"""YouTube Best Video Finder — search and get the single best video via AI."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import streamlit as st
import yt_dlp
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

load_dotenv()

# ---------------------------------------------------------------------------
# YouTube collector
# ---------------------------------------------------------------------------


@dataclass
class VideoInfo:
    video_id: str
    title: str
    url: str
    channel: str
    description: str
    duration_seconds: int | None
    view_count: int | None
    like_count: int | None
    upload_date: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        duration = _format_duration(self.duration_seconds)
        views = f"{self.view_count:,}" if self.view_count else "unknown"
        likes = f"{self.like_count:,}" if self.like_count else "unknown"
        uploaded = self.upload_date or "unknown"
        desc = (self.description or "")[:400].replace("\n", " ")
        return (
            f"Title: {self.title}\n"
            f"Channel: {self.channel}\n"
            f"URL: {self.url}\n"
            f"Duration: {duration} | Views: {views} | Likes: {likes} | Uploaded: {uploaded}\n"
            f"Description: {desc}"
        )


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def search_videos(query: str, max_results: int = 15) -> list[VideoInfo]:
    search_url = f"ytsearch{max_results}:{query}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
    }

    videos: list[VideoInfo] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(search_url, download=False)

    for entry in result.get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id") or ""
        if not video_id:
            continue
        videos.append(
            VideoInfo(
                video_id=video_id,
                title=entry.get("title") or "Untitled",
                url=f"https://www.youtube.com/watch?v={video_id}",
                channel=entry.get("channel") or entry.get("uploader") or "Unknown",
                description=entry.get("description") or "",
                duration_seconds=entry.get("duration"),
                view_count=entry.get("view_count"),
                like_count=entry.get("like_count"),
                upload_date=entry.get("upload_date"),
            )
        )
    return videos


# ---------------------------------------------------------------------------
# AI evaluator
# ---------------------------------------------------------------------------

PLACEHOLDER_KEYS = {"", "sk-your-key-here", "your-api-key", "sk-your-****here"}


def is_valid_api_key(key: str | None) -> bool:
    if not key:
        return False
    normalized = key.strip()
    if normalized in PLACEHOLDER_KEYS or normalized.startswith("sk-your"):
        return False
    return normalized.startswith("sk-") and len(normalized) > 20


@dataclass
class EvaluationResult:
    best_video_id: str
    reasoning: str
    confidence: str
    runner_up_title: str | None = None


SYSTEM_PROMPT = """You are an expert at evaluating YouTube videos for quality and relevance.

Given a user's search query and a list of candidate videos with metadata, choose the SINGLE best video.

Consider:
- Relevance to the exact search intent (most important)
- Educational value and clarity of title/description
- Credibility of the channel
- View count and engagement as weak signals (popular != best)
- Recency when the topic benefits from up-to-date content
- Appropriate length for the topic

Respond ONLY with valid JSON in this exact shape:
{
  "best_video_id": "<youtube video id>",
  "reasoning": "<2-4 sentences explaining why this is the best pick>",
  "confidence": "<high|medium|low>",
  "runner_up_title": "<title of second-best option or null>"
}"""


def _build_user_prompt(query: str, videos: list[VideoInfo]) -> str:
    lines = [f'Search query: "{query}"', "", "Candidate videos:", ""]
    for i, video in enumerate(videos, start=1):
        lines.append(f"--- Video {i} (id: {video.video_id}) ---")
        lines.append(video.summary())
        lines.append("")
    lines.append("Pick the single best video for this search.")
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def evaluate_videos(
    query: str,
    videos: list[VideoInfo],
    *,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> EvaluationResult:
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not is_valid_api_key(key):
        raise ValueError(
            "A valid OpenAI API key is required. Get one at "
            "https://platform.openai.com/api-keys and paste it in the sidebar."
        )

    client = OpenAI(api_key=key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, videos)},
            ],
        )
    except AuthenticationError as exc:
        raise ValueError(
            "Invalid OpenAI API key. Create a new key at "
            "https://platform.openai.com/api-keys and update the sidebar."
        ) from exc
    except RateLimitError as exc:
        raise ValueError(
            "OpenAI quota exceeded for this API key. Add billing/credits at "
            "https://platform.openai.com/settings/organization/billing/overview."
        ) from exc

    data = _parse_response(response.choices[0].message.content or "{}")
    best_id = data.get("best_video_id", "")
    valid_ids = {v.video_id for v in videos}
    if best_id not in valid_ids:
        best_id = videos[0].video_id

    return EvaluationResult(
        best_video_id=best_id,
        reasoning=data.get("reasoning", "Selected as the best match for your query."),
        confidence=data.get("confidence", "medium"),
        runner_up_title=data.get("runner_up_title"),
    )


def fallback_pick(query: str, videos: list[VideoInfo]) -> EvaluationResult:
    query_words = set(query.lower().split())

    def score(video: VideoInfo) -> float:
        title_words = set(video.title.lower().split())
        overlap = len(query_words & title_words)
        views = video.view_count or 0
        likes = video.like_count or 0
        return overlap * 1000 + (likes * 0.001) + (views * 0.000001)

    ranked = sorted(videos, key=score, reverse=True)
    best = ranked[0]
    runner_up = ranked[1].title if len(ranked) > 1 else None
    return EvaluationResult(
        best_video_id=best.video_id,
        reasoning=(
            f'Heuristic pick (no AI key): "{best.title}" scored highest on title relevance '
            f"and engagement signals. Add an OpenAI API key for smarter evaluation."
        ),
        confidence="low",
        runner_up_title=runner_up,
    )


def _default_api_key() -> str:
    """Read API key from Streamlit Cloud secrets or local .env."""
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

SUGGESTED_TOPICS = [
    "Python for beginners",
    "Machine learning explained",
    "React tutorial 2024",
    "How to invest in stocks",
    "Guitar lessons easy songs",
    "Photoshop for beginners",
]

CONFIDENCE_META = {
    "high": ("High confidence", "#22c55e", 95),
    "medium": ("Medium confidence", "#eab308", 70),
    "low": ("Low confidence", "#ef4444", 45),
}


def _thumb_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"


def _format_count(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 1200px;
        }

        .hero {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 45%, #0f3460 100%);
            border-radius: 20px;
            padding: 2.2rem 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        }

        .hero h1 {
            color: #fff;
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.03em;
        }

        .hero p {
            color: rgba(255,255,255,0.75);
            font-size: 1.05rem;
            margin: 0;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(255,0,0,0.15);
            color: #ff6b6b;
            border: 1px solid rgba(255,80,80,0.35);
            border-radius: 999px;
            padding: 0.3rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 1rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0 0.5rem 0;
        }

        .feature-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            color: #fff;
        }

        .feature-card .icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
        .feature-card .title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.25rem; }
        .feature-card .desc { color: rgba(255,255,255,0.6); font-size: 0.82rem; line-height: 1.4; }

        .result-banner {
            background: linear-gradient(90deg, #065f46, #047857);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            color: #ecfdf5;
            font-weight: 600;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .video-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }

        .video-card:hover { border-color: #374151; }

        .video-card.winner {
            border-color: #22c55e;
            box-shadow: 0 0 0 1px rgba(34,197,94,0.25), 0 8px 30px rgba(34,197,94,0.12);
        }

        .video-card .rank {
            display: inline-block;
            background: #1f2937;
            color: #9ca3af;
            border-radius: 8px;
            padding: 0.15rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .video-card.winner .rank {
            background: #14532d;
            color: #86efac;
        }

        .video-card img {
            width: 100%;
            border-radius: 10px;
            aspect-ratio: 16/9;
            object-fit: cover;
        }

        .video-card .title {
            color: #f9fafb;
            font-weight: 600;
            font-size: 0.92rem;
            margin: 0.6rem 0 0.3rem;
            line-height: 1.35;
        }

        .video-card .meta {
            color: #9ca3af;
            font-size: 0.78rem;
        }

        .ai-box {
            background: linear-gradient(160deg, #1e1b4b, #312e81);
            border: 1px solid rgba(129,140,248,0.25);
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            color: #e0e7ff;
        }

        .ai-box h3 {
            color: #c7d2fe;
            margin: 0 0 0.75rem 0;
            font-size: 1rem;
            font-weight: 700;
        }

        .ai-box p {
            color: rgba(224,231,255,0.9);
            line-height: 1.6;
            margin: 0;
            font-size: 0.95rem;
        }

        .chip-label {
            color: #6b7280;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.75rem 0 0.4rem 0;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a, #111827);
        }

        div[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #dc2626, #ef4444) !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.2rem !important;
            box-shadow: 0 4px 20px rgba(239,68,68,0.35) !important;
        }

        .stButton > button[kind="secondary"] {
            border-radius: 999px !important;
            border: 1px solid #d1d5db !important;
            font-size: 0.82rem !important;
        }

        @media (max-width: 768px) {
            .feature-grid { grid-template-columns: 1fr; }
            .hero h1 { font-size: 1.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "last_query": "",
        "last_result": None,
        "search_history": [],
        "pending_query": "",
        "trigger_search": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def find_best_video(
    query: str,
    max_results: int,
    api_key: str | None,
    use_ai: bool,
) -> tuple[VideoInfo, EvaluationResult, list[VideoInfo]]:
    videos = search_videos(query, max_results=max_results)
    if not videos:
        raise RuntimeError("No videos found for this search. Try different keywords.")

    if use_ai and is_valid_api_key(api_key):
        result = evaluate_videos(query, videos, api_key=api_key)
    else:
        result = fallback_pick(query, videos)

    best = next(v for v in videos if v.video_id == result.best_video_id)
    return best, result, videos


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">AI-Powered Video Discovery</div>
            <h1>Find the single best YouTube video</h1>
            <p>We search dozens of videos, analyze them with AI, and hand you the winner — no more endless scrolling.</p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="icon">🔍</div>
                    <div class="title">Smart Search</div>
                    <div class="desc">Pulls real metadata from YouTube — titles, views, descriptions & more.</div>
                </div>
                <div class="feature-card">
                    <div class="icon">🧠</div>
                    <div class="title">AI Evaluation</div>
                    <div class="desc">Compares every candidate and picks the one that best matches your intent.</div>
                </div>
                <div class="feature-card">
                    <div class="icon">🏆</div>
                    <div class="title">One Winner</div>
                    <div class="desc">No overwhelming lists — just the best video with a clear explanation why.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_video_card(video: VideoInfo, rank: int, is_winner: bool = False) -> None:
    winner_class = " winner" if is_winner else ""
    badge = "🏆 BEST PICK" if is_winner else f"#{rank}"
    st.markdown(
        f"""
        <div class="video-card{winner_class}">
            <span class="rank">{badge}</span>
            <img src="{_thumb_url(video.video_id)}" alt="thumbnail"/>
            <div class="title">{video.title}</div>
            <div class="meta">{video.channel} · {_format_count(video.view_count)} views · {_format_duration(video.duration_seconds)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(
    query: str,
    best: VideoInfo,
    evaluation: EvaluationResult,
    all_videos: list[VideoInfo],
    used_ai: bool,
) -> None:
    st.markdown(
        f'<div class="result-banner">✅ Best match for <em>"{query}"</em> — evaluated {len(all_videos)} videos</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Views", _format_count(best.view_count))
    m2.metric("Likes", _format_count(best.like_count))
    m3.metric("Duration", _format_duration(best.duration_seconds))
    m4.metric("AI Mode", "Smart" if used_ai else "Basic")

    if used_ai:
        label, _, pct = CONFIDENCE_META.get(
            evaluation.confidence, ("Unknown", "#6b7280", 50)
        )
        st.progress(pct / 100, text=f"AI {label} — {pct}%")
    else:
        st.info(
            "**Basic mode** — ranking uses title keywords + views/likes only. "
            "Add an OpenAI API key in the sidebar and enable **AI evaluation** for high-confidence picks."
        )

    tab_winner, tab_compare, tab_ai = st.tabs(["🏆 Winner", "📊 All Candidates", "🧠 AI Analysis"])

    with tab_winner:
        left, right = st.columns([1.6, 1])
        with left:
            st.video(best.url)
            st.link_button("▶ Open on YouTube", best.url, use_container_width=True, type="primary")
        with right:
            st.markdown(f"### {best.title}")
            st.caption(f"📺 {best.channel}")
            if best.description:
                with st.expander("Video description", expanded=False):
                    st.write(best.description[:600] + ("..." if len(best.description) > 600 else ""))

    with tab_compare:
        st.caption("All videos we found and evaluated for your search.")
        cols = st.columns(3)
        for i, video in enumerate(all_videos):
            with cols[i % 3]:
                render_video_card(video, i + 1, is_winner=video.video_id == best.video_id)
                st.link_button("Watch", video.url, key=f"watch_{video.video_id}", use_container_width=True)

    with tab_ai:
        st.markdown(
            f"""
            <div class="ai-box">
                <h3>Why this video won</h3>
                <p>{evaluation.reasoning}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if evaluation.runner_up_title:
            st.info(f"🥈 **Runner-up:** {evaluation.runner_up_title}")
        if not used_ai:
            st.warning("Using basic ranking. Add an OpenAI API key in the sidebar for smarter AI picks.")


def run_search(query: str, api_key: str | None, use_ai: bool, max_results: int) -> None:
    progress = st.progress(0, text="Starting search...")
    try:
        progress.progress(15, text="🔍 Searching YouTube...")
        videos = search_videos(query, max_results=max_results)
        if not videos:
            raise RuntimeError("No videos found for this search. Try different keywords.")

        progress.progress(55, text=f"📥 Collected {len(videos)} videos — analyzing...")
        if use_ai and is_valid_api_key(api_key):
            try:
                result = evaluate_videos(query, videos, api_key=api_key)
                used_ai = True
            except ValueError as exc:
                # Graceful fallback when key is invalid or quota is exhausted.
                st.warning(f"{exc} Switched to Basic mode for this search.")
                result = fallback_pick(query, videos)
                used_ai = False
        else:
            result = fallback_pick(query, videos)
            used_ai = False

        progress.progress(90, text="🏆 Picking the winner...")
        best = next(v for v in videos if v.video_id == result.best_video_id)
        progress.progress(100, text="Done!")

        st.session_state.last_query = query
        st.session_state.last_result = (best, result, videos, used_ai)
        if query not in st.session_state.search_history:
            st.session_state.search_history = ([query] + st.session_state.search_history)[:8]
        st.balloons()
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
    finally:
        progress.empty()


st.set_page_config(
    page_title="YouTube Best Video Finder",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
inject_styles()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        value=_default_api_key(),
        type="password",
        help="Required for AI-powered evaluation.",
    )
    key_ok = is_valid_api_key(api_key)
    if api_key and not key_ok:
        st.warning("Invalid key — using basic mode.")
    use_ai = st.toggle("🧠 AI evaluation", value=key_ok, disabled=not key_ok)
    max_results = st.slider("Videos to scan", 5, 25, 15)

    st.divider()
    st.markdown("**Recent searches**")
    if st.session_state.search_history:
        for past in st.session_state.search_history:
            if st.button(f"↩ {past}", key=f"hist_{past}", use_container_width=True):
                st.session_state.pending_query = past
                st.session_state.trigger_search = True
                st.rerun()
    else:
        st.caption("Your searches will appear here.")

    st.divider()
    with st.expander("How it works"):
        st.markdown(
            "1. **Search** — finds related YouTube videos\n"
            "2. **Collect** — gathers views, likes, descriptions\n"
            "3. **Evaluate** — AI scores every candidate\n"
            "4. **Win** — you get the single best pick"
        )

render_hero()

# Quick-topic chips
st.markdown('<p class="chip-label">Try a popular topic</p>', unsafe_allow_html=True)
chip_cols = st.columns(len(SUGGESTED_TOPICS))
for i, topic in enumerate(SUGGESTED_TOPICS):
    with chip_cols[i]:
        if st.button(topic, key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_query = topic
            st.session_state.trigger_search = True
            st.rerun()

# Main search bar
search_col, btn_col = st.columns([5, 1])
with search_col:
    query_input = st.text_input(
        "Search",
        value=st.session_state.pending_query or st.session_state.last_query,
        placeholder="What do you want to learn? e.g. Python data analysis tutorial",
        label_visibility="collapsed",
    )
with btn_col:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# Handle search triggers
active_query = ""
if search_clicked and query_input.strip():
    active_query = query_input.strip()
elif st.session_state.trigger_search and st.session_state.pending_query:
    active_query = st.session_state.pending_query.strip()

st.session_state.trigger_search = False
st.session_state.pending_query = ""

if active_query:
    run_search(active_query, api_key or None, use_ai, max_results)

# Show cached results (persists across chip clicks / reruns until new search)
if st.session_state.last_result:
    best, evaluation, all_videos, used_ai = st.session_state.last_result
    st.divider()
    render_results(st.session_state.last_query, best, evaluation, all_videos, used_ai)
elif not active_query:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🎬 Learn anything")
        st.caption("Tutorials, courses, how-tos — any topic on YouTube.")
    with c2:
        st.markdown("#### ⚡ Save time")
        st.caption("Skip watching 10 mediocre videos. Get the best one instantly.")
    with c3:
        st.markdown("#### 🔑 Add AI key")
        st.caption("Paste your OpenAI key in the sidebar for smarter results.")
