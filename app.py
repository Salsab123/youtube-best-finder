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
from openai import AuthenticationError, OpenAI

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

st.set_page_config(
    page_title="YouTube Best Video Finder",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 YouTube Best Video Finder")
st.caption(
    "Search any topic — we collect related YouTube videos and use AI to pick the single best one."
)


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


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        value=_default_api_key(),
        type="password",
        help="Required for AI-powered evaluation. Get one at platform.openai.com/api-keys",
    )
    key_ok = is_valid_api_key(api_key)
    if api_key and not key_ok:
        st.warning("Invalid or placeholder API key. AI is off until you add a real key.")
    use_ai = st.toggle("Use AI evaluation", value=key_ok, disabled=not key_ok)
    max_results = st.slider("Videos to evaluate", min_value=5, max_value=25, value=15)
    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Search YouTube for related videos\n"
        "2. Collect title, description, views, etc.\n"
        "3. AI reviews all candidates\n"
        "4. You get the single best pick"
    )

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "What do you want to learn or find?",
        placeholder="e.g. Python list comprehensions tutorial for beginners",
    )
    submitted = st.form_submit_button("Find Best Video", type="primary", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        try:
            with st.status("Working...", expanded=True) as status:
                st.write("🔍 Searching YouTube...")
                best, evaluation, all_videos = find_best_video(
                    query.strip(),
                    max_results=max_results,
                    api_key=api_key or None,
                    use_ai=use_ai,
                )
                status.update(label="Done!", state="complete", expanded=False)

            st.success("Best video found!")

            col_video, col_info = st.columns([3, 2])

            with col_video:
                st.subheader(best.title)
                st.video(best.url)
                st.link_button("Open on YouTube", best.url, use_container_width=True)

            with col_info:
                st.markdown("### Why this video?")
                confidence_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                icon = confidence_colors.get(evaluation.confidence, "⚪")
                st.markdown(f"**Confidence:** {icon} {evaluation.confidence.title()}")
                st.write(evaluation.reasoning)
                if evaluation.runner_up_title:
                    st.markdown(f"**Runner-up:** {evaluation.runner_up_title}")

                st.divider()
                st.markdown("**Channel**")
                st.write(best.channel)
                st.markdown("**Stats**")
                views = f"{best.view_count:,}" if best.view_count else "—"
                likes = f"{best.like_count:,}" if best.like_count else "—"
                st.write(f"Views: {views} · Likes: {likes}")

            with st.expander(f"All {len(all_videos)} videos evaluated", expanded=False):
                for i, video in enumerate(all_videos, start=1):
                    marker = " ✅" if video.video_id == best.video_id else ""
                    st.markdown(f"**{i}. {video.title}**{marker}")
                    st.caption(f"{video.channel} · {video.url}")

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
else:
    st.info("Enter a topic above and click **Find Best Video** to get started.")
