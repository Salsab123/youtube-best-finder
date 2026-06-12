"""YouTube Best Video Finder — search and get the single best video via AI."""

import os

import streamlit as st
from dotenv import load_dotenv

from ai_evaluator import EvaluationResult, evaluate_videos, fallback_pick, is_valid_api_key
from youtube_collector import VideoInfo, search_videos

load_dotenv()

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


# Sidebar settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
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

# Search form
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
