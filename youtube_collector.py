"""Collect YouTube video metadata for a search query."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import yt_dlp


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
    """Search YouTube and return metadata for related videos."""
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

    entries = result.get("entries") or []
    for entry in entries:
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
