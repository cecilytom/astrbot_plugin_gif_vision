"""Media and event detection helpers for QQ GIFs."""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from .frames import image_info
from .prompt import SENTINEL_PREFIX


@dataclass(frozen=True)
class GifCandidate:
    ref: str
    where: str  # image_urls | extra_parts | mface
    index: int = 0
    summary: str = ""


async def read_ref_bytes(ref: str) -> bytes | None:
    """Resolve an AstrBot media reference to bytes, using the core resolver."""

    if not isinstance(ref, str) or not ref.strip():
        return None
    try:
        from astrbot.core.utils.media_utils import MediaResolver

        return await MediaResolver(ref, media_type="image").to_bytes()
    except Exception:
        return None


def read_local_file(path: str) -> bytes | None:
    try:
        data = Path(path).expanduser().read_bytes()
    except OSError:
        return None
    return data


def is_animated_image(data: bytes) -> bool:
    if not data:
        return False
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return True
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            return int(getattr(image, "n_frames", 1) or 1) > 1
    except Exception:
        return False


def iter_request_image_refs(req: Any) -> list[GifCandidate]:
    """Return image references from the current request, deduplicated."""

    refs: list[GifCandidate] = []
    seen: set[str] = set()

    def add(value: Any, where: str, index: int) -> None:
        if not isinstance(value, str):
            return
        value = value.strip()
        if not value or value in seen:
            return
        seen.add(value)
        refs.append(GifCandidate(ref=value, where=where, index=index))

    for index, ref in enumerate(getattr(req, "image_urls", None) or []):
        add(ref, "image_urls", index)

    for index, part in enumerate(
        getattr(req, "extra_user_content_parts", None) or []
    ):
        if isinstance(part, dict):
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                add(image_url.get("url"), "extra_parts", index)
            elif isinstance(image_url, str):
                add(image_url, "extra_parts", index)
        else:
            image_url = getattr(part, "image_url", None)
            if image_url is None:
                continue
            url = getattr(image_url, "url", None)
            if url is None and isinstance(image_url, dict):
                url = image_url.get("url")
            add(url, "extra_parts", index)

    return refs


def parts_text(part: Any) -> str:
    if isinstance(part, dict):
        text = part.get("text") or part.get("content")
        return str(text) if text else ""
    text = getattr(part, "text", None)
    if text is None:
        text = getattr(part, "content", None)
    return str(text) if text else ""


def existing_description(req: Any) -> bool:
    """Detect descriptions already injected by core/companion plugins."""

    markers = (
        "<image_caption>",
        "<gif_analysis>",
        "图像表达意图",
        "用户本轮图片视觉摘要",
        "GIF 已转换为 PNG 帧",
        SENTINEL_PREFIX,
    )
    texts = [
        str(getattr(req, "prompt", "") or ""),
        str(getattr(req, "system_prompt", "") or ""),
    ]
    for part in getattr(req, "extra_user_content_parts", None) or []:
        texts.append(parts_text(part))
    return any(marker in text for text in texts for marker in markers)


def request_has_text(req: Any, needle: str) -> bool:
    if not needle:
        return False
    texts = [
        str(getattr(req, "prompt", "") or ""),
        str(getattr(req, "system_prompt", "") or ""),
    ]
    for part in getattr(req, "extra_user_content_parts", None) or []:
        texts.append(parts_text(part))
    return any(needle in text for text in texts)


def is_group_event(event: Any) -> bool:
    getter = getattr(event, "get_group_id", None)
    if callable(getter):
        try:
            if getter():
                return True
        except Exception:
            pass
    origin = str(getattr(event, "unified_msg_origin", "") or "")
    return "GroupMessage" in origin


def question_from_event(event: Any, req: Any = None) -> str:
    message = str(getattr(event, "message_str", "") or "")
    if not message and req is not None:
        message = str(getattr(req, "prompt", "") or "")
    message = re.sub(
        r"<system_reminder>.*?</system_reminder>",
        " ",
        message,
        flags=re.DOTALL | re.IGNORECASE,
    )
    message = re.sub(r"@\S+\s*", " ", message)
    for placeholder in ("[图片]", "[Image]", "[GIF]", "[动图]"):
        message = message.replace(placeholder, " ")
    message = re.sub(r"\s+", " ", message).strip()
    return message[:300]


def mface_candidates(event: Any) -> list[GifCandidate]:
    """Best-effort recovery for NapCat market-face segments."""

    message_obj = getattr(event, "message_obj", None)
    raw = getattr(message_obj, "raw_message", None)
    if raw is None:
        return []
    segments = raw.get("message") if hasattr(raw, "get") else None
    if not isinstance(segments, list):
        return []
    result: list[GifCandidate] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or segment.get("type") != "mface":
            continue
        data = segment.get("data") or {}
        emoji_id = str(data.get("emoji_id") or "").strip()
        if not re.fullmatch(r"[0-9A-Fa-f]{4,64}", emoji_id):
            continue
        summary = str(data.get("summary") or "").strip()
        url = (
            "https://gxh.vip.qq.com/club/item/parcel/item/"
            f"{emoji_id[:2]}/{emoji_id}/raw300.gif"
        )
        result.append(
            GifCandidate(
                ref=url,
                where="mface",
                index=index,
                summary=summary,
            )
        )
    return result


def bytes_from_image_component(component: Any) -> bytes | None:
    path = getattr(component, "path", None)
    if isinstance(path, str) and path:
        try:
            return Path(path).read_bytes()
        except OSError:
            pass
    file_ref = getattr(component, "file", None)
    if isinstance(file_ref, str) and file_ref.startswith("base64://"):
        try:
            return base64.b64decode(file_ref.removeprefix("base64://"))
        except Exception:
            pass
    url = getattr(component, "url", None) or file_ref
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return None  # caller resolves asynchronously
    return None


def image_metadata(data: bytes) -> dict:
    return image_info(data)


__all__ = [
    "GifCandidate",
    "bytes_from_image_component",
    "existing_description",
    "image_metadata",
    "is_animated_image",
    "is_group_event",
    "iter_request_image_refs",
    "mface_candidates",
    "parts_text",
    "question_from_event",
    "read_local_file",
    "read_ref_bytes",
    "request_has_text",
]
