"""Request mutation helpers shared by hooks, commands, and the WebUI."""

from __future__ import annotations

from typing import Any

from .detect import (
    existing_description,
    is_group_event,
    iter_request_image_refs,
    parts_text,
    question_from_event,
    request_has_text,
)

_DONE_ATTR = "_gif_vision_done"
_PROCESSED_ATTR = "_gif_vision_processed"
_RESULTS_EXTRA = "_gif_vision_results"


def mark_done(req: Any) -> None:
    try:
        setattr(req, _DONE_ATTR, True)
    except Exception:
        pass


def is_done(req: Any) -> bool:
    return bool(getattr(req, _DONE_ATTR, False))


def mark_processed(req: Any) -> None:
    try:
        setattr(req, _PROCESSED_ATTR, True)
    except Exception:
        pass


def is_processed(req: Any) -> bool:
    return bool(getattr(req, _PROCESSED_ATTR, False))


def append_text_part(req: Any, text: str) -> bool:
    if not text:
        return False
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        parts = []
        try:
            setattr(req, "extra_user_content_parts", parts)
        except Exception:
            return False
    try:
        from astrbot.core.agent.message import TextPart

        part = TextPart(text=text).mark_as_temp()
    except Exception:
        part = {"type": "text", "text": text, "_no_save": True}
    parts.append(part)
    return True


def remove_image_refs(req: Any, refs: set[str]) -> int:
    if not refs:
        return 0
    removed = 0
    image_urls = getattr(req, "image_urls", None)
    if isinstance(image_urls, list):
        kept = []
        for ref in image_urls:
            if isinstance(ref, str) and ref in refs:
                removed += 1
            else:
                kept.append(ref)
        try:
            setattr(req, "image_urls", kept)
        except Exception:
            pass

    parts = getattr(req, "extra_user_content_parts", None)
    if isinstance(parts, list):
        kept_parts = []
        for part in parts:
            if isinstance(part, dict):
                if part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if isinstance(url, str) and url in refs:
                        removed += 1
                        continue
                kept_parts.append(part)
                continue
            image_url = getattr(part, "image_url", None)
            url = getattr(image_url, "url", None)
            if isinstance(url, str) and url in refs:
                removed += 1
                continue
            kept_parts.append(part)
        try:
            setattr(req, "extra_user_content_parts", kept_parts)
        except Exception:
            pass
    return removed


def set_event_extra(event: Any, key: str, value: Any) -> None:
    if event is None:
        return
    setter = getattr(event, "set_extra", None)
    if callable(setter):
        try:
            setter(key, value)
            return
        except Exception:
            pass
    try:
        setattr(event, key, value)
    except Exception:
        pass


def get_event_extra(event: Any, key: str, default: Any = None) -> Any:
    if event is None:
        return default
    getter = getattr(event, "get_extra", None)
    if callable(getter):
        try:
            value = getter(key)
            return default if value is None else value
        except Exception:
            pass
    return getattr(event, key, default)


def set_results(event: Any, results: list[dict[str, Any]]) -> None:
    set_event_extra(event, _RESULTS_EXTRA, results)


def get_results(event: Any) -> list[dict[str, Any]]:
    value = get_event_extra(event, _RESULTS_EXTRA, [])
    return value if isinstance(value, list) else []


__all__ = [
    "append_text_part",
    "existing_description",
    "get_event_extra",
    "get_results",
    "is_done",
    "is_group_event",
    "is_processed",
    "iter_request_image_refs",
    "mark_done",
    "mark_processed",
    "parts_text",
    "question_from_event",
    "remove_image_refs",
    "request_has_text",
    "set_event_extra",
    "set_results",
]
