"""Configuration model for the GIF vision plugin.

The module deliberately has no AstrBot imports so that tests can exercise the
configuration coercion without a running bot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if result != result:  # NaN
        result = default
    return max(minimum, min(maximum, result))


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[\n,]", value) if item.strip()]
    return []


def _source_id(value: Any) -> str:
    candidate = _as_str(value, "gif_vision").lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,127}", candidate):
        return candidate
    return "gif_vision"


@dataclass
class GifConfig:
    # General
    enable: bool = True
    provider_id: str = ""
    fallback_provider_ids: list[str] = field(default_factory=list)
    embedding_provider_id: str = ""
    embedding_enabled: bool = True

    # Companion integration
    return_to_companion: bool = True
    companion_source_id: str = "gif_vision"
    companion_lane: str = "scene"
    companion_priority: int = 55
    companion_max_age_seconds: int = 900
    companion_hook_priority: int = -259500
    companion_register_extension: bool = True

    # Media / vision
    max_gifs_per_message: int = 2
    frame_count: int = 6
    frame_max_edge: int = 768
    max_gif_mb: float = 15.0
    vision_timeout_sec: float = 25.0
    max_output_chars: int = 1200
    vision_with_question: bool = True
    vision_describe_prompt: str = ""
    vision_qa_prompt: str = ""

    # Scope / behaviour
    session_scope: str = "all"
    skip_if_existing_description: bool = True
    keep_original_image: bool = False
    handle_mface: bool = True

    # Library
    library_enable: bool = True
    library_dir: str = ""
    library_max_mb: float = 512.0
    retention_days: int = 0
    near_duplicate_hamming: int = 12
    near_duplicate_store: bool = True
    search_top_k: int = 5
    search_min_score: float = 0.55
    vector_search_max_rows: int = 5000
    send_max_mb: float = 15.0
    max_concurrent: int = 2

    # WebUI
    webui_page_size: int = 24
    webui_thumb_edge: int = 320
    webui_preview_max_mb: float = 4.0
    webui_import_enable: bool = True

    # Diagnostics
    debug_log: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "GifConfig":
        data = dict(raw or {})
        cfg = cls(
            enable=_as_bool(data.get("enable"), cls.enable),
            provider_id=_as_str(data.get("provider_id"), ""),
            fallback_provider_ids=_as_list(data.get("fallback_provider_ids")),
            embedding_provider_id=_as_str(data.get("embedding_provider_id"), ""),
            embedding_enabled=_as_bool(
                data.get("embedding_enabled"), cls.embedding_enabled
            ),
            return_to_companion=_as_bool(
                data.get("return_to_companion"), cls.return_to_companion
            ),
            companion_source_id=_source_id(data.get("companion_source_id")),
            companion_lane=_as_str(data.get("companion_lane"), "scene").lower(),
            companion_priority=_as_int(
                data.get("companion_priority"), 55, -100000, 100000
            ),
            companion_max_age_seconds=_as_int(
                data.get("companion_max_age_seconds"), 900, 0, 86400
            ),
            companion_hook_priority=_as_int(
                data.get("companion_hook_priority"), -259500, -1000000, 1000000
            ),
            companion_register_extension=_as_bool(
                data.get("companion_register_extension"), True
            ),
            max_gifs_per_message=_as_int(
                data.get("max_gifs_per_message"), 2, 1, 5
            ),
            frame_count=_as_int(data.get("frame_count"), 6, 2, 9),
            frame_max_edge=_as_int(data.get("frame_max_edge"), 768, 64, 2048),
            max_gif_mb=_as_float(data.get("max_gif_mb"), 15.0, 1.0, 100.0),
            vision_timeout_sec=_as_float(
                data.get("vision_timeout_sec"), 25.0, 5.0, 120.0
            ),
            max_output_chars=_as_int(
                data.get("max_output_chars"), 1200, 200, 4000
            ),
            vision_with_question=_as_bool(
                data.get("vision_with_question"), True
            ),
            vision_describe_prompt=_as_str(data.get("vision_describe_prompt")),
            vision_qa_prompt=_as_str(data.get("vision_qa_prompt")),
            session_scope=_as_str(data.get("session_scope"), "all").lower(),
            skip_if_existing_description=_as_bool(
                data.get("skip_if_existing_description"), True
            ),
            keep_original_image=_as_bool(
                data.get("keep_original_image"), False
            ),
            handle_mface=_as_bool(data.get("handle_mface"), True),
            library_enable=_as_bool(data.get("library_enable"), True),
            library_dir=_as_str(data.get("library_dir"), ""),
            library_max_mb=_as_float(
                data.get("library_max_mb"), 512.0, 16.0, 10240.0
            ),
            retention_days=_as_int(data.get("retention_days"), 0, 0, 3650),
            near_duplicate_hamming=_as_int(
                data.get("near_duplicate_hamming"), 12, 0, 192
            ),
            near_duplicate_store=_as_bool(
                data.get("near_duplicate_store"), True
            ),
            search_top_k=_as_int(data.get("search_top_k"), 5, 1, 20),
            search_min_score=_as_float(
                data.get("search_min_score"), 0.55, 0.0, 1.0
            ),
            vector_search_max_rows=_as_int(
                data.get("vector_search_max_rows"), 5000, 100, 50000
            ),
            send_max_mb=_as_float(data.get("send_max_mb"), 15.0, 1.0, 100.0),
            max_concurrent=_as_int(data.get("max_concurrent"), 2, 1, 8),
            webui_page_size=_as_int(data.get("webui_page_size"), 24, 1, 50),
            webui_thumb_edge=_as_int(
                data.get("webui_thumb_edge"), 320, 64, 1024
            ),
            webui_preview_max_mb=_as_float(
                data.get("webui_preview_max_mb"), 4.0, 1.0, 50.0
            ),
            webui_import_enable=_as_bool(
                data.get("webui_import_enable"), True
            ),
            debug_log=_as_bool(data.get("debug_log"), False),
        )
        return cfg.normalized()

    def normalized(self) -> "GifConfig":
        scope = self.session_scope
        if scope not in {"all", "group", "private"}:
            scope = "all"
        lane = self.companion_lane
        if lane not in {"scene", "memory"}:
            lane = "scene"
        return replace(
            self,
            session_scope=scope,
            companion_lane=lane,
            provider_id=self.provider_id.strip(),
            embedding_provider_id=self.embedding_provider_id.strip(),
            companion_source_id=_source_id(self.companion_source_id),
            fallback_provider_ids=[
                item
                for item in dict.fromkeys(self.fallback_provider_ids)
                if item
            ],
            max_gif_mb=max(1.0, min(100.0, float(self.max_gif_mb))),
            vision_timeout_sec=max(
                5.0, min(120.0, float(self.vision_timeout_sec))
            ),
            library_max_mb=max(16.0, min(10240.0, float(self.library_max_mb))),
            send_max_mb=max(1.0, min(100.0, float(self.send_max_mb))),
            webui_preview_max_mb=max(
                1.0, min(50.0, float(self.webui_preview_max_mb))
            ),
            search_min_score=max(0.0, min(1.0, float(self.search_min_score))),
        )

    def should_handle_event(self, *, is_group: bool) -> bool:
        if self.session_scope == "all":
            return True
        if self.session_scope == "group":
            return is_group
        return not is_group

    def prompt_defaults(self) -> tuple[str, str]:
        return (self.vision_describe_prompt, self.vision_qa_prompt)
