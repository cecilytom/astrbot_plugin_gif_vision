"""AstrBot provider resolution for vision descriptions and embeddings."""

from __future__ import annotations

import asyncio
from typing import Any

from .config import GifConfig
from .fingerprint import normalize_embedding
from .frames import Frame


class VisionError(RuntimeError):
    pass


class ProviderResolver:
    def __init__(self, context: Any, config: GifConfig, logger: Any = None) -> None:
        self.context = context
        self.config = config
        self.logger = logger
        self.last_vision_error = ""
        self.last_embedding_error = ""

    # ------------------------------------------------------------- vision
    @staticmethod
    def _supports_image(provider: Any) -> bool:
        provider_config = getattr(provider, "provider_config", {}) or {}
        modalities = provider_config.get("modalities")
        if isinstance(modalities, list) and modalities:
            return "image" in modalities
        return True

    @staticmethod
    def _provider_id(provider: Any) -> str:
        provider_config = getattr(provider, "provider_config", {}) or {}
        return str(provider_config.get("id") or "")

    @staticmethod
    def _provider_model(provider: Any) -> str:
        getter = getattr(provider, "get_model", None)
        if callable(getter):
            try:
                return str(getter() or "")
            except Exception:
                pass
        provider_config = getattr(provider, "provider_config", {}) or {}
        return str(provider_config.get("model") or "")

    def _by_id(self, provider_id: str) -> Any | None:
        if not provider_id:
            return None
        getter = getattr(self.context, "get_provider_by_id", None)
        if not callable(getter):
            return None
        try:
            provider = getter(provider_id)
        except Exception:
            return None
        if provider is not None and self._supports_image(provider):
            return provider
        return None

    async def vision_provider(self, event: Any = None) -> Any | None:
        for provider_id in [self.config.provider_id, *self.config.fallback_provider_ids]:
            provider = self._by_id(provider_id)
            if provider is not None:
                return provider

        getter = getattr(self.context, "get_using_provider_async", None)
        if callable(getter):
            try:
                umo = getattr(event, "unified_msg_origin", None)
                provider = await getter(umo=umo)
            except Exception:
                provider = None
            if provider is not None and self._supports_image(provider):
                return provider

        # Core image-caption provider is a sensible last resort.
        try:
            umo = getattr(event, "unified_msg_origin", None)
            config = self.context.get_config(umo=umo)
            provider_settings = config.get("provider_settings", {}) if config else {}
            caption_id = str(
                provider_settings.get("default_image_caption_provider_id") or ""
            )
            provider = self._by_id(caption_id)
            if provider is not None:
                return provider
        except Exception:
            pass
        return None

    async def describe(
        self,
        provider: Any,
        prompt: str,
        frames: list[Frame],
        *,
        timeout_sec: float,
    ) -> str:
        if provider is None:
            raise VisionError("没有可用的视觉 provider")
        if not frames:
            raise VisionError("没有抽到可用的动图帧")
        image_urls = [frame.data_url for frame in frames]
        try:
            response = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    image_urls=image_urls,
                    contexts=[],
                ),
                timeout=max(1.0, float(timeout_sec)),
            )
        except asyncio.TimeoutError as exc:
            self.last_vision_error = "视觉模型超时"
            raise VisionError(self.last_vision_error) from exc
        except Exception as exc:
            self.last_vision_error = f"{type(exc).__name__}: {exc}"
            raise VisionError(self.last_vision_error) from exc

        text = str(getattr(response, "completion_text", "") or "").strip()
        if not text:
            text = str(getattr(response, "reasoning_content", "") or "").strip()
        if not text:
            self.last_vision_error = "视觉模型返回空内容"
            raise VisionError(self.last_vision_error)
        self.last_vision_error = ""
        return text

    def vision_info(self, provider: Any = None) -> dict[str, Any]:
        if provider is None:
            return {
                "id": "",
                "model": "",
                "configured_id": self.config.provider_id,
            }
        return {
            "id": self._provider_id(provider),
            "model": self._provider_model(provider),
            "configured_id": self.config.provider_id,
        }

    # ---------------------------------------------------------- embeddings
    def embedding_provider(self) -> Any | None:
        getter = getattr(self.context, "get_all_embedding_providers", None)
        if not callable(getter):
            return None
        try:
            providers = list(getter() or [])
        except Exception:
            return None
        if not providers:
            return None
        if self.config.embedding_provider_id:
            for provider in providers:
                if self._provider_id(provider) == self.config.embedding_provider_id:
                    return provider
        for provider in providers:
            if callable(getattr(provider, "get_embedding", None)):
                return provider
        return None

    def embedding_info(self) -> dict[str, Any]:
        provider = self.embedding_provider()
        if provider is None:
            return {
                "id": "",
                "dimension": 0,
                "enabled": self.config.embedding_enabled,
                "available": False,
            }
        dimension = 0
        getter = getattr(provider, "get_dim", None)
        if callable(getter):
            try:
                dimension = int(getter())
            except Exception:
                dimension = 0
        return {
            "id": self._provider_id(provider),
            "model": self._provider_model(provider),
            "dimension": dimension,
            "enabled": self.config.embedding_enabled,
            "available": True,
        }

    async def embed(self, text: str) -> list[float] | None:
        if not self.config.embedding_enabled:
            return None
        provider = self.embedding_provider()
        if provider is None:
            self.last_embedding_error = "没有可用的 embedding provider"
            return None
        try:
            values = await asyncio.wait_for(
                provider.get_embedding(text),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            self.last_embedding_error = "embedding 超时"
            return None
        except Exception as exc:
            self.last_embedding_error = f"{type(exc).__name__}: {exc}"
            return None
        if not values:
            self.last_embedding_error = "embedding 返回空向量"
            return None
        self.last_embedding_error = ""
        return normalize_embedding([float(item) for item in values])


__all__ = ["ProviderResolver", "VisionError"]
