"""Bridge to the public extension API of the companion plugin.

The companion plugin (display name “我会永远陪着你”) publishes
``get_private_companion_api()`` from its module and accepts typed
``ContextContribution`` sections through ``add_context_contribution``.  We only
use that public protocol; this module never imports or patches the companion
internals.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from .config import GifConfig
from .prompt import build_contribution_content, sentinel_for


class CompanionBridge:
    def __init__(
        self,
        context: Any,
        config: GifConfig,
        logger: Any = None,
        *,
        api: Any = None,
    ) -> None:
        self.context = context
        self.config = config
        self.logger = logger
        self._api = api
        self._registered = False
        self._last_error = ""
        self._last_success = 0.0
        self._contribution_count = 0

    # ------------------------------------------------------------- lookup
    @staticmethod
    def _is_active(api: Any) -> bool:
        lifecycle = getattr(api, "bridge_lifecycle_status", None)
        if not callable(lifecycle):
            return True
        try:
            status = lifecycle()
        except Exception:
            return False
        return isinstance(status, dict) and status.get("active") is True

    def _from_modules(self) -> Any | None:
        names = (
            "data.plugins.astrbot_plugin_private_companion.main",
            "astrbot_plugin_private_companion.main",
        )
        suffixes = tuple(name.removeprefix("data.plugins.") for name in names)
        modules = [sys.modules.get(name) for name in names]
        modules.extend(
            module
            for name, module in list(sys.modules.items())
            if module is not None and any(name.endswith(suffix) for suffix in suffixes)
        )
        for module in modules:
            if module is None:
                continue
            getter = getattr(module, "get_private_companion_api", None)
            if not callable(getter):
                continue
            try:
                api = getter()
            except Exception:
                api = None
            if api is not None and self._is_active(api):
                return api
        return None

    def _from_registry(self) -> Any | None:
        getter = getattr(self.context, "get_registered_star", None)
        if not callable(getter):
            return None
        try:
            metadata = getter("astrbot_plugin_private_companion")
            instance = getattr(metadata, "star_cls", None) if metadata else None
            api = getattr(instance, "extension_api", None)
        except Exception:
            return None
        if api is not None and self._is_active(api):
            return api
        return None

    def resolve_api(self) -> Any | None:
        if self._api is not None:
            if self._is_active(self._api):
                return self._api
            self._api = None
        api = self._from_modules() or self._from_registry()
        if api is not None:
            self._api = api
        return api

    def is_available(self) -> bool:
        return self.resolve_api() is not None

    # --------------------------------------------------------- control plane
    def register_extension(self) -> bool:
        if self._registered or not self.config.companion_register_extension:
            return self._registered
        api = self.resolve_api()
        if api is None:
            return False
        register = getattr(api, "register_extension", None)
        if not callable(register):
            return False
        try:
            result = register(
                {
                    "id": self.config.companion_source_id,
                    "version": "0.1",
                    "sdk_version": "0.1",
                    "display_name": "看动图",
                    "supported_platforms": ["aiocqhttp"],
                }
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self.logger:
                self.logger.debug("[GifVision] 陪伴扩展注册失败: %s", exc)
            return False
        if isinstance(result, dict) and result.get("ok") is False:
            self._last_error = str(result.get("reason") or "register_extension failed")
            return False
        self._registered = True
        return True

    # ---------------------------------------------------------- contribution
    def contribute(
        self,
        req: Any,
        event: Any,
        *,
        description: str,
        sha_short: str,
        is_group: bool,
        question: str = "",
    ) -> bool:
        if not self.config.return_to_companion:
            return False
        api = self.resolve_api()
        if api is None:
            return False
        self.register_extension()
        add = getattr(api, "add_context_contribution", None)
        if not callable(add):
            self._last_error = "companion API has no add_context_contribution"
            return False
        content = build_contribution_content(description, sha_short, question)
        contribution = {
            "lane": self.config.companion_lane,
            "key": f"gif_{sha_short}",
            "content": content,
            "evidence": self.config.companion_source_id,
            "priority": int(self.config.companion_priority),
            "max_age_seconds": int(self.config.companion_max_age_seconds),
            "visibility": "public" if is_group else "private",
            "source": self.config.companion_source_id,
            "source_refs": [f"gif:{sha_short}"],
            "revision": sha_short,
        }
        try:
            ok = bool(
                add(
                    req,
                    contribution,
                    event=event,
                    source_id=self.config.companion_source_id,
                )
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self.logger:
                self.logger.debug("[GifVision] 陪伴上下文注入失败: %s", exc)
            return False
        if ok:
            self._last_error = ""
            self._last_success = time.time()
            self._contribution_count += 1
        else:
            self._last_error = "add_context_contribution returned false"
        return ok

    @staticmethod
    def sentinel(sha_short: str) -> str:
        return sentinel_for(sha_short)

    def status(self) -> dict[str, Any]:
        api = self.resolve_api()
        return {
            "available": api is not None,
            "source_id": self.config.companion_source_id,
            "registered": self._registered,
            "contribution_count": self._contribution_count,
            "last_success": self._last_success,
            "last_error": self._last_error,
            "enabled": self.config.return_to_companion,
        }


__all__ = ["CompanionBridge"]
