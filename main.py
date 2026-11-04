"""AstrBot plugin: understand QQ GIFs, remember them, and show a WebUI library."""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Reply
from astrbot.api.star import Context, Star

try:
    from astrbot.core.star.filter.command import GreedyStr
except Exception:  # pragma: no cover - older AstrBot compatibility
    GreedyStr = str  # type: ignore[assignment,misc]

from .gif_vision.companion import CompanionBridge
from .gif_vision.config import GifConfig
from .gif_vision.detect import (
    existing_description,
    is_animated_image,
    is_group_event,
    mface_candidates,
    question_from_event,
    read_local_file,
    read_ref_bytes,
)
from .gif_vision.engine import GifEngine, ProcessedGif
from .gif_vision.inject import (
    append_text_part,
    get_results,
    is_done,
    is_processed,
    mark_done,
    mark_processed,
    remove_image_refs,
    request_has_text,
    set_results,
)
from .gif_vision.library import GifLibrary
from .gif_vision.prompt import sentinel_for, short_preview, wrap_for_injection
from .gif_vision.provider import ProviderResolver
from .gif_vision.webui import GifWebUI

PLUGIN_NAME = "astrbot_plugin_gif_vision"
COMPANION_HOOK_PRIORITY = -259500
DIRECT_HOOK_PRIORITY = -300000


class GifVisionPlugin(Star):
    PLUGIN_NAME = PLUGIN_NAME

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.context = context
        self.raw_config = dict(config or {})
        self.config = GifConfig.from_mapping(self.raw_config)
        self.logger = logger
        self.PLUGIN_NAME = PLUGIN_NAME
        self._started_at = time.time()

        if self.config.companion_hook_priority != COMPANION_HOOK_PRIORITY:
            logger.warning(
                "[GifVision] companion_hook_priority 配置为 %s，但钩子优先级在类定义时固定为 %s；"
                "如需改序请直接修改代码常量。",
                self.config.companion_hook_priority,
                COMPANION_HOOK_PRIORITY,
            )

        self.library: GifLibrary | None = None
        if self.config.library_enable:
            try:
                self.library = GifLibrary(self._library_root(), logger)
            except Exception as exc:  # noqa: BLE001
                logger.error("[GifVision] 初始化 GIF 库失败: %s", exc, exc_info=True)
        self.companion = CompanionBridge(context, self.config, logger)
        self.providers = ProviderResolver(context, self.config, logger)
        self.engine = GifEngine(
            self.config, self.library, self.providers, logger
        )
        self._companion_retry_task: asyncio.Task | None = None
        self.webui = GifWebUI(self)
        self.webui.register(context)

        logger.info(
            "[GifVision] 初始化完成 | scope=%s | frames=%d | library=%s | companion=%s",
            self.config.session_scope,
            self.config.frame_count,
            "on" if self.library is not None else "off",
            "on" if self.config.return_to_companion else "off",
        )

    # ------------------------------------------------------------------ setup
    def _library_root(self) -> Path:
        try:
            from astrbot.core.utils.astrbot_path import (
                get_astrbot_plugin_data_path,
            )

            base = Path(get_astrbot_plugin_data_path())
        except Exception:
            base = Path.home() / ".astrbot" / "data" / "plugin_data"
        custom = self.config.library_dir.strip()
        if custom:
            path = Path(custom).expanduser()
            if not path.is_absolute():
                path = base / path
            return path
        return base / PLUGIN_NAME

    @filter.on_astrbot_loaded()
    async def _on_astrbot_loaded(self) -> None:
        registered = False
        if self.companion is not None:
            try:
                registered = self.companion.register_extension()
            except Exception:
                registered = False
        if not registered and self.companion is not None:
            self._companion_retry_task = asyncio.create_task(
                self._retry_companion_registration()
            )
        if self.library is not None:
            try:
                await self.library.prune(
                    retention_days=self.config.retention_days,
                    max_mb=self.config.library_max_mb,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[GifVision] 启动清理失败: %s", exc)
        logger.info(
            "[GifVision] ready | version=0.1.0 | companion=%s",
            self.companion.status() if self.companion else {},
        )

    async def _retry_companion_registration(self) -> None:
        for _ in range(12):
            await asyncio.sleep(5)
            try:
                if self.companion.register_extension():
                    logger.info(
                        "[GifVision] 陪伴扩展已注册 | source=%s",
                        self.config.companion_source_id,
                    )
                    return
            except Exception:
                continue

    async def terminate(self) -> None:
        task = self._companion_retry_task
        if task is not None and not task.done():
            task.cancel()
        try:
            if self.library is not None:
                self.library.close()
        except Exception:
            pass

    # --------------------------------------------------------------- hook one
    @filter.on_llm_request(priority=COMPANION_HOOK_PRIORITY)
    async def contribute_to_companion(
        self, event: AstrMessageEvent, req: Any
    ) -> None:
        try:
            await self._handle_companion(event, req)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[GifVision] 陪伴回传钩子异常: %s", exc, exc_info=True
            )

    async def _handle_companion(self, event: Any, req: Any) -> None:
        if req is None or not self.config.enable:
            return
        if not self.config.return_to_companion:
            return
        if not self.config.should_handle_event(
            is_group=is_group_event(event)
        ):
            return
        if is_done(req) or is_processed(req):
            return
        if (
            self.config.skip_if_existing_description
            and existing_description(req)
        ):
            mark_done(req)
            return
        if not self.companion.is_available():
            return

        candidates = await self._collect_candidates(event, req)
        if not candidates:
            return
        question = question_from_event(event, req)
        results = await self._process_candidates(
            event, req, candidates, question
        )
        if not results:
            return

        is_group = is_group_event(event)
        contributions: list[dict[str, Any]] = []
        for item, _candidate in results:
            ok = self.companion.contribute(
                req,
                event,
                description=item.description,
                sha_short=item.sha_short,
                is_group=is_group,
                question=question,
            )
            contributions.append(
                {
                    "sha": item.sha256,
                    "sha_short": item.sha_short,
                    "sentinel": sentinel_for(item.sha_short),
                    "description": item.description,
                    "record_id": item.record.id if item.record else None,
                    "ok": bool(ok),
                }
            )
            if self.config.debug_log:
                logger.info(
                    "[GifVision] companion contribution | sha=%s | ok=%s",
                    item.sha_short,
                    ok,
                )
        set_results(event, contributions)
        mark_processed(req)
        logger.info(
            "[GifVision] companion contribution staged | gifs=%d | ok=%d",
            len(contributions),
            sum(1 for item in contributions if item["ok"]),
        )

    # --------------------------------------------------------------- hook two
    @filter.on_llm_request(priority=DIRECT_HOOK_PRIORITY)
    async def inject_direct(
        self, event: AstrMessageEvent, req: Any
    ) -> None:
        try:
            await self._handle_direct(event, req)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[GifVision] 直接注入钩子异常: %s", exc, exc_info=True
            )

    async def _handle_direct(self, event: Any, req: Any) -> None:
        if req is None or not self.config.enable:
            return
        if not self.config.should_handle_event(
            is_group=is_group_event(event)
        ):
            return
        if is_done(req):
            return

        stored = get_results(event)
        if stored:
            missing = []
            for item in stored:
                sentinel = str(item.get("sentinel") or "")
                if item.get("ok") and sentinel and request_has_text(req, sentinel):
                    continue
                if item.get("description"):
                    missing.append(item)
            for index, item in enumerate(missing, 1):
                append_text_part(
                    req,
                    wrap_for_injection(
                        str(item.get("description") or ""),
                        index=index,
                        total=len(missing),
                    ),
                )
            mark_done(req)
            if missing:
                logger.info(
                    "[GifVision] direct fallback | missing=%d | companion_images_untouched",
                    len(missing),
                )
            else:
                logger.info(
                    "[GifVision] companion contribution verified | images_untouched"
                )
            return

        if is_processed(req):
            mark_done(req)
            return
        if (
            self.config.skip_if_existing_description
            and existing_description(req)
        ):
            mark_done(req)
            return

        candidates = await self._collect_candidates(event, req)
        if not candidates:
            mark_done(req)
            return
        question = question_from_event(event, req)
        results = await self._process_candidates(
            event, req, candidates, question
        )
        if not results:
            mark_done(req)
            return

        for index, (item, _candidate) in enumerate(results, 1):
            append_text_part(
                req,
                wrap_for_injection(
                    item.description,
                    index=index,
                    total=len(results),
                ),
            )

        # Companion mode owns image_urls. Standalone mode may drop the raw GIF
        # after a successful description to avoid the core montage duplicate.
        companion_active = (
            self.config.return_to_companion and self.companion.is_available()
        )
        if (
            not companion_active
            and not self.config.keep_original_image
        ):
            refs = {
                candidate.ref
                for _item, candidate in results
                if candidate.where != "mface"
            }
            removed = remove_image_refs(req, refs)
            if self.config.debug_log:
                logger.info("[GifVision] removed %d original refs", removed)
        mark_done(req)
        logger.info(
            "[GifVision] direct injection | gifs=%d | companion=%s",
            len(results),
            companion_active,
        )

    # -------------------------------------------------------------- utilities
    def _source(self, event: Any) -> str:
        return "qq_group" if is_group_event(event) else "qq_private"

    async def _collect_candidates(
        self, event: Any, req: Any
    ) -> list[Any]:
        from .gif_vision.detect import iter_request_image_refs

        candidates = list(iter_request_image_refs(req))
        if self.config.handle_mface:
            seen = {candidate.ref for candidate in candidates}
            for candidate in mface_candidates(event):
                if candidate.ref not in seen:
                    candidates.append(candidate)
                    seen.add(candidate.ref)
        return candidates[: self.config.max_gifs_per_message]

    async def _process_candidates(
        self,
        event: Any,
        req: Any,
        candidates: list[Any],
        question: str,
    ) -> list[tuple[ProcessedGif, Any]]:
        results: list[tuple[ProcessedGif, Any]] = []
        for candidate in candidates:
            if len(results) >= self.config.max_gifs_per_message:
                break
            data = await read_ref_bytes(candidate.ref)
            if not data or not is_animated_image(data):
                continue
            item = await self.engine.process_bytes(
                data,
                source=self._source(event),
                question=question,
                summary=candidate.summary,
                event=event,
            )
            if item is not None:
                results.append((item, candidate))
        return results

    # --------------------------------------------------------------- commands
    @staticmethod
    def _admin(event: Any) -> bool:
        return str(getattr(event, "role", "") or "").lower() == "admin"

    @staticmethod
    def _message_images(event: Any) -> list[Any]:
        components = list(getattr(event.message_obj, "message", None) or [])
        extra: list[Any] = []
        for component in components:
            if isinstance(component, Reply):
                chain = getattr(component, "chain", None) or []
                extra.extend(chain)
        return [
            component
            for component in [*components, *extra]
            if isinstance(component, Image)
        ]

    @staticmethod
    async def _component_bytes(component: Any) -> bytes | None:
        path = getattr(component, "path", None)
        if isinstance(path, str) and path:
            try:
                return Path(path).read_bytes()
            except OSError:
                pass
        try:
            path = await component.convert_to_file_path()
            return Path(path).read_bytes()
        except Exception:
            pass
        try:
            encoded = await component.convert_to_base64()
            return base64.b64decode(encoded)
        except Exception:
            return None

    @filter.command("gif", alias={"动图"})
    async def cmd_gif(self, event: AstrMessageEvent):
        if not self.config.enable:
            yield event.plain_result("动图识别已关闭。")
            return
        images = self._message_images(event)
        if not images:
            yield event.plain_result(
                "用法：引用一张动图发送 /gif，或直接发送动图并 @我。"
            )
            return
        results: list[ProcessedGif] = []
        for component in images[: self.config.max_gifs_per_message]:
            data = await self._component_bytes(component)
            if not data or not is_animated_image(data):
                continue
            item = await self.engine.process_bytes(
                data,
                source=self._source(event),
                question="",
                event=event,
            )
            if item is not None:
                results.append(item)
        if not results:
            yield event.plain_result("没有识别到可处理的动图。")
            return
        lines = ["动图识别结果："]
        for item in results:
            record_id = f"#{item.record.id} " if item.record else ""
            lines.append(f"{record_id}{short_preview(item.description, 600)}")
        yield event.plain_result("\n".join(lines))

    @filter.command("gifimport", alias={"导入动图"})
    async def cmd_gifimport(self, event: AstrMessageEvent, path: GreedyStr):
        if not self._admin(event):
            yield event.plain_result("该指令仅管理员可用。")
            return
        data = read_local_file(str(path).strip().strip('"').strip("'"))
        if not data:
            yield event.plain_result("读取文件失败。")
            return
        item = await self.engine.process_bytes(
            data,
            source="local",
            question="",
            event=event,
        )
        if item is None:
            yield event.plain_result(
                f"处理失败：{self.engine.last_error or '不是动图或视觉理解失败'}"
            )
            return
        record_id = f"#{item.record.id} " if item.record else ""
        yield event.plain_result(
            f"已导入并保存 {record_id}\n{short_preview(item.description, 800)}"
        )

    @filter.command("gifsearch", alias={"找动图"})
    async def cmd_gifsearch(self, event: AstrMessageEvent, query: GreedyStr):
        query = str(query).strip()
        if not query:
            yield event.plain_result("用法：/gifsearch <关键词或描述>")
            return
        results, mode = await self.engine.search(query)
        if not results:
            yield event.plain_result("没有找到匹配的动图。")
            return
        lines = [f"匹配到 {len(results)} 条（{mode}）："]
        for index, (record, score) in enumerate(results, 1):
            score_text = f" {score:.3f}" if mode == "vector" else ""
            lines.append(
                f"{index}. #{record.id}{score_text} "
                f"{short_preview(record.description, 100)}"
            )
        yield event.plain_result("\n".join(lines))

    @filter.command("gifsend", alias={"发动图"})
    async def cmd_gifsend(self, event: AstrMessageEvent, query: GreedyStr):
        if self.library is None:
            yield event.plain_result("GIF 库未启用。")
            return
        raw = str(query).strip()
        if not raw:
            yield event.plain_result("用法：/gifsend <id 或描述>")
            return
        record = None
        score = 0.0
        mode = "id"
        if raw.isdigit():
            record = await self.library.get_by_id(int(raw))
        else:
            results, mode = await self.engine.search(raw, top_k=1)
            if results:
                record, score = results[0]
                if mode == "vector" and score < self.config.search_min_score:
                    record = None
        if record is None:
            yield event.plain_result("没有找到匹配的动图。")
            return
        if record.size > int(self.config.send_max_mb * 1024 * 1024):
            yield event.plain_result(
                f"该动图超过发送上限 {self.config.send_max_mb:g}MB。"
            )
            return
        path = await self.library.path_for(record.id)
        if path is None:
            yield event.plain_result("动图文件不存在。")
            return
        yield event.image_result(str(path))

    @filter.command("gifstats", alias={"动图统计"})
    async def cmd_gifstats(self, event: AstrMessageEvent):
        if not self._admin(event):
            yield event.plain_result("该指令仅管理员可用。")
            return
        if self.library is None:
            yield event.plain_result("GIF 库未启用。")
            return
        stats = await self.library.stats()
        embedding = self.providers.embedding_info()
        companion = self.companion.status()
        lines = [
            "动图库统计：",
            f"总数：{stats.get('total', 0)}",
            f"磁盘：{stats.get('total_bytes', 0) / 1024 / 1024:.1f} MB",
            f"有向量：{stats.get('with_embedding', 0)} / 缺向量：{stats.get('without_embedding', 0)}",
            f"embedding：{embedding.get('id') or '-'} ({embedding.get('dimension', 0)} dim)",
            f"陪伴插件：{'已连接' if companion.get('available') else '未连接'}",
            f"最近错误：{self.engine.last_error or '无'}",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("gifforget", alias={"忘掉动图"})
    async def cmd_gifforget(self, event: AstrMessageEvent, record_id: GreedyStr):
        if not self._admin(event):
            yield event.plain_result("该指令仅管理员可用。")
            return
        if self.library is None:
            yield event.plain_result("GIF 库未启用。")
            return
        raw = str(record_id).strip()
        if not raw.isdigit():
            yield event.plain_result("用法：/gifforget <id>")
            return
        removed = await self.library.delete(int(raw))
        yield event.plain_result("已删除。" if removed else "没有找到该记录。")


__all__ = ["GifVisionPlugin", "PLUGIN_NAME"]
