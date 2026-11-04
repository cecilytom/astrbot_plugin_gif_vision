"""WebUI APIs for the GIF library page.

The page itself lives in ``pages/gif_library`` and talks to these handlers
through AstrBot's plugin-page bridge.  Handlers are registered with
``Context.register_web_api`` under ``/<plugin_name>/<route>``; the WebUI exposes
them as ``/api/v1/plugins/extensions/<plugin_name>/<route>``.
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from pathlib import Path
from typing import Any

try:  # Quart is present in the AstrBot dashboard runtime.
    from quart import jsonify, request, send_file
except Exception:  # pragma: no cover - allows pure tests without Quart
    jsonify = None  # type: ignore[assignment]
    request = None  # type: ignore[assignment]
    send_file = None  # type: ignore[assignment]

from .config import GifConfig
from .frames import gif_data_url, thumbnail_data_url
from .library import GifRecord


class WebUIError(RuntimeError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class GifWebUI:
    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self.config: GifConfig = plugin.config
        self.logger = getattr(plugin, "logger", None)

    # --------------------------------------------------------------- routes
    def register(self, context: Any) -> None:
        routes = [
            ("library/stats", self._api_stats, ["GET"], "GIF library stats"),
            ("library/list", self._api_list, ["GET"], "List or search GIFs"),
            ("library/item", self._api_item, ["GET"], "Read one GIF record"),
            ("library/thumb", self._api_thumb, ["GET"], "Read GIF thumbnail"),
            ("library/import", self._api_import, ["POST"], "Import a GIF"),
            ("library/delete", self._api_delete, ["POST"], "Delete a GIF"),
            (
                "library/redescribe",
                self._api_redescribe,
                ["POST"],
                "Re-describe a GIF",
            ),
            (
                "library/rebuild_embeddings",
                self._api_rebuild_embeddings,
                ["POST"],
                "Rebuild missing embeddings",
            ),
            ("library/export", self._api_export, ["GET"], "Export GIF metadata"),
            ("library/health", self._api_health, ["GET"], "GIF service health"),
        ]
        for route, handler, methods, desc in routes:
            self._register(context, route, handler, methods, desc)
        self._register_raw(
            context,
            "library/download",
            self._api_download,
            ["GET"],
            "Download the original GIF",
        )

    def _register(
        self,
        context: Any,
        route: str,
        handler: Any,
        methods: list[str],
        desc: str,
    ) -> None:
        async def wrapped() -> Any:
            try:
                data = await handler()
            except WebUIError as exc:
                return self._error(str(exc), exc.status)
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.warning(
                        "[GifVision][WebUI] %s failed: %s",
                        route,
                        exc,
                        exc_info=True,
                    )
                return self._error(f"接口执行失败: {type(exc).__name__}")
            return jsonify({"status": "ok", "data": data})

        wrapped.__name__ = f"gif_vision_{route.replace('/', '_')}"
        context.register_web_api(
            f"/{getattr(self.plugin, 'PLUGIN_NAME', 'astrbot_plugin_gif_vision')}/{route}",
            wrapped,
            methods,
            desc,
        )

    def _register_raw(
        self,
        context: Any,
        route: str,
        handler: Any,
        methods: list[str],
        desc: str,
    ) -> None:
        async def wrapped() -> Any:
            try:
                return await handler()
            except WebUIError as exc:
                return self._error(str(exc), exc.status)
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.warning(
                        "[GifVision][WebUI] %s failed: %s",
                        route,
                        exc,
                        exc_info=True,
                    )
                return self._error(f"接口执行失败: {type(exc).__name__}")

        wrapped.__name__ = f"gif_vision_{route.replace('/', '_')}"
        context.register_web_api(
            f"/{getattr(self.plugin, 'PLUGIN_NAME', 'astrbot_plugin_gif_vision')}/{route}",
            wrapped,
            methods,
            desc,
        )

    @staticmethod
    def _error(message: str, status: int = 400) -> Any:
        response = jsonify({"status": "error", "message": message})
        try:
            return response, status
        except Exception:
            return response

    @property
    def library(self) -> Any:
        return getattr(self.plugin, "library", None)

    @property
    def engine(self) -> Any:
        return getattr(self.plugin, "engine", None)

    @property
    def providers(self) -> Any:
        return getattr(self.plugin, "providers", None)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _query_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(request.args.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _query_bool(name: str, default: bool = False) -> bool:
        raw = str(request.args.get(name, "1" if default else "0")).lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _record_id(raw: Any) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise WebUIError("缺少合法的 GIF id")
        if value <= 0:
            raise WebUIError("缺少合法的 GIF id")
        return value

    async def _body(self) -> dict[str, Any]:
        try:
            data = await request.get_json(silent=True)
        except Exception:
            data = None
        return data if isinstance(data, dict) else {}

    async def _read_record(self, record_id: int) -> GifRecord:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        record = await self.library.get_by_id(record_id)
        if record is None:
            raise WebUIError("GIF 不存在", 404)
        return record

    # ------------------------------------------------------------- handlers
    async def _api_stats(self) -> dict[str, Any]:
        if getattr(self.plugin, "companion", None) is not None:
            try:
                self.plugin.companion.register_extension()
            except Exception:
                pass
        library = self.library
        stats = await library.stats() if library is not None else {}
        # Do not expose absolute paths in the page payload.
        stats.pop("root", None)
        stats.pop("db_path", None)
        if library is not None:
            stats["root_name"] = library.root.name
            stats["db_name"] = library.db_path.name
        provider = None
        if self.providers is not None:
            try:
                provider = await self.providers.vision_provider(None)
            except Exception:
                provider = None
        return {
            "library": stats,
            "vision": (
                self.providers.vision_info(provider)
                if self.providers is not None
                else {}
            ),
            "embedding": (
                self.providers.embedding_info()
                if self.providers is not None
                else {}
            ),
            "companion": (
                self.plugin.companion.status()
                if getattr(self.plugin, "companion", None) is not None
                else {}
            ),
            "cache": (
                self.engine.cache_stats()
                if self.engine is not None
                else {}
            ),
            "last_error": (
                self.engine.last_error if self.engine is not None else ""
            ),
            "config": {
                "provider_id": self.config.provider_id,
                "embedding_enabled": self.config.embedding_enabled,
                "return_to_companion": self.config.return_to_companion,
                "library_enable": self.config.library_enable,
                "frame_count": self.config.frame_count,
                "webui_import_enable": self.config.webui_import_enable,
            },
        }

    async def _api_list(self) -> dict[str, Any]:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        page = self._query_int("page", 1, 1, 100000)
        page_size = self._query_int(
            "page_size", self.config.webui_page_size, 1, 50
        )
        query = str(request.args.get("query", "") or "").strip()[:200]
        sort = str(request.args.get("sort", "recent") or "recent")
        missing = self._query_bool("missing_embedding", False)
        prefer_like = (
            str(request.args.get("mode", "auto") or "auto").lower() == "like"
        )

        if query and not prefer_like:
            results, mode = await self.engine.search(
                query, top_k=max(page_size, self.config.search_top_k)
            )
            if mode == "vector":
                items = []
                for record, score in results:
                    item = record.public_dict()
                    item["score"] = round(float(score), 4)
                    items.append(item)
                return {
                    "items": items,
                    "total": len(items),
                    "page": 1,
                    "page_size": page_size,
                    "mode": mode,
                    "query": query,
                }

        records, total = await self.library.list_records(
            page=page,
            page_size=page_size,
            query=query,
            sort=sort,
            missing_embedding=missing,
        )
        return {
            "items": [record.public_dict() for record in records],
            "total": total,
            "page": page,
            "page_size": page_size,
            "mode": "like" if query else "list",
            "query": query,
        }

    async def _api_item(self) -> dict[str, Any]:
        record = await self._read_record(
            self._record_id(request.args.get("id"))
        )
        item = record.public_dict()
        item["command"] = f"/gifsend {record.id}"
        item["sha256"] = record.sha256
        item["phash"] = record.phash
        item["canonical_id"] = record.canonical_id
        preview_path = None
        if self.library is not None:
            try:
                preview_path = await self.library.ensure_thumb(
                    record.id, self.config.webui_thumb_edge
                )
            except Exception:
                preview_path = None
        if preview_path and Path(preview_path).exists():
            try:
                item["preview_data_url"] = thumbnail_data_url(
                    Path(preview_path).read_bytes()
                )
            except OSError:
                item["preview_data_url"] = None
        else:
            item["preview_data_url"] = None

        item["gif_data_url"] = None
        item["too_large_for_preview"] = False
        item["can_preview"] = False
        include_gif = self._query_bool("include_gif", False)
        if include_gif:
            limit = int(self.config.webui_preview_max_mb * 1024 * 1024)
            if record.size > limit:
                item["too_large_for_preview"] = True
            elif self.library is not None:
                path = await self.library.path_for(record.id)
                if path is not None:
                    try:
                        data = await asyncio.to_thread(path.read_bytes)
                        item["gif_data_url"] = gif_data_url(data)
                        item["can_preview"] = True
                    except OSError:
                        item["gif_data_url"] = None
        return item

    async def _api_thumb(self) -> dict[str, Any]:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        record_id = self._record_id(request.args.get("id"))
        path = await self.library.ensure_thumb(
            record_id, self.config.webui_thumb_edge
        )
        if path is None or not Path(path).exists():
            raise WebUIError("缩略图不可用", 404)
        try:
            data = await asyncio.to_thread(Path(path).read_bytes)
        except OSError as exc:
            raise WebUIError(f"缩略图读取失败: {exc}", 500)
        return {"data_url": thumbnail_data_url(data)}

    async def _api_download(self) -> Any:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        record_id = self._record_id(request.args.get("id"))
        path = await self.library.path_for(record_id)
        if path is None:
            raise WebUIError("GIF 文件不存在", 404)
        if send_file is None:
            raise WebUIError("当前环境不支持文件下载", 503)
        return await send_file(
            str(path),
            mimetype="image/gif",
            as_attachment=True,
            attachment_filename=f"gif_{record_id}.gif",
        )

    async def _api_import(self) -> dict[str, Any]:
        if not self.config.webui_import_enable:
            raise WebUIError("WebUI 导入已关闭", 403)
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        files = await request.files
        upload = files.get("file")
        if upload is None:
            raise WebUIError("缺少上传文件字段 file")
        filename = str(getattr(upload, "filename", "") or "upload.gif")
        suffix = Path(filename).suffix.lower() or ".gif"
        if suffix not in {".gif", ".png", ".webp", ".apng"}:
            raise WebUIError("只支持 GIF/APNG/WebP 动图")
        temp_path = self.library.tmp_dir / f"upload_{uuid.uuid4().hex}{suffix}"
        try:
            await upload.save(str(temp_path))
            size = temp_path.stat().st_size
            if size > int(self.config.max_gif_mb * 1024 * 1024):
                raise WebUIError("文件超过 max_gif_mb 限制", 413)
            data = await asyncio.to_thread(temp_path.read_bytes)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        result = await self.engine.process_bytes(
            data, source="webui", question="", event=None
        )
        if result is None:
            raise WebUIError(
                f"无法识别的动图或视觉理解失败: {self.engine.last_error or '未知原因'}"
            )
        item = (
            result.record.public_dict()
            if result.record is not None
            else {
                "id": None,
                "sha256_short": result.sha_short,
                "description": result.description,
            }
        )
        item["new"] = result.is_new
        item["near_hit"] = result.near_hit
        return item

    async def _api_delete(self) -> dict[str, Any]:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        body = await self._body()
        record_id = self._record_id(body.get("id"))
        removed = await self.library.delete(record_id)
        if not removed:
            raise WebUIError("GIF 不存在", 404)
        return {"deleted": record_id}

    async def _api_redescribe(self) -> dict[str, Any]:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        body = await self._body()
        record = await self._read_record(self._record_id(body.get("id")))
        path = await self.library.path_for(record.id)
        if path is None:
            raise WebUIError("GIF 文件不存在", 404)
        data = await asyncio.to_thread(path.read_bytes)
        result = await self.engine.redescribe(record, data, source="webui")
        if result is None:
            raise WebUIError(
                f"重新描述失败: {self.engine.last_error or '未知原因'}"
            )
        return result.record.public_dict() if result.record else {
            "description": result.description
        }

    async def _api_rebuild_embeddings(self) -> dict[str, Any]:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        if self.providers is None or not self.config.embedding_enabled:
            raise WebUIError("embedding 未启用")
        body = await self._body()
        try:
            limit = int(body.get("limit") or 20)
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(50, limit))
        info = self.providers.embedding_info()
        dimension = int(info.get("dimension") or 0)
        records = await self.library.records_missing_embeddings(dimension, limit)
        updated = 0
        failed = 0
        for record in records:
            vector = await self.providers.embed(record.description)
            if not vector:
                failed += 1
                continue
            from .fingerprint import embedding_to_blob

            await self.library.update_embedding(
                record.id,
                embedding=embedding_to_blob(vector),
                dimension=len(vector),
                model=str(info.get("id") or ""),
            )
            updated += 1
        return {
            "scanned": len(records),
            "updated": updated,
            "failed": failed,
            "embedding_provider": info.get("id", ""),
        }

    async def _api_export(self) -> list[dict[str, Any]]:
        if self.library is None:
            raise WebUIError("持久化库未启用", 503)
        return await self.library.export_records()

    async def _api_health(self) -> dict[str, Any]:
        if getattr(self.plugin, "companion", None) is not None:
            try:
                self.plugin.companion.register_extension()
            except Exception:
                pass
        started = time.monotonic()
        embedding_ok = False
        dimension = 0
        if self.providers is not None and self.config.embedding_enabled:
            vector = await self.providers.embed("gif vision health check")
            embedding_ok = bool(vector)
            dimension = len(vector) if vector else 0
        vision_provider = None
        if self.providers is not None:
            try:
                vision_provider = await self.providers.vision_provider(None)
            except Exception:
                vision_provider = None
        return {
            "embedding_ok": embedding_ok,
            "embedding_dimension": dimension,
            "embedding_error": (
                self.providers.last_embedding_error
                if self.providers is not None
                else ""
            ),
            "vision": (
                self.providers.vision_info(vision_provider)
                if self.providers is not None
                else {}
            ),
            "companion": (
                self.plugin.companion.status()
                if getattr(self.plugin, "companion", None) is not None
                else {}
            ),
            "last_error": (
                self.engine.last_error if self.engine is not None else ""
            ),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }


__all__ = ["GifWebUI", "WebUIError"]
