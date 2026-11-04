"""GIF understanding orchestration shared by hooks, commands and WebUI."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .cache import L1Cache
from .config import GifConfig
from .detect import is_animated_image, read_ref_bytes
from .fingerprint import phash_of_frames, sha256_bytes
from .frames import Frame, extract_frames, make_thumbnail
from .library import GifLibrary, GifRecord
from .prompt import build_vision_prompt, clean_description
from .provider import ProviderResolver, VisionError


@dataclass
class ProcessedGif:
    description: str
    sha256: str
    sha_short: str
    phash: str
    record: GifRecord | None
    is_new: bool
    near_hit: bool
    elapsed_sec: float
    vision_provider: str = ""
    vision_model: str = ""
    frame_count: int = 0
    embedding: list[float] | None = None


class GifEngine:
    def __init__(
        self,
        config: GifConfig,
        library: GifLibrary | None,
        providers: ProviderResolver,
        logger: Any = None,
    ) -> None:
        self.config = config
        self.library = library
        self.providers = providers
        self.logger = logger
        self.last_error = ""
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrent))
        self._l1: L1Cache[ProcessedGif] = L1Cache(ttl_sec=120, max_entries=64)

    async def process_bytes(
        self,
        data: bytes,
        *,
        source: str,
        question: str = "",
        summary: str = "",
        event: Any = None,
        force_vision: bool = False,
    ) -> ProcessedGif | None:
        if not data or not is_animated_image(data):
            return None
        started = time.monotonic()
        sha = sha256_bytes(data)
        sha_short = sha[:12]
        phash = await asyncio.to_thread(phash_of_frames, data)
        question = (question or "").strip()[:300]

        if not force_vision:
            if self.library is not None:
                exact = await self.library.get_by_sha(sha)
                if exact is not None:
                    await self.library.touch(exact.id)
                    return ProcessedGif(
                        description=exact.description,
                        sha256=sha,
                        sha_short=sha_short,
                        phash=exact.phash or phash,
                        record=exact,
                        is_new=False,
                        near_hit=False,
                        elapsed_sec=time.monotonic() - started,
                        vision_provider=exact.vision_provider,
                        vision_model=exact.vision_model,
                        frame_count=exact.frames or 0,
                    )
                near = await self.library.find_near(
                    phash, self.config.near_duplicate_hamming
                )
                if near is not None and near.sha256 != sha:
                    if not self.config.near_duplicate_store:
                        await self.library.touch(near.id)
                        return ProcessedGif(
                            description=near.description,
                            sha256=sha,
                            sha_short=sha_short,
                            phash=phash,
                            record=near,
                            is_new=False,
                            near_hit=True,
                            elapsed_sec=time.monotonic() - started,
                            vision_provider=near.vision_provider,
                            vision_model=near.vision_model,
                            frame_count=near.frames or 0,
                        )
                    thumb = await asyncio.to_thread(
                        make_thumbnail, data, self.config.webui_thumb_edge
                    )
                    embedding = near.embedding
                    embedding_dim = near.embedding_dim
                    embedding_model = near.embedding_model
                    if embedding is None and self.config.embedding_enabled:
                        vector = await self.providers.embed(near.description)
                        if vector:
                            from .fingerprint import embedding_to_blob

                            embedding = embedding_to_blob(vector)
                            embedding_dim = len(vector)
                            embedding_model = self.providers.embedding_info().get(
                                "id", ""
                            )
                    try:
                        record = await self.library.store(
                            data=data,
                            sha256=sha,
                            phash=phash,
                            description=near.description,
                            embedding=embedding,
                            embedding_dim=embedding_dim,
                            embedding_model=embedding_model,
                            vision_provider=near.vision_provider,
                            vision_model=near.vision_model,
                            canonical_id=near.id,
                            source=source,
                            summary=summary,
                            thumb_data=thumb,
                        )
                    except Exception:
                        record = near
                    return ProcessedGif(
                        description=near.description,
                        sha256=sha,
                        sha_short=sha_short,
                        phash=phash,
                        record=record,
                        is_new=True,
                        near_hit=True,
                        elapsed_sec=time.monotonic() - started,
                        vision_provider=near.vision_provider,
                        vision_model=near.vision_model,
                        frame_count=near.frames or 0,
                        embedding=None,
                    )
            else:
                cached = self._l1.get(sha)
                if cached is not None:
                    return cached

        try:
            async with self._semaphore:
                frames = await asyncio.to_thread(
                    extract_frames,
                    data,
                    self.config.frame_count,
                    self.config.frame_max_edge,
                )
                if not frames:
                    self.last_error = "没有抽到可用的动图帧"
                    return None
                provider = await self.providers.vision_provider(event)
                if provider is None:
                    self.last_error = "没有可用的视觉 provider"
                    return None
                prompt = build_vision_prompt(
                    self.config,
                    question,
                    len(frames),
                    summary,
                )
                raw = await self.providers.describe(
                    provider,
                    prompt,
                    frames,
                    timeout_sec=self.config.vision_timeout_sec,
                )
                description = clean_description(
                    raw, self.config.max_output_chars
                )
                if not description:
                    self.last_error = "视觉模型返回空描述"
                    return None
                embedding = await self.providers.embed(description)
                thumb = await asyncio.to_thread(
                    make_thumbnail, data, self.config.webui_thumb_edge
                )
                vision_info = self.providers.vision_info(provider)
                record: GifRecord | None = None
                if self.library is not None:
                    from .fingerprint import embedding_to_blob

                    try:
                        record = await self.library.store(
                            data=data,
                            sha256=sha,
                            phash=phash,
                            description=description,
                            embedding=(
                                embedding_to_blob(embedding)
                                if embedding
                                else None
                            ),
                            embedding_dim=len(embedding) if embedding else None,
                            embedding_model=(
                                self.providers.embedding_info().get("id", "")
                                if embedding
                                else None
                            ),
                            vision_provider=vision_info.get("id", ""),
                            vision_model=vision_info.get("model", ""),
                            source=source,
                            summary=summary,
                            thumb_data=thumb,
                        )
                        await self.library.prune(
                            retention_days=self.config.retention_days,
                            max_mb=self.config.library_max_mb,
                        )
                    except Exception as exc:
                        self.last_error = f"入库失败: {type(exc).__name__}: {exc}"
                        if self.logger:
                            self.logger.warning("[GifVision] %s", self.last_error)
                result = ProcessedGif(
                    description=description,
                    sha256=sha,
                    sha_short=sha_short,
                    phash=phash,
                    record=record,
                    is_new=True,
                    near_hit=False,
                    elapsed_sec=time.monotonic() - started,
                    vision_provider=vision_info.get("id", ""),
                    vision_model=vision_info.get("model", ""),
                    frame_count=len(frames),
                    embedding=embedding,
                )
                if self.library is None:
                    self._l1.put(sha, result)
                self.last_error = ""
                return result
        except VisionError as exc:
            self.last_error = str(exc)
            if self.logger:
                self.logger.warning("[GifVision] 视觉理解失败: %s", exc)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.logger:
                self.logger.warning(
                    "[GifVision] 处理动图失败: %s",
                    self.last_error,
                    exc_info=True,
                )
            return None

    async def process_ref(
        self,
        ref: str,
        *,
        source: str,
        question: str = "",
        summary: str = "",
        event: Any = None,
        force_vision: bool = False,
    ) -> ProcessedGif | None:
        data = await read_ref_bytes(ref)
        if not data:
            return None
        return await self.process_bytes(
            data,
            source=source,
            question=question,
            summary=summary,
            event=event,
            force_vision=force_vision,
        )

    async def redescribe(
        self,
        record: GifRecord,
        data: bytes,
        *,
        source: str = "webui",
        question: str = "",
    ) -> ProcessedGif | None:
        result = await self.process_bytes(
            data,
            source=source,
            question=question,
            summary=record.summary,
            force_vision=True,
        )
        if result is None or self.library is None:
            return result
        from .fingerprint import embedding_to_blob

        # process_bytes stored a new record keyed by the same SHA only when it
        # did not already exist; update the existing row with the fresh prose.
        if result.embedding:
            embedding = embedding_to_blob(result.embedding)
            embedding_dim = len(result.embedding)
            embedding_model = self.providers.embedding_info().get("id", "")
        else:
            embedding = record.embedding
            embedding_dim = record.embedding_dim
            embedding_model = record.embedding_model
        updated = await self.library.update_description(
            record.id,
            description=result.description,
            embedding=embedding,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
            vision_provider=result.vision_provider,
            vision_model=result.vision_model,
        )
        if updated is not None:
            result.record = updated
        return result

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        prefer_like: bool = False,
    ) -> tuple[list[tuple[GifRecord, float]], str]:
        if self.library is None:
            return [], "none"
        query = (query or "").strip()
        limit = max(1, min(20, int(top_k or self.config.search_top_k)))
        if query and self.config.embedding_enabled and not prefer_like:
            vector = await self.providers.embed(query)
            if vector:
                results = await self.library.search_by_embedding(
                    vector,
                    top_k=limit,
                    max_rows=self.config.vector_search_max_rows,
                )
                return results, "vector"
        records, _ = await self.library.list_records(
            page=1,
            page_size=limit,
            query=query,
            sort="uses" if not query else "recent",
        )
        return [(record, 0.0) for record in records], "like"

    def cache_stats(self) -> dict[str, Any]:
        stats = self._l1.stats()
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "stores": stats.stores,
            "evictions": stats.evictions,
            "size": stats.size,
        }


__all__ = ["GifEngine", "ProcessedGif"]
