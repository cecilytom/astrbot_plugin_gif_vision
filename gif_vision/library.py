"""Persistent GIF library: original files, SQLite metadata, embeddings."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fingerprint import (
    blob_to_embedding,
    cosine_similarity,
    hamming_distance_hex,
)
from .frames import image_info, make_thumbnail


@dataclass
class GifRecord:
    id: int
    sha256: str
    phash: str
    path: str
    thumb_path: str | None
    size: int
    width: int | None
    height: int | None
    frames: int | None
    duration_ms: int | None
    description: str
    embedding: bytes | None
    embedding_dim: int | None
    embedding_model: str | None
    vision_provider: str
    vision_model: str
    canonical_id: int | None
    source: str
    summary: str
    created_at: float
    last_used_at: float
    use_count: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "GifRecord":
        return cls(
            id=int(row["id"]),
            sha256=str(row["sha256"]),
            phash=str(row["phash"] or ""),
            path=str(row["path"]),
            thumb_path=str(row["thumb_path"]) if row["thumb_path"] else None,
            size=int(row["size"] or 0),
            width=int(row["width"]) if row["width"] is not None else None,
            height=int(row["height"]) if row["height"] is not None else None,
            frames=int(row["frames"]) if row["frames"] is not None else None,
            duration_ms=(
                int(row["duration_ms"])
                if row["duration_ms"] is not None
                else None
            ),
            description=str(row["description"] or ""),
            embedding=bytes(row["embedding"]) if row["embedding"] else None,
            embedding_dim=(
                int(row["embedding_dim"])
                if row["embedding_dim"] is not None
                else None
            ),
            embedding_model=(
                str(row["embedding_model"]) if row["embedding_model"] else None
            ),
            vision_provider=str(row["vision_provider"] or ""),
            vision_model=str(row["vision_model"] or ""),
            canonical_id=(
                int(row["canonical_id"])
                if row["canonical_id"] is not None
                else None
            ),
            source=str(row["source"] or ""),
            summary=str(row["summary"] or ""),
            created_at=float(row["created_at"] or 0.0),
            last_used_at=float(row["last_used_at"] or 0.0),
            use_count=int(row["use_count"] or 0),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sha256_short": self.sha256[:12],
            "description": self.description,
            "size": self.size,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
            "source": self.source,
            "summary": self.summary,
            "has_embedding": bool(self.embedding),
            "has_thumb": bool(self.thumb_path),
            "canonical_id": self.canonical_id,
        }


class GifLibrary:
    def __init__(self, root: Path, logger: Any = None) -> None:
        self.root = Path(root)
        self.logger = logger
        self.gifs_dir = self.root / "gifs"
        self.thumbs_dir = self.root / "thumbs"
        self.tmp_dir = self.root / "tmp"
        for directory in (self.root, self.gifs_dir, self.thumbs_dir, self.tmp_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "library.sqlite3"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ---------------------------------------------------------------- setup
    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS gifs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  sha256 TEXT UNIQUE NOT NULL,
                  phash TEXT NOT NULL DEFAULT '',
                  path TEXT NOT NULL,
                  thumb_path TEXT,
                  size INTEGER NOT NULL DEFAULT 0,
                  width INTEGER,
                  height INTEGER,
                  frames INTEGER,
                  duration_ms INTEGER,
                  description TEXT NOT NULL,
                  embedding BLOB,
                  embedding_dim INTEGER,
                  embedding_model TEXT,
                  vision_provider TEXT,
                  vision_model TEXT,
                  canonical_id INTEGER,
                  source TEXT,
                  summary TEXT,
                  created_at REAL NOT NULL,
                  last_used_at REAL NOT NULL,
                  use_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_gifs_phash ON gifs(phash);
                CREATE INDEX IF NOT EXISTS idx_gifs_last_used ON gifs(last_used_at);
                CREATE INDEX IF NOT EXISTS idx_gifs_created ON gifs(created_at);
                """
            )
            self._conn.execute("PRAGMA user_version=1")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass

    # -------------------------------------------------------------- helpers
    def _gif_path(self, sha: str) -> Path:
        return self.gifs_dir / sha[:2] / sha[2:4] / f"{sha}.gif"

    def _thumb_path(self, sha: str) -> Path:
        return self.thumbs_dir / sha[:2] / f"{sha}.jpg"

    def _safe_remove(self, path: str | Path | None) -> None:
        if not path:
            return
        try:
            resolved = Path(path).resolve()
            root = self.root.resolve()
            if not resolved.is_relative_to(root):
                return
            resolved.unlink(missing_ok=True)
        except Exception:
            return

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
        try:
            temp.write_bytes(data)
            os.replace(temp, path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except OSError:
            pass
        return total

    def _row_by_sha(self, sha: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM gifs WHERE sha256 = ?", (sha,)
        ).fetchone()

    def _row_by_id(self, record_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM gifs WHERE id = ?", (int(record_id),)
        ).fetchone()

    def _touch_sync(self, record_id: int) -> None:
        now = time.time()
        self._conn.execute(
            "UPDATE gifs SET last_used_at = ?, use_count = use_count + 1 WHERE id = ?",
            (now, int(record_id)),
        )
        self._conn.commit()

    # --------------------------------------------------------------- reads
    async def get_by_sha(self, sha: str) -> GifRecord | None:
        def work() -> GifRecord | None:
            with self._lock:
                row = self._row_by_sha(sha)
                return GifRecord.from_row(row) if row else None

        return await asyncio.to_thread(work)

    async def get_by_id(self, record_id: int) -> GifRecord | None:
        def work() -> GifRecord | None:
            with self._lock:
                row = self._row_by_id(record_id)
                return GifRecord.from_row(row) if row else None

        return await asyncio.to_thread(work)

    def _touch_with_lock(self, record_id: int) -> None:
        with self._lock:
            self._touch_sync(record_id)

    async def touch(self, record_id: int) -> None:
        await asyncio.to_thread(self._touch_with_lock, record_id)

    async def find_near(self, phash: str, max_distance: int) -> GifRecord | None:
        def work() -> GifRecord | None:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM gifs WHERE phash != '' "
                    "ORDER BY last_used_at DESC LIMIT 4000"
                ).fetchall()
                best_row: sqlite3.Row | None = None
                best_distance = max_distance + 1
                for row in rows:
                    distance = hamming_distance_hex(phash, str(row["phash"] or ""))
                    if distance < best_distance:
                        best_distance = distance
                        best_row = row
                if best_row is None or best_distance > max_distance:
                    return None
                return GifRecord.from_row(best_row)

        return await asyncio.to_thread(work)

    async def list_records(
        self,
        *,
        page: int = 1,
        page_size: int = 24,
        query: str = "",
        sort: str = "recent",
        missing_embedding: bool = False,
    ) -> tuple[list[GifRecord], int]:
        page = max(1, int(page))
        page_size = max(1, min(50, int(page_size)))
        order = {
            "recent": "last_used_at DESC, id DESC",
            "oldest": "created_at ASC, id ASC",
            "newest": "created_at DESC, id DESC",
            "uses": "use_count DESC, last_used_at DESC",
            "size": "size DESC, last_used_at DESC",
        }.get(sort, "last_used_at DESC, id DESC")

        def work() -> tuple[list[GifRecord], int]:
            with self._lock:
                clauses: list[str] = []
                params: list[Any] = []
                if query:
                    clauses.append("description LIKE ?")
                    params.append(f"%{query}%")
                if missing_embedding:
                    clauses.append("(embedding IS NULL OR embedding_dim IS NULL)")
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                total_row = self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM gifs {where}", params
                ).fetchone()
                total = int(total_row["c"] if total_row else 0)
                rows = self._conn.execute(
                    f"SELECT * FROM gifs {where} ORDER BY {order} LIMIT ? OFFSET ?",
                    [*params, page_size, (page - 1) * page_size],
                ).fetchall()
                return [GifRecord.from_row(row) for row in rows], total

        return await asyncio.to_thread(work)

    async def search_by_embedding(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        max_rows: int = 5000,
    ) -> list[tuple[GifRecord, float]]:
        if not query_vector:
            return []

        def work() -> list[tuple[GifRecord, float]]:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM gifs WHERE embedding IS NOT NULL "
                    "ORDER BY last_used_at DESC LIMIT ?",
                    (max(1, min(50000, int(max_rows))),),
                ).fetchall()
                scored: list[tuple[GifRecord, float]] = []
                for row in rows:
                    record = GifRecord.from_row(row)
                    vector = blob_to_embedding(record.embedding)
                    if not vector:
                        continue
                    score = cosine_similarity(query_vector, vector)
                    if score < 0:
                        continue
                    scored.append((record, score))
                scored.sort(key=lambda item: item[1], reverse=True)
                return scored[: max(1, min(50, int(top_k)))]

        return await asyncio.to_thread(work)

    async def records_missing_embeddings(
        self, dimension: int, limit: int = 20
    ) -> list[GifRecord]:
        def work() -> list[GifRecord]:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM gifs WHERE embedding IS NULL "
                    "OR embedding_dim IS NULL OR embedding_dim != ? "
                    "ORDER BY created_at ASC LIMIT ?",
                    (int(dimension), max(1, min(200, int(limit)))),
                ).fetchall()
                return [GifRecord.from_row(row) for row in rows]

        return await asyncio.to_thread(work)

    async def export_records(self) -> list[dict[str, Any]]:
        def work() -> list[dict[str, Any]]:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM gifs ORDER BY id ASC"
                ).fetchall()
                result: list[dict[str, Any]] = []
                for row in rows:
                    record = GifRecord.from_row(row)
                    item = record.public_dict()
                    item["description"] = record.description
                    item["sha256"] = record.sha256
                    item["phash"] = record.phash
                    result.append(item)
                return result

        return await asyncio.to_thread(work)

    async def stats(self) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS total, "
                    "COALESCE(SUM(size), 0) AS bytes, "
                    "COALESCE(SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END), 0) AS with_embedding, "
                    "COALESCE(MIN(created_at), 0) AS oldest, "
                    "COALESCE(MAX(created_at), 0) AS newest "
                    "FROM gifs"
                ).fetchone()
                total = int(row["total"] if row else 0)
                with_embedding = int(row["with_embedding"] if row else 0)
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                return {
                    "total": total,
                    "with_embedding": with_embedding,
                    "without_embedding": max(0, total - with_embedding),
                    "total_bytes": int(row["bytes"] if row else 0),
                    "db_bytes": db_size,
                    "disk_bytes": self._dir_size(self.gifs_dir)
                    + self._dir_size(self.thumbs_dir),
                    "oldest": float(row["oldest"] if row else 0.0),
                    "newest": float(row["newest"] if row else 0.0),
                    "db_path": str(self.db_path),
                    "root": str(self.root),
                }

        return await asyncio.to_thread(work)

    # -------------------------------------------------------------- writes
    async def store(
        self,
        *,
        data: bytes,
        sha256: str,
        phash: str,
        description: str,
        embedding: bytes | None = None,
        embedding_dim: int | None = None,
        embedding_model: str | None = None,
        vision_provider: str = "",
        vision_model: str = "",
        canonical_id: int | None = None,
        source: str = "",
        summary: str = "",
        thumb_data: bytes | None = None,
    ) -> GifRecord:
        info = image_info(data)
        return await asyncio.to_thread(
            self._store_sync,
            data,
            sha256,
            phash,
            description,
            embedding,
            embedding_dim,
            embedding_model,
            vision_provider,
            vision_model,
            canonical_id,
            source,
            summary,
            thumb_data,
            info,
        )

    def _store_sync(
        self,
        data: bytes,
        sha256: str,
        phash: str,
        description: str,
        embedding: bytes | None,
        embedding_dim: int | None,
        embedding_model: str | None,
        vision_provider: str,
        vision_model: str,
        canonical_id: int | None,
        source: str,
        summary: str,
        thumb_data: bytes | None,
        info: dict[str, Any],
    ) -> GifRecord:
        with self._lock:
            existing = self._row_by_sha(sha256)
            if existing:
                self._touch_sync(int(existing["id"]))
                row = self._row_by_sha(sha256)
                return GifRecord.from_row(row if row else existing)

            path = self._gif_path(sha256)
            if not path.exists():
                self._atomic_write(path, data)
            thumb_path: Path | None = None
            if thumb_data:
                thumb_path = self._thumb_path(sha256)
                self._atomic_write(thumb_path, thumb_data)

            now = time.time()
            self._conn.execute(
                """
                INSERT INTO gifs (
                  sha256, phash, path, thumb_path, size, width, height,
                  frames, duration_ms, description, embedding, embedding_dim,
                  embedding_model, vision_provider, vision_model, canonical_id,
                  source, summary, created_at, last_used_at, use_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    sha256,
                    phash,
                    str(path),
                    str(thumb_path) if thumb_path else None,
                    len(data),
                    info.get("width"),
                    info.get("height"),
                    info.get("frames"),
                    info.get("duration_ms"),
                    description,
                    embedding,
                    embedding_dim,
                    embedding_model,
                    vision_provider,
                    vision_model,
                    canonical_id,
                    source,
                    summary,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._row_by_sha(sha256)
            if row is None:
                raise RuntimeError("failed to persist GIF record")
            return GifRecord.from_row(row)

    async def update_description(
        self,
        record_id: int,
        *,
        description: str,
        embedding: bytes | None,
        embedding_dim: int | None,
        embedding_model: str | None,
        vision_provider: str,
        vision_model: str,
    ) -> GifRecord | None:
        def work() -> GifRecord | None:
            with self._lock:
                self._conn.execute(
                    "UPDATE gifs SET description = ?, embedding = ?, "
                    "embedding_dim = ?, embedding_model = ?, vision_provider = ?, "
                    "vision_model = ?, last_used_at = ? WHERE id = ?",
                    (
                        description,
                        embedding,
                        embedding_dim,
                        embedding_model,
                        vision_provider,
                        vision_model,
                        time.time(),
                        int(record_id),
                    ),
                )
                self._conn.commit()
                row = self._row_by_id(record_id)
                return GifRecord.from_row(row) if row else None

        return await asyncio.to_thread(work)

    async def update_embedding(
        self,
        record_id: int,
        *,
        embedding: bytes,
        dimension: int,
        model: str,
    ) -> None:
        def work() -> None:
            with self._lock:
                self._conn.execute(
                    "UPDATE gifs SET embedding = ?, embedding_dim = ?, "
                    "embedding_model = ? WHERE id = ?",
                    (embedding, int(dimension), model, int(record_id)),
                )
                self._conn.commit()

        await asyncio.to_thread(work)

    async def delete(self, record_id: int) -> bool:
        def work() -> bool:
            with self._lock:
                row = self._row_by_id(record_id)
                if row is None:
                    return False
                record = GifRecord.from_row(row)
                self._conn.execute("DELETE FROM gifs WHERE id = ?", (int(record_id),))
                self._conn.commit()
                self._safe_remove(record.path)
                self._safe_remove(record.thumb_path)
                return True

        return await asyncio.to_thread(work)

    async def ensure_thumb(self, record_id: int, edge: int) -> Path | None:
        record = await self.get_by_id(record_id)
        if record is None:
            return None
        if record.thumb_path and Path(record.thumb_path).exists():
            return Path(record.thumb_path)
        data = await asyncio.to_thread(self._read_gif, record)
        if not data:
            return None
        thumb = await asyncio.to_thread(make_thumbnail, data, edge)
        if not thumb:
            return None
        path = self._thumb_path(record.sha256)
        await asyncio.to_thread(self._atomic_write, path, thumb)
        await asyncio.to_thread(self._set_thumb, record_id, str(path))
        return path

    def _read_gif(self, record: GifRecord) -> bytes | None:
        try:
            path = Path(record.path).resolve()
            if not path.is_relative_to(self.root.resolve()):
                return None
            return path.read_bytes()
        except (OSError, ValueError):
            return None

    def _set_thumb(self, record_id: int, thumb_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE gifs SET thumb_path = ? WHERE id = ?",
                (thumb_path, int(record_id)),
            )
            self._conn.commit()

    async def path_for(self, record_id: int) -> Path | None:
        record = await self.get_by_id(record_id)
        if record is None:
            return None
        path = Path(record.path)
        try:
            if path.exists() and path.resolve().is_relative_to(self.root.resolve()):
                return path
        except OSError:
            return None
        return None

    async def prune(self, *, retention_days: int, max_mb: float) -> int:
        cutoff = 0.0
        if retention_days > 0:
            cutoff = time.time() - retention_days * 86400
        max_bytes = max(0.001, float(max_mb)) * 1024 * 1024

        def work() -> int:
            removed = 0
            with self._lock:
                rows = self._conn.execute("SELECT * FROM gifs").fetchall()
                records = [GifRecord.from_row(row) for row in rows]
                if cutoff:
                    for record in records:
                        if record.created_at < cutoff:
                            self._conn.execute(
                                "DELETE FROM gifs WHERE id = ?", (record.id,)
                            )
                            self._safe_remove(record.path)
                            self._safe_remove(record.thumb_path)
                            removed += 1
                    self._conn.commit()
                    records = [
                        record for record in records if record.created_at >= cutoff
                    ]
                total = sum(record.size for record in records)
                if total <= max_bytes:
                    return removed
                for record in sorted(records, key=lambda item: item.last_used_at):
                    if total <= max_bytes:
                        break
                    self._conn.execute(
                        "DELETE FROM gifs WHERE id = ?", (record.id,)
                    )
                    self._safe_remove(record.path)
                    self._safe_remove(record.thumb_path)
                    total -= record.size
                    removed += 1
                self._conn.commit()
                return removed

        return await asyncio.to_thread(work)


__all__ = ["GifLibrary", "GifRecord"]
