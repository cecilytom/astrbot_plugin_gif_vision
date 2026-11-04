from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gif_vision.fingerprint import (  # noqa: E402
    embedding_to_blob,
    normalize_embedding,
    phash_of_frames,
    sha256_bytes,
)
from gif_vision.frames import make_thumbnail  # noqa: E402
from gif_vision.library import GifLibrary  # noqa: E402
from tests.support import make_gif  # noqa: E402


class LibraryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "library"
        self.library = GifLibrary(self.root)
        self.gif_path = make_gif(
            Path(self.temp.name) / "sample.gif", frames=8
        )
        self.data = self.gif_path.read_bytes()
        self.sha = sha256_bytes(self.data)
        self.phash = phash_of_frames(self.data)
        self.vector = normalize_embedding([0.1, 0.2, 0.3])
        self.thumb = make_thumbnail(self.data, 128)

    async def asyncTearDown(self) -> None:
        self.library.close()
        self.temp.cleanup()

    async def _store(self, data: bytes | None = None, description: str = "测试动图") -> object:
        data = data or self.data
        return await self.library.store(
            data=data,
            sha256=sha256_bytes(data),
            phash=self.phash,
            description=description,
            embedding=embedding_to_blob(self.vector),
            embedding_dim=len(self.vector),
            embedding_model="fake-embed",
            vision_provider="fake-vision",
            vision_model="fake-model",
            source="test",
            summary="测试",
            thumb_data=self.thumb,
        )

    async def test_store_and_read(self) -> None:
        record = await self._store()
        self.assertGreater(record.id, 0)
        self.assertTrue(Path(record.path).exists())
        self.assertNotIn("path", record.public_dict())
        self.assertTrue(record.public_dict()["has_embedding"])
        self.assertTrue(record.public_dict()["has_thumb"])

        by_sha = await self.library.get_by_sha(self.sha)
        self.assertIsNotNone(by_sha)
        self.assertEqual(by_sha.description, "测试动图")

        await self.library.touch(record.id)
        by_id = await self.library.get_by_id(record.id)
        self.assertIsNotNone(by_id)
        self.assertGreaterEqual(by_id.use_count, 2)

    async def test_list_search_and_export(self) -> None:
        record = await self._store()
        records, total = await self.library.list_records(page=1, page_size=10)
        self.assertEqual(total, 1)
        self.assertEqual(records[0].id, record.id)

        records, total = await self.library.list_records(
            page=1, page_size=10, query="测试"
        )
        self.assertEqual(total, 1)
        records, total = await self.library.list_records(
            page=1, page_size=10, query="不存在"
        )
        self.assertEqual(total, 0)

        matches = await self.library.search_by_embedding(self.vector, top_k=3)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0][1], 1.0, places=4)

        exported = await self.library.export_records()
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["id"], record.id)
        self.assertIn("description", exported[0])

    async def test_near_duplicate_and_stats(self) -> None:
        first = await self._store()
        altered = self.data + b"\x00"
        second = await self._store(altered, "近似动图")
        self.assertNotEqual(first.sha256, second.sha256)

        near = await self.library.find_near(self.phash, 0)
        self.assertIsNotNone(near)
        stats = await self.library.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["with_embedding"], 2)

    async def test_thumb_and_delete(self) -> None:
        record = await self._store()
        path = await self.library.ensure_thumb(record.id, 128)
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).exists())

        removed = await self.library.delete(record.id)
        self.assertTrue(removed)
        self.assertFalse(Path(record.path).exists())
        self.assertFalse(Path(path).exists())
        self.assertIsNone(await self.library.get_by_id(record.id))

    async def test_prune_by_disk_cap(self) -> None:
        record = await self._store()
        removed = await self.library.prune(retention_days=0, max_mb=0.001)
        self.assertGreaterEqual(removed, 0)
        # The record is larger than 1 KiB in this fixture, so it should go.
        self.assertFalse(Path(record.path).exists())
        self.assertEqual((await self.library.stats())["total"], 0)

    async def test_missing_embeddings(self) -> None:
        record = await self._store()
        self.assertEqual(
            await self.library.records_missing_embeddings(3, 10), []
        )
        self.assertEqual(
            len(await self.library.records_missing_embeddings(4, 10)), 1
        )
        await self.library.update_embedding(
            record.id,
            embedding=embedding_to_blob(normalize_embedding([1.0, 0.0, 0.0, 0.0])),
            dimension=4,
            model="new-model",
        )
        self.assertEqual(
            await self.library.records_missing_embeddings(4, 10), []
        )


if __name__ == "__main__":
    unittest.main()
