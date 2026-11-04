from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gif_vision import webui  # noqa: E402
from gif_vision.config import GifConfig  # noqa: E402
from gif_vision.fingerprint import phash_of_frames, sha256_bytes  # noqa: E402
from gif_vision.library import GifLibrary  # noqa: E402
from gif_vision.webui import GifWebUI, WebUIError  # noqa: E402
from tests.support import make_gif  # noqa: E402


class FakeRequest:
    def __init__(self, args=None) -> None:
        self.args = args or {}

    async def get_json(self, silent=True):
        return {}


class FakeEngine:
    def __init__(self, results=None) -> None:
        self.results = results or []
        self.last_error = ""

    async def search(self, query, top_k=None):
        return self.results, "vector" if self.results else "like"


class WebUITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.library = GifLibrary(Path(self.temp.name) / "library")
        self.gif = make_gif(Path(self.temp.name) / "a.gif", frames=6)
        self.data = self.gif.read_bytes()
        await self.library.store(
            data=self.data,
            sha256=sha256_bytes(self.data),
            phash=phash_of_frames(self.data),
            description="测试红色方块在移动",
            embedding=None,
            source="test",
        )
        self.config = GifConfig()
        self.plugin = SimpleNamespace(
            config=self.config,
            library=self.library,
            engine=FakeEngine(),
            providers=None,
            companion=None,
            logger=None,
        )
        self.ui = GifWebUI(self.plugin)
        self.original_request = webui.request
        webui.request = FakeRequest()

    async def asyncTearDown(self) -> None:
        webui.request = self.original_request
        self.library.close()
        self.temp.cleanup()

    async def test_list_records(self) -> None:
        webui.request = FakeRequest({"page": "1", "page_size": "10"})
        result = await self.ui._api_list()
        self.assertEqual(result["mode"], "list")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["id"], 1)
        self.assertNotIn("path", result["items"][0])

    async def test_like_search(self) -> None:
        webui.request = FakeRequest(
            {"page": "1", "page_size": "10", "query": "红色", "mode": "like"}
        )
        result = await self.ui._api_list()
        self.assertEqual(result["mode"], "like")
        self.assertEqual(result["total"], 1)

    async def test_vector_search(self) -> None:
        record = await self.library.get_by_id(1)
        self.plugin.engine = FakeEngine([(record, 0.91)])
        webui.request = FakeRequest(
            {"page": "1", "page_size": "10", "query": "红色", "mode": "auto"}
        )
        result = await self.ui._api_list()
        self.assertEqual(result["mode"], "vector")
        self.assertAlmostEqual(result["items"][0]["score"], 0.91)

    def test_record_id_validation(self) -> None:
        self.assertEqual(GifWebUI._record_id("12"), 12)
        with self.assertRaises(WebUIError):
            GifWebUI._record_id("0")
        with self.assertRaises(WebUIError):
            GifWebUI._record_id("abc")


if __name__ == "__main__":
    unittest.main()
