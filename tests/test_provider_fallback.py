from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gif_vision.config import GifConfig  # noqa: E402
from gif_vision.frames import Frame  # noqa: E402
from gif_vision.provider import ProviderResolver, VisionError  # noqa: E402


class FakeResponse:
    def __init__(self, text: str = "画面里有一个红色方块在移动。") -> None:
        self.completion_text = text
        self.reasoning_content = ""


class FakeProvider:
    def __init__(self, provider_id: str, modalities: list[str], text: str = "") -> None:
        self.provider_config = {
            "id": provider_id,
            "modalities": modalities,
            "model": f"{provider_id}-model",
        }
        self.text = text
        self.calls = 0

    def get_model(self) -> str:
        return str(self.provider_config.get("model") or "")

    async def text_chat(self, **kwargs):
        self.calls += 1
        return FakeResponse(self.text or "画面里有一个红色方块在移动。")


class FakeEmbeddingProvider:
    def __init__(self, model: str = "fake-embed") -> None:
        self.provider_config = {"id": model, "model": model}
        self.calls = 0

    def get_dim(self) -> int:
        return 3

    async def get_embedding(self, text: str):
        self.calls += 1
        return [3.0, 4.0, 0.0]


class FakeContext:
    def __init__(self, providers, current=None, embeddings=None, caption_id=""):
        self.providers = {item.provider_config["id"]: item for item in providers}
        self.current = current
        self.embeddings = embeddings or []
        self.caption_id = caption_id

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)

    async def get_using_provider_async(self, umo=None):
        return self.current

    def get_config(self, umo=None):
        return {
            "provider_settings": {
                "default_image_caption_provider_id": self.caption_id
            }
        }

    def get_all_embedding_providers(self):
        return self.embeddings


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_falls_back_to_vision_provider(self) -> None:
        text_only = FakeProvider("text", ["text"])
        vision = FakeProvider("vision", ["text", "image"], "红色方块")
        context = FakeContext([text_only, vision], current=text_only)
        config = GifConfig(provider_id="", fallback_provider_ids=["vision"])
        resolver = ProviderResolver(context, config)
        provider = await resolver.vision_provider(None)
        self.assertIs(provider, vision)

    async def test_uses_current_if_vision(self) -> None:
        vision = FakeProvider("vision", ["image"], "红色方块")
        context = FakeContext([vision], current=vision)
        resolver = ProviderResolver(context, GifConfig())
        self.assertIs(await resolver.vision_provider(None), vision)

    async def test_describe_and_embed(self) -> None:
        vision = FakeProvider("vision", ["image"], "一个人在吃薯片。")
        embedding = FakeEmbeddingProvider()
        context = FakeContext([vision], current=vision, embeddings=[embedding])
        resolver = ProviderResolver(context, GifConfig())
        frames = [
            Frame(index=0, png_bytes=b"x", data_url="data:image/png;base64,eA==")
        ]
        text = await resolver.describe(
            vision, "描述", frames, timeout_sec=3
        )
        self.assertIn("薯片", text)
        self.assertEqual(vision.calls, 1)
        vector = await resolver.embed("测试")
        self.assertIsNotNone(vector)
        self.assertAlmostEqual(vector[0], 0.6, places=5)
        info = resolver.embedding_info()
        self.assertEqual(info["dimension"], 3)
        self.assertTrue(info["available"])

    async def test_no_provider_error(self) -> None:
        resolver = ProviderResolver(FakeContext([]), GifConfig())
        self.assertIsNone(await resolver.vision_provider(None))
        with self.assertRaises(VisionError):
            await resolver.describe(None, "x", [], timeout_sec=1)
        self.assertIsNone(await resolver.embed("x"))


if __name__ == "__main__":
    unittest.main()
