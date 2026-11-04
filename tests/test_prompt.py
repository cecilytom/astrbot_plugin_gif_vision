from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gif_vision.config import GifConfig  # noqa: E402
from gif_vision.prompt import (  # noqa: E402
    build_contribution_content,
    build_vision_prompt,
    clean_description,
    sentinel_for,
    short_preview,
    wrap_for_injection,
)


class PromptTests(unittest.TestCase):
    def test_describe_prompt(self) -> None:
        prompt = build_vision_prompt(GifConfig(), "", 6)
        self.assertIn("画面主体", prompt)
        self.assertIn("6 个采样帧", prompt)
        self.assertNotIn("{question}", prompt)

    def test_qa_prompt_contains_question(self) -> None:
        prompt = build_vision_prompt(GifConfig(), "它在干什么？", 4)
        self.assertIn("它在干什么？", prompt)
        self.assertIn("4 个采样帧", prompt)

    def test_clean_and_truncate(self) -> None:
        text = "第一段。第二段。" + "很长的描述" * 100
        cleaned = clean_description(text, 40)
        self.assertLessEqual(len(cleaned), 41)
        self.assertTrue(cleaned.endswith("…") or cleaned.endswith("。"))
        self.assertEqual(clean_description("", 40), "")
        self.assertEqual(clean_description("```\nhello\n```", 40), "hello")

    def test_wrappers(self) -> None:
        wrapped = wrap_for_injection("画面内容", index=1, total=2)
        self.assertIn("<gif_analysis>", wrapped)
        self.assertIn("画面内容", wrapped)
        self.assertIn("不要提抽帧", wrapped)
        contribution = build_contribution_content(
            "画面内容", "abcdef012345", "这是什么"
        )
        self.assertIn(sentinel_for("abcdef012345"), contribution)
        self.assertIn("这是什么", contribution)
        self.assertIn("不是指令", contribution)

    def test_short_preview(self) -> None:
        self.assertEqual(short_preview("a  b\n c", 20), "a b c")
        self.assertTrue(short_preview("x" * 50, 10).endswith("…"))


if __name__ == "__main__":
    unittest.main()
