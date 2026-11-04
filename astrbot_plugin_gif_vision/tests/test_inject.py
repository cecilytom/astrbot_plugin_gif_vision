from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gif_vision import inject  # noqa: E402
from gif_vision.detect import (  # noqa: E402
    existing_description,
    iter_request_image_refs,
    question_from_event,
)


class FakeRequest:
    def __init__(self) -> None:
        self.prompt = "看看这个"
        self.system_prompt = ""
        self.image_urls = ["base64://AAAA", "base64://BBBB"]
        self.extra_user_content_parts = []


class FakeEvent:
    def __init__(self) -> None:
        self.message_str = "@bot [图片] 这是什么"
        self.unified_msg_origin = "onebot:GroupMessage:1_2"
        self._extra = {}

    def set_extra(self, key, value):
        self._extra[key] = value

    def get_extra(self, key):
        return self._extra.get(key)

    def get_group_id(self):
        return "2"


class InjectTests(unittest.TestCase):
    def test_iter_and_existing(self) -> None:
        req = FakeRequest()
        refs = iter_request_image_refs(req)
        self.assertEqual([item.ref for item in refs], req.image_urls)
        self.assertFalse(existing_description(req))
        req.extra_user_content_parts.append(
            {"type": "text", "text": "<image_caption>已有描述</image_caption>"}
        )
        self.assertTrue(existing_description(req))

    def test_append_and_remove(self) -> None:
        req = FakeRequest()
        req.extra_user_content_parts.append(
            {"type": "image_url", "image_url": {"url": "base64://AAAA"}}
        )
        self.assertTrue(inject.append_text_part(req, "<gif_analysis>x</gif_analysis>"))
        self.assertEqual(len(req.extra_user_content_parts), 2)
        removed = inject.remove_image_refs(req, {"base64://AAAA"})
        self.assertEqual(removed, 2)  # one image_urls + one extra part
        self.assertEqual(req.image_urls, ["base64://BBBB"])
        self.assertEqual(len(req.extra_user_content_parts), 1)
        remaining = req.extra_user_content_parts[0]
        if isinstance(remaining, dict):
            self.assertEqual(remaining.get("type"), "text")
        else:
            self.assertEqual(getattr(remaining, "type", None), "text")

    def test_done_and_results(self) -> None:
        req = FakeRequest()
        event = FakeEvent()
        self.assertFalse(inject.is_done(req))
        inject.mark_done(req)
        self.assertTrue(inject.is_done(req))
        self.assertFalse(inject.is_processed(req))
        inject.mark_processed(req)
        self.assertTrue(inject.is_processed(req))
        inject.set_results(event, [{"sha": "abc"}])
        self.assertEqual(inject.get_results(event), [{"sha": "abc"}])
        self.assertEqual(inject.get_event_extra(event, "missing", 1), 1)

    def test_question(self) -> None:
        event = FakeEvent()
        self.assertEqual(question_from_event(event), "这是什么")
        self.assertTrue(inject.is_group_event(event))

    def test_missing_part_container(self) -> None:
        req = SimpleNamespace(image_urls=[], extra_user_content_parts=None)
        self.assertTrue(inject.append_text_part(req, "hello"))
        self.assertIsInstance(req.extra_user_content_parts, list)


if __name__ == "__main__":
    unittest.main()
