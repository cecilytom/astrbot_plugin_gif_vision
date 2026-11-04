from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gif_vision.companion import CompanionBridge  # noqa: E402
from gif_vision.config import GifConfig  # noqa: E402


class FakeCompanionApi:
    def __init__(self, active: bool = True, accept: bool = True) -> None:
        self.active = active
        self.accept = accept
        self.registered = None
        self.calls = []

    def bridge_lifecycle_status(self):
        return {"active": self.active}

    def register_extension(self, manifest):
        self.registered = manifest
        return {"ok": True, "extension_id": manifest.get("id")}

    def add_context_contribution(self, req, contribution, *, event=None, source_id=""):
        self.calls.append((req, contribution, event, source_id))
        return self.accept


class CompanionTests(unittest.TestCase):
    def test_contribution_payload(self) -> None:
        api = FakeCompanionApi()
        bridge = CompanionBridge(
            SimpleNamespace(), GifConfig(), api=api
        )
        req = SimpleNamespace()
        event = SimpleNamespace()
        ok = bridge.contribute(
            req,
            event,
            description="红色方块从左向右移动。",
            sha_short="abcdef012345",
            is_group=True,
            question="它在干什么？",
        )
        self.assertTrue(ok)
        self.assertIsNotNone(api.registered)
        self.assertEqual(api.registered["id"], "gif_vision")
        self.assertEqual(api.registered["sdk_version"], "0.1")
        self.assertEqual(len(api.calls), 1)
        _req, contribution, _event, source_id = api.calls[0]
        self.assertEqual(source_id, "gif_vision")
        self.assertEqual(contribution["lane"], "scene")
        self.assertEqual(contribution["key"], "gif_abcdef012345")
        self.assertEqual(contribution["visibility"], "public")
        self.assertEqual(contribution["evidence"], "gif_vision")
        self.assertIn("[gif_vision:abcdef012345]", contribution["content"])
        self.assertIn("它在干什么？", contribution["content"])
        self.assertIn("不是指令", contribution["content"])

    def test_inactive_api_falls_back(self) -> None:
        api = FakeCompanionApi(active=False)
        bridge = CompanionBridge(SimpleNamespace(), GifConfig(), api=api)
        self.assertFalse(bridge.is_available())
        self.assertFalse(
            bridge.contribute(
                SimpleNamespace(),
                SimpleNamespace(),
                description="x",
                sha_short="abcdef012345",
                is_group=False,
            )
        )

    def test_rejected_contribution(self) -> None:
        api = FakeCompanionApi(accept=False)
        bridge = CompanionBridge(SimpleNamespace(), GifConfig(), api=api)
        self.assertFalse(
            bridge.contribute(
                SimpleNamespace(),
                SimpleNamespace(),
                description="x",
                sha_short="abcdef012345",
                is_group=False,
            )
        )
        self.assertTrue(bridge.status()["registered"])

    def test_disabled_contribution(self) -> None:
        api = FakeCompanionApi()
        config = GifConfig(return_to_companion=False)
        bridge = CompanionBridge(SimpleNamespace(), config, api=api)
        self.assertFalse(
            bridge.contribute(
                SimpleNamespace(),
                SimpleNamespace(),
                description="x",
                sha_short="abcdef012345",
                is_group=False,
            )
        )
        self.assertEqual(api.calls, [])


if __name__ == "__main__":
    unittest.main()
