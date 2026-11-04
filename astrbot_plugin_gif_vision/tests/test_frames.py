from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gif_vision.frames import (  # noqa: E402
    extract_frames,
    make_thumbnail,
    select_frame_indices,
)
from tests.support import make_gif, make_static_png  # noqa: E402


class FrameTests(unittest.TestCase):
    def test_select_frame_indices(self) -> None:
        self.assertEqual(select_frame_indices(1, 6), [0])
        self.assertEqual(select_frame_indices(3, 6), [0, 1, 2])
        indices = select_frame_indices(73, 6)
        self.assertEqual(len(indices), 6)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 72)
        self.assertEqual(indices, sorted(set(indices)))

    def test_extract_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = make_gif(Path(temp) / "sample.gif", frames=10)
            data = path.read_bytes()
            frames = extract_frames(data, count=6, max_edge=128)
            self.assertEqual(len(frames), 6)
            self.assertEqual(frames[0].index, 0)
            self.assertEqual(frames[-1].index, 9)
            self.assertTrue(frames[0].png_bytes.startswith(b"\x89PNG"))
            self.assertTrue(frames[0].data_url.startswith("data:image/png;base64,"))
            for frame in frames:
                self.assertLessEqual(len(frame.png_bytes), 200_000)

    def test_static_image_has_no_frames(self) -> None:
        self.assertEqual(extract_frames(make_static_png(), 6, 128), [])

    def test_make_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = make_gif(Path(temp) / "thumb.gif", frames=6)
            thumb = make_thumbnail(path.read_bytes(), 200)
            self.assertIsNotNone(thumb)
            self.assertTrue(thumb.startswith(b"\xff\xd8"))

    def test_bad_data(self) -> None:
        self.assertEqual(extract_frames(b"not an image", 6, 128), [])
        self.assertIsNone(make_thumbnail(b"not an image", 200))


if __name__ == "__main__":
    unittest.main()
