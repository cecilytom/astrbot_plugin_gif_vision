from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gif_vision.fingerprint import (  # noqa: E402
    blob_to_embedding,
    cosine_similarity,
    embedding_to_blob,
    hamming_distance_hex,
    normalize_embedding,
    phash_of_frames,
    sha256_bytes,
)
from tests.support import make_gif  # noqa: E402


class FingerprintTests(unittest.TestCase):
    def test_sha_and_phash_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = make_gif(Path(temp) / "a.gif", frames=8)
            data = path.read_bytes()
            self.assertEqual(sha256_bytes(data), sha256_bytes(data))
            left = phash_of_frames(data)
            right = phash_of_frames(data)
            self.assertEqual(left, right)
            self.assertEqual(len(left), 48)
            self.assertEqual(hamming_distance_hex(left, right), 0)

    def test_different_gifs_have_large_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            a = make_gif(Path(temp) / "a.gif", frames=8).read_bytes()
            b = make_gif(
                Path(temp) / "b.gif", frames=8, size=(160, 120)
            ).read_bytes()
            distance = hamming_distance_hex(
                phash_of_frames(a), phash_of_frames(b)
            )
            self.assertGreater(distance, 12)

    def test_embedding_helpers(self) -> None:
        vector = normalize_embedding([3.0, 4.0])
        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)
        blob = embedding_to_blob(vector)
        restored = blob_to_embedding(blob)
        self.assertEqual(len(restored), 2)
        self.assertAlmostEqual(restored[0], 0.6, places=5)
        self.assertAlmostEqual(cosine_similarity(vector, restored), 1.0, places=5)
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), -1.0)
        self.assertEqual(blob_to_embedding(b""), [])

    def test_phash_fallback_for_bad_data(self) -> None:
        value = phash_of_frames(b"not a gif")
        self.assertEqual(len(value), 48)


if __name__ == "__main__":
    unittest.main()
