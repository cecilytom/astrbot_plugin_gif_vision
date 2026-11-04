"""Hashes and vectors used by the persistent GIF library."""

from __future__ import annotations

import hashlib
import io
import math
from array import array

from PIL import Image as PILImage

from .frames import select_frame_indices

_LANCZOS = getattr(PILImage, "Resampling", PILImage).LANCZOS


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(image: PILImage.Image) -> int:
    """64-bit difference hash of one image."""

    gray = image.convert("L").resize((9, 8), _LANCZOS)
    try:
        pixels = list(gray.getdata())
    finally:
        gray.close()
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (
                1 if pixels[row * 9 + column] > pixels[row * 9 + column + 1] else 0
            )
    return value


def phash_of_frames(data: bytes, max_frames: int = 3) -> str:
    """Return a 192-bit (3 x 64) perceptual hash for an animated image."""

    try:
        with PILImage.open(io.BytesIO(data)) as image:
            frames = int(getattr(image, "n_frames", 1) or 1)
            indices = select_frame_indices(frames, max(1, max_frames))
            parts: list[str] = []
            for index in indices:
                try:
                    image.seek(index)
                    parts.append(f"{dhash(image):016x}")
                except Exception:
                    continue
            if parts:
                return "".join(parts)
    except Exception:
        pass
    # Far better than nothing for non-decodable inputs.
    return sha256_bytes(data)[:48]


def hamming_distance_hex(left: str, right: str) -> int:
    if not left or not right:
        return 192
    if len(left) != len(right):
        return 192
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 192


def normalize_embedding(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in values))
    if not norm or not math.isfinite(norm):
        return [0.0 for _ in values]
    return [float(item) / norm for item in values]


def embedding_to_blob(values: list[float]) -> bytes:
    return array("f", [float(item) for item in values]).tobytes()


def blob_to_embedding(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    values = array("f")
    try:
        values.frombytes(blob)
    except Exception:
        return []
    return list(values)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


__all__ = [
    "blob_to_embedding",
    "cosine_similarity",
    "dhash",
    "embedding_to_blob",
    "hamming_distance_hex",
    "normalize_embedding",
    "phash_of_frames",
    "sha256_bytes",
]
