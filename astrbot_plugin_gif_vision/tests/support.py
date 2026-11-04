"""Shared test helpers (no AstrBot imports)."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw


def make_gif(
    path: Path,
    *,
    frames: int = 8,
    size: tuple[int, int] = (120, 90),
    duration: int = 80,
) -> Path:
    images = []
    for index in range(frames):
        image = Image.new("RGB", size, (250, 250, 250))
        draw = ImageDraw.Draw(image)
        x = 8 + index * max(1, (size[0] - 40) // max(1, frames - 1))
        draw.rectangle((x, 20, x + 24, 44), fill=(220, 44, 44))
        draw.text((8, 58), f"frame {index}", fill=(20, 20, 20))
        images.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()
    return path


def make_static_png(size: tuple[int, int] = (40, 30)) -> bytes:
    image = Image.new("RGB", size, (20, 120, 220))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    image.close()
    return buffer.getvalue()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()
