"""GIF frame sampling and thumbnail helpers."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image as PILImage
from PIL import ImageSequence

_LANCZOS = getattr(PILImage, "Resampling", PILImage).LANCZOS


@dataclass(frozen=True)
class Frame:
    index: int
    png_bytes: bytes
    data_url: str

    @property
    def size(self) -> int:
        return len(self.png_bytes)


def select_frame_indices(total: int, count: int) -> list[int]:
    """Pick evenly spaced frame indexes, always including first and last."""

    if total <= 0:
        return []
    count = max(1, min(int(count), int(total)))
    if count == 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def _flatten_frame(frame: PILImage.Image) -> PILImage.Image:
    """Composite a frame over white so transparent GIF cells are readable."""

    rgba = frame.convert("RGBA")
    try:
        canvas = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
        try:
            canvas.alpha_composite(rgba)
            return canvas.convert("RGB")
        finally:
            canvas.close()
    finally:
        rgba.close()


def extract_frames(
    data: bytes,
    count: int = 6,
    max_edge: int = 768,
) -> list[Frame]:
    """Return evenly sampled PNG frames for an animated image.

    Static images return an empty list; the caller should leave them to the
    normal image pipeline.
    """

    try:
        with PILImage.open(io.BytesIO(data)) as image:
            frames = int(getattr(image, "n_frames", 1) or 1)
            if frames <= 1:
                return []
            indices = select_frame_indices(frames, count)
            result: list[Frame] = []
            for index in indices:
                try:
                    image.seek(index)
                    rgb = _flatten_frame(image)
                    try:
                        if max_edge > 0 and max(rgb.size) > max_edge:
                            rgb.thumbnail(
                                (max_edge, max_edge), _LANCZOS
                            )
                        buffer = io.BytesIO()
                        rgb.save(buffer, "PNG", optimize=True)
                        png_bytes = buffer.getvalue()
                    finally:
                        rgb.close()
                    result.append(
                        Frame(
                            index=index,
                            png_bytes=png_bytes,
                            data_url="data:image/png;base64,"
                            + base64.b64encode(png_bytes).decode("ascii"),
                        )
                    )
                except Exception:
                    continue
            return result
    except Exception:
        return []


def make_thumbnail(data: bytes, edge: int = 320) -> bytes | None:
    """Build a 2x2 JPEG contact sheet; return None when no image can be read."""

    edge = max(64, int(edge))
    cell = max(32, edge // 2)
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            frames = int(getattr(image, "n_frames", 1) or 1)
            if frames > 1:
                indices = select_frame_indices(frames, 4)
            else:
                indices = [0]
            canvas = PILImage.new("RGB", (cell * 2, cell * 2), (255, 255, 255))
            try:
                for position, index in enumerate(indices[:4]):
                    try:
                        image.seek(index)
                        frame = _flatten_frame(image)
                        try:
                            frame.thumbnail((cell, cell), _LANCZOS)
                            x = (position % 2) * cell
                            y = (position // 2) * cell
                            canvas.paste(frame, (x, y))
                        finally:
                            frame.close()
                    except Exception:
                        continue
                buffer = io.BytesIO()
                canvas.save(buffer, "JPEG", quality=82, optimize=True)
                return buffer.getvalue()
            finally:
                canvas.close()
    except Exception:
        return None


def thumbnail_data_url(thumb_bytes: bytes | None) -> str | None:
    if not thumb_bytes:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(thumb_bytes).decode(
        "ascii"
    )


def image_info(data: bytes) -> dict:
    """Read basic dimensions/frame metadata for the library record."""

    info: dict = {
        "width": None,
        "height": None,
        "frames": None,
        "duration_ms": None,
    }
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            info["width"], info["height"] = image.size
            info["frames"] = int(getattr(image, "n_frames", 1) or 1)
            duration = image.info.get("duration")
            if duration is not None:
                try:
                    info["duration_ms"] = int(float(duration))
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return info


def gif_data_url(data: bytes, mime: str = "image/gif") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


__all__ = [
    "Frame",
    "extract_frames",
    "gif_data_url",
    "image_info",
    "make_thumbnail",
    "select_frame_indices",
    "thumbnail_data_url",
]
