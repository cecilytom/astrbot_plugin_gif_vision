"""Manual probe: real GIF -> frames -> current DeepSeek vision model -> library hit.

Run from the plugin root:

    python tests/manual_probe.py
    python tests/manual_probe.py "C:\\path\\to\\sample.gif"

This script intentionally does not import AstrBot.  It reads the runtime
``cmd_config.json`` only to find the configured DeepSeek key/base/model.
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gif_vision.config import GifConfig  # noqa: E402
from gif_vision.fingerprint import phash_of_frames, sha256_bytes  # noqa: E402
from gif_vision.frames import extract_frames  # noqa: E402
from gif_vision.prompt import build_vision_prompt  # noqa: E402

CONFIG_PATH = Path.home() / ".astrbot" / "data" / "cmd_config.json"
TEMP_DIR = Path.home() / ".astrbot" / "data" / "temp"


def find_latest_gif() -> Path | None:
    if not TEMP_DIR.exists():
        return None
    files = sorted(
        TEMP_DIR.glob("media_image_*.gif"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def load_deepseek_config() -> tuple[str, str, str]:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    source = next(
        (
            item
            for item in raw.get("provider_sources", [])
            if "deepseek" in str(item.get("api_base", ""))
        ),
        None,
    )
    if source is None:
        raise RuntimeError("cmd_config.json 中没有 DeepSeek provider")
    key = str((source.get("key") or [""])[0])
    base = str(source.get("api_base") or "https://api.deepseek.com/v1").rstrip("/")
    model = "deepseek-flash"
    for provider in raw.get("provider", []):
        if provider.get("provider_source_id") == source.get("id"):
            model = str(provider.get("model") or model)
            break
    return key, base, model


def chat(base: str, key: str, model: str, prompt: str, frames) -> tuple[int, str]:
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append(
            {"type": "image_url", "image_url": {"url": frame.data_url}}
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2500,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8", "replace")
            data = json.loads(body)
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content") or message.get("reasoning_content") or ""
            return response.status, str(text)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:3000]
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_gif()
    if path is None or not path.exists():
        print("没有找到 GIF；请传入路径。")
        return 2
    data = path.read_bytes()
    frames = extract_frames(data, count=6, max_edge=768)
    if not frames:
        print(f"{path.name} 不是动图或无法解码。")
        return 2
    prompt = build_vision_prompt(GifConfig(), "", 6)
    key, base, model = load_deepseek_config()
    print(f"file={path.name} bytes={len(data)} frames={len(frames)}")
    print(f"provider={base} model={model}")
    print(f"sha={sha256_bytes(data)[:16]} phash={phash_of_frames(data)}")
    status, text = chat(base, key, model, prompt, frames)
    print(f"status={status}")
    print(text[:3000])
    if status == 200 and text.strip():
        print("\n[ok] vision description generated")
        return 0
    print("\n[failed] no usable description")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
