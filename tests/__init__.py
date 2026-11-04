"""Test package bootstrap.

The DSH Windows sandbox runs Python under a restricted token.  ``tempfile``
creates its random directories with mode 0o700, which on this host produces a
DACL the restricted token cannot write into.  Tests therefore create their
temporary directories with a plain ``Path.mkdir()`` under the plugin workspace,
which inherits the sandbox grant correctly.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

_TEST_ROOT = Path(__file__).resolve().parents[1] / ".testwork"


def _mkdtemp(suffix: str | None = None, prefix: str | None = None, dir: str | None = None) -> str:
    base = Path(dir) if dir else _TEST_ROOT
    base.mkdir(parents=True, exist_ok=True)
    name = f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
    path = base / name
    path.mkdir()
    return str(path)


tempfile.mkdtemp = _mkdtemp  # type: ignore[assignment]
