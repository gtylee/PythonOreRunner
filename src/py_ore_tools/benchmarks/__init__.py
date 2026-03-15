from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[2]
_LEGACY_DIR = _REPO_ROOT / "legacy" / "py_ore_tools" / "benchmarks"
_CANONICAL_DIR = _REPO_ROOT / "src" / "pythonore" / "benchmarks"

__path__ = [str(_PKG_DIR)]
for extra in (_CANONICAL_DIR, _LEGACY_DIR):
    if extra.exists():
        __path__.append(str(extra))
