from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[2]
_LEGACY_DIR = _REPO_ROOT / "legacy" / "native_xva_interface" / "demos"

__path__ = [str(_PKG_DIR)]
if _LEGACY_DIR.exists():
    __path__.append(str(_LEGACY_DIR))
