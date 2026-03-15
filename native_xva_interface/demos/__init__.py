from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_LEGACY_DIR = _PKG_DIR.parents[1] / "legacy" / "native_xva_interface" / "demos"

__path__ = [str(_PKG_DIR)]
if _LEGACY_DIR.exists():
    __path__.append(str(_LEGACY_DIR))
