from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"

if _SRC_ROOT.exists():
    src_text = str(_SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
