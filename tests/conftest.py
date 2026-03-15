from __future__ import annotations

import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    src_text = str(_SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def find_engine_repo_root() -> Path | None:
    env_root = os.getenv("ENGINE_REPO_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))

    repo_root = _REPO_ROOT
    candidates.extend(
        [
            repo_root,
            repo_root.parent / "Engine",
            Path("/Users/gordonlee/Documents/Engine"),
        ]
    )

    for candidate in candidates:
        if (candidate / "Examples").exists():
            return candidate
    return None


def require_engine_repo_root() -> Path:
    root = find_engine_repo_root()
    if root is None:
        import pytest

        pytest.skip("Engine checkout with Examples/ is not available")
    return root
