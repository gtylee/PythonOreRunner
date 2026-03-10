from __future__ import annotations

import os
from pathlib import Path


def find_engine_repo_root() -> Path | None:
    env_root = os.getenv("ENGINE_REPO_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))

    repo_root = Path(__file__).resolve().parents[1]
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
