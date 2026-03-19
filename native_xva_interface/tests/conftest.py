from __future__ import annotations

import sys
from pathlib import Path


# Allow `import native_xva_interface` when pytest is launched from the repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def find_engine_repo_root() -> Path | None:
    import os

    env_root = os.getenv("ENGINE_REPO_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            PACKAGE_ROOT,
            PACKAGE_ROOT.parent / "Engine",
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


def require_examples_repo_root() -> Path:
    local_examples = PACKAGE_ROOT / "Examples"
    if local_examples.exists():
        return PACKAGE_ROOT
    return require_engine_repo_root()
