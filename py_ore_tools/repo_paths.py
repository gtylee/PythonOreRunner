from __future__ import annotations

import os
from pathlib import Path


PYTHONORERUNNER_ROOT = Path(__file__).resolve().parents[1]


def pythonorerunner_root() -> Path:
    return PYTHONORERUNNER_ROOT


def find_engine_repo_root() -> Path | None:
    env_root = os.getenv("ENGINE_REPO_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            PYTHONORERUNNER_ROOT,
            PYTHONORERUNNER_ROOT.parent / "Engine",
            Path("/Users/gordonlee/Documents/Engine"),
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "Examples").exists():
            return candidate
    return None


def require_engine_repo_root() -> Path:
    engine_root = find_engine_repo_root()
    if engine_root is None:
        raise FileNotFoundError(
            "Could not locate an Engine checkout with Examples/. "
            "Set ENGINE_REPO_ROOT to your ORE Engine repository."
        )
    return engine_root


def default_ore_bin() -> Path:
    return require_engine_repo_root() / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore"


def local_parity_artifacts_root() -> Path:
    return PYTHONORERUNNER_ROOT / "parity_artifacts"
