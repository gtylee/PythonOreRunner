from __future__ import annotations

import os
from pathlib import Path


PYTHONORERUNNER_ROOT = Path(__file__).resolve().parents[1]


def pythonorerunner_root() -> Path:
    return PYTHONORERUNNER_ROOT


def local_examples_root() -> Path | None:
    examples_root = PYTHONORERUNNER_ROOT / "Examples"
    if examples_root.exists():
        return examples_root
    return None


def find_examples_repo_root() -> Path | None:
    local_root = local_examples_root()
    if local_root is not None:
        return PYTHONORERUNNER_ROOT

    return find_engine_repo_root()


def require_examples_repo_root() -> Path:
    examples_root = find_examples_repo_root()
    if examples_root is None:
        raise FileNotFoundError(
            "Could not locate a repository root with Examples/. "
            "Vendor the required examples or set ENGINE_REPO_ROOT."
        )
    return examples_root


def find_engine_repo_root() -> Path | None:
    env_root = os.getenv("ENGINE_REPO_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
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
