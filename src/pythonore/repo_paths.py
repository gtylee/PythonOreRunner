from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


PYTHONORERUNNER_ROOT = Path(__file__).resolve().parents[2]


def pythonorerunner_root() -> Path:
    return PYTHONORERUNNER_ROOT


def local_examples_root() -> Optional[Path]:
    examples_root = PYTHONORERUNNER_ROOT / "Examples"
    if examples_root.exists():
        return examples_root
    return None


def find_examples_repo_root() -> Optional[Path]:
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


def find_engine_repo_root() -> Optional[Path]:
    env_root = os.getenv("ENGINE_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "Examples").exists():
            return candidate
        return None

    candidates: List[Path] = []

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


def find_ore_bin() -> Optional[Path]:
    env_exe = os.getenv("ORE_EXE")
    candidates: List[Path] = []
    if env_exe:
        candidate = Path(env_exe).expanduser().resolve()
        if candidate.exists():
            return candidate
        return None

    engine_root = find_engine_repo_root()
    if engine_root is not None:
        candidates.extend(
            [
                engine_root / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore",
                engine_root / "App" / "ore",
            ]
        )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def require_ore_bin() -> Path:
    ore_bin = find_ore_bin()
    if ore_bin is None:
        raise FileNotFoundError(
            "Could not locate an ORE executable. "
            "Set ORE_EXE or provide an Engine checkout with a built App/ore binary."
        )
    return ore_bin


def local_parity_artifacts_root() -> Path:
    return PYTHONORERUNNER_ROOT / "parity_artifacts"
