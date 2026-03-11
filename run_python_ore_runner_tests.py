from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _default_targets(repo_root: Path) -> list[str]:
    base = repo_root / "Tools" / "PythonOreRunner"
    return [
        str(base / "tests"),
        str(base / "native_xva_interface" / "tests"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the PythonOreRunner pytest suite with the correct PYTHONPATH."
    )
    _, pytest_args = parser.parse_known_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    py_path = str(repo_root / "Tools" / "PythonOreRunner")
    env["PYTHONPATH"] = py_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else py_path

    explicit_paths = [a for a in pytest_args if not a.startswith("-")]
    pytest_targets = explicit_paths if explicit_paths else _default_targets(repo_root)
    extra_flags = [a for a in pytest_args if a.startswith("-")]

    cmd = [sys.executable, "-m", "pytest", *pytest_targets, *extra_flags]
    return subprocess.run(cmd, cwd=repo_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
