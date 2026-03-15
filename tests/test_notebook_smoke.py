from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_name", "timeout"),
    [
        ("01_python_to_ore_swig_dataclasses.py", 90),
        ("05_joint_python_and_ore_workflow.py", 180),
    ],
)
def test_notebook_series_script_runs(script_name: str, timeout: int):
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "notebook_series" / script_name)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=timeout,
    )
