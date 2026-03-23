from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_SCRIPTS = [
    "check_demo_fx_examples.py",
    "check_demo_fx_profiles_xva.py",
    "check_lgm_irs_xva_calc.py",
    "diagnose_cashflow_triptych.py",
    "diagnose_ore_snapshot_leg_bias.py",
    "dump_ore_discount_factors.py",
    "dump_ore_lgm_rng_parity_case.py",
    "example.py",
    "example_basic.py",
    "example_ore_snapshot.py",
    "example_systemic.py",
    "plot_ore_snapshot_epe_ene_semianalytic.py",
    "run_ore_snapshot_native_xva.py",
    "run_ore_snapshot_sensitivity_compare.py",
    "run_xva_regression_pack.py",
    "strict_native_vs_py_lgm_example.py",
]


def _run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_compat_shims_import_without_sitecustomize() -> None:
    env = {"PYTHONPATH": str(REPO_ROOT)}
    proc = _run(
        [
            sys.executable,
            "-c",
            "import py_ore_tools, native_xva_interface; print('ok')",
        ],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_root_scripts_compile() -> None:
    proc = _run([sys.executable, "-m", "py_compile", *ROOT_SCRIPTS])
    assert proc.returncode == 0, proc.stderr


def test_engine_bound_examples_exit_gracefully_without_engine(tmp_path: Path) -> None:
    bogus_root = tmp_path / "missing-engine"
    env = {
        "ENGINE_REPO_ROOT": str(bogus_root),
        "ORE_EXE": str(bogus_root / "ore"),
        "FIN_SYSTEM_FOLDER": str(bogus_root / "ToyExample"),
    }
    for script in ("example.py", "example_basic.py", "example_systemic.py"):
        proc = _run([sys.executable, script], env=env)
        assert proc.returncode == 0, (script, proc.stderr)
        assert "requires" in proc.stdout or "Set " in proc.stdout
