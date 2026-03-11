#!/usr/bin/env python3
"""Run a focused sticky-MPOR verification sweep.

This script does two things:
1. Fresh ORE-vs-Python LGM reruns on a small set of parity cases with
   sticky MPOR enabled and DVA turned on.
2. A native ORE classic-vs-AMC-CG check on the AmericanMonteCarlo example
   that already carries sticky closeout settings in its simulation config.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from example_ore_snapshot_side_by_side import _prepare_fresh_ore_run
from native_xva_interface import XVALoader, XVAEngine, PythonLgmAdapter


REPO_ROOT = TOOLS_DIR.parent.parent
DEFAULT_CASES = (
    "flat_EUR_5Y_A",
    "flat_EUR_5Y_B",
    "flat_EUR_10Y_A",
    "flat_USD_5Y_A",
    "flat_GBP_5Y_A",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=int, default=2000)
    p.add_argument("--rng", choices=("numpy", "ore_parity"), default="ore_parity")
    p.add_argument("--mpor-days", type=int, default=10)
    p.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    p.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "sticky_mpor_verification_latest.json",
    )
    return p.parse_args()


def _ore_bin() -> Path:
    for candidate in (REPO_ROOT / "build" / "App" / "ore", REPO_ROOT / "build" / "ore" / "App" / "ore"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ORE executable not found under local build tree")


def _case_dir(name: str) -> Path:
    return (
        REPO_ROOT
        / "Tools"
        / "PythonOreRunner"
        / "parity_artifacts"
        / "multiccy_benchmark_final"
        / "cases"
        / name
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader((line.lstrip("#") for line in handle if line.strip())))


def _load_ore_xva(output_dir: Path) -> dict[str, float]:
    for row in _read_csv_rows(output_dir / "xva.csv"):
        if not row.get("TradeId", "").strip():
            return {
                "CVA": float(row.get("CVA", 0.0) or 0.0),
                "DVA": float(row.get("DVA", 0.0) or 0.0),
                "FBA": float(row.get("FBA", 0.0) or 0.0),
                "FCA": float(row.get("FCA", 0.0) or 0.0),
            }
    return {}


def _load_ore_pv(output_dir: Path) -> float:
    rows = _read_csv_rows(output_dir / "npv.csv")
    if not rows:
        return 0.0
    row = rows[0]
    return float(row.get("NPV(Base)", row.get("NPV", 0.0)) or 0.0)


def _load_ore_netting_profile(output_dir: Path, netting_set: str) -> list[dict[str, float]]:
    rows = _read_csv_rows(output_dir / f"exposure_nettingset_{netting_set}.csv")
    return [
        {
            "time": float(r.get("Time", 0.0) or 0.0),
            "epe": float(r.get("EPE", 0.0) or 0.0),
            "ene": float(r.get("ENE", 0.0) or 0.0),
        }
        for r in rows
    ]


def _nearest_row(time_value: float, rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["time"] - time_value))


def _safe_rel(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(rhs), 1.0)


def _run_case(case_name: str, paths: int, rng: str, mpor_days: int, ore_bin: Path) -> dict[str, object]:
    case_dir = _case_dir(case_name)
    input_dir = case_dir / "Input"
    snap = XVALoader.from_files(str(input_dir), ore_file="ore.xml")
    snap = replace(
        snap,
        config=replace(
            snap.config,
            analytics=tuple(m for m in ("CVA", "DVA", "FVA", "MVA") if m in snap.config.analytics or m == "DVA"),
            num_paths=paths,
            params={
                **snap.config.params,
                "python.lgm_rng_mode": rng,
                "python.mpor_days": str(mpor_days),
            },
        ),
    )

    py_result = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snap).run(return_cubes=True)
    fresh_ore_xml, output_dir, meta = _prepare_fresh_ore_run(case_dir, mpor_days=mpor_days, force_dva=True)
    cp = subprocess.run([str(ore_bin), str(fresh_ore_xml)], cwd=str(fresh_ore_xml.parent.parent), capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"{case_name}: fresh ORE run failed: {cp.stderr[-1000:] or cp.stdout[-1000:]}")

    ore_xva = _load_ore_xva(output_dir)
    ore_pv = _load_ore_pv(output_dir)
    ns = sorted(py_result.cube("exposure_cube").payload.keys())[0]
    py_exp = py_result.cube("exposure_cube").payload[ns]
    ore_profile = _load_ore_netting_profile(output_dir, ns)
    ore_start = _nearest_row(0.0, ore_profile) or {"epe": 0.0, "ene": 0.0, "time": 0.0}
    ore_early = _nearest_row(py_exp["times"][5], ore_profile) or ore_start

    return {
        "case": case_name,
        "fresh_run_root": meta["run_root"],
        "pv_py": float(py_result.pv_total),
        "pv_ore": ore_pv,
        "pv_abs_diff": abs(float(py_result.pv_total) - ore_pv),
        "cva_py": float(py_result.xva_by_metric.get("CVA", 0.0)),
        "cva_ore": float(ore_xva.get("CVA", 0.0)),
        "cva_rel_diff": _safe_rel(float(py_result.xva_by_metric.get("CVA", 0.0)), float(ore_xva.get("CVA", 0.0))),
        "dva_py": float(py_result.xva_by_metric.get("DVA", 0.0)),
        "dva_ore": float(ore_xva.get("DVA", 0.0)),
        "dva_rel_diff": _safe_rel(float(py_result.xva_by_metric.get("DVA", 0.0)), float(ore_xva.get("DVA", 0.0))),
        "valuation_epe0_py": float(py_exp["valuation_epe"][0]),
        "valuation_epe0_ore": float(ore_start["epe"]),
        "valuation_ene0_py": float(py_exp["valuation_ene"][0]),
        "valuation_ene0_ore": float(ore_start["ene"]),
        "valuation_epe_early_py": float(py_exp["valuation_epe"][5]),
        "valuation_epe_early_ore": float(ore_early["epe"]),
        "valuation_ene_early_py": float(py_exp["valuation_ene"][5]),
        "valuation_ene_early_ore": float(ore_early["ene"]),
        "closeout_epe0_py": float(py_exp["closeout_epe"][0]),
        "closeout_ene0_py": float(py_exp["closeout_ene"][0]),
    }


def _rewrite_output_path(root: ET.Element, output_subdir: str) -> None:
    for node in root.findall("./Setup/Parameter[@name='outputPath']"):
        node.text = output_subdir


def _rewrite_input_references(root: ET.Element, input_dir: Path, source_input_dir: Path) -> None:
    for node in root.findall(".//Parameter"):
        name = node.attrib.get("name", "")
        text = (node.text or "").strip()
        if name == "inputPath":
            node.text = "Input"
            continue
        if not text:
            continue
        if name in {"dimEvolutionFile", "dimRegressionFiles"}:
            node.text = Path(text).name
            continue
        if name.endswith("File") and name not in {
            "outputFile",
            "outputFileName",
            "cubeFile",
            "scenarioFile",
            "aggregationScenarioDataFileName",
            "rawCubeOutputFile",
            "netCubeOutputFile",
            "logFile",
        }:
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = (source_input_dir / candidate).resolve()
            node.text = str(candidate)


def _run_native_amccg_check(ore_bin: Path) -> dict[str, object]:
    src = REPO_ROOT / "Examples" / "AmericanMonteCarlo" / "Input"
    run_root = REPO_ROOT / "Tools" / "PythonOreRunner" / "_fresh_amccg_runs" / f"amccg_{int(time.time() * 1000)}"
    input_dir = run_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, input_dir, dirs_exist_ok=True)

    for ore_name, out_subdir in (("ore_classic.xml", "Output/classic"), ("ore_amccg.xml", "Output/amccg")):
        ore_path = input_dir / ore_name
        root = ET.parse(ore_path).getroot()
        for analytic in root.findall("./Analytics/Analytic[@type='cashflow']"):
            for node in analytic.findall("./Parameter[@name='active']"):
                node.text = "N"
        _rewrite_output_path(root, out_subdir)
        _rewrite_input_references(root, input_dir, src)
        ET.ElementTree(root).write(ore_path, encoding="utf-8", xml_declaration=True)
        cp = subprocess.run([str(ore_bin), str(ore_path.relative_to(run_root))], cwd=str(run_root), capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"{ore_name}: {cp.stderr[-1000:] or cp.stdout[-1000:]}")

    classic_dir = run_root / "Output" / "classic"
    amccg_dir = run_root / "Output" / "amccg"
    classic_xva = _load_ore_xva(classic_dir)
    amccg_xva = _load_ore_xva(amccg_dir)
    classic_npv = _load_ore_pv(classic_dir)
    amccg_npv = _load_ore_pv(amccg_dir)
    classic_rows = _read_csv_rows(classic_dir / "exposure_nettingset_CPTY_A.csv")
    amccg_rows = _read_csv_rows(amccg_dir / "exposure_nettingset_CPTY_A.csv")

    return {
        "run_root": str(run_root),
        "classic_npv": classic_npv,
        "amccg_npv": amccg_npv,
        "npv_abs_diff": abs(classic_npv - amccg_npv),
        "classic_cva": float(classic_xva.get("CVA", 0.0)),
        "amccg_cva": float(amccg_xva.get("CVA", 0.0)),
        "cva_rel_diff": _safe_rel(float(amccg_xva.get("CVA", 0.0)), float(classic_xva.get("CVA", 0.0))),
        "classic_exposure_rows": len(classic_rows),
        "amccg_exposure_rows": len(amccg_rows),
        "simulation_closeout_lag": (ET.parse(input_dir / "simulation_amccg.xml").getroot().findtext("./Parameters/CloseOutLag") or "").strip(),
        "simulation_mpor_mode": (ET.parse(input_dir / "simulation_amccg.xml").getroot().findtext("./Parameters/MporMode") or "").strip(),
    }


def main() -> int:
    args = _parse_args()
    ore_bin = _ore_bin()

    case_rows: list[dict[str, object]] = []
    for case_name in args.cases:
        row = _run_case(case_name, paths=args.paths, rng=args.rng, mpor_days=args.mpor_days, ore_bin=ore_bin)
        case_rows.append(row)
        print(
            f"{case_name}: PV abs={row['pv_abs_diff']:.2f} "
            f"CVA rel={100.0 * float(row['cva_rel_diff']):.2f}% "
            f"DVA rel={100.0 * float(row['dva_rel_diff']):.2f}% "
            f"ValEPE0 PY/ORE={row['valuation_epe0_py']:.2f}/{row['valuation_epe0_ore']:.2f}"
        )

    try:
        amccg = _run_native_amccg_check(ore_bin)
        print(
            "amccg_check: "
            f"NPV abs={amccg['npv_abs_diff']:.6f} "
            f"CVA rel={100.0 * float(amccg['cva_rel_diff']):.2f}% "
            f"rows classic/amccg={amccg['classic_exposure_rows']}/{amccg['amccg_exposure_rows']} "
            f"closeout={amccg['simulation_closeout_lag']} mode={amccg['simulation_mpor_mode']}"
        )
    except Exception as exc:
        amccg = {"error": str(exc)}
        print(f"amccg_check: error={exc}")

    payload = {
        "paths": args.paths,
        "rng": args.rng,
        "mpor_days": args.mpor_days,
        "cases": case_rows,
        "amccg_check": amccg,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
