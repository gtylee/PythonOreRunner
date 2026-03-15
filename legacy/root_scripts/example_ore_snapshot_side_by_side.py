#!/usr/bin/env python3
"""Run an ORE snapshot side by side through Python LGM and ORE outputs.

Usage:
    PYTHONPATH=Tools/PythonOreRunner \\
    python Tools/PythonOreRunner/example_ore_snapshot_side_by_side.py \\
      --case-dir Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A \\
      --paths 2000 \\
      --rng ore_parity
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def _default_case_dir() -> Path:
    return TOOLS_DIR / "parity_artifacts" / "multiccy_benchmark_final" / "cases" / "flat_EUR_5Y_A"


def _samples_from_simulation_xml(xml: str) -> int | None:
    m = re.search(r"<Samples>\s*(\d+)\s*</Samples>", xml)
    return int(m.group(1)) if m else None


def _to_float(s: str | None) -> float:
    if s in (None, "", "#N/A"):
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return []
        header = [h.strip().lstrip("#") for h in first.strip().split(",")]
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [v.strip() for v in line.split(",")]
            if len(vals) < len(header):
                vals.extend([""] * (len(header) - len(vals)))
            rows.append(dict(zip(header, vals)))
        return rows


def _read_ore_xva(output_dir: Path) -> dict[str, float]:
    rows = _read_csv_rows(output_dir / "xva.csv")
    for row in rows:
        if not row.get("TradeId", "").strip():
            return {
                "CVA": _to_float(row.get("CVA")),
                "DVA": _to_float(row.get("DVA")),
                "FBA": _to_float(row.get("FBA")),
                "FCA": _to_float(row.get("FCA")),
                "FVA": _to_float(row.get("FBA")) + _to_float(row.get("FCA")),
            }
    return {}


def _read_case_xva_flags(ore_xml: Path) -> dict[str, bool]:
    out = {"CVA": True, "DVA": False, "FVA": False, "MVA": False}
    try:
        root = ET.parse(ore_xml).getroot()
    except Exception:
        return out
    analytic = root.find("./Analytics/Analytic[@type='xva']")
    if analytic is None:
        return out
    params = {
        n.attrib.get("name", "").strip(): (n.text or "").strip().upper()
        for n in analytic.findall("./Parameter")
    }
    out["CVA"] = params.get("cva", "Y") == "Y"
    out["DVA"] = params.get("dva", "N") == "Y"
    out["FVA"] = params.get("fva", "N") == "Y"
    out["MVA"] = params.get("mva", "N") == "Y"
    return out


def _repo_root() -> Path:
    return TOOLS_DIR.parent.parent


def _locate_ore_exe() -> Path:
    repo = _repo_root()
    for candidate in (repo / "build" / "App" / "ore", repo / "build" / "ore" / "App" / "ore"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ORE executable not found under build/App/ore or build/ore/App/ore")


def _set_param(parent: ET.Element, name: str, value: str) -> None:
    for node in parent.findall("./Parameter"):
        if node.attrib.get("name") == name:
            node.text = value
            return
    ET.SubElement(parent, "Parameter", {"name": name}).text = value


def _fresh_run_root(case_dir: Path) -> Path:
    return (
        _repo_root()
        / "Tools"
        / "PythonOreRunner"
        / "_fresh_ore_runs"
        / f"{case_dir.name}_{int(time.time() * 1000)}"
    )


def _prepare_fresh_ore_run(case_dir: Path, mpor_days: int | None, force_dva: bool) -> tuple[Path, Path, dict[str, object]]:
    source_input = case_dir / "Input"
    source_ore_xml = source_input / "ore.xml"
    source_sim_xml = source_input / "simulation.xml"
    run_root = _fresh_run_root(case_dir)
    input_dir = run_root / "Input"
    output_dir = run_root / "Output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    fresh_sim_xml = input_dir / "simulation.xml"
    shutil.copy2(source_sim_xml, fresh_sim_xml)

    sim_root = ET.parse(fresh_sim_xml).getroot()
    sim_params = sim_root.find("./Parameters")
    if sim_params is None:
        sim_params = ET.SubElement(sim_root, "Parameters")
    if mpor_days is not None:
        _set_param(sim_params, "CloseOutLag", f"{int(mpor_days)}D")
        _set_param(sim_params, "MporMode", "StickyDate")
    ET.ElementTree(sim_root).write(fresh_sim_xml, encoding="utf-8", xml_declaration=True)

    ore_root = ET.parse(source_ore_xml).getroot()
    output_relative_names = {
        "logFile",
        "outputFile",
        "outputFileName",
        "cubeFile",
        "aggregationScenarioDataFileName",
        "scenarioFile",
        "rawCubeOutputFile",
        "netCubeOutputFile",
    }
    for analytic in ore_root.findall("./Analytics/Analytic"):
        analytic_type = analytic.attrib.get("type")
        if analytic_type == "simulation":
            _set_param(analytic, "simulationConfigFile", str(fresh_sim_xml))
            _set_param(analytic, "cubeFile", "cube.csv.gz")
            _set_param(analytic, "aggregationScenarioDataFileName", "scenariodata.csv.gz")
        elif analytic_type == "xva":
            if force_dva:
                _set_param(analytic, "dva", "Y")
            _set_param(analytic, "cubeFile", "cube.csv.gz")
            _set_param(analytic, "scenarioFile", "scenariodata.csv.gz")
            _set_param(analytic, "rawCubeOutputFile", "rawcube.csv")
            _set_param(analytic, "netCubeOutputFile", "netcube.csv")
        elif analytic_type in ("npv", "cashflow", "curves"):
            file_param = "outputFileName"
            current = ""
            for node in analytic.findall("./Parameter"):
                if node.attrib.get("name") == file_param:
                    current = (node.text or "").strip()
                    break
            if current:
                _set_param(analytic, file_param, Path(current).name)

    for node in ore_root.findall(".//Parameter"):
        name = node.attrib.get("name", "")
        text = (node.text or "").strip()
        if name == "inputPath":
            node.text = str(source_input)
            continue
        if name == "outputPath":
            node.text = str(output_dir)
            continue
        if name in output_relative_names and text:
            node.text = Path(text).name
            continue
        if name.endswith("File") and text and name not in {"simulationConfigFile"}:
            if Path(text).is_absolute():
                node.text = str(Path(text))
            elif name not in {
                *output_relative_names,
            }:
                node.text = str((source_input / text).resolve())

    fresh_ore_xml = input_dir / "ore.xml"
    ET.ElementTree(ore_root).write(fresh_ore_xml, encoding="utf-8", xml_declaration=True)
    return fresh_ore_xml, output_dir, {
        "run_root": str(run_root),
        "fresh_ore_xml": str(fresh_ore_xml),
        "fresh_simulation_xml": str(fresh_sim_xml),
    }


def _run_fresh_ore_case(case_dir: Path, mpor_days: int | None, force_dva: bool) -> tuple[Path, Path, dict[str, object]]:
    fresh_ore_xml, output_dir, meta = _prepare_fresh_ore_run(case_dir, mpor_days=mpor_days, force_dva=force_dva)
    ore_exe = _locate_ore_exe()
    cp = subprocess.run(
        [str(ore_exe), str(fresh_ore_xml)],
        cwd=str(fresh_ore_xml.parent.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    meta = dict(meta)
    meta.update(
        {
            "ore_exe": str(ore_exe),
            "returncode": cp.returncode,
            "stdout_tail": cp.stdout[-2000:],
            "stderr_tail": cp.stderr[-2000:],
        }
    )
    if cp.returncode != 0:
        raise RuntimeError(meta["stderr_tail"] or meta["stdout_tail"] or "Fresh ORE run failed")
    return fresh_ore_xml, output_dir, meta


def _read_ore_pv(output_dir: Path) -> float:
    rows = _read_csv_rows(output_dir / "npv.csv")
    if not rows:
        return 0.0
    row = rows[0]
    for key in ("NPV(Base)", "NPV"):
        if key in row:
            return _to_float(row.get(key))
    return 0.0


def _read_first_exposure(output_dir: Path) -> tuple[float, float]:
    files = sorted(output_dir.glob("exposure_trade_*.csv"))
    if not files:
        return 0.0, 0.0
    rows = _read_csv_rows(files[0])
    if not rows:
        return 0.0, 0.0
    row = rows[0]
    return _to_float(row.get("EPE")), _to_float(row.get("ENE"))


def _read_netting_exposure_profile(output_dir: Path, netting_set: str) -> list[dict[str, float]]:
    path = output_dir / f"exposure_nettingset_{netting_set}.csv"
    rows = _read_csv_rows(path)
    out: list[dict[str, float]] = []
    for row in rows:
        out.append(
            {
                "time": _to_float(row.get("Time")),
                "epe": _to_float(row.get("EPE")),
                "ene": _to_float(row.get("ENE")),
            }
        )
    return out


def _nearest_profile_rows(
    py_times: list[float], py_epe: list[float], py_ene: list[float], ore_profile: list[dict[str, float]]
) -> list[dict[str, float]]:
    if not ore_profile:
        return []
    out: list[dict[str, float]] = []
    ore_times = [row["time"] for row in ore_profile]
    for t, epe, ene in zip(py_times, py_epe, py_ene):
        idx = min(range(len(ore_times)), key=lambda i: abs(ore_times[i] - t))
        row = ore_profile[idx]
        out.append(
            {
                "py_time": t,
                "ore_time": row["time"],
                "py_epe": epe,
                "ore_epe": row["epe"],
                "py_ene": ene,
                "ore_ene": row["ene"],
            }
        )
    return out


def _print_exposure_alignment(result, output_dir: Path) -> None:
    if "exposure_cube" not in result.cubes:
        return
    payload = result.cube("exposure_cube").payload
    if not payload:
        return
    ns = sorted(payload.keys())[0]
    py = payload[ns]
    ore_profile = _read_netting_exposure_profile(output_dir, ns)
    valuation_aligned = _nearest_profile_rows(py["times"], py["valuation_epe"], py["valuation_ene"], ore_profile)
    if not valuation_aligned:
        print("\nExposure")
        print(f"Python netting set: {ns}")
        print("ORE netting-set exposure profile not found.")
        return

    first = valuation_aligned[0]
    last = valuation_aligned[min(5, len(valuation_aligned) - 1)]
    py_peak_epe = max(py["valuation_epe"])
    py_peak_ene = max(py["valuation_ene"])
    ore_peak_epe = max(row["epe"] for row in ore_profile)
    ore_peak_ene = max(row["ene"] for row in ore_profile)
    closeout_start_epe = py["closeout_epe"][0]
    closeout_start_ene = py["closeout_ene"][0]
    closeout_peak_epe = max(py["closeout_epe"])
    closeout_peak_ene = max(py["closeout_ene"])

    print("\nExposure")
    print(f"Netting set: {ns}")
    print(
        "Valuation start: "
        f"PY t={first['py_time']:.6f} EPE={_fmt(first['py_epe'])} ENE={_fmt(first['py_ene'])} | "
        f"ORE t={first['ore_time']:.6f} EPE={_fmt(first['ore_epe'])} ENE={_fmt(first['ore_ene'])}"
    )
    print(
        "Valuation early: "
        f"PY t={last['py_time']:.6f} EPE={_fmt(last['py_epe'])} ENE={_fmt(last['py_ene'])} | "
        f"ORE t={last['ore_time']:.6f} EPE={_fmt(last['ore_epe'])} ENE={_fmt(last['ore_ene'])}"
    )
    print(
        f"Valuation peak EPE: PY={_fmt(py_peak_epe)} ORE={_fmt(ore_peak_epe)} | "
        f"Valuation peak ENE: PY={_fmt(py_peak_ene)} ORE={_fmt(ore_peak_ene)}"
    )
    print(
        f"Sticky closeout start: PY EPE={_fmt(closeout_start_epe)} ENE={_fmt(closeout_start_ene)}"
    )
    print(
        f"Sticky closeout peak EPE: PY={_fmt(closeout_peak_epe)} | "
        f"Sticky closeout peak ENE: PY={_fmt(closeout_peak_ene)}"
    )


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


def _print_metric_table(py: dict[str, float], ore: dict[str, float], ore_enabled: dict[str, bool]) -> None:
    metrics = ("PV", "CVA", "DVA", "FVA", "FBA", "FCA")
    print("\nMetric    Python LGM        ORE        Diff")
    print("------  ------------  ------------  ------------")
    for metric in metrics:
        py_v = py.get(metric, 0.0)
        enabled = ore_enabled.get(metric, ore_enabled.get("FVA", True) if metric in ("FBA", "FCA") else True)
        if metric != "PV" and not enabled:
            print(f"{metric:>6}  {_fmt(py_v):>12}  {'disabled':>12}  {'n/a':>12}")
            continue
        ore_v = ore.get(metric, 0.0)
        diff = py_v - ore_v
        print(f"{metric:>6}  {_fmt(py_v):>12}  {_fmt(ore_v):>12}  {_fmt(diff):>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact ore_snapshot side-by-side: Python LGM vs ORE")
    parser.add_argument("--case-dir", type=Path, default=_default_case_dir())
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--rng", choices=["numpy", "ore_parity"], default="ore_parity")
    parser.add_argument("--mpor-days", type=int, default=None, help="Force Python-side sticky MPOR days override")
    parser.add_argument("--rerun-ore", action="store_true", help="Run a fresh native ORE case instead of reading existing Output/")
    parser.add_argument("--ore-dva", action="store_true", help="Force DVA=Y in the fresh native ORE rerun")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    input_dir = case_dir / "Input"
    ore_xml = input_dir / "ore.xml"

    from native_xva_interface import XVALoader, XVAEngine, PythonLgmAdapter

    snap = XVALoader.from_files(str(input_dir), ore_file="ore.xml")
    num_paths = args.paths
    if num_paths is None and "simulation.xml" in snap.config.xml_buffers:
        num_paths = _samples_from_simulation_xml(snap.config.xml_buffers["simulation.xml"])
    if num_paths is None or num_paths <= 0:
        num_paths = 2000

    snap = replace(
        snap,
        config=replace(
            snap.config,
            analytics=tuple(
                m for m in ("CVA", "DVA", "FVA", "MVA")
                if (
                    m in snap.config.analytics
                    or (m == "DVA" and args.ore_dva)
                )
            ) or snap.config.analytics,
            num_paths=num_paths,
            params={
                **snap.config.params,
                "python.lgm_rng_mode": args.rng,
                **({"python.mpor_days": str(args.mpor_days)} if args.mpor_days is not None else {}),
            },
        ),
    )

    result = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snap).run(return_cubes=True)

    py_metrics = {
        "PV": float(result.pv_total),
        "CVA": float(result.xva_by_metric.get("CVA", 0.0)),
        "DVA": float(result.xva_by_metric.get("DVA", 0.0)),
        "FVA": float(result.xva_by_metric.get("FVA", 0.0)),
        "FBA": float(result.xva_by_metric.get("FBA", 0.0)),
        "FCA": float(result.xva_by_metric.get("FCA", 0.0)),
    }
    fresh_meta: dict[str, object] | None = None
    output_dir = case_dir / "Output"
    ore_flags_xml = ore_xml
    if args.rerun_ore:
        ore_flags_xml, output_dir, fresh_meta = _run_fresh_ore_case(
            case_dir,
            mpor_days=args.mpor_days,
            force_dva=args.ore_dva,
        )

    ore_enabled = _read_case_xva_flags(ore_flags_xml)
    ore_metrics = {"PV": _read_ore_pv(output_dir), **_read_ore_xva(output_dir)}
    ore_epe0, ore_ene0 = _read_first_exposure(output_dir)

    print("ORE Snapshot Side by Side")
    print(f"Case: {case_dir}")
    print(f"As-of: {snap.config.asof}")
    print(f"Trades: {len(snap.portfolio.trades)}")
    print(f"Paths: {snap.config.num_paths}")
    print(f"RNG: {args.rng}")
    print(f"ORE enabled analytics: {', '.join(k for k, v in ore_enabled.items() if v)}")
    if fresh_meta is not None:
        print(f"Fresh ORE run: {fresh_meta['run_root']}")
        print(f"ORE executable: {fresh_meta['ore_exe']}")

    print("\nMPOR")
    print(f"Enabled: {result.metadata.get('mpor_enabled')}")
    print(f"Days: {result.metadata.get('mpor_days')}")
    print(f"Mode: {result.metadata.get('mpor_mode')}")
    print(f"Source: {result.metadata.get('mpor_source')}")
    print(f"Valuation grid size: {result.metadata.get('valuation_grid_size')}")
    print(f"Closeout grid size: {result.metadata.get('closeout_grid_size')}")

    _print_metric_table(py_metrics, ore_metrics, ore_enabled)

    _print_exposure_alignment(result, output_dir)

    print("\nMetadata")
    print(f"Engine: {result.metadata.get('engine')}")
    print(f"Path count: {result.metadata.get('path_count')}")
    print(f"Input provenance: {result.metadata.get('input_provenance')}")
    if fresh_meta is not None:
        print(f"Fresh ORE xml: {fresh_meta['fresh_ore_xml']}")


if __name__ == "__main__":
    main()
