#!/usr/bin/env python3
"""
Run ore_snapshot case (flat_EUR_5Y_A) via native_xva_interface (XVALoader + PythonLgmAdapter),
then compare Python vs ORE: CVA, DVA, EPE/ENE, PV.

Usage (from repo root):
  PYTHONPATH=Tools/PythonOreRunner python Tools/PythonOreRunner/run_ore_snapshot_native_xva.py [--paths 2000] [--seed 42]

ORE output is read from the case Output/ folder (xva.csv, exposure_trade_*.csv, npv.csv).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

def _repo_tools() -> Path:
    here = Path(__file__).resolve()
    tools = here.parent
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    return tools


def _default_case_dir() -> Path:
    return _repo_tools() / "parity_artifacts" / "multiccy_benchmark_final" / "cases" / "flat_EUR_5Y_A"


def _read_ore_xva(csv_path: Path) -> dict:
    """Parse ORE xva.csv (netting-set row: no TradeId)."""
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return out
    header = [h.strip() for h in lines[0].lstrip("#").split(",")]
    for line in lines[1:]:
        row = [c.strip() for c in line.split(",")]
        if len(row) < len(header):
            continue
        rec = dict(zip(header, row))
        trade_id = rec.get("TradeId", "").strip()
        if not trade_id:
            out["CVA"] = _to_float(rec.get("CVA"))
            out["DVA"] = _to_float(rec.get("DVA"))
            out["FBA"] = _to_float(rec.get("FBA"))
            out["FCA"] = _to_float(rec.get("FCA"))
            out["BaselEPE"] = _to_float(rec.get("BaselEPE"))
            out["BaselEEPE"] = _to_float(rec.get("BaselEEPE"))
            break
    return out


def _read_ore_npv(csv_path: Path) -> float:
    """First trade NPV(Base) from npv.csv."""
    if not csv_path.exists():
        return 0.0
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        return 0.0
    header = [h.strip() for h in lines[0].lstrip("#").split(",")]
    idx = next((i for i, h in enumerate(header) if "NPV" in h and "Base" in h), None)
    if idx is None:
        idx = next((i for i, h in enumerate(header) if h == "NPV"), 0)
    row = [c.strip() for c in lines[1].split(",")]
    return _to_float(row[idx]) if idx < len(row) else 0.0


def _read_ore_exposure_epe_ene(csv_path: Path) -> tuple[float, float]:
    """Time 0 EPE and ENE from exposure_trade_*.csv (first data row after header)."""
    if not csv_path.exists():
        return 0.0, 0.0
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        return 0.0, 0.0
    header = [h.strip() for h in lines[0].lstrip("#").split(",")]
    try:
        epe_i = header.index("EPE")
        ene_i = header.index("ENE")
    except ValueError:
        return 0.0, 0.0
    row = [c.strip() for c in lines[1].split(",")]
    epe = _to_float(row[epe_i]) if epe_i < len(row) else 0.0
    ene = _to_float(row[ene_i]) if ene_i < len(row) else 0.0
    return epe, ene


def _to_float(s: str) -> float:
    if s is None or s == "" or s == "#N/A":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _samples_from_simulation_xml(xml: str) -> int | None:
    m = re.search(r"<Samples>\s*(\d+)\s*</Samples>", xml)
    return int(m.group(1)) if m else None


def main() -> None:
    _repo_tools()
    parser = argparse.ArgumentParser(description="Run ore_snapshot via native_xva_interface and compare PY vs ORE")
    parser.add_argument("--case-dir", type=Path, default=None, help="Case dir (default: flat_EUR_5Y_A)")
    parser.add_argument("--paths", type=int, default=None, help="Override num_paths (default: from simulation.xml)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (used for display; runtime uses simulation.xml)")
    parser.add_argument("--rng", choices=["numpy", "ore_parity"], default="ore_parity", help="Python LGM RNG mode")
    args = parser.parse_args()

    case_dir = args.case_dir or _default_case_dir()
    input_dir = case_dir / "Input"
    output_dir = case_dir / "Output"
    ore_xml = input_dir / "ore.xml"
    if not ore_xml.exists():
        print(f"ORE file not found: {ore_xml}")
        sys.exit(1)

    from dataclasses import replace
    from native_xva_interface import XVALoader, XVAEngine, PythonLgmAdapter, XVAConfig

    print("Loading snapshot via XVALoader.from_files (native_xva_interface)...")
    snapshot = XVALoader.from_files(str(input_dir), ore_file="ore.xml")

    num_paths = args.paths
    if num_paths is None and "simulation.xml" in snapshot.config.xml_buffers:
        num_paths = _samples_from_simulation_xml(snapshot.config.xml_buffers["simulation.xml"])
    if num_paths is None or num_paths <= 0:
        num_paths = 2000
    cfg = replace(
        snapshot.config,
        num_paths=num_paths,
        params={**snapshot.config.params, "python.lgm_rng_mode": args.rng},
    )
    snapshot = replace(snapshot, config=cfg)

    print(f"  asof: {snapshot.config.asof}, base_currency: {snapshot.config.base_currency}")
    print(f"  trades: {len(snapshot.portfolio.trades)}, num_paths: {snapshot.config.num_paths}, rng: {args.rng}")

    print("\nRunning Python LGM (PythonLgmAdapter)...")
    engine = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False))
    result = engine.create_session(snapshot).run(return_cubes=False)

    py_pv = result.pv_total
    py_cva = result.xva_by_metric.get("CVA", 0.0)
    py_dva = result.xva_by_metric.get("DVA", 0.0)
    py_fba = result.xva_by_metric.get("FBA", 0.0)
    py_fca = result.xva_by_metric.get("FCA", 0.0)
    py_epe = result.exposure_by_netting_set
    py_epe_total = sum(py_epe.values()) if py_epe else 0.0

    print("\nReading ORE output from", output_dir)
    ore_xva = _read_ore_xva(output_dir / "xva.csv")
    ore_pv = _read_ore_npv(output_dir / "npv.csv")
    ore_cva = ore_xva.get("CVA", 0.0)
    ore_dva = ore_xva.get("DVA", 0.0)
    ore_fba = ore_xva.get("FBA", 0.0)
    ore_fca = ore_xva.get("FCA", 0.0)
    ore_basel_epe = ore_xva.get("BaselEPE", 0.0)
    ore_basel_eepe = ore_xva.get("BaselEEPE", 0.0)
    exposure_files = list(output_dir.glob("exposure_trade_*.csv"))
    ore_epe0, ore_ene0 = 0.0, 0.0
    if exposure_files:
        ore_epe0, ore_ene0 = _read_ore_exposure_epe_ene(exposure_files[0])

    print("\n--- PY vs ORE ---")
    print(f"  PV (t0):     PY={py_pv:.2f}   ORE={ore_pv:.2f}   diff={py_pv - ore_pv:.2f}")
    print(f"  CVA:         PY={py_cva:.2f}   ORE={ore_cva:.2f}   diff={py_cva - ore_cva:.2f}   rel={_rel(py_cva, ore_cva):.1%}")
    print(f"  DVA:         PY={py_dva:.2f}   ORE={ore_dva:.2f}   diff={py_dva - ore_dva:.2f}   rel={_rel(py_dva, ore_dva):.1%}")
    print(f"  FBA:         PY={py_fba:.2f}   ORE={ore_fba:.2f}   diff={py_fba - ore_fba:.2f}   rel={_rel(py_fba, ore_fba):.1%}")
    print(f"  FCA:         PY={py_fca:.2f}   ORE={ore_fca:.2f}   diff={py_fca - ore_fca:.2f}   rel={_rel(py_fca, ore_fca):.1%}")
    print(f"  EPE (t0):    PY={py_epe_total:.2f}   ORE(row1)={ore_epe0:.2f}   ORE(BaselEPE)={ore_basel_epe:.2f}")
    print(f"  ENE (t0):    ORE(row1)={ore_ene0:.2f}")
    print(f"  Basel EEPE:  ORE={ore_basel_eepe:.2f}")
    print("\nPython result metadata:", result.metadata.get("engine"), result.metadata.get("path_count"), result.metadata.get("python_lgm_rng_mode"))
    if abs(py_pv - ore_pv) > 1000:
        print("(Known: PV/EPE gap can be from leg sign, schedule, or curve mapping; DVA=0 if own-credit not configured.)")


def _rel(py_val: float, ore_val: float) -> float:
    if ore_val == 0:
        return 0.0
    return (py_val - ore_val) / abs(ore_val)


if __name__ == "__main__":
    main()
