#!/usr/bin/env python3
"""example_ore_snapshot.py — Load ORE inputs from a single ore.xml and run a quick LGM parity check.

Demonstrates how OreSnapshot replaces the scattered --discount-column /
--forward-column / --simulation-xml flags in compare_ore_python_lgm.py.

Usage
-----
    # From the repo root:
    python Tools/PythonOreRunner/example_ore_snapshot.py

    # Point at a different ore.xml:
    python Tools/PythonOreRunner/example_ore_snapshot.py \
        --ore-xml Examples/Exposure/Input/ore_measure_lgm.xml \
        --paths 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# Make py_ore_tools importable when run from repo root.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from py_ore_tools.lgm import simulate_lgm_measure
from py_ore_tools.irs_xva_utils import (
    swap_npv_from_ore_legs_dual_curve,
)
from py_ore_tools.ore_snapshot import OreSnapshot, load_from_ore_xml


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    t_start = perf_counter()

    # ------------------------------------------------------------------
    # 1.  Load everything from ore.xml  — one call, no flags
    # ------------------------------------------------------------------
    print(f"\nLoading ORE snapshot from: {args.ore_xml}")
    snap = load_from_ore_xml(args.ore_xml)
    print(snap)
    print()

    _print_section("Snapshot summary")
    print(f"  As-of date      : {snap.asof_date}")
    print(f"  Trade           : {snap.trade_id}  (cpty: {snap.counterparty})")
    print(f"  Measure         : {snap.measure}")
    print(f"  LGM alpha source: {snap.alpha_source}")
    print(f"  Domestic ccy    : {snap.domestic_ccy}")
    print(f"  Discount column : {snap.discount_column}")
    print(f"  Forward column  : {snap.forward_column}")
    print(f"  Node tenors     : {snap.node_tenors}")
    print(f"  Exposure pts    : {snap.exposure_times.size}")
    print(f"  ORE t0 NPV      : {snap.ore_t0_npv:,.2f}")
    print(f"  ORE CVA         : {snap.ore_cva:,.2f}")
    print(f"  Recovery rate   : {snap.recovery:.2%}")
    print()

    _print_section("LGM parameters")
    p = snap.lgm_params
    print(f"  alpha_times  = {p.alpha_times}")
    print(f"  alpha_values = {p.alpha_values}")
    print(f"  kappa_times  = {p.kappa_times}")
    print(f"  kappa_values = {p.kappa_values}")
    print(f"  shift        = {p.shift}")
    print(f"  scaling      = {p.scaling}")
    print()

    # ------------------------------------------------------------------
    # 2.  Build model + simulate
    # ------------------------------------------------------------------
    _print_section(f"Simulating {args.paths:,} paths (seed={args.seed})")
    model = snap.build_model(alpha_scale=args.alpha_scale)
    rng = np.random.default_rng(args.seed)

    t_sim = perf_counter()
    x = simulate_lgm_measure(model, snap.exposure_times, args.paths, rng=rng)
    print(f"  Simulation done in {perf_counter() - t_sim:.2f}s")
    print()

    # ------------------------------------------------------------------
    # 3.  Pathwise swap re-pricing
    # ------------------------------------------------------------------
    _print_section("Re-pricing swap on simulated paths")
    t_reprice = perf_counter()

    npv = np.zeros((snap.exposure_times.size, args.paths), dtype=float)
    for i, t in enumerate(snap.exposure_times):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(
            model,
            snap.p0_disc,
            snap.p0_fwd,
            snap.legs,
            float(t),
            x[i, :],
        )

    epe = np.mean(np.maximum(npv, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv, 0.0), axis=1)
    ee = np.mean(npv, axis=1)
    print(f"  Re-pricing done in {perf_counter() - t_reprice:.2f}s")
    print()

    # ------------------------------------------------------------------
    # 4.  CVA
    # ------------------------------------------------------------------
    _print_section("CVA computation")
    q = snap.survival_probability(snap.exposure_times)
    dfs = snap.discount_factors(snap.exposure_times)
    dpd = np.diff(np.concatenate(([1.0], q)), prepend=0.0)[1:]   # Δ(1-Q)
    # Recompute: dPD_i = max(Q_{i-1} - Q_i, 0)
    dpd = np.empty(q.size)
    q_prev = 1.0
    for i in range(q.size):
        dpd[i] = max(q_prev - q[i], 0.0)
        q_prev = q[i]

    lgd = 1.0 - snap.recovery
    cva_terms = lgd * dfs * epe * dpd
    py_cva = float(np.sum(cva_terms))

    # ------------------------------------------------------------------
    # 5.  Parity report
    # ------------------------------------------------------------------
    _print_section("Parity report  (Python LGM  vs  ORE)")

    ore_epe = snap.ore_epe
    ore_ene = snap.ore_ene

    rel_epe = _safe_rel_err(epe, ore_epe)
    rel_ene = _safe_rel_err(ene, ore_ene)
    cva_rel = abs(py_cva - snap.ore_cva) / max(abs(snap.ore_cva), 1.0)

    print(f"  ORE CVA     : {snap.ore_cva:>12,.2f}")
    print(f"  Python CVA  : {py_cva:>12,.2f}")
    print(f"  CVA abs diff: {py_cva - snap.ore_cva:>+12,.2f}")
    print(f"  CVA rel diff: {cva_rel:.4%}")
    print()
    print(
        f"  EPE rel diff  median={np.median(rel_epe):.4%}  "
        f"p95={np.quantile(rel_epe, 0.95):.4%}  "
        f"max={np.max(rel_epe):.4%}"
    )
    print(
        f"  ENE rel diff  median={np.median(rel_ene):.4%}  "
        f"p95={np.quantile(rel_ene, 0.95):.4%}  "
        f"max={np.max(rel_ene):.4%}"
    )
    print()

    # Brief per-date table (first 10 and last 3 rows)
    _print_exposure_table(snap.exposure_times, snap.exposure_dates, ore_epe, epe, rel_epe)

    total = perf_counter() - t_start
    print(f"\nTotal wall time: {total:.1f}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.maximum(np.abs(b), 1.0)
    return np.abs(a - b) / den


def _print_section(title: str) -> None:
    print(f"{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_exposure_table(
    times: np.ndarray,
    dates: np.ndarray,
    ore_epe: np.ndarray,
    py_epe: np.ndarray,
    rel_err: np.ndarray,
) -> None:
    header = f"  {'Date':<14}  {'Time':>6}  {'ORE EPE':>12}  {'PY EPE':>12}  {'Rel err':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    n = times.size
    idx_to_show = list(range(min(10, n))) + (list(range(max(10, n - 3), n)) if n > 10 else [])
    prev = -1
    for i in idx_to_show:
        if prev >= 0 and i > prev + 1:
            print("  ...")
        print(
            f"  {str(dates[i]):<14}  {times[i]:>6.3f}  "
            f"{ore_epe[i]:>12,.0f}  {py_epe[i]:>12,.0f}  {rel_err[i]:>7.2%}"
        )
        prev = i
    print()


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_xml = repo_root / "Examples/Exposure/Input/ore_measure_lgm.xml"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ore-xml", type=Path, default=default_xml,
        help="Path to the ORE root XML file (default: %(default)s)",
    )
    p.add_argument(
        "--paths", type=int, default=5000,
        help="Number of Monte Carlo paths (default: %(default)s)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: %(default)s)",
    )
    p.add_argument(
        "--alpha-scale", type=float, default=1.0,
        help="Multiplicative scale on LGM alpha (default: %(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
