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
    compute_xva_from_exposure_profile,
    compute_realized_float_coupons,
    deflate_lgm_npv_paths,
    survival_probability_from_hazard,
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
    snap = load_from_ore_xml(args.ore_xml, anchor_t0_npv=args.anchor_t0_npv)
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
    print(f"  XVA disc column : {snap.xva_discount_column}")
    print(f"  Node tenors     : {snap.node_tenors}")
    print(f"  Exposure pts    : {snap.exposure_times.size}")
    print(f"  Model DC        : {snap.model_day_counter}")
    print(f"  Report DC       : {snap.report_day_counter}")
    print(f"  ORE t0 NPV      : {snap.ore_t0_npv:,.2f}")
    print(f"  ORE CVA / DVA   : {snap.ore_cva:,.2f} / {snap.ore_dva:,.2f}")
    if snap.ore_fba != 0 or snap.ore_fca != 0:
        print(f"  ORE FBA / FCA   : {snap.ore_fba:,.2f} / {snap.ore_fca:,.2f}")
    print(f"  Recovery rate   : {snap.recovery:.2%}")
    if snap.own_hazard_times is not None:
        print(f"  Own (DVA) credit: from market (hazard pillars: {snap.own_hazard_times.size})")
    else:
        print(f"  Own (DVA) credit: fallback (--own-hazard {args.own_hazard})")
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
    x, x_all, sim_times = _simulate_with_fixing_grid(
        model=model,
        exposure_times=snap.exposure_model_times,
        fixing_times=np.asarray(snap.legs.get("float_fixing_time", []), dtype=float),
        n_paths=args.paths,
        rng=rng,
    )
    realized_coupon = compute_realized_float_coupons(
        model=model,
        p0_disc=snap.p0_disc,
        p0_fwd=snap.p0_fwd,
        legs=snap.legs,
        sim_times=sim_times,
        x_paths_on_sim_grid=x_all,
    )
    print(f"  Simulation done in {perf_counter() - t_sim:.2f}s")
    print()

    # ------------------------------------------------------------------
    # 3.  Pathwise swap re-pricing
    # ------------------------------------------------------------------
    _print_section("Re-pricing swap on simulated paths")
    t_reprice = perf_counter()

    npv = np.zeros((snap.exposure_model_times.size, args.paths), dtype=float)
    for i, t in enumerate(snap.exposure_model_times):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(
            model,
            snap.p0_disc,
            snap.p0_fwd,
            snap.legs,
            float(t),
            x[i, :],
            realized_float_coupon=realized_coupon,
        )
    if args.flip_npv_perspective:
        npv = -npv

    xva_mode = str(args.xva_mode).strip().lower()
    if xva_mode == "ore":
        npv_xva = deflate_lgm_npv_paths(
            model=model,
            p0_disc=snap.p0_disc,
            times=snap.exposure_model_times,
            x_paths=x,
            npv_paths=npv,
        )
    else:
        npv_xva = npv

    epe = np.mean(np.maximum(npv_xva, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv_xva, 0.0), axis=1)
    ee = np.mean(npv_xva, axis=1)
    print(f"  Re-pricing done in {perf_counter() - t_reprice:.2f}s")
    print()

    # ------------------------------------------------------------------
    # 4.  CVA / DVA / FVA
    # ------------------------------------------------------------------
    _print_section("CVA / DVA / FVA computation")
    times = snap.exposure_model_times
    q = snap.survival_probability(times)
    dfs = snap.discount_factors(times)
    ore_style_xva = xva_mode == "ore"

    # Counterparty default prob (for CVA)
    dpd = np.empty(q.size)
    q_prev = 1.0
    for i in range(q.size):
        dpd[i] = max(q_prev - q[i], 0.0)
        q_prev = q[i]

    lgd = 1.0 - snap.recovery
    cva_terms = lgd * epe * dpd if ore_style_xva else lgd * dfs * epe * dpd
    py_cva = float(np.sum(cva_terms))

    # Own (bank) default prob for DVA — use snapshot hazard when loaded from ORE market
    if (
        snap.own_hazard_times is not None
        and snap.own_hazard_rates is not None
        and snap.own_recovery is not None
    ):
        q_own = survival_probability_from_hazard(
            times, snap.own_hazard_times, snap.own_hazard_rates
        )
        lgd_own = 1.0 - snap.own_recovery
    else:
        q_own = survival_probability_from_hazard(
            times,
            np.array([0.5, 1.0, 5.0, 10.0]),
            np.full(4, args.own_hazard),
        )
        lgd_own = 1.0 - args.own_recovery
    dpd_own = np.empty(q_own.size)
    q_prev = 1.0
    for i in range(q_own.size):
        dpd_own[i] = max(q_prev - q_own[i], 0.0)
        q_prev = q_own[i]
    dva_terms = lgd_own * ene * dpd_own if ore_style_xva else lgd_own * dfs * ene * dpd_own
    py_dva = float(np.sum(dva_terms))

    funding_kwargs = {}
    if snap.p0_borrow is not None and snap.p0_lend is not None:
        funding_kwargs = {
            "funding_discount_borrow": np.asarray([snap.p0_borrow(float(t)) for t in times], dtype=float),
            "funding_discount_lend": np.asarray([snap.p0_lend(float(t)) for t in times], dtype=float),
            "funding_discount_ois": np.asarray([snap.p0_xva_disc(float(t)) for t in times], dtype=float),
        }
    xva_pack = compute_xva_from_exposure_profile(
        times=times,
        epe=epe,
        ene=ene,
        discount=dfs,
        survival_cpty=q,
        survival_own=q_own,
        recovery_cpty=snap.recovery,
        recovery_own=(1.0 - lgd_own),
        funding_spread=args.borrow_spread,
        exposure_discounting="numeraire_deflated" if ore_style_xva else "discount_curve",
        **funding_kwargs,
    )
    py_fca = float(xva_pack.get("fca", 0.0))
    py_fba = float(xva_pack.get("fba", 0.0))
    py_fva = float(xva_pack["fva"])

    # ------------------------------------------------------------------
    # 5.  PV & EPE/ENE diagnostic (is the gap from PV or from exposure?)
    # ------------------------------------------------------------------
    _print_section("PV & EPE/ENE diagnostic")
    ore_epe = snap.ore_epe
    ore_ene = snap.ore_ene
    t0 = times[0]
    py_t0_npv = float(np.mean(npv[0, :]))
    ore_t0_npv = snap.ore_t0_npv
    pv_diff = py_t0_npv - ore_t0_npv
    print(f"  t=0 grid point    : {t0:.6f}")
    print(f"  Python t0 NPV (EE[0]): {py_t0_npv:>12,.2f}")
    print(f"  ORE t0 NPV        : {ore_t0_npv:>12,.2f}  (diff: {pv_diff:+.2f})")
    if (py_t0_npv > 0) != (ore_t0_npv > 0):
        print(f"  >>> Sign mismatch: Python and ORE t0 NPV have opposite signs. Check Payer convention in")
        print(f"      load_swap_legs_from_portfolio (ORE may use Payer=true = we receive) or curve build.")
    print()

    # CVA: how much would we get if we used ORE EPE with our formula?
    cva_if_ore_epe = float(np.sum(lgd * ore_epe * dpd)) if ore_style_xva else float(np.sum(lgd * dfs * ore_epe * dpd))
    print(f"  CVA with Python EPE : {py_cva:>12,.2f}")
    print(f"  CVA with ORE EPE    : {cva_if_ore_epe:>12,.2f}  (same formula, ORE exposure)")
    print(f"  ORE CVA             : {snap.ore_cva:>12,.2f}")
    print(f"  → CVA gap from EPE  : {py_cva - cva_if_ore_epe:+.2f}  (Python−ORE exposure)")
    print(f"  → CVA gap from rest : {cva_if_ore_epe - snap.ore_cva:+.2f}  (formula/df/dpd vs ORE)")
    print()

    # DVA: same check
    dva_if_ore_ene = float(np.sum(lgd_own * ore_ene * dpd_own)) if ore_style_xva else float(np.sum(lgd_own * dfs * ore_ene * dpd_own))
    print(f"  DVA with Python ENE : {py_dva:>12,.2f}")
    print(f"  DVA with ORE ENE    : {dva_if_ore_ene:>12,.2f}  (same formula, ORE exposure)")
    print(f"  ORE DVA             : {snap.ore_dva:>12,.2f}")
    print(f"  → DVA gap from ENE  : {py_dva - dva_if_ore_ene:+.2f}  (Python−ORE exposure)")
    print(f"  → DVA gap from rest : {dva_if_ore_ene - snap.ore_dva:+.2f}  (formula/df/dpd vs ORE)")
    print()

    # Profile scale (discounted EPE/ENE for intuition)
    dt = np.diff(times, prepend=0.0)
    if ore_style_xva:
        py_epe_disc = float(np.sum(epe * dt))
        ore_epe_disc = float(np.sum(ore_epe * dt))
        py_ene_disc = float(np.sum(ene * dt))
        ore_ene_disc = float(np.sum(ore_ene * dt))
        print(f"  Sum(EPE*dt)     Python: {py_epe_disc:>12,.0f}  ORE: {ore_epe_disc:>12,.0f}")
        print(f"  Sum(ENE*dt)     Python: {py_ene_disc:>12,.0f}  ORE: {ore_ene_disc:>12,.0f}")
    else:
        py_epe_disc = float(np.sum(dfs * epe * dt))
        ore_epe_disc = float(np.sum(dfs * ore_epe * dt))
        py_ene_disc = float(np.sum(dfs * ene * dt))
        ore_ene_disc = float(np.sum(dfs * ore_ene * dt))
        print(f"  Sum(df*EPE*dt)  Python: {py_epe_disc:>12,.0f}  ORE: {ore_epe_disc:>12,.0f}")
        print(f"  Sum(df*ENE*dt)  Python: {py_ene_disc:>12,.0f}  ORE: {ore_ene_disc:>12,.0f}")
    print()

    # ------------------------------------------------------------------
    # 6.  Parity report
    # ------------------------------------------------------------------
    _print_section("Parity report  (Python LGM  vs  ORE)")

    rel_epe = _safe_rel_err(epe, ore_epe)
    rel_ene = _safe_rel_err(ene, ore_ene)
    cva_rel = abs(py_cva - snap.ore_cva) / max(abs(snap.ore_cva), 1.0)
    dva_rel = abs(py_dva - snap.ore_dva) / max(abs(snap.ore_dva), 1.0)

    print(f"  ORE CVA     : {snap.ore_cva:>12,.2f}  |  Python CVA  : {py_cva:>12,.2f}  (rel diff: {cva_rel:.4%})")
    print(f"  ORE DVA     : {snap.ore_dva:>12,.2f}  |  Python DVA  : {py_dva:>12,.2f}  (rel diff: {dva_rel:.4%})")
    print(f"  ORE FBA     : {snap.ore_fba:>12,.2f}  |  Python FBA  : {py_fba:>12,.2f}")
    print(f"  ORE FCA     : {snap.ore_fca:>12,.2f}  |  Python FCA  : {py_fca:>12,.2f}")
    ore_xva = snap.ore_cva - snap.ore_dva + snap.ore_fba + snap.ore_fca
    py_xva = py_cva - py_dva + py_fva
    print(f"  XVA total   : {ore_xva:>12,.2f}  |  Python total: {py_xva:>12,.2f}")
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


def _simulate_with_fixing_grid(
    model,
    exposure_times: np.ndarray,
    fixing_times: np.ndarray,
    n_paths: int,
    rng,
    draw_order: str = "time_major",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate on exposure grid plus fixing grid, return aligned views."""
    if fixing_times is None or fixing_times.size == 0:
        x_exp = simulate_lgm_measure(model, exposure_times, n_paths, rng=rng, x0=0.0, draw_order=draw_order)
        return x_exp, x_exp, exposure_times

    extra = np.asarray(fixing_times, dtype=float)
    extra = extra[extra > 1.0e-12]
    if extra.size == 0:
        x_exp = simulate_lgm_measure(model, exposure_times, n_paths, rng=rng, x0=0.0, draw_order=draw_order)
        return x_exp, x_exp, exposure_times

    sim_times = np.unique(np.concatenate((exposure_times, extra)))
    x_all = simulate_lgm_measure(model, sim_times, n_paths, rng=rng, x0=0.0, draw_order=draw_order)
    idx = np.searchsorted(sim_times, exposure_times)
    if not np.allclose(sim_times[idx], exposure_times, atol=1e-12, rtol=0.0):
        raise ValueError("failed to align exposure times on simulated grid")
    return x_all[idx, :], x_all, sim_times


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
    p.add_argument(
        "--xva-mode", choices=("classic", "ore"), default="ore",
        help="Use classic df-weighted Python XVA or ORE-style numeraire-deflated XVA (default: %(default)s)",
    )
    p.add_argument(
        "--own-hazard", type=float, default=0.01,
        help="Flat hazard rate for own (bank) survival, used for DVA (default: %(default)s)",
    )
    p.add_argument(
        "--own-recovery", type=float, default=0.4,
        help="Own recovery rate for DVA (default: %(default)s)",
    )
    p.add_argument(
        "--borrow-spread", type=float, default=0.0,
        help="Funding spread on EPE for FCA (default: %(default)s)",
    )
    p.add_argument(
        "--lend-spread", type=float, default=0.0,
        help="Funding spread on ENE for FBA (default: %(default)s)",
    )
    p.add_argument(
        "--anchor-t0-npv", action="store_true",
        help="Shift float spread so Python t0 NPV matches ORE (dual-curve can overshoot)",
    )
    p.add_argument(
        "--flip-npv-perspective", action="store_true",
        help="Negate pathwise NPV so Python matches ORE sign (use if t0 NPV has opposite sign to ORE)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
