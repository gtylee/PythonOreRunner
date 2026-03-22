#!/usr/bin/env python3
"""Diagnose a locked-coupon quantile PFE proxy against Python swap simulation.

This script builds a low-dimensional trade-level PFE approximation for vanilla
rate swaps under the Python LGM:

- each already-fixed coupon is replaced by its fixing-time LGM quantile value
- the residual live swap is valued at the current-state quantile
- trade PFE is approximated by the positive branch across the low/high state
  quantile evaluations

It then compares that proxy to the full Python pathwise PFE built from the same
trade, curves and model parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pythonore.io.ore_snapshot import load_from_ore_xml
from pythonore.compute.lgm import (
    ORE_SOBOL_BROWNIAN_BRIDGE_SEQUENCE_TYPE,
    make_ore_gaussian_rng,
    simulate_lgm_measure,
)
from pythonore.compute.irs_xva_utils import (
    compute_realized_float_coupons,
    swap_npv_from_ore_legs_dual_curve,
)
from pythonore.runtime.exposure_profiles import ore_pfe_quantile


def _default_cases_glob() -> Path:
    return ROOT / "parity_artifacts" / "multiccy_benchmark_final" / "cases"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare a locked-coupon quantile trade PFE proxy to Python simulation."
    )
    p.add_argument(
        "--ore-xml",
        type=Path,
        default=None,
        help="Single case ore.xml to analyse in detail.",
    )
    p.add_argument(
        "--cases-root",
        type=Path,
        default=_default_cases_glob(),
        help="Benchmark cases root used for portfolio sweeps.",
    )
    p.add_argument(
        "--case-filter",
        type=str,
        default="",
        help="Substring filter applied to case directory names in sweep mode.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of cases in sweep mode.",
    )
    p.add_argument(
        "--paths",
        type=int,
        nargs="+",
        default=[10000],
        help="Simulation path counts to compare against the proxy.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument(
        "--lgm-param-source",
        type=str,
        default="simulation_xml",
        choices=("auto", "calibration_xml", "simulation_xml", "ore", "provided"),
    )
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def _iter_case_xmls(args: argparse.Namespace) -> list[Path]:
    if args.ore_xml is not None:
        return [args.ore_xml.resolve()]
    cases = sorted(Path(args.cases_root).glob("*/Input/ore.xml"))
    if args.case_filter:
        token = args.case_filter.strip().lower()
        cases = [p for p in cases if token in p.parent.parent.name.lower()]
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    return [p.resolve() for p in cases]


def _coupon_quantiles(snap, model, q_low: float, q_high: float) -> dict[str, Any]:
    fix_t = np.asarray(snap.legs.get("float_fixing_time", snap.legs["float_start_time"]), dtype=float)
    idx_tau_all = np.asarray(
        snap.legs.get("float_index_accrual", snap.legs["float_accrual"]),
        dtype=float,
    )
    spr_all = np.asarray(snap.legs["float_spread"], dtype=float)
    s_all = np.asarray(snap.legs["float_start_time"], dtype=float)
    e_all = np.asarray(snap.legs["float_end_time"], dtype=float)
    quoted_all = np.asarray(snap.legs.get("float_coupon", np.zeros_like(fix_t)), dtype=float)
    norm = NormalDist()
    z_low = norm.inv_cdf(q_low)
    z_high = norm.inv_cdf(q_high)

    q_coupon_low = np.zeros_like(fix_t)
    q_coupon_high = np.zeros_like(fix_t)
    rows: list[dict[str, float]] = []

    for k, ft in enumerate(fix_t):
        sigma = math.sqrt(max(model.zeta(float(ft)), 0.0))
        p_ft = float(snap.p0_disc(float(ft)))
        s = float(s_all[k])
        e = float(e_all[k])
        idx_tau = float(idx_tau_all[k])
        spr = float(spr_all[k])

        def cpn(xv: float) -> float:
            p_t_s_d = float(model.discount_bond(float(ft), s, xv, p_ft, float(snap.p0_disc(s))))
            p_t_e_d = float(model.discount_bond(float(ft), e, xv, p_ft, float(snap.p0_disc(e))))
            bt = float(snap.p0_fwd(float(ft)) / snap.p0_disc(float(ft)))
            bs = float(snap.p0_fwd(s) / snap.p0_disc(s))
            be = float(snap.p0_fwd(e) / snap.p0_disc(e))
            return (p_t_s_d * (bs / bt) / (p_t_e_d * (be / bt)) - 1.0) / idx_tau + spr

        x_low = sigma * z_low
        x_high = sigma * z_high
        q_coupon_low[k] = cpn(x_low)
        q_coupon_high[k] = cpn(x_high)
        rows.append(
            {
                "coupon_index": float(k + 1),
                "fix_t": float(ft),
                "x_q_low": float(x_low),
                "x_q_high": float(x_high),
                "quoted_coupon": float(quoted_all[k]),
                "coupon_q_low": float(q_coupon_low[k]),
                "coupon_q_high": float(q_coupon_high[k]),
            }
        )

    return {
        "fix_t": fix_t,
        "quoted": quoted_all,
        "q_coupon_low": q_coupon_low,
        "q_coupon_high": q_coupon_high,
        "rows": rows,
        "z_low": float(z_low),
        "z_high": float(z_high),
    }


def _run_case(ore_xml: Path, *, paths: int, seed: int, quantile: float, lgm_param_source: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    snap = load_from_ore_xml(ore_xml, lgm_param_source=lgm_param_source)
    model = snap.build_model()
    exp_times = np.asarray(snap.exposure_model_times, dtype=float)
    exp_dates = np.asarray(snap.exposure_dates)
    fix_payload = _coupon_quantiles(snap, model, 1.0 - quantile, quantile)
    fix_t = np.asarray(fix_payload["fix_t"], dtype=float)

    sim_times = np.unique(np.concatenate([exp_times, fix_t]))
    sim_times.sort()
    rng = make_ore_gaussian_rng(seed, ORE_SOBOL_BROWNIAN_BRIDGE_SEQUENCE_TYPE)
    if hasattr(rng, "configure_time_grid"):
        rng.configure_time_grid(sim_times)
    x = simulate_lgm_measure(
        model,
        sim_times,
        paths,
        rng=rng,
        x0=0.0,
        draw_order="ore_path_major",
    )
    realized = compute_realized_float_coupons(model, snap.p0_disc, snap.p0_fwd, snap.legs, sim_times, x)
    exp_idx = np.searchsorted(sim_times, exp_times)

    npv = np.zeros((exp_times.size, x.shape[1]), dtype=float)
    for i, (t, j) in enumerate(zip(exp_times, exp_idx)):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(
            model,
            snap.p0_disc,
            snap.p0_fwd,
            snap.legs,
            float(t),
            x[j, :],
            realized_float_coupon=realized,
        )

    pfe_sim = ore_pfe_quantile(npv, quantile)
    quoted_all = np.asarray(fix_payload["quoted"], dtype=float)
    q_coupon_low = np.asarray(fix_payload["q_coupon_low"], dtype=float)
    q_coupon_high = np.asarray(fix_payload["q_coupon_high"], dtype=float)
    z_low = float(fix_payload["z_low"])
    z_high = float(fix_payload["z_high"])

    def _proxy_curve_for_z(z_state: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pfe_curve = np.zeros_like(exp_times)
        pfe_low_curve = np.zeros_like(exp_times)
        pfe_high_curve = np.zeros_like(exp_times)
        q_coupon_curve = _coupon_quantiles(snap, model, 1.0 - quantile, quantile)
        q_coupon_low_curve = np.asarray(q_coupon_curve["q_coupon_low"], dtype=float)
        q_coupon_high_curve = np.asarray(q_coupon_curve["q_coupon_high"], dtype=float)
        if abs(float(q_coupon_curve["z_low"]) - float(z_state)) > 1.0e-12:
            # Rebuild coupon quantiles for the chosen state z.
            q_coupon_low_curve = np.zeros_like(fix_t)
            q_coupon_high_curve = np.zeros_like(fix_t)
            idx_tau_all_local = np.asarray(
                snap.legs.get("float_index_accrual", snap.legs["float_accrual"]),
                dtype=float,
            )
            spr_all_local = np.asarray(snap.legs["float_spread"], dtype=float)
            s_all_local = np.asarray(snap.legs["float_start_time"], dtype=float)
            e_all_local = np.asarray(snap.legs["float_end_time"], dtype=float)
            for k, ft in enumerate(fix_t):
                sigma_fix = math.sqrt(max(model.zeta(float(ft)), 0.0))
                xv = sigma_fix * float(z_state)
                p_ft = float(snap.p0_disc(float(ft)))
                s = float(s_all_local[k])
                e = float(e_all_local[k])
                idx_tau = float(idx_tau_all_local[k])
                spr = float(spr_all_local[k])
                p_t_s_d = float(model.discount_bond(float(ft), s, xv, p_ft, float(snap.p0_disc(s))))
                p_t_e_d = float(model.discount_bond(float(ft), e, xv, p_ft, float(snap.p0_disc(e))))
                bt = float(snap.p0_fwd(float(ft)) / snap.p0_disc(float(ft)))
                bs = float(snap.p0_fwd(s) / snap.p0_disc(s))
                be = float(snap.p0_fwd(e) / snap.p0_disc(e))
                cpn = (p_t_s_d * (bs / bt) / (p_t_e_d * (be / bt)) - 1.0) / idx_tau + spr
                q_coupon_low_curve[k] = cpn
                q_coupon_high_curve[k] = cpn
        for i, t in enumerate(exp_times):
            sigma = math.sqrt(max(model.zeta(float(t)), 0.0))
            x_state = np.array([sigma * float(z_state)], dtype=float)
            rc = quoted_all.reshape(-1, 1).copy()
            locked = fix_t <= float(t) + 1.0e-12
            rc[locked, 0] = q_coupon_low_curve[locked]
            v_state = float(
                swap_npv_from_ore_legs_dual_curve(
                    model,
                    snap.p0_disc,
                    snap.p0_fwd,
                    snap.legs,
                    float(t),
                    x_state,
                    realized_float_coupon=rc,
                )[0]
            )
            pfe_low_curve[i] = max(v_state, 0.0)
            pfe_high_curve[i] = max(v_state, 0.0)
            pfe_curve[i] = max(v_state, 0.0)
        return pfe_curve, pfe_low_curve, pfe_high_curve

    pfe_approx, pfe_low, pfe_high = _proxy_curve_for_z(z_low)

    abs_diff = np.abs(pfe_approx - pfe_sim)
    rel_diff = abs_diff / np.maximum(np.abs(pfe_sim), 1.0)
    peak_idx = int(np.argmax(pfe_sim))
    one_idx = int(np.argmin(np.abs(exp_times - 1.0)))
    worst_idx = int(np.argmax(rel_diff))

    def _solve_effective_z(i: int) -> float:
        target = float(pfe_sim[i])
        if target <= 1.0e-12:
            return float("nan")
        lo, hi = -4.0, 0.0
        vlo = _proxy_curve_for_z(lo)[0][i]
        vhi = _proxy_curve_for_z(hi)[0][i]
        if target >= vlo:
            return lo
        if target <= vhi:
            return hi
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            vmid = _proxy_curve_for_z(mid)[0][i]
            if vmid > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    z_eff = np.asarray([_solve_effective_z(i) for i in range(exp_times.size)], dtype=float)
    active = np.isfinite(z_eff)
    if np.any(active):
        z_cal = float(np.median(z_eff[active]))
        pfe_cal, _, _ = _proxy_curve_for_z(z_cal)
        cal_abs = np.abs(pfe_cal - pfe_sim)
        cal_rel = cal_abs / np.maximum(np.abs(pfe_sim), 1.0)
        calibrated = {
            "state_z": z_cal,
            "pfe_mean_abs": float(np.mean(cal_abs)),
            "pfe_p95_abs": float(np.quantile(cal_abs, 0.95)),
            "pfe_mean_rel": float(np.mean(cal_rel)),
            "pfe_p95_rel": float(np.quantile(cal_rel, 0.95)),
        }
    else:
        calibrated = None

    rows = []
    for i in range(exp_times.size):
        rows.append(
            {
                "date": str(exp_dates[i]),
                "time": float(exp_times[i]),
                "pfe_sim": float(pfe_sim[i]),
                "pfe_approx": float(pfe_approx[i]),
                "pfe_low": float(pfe_low[i]),
                "pfe_high": float(pfe_high[i]),
                "abs_diff": float(abs_diff[i]),
                "rel_diff": float(rel_diff[i]),
            }
        )

    return {
        "case": ore_xml.parent.parent.name,
        "ore_xml": str(ore_xml),
        "paths": int(paths),
        "seed": int(seed),
        "quantile": float(quantile),
        "runtime_sec": float(time.perf_counter() - t0),
        "summary": {
            "pfe_mean_abs": float(np.mean(abs_diff)),
            "pfe_p95_abs": float(np.quantile(abs_diff, 0.95)),
            "pfe_max_abs": float(np.max(abs_diff)),
            "pfe_mean_rel": float(np.mean(rel_diff)),
            "pfe_p95_rel": float(np.quantile(rel_diff, 0.95)),
            "pfe_max_rel": float(np.max(rel_diff)),
            "one_year": rows[one_idx],
            "peak": rows[peak_idx],
            "worst": rows[worst_idx],
            "calibrated_state_quantile": calibrated,
        },
        "coupon_quantiles": fix_payload["rows"],
        "profile_rows": rows,
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "paths",
        "runtime_sec",
        "pfe_mean_abs",
        "pfe_p95_abs",
        "pfe_max_abs",
        "pfe_mean_rel",
        "pfe_p95_rel",
        "pfe_max_rel",
        "peak_time",
        "peak_sim",
        "peak_approx",
        "peak_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if "summary" not in result:
                continue
            peak = result["summary"]["peak"]
            writer.writerow(
                {
                    "case": result["case"],
                    "paths": result["paths"],
                    "runtime_sec": result["runtime_sec"],
                    "pfe_mean_abs": result["summary"]["pfe_mean_abs"],
                    "pfe_p95_abs": result["summary"]["pfe_p95_abs"],
                    "pfe_max_abs": result["summary"]["pfe_max_abs"],
                    "pfe_mean_rel": result["summary"]["pfe_mean_rel"],
                    "pfe_p95_rel": result["summary"]["pfe_p95_rel"],
                    "pfe_max_rel": result["summary"]["pfe_max_rel"],
                    "peak_time": peak["time"],
                    "peak_sim": peak["pfe_sim"],
                    "peak_approx": peak["pfe_approx"],
                    "peak_rel": peak["rel_diff"],
                }
            )


def main() -> None:
    args = _parse_args()
    case_xmls = _iter_case_xmls(args)
    if not case_xmls:
        raise SystemExit("No cases matched")

    results: list[dict[str, Any]] = []
    for ore_xml in case_xmls:
        for n_paths in args.paths:
            try:
                results.append(
                    _run_case(
                        ore_xml,
                        paths=int(n_paths),
                        seed=int(args.seed),
                        quantile=float(args.quantile),
                        lgm_param_source=str(args.lgm_param_source),
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "case": ore_xml.parent.parent.name,
                        "ore_xml": str(ore_xml),
                        "paths": int(n_paths),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    success = [r for r in results if "summary" in r]
    aggregate: dict[str, Any] = {"cases": len(success)}
    if success:
        by_paths = sorted({int(r["paths"]) for r in success})
        agg_rows = []
        for n_paths in by_paths:
            subset = [r for r in success if int(r["paths"]) == int(n_paths)]
            p95 = np.asarray([r["summary"]["pfe_p95_rel"] for r in subset], dtype=float)
            mean_rel = np.asarray([r["summary"]["pfe_mean_rel"] for r in subset], dtype=float)
            peak_rel = np.asarray([r["summary"]["peak"]["rel_diff"] for r in subset], dtype=float)
            agg_rows.append(
                {
                    "paths": int(n_paths),
                    "cases": int(len(subset)),
                    "pfe_p95_rel_median": float(np.median(p95)),
                    "pfe_p95_rel_mean": float(np.mean(p95)),
                    "pfe_p95_rel_max": float(np.max(p95)),
                    "pfe_mean_rel_median": float(np.median(mean_rel)),
                    "peak_rel_median": float(np.median(peak_rel)),
                    "peak_rel_max": float(np.max(peak_rel)),
                }
            )
        aggregate["by_paths"] = agg_rows

    payload = {
        "cases_root": str(args.cases_root.resolve()),
        "paths": [int(x) for x in args.paths],
        "seed": int(args.seed),
        "quantile": float(args.quantile),
        "results": results,
        "aggregate": aggregate,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.out_csv is not None:
        _write_csv(args.out_csv, results)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
