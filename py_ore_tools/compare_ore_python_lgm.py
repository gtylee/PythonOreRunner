#!/usr/bin/env python3
"""Compare ORE LGM IRS exposure/CVA against Python LGM implementation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lgm import LGM1F, LGMParams, simulate_lgm_measure
from irs_xva_utils import (
    apply_parallel_float_spread_shift_to_match_npv,
    build_discount_curve_from_discount_pairs,
    calibrate_float_spreads_from_coupon,
    load_ore_default_curve_inputs,
    load_ore_discount_pairs_from_curves,
    load_ore_exposure_profile,
    load_ore_legs_from_flows,
    load_simulation_yield_tenors,
    load_swap_legs_from_portfolio,
    parse_lgm_params_from_simulation_xml,
    parse_lgm_params_from_calibration_xml,
    survival_probability_from_hazard,
    swap_npv_from_ore_legs,
    swap_npv_from_ore_legs_dual_curve,
)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_exposure_input = repo_root / "Examples/Exposure/Input"
    default_exposure_output = repo_root / "Examples/Exposure/Output"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["fixed", "calibrated"], default="fixed")
    p.add_argument("--repo-root", type=Path, default=repo_root)
    p.add_argument("--ore-input-xml", type=Path, default=None)
    p.add_argument("--simulation-xml", type=Path, default=None)
    p.add_argument("--ore-output-dir", type=Path, default=None)
    p.add_argument("--discount-column", default="EUR-EONIA")
    p.add_argument("--forward-column", default=None, help="Forwarding curve column in curves.csv (defaults to discount column)")
    p.add_argument("--cpty", default="CPTY_A")
    p.add_argument("--trade-id", default="Swap_20")
    p.add_argument("--model-ccy", default="EUR", help="IR model currency key in simulation/calibration xml")
    p.add_argument("--swap-source", choices=["flows", "trade"], default="trade")
    p.add_argument("--portfolio-xml", type=Path, default=None)
    p.add_argument("--paths", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anchor-t0-npv", action="store_true", default=False)
    p.add_argument("--alpha-source", choices=["auto", "simulation", "calibration"], default="auto")
    p.add_argument("--calibration-xml", type=Path, default=None)
    p.add_argument("--alpha-scale", type=float, default=1.0)
    p.add_argument("--fit-alpha-scale", action="store_true")
    p.add_argument("--alpha-fit-min", type=float, default=0.6)
    p.add_argument("--alpha-fit-max", type=float, default=1.4)
    p.add_argument("--alpha-fit-steps", type=int, default=17)
    p.add_argument("--pathwise-fixing-lock", action="store_true", default=False)
    p.add_argument("--no-node-tenor-interp", action="store_true", default=False)
    p.add_argument("--no-coupon-spread-calibration", action="store_true", default=False)
    p.add_argument("--artifact-root", type=Path, default=repo_root / "Tools/PythonOreRunner/parity_artifacts")
    p.add_argument("--out-prefix", default="compare")

    ns = p.parse_args()

    if ns.ore_input_xml is None:
        fname = "ore_measure_lgm_fixed.xml" if ns.scenario == "fixed" else "ore_measure_lgm.xml"
        ns.ore_input_xml = default_exposure_input / fname
    if ns.simulation_xml is None:
        fname = "simulation_lgm_fixed.xml" if ns.scenario == "fixed" else "simulation_lgm.xml"
        ns.simulation_xml = default_exposure_input / fname
    if ns.ore_output_dir is None:
        dname = "measure_lgm_fixed" if ns.scenario == "fixed" else "measure_lgm"
        ns.ore_output_dir = default_exposure_output / dname

    return ns


def _parse_ore_setup(ore_input_xml: Path) -> dict:
    root = ET.parse(ore_input_xml).getroot()
    setup = root.find("./Setup")
    if setup is None:
        raise ValueError(f"missing Setup node in {ore_input_xml}")
    params = {n.attrib.get("name", ""): (n.text or "").strip() for n in setup.findall("./Parameter")}
    if "asofDate" not in params:
        raise ValueError(f"missing Setup/asofDate in {ore_input_xml}")
    return params


def _load_ore_cva(xva_csv: Path, cpty: str) -> float:
    with open(xva_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get("NettingSetId", "") == cpty and row.get(tid_key, "") == "":
                return float(row["CVA"])
    raise ValueError(f"could not find aggregate CVA row for netting set '{cpty}' in {xva_csv}")


def _load_ore_trade_npv(npv_csv: Path, trade_id: str) -> float:
    with open(npv_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get(tid_key, "") == trade_id:
                return float(row["NPV"])
    raise ValueError(f"could not find trade '{trade_id}' in {npv_csv}")


def _safe_rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.maximum(np.abs(b), 1.0)
    return np.abs(a - b) / den


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _max_rel_excluding_tiny(rel: np.ndarray, ref: np.ndarray, threshold: float = 1000.0) -> float:
    mask = np.abs(ref) >= threshold
    if not np.any(mask):
        return float(np.max(rel))
    return float(np.max(rel[mask]))


def _simulate_with_optional_fixing_grid(
    model: LGM1F,
    exposure_times: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
    fixing_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate on exposure grid or union(exposure, fixing>0) grid."""
    if fixing_times is None or fixing_times.size == 0:
        x_exp = simulate_lgm_measure(model, exposure_times, n_paths, rng=rng, x0=0.0)
        return x_exp, x_exp, exposure_times

    extra = np.asarray(fixing_times, dtype=float)
    extra = extra[extra > 1.0e-12]
    if extra.size == 0:
        x_exp = simulate_lgm_measure(model, exposure_times, n_paths, rng=rng, x0=0.0)
        return x_exp, x_exp, exposure_times

    sim_times = np.unique(np.concatenate((exposure_times, extra)))
    x_all = simulate_lgm_measure(model, sim_times, n_paths, rng=rng, x0=0.0)
    idx = np.searchsorted(sim_times, exposure_times)
    if not np.allclose(sim_times[idx], exposure_times, atol=1e-12, rtol=0.0):
        raise ValueError("failed to align exposure times on simulated grid")
    return x_all[idx, :], x_all, sim_times


def _compute_realized_float_coupons(
    model: LGM1F,
    p0_disc,
    p0_fwd,
    legs: dict,
    sim_times: np.ndarray,
    x_paths_on_sim_grid: np.ndarray,
) -> np.ndarray:
    """Pathwise full floating coupon (forward+spread) locked at each fixing time."""
    s = np.asarray(legs["float_start_time"], dtype=float)
    e = np.asarray(legs["float_end_time"], dtype=float)
    tau = np.asarray(legs["float_accrual"], dtype=float)
    spr = np.asarray(legs["float_spread"], dtype=float)
    fix_t = np.asarray(legs.get("float_fixing_time", s), dtype=float)
    quoted_coupon = np.asarray(legs.get("float_coupon", np.zeros_like(s)), dtype=float)

    n_cf = s.size
    n_paths = x_paths_on_sim_grid.shape[1]
    out = np.zeros((n_cf, n_paths), dtype=float)

    for i in range(n_cf):
        if tau[i] <= 0.0:
            out[i, :] = quoted_coupon[i]
            continue
        ft = float(fix_t[i])
        if ft <= 1.0e-12:
            ps = float(p0_fwd(max(0.0, float(s[i]))))
            pe = float(p0_fwd(float(e[i])))
            fwd = (ps / pe - 1.0) / float(tau[i])
            out[i, :] = fwd + float(spr[i])
            continue
        j = int(np.searchsorted(sim_times, ft))
        if j >= sim_times.size or abs(float(sim_times[j]) - ft) > 1.0e-12:
            raise ValueError(f"fixing time {ft} not present on simulation grid")
        x_fix = x_paths_on_sim_grid[j, :]
        p_ft = float(p0_disc(ft))
        p_t_s_d = model.discount_bond(ft, float(s[i]), x_fix, p_ft, float(p0_disc(float(s[i]))))
        p_t_e_d = model.discount_bond(ft, float(e[i]), x_fix, p_ft, float(p0_disc(float(e[i]))))
        bt = float(p0_fwd(ft) / p0_disc(ft))
        bs = float(p0_fwd(float(s[i])) / p0_disc(float(s[i])))
        be = float(p0_fwd(float(e[i])) / p0_disc(float(e[i])))
        p_t_s_f = p_t_s_d * (bs / bt)
        p_t_e_f = p_t_e_d * (be / bt)
        fwd_path = (p_t_s_f / p_t_e_f - 1.0) / float(tau[i])
        out[i, :] = fwd_path + float(spr[i])
    return out


def main() -> None:
    args = _parse_args()

    t0 = perf_counter()

    ore_setup = _parse_ore_setup(args.ore_input_xml)
    exposure_csv = args.ore_output_dir / f"exposure_trade_{args.trade_id}.csv"
    curves_csv = args.ore_output_dir / "curves.csv"
    flows_csv = args.ore_output_dir / "flows.csv"
    xva_csv = args.ore_output_dir / "xva.csv"
    npv_csv = args.ore_output_dir / "npv.csv"

    if not exposure_csv.exists():
        raise FileNotFoundError(exposure_csv)
    if not curves_csv.exists():
        raise FileNotFoundError(curves_csv)
    if not flows_csv.exists():
        raise FileNotFoundError(flows_csv)
    if not xva_csv.exists():
        raise FileNotFoundError(xva_csv)
    if not npv_csv.exists():
        raise FileNotFoundError(npv_csv)

    exposure = load_ore_exposure_profile(str(exposure_csv))
    times = exposure["time"]

    curve_t, curve_df = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=args.discount_column)
    p0_disc = build_discount_curve_from_discount_pairs(list(zip(curve_t, curve_df)))
    if args.forward_column is None:
        args.forward_column = args.discount_column
    fwd_t, fwd_df = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=args.forward_column)
    p0_fwd = build_discount_curve_from_discount_pairs(list(zip(fwd_t, fwd_df)))

    setup_dir = args.ore_input_xml.parent
    if args.portfolio_xml is None:
        args.portfolio_xml = setup_dir / ore_setup.get("portfolioFile", "portfolio_singleswap.xml")
    asof_date = str(exposure["date"][0])
    if args.swap_source == "trade":
        legs = load_swap_legs_from_portfolio(str(args.portfolio_xml), trade_id=args.trade_id, asof_date=asof_date)
    else:
        legs = load_ore_legs_from_flows(str(flows_csv), trade_id=args.trade_id, asof_date=asof_date)
    if not args.no_node_tenor_interp:
        legs["node_tenors"] = load_simulation_yield_tenors(str(args.simulation_xml))
    if not args.no_coupon_spread_calibration:
        legs = calibrate_float_spreads_from_coupon(legs, p0_fwd, t0=0.0)
    if args.anchor_t0_npv:
        ore_trade_npv = _load_ore_trade_npv(npv_csv, args.trade_id)
        legs = apply_parallel_float_spread_shift_to_match_npv(legs, p0_disc, ore_trade_npv, t0=0.0)

    if args.calibration_xml is None:
        args.calibration_xml = args.ore_output_dir / "calibration.xml"

    alpha_source = args.alpha_source
    if alpha_source == "auto":
        alpha_source = "calibration" if args.calibration_xml.exists() and args.scenario == "calibrated" else "simulation"

    if alpha_source == "calibration":
        if not args.calibration_xml.exists():
            raise FileNotFoundError(f"calibration xml not found: {args.calibration_xml}")
        params_dict = parse_lgm_params_from_calibration_xml(str(args.calibration_xml), ccy_key=args.model_ccy)
    else:
        params_dict = parse_lgm_params_from_simulation_xml(str(args.simulation_xml), ccy_key=args.model_ccy)
        if args.scenario == "calibrated" and params_dict["calibrate_vol"]:
            print(
                "WARNING: Simulation config has calibrated LGM vol; using simulation initial values. "
                "Set --alpha-source calibration when calibration.xml is available."
            )

    def build_model(alpha_scale: float) -> LGM1F:
        return LGM1F(
            LGMParams(
                alpha_times=tuple(float(x) for x in params_dict["alpha_times"]),
                alpha_values=tuple(float(x) * alpha_scale for x in params_dict["alpha_values"]),
                kappa_times=tuple(float(x) for x in params_dict["kappa_times"]),
                kappa_values=tuple(float(x) for x in params_dict["kappa_values"]),
                shift=float(params_dict["shift"]),
                scaling=float(params_dict["scaling"]),
            )
        )

    t1 = perf_counter()
    fixing_times = np.asarray(legs.get("float_fixing_time", legs.get("float_start_time", np.array([], dtype=float))), dtype=float)
    if args.fit_alpha_scale:
        scales = np.linspace(args.alpha_fit_min, args.alpha_fit_max, args.alpha_fit_steps)
        best_scale = None
        best_rmse = float("inf")
        for s in scales:
            model_s = build_model(float(s))
            rng_s = np.random.default_rng(args.seed)
            x_s, x_all_s, sim_times_s = _simulate_with_optional_fixing_grid(
                model_s,
                times,
                args.paths,
                rng=rng_s,
                fixing_times=fixing_times if args.pathwise_fixing_lock else None,
            )
            realized_coupon_s = None
            if args.pathwise_fixing_lock:
                realized_coupon_s = _compute_realized_float_coupons(model_s, p0_disc, p0_fwd, legs, sim_times_s, x_all_s)
            npv_s = np.zeros_like(x_s)
            for i, t in enumerate(times):
                if args.forward_column == args.discount_column:
                    if realized_coupon_s is None:
                        npv_s[i, :] = swap_npv_from_ore_legs(model_s, p0_disc, legs, float(t), x_s[i, :])
                    else:
                        npv_s[i, :] = swap_npv_from_ore_legs_dual_curve(
                            model_s,
                            p0_disc,
                            p0_disc,
                            legs,
                            float(t),
                            x_s[i, :],
                            realized_float_coupon=realized_coupon_s,
                        )
                else:
                    npv_s[i, :] = swap_npv_from_ore_legs_dual_curve(
                        model_s,
                        p0_disc,
                        p0_fwd,
                        legs,
                        float(t),
                        x_s[i, :],
                        realized_float_coupon=realized_coupon_s,
                    )
            epe_s = np.mean(np.maximum(npv_s, 0.0), axis=1)
            rmse = float(np.sqrt(np.mean(np.square(epe_s - exposure["epe"]))))
            if rmse < best_rmse:
                best_rmse = rmse
                best_scale = float(s)
        assert best_scale is not None
        args.alpha_scale = best_scale
        print(f"Best alpha scale by EPE RMSE: {best_scale:.6f} (RMSE={best_rmse:,.2f})")

    model = build_model(float(args.alpha_scale))
    rng = np.random.default_rng(args.seed)
    x_paths, x_all_paths, sim_times = _simulate_with_optional_fixing_grid(
        model,
        times,
        args.paths,
        rng=rng,
        fixing_times=fixing_times if args.pathwise_fixing_lock else None,
    )
    realized_coupon = None
    if args.pathwise_fixing_lock:
        realized_coupon = _compute_realized_float_coupons(model, p0_disc, p0_fwd, legs, sim_times, x_all_paths)
    t2 = perf_counter()

    npv = np.zeros_like(x_paths)
    for i, t in enumerate(times):
        if args.forward_column == args.discount_column:
            if realized_coupon is None:
                npv[i, :] = swap_npv_from_ore_legs(model, p0_disc, legs, float(t), x_paths[i, :])
            else:
                npv[i, :] = swap_npv_from_ore_legs_dual_curve(
                    model,
                    p0_disc,
                    p0_disc,
                    legs,
                    float(t),
                    x_paths[i, :],
                    realized_float_coupon=realized_coupon,
                )
        else:
            npv[i, :] = swap_npv_from_ore_legs_dual_curve(
                model,
                p0_disc,
                p0_fwd,
                legs,
                float(t),
                x_paths[i, :],
                realized_float_coupon=realized_coupon,
            )
    ee = np.mean(npv, axis=1)
    epe = np.mean(np.maximum(npv, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv, 0.0), axis=1)
    t3 = perf_counter()

    examples_input = (setup_dir / "../../Input").resolve()
    market_cfg = (setup_dir / ore_setup["marketConfigFile"]).resolve()
    market_data = (setup_dir / ore_setup["marketDataFile"]).resolve()
    if not market_cfg.exists() and (examples_input / "todaysmarket.xml").exists():
        market_cfg = examples_input / "todaysmarket.xml"
    if not market_data.exists() and (examples_input / "market_20160205_flat.txt").exists():
        market_data = examples_input / "market_20160205_flat.txt"

    default_inputs = load_ore_default_curve_inputs(str(market_cfg), str(market_data), cpty_name=args.cpty)
    q = survival_probability_from_hazard(times, default_inputs["hazard_times"], default_inputs["hazard_rates"])
    dpd = np.empty_like(q)
    q_prev = 1.0
    for i in range(q.size):
        dpd[i] = max(q_prev - q[i], 0.0)
        q_prev = q[i]
    lgd = 1.0 - float(default_inputs["recovery"])
    dfs = np.asarray([p0_disc(float(t)) for t in times], dtype=float)
    cva_terms = lgd * dfs * epe * dpd
    py_cva = float(np.sum(cva_terms))

    ore_cva = _load_ore_cva(xva_csv, cpty=args.cpty)
    ore_epe = exposure["epe"]
    ore_ene = exposure["ene"]

    rel_epe = _safe_rel_err(epe, ore_epe)
    rel_ene = _safe_rel_err(ene, ore_ene)

    t4 = perf_counter()

    out_dir = args.artifact_root / args.scenario
    _ensure_dir(out_dir)

    diagnostics_csv = out_dir / f"{args.out_prefix}_diagnostics.csv"
    cva_csv = out_dir / f"{args.out_prefix}_cva_terms.csv"
    summary_json = out_dir / f"{args.out_prefix}_summary.json"

    diag_rows: list[list[object]] = []
    for i in range(times.size):
        diag_rows.append(
            [
                str(exposure["date"][i]),
                float(times[i]),
                float(ore_epe[i]),
                float(epe[i]),
                float(epe[i] - ore_epe[i]),
                float(rel_epe[i]),
                float(ore_ene[i]),
                float(ene[i]),
                float(ene[i] - ore_ene[i]),
                float(rel_ene[i]),
            ]
        )
    _write_csv(
        diagnostics_csv,
        [
            "Date",
            "Time",
            "ORE_EPE",
            "PY_EPE",
            "AbsDiff_EPE",
            "RelDiff_EPE",
            "ORE_ENE",
            "PY_ENE",
            "AbsDiff_ENE",
            "RelDiff_ENE",
        ],
        diag_rows,
    )

    cva_rows: list[list[object]] = []
    for i in range(times.size):
        cva_rows.append(
            [
                str(exposure["date"][i]),
                float(times[i]),
                float(dfs[i]),
                float(q[i]),
                float(dpd[i]),
                float(epe[i]),
                float(lgd),
                float(cva_terms[i]),
            ]
        )
    _write_csv(
        cva_csv,
        ["Date", "Time", "DF", "Survival", "dPD", "PY_EPE", "LGD", "TermContribution"],
        cva_rows,
    )

    summary = {
        "scenario": args.scenario,
        "trade_id": args.trade_id,
        "model_ccy": args.model_ccy,
        "swap_source": args.swap_source,
        "portfolio_xml": str(args.portfolio_xml),
        "counterparty": args.cpty,
        "paths": args.paths,
        "seed": args.seed,
        "ore_input_xml": str(args.ore_input_xml),
        "simulation_xml": str(args.simulation_xml),
        "ore_output_dir": str(args.ore_output_dir),
        "discount_column": args.discount_column,
        "forward_column": args.forward_column,
        "alpha_source": alpha_source,
        "alpha_scale": float(args.alpha_scale),
        "pathwise_fixing_lock": bool(args.pathwise_fixing_lock),
        "use_node_tenor_interp": not bool(args.no_node_tenor_interp),
        "use_coupon_spread_calibration": not bool(args.no_coupon_spread_calibration),
        "calibration_xml": str(args.calibration_xml),
        "ore_cva": ore_cva,
        "py_cva": py_cva,
        "cva_abs_diff": py_cva - ore_cva,
        "cva_rel_diff": abs(py_cva - ore_cva) / max(abs(ore_cva), 1.0),
        "epe_rel_median": float(np.median(rel_epe)),
        "epe_rel_p95": float(np.quantile(rel_epe, 0.95)),
        "epe_rel_max": float(np.max(rel_epe)),
        "epe_rel_max_excl_tiny_ref": _max_rel_excluding_tiny(rel_epe, ore_epe),
        "ene_rel_median": float(np.median(rel_ene)),
        "ene_rel_p95": float(np.quantile(rel_ene, 0.95)),
        "ene_rel_max": float(np.max(rel_ene)),
        "ene_rel_max_excl_tiny_ref": _max_rel_excluding_tiny(rel_ene, ore_ene),
        "recovery": float(default_inputs["recovery"]),
        "curve_handle": str(default_inputs["curve_handle"]),
        "timings_seconds": {
            "setup": t1 - t0,
            "simulate": t2 - t1,
            "revalue": t3 - t2,
            "credit_and_report": t4 - t3,
            "total": t4 - t0,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Freeze a reproducible artifact set for parity debugging.
    artifact_sources = {
        "times": {"times_exposure.csv": exposure_csv},
        "curves": {"curves.csv": curves_csv},
        "flows": {"flows.csv": flows_csv},
        "exposure": {f"exposure_trade_{args.trade_id}.csv": exposure_csv},
        "xva": {"xva.csv": xva_csv},
    }
    for subdir, files in artifact_sources.items():
        dest = out_dir / subdir
        _ensure_dir(dest)
        for name, src in files.items():
            dst = dest / name
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())

    print(f"Scenario: {args.scenario}")
    print(f"Trade: {args.trade_id}  Counterparty: {args.cpty}")
    print(f"ORE CVA: {ore_cva:,.2f}")
    print(f"PY  CVA: {py_cva:,.2f}")
    print(f"CVA abs diff: {py_cva - ore_cva:,.2f}")
    print(f"CVA rel diff: {summary['cva_rel_diff']:.4%}")
    print(
        "EPE rel diff (median/p95/max): "
        f"{summary['epe_rel_median']:.4%} / {summary['epe_rel_p95']:.4%} / {summary['epe_rel_max']:.4%}"
    )
    print(f"EPE rel max excl tiny ref: {summary['epe_rel_max_excl_tiny_ref']:.4%}")
    print(
        "ENE rel diff (median/p95/max): "
        f"{summary['ene_rel_median']:.4%} / {summary['ene_rel_p95']:.4%} / {summary['ene_rel_max']:.4%}"
    )
    print(f"ENE rel max excl tiny ref: {summary['ene_rel_max_excl_tiny_ref']:.4%}")
    print(f"Diagnostics: {diagnostics_csv}")
    print(f"CVA terms:   {cva_csv}")
    print(f"Summary:     {summary_json}")


if __name__ == "__main__":
    main()
