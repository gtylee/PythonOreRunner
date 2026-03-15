#!/usr/bin/env python3
"""Compare ORE exposure/CVA vs Python LGM(+FX) for IRS/FXFwd/XCCY cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

from pythonore.compute.irs_xva_utils import (
    build_discount_curve_from_discount_pairs,
    load_ore_default_curve_inputs,
    load_ore_discount_pairs_from_curves,
    load_ore_exposure_profile,
    load_ore_legs_from_flows,
    load_swap_legs_from_portfolio,
    parse_lgm_params_from_simulation_xml,
    survival_probability_from_hazard,
)
from pythonore.compute.lgm import LGM1F, LGMParams, simulate_lgm_measure
from pythonore.compute.lgm_fx_hybrid import LgmFxHybrid, MultiCcyLgmParams
from pythonore.compute.lgm_fx_xva_utils import (
    FxForwardDef,
    XccyFloatLegDef,
    aggregate_exposure_profile,
    cva_terms_from_profile,
    fx_forward_npv,
    single_ccy_irs_npv,
    xccy_float_float_swap_npv,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product", choices=["irs_single", "fx_forward", "xccy_float_float"], required=True)
    p.add_argument("--ore-output-dir", type=Path, required=True)
    p.add_argument("--simulation-xml", type=Path, required=True)
    p.add_argument("--portfolio-xml", type=Path, default=None)
    p.add_argument("--trade-id", required=True)
    p.add_argument("--cpty", default="CPTY_A")
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--todaysmarket-xml", type=Path, required=True)
    p.add_argument("--market-data-file", type=Path, required=True)

    p.add_argument("--discount-column", default="EUR-EONIA")
    p.add_argument("--forward-column", default=None)
    p.add_argument("--model-ccy", default="EUR")
    p.add_argument(
        "--cva-discount-mode",
        choices=["curve", "none"],
        default="curve",
        help="CVA discounting convention: curve discount factors or no discounting.",
    )

    p.add_argument("--product-json", type=Path, default=None, help="Required for fx_forward/xccy_float_float")
    p.add_argument("--fx-vol", type=float, default=0.12)
    p.add_argument("--fx-corr-ir-dom", type=float, default=0.0)
    p.add_argument("--fx-corr-ir-for", type=float, default=0.0)

    return p.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_ore_cva(xva_csv: Path, cpty: str) -> float:
    with open(xva_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        tid_key = "TradeId" if r.fieldnames and "TradeId" in r.fieldnames else "#TradeId"
        for row in r:
            if row.get("NettingSetId", "") == cpty and row.get(tid_key, "") == "":
                return float(row["CVA"])
    raise ValueError(f"netting set '{cpty}' aggregate CVA not found in {xva_csv}")


def _read_ore_epe_ene(exposure_trade_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prof = load_ore_exposure_profile(str(exposure_trade_csv))
    return prof["time"], prof["epe"], prof["ene"]


def _discount_grid_from_curve(curve, t: np.ndarray) -> np.ndarray:
    return np.asarray([float(curve(float(x))) for x in t], dtype=float)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _run_irs_single(args: argparse.Namespace, times: np.ndarray) -> np.ndarray:
    curves_csv = args.ore_output_dir / "curves.csv"
    flows_csv = args.ore_output_dir / "flows.csv"
    t_d, df_d = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=args.discount_column)
    p0_d = build_discount_curve_from_discount_pairs(list(zip(t_d, df_d)))
    fwd_col = args.forward_column or args.discount_column
    t_f, df_f = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=fwd_col)
    p0_f = build_discount_curve_from_discount_pairs(list(zip(t_f, df_f)))

    sim = parse_lgm_params_from_simulation_xml(str(args.simulation_xml), ccy_key=args.model_ccy)
    model = LGM1F(
        LGMParams(
            alpha_times=tuple(float(x) for x in sim["alpha_times"]),
            alpha_values=tuple(float(x) for x in sim["alpha_values"]),
            kappa_times=tuple(float(x) for x in sim["kappa_times"]),
            kappa_values=tuple(float(x) for x in sim["kappa_values"]),
            shift=float(sim["shift"]),
            scaling=float(sim["scaling"]),
        )
    )

    exposure_csv = args.ore_output_dir / f"exposure_trade_{args.trade_id}.csv"
    date0 = load_ore_exposure_profile(str(exposure_csv))["date"][0]
    if args.portfolio_xml and args.portfolio_xml.exists():
        legs = load_swap_legs_from_portfolio(str(args.portfolio_xml), args.trade_id, asof_date=str(date0))
    else:
        legs = load_ore_legs_from_flows(str(flows_csv), trade_id=args.trade_id, asof_date=str(date0))

    rng = np.random.default_rng(args.seed)
    x = simulate_lgm_measure(model, times, args.paths, rng=rng)
    out = np.zeros_like(x)
    for i, t in enumerate(times):
        out[i, :] = single_ccy_irs_npv(model, p0_d, p0_f, legs, float(t), x[i, :])
    return out


def _build_hybrid_for_pair(
    args: argparse.Namespace,
    base: str,
    quote: str,
) -> LgmFxHybrid:
    sim_dom = parse_lgm_params_from_simulation_xml(str(args.simulation_xml), ccy_key=quote)
    sim_for = parse_lgm_params_from_simulation_xml(str(args.simulation_xml), ccy_key=base)

    p_dom = LGMParams(
        alpha_times=tuple(float(x) for x in sim_dom["alpha_times"]),
        alpha_values=tuple(float(x) for x in sim_dom["alpha_values"]),
        kappa_times=tuple(float(x) for x in sim_dom["kappa_times"]),
        kappa_values=tuple(float(x) for x in sim_dom["kappa_values"]),
        shift=float(sim_dom["shift"]),
        scaling=float(sim_dom["scaling"]),
    )
    p_for = LGMParams(
        alpha_times=tuple(float(x) for x in sim_for["alpha_times"]),
        alpha_values=tuple(float(x) for x in sim_for["alpha_values"]),
        kappa_times=tuple(float(x) for x in sim_for["kappa_times"]),
        kappa_values=tuple(float(x) for x in sim_for["kappa_values"]),
        shift=float(sim_for["shift"]),
        scaling=float(sim_for["scaling"]),
    )

    pair = f"{base}/{quote}"
    corr = np.array(
        [
            [1.0, 0.0, float(args.fx_corr_ir_for)],
            [0.0, 1.0, float(args.fx_corr_ir_dom)],
            [float(args.fx_corr_ir_for), float(args.fx_corr_ir_dom), 1.0],
        ],
        dtype=float,
    )
    params = MultiCcyLgmParams(
        ir_params={base: p_for, quote: p_dom},
        fx_vols={pair: (tuple(), (float(args.fx_vol),))},
        corr=corr,
    )
    return LgmFxHybrid(params)


def _run_fx_forward(args: argparse.Namespace, times: np.ndarray) -> np.ndarray:
    if args.product_json is None:
        raise ValueError("--product-json is required for fx_forward")
    spec = json.loads(args.product_json.read_text(encoding="utf-8"))
    fx_def = FxForwardDef(
        trade_id=str(spec["trade_id"]),
        pair=str(spec["pair"]),
        notional_base=float(spec["notional_base"]),
        strike=float(spec["strike"]),
        maturity=float(spec["maturity"]),
    )
    base, quote = fx_def.pair.upper().replace("-", "/").split("/")

    curves_csv = args.ore_output_dir / "curves.csv"
    t_q, df_q = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=str(spec["discount_columns"][quote]))
    p0_q = build_discount_curve_from_discount_pairs(list(zip(t_q, df_q)))
    t_b, df_b = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=str(spec["discount_columns"][base]))
    p0_b = build_discount_curve_from_discount_pairs(list(zip(t_b, df_b)))

    hybrid = _build_hybrid_for_pair(args, base, quote)
    rng = np.random.default_rng(args.seed)
    rd = np.asarray([-(np.log(p0_q(max(x, 1.0e-8))) / max(x, 1.0e-8)) for x in np.maximum(times, 1.0e-8)])
    rf = np.asarray([-(np.log(p0_b(max(x, 1.0e-8))) / max(x, 1.0e-8)) for x in np.maximum(times, 1.0e-8)])
    mu = float(np.mean(rd - rf))

    paths = hybrid.simulate_paths(
        times,
        args.paths,
        rng,
        log_s0={fx_def.pair: float(np.log(float(spec["spot0"])))},
        rd_minus_rf={fx_def.pair: mu},
    )

    x_q = paths["x"][quote]
    x_b = paths["x"][base]
    s = paths["s"][fx_def.pair]

    out = np.zeros((times.size, args.paths), dtype=float)
    for i, t in enumerate(times):
        out[i, :] = fx_forward_npv(
            hybrid,
            fx_def,
            float(t),
            s[i, :],
            x_q[i, :],
            x_b[i, :],
            p0_q,
            p0_b,
        )
    return out


def _run_xccy_float_float(args: argparse.Namespace, times: np.ndarray) -> np.ndarray:
    if args.product_json is None:
        raise ValueError("--product-json is required for xccy_float_float")
    spec = json.loads(args.product_json.read_text(encoding="utf-8"))
    dccy = str(spec["domestic_ccy"]).upper()
    leg1_spec = spec["leg1"]
    leg2_spec = spec["leg2"]

    def mk_leg(x: dict) -> XccyFloatLegDef:
        return XccyFloatLegDef(
            ccy=str(x["ccy"]).upper(),
            pay_time=np.asarray(x["pay_time"], dtype=float),
            start_time=np.asarray(x["start_time"], dtype=float),
            end_time=np.asarray(x["end_time"], dtype=float),
            accrual=np.asarray(x["accrual"], dtype=float),
            notional=np.asarray(x["notional"], dtype=float),
            spread=np.asarray(x["spread"], dtype=float),
            sign=np.asarray(x["sign"], dtype=float),
        )

    leg1 = mk_leg(leg1_spec)
    leg2 = mk_leg(leg2_spec)
    ccy_set = sorted({leg1.ccy, leg2.ccy, dccy})
    if len(ccy_set) != 2:
        raise ValueError("v1 xccy_float_float supports exactly two currencies")
    base, quote = (ccy_set[0], ccy_set[1])
    pair = f"{base}/{quote}"

    hybrid = _build_hybrid_for_pair(args, base, quote)

    curves_csv = args.ore_output_dir / "curves.csv"
    disc_cols = spec["discount_columns"]
    fwd_cols = spec["forward_columns"]

    dcurves = {}
    fcurves = {}
    for c in ccy_set:
        t_d, df_d = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=str(disc_cols[c]))
        t_f, df_f = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=str(fwd_cols[c]))
        dcurves[c] = build_discount_curve_from_discount_pairs(list(zip(t_d, df_d)))
        fcurves[c] = build_discount_curve_from_discount_pairs(list(zip(t_f, df_f)))

    rng = np.random.default_rng(args.seed)
    paths = hybrid.simulate_paths(
        times,
        args.paths,
        rng,
        log_s0={pair: float(np.log(float(spec["spot0"])))}
    )
    x_by = {c: paths["x"][c] for c in ccy_set}
    s_by = {pair: paths["s"][pair]}

    out = np.zeros((times.size, args.paths), dtype=float)
    for i, t in enumerate(times):
        x_t = {c: x_by[c][i, :] for c in ccy_set}
        s_t = {pair: s_by[pair][i, :]}
        out[i, :] = xccy_float_float_swap_npv(
            hybrid,
            dccy,
            leg1,
            leg2,
            float(t),
            x_t,
            s_t,
            dcurves,
            fcurves,
        )
    return out


def main() -> None:
    args = _parse_args()
    t0 = perf_counter()

    exposure_csv = args.ore_output_dir / f"exposure_trade_{args.trade_id}.csv"
    xva_csv = args.ore_output_dir / "xva.csv"
    if not exposure_csv.exists():
        raise FileNotFoundError(exposure_csv)
    if not xva_csv.exists():
        raise FileNotFoundError(xva_csv)

    times, ore_epe, ore_ene = _read_ore_epe_ene(exposure_csv)
    if args.product == "irs_single":
        npv_paths = _run_irs_single(args, times)
    elif args.product == "fx_forward":
        npv_paths = _run_fx_forward(args, times)
    else:
        npv_paths = _run_xccy_float_float(args, times)

    exp = aggregate_exposure_profile(npv_paths)
    py_epe = exp["epe"]
    py_ene = exp["ene"]
    py_ee = exp["ee"]

    credit = load_ore_default_curve_inputs(str(args.todaysmarket_xml), str(args.market_data_file), cpty_name=args.cpty)
    surv = survival_probability_from_hazard(times, np.asarray(credit["hazard_times"]), np.asarray(credit["hazard_rates"]))

    curves_csv = args.ore_output_dir / "curves.csv"
    t_d, df_d = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=args.discount_column)
    p0_d = build_discount_curve_from_discount_pairs(list(zip(t_d, df_d)))
    if args.cva_discount_mode == "none":
        df_grid = np.ones_like(times)
    else:
        df_grid = _discount_grid_from_curve(p0_d, times)

    cva_dec = cva_terms_from_profile(times, py_epe, df_grid, surv, float(credit["recovery"]))
    ore_cva = _load_ore_cva(xva_csv, args.cpty)
    py_cva = float(cva_dec["cva"][0])

    rel = np.abs(py_epe - ore_epe) / np.maximum(np.abs(ore_epe), 1.0)
    rows_diag = []
    for i in range(times.size):
        rows_diag.append(
            [
                i,
                float(times[i]),
                float(ore_epe[i]),
                float(py_epe[i]),
                float(py_epe[i] - ore_epe[i]),
                float(rel[i]),
                float(ore_ene[i]),
                float(py_ene[i]),
                float(py_ee[i]),
            ]
        )

    rows_cva = []
    for i in range(times.size):
        rows_cva.append(
            [
                i,
                float(times[i]),
                float(py_epe[i]),
                float(df_grid[i]),
                float(surv[i]),
                float(cva_dec["dpd"][i]),
                float(cva_dec["terms"][i]),
            ]
        )

    _ensure_dir(args.artifact_dir)
    diag_csv = args.artifact_dir / f"{args.trade_id}_diagnostics.csv"
    cva_csv = args.artifact_dir / f"{args.trade_id}_cva_terms.csv"
    summary_json = args.artifact_dir / f"{args.trade_id}_summary.json"

    _write_csv(
        diag_csv,
        ["idx", "time", "ORE_EPE", "PY_EPE", "EPE_abs_diff", "EPE_rel_diff", "ORE_ENE", "PY_ENE", "PY_EE"],
        rows_diag,
    )
    _write_csv(
        cva_csv,
        ["idx", "time", "PY_EPE", "DF", "Q", "dPD", "LGDxEPE_DF_dPD"],
        rows_cva,
    )

    summary = {
        "product": args.product,
        "trade_id": args.trade_id,
        "paths": int(args.paths),
        "seed": int(args.seed),
        "ore_cva": ore_cva,
        "py_cva": py_cva,
        "cva_abs_diff": py_cva - ore_cva,
        "cva_rel_diff": (py_cva - ore_cva) / ore_cva if abs(ore_cva) > 1.0e-12 else float("inf"),
        "epe_rel_median": float(np.median(rel)),
        "epe_rel_p95": float(np.percentile(rel, 95.0)),
        "epe_rel_max": float(np.max(rel)),
        "runtime_seconds": float(perf_counter() - t0),
        "cva_discount_mode": args.cva_discount_mode,
        "files": {
            "diagnostics": str(diag_csv),
            "cva_terms": str(cva_csv),
            "summary": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"ORE CVA: {ore_cva:,.2f}")
    print(f"PY  CVA: {py_cva:,.2f}")
    print(f"CVA rel diff: {summary['cva_rel_diff']:.4%}")
    print(f"Median EPE rel diff: {summary['epe_rel_median']:.4%}")
    print(f"P95 EPE rel diff: {summary['epe_rel_p95']:.4%}")
    print(f"Diagnostics: {diag_csv}")
    print(f"CVA terms:   {cva_csv}")
    print(f"Summary:     {summary_json}")


if __name__ == "__main__":
    main()
