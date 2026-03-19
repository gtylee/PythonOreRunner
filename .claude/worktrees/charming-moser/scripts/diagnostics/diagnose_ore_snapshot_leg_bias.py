#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_ore_tools.lgm import simulate_lgm_measure
from py_ore_tools.ore_snapshot import load_from_ore_xml
from py_ore_tools.irs_xva_utils import _discount_bond_block


def _dual_curve_components(model, p0_disc, p0_fwd, legs, t: float, x_t: np.ndarray):
    x = np.asarray(x_t, dtype=float)
    p_t_d = p0_disc(t)

    node_tenors = np.asarray(legs.get("node_tenors", np.array([], dtype=float)), dtype=float)
    use_nodes = node_tenors.size > 0
    grid = t + node_tenors if use_nodes else np.array([], dtype=float)
    bt = p0_fwd(t) / p0_disc(t)

    p_nodes_d = None
    logp = None
    slope = None
    if use_nodes:
        p_nodes_d = _discount_bond_block(model, p0_disc, t, grid, x, p_t_d)
        logp = np.log(np.clip(p_nodes_d, 1.0e-18, None))
        slope = (logp[-1] - logp[-2]) / max(grid[-1] - grid[-2], 1.0e-12)

    def interp_from_nodes_batch(T: np.ndarray) -> np.ndarray:
        maturities = np.asarray(T, dtype=float)
        if maturities.size == 0:
            return np.empty((0, x.size), dtype=float)
        if not use_nodes:
            return _discount_bond_block(model, p0_disc, t, maturities, x, p_t_d)

        out = np.empty((maturities.size, x.size), dtype=float)
        immediate = maturities <= t + 1.0e-14
        if np.any(immediate):
            out[immediate, :] = 1.0

        exact_short = (~immediate) & (maturities <= grid[0])
        if np.any(exact_short):
            out[exact_short, :] = _discount_bond_block(model, p0_disc, t, maturities[exact_short], x, p_t_d)

        extrap = maturities >= grid[-1]
        extrap &= ~immediate
        extrap &= ~exact_short
        if np.any(extrap):
            out[extrap, :] = np.exp(logp[-1][None, :] + (maturities[extrap] - grid[-1])[:, None] * slope[None, :])

        interior = ~(immediate | exact_short | extrap)
        if np.any(interior):
            mids = maturities[interior]
            j = np.searchsorted(grid, mids, side="right")
            left = j - 1
            denom = np.maximum(grid[j] - grid[left], 1.0e-12)
            w = (mids - grid[left]) / denom
            out[interior, :] = np.exp((1.0 - w)[:, None] * logp[left, :] + w[:, None] * logp[j, :])

        return out

    def map_forward_bond_from_disc_batch(T: np.ndarray, p_t_T_disc: np.ndarray) -> np.ndarray:
        maturities = np.asarray(T, dtype=float)
        bT = np.fromiter((float(p0_fwd(float(m))) / p0_disc(float(m)) for m in maturities), dtype=float, count=maturities.size)
        return p_t_T_disc * (bT / bt)[:, None]

    fixed_pv = np.zeros_like(x, dtype=float)
    float_pv = np.zeros_like(x, dtype=float)

    mask_f = legs["fixed_pay_time"] > t + 1e-12
    if np.any(mask_f):
        pay = legs["fixed_pay_time"][mask_f]
        disc = interp_from_nodes_batch(pay)
        cash = legs["fixed_amount"][mask_f]
        fixed_pv += np.sum(cash[:, None] * disc, axis=0)

    fix_t = np.asarray(legs.get("float_fixing_time", legs["float_start_time"]), dtype=float)
    pay_all = legs["float_pay_time"]
    live = pay_all > t + 1e-12
    if np.any(live):
        s = legs["float_start_time"][live]
        e = legs["float_end_time"][live]
        pay = pay_all[live]
        tau = legs["float_accrual"][live]
        n = legs["float_notional"][live]
        sign = legs["float_sign"][live]
        spread = legs["float_spread"][live]
        fixed = fix_t[live] <= t + 1.0e-12

        p_tp_d = interp_from_nodes_batch(pay)
        amount = np.zeros((pay.size, x.size), dtype=float)

        if np.any(fixed):
            coupon_fix = np.tile(legs["float_coupon"][live][fixed][:, None], (1, x_t.size))
            amount[fixed, :] = sign[fixed, None] * n[fixed, None] * coupon_fix * tau[fixed, None]

        if np.any(~fixed):
            s2 = s[~fixed]
            e2 = e[~fixed]
            tau2 = tau[~fixed]
            n2 = n[~fixed]
            sign2 = sign[~fixed]
            spread2 = spread[~fixed]
            p_ts_d2 = interp_from_nodes_batch(s2)
            p_te_d2 = interp_from_nodes_batch(e2)
            p_ts_f2 = map_forward_bond_from_disc_batch(s2, p_ts_d2)
            p_te_f2 = map_forward_bond_from_disc_batch(e2, p_te_d2)
            fwd2 = (p_ts_f2 / p_te_f2 - 1.0) / tau2[:, None]
            amount[~fixed, :] = sign2[:, None] * n2[:, None] * (fwd2 + spread2[:, None]) * tau2[:, None]

        float_pv += np.sum(amount * p_tp_d, axis=0)

    return fixed_pv, float_pv


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose fixed/float leg bias for OreSnapshot parity")
    parser.add_argument("--ore-xml", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snap = load_from_ore_xml(args.ore_xml)
    model = snap.build_model()
    rng = np.random.default_rng(args.seed)
    x = simulate_lgm_measure(model, snap.exposure_model_times, n_paths=args.paths, rng=rng)

    n_t = snap.exposure_model_times.size
    ee_fix = np.zeros(n_t)
    ee_flt = np.zeros(n_t)
    ee_tot = np.zeros(n_t)
    epe = np.zeros(n_t)
    ene = np.zeros(n_t)

    for i, t in enumerate(snap.exposure_model_times):
        pv_fix, pv_flt = _dual_curve_components(model, snap.p0_disc, snap.p0_fwd, snap.legs, float(t), x[i, :])
        pv = pv_fix + pv_flt
        ee_fix[i] = float(np.mean(pv_fix))
        ee_flt[i] = float(np.mean(pv_flt))
        ee_tot[i] = float(np.mean(pv))
        epe[i] = float(np.mean(np.maximum(pv, 0.0)))
        ene[i] = float(np.mean(np.maximum(-pv, 0.0)))

    out_csv = args.ore_xml.resolve().parents[1] / "Output" / "leg_bias_diagnostic.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    hdr = "Date,Time,ORE_EPE,PY_EPE,EPE_Diff,ORE_ENE,PY_ENE,ENE_Diff,PY_EE_Total,PY_EE_FixedLeg,PY_EE_FloatLeg\n"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(hdr)
        for i in range(n_t):
            f.write(
                f"{snap.exposure_dates[i]},{snap.exposure_times[i]:.6f},{snap.ore_epe[i]:.6f},{epe[i]:.6f},{(epe[i]-snap.ore_epe[i]):.6f},"
                f"{snap.ore_ene[i]:.6f},{ene[i]:.6f},{(ene[i]-snap.ore_ene[i]):.6f},{ee_tot[i]:.6f},{ee_fix[i]:.6f},{ee_flt[i]:.6f}\n"
            )

    epe_gap = epe - snap.ore_epe
    ene_gap = ene - snap.ore_ene
    print(f"Wrote: {out_csv}")
    print(f"Mean(EPE gap PY-ORE): {np.mean(epe_gap):.2f}")
    print(f"Mean(ENE gap PY-ORE): {np.mean(ene_gap):.2f}")
    print(f"Mean PY EE fixed leg: {np.mean(ee_fix):.2f}")
    print(f"Mean PY EE float leg: {np.mean(ee_flt):.2f}")
    print(f"At t0: EE fixed={ee_fix[0]:.2f}, EE float={ee_flt[0]:.2f}, EE total={ee_tot[0]:.2f}")


if __name__ == "__main__":
    main()
