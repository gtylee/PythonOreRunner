#!/usr/bin/env python3
"""Benchmark vectorized bond pricing kernels on NumPy vs torch.

This benchmark uses the Example_18 bond fixtures to build realistic compiled
bond trades and a base one-scenario grid, then fans that out into synthetic
multi-scenario tensors. The goal is to time the pricing kernels directly:

- scalar loop baseline using the single-scenario NumPy path
- vectorized NumPy kernel
- vectorized torch kernel on CPU / MPS when available
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_ore_tools.bond_pricing import (  # noqa: E402
    BondScenarioGrid,
    _accrued_amount,
    _bond_npv,
    _curve_from_flow_discounts,
    _load_security_spread,
    _time_from_dates,
    _adjust_date,
    build_bond_scenario_grid_numpy,
    compile_bond_trade,
    load_bond_trade_spec,
    price_bond_scenarios_numpy,
    price_bond_scenarios_torch,
    price_bond_single_numpy,
    torch,
)
from py_ore_tools.irs_xva_utils import load_ore_default_curve_inputs  # noqa: E402


EX18_IN = REPO_ROOT / "Examples" / "Legacy" / "Example_18" / "Input"
EX18_OUT = REPO_ROOT / "Examples" / "Legacy" / "Example_18" / "ExpectedOutput"
SHARED_IN = REPO_ROOT / "Examples" / "Input"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", type=int, default=[1000, 10000, 50000])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", nargs="+", default=["cpu", "gpu"])
    p.add_argument("--trades", nargs="+", default=["Bond_Fixed", "FwdBond_Fixed"])
    return p.parse_args(argv)


def _torch_devices(requested):
    out = []
    has_mps = bool(torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    for item in requested:
        if item == "cpu":
            out.append("cpu")
        elif item == "gpu" and has_mps:
            out.append("mps")
        elif item == "mps" and has_mps:
            out.append("mps")
    return list(dict.fromkeys(out))


def _bench(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        vals.append(perf_counter() - t0)
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.std(ddof=0))


def _error_metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    ref64 = np.asarray(ref, dtype=np.float64)
    got64 = np.asarray(got, dtype=np.float64)
    diff = got64 - ref64
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_scale = max(float(np.max(np.abs(ref64))), 1.0e-12)
    return {
        "parity_max_abs": max_abs,
        "parity_rmse": rmse,
        "parity_max_abs_rel_to_refmax": max_abs / ref_scale,
    }


def _single_grid_from_example18(trade_id: str):
    asof = "2016-02-05"
    spec, engine_spec = load_bond_trade_spec(
        portfolio_xml=EX18_IN / "portfolio.xml",
        trade_id=trade_id,
        reference_data_path=EX18_IN / "referencedata.xml",
        pricingengine_path=SHARED_IN / "pricingengine.xml",
        flows_csv=EX18_OUT / "flows.csv",
    )
    try:
        credit = load_ore_default_curve_inputs(
            str(SHARED_IN / "todaysmarket.xml"),
            str(SHARED_IN / "market_20160205_flat.txt"),
            cpty_name=spec.credit_curve_id,
        )
        hazard_times = np.asarray(credit["hazard_times"], dtype=float)
        hazard_rates = np.asarray(credit["hazard_rates"], dtype=float)
        recovery_rate = float(credit["recovery"])
    except Exception:
        hazard_times = np.asarray([50.0], dtype=float)
        hazard_rates = np.asarray([0.0], dtype=float)
        recovery_rate = 0.0
    security_spread = _load_security_spread(SHARED_IN / "market_20160205_flat.txt", spec.security_id)
    curve = _curve_from_flow_discounts(
        EX18_OUT / "flows.csv",
        trade_id,
        date.fromisoformat(asof),
        "A365F",
        forward_underlying_only=spec.trade_type == "ForwardBond",
    )
    compiled = compile_bond_trade(spec, asof_date=asof, day_counter="A365F", engine_spec=engine_spec)
    underlying_compiled = compiled if compiled.trade_type == "Bond" else replace(compiled, trade_type="Bond")
    base_grid = build_bond_scenario_grid_numpy(
        underlying_compiled,
        discount_curve=curve,
        income_curve=curve,
        hazard_times=hazard_times,
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
        security_spread=security_spread,
        engine_spec=engine_spec,
    )
    if spec.trade_type != "ForwardBond":
        return compiled, base_grid

    asof_date = date.fromisoformat(asof)
    _, forward_value = _bond_npv(
        spec,
        asof_date=asof_date,
        day_counter="A365F",
        discount_curve=curve,
        income_curve=curve,
        hazard_times=hazard_times,
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
        security_spread=security_spread,
        engine_spec=engine_spec,
        npv_date=spec.forward_maturity_date,
        settlement_date=spec.forward_maturity_date,
        conditional_on_survival=True,
    )
    bond_settlement_date = _adjust_date(spec.forward_maturity_date + timedelta(days=spec.settlement_days), "F", spec.calendar)
    accrued = _accrued_amount(spec, bond_settlement_date)
    payoff_discount = float(curve(max(_time_from_dates(asof_date, bond_settlement_date, "A365F"), 0.0)))
    premium_discount = None
    if spec.compensation_payment and spec.compensation_payment_date and spec.compensation_payment_date > asof_date:
        premium_discount = float(curve(max(_time_from_dates(asof_date, spec.compensation_payment_date, "A365F"), 0.0)))
    forward_grid = BondScenarioGrid(
        discount_to_pay=base_grid.discount_to_pay,
        income_to_npv=base_grid.income_to_npv,
        income_to_settlement=base_grid.income_to_settlement,
        survival_to_pay=base_grid.survival_to_pay,
        recovery_discount_mid=base_grid.recovery_discount_mid,
        recovery_default_prob=base_grid.recovery_default_prob,
        recovery_rate=base_grid.recovery_rate,
        forward_dirty_value=np.asarray([forward_value], dtype=float),
        accrued_at_bond_settlement=np.asarray([accrued], dtype=float),
        payoff_discount=np.asarray([payoff_discount], dtype=float),
        premium_discount=None if premium_discount is None else np.asarray([premium_discount], dtype=float),
    )
    return compiled, forward_grid


def _expand_grid(compiled, base_grid: BondScenarioGrid, n_scenarios: int, seed: int) -> BondScenarioGrid:
    rng = np.random.default_rng(seed)
    curve_shift = rng.normal(0.0, 0.0025, size=n_scenarios)
    hazard_shift = np.abs(rng.normal(0.0, 0.02, size=n_scenarios))
    rec_shift = rng.normal(0.0, 0.03, size=n_scenarios)
    pay_t = compiled.pay_times[None, :]
    rec_mid_t = (compiled.recovery_start_times + compiled.recovery_end_times) * 0.5
    rec_mid_t = rec_mid_t[None, :] if rec_mid_t.size else np.zeros((1, 0), dtype=float)

    base_discount = np.asarray(base_grid.discount_to_pay, dtype=float)
    base_survival = np.asarray(base_grid.survival_to_pay, dtype=float)
    base_rec_disc = np.asarray(base_grid.recovery_discount_mid, dtype=float)
    base_rec_prob = np.asarray(base_grid.recovery_default_prob, dtype=float)

    discount_to_pay = np.repeat(base_discount, n_scenarios, axis=0) * np.exp(-curve_shift[:, None] * pay_t)
    survival_to_pay = np.clip(np.repeat(base_survival, n_scenarios, axis=0) * np.exp(-hazard_shift[:, None] * pay_t), 0.0, 1.0)
    recovery_discount_mid = (
        np.repeat(base_rec_disc, n_scenarios, axis=0) * np.exp(-curve_shift[:, None] * rec_mid_t) if base_rec_disc.size else np.zeros((n_scenarios, 0), dtype=float)
    )
    recovery_default_prob = (
        np.clip(np.repeat(base_rec_prob, n_scenarios, axis=0) * (1.0 + hazard_shift[:, None]), 0.0, 1.0)
        if base_rec_prob.size
        else np.zeros((n_scenarios, 0), dtype=float)
    )
    recovery_rate = np.clip(float(base_grid.recovery_rate[0]) + rec_shift, 0.0, 0.95)

    forward_dirty_value = None
    accrued = None
    payoff_discount = None
    premium_discount = None
    if compiled.trade_type == "ForwardBond":
        level = rng.normal(0.0, 25000.0, size=n_scenarios)
        forward_dirty_value = np.repeat(np.asarray(base_grid.forward_dirty_value, dtype=float), n_scenarios, axis=0) + level
        accrued = np.repeat(np.asarray(base_grid.accrued_at_bond_settlement, dtype=float), n_scenarios, axis=0)
        payoff_discount = np.clip(np.repeat(np.asarray(base_grid.payoff_discount, dtype=float), n_scenarios, axis=0) * np.exp(-0.25 * curve_shift), 0.0, None)
        if base_grid.premium_discount is not None:
            premium_discount = np.clip(np.repeat(np.asarray(base_grid.premium_discount, dtype=float), n_scenarios, axis=0) * np.exp(-0.25 * curve_shift), 0.0, None)

    return BondScenarioGrid(
        discount_to_pay=discount_to_pay,
        income_to_npv=np.repeat(np.asarray(base_grid.income_to_npv, dtype=float), n_scenarios, axis=0),
        income_to_settlement=np.repeat(np.asarray(base_grid.income_to_settlement, dtype=float), n_scenarios, axis=0),
        survival_to_pay=survival_to_pay,
        recovery_discount_mid=recovery_discount_mid,
        recovery_default_prob=recovery_default_prob,
        recovery_rate=recovery_rate,
        forward_dirty_value=forward_dirty_value,
        accrued_at_bond_settlement=accrued,
        payoff_discount=payoff_discount,
        premium_discount=premium_discount,
    )


def _grid_row(grid: BondScenarioGrid, i: int) -> BondScenarioGrid:
    def _slice(value):
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 1:
            return arr[i : i + 1]
        return arr[i : i + 1, :]

    return BondScenarioGrid(
        discount_to_pay=_slice(grid.discount_to_pay),
        income_to_npv=_slice(grid.income_to_npv),
        income_to_settlement=_slice(grid.income_to_settlement),
        survival_to_pay=_slice(grid.survival_to_pay),
        recovery_discount_mid=_slice(grid.recovery_discount_mid),
        recovery_default_prob=_slice(grid.recovery_default_prob),
        recovery_rate=_slice(grid.recovery_rate),
        forward_dirty_value=_slice(grid.forward_dirty_value),
        accrued_at_bond_settlement=_slice(grid.accrued_at_bond_settlement),
        payoff_discount=_slice(grid.payoff_discount),
        premium_discount=_slice(grid.premium_discount),
    )


def _scalar_loop(compiled, grid: BondScenarioGrid) -> np.ndarray:
    out = np.empty(grid.n_scenarios(), dtype=float)
    for i in range(grid.n_scenarios()):
        out[i] = price_bond_single_numpy(compiled, _grid_row(grid, i))
    return out


def _torch_run(compiled, grid: BondScenarioGrid, device: str, return_numpy: bool):
    with torch.inference_mode():
        out = price_bond_scenarios_torch(compiled, grid, device=device)
        if return_numpy:
            return out.detach().cpu().numpy()
        return out


def main(argv=None):
    args = _parse_args(argv)
    devices = _torch_devices(args.devices)
    rows = []
    for trade_id in args.trades:
        compiled, base_grid = _single_grid_from_example18(trade_id)
        for n_scenarios in args.scenarios:
            grid = _expand_grid(compiled, base_grid, n_scenarios, args.seed + n_scenarios)
            scalar_ref = _scalar_loop(compiled, grid)
            mean_sec, min_sec, std_sec = _bench(lambda: _scalar_loop(compiled, grid), args.repeats, args.warmup)
            rows.append(
                {
                    "trade_id": trade_id,
                    "trade_type": compiled.trade_type,
                    "mode": "numpy/scalar_loop",
                    "n_scenarios": n_scenarios,
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                }
            )
            np_out = price_bond_scenarios_numpy(compiled, grid)
            mean_sec, min_sec, std_sec = _bench(lambda: price_bond_scenarios_numpy(compiled, grid), args.repeats, args.warmup)
            row = {
                "trade_id": trade_id,
                "trade_type": compiled.trade_type,
                "mode": "numpy/vectorized",
                "n_scenarios": n_scenarios,
                "mean_sec": mean_sec,
                "min_sec": min_sec,
                "std_sec": std_sec,
                "speedup_vs_scalar_mean": rows[-1]["mean_sec"] / mean_sec if mean_sec > 0.0 else float("inf"),
            }
            row.update(_error_metrics(scalar_ref, np_out))
            rows.append(row)
            if torch is None:
                continue
            for device in devices:
                if device == "cpu":
                    th_out = _torch_run(compiled, grid, device, True)
                    mean_sec, min_sec, std_sec = _bench(lambda d=device: _torch_run(compiled, grid, d, True), args.repeats, args.warmup)
                    row = {
                        "trade_id": trade_id,
                        "trade_type": compiled.trade_type,
                        "mode": "torch/cpu/vectorized",
                        "n_scenarios": n_scenarios,
                        "mean_sec": mean_sec,
                        "min_sec": min_sec,
                        "std_sec": std_sec,
                        "speedup_vs_scalar_mean": rows[-2]["mean_sec"] / mean_sec if mean_sec > 0.0 else float("inf"),
                        "speedup_vs_numpy_mean": rows[-1]["mean_sec"] / mean_sec if mean_sec > 0.0 else float("inf"),
                    }
                    row.update(_error_metrics(np_out, th_out))
                    rows.append(row)
                elif device == "mps":
                    th_dev = _torch_run(compiled, grid, device, False)
                    mean_sec, min_sec, std_sec = _bench(lambda d=device: _torch_run(compiled, grid, d, False), args.repeats, args.warmup)
                    row = {
                        "trade_id": trade_id,
                        "trade_type": compiled.trade_type,
                        "mode": "torch/mps/vectorized_device_only",
                        "n_scenarios": n_scenarios,
                        "mean_sec": mean_sec,
                        "min_sec": min_sec,
                        "std_sec": std_sec,
                    }
                    row.update(_error_metrics(np_out, th_dev.detach().cpu().numpy()))
                    rows.append(row)
                    th_host = _torch_run(compiled, grid, device, True)
                    mean_sec, min_sec, std_sec = _bench(lambda d=device: _torch_run(compiled, grid, d, True), args.repeats, args.warmup)
                    row = {
                        "trade_id": trade_id,
                        "trade_type": compiled.trade_type,
                        "mode": "torch/mps/vectorized_host_copy",
                        "n_scenarios": n_scenarios,
                        "mean_sec": mean_sec,
                        "min_sec": min_sec,
                        "std_sec": std_sec,
                    }
                    row.update(_error_metrics(np_out, th_host))
                    rows.append(row)
    print(json.dumps({"devices": devices, "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
