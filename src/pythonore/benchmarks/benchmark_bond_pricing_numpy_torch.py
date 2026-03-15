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
from dataclasses import replace
from datetime import date
from datetime import timedelta
from pathlib import Path
from time import perf_counter
import sys
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

REPO_ROOT = Path(__file__).resolve().parents[3]

from pythonore.compute.bond_pricing import (
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
    prepare_bond_scenario_grid_batch_torch,
    prepare_bond_scenario_grid_torch,
    prepare_bond_trade_batch_torch,
    prepare_bond_trade_torch,
    price_bond_scenarios_numpy,
    price_bond_scenarios_torch,
    price_bond_scenarios_torch_batched_preloaded,
    price_bond_scenarios_torch_preloaded,
    torch,
)
from pythonore.compute.irs_xva_utils import load_ore_default_curve_inputs


EX18_IN = REPO_ROOT / "Examples" / "Legacy" / "Example_18" / "Input"
EX18_OUT = REPO_ROOT / "Examples" / "Legacy" / "Example_18" / "ExpectedOutput"
SHARED_IN = REPO_ROOT / "Examples" / "Input"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", type=int, default=[2000, 5000])
    p.add_argument("--bonds", nargs="+", type=int, default=[100, 500], help="Repeat same bond this many times.")
    p.add_argument(
        "--scalar-max-work",
        type=int,
        default=1000,
        help="Skip scalar-loop baseline when n_bonds * n_scenarios exceeds this threshold (0 disables scalar baseline).",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", nargs="+", default=["cpu", "gpu"])
    p.add_argument("--trades", nargs="+", default=["Bond_Fixed", "FwdBond_Fixed"])
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON in addition to the table report.")
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
    # Baseline only: call canonical NumPy scenario kernel one scenario at a time.
    out = np.empty(grid.n_scenarios(), dtype=float)
    for i in range(grid.n_scenarios()):
        out[i] = float(price_bond_scenarios_numpy(compiled, _grid_row(grid, i))[0])
    return out


def _torch_run(compiled, grid: BondScenarioGrid, device: str, return_numpy: bool):
    with torch.inference_mode():
        out = price_bond_scenarios_torch(compiled, grid, device=device)
        if return_numpy:
            return out.detach().cpu().numpy()
        return out


def _torch_run_preloaded(compiled, prepared_trade, prepared_grid, return_numpy: bool):
    with torch.inference_mode():
        pv = price_bond_scenarios_torch_preloaded(compiled, prepared_trade, prepared_grid)
        if return_numpy:
            return pv.detach().cpu().numpy()
        return pv


def _torch_run_batched_preloaded(prepared_trade_batch, prepared_grid_batch, return_numpy: bool):
    with torch.inference_mode():
        pv = price_bond_scenarios_torch_batched_preloaded(prepared_trade_batch, prepared_grid_batch)
        if return_numpy:
            return pv.detach().cpu().numpy()
        return pv


def _repeat_same_bond(fn, n_bonds: int):
    out = None
    for _ in range(int(n_bonds)):
        out = fn()
    return out


def _fmt_sec(value: float) -> str:
    return f"{value:.4f}s"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _fmt_sci(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3e}"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _make_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([line, head, line, *body, line])


def _render_report(*, rows, devices, scenarios, bonds, trades, repeats, warmup, seed):
    lines = []
    lines.append("Bond Pricing NumPy vs Torch Benchmark Report")
    lines.append("=" * 42)
    lines.append("Configuration:")
    lines.append(f"  - Trades tested  : {', '.join(trades)}")
    lines.append(f"  - Scenarios      : {', '.join(_fmt_int(int(n)) for n in scenarios)}")
    lines.append(f"  - Bonds repeated : {', '.join(_fmt_int(int(n)) for n in bonds)}")
    lines.append(f"  - Repeats/warmup : {repeats}/{warmup}")
    lines.append(f"  - Seed           : {seed}")
    lines.append(f"  - Devices used   : {', '.join(devices) if devices else 'cpu only'}")
    lines.append("")
    lines.append("How to read:")
    lines.append("  - mean/min/std are runtimes over repeats.")
    lines.append("  - numpy/vectorized uses the host NumPy scenario kernel.")
    lines.append("  - torch/cpu and torch/mps reuse preloaded tensors on the target device.")
    lines.append("  - speedup_vs_scalar compares against one-scenario-at-a-time NumPy baseline when available.")
    lines.append("  - scalar baseline may be skipped automatically for large n_bonds*n_scenarios workloads.")
    lines.append("  - speedup_vs_numpy compares against numpy/vectorized.")
    lines.append("  - parity metrics are relative to NumPy outputs for that scenario.")
    lines.append("")

    grouped = {}
    for r in rows:
        grouped.setdefault((str(r["trade_id"]), int(r["n_scenarios"]), int(r["n_bonds"])), []).append(r)

    for trade_id, n_scenarios, n_bonds in sorted(grouped.keys(), key=lambda x: (x[0], x[2], x[1])):
        group = grouped[(trade_id, n_scenarios, n_bonds)]
        table_rows = []
        for r in sorted(group, key=lambda item: str(item["mode"])):
            table_rows.append(
                (
                    str(r["mode"]),
                    _fmt_sec(float(r["mean_sec"])),
                    _fmt_sec(float(r["min_sec"])),
                    _fmt_sec(float(r["std_sec"])),
                    _fmt_speedup(r.get("speedup_vs_scalar_mean")),
                    _fmt_speedup(r.get("speedup_vs_numpy_mean")),
                    _fmt_sci(r.get("parity_max_abs")),
                    _fmt_sci(r.get("parity_rmse")),
                    _fmt_sci(r.get("parity_max_abs_rel_to_refmax")),
                )
            )
        lines.append(
            f"Results for trade={trade_id}, n_bonds={_fmt_int(n_bonds)}, n_scenarios={_fmt_int(n_scenarios)}"
        )
        lines.append(
            _make_table(
                [
                    "mode",
                    "mean",
                    "min",
                    "std",
                    "speedup_vs_scalar",
                    "speedup_vs_numpy",
                    "parity_max_abs",
                    "parity_rmse",
                    "parity_max_abs_rel",
                ],
                table_rows,
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    args = _parse_args(argv)
    devices = _torch_devices(args.devices)
    rows = []
    for trade_id in args.trades:
        compiled, base_grid = _single_grid_from_example18(trade_id)
        for n_bonds in args.bonds:
            for n_scenarios in args.scenarios:
                grid = _expand_grid(compiled, base_grid, n_scenarios, args.seed + n_scenarios)
                work = int(n_bonds) * int(n_scenarios)
                scalar_enabled = args.scalar_max_work > 0 and work <= int(args.scalar_max_work)
                scalar_ref = _scalar_loop(compiled, grid) if scalar_enabled else None
                scalar_mean = None
                if scalar_enabled:
                    mean_sec, min_sec, std_sec = _bench(
                        lambda: _repeat_same_bond(lambda: _scalar_loop(compiled, grid), n_bonds),
                        args.repeats,
                        args.warmup,
                    )
                    scalar_mean = mean_sec
                    rows.append(
                        {
                            "trade_id": trade_id,
                            "trade_type": compiled.trade_type,
                            "mode": "numpy/scalar_loop",
                            "n_bonds": int(n_bonds),
                            "n_scenarios": n_scenarios,
                            "mean_sec": mean_sec,
                            "min_sec": min_sec,
                            "std_sec": std_sec,
                            "scalar_skipped": False,
                        }
                    )
                np_out = price_bond_scenarios_numpy(compiled, grid)
                mean_sec, min_sec, std_sec = _bench(
                    lambda: _repeat_same_bond(lambda: price_bond_scenarios_numpy(compiled, grid), n_bonds),
                    args.repeats,
                    args.warmup,
                )
                numpy_mean = mean_sec
                row = {
                    "trade_id": trade_id,
                    "trade_type": compiled.trade_type,
                    "mode": "numpy/vectorized",
                    "n_bonds": int(n_bonds),
                    "n_scenarios": n_scenarios,
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                    "speedup_vs_scalar_mean": (scalar_mean / mean_sec) if (scalar_mean is not None and mean_sec > 0.0) else None,
                    "scalar_skipped": not scalar_enabled,
                }
                if scalar_ref is not None:
                    row.update(_error_metrics(scalar_ref, np_out))
                rows.append(row)
                if torch is None:
                    continue
                for device in devices:
                    if int(n_bonds) == 1:
                        prepared_trade = prepare_bond_trade_torch(compiled, device=device)
                        prepared_grid = prepare_bond_scenario_grid_torch(grid, device=device)
                    else:
                        prepared_trade_batch = prepare_bond_trade_batch_torch(compiled, device=device, repeat=int(n_bonds))
                        prepared_grid_batch = prepare_bond_scenario_grid_batch_torch(grid, device=device, repeat=int(n_bonds))
                    if device == "cpu":
                        if int(n_bonds) == 1:
                            th_out = _torch_run_preloaded(compiled, prepared_trade, prepared_grid, True)
                            mean_sec, min_sec, std_sec = _bench(
                                lambda pt=prepared_trade, pg=prepared_grid: _torch_run_preloaded(compiled, pt, pg, True),
                                args.repeats,
                                args.warmup,
                            )
                        else:
                            th_out = _torch_run_batched_preloaded(prepared_trade_batch, prepared_grid_batch, True)
                            mean_sec, min_sec, std_sec = _bench(
                                lambda pt=prepared_trade_batch, pg=prepared_grid_batch: _torch_run_batched_preloaded(pt, pg, True),
                                args.repeats,
                                args.warmup,
                            )
                        row = {
                            "trade_id": trade_id,
                            "trade_type": compiled.trade_type,
                            "mode": "torch/cpu",
                            "n_bonds": int(n_bonds),
                            "n_scenarios": n_scenarios,
                            "mean_sec": mean_sec,
                            "min_sec": min_sec,
                            "std_sec": std_sec,
                            "speedup_vs_scalar_mean": (scalar_mean / mean_sec) if (scalar_mean is not None and mean_sec > 0.0) else None,
                            "speedup_vs_numpy_mean": numpy_mean / mean_sec if mean_sec > 0.0 else float("inf"),
                            "scalar_skipped": not scalar_enabled,
                        }
                        ref = np_out if int(n_bonds) == 1 else np.broadcast_to(np_out, th_out.shape)
                        row.update(_error_metrics(ref, th_out))
                        rows.append(row)
                    elif device == "mps":
                        if int(n_bonds) == 1:
                            th_dev = _torch_run_preloaded(compiled, prepared_trade, prepared_grid, False)
                            mean_sec, min_sec, std_sec = _bench(
                                lambda pt=prepared_trade, pg=prepared_grid: _torch_run_preloaded(compiled, pt, pg, False),
                                args.repeats,
                                args.warmup,
                            )
                        else:
                            th_dev = _torch_run_batched_preloaded(prepared_trade_batch, prepared_grid_batch, False)
                            mean_sec, min_sec, std_sec = _bench(
                                lambda pt=prepared_trade_batch, pg=prepared_grid_batch: _torch_run_batched_preloaded(pt, pg, False),
                                args.repeats,
                                args.warmup,
                            )
                        row = {
                            "trade_id": trade_id,
                            "trade_type": compiled.trade_type,
                            "mode": "torch/mps",
                            "n_bonds": int(n_bonds),
                            "n_scenarios": n_scenarios,
                            "mean_sec": mean_sec,
                            "min_sec": min_sec,
                            "std_sec": std_sec,
                            "speedup_vs_scalar_mean": (scalar_mean / mean_sec) if (scalar_mean is not None and mean_sec > 0.0) else None,
                            "speedup_vs_numpy_mean": numpy_mean / mean_sec if mean_sec > 0.0 else float("inf"),
                            "scalar_skipped": not scalar_enabled,
                        }
                        got = th_dev.detach().cpu().numpy()
                        ref = np_out if int(n_bonds) == 1 else np.broadcast_to(np_out, got.shape)
                        row.update(_error_metrics(ref, got))
                        rows.append(row)
    payload = {
        "metadata": {
            "seed": int(args.seed),
            "scenarios": [int(n) for n in args.scenarios],
            "bonds": [int(n) for n in args.bonds],
            "scalar_max_work": int(args.scalar_max_work),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "trades": list(args.trades),
            "devices": devices,
        },
        "results": rows,
    }
    print(
        _render_report(
            rows=rows,
            devices=devices,
            scenarios=args.scenarios,
            bonds=args.bonds,
            trades=args.trades,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
        )
    )
    if args.json:
        print("Machine-readable JSON:")
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
