#!/usr/bin/env python3
"""Benchmark callable-bond pricing kernels on NumPy vs torch.

This benchmark uses the real callable-bond example fixtures and compares the
pricing kernels directly after the scenario pack has already been built.
Pack/model construction is intentionally excluded from the timed section.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

REPO_ROOT = Path(__file__).resolve().parents[3]

from pythonore.compute.bond_pricing import (
    _build_lgm_model_for_callable,
    _fit_curve_for_currency,
    _load_callable_option_curve_from_reference_output,
    _load_security_spread,
    build_callable_bond_scenario_pack,
    compile_callable_bond_trade,
    load_callable_bond_trade_spec,
    price_callable_bond_scenarios_numpy,
    price_callable_bond_scenarios_torch,
    price_callable_bond_scenarios_torch_batched,
    torch,
)
from pythonore.compute.irs_xva_utils import load_ore_default_curve_inputs


CALLABLE_IN = REPO_ROOT / "Examples" / "Exposure" / "Input"
SHARED_IN = REPO_ROOT / "Examples" / "Input"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", type=int, default=[10, 50])
    p.add_argument("--bonds", nargs="+", type=int, default=[1, 8], help="Repeat same callable bond this many times.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--devices", nargs="+", default=["cpu", "gpu"])
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


def _callable_fixture(n_scenarios: int):
    ore_xml = CALLABLE_IN / "ore_callable_bond.xml"
    portfolio_xml = CALLABLE_IN / "portfolio_callablebond.xml"
    reference_xml = CALLABLE_IN / "reference_data_callablebond.xml"
    pe_xml = CALLABLE_IN / "pricingengine_callablebond.xml"
    todaysmarket_xml = SHARED_IN / "todaysmarket.xml"
    market_data_file = SHARED_IN / "market_20160205.txt"

    spec, engine = load_callable_bond_trade_spec(
        portfolio_xml=portfolio_xml,
        trade_id="CallableBondTrade",
        reference_data_path=reference_xml,
        pricingengine_path=pe_xml,
    )
    compiled = compile_callable_bond_trade(spec, engine, asof_date="2016-02-05", day_counter="A365F")
    model = _build_lgm_model_for_callable(
        ore_xml=ore_xml,
        pricingengine_path=pe_xml,
        todaysmarket_xml=todaysmarket_xml,
        market_data_file=market_data_file,
        currency=spec.currency,
        maturity_date=max(cf.pay_date for cf in spec.bond.cashflows),
        asof_date=date(2016, 2, 5),
    )
    native_curve = _load_callable_option_curve_from_reference_output(
        ore_xml,
        todaysmarket_xml=todaysmarket_xml,
        curve_id=spec.reference_curve_id,
        asof_date="2016-02-05",
        day_counter="A365F",
    )
    rollback_curve = native_curve[0] if native_curve is not None else _fit_curve_for_currency(ore_xml, spec.currency)
    stripped_curve = _fit_curve_for_currency(ore_xml, spec.currency)
    credit = load_ore_default_curve_inputs(
        str(todaysmarket_xml),
        str(market_data_file),
        cpty_name=spec.credit_curve_id,
    )
    security_spread = _load_security_spread(market_data_file, spec.security_id)
    pack = build_callable_bond_scenario_pack(
        compiled,
        reference_curves=[rollback_curve] * int(n_scenarios),
        models=[model] * int(n_scenarios),
        hazard_times=np.asarray(credit["hazard_times"], dtype=float),
        hazard_rates=np.repeat(np.asarray(credit["hazard_rates"], dtype=float).reshape(1, -1), int(n_scenarios), axis=0),
        recovery_rate=np.full(int(n_scenarios), float(credit["recovery"]), dtype=float),
        security_spread=np.full(int(n_scenarios), security_spread, dtype=float),
        stripped_discount_curves=[stripped_curve] * int(n_scenarios),
        stripped_income_curves=[stripped_curve] * int(n_scenarios),
    )
    return compiled, pack


def _fmt_sec(value: float) -> str:
    return f"{value:.4f}s"


def _fmt_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _fmt_sci(value: float | None) -> str:
    return "-" if value is None else f"{value:.3e}"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _make_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([line, head, line, *body, line])


def _render_report(*, rows, devices, scenarios, bonds, repeats, warmup, chunk_size):
    lines = []
    lines.append("Callable Bond Pricing NumPy vs Torch Benchmark Report")
    lines.append("=" * 50)
    lines.append("Configuration:")
    lines.append(f"  - Scenarios      : {', '.join(_fmt_int(int(n)) for n in scenarios)}")
    lines.append(f"  - Bonds repeated : {', '.join(_fmt_int(int(n)) for n in bonds)}")
    lines.append(f"  - Repeats/warmup : {repeats}/{warmup}")
    lines.append(f"  - Chunk size     : {chunk_size}")
    lines.append(f"  - Devices used   : {', '.join(devices) if devices else 'cpu only'}")
    lines.append("")
    lines.append("How to read:")
    lines.append("  - Timings exclude callable scenario-pack/model construction.")
    lines.append("  - numpy loops over repeated same-bond requests.")
    lines.append("  - torch/cpu uses the callable torch kernel directly; for n_bonds>1 it uses the batched callable wrapper.")
    lines.append("  - torch/mps does the same on Apple MPS when available.")
    lines.append("")

    grouped = {}
    for r in rows:
        grouped.setdefault((int(r["n_bonds"]), int(r["n_scenarios"])), []).append(r)
    for n_bonds, n_scenarios in sorted(grouped.keys()):
        table_rows = []
        for r in sorted(grouped[(n_bonds, n_scenarios)], key=lambda item: str(item["mode"])):
            table_rows.append(
                (
                    str(r["mode"]),
                    _fmt_sec(float(r["mean_sec"])),
                    _fmt_sec(float(r["min_sec"])),
                    _fmt_sec(float(r["std_sec"])),
                    _fmt_speedup(r.get("speedup_vs_numpy_mean")),
                    _fmt_sci(r.get("parity_max_abs")),
                    _fmt_sci(r.get("parity_rmse")),
                    _fmt_sci(r.get("parity_max_abs_rel_to_refmax")),
                )
            )
        lines.append(f"Results for n_bonds={_fmt_int(n_bonds)}, n_scenarios={_fmt_int(n_scenarios)}")
        lines.append(
            _make_table(
                ["mode", "mean", "min", "std", "speedup_vs_numpy", "parity_max_abs", "parity_rmse", "parity_max_abs_rel"],
                table_rows,
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    args = _parse_args(argv)
    devices = _torch_devices(args.devices)
    rows = []
    for n_scenarios in args.scenarios:
        compiled, pack = _callable_fixture(int(n_scenarios))
        np_single = price_callable_bond_scenarios_numpy(compiled, pack, chunk_size=args.chunk_size)
        for n_bonds in args.bonds:
            n_bonds = int(n_bonds)
            mean_sec, min_sec, std_sec = _bench(
                lambda nb=n_bonds: np.stack(
                    [price_callable_bond_scenarios_numpy(compiled, pack, chunk_size=args.chunk_size) for _ in range(nb)],
                    axis=0,
                ),
                args.repeats,
                args.warmup,
            )
            np_ref = np_single if n_bonds == 1 else np.broadcast_to(np_single, (n_bonds, np_single.shape[0]))
            rows.append(
                {
                    "mode": "numpy",
                    "n_bonds": n_bonds,
                    "n_scenarios": int(n_scenarios),
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                }
            )
            numpy_mean = mean_sec
            if torch is None:
                continue
            for device in devices:
                if n_bonds == 1:
                    th_out = price_callable_bond_scenarios_torch(compiled, pack, device=device, chunk_size=args.chunk_size)
                    bench_fn = lambda d=device: price_callable_bond_scenarios_torch(compiled, pack, device=d, chunk_size=args.chunk_size)
                else:
                    th_out = price_callable_bond_scenarios_torch_batched(
                        compiled,
                        pack,
                        device=device,
                        chunk_size=args.chunk_size,
                        repeat=n_bonds,
                    )
                    bench_fn = lambda d=device, nb=n_bonds: price_callable_bond_scenarios_torch_batched(
                        compiled,
                        pack,
                        device=d,
                        chunk_size=args.chunk_size,
                        repeat=nb,
                    )
                if device == "mps":
                    torch.mps.synchronize()
                mean_sec, min_sec, std_sec = _bench(bench_fn, args.repeats, args.warmup)
                got = th_out.detach().cpu().numpy()
                rows.append(
                    {
                        "mode": f"torch/{device}",
                        "n_bonds": n_bonds,
                        "n_scenarios": int(n_scenarios),
                        "mean_sec": mean_sec,
                        "min_sec": min_sec,
                        "std_sec": std_sec,
                        "speedup_vs_numpy_mean": numpy_mean / mean_sec if mean_sec > 0.0 else float("inf"),
                        **_error_metrics(np_ref, got),
                    }
                )

    payload = {
        "metadata": {
            "scenarios": [int(x) for x in args.scenarios],
            "bonds": [int(x) for x in args.bonds],
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "chunk_size": int(args.chunk_size),
            "devices": devices,
        },
        "results": rows,
    }
    print(_render_report(rows=rows, devices=devices, scenarios=args.scenarios, bonds=args.bonds, repeats=args.repeats, warmup=args.warmup, chunk_size=args.chunk_size))
    if args.json:
        print("Machine-readable JSON:")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
