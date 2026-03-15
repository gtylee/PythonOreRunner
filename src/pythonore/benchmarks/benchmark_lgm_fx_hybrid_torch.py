#!/usr/bin/env python3
"""Benchmark NumPy vs torch for hybrid IR/FX path simulation."""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

from pythonore.compute.lgm import LGMParams
from pythonore.compute.lgm_fx_hybrid import LgmFxHybrid, MultiCcyLgmParams
from pythonore.compute.lgm_fx_hybrid_torch import TorchLgmFxHybrid, simulate_hybrid_paths_torch


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", type=int, default=[10000, 50000, 100000])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--horizon", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", nargs="+", default=["cpu", "gpu"])
    return p.parse_args(argv)


def _build_models():
    eur = LGMParams(alpha_times=(1.0,), alpha_values=(0.01, 0.01), kappa_times=(), kappa_values=(0.03,))
    usd = LGMParams(alpha_times=(1.0,), alpha_values=(0.012, 0.012), kappa_times=(), kappa_values=(0.02,))
    gbp = LGMParams(alpha_times=(1.0,), alpha_values=(0.011, 0.011), kappa_times=(), kappa_values=(0.025,))
    corr = np.array(
        [
            [1.0, 0.20, 0.10, 0.15, 0.05],
            [0.20, 1.0, 0.25, 0.10, 0.08],
            [0.10, 0.25, 1.0, 0.06, 0.12],
            [0.15, 0.10, 0.06, 1.0, 0.30],
            [0.05, 0.08, 0.12, 0.30, 1.0],
        ],
        dtype=float,
    )
    params = MultiCcyLgmParams(
        ir_params={"EUR": eur, "USD": usd, "GBP": gbp},
        fx_vols={"EUR/USD": (tuple(), (0.15,)), "GBP/USD": (tuple(), (0.13,))},
        corr=corr,
    )
    return LgmFxHybrid(params), TorchLgmFxHybrid(params)


def _expand_devices(requested):
    import torch
    out = []
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
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


def _fmt_sec(value: float) -> str:
    return f"{value:.4f}s"


def _fmt_sci(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3e}"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


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


def _render_report(rows, *, devices, paths, steps, horizon, repeats, warmup, seed):
    lines = []
    lines.append("LGM FX Hybrid Benchmark Report")
    lines.append("=" * 30)
    lines.append("Configuration:")
    lines.append(f"  - Paths tested : {', '.join(str(p) for p in paths)}")
    lines.append(f"  - Time grid    : {steps} steps over {horizon:.2f} years")
    lines.append(f"  - Repeats      : {repeats} (warmup={warmup})")
    lines.append(f"  - Seed         : {seed}")
    lines.append(f"  - Devices used : {', '.join(devices) if devices else 'cpu only'}")
    lines.append("")
    lines.append("How to read this report:")
    lines.append("  - mean/min/std are runtime statistics across repeats.")
    lines.append("  - speedup is measured against NumPy for the same path count.")
    lines.append("  - parity_max_abs is the max absolute difference vs NumPy reference.")
    lines.append("")

    rows_by_paths = {}
    for r in rows:
        rows_by_paths.setdefault(int(r["n_paths"]), []).append(r)

    for n_paths in sorted(rows_by_paths):
        group = rows_by_paths[n_paths]
        base = next((r for r in group if str(r["mode"]).startswith("numpy/")), None)
        base_mean = float(base["mean_sec"]) if base else None
        table_rows = []
        for r in sorted(group, key=lambda item: item["mode"]):
            mean_sec = float(r["mean_sec"])
            speedup = (base_mean / mean_sec) if base_mean is not None and mean_sec > 0.0 else None
            table_rows.append(
                (
                    str(r["mode"]),
                    _fmt_sec(mean_sec),
                    _fmt_sec(float(r["min_sec"])),
                    _fmt_sec(float(r["std_sec"])),
                    _fmt_speedup(speedup),
                    _fmt_sci(r.get("parity_max_abs")),
                )
            )

        lines.append(f"Results for n_paths = {n_paths}")
        lines.append(
            _make_table(
                ["mode", "mean", "min", "std", "speedup_vs_numpy", "parity_max_abs"],
                table_rows,
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    args = _parse_args(argv)
    np_model, torch_model = _build_models()
    times = np.linspace(0.0, float(args.horizon), int(args.steps) + 1, dtype=float)
    devices = _expand_devices(args.devices)
    rows = []

    for n_paths in args.paths:
        shared = np.random.default_rng(args.seed).standard_normal((times.size - 1, np_model.n_factors, n_paths))
        ref = np_model.simulate_paths(times, n_paths, rng=np.random.default_rng(args.seed), log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)})
        mean_sec, min_sec, std_sec = _bench(
            lambda n=n_paths: np_model.simulate_paths(times, n, rng=np.random.default_rng(args.seed), log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)}),
            args.repeats,
            args.warmup,
        )
        rows.append({"mode": "numpy/cpu/hybrid_sim", "n_paths": n_paths, "mean_sec": mean_sec, "min_sec": min_sec, "std_sec": std_sec})
        for device in devices:
            out = simulate_hybrid_paths_torch(
                torch_model,
                times,
                n_paths,
                normal_draws=shared,
                log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)},
                device=device,
                return_numpy=True,
            )
            mean_sec, min_sec, std_sec = _bench(
                lambda n=n_paths, d=device: simulate_hybrid_paths_torch(
                    torch_model,
                    times,
                    n,
                    rng=np.random.default_rng(args.seed),
                    log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)},
                    device=d,
                    return_numpy=True,
                ),
                args.repeats,
                args.warmup,
            )
            max_abs = 0.0
            max_abs = max(max_abs, float(np.max(np.abs(out["x"]["EUR"] - ref["x"]["EUR"]))))
            max_abs = max(max_abs, float(np.max(np.abs(out["x"]["USD"] - ref["x"]["USD"]))))
            max_abs = max(max_abs, float(np.max(np.abs(out["x"]["GBP"] - ref["x"]["GBP"]))))
            max_abs = max(max_abs, float(np.max(np.abs(out["s"]["EUR/USD"] - ref["s"]["EUR/USD"]))))
            max_abs = max(max_abs, float(np.max(np.abs(out["s"]["GBP/USD"] - ref["s"]["GBP/USD"]))))
            rows.append({"mode": f"torch/{device}/hybrid_sim", "n_paths": n_paths, "mean_sec": mean_sec, "min_sec": min_sec, "std_sec": std_sec, "parity_max_abs": max_abs})
    report = _render_report(
        rows,
        devices=devices,
        paths=args.paths,
        steps=args.steps,
        horizon=args.horizon,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
    )
    print(report)
    print("Machine-readable JSON:")
    print(json.dumps({"devices": devices, "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
