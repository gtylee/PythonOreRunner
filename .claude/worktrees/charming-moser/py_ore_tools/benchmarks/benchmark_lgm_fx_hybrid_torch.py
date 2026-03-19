#!/usr/bin/env python3
"""Benchmark NumPy vs torch for hybrid IR/FX path simulation."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from py_ore_tools.lgm import LGMParams
from py_ore_tools.lgm_fx_hybrid import LgmFxHybrid, MultiCcyLgmParams
from py_ore_tools.lgm_fx_hybrid_torch import TorchLgmFxHybrid, simulate_hybrid_paths_torch


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
    print(json.dumps({"devices": devices, "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
