#!/usr/bin/env python3
"""Benchmark a batched multi-ccy FX forward portfolio in NumPy vs torch."""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

from pythonore.compute.irs_xva_utils import build_discount_curve_from_zero_rate_pairs, survival_probability_from_hazard
from pythonore.compute.lgm_fx_hybrid import MultiCcyLgmParams, LgmFxHybrid
from pythonore.compute.lgm_fx_hybrid_torch import TorchLgmFxHybrid, simulate_hybrid_paths_torch
from pythonore.compute.lgm_fx_xva_utils import FxForwardDef, aggregate_exposure_profile, cva_terms_from_profile, fx_forward_npv, fx_forward_portfolio_npv_paths_torch
from pythonore.compute.lgm import LGMParams


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", type=int, default=[10000, 50000, 100000])
    p.add_argument("--trades", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--devices", nargs="+", default=["cpu", "gpu"])
    return p.parse_args(argv)


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


def _build_hybrid():
    eur = LGMParams.constant(alpha=0.01, kappa=0.03)
    usd = LGMParams.constant(alpha=0.012, kappa=0.02)
    gbp = LGMParams.constant(alpha=0.011, kappa=0.025)
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
        fx_vols={"EUR/USD": (tuple(), (0.12,)), "GBP/USD": (tuple(), (0.11,))},
        corr=corr,
    )
    return LgmFxHybrid(params), TorchLgmFxHybrid(params)


def _curves():
    return {
        "USD": build_discount_curve_from_zero_rate_pairs([(0.0, 0.03), (5.0, 0.03)]),
        "EUR": build_discount_curve_from_zero_rate_pairs([(0.0, 0.02), (5.0, 0.02)]),
        "GBP": build_discount_curve_from_zero_rate_pairs([(0.0, 0.025), (5.0, 0.025)]),
    }


def _portfolio(n_trades):
    trades = []
    maturities = np.linspace(0.5, 2.5, n_trades)
    for i, mat in enumerate(maturities):
        if i % 2 == 0:
            trades.append(FxForwardDef(f"FX_{i}", "EUR/USD", 1_000_000 + 5000 * i, 1.10 + 0.0005 * i, float(mat)))
        else:
            trades.append(FxForwardDef(f"FX_{i}", "GBP/USD", 1_200_000 + 5000 * i, 1.28 + 0.0007 * i, float(mat)))
    return trades


def _times():
    return np.linspace(0.0, 2.5, 33, dtype=float)


def _numpy_pipeline(hybrid, trades, times, n_paths, seed):
    curves = _curves()
    sim = hybrid.simulate_paths(times, n_paths, rng=np.random.default_rng(seed), log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)}, rd_minus_rf={"EUR/USD": 0.01, "GBP/USD": 0.005})
    npv = np.zeros((times.size, n_paths), dtype=float)
    for fx in trades:
        base, quote = fx.pair.split("/")
        s = sim["s"][fx.pair]
        x_dom = sim["x"][quote]
        x_for = sim["x"][base]
        for i, t in enumerate(times):
            npv[i, :] += fx_forward_npv(hybrid, fx, float(t), s[i, :], x_dom[i, :], x_for[i, :], curves[quote], curves[base])
    exp = aggregate_exposure_profile(npv)
    df = np.asarray([curves["USD"](float(t)) for t in times], dtype=float)
    q = survival_probability_from_hazard(times, np.array([2.5]), np.array([0.015]))
    cva = float(cva_terms_from_profile(times, exp["epe"], df, q, recovery=0.4)["cva"][0])
    return npv, cva


def _torch_pipeline(hybrid_np, hybrid_t, trades, times, n_paths, seed, device):
    curves = _curves()
    sim = simulate_hybrid_paths_torch(hybrid_t, times, n_paths, rng=np.random.default_rng(seed), log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)}, rd_minus_rf={"EUR/USD": 0.01, "GBP/USD": 0.005}, device=device, return_numpy=False)
    npv = fx_forward_portfolio_npv_paths_torch(hybrid_np, trades, times, sim, curves, curves, return_numpy=True)
    exp = aggregate_exposure_profile(npv)
    df = np.asarray([curves["USD"](float(t)) for t in times], dtype=float)
    q = survival_probability_from_hazard(times, np.array([2.5]), np.array([0.015]))
    cva = float(cva_terms_from_profile(times, exp["epe"], df, q, recovery=0.4)["cva"][0])
    return npv, cva


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
    devices = _expand_devices(args.devices)
    hybrid_np, hybrid_t = _build_hybrid()
    times = _times()
    trades = _portfolio(args.trades)
    rows = []
    for n_paths in args.paths:
        ref_npv, ref_cva = _numpy_pipeline(hybrid_np, trades, times, n_paths, args.seed)
        mean_sec, min_sec, std_sec = _bench(lambda n=n_paths: _numpy_pipeline(hybrid_np, trades, times, n, args.seed), args.repeats, args.warmup)
        rows.append({"mode": "numpy/cpu/fx_portfolio", "n_paths": n_paths, "n_trades": args.trades, "mean_sec": mean_sec, "min_sec": min_sec, "std_sec": std_sec})
        for device in devices:
            npv, cva = _torch_pipeline(hybrid_np, hybrid_t, trades, times, n_paths, args.seed, device)
            mean_sec, min_sec, std_sec = _bench(lambda n=n_paths, d=device: _torch_pipeline(hybrid_np, hybrid_t, trades, times, n, args.seed, d), args.repeats, args.warmup)
            rows.append(
                {
                    "mode": f"torch/{device}/fx_portfolio",
                    "n_paths": n_paths,
                    "n_trades": args.trades,
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                    "npv_parity_max_abs": float(np.max(np.abs(npv - ref_npv))),
                    "cva_abs_diff": abs(float(cva) - float(ref_cva)),
                }
            )
    print(json.dumps({"devices": devices, "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
