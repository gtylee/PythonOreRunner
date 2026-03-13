#!/usr/bin/env python3
"""Benchmark front-to-back FX forward profile/XVA in NumPy vs torch."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from py_ore_tools.lgm_fx_xva_utils import run_fx_forward_profile_xva, run_fx_forward_profile_xva_torch


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", type=int, default=[10000, 50000])
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


def _kwargs(n_paths, seed):
    return dict(
        name="FXFWD_BENCH",
        pair="EUR/USD",
        maturity=1.0,
        spot0=1.1,
        strike=1.12,
        notional_base=1_000_000,
        dom_zero_rate=[(0.0, 0.03), (5.0, 0.03)],
        for_zero_rate=[(0.0, 0.02), (5.0, 0.02)],
        fx_vol=0.12,
        n_paths=n_paths,
        seed=seed,
    )


def main(argv=None):
    args = _parse_args(argv)
    devices = _expand_devices(args.devices)
    rows = []
    for n_paths in args.paths:
        ref = run_fx_forward_profile_xva(**_kwargs(n_paths, args.seed))
        mean_sec, min_sec, std_sec = _bench(lambda n=n_paths: run_fx_forward_profile_xva(**_kwargs(n, args.seed)), args.repeats, args.warmup)
        rows.append({"mode": "numpy/cpu/fx_forward_profile_xva", "n_paths": n_paths, "mean_sec": mean_sec, "min_sec": min_sec, "std_sec": std_sec})
        for device in devices:
            out = run_fx_forward_profile_xva_torch(**_kwargs(n_paths, args.seed), device=device)
            mean_sec, min_sec, std_sec = _bench(
                lambda n=n_paths, d=device: run_fx_forward_profile_xva_torch(**_kwargs(n, args.seed), device=d),
                args.repeats,
                args.warmup,
            )
            rows.append(
                {
                    "mode": f"torch/{device}/fx_forward_profile_xva",
                    "n_paths": n_paths,
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                    "npv_parity_max_abs": float(np.max(np.abs(out["npv_paths"] - ref["npv_paths"]))),
                    "cva_abs_diff": abs(float(out["cva"]) - float(ref["cva"])),
                    "xva_total_abs_diff": abs(float(out["xva_total"]) - float(ref["xva_total"])),
                }
            )
    print(json.dumps({"devices": devices, "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
