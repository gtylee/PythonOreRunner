#!/usr/bin/env python3
"""Benchmark simulate + swap pricing + deflation on NumPy CPU vs torch CPU/MPS."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from py_ore_tools.irs_xva_utils import deflate_lgm_npv_paths, swap_npv_from_ore_legs_dual_curve
from py_ore_tools.lgm import LGM1F, LGMParams, simulate_lgm_measure
from py_ore_tools.lgm_torch import TorchLGM1F, simulate_lgm_measure_torch
from py_ore_tools.lgm_torch_xva import (
    TorchDiscountCurve,
    deflate_lgm_npv_paths_torch_batched,
    swap_npv_paths_from_ore_legs_dual_curve_torch,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=int, nargs="+", default=[10000, 50000, 100000])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", type=str, nargs="+", default=["cpu", "gpu"])
    return p.parse_args(argv)


def _torch_env():
    has_mps = bool(torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return {"cpu": True, "mps": has_mps}


def _expand_devices(requested):
    env = _torch_env()
    out = []
    for item in requested:
        if item == "cpu":
            out.append("cpu")
        elif item == "gpu" and env["mps"]:
            out.append("mps")
        elif item == "mps" and env["mps"]:
            out.append("mps")
    return list(dict.fromkeys(out))


def _flat_curve(rate: float, horizon: float = 30.0, n: int = 121):
    times = np.linspace(0.0, horizon, n, dtype=float)
    dfs = np.exp(-rate * times)
    return times, dfs


def _build_model():
    params = LGMParams(
        alpha_times=(1.0, 5.0, 10.0),
        alpha_values=(0.012, 0.018, 0.015, 0.010),
        kappa_times=(3.0, 12.0),
        kappa_values=(0.025, 0.018, 0.012),
        shift=0.0,
        scaling=1.0,
    )
    return LGM1F(params), TorchLGM1F(params)


def _build_times():
    return np.linspace(0.0, 10.0, 41, dtype=float)


def _build_legs():
    pay = np.arange(0.5, 10.0 + 1.0e-12, 0.5, dtype=float)
    start = np.arange(0.0, 9.5 + 1.0e-12, 0.5, dtype=float)
    end = pay.copy()
    fixed_rate = 0.026
    notional = 1_000_000.0
    return {
        "fixed_pay_time": pay,
        "fixed_amount": np.full(pay.size, -notional * fixed_rate * 0.5, dtype=float),
        "float_pay_time": pay,
        "float_start_time": start,
        "float_end_time": end,
        "float_fixing_time": start.copy(),
        "float_accrual": np.full(pay.size, 0.5, dtype=float),
        "float_index_accrual": np.full(pay.size, 0.5, dtype=float),
        "float_notional": np.full(pay.size, notional, dtype=float),
        "float_sign": np.full(pay.size, 1.0, dtype=float),
        "float_spread": np.full(pay.size, 0.0010, dtype=float),
        "float_coupon": np.full(pay.size, 0.02, dtype=float),
    }


def _bench(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        durations.append(perf_counter() - t0)
    arr = np.asarray(durations, dtype=float)
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


def _numpy_pipeline(model, disc_t, disc_df, fwd_t, fwd_df, legs, times, n_paths, seed):
    p0_disc = lambda t: float(np.interp(float(t), disc_t, disc_df))
    p0_fwd = lambda t: float(np.interp(float(t), fwd_t, fwd_df))
    x = simulate_lgm_measure(model, times, n_paths, rng=np.random.default_rng(seed))
    npv = np.empty_like(x)
    for i, ti in enumerate(times):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(model, p0_disc, p0_fwd, legs, float(ti), x[i, :])
    return deflate_lgm_npv_paths(model, p0_disc, times, x, npv)


def _torch_pipeline(model, disc_curve, fwd_curve, legs, times, n_paths, seed, device, return_numpy):
    with torch.inference_mode():
        x = simulate_lgm_measure_torch(model, times, n_paths, rng=np.random.default_rng(seed), device=device, return_numpy=return_numpy)
        npv = swap_npv_paths_from_ore_legs_dual_curve_torch(model, disc_curve, fwd_curve, legs, times, x, return_numpy=return_numpy)
        return deflate_lgm_npv_paths_torch_batched(model, disc_curve, times, x, npv, return_numpy=return_numpy)


def main(argv: list[str] | None = None):
    if torch is None:
        raise SystemExit("torch is required")
    args = _parse_args(argv)
    devices = _expand_devices(args.devices)
    np_model, torch_model = _build_model()
    times = _build_times()
    legs = _build_legs()
    disc_t, disc_df = _flat_curve(0.02, horizon=30.0)
    fwd_t, fwd_df = _flat_curve(0.0175, horizon=30.0)
    results = []

    for n_paths in args.paths:
        np_ref = _numpy_pipeline(np_model, disc_t, disc_df, fwd_t, fwd_df, legs, times, n_paths, args.seed)
        mean_sec, min_sec, std_sec = _bench(
            lambda n=n_paths: _numpy_pipeline(np_model, disc_t, disc_df, fwd_t, fwd_df, legs, times, n, args.seed),
            args.repeats,
            args.warmup,
        )
        results.append(
            {
                "mode": "numpy/cpu/full_pipeline",
                "n_paths": n_paths,
                "mean_sec": mean_sec,
                "min_sec": min_sec,
                "std_sec": std_sec,
            }
        )
        for device in devices:
            disc_curve = TorchDiscountCurve(disc_t, disc_df, device=device)
            fwd_curve = TorchDiscountCurve(fwd_t, fwd_df, device=device, dtype=disc_curve.dtype)
            if device == "cpu":
                out = _torch_pipeline(torch_model, disc_curve, fwd_curve, legs, times, n_paths, args.seed, device, True)
                disc_curve32 = TorchDiscountCurve(disc_t, disc_df, device=device, dtype=torch.float32)
                fwd_curve32 = TorchDiscountCurve(fwd_t, fwd_df, device=device, dtype=torch.float32)
                out32 = _torch_pipeline(torch_model, disc_curve32, fwd_curve32, legs, times, n_paths, args.seed, device, True)
                mean_sec, min_sec, std_sec = _bench(
                    lambda n=n_paths: _torch_pipeline(torch_model, disc_curve, fwd_curve, legs, times, n, args.seed, device, True),
                    args.repeats,
                    args.warmup,
                )
                row = {
                    "mode": "torch/cpu/full_pipeline",
                    "n_paths": n_paths,
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                }
                row.update(_error_metrics(np_ref, out))
                row.update({f"{k}_vs_cpu_fp32": v for k, v in _error_metrics(out32, out).items()})
                results.append(row)
            elif device == "mps":
                out_dev = _torch_pipeline(torch_model, disc_curve, fwd_curve, legs, times, n_paths, args.seed, device, False)
                cpu_fp32_disc = TorchDiscountCurve(disc_t, disc_df, device="cpu", dtype=torch.float32)
                cpu_fp32_fwd = TorchDiscountCurve(fwd_t, fwd_df, device="cpu", dtype=torch.float32)
                cpu_fp32 = _torch_pipeline(torch_model, cpu_fp32_disc, cpu_fp32_fwd, legs, times, n_paths, args.seed, "cpu", True)
                mean_sec, min_sec, std_sec = _bench(
                    lambda n=n_paths: _torch_pipeline(torch_model, disc_curve, fwd_curve, legs, times, n, args.seed, device, False),
                    args.repeats,
                    args.warmup,
                )
                out_dev_np = out_dev.detach().cpu().numpy()
                row = {
                    "mode": "torch/mps/full_pipeline_device_only",
                    "n_paths": n_paths,
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                }
                row.update(_error_metrics(np_ref, out_dev_np))
                row.update({f"{k}_vs_cpu_fp32": v for k, v in _error_metrics(cpu_fp32, out_dev_np).items()})
                results.append(row)
                out_host = _torch_pipeline(torch_model, disc_curve, fwd_curve, legs, times, n_paths, args.seed, device, True)
                mean_sec, min_sec, std_sec = _bench(
                    lambda n=n_paths: _torch_pipeline(torch_model, disc_curve, fwd_curve, legs, times, n, args.seed, device, True),
                    args.repeats,
                    args.warmup,
                )
                row = {
                    "mode": "torch/mps/full_pipeline_to_numpy",
                    "n_paths": n_paths,
                    "mean_sec": mean_sec,
                    "min_sec": min_sec,
                    "std_sec": std_sec,
                }
                row.update(_error_metrics(np_ref, out_host))
                row.update({f"{k}_vs_cpu_fp32": v for k, v in _error_metrics(cpu_fp32, out_host).items()})
                results.append(row)
    print(json.dumps({"devices": devices, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
