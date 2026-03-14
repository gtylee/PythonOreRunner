#!/usr/bin/env python3
"""Example: 10-ccy FX-forward portfolio XVA in NumPy vs torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from py_ore_tools.irs_xva_utils import compute_xva_from_npv_paths, survival_probability_from_hazard
from py_ore_tools.lgm import LGMParams
from py_ore_tools.lgm_fx_hybrid import LgmFxHybrid, MultiCcyLgmParams
from py_ore_tools.lgm_fx_hybrid_torch import TorchLgmFxHybrid, simulate_hybrid_paths_torch
from py_ore_tools.lgm_fx_xva_utils import FxForwardDef, fx_forward_portfolio_npv_paths

try:
    import torch
except ImportError:  # pragma: no cover - demo exits cleanly without torch
    torch = None


CCYS = ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK")
USD_CROSSES = ("EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "USD/SEK", "USD/NOK")
ZERO_RATES = {
    "USD": 0.030,
    "EUR": 0.020,
    "GBP": 0.025,
    "JPY": 0.010,
    "CHF": 0.012,
    "AUD": 0.032,
    "CAD": 0.029,
    "NZD": 0.034,
    "SEK": 0.022,
    "NOK": 0.028,
}
ALPHA_BY_CCY = {
    "USD": 0.012,
    "EUR": 0.010,
    "GBP": 0.011,
    "JPY": 0.009,
    "CHF": 0.0095,
    "AUD": 0.0125,
    "CAD": 0.0115,
    "NZD": 0.013,
    "SEK": 0.0105,
    "NOK": 0.011,
}
KAPPA_BY_CCY = {
    "USD": 0.020,
    "EUR": 0.030,
    "GBP": 0.025,
    "JPY": 0.018,
    "CHF": 0.019,
    "AUD": 0.024,
    "CAD": 0.022,
    "NZD": 0.023,
    "SEK": 0.021,
    "NOK": 0.0225,
}
SPOT_BY_PAIR = {
    "EUR/USD": 1.10,
    "GBP/USD": 1.27,
    "USD/JPY": 148.0,
    "USD/CHF": 0.88,
    "AUD/USD": 0.66,
    "USD/CAD": 1.35,
    "NZD/USD": 0.61,
    "USD/SEK": 10.4,
    "USD/NOK": 10.7,
}
FX_VOL_BY_PAIR = {
    "EUR/USD": 0.12,
    "GBP/USD": 0.11,
    "USD/JPY": 0.10,
    "USD/CHF": 0.095,
    "AUD/USD": 0.13,
    "USD/CAD": 0.10,
    "NZD/USD": 0.135,
    "USD/SEK": 0.11,
    "USD/NOK": 0.115,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--trades", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "mps", "all"), default="all")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def _build_hybrid() -> tuple[LgmFxHybrid, TorchLgmFxHybrid]:
    ir_params = {
        ccy: LGMParams.constant(alpha=ALPHA_BY_CCY[ccy], kappa=KAPPA_BY_CCY[ccy])
        for ccy in CCYS
    }
    fx_vols = {pair: (tuple(), (FX_VOL_BY_PAIR[pair],)) for pair in USD_CROSSES}
    n_ir = len(CCYS)
    n_fx = len(USD_CROSSES)
    n_factors = n_ir + n_fx
    corr = np.empty((n_factors, n_factors), dtype=float)
    for i in range(n_factors):
        for j in range(n_factors):
            if i == j:
                corr[i, j] = 1.0
            elif i < n_ir and j < n_ir:
                corr[i, j] = 0.35 * np.exp(-abs(i - j) / 4.0)
            elif i >= n_ir and j >= n_ir:
                corr[i, j] = 0.40 * np.exp(-abs(i - j) / 3.0)
            else:
                corr[i, j] = 0.15 * np.exp(-abs(i - j) / 5.0)
    corr = 0.5 * (corr + corr.T)
    params = MultiCcyLgmParams(ir_params=ir_params, fx_vols=fx_vols, corr=corr)
    return LgmFxHybrid(params), TorchLgmFxHybrid(params)


def _curve(rate: float):
    return lambda t: float(np.exp(-rate * float(t)))


def _curves():
    return {ccy: _curve(rate) for ccy, rate in ZERO_RATES.items()}


def _times() -> np.ndarray:
    return np.linspace(0.0, 2.5, 33, dtype=float)


def _portfolio(n_trades: int) -> list[FxForwardDef]:
    trades: list[FxForwardDef] = []
    maturities = np.linspace(0.5, 2.5, n_trades, dtype=float)
    pairs = list(USD_CROSSES)
    for i, maturity in enumerate(maturities):
        pair = pairs[i % len(pairs)]
        spot = SPOT_BY_PAIR[pair]
        strike_bump = 0.0005 * ((i % len(pairs)) + 1) / len(pairs)
        strike = spot * (1.0 + (-1.0 if i % 3 == 0 else 1.0) * strike_bump)
        trades.append(
            FxForwardDef(
                trade_id=f"FX_{pair.replace('/', '')}_{i:03d}",
                pair=pair,
                notional_base=1_000_000.0 + 7_500.0 * i,
                strike=float(strike),
                maturity=float(maturity),
            )
        )
    return trades


def _sim_inputs():
    log_s0 = {pair: np.log(spot) for pair, spot in SPOT_BY_PAIR.items()}
    rd_minus_rf = {}
    for pair in USD_CROSSES:
        base, quote = pair.split("/")
        rd_minus_rf[pair] = ZERO_RATES[quote] - ZERO_RATES[base]
    return log_s0, rd_minus_rf


def _simulate_numpy(hybrid: LgmFxHybrid, times: np.ndarray, n_paths: int, seed: int):
    log_s0, rd_minus_rf = _sim_inputs()
    return hybrid.simulate_paths(
        times,
        n_paths,
        rng=np.random.default_rng(seed),
        log_s0=log_s0,
        rd_minus_rf=rd_minus_rf,
    )


def _simulate_torch(hybrid: TorchLgmFxHybrid, times: np.ndarray, n_paths: int, seed: int, device: str):
    log_s0, rd_minus_rf = _sim_inputs()
    return simulate_hybrid_paths_torch(
        hybrid,
        times,
        n_paths,
        rng=np.random.default_rng(seed),
        log_s0=log_s0,
        rd_minus_rf=rd_minus_rf,
        device=device,
        return_numpy=False,
    )


def _xva_pack(times: np.ndarray, npv_paths: np.ndarray) -> dict[str, object]:
    discount = np.asarray([_curves()["USD"](float(t)) for t in times], dtype=float)
    q_cpty = survival_probability_from_hazard(times, np.array([2.5], dtype=float), np.array([0.015], dtype=float))
    q_own = survival_probability_from_hazard(times, np.array([2.5], dtype=float), np.array([0.010], dtype=float))
    return compute_xva_from_npv_paths(
        times=times,
        npv_paths=npv_paths,
        discount=discount,
        survival_cpty=q_cpty,
        survival_own=q_own,
        recovery_cpty=0.40,
        recovery_own=0.40,
        funding_spread=0.0015,
        exposure_discounting="discount_curve",
    )


def _numpy_pipeline(hybrid: LgmFxHybrid, trades: list[FxForwardDef], times: np.ndarray, n_paths: int, seed: int):
    curves = _curves()
    sim = _simulate_numpy(hybrid, times, n_paths, seed)
    npv = fx_forward_portfolio_npv_paths(
        hybrid,
        trades,
        times,
        sim,
        curves,
        curves,
        tensor_backend="numpy",
        return_numpy=True,
    )
    return npv, _xva_pack(times, npv)


def _torch_pipeline(
    hybrid_np: LgmFxHybrid,
    hybrid_t: TorchLgmFxHybrid,
    trades: list[FxForwardDef],
    times: np.ndarray,
    n_paths: int,
    seed: int,
    device: str,
):
    curves = _curves()
    sim = _simulate_torch(hybrid_t, times, n_paths, seed, device)
    npv = fx_forward_portfolio_npv_paths(
        hybrid_np,
        trades,
        times,
        sim,
        curves,
        curves,
        tensor_backend="torch-mps" if device == "mps" else "torch-cpu",
        return_numpy=True,
    )
    return npv, _xva_pack(times, npv)


def _summary(pack: dict[str, object]) -> dict[str, float]:
    return {
        "cva": float(pack["cva"]),
        "dva": float(pack["dva"]),
        "fva": float(pack["fva"]),
        "xva_total": float(pack["xva_total"]),
        "ee_t0": float(np.asarray(pack["ee"], dtype=float)[0]),
        "epe_max": float(np.max(np.asarray(pack["epe"], dtype=float))),
        "ene_max": float(np.max(np.asarray(pack["ene"], dtype=float))),
    }


def _parity(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    diff = np.asarray(got, dtype=float) - np.asarray(ref, dtype=float)
    return {
        "npv_max_abs": float(np.max(np.abs(diff))),
        "npv_rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _bench(fn, *, repeats: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        vals.append(perf_counter() - t0)
    arr = np.asarray(vals, dtype=float)
    return {
        "mean_sec": float(arr.mean()),
        "min_sec": float(arr.min()),
        "std_sec": float(arr.std(ddof=0)),
    }


def _available_torch_devices() -> list[str]:
    out = ["cpu"]
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        out.append("mps")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if torch is None:
        raise SystemExit("torch is required for this demo")
    available_devices = _available_torch_devices()
    if args.device == "mps" and "mps" not in available_devices:
        raise SystemExit("requested device 'mps' is not available")

    hybrid_np, hybrid_t = _build_hybrid()
    times = _times()
    trades = _portfolio(args.trades)

    npv_numpy, xva_numpy = _numpy_pipeline(hybrid_np, trades, times, args.paths, args.seed)
    numpy_timing = _bench(
        lambda: _numpy_pipeline(hybrid_np, trades, times, args.paths, args.seed),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    torch_devices = available_devices if args.device == "all" else [args.device]
    torch_results: dict[str, dict[str, object]] = {}
    for device in torch_devices:
        npv_torch, xva_torch = _torch_pipeline(hybrid_np, hybrid_t, trades, times, args.paths, args.seed, device)
        torch_timing = _bench(
            lambda d=device: _torch_pipeline(hybrid_np, hybrid_t, trades, times, args.paths, args.seed, d),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        torch_results[device] = {
            "summary": _summary(xva_torch),
            "timing": {
                **torch_timing,
                "speedup_torch_vs_numpy": numpy_timing["mean_sec"] / max(torch_timing["mean_sec"], 1.0e-12),
            },
            "parity": {
                **_parity(npv_numpy, npv_torch),
                "xva_total_abs_diff": abs(float(xva_torch["xva_total"]) - float(xva_numpy["xva_total"])),
                "cva_abs_diff": abs(float(xva_torch["cva"]) - float(xva_numpy["cva"])),
                "dva_abs_diff": abs(float(xva_torch["dva"]) - float(xva_numpy["dva"])),
                "fva_abs_diff": abs(float(xva_torch["fva"]) - float(xva_numpy["fva"])),
            },
        }

    result = {
        "currencies": list(CCYS),
        "trade_pairs": list(USD_CROSSES),
        "paths": args.paths,
        "trades": args.trades,
        "seed": args.seed,
        "device": args.device,
        "torch_devices_tested": torch_devices,
        "numpy": _summary(xva_numpy),
        "timing": {"numpy": numpy_timing},
        "torch": torch_results,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("LGM FX portfolio XVA example")
    print("=" * 80)
    print(f"{'Settings':<22} {'Value'}")
    print("=" * 80)
    print(f"{'currencies':<22} {', '.join(CCYS)}")
    print(f"{'pairs':<22} {', '.join(USD_CROSSES)}")
    print(f"{'trades':<22} {args.trades:,}")
    print(f"{'paths':<22} {args.paths:,}")
    print(f"{'seed':<22} {args.seed:,}")
    print(f"{'device':<22} {args.device}")
    print("=" * 80)
    print()

    numbers_headers = (
        f"{'Engine':<16}"
        f"{'CVA':>18} {'DVA':>18} {'FVA':>18} {'XVA':>18} "
        f"{'NPV max abs':>18} {'NPV RMSE':>18} {'XVA diff':>18}"
    )
    print(numbers_headers)
    print("-" * len(numbers_headers))
    print(
        f"{'NumPy':<16}"
        f"{result['numpy']['cva']:18,.3f}"
        f"{result['numpy']['dva']:18,.3f}"
        f"{result['numpy']['fva']:18,.3f}"
        f"{result['numpy']['xva_total']:18,.3f}"
        f"{'':>18}{'':>18}{'':>18}"
    )
    for device in torch_devices:
        torch_row = result["torch"][device]
        print(
            f"{('Torch ' + device.upper()):<16}"
            f"{torch_row['summary']['cva']:18,.3f}"
            f"{torch_row['summary']['dva']:18,.3f}"
            f"{torch_row['summary']['fva']:18,.3f}"
            f"{torch_row['summary']['xva_total']:18,.3f}"
            f"{torch_row['parity']['npv_max_abs']:18.3e}"
            f"{torch_row['parity']['npv_rmse']:18.3e}"
            f"{torch_row['parity']['xva_total_abs_diff']:18.3e}"
        )
    print("=" * 80)
    print()

    timing_header = f"{'Engine':<16}{'Mean sec':>18} {'Speedup':>18}"
    print(timing_header)
    print("-" * len(timing_header))
    print(f"{'NumPy':<16}{result['timing']['numpy']['mean_sec']:18,.3f}{'':>18}")
    for device in torch_devices:
        torch_row = result["torch"][device]
        print(
            f"{('Torch ' + device.upper()):<16}"
            f"{torch_row['timing']['mean_sec']:18,.3f}"
            f"{torch_row['timing']['speedup_torch_vs_numpy']:18,.2f}x"
        )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
