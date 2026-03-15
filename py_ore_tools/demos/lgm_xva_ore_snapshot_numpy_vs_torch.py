#!/usr/bin/env python3
"""Example: ORE snapshot swap/XVA in NumPy vs torch."""

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

from py_ore_tools.irs_xva_utils import (
    compute_realized_float_coupons,
    compute_xva_from_npv_paths,
    deflate_lgm_npv_paths,
    survival_probability_from_hazard,
    swap_npv_from_ore_legs_dual_curve,
)
from py_ore_tools.lgm_torch import simulate_lgm_measure_torch
from py_ore_tools.ore_snapshot import load_from_ore_xml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_xml = repo_root / "Examples/Exposure/Input/ore_measure_lgm.xml"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ore-xml", type=Path, default=default_xml)
    p.add_argument("--paths", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "mps", "all"), default="all")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--xva-mode", choices=("classic", "ore"), default="ore")
    p.add_argument("--anchor-t0-npv", action="store_true")
    p.add_argument("--flip-npv-perspective", action="store_true")
    p.add_argument("--own-hazard", type=float, default=0.01)
    p.add_argument("--own-recovery", type=float, default=0.4)
    p.add_argument("--borrow-spread", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def _available_torch_devices() -> list[str]:
    out = ["cpu"]
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        out.append("mps")
    return out


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


def _shared_sim_grid(exposure_times: np.ndarray, fixing_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extra = np.asarray(fixing_times, dtype=float)
    extra = extra[extra > 1.0e-12]
    sim_times = np.unique(np.concatenate((np.asarray(exposure_times, dtype=float), extra)))
    idx = np.searchsorted(sim_times, exposure_times)
    if not np.allclose(sim_times[idx], exposure_times, atol=1.0e-12, rtol=0.0):
        raise ValueError("failed to align exposure times on simulation grid")
    return sim_times, idx


def _own_survival(snap, times: np.ndarray, own_hazard: float, own_recovery: float):
    if snap.own_hazard_times is not None and snap.own_hazard_rates is not None and snap.own_recovery is not None:
        return (
            survival_probability_from_hazard(times, snap.own_hazard_times, snap.own_hazard_rates),
            float(snap.own_recovery),
        )
    return (
        survival_probability_from_hazard(times, np.array([0.5, 1.0, 5.0, 10.0]), np.full(4, own_hazard, dtype=float)),
        float(own_recovery),
    )


def _xva_pack(snap, times: np.ndarray, npv_paths: np.ndarray, *, xva_mode: str, own_hazard: float, own_recovery: float, borrow_spread: float):
    discount = snap.discount_factors(times)
    q_cpty = snap.survival_probability(times)
    q_own, rec_own = _own_survival(snap, times, own_hazard, own_recovery)
    funding_kwargs = {}
    if snap.p0_borrow is not None and snap.p0_lend is not None and snap.p0_xva_disc is not None:
        funding_kwargs = {
            "funding_discount_borrow": np.asarray([snap.p0_borrow(float(t)) for t in times], dtype=float),
            "funding_discount_lend": np.asarray([snap.p0_lend(float(t)) for t in times], dtype=float),
            "funding_discount_ois": np.asarray([snap.p0_xva_disc(float(t)) for t in times], dtype=float),
        }
    return compute_xva_from_npv_paths(
        times=times,
        npv_paths=npv_paths,
        discount=discount,
        survival_cpty=q_cpty,
        survival_own=q_own,
        recovery_cpty=snap.recovery,
        recovery_own=rec_own,
        funding_spread=borrow_spread,
        exposure_discounting="numeraire_deflated" if xva_mode == "ore" else "discount_curve",
        **funding_kwargs,
    )


def _summary(pack: dict[str, object]) -> dict[str, float]:
    return {
        "npv_t0": float(np.asarray(pack["ee"], dtype=float)[0]),
        "cva": float(pack["cva"]),
        "dva": float(pack["dva"]),
        "fba": float(pack.get("fba", 0.0)),
        "fca": float(pack.get("fca", 0.0)),
        "fva": float(pack["fva"]),
        "xva_total": float(pack["xva_total"]),
        "epe_max": float(np.max(np.asarray(pack["epe"], dtype=float))),
        "ene_max": float(np.max(np.asarray(pack["ene"], dtype=float))),
    }


def _parity(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    diff = np.asarray(got, dtype=float) - np.asarray(ref, dtype=float)
    return {
        "npv_max_abs": float(np.max(np.abs(diff))),
        "npv_rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _simulate_lgm_from_draws_numpy(model, times: np.ndarray, normal_draws: np.ndarray, x0: float = 0.0) -> np.ndarray:
    zeta_grid = np.asarray(model.zeta(times), dtype=float)
    step_scales = np.sqrt(np.maximum(np.diff(zeta_grid), 0.0))
    if normal_draws.shape != (times.size - 1, normal_draws.shape[1]):
        raise ValueError("normal_draws must have shape (n_steps, n_paths)")
    out = np.empty((times.size, normal_draws.shape[1]), dtype=float)
    out[0, :] = float(x0)
    out[1:, :] = float(x0) + np.cumsum(step_scales[:, None] * normal_draws, axis=0)
    return out


def _ore_reference(snap) -> dict[str, float]:
    return {
        "ore_t0_npv": float(snap.ore_t0_npv),
        "ore_cva": float(snap.ore_cva),
        "ore_dva": float(snap.ore_dva),
        "ore_fba": float(snap.ore_fba),
        "ore_fca": float(snap.ore_fca),
        "ore_xva_total": float(snap.ore_cva - snap.ore_dva + snap.ore_fba + snap.ore_fca),
    }


def _numpy_pipeline(snap, sim_times: np.ndarray, exposure_idx: np.ndarray, normal_draws: np.ndarray, *, xva_mode: str, flip_npv_perspective: bool, own_hazard: float, own_recovery: float, borrow_spread: float):
    model = snap.build_model()
    x_all = _simulate_lgm_from_draws_numpy(model, sim_times, normal_draws)
    realized_coupon = compute_realized_float_coupons(
        model=model,
        p0_disc=snap.p0_disc,
        p0_fwd=snap.p0_fwd,
        legs=snap.legs,
        sim_times=sim_times,
        x_paths_on_sim_grid=x_all,
    )
    x_exp = x_all[exposure_idx, :]
    npv = np.empty_like(x_exp)
    for i, ti in enumerate(snap.exposure_model_times):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(
            model,
            snap.p0_disc,
            snap.p0_fwd,
            snap.legs,
            float(ti),
            x_exp[i, :],
            realized_float_coupon=realized_coupon,
        )
    if flip_npv_perspective:
        npv = -npv
    npv_xva = deflate_lgm_npv_paths(model, snap.p0_disc, snap.exposure_model_times, x_exp, npv) if xva_mode == "ore" else npv
    return npv_xva, _xva_pack(
        snap,
        snap.exposure_model_times,
        npv_xva,
        xva_mode=xva_mode,
        own_hazard=own_hazard,
        own_recovery=own_recovery,
        borrow_spread=borrow_spread,
    )


def _torch_pipeline(snap, sim_times: np.ndarray, exposure_idx: np.ndarray, normal_draws: np.ndarray, *, device: str, xva_mode: str, flip_npv_perspective: bool, own_hazard: float, own_recovery: float, borrow_spread: float):
    model = snap.build_model()
    x_all = simulate_lgm_measure_torch(model, sim_times, normal_draws.shape[1], normal_draws=normal_draws, device=device, return_numpy=True)
    realized_coupon = compute_realized_float_coupons(
        model=snap.build_model(),
        p0_disc=snap.p0_disc,
        p0_fwd=snap.p0_fwd,
        legs=snap.legs,
        sim_times=sim_times,
        x_paths_on_sim_grid=x_all,
    )
    x_exp = x_all[exposure_idx, :]
    npv = np.empty_like(x_exp)
    for i, ti in enumerate(snap.exposure_model_times):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(
            model,
            snap.p0_disc,
            snap.p0_fwd,
            snap.legs,
            float(ti),
            x_exp[i, :],
            realized_float_coupon=realized_coupon,
        )
    if flip_npv_perspective:
        npv = -npv
    npv_xva = (
        deflate_lgm_npv_paths(model, snap.p0_disc, snap.exposure_model_times, x_exp, npv)
        if xva_mode == "ore"
        else npv
    )
    return npv_xva, _xva_pack(
        snap,
        snap.exposure_model_times,
        npv_xva,
        xva_mode=xva_mode,
        own_hazard=own_hazard,
        own_recovery=own_recovery,
        borrow_spread=borrow_spread,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if torch is None:
        raise SystemExit("torch is required for this demo")
    available_devices = _available_torch_devices()
    if args.device == "mps" and "mps" not in available_devices:
        raise SystemExit("requested device 'mps' is not available")

    snap = load_from_ore_xml(args.ore_xml, anchor_t0_npv=args.anchor_t0_npv)
    fixing_times = np.asarray(snap.legs.get("float_fixing_time", []), dtype=float)
    sim_times, exposure_idx = _shared_sim_grid(snap.exposure_model_times, fixing_times)
    normal_draws = np.random.default_rng(args.seed).standard_normal((sim_times.size - 1, args.paths))

    npv_numpy, xva_numpy = _numpy_pipeline(
        snap,
        sim_times,
        exposure_idx,
        normal_draws,
        xva_mode=args.xva_mode,
        flip_npv_perspective=args.flip_npv_perspective,
        own_hazard=args.own_hazard,
        own_recovery=args.own_recovery,
        borrow_spread=args.borrow_spread,
    )
    numpy_timing = _bench(
        lambda: _numpy_pipeline(
            snap,
            sim_times,
            exposure_idx,
            normal_draws,
            xva_mode=args.xva_mode,
            flip_npv_perspective=args.flip_npv_perspective,
            own_hazard=args.own_hazard,
            own_recovery=args.own_recovery,
            borrow_spread=args.borrow_spread,
        ),
        repeats=args.repeats,
        warmup=args.warmup,
    )

    torch_devices = available_devices if args.device == "all" else [args.device]
    torch_results: dict[str, dict[str, object]] = {}
    for device in torch_devices:
        npv_torch, xva_torch = _torch_pipeline(
            snap,
            sim_times,
            exposure_idx,
            normal_draws,
            device=device,
            xva_mode=args.xva_mode,
            flip_npv_perspective=args.flip_npv_perspective,
            own_hazard=args.own_hazard,
            own_recovery=args.own_recovery,
            borrow_spread=args.borrow_spread,
        )
        torch_timing = _bench(
            lambda d=device: _torch_pipeline(
                snap,
                sim_times,
                exposure_idx,
                normal_draws,
                device=d,
                xva_mode=args.xva_mode,
                flip_npv_perspective=args.flip_npv_perspective,
                own_hazard=args.own_hazard,
                own_recovery=args.own_recovery,
                borrow_spread=args.borrow_spread,
            ),
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
                "fba_abs_diff": abs(float(xva_torch.get("fba", 0.0)) - float(xva_numpy.get("fba", 0.0))),
                "fca_abs_diff": abs(float(xva_torch.get("fca", 0.0)) - float(xva_numpy.get("fca", 0.0))),
                "fva_abs_diff": abs(float(xva_torch["fva"]) - float(xva_numpy["fva"])),
            },
        }

    result = {
        "ore_xml": str(args.ore_xml),
        "trade_id": str(snap.trade_id),
        "counterparty": str(snap.counterparty),
        "netting_set_id": str(snap.netting_set_id),
        "paths": args.paths,
        "seed": args.seed,
        "device": args.device,
        "xva_mode": args.xva_mode,
        "torch_devices_tested": torch_devices,
        "ore_reference": _ore_reference(snap),
        "numpy": _summary(xva_numpy),
        "timing": {"numpy": numpy_timing},
        "torch": torch_results,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("ORE snapshot XVA example")
    print("=" * 80)
    print(f"{'ore_xml':<22} {args.ore_xml}")
    print(f"{'trade_id':<22} {snap.trade_id}")
    print(f"{'counterparty':<22} {snap.counterparty}")
    print(f"{'netting_set_id':<22} {snap.netting_set_id}")
    print(f"{'paths':<22} {args.paths:,}")
    print(f"{'seed':<22} {args.seed}")
    print(f"{'device':<22} {args.device}")
    print(f"{'xva_mode':<22} {args.xva_mode}")
    print("=" * 80)
    print()

    numbers_headers = (
        f"{'Engine':<16}"
        f"{'NPV':>18} {'CVA':>18} {'DVA':>18} {'FBA':>18} {'FCA':>18} {'FVA':>18} {'XVA':>18} "
        f"{'NPV max abs':>18} {'NPV RMSE':>18} {'XVA diff':>18}"
    )
    print(numbers_headers)
    print("-" * len(numbers_headers))
    print(
        f"{'ORE ref':<16}"
        f"{result['ore_reference']['ore_t0_npv']:18,.3f}"
        f"{result['ore_reference']['ore_cva']:18,.3f}"
        f"{result['ore_reference']['ore_dva']:18,.3f}"
        f"{result['ore_reference']['ore_fba']:18,.3f}"
        f"{result['ore_reference']['ore_fca']:18,.3f}"
        f"{(result['ore_reference']['ore_fba'] + result['ore_reference']['ore_fca']):18,.3f}"
        f"{result['ore_reference']['ore_xva_total']:18,.3f}"
        f"{'':>18}{'':>18}{'':>18}"
    )
    print(
        f"{'NumPy':<16}"
        f"{result['numpy']['npv_t0']:18,.3f}"
        f"{result['numpy']['cva']:18,.3f}"
        f"{result['numpy']['dva']:18,.3f}"
        f"{result['numpy']['fba']:18,.3f}"
        f"{result['numpy']['fca']:18,.3f}"
        f"{result['numpy']['fva']:18,.3f}"
        f"{result['numpy']['xva_total']:18,.3f}"
        f"{'':>18}{'':>18}{'':>18}"
    )
    for device in torch_devices:
        row = result["torch"][device]
        print(
            f"{('Torch ' + device.upper()):<16}"
            f"{row['summary']['npv_t0']:18,.3f}"
            f"{row['summary']['cva']:18,.3f}"
            f"{row['summary']['dva']:18,.3f}"
            f"{row['summary']['fba']:18,.3f}"
            f"{row['summary']['fca']:18,.3f}"
            f"{row['summary']['fva']:18,.3f}"
            f"{row['summary']['xva_total']:18,.3f}"
            f"{row['parity']['npv_max_abs']:18.3e}"
            f"{row['parity']['npv_rmse']:18.3e}"
            f"{row['parity']['xva_total_abs_diff']:18.3e}"
        )
    print("=" * 80)
    print()

    timing_header = f"{'Engine':<16}{'Mean sec':>18} {'Speedup':>18}"
    print(timing_header)
    print("-" * len(timing_header))
    print(f"{'NumPy':<16}{result['timing']['numpy']['mean_sec']:18,.3f}{'':>18}")
    for device in torch_devices:
        row = result["torch"][device]
        print(
            f"{('Torch ' + device.upper()):<16}"
            f"{row['timing']['mean_sec']:18,.3f}"
            f"{row['timing']['speedup_torch_vs_numpy']:18,.2f}x"
        )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
