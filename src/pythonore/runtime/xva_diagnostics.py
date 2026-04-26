from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pythonore.compute.irs_xva_utils import (
    curve_values,
    compute_xva_from_exposure_profile,
    load_ore_exposure_profile,
    survival_probability_from_hazard,
)
from pythonore.mapping.mapper import map_snapshot
from pythonore.io.loader import XVALoader
from pythonore.io.ore_snapshot import load_from_ore_xml
from pythonore.runtime.runtime import PythonLgmAdapter, XVAEngine


def compare_native_exposure_to_ore(
    case_dir: str | Path,
    *,
    paths: int | None = None,
    seed: int | None = None,
    rng_mode: str = "ore_parity",
    use_ore_output_curves: bool = True,
    compare_backends: bool = True,
    abs_threshold: float = 1000.0,
    rel_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare native Python XVA exposure rows with ORE's exposure report.

    The native run is built from ORE input XML, market data, curves, fixings, and
    portfolio data. ORE ``flows.csv`` is not used by this diagnostic.
    """
    case_root = Path(case_dir).resolve()
    input_dir = case_root / "Input"
    output_dir = case_root / "Output"
    ore_xml = input_dir / "ore.xml"
    if not ore_xml.exists():
        raise FileNotFoundError(f"ORE input not found: {ore_xml}")

    snapshot = XVALoader.from_files(str(input_dir), ore_file="ore.xml")
    if paths is not None or seed is not None or rng_mode or use_ore_output_curves:
        runtime = snapshot.config.runtime
        if runtime is not None and runtime.simulation is not None and seed is not None:
            runtime = replace(runtime, simulation=replace(runtime.simulation, seed=int(seed)))
        params = {**snapshot.config.params, "python.lgm_rng_mode": str(rng_mode)}
        if use_ore_output_curves and (output_dir / "curves.csv").exists():
            params["python.use_ore_output_curves"] = "Y"
        params["python.store_npv_cube_paths"] = "Y"
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=int(paths if paths is not None else snapshot.config.num_paths),
                runtime=runtime,
                params=params,
            ),
        )

    result, primary_seconds = _run_native(snapshot)
    trade_id = snapshot.portfolio.trades[0].trade_id if snapshot.portfolio.trades else ""
    netting_set_id = snapshot.portfolio.trades[0].netting_set if snapshot.portfolio.trades else "PORTFOLIO"
    native_profile = result.exposure_profiles_by_trade.get(trade_id)
    if native_profile is None:
        native_profile = result.exposure_profiles_by_netting_set.get(netting_set_id)
    if native_profile is None:
        raise ValueError(f"native result has no exposure profile for trade '{trade_id}' or netting set '{netting_set_id}'")

    exposure_csv = output_dir / f"exposure_trade_{trade_id}.csv"
    if not exposure_csv.exists():
        matches = sorted(output_dir.glob("exposure_trade_*.csv"))
        if not matches:
            raise FileNotFoundError(f"ORE exposure report not found under {output_dir}")
        exposure_csv = matches[0]
    ore_profile = load_ore_exposure_profile(str(exposure_csv))
    ore_rawcube = _load_ore_rawcube_profile(output_dir / "rawcube.csv", trade_id)

    ore_dates = [str(x) for x in ore_profile.get("date", [])]
    ore_times = np.asarray(ore_profile.get("time", []), dtype=float)
    ore_epe = np.asarray(ore_profile.get("epe", []), dtype=float)
    ore_ene = np.asarray(ore_profile.get("ene", []), dtype=float)

    native_dates = [str(x) for x in native_profile.get("dates", [])]
    native_times = np.asarray(native_profile.get("times", []), dtype=float)
    py_epe = np.asarray(native_profile.get("closeout_epe", []), dtype=float)
    py_ene = np.asarray(native_profile.get("closeout_ene", []), dtype=float)

    alignment = _align_exposure_grids(ore_dates, ore_times, native_dates, native_times)
    if not alignment:
        raise ValueError(f"exposure grids have no shared rows: ORE={ore_times.shape}, native={native_times.shape}")
    ore_idx = np.asarray([a[0] for a in alignment], dtype=int)
    py_idx = np.asarray([a[1] for a in alignment], dtype=int)
    grid_mismatch = int(ore_times.size != native_times.size or len(alignment) != int(ore_times.size))

    ore_dates = [ore_dates[i] for i in ore_idx]
    ore_times = ore_times[ore_idx]
    ore_epe = ore_epe[ore_idx]
    ore_ene = ore_ene[ore_idx]
    py_epe = py_epe[py_idx]
    py_ene = py_ene[py_idx]

    epe_abs = np.abs(py_epe - ore_epe)
    ene_abs = np.abs(py_ene - ore_ene)
    epe_rel = _safe_rel(py_epe, ore_epe)
    ene_rel = _safe_rel(py_ene, ore_ene)
    material = (epe_abs > float(abs_threshold)) & (epe_rel > float(rel_threshold))
    first_idx = int(np.argmax(material)) if np.any(material) else None
    worst_idx = int(np.argmax(epe_abs)) if epe_abs.size else None

    ore_snap = load_from_ore_xml(ore_xml, anchor_t0_npv=False)
    xva_split = _xva_formula_split(ore_snap, ore_times, ore_epe, ore_ene, py_epe, py_ene)

    pointwise = [
        {
            "Date": ore_dates[i] if i < len(ore_dates) else native_dates[i] if i < len(native_dates) else "",
            "Time": float(ore_times[i]),
            "OreEPE": float(ore_epe[i]),
            "PythonEPE": float(py_epe[i]),
            "EPEAbsDiff": float(epe_abs[i]),
            "EPERelDiff": float(epe_rel[i]),
            "OreENE": float(ore_ene[i]),
            "PythonENE": float(py_ene[i]),
            "ENEAbsDiff": float(ene_abs[i]),
            "ENERelDiff": float(ene_rel[i]),
        }
        for i in range(int(ore_times.size))
    ]

    t0_diagnostics = _t0_leg_split(snapshot, output_dir, trade_id)
    floating_coupon_comparison = list(t0_diagnostics.pop("floating_coupon_comparison", []))

    summary = {
        "case_dir": str(case_root),
        "trade_id": trade_id,
        "netting_set_id": netting_set_id,
        "paths": int(snapshot.config.num_paths),
        "rng_mode": str(rng_mode),
        "market_source": str(result.metadata.get("input_provenance", {}).get("market", "")),
        "irs_pricing_backend": str(result.metadata.get("irs_pricing_backend", "")),
        "runtime_seconds": float(primary_seconds),
        "ore_grid_points": int(len(ore_profile.get("time", []))),
        "native_grid_points": int(native_times.size),
        "grid_points": int(ore_times.size),
        "grid_mismatch": bool(grid_mismatch),
        "native_only_grid_points": int(native_times.size - len({a[1] for a in alignment})),
        "ore_unmatched_grid_points": int(len(ore_profile.get("time", [])) - len({a[0] for a in alignment})),
        "first_material_epe_divergence": _row_summary(pointwise, first_idx),
        "worst_epe_divergence": _row_summary(pointwise, worst_idx),
        "median_epe_rel_diff": float(np.median(epe_rel)) if epe_rel.size else 0.0,
        "p95_epe_rel_diff": float(np.percentile(epe_rel, 95.0)) if epe_rel.size else 0.0,
        "max_epe_abs_diff": float(np.max(epe_abs)) if epe_abs.size else 0.0,
        "median_ene_rel_diff": float(np.median(ene_rel)) if ene_rel.size else 0.0,
        "p95_ene_rel_diff": float(np.percentile(ene_rel, 95.0)) if ene_rel.size else 0.0,
        "max_ene_abs_diff": float(np.max(ene_abs)) if ene_abs.size else 0.0,
        **xva_split,
        **_native_cube_split(result, trade_id, ore_dates, ore_rawcube),
        **t0_diagnostics,
    }
    if compare_backends:
        summary["backend_comparison"] = _compare_numpy_torch_backends(snapshot)
    return {"summary": summary, "pointwise": pointwise, "floating_coupon_comparison": floating_coupon_comparison}


def write_exposure_diagnostic(result: Mapping[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(result.get("pointwise", []))
    if rows:
        with open(out / "pointwise.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    coupon_rows = list(result.get("floating_coupon_comparison", []))
    if coupon_rows:
        with open(out / "floating_coupons.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(coupon_rows[0].keys()))
            writer.writeheader()
            writer.writerows(coupon_rows)


def _run_native(snapshot: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snapshot).run(return_cubes=True)
    return result, time.perf_counter() - start


def _snapshot_with_backend(snapshot: Any, backend: str) -> Any:
    params = {**snapshot.config.params, "python.store_npv_cube_paths": "Y"}
    if backend == "numpy":
        params["python.torch_device"] = "numpy"
    elif backend:
        params["python.torch_device"] = backend
    return replace(snapshot, config=replace(snapshot.config, params=params))


def _compare_numpy_torch_backends(snapshot: Any) -> dict[str, Any]:
    numpy_snapshot = _snapshot_with_backend(snapshot, "numpy")
    torch_snapshot = _snapshot_with_backend(snapshot, "cpu")
    numpy_result, numpy_seconds = _run_native(numpy_snapshot)
    torch_result, torch_seconds = _run_native(torch_snapshot)
    trade_id = snapshot.portfolio.trades[0].trade_id if snapshot.portfolio.trades else ""
    return {
        "numpy_backend": str(numpy_result.metadata.get("irs_pricing_backend", "")),
        "torch_backend": str(torch_result.metadata.get("irs_pricing_backend", "")),
        "numpy_seconds": float(numpy_seconds),
        "torch_seconds": float(torch_seconds),
        "torch_speedup_vs_numpy": float(numpy_seconds / torch_seconds) if torch_seconds > 0.0 else 0.0,
        "pv_diff": float(torch_result.pv_total) - float(numpy_result.pv_total),
        "cva_diff": float(torch_result.xva_by_metric.get("CVA", 0.0)) - float(numpy_result.xva_by_metric.get("CVA", 0.0)),
        "dva_diff": float(torch_result.xva_by_metric.get("DVA", 0.0)) - float(numpy_result.xva_by_metric.get("DVA", 0.0)),
        **_backend_cube_diff(numpy_result, torch_result, trade_id),
    }


def _backend_cube_diff(numpy_result: Any, torch_result: Any, trade_id: str) -> dict[str, float]:
    numpy_cube = numpy_result.cubes.get("npv_cube") if getattr(numpy_result, "cubes", None) else None
    torch_cube = torch_result.cubes.get("npv_cube") if getattr(torch_result, "cubes", None) else None
    if numpy_cube is None or torch_cube is None:
        return {}
    np_payload = numpy_cube.payload.get(trade_id, {})
    th_payload = torch_cube.payload.get(trade_id, {})
    out: dict[str, float] = {}
    for key in ("npv_paths", "npv_xva_paths"):
        a = np.asarray(np_payload.get(key, []), dtype=float)
        b = np.asarray(th_payload.get(key, []), dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue
        prefix = "raw" if key == "npv_paths" else "xva"
        out[f"{prefix}_cube_max_abs_diff"] = float(np.max(np.abs(b - a)))
        out[f"{prefix}_cube_mean_abs_diff"] = float(np.mean(np.abs(b - a)))
    return out


def _safe_rel(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(values - reference) / np.maximum(np.abs(reference), 1.0)


def _align_exposure_grids(
    ore_dates: Sequence[str],
    ore_times: np.ndarray,
    native_dates: Sequence[str],
    native_times: np.ndarray,
) -> list[tuple[int, int]]:
    by_date: dict[str, int] = {str(d): i for i, d in enumerate(native_dates) if str(d)}
    out: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, d in enumerate(ore_dates):
        j = by_date.get(str(d))
        if j is not None:
            out.append((i, j))
            used.add(j)
            continue
        if native_times.size == 0:
            continue
        deltas = np.abs(native_times - float(ore_times[i]))
        for cand in np.argsort(deltas):
            j = int(cand)
            if j in used:
                continue
            # ORE reports actual calendar year fractions while the simulation
            # grid can be regular tenor fractions. Accept nearest monthly row,
            # but do not jump across more than roughly half a monthly bucket.
            if float(deltas[j]) <= 0.05:
                out.append((i, j))
                used.add(j)
            break
    return out


def _row_summary(rows: Sequence[Mapping[str, Any]], idx: int | None) -> dict[str, Any] | None:
    if idx is None or idx < 0 or idx >= len(rows):
        return None
    row = dict(rows[idx])
    row["Index"] = int(idx)
    return row


def _xva_formula_split(
    snap: Any,
    times: np.ndarray,
    ore_epe: np.ndarray,
    ore_ene: np.ndarray,
    py_epe: np.ndarray,
    py_ene: np.ndarray,
) -> dict[str, float]:
    if times.size < 2:
        return {}
    q_c = snap.survival_probability(times)
    q_b = survival_probability_from_hazard(times, snap.own_hazard_times, snap.own_hazard_rates)
    discount = np.asarray([snap.p0_xva_disc(float(t)) for t in times], dtype=float)
    kwargs = {
        "times": times,
        "discount": discount,
        "survival_cpty": q_c,
        "survival_own": q_b,
        "recovery_cpty": snap.recovery,
        "recovery_own": float(snap.own_recovery),
        "exposure_discounting": "numeraire_deflated",
    }
    ore_pack = compute_xva_from_exposure_profile(epe=ore_epe, ene=ore_ene, **kwargs)
    py_pack = compute_xva_from_exposure_profile(epe=py_epe, ene=py_ene, **kwargs)
    return {
        "ore_report_cva": float(snap.ore_cva),
        "ore_report_dva": float(snap.ore_dva),
        "python_formula_on_ore_epe_cva": float(ore_pack["cva"]),
        "python_formula_on_python_epe_cva": float(py_pack["cva"]),
        "python_formula_on_ore_ene_dva": float(ore_pack["dva"]),
        "python_formula_on_python_ene_dva": float(py_pack["dva"]),
    }


def _load_ore_rawcube_profile(path: Path, trade_id: str) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    grouped: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_trade = row.get("#Id") or row.get("Id") or ""
            if row_trade != trade_id:
                continue
            date = str(row.get("Date", "")).strip()
            if not date:
                continue
            try:
                grouped.setdefault(date, []).append(float(row.get("Value", "0") or 0.0))
            except ValueError:
                continue
    out: dict[str, dict[str, float]] = {}
    for date, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        out[date] = {
            "mean": float(np.mean(arr)),
            "epe": float(np.mean(np.maximum(arr, 0.0))),
            "ene": float(np.mean(np.maximum(-arr, 0.0))),
            "paths": int(arr.size),
        }
    return out


def _profile_from_matrix(dates: Sequence[str], matrix: Any) -> dict[str, dict[str, float]]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        return {}
    out: dict[str, dict[str, float]] = {}
    for i, date in enumerate(dates):
        if i >= arr.shape[0]:
            break
        row = arr[i, :]
        out[str(date)] = {
            "mean": float(np.mean(row)),
            "epe": float(np.mean(np.maximum(row, 0.0))),
            "ene": float(np.mean(np.maximum(-row, 0.0))),
            "paths": int(row.size),
        }
    return out


def _native_cube_split(
    result: Any,
    trade_id: str,
    ore_dates: Sequence[str],
    ore_rawcube: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    npv_cube = result.cubes.get("npv_cube") if getattr(result, "cubes", None) else None
    payload = npv_cube.payload.get(trade_id, {}) if npv_cube is not None else {}
    dates = [str(x) for x in payload.get("dates", [])]
    if not dates:
        dates = list(ore_dates)
    native_raw = _profile_from_matrix(dates, payload.get("npv_paths", []))
    native_xva = _profile_from_matrix(dates, payload.get("npv_xva_paths", []))
    common = [d for d in ore_dates if d in ore_rawcube and d in native_raw and d in native_xva]
    if not common:
        return {}

    def max_diff(native: Mapping[str, Mapping[str, float]], field: str) -> float:
        return float(max(abs(float(native[d][field]) - float(ore_rawcube[d][field])) for d in common))

    def first_row(native: Mapping[str, Mapping[str, float]], field: str) -> dict[str, Any]:
        d = common[0]
        return {
            "date": d,
            "ore": float(ore_rawcube[d][field]),
            "python": float(native[d][field]),
            "diff": float(native[d][field]) - float(ore_rawcube[d][field]),
        }

    return {
        "ore_rawcube_points": int(len(ore_rawcube)),
        "cube_comparison_points": int(len(common)),
        "native_raw_cube_paths": int(native_raw[common[0]]["paths"]),
        "ore_raw_cube_paths_first_point": int(ore_rawcube[common[0]]["paths"]),
        "native_raw_vs_ore_max_epe_abs_diff": max_diff(native_raw, "epe"),
        "native_xva_vs_ore_max_epe_abs_diff": max_diff(native_xva, "epe"),
        "native_raw_vs_ore_t0_mean": first_row(native_raw, "mean"),
        "native_xva_vs_ore_t0_mean": first_row(native_xva, "mean"),
    }


def _t0_leg_split(snapshot: Any, output_dir: Path, trade_id: str) -> dict[str, Any]:
    try:
        adapter = PythonLgmAdapter(fallback_to_swig=False)
        adapter._ensure_py_lgm_imports()
        mapped = map_snapshot(snapshot)
        inputs = adapter._extract_inputs(snapshot, mapped)
        spec = next((s for s in inputs.trade_specs if s.trade.trade_id == trade_id), None)
        if spec is None or spec.legs is None or spec.kind != "IRS":
            return {}
        legs = spec.legs
        p_disc = inputs.discount_curves[spec.ccy]
        index_name = str(legs.get("float_index", spec.trade.additional_fields.get("index", "")))
        p_fwd = adapter._resolve_index_curve(inputs, spec.ccy, index_name)

        fixed_pay = np.asarray(legs.get("fixed_pay_time", []), dtype=float)
        fixed_amount = np.asarray(legs.get("fixed_amount", []), dtype=float)
        fixed_pv = float(np.sum(fixed_amount * np.asarray(curve_values(p_disc, fixed_pay), dtype=float))) if fixed_pay.size else 0.0

        start = np.asarray(legs.get("float_start_time", []), dtype=float)
        end = np.asarray(legs.get("float_end_time", []), dtype=float)
        pay = np.asarray(legs.get("float_pay_time", []), dtype=float)
        tau = np.asarray(legs.get("float_accrual", []), dtype=float)
        index_tau = np.asarray(legs.get("float_index_accrual", tau), dtype=float)
        notional = np.asarray(legs.get("float_notional", []), dtype=float)
        sign = np.asarray(legs.get("float_sign", []), dtype=float)
        spread = np.nan_to_num(np.asarray(legs.get("float_spread", np.zeros_like(tau)), dtype=float), nan=0.0)
        native_coupon = np.asarray([], dtype=float)
        native_amount = np.asarray([], dtype=float)
        native_pv = np.asarray([], dtype=float)
        float_pv = 0.0
        if pay.size:
            gearing = np.asarray(legs.get("float_gearing", np.ones_like(tau)), dtype=float)
            p_s = np.asarray(curve_values(p_fwd, start), dtype=float)
            p_e = np.asarray(curve_values(p_fwd, end), dtype=float)
            fwd = np.nan_to_num((p_s / p_e - 1.0) / np.maximum(index_tau, 1.0e-18), nan=0.0)
            native_coupon = np.nan_to_num(gearing * fwd + spread, nan=0.0)
            native_amount = sign * notional * native_coupon * tau
            native_pv = native_amount * np.asarray(curve_values(p_disc, pay), dtype=float)
            float_pv = float(np.sum(native_pv))
        native = {
            "fixed_pv": fixed_pv,
            "float_pv": float_pv,
            "net_pv": fixed_pv + float_pv,
        }
        ore, coupon, coupon_rows = _ore_flow_leg_split(
            output_dir / "flows.csv",
            trade_id,
            native_coupon,
            native_amount,
            native_pv,
        )
        return {
            "t0_leg_split": {"native": native, "ore_flow_report": ore, "coupon_comparison": coupon},
            "floating_coupon_comparison": coupon_rows,
        }
    except Exception as exc:
        return {"t0_leg_split_error": str(exc)}


def _ore_flow_leg_split(
    path: Path,
    trade_id: str,
    native_float_coupons: np.ndarray,
    native_float_amounts: np.ndarray,
    native_float_pvs: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        return {}, {}, []
    fixed_pv = 0.0
    float_pv = 0.0
    float_rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("#TradeId") != trade_id and row.get("TradeId") != trade_id:
                continue
            try:
                pv = float(row.get("PresentValue(Base)", "0") or 0.0)
            except ValueError:
                pv = 0.0
            if row.get("LegNo") == "0":
                fixed_pv += pv
            elif row.get("LegNo") == "1":
                float_pv += pv
                float_rows.append(dict(row))
    coupon_cmp: dict[str, Any] = {}
    coupon_rows: list[dict[str, Any]] = []
    if float_rows and native_float_coupons.size:
        n = min(len(float_rows), int(native_float_coupons.size))
        ore_coupons = np.asarray([_float_or_nan(r.get("Coupon", "nan")) for r in float_rows[:n]], dtype=float)
        ore_pvs = np.asarray([_float_or_nan(r.get("PresentValue(Base)", "nan")) for r in float_rows[:n]], dtype=float)
        ore_amounts = np.asarray([_float_or_nan(r.get("Amount", "nan")) for r in float_rows[:n]], dtype=float)
        diff = np.asarray(native_float_coupons[:n], dtype=float) - ore_coupons
        pv_diff = np.asarray(native_float_pvs[:n], dtype=float) - ore_pvs
        worst = int(np.argmax(np.abs(diff))) if diff.size else 0
        coupon_cmp = {
            "points": int(n),
            "max_abs_coupon_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "worst_coupon_index": int(worst + 1) if diff.size else 0,
            "worst_coupon_diff": float(diff[worst]) if diff.size else 0.0,
            "max_abs_pv_diff": float(np.nanmax(np.abs(pv_diff))) if pv_diff.size else 0.0,
        }
        for i, row in enumerate(float_rows[:n]):
            coupon_rows.append(
                {
                    "CouponNo": int(i + 1),
                    "AccrualStartDate": row.get("AccrualStartDate", ""),
                    "AccrualEndDate": row.get("AccrualEndDate", ""),
                    "PayDate": row.get("PayDate", ""),
                    "FixingDate": row.get("FixingDate", ""),
                    "OreCoupon": float(ore_coupons[i]),
                    "NativeCoupon": float(native_float_coupons[i]),
                    "CouponDiff": float(diff[i]),
                    "OreAmount": float(ore_amounts[i]),
                    "NativeAmount": float(native_float_amounts[i]),
                    "AmountDiff": float(native_float_amounts[i] - ore_amounts[i]),
                    "OrePV": float(ore_pvs[i]),
                    "NativePV": float(native_float_pvs[i]),
                    "PVDiff": float(pv_diff[i]),
                }
            )
    return {"fixed_pv": fixed_pv, "float_pv": float_pv, "net_pv": fixed_pv + float_pv}, coupon_cmp, coupon_rows


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare native Python XVA exposure rows against ORE exposure reports.")
    p.add_argument("case_dir", type=Path)
    p.add_argument("--paths", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rng", choices=["numpy", "ore_parity"], default="ore_parity")
    p.add_argument(
        "--fit-market-quotes",
        action="store_true",
        help="Do not consume ORE Output/curves.csv; fit native curves from market quotes instead.",
    )
    p.add_argument("--skip-backend-compare", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = compare_native_exposure_to_ore(
        args.case_dir,
        paths=args.paths,
        seed=args.seed,
        rng_mode=args.rng,
        use_ore_output_curves=not bool(args.fit_market_quotes),
        compare_backends=not bool(args.skip_backend_compare),
    )
    if args.output_dir is not None:
        write_exposure_diagnostic(result, args.output_dir)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
