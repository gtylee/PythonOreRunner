#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _tools_dir() -> Path:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return here


def _default_case_dir() -> Path:
    return (
        _tools_dir()
        / "parity_artifacts"
        / "multiccy_benchmark_final"
        / "cases"
        / "flat_EUR_5Y_A"
    )


def _parse_times(txt: str) -> list[float]:
    out = []
    for token in txt.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _zero_from_df(df: float, t: float) -> float:
    if t <= 0.0:
        return 0.0
    return -math.log(max(df, 1.0e-16)) / t


def _ore_bucket_weight(shift_times: Sequence[float], bucket: int, t: float) -> float:
    t = float(t)
    tenors = [float(x) for x in shift_times]
    j = int(bucket)
    t1 = tenors[j]
    if len(tenors) == 1:
        return 1.0
    if j == 0:
        t2 = tenors[j + 1]
        if t <= t1:
            return 1.0
        if t <= t2:
            return (t2 - t) / (t2 - t1)
        return 0.0
    if j == len(tenors) - 1:
        t0 = tenors[j - 1]
        if t >= t0 and t <= t1:
            return (t - t0) / (t1 - t0)
        if t > t1:
            return 1.0
        return 0.0
    t0 = tenors[j - 1]
    t2 = tenors[j + 1]
    if t >= t0 and t <= t1:
        return (t - t0) / (t1 - t0)
    if t > t1 and t <= t2:
        return (t2 - t) / (t2 - t1)
    return 0.0


def _find_matching_entry(entries: Iterable, factor: str):
    factor_u = factor.strip().upper()
    for entry in entries:
        if entry.normalized_factor.upper() == factor_u:
            return entry
        if entry.factor.strip().upper() == factor_u:
            return entry
    return None


def _curve_fn_for_factor(inputs, curve_spec: dict[str, object]):
    kind = str(curve_spec.get("kind", "")).lower()
    if kind == "discount":
        return inputs.discount_curves[str(curve_spec["ccy"]).upper()]
    if kind == "forward":
        ccy = str(curve_spec["ccy"]).upper()
        tenor = str(curve_spec["index_tenor"]).upper()
        return inputs.forward_curves_by_tenor[ccy][tenor]
    raise ValueError(f"Unsupported curve kind for bucket diagnostic: {kind}")


def main() -> None:
    _tools_dir()
    from native_xva_interface import OreSnapshotPythonLgmSensitivityComparator
    from native_xva_interface.mapper import map_snapshot
    from native_xva_interface.runtime import PythonLgmAdapter
    from native_xva_interface.sensitivity import _curve_factor_specs_from_ore_entries

    parser = argparse.ArgumentParser(
        description="Diagnose ORE vs Python bucket shock behavior for a chosen XVA sensitivity factor."
    )
    parser.add_argument("--case-dir", type=Path, default=_default_case_dir())
    parser.add_argument("--ore-file", default="ore.xml")
    parser.add_argument("--metric", default="CVA")
    parser.add_argument("--factor", required=True, help="Normalized factor or ORE factor, e.g. zero:EUR:1Y")
    parser.add_argument("--sample-times", default="1,2,3,5")
    parser.add_argument("--netting-set", default=None)
    args = parser.parse_args()

    comparator, snapshot = OreSnapshotPythonLgmSensitivityComparator.from_case_dir(args.case_dir, ore_file=args.ore_file)
    output_dir = snapshot.config.params.get("outputPath") or str(args.case_dir / "Output")
    ore_entries = comparator.load_ore_zero_sensitivities(output_dir, metric=args.metric, netting_set_id=args.netting_set)
    entry = _find_matching_entry(ore_entries, args.factor)
    if entry is None:
        raise SystemExit(f"Factor '{args.factor}' not found in ORE zero sensitivity rows under {output_dir}")

    curve_specs = _curve_factor_specs_from_ore_entries(ore_entries)
    curve_spec = curve_specs.get(entry.normalized_factor)
    if curve_spec is None:
        raise SystemExit(f"Factor '{entry.normalized_factor}' is not a curve factor")

    base_snapshot = comparator.engine.prepare_sensitivity_snapshot(
        snapshot,
        curve_fit_mode="ore_fit",
        use_ore_output_curves=False,
        freeze_float_spreads=True,
    )
    frozen = base_snapshot.config.params.get("python.frozen_float_spreads")
    up_snapshot = comparator._bump_snapshot_curve(base_snapshot, curve_spec, entry.shift_size, frozen)

    adapter = comparator.engine.adapter
    if not isinstance(adapter, PythonLgmAdapter):
        raise SystemExit("This diagnostic expects a PythonLgmAdapter-backed engine")

    adapter._ensure_py_lgm_imports()
    base_inputs = adapter._extract_inputs(base_snapshot, map_snapshot(base_snapshot))
    up_inputs = adapter._extract_inputs(up_snapshot, map_snapshot(up_snapshot))

    base_curve = _curve_fn_for_factor(base_inputs, curve_spec)
    up_curve = _curve_fn_for_factor(up_inputs, curve_spec)

    base_result = comparator.engine.create_session(base_snapshot).run(return_cubes=False)
    up_result = comparator.engine.create_session(up_snapshot).run(return_cubes=False)

    sample_times = _parse_times(args.sample_times)
    shift_times = [float(x) for x in curve_spec.get("node_times", [])]
    target_time = float(curve_spec.get("target_time", 0.0))
    try:
        bucket = shift_times.index(target_time)
    except ValueError:
        bucket = min(range(len(shift_times)), key=lambda i: abs(shift_times[i] - target_time))

    print(f"metric={args.metric}")
    print(f"factor={entry.normalized_factor}")
    print(f"ore_factor={entry.factor}")
    print(f"shift_size={entry.shift_size}")
    print(f"bucket_index={bucket}")
    print(f"bucket_times={','.join(str(x) for x in shift_times)}")
    print(f"base_metric={float(base_result.xva_by_metric.get(args.metric, 0.0)):.6f}")
    print(f"up_metric={float(up_result.xva_by_metric.get(args.metric, 0.0)):.6f}")
    print(f"python_bump_change={float(up_result.xva_by_metric.get(args.metric, 0.0)) - float(base_result.xva_by_metric.get(args.metric, 0.0)):.6f}")
    print(f"ore_delta={entry.delta:.6f}")
    print(f"base_pv_total={float(base_result.pv_total):.6f}")
    print(f"up_pv_total={float(up_result.pv_total):.6f}")
    print("time  ore_w  ore_zero_shift  base_df  up_df  py_zero_shift")
    for t in sample_times:
        w = _ore_bucket_weight(shift_times, bucket, t)
        ore_zero_shift = w * float(entry.shift_size)
        base_df = float(base_curve(t))
        up_df = float(up_curve(t))
        py_zero_shift = _zero_from_df(up_df, t) - _zero_from_df(base_df, t)
        print(
            f"{t:4.1f}  {w:5.2f}  {ore_zero_shift:14.8f}  {base_df:7.6f}  {up_df:7.6f}  {py_zero_shift:13.8f}"
        )


if __name__ == "__main__":
    main()
