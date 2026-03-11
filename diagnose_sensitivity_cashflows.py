#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


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


def main() -> None:
    _tools_dir()
    from native_xva_interface import OreSnapshotPythonLgmSensitivityComparator
    from native_xva_interface.mapper import map_snapshot
    from native_xva_interface.runtime import PythonLgmAdapter
    from native_xva_interface.sensitivity import _curve_factor_specs_from_ore_entries

    parser = argparse.ArgumentParser(
        description="Decompose Python t0 cashflow PV response for a chosen ORE XVA sensitivity factor."
    )
    parser.add_argument("--case-dir", type=Path, default=_default_case_dir())
    parser.add_argument("--ore-file", default="ore.xml")
    parser.add_argument("--metric", default="CVA")
    parser.add_argument("--factor", required=True, help="Normalized factor or ORE factor, e.g. zero:EUR:1Y")
    parser.add_argument("--netting-set", default=None)
    args = parser.parse_args()

    comparator, snapshot = OreSnapshotPythonLgmSensitivityComparator.from_case_dir(args.case_dir, ore_file=args.ore_file)
    output_dir = snapshot.config.params.get("outputPath") or str(args.case_dir / "Output")
    ore_entries = comparator.load_ore_zero_sensitivities(output_dir, metric=args.metric, netting_set_id=args.netting_set)
    factor_key = args.factor.strip().upper()
    entry = None
    for candidate in ore_entries:
        if candidate.normalized_factor.upper() == factor_key or candidate.factor.strip().upper() == factor_key:
            entry = candidate
            break
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
    spec = base_inputs.trade_specs[0]
    if spec.kind != "IRS" or spec.legs is None:
        raise SystemExit("This diagnostic currently supports a single IRS trade only")

    legs = spec.legs
    ccy = str(spec.trade.additional_fields.get("currency", spec.ccy)).upper()
    fwd_tenor = str(legs.get("float_index_tenor", "")).upper()
    if not fwd_tenor:
        raise SystemExit("Trade legs are missing float_index_tenor")

    p0d_base = base_inputs.discount_curves[ccy]
    p0d_up = up_inputs.discount_curves[ccy]
    p0f_base = base_inputs.forward_curves_by_tenor[ccy][fwd_tenor]
    p0f_up = up_inputs.forward_curves_by_tenor[ccy][fwd_tenor]

    print(f"metric={args.metric}")
    print(f"factor={entry.normalized_factor}")
    print(f"ore_factor={entry.factor}")
    print(f"shift_size={entry.shift_size:.8f}")
    print(f"trade_id={spec.trade.trade_id}")
    print(f"currency={ccy}")
    print(f"forward_tenor={fwd_tenor}")

    fixed_base = 0.0
    fixed_up = 0.0
    print("fixed_leg")
    print("i pay amount pv_base pv_up diff")
    for i in range(len(legs["fixed_pay_time"])):
        pay = float(legs["fixed_pay_time"][i])
        amount = float(legs["fixed_amount"][i])
        pv_base = amount * float(p0d_base(pay))
        pv_up = amount * float(p0d_up(pay))
        fixed_base += pv_base
        fixed_up += pv_up
        print(f"{i} {pay:.6f} {amount:.6f} {pv_base:.6f} {pv_up:.6f} {pv_up - pv_base:.6f}")

    float_base = 0.0
    float_up = 0.0
    print("float_leg")
    print("i start end pay flow_amount model_amount_base model_amount_up pv_base pv_up diff")
    for i in range(len(legs["float_pay_time"])):
        start = float(legs["float_start_time"][i])
        end = float(legs["float_end_time"][i])
        pay = float(legs["float_pay_time"][i])
        tau = float(legs["float_accrual"][i])
        sign = float(legs["float_sign"][i])
        notional = float(legs["float_notional"][i])
        spread = float(legs["float_spread"][i])
        flow_amount = float(legs["float_amount"][i])
        fwd_base = (float(p0f_base(start)) / float(p0f_base(end)) - 1.0) / tau
        fwd_up = (float(p0f_up(start)) / float(p0f_up(end)) - 1.0) / tau
        amount_base = sign * notional * (fwd_base + spread) * tau
        amount_up = sign * notional * (fwd_up + spread) * tau
        pv_base = amount_base * float(p0d_base(pay))
        pv_up = amount_up * float(p0d_up(pay))
        float_base += pv_base
        float_up += pv_up
        print(
            f"{i} {start:.6f} {end:.6f} {pay:.6f} {flow_amount:.6f} {amount_base:.6f} {amount_up:.6f} "
            f"{pv_base:.6f} {pv_up:.6f} {pv_up - pv_base:.6f}"
        )

    base_result = comparator.engine.create_session(base_snapshot).run(return_cubes=False)
    up_result = comparator.engine.create_session(up_snapshot).run(return_cubes=False)
    print("totals")
    print(f"fixed_base={fixed_base:.6f}")
    print(f"fixed_up={fixed_up:.6f}")
    print(f"fixed_diff={fixed_up - fixed_base:.6f}")
    print(f"float_base={float_base:.6f}")
    print(f"float_up={float_up:.6f}")
    print(f"float_diff={float_up - float_base:.6f}")
    print(f"pv_total_base={fixed_base + float_base:.6f}")
    print(f"pv_total_up={fixed_up + float_up:.6f}")
    print(f"pv_total_diff={fixed_up + float_up - fixed_base - float_base:.6f}")
    print(f"xva_base={float(base_result.xva_by_metric.get(args.metric, 0.0)):.6f}")
    print(f"xva_up={float(up_result.xva_by_metric.get(args.metric, 0.0)):.6f}")
    print(f"python_bump_change={float(up_result.xva_by_metric.get(args.metric, 0.0)) - float(base_result.xva_by_metric.get(args.metric, 0.0)):.6f}")
    print(f"ore_delta={entry.delta:.6f}")


if __name__ == "__main__":
    main()
