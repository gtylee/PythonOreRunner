#!/usr/bin/env python3
"""Micro-profile one native sensitivity factor on a small ORE snapshot case."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path
from dataclasses import replace

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pythonore.io.loader import XVALoader
from pythonore.runtime.sensitivity import OreSnapshotPythonLgmSensitivityComparator, _prune_native_factor_setup_for_portfolio
from pythonore.workflows.ore_snapshot_cli import _parse_sensitivity_analytic_params, _parse_sensitivity_factor_setup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a single native sensitivity factor on a small case.")
    parser.add_argument("case_input_dir", type=Path, help="Case Input directory containing ore.xml")
    parser.add_argument("--factor", default=None, help="Normalized factor to profile. Defaults to the first pruned factor.")
    parser.add_argument(
        "--lgm-param-source",
        choices=("auto", "simulation_xml", "ore"),
        default="simulation_xml",
        help="LGM parameter source for the native sensitivity pricing step.",
    )
    parser.add_argument("--top", type=int, default=25, help="Number of cProfile rows to print")
    return parser.parse_args()


def _resolve_factor(case_input_dir: Path, factor_override: str | None):
    ore_xml = case_input_dir / "ore.xml"
    snapshot = XVALoader.from_files(str(case_input_dir), ore_file="ore.xml")
    sensi_params = _parse_sensitivity_analytic_params(ore_xml)
    factor_shifts, curve_specs, factor_labels = _parse_sensitivity_factor_setup(ore_xml, sensi_params=sensi_params)
    pruned_shifts, pruned_specs, pruned_labels, pruned_count = _prune_native_factor_setup_for_portfolio(
        snapshot,
        factor_shifts=factor_shifts,
        curve_factor_specs=curve_specs,
        factor_labels=factor_labels,
    )
    factor = factor_override or sorted(pruned_shifts)[0]
    if factor not in pruned_shifts:
        raise ValueError(f"factor '{factor}' not found in pruned factor set")
    return snapshot, factor, pruned_shifts, pruned_specs, pruned_labels, pruned_count


def main() -> int:
    args = _parse_args()
    case_input_dir = args.case_input_dir.resolve()
    snapshot, factor, factor_shifts, curve_specs, factor_labels, pruned_count = _resolve_factor(
        case_input_dir, args.factor
    )
    snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            params={
                **dict(snapshot.config.params),
                "python.lgm_param_source": str(args.lgm_param_source),
            },
        ),
    )

    comparator = OreSnapshotPythonLgmSensitivityComparator()
    native_snapshot = comparator.engine.prepare_sensitivity_snapshot(
        snapshot,
        curve_fit_mode="ore_fit",
        use_ore_output_curves=False,
        freeze_float_spreads=True,
    )
    frozen_float_spreads = native_snapshot.config.params.get("python.frozen_float_spreads")
    base_mapped = comparator._mapped_inputs_with_market(native_snapshot, comparator.engine.create_session(native_snapshot).state.mapped_inputs)
    quote_map = comparator._discover_supported_quotes(native_snapshot)
    curve_spec = curve_specs.get(factor)
    quote_entries = quote_map.get(factor, [])
    quote = quote_entries[0][1] if quote_entries else None
    shift_size = float(factor_shifts[factor])
    bump_mode = "quote_value"

    started = time.perf_counter()
    base_npv = comparator._price_snapshot_t0_npv(native_snapshot, mapped=base_mapped)
    base_elapsed = time.perf_counter() - started

    if curve_spec is not None:
        started = time.perf_counter()
        up_snapshot = comparator._bump_snapshot_curve(native_snapshot, curve_spec, shift_size, frozen_float_spreads)
        bump_up_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        down_snapshot = comparator._bump_snapshot_curve(native_snapshot, curve_spec, -shift_size, frozen_float_spreads)
        bump_down_elapsed = time.perf_counter() - started
    else:
        if quote is None:
            raise ValueError(f"factor '{factor}' has no quote entries")
        up_values, down_values = comparator._bumped_quote_values(
            quotes=[q for _, q in quote_entries],
            normalized_factor=factor,
            shift_size=shift_size,
            bump_mode=bump_mode,
        )
        started = time.perf_counter()
        up_snapshot = comparator.engine.prepare_sensitivity_snapshot(
            comparator._bump_snapshot_quotes(native_snapshot, quote_entries, up_values),
            curve_fit_mode="ore_fit",
            use_ore_output_curves=False,
            frozen_float_spreads=frozen_float_spreads,
        )
        bump_up_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        down_snapshot = comparator.engine.prepare_sensitivity_snapshot(
            comparator._bump_snapshot_quotes(native_snapshot, quote_entries, down_values),
            curve_fit_mode="ore_fit",
            use_ore_output_curves=False,
            frozen_float_spreads=frozen_float_spreads,
        )
        bump_down_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    up_mapped = comparator._mapped_inputs_with_market(up_snapshot, base_mapped)
    up_map_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    down_mapped = comparator._mapped_inputs_with_market(down_snapshot, base_mapped)
    down_map_elapsed = time.perf_counter() - started

    prof = cProfile.Profile()
    prof.enable()
    started = time.perf_counter()
    up_npv = comparator._price_snapshot_t0_npv(up_snapshot, mapped=up_mapped)
    up_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    down_npv = comparator._price_snapshot_t0_npv(down_snapshot, mapped=down_mapped)
    down_elapsed = time.perf_counter() - started
    prof.disable()

    print(f"case_input_dir      : {case_input_dir}")
    print(f"selected_factor     : {factor}")
    print(f"pruned_factor_count : {len(factor_shifts)}")
    print(f"pruned_out_count    : {pruned_count}")
    print(f"base_npv            : {base_npv:.12f}")
    print(f"up_npv              : {up_npv:.12f}")
    print(f"down_npv            : {down_npv:.12f}")
    print(f"base_price_sec      : {base_elapsed:.6f}")
    print(f"bump_up_build_sec   : {bump_up_elapsed:.6f}")
    print(f"bump_down_build_sec : {bump_down_elapsed:.6f}")
    print(f"up_map_sec          : {up_map_elapsed:.6f}")
    print(f"down_map_sec        : {down_map_elapsed:.6f}")
    print(f"up_price_sec        : {up_elapsed:.6f}")
    print(f"down_price_sec      : {down_elapsed:.6f}")
    print()
    stats = pstats.Stats(prof)
    stats.sort_stats("cumulative").print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
