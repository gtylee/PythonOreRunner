#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _default_case_dir() -> Path:
    return Path(__file__).resolve().parent / "parity_artifacts" / "multiccy_benchmark_final" / "cases" / "flat_EUR_5Y_A"


def main() -> None:
    from native_xva_interface import OreSnapshotPythonLgmSensitivityComparator

    parser = argparse.ArgumentParser(description="Run Python-LGM ore_snapshot sensitivities and compare with ORE zero-sensitivity CSVs.")
    parser.add_argument("--case-dir", type=Path, default=_default_case_dir())
    parser.add_argument("--ore-file", default="ore.xml")
    parser.add_argument("--metric", default="CVA")
    parser.add_argument("--netting-set", default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    comparator, snapshot = OreSnapshotPythonLgmSensitivityComparator.from_case_dir(args.case_dir, ore_file=args.ore_file)
    result = comparator.compare(snapshot, metric=args.metric, netting_set_id=args.netting_set)

    print(f"metric={result['metric']}")
    print(f"python factors={len(result['python'])}")
    print(f"ore factors={len(result['ore'])}")
    print(f"matched={len(result['comparisons'])}")
    if result.get("unsupported_ore") or result.get("unsupported_python"):
        print(
            f"unsupported credit factors="
            f"{max(len(result.get('unsupported_ore', [])), len(result.get('unsupported_python', [])))}"
        )
    unsupported_factors = result.get("unsupported_factors", [])
    if unsupported_factors:
        print("unsupported factors:")
        for factor in unsupported_factors:
            print(f"  {factor}")
    for note in result.get("notes", []):
        print(note)
    if result["unmatched_ore"]:
        print(f"unmatched ORE factors={len(result['unmatched_ore'])}")
        for factor in result["unmatched_ore"][: args.top]:
            print(f"  {factor}")
    if result["unmatched_python"]:
        print(f"unmatched Python factors={len(result['unmatched_python'])}")
        for factor in result["unmatched_python"][: args.top]:
            print(f"  {factor}")

    for row in result["comparisons"][: args.top]:
        print(
            f"{row.normalized_factor}: "
            f"PY={row.python_delta:.6f} "
            f"ORE={row.ore_delta:.6f} "
            f"diff={row.delta_diff:.6f} "
            f"rel={row.delta_rel_diff:.2%} "
            f"quote={row.python_quote_key}"
        )


if __name__ == "__main__":
    main()
