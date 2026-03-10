#!/usr/bin/env python3
"""Quick benchmark for per-currency discount-factor extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from py_ore_tools.ore_snapshot import extract_discount_factors_by_currency


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    default_xml = repo_root / "Examples/Exposure/Input/ore_measure_lgm.xml"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ore-xml", type=Path, default=default_xml)
    p.add_argument("--runs", type=int, default=30, help="Number of extraction runs")
    p.add_argument("--config-id", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Warmup to amortize import/runtime effects.
    _ = extract_discount_factors_by_currency(args.ore_xml, configuration_id=args.config_id)

    start = perf_counter()
    n_points = 0
    for _ in range(args.runs):
        payload = extract_discount_factors_by_currency(
            args.ore_xml,
            configuration_id=args.config_id,
        )
        n_points += sum(len(v["times"]) for v in payload.values())
    elapsed = perf_counter() - start
    per_run_ms = 1000.0 * elapsed / max(args.runs, 1)

    print(f"runs={args.runs}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"ms_per_run={per_run_ms:.3f}")
    print(f"points_processed={n_points}")


if __name__ == "__main__":
    main()
