#!/usr/bin/env python3
"""Minimal quickstart for the canonical ORE snapshot loader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pythonore.io.ore_snapshot import load_from_ore_xml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ore-xml",
        type=Path,
        default=REPO_ROOT / "Examples" / "Exposure" / "Input" / "ore_measure_lgm.xml",
        help="Path to an ORE ore.xml file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    snapshot = load_from_ore_xml(args.ore_xml)

    print("Loaded ORE snapshot")
    print(f"  ore_xml        : {snapshot.ore_xml_path}")
    print(f"  asof           : {snapshot.asof_date}")
    print(f"  trade_id       : {snapshot.trade_id}")
    print(f"  counterparty   : {snapshot.counterparty}")
    print(f"  base_ccy       : {snapshot.domestic_ccy}")
    print(f"  measure        : {snapshot.measure}")
    print(f"  alpha_source   : {snapshot.alpha_source}")
    print(f"  discount_curve : {snapshot.discount_column}")
    print(f"  forward_curve  : {snapshot.forward_column}")
    print(f"  ore_t0_npv     : {snapshot.ore_t0_npv:.6f}")
    print(f"  ore_cva        : {snapshot.ore_cva:.6f}")
    print(f"  exposure_pts   : {snapshot.exposure_times.size}")
    print()
    print("Canonical modules:")
    print("  pythonore.io.ore_snapshot")
    print("  pythonore.workflows.ore_snapshot_cli")
    print()
    print("CLI quickstart:")
    print(f"  python3 -m pythonore.apps.ore_snapshot_cli {snapshot.ore_xml_path} --xva --paths 10000")


if __name__ == "__main__":
    main()
