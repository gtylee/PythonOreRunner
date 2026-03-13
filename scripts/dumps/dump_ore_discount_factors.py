#!/usr/bin/env python3
"""Dump per-currency discount factors from ORE artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_ore_tools.ore_snapshot import (
    discount_factors_to_dataframe,
    dump_discount_factors_json,
    extract_discount_factors_by_currency,
)
from py_ore_tools.repo_paths import require_examples_repo_root


def _parse_args() -> argparse.Namespace:
    default_xml = require_examples_repo_root() / "Examples" / "Exposure" / "Input" / "ore_measure_lgm.xml"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ore-xml",
        type=Path,
        default=default_xml,
        help="Path to ORE root XML (default: %(default)s)",
    )
    p.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Todaysmarket configuration id override (defaults to curves analytic config)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for JSON payload",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output path for long-format CSV",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    payload_json = dump_discount_factors_json(
        ore_xml_path=args.ore_xml,
        configuration_id=args.config_id,
        indent=args.indent,
    )
    payload = extract_discount_factors_by_currency(
        ore_xml_path=args.ore_xml,
        configuration_id=args.config_id,
    )

    if args.json_out is not None:
        args.json_out.write_text(payload_json + "\n", encoding="utf-8")
        print(f"Wrote JSON payload: {args.json_out}")
    else:
        print(payload_json)

    if args.csv_out is not None:
        df = discount_factors_to_dataframe(payload)
        df.to_csv(args.csv_out, index=False)
        print(f"Wrote CSV payload: {args.csv_out}")


if __name__ == "__main__":
    main()
