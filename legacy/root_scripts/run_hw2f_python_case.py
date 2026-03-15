#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from py_ore_tools.hw2f_integration import compare_hw2f_python_vs_ore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare pure-Python HW 2F swap pricing with ORE output.")
    parser.add_argument("case_dir", type=Path, help="Case directory containing Input/ and Output/.")
    parser.add_argument("--discount-column", default="EUR-EONIA", help="curves.csv discount column to use.")
    parser.add_argument("--trade-id", default=None, help="Optional explicit trade id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_hw2f_python_vs_ore(
        args.case_dir,
        discount_column=args.discount_column,
        trade_id=args.trade_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
