#!/usr/bin/env python3
"""Convenience entrypoint for the tuned USD limited-sensitivity benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_ore_snapshot_usd_one_trade_limited_sensi import main as _benchmark_main


def main() -> int:
    argv = list(sys.argv[1:])
    defaults = {
        "--trade-count": "1000",
        "--trade-mix": "mixed",
        "--paths": "2000",
        "--sensi-tenors": "1Y,2Y,5Y,10Y,20Y,30Y",
    }
    present = {arg for arg in argv if arg.startswith("--")}
    merged: list[str] = []
    for flag, value in defaults.items():
        if flag not in present:
            merged.extend([flag, value])
    merged.extend(argv)
    sys.argv = [sys.argv[0], *merged]
    return _benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
