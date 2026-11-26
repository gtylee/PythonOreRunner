#!/usr/bin/env python3
"""Run each ORE portfolio trade through Python LGM without SWIG fallback."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for source_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from pythonore.io.loader import XVALoader
from pythonore.runtime.runtime import PythonLgmAdapter, XVAEngine


FIELDS = (
    "trade_id",
    "trade_type",
    "product",
    "status",
    "pv",
    "peak_epe",
    "cva",
    "python_trades",
    "fallback_trades",
    "input_fallbacks",
    "model_params_source",
    "market_source",
    "grid_source",
    "model_currency",
    "error",
)


def _product_name(trade: Any) -> str:
    product = trade.product
    payload = getattr(product, "payload", None)
    if isinstance(payload, dict):
        return str(payload.get("trade_type") or type(product).__name__)
    return type(product).__name__


def _single_trade_snapshot(snapshot: Any, trade: Any, paths: int, rng: str) -> Any:
    params = {
        **snapshot.config.params,
        "python.lgm_rng_mode": rng,
        "python.torch_device": "numpy",
        "python.progress": "N",
    }
    return replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=paths,
            analytics=("CVA",),
            params=params,
        ),
    )


def _run_trade(
    snapshot: Any,
    trade: Any,
    paths: int,
    rng: str,
    engine: XVAEngine,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trade_id": trade.trade_id,
        "trade_type": trade.trade_type,
        "product": _product_name(trade),
        "status": "FAIL",
        "pv": None,
        "peak_epe": None,
        "cva": None,
        "python_trades": 0,
        "fallback_trades": 0,
        "input_fallbacks": "",
        "model_params_source": "",
        "market_source": "",
        "grid_source": "",
        "model_currency": "",
        "error": "",
    }
    try:
        result = engine.create_session(_single_trade_snapshot(snapshot, trade, paths, rng)).run(
            return_cubes=False
        )
        coverage = result.metadata.get("coverage", {})
        exposure_rows = result.reports.get("exposure", [])
        peak_epe = max((float(item.get("EPE", 0.0)) for item in exposure_rows), default=0.0)
        pv = float(result.pv_total)
        cva = float(result.xva_by_metric.get("CVA", 0.0))
        input_fallbacks = list(result.metadata.get("input_fallbacks", []))
        provenance = result.metadata.get("input_provenance", {})
        row.update(
            pv=pv,
            peak_epe=peak_epe,
            cva=cva,
            python_trades=int(coverage.get("python_trades", 0)),
            fallback_trades=int(coverage.get("fallback_trades", 0)),
            input_fallbacks=";".join(map(str, input_fallbacks)),
            model_params_source=provenance.get("model_params", ""),
            market_source=provenance.get("market", ""),
            grid_source=provenance.get("grid", ""),
            model_currency=result.metadata.get("model_currency", ""),
        )
        if (
            row["python_trades"] == 1
            and row["fallback_trades"] == 0
            and not coverage.get("unsupported")
            and all(math.isfinite(value) for value in (pv, peak_epe, cva))
        ):
            row["status"] = "CHECK_INPUT_FALLBACK" if input_fallbacks else "PASS"
        else:
            row["error"] = "; ".join(map(str, coverage.get("unsupported", []))) or "non-native or non-finite result"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run each trade in an ORE Input portfolio separately with PythonLgmAdapter "
            "and SWIG fallback disabled. Results are execution coverage, not ORE parity."
        )
    )
    parser.add_argument("input_dir", type=Path, help="ORE Input directory containing ore.xml")
    parser.add_argument("--ore-file", default="ore.xml")
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument(
        "--rng",
        choices=("numpy", "ore_parity", "ore_sobol_bridge", "ore_sobol_direct"),
        default="ore_parity",
        help="RNG mode; direct Sobol only supports single-factor, base-currency IR-only cases",
    )
    parser.add_argument("--output", type=Path, help="Write a CSV report (refuses to overwrite an existing file)")
    args = parser.parse_args()

    if args.paths <= 0:
        parser.error("--paths must be positive")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    snapshot = XVALoader.from_files(str(args.input_dir), ore_file=args.ore_file)
    # Reuse the adapter so this matrix also exercises cache isolation across trades.
    engine = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False))
    rows = [
        _run_trade(snapshot, trade, args.paths, args.rng, engine)
        for trade in snapshot.portfolio.trades
    ]

    print("Native-only execution matrix (not an ORE parity report)")
    print(f"Trades: {len(rows)} | paths per trade: {args.paths} | RNG: {args.rng}")
    print("status | trade_id | product | PV | peak EPE | CVA")
    for row in rows:
        print(
            f"{row['status']} | {row['trade_id']} | {row['product']} | "
            f"{row['pv']} | {row['peak_epe']} | {row['cva']}"
        )
        if row["error"] or row["input_fallbacks"]:
            print(f"  detail: {row['error'] or row['input_fallbacks']}")

    counts = {status: sum(row["status"] == status for row in rows) for status in ("PASS", "CHECK_INPUT_FALLBACK", "FAIL")}
    print("Summary:", ", ".join(f"{key}={value}" for key, value in counts.items()))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {args.output}")

    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
