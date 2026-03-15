#!/usr/bin/env python3
"""Fast Bermudan grid-pricing benchmark: ORE grid PV vs Python backward PV."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Sequence

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

from pythonore.benchmarks.benchmark_ore_ir_options import (
    BERM_BACKWARD_N_GRID,
    BERM_TRADE_SPECS,
    _berm_trade_xml,
    _build_berm_from_ore_flows,
    _build_model_from_ore_calibration,
    _build_simulation_stub_model,
    _ore_classic_calibration_xml,
    _simulation_classic_xml,
    _write,
)
from pythonore.compute.lgm import LGM1F, LGMParams
from pythonore.compute.lgm_ir_options import bermudan_backward_price
from pythonore.compute.irs_xva_utils import (
    build_discount_curve_from_discount_pairs,
    load_ore_discount_pairs_from_curves,
)
from pythonore.repo_paths import default_ore_bin, local_parity_artifacts_root, require_engine_repo_root

REPO_ROOT = require_engine_repo_root()
EXAMPLES_INPUT = REPO_ROOT / "Examples" / "Input"
EXPOSURE_INPUT = REPO_ROOT / "Examples" / "Exposure" / "Input"
ORE_BIN_DEFAULT = default_ore_bin()


@dataclass(frozen=True)
class GridRow:
    trade_id: str
    ore_pv: float
    py_pv_same_curve: float
    py_pv_main: float
    model_source: str


@dataclass(frozen=True)
class CurveRun:
    discount_column: str
    forward_column: str
    ore_market: str
    curves_csv: Path
    flows_csv: Path
    npv_csv: Path
    calibration_xml: Path


def _parse_curve_pair(spec: str) -> tuple[str, str]:
    left, sep, right = spec.partition(":")
    if not sep or not left.strip() or not right.strip():
        raise argparse.ArgumentTypeError("curve pair must be DISCOUNT:FORWARD")
    return left.strip(), right.strip()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ore-bin", type=Path, default=ORE_BIN_DEFAULT)
    p.add_argument(
        "--output-root",
        type=Path,
        default=local_parity_artifacts_root() / "ir_grid_ore_benchmark",
    )
    p.add_argument("--discount-column", default="EUR-EONIA")
    p.add_argument("--forward-column", default="EUR-EURIBOR-6M")
    p.add_argument(
        "--curve-pair",
        dest="curve_pairs",
        action="append",
        type=_parse_curve_pair,
        default=[],
        help="Additional curve pair to compare, formatted as DISCOUNT:FORWARD. May be repeated.",
    )
    p.add_argument("--model-source", choices=("classic_calibration", "simulation_stub"), default="classic_calibration")
    p.add_argument("--alpha-scale", type=float, default=1.0)
    p.add_argument("--kappa-override", type=float, default=None)
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON payload to stdout as well.")
    return p.parse_args(argv)


def _fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.3%}"


def _make_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([line, header_line, line, *body, line])


def _berm_portfolio_xml() -> str:
    berms_xml = "".join(_berm_trade_xml(str(spec["trade_id"]), float(spec["fixed_rate"])) for spec in BERM_TRADE_SPECS)
    return f"""<?xml version="1.0"?>
<Portfolio>
{berms_xml}
</Portfolio>
"""


def _ore_pricing_only_xml(output_dir: Path, input_dir: Path, *, pricing_market: str) -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">2016-02-05</Parameter>
    <Parameter name="inputPath">{input_dir.as_posix()}</Parameter>
    <Parameter name="outputPath">{output_dir.as_posix()}</Parameter>
    <Parameter name="logFile">log.txt</Parameter>
    <Parameter name="logMask">31</Parameter>
    <Parameter name="marketDataFile">{(EXAMPLES_INPUT / "market_20160205_flat.txt").as_posix()}</Parameter>
    <Parameter name="fixingDataFile">{(EXAMPLES_INPUT / "fixings_20160205.txt").as_posix()}</Parameter>
    <Parameter name="implyTodaysFixings">Y</Parameter>
    <Parameter name="curveConfigFile">{(EXAMPLES_INPUT / "curveconfig.xml").as_posix()}</Parameter>
    <Parameter name="conventionsFile">{(EXAMPLES_INPUT / "conventions.xml").as_posix()}</Parameter>
    <Parameter name="marketConfigFile">{(EXAMPLES_INPUT / "todaysmarket.xml").as_posix()}</Parameter>
    <Parameter name="pricingEnginesFile">{(EXAMPLES_INPUT / "pricingengine.xml").as_posix()}</Parameter>
    <Parameter name="portfolioFile">{(input_dir / "portfolio.xml").as_posix()}</Parameter>
    <Parameter name="observationModel">None</Parameter>
    <Parameter name="continueOnError">false</Parameter>
    <Parameter name="calendarAdjustment">{(EXAMPLES_INPUT / "calendaradjustment.xml").as_posix()}</Parameter>
  </Setup>
  <Markets>
    <Parameter name="lgmcalibration">{pricing_market}</Parameter>
    <Parameter name="pricing">{pricing_market}</Parameter>
  </Markets>
  <Analytics>
    <Analytic type="npv"><Parameter name="active">Y</Parameter><Parameter name="baseCurrency">EUR</Parameter><Parameter name="outputFileName">npv.csv</Parameter></Analytic>
    <Analytic type="cashflow"><Parameter name="active">Y</Parameter><Parameter name="outputFileName">flows.csv</Parameter></Analytic>
    <Analytic type="curves"><Parameter name="active">Y</Parameter><Parameter name="configuration">default</Parameter><Parameter name="grid">240,1M</Parameter><Parameter name="outputFileName">curves.csv</Parameter></Analytic>
  </Analytics>
</ORE>
"""


def _load_trade_npv(npv_csv: Path, trade_id: str) -> float:
    with open(npv_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get(tid_key, "") == trade_id:
                return float(row["NPV"])
    raise ValueError(f"trade {trade_id} not found in {npv_csv}")


def _render_report(rows: list[GridRow], ore_seconds: float, *, discount_column: str, forward_column: str, ore_market: str, model_source: str, alpha_scale: float, kappa_override: float | None) -> str:
    table_rows = [
        (
            row.trade_id,
            _fmt_num(row.ore_pv),
            _fmt_num(row.py_pv_main),
            _fmt_pct((row.py_pv_main - row.ore_pv) / max(abs(row.ore_pv), 1.0)),
            _fmt_num(row.py_pv_same_curve),
            _fmt_pct((row.py_pv_same_curve - row.ore_pv) / max(abs(row.ore_pv), 1.0)),
            row.model_source,
        )
        for row in rows
    ]
    lines = [
        "ORE IR Bermudan Grid Benchmark",
        "=" * 29,
        f"ORE pricing runtime: {ore_seconds:.3f}s",
        f"Python backward grid: n_grid={BERM_BACKWARD_N_GRID}",
        f"Discount curve: {discount_column}",
        f"Forward curve: {forward_column}",
        f"ORE market: {ore_market}",
        f"Base model source: {model_source}",
        f"Alpha scale: {alpha_scale:.6f}",
        f"Kappa override: {kappa_override if kappa_override is not None else 'calibration'}",
        "",
        _make_table(
            ["trade_id", "ORE_PV", "PY_main", "main_rel", "PY_same_curve", "same_curve_rel", "model_source"],
            table_rows,
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _effective_curve_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    pairs = [(str(args.discount_column), str(args.forward_column))]
    for pair in args.curve_pairs:
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _ore_market_for_curve_pair(discount_column: str, forward_column: str) -> str:
    pair = (str(discount_column), str(forward_column))
    if pair == ("EUR-EONIA", "EUR-EURIBOR-6M"):
        return "default"
    if pair == ("EUR-EURIBOR-6M", "EUR-EURIBOR-6M"):
        return "libor"
    raise ValueError(
        f"curve pair {discount_column}:{forward_column} is not a valid ORE benchmark market for this trade; "
        "supported benchmark pairs are EUR-EONIA:EUR-EURIBOR-6M and EUR-EURIBOR-6M:EUR-EURIBOR-6M"
    )


def main() -> None:
    args = _parse_args()
    if not args.ore_bin.exists():
        raise FileNotFoundError(args.ore_bin)

    out_root = args.output_root
    curve_pairs = _effective_curve_pairs(args)
    curve_runs: list[CurveRun] = []
    ore_seconds_total = 0.0

    for discount_column, forward_column in curve_pairs:
        ore_market = _ore_market_for_curve_pair(discount_column, forward_column)
        run_root = out_root / f"market_{ore_market}"
        inp = run_root / "Input"
        out = run_root / "Output"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)

        _write(inp / "portfolio.xml", _berm_portfolio_xml())
        _write(inp / "simulation_classic.xml", _simulation_classic_xml(256, 42))
        _write(inp / "netting.xml", (EXPOSURE_INPUT / "netting.xml").read_text(encoding="utf-8"))
        _write(inp / "ore.xml", _ore_pricing_only_xml(out, inp, pricing_market=ore_market))
        _write(inp / "ore_classic_calibration.xml", _ore_classic_calibration_xml(out, inp, inp / "simulation_classic.xml"))

        t0 = perf_counter()
        proc = subprocess.run([args.ore_bin.as_posix(), (inp / "ore.xml").as_posix()], capture_output=True, text=True)
        ore_seconds_total += perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"ORE pricing-only run failed for market {ore_market}:\n{proc.stdout}\n{proc.stderr}")

        classic_proc = subprocess.run(
            [args.ore_bin.as_posix(), (inp / "ore_classic_calibration.xml").as_posix()],
            capture_output=True,
            text=True,
        )
        if classic_proc.returncode != 0:
            raise RuntimeError(f"ORE classic calibration run failed for market {ore_market}:\n{classic_proc.stdout}\n{classic_proc.stderr}")

        curves_csv = out / "curves.csv"
        flows_csv = out / "flows.csv"
        npv_csv = out / "npv.csv"
        calibration_xml = out / "classic" / "calibration.xml"
        if not (curves_csv.exists() and flows_csv.exists() and npv_csv.exists() and calibration_xml.exists()):
            raise FileNotFoundError(f"expected pricing/calibration outputs missing for market {ore_market}")
        curve_runs.append(
            CurveRun(
                discount_column=discount_column,
                forward_column=forward_column,
                ore_market=ore_market,
                curves_csv=curves_csv,
                flows_csv=flows_csv,
                npv_csv=npv_csv,
                calibration_xml=calibration_xml,
            )
        )

    effective_model_source = args.model_source
    if args.alpha_scale != 1.0 or args.kappa_override is not None:
        effective_model_source = f"{effective_model_source}_scaled"
    report_runs: list[dict[str, object]] = []
    for run in curve_runs:
        base_model = (
            _build_model_from_ore_calibration(run.calibration_xml, ccy_key="EUR")
            if args.model_source == "classic_calibration"
            else _build_simulation_stub_model()
        )
        kappa_value = float(base_model.params.kappa_values[0]) if len(base_model.params.kappa_values) else 0.0
        if args.kappa_override is not None:
            kappa_value = float(args.kappa_override)
        model = LGM1F(
            LGMParams(
                alpha_times=tuple(float(x) for x in base_model.params.alpha_times),
                alpha_values=tuple(float(x) * float(args.alpha_scale) for x in base_model.params.alpha_values),
                kappa_times=tuple(float(x) for x in base_model.params.kappa_times),
                kappa_values=(kappa_value,),
                shift=float(base_model.params.shift),
                scaling=float(base_model.params.scaling),
            )
        )
        disc_t, disc_df = load_ore_discount_pairs_from_curves(run.curves_csv.as_posix(), discount_column=run.discount_column)
        p0_disc = build_discount_curve_from_discount_pairs(list(zip(disc_t.tolist(), disc_df.tolist())))
        fwd_t, fwd_df = load_ore_discount_pairs_from_curves(run.curves_csv.as_posix(), discount_column=run.forward_column)
        p0_fwd = build_discount_curve_from_discount_pairs(list(zip(fwd_t.tolist(), fwd_df.tolist())))

        rows: list[GridRow] = []
        for spec in BERM_TRADE_SPECS:
            trade_id = str(spec["trade_id"])
            berm = _build_berm_from_ore_flows(run.flows_csv, asof=date(2016, 2, 5), trade_id=trade_id)
            rows.append(
                GridRow(
                    trade_id=trade_id,
                    ore_pv=_load_trade_npv(run.npv_csv, trade_id),
                    py_pv_same_curve=float(bermudan_backward_price(model, p0_disc, p0_disc, berm, n_grid=BERM_BACKWARD_N_GRID).price),
                    py_pv_main=float(bermudan_backward_price(model, p0_disc, p0_fwd, berm, n_grid=BERM_BACKWARD_N_GRID).price),
                    model_source=effective_model_source,
                )
            )
        report_runs.append(
            {
                "discount_column": run.discount_column,
                "forward_column": run.forward_column,
                "ore_market": run.ore_market,
                "rows": rows,
            }
        )

    reports = [
        _render_report(
            run["rows"],
            ore_seconds_total,
            discount_column=str(run["discount_column"]),
            forward_column=str(run["forward_column"]),
            ore_market=str(run["ore_market"]),
            model_source=args.model_source,
            alpha_scale=float(args.alpha_scale),
            kappa_override=args.kappa_override,
        ).rstrip()
        for run in report_runs
    ]
    print("\n\n".join(reports) + "\n", end="")
    if args.json:
        print("Machine-readable JSON:")
        payload = {
            "ore_seconds": ore_seconds_total,
            "berm_backward_n_grid": BERM_BACKWARD_N_GRID,
            "model_source": args.model_source,
            "alpha_scale": float(args.alpha_scale),
            "kappa_override": args.kappa_override,
        }
        if len(report_runs) == 1:
            run = report_runs[0]
            payload.update(
                {
                    "discount_column": run["discount_column"],
                    "forward_column": run["forward_column"],
                    "ore_market": run["ore_market"],
                    "rows": [row.__dict__ for row in run["rows"]],
                }
            )
        else:
            payload["runs"] = [
                {
                    "discount_column": run["discount_column"],
                    "forward_column": run["forward_column"],
                    "ore_market": run["ore_market"],
                    "rows": [row.__dict__ for row in run["rows"]],
                }
                for run in report_runs
            ]
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
