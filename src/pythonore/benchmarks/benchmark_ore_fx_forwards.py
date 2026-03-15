#!/usr/bin/env python3
"""Run focused ORE benchmarks for demo FX forwards and compare with Python t0 formula."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

from pythonore.repo_paths import default_ore_bin, local_parity_artifacts_root, require_engine_repo_root

REPO_ROOT = require_engine_repo_root()
EXAMPLES_INPUT = REPO_ROOT / "Examples" / "Input"
ORE_BIN_DEFAULT = default_ore_bin()


@dataclass(frozen=True)
class FwdCase:
    case_id: str
    pair: str
    maturity_years: int
    bought_ccy: str
    sold_ccy: str
    notional_base: float
    strike: float
    spot0: float
    dom_zero_rate: float
    for_zero_rate: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ore-bin", type=Path, default=ORE_BIN_DEFAULT)
    p.add_argument(
        "--output-root",
        type=Path,
        default=local_parity_artifacts_root() / "fxfwd_ore_benchmark",
    )
    return p.parse_args()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _start_end(maturity_years: int) -> tuple[str, str]:
    start = date(2016, 3, 1)
    end = date(2016 + maturity_years, 3, 1)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    return start_s, end_s


def _curve_column(ccy: str) -> str:
    c = ccy.upper()
    if c == "USD":
        return "USD-LIBOR-3M"
    if c == "GBP":
        return "GBP-LIBOR-6M"
    if c == "EUR":
        return "EUR-EURIBOR-6M"
    if c == "CAD":
        return "CAD-CDOR-3M"
    raise ValueError(f"unsupported currency '{ccy}'")


def _discount_from_curves_csv(curves_csv: Path, col: str, t_years: float) -> float:
    with open(curves_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty curves csv: {curves_csv}")
    d0 = date.fromisoformat(rows[0]["Date"])
    xs = []
    ys = []
    for r in rows:
        d = date.fromisoformat(r["Date"])
        xs.append((d - d0).days / 365.0)
        ys.append(float(r[col]))
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    if t_years <= xa[0]:
        return float(ya[0])
    if t_years >= xa[-1]:
        return float(ya[-1])
    return float(np.interp(t_years, xa, ya))


def _spot_from_market(market_file: Path, base: str, quote: str, fallback: float) -> float:
    k1 = f"FX/RATE/{base}/{quote}"
    k2 = f"FX/RATE/{quote}/{base}"
    with open(market_file, encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 3:
                continue
            if p[1] == k1:
                return float(p[2])
            if p[1] == k2:
                return 1.0 / float(p[2])
    return float(fallback)


def _spot_from_marketdata_csv(marketdata_csv: Path, base: str, quote: str) -> float:
    k1 = f"FX/RATE/{base}/{quote}"
    k2 = f"FX/RATE/{quote}/{base}"
    quotes: dict[str, float] = {}
    with open(marketdata_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            key = row[1].strip()
            if key.startswith("FX/RATE/"):
                quotes[key] = float(row[2])
    if k1 in quotes:
        return quotes[k1]
    if k2 in quotes:
        return 1.0 / quotes[k2]

    # Triangulate through EUR if possible.
    eur_base = f"FX/RATE/EUR/{base}"
    eur_quote = f"FX/RATE/EUR/{quote}"
    if eur_base in quotes and eur_quote in quotes and quotes[eur_base] != 0.0:
        return quotes[eur_quote] / quotes[eur_base]
    base_eur = f"FX/RATE/{base}/EUR"
    quote_eur = f"FX/RATE/{quote}/EUR"
    if base_eur in quotes and quote_eur in quotes and quotes[quote_eur] != 0.0:
        return quotes[base_eur] / quotes[quote_eur]
    return float("nan")
    return start, end


def _portfolio_xml(case: FwdCase) -> str:
    _, end = _start_end(case.maturity_years)
    sold_amt = case.notional_base * case.strike
    return f"""<?xml version=\"1.0\"?>
<Portfolio>
  <Trade id=\"{case.case_id}\">
    <TradeType>FxForward</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <FxForwardData>
      <ValueDate>{end}</ValueDate>
      <BoughtCurrency>{case.bought_ccy}</BoughtCurrency>
      <BoughtAmount>{case.notional_base:.6f}</BoughtAmount>
      <SoldCurrency>{case.sold_ccy}</SoldCurrency>
      <SoldAmount>{sold_amt:.6f}</SoldAmount>
    </FxForwardData>
  </Trade>
</Portfolio>
"""


def _ore_xml(output_dir: Path, market_file: Path, tm_file: Path, portfolio_file: Path, base_ccy: str) -> str:
    return f"""<?xml version=\"1.0\"?>
<ORE>
  <Setup>
    <Parameter name=\"asofDate\">2016-02-05</Parameter>
    <Parameter name=\"inputPath\">{portfolio_file.parent.as_posix()}</Parameter>
    <Parameter name=\"outputPath\">{output_dir.as_posix()}</Parameter>
    <Parameter name=\"logFile\">log.txt</Parameter>
    <Parameter name=\"logMask\">31</Parameter>
    <Parameter name=\"marketDataFile\">{market_file.as_posix()}</Parameter>
    <Parameter name=\"fixingDataFile\">{(EXAMPLES_INPUT / "fixings_20160205.txt").as_posix()}</Parameter>
    <Parameter name=\"implyTodaysFixings\">Y</Parameter>
    <Parameter name=\"curveConfigFile\">{(EXAMPLES_INPUT / "curveconfig.xml").as_posix()}</Parameter>
    <Parameter name=\"conventionsFile\">{(EXAMPLES_INPUT / "conventions.xml").as_posix()}</Parameter>
    <Parameter name=\"marketConfigFile\">{tm_file.as_posix()}</Parameter>
    <Parameter name=\"pricingEnginesFile\">{(EXAMPLES_INPUT / "pricingengine.xml").as_posix()}</Parameter>
    <Parameter name=\"portfolioFile\">{portfolio_file.as_posix()}</Parameter>
    <Parameter name=\"observationModel\">None</Parameter>
    <Parameter name=\"continueOnError\">false</Parameter>
    <Parameter name=\"calendarAdjustment\">{(EXAMPLES_INPUT / "calendaradjustment.xml").as_posix()}</Parameter>
  </Setup>
  <Markets>
    <Parameter name=\"pricing\">libor</Parameter>
  </Markets>
  <Analytics>
    <Analytic type=\"npv\">
      <Parameter name=\"active\">Y</Parameter>
      <Parameter name=\"baseCurrency\">{base_ccy}</Parameter>
      <Parameter name=\"outputFileName\">npv.csv</Parameter>
    </Analytic>
    <Analytic type=\"curves\">
      <Parameter name=\"active\">Y</Parameter>
      <Parameter name=\"configuration\">default</Parameter>
      <Parameter name=\"grid\">240,1M</Parameter>
      <Parameter name=\"outputFileName\">curves.csv</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def _load_ore_trade_npv(npv_csv: Path, trade_id: str) -> float:
    with open(npv_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get(tid_key, "") == trade_id:
                return float(row["NPV"])
    raise ValueError(f"trade '{trade_id}' not found in {npv_csv}")


def _python_t0_npv(case: FwdCase, curves_csv: Path, market_file: Path, marketdata_csv: Path) -> float:
    T = float(case.maturity_years)
    p_dom = _discount_from_curves_csv(curves_csv, _curve_column(case.sold_ccy), T)
    p_for = _discount_from_curves_csv(curves_csv, _curve_column(case.bought_ccy), T)
    spot = _spot_from_marketdata_csv(marketdata_csv, case.bought_ccy, case.sold_ccy)
    if not np.isfinite(spot) or spot <= 0.0:
        spot = _spot_from_market(market_file, case.bought_ccy, case.sold_ccy, fallback=case.spot0)
    fwd = spot * p_for / p_dom
    return case.notional_base * p_dom * (fwd - case.strike)


def _market_files_for_case(case: FwdCase) -> tuple[Path, Path]:
    # Reuse CAD-augmented files if available for USD/CAD.
    if "CAD" in (case.bought_ccy, case.sold_ccy):
        cad_tm = REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "multiccy_benchmark_final" / "shared" / "todaysmarket_with_cad.xml"
        cad_mkt = REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "multiccy_benchmark_final" / "shared" / "market_20160205_flat_with_cad.txt"
        if cad_tm.exists() and cad_mkt.exists():
            return cad_mkt, cad_tm
    return EXAMPLES_INPUT / "market_20160205_flat.txt", EXAMPLES_INPUT / "todaysmarket.xml"


def main() -> None:
    args = _parse_args()
    if not args.ore_bin.exists():
        raise FileNotFoundError(args.ore_bin)

    cases = [
        FwdCase(
            case_id="FXFWD_GBPUSD_1Y",
            pair="GBP/USD",
            maturity_years=1,
            bought_ccy="GBP",
            sold_ccy="USD",
            notional_base=10_000_000.0,
            strike=1.2850,
            spot0=1.2700,
            dom_zero_rate=0.0475,
            for_zero_rate=0.0400,
        ),
        FwdCase(
            case_id="FXFWD_USDCAD_2Y",
            pair="USD/CAD",
            maturity_years=2,
            bought_ccy="USD",
            sold_ccy="CAD",
            notional_base=10_000_000.0,
            strike=1.3600,
            spot0=1.3400,
            dom_zero_rate=0.0360,
            for_zero_rate=0.0475,
        ),
    ]

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        case_dir = out_root / case.case_id
        inp = case_dir / "Input"
        out = case_dir / "Output"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)

        market_file, tm_file = _market_files_for_case(case)
        portfolio = inp / "portfolio.xml"
        ore_xml = inp / "ore.xml"
        _write(portfolio, _portfolio_xml(case))
        _write(ore_xml, _ore_xml(out, market_file, tm_file, portfolio, base_ccy=case.sold_ccy))

        t0 = perf_counter()
        cp = subprocess.run([str(args.ore_bin), str(ore_xml)], cwd=str(case_dir), capture_output=True, text=True, check=False)
        ore_sec = perf_counter() - t0

        status = "ok" if cp.returncode == 0 else "ore_error"
        ore_npv = float("nan")
        py_npv = float("nan")
        abs_diff = float("nan")
        rel_diff = float("nan")
        (case_dir / "ore_stdout.log").write_text(cp.stdout, encoding="utf-8")
        (case_dir / "ore_stderr.log").write_text(cp.stderr, encoding="utf-8")

        npv_csv = out / "npv.csv"
        if status == "ok" and npv_csv.exists():
            npv_csv = out / "npv.csv"
            ore_npv = _load_ore_trade_npv(npv_csv, case.case_id)
            py_npv = _python_t0_npv(case, out / "curves.csv", market_file, out / "marketdata.csv")
            abs_diff = py_npv - ore_npv
            rel_diff = abs_diff / ore_npv if abs(ore_npv) > 1.0e-12 else float("inf")
        elif status == "ok":
            status = "ore_missing_npv"

        rows.append(
            {
                "case_id": case.case_id,
                "pair": case.pair,
                "status": status,
                "ore_seconds": ore_sec,
                "ore_npv": ore_npv,
                "python_npv": py_npv,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "output_dir": str(out),
            }
        )

    (out_root / "benchmark_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with open(out_root / "benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "pair", "status", "ore_seconds", "ore_npv", "python_npv", "abs_diff", "rel_diff", "output_dir"])
        for r in rows:
            w.writerow([r[k] for k in ["case_id", "pair", "status", "ore_seconds", "ore_npv", "python_npv", "abs_diff", "rel_diff", "output_dir"]])

    for r in rows:
        print(
            f"{r['case_id']} ({r['pair']}): status={r['status']}, "
            f"ORE NPV={r['ore_npv']}, PY NPV={r['python_npv']}, rel_diff={r['rel_diff']}, t={r['ore_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
