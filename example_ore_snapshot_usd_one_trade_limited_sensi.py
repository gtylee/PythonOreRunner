#!/usr/bin/env python3
"""Generate and run a one-trade USD snapshot case with a tiny sensitivity grid."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pythonore.apps import ore_snapshot_cli
from example_ore_snapshot_usd_all_rates_products import _simulation_xml as _full_simulation_xml


PRODUCTS_INPUT = REPO_ROOT / "Examples" / "Products" / "Input"
DEFAULT_CASE_ROOT = REPO_ROOT / "Examples" / "Generated" / "USD_OneTradeLimitedSensi"
ASOF_DATE = "2025-02-10"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a one-trade USD case with a tiny sensitivity grid.")
    parser.add_argument("--trade-count", type=int, default=300)
    parser.add_argument("--sensi-tenors", default="6M,1Y,18M,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y,25Y,30Y,35Y,40Y")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--paths", type=int, default=96)
    parser.add_argument(
        "--lgm-param-source",
        choices=("auto", "calibration_xml", "simulation_xml", "ore"),
        default="simulation_xml",
    )
    return parser.parse_args()


def _ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace the generated case.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _irs_trade_xml(idx: int) -> str:
    suffix = f"{idx:04d}"
    fixed_rate = 0.04 + 0.0005 * (idx - 1)
    return f"""  <Trade id="IRS_USD_{suffix}">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData><Rates><Rate>{fixed_rate:.6f}</Rate></Rates></FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads><Spread>0.0000</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _portfolio_xml(trade_count: int) -> str:
    trades = "\n".join(_irs_trade_xml(i) for i in range(1, trade_count + 1))
    return f"<Portfolio>\n{trades}\n</Portfolio>\n"


def _ore_xml() -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">{ASOF_DATE}</Parameter>
    <Parameter name="baseCurrency">USD</Parameter>
    <Parameter name="marketDataFile">{PRODUCTS_INPUT / "marketdata.csv"}</Parameter>
    <Parameter name="fixingDataFile">{PRODUCTS_INPUT / "fixings.csv"}</Parameter>
    <Parameter name="curveConfigFile">{PRODUCTS_INPUT / "curveconfig.xml"}</Parameter>
    <Parameter name="conventionsFile">{PRODUCTS_INPUT / "conventions.xml"}</Parameter>
    <Parameter name="marketConfigFile">{PRODUCTS_INPUT / "todaysmarket.xml"}</Parameter>
    <Parameter name="pricingEnginesFile">{PRODUCTS_INPUT / "pricingengine.xml"}</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
    <Parameter name="referenceDataFile">{PRODUCTS_INPUT / "referencedata.xml"}</Parameter>
    <Parameter name="scriptLibrary">{PRODUCTS_INPUT / "scriptlibrary.xml"}</Parameter>
    <Parameter name="outputPath">Output</Parameter>
    <Parameter name="observationModel">None</Parameter>
    <Parameter name="continueOnError">false</Parameter>
    <Parameter name="buildFailedTrades">true</Parameter>
  </Setup>
  <Markets>
    <Parameter name="lgmcalibration">default</Parameter>
    <Parameter name="fxcalibration">default</Parameter>
    <Parameter name="pricing">default</Parameter>
    <Parameter name="simulation">default</Parameter>
  </Markets>
  <Analytics>
    <Analytic type="npv">
      <Parameter name="active">Y</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
      <Parameter name="outputFileName">npv.csv</Parameter>
    </Analytic>
    <Analytic type="cashflow">
      <Parameter name="active">Y</Parameter>
      <Parameter name="outputFileName">flows.csv</Parameter>
    </Analytic>
    <Analytic type="curves">
      <Parameter name="active">Y</Parameter>
      <Parameter name="configuration">default</Parameter>
      <Parameter name="grid">240,1M</Parameter>
      <Parameter name="outputFileName">curves.csv</Parameter>
    </Analytic>
    <Analytic type="simulation">
      <Parameter name="active">N</Parameter>
      <Parameter name="simulationConfigFile">simulation.xml</Parameter>
      <Parameter name="pricingEnginesFile">{PRODUCTS_INPUT / "pricingengine.xml"}</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
    </Analytic>
    <Analytic type="sensitivity">
      <Parameter name="active">Y</Parameter>
      <Parameter name="sensitivityConfigFile">sensitivity.xml</Parameter>
      <Parameter name="sensitivityOutputFile">sensitivity.csv</Parameter>
      <Parameter name="scenarioOutputFile">scenario.csv</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def _sensitivity_xml(sensi_tenors: str) -> str:
    return """<?xml version="1.0"?>
<SensitivityAnalysis>
  <DiscountCurves>
    <DiscountCurve ccy="USD">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </DiscountCurve>
  </DiscountCurves>
  <IndexCurves>
    <IndexCurve index="USD-LIBOR-3M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </IndexCurve>
  </IndexCurves>
</SensitivityAnalysis>
""".format(tenors=sensi_tenors)


def _write_case(case_root: Path, *, trade_count: int, sensi_tenors: str) -> Path:
    input_dir = case_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "ore.xml").write_text(_ore_xml(), encoding="utf-8")
    (input_dir / "portfolio.xml").write_text(_portfolio_xml(trade_count), encoding="utf-8")
    (input_dir / "simulation.xml").write_text(_full_simulation_xml(), encoding="utf-8")
    (input_dir / "netting.xml").write_text("<NettingSetDefinitions/>\n", encoding="utf-8")
    (input_dir / "sensitivity.xml").write_text(_sensitivity_xml(sensi_tenors), encoding="utf-8")
    return input_dir / "ore.xml"


def main() -> int:
    args = _parse_args()
    case_root = args.case_root.resolve()
    artifact_root = (args.artifact_root or (case_root / "artifacts")).resolve()
    _ensure_clean_dir(case_root, overwrite=args.overwrite)
    if args.trade_count <= 0:
        raise ValueError("--trade-count must be > 0")
    ore_xml = _write_case(case_root, trade_count=args.trade_count, sensi_tenors=args.sensi_tenors)
    print(f"case_root      : {case_root}")
    print(f"ore_xml        : {ore_xml}")
    print(f"artifact_root  : {artifact_root}")
    print(f"trade_count    : {args.trade_count}")
    tenor_count = len([t for t in str(args.sensi_tenors).split(",") if t.strip()])
    print(f"sensi_factors  : {2 * tenor_count} (USD discount + USD-LIBOR-3M across {args.sensi_tenors})")
    argv = [
        str(ore_xml),
        "--price",
        "--sensi",
        "--sensi-progress",
        "--sensi-metric",
        "NPV",
        "--paths",
        str(args.paths),
        "--lgm-param-source",
        str(args.lgm_param_source),
        "--skip-input-validation",
        "--output-root",
        str(artifact_root),
    ]
    return ore_snapshot_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
