#!/usr/bin/env python3
"""Generate and run a USD SOFR basis-swap ORE snapshot example.

This example uses the modern Products input bundle so that USD-SOFR curves,
conventions and market mappings are available.

Generated trades:

- USD-SOFR-3M vs USD-LIBOR-3M
- USD-SOFR-3M vs USD-SIFMA

It defaults to one trade of each type and runs pricing plus XVA.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pythonore.apps import ore_snapshot_cli


PRODUCTS_INPUT = REPO_ROOT / "Examples" / "Products" / "Input"
DEFAULT_CASE_ROOT = REPO_ROOT / "Examples" / "Generated" / "USD_SOFRBasisSnapshot"
ASOF_DATE = "2025-02-10"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a USD SOFR basis-swap ORE snapshot case and optionally price it with the snapshot CLI."
    )
    parser.add_argument("--count-per-type", type=int, default=1)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--price-only", action="store_true")
    parser.add_argument(
        "--lgm-param-source",
        choices=("auto", "calibration_xml", "simulation_xml", "ore"),
        default="auto",
    )
    return parser.parse_args()


def _ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace the generated case.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _trade_blocks(count_per_type: int) -> list[str]:
    blocks: list[str] = []
    for i in range(1, count_per_type + 1):
        suffix = f"{i:03d}"
        blocks.extend([_sofr3m_libor3m_trade_xml(suffix), _sofr3m_sifma_trade_xml(suffix)])
    return blocks


def _sofr3m_libor3m_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="BASIS_USD_SOFR3M_LIB3M_{suffix}">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR-3M</Index>
          <Spreads>
            <Spread>0.0000</Spread>
          </Spreads>
          <IsInArrears>true</IsInArrears>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2025-02-10</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0010</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2025-02-10</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _sofr3m_sifma_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="BASIS_USD_SOFR3M_SIFMA_{suffix}">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR-3M</Index>
          <Spreads>
            <Spread>0.0005</Spread>
          </Spreads>
          <IsInArrears>true</IsInArrears>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2025-02-10</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>ACT/ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SIFMA</Index>
          <Spreads>
            <Spread>0.0005</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>1</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2025-02-10</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>US-NYSE</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _portfolio_xml(count_per_type: int) -> str:
    trades = "\n".join(_trade_blocks(count_per_type))
    return f"""<?xml version="1.0"?>
<Portfolio>
{trades}
</Portfolio>
"""


def _simulation_xml() -> str:
    return """<?xml version="1.0"?>
<Simulation>
  <Parameters>
    <Discretization>Exact</Discretization>
    <Grid>81,6M</Grid>
    <Calendar>USD</Calendar>
    <Sequence>SobolBrownianBridge</Sequence>
    <Scenario>Simple</Scenario>
    <Seed>42</Seed>
    <Samples>256</Samples>
  </Parameters>
  <CrossAssetModel>
    <DomesticCcy>USD</DomesticCcy>
    <Currencies>
      <Currency>USD</Currency>
    </Currencies>
    <BootstrapTolerance>0.0001</BootstrapTolerance>
    <InterestRateModels>
      <LGM ccy="USD">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility>
          <Calibrate>Y</Calibrate>
          <VolatilityType>Hagan</VolatilityType>
          <ParamType>Piecewise</ParamType>
          <TimeGrid>1.0,2.0,3.0,5.0,10.0</TimeGrid>
          <InitialValue>0.01,0.01,0.01,0.01,0.01,0.01</InitialValue>
        </Volatility>
        <Reversion>
          <Calibrate>N</Calibrate>
          <ReversionType>HullWhite</ReversionType>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>0.03</InitialValue>
        </Reversion>
        <CalibrationSwaptions>
          <Expiries>6M,1Y,2Y,3Y,5Y,10Y</Expiries>
          <Terms>1Y,2Y,3Y,5Y,10Y,30Y</Terms>
          <Strikes/>
        </CalibrationSwaptions>
        <ParameterTransformation>
          <ShiftHorizon>0.0</ShiftHorizon>
          <Scaling>1.0</Scaling>
        </ParameterTransformation>
      </LGM>
    </InterestRateModels>
    <ForeignExchangeModels/>
    <InstantaneousCorrelations/>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>USD</BaseCurrency>
    <Currencies>
      <Currency>USD</Currency>
    </Currencies>
    <YieldCurves>
      <Configuration>
        <Tenors>3M,6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y</Tenors>
        <Interpolation>LogLinear</Interpolation>
        <Extrapolation>Y</Extrapolation>
      </Configuration>
    </YieldCurves>
    <Indices>
      <Index>USD-SOFR</Index>
      <Index>USD-SOFR-3M</Index>
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-SIFMA</Index>
    </Indices>
    <DefaultCurves>
      <Names/>
      <Tenors>6M,1Y,2Y</Tenors>
    </DefaultCurves>
    <SwaptionVolatilities>
      <ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay>
      <Currencies>
        <Currency>USD</Currency>
      </Currencies>
      <Expiries>6M,1Y,2Y,3Y,5Y,10Y</Expiries>
      <Terms>1Y,2Y,3Y,5Y,10Y,30Y</Terms>
    </SwaptionVolatilities>
    <AggregationScenarioDataCurrencies>
      <Currency>USD</Currency>
    </AggregationScenarioDataCurrencies>
    <AggregationScenarioDataIndices>
      <Index>USD-SOFR</Index>
      <Index>USD-SOFR-3M</Index>
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-SIFMA</Index>
    </AggregationScenarioDataIndices>
  </Market>
</Simulation>
"""


def _ore_xml() -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">{ASOF_DATE}</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output</Parameter>
    <Parameter name="logFile">log.txt</Parameter>
    <Parameter name="logMask">31</Parameter>
    <Parameter name="marketDataFile">{PRODUCTS_INPUT / "marketdata.csv"}</Parameter>
    <Parameter name="fixingDataFile">{PRODUCTS_INPUT / "fixings.csv"}</Parameter>
    <Parameter name="implyTodaysFixings">N</Parameter>
    <Parameter name="curveConfigFile">{PRODUCTS_INPUT / "curveconfig.xml"}</Parameter>
    <Parameter name="conventionsFile">{PRODUCTS_INPUT / "conventions.xml"}</Parameter>
    <Parameter name="marketConfigFile">{PRODUCTS_INPUT / "todaysmarket.xml"}</Parameter>
    <Parameter name="pricingEnginesFile">{PRODUCTS_INPUT / "pricingengine.xml"}</Parameter>
    <Parameter name="portfolioFile">portfolio_usd_sofr_basis_swaps.xml</Parameter>
    <Parameter name="referenceDataFile">{PRODUCTS_INPUT / "referencedata.xml"}</Parameter>
    <Parameter name="scriptLibrary">{PRODUCTS_INPUT / "scriptlibrary.xml"}</Parameter>
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
    <Analytic type="xva">
      <Parameter name="active">N</Parameter>
      <Parameter name="csaFile">netting.xml</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def _write_files(case_root: Path, *, count_per_type: int) -> tuple[Path, Path]:
    input_dir = case_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    ore_xml = input_dir / "ore.xml"
    portfolio_xml = input_dir / "portfolio_usd_sofr_basis_swaps.xml"
    simulation_xml = input_dir / "simulation.xml"
    netting_xml = input_dir / "netting.xml"

    ore_xml.write_text(_ore_xml(), encoding="utf-8")
    portfolio_xml.write_text(_portfolio_xml(count_per_type), encoding="utf-8")
    simulation_xml.write_text(_simulation_xml(), encoding="utf-8")
    netting_xml.write_text("<NettingSetDefinitions/>\n", encoding="utf-8")
    return ore_xml, portfolio_xml


def _run_cli(
    ore_xml: Path,
    artifact_root: Path,
    *,
    price_only: bool,
    paths: int,
    lgm_param_source: str,
) -> int:
    argv = [str(ore_xml), "--price"]
    if not price_only:
        argv.extend(["--xva", "--paths", str(paths)])
    argv.extend(["--lgm-param-source", str(lgm_param_source)])
    argv.extend(["--output-root", str(artifact_root)])
    return ore_snapshot_cli.main(argv)


def _case_slug(ore_xml: Path) -> str:
    parent = ore_xml.resolve().parents[1].name
    return parent or ore_xml.stem


def _product_counts(count_per_type: int) -> Iterable[tuple[str, int]]:
    yield ("USD-SOFR-3M vs USD-LIBOR-3M", count_per_type)
    yield ("USD-SOFR-3M vs USD-SIFMA", count_per_type)


def main() -> int:
    args = _parse_args()
    if args.count_per_type <= 0:
        raise ValueError("--count-per-type must be > 0")

    case_root = args.case_root.resolve()
    artifact_root = (args.artifact_root or (case_root / "artifacts")).resolve()

    _ensure_clean_dir(case_root, overwrite=args.overwrite)
    ore_xml, portfolio_xml = _write_files(case_root, count_per_type=args.count_per_type)

    print("Generated USD SOFR basis-swap snapshot example")
    print(f"  case_root      : {case_root}")
    print(f"  ore_xml        : {ore_xml}")
    print(f"  portfolio_xml  : {portfolio_xml}")
    print(f"  artifact_root  : {artifact_root}")
    print("  product_counts :")
    for name, count in _product_counts(args.count_per_type):
        print(f"    - {name}: {count}")
    print()
    print("Next scale-up:")
    print("  python3 example_ore_snapshot_usd_sofr_basis_swaps.py --count-per-type 100 --overwrite")
    if args.price_only:
        print("  run_mode       : price only")
    else:
        print(f"  run_mode       : price + xva (paths={args.paths})")
    print(f"  lgm_param_src  : {args.lgm_param_source}")

    if args.no_run:
        return 0

    print()
    if args.price_only:
        print("Running ore_snapshot_cli --price ...")
    else:
        print(f"Running ore_snapshot_cli --price --xva --paths {args.paths} ...")
    rc = _run_cli(
        ore_xml,
        artifact_root,
        price_only=args.price_only,
        paths=args.paths,
        lgm_param_source=args.lgm_param_source,
    )
    if rc == 0:
        case_dir = artifact_root / _case_slug(ore_xml)
        print()
        if args.price_only:
            print("Pricing run completed")
        else:
            print("Pricing and XVA run completed")
        print(f"  report_dir     : {case_dir}")
        print(f"  summary_json   : {case_dir / 'summary.json'}")
        print(f"  report_md      : {case_dir / 'report.md'}")
        print(f"  compare_csv    : {case_dir / 'comparison.csv'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
