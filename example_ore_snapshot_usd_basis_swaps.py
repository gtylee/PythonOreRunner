#!/usr/bin/env python3
"""Generate and run a USD floating-vs-floating ORE snapshot example.

This example writes a USD-only ORE input case with:

- USD-LIBOR-3M vs USD-LIBOR-6M tenor basis swaps
- USD-LIBOR-3M vs USD-SIFMA basis swaps

It defaults to one trade of each type so the first run stays easy to validate.
The run mode defaults to pricing plus XVA via ``ore_snapshot_cli``.

Example:
    python3 example_ore_snapshot_usd_basis_swaps.py
    python3 example_ore_snapshot_usd_basis_swaps.py --count-per-type 100
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


COMMON_INPUT = REPO_ROOT / "Examples" / "Input"
DEFAULT_CASE_ROOT = REPO_ROOT / "Examples" / "Generated" / "USD_BasisSwapsSnapshot"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a USD floating-vs-floating ORE snapshot case and optionally price it with the snapshot CLI."
    )
    parser.add_argument(
        "--count-per-type",
        type=int,
        default=1,
        help="Number of trades to generate for each product type. Defaults to 1.",
    )
    parser.add_argument(
        "--case-root",
        type=Path,
        default=DEFAULT_CASE_ROOT,
        help="Directory where the generated ORE case will be written.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Snapshot CLI output root. Defaults to <case-root>/artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the generated case directory if it already exists.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only generate the case files, do not run ore_snapshot_cli.",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=32,
        help="Python snapshot XVA path count passed to ore_snapshot_cli. Defaults to 32.",
    )
    parser.add_argument(
        "--price-only",
        action="store_true",
        help="Run pricing only. By default the script runs pricing plus XVA.",
    )
    parser.add_argument(
        "--include-fedfunds",
        action="store_true",
        help="Also include USD-FedFunds averaged vs USD-LIBOR-3M swaps.",
    )
    return parser.parse_args()


def _ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Re-run with --overwrite to replace the generated case."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _trade_blocks(count_per_type: int, *, include_fedfunds: bool) -> list[str]:
    blocks: list[str] = []
    for i in range(1, count_per_type + 1):
        suffix = f"{i:03d}"
        blocks.extend(
            [
                _libor_3m_6m_basis_trade_xml(suffix),
                _libor_3m_sifma_basis_trade_xml(suffix),
            ]
        )
        if include_fedfunds:
            blocks.append(_fedfunds_libor_basis_trade_xml(suffix))
    return blocks


def _libor_3m_6m_basis_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="BASIS_USD_LIB3M_LIB6M_{suffix}">
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
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-6M</Index>
          <Spreads>
            <Spread>0.0000</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20160209</StartDate>
            <EndDate>20260209</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>US</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
            <EndOfMonth/>
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
            <Spread>0.0015</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20160209</StartDate>
            <EndDate>20260209</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>US</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
            <EndOfMonth/>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _libor_3m_sifma_basis_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="BASIS_USD_LIB3M_SIFMA_{suffix}">
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
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads/>
          <Gearings>
            <Gearing>0.8</Gearing>
          </Gearings>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2016-02-08</StartDate>
            <EndDate>2026-02-08</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>US with Libor impact</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Backward</Rule>
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
            <StartDate>2016-02-08</StartDate>
            <EndDate>2026-02-08</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>US-NYSE</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Backward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _fedfunds_libor_basis_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="BASIS_USD_FEDFUNDS_LIB3M_{suffix}">
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
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-FedFunds</Index>
          <Spreads>
            <Spread>0.0002</Spread>
          </Spreads>
          <IsAveraged>true</IsAveraged>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20160209</StartDate>
            <EndDate>20210209</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>US</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Backward</Rule>
            <EndOfMonth>false</EndOfMonth>
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
            <StartDate>20160209</StartDate>
            <EndDate>20210209</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>US</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Backward</Rule>
            <EndOfMonth>false</EndOfMonth>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _portfolio_xml(count_per_type: int, *, include_fedfunds: bool) -> str:
    trades = "\n".join(_trade_blocks(count_per_type, include_fedfunds=include_fedfunds))
    return f"""<?xml version="1.0"?>
<Portfolio>
{trades}
</Portfolio>
"""


def _simulation_xml(*, include_fedfunds: bool) -> str:
    extra_indices = "      <Index>USD-FedFunds</Index>\n" if include_fedfunds else ""
    return f"""<?xml version="1.0"?>
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
          <Expiries>1Y,2Y,4Y,6Y,8Y,10Y,12Y,14Y,16Y,18Y,19Y</Expiries>
          <Terms>19Y,18Y,16Y,14Y,12Y,10Y,8Y,6Y,4Y,2Y,1Y</Terms>
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
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-LIBOR-6M</Index>
      <Index>USD-SIFMA</Index>
{extra_indices}    </Indices>
    <DefaultCurves>
      <Names/>
      <Tenors>6M,1Y,2Y</Tenors>
    </DefaultCurves>
    <SwaptionVolatilities>
      <ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay>
      <Currencies>
        <Currency>USD</Currency>
      </Currencies>
      <Expiries>6M,1Y,2Y,3Y,5Y,10Y,12Y,15Y,20Y</Expiries>
      <Terms>1Y,2Y,3Y,4Y,5Y,7Y,10Y,15Y,20Y,30Y</Terms>
    </SwaptionVolatilities>
    <AggregationScenarioDataCurrencies>
      <Currency>USD</Currency>
    </AggregationScenarioDataCurrencies>
    <AggregationScenarioDataIndices>
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-LIBOR-6M</Index>
      <Index>USD-SIFMA</Index>
{extra_indices}    </AggregationScenarioDataIndices>
  </Market>
</Simulation>
"""


def _ore_xml() -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">2016-02-05</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output</Parameter>
    <Parameter name="logFile">log.txt</Parameter>
    <Parameter name="logMask">31</Parameter>
    <Parameter name="marketDataFile">{COMMON_INPUT / "market_20160205_flat.txt"}</Parameter>
    <Parameter name="fixingDataFile">{COMMON_INPUT / "fixings_20160205.txt"}</Parameter>
    <Parameter name="implyTodaysFixings">Y</Parameter>
    <Parameter name="curveConfigFile">{COMMON_INPUT / "curveconfig.xml"}</Parameter>
    <Parameter name="conventionsFile">{COMMON_INPUT / "conventions.xml"}</Parameter>
    <Parameter name="marketConfigFile">{COMMON_INPUT / "todaysmarket.xml"}</Parameter>
    <Parameter name="pricingEnginesFile">{COMMON_INPUT / "pricingengine.xml"}</Parameter>
    <Parameter name="portfolioFile">portfolio_usd_basis_swaps.xml</Parameter>
    <Parameter name="observationModel">Disable</Parameter>
  </Setup>
  <Markets>
    <Parameter name="lgmcalibration">libor</Parameter>
    <Parameter name="fxcalibration">libor</Parameter>
    <Parameter name="eqcalibration">libor</Parameter>
    <Parameter name="pricing">libor</Parameter>
    <Parameter name="simulation">libor</Parameter>
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
      <Parameter name="pricingEnginesFile">{COMMON_INPUT / "pricingengine.xml"}</Parameter>
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


def _write_files(
    case_root: Path,
    *,
    count_per_type: int,
    include_fedfunds: bool,
) -> tuple[Path, Path]:
    input_dir = case_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    ore_xml = input_dir / "ore.xml"
    portfolio_xml = input_dir / "portfolio_usd_basis_swaps.xml"
    simulation_xml = input_dir / "simulation.xml"
    netting_xml = input_dir / "netting.xml"

    ore_xml.write_text(_ore_xml(), encoding="utf-8")
    portfolio_xml.write_text(
        _portfolio_xml(count_per_type, include_fedfunds=include_fedfunds),
        encoding="utf-8",
    )
    simulation_xml.write_text(_simulation_xml(include_fedfunds=include_fedfunds), encoding="utf-8")
    netting_xml.write_text("<NettingSetDefinitions/>\n", encoding="utf-8")

    return ore_xml, portfolio_xml


def _run_cli(ore_xml: Path, artifact_root: Path, *, price_only: bool, paths: int) -> int:
    argv = [str(ore_xml), "--price"]
    if not price_only:
        argv.extend(["--xva", "--paths", str(paths)])
    argv.extend(["--output-root", str(artifact_root)])
    return ore_snapshot_cli.main(argv)


def _case_slug(ore_xml: Path) -> str:
    parent = ore_xml.resolve().parents[1].name
    return parent or ore_xml.stem


def _product_counts(count_per_type: int, *, include_fedfunds: bool) -> Iterable[tuple[str, int]]:
    yield ("USD-LIBOR-3M vs USD-LIBOR-6M", count_per_type)
    yield ("USD-LIBOR-3M vs USD-SIFMA", count_per_type)
    if include_fedfunds:
        yield ("USD-FedFunds vs USD-LIBOR-3M", count_per_type)


def main() -> int:
    args = _parse_args()
    if args.count_per_type <= 0:
        raise ValueError("--count-per-type must be > 0")

    case_root = args.case_root.resolve()
    artifact_root = (args.artifact_root or (case_root / "artifacts")).resolve()

    _ensure_clean_dir(case_root, overwrite=args.overwrite)
    ore_xml, portfolio_xml = _write_files(
        case_root,
        count_per_type=args.count_per_type,
        include_fedfunds=args.include_fedfunds,
    )

    print("Generated USD floating-vs-floating rates snapshot example")
    print(f"  case_root      : {case_root}")
    print(f"  ore_xml        : {ore_xml}")
    print(f"  portfolio_xml  : {portfolio_xml}")
    print(f"  artifact_root  : {artifact_root}")
    print("  product_counts :")
    for name, count in _product_counts(args.count_per_type, include_fedfunds=args.include_fedfunds):
        print(f"    - {name}: {count}")
    print()
    print("Next scale-up:")
    print("  python3 example_ore_snapshot_usd_basis_swaps.py --count-per-type 100 --overwrite")
    if args.price_only:
        print("  run_mode       : price only")
    else:
        print(f"  run_mode       : price + xva (paths={args.paths})")

    if args.no_run:
        return 0

    print()
    if args.price_only:
        print("Running ore_snapshot_cli --price ...")
    else:
        print(f"Running ore_snapshot_cli --price --xva --paths {args.paths} ...")
    rc = _run_cli(ore_xml, artifact_root, price_only=args.price_only, paths=args.paths)
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
