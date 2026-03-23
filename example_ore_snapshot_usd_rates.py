#!/usr/bin/env python3
"""Generate and run a USD rates-only ORE snapshot example.

This example writes a small self-contained ORE input case with:

- IRS
- Cap
- Floor
- CMS swap
- CMS spread swap
- European swaption

It defaults to one trade of each type so the first run stays easy to validate.
The run mode defaults to pricing plus XVA via `ore_snapshot_cli`.
Once that works, scale with `--count-per-type 100`.

Example:
    python3 example_ore_snapshot_usd_rates.py
    python3 example_ore_snapshot_usd_rates.py --count-per-type 100
    python3 example_ore_snapshot_usd_rates.py --price-only
    python3 example_ore_snapshot_usd_rates.py --no-run
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
DEFAULT_CASE_ROOT = REPO_ROOT / "Examples" / "Generated" / "USD_RatesSnapshot"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a USD rates-only ORE snapshot case and optionally price it with the snapshot CLI."
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
    return parser.parse_args()


def _ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Re-run with --overwrite to replace the generated case."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _trade_blocks(count_per_type: int) -> list[str]:
    blocks: list[str] = []
    for i in range(1, count_per_type + 1):
        suffix = f"{i:03d}"
        blocks.extend(
            [
                _irs_trade_xml(suffix),
                _cap_trade_xml(suffix),
                _floor_trade_xml(suffix),
                _cms_trade_xml(suffix),
                _cmsspread_trade_xml(suffix),
                _swaption_trade_xml(suffix),
            ]
        )
    return blocks


def _irs_trade_xml(suffix: str) -> str:
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
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>0.025</Rate>
          </Rates>
        </FixedLegData>
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
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0</Spread>
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


def _cap_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="CAP_USD_{suffix}">
    <TradeType>CapFloor</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <CapFloorData>
      <LongShort>Long</LongShort>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <DayCounter>ACT/360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <Notionals>
          <Notional>1000000</Notional>
        </Notionals>
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
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps>
        <Cap>0.04</Cap>
      </Caps>
      <Floors/>
    </CapFloorData>
  </Trade>"""


def _floor_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="FLOOR_USD_{suffix}">
    <TradeType>CapFloor</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <CapFloorData>
      <LongShort>Long</LongShort>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <DayCounter>ACT/360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <Notionals>
          <Notional>1000000</Notional>
        </Notionals>
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
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps/>
      <Floors>
        <Floor>0.01</Floor>
      </Floors>
    </CapFloorData>
  </Trade>"""


def _cms_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="CMS_SWAP_USD_{suffix}">
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
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>0.028</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20160209</StartDate>
            <EndDate>20360209</EndDate>
            <Tenor>1Y</Tenor>
            <Calendar>US</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
            <EndOfMonth/>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>CMS</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <CMSLegData>
          <Index>USD-CMS-30Y</Index>
          <Spreads>
            <Spread>0.0</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </CMSLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20160209</StartDate>
            <EndDate>20360209</EndDate>
            <Tenor>6M</Tenor>
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


def _cmsspread_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="CMSSPREAD_USD_{suffix}">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>CMSSpread</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <CMSSpreadLegData>
          <Index1>USD-CMS-10Y</Index1>
          <Index2>USD-CMS-1Y</Index2>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
          <Caps>
            <Cap>0.06</Cap>
          </Caps>
          <Floors>
            <Floor>0.00</Floor>
          </Floors>
          <Gearings>
            <Gearing>2.0</Gearing>
          </Gearings>
          <Spreads>
            <Spread>0.001</Spread>
          </Spreads>
          <NakedOption>false</NakedOption>
        </CMSSpreadLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20160209</StartDate>
            <EndDate>20310209</EndDate>
            <Tenor>1Y</Tenor>
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
            <Spread>0.002</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20160209</StartDate>
            <EndDate>20310209</EndDate>
            <Tenor>6M</Tenor>
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


def _swaption_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="SWAPTION_USD_{suffix}">
    <TradeType>Swaption</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwaptionData>
      <OptionData>
        <LongShort>Long</LongShort>
        <OptionType>Call</OptionType>
        <Style>European</Style>
        <Settlement>Physical</Settlement>
        <PayOffAtExpiry>false</PayOffAtExpiry>
        <ExerciseDates>
          <ExerciseDate>2021-02-09</ExerciseDate>
        </ExerciseDates>
      </OptionData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>ModifiedFollowing</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0</Spread>
          </Spreads>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2021-02-09</StartDate>
            <EndDate>2031-02-09</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>US</Calendar>
            <Convention>ModifiedFollowing</Convention>
            <TermConvention>ModifiedFollowing</TermConvention>
            <Rule>Forward</Rule>
            <EndOfMonth/>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>Following</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>0.03</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2021-02-09</StartDate>
            <EndDate>2031-02-09</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>US</Calendar>
            <Convention>Following</Convention>
            <TermConvention>Following</TermConvention>
            <Rule>Forward</Rule>
            <EndOfMonth/>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwaptionData>
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
    </Indices>
    <SwapIndices>
      <SwapIndex>
        <Name>USD-CMS-1Y</Name>
        <DiscountingIndex>USD-LIBOR-3M</DiscountingIndex>
      </SwapIndex>
      <SwapIndex>
        <Name>USD-CMS-10Y</Name>
        <DiscountingIndex>USD-LIBOR-3M</DiscountingIndex>
      </SwapIndex>
      <SwapIndex>
        <Name>USD-CMS-30Y</Name>
        <DiscountingIndex>USD-LIBOR-3M</DiscountingIndex>
      </SwapIndex>
    </SwapIndices>
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
    </AggregationScenarioDataIndices>
  </Market>
</Simulation>
"""


def _ore_xml(case_input_dir: Path) -> str:
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
    <Parameter name="portfolioFile">portfolio_usd_rates.xml</Parameter>
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


def _write_files(case_root: Path, *, count_per_type: int) -> tuple[Path, Path]:
    input_dir = case_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    ore_xml = input_dir / "ore.xml"
    portfolio_xml = input_dir / "portfolio_usd_rates.xml"
    simulation_xml = input_dir / "simulation.xml"
    netting_xml = input_dir / "netting.xml"

    ore_xml.write_text(_ore_xml(input_dir), encoding="utf-8")
    portfolio_xml.write_text(_portfolio_xml(count_per_type), encoding="utf-8")
    simulation_xml.write_text(_simulation_xml(), encoding="utf-8")
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


def _product_counts(count_per_type: int) -> Iterable[tuple[str, int]]:
    yield ("IRS", count_per_type)
    yield ("Cap", count_per_type)
    yield ("Floor", count_per_type)
    yield ("CMS", count_per_type)
    yield ("CMSSpread", count_per_type)
    yield ("Swaption", count_per_type)


def main() -> int:
    args = _parse_args()
    if args.count_per_type <= 0:
        raise ValueError("--count-per-type must be > 0")

    case_root = args.case_root.resolve()
    artifact_root = (args.artifact_root or (case_root / "artifacts")).resolve()

    _ensure_clean_dir(case_root, overwrite=args.overwrite)
    ore_xml, portfolio_xml = _write_files(case_root, count_per_type=args.count_per_type)

    print("Generated USD rates snapshot example")
    print(f"  case_root      : {case_root}")
    print(f"  ore_xml        : {ore_xml}")
    print(f"  portfolio_xml  : {portfolio_xml}")
    print(f"  artifact_root  : {artifact_root}")
    print("  product_counts :")
    for name, count in _product_counts(args.count_per_type):
        print(f"    - {name}: {count}")
    print()
    print("Next scale-up:")
    print("  python3 example_ore_snapshot_usd_rates.py --count-per-type 100 --overwrite")
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
