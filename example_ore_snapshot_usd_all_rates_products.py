#!/usr/bin/env python3
"""Generate and run a broad USD rates book under the modern Products bundle."""

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
DEFAULT_CASE_ROOT = REPO_ROOT / "Examples" / "Generated" / "USD_AllRatesProductsSnapshot"
ASOF_DATE = "2025-02-10"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a broad USD rates-only ORE snapshot case for pricing, XVA, and sensitivity."
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
        suffix = f"{i:04d}"
        batch = [
            _irs_trade_xml(suffix),
            _cap_trade_xml(suffix),
            _floor_trade_xml(suffix),
            _swaption_trade_xml(suffix),
            _basis_libor_3m_6m_trade_xml(suffix),
            _basis_libor_3m_sifma_trade_xml(suffix),
            _basis_fedfunds_libor_3m_trade_xml(suffix),
            _basis_sofr_3m_libor_3m_trade_xml(suffix),
            _basis_sofr_3m_sifma_trade_xml(suffix),
        ]
        blocks.extend(batch)
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
            <Rate>0.040</Rate>
          </Rates>
        </FixedLegData>
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
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0000</Spread>
          </Spreads>
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


def _cap_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="CAP_USD_SOFR3M_{suffix}">
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
        <DayCounter>ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <Notionals>
          <Notional>1000000</Notional>
        </Notionals>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2030-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
        <FloatingLegData>
          <Index>USD-SOFR-3M</Index>
          <Spreads>
            <Spread>0.0</Spread>
          </Spreads>
          <IsInArrears>true</IsInArrears>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps>
        <Cap>0.05</Cap>
      </Caps>
      <Floors/>
    </CapFloorData>
  </Trade>"""


def _floor_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="FLOOR_USD_SOFR3M_{suffix}">
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
        <DayCounter>ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <Notionals>
          <Notional>1000000</Notional>
        </Notionals>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2030-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
        <FloatingLegData>
          <Index>USD-SOFR-3M</Index>
          <Spreads>
            <Spread>0.0</Spread>
          </Spreads>
          <IsInArrears>true</IsInArrears>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps/>
      <Floors>
        <Floor>0.02</Floor>
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
            <Rate>0.040</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2045-02-10</EndDate>
            <Tenor>1Y</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
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
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2045-02-10</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
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
          <Index1>USD-CMS-30Y</Index1>
          <Index2>USD-CMS-2Y</Index2>
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
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2040-02-10</EndDate>
            <Tenor>1Y</Tenor>
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
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2040-02-10</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _digital_cmsspread_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="DIGITAL_CMSSPREAD_USD_{suffix}">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>DigitalCMSSpread</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <DigitalCMSSpreadLegData>
          <CMSSpreadLegData>
            <Index1>USD-CMS-30Y</Index1>
            <Index2>USD-CMS-2Y</Index2>
            <IsInArrears>false</IsInArrears>
            <FixingDays>2</FixingDays>
            <Gearings>
              <Gearing>2.0</Gearing>
            </Gearings>
            <Spreads>
              <Spread>0.001</Spread>
            </Spreads>
          </CMSSpreadLegData>
          <CallPosition>Long</CallPosition>
          <IsCallATMIncluded>false</IsCallATMIncluded>
          <CallStrikes>
            <Strike>0.002</Strike>
          </CallStrikes>
          <CallPayoffs>
            <Payoff>0.002</Payoff>
          </CallPayoffs>
          <PutPosition>Long</PutPosition>
          <IsPutATMIncluded>false</IsPutATMIncluded>
          <PutStrikes>
            <Strike>0.0005</Strike>
          </PutStrikes>
          <PutPayoffs>
            <Payoff>0.001</Payoff>
          </PutPayoffs>
        </DigitalCMSSpreadLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2030-02-10</EndDate>
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
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2030-02-10</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
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
          <ExerciseDate>2027-02-10</ExerciseDate>
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
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0</Spread>
          </Spreads>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2027-02-10</StartDate>
            <EndDate>2037-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
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
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>0.040</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>2027-02-10</StartDate>
            <EndDate>2037-02-10</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>F</Convention>
            <TermConvention>F</TermConvention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwaptionData>
  </Trade>"""


def _basis_libor_3m_6m_trade_xml(suffix: str) -> str:
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
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-6M</Index>
          <Spreads><Spread>0.0000</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>6M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads><Spread>0.0015</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _basis_libor_3m_sifma_trade_xml(suffix: str) -> str:
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
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads/>
          <Gearings><Gearing>0.8</Gearing></Gearings>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>ACT/ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SIFMA</Index>
          <Spreads><Spread>0.0005</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>1</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>3M</Tenor><Calendar>US-NYSE</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _basis_fedfunds_libor_3m_trade_xml(suffix: str) -> str:
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
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-FedFunds</Index>
          <Spreads><Spread>0.0002</Spread></Spreads>
          <IsAveraged>true</IsAveraged>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2030-02-10</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Backward</Rule><EndOfMonth>false</EndOfMonth></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads><Spread>0.0010</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2030-02-10</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Backward</Rule><EndOfMonth>false</EndOfMonth></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _basis_sofr_3m_libor_3m_trade_xml(suffix: str) -> str:
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
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR-3M</Index>
          <Spreads><Spread>0.0000</Spread></Spreads>
          <IsInArrears>true</IsInArrears>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads><Spread>0.0010</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _basis_sofr_3m_sifma_trade_xml(suffix: str) -> str:
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
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR-3M</Index>
          <Spreads><Spread>0.0005</Spread></Spreads>
          <IsInArrears>true</IsInArrears>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>ACT/ACT</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SIFMA</Index>
          <Spreads><Spread>0.0005</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>1</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>{ASOF_DATE}</StartDate><EndDate>2035-02-10</EndDate><Tenor>3M</Tenor><Calendar>US-NYSE</Calendar><Convention>MF</Convention><Rule>Forward</Rule></Rules></ScheduleData>
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
      <Index>USD-FedFunds</Index>
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-LIBOR-6M</Index>
      <Index>USD-SIFMA</Index>
      <Index>USD-SOFR</Index>
      <Index>USD-SOFR-3M</Index>
    </Indices>
    <SwapIndices>
      <SwapIndex>
        <Name>USD-CMS-2Y</Name>
        <DiscountingIndex>USD-SOFR</DiscountingIndex>
      </SwapIndex>
      <SwapIndex>
        <Name>USD-CMS-30Y</Name>
        <DiscountingIndex>USD-SOFR</DiscountingIndex>
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
      <Expiries>6M,1Y,2Y,3Y,5Y,10Y</Expiries>
      <Terms>1Y,2Y,3Y,5Y,10Y,30Y</Terms>
    </SwaptionVolatilities>
    <AggregationScenarioDataCurrencies>
      <Currency>USD</Currency>
    </AggregationScenarioDataCurrencies>
    <AggregationScenarioDataIndices>
      <Index>USD-FedFunds</Index>
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-LIBOR-6M</Index>
      <Index>USD-SIFMA</Index>
      <Index>USD-SOFR</Index>
      <Index>USD-SOFR-3M</Index>
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
    <Parameter name="portfolioFile">portfolio_usd_all_rates_products.xml</Parameter>
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
    <Analytic type="sensitivity">
      <Parameter name="active">Y</Parameter>
      <Parameter name="sensitivityConfigFile">sensitivity.xml</Parameter>
      <Parameter name="sensitivityOutputFile">sensitivity.csv</Parameter>
      <Parameter name="scenarioOutputFile">scenario.csv</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def _sensitivity_xml() -> str:
    return """<?xml version="1.0"?>
<SensitivityAnalysis>
  <DiscountCurves>
    <DiscountCurve ccy="USD">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>1M,3M,6M,1Y,2Y,5Y,10Y,30Y</ShiftTenors>
    </DiscountCurve>
  </DiscountCurves>
  <IndexCurves>
    <IndexCurve index="USD-FedFunds">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>1M,3M,6M,1Y,2Y,5Y,10Y,30Y</ShiftTenors>
    </IndexCurve>
    <IndexCurve index="USD-LIBOR-3M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>1M,3M,6M,1Y,2Y,5Y,10Y,30Y</ShiftTenors>
    </IndexCurve>
    <IndexCurve index="USD-LIBOR-6M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>1M,3M,6M,1Y,2Y,5Y,10Y,30Y</ShiftTenors>
    </IndexCurve>
    <IndexCurve index="USD-SIFMA">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>1M,3M,6M,1Y,2Y,5Y,10Y,30Y</ShiftTenors>
    </IndexCurve>
    <IndexCurve index="USD-SOFR">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>1M,3M,6M,1Y,2Y,5Y,10Y,30Y</ShiftTenors>
    </IndexCurve>
    <IndexCurve index="USD-SOFR-3M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>1M,3M,6M,1Y,2Y,5Y,10Y,30Y</ShiftTenors>
    </IndexCurve>
  </IndexCurves>
</SensitivityAnalysis>
"""


def _write_files(case_root: Path, *, count_per_type: int) -> tuple[Path, Path]:
    input_dir = case_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    ore_xml = input_dir / "ore.xml"
    portfolio_xml = input_dir / "portfolio_usd_all_rates_products.xml"
    simulation_xml = input_dir / "simulation.xml"
    netting_xml = input_dir / "netting.xml"
    sensitivity_xml = input_dir / "sensitivity.xml"

    ore_xml.write_text(_ore_xml(), encoding="utf-8")
    portfolio_xml.write_text(
        _portfolio_xml(count_per_type),
        encoding="utf-8",
    )
    simulation_xml.write_text(_simulation_xml(), encoding="utf-8")
    netting_xml.write_text("<NettingSetDefinitions/>\n", encoding="utf-8")
    sensitivity_xml.write_text(_sensitivity_xml(), encoding="utf-8")
    return ore_xml, portfolio_xml


def _run_cli(ore_xml: Path, artifact_root: Path, *, price_only: bool, paths: int, lgm_param_source: str) -> int:
    argv = [str(ore_xml), "--price"]
    if not price_only:
        argv.extend(["--xva", "--sensi", "--paths", str(paths)])
    argv.extend(["--lgm-param-source", str(lgm_param_source), "--output-root", str(artifact_root)])
    return ore_snapshot_cli.main(argv)


def _case_slug(ore_xml: Path) -> str:
    parent = ore_xml.resolve().parents[1].name
    return parent or ore_xml.stem


def _product_counts(count_per_type: int) -> Iterable[tuple[str, int]]:
    yield ("IRS", count_per_type)
    yield ("Cap", count_per_type)
    yield ("Floor", count_per_type)
    yield ("Swaption", count_per_type)
    yield ("Basis LIBOR3M/LIBOR6M", count_per_type)
    yield ("Basis LIBOR3M/SIFMA", count_per_type)
    yield ("Basis FedFunds/LIBOR3M", count_per_type)
    yield ("Basis SOFR3M/LIBOR3M", count_per_type)
    yield ("Basis SOFR3M/SIFMA", count_per_type)


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
    )

    print("Generated broad USD rates snapshot example")
    print(f"  case_root      : {case_root}")
    print(f"  ore_xml        : {ore_xml}")
    print(f"  portfolio_xml  : {portfolio_xml}")
    print(f"  artifact_root  : {artifact_root}")
    print("  product_counts :")
    for name, count in _product_counts(args.count_per_type):
        print(f"    - {name}: {count}")
    print("  note           : CMS-family trades are excluded to keep native XVA/sensitivity output working")
    print()
    print("Next scale-up:")
    print("  python3 example_ore_snapshot_usd_all_rates_products.py --count-per-type 100 --overwrite")
    if args.price_only:
        print("  run_mode       : price only")
    else:
        print(f"  run_mode       : price + xva + sensitivity (paths={args.paths})")
    print(f"  lgm_param_src  : {args.lgm_param_source}")

    if args.no_run:
        return 0

    print()
    if args.price_only:
        print("Running ore_snapshot_cli --price ...")
    else:
        print(f"Running ore_snapshot_cli --price --xva --sensi --paths {args.paths} ...")
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
            print("Pricing, XVA, and sensitivity run completed")
        print(f"  report_dir     : {case_dir}")
        print(f"  summary_json   : {case_dir / 'summary.json'}")
        print(f"  report_md      : {case_dir / 'report.md'}")
        print(f"  compare_csv    : {case_dir / 'comparison.csv'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
