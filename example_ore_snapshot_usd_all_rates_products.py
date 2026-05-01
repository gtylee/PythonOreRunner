#!/usr/bin/env python3
"""Generate and run a broad USD rates book under the modern Products bundle."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
import time
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
    parser.add_argument("--timing-breakdown", action="store_true")
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
            _amortizing_irs_trade_xml(suffix),
            _cap_trade_xml(suffix),
            _floor_trade_xml(suffix),
            _native_cap_trade_xml(suffix),
            _native_floor_trade_xml(suffix),
            _cashflow_trade_xml(suffix),
            _swaption_trade_xml(suffix),
            _bermudan_swaption_trade_xml(suffix),
            _cms_trade_xml(suffix),
            _cmsspread_trade_xml(suffix),
            _digital_cmsspread_trade_xml(suffix),
            _basis_libor_3m_6m_trade_xml(suffix),
            _basis_libor_3m_sifma_trade_xml(suffix),
            _basis_fedfunds_libor_3m_trade_xml(suffix),
            _basis_sofr_3m_libor_3m_trade_xml(suffix),
            _basis_sofr_3m_sifma_trade_xml(suffix),
            _basis_sofr_tonar_xccy_trade_xml(suffix),
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


def _amortizing_irs_trade_xml(suffix: str) -> str:
    fixed_notionals = "\n".join(
        f"          <Notional>{n}</Notional>"
        for n in (10_000_000, 9_500_000, 9_000_000, 8_500_000, 8_000_000, 7_500_000)
    )
    float_notionals = "\n".join(
        f"          <Notional>{n}</Notional>"
        for n in (
            10_000_000, 10_000_000,
            9_500_000, 9_500_000,
            9_000_000, 9_000_000,
            8_500_000, 8_500_000,
            8_000_000, 8_000_000,
            7_500_000, 7_500_000,
        )
    )
    return f"""  <Trade id="IRS_AMORT_USD_{suffix}">
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
{fixed_notionals}
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>0.0375</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2028-02-10</EndDate>
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
{float_notionals}
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0005</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2028-02-10</EndDate>
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


def _native_cap_trade_xml(suffix: str) -> str:
    notionals = "\n".join(
        f"          <Notional>{n}</Notional>"
        for n in (2_000_000, 1_900_000, 1_800_000, 1_700_000, 1_600_000, 1_500_000, 1_400_000, 1_300_000)
    )
    return f"""  <Trade id="CAP_USD_LIB3M_{suffix}">
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
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <Notionals>
{notionals}
        </Notionals>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2027-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0000</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps>
        <Cap>0.045</Cap>
      </Caps>
      <Floors/>
    </CapFloorData>
  </Trade>"""


def _native_floor_trade_xml(suffix: str) -> str:
    notionals = "\n".join(
        f"          <Notional>{n}</Notional>"
        for n in (2_000_000, 1_900_000, 1_800_000, 1_700_000, 1_600_000, 1_500_000, 1_400_000, 1_300_000)
    )
    return f"""  <Trade id="FLOOR_USD_LIB3M_{suffix}">
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
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <Notionals>
{notionals}
        </Notionals>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2027-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>0.0000</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps/>
      <Floors>
        <Floor>0.015</Floor>
      </Floors>
    </CapFloorData>
  </Trade>"""


def _cashflow_trade_xml(suffix: str) -> str:
    amount = 100_000 + (int(suffix) - 1) * 2_500
    month = ((int(suffix) - 1) % 9) + 3
    pay_date = f"2026-{month:02d}-10"
    return f"""  <Trade id="CASHFLOW_USD_{suffix}">
    <TradeType>Cashflow</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <CashflowData>
      <PaymentDate>{pay_date}</PaymentDate>
      <Amount>{amount:.2f}</Amount>
      <Currency>USD</Currency>
    </CashflowData>
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


def _bermudan_swaption_trade_xml(suffix: str) -> str:
    start_year = 2027 + ((int(suffix) - 1) % 2)
    end_year = start_year + 10
    return f"""  <Trade id="BERMUDAN_SWAPTION_USD_{suffix}">
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
        <Style>Bermudan</Style>
        <Settlement>Physical</Settlement>
        <PayOffAtExpiry>false</PayOffAtExpiry>
        <ExerciseDates>
          <ExerciseDate>{start_year}-02-10</ExerciseDate>
          <ExerciseDate>{start_year + 1}-02-10</ExerciseDate>
          <ExerciseDate>{start_year + 2}-02-10</ExerciseDate>
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
            <Spread>0.0000</Spread>
          </Spreads>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start_year}-02-10</StartDate>
            <EndDate>{end_year}-02-10</EndDate>
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
            <Rate>0.039</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start_year}-02-10</StartDate>
            <EndDate>{end_year}-02-10</EndDate>
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


def _basis_sofr_tonar_xccy_trade_xml(suffix: str) -> str:
    return f"""  <Trade id="XCCY_USD_SOFR_JPY_TONAR_{suffix}">
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
          <Notional>100000000</Notional>
          <Exchanges>
            <NotionalInitialExchange>true</NotionalInitialExchange>
            <NotionalFinalExchange>true</NotionalFinalExchange>
          </Exchanges>
        </Notionals>
        <DayCounter>ACT/360</DayCounter>
        <PaymentConvention>ModifiedFollowing</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR</Index>
          <Spreads><Spread>0.0000</Spread></Spreads>
          <Gearings><Gearing>1.0</Gearing></Gearings>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>2025-04-10</StartDate><EndDate>2028-04-10</EndDate><Tenor>3M</Tenor><Calendar>US,JP</Calendar><Convention>ModifiedFollowing</Convention><TermConvention>ModifiedFollowing</TermConvention><Rule>Forward</Rule><EndOfMonth>false</EndOfMonth></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>JPY</Currency>
        <Notionals>
          <Notional>15222818282</Notional>
          <FXReset>
            <ForeignCurrency>USD</ForeignCurrency>
            <ForeignAmount>100000000</ForeignAmount>
            <FXIndex>FX-BOE-USD-JPY</FXIndex>
            <FixingDays>2</FixingDays>
          </FXReset>
          <Exchanges>
            <NotionalInitialExchange>true</NotionalInitialExchange>
            <NotionalFinalExchange>true</NotionalFinalExchange>
          </Exchanges>
        </Notionals>
        <DayCounter>ACT/360</DayCounter>
        <PaymentConvention>ModifiedFollowing</PaymentConvention>
        <FloatingLegData>
          <Index>JPY-TONAR</Index>
          <Spreads><Spread>0.0000</Spread></Spreads>
          <Gearings><Gearing>1.0</Gearing></Gearings>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>2025-04-10</StartDate><EndDate>2028-04-10</EndDate><Tenor>3M</Tenor><Calendar>US,JP</Calendar><Convention>ModifiedFollowing</Convention><TermConvention>ModifiedFollowing</TermConvention><Rule>Forward</Rule><EndOfMonth>false</EndOfMonth></Rules></ScheduleData>
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
      <Currency>JPY</Currency>
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
      <LGM ccy="JPY">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility>
          <Calibrate>N</Calibrate>
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
    <ForeignExchangeModels>
      <CrossCcyLGM foreignCcy="JPY">
        <DomesticCcy>USD</DomesticCcy>
        <CalibrationType>Bootstrap</CalibrationType>
        <Sigma>
          <Calibrate>N</Calibrate>
          <ParamType>Piecewise</ParamType>
          <TimeGrid>1.0,2.0,3.0,5.0,7.0,10.0</TimeGrid>
          <InitialValue>0.10,0.10,0.10,0.10,0.10,0.10,0.10</InitialValue>
        </Sigma>
        <CalibrationOptions>
          <Expiries>1Y,2Y,3Y,5Y,10Y</Expiries>
          <Strikes/>
        </CalibrationOptions>
      </CrossCcyLGM>
    </ForeignExchangeModels>
    <InstantaneousCorrelations>
      <Correlation factor1="IR:USD" factor2="IR:JPY">0.30</Correlation>
      <Correlation factor1="IR:USD" factor2="FX:USDJPY">0.00</Correlation>
      <Correlation factor1="IR:JPY" factor2="FX:USDJPY">0.00</Correlation>
    </InstantaneousCorrelations>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>USD</BaseCurrency>
    <Currencies>
      <Currency>USD</Currency>
      <Currency>JPY</Currency>
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
      <Index>JPY-TONAR</Index>
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
    <FxVolatilities>
      <ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay>
      <CurrencyPairs>
        <CurrencyPair>USDJPY</CurrencyPair>
      </CurrencyPairs>
      <Expiries>1M,3M,6M,1Y,2Y,3Y,5Y,10Y</Expiries>
    </FxVolatilities>
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
      <Currency>JPY</Currency>
    </AggregationScenarioDataCurrencies>
    <AggregationScenarioDataIndices>
      <Index>USD-FedFunds</Index>
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-LIBOR-6M</Index>
      <Index>USD-SIFMA</Index>
      <Index>USD-SOFR</Index>
      <Index>USD-SOFR-3M</Index>
      <Index>JPY-TONAR</Index>
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


def _run_cli(
    ore_xml: Path,
    artifact_root: Path,
    *,
    price: bool = True,
    xva: bool = True,
    sensi: bool = True,
    sensi_metric: str = "CVA",
    sensi_progress: bool = True,
    paths: int,
    lgm_param_source: str,
) -> int:
    argv = [str(ore_xml), "--price"]
    if xva:
        argv.append("--xva")
    if sensi:
        argv.extend(["--sensi", "--sensi-metric", str(sensi_metric)])
        if sensi_progress:
            argv.append("--sensi-progress")
    if xva or sensi:
        argv.extend(["--paths", str(paths)])
    argv.extend(["--lgm-param-source", str(lgm_param_source), "--output-root", str(artifact_root)])
    return ore_snapshot_cli.main(argv)


def _timed_run(label: str, fn) -> tuple[int, float]:
    started = time.perf_counter()
    rc = int(fn())
    return rc, time.perf_counter() - started


def _case_slug(ore_xml: Path) -> str:
    parent = ore_xml.resolve().parents[1].name
    return parent or ore_xml.stem


def _product_counts(count_per_type: int) -> Iterable[tuple[str, int]]:
    yield ("IRS", count_per_type)
    yield ("Amortizing IRS", count_per_type)
    yield ("SOFR Cap", count_per_type)
    yield ("SOFR Floor", count_per_type)
    yield ("LIBOR Cap", count_per_type)
    yield ("LIBOR Floor", count_per_type)
    yield ("Cashflow", count_per_type)
    yield ("Swaption", count_per_type)
    yield ("Bermudan Swaption", count_per_type)
    yield ("CMS Swap", count_per_type)
    yield ("CMS Spread Swap", count_per_type)
    yield ("Digital CMS Spread Swap", count_per_type)
    yield ("Basis LIBOR3M/LIBOR6M", count_per_type)
    yield ("Basis LIBOR3M/SIFMA", count_per_type)
    yield ("Basis FedFunds/LIBOR3M", count_per_type)
    yield ("Basis SOFR3M/LIBOR3M", count_per_type)
    yield ("Basis SOFR3M/SIFMA", count_per_type)
    yield ("XCCY SOFR/TONAR", count_per_type)


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
    print("  note           : book includes CMS, amortizing, SIFMA, overnight SOFR/TONAR, and USD/JPY XCCY native IR trades")
    print()
    print("Next scale-up:")
    print("  python3 example_ore_snapshot_usd_all_rates_products.py --count-per-type 100 --overwrite")
    if args.price_only:
        print("  run_mode       : price only")
    elif args.timing_breakdown:
        print(f"  run_mode       : timing breakdown (paths={args.paths})")
    else:
        print(f"  run_mode       : price + xva + sensitivity (paths={args.paths}, live sensi progress)")
    print(f"  lgm_param_src  : {args.lgm_param_source}")

    if args.no_run:
        return 0

    print()
    if args.price_only:
        print("Running ore_snapshot_cli --price ...")
        rc = _run_cli(
            ore_xml,
            artifact_root,
            xva=False,
            sensi=False,
            paths=args.paths,
            lgm_param_source=args.lgm_param_source,
        )
    elif args.timing_breakdown:
        print(f"Running timing breakdown with paths={args.paths} ...")
        runs = [
            (
                "price+xva",
                artifact_root / "timing_price_xva",
                dict(xva=True, sensi=False, sensi_progress=False),
            ),
            (
                "price+sensi(npv)",
                artifact_root / "timing_price_sensi_npv",
                dict(xva=False, sensi=True, sensi_metric="NPV", sensi_progress=False),
            ),
            (
                "price+xva+sensi(cva)",
                artifact_root / "timing_price_xva_sensi_cva",
                dict(xva=True, sensi=True, sensi_metric="CVA", sensi_progress=True),
            ),
        ]
        timing_rows: list[tuple[str, int, float, Path]] = []
        rc = 0
        for label, run_artifact_root, kwargs in runs:
            print()
            print(f"  timing_run     : {label}")
            run_rc, elapsed = _timed_run(
                label,
                lambda run_artifact_root=run_artifact_root, kwargs=kwargs: _run_cli(
                    ore_xml,
                    run_artifact_root,
                    paths=args.paths,
                    lgm_param_source=args.lgm_param_source,
                    **kwargs,
                ),
            )
            timing_rows.append((label, run_rc, elapsed, run_artifact_root))
            if run_rc != 0 and rc == 0:
                rc = run_rc
        print()
        print("Timing summary")
        for label, run_rc, elapsed, run_artifact_root in timing_rows:
            print(f"  {label:<22} rc={run_rc} elapsed={elapsed:.6f} sec artifacts={run_artifact_root}")
    else:
        print(f"Running ore_snapshot_cli --price --xva --sensi --sensi-progress --paths {args.paths} ...")
        rc = _run_cli(
            ore_xml,
            artifact_root,
            xva=True,
            sensi=True,
            sensi_metric="CVA",
            sensi_progress=True,
            paths=args.paths,
            lgm_param_source=args.lgm_param_source,
        )
    if rc == 0:
        case_dir = artifact_root / _case_slug(ore_xml)
        print()
        if args.price_only:
            print("Pricing run completed")
        elif args.timing_breakdown:
            print("Timing breakdown completed")
        else:
            print("Pricing, XVA, and sensitivity run completed")
        if not args.timing_breakdown:
            print(f"  report_dir     : {case_dir}")
            print(f"  summary_json   : {case_dir / 'summary.json'}")
            print(f"  report_md      : {case_dir / 'report.md'}")
            print(f"  compare_csv    : {case_dir / 'comparison.csv'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
