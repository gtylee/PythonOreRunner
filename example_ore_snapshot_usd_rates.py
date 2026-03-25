#!/usr/bin/env python3
"""Generate and run a USD rates-only ORE snapshot example.

This example writes a small self-contained ORE input case with:

- IRS
- Cap
- Floor
- CMS swap
- CMS spread swap
- Digital CMS spread swap
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
    parser.add_argument(
        "--include-digital-cmsspread",
        action="store_true",
        help="Include Digital CMSSpread swaps in the generated portfolio.",
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


def _trade_blocks(count_per_type: int, *, include_digital_cmsspread: bool) -> list[str]:
    blocks: list[str] = []
    for i in range(1, count_per_type + 1):
        suffix = f"{i:03d}"
        batch = [
            _irs_trade_xml(suffix),
            _cap_trade_xml(suffix),
            _floor_trade_xml(suffix),
            _cms_trade_xml(suffix),
            _cmsspread_trade_xml(suffix),
            _swaption_trade_xml(suffix),
        ]
        if include_digital_cmsspread:
            batch.append(_digital_cmsspread_trade_xml(suffix))
        blocks.extend(batch)
    return blocks


def _suffix_index(suffix: str) -> int:
    return max(int(suffix), 1)


def _add_months(yyyymmdd: str, months: int) -> str:
    year = int(yyyymmdd[0:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])
    month0 = (month - 1) + int(months)
    year += month0 // 12
    month = (month0 % 12) + 1
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        month_days = 29 if leap else 28
    elif month in {4, 6, 9, 11}:
        month_days = 30
    else:
        month_days = 31
    day = min(day, month_days)
    return f"{year:04d}{month:02d}{day:02d}"


def _compact_to_iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _rate_book_variant(suffix: str) -> dict[str, float | str]:
    idx = _suffix_index(suffix) - 1
    start_shift_months = (idx % 12) * 3
    maturity_shift_months = ((idx // 12) % 5 - 1) * 12
    notional_scale = 1.0 + 0.05 * (idx % 5)
    coupon_shift = (idx % 9 - 4) * 0.0004
    spread_shift = (idx % 7 - 3) * 0.00025
    strike_shift = (idx % 5 - 2) * 0.0015
    return {
        "start_shift_months": start_shift_months,
        "maturity_shift_months": maturity_shift_months,
        "notional_scale": notional_scale,
        "coupon_shift": coupon_shift,
        "spread_shift": spread_shift,
        "strike_shift": strike_shift,
    }


def _irs_trade_xml(suffix: str) -> str:
    v = _rate_book_variant(suffix)
    start = _add_months("20160209", int(v["start_shift_months"]))
    end = _add_months("20260209", int(v["start_shift_months"]) + int(v["maturity_shift_months"]))
    notional = int(round(10_000_000 * float(v["notional_scale"])))
    fixed_rate = 0.025 + float(v["coupon_shift"])
    float_spread = float(v["spread_shift"])
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>{fixed_rate:.6f}</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>{float_spread:.6f}</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
    v = _rate_book_variant(suffix)
    start = _add_months("20160209", int(v["start_shift_months"]))
    end = _add_months("20260209", int(v["start_shift_months"]) + int(v["maturity_shift_months"]))
    notional = int(round(1_000_000 * float(v["notional_scale"])))
    strike = max(0.005, 0.04 + float(v["strike_shift"]))
    spread = float(v["spread_shift"])
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
          <Notional>{notional}</Notional>
        </Notionals>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
            <Spread>{spread:.6f}</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps>
        <Cap>{strike:.6f}</Cap>
      </Caps>
      <Floors/>
    </CapFloorData>
  </Trade>"""


def _floor_trade_xml(suffix: str) -> str:
    v = _rate_book_variant(suffix)
    start = _add_months("20160209", int(v["start_shift_months"]))
    end = _add_months("20260209", int(v["start_shift_months"]) + int(v["maturity_shift_months"]))
    notional = int(round(1_000_000 * float(v["notional_scale"])))
    strike = max(0.0001, 0.01 + 0.5 * float(v["strike_shift"]))
    spread = float(v["spread_shift"])
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
          <Notional>{notional}</Notional>
        </Notionals>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
            <Spread>{spread:.6f}</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps/>
      <Floors>
        <Floor>{strike:.6f}</Floor>
      </Floors>
    </CapFloorData>
  </Trade>"""


def _cms_trade_xml(suffix: str) -> str:
    v = _rate_book_variant(suffix)
    start = _add_months("20160209", int(v["start_shift_months"]))
    end = _add_months("20360209", int(v["start_shift_months"]) + int(v["maturity_shift_months"]))
    notional = int(round(10_000_000 * float(v["notional_scale"])))
    fixed_rate = 0.028 + 0.75 * float(v["coupon_shift"])
    spread = float(v["spread_shift"])
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>{fixed_rate:.6f}</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <CMSLegData>
          <Index>USD-CMS-30Y</Index>
          <Spreads>
            <Spread>{spread:.6f}</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </CMSLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
    v = _rate_book_variant(suffix)
    start = _add_months("20160209", int(v["start_shift_months"]))
    end = _add_months("20310209", int(v["start_shift_months"]) + int(v["maturity_shift_months"]))
    notional = int(round(10_000_000 * float(v["notional_scale"])))
    cms_spread = 0.001 + float(v["spread_shift"])
    libor_spread = 0.002 + 0.5 * float(v["spread_shift"])
    cap = max(0.02, 0.06 + float(v["strike_shift"]))
    gearing = 2.0 + 0.1 * ((_suffix_index(suffix) - 1) % 3)
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <CMSSpreadLegData>
          <Index1>USD-CMS-10Y</Index1>
          <Index2>USD-CMS-1Y</Index2>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
          <Caps>
            <Cap>{cap:.6f}</Cap>
          </Caps>
          <Floors>
            <Floor>0.00</Floor>
          </Floors>
          <Gearings>
            <Gearing>{gearing:.6f}</Gearing>
          </Gearings>
          <Spreads>
            <Spread>{cms_spread:.6f}</Spread>
          </Spreads>
          <NakedOption>false</NakedOption>
        </CMSSpreadLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-6M</Index>
          <Spreads>
            <Spread>{libor_spread:.6f}</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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


def _digital_cmsspread_trade_xml(suffix: str) -> str:
    v = _rate_book_variant(suffix)
    start = _add_months("20160209", int(v["start_shift_months"]))
    end = _add_months("20260209", int(v["start_shift_months"]) + int(v["maturity_shift_months"]))
    notional = int(round(10_000_000 * float(v["notional_scale"])))
    cms_spread = 0.001 + float(v["spread_shift"])
    libor_spread = 0.002 + 0.5 * float(v["spread_shift"])
    call_strike = max(0.0001, 0.002 + 0.4 * float(v["strike_shift"]))
    put_strike = max(0.0001, 0.0005 + 0.2 * float(v["strike_shift"]))
    gearing = 2.0 + 0.1 * ((_suffix_index(suffix) - 1) % 3)
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <DigitalCMSSpreadLegData>
          <CMSSpreadLegData>
            <Index1>USD-CMS-10Y</Index1>
            <Index2>USD-CMS-1Y</Index2>
            <IsInArrears>false</IsInArrears>
            <FixingDays>2</FixingDays>
            <Gearings>
              <Gearing>{gearing:.6f}</Gearing>
            </Gearings>
            <Spreads>
              <Spread>{cms_spread:.6f}</Spread>
            </Spreads>
          </CMSSpreadLegData>
          <CallPosition>Long</CallPosition>
          <IsCallATMIncluded>false</IsCallATMIncluded>
          <CallStrikes>
            <Strike>{call_strike:.6f}</Strike>
          </CallStrikes>
          <CallPayoffs>
            <Payoff>0.002</Payoff>
          </CallPayoffs>
          <PutPosition>Long</PutPosition>
          <IsPutATMIncluded>false</IsPutATMIncluded>
          <PutStrikes>
            <Strike>{put_strike:.6f}</Strike>
          </PutStrikes>
          <PutPayoffs>
            <Payoff>0.001</Payoff>
          </PutPayoffs>
        </DigitalCMSSpreadLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-6M</Index>
          <Spreads>
            <Spread>{libor_spread:.6f}</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
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
    v = _rate_book_variant(suffix)
    exercise = _add_months("20210209", int(v["start_shift_months"]))
    end = _add_months("20310209", int(v["start_shift_months"]) + int(v["maturity_shift_months"]))
    notional = int(round(10_000_000 * float(v["notional_scale"])))
    fixed_rate = 0.03 + float(v["coupon_shift"])
    spread = float(v["spread_shift"])
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
          <ExerciseDate>{_compact_to_iso(exercise)}</ExerciseDate>
        </ExerciseDates>
      </OptionData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>ModifiedFollowing</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads>
            <Spread>{spread:.6f}</Spread>
          </Spreads>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{_compact_to_iso(exercise)}</StartDate>
            <EndDate>{_compact_to_iso(end)}</EndDate>
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
          <Notional>{notional}</Notional>
        </Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>Following</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>{fixed_rate:.6f}</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{_compact_to_iso(exercise)}</StartDate>
            <EndDate>{_compact_to_iso(end)}</EndDate>
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


def _portfolio_xml(count_per_type: int, *, include_digital_cmsspread: bool) -> str:
    trades = "\n".join(_trade_blocks(count_per_type, include_digital_cmsspread=include_digital_cmsspread))
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


def _write_files(
    case_root: Path,
    *,
    count_per_type: int,
    include_digital_cmsspread: bool,
) -> tuple[Path, Path]:
    input_dir = case_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    ore_xml = input_dir / "ore.xml"
    portfolio_xml = input_dir / "portfolio_usd_rates.xml"
    simulation_xml = input_dir / "simulation.xml"
    netting_xml = input_dir / "netting.xml"

    ore_xml.write_text(_ore_xml(input_dir), encoding="utf-8")
    portfolio_xml.write_text(
        _portfolio_xml(count_per_type, include_digital_cmsspread=include_digital_cmsspread),
        encoding="utf-8",
    )
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


def _product_counts(count_per_type: int, *, include_digital_cmsspread: bool) -> Iterable[tuple[str, int]]:
    yield ("IRS", count_per_type)
    yield ("Cap", count_per_type)
    yield ("Floor", count_per_type)
    yield ("CMS", count_per_type)
    yield ("CMSSpread", count_per_type)
    if include_digital_cmsspread:
        yield ("DigitalCMSSpread", count_per_type)
    yield ("Swaption", count_per_type)


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
        include_digital_cmsspread=args.include_digital_cmsspread,
    )

    print("Generated USD rates snapshot example")
    print(f"  case_root      : {case_root}")
    print(f"  ore_xml        : {ore_xml}")
    print(f"  portfolio_xml  : {portfolio_xml}")
    print(f"  artifact_root  : {artifact_root}")
    print("  product_counts :")
    for name, count in _product_counts(
        args.count_per_type,
        include_digital_cmsspread=args.include_digital_cmsspread,
    ):
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
