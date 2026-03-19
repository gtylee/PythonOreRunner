"""Reusable IRS/XVA helpers for LGM notebook examples.

This module is the main adapter between ORE artefacts and the lightweight Python
pricing routines in this directory.  Its responsibilities are deliberately practical:

- read ORE-style XML / CSV inputs such as portfolio, curves, flows and simulation cfg
- reconstruct coupon schedules / leg arrays in a compact NumPy-friendly form
- price swaps pathwise with the same sign and leg conventions used by ORE exports
- derive exposure and default inputs that can be compared against ORE profile output

Most comments below explain those conventions, especially where the code is making a
small modelling approximation so that ORE inputs can still be reused.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import csv
from datetime import date, datetime, timedelta
import re
import xml.etree.ElementTree as ET
import numpy as np


_TIME_YEAR_BASIS = 365.25


def _days_in_year(year: int) -> int:
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


def _year_fraction_actual_actual(start: date, end: date) -> float:
    if end == start:
        return 0.0
    sign = 1.0
    if end < start:
        start, end = end, start
        sign = -1.0

    total = 0.0
    cur = start
    while cur.year < end.year:
        year_end = date(cur.year + 1, 1, 1)
        total += (year_end - cur).days / _days_in_year(cur.year)
        cur = year_end
    total += (end - cur).days / _days_in_year(cur.year)
    return sign * total


def _time_from_dates(start: date, end: date, day_counter: str) -> float:
    dc = (day_counter or "A365F").strip().upper()
    if dc in ("A365F", "A365", "ACT/365(FIXED)", "ACTUAL/365(FIXED)", "ACTUAL365FIXED"):
        return (end - start).days / 365.0
    if dc in ("AAISDA", "ACTUALACTUAL(ISDA)", "ACT/ACT(ISDA)", "ACTUALACTUALISDA"):
        return _year_fraction_actual_actual(start, end)
    return _year_fraction_actual_actual(start, end)


def build_discount_curve_from_zero_rate_pairs(
    rate_pairs: Sequence[Tuple[float, float]],
    compounding: str = "continuous",
) -> Callable[[float], float]:
    """Build discount factor function p0(t) from (time, zero_rate) pairs.

    Uses linear interpolation in zero rates and flat extrapolation at both ends.
    """
    if len(rate_pairs) < 2:
        raise ValueError("rate_pairs must contain at least two (time, rate) points")

    arr = np.asarray(rate_pairs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("rate_pairs must be a sequence of (time, rate) pairs")

    times = arr[:, 0]
    rates = arr[:, 1]

    if np.any(times < 0.0):
        raise ValueError("curve times must be non-negative")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("curve times must be strictly increasing")

    def zero_rate(t: float) -> float:
        if t <= times[0]:
            return float(rates[0])
        if t >= times[-1]:
            return float(rates[-1])
        return float(np.interp(t, times, rates))

    if compounding == "continuous":
        return lambda t: float(np.exp(-zero_rate(float(t)) * float(t)))
    if compounding == "simple":
        return lambda t: float(1.0 / (1.0 + zero_rate(float(t)) * float(t)))

    raise ValueError("compounding must be either 'continuous' or 'simple'")


def build_discount_curve_from_discount_pairs(
    discount_pairs: Sequence[Tuple[float, float]],
) -> Callable[[float], float]:
    """Build discount function p0(t) from (time, discount_factor) pairs."""
    if len(discount_pairs) < 2:
        raise ValueError("discount_pairs must contain at least two points")

    arr = np.asarray(discount_pairs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("discount_pairs must be a sequence of (time, discount) pairs")

    times = arr[:, 0]
    dfs = arr[:, 1]
    if np.any(times < 0.0):
        raise ValueError("discount curve times must be non-negative")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("discount curve times must be strictly increasing")
    if np.any(dfs <= 0.0):
        raise ValueError("discount factors must be strictly positive")

    def p0(t: float) -> float:
        t = float(t)
        if t <= times[0]:
            return float(dfs[0])
        if t >= times[-1]:
            return float(dfs[-1])
        return float(np.interp(t, times, dfs))

    return p0


def build_swap_schedules(trade_def: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build fixed and floating schedules from an ORE-like swap definition.

    This is a simplified helper for toy trades where the ORE trade definition is
    already normalised into year-fraction times.
    """
    maturity = float(trade_def["SwapData"]["End"])
    fixed_leg = trade_def["SwapData"]["LegData"][0]

    front_stub = float(fixed_leg.get("FrontStub", 0.0))
    if front_stub > 0.0:
        fixed_dates = np.concatenate((np.array([front_stub]), np.arange(front_stub + 0.5, maturity + 1e-12, 0.5)))
    else:
        fixed_dates = np.arange(0.5, maturity + 1e-12, 0.5)

    float_dates = np.arange(0.25, maturity + 1e-12, 0.25)
    return fixed_dates, float_dates, maturity


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    mdays = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(d.day, mdays[m - 1])
    return date(y, m, day)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _is_target_holiday(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    if (d.month, d.day) in ((1, 1), (5, 1), (12, 25), (12, 26)):
        return True
    es = _easter_sunday(d.year)
    if d == es - timedelta(days=2) or d == es + timedelta(days=1):
        return True
    return False


def _calendar_tokens(calendar: str) -> list[str]:
    toks = [x.strip().upper() for x in (calendar or "TARGET").split(",") if x.strip()]
    return toks if toks else ["TARGET"]


def _is_business_day(d: date, calendar: str) -> bool:
    if d.weekday() >= 5:
        return False
    for c in _calendar_tokens(calendar):
        if c == "TARGET" and _is_target_holiday(d):
            return False
    return True


def _adjust_date(d: date, convention: str, calendar: str) -> date:
    c = (convention or "F").upper()
    if c in ("U", "UNADJUSTED"):
        return d
    if _is_business_day(d, calendar):
        return d
    if c in ("F", "FOLLOWING", "MF", "MODIFIEDFOLLOWING"):
        x = d
        while not _is_business_day(x, calendar):
            x += timedelta(days=1)
        if c in ("MF", "MODIFIEDFOLLOWING") and x.month != d.month:
            x = d
            while not _is_business_day(x, calendar):
                x -= timedelta(days=1)
        return x
    if c in ("P", "PRECEDING"):
        x = d
        while not _is_business_day(x, calendar):
            x -= timedelta(days=1)
        return x
    return d


def _advance_business_days(d: date, n: int, calendar: str) -> date:
    if n == 0:
        return d
    step = 1 if n > 0 else -1
    remaining = abs(n)
    x = d
    while remaining > 0:
        x += timedelta(days=step)
        if _is_business_day(x, calendar):
            remaining -= 1
    return x


def _year_fraction(start: date, end: date, day_counter: str) -> float:
    dc = day_counter.upper()
    if dc == "A360":
        return (end - start).days / 360.0
    if dc == "A365":
        return (end - start).days / 365.0
    if dc == "30/360":
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 == 30 else end.day
        return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0
    # fallback
    return (end - start).days / 365.0


def _parse_tenor_to_months(tenor: str) -> int:
    tenor = tenor.strip().upper()
    m = re.match(r"^(\d+)([MY])$", tenor)
    if m is None:
        raise ValueError(f"unsupported tenor '{tenor}'")
    n = int(m.group(1))
    return n * 12 if m.group(2) == "Y" else n


def parse_tenor_to_years(tenor: str) -> float:
    tenor = tenor.strip().upper()
    m = re.match(r"^(\d+)([YMWD])$", tenor)
    if m is None:
        raise ValueError(f"unsupported tenor '{tenor}'")
    n = float(m.group(1))
    u = m.group(2)
    if u == "Y":
        return n
    if u == "M":
        return n / 12.0
    if u == "W":
        return n / 52.0
    return n / 365.0


def load_simulation_yield_tenors(simulation_xml: str) -> np.ndarray:
    root = ET.parse(simulation_xml).getroot()
    ten = root.findtext("./Market/YieldCurves/Configuration/Tenors")
    if ten is None:
        raise ValueError("simulation xml missing Market/YieldCurves/Configuration/Tenors")
    vals = [parse_tenor_to_years(x.strip()) for x in ten.split(",") if x.strip()]
    arr = np.asarray(sorted(vals), dtype=float)
    if arr.size == 0:
        raise ValueError("no yield tenors parsed from simulation xml")
    return arr


def _build_schedule(
    start: date,
    end: date,
    tenor: str,
    calendar: str,
    convention: str,
    pay_convention: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build coupon period boundaries from the same rule ingredients ORE stores in
    # ``ScheduleData/Rules``. ORE advances the unadjusted anchor dates and only then
    # applies business-day adjustment to each boundary independently.
    months = _parse_tenor_to_months(tenor)
    boundaries = [start]
    s = start
    while s < end:
        s = _add_months(s, months)
        if s > end:
            s = end
        boundaries.append(s)
    adjusted = [_adjust_date(d, convention, calendar) for d in boundaries]
    starts = adjusted[:-1]
    ends = adjusted[1:]
    pay = [_adjust_date(d, pay_convention or convention, calendar) for d in boundaries[1:]]
    return np.asarray(starts, dtype=object), np.asarray(ends, dtype=object), pay


def build_irregular_exposure_grid(maturity: float) -> np.ndarray:
    """Monthly to 2Y, quarterly to 5Y, semiannual thereafter.

    This is a useful stand-in when an explicit ORE simulation grid is not being read
    from XML.  It is denser near the front where exposure profiles move fastest.
    """
    grid_1 = np.arange(0.0, 2.0 + 1e-12, 1.0 / 12.0)
    grid_2 = np.arange(2.25, min(5.0, maturity) + 1e-12, 0.25)
    grid_3 = np.arange(5.5, maturity + 1e-12, 0.5) if maturity > 5.0 else np.array([], dtype=float)
    return np.unique(np.concatenate((grid_1, grid_2, grid_3)))


def load_ore_exposure_times(exposure_trade_csv: str) -> np.ndarray:
    """Load exposure times from ORE's exposure_trade_*.csv file."""
    data = np.genfromtxt(exposure_trade_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.asarray(data["Time"], dtype=float)


def load_ore_exposure_profile(exposure_trade_csv: str) -> Dict[str, np.ndarray]:
    """Load ORE exposure trade profile with robust column handling.

    ORE has historically emitted either ``TradeId`` or ``#TradeId`` depending on the
    report writer; this helper accepts either so notebook code does not care.
    """
    rows = []
    with open(exposure_trade_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"exposure file '{exposure_trade_csv}' is empty")

    trade_key = "TradeId" if "TradeId" in rows[0] else "#TradeId"
    return {
        "trade_id": np.asarray([r.get(trade_key, "") for r in rows], dtype=object),
        "date": np.asarray([r["Date"] for r in rows], dtype=object),
        "time": np.asarray([float(r["Time"]) for r in rows], dtype=float),
        "epe": np.asarray([float(r["EPE"]) for r in rows], dtype=float),
        "ene": np.asarray([float(r["ENE"]) for r in rows], dtype=float),
    }


def load_ore_discount_pairs_from_curves(
    curves_csv: str,
    discount_column: str = "EUR-EONIA",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load (time, discount) pairs from ORE curves.csv.

    The returned arrays are immediately usable with ``build_discount_curve_from_discount_pairs``.
    """
    data = load_ore_discount_pairs_by_columns(curves_csv, [discount_column])
    return data[discount_column]


def load_ore_discount_pairs_by_columns(
    curves_csv: str,
    discount_columns: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load discount-factor pillars for multiple curves.csv columns in one pass.

    This helper is designed for multi-currency extraction workflows where several
    currencies may map to the same or different curves.csv columns.
    """
    columns = [c for c in discount_columns if c]
    if not columns:
        raise ValueError("discount_columns must contain at least one column name")

    requested = list(dict.fromkeys(columns))
    rows = []
    with open(curves_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("curves.csv appears empty")
    if "Date" not in rows[0]:
        raise ValueError("curves.csv missing Date column")
    missing = [c for c in requested if c not in rows[0]]
    if missing:
        raise ValueError(f"curves.csv missing requested discount columns: {missing}")

    d0 = datetime.strptime(rows[0]["Date"], "%Y-%m-%d")
    times = []
    by_col: Dict[str, list[float]] = {c: [] for c in requested}
    for r in rows:
        d = datetime.strptime(r["Date"], "%Y-%m-%d")
        times.append((d - d0).days / 365.0)
        for c in requested:
            by_col[c].append(float(r[c]))

    times_arr = np.asarray(times, dtype=float)
    uniq_t, idx = np.unique(times_arr, return_index=True)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for c in requested:
        dfs_arr = np.asarray(by_col[c], dtype=float)
        uniq_df = dfs_arr[idx].copy()
        if uniq_t[0] > 1.0e-12:
            t = np.insert(uniq_t, 0, 0.0)
            df = np.insert(uniq_df, 0, 1.0)
        else:
            t = uniq_t.copy()
            df = uniq_df
            if t[0] == 0.0:
                df[0] = 1.0
        out[c] = (t, df)
    return out


def load_ore_legs_from_flows(
    flows_csv: str,
    trade_id: str = "Swap_20",
    asof_date: Optional[str] = None,
    time_day_counter: str = "ActualActual(ISDA)",
) -> Dict[str, np.ndarray]:
    """Parse ORE flows.csv and return fixed/floating leg cashflow definitions in year-fraction times.

    Returned keys:
      fixed_pay_time, fixed_accrual, fixed_rate, fixed_notional, fixed_sign
      float_pay_time, float_start_time, float_end_time, float_accrual, float_spread, float_notional, float_sign

    This is the most direct route from ORE cashflow reports into the swap pricers in
    this module. The key subtlety is that the *leg* payer/receiver orientation is
    constant even when coupon rates go negative. We therefore infer a constant leg
    sign from ``Amount / Coupon`` where possible instead of taking ``sign(Amount)``
    row-by-row, which would flip signs incorrectly on negative-rate coupons.
    """
    rows = []
    with open(flows_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        trade_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get(trade_key, "") == trade_id and row["FlowType"].startswith("Interest"):
                rows.append(row)

    if not rows:
        raise ValueError(f"no interest cashflows for trade_id='{trade_id}' found in {flows_csv}")

    if asof_date is None:
        # Fallback for stand-alone use; parity scripts should pass the true ORE as-of date.
        asof = min(datetime.strptime(r["AccrualStartDate"], "%Y-%m-%d") for r in rows)
    else:
        asof = datetime.strptime(asof_date, "%Y-%m-%d")

    def to_time(date_str: str) -> float:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return _time_from_dates(asof.date(), d.date(), time_day_counter)

    fixed = [r for r in rows if r["LegNo"] == "0"]
    floating = [r for r in rows if r["LegNo"] == "1"]
    if not fixed or not floating:
        raise ValueError("could not split fixed/floating legs from flows.csv")

    # sort by pay date
    fixed.sort(key=lambda r: r["PayDate"])
    floating.sort(key=lambda r: r["PayDate"])

    out: Dict[str, np.ndarray] = {}

    def infer_leg_sign(rows: list[dict[str, str]]) -> float:
        signed = []
        for r in rows:
            amt = float(r["Amount"])
            cpn_txt = r.get("Coupon", "")
            try:
                cpn = float(cpn_txt)
            except Exception:
                cpn = 0.0
            if abs(cpn) > 1.0e-14:
                signed.append(np.sign(amt / cpn))
        if signed:
            s = float(np.sign(np.median(np.asarray(signed, dtype=float))))
            return s if abs(s) > 0.0 else 1.0
        nonzero_amt = [np.sign(float(r["Amount"])) for r in rows if abs(float(r["Amount"])) > 1.0e-14]
        if nonzero_amt:
            s = float(np.sign(np.median(np.asarray(nonzero_amt, dtype=float))))
            return s if abs(s) > 0.0 else 1.0
        return 1.0

    fixed_leg_sign = infer_leg_sign(fixed)
    float_leg_sign = infer_leg_sign(floating)

    out["fixed_pay_time"] = np.asarray([to_time(r["PayDate"]) for r in fixed], dtype=float)
    out["fixed_accrual"] = np.asarray([float(r["Accrual"]) for r in fixed], dtype=float)
    out["fixed_rate"] = np.asarray([float(r["Coupon"]) for r in fixed], dtype=float)
    out["fixed_notional"] = np.asarray([float(r["Notional"]) for r in fixed], dtype=float)
    out["fixed_sign"] = np.full(len(fixed), fixed_leg_sign, dtype=float)
    out["fixed_amount"] = np.asarray([float(r["Amount"]) for r in fixed], dtype=float)

    out["float_pay_time"] = np.asarray([to_time(r["PayDate"]) for r in floating], dtype=float)
    out["float_start_time"] = np.asarray([to_time(r["AccrualStartDate"]) for r in floating], dtype=float)
    out["float_end_time"] = np.asarray([to_time(r["AccrualEndDate"]) for r in floating], dtype=float)
    out["float_accrual"] = np.asarray([float(r["Accrual"]) for r in floating], dtype=float)
    out["float_notional"] = np.asarray([float(r["Notional"]) for r in floating], dtype=float)
    out["float_sign"] = np.full(len(floating), float_leg_sign, dtype=float)
    out["float_coupon"] = np.asarray([float(r["Coupon"]) for r in floating], dtype=float)
    out["float_amount"] = np.asarray([float(r["Amount"]) for r in floating], dtype=float)
    # Flow exports typically contain the all-in projected coupon rather than a clean
    # decomposition into index forward + spread.  Start with zero spread and let a
    # later calibration helper infer the spread if the user wants dual-curve parity.
    out["float_spread"] = np.zeros_like(out["float_accrual"])
    # Fixing time: flows.csv has no explicit fixing date; use period start (standard convention).
    out["float_fixing_time"] = np.asarray(out["float_start_time"], dtype=float)

    return out


def load_swap_legs_from_portfolio(
    portfolio_xml: str,
    trade_id: str,
    asof_date: str,
    time_day_counter: str = "ActualActual(ISDA)",
) -> Dict[str, np.ndarray]:
    """Build swap legs from ORE trade definition (no flows.csv dependency).

    This reconstructs the same leg arrays from the portfolio XML itself, which is
    useful when flow reports are unavailable or when a schedule needs to be rebuilt
    from the canonical ORE trade description.
    """
    root = ET.parse(portfolio_xml).getroot()
    trade = None
    for t in root.findall("./Trade"):
        if t.attrib.get("id", "") == trade_id:
            trade = t
            break
    if trade is None:
        raise ValueError(f"trade '{trade_id}' not found in {portfolio_xml}")
    if (trade.findtext("./TradeType") or "").strip() != "Swap":
        raise ValueError("only Swap trade type is supported")

    swap = trade.find("./SwapData")
    legs_xml = swap.findall("./LegData") if swap is not None else []
    if len(legs_xml) != 2:
        raise ValueError("expected exactly 2 legs in swap trade")

    asof = datetime.strptime(asof_date, "%Y-%m-%d").date()

    out: Dict[str, np.ndarray] = {}
    for lx in legs_xml:
        ltype = (lx.findtext("./LegType") or "").strip()
        payer = (lx.findtext("./Payer") or "").strip().lower() == "true"
        # ORE convention: Payer=true means we pay this leg (outflow); Payer=false means we receive (inflow).
        sign = -1.0 if payer else 1.0
        notional = float((lx.findtext("./Notionals/Notional") or "0").strip())
        dc = (lx.findtext("./DayCounter") or "A365").strip()
        pay_conv = (lx.findtext("./PaymentConvention") or "F").strip()
        rules = lx.find("./ScheduleData/Rules")
        if rules is None:
            raise ValueError("missing ScheduleData/Rules")
        start = _parse_yyyymmdd((rules.findtext("./StartDate") or "").strip())
        end = _parse_yyyymmdd((rules.findtext("./EndDate") or "").strip())
        tenor = (rules.findtext("./Tenor") or "").strip()
        cal = (rules.findtext("./Calendar") or "TARGET").strip()
        conv = (rules.findtext("./Convention") or pay_conv).strip()

        s_dates, e_dates, p_dates = _build_schedule(start, end, tenor, cal, conv, pay_convention=pay_conv)
        s_t = np.asarray([_time_from_dates(asof, d, time_day_counter) for d in s_dates], dtype=float)
        e_t = np.asarray([_time_from_dates(asof, d, time_day_counter) for d in e_dates], dtype=float)
        p_t = np.asarray([_time_from_dates(asof, d, time_day_counter) for d in p_dates], dtype=float)
        accr = np.asarray([_year_fraction(sd, ed, dc) for sd, ed in zip(s_dates, e_dates)], dtype=float)

        if ltype == "Fixed":
            rate = float((lx.findtext("./FixedLegData/Rates/Rate") or "0").strip())
            out["fixed_pay_time"] = p_t
            out["fixed_accrual"] = accr
            out["fixed_rate"] = np.full_like(accr, rate)
            out["fixed_notional"] = np.full_like(accr, notional)
            out["fixed_sign"] = np.full_like(accr, sign)
            out["fixed_amount"] = sign * notional * rate * accr
        elif ltype == "Floating":
            spread = float((lx.findtext("./FloatingLegData/Spreads/Spread") or "0").strip())
            fixing_days = int((lx.findtext("./FloatingLegData/FixingDays") or "2").strip())
            float_index = (lx.findtext("./FloatingLegData/Index") or "").strip().upper()
            float_index_tenor = float_index.split("-")[-1].upper() if "-" in float_index else ""
            out["float_pay_time"] = p_t
            out["float_start_time"] = s_t
            out["float_end_time"] = e_t
            out["float_accrual"] = accr
            out["float_notional"] = np.full_like(accr, notional)
            out["float_sign"] = np.full_like(accr, sign)
            out["float_spread"] = np.full_like(accr, spread)
            out["float_coupon"] = np.zeros_like(accr)
            out["float_amount"] = np.zeros_like(accr)
            # ORE stores fixing lag as business days relative to the accrual start.
            fix_dates = [_advance_business_days(sd, -fixing_days, cal) for sd in s_dates]
            out["float_fixing_time"] = np.asarray([_time_from_dates(asof, fd, time_day_counter) for fd in fix_dates], dtype=float)
            out["float_index"] = float_index
            out["float_index_tenor"] = float_index_tenor
        else:
            raise ValueError(f"unsupported leg type '{ltype}'")

    required = [
        "fixed_pay_time",
        "fixed_accrual",
        "fixed_rate",
        "fixed_notional",
        "fixed_sign",
        "fixed_amount",
        "float_pay_time",
        "float_start_time",
        "float_end_time",
        "float_accrual",
        "float_notional",
        "float_sign",
        "float_spread",
        "float_coupon",
        "float_amount",
        "float_fixing_time",
    ]
    for k in required:
        if k not in out:
            raise ValueError(f"incomplete swap leg data, missing '{k}'")
    return out


def remaining_schedule(dates: np.ndarray, t: float) -> np.ndarray:
    return dates[dates > t + 1e-12]


def accruals_from_dates(rem_dates: np.ndarray, t: float) -> np.ndarray:
    if rem_dates.size == 0:
        return rem_dates
    prev = np.concatenate(([t], rem_dates[:-1]))
    return rem_dates - prev


def forward_simple_from_discounts(p_start: np.ndarray, p_end: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return (p_start / p_end - 1.0) / tau


def _remaining_schedule_from_index(dates: np.ndarray, t: float) -> np.ndarray:
    start = int(np.searchsorted(dates, t + 1.0e-12, side="right"))
    return dates[start:]


def _discount_bond_block(model, p0, t: float, maturities: np.ndarray, x_t: np.ndarray, p_t: float) -> np.ndarray:
    if maturities.size == 0:
        return np.empty((0, x_t.size), dtype=float)
    p_T = np.fromiter((float(p0(float(T))) for T in maturities), dtype=float, count=maturities.size)
    return model.discount_bond_paths(t, maturities, x_t, p_t, p_T)


def payer_swap_npv_at_time(model, p0, fixed_dates, float_dates, t, x_t, fixed_rate, trade_def):
    """Payer fixed IRS value = floating leg PV - fixed leg PV.

    This is the minimal single-curve identity used in older notebook examples before
    the more explicit ORE-leg-array valuation helpers were added.
    """
    notional_fixed = trade_def["SwapData"]["LegData"][0]["Notional"]
    notional_float = trade_def["SwapData"]["LegData"][1]["Notional"]
    x = np.asarray(x_t, dtype=float)

    rem_fixed = _remaining_schedule_from_index(fixed_dates, t)
    rem_float = _remaining_schedule_from_index(float_dates, t)

    if rem_fixed.size == 0 and rem_float.size == 0:
        return np.zeros_like(x, dtype=float)

    p_t = p0(t)

    fixed_leg_pv = np.zeros_like(x, dtype=float)
    if rem_fixed.size > 0:
        p_t_T_fixed = _discount_bond_block(model, p0, t, rem_fixed, x, p_t)
        tau_fixed = accruals_from_dates(rem_fixed, t)
        fixed_leg_pv = notional_fixed * fixed_rate * (tau_fixed @ p_t_T_fixed)

    float_leg_pv = np.zeros_like(x, dtype=float)
    if rem_float.size > 0:
        p_t_T_float = _discount_bond_block(model, p0, t, rem_float, x, p_t)
        p_start = np.empty_like(p_t_T_float)
        p_start[0, :] = 1.0
        if p_t_T_float.shape[0] > 1:
            p_start[1:, :] = p_t_T_float[:-1, :]
        float_leg_pv = notional_float * np.sum(p_start - p_t_T_float, axis=0)

    return float_leg_pv - fixed_leg_pv


def swap_npv_from_ore_legs(model, p0, legs: Dict[str, np.ndarray], t: float, x_t: np.ndarray) -> np.ndarray:
    """Pathwise swap NPV from ORE-like leg arrays parsed from flows.csv.

    The leg arrays are assumed to follow ORE's cashflow sign convention, so the
    function can just sum discounted fixed and floating cashflows from the netting
    set perspective.
    """
    x = np.asarray(x_t, dtype=float)
    pv = np.zeros_like(x, dtype=float)
    p_t = p0(t)

    # Fixed leg
    mask_f = legs["fixed_pay_time"] > t + 1e-12
    if np.any(mask_f):
        pay = legs["fixed_pay_time"][mask_f]
        disc = _discount_bond_block(model, p0, t, pay, x, p_t)
        cash = legs["fixed_amount"][mask_f]
        pv += np.sum(cash[:, None] * disc, axis=0)

    # Floating coupons are handled in the same buckets practitioners use when
    # reconciling against ORE cashflows:
    # future periods still need projection, current periods are already fixed, and
    # past periods have dropped out.
    # 1) Not yet started: model projected forward.
    mask_future = (legs["float_pay_time"] > t + 1e-12) & (legs["float_start_time"] >= t - 1e-12)
    if np.any(mask_future):
        s = legs["float_start_time"][mask_future]
        e = legs["float_end_time"][mask_future]
        pay = legs["float_pay_time"][mask_future]
        tau = legs["float_accrual"][mask_future]
        n = legs["float_notional"][mask_future]
        sign = legs["float_sign"][mask_future]
        spread = legs["float_spread"][mask_future]

        p_ts = _discount_bond_block(model, p0, t, s, x, p_t)
        p_te = _discount_bond_block(model, p0, t, e, x, p_t)
        p_tp = _discount_bond_block(model, p0, t, pay, x, p_t)
        fwd = (p_ts / p_te - 1.0) / tau[:, None]
        cash = sign[:, None] * n[:, None] * (fwd + spread[:, None]) * tau[:, None]
        pv += np.sum(cash * p_tp, axis=0)

    # 2) In accrual period: treat coupon as already fixed and discount deterministic cashflow.
    mask_in_period = (
        (legs["float_pay_time"] > t + 1e-12)
        & (legs["float_start_time"] < t - 1e-12)
        & (legs["float_end_time"] >= t - 1e-12)
    )
    if np.any(mask_in_period):
        pay = legs["float_pay_time"][mask_in_period]
        amount = legs["float_amount"][mask_in_period]
        p_tp = _discount_bond_block(model, p0, t, pay, x, p_t)
        pv += np.sum(amount[:, None] * p_tp, axis=0)

    return pv


def swap_npv_from_ore_legs_dual_curve(
    model,
    p0_disc,
    p0_fwd,
    legs: Dict[str, np.ndarray],
    t: float,
    x_t: np.ndarray,
    realized_float_coupon: Optional[np.ndarray] = None,
    use_node_interpolation: bool = False,
) -> np.ndarray:
    """Pathwise swap NPV with discounting on p0_disc and forwarding on p0_fwd.

    Forwarding curve discount factors are mapped from discounting curve state using
    a deterministic basis ratio:
      P_f(t,T) = P_d(t,T) * (B(T) / B(t)),  B(u)=P_f(0,u)/P_d(0,u)

    This is one of the key ORE-related approximations in the Python toolkit. ORE can
    carry richer multi-curve state; here we reuse a single LGM factor and transport
    it onto the forwarding curve through the deterministic t=0 basis term structure.

    ``use_node_interpolation`` keeps the older "simulate node tenors, then interpolate
    discount factors" workflow available for diagnostics.  Exact discount-bond
    evaluation is the default because it is materially closer to ORE XVA exposure
    profiles than the node interpolation shortcut on the benchmark parity cases.
    """
    x = np.asarray(x_t, dtype=float)
    pv = np.zeros_like(x, dtype=float)
    p_t_d = p0_disc(t)

    node_tenors = np.asarray(legs.get("node_tenors", np.array([], dtype=float)), dtype=float)
    # At t=0 we know the exact initial curves, so interpolating off simulation nodes
    # only introduces approximation error. This was the main source of the large
    # t0 parity gap against ORE for swap cashflows.
    use_nodes = bool(use_node_interpolation) and node_tenors.size > 0 and t > 1.0e-14
    grid = t + node_tenors if use_nodes else np.array([], dtype=float)
    bt = p0_fwd(t) / p0_disc(t)
    p_nodes_d = None
    logp = None
    slope = None
    if use_nodes:
        # Some ORE workflows expose simulation node tenors rather than requiring bond
        # evaluation at every maturity.  When available, interpolate in log-discount
        # space off those nodes to mimic the "read simulated curve nodes" workflow.
        p_nodes_d = _discount_bond_block(model, p0_disc, t, grid, x, p_t_d)
        logp = np.log(np.clip(p_nodes_d, 1.0e-18, None))
        slope = (logp[-1] - logp[-2]) / max(grid[-1] - grid[-2], 1.0e-12)

    def interp_from_nodes_batch(T: np.ndarray) -> np.ndarray:
        maturities = np.asarray(T, dtype=float)
        if maturities.ndim != 1:
            raise ValueError("maturities must be one-dimensional")
        if maturities.size == 0:
            return np.empty((0, x.size), dtype=float)
        if not use_nodes:
            return _discount_bond_block(model, p0_disc, t, maturities, x, p_t_d)

        out = np.empty((maturities.size, x.size), dtype=float)
        immediate = maturities <= t + 1.0e-14
        if np.any(immediate):
            out[immediate, :] = 1.0

        exact_short = (~immediate) & (maturities <= grid[0])
        if np.any(exact_short):
            out[exact_short, :] = _discount_bond_block(model, p0_disc, t, maturities[exact_short], x, p_t_d)

        extrap = maturities >= grid[-1]
        extrap &= ~immediate
        extrap &= ~exact_short
        if np.any(extrap):
            out[extrap, :] = np.exp(logp[-1][None, :] + (maturities[extrap] - grid[-1])[:, None] * slope[None, :])

        interior = ~(immediate | exact_short | extrap)
        if np.any(interior):
            mids = maturities[interior]
            j = np.searchsorted(grid, mids, side="right")
            left = j - 1
            denom = np.maximum(grid[j] - grid[left], 1.0e-12)
            w = (mids - grid[left]) / denom
            out[interior, :] = np.exp((1.0 - w)[:, None] * logp[left, :] + w[:, None] * logp[j, :])

        return out

    def map_forward_bond_from_disc_batch(T: np.ndarray, p_t_T_disc: np.ndarray) -> np.ndarray:
        maturities = np.asarray(T, dtype=float)
        bT = np.fromiter((float(p0_fwd(float(m))) / p0_disc(float(m)) for m in maturities), dtype=float, count=maturities.size)
        return p_t_T_disc * (bT / bt)[:, None]

    # Fixed leg discounted on discount curve
    mask_f = legs["fixed_pay_time"] > t + 1e-12
    if np.any(mask_f):
        pay = legs["fixed_pay_time"][mask_f]
        disc = interp_from_nodes_batch(pay)
        cash = legs["fixed_amount"][mask_f]
        pv += np.sum(cash[:, None] * disc, axis=0)

    fix_t = np.asarray(legs.get("float_fixing_time", legs["float_start_time"]), dtype=float)
    pay_all = legs["float_pay_time"]
    live = pay_all > t + 1e-12
    if np.any(live):
        s = legs["float_start_time"][live]
        e = legs["float_end_time"][live]
        pay = pay_all[live]
        tau = legs["float_accrual"][live]
        n = legs["float_notional"][live]
        sign = legs["float_sign"][live]
        spread = legs["float_spread"][live]
        fixed = fix_t[live] <= t + 1.0e-12

        p_tp_d = interp_from_nodes_batch(pay)
        amount = np.zeros((pay.size, x.size), dtype=float)

        # Fixed coupons: fixing already known, only discounting remains.
        if np.any(fixed):
            if "float_amount" in legs and realized_float_coupon is None:
                fixed_amount = np.asarray(legs["float_amount"][live][fixed], dtype=float)
                amount[fixed, :] = np.tile(fixed_amount[:, None], (1, x_t.size))
            else:
                if realized_float_coupon is not None:
                    coupon_fix = realized_float_coupon[live][fixed]
                else:
                    coupon_fix = np.tile(legs["float_coupon"][live][fixed][:, None], (1, x_t.size))
                amount[fixed, :] = sign[fixed, None] * n[fixed, None] * coupon_fix * tau[fixed, None]

        # Unfixed coupons: project with the forwarding curve, discount with the
        # discounting curve, exactly as in an OIS-discounted ORE setup.
        if np.any(~fixed):
            s2 = s[~fixed]
            e2 = e[~fixed]
            tau2 = tau[~fixed]
            n2 = n[~fixed]
            sign2 = sign[~fixed]
            spread2 = spread[~fixed]
            p_ts_d2 = interp_from_nodes_batch(s2)
            p_te_d2 = interp_from_nodes_batch(e2)
            p_ts_f2 = map_forward_bond_from_disc_batch(s2, p_ts_d2)
            p_te_f2 = map_forward_bond_from_disc_batch(e2, p_te_d2)
            fwd2 = (p_ts_f2 / p_te_f2 - 1.0) / tau2[:, None]
            amount[~fixed, :] = sign2[:, None] * n2[:, None] * (fwd2 + spread2[:, None]) * tau2[:, None]

        pv += np.sum(amount * p_tp_d, axis=0)

    return pv


def compute_realized_float_coupons(
    model,
    p0_disc,
    p0_fwd,
    legs: Dict[str, np.ndarray],
    sim_times: np.ndarray,
    x_paths_on_sim_grid: np.ndarray,
) -> np.ndarray:
    """Pathwise full floating coupon (forward+spread) locked at each fixing time.

    This belongs in the core helper module because it is required for parity with
    ORE whenever coupons are already fixed at a given exposure date.  Using static
    quoted coupons can materially bias the floating leg and therefore EPE/ENE.
    """
    s = np.asarray(legs["float_start_time"], dtype=float)
    e = np.asarray(legs["float_end_time"], dtype=float)
    tau = np.asarray(legs["float_accrual"], dtype=float)
    spr = np.asarray(legs["float_spread"], dtype=float)
    fix_t = np.asarray(legs.get("float_fixing_time", s), dtype=float)
    quoted_coupon = np.asarray(legs.get("float_coupon", np.zeros_like(s)), dtype=float)

    n_cf = s.size
    n_paths = x_paths_on_sim_grid.shape[1]
    out = np.zeros((n_cf, n_paths), dtype=float)

    for i in range(n_cf):
        if tau[i] <= 0.0:
            out[i, :] = quoted_coupon[i]
            continue
        ft = float(fix_t[i])
        if ft <= 1.0e-12:
            ps = float(p0_fwd(max(0.0, float(s[i]))))
            pe = float(p0_fwd(float(e[i])))
            fwd = (ps / pe - 1.0) / float(tau[i])
            out[i, :] = fwd + float(spr[i])
            continue
        j = int(np.searchsorted(sim_times, ft))
        if j >= sim_times.size or abs(float(sim_times[j]) - ft) > 1.0e-12:
            raise ValueError(f"fixing time {ft} not present on simulation grid")
        x_fix = x_paths_on_sim_grid[j, :]
        p_ft = float(p0_disc(ft))
        p_t_s_d = model.discount_bond(ft, float(s[i]), x_fix, p_ft, float(p0_disc(float(s[i]))))
        p_t_e_d = model.discount_bond(ft, float(e[i]), x_fix, p_ft, float(p0_disc(float(e[i]))))
        bt = float(p0_fwd(ft) / p0_disc(ft))
        bs = float(p0_fwd(float(s[i])) / p0_disc(float(s[i])))
        be = float(p0_fwd(float(e[i])) / p0_disc(float(e[i])))
        p_t_s_f = p_t_s_d * (bs / bt)
        p_t_e_f = p_t_e_d * (be / bt)
        fwd_path = (p_t_s_f / p_t_e_f - 1.0) / float(tau[i])
        out[i, :] = fwd_path + float(spr[i])
    return out


def calibrate_float_spreads_from_coupon(
    legs: Dict[str, np.ndarray],
    p0_fwd: Callable[[float], float],
    t0: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Return a copy of legs with float_spread inferred from flow coupons at t0.

    For each future coupon i:
      spread_i = coupon_i - (P_f(t0,s_i)/P_f(t0,e_i)-1)/tau_i

    This is useful when ORE has exported an all-in coupon but the Python pricer wants
    a clean explicit spread component for dual-curve revaluation.
    """
    out = {k: np.array(v, copy=True) for k, v in legs.items()}
    spread = np.array(out.get("float_spread", np.zeros_like(out["float_accrual"])), copy=True, dtype=float)

    for i in range(spread.size):
        s = float(out["float_start_time"][i])
        e = float(out["float_end_time"][i])
        tau = float(out["float_accrual"][i])
        if e <= t0 + 1e-12 or tau <= 0.0:
            continue
        ps = float(p0_fwd(max(t0, s)))
        pe = float(p0_fwd(e))
        fwd0 = (ps / pe - 1.0) / tau
        if "float_coupon" in out and out["float_coupon"][i] != 0.0:
            spread[i] = float(out["float_coupon"][i]) - fwd0
        out["float_coupon"][i] = fwd0 + spread[i]

    out["float_spread"] = spread
    return out


def apply_parallel_float_spread_shift_to_match_npv(
    legs: Dict[str, np.ndarray],
    p0_disc: Callable[[float], float],
    target_npv: float,
    t0: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Return a copy of legs with a parallel spread shift so t0 PV matches target_npv.

    In reconciliation exercises this is a practical way to absorb small convention or
    curve-construction differences relative to ORE into a single deterministic spread
    adjustment.
    """
    out = {k: np.array(v, copy=True) for k, v in legs.items()}

    node_tenors = np.asarray(out.get("node_tenors", np.array([], dtype=float)), dtype=float)

    def df_pay(T: float) -> float:
        if node_tenors.size == 0:
            return float(p0_disc(T))
        grid = t0 + node_tenors
        vals = np.asarray([p0_disc(t) for t in grid], dtype=float)
        if T <= t0 + 1.0e-14:
            return 1.0
        if T <= grid[0]:
            return float(p0_disc(T))
        logv = np.log(np.clip(vals, 1.0e-18, None))
        if T >= grid[-1]:
            t1, t2 = grid[-2], grid[-1]
            slope = (logv[-1] - logv[-2]) / max(t2 - t1, 1.0e-12)
            return float(np.exp(logv[-1] + slope * (T - t2)))
        j = int(np.searchsorted(grid, T, side="right"))
        t1, t2 = grid[j - 1], grid[j]
        w = (T - t1) / max(t2 - t1, 1.0e-12)
        return float(np.exp((1.0 - w) * logv[j - 1] + w * logv[j]))

    # Rebuild a deterministic t0 PV from the current leg setup, then solve the
    # one-dimensional shift that makes the helper trade line up with the target PV.
    mask_f = out["fixed_pay_time"] > t0 + 1e-12
    pv_fixed = float(np.sum(out["fixed_amount"][mask_f] * np.array([df_pay(T) for T in out["fixed_pay_time"][mask_f]])))

    mask_l = out["float_pay_time"] > t0 + 1e-12
    if not np.any(mask_l):
        return out

    s = out["float_start_time"][mask_l]
    e = out["float_end_time"][mask_l]
    tau = out["float_accrual"][mask_l]
    sign = out["float_sign"][mask_l]
    n = out["float_notional"][mask_l]
    spr = out["float_spread"][mask_l]
    df_pay_arr = np.array([df_pay(T) for T in out["float_pay_time"][mask_l]])

    fwd = np.array([(p0_disc(si) / p0_disc(ei) - 1.0) / ti for si, ei, ti in zip(s, e, tau)], dtype=float)
    pv_float = float(np.sum(sign * n * (fwd + spr) * tau * df_pay_arr))
    pv_curr = pv_fixed + pv_float

    denom = float(np.sum(sign * n * tau * df_pay_arr))
    if abs(denom) < 1.0e-16:
        return out

    shift = (target_npv - pv_curr) / denom
    out["float_spread"][mask_l] = out["float_spread"][mask_l] + shift
    return out


def par_rate_from_trade(model, p0, fixed_dates, float_dates, trade_def) -> float:
    x0_vec = np.array([0.0])
    notional_fixed = trade_def["SwapData"]["LegData"][0]["Notional"]
    pv01_0 = notional_fixed * np.sum(accruals_from_dates(fixed_dates, 0.0) * np.array([p0(T) for T in fixed_dates]))
    float_pv_0 = payer_swap_npv_at_time(model, p0, fixed_dates, float_dates, 0.0, x0_vec, 0.0, trade_def)[0]
    return float(float_pv_0 / pv01_0)


def parse_lgm_params_from_simulation_xml(simulation_xml: str, ccy_key: str = "EUR") -> Dict[str, object]:
    """Extract LGM parameterization from an ORE simulation config.

    This reads the same calibration template data ORE uses before calibration, not
    necessarily the final calibrated parameters.
    """
    root = ET.parse(simulation_xml).getroot()
    models = root.find("./CrossAssetModel/InterestRateModels")
    if models is None:
        raise ValueError("simulation xml is missing CrossAssetModel/InterestRateModels")

    node = models.find(f"./LGM[@ccy='{ccy_key}']")
    if node is None:
        node = models.find("./LGM[@ccy='default']")
    if node is None:
        raise ValueError(f"no LGM node found for ccy '{ccy_key}' or 'default'")

    vol_node = node.find("./Volatility")
    rev_node = node.find("./Reversion")
    trans_node = node.find("./ParameterTransformation")
    if vol_node is None or rev_node is None or trans_node is None:
        raise ValueError("LGM node is missing Volatility/Reversion/ParameterTransformation blocks")

    def parse_grid(txt: Optional[str]) -> np.ndarray:
        txt = (txt or "").strip()
        if not txt:
            return np.array([], dtype=float)
        return np.asarray([float(x.strip()) for x in txt.split(",") if x.strip()], dtype=float)

    alpha_times = parse_grid(vol_node.findtext("./TimeGrid"))
    alpha_vals = parse_grid(vol_node.findtext("./InitialValue"))
    kappa_times = parse_grid(rev_node.findtext("./TimeGrid"))
    kappa_vals = parse_grid(rev_node.findtext("./InitialValue"))
    shift = float((trans_node.findtext("./ShiftHorizon") or "0").strip())
    scaling = float((trans_node.findtext("./Scaling") or "1").strip())

    calibrate_vol = (vol_node.findtext("./Calibrate") or "N").strip().upper() == "Y"
    calibrate_kappa = (rev_node.findtext("./Calibrate") or "N").strip().upper() == "Y"

    return {
        "alpha_times": alpha_times,
        "alpha_values": alpha_vals,
        "kappa_times": kappa_times,
        "kappa_values": kappa_vals,
        "shift": shift,
        "scaling": scaling,
        "calibrate_vol": calibrate_vol,
        "calibrate_kappa": calibrate_kappa,
    }


def parse_lgm_params_from_calibration_xml(calibration_xml: str, ccy_key: str = "EUR") -> Dict[str, object]:
    """Extract calibrated LGM parameters from ORE calibration.xml (CrossAssetModelData).

    Use this when you want the post-calibration numbers produced by ORE rather than
    the starting configuration from the simulation XML.
    """
    root = ET.parse(calibration_xml).getroot()
    models = root.find("./InterestRateModels")
    if models is None:
        raise ValueError("calibration.xml is missing InterestRateModels")

    node = models.find(f"./LGM[@key='{ccy_key}']")
    if node is None:
        # Some files may still use ccy attribute.
        node = models.find(f"./LGM[@ccy='{ccy_key}']")
    if node is None:
        raise ValueError(f"no LGM node found for key '{ccy_key}' in calibration xml")

    vol_node = node.find("./Volatility")
    rev_node = node.find("./Reversion")
    trans_node = node.find("./ParameterTransformation")
    if vol_node is None or rev_node is None or trans_node is None:
        raise ValueError("LGM node is missing Volatility/Reversion/ParameterTransformation blocks")

    def parse_grid(txt: Optional[str]) -> np.ndarray:
        txt = (txt or "").strip()
        if not txt:
            return np.array([], dtype=float)
        return np.asarray([float(x.strip()) for x in txt.split(",") if x.strip()], dtype=float)

    alpha_times = parse_grid(vol_node.findtext("./TimeGrid"))
    alpha_vals = parse_grid(vol_node.findtext("./InitialValue"))
    kappa_times = parse_grid(rev_node.findtext("./TimeGrid"))
    kappa_vals = parse_grid(rev_node.findtext("./InitialValue"))
    shift = float((trans_node.findtext("./ShiftHorizon") or "0").strip())
    scaling = float((trans_node.findtext("./Scaling") or "1").strip())

    return {
        "alpha_times": alpha_times,
        "alpha_values": alpha_vals,
        "kappa_times": kappa_times,
        "kappa_values": kappa_vals,
        "shift": shift,
        "scaling": scaling,
        "calibrate_vol": False,
        "calibrate_kappa": False,
    }


_TENOR_RE = re.compile(r"^([0-9]+)([YMWD])$")


def _tenor_to_years(tenor: str) -> float:
    m = _TENOR_RE.match(tenor.strip().upper())
    if m is None:
        raise ValueError(f"unsupported tenor '{tenor}'")
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "Y":
        return n
    if unit == "M":
        return n / 12.0
    if unit == "W":
        return n / 52.0
    return n / 365.0


def load_ore_default_curve_inputs(
    todaysmarket_xml: str,
    market_data_file: str,
    cpty_name: str = "CPTY_A",
) -> Dict[str, object]:
    """Load hazard/recovery inputs for a named counterparty from ORE market data files.

    This parses the ORE market-data naming convention for default curves and returns
    arrays suitable for the lightweight survival-probability helper below.
    """
    tm_root = ET.parse(todaysmarket_xml).getroot()
    dcurves = tm_root.find("./DefaultCurves[@id='default']")
    if dcurves is None:
        raise ValueError("todaysmarket.xml missing DefaultCurves id='default'")
    mapping = {n.attrib.get("name", ""): (n.text or "").strip() for n in dcurves.findall("./DefaultCurve")}
    if cpty_name not in mapping:
        raise ValueError(f"default curve mapping for '{cpty_name}' not found in todaysmarket.xml")

    # Default/USD/CPTY_A_SR_USD -> name=CPTY_A, seniority=SR, ccy=USD
    dc = mapping[cpty_name].split("/")
    if len(dc) < 3:
        raise ValueError(f"unexpected default curve handle '{mapping[cpty_name]}'")
    suffix = dc[-1]
    m = re.match(r"(.+)_([A-Z]+)_([A-Z]{3})$", suffix)
    if m is None:
        raise ValueError(f"cannot parse default curve suffix '{suffix}'")
    ref_name = m.group(1)
    seniority = m.group(2)
    ccy = m.group(3)

    hazard_prefix = f"HAZARD_RATE/RATE/{ref_name}/{seniority}/{ccy}/"
    cds_prefix = f"CDS/CREDIT_SPREAD/{ref_name}/{seniority}/{ccy}/"
    recovery_key = f"RECOVERY_RATE/RATE/{ref_name}/{seniority}/{ccy}"

    hazard_points = []
    cds_points = []
    recovery = None
    with open(market_data_file, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 3:
                continue
            key = toks[1]
            val = float(toks[2])
            if key == recovery_key:
                recovery = val
            elif key.startswith(hazard_prefix):
                tenor = key[len(hazard_prefix) :]
                hazard_points.append((_tenor_to_years(tenor), val))
            elif key.startswith(cds_prefix):
                tenor = key[len(cds_prefix) :]
                cds_points.append((_tenor_to_years(tenor), val))

    if recovery is None:
        raise ValueError(f"recovery not found for key '{recovery_key}'")
    if not hazard_points and cds_points:
        # ORE builds a default term structure from CDS spreads plus recovery.
        # For the lightweight Python path we use the same first-order flat-hazard
        # approximation per pillar: lambda ~= spread / LGD.
        lgd = max(1.0 - float(recovery), 1.0e-12)
        hazard_points = sorted((t, s / lgd) for t, s in cds_points)
    if not hazard_points:
        raise ValueError(
            f"hazard curve points not found for prefix '{hazard_prefix}' or '{cds_prefix}'"
        )
    hazard_points = sorted(hazard_points, key=lambda p: p[0])
    times = np.asarray([p[0] for p in hazard_points], dtype=float)
    hazard = np.asarray([p[1] for p in hazard_points], dtype=float)
    return {
        "counterparty": cpty_name,
        "curve_handle": mapping[cpty_name],
        "reference_name": ref_name,
        "seniority": seniority,
        "ccy": ccy,
        "recovery": float(recovery),
        "hazard_times": times,
        "hazard_rates": hazard,
    }


def survival_probability_from_hazard(
    times: np.ndarray,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
) -> np.ndarray:
    """Piecewise-flat hazard survival probabilities at requested times.

    ORE default curves are typically represented as pillar tenors with piecewise-flat
    hazards between pillars; this helper reproduces that simple survival construction.
    """
    t = np.asarray(times, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("times must be non-negative")
    if np.any(np.diff(hazard_times) < 0.0):
        raise ValueError("hazard_times must be non-decreasing")
    if hazard_times.size != hazard_rates.size:
        raise ValueError("hazard_times and hazard_rates must have same size")
    if hazard_times.size == 0:
        raise ValueError("hazard curve is empty")

    knots = np.concatenate(([0.0], hazard_times))
    lambdas = np.asarray(hazard_rates, dtype=float)
    out = np.empty_like(t)
    for i, x in enumerate(t):
        acc = 0.0
        for j in range(lambdas.size):
            a = knots[j]
            b = min(x, knots[j + 1])
            if b > a:
                acc += lambdas[j] * (b - a)
            if x <= knots[j + 1]:
                break
        if x > knots[-1]:
            acc += lambdas[-1] * (x - knots[-1])
        out[i] = np.exp(-acc)
    return out


def aggregate_exposure_profile_from_npv_paths(npv_paths: np.ndarray) -> Dict[str, np.ndarray]:
    """Return EE/EPE/ENE profiles from pathwise NPV matrix [n_times, n_paths].

    The ENE convention here is positive magnitude:
      ENE(t) = E[max(-V(t), 0)].
    """
    v = np.asarray(npv_paths, dtype=float)
    if v.ndim != 2:
        raise ValueError("npv_paths must be 2D [n_times, n_paths]")
    return {
        "ee": np.mean(v, axis=1),
        "epe": np.mean(np.maximum(v, 0.0), axis=1),
        "ene": np.mean(np.maximum(-v, 0.0), axis=1),
    }


def deflate_lgm_npv_paths(
    model,
    p0_disc: Callable[[float], float],
    times: np.ndarray,
    x_paths: np.ndarray,
    npv_paths: np.ndarray,
) -> np.ndarray:
    """Return ORE-style numeraire-deflated LGM NPVs on a common simulation grid.

    ORE's XVA cube stores base-ccy NPVs divided by the simulation numeraire. This
    helper applies the same transformation to raw Python LGM NPVs:

      deflated_npv(t) = npv(t) / N(t)

    where ``N(t)`` is the LGM numeraire under ``p0_disc``.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(x_paths, dtype=float)
    v = np.asarray(npv_paths, dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if x.ndim != 2 or v.ndim != 2:
        raise ValueError("x_paths and npv_paths must be 2D [n_times, n_paths]")
    if x.shape != v.shape:
        raise ValueError("x_paths and npv_paths must have the same shape")
    if x.shape[0] != t.size:
        raise ValueError("times length must match the first dimension of x_paths/npv_paths")

    out = np.empty_like(v, dtype=float)
    for i, ti in enumerate(t):
        out[i, :] = v[i, :] / model.numeraire_lgm(float(ti), x[i, :], p0_disc)
    return out


def compute_xva_from_exposure_profile(
    times: np.ndarray,
    epe: np.ndarray,
    ene: np.ndarray,
    discount: np.ndarray,
    survival_cpty: np.ndarray,
    survival_own: np.ndarray,
    recovery_cpty: float = 0.40,
    recovery_own: float = 0.40,
    funding_spread: float = 0.0,
    exposure_discounting: str = "discount_curve",
    funding_discount_borrow: Optional[np.ndarray] = None,
    funding_discount_lend: Optional[np.ndarray] = None,
    funding_discount_ois: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | float]:
    """Compute unilateral CVA/DVA/FVA and incremental terms on one exposure grid."""
    t = np.asarray(times, dtype=float)
    epe_arr = np.asarray(epe, dtype=float)
    ene_arr = np.asarray(ene, dtype=float)
    df = np.asarray(discount, dtype=float)
    q_c = np.asarray(survival_cpty, dtype=float)
    q_b = np.asarray(survival_own, dtype=float)

    if not (t.shape == epe_arr.shape == ene_arr.shape == df.shape == q_c.shape == q_b.shape):
        raise ValueError("times/epe/ene/discount/survivals must share the same 1D shape")
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if t.size < 2:
        raise ValueError("need at least two time points")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")

    mode = str(exposure_discounting).strip().lower()
    if mode not in ("discount_curve", "numeraire_deflated"):
        raise ValueError("exposure_discounting must be 'discount_curve' or 'numeraire_deflated'")

    lgd_c = 1.0 - float(recovery_cpty)
    lgd_b = 1.0 - float(recovery_own)
    f_spread = float(funding_spread)

    dt = np.diff(t)
    dpd_c = np.zeros_like(t)
    dpd_b = np.zeros_like(t)
    dpd_c[1:] = np.clip(q_c[:-1] - q_c[1:], 0.0, None)
    dpd_b[1:] = np.clip(q_b[:-1] - q_b[1:], 0.0, None)

    discount_weight = df if mode == "discount_curve" else np.ones_like(df)

    cva_terms = np.zeros_like(t)
    dva_terms = np.zeros_like(t)
    fva_terms = np.zeros_like(t)
    fba_terms = np.zeros_like(t)
    fca_terms = np.zeros_like(t)
    cva_terms[1:] = lgd_c * discount_weight[1:] * epe_arr[1:] * dpd_c[1:]
    dva_terms[1:] = lgd_b * discount_weight[1:] * ene_arr[1:] * dpd_b[1:]
    if funding_discount_borrow is not None or funding_discount_lend is not None or funding_discount_ois is not None:
        if funding_discount_borrow is None or funding_discount_lend is None or funding_discount_ois is None:
            raise ValueError(
                "funding_discount_borrow, funding_discount_lend and funding_discount_ois must be provided together"
            )
        df_b = np.asarray(funding_discount_borrow, dtype=float)
        df_l = np.asarray(funding_discount_lend, dtype=float)
        df_o = np.asarray(funding_discount_ois, dtype=float)
        if not (df_b.shape == df_l.shape == df_o.shape == t.shape):
            raise ValueError("funding discount arrays must share the same 1D shape as times")
        surv_joint = q_c[:-1] * q_b[:-1]
        dcf_borrow = df_b[:-1] / df_b[1:] - df_o[:-1] / df_o[1:]
        dcf_lend = df_l[:-1] / df_l[1:] - df_o[:-1] / df_o[1:]
        fca_terms[1:] = surv_joint * epe_arr[1:] * dcf_borrow
        fba_terms[1:] = surv_joint * ene_arr[1:] * dcf_lend
        fva_terms = fba_terms + fca_terms
    else:
        fva_terms[1:] = f_spread * discount_weight[1:] * epe_arr[1:] * dt

    cva = float(np.sum(cva_terms))
    dva = float(np.sum(dva_terms))
    fba = float(np.sum(fba_terms))
    fca = float(np.sum(fca_terms))
    fva = float(np.sum(fva_terms))
    return {
        "dt": dt,
        "dpd_cpty": dpd_c,
        "dpd_own": dpd_b,
        "cva_terms": cva_terms,
        "dva_terms": dva_terms,
        "fba_terms": fba_terms,
        "fca_terms": fca_terms,
        "fva_terms": fva_terms,
        "cva": cva,
        "dva": dva,
        "fba": fba,
        "fca": fca,
        "fva": fva,
        "xva_total": float(cva - dva + fva),
    }


def compute_xva_from_npv_paths(
    times: np.ndarray,
    npv_paths: np.ndarray,
    discount: np.ndarray,
    survival_cpty: np.ndarray,
    survival_own: np.ndarray,
    recovery_cpty: float = 0.40,
    recovery_own: float = 0.40,
    funding_spread: float = 0.0,
    exposure_discounting: str = "discount_curve",
    funding_discount_borrow: Optional[np.ndarray] = None,
    funding_discount_lend: Optional[np.ndarray] = None,
    funding_discount_ois: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | float]:
    """Compute EE/EPE/ENE and XVA stack from one pathwise netting set matrix."""
    exp = aggregate_exposure_profile_from_npv_paths(npv_paths)
    xva = compute_xva_from_exposure_profile(
        times=times,
        epe=exp["epe"],
        ene=exp["ene"],
        discount=discount,
        survival_cpty=survival_cpty,
        survival_own=survival_own,
        recovery_cpty=recovery_cpty,
        recovery_own=recovery_own,
        funding_spread=funding_spread,
        exposure_discounting=exposure_discounting,
        funding_discount_borrow=funding_discount_borrow,
        funding_discount_lend=funding_discount_lend,
        funding_discount_ois=funding_discount_ois,
    )
    return {**exp, **xva}


def aggregate_portfolio_npv_paths(
    npv_paths_by_trade: Mapping[str, np.ndarray],
    trade_to_netting_set: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, np.ndarray] | np.ndarray]:
    """Aggregate trade path matrices into netting-set matrices and total portfolio."""
    if not npv_paths_by_trade:
        raise ValueError("npv_paths_by_trade must not be empty")

    ref_shape: Optional[Tuple[int, int]] = None
    by_ns: Dict[str, np.ndarray] = {}
    for trade_id, mat in npv_paths_by_trade.items():
        arr = np.asarray(mat, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"trade '{trade_id}' path matrix must be 2D [n_times, n_paths]")
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(f"trade '{trade_id}' has shape {arr.shape}, expected {ref_shape}")

        ns = "PORTFOLIO"
        if trade_to_netting_set is not None:
            ns = str(trade_to_netting_set.get(trade_id, "PORTFOLIO"))

        if ns not in by_ns:
            by_ns[ns] = np.zeros_like(arr)
        by_ns[ns] += arr

    portfolio = np.zeros(ref_shape, dtype=float)  # type: ignore[arg-type]
    for arr in by_ns.values():
        portfolio += arr
    return {"by_netting_set": by_ns, "portfolio": portfolio}


def compute_portfolio_xva_from_trade_paths(
    times: np.ndarray,
    npv_paths_by_trade: Mapping[str, np.ndarray],
    discount: np.ndarray,
    survival_cpty: np.ndarray,
    survival_own: np.ndarray,
    recovery_cpty: float = 0.40,
    recovery_own: float = 0.40,
    funding_spread: float = 0.0,
    trade_to_netting_set: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """Compute netting-set and portfolio XVA from trade-level path matrices."""
    agg = aggregate_portfolio_npv_paths(npv_paths_by_trade, trade_to_netting_set=trade_to_netting_set)
    by_ns = agg["by_netting_set"]
    if not isinstance(by_ns, dict):
        raise ValueError("internal aggregation error: expected netting-set mapping")

    ns_out: Dict[str, Dict[str, np.ndarray | float]] = {}
    cva_sum = 0.0
    dva_sum = 0.0
    fva_sum = 0.0
    for ns, mat in by_ns.items():
        pack = compute_xva_from_npv_paths(
            times=times,
            npv_paths=mat,
            discount=discount,
            survival_cpty=survival_cpty,
            survival_own=survival_own,
            recovery_cpty=recovery_cpty,
            recovery_own=recovery_own,
            funding_spread=funding_spread,
        )
        ns_out[ns] = pack
        cva_sum += float(pack["cva"])
        dva_sum += float(pack["dva"])
        fva_sum += float(pack["fva"])

    portfolio_paths = np.asarray(agg["portfolio"], dtype=float)
    portfolio_pack = compute_xva_from_npv_paths(
        times=times,
        npv_paths=portfolio_paths,
        discount=discount,
        survival_cpty=survival_cpty,
        survival_own=survival_own,
        recovery_cpty=recovery_cpty,
        recovery_own=recovery_own,
        funding_spread=funding_spread,
    )
    return {
        "by_netting_set": ns_out,
        "portfolio": portfolio_pack,
        "sum_by_netting_set": {
            "cva": cva_sum,
            "dva": dva_sum,
            "fva": fva_sum,
            "xva_total": float(cva_sum - dva_sum + fva_sum),
        },
    }
