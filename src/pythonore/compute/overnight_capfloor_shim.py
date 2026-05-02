"""Local shim for overnight cap/floor pricing.

This module keeps the current Python approximation for ORE's overnight capped /
floored coupon handling in one place. It is intentionally not a native binding
to QuantExt's proxy-volatility classes.
"""
from __future__ import annotations

import math
from functools import lru_cache
from datetime import datetime
from typing import Any, Mapping

import numpy as np


def _coupon_include_spread(coupon: Any) -> bool:
    fn = getattr(coupon, "includeSpread", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


_CAPFLOOR_SURFACE_PERIOD_CACHE: dict[tuple[int, str], str | None] = {}
_QL_OVERNIGHT_INDEX_CACHE: dict[tuple[str, int], Any] = {}
_QL_CURVE_HANDLE_CACHE: dict[tuple[object, ...], Any] = {}
_QL_OVERNIGHT_RATE_CACHE: dict[tuple[object, ...], float] = {}
_QL_AVERAGE_OVERNIGHT_RATE_CACHE: dict[tuple[object, ...], float] = {}
_QL_OPTIONLET_VOL_CACHE: dict[tuple[object, ...], Any] = {}


def _past_fixing_or_none(ql_index: Any, fixing_date: Any) -> float | None:
    try:
        past_fixing = ql_index.pastFixing(fixing_date)
    except Exception:
        return None
    try:
        import QuantLib as ql

        null_real = getattr(ql, "NullReal", None)
        if callable(null_real):
            try:
                if past_fixing == null_real():
                    return None
            except Exception:
                pass
    except Exception:
        pass
    return float(past_fixing)


def _build_ql_overnight_index(overnight_index: str, overnight_handle: Any) -> Any:
    import QuantLib as ql

    cache_key = (overnight_index, id(overnight_handle))
    cached = _QL_OVERNIGHT_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if overnight_index == "SOFR":
        idx = ql.Sofr(overnight_handle)
    elif overnight_index == "FEDFUNDS":
        idx = ql.FedFunds(overnight_handle)
    elif overnight_index == "SONIA":
        idx = ql.Sonia(overnight_handle)
    elif overnight_index == "ESTR":
        idx = ql.Estr(overnight_handle)
    elif overnight_index == "SARON":
        idx = ql.Saron(overnight_handle)
    elif overnight_index == "TONAR":
        if hasattr(ql, "Tonar"):
            idx = ql.Tonar(overnight_handle)
        else:
            idx = ql.OvernightIndex(
                "TONAR",
                0,
                ql.JPYCurrency(),
                ql.Japan(),
                ql.Actual365Fixed(),
                overnight_handle,
            )
    else:
        raise AttributeError(f"unsupported overnight index '{overnight_index}'")
    _QL_OVERNIGHT_INDEX_CACHE[cache_key] = idx
    return idx


def _curve_handle_from_curve(
    eval_date: Any,
    curve: Any,
    *,
    extra_times: list[float] | None = None,
    dates: list[Any] | tuple[Any, ...] | None = None,
    dfs: list[float] | tuple[float, ...] | None = None,
) -> Any:
    import QuantLib as ql
    cache_key = (
        int(eval_date.serialNumber()) if hasattr(eval_date, "serialNumber") else str(eval_date),
        id(curve),
        tuple(extra_times) if extra_times else None,
        tuple(dates) if dates is not None else None,
        tuple(dfs) if dfs is not None else None,
    )
    cached = _QL_CURVE_HANDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if dates is not None and dfs is not None:
        ql_nodes: dict[int, tuple[Any, float]] = {}
        for d, df in zip(dates, dfs):
            try:
                if isinstance(d, str):
                    qd = ql.DateParser.parseISO(d)
                elif hasattr(d, "year") and hasattr(d, "month"):
                    day = int(getattr(d, "dayOfMonth", lambda: getattr(d, "day", 1))())
                    month = int(d.month())
                    year = int(d.year())
                    qd = ql.Date(day, month, year)
                else:
                    continue
                ql_nodes[int(qd.serialNumber())] = (qd, max(float(df), 1.0e-10))
            except Exception:
                continue
        ql_dates = []
        ql_dfs = []
        for _, (qd, df) in sorted(ql_nodes.items(), key=lambda item: item[0]):
            ql_dates.append(qd)
            ql_dfs.append(df)
        if ql_dates and ql_dfs:
            handle = ql.YieldTermStructureHandle(ql.DiscountCurve(ql_dates, ql_dfs, ql.Actual365Fixed()))
            _QL_CURVE_HANDLE_CACHE[cache_key] = handle
            return handle

    grid = {0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0}
    if extra_times:
        for t in extra_times:
            try:
                grid.add(max(float(t), 0.0))
            except Exception:
                continue
    grid = sorted(grid)
    ql_nodes = {int(eval_date.serialNumber()): (eval_date, 1.0)}
    for tt in grid[1:]:
        qd = eval_date + int(round(365.25 * tt))
        ql_nodes[int(qd.serialNumber())] = (qd, max(float(curve(tt)), 1.0e-10))
    ql_dates = []
    ql_dfs = []
    for _, (qd, df) in sorted(ql_nodes.items(), key=lambda item: item[0]):
        ql_dates.append(qd)
        ql_dfs.append(df)
    handle = ql.YieldTermStructureHandle(ql.DiscountCurve(ql_dates, ql_dfs, ql.Actual365Fixed()))
    _QL_CURVE_HANDLE_CACHE[cache_key] = handle
    return handle


def _overnight_coupon_rate_exact(coupon: Any, *, ql_index: Any, eval_date: Any) -> float:
    """Replicate QuantExt::OvernightIndexedCoupon::compute() as closely as possible."""
    import QuantLib as ql
    cache_key = (
        "overnight",
        int(eval_date.serialNumber()) if hasattr(eval_date, "serialNumber") else str(eval_date),
        id(coupon),
        id(ql_index),
    )
    cached = _QL_OVERNIGHT_RATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fixing_dates = list(coupon.fixingDates())
    value_dates = list(coupon.valueDates())
    dts = list(coupon.dt())
    if not fixing_dates or not value_dates or not dts:
        raise ValueError("empty overnight coupon schedule")

    n = len(dts)
    rate_cutoff = int(coupon.rateCutoff()) if hasattr(coupon, "rateCutoff") else 0
    if rate_cutoff >= n:
        raise ValueError("rate cutoff must be less than number of fixings in period")
    n_cutoff = n - rate_cutoff

    compound_factor = 1.0
    compound_factor_without_spread = 1.0
    i = 0
    today = eval_date

    while i < n and fixing_dates[min(i, n_cutoff)] < today:
        fixing_date = fixing_dates[min(i, n_cutoff)]
        past_fixing = _past_fixing_or_none(ql_index, fixing_date)
        if past_fixing is None:
            raise ValueError(f"missing fixing for {fixing_date}")
        if _coupon_include_spread(coupon):
            compound_factor_without_spread *= 1.0 + past_fixing * dts[i]
            past_fixing += coupon.spread()
        compound_factor *= 1.0 + past_fixing * dts[i]
        i += 1

    if i < n and fixing_dates[min(i, n_cutoff)] == today:
        try:
            past_fixing = _past_fixing_or_none(ql_index, today)
            if past_fixing is not None:
                if _coupon_include_spread(coupon):
                    compound_factor_without_spread *= 1.0 + past_fixing * dts[i]
                    past_fixing += coupon.spread()
                compound_factor *= 1.0 + past_fixing * dts[i]
                i += 1
        except Exception:
            pass

    if i < n:
        curve = ql_index.forwardingTermStructure()
        if curve is None:
            raise ValueError(f"null term structure set to this instance of {ql_index.name()}")

        start_discount = curve.discount(value_dates[i])
        end_discount = curve.discount(value_dates[max(n_cutoff, i)])
        if n_cutoff < n:
            discount_cutoff_date = curve.discount(value_dates[n_cutoff] + 1) / curve.discount(value_dates[n_cutoff])
            end_discount *= discount_cutoff_date ** (value_dates[n] - value_dates[n_cutoff])

        compound_factor *= start_discount / end_discount
        if _coupon_include_spread(coupon):
            compound_factor_without_spread *= start_discount / end_discount
            tau = ql_index.dayCounter().yearFraction(value_dates[i], value_dates[-1]) / (value_dates[-1] - value_dates[i])
            compound_factor *= (1.0 + tau * coupon.spread()) ** int(value_dates[-1] - value_dates[i])

    tau = ql_index.dayCounter().yearFraction(value_dates[0], value_dates[-1])
    if tau <= 0.0:
        raise ValueError("invalid overnight coupon accrual period")
    rate = (compound_factor - 1.0) / tau
    swaplet_rate = coupon.gearing() * rate
    if not _coupon_include_spread(coupon):
        swaplet_rate += coupon.spread()
    out = float(swaplet_rate)
    _QL_OVERNIGHT_RATE_CACHE[cache_key] = out
    return out


def _quant_ext_overnight_value_dates(
    *,
    ql_index: Any,
    eval_date: Any,
    start_date: Any,
    end_date: Any,
    lookback_days: int,
    rate_cutoff: int,
    telescopic_value_dates: bool = False,
) -> list[Any]:
    import QuantLib as ql

    calendar = ql_index.fixingCalendar()
    bdc = ql_index.businessDayConvention()
    value_start = start_date
    value_end = end_date
    if lookback_days != 0:
        lookback_bdc = ql.Preceding if lookback_days > 0 else ql.Following
        value_start = calendar.advance(value_start, -lookback_days, ql.Days, lookback_bdc)
        value_end = calendar.advance(value_end, -lookback_days, ql.Days, lookback_bdc)

    tmp_end_date = value_end
    if telescopic_value_dates:
        tmp_end_date = calendar.advance(max(value_start, eval_date), 7, ql.Days, ql.Following)
        tmp_end_date = min(tmp_end_date, value_end)
    if not value_start < tmp_end_date:
        raise ValueError("invalid overnight value-date period")

    value_dates = [value_start]
    next_date = calendar.advance(value_start, 1, ql.Days, ql.Following)
    while next_date < tmp_end_date:
        if next_date != value_dates[-1]:
            value_dates.append(next_date)
        next_date = calendar.advance(next_date, 1, ql.Days, ql.Following)
    if value_dates[-1] != tmp_end_date:
        value_dates.append(tmp_end_date)

    if value_dates[0] != value_start:
        value_dates.insert(0, value_start)

    if telescopic_value_dates:
        tmp2 = calendar.adjust(value_end, bdc)
        tmp1 = calendar.advance(tmp2, -max(int(rate_cutoff), 1), ql.Days, ql.Preceding)
        while tmp1 <= tmp2:
            if tmp1 > value_dates[-1]:
                value_dates.append(tmp1)
            tmp1 = calendar.advance(tmp1, 1, ql.Days, ql.Following)

    if len(value_dates) < 2 + int(rate_cutoff):
        raise ValueError("degenerate overnight schedule")
    value_dates[0] = value_start
    value_dates[-1] = value_end
    return value_dates


def _quant_ext_overnight_coupon_rate(
    *,
    ql_index: Any,
    eval_date: Any,
    start_date: Any,
    end_date: Any,
    spread: float,
    gearing: float,
    payment_day_counter: Any,
    lookback_days: int,
    rate_cutoff: int,
    fixing_days: int,
    include_spread: bool = False,
    telescopic_value_dates: bool = False,
) -> tuple[float, list[Any]]:
    import QuantLib as ql

    value_dates = _quant_ext_overnight_value_dates(
        ql_index=ql_index,
        eval_date=eval_date,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        rate_cutoff=rate_cutoff,
        telescopic_value_dates=telescopic_value_dates,
    )
    n = len(value_dates) - 1
    if int(rate_cutoff) >= n:
        raise ValueError("rate cutoff must be less than number of fixings in period")

    calendar = ql_index.fixingCalendar()
    fixing_dates = [
        calendar.advance(value_dates[i], -int(fixing_days), ql.Days, ql.Preceding)
        for i in range(n)
    ]
    dts = [ql_index.dayCounter().yearFraction(value_dates[i], value_dates[i + 1]) for i in range(n)]
    n_cutoff = n - int(rate_cutoff)
    compound_factor = 1.0
    compound_factor_without_spread = 1.0
    i = 0

    while i < n and fixing_dates[min(i, n_cutoff)] < eval_date:
        fixing_date = fixing_dates[min(i, n_cutoff)]
        past_fixing = _past_fixing_or_none(ql_index, fixing_date)
        if past_fixing is None:
            raise ValueError(f"missing fixing for {fixing_date}")
        if include_spread:
            compound_factor_without_spread *= 1.0 + past_fixing * dts[i]
            past_fixing += spread
        compound_factor *= 1.0 + past_fixing * dts[i]
        i += 1

    if i < n and fixing_dates[min(i, n_cutoff)] == eval_date:
        past_fixing = _past_fixing_or_none(ql_index, eval_date)
        if past_fixing is not None:
            if include_spread:
                compound_factor_without_spread *= 1.0 + past_fixing * dts[i]
                past_fixing += spread
            compound_factor *= 1.0 + past_fixing * dts[i]
            i += 1

    if i < n:
        curve = ql_index.forwardingTermStructure()
        if curve is None:
            raise ValueError(f"null term structure set to this instance of {ql_index.name()}")
        start_discount = curve.discount(value_dates[i])
        end_discount = curve.discount(value_dates[max(n_cutoff, i)])
        if n_cutoff < n:
            discount_cutoff_date = curve.discount(value_dates[n_cutoff] + 1) / curve.discount(value_dates[n_cutoff])
            end_discount *= discount_cutoff_date ** int(value_dates[n] - value_dates[n_cutoff])
        compound_factor *= start_discount / end_discount
        if include_spread:
            compound_factor_without_spread *= start_discount / end_discount
            tau_daily = ql_index.dayCounter().yearFraction(value_dates[i], value_dates[-1]) / int(
                value_dates[-1] - value_dates[i]
            )
            compound_factor *= (1.0 + tau_daily * spread) ** int(value_dates[-1] - value_dates[i])

    tau = payment_day_counter.yearFraction(value_dates[0], value_dates[-1])
    if tau <= 0.0:
        raise ValueError("invalid overnight coupon accrual period")
    rate = (compound_factor - 1.0) / tau
    swaplet_rate = float(gearing) * rate
    if not include_spread:
        swaplet_rate += float(spread)
    return float(swaplet_rate), fixing_dates


def _average_overnight_coupon_rate_exact(
    *,
    ql_index: Any,
    runtime: Any,
    snapshot: Any,
    leg: Mapping[str, object],
    t: float,
    coupon_index: int,
) -> float:
    """Replicate QuantExt::AverageONIndexedCouponPricer::swapletRate().

    The future part uses the Takada/log-discount forecast rather than a simple
    forward-rate approximation.
    """
    import QuantLib as ql
    cache_key = (
        "average",
        int(float(t) * 1_000_000.0),
        int(coupon_index),
        id(ql_index),
        id(runtime),
        id(snapshot),
    )
    cached = _QL_AVERAGE_OVERNIGHT_RATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    start_dates = np.asarray(leg.get("start_date", []), dtype=object)
    end_dates = np.asarray(leg.get("end_date", []), dtype=object)
    pay_dates = np.asarray(leg.get("pay_date", []), dtype=object)
    fixing_dates = np.asarray(leg.get("fixing_date", []), dtype=object)

    def _to_qdate(value: object) -> ql.Date:
        if isinstance(value, str):
            return ql.DateParser.parseISO(value)
        if hasattr(value, "year") and hasattr(value, "month"):
            day = int(getattr(value, "dayOfMonth", lambda: getattr(value, "day", 1))())
            month = int(value.month())
            year = int(value.year())
            return ql.Date(day, month, year)
        raise ValueError("invalid date value")

    if coupon_index < start_dates.size:
        start_date = _to_qdate(start_dates[coupon_index])
    else:
        start_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(np.asarray(leg.get("start_time", []), dtype=float)[coupon_index])))
    if coupon_index < end_dates.size:
        end_date = _to_qdate(end_dates[coupon_index])
    else:
        end_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(np.asarray(leg.get("end_time", []), dtype=float)[coupon_index])))
    if fixing_dates.size > coupon_index:
        fixing_date = _to_qdate(fixing_dates[coupon_index])
    else:
        fixing_time = float(np.asarray(leg.get("fixing_time", leg.get("start_time", [])), dtype=float)[coupon_index])
        fixing_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, fixing_time))
    if coupon_index < pay_dates.size:
        pay_date = _to_qdate(pay_dates[coupon_index])
    else:
        pay_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(np.asarray(leg.get("pay_time", leg.get("end_time", [])), dtype=float)[coupon_index])))
    _ = pay_date  # keep parity with the C++ coupon shape even though pay date is not used directly here

    fixing_calendar = ql_index.fixingCalendar()
    day_counter = ql_index.dayCounter()

    lookback_days = int(leg.get("lookback_days", 0) or 0)
    rate_cutoff = int(leg.get("rate_cutoff", 0) or 0)
    fixing_days = int(leg.get("fixing_days", 2) or 2)
    gearing = float(np.asarray(leg.get("gearing", [1.0]), dtype=float)[coupon_index])
    spread = float(np.asarray(leg.get("spread", [0.0]), dtype=float)[coupon_index])

    value_start = start_date
    value_end = end_date
    if lookback_days != 0:
        bdc = ql.Preceding if lookback_days > 0 else ql.Following
        value_start = fixing_calendar.advance(value_start, -lookback_days, ql.Days, bdc)
        value_end = fixing_calendar.advance(value_end, -lookback_days, ql.Days, bdc)

    try:
        schedule = ql.MakeSchedule().from_(value_start).to(value_end).withTenor(ql.Period(1, ql.Days)).withCalendar(
            fixing_calendar
        ).withConvention(fixing_calendar.businessDayConvention()).backwards()
        value_dates = list(schedule.dates())
    except Exception:
        value_dates = [value_start, value_end]
    if value_dates[0] != value_start:
        value_dates[0] = value_start
    if value_dates[-1] != value_end:
        value_dates[-1] = value_end
    n = len(value_dates) - 1
    if n <= 0:
        raise ValueError("degenerate average overnight schedule")
    n_cutoff = max(0, n - rate_cutoff)
    fixing_dates_schedule = [
        fixing_calendar.advance(value_dates[i], -fixing_days, ql.Days, ql.Preceding) for i in range(n)
    ]
    dts = [day_counter.yearFraction(value_dates[i], value_dates[i + 1]) for i in range(n)]
    today = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(t)))
    accumulated_rate = 0.0
    i = 0
    while i < n and fixing_dates_schedule[min(i, n_cutoff)] < today:
        past_fixing = _past_fixing_or_none(ql_index, fixing_dates_schedule[min(i, n_cutoff)])
        if past_fixing is None:
            raise ValueError(f"missing fixing for {fixing_dates_schedule[min(i, n_cutoff)]}")
        accumulated_rate += past_fixing * dts[i]
        i += 1
    if i < n and fixing_dates_schedule[min(i, n_cutoff)] == today:
        try:
            past_fixing = _past_fixing_or_none(ql_index, today)
            if past_fixing is not None:
                accumulated_rate += past_fixing * dts[i]
                i += 1
        except Exception:
            pass
    if i < n:
        curve = ql_index.forwardingTermStructure()
        if curve is None:
            raise ValueError(f"null term structure set to this instance of {ql_index.name()}")
        start_discount = curve.discount(value_dates[i])
        end_discount = curve.discount(value_dates[max(n_cutoff, i)])
        if n_cutoff < n:
            discount_cutoff_date = curve.discount(value_dates[n_cutoff] + 1) / curve.discount(value_dates[n_cutoff])
            end_discount *= discount_cutoff_date ** (value_dates[n] - value_dates[n_cutoff])
        accumulated_rate += math.log(start_discount / end_discount)
    tau = day_counter.yearFraction(value_dates[0], value_dates[-1])
    if tau <= 0.0:
        raise ValueError("invalid average overnight coupon accrual period")
    out = float(gearing * accumulated_rate / tau + spread)
    _QL_AVERAGE_OVERNIGHT_RATE_CACHE[cache_key] = out
    return out


def _parse_overnight_index_name(index_name: str) -> str | None:
    index_upper = str(index_name).strip().upper()
    if "SOFR" in index_upper:
        return "SOFR"
    if "FEDFUNDS" in index_upper:
        return "FEDFUNDS"
    if "SONIA" in index_upper:
        return "SONIA"
    if "EONIA" in index_upper or "ESTR" in index_upper or "ESTER" in index_upper:
        return "ESTR"
    if "SARON" in index_upper:
        return "SARON"
    if "TONAR" in index_upper:
        return "TONAR"
    return None


def _parse_ore_tenor_to_years(tenor: str) -> float:
    text = str(tenor).strip().upper()
    if not text:
        return 0.0
    unit = text[-1]
    value = float(text[:-1])
    if unit == "D":
        return value / 365.0
    if unit == "W":
        return 7.0 * value / 365.0
    if unit == "M":
        return value / 12.0
    if unit == "Y":
        return value
    return float(text)


def _ql_calendar(name: str):
    import QuantLib as ql

    text = str(name).strip().upper()
    if "US" in text:
        return ql.UnitedStates(ql.UnitedStates.Settlement)
    if "JP" in text or "JAPAN" in text:
        return ql.Japan()
    if "GB" in text or "UK" in text or "LONDON" in text:
        return ql.UnitedKingdom()
    if "CH" in text or "SWITZERLAND" in text:
        return ql.Switzerland()
    if "TARGET" in text or "EU" in text:
        return ql.TARGET()
    return ql.TARGET()


def _ql_business_day_convention(name: str):
    import QuantLib as ql

    text = str(name).strip().upper().replace(" ", "")
    if text in {"MF", "MODFOLLOWING", "MODIFIEDFOLLOWING"}:
        return ql.ModifiedFollowing
    if text in {"P", "PRECEDING"}:
        return ql.Preceding
    if text in {"U", "UNADJUSTED"}:
        return ql.Unadjusted
    return ql.Following


def _capfloor_curve_config(snapshot: Any, *, ccy: str) -> dict[str, str]:
    import xml.etree.ElementTree as ET

    out: dict[str, str] = {}
    xml_buffers = getattr(snapshot.config, "xml_buffers", {}) or {}
    xml = xml_buffers.get("curveconfig.xml")
    if not xml:
        return out
    try:
        root = ET.fromstring(xml)
    except Exception:
        return out
    ccy_upper = str(ccy).upper()
    for node in root.findall(".//CapFloorVolatility"):
        curve_id = (node.findtext("./CurveId") or "").strip().upper()
        index = (node.findtext("./Index") or "").strip().upper()
        if curve_id != ccy_upper and ccy_upper not in {curve_id[:3], index[:3]}:
            continue
        for key in (
            "Calendar",
            "BusinessDayConvention",
            "DayCounter",
            "InterpolationMethod",
            "InterpolateOn",
            "TimeInterpolation",
            "StrikeInterpolation",
            "InputType",
            "VolatilityType",
            "OutputVolatilityType",
            "RateComputationPeriod",
        ):
            value = (node.findtext(f"./{key}") or "").strip()
            if value:
                out[key] = value
        break
    return out


def _ql_optionlet_volatility(
    runtime: Any,
    inputs: Any,
    snapshot: Any,
    *,
    ccy: str,
) -> Any | None:
    try:
        import QuantLib as ql
    except Exception:
        return None

    if str(ccy).upper() != "USD":
        return None
    config = _capfloor_curve_config(snapshot, ccy=ccy)
    interpolate_on = str(config.get("InterpolateOn", "")).strip().upper()
    input_type = str(config.get("InputType", "")).strip().upper()
    vol_type = str(config.get("VolatilityType", "Normal")).strip().upper()
    if interpolate_on not in {"OPTIONLETVOLATILITIES", "OPTIONLET"} or input_type not in {"TERMVOLATILITIES", ""}:
        return None

    cache_key = (
        id(snapshot.market.raw_quotes),
        id(inputs),
        str(ccy).upper(),
        tuple(sorted(config.items())),
    )
    cached = _QL_OPTIONLET_VOL_CACHE.get(cache_key)
    if cached is not None or cache_key in _QL_OPTIONLET_VOL_CACHE:
        return cached

    points: dict[str, dict[float, float]] = {}
    ccy_upper = str(ccy).upper()
    for quote in snapshot.market.raw_quotes:
        raw_key = str(getattr(quote, "key", "")).strip().upper()
        if not raw_key.startswith(f"CAPFLOOR/RATE_NVOL/{ccy_upper}/"):
            continue
        parts = raw_key.split("/")
        if len(parts) < 8:
            continue
        expiry_txt = parts[3].strip().upper()
        strike_txt = parts[-1]
        try:
            points.setdefault(expiry_txt, {})[float(strike_txt)] = float(getattr(quote, "value", 0.0))
        except Exception:
            continue
    if not points:
        _QL_OPTIONLET_VOL_CACHE[cache_key] = None
        return None

    expiries = sorted(points, key=_parse_ore_tenor_to_years)
    strikes = sorted({strike for row in points.values() for strike in row})
    matrix = ql.Matrix(len(expiries), len(strikes))
    for i, expiry in enumerate(expiries):
        row = points[expiry]
        finite_values = [v for v in row.values() if math.isfinite(float(v))]
        fallback = float(finite_values[0]) if finite_values else 0.01
        for j, strike in enumerate(strikes):
            matrix[i][j] = float(row.get(strike, fallback))

    eval_date = ql.DateParser.parseISO(runtime._normalized_asof(snapshot))
    ql.Settings.instance().evaluationDate = eval_date
    periods = [ql.Period(expiry) for expiry in expiries]
    calendar = _ql_calendar(config.get("Calendar", ccy_upper))
    convention = _ql_business_day_convention(config.get("BusinessDayConvention", "Following"))
    surface = ql.CapFloorTermVolSurface(
        0,
        calendar,
        convention,
        periods,
        strikes,
        matrix,
        ql.Actual365Fixed(),
    )
    surface.enableExtrapolation()

    date_nodes = getattr(inputs, "discount_curve_dates", {}).get(ccy_upper)
    df_nodes = getattr(inputs, "discount_curve_dfs", {}).get(ccy_upper)
    discount_curve = runtime._resolve_index_curve(inputs, ccy_upper, f"{ccy_upper}-SOFR")
    discount_handle = _curve_handle_from_curve(
        eval_date,
        discount_curve,
        extra_times=list(getattr(inputs, "times", [])),
        dates=list(date_nodes) if date_nodes else None,
        dfs=list(df_nodes) if df_nodes else None,
    )
    period_txt = str(config.get("RateComputationPeriod", "3M") or "3M").strip() or "3M"
    try:
        index = ql.USDLibor(ql.Period(period_txt), discount_handle) if ccy_upper == "USD" else ql.IborIndex(
            f"{ccy_upper}-{period_txt}",
            ql.Period(period_txt),
            0,
            ql.USDCurrency() if ccy_upper == "USD" else ql.EURCurrency(),
            calendar,
            convention,
            False,
            ql.Actual360(),
            discount_handle,
        )
        stripped = ql.OptionletStripper1(
            surface,
            index,
            ql.nullDouble(),
            1.0e-12,
            100,
            discount_handle,
            ql.Normal if vol_type != "SHIFTEDLOGNORMAL" else ql.ShiftedLognormal,
            0.0,
            True,
        )
        first_vols = list(stripped.optionletVolatilities(0))
        first_strikes = list(stripped.optionletStrikes(0))
        raw_first = points[expiries[0]]
        raw_zero = float(raw_first.get(0.0, np.interp(0.0, sorted(raw_first), [raw_first[k] for k in sorted(raw_first)])))
        stripped_zero = float(np.interp(0.0, first_strikes, first_vols))
        # ORE's cap/floor curve is configured with InterpolateOn=OptionletVolatilities.
        # The local runtime still uses the raw term-vol lookup for expiry/strike
        # interpolation, but applies the short-end bootstrap uplift implied by
        # QuantLib's optionlet stripper. This closes the overnight floorlet scale
        # without routing the default path through ORE output anchors.
        adapter = stripped_zero / raw_zero if raw_zero > 0.0 and math.isfinite(stripped_zero) else None
    except Exception:
        adapter = None
    _QL_OPTIONLET_VOL_CACHE[cache_key] = adapter
    return adapter


class ProxyOptionletVolatilityReplica:
    """Small Python replica of QuantExt::ProxyOptionletVolatility.

    The object is intentionally lightweight: it only exposes the behavior the
    runtime needs, namely ATM adjustment and scaled normal-vol lookup.
    """

    def __init__(
        self,
        runtime: Any,
        inputs: Any,
        snapshot: Any,
        *,
        ccy: str,
        index_name: str,
        base_rate_computation_period: str,
        target_rate_computation_period: str,
        scaling_factor: float = 1.0,
        optionlet_volatility: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.inputs = inputs
        self.snapshot = snapshot
        self.ccy = ccy
        self.index_name = index_name
        self.base_rate_computation_period = str(base_rate_computation_period).strip()
        self.target_rate_computation_period = str(target_rate_computation_period).strip()
        self.scaling_factor = float(scaling_factor)
        self.optionlet_volatility = optionlet_volatility

    def atm_level(self, fixing_date: Any, rate_computation_period: str) -> float | None:
        return overnight_atm_level(
            self.runtime,
            self.inputs,
            ccy=self.ccy,
            index_name=self.index_name,
            fixing_date=fixing_date,
            rate_computation_period=rate_computation_period,
        )

    def adjusted_strike(self, fixing_date: Any, strike: float) -> float:
        if not self.base_rate_computation_period or not self.target_rate_computation_period:
            return float(strike)
        if self.base_rate_computation_period == self.target_rate_computation_period:
            return float(strike)
        base_atm = self.atm_level(fixing_date, self.base_rate_computation_period)
        target_atm = self.atm_level(fixing_date, self.target_rate_computation_period)
        if base_atm is None or target_atm is None:
            return float(strike)
        return float(strike + base_atm - target_atm)

    def volatility(self, *, expiry_time: float, strike: float, fixing_date: Any) -> float:
        adjusted_strike = self.adjusted_strike(fixing_date, strike)
        if isinstance(self.optionlet_volatility, (float, int)):
            return float(
                self.runtime._capfloor_normal_vol(
                    self.snapshot,
                    ccy=self.ccy,
                    expiry_time=expiry_time,
                    strike=adjusted_strike,
                )
                * float(self.optionlet_volatility)
                * self.scaling_factor
            )
        if self.optionlet_volatility is not None and fixing_date is not None:
            try:
                return float(self.optionlet_volatility.volatility(fixing_date, adjusted_strike) * self.scaling_factor)
            except Exception:
                pass
        return float(
            self.runtime._capfloor_normal_vol(
                self.snapshot,
                ccy=self.ccy,
                expiry_time=expiry_time,
                strike=adjusted_strike,
            )
            * self.scaling_factor
        )


class BlackOvernightIndexedCouponPricerReplica:
    """Small Python replica of QuantExt::BlackOvernightIndexedCouponPricer."""

    def __init__(
        self,
        runtime: Any,
        inputs: Any,
        snapshot: Any,
        *,
        ccy: str,
        index_name: str,
        base_rate_computation_period: str,
        target_rate_computation_period: str,
        scaling_factor: float = 1.0,
        effective_volatility_input: bool = False,
        optionlet_volatility: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.inputs = inputs
        self.snapshot = snapshot
        self.effective_volatility_input = bool(effective_volatility_input)
        self.proxy_vol = ProxyOptionletVolatilityReplica(
            runtime,
            inputs,
            snapshot,
            ccy=ccy,
            index_name=index_name,
            base_rate_computation_period=base_rate_computation_period,
            target_rate_computation_period=target_rate_computation_period,
            scaling_factor=scaling_factor,
            optionlet_volatility=optionlet_volatility,
        )

    def global_coupon_rate(
        self,
        *,
        raw_rate: np.ndarray,
        cap: object,
        floor: object,
        naked_option: bool,
        expiry_time: float,
        fixing_date: Any,
        fixing_dates: list[Any] | None = None,
        asof_date: datetime.date | None = None,
    ) -> np.ndarray:
        def _optional_float(value: object) -> float | None:
            if value is None:
                return None
            try:
                out = float(value)
            except Exception:
                return None
            return out if math.isfinite(out) else None

        cap_value = _optional_float(cap)
        floor_value = _optional_float(floor)
        coupon_rate = np.zeros_like(raw_rate, dtype=float) if naked_option else raw_rate.copy()
        if asof_date is None:
            asof_date = datetime.fromisoformat(self.runtime._normalized_asof(self.snapshot)).date()

        def _stddev(strike: float) -> np.ndarray:
            if self.effective_volatility_input or not fixing_dates:
                vol = self.proxy_vol.volatility(expiry_time=expiry_time, strike=strike, fixing_date=fixing_date)
                return np.full_like(raw_rate, vol * math.sqrt(max(expiry_time, 0.0)), dtype=float)

            fixing_start_py = datetime(
                fixing_dates[0].year(), fixing_dates[0].month(), fixing_dates[0].dayOfMonth()
            ).date()
            fixing_end_py = datetime(
                fixing_dates[-1].year(), fixing_dates[-1].month(), fixing_dates[-1].dayOfMonth()
            ).date()
            fixing_start_time = float(self.runtime._irs_utils._time_from_dates(asof_date, fixing_start_py, "A365F"))
            fixing_end_time = float(self.runtime._irs_utils._time_from_dates(asof_date, fixing_end_py, "A365F"))
            vol = self.proxy_vol.volatility(
                expiry_time=max(fixing_start_time, 1.0 / 365.0),
                strike=strike,
                fixing_date=fixing_dates[0],
            )
            T = max(fixing_start_time, 0.0)
            if abs(fixing_end_time - T) > 1.0e-12:
                denom = max((fixing_end_time - fixing_start_time) ** 2, 1.0e-18)
                T += ((fixing_end_time - T) ** 3) / denom / 3.0
            return np.full_like(raw_rate, vol * math.sqrt(max(T, 0.0)), dtype=float)

        if floor_value is not None:
            stddev = _stddev(floor_value)
            coupon_rate = coupon_rate + self.runtime._normal_option_rate(raw_rate, floor_value, stddev, is_call=False)
        if cap_value is not None:
            stddev = _stddev(cap_value)
            coupon_rate = coupon_rate - self.runtime._normal_option_rate(raw_rate, cap_value, stddev, is_call=True)
        return coupon_rate

    def local_coupon_rate(
        self,
        *,
        raw_rate: np.ndarray,
        cap: object,
        floor: object,
        naked_option: bool,
        option_stddev: np.ndarray | None,
    ) -> np.ndarray:
        return _apply_local_cap_floor(
            self.runtime,
            raw_rate,
            cap=cap,
            floor=floor,
            naked_option=naked_option,
            option_stddev=option_stddev,
        )


def capfloor_surface_rate_computation_period(snapshot: Any, *, ccy: str) -> str | None:
    cache_key = (id(snapshot.market.raw_quotes), str(ccy).upper())
    cached = _CAPFLOOR_SURFACE_PERIOD_CACHE.get(cache_key)
    if cached is not None or cache_key in _CAPFLOOR_SURFACE_PERIOD_CACHE:
        return cached
    if hasattr(snapshot, "config"):
        config = _capfloor_curve_config(snapshot, ccy=ccy)
        configured_period = str(config.get("RateComputationPeriod", "")).strip()
        if configured_period:
            _CAPFLOOR_SURFACE_PERIOD_CACHE[cache_key] = configured_period
            return configured_period
    for quote in snapshot.market.raw_quotes:
        raw_key = str(getattr(quote, "key", "")).strip().upper()
        if not raw_key.startswith(f"CAPFLOOR/RATE_NVOL/{ccy.upper()}/"):
            continue
        parts = raw_key.split("/")
        if len(parts) < 5:
            continue
        period = parts[4].strip()
        result = period or None
        _CAPFLOOR_SURFACE_PERIOD_CACHE[cache_key] = result
        return result
    _CAPFLOOR_SURFACE_PERIOD_CACHE[cache_key] = None
    return None


def overnight_atm_level(
    runtime: Any,
    inputs: Any,
    *,
    ccy: str,
    index_name: str,
    fixing_date: Any,
    rate_computation_period: str,
) -> float | None:
    try:
        import QuantLib as ql
    except Exception:
        return None
    overnight_index = _parse_overnight_index_name(index_name)
    if overnight_index is None:
        return None
    index_curve = runtime._resolve_index_curve(inputs, ccy, index_name)
    date_nodes = getattr(inputs, "discount_curve_dates", {}).get(ccy.upper())
    df_nodes = getattr(inputs, "discount_curve_dfs", {}).get(ccy.upper())
    overnight_handle = _curve_handle_from_curve(
        ql.Settings.instance().evaluationDate,
        index_curve,
        extra_times=list(getattr(inputs, "times", [])),
        dates=list(date_nodes) if date_nodes else None,
        dfs=list(df_nodes) if df_nodes else None,
    )
    ql_index = _build_ql_overnight_index(overnight_index, overnight_handle)
    today = ql.Settings.instance().evaluationDate
    try:
        start = ql_index.valueDate(fixing_date)
        end = ql_index.fixingCalendar().advance(start, ql.Period(rate_computation_period))
        adj_start = start if start > today else today
        adj_end = end if end > adj_start + 1 else adj_start + 1
        coupon = ql.OvernightIndexedCoupon(adj_end, 1.0, adj_start, adj_end, ql_index, 1.0, 0.0, adj_start, adj_end, ql.Actual360(), False, ql.RateAveraging.Compound, 0, 0, False)
        return _overnight_coupon_rate_exact(coupon, ql_index=ql_index, eval_date=today)
    except Exception:
        return None


def _option_stddev(runtime: Any, snapshot: Any, *, ccy: str, expiry_time: float, strike: float) -> float:
    vol = runtime._capfloor_normal_vol(snapshot, ccy=ccy, expiry_time=expiry_time, strike=strike)
    return float(vol * math.sqrt(max(expiry_time, 0.0)))


def _maybe_adjust_strike(
    proxy: ProxyOptionletVolatilityReplica,
    *,
    fixing_dates: list[Any],
    strike: float,
) -> float:
    return proxy.adjusted_strike(fixing_dates[-1], strike)


def _overnight_static_cache_key(
    runtime: Any,
    snapshot: Any,
    *,
    ccy: str,
    index_name: str,
    surface_period: str,
    target_period: str,
    leg: Mapping[str, object],
) -> tuple[Any, ...]:
    start = np.asarray(leg.get("start_time", []), dtype=float)
    end = np.asarray(leg.get("end_time", []), dtype=float)
    pay = np.asarray(leg.get("pay_time", []), dtype=float)
    return (
        id(snapshot),
        id(leg),
        str(ccy).upper(),
        str(index_name).upper(),
        str(surface_period).strip(),
        str(target_period).strip(),
        int(leg.get("lookback_days", 0) or 0),
        int(leg.get("rate_cutoff", 0) or 0),
        int(leg.get("fixing_days", 0) or 0),
        bool(leg.get("apply_observation_shift", False)),
        bool(leg.get("naked_option", False)),
        bool(leg.get("local_cap_floor", False)),
        start.size,
        end.size,
        pay.size,
    )


def _build_overnight_static_state(
    runtime: Any,
    inputs: Any,
    snapshot: Any,
    *,
    ccy: str,
    index_name: str,
    surface_period: str,
    target_period: str,
    leg: Mapping[str, object],
) -> dict[str, Any] | None:
    try:
        import QuantLib as ql
    except Exception:
        return None

    eval_date = ql.DateParser.parseISO(runtime._normalized_asof(snapshot))
    ql.Settings.instance().evaluationDate = eval_date
    overnight_index = _parse_overnight_index_name(index_name)
    if overnight_index is None:
        return None
    index_curve = runtime._resolve_index_curve(inputs, ccy, index_name)
    overnight_handle = _curve_handle_from_curve(
        eval_date,
        index_curve,
        extra_times=list(getattr(inputs, "times", [])),
    )
    ql_index = _build_ql_overnight_index(overnight_index, overnight_handle)
    optionlet_volatility = _ql_optionlet_volatility(
        runtime,
        inputs,
        snapshot,
        ccy=ccy,
    )
    proxy = ProxyOptionletVolatilityReplica(
        runtime,
        inputs,
        snapshot,
        ccy=ccy,
        index_name=index_name,
        base_rate_computation_period=surface_period,
        target_rate_computation_period=target_period,
        scaling_factor=1.0,
        optionlet_volatility=optionlet_volatility,
    )
    pricer = BlackOvernightIndexedCouponPricerReplica(
        runtime,
        inputs,
        snapshot,
        ccy=ccy,
        index_name=index_name,
        base_rate_computation_period=surface_period,
        target_rate_computation_period=target_period,
        scaling_factor=1.0,
        effective_volatility_input=True,
        optionlet_volatility=optionlet_volatility,
    )

    start = np.asarray(leg.get("start_time", []), dtype=float)
    end = np.asarray(leg.get("end_time", []), dtype=float)
    pay_times = np.asarray(leg.get("pay_time", end), dtype=float)
    curve_times = set(float(t) for t in getattr(inputs, "times", []) if np.isfinite(float(t)))
    for arr in (start, end, pay_times, np.asarray(leg.get("fixing_time", []), dtype=float)):
        for t_value in arr:
            if np.isfinite(float(t_value)):
                curve_times.add(max(float(t_value), 0.0))
    curve_times.add(0.0)
    curve_nodes: dict[str, float] = {}
    for t_value in sorted(curve_times):
        try:
            curve_nodes[runtime._date_from_time_cached(snapshot, float(t_value))] = float(index_curve(float(t_value)))
        except Exception:
            continue
    curve_dates = sorted(curve_nodes)
    curve_dfs = [curve_nodes[d] for d in curve_dates]
    if curve_dates and curve_dfs:
        overnight_handle = _curve_handle_from_curve(
            eval_date,
            index_curve,
            dates=curve_dates,
            dfs=curve_dfs,
        )
        ql_index = _build_ql_overnight_index(overnight_index, overnight_handle)
    fixings = runtime._fixings_lookup(snapshot)
    for (fixing_index, fixing_date), fixing_value in fixings.items():
        if str(fixing_index).upper() != str(index_name).upper():
            continue
        try:
            ql_fixing_date = ql.DateParser.parseISO(str(fixing_date))
            if ql_fixing_date <= eval_date:
                ql_index.addFixing(ql_fixing_date, float(fixing_value), True)
        except Exception:
            continue
    lookback_days = int(leg.get("lookback_days", 0) or 0)
    lockout_days = int(leg.get("rate_cutoff", 0) or 0)
    fixing_days = int(leg.get("fixing_days", 0) or 0)
    is_averaged = bool(leg.get("is_averaged", False))
    spreads = np.asarray(leg.get("spread", np.zeros(start.shape)), dtype=float)
    gearings = np.asarray(leg.get("gearing", np.ones(start.shape)), dtype=float)
    raw_rates = np.zeros(start.size, dtype=float)
    fixing_dates: list[Any] = []
    expiry_times = np.zeros(start.size, dtype=float)
    asof_date = datetime.fromisoformat(runtime._normalized_asof(snapshot)).date()
    for i in range(start.size):
        if is_averaged:
            if "fixing_date" in leg and i < len(np.asarray(leg.get("fixing_date", []), dtype=object)):
                fixing_py = datetime.fromisoformat(str(np.asarray(leg.get("fixing_date", []), dtype=object)[i])).date()
            else:
                fixing_py = datetime.fromisoformat(runtime._date_from_time_cached(snapshot, float(start[i]))).date()
            fixing_dates.append(ql.Date(fixing_py.day, fixing_py.month, fixing_py.year))
            expiry_times[i] = float(runtime._irs_utils._time_from_dates(asof_date, fixing_py, "A365F"))
            raw_rates[i] = 0.0
            continue
        start_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(start[i])))
        end_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(end[i])))
        raw_rate, coupon_fixings = _quant_ext_overnight_coupon_rate(
            ql_index=ql_index,
            eval_date=eval_date,
            start_date=start_date,
            end_date=end_date,
            spread=float(spreads[i]) if i < spreads.size else 0.0,
            gearing=float(gearings[i]) if i < gearings.size else 1.0,
            payment_day_counter=ql.Actual360(),
            lookback_days=lookback_days,
            rate_cutoff=lockout_days,
            fixing_days=fixing_days,
        )
        if not coupon_fixings:
            continue
        fixing_date = coupon_fixings[len(coupon_fixings) - 1 - lockout_days]
        fixing_dates.append(fixing_date)
        fixing_py = datetime(fixing_date.year(), fixing_date.month(), fixing_date.dayOfMonth()).date()
        expiry_times[i] = float(runtime._irs_utils._time_from_dates(asof_date, fixing_py, "A365F"))
        raw_rates[i] = raw_rate
    return {
        "ql_index": ql_index,
        "pricer": pricer,
        "proxy": proxy,
        "raw_rates": raw_rates,
        "fixing_dates": fixing_dates,
        "expiry_times": expiry_times,
        "asof_date": asof_date,
    }


def price_average_overnight_coupon_paths(
    runtime: Any,
    *,
    model: Any,
    inputs: Any,
    leg: Mapping[str, object],
    ccy: str,
    t: float,
    x_t: np.ndarray,
    snapshot: Any,
) -> np.ndarray:
    """Price an averaged overnight leg using the Takada/log-DF forecast."""
    x_arr = np.asarray(x_t, dtype=float)
    start = np.asarray(leg.get("start_time", []), dtype=float)
    coupons = np.zeros((start.size, x_arr.size), dtype=float)
    if not bool(leg.get("overnight_indexed", False)) or not bool(leg.get("is_averaged", False)):
        return coupons

    index_name = str(leg.get("index_name", "")).strip().upper()
    surface_period = capfloor_surface_rate_computation_period(snapshot, ccy=ccy) or ""
    target_period = str(leg.get("schedule_tenor", "")).strip() or "3M"
    cache = getattr(runtime, "_overnight_capfloor_static_cache", None)
    if cache is None:
        cache = {}
        setattr(runtime, "_overnight_capfloor_static_cache", cache)
    cache_key = _overnight_static_cache_key(
        runtime,
        snapshot,
        ccy=ccy,
        index_name=index_name,
        surface_period=surface_period,
        target_period=target_period,
        leg=leg,
    )
    static_state = cache.get(cache_key)
    if static_state is None:
        static_state = _build_overnight_static_state(
            runtime,
            inputs,
            snapshot,
            ccy=ccy,
            index_name=index_name,
            surface_period=surface_period,
            target_period=target_period,
            leg=leg,
        )
        cache[cache_key] = static_state
    if static_state is None:
        return coupons

    for i in range(start.size):
        try:
            coupons[i, :] = _average_overnight_coupon_rate_exact(
                ql_index=static_state["ql_index"],
                runtime=runtime,
                snapshot=snapshot,
                leg=leg,
                t=t,
                coupon_index=i,
            )
        except Exception:
            coupons[i, :] = 0.0
    if leg.get("cap") is not None or leg.get("floor") is not None:
        naked_option = bool(leg.get("naked_option", False))
        for i in range(start.size):
            coupons[i, :] = _apply_local_cap_floor(
                runtime,
                coupons[i, :],
                cap=leg.get("cap"),
                floor=leg.get("floor"),
                naked_option=naked_option,
                option_stddev=np.zeros_like(x_arr, dtype=float),
            )
    return coupons


def _apply_local_cap_floor(
    runtime: Any,
    raw_rate: np.ndarray,
    *,
    cap: object,
    floor: object,
    naked_option: bool,
    option_stddev: np.ndarray | None,
) -> np.ndarray:
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    cap_value = _optional_float(cap)
    floor_value = _optional_float(floor)
    coupon_rate = raw_rate.copy() if not naked_option else np.zeros_like(raw_rate, dtype=float)
    stddev = option_stddev if option_stddev is not None else np.zeros_like(raw_rate, dtype=float)
    if floor_value is not None:
        coupon_rate = np.maximum(coupon_rate, floor_value) if not naked_option else runtime._normal_option_rate(
            raw_rate,
            floor_value,
            stddev,
            is_call=False,
        )
    if cap_value is not None:
        coupon_rate = np.minimum(coupon_rate, cap_value) if not naked_option else coupon_rate - runtime._normal_option_rate(
            raw_rate,
            cap_value,
            stddev,
            is_call=True,
        )
    return coupon_rate


def price_overnight_capfloor_coupon_paths(
    runtime: Any,
    *,
    inputs: Any,
    leg: Mapping[str, object],
    ccy: str,
    t: float,
    x_t: np.ndarray,
    snapshot: Any,
) -> np.ndarray:
    """Price the current Python approximation for an overnight cap/floor leg."""

    x_arr = np.asarray(x_t, dtype=float)
    start = np.asarray(leg.get("start_time", []), dtype=float)
    end = np.asarray(leg.get("end_time", []), dtype=float)
    coupons = np.zeros((start.size, x_arr.size), dtype=float)

    if not bool(leg.get("overnight_indexed", False)):
        return coupons

    index_name = str(leg.get("index_name", "")).strip().upper()
    surface_period = capfloor_surface_rate_computation_period(snapshot, ccy=ccy) or ""
    target_period = str(leg.get("schedule_tenor", "")).strip() or "3M"
    cache = getattr(runtime, "_overnight_capfloor_static_cache", None)
    if cache is None:
        cache = {}
        setattr(runtime, "_overnight_capfloor_static_cache", cache)
    cache_key = _overnight_static_cache_key(
        runtime,
        snapshot,
        ccy=ccy,
        index_name=index_name,
        surface_period=surface_period,
        target_period=target_period,
        leg=leg,
    )
    static_state = cache.get(cache_key)
    if static_state is None:
        static_state = _build_overnight_static_state(
            runtime,
            inputs,
            snapshot,
            ccy=ccy,
            index_name=index_name,
            surface_period=surface_period,
            target_period=target_period,
            leg=leg,
        )
        cache[cache_key] = static_state
    if static_state is None:
        return coupons

    pricer = static_state["pricer"]
    proxy = static_state["proxy"]
    raw_rates = np.asarray(static_state["raw_rates"], dtype=float)
    fixing_dates = list(static_state["fixing_dates"])
    expiry_times = np.asarray(static_state["expiry_times"], dtype=float)
    asof_date = static_state["asof_date"]
    naked_option = bool(leg.get("naked_option", False))
    local_cap_floor = bool(leg.get("local_cap_floor", False))

    for i in range(start.size):
        raw_rate = np.full_like(x_arr, float(raw_rates[i]), dtype=float)
        floor = leg.get("floor")
        cap = leg.get("cap")
        if local_cap_floor:
            if naked_option:
                coupons[i, :] = pricer.global_coupon_rate(
                    raw_rate=raw_rate,
                    cap=cap,
                    floor=floor,
                    naked_option=True,
                    expiry_time=float(expiry_times[i]),
                    fixing_date=fixing_dates[i] if i < len(fixing_dates) else None,
                    fixing_dates=fixing_dates,
                    asof_date=asof_date,
                )
                continue
            coupons[i, :] = pricer.local_coupon_rate(
                raw_rate=raw_rate,
                cap=cap,
                floor=floor,
                naked_option=naked_option,
                option_stddev=np.full_like(x_arr, 0.0, dtype=float),
            )
            continue

        coupons[i, :] = pricer.global_coupon_rate(
            raw_rate=raw_rate,
            cap=cap,
            floor=floor,
            naked_option=naked_option,
            expiry_time=float(expiry_times[i]),
            fixing_date=fixing_dates[i] if i < len(fixing_dates) else None,
            fixing_dates=fixing_dates,
            asof_date=asof_date,
        )
    return coupons
