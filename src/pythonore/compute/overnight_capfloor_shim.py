"""Local shim for overnight cap/floor pricing.

This module keeps the current Python approximation for ORE's overnight capped /
floored coupon handling in one place. It is intentionally not a native binding
to QuantExt's proxy-volatility classes.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

import numpy as np


def _build_ql_overnight_index(overnight_index: str, overnight_handle: Any) -> Any:
    import QuantLib as ql

    if overnight_index == "SOFR":
        return ql.Sofr(overnight_handle)
    if overnight_index == "FEDFUNDS":
        return ql.FedFunds(overnight_handle)
    if overnight_index == "SONIA":
        return ql.Sonia(overnight_handle)
    if overnight_index == "ESTR":
        return ql.Estr(overnight_handle)
    if overnight_index == "SARON":
        return ql.Saron(overnight_handle)
    return ql.Tonar(overnight_handle)


def _curve_handle_from_curve(eval_date: Any, curve: Any) -> Any:
    import QuantLib as ql

    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    dates = [eval_date]
    dfs = [1.0]
    for tt in grid[1:]:
        dates.append(eval_date + int(round(365.25 * tt)))
        dfs.append(max(float(curve(tt)), 1.0e-10))
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))


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


def capfloor_surface_rate_computation_period(snapshot: Any, *, ccy: str) -> str | None:
    for quote in snapshot.market.raw_quotes:
        raw_key = str(getattr(quote, "key", "")).strip().upper()
        if not raw_key.startswith(f"CAPFLOOR/RATE_NVOL/{ccy.upper()}/"):
            continue
        parts = raw_key.split("/")
        if len(parts) < 5:
            continue
        period = parts[4].strip()
        return period or None
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
    overnight_handle = _curve_handle_from_curve(ql.Settings.instance().evaluationDate, index_curve)
    ql_index = _build_ql_overnight_index(overnight_index, overnight_handle)
    today = ql.Settings.instance().evaluationDate
    try:
        start = ql_index.valueDate(fixing_date)
        end = ql_index.fixingCalendar().advance(start, ql.Period(rate_computation_period))
        adj_start = start if start > today else today
        adj_end = end if end > adj_start + 1 else adj_start + 1
        coupon = ql.OvernightIndexedCoupon(
            adj_end,
            1.0,
            adj_start,
            adj_end,
            ql_index,
        )
        coupon.setPricer(ql.OvernightIndexedCouponPricer())
        return float(coupon.rate())
    except Exception:
        return None


def _option_stddev(runtime: Any, snapshot: Any, *, ccy: str, expiry_time: float, strike: float) -> float:
    vol = runtime._capfloor_normal_vol(snapshot, ccy=ccy, expiry_time=expiry_time, strike=strike)
    return float(vol * math.sqrt(max(expiry_time, 0.0)))


def _maybe_adjust_strike(
    runtime: Any,
    inputs: Any,
    snapshot: Any,
    *,
    ccy: str,
    index_name: str,
    fixing_dates: list[Any],
    strike: float,
    target_period: str,
) -> float:
    surface_period = capfloor_surface_rate_computation_period(snapshot, ccy=ccy)
    if not surface_period or surface_period == target_period:
        return strike
    base_atm = overnight_atm_level(
        runtime,
        inputs,
        ccy=ccy,
        index_name=index_name,
        fixing_date=fixing_dates[-1],
        rate_computation_period=surface_period,
    )
    target_atm = overnight_atm_level(
        runtime,
        inputs,
        ccy=ccy,
        index_name=index_name,
        fixing_date=fixing_dates[-1],
        rate_computation_period=target_period,
    )
    if base_atm is None or target_atm is None:
        return strike
    return strike + base_atm - target_atm


def _apply_local_cap_floor(
    runtime: Any,
    raw_rate: np.ndarray,
    *,
    cap: object,
    floor: object,
    naked_option: bool,
    option_stddev: np.ndarray | None,
) -> np.ndarray:
    coupon_rate = raw_rate.copy() if not naked_option else np.zeros_like(raw_rate, dtype=float)
    stddev = option_stddev if option_stddev is not None else np.zeros_like(raw_rate, dtype=float)
    if floor is not None:
        coupon_rate = np.maximum(coupon_rate, float(floor)) if not naked_option else runtime._normal_option_rate(
            raw_rate,
            float(floor),
            stddev,
            is_call=False,
        )
    if cap is not None:
        coupon_rate = np.minimum(coupon_rate, float(cap)) if not naked_option else coupon_rate - runtime._normal_option_rate(
            raw_rate,
            float(cap),
            stddev,
            is_call=True,
        )
    return coupon_rate


def _apply_global_cap_floor(
    runtime: Any,
    snapshot: Any,
    raw_rate: np.ndarray,
    *,
    ccy: str,
    expiry_time: float,
    cap: object,
    floor: object,
    naked_option: bool,
) -> np.ndarray:
    coupon_rate = np.zeros_like(raw_rate, dtype=float) if naked_option else raw_rate.copy()
    if floor is not None:
        vol = runtime._capfloor_normal_vol(snapshot, ccy=ccy, expiry_time=expiry_time, strike=float(floor))
        stddev = np.full_like(raw_rate, vol * math.sqrt(max(expiry_time, 0.0)), dtype=float)
        coupon_rate = coupon_rate + runtime._normal_option_rate(raw_rate, float(floor), stddev, is_call=False)
    if cap is not None:
        vol = runtime._capfloor_normal_vol(snapshot, ccy=ccy, expiry_time=expiry_time, strike=float(cap))
        stddev = np.full_like(raw_rate, vol * math.sqrt(max(expiry_time, 0.0)), dtype=float)
        coupon_rate = coupon_rate - runtime._normal_option_rate(raw_rate, float(cap), stddev, is_call=True)
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

    try:
        import QuantLib as ql
    except Exception:
        return np.zeros((0, np.asarray(x_t, dtype=float).size), dtype=float)

    x_arr = np.asarray(x_t, dtype=float)
    start = np.asarray(leg.get("start_time", []), dtype=float)
    end = np.asarray(leg.get("end_time", []), dtype=float)
    fixing = np.asarray(leg.get("fixing_time", start), dtype=float)
    coupons = np.zeros((start.size, x_arr.size), dtype=float)

    if not bool(leg.get("overnight_indexed", False)):
        return coupons

    index_name = str(leg.get("index_name", "")).strip().upper()
    overnight_index = _parse_overnight_index_name(index_name)
    if overnight_index is None:
        return coupons

    eval_date = ql.DateParser.parseISO(runtime._normalized_asof(snapshot))
    ql.Settings.instance().evaluationDate = eval_date
    index_curve = runtime._resolve_index_curve(inputs, ccy, index_name)
    overnight_handle = _curve_handle_from_curve(eval_date, index_curve)
    ql_index = _build_ql_overnight_index(overnight_index, overnight_handle)

    lookback_days = int(leg.get("lookback_days", 0) or 0)
    lockout_days = int(leg.get("rate_cutoff", 0) or 0)
    apply_observation_shift = bool(leg.get("apply_observation_shift", False))
    naked_option = bool(leg.get("naked_option", False))
    local_cap_floor = bool(leg.get("local_cap_floor", False))
    asof_date = datetime.fromisoformat(runtime._normalized_asof(snapshot)).date()

    for i in range(start.size):
        start_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(start[i])))
        end_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(end[i])))
        pay_times = np.asarray(leg.get("pay_time", end), dtype=float)
        pay_date = ql.DateParser.parseISO(runtime._date_from_time_cached(snapshot, float(pay_times[i])))
        coupon = ql.OvernightIndexedCoupon(
            pay_date,
            1.0,
            start_date,
            end_date,
            ql_index,
            1.0,
            0.0,
            start_date,
            end_date,
            ql.Actual360(),
            False,
            ql.RateAveraging.Compound,
            lookback_days,
            lockout_days,
            apply_observation_shift,
        )
        fixing_dates = list(coupon.fixingDates())
        last_fixing_date = fixing_dates[-1]
        fixing_py = datetime(last_fixing_date.year(), last_fixing_date.month(), last_fixing_date.dayOfMonth()).date()
        expiry_time = float(runtime._irs_utils._time_from_dates(asof_date, fixing_py, "A365F"))
        raw_rate = np.full_like(x_arr, float(coupon.rate()), dtype=float)
        floor = leg.get("floor")
        cap = leg.get("cap")
        option_stddev = None
        if floor is not None or cap is not None:
            strike = _maybe_adjust_strike(
                runtime,
                inputs,
                snapshot,
                ccy=ccy,
                index_name=index_name,
                fixing_dates=fixing_dates,
                strike=float(floor if floor is not None else cap),
                target_period=str(leg.get("schedule_tenor", "")).strip() or "3M",
            )
            fixing_start_py = datetime(
                fixing_dates[0].year(), fixing_dates[0].month(), fixing_dates[0].dayOfMonth()
            ).date()
            fixing_end_py = datetime(
                fixing_dates[-1].year(), fixing_dates[-1].month(), fixing_dates[-1].dayOfMonth()
            ).date()
            fixing_start_time = float(runtime._irs_utils._time_from_dates(asof_date, fixing_start_py, "A365F"))
            fixing_end_time = float(runtime._irs_utils._time_from_dates(asof_date, fixing_end_py, "A365F"))
            T = max(fixing_start_time, 0.0)
            if abs(fixing_end_time - T) > 1.0e-12:
                denom = max((fixing_end_time - fixing_start_time) ** 2, 1.0e-18)
                T += ((fixing_end_time - T) ** 3) / denom / 3.0
            option_stddev = np.full_like(
                raw_rate,
                _option_stddev(runtime, snapshot, ccy=ccy, expiry_time=max(fixing_start_time, 1.0 / 365.0), strike=strike)
                * math.sqrt(max(T / max(fixing_start_time, 1.0 / 365.0), 0.0)),
                dtype=float,
            )

        if local_cap_floor:
            coupons[i, :] = _apply_local_cap_floor(
                runtime,
                raw_rate,
                cap=cap,
                floor=floor,
                naked_option=naked_option,
                option_stddev=option_stddev,
            )
            continue

        coupons[i, :] = _apply_global_cap_floor(
            runtime,
            snapshot,
            raw_rate,
            ccy=ccy,
            expiry_time=expiry_time,
            cap=cap,
            floor=floor,
            naked_option=naked_option,
        )
    return coupons
