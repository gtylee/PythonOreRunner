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


def _coupon_include_spread(coupon: Any) -> bool:
    fn = getattr(coupon, "includeSpread", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


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
    if overnight_index == "TONAR":
        if hasattr(ql, "Tonar"):
            return ql.Tonar(overnight_handle)
        return ql.OvernightIndex(
            "TONAR",
            0,
            ql.JPYCurrency(),
            ql.Japan(),
            ql.Actual365Fixed(),
            overnight_handle,
        )
    raise AttributeError(f"unsupported overnight index '{overnight_index}'")


def _curve_handle_from_curve(
    eval_date: Any,
    curve: Any,
    *,
    extra_times: list[float] | None = None,
    dates: list[Any] | tuple[Any, ...] | None = None,
    dfs: list[float] | tuple[float, ...] | None = None,
) -> Any:
    import QuantLib as ql

    if dates is not None and dfs is not None:
        ql_dates = []
        ql_dfs = []
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
                ql_dates.append(qd)
                ql_dfs.append(max(float(df), 1.0e-10))
            except Exception:
                continue
        if ql_dates and ql_dfs:
            return ql.YieldTermStructureHandle(ql.DiscountCurve(ql_dates, ql_dfs, ql.Actual365Fixed()))

    grid = {0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0}
    if extra_times:
        for t in extra_times:
            try:
                grid.add(max(float(t), 0.0))
            except Exception:
                continue
    grid = sorted(grid)
    ql_dates = [eval_date]
    ql_dfs = [1.0]
    for tt in grid[1:]:
        ql_dates.append(eval_date + int(round(365.25 * tt)))
        ql_dfs.append(max(float(curve(tt)), 1.0e-10))
    return ql.YieldTermStructureHandle(ql.DiscountCurve(ql_dates, ql_dfs, ql.Actual365Fixed()))


def _overnight_coupon_rate_exact(coupon: Any, *, ql_index: Any, eval_date: Any) -> float:
    """Replicate QuantExt::OvernightIndexedCoupon::compute() as closely as possible."""
    import QuantLib as ql

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
    return float(swaplet_rate)


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
    return float(gearing * accumulated_rate / tau + spread)


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
    ) -> None:
        self.runtime = runtime
        self.inputs = inputs
        self.snapshot = snapshot
        self.ccy = ccy
        self.index_name = index_name
        self.base_rate_computation_period = str(base_rate_computation_period).strip()
        self.target_rate_computation_period = str(target_rate_computation_period).strip()
        self.scaling_factor = float(scaling_factor)

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

        if floor is not None:
            stddev = _stddev(float(floor))
            coupon_rate = coupon_rate + self.runtime._normal_option_rate(raw_rate, float(floor), stddev, is_call=False)
        if cap is not None:
            stddev = _stddev(float(cap))
            coupon_rate = coupon_rate - self.runtime._normal_option_rate(raw_rate, float(cap), stddev, is_call=True)
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
    date_nodes = getattr(inputs, "discount_curve_dates", {}).get(ccy.upper())
    df_nodes = getattr(inputs, "discount_curve_dfs", {}).get(ccy.upper())
    overnight_handle = _curve_handle_from_curve(
        eval_date,
        index_curve,
        extra_times=list(getattr(inputs, "times", [])),
        dates=list(date_nodes) if date_nodes else None,
        dfs=list(df_nodes) if df_nodes else None,
    )
    ql_index = _build_ql_overnight_index(overnight_index, overnight_handle)
    proxy = ProxyOptionletVolatilityReplica(
        runtime,
        inputs,
        snapshot,
        ccy=ccy,
        index_name=index_name,
        base_rate_computation_period=surface_period,
        target_rate_computation_period=target_period,
        scaling_factor=1.0,
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
    )

    start = np.asarray(leg.get("start_time", []), dtype=float)
    end = np.asarray(leg.get("end_time", []), dtype=float)
    pay_times = np.asarray(leg.get("pay_time", end), dtype=float)
    lookback_days = int(leg.get("lookback_days", 0) or 0)
    lockout_days = int(leg.get("rate_cutoff", 0) or 0)
    apply_observation_shift = bool(leg.get("apply_observation_shift", False))
    is_averaged = bool(leg.get("is_averaged", False))
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
        coupon_fixings = list(coupon.fixingDates())
        if not coupon_fixings:
            continue
        fixing_dates.append(coupon_fixings[-1])
        fixing_py = datetime(coupon_fixings[-1].year(), coupon_fixings[-1].month(), coupon_fixings[-1].dayOfMonth()).date()
        expiry_times[i] = float(runtime._irs_utils._time_from_dates(asof_date, fixing_py, "A365F"))
        raw_rates[i] = _overnight_coupon_rate_exact(coupon, ql_index=ql_index, eval_date=eval_date)
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
