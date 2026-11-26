from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
import math
import re
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np


_TENOR_RE = re.compile(r"^(\d+)([DWMY])$", re.IGNORECASE)


@lru_cache(maxsize=32)
def _curve_config(xml: str, index_name: str) -> dict[str, str] | None:
    if not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None

    target = str(index_name).strip().upper()
    for node in root.findall("./CapFloorVolatilities/CapFloorVolatility"):
        curve_id = (node.findtext("./CurveId") or "").strip().upper()
        proxy = node.find("./ProxyConfig")
        configured_index = (node.findtext("./Index") or "").strip().upper()
        if curve_id != target and configured_index != target:
            continue

        source = proxy.find("./Source") if proxy is not None else None
        source_curve = (source.findtext("./CurveId") or "").strip() if source is not None else ""
        surface_node = node
        if source_curve:
            surface_curve_id = source_curve.split("/")[-1].strip().upper()
            surface_node = next(
                (
                    candidate
                    for candidate in root.findall("./CapFloorVolatilities/CapFloorVolatility")
                    if (candidate.findtext("./CurveId") or "").strip().upper() == surface_curve_id
                ),
                node,
            )
        fields = (
            "Calendar",
            "BusinessDayConvention",
            "DayCounter",
            "TimeInterpolation",
            "StrikeInterpolation",
            "InputType",
            "InterpolateOn",
            "VolatilityType",
            "RateComputationPeriod",
            "SettlementDays",
            "IncludeAtm",
        )
        out = {name: (surface_node.findtext(f"./{name}") or "").strip() for name in fields}
        # ORE defaults cap/floor surface inputs to term volatilities.
        if not out["InputType"]:
            out["InputType"] = "TermVolatilities"
        if source_curve:
            out["SurfaceCurveId"] = surface_curve_id
            out["SourceIndex"] = (source.findtext("./Index") or "").strip().upper()
            out["SourcePeriod"] = (source.findtext("./RateComputationPeriod") or "").strip()
            target_node = proxy.find("./Target")
            out["TargetPeriod"] = (
                (target_node.findtext("./RateComputationPeriod") or "").strip()
                if target_node is not None
                else ""
            )
            out["IsProxy"] = "true"
        else:
            out["SurfaceCurveId"] = curve_id
            out["SourceIndex"] = configured_index
            out["SourcePeriod"] = out.get("RateComputationPeriod", "")
            out["TargetPeriod"] = out.get("RateComputationPeriod", "")
            out["IsProxy"] = "false"
        return out
    return None


def proxy_capfloor_fallback_reason(curve_config_xml: str, index_name: str) -> str | None:
    """Explain proxy cap/floor configurations not faithfully handled natively."""
    config = _curve_config(curve_config_xml, index_name)
    if not config or config.get("IsProxy") != "true":
        return None
    input_type = config.get("InputType", "TermVolatilities").strip().upper()
    if input_type == "TERMVOLATILITIES":
        return (
            "ORE proxy source is a term-vol surface and requires cap-to-optionlet stripping; "
            "Python native pricing is disabled for this configuration"
        )
    if input_type not in {"OPTIONLETVOLATILITIES", "OPTIONLET"}:
        return f"ORE proxy cap/floor input type {input_type!r} is not supported by Python native pricing"
    if "SOFR" not in config.get("SourceIndex", "").strip().upper():
        return "Python native proxy cap/floor pricing currently supports only SOFR source indices"
    if config.get("SourcePeriod", "").strip().upper() != "1D":
        return "Python native proxy cap/floor pricing currently supports only a 1D overnight source index"
    try:
        settlement_days = int(config.get("SettlementDays", "0") or 0)
    except ValueError:
        return "Python native proxy cap/floor pricing cannot parse the source settlement-day convention"
    if settlement_days != 0:
        return "Python native proxy cap/floor pricing currently requires zero source settlement days"
    return None


@lru_cache(maxsize=32)
def _simulation_capfloor_vol_settings(xml: str) -> tuple[bool, str]:
    if not xml:
        return False, "CONSTANTVARIANCE"
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False, "CONSTANTVARIANCE"
    node = root.find("./Market/CapFloorVolatilities")
    if node is None:
        return False, "CONSTANTVARIANCE"
    simulate = (node.findtext("./Simulate") or "false").strip().lower() in {"y", "yes", "true", "1"}
    decay = (node.findtext("./ReactionToTimeDecay") or "ConstantVariance").strip().upper()
    return simulate, decay


@lru_cache(maxsize=64)
def _surface_quotes(
    raw_quotes: tuple[object, ...], curve_id: str, period: str
) -> tuple[tuple[str, tuple[tuple[float, float], ...], float | None], ...]:
    prefix = f"CAPFLOOR/RATE_NVOL/{curve_id.strip().upper()}/"
    points: dict[str, dict[float, float]] = {}
    atm_points: dict[str, float] = {}
    for quote in raw_quotes:
        key = str(getattr(quote, "key", "")).strip().upper()
        if not key.startswith(prefix):
            continue
        parts = key.split("/")
        if len(parts) < 5:
            continue
        if period and len(parts) >= 5 and parts[4].upper() != period.upper():
            continue
        try:
            volatility = float(getattr(quote, "value"))
        except (TypeError, ValueError):
            continue
        if len(parts) >= 8 and parts[-3] == "1" and parts[-2] == "1":
            if math.isfinite(volatility) and volatility >= 0.0:
                atm_points[parts[3].upper()] = volatility
            continue
        try:
            strike = float(parts[-1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(strike) and math.isfinite(volatility) and volatility >= 0.0:
            points.setdefault(parts[3].upper(), {})[strike] = volatility
    return tuple(
        (expiry, tuple(sorted(points.get(expiry, {}).items())), atm_points.get(expiry))
        for expiry in sorted(points.keys() | atm_points.keys(), key=_tenor_sort_key)
    )


def _tenor_sort_key(tenor: str) -> float:
    match = _TENOR_RE.match(tenor.strip())
    if not match:
        return float("inf")
    amount, unit = int(match.group(1)), match.group(2).upper()
    return amount * {"D": 1.0 / 365.0, "W": 7.0 / 365.0, "M": 1.0 / 12.0, "Y": 1.0}[unit]


def _advance_tenor(value: date, tenor: str, irs_utils: Any) -> date:
    match = _TENOR_RE.match(str(tenor).strip())
    if not match:
        raise ValueError(f"invalid tenor {tenor!r}")
    amount, unit = int(match.group(1)), match.group(2).upper()
    if unit == "D":
        return value + timedelta(days=amount)
    if unit == "W":
        return value + timedelta(days=7 * amount)
    months = amount if unit == "M" else amount * 12
    return irs_utils._add_months(value, months)


def _calendar_name(value: str) -> str:
    upper = str(value).strip().upper()
    if "US" in upper or "NEW YORK" in upper:
        return "US"
    if "TARGET" in upper or "EURO" in upper:
        return "TARGET"
    return upper or "US"


def _day_counter_name(value: str) -> str:
    upper = str(value).strip().upper().replace(" ", "")
    if "360" in upper:
        return "A360"
    if "ACTUAL/ACTUAL" in upper or upper in {"ACT", "ACT/ACT"}:
        return "ACT/ACT"
    return "A365F"


def _surface_volatility(
    adapter: Any,
    snapshot: Any,
    config: dict[str, str],
    *,
    asof: date,
    fixing_date: date,
    strike: float,
    option_time: float | None = None,
    atm_strike_for_expiry: Any = None,
) -> float | None:
    vol_type = config.get("VolatilityType", "Normal").strip().upper()
    input_type = config.get("InputType", "").strip().upper()
    if vol_type not in {"NORMAL", ""} or input_type not in {"OPTIONLETVOLATILITIES", "OPTIONLET"}:
        return None

    points = _surface_quotes(
        tuple(snapshot.market.raw_quotes),
        config.get("SurfaceCurveId", ""),
        config.get("SourcePeriod", ""),
    )
    if not points:
        return None

    irs = adapter._irs_utils
    calendar = _calendar_name(config.get("Calendar", "US"))
    convention = config.get("BusinessDayConvention", "Following")
    day_counter = _day_counter_name(config.get("DayCounter", "Actual/365 (Fixed)"))
    expiries: list[tuple[date, float, tuple[tuple[float, float], ...], float | None]] = []
    for tenor, strike_vols, atm_volatility in points:
        try:
            expiry_date = _advance_tenor(asof, tenor, irs)
            expiry_date = irs._adjust_date(expiry_date, convention, calendar)
            # ORE anchors 1D OIS optionlet quote tenors to the final daily
            # fixing date in the last cap coupon, normally one business day
            # before the adjusted cap maturity.
            if (
                config.get("IsProxy") == "true"
                and input_type in {"OPTIONLETVOLATILITIES", "OPTIONLET"}
                and config.get("SourcePeriod", "").strip().upper() == "1D"
                and int(config.get("SettlementDays", "0") or 0) == 0
            ):
                expiry_date = max(
                    asof + timedelta(days=1),
                    irs._advance_business_days(expiry_date, -1, calendar),
                )
            expiry_time = float(irs._time_from_dates(asof, expiry_date, day_counter))
        except (TypeError, ValueError):
            continue
        include_atm = config.get("IncludeAtm", "false").strip().lower() in {"true", "yes", "y", "1"}
        if include_atm and atm_volatility is not None and atm_strike_for_expiry is not None:
            try:
                atm_strike = float(atm_strike_for_expiry(expiry_date))
                if math.isfinite(atm_strike):
                    strike_map = dict(strike_vols)
                    strike_map.setdefault(atm_strike, atm_volatility)
                    strike_vols = tuple(sorted(strike_map.items()))
            except (TypeError, ValueError, ArithmeticError):
                return None
        expiries.append((expiry_date, expiry_time, strike_vols, atm_volatility))
    if not expiries:
        return None

    time_interp = config.get("TimeInterpolation", "Linear").strip().upper().replace("_", "")

    def vol_at(row: tuple[tuple[float, float], ...]) -> float | None:
        if not row:
            return None
        strikes = np.asarray([point[0] for point in row], dtype=float)
        vols = np.asarray([point[1] for point in row], dtype=float)
        method = config.get("StrikeInterpolation", "LinearFlat").strip().upper()
        if method in {"BACKWARDFLAT", "PREVIOUS", "FLAT"}:
            idx = int(np.searchsorted(strikes, strike, side="left"))
            idx = min(max(idx, 0), strikes.size - 1)
            return float(vols[idx])
        return float(np.interp(float(strike), strikes, vols, left=vols[0], right=vols[-1]))

    target_time = (
        max(float(option_time), 0.0)
        if option_time is not None
        else float(irs._time_from_dates(asof, fixing_date, day_counter))
    )
    expiry_times = np.asarray([point[1] for point in expiries], dtype=float)
    if time_interp in {"BACKWARDFLAT", "BACKWARD"}:
        idx = int(np.searchsorted(expiry_times, target_time, side="left"))
        idx = min(max(idx, 0), len(expiries) - 1)
        return vol_at(expiries[idx][2])
    if time_interp in {"FORWARDFLAT", "FORWARD"}:
        idx = int(np.searchsorted(expiry_times, target_time, side="right")) - 1
        idx = min(max(idx, 0), len(expiries) - 1)
        return vol_at(expiries[idx][2])

    vols = np.asarray([vol_at(point[2]) for point in expiries], dtype=float)
    valid = np.isfinite(vols)
    if not np.any(valid):
        return None
    return float(
        np.interp(
            target_time,
            expiry_times[valid],
            vols[valid],
            left=vols[valid][0],
            right=vols[valid][-1],
        )
    )


def _lookup_fixing(fixings: dict[tuple[str, str], float], index: str, fixing_date: date) -> float | None:
    index_key = str(index).strip().upper()
    date_key = fixing_date.isoformat()
    aliases = (index_key, index_key.split("-")[-1])
    for alias in aliases:
        value = fixings.get((alias, date_key))
        if value is not None:
            return float(value)
    return None


def _target_index_atm_forward(
    adapter: Any,
    *,
    index_name: str,
    period: str,
    fixing_date: date,
    asof: date,
    curve: Any,
    model_day_counter: str,
) -> float | None:
    """Forecast the proxy target index using ORE's fixing/value-date convention."""
    index = str(index_name).strip().upper()
    if not index.startswith("USD-LIBOR-"):
        return None

    irs = adapter._irs_utils
    calendar = "US"
    fixing_days = 2
    fixing_date = irs._adjust_date(fixing_date, "Following", calendar)
    value_date = irs._advance_business_days(fixing_date, fixing_days, calendar)
    maturity_date = _advance_tenor(value_date, period, irs)
    maturity_date = irs._adjust_date(maturity_date, "ModifiedFollowing", calendar)
    index_day_counter = irs._infer_index_day_counter(index, fallback="A360")
    accrual = float(irs._time_from_dates(value_date, maturity_date, index_day_counter))
    if accrual <= 0.0:
        return None

    value_time = float(irs._time_from_dates(asof, value_date, model_day_counter))
    maturity_time = float(irs._time_from_dates(asof, maturity_date, model_day_counter))
    if maturity_time <= value_time:
        return None
    return (float(curve(value_time)) / float(curve(maturity_time)) - 1.0) / accrual


def _overnight_proxy_source_atm_level(
    adapter: Any,
    curve: Any,
    *,
    fixing_date: date,
    asof: date,
    model_day_counter: str,
    calendar: str,
) -> float:
    irs = adapter._irs_utils
    source_start = irs._adjust_date(fixing_date, "Following", calendar)
    source_end = irs._advance_business_days(source_start, 1, calendar)
    source_start_time = float(irs._time_from_dates(asof, source_start, model_day_counter))
    source_end_time = float(irs._time_from_dates(asof, source_end, model_day_counter))
    source_accrual = (source_end - source_start).days / 360.0
    if source_accrual <= 0.0:
        raise ValueError("invalid overnight proxy accrual")
    return (float(curve(source_start_time)) / float(curve(source_end_time)) - 1.0) / source_accrual


def _market_capfloor_terms(adapter: Any, spec: Any, ctx: Any) -> dict[str, Any] | None:
    state = spec.sticky_state
    if not isinstance(state, dict):
        return None
    definition = state.get("definition")
    index_name = str(state.get("index_name", "")).strip().upper()
    if definition is None or not index_name:
        return None

    xml_buffers = getattr(ctx.snapshot.config, "xml_buffers", {}) or {}
    config = _curve_config(str(xml_buffers.get("curveconfig.xml", "")), index_name)
    if not config or config.get("IsProxy") != "true":
        return None
    simulate_capfloor_vols, time_decay = _simulation_capfloor_vol_settings(
        str(xml_buffers.get("simulation.xml", ""))
    )
    source_index = config.get("SourceIndex", "").strip().upper()
    if not source_index or "SOFR" not in source_index:
        return None
    source_period = config.get("SourcePeriod", "").strip()
    target_period = config.get("TargetPeriod", "").strip()
    if not source_period or not target_period:
        return None

    asof = date.fromisoformat(adapter._normalized_asof(ctx.snapshot))
    irs = adapter._irs_utils
    model_day_counter = adapter._day_counter_from_sim_xml(xml_buffers.get("simulation.xml", ""))
    p_disc = ctx.inputs.discount_curves.get(spec.ccy.upper())
    if p_disc is None:
        return None
    try:
        p_target = adapter._resolve_index_curve(ctx.inputs, spec.ccy, index_name)
        p_source = adapter._resolve_index_curve(ctx.inputs, spec.ccy, source_index)
    except Exception:
        return None

    fixing_dates = getattr(definition, "fixing_date", None)
    if fixing_dates is None:
        return None
    fixing_dates = np.asarray(fixing_dates, dtype=object).reshape(-1)
    start_times = np.asarray(definition.start_time, dtype=float).reshape(-1)
    end_times = np.asarray(definition.end_time, dtype=float).reshape(-1)
    pay_times = np.asarray(definition.pay_time, dtype=float).reshape(-1)
    accruals = np.asarray(definition.accrual, dtype=float).reshape(-1)
    notionals = np.asarray(definition.notional, dtype=float).reshape(-1)
    strikes = np.asarray(definition.strike, dtype=float).reshape(-1)
    gearings = np.asarray(
        definition.gearing if definition.gearing is not None else np.ones_like(strikes), dtype=float
    ).reshape(-1)
    spreads = np.asarray(
        definition.spread if definition.spread is not None else np.zeros_like(strikes), dtype=float
    ).reshape(-1)
    size = min(
        len(fixing_dates),
        start_times.size,
        end_times.size,
        pay_times.size,
        accruals.size,
        notionals.size,
        strikes.size,
        gearings.size,
        spreads.size,
    )
    if size == 0:
        return None

    fixings = adapter._fixings_lookup(ctx.snapshot)
    position = float(definition.position)
    is_cap = str(definition.option_type).strip().lower() == "cap"
    source_calendar = _calendar_name(config.get("Calendar", "US"))
    state_index_accrual = state.get("index_accrual")
    if state_index_accrual is None:
        index_accruals = accruals
    else:
        index_accruals = np.asarray(state_index_accrual, dtype=float).reshape(-1)
    if index_accruals.size < size:
        return None

    def source_atm_level(source_fixing_date: date) -> float:
        if source_period.upper() != "1D":
            raise ValueError("only 1D overnight proxy ATM levels are supported")
        return _overnight_proxy_source_atm_level(
            adapter,
            p_source,
            fixing_date=source_fixing_date,
            asof=asof,
            model_day_counter=model_day_counter,
            calendar=source_calendar,
        )

    target_forwards = np.empty(size, dtype=float)
    target_atm_forwards = np.empty(size, dtype=float)
    proxy_shifts = np.empty(size, dtype=float)
    volatilities = np.empty(size, dtype=float)
    expiry_times = np.empty(size, dtype=float)
    effective_strikes = np.empty(size, dtype=float)
    call_flags = np.empty(size, dtype=bool)
    initial_fixings: list[float | None] = []

    for i in range(size):
        try:
            fixing_date = date.fromisoformat(str(fixing_dates[i])[:10])
        except (TypeError, ValueError):
            return None

        gearing = float(gearings[i])
        spread = float(spreads[i])
        strike = float(strikes[i])
        if abs(gearing) < 1.0e-14:
            return None

        fixing = _lookup_fixing(fixings, index_name, fixing_date)
        # ORE prices the coupon already accruing at valuation from the projection curve.
        project_current_coupon = fixing_date <= asof and abs(float(start_times[i])) <= 1.0e-12
        if fixing is not None and fixing_date <= asof and not project_current_coupon:
            target_forward = fixing
        else:
            try:
                # ORE forecasts the coupon over its actual schedule dates, which
                # can differ by a business day from advancing the index tenor.
                value_time = max(float(start_times[i]), 0.0)
                maturity_time = float(end_times[i])
                index_accrual = float(index_accruals[i])
                if index_accrual <= 0.0 or maturity_time <= value_time:
                    return None
                target_forward = (float(p_target(value_time)) / float(p_target(maturity_time)) - 1.0) / index_accrual
            except Exception:
                return None

        target_atm_forward = fixing if fixing is not None and fixing_date <= asof else None
        if target_atm_forward is None:
            try:
                target_atm_forward = _target_index_atm_forward(
                    adapter,
                    index_name=index_name,
                    period=target_period,
                    fixing_date=fixing_date,
                    asof=asof,
                    curve=p_target,
                    model_day_counter=model_day_counter,
                )
            except Exception:
                return None
        if target_atm_forward is None:
            target_atm_forward = target_forward

        adjusted_strike = (strike - spread) / gearing
        source_fixing = _lookup_fixing(fixings, source_index, fixing_date)
        if source_fixing is None:
            try:
                source_fixing = source_atm_level(fixing_date)
            except Exception:
                return None
        proxy_shift = float(source_fixing) - float(target_atm_forward)
        expiry = max(
            float(
                irs._time_from_dates(
                    asof,
                    fixing_date,
                    _day_counter_name(config.get("DayCounter", "Actual/365 (Fixed)")),
                )
            ),
            0.0,
        )
        initial_fixing = None
        if fixing_date <= asof:
            initial_fixing = target_forward if project_current_coupon else fixing
        if initial_fixing is None:
            volatility = _surface_volatility(
                adapter,
                ctx.snapshot,
                config,
                asof=asof,
                fixing_date=fixing_date,
                strike=adjusted_strike + proxy_shift,
                atm_strike_for_expiry=source_atm_level,
            )
            if volatility is None or not math.isfinite(volatility):
                return None
        else:
            volatility = 0.0

        target_forwards[i] = float(target_forward)
        target_atm_forwards[i] = float(target_atm_forward)
        proxy_shifts[i] = proxy_shift
        volatilities[i] = float(volatility)
        expiry_times[i] = expiry
        effective_strikes[i] = adjusted_strike
        call_flags[i] = is_cap if gearing > 0.0 else not is_cap
        initial_fixings.append(None if initial_fixing is None else float(initial_fixing))

    fixing_time = np.asarray(
        definition.fixing_time if definition.fixing_time is not None else definition.start_time,
        dtype=float,
    ).reshape(-1)[:size]
    return {
        "config": config,
        "simulate_capfloor_vols": simulate_capfloor_vols,
        "time_decay": time_decay,
        "definition": definition,
        "index_name": index_name,
        "source_index": source_index,
        "source_atm_level": source_atm_level,
        "p_disc": p_disc,
        "p_target": p_target,
        "target_forwards": target_forwards,
        "target_atm_forwards": target_atm_forwards,
        "proxy_shifts": proxy_shifts,
        "volatilities": volatilities,
        "expiry_times": expiry_times,
        "effective_strikes": effective_strikes,
        "call_flags": call_flags,
        "initial_fixings": tuple(initial_fixings),
        "fixing_time": fixing_time,
        "index_accrual": index_accruals[:size],
        "position": position,
        "is_cap": is_cap,
        "size": size,
    }


def market_capfloor_t0_value(adapter: Any, spec: Any, ctx: Any) -> float | None:
    """Price a vanilla cap/floor at t0 using ORE's configured normal-vol surface."""
    terms = _market_capfloor_terms(adapter, spec, ctx)
    if terms is None:
        return None

    definition = terms["definition"]
    p_disc = terms["p_disc"]
    total = 0.0
    for i in range(int(terms["size"])):
        gearing = float(definition.gearing[i]) if definition.gearing is not None else 1.0
        spread = float(definition.spread[i]) if definition.spread is not None else 0.0
        initial_fixing = terms["initial_fixings"][i]
        if initial_fixing is not None:
            effective_rate = gearing * initial_fixing + spread
            if terms["is_cap"]:
                option_rate = max(effective_rate - float(definition.strike[i]), 0.0)
            else:
                option_rate = max(float(definition.strike[i]) - effective_rate, 0.0)
        else:
            option_rate = float(
                adapter._normal_option_rate(
                    np.asarray([terms["target_forwards"][i]], dtype=float),
                    float(terms["effective_strikes"][i]),
                    np.asarray(
                        [terms["volatilities"][i] * math.sqrt(terms["expiry_times"][i])],
                        dtype=float,
                    ),
                    is_call=bool(terms["call_flags"][i]),
                )[0]
            )
            option_rate *= abs(gearing)
        total += (
            float(terms["position"])
            * float(definition.notional[i])
            * float(definition.accrual[i])
            * float(p_disc(float(definition.pay_time[i])))
            * option_rate
        )
    return float(total)


def _market_realized_forward_paths(
    adapter: Any,
    ctx: Any,
    terms: dict[str, Any],
    start: np.ndarray,
    end: np.ndarray,
    index_accrual: np.ndarray,
    fixing_dates: np.ndarray,
) -> np.ndarray:
    ir_options = adapter._ir_options_mod
    times = np.asarray(ctx.inputs.times, dtype=float)
    x_paths = np.asarray(ctx.x_paths, dtype=float)
    fixing_time = np.asarray(terms["fixing_time"], dtype=float)
    realized = np.zeros((fixing_time.size, x_paths.shape[1]), dtype=float)
    fixings = adapter._fixings_lookup(ctx.snapshot)
    asof = date.fromisoformat(adapter._normalized_asof(ctx.snapshot))

    for i, fixing_date_value in enumerate(fixing_dates):
        try:
            fixing_date = date.fromisoformat(str(fixing_date_value)[:10])
        except ValueError:
            continue
        known_fixing = terms["initial_fixings"][i]
        if known_fixing is None:
            known_fixing = _lookup_fixing(fixings, terms["index_name"], fixing_date)
        if known_fixing is not None and fixing_date <= asof:
            realized[i, :] = known_fixing

    nonpositive = np.flatnonzero(fixing_time <= 1.0e-12)
    for i in nonpositive:
        if fixing_dates[i] and _lookup_fixing(
            fixings,
            terms["index_name"],
            date.fromisoformat(str(fixing_dates[i])[:10]),
        ) is not None:
            continue
        p_start = float(terms["p_target"](max(0.0, float(start[i]))))
        p_end = float(terms["p_target"](float(end[i])))
        realized[i, :] = (p_start / p_end - 1.0) / max(float(index_accrual[i]), 1.0e-18)

    future = np.flatnonzero(fixing_time > 1.0e-12)
    for fixing_t in np.unique(fixing_time[future]):
        indices = future[np.abs(fixing_time[future] - fixing_t) <= 1.0e-12]
        x_fix = ir_options.interpolate_path_grid(times, x_paths, float(fixing_t))[0, :]
        realized[indices, :] = ir_options.forward_rate_from_bonds(
            ctx.model,
            terms["p_disc"],
            terms["p_target"],
            float(fixing_t),
            x_fix,
            start[indices],
            end[indices],
            index_accrual[indices],
            fixing_time=float(fixing_t),
        )
    return realized


def market_capfloor_npv_paths(adapter: Any, spec: Any, ctx: Any) -> np.ndarray | None:
    """Revalue cap/floor optionlets pathwise using the configured market normal vol."""
    terms = _market_capfloor_terms(adapter, spec, ctx)
    if terms is None:
        return None
    if terms["simulate_capfloor_vols"] or terms["time_decay"] != "CONSTANTVARIANCE":
        return None

    definition = terms["definition"]
    model = ctx.model
    p_disc = terms["p_disc"]
    p_fwd = terms["p_target"]
    ir_options = adapter._ir_options_mod
    times = np.asarray(ctx.inputs.times, dtype=float)
    x_paths = np.asarray(ctx.x_paths, dtype=float)
    if x_paths.shape[0] != times.size:
        return None

    start = np.asarray(definition.start_time, dtype=float)[: terms["size"]]
    end = np.asarray(definition.end_time, dtype=float)[: terms["size"]]
    pay = np.asarray(definition.pay_time, dtype=float)[: terms["size"]]
    accrual = np.asarray(definition.accrual, dtype=float)[: terms["size"]]
    notional = np.asarray(definition.notional, dtype=float)[: terms["size"]]
    strike = np.asarray(definition.strike, dtype=float)[: terms["size"]]
    gearing = np.asarray(
        definition.gearing if definition.gearing is not None else np.ones_like(definition.strike), dtype=float
    )[: terms["size"]]
    spread = np.asarray(
        definition.spread if definition.spread is not None else np.zeros_like(definition.strike), dtype=float
    )[: terms["size"]]
    fixing_time = np.asarray(terms["fixing_time"], dtype=float)
    fixing_dates = np.asarray(definition.fixing_date, dtype=object)[: terms["size"]]
    index_accrual = np.asarray(terms["index_accrual"], dtype=float)
    realized_forward = _market_realized_forward_paths(
        adapter,
        ctx,
        terms,
        start,
        end,
        index_accrual,
        fixing_dates,
    )

    asof = date.fromisoformat(adapter._normalized_asof(ctx.snapshot))
    vol_day_counter = _day_counter_name(terms["config"].get("DayCounter", "Actual/365 (Fixed)"))
    remaining_expiries = np.zeros((times.size, terms["size"]), dtype=float)
    rolled_volatilities = np.zeros_like(remaining_expiries)
    # ORE's dynamic optionlet structure queries its source surface at remaining expiry.
    for time_idx, t in enumerate(times):
        valuation_date = date.fromisoformat(adapter._date_from_time_cached(ctx.snapshot, float(t)))
        for i in range(int(terms["size"])):
            if float(t) >= float(fixing_time[i]) - 1.0e-12:
                continue
            fixing_date = date.fromisoformat(str(fixing_dates[i])[:10])
            remaining = max(
                float(adapter._irs_utils._time_from_dates(valuation_date, fixing_date, vol_day_counter)),
                0.0,
            )
            volatility = _surface_volatility(
                adapter,
                ctx.snapshot,
                terms["config"],
                asof=asof,
                fixing_date=fixing_date,
                strike=float(terms["effective_strikes"][i] + terms["proxy_shifts"][i]),
                option_time=remaining,
                atm_strike_for_expiry=terms["source_atm_level"],
            )
            if volatility is None or not math.isfinite(volatility):
                return None
            remaining_expiries[time_idx, i] = remaining
            rolled_volatilities[time_idx, i] = float(volatility)

    final_pay = float(np.max(pay)) if pay.size else 0.0
    values = np.zeros_like(x_paths, dtype=float)
    for time_idx, t in enumerate(times):
        if t < 0.0:
            return None
        if t >= final_pay - 1.0e-12:
            continue
        live = pay >= float(t) - 1.0e-12
        if not np.any(live):
            continue
        coupon_indices = np.flatnonzero(live)
        x_t = x_paths[time_idx, :]
        p_t = float(p_disc(float(t)))
        discount_bonds = model.discount_bond_paths(
            float(t),
            pay[live],
            x_t,
            p_t,
            lambda maturity: float(p_disc(float(maturity))),
        )
        coupon_values = np.zeros((coupon_indices.size, x_paths.shape[1]), dtype=float)
        fixed = fixing_time[live] <= float(t) + 1.0e-12
        if np.any(fixed):
            fixed_global = coupon_indices[fixed]
            fixed_rates = realized_forward[fixed_global, :]
            effective_rates = gearing[fixed_global, None] * fixed_rates + spread[fixed_global, None]
            if terms["is_cap"]:
                payoff = np.maximum(effective_rates - strike[fixed_global, None], 0.0)
            else:
                payoff = np.maximum(strike[fixed_global, None] - effective_rates, 0.0)
            coupon_values[fixed, :] = (
                float(terms["position"])
                * notional[fixed_global, None]
                * accrual[fixed_global, None]
                * payoff
            )

        unfixed = ~fixed
        if np.any(unfixed):
            unfixed_global = coupon_indices[unfixed]
            forward = ir_options.forward_rate_from_bonds(
                model,
                p_disc,
                p_fwd,
                float(t),
                x_t,
                start[unfixed_global],
                end[unfixed_global],
                index_accrual[unfixed_global],
                fixing_time=float(t),
            )
            for local_idx, global_idx in enumerate(unfixed_global):
                remaining = float(remaining_expiries[time_idx, global_idx])
                stddev = float(rolled_volatilities[time_idx, global_idx]) * math.sqrt(remaining)
                option_rate = adapter._normal_option_rate(
                    np.asarray(forward[local_idx, :], dtype=float),
                    float(terms["effective_strikes"][global_idx]),
                    np.full(x_paths.shape[1], stddev, dtype=float),
                    is_call=bool(terms["call_flags"][global_idx]),
                )
                local_coupon_idx = int(np.flatnonzero(unfixed)[local_idx])
                coupon_values[local_coupon_idx, :] = (
                    float(terms["position"])
                    * notional[global_idx]
                    * accrual[global_idx]
                    * abs(gearing[global_idx])
                    * option_rate
                )
        values[time_idx, :] = np.sum(coupon_values * discount_bonds, axis=0)
    return values


__all__ = ["market_capfloor_npv_paths", "market_capfloor_t0_value"]
