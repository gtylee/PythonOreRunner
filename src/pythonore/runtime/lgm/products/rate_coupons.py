from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np


def capped_floored_rate(
    raw_rate: np.ndarray,
    cap: Optional[float] = None,
    floor: Optional[float] = None,
) -> np.ndarray:
    out = np.asarray(raw_rate, dtype=float).copy()
    if floor is not None:
        out = np.maximum(out, float(floor))
    if cap is not None:
        out = np.minimum(out, float(cap))
    return out


def digital_option_rate(
    raw_rate: np.ndarray,
    strike: float,
    payoff: float,
    *,
    is_call: bool,
    long_short: float,
    fixed_mode: bool,
    atm_included: bool,
    capped_rate_fn: Optional[Callable[[float, float], np.ndarray]] = None,
) -> np.ndarray:
    if math.isnan(float(strike)):
        return np.zeros_like(raw_rate, dtype=float)
    strike_value = float(strike)
    eps = 1.0e-4
    if is_call and abs(strike_value) < eps / 2.0:
        strike_value = eps / 2.0

    if fixed_mode:
        if is_call:
            hit = raw_rate >= strike_value if atm_included else raw_rate > strike_value
        else:
            hit = raw_rate <= strike_value if atm_included else raw_rate < strike_value
        step = float(payoff) if not math.isnan(float(payoff)) else strike_value
        if math.isnan(float(payoff)):
            vanilla = np.maximum(raw_rate - strike_value, 0.0) if is_call else np.maximum(strike_value - raw_rate, 0.0)
            return float(long_short) * (step * hit.astype(float) + (vanilla if is_call else -vanilla))
        return float(long_short) * step * hit.astype(float)

    right = strike_value + eps / 2.0
    left = strike_value - eps / 2.0
    if capped_rate_fn is not None:
        next_rate = capped_rate_fn(right if is_call else math.nan, math.nan if is_call else right)
        prev_rate = capped_rate_fn(left if is_call else math.nan, math.nan if is_call else left)
    elif is_call:
        next_rate = capped_floored_rate(raw_rate, cap=right)
        prev_rate = capped_floored_rate(raw_rate, cap=left)
    else:
        next_rate = capped_floored_rate(raw_rate, floor=right)
        prev_rate = capped_floored_rate(raw_rate, floor=left)
    step = float(payoff) if not math.isnan(float(payoff)) else strike_value
    option_rate = step * (next_rate - prev_rate) / eps
    if math.isnan(float(payoff)):
        if capped_rate_fn is not None:
            at_strike = capped_rate_fn(strike_value if is_call else math.nan, math.nan if is_call else strike_value)
        else:
            at_strike = capped_floored_rate(raw_rate, cap=strike_value) if is_call else capped_floored_rate(raw_rate, floor=strike_value)
        vanilla = raw_rate - at_strike if is_call else -raw_rate + at_strike
        option_rate = option_rate + vanilla if is_call else option_rate - vanilla
    return float(long_short) * option_rate


def rate_leg_pricing_cache_key(ccy: str, leg: Dict[str, object]) -> tuple[object, ...]:
    kind = str(leg.get("kind", "")).upper()
    key: list[object] = [ccy.upper(), kind]
    scalar_fields = (
        "notional",
        "sign",
        "index_name",
        "index_name_1",
        "index_name_2",
        "fixing_days",
        "is_in_arrears",
        "is_averaged",
        "has_sub_periods",
        "day_counter",
        "call_strike",
        "call_payoff",
        "put_strike",
        "put_payoff",
        "call_position",
        "put_position",
        "is_call_atm_included",
        "is_put_atm_included",
        "naked_option",
        "cap",
        "floor",
    )
    for field in scalar_fields:
        value = leg.get(field)
        if isinstance(value, np.ndarray):
            continue
        key.append((field, value))
    array_fields = (
        "pay_time",
        "start_time",
        "end_time",
        "pay_date",
        "start_date",
        "end_date",
        "fixing_date",
        "fixing_time",
        "amount",
        "spread",
        "gearing",
        "accrual",
        "index_accrual",
        "quoted_coupon",
        "is_historically_fixed",
    )
    for field in array_fields:
        if field not in leg:
            continue
        arr = np.asarray(leg.get(field))
        key.append((field, str(arr.dtype), tuple(arr.tolist())))
    return tuple(key)


def torch_curve_from_handle(
    curve_cache: Dict[tuple[str, str], object],
    torch_curve_ctor: object,
    torch_device: str,
    key: tuple[str, str],
    curve: Callable[[float], float],
    sample_times: np.ndarray,
):
    cached = curve_cache.get(key)
    if cached is not None:
        return cached
    pts = np.unique(np.asarray(sample_times, dtype=float))
    pts = pts[np.isfinite(pts)]
    pts.sort()
    cached = torch_curve_ctor(
        times=pts,
        dfs=np.asarray([float(curve(float(t))) for t in pts], dtype=float),
        device=torch_device,
    )
    curve_cache[key] = cached
    return cached


__all__ = [
    "capped_floored_rate",
    "digital_option_rate",
    "rate_leg_pricing_cache_key",
    "torch_curve_from_handle",
]
