from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from typing import Optional


@dataclasses.dataclass(frozen=True)
class RateFutureModelParams:
    model: str = "none"
    mean_reversion: float = 0.03
    volatility: float = 0.0


@dataclasses.dataclass(frozen=True)
class RateFutureQuote:
    future_type: str
    ccy: str
    price: float
    contract_label: str
    contract_start: date
    contract_end: date
    accrual_years: float
    start_time_years: float
    end_time_years: float
    quote_key: str = ""
    exchange: str = ""
    underlying_tenor: str = ""
    index: str = ""
    convexity_adjustment: float = 0.0


def futures_price_to_rate(price: float) -> float:
    return (100.0 - float(price)) / 100.0


def future_convexity_adjustment(
    quote: RateFutureQuote,
    model_params: Optional[RateFutureModelParams] = None,
) -> float:
    if abs(float(quote.convexity_adjustment)) > 0.0:
        return float(quote.convexity_adjustment)
    if model_params is None:
        return 0.0

    model = str(model_params.model).strip().lower()
    if model in ("", "none"):
        return 0.0
    if model not in ("hw", "hull_white", "lgm"):
        raise ValueError(f"Unsupported futures convexity model '{model_params.model}'")

    a = max(float(model_params.mean_reversion), 1.0e-8)
    sigma = max(float(model_params.volatility), 0.0)
    if sigma == 0.0:
        return 0.0

    # Simple one-factor Gaussian approximation for the futures-vs-forward bias.
    delta = max(float(quote.accrual_years), 1.0e-12)
    start = max(float(quote.start_time_years), 0.0)
    b = (1.0 - pow(2.718281828459045, -a * delta)) / a
    variance_term = (1.0 - pow(2.718281828459045, -2.0 * a * start)) / (2.0 * a)
    return 0.5 * sigma * sigma * b * b * variance_term


def future_forward_rate(
    quote: RateFutureQuote,
    model_params: Optional[RateFutureModelParams] = None,
) -> tuple[float, float, float]:
    futures_rate = futures_price_to_rate(quote.price)
    convexity = future_convexity_adjustment(quote, model_params)
    return futures_rate - convexity, futures_rate, convexity


def future_to_fra_rate(
    quote: RateFutureQuote,
    model_params: Optional[RateFutureModelParams] = None,
) -> float:
    forward, _, _ = future_forward_rate(quote, model_params)
    return forward


def month_code_to_int(value: str) -> int:
    month = str(value).strip().upper()
    if month in {"F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"}:
        return {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}[month]
    return int(month)


def parse_contract_month(label: str) -> tuple[int, int]:
    text = str(label).strip()
    if len(text) == 7 and text[4] == "-":
        return int(text[:4]), int(text[5:7])
    if len(text) == 6 and text[:4].isdigit():
        return int(text[:4]), int(text[4:6])
    raise ValueError(f"Unsupported future contract label '{label}'")


def add_months(value: date, months: int) -> date:
    year = value.year + (value.month - 1 + months) // 12
    month = (value.month - 1 + months) % 12 + 1
    month_lengths = [31, 29 if _is_leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(value.day, month_lengths[month - 1])
    return date(year, month, day)


def build_rate_future_quote(
    *,
    asof_date: date,
    future_type: str,
    ccy: str,
    price: float,
    contract_label: str,
    underlying_tenor: str,
    quote_key: str = "",
    exchange: str = "",
    index: str = "",
    contract_start: date | None = None,
    contract_end: date | None = None,
    convexity_adjustment: float = 0.0,
) -> RateFutureQuote:
    tenor_months = tenor_to_months(underlying_tenor)
    start = contract_start or infer_future_start_date(contract_label, future_type=future_type)
    end = contract_end or add_months(start, tenor_months)
    start_time = year_fraction_365(asof_date, start)
    end_time = year_fraction_365(asof_date, end)
    accrual = max(year_fraction_365(start, end), 1.0e-12)
    return RateFutureQuote(
        future_type=str(future_type).upper(),
        ccy=str(ccy).upper(),
        price=float(price),
        contract_label=str(contract_label),
        contract_start=start,
        contract_end=end,
        accrual_years=accrual,
        start_time_years=max(start_time, 0.0),
        end_time_years=max(end_time, 0.0),
        quote_key=str(quote_key),
        exchange=str(exchange),
        underlying_tenor=str(underlying_tenor).upper(),
        index=str(index),
        convexity_adjustment=float(convexity_adjustment),
    )


def infer_future_start_date(contract_label: str, *, future_type: str) -> date:
    year, month = parse_contract_month(contract_label)
    if str(future_type).upper() == "MM_FUTURE" and month in (3, 6, 9, 12):
        return third_wednesday(year, month)
    return date(year, month, 1)


def third_wednesday(year: int, month: int) -> date:
    first = date(year, month, 1)
    weekday = first.weekday()
    days_to_wed = (2 - weekday) % 7
    return first + timedelta(days=days_to_wed + 14)


def tenor_to_months(text: str) -> int:
    value = str(text).strip().upper()
    if not value:
        raise ValueError("Tenor must be provided for rate futures")
    if value.endswith("M"):
        return int(value[:-1])
    if value.endswith("Y"):
        return int(value[:-1]) * 12
    raise ValueError(f"Unsupported tenor '{text}'")


def year_fraction_365(start: date, end: date) -> float:
    return (end - start).days / 365.0


def _is_leap(year: int) -> bool:
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)
