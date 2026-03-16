from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class InflationCurve:
    index: str
    curve_type: str
    times: tuple[float, ...]
    rates: tuple[float, ...]

    def rate(self, t: float) -> float:
        x = max(float(t), 0.0)
        if not self.times:
            return 0.0
        times = np.asarray(self.times, dtype=float)
        rates = np.asarray(self.rates, dtype=float)
        if x <= times[0]:
            return float(rates[0])
        if x >= times[-1]:
            return float(rates[-1])
        return float(np.interp(x, times, rates))

    def growth(self, t: float) -> float:
        r = self.rate(float(t))
        return (1.0 + r) ** max(float(t), 0.0)


@dataclass(frozen=True)
class InflationLgmParams:
    index: str
    currency: str
    volatility_times: tuple[float, ...]
    volatility_values: tuple[float, ...]
    reversion_times: tuple[float, ...]
    reversion_values: tuple[float, ...]
    shift: float = 0.0
    scaling: float = 1.0


@dataclass(frozen=True)
class JarrowYildirimParams:
    index: str
    currency: str
    real_rate_vol_times: tuple[float, ...]
    real_rate_vol_values: tuple[float, ...]
    real_rate_rev_times: tuple[float, ...]
    real_rate_rev_values: tuple[float, ...]
    index_vol_times: tuple[float, ...]
    index_vol_values: tuple[float, ...]
    shift: float = 0.0
    scaling: float = 1.0


@dataclass(frozen=True)
class InflationCalibrationInstrument:
    family: str
    tenor_or_maturity: str
    strike: float | None = None
    option_type: str | None = None
    parameter: str | None = None


@dataclass(frozen=True)
class InflationModelSpec:
    family: str
    index: str
    currency: str
    params: InflationLgmParams | JarrowYildirimParams
    calibration_instruments: tuple[InflationCalibrationInstrument, ...] = ()


@dataclass(frozen=True)
class InflationSwapLeg:
    leg_type: str
    currency: str
    payer: bool
    notional: float
    schedule_dates: tuple[str, ...]
    start_date: str | None = None
    end_date: str | None = None
    rate: float | None = None
    index: str | None = None
    base_cpi: float | None = None
    observation_lag: str | None = None
    subtract_notional: bool = False


@dataclass(frozen=True)
class InflationSwapDefinition:
    trade_id: str
    currency: str
    maturity_years: float
    fixed_leg: InflationSwapLeg | None = None
    float_leg: InflationSwapLeg | None = None
    inflation_leg: InflationSwapLeg | None = None


@dataclass(frozen=True)
class InflationCapFloorDefinition:
    trade_id: str
    currency: str
    inflation_type: str
    option_type: str
    index: str
    strike: float
    notional: float
    maturity_years: float
    base_cpi: float | None = None
    observation_lag: str | None = None
    long_short: str = "Long"


def _parse_csv_floats(text: str | None) -> tuple[float, ...]:
    raw = (text or "").strip()
    if not raw:
        return ()
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _parse_tenor_years(text: str) -> float:
    m = re.fullmatch(r"\s*(\d+)\s*([DWMY])\s*", str(text).upper())
    if not m:
        raise ValueError(f"unsupported tenor '{text}'")
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "D":
        return n / 365.25
    if unit == "W":
        return 7.0 * n / 365.25
    if unit == "M":
        return n / 12.0
    return n


def parse_inflation_models_from_simulation_xml(xml_path: str | Path) -> dict[str, InflationModelSpec]:
    root = ET.parse(str(xml_path)).getroot()
    section = root.find("./CrossAssetModel/InflationIndexModels")
    if section is None:
        return {}
    out: dict[str, InflationModelSpec] = {}
    for node in list(section):
        tag = node.tag.strip()
        index = (node.attrib.get("index", "") or "").strip()
        currency = (node.findtext("./Currency") or "").strip()
        if not index:
            continue
        if tag == "LGM":
            params = InflationLgmParams(
                index=index,
                currency=currency,
                volatility_times=_parse_csv_floats(node.findtext("./Volatility/TimeGrid")),
                volatility_values=_parse_csv_floats(node.findtext("./Volatility/InitialValue")),
                reversion_times=_parse_csv_floats(node.findtext("./Reversion/TimeGrid")),
                reversion_values=_parse_csv_floats(node.findtext("./Reversion/InitialValue")),
                shift=float(node.findtext("./ParameterTransformation/ShiftHorizon") or 0.0),
                scaling=float(node.findtext("./ParameterTransformation/Scaling") or 1.0),
            )
            inst: list[InflationCalibrationInstrument] = []
            for expiry, strike in zip(
                (x.strip() for x in (node.findtext("./CalibrationCapFloors/Expiries") or "").split(",") if x.strip()),
                _parse_csv_floats(node.findtext("./CalibrationCapFloors/Strikes")),
            ):
                inst.append(
                    InflationCalibrationInstrument(
                        family="ZeroInflationCapFloor",
                        tenor_or_maturity=expiry,
                        strike=float(strike),
                        option_type=(node.findtext("./CalibrationCapFloors/CapFloor") or "").strip(),
                    )
                )
            out[index] = InflationModelSpec("LGM", index, currency, params, tuple(inst))
        elif tag == "JarrowYildirim":
            params = JarrowYildirimParams(
                index=index,
                currency=currency,
                real_rate_vol_times=_parse_csv_floats(node.findtext("./RealRate/Volatility/TimeGrid")),
                real_rate_vol_values=_parse_csv_floats(node.findtext("./RealRate/Volatility/InitialValue")),
                real_rate_rev_times=_parse_csv_floats(node.findtext("./RealRate/Reversion/TimeGrid")),
                real_rate_rev_values=_parse_csv_floats(node.findtext("./RealRate/Reversion/InitialValue")),
                index_vol_times=_parse_csv_floats(node.findtext("./Index/Volatility/TimeGrid")),
                index_vol_values=_parse_csv_floats(node.findtext("./Index/Volatility/InitialValue")),
                shift=float(node.findtext("./RealRate/ParameterTransformation/ShiftHorizon") or 0.0),
                scaling=float(node.findtext("./RealRate/ParameterTransformation/Scaling") or 1.0),
            )
            inst: list[InflationCalibrationInstrument] = []
            for basket in node.findall("./CalibrationBaskets/CalibrationBasket"):
                parameter = (basket.attrib.get("parameter", "") or "").strip()
                for capfloor in basket.findall("./CpiCapFloor"):
                    inst.append(
                        InflationCalibrationInstrument(
                            family="ZeroInflationCapFloor",
                            tenor_or_maturity=(capfloor.findtext("./Maturity") or "").strip(),
                            strike=float(capfloor.findtext("./Strike") or 0.0),
                            option_type=(capfloor.findtext("./Type") or "").strip(),
                            parameter=parameter,
                        )
                    )
                for yoy in basket.findall("./YoYSwap"):
                    inst.append(
                        InflationCalibrationInstrument(
                            family="YoYSwap",
                            tenor_or_maturity=(yoy.findtext("./Tenor") or "").strip(),
                            parameter=parameter,
                        )
                    )
            out[index] = InflationModelSpec("JarrowYildirim", index, currency, params, tuple(inst))
    return out


def load_inflation_curve_from_market_data(
    market_data_path: str | Path,
    asof_date: str,
    index: str,
    *,
    curve_type: str = "ZC",
) -> InflationCurve:
    asof_tokens = {str(asof_date).replace("-", ""), str(asof_date)}
    prefix = "ZC_INFLATIONSWAP/RATE" if str(curve_type).upper() == "ZC" else "YY_INFLATIONSWAP/RATE"
    times: list[float] = []
    rates: list[float] = []
    with open(market_data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            parts = txt.split()
            if len(parts) < 3 or parts[0] not in asof_tokens:
                continue
            key = parts[1]
            match = re.fullmatch(rf"{re.escape(prefix)}/{re.escape(index)}/([^/]+)", key)
            if not match:
                continue
            try:
                times.append(_parse_tenor_years(match.group(1)))
                rates.append(float(parts[2]))
            except Exception:
                continue
    if not times:
        raise ValueError(f"no {curve_type} inflation quotes found for index '{index}' in {market_data_path}")
    ordered = sorted(zip(times, rates), key=lambda item: item[0])
    return InflationCurve(index=index, curve_type=str(curve_type).upper(), times=tuple(x for x, _ in ordered), rates=tuple(y for _, y in ordered))


def load_zero_inflation_surface_quote(
    market_data_path: str | Path,
    asof_date: str,
    index: str,
    maturity: str,
    strike: float,
    option_type: str,
) -> float | None:
    asof_tokens = {str(asof_date).replace("-", ""), str(asof_date)}
    opt = "C" if str(option_type).strip().lower().startswith("c") else "F"
    strike_txt = f"{float(strike):.3f}".rstrip("0").rstrip(".")
    candidates = {
        f"ZC_INFLATIONCAPFLOOR/PRICE/{index}/{maturity}/{opt}/{strike_txt}",
        f"ZC_INFLATIONCAPFLOOR/PRICE/{index}/{maturity}/{opt}/{float(strike):.2f}",
        f"ZC_INFLATIONCAPFLOOR/PRICE/{index}/{maturity}/{opt}/{float(strike):.4f}",
    }
    with open(market_data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            parts = txt.split()
            if len(parts) < 3 or parts[0] not in asof_tokens:
                continue
            if parts[1] in candidates:
                return float(parts[2])
    return None


def project_index_level(base_cpi: float, curve: InflationCurve, maturity_years: float) -> float:
    return float(base_cpi) * float(curve.growth(maturity_years))


def inflation_swap_payment_times(maturity_years: float, schedule_tenor: str = "1Y") -> tuple[float, ...]:
    step = max(_parse_tenor_years(schedule_tenor), 1.0 / 365.25)
    maturity = max(float(maturity_years), 0.0)
    if maturity <= 0.0:
        return ()
    count = max(int(math.ceil(maturity / step - 1.0e-12)), 1)
    times = [min(i * step, maturity) for i in range(1, count + 1)]
    if abs(times[-1] - maturity) > 1.0e-10:
        times.append(maturity)
    return tuple(float(x) for x in times)


def _forward_discount(discount_curve: Callable[[float], float], valuation_time: float, payment_time: float) -> float:
    if float(payment_time) <= float(valuation_time) + 1.0e-12:
        return 0.0
    p_t = max(float(discount_curve(float(valuation_time))), 1.0e-18)
    p_u = max(float(discount_curve(float(payment_time))), 0.0)
    return p_u / p_t


def price_zero_coupon_cpi_swap(
    notional: float,
    maturity_years: float,
    fixed_rate: float,
    base_cpi: float,
    inflation_curve: InflationCurve,
    discount_curve: Callable[[float], float],
    *,
    receive_inflation: bool = True,
    subtract_inflation_notional: bool = False,
) -> float:
    growth = inflation_curve.growth(maturity_years)
    inflation_payoff = float(notional) * (growth - (1.0 if subtract_inflation_notional else 0.0))
    fixed_payoff = float(notional) * ((1.0 + float(fixed_rate)) ** float(maturity_years) - 1.0)
    payoff = inflation_payoff - fixed_payoff if receive_inflation else fixed_payoff - inflation_payoff
    return float(discount_curve(float(maturity_years))) * payoff


def price_zero_coupon_cpi_swap_at_time(
    notional: float,
    maturity_years: float,
    fixed_rate: float,
    base_cpi: float,
    inflation_curve: InflationCurve,
    discount_curve: Callable[[float], float],
    valuation_time: float,
    *,
    receive_inflation: bool = True,
    subtract_inflation_notional: bool = False,
) -> float:
    if float(valuation_time) >= float(maturity_years) - 1.0e-12:
        return 0.0
    growth = inflation_curve.growth(maturity_years)
    inflation_payoff = float(notional) * (growth - (1.0 if subtract_inflation_notional else 0.0))
    fixed_payoff = float(notional) * ((1.0 + float(fixed_rate)) ** float(maturity_years) - 1.0)
    payoff = inflation_payoff - fixed_payoff if receive_inflation else fixed_payoff - inflation_payoff
    return _forward_discount(discount_curve, float(valuation_time), float(maturity_years)) * payoff


def price_yoy_swap(
    notional: float,
    payment_times: Sequence[float],
    fixed_rate: float,
    inflation_curve: InflationCurve,
    discount_curve: Callable[[float], float],
    *,
    receive_inflation: bool = True,
) -> float:
    pv = 0.0
    prev_t = 0.0
    for t in payment_times:
        tau = max(float(t) - prev_t, 0.0)
        prev_t = float(t)
        yoy = inflation_curve.rate(float(t))
        coupon = (yoy - float(fixed_rate)) * tau * float(notional)
        pv += float(discount_curve(float(t))) * (coupon if receive_inflation else -coupon)
    return pv


def price_yoy_swap_at_time(
    notional: float,
    payment_times: Sequence[float],
    fixed_rate: float,
    inflation_curve: InflationCurve,
    discount_curve: Callable[[float], float],
    valuation_time: float,
    *,
    receive_inflation: bool = True,
) -> float:
    pv = 0.0
    prev_t = 0.0
    val_t = float(valuation_time)
    for t in payment_times:
        pay_t = float(t)
        tau = max(pay_t - prev_t, 0.0)
        yoy = inflation_curve.rate(pay_t)
        coupon = (yoy - float(fixed_rate)) * tau * float(notional)
        if pay_t > val_t + 1.0e-12:
            pv += _forward_discount(discount_curve, val_t, pay_t) * (coupon if receive_inflation else -coupon)
        prev_t = pay_t
    return pv


def price_inflation_capfloor(
    definition: InflationCapFloorDefinition,
    inflation_curve: InflationCurve,
    discount_curve: Callable[[float], float],
    *,
    market_surface_price: float | None = None,
) -> float:
    sign = 1.0 if str(definition.long_short).upper() != "SHORT" else -1.0
    if market_surface_price is not None:
        return sign * float(definition.notional) * float(market_surface_price)
    maturity = float(definition.maturity_years)
    forward = inflation_curve.rate(maturity) if definition.inflation_type.upper() == "YY" else inflation_curve.growth(maturity) - 1.0
    if definition.option_type.lower() == "cap":
        payoff = max(forward - float(definition.strike), 0.0)
    else:
        payoff = max(float(definition.strike) - forward, 0.0)
    return sign * float(definition.notional) * float(discount_curve(maturity)) * payoff


def simulate_inflation_index_paths(
    base_cpi: float,
    curve: InflationCurve,
    times: Sequence[float],
    n_paths: int,
) -> np.ndarray:
    levels = np.asarray([project_index_level(base_cpi, curve, float(t)) for t in times], dtype=float)
    return np.repeat(levels[:, None], int(n_paths), axis=1)


__all__ = [
    "InflationCalibrationInstrument",
    "InflationCapFloorDefinition",
    "InflationCurve",
    "InflationLgmParams",
    "InflationModelSpec",
    "inflation_swap_payment_times",
    "InflationSwapDefinition",
    "InflationSwapLeg",
    "JarrowYildirimParams",
    "load_inflation_curve_from_market_data",
    "load_zero_inflation_surface_quote",
    "parse_inflation_models_from_simulation_xml",
    "price_inflation_capfloor",
    "price_yoy_swap",
    "price_yoy_swap_at_time",
    "price_zero_coupon_cpi_swap",
    "price_zero_coupon_cpi_swap_at_time",
    "project_index_level",
    "simulate_inflation_index_paths",
]
