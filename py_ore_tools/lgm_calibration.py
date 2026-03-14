"""ORE-style external LGM calibration helpers.

This module implements the Python-side orchestration around ORE's IR LGM
calibration workflow:

- ORE-shaped config dataclasses
- ORE-style swaption basket construction with strike fallback handling
- batch / per-currency calibration entry points
- a QuantLib-backed execution backend for the subset that can be reproduced
  cleanly without QuantExt bindings

The calibration backend is intentionally explicit about its support matrix.
When QuantExt / ORE LGM bindings are unavailable, the code uses QuantLib's
``Gsr`` model where the representation is equivalent or practically close
enough to support calibration parity for common ORE setups:

- HullWhite volatility + HullWhite reversion: supported

Other combinations fail fast with a descriptive error instead of silently
producing a different model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, Union

try:
    import QuantLib as ql
except ImportError:  # pragma: no cover - exercised in environments without QuantLib
    ql = None


class LgmCalibrationError(RuntimeError):
    """Raised when the external LGM calibration cannot be completed."""


class CalibrationType(str, Enum):
    NONE = "None"
    BOOTSTRAP = "Bootstrap"
    BEST_FIT = "BestFit"


class ParamType(str, Enum):
    CONSTANT = "Constant"
    PIECEWISE = "Piecewise"


class ReversionType(str, Enum):
    HULL_WHITE = "HullWhite"
    HAGAN = "Hagan"


class VolatilityType(str, Enum):
    HULL_WHITE = "HullWhite"
    HAGAN = "Hagan"


class FloatSpreadMapping(str, Enum):
    NEXT_COUPON = "nextCoupon"
    PRO_RATA = "proRata"
    SIMPLE = "simple"


class FallbackType(str, Enum):
    NO_FALLBACK = "NoFallback"
    FALLBACK_RULE_1 = "FallbackRule1"


_DATE_PATTERNS = ("%Y-%m-%d", "%Y%m%d")
_TENOR_PATTERN = re.compile(r"^\s*(\d+)\s*([YMWD])\s*$", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_TIME_TO_DAYS = 365.25
_MAX_ATM_STDDEV = 3.0


def _require_quantlib() -> None:
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for LGM calibration")


def _as_list_of_floats(values: Iterable[Any], name: str) -> list[float]:
    out = [float(v) for v in values]
    for v in out:
        if not math.isfinite(v):
            raise ValueError(f"{name} contains a non-finite value")
    return out


def _normalise_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {"y", "yes", "true", "1"}:
            return True
        if txt in {"n", "no", "false", "0"}:
            return False
    return bool(value)


def _coerce_enum(enum_cls: type[Enum], value: Any, default: Optional[Enum] = None) -> Enum:
    if value is None:
        if default is None:
            raise ValueError(f"missing value for {enum_cls.__name__}")
        return default
    if isinstance(value, enum_cls):
        return value
    for member in enum_cls:
        if str(value).strip().lower() == member.value.lower():
            return member
    raise ValueError(f"invalid {enum_cls.__name__} value '{value}'")


def _to_quantlib_period(text: str):
    _require_quantlib()
    txt = str(text).strip()
    if not txt:
        raise ValueError("empty tenor/period string")
    return ql.Period(txt)


def _try_parse_quantlib_date(text: str):
    _require_quantlib()
    txt = str(text).strip()
    for fmt in _DATE_PATTERNS:
        try:
            import datetime as _dt

            dt = _dt.datetime.strptime(txt, fmt).date()
            return ql.Date(dt.day, dt.month, dt.year)
        except ValueError:
            continue
    return None


def _parse_date_or_period(text: str):
    date_value = _try_parse_quantlib_date(text)
    if date_value is not None:
        return date_value, None, True
    return None, _to_quantlib_period(text), False


def _parse_period_to_years(text: str) -> float:
    match = _TENOR_PATTERN.match(str(text))
    if match is None:
        raise ValueError(f"unsupported tenor '{text}'")
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "Y":
        return value
    if unit == "M":
        return value / 12.0
    if unit == "W":
        return value / 52.0
    return value / 365.0


def _parse_strike_value(text: str) -> Optional[float]:
    txt = str(text).strip()
    if not txt or txt.upper() in {"ATM", "ATMF"}:
        return None
    if not _NUMBER_PATTERN.match(txt):
        raise ValueError(f"unsupported strike '{text}', expected ATM/ATMF or absolute numeric value")
    return float(txt)


def _time_to_step_date(reference_date, time_value: float):
    _require_quantlib()
    return reference_date + int(round(float(time_value) * _TIME_TO_DAYS))


def _vector_push(vector, values: Iterable[Any]) -> None:
    for value in values:
        vector.push_back(value)


def _build_date_vector(values: Sequence[Any]):
    _require_quantlib()
    out = ql.DateVector()
    _vector_push(out, values)
    return out


def _build_quote_handle_vector(values: Sequence[float]):
    _require_quantlib()
    out = ql.QuoteHandleVector()
    for value in values:
        out.push_back(ql.QuoteHandle(ql.SimpleQuote(float(value))))
    return out


def _build_bool_vector(values: Sequence[bool]):
    _require_quantlib()
    out = ql.BoolVector()
    for value in values:
        out.push_back(bool(value))
    return out


def _build_black_helper_vector(values: Sequence[Any]):
    _require_quantlib()
    out = ql.BlackCalibrationHelperVector()
    for value in values:
        out.push_back(value)
    return out


def _unwrap_handle(value: Any) -> Any:
    current_link = getattr(value, "currentLink", None)
    if callable(current_link):
        return current_link()
    return value


def _option_date_from_tenor(svts, period):
    option_date_from_tenor = getattr(svts, "optionDateFromTenor", None)
    if callable(option_date_from_tenor):
        return option_date_from_tenor(period)
    ref_date = svts.referenceDate() if hasattr(svts, "referenceDate") else ql.Settings.instance().evaluationDate
    return ref_date + period


def _swap_length_from_period(svts, period) -> float:
    swap_length = getattr(svts, "swapLength", None)
    if callable(swap_length):
        return float(swap_length(period))
    return _parse_period_to_years(str(period))


def _swap_length_from_dates(svts, start_date, end_date) -> float:
    swap_length = getattr(svts, "swapLength", None)
    if callable(swap_length):
        return float(swap_length(start_date, end_date))
    return ql.Actual365Fixed().yearFraction(start_date, end_date)


def _surface_vol_and_shift(svts, meta: Mapping[str, Any], strike_value: float | None) -> tuple[float, float]:
    strike = ql.nullDouble() if strike_value is None else strike_value
    def _is_shifted(expiry_arg, term_arg) -> bool:
        try:
            return svts.smileSection(expiry_arg, term_arg).volatilityType() == ql.ShiftedLognormal
        except Exception:
            return False
    if meta["expiry_date_based"] and not meta["term_date_based"]:
        vol = float(svts.volatility(meta["expiry_date"], meta["term_period"], strike))
        shift = float(svts.shift(meta["expiry_date"], meta["term_period"])) if _is_shifted(meta["expiry_date"], meta["term_period"]) else 0.0
        return vol, shift

    expiry_date = meta["expiry_date"] if meta["expiry_date_based"] else _option_date_from_tenor(svts, meta["expiry_period"])
    expiry_time = ql.Actual365Fixed().yearFraction(svts.referenceDate(), expiry_date)
    term_time = float(meta["term_time"])
    vol = float(svts.volatility(expiry_time, term_time, strike))
    shift = float(svts.shift(expiry_time, term_time)) if _is_shifted(expiry_time, term_time) else 0.0
    return vol, shift


@dataclass(frozen=True)
class ParameterBlockConfig:
    calibrate: bool = False
    type: Union[ReversionType, VolatilityType, str] = VolatilityType.HULL_WHITE
    param_type: Union[ParamType, str] = ParamType.CONSTANT
    time_grid: tuple[float, ...] = ()
    initial_values: tuple[float, ...] = (0.0,)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        type_enum: type[Enum],
        default_type: Enum,
        default_initial: Sequence[float],
    ) -> "ParameterBlockConfig":
        if not isinstance(data, Mapping):
            raise TypeError("parameter block config must be a mapping")
        return cls(
            calibrate=_normalise_bool(data.get("calibrate"), False),
            type=_coerce_enum(type_enum, data.get("type"), default_type),
            param_type=_coerce_enum(ParamType, data.get("param_type"), ParamType.CONSTANT),
            time_grid=tuple(_as_list_of_floats(data.get("time_grid", []), "time_grid")),
            initial_values=tuple(
                _as_list_of_floats(data.get("initial_values", default_initial), "initial_values")
            ),
        )


@dataclass(frozen=True)
class ParameterTransformationConfig:
    shift_horizon: float = 0.0
    scaling: float = 1.0

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ParameterTransformationConfig":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("parameter_transformation must be a mapping")
        shift_horizon = float(data.get("shift_horizon", 0.0))
        scaling = float(data.get("scaling", 1.0))
        if shift_horizon < 0.0:
            raise ValueError("shift_horizon must be non-negative")
        if scaling <= 0.0:
            raise ValueError("scaling must be strictly positive")
        return cls(shift_horizon=shift_horizon, scaling=scaling)


@dataclass(frozen=True)
class SwaptionSpec:
    expiry: str
    term: str
    strike: str = "ATM"


@dataclass(frozen=True)
class CurrencyLgmConfig:
    currency: str
    calibration_type: CalibrationType = CalibrationType.NONE
    volatility: ParameterBlockConfig = field(
        default_factory=lambda: ParameterBlockConfig(
            calibrate=False,
            type=VolatilityType.HULL_WHITE,
            param_type=ParamType.CONSTANT,
            time_grid=(),
            initial_values=(0.01,),
        )
    )
    reversion: ParameterBlockConfig = field(
        default_factory=lambda: ParameterBlockConfig(
            calibrate=False,
            type=ReversionType.HULL_WHITE,
            param_type=ParamType.CONSTANT,
            time_grid=(),
            initial_values=(0.03,),
        )
    )
    calibration_swaptions: tuple[SwaptionSpec, ...] = ()
    parameter_transformation: ParameterTransformationConfig = field(default_factory=ParameterTransformationConfig)
    float_spread_mapping: FloatSpreadMapping = FloatSpreadMapping.PRO_RATA
    reference_calibration_grid: str = ""
    bootstrap_tolerance: float = 1.0e-3
    continue_on_error: bool = False

    @classmethod
    def from_dict(cls, currency: str, data: Mapping[str, Any]) -> "CurrencyLgmConfig":
        if not isinstance(data, Mapping):
            raise TypeError("currency config must be a mapping")

        swaptions_raw = data.get("calibration_swaptions", ())
        swaptions: list[SwaptionSpec] = []
        if isinstance(swaptions_raw, Mapping):
            expiries = list(swaptions_raw.get("expiries", []))
            terms = list(swaptions_raw.get("terms", []))
            strikes = list(swaptions_raw.get("strikes", []))
            if not strikes:
                strikes = ["ATM"] * len(expiries)
            if not (len(expiries) == len(terms) == len(strikes)):
                raise ValueError("calibration_swaptions expiries/terms/strikes size mismatch")
            swaptions = [
                SwaptionSpec(str(expiries[i]).strip(), str(terms[i]).strip(), str(strikes[i]).strip())
                for i in range(len(expiries))
            ]
        else:
            for item in swaptions_raw:
                if isinstance(item, SwaptionSpec):
                    swaptions.append(item)
                elif isinstance(item, Mapping):
                    swaptions.append(
                        SwaptionSpec(
                            expiry=str(item["expiry"]).strip(),
                            term=str(item["term"]).strip(),
                            strike=str(item.get("strike", "ATM")).strip(),
                        )
                    )
                else:
                    raise TypeError("calibration_swaptions entries must be mappings or SwaptionSpec instances")

        return cls(
            currency=str(currency).strip(),
            calibration_type=_coerce_enum(CalibrationType, data.get("calibration_type"), CalibrationType.NONE),
            volatility=ParameterBlockConfig.from_dict(
                data.get("volatility", {}),
                type_enum=VolatilityType,
                default_type=VolatilityType.HULL_WHITE,
                default_initial=(0.01,),
            ),
            reversion=ParameterBlockConfig.from_dict(
                data.get("reversion", {}),
                type_enum=ReversionType,
                default_type=ReversionType.HULL_WHITE,
                default_initial=(0.03,),
            ),
            calibration_swaptions=tuple(swaptions),
            parameter_transformation=ParameterTransformationConfig.from_dict(data.get("parameter_transformation")),
            float_spread_mapping=_coerce_enum(
                FloatSpreadMapping, data.get("float_spread_mapping"), FloatSpreadMapping.PRO_RATA
            ),
            reference_calibration_grid=str(data.get("reference_calibration_grid", "") or ""),
            bootstrap_tolerance=float(data.get("bootstrap_tolerance", 1.0e-3)),
            continue_on_error=_normalise_bool(data.get("continue_on_error"), False),
        )


@dataclass(frozen=True)
class LgmCalibrationConfig:
    currencies: tuple[CurrencyLgmConfig, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LgmCalibrationConfig":
        if not isinstance(data, Mapping):
            raise TypeError("batch config must be a mapping")
        currencies = data.get("currencies")
        if isinstance(currencies, Mapping):
            configs = [CurrencyLgmConfig.from_dict(ccy, cfg) for ccy, cfg in currencies.items()]
        elif isinstance(currencies, Sequence):
            configs = []
            for item in currencies:
                if isinstance(item, CurrencyLgmConfig):
                    configs.append(item)
                elif isinstance(item, Mapping):
                    currency = item.get("currency")
                    if not currency:
                        raise ValueError("currency config entry is missing 'currency'")
                    configs.append(CurrencyLgmConfig.from_dict(str(currency), item))
                else:
                    raise TypeError("currencies must contain mappings or CurrencyLgmConfig instances")
        else:
            raise ValueError("batch config must contain a 'currencies' mapping or sequence")
        return cls(tuple(configs))


@dataclass(frozen=True)
class LgmMarketInputs:
    swaption_vol_surface: Any
    swap_index: Any
    short_swap_index: Any
    calibration_discount_curve: Any | None = None
    model_discount_curve: Any | None = None


class MarketProvider(Protocol):
    def get_lgm_market_inputs(self, currency: str, config: CurrencyLgmConfig) -> LgmMarketInputs:
        ...


@dataclass(frozen=True)
class BasketInstrument:
    index: int
    expiry: str
    term: str
    strike_input: str
    strike_used: float | None
    fallback_type: FallbackType
    market_vol: float
    shift: float
    helper: Any
    vol_quote: Any
    expiry_date: Any
    maturity_date: Any
    expiry_time: float
    maturity_time: float
    swap_length: float
    atm_forward: float
    annuity: float
    vega: float
    std_dev: float
    ibor_index_name: str


@dataclass(frozen=True)
class LgmCalibrationPointResult:
    index: int
    expiry: str
    term: str
    strike_input: str
    strike_used: float | None
    fallback_type: FallbackType
    market_vol: float
    model_vol: float | None
    market_value: float
    model_value: float
    calibration_error: float
    expiry_time: float
    swap_length: float
    atm_forward: float
    annuity: float
    vega: float
    std_dev: float


@dataclass(frozen=True)
class CalibratedParameterBlock:
    type: str
    param_type: str
    time_grid: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class LgmCalibrationResult:
    currency: str
    valid: bool
    rmse: float
    calibration_type: str
    volatility: CalibratedParameterBlock
    reversion: CalibratedParameterBlock
    points: tuple[LgmCalibrationPointResult, ...]
    float_spread_mapping: str
    backend: str


def _resolve_market_inputs(
    config: CurrencyLgmConfig,
    market_provider: Union[MarketProvider, Mapping[str, LgmMarketInputs], Callable[[str, CurrencyLgmConfig], LgmMarketInputs], LgmMarketInputs],
) -> LgmMarketInputs:
    if isinstance(market_provider, LgmMarketInputs):
        return market_provider
    if isinstance(market_provider, Mapping):
        return market_provider[config.currency]
    if hasattr(market_provider, "get_lgm_market_inputs"):
        return market_provider.get_lgm_market_inputs(config.currency, config)
    if callable(market_provider):
        return market_provider(config.currency, config)
    raise TypeError("unsupported market_provider type")


def _build_reference_calibration_dates(reference_date, grid_spec: str) -> list[Any]:
    _require_quantlib()
    if not grid_spec.strip():
        return []
    dates = []
    for token in grid_spec.split(","):
        txt = token.strip()
        if not txt:
            continue
        dt = _try_parse_quantlib_date(txt)
        if dt is not None:
            dates.append(dt)
        else:
            dates.append(reference_date + _to_quantlib_period(txt))
    dates.sort()
    return dates


def _get_expiry_and_term(config: CurrencyLgmConfig, instrument: SwaptionSpec, market: LgmMarketInputs):
    _require_quantlib()
    expiry_date, expiry_period, expiry_date_based = _parse_date_or_period(instrument.expiry)
    term_date, term_period, term_date_based = _parse_date_or_period(instrument.term)
    svts = _unwrap_handle(market.swaption_vol_surface)
    swap_index = market.swap_index

    if term_date_based:
        tmp_expiry = expiry_date if expiry_date_based else _option_date_from_tenor(svts, expiry_period)
        tmp_start = swap_index.iborIndex().valueDate(
            swap_index.iborIndex().fixingCalendar().adjust(tmp_expiry)
        )
        min_term_date = tmp_start + ql.Period(1, ql.Months)
        if term_date < min_term_date:
            term_date = min_term_date
        term_time = _swap_length_from_dates(svts, tmp_start, term_date)
    else:
        term_time = _swap_length_from_period(svts, term_period)
        if term_time < 1.0 / 12.0:
            term_time = 1.0 / 12.0
            term_period = ql.Period(1, ql.Months)

    return {
        "expiry_date_based": expiry_date_based,
        "term_date_based": term_date_based,
        "expiry_date": expiry_date,
        "expiry_period": expiry_period,
        "term_date": term_date,
        "term_period": term_period,
        "term_time": term_time,
    }


def _select_swap_index(term_time: float, market: LgmMarketInputs):
    _require_quantlib()
    short_tenor = market.short_swap_index.tenor().length() * (
        1.0 if market.short_swap_index.tenor().units() == ql.Years else 1.0 / 12.0
    )
    use_long = round(term_time) > short_tenor
    index = market.swap_index if use_long else market.short_swap_index
    ibor_index = index.iborIndex()
    fixed_leg_tenor = index.fixedLegTenor()
    fixed_day_counter = index.dayCounter()
    float_day_counter = ibor_index.dayCounter()
    settlement_days = ql.nullInt()
    averaging_method = ql.RateAveraging.Compound
    overnight_cls = getattr(ql, "OvernightIndexedSwapIndex", None)
    if overnight_cls is not None and isinstance(index, overnight_cls):
        settlement_days = index.fixingDays()
        averaging_method = index.averagingMethod()
    return {
        "swap_index": index,
        "ibor_index": ibor_index,
        "fixed_leg_tenor": fixed_leg_tenor,
        "fixed_day_counter": fixed_day_counter,
        "float_day_counter": float_day_counter,
        "settlement_days": settlement_days,
        "averaging_method": averaging_method,
    }


def _helper_stats(helper, market_discount_curve, market_vol: float, shift: float) -> tuple[float, float, float, float]:
    _require_quantlib()
    swap = helper.underlyingSwap()
    swap.setPricingEngine(ql.DiscountingSwapEngine(market_discount_curve))
    atm_forward = float(swap.fairRate())
    annuity = abs(float(swap.fixedLegBPS())) / 1.0e-4
    expiry_date = helper.swaptionExpiryDate()
    expiry_time = float(market_discount_curve.timeFromReference(expiry_date))
    if expiry_time > 0.0:
        std_dev = abs(market_vol) * math.sqrt(expiry_time)
        if helper.volatilityType() == ql.ShiftedLognormal:
            std_dev *= abs(atm_forward + shift)
    else:
        std_dev = 0.0
    eps = 1.0e-4
    vol_up = max(market_vol + eps, 1.0e-8)
    vol_dn = max(market_vol - eps, 1.0e-8)
    vega = (helper.blackPrice(vol_up) - helper.blackPrice(vol_dn)) / (vol_up - vol_dn)
    return atm_forward, annuity, float(vega), float(std_dev)


def _create_swaption_helper(
    meta: Mapping[str, Any],
    strike_value: float | None,
    market_vol: float,
    shift: float,
    market: LgmMarketInputs,
):
    _require_quantlib()
    svts = _unwrap_handle(market.swaption_vol_surface)
    def _helper_vol_type():
        try:
            if meta["expiry_date_based"] and not meta["term_date_based"]:
                return svts.smileSection(meta["expiry_date"], meta["term_period"]).volatilityType()
            expiry_date = meta["expiry_date"] if meta["expiry_date_based"] else _option_date_from_tenor(svts, meta["expiry_period"])
            expiry_time = ql.Actual365Fixed().yearFraction(svts.referenceDate(), expiry_date)
            return svts.smileSection(expiry_time, meta["term_time"]).volatilityType()
        except Exception:
            return ql.Normal
    vol_type = _helper_vol_type()
    quote = ql.SimpleQuote(float(market_vol))
    vol_handle = ql.QuoteHandle(quote)
    selector = _select_swap_index(meta["term_time"], market)

    helper = None
    if meta["expiry_date_based"] and meta["term_date_based"]:
        helper = ql.SwaptionHelper(
            meta["expiry_date"],
            meta["term_date"],
            vol_handle,
            selector["ibor_index"],
            selector["fixed_leg_tenor"],
            selector["fixed_day_counter"],
            selector["float_day_counter"],
            market.calibration_discount_curve,
            ql.BlackCalibrationHelper.RelativePriceError,
            ql.nullDouble() if strike_value is None else strike_value,
            1.0,
            vol_type,
            shift,
            selector["settlement_days"],
            selector["averaging_method"],
        )
    elif meta["expiry_date_based"] and not meta["term_date_based"]:
        helper = ql.SwaptionHelper(
            meta["expiry_date"],
            meta["term_period"],
            vol_handle,
            selector["ibor_index"],
            selector["fixed_leg_tenor"],
            selector["fixed_day_counter"],
            selector["float_day_counter"],
            market.calibration_discount_curve,
            ql.BlackCalibrationHelper.RelativePriceError,
            ql.nullDouble() if strike_value is None else strike_value,
            1.0,
            vol_type,
            shift,
            selector["settlement_days"],
            selector["averaging_method"],
        )
    elif not meta["expiry_date_based"] and meta["term_date_based"]:
        expiry_date = _option_date_from_tenor(svts, meta["expiry_period"])
        helper = ql.SwaptionHelper(
            expiry_date,
            meta["term_date"],
            vol_handle,
            selector["ibor_index"],
            selector["fixed_leg_tenor"],
            selector["fixed_day_counter"],
            selector["float_day_counter"],
            market.calibration_discount_curve,
            ql.BlackCalibrationHelper.RelativePriceError,
            ql.nullDouble() if strike_value is None else strike_value,
            1.0,
            vol_type,
            shift,
            selector["settlement_days"],
            selector["averaging_method"],
        )
    else:
        helper = ql.SwaptionHelper(
            meta["expiry_period"],
            meta["term_period"],
            vol_handle,
            selector["ibor_index"],
            selector["fixed_leg_tenor"],
            selector["fixed_day_counter"],
            selector["float_day_counter"],
            market.calibration_discount_curve,
            ql.BlackCalibrationHelper.RelativePriceError,
            ql.nullDouble() if strike_value is None else strike_value,
            1.0,
            vol_type,
            shift,
            selector["settlement_days"],
            selector["averaging_method"],
        )

    atm_forward, annuity, vega, std_dev = _helper_stats(
        helper, market.calibration_discount_curve, market_vol, shift
    )
    fallback_type = FallbackType.NO_FALLBACK
    updated_strike = strike_value
    expiry_time = float(market.calibration_discount_curve.timeFromReference(helper.swaptionExpiryDate()))
    atm_std_dev = abs(market_vol) * math.sqrt(max(expiry_time, 0.0))
    if helper.volatilityType() == ql.ShiftedLognormal:
        atm_std_dev *= abs(atm_forward + shift)
    if updated_strike is not None and abs(updated_strike - atm_forward) > _MAX_ATM_STDDEV * atm_std_dev:
        if updated_strike > atm_forward:
            updated_strike = atm_forward + _MAX_ATM_STDDEV * atm_std_dev
        else:
            updated_strike = atm_forward - _MAX_ATM_STDDEV * atm_std_dev
        fallback_type = FallbackType.FALLBACK_RULE_1
        return _create_swaption_helper(meta, updated_strike, market_vol, shift, market)[:-1] + (fallback_type,)

    maturity_date = helper.swaptionMaturityDate()
    expiry_time = float(market.calibration_discount_curve.timeFromReference(helper.swaptionExpiryDate()))
    maturity_time = float(market.calibration_discount_curve.timeFromReference(maturity_date))
    swap_length = max(maturity_time - expiry_time, 0.0)
    return (
        helper,
        quote,
        updated_strike,
        atm_forward,
        annuity,
        vega,
        std_dev,
        expiry_time,
        maturity_time,
        swap_length,
        selector["ibor_index"].name(),
        fallback_type,
    )


def build_lgm_swaption_basket(config: CurrencyLgmConfig, market_inputs: LgmMarketInputs) -> tuple[BasketInstrument, ...]:
    """Build an ORE-style calibration basket for one currency."""

    _require_quantlib()
    svts = _unwrap_handle(market_inputs.swaption_vol_surface)
    ref_grid = _build_reference_calibration_dates(
        market_inputs.calibration_discount_curve.referenceDate(), config.reference_calibration_grid
    )
    last_ref_cal_date = ql.Date.minDate()
    basket: list[BasketInstrument] = []

    for idx, spec in enumerate(config.calibration_swaptions):
        meta = _get_expiry_and_term(config, spec, market_inputs)
        strike_value = _parse_strike_value(spec.strike)
        market_vol, shift = _surface_vol_and_shift(svts, meta, strike_value)

        (
            helper,
            quote,
            updated_strike,
            atm_forward,
            annuity,
            vega,
            std_dev,
            expiry_time,
            maturity_time,
            swap_length,
            ibor_index_name,
            fallback_type,
        ) = _create_swaption_helper(meta, strike_value, market_vol, shift, market_inputs)

        if ref_grid:
            expiry_date = helper.swaptionExpiryDate()
            keep = False
            for candidate in ref_grid:
                if candidate >= expiry_date and candidate > last_ref_cal_date:
                    keep = True
                    last_ref_cal_date = candidate
                    break
            if not keep:
                continue

        basket.append(
            BasketInstrument(
                index=idx,
                expiry=spec.expiry,
                term=spec.term,
                strike_input=spec.strike,
                strike_used=updated_strike,
                fallback_type=fallback_type,
                market_vol=market_vol,
                shift=shift,
                helper=helper,
                vol_quote=quote,
                expiry_date=helper.swaptionExpiryDate(),
                maturity_date=helper.swaptionMaturityDate(),
                expiry_time=expiry_time,
                maturity_time=maturity_time,
                swap_length=swap_length,
                atm_forward=atm_forward,
                annuity=annuity,
                vega=vega,
                std_dev=std_dev,
                ibor_index_name=ibor_index_name,
            )
        )

    return tuple(basket)


def _effective_parameter_block(
    block: ParameterBlockConfig,
    *,
    calibrate: bool,
    calibration_type: CalibrationType,
    swaption_expiry_times: Sequence[float],
    swaption_maturity_times: Sequence[float],
) -> tuple[list[float], list[float]]:
    times = list(block.time_grid)
    values = list(block.initial_values)
    if block.param_type == ParamType.CONSTANT:
        if times:
            raise ValueError("constant parameter type expects an empty time grid")
        if len(values) != 1:
            raise ValueError("constant parameter type expects a single initial value")
        return times, values

    if len(values) != len(times) + 1:
        raise ValueError("piecewise parameter block expects len(values) == len(time_grid) + 1")

    if calibrate and calibration_type == CalibrationType.BOOTSTRAP:
        if not values:
            raise ValueError("bootstrap calibration requires at least one initial value")
        return list(swaption_expiry_times), [values[0]] * (len(swaption_expiry_times) + 1)

    return times, values


def _effective_reversion_block(
    block: ParameterBlockConfig,
    *,
    calibrate: bool,
    calibration_type: CalibrationType,
    swaption_maturity_times: Sequence[float],
) -> tuple[list[float], list[float]]:
    times = list(block.time_grid)
    values = list(block.initial_values)
    if block.param_type == ParamType.CONSTANT:
        if times:
            raise ValueError("constant parameter type expects an empty time grid")
        if len(values) != 1:
            raise ValueError("constant parameter type expects a single initial value")
        return times, values

    if len(values) != len(times) + 1:
        raise ValueError("piecewise parameter block expects len(values) == len(time_grid) + 1")

    if calibrate and calibration_type == CalibrationType.BOOTSTRAP:
        if not values:
            raise ValueError("bootstrap calibration requires at least one initial value")
        return list(swaption_maturity_times), [values[0]] * (len(swaption_maturity_times) + 1)

    return times, values


class QuantLibGsrCalibrationBackend:
    """QuantLib GSR-backed execution path for the supported LGM subset."""

    name = "quantlib_gsr"

    def __init__(self, config: CurrencyLgmConfig, market_inputs: LgmMarketInputs, basket: Sequence[BasketInstrument]):
        _require_quantlib()
        self.config = config
        self.market_inputs = market_inputs
        self.basket = list(basket)
        self.optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
        self.end_criteria = ql.EndCriteria(1000, 500, 1.0e-8, 1.0e-8, 1.0e-8)

    def _check_support(self) -> None:
        if self.config.reversion.type == ReversionType.HAGAN:
            raise NotImplementedError("Hagan reversion parametrization requires QuantExt LGM bindings")
        if self.config.volatility.type == VolatilityType.HAGAN:
            raise NotImplementedError(
                "Hagan volatility parametrization requires QuantExt LGM bindings for source-faithful calibration"
            )

    def _build_gsr(self):
        self._check_support()
        expiry_times = [inst.expiry_time for inst in self.basket]
        maturity_times = [inst.maturity_time for inst in self.basket]
        effective_vol_times, effective_vol_values = _effective_parameter_block(
            self.config.volatility,
            calibrate=self.config.volatility.calibrate,
            calibration_type=self.config.calibration_type,
            swaption_expiry_times=expiry_times[:-1] if expiry_times else [],
            swaption_maturity_times=maturity_times,
        )
        effective_rev_times, effective_rev_values = _effective_reversion_block(
            self.config.reversion,
            calibrate=self.config.reversion.calibrate,
            calibration_type=self.config.calibration_type,
            swaption_maturity_times=maturity_times,
        )

        vol_dates = [_time_to_step_date(self.market_inputs.calibration_discount_curve.referenceDate(), t) for t in effective_vol_times]
        rev_dates = [_time_to_step_date(self.market_inputs.calibration_discount_curve.referenceDate(), t) for t in effective_rev_times]

        # Use the last basket maturity as the GSR numeraire time instead of
        # a hardcoded 60Y.  The calibration discount curve may not extend to 60Y
        # (e.g. ~50Y for standard EUR curves), which causes QuantLib to raise
        # "time (60) is past max curve time" and silently disables calibration.
        # Adding a 2Y buffer beyond the last maturity is always sufficient and
        # stays well within the curve range for typical callable bond tenors.
        numeraire_time = (max(maturity_times) + 2.0) if maturity_times else 60.0
        gsr = ql.Gsr(
            self.market_inputs.model_discount_curve,
            _build_date_vector(vol_dates),
            _build_quote_handle_vector(effective_vol_values),
            _build_quote_handle_vector(effective_rev_values),
            numeraire_time,
        )
        engine = ql.Gaussian1dSwaptionEngine(gsr, 64, 7.0, True, False, self.market_inputs.calibration_discount_curve)
        for inst in self.basket:
            inst.helper.setPricingEngine(engine)
            update = getattr(inst.helper, "update", None)
            if callable(update):
                update()
        return gsr, engine, effective_vol_times, effective_vol_values, effective_rev_times, effective_rev_values

    def _precheck_bootstrap_vol(self, model, n_reversions: int) -> None:
        if not (self.config.volatility.calibrate and self.config.calibration_type == CalibrationType.BOOTSTRAP):
            return
        params = ql.Array(model.params())
        original = ql.Array(params)
        for j, inst in enumerate(self.basket):
            market_value = float(inst.helper.marketValue())
            model_value = float(inst.helper.modelValue())
            ratio = model_value / market_value if abs(market_value) > 1.0e-16 else 1.0
            if ratio >= 1.0e-4:
                continue
            sigma_idx = n_reversions + min(j, len(params) - n_reversions - 1)
            tuned = False
            for _ in range(10):
                params[sigma_idx] *= 1.3
                model.setParams(params)
                model.generateArguments()
                market_value = float(inst.helper.marketValue())
                model_value = float(inst.helper.modelValue())
                ratio = model_value / market_value if abs(market_value) > 1.0e-16 else 1.0
                if ratio >= 1.0e-4:
                    tuned = True
                    break
            if not tuned:
                params[sigma_idx] = original[sigma_idx]
                model.setParams(params)
                model.generateArguments()

    def _fix_parameters_mask(self, model, move_sigma_index: Optional[int] = None, move_reversion_index: Optional[int] = None):
        params = list(model.params())
        n_reversions = len(model.reversion())
        n_sigmas = len(model.volatility())
        mask = [True] * len(params)
        if move_sigma_index is not None:
            mask[n_reversions + move_sigma_index] = False
        if move_reversion_index is not None:
            mask[move_reversion_index] = False
        if move_sigma_index is None and move_reversion_index is None:
            return mask
        if move_sigma_index is not None and move_reversion_index is None:
            for i in range(n_reversions, n_reversions + n_sigmas):
                mask[i] = True
            mask[n_reversions + move_sigma_index] = False
        if move_reversion_index is not None and move_sigma_index is None:
            for i in range(n_reversions):
                mask[i] = True
            mask[move_reversion_index] = False
        return mask

    def _run_calibration(self, model) -> None:
        helpers = _build_black_helper_vector([inst.helper for inst in self.basket])
        n_reversions = len(model.reversion())
        no_constraint = ql.NoConstraint()

        self._precheck_bootstrap_vol(model, n_reversions)

        if (
            self.config.volatility.calibrate
            and not self.config.reversion.calibrate
            and self.config.calibration_type == CalibrationType.BOOTSTRAP
        ):
            model.calibrateVolatilitiesIterative(helpers, self.optimization_method, self.end_criteria)
            return

        if (
            self.config.reversion.calibrate
            and not self.config.volatility.calibrate
            and self.config.calibration_type == CalibrationType.BOOTSTRAP
        ):
            for i, inst in enumerate(self.basket):
                vec = _build_black_helper_vector([inst.helper])
                mask = _build_bool_vector(self._fix_parameters_mask(model, move_reversion_index=min(i, len(model.reversion()) - 1)))
                model.calibrate(vec, self.optimization_method, self.end_criteria, no_constraint, ql.DoubleVector(), mask)
            return

        if self.config.calibration_type == CalibrationType.BOOTSTRAP:
            raise LgmCalibrationError(
                "Bootstrap calibration can only be used when either volatility or reversion is fixed"
            )

        if self.config.volatility.calibrate and not self.config.reversion.calibrate:
            mask = [True] * len(model.params())
            for i in range(len(model.reversion()), len(model.params())):
                mask[i] = False
            model.calibrate(helpers, self.optimization_method, self.end_criteria, no_constraint, ql.DoubleVector(), _build_bool_vector(mask))
            return

        if self.config.reversion.calibrate and not self.config.volatility.calibrate:
            mask = [True] * len(model.params())
            for i in range(len(model.reversion())):
                mask[i] = False
            model.calibrate(helpers, self.optimization_method, self.end_criteria, no_constraint, ql.DoubleVector(), _build_bool_vector(mask))
            return

        if self.config.reversion.calibrate and self.config.volatility.calibrate:
            mask = [False] * len(model.params())
            model.calibrate(helpers, self.optimization_method, self.end_criteria, no_constraint, ql.DoubleVector(), _build_bool_vector(mask))

    def _reported_volatility_values(
        self,
        calibrated_sigma: Sequence[float],
        calibrated_reversion: Sequence[float],
    ) -> tuple[float, ...]:
        return tuple(float(v) for v in calibrated_sigma)

    def calibrate(self) -> LgmCalibrationResult:
        if not self.basket:
            raise LgmCalibrationError(f"currency '{self.config.currency}' has an empty calibration basket")

        model, _, effective_vol_times, _, effective_rev_times, _ = self._build_gsr()

        error = float("inf")
        valid = False
        try:
            self._run_calibration(model)
            sq = []
            points = []
            for inst in self.basket:
                market_value = float(inst.helper.marketValue())
                model_value = float(inst.helper.modelValue())
                sq.append((model_value - market_value) ** 2)
                try:
                    model_vol = float(inst.helper.impliedVolatility(model_value, 1.0e-8, 500, 1.0e-8, 5.0))
                except RuntimeError:
                    model_vol = None
                points.append(
                    LgmCalibrationPointResult(
                        index=inst.index,
                        expiry=inst.expiry,
                        term=inst.term,
                        strike_input=inst.strike_input,
                        strike_used=inst.strike_used,
                        fallback_type=inst.fallback_type,
                        market_vol=inst.market_vol,
                        model_vol=model_vol,
                        market_value=market_value,
                        model_value=model_value,
                        calibration_error=float(inst.helper.calibrationError()),
                        expiry_time=inst.expiry_time,
                        swap_length=inst.swap_length,
                        atm_forward=inst.atm_forward,
                        annuity=inst.annuity,
                        vega=inst.vega,
                        std_dev=inst.std_dev,
                    )
                )
            error = math.sqrt(sum(sq) / len(sq))
            valid = (error < self.config.bootstrap_tolerance) or (
                self.config.calibration_type == CalibrationType.BEST_FIT and math.isfinite(error)
            )
            if not valid and not self.config.continue_on_error:
                raise LgmCalibrationError(
                    f"LGM ({self.config.currency}) calibration target function value ({error}) exceeds tolerance ({self.config.bootstrap_tolerance})"
                )
        except Exception:
            if not self.config.continue_on_error:
                raise
            points = []
            for inst in self.basket:
                market_value = float(inst.helper.marketValue())
                model_value = float(inst.helper.modelValue())
                points.append(
                    LgmCalibrationPointResult(
                        index=inst.index,
                        expiry=inst.expiry,
                        term=inst.term,
                        strike_input=inst.strike_input,
                        strike_used=inst.strike_used,
                        fallback_type=inst.fallback_type,
                        market_vol=inst.market_vol,
                        model_vol=None,
                        market_value=market_value,
                        model_value=model_value,
                        calibration_error=float(inst.helper.calibrationError()),
                        expiry_time=inst.expiry_time,
                        swap_length=inst.swap_length,
                        atm_forward=inst.atm_forward,
                        annuity=inst.annuity,
                        vega=inst.vega,
                        std_dev=inst.std_dev,
                    )
                )
            error = float("inf")
            valid = False

        calibrated_sigma = tuple(float(v) for v in model.volatility())
        calibrated_reversion = tuple(float(v) for v in model.reversion())
        return LgmCalibrationResult(
            currency=self.config.currency,
            valid=valid,
            rmse=error,
            calibration_type=self.config.calibration_type.value,
            volatility=CalibratedParameterBlock(
                type=str(self.config.volatility.type.value),
                param_type=str(self.config.volatility.param_type.value),
                time_grid=tuple(float(v) for v in effective_vol_times),
                values=self._reported_volatility_values(calibrated_sigma, calibrated_reversion),
            ),
            reversion=CalibratedParameterBlock(
                type=str(self.config.reversion.type.value),
                param_type=str(self.config.reversion.param_type.value),
                time_grid=tuple(float(v) for v in effective_rev_times),
                values=tuple(float(v) for v in calibrated_reversion),
            ),
            points=tuple(points),
            float_spread_mapping=self.config.float_spread_mapping.value,
            backend=self.name,
        )


def calibrate_lgm_currency(
    currency_config: Union[CurrencyLgmConfig, Mapping[str, Any]],
    market_provider: Union[MarketProvider, Mapping[str, LgmMarketInputs], Callable[[str, CurrencyLgmConfig], LgmMarketInputs], LgmMarketInputs],
) -> LgmCalibrationResult:
    """Calibrate one currency's LGM model externally."""

    if isinstance(currency_config, Mapping):
        currency = currency_config.get("currency")
        if not currency:
            raise ValueError("currency config mapping must include 'currency'")
        config = CurrencyLgmConfig.from_dict(str(currency), currency_config)
    else:
        config = currency_config
    market_inputs = _resolve_market_inputs(config, market_provider)
    if market_inputs.calibration_discount_curve is None:
        market_inputs = LgmMarketInputs(
            swaption_vol_surface=market_inputs.swaption_vol_surface,
            swap_index=market_inputs.swap_index,
            short_swap_index=market_inputs.short_swap_index,
            calibration_discount_curve=market_inputs.swap_index.discountingTermStructure(),
            model_discount_curve=market_inputs.model_discount_curve
            or market_inputs.swap_index.discountingTermStructure(),
        )
    elif market_inputs.model_discount_curve is None:
        market_inputs = LgmMarketInputs(
            swaption_vol_surface=market_inputs.swaption_vol_surface,
            swap_index=market_inputs.swap_index,
            short_swap_index=market_inputs.short_swap_index,
            calibration_discount_curve=market_inputs.calibration_discount_curve,
            model_discount_curve=market_inputs.swap_index.discountingTermStructure(),
        )
    basket = build_lgm_swaption_basket(config, market_inputs)
    backend = QuantLibGsrCalibrationBackend(config, market_inputs, basket)
    return backend.calibrate()


def calibrate_lgm_batch(
    batch_config: Union[LgmCalibrationConfig, Mapping[str, Any], Sequence[CurrencyLgmConfig]],
    market_provider: Union[MarketProvider, Mapping[str, LgmMarketInputs], Callable[[str, CurrencyLgmConfig], LgmMarketInputs]],
) -> dict[str, LgmCalibrationResult]:
    """Calibrate a batch of currencies."""

    if isinstance(batch_config, LgmCalibrationConfig):
        config = batch_config
    elif isinstance(batch_config, Mapping):
        config = LgmCalibrationConfig.from_dict(batch_config)
    else:
        config = LgmCalibrationConfig(tuple(batch_config))
    return {cfg.currency: calibrate_lgm_currency(cfg, market_provider) for cfg in config.currencies}


__all__ = [
    "BasketInstrument",
    "CalibratedParameterBlock",
    "CalibrationType",
    "CurrencyLgmConfig",
    "FallbackType",
    "FloatSpreadMapping",
    "LgmCalibrationConfig",
    "LgmCalibrationError",
    "LgmCalibrationPointResult",
    "LgmCalibrationResult",
    "LgmMarketInputs",
    "MarketProvider",
    "ParamType",
    "ParameterBlockConfig",
    "ParameterTransformationConfig",
    "QuantLibGsrCalibrationBackend",
    "ReversionType",
    "SwaptionSpec",
    "VolatilityType",
    "build_lgm_swaption_basket",
    "calibrate_lgm_batch",
    "calibrate_lgm_currency",
]
