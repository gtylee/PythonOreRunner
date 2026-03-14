"""Python bond pricing helpers for `ore_snapshot_cli`.

This module is intentionally not a generic bond library. It is a narrow port of
the ORE C++ price-only path used by the snapshot CLI when the first trade is a
`Bond`, `ForwardBond`, or `CallableBond`.

Source mapping to ORE C++:
- `ored/portfolio/bond.cpp`
  Mirrors how ORE turns `BondData` into a vanilla `QuantLib::Bond` with
  schedule-driven coupon cashflows and a redemption flow.
- `qle/pricingengines/discountingriskybondengine.cpp`
  Mirrors the risky-bond NPV logic: discounted expected cashflows plus expected
  recovery, with optional treatment of security spread as either a discount
  spread or an extra credit spread.
- `ored/portfolio/forwardbond.cpp`
  Mirrors how ORE reuses the underlying bond definition plus settlement /
  premium fields from `ForwardBondData`.
- `qle/pricingengines/discountingforwardbondengine.cpp`
  Mirrors the forward-bond payoff shape: underlying forward bond value minus
  strike, discounted to the effective settlement date, with optional premium.
- `ored/portfolio/callablebond.cpp`
  Mirrors how ORE merges `CallableBondData` with reference data, reuses the
  underlying bond construction, and builds call / put schedules.
- `qle/pricingengines/numericlgmcallablebondengine.cpp`
  Mirrors the deterministic 1D LGM rollback used for callable bond pricing,
  including American exercise expansion and clean/dirty call-price handling.

The implementation here is deliberately pragmatic:
- it focuses on price-only parity for `Bond` / `ForwardBond`
- it uses existing PythonOreRunner utilities where that is already faithful
- when native ORE curve outputs are unavailable, it falls back to either
  flow-implied discount factors or a fitted market curve rather than requiring
  ORE-SWIG at runtime
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional in some environments
    torch = None

try:
    import QuantLib as ql
except Exception:  # pragma: no cover - QuantLib is optional in some environments
    ql = None

from py_ore_tools.irs_xva_utils import (
    _parse_yyyymmdd,
    load_ore_default_curve_inputs,
    parse_lgm_params_from_simulation_xml,
    survival_probability_from_hazard,
)
from py_ore_tools.lgm import LGM1F, LGMParams
from py_ore_tools.lgm_calibration import (
    CalibrationType,
    CurrencyLgmConfig,
    LgmMarketInputs,
    ParamType,
    ReversionType,
    SwaptionSpec,
    VolatilityType,
    calibrate_lgm_currency,
)
from py_ore_tools.lgm_ir_options import _convolution_nodes_and_weights, _convolution_rollback
from py_ore_tools.ore_snapshot import (
    _resolve_ore_path,
    build_discount_curve_from_discount_pairs,
    fit_discount_curves_from_ore_market,
)


def _parse_any_date(text: str) -> date:
    txt = (text or "").strip()
    if not txt:
        raise ValueError("empty date string")
    if "." in txt:
        return datetime.strptime(txt, "%d.%m.%Y").date()
    return _parse_yyyymmdd(txt)


def _parse_bool(text: str | None, default: bool = False) -> bool:
    if text is None or str(text).strip() == "":
        return default
    return str(text).strip().lower() in {"true", "t", "yes", "y", "1"}


def _norm_dc(name: str) -> str:
    return (name or "A365F").strip().upper().replace("ACTUAL/", "ACT/").replace(" ", "")


def _time_from_dates(asof: date, d: date, day_counter: str) -> float:
    dc = _norm_dc(day_counter)
    if dc in {"A365", "A365F", "ACT/365", "ACT/365F"}:
        return (d - asof).days / 365.0
    if dc in {"A360", "ACT/360"}:
        return (d - asof).days / 360.0
    return (d - asof).days / 365.0


def _year_fraction(start: date, end: date, day_counter: str) -> float:
    dc = _norm_dc(day_counter)
    if dc in {"A360", "ACT/360"}:
        return (end - start).days / 360.0
    if dc in {"30/360", "30E/360"}:
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 == 30 else end.day
        return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0
    return (end - start).days / 365.0


def _is_target_holiday(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    if (d.month, d.day) in {(1, 1), (5, 1), (12, 25), (12, 26)}:
        return True
    y = d.year
    a = y % 19
    b = y // 100
    c = y % 100
    d0 = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d0 - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = date(y, month, day)
    return d in {easter - timedelta(days=2), easter + timedelta(days=1)}


def _is_business_day(d: date, calendar: str) -> bool:
    if d.weekday() >= 5:
        return False
    tokens = [x.strip().upper() for x in (calendar or "TARGET").split(",") if x.strip()]
    if not tokens:
        tokens = ["TARGET"]
    for token in tokens:
        if token == "TARGET" and _is_target_holiday(d):
            return False
    return True


def _adjust_date(d: date, convention: str, calendar: str) -> date:
    c = (convention or "F").strip().upper()
    if c in {"U", "UNADJUSTED"}:
        return d
    if _is_business_day(d, calendar):
        return d
    if c in {"F", "FOLLOWING", "MF", "MODIFIEDFOLLOWING"}:
        x = d
        while not _is_business_day(x, calendar):
            x += timedelta(days=1)
        if c in {"MF", "MODIFIEDFOLLOWING"} and x.month != d.month:
            x = d
            while not _is_business_day(x, calendar):
                x -= timedelta(days=1)
        return x
    x = d
    while not _is_business_day(x, calendar):
        x -= timedelta(days=1)
    return x


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    mdays = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(y, m, min(d.day, mdays[m - 1]))


def _parse_tenor_months(tenor: str) -> int:
    m = re.match(r"^\s*(\d+)\s*([MY])\s*$", (tenor or "").upper())
    if m is None:
        raise ValueError(f"unsupported tenor '{tenor}'")
    n = int(m.group(1))
    return n * 12 if m.group(2) == "Y" else n


def _build_schedule(start: date, end: date, tenor: str, calendar: str, convention: str, term_convention: str, rule: str) -> list[date]:
    rule_norm = (rule or "Forward").strip().lower()
    if rule_norm == "zero":
        return [_adjust_date(start, convention, calendar), _adjust_date(end, term_convention or convention, calendar)]
    months = _parse_tenor_months(tenor)
    raw = [start]
    if rule_norm == "backward":
        tmp = [end]
        x = end
        while x > start:
            x = _add_months(x, -months)
            if x < start:
                x = start
            tmp.append(x)
        raw = list(reversed(tmp))
    else:
        x = start
        while x < end:
            x = _add_months(x, months)
            if x > end:
                x = end
            raw.append(x)
    out = [_adjust_date(raw[0], convention, calendar)]
    for d in raw[1:-1]:
        out.append(_adjust_date(d, convention, calendar))
    out.append(_adjust_date(raw[-1], term_convention or convention, calendar))
    return out


@dataclass(frozen=True)
class BondCashflow:
    pay_date: date
    amount: float
    flow_type: str
    accrual_start: date | None = None
    accrual_end: date | None = None
    nominal: float | None = None


@dataclass(frozen=True)
class BondEngineSpec:
    timestep_months: int = 3
    spread_on_income_curve: bool = True
    treat_security_spread_as_credit_spread: bool = False
    include_past_cashflows: bool = False


@dataclass(frozen=True)
class BondTradeSpec:
    trade_id: str
    trade_type: str
    currency: str
    payer: bool
    security_id: str
    credit_curve_id: str
    reference_curve_id: str
    income_curve_id: str
    settlement_days: int
    calendar: str
    issue_date: date
    bond_notional: float
    cashflows: tuple[BondCashflow, ...]
    forward_maturity_date: date | None = None
    forward_settlement_date: date | None = None
    forward_amount: float | None = None
    long_in_forward: bool = True
    settlement_dirty: bool = True
    compensation_payment: float = 0.0
    compensation_payment_date: date | None = None


@dataclass(frozen=True)
class CallableExerciseSpec:
    exercise_date: date
    exercise_type: str
    price: float
    price_type: str
    include_accrual: bool


@dataclass(frozen=True)
class CallableBondEngineSpec:
    model_family: str = "LGM"
    engine_variant: str = "Grid"
    spread_on_income_curve: bool = True
    exercise_time_steps_per_year: int = 24
    grid_sy: float = 3.0
    grid_ny: int = 10
    grid_sx: float = 6.0
    grid_nx: int = 30
    fd_max_time: float = 50.0
    fd_state_grid_points: int = 121
    fd_time_steps_per_year: int = 24
    fd_mesher_epsilon: float = 1.0e-4
    fd_scheme: str = "Douglas"
    generate_additional_results: bool = True


@dataclass(frozen=True)
class CallableBondTradeSpec:
    trade_id: str
    currency: str
    security_id: str
    credit_curve_id: str
    reference_curve_id: str
    income_curve_id: str
    settlement_days: int
    calendar: str
    issue_date: date
    bond_notional: float
    bond: BondTradeSpec
    call_data: tuple[CallableExerciseSpec, ...]
    put_data: tuple[CallableExerciseSpec, ...]


@dataclass(frozen=True)
class CompiledBondTrade:
    """Trade-time representation for scenario pricing.

    This is the vectorization boundary. XML parsing and schedule reconstruction
    happen once into dense arrays; scenario evaluation then becomes pure array
    algebra over `[n_scenarios, n_flows]` style tensors.
    """

    trade_id: str
    trade_type: str
    currency: str
    security_id: str
    credit_curve_id: str
    reference_curve_id: str
    income_curve_id: str
    bond_notional: float

    pay_times: np.ndarray
    amounts: np.ndarray
    is_coupon: np.ndarray
    recovery_nominals: np.ndarray
    recovery_start_times: np.ndarray
    recovery_end_times: np.ndarray

    maturity_time: float

    forward_maturity_time: float | None = None
    forward_settlement_time: float | None = None
    forward_amount: float | None = None
    settlement_dirty: bool = True
    long_in_forward: bool = True
    compensation_payment: float = 0.0
    compensation_payment_time: float | None = None

    engine_spec: BondEngineSpec = BondEngineSpec()


@dataclass(frozen=True)
class BondScenarioGrid:
    """Scenario inputs already sampled on the compiled trade times.

    The intended use is:
    - build trade-specific times once
    - sample scenario curves / hazard inputs onto those times outside the hot loop
    - feed the sampled arrays into NumPy or Torch pricing kernels
    """

    discount_to_pay: np.ndarray
    income_to_npv: np.ndarray
    income_to_settlement: np.ndarray
    survival_to_pay: np.ndarray

    recovery_discount_mid: np.ndarray
    recovery_default_prob: np.ndarray
    recovery_rate: np.ndarray

    forward_dirty_value: np.ndarray | None = None
    accrued_at_bond_settlement: np.ndarray | None = None
    payoff_discount: np.ndarray | None = None
    premium_discount: np.ndarray | None = None

    def n_scenarios(self) -> int:
        return int(np.asarray(self.income_to_npv).shape[0])

    def as_single(self) -> "BondScenarioGrid":
        """Return a shape-safe single-scenario view.

        This is the adapter for legacy scalar pricing. Single pricing is just
        the vectorized pricing path with `n_scenarios = 1`.
        """

        return BondScenarioGrid(
            discount_to_pay=np.asarray(self.discount_to_pay, dtype=float).reshape(1, -1),
            income_to_npv=np.asarray(self.income_to_npv, dtype=float).reshape(1),
            income_to_settlement=np.asarray(self.income_to_settlement, dtype=float).reshape(1),
            survival_to_pay=np.asarray(self.survival_to_pay, dtype=float).reshape(1, -1),
            recovery_discount_mid=np.asarray(self.recovery_discount_mid, dtype=float).reshape(1, -1),
            recovery_default_prob=np.asarray(self.recovery_default_prob, dtype=float).reshape(1, -1),
            recovery_rate=np.asarray(self.recovery_rate, dtype=float).reshape(1),
            forward_dirty_value=None if self.forward_dirty_value is None else np.asarray(self.forward_dirty_value, dtype=float).reshape(1),
            accrued_at_bond_settlement=None if self.accrued_at_bond_settlement is None else np.asarray(self.accrued_at_bond_settlement, dtype=float).reshape(1),
            payoff_discount=None if self.payoff_discount is None else np.asarray(self.payoff_discount, dtype=float).reshape(1),
            premium_discount=None if self.premium_discount is None else np.asarray(self.premium_discount, dtype=float).reshape(1),
        )


def compile_bond_trade(
    spec: BondTradeSpec,
    *,
    asof_date: str | date,
    day_counter: str,
    engine_spec: BondEngineSpec | None = None,
) -> CompiledBondTrade:
    """Compile a parsed trade into dense arrays for scenario evaluation."""

    asof = asof_date if isinstance(asof_date, date) else _parse_any_date(str(asof_date))
    live_cashflows = [cf for cf in spec.cashflows if not _cashflow_has_occurred(cf.pay_date, asof, False)]
    coupon_rows = [cf for cf in live_cashflows if cf.accrual_start is not None and cf.accrual_end is not None and cf.nominal is not None]
    pay_times = np.asarray([_time_from_dates(asof, cf.pay_date, day_counter) for cf in live_cashflows], dtype=float)
    amounts = np.asarray([float(cf.amount) for cf in live_cashflows], dtype=float)
    is_coupon = np.asarray([cf.accrual_start is not None and cf.accrual_end is not None and cf.nominal is not None for cf in live_cashflows], dtype=bool)
    recovery_nominals = np.asarray([float(cf.nominal) for cf in coupon_rows], dtype=float) if coupon_rows else np.zeros(0, dtype=float)
    recovery_start_times = (
        np.asarray([_time_from_dates(asof, cf.accrual_start, day_counter) for cf in coupon_rows], dtype=float)
        if coupon_rows
        else np.zeros(0, dtype=float)
    )
    recovery_end_times = (
        np.asarray([_time_from_dates(asof, cf.accrual_end, day_counter) for cf in coupon_rows], dtype=float)
        if coupon_rows
        else np.zeros(0, dtype=float)
    )
    maturity_time = float(np.max(pay_times)) if pay_times.size else 0.0
    return CompiledBondTrade(
        trade_id=spec.trade_id,
        trade_type=spec.trade_type,
        currency=spec.currency,
        security_id=spec.security_id,
        credit_curve_id=spec.credit_curve_id,
        reference_curve_id=spec.reference_curve_id,
        income_curve_id=spec.income_curve_id,
        bond_notional=float(spec.bond_notional),
        pay_times=pay_times,
        amounts=amounts,
        is_coupon=is_coupon,
        recovery_nominals=recovery_nominals,
        recovery_start_times=recovery_start_times,
        recovery_end_times=recovery_end_times,
        maturity_time=maturity_time,
        forward_maturity_time=None if spec.forward_maturity_date is None else _time_from_dates(asof, spec.forward_maturity_date, day_counter),
        forward_settlement_time=None if spec.forward_settlement_date is None else _time_from_dates(asof, spec.forward_settlement_date, day_counter),
        forward_amount=None if spec.forward_amount is None else float(spec.forward_amount),
        settlement_dirty=bool(spec.settlement_dirty),
        long_in_forward=bool(spec.long_in_forward),
        compensation_payment=float(spec.compensation_payment),
        compensation_payment_time=None if spec.compensation_payment_date is None else _time_from_dates(asof, spec.compensation_payment_date, day_counter),
        engine_spec=engine_spec or BondEngineSpec(),
    )


def _ensure_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def _ensure_1d_float(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def validate_bond_scenario_grid(compiled: CompiledBondTrade, grid: BondScenarioGrid) -> None:
    discount_to_pay = _ensure_2d_float("discount_to_pay", grid.discount_to_pay)
    survival_to_pay = _ensure_2d_float("survival_to_pay", grid.survival_to_pay)
    income_to_npv = _ensure_1d_float("income_to_npv", grid.income_to_npv)
    income_to_settlement = _ensure_1d_float("income_to_settlement", grid.income_to_settlement)
    recovery_discount_mid = _ensure_2d_float("recovery_discount_mid", grid.recovery_discount_mid)
    recovery_default_prob = _ensure_2d_float("recovery_default_prob", grid.recovery_default_prob)
    recovery_rate = _ensure_1d_float("recovery_rate", grid.recovery_rate)
    n_scen = income_to_npv.shape[0]
    if discount_to_pay.shape != (n_scen, compiled.pay_times.size):
        raise ValueError(f"discount_to_pay expected shape {(n_scen, compiled.pay_times.size)}, got {discount_to_pay.shape}")
    if survival_to_pay.shape != (n_scen, compiled.pay_times.size):
        raise ValueError(f"survival_to_pay expected shape {(n_scen, compiled.pay_times.size)}, got {survival_to_pay.shape}")
    if income_to_settlement.shape[0] != n_scen or recovery_rate.shape[0] != n_scen:
        raise ValueError("scenario scalars must share the same scenario count")
    if recovery_discount_mid.shape != (n_scen, compiled.recovery_nominals.size):
        raise ValueError(f"recovery_discount_mid expected shape {(n_scen, compiled.recovery_nominals.size)}, got {recovery_discount_mid.shape}")
    if recovery_default_prob.shape != (n_scen, compiled.recovery_nominals.size):
        raise ValueError(f"recovery_default_prob expected shape {(n_scen, compiled.recovery_nominals.size)}, got {recovery_default_prob.shape}")
    if compiled.trade_type == "ForwardBond":
        for name in ("forward_dirty_value", "accrued_at_bond_settlement", "payoff_discount"):
            value = getattr(grid, name)
            if value is None:
                raise ValueError(f"{name} is required for ForwardBond scenario pricing")
            arr = _ensure_1d_float(name, value)
            if arr.shape[0] != n_scen:
                raise ValueError(f"{name} must have length {n_scen}, got {arr.shape[0]}")
        if (
            compiled.compensation_payment
            and compiled.compensation_payment_time is not None
            and compiled.compensation_payment_time > 0.0
            and grid.premium_discount is None
        ):
            raise ValueError("premium_discount is required when compensation_payment_time is set")


def build_bond_scenario_grid_numpy(
    compiled: CompiledBondTrade,
    *,
    discount_curve,
    income_curve,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
    recovery_rate: float,
    security_spread: float = 0.0,
    engine_spec: BondEngineSpec | None = None,
    npv_time: float = 0.0,
    settlement_time: float = 0.0,
    conditional_on_survival: bool = False,
    forward_dirty_value: float | None = None,
    accrued_at_bond_settlement: float | None = None,
    payoff_discount: float | None = None,
    premium_discount: float | None = None,
) -> BondScenarioGrid:
    """Convenience builder for the legacy single-scenario case.

    This samples scalar curve/hazard inputs onto a compiled trade and returns a
    one-scenario grid that can be fed directly into the vectorized NumPy kernel.
    """

    engine_spec = engine_spec or compiled.engine_spec
    extra_credit = (
        security_spread / max(1.0 - float(recovery_rate), 1.0e-12)
        if engine_spec.treat_security_spread_as_credit_spread
        else 0.0
    )
    eff_discount = discount_curve if engine_spec.treat_security_spread_as_credit_spread else _apply_spread(discount_curve, security_spread)
    eff_income = income_curve
    if security_spread and not engine_spec.treat_security_spread_as_credit_spread and engine_spec.spread_on_income_curve:
        eff_income = _apply_spread(income_curve, security_spread)
    df_npv = float(eff_income(max(npv_time, 0.0)))
    df_settlement = float(eff_income(max(settlement_time, 0.0)))
    sp_npv = 1.0 if not conditional_on_survival else float(
        _survival_from_piecewise(np.asarray([max(npv_time, 0.0)]), hazard_times, hazard_rates, extra_credit)[0]
    )
    sp_settlement = float(_survival_from_piecewise(np.asarray([max(settlement_time, 0.0)]), hazard_times, hazard_rates, extra_credit)[0]) / (
        1.0 if not conditional_on_survival else max(sp_npv, 1.0e-18)
    )
    discount_to_pay = np.asarray([float(eff_discount(max(t, 0.0))) / max(df_npv, 1.0e-18) for t in compiled.pay_times], dtype=float)[None, :]
    survival_to_pay = np.asarray(
        [
            float(_survival_from_piecewise(np.asarray([max(t, 0.0)]), hazard_times, hazard_rates, extra_credit)[0]) / max(sp_npv, 1.0e-18)
            for t in compiled.pay_times
        ],
        dtype=float,
    )[None, :]
    recovery_discount_mid = np.zeros((1, compiled.recovery_nominals.size), dtype=float)
    recovery_default_prob = np.zeros((1, compiled.recovery_nominals.size), dtype=float)
    for i, (s, e) in enumerate(zip(compiled.recovery_start_times, compiled.recovery_end_times)):
        s_eff = max(float(s), 0.0)
        e_eff = max(float(e), 0.0)
        if e_eff <= s_eff:
            continue
        default_prob = _default_probability(s_eff, e_eff, hazard_times, hazard_rates, extra_credit) / max(sp_npv, 1.0e-18)
        mid_t = 0.5 * (s_eff + e_eff)
        recovery_default_prob[0, i] = default_prob
        recovery_discount_mid[0, i] = float(eff_discount(max(mid_t, 0.0))) / max(df_npv, 1.0e-18)
    if payoff_discount is None:
        payoff_discount = float(eff_discount(max(settlement_time, 0.0)))
        # For the vectorized forward path we supply the already-discounted payoff
        # factor used in the C++ `DiscountingForwardBondEngine`, not the bond
        # engine's settlement compounding factor.
    return BondScenarioGrid(
        discount_to_pay=discount_to_pay,
        income_to_npv=np.asarray([df_npv], dtype=float),
        income_to_settlement=np.asarray([df_settlement], dtype=float),
        survival_to_pay=survival_to_pay,
        recovery_discount_mid=recovery_discount_mid,
        recovery_default_prob=recovery_default_prob,
        recovery_rate=np.asarray([float(recovery_rate)], dtype=float),
        forward_dirty_value=None if forward_dirty_value is None else np.asarray([float(forward_dirty_value)], dtype=float),
        accrued_at_bond_settlement=None if accrued_at_bond_settlement is None else np.asarray([float(accrued_at_bond_settlement)], dtype=float),
        payoff_discount=None if payoff_discount is None else np.asarray([float(payoff_discount)], dtype=float),
        premium_discount=None if premium_discount is None else np.asarray([float(premium_discount)], dtype=float),
    )


def price_bond_scenarios_numpy(compiled: CompiledBondTrade, grid: BondScenarioGrid) -> np.ndarray:
    """Vectorized NumPy evaluator.

    This is the canonical scenario pricing kernel. Single-scenario pricing
    should use this same function with `n_scenarios=1`.
    """

    validate_bond_scenario_grid(compiled, grid)
    cash_pv = (np.asarray(grid.discount_to_pay, dtype=float) * np.asarray(grid.survival_to_pay, dtype=float)) * compiled.amounts[None, :]
    pv = np.sum(cash_pv, axis=1)
    if compiled.recovery_nominals.size:
        pv += np.sum(
            np.asarray(grid.recovery_discount_mid, dtype=float)
            * np.asarray(grid.recovery_default_prob, dtype=float)
            * compiled.recovery_nominals[None, :]
            * np.asarray(grid.recovery_rate, dtype=float)[:, None],
            axis=1,
        )
    if compiled.trade_type == "ForwardBond":
        forward_dirty_value = np.asarray(grid.forward_dirty_value, dtype=float)
        accrued = np.asarray(grid.accrued_at_bond_settlement, dtype=float)
        strike = float(compiled.forward_amount or 0.0) + np.where(compiled.settlement_dirty, 0.0, accrued)
        raw = np.where(compiled.long_in_forward, forward_dirty_value - strike, strike - forward_dirty_value)
        pv = raw * np.asarray(grid.payoff_discount, dtype=float)
        if (
            compiled.compensation_payment_time is not None
            and compiled.compensation_payment_time > 0.0
            and compiled.compensation_payment
        ):
            prem = float(compiled.compensation_payment) * np.asarray(grid.premium_discount, dtype=float)
            pv = pv + np.where(compiled.long_in_forward, -prem, prem)
    return np.asarray(pv, dtype=float)


def price_bond_scenarios_torch(compiled: CompiledBondTrade, grid: BondScenarioGrid, *, device: str = "cpu"):
    """Vectorized Torch evaluator mirroring the NumPy kernel."""

    if torch is None:
        raise ImportError("torch is required for price_bond_scenarios_torch()")
    validate_bond_scenario_grid(compiled, grid)
    target = torch.device(device)
    dtype = torch.float32 if target.type == "mps" else torch.float64
    amounts = torch.as_tensor(compiled.amounts, dtype=dtype, device=target)
    recovery_nominals = torch.as_tensor(compiled.recovery_nominals, dtype=dtype, device=target)
    discount_to_pay = torch.as_tensor(grid.discount_to_pay, dtype=dtype, device=target)
    survival_to_pay = torch.as_tensor(grid.survival_to_pay, dtype=dtype, device=target)
    pv = torch.sum(discount_to_pay * survival_to_pay * amounts.unsqueeze(0), dim=1)
    if compiled.recovery_nominals.size:
        recovery_discount_mid = torch.as_tensor(grid.recovery_discount_mid, dtype=dtype, device=target)
        recovery_default_prob = torch.as_tensor(grid.recovery_default_prob, dtype=dtype, device=target)
        recovery_rate = torch.as_tensor(grid.recovery_rate, dtype=dtype, device=target)
        pv = pv + torch.sum(
            recovery_discount_mid * recovery_default_prob * recovery_nominals.unsqueeze(0) * recovery_rate.unsqueeze(1),
            dim=1,
        )
    if compiled.trade_type == "ForwardBond":
        forward_dirty_value = torch.as_tensor(grid.forward_dirty_value, dtype=dtype, device=target)
        accrued = torch.as_tensor(grid.accrued_at_bond_settlement, dtype=dtype, device=target)
        strike = torch.full_like(forward_dirty_value, float(compiled.forward_amount or 0.0))
        if not compiled.settlement_dirty:
            strike = strike + accrued
        raw = forward_dirty_value - strike if compiled.long_in_forward else strike - forward_dirty_value
        pv = raw * torch.as_tensor(grid.payoff_discount, dtype=dtype, device=target)
        if (
            compiled.compensation_payment_time is not None
            and compiled.compensation_payment_time > 0.0
            and compiled.compensation_payment
        ):
            prem = float(compiled.compensation_payment) * torch.as_tensor(grid.premium_discount, dtype=dtype, device=target)
            pv = pv + (-prem if compiled.long_in_forward else prem)
    return pv


def price_bond_single_numpy(compiled: CompiledBondTrade, grid: BondScenarioGrid) -> float:
    """Single-scenario special case of the vectorized NumPy kernel."""

    return float(price_bond_scenarios_numpy(compiled, grid.as_single())[0])


def _resolve_trade(root: ET.Element, trade_id: str) -> ET.Element:
    for trade in root.findall("./Trade"):
        if (trade.attrib.get("id", "") or "").strip() == trade_id:
            return trade
    raise ValueError(f"trade '{trade_id}' not found in portfolio")


def _scale_cashflows(cashflows: tuple[BondCashflow, ...], scale: float) -> tuple[BondCashflow, ...]:
    if abs(float(scale) - 1.0) <= 1.0e-16:
        return cashflows
    out: list[BondCashflow] = []
    for cf in cashflows:
        out.append(
            BondCashflow(
                pay_date=cf.pay_date,
                amount=float(cf.amount) * float(scale),
                flow_type=cf.flow_type,
                accrual_start=cf.accrual_start,
                accrual_end=cf.accrual_end,
                nominal=None if cf.nominal is None else float(cf.nominal) * float(scale),
            )
        )
    return tuple(out)


def _merge_reference_data(bond_node: ET.Element, reference_data_path: Path | None) -> ET.Element:
    # ORE's `Bond::build()` first populates the inline `BondData` from reference
    # data when a security id is present. We mimic the minimal part needed for
    # the snapshot CLI: fill missing top-level bond fields and `LegData` from
    # `ReferenceDatum[id]/BondReferenceData`, but leave inline overrides intact.
    merged = ET.fromstring(ET.tostring(bond_node, encoding="unicode"))
    security_id = (merged.findtext("./SecurityId") or "").strip()
    if not security_id or reference_data_path is None or not reference_data_path.exists():
        return merged
    rd_root = ET.parse(reference_data_path).getroot()
    ref_node = rd_root.find(f"./ReferenceDatum[@id='{security_id}']/BondReferenceData")
    if ref_node is None:
        return merged
    for child in ref_node:
        if merged.find(child.tag) is None:
            merged.append(ET.fromstring(ET.tostring(child, encoding="unicode")))
        elif child.tag == "LegData" and merged.find("./LegData") is None:
            merged.append(ET.fromstring(ET.tostring(child, encoding="unicode")))
    return merged


def _merge_callable_reference_data(callable_node: ET.Element, reference_data_path: Path | None) -> ET.Element:
    merged = ET.fromstring(ET.tostring(callable_node, encoding="unicode"))
    security_id = (merged.findtext("./BondData/SecurityId") or "").strip()
    if not security_id or reference_data_path is None or not reference_data_path.exists():
        return merged
    rd_root = ET.parse(reference_data_path).getroot()
    ref_node = rd_root.find(f"./ReferenceDatum[@id='{security_id}']/CallableBondReferenceData")
    if ref_node is None:
        return merged
    bond_node = merged.find("./BondData")
    ref_bond = ref_node.find("./BondData")
    if bond_node is None and ref_bond is not None:
        merged.insert(0, ET.fromstring(ET.tostring(ref_bond, encoding="unicode")))
    elif bond_node is not None and ref_bond is not None:
        bond_merged = _merge_reference_data(bond_node, None)
        for child in ref_bond:
            if bond_merged.find(child.tag) is None:
                bond_merged.append(ET.fromstring(ET.tostring(child, encoding="unicode")))
            elif child.tag == "LegData" and bond_merged.find("./LegData") is None:
                bond_merged.append(ET.fromstring(ET.tostring(child, encoding="unicode")))
        merged.remove(bond_node)
        merged.insert(0, bond_merged)
    for tag in ("CallData", "PutData"):
        if merged.find(f"./{tag}") is None and ref_node.find(f"./{tag}") is not None:
            merged.append(ET.fromstring(ET.tostring(ref_node.find(f"./{tag}"), encoding="unicode")))
    return merged


def _parse_scheduled_block(node: ET.Element | None, container_tag: str, item_tag: str, parser) -> tuple[list[object], list[date | None]]:
    if node is None:
        return [], []
    parent = node.find(f"./{container_tag}")
    if parent is None:
        return [], []
    items = parent.findall(f"./{item_tag}")
    values: list[object] = []
    starts: list[date | None] = []
    for item in items:
        txt = (item.text or "").strip()
        if not txt:
            continue
        values.append(parser(txt))
        start_txt = (item.attrib.get("startDate", "") or "").strip()
        starts.append(_parse_any_date(start_txt) if start_txt else None)
    if not values and (parent.text or "").strip():
        values = [parser((parent.text or "").strip())]
        starts = [None]
    return values, starts


def _parse_call_schedule_dates(node: ET.Element) -> list[date]:
    dates_parent = node.find("./ScheduleData/Dates/Dates")
    if dates_parent is not None:
        dates = [_parse_any_date((n.text or "").strip()) for n in dates_parent.findall("./Date") if (n.text or "").strip()]
        if dates:
            return dates
    rules = node.find("./ScheduleData/Rules")
    if rules is None:
        raise ValueError("callable bond call/put data is missing schedule dates")
    start = _parse_any_date(rules.findtext("./StartDate") or "")
    end = _parse_any_date(rules.findtext("./EndDate") or "")
    tenor = (rules.findtext("./Tenor") or "").strip()
    calendar = (rules.findtext("./Calendar") or "TARGET").strip()
    convention = (rules.findtext("./Convention") or "F").strip()
    term_convention = (rules.findtext("./TermConvention") or convention).strip()
    rule = (rules.findtext("./Rule") or "Forward").strip()
    return _build_schedule(start, end, tenor, calendar, convention, term_convention, rule)


def _expand_scheduled_values(values: list[object], starts: list[date | None], schedule_dates: list[date], default):
    if not schedule_dates:
        return []
    if not values:
        return [default for _ in schedule_dates]
    if starts and any(s is not None for s in starts):
        out: list[object] = []
        schedule_plus = schedule_dates + [date.max]
        normalized_starts = [date.min if s is None else s for s in starts]
        for d in schedule_plus[:-1]:
            idx = 0
            for i, s in enumerate(normalized_starts):
                if s <= d:
                    idx = i
            out.append(values[idx])
        return out
    if len(values) == 1:
        return [values[0] for _ in schedule_dates]
    if len(values) == len(schedule_dates):
        return list(values)
    raise ValueError("callable bond schedule block length does not match exercise dates")


def _parse_callability_data(node: ET.Element | None) -> tuple[CallableExerciseSpec, ...]:
    if node is None or not list(node):
        return ()
    schedule_dates = _parse_call_schedule_dates(node)
    styles, style_starts = _parse_scheduled_block(node, "Styles", "Style", lambda x: x.strip())
    prices, price_starts = _parse_scheduled_block(node, "Prices", "Price", lambda x: float(x))
    price_types, price_type_starts = _parse_scheduled_block(node, "PriceTypes", "PriceType", lambda x: x.strip())
    include_accruals, include_starts = _parse_scheduled_block(node, "IncludeAccruals", "IncludeAccrual", lambda x: _parse_bool(x, True))
    styles_n = _expand_scheduled_values(styles, style_starts, schedule_dates, "Bermudan")
    prices_n = _expand_scheduled_values(prices, price_starts, schedule_dates, 1.0)
    price_types_n = _expand_scheduled_values(price_types, price_type_starts, schedule_dates, "Clean")
    include_n = _expand_scheduled_values(include_accruals, include_starts, schedule_dates, True)
    out: list[CallableExerciseSpec] = []
    for i, d in enumerate(schedule_dates):
        style = str(styles_n[i]).strip()
        if style == "American":
            exercise_type = "FromThisDateOn" if i < len(schedule_dates) - 1 else "OnThisDate"
        elif style == "Bermudan":
            exercise_type = "OnThisDate"
        else:
            raise ValueError(f"unsupported callable bond exercise style '{style}'")
        out.append(
            CallableExerciseSpec(
                exercise_date=d,
                exercise_type=exercise_type,
                price=float(prices_n[i]),
                price_type=str(price_types_n[i]).strip() or "Clean",
                include_accrual=bool(include_n[i]),
            )
        )
    return tuple(out)


def _parse_fixed_leg_cashflows(leg_node: ET.Element, *, sign: float) -> tuple[str, str, tuple[BondCashflow, ...]]:
    currency = (leg_node.findtext("./Currency") or "").strip()
    notional = float((leg_node.findtext("./Notionals/Notional") or "0").strip())
    dc = (leg_node.findtext("./DayCounter") or "A365F").strip()
    rate = float((leg_node.findtext("./FixedLegData/Rates/Rate") or "0").strip())
    payment_convention = (leg_node.findtext("./PaymentConvention") or "F").strip()
    rules = leg_node.find("./ScheduleData/Rules")
    if rules is None:
        raise ValueError("bond fixed leg missing ScheduleData/Rules")
    start = _parse_any_date(rules.findtext("./StartDate") or "")
    end = _parse_any_date(rules.findtext("./EndDate") or "")
    tenor = (rules.findtext("./Tenor") or "").strip()
    calendar = (rules.findtext("./Calendar") or "TARGET").strip()
    convention = (rules.findtext("./Convention") or payment_convention).strip()
    term_convention = (rules.findtext("./TermConvention") or convention).strip()
    rule = (rules.findtext("./Rule") or "Forward").strip()
    schedule = _build_schedule(start, end, tenor, calendar, convention, term_convention, rule)
    # ORE ultimately prices bonds off the cashflow leg that comes out of
    # `BondData` -> coupon schedule -> `QuantLib::Bond`. For the Python path we
    # reconstruct those deterministic cashflows directly instead of constructing
    # a live QuantLib bond object and then re-extracting the same dates/amounts.
    flows: list[BondCashflow] = []
    if len(schedule) < 2:
        raise ValueError("bond schedule produced fewer than two dates")
    for s, e in zip(schedule[:-1], schedule[1:]):
        accrual = _year_fraction(s, e, dc)
        amount = sign * notional * rate * accrual
        flows.append(BondCashflow(pay_date=e, amount=amount, flow_type="Interest", accrual_start=s, accrual_end=e, nominal=notional))
    flows.append(BondCashflow(pay_date=schedule[-1], amount=sign * notional, flow_type="Notional", nominal=notional))
    return currency, dc, tuple(flows)


def _load_bond_cashflows_from_flows(flows_csv: Path, trade_id: str, *, forward_underlying_only: bool = False) -> tuple[BondCashflow, ...]:
    # When ORE `flows.csv` is available, prefer it over a locally rebuilt
    # schedule. This follows the same parity philosophy already used elsewhere
    # in the repo for swaps: ORE's emitted cashflow schedule is the most
    # authoritative source for pay dates, accrued amounts, and redemption timing.
    rows: list[BondCashflow] = []
    with open(flows_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        trade_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if (row.get(trade_key, "") or "").strip() != trade_id:
                continue
            pay_date = _parse_any_date(row.get("PayDate", ""))
            flow_type = (row.get("FlowType", "") or "").strip()
            if forward_underlying_only:
                if not flow_type.startswith("Bond_"):
                    continue
                flow_type = flow_type[len("Bond_") :]
            amount = float((row.get("Amount", "0") or "0").replace("#N/A", "0"))
            nominal_text = (row.get("Notional", "") or "").strip()
            nominal = None if nominal_text in {"", "#N/A"} else float(nominal_text)
            accrual_start = None
            accrual_end = None
            start_text = (row.get("AccrualStartDate", "") or "").strip()
            end_text = (row.get("AccrualEndDate", "") or "").strip()
            if start_text and start_text != "#N/A":
                accrual_start = _parse_any_date(start_text)
            if end_text and end_text != "#N/A":
                accrual_end = _parse_any_date(end_text)
            rows.append(BondCashflow(pay_date=pay_date, amount=amount, flow_type=flow_type, accrual_start=accrual_start, accrual_end=accrual_end, nominal=nominal))
    if not rows:
        raise ValueError(f"no bond flows found for trade '{trade_id}' in {flows_csv}")
    rows.sort(key=lambda x: (x.pay_date, 0 if "Interest" in x.flow_type else 1))
    return tuple(rows)


def _load_engine_spec(pricingengine_path: Path | None, trade_type: str) -> BondEngineSpec:
    # Only port the engine fields that materially affect the price formulas we
    # reproduce here. This is taken from the ORE product docs and the
    # corresponding builders / pricing engines:
    # - TimestepPeriod
    # - SpreadOnIncomeCurve
    # - TreatSecuritySpreadAsCreditSpread
    # - IncludePastCashflows
    if pricingengine_path is None or not pricingengine_path.exists():
        return BondEngineSpec()
    root = ET.parse(pricingengine_path).getroot()
    product = root.find(f"./Product[@type='{trade_type}']")
    if product is None:
        return BondEngineSpec()
    model_params = {
        (n.attrib.get("name", "") or "").strip(): (n.text or "").strip()
        for n in product.findall("./ModelParameters/Parameter")
    }
    engine_params = {
        (n.attrib.get("name", "") or "").strip(): (n.text or "").strip()
        for n in product.findall("./EngineParameters/Parameter")
    }
    step = (engine_params.get("TimestepPeriod") or "3M").strip().upper()
    months = int(step[:-1]) if step.endswith("M") and step[:-1].isdigit() else 3
    return BondEngineSpec(
        timestep_months=max(months, 1),
        spread_on_income_curve=_parse_bool(engine_params.get("SpreadOnIncomeCurve"), True),
        treat_security_spread_as_credit_spread=_parse_bool(model_params.get("TreatSecuritySpreadAsCreditSpread"), False),
        include_past_cashflows=_parse_bool(engine_params.get("IncludePastCashflows"), False),
    )


def _load_callable_engine_spec(pricingengine_path: Path | None) -> CallableBondEngineSpec:
    if pricingengine_path is None or not pricingengine_path.exists():
        return CallableBondEngineSpec()
    root = ET.parse(pricingengine_path).getroot()
    product = root.find("./Product[@type='CallableBond']")
    if product is None:
        return CallableBondEngineSpec()
    model = (product.findtext("./Model") or "LGM").strip()
    engine = (product.findtext("./Engine") or "Grid").strip()
    model_params = {
        (n.attrib.get("name", "") or "").strip(): (n.text or "").strip()
        for n in product.findall("./ModelParameters/Parameter")
    }
    engine_params = {
        (n.attrib.get("name", "") or "").strip(): (n.text or "").strip()
        for n in product.findall("./EngineParameters/Parameter")
    }
    variant = engine if engine in {"Grid", "FD"} else "Grid"
    state_grid_points = int((engine_params.get("StateGridPoints") or "121").strip() or "121")
    if state_grid_points % 2 == 0:
        state_grid_points += 1
    return CallableBondEngineSpec(
        model_family="LGM" if model in {"LGM", "CrossAssetModel"} else model,
        engine_variant=variant,
        spread_on_income_curve=_parse_bool(engine_params.get("SpreadOnIncomeCurve"), True),
        exercise_time_steps_per_year=max(int((model_params.get("ExerciseTimeStepsPerYear") or "24").strip() or "24"), 0),
        grid_sy=float((engine_params.get("sy") or "3.0").strip() or "3.0"),
        grid_ny=max(int((engine_params.get("ny") or "10").strip() or "10"), 1),
        grid_sx=float((engine_params.get("sx") or "6.0").strip() or "6.0"),
        # The callable-bond rollback is sensitive to the x-grid density because
        # exercise amounts are compared against continuation values on the
        # discretised state grid. ORE's convolution solver uses interpolation on
        # this grid; a denser default noticeably reduces residual parity error
        # without changing product semantics.
        grid_nx=max(int((engine_params.get("nx") or "30").strip() or "30"), 30),
        fd_max_time=float((engine_params.get("MaxTime") or "50.0").strip() or "50.0"),
        fd_state_grid_points=state_grid_points,
        fd_time_steps_per_year=max(int((engine_params.get("TimeStepsPerYear") or "24").strip() or "24"), 1),
        fd_mesher_epsilon=float((engine_params.get("MesherEpsilon") or "1e-4").strip() or "1e-4"),
        fd_scheme=(engine_params.get("Scheme") or "Douglas").strip() or "Douglas",
        generate_additional_results=_parse_bool(
            {
                (n.attrib.get("name", "") or "").strip(): (n.text or "").strip()
                for n in root.findall("./GlobalParameters/Parameter")
            }.get("GenerateAdditionalResults"),
            True,
        ),
    )


def load_bond_trade_spec(
    *,
    portfolio_xml: Path,
    trade_id: str,
    reference_data_path: Path | None,
    pricingengine_path: Path | None,
    flows_csv: Path | None,
) -> tuple[BondTradeSpec, BondEngineSpec]:
    root = ET.parse(portfolio_xml).getroot()
    trade = _resolve_trade(root, trade_id)
    trade_type = (trade.findtext("./TradeType") or "").strip()
    if trade_type not in {"Bond", "ForwardBond"}:
        raise ValueError(f"unsupported bond trade type '{trade_type}'")
    if trade_type == "Bond":
        bond_node = _merge_reference_data(trade.find("./BondData"), reference_data_path)
        payer = _parse_bool(bond_node.findtext("./LegData/Payer"), False)
        sign = -1.0 if payer else 1.0
        currency, _, cashflows = _parse_fixed_leg_cashflows(bond_node.find("./LegData"), sign=sign)
        spec = BondTradeSpec(
            trade_id=trade_id,
            trade_type=trade_type,
            currency=currency,
            payer=payer,
            security_id=(bond_node.findtext("./SecurityId") or "").strip(),
            credit_curve_id=(bond_node.findtext("./CreditCurveId") or "").strip(),
            reference_curve_id=(bond_node.findtext("./ReferenceCurveId") or "").strip(),
            income_curve_id=(bond_node.findtext("./IncomeCurveId") or "").strip(),
            settlement_days=int((bond_node.findtext("./SettlementDays") or "0").strip() or "0"),
            calendar=(bond_node.findtext("./Calendar") or "TARGET").strip(),
            issue_date=_parse_any_date(bond_node.findtext("./IssueDate") or ""),
            bond_notional=float((bond_node.findtext("./BondNotional") or "1").strip() or "1"),
            cashflows=_load_bond_cashflows_from_flows(flows_csv, trade_id) if flows_csv and flows_csv.exists() else cashflows,
        )
    else:
        # This follows `ForwardBond::fromXML()` / `ForwardBond::build()` in ORE:
        # the underlying bond is parsed first, then settlement / premium /
        # long-short forward metadata are applied around it.
        fwd_node = trade.find("./ForwardBondData")
        if fwd_node is None:
            raise ValueError("ForwardBond missing ForwardBondData")
        bond_node = _merge_reference_data(fwd_node.find("./BondData"), reference_data_path)
        payer = _parse_bool(bond_node.findtext("./LegData/Payer"), False)
        sign = -1.0 if payer else 1.0
        currency, dc, parsed_cashflows = _parse_fixed_leg_cashflows(bond_node.find("./LegData"), sign=sign)
        settlement = fwd_node.find("./SettlementData")
        if settlement is None:
            raise ValueError("ForwardBond missing SettlementData")
        spec = BondTradeSpec(
            trade_id=trade_id,
            trade_type=trade_type,
            currency=currency,
            payer=payer,
            security_id=(bond_node.findtext("./SecurityId") or "").strip(),
            credit_curve_id=(bond_node.findtext("./CreditCurveId") or "").strip(),
            reference_curve_id=(bond_node.findtext("./ReferenceCurveId") or "").strip(),
            income_curve_id=(bond_node.findtext("./IncomeCurveId") or "").strip(),
            settlement_days=int((bond_node.findtext("./SettlementDays") or "0").strip() or "0"),
            calendar=(bond_node.findtext("./Calendar") or "TARGET").strip(),
            issue_date=_parse_any_date(bond_node.findtext("./IssueDate") or ""),
            bond_notional=float((bond_node.findtext("./BondNotional") or "1").strip() or "1"),
            cashflows=_load_bond_cashflows_from_flows(flows_csv, trade_id, forward_underlying_only=True) if flows_csv and flows_csv.exists() else parsed_cashflows,
            forward_maturity_date=_parse_any_date(settlement.findtext("./ForwardMaturityDate") or ""),
            forward_settlement_date=_parse_any_date(settlement.findtext("./ForwardSettlementDate") or settlement.findtext("./ForwardMaturityDate") or ""),
            forward_amount=float((settlement.findtext("./Amount") or "nan").strip()),
            long_in_forward=_parse_bool(fwd_node.findtext("./LongInForward"), True),
            settlement_dirty=_parse_bool(settlement.findtext("./SettlementDirty"), True),
            compensation_payment=float((fwd_node.findtext("./PremiumData/Amount") or "0").strip() or "0"),
            compensation_payment_date=_parse_any_date(fwd_node.findtext("./PremiumData/Date") or settlement.findtext("./ForwardMaturityDate") or ""),
        )
    return spec, _load_engine_spec(pricingengine_path, trade_type)


def load_callable_bond_trade_spec(
    *,
    portfolio_xml: Path,
    trade_id: str,
    reference_data_path: Path | None,
    pricingengine_path: Path | None,
) -> tuple[CallableBondTradeSpec, CallableBondEngineSpec]:
    root = ET.parse(portfolio_xml).getroot()
    trade = _resolve_trade(root, trade_id)
    trade_type = (trade.findtext("./TradeType") or "").strip()
    if trade_type != "CallableBond":
        raise ValueError(f"unsupported callable trade type '{trade_type}'")
    callable_node = trade.find("./CallableBondData")
    if callable_node is None:
        raise ValueError("CallableBond missing CallableBondData")
    merged = _merge_callable_reference_data(callable_node, reference_data_path)
    bond_node = merged.find("./BondData")
    if bond_node is None:
        raise ValueError("CallableBondData missing BondData")
    payer = _parse_bool(bond_node.findtext("./LegData/Payer"), False)
    sign = -1.0 if payer else 1.0
    currency, _, parsed_cashflows = _parse_fixed_leg_cashflows(bond_node.find("./LegData"), sign=sign)
    scale = float((bond_node.findtext("./BondNotional") or "1").strip() or "1")
    scaled_cashflows = _scale_cashflows(parsed_cashflows, scale)
    bond_spec = BondTradeSpec(
        trade_id=trade_id,
        trade_type="Bond",
        currency=currency,
        payer=payer,
        security_id=(bond_node.findtext("./SecurityId") or "").strip(),
        credit_curve_id=(bond_node.findtext("./CreditCurveId") or "").strip(),
        reference_curve_id=(bond_node.findtext("./ReferenceCurveId") or "").strip(),
        income_curve_id=(bond_node.findtext("./IncomeCurveId") or "").strip(),
        settlement_days=int((bond_node.findtext("./SettlementDays") or "0").strip() or "0"),
        calendar=(bond_node.findtext("./Calendar") or "TARGET").strip(),
        issue_date=_parse_any_date(bond_node.findtext("./IssueDate") or ""),
        bond_notional=scale,
        cashflows=scaled_cashflows,
    )
    spec = CallableBondTradeSpec(
        trade_id=trade_id,
        currency=bond_spec.currency,
        security_id=bond_spec.security_id,
        credit_curve_id=bond_spec.credit_curve_id,
        reference_curve_id=bond_spec.reference_curve_id,
        income_curve_id=bond_spec.income_curve_id,
        settlement_days=bond_spec.settlement_days,
        calendar=bond_spec.calendar,
        issue_date=bond_spec.issue_date,
        bond_notional=bond_spec.bond_notional,
        bond=bond_spec,
        call_data=_parse_callability_data(merged.find("./CallData")),
        put_data=_parse_callability_data(merged.find("./PutData")),
    )
    return spec, _load_callable_engine_spec(pricingengine_path)


def _load_security_spread(market_data_file: Path, security_id: str) -> float:
    key = f"BOND/YIELD_SPREAD/{security_id}"
    with open(market_data_file, encoding="utf-8") as handle:
        for line in handle:
            toks = line.strip().split()
            if len(toks) >= 3 and toks[1] == key:
                return float(toks[2])
    return 0.0


def _apply_spread(curve, spread: float):
    if abs(spread) <= 1.0e-16:
        return curve
    return lambda t: float(curve(float(t))) * float(np.exp(-spread * max(float(t), 0.0)))


def _fit_curve_for_currency(ore_xml: Path, currency: str):
    fitted = fit_discount_curves_from_ore_market(ore_xml)
    if currency not in fitted:
        raise ValueError(f"no fitted market curve available for currency '{currency}'")
    payload = fitted[currency]
    return build_discount_curve_from_discount_pairs(list(zip(payload["times"], payload["dfs"])))


def _curve_from_flow_discounts(flows_csv: Path, trade_id: str, asof_date: date, day_counter: str, *, forward_underlying_only: bool = False):
    # Some legacy example cases ship `flows.csv` / `npv.csv` but not `curves.csv`.
    # In the risk-free / zero-spread cases, ORE's own cashflow report already
    # contains discount factors per flow, so we can use those as curve pillars.
    # This is not a generic market reconstruction; it is a parity-oriented
    # fallback for the snapshot CLI's price-only bond checks.
    pairs: list[tuple[float, float]] = []
    with open(flows_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        trade_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if (row.get(trade_key, "") or "").strip() != trade_id:
                continue
            flow_type = (row.get("FlowType", "") or "").strip()
            if forward_underlying_only:
                if not flow_type.startswith("Bond_"):
                    continue
            elif flow_type == "ForwardValue":
                continue
            df_text = (row.get("DiscountFactor", "") or "").strip()
            if not df_text or df_text == "#N/A":
                continue
            pay_date = _parse_any_date(row.get("PayDate", ""))
            t = _time_from_dates(asof_date, pay_date, day_counter)
            if t <= 0.0:
                continue
            pairs.append((t, float(df_text)))
    if not pairs:
        raise ValueError(f"no flow discount factors found for trade '{trade_id}' in {flows_csv}")
    # Anchor the curve at the valuation date. ORE discount factors in the flow
    # report are quoted from the asof date, so `P(0,0)` must be exactly 1.
    # Without this anchor, the first flow DF can leak into `curve(0)` via the
    # interpolator and the risky-bond engine will overstate PV because it divides
    # future discounts by `df_npv = curve(0)`.
    pairs.append((0.0, 1.0))
    pairs = sorted(set((round(t, 12), df) for t, df in pairs))
    return build_discount_curve_from_discount_pairs(pairs)


def _survival_from_piecewise(times: np.ndarray, hazard_times: np.ndarray, hazard_rates: np.ndarray, extra_hazard: float = 0.0) -> np.ndarray:
    if abs(extra_hazard) <= 1.0e-16:
        return survival_probability_from_hazard(times, hazard_times, hazard_rates)
    return survival_probability_from_hazard(times, hazard_times, hazard_rates + extra_hazard)


def _default_probability(start_t: float, end_t: float, hazard_times: np.ndarray, hazard_rates: np.ndarray, extra_hazard: float = 0.0) -> float:
    start = _survival_from_piecewise(np.asarray([start_t], dtype=float), hazard_times, hazard_rates, extra_hazard)[0]
    end = _survival_from_piecewise(np.asarray([end_t], dtype=float), hazard_times, hazard_rates, extra_hazard)[0]
    return max(start - end, 0.0)


def _cashflow_has_occurred(cf_date: date, ref_date: date, include_ref_date_flows: bool = False) -> bool:
    return cf_date < ref_date or (cf_date == ref_date and not include_ref_date_flows)


def _bond_npv(
    spec: BondTradeSpec,
    *,
    asof_date: date,
    day_counter: str,
    discount_curve,
    income_curve,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
    recovery_rate: float,
    security_spread: float,
    engine_spec: BondEngineSpec,
    npv_date: date | None = None,
    settlement_date: date | None = None,
    conditional_on_survival: bool = False,
) -> tuple[float, float]:
    # This is the core port of QuantExt's `DiscountingRiskyBondEngine`.
    #
    # Mapping to C++ concepts:
    # - `npv_date` / `settlement_date` correspond to the engine's forward-NPV and
    #   settlement evaluation points.
    # - `discount_curve` corresponds to the benchmark / risky discounting curve
    #   after any security-spread-on-curve adjustment.
    # - `income_curve` corresponds to the compounding curve used for settlement
    #   value (`compoundFactorSettlement` in the C++ engine).
    # - `conditional_on_survival` is used by ORE when valuing forward bond
    #   positions off the underlying bond's forward price.
    npv_date = npv_date or asof_date
    settlement_date = settlement_date or npv_date
    include_ref_date_flows = False
    npv_t = _time_from_dates(asof_date, npv_date, day_counter)
    sett_t = _time_from_dates(asof_date, settlement_date, day_counter)
    extra_credit = security_spread / max(1.0 - float(recovery_rate), 1.0e-12) if engine_spec.treat_security_spread_as_credit_spread else 0.0
    # ORE has two spread treatments:
    # 1. default: apply security spread as a zero spread on the benchmark curve
    # 2. TreatSecuritySpreadAsCreditSpread=true: leave the benchmark curve
    #    unchanged and add the spread / (1 - recovery) onto the hazard curve
    eff_discount = discount_curve if engine_spec.treat_security_spread_as_credit_spread else _apply_spread(discount_curve, security_spread)
    eff_income = income_curve
    if security_spread and not engine_spec.treat_security_spread_as_credit_spread and engine_spec.spread_on_income_curve:
        eff_income = _apply_spread(income_curve, security_spread)
    df_npv = float(eff_income(max(npv_t, 0.0)))
    sp_npv = 1.0 if not conditional_on_survival else float(_survival_from_piecewise(np.asarray([max(npv_t, 0.0)]), hazard_times, hazard_rates, extra_credit)[0])
    df_settl = float(eff_income(max(sett_t, 0.0)))
    sp_settl = float(_survival_from_piecewise(np.asarray([max(sett_t, 0.0)]), hazard_times, hazard_rates, extra_credit)[0]) / (1.0 if not conditional_on_survival else max(sp_npv, 1.0e-18))
    compound_factor_settlement = (df_npv * sp_npv) / max(df_settl * sp_settl, 1.0e-18)
    npv = 0.0
    before_settlement = 0.0
    timestep = max(int(engine_spec.timestep_months), 1)
    for cf in spec.cashflows:
        if _cashflow_has_occurred(cf.pay_date, npv_date, include_ref_date_flows):
            continue
        cf_t = _time_from_dates(asof_date, cf.pay_date, day_counter)
        df = float(eff_discount(max(cf_t, 0.0))) / max(df_npv, 1.0e-18)
        surv = float(_survival_from_piecewise(np.asarray([max(cf_t, 0.0)]), hazard_times, hazard_rates, extra_credit)[0]) / max(sp_npv, 1.0e-18)
        pv = float(cf.amount) * df * surv
        if _cashflow_has_occurred(cf.pay_date, settlement_date, include_ref_date_flows):
            before_settlement += pv
        else:
            npv += pv
        if cf.accrual_start is not None and cf.accrual_end is not None and cf.nominal is not None:
            # Coupon recovery contribution. This follows the C++ engine's
            # `recoveryContribution(...)` call for coupon cashflows: expected
            # default probability over the coupon accrual period times recovery
            # of the coupon nominal, discounted to the valuation point.
            start = max(npv_date, cf.accrual_start)
            end = cf.accrual_end
            if start < end and recovery_rate > 0.0:
                start_t = _time_from_dates(asof_date, start, day_counter)
                end_t = _time_from_dates(asof_date, end, day_counter)
                default_prob = _default_probability(start_t, end_t, hazard_times, hazard_rates, extra_credit) / max(sp_npv, 1.0e-18)
                mid = start + (end - start) / 2
                mid_t = _time_from_dates(asof_date, mid, day_counter)
                npv += float(cf.nominal) * float(recovery_rate) * default_prob * float(eff_discount(max(mid_t, 0.0))) / max(df_npv, 1.0e-18)
    if len(spec.cashflows) == 1 and spec.cashflows[0].flow_type.endswith("Notional") and spec.cashflows[0].nominal is not None and recovery_rate > 0.0:
        # Zero-coupon bond special case. QuantExt handles this separately by
        # stepping through the life of the redemption with `timestepPeriod_`
        # because there are no coupon accrual periods to attach recovery to.
        redemption = spec.cashflows[0]
        start = npv_date
        while start < redemption.pay_date:
            step = min(redemption.pay_date, _add_months(start, timestep))
            start_t = _time_from_dates(asof_date, start, day_counter)
            end_t = _time_from_dates(asof_date, step, day_counter)
            default_prob = _default_probability(start_t, end_t, hazard_times, hazard_rates, extra_credit) / max(sp_npv, 1.0e-18)
            mid = start + (step - start) / 2
            mid_t = _time_from_dates(asof_date, mid, day_counter)
            npv += float(redemption.nominal) * float(recovery_rate) * default_prob * float(eff_discount(max(mid_t, 0.0))) / max(df_npv, 1.0e-18)
            start = step
    return float(npv + before_settlement), float(npv * compound_factor_settlement)


def _accrued_amount(spec: BondTradeSpec, settlement_date: date) -> float:
    # Forward bond settlement can be dirty or clean. ORE's
    # `DiscountingForwardBondEngine` adds accrued amount to the strike only when
    # settlement is clean. We compute the same accrued stub off the underlying
    # bond cashflow period that contains the bond settlement date.
    for cf in spec.cashflows:
        if cf.accrual_start is None or cf.accrual_end is None or cf.nominal is None:
            continue
        if cf.accrual_start <= settlement_date < cf.accrual_end and "Interest" in cf.flow_type:
            full_accrual = _year_fraction(cf.accrual_start, cf.accrual_end, "ACT/ACT")
            earned = _year_fraction(cf.accrual_start, settlement_date, "ACT/ACT")
            if full_accrual <= 0.0:
                return 0.0
            return float(cf.amount) * earned / full_accrual
    return 0.0


def _resolve_simulation_config_path(ore_xml: Path) -> Path | None:
    root = ET.parse(ore_xml).getroot()
    base = ore_xml.resolve().parent
    run_dir = base.parent
    input_dir = base
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in root.findall("./Setup/Parameter")
    }
    if setup_params.get("inputPath"):
        input_dir = (run_dir / setup_params["inputPath"]).resolve()
    for analytic in root.findall("./Analytics/Analytic[@type='simulation']"):
        for param in analytic.findall("./Parameter"):
            if (param.attrib.get("name", "") or "").strip() == "simulationConfigFile" and (param.text or "").strip():
                return (input_dir / (param.text or "").strip()).resolve()
    return None


def _resolve_curveconfig_path(ore_xml: Path) -> Path | None:
    root = ET.parse(ore_xml).getroot()
    base = ore_xml.resolve().parent
    run_dir = base.parent
    input_dir = base
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in root.findall("./Setup/Parameter")
    }
    if setup_params.get("inputPath"):
        input_dir = (run_dir / setup_params["inputPath"]).resolve()
    raw = setup_params.get("curveConfigFile", "../../Input/curveconfig.xml")
    return _resolve_ore_path(raw, input_dir)


def _ql_day_counter(name: str):
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for callable bond calibration")
    norm = _norm_dc(name)
    if norm in {"A360", "ACT/360"}:
        return ql.Actual360()
    if norm in {"30/360", "30E/360"}:
        return ql.Thirty360(ql.Thirty360.BondBasis)
    return ql.Actual365Fixed()


def _ql_calendar(name: str):
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for callable bond calibration")
    return ql.TARGET()


def _ql_bdc(name: str):
    txt = (name or "Following").strip().lower()
    if txt in {"modifiedfollowing", "modified following", "mf"}:
        return ql.ModifiedFollowing
    if txt in {"preceding", "p"}:
        return ql.Preceding
    if txt in {"unadjusted", "u"}:
        return ql.Unadjusted
    return ql.Following


def _parse_reference_grid_dates(asof_date: date, maturity_date: date, grid_text: str) -> list[date]:
    parts = [x.strip() for x in (grid_text or "").split(",") if x.strip()]
    if len(parts) != 2 or not parts[0].isdigit():
        return []
    count = int(parts[0])
    step = parts[1]
    try:
        months = _parse_tenor_months(step)
    except Exception:
        return []
    dates: list[date] = []
    current = asof_date
    for _ in range(max(count, 0)):
        current = _add_months(current, months)
        if current >= maturity_date:
            break
        dates.append(current)
    return dates


def _load_ore_swaption_surface_spec(todaysmarket_xml: Path, curveconfig_path: Path, currency: str) -> dict[str, object]:
    tm_root = ET.parse(todaysmarket_xml).getroot()
    mapping = tm_root.find("./SwaptionVolatilities[@id='default']")
    if mapping is None:
        raise ValueError("todaysmarket.xml missing SwaptionVolatilities id='default'")
    handle = mapping.findtext(f"./SwaptionVolatility[@currency='{currency}']")
    if not handle:
        raise ValueError(f"todaysmarket.xml missing swaption volatility mapping for currency '{currency}'")
    curve_id = handle.strip().split("/")[-1]
    cfg_root = ET.parse(curveconfig_path).getroot()
    node = None
    for candidate in cfg_root.findall(".//SwaptionVolatility"):
        if (candidate.findtext("./CurveId") or "").strip() == curve_id:
            node = candidate
            break
    if node is None:
        raise ValueError(f"curveconfig.xml missing SwaptionVolatility with CurveId '{curve_id}'")
    option_tenors = [x.strip() for x in (node.findtext("./OptionTenors") or "").replace("\n", "").split(",") if x.strip()]
    swap_tenors = [x.strip() for x in (node.findtext("./SwapTenors") or "").replace("\n", "").split(",") if x.strip()]
    return {
        "curve_id": curve_id,
        "volatility_type": (node.findtext("./VolatilityType") or "Normal").strip(),
        "day_counter": (node.findtext("./DayCounter") or "Actual/365 (Fixed)").strip(),
        "calendar": (node.findtext("./Calendar") or "TARGET").strip(),
        "business_day_convention": (node.findtext("./BusinessDayConvention") or "Following").strip(),
        "option_tenors": option_tenors,
        "swap_tenors": swap_tenors,
        "short_swap_index_base": (node.findtext("./ShortSwapIndexBase") or "").strip(),
        "swap_index_base": (node.findtext("./SwapIndexBase") or "").strip(),
    }


def _load_ore_swaption_quotes(
    market_data_file: Path,
    *,
    currency: str,
    option_tenors: list[str],
    swap_tenors: list[str],
    volatility_type: str,
) -> np.ndarray:
    quote_kind = "RATE_NVOL" if (volatility_type or "").strip().lower() == "normal" else "RATE_LNVOL"
    prefix = f"SWAPTION/{quote_kind}/{currency}/"
    quotes: dict[tuple[str, str], float] = {}
    with open(market_data_file, encoding="utf-8") as handle:
        for line in handle:
            toks = line.strip().split()
            if len(toks) < 3:
                continue
            key = toks[1]
            if not key.startswith(prefix):
                continue
            parts = key.split("/")
            if len(parts) < 6:
                continue
            expiry = parts[3].strip()
            term = parts[4].strip()
            strike = parts[5].strip().upper()
            if strike != "ATM":
                continue
            quotes[(expiry, term)] = float(toks[2])
    matrix = np.empty((len(option_tenors), len(swap_tenors)), dtype=float)
    for i, expiry in enumerate(option_tenors):
        for j, term in enumerate(swap_tenors):
            key = (expiry, term)
            if key not in quotes:
                raise ValueError(f"missing swaption quote '{quote_kind}/{currency}/{expiry}/{term}/ATM'")
            matrix[i, j] = quotes[key]
    return matrix


def _build_ql_discount_curve(ore_xml: Path, currency: str, asof_date: date):
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for callable bond calibration")
    fitted = fit_discount_curves_from_ore_market(ore_xml)
    if currency not in fitted:
        raise ValueError(f"no fitted market curve available for currency '{currency}'")
    payload = fitted[currency]
    base_curve = build_discount_curve_from_discount_pairs(list(zip(payload["times"], payload["dfs"])))
    dates = [ql.Date(asof_date.day, asof_date.month, asof_date.year)]
    dfs = [1.0]
    max_time = max(float(payload["times"][-1]) if payload["times"] else 0.0, 61.0)
    sample_times = sorted(
        set(
            [round(float(t), 8) for t in payload["times"] if float(t) > 0.0]
            + [round(0.25 * i, 8) for i in range(1, int(math.ceil(max_time / 0.25)) + 1)]
        )
    )
    for tt in sample_times:
        qd = ql.Date(asof_date.day, asof_date.month, asof_date.year) + int(round(tt * 365.0))
        if qd <= dates[-1]:
            continue
        dates.append(qd)
        dfs.append(float(base_curve(tt)))
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))


def _build_callable_lgm_market_inputs(
    *,
    ore_xml: Path,
    todaysmarket_xml: Path,
    market_data_file: Path,
    currency: str,
    asof_date: date,
) -> LgmMarketInputs:
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for callable bond calibration")
    ql.Settings.instance().evaluationDate = ql.Date(asof_date.day, asof_date.month, asof_date.year)
    curveconfig_path = _resolve_curveconfig_path(ore_xml)
    if curveconfig_path is None or not curveconfig_path.exists():
        raise FileNotFoundError("curveconfig.xml could not be resolved for callable bond calibration")
    spec = _load_ore_swaption_surface_spec(todaysmarket_xml, curveconfig_path, currency)
    quotes = _load_ore_swaption_quotes(
        market_data_file,
        currency=currency,
        option_tenors=list(spec["option_tenors"]),
        swap_tenors=list(spec["swap_tenors"]),
        volatility_type=str(spec["volatility_type"]),
    )
    option_periods = ql.PeriodVector()
    for tenor in spec["option_tenors"]:
        option_periods.push_back(ql.Period(str(tenor)))
    swap_periods = ql.PeriodVector()
    for tenor in spec["swap_tenors"]:
        swap_periods.push_back(ql.Period(str(tenor)))
    vols = ql.Matrix(quotes.shape[0], quotes.shape[1])
    for i in range(quotes.shape[0]):
        for j in range(quotes.shape[1]):
            vols[i][j] = float(quotes[i, j])
    dc = _ql_day_counter(str(spec["day_counter"]))
    cal = _ql_calendar(str(spec["calendar"]))
    bdc = _ql_bdc(str(spec["business_day_convention"]))
    vol_type = ql.Normal if str(spec["volatility_type"]).strip().lower() == "normal" else ql.ShiftedLognormal
    vol_surface = ql.SwaptionVolatilityMatrix(cal, bdc, option_periods, swap_periods, vols, dc, False, vol_type)
    discount_curve = _build_ql_discount_curve(ore_xml, currency, asof_date)
    short_tenor = str(spec["short_swap_index_base"]).strip().split("-")[-1] or "1Y"
    long_tenor = str(spec["swap_index_base"]).strip().split("-")[-1] or "30Y"
    short_index = ql.EuriborSwapIsdaFixA(ql.Period(short_tenor), discount_curve)
    long_index = ql.EuriborSwapIsdaFixA(ql.Period(long_tenor), discount_curve)
    return LgmMarketInputs(
        swaption_vol_surface=ql.SwaptionVolatilityStructureHandle(vol_surface),
        swap_index=long_index,
        short_swap_index=short_index,
        calibration_discount_curve=discount_curve,
        model_discount_curve=discount_curve,
    )


@lru_cache(maxsize=16)
def _try_calibrate_callable_lgm(
    *,
    ore_xml: Path,
    pricingengine_path: Path | None,
    todaysmarket_xml: Path,
    market_data_file: Path,
    currency: str,
    maturity_date: date,
    asof_date: date,
) -> LGM1F | None:
    if ql is None:
        return None
    model_params: dict[str, str] = {}
    if pricingengine_path is not None and pricingengine_path.exists():
        root = ET.parse(pricingengine_path).getroot()
        product = root.find("./Product[@type='CallableBond']")
        if product is not None:
            model_params = {
                (n.attrib.get("name", "") or "").strip(): (n.text or "").strip()
                for n in product.findall("./ModelParameters/Parameter")
            }
    calibration = (model_params.get("Calibration") or "None").strip()
    strategy = (model_params.get("CalibrationStrategy") or "None").strip()
    if calibration == "None" or strategy not in {"CoterminalATM", "CoterminalDealStrike"}:
        return None
    try:
        market_inputs = _build_callable_lgm_market_inputs(
            ore_xml=ore_xml,
            todaysmarket_xml=todaysmarket_xml,
            market_data_file=market_data_file,
            currency=currency,
            asof_date=asof_date,
        )
        expiries = _parse_reference_grid_dates(
            asof_date,
            maturity_date,
            model_params.get("ReferenceCalibrationGrid", ""),
        )
        if not expiries:
            return None
        sigma = [float(x.strip()) for x in (model_params.get("Volatility") or "0.01").split(",") if x.strip()]
        sigma_times = [float(x.strip()) for x in (model_params.get("VolatilityTimes") or "").split(",") if x.strip()]
        lambda_value = float((model_params.get("Reversion") or "0.03").strip() or "0.03")
        result = calibrate_lgm_currency(
            CurrencyLgmConfig(
                currency=currency,
                calibration_type=CalibrationType.BOOTSTRAP if calibration == "Bootstrap" else CalibrationType.BEST_FIT,
                volatility=replace(
                    CurrencyLgmConfig(currency=currency).volatility,
                    calibrate=True,
                    # The external calibration backend is QuantLib GSR-based
                    # and only supports the Hull-White parametrisation. ORE's
                    # callable-bond config here requests Hagan volatility, but
                    # calibrating the same coterminal basket with the available
                    # HW/GSR engine is still much closer to native ORE than
                    # keeping the fixed sigma/lambda fallback.
                    type=VolatilityType.HULL_WHITE,
                    param_type=ParamType.PIECEWISE if calibration == "Bootstrap" else ParamType.CONSTANT,
                    time_grid=tuple(sigma_times),
                    initial_values=tuple(sigma),
                ),
                reversion=replace(
                    CurrencyLgmConfig(currency=currency).reversion,
                    calibrate=False,
                    type=ReversionType.HULL_WHITE if (model_params.get("ReversionType") or "HullWhite").strip() == "HullWhite" else ReversionType.HAGAN,
                    param_type=ParamType.CONSTANT,
                    time_grid=(),
                    initial_values=(lambda_value,),
                ),
                calibration_swaptions=tuple(
                    [
                        # Mirror ORE's coterminal basket: calibration expiry grid
                        # before maturity, each swap maturing on the callable
                        # bond's maturity date.
                        SwaptionSpec(
                            expiry=d.isoformat(),
                            term=maturity_date.isoformat(),
                            strike="ATM",
                        )
                        for d in expiries
                    ]
                ),
                # `lgm_calibration.py` expects the basket to be provided
                # explicitly; its `reference_calibration_grid` helper parses a
                # list of dates / periods rather than ORE's `400,3M` date-grid
                # shorthand. We already expanded that shorthand into the
                # coterminal expiry list above, so leave the secondary filter
                # empty here.
                reference_calibration_grid="",
                bootstrap_tolerance=float((model_params.get("Tolerance") or "1e-4").strip() or "1e-4"),
                continue_on_error=False,
            ),
            market_inputs,
        )
    except Exception:
        return None
    alpha_values = tuple(max(float(x), 0.0) for x in result.volatility.values)
    params = LGMParams(
        alpha_times=tuple(float(x) for x in result.volatility.time_grid),
        # QuantLib's bootstrap can leave a handful of sigma buckets at tiny
        # negative values from numerical noise (order 1e-8 to 1e-7). The Python
        # LGM kernel requires non-negative alpha, so floor those residuals at 0.
        alpha_values=alpha_values,
        kappa_times=tuple(float(x) for x in result.reversion.time_grid),
        kappa_values=tuple(float(x) for x in result.reversion.values),
        shift=0.0,
        scaling=1.0,
    )
    return LGM1F(params)


def _build_lgm_model_for_callable(
    *,
    ore_xml: Path,
    pricingengine_path: Path | None,
    todaysmarket_xml: Path,
    market_data_file: Path,
    currency: str,
    maturity_date: date,
    asof_date: date,
) -> LGM1F:
    calibrated = _try_calibrate_callable_lgm(
        ore_xml=ore_xml,
        pricingengine_path=pricingengine_path,
        todaysmarket_xml=todaysmarket_xml,
        market_data_file=market_data_file,
        currency=currency,
        maturity_date=maturity_date,
        asof_date=asof_date,
    )
    if calibrated is not None:
        return calibrated
    sim_cfg = _resolve_simulation_config_path(ore_xml)
    payload = None
    if sim_cfg is not None and sim_cfg.exists():
        try:
            payload = parse_lgm_params_from_simulation_xml(str(sim_cfg), ccy_key=currency)
        except Exception:
            payload = None
        if payload is None:
            try:
                payload = parse_lgm_params_from_simulation_xml(str(sim_cfg), ccy_key="default")
            except Exception:
                payload = None
    if payload is None:
        model_params = {}
        if pricingengine_path is not None and pricingengine_path.exists():
            root = ET.parse(pricingengine_path).getroot()
            product = root.find("./Product[@type='CallableBond']")
            if product is not None:
                model_params = {
                    (n.attrib.get("name", "") or "").strip(): (n.text or "").strip()
                    for n in product.findall("./ModelParameters/Parameter")
                }
        vol = float((model_params.get("Volatility") or "0.01").strip() or "0.01")
        rev = float((model_params.get("Reversion") or "0.03").strip() or "0.03")
        payload = {
            "alpha_times": np.asarray([], dtype=float),
            "alpha_values": np.asarray([vol], dtype=float),
            "kappa_times": np.asarray([], dtype=float),
            "kappa_values": np.asarray([rev], dtype=float),
            "shift": 0.0,
            "scaling": 1.0,
        }
    params = LGMParams(
        alpha_times=tuple(float(x) for x in np.asarray(payload["alpha_times"], dtype=float)),
        alpha_values=tuple(float(x) for x in np.asarray(payload["alpha_values"], dtype=float)),
        kappa_times=tuple(float(x) for x in np.asarray(payload["kappa_times"], dtype=float)),
        kappa_values=tuple(float(x) for x in np.asarray(payload["kappa_values"], dtype=float)),
        shift=float(payload.get("shift", 0.0)),
        scaling=float(payload.get("scaling", 1.0)),
    )
    return LGM1F(params)


def _effective_bond_discount_curve(reference_curve, hazard_times: np.ndarray, hazard_rates: np.ndarray, recovery_rate: float, security_spread: float):
    rr = float(recovery_rate)
    spread = float(security_spread)

    def curve(t: float) -> float:
        tt = max(float(t), 0.0)
        ref = float(reference_curve(tt))
        surv = float(_survival_from_piecewise(np.asarray([tt], dtype=float), hazard_times, hazard_rates)[0])
        return ref * (surv ** max(1.0 - rr, 0.0)) * math.exp(-spread * tt)

    return curve


def _callable_notional_schedule(spec: CallableBondTradeSpec, asof_date: date, day_counter: str) -> tuple[np.ndarray, np.ndarray]:
    coupon_rows = [
        cf
        for cf in spec.bond.cashflows
        if cf.accrual_start is not None and cf.accrual_end is not None and cf.nominal is not None and not _cashflow_has_occurred(cf.pay_date, asof_date, False)
    ]
    if not coupon_rows:
        return np.zeros(0, dtype=float), np.asarray([float(spec.bond_notional)], dtype=float)
    notionals = [float(coupon_rows[0].nominal or spec.bond_notional)]
    change_times: list[float] = []
    last_nominal = notionals[0]
    for cf in coupon_rows:
        nominal = float(cf.nominal or last_nominal)
        if not math.isclose(nominal, last_nominal, rel_tol=0.0, abs_tol=1.0e-12):
            change_times.append(_time_from_dates(asof_date, cf.pay_date, day_counter))
            notionals.append(nominal)
            last_nominal = nominal
    return np.asarray(change_times, dtype=float), np.asarray(notionals, dtype=float)


def _callable_notional_at_time(change_times: np.ndarray, notionals: np.ndarray, t: float) -> float:
    if notionals.size == 0:
        return 0.0
    idx = int(np.searchsorted(change_times, float(t), side="right"))
    return float(notionals[min(idx, notionals.size - 1)])


def _callable_accrual_at_time(spec: CallableBondTradeSpec, asof_date: date, day_counter: str, t: float) -> float:
    tt = float(t)
    for cf in spec.bond.cashflows:
        if cf.accrual_start is None or cf.accrual_end is None or cf.nominal is None:
            continue
        pay_t = _time_from_dates(asof_date, cf.pay_date, day_counter)
        start_t = _time_from_dates(asof_date, cf.accrual_start, day_counter)
        end_t = _time_from_dates(asof_date, cf.accrual_end, day_counter)
        if pay_t > tt > start_t and end_t > start_t:
            return (tt - start_t) / max(end_t - start_t, 1.0e-18) * float(cf.amount)
    return 0.0


def _callable_price_amount(exercise: CallableExerciseSpec, notional: float, accruals: float) -> float:
    amount = float(exercise.price) * float(notional)
    if str(exercise.price_type).strip() == "Clean":
        amount += float(accruals)
    if not exercise.include_accrual:
        amount -= float(accruals)
    return amount


def _callable_effective_income_curve(reference_curve, income_curve, security_spread: float, spread_on_income_curve: bool):
    if spread_on_income_curve and abs(float(security_spread)) > 1.0e-16:
        return _apply_spread(income_curve or reference_curve, float(security_spread))
    return income_curve or reference_curve


def _build_callable_grid_times(
    mandatory_times: np.ndarray,
    *,
    max_time: float,
    time_steps_per_year: int,
) -> np.ndarray:
    base = [0.0]
    if max_time > 0.0 and time_steps_per_year > 0:
        steps = max(int(round(float(time_steps_per_year) * float(max_time) + 0.5)), 1)
        base.extend(np.linspace(0.0, float(max_time), steps + 1).tolist())
    base.extend([float(x) for x in np.asarray(mandatory_times, dtype=float) if float(x) > 0.0])
    return np.asarray(sorted(set(round(x, 12) for x in base)), dtype=float)


def _find_time_index(grid: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(grid, float(t), side="left"))
    if idx >= grid.size:
        return grid.size - 1
    if idx > 0 and abs(grid[idx - 1] - float(t)) <= abs(grid[idx] - float(t)):
        return idx - 1
    return idx


def _register_callable_exercises(
    grid: np.ndarray,
    exercises: tuple[CallableExerciseSpec, ...],
    *,
    asof_date: date,
    day_counter: str,
) -> dict[int, CallableExerciseSpec]:
    out: dict[int, CallableExerciseSpec] = {}
    for i, ex in enumerate(exercises):
        ex_time = _time_from_dates(asof_date, ex.exercise_date, day_counter)
        if ex.exercise_type == "OnThisDate":
            start_idx = _find_time_index(grid, ex_time)
            out[start_idx] = ex
            continue
        start_idx = _find_time_index(grid, ex_time)
        next_time = _time_from_dates(asof_date, exercises[i + 1].exercise_date, day_counter) if i + 1 < len(exercises) else ex_time
        end_idx = max(_find_time_index(grid, next_time) - 1, start_idx)
        for j in range(start_idx, end_idx + 1):
            out[j] = ex
    return out


def _callable_underlying_reduced_value(
    spec: CallableBondTradeSpec,
    model: LGM1F,
    eff_discount_curve,
    *,
    asof_date: date,
    day_counter: str,
    t: float,
    x_grid: np.ndarray,
) -> np.ndarray:
    value = np.zeros_like(np.asarray(x_grid, dtype=float))
    for cf in spec.bond.cashflows:
        pay_t = _time_from_dates(asof_date, cf.pay_date, day_counter)
        if pay_t <= t + 1.0e-12:
            continue
        value += float(cf.amount) * np.asarray(model.discount_bond(float(t), float(pay_t), x_grid, eff_discount_curve, eff_discount_curve), dtype=float)
    num = np.asarray(model.numeraire_lgm(float(t), x_grid, eff_discount_curve), dtype=float)
    return value / num


def _rollback_callable_bond_lgm_value(
    spec: CallableBondTradeSpec,
    engine_spec: CallableBondEngineSpec,
    *,
    asof_date: date,
    day_counter: str,
    reference_curve,
    income_curve,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
    recovery_rate: float,
    security_spread: float,
    model: LGM1F,
) -> float:
    eff_discount_curve = _effective_bond_discount_curve(reference_curve, hazard_times, hazard_rates, recovery_rate, security_spread)
    eff_income_curve = _callable_effective_income_curve(reference_curve, income_curve, security_spread, engine_spec.spread_on_income_curve)

    pay_times = np.asarray(
        [
            _time_from_dates(asof_date, cf.pay_date, day_counter)
            for cf in spec.bond.cashflows
            if not _cashflow_has_occurred(cf.pay_date, asof_date, False)
        ],
        dtype=float,
    )
    exercise_times = np.asarray(
        [
            _time_from_dates(asof_date, ex.exercise_date, day_counter)
            for ex in spec.call_data + spec.put_data
            if ex.exercise_date > asof_date
        ],
        dtype=float,
    )
    mandatory_times = np.asarray(sorted(set([float(x) for x in np.concatenate((pay_times, exercise_times)) if float(x) > 0.0])), dtype=float)
    max_time = float(np.max(mandatory_times)) if mandatory_times.size else 0.0
    if engine_spec.engine_variant == "Grid":
        mx = max(int(round(float(engine_spec.grid_sx) * float(engine_spec.grid_nx))), 1)
        y_nodes, y_weights = _convolution_nodes_and_weights(float(engine_spec.grid_sy), int(engine_spec.grid_ny))
        effective_steps = max(int(engine_spec.exercise_time_steps_per_year), 0)
    else:
        mx = max((int(engine_spec.fd_state_grid_points) - 1) // 2, 1)
        nx = max(int(round(mx / max(6.0, 1.0e-12))), 1)
        y_nodes, y_weights = _convolution_nodes_and_weights(3.0, 10)
        engine_spec = replace(engine_spec, grid_nx=nx)
        effective_steps = max(int(engine_spec.exercise_time_steps_per_year), int(engine_spec.fd_time_steps_per_year))
    grid = _build_callable_grid_times(mandatory_times, max_time=max(max_time, engine_spec.fd_max_time if engine_spec.engine_variant == "FD" else max_time), time_steps_per_year=effective_steps)
    if grid[-1] > max_time and max_time > 0.0:
        grid = grid[grid <= max_time + 1.0e-12]
        if grid.size == 0 or grid[-1] < max_time:
            grid = np.append(grid, max_time)
    zeta_grid = np.asarray(model.zeta(grid), dtype=float)
    x_grids = []
    for v in zeta_grid:
        if abs(float(v)) <= 1.0e-18:
            x_grids.append(np.zeros(2 * mx + 1, dtype=float))
        else:
            dx = np.sqrt(max(float(v), 0.0)) / float(max(engine_spec.grid_nx, 1))
            x_grids.append(dx * (np.arange(2 * mx + 1, dtype=float) - mx))
    call_map = _register_callable_exercises(
        grid,
        tuple(ex for ex in spec.call_data if ex.exercise_date > asof_date),
        asof_date=asof_date,
        day_counter=day_counter,
    )
    put_map = _register_callable_exercises(
        grid,
        tuple(ex for ex in spec.put_data if ex.exercise_date > asof_date),
        asof_date=asof_date,
        day_counter=day_counter,
    )
    cashflow_map: dict[int, list[BondCashflow]] = {}
    for cf in spec.bond.cashflows:
        if _cashflow_has_occurred(cf.pay_date, asof_date, False):
            continue
        idx = _find_time_index(grid, _time_from_dates(asof_date, cf.pay_date, day_counter))
        cashflow_map.setdefault(idx, []).append(cf)
    change_times, notionals = _callable_notional_schedule(spec, asof_date, day_counter)

    values = np.zeros_like(x_grids[-1], dtype=float)
    for i in range(len(grid) - 1, 0, -1):
        t_from = float(grid[i])
        t_to = float(grid[i - 1])
        grid_i = x_grids[i]
        if i < len(grid) - 1:
            values = _convolution_rollback(
                values,
                zeta_t1=float(zeta_grid[i + 1]),
                zeta_t0=float(zeta_grid[i]),
                mx=mx,
                nx=int(engine_spec.grid_nx),
                y_nodes=y_nodes,
                y_weights=y_weights,
            )
        if i in cashflow_map:
            num = np.asarray(model.numeraire_lgm(t_from, grid_i, eff_discount_curve), dtype=float)
            for cf in cashflow_map[i]:
                values = values + float(cf.amount) / num
        if i in call_map or i in put_map:
            continuation = values.copy()
            if i in call_map:
                ex = call_map[i]
                amt = _callable_price_amount(
                    ex,
                    _callable_notional_at_time(change_times, notionals, t_from),
                    _callable_accrual_at_time(spec, asof_date, day_counter, t_from),
                )
                num = np.asarray(model.numeraire_lgm(t_from, grid_i, eff_discount_curve), dtype=float)
                continuation = np.minimum(continuation, float(amt) / num)
            if i in put_map:
                ex = put_map[i]
                amt = _callable_price_amount(
                    ex,
                    _callable_notional_at_time(change_times, notionals, t_from),
                    _callable_accrual_at_time(spec, asof_date, day_counter, t_from),
                )
                num = np.asarray(model.numeraire_lgm(t_from, grid_i, eff_discount_curve), dtype=float)
                continuation = np.maximum(continuation, float(amt) / num)
            values = continuation
    if len(grid) > 1:
        values = _convolution_rollback(
            values,
            zeta_t1=float(zeta_grid[1]),
            zeta_t0=float(zeta_grid[0]),
            mx=mx,
            nx=int(engine_spec.grid_nx),
            y_nodes=y_nodes,
            y_weights=y_weights,
        )
    return float(values[mx]) / max(float(eff_income_curve(0.0)), 1.0e-18)


def _price_callable_bond_lgm(
    spec: CallableBondTradeSpec,
    engine_spec: CallableBondEngineSpec,
    *,
    asof_date: date,
    day_counter: str,
    reference_curve,
    income_curve,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
    recovery_rate: float,
    security_spread: float,
    model: LGM1F,
) -> dict[str, float | int | str]:
    price = _rollback_callable_bond_lgm_value(
        spec,
        engine_spec,
        asof_date=asof_date,
        day_counter=day_counter,
        reference_curve=reference_curve,
        income_curve=income_curve,
        hazard_times=hazard_times,
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
        security_spread=security_spread,
        model=model,
    )
    stripped = _rollback_callable_bond_lgm_value(
        replace(spec, call_data=(), put_data=()),
        engine_spec,
        asof_date=asof_date,
        day_counter=day_counter,
        reference_curve=reference_curve,
        income_curve=income_curve,
        hazard_times=hazard_times,
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
        security_spread=security_spread,
        model=model,
    )
    maturity_date = max(cf.pay_date for cf in spec.bond.cashflows)
    return {
        "py_npv": float(price),
        "py_settlement_value": float(price),
        "maturity_date": maturity_date.isoformat(),
        "maturity_time": float(_time_from_dates(asof_date, maturity_date, day_counter)),
        "stripped_bond_npv": float(stripped),
        "embedded_option_value": float(stripped - price),
        "call_schedule_count": int(len(spec.call_data)),
        "put_schedule_count": int(len(spec.put_data)),
        "exercise_time_steps_per_year": int(engine_spec.exercise_time_steps_per_year),
        "callable_model_family": str(engine_spec.model_family),
        "callable_engine_variant": str(engine_spec.engine_variant),
    }


def price_bond_trade(
    *,
    ore_xml: Path,
    portfolio_xml: Path,
    trade_id: str,
    asof_date: str,
    model_day_counter: str,
    market_data_file: Path,
    todaysmarket_xml: Path,
    reference_data_path: Path | None,
    pricingengine_path: Path | None,
    flows_csv: Path | None,
) -> dict[str, object]:
    # Entry point used by `ore_snapshot_cli` price-only mode.
    #
    # High-level ORE mapping:
    # 1. parse trade + reference data
    # 2. load engine parameters that influence pricing
    # 3. load credit inputs and security spread from ORE market data
    # 4. build / infer a discount curve
    # 5. price either:
    #    - a risky bond directly, or
    #    - a forward contract on that underlying risky bond
    asof = _parse_any_date(asof_date)
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade = _resolve_trade(portfolio_root, trade_id)
    trade_type = (trade.findtext("./TradeType") or "").strip()
    if trade_type == "CallableBond":
        spec, engine_spec = load_callable_bond_trade_spec(
            portfolio_xml=portfolio_xml,
            trade_id=trade_id,
            reference_data_path=reference_data_path,
            pricingengine_path=pricingengine_path,
        )
    else:
        spec, engine_spec = load_bond_trade_spec(
            portfolio_xml=portfolio_xml,
            trade_id=trade_id,
            reference_data_path=reference_data_path,
            pricingengine_path=pricingengine_path,
            flows_csv=flows_csv,
        )
    try:
        credit = load_ore_default_curve_inputs(str(todaysmarket_xml), str(market_data_file), cpty_name=spec.credit_curve_id)
        hazard_times = np.asarray(credit["hazard_times"], dtype=float)
        hazard_rates = np.asarray(credit["hazard_rates"], dtype=float)
        recovery_rate = float(credit["recovery"])
    except Exception:
        hazard_times = np.asarray([50.0], dtype=float)
        hazard_rates = np.asarray([0.0], dtype=float)
        recovery_rate = 0.0
    security_recovery_key = f"RECOVERY_RATE/RATE/{spec.security_id}"
    with open(market_data_file, encoding="utf-8") as handle:
        for line in handle:
            toks = line.strip().split()
            if len(toks) >= 3 and toks[1] == security_recovery_key:
                recovery_rate = float(toks[2])
                break
    security_spread = _load_security_spread(market_data_file, spec.security_id)
    use_flow_curve = (
        trade_type != "CallableBond"
        and isinstance(spec, BondTradeSpec)
        and
        flows_csv is not None
        and flows_csv.exists()
        and abs(security_spread) <= 1.0e-16
        and np.max(np.abs(hazard_rates)) <= 1.0e-16
    )
    if use_flow_curve:
        base_curve = _curve_from_flow_discounts(
            flows_csv,
            trade_id,
            asof,
            model_day_counter,
            forward_underlying_only=spec.trade_type == "ForwardBond",
        )
    else:
        base_curve = _fit_curve_for_currency(ore_xml, spec.currency)
    income_curve = base_curve
    if trade_type == "CallableBond":
        model = _build_lgm_model_for_callable(
            ore_xml=ore_xml,
            pricingengine_path=pricingengine_path,
            todaysmarket_xml=todaysmarket_xml,
            market_data_file=market_data_file,
            currency=spec.currency,
            maturity_date=max(cf.pay_date for cf in spec.bond.cashflows),
            asof_date=asof,
        )
        result = _price_callable_bond_lgm(
            spec,
            engine_spec,
            asof_date=asof,
            day_counter=model_day_counter,
            reference_curve=base_curve,
            income_curve=income_curve,
            hazard_times=hazard_times,
            hazard_rates=hazard_rates,
            recovery_rate=recovery_rate,
            security_spread=security_spread,
            model=model,
        )
        result.update(
            {
                "trade_type": "CallableBond",
                "currency": spec.currency,
                "security_id": spec.security_id,
                "credit_curve_id": spec.credit_curve_id,
                "reference_curve_id": spec.reference_curve_id,
                "income_curve_id": spec.income_curve_id,
                "security_spread": float(security_spread),
                "recovery_rate": float(recovery_rate),
                "spread_on_income_curve": bool(engine_spec.spread_on_income_curve),
                "settlement_dirty": True,
            }
        )
        return result
    compiled = compile_bond_trade(spec, asof_date=asof, day_counter=model_day_counter, engine_spec=engine_spec)
    underlying_compiled = compiled if compiled.trade_type == "Bond" else replace(compiled, trade_type="Bond")
    spot_grid = build_bond_scenario_grid_numpy(
        underlying_compiled,
        discount_curve=base_curve,
        income_curve=income_curve,
        hazard_times=hazard_times,
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
        security_spread=security_spread,
        engine_spec=engine_spec,
        settlement_time=0.0,
        conditional_on_survival=False,
    )
    value = price_bond_single_numpy(underlying_compiled, spot_grid)
    settlement_value = value
    if spec.trade_type == "ForwardBond":
        # Port of `DiscountingForwardBondEngine::calculate()`:
        # - first compute the underlying bond forward value at the forward
        #   maturity date
        # - then form `(forwardPrice - strikeAmount) * discount`
        # - then apply long/short sign and optional premium cashflow
        _, forward_value = _bond_npv(
            spec,
            asof_date=asof,
            day_counter=model_day_counter,
            discount_curve=base_curve,
            income_curve=income_curve,
            hazard_times=hazard_times,
            hazard_rates=hazard_rates,
            recovery_rate=recovery_rate,
            security_spread=security_spread,
            engine_spec=engine_spec,
            npv_date=spec.forward_maturity_date,
            settlement_date=spec.forward_maturity_date,
            conditional_on_survival=True,
        )
        bond_settlement_date = _adjust_date(spec.forward_maturity_date + timedelta(days=spec.settlement_days), "F", spec.calendar)
        accrued = _accrued_amount(spec, bond_settlement_date)
        eff_settlement_date = bond_settlement_date
        forward_grid = BondScenarioGrid(
            discount_to_pay=spot_grid.discount_to_pay,
            income_to_npv=spot_grid.income_to_npv,
            income_to_settlement=spot_grid.income_to_settlement,
            survival_to_pay=spot_grid.survival_to_pay,
            recovery_discount_mid=spot_grid.recovery_discount_mid,
            recovery_default_prob=spot_grid.recovery_default_prob,
            recovery_rate=spot_grid.recovery_rate,
            forward_dirty_value=np.asarray([float(forward_value)], dtype=float),
            accrued_at_bond_settlement=np.asarray([float(accrued)], dtype=float),
            payoff_discount=np.asarray([float(base_curve(max(_time_from_dates(asof, eff_settlement_date, model_day_counter), 0.0)))], dtype=float),
            premium_discount=(
                None
                if not (spec.compensation_payment and spec.compensation_payment_date > asof)
                else np.asarray([float(base_curve(max(_time_from_dates(asof, spec.compensation_payment_date, model_day_counter), 0.0)))], dtype=float)
            ),
        )
        value = price_bond_single_numpy(compiled, forward_grid)
        settlement_value = value
    maturity_date = max(cf.pay_date for cf in spec.cashflows)
    maturity_time = _time_from_dates(asof, maturity_date, model_day_counter)
    return {
        "trade_type": spec.trade_type,
        "currency": spec.currency,
        "py_npv": float(value),
        "py_settlement_value": float(settlement_value),
        "maturity_date": maturity_date.isoformat(),
        "maturity_time": float(maturity_time),
        "security_id": spec.security_id,
        "credit_curve_id": spec.credit_curve_id,
        "reference_curve_id": spec.reference_curve_id,
        "income_curve_id": spec.income_curve_id,
        "security_spread": float(security_spread),
        "recovery_rate": float(recovery_rate),
        "spread_on_income_curve": bool(engine_spec.spread_on_income_curve),
        "treat_security_spread_as_credit_spread": bool(engine_spec.treat_security_spread_as_credit_spread),
        "settlement_dirty": bool(spec.settlement_dirty),
    }
