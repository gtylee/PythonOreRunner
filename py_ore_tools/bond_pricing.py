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
    _fit_quantlib_helper_eur_nodes,
    _resolve_ore_path,
    build_discount_curve_from_discount_pairs,
    extract_market_instruments_by_currency,
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


@dataclass(frozen=True)
class CompiledCallableCashflowState:
    amount: float
    pay_time: float
    pay_index: int
    belongs_to_underlying_max_time: float
    max_estimation_time: float | None
    exact_estimation_time: float | None
    coupon_start_time: float | None
    coupon_end_time: float | None


@dataclass(frozen=True)
class CompiledCallableBondTrade:
    trade_id: str
    currency: str
    security_id: str
    credit_curve_id: str
    reference_curve_id: str
    income_curve_id: str
    bond_notional: float
    day_counter: str

    stripped_bond: CompiledBondTrade
    engine_spec: CallableBondEngineSpec

    grid_times: np.ndarray
    k_grid: np.ndarray
    call_amounts: np.ndarray
    put_amounts: np.ndarray
    call_active: np.ndarray
    put_active: np.ndarray
    cashflows: tuple[CompiledCallableCashflowState, ...]
    cf_amounts: np.ndarray
    cf_pay_indices: np.ndarray
    cf_belongs_to_underlying_max_time: np.ndarray
    cf_max_estimation_time: np.ndarray
    cf_exact_estimation_time: np.ndarray
    cf_coupon_start_time: np.ndarray
    cf_coupon_end_time: np.ndarray

    center_index: int
    grid_nx: int
    y_nodes: np.ndarray
    y_weights: np.ndarray


@dataclass(frozen=True)
class CallableBondScenarioPack:
    p0_grid: np.ndarray
    h_grid: np.ndarray
    zeta_grid: np.ndarray
    stripped_grid: BondScenarioGrid

    def n_scenarios(self) -> int:
        return int(np.asarray(self.p0_grid, dtype=float).shape[0])

    def slice(self, start: int, end: int) -> "CallableBondScenarioPack":
        return CallableBondScenarioPack(
            p0_grid=np.asarray(self.p0_grid, dtype=float)[start:end],
            h_grid=np.asarray(self.h_grid, dtype=float)[start:end],
            zeta_grid=np.asarray(self.zeta_grid, dtype=float)[start:end],
            stripped_grid=BondScenarioGrid(
                discount_to_pay=np.asarray(self.stripped_grid.discount_to_pay, dtype=float)[start:end],
                income_to_npv=np.asarray(self.stripped_grid.income_to_npv, dtype=float)[start:end],
                income_to_settlement=np.asarray(self.stripped_grid.income_to_settlement, dtype=float)[start:end],
                survival_to_pay=np.asarray(self.stripped_grid.survival_to_pay, dtype=float)[start:end],
                recovery_discount_mid=np.asarray(self.stripped_grid.recovery_discount_mid, dtype=float)[start:end],
                recovery_default_prob=np.asarray(self.stripped_grid.recovery_default_prob, dtype=float)[start:end],
                recovery_rate=np.asarray(self.stripped_grid.recovery_rate, dtype=float)[start:end],
                forward_dirty_value=None if self.stripped_grid.forward_dirty_value is None else np.asarray(self.stripped_grid.forward_dirty_value, dtype=float)[start:end],
                accrued_at_bond_settlement=None if self.stripped_grid.accrued_at_bond_settlement is None else np.asarray(self.stripped_grid.accrued_at_bond_settlement, dtype=float)[start:end],
                payoff_discount=None if self.stripped_grid.payoff_discount is None else np.asarray(self.stripped_grid.payoff_discount, dtype=float)[start:end],
                premium_discount=None if self.stripped_grid.premium_discount is None else np.asarray(self.stripped_grid.premium_discount, dtype=float)[start:end],
            ),
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


def _scenario_curve_list(curves, n_scenarios: int | None, name: str) -> tuple[list[object], int]:
    if callable(curves):
        if n_scenarios is None:
            n_scenarios = 1
        return [curves] * int(n_scenarios), int(n_scenarios)
    seq = list(curves)
    if not seq:
        raise ValueError(f"{name} is empty")
    if n_scenarios is None:
        n_scenarios = len(seq)
    elif len(seq) == 1 and n_scenarios > 1:
        seq = seq * int(n_scenarios)
    elif len(seq) != int(n_scenarios):
        raise ValueError(f"{name} must contain 1 or {n_scenarios} curves, got {len(seq)}")
    return seq, int(n_scenarios)


def _scenario_values(value, n_scenarios: int, name: str, *, allow_none: bool = False) -> np.ndarray | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(int(n_scenarios), float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1d, got shape {arr.shape}")
    if arr.size == 1 and int(n_scenarios) > 1:
        return np.full(int(n_scenarios), float(arr[0]), dtype=float)
    if arr.size != int(n_scenarios):
        raise ValueError(f"{name} must have length {n_scenarios}, got {arr.size}")
    return np.asarray(arr, dtype=float)


def build_bond_scenario_grid_from_scenarios(
    compiled: CompiledBondTrade,
    *,
    discount_curves,
    income_curves=None,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
    recovery_rate: float | np.ndarray,
    security_spread: float | np.ndarray = 0.0,
    engine_spec: BondEngineSpec | None = None,
    npv_time: float = 0.0,
    settlement_time: float = 0.0,
    conditional_on_survival: bool = False,
    forward_dirty_value: float | np.ndarray | None = None,
    accrued_at_bond_settlement: float | np.ndarray | None = None,
    payoff_discount: float | np.ndarray | None = None,
    premium_discount: float | np.ndarray | None = None,
) -> BondScenarioGrid:
    """Sample many scenario curves / hazards onto a compiled trade in one call.

    This is the multi-scenario counterpart to `build_bond_scenario_grid_numpy`.
    It keeps the hot pricing path in NumPy / Torch while allowing scenario
    market data to arrive as Python callables plus hazard vectors.
    """

    engine_spec = engine_spec or compiled.engine_spec
    discount_curve_list, n_scenarios = _scenario_curve_list(discount_curves, None, "discount_curves")
    if income_curves is None:
        income_curve_list = discount_curve_list
    else:
        income_curve_list, _ = _scenario_curve_list(income_curves, n_scenarios, "income_curves")

    hazard_times = np.asarray(hazard_times, dtype=float)
    hazard_rates_arr = np.asarray(hazard_rates, dtype=float)
    if hazard_rates_arr.ndim == 1:
        hazard_rates_arr = np.repeat(hazard_rates_arr.reshape(1, -1), n_scenarios, axis=0)
    elif hazard_rates_arr.ndim == 2:
        if hazard_rates_arr.shape[0] == 1 and n_scenarios > 1:
            hazard_rates_arr = np.repeat(hazard_rates_arr, n_scenarios, axis=0)
        elif hazard_rates_arr.shape[0] != n_scenarios:
            raise ValueError(
                f"hazard_rates must have shape ({n_scenarios}, m) or (m,), got {hazard_rates_arr.shape}"
            )
    else:
        raise ValueError(f"hazard_rates must be 1d or 2d, got shape {hazard_rates_arr.shape}")
    if hazard_rates_arr.shape[1] != hazard_times.size:
        raise ValueError(
            f"hazard_rates tenor axis must match hazard_times, got {hazard_rates_arr.shape[1]} vs {hazard_times.size}"
        )

    recovery_rates = _scenario_values(recovery_rate, n_scenarios, "recovery_rate")
    security_spreads = _scenario_values(security_spread, n_scenarios, "security_spread")

    discount_to_pay = np.zeros((n_scenarios, compiled.pay_times.size), dtype=float)
    survival_to_pay = np.zeros((n_scenarios, compiled.pay_times.size), dtype=float)
    recovery_discount_mid = np.zeros((n_scenarios, compiled.recovery_nominals.size), dtype=float)
    recovery_default_prob = np.zeros((n_scenarios, compiled.recovery_nominals.size), dtype=float)
    income_to_npv = np.zeros(n_scenarios, dtype=float)
    income_to_settlement = np.zeros(n_scenarios, dtype=float)

    for i in range(n_scenarios):
        extra_credit = (
            float(security_spreads[i]) / max(1.0 - float(recovery_rates[i]), 1.0e-12)
            if engine_spec.treat_security_spread_as_credit_spread
            else 0.0
        )
        eff_discount = (
            discount_curve_list[i]
            if engine_spec.treat_security_spread_as_credit_spread
            else _apply_spread(discount_curve_list[i], float(security_spreads[i]))
        )
        eff_income = income_curve_list[i]
        if (
            abs(float(security_spreads[i])) > 1.0e-16
            and not engine_spec.treat_security_spread_as_credit_spread
            and engine_spec.spread_on_income_curve
        ):
            eff_income = _apply_spread(income_curve_list[i], float(security_spreads[i]))
        df_npv = float(eff_income(max(npv_time, 0.0)))
        df_settlement = float(eff_income(max(settlement_time, 0.0)))
        income_to_npv[i] = df_npv
        income_to_settlement[i] = df_settlement
        sp_npv = 1.0 if not conditional_on_survival else float(
            _survival_from_piecewise(
                np.asarray([max(npv_time, 0.0)], dtype=float),
                hazard_times,
                hazard_rates_arr[i],
                extra_credit,
            )[0]
        )
        for j, pay_t in enumerate(compiled.pay_times):
            pay_t_eff = max(float(pay_t), 0.0)
            discount_to_pay[i, j] = float(eff_discount(pay_t_eff)) / max(df_npv, 1.0e-18)
            survival_to_pay[i, j] = float(
                _survival_from_piecewise(
                    np.asarray([pay_t_eff], dtype=float),
                    hazard_times,
                    hazard_rates_arr[i],
                    extra_credit,
                )[0]
            ) / max(sp_npv, 1.0e-18)
        for j, (start_t, end_t) in enumerate(zip(compiled.recovery_start_times, compiled.recovery_end_times)):
            s_eff = max(float(start_t), 0.0)
            e_eff = max(float(end_t), 0.0)
            if e_eff <= s_eff:
                continue
            recovery_default_prob[i, j] = _default_probability(
                s_eff,
                e_eff,
                hazard_times,
                hazard_rates_arr[i],
                extra_credit,
            ) / max(sp_npv, 1.0e-18)
            recovery_discount_mid[i, j] = float(eff_discount(0.5 * (s_eff + e_eff))) / max(df_npv, 1.0e-18)

    return BondScenarioGrid(
        discount_to_pay=discount_to_pay,
        income_to_npv=income_to_npv,
        income_to_settlement=income_to_settlement,
        survival_to_pay=survival_to_pay,
        recovery_discount_mid=recovery_discount_mid,
        recovery_default_prob=recovery_default_prob,
        recovery_rate=np.asarray(recovery_rates, dtype=float),
        forward_dirty_value=_scenario_values(forward_dirty_value, n_scenarios, "forward_dirty_value", allow_none=True),
        accrued_at_bond_settlement=_scenario_values(
            accrued_at_bond_settlement,
            n_scenarios,
            "accrued_at_bond_settlement",
            allow_none=True,
        ),
        payoff_discount=_scenario_values(payoff_discount, n_scenarios, "payoff_discount", allow_none=True),
        premium_discount=_scenario_values(premium_discount, n_scenarios, "premium_discount", allow_none=True),
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


def compile_callable_bond_trade(
    spec: CallableBondTradeSpec,
    engine_spec: CallableBondEngineSpec,
    *,
    asof_date: str | date,
    day_counter: str,
) -> CompiledCallableBondTrade:
    asof = asof_date if isinstance(asof_date, date) else _parse_any_date(str(asof_date))
    pay_times = np.asarray(
        [
            _time_from_dates(asof, cf.pay_date, day_counter)
            for cf in spec.bond.cashflows
            if not _cashflow_has_occurred(cf.pay_date, asof, False)
        ],
        dtype=float,
    )
    exercise_times = np.asarray(
        [
            _time_from_dates(asof, ex.exercise_date, day_counter)
            for ex in spec.call_data + spec.put_data
            if ex.exercise_date > asof
        ],
        dtype=float,
    )
    mandatory_times = np.asarray(sorted(set([float(x) for x in np.concatenate((pay_times, exercise_times)) if float(x) > 0.0])), dtype=float)
    max_time = float(np.max(mandatory_times)) if mandatory_times.size else 0.0
    if engine_spec.engine_variant == "Grid":
        mx = max(int(round(float(engine_spec.grid_sx) * float(engine_spec.grid_nx))), 1)
        grid_nx = int(engine_spec.grid_nx)
        y_nodes, y_weights = _convolution_nodes_and_weights(float(engine_spec.grid_sy), int(engine_spec.grid_ny))
        effective_steps = max(int(engine_spec.exercise_time_steps_per_year), 0)
        grid_max_time = max_time
    else:
        mx = max((int(engine_spec.fd_state_grid_points) - 1) // 2, 1)
        grid_nx = max(int(round(mx / max(6.0, 1.0e-12))), 1)
        y_nodes, y_weights = _convolution_nodes_and_weights(3.0, 10)
        effective_steps = max(int(engine_spec.exercise_time_steps_per_year), int(engine_spec.fd_time_steps_per_year))
        grid_max_time = max(max_time, float(engine_spec.fd_max_time))
    grid = _build_callable_grid_times(mandatory_times, max_time=grid_max_time, time_steps_per_year=effective_steps)
    if grid[-1] > max_time and max_time > 0.0:
        grid = grid[grid <= max_time + 1.0e-12]
        if grid.size == 0 or grid[-1] < max_time:
            grid = np.append(grid, max_time)

    call_active = np.zeros(grid.size, dtype=bool)
    put_active = np.zeros(grid.size, dtype=bool)
    call_amounts = np.zeros(grid.size, dtype=float)
    put_amounts = np.zeros(grid.size, dtype=float)
    change_times, notionals = _callable_notional_schedule(spec, asof, day_counter)
    notionals_on_grid = np.asarray([_callable_notional_at_time(change_times, notionals, float(t)) for t in grid], dtype=float)
    accruals_on_grid = np.asarray([_callable_accrual_at_time(spec, asof, day_counter, float(t)) for t in grid], dtype=float)

    call_map = _register_callable_exercises(
        grid,
        tuple(ex for ex in spec.call_data if ex.exercise_date > asof),
        asof_date=asof,
        day_counter=day_counter,
    )
    put_map = _register_callable_exercises(
        grid,
        tuple(ex for ex in spec.put_data if ex.exercise_date > asof),
        asof_date=asof,
        day_counter=day_counter,
    )
    for idx, ex in call_map.items():
        call_active[idx] = True
        call_amounts[idx] = _callable_price_amount(ex, notionals_on_grid[idx], accruals_on_grid[idx])
    for idx, ex in put_map.items():
        put_active[idx] = True
        put_amounts[idx] = _callable_price_amount(ex, notionals_on_grid[idx], accruals_on_grid[idx])

    cashflows = []
    raw_cashflows = _build_callable_cashflow_states(spec, asof_date=asof, day_counter=day_counter)
    for cf in raw_cashflows:
        cashflows.append(
            CompiledCallableCashflowState(
                amount=float(cf.amount),
                pay_time=float(cf.pay_time),
                pay_index=_find_time_index(grid, float(cf.pay_time)),
                belongs_to_underlying_max_time=float(cf.belongs_to_underlying_max_time),
                max_estimation_time=None if cf.max_estimation_time is None else float(cf.max_estimation_time),
                exact_estimation_time=None if cf.exact_estimation_time is None else float(cf.exact_estimation_time),
                coupon_start_time=None if cf.coupon_start_time is None else float(cf.coupon_start_time),
                coupon_end_time=None if cf.coupon_end_time is None else float(cf.coupon_end_time),
            )
        )
    cf_amounts = np.asarray([float(cf.amount) for cf in raw_cashflows], dtype=float)
    cf_pay_indices = np.asarray([_find_time_index(grid, float(cf.pay_time)) for cf in raw_cashflows], dtype=int)
    cf_belongs_to_underlying_max_time = np.asarray([float(cf.belongs_to_underlying_max_time) for cf in raw_cashflows], dtype=float)
    cf_max_estimation_time = np.asarray(
        [np.nan if cf.max_estimation_time is None else float(cf.max_estimation_time) for cf in raw_cashflows],
        dtype=float,
    )
    cf_exact_estimation_time = np.asarray(
        [np.nan if cf.exact_estimation_time is None else float(cf.exact_estimation_time) for cf in raw_cashflows],
        dtype=float,
    )
    cf_coupon_start_time = np.asarray(
        [np.nan if cf.coupon_start_time is None else float(cf.coupon_start_time) for cf in raw_cashflows],
        dtype=float,
    )
    cf_coupon_end_time = np.asarray(
        [np.nan if cf.coupon_end_time is None else float(cf.coupon_end_time) for cf in raw_cashflows],
        dtype=float,
    )

    stripped_spec = replace(
        spec.bond,
        trade_type="Bond",
    )
    stripped_engine = BondEngineSpec(
        timestep_months=6,
        spread_on_income_curve=engine_spec.spread_on_income_curve,
        treat_security_spread_as_credit_spread=False,
        include_past_cashflows=False,
    )
    return CompiledCallableBondTrade(
        trade_id=spec.trade_id,
        currency=spec.currency,
        security_id=spec.security_id,
        credit_curve_id=spec.credit_curve_id,
        reference_curve_id=spec.reference_curve_id,
        income_curve_id=spec.income_curve_id,
        bond_notional=float(spec.bond_notional),
        day_counter=str(day_counter),
        stripped_bond=compile_bond_trade(stripped_spec, asof_date=asof, day_counter=day_counter, engine_spec=stripped_engine),
        engine_spec=engine_spec,
        grid_times=np.asarray(grid, dtype=float),
        k_grid=(np.arange(2 * mx + 1, dtype=float) - mx),
        call_amounts=call_amounts,
        put_amounts=put_amounts,
        call_active=call_active,
        put_active=put_active,
        cashflows=tuple(cashflows),
        cf_amounts=cf_amounts,
        cf_pay_indices=cf_pay_indices,
        cf_belongs_to_underlying_max_time=cf_belongs_to_underlying_max_time,
        cf_max_estimation_time=cf_max_estimation_time,
        cf_exact_estimation_time=cf_exact_estimation_time,
        cf_coupon_start_time=cf_coupon_start_time,
        cf_coupon_end_time=cf_coupon_end_time,
        center_index=mx,
        grid_nx=grid_nx,
        y_nodes=np.asarray(y_nodes, dtype=float),
        y_weights=np.asarray(y_weights, dtype=float),
    )


def _scenario_model_list(models, n_scenarios: int | None) -> tuple[list[LGM1F], int]:
    if isinstance(models, (LGM1F, LGMParams, dict)):
        if n_scenarios is None:
            n_scenarios = 1
        models = [models] * int(n_scenarios)
    seq = list(models)
    if not seq:
        raise ValueError("models is empty")
    if n_scenarios is None:
        n_scenarios = len(seq)
    elif len(seq) == 1 and int(n_scenarios) > 1:
        seq = seq * int(n_scenarios)
    elif len(seq) != int(n_scenarios):
        raise ValueError(f"models must contain 1 or {n_scenarios} entries, got {len(seq)}")

    out: list[LGM1F] = []
    for item in seq:
        if isinstance(item, LGM1F):
            out.append(item)
            continue
        if isinstance(item, LGMParams):
            out.append(LGM1F(item))
            continue
        if isinstance(item, dict):
            out.append(
                LGM1F(
                    LGMParams(
                        alpha_times=tuple(float(x) for x in np.asarray(item.get("alpha_times", []), dtype=float)),
                        alpha_values=tuple(float(x) for x in np.asarray(item.get("alpha_values", [0.01]), dtype=float)),
                        kappa_times=tuple(float(x) for x in np.asarray(item.get("kappa_times", []), dtype=float)),
                        kappa_values=tuple(float(x) for x in np.asarray(item.get("kappa_values", [0.03]), dtype=float)),
                        shift=float(item.get("shift", 0.0)),
                        scaling=float(item.get("scaling", 1.0)),
                    )
                )
            )
            continue
        raise TypeError(f"unsupported LGM model payload type '{type(item).__name__}'")
    return out, int(n_scenarios)


def build_callable_bond_scenario_pack(
    compiled: CompiledCallableBondTrade,
    *,
    reference_curves,
    models,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
    recovery_rate: float | np.ndarray,
    security_spread: float | np.ndarray = 0.0,
    stripped_discount_curves=None,
    stripped_income_curves=None,
) -> CallableBondScenarioPack:
    reference_curve_list, n_scenarios = _scenario_curve_list(reference_curves, None, "reference_curves")
    model_list, n_scenarios = _scenario_model_list(models, n_scenarios)
    hazard_times = np.asarray(hazard_times, dtype=float)
    hazard_rates_arr = np.asarray(hazard_rates, dtype=float)
    if hazard_rates_arr.ndim == 1:
        hazard_rates_arr = np.repeat(hazard_rates_arr.reshape(1, -1), n_scenarios, axis=0)
    elif hazard_rates_arr.ndim == 2:
        if hazard_rates_arr.shape[0] == 1 and n_scenarios > 1:
            hazard_rates_arr = np.repeat(hazard_rates_arr, n_scenarios, axis=0)
        elif hazard_rates_arr.shape[0] != n_scenarios:
            raise ValueError(f"hazard_rates must have shape ({n_scenarios}, m) or (m,), got {hazard_rates_arr.shape}")
    else:
        raise ValueError(f"hazard_rates must be 1d or 2d, got shape {hazard_rates_arr.shape}")
    if hazard_rates_arr.shape[1] != hazard_times.size:
        raise ValueError(f"hazard_rates tenor axis must match hazard_times, got {hazard_rates_arr.shape[1]} vs {hazard_times.size}")

    recovery_rates = _scenario_values(recovery_rate, n_scenarios, "recovery_rate")
    security_spreads = _scenario_values(security_spread, n_scenarios, "security_spread")
    p0_grid = np.zeros((n_scenarios, compiled.grid_times.size), dtype=float)
    h_grid = np.zeros((n_scenarios, compiled.grid_times.size), dtype=float)
    zeta_grid = np.zeros((n_scenarios, compiled.grid_times.size), dtype=float)

    for i, (curve, model) in enumerate(zip(reference_curve_list, model_list)):
        eff_discount_curve = _effective_bond_discount_curve(
            curve,
            hazard_times,
            hazard_rates_arr[i],
            float(recovery_rates[i]),
            float(security_spreads[i]),
        )
        p0_grid[i, :] = np.asarray([float(eff_discount_curve(float(t))) for t in compiled.grid_times], dtype=float)
        h_grid[i, :] = np.asarray(model.H(compiled.grid_times), dtype=float)
        zeta_grid[i, :] = np.asarray(model.zeta(compiled.grid_times), dtype=float)

    stripped_discount_curves = reference_curves if stripped_discount_curves is None else stripped_discount_curves
    stripped_income_curves = stripped_discount_curves if stripped_income_curves is None else stripped_income_curves
    stripped_grid = build_bond_scenario_grid_from_scenarios(
        compiled.stripped_bond,
        discount_curves=stripped_discount_curves,
        income_curves=stripped_income_curves,
        hazard_times=hazard_times,
        hazard_rates=hazard_rates_arr,
        recovery_rate=recovery_rates,
        security_spread=security_spreads,
        engine_spec=compiled.stripped_bond.engine_spec,
    )
    return CallableBondScenarioPack(
        p0_grid=p0_grid,
        h_grid=h_grid,
        zeta_grid=zeta_grid,
        stripped_grid=stripped_grid,
    )


def build_callable_bond_scenario_pack_from_arrays(
    compiled: CompiledCallableBondTrade,
    *,
    p0_grid: np.ndarray,
    h_grid: np.ndarray,
    zeta_grid: np.ndarray,
    stripped_grid: BondScenarioGrid,
) -> CallableBondScenarioPack:
    pack = CallableBondScenarioPack(
        p0_grid=np.asarray(p0_grid, dtype=float),
        h_grid=np.asarray(h_grid, dtype=float),
        zeta_grid=np.asarray(zeta_grid, dtype=float),
        stripped_grid=stripped_grid,
    )
    validate_callable_bond_scenario_pack(compiled, pack)
    return pack


def validate_callable_bond_scenario_pack(compiled: CompiledCallableBondTrade, pack: CallableBondScenarioPack) -> None:
    p0_grid = _ensure_2d_float("p0_grid", pack.p0_grid)
    h_grid = _ensure_2d_float("h_grid", pack.h_grid)
    zeta_grid = _ensure_2d_float("zeta_grid", pack.zeta_grid)
    expected = (pack.n_scenarios(), compiled.grid_times.size)
    for name, arr in (("p0_grid", p0_grid), ("h_grid", h_grid), ("zeta_grid", zeta_grid)):
        if arr.shape != expected:
            raise ValueError(f"{name} expected shape {expected}, got {arr.shape}")
    validate_bond_scenario_grid(compiled.stripped_bond, pack.stripped_grid)
    if pack.stripped_grid.n_scenarios() != pack.n_scenarios():
        raise ValueError("stripped_grid scenario count must match callable scenario count")


def _callable_coupon_ratio_compiled(cf: CompiledCallableCashflowState, t: float) -> float:
    if cf.coupon_start_time is None or cf.coupon_end_time is None:
        return 1.0
    denom = float(cf.coupon_end_time) - float(cf.coupon_start_time)
    if denom <= 1.0e-18:
        return 0.0
    return max(0.0, min(1.0, (float(cf.coupon_end_time) - float(t)) / denom))


def _callable_coupon_ratio_from_arrays(start_t: float, end_t: float, t: float) -> float:
    if np.isnan(start_t) or np.isnan(end_t):
        return 1.0
    denom = float(end_t) - float(start_t)
    if denom <= 1.0e-18:
        return 0.0
    return max(0.0, min(1.0, (float(end_t) - float(t)) / denom))


def _convolution_rollback_batch_numpy(
    values: np.ndarray,
    *,
    zeta_t1: np.ndarray,
    zeta_t0: np.ndarray,
    mx: int,
    nx: int,
    y_nodes: np.ndarray,
    y_weights: np.ndarray,
) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = v.copy()
    z1 = np.asarray(zeta_t1, dtype=float).reshape(-1)
    z0 = np.asarray(zeta_t0, dtype=float).reshape(-1)
    if v.ndim != 2 or v.shape[0] != z1.size or z0.size != z1.size:
        raise ValueError("batch rollback shape mismatch")

    last = 2 * mx
    changed = np.abs(z1 - z0) > 1.0e-18
    if not np.any(changed):
        return out

    idx = np.nonzero(changed)[0]
    vv = v[idx]
    zz1 = np.maximum(z1[idx], 0.0)
    zz0 = np.maximum(z0[idx], 0.0)
    sigma = np.sqrt(zz1)
    dx = sigma / float(nx)

    zero_mask = zz0 <= 1.0e-18
    if np.any(zero_mask):
        zi = idx[zero_mask]
        kp = y_nodes[None, :] * (sigma[zero_mask] / dx[zero_mask])[:, None] + mx
        kk = np.floor(kp).astype(np.int64, copy=False)
        alpha = kp - kk
        beta = 1.0 - alpha
        left = np.clip(kk, 0, last)
        right = np.clip(kk + 1, 0, last)
        interp = alpha * np.take_along_axis(v[zi], right, axis=1) + beta * np.take_along_axis(v[zi], left, axis=1)
        acc = interp @ y_weights
        out[zi, :] = acc[:, None]

    pos_mask = ~zero_mask
    if np.any(pos_mask):
        zi = idx[pos_mask]
        std = np.sqrt(np.maximum(zz1[pos_mask] - zz0[pos_mask], 0.0))
        dx2 = np.sqrt(zz0[pos_mask]) / float(nx)
        k_grid = (np.arange(last + 1, dtype=float) - mx)[None, :, None]
        kp = ((dx2[:, None, None] * k_grid) + (std[:, None, None] * y_nodes[None, None, :])) / dx[pos_mask][:, None, None] + mx
        kk = np.floor(kp).astype(np.int64, copy=False)
        alpha = kp - kk
        beta = 1.0 - alpha
        left = np.clip(kk, 0, last)
        right = np.clip(kk + 1, 0, last)
        base = v[zi][:, :, None]
        interp = alpha * np.take_along_axis(base, right, axis=1) + beta * np.take_along_axis(base, left, axis=1)
        out[zi, :] = np.tensordot(interp, y_weights, axes=([2], [0]))
    return out


def _convolution_rollback_batch_torch(
    values,
    *,
    zeta_t1,
    zeta_t0,
    mx: int,
    nx: int,
    y_nodes,
    y_weights,
):
    if torch is None:
        raise ImportError("torch is required for _convolution_rollback_batch_torch()")
    v = values
    out = v.clone()
    changed = torch.abs(zeta_t1 - zeta_t0) > 1.0e-18
    if not torch.any(changed):
        return out

    last = 2 * mx
    idx = torch.nonzero(changed, as_tuple=False).squeeze(1)
    vv = v.index_select(0, idx)
    zz1 = torch.clamp(zeta_t1.index_select(0, idx), min=0.0)
    zz0 = torch.clamp(zeta_t0.index_select(0, idx), min=0.0)
    sigma = torch.sqrt(zz1)
    dx = sigma / float(nx)

    zero_mask = zz0 <= 1.0e-18
    if torch.any(zero_mask):
        zi = idx[zero_mask]
        kp = y_nodes.unsqueeze(0) * (sigma[zero_mask] / dx[zero_mask]).unsqueeze(1) + float(mx)
        kk = torch.floor(kp).to(torch.long)
        alpha = kp - kk.to(kp.dtype)
        beta = 1.0 - alpha
        left = torch.clamp(kk, 0, last)
        right = torch.clamp(kk + 1, 0, last)
        base = v.index_select(0, zi)
        interp = alpha * torch.gather(base, 1, right) + beta * torch.gather(base, 1, left)
        acc = interp @ y_weights
        out[zi, :] = acc.unsqueeze(1).expand(-1, last + 1)

    pos_mask = ~zero_mask
    if torch.any(pos_mask):
        zi = idx[pos_mask]
        std = torch.sqrt(torch.clamp(zz1[pos_mask] - zz0[pos_mask], min=0.0))
        dx2 = torch.sqrt(zz0[pos_mask]) / float(nx)
        k_grid = (torch.arange(last + 1, dtype=v.dtype, device=v.device) - float(mx)).view(1, last + 1, 1)
        kp = ((dx2.view(-1, 1, 1) * k_grid) + (std.view(-1, 1, 1) * y_nodes.view(1, 1, -1))) / dx[pos_mask].view(-1, 1, 1) + float(mx)
        kk = torch.floor(kp).to(torch.long)
        alpha = kp - kk.to(kp.dtype)
        beta = 1.0 - alpha
        left = torch.clamp(kk, 0, last)
        right = torch.clamp(kk + 1, 0, last)
        base = v.index_select(0, zi).unsqueeze(2).expand(-1, -1, y_nodes.numel())
        interp = alpha * torch.gather(base, 1, right) + beta * torch.gather(base, 1, left)
        out[zi, :] = torch.matmul(interp, y_weights)
    return out


def _price_callable_bond_scenarios_numpy_chunk(compiled: CompiledCallableBondTrade, pack: CallableBondScenarioPack) -> np.ndarray:
    n_scenarios = pack.n_scenarios()
    n_states = 2 * compiled.center_index + 1
    option_values = np.zeros((n_scenarios, n_states), dtype=float)
    underlying_npv = np.zeros((n_scenarios, n_states), dtype=float)
    provisional_npv = np.zeros((n_scenarios, n_states), dtype=float)
    cf_cache: list[np.ndarray | None] = [None] * len(compiled.cashflows)
    cf_status = ["Open"] * len(compiled.cashflows)
    stripped = price_bond_scenarios_numpy(compiled.stripped_bond, pack.stripped_grid)

    for i in range(compiled.grid_times.size - 1, 0, -1):
        t_from = float(compiled.grid_times[i])
        z_t = np.asarray(pack.zeta_grid[:, i], dtype=float)
        h_t = np.asarray(pack.h_grid[:, i], dtype=float)
        p0_t = np.asarray(pack.p0_grid[:, i], dtype=float)
        dx = np.sqrt(np.maximum(z_t, 0.0)) / float(max(compiled.grid_nx, 1))
        x_grid = dx[:, None] * compiled.k_grid[None, :]
        reduced_cache: dict[int, np.ndarray] = {}
        if i < compiled.grid_times.size - 1:
            option_values = _convolution_rollback_batch_numpy(
                option_values,
                zeta_t1=pack.zeta_grid[:, i + 1],
                zeta_t0=pack.zeta_grid[:, i],
                mx=compiled.center_index,
                nx=compiled.grid_nx,
                y_nodes=compiled.y_nodes,
                y_weights=compiled.y_weights,
            )
            underlying_npv = _convolution_rollback_batch_numpy(
                underlying_npv,
                zeta_t1=pack.zeta_grid[:, i + 1],
                zeta_t0=pack.zeta_grid[:, i],
                mx=compiled.center_index,
                nx=compiled.grid_nx,
                y_nodes=compiled.y_nodes,
                y_weights=compiled.y_weights,
            )
            if i == 1:
                provisional_npv = _convolution_rollback_batch_numpy(
                    provisional_npv,
                    zeta_t1=pack.zeta_grid[:, i + 1],
                    zeta_t0=pack.zeta_grid[:, i],
                    mx=compiled.center_index,
                    nx=compiled.grid_nx,
                    y_nodes=compiled.y_nodes,
                    y_weights=compiled.y_weights,
                )
            for j, cached in enumerate(cf_cache):
                if cached is None:
                    continue
                cf_cache[j] = _convolution_rollback_batch_numpy(
                    cached,
                    zeta_t1=pack.zeta_grid[:, i + 1],
                    zeta_t0=pack.zeta_grid[:, i],
                    mx=compiled.center_index,
                    nx=compiled.grid_nx,
                    y_nodes=compiled.y_nodes,
                    y_weights=compiled.y_weights,
                )

        provisional_npv = np.zeros_like(provisional_npv)
        for j in range(compiled.cf_amounts.size):
            if cf_status[j] == "Done":
                continue
            belongs_to_underlying = (
                t_from < float(compiled.cf_belongs_to_underlying_max_time[j])
                or _callable_close_enough(t_from, float(compiled.cf_belongs_to_underlying_max_time[j]))
            )
            coupon_ratio = _callable_coupon_ratio_from_arrays(
                float(compiled.cf_coupon_start_time[j]),
                float(compiled.cf_coupon_end_time[j]),
                t_from,
            )
            if belongs_to_underlying and coupon_ratio > 0.0:
                if cf_status[j] == "Cached":
                    underlying_npv = underlying_npv + np.asarray(cf_cache[j], dtype=float)
                    cf_cache[j] = None
                    cf_status[j] = "Done"
                elif (
                    not np.isnan(compiled.cf_max_estimation_time[j])
                    and (t_from < float(compiled.cf_max_estimation_time[j]) or _callable_close_enough(t_from, float(compiled.cf_max_estimation_time[j])))
                ):
                    pay_idx = int(compiled.cf_pay_indices[j])
                    reduced = reduced_cache.get(pay_idx)
                    if reduced is None:
                        h_pay = np.asarray(pack.h_grid[:, pay_idx], dtype=float)
                        p0_pay = np.asarray(pack.p0_grid[:, pay_idx], dtype=float)
                        reduced = p0_pay[:, None] * np.exp(
                            -h_pay[:, None] * x_grid - 0.5 * (h_pay * h_pay * z_t)[:, None]
                        )
                        reduced_cache[pay_idx] = reduced
                    reduced = float(compiled.cf_amounts[j]) * reduced
                    underlying_npv = underlying_npv + reduced
                    cf_status[j] = "Done"
                else:
                    pay_idx = int(compiled.cf_pay_indices[j])
                    reduced = reduced_cache.get(pay_idx)
                    if reduced is None:
                        h_pay = np.asarray(pack.h_grid[:, pay_idx], dtype=float)
                        p0_pay = np.asarray(pack.p0_grid[:, pay_idx], dtype=float)
                        reduced = p0_pay[:, None] * np.exp(
                            -h_pay[:, None] * x_grid - 0.5 * (h_pay * h_pay * z_t)[:, None]
                        )
                        reduced_cache[pay_idx] = reduced
                    provisional_npv = provisional_npv + float(compiled.cf_amounts[j]) * reduced
            elif (
                np.isnan(compiled.cf_max_estimation_time[j])
                and not np.isnan(compiled.cf_exact_estimation_time[j])
                and (t_from < float(compiled.cf_exact_estimation_time[j]) or _callable_close_enough(t_from, float(compiled.cf_exact_estimation_time[j])))
                and cf_status[j] == "Open"
            ):
                pay_idx = int(compiled.cf_pay_indices[j])
                reduced = reduced_cache.get(pay_idx)
                if reduced is None:
                    h_pay = np.asarray(pack.h_grid[:, pay_idx], dtype=float)
                    p0_pay = np.asarray(pack.p0_grid[:, pay_idx], dtype=float)
                    reduced = p0_pay[:, None] * np.exp(
                        -h_pay[:, None] * x_grid - 0.5 * (h_pay * h_pay * z_t)[:, None]
                    )
                    reduced_cache[pay_idx] = reduced
                cf_cache[j] = float(compiled.cf_amounts[j]) * reduced
                cf_status[j] = "Cached"
        if compiled.call_active[i] or compiled.put_active[i]:
            continuation = option_values.copy()
            amount_over_num = p0_t[:, None] * np.exp(-h_t[:, None] * x_grid - 0.5 * (h_t * h_t * z_t)[:, None])
            if compiled.call_active[i]:
                continuation = np.minimum(continuation, float(compiled.call_amounts[i]) * amount_over_num - (underlying_npv + provisional_npv))
            if compiled.put_active[i]:
                continuation = np.maximum(continuation, float(compiled.put_amounts[i]) * amount_over_num - underlying_npv + provisional_npv)
            option_values = continuation
    if compiled.grid_times.size > 1:
        option_values = _convolution_rollback_batch_numpy(
            option_values,
            zeta_t1=pack.zeta_grid[:, 1],
            zeta_t0=pack.zeta_grid[:, 0],
            mx=compiled.center_index,
            nx=compiled.grid_nx,
            y_nodes=compiled.y_nodes,
            y_weights=compiled.y_weights,
        )
    option_value = option_values[:, compiled.center_index]
    return np.asarray(stripped + option_value, dtype=float)


def price_callable_bond_scenarios_numpy(
    compiled: CompiledCallableBondTrade,
    pack: CallableBondScenarioPack,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    validate_callable_bond_scenario_pack(compiled, pack)
    n_scenarios = pack.n_scenarios()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if n_scenarios <= chunk_size:
        return _price_callable_bond_scenarios_numpy_chunk(compiled, pack)
    out = []
    for start in range(0, n_scenarios, chunk_size):
        end = min(start + chunk_size, n_scenarios)
        out.append(_price_callable_bond_scenarios_numpy_chunk(compiled, pack.slice(start, end)))
    return np.concatenate(out, axis=0)


def _price_callable_bond_scenarios_torch_chunk(
    compiled: CompiledCallableBondTrade,
    pack: CallableBondScenarioPack,
    *,
    device: str,
):
    if torch is None:
        raise ImportError("torch is required for price_callable_bond_scenarios_torch()")
    target = torch.device(device)
    dtype = torch.float32 if target.type == "mps" else torch.float64
    n_scenarios = pack.n_scenarios()
    n_states = 2 * compiled.center_index + 1
    option_values = torch.zeros((n_scenarios, n_states), dtype=dtype, device=target)
    underlying_npv = torch.zeros((n_scenarios, n_states), dtype=dtype, device=target)
    provisional_npv = torch.zeros((n_scenarios, n_states), dtype=dtype, device=target)
    cf_cache: list[object | None] = [None] * len(compiled.cashflows)
    cf_status = ["Open"] * len(compiled.cashflows)
    p0_grid = torch.as_tensor(pack.p0_grid, dtype=dtype, device=target)
    h_grid = torch.as_tensor(pack.h_grid, dtype=dtype, device=target)
    zeta_grid = torch.as_tensor(pack.zeta_grid, dtype=dtype, device=target)
    k_grid = torch.as_tensor(compiled.k_grid, dtype=dtype, device=target)
    y_nodes = torch.as_tensor(compiled.y_nodes, dtype=dtype, device=target)
    y_weights = torch.as_tensor(compiled.y_weights, dtype=dtype, device=target)
    stripped = price_bond_scenarios_torch(compiled.stripped_bond, pack.stripped_grid, device=device)

    for i in range(compiled.grid_times.size - 1, 0, -1):
        t_from = float(compiled.grid_times[i])
        z_t = zeta_grid[:, i]
        h_t = h_grid[:, i]
        p0_t = p0_grid[:, i]
        dx = torch.sqrt(torch.clamp(z_t, min=0.0)) / float(max(compiled.grid_nx, 1))
        x_grid = dx.unsqueeze(1) * k_grid.unsqueeze(0)
        reduced_cache: dict[int, object] = {}
        if i < compiled.grid_times.size - 1:
            option_values = _convolution_rollback_batch_torch(
                option_values,
                zeta_t1=zeta_grid[:, i + 1],
                zeta_t0=zeta_grid[:, i],
                mx=compiled.center_index,
                nx=compiled.grid_nx,
                y_nodes=y_nodes,
                y_weights=y_weights,
            )
            underlying_npv = _convolution_rollback_batch_torch(
                underlying_npv,
                zeta_t1=zeta_grid[:, i + 1],
                zeta_t0=zeta_grid[:, i],
                mx=compiled.center_index,
                nx=compiled.grid_nx,
                y_nodes=y_nodes,
                y_weights=y_weights,
            )
            if i == 1:
                provisional_npv = _convolution_rollback_batch_torch(
                    provisional_npv,
                    zeta_t1=zeta_grid[:, i + 1],
                    zeta_t0=zeta_grid[:, i],
                    mx=compiled.center_index,
                    nx=compiled.grid_nx,
                    y_nodes=y_nodes,
                    y_weights=y_weights,
                )
            for j, cached in enumerate(cf_cache):
                if cached is None:
                    continue
                cf_cache[j] = _convolution_rollback_batch_torch(
                    cached,
                    zeta_t1=zeta_grid[:, i + 1],
                    zeta_t0=zeta_grid[:, i],
                    mx=compiled.center_index,
                    nx=compiled.grid_nx,
                    y_nodes=y_nodes,
                    y_weights=y_weights,
                )
        provisional_npv = torch.zeros_like(provisional_npv)
        for j in range(compiled.cf_amounts.size):
            if cf_status[j] == "Done":
                continue
            belongs_to_underlying = (
                t_from < float(compiled.cf_belongs_to_underlying_max_time[j])
                or _callable_close_enough(t_from, float(compiled.cf_belongs_to_underlying_max_time[j]))
            )
            coupon_ratio = _callable_coupon_ratio_from_arrays(
                float(compiled.cf_coupon_start_time[j]),
                float(compiled.cf_coupon_end_time[j]),
                t_from,
            )
            if belongs_to_underlying and coupon_ratio > 0.0:
                if cf_status[j] == "Cached":
                    underlying_npv = underlying_npv + cf_cache[j]
                    cf_cache[j] = None
                    cf_status[j] = "Done"
                elif (
                    not np.isnan(compiled.cf_max_estimation_time[j])
                    and (t_from < float(compiled.cf_max_estimation_time[j]) or _callable_close_enough(t_from, float(compiled.cf_max_estimation_time[j])))
                ):
                    pay_idx = int(compiled.cf_pay_indices[j])
                    reduced = reduced_cache.get(pay_idx)
                    if reduced is None:
                        h_pay = h_grid[:, pay_idx]
                        p0_pay = p0_grid[:, pay_idx]
                        reduced = p0_pay.unsqueeze(1) * torch.exp(
                            -h_pay.unsqueeze(1) * x_grid - 0.5 * (h_pay * h_pay * z_t).unsqueeze(1)
                        )
                        reduced_cache[pay_idx] = reduced
                    reduced = float(compiled.cf_amounts[j]) * reduced
                    underlying_npv = underlying_npv + reduced
                    cf_status[j] = "Done"
                else:
                    pay_idx = int(compiled.cf_pay_indices[j])
                    reduced = reduced_cache.get(pay_idx)
                    if reduced is None:
                        h_pay = h_grid[:, pay_idx]
                        p0_pay = p0_grid[:, pay_idx]
                        reduced = p0_pay.unsqueeze(1) * torch.exp(
                            -h_pay.unsqueeze(1) * x_grid - 0.5 * (h_pay * h_pay * z_t).unsqueeze(1)
                        )
                        reduced_cache[pay_idx] = reduced
                    provisional_npv = provisional_npv + float(compiled.cf_amounts[j]) * reduced
            elif (
                np.isnan(compiled.cf_max_estimation_time[j])
                and not np.isnan(compiled.cf_exact_estimation_time[j])
                and (t_from < float(compiled.cf_exact_estimation_time[j]) or _callable_close_enough(t_from, float(compiled.cf_exact_estimation_time[j])))
                and cf_status[j] == "Open"
            ):
                pay_idx = int(compiled.cf_pay_indices[j])
                reduced = reduced_cache.get(pay_idx)
                if reduced is None:
                    h_pay = h_grid[:, pay_idx]
                    p0_pay = p0_grid[:, pay_idx]
                    reduced = p0_pay.unsqueeze(1) * torch.exp(
                        -h_pay.unsqueeze(1) * x_grid - 0.5 * (h_pay * h_pay * z_t).unsqueeze(1)
                    )
                    reduced_cache[pay_idx] = reduced
                cf_cache[j] = float(compiled.cf_amounts[j]) * reduced
                cf_status[j] = "Cached"
        if bool(compiled.call_active[i]) or bool(compiled.put_active[i]):
            continuation = option_values.clone()
            amount_over_num = p0_t.unsqueeze(1) * torch.exp(-h_t.unsqueeze(1) * x_grid - 0.5 * (h_t * h_t * z_t).unsqueeze(1))
            if bool(compiled.call_active[i]):
                continuation = torch.minimum(continuation, float(compiled.call_amounts[i]) * amount_over_num - (underlying_npv + provisional_npv))
            if bool(compiled.put_active[i]):
                continuation = torch.maximum(continuation, float(compiled.put_amounts[i]) * amount_over_num - underlying_npv + provisional_npv)
            option_values = continuation
    if compiled.grid_times.size > 1:
        option_values = _convolution_rollback_batch_torch(
            option_values,
            zeta_t1=zeta_grid[:, 1],
            zeta_t0=zeta_grid[:, 0],
            mx=compiled.center_index,
            nx=compiled.grid_nx,
            y_nodes=y_nodes,
            y_weights=y_weights,
        )
    option_value = option_values[:, compiled.center_index]
    return stripped + option_value


def price_callable_bond_scenarios_torch(
    compiled: CompiledCallableBondTrade,
    pack: CallableBondScenarioPack,
    *,
    device: str = "cpu",
    chunk_size: int = 512,
):
    if torch is None:
        raise ImportError("torch is required for price_callable_bond_scenarios_torch()")
    validate_callable_bond_scenario_pack(compiled, pack)
    n_scenarios = pack.n_scenarios()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if n_scenarios <= chunk_size:
        return _price_callable_bond_scenarios_torch_chunk(compiled, pack, device=device)
    chunks = []
    for start in range(0, n_scenarios, chunk_size):
        end = min(start + chunk_size, n_scenarios)
        chunks.append(_price_callable_bond_scenarios_torch_chunk(compiled, pack.slice(start, end), device=device))
    return torch.cat(chunks, dim=0)


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
    if ref_node is None and "_FWDEXP_" in security_id:
        base_security_id = security_id.split("_FWDEXP_", 1)[0].strip()
        if base_security_id:
            ref_node = rd_root.find(f"./ReferenceDatum[@id='{base_security_id}']/BondReferenceData")
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
        scale = float((bond_node.findtext("./BondNotional") or "1").strip() or "1")
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
            bond_notional=scale,
            cashflows=_load_bond_cashflows_from_flows(flows_csv, trade_id) if flows_csv and flows_csv.exists() else _scale_cashflows(cashflows, scale),
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
        scale = float((bond_node.findtext("./BondNotional") or "1").strip() or "1")
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
            bond_notional=scale,
            cashflows=_load_bond_cashflows_from_flows(flows_csv, trade_id, forward_underlying_only=True) if flows_csv and flows_csv.exists() else _scale_cashflows(parsed_cashflows, scale),
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


def _resolve_case_output_dir_local(ore_xml: Path) -> Path:
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.resolve().parent
    run_dir = base.parent
    return (run_dir / setup_params.get("outputPath", "Output")).resolve()


def _expected_output_root_local(ore_xml: Path) -> Path | None:
    root = ore_xml.resolve().parents[1] / "ExpectedOutput"
    return root if root.exists() else None


def _expected_output_variant_dir_local(ore_xml: Path) -> Path | None:
    root = _expected_output_root_local(ore_xml)
    if root is None:
        return None
    case_dir = ore_xml.resolve().parents[1]
    stem = ore_xml.stem
    if case_dir.name == "Example_10":
        mapping = {
            "ore.xml": None,
            "ore_iah_0.xml": "collateral_iah_0",
            "ore_iah_1.xml": "collateral_iah_1",
            "ore_mpor.xml": "collateral_mpor",
            "ore_mta.xml": "collateral_mta",
            "ore_threshold.xml": "collateral_threshold",
            "ore_threshold_break.xml": "collateral_threshold_break",
            "ore_threshold_dim.xml": "collateral_threshold_dim",
        }
        target = mapping.get(ore_xml.name)
        return None if target is None else root / target
    if case_dir.name == "Example_31":
        mapping = {
            "ore.xml": None,
            "ore_mpor.xml": "collateral_mpor",
            "ore_dim.xml": "collateral_dim",
            "ore_ddv.xml": "collateral_ddv",
        }
        target = mapping.get(ore_xml.name)
        return None if target is None else root / target
    if case_dir.name == "Example_13":
        if stem.startswith("ore_A"):
            return root / "case_A_eur_swap"
        if stem.startswith("ore_B"):
            return root / "case_B_eur_swaption"
        if stem.startswith("ore_C"):
            return root / "case_C_usd_swap"
        if stem.startswith("ore_D"):
            return root / "case_D_eurusd_swap"
        if stem.startswith("ore_E"):
            return root / "case_E_fxopt"
    return None


def _reference_filename_candidates_local(ore_xml: Path, filename: str) -> list[str]:
    candidates = [filename]
    stem = ore_xml.stem
    variant = stem[4:] if stem.startswith("ore_") else stem
    if not variant:
        return candidates
    name = Path(filename).stem
    suffix = Path(filename).suffix
    variants = [variant]
    if "_" in variant:
        head = variant.split("_", 1)[0]
        if head and head not in variants:
            variants.append(head)
    for item in variants:
        candidate = f"{name}_{item}{suffix}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _reference_output_dirs_local(ore_xml: Path) -> list[Path]:
    dirs: list[Path] = []
    actual = _resolve_case_output_dir_local(ore_xml)
    if actual.exists():
        dirs.append(actual)
    variant = _expected_output_variant_dir_local(ore_xml)
    if variant is not None and variant.exists() and variant not in dirs:
        dirs.append(variant)
    root = _expected_output_root_local(ore_xml)
    if root is not None and root.exists() and root not in dirs:
        dirs.append(root)
    return dirs


def _find_reference_output_file_local(ore_xml: Path, filename: str) -> Path | None:
    for directory in _reference_output_dirs_local(ore_xml):
        for candidate in _reference_filename_candidates_local(ore_xml, filename):
            path = directory / candidate
            if path.exists() and path.is_file():
                return path
    return None


def _resolve_curve_column_from_id(todaysmarket_xml: Path, curve_id: str) -> str:
    from py_ore_tools import ore_snapshot as ore_snapshot_mod

    root = ET.parse(todaysmarket_xml).getroot()
    curve_id = (curve_id or "").strip()
    if not curve_id:
        raise ValueError("curve id is empty")

    # Most bond reference ids are already the curves.csv alias names, e.g.
    # `EUR-EURIBOR-3M`. Fall back to handle resolution only if needed.
    for fc_group in root.findall("./IndexForwardingCurves"):
        for idx_elem in fc_group.findall("./Index"):
            if (idx_elem.attrib.get("name", "") or "").strip() == curve_id:
                return curve_id
    for yc_group in root.findall("./YieldCurves"):
        for yc in yc_group.findall("./YieldCurve"):
            if (yc.attrib.get("name", "") or "").strip() == curve_id:
                return curve_id

    handle = curve_id
    if "/" not in handle:
        for fc_group in root.findall("./IndexForwardingCurves"):
            for idx_elem in fc_group.findall("./Index"):
                if (idx_elem.attrib.get("name", "") or "").strip() == curve_id:
                    handle = (idx_elem.text or "").strip()
                    break
        if "/" not in handle:
            for yc_group in root.findall("./YieldCurves"):
                for yc in yc_group.findall("./YieldCurve"):
                    if (yc.attrib.get("name", "") or "").strip() == curve_id:
                        handle = (yc.text or "").strip()
                        break

    if handle == curve_id and "/" not in handle:
        return curve_id
    return ore_snapshot_mod._handle_to_curve_name(root, handle)


def _load_curve_from_reference_output(
    ore_xml: Path,
    *,
    todaysmarket_xml: Path,
    curve_id: str,
    asof_date: str,
    day_counter: str,
) -> tuple[object, str, Path] | None:
    from py_ore_tools import ore_snapshot as ore_snapshot_mod

    curves_csv = _find_reference_output_file_local(ore_xml, "curves.csv")
    if curves_csv is None:
        return None
    column = _resolve_curve_column_from_id(todaysmarket_xml, curve_id)
    _, times, dfs = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
        str(curves_csv), [column], asof_date=asof_date, day_counter=day_counter
    )[column]
    return build_discount_curve_from_discount_pairs(list(zip(times, dfs))), column, curves_csv


def _load_callable_option_curve_from_reference_output(
    ore_xml: Path,
    *,
    todaysmarket_xml: Path,
    curve_id: str,
    asof_date: str,
    day_counter: str,
) -> tuple[object, str, Path] | None:
    # Callable-bond parity improved when the rollback uses the native ORE
    # reference curve from `curves.csv`, but the stripped risky bond remains on
    # the existing fitted curve. Some runs only emit `curves.csv` in a sibling
    # output variant (e.g. `_curves` rather than `_npv_only`), so try a small
    # set of source-faithful neighbours before giving up.
    direct = _load_curve_from_reference_output(
        ore_xml,
        todaysmarket_xml=todaysmarket_xml,
        curve_id=curve_id,
        asof_date=asof_date,
        day_counter=day_counter,
    )
    if direct is not None:
        return direct

    stem = ore_xml.stem
    sibling_stems: list[str] = []
    for src, dst in (
        ("_npv_only", "_curves"),
        ("_npv_additional", "_curves"),
        ("_npv", "_curves"),
    ):
        if stem.endswith(src):
            sibling_stems.append(stem[: -len(src)] + dst)
    if stem not in sibling_stems:
        sibling_stems.append(stem + "_curves")

    for sibling_stem in sibling_stems:
        sibling = ore_xml.with_name(f"{sibling_stem}{ore_xml.suffix}")
        if not sibling.exists():
            continue
        loaded = _load_curve_from_reference_output(
            sibling,
            todaysmarket_xml=todaysmarket_xml,
            curve_id=curve_id,
            asof_date=asof_date,
            day_counter=day_counter,
        )
        if loaded is not None:
            return loaded
    return None


def _sample_ql_curve(curve_handle, asof_date: date):
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for this curve build")
    curve = curve_handle.currentLink() if hasattr(curve_handle, "currentLink") else curve_handle
    pairs: list[tuple[float, float]] = [(0.0, 1.0)]
    curve.enableExtrapolation()
    if hasattr(curve, "dates"):
        for d in list(curve.dates())[1:]:
            t = float(curve.timeFromReference(d))
            if t <= 1.0e-12:
                continue
            pairs.append((t, float(curve.discount(d))))
    if len(pairs) == 1:
        max_time = 61.0
        for i in range(1, int(math.ceil(max_time / 0.25)) + 1):
            t = 0.25 * i
            pairs.append((t, float(curve.discount(t))))
    return build_discount_curve_from_discount_pairs(pairs)


def _build_ql_eur_projection_curve(
    *,
    asof_date: date,
    instruments: list[dict[str, object]],
    tenor: str,
):
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for this curve build")
    ql.Settings.instance().evaluationDate = ql.Date(asof_date.day, asof_date.month, asof_date.year)

    ois_instruments = _filter_instruments_for_discount_handle(instruments, discount_handle="Yield/EUR/EUR1D")
    ois_times, ois_zeros = _fit_quantlib_helper_eur_nodes(asof_date.isoformat(), ois_instruments)
    ois_pairs = [(float(t), float(math.exp(-float(z) * float(t)))) for t, z in zip(ois_times, ois_zeros)]
    ois_base = build_discount_curve_from_discount_pairs(ois_pairs)
    ois_dates = [ql.Date(asof_date.day, asof_date.month, asof_date.year)]
    ois_dfs = [1.0]
    for tt in sorted(set(round(float(t), 8) for t in ois_times if float(t) > 0.0)):
        qd = ql.Date(asof_date.day, asof_date.month, asof_date.year) + int(round(tt * 365.0))
        if qd <= ois_dates[-1]:
            continue
        ois_dates.append(qd)
        ois_dfs.append(float(ois_base(tt)))
    discount_curve = ql.RelinkableYieldTermStructureHandle()
    discount_curve.linkTo(ql.DiscountCurve(ois_dates, ois_dfs, ql.Actual365Fixed()))

    projection_curve = ql.RelinkableYieldTermStructureHandle()
    cal = ql.TARGET()
    zero_dc = ql.Actual365Fixed()
    fixed_dc = ql.Thirty360(ql.Thirty360.BondBasis)
    period = ql.Period(str(tenor))
    if str(tenor).upper() == "3M":
        index = ql.Euribor3M(projection_curve)
    elif str(tenor).upper() == "6M":
        index = ql.Euribor6M(projection_curve)
    else:
        raise ValueError(f"unsupported EUR projection tenor '{tenor}'")

    mm: list[dict[str, object]] = []
    fra: list[dict[str, object]] = []
    irs: list[dict[str, object]] = []
    for ins in instruments:
        tpe = str(ins.get("instrument_type", "")).upper()
        key = str(ins.get("quote_key", "")).upper()
        if tpe == "MM":
            mm.append(ins)
        elif key.startswith(f"FRA/RATE/EUR/") and key.endswith(f"/{str(tenor).upper()}"):
            fra.append(ins)
        elif tpe == "IR_SWAP" and f"/{str(tenor).upper()}/" in key:
            irs.append(ins)

    def _quote(ins: dict[str, object]) -> ql.QuoteHandle:
        return ql.QuoteHandle(ql.SimpleQuote(float(ins["quote_value"])))

    def _period(text: str) -> ql.Period:
        return ql.Period(str(text).upper())

    helpers: list[ql.RateHelper] = []
    helper_pillars: set[int] = set()

    def _append_helper(helper) -> None:
        pillar = int(helper.latestDate().serialNumber())
        if pillar in helper_pillars:
            return
        helper_pillars.add(pillar)
        helpers.append(helper)
    for ins in sorted(mm, key=lambda x: float(x["maturity"])):
        key = str(ins.get("quote_key", "")).upper()
        parts = key.split("/")
        if len(parts) < 5:
            continue
        dep_tenor = parts[-1]
        fixing_days = int(parts[3][:-1]) if parts[3].endswith("D") and parts[3][:-1].isdigit() else 2
        _append_helper(
            ql.DepositRateHelper(
                _quote(ins),
                _period(dep_tenor),
                fixing_days,
                cal,
                ql.ModifiedFollowing,
                False,
                ql.Actual360(),
            )
        )
    for ins in sorted(fra, key=lambda x: float(x["maturity"])):
        key = str(ins.get("quote_key", "")).upper()
        parts = key.split("/")
        if len(parts) < 5:
            continue
        start_period = _period(parts[3])
        _append_helper(ql.FraRateHelper(_quote(ins), start_period, index))
    for ins in sorted(irs, key=lambda x: float(x["maturity"])):
        key = str(ins.get("quote_key", "")).upper()
        parts = key.split("/")
        if len(parts) < 6:
            continue
        swap_tenor = parts[-1]
        _append_helper(
            ql.SwapRateHelper(
                _quote(ins),
                _period(swap_tenor),
                cal,
                ql.Annual,
                ql.ModifiedFollowing,
                fixed_dc,
                index,
                ql.QuoteHandle(),
                ql.Period(0, ql.Days),
                discount_curve,
            )
        )
    curve = ql.PiecewiseLogLinearDiscount(
        ql.Date(asof_date.day, asof_date.month, asof_date.year),
        helpers,
        zero_dc,
    )
    projection_curve.linkTo(curve)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def _curve_from_zero_quotes(
    *,
    asof_date: date,
    instruments: list[dict[str, object]],
    curve_tag: str,
):
    pairs: list[tuple[float, float]] = [(0.0, 1.0)]
    for ins in instruments:
        key = str(ins.get("quote_key", "")).upper()
        if f"/{curve_tag.upper()}/" not in key:
            continue
        parts = key.split("/")
        tenor = parts[-1]
        try:
            t = _parse_period_to_years(tenor)
        except Exception:
            continue
        r = float(ins["quote_value"])
        pairs.append((float(t), float(math.exp(-r * float(t)))))
    if len(pairs) <= 1:
        raise ValueError(f"no zero quotes found for curve tag '{curve_tag}'")
    return build_discount_curve_from_discount_pairs(sorted(set((round(t, 12), df) for t, df in pairs)))


def _fit_curve_for_id(ore_xml: Path, currency: str, curve_id: str | None, asof_date: date):
    curve_name = (curve_id or "").strip().upper()
    if not curve_name:
        return _fit_curve_for_currency(ore_xml, currency)
    if currency.upper() == "EUR":
        try:
            payload = extract_market_instruments_by_currency(
                ore_xml,
                instrument_types=("ZERO", "MM", "FRA", "IR_SWAP", "OIS"),
            )
            instruments = list(payload.get(currency, {}).get("instruments", []))
            if curve_name.endswith("1D") or curve_name.endswith("EONIA"):
                ql_curve = _build_ql_discount_curve_for_callable_calibration(
                    ore_xml=ore_xml,
                    todaysmarket_xml=_resolve_todaysmarket_path(ore_xml) or Path(),
                    currency=currency,
                    asof_date=asof_date,
                )
                return _sample_ql_curve(ql_curve, asof_date)
            if curve_name.endswith("3M"):
                return _sample_ql_curve(_build_ql_eur_projection_curve(asof_date=asof_date, instruments=instruments, tenor="3M"), asof_date)
            if curve_name.endswith("6M"):
                return _sample_ql_curve(_build_ql_eur_projection_curve(asof_date=asof_date, instruments=instruments, tenor="6M"), asof_date)
            if "BOND_YIELD_EUR" in curve_name:
                return _curve_from_zero_quotes(asof_date=asof_date, instruments=instruments, curve_tag="BOND_YIELD_EUR")
        except Exception:
            pass
    return _fit_curve_for_currency(ore_xml, currency)


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


def _resolve_todaysmarket_path(ore_xml: Path) -> Path | None:
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
    raw = setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")
    return _resolve_ore_path(raw, input_dir)


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
    # ORE expands ReferenceCalibrationGrid through DateGrid(grid), whose
    # defaults are TARGET + Following. Using raw month addition here shifts
    # some expiries onto weekends and changes the callable LGM calibration
    # basket materially, especially for the put-heavy example.
    if ql is not None:
        cal = ql.TARGET()
        ql_today = ql.Date(asof_date.day, asof_date.month, asof_date.year)
        ql_maturity = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)
        tenor = ql.Period(int(months), ql.Months)
        for i in range(max(count, 0)):
            d = cal.advance(ql_today, (i + 1) * tenor, ql.Following, False)
            if d >= ql_maturity:
                break
            dates.append(date(d.year(), int(d.month()), d.dayOfMonth()))
        return dates

    current = asof_date
    for _ in range(max(count, 0)):
        current = _add_months(current, months)
        while current.weekday() >= 5:
            current += timedelta(days=1)
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


def _resolve_todaysmarket_discount_handle(
    todaysmarket_xml: Path,
    *,
    currency: str,
    configuration_id: str,
) -> str | None:
    root = ET.parse(todaysmarket_xml).getroot()
    cfg = root.find(f"./Configuration[@id='{configuration_id}']")
    if cfg is None:
        return None
    disc_curves_id = (cfg.findtext("./DiscountingCurvesId") or "").strip()
    if not disc_curves_id:
        return None
    disc_curves = root.find(f"./DiscountingCurves[@id='{disc_curves_id}']")
    if disc_curves is None:
        return None
    node = disc_curves.find(f"./DiscountingCurve[@currency='{currency}']")
    if node is None:
        return None
    handle = (node.text or "").strip()
    return handle or None


def _filter_instruments_for_discount_handle(
    instruments: list[dict[str, object]],
    *,
    discount_handle: str,
) -> list[dict[str, object]]:
    handle = str(discount_handle).strip().upper()
    if not handle:
        return []

    # ORE's callable-bond LGM builder calibrates off the specific discount
    # curve handle resolved from MarketContext::irCalibration. For the EUR
    # callable example this is the 1D/OIS curve, so using the full generic EUR
    # market fit mixes in 3M/6M swap instruments that are not part of the
    # native calibration curve family.
    if handle.endswith("1D"):
        out: list[dict[str, object]] = []
        for ins in instruments:
            tpe = str(ins.get("instrument_type", "")).upper()
            key = str(ins.get("quote_key", "")).upper()
            if tpe == "MM" and key.startswith("MM/RATE/") and key.endswith("/1D"):
                out.append(ins)
            elif tpe == "IR_SWAP" and "/1D/" in key:
                out.append(ins)
        return out

    return []


def _build_ql_discount_curve_for_callable_calibration(
    *,
    ore_xml: Path,
    todaysmarket_xml: Path,
    currency: str,
    asof_date: date,
):
    if ql is None:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("QuantLib Python bindings are required for callable bond calibration")

    discount_handle = _resolve_todaysmarket_discount_handle(
        todaysmarket_xml,
        currency=currency,
        configuration_id="collateral_inccy",
    ) or _resolve_todaysmarket_discount_handle(
        todaysmarket_xml,
        currency=currency,
        configuration_id="default",
    )

    if discount_handle and currency.upper() == "EUR":
        try:
            payload = extract_market_instruments_by_currency(
                ore_xml,
                instrument_types=("ZERO", "MM", "IR_SWAP", "OIS"),
            )
            instruments = list(payload.get(currency, {}).get("instruments", []))
            filtered = _filter_instruments_for_discount_handle(instruments, discount_handle=discount_handle)
            if filtered:
                times, zeros = _fit_quantlib_helper_eur_nodes(asof_date.isoformat(), filtered)
                pairs = [(float(t), float(math.exp(-float(z) * float(t)))) for t, z in zip(times, zeros)]
                base_curve = build_discount_curve_from_discount_pairs(pairs)
                dates = [ql.Date(asof_date.day, asof_date.month, asof_date.year)]
                dfs = [1.0]
                for tt in sorted(set(round(float(t), 8) for t in times if float(t) > 0.0)):
                    qd = ql.Date(asof_date.day, asof_date.month, asof_date.year) + int(round(tt * 365.0))
                    if qd <= dates[-1]:
                        continue
                    dates.append(qd)
                    dfs.append(float(base_curve(tt)))
                return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))
        except Exception:
            pass

    return _build_ql_discount_curve(ore_xml, currency, asof_date)


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
    discount_curve = _build_ql_discount_curve_for_callable_calibration(
        ore_xml=ore_xml,
        todaysmarket_xml=todaysmarket_xml,
        currency=currency,
        asof_date=asof_date,
    )
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
    tt = float(t)
    idx = int(np.searchsorted(grid, tt, side="left"))
    if idx < grid.size and abs(grid[idx] - tt) <= 1.0e-10:
        return idx
    if idx > 0 and abs(grid[idx - 1] - tt) <= 1.0e-10:
        return idx - 1
    if idx >= grid.size:
        return grid.size - 1
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


@dataclass(frozen=True)
class _CallableCashflowState:
    amount: float
    pay_time: float
    belongs_to_underlying_max_time: float
    max_estimation_time: float | None
    exact_estimation_time: float | None
    coupon_start_time: float | None
    coupon_end_time: float | None


@dataclass(frozen=True)
class CallableRollbackTraceRow:
    grid_index: int
    time: float
    notional: float
    accrual: float
    flow_amount: float
    call_price: float | None
    put_price: float | None
    underlying_center: float
    provisional_center: float
    option_before_call_center: float
    option_after_call_center: float
    option_after_put_center: float


def _build_callable_cashflow_states(
    spec: CallableBondTradeSpec,
    *,
    asof_date: date,
    day_counter: str,
) -> list[_CallableCashflowState]:
    out: list[_CallableCashflowState] = []
    for cf in spec.bond.cashflows:
        pay_t = _time_from_dates(asof_date, cf.pay_date, day_counter)
        if pay_t <= 0.0:
            continue
        if cf.accrual_start is not None and cf.accrual_end is not None:
            start_t = _time_from_dates(asof_date, cf.accrual_start, day_counter)
            end_t = _time_from_dates(asof_date, cf.accrual_end, day_counter)
            out.append(
                _CallableCashflowState(
                    amount=float(cf.amount),
                    pay_time=float(pay_t),
                    belongs_to_underlying_max_time=float(end_t),
                    max_estimation_time=float(pay_t),
                    exact_estimation_time=None,
                    coupon_start_time=float(start_t),
                    coupon_end_time=float(end_t),
                )
            )
        else:
            out.append(
                _CallableCashflowState(
                    amount=float(cf.amount),
                    pay_time=float(pay_t),
                    belongs_to_underlying_max_time=float(pay_t),
                    max_estimation_time=float(pay_t),
                    exact_estimation_time=None,
                    coupon_start_time=None,
                    coupon_end_time=None,
                )
            )
    return out


def _callable_close_enough(a: float, b: float, tol: float = 1.0e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def _callable_is_part_of_underlying(cf: _CallableCashflowState, t: float) -> bool:
    return float(t) < cf.belongs_to_underlying_max_time or _callable_close_enough(float(t), cf.belongs_to_underlying_max_time)


def _callable_can_be_estimated(cf: _CallableCashflowState, t: float) -> bool:
    tt = float(t)
    if cf.max_estimation_time is not None:
        return tt < cf.max_estimation_time or _callable_close_enough(tt, cf.max_estimation_time)
    if cf.exact_estimation_time is not None:
        return _callable_close_enough(tt, cf.exact_estimation_time)
    return False


def _callable_must_be_estimated(cf: _CallableCashflowState, t: float) -> bool:
    if cf.max_estimation_time is not None or cf.exact_estimation_time is None:
        return False
    tt = float(t)
    return tt < cf.exact_estimation_time or _callable_close_enough(tt, cf.exact_estimation_time)


def _callable_coupon_ratio(cf: _CallableCashflowState, t: float) -> float:
    if cf.coupon_start_time is None or cf.coupon_end_time is None:
        return 1.0
    denom = cf.coupon_end_time - cf.coupon_start_time
    if denom <= 1.0e-18:
        return 0.0
    return max(0.0, min(1.0, (cf.coupon_end_time - float(t)) / denom))


def _callable_cashflow_reduced_pv(
    cf: _CallableCashflowState,
    model: LGM1F,
    eff_discount_curve,
    *,
    t: float,
    x_grid: np.ndarray,
) -> np.ndarray:
    disc = np.asarray(model.discount_bond(float(t), float(cf.pay_time), x_grid, eff_discount_curve, eff_discount_curve), dtype=float)
    num = np.asarray(model.numeraire_lgm(float(t), x_grid, eff_discount_curve), dtype=float)
    return float(cf.amount) * disc / num


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
    trace_rows: list[CallableRollbackTraceRow] | None = None,
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
    change_times, notionals = _callable_notional_schedule(spec, asof_date, day_counter)
    cashflows = _build_callable_cashflow_states(spec, asof_date=asof_date, day_counter=day_counter)
    cf_status = ["Open"] * len(cashflows)
    cf_cache: list[np.ndarray | None] = [None] * len(cashflows)

    # Mirror the ORE engine loop more directly:
    # - `underlying_npv` is the reduced-form value of already-activated bond
    #   cashflows that remain part of the underlying
    # - `provisional_npv` captures flows that are part of the underlying but
    #   cannot yet be fully estimated at the exercise time
    # For the current callable bond examples all coupons are fixed, so the
    # provisional bucket stays zero; we still keep the state split because the
    # call/put formulas in ORE use it asymmetrically.
    underlying_npv = np.zeros_like(x_grids[-1], dtype=float)
    option_values = np.zeros_like(x_grids[-1], dtype=float)
    provisional_npv = np.zeros_like(x_grids[-1], dtype=float)
    for i in range(len(grid) - 1, 0, -1):
        t_from = float(grid[i])
        grid_i = x_grids[i]
        if i < len(grid) - 1:
            option_values = _convolution_rollback(
                option_values,
                zeta_t1=float(zeta_grid[i + 1]),
                zeta_t0=float(zeta_grid[i]),
                mx=mx,
                nx=int(engine_spec.grid_nx),
                y_nodes=y_nodes,
                y_weights=y_weights,
            )
            underlying_npv = _convolution_rollback(
                underlying_npv,
                zeta_t1=float(zeta_grid[i + 1]),
                zeta_t0=float(zeta_grid[i]),
                mx=mx,
                nx=int(engine_spec.grid_nx),
                y_nodes=y_nodes,
                y_weights=y_weights,
            )
            if i == 1:
                provisional_npv = _convolution_rollback(
                    provisional_npv,
                    zeta_t1=float(zeta_grid[i + 1]),
                    zeta_t0=float(zeta_grid[i]),
                    mx=mx,
                    nx=int(engine_spec.grid_nx),
                    y_nodes=y_nodes,
                    y_weights=y_weights,
                )
            for j, cached in enumerate(cf_cache):
                if cached is None:
                    continue
                cf_cache[j] = _convolution_rollback(
                    np.asarray(cached, dtype=float),
                    zeta_t1=float(zeta_grid[i + 1]),
                    zeta_t0=float(zeta_grid[i]),
                    mx=mx,
                    nx=int(engine_spec.grid_nx),
                    y_nodes=y_nodes,
                    y_weights=y_weights,
                )

        provisional_npv = np.zeros_like(provisional_npv)
        flow_amount = 0.0
        for j, cf in enumerate(cashflows):
            if cf_status[j] == "Done":
                continue
            if _callable_is_part_of_underlying(cf, t_from) and _callable_coupon_ratio(cf, t_from) > 0.0:
                if cf_status[j] == "Cached":
                    underlying_npv = underlying_npv + np.asarray(cf_cache[j], dtype=float)
                    cf_cache[j] = None
                    cf_status[j] = "Done"
                elif _callable_can_be_estimated(cf, t_from):
                    flow_amount += float(cf.amount)
                    underlying_npv = underlying_npv + _callable_cashflow_reduced_pv(
                        cf,
                        model,
                        eff_discount_curve,
                        t=t_from,
                        x_grid=grid_i,
                    )
                    cf_status[j] = "Done"
                else:
                    provisional_npv = provisional_npv + _callable_cashflow_reduced_pv(
                        cf,
                        model,
                        eff_discount_curve,
                        t=t_from,
                        x_grid=grid_i,
                    )
            elif _callable_must_be_estimated(cf, t_from) and cf_status[j] == "Open":
                cf_cache[j] = _callable_cashflow_reduced_pv(
                    cf,
                    model,
                    eff_discount_curve,
                    t=t_from,
                    x_grid=grid_i,
                )
                cf_status[j] = "Cached"
        if i in call_map or i in put_map:
            continuation = option_values.copy()
            option_before_call_center = float(option_values[mx])
            option_after_call_center = option_before_call_center
            option_after_put_center = option_before_call_center
            notional = _callable_notional_at_time(change_times, notionals, t_from)
            accrual = _callable_accrual_at_time(spec, asof_date, day_counter, t_from)
            call_price = None
            put_price = None
            if i in call_map:
                ex = call_map[i]
                call_price = float(ex.price)
                amt = _callable_price_amount(
                    ex,
                    notional,
                    accrual,
                )
                num = np.asarray(model.numeraire_lgm(t_from, grid_i, eff_discount_curve), dtype=float)
                continuation = np.minimum(continuation, float(amt) / num - (underlying_npv + provisional_npv))
                option_after_call_center = float(continuation[mx])
            if i in put_map:
                ex = put_map[i]
                put_price = float(ex.price)
                amt = _callable_price_amount(
                    ex,
                    notional,
                    accrual,
                )
                num = np.asarray(model.numeraire_lgm(t_from, grid_i, eff_discount_curve), dtype=float)
                continuation = np.maximum(continuation, float(amt) / num - underlying_npv + provisional_npv)
                option_after_put_center = float(continuation[mx])
            option_values = continuation
            if trace_rows is not None:
                trace_rows.append(
                    CallableRollbackTraceRow(
                        grid_index=i,
                        time=t_from,
                        notional=float(notional),
                        accrual=float(accrual),
                        flow_amount=float(flow_amount),
                        call_price=call_price,
                        put_price=put_price,
                        underlying_center=float(underlying_npv[mx]),
                        provisional_center=float(provisional_npv[mx]),
                        option_before_call_center=float(option_before_call_center),
                        option_after_call_center=float(option_after_call_center),
                        option_after_put_center=float(option_after_put_center),
                    )
                )
    if len(grid) > 1:
        option_values = _convolution_rollback(
            option_values,
            zeta_t1=float(zeta_grid[1]),
            zeta_t0=float(zeta_grid[0]),
            mx=mx,
            nx=int(engine_spec.grid_nx),
            y_nodes=y_nodes,
            y_weights=y_weights,
        )
    return float(option_values[mx]) / max(float(eff_income_curve(0.0)), 1.0e-18)


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
    include_trace: bool = False,
) -> dict[str, float | int | str]:
    trace_rows: list[CallableRollbackTraceRow] | None = [] if include_trace else None
    option_value = _rollback_callable_bond_lgm_value(
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
        trace_rows=trace_rows,
    )
    stripped, _ = _bond_npv(
        spec.bond,
        asof_date=asof_date,
        day_counter=day_counter,
        discount_curve=reference_curve,
        income_curve=income_curve,
        hazard_times=hazard_times,
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
        security_spread=security_spread,
        engine_spec=BondEngineSpec(
            timestep_months=6,
            spread_on_income_curve=engine_spec.spread_on_income_curve,
            treat_security_spread_as_credit_spread=False,
            include_past_cashflows=False,
        ),
    )
    price = float(stripped + option_value)
    maturity_date = max(cf.pay_date for cf in spec.bond.cashflows)
    result = {
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
    if trace_rows is not None:
        result["trace_rows"] = trace_rows
    return result


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
        callable_reference_curve = base_curve
        callable_reference_curve_column = ""
        callable_reference_curve_csv = ""
        if spec.reference_curve_id:
            native_curve = _load_callable_option_curve_from_reference_output(
                ore_xml,
                todaysmarket_xml=todaysmarket_xml,
                curve_id=spec.reference_curve_id,
                asof_date=asof_date,
                day_counter=model_day_counter,
            )
            if native_curve is not None:
                callable_reference_curve, callable_reference_curve_column, callable_reference_curve_csv = native_curve
        result = _price_callable_bond_lgm(
            spec,
            engine_spec,
            asof_date=asof,
            day_counter=model_day_counter,
            reference_curve=callable_reference_curve,
            income_curve=callable_reference_curve,
            hazard_times=hazard_times,
            hazard_rates=hazard_rates,
            recovery_rate=recovery_rate,
            security_spread=security_spread,
            model=model,
        )
        if callable_reference_curve is not base_curve:
            stripped, _ = _bond_npv(
                spec.bond,
                asof_date=asof,
                day_counter=model_day_counter,
                discount_curve=base_curve,
                income_curve=income_curve,
                hazard_times=hazard_times,
                hazard_rates=hazard_rates,
                recovery_rate=recovery_rate,
                security_spread=security_spread,
                engine_spec=BondEngineSpec(
                    timestep_months=6,
                    spread_on_income_curve=engine_spec.spread_on_income_curve,
                    treat_security_spread_as_credit_spread=False,
                    include_past_cashflows=False,
                ),
            )
            option_value = float(result["py_npv"]) - float(result["stripped_bond_npv"])
            result["stripped_bond_npv"] = float(stripped)
            result["py_npv"] = float(stripped + option_value)
            result["py_settlement_value"] = float(result["py_npv"])
            result["embedded_option_value"] = float(stripped - float(result["py_npv"]))
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
                "callable_option_reference_curve_id": spec.reference_curve_id,
                "callable_option_reference_curve_column": callable_reference_curve_column,
                "callable_option_reference_curve_csv": callable_reference_curve_csv,
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
