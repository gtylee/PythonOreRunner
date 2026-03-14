from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from py_ore_tools.irs_xva_utils import (
    _parse_yyyymmdd,
    load_ore_default_curve_inputs,
    survival_probability_from_hazard,
)
from py_ore_tools.ore_snapshot import (
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


def _resolve_trade(root: ET.Element, trade_id: str) -> ET.Element:
    for trade in root.findall("./Trade"):
        if (trade.attrib.get("id", "") or "").strip() == trade_id:
            return trade
    raise ValueError(f"trade '{trade_id}' not found in portfolio")


def _merge_reference_data(bond_node: ET.Element, reference_data_path: Path | None) -> ET.Element:
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
    npv_date = npv_date or asof_date
    settlement_date = settlement_date or npv_date
    include_ref_date_flows = False
    npv_t = _time_from_dates(asof_date, npv_date, day_counter)
    sett_t = _time_from_dates(asof_date, settlement_date, day_counter)
    extra_credit = security_spread / max(1.0 - float(recovery_rate), 1.0e-12) if engine_spec.treat_security_spread_as_credit_spread else 0.0
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
    asof = _parse_any_date(asof_date)
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
    value, settlement_value = _bond_npv(
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
    )
    if spec.trade_type == "ForwardBond":
        forward_value, _ = _bond_npv(
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
        strike = float(spec.forward_amount or 0.0) + (0.0 if spec.settlement_dirty else accrued)
        eff_settlement_date = bond_settlement_date
        discount = float(base_curve(max(_time_from_dates(asof, eff_settlement_date, model_day_counter), 0.0)))
        if spec.long_in_forward:
            value = (forward_value - strike) * discount
        else:
            value = (strike - forward_value) * discount
        if spec.compensation_payment and spec.compensation_payment_date > asof:
            cmp_df = float(base_curve(max(_time_from_dates(asof, spec.compensation_payment_date, model_day_counter), 0.0)))
            value += (-spec.compensation_payment if spec.long_in_forward else spec.compensation_payment) * cmp_df
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
