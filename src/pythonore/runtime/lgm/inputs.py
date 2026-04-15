from __future__ import annotations

from datetime import datetime
import warnings
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from pythonore.domain.dataclasses import IRS, Trade
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.lgm.market import _default_index_for_ccy, _forward_index_family, _parse_tenor_to_years


def _fallback_exposure_grid(snapshot) -> np.ndarray:
    max_mat = float(snapshot.config.horizon_years)
    for t in snapshot.portfolio.trades:
        p = t.product
        m = getattr(p, "maturity_years", None)
        if isinstance(m, (int, float)):
            max_mat = max(max_mat, float(m))
    if max_mat <= 0.0:
        max_mat = 1.0
    steps = max(4, int(np.ceil(max_mat * 4.0)))
    return np.linspace(0.0, max_mat, steps + 1)


def _irs_schedule_bounds(product: IRS, asof: str | None) -> Tuple[float, float]:
    if product.start_date and asof:
        start = str(product.start_date).strip()
        ref = str(asof).strip()
        start_offset = (datetime.strptime(start, "%Y-%m-%d") - datetime.strptime(ref, "%Y-%m-%d")).days / 365.25
    else:
        start_offset = 0.0
    if product.end_date and asof:
        end = str(product.end_date).strip()
        ref = str(asof).strip()
        end_offset = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(ref, "%Y-%m-%d")).days / 365.25
    else:
        end_offset = start_offset + float(product.maturity_years)
    end_offset = max(end_offset, start_offset + 1.0 / 365.25)
    return float(start_offset), float(end_offset)


def _schedule_periods(start: float, end: float, tenor: str, rule: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = _parse_tenor_to_years(tenor)
    if step <= 0.0:
        raise EngineRunError(f"Unsupported IRS tenor '{tenor}'")
    rule_name = str(rule or "Forward").strip().lower()
    starts: List[float] = []
    stops: List[float] = []
    if rule_name == "backward":
        current = float(end)
        while current > start + 1.0e-12:
            prev = max(start, current - step)
            starts.append(prev)
            stops.append(current)
            current = prev
        starts.reverse()
        stops.reverse()
    else:
        current = float(start)
        while current < end - 1.0e-12:
            nxt = min(end, current + step)
            starts.append(current)
            stops.append(nxt)
            current = nxt
    start_arr = np.asarray(starts, dtype=float)
    stop_arr = np.asarray(stops, dtype=float)
    return stop_arr.copy(), start_arr, stop_arr


def _tenor_to_years(tenor: str) -> float:
    return _parse_tenor_to_years(tenor)


def _trade_notional(trade: Trade) -> float:
    p = trade.product
    n = getattr(p, "notional", None)
    if isinstance(n, (int, float)):
        return abs(float(n))
    return 0.0


def _parse_swaption_premium_records(trade_root: ET.Element) -> Tuple[Dict[str, object], ...]:
    def _parse_date_text(text: str) -> str | None:
        value = str(text or "").strip()
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%d/%m/%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
        return value

    def _append_record(
        records: List[Dict[str, object]],
        *,
        amount_text: str | None,
        currency_text: str | None,
        pay_date_text: str | None,
        source: str,
    ) -> None:
        amount_value = str(amount_text or "").strip()
        pay_date_value = _parse_date_text(pay_date_text or "")
        if not amount_value or not pay_date_value:
            return
        try:
            amount = float(amount_value)
        except ValueError:
            return
        currency_value = str(currency_text or "").strip().upper()
        if not currency_value:
            currency_value = str(
                trade_root.findtext("./SwaptionData/LegData/Currency")
                or trade_root.findtext("./SwaptionData/OptionData/PremiumCurrency")
                or ""
            ).strip().upper()
        records.append(
            {
                "amount": float(amount),
                "currency": currency_value,
                "pay_date": pay_date_value,
                "source": source,
            }
        )

    records: List[Dict[str, object]] = []
    for premium in trade_root.findall("./SwaptionData/OptionData/Premiums/Premium"):
        _append_record(
            records,
            amount_text=premium.findtext("./Amount") or premium.findtext("./PremiumAmount"),
            currency_text=premium.findtext("./Currency") or premium.findtext("./PremiumCurrency"),
            pay_date_text=premium.findtext("./PayDate") or premium.findtext("./PremiumPayDate") or premium.findtext("./PremiumDate"),
            source="nested",
        )
    _append_record(
        records,
        amount_text=trade_root.findtext("./SwaptionData/OptionData/PremiumAmount"),
        currency_text=trade_root.findtext("./SwaptionData/OptionData/PremiumCurrency"),
        pay_date_text=trade_root.findtext("./SwaptionData/OptionData/PremiumPayDate")
        or trade_root.findtext("./SwaptionData/OptionData/PremiumDate"),
        source="flat",
    )
    return tuple(records)


def _build_irs_legs_from_trade(trade: Trade, asof: str | None = None) -> Dict[str, np.ndarray]:
    warnings.warn(
        f"Using fallback leg schedule for trade {trade.trade_id} "
        "(no flows.csv or portfolio.xml available). "
        "Schedule timing is generated from the IRS dataclass fields and remains approximate for calendars/day-count roll rules.",
        UserWarning,
        stacklevel=3,
    )
    p = trade.product
    if not isinstance(p, IRS):
        raise EngineRunError(f"Cannot build IRS legs for non-IRS trade {trade.trade_id}")
    start_offset, end_offset = _irs_schedule_bounds(p, asof)
    fixed_pay, fixed_start, fixed_end = _schedule_periods(start_offset, end_offset, p.fixed_leg_tenor, p.fixed_schedule_rule)
    float_pay, float_start, float_end = _schedule_periods(start_offset, end_offset, p.float_leg_tenor, p.float_schedule_rule)
    fixing = float_start - (float(p.fixing_days) / 365.25)
    fixed_sign = -1.0 if p.pay_fixed else 1.0
    float_sign = -fixed_sign
    float_index = str(p.float_index or trade.additional_fields.get("index", _default_index_for_ccy(p.ccy))).upper()
    float_index_tenor = float_index.split("-")[-1].upper() if "-" in float_index else ""
    overnight_indexed = _forward_index_family(float_index) == "1D"
    return {
        "fixed_pay_time": fixed_pay,
        "fixed_accrual": np.maximum(fixed_end - fixed_start, 0.0),
        "fixed_rate": np.full(fixed_pay.shape, float(p.fixed_rate)),
        "fixed_notional": np.full(fixed_pay.shape, float(p.notional)),
        "fixed_sign": np.full(fixed_pay.shape, fixed_sign),
        "fixed_amount": fixed_sign * float(p.notional) * float(p.fixed_rate) * np.maximum(fixed_end - fixed_start, 0.0),
        "float_pay_time": float_pay,
        "float_start_time": float_start,
        "float_end_time": float_end,
        "float_accrual": np.maximum(float_end - float_start, 0.0),
        "float_notional": np.full(float_pay.shape, float(p.notional)),
        "float_sign": np.full(float_pay.shape, float_sign),
        "float_spread": np.full(float_pay.shape, float(p.float_spread)),
        "float_coupon": np.zeros(float_pay.shape),
        "float_amount": np.zeros(float_pay.shape),
        "float_fixing_time": fixing,
        "float_is_historically_fixed": np.zeros(float_pay.shape, dtype=bool),
        "float_index": float_index,
        "float_index_tenor": float_index_tenor,
        "float_overnight_indexed": overnight_indexed,
        "float_lookback_days": 0,
        "float_rate_cutoff": 0,
        "float_naked_option": False,
        "float_local_cap_floor": False,
        "float_cap": None,
        "float_floor": None,
        "float_apply_observation_shift": False,
        "float_gearing": np.ones(float_pay.shape),
        "float_is_in_arrears": True,
        "float_fixing_days": int(p.fixing_days),
    }


__all__ = [
    "_build_irs_legs_from_trade",
    "_fallback_exposure_grid",
    "_irs_schedule_bounds",
    "_schedule_periods",
    "_parse_swaption_premium_records",
    "_tenor_to_years",
    "_trade_notional",
]
