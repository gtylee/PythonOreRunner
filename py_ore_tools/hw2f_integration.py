from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np

from .hw2f import HW2FModel, parse_hw2f_params_from_simulation_xml, simulate_hw_ba_euler
from .irs_xva_utils import (
    build_discount_curve_from_discount_pairs,
    load_ore_discount_pairs_from_curves,
    load_swap_legs_from_portfolio,
)


@dataclass(frozen=True)
class HW2FCaseContext:
    case_dir: Path
    model: HW2FModel
    discount_curve: object
    legs: dict[str, np.ndarray]
    trade_id: str
    asof_date: str


def load_hw2f_case_context(
    case_dir: str | Path,
    *,
    discount_column: str = "EUR-EONIA",
    trade_id: Optional[str] = None,
) -> HW2FCaseContext:
    case_dir = Path(case_dir).resolve()
    input_dir = case_dir / "Input"
    output_dir = case_dir / "Output"
    ore_xml = input_dir / "ore.xml"
    simulation_xml = input_dir / "simulation.xml"
    portfolio_xml = input_dir / "portfolio.xml"
    curves_csv = output_dir / "curves.csv"

    params = parse_hw2f_params_from_simulation_xml(simulation_xml)
    model = HW2FModel(params)

    curve_t, curve_df = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=discount_column)
    discount_curve = build_discount_curve_from_discount_pairs(list(zip(curve_t, curve_df)))

    asof_date = _read_asof_date(ore_xml)
    resolved_trade_id = trade_id or _first_trade_id(portfolio_xml)
    legs = load_swap_legs_from_portfolio(str(portfolio_xml), resolved_trade_id, asof_date)

    return HW2FCaseContext(
        case_dir=case_dir,
        model=model,
        discount_curve=discount_curve,
        legs=legs,
        trade_id=resolved_trade_id,
        asof_date=asof_date,
    )


def swap_npv_from_ore_legs_hw(
    model: HW2FModel,
    p0,
    legs: dict[str, np.ndarray],
    t: float,
    x_t: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_t, dtype=float)
    if x.ndim != 2 or x.shape[1] != model.n:
        raise ValueError(f"x_t must have shape (n_paths, {model.n})")
    pv = np.zeros(x.shape[0], dtype=float)
    p_t = p0(float(t))

    mask_f = legs["fixed_pay_time"] > t + 1.0e-12
    if np.any(mask_f):
        pay = legs["fixed_pay_time"][mask_f]
        disc = model.discount_bond_paths(t, pay, x, p_t, lambda T: p0(float(T)))
        cash = legs["fixed_amount"][mask_f]
        pv += np.sum(cash[:, None] * disc, axis=0)

    mask_float = legs["float_pay_time"] > t + 1.0e-12
    if np.any(mask_float):
        s = legs["float_start_time"][mask_float]
        e = legs["float_end_time"][mask_float]
        pay = legs["float_pay_time"][mask_float]
        tau = legs["float_accrual"][mask_float]
        n = legs["float_notional"][mask_float]
        sign = legs["float_sign"][mask_float]
        spread = legs["float_spread"][mask_float]

        fixed = legs["float_fixing_time"][mask_float] <= t + 1.0e-12
        p_tp = model.discount_bond_paths(t, pay, x, p_t, lambda T: p0(float(T)))

        cash = np.zeros((pay.size, x.shape[0]), dtype=float)
        if np.any(fixed):
            coupon = legs["float_coupon"][mask_float][fixed]
            cash[fixed, :] = sign[fixed, None] * n[fixed, None] * coupon[:, None] * tau[fixed, None]

        if np.any(~fixed):
            p_ts = model.discount_bond_paths(t, s[~fixed], x, p_t, lambda T: p0(float(T)))
            p_te = model.discount_bond_paths(t, e[~fixed], x, p_t, lambda T: p0(float(T)))
            fwd = (p_ts / p_te - 1.0) / tau[~fixed, None]
            cash[~fixed, :] = sign[~fixed, None] * n[~fixed, None] * (fwd + spread[~fixed, None]) * tau[~fixed, None]

        pv += np.sum(cash * p_tp, axis=0)

    return pv


def price_hw2f_swap_t0(case_dir: str | Path, *, discount_column: str = "EUR-EONIA", trade_id: Optional[str] = None) -> float:
    ctx = load_hw2f_case_context(case_dir, discount_column=discount_column, trade_id=trade_id)
    x0 = np.zeros((1, ctx.model.n), dtype=float)
    return float(swap_npv_from_ore_legs_hw(ctx.model, ctx.discount_curve, ctx.legs, 0.0, x0)[0])


def simulate_hw2f_exposure_paths(
    case_dir: str | Path,
    *,
    n_paths: int = 1000,
    times: Optional[np.ndarray] = None,
    seed: int = 42,
    discount_column: str = "EUR-EONIA",
    trade_id: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = load_hw2f_case_context(case_dir, discount_column=discount_column, trade_id=trade_id)
    if times is None:
        times = _simulation_grid_times(ctx.case_dir / "Input" / "simulation.xml")
    rng = np.random.default_rng(seed)
    x_paths, aux_paths = simulate_hw_ba_euler(ctx.model, times, n_paths=n_paths, rng=rng)
    npv_paths = np.empty((len(times), n_paths), dtype=float)
    for i, t in enumerate(times):
        npv_paths[i, :] = swap_npv_from_ore_legs_hw(ctx.model, ctx.discount_curve, ctx.legs, float(t), x_paths[i, :, :])
    return np.asarray(times, dtype=float), x_paths, npv_paths


def read_ore_t0_npv(case_dir: str | Path, trade_id: Optional[str] = None) -> float:
    npv_csv = Path(case_dir).resolve() / "Output" / "npv.csv"
    with open(npv_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"npv.csv is empty: {npv_csv}")
    if trade_id is None:
        row = rows[0]
    else:
        row = next((r for r in rows if (r.get("#TradeId") or r.get("TradeId")) == trade_id), None)
        if row is None:
            raise ValueError(f"trade '{trade_id}' not found in {npv_csv}")
    return float(row.get("NPV(Base)") or row.get("NPV"))


def compare_hw2f_python_vs_ore(case_dir: str | Path, *, discount_column: str = "EUR-EONIA", trade_id: Optional[str] = None) -> dict[str, float | str]:
    py_npv = price_hw2f_swap_t0(case_dir, discount_column=discount_column, trade_id=trade_id)
    ctx = load_hw2f_case_context(case_dir, discount_column=discount_column, trade_id=trade_id)
    ore_npv = read_ore_t0_npv(case_dir, trade_id=ctx.trade_id)
    diff = py_npv - ore_npv
    rel = 0.0 if ore_npv == 0.0 else diff / abs(ore_npv)
    return {
        "trade_id": ctx.trade_id,
        "python_t0_npv": py_npv,
        "ore_t0_npv": ore_npv,
        "diff": diff,
        "rel_diff": rel,
    }


def _read_asof_date(ore_xml: Path) -> str:
    root = ET.parse(ore_xml).getroot()
    for node in root.findall("./Setup/Parameter"):
        if node.attrib.get("name") == "asofDate":
            return (node.text or "").strip()
    raise ValueError(f"asofDate not found in {ore_xml}")


def _first_trade_id(portfolio_xml: Path) -> str:
    root = ET.parse(portfolio_xml).getroot()
    trade = root.find("./Trade")
    if trade is None:
        raise ValueError(f"no Trade node found in {portfolio_xml}")
    trade_id = trade.attrib.get("id", "").strip()
    if not trade_id:
        raise ValueError(f"first Trade in {portfolio_xml} is missing id")
    return trade_id


def _simulation_grid_times(simulation_xml: Path) -> np.ndarray:
    root = ET.parse(simulation_xml).getroot()
    grid_text = (root.findtext("./Parameters/Grid") or "").strip()
    if "," not in grid_text:
        raise ValueError(f"unsupported Grid format '{grid_text}'")
    count_text, tenor_text = [x.strip() for x in grid_text.split(",", 1)]
    count = int(count_text)
    tenor = tenor_text.upper()
    if tenor.endswith("Y"):
        step = float(tenor[:-1])
    elif tenor.endswith("M"):
        step = float(tenor[:-1]) / 12.0
    elif tenor.endswith("W"):
        step = float(tenor[:-1]) / 52.0
    elif tenor.endswith("D"):
        step = float(tenor[:-1]) / 365.0
    else:
        raise ValueError(f"unsupported grid tenor '{tenor}'")
    return np.asarray([i * step for i in range(count + 1)], dtype=float)


__all__ = [
    "HW2FCaseContext",
    "load_hw2f_case_context",
    "swap_npv_from_ore_legs_hw",
    "price_hw2f_swap_t0",
    "simulate_hw2f_exposure_paths",
    "read_ore_t0_npv",
    "compare_hw2f_python_vs_ore",
]
