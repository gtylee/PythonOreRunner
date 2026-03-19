#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_ore_tools.irs_xva_utils import load_swap_legs_from_portfolio
from py_ore_tools.ore_snapshot import (
    _date_from_time_with_day_counter,
    _resolve_ore_run_files,
    load_from_ore_xml,
)
from py_ore_tools.repo_paths import require_examples_repo_root


def _parse_args() -> argparse.Namespace:
    default_xml = require_examples_repo_root() / "Examples" / "Exposure" / "Input" / "ore_measure_lgm.xml"
    p = argparse.ArgumentParser(
        description="Compare ORE direct cashflows, Python cashflows, and XVA-state cashflow treatment."
    )
    p.add_argument("--ore-xml", type=Path, default=default_xml)
    p.add_argument("--trade-id", type=str, default=None)
    p.add_argument("--obs-date", type=str, default=None, help="Observation date for XVA-state classification")
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def _read_interest_flows(flows_csv: Path, trade_id: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(flows_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        trade_key = "TradeId" if "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get(trade_key, "") != trade_id:
                continue
            if not row.get("FlowType", "").startswith("Interest"):
                continue
            rows.append(row)
    return rows


def _raw_flow_key(row: Dict[str, str]) -> Tuple[str, str]:
    leg = "fixed" if row.get("LegNo", "") == "0" else "float"
    return leg, row["PayDate"]


def _amount_from_coupon(sign: float, notional: float, coupon: float, accrual: float) -> float:
    return float(sign) * float(notional) * float(coupon) * float(accrual)


def _rows_from_python_legs(legs: Dict[str, np.ndarray], snap, label: str) -> Dict[Tuple[str, str], Dict[str, object]]:
    out: Dict[Tuple[str, str], Dict[str, object]] = {}

    for i, pay_time in enumerate(np.asarray(legs["fixed_pay_time"], dtype=float)):
        pay_date = _iso_from_model_time(snap, float(pay_time))
        coupon = float(np.asarray(legs["fixed_rate"], dtype=float)[i])
        accrual = float(np.asarray(legs["fixed_accrual"], dtype=float)[i])
        notional = float(np.asarray(legs["fixed_notional"], dtype=float)[i])
        sign = float(np.asarray(legs["fixed_sign"], dtype=float)[i])
        amount = float(np.asarray(legs.get("fixed_amount"), dtype=float)[i]) if "fixed_amount" in legs else _amount_from_coupon(sign, notional, coupon, accrual)
        out[("fixed", pay_date)] = {
            f"{label}_pay_date": pay_date,
            f"{label}_pay_time_model": float(pay_time),
            f"{label}_pay_time_report": snap.report_time_from_date(pay_date),
            f"{label}_accrual": accrual,
            f"{label}_coupon": coupon,
            f"{label}_notional": notional,
            f"{label}_sign": sign,
            f"{label}_amount": amount,
        }

    float_fixing = np.asarray(legs.get("float_fixing_time", legs["float_start_time"]), dtype=float)
    float_coupon = np.asarray(legs.get("float_coupon", np.zeros_like(legs["float_accrual"])), dtype=float)
    float_amount = (
        np.asarray(legs["float_amount"], dtype=float)
        if "float_amount" in legs
        else np.asarray(
            [
                _amount_from_coupon(s, n, c, a)
                for s, n, c, a in zip(
                    np.asarray(legs["float_sign"], dtype=float),
                    np.asarray(legs["float_notional"], dtype=float),
                    float_coupon,
                    np.asarray(legs["float_accrual"], dtype=float),
                )
            ],
            dtype=float,
        )
    )

    for i, pay_time in enumerate(np.asarray(legs["float_pay_time"], dtype=float)):
        pay_date = _iso_from_model_time(snap, float(pay_time))
        start_time = float(np.asarray(legs["float_start_time"], dtype=float)[i])
        end_time = float(np.asarray(legs["float_end_time"], dtype=float)[i])
        fixing_time = float(float_fixing[i])
        out[("float", pay_date)] = {
            f"{label}_pay_date": pay_date,
            f"{label}_pay_time_model": float(pay_time),
            f"{label}_pay_time_report": snap.report_time_from_date(pay_date),
            f"{label}_start_date": _iso_from_model_time(snap, start_time),
            f"{label}_end_date": _iso_from_model_time(snap, end_time),
            f"{label}_fixing_date": _iso_from_model_time(snap, fixing_time),
            f"{label}_start_time_model": start_time,
            f"{label}_end_time_model": end_time,
            f"{label}_fixing_time_model": fixing_time,
            f"{label}_accrual": float(np.asarray(legs["float_accrual"], dtype=float)[i]),
            f"{label}_coupon": float(float_coupon[i]),
            f"{label}_notional": float(np.asarray(legs["float_notional"], dtype=float)[i]),
            f"{label}_sign": float(np.asarray(legs["float_sign"], dtype=float)[i]),
            f"{label}_spread": float(np.asarray(legs.get("float_spread", np.zeros_like(legs["float_accrual"])), dtype=float)[i]),
            f"{label}_amount": float(float_amount[i]),
        }
    return out


def _rows_from_ore_flows(rows: Iterable[Dict[str, str]], snap) -> Dict[Tuple[str, str], Dict[str, object]]:
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = _raw_flow_key(row)
        pay_date = row["PayDate"]
        base = {
            "ore_flow_leg": key[0],
            "ore_flow_pay_date": pay_date,
            "ore_flow_pay_time_model": snap.model_time_from_date(pay_date),
            "ore_flow_pay_time_report": snap.report_time_from_date(pay_date),
            "ore_flow_accrual": float(row.get("Accrual", "0") or 0.0),
            "ore_flow_coupon": float(row.get("Coupon", "0") or 0.0),
            "ore_flow_notional": float(row.get("Notional", "0") or 0.0),
            "ore_flow_amount": float(row.get("Amount", "0") or 0.0),
            "ore_flow_sign": float(np.sign(float(row.get("Amount", "0") or 0.0))),
        }
        if key[0] == "float":
            start_date = row.get("AccrualStartDate", "")
            end_date = row.get("AccrualEndDate", "")
            base.update(
                {
                    "ore_flow_start_date": start_date,
                    "ore_flow_end_date": end_date,
                    "ore_flow_start_time_model": snap.model_time_from_date(start_date) if start_date else "",
                    "ore_flow_end_time_model": snap.model_time_from_date(end_date) if end_date else "",
                    "ore_flow_fixing_date": start_date,
                    "ore_flow_fixing_time_model": snap.model_time_from_date(start_date) if start_date else "",
                }
            )
        out[key] = base
    return out


def _iso_from_model_time(snap, t: float) -> str:
    return _date_from_time_with_day_counter(snap.asof_date, float(t), snap.model_day_counter)


def _classify_xva_state(row: Dict[str, object], obs_model_time: float) -> str:
    pay_time = row.get("py_active_pay_time_model")
    if pay_time in ("", None) or float(pay_time) <= obs_model_time + 1.0e-12:
        return "dead"
    leg = row.get("leg")
    if leg == "fixed":
        return "fixed_alive"
    fixing_time = row.get("py_active_fixing_time_model")
    if fixing_time in ("", None):
        return "projected"
    return "fixed_alive" if float(fixing_time) <= obs_model_time + 1.0e-12 else "projected"


def _float_or_blank(value: object) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.10f}"


def main() -> None:
    args = _parse_args()
    snap = load_from_ore_xml(args.ore_xml, trade_id=args.trade_id)
    ore_xml, _, _, _, output_path = _resolve_ore_run_files(args.ore_xml)
    flows_csv = output_path / "flows.csv"
    if not flows_csv.exists():
        raise FileNotFoundError(f"flows.csv not found under {output_path}")

    ore_root = ET.parse(ore_xml).getroot()
    portfolio_rel = next(
        (
            (n.text or "").strip()
            for n in ore_root.findall("./Setup/Parameter")
            if n.attrib.get("name", "") == "portfolioFile"
        ),
        "portfolio.xml",
    )
    portfolio_xml = Path(portfolio_rel)
    if not portfolio_xml.is_absolute():
        portfolio_xml = (ore_xml.parent / portfolio_xml).resolve()

    ore_flow_rows = _rows_from_ore_flows(_read_interest_flows(flows_csv, snap.trade_id), snap)
    py_portfolio_rows = _rows_from_python_legs(
        load_swap_legs_from_portfolio(str(portfolio_xml), snap.trade_id, snap.asof_date),
        snap,
        "py_portfolio",
    )
    py_active_rows = _rows_from_python_legs(snap.legs, snap, "py_active")

    obs_date = args.obs_date or str(snap.exposure_dates[min(1, len(snap.exposure_dates) - 1)])
    obs_model_time = snap.model_time_from_date(obs_date)
    obs_report_time = snap.report_time_from_date(obs_date)

    keys = sorted(set(ore_flow_rows) | set(py_portfolio_rows) | set(py_active_rows), key=lambda x: (x[0], x[1]))
    out_csv = args.out_csv or (output_path / "cashflow_triptych.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "leg",
        "pay_date",
        "obs_date",
        "obs_time_model",
        "obs_time_report",
        "xva_state_at_obs",
        "ore_flow_leg",
        "ore_flow_pay_date",
        "ore_flow_pay_time_model",
        "ore_flow_pay_time_report",
        "ore_flow_start_date",
        "ore_flow_end_date",
        "ore_flow_fixing_date",
        "ore_flow_accrual",
        "ore_flow_coupon",
        "ore_flow_notional",
        "ore_flow_sign",
        "ore_flow_amount",
        "py_portfolio_pay_date",
        "py_portfolio_pay_time_model",
        "py_portfolio_pay_time_report",
        "py_portfolio_start_date",
        "py_portfolio_end_date",
        "py_portfolio_fixing_date",
        "py_portfolio_accrual",
        "py_portfolio_coupon",
        "py_portfolio_notional",
        "py_portfolio_sign",
        "py_portfolio_amount",
        "py_active_pay_date",
        "py_active_pay_time_model",
        "py_active_pay_time_report",
        "py_active_start_date",
        "py_active_end_date",
        "py_active_fixing_date",
        "py_active_accrual",
        "py_active_coupon",
        "py_active_notional",
        "py_active_sign",
        "py_active_amount",
        "diff_portfolio_vs_ore_amount",
        "diff_active_vs_ore_amount",
        "diff_active_vs_portfolio_amount",
    ]

    mismatch_portfolio = 0
    mismatch_active = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in keys:
            row = {"leg": key[0], "pay_date": key[1], "obs_date": obs_date}
            row.update(ore_flow_rows.get(key, {}))
            row.update(py_portfolio_rows.get(key, {}))
            row.update(py_active_rows.get(key, {}))
            row["obs_time_model"] = _float_or_blank(obs_model_time)
            row["obs_time_report"] = _float_or_blank(obs_report_time)
            row["xva_state_at_obs"] = _classify_xva_state(row, obs_model_time)

            ore_amt = row.get("ore_flow_amount", "")
            port_amt = row.get("py_portfolio_amount", "")
            active_amt = row.get("py_active_amount", "")

            if ore_amt != "" and port_amt != "":
                row["diff_portfolio_vs_ore_amount"] = float(port_amt) - float(ore_amt)
                if abs(row["diff_portfolio_vs_ore_amount"]) > 1.0e-8:
                    mismatch_portfolio += 1
            else:
                row["diff_portfolio_vs_ore_amount"] = ""

            if ore_amt != "" and active_amt != "":
                row["diff_active_vs_ore_amount"] = float(active_amt) - float(ore_amt)
                if abs(row["diff_active_vs_ore_amount"]) > 1.0e-8:
                    mismatch_active += 1
            else:
                row["diff_active_vs_ore_amount"] = ""

            if port_amt != "" and active_amt != "":
                row["diff_active_vs_portfolio_amount"] = float(active_amt) - float(port_amt)
            else:
                row["diff_active_vs_portfolio_amount"] = ""

            writer.writerow(row)

    print(f"Wrote: {out_csv}")
    print(f"Trade: {snap.trade_id}")
    print(f"Observation date: {obs_date}  model={obs_model_time:.10f}  report={obs_report_time:.10f}")
    print(f"Rows: {len(keys)}")
    print(f"Portfolio-vs-ORE amount mismatches: {mismatch_portfolio}")
    print(f"Active-vs-ORE amount mismatches: {mismatch_active}")


if __name__ == "__main__":
    main()
