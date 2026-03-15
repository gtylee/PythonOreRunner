#!/usr/bin/env python3
"""Benchmark Cap/Floor and Bermudan Swaption vs ORE (PV + CVA)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
import sys

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

from pythonore.compute.lgm import LGM1F, LGMParams, simulate_lgm_measure
from pythonore.compute.lgm_fx_xva_utils import aggregate_exposure_profile, apply_mpor_closeout, cva_terms_from_profile
from pythonore.compute.lgm_ir_options import BermudanSwaptionDef, CapFloorDef, bermudan_npv_paths, capfloor_npv_paths
from pythonore.compute.irs_xva_utils import (
    build_discount_curve_from_discount_pairs,
    load_ore_default_curve_inputs,
    load_ore_discount_pairs_from_curves,
    load_ore_legs_from_flows,
    load_ore_exposure_profile,
    load_ore_exposure_times,
    survival_probability_from_hazard,
    swap_npv_from_ore_legs_dual_curve,
)
from pythonore.repo_paths import default_ore_bin, local_parity_artifacts_root, require_engine_repo_root

REPO_ROOT = require_engine_repo_root()
EXAMPLES_INPUT = REPO_ROOT / "Examples" / "Input"
EXPOSURE_INPUT = REPO_ROOT / "Examples" / "Exposure" / "Input"
ORE_BIN_DEFAULT = default_ore_bin()


@dataclass(frozen=True)
class Case:
    trade_id: str
    trade_type: str  # CAP or BERM


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ore-bin", type=Path, default=ORE_BIN_DEFAULT)
    p.add_argument(
        "--output-root",
        type=Path,
        default=local_parity_artifacts_root() / "ir_options_ore_benchmark",
    )
    p.add_argument("--ore-samples", type=int, default=2000)
    p.add_argument("--ore-seed", type=int, default=42)
    p.add_argument("--python-paths", type=int, default=10000)
    p.add_argument("--python-seed", type=int, default=4242)
    p.add_argument("--cpty-hazard", type=float, default=None, help="Override counterparty flat hazard for Python XVA.")
    p.add_argument("--own-hazard", type=float, default=0.010, help="Own flat hazard for Python DVA/FVA.")
    p.add_argument("--recovery-cpty", type=float, default=None, help="Override counterparty recovery for Python XVA.")
    p.add_argument("--recovery-own", type=float, default=0.40, help="Own recovery for Python DVA.")
    p.add_argument("--funding-spread", type=float, default=0.0015, help="Flat funding spread for Python FCA/FBA proxies.")
    p.add_argument("--mpor-days", type=float, default=0.0, help="Apply Python-side closeout mapping V(t)->V(t+MPOR).")
    p.add_argument(
        "--closeout-mode",
        choices=("sticky", "nonsticky"),
        default="sticky",
        help="Closeout mapping mode for Python side (nonsticky currently falls back to sticky map).",
    )
    return p.parse_args()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cap_trade_xml(trade_id: str, strike: float) -> str:
    return f"""  <Trade id="{trade_id}">
    <TradeType>CapFloor</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <CapFloorData>
      <LongShort>Long</LongShort>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>EUR</Currency>
        <DayCounter>ACT/365</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <ScheduleData>
          <Rules>
            <StartDate>20160805</StartDate>
            <EndDate>20180205</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>TARGET</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
        <FloatingLegData>
          <Index>EUR-EURIBOR-3M</Index>
          <Spreads>
            <Spread>0.0</Spread>
          </Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
      </LegData>
      <Caps>
        <Cap>{strike:.6f}</Cap>
      </Caps>
      <Floors/>
    </CapFloorData>
  </Trade>
"""


def _portfolio_xml() -> str:
    # As-of in ORE run will be 2016-02-05.
    cap_defs = [
        ("CAP_EUR_2Y_K00", 0.00),
        ("CAP_EUR_2Y_K01", 0.01),
        ("CAP_EUR_2Y_K02", 0.02),
        ("CAP_EUR_2Y", 0.03),
        ("CAP_EUR_2Y_K05", 0.05),
        ("CAP_EUR_2Y_OTM", 0.08),
    ]
    caps_xml = "".join(_cap_trade_xml(tid, k) for tid, k in cap_defs)
    return f"""<?xml version="1.0"?>
<Portfolio>
{caps_xml}

  <Trade id="BERM_EUR_5Y">
    <TradeType>Swaption</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwaptionData>
      <OptionData>
        <LongShort>Long</LongShort>
        <Style>Bermudan</Style>
        <Settlement>Physical</Settlement>
        <PayOffAtExpiry>false</PayOffAtExpiry>
        <ExerciseDates>
          <ExerciseDate>20170205</ExerciseDate>
          <ExerciseDate>20180205</ExerciseDate>
          <ExerciseDate>20190205</ExerciseDate>
        </ExerciseDates>
      </OptionData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>EUR</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>ACT/365</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>EUR-EURIBOR-6M</Index>
          <Spreads>
            <Spread>0.0</Spread>
          </Spreads>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20170205</StartDate>
            <EndDate>20210205</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>TARGET</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>false</Payer>
        <Currency>EUR</Currency>
        <Notionals>
          <Notional>10000000</Notional>
        </Notionals>
        <DayCounter>ACT/365</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FixedLegData>
          <Rates>
            <Rate>0.032</Rate>
          </Rates>
        </FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>20170205</StartDate>
            <EndDate>20210205</EndDate>
            <Tenor>1Y</Tenor>
            <Calendar>TARGET</Calendar>
            <Convention>MF</Convention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwaptionData>
  </Trade>
</Portfolio>
"""


def _simulation_xml(samples: int, seed: int) -> str:
    src = EXPOSURE_INPUT / "simulation_lgm_fixed.xml"
    root = ET.parse(src).getroot()
    params = root.find("./Parameters")
    if params is None:
        raise ValueError("simulation_lgm_fixed.xml missing Parameters")
    node = params.find("./Samples")
    if node is None:
        node = ET.SubElement(params, "Samples")
    node.text = str(int(samples))
    node = params.find("./Seed")
    if node is None:
        node = ET.SubElement(params, "Seed")
    node.text = str(int(seed))
    return ET.tostring(root, encoding="unicode")


def _ore_xml(output_dir: Path, input_dir: Path, simulation_file: Path) -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">2016-02-05</Parameter>
    <Parameter name="inputPath">{input_dir.as_posix()}</Parameter>
    <Parameter name="outputPath">{output_dir.as_posix()}</Parameter>
    <Parameter name="logFile">log.txt</Parameter>
    <Parameter name="logMask">31</Parameter>
    <Parameter name="marketDataFile">{(EXAMPLES_INPUT / "market_20160205_flat.txt").as_posix()}</Parameter>
    <Parameter name="fixingDataFile">{(EXAMPLES_INPUT / "fixings_20160205.txt").as_posix()}</Parameter>
    <Parameter name="implyTodaysFixings">Y</Parameter>
    <Parameter name="curveConfigFile">{(EXAMPLES_INPUT / "curveconfig.xml").as_posix()}</Parameter>
    <Parameter name="conventionsFile">{(EXAMPLES_INPUT / "conventions.xml").as_posix()}</Parameter>
    <Parameter name="marketConfigFile">{(EXAMPLES_INPUT / "todaysmarket.xml").as_posix()}</Parameter>
    <Parameter name="pricingEnginesFile">{(EXAMPLES_INPUT / "pricingengine.xml").as_posix()}</Parameter>
    <Parameter name="portfolioFile">{(input_dir / "portfolio.xml").as_posix()}</Parameter>
    <Parameter name="observationModel">None</Parameter>
    <Parameter name="continueOnError">false</Parameter>
    <Parameter name="calendarAdjustment">{(EXAMPLES_INPUT / "calendaradjustment.xml").as_posix()}</Parameter>
  </Setup>
  <Markets>
    <Parameter name="lgmcalibration">libor</Parameter>
    <Parameter name="pricing">libor</Parameter>
    <Parameter name="simulation">libor</Parameter>
  </Markets>
  <Analytics>
    <Analytic type="npv"><Parameter name="active">Y</Parameter><Parameter name="baseCurrency">EUR</Parameter><Parameter name="outputFileName">npv.csv</Parameter></Analytic>
    <Analytic type="cashflow"><Parameter name="active">Y</Parameter><Parameter name="outputFileName">flows.csv</Parameter></Analytic>
    <Analytic type="curves"><Parameter name="active">Y</Parameter><Parameter name="configuration">default</Parameter><Parameter name="grid">240,1M</Parameter><Parameter name="outputFileName">curves.csv</Parameter></Analytic>
    <Analytic type="simulation">
      <Parameter name="active">Y</Parameter>
      <Parameter name="simulationConfigFile">{simulation_file.as_posix()}</Parameter>
      <Parameter name="pricingEnginesFile">{(EXAMPLES_INPUT / "pricingengine.xml").as_posix()}</Parameter>
      <Parameter name="baseCurrency">EUR</Parameter>
      <Parameter name="cubeFile">cube.csv.gz</Parameter>
      <Parameter name="aggregationScenarioDataFileName">scenariodata.csv.gz</Parameter>
    </Analytic>
    <Analytic type="xva">
      <Parameter name="active">Y</Parameter>
      <Parameter name="useXvaRunner">N</Parameter>
      <Parameter name="csaFile">{(input_dir / "netting.xml").as_posix()}</Parameter>
      <Parameter name="cubeFile">cube.csv.gz</Parameter>
      <Parameter name="scenarioFile">scenariodata.csv.gz</Parameter>
      <Parameter name="baseCurrency">EUR</Parameter>
      <Parameter name="exposureProfiles">Y</Parameter>
      <Parameter name="exposureProfilesByTrade">Y</Parameter>
      <Parameter name="cva">Y</Parameter>
      <Parameter name="dva">Y</Parameter>
      <Parameter name="fva">Y</Parameter>
      <Parameter name="rawCubeOutputFile">rawcube.csv</Parameter>
      <Parameter name="netCubeOutputFile">netcube.csv</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def _load_trade_npv(npv_csv: Path, trade_id: str) -> float:
    with open(npv_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        tid_key = "TradeId" if r.fieldnames and "TradeId" in r.fieldnames else "#TradeId"
        for row in r:
            if row.get(tid_key, "") == trade_id:
                return float(row["NPV"])
    raise ValueError(f"trade {trade_id} not found in {npv_csv}")


def _parse_xva_float(s: str) -> float:
    x = (s or "").strip()
    if x in ("", "#N/A", "N/A", "nan", "NaN"):
        return float("nan")
    return float(x)


def _load_trade_xva(xva_csv: Path, trade_id: str) -> dict:
    with open(xva_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        tid_key = "TradeId" if r.fieldnames and "TradeId" in r.fieldnames else "#TradeId"
        for row in r:
            if row.get(tid_key, "") == trade_id:
                return {
                    "cva": _parse_xva_float(row.get("CVA", "")),
                    "dva": _parse_xva_float(row.get("DVA", "")),
                    "fca": _parse_xva_float(row.get("FCA", "")),
                    "fba": _parse_xva_float(row.get("FBA", "")),
                }
    raise ValueError(f"trade {trade_id} not found in {xva_csv}")


def _trade_cva_from_ore_exposure(
    exposure_trade_csv: Path,
    p0_disc,
    hazard_times: np.ndarray,
    hazard_rates: np.ndarray,
    recovery: float,
) -> float:
    prof = load_ore_exposure_profile(exposure_trade_csv.as_posix())
    t = np.asarray(prof["times"], dtype=float)
    epe = np.asarray(prof["epe"], dtype=float)
    q = survival_probability_from_hazard(t, hazard_times, hazard_rates)
    df = np.asarray([p0_disc(float(x)) for x in t], dtype=float)
    return float(cva_terms_from_profile(t, epe, df, q, recovery=recovery)["cva"][0])


def _py_cap_profile(model: LGM1F, p0_disc, p0_fwd, times: np.ndarray, n_paths: int, seed: int, strike: float) -> np.ndarray:
    x = simulate_lgm_measure(model, times, n_paths=n_paths, rng=np.random.default_rng(seed))
    start = np.arange(0.5, 2.0, 0.25, dtype=float)
    end = start + 0.25
    cf = CapFloorDef(
        trade_id="CAP_EUR_2Y",
        ccy="EUR",
        option_type="cap",
        start_time=start,
        end_time=end,
        pay_time=end.copy(),
        accrual=np.full_like(start, 0.25),
        notional=np.full_like(start, 10_000_000.0),
        strike=np.full_like(start, float(strike)),
        fixing_time=start.copy(),
        position=1.0,
    )
    return capfloor_npv_paths(model, p0_disc, p0_fwd, cf, times, x, lock_fixings=True)


def _py_berm_profile(model: LGM1F, p0_disc, p0_fwd, times: np.ndarray, n_paths: int, seed: int) -> np.ndarray:
    x = simulate_lgm_measure(model, times, n_paths=n_paths, rng=np.random.default_rng(seed))
    fixed_pay = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
    float_start = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=float)
    float_end = float_start + 0.5
    legs = {
        "fixed_pay_time": fixed_pay,
        "fixed_accrual": np.ones_like(fixed_pay),
        "fixed_rate": np.full_like(fixed_pay, 0.032),
        "fixed_notional": np.full_like(fixed_pay, 10_000_000.0),
        "fixed_sign": np.full_like(fixed_pay, -1.0),
        "fixed_amount": np.full_like(fixed_pay, -320_000.0),
        "float_pay_time": float_end,
        "float_start_time": float_start,
        "float_end_time": float_end,
        "float_accrual": np.full_like(float_start, 0.5),
        "float_notional": np.full_like(float_start, 10_000_000.0),
        "float_sign": np.full_like(float_start, 1.0),
        "float_spread": np.zeros_like(float_start),
        "float_coupon": np.zeros_like(float_start),
    }
    berm = BermudanSwaptionDef(
        trade_id="BERM_EUR_5Y",
        exercise_times=np.array([1.0, 2.0, 3.0], dtype=float),
        underlying_legs=legs,
        exercise_sign=1.0,
    )
    return bermudan_npv_paths(model, p0_disc, p0_fwd, berm, times, x, basis_degree=2, itm_only=True)


def _berm_lsmc_train_price(
    model: LGM1F,
    p0_disc,
    p0_fwd,
    berm: BermudanSwaptionDef,
    times: np.ndarray,
    n_train: int,
    n_price: int,
    seed_train: int,
    seed_price: int,
    degree: int,
    itm_only: bool,
    feature_set: str = "x",
) -> float:
    x_tr = simulate_lgm_measure(model, times, n_paths=n_train, rng=np.random.default_rng(seed_train))
    x_pr = simulate_lgm_measure(model, times, n_paths=n_price, rng=np.random.default_rng(seed_price))

    ex = np.asarray(berm.exercise_times, dtype=float)
    ex_idx = np.searchsorted(times, ex)
    flags = np.zeros(times.size, dtype=bool)
    flags[ex_idx] = True

    def _basis(x: np.ndarray, swap: np.ndarray | None) -> np.ndarray:
        cols = [np.ones_like(x)]
        for d in range(1, degree + 1):
            cols.append(np.power(x, d))
        if feature_set == "x_swap":
            if swap is None:
                raise ValueError("swap feature requires swap values")
            cols.extend([swap, swap * swap, x * swap])
        return np.column_stack(cols)

    betas = {}
    v = np.zeros_like(x_tr)
    for i in range(times.size - 2, -1, -1):
        disc = model.discount_bond(
            float(times[i]),
            float(times[i + 1]),
            x_tr[i, :],
            float(p0_disc(float(times[i]))),
            float(p0_disc(float(times[i + 1]))),
        )
        cont = disc * v[i + 1, :]
        if not flags[i]:
            v[i, :] = cont
            continue
        swap = swap_npv_from_ore_legs_dual_curve(model, p0_disc, p0_fwd, berm.underlying_legs, float(times[i]), x_tr[i, :])
        exer = np.maximum(float(berm.exercise_sign) * swap, 0.0)
        mask = np.ones_like(exer, dtype=bool)
        if itm_only:
            mask = exer > 1.0e-14
            if np.count_nonzero(mask) < max(16, degree + 4):
                mask = np.ones_like(mask, dtype=bool)
        a = _basis(x_tr[i, mask], swap[mask])
        b = cont[mask]
        beta, *_ = np.linalg.lstsq(a, b, rcond=None)
        betas[i] = beta
        ch = _basis(x_tr[i, :], swap) @ beta
        v[i, :] = np.where(exer >= ch, exer, cont)

    vp = np.zeros_like(x_pr)
    for i in range(times.size - 2, -1, -1):
        disc = model.discount_bond(
            float(times[i]),
            float(times[i + 1]),
            x_pr[i, :],
            float(p0_disc(float(times[i]))),
            float(p0_disc(float(times[i + 1]))),
        )
        cont = disc * vp[i + 1, :]
        if not flags[i]:
            vp[i, :] = cont
            continue
        swap = swap_npv_from_ore_legs_dual_curve(model, p0_disc, p0_fwd, berm.underlying_legs, float(times[i]), x_pr[i, :])
        exer = np.maximum(float(berm.exercise_sign) * swap, 0.0)
        ch = _basis(x_pr[i, :], swap) @ betas[i]
        vp[i, :] = np.where(exer >= ch, exer, cont)

    return float(np.mean(vp[0, :]))


def _build_berm_from_ore_flows(flows_csv: Path, asof: date) -> BermudanSwaptionDef:
    def _t(d: str) -> float:
        return (date.fromisoformat(d) - asof).days / 365.0

    rows = []
    with open(flows_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        tid_key = "TradeId" if r.fieldnames and "TradeId" in r.fieldnames else "#TradeId"
        for row in r:
            if row.get(tid_key, "") == "BERM_EUR_5Y" and row.get("FlowType", "").startswith("Interest"):
                rows.append(row)
    if not rows:
        raise ValueError("no Bermudan rows found in flows.csv")

    leg0 = [x for x in rows if x.get("LegNo", "") == "0"]
    leg1 = [x for x in rows if x.get("LegNo", "") == "1"]
    if not leg0 or not leg1:
        raise ValueError("failed to split Bermudan legs from flows.csv")

    # In this trade, leg0 is floating projected and leg1 is fixed.
    flt = sorted(leg0, key=lambda x: x["PayDate"])
    fix = sorted(leg1, key=lambda x: x["PayDate"])

    fixed_pay = np.array([_t(x["PayDate"]) for x in fix], dtype=float)
    fixed_accr = np.array([float(x["Accrual"]) for x in fix], dtype=float)
    fixed_rate = np.array([float(x["Coupon"]) for x in fix], dtype=float)
    fixed_notional = np.array([float(x["Notional"]) for x in fix], dtype=float)
    fixed_sign = np.array([np.sign(float(x["Amount"])) for x in fix], dtype=float)
    fixed_amount = np.array([float(x["Amount"]) for x in fix], dtype=float)

    float_start = np.array([_t(x["AccrualStartDate"]) for x in flt], dtype=float)
    float_end = np.array([_t(x["AccrualEndDate"]) for x in flt], dtype=float)
    float_pay = np.array([_t(x["PayDate"]) for x in flt], dtype=float)
    float_accr = np.array([float(x["Accrual"]) for x in flt], dtype=float)
    float_notional = np.array([float(x["Notional"]) for x in flt], dtype=float)
    float_sign = np.array([np.sign(float(x["Amount"])) for x in flt], dtype=float)

    legs = {
        "fixed_pay_time": fixed_pay,
        "fixed_accrual": fixed_accr,
        "fixed_rate": fixed_rate,
        "fixed_notional": fixed_notional,
        "fixed_sign": fixed_sign,
        "fixed_amount": fixed_amount,
        "float_pay_time": float_pay,
        "float_start_time": float_start,
        "float_end_time": float_end,
        "float_accrual": float_accr,
        "float_notional": float_notional,
        "float_sign": float_sign,
        "float_spread": np.zeros_like(float_accr),
        "float_coupon": np.zeros_like(float_accr),
    }
    exercise_times = np.array([_t("2017-02-05"), _t("2018-02-05"), _t("2019-02-05")], dtype=float)
    return BermudanSwaptionDef(
        trade_id="BERM_EUR_5Y",
        exercise_times=exercise_times,
        underlying_legs=legs,
        exercise_sign=1.0,
    )


def _fallback_times_for_case(case: Case) -> np.ndarray:
    if case.trade_type == "CAP":
        return np.linspace(0.0, 2.0, 25)
    return np.linspace(0.0, 5.0, 61)


def _map_to_effective_exercise_times(grid_times: np.ndarray, exercise_times: np.ndarray) -> np.ndarray:
    """ORE-style effective exercise mapping: first simulation date >= contract exercise date."""
    g = np.asarray(grid_times, dtype=float)
    e = np.asarray(exercise_times, dtype=float)
    if g.ndim != 1 or e.ndim != 1:
        raise ValueError("grid_times and exercise_times must be one-dimensional")
    if g.size == 0:
        raise ValueError("grid_times must be non-empty")
    if np.any(np.diff(g) <= 0.0):
        raise ValueError("grid_times must be strictly increasing")
    idx = np.searchsorted(g, e, side="left")
    idx = np.clip(idx, 0, g.size - 1)
    return g[idx]


def _compute_py_xva(
    npv_paths: np.ndarray,
    times: np.ndarray,
    p0_disc,
    cpty_hazard_times: np.ndarray,
    cpty_hazard_rates: np.ndarray,
    recovery_cpty: float,
    own_hazard_times: np.ndarray,
    own_hazard_rates: np.ndarray,
    recovery_own: float,
    funding_spread: float,
    mpor_days: float = 0.0,
    closeout_mode: str = "sticky",
) -> dict:
    npv_eff = apply_mpor_closeout(
        npv_paths,
        times,
        mpor_years=max(float(mpor_days), 0.0) / 365.0,
        sticky=(closeout_mode.lower() == "sticky"),
    )
    prof = aggregate_exposure_profile(npv_eff)
    epe = prof["epe"]
    ene = prof["ene"]  # negative by construction
    q_cpty = survival_probability_from_hazard(times, cpty_hazard_times, cpty_hazard_rates)
    q_own = survival_probability_from_hazard(times, own_hazard_times, own_hazard_rates)
    df = np.asarray([p0_disc(float(t)) for t in times], dtype=float)
    cva = float(cva_terms_from_profile(times, epe, df, q_cpty, recovery=recovery_cpty)["cva"][0])
    dva = float(cva_terms_from_profile(times, -ene, df, q_own, recovery=recovery_own)["cva"][0])
    dpd_own = np.zeros_like(times)
    dpd_own[1:] = np.clip(q_own[:-1] - q_own[1:], 0.0, None)
    fca = float(np.sum(float(funding_spread) * epe * df * dpd_own))
    fba = float(np.sum(float(funding_spread) * (-ene) * df * dpd_own))
    return {"cva": cva, "dva": dva, "fca": fca, "fba": fba}


def main() -> None:
    args = _parse_args()
    if not args.ore_bin.exists():
        raise FileNotFoundError(args.ore_bin)

    out_root = args.output_root
    inp = out_root / "Input"
    out = out_root / "Output"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    _write(inp / "portfolio.xml", _portfolio_xml())
    _write(inp / "simulation.xml", _simulation_xml(args.ore_samples, args.ore_seed))
    _write(inp / "netting.xml", (EXPOSURE_INPUT / "netting.xml").read_text(encoding="utf-8"))
    _write(inp / "ore.xml", _ore_xml(out, inp, inp / "simulation.xml"))

    t0 = perf_counter()
    cmd = [args.ore_bin.as_posix(), (inp / "ore.xml").as_posix()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_ore = perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"ORE run failed:\n{proc.stdout}\n{proc.stderr}")

    curves_csv = out / "curves.csv"
    npv_csv = out / "npv.csv"
    xva_csv = out / "xva.csv"
    if not (curves_csv.exists() and npv_csv.exists() and xva_csv.exists()):
        raise FileNotFoundError("expected ORE outputs missing")

    disc_t, disc_df = load_ore_discount_pairs_from_curves(curves_csv.as_posix(), discount_column="EUR-EONIA")
    p0_disc = build_discount_curve_from_discount_pairs(list(zip(disc_t.tolist(), disc_df.tolist())))
    p0_fwd = p0_disc
    f3_t, f3_df = load_ore_discount_pairs_from_curves(curves_csv.as_posix(), discount_column="EUR-EURIBOR-3M")
    p0_fwd_3m = build_discount_curve_from_discount_pairs(list(zip(f3_t.tolist(), f3_df.tolist())))

    lgm_params = LGMParams(
        alpha_times=(1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0),
        alpha_values=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
        kappa_times=(),
        kappa_values=(0.03,),
        shift=0.0,
        scaling=1.0,
    )
    model = LGM1F(lgm_params)
    berm_from_ore = _build_berm_from_ore_flows(out / "flows.csv", asof=date(2016, 2, 5))

    default_inputs = load_ore_default_curve_inputs(
        todaysmarket_xml=(EXAMPLES_INPUT / "todaysmarket.xml").as_posix(),
        market_data_file=(EXAMPLES_INPUT / "market_20160205_flat.txt").as_posix(),
        cpty_name="CPTY_A",
    )
    hazard_times = default_inputs["hazard_times"]
    hazard_rates = default_inputs["hazard_rates"]
    recovery = float(default_inputs["recovery"])

    cases = [
        Case(trade_id="CAP_EUR_2Y_K00", trade_type="CAP"),
        Case(trade_id="CAP_EUR_2Y_K01", trade_type="CAP"),
        Case(trade_id="CAP_EUR_2Y_K02", trade_type="CAP"),
        Case(trade_id="CAP_EUR_2Y", trade_type="CAP"),
        Case(trade_id="CAP_EUR_2Y_K05", trade_type="CAP"),
        Case(trade_id="CAP_EUR_2Y_OTM", trade_type="CAP"),
        Case(trade_id="BERM_EUR_5Y", trade_type="BERM"),
    ]

    rows = []
    for i, c in enumerate(cases):
        ore_pv = _load_trade_npv(npv_csv, c.trade_id)
        ore_cva = float("nan")
        ore_dva = float("nan")
        ore_fca = float("nan")
        ore_fba = float("nan")
        try:
            ore_xva = _load_trade_xva(xva_csv, c.trade_id)
            ore_cva = float(ore_xva["cva"])
            ore_dva = float(ore_xva["dva"])
            ore_fca = float(ore_xva["fca"])
            ore_fba = float(ore_xva["fba"])
        except ValueError:
            exposure_trade_file = out / f"exposure_trade_{c.trade_id}.csv"
            if exposure_trade_file.exists():
                ore_cva = _trade_cva_from_ore_exposure(
                    exposure_trade_file,
                    p0_disc,
                    hazard_times,
                    hazard_rates,
                    recovery,
                )
        exposure_trade_file = out / f"exposure_trade_{c.trade_id}.csv"
        if exposure_trade_file.exists():
            times = load_ore_exposure_times(exposure_trade_file.as_posix())
        else:
            times = _fallback_times_for_case(c)

        t1 = perf_counter()
        if c.trade_type == "CAP":
            strike_map = {
                "CAP_EUR_2Y_K00": 0.00,
                "CAP_EUR_2Y_K01": 0.01,
                "CAP_EUR_2Y_K02": 0.02,
                "CAP_EUR_2Y": 0.03,
                "CAP_EUR_2Y_K05": 0.05,
                "CAP_EUR_2Y_OTM": 0.08,
            }
            strike = strike_map[c.trade_id]
            npv_paths = _py_cap_profile(model, p0_disc, p0_fwd_3m, times, args.python_paths, args.python_seed + i, strike=strike)
        else:
            x = simulate_lgm_measure(model, times, n_paths=args.python_paths, rng=np.random.default_rng(args.python_seed + i))
            ex_adj = _map_to_effective_exercise_times(times, berm_from_ore.exercise_times)
            berm_on_grid = BermudanSwaptionDef(
                trade_id=berm_from_ore.trade_id,
                exercise_times=ex_adj,
                underlying_legs=berm_from_ore.underlying_legs,
                exercise_sign=berm_from_ore.exercise_sign,
            )
            npv_paths = bermudan_npv_paths(model, p0_disc, p0_fwd, berm_on_grid, times, x, basis_degree=2, itm_only=True)
        py_time = perf_counter() - t1

        py_pv = float(np.mean(npv_paths[0, :]))
        cpty_ht = hazard_times
        cpty_hr = hazard_rates
        cpty_rec = recovery
        if args.cpty_hazard is not None:
            cpty_ht = np.array([float(times[-1])], dtype=float)
            cpty_hr = np.array([float(args.cpty_hazard)], dtype=float)
        if args.recovery_cpty is not None:
            cpty_rec = float(args.recovery_cpty)

        own_ht = np.array([float(times[-1])], dtype=float)
        own_hr = np.array([float(args.own_hazard)], dtype=float)
        py_xva = _compute_py_xva(
            npv_paths,
            times,
            p0_disc,
            cpty_ht,
            cpty_hr,
            cpty_rec,
            own_ht,
            own_hr,
            float(args.recovery_own),
            float(args.funding_spread),
            mpor_days=args.mpor_days,
            closeout_mode=args.closeout_mode,
        )
        py_cva = float(py_xva["cva"])
        py_dva = float(py_xva["dva"])
        py_fca = float(py_xva["fca"])
        py_fba = float(py_xva["fba"])
        cva_abs = py_cva - ore_cva if np.isfinite(ore_cva) else float("nan")
        cva_rel = cva_abs / max(abs(ore_cva), 1.0) if np.isfinite(ore_cva) else float("nan")

        rows.append(
            {
                "trade_id": c.trade_id,
                "type": c.trade_type,
                "ore_pv": ore_pv,
                "py_pv": py_pv,
                "pv_abs_diff": py_pv - ore_pv,
                "pv_rel_diff": (py_pv - ore_pv) / max(abs(ore_pv), 1.0),
                "ore_cva": ore_cva,
                "py_cva": py_cva,
                "ore_dva": ore_dva,
                "py_dva": py_dva,
                "ore_fca": ore_fca,
                "py_fca": py_fca,
                "ore_fba": ore_fba,
                "py_fba": py_fba,
                "cva_abs_diff": cva_abs,
                "cva_rel_diff": cva_rel,
                "py_compute_s": py_time,
            }
        )

    # Bermudan diagnostics: curve choice and regression settings.
    berm_times = load_ore_exposure_times((out / "exposure_trade_BERM_EUR_5Y.csv").as_posix())
    x_dbg = simulate_lgm_measure(model, berm_times, n_paths=args.python_paths, rng=np.random.default_rng(args.python_seed + 99))
    ex_adj = _map_to_effective_exercise_times(berm_times, berm_from_ore.exercise_times)
    berm_on_grid = BermudanSwaptionDef(
        trade_id=berm_from_ore.trade_id,
        exercise_times=ex_adj,
        underlying_legs=berm_from_ore.underlying_legs,
        exercise_sign=berm_from_ore.exercise_sign,
    )
    # Dual-curve forward from ORE curve column.
    fwd_t, fwd_df = load_ore_discount_pairs_from_curves(curves_csv.as_posix(), discount_column="EUR-EURIBOR-6M")
    p0_fwd_6m = build_discount_curve_from_discount_pairs(list(zip(fwd_t.tolist(), fwd_df.tolist())))
    berm_dbg = {}
    for deg in (1, 2, 3):
        for itm in (True, False):
            key = f"dual_deg{deg}_{'itm' if itm else 'all'}"
            v = bermudan_npv_paths(model, p0_disc, p0_fwd_6m, berm_on_grid, berm_times, x_dbg, basis_degree=deg, itm_only=itm)
            berm_dbg[key] = float(np.mean(v[0, :]))
    berm_method_grid = {}
    ore_berm_pv = _load_trade_npv(npv_csv, "BERM_EUR_5Y")
    for deg in (1, 2, 3, 4):
        for itm in (True, False):
            for feat in ("x", "x_swap"):
                for fwd_name, fwd_curve in (("single", p0_disc), ("dual6m", p0_fwd_6m)):
                    k = f"{fwd_name}_{feat}_deg{deg}_{'itm' if itm else 'all'}"
                    pv = _berm_lsmc_train_price(
                        model,
                        p0_disc,
                        fwd_curve,
                        berm_on_grid,
                        berm_times,
                        n_train=max(4000, args.python_paths // 2),
                        n_price=max(4000, args.python_paths // 2),
                        seed_train=args.python_seed + 100 + deg,
                        seed_price=args.python_seed + 200 + deg,
                        degree=deg,
                        itm_only=itm,
                        feature_set=feat,
                    )
                    berm_method_grid[k] = {
                        "pv": pv,
                        "abs_diff_vs_ore": pv - ore_berm_pv,
                        "rel_diff_vs_ore": (pv - ore_berm_pv) / max(abs(ore_berm_pv), 1.0),
                    }
    best_key = min(berm_method_grid.keys(), key=lambda k: abs(berm_method_grid[k]["abs_diff_vs_ore"]))
    # Also compare single-curve dual-curve underlying swap at t0.
    legs_ore = load_ore_legs_from_flows((out / "flows.csv").as_posix(), trade_id="BERM_EUR_5Y", asof_date="2016-02-05")
    swap_single = float(swap_npv_from_ore_legs_dual_curve(model, p0_disc, p0_disc, legs_ore, 0.0, np.array([0.0]))[0])
    swap_dual = float(swap_npv_from_ore_legs_dual_curve(model, p0_disc, p0_fwd_6m, legs_ore, 0.0, np.array([0.0]))[0])

    with open(out_root / "benchmark_ir_options_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ore_seconds": t_ore,
                "ore_samples": int(args.ore_samples),
                "python_paths": int(args.python_paths),
                "rows": rows,
                "bermudan_diagnostics": {
                    "swap_t0_single_curve": swap_single,
                    "swap_t0_dual_curve": swap_dual,
                    "closeout": {"mode": args.closeout_mode, "mpor_days": float(args.mpor_days)},
                    "lsmc_variants": berm_dbg,
                    "method_grid": berm_method_grid,
                    "best_method": {"key": best_key, **berm_method_grid[best_key]},
                },
            },
            f,
            indent=2,
        )
    with open(out_root / "benchmark_ir_options_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(
        json.dumps(
            {
                "ore_seconds": t_ore,
                "rows": rows,
                "bermudan_diagnostics": {
                    "lsmc_variants": berm_dbg,
                    "best_method": {"key": best_key, **berm_method_grid[best_key]},
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
