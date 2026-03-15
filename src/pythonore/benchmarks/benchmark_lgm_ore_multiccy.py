#!/usr/bin/env python3
"""Benchmark Python LGM IRS CVA parity vs ORE across currencies, markets, maturities and conventions."""

from __future__ import annotations

import argparse
import json
import shutil
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

from pythonore.compute.lgm import LGM1F, LGMParams
from pythonore.compute.irs_xva_utils import (
    build_discount_curve_from_discount_pairs,
    load_ore_discount_pairs_from_curves,
    load_swap_legs_from_portfolio,
    parse_lgm_params_from_simulation_xml,
    swap_npv_from_ore_legs_dual_curve,
)
from pythonore.repo_paths import default_ore_bin, local_parity_artifacts_root, pythonorerunner_root, require_engine_repo_root


REPO_ROOT = require_engine_repo_root()
ORE_BIN_DEFAULT = default_ore_bin()
EXAMPLES_INPUT = REPO_ROOT / "Examples" / "Input"
EXPOSURE_INPUT = REPO_ROOT / "Examples" / "Exposure" / "Input"
COMPARE_SCRIPT = pythonorerunner_root() / "py_ore_tools" / "demos" / "compare_ore_python_lgm.py"


@dataclass(frozen=True)
class ConvProfile:
    key: str
    fixed_tenor: str
    fixed_day_counter: str
    fixed_payment_convention: str
    float_tenor: str
    float_day_counter: str
    float_payment_convention: str
    fixing_days: int


@dataclass(frozen=True)
class CaseDef:
    market_key: str
    ccy: str
    maturity_years: int
    conv_key: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ore-bin", type=Path, default=ORE_BIN_DEFAULT)
    p.add_argument("--output-root", type=Path, default=local_parity_artifacts_root() / "multiccy_benchmark")
    p.add_argument("--ore-samples", type=int, default=128)
    p.add_argument("--python-paths", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=0, help="0 means run all generated cases")
    return p.parse_args()


def _iso_to_yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _start_end_dates(maturity_years: int) -> tuple[str, str]:
    start = date(2016, 3, 1)
    end = date(start.year + maturity_years, start.month, start.day)
    return _iso_to_yyyymmdd(start), _iso_to_yyyymmdd(end)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _augment_todaysmarket_for_cad(src: Path, dst: Path) -> None:
    root = ET.parse(src).getroot()
    changed = False

    for dnode in root.findall("./DiscountingCurves"):
        did = dnode.attrib.get("id", "")
        if did == "inccy_swap":
            if not any(x.attrib.get("currency", "") == "CAD" for x in dnode.findall("./DiscountingCurve")):
                n = ET.SubElement(dnode, "DiscountingCurve")
                n.attrib["currency"] = "CAD"
                n.text = "Yield/CAD/CAD3M"
                changed = True
        if did == "ois":
            if not any(x.attrib.get("currency", "") == "CAD" for x in dnode.findall("./DiscountingCurve")):
                n = ET.SubElement(dnode, "DiscountingCurve")
                n.attrib["currency"] = "CAD"
                n.text = "Yield/CAD/CAD1D"
                changed = True

    for inode in root.findall("./IndexForwardingCurves"):
        if inode.attrib.get("id", "") != "default":
            continue
        names = {x.attrib.get("name", "") for x in inode.findall("./Index")}
        if "CAD-CDOR-3M" not in names:
            n = ET.SubElement(inode, "Index")
            n.attrib["name"] = "CAD-CDOR-3M"
            n.text = "Yield/CAD/CAD3M"
            changed = True
        if "CAD-CORRA" not in names:
            n = ET.SubElement(inode, "Index")
            n.attrib["name"] = "CAD-CORRA"
            n.text = "Yield/CAD/CAD1D"
            changed = True

    if not changed:
        shutil.copy2(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)


def _augment_market_for_cad(src: Path, dst: Path) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    out = list(lines)
    seen = set(lines)
    quote_map: dict[str, tuple[str, str]] = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 3 and not parts[0].startswith("#"):
            quote_map[parts[1]] = (parts[0], parts[2])
    prefixes = (
        "MM/RATE/USD/",
        "FRA/RATE/USD/",
        "IR_SWAP/RATE/USD/",
        "BASIS_SWAP/BASIS_SPREAD/",
    )

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        q = parts[1]
        if not q.startswith(prefixes):
            continue
        if "/USD/" not in q:
            continue
        q_cad = q.replace("/USD/", "/CAD/")
        new_ln = f"{parts[0]} {q_cad} {parts[2]}"
        if new_ln not in seen:
            seen.add(new_ln)
            out.append(new_ln)

    fx_line = "20160205 FX/RATE/USD/CAD 1.3000"
    if fx_line not in seen:
        out.append(fx_line)
    mm_line = "20160205 MM/RATE/CAD/0D/1D 0.0100"
    if mm_line not in seen:
        out.append(mm_line)
    # CAD OIS config expects 12M node explicitly; add alias from 1Y if available.
    alias_from = "IR_SWAP/RATE/CAD/2D/1D/1Y"
    alias_to = "IR_SWAP/RATE/CAD/2D/1D/12M"
    if not any(alias_to in ln for ln in out):
        for ln in out:
            parts = ln.split()
            if len(parts) >= 3 and parts[1] == alias_from:
                out.append(f"{parts[0]} {alias_to} {parts[2]}")
                break

    # Ensure mandatory CAD MM/FRA/swap quotes from curveconfig are present.
    curve_cfg = EXAMPLES_INPUT / "curveconfig.xml"
    req = set()
    for m in curve_cfg.read_text(encoding="utf-8").splitlines():
        s = m.strip()
        if "/CAD/" in s and (
            "MM/RATE/" in s
            or "FRA/RATE/" in s
            or "IR_SWAP/RATE/" in s
            or "BASIS_SWAP/BASIS_SPREAD/" in s
        ):
            a = s.find(">")
            b = s.rfind("<")
            if a >= 0 and b > a:
                q = s[a + 1 : b]
                req.add(q)

    current_quotes = {ln.split()[1] for ln in out if len(ln.split()) >= 3 and not ln.startswith("#")}
    for qcad in sorted(req):
        if qcad in current_quotes:
            continue
        qusd = qcad.replace("/CAD/", "/USD/")
        if qusd in quote_map:
            d, v = quote_map[qusd]
            out.append(f"{d} {qcad} {v}")
            current_quotes.add(qcad)
            continue
        # Fallback for odd monthly nodes not present in source
        d = "20160205"
        v = "0.0100"
        if "IR_SWAP/RATE/CAD/" in qcad:
            d2, v2 = quote_map.get("IR_SWAP/RATE/USD/2D/1D/1Y", (d, v))
            d, v = d2, v2
        elif "FRA/RATE/CAD/" in qcad:
            d2, v2 = quote_map.get("FRA/RATE/USD/0M/3M", (d, v))
            d, v = d2, v2
        out.append(f"{d} {qcad} {v}")
        current_quotes.add(qcad)

    _write_text(dst, "\n".join(out) + "\n")


def _simulation_xml(ccy: str, fwd_index: str, samples: int, seed: int) -> str:
    return f"""<?xml version="1.0"?>
<Simulation>
  <Parameters>
    <Discretization>Exact</Discretization>
    <Grid>121,1M</Grid>
    <Calendar>TARGET</Calendar>
    <Sequence>SobolBrownianBridge</Sequence>
    <Scenario>Simple</Scenario>
    <Seed>{seed}</Seed>
    <Samples>{samples}</Samples>
    <Ordering>Steps</Ordering>
    <DirectionIntegers>JoeKuoD7</DirectionIntegers>
  </Parameters>
  <CrossAssetModel>
    <DomesticCcy>{ccy}</DomesticCcy>
    <Currencies>
      <Currency>{ccy}</Currency>
    </Currencies>
    <BootstrapTolerance>0.0001</BootstrapTolerance>
    <Measure>LGM</Measure>
    <InterestRateModels>
      <LGM ccy="{ccy}">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility>
          <Calibrate>N</Calibrate>
          <VolatilityType>Hagan</VolatilityType>
          <ParamType>Piecewise</ParamType>
          <TimeGrid>1.0, 2.0, 3.0, 5.0, 7.0, 10.0</TimeGrid>
          <InitialValue>0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01</InitialValue>
        </Volatility>
        <Reversion>
          <Calibrate>N</Calibrate>
          <ReversionType>HullWhite</ReversionType>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>0.03</InitialValue>
        </Reversion>
        <ParameterTransformation>
          <ShiftHorizon>0.0</ShiftHorizon>
          <Scaling>1.0</Scaling>
        </ParameterTransformation>
      </LGM>
    </InterestRateModels>
    <InstantaneousCorrelations/>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>{ccy}</BaseCurrency>
    <Currencies>
      <Currency>{ccy}</Currency>
    </Currencies>
    <YieldCurves>
      <Configuration>
        <Tenors>3M,6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y</Tenors>
        <Interpolation>LogLinear</Interpolation>
        <Extrapolation>Y</Extrapolation>
      </Configuration>
    </YieldCurves>
    <Indices>
      <Index>{fwd_index}</Index>
    </Indices>
    <DefaultCurves>
      <Names/>
      <Tenors>6M,1Y,2Y</Tenors>
    </DefaultCurves>
    <AggregationScenarioDataCurrencies>
      <Currency>{ccy}</Currency>
    </AggregationScenarioDataCurrencies>
    <AggregationScenarioDataIndices>
      <Index>{fwd_index}</Index>
    </AggregationScenarioDataIndices>
  </Market>
</Simulation>
"""


def _portfolio_xml(
    trade_id: str,
    ccy: str,
    fwd_index: str,
    maturity_years: int,
    conv: ConvProfile,
    fixed_rate: float = 0.02,
    notional: float = 10_000_000.0,
) -> str:
    start, end = _start_end_dates(maturity_years)
    return f"""<?xml version="1.0"?>
<Portfolio>
  <Trade id="{trade_id}">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>false</Payer>
        <Currency>{ccy}</Currency>
        <Notionals><Notional>{notional:.6f}</Notional></Notionals>
        <DayCounter>{conv.fixed_day_counter}</DayCounter>
        <PaymentConvention>{conv.fixed_payment_convention}</PaymentConvention>
        <FixedLegData><Rates><Rate>{fixed_rate:.8f}</Rate></Rates></FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
            <Tenor>{conv.fixed_tenor}</Tenor>
            <Calendar>TARGET</Calendar>
            <Convention>{conv.fixed_payment_convention}</Convention>
            <TermConvention>{conv.fixed_payment_convention}</TermConvention>
            <Rule>Forward</Rule>
            <EndOfMonth/>
            <FirstDate/>
            <LastDate/>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>{ccy}</Currency>
        <Notionals><Notional>{notional:.6f}</Notional></Notionals>
        <DayCounter>{conv.float_day_counter}</DayCounter>
        <PaymentConvention>{conv.float_payment_convention}</PaymentConvention>
        <FloatingLegData>
          <Index>{fwd_index}</Index>
          <Spreads><Spread>0.00000000</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>{conv.fixing_days}</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{start}</StartDate>
            <EndDate>{end}</EndDate>
            <Tenor>{conv.float_tenor}</Tenor>
            <Calendar>TARGET</Calendar>
            <Convention>{conv.float_payment_convention}</Convention>
            <TermConvention>{conv.float_payment_convention}</TermConvention>
            <Rule>Forward</Rule>
            <EndOfMonth/>
            <FirstDate/>
            <LastDate/>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>
</Portfolio>
"""


def _ore_xml(
    output_dir: Path,
    market_file: Path,
    todaysmarket_file: Path,
    portfolio_file: Path,
    simulation_file: Path,
    base_ccy: str,
    xva_active: bool,
) -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">2016-02-05</Parameter>
    <Parameter name="inputPath">{portfolio_file.parent.as_posix()}</Parameter>
    <Parameter name="outputPath">{output_dir.as_posix()}</Parameter>
    <Parameter name="logFile">log.txt</Parameter>
    <Parameter name="logMask">31</Parameter>
    <Parameter name="marketDataFile">{market_file.as_posix()}</Parameter>
    <Parameter name="fixingDataFile">{(EXAMPLES_INPUT / "fixings_20160205.txt").as_posix()}</Parameter>
    <Parameter name="implyTodaysFixings">Y</Parameter>
    <Parameter name="curveConfigFile">{(EXAMPLES_INPUT / "curveconfig.xml").as_posix()}</Parameter>
    <Parameter name="conventionsFile">{(EXAMPLES_INPUT / "conventions.xml").as_posix()}</Parameter>
    <Parameter name="marketConfigFile">{todaysmarket_file.as_posix()}</Parameter>
    <Parameter name="pricingEnginesFile">{(EXAMPLES_INPUT / "pricingengine.xml").as_posix()}</Parameter>
    <Parameter name="portfolioFile">{portfolio_file.as_posix()}</Parameter>
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
    <Analytic type="npv">
      <Parameter name="active">Y</Parameter>
      <Parameter name="baseCurrency">{base_ccy}</Parameter>
      <Parameter name="outputFileName">npv.csv</Parameter>
    </Analytic>
    <Analytic type="cashflow">
      <Parameter name="active">Y</Parameter>
      <Parameter name="outputFileName">flows.csv</Parameter>
    </Analytic>
    <Analytic type="curves">
      <Parameter name="active">Y</Parameter>
      <Parameter name="configuration">default</Parameter>
      <Parameter name="grid">240,1M</Parameter>
      <Parameter name="outputFileName">curves.csv</Parameter>
    </Analytic>
    <Analytic type="simulation">
      <Parameter name="active">Y</Parameter>
      <Parameter name="simulationConfigFile">{simulation_file.as_posix()}</Parameter>
      <Parameter name="pricingEnginesFile">{(EXAMPLES_INPUT / "pricingengine.xml").as_posix()}</Parameter>
      <Parameter name="baseCurrency">{base_ccy}</Parameter>
      <Parameter name="cubeFile">cube.csv.gz</Parameter>
      <Parameter name="aggregationScenarioDataFileName">scenariodata.csv.gz</Parameter>
    </Analytic>
    <Analytic type="xva">
      <Parameter name="active">{"Y" if xva_active else "N"}</Parameter>
      <Parameter name="useXvaRunner">N</Parameter>
      <Parameter name="csaFile">{(EXPOSURE_INPUT / "netting.xml").as_posix()}</Parameter>
      <Parameter name="cubeFile">cube.csv.gz</Parameter>
      <Parameter name="scenarioFile">scenariodata.csv.gz</Parameter>
      <Parameter name="baseCurrency">{base_ccy}</Parameter>
      <Parameter name="exposureProfiles">Y</Parameter>
      <Parameter name="quantile">0.95</Parameter>
      <Parameter name="calculationType">Symmetric</Parameter>
      <Parameter name="allocationMethod">None</Parameter>
      <Parameter name="exerciseNextBreak">N</Parameter>
      <Parameter name="cva">Y</Parameter>
      <Parameter name="dva">N</Parameter>
      <Parameter name="dvaName">BANK</Parameter>
      <Parameter name="fva">N</Parameter>
      <Parameter name="colva">N</Parameter>
      <Parameter name="rawCubeOutputFile">rawcube.csv</Parameter>
      <Parameter name="netCubeOutputFile">netcube.csv</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def _run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def _market_file_for_case(case: CaseDef, market_files: dict[str, Path], cad_market_files: dict[str, Path]) -> Path:
    if case.ccy == "CAD":
        return cad_market_files[case.market_key]
    return market_files[case.market_key]


def _forward_index(ccy: str, conv: ConvProfile) -> str:
    if ccy == "EUR":
        return "EUR-EURIBOR-3M" if conv.float_tenor == "3M" else "EUR-EURIBOR-6M"
    if ccy == "USD":
        return "USD-LIBOR-3M" if conv.float_tenor == "3M" else "USD-LIBOR-6M"
    if ccy == "GBP":
        return "GBP-LIBOR-3M" if conv.float_tenor == "3M" else "GBP-LIBOR-6M"
    if ccy == "CAD":
        return "CAD-CDOR-3M"
    raise ValueError(f"unsupported ccy {ccy}")


def _discount_column(ccy: str) -> str:
    if ccy == "EUR":
        return "EUR-EURIBOR-6M"
    if ccy == "USD":
        return "USD-LIBOR-3M"
    if ccy == "GBP":
        return "GBP-LIBOR-6M"
    if ccy == "CAD":
        return "CAD-CDOR-3M"
    raise ValueError(f"unsupported ccy {ccy}")


def _build_cases() -> tuple[list[CaseDef], dict[str, ConvProfile]]:
    convs = {
        "A": ConvProfile("A", "1Y", "30/360", "MF", "6M", "A360", "MF", 2),
        "B": ConvProfile("B", "6M", "A365", "F", "3M", "A365", "F", 2),
    }
    cases: list[CaseDef] = []
    for market_key in ("flat",):
        for ccy in ("EUR", "USD", "GBP", "CAD"):
            for maturity in (5, 10):
                for conv in ("A", "B"):
                    cases.append(CaseDef(market_key, ccy, maturity, conv))
    for market_key in ("full",):
        for ccy in ("EUR", "USD", "GBP", "CAD"):
            for maturity in (5, 10):
                cases.append(CaseDef(market_key, ccy, maturity, "A"))
    return cases, convs


def _load_ore_trade_npv(npv_csv: Path, trade_id: str) -> float:
    lines = npv_csv.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"empty npv csv: {npv_csv}")
    header = lines[0].split(",")
    rows = [x.split(",") for x in lines[1:] if x.strip()]
    tid_idx = header.index("#TradeId") if "#TradeId" in header else header.index("TradeId")
    npv_idx = header.index("NPV")
    for r in rows:
        if r[tid_idx] == trade_id:
            return float(r[npv_idx])
    raise ValueError(f"trade {trade_id} not found in {npv_csv}")


def _python_t0_npv(simulation_xml: Path, curves_csv: Path, portfolio_xml: Path, trade_id: str, model_ccy: str, disc_col: str, fwd_col: str) -> float:
    t_d, df_d = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=disc_col)
    t_f, df_f = load_ore_discount_pairs_from_curves(str(curves_csv), discount_column=fwd_col)
    p0_d = build_discount_curve_from_discount_pairs(list(zip(t_d, df_d)))
    p0_f = build_discount_curve_from_discount_pairs(list(zip(t_f, df_f)))
    params = parse_lgm_params_from_simulation_xml(str(simulation_xml), ccy_key=model_ccy)
    model = LGM1F(
        LGMParams(
            alpha_times=tuple(float(x) for x in params["alpha_times"]),
            alpha_values=tuple(float(x) for x in params["alpha_values"]),
            kappa_times=tuple(float(x) for x in params["kappa_times"]),
            kappa_values=tuple(float(x) for x in params["kappa_values"]),
            shift=float(params["shift"]),
            scaling=float(params["scaling"]),
        )
    )
    legs = load_swap_legs_from_portfolio(str(portfolio_xml), trade_id=trade_id, asof_date="2016-02-05")
    x0 = np.array([0.0], dtype=float)
    return float(swap_npv_from_ore_legs_dual_curve(model, p0_d, p0_f, legs, 0.0, x0)[0])


def main() -> None:
    args = _parse_args()
    if not args.ore_bin.exists():
        raise FileNotFoundError(f"ore binary not found: {args.ore_bin}")
    if not COMPARE_SCRIPT.exists():
        raise FileNotFoundError(COMPARE_SCRIPT)

    run_root = args.output_root
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    market_files = {
        "flat": EXAMPLES_INPUT / "market_20160205_flat.txt",
        "full": EXAMPLES_INPUT / "market_20160205.txt",
    }
    cad_market_files = {
        "flat": run_root / "shared" / "market_20160205_flat_with_cad.txt",
        "full": run_root / "shared" / "market_20160205_with_cad.txt",
    }
    for k, src in market_files.items():
        _augment_market_for_cad(src, cad_market_files[k])

    todaysmarket_src = EXAMPLES_INPUT / "todaysmarket.xml"
    todaysmarket_cad = run_root / "shared" / "todaysmarket_with_cad.xml"
    _augment_todaysmarket_for_cad(todaysmarket_src, todaysmarket_cad)

    cases, conv_profiles = _build_cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    py_art_root = run_root / "python_compare"
    py_art_root.mkdir(parents=True, exist_ok=True)

    results = []
    bench_t0 = perf_counter()
    for idx, case in enumerate(cases, start=1):
        case_t0 = perf_counter()
        conv = conv_profiles[case.conv_key]
        conv_for_case = conv
        if case.ccy == "CAD" and conv.float_tenor == "6M":
            conv_for_case = ConvProfile("A_CAD", "1Y", "30/360", "MF", "3M", "A360", "MF", 2)

        trade_id = f"SWAP_{case.ccy}_{case.maturity_years}Y_{conv_for_case.key}_{case.market_key}".replace("-", "_")
        case_id = f"{case.market_key}_{case.ccy}_{case.maturity_years}Y_{conv_for_case.key}"
        case_dir = run_root / "cases" / case_id
        input_dir = case_dir / "Input"
        output_dir = case_dir / "Output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        fwd_index = _forward_index(case.ccy, conv_for_case)
        disc_col = _discount_column(case.ccy)
        mkt_file = _market_file_for_case(case, market_files, cad_market_files)
        tm_file = todaysmarket_cad if case.ccy == "CAD" else todaysmarket_src
        xva_active = case.ccy != "CAD"

        portfolio_xml = input_dir / "portfolio.xml"
        simulation_xml = input_dir / "simulation.xml"
        ore_xml = input_dir / "ore.xml"

        _write_text(portfolio_xml, _portfolio_xml(trade_id, case.ccy, fwd_index, case.maturity_years, conv_for_case))
        _write_text(simulation_xml, _simulation_xml(case.ccy, fwd_index, args.ore_samples, args.seed))
        _write_text(ore_xml, _ore_xml(output_dir, mkt_file, tm_file, portfolio_xml, simulation_xml, case.ccy, xva_active))

        ore_t0 = perf_counter()
        ore_run = _run_cmd([str(args.ore_bin), str(ore_xml)], cwd=case_dir)
        ore_seconds = perf_counter() - ore_t0
        if ore_run.returncode != 0:
            results.append(
                {
                    "case_id": case_id,
                    "status": "ore_error",
                    "ore_returncode": ore_run.returncode,
                    "ore_stderr_tail": ore_run.stderr.splitlines()[-1] if ore_run.stderr else "",
                    "ore_seconds": ore_seconds,
                    "total_case_seconds": perf_counter() - case_t0,
                }
            )
            continue

        if not xva_active:
            try:
                npv_t0 = perf_counter()
                ore_npv = _load_ore_trade_npv(output_dir / "npv.csv", trade_id)
                py_npv = _python_t0_npv(simulation_xml, output_dir / "curves.csv", portfolio_xml, trade_id, case.ccy, disc_col, fwd_index)
                npv_seconds = perf_counter() - npv_t0
                rec = {
                    "case_id": case_id,
                    "status": "npv_only",
                    "market_key": case.market_key,
                    "ccy": case.ccy,
                    "maturity_years": case.maturity_years,
                    "conv_key": conv_for_case.key,
                    "trade_id": trade_id,
                    "discount_column": disc_col,
                    "forward_column": fwd_index,
                    "ore_npv": float(ore_npv),
                    "py_npv": float(py_npv),
                    "npv_rel_diff": abs(py_npv - ore_npv) / max(abs(ore_npv), 1.0),
                    "ore_xml": str(ore_xml),
                    "simulation_xml": str(simulation_xml),
                    "portfolio_xml": str(portfolio_xml),
                    "output_dir": str(output_dir),
                    "ore_seconds": ore_seconds,
                    "npv_seconds": npv_seconds,
                    "total_case_seconds": perf_counter() - case_t0,
                }
                results.append(rec)
                print(f"[{idx}/{len(cases)}] {case_id}: npv_rel_diff={rec['npv_rel_diff']:.4%} (NPV-only)")
            except Exception as e:
                results.append(
                    {
                        "case_id": case_id,
                        "status": "npv_only_error",
                        "error": str(e),
                        "ore_seconds": ore_seconds,
                        "total_case_seconds": perf_counter() - case_t0,
                    }
                )
            continue

        cmp_t0 = perf_counter()
        compare_cmd = [
            "python3",
            str(COMPARE_SCRIPT),
            "--scenario",
            "fixed",
            "--ore-input-xml",
            str(ore_xml),
            "--simulation-xml",
            str(simulation_xml),
            "--ore-output-dir",
            str(output_dir),
            "--discount-column",
            disc_col,
            "--forward-column",
            fwd_index,
            "--cpty",
            "CPTY_A",
            "--trade-id",
            trade_id,
            "--swap-source",
            "trade",
            "--portfolio-xml",
            str(portfolio_xml),
            "--paths",
            str(args.python_paths),
            "--seed",
            str(args.seed),
            "--alpha-source",
            "simulation",
            "--model-ccy",
            case.ccy,
            "--no-coupon-spread-calibration",
            "--artifact-root",
            str(py_art_root),
            "--out-prefix",
            case_id,
        ]
        cmp_run = _run_cmd(compare_cmd, cwd=case_dir)
        compare_seconds = perf_counter() - cmp_t0
        summary_path = py_art_root / "fixed" / f"{case_id}_summary.json"
        if cmp_run.returncode != 0 or not summary_path.exists():
            results.append(
                {
                    "case_id": case_id,
                    "status": "compare_error",
                    "compare_returncode": cmp_run.returncode,
                    "compare_stderr_tail": cmp_run.stderr.splitlines()[-1] if cmp_run.stderr else "",
                    "ore_seconds": ore_seconds,
                    "compare_seconds": compare_seconds,
                    "total_case_seconds": perf_counter() - case_t0,
                }
            )
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rec = {
            "case_id": case_id,
            "status": "ok",
            "market_key": case.market_key,
            "ccy": case.ccy,
            "maturity_years": case.maturity_years,
            "conv_key": conv_for_case.key,
            "trade_id": trade_id,
            "discount_column": disc_col,
            "forward_column": fwd_index,
            "ore_cva": float(summary["ore_cva"]),
            "py_cva": float(summary["py_cva"]),
            "cva_rel_diff": float(summary["cva_rel_diff"]),
            "ore_xml": str(ore_xml),
            "simulation_xml": str(simulation_xml),
            "portfolio_xml": str(portfolio_xml),
            "output_dir": str(output_dir),
            "summary_json": str(summary_path),
            "ore_seconds": ore_seconds,
            "compare_seconds": compare_seconds,
            "total_case_seconds": perf_counter() - case_t0,
        }
        results.append(rec)
        print(f"[{idx}/{len(cases)}] {case_id}: rel_diff={rec['cva_rel_diff']:.4%}")

    out_json = run_root / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    ok = [r for r in results if r.get("status") == "ok"]
    npv_only = [r for r in results if r.get("status") == "npv_only"]
    ok_sorted = sorted(ok, key=lambda x: abs(float(x["cva_rel_diff"])))
    total_elapsed = perf_counter() - bench_t0

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        a = np.asarray(vals, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)),
            "max": float(np.max(a)),
        }

    ore_stats = _stats([float(r.get("ore_seconds", 0.0)) for r in results if "ore_seconds" in r])
    cmp_stats = _stats([float(r.get("compare_seconds", 0.0)) for r in ok if "compare_seconds" in r])
    npv_stats = _stats([float(r.get("npv_seconds", 0.0)) for r in npv_only if "npv_seconds" in r])
    total_case_stats = _stats([float(r.get("total_case_seconds", 0.0)) for r in results if "total_case_seconds" in r])
    slowest = sorted(
        [r for r in results if "total_case_seconds" in r],
        key=lambda x: float(x["total_case_seconds"]),
        reverse=True,
    )[:10]
    lines = [
        "# LGM ORE vs Python Multicurrency Benchmark",
        "",
        f"- Cases run: {len(results)}",
        f"- Successful: {len(ok)}",
        f"- NPV-only (CAD): {len(npv_only)}",
        f"- ORE samples per run: {args.ore_samples}",
        f"- Python paths per run: {args.python_paths}",
        f"- Seed: {args.seed}",
        f"- Total benchmark wall time: {total_elapsed:.2f}s",
        "",
        "## Performance",
        "",
        "| Stage | Mean (s) | Median (s) | P95 (s) | Max (s) |",
        "|---|---:|---:|---:|---:|",
        f"| ORE run | {ore_stats['mean']:.2f} | {ore_stats['median']:.2f} | {ore_stats['p95']:.2f} | {ore_stats['max']:.2f} |",
        f"| Python compare (CVA cases) | {cmp_stats['mean']:.2f} | {cmp_stats['median']:.2f} | {cmp_stats['p95']:.2f} | {cmp_stats['max']:.2f} |",
        f"| Python NPV-only (CAD) | {npv_stats['mean']:.4f} | {npv_stats['median']:.4f} | {npv_stats['p95']:.4f} | {npv_stats['max']:.4f} |",
        f"| End-to-end per case | {total_case_stats['mean']:.2f} | {total_case_stats['median']:.2f} | {total_case_stats['p95']:.2f} | {total_case_stats['max']:.2f} |",
        "",
        "### Slowest Cases",
        "",
        "| Case | Status | Total Case Seconds | ORE Seconds | Compare Seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in slowest:
        lines.append(
            f"| {r['case_id']} | {r['status']} | {float(r.get('total_case_seconds', 0.0)):.2f} | "
            f"{float(r.get('ore_seconds', 0.0)):.2f} | {float(r.get('compare_seconds', 0.0)):.2f} |"
        )

    lines += [
        "",
        "## Best Cases (By |CVA Rel Diff|)",
        "",
        "| Case | Ccy | Market | Tenor | Conv | ORE CVA | PY CVA | Rel Diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ok_sorted[:10]:
        lines.append(
            f"| {r['case_id']} | {r['ccy']} | {r['market_key']} | {r['maturity_years']}Y | {r['conv_key']} | "
            f"{r['ore_cva']:.2f} | {r['py_cva']:.2f} | {r['cva_rel_diff']:.4%} |"
        )

    lines += [
        "",
        "## CAD NPV-Only Cases",
        "",
        "| Case | Ccy | Market | Tenor | Conv | ORE NPV | PY NPV | Rel Diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(npv_only, key=lambda x: abs(float(x["npv_rel_diff"]))):
        lines.append(
            f"| {r['case_id']} | {r['ccy']} | {r['market_key']} | {r['maturity_years']}Y | {r['conv_key']} | "
            f"{r['ore_npv']:.2f} | {r['py_npv']:.2f} | {r['npv_rel_diff']:.4%} |"
        )

    lines += [
        "",
        "## All Successful Cases",
        "",
        "| Case | Ccy | Market | Tenor | Conv | ORE CVA | PY CVA | Rel Diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ok_sorted:
        lines.append(
            f"| {r['case_id']} | {r['ccy']} | {r['market_key']} | {r['maturity_years']}Y | {r['conv_key']} | "
            f"{r['ore_cva']:.2f} | {r['py_cva']:.2f} | {r['cva_rel_diff']:.4%} |"
        )

    lines += [
        "",
        "## ORE Settings Storage",
        "",
        "Per case, settings and outputs are stored under:",
        f"- `{(run_root / 'cases').as_posix()}`",
        "",
        "Each case directory contains:",
        "- `Input/ore.xml`",
        "- `Input/simulation.xml`",
        "- `Input/portfolio.xml`",
        "- `Output/` (ORE run outputs)",
        "",
        f"Machine-readable index: `{out_json.as_posix()}`",
    ]
    report_md = run_root / "LGM_MULTICCY_BENCHMARK_REPORT.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Results: {out_json}")
    print(f"Report:  {report_md}")


if __name__ == "__main__":
    main()
