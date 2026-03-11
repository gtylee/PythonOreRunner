#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent

if str(TOOLS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TOOLS_DIR))

from py_ore_tools.repo_paths import default_ore_bin, local_parity_artifacts_root, require_engine_repo_root


def _bootstrap() -> None:
    import sys

    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))


def _locate_ore_exe() -> Path:
    candidate = default_ore_bin()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"ORE executable not found: {candidate}")


def _case_input_dir(case_name: str) -> Path:
    return TOOLS_DIR / "parity_artifacts" / "bermudan_method_compare" / case_name / "Input"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ore_market_quote_for_factor(normalized_factor: str) -> str:
    mapping = {
        "zero:EUR:10Y": "IR_SWAP/RATE/EUR/2D/1D/10Y",
        "fwd:EUR:6M:10Y": "IR_SWAP/RATE/EUR/2D/6M/10Y",
    }
    try:
        return mapping[normalized_factor]
    except KeyError as exc:
        raise KeyError(f"No direct ORE quote mapping for factor '{normalized_factor}'") from exc


def _prepare_ore_sensitivity_run(case_input: Path, output_root: Path, shift_size: float) -> Path:
    run_root = output_root / case_input.parent.name
    if run_root.exists():
        shutil.rmtree(run_root)
    (run_root / "Input").mkdir(parents=True, exist_ok=True)
    (run_root / "Output").mkdir(parents=True, exist_ok=True)

    _write_text(
        run_root / "Input" / "sensitivity.xml",
        textwrap.dedent(
            f"""\
            <?xml version="1.0"?>
            <SensitivityAnalysis>
              <DiscountCurves>
                <DiscountCurve ccy="EUR">
                  <ShiftType>Absolute</ShiftType>
                  <ShiftSize>{shift_size:.10f}</ShiftSize>
                  <ShiftScheme>Forward</ShiftScheme>
                  <ShiftTenors>10Y</ShiftTenors>
                  <ParConversion>
                    <Instruments>OIS</Instruments>
                    <SingleCurve>true</SingleCurve>
                    <Conventions>
                      <Convention id="DEP">EUR-EONIA-CONVENTIONS</Convention>
                      <Convention id="OIS">EUR-OIS-CONVENTIONS</Convention>
                    </Conventions>
                  </ParConversion>
                </DiscountCurve>
              </DiscountCurves>
              <IndexCurves>
                <IndexCurve index="EUR-EURIBOR-6M">
                  <ShiftType>Absolute</ShiftType>
                  <ShiftSize>{shift_size:.10f}</ShiftSize>
                  <ShiftScheme>Forward</ShiftScheme>
                  <ShiftTenors>10Y</ShiftTenors>
                  <ParConversion>
                    <Instruments>FRA, IRS</Instruments>
                    <SingleCurve>false</SingleCurve>
                    <DiscountCurve>EUR1D</DiscountCurve>
                    <Conventions>
                      <Convention id="DEP">EUR-DEPOSIT</Convention>
                      <Convention id="FRA">EUR-6M-FRA-CONVENTIONS</Convention>
                      <Convention id="IRS">EUR-6M-SWAP-CONVENTIONS</Convention>
                    </Conventions>
                  </ParConversion>
                </IndexCurve>
              </IndexCurves>
              <YieldCurves/>
              <FxSpots/>
              <FxVolatilities/>
              <SwaptionVolatilities/>
              <CapFloorVolatilities/>
              <CDSVolatilities/>
              <CreditCurves/>
              <EquitySpots/>
              <EquityVolatilities/>
              <ZeroInflationIndexCurves/>
              <YYInflationIndexCurves/>
              <BaseCorrelations/>
              <SecuritySpreads/>
              <Correlations/>
              <CrossGammaFilter/>
            </SensitivityAnalysis>
            """
        ),
    )
    _write_text(
        run_root / "Input" / "simulation_sensitivity.xml",
        textwrap.dedent(
            """\
            <Simulation>
              <Market>
                <BaseCurrency>EUR</BaseCurrency>
                <Currencies>
                  <Currency>EUR</Currency>
                </Currencies>
                <YieldCurves>
                  <Configuration curve="">
                    <Tenors>6M,1Y,2Y,3Y,5Y,7Y,10Y,15Y,20Y,30Y</Tenors>
                    <Interpolation>LogLinear</Interpolation>
                    <Extrapolation>true</Extrapolation>
                  </Configuration>
                </YieldCurves>
                <Indices>
                  <Index>EUR-EONIA</Index>
                  <Index>EUR-EURIBOR-3M</Index>
                  <Index>EUR-EURIBOR-6M</Index>
                </Indices>
                <SwapIndices>
                  <SwapIndex>
                    <Name>EUR-CMS-1Y</Name>
                    <DiscountingIndex>EUR-EONIA</DiscountingIndex>
                  </SwapIndex>
                  <SwapIndex>
                    <Name>EUR-CMS-30Y</Name>
                    <DiscountingIndex>EUR-EONIA</DiscountingIndex>
                  </SwapIndex>
                </SwapIndices>
                <SwaptionVolatilities>
                  <Simulate>true</Simulate>
                  <ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay>
                  <Currencies>
                    <Currency>EUR</Currency>
                  </Currencies>
                  <Expiries>1Y,2Y,3Y,5Y,10Y,15Y,20Y,30Y</Expiries>
                  <Terms>1Y,2Y,3Y,5Y,10Y,15Y,20Y,30Y</Terms>
                  <DayCounters>
                    <DayCounter ccy="">A365</DayCounter>
                  </DayCounters>
                </SwaptionVolatilities>
              </Market>
            </Simulation>
            """
        ),
    )
    _write_text(
        run_root / "Input" / "todaysmarket.xml",
        textwrap.dedent(
            """\
            <?xml version="1.0"?>
            <TodaysMarket>
              <Configuration id="default">
                <DiscountingCurvesId>xois_eur</DiscountingCurvesId>
                <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
                <SwaptionVolatilitiesId>default</SwaptionVolatilitiesId>
                <SwapIndexCurvesId>default</SwapIndexCurvesId>
              </Configuration>
              <Configuration id="inccy">
                <DiscountingCurvesId>ois</DiscountingCurvesId>
                <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
                <SwaptionVolatilitiesId>default</SwaptionVolatilitiesId>
                <SwapIndexCurvesId>default</SwapIndexCurvesId>
              </Configuration>
              <Configuration id="xois_eur">
                <DiscountingCurvesId>xois_eur</DiscountingCurvesId>
                <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
                <SwaptionVolatilitiesId>default</SwaptionVolatilitiesId>
                <SwapIndexCurvesId>default</SwapIndexCurvesId>
              </Configuration>
              <Configuration id="collateral_inccy">
                <DiscountingCurvesId>ois</DiscountingCurvesId>
                <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
                <SwaptionVolatilitiesId>default</SwaptionVolatilitiesId>
                <SwapIndexCurvesId>default</SwapIndexCurvesId>
              </Configuration>
              <DiscountingCurves id="ois">
                <DiscountingCurve currency="EUR">Yield/EUR/EUR1D</DiscountingCurve>
              </DiscountingCurves>
              <DiscountingCurves id="xois_eur">
                <DiscountingCurve currency="EUR">Yield/EUR/EUR1D</DiscountingCurve>
              </DiscountingCurves>
              <IndexForwardingCurves id="default">
                <Index name="EUR-EONIA">Yield/EUR/EUR1D</Index>
                <Index name="EUR-EURIBOR-3M">Yield/EUR/EUR3M</Index>
                <Index name="EUR-EURIBOR-6M">Yield/EUR/EUR6M</Index>
              </IndexForwardingCurves>
              <SwapIndexCurves id="default">
                <SwapIndex name="EUR-CMS-1Y">
                  <Discounting>EUR-EONIA</Discounting>
                </SwapIndex>
                <SwapIndex name="EUR-CMS-30Y">
                  <Discounting>EUR-EONIA</Discounting>
                </SwapIndex>
              </SwapIndexCurves>
              <SwaptionVolatilities id="default">
                <SwaptionVolatility currency="EUR">SwaptionVolatility/EUR/EUR_SW_N</SwaptionVolatility>
              </SwaptionVolatilities>
            </TodaysMarket>
            """
        ),
    )

    root = ET.parse(case_input / "ore_classic.xml").getroot()
    setup = root.find("Setup")
    assert setup is not None
    for node in setup.findall("Parameter"):
        name = node.attrib.get("name", "")
        if name == "inputPath":
            node.text = str(run_root / "Input")
        elif name == "outputPath":
            node.text = str(run_root / "Output")
        elif name == "portfolioFile":
            node.text = str(case_input / "portfolio.xml")
        elif name == "pricingEnginesFile":
            node.text = str(case_input / "pricingengine.xml")
        elif name == "marketDataFile":
            node.text = str(case_input / "market_20160205_flat_fixed_fxfwd.txt")
        elif name == "fixingDataFile":
            node.text = str(REPO_ROOT / "Examples" / "Input" / "fixings_20160205.txt")
        elif name == "curveConfigFile":
            node.text = str(REPO_ROOT / "Examples" / "Input" / "curveconfig.xml")
        elif name == "conventionsFile":
            node.text = str(REPO_ROOT / "Examples" / "Input" / "conventions.xml")
        elif name == "marketConfigFile":
            node.text = str(run_root / "Input" / "todaysmarket.xml")

    analytics = root.find("Analytics")
    assert analytics is not None
    for child in list(analytics):
        analytics.remove(child)

    for analytic_type, params in (
        ("npv", {"active": "Y", "baseCurrency": "EUR", "outputFileName": "npv.csv"}),
        (
            "sensitivity",
            {
                "active": "Y",
                "marketConfigFile": str(run_root / "Input" / "simulation_sensitivity.xml"),
                "sensitivityConfigFile": str(run_root / "Input" / "sensitivity.xml"),
                "pricingEnginesFile": str(case_input / "pricingengine.xml"),
                "scenarioOutputFile": "scenario.csv",
                "sensitivityOutputFile": "sensitivity.csv",
                "outputSensitivityThreshold": "0.0",
                "recalibrateModels": "Y",
            },
        ),
    ):
        analytic = ET.SubElement(analytics, "Analytic", {"type": analytic_type})
        for key, value in params.items():
            param = ET.SubElement(analytic, "Parameter", {"name": key})
            param.text = value

    ET.ElementTree(root).write(run_root / "Input" / "ore.xml", encoding="utf-8", xml_declaration=True)
    return run_root


def _run_ore(run_root: Path, ore_bin: Path) -> None:
    cp = subprocess.run(
        [str(ore_bin), str(run_root / "Input" / "ore.xml")],
        cwd=str(run_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"ORE run failed with return code {cp.returncode}\n{cp.stdout}\n{cp.stderr}")


def _read_trade_npv(path: Path, trade_id: str) -> float:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "#TradeId" if "#TradeId" in (reader.fieldnames or []) else "TradeId"
        for row in reader:
            if row.get(tid_key, "").strip() == trade_id:
                return float(row["NPV"])
    raise ValueError(f"Trade {trade_id} not found in {path}")


def _write_direct_npv_case(case_input: Path, run_root: Path, market_file: Path, output_dir: Path) -> Path:
    root = ET.parse(case_input / "ore_classic.xml").getroot()
    setup = root.find("Setup")
    assert setup is not None
    examples_input = require_engine_repo_root() / "Examples" / "Input"
    for node in setup.findall("Parameter"):
        name = node.attrib.get("name", "")
        if name == "inputPath":
            node.text = str(run_root / "Input")
        elif name == "outputPath":
            node.text = str(output_dir)
        elif name == "portfolioFile":
            node.text = str(case_input / "portfolio.xml")
        elif name == "pricingEnginesFile":
            node.text = str(case_input / "pricingengine.xml")
        elif name == "marketDataFile":
            node.text = str(market_file)
        elif name == "fixingDataFile":
            node.text = str(examples_input / "fixings_20160205.txt")
        elif name == "curveConfigFile":
            node.text = str(examples_input / "curveconfig.xml")
        elif name == "conventionsFile":
            node.text = str(examples_input / "conventions.xml")
        elif name == "marketConfigFile":
            node.text = str(examples_input / "todaysmarket.xml")

    analytics = root.find("Analytics")
    assert analytics is not None
    for child in list(analytics):
        analytics.remove(child)
    analytic = ET.SubElement(analytics, "Analytic", {"type": "npv"})
    for key, value in (("active", "Y"), ("baseCurrency", "EUR"), ("outputFileName", "npv.csv")):
        param = ET.SubElement(analytic, "Parameter", {"name": key})
        param.text = value

    xml_path = run_root / "Input" / f"ore_direct_{output_dir.name}.xml"
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


def _direct_ore_quote_bump_change(
    case_input: Path,
    run_root: Path,
    ore_bin: Path,
    normalized_factor: str,
    trade_id: str,
    shift_size: float,
) -> float:
    quote_key = _ore_market_quote_for_factor(normalized_factor)
    market_lines = (case_input / "market_20160205_flat_fixed_fxfwd.txt").read_text(encoding="utf-8").splitlines()
    bumped_lines: list[str] = []
    matched = False
    for line in market_lines:
        if line.strip().startswith(f"20160205 {quote_key} "):
            parts = line.split()
            parts[-1] = f"{float(parts[-1]) + shift_size:.10f}"
            line = " ".join(parts)
            matched = True
        bumped_lines.append(line)
    if not matched:
        raise KeyError(f"Quote '{quote_key}' not found in {case_input / 'market_20160205_flat_fixed_fxfwd.txt'}")

    market_file = run_root / "Input" / f"market_direct_{normalized_factor.replace(':', '_')}.txt"
    market_file.write_text("\n".join(bumped_lines) + "\n", encoding="utf-8")
    output_dir = run_root / "Output" / f"direct_{normalized_factor.replace(':', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _write_direct_npv_case(case_input, run_root, market_file, output_dir)
    cp = subprocess.run([str(ore_bin), str(xml_path)], cwd=str(run_root), capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"ORE direct quote bump run failed with return code {cp.returncode}\n{cp.stdout}\n{cp.stderr}")
    bumped_npv = _read_trade_npv(output_dir / "npv.csv", trade_id)
    base_npv = _read_trade_npv(case_input.parent / "Output" / "classic" / "npv.csv", trade_id)
    return float(bumped_npv - base_npv)


def _read_ore_sensitivity_rows(path: Path, trade_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "#TradeId" if "#TradeId" in (reader.fieldnames or []) else "TradeId"
        for row in reader:
            if row.get(tid_key, "").strip() != trade_id:
                continue
            rows.append(
                {
                    "factor": row["Factor_1"].strip(),
                    "base_npv": float(row["Base NPV"]),
                    "delta_bump_change": float(row["Delta"]),
                }
            )
    return rows


def _python_node_bump_rows(case_input: Path, method: str, num_paths: int, seed: int, shift_size: float) -> list[dict[str, Any]]:
    _bootstrap()
    from native_xva_interface.bermudan import _build_bermudan_context, _curve_node_deltas
    from native_xva_interface.loader import XVALoader

    snapshot = XVALoader.from_files(str(case_input), ore_file="ore_classic.xml")
    ctx = _build_bermudan_context(
        snapshot,
        trade_id="BermSwp",
        method=method,
        num_paths=num_paths,
        seed=seed,
        basis_degree=2,
        curve_mode="market_fit",
    )
    curve_bundle = ctx["curve_bundle"]
    disc_fit = curve_bundle["disc_fit"]
    fwd_fit = curve_bundle["fwd_fit"]
    base_disc = curve_bundle["p0_disc"]
    base_fwd = curve_bundle["p0_fwd"]

    rows: list[dict[str, Any]] = []
    if disc_fit:
        node_times = np.asarray(disc_fit["times"], dtype=float)
        node_times = node_times[node_times > 1.0e-12]
        deltas = _curve_node_deltas(ctx, base_disc, base_fwd, node_times, curve_side="discount", shift_size=shift_size)
        for t, d in zip(node_times, deltas):
            rows.append(
                {
                    "normalized_factor": f"zero:EUR:{_years_to_tenor(t)}",
                    "python_node_derivative": float(d),
                    "python_node_bump_change": float(d * shift_size),
                }
            )
    if fwd_fit:
        node_times = np.asarray(fwd_fit["times"], dtype=float)
        node_times = node_times[node_times > 1.0e-12]
        deltas = _curve_node_deltas(ctx, base_disc, base_fwd, node_times, curve_side="forward", shift_size=shift_size)
        for t, d in zip(node_times, deltas):
            rows.append(
                {
                    "normalized_factor": f"fwd:EUR:6M:{_years_to_tenor(t)}",
                    "python_node_derivative": float(d),
                    "python_node_bump_change": float(d * shift_size),
                }
            )
    return rows


def _years_to_tenor(years: float) -> str:
    y = float(years)
    if abs(y - round(y)) < 1.0e-8:
        return f"{int(round(y))}Y"
    months = int(round(y * 12.0))
    if months % 12 == 0:
        return f"{months // 12}Y"
    return f"{months}M"


def _normalize_ore_factor(factor: str) -> str:
    parts = factor.split("/")
    if parts[0] == "DiscountCurve":
        return f"zero:{parts[1].upper()}:{parts[-1].upper()}"
    if parts[0] == "IndexCurve":
        ccy = parts[1].split("-")[0].upper()
        tenor = parts[1].split("-")[-1].upper()
        return f"fwd:{ccy}:{tenor}:{parts[-1].upper()}"
    return factor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ORE Bermudan PV sensitivity comparison against the Python Bermudan pricer.")
    parser.add_argument("--case-name", default="berm_200bp")
    parser.add_argument("--method", default="backward", choices=("backward", "lsmc"))
    parser.add_argument("--num-paths", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shift-size", type=float, default=1.0e-4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=local_parity_artifacts_root() / "bermudan_sensitivity_compare",
    )
    args = parser.parse_args()

    _bootstrap()
    from native_xva_interface import price_bermudan_from_ore_case, price_bermudan_with_sensis_from_ore_case

    case_input = _case_input_dir(args.case_name)
    ore_bin = _locate_ore_exe()
    ore_run_root = _prepare_ore_sensitivity_run(case_input, args.output_root, args.shift_size)
    _run_ore(ore_run_root, ore_bin)

    ore_price = _read_trade_npv(ore_run_root / "Output" / "npv.csv", "BermSwp")
    ore_rows = _read_ore_sensitivity_rows(ore_run_root / "Output" / "sensitivity.csv", "BermSwp")
    ore_by_factor = {_normalize_ore_factor(row["factor"]): row for row in ore_rows}

    py_price = price_bermudan_from_ore_case(
        case_input,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method=args.method,
        num_paths=args.num_paths,
        seed=args.seed,
        curve_mode="market_fit",
    )
    py_quote_full = price_bermudan_with_sensis_from_ore_case(
        case_input,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method=args.method,
        num_paths=args.num_paths,
        seed=args.seed,
        factors=["IR_SWAP/RATE/EUR/2D/1D/10Y", "IR_SWAP/RATE/EUR/2D/6M/10Y"],
        shift_size=args.shift_size,
        sensitivity_mode="full_reprice",
    )
    py_quote_fast = price_bermudan_with_sensis_from_ore_case(
        case_input,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method=args.method,
        num_paths=args.num_paths,
        seed=args.seed,
        factors=["IR_SWAP/RATE/EUR/2D/1D/10Y", "IR_SWAP/RATE/EUR/2D/6M/10Y"],
        shift_size=args.shift_size,
        sensitivity_mode="fast_curve_jacobian",
    )
    py_node_rows = _python_node_bump_rows(case_input, args.method, args.num_paths, args.seed, args.shift_size)
    py_node_by_factor = {row["normalized_factor"]: row for row in py_node_rows}
    py_quote_full_by_factor = {
        "zero:EUR:10Y": next(s for s in py_quote_full.sensitivities if s.factor == "IR_SWAP/RATE/EUR/2D/1D/10Y"),
        "fwd:EUR:6M:10Y": next(s for s in py_quote_full.sensitivities if s.factor == "IR_SWAP/RATE/EUR/2D/6M/10Y"),
    }
    py_quote_fast_by_factor = {
        "zero:EUR:10Y": next(s for s in py_quote_fast.sensitivities if s.factor == "IR_SWAP/RATE/EUR/2D/1D/10Y"),
        "fwd:EUR:6M:10Y": next(s for s in py_quote_fast.sensitivities if s.factor == "IR_SWAP/RATE/EUR/2D/6M/10Y"),
    }

    rows: list[dict[str, Any]] = []
    for normalized_factor in ("zero:EUR:10Y", "fwd:EUR:6M:10Y"):
        ore_row = ore_by_factor.get(normalized_factor, {})
        node_row = py_node_by_factor.get(normalized_factor, {})
        quote_full = py_quote_full_by_factor.get(normalized_factor)
        quote_fast = py_quote_fast_by_factor.get(normalized_factor)
        ore_direct_quote_bump_change = _direct_ore_quote_bump_change(
            case_input=case_input,
            run_root=ore_run_root,
            ore_bin=ore_bin,
            normalized_factor=normalized_factor,
            trade_id="BermSwp",
            shift_size=args.shift_size,
        )
        rows.append(
            {
                "normalized_factor": normalized_factor,
                "ore_factor": ore_row.get("factor"),
                "ore_bump_change": ore_row.get("delta_bump_change"),
                "ore_direct_quote_bump_change": ore_direct_quote_bump_change,
                "python_node_bump_change": node_row.get("python_node_bump_change"),
                "python_node_minus_ore": None
                if ore_row.get("delta_bump_change") is None or node_row.get("python_node_bump_change") is None
                else float(node_row["python_node_bump_change"] - ore_row["delta_bump_change"]),
                "python_quote_full_minus_ore_direct": None
                if quote_full is None
                else float(quote_full.delta * args.shift_size - ore_direct_quote_bump_change),
                "python_quote_full_bump_change": None if quote_full is None else float(quote_full.delta * args.shift_size),
                "python_quote_fast_bump_change": None if quote_fast is None else float(quote_fast.delta * args.shift_size),
            }
        )

    payload = {
        "case_name": args.case_name,
        "trade_id": "BermSwp",
        "method": args.method,
        "shift_size": args.shift_size,
        "python_price": py_price.price,
        "ore_price": ore_price,
        "price_diff": py_price.price - ore_price,
        "ore_run_root": str(ore_run_root),
        "rows": rows,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / f"{args.case_name}_{args.method}.json"
    csv_path = args.output_root / f"{args.case_name}_{args.method}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"case={args.case_name}")
    print(f"python_price={py_price.price:.6f}")
    print(f"ore_price={ore_price:.6f}")
    print(f"price_diff={py_price.price - ore_price:.6f}")
    for row in rows:
        print(
            f"{row['normalized_factor']}: ore={row['ore_bump_change']:.6f} "
            f"ore_direct={row['ore_direct_quote_bump_change']:.6f} "
            f"py_node={row['python_node_bump_change']:.6f} "
            f"py_quote_full={row['python_quote_full_bump_change']:.6f} "
            f"py_quote_fast={row['python_quote_fast_bump_change']:.6f}"
        )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
