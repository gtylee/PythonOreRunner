#!/usr/bin/env python3
"""Compare Python and ORE Bermudan pricing methods on a few native ORE cases."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional
import xml.etree.ElementTree as ET

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from native_xva_interface import price_bermudan_from_ore_case

REPO_ROOT = Path(__file__).resolve().parents[4]
ORE_BIN_DEFAULT = REPO_ROOT / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore"
AMC_INPUT_DIR = REPO_ROOT / "Examples" / "AmericanMonteCarlo" / "Input"
AMC_EXPECTED_OUTPUT_DIR = REPO_ROOT / "Examples" / "AmericanMonteCarlo" / "ExpectedOutput" / "amc"
EXAMPLES_INPUT_DIR = REPO_ROOT / "Examples" / "Input"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "bermudan_method_compare"


@dataclass(frozen=True)
class BermudanVariant:
    name: str
    fixed_rate: float


@dataclass(frozen=True)
class ComparisonRow:
    case_name: str
    trade_id: str
    fixed_rate: float
    py_lsmc: float
    py_backward: float
    py_lsmc_calibrated: Optional[float]
    py_backward_calibrated: Optional[float]
    ore_classic: float
    ore_amc: Optional[float]
    ore_amc_source: str
    py_lsmc_minus_ore_classic: float
    py_backward_minus_ore_classic: float
    py_lsmc_calibrated_minus_ore_classic: Optional[float]
    py_backward_calibrated_minus_ore_classic: Optional[float]
    ore_amc_minus_ore_classic: Optional[float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ore-bin", type=Path, default=ORE_BIN_DEFAULT)
    parser.add_argument("--input-template", type=Path, default=AMC_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trade-id", default="BermSwp")
    parser.add_argument("--py-paths", type=int, default=4096)
    parser.add_argument("--py-seed", type=int, default=42)
    parser.add_argument("--basis-degree", type=int, default=2)
    parser.add_argument("--run-calibration", action="store_true")
    return parser.parse_args()


def _variants() -> tuple[BermudanVariant, ...]:
    return (
        BermudanVariant("berm_100bp", 0.0100),
        BermudanVariant("berm_200bp", 0.0200),
        BermudanVariant("berm_300bp", 0.0300),
    )


def _run() -> int:
    args = _parse_args()
    if not args.ore_bin.exists():
        raise FileNotFoundError(f"ORE binary not found: {args.ore_bin}")
    if not args.input_template.exists():
        raise FileNotFoundError(f"AMC example input not found: {args.input_template}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[ComparisonRow] = []

    for variant in _variants():
        case_root = args.output_root / variant.name
        input_dir = case_root / "Input"
        _prepare_case(args.input_template, input_dir, variant.fixed_rate, args.trade_id)
        _run_ore_case(args.ore_bin, case_root, input_dir / "ore_classic.xml")
        if args.run_calibration:
            _run_ore_case(args.ore_bin, case_root, _ensure_calibration_config(input_dir))

        py_lsmc = price_bermudan_from_ore_case(
            input_dir,
            ore_file="ore_classic.xml",
            trade_id=args.trade_id,
            method="lsmc",
            num_paths=args.py_paths,
            seed=args.py_seed,
            basis_degree=args.basis_degree,
            curve_mode="market_fit",
        ).price
        py_backward = price_bermudan_from_ore_case(
            input_dir,
            ore_file="ore_classic.xml",
            trade_id=args.trade_id,
            method="backward",
            num_paths=args.py_paths,
            seed=args.py_seed,
            basis_degree=args.basis_degree,
            curve_mode="market_fit",
        ).price
        py_lsmc_calibrated = None
        py_backward_calibrated = None
        if args.run_calibration:
            py_lsmc_calibrated = price_bermudan_from_ore_case(
                input_dir,
                ore_file="ore_classic_calibration.xml",
                trade_id=args.trade_id,
                method="lsmc",
                num_paths=args.py_paths,
                seed=args.py_seed,
                basis_degree=args.basis_degree,
                curve_mode="market_fit",
            ).price
            py_backward_calibrated = price_bermudan_from_ore_case(
                input_dir,
                ore_file="ore_classic_calibration.xml",
                trade_id=args.trade_id,
                method="backward",
                num_paths=args.py_paths,
                seed=args.py_seed,
                basis_degree=args.basis_degree,
                curve_mode="market_fit",
            ).price
        ore_classic = _load_trade_npv(case_root / "Output" / "classic" / "npv.csv", args.trade_id)
        ore_amc, ore_amc_source = _resolve_ore_amc_price(args.ore_bin, case_root, input_dir / "ore_amc.xml", args.trade_id, variant.fixed_rate)
        rows.append(
            ComparisonRow(
                case_name=variant.name,
                trade_id=args.trade_id,
                fixed_rate=variant.fixed_rate,
                py_lsmc=py_lsmc,
                py_backward=py_backward,
                py_lsmc_calibrated=py_lsmc_calibrated,
                py_backward_calibrated=py_backward_calibrated,
                ore_classic=ore_classic,
                ore_amc=ore_amc,
                ore_amc_source=ore_amc_source,
                py_lsmc_minus_ore_classic=py_lsmc - ore_classic,
                py_backward_minus_ore_classic=py_backward - ore_classic,
                py_lsmc_calibrated_minus_ore_classic=None if py_lsmc_calibrated is None else py_lsmc_calibrated - ore_classic,
                py_backward_calibrated_minus_ore_classic=None if py_backward_calibrated is None else py_backward_calibrated - ore_classic,
                ore_amc_minus_ore_classic=None if ore_amc is None else ore_amc - ore_classic,
            )
        )

    _write_outputs(args.output_root, rows)
    _print_summary(rows)
    return 0


def _prepare_case(template_input_dir: Path, input_dir: Path, fixed_rate: float, trade_id: str) -> None:
    shutil.copytree(template_input_dir, input_dir, dirs_exist_ok=True)
    portfolio_path = input_dir / "portfolio.xml"
    tree = ET.parse(portfolio_path)
    root = tree.getroot()
    for child in list(root.findall("./Trade")):
        if child.get("id") != trade_id:
            root.remove(child)
    trade = root.find(f"./Trade[@id='{trade_id}']")
    if trade is None:
        raise ValueError(f"Trade '{trade_id}' not found in {portfolio_path}")
    rate_node = trade.find("./SwaptionData/LegData[LegType='Fixed']/FixedLegData/Rates/Rate")
    if rate_node is None:
        raise ValueError(f"Fixed rate node not found for trade '{trade_id}' in {portfolio_path}")
    rate_node.text = f"{fixed_rate:.6f}"
    tree.write(portfolio_path, encoding="utf-8", xml_declaration=True)
    _rewrite_ore_config_paths(input_dir / "ore_classic.xml")
    _rewrite_ore_config_paths(input_dir / "ore_amc.xml")


def _rewrite_ore_config_paths(ore_xml_path: Path) -> None:
    tree = ET.parse(ore_xml_path)
    root = tree.getroot()
    replacements = {
        "fixingDataFile": EXAMPLES_INPUT_DIR / "fixings_20160205.txt",
        "curveConfigFile": EXAMPLES_INPUT_DIR / "curveconfig.xml",
        "conventionsFile": EXAMPLES_INPUT_DIR / "conventions.xml",
        "marketConfigFile": EXAMPLES_INPUT_DIR / "todaysmarket.xml",
    }
    for param_name, path in replacements.items():
        node = root.find(f"./Setup/Parameter[@name='{param_name}']")
        if node is None:
            raise ValueError(f"Parameter '{param_name}' not found in {ore_xml_path}")
        node.text = str(path)
    tree.write(ore_xml_path, encoding="utf-8", xml_declaration=True)


def _ensure_calibration_config(input_dir: Path) -> Path:
    ore_xml_path = input_dir / "ore_classic_calibration.xml"
    if ore_xml_path.exists():
        return ore_xml_path

    source_path = input_dir / "ore_classic.xml"
    tree = ET.parse(source_path)
    root = tree.getroot()
    analytics = root.find("./Analytics")
    if analytics is None:
        raise ValueError(f"Analytics node not found in {source_path}")

    calibration = analytics.find("./Analytic[@type='calibration']")
    if calibration is None:
        calibration = ET.SubElement(analytics, "Analytic", {"type": "calibration"})
        active = ET.SubElement(calibration, "Parameter", {"name": "active"})
        active.text = "Y"
        config_file = ET.SubElement(calibration, "Parameter", {"name": "configFile"})
        config_file.text = "simulation_classic.xml"
        output_file = ET.SubElement(calibration, "Parameter", {"name": "outputFile"})
        output_file.text = "calibration.csv"
    else:
        active = calibration.find("./Parameter[@name='active']")
        if active is None:
            active = ET.SubElement(calibration, "Parameter", {"name": "active"})
        active.text = "Y"

    tree.write(ore_xml_path, encoding="utf-8", xml_declaration=True)
    return ore_xml_path


def _run_ore_case(ore_bin: Path, case_root: Path, ore_input: Path) -> None:
    subprocess.run(
        [str(ore_bin), str(ore_input.relative_to(case_root))],
        cwd=case_root,
        check=True,
    )


def _resolve_ore_amc_price(
    ore_bin: Path,
    case_root: Path,
    ore_input: Path,
    trade_id: str,
    fixed_rate: float,
) -> tuple[Optional[float], str]:
    try:
        _run_ore_case(ore_bin, case_root, ore_input)
        return _load_trade_npv(case_root / "Output" / "amc" / "npv.csv", trade_id), "live_run"
    except subprocess.CalledProcessError as exc:
        if abs(fixed_rate - 0.02) <= 1.0e-12 and (AMC_EXPECTED_OUTPUT_DIR / "npv.csv").exists():
            return _load_trade_npv(AMC_EXPECTED_OUTPUT_DIR / "npv.csv", trade_id), "expected_output_fallback"
        return None, f"unavailable(returncode={exc.returncode})"


def _load_trade_npv(npv_csv: Path, trade_id: str) -> float:
    with open(npv_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get(tid_key, "").strip() == trade_id:
                return float(row["NPV"])
    raise ValueError(f"Trade '{trade_id}' not found in {npv_csv}")


def _write_outputs(output_root: Path, rows: list[ComparisonRow]) -> None:
    csv_path = output_root / "comparison.csv"
    json_path = output_root / "comparison.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")


def _print_summary(rows: list[ComparisonRow]) -> None:
    print(
        "case_name,fixed_rate,py_lsmc,py_backward,py_lsmc_calibrated,py_backward_calibrated,"
        "ore_classic,ore_amc,ore_amc_source"
    )
    for row in rows:
        ore_amc = "NA" if row.ore_amc is None else f"{row.ore_amc:.6f}"
        py_lsmc_calibrated = "NA" if row.py_lsmc_calibrated is None else f"{row.py_lsmc_calibrated:.6f}"
        py_backward_calibrated = "NA" if row.py_backward_calibrated is None else f"{row.py_backward_calibrated:.6f}"
        print(
            f"{row.case_name},{row.fixed_rate:.4f},{row.py_lsmc:.6f},{row.py_backward:.6f},"
            f"{py_lsmc_calibrated},{py_backward_calibrated},"
            f"{row.ore_classic:.6f},{ore_amc},{row.ore_amc_source}"
        )


if __name__ == "__main__":
    raise SystemExit(_run())
