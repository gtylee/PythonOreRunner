from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from pythonore.repo_paths import default_ore_bin, local_examples_root, require_engine_repo_root


@dataclass(frozen=True)
class Hw2FCasePaths:
    case_dir: Path
    input_dir: Path
    output_dir: Path
    ore_xml: Path
    simulation_xml: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_shared_input_dir() -> Path:
    examples_root = local_examples_root()
    if examples_root is not None:
        shared = examples_root / "Input"
        if shared.exists():
            return shared
    return require_engine_repo_root() / "Examples" / "Input"


def _default_exposure_input_dir() -> Path:
    return require_engine_repo_root() / "Examples" / "Exposure" / "Input"


def _default_ore_exe() -> Path:
    candidates: list[Path] = []
    env_value = os.environ.get("ORE_EXE")
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(default_ore_bin())
    repo_root = require_engine_repo_root()
    candidates.extend(
        [
            repo_root / "build" / "App" / "ore",
            repo_root / "build" / "ore" / "App" / "ore",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ORE executable not found. Set ORE_EXE or build App/ore.")


def _normalize_case_paths(input_dir: str | Path) -> Hw2FCasePaths:
    input_dir = Path(input_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    case_dir = input_dir.parent
    output_dir = case_dir / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Hw2FCasePaths(
        case_dir=case_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        ore_xml=input_dir / "ore.xml",
        simulation_xml=input_dir / "simulation.xml",
    )


def _resolve_base_input_dir(base_case: str | Path | None) -> Path:
    if base_case is None:
        return _default_exposure_input_dir()
    base_path = Path(base_case).resolve()
    if base_path.is_file():
        if base_path.name == "ore.xml":
            return base_path.parent
        raise FileNotFoundError(f"Unsupported base case file: {base_path}")
    if (base_path / "Input").exists():
        return (base_path / "Input").resolve()
    if (base_path / "ore.xml").exists():
        return base_path
    raise FileNotFoundError(f"Could not resolve base case input directory from {base_case}")


def _copy_default_case_inputs(target_input_dir: Path, base_input_dir: Path) -> None:
    target_input_dir.mkdir(parents=True, exist_ok=True)

    shared_input_dir = _default_shared_input_dir()
    engine_shared_input_dir = require_engine_repo_root() / "Examples" / "Input"
    shared_files = {
        "conventions.xml": ("conventions.xml", shared_input_dir / "conventions.xml"),
        "curveconfig.xml": ("curveconfig.xml", shared_input_dir / "curveconfig.xml"),
        "todaysmarket.xml": ("todaysmarket.xml", shared_input_dir / "todaysmarket.xml"),
        "pricingengine.xml": ("pricingengine.xml", shared_input_dir / "pricingengine.xml"),
        "calendaradjustment.xml": ("calendaradjustment.xml", shared_input_dir / "calendaradjustment.xml"),
        "currencies.xml": ("currencies.xml", shared_input_dir / "currencies.xml"),
        "market.txt": ("market.txt", shared_input_dir / "market_20160205_flat.txt"),
        "fixings.txt": ("fixings.txt", shared_input_dir / "fixings_20160205.txt"),
    }
    for target_name, (base_name, fallback_source) in shared_files.items():
        source = base_input_dir / base_name
        if not source.exists():
            source = fallback_source
        if not source.exists():
            source = engine_shared_input_dir / fallback_source.name
        shutil.copy2(source, target_input_dir / target_name)

    portfolio_source = base_input_dir / "portfolio_swap.xml"
    if not portfolio_source.exists():
        portfolio_source = base_input_dir / "portfolio.xml"
    if not portfolio_source.exists():
        raise FileNotFoundError(f"No portfolio file found in {base_input_dir}")
    shutil.copy2(portfolio_source, target_input_dir / "portfolio.xml")

    netting_source = base_input_dir / "netting.xml"
    if not netting_source.exists():
        netting_source = _default_exposure_input_dir() / "netting.xml"
    shutil.copy2(netting_source, target_input_dir / "netting.xml")


def _validate_sigma_kappa_inputs(
    sigma: Iterable[Iterable[Iterable[float]]],
    kappa: Iterable[Iterable[float]],
    times: Iterable[float],
) -> tuple[list[list[list[float]]], list[list[float]], list[float], str]:
    sigma_list = [[list(map(float, row)) for row in matrix] for matrix in sigma]
    kappa_list = [list(map(float, bucket)) for bucket in kappa]
    times_list = [float(t) for t in times]

    if not sigma_list:
        raise ValueError("sigma must contain at least one 2x2 matrix")
    if not kappa_list:
        raise ValueError("kappa must contain at least one 2-element vector")
    if len(sigma_list) != len(kappa_list):
        raise ValueError("sigma and kappa must have the same number of buckets")

    for idx, matrix in enumerate(sigma_list):
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError(f"sigma bucket {idx} must be a 2x2 matrix")

    for idx, bucket in enumerate(kappa_list):
        if len(bucket) != 2:
            raise ValueError(f"kappa bucket {idx} must be a 2-element vector")

    expected_times = max(len(sigma_list) - 1, 0)
    if len(times_list) != expected_times:
        raise ValueError(
            f"times length must be {expected_times} for {len(sigma_list)} sigma/kappa bucket(s); got {len(times_list)}"
        )
    if any(b <= a for a, b in zip(times_list, times_list[1:])):
        raise ValueError("times must be strictly increasing")

    param_type = "Constant" if not times_list else "Piecewise"
    return sigma_list, kappa_list, times_list, param_type


def _format_number(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return format(float(value), ".15g")


def _comma_join(values: Iterable[float]) -> str:
    return ",".join(_format_number(v) for v in values)


def _build_simulation_xml(
    sigma: list[list[list[float]]],
    kappa: list[list[float]],
    times: list[float],
    param_type: str,
    *,
    measure: str,
    samples: int,
    seed: int,
    grid: str,
) -> ET.ElementTree:
    root = ET.Element("Simulation")

    params = ET.SubElement(root, "Parameters")
    for tag, text in (
        ("Discretization", "Exact"),
        ("Grid", grid),
        ("Calendar", "EUR"),
        ("Sequence", "SobolBrownianBridge"),
        ("Scenario", "Simple"),
        ("Seed", str(seed)),
        ("Samples", str(samples)),
        ("Ordering", "Steps"),
        ("DirectionIntegers", "JoeKuoD7"),
    ):
        ET.SubElement(params, tag).text = text

    cam = ET.SubElement(root, "CrossAssetModel")
    ET.SubElement(cam, "DomesticCcy").text = "EUR"
    currencies = ET.SubElement(cam, "Currencies")
    ET.SubElement(currencies, "Currency").text = "EUR"
    ET.SubElement(cam, "Discretization").text = "Euler"
    ET.SubElement(cam, "Measure").text = measure
    ET.SubElement(cam, "BootstrapTolerance").text = "0.0001"

    ir_models = ET.SubElement(cam, "InterestRateModels")
    hw_model = ET.SubElement(ir_models, "HWModel", {"key": "default"})
    ET.SubElement(hw_model, "CalibrationType").text = "None"

    reversion = ET.SubElement(hw_model, "Reversion")
    ET.SubElement(reversion, "Calibrate").text = "N"
    ET.SubElement(reversion, "ParamType").text = param_type
    ET.SubElement(reversion, "TimeGrid").text = _comma_join(times)
    reversion_initial = ET.SubElement(reversion, "InitialValue")
    for bucket in kappa:
        ET.SubElement(reversion_initial, "Kappa").text = _comma_join(bucket)

    volatility = ET.SubElement(hw_model, "Volatility")
    ET.SubElement(volatility, "Calibrate").text = "N"
    ET.SubElement(volatility, "ParamType").text = param_type
    ET.SubElement(volatility, "TimeGrid").text = _comma_join(times)
    volatility_initial = ET.SubElement(volatility, "InitialValue")
    for matrix in sigma:
        sigma_node = ET.SubElement(volatility_initial, "Sigma")
        for row in matrix:
            ET.SubElement(sigma_node, "Row").text = _comma_join(row)

    ET.SubElement(cam, "ForeignExchangeModels")
    ET.SubElement(cam, "InstantaneousCorrelations")

    market = ET.SubElement(root, "Market")
    ET.SubElement(market, "BaseCurrency").text = "EUR"
    market_ccys = ET.SubElement(market, "Currencies")
    ET.SubElement(market_ccys, "Currency").text = "EUR"

    yield_curves = ET.SubElement(market, "YieldCurves")
    config = ET.SubElement(yield_curves, "Configuration")
    ET.SubElement(config, "Tenors").text = "3M,6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y"
    ET.SubElement(config, "Interpolation").text = "LogLinear"
    ET.SubElement(config, "Extrapolation").text = "Y"

    indices = ET.SubElement(market, "Indices")
    for index_name in ("EUR-EURIBOR-6M", "EUR-EURIBOR-3M", "EUR-EONIA"):
        ET.SubElement(indices, "Index").text = index_name

    swap_indices = ET.SubElement(market, "SwapIndices")
    for swap_index_name in ("EUR-CMS-1Y", "EUR-CMS-2Y", "EUR-CMS-10Y", "EUR-CMS-30Y"):
        swap_index = ET.SubElement(swap_indices, "SwapIndex")
        ET.SubElement(swap_index, "Name").text = swap_index_name
        ET.SubElement(swap_index, "DiscountingIndex").text = "EUR-EONIA"

    agg_ccys = ET.SubElement(market, "AggregationScenarioDataCurrencies")
    ET.SubElement(agg_ccys, "Currency").text = "EUR"
    agg_indices = ET.SubElement(market, "AggregationScenarioDataIndices")
    for index_name in ("EUR-EURIBOR-3M", "EUR-EONIA"):
        ET.SubElement(agg_indices, "Index").text = index_name

    return ET.ElementTree(root)


def _build_ore_xml(case_paths: Hw2FCasePaths) -> ET.ElementTree:
    root = ET.Element("ORE")

    setup = ET.SubElement(root, "Setup")
    setup_params = (
        ("asofDate", "2016-02-05"),
        ("inputPath", str(case_paths.input_dir)),
        ("outputPath", str(case_paths.output_dir)),
        ("logFile", "log.txt"),
        ("logMask", "31"),
        ("marketDataFile", str(case_paths.input_dir / "market.txt")),
        ("fixingDataFile", str(case_paths.input_dir / "fixings.txt")),
        ("implyTodaysFixings", "Y"),
        ("curveConfigFile", str(case_paths.input_dir / "curveconfig.xml")),
        ("conventionsFile", str(case_paths.input_dir / "conventions.xml")),
        ("marketConfigFile", str(case_paths.input_dir / "todaysmarket.xml")),
        ("pricingEnginesFile", str(case_paths.input_dir / "pricingengine.xml")),
        ("portfolioFile", str(case_paths.input_dir / "portfolio.xml")),
        ("observationModel", "None"),
        ("continueOnError", "false"),
        ("calendarAdjustment", str(case_paths.input_dir / "calendaradjustment.xml")),
        ("currencyConfiguration", str(case_paths.input_dir / "currencies.xml")),
    )
    for name, value in setup_params:
        ET.SubElement(setup, "Parameter", {"name": name}).text = value

    markets = ET.SubElement(root, "Markets")
    for name, value in (
        ("lgmcalibration", "collateral_inccy"),
        ("fxcalibration", "xois_eur"),
        ("pricing", "xois_eur"),
        ("simulation", "xois_eur"),
    ):
        ET.SubElement(markets, "Parameter", {"name": name}).text = value

    analytics = ET.SubElement(root, "Analytics")

    analytic = ET.SubElement(analytics, "Analytic", {"type": "npv"})
    for name, value in (
        ("active", "Y"),
        ("baseCurrency", "EUR"),
        ("outputFileName", "npv.csv"),
        ("additionalResults", "Y"),
        ("additionalResultsReportPrecision", "12"),
    ):
        ET.SubElement(analytic, "Parameter", {"name": name}).text = value

    analytic = ET.SubElement(analytics, "Analytic", {"type": "cashflow"})
    for name, value in (("active", "Y"), ("outputFileName", "flows.csv")):
        ET.SubElement(analytic, "Parameter", {"name": name}).text = value

    analytic = ET.SubElement(analytics, "Analytic", {"type": "curves"})
    for name, value in (
        ("active", "Y"),
        ("configuration", "default"),
        ("grid", "240,1M"),
        ("outputFileName", "curves.csv"),
        ("outputTodaysMarketCalibration", "Y"),
    ):
        ET.SubElement(analytic, "Parameter", {"name": name}).text = value

    analytic = ET.SubElement(analytics, "Analytic", {"type": "simulation"})
    for name, value in (
        ("active", "Y"),
        ("simulationConfigFile", str(case_paths.simulation_xml)),
        ("pricingEnginesFile", str(case_paths.input_dir / "pricingengine.xml")),
        ("baseCurrency", "EUR"),
        ("observationModel", "Disable"),
        ("cubeFile", "cube.csv.gz"),
        ("aggregationScenarioDataFileName", "scenariodata.csv.gz"),
    ):
        ET.SubElement(analytic, "Parameter", {"name": name}).text = value

    analytic = ET.SubElement(analytics, "Analytic", {"type": "xva"})
    for name, value in (
        ("active", "Y"),
        ("useXvaRunner", "N"),
        ("csaFile", str(case_paths.input_dir / "netting.xml")),
        ("cubeFile", "cube.csv.gz"),
        ("scenarioFile", "scenariodata.csv.gz"),
        ("baseCurrency", "EUR"),
        ("exposureProfiles", "Y"),
        ("exposureProfilesByTrade", "Y"),
        ("cva", "Y"),
        ("dva", "N"),
        ("fva", "N"),
        ("rawCubeOutputFile", "rawcube.csv"),
        ("netCubeOutputFile", "netcube.csv"),
    ):
        ET.SubElement(analytic, "Parameter", {"name": name}).text = value

    return ET.ElementTree(root)


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_hw2f_case(
    input_dir: str | Path,
    sigma: Iterable[Iterable[Iterable[float]]],
    kappa: Iterable[Iterable[float]],
    times: Iterable[float],
    base_case: str | Path | None = None,
    measure: str = "BA",
    *,
    samples: int = 32,
    seed: int = 42,
    grid: str = "24,6M",
) -> Hw2FCasePaths:
    sigma_list, kappa_list, times_list, param_type = _validate_sigma_kappa_inputs(sigma, kappa, times)
    case_paths = _normalize_case_paths(input_dir)
    base_input_dir = _resolve_base_input_dir(base_case)

    _copy_default_case_inputs(case_paths.input_dir, base_input_dir)

    simulation_tree = _build_simulation_xml(
        sigma_list,
        kappa_list,
        times_list,
        param_type,
        measure=measure,
        samples=samples,
        seed=seed,
        grid=grid,
    )
    ore_tree = _build_ore_xml(case_paths)

    _write_xml(simulation_tree, case_paths.simulation_xml)
    _write_xml(ore_tree, case_paths.ore_xml)

    return case_paths


def run_ore_case(case_dir: str | Path, ore_exe: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    case_dir = Path(case_dir).resolve()
    ore_xml = case_dir / "Input" / "ore.xml"
    if not ore_xml.exists():
        raise FileNotFoundError(f"ORE input file not found: {ore_xml}")
    ore_bin = Path(ore_exe).resolve() if ore_exe is not None else _default_ore_exe()
    result = subprocess.run(
        [str(ore_bin), str(ore_xml)],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ORE run failed")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def read_ore_results(case_dir: str | Path) -> dict[str, object]:
    case_dir = Path(case_dir).resolve()
    output_dir = case_dir / "Output"

    npv_rows = _read_csv(output_dir / "npv.csv")
    xva_rows = _read_csv(output_dir / "xva.csv")
    trade_exposure_files = sorted(output_dir.glob("exposure_trade_*.csv"))
    netting_exposure_files = sorted(output_dir.glob("exposure_nettingset_*.csv"))

    result: dict[str, object] = {
        "case_dir": str(case_dir),
        "output_dir": str(output_dir),
        "npv_rows": npv_rows,
        "xva_rows": xva_rows,
        "trade_exposure_files": [str(path) for path in trade_exposure_files],
        "netting_exposure_files": [str(path) for path in netting_exposure_files],
    }

    if npv_rows:
        first_npv = npv_rows[0]
        result["npv"] = {
            "trade_id": first_npv.get("#TradeId") or first_npv.get("TradeId"),
            "base_npv": _safe_float(first_npv.get("NPV(Base)") or first_npv.get("NPV")),
        }

    aggregate_xva = next((row for row in xva_rows if not (row.get("#TradeId") or row.get("TradeId"))), None)
    if aggregate_xva is not None:
        result["xva"] = {
            "cva": _safe_float(aggregate_xva.get("CVA")),
            "dva": _safe_float(aggregate_xva.get("DVA")),
            "fba": _safe_float(aggregate_xva.get("FBA")),
            "fca": _safe_float(aggregate_xva.get("FCA")),
            "basel_epe": _safe_float(aggregate_xva.get("BaselEPE")),
            "basel_eepe": _safe_float(aggregate_xva.get("BaselEEPE")),
        }

    return result


def _safe_float(value: str | None) -> float | None:
    if value in (None, "", "#N/A"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _default_sigma() -> list[list[list[float]]]:
    return [[[0.002, 0.008], [0.009, 0.001]]]


def _default_kappa() -> list[list[float]]:
    return [[0.01, 0.2]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run an ORE HW 2F case from Python.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=_repo_root() / "Tools" / "PythonOreRunner" / "_generated_hw2f_case",
        help="Target case directory containing Input/ and Output/.",
    )
    parser.add_argument(
        "--sigma-json",
        default=json.dumps(_default_sigma()),
        help="JSON list of 2x2 sigma matrices, e.g. '[[[0.002,0.008],[0.009,0.001]]]'",
    )
    parser.add_argument(
        "--kappa-json",
        default=json.dumps(_default_kappa()),
        help="JSON list of 2-element kappa vectors, e.g. '[[0.01,0.2]]'",
    )
    parser.add_argument(
        "--times-json",
        default="[]",
        help="JSON list of piecewise bucket times. Use [] for constant parameters.",
    )
    parser.add_argument("--measure", default="BA", help="Cross-asset model measure.")
    parser.add_argument("--samples", type=int, default=32, help="Simulation path count.")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed.")
    parser.add_argument("--grid", default="24,6M", help="Simulation grid string.")
    parser.add_argument("--base-case", type=Path, default=None, help="Optional base case root, Input dir, or ore.xml.")
    parser.add_argument("--ore-exe", type=Path, default=None, help="Optional explicit ore executable.")
    parser.add_argument("--build-only", action="store_true", help="Write the case but do not run ORE.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_paths = build_hw2f_case(
        args.case_dir / "Input",
        sigma=json.loads(args.sigma_json),
        kappa=json.loads(args.kappa_json),
        times=json.loads(args.times_json),
        base_case=args.base_case,
        measure=args.measure,
        samples=args.samples,
        seed=args.seed,
        grid=args.grid,
    )
    if args.build_only:
        print(json.dumps({"case_dir": str(case_paths.case_dir), "built": True}, indent=2))
        return 0

    run_ore_case(case_paths.case_dir, args.ore_exe)
    results = read_ore_results(case_paths.case_dir)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
