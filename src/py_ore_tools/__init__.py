from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]
_LEGACY_DIR = _REPO_ROOT / "legacy" / "py_ore_tools"
_SRC = _REPO_ROOT / "src" / "pythonore"

_CACHE_ROOT = _REPO_ROOT / ".cache" / "py_ore_tools"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_FONTCONFIG_CACHE = _CACHE_ROOT / "fontconfig"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_FONTCONFIG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

__path__ = [str(_PKG_DIR)]
for extra in (
    _LEGACY_DIR,
    _SRC,
    _SRC / "compute",
    _SRC / "io",
    _SRC / "workflows",
    _SRC / "benchmarks",
    _SRC / "demos",
):
    if extra.exists():
        __path__.append(str(extra))

_EXPORTS = {
    "OreBasic": ("pythonore.ore", "OreBasic"),
    "LGM1F": ("pythonore.compute.lgm", "LGM1F"),
    "LGMParams": ("pythonore.compute.lgm", "LGMParams"),
    "ORE_PARITY_SEQUENCE_TYPE": ("pythonore.compute.lgm", "ORE_PARITY_SEQUENCE_TYPE"),
    "OreMersenneTwisterGaussianRng": ("pythonore.compute.lgm", "OreMersenneTwisterGaussianRng"),
    "make_ore_gaussian_rng": ("pythonore.compute.lgm", "make_ore_gaussian_rng"),
    "simulate_lgm_measure": ("pythonore.compute.lgm", "simulate_lgm_measure"),
    "simulate_ba_measure": ("pythonore.compute.lgm", "simulate_ba_measure"),
    "build_discount_curve_from_discount_pairs": ("pythonore.compute.irs_xva_utils", "build_discount_curve_from_discount_pairs"),
    "build_discount_curve_from_zero_rate_pairs": ("pythonore.compute.irs_xva_utils", "build_discount_curve_from_zero_rate_pairs"),
    "CurveDFPayload": ("pythonore.io.ore_snapshot", "CurveDFPayload"),
    "OreSnapshot": ("pythonore.io.ore_snapshot", "OreSnapshot"),
    "load_from_ore_xml": ("pythonore.io.ore_snapshot", "load_from_ore_xml"),
    "validate_ore_input_snapshot": ("pythonore.io.ore_snapshot", "validate_ore_input_snapshot"),
    "ore_input_validation_dataframe": ("pythonore.io.ore_snapshot", "ore_input_validation_dataframe"),
    "validate_xva_snapshot_dataclasses": ("pythonore.io.ore_snapshot", "validate_xva_snapshot_dataclasses"),
    "xva_snapshot_validation_dataframe": ("pythonore.io.ore_snapshot", "xva_snapshot_validation_dataframe"),
    "extract_discount_factors_by_currency": ("pythonore.io.ore_snapshot", "extract_discount_factors_by_currency"),
    "discount_factors_to_dataframe": ("pythonore.io.ore_snapshot", "discount_factors_to_dataframe"),
    "extract_market_instruments_by_currency": ("pythonore.io.ore_snapshot", "extract_market_instruments_by_currency"),
    "RateFutureModelParams": ("pythonore.compute.rate_futures", "RateFutureModelParams"),
    "InflationCurve": ("pythonore.compute.inflation", "InflationCurve"),
    "InflationLgmParams": ("pythonore.compute.inflation", "InflationLgmParams"),
    "JarrowYildirimParams": ("pythonore.compute.inflation", "JarrowYildirimParams"),
    "InflationModelSpec": ("pythonore.compute.inflation", "InflationModelSpec"),
    "parse_inflation_models_from_simulation_xml": (
        "pythonore.compute.inflation",
        "parse_inflation_models_from_simulation_xml",
    ),
    "load_inflation_curve_from_market_data": (
        "pythonore.compute.inflation",
        "load_inflation_curve_from_market_data",
    ),
    "load_zero_inflation_surface_quote": (
        "pythonore.compute.inflation",
        "load_zero_inflation_surface_quote",
    ),
    "price_zero_coupon_cpi_swap": ("pythonore.compute.inflation", "price_zero_coupon_cpi_swap"),
    "price_yoy_swap": ("pythonore.compute.inflation", "price_yoy_swap"),
    "price_inflation_capfloor": ("pythonore.compute.inflation", "price_inflation_capfloor"),
    "simulate_inflation_index_paths": ("pythonore.compute.inflation", "simulate_inflation_index_paths"),
    "extract_market_instruments_by_currency_from_quotes": (
        "pythonore.io.ore_snapshot",
        "extract_market_instruments_by_currency_from_quotes",
    ),
    "fit_discount_curves_from_ore_market": ("pythonore.io.ore_snapshot", "fit_discount_curves_from_ore_market"),
    "fit_discount_curves_from_programmatic_quotes": (
        "pythonore.io.ore_snapshot",
        "fit_discount_curves_from_programmatic_quotes",
    ),
    "fitted_curves_to_dataframe": ("pythonore.io.ore_snapshot", "fitted_curves_to_dataframe"),
    "quote_dicts_from_pairs": ("pythonore.io.ore_snapshot", "quote_dicts_from_pairs"),
    "parse_lgm_params_from_calibration_xml": (
        "pythonore.compute.irs_xva_utils",
        "parse_lgm_params_from_calibration_xml",
    ),
    "parse_lgm_params_from_simulation_xml": (
        "pythonore.compute.irs_xva_utils",
        "parse_lgm_params_from_simulation_xml",
    ),
    "DEFAULT_BASELINE_ROOT": ("pythonore.workflows.examples_regression", "DEFAULT_BASELINE_ROOT"),
    "DEFAULT_MANIFEST": ("pythonore.workflows.examples_regression", "DEFAULT_MANIFEST"),
    "ExampleRegressionCase": ("pythonore.workflows.examples_regression", "ExampleRegressionCase"),
    "ExampleRegressionResult": ("pythonore.workflows.examples_regression", "ExampleRegressionResult"),
    "compare_baselines": ("pythonore.workflows.examples_regression", "compare_baselines"),
    "load_manifest": ("pythonore.workflows.examples_regression", "load_manifest"),
    "refresh_baselines": ("pythonore.workflows.examples_regression", "refresh_baselines"),
}

__all__ = sorted(_EXPORTS)

_SUBMODULES = {
    "bond_pricing": "pythonore.compute.bond_pricing",
    "file_lists": "pythonore.file_lists",
    "hw2f": "pythonore.hw2f",
    "hw2f_integration": "pythonore.hw2f_integration",
    "hw2f_ore_runner": "pythonore.hw2f_ore_runner",
    "irs_xva_utils": "pythonore.compute.irs_xva_utils",
    "inflation": "pythonore.compute.inflation",
    "lgm": "pythonore.compute.lgm",
    "lgm_calibration": "pythonore.compute.lgm_calibration",
    "lgm_fx_hybrid": "pythonore.compute.lgm_fx_hybrid",
    "lgm_fx_hybrid_torch": "pythonore.compute.lgm_fx_hybrid_torch",
    "lgm_fx_xva_utils": "pythonore.compute.lgm_fx_xva_utils",
    "lgm_ir_options": "pythonore.compute.lgm_ir_options",
    "lgm_torch": "pythonore.compute.lgm_torch",
    "lgm_torch_xva": "pythonore.compute.lgm_torch_xva",
    "ore": "pythonore.ore",
    "ore_parity_artifacts": "pythonore.parity_artifacts",
    "ore_snapshot": "pythonore.io.ore_snapshot",
    "ore_snapshot_cli": "pythonore.workflows.ore_snapshot_cli",
    "plotter": "pythonore.plotter",
    "rate_futures": "pythonore.compute.rate_futures",
    "repo_paths": "pythonore.repo_paths",
}

for old_name, new_name in _SUBMODULES.items():
    sys.modules.setdefault(f"{__name__}.{old_name}", import_module(new_name))


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
