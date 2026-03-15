from __future__ import annotations

import os
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
    "RateFutureModelParams": ("pythonore.compute.rate_futures", "RateFutureModelParams"),
    "extract_market_instruments_by_currency_from_quotes": (
        "pythonore.compute.rate_futures",
        "extract_market_instruments_by_currency_from_quotes",
    ),
    "fit_discount_curves_from_programmatic_quotes": (
        "pythonore.compute.rate_futures",
        "fit_discount_curves_from_programmatic_quotes",
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


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
