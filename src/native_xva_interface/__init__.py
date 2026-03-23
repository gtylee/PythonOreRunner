from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]
_LEGACY_DIR = _REPO_ROOT / "legacy" / "native_xva_interface"
_SRC_ROOT = _REPO_ROOT / "src"
_SRC = _REPO_ROOT / "src" / "pythonore"

if _SRC_ROOT.exists():
    src_root_text = str(_SRC_ROOT)
    if src_root_text not in sys.path:
        sys.path.insert(0, src_root_text)

__path__ = [str(_PKG_DIR)]
for extra in (
    _LEGACY_DIR,
    _SRC / "domain",
    _SRC / "io",
    _SRC / "mapping",
    _SRC / "runtime",
):
    if extra.exists():
        __path__.append(str(extra))

_EXPORTS = {
    "CollateralBalance": ("pythonore.domain.dataclasses", "CollateralBalance"),
    "CollateralConfig": ("pythonore.domain.dataclasses", "CollateralConfig"),
    "BermudanBenchmarkResult": ("pythonore.runtime.bermudan", "BermudanBenchmarkResult"),
    "BermudanPricingResult": ("pythonore.runtime.bermudan", "BermudanPricingResult"),
    "BermudanPvSensitivity": ("pythonore.runtime.bermudan", "BermudanPvSensitivity"),
    "BermudanSensitivityBenchmarkResult": ("pythonore.runtime.bermudan", "BermudanSensitivityBenchmarkResult"),
    "BermudanSensitivityBenchmarkRow": ("pythonore.runtime.bermudan", "BermudanSensitivityBenchmarkRow"),
    "BermudanSwaption": ("pythonore.domain.dataclasses", "BermudanSwaption"),
    "ConventionsConfig": ("pythonore.domain.dataclasses", "ConventionsConfig"),
    "ConflictError": ("pythonore.runtime.exceptions", "ConflictError"),
    "CounterpartyConfig": ("pythonore.domain.dataclasses", "CounterpartyConfig"),
    "CreditCurve": ("pythonore.domain.dataclasses", "CreditCurve"),
    "CreditEntityConfig": ("pythonore.domain.dataclasses", "CreditEntityConfig"),
    "CreditSimulationConfig": ("pythonore.domain.dataclasses", "CreditSimulationConfig"),
    "CrossAssetModelConfig": ("pythonore.domain.dataclasses", "CrossAssetModelConfig"),
    "CubeAccessor": ("pythonore.runtime.results", "CubeAccessor"),
    "Curve": ("pythonore.domain.dataclasses", "Curve"),
    "CurveConfig": ("pythonore.domain.dataclasses", "CurveConfig"),
    "CurvePoint": ("pythonore.domain.dataclasses", "CurvePoint"),
    "DeterministicToyAdapter": ("pythonore.runtime.runtime", "DeterministicToyAdapter"),
    "DIMMarginComponents": ("pythonore.runtime.results", "DIMMarginComponents"),
    "DIMResult": ("pythonore.runtime.results", "DIMResult"),
    "EngineRunError": ("pythonore.runtime.exceptions", "EngineRunError"),
    "EuropeanOption": ("pythonore.domain.dataclasses", "EuropeanOption"),
    "FixingPoint": ("pythonore.domain.dataclasses", "FixingPoint"),
    "FixingsData": ("pythonore.domain.dataclasses", "FixingsData"),
    "FXForward": ("pythonore.domain.dataclasses", "FXForward"),
    "FXQuote": ("pythonore.domain.dataclasses", "FXQuote"),
    "FXQuotes": ("pythonore.domain.dataclasses", "FXQuotes"),
    "GenericProduct": ("pythonore.domain.dataclasses", "GenericProduct"),
    "InputCompatibilityError": ("pythonore.runtime.exceptions", "InputCompatibilityError"),
    "IRS": ("pythonore.domain.dataclasses", "IRS"),
    "MappedInputs": ("pythonore.mapping.mapper", "MappedInputs"),
    "MappingError": ("pythonore.runtime.exceptions", "MappingError"),
    "MarketData": ("pythonore.domain.dataclasses", "MarketData"),
    "MarketQuote": ("pythonore.domain.dataclasses", "MarketQuote"),
    "MporConfig": ("pythonore.domain.dataclasses", "MporConfig"),
    "NettingConfig": ("pythonore.domain.dataclasses", "NettingConfig"),
    "NettingSet": ("pythonore.domain.dataclasses", "NettingSet"),
    "ORESensitivityEntry": ("pythonore.runtime.sensitivity", "ORESensitivityEntry"),
    "ORESwigAdapter": ("pythonore.runtime.runtime", "ORESwigAdapter"),
    "OreSnapshotPythonLgmSensitivityComparator": (
        "pythonore.runtime.sensitivity",
        "OreSnapshotPythonLgmSensitivityComparator",
    ),
    "ParityCheckResult": ("pythonore.runtime.parity", "ParityCheckResult"),
    "ParityTolerance": ("pythonore.runtime.parity", "ParityTolerance"),
    "Portfolio": ("pythonore.domain.dataclasses", "Portfolio"),
    "PricingEngineConfig": ("pythonore.domain.dataclasses", "PricingEngineConfig"),
    "Product": ("pythonore.domain.dataclasses", "Product"),
    "PythonDeltaGammaVarHelper": ("pythonore.runtime.dim", "PythonDeltaGammaVarHelper"),
    "PythonDimInput": ("pythonore.runtime.dim", "PythonDimInput"),
    "PythonDynamicDeltaVarCalculator": ("pythonore.runtime.dim", "PythonDynamicDeltaVarCalculator"),
    "PythonDynamicSimmCalculator": ("pythonore.runtime.dim", "PythonDynamicSimmCalculator"),
    "PythonLgmAdapter": ("pythonore.runtime.runtime", "PythonLgmAdapter"),
    "PythonSensitivityEntry": ("pythonore.runtime.sensitivity", "PythonSensitivityEntry"),
    "PythonSimpleDynamicSimm": ("pythonore.runtime.dim", "PythonSimpleDynamicSimm"),
    "PythonSimpleDynamicSimmConfig": ("pythonore.runtime.dim", "PythonSimpleDynamicSimmConfig"),
    "PythonVarDimInput": ("pythonore.runtime.dim", "PythonVarDimInput"),
    "RuntimeConfig": ("pythonore.domain.dataclasses", "RuntimeConfig"),
    "SensitivityComparisonEntry": ("pythonore.runtime.sensitivity", "SensitivityComparisonEntry"),
    "SimulationConfig": ("pythonore.domain.dataclasses", "SimulationConfig"),
    "SimulationMarketConfig": ("pythonore.domain.dataclasses", "SimulationMarketConfig"),
    "SourceMeta": ("pythonore.domain.dataclasses", "SourceMeta"),
    "TodaysMarketConfig": ("pythonore.domain.dataclasses", "TodaysMarketConfig"),
    "Trade": ("pythonore.domain.dataclasses", "Trade"),
    "ValidationError": ("pythonore.runtime.exceptions", "ValidationError"),
    "XVAAnalyticConfig": ("pythonore.domain.dataclasses", "XVAAnalyticConfig"),
    "XVAConfig": ("pythonore.domain.dataclasses", "XVAConfig"),
    "XVAEngine": ("pythonore.runtime.runtime", "XVAEngine"),
    "XVALoader": ("pythonore.io.loader", "XVALoader"),
    "XVAResult": ("pythonore.runtime.results", "XVAResult"),
    "XVASession": ("pythonore.runtime.runtime", "XVASession"),
    "XVASnapshot": ("pythonore.domain.dataclasses", "XVASnapshot"),
    "benchmark_bermudan_from_ore_case": ("pythonore.runtime.bermudan", "benchmark_bermudan_from_ore_case"),
    "benchmark_bermudan_sensitivities_from_ore_case": (
        "pythonore.runtime.bermudan",
        "benchmark_bermudan_sensitivities_from_ore_case",
    ),
    "build_input_parameters": ("pythonore.mapping.mapper", "build_input_parameters"),
    "compare_results": ("pythonore.runtime.parity", "compare_results"),
    "classify_portfolio_support": ("pythonore.runtime.runtime", "classify_portfolio_support"),
    "map_snapshot": ("pythonore.mapping.mapper", "map_snapshot"),
    "merge_snapshots": ("pythonore.io.loader", "merge_snapshots"),
    "price_bermudan_from_ore_case": ("pythonore.runtime.bermudan", "price_bermudan_from_ore_case"),
    "price_bermudan_with_sensis_from_ore_case": (
        "pythonore.runtime.bermudan",
        "price_bermudan_with_sensis_from_ore_case",
    ),
    "stress_classic_fixing_lines": ("pythonore.runtime.stress_classic_templates", "stress_classic_fixing_lines"),
    "stress_classic_market_lines": ("pythonore.runtime.stress_classic_templates", "stress_classic_market_lines"),
    "stress_classic_native_preset": ("pythonore.runtime.presets", "stress_classic_native_preset"),
    "stress_classic_native_runtime": ("pythonore.runtime.presets", "stress_classic_native_runtime"),
    "stress_classic_xml_buffers": ("pythonore.runtime.stress_classic_templates", "stress_classic_xml_buffers"),
}

__all__ = sorted(_EXPORTS)

_SUBMODULES = {
    "bermudan": "pythonore.runtime.bermudan",
    "dataclasses": "pythonore.domain.dataclasses",
    "dim": "pythonore.runtime.dim",
    "exceptions": "pythonore.runtime.exceptions",
    "loader": "pythonore.io.loader",
    "mapper": "pythonore.mapping.mapper",
    "parity": "pythonore.runtime.parity",
    "presets": "pythonore.runtime.presets",
    "results": "pythonore.runtime.results",
    "runtime": "pythonore.runtime.runtime",
    "sensitivity": "pythonore.runtime.sensitivity",
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
