from importlib import import_module

_EXPORTS = {
    "BermudanBenchmarkResult": ("pythonore.runtime.bermudan", "BermudanBenchmarkResult"),
    "BermudanPricingResult": ("pythonore.runtime.bermudan", "BermudanPricingResult"),
    "BermudanPvSensitivity": ("pythonore.runtime.bermudan", "BermudanPvSensitivity"),
    "BermudanSensitivityBenchmarkResult": ("pythonore.runtime.bermudan", "BermudanSensitivityBenchmarkResult"),
    "BermudanSensitivityBenchmarkRow": ("pythonore.runtime.bermudan", "BermudanSensitivityBenchmarkRow"),
    "benchmark_bermudan_from_ore_case": ("pythonore.runtime.bermudan", "benchmark_bermudan_from_ore_case"),
    "benchmark_bermudan_sensitivities_from_ore_case": ("pythonore.runtime.bermudan", "benchmark_bermudan_sensitivities_from_ore_case"),
    "price_bermudan_from_ore_case": ("pythonore.runtime.bermudan", "price_bermudan_from_ore_case"),
    "price_bermudan_with_sensis_from_ore_case": ("pythonore.runtime.bermudan", "price_bermudan_with_sensis_from_ore_case"),
    "PythonDeltaGammaVarHelper": ("pythonore.runtime.dim", "PythonDeltaGammaVarHelper"),
    "PythonDimInput": ("pythonore.runtime.dim", "PythonDimInput"),
    "PythonDynamicDeltaVarCalculator": ("pythonore.runtime.dim", "PythonDynamicDeltaVarCalculator"),
    "PythonDynamicSimmCalculator": ("pythonore.runtime.dim", "PythonDynamicSimmCalculator"),
    "PythonSimpleDynamicSimm": ("pythonore.runtime.dim", "PythonSimpleDynamicSimm"),
    "PythonSimpleDynamicSimmConfig": ("pythonore.runtime.dim", "PythonSimpleDynamicSimmConfig"),
    "PythonVarDimInput": ("pythonore.runtime.dim", "PythonVarDimInput"),
    "ConflictError": ("pythonore.runtime.exceptions", "ConflictError"),
    "EngineRunError": ("pythonore.runtime.exceptions", "EngineRunError"),
    "InputCompatibilityError": ("pythonore.runtime.exceptions", "InputCompatibilityError"),
    "MappingError": ("pythonore.runtime.exceptions", "MappingError"),
    "ValidationError": ("pythonore.runtime.exceptions", "ValidationError"),
    "stress_classic_native_preset": ("pythonore.runtime.presets", "stress_classic_native_preset"),
    "stress_classic_native_runtime": ("pythonore.runtime.presets", "stress_classic_native_runtime"),
    "DeterministicToyAdapter": ("pythonore.runtime.runtime", "DeterministicToyAdapter"),
    "classify_portfolio_support": ("pythonore.runtime.runtime", "classify_portfolio_support"),
    "ORESwigAdapter": ("pythonore.runtime.runtime", "ORESwigAdapter"),
    "PythonLgmAdapter": ("pythonore.runtime.runtime", "PythonLgmAdapter"),
    "XVAEngine": ("pythonore.runtime.runtime", "XVAEngine"),
    "XVASession": ("pythonore.runtime.runtime", "XVASession"),
    "OreSnapshotPythonLgmSensitivityComparator": ("pythonore.runtime.sensitivity", "OreSnapshotPythonLgmSensitivityComparator"),
    "stress_classic_fixing_lines": ("pythonore.runtime.stress_classic_templates", "stress_classic_fixing_lines"),
    "stress_classic_market_lines": ("pythonore.runtime.stress_classic_templates", "stress_classic_market_lines"),
    "stress_classic_xml_buffers": ("pythonore.runtime.stress_classic_templates", "stress_classic_xml_buffers"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
