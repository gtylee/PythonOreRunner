from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "XVASnapshot": ("pythonore.domain", "XVASnapshot"),
    "XVAConfig": ("pythonore.domain", "XVAConfig"),
    "Trade": ("pythonore.domain", "Trade"),
    "Portfolio": ("pythonore.domain", "Portfolio"),
    "MarketData": ("pythonore.domain", "MarketData"),
    "XVALoader": ("pythonore.io", "XVALoader"),
    "merge_snapshots": ("pythonore.io", "merge_snapshots"),
    "map_snapshot": ("pythonore.mapping", "map_snapshot"),
    "build_input_parameters": ("pythonore.mapping", "build_input_parameters"),
    "XVAEngine": ("pythonore.runtime", "XVAEngine"),
    "XVASession": ("pythonore.runtime", "XVASession"),
    "ORESwigAdapter": ("pythonore.runtime", "ORESwigAdapter"),
    "PythonLgmAdapter": ("pythonore.runtime", "PythonLgmAdapter"),
    "DeterministicToyAdapter": ("pythonore.runtime", "DeterministicToyAdapter"),
    "classify_portfolio_support": ("pythonore.runtime", "classify_portfolio_support"),
    "BufferCaseInputs": ("pythonore.workflows", "BufferCaseInputs"),
    "PurePythonRunOptions": ("pythonore.workflows", "PurePythonRunOptions"),
    "PurePythonCaseResult": ("pythonore.workflows", "PurePythonCaseResult"),
    "OreSnapshotApp": ("pythonore.workflows", "OreSnapshotApp"),
    "run_case_from_buffers": ("pythonore.workflows", "run_case_from_buffers"),
    "ExampleRegressionCase": ("pythonore.workflows", "ExampleRegressionCase"),
    "ExampleRegressionResult": ("pythonore.workflows", "ExampleRegressionResult"),
    "DEFAULT_MANIFEST": ("pythonore.workflows", "DEFAULT_MANIFEST"),
    "DEFAULT_BASELINE_ROOT": ("pythonore.workflows", "DEFAULT_BASELINE_ROOT"),
    "load_manifest": ("pythonore.workflows", "load_manifest"),
    "refresh_baselines": ("pythonore.workflows", "refresh_baselines"),
    "compare_baselines": ("pythonore.workflows", "compare_baselines"),
    "PayoffModuleIR": ("pythonore.payoff_ir", "PayoffModuleIR"),
    "lower_ore_script": ("pythonore.payoff_ir", "lower_ore_script"),
    "lower_python_payoff": ("pythonore.payoff_ir", "lower_python_payoff"),
    "emit_ore_script": ("pythonore.payoff_ir", "emit_ore_script"),
    "normalize_module": ("pythonore.payoff_ir", "normalize_module"),
    "validate_module": ("pythonore.payoff_ir", "validate_module"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
