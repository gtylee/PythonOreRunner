from __future__ import annotations

from importlib import import_module
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_SRC_IMPL = _PKG_DIR.parent / "src" / "pythonore"

__path__ = [str(_PKG_DIR)]
if _SRC_IMPL.exists():
    __path__.append(str(_SRC_IMPL))

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
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
