from importlib import import_module

_EXPORTS = {
    "BufferCaseInputs": ("pythonore.workflows.ore_snapshot_cli", "BufferCaseInputs"),
    "OreSnapshotApp": ("pythonore.workflows.ore_snapshot_cli", "OreSnapshotApp"),
    "PurePythonCaseResult": ("pythonore.workflows.ore_snapshot_cli", "PurePythonCaseResult"),
    "PurePythonRunOptions": ("pythonore.workflows.ore_snapshot_cli", "PurePythonRunOptions"),
    "run_case_from_buffers": ("pythonore.workflows.ore_snapshot_cli", "run_case_from_buffers"),
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
