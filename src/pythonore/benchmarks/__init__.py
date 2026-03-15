from importlib import import_module

_MODULE_EXPORTS = {
    "benchmark_lgm_fx_forward_torch": "py_ore_tools.benchmarks.benchmark_lgm_fx_forward_torch",
    "benchmark_lgm_fx_hybrid_torch": "py_ore_tools.benchmarks.benchmark_lgm_fx_hybrid_torch",
    "benchmark_lgm_fx_portfolio_torch": "py_ore_tools.benchmarks.benchmark_lgm_fx_portfolio_torch",
    "benchmark_lgm_torch": "py_ore_tools.benchmarks.benchmark_lgm_torch",
    "benchmark_lgm_torch_swap": "py_ore_tools.benchmarks.benchmark_lgm_torch_swap",
}

__all__ = sorted(_MODULE_EXPORTS)


def __getattr__(name: str):
    if name not in _MODULE_EXPORTS:
        raise AttributeError(name)
    value = import_module(_MODULE_EXPORTS[name])
    globals()[name] = value
    return value
