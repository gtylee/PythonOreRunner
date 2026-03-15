from importlib import import_module

_MODULE_EXPORTS = {
    "benchmark_bond_pricing_numpy_torch": "pythonore.benchmarks.benchmark_bond_pricing_numpy_torch",
    "benchmark_discount_factor_extractor": "pythonore.benchmarks.benchmark_discount_factor_extractor",
    "benchmark_lgm_fx_forward_torch": "pythonore.benchmarks.benchmark_lgm_fx_forward_torch",
    "benchmark_lgm_fx_hybrid_ore": "pythonore.benchmarks.benchmark_lgm_fx_hybrid_ore",
    "benchmark_lgm_fx_hybrid_torch": "pythonore.benchmarks.benchmark_lgm_fx_hybrid_torch",
    "benchmark_lgm_fx_portfolio_torch": "pythonore.benchmarks.benchmark_lgm_fx_portfolio_torch",
    "benchmark_lgm_ore_multiccy": "pythonore.benchmarks.benchmark_lgm_ore_multiccy",
    "benchmark_lgm_torch": "pythonore.benchmarks.benchmark_lgm_torch",
    "benchmark_lgm_torch_swap": "pythonore.benchmarks.benchmark_lgm_torch_swap",
    "benchmark_ore_fx_forwards": "pythonore.benchmarks.benchmark_ore_fx_forwards",
    "benchmark_ore_fx_forwards_xva": "pythonore.benchmarks.benchmark_ore_fx_forwards_xva",
    "benchmark_ore_ir_options": "pythonore.benchmarks.benchmark_ore_ir_options",
}

__all__ = sorted(_MODULE_EXPORTS)


def __getattr__(name: str):
    if name not in _MODULE_EXPORTS:
        raise AttributeError(name)
    value = import_module(_MODULE_EXPORTS[name])
    globals()[name] = value
    return value
