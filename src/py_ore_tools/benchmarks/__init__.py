from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[2]
_LEGACY_DIR = _REPO_ROOT / "legacy" / "py_ore_tools" / "benchmarks"
_CANONICAL_DIR = _REPO_ROOT / "src" / "pythonore" / "benchmarks"

__path__ = [str(_PKG_DIR)]
for extra in (_CANONICAL_DIR, _LEGACY_DIR):
    if extra.exists():
        __path__.append(str(extra))

_CANONICAL_BENCHMARKS = (
    "benchmark_bond_pricing_numpy_torch",
    "benchmark_discount_factor_extractor",
    "benchmark_lgm_calibration_parity",
    "benchmark_lgm_fx_forward_torch",
    "benchmark_lgm_fx_hybrid_ore",
    "benchmark_lgm_fx_hybrid_torch",
    "benchmark_lgm_fx_portfolio_torch",
    "benchmark_lgm_ore_multiccy",
    "benchmark_lgm_torch",
    "benchmark_lgm_torch_swap",
    "benchmark_ore_fx_forwards",
    "benchmark_ore_fx_forwards_xva",
    "benchmark_ore_ir_options",
)

for module_name in _CANONICAL_BENCHMARKS:
    module = import_module(f"pythonore.benchmarks.{module_name}")
    sys.modules[f"{__name__}.{module_name}"] = module
    sys.modules.setdefault(f"pythonore.benchmarks.{module_name}", module)
    globals()[module_name] = module
