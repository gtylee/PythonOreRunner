from __future__ import annotations

from importlib import import_module
import sys

_mod = import_module("pythonore.benchmarks.benchmark_lgm_torch")
sys.modules[__name__] = _mod
