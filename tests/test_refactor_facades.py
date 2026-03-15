from __future__ import annotations

import importlib


def test_native_xva_loader_is_canonical_module():
    old_mod = importlib.import_module("native_xva_interface.loader")
    new_mod = importlib.import_module("pythonore.io.loader")
    assert old_mod is new_mod


def test_py_ore_tools_lgm_is_canonical_module():
    old_mod = importlib.import_module("py_ore_tools.lgm")
    new_mod = importlib.import_module("pythonore.compute.lgm")
    assert old_mod is new_mod


def test_py_ore_tools_lgm_torch_xva_is_canonical_module():
    old_mod = importlib.import_module("py_ore_tools.lgm_torch_xva")
    new_mod = importlib.import_module("pythonore.compute.lgm_torch_xva")
    assert old_mod is new_mod


def test_py_ore_tools_package_exports_canonical_compute_symbols():
    import py_ore_tools
    from pythonore.compute.lgm import LGM1F, LGMParams

    assert py_ore_tools.LGM1F is LGM1F
    assert py_ore_tools.LGMParams is LGMParams


def test_old_benchmark_wrapper_re_exports_canonical_main():
    old_mod = importlib.import_module("py_ore_tools.benchmarks.benchmark_lgm_torch")
    new_mod = importlib.import_module("pythonore.benchmarks.benchmark_lgm_torch")
    assert old_mod.main is new_mod.main
