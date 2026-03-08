# ORE-Based LGM+FX Hybrid Parity Framework

## Summary
This extension adds a multi-currency LGM + FX hybrid layer on top of the existing 1F LGM IRS parity workflow.

Implemented scope:
- Canonical parity artifact layout with reproducible manifests and command logs
- Multi-ccy model API (`MultiCcyLgmParams`, `LgmFxHybrid`)
- Product valuation utilities for IRS / FX Forward / XCCY Float-Float
- Reconciliation harness with per-date ORE-vs-Python exposure diagnostics and CVA decomposition
- Calibrated-parameter export utility for replayable calibrated runs

## New Modules
- [lgm_fx_hybrid.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/lgm_fx_hybrid.py)
  - `MultiCcyLgmParams`
  - `LgmFxHybrid.simulate_paths(...)`
  - `zc_bond(...)`
  - `fx_forward(...)`
  - Correlation PSD handling with deterministic eigenvalue clipping fallback

- [lgm_fx_xva_utils.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/lgm_fx_xva_utils.py)
  - `fx_forward_npv(...)`
  - `single_ccy_irs_npv(...)`
  - `xccy_float_float_swap_npv(...)`
  - `aggregate_exposure_profile(...)`
  - `cva_terms_from_profile(...)`

- [ore_parity_artifacts.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/ore_parity_artifacts.py)
  - Canonical case layout: `times/`, `curves/`, `trades/`, `calibration/`, `exposure/`, `xva/`, `perf/`
  - Manifest + command logging

## Reconciliation and Benchmark Scripts
- [compare_ore_python_lgm_fx.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/demos/compare_ore_python_lgm_fx.py)
  - Product modes: `irs_single`, `fx_forward`, `xccy_float_float`
  - Outputs:
    - `<trade>_diagnostics.csv` with `ORE_EPE/PY_EPE` + abs/rel diffs, plus ENE/EE columns
    - `<trade>_cva_terms.csv` with per-date `EPE*DF*dPD*LGD` decomposition
    - `<trade>_summary.json` with CVA summary metrics and error stats

- [benchmark_lgm_fx_hybrid_ore.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/benchmarks/benchmark_lgm_fx_hybrid_ore.py)
  - Benchmark matrix generation across:
    - products: IRS, FXFWD, XCCY
    - markets: flat, full
    - currencies/pairs: EUR/USD/GBP/CAD and GBP/USD, USD/CAD
    - maturities: 1Y, 5Y, 10Y
    - conventions: A, B
  - Supports `fixed` and `calibrated` run modes
  - Stores per-case ORE settings and run commands for exact replay

## Calibration Export
- [export_ore_lgm_calibration.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/demos/export_ore_lgm_calibration.py)
  - Converts ORE calibration XML into normalized JSON and CSV
  - Intended for calibrated parity replay and drift control

## Notes
- Existing `LGM1F` module and tests remain unchanged.
- Hybrid implementation is designed to be composed with existing single-ccy utilities.
- For strict production parity, keep ORE outputs as truth source for conventions, exposure grid, and XVA aggregation.
