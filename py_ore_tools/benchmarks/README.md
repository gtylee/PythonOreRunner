# Benchmarks

Benchmark entrypoints in this folder are intended to run from the standalone
`PythonOreRunner` checkout while reading ORE example inputs from a separate
`Engine` checkout.

## Prerequisites

- Set `ENGINE_REPO_ROOT` to your ORE `Engine` repository.
- Build the `ore` executable in that checkout, or pass `--ore-bin`.
- Install Python dependencies from [`requirements.txt`](/Users/gordonlee/Documents/PythonOreRunner/requirements.txt).

Typical setup:

```bash
export ENGINE_REPO_ROOT=/path/to/Engine
export ORE_EXE="$ENGINE_REPO_ROOT/build/apple-make-relwithdebinfo-arm64/App/ore"
```

Notes:

- If `--ore-bin` is omitted, scripts default to the `ore` binary under `ENGINE_REPO_ROOT`.
- Generated outputs are written under this repo's local `parity_artifacts/`, not back into `Engine`.
- Scripts can also be run with `python -m` if you prefer, but direct invocation from the repo root is supported.

## Scripts

### `benchmark_discount_factor_extractor.py`

Purpose:
Measure the cost of extracting per-currency discount-factor pillars from an ORE XML case.

Inputs:
- `ENGINE_REPO_ROOT/Examples/Exposure/Input/ore_measure_lgm.xml` by default
- Optional `--ore-xml` override

Outputs:
- Prints timing summary only

Example:

```bash
python py_ore_tools/benchmarks/benchmark_discount_factor_extractor.py --runs 10
```

### `benchmark_ore_fx_forwards.py`

Purpose:
Run focused ORE FX forward valuation cases and compare ORE t0 NPV with the Python closed-form result.

Inputs:
- ORE example market/config files from `ENGINE_REPO_ROOT/Examples/Input`

Outputs:
- Local result set under `parity_artifacts/fxfwd_ore_benchmark/`

Example:

```bash
python py_ore_tools/benchmarks/benchmark_ore_fx_forwards.py
```

### `benchmark_ore_fx_forwards_xva.py`

Purpose:
Run ORE XVA benchmark cases for demo FX forwards.

Inputs:
- ORE input/config files from `ENGINE_REPO_ROOT/Examples/Input`
- Exposure/XVA support files from `ENGINE_REPO_ROOT/Examples/Exposure/Input`

Outputs:
- Local result set under `parity_artifacts/fxfwd_ore_xva_benchmark/`

Example:

```bash
python py_ore_tools/benchmarks/benchmark_ore_fx_forwards_xva.py --samples 2000
```

### `benchmark_lgm_ore_multiccy.py`

Purpose:
Generate and run multi-currency IRS parity benchmark cases across currencies, markets, maturities, and convention sets.

Inputs:
- ORE example market/config files from `ENGINE_REPO_ROOT/Examples/Input`
- Exposure support files from `ENGINE_REPO_ROOT/Examples/Exposure/Input`

Outputs:
- Local result set under `parity_artifacts/multiccy_benchmark/`

Example:

```bash
python py_ore_tools/benchmarks/benchmark_lgm_ore_multiccy.py --max-cases 2
```

### `benchmark_lgm_fx_hybrid_ore.py`

Purpose:
Build and optionally execute ORE-based LGM+FX hybrid parity benchmark cases for IRS, FX forwards, and XCCY swaps.

Inputs:
- ORE example market/config files from `ENGINE_REPO_ROOT/Examples/Input`
- Exposure support files from `ENGINE_REPO_ROOT/Examples/Exposure/Input`

Outputs:
- Local case layouts and results under `parity_artifacts/lgm_fx_hybrid_benchmark/`

Example:

```bash
python py_ore_tools/benchmarks/benchmark_lgm_fx_hybrid_ore.py --max-cases 2
python py_ore_tools/benchmarks/benchmark_lgm_fx_hybrid_ore.py --execute --max-cases 2
```

### `benchmark_ore_ir_options.py`

Purpose:
Benchmark cap/floor and Bermudan swaption PV and XVA against ORE.

Inputs:
- ORE example market/config files from `ENGINE_REPO_ROOT/Examples/Input`
- Exposure support files from `ENGINE_REPO_ROOT/Examples/Exposure/Input`

Outputs:
- Local result set under `parity_artifacts/ir_options_ore_benchmark/`

Example:

```bash
python py_ore_tools/benchmarks/benchmark_ore_ir_options.py
```

## Troubleshooting

- `Could not locate an Engine checkout with Examples/`:
  Set `ENGINE_REPO_ROOT` explicitly.
- `ModuleNotFoundError` when invoking a script directly:
  Run from the repository root so the script's local import bootstrap resolves correctly.
- Missing `ore` binary:
  Build ORE in the `Engine` checkout or pass `--ore-bin /path/to/ore`.
- Large outputs:
  Benchmark outputs are intentionally ignored by git under `parity_artifacts/`.
