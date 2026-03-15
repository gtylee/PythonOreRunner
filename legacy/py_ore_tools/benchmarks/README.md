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
- `ENGINE_REPO_ROOT/Examples/Exposure/Input/ore_measure_lgm.xml` when available
- otherwise the vendored fixture under `py_ore_tools/benchmarks/fixtures/discount_factor_extractor/`
- Optional `--ore-xml` override

Outputs:
- Prints timing summary only

Example:

```bash
python py_ore_tools/benchmarks/benchmark_discount_factor_extractor.py --runs 10
```

This benchmark can run without any `Engine` checkout.

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

Notes:
- The Python compare path now defaults to the stronger local parity settings used in this repo: pathwise fixing lock on and `alpha_scale=1.05`.

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

### `benchmark_bond_pricing_numpy_torch.py`

Purpose:
Benchmark the new compiled-trade bond scenario pricers with a scalar-loop
baseline plus NumPy and torch side by side.

Inputs:
- Local `Example_18` bond fixtures already vendored in this repo

Outputs:
- Prints timing and parity JSON only

Example:

```bash
python py_ore_tools/benchmarks/benchmark_bond_pricing_numpy_torch.py --scenarios 1000 10000 --devices cpu gpu
```

### `benchmark_ore_ir_options.py`

Purpose:
Benchmark cap/floor and Bermudan swaption PV and XVA against ORE.

Current Bermudan benchmark rule:
- keep the main ORE PV/XVA run in `Output/`
- also run a classic calibration pass in `Output/classic/`
- build the Python Bermudan LGM model from `Output/classic/calibration.xml` when present

On this repo that matters a lot more than the old flat simulation stub. With the classic calibration source restored, the 5Y Bermudan backward PV rows moved from about `+8.4% / +93.3% / -0.04%` to about `-0.67% / +27.8% / -2.82%` for `BERM_EUR_5Y / LOWK / HIGHK` at `--ore-samples 1024`.

Inputs:
- ORE example market/config files from `ENGINE_REPO_ROOT/Examples/Input`
- Exposure support files from `ENGINE_REPO_ROOT/Examples/Exposure/Input`

Outputs:
- Local result set under `parity_artifacts/ir_options_ore_benchmark/`

Notes:
- The EUR cap cases in this benchmark should use `EUR-EURIBOR-6M` with a `6M` coupon schedule. The example ORE market configuration exposes the EUR normal cap/floor surface on `6M`, not `3M`, so a `3M` benchmark setup creates a false parity gap.
- Cap PV parity is best obtained here from a Bachelier caplet sum using the quoted `CAPFLOOR/RATE_NVOL/EUR/2Y/6M/...` normal vols. The old LGM-only cap path materially underpriced deep OTM tails or overstated low-strike caps.

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
