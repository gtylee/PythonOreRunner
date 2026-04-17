# PythonOreRunner

Python tools for loading ORE-style cases, running native Python pricing/XVA flows, and comparing Python results against ORE reference outputs.

The main maintained code lives under [`src/pythonore/`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore). Legacy import surfaces such as `py_ore_tools` remain available as compatibility shims.

## What This Repo Is For

Use this repo when you want to:

- load an ORE case from a single `ore.xml`
- inspect or validate the linked ORE inputs
- run pricing, XVA, and sensitivity workflows in native Python
- compare Python results to existing ORE output files
- benchmark or diagnose parity gaps between Python and ORE

There are two broad workflows:

- Python-first workflow:
  `ore.xml` -> `OreSnapshot` / `XVASnapshot` -> Python runtime -> reports
- ORE-integrated workflow:
  dataclasses / mapper -> ORE XML + ORE executable -> ORE output -> Python comparison / diagnostics

## Current Native Coverage

The maintained native path is strongest for:

- vanilla IRS and related mapped rate swaps
- generic fixed/floating rate swaps, including basis swaps
- generic CMS, CMS spread, and digital CMS spread swaps
- generic cap/floor trades, including in-arrears SOFR-style cap/floors
- European and Bermudan swaptions
- deterministic cashflows
- FX forwards
- inflation swaps and inflation cap/floor flows already supported by the runtime
- XVA exposure, CVA, DVA, FVA, and related parity-style reporting for supported products

Parity and workflow notes:

- Bermudan swaptions use the grid/backward pricing path for trade NPV checks, while XVA/exposure uses the LSMC path.
- The USD Bermudan case in the generated mixed book now picks up trade-specific GSR calibration from `pricingengine.xml` when swaption market quotes are available.
- The runtime handles both normal and lognormal swaption vol surfaces, including the USD Libor index support needed by the Bermudan calibration helper.

Not everything in ORE is natively priced here. Unsupported trades may still require ORE or SWIG-backed flows.

## Repository Layout

| Path | Purpose |
|---|---|
| [`src/pythonore/`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore) | Main package: IO, runtime, workflows, mapping, compute, apps |
| [`py_ore_tools/`](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools) | Compatibility package and legacy entrypoints |
| [`native_xva_interface/`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface) | Compatibility-facing interface layer |
| [`tests/`](/Users/gordonlee/Documents/PythonOreRunner/tests) | Test suite |
| [`scripts/`](/Users/gordonlee/Documents/PythonOreRunner/scripts) | Diagnostics, dumps, plots, ad hoc utilities |
| [`notebook_series/`](/Users/gordonlee/Documents/PythonOreRunner/notebook_series) | Notebook walkthroughs |
| [`parity_artifacts/`](/Users/gordonlee/Documents/PythonOreRunner/parity_artifacts) | Stored parity inputs and outputs |
| [`example_ore_snapshot.py`](/Users/gordonlee/Documents/PythonOreRunner/example_ore_snapshot.py) | Root-level snapshot quickstart |

## Installation

Requirements:

- Python 3.8+
- packages from [`requirements.txt`](/Users/gordonlee/Documents/PythonOreRunner/requirements.txt)

Install:

```bash
pip install -r requirements.txt
```

If you want to use the ORE executable for reference generation or calibration, also set:

```bash
export ENGINE_REPO_ROOT=/path/to/Engine
export ORE_EXE="$ENGINE_REPO_ROOT/build/.../App/ore"
```

## Quick Start

### 1. Preflight an ORE case

```bash
python -m py_ore_tools.ore_snapshot_cli \
  parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A/Input/ore.xml \
  --preflight
```

This checks that the case links resolve and reports whether the portfolio looks natively supported.

### 2. Run native Python XVA

```bash
python -m py_ore_tools.ore_snapshot_cli \
  parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A/Input/ore.xml \
  --xva \
  --paths 2000 \
  --rng ore_sobol_bridge \
  --lgm-param-source simulation_xml
```

This keeps the run on the Python side and avoids the expensive ORE calibration subprocess.

### 3. Run pricing only

```bash
python -m py_ore_tools.ore_snapshot_cli \
  parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A/Input/ore.xml \
  --price \
  --lgm-param-source simulation_xml
```

## Large Mixed-Book Benchmark

If you want a quick native stress test on a broad USD rates book, use the generated all-products case.

### 1. Generate a broad mixed rates book

This generator creates:

- IRS and amortising IRS
- basis swaps
- CMS, CMS spread, and digital CMS spread swaps
- cap/floor trades including in-arrears SOFR cap/floors
- European and Bermudan swaptions
- deterministic cashflows

Example:

```bash
python3 example_ore_snapshot_usd_all_rates_products.py \
  --count-per-type 134 \
  --case-root Examples/Generated/USD_AllRatesProductsSnapshot_134PerType \
  --overwrite \
  --no-run
```

That produces:

- `2278` total trades
- current native coverage on this generated book: `2278/2278`

### 2. Run a large native XVA benchmark

```bash
python -m py_ore_tools.ore_snapshot_cli \
  Examples/Generated/USD_AllRatesProductsSnapshot_134PerType/Input/ore.xml \
  --xva \
  --paths 2000 \
  --rng ore_sobol_bridge \
  --lgm-param-source simulation_xml
```

If you want the heavier stress run we used for hotspot work, increase to:

```bash
python -m py_ore_tools.ore_snapshot_cli \
  Examples/Generated/USD_AllRatesProductsSnapshot_134PerType/Input/ore.xml \
  --xva \
  --paths 10000 \
  --rng ore_sobol_bridge \
  --lgm-param-source simulation_xml
```

### 3. Preflight native support quickly

```bash
python -m py_ore_tools.ore_snapshot_cli \
  Examples/Generated/USD_AllRatesProductsSnapshot_134PerType/Input/ore.xml \
  --preflight
```

This is the fastest way to confirm whether a generated mixed book is fully native-supported before running the larger XVA job.

## LGM Parameter Source

The snapshot loader can populate LGM parameters from several sources.

- `auto`
  Default behavior. Prefer an existing `calibration.xml`, otherwise try runtime ORE calibration, otherwise fall back to `simulation.xml`.
- `calibration_xml`
  Use an existing `calibration.xml`. Do not run runtime ORE calibration.
- `simulation_xml`
  Parse model parameters directly from `simulation.xml`.
- `ore`
  Force runtime ORE calibration through the ORE executable.
- `provided`
  Use a supplied `LGMParams` object or parameter payload from Python code.

Important performance note:

- `auto` may invoke the ORE executable and is much slower
- `simulation_xml` and `provided` are the fast Python-only paths

On the flat EUR 5Y benchmark case, the current Python path was:

- `auto`: about `11.2s`
- `simulation_xml`: about `0.13s`
- `provided`: about `0.13s`

Recommended practical default for parity-quality Python runs:

- `--paths 10000`
- `--rng ore_sobol_bridge`
- `--xva-mode ore`
- `--lgm-param-source simulation_xml`

On the flat EUR 5Y benchmark, that setup currently gives:

- runtime about `0.49s`
- EPE `p95` under `1%`
- ENE `p95` about `2.2%`
- PFE `p95` about `3.7%`

That is a good default when you want fast Python-native runs with solid parity quality and without invoking the ORE executable.

## Single-Swap 99% Benchmark Evidence

For single vanilla swap migration work, a useful benchmark triangle is:

- ORE at modest path count
- Python Monte Carlo
- Python locked-coupon quantile proxy

The locked-coupon quantile proxy is a fast structural benchmark built from:

- the current LGM state quantile at each exposure date
- fixing-time quantiles for already-locked coupons

This matters when ORE high-path runs are too expensive. If Python Monte Carlo stays close to the proxy, then a modest Python-vs-ORE gap is more likely to be Monte Carlo / engine parity noise than a broken Python swap valuation.

At `99%` PFE and `10000` Python paths, the wider single-swap benchmark sweep currently looks like this:

- `18` successful cases
- median curve-level `PFE p95 rel`: `2.37%`
- mean curve-level `PFE p95 rel`: `2.44%`
- max curve-level `PFE p95 rel`: `4.09%`
- median peak-point relative error: `0.94%`
- max peak-point relative error: `2.45%`
- median one-year relative error: `1.93%`
- max one-year relative error: `3.53%`

Interpretation:

- this is strong evidence that the Python single-swap `99%` PFE shape is structurally correct at `10000` paths
- the proxy is tight enough to support migration discussions even when ORE cannot afford a very high-path confirmation run
- this evidence is strongest for single vanilla swaps and similar low-dimensional IR trades, not for large netted portfolios

## CLI

The main CLI is built in [`ore_snapshot_cli.py`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore/workflows/ore_snapshot_cli.py) and is exposed through the compatibility module:

```bash
python -m py_ore_tools.ore_snapshot_cli <path-to-ore.xml> [options]
```

### Main command shape

```bash
python -m py_ore_tools.ore_snapshot_cli \
  path/to/ore.xml \
  [--price] [--xva] [--sensi] [--preflight] \
  [--paths N] [--seed N] [--rng MODE] \
  [--xva-mode MODE] [--lgm-param-source SOURCE]
```

### What The CLI Produces

Depending on the flags, the CLI can:

- validate the case
- run native Python pricing
- run native Python XVA
- compare Python outputs to ORE output files already present in the case
- write artifacts under an output root
- generate report packs across multiple cases

## CLI Argument Reference

### Informational flags

- `-v`, `--version`
  Print version information.
- `-h`, `--hash`
  Print the build / hash style identifier used by the tool.
- `--help`
  Print help text.

### Mode selection

- `--price`
  Run pricing.
- `--xva`
  Run XVA.
- `--sensi`
  Run sensitivity comparison flow.
- `--preflight`
  Validate the case and native support without running a full pricing/XVA pass.
- `--pack`
  Run a batch/pack workflow over one or more cases.
- `--report-examples`
  Build example report bundles from configured example cases.

If you do not specify a mode, higher-level entrypoints may infer one, but in normal use you should pass the mode you want explicitly.

### Case and output selection

- positional `ore_xml`
  Path to the root ORE XML file for the case.
- `--case`
  Add a case name to a pack/report run. Can be passed multiple times.
- `--output-root PATH`
  Root folder for generated artifacts.
  Default:
  [`parity_artifacts/ore_snapshot_cli`](/Users/gordonlee/Documents/PythonOreRunner/parity_artifacts/ore_snapshot_cli)
- `--report-root PATH`
  Override report output location for report workflows.
- `--ore-output-only`
  Restrict some flows to ORE reference output handling rather than full Python recomputation where supported.

### Simulation controls

- `--paths N`
  Number of Monte Carlo paths to run in Python.
  If omitted, the loader uses the case sample count from `simulation.xml`.
- `--seed N`
  Random seed for the Python RNG mode.
  Default: `42`
- `--rng MODE`
  RNG construction for the Python simulation.

Supported values:

- `numpy`
  Plain NumPy RNG.
- `ore_parity`
  Mersenne Twister parity-style generator.
- `ore_parity_antithetic`
  Mersenne Twister antithetic variant.
- `ore_sobol`
  Sobol generator without the bridge path used in the main parity flow.
- `ore_sobol_bridge`
  Current best ORE-parity Sobol/Brownian-bridge path in Python.

Recommended default for current parity work:

- `--rng ore_sobol_bridge`

### XVA calculation controls

- `--xva-mode MODE`
  Choose how the Python path interprets XVA exposure handling.

Values:

- `classic`
  Simpler historical/native mode.
- `ore`
  ORE-style exposure and reporting semantics.

Recommended default:

- `--xva-mode ore`

### LGM parameter controls

- `--lgm-param-source SOURCE`
  Select where LGM parameters come from.

Values:

- `auto`
- `calibration_xml`
- `simulation_xml`
- `ore`
- `provided`

In CLI use, `provided` is mostly for programmatic entrypoints. The practical CLI choices are:

- `auto` for ORE-compatible default behavior
- `simulation_xml` for fast Python-only runs
- `calibration_xml` when you already have a trusted calibration file
- `ore` when you explicitly want runtime ORE calibration

### Pricing / parity controls

- `--anchor-t0-npv`
  Apply a float-spread anchoring adjustment so the Python t0 NPV matches ORE more closely in specific setups.
  Use carefully. It is not universally beneficial for dual-curve parity.

### Credit / XVA assumptions

- `--own-hazard FLOAT`
  Bank/own hazard rate used in DVA-style logic where relevant.
  Default: `0.01`
- `--own-recovery FLOAT`
  Bank/own recovery assumption.
  Default: `0.4`
- `--netting-set ID`
  Override or target a specific netting set in relevant workflows.

### Sensitivity controls

- `--sensi-metric NAME`
  Sensitivity metric to compare.
  Default: `CVA`
- `--top N`
  Limit top findings or rows shown in sensitivity/report outputs.
  Default: `10`

### Parity thresholds

These thresholds affect pass/fail style reporting in comparison workflows.

- `--max-npv-abs-diff FLOAT`
  Default: `1000.0`
- `--max-cva-rel FLOAT`
  Default: `0.05`
- `--max-dva-rel FLOAT`
  Default: `0.05`
- `--max-fba-rel FLOAT`
  Default: `0.05`
- `--max-fca-rel FLOAT`
  Default: `0.05`

These are relative tolerances except for `--max-npv-abs-diff`, which is absolute.

### Reporting / example workflow options

- `--report-workers N`
  Parallel worker count for report generation.
  Default: `12`
- `--report-refresh-every N`
  Refresh cadence for long-running report output.
  Default: `1`
- `--report-top-buckets N`
  Number of top report buckets to keep.
  Default: `10`
- `--example NAME`
  Run a benchmark/example preset.
  Choices:
  `lgm_torch`, `lgm_torch_swap`, `lgm_fx_hybrid`, `lgm_fx_forward`, `lgm_fx_portfolio`, `lgm_fx_portfolio_256`
- `--example-path-counts N [N ...]`
  Path counts for example benchmark runs.
  Default: `10000 50000`
- `--example-devices DEV [DEV ...]`
  Devices to use for example benchmark runs.
- `--tensor-backend`
  Tensor backend selection.
  Values:
  `auto`, `numpy`, `torch-cpu`, `torch-mps`
- `--example-repeats N`
  Default: `2`
- `--example-warmup N`
  Default: `1`
- `--example-trades N`
  Default: `64`

## Practical CLI Recipes

### Fast Python-only parity check

```bash
python -m py_ore_tools.ore_snapshot_cli \
  parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A/Input/ore.xml \
  --xva \
  --paths 2000 \
  --rng ore_sobol_bridge \
  --xva-mode ore \
  --lgm-param-source simulation_xml
```

### Use an existing calibration file, but do not spawn ORE

```bash
python -m py_ore_tools.ore_snapshot_cli \
  path/to/ore.xml \
  --xva \
  --rng ore_sobol_bridge \
  --lgm-param-source calibration_xml
```

### Force ORE runtime calibration

```bash
python -m py_ore_tools.ore_snapshot_cli \
  path/to/ore.xml \
  --xva \
  --lgm-param-source ore
```

### Pricing-only smoke test

```bash
python -m py_ore_tools.ore_snapshot_cli \
  path/to/ore.xml \
  --price \
  --lgm-param-source simulation_xml
```

## Programmatic Use

Load an ORE case directly:

```python
from pythonore.io.ore_snapshot import load_from_ore_xml

snap = load_from_ore_xml(
    "path/to/ore.xml",
    lgm_param_source="simulation_xml",
)
model = snap.build_model()
```

Run a Python snapshot case directly:

```python
from pathlib import Path
from pythonore.workflows.ore_snapshot_cli import _compute_snapshot_case

result = _compute_snapshot_case(
    Path("path/to/ore.xml"),
    paths=2000,
    seed=42,
    rng_mode="ore_sobol_bridge",
    anchor_t0_npv=False,
    own_hazard=0.01,
    own_recovery=0.4,
    xva_mode="ore",
    lgm_param_source="simulation_xml",
)
```

Use pre-supplied params:

```python
from pythonore.io.ore_snapshot import load_from_ore_xml

base = load_from_ore_xml("path/to/ore.xml", lgm_param_source="simulation_xml")

snap = load_from_ore_xml(
    "path/to/ore.xml",
    lgm_param_source="provided",
    provided_lgm_params=base.lgm_params,
)
```

## Tests

Run the main tests:

```bash
python -m pytest tests -q
```

Refresh or compare curated example baselines:

```bash
python -m pythonore.apps.examples_regression refresh
python -m pythonore.apps.examples_regression compare
```

## Further Reading

- [`example_ore_snapshot.py`](/Users/gordonlee/Documents/PythonOreRunner/example_ore_snapshot.py)
- [`scripts/README.md`](/Users/gordonlee/Documents/PythonOreRunner/scripts/README.md)
- [`notebook_series/05_1_python_only_workflow.ipynb`](/Users/gordonlee/Documents/PythonOreRunner/notebook_series/05_1_python_only_workflow.ipynb)

## Notes

- ORE itself is a separate project: [Open Risk Engine](https://github.com/OpenSourceRisk/Engine)
- This repo can integrate with ORE, but the fast Python-native path does not require invoking the ORE executable when you use `--lgm-param-source simulation_xml` or supplied params
