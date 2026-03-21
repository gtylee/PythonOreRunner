# PythonOreRunner

Python utilities and a native XVA interface for [Open Risk Engine (ORE)](https://github.com/OpenSourceRisk/Engine). Use this to run ORE configs from Python, compare ORE vs Python LGM results, and drive XVA workflows.

## Python-first native workflow

The maintained Python-native path is:

`XVASnapshot` -> `PythonLgmAdapter(fallback_to_swig=False)` -> Python pricing/XVA reports

This path runs entirely in native Python for the product families supported below. Use it when you want an in-memory workflow with no ORE binary and no SWIG dependency.

The quickest starting points are:

- [`notebook_series/05_1_python_only_workflow.ipynb`](/Users/gordonlee/Documents/PythonOreRunner/notebook_series/05_1_python_only_workflow.ipynb) for the canonical in-memory walkthrough
- [`example_ore_snapshot.py`](/Users/gordonlee/Documents/PythonOreRunner/example_ore_snapshot.py) for the ORE-XML-to-Python-snapshot bridge
- `python -m pythonore.apps.ore_snapshot_cli ... --xva` for the Python-first CLI flow

### Native capability matrix

| Area | Native Python | Fallback-only | Not supported |
|------|---------------|---------------|---------------|
| Rates | IRS, generic rate swaps already mapped to native legs, caps/floors, Bermudan swaption flow, rate futures | Some remaining non-native swap families when loaded as unsupported generic trades | Full ORE rates surface parity is not claimed |
| FX | FX forwards, XCCY float-float helper flow, vanilla FX option helper pricing | Products that route through unsupported trade types in the runtime | Smile/barrier/exotic support |
| Inflation | Inflation swaps and inflation cap/floor runtime flow | None intended for the current native set | Broader inflation product coverage |
| XVA metrics | CVA, DVA, FVA, MVA via Python DIM feeder | Any metric only available from ORE reports on unsupported trades | KVA |
| Equity / Commodity | Dataclasses only for schema / interop | ORE SWIG may price some loaded cases if available | Native Python pricing is not implemented |

Equity and commodity dataclasses remain in the model for compatibility and mapping, but they should be treated as schema/interop types, not native pricers.

Use it for fast prototyping, regression tests, teaching, or as the Python leg in ORE-vs-Python parity runs. The maintained implementation now lives under `src/pythonore/`; `py_ore_tools/` and `native_xva_interface/` remain compatibility-facing packages and script entrypoints.

## Layout

| Path | Description |
|------|-------------|
| `src/pythonore/` | Canonical Python package layout: shared domain types, IO, mapping, runtime, workflows, and app entrypoints |
| `py_ore_tools/` | Compatibility package for the standalone Python LGM, ORE runner, and thin legacy entrypoints |
| `native_xva_interface/` | Compatibility package for dataclass loaders, ORE-SWIG and Python LGM adapters; see [native_xva_interface/README.md](native_xva_interface/README.md) |
| `legacy/` | Relocated legacy demos, benchmarks, notebooks, docs, writeups, and former root scripts from the pre-`src` package layout |
| `example_ore_snapshot.py` | Single root quickstart demo for the canonical ORE snapshot flow |
| `scripts/` | Ad hoc checks, diagnostics, dumps, plots, and parity comparison utilities; see [scripts/README.md](/Users/gordonlee/Documents/PythonOreRunner/scripts/README.md) |
| `notebook_series/legacy/` | Older standalone demo notebooks retained for reference |
| `docs/` | Project notes and longer-form writeups that do not belong at the root |
| `tests/` | Unit tests for `py_ore_tools` |
| `parity_artifacts/` | Generated benchmark/parity outputs (optional; can be recreated) |
| `regression_artifacts/examples_python/` | Central Python-first regression baselines generated from curated cases under `Examples/` |

## Architecture

The maintained Python flows now separate into two explicit paths:

- Python-first: dataclasses -> loader/programmatic build -> Python compute -> snapshot/report artifacts
- ORE integration: dataclasses -> XML mapper -> oreapp / SWIG runtime -> snapshot-compatible artifacts

The canonical import surface for new code lives under [`src/pythonore/`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore), while [`py_ore_tools/`](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools) and [`native_xva_interface/`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface) remain compatibility-facing packages.

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt). Core: `numpy`, `pandas`; examples/plots: `matplotlib`, `networkx`.

```bash
pip install -r requirements.txt
```

## Quick Start

This repo includes an ORE-style Python CLI flow: point it at an ORE case directory with `Input/` and `Output/`, run the case, then inspect the same style of artifacts you would expect from an ORE run.

If you have an Engine checkout available:

```bash
export ENGINE_REPO_ROOT=/path/to/Engine
export ORE_EXE="$ENGINE_REPO_ROOT/build/apple-make-relwithdebinfo-arm64/App/ore"

cd Examples/XvaRisk
"$ORE_EXE" Input/ore_stress_classic.xml
```

That runs a standard vendored ORE case against the stress portfolio in [`Examples/XvaRisk/Input/portfolio_stress.xml`](/Users/gordonlee/Documents/PythonOreRunner/Examples/XvaRisk/Input/portfolio_stress.xml). Results are written under [`Examples/XvaRisk/Output/stress/classic/`](/Users/gordonlee/Documents/PythonOreRunner/Examples/XvaRisk/Output/stress/classic), typically including `npv.csv`, `flows.csv`, `log.txt`, and the configured stress / XVA reports for that case.

For a simpler single-`ore.xml` Python-side parity run without calling the ORE binary:

```bash
python -m py_ore_tools.ore_snapshot_cli \
  Examples/Exposure/Input/ore_measure_lgm.xml \
  --xva \
  --paths 10000 \
  --rng ore_parity
```

That loads the ORE-style XML inputs, runs the standalone Python LGM parity/XVA path, and prints the pricing, XVA, and parity diagnostics directly in the terminal.

## Running tests

From the project root:

```bash
# py_ore_tools tests
python -m pytest tests/ -q

# native_xva_interface tests (add project root to PYTHONPATH)
PYTHONPATH=. python -m pytest native_xva_interface/tests/ -q
```

To refresh or compare the curated Python-first example baselines:

```bash
python -m pythonore.apps.examples_regression refresh
python -m pythonore.apps.examples_regression compare
```

## Examples

- **`example_basic.py`** / **`example.py`** – Run ORE from Python and inspect NPV/XVA. Expects an ORE executable and an ORE input folder (e.g. from an ORE repo).
- **`example_ore_snapshot.py`** – Load ORE XML into the canonical Python snapshot object and bridge into the Python-first workflow.
- **`example_systemic.py`** – Uses `OreBasic` with network/graph dependencies.

Set `ORE_EXE` to the path of the `ore` binary. Set `ORE_EXAMPLE_DIR` to an ORE example folder (for example `path/to/ORE/Examples/Legacy/Example_1`). If unset, examples will auto-discover `ENGINE_REPO_ROOT` and default to `Examples/Legacy/Example_1`.

For a standalone checkout, set `ENGINE_REPO_ROOT` to your ORE `Engine` repo. The benchmark and example scripts will use that for `Examples/` inputs and the default `ore` binary, while writing any generated benchmark output under this repo's local `parity_artifacts/`.

## Benchmarks

Benchmark implementations live under `src/pythonore/benchmarks/`; historical benchmark content now lives under [`legacy/py_ore_tools/benchmarks/`](/Users/gordonlee/Documents/PythonOreRunner/legacy/py_ore_tools/benchmarks), with `py_ore_tools.benchmarks` left as an import shim. Typical usage from this repo root:

```bash
export ENGINE_REPO_ROOT=/path/to/Engine

python py_ore_tools/benchmarks/benchmark_discount_factor_extractor.py --runs 10
python py_ore_tools/benchmarks/benchmark_ore_fx_forwards.py
python py_ore_tools/benchmarks/benchmark_ore_fx_forwards_xva.py --samples 2000
python py_ore_tools/benchmarks/benchmark_lgm_ore_multiccy.py --max-cases 2
python py_ore_tools/benchmarks/benchmark_lgm_fx_hybrid_ore.py --max-cases 2
python py_ore_tools/benchmarks/benchmark_ore_ir_options.py
```

These scripts read market/example inputs from `ENGINE_REPO_ROOT/Examples/...` and emit local results under `parity_artifacts/`.

See [`py_ore_tools/benchmarks/README.md`](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/benchmarks/README.md) for per-script prerequisites, inputs, and output locations.

## Diagnostics And Utilities

Ad hoc root scripts now live under [`scripts/`](/Users/gordonlee/Documents/PythonOreRunner/scripts), leaving `tests/` and `py_ore_tools/benchmarks/` as the maintained automated surfaces.

## Notebook Series

The main notebook walkthrough lives under [`notebook_series/`](/Users/gordonlee/Documents/PythonOreRunner/notebook_series). The helpers prefer local vendored `Examples/` and local `parity_artifacts/`, and fall back to `ENGINE_REPO_ROOT` when a live ORE binary or non-vendored Engine inputs are required.

For the native Python story, start with [`notebook_series/05_1_python_only_workflow.ipynb`](/Users/gordonlee/Documents/PythonOreRunner/notebook_series/05_1_python_only_workflow.ipynb). That is the canonical end-to-end in-memory flow: programmatic snapshot build, validation, support preflight, native session run, and incremental updates.

Older one-off demo notebooks now live under [`notebook_series/legacy/`](/Users/gordonlee/Documents/PythonOreRunner/notebook_series/legacy).

To regenerate the notebooks after editing the builder:

```bash
python3 notebook_series/build_series.py
```

## License and ORE

ORE itself is separate; build and install it from the [ORE repository](https://github.com/OpenSourceRisk/Engine). This project is an add-on layer and does not replace ORE.
