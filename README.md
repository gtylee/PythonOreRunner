# PythonOreRunner

Python utilities and a native XVA interface for [Open Risk Engine (ORE)](https://github.com/OpenSourceRisk/Engine). Use this to run ORE configs from Python, compare ORE vs Python LGM results, and drive XVA workflows.

## Standalone Python LGM model

**`py_ore_tools`** includes a **standalone Python LGM (Linear Gaussian Model)** implementation that consumes **ORE-style inputs** but runs entirely in **native Python**—no ORE binary or SWIG build required. Same curves, same conventions, same exposure/XVA workflow, so you can iterate quickly and compare directly to ORE.

**Supported products and features:**

- **IRS** – single- and multi-currency interest rate swaps, dual-curve discount/forward, ORE parity-oriented conventions  
- **FX forwards** – multi-currency LGM–FX hybrid, FX forwards and XCCY float-float  
- **Caps & floors** – pathwise cap/floor valuation and exposure  
- **Bermudan swaptions** – exercise-date handling and pathwise NPV  
- **Multi-currency** – LGM–FX hybrid with correlated IR and FX, aggregate exposure and CVA/FVA-style XVA terms  

Use it for fast prototyping, regression tests, teaching, or as the Python leg in ORE-vs-Python parity runs. See `py_ore_tools/lgm.py`, `lgm_fx_hybrid.py`, `lgm_ir_options.py`, `irs_xva_utils.py`, and `lgm_fx_xva_utils.py` for the core logic; demos and benchmarks under `py_ore_tools/demos/` and `py_ore_tools/benchmarks/`.

## Layout

| Path | Description |
|------|-------------|
| `py_ore_tools/` | Standalone Python LGM (IRS, FX fwd, cap/floor, Bermudans, multi-ccy), ORE runner, XVA helpers, benchmarks, demos |
| `native_xva_interface/` | Python dataclass loaders, ORE-SWIG and Python LGM adapters; see [native_xva_interface/README.md](native_xva_interface/README.md) |
| `example*.py` | Example scripts using `OreBasic` and snapshot tools |
| `tests/` | Unit tests for `py_ore_tools` |
| `parity_artifacts/` | Generated benchmark/parity outputs (optional; can be recreated) |

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt). Core: `numpy`, `pandas`; examples/plots: `matplotlib`, `networkx`.

```bash
pip install -r requirements.txt
```

## Running tests

From the project root:

```bash
# py_ore_tools tests
python -m pytest tests/ -q

# native_xva_interface tests (add project root to PYTHONPATH)
PYTHONPATH=. python -m pytest native_xva_interface/tests/ -q
```

## Examples

- **`example_basic.py`** / **`example.py`** – Run ORE from Python and inspect NPV/XVA. Expects an ORE executable and an ORE input folder (e.g. from an ORE repo).
- **`example_ore_snapshot.py`** – Load ORE XML into a snapshot and run the **standalone Python LGM** (no ORE binary) for exposure/XVA.
- **`example_systemic.py`** – Uses `OreBasic` with network/graph dependencies.

Set `ORE_EXE` to the path of the `ore` binary. Set `ORE_EXAMPLE_DIR` to an ORE example folder (for example `path/to/ORE/Examples/Legacy/Example_1`). If unset, examples will auto-discover `ENGINE_REPO_ROOT` and default to `Examples/Legacy/Example_1`.

For a standalone checkout, set `ENGINE_REPO_ROOT` to your ORE `Engine` repo. The benchmark and example scripts will use that for `Examples/` inputs and the default `ore` binary, while writing any generated benchmark output under this repo's local `parity_artifacts/`.

## Benchmarks

Benchmark entrypoints live under `py_ore_tools/benchmarks/`. Typical usage from this repo root:

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

## License and ORE

ORE itself is separate; build and install it from the [ORE repository](https://github.com/OpenSourceRisk/Engine). This project is an add-on layer and does not replace ORE.
