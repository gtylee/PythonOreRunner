# PythonOreRunner

Python utilities and a native XVA interface for [Open Risk Engine (ORE)](https://github.com/OpenSourceRisk/Engine). Use this to run ORE configs from Python, compare ORE vs Python LGM results, and drive XVA workflows.

## Layout

| Path | Description |
|------|-------------|
| `py_ore_tools/` | ORE runner, LGM/FX models, IRS and XVA helpers, benchmarks, demos |
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
- **`example_ore_snapshot.py`** – Load ORE XML into a snapshot and run Python LGM exposure/XVA.
- **`example_systemic.py`** – Uses `OreBasic` with network/graph dependencies.

Set `ORE_EXE` to the path of the `ore` binary. Set `ORE_EXAMPLE_DIR` to an ORE Input folder (e.g. `path/to/ORE/Examples/Example_7`). If unset, examples assume this repo lives under an ORE tree at `../..` with `Examples/Example_7` and `build/.../App/ore`.

## License and ORE

ORE itself is separate; build and install it from the [ORE repository](https://github.com/OpenSourceRisk/Engine). This project is an add-on layer and does not replace ORE.
