# Native XVA Interface

`native_xva_interface` is the typed Python API layer that sits between ORE-style inputs and two runtime paths:

- `ORESwigAdapter` for ORE-SWIG-backed execution
- `PythonLgmAdapter` for the lighter Python LGM parity path

It is not the main repo CLI by itself. The user-facing ORE-style CLI in this repo is:

```bash
python -m py_ore_tools.ore_snapshot_cli path/to/ore.xml --price --xva
```

Use `native_xva_interface` when you want to load, inspect, modify, merge, or run snapshots as Python objects.

## What It Provides

- typed dataclasses for market, fixings, portfolio, netting, collateral, runtime config, and results
- `XVALoader.from_files(...)` for standard ORE `Input/` directories
- `merge_snapshots(...)` to overlay programmatic changes on top of file-loaded cases
- `map_snapshot(...)` to produce runtime-ready XML buffers
- runtime adapters and `XVAEngine`
- preset / template helpers for known-good stress-classic configurations

## File-Based Flow

Load a normal ORE input bundle and run it through the native interface:

```python
from native_xva_interface import ORESwigAdapter, XVAEngine, XVALoader

snapshot = XVALoader.from_files("Examples/XvaRisk/Input", ore_file="ore_stress_classic.xml")
result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
print(result.xva_by_metric)
```

This keeps the ORE-style file layout as the source of truth and exposes it as Python dataclasses.

## Buffer-Based Flow

The important internal contract is `snapshot.config.xml_buffers`.

`XVALoader.from_files(...)` already loads known XML files such as:

- `portfolio.xml`
- `netting.xml`
- `pricingengine.xml`
- `todaysmarket.xml`
- `curveconfig.xml`
- `conventions.xml`
- `simulation.xml`
- `creditsimulation.xml` when available

You can inspect or override those buffers directly:

```python
from dataclasses import replace

from native_xva_interface import ORESwigAdapter, XVAEngine, XVALoader

base = XVALoader.from_files("Examples/XvaRisk/Input", ore_file="ore_stress_classic.xml")

xml_buffers = dict(base.config.xml_buffers)
xml_buffers["simulation.xml"] = xml_buffers["simulation.xml"].replace("<Samples>64</Samples>", "<Samples>256</Samples>")

snapshot = replace(base, config=replace(base.config, xml_buffers=xml_buffers, num_paths=256))
result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
```

You can also build known-good embedded XML buffers without reading files from disk:

```python
from native_xva_interface import stress_classic_xml_buffers

xml_buffers = stress_classic_xml_buffers(num_paths=128)
print(sorted(xml_buffers))
```

That is the right pattern when you want an in-memory runtime payload or need to inject exact XML blocks into a programmatic snapshot.

## Command-Line Entry Points

### Main ORE-style CLI

For front-door parity / reporting runs, use the repo CLI:

```bash
python -m py_ore_tools.ore_snapshot_cli Examples/Exposure/Input/ore_measure_lgm.xml --xva --paths 10000
```

### Native Interface Benchmarks / Demos

`native_xva_interface` does ship a few direct executable scripts, but they are targeted demos or benchmarks, not the main user CLI surface. Examples:

```bash
python native_xva_interface/demos/run_large_fx_universe_benchmark.py --help
python native_xva_interface/demos/run_quick_swig_100_paths.py
python native_xva_interface/demos/run_pure_programmatic_swig.py
```

## Mixed File + Programmatic Overrides

If you want to keep the ORE input bundle but override parts of the snapshot in Python, use `from_mixed(...)` or `merge_snapshots(...)`.

```python
from native_xva_interface import IRS, Portfolio, Trade, XVASnapshot, XVALoader, merge_snapshots

base = XVALoader.from_files("Examples/XvaRisk/Input", ore_file="ore_stress_classic.xml")

override = XVASnapshot(
    market=base.market,
    fixings=base.fixings,
    portfolio=Portfolio(
        trades=(
            Trade(
                trade_id="IRS_NEW",
                counterparty="CPTY_A",
                netting_set="CPTY_A",
                trade_type="Swap",
                product=IRS(ccy="EUR", notional=5_000_000, fixed_rate=0.02, maturity_years=5.0),
            ),
        )
    ),
    netting=base.netting,
    collateral=base.collateral,
    config=base.config,
)

snapshot = merge_snapshots(base, override, on_conflict="override")
```

## Related Files

- [`loader.py`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/loader.py)
- [`mapper.py`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/mapper.py)
- [`runtime.py`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/runtime.py)
- [`presets.py`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/presets.py)
- [`stress_classic_templates.py`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/stress_classic_templates.py)
- [`docs/NATIVE_XVA_STATUS.md`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/docs/NATIVE_XVA_STATUS.md)

## Tests

From repo root:

```bash
PYTHONPATH=. python3 -m pytest native_xva_interface/tests -q
```

## Demos And Notebooks

- Demos: [`native_xva_interface/demos/`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/demos)
- Notebooks: [`native_xva_interface/notebooks/`](/Users/gordonlee/Documents/PythonOreRunner/native_xva_interface/notebooks)
