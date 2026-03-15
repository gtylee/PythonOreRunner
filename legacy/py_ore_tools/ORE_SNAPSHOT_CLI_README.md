# ORE Snapshot CLI

This document describes the current implemented state of the ORE-compatible Python CLI in:

- [ore_snapshot_cli.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/ore_snapshot_cli.py)

It covers:

- the `ore.exe`-shaped command-line interface
- what the CLI actually runs
- what artifacts it writes
- how ORE-format output parity is handled
- the pure-Python in-memory API
- the current limits of the implementation

## Purpose

The CLI exists to run front-to-back replication workflows around `py_ore_tools.ore_snapshot` while keeping the top-level invocation shape familiar to ORE users.

It is designed for:

- pricing parity
- XVA parity
- sensitivity comparison
- pack/regression runs
- ORE-style terminal output
- ORE-style report emission
- in-memory object-based usage from Python

It is not a wrapper that invokes the native `ore` binary. The active engine is Python. Native ORE outputs are used as stored references for comparison and parity diagnostics.

## Top-Level CLI Shape

The CLI intentionally keeps the same entry shape as `ore.exe`:

```bash
python -m py_ore_tools.ore_snapshot_cli path/to/ore.xml
python -m py_ore_tools.ore_snapshot_cli -v
python -m py_ore_tools.ore_snapshot_cli --hash
```

Supported compatibility flags:

- positional `ore.xml`
- `-v` / `--version`
- `-h` / `--hash`

Python-only extensions:

- `--price`
- `--xva`
- `--sensi`
- `--pack`
- `--case`
- `--output-root`
- `--ore-output-only`
- `--paths`
- `--seed`
- `--rng`
- `--xva-mode`
- `--anchor-t0-npv`
- `--own-hazard`
- `--own-recovery`
- `--netting-set`
- `--sensi-metric`
- `--top`
- `--max-npv-abs-diff`
- `--max-cva-rel`
- `--max-dva-rel`
- `--max-fba-rel`
- `--max-fca-rel`

## Default Behavior

When you run:

```bash
python -m py_ore_tools.ore_snapshot_cli path/to/ore.xml
```

the CLI:

1. reads `ore.xml`
2. infers active analytics from the ORE config
3. runs the supported Python pricing/XVA/sensitivity workflow
4. compares Python outputs against stored ORE outputs where applicable
5. writes artifacts
6. prints an ORE-style terminal summary

Mode inference is based on the active analytic blocks in `ore.xml`. Explicit flags such as `--price` or `--xva` override inference.

## Implemented Modes

### Pricing

Pricing mode computes a Python t0 NPV and compares against ORE `npv.csv`.

Implemented behavior:

- uses the snapshot/minimal pricing loaders in `py_ore_tools`
- loads curves from ORE `curves.csv`
- loads trade structure from `flows.csv` when available
- falls back to `portfolio.xml` leg extraction if `flows.csv` is missing
- writes an ORE-format `npv.csv`

Price-only mode has been relaxed so it does not require full XVA outputs. It can run from a partial ORE output set as long as the pricing-critical inputs exist.

### XVA

XVA mode computes Python exposure and XVA metrics and compares them against stored ORE outputs such as:

- `xva.csv`
- `exposure_trade_<trade_id>.csv`
- `exposure_nettingset_<netting_set_id>.csv`

Implemented behavior:

- ORE-parity RNG mode is available via `--rng ore_parity`
- numeraire-deflated XVA mode is available via `--xva-mode ore`
- fallback own-credit settings are supported if not provided by the case
- Basel EPE / EEPE are reported using the same 1Y time-weighted convention as the ORE aggregate report

### Sensitivity

Sensitivity mode uses the existing comparator path:

- `native_xva_interface.OreSnapshotPythonLgmSensitivityComparator`

Implemented behavior:

- supports `--sensi`
- supports `--sensi-metric`
- supports `--netting-set`
- supports `--top`
- adds sensitivity results into the case summary and pack summary

### Pack

Pack mode runs the same case logic across multiple ORE case XMLs and writes pack-level outputs.

Implemented behavior:

- `--pack`
- optional repeated `--case`
- per-case results under a pack root
- aggregate pack outputs:
  - `summary.json`
  - `results.csv`
  - `summary.md`

## ORE-Style Terminal Output

The CLI now emits an ORE-like console flow, including:

- `Loading inputs ... OK`
- `Requested analytics ...`
- `Pricing: Build ... OK`
- `XVA: Build ... OK`
- cube/progress output
- `XVA: Aggregation ... OK`
- `Writing reports... OK`
- `Writing cubes... OK`
- `run time: ... sec`
- `ORE done.`

After the ORE-style footer, the CLI also prints a concise Python parity summary with:

- `ore_xml`
- `trade_id`
- active modes
- `pass_all`
- pricing summary
- XVA summary
- sensitivity summary when present

This is intentionally close to native ORE output, but it is not byte-identical.

## ORE-Format Artifact Output

The CLI writes ORE-style reports into the artifact case directory.

Implemented ORE-format generation / copying includes:

- `npv.csv`
- `xva.csv`
- `exposure_trade_<trade_id>.csv`
- `exposure_nettingset_<netting_set_id>.csv`

The CLI also mirrors the remaining native ORE files from the case `Output/` directory when present, including files such as:

- `flows.csv`
- `curves.csv`
- `cube.csv.gz`
- `colva_nettingset_*.csv`
- `cva_sensitivity_nettingset_*.csv`
- `scenariodata.csv.gz`
- logs and runtime files

This means the Python artifact directory can be made structurally very close to a native ORE `Output/` directory.

### `--ore-output-only`

If you use:

```bash
python -m py_ore_tools.ore_snapshot_cli path/to/ore.xml --ore-output-only
```

the CLI suppresses the Python-specific extra files and writes only the ORE-style output set.

Without `--ore-output-only`, the CLI additionally writes:

- `summary.json`
- `comparison.csv`
- `input_validation.csv`
- `report.md`

## Python-Specific Artifact Semantics

### `summary.json`

This is the primary machine-readable case summary and includes:

- case metadata
- selected modes
- pricing comparison
- XVA comparison
- parity completeness
- diagnostics
- input validation
- pass/fail flags

### `comparison.csv`

This is a flattened view of the case summary for quick spreadsheet inspection. It includes rows from:

- `pricing`
- `xva`
- `diagnostics`
- `sensitivity` except top-comparison detail rows

### `input_validation.csv`

This is derived from the ORE snapshot input validation routines and is intended to expose broken links and unsupported/missing case elements early.

### `report.md`

This is a readable markdown case report including pricing, XVA, parity, sensitivity, and input validation sections.

## Current Diagnostics and Hardening

The CLI includes additional diagnostics that were added during parity debugging. These are available in `summary.json` under `diagnostics`.

Implemented diagnostics include:

- `ore_samples`
- `python_paths`
- `sample_count_mismatch`
- `float_fixing_source`
- `float_index_day_counter`
- `float_spread_abs_median`
- `float_spread_abs_max`
- `warnings`

These were added specifically to catch and explain the kinds of issues that previously caused misleading parity gaps:

- comparing Python `5000` paths against stored ORE `500` samples
- using coupon dates instead of fixing dates
- mixing coupon accrual and floating-index accrual conventions

## Parity Rules Learned and Implemented

The current implementation includes several parity fixes that matter materially:

- `fixingDate` from `flows.csv` is now used when present
- index accrual and coupon accrual are handled separately
- `A360` support was added to the relevant year-fraction helper path
- `Maturity` and `MaturityTime` in generated `npv.csv` are sourced correctly
- `BaselEPE` / `BaselEEPE` in generated `xva.csv` follow the 1Y time-weighted exposure convention instead of `max(EPE)`
- pass/fail gating only applies to XVA metrics actually requested by the case

These are important because they converted several earlier apparent “bugs” into either correct metadata behavior or normal Monte Carlo variation.

## In-Memory Python API

In addition to the CLI, the same logic is exposed as a pure-Python, object-returning API.

Implemented exports from [__init__.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/__init__.py):

- `BufferCaseInputs`
- `PurePythonRunOptions`
- `PurePythonCaseResult`
- `OreSnapshotApp`
- `run_case_from_buffers`

### `BufferCaseInputs`

This represents a virtual ORE case in memory.

Current fields:

- `input_files: dict[str, str]`
- `output_files: dict[str, str]`
- `ore_xml_name: str = "ore.xml"`

Important current meaning:

- `input_files` means virtual `Input/` files
- `output_files` means virtual existing ORE `Output/` files supplied as reference data

`output_files` does not currently mean “which files should be returned”.

### `PurePythonRunOptions`

Current fields include:

- `engine`
- `price`
- `xva`
- `sensi`
- `paths`
- `seed`
- `rng`
- `xva_mode`
- tolerance settings
- `ore_output_only`

### Engine Modes

The in-memory API now supports explicit engine selection via `PurePythonRunOptions.engine`.

Implemented values:

- `compare`
- `python`
- `ore`

#### `engine="compare"`

This is the default.

Behavior:

- runs the Python engine
- compares against supplied ORE output buffers
- returns comparison rows
- returns report markdown
- returns generated/mirrored ORE-style files

#### `engine="python"`

Behavior:

- runs the Python engine
- returns Python-side objects only
- strips ORE comparison fields from the summary sections
- does not return comparison rows
- still returns generated ORE-style outputs in-memory

This mode is useful if the caller wants a pure Python object result without treating stored ORE outputs as the primary output contract.

#### `engine="ore"`

Behavior:

- does not run the Python engine
- parses supplied ORE output buffers into an object summary
- returns ORE reference objects
- mirrors the supplied ORE output files in-memory

This is useful if the caller wants a consistent object model for the ORE side only.

### `run_case_from_buffers(...)`

This is the direct function API for in-memory execution.

Example:

```python
from pathlib import Path
from py_ore_tools import BufferCaseInputs, PurePythonRunOptions, run_case_from_buffers

case_root = Path(
    "/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A"
)

input_files = {
    path.name: path.read_text(encoding="utf-8")
    for path in (case_root / "Input").iterdir()
    if path.is_file()
}
output_files = {
    path.name: path.read_text(encoding="utf-8")
    for path in (case_root / "Output").iterdir()
    if path.is_file() and path.suffix in {".xml", ".csv", ".txt"}
}

result = run_case_from_buffers(
    BufferCaseInputs(
        input_files=input_files,
        output_files=output_files,
    ),
    PurePythonRunOptions(
        engine="compare",
        price=True,
        xva=True,
        paths=500,
        seed=42,
        rng="ore_parity",
    ),
)

print(result.summary["trade_id"])
print(result.summary["pricing"])
print(result.summary["xva"])
print(result.report_markdown[:200])
```

### `OreSnapshotApp`

This is the wrapper class that gives the API a more `OreApp`-style feel.

Implemented constructors:

- `OreSnapshotApp.from_strings(...)`
- `OreSnapshotApp.from_buffers(...)`

Implemented method:

- `.run()`

Example:

```python
from py_ore_tools import OreSnapshotApp, PurePythonRunOptions

app = OreSnapshotApp.from_strings(
    input_files=input_files,
    output_files=output_files,
    options=PurePythonRunOptions(
        engine="python",
        price=True,
        xva=True,
        paths=500,
    ),
)

result = app.run()
print(result.summary["pricing"]["py_t0_npv"])
```

## Return Object Shape

The in-memory API returns `PurePythonCaseResult`.

Current fields:

- `summary`
- `comparison_rows`
- `input_validation_rows`
- `report_markdown`
- `ore_output_files`

### `summary`

This is the structured case result object. Its content depends on the engine mode.

### `comparison_rows`

Present in `compare` mode. Empty in `python` and `ore` modes.

### `input_validation_rows`

Always available as a row-based validation view.

### `report_markdown`

Present in all modes. In `compare` mode it is read from the written artifact; in non-compare modes it is rendered directly from the summary.

### `ore_output_files`

This contains the returned ORE-style output files as strings, keyed by filename.

Current behavior:

- in `compare` mode it includes the generated and mirrored ORE-style files
- in `python` mode it includes generated ORE-style files
- in `ore` mode it includes mirrored ORE reference files

## What Is Not Yet Cleanly Named

The current field name `BufferCaseInputs.output_files` is somewhat misleading.

What it actually means today:

- existing ORE `Output/` files supplied as reference input

What it does not mean:

- requested output file names
- an output destination directory

This should likely be renamed in a future cleanup to something like:

- `ore_output_files`
- `reference_output_files`

with a separate optional field for requested returned files if selective emission becomes important.

## Current Limitations

### The CLI does not invoke native ORE

The CLI compares against existing ORE artifacts. It does not call the C++ `ore` executable.

### The in-memory API is still parity-oriented

Even `engine="python"` is built on the parity workflow. That means some modes still expect ORE-style case structure and, depending on the path, may still benefit from supplied ORE outputs.

This is intentionally pragmatic: the object API reuses the same validated path as the CLI rather than creating a second independent execution stack.

### Sensitivity still depends on the comparator path

Sensitivity runs are routed through the existing comparator integration rather than a standalone pure-Python sensitivity engine.

### Terminal parity is visual, not literal

The terminal output is intentionally close to ORE, but it is not intended to be byte-for-byte identical to native ORE output.

## Validation Status

The implemented CLI and object API are covered by:

- [test_ore_snapshot_cli.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/tests/test_ore_snapshot_cli.py)

Current tested coverage includes:

- version/hash compatibility flags
- normal CLI invocation shape
- mode inference
- per-case artifact generation
- pack artifact generation
- `--ore-output-only`
- price-only relaxed loading
- sensitivity path
- parity diagnostics
- XVA metric gating
- in-memory compare mode
- in-memory Python mode
- in-memory ORE mode
- `OreSnapshotApp.from_strings(...).run()`

## Recommended Usage Patterns

### For command-line parity work

Use:

```bash
python -m py_ore_tools.ore_snapshot_cli path/to/ore.xml --price --xva
```

### For regression packs

Use:

```bash
python -m py_ore_tools.ore_snapshot_cli path/to/ore.xml --pack --price --xva
```

### For notebook or service integration

Use:

- `run_case_from_buffers(...)` for direct functional style
- `OreSnapshotApp.from_strings(...).run()` for app-style orchestration

### For ORE-like drop-in output directories

Use:

```bash
python -m py_ore_tools.ore_snapshot_cli path/to/ore.xml --price --xva --ore-output-only
```

## Practical Summary

What is implemented now:

- ORE-compatible top-level CLI shape
- pricing/XVA/sensitivity/pack flows
- ORE-style terminal output
- ORE-style report output
- Python-specific summary/report output
- in-memory object API
- explicit `compare` / `python` / `ore` engine selection
- `OreSnapshotApp` wrapper class

What the current implementation is best thought of as:

- a Python ORE parity runner
- an ORE-shaped reporting adapter
- an in-memory app-style interface for the same runtime

It is not yet:

- a native ORE process wrapper
- a fully independent Python replacement for every ORE runtime path
- a finalized public API with perfectly settled naming
