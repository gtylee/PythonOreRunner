# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 01. Python Dataclasses -> ORE-SWIG

This notebook is the start of the series. The goal is to make the Python-facing snapshot model concrete before we
talk about loaders, calibration, or XVA numbers. The core question here is simple: what exactly do we hand around
in Python before either engine runs?

**Purpose**
- show the shape of the `XVASnapshot` object
- prove that the snapshot survives serialization without changing meaning
- show how the same snapshot is mapped into ORE-style runtime inputs

**Prerequisites**
- the main path is Python-only and should run anywhere
- the ORE-SWIG adapter check is optional and skips cleanly if the bindings are unavailable

**What you will learn**
- which dataclasses make up the snapshot
- what `stable_key()` is protecting
- which files and XML payloads are synthesized by `map_snapshot()`
"""

# %% cell 1
from pathlib import Path
import os
import sys

def _is_pythonorerunner_root(path: Path) -> bool:
    return (
        (path / "notebook_series" / "series_helpers.py").exists()
        and (path / "native_xva_interface").exists()
        and (path / "py_ore_tools").exists()
    )

def _is_engine_root(path: Path) -> bool:
    return (path / "Tools" / "PythonOreRunner" / "notebook_series" / "series_helpers.py").exists()

def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if _is_pythonorerunner_root(candidate) or _is_engine_root(candidate):
            return candidate
    repo_hint = Path("/Users/gordonlee/Documents/PythonOreRunner")
    if _is_pythonorerunner_root(repo_hint):
        return repo_hint
    repo_hint = Path("/Users/gordonlee/Documents/Engine")
    if _is_engine_root(repo_hint):
        return repo_hint
    raise RuntimeError("Could not locate a PythonOreRunner or Engine repo root from the current notebook working directory")

def _pythonorerunner_root(repo_root: Path) -> Path:
    if _is_pythonorerunner_root(repo_root):
        return repo_root
    return repo_root / "Tools" / "PythonOreRunner"

REPO_ROOT = _find_repo_root(Path.cwd())
NOTEBOOK_DIR = _pythonorerunner_root(REPO_ROOT) / "notebook_series"
for path in (NOTEBOOK_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mplconfig")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import series_helpers as nh

try:
    from IPython.display import display
except Exception:
    def display(obj):
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

repo = nh.bootstrap_notebook_env(REPO_ROOT)
nh.apply_plot_style()
print(repo)
RUN_ORE_SWIG = os.getenv("RUN_ORE_SWIG_DEMOS") == "1"

# %% cell 2
"""
## Inputs we reuse from the repo

This notebook stays close to the tested dataclass and adapter surface already used elsewhere in the repo:
- `native_xva_interface/tests/test_step1_dataclasses.py`
- `native_xva_interface/tests/test_ore_swig_adapter.py`
- `native_xva_interface/notebooks/programmatic_ore_swig_demo.py`
"""

# %% cell 3
from native_xva_interface import XVASnapshot, map_snapshot, DeterministicToyAdapter, ORESwigAdapter

# Build one small snapshot entirely from Python so we can inspect the structure directly.
snapshot = nh.make_programmatic_snapshot(num_paths=128)
round_trip = XVASnapshot.from_dict(snapshot.to_dict())
mapped = map_snapshot(snapshot)

display(nh.snapshot_overview(snapshot))
display(nh.trade_frame(snapshot))
nh.plot_snapshot_composition(snapshot, title="Programmatic snapshot: component count and quote mix")

# %% cell 4
"""
Read the chart above as an inventory check. The left panel shows how much business content sits inside the snapshot.
The right panel shows that market data is already grouped into recognizable quote families before any ORE call.
"""

# %% cell 5
# These checks make the object contract explicit.
validation = pd.DataFrame(
    [
        {"check": "round-trip equality", "value": round_trip.to_dict() == snapshot.to_dict()},
        {"check": "stable_key preserved", "value": round_trip.stable_key() == snapshot.stable_key()},
        {"check": "analytics", "value": ",".join(snapshot.config.analytics)},
        {"check": "market quote count", "value": len(snapshot.market.raw_quotes)},
    ]
)
display(validation)
display(nh.quote_family_frame(snapshot))
nh.plot_mapping_pipeline()

# %% cell 6
"""
## Before and after mapping

`map_snapshot()` is the bridge between the Python object model and the ORE runtime shape. The important point is
not the exact XML syntax; it is that the notebook can show, in one place, what leaves the dataclass layer.
"""

# %% cell 7
# Summarize the generated runtime payload instead of dumping raw files first.
display(nh.mapped_input_summary(mapped))
display(nh.xml_buffer_summary(mapped))
nh.plot_xml_buffer_sizes(mapped, title="Generated XML payload sizes after mapping")

preview_rows = []
for idx in range(5):
    preview_rows.append(
        {
            "market_data_line": mapped.market_data_lines[idx] if idx < len(mapped.market_data_lines) else "...",
            "fixing_data_line": mapped.fixing_data_lines[idx] if idx < len(mapped.fixing_data_lines) else "...",
        }
    )
preview = pd.DataFrame(preview_rows)
display(preview)

# %% cell 8
"""
The size chart is useful for orientation. In practice, a handful of XML documents carry nearly all of the runtime
structure, while market and fixing lines stay in flat text form.
"""

# %% cell 9
# The toy adapter is a fast contract check: the snapshot can already flow through the engine boundary.
toy_result, toy_elapsed = nh.run_adapter(snapshot, DeterministicToyAdapter())
print(f"Toy adapter elapsed: {toy_elapsed:.4f}s")
display(nh.result_metrics_frame(toy_result))

# %% cell 10
"""
## Optional SWIG run

This is intentionally a boundary check, not a parity claim. The point is that the same snapshot can be handed to
an ORE-backed adapter without rewriting the notebook around engine-specific inputs.
"""

# %% cell 11
swig = nh.swig_status()
print(swig["message"])

if swig["available"] and RUN_ORE_SWIG:
    # Run the same snapshot through the ORE-backed adapter only when the user asks for it.
    swig_result, swig_elapsed = nh.run_adapter(snapshot, ORESwigAdapter())
    print(f"ORE-SWIG elapsed: {swig_elapsed:.4f}s")
    display(nh.result_metrics_frame(swig_result))
elif swig["available"]:
    print("Skipping ORE-SWIG adapter execution. Set RUN_ORE_SWIG_DEMOS=1 to enable it.")
else:
    print("Skipping ORE-SWIG adapter execution in this environment.")

# %% cell 12
"""
## Key takeaways

- The dataclass layer already carries enough structure to be the common hand-off point for both engines.
- `stable_key()` matters because notebooks and tests need a cheap way to detect a real change in economic content.
- `map_snapshot()` is the explicit boundary where Python objects turn into ORE-style runtime payloads.

## Where this connects next

The next notebook moves from a hand-built snapshot to a loader-backed snapshot built from a fresh ORE run.
"""

