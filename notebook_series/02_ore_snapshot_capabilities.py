# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 02. ORE Snapshot and Its Abilities

This notebook shifts from a hand-built snapshot to a loader-backed snapshot built from a fresh ORE run. The goal is
to make the snapshot itself feel like the primary artifact: something you can inspect, audit, serialize, and hand to
both Python and ORE code without reopening the raw files.

**Purpose**
- show what the loader actually materializes
- keep the data fresh by generating the ORE outputs inside the notebook run
- introduce the stricter parity-grade `OreSnapshot` audit

**What you will learn**
- what comes back from `XVALoader.from_files(...)`
- how to inspect trades, quotes, netting, and generated XML without digging through the repo tree
- how to tell whether the resulting case is parity-ready
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
from native_xva_interface import map_snapshot, DeterministicToyAdapter

# Run ORE now, then load the fresh inputs and outputs into Python-facing snapshot objects.
snapshot, ore_snapshot, fresh_meta = nh.load_fresh_case_snapshots("flat_EUR_5Y_A", label="notebook_02")
mapped = map_snapshot(snapshot)

print("Fresh run root:", fresh_meta["run_root"])
print("Fresh ORE xml:", fresh_meta["ore_xml"])
print("Fresh output dir:", fresh_meta["output_dir"])
display(nh.snapshot_overview(snapshot))
nh.plot_snapshot_composition(snapshot, title="Fresh loader-backed snapshot: inventory and market mix")

# %% cell 3
"""
## Inputs we reuse from the repo

The notebook starts from the aligned benchmark case but does not trust its checked-in outputs. Instead it executes a
new ORE run first and only then loads the results. That means the portfolio, curves, exposure files, and XVA outputs
shown below are all from the current run.

The loader-backed path still follows the same code used in the tests and demos:
- `native_xva_interface/loader.py`
- `native_xva_interface/tests/test_step2_loader.py`
- `native_xva_interface/tests/test_step3_merge.py`
"""

# %% cell 4
trade_df = nh.trade_frame(snapshot)
display(trade_df.head(12))
display(nh.netting_frame(snapshot).head(12))
display(nh.collateral_frame(snapshot).head(12))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
nh.plot_ranked_bars(nh.quote_family_frame(snapshot), "family", "count", title="Largest quote families", color=nh.PALETTE["teal"], top_n=8, ax=axes[0])
nh.plot_ranked_bars(trade_df, "trade_id", "notional", title="Trade notionals in the loaded snapshot", color=nh.PALETTE["gold"], ax=axes[1])
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 5
"""
## Snapshot as a reusable artifact

The loader has already normalized XML, market text, fixings, and netting configuration into one object. That is the
practical value of the snapshot: most downstream code can stop caring which original file carried which piece.
"""

# %% cell 6
snapshot_dict = snapshot.to_dict()
snapshot_json = nh.to_pretty_json(snapshot_dict, limit=1800)
print(snapshot_json)

display(nh.mapped_input_summary(mapped))
display(nh.xml_buffer_summary(mapped).head(12))
nh.plot_xml_buffer_sizes(mapped, title="Mapped XML payload sizes for the fresh loader snapshot")

# %% cell 7
"""
## Parity completeness audit

A runtime snapshot being usable is not the same thing as a case being parity-ready. The `OreSnapshot` audit is more
strict: it checks whether the run captured enough schedule, curve, credit, funding, and exposure information for a
fair Python-vs-ORE comparison.
"""

# %% cell 8
if ore_snapshot is not None:
    parity_report = ore_snapshot.parity_completeness_report()
    parity_df = ore_snapshot.parity_completeness_dataframe()
    display(parity_df)
    print("Parity issues:", parity_report["issues"])
    print("Parity ready:", parity_report["parity_ready"])
    comparability_df = parity_df[parity_df["section"] == "comparability"].copy()
    comparability_df["comparable"] = comparability_df["value"].astype(bool)
    nh.plot_boolean_matrix(comparability_df, row_col="field", value_cols=["comparable"], title="Which XVA blocks are comparable?")
else:
    print("No parity-grade OreSnapshot audit available for this loader bundle.")

# %% cell 9
"""
The audit table is the one to trust when you want to compare engines. It separates “the case runs” from “the case is
economically lined up well enough to compare specific XVA numbers.”
"""

# %% cell 10
# One cheap engine pass shows that the fresh snapshot is immediately executable from Python.
toy_result, toy_elapsed = nh.run_adapter(snapshot, DeterministicToyAdapter())
print(f"Deterministic toy run elapsed: {toy_elapsed:.4f}s")
display(nh.result_metrics_frame(toy_result))

# %% cell 11
"""
## Key takeaways

- The snapshot is the cleanest inspection boundary in the current Python-facing stack.
- Fresh ORE output generation removes the ambiguity of stale checked-in files.
- The parity audit is the right place to decide what can be compared, not an afterthought.

## Where this connects next

The next notebook zooms into the market side: curve extraction, curve fitting, and how those outputs feed the LGM model.
"""

