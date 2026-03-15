# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 05. Calling Python and ORE Together

This notebook closes the series with two distinct stories that should not be mixed:
1. a fresh run of an **aligned benchmark case**, used as the main Python-vs-ORE comparison
2. a **live ORE-SWIG workflow** on the native snapshot interface, used to show the end-to-end runtime path

**Purpose**
- show the cleanest current in-memory comparison baseline
- keep the live SWIG workflow visible without overselling it as the main regression harness
- end the series with an explicit recommendation on how to use both paths

**What you will learn**
- how to run the canonical in-memory `OreSnapshotApp` comparison path
- how to audit that case before trusting the numbers
- how the live ORE-SWIG workflow fits into the current prototype story
"""

# %% cell 1
from pathlib import Path
import os
import sys

def _pythonorerunner_root(candidate: Path) -> bool:
    return (
        (candidate / "notebook_series" / "series_helpers.py").exists()
        and ((candidate / "pythonore").exists() or (candidate / "src" / "pythonore").exists())
    )

def _engine_root(candidate: Path) -> bool:
    return (candidate / "Tools" / "PythonOreRunner" / "notebook_series" / "series_helpers.py").exists()

def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if _pythonorerunner_root(candidate) or _engine_root(candidate):
            return candidate
    repo_hint = Path("/Users/gordonlee/Documents/Engine")
    if _engine_root(repo_hint):
        return repo_hint
    standalone_hint = Path("/Users/gordonlee/Documents/PythonOreRunner")
    if _pythonorerunner_root(standalone_hint):
        return standalone_hint
    raise RuntimeError("Could not locate the notebook repo root from the current notebook working directory")

REPO_ROOT = _find_repo_root(Path.cwd())
NOTEBOOK_DIR = REPO_ROOT / "notebook_series" if _pythonorerunner_root(REPO_ROOT) else REPO_ROOT / "Tools" / "PythonOreRunner" / "notebook_series"
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

# %% cell 2
from native_xva_interface import PythonLgmAdapter

swig = nh.swig_status()
print(swig["message"])

# Use the canonical in-memory OreSnapshotApp surface first. This is the maintained regression path.
app_ore_xml = nh.default_live_parity_ore_xml()
app_result, app_meta = nh.run_ore_snapshot_app_case(app_ore_xml, engine="compare", price=True, xva=True, paths=1000)
app_comparison = nh.ore_snapshot_app_metric_frame(app_result)
app_validation = pd.DataFrame(app_result.input_validation_rows)
live_snapshot, aligned_ore_snapshot, live_meta = nh.load_case_snapshots(app_ore_xml)
live_py_result, live_py_elapsed = nh.run_adapter(live_snapshot, PythonLgmAdapter(fallback_to_swig=False))

print("App source ORE xml:", app_meta["ore_xml"])
print("App input dir:", app_meta["input_dir"])
print("App output dir:", app_meta["output_dir"])
print("App elapsed (s):", round(app_meta["elapsed_sec"], 4))
print("Trade ids:", live_meta["trade_ids"])
print("Python adapter elapsed (s):", round(live_py_elapsed, 4))

# %% cell 3
"""
## Inputs we reuse from the repo

The comparison starts from the repo-backed parity case and runs it through the canonical in-memory app surface:
- `pythonore.workflows.OreSnapshotApp`
- `pythonore.workflows.run_case_from_buffers`
- parity notes in `SKILL.md`

The important operational point is that the notebook receives the comparison payload back in Python without creating
new persistent output folders in the repo.
"""

# %% cell 4
display(nh.ore_snapshot_app_summary_frame(app_result))
display(app_comparison)
aligned_plot = app_comparison[app_comparison["metric"].isin(["PV", "CVA", "DVA"])].copy()
nh.plot_metric_comparison(aligned_plot, "python_lgm", "ore_output", title="OreSnapshotApp compare mode: Python vs ORE reference")
nh.plot_metric_delta(aligned_plot, title="OreSnapshotApp compare mode: Python minus ORE")

# %% cell 5
"""
The paired charts above answer two different questions. The grouped bars show level agreement. The delta chart shows
whether the remaining gap is economically small or still material relative to the metric being compared.
"""

# %% cell 6
"""
## Runtime comparison on the aligned case

The numerical comparison is only half of the story. The table below shows the in-memory app cost and the direct
Python adapter cost on the same case.
"""

# %% cell 7
perf_df = pd.DataFrame(
    [
        {"engine": "ore_snapshot_app_compare", "elapsed_sec": app_meta["elapsed_sec"]},
        {"engine": "python_lgm_adapter", "elapsed_sec": live_py_elapsed},
    ]
)
perf_df["speed_ratio_vs_python"] = perf_df["elapsed_sec"] / max(live_py_elapsed, 1e-12)
display(perf_df)

fig, ax = plt.subplots(figsize=(8.0, 4.2))
nh.plot_bar_frame(perf_df, "engine", "elapsed_sec", title="In-memory app vs direct Python adapter", color=nh.PALETTE["rose"], ax=ax)
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 8
"""
## Parity readiness on the aligned case

Before trusting the comparison, audit the case. This is the step that prevents us from comparing “same trade name”
while still using different schedules, curves, or enabled analytics.
"""

# %% cell 9
aligned_parity_report = aligned_ore_snapshot.parity_completeness_report()
aligned_parity_df = aligned_ore_snapshot.parity_completeness_dataframe()
display(aligned_parity_df)
print("Comparability:", aligned_parity_report["comparability"])
print("Issues:", aligned_parity_report["issues"])
comparability_df = aligned_parity_df[aligned_parity_df["section"] == "comparability"].copy()
comparability_df["comparable"] = comparability_df["value"].astype(bool)
nh.plot_boolean_matrix(comparability_df, row_col="field", value_cols=["comparable"], title="Aligned case: comparable XVA blocks")

# %% cell 10
"""
The in-memory app case is the main numerical comparison in this notebook. One important caveat remains:

- the underlying ORE case still determines which analytics are actually comparable

That is why funding-related metrics are not always treated as comparable. The audit makes that visible
rather than leaving it implicit.
"""

# %% cell 11
"""
## Live Native Snapshot Workflow

The app comparison above is the maintained regression surface. This section keeps the direct native snapshot
interface visible on the same case, so the reader can still see how the `XVALoader` + `PythonLgmAdapter` path
fits into the overall workflow.
"""

# %% cell 12
display(nh.snapshot_overview(live_snapshot))
display(nh.trade_frame(live_snapshot))
display(app_validation.head(12))
display(nh.result_metrics_frame(live_py_result))

# %% cell 13
"""
Read this section differently from the app comparison above. The point here is not to create another benchmark table,
but to keep the direct native runtime path visible. That path is still useful when you want explicit control over the
dataclass snapshot and adapter boundary.
"""

# %% cell 14
"""
## Stored parity context from the repo

The parity report below comes from a different common-snapshot setup with the full XVA stack enabled. It is still
useful because it summarizes the broader prototype status: DVA and FVA are closer, while CVA is still the main gap.
"""

# %% cell 15
parity_df, parity_summary = nh.parse_parity_report()
display(parity_df)
print(parity_summary)
nh.plot_metric_delta(parity_df, title="Stored parity report: ORE minus Python")

capability_df = nh.capability_matrix_frame(swig["available"])
display(capability_df)
nh.plot_boolean_matrix(capability_df, row_col="capability", value_cols=["python", "ore_swig"], title="Current capability split")

# %% cell 16
"""
## Key takeaways

- Use the fresh aligned benchmark case when you want a clean Python-vs-ORE comparison.
- Use the live/native parity case when you want to show that Python can also match a fresh ORE run on a case tuned for parity, including `FBA/FCA`.
- Keep the parity audit in the notebook, because it tells you which comparisons are actually fair.

## End of series

These five notebooks now form one continuous story: Python snapshot construction, fresh ORE-backed loading, market
calibration, Python LGM XVA, and a final Python-and-ORE workflow that distinguishes aligned comparisons from live demos.
"""

