# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 05. Calling Python and ORE Together

This notebook closes the series with two distinct stories that should not be mixed:
1. a fresh run of an **aligned benchmark case**, used as the main Python-vs-ORE comparison
2. a **live ORE-SWIG workflow** on the native snapshot interface, used to show the end-to-end runtime path

**Purpose**
- show the cleanest current comparison baseline
- keep the live SWIG workflow visible without overselling it as the parity benchmark
- end the series with an explicit recommendation on how to use both paths

**What you will learn**
- how to run a fresh aligned ORE case and compare it to Python
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
        and (candidate / "py_ore_tools").exists()
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

# Run a fresh aligned benchmark case first. This is the comparison to trust.
py_result, aligned_comparison, aligned_meta, aligned_ore_snapshot = nh.run_aligned_case_compare("flat_EUR_5Y_A", paths=2000)
print("Aligned case:", aligned_meta["case_name"])
print("Source case directory:", aligned_meta["source_case_dir"])
print("Fresh run root:", aligned_meta["fresh_run_root"])
print("Fresh ORE xml:", aligned_meta["fresh_ore_xml"])
print("Fresh output dir:", aligned_meta["fresh_output_dir"])
print("Trade ids:", aligned_meta["trade_ids"])
print("Requested metrics in original ORE case:", aligned_meta["requested_metrics_in_case"])
print("Python elapsed (s):", round(aligned_meta["python_elapsed_sec"], 4))
print("ORE fresh run elapsed (s):", round(aligned_meta["ore_elapsed_sec"], 4))

# Run a fresh live/native parity case that is known to line up on the current code path.
live_snapshot, live_py_result, live_comparison, live_perf_df, live_meta = nh.run_live_measure_lgm_compare(paths=1000)
print("Live/native source ORE xml:", live_meta["source_ore_xml"])
print("Live/native fresh run root:", live_meta["fresh_run_root"])
print("Live/native trade ids:", live_meta["trade_ids"])

# %% cell 3
"""
## Inputs we reuse from the repo

The aligned comparison still starts from the repo's benchmark case definition:
- `Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A`
- `run_ore_snapshot_native_xva.py`
- parity notes in `SKILL.md`

The difference is that this notebook now regenerates the ORE outputs during execution, so the comparison is based on
fresh run data rather than on the checked-in output folder.
"""

# %% cell 4
display(aligned_comparison)
aligned_plot = aligned_comparison[aligned_comparison["metric"].isin(["PV", "CVA", "DVA"])].copy()
nh.plot_metric_comparison(aligned_plot, "python_lgm", "ore_output", title="Aligned benchmark case: Python vs fresh ORE output")
nh.plot_metric_delta(aligned_plot, title="Aligned benchmark case: Python minus ORE")

# %% cell 5
"""
The paired charts above answer two different questions. The grouped bars show level agreement. The delta chart shows
whether the remaining gap is economically small or still material relative to the metric being compared.
"""

# %% cell 6
"""
## Runtime comparison on the aligned case

The numerical comparison is only half of the story. The table below shows the wall-clock cost of the Python LGM run
versus the fresh ORE executable run used to produce the benchmark outputs for this notebook execution.
"""

# %% cell 7
perf_df = pd.DataFrame(
    [
        {"engine": "python_lgm", "elapsed_sec": aligned_meta["python_elapsed_sec"]},
        {"engine": "ore_fresh_run", "elapsed_sec": aligned_meta["ore_elapsed_sec"]},
    ]
)
perf_df["speed_ratio_vs_python"] = perf_df["elapsed_sec"] / max(aligned_meta["python_elapsed_sec"], 1e-12)
display(perf_df)

fig, ax = plt.subplots(figsize=(8.0, 4.2))
nh.plot_bar_frame(perf_df, "engine", "elapsed_sec", title="Aligned case runtime: Python vs ORE", color=nh.PALETTE["rose"], ax=ax)
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
The aligned case is the main numerical comparison in this notebook. One important caveat remains:

- `flat_EUR_5Y_A` only requests `CVA` in the underlying ORE case

That is why funding-related metrics are not treated as comparable in the aligned section. The audit makes that visible
rather than leaving it implicit.
"""

# %% cell 11
"""
## Live Native Parity Case

The aligned benchmark above is file-based. This section uses a fresh live/native case from the current ORE example
set, chosen because it already lines up well in the regression pack: `ore_measure_lgm_fixed.xml`.

Unlike the old stress-classic demo, this is a real parity section. It is the right place to look at live `FBA/FCA`
because both engines are materially closer on this case.
"""

# %% cell 12
display(nh.snapshot_overview(live_snapshot))
display(nh.trade_frame(live_snapshot))
display(live_perf_df)
display(live_comparison)

live_plot = live_comparison[live_comparison["metric"].isin(["CVA", "DVA", "FBA", "FCA"])].copy()
nh.plot_metric_comparison(live_plot, "python_lgm", "ore_output", title="Live/native parity case: Python vs fresh ORE")
nh.plot_metric_delta(live_plot, title="Live/native parity case: Python minus ORE")

# %% cell 13
"""
Read this section differently from the stress-classic workflow demo that appeared earlier in the series. Here the
numbers are supposed to match. The point is to show that the current native Python path can in fact track a fresh ORE
live run on a case that is set up for parity rather than for broad stress-style coverage.
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

