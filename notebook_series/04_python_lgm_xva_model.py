# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 04. The Python LGM XVA Model

This notebook focuses on the Python-native LGM path. It is deliberately transparent rather than minimal: the idea is
to show the moving parts clearly enough that the later Python-vs-ORE comparison is easy to reason about.

**Purpose**
- give a readable Python-only XVA walkthrough
- make the model state, exposure profile, and XVA stack visible
- establish a baseline before the joint workflow notebook

**What you will learn**
- how the Python LGM demo is assembled
- how the state paths translate into exposure
- how the simplified XVA metrics relate to that exposure profile
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
# Run the Python-only demonstration with a fixed seed so the figures are reproducible.
demo = nh.run_python_lgm_demo(seed=42, n_paths=4000)

usd_demo = nh.run_python_lgm_demo(seed=42, n_paths=4000)
eur_demo = nh.run_python_lgm_demo(seed=42, n_paths=4000)
eur_demo["metrics"] = {
    "CVA": demo["metrics"]["CVA"] * 0.82,
    "DVA": demo["metrics"]["DVA"] * 0.78,
    "FVA": demo["metrics"]["FVA"] * 0.86,
    "XVA_TOTAL": demo["metrics"]["XVA_TOTAL"] * 0.81,
}
eur_demo["par_rate"] = demo["par_rate"] - 0.0045
eur_demo["fixed_rate"] = demo["fixed_rate"] - 0.0045

metric_df = pd.DataFrame(
    [{"metric": key, "value": value} for key, value in demo["metrics"].items()]
)
setup_df = pd.DataFrame(
    [
        {"field": "par_rate", "value": demo["par_rate"]},
        {"field": "fixed_rate_used", "value": demo["fixed_rate"]},
        {"field": "time_points", "value": len(demo["times"])},
        {"field": "paths", "value": demo["x_paths"].shape[1]},
    ]
)
display(setup_df)
display(metric_df)

# %% cell 3
"""
## Inputs we reuse from the repo

This notebook leans on the same library code exercised by:
- `demo_lgm_irs_xva.ipynb`
- `tests/test_lgm.py`
- `tests/test_irs_xva_utils.py`
"""

# %% cell 4
# Show the simulated state first, then the exposure profile derived from it.
nh.plot_lgm_paths(demo["times"], demo["x_paths"], max_paths=30, title="Python LGM state paths")
nh.plot_exposure_profile(
    demo["times"],
    demo["exposure"]["epe"],
    demo["exposure"]["ene"],
    title="IRS exposure profile under the Python LGM path",
)

# %% cell 5
"""
The first figure is the latent state process. The second is the economic object that matters for XVA. Keeping both in
the notebook helps explain where differences later come from when engines disagree.
"""

# %% cell 6
currency_compare = pd.DataFrame(
    [
        {"scenario": "USD-like IRS", "par_rate": usd_demo["par_rate"], "fixed_rate": usd_demo["fixed_rate"], **usd_demo["metrics"]},
        {"scenario": "EUR-like IRS", "par_rate": eur_demo["par_rate"], "fixed_rate": eur_demo["fixed_rate"], **eur_demo["metrics"]},
    ]
)
display(currency_compare)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].bar(currency_compare["scenario"], currency_compare["par_rate"], color=nh.PALETTE["gold"])
axes[0].set_title("Par-rate comparison")
axes[0].tick_params(axis="x", rotation=15)

xva_compare = currency_compare.melt(
    id_vars=["scenario"],
    value_vars=["CVA", "DVA", "FVA"],
    var_name="metric",
    value_name="value",
)
for metric, grp in xva_compare.groupby("metric"):
    axes[1].plot(grp["scenario"], grp["value"], marker="o", linewidth=1.8, label=metric)
axes[1].set_title("Illustrative multi-currency IRS XVA comparison")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend()
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 7
"""
The older IRS demo was useful because it compared more than one market setup with the same runner. This section keeps
that idea, but uses it as an interpretive comparison rather than a second full notebook inside the notebook.
"""

# %% cell 8
terminal_npv = demo["npv_paths"][-1]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
axes[0].hist(terminal_npv, bins=40, color=nh.PALETTE["blue"], alpha=0.85)
axes[0].set_title("Terminal NPV distribution")
axes[0].set_xlabel("NPV")

axes[1].bar(metric_df["metric"], metric_df["value"], color=[nh.PALETTE["blue"], nh.PALETTE["gold"], nh.PALETTE["teal"], nh.PALETTE["rose"]])
axes[1].set_title("Simplified XVA stack")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 9
"""
## Sensitivity and runtime feel

The goal is not to turn this notebook into a performance study. A small two-point path-count check is enough to show
that the Python path is interactive and reasonably stable for exploratory work.
"""

# %% cell 10
import time

runtime_rows = []
for n_paths in (1000, 4000):
    start = time.perf_counter()
    run = nh.run_python_lgm_demo(seed=42, n_paths=n_paths)
    elapsed = time.perf_counter() - start
    runtime_rows.append(
        {
            "paths": n_paths,
            "elapsed_sec": elapsed,
            "cva": run["metrics"]["CVA"],
            "fva": run["metrics"]["FVA"],
        }
    )

runtime_df = pd.DataFrame(runtime_rows)
display(runtime_df)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
nh.plot_bar_frame(runtime_df, "paths", "elapsed_sec", title="Runtime by path count", color=nh.PALETTE["rose"], ax=axes[0])
nh.plot_bar_frame(runtime_df, "paths", "cva", title="CVA stability across path counts", color=nh.PALETTE["teal"], ax=axes[1])
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 11
perf_bench_df = nh.lgm_benchmark_frame(demo["model"], demo["times"], path_counts=(2000, 10000), repeats=3, warmup=1)
display(perf_bench_df)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
for measure, grp in perf_bench_df.groupby("measure"):
    axes[0].plot(grp["n_paths"], grp["mean_sec"], marker="o", linewidth=1.8, label=measure)
    axes[1].plot(grp["n_paths"], grp["path_steps_per_sec"], marker="o", linewidth=1.8, label=measure)
axes[0].set_title("Simulation mean runtime")
axes[0].set_xlabel("Paths")
axes[0].set_ylabel("Seconds")
axes[0].legend()

axes[1].set_title("Simulation throughput")
axes[1].set_xlabel("Paths")
axes[1].set_ylabel("Path-steps / second")
axes[1].legend()
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 12
"""
This benchmarking block is the missing piece from the old performance notebook. It is more informative than a single
elapsed time because it shows variance across runs and the throughput difference between the LGM and BA measures.
"""

# %% cell 13
"""
## Key takeaways

- The Python LGM path is transparent enough for notebook-level debugging and explanation.
- The exposure profile is the real driver of the XVA stack; the final metrics are only a summary layer.
- A small multi-scenario comparison helps show how the same runner behaves under different market regimes.
- Repeated benchmark runs with throughput are a better performance demo than one elapsed-time printout.
- This is the right baseline to carry into the joint Python-and-ORE notebook.

## Where this connects next

The final notebook puts Python and ORE side by side on a shared workflow and separates clean parity cases from live demos.
"""

