# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 05. Calling Python and ORE Together

The main comparison here is **live only**: the same native `XVASnapshot` is valued with
`PythonLgmAdapter(fallback_to_swig=False)` and, when SWIG is available, `ORESwigAdapter()` (`OREApp`).
There is **no** regression-style comparison against saved `Output/*.csv` files.

**Purpose**
- show an apples-to-apples Python LGM vs in-process ORE run on one snapshot
- audit the case before trusting the numbers
- show **cold vs warm** wall time: first full stack (adapter `__init__` + engine + session + run) vs **run-only** (`session.run` on a reused adapter). Python’s cold path is tiny; ORE-SWIG cold path captures SWIG/`OREApp` warmup.

**What you will learn**
- how to load a repo parity case into `XVASnapshot` and run both engines explicitly
- how to read the dual-engine metric table and charts
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
from native_xva_interface import ORESwigAdapter, PythonLgmAdapter
from pythonore.io.ore_snapshot import validate_ore_input_snapshot, ore_input_validation_dataframe

swig = nh.swig_status()
print(swig["message"])

compare_paths = 250
app_ore_xml = nh.default_live_parity_ore_xml()
app_validation = ore_input_validation_dataframe(validate_ore_input_snapshot(app_ore_xml))
live_snapshot, aligned_ore_snapshot, live_meta = nh.load_case_snapshots(
    app_ore_xml, num_paths=compare_paths
)
# Enable DVA and FVA — the parity case XML defaults both to "N".
# FVA uses the key "FVA" which triggers FBA + FCA in PythonLgmAdapter.
# RuntimeConfig.xva_analytic carries the own-name and funding curve names needed
# by both the Python adapter (fallback hazard / spread approximation) and ORE-SWIG
# (setDvaName / setFvaBorrowingCurve / setFvaLendingCurve in build_input_parameters).
from dataclasses import replace as _dc_replace
from native_xva_interface import RuntimeConfig, XVAAnalyticConfig
_xva_analytic = XVAAnalyticConfig(
    dva_name="BANK",
    fva_borrowing_curve="BANK_EUR_BORROW",
    fva_lending_curve="BANK_EUR_LEND",
)
_existing_runtime = live_snapshot.config.runtime
_new_runtime = (
    _dc_replace(_existing_runtime, xva_analytic=_xva_analytic)
    if _existing_runtime is not None
    else RuntimeConfig(xva_analytic=_xva_analytic)
)
live_snapshot = _dc_replace(
    live_snapshot,
    config=_dc_replace(
        live_snapshot.config,
        analytics=("CVA", "DVA", "FVA"),
        runtime=_new_runtime,
    ),
)

# Explicit dual-engine run (no SWIG fallback inside Python LGM).
# Cold: build_adapter=... times adapter __init__ + XVAEngine + create_session + run (all one-time setup included).
# Warm: reuse adapter; warm=True times only session.run (steady-state per-valuation cost).
# Both cold and warm return full results — XVAs are shown from both timing passes below.
live_py_cold_result, live_py_cold_elapsed = nh.run_adapter(
    live_snapshot, build_adapter=lambda: PythonLgmAdapter(fallback_to_swig=False)
)
py_adapter = PythonLgmAdapter(fallback_to_swig=False)
live_py_warm_result, live_py_warm_elapsed = nh.run_adapter(live_snapshot, py_adapter, warm=True)

live_ore_cold_result = None
live_ore_cold_elapsed = None
live_ore_warm_result = None
live_ore_warm_elapsed = None
live_cold_compare = None
live_adapter_compare = None
live_ore_error = None
if swig["available"]:
    try:
        live_ore_cold_result, live_ore_cold_elapsed = nh.run_adapter(
            live_snapshot, build_adapter=lambda: ORESwigAdapter()
        )
    except Exception as exc:
        live_ore_error = str(exc)
        print("ORE-SWIG cold path failed (adapter + session + run):", exc)
else:
    print("Skipping explicit ORE-SWIG run: ORE module not available in this kernel.")

if swig["available"] and live_ore_error is None:
    try:
        ore_adapter = ORESwigAdapter()
        live_ore_warm_result, live_ore_warm_elapsed = nh.run_adapter(
            live_snapshot, ore_adapter, warm=True
        )
        live_cold_compare = nh.compare_results_frame(
            "python_lgm", live_py_cold_result, "ore_swig", live_ore_cold_result
        )
        live_adapter_compare = nh.compare_results_frame(
            "python_lgm", live_py_warm_result, "ore_swig", live_ore_warm_result
        )
    except Exception as exc:
        live_ore_error = str(exc)
        print("ORE-SWIG warm run failed:", exc)

print("Case ore.xml:", live_meta["ore_xml"])
print("Input dir:", live_meta["input_dir"])
print("Trade ids:", live_meta["trade_ids"])
print("Native snapshot num_paths (Python + ORE-SWIG):", live_meta["num_paths"])
print("Python cold (init+session+run) (s):", round(live_py_cold_elapsed, 4))
print("Python warm (run only) (s):", round(live_py_warm_elapsed, 4))
if live_ore_cold_elapsed is not None:
    print("ORE-SWIG cold (init+session+run) (s):", round(live_ore_cold_elapsed, 4))
if live_ore_warm_elapsed is not None:
    print("ORE-SWIG warm (run only) (s):", round(live_ore_warm_elapsed, 4))

# %% cell 3
"""
## Live dual-engine comparison on the repo parity case

`load_case_snapshots` builds the native `XVASnapshot` used by both adapters (`compare_paths` in the setup cell).
ORE numbers below come **only** from `ORESwigAdapter` / `OREApp` in this session — not from precomputed CSVs.

Input validation is a preflight on `ore.xml` (linkages, markets, etc.); it does not substitute for the parity audit
later in the notebook.
"""

# %% cell 4
XVA_METRICS = ["CVA", "DVA", "FBA", "FCA"]
_template = pd.DataFrame({"metric": XVA_METRICS})

def _xva_rows(compare_df):
    # left-join against template so all four metrics always appear; missing → 0
    return _template.merge(compare_df, on="metric", how="left").fillna(0.0)

def _py_xva_rows(result):
    # Python-only fallback when no ORE comparison is available
    return pd.DataFrame([
        {"metric": m, "python_lgm": result.xva_by_metric.get(m, 0.0)}
        for m in XVA_METRICS
    ])

display(nh.snapshot_overview(live_snapshot))
display(nh.trade_frame(live_snapshot))
display(app_validation.head(12))

# Cold run — Python XVAs always shown; ORE comparison added when available
print("=== Cold run (init + session + run): CVA / DVA / FVA ===")
if live_cold_compare is not None:
    cold_xva = _xva_rows(live_cold_compare)
    display(cold_xva)
    nh.plot_metric_comparison(
        cold_xva, "python_lgm", "ore_swig",
        title="Cold: Python LGM vs ORE-SWIG — CVA / DVA / FVA",
    )
    nh.plot_metric_delta(cold_xva, title="Cold: ORE-SWIG minus Python LGM")
else:
    display(_py_xva_rows(live_py_cold_result))
    print("(ORE-SWIG unavailable for cold comparison" + (f": {live_ore_error}" if live_ore_error else "") + ")")

# Warm run — same structure
print("=== Warm run (run only): CVA / DVA / FVA ===")
if live_adapter_compare is not None:
    warm_xva = _xva_rows(live_adapter_compare)
    display(warm_xva)
    nh.plot_metric_comparison(
        warm_xva, "python_lgm", "ore_swig",
        title="Warm: Python LGM vs ORE-SWIG — CVA / DVA / FVA",
    )
    nh.plot_metric_delta(warm_xva, title="Warm: ORE-SWIG minus Python LGM")
else:
    display(_py_xva_rows(live_py_warm_result))
    print("(ORE-SWIG unavailable for warm comparison" + (f": {live_ore_error}" if live_ore_error else "") + ")")

# %% cell 5
"""
Both cold and warm runs produce the same XVA numbers on the same snapshot — only the wall time differs.
`delta` is **ORE-SWIG minus Python LGM**. FVA here means FBA + FCA; if either engine does not compute
them they will appear as 0.
"""

# %% cell 6
"""
## Runtime comparison on the aligned case

**Cold** rows time the **first** end-to-end pass: ``run_adapter(..., build_adapter=lambda: …)`` includes adapter
construction, ``XVAEngine``, ``create_session`` (and ``map_snapshot``), and ``session.run``. For ORE-SWIG this is
where module/`OREApp` warmup shows up.

**Warm** rows time **only** ``session.run`` on an adapter that already exists (``run_adapter(..., warm=True)``) —
the usual “many valuations in one batch” view. Python’s adapter is lightweight, so cold ≈ warm; ORE-SWIG typically
has a large **cold − warm** gap. Timings use ``time.perf_counter`` (wall time, not CPU).
"""

# %% cell 7
perf_rows = [
    {"label": "Python: cold (init + session + run)", "elapsed_sec": live_py_cold_elapsed},
    {"label": "Python: warm (run only)", "elapsed_sec": live_py_warm_elapsed},
]
if live_ore_cold_elapsed is not None:
    perf_rows.append(
        {"label": "ORE-SWIG: cold (init + session + run)", "elapsed_sec": live_ore_cold_elapsed}
    )
if live_ore_warm_elapsed is not None:
    perf_rows.append({"label": "ORE-SWIG: warm (run only)", "elapsed_sec": live_ore_warm_elapsed})
perf_df = pd.DataFrame(perf_rows)
perf_df["speed_ratio_vs_python_warm"] = perf_df["elapsed_sec"] / max(live_py_warm_elapsed, 1e-12)
display(perf_df)
ore_setup = (
    round(live_ore_cold_elapsed - live_ore_warm_elapsed, 4)
    if live_ore_cold_elapsed is not None and live_ore_warm_elapsed is not None
    else None
)
print(
    "Amortizable setup on first full pass (cold − warm): Python",
    round(live_py_cold_elapsed - live_py_warm_elapsed, 4),
    "s | ORE-SWIG",
    f"{ore_setup} s" if ore_setup is not None else "n/a",
)

fig, ax = plt.subplots(figsize=(9.5, 4.5))
nh.plot_ranked_bars(
    perf_df,
    "label",
    "elapsed_sec",
    title="Cold vs warm: ORE-SWIG warmup vs small Python adapter cost",
    color=nh.PALETTE["rose"],
    ax=ax,
)
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
The live dual-engine table is only as fair as the case setup. One important caveat remains:

- the underlying ORE case still determines which analytics are actually comparable

That is why funding-related metrics are not always treated as comparable. The audit makes that visible
rather than leaving it implicit.
"""

# %% cell 11
"""
## Stored parity context from the repo

The parity report below comes from a different common-snapshot setup with the full XVA stack enabled. It is still
useful because it summarizes the broader prototype status: DVA and FVA are closer, while CVA is still the main gap.
"""

# %% cell 12
parity_df, parity_summary = nh.parse_parity_report()
display(parity_df)
print(parity_summary)
nh.plot_metric_delta(parity_df, title="Stored parity report: ORE minus Python")

capability_df = nh.capability_matrix_frame(swig["available"])
display(capability_df)
nh.plot_boolean_matrix(capability_df, row_col="capability", value_cols=["python", "ore_swig"], title="Current capability split")

# %% cell 13
"""
## Key takeaways

- For **Python vs ORE** on one loaded case, use explicit `PythonLgmAdapter` + `ORESwigAdapter` — both consume the same
  `XVASnapshot` and ORE’s side is **live** `OREApp` output, not files under `Output/`.
- **Cold vs warm** wall times make ORE-SWIG’s one-time setup visible next to Python (where cold ≈ warm); use **warm**
  rows to compare steady-state pricing speed after adapters exist.
- Keep the parity audit in the notebook; it tells you which metrics are actually comparable.
- Regression-style checks against committed CSVs live in the CLI / harness, not in this notebook.

## End of series

These five notebooks now form one continuous story: Python snapshot construction, ORE-backed loading, market
calibration, Python LGM XVA, and a final joint workflow focused on **live** dual-engine comparison.
"""

