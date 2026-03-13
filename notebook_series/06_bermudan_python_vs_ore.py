# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 06. Bermudan Swaption Pricing: Python vs ORE

This notebook focuses on the native Bermudan swaption path that was added on top of the Python LGM stack.
The objective is narrower than the XVA notebooks: price a few Bermudans, compare the Python methods against
ORE classic, and inspect one case in more detail.

**Purpose**
- show the current Python Bermudan pricing methods side by side with ORE classic
- make the calibration-vs-simulation model-source point explicit
- inspect one benchmark case through its exercise diagnostics

**What you will learn**
- how close Python `backward` is to ORE classic on the current benchmark pack
- why Python `lsmc` is still useful as a control but is not the parity target
- what the Bermudan exercise diagnostics look like on the base case
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
"""
## Inputs reused from the repo

This notebook uses the native Bermudan benchmark pack and the Python Bermudan pricer already exercised by the regression tests:
- `Tools/PythonOreRunner/parity_artifacts/bermudan_method_compare`
- `native_xva_interface/bermudan.py`
- `native_xva_interface/tests/test_bermudan_pricer.py`
"""

# %% cell 3
berm_comp = nh.load_bermudan_method_comparison()
display(berm_comp)

berm_overview = berm_comp[["case_name", "fixed_rate", "py_lsmc", "py_backward", "ore_classic", "ore_amc", "ore_amc_source"]].copy()
display(berm_overview)
nh.plot_bermudan_method_levels(berm_comp, title="Stored Bermudan benchmark pack: Python vs ORE classic")
nh.plot_bermudan_ore_deltas(berm_comp, title="Stored Bermudan benchmark pack: Python minus ORE classic")

# %% cell 4
"""
The main thing to read off the two plots is that `py_backward` is now the ORE-classic parity method on this
benchmark pack. `py_lsmc` is still informative because it stays close to `backward`, but it is a control, not
the target engine.
"""

# %% cell 5
"""
## Detailed base case: 2% Bermudan

The `berm_200bp` case is the cleanest single example because it sits near the middle of the pack and also has
an AMC fallback number stored in the comparison CSV. Here we reprice the case directly through the Python API
so the notebook reflects the current code rather than only the stored summary file.
"""

# %% cell 6
berm_summary, berm_diag, berm_speed, berm_meta = nh.run_bermudan_case_summary("berm_200bp", num_paths=4096, seed=42)
display(pd.DataFrame([berm_meta]))
display(berm_summary)
display(berm_speed)
display(berm_diag)

nh.plot_metric_delta(
    berm_summary.rename(columns={"method": "metric", "delta_vs_ore": "delta"}),
    title="berm_200bp: Python method minus ORE classic",
)
fig, ax = plt.subplots(figsize=(9.0, 4.2))
nh.plot_bar_frame(
    berm_speed,
    "engine",
    "elapsed_sec",
    title="berm_200bp runtime view: Python methods and stored ORE timings",
    color=nh.PALETTE["gold"],
    ax=ax,
)
plt.tight_layout()
plt.show()
plt.close(fig)
nh.plot_bermudan_exercise_diagnostics(berm_diag, title="berm_200bp exercise diagnostics")

# %% cell 7
"""
The summary table should show `model_param_source = calibration` for both Python methods. That is important:
on these benchmark cases the best parity source is the actual `calibration.xml` emitted by ORE classic, not a
Python-side reconstruction of the trade-specific builder path.
"""

# %% cell 8
"""
The timing table mixes two sources on purpose:

- Python timings are measured live in the notebook with wall-clock time
- ORE timings come from the stored `pricingstats.csv` and `runtimes.csv` written by the benchmark run

So this section is best read as an operational runtime view, not as a strict apples-to-apples microbenchmark.
"""

# %% cell 9
"""
## Interpreting the diagnostics

The exercise-probability chart answers “when does the option matter most?”. The boundary-state chart answers
“where is the continuation-versus-exercise switch happening in the LGM state variable?”. Those are the quickest
sanity checks when the aggregate price looks wrong.
"""

# %% cell 10
"""
## Sensitivity definition example: direct quote bump vs ORE sensitivity analytic

The biggest open issue is not the Bermudan price itself. It is the meaning of the sensitivity number.

For the `berm_200bp` case we now run three calculations live in the notebook:
- ORE `sensitivity.csv`
- direct ORE quote bump-and-reprice
- Python quote bump-and-reprice

This is important because ORE's sensitivity analytic is not the same as "edit one line in the market file and
reprice". It goes through the sensitivity scenario engine and par-conversion layer first.
"""

# %% cell 11
sens_payload, sens_meta, sens_rows, sens_config, sens_scenario = nh.run_bermudan_sensitivity_comparison(
    "berm_200bp",
    method="backward",
    num_paths=256,
    seed=17,
    shift_size=1.0e-4,
)
display(sens_meta)
display(sens_rows)
nh.plot_bermudan_sensitivity_triplet(
    sens_rows,
    title="berm_200bp: ORE sensitivity analytic vs direct ORE quote bump vs Python quote bump",
)

# %% cell 12
"""
The expected pattern is:

- `python_quote_bump` is close to `ore_direct_quote_bump`
- both can disagree sharply with `ore_sensitivity_csv`

That means the remaining gap is not the Bermudan backward engine. It is the sensitivity-definition layer used by
ORE's analytic.
"""

# %% cell 13
display(sens_config)

focus_cols = ["#TradeId", "Factor", "ScenarioDescription", "Base NPV", "Scenario NPV", "Difference"]
available_cols = [c for c in focus_cols if c in sens_scenario.columns]
if not available_cols:
    available_cols = list(sens_scenario.columns)
display(sens_scenario[available_cols])

# %% cell 14
"""
Read the two ORE tables together:

- `sensitivity_config.csv` shows which internal sensitivity key actually moved
- `scenario.csv` shows the up/down scenario NPVs produced by the analytic

On this case the reported factor label `IndexCurve/EUR-EURIBOR-6M/0/10Y` is associated with an internal node key,
not a literal raw-market-file quote bump. That is why the direct ORE bump and the sensitivity analytic can have
different signs.
"""

# %% cell 15
direct_vs_analytic = sens_rows[
    [
        "normalized_factor",
        "ore_bump_change",
        "ore_direct_quote_bump_change",
        "python_quote_full_bump_change",
        "python_quote_full_minus_ore_direct",
    ]
].copy()
direct_vs_analytic["analytic_minus_direct_ore"] = (
    direct_vs_analytic["ore_bump_change"] - direct_vs_analytic["ore_direct_quote_bump_change"]
)
display(direct_vs_analytic)

# %% cell 16
"""
The forward `EUR 6M 10Y` row is the headline example:

- ORE sensitivity analytic says the bump change is positive
- direct ORE quote bump says the bump change is negative
- Python quote bump is close to direct ORE quote bump

So if the benchmark objective is "match direct market quote bumping", Python is already on the right side of the
comparison. Matching `sensitivity.csv` would require replicating ORE's par-conversion / sensitivity-scenario
machinery, not changing the Bermudan pricer.
"""

# %% cell 17
berm_rank = berm_comp[["case_name", "py_backward_abs_rel_diff"]].copy()
berm_rank["py_backward_abs_rel_diff_bp"] = 1.0e4 * berm_rank["py_backward_abs_rel_diff"]
display(berm_rank)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
nh.plot_bar_frame(
    berm_rank.rename(columns={"py_backward_abs_rel_diff_bp": "abs_rel_diff_bp"}),
    "case_name",
    "abs_rel_diff_bp",
    title="Backward parity gap vs ORE classic (basis points of relative error)",
    color=nh.PALETTE["teal"],
    ax=ax,
)
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 18
"""
## Key takeaways

- Use the stored Bermudan comparison pack for the quick multi-case view.
- Use the direct `berm_200bp` repricing section when you want current diagnostics from the live Python code.
- For sensitivities, distinguish direct quote bumping from ORE's sensitivity analytic. They are not the same calculation.
- Treat `backward` as the ORE-classic parity engine and `lsmc` as the control.
- On these Bermudan cases, `calibration.xml` is the right model source whenever ORE has already produced it.
"""

