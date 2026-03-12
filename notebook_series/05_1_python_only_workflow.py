# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 05_1. Python-Only Workflow

This companion notebook replaces the mixed-engine workflow with a pure Python session workflow. The goal is to show
what can already be done interactively with the native snapshot model and the Python LGM adapter alone.

**Purpose**
- run the Python LGM adapter on a programmatic snapshot
- show how market and portfolio updates flow through one session
- keep the workflow focused on reusable Python objects rather than external runs

**What you will learn**
- how to launch a Python-only XVA session
- how to compare base, market-bumped, and portfolio-updated runs
- which pieces of the workflow are already interactive without any external engine
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
    raise RuntimeError("Could not locate the Engine repo root from the current notebook working directory")

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
from dataclasses import replace

from native_xva_interface import FXForward, PythonLgmAdapter, Trade, XVAEngine

snapshot = nh.make_programmatic_snapshot(num_paths=768)
adapter = PythonLgmAdapter(fallback_to_swig=False)
session = XVAEngine(adapter=adapter).create_session(snapshot)

base_result = session.run(return_cubes=False)
display(nh.snapshot_overview(snapshot))
display(nh.trade_frame(snapshot))
display(nh.quote_family_frame(snapshot))
display(nh.result_metrics_frame(base_result))

# %% cell 3
"""
## Base run

The base snapshot is intentionally small. That keeps the session updates easy to reason about and makes it obvious
which change caused which metric move.
"""

# %% cell 4
nh.plot_snapshot_composition(snapshot, title="Python-only session: snapshot inventory and quote mix")

# %% cell 5
"""
## Market update

First change a small set of market quotes and rerun the same session. This isolates the market sensitivity of the
current portfolio without rebuilding a new notebook state from scratch.
"""

# %% cell 6
bumped_quotes = []
for quote in snapshot.market.raw_quotes:
    if quote.key == "ZERO/RATE/EUR/1Y":
        bumped_quotes.append(replace(quote, value=quote.value + 0.0010))
    elif quote.key == "FX/EUR/USD":
        bumped_quotes.append(replace(quote, value=quote.value + 0.0150))
    else:
        bumped_quotes.append(quote)
bumped_market = replace(snapshot.market, raw_quotes=tuple(bumped_quotes))
session.update_market(bumped_market)
market_result = session.run(return_cubes=False)

market_compare = nh.compare_results_frame("base", base_result, "market_bump", market_result)
display(market_compare)
nh.plot_metric_comparison(market_compare, "base", "market_bump", title="Python-only session: base vs market bump")
nh.plot_metric_delta(market_compare, title="Python-only session: market bump minus base")

# %% cell 7
"""
The delta view is more informative than the level view once the baseline is familiar. It shows which metrics are
genuinely moving and which are mostly stable to this small bump.
"""

# %% cell 8
"""
## Portfolio update

Next add one trade and reprice through the same Python session. This is the workflow analogue of a trader-side
portfolio patch rather than a market move.
"""

# %% cell 9
session.update_market(snapshot.market)
session.update_portfolio(
    add=[
        Trade(
            trade_id="FXFWD_PATCH_2",
            counterparty="CP_A",
            netting_set="NS_EUR",
            trade_type="FxForward",
            product=FXForward(
                pair="EURUSD",
                notional=1_500_000,
                strike=1.12,
                maturity_years=1.5,
                buy_base=False,
            ),
        )
    ]
)
portfolio_result = session.run(return_cubes=False)

portfolio_compare = nh.compare_results_frame("base", base_result, "portfolio_patch", portfolio_result)
display(portfolio_compare)
nh.plot_metric_comparison(portfolio_compare, "base", "portfolio_patch", title="Python-only session: base vs portfolio patch")
nh.plot_metric_delta(portfolio_compare, title="Python-only session: portfolio patch minus base")

# %% cell 10
workflow_summary = pd.DataFrame(
    [
        {"run": "base", "pv": base_result.pv_total, "xva_total": base_result.xva_total},
        {"run": "market_bump", "pv": market_result.pv_total, "xva_total": market_result.xva_total},
        {"run": "portfolio_patch", "pv": portfolio_result.pv_total, "xva_total": portfolio_result.xva_total},
    ]
)
display(workflow_summary)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
nh.plot_bar_frame(workflow_summary, "run", "pv", title="PV by workflow step", color=nh.PALETTE["blue"], ax=axes[0])
nh.plot_bar_frame(workflow_summary, "run", "xva_total", title="XVA total by workflow step", color=nh.PALETTE["gold"], ax=axes[1])
plt.tight_layout()
plt.show()
plt.close(fig)

# %% cell 11
"""
## Capabilities in this workflow

The capability table is still useful here, but it is read purely as a Python-side checklist rather than a split
between engines.
"""

# %% cell 12
capability_df = pd.DataFrame(
    [
        {"capability": "Programmatic snapshot build", "python_only": True},
        {"capability": "Session market updates", "python_only": True},
        {"capability": "Session portfolio updates", "python_only": True},
        {"capability": "Pathwise exposure and XVA metrics", "python_only": True},
        {"capability": "Fresh external run required", "python_only": False},
    ]
)
display(capability_df)
nh.plot_boolean_matrix(capability_df, row_col="capability", value_cols=["python_only"], title="Python-only workflow capabilities")

# %% cell 13
"""
## Key takeaways

- The native Python session is already enough for interactive market and portfolio iteration.
- Base, market-bumped, and portfolio-patched runs are easiest to compare in one persistent session.
- The notebook is most useful as a workflow demo when the snapshot stays small and explicit.
"""

