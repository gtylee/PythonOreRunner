# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
"""
# 01. Python Dataclasses -> Runtime Inputs

This notebook is the object-model starting point for the series. The canonical end-to-end native workflow lives in
`05_1_python_only_workflow`; this notebook focuses on the question underneath it: what exactly do we hand around
in Python before either engine runs?

**Purpose**
- show the shape of the `XVASnapshot` object
- prove that the snapshot survives serialization without changing meaning
- show how the same snapshot is mapped into ORE-style runtime inputs
- set up the object model used by the Python-only workflow notebook

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
"""
## Inputs we reuse from the repo

This notebook stays close to the tested dataclass and adapter surface already used elsewhere in the repo:
- `native_xva_interface/tests/test_step1_dataclasses.py`
- `native_xva_interface/tests/test_ore_swig_adapter.py`
- `native_xva_interface/notebooks/programmatic_ore_swig_demo.py`
"""

# %% cell 3
from dataclasses import replace

from native_xva_interface import (
    XVASnapshot,
    XVAEngine,
    Trade,
    FXForward,
    map_snapshot,
    DeterministicToyAdapter,
    ORESwigAdapter,
)

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
## The same API with a real ORE bundle

The programmatic snapshot is the cleanest place to start, but the same `XVASnapshot` type is also what the
loader produces from a real ORE input directory. This mirrors the "minimal real bundle" check in
`PythonIntegration/native_xva_interface/demo_achieved.py`.
"""

# %% cell 11
loaded_snapshot = nh.load_base_snapshot(num_paths=32)
loaded_result, loaded_elapsed = nh.run_adapter(loaded_snapshot, DeterministicToyAdapter())

loaded_summary = pd.DataFrame(
    [
        {"field": "source path", "value": getattr(loaded_snapshot.config.source_meta, "path", "") or "(in-memory)"},
        {"field": "asof", "value": loaded_snapshot.config.asof},
        {"field": "base_currency", "value": loaded_snapshot.config.base_currency},
        {"field": "trades", "value": len(loaded_snapshot.portfolio.trades)},
        {"field": "market quotes", "value": len(loaded_snapshot.market.raw_quotes)},
        {"field": "fixings", "value": len(loaded_snapshot.fixings.points)},
        {"field": "num_paths", "value": loaded_snapshot.config.num_paths},
    ]
)
display(loaded_summary)
print(f"Loaded snapshot toy elapsed: {loaded_elapsed:.4f}s")
display(nh.result_metrics_frame(loaded_result))

# %% cell 12
"""
That is the first architectural payoff: the notebook does not need a separate object model for "real ORE files"
versus "hand-built Python trades". Both paths converge to the same runtime contract.
"""

# %% cell 13
"""
## Incremental workflow on top of the snapshot

`demo_small_xva.py` is older and uses a different API shape, but the workflow idea still applies: build the
state once, then apply targeted market and portfolio changes. In the current interface this is handled by
`XVAEngine.create_session(...)` plus `update_market(...)` / `update_portfolio(...)`.
"""

# %% cell 14
session = XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot)
base_session_result = session.run(return_cubes=False)

bumped_quotes = []
for quote in snapshot.market.raw_quotes:
    if quote.key == "ZERO/RATE/EUR/1Y":
        bumped_quotes.append(replace(quote, value=quote.value + 0.0010))
    else:
        bumped_quotes.append(quote)
session.update_market(replace(snapshot.market, raw_quotes=tuple(bumped_quotes)))
after_market_result = session.run(return_cubes=False)

session.update_portfolio(
    add=[
        Trade(
            trade_id="FXFWD_PATCH_1",
            counterparty="CP_A",
            netting_set="NS_EUR",
            trade_type="FxForward",
            product=FXForward(
                pair="EURUSD",
                notional=1_000_000,
                strike=1.08,
                maturity_years=0.5,
                buy_base=False,
            ),
        )
    ],
    amend=[("FXFWD_DEMO_1", {"product": {"strike": 1.10}})],
)
after_portfolio_result = session.run(return_cubes=False)

incremental = pd.DataFrame(
    [
        {
            "run": "base session",
            "pv": base_session_result.pv_total,
            "xva_total": base_session_result.xva_total,
            "rebuild_counts": str(base_session_result.metadata.get("rebuild_counts", {})),
        },
        {
            "run": "after market update",
            "pv": after_market_result.pv_total,
            "xva_total": after_market_result.xva_total,
            "rebuild_counts": str(after_market_result.metadata.get("rebuild_counts", {})),
        },
        {
            "run": "after portfolio update",
            "pv": after_portfolio_result.pv_total,
            "xva_total": after_portfolio_result.xva_total,
            "rebuild_counts": str(after_portfolio_result.metadata.get("rebuild_counts", {})),
        },
    ]
)
display(incremental)

# %% cell 15
"""
The exact numbers here come from the toy adapter, so the point is not market realism. The point is that the
snapshot is not just a static serialization object; it is also the unit of incremental session updates.
"""

# %% cell 16
"""
## Optional SWIG run

This is intentionally a boundary check, not a parity claim. The point is that the same snapshot can be handed to
an ORE-backed adapter without rewriting the notebook around engine-specific inputs.
"""

# %% cell 17
swig = nh.swig_status()
print(swig["message"])

if swig["available"]:
    swig_result, swig_elapsed = nh.run_adapter(snapshot, ORESwigAdapter())
    print(f"ORE-SWIG elapsed: {swig_elapsed:.4f}s")
    display(nh.result_metrics_frame(swig_result))
else:
    print("Skipping ORE-SWIG adapter execution in this environment.")

# %% cell 18
"""
## Key takeaways

- The dataclass layer already carries enough structure to be the common hand-off point for both engines.
- The same snapshot contract works for a hand-built programmatic demo and for a snapshot loaded from real ORE files.
- Session updates build on the same object model, so market and portfolio changes do not require a different API tier.
- `stable_key()` matters because notebooks and tests need a cheap way to detect a real change in economic content.
- `map_snapshot()` is the explicit boundary where Python objects turn into ORE-style runtime payloads.

## Where this connects next

The next notebook moves from a hand-built snapshot to a loader-backed snapshot built from a fresh ORE run.
"""
