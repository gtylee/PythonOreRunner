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
from dataclasses import replace

from native_xva_interface import (
    CollateralBalance,
    CollateralConfig,
    FixingPoint,
    FixingsData,
    MarketData,
    MarketQuote,
    NettingConfig,
    NettingSet,
    XVAConfig,
    map_snapshot,
    DeterministicToyAdapter,
)
from py_ore_tools import validate_xva_snapshot_dataclasses, xva_snapshot_validation_dataframe

# Load an existing repo case, then run the canonical in-memory OreSnapshotApp on buffers.
source_ore_xml = nh.default_live_parity_ore_xml()
snapshot, ore_snapshot, case_meta = nh.load_case_snapshots(source_ore_xml)
app_result, app_meta = nh.run_ore_snapshot_app_case(source_ore_xml, engine="compare", price=True, xva=True, paths=500)
mapped = map_snapshot(snapshot)

print("Source ORE xml:", case_meta["ore_xml"])
print("Input dir:", case_meta["input_dir"])
print("Output dir:", case_meta["output_dir"])
print("OreSnapshotApp elapsed (s):", round(app_meta["elapsed_sec"], 4))
display(nh.snapshot_overview(snapshot))
nh.plot_snapshot_composition(snapshot, title="Fresh loader-backed snapshot: inventory and market mix")

# %% cell 3
"""
## Inputs we reuse from the repo

The notebook starts from a repo-backed ORE case and uses the canonical in-memory `OreSnapshotApp` surface for the
parity workflow. That keeps the demonstration programmatic and avoids creating persistent notebook output folders.

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
## Canonical ORE app run without persistent notebook outputs

The same case can be run through the buffer-based `OreSnapshotApp` API. It materializes the case in a temporary
workspace internally, returns the comparison and validation payloads to Python, and leaves no new repo artifacts
behind.
"""

# %% cell 8
display(nh.ore_snapshot_app_summary_frame(app_result))
display(pd.DataFrame(app_result.comparison_rows))
display(pd.DataFrame(app_result.input_validation_rows).head(12))

# %% cell 9
"""
## Parity completeness audit

A runtime snapshot being usable is not the same thing as a case being parity-ready. The `OreSnapshot` audit is more
strict: it checks whether the run captured enough schedule, curve, credit, funding, and exposure information for a
fair Python-vs-ORE comparison.
"""

# %% cell 10
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

# %% cell 11
"""
The audit table is the one to trust when you want to compare engines. It separates “the case runs” from “the case is
economically lined up well enough to compare specific XVA numbers.”
"""

# %% cell 12
"""
## File-backed validation as a fix list

The XML/file validator is meant to answer a more operational question: if the case is not linked up cleanly,
what exactly should be changed next? The report below is shown as a fix list rather than a generic status dump.
"""

# %% cell 13
from py_ore_tools import validate_ore_input_snapshot, ore_input_validation_dataframe

file_validation = validate_ore_input_snapshot(case_meta["ore_xml"])
file_validation_df = ore_input_validation_dataframe(file_validation)
display(file_validation_df[file_validation_df["section"] == "action_items"])

print("Input links valid:", file_validation["input_links_valid"])
for item in file_validation.get("action_items", []):
    print(f"[{item['severity']}] {item['code']}")
    print("  failed :", item["what_failed"])
    print("  fix    :", item["what_to_fix"])
    print("  where  :", ", ".join(str(x) for x in item.get("where_to_fix", [])))

# %% cell 14
"""
## Dataclass validation: good and bad examples

The file-backed validator checks the ORE XML chain. There is now a second validator for the in-memory dataclasses
themselves. That matters when snapshots are built programmatically, merged, or modified before any engine run.

The good example below is a deliberately well-formed programmatic snapshot. The fresh loader-backed snapshot above
is useful for inspection, but it is not a clean teaching example for this validator because loader bundles can be
operationally valid while still failing stricter in-memory consistency checks.

The bad example breaks a few basic contracts on purpose:
- quote date no longer matches the snapshot asof
- a fixing is placed after the asof date
- the trade references a netting set that is not defined
- collateral references another unknown netting set
- analytics includes an unsupported metric
"""

# %% cell 15
good_snapshot = nh.make_programmatic_snapshot(num_paths=128)
good_report = validate_xva_snapshot_dataclasses(good_snapshot)
good_df = xva_snapshot_validation_dataframe(good_report)
print("Good snapshot valid:", good_report["snapshot_valid"])
print("Good snapshot issues:", good_report["issues"])
display(good_df)
display(good_df[good_df["section"] == "action_items"])

# %% cell 16
bad_snapshot = replace(
    good_snapshot,
    market=replace(
        good_snapshot.market,
        raw_quotes=(
            MarketQuote(
                date="1999-01-01",
                key=good_snapshot.market.raw_quotes[0].key,
                value=good_snapshot.market.raw_quotes[0].value,
            ),
        ) if good_snapshot.market.raw_quotes else (
            MarketQuote(date="1999-01-01", key="FX/RATE/EUR/USD", value=1.10),
        ),
    ),
    fixings=FixingsData(
        points=(FixingPoint(date="2099-01-01", index="USD-LIBOR-3M", value=0.05),)
    ),
    portfolio=replace(
        good_snapshot.portfolio,
        trades=(
            replace(good_snapshot.portfolio.trades[0], netting_set="BROKEN_NS"),
        ) if good_snapshot.portfolio.trades else good_snapshot.portfolio.trades,
    ),
    netting=NettingConfig(netting_sets={"NS_OK": NettingSet(netting_set_id="NS_OK", counterparty="CP_A")}),
    collateral=CollateralConfig(balances=(CollateralBalance(netting_set_id="BROKEN_COLLATERAL", currency=good_snapshot.config.base_currency),)),
    config=replace(good_snapshot.config, analytics=("CVA", "NOT_A_METRIC")),
)

bad_report = validate_xva_snapshot_dataclasses(bad_snapshot)
bad_df = xva_snapshot_validation_dataframe(bad_report)
print("Bad snapshot valid:", bad_report["snapshot_valid"])
print("Bad snapshot issues:", bad_report["issues"])
display(bad_df)
display(bad_df[bad_df["section"] == "action_items"])
for item in bad_report.get("action_items", []):
    print(f"[{item['severity']}] {item['code']}")
    print("  failed :", item["what_failed"])
    print("  fix    :", item["what_to_fix"])
    print("  where  :", ", ".join(str(x) for x in item.get("where_to_fix", [])))

# %% cell 17
# One cheap engine pass shows that the fresh snapshot is immediately executable from Python.
toy_result, toy_elapsed = nh.run_adapter(snapshot, DeterministicToyAdapter())
print(f"Deterministic toy run elapsed: {toy_elapsed:.4f}s")
display(nh.result_metrics_frame(toy_result))

# %% cell 18
"""
## Key takeaways

- The snapshot is the cleanest inspection boundary in the current Python-facing stack.
- The canonical `OreSnapshotApp` path keeps the notebook run in-memory and avoids new persistent output folders.
- The parity audit is the right place to decide what can be compared, not an afterthought.

## Where this connects next

The next notebook zooms into the market side: curve extraction, curve fitting, and how those outputs feed the LGM model.
"""

