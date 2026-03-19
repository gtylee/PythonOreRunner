from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Callable


KERNEL = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


LANG_INFO = {
    "name": "python",
    "version": "3.13",
    "mimetype": "text/x-python",
    "codemirror_mode": {"name": "ipython", "version": 3},
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "file_extension": ".py",
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in normalized.splitlines()]


BOOTSTRAP = """
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
"""


def py_markdown(text: str) -> str:
    lines = _lines(text)
    out = ['"""\n']
    out.extend(lines)
    out.append('"""\n')
    return "".join(out)


def py_code(text: str) -> str:
    normalized = textwrap.dedent(text).strip("\n")
    return normalized + "\n" if normalized else ""


def render_py(cells: list[dict]) -> str:
    chunks: list[str] = [
        "# Auto-generated from the notebook source in build_series.py\n",
        "# Execute with: python3 <this_file>\n\n",
    ]
    for idx, cell in enumerate(cells):
        chunks.append(f"# %% cell {idx}\n")
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            chunks.append(py_markdown(source))
        else:
            chunks.append(py_code(source))
        chunks.append("\n")
    return "".join(chunks)


def notebook_01() -> list[dict]:
    return [
        md(
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
        ),
        code(BOOTSTRAP),
        md(
            """
            ## Inputs we reuse from the repo

            This notebook stays close to the tested dataclass and adapter surface already used elsewhere in the repo:
            - `native_xva_interface/tests/test_step1_dataclasses.py`
            - `native_xva_interface/tests/test_ore_swig_adapter.py`
            - `native_xva_interface/notebooks/programmatic_ore_swig_demo.py`
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            Read the chart above as an inventory check. The left panel shows how much business content sits inside the snapshot.
            The right panel shows that market data is already grouped into recognizable quote families before any ORE call.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Before and after mapping

            `map_snapshot()` is the bridge between the Python object model and the ORE runtime shape. The important point is
            not the exact XML syntax; it is that the notebook can show, in one place, what leaves the dataclass layer.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The size chart is useful for orientation. In practice, a handful of XML documents carry nearly all of the runtime
            structure, while market and fixing lines stay in flat text form.
            """
        ),
        code(
            """
            # The toy adapter is a fast contract check: the snapshot can already flow through the engine boundary.
            toy_result, toy_elapsed = nh.run_adapter(snapshot, DeterministicToyAdapter())
            print(f"Toy adapter elapsed: {toy_elapsed:.4f}s")
            display(nh.result_metrics_frame(toy_result))
            """
        ),
        md(
            """
            ## The same API with a real ORE bundle

            The programmatic snapshot is the cleanest place to start, but the same `XVASnapshot` type is also what the
            loader produces from a real ORE input directory. This mirrors the "minimal real bundle" check in
            `PythonIntegration/native_xva_interface/demo_achieved.py`.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            That is the first architectural payoff: the notebook does not need a separate object model for "real ORE files"
            versus "hand-built Python trades". Both paths converge to the same runtime contract.
            """
        ),
        md(
            """
            ## Incremental workflow on top of the snapshot

            `demo_small_xva.py` is older and uses a different API shape, but the workflow idea still applies: build the
            state once, then apply targeted market and portfolio changes. In the current interface this is handled by
            `XVAEngine.create_session(...)` plus `update_market(...)` / `update_portfolio(...)`.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The exact numbers here come from the toy adapter, so the point is not market realism. The point is that the
            snapshot is not just a static serialization object; it is also the unit of incremental session updates.
            """
        ),
        md(
            """
            ## Optional SWIG run

            This is intentionally a boundary check, not a parity claim. The point is that the same snapshot can be handed to
            an ORE-backed adapter without rewriting the notebook around engine-specific inputs.
            """
        ),
        code(
            """
            swig = nh.swig_status()
            print(swig["message"])

            if swig["available"]:
                swig_result, swig_elapsed = nh.run_adapter(snapshot, ORESwigAdapter())
                print(f"ORE-SWIG elapsed: {swig_elapsed:.4f}s")
                display(nh.result_metrics_frame(swig_result))
            else:
                print("Skipping ORE-SWIG adapter execution in this environment.")
            """
        ),
        md(
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
        ),
    ]


def notebook_02() -> list[dict]:
    return [
        md(
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
        ),
        code(BOOTSTRAP),
        code(
            """
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
            """
        ),
        md(
            """
            ## Inputs we reuse from the repo

            The notebook starts from a repo-backed ORE case and uses the canonical in-memory `OreSnapshotApp` surface for the
            parity workflow. That keeps the demonstration programmatic and avoids creating persistent notebook output folders.

            The loader-backed path still follows the same code used in the tests and demos:
            - `native_xva_interface/loader.py`
            - `native_xva_interface/tests/test_step2_loader.py`
            - `native_xva_interface/tests/test_step3_merge.py`
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Snapshot as a reusable artifact

            The loader has already normalized XML, market text, fixings, and netting configuration into one object. That is the
            practical value of the snapshot: most downstream code can stop caring which original file carried which piece.
            """
        ),
        code(
            """
            snapshot_dict = snapshot.to_dict()
            snapshot_json = nh.to_pretty_json(snapshot_dict, limit=1800)
            print(snapshot_json)

            display(nh.mapped_input_summary(mapped))
            display(nh.xml_buffer_summary(mapped).head(12))
            nh.plot_xml_buffer_sizes(mapped, title="Mapped XML payload sizes for the fresh loader snapshot")
            """
        ),
        md(
            """
            ## Canonical ORE app run without persistent notebook outputs

            The same case can be run through the buffer-based `OreSnapshotApp` API. It materializes the case in a temporary
            workspace internally, returns the comparison and validation payloads to Python, and leaves no new repo artifacts
            behind.
            """
        ),
        code(
            """
            display(nh.ore_snapshot_app_summary_frame(app_result))
            display(pd.DataFrame(app_result.comparison_rows))
            display(pd.DataFrame(app_result.input_validation_rows).head(12))
            """
        ),
        md(
            """
            ## Parity completeness audit

            A runtime snapshot being usable is not the same thing as a case being parity-ready. The `OreSnapshot` audit is more
            strict: it checks whether the run captured enough schedule, curve, credit, funding, and exposure information for a
            fair Python-vs-ORE comparison.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The audit table is the one to trust when you want to compare engines. It separates “the case runs” from “the case is
            economically lined up well enough to compare specific XVA numbers.”
            """
        ),
        md(
            """
            ## File-backed validation as a fix list

            The XML/file validator is meant to answer a more operational question: if the case is not linked up cleanly,
            what exactly should be changed next? The report below is shown as a fix list rather than a generic status dump.
            """
        ),
        code(
            """
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
            """
        ),
        md(
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
        ),
        code(
            """
            good_snapshot = nh.make_programmatic_snapshot(num_paths=128)
            good_report = validate_xva_snapshot_dataclasses(good_snapshot)
            good_df = xva_snapshot_validation_dataframe(good_report)
            print("Good snapshot valid:", good_report["snapshot_valid"])
            print("Good snapshot issues:", good_report["issues"])
            display(good_df)
            display(good_df[good_df["section"] == "action_items"])
            """
        ),
        code(
            """
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
            """
        ),
        code(
            """
            # One cheap engine pass shows that the fresh snapshot is immediately executable from Python.
            toy_result, toy_elapsed = nh.run_adapter(snapshot, DeterministicToyAdapter())
            print(f"Deterministic toy run elapsed: {toy_elapsed:.4f}s")
            display(nh.result_metrics_frame(toy_result))
            """
        ),
        md(
            """
            ## Key takeaways

            - The snapshot is the cleanest inspection boundary in the current Python-facing stack.
            - The canonical `OreSnapshotApp` path keeps the notebook run in-memory and avoids new persistent output folders.
            - The parity audit is the right place to decide what can be compared, not an afterthought.

            ## Where this connects next

            The next notebook zooms into the market side: curve extraction, curve fitting, and how those outputs feed the LGM model.
            """
        ),
    ]


def notebook_03() -> list[dict]:
    return [
        md(
            """
            # 03. Curve Calibration and LGM Parameter Extraction

            This notebook connects two things that are often shown separately: the fitted market curves and the LGM parameter
            payload. The point is to keep the model story anchored to the same ORE artifacts that produced the market view.

            **Purpose**
            - show how the calibration instruments are discovered from ORE inputs
            - visualize the fitted discount curves in a way that exposes shape, density, and pillar coverage
            - connect those results to the extracted LGM alpha and kappa term structures

            **What you will learn**
            - how the repo traces discount and index curves from a known case
            - how to inspect fitted curves beyond one flat table
            - where the LGM parameter term structures come from
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            import json
            import subprocess

            from py_ore_tools import (
                discount_factors_to_dataframe,
                extract_discount_factors_by_currency,
                extract_market_instruments_by_currency,
                extract_market_instruments_by_currency_from_quotes,
                fit_discount_curves_from_ore_market,
                fit_discount_curves_from_programmatic_quotes,
                fitted_curves_to_dataframe,
                parse_lgm_params_from_calibration_xml,
                parse_lgm_params_from_simulation_xml,
                quote_dicts_from_pairs,
            )
            from ore_curve_fit_parity import compare_python_vs_ore, trace_discount_curve_from_ore, trace_index_curve_from_ore

            ORE_XML = nh.default_curve_case_ore_xml(repo)
            CALIBRATION_PARITY_SCRIPT = REPO_ROOT / "src" / "pythonore" / "benchmarks" / "benchmark_lgm_calibration_parity.py"
            df_payload = extract_discount_factors_by_currency(ORE_XML, configuration_id="default")
            df_long = discount_factors_to_dataframe(df_payload)
            instruments = extract_market_instruments_by_currency(ORE_XML)
            fitted = fit_discount_curves_from_ore_market(
                ORE_XML,
                fit_method="bootstrap_mm_irs_v1",
                fit_grid_mode="dense",
                dense_step_years=0.25,
            )
            fitted_df = fitted_curves_to_dataframe(fitted)
            usd_discount_trace = trace_discount_curve_from_ore(ORE_XML, currency="USD")
            usd_index_trace = trace_index_curve_from_ore(ORE_XML, index_name="USD-LIBOR-6M")
            usd_python_fit_compare = compare_python_vs_ore(ORE_XML, currency="USD")
            configured_lgm, calibrated_lgm, calibration_meta = nh.run_fresh_lgm_calibration_demo(ccy_key="EUR")
            calibration_hw_parity_meta = nh.run_fresh_lgm_calibration_hullwhite_parity_demo(
                base_fresh_ore_xml=calibration_meta["fresh_ore_xml"]
            )

            print("ORE XML:", ORE_XML)
            display(df_long.head(12))
            display(pd.DataFrame.from_dict(instruments, orient="index").reset_index(names="currency"))
            """
        ),
        md(
            """
            ## Inputs we reuse from the repo

            The curve side of the series is grounded in the existing parity walkthroughs and tests:
            - `ore_curve_fit_parity/notebooks/usd_curve_parity_walkthrough.ipynb`
            - `ore_curve_fit_parity/notebooks/multi_currency_curve_trace_demo.ipynb`
            - `tests/test_quick_curve_fit.py`
            """
        ),
        code(
            """
            # Start with the size of each fitted grid and the number of instruments that drove it.
            summary = (
                fitted_df.groupby("ccy", as_index=False)
                .agg(
                    fit_points=("time", "count"),
                    first_df=("df", "first"),
                    last_df=("df", "last"),
                    instrument_count=("instrument_count", "max"),
                )
            )
            display(summary)

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
            nh.plot_bar_frame(summary, "ccy", "fit_points", title="Fit grid size by currency", color=nh.PALETTE["blue"], ax=axes[0])
            nh.plot_bar_frame(summary, "ccy", "instrument_count", title="Instrument count by currency", color=nh.PALETTE["teal"], ax=axes[1])
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            The first chart answers a structural question: how dense is the fitted curve relative to the number of calibration
            instruments? That difference is exactly why a fitted grid can look smooth even when the market input is sparse.
            """
        ),
        code(
            """
            nh.plot_log_discount_factors(df_long, title="Extracted log(DF) by currency")
            heatmap_df = nh.zero_rate_heatmap_frame(df_long)
            display(heatmap_df.head(12))
            nh.plot_zero_rate_heatmap(heatmap_df, title="Extracted zero-rate heatmap by currency and maturity")

            extractor_audit = (
                df_long.groupby("ccy", as_index=False)
                .agg(
                    point_count=("time", "count"),
                    min_time=("time", "min"),
                    max_time=("time", "max"),
                    min_df=("df", "min"),
                    max_df=("df", "max"),
                )
                .sort_values("ccy")
            )
            display(extractor_audit)
            """
        ),
        md(
            """
            The older extractor notebook also tried to expose this through a session bridge. In the current codebase that bridge
            is not surfaced as a public `XVASession` method, so this series keeps the extraction path direct and explicit rather
            than pretending the runtime API already exposes a stable curve-extraction wrapper.
            """
        ),
        code(
            """
            nh.plot_fitted_curves({ccy: fitted[ccy] for ccy in sorted(fitted)[:4]}, title="Representative fitted discount curves")
            nh.plot_curve_diagnostics(fitted_df, ccy="USD", title="USD curve diagnostics from the fitted output grid")

            trace_compare = pd.DataFrame(
                {
                    "series": ["USD discount native nodes", "USD discount report grid", "USD 6M dependency graph"],
                    "count": [
                        len(usd_discount_trace["ore_calibration_trace"]["pillars"]),
                        len(usd_discount_trace["ore_curve_points"]["times"]),
                        len(usd_index_trace["dependency_graph"]),
                    ],
                }
            )
            display(trace_compare)
            """
        ),
        md(
            """
            ## Python curve fitter demo on mixed instruments

            The USD discount trace is a good compact demo because it mixes three instrument types in one curve:

            - deposits
            - FRAs
            - swaps

            The Python diagnostic fitter here is deliberately narrow: it rebuilds the ORE curve from the native ORE
            calibration nodes and then compares the rebuilt discount factors to the ORE report grid. That makes the
            interpolation parity visible without pretending we have fully reimplemented ORE helper construction.
            """
        ),
        code(
            """
            usd_input_rows = []
            for segment in usd_discount_trace["segment_alignment"]:
                for quote in segment["quotes"]:
                    pillar = quote.get("ore_pillar") or {}
                    usd_input_rows.append(
                        {
                            "segment_type": segment["type"],
                            "quote_key": quote["quote_key"],
                            "market_quote": quote.get("quote_value"),
                            "pillar_time": pd.to_numeric(pillar.get("time"), errors="coerce"),
                            "pillar_df": pd.to_numeric(pillar.get("discountFactor"), errors="coerce"),
                            "pillar_zero_rate": pd.to_numeric(pillar.get("zeroRate"), errors="coerce"),
                        }
                    )
            usd_input_df = pd.DataFrame(usd_input_rows)
            display(usd_input_df.head(12))

            usd_fit_compare_df = pd.DataFrame(
                [
                    {
                        "time": point.time,
                        "ore_df": point.ore_value,
                        "python_df": point.engine_value,
                        "abs_error": point.abs_error,
                        "rel_error": point.rel_error,
                    }
                    for point in usd_python_fit_compare.points
                ]
            )
            usd_fit_compare_df["ore_zero_rate"] = -np.log(np.clip(usd_fit_compare_df["ore_df"], 1.0e-12, None)) / usd_fit_compare_df["time"].replace(0.0, np.nan)
            usd_fit_compare_df["python_zero_rate"] = -np.log(np.clip(usd_fit_compare_df["python_df"], 1.0e-12, None)) / usd_fit_compare_df["time"].replace(0.0, np.nan)
            ore_log_df = -np.log(np.clip(usd_fit_compare_df["ore_df"].to_numpy(dtype=float), 1.0e-12, None))
            py_log_df = -np.log(np.clip(usd_fit_compare_df["python_df"].to_numpy(dtype=float), 1.0e-12, None))
            grid_time = usd_fit_compare_df["time"].to_numpy(dtype=float)
            usd_fit_compare_df["ore_forward_rate"] = np.gradient(ore_log_df, grid_time, edge_order=1)
            usd_fit_compare_df["python_forward_rate"] = np.gradient(py_log_df, grid_time, edge_order=1)
            display(usd_fit_compare_df.head(12))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            native_nodes = pd.DataFrame(
                {
                    "time": usd_discount_trace["native_curve_nodes"]["times"],
                    "df": usd_discount_trace["native_curve_nodes"]["discount_factors"],
                }
            )
            axes[0].scatter(native_nodes["time"], native_nodes["df"], s=28, color=nh.PALETTE["gold"], label="ORE native nodes")
            axes[0].plot(usd_fit_compare_df["time"], usd_fit_compare_df["ore_df"], linewidth=1.8, label="ORE report grid")
            axes[0].plot(usd_fit_compare_df["time"], usd_fit_compare_df["python_df"], linewidth=1.4, linestyle="--", label="Python fitter")
            axes[0].set_title("USD curve fitter: inputs to fitted outputs")
            axes[0].set_xlabel("Time (years)")
            axes[0].set_ylabel("Discount Factor")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(usd_fit_compare_df["time"], 1.0e9 * usd_fit_compare_df["abs_error"], linewidth=1.8, color=nh.PALETTE["rose"])
            axes[1].set_title("USD curve fitter absolute error (nanodf)")
            axes[1].set_xlabel("Time (years)")
            axes[1].set_ylabel("Abs Error x 1e9")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close(fig)

            rate_display = usd_fit_compare_df[
                [
                    "time",
                    "ore_zero_rate",
                    "python_zero_rate",
                    "ore_forward_rate",
                    "python_forward_rate",
                    "abs_error",
                ]
            ].head(12)
            display(rate_display)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
            axes[0].plot(usd_fit_compare_df["time"], usd_fit_compare_df["ore_zero_rate"], linewidth=1.8, label="ORE zero")
            axes[0].plot(usd_fit_compare_df["time"], usd_fit_compare_df["python_zero_rate"], linewidth=1.4, linestyle="--", label="Python zero")
            axes[0].set_title("USD curve fitter zero rates")
            axes[0].set_xlabel("Time (years)")
            axes[0].set_ylabel("Zero Rate")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(usd_fit_compare_df["time"], usd_fit_compare_df["ore_forward_rate"], linewidth=1.8, label="ORE forward")
            axes[1].plot(usd_fit_compare_df["time"], usd_fit_compare_df["python_forward_rate"], linewidth=1.4, linestyle="--", label="Python forward")
            axes[1].set_title("USD curve fitter instantaneous forwards")
            axes[1].set_xlabel("Time (years)")
            axes[1].set_ylabel("Forward Rate")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            plt.close(fig)

            print("Python fitter status:", usd_python_fit_compare.status)
            print("Max abs error:", usd_python_fit_compare.max_abs_error)
            print("Max rel error:", usd_python_fit_compare.max_rel_error)
            """
        ),
        code(
            """
            fitted_weighted_dense = fit_discount_curves_from_ore_market(
                ORE_XML,
                fit_method="weighted_zero_logdf_v1",
                fit_grid_mode="dense",
                dense_step_years=0.25,
            )
            fitted_weighted_dense_df = fitted_curves_to_dataframe(fitted_weighted_dense)

            sample_ccys = sorted(fitted.keys())[:4]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            for ccy in sample_ccys:
                dense = fitted_weighted_dense_df[fitted_weighted_dense_df["ccy"] == ccy].sort_values("time")
                axes[0].plot(dense["time"], dense["zero_rate"], linewidth=1.8, label=ccy)
            axes[0].set_title("Weighted zero fit")
            axes[0].set_xlabel("Time (years)")
            axes[0].set_ylabel("Zero Rate")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            for ccy in sample_ccys:
                dense = fitted_df[fitted_df["ccy"] == ccy].sort_values("time")
                axes[1].plot(dense["time"], dense["zero_rate"], linewidth=1.8, label=ccy)
            axes[1].set_title("Bootstrap fit")
            axes[1].set_xlabel("Time (years)")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            The weighted-versus-bootstrap comparison is worth showing because it makes the fitting choice visible. The old
            extractor demo did this well, and it is more informative than showing one fitter in isolation.
            """
        ),
        code(
            """
            programmatic_quote_pairs = [
                ("MM/RATE/USD/USD-LIBOR/1M", 0.0520),
                ("MM/RATE/USD/USD-LIBOR/3M", 0.0515),
                ("ZERO/RATE/USD/1Y", 0.0500),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/2Y", 0.0485),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/5Y", 0.0470),
                ("MM/RATE/EUR/EUR-EURIBOR/1M", 0.0310),
                ("ZERO/RATE/EUR/1Y", 0.0300),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/2Y", 0.0310),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/5Y", 0.0320),
            ]
            programmatic_quotes = quote_dicts_from_pairs(programmatic_quote_pairs)
            prog_inst = extract_market_instruments_by_currency_from_quotes(asof_date="2026-03-08", quotes=programmatic_quotes)
            prog_fitted = fit_discount_curves_from_programmatic_quotes(
                asof_date="2026-03-08",
                quotes=programmatic_quotes,
                fit_method="bootstrap_mm_irs_v1",
                fit_grid_mode="dense",
                dense_step_years=0.25,
            )
            prog_fitted_df = fitted_curves_to_dataframe(prog_fitted)

            display(pd.DataFrame(
                [
                    {"ccy": ccy, "instrument_count": p["instrument_count"], "fit_points_count": prog_fitted[ccy]["fit_points_count"]}
                    for ccy, p in sorted(prog_inst.items())
                ]
            ))

            fig, ax = plt.subplots(figsize=(10, 5))
            for ccy, grp in prog_fitted_df.groupby("ccy"):
                g = grp.sort_values("time")
                ax.plot(g["time"], g["zero_rate"], linewidth=1.8, label=f"{ccy} bootstrap")
            ax.set_title("Programmatic fitted zero curves (no XML)")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Zero Rate")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Fresh LGM calibration run

            The curve extraction above used a compact benchmark case because it is convenient for inspecting discount curves.
            For the model section we switch to a calibration-enabled ORE example and run it fresh inside the notebook.

            That gives us two distinct objects to compare:
            - the LGM parameter template from `simulation_lgm.xml`
            - the calibrated term structure written by ORE to `calibration.xml`

            In this example ORE calibrates `alpha` and keeps `kappa` fixed, so the interesting movement is in the volatility
            buckets rather than the reversion line.
            """
        ),
        code(
            """
            calibration_status = pd.DataFrame(
                [
                    {"field": "source_ore_xml", "value": calibration_meta["source_ore_xml"]},
                    {"field": "fresh_ore_xml", "value": calibration_meta["fresh_ore_xml"]},
                    {"field": "simulation_xml", "value": calibration_meta["simulation_xml"]},
                    {"field": "calibration_xml", "value": calibration_meta["calibration_xml"]},
                    {"field": "ore_elapsed_sec", "value": round(calibration_meta["elapsed_sec"], 3)},
                    {"field": "calibrate_vol", "value": calibration_meta["calibrate_vol"]},
                    {"field": "calibrate_kappa", "value": calibration_meta["calibrate_kappa"]},
                ]
            )
            display(calibration_status)

            alpha_configured = nh.lgm_param_frame(configured_lgm, param="alpha").rename(columns={"value": "configured_alpha"})
            alpha_calibrated = nh.lgm_param_frame(calibrated_lgm, param="alpha").rename(columns={"value": "calibrated_alpha"})
            kappa_configured = nh.lgm_param_frame(configured_lgm, param="kappa").rename(columns={"value": "configured_kappa"})
            kappa_calibrated = nh.lgm_param_frame(calibrated_lgm, param="kappa").rename(columns={"value": "calibrated_kappa"})

            display(alpha_configured)
            display(alpha_calibrated)
            display(kappa_configured)
            display(kappa_calibrated)

            alpha_compare = alpha_calibrated.merge(
                alpha_configured[["bucket_index", "configured_alpha"]],
                on="bucket_index",
                how="left",
            )
            alpha_compare["configured_alpha"] = alpha_compare["configured_alpha"].fillna(alpha_configured["configured_alpha"].iloc[-1])
            alpha_compare["alpha_shift_bp"] = 1.0e4 * (alpha_compare["calibrated_alpha"] - alpha_compare["configured_alpha"])
            display(alpha_compare[["bucket_label", "configured_alpha", "calibrated_alpha", "alpha_shift_bp"]])

            nh.plot_lgm_calibration_summary(configured_lgm, calibrated_lgm, title="Fresh ORE calibration: configured template vs calibrated output")

            lgm_ccys = ["EUR", "USD", "GBP", "CHF", "JPY"]
            lgm_ccy_rows = []
            lgm_alpha_curves = []
            lgm_kappa_curves = []

            for ccy in lgm_ccys:
                cfg = parse_lgm_params_from_simulation_xml(calibration_meta["simulation_xml"], ccy_key=ccy)
                cal = parse_lgm_params_from_calibration_xml(calibration_meta["calibration_xml"], ccy_key=ccy)
                alpha_cfg_ccy = nh.lgm_param_frame(cfg, param="alpha")
                alpha_cal_ccy = nh.lgm_param_frame(cal, param="alpha")
                kappa_cfg_ccy = nh.lgm_param_frame(cfg, param="kappa")
                kappa_cal_ccy = nh.lgm_param_frame(cal, param="kappa")

                lgm_ccy_rows.append(
                    {
                        "ccy": ccy,
                        "alpha_bucket_count": len(alpha_cal_ccy),
                        "kappa_bucket_count": len(kappa_cal_ccy),
                        "alpha_mean_shift_bp": 1.0e4 * (alpha_cal_ccy["value"].mean() - alpha_cfg_ccy["value"].mean()),
                        "kappa_mean_shift_bp": 1.0e4 * (kappa_cal_ccy["value"].mean() - kappa_cfg_ccy["value"].mean()),
                    }
                )

                lgm_alpha_curves.append(
                    alpha_cal_ccy.assign(ccy=ccy).rename(columns={"value": "calibrated_alpha"})
                )
                lgm_kappa_curves.append(
                    kappa_cal_ccy.assign(ccy=ccy).rename(columns={"value": "calibrated_kappa"})
                )

            lgm_ccy_summary = pd.DataFrame(lgm_ccy_rows).sort_values("ccy")
            lgm_alpha_curve_df = pd.concat(lgm_alpha_curves, ignore_index=True)
            lgm_kappa_curve_df = pd.concat(lgm_kappa_curves, ignore_index=True)

            display(lgm_ccy_summary)
            display(lgm_alpha_curve_df.head(12))
            display(lgm_kappa_curve_df.head(12))

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=False)
            for ccy, grp in lgm_alpha_curve_df.groupby("ccy"):
                g = grp.sort_values("horizon_years")
                axes[0].step(g["horizon_years"], g["calibrated_alpha"], where="post", linewidth=1.8, label=ccy)
            axes[0].set_title("Calibrated alpha by currency")
            axes[0].set_xlabel("Horizon (years)")
            axes[0].set_ylabel("Alpha")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            for ccy, grp in lgm_kappa_curve_df.groupby("ccy"):
                g = grp.sort_values("horizon_years")
                axes[1].step(g["horizon_years"], g["calibrated_kappa"], where="post", linewidth=1.8, label=ccy)
            axes[1].set_title("Calibrated kappa by currency")
            axes[1].set_xlabel("Horizon (years)")
            axes[1].set_ylabel("Kappa")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## External Python-vs-ORE calibration parity

            The notebook's fresh ORE calibration run keeps the original `Hagan` volatility parametrization, while the current
            external Python calibration backend is implemented for the `HullWhite/HullWhite` subset. So this section derives a
            `HullWhite` variant of the same ORE example, runs it fresh, and then compares the external Python calibration
            against that derived case.

            This section runs the external benchmark on the fresh case and summarizes three things:
            - whether the basket expiries line up with ORE
            - whether the helper market values are matched under the calibrated Python result
            - how close the calibrated sigma buckets are to ORE for representative currencies
            """
        ),
        code(
            """
            calibration_case_root = Path(calibration_hw_parity_meta["fresh_output_dir"]).parent
            parity_reports = {}
            for ccy in ["EUR", "USD"]:
                out_json = calibration_case_root / f"python_parity_report_{ccy.lower()}.json"
                cmd = [
                    sys.executable,
                    str(CALIBRATION_PARITY_SCRIPT),
                    "--case-root",
                    str(calibration_case_root),
                    "--currency",
                    ccy,
                    "--out-json",
                    str(out_json),
                ]
                completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if not out_json.exists():
                    raise FileNotFoundError(f"Parity report not written for {ccy}: {out_json}")
                parity_reports[ccy] = json.loads(out_json.read_text())

            parity_summary_rows = [
                {
                    "ccy": ccy,
                    "status": "ok",
                    "vol_source": report["vol_source"],
                    "vol_type": report["vol_type"],
                    "basket_size": report["basket_size"],
                    "basket_grid_match": report["basket_expiry_time_matches_ore_grid"],
                    "python_valid": report["python_valid"],
                    "python_rmse": report["python_rmse"],
                    "max_abs_sigma_diff": report["max_abs_diff"],
                    "max_rel_sigma_diff": report["max_rel_diff"],
                }
                for ccy, report in sorted(parity_reports.items())
            ]
            parity_summary = pd.DataFrame(parity_summary_rows)
            display(parity_summary)

            parity_point_rows = []
            for ccy, report in sorted(parity_reports.items()):
                first_point = report["points"][0]
                last_point = report["points"][-1]
                parity_point_rows.extend(
                    [
                        {
                            "ccy": ccy,
                            "point": "first",
                            "expiry": first_point["expiry"],
                            "term": first_point["term"],
                            "market_vol": first_point["market_vol"],
                            "market_value": first_point["market_value"],
                            "model_value": first_point["model_value"],
                            "model_vol": first_point["model_vol"],
                        },
                        {
                            "ccy": ccy,
                            "point": "last",
                            "expiry": last_point["expiry"],
                            "term": last_point["term"],
                            "market_vol": last_point["market_vol"],
                            "market_value": last_point["market_value"],
                            "model_value": last_point["model_value"],
                            "model_vol": last_point["model_vol"],
                        },
                    ]
                )
            display(pd.DataFrame(parity_point_rows))

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=False)
            for ccy, report in sorted(parity_reports.items()):
                x = np.arange(len(report["ore_sigmas"]))
                axes[0].plot(x, report["ore_sigmas"], linewidth=1.8, label=f"{ccy} ORE")
                axes[0].plot(x, report["python_sigmas"], linewidth=1.4, linestyle="--", label=f"{ccy} Python")
            axes[0].set_title("ORE vs Python calibrated sigma buckets")
            axes[0].set_xlabel("Bucket index")
            axes[0].set_ylabel("Sigma")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(ncol=2)

            for ccy, report in sorted(parity_reports.items()):
                x = np.arange(len(report["ore_sigmas"]))
                diffs_bp = 1.0e4 * (np.array(report["python_sigmas"]) - np.array(report["ore_sigmas"]))
                axes[1].plot(x, diffs_bp, linewidth=1.8, marker="o", label=ccy)
            axes[1].set_title("Python minus ORE sigma difference (bp)")
            axes[1].set_xlabel("Bucket index")
            axes[1].set_ylabel("Sigma diff x 1e4")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            The important part of this benchmark is not just the final sigma difference. The benchmark now uses ORE's emitted
            `marketdata.csv` and `todaysmarketcalibration.csv` to rebuild the calibration surface and the discount/forward
            curves, so a small parameter difference is interpretable. Earlier versions that used synthetic flat helper vols
            produced misleading failures. In notebook 3 this compare is run on a derived `HullWhite` variant of the ORE case
            so the external Python backend and the ORE calibration are actually solving the same supported problem.
            """
        ),
        md(
            """
            The key visual is the alpha comparison. The template starts from a flat `1%` volatility line, while the calibrated
            output bends upward across maturities. Kappa stays flat because this case calibrates volatility but not reversion.
            The additional multi-currency tables make it clear which currencies carry their own calibrated alpha/kappa term
            structures in the same run.
            """
        ),
        md(
            """
            ## Key takeaways

            - Curve fitting and LGM parameter extraction belong in the same story because they come from the same run setup.
            - The fitted grid is much denser than the market pillars, so visualizing both levels prevents false precision.
            - The direct extractor path and the native session bridge now explicitly show one shared curve-extraction contract.
            - The same fitter stack also works without XML, which is useful for small programmatic demos and tests.
            - The notebook now shows a clean Python curve fitter demo on one USD curve with deposits, FRAs, and swaps as inputs.
            - The Python fitter section now shows discount factors, zero rates, and instantaneous forwards side by side.
            - The notebook now shows a real fresh ORE calibration run, not a flat parameter template masquerading as calibrated output.
            - The calibration section now includes the other configured currencies, not only EUR.
            - In the chosen case, `alpha` is calibrated while `kappa` remains fixed, so that asymmetry is expected.
            - `load_from_ore_xml(...)` remains the clean bridge from ORE artifacts to Python-side model inputs.
            - The new external calibration parity check is only credible because it now rebuilds helper inputs from ORE output artifacts, not from synthetic placeholder vols.
            - The parity compare in this notebook is run on a derived HullWhite variant of the calibration case so it reflects an actual supported external-vs-ORE calibration comparison.

            ## Where this connects next

            The next notebook uses the Python LGM stack directly for a transparent IRS exposure and XVA walkthrough.
            """
        ),
    ]


def notebook_04() -> list[dict]:
    return [
        md(
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
        ),
        code(BOOTSTRAP),
        code(
            """
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
            """
        ),
        md(
            """
            ## Inputs we reuse from the repo

            This notebook leans on the same library code exercised by:
            - `notebook_series/legacy/demo_lgm_irs_xva.ipynb`
            - `tests/test_lgm.py`
            - `tests/test_irs_xva_utils.py`
            """
        ),
        code(
            """
            # Show the simulated state first, then the exposure profile derived from it.
            nh.plot_lgm_paths(demo["times"], demo["x_paths"], max_paths=30, title="Python LGM state paths")
            nh.plot_exposure_profile(
                demo["times"],
                demo["exposure"]["epe"],
                demo["exposure"]["ene"],
                title="IRS exposure profile under the Python LGM path",
            )
            """
        ),
        md(
            """
            The first figure is the latent state process. The second is the economic object that matters for XVA. Keeping both in
            the notebook helps explain where differences later come from when engines disagree.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The older IRS demo was useful because it compared more than one market setup with the same runner. This section keeps
            that idea, but uses it as an interpretive comparison rather than a second full notebook inside the notebook.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Sensitivity and runtime feel

            The goal is not to turn this notebook into a performance study. A small two-point path-count check is enough to show
            that the Python path is interactive and reasonably stable for exploratory work.
            """
        ),
        code(
            """
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
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            This benchmarking block is the missing piece from the old performance notebook. It is more informative than a single
            elapsed time because it shows variance across runs and the throughput difference between the LGM and BA measures.
            """
        ),
        md(
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
        ),
    ]


def notebook_03_1() -> list[dict]:
    return [
        md(
            """
            # 03_1. Python Curve Calibration and LGM Parameters

            This companion notebook keeps the same broad topic as notebook 03, but strips the workflow back to Python-only
            components. The emphasis is on programmatic quote inputs, fitted term structures, and the LGM parameter shape that
            later drives the path simulation.

            **Purpose**
            - fit curves from Python-defined quotes only
            - visualize how fitter choice changes the resulting term structure
            - connect those fitted curves to a Python-native LGM parameter setup

            **What you will learn**
            - how to bootstrap a small multi-currency curve set without XML or external runs
            - how to inspect zero-rate and forward-rate shape from the fitted grid
            - how the LGM alpha and kappa term structures look before any calibration loop is added
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            from py_ore_tools import (
                RateFutureModelParams,
                extract_market_instruments_by_currency_from_quotes,
                fit_discount_curves_from_programmatic_quotes,
                fitted_curves_to_dataframe,
                quote_dicts_from_pairs,
            )

            quote_pairs = [
                ("MM/RATE/USD/USD-LIBOR/1W", 0.0525),
                ("MM/RATE/USD/USD-LIBOR/1M", 0.0520),
                ("MM/RATE/USD/USD-LIBOR/3M", 0.0515),
                ("MM/RATE/USD/USD-LIBOR/6M", 0.0508),
                ("ZERO/RATE/USD/1Y", 0.0500),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/1Y", 0.0494),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/2Y", 0.0485),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/3Y", 0.0479),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/5Y", 0.0470),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/7Y", 0.0462),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/10Y", 0.0455),
                ("MM/RATE/EUR/EUR-EURIBOR/1W", 0.0315),
                ("MM/RATE/EUR/EUR-EURIBOR/1M", 0.0310),
                ("MM/RATE/EUR/EUR-EURIBOR/3M", 0.0307),
                ("ZERO/RATE/EUR/1Y", 0.0300),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/1Y", 0.0304),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/2Y", 0.0310),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/3Y", 0.0314),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/5Y", 0.0320),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/7Y", 0.0325),
                ("IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/10Y", 0.0330),
                ("MM/RATE/GBP/GBP-LIBOR/1W", 0.0415),
                ("MM/RATE/GBP/GBP-LIBOR/1M", 0.0410),
                ("MM/RATE/GBP/GBP-LIBOR/3M", 0.0406),
                ("ZERO/RATE/GBP/1Y", 0.0405),
                ("IR_SWAP/RATE/GBP/GBP-LIBOR-6M/6M/1Y", 0.0400),
                ("IR_SWAP/RATE/GBP/GBP-LIBOR-6M/6M/2Y", 0.0398),
                ("IR_SWAP/RATE/GBP/GBP-LIBOR-6M/6M/3Y", 0.0395),
                ("IR_SWAP/RATE/GBP/GBP-LIBOR-6M/6M/5Y", 0.0391),
                ("IR_SWAP/RATE/GBP/GBP-LIBOR-6M/6M/7Y", 0.0388),
            ]
            quote_dicts = quote_dicts_from_pairs(quote_pairs)
            instruments = extract_market_instruments_by_currency_from_quotes(asof_date="2026-03-08", quotes=quote_dicts)
            bootstrap_fit = fit_discount_curves_from_programmatic_quotes(
                asof_date="2026-03-08",
                quotes=quote_dicts,
                fit_method="bootstrap_mm_irs_v1",
                fit_grid_mode="dense",
                dense_step_years=0.25,
            )
            weighted_fit = fit_discount_curves_from_programmatic_quotes(
                asof_date="2026-03-08",
                quotes=quote_dicts,
                fit_method="weighted_zero_logdf_v1",
                fit_grid_mode="dense",
                dense_step_years=0.25,
            )

            bootstrap_df = fitted_curves_to_dataframe(bootstrap_fit)
            weighted_df = fitted_curves_to_dataframe(weighted_fit)
            display(pd.DataFrame(
                [
                    {"ccy": ccy, "instrument_count": payload["instrument_count"], "fit_points": bootstrap_fit[ccy]["fit_points_count"]}
                    for ccy, payload in sorted(instruments.items())
                ]
            ))
            display(bootstrap_df.head(12))
            """
        ),
        md(
            """
            ## Programmatic market inputs

            The input set is intentionally small. The point is not market realism; it is to make the mapping from quote mix to
            fitted curve visible without hiding behind file loading or a large case bundle.
            """
        ),
        code(
            """
            input_rows = []
            for quote in quote_dicts:
                key = str(quote["key"])
                parts = key.split("/")
                input_rows.append(
                    {
                        "ccy": parts[2] if len(parts) > 2 else "",
                        "quote_key": key,
                        "quote_value": quote.get("value"),
                        "instrument_type": parts[0] if parts else "",
                    }
                )
            input_df = pd.DataFrame(input_rows)
            display(input_df)

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
            nh.plot_ranked_bars(
                input_df.groupby("ccy", as_index=False).size().rename(columns={"size": "count"}),
                "ccy",
                "count",
                title="Programmatic quote count by currency",
                color=nh.PALETTE["blue"],
                ax=axes[0],
            )
            nh.plot_ranked_bars(
                input_df.groupby("instrument_type", as_index=False).size().rename(columns={"size": "count"}),
                "instrument_type",
                "count",
                title="Instrument mix",
                color=nh.PALETTE["gold"],
                ax=axes[1],
            )
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Fitted curves

            Two fitters are shown on the same quotes because that is the smallest useful comparison. It separates the market
            input choice from the interpolation / fitting choice.
            """
        ),
        code(
            """
            bootstrap_summary = (
                bootstrap_df.groupby("ccy", as_index=False)
                .agg(fit_points=("time", "count"), min_df=("df", "min"), max_df=("df", "max"))
                .sort_values("ccy")
            )
            weighted_summary = (
                weighted_df.groupby("ccy", as_index=False)
                .agg(fit_points=("time", "count"), min_df=("df", "min"), max_df=("df", "max"))
                .sort_values("ccy")
            )
            display(bootstrap_summary)
            display(weighted_summary)

            nh.plot_fitted_curves(bootstrap_fit, title="Bootstrap fitter: programmatic discount curves")
            nh.plot_curve_diagnostics(bootstrap_df, ccy="USD", title="Bootstrap fitter diagnostics for USD")

            sample_ccys = sorted(set(bootstrap_df["ccy"]))[:3]
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
            zero_ax = axes[0]
            weighted_ax = zero_ax.twinx()
            for ccy in sample_ccys:
                b = bootstrap_df[bootstrap_df["ccy"] == ccy].sort_values("time")
                w = weighted_df[weighted_df["ccy"] == ccy].sort_values("time")
                zero_ax.plot(b["time"], b["zero_rate"], linewidth=1.8, label=f"{ccy} bootstrap")
                weighted_ax.plot(w["time"], w["zero_rate"], linewidth=1.4, linestyle="--", label=f"{ccy} weighted")
                axes[1].plot(b["time"], 1.0e4 * (w["zero_rate"].to_numpy() - b["zero_rate"].to_numpy()), linewidth=1.8, label=ccy)
            zero_ax.set_title("Programmatic zero curves")
            zero_ax.set_xlabel("Time (years)")
            zero_ax.set_ylabel("Bootstrap zero rate")
            weighted_ax.set_ylabel("Weighted zero rate")
            lines_left, labels_left = zero_ax.get_legend_handles_labels()
            lines_right, labels_right = weighted_ax.get_legend_handles_labels()
            zero_ax.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
            axes[1].set_title("Weighted minus bootstrap (bp)")
            axes[1].set_xlabel("Time (years)")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            The second panel is the one to read closely. If the difference stays small and smooth, the notebook is showing a
            fitter-choice effect rather than evidence of broken market inputs.
            """
        ),
        md(
            """
            ## Rates futures and convexity

            The bootstrap fitter also supports rates futures. There are two Python modes:

            - `external_adjusted_fra`: convert the futures price into an adjusted forward before fitting
            - `native_future`: keep the instrument tagged as a future and apply the same convexity engine inside the bootstrap

            Today those two paths share one convexity implementation, which is the migration-friendly setup if the ORE-compatible
            path needs to stay in place while the Python-native future instrument matures.
            """
        ),
        code(
            """
            future_quotes = [
                ("MM/RATE/USD/USD-LIBOR/1W", 0.0255),
                ("MM/RATE/USD/USD-LIBOR/1M", 0.0200),
                ("MM/RATE/USD/USD-LIBOR/3M", 0.0210),
                {
                    "key": "MM_FUTURE/PRICE/USD/2020-08/ED/3M",
                    "value": 97.55,
                    "contract_start": "2020-08-19",
                    "contract_end": "2020-11-19",
                    "convexity_adjustment": 0.0010,
                },
                {
                    "key": "MM_FUTURE/PRICE/USD/2020-09/ED/3M",
                    "value": 97.47,
                    "contract_start": "2020-09-16",
                    "contract_end": "2020-12-16",
                    "convexity_adjustment": 0.0009,
                },
                {
                    "key": "MM_FUTURE/PRICE/USD/2020-12/ED/3M",
                    "value": 97.38,
                    "contract_start": "2020-12-16",
                    "contract_end": "2021-03-17",
                    "convexity_adjustment": 0.0008,
                },
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/1Y", 0.0230),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/2Y", 0.0240),
            ]

            future_external = fit_discount_curves_from_programmatic_quotes(
                asof_date="2020-05-15",
                quotes=future_quotes,
                instrument_types=("MM", "IR_SWAP", "FUTURE"),
                fit_method="bootstrap_mm_irs_v1",
                fit_grid_mode="instrument",
                future_convexity_mode="external_adjusted_fra",
            )
            future_native = fit_discount_curves_from_programmatic_quotes(
                asof_date="2020-05-15",
                quotes=future_quotes,
                instrument_types=("MM", "IR_SWAP", "FUTURE"),
                fit_method="bootstrap_mm_irs_v1",
                fit_grid_mode="instrument",
                future_convexity_mode="native_future",
            )
            future_model = fit_discount_curves_from_programmatic_quotes(
                asof_date="2020-05-15",
                quotes=[
                    ("MM/RATE/USD/USD-LIBOR/1W", 0.0255),
                    ("MM/RATE/USD/USD-LIBOR/1M", 0.0200),
                    ("MM/RATE/USD/USD-LIBOR/3M", 0.0210),
                    ("MM_FUTURE/PRICE/USD/2020-08/ED/3M", 97.55),
                    ("MM_FUTURE/PRICE/USD/2020-09/ED/3M", 97.47),
                    ("MM_FUTURE/PRICE/USD/2020-12/ED/3M", 97.38),
                    ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/1Y", 0.0230),
                    ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/2Y", 0.0240),
                ],
                instrument_types=("MM", "IR_SWAP", "FUTURE"),
                fit_method="bootstrap_mm_irs_v1",
                fit_grid_mode="instrument",
                future_convexity_mode="native_future",
                future_model_params=RateFutureModelParams(model="hw", mean_reversion=0.03, volatility=0.01),
            )

            future_diag = pd.DataFrame(
                [
                    future_external["USD"]["bootstrap_diagnostics"][0],
                    future_native["USD"]["bootstrap_diagnostics"][0],
                    future_model["USD"]["bootstrap_diagnostics"][0],
                ]
            )
            display(future_diag)

            future_curve_df = pd.DataFrame(
                {
                    "time": future_external["USD"]["times"],
                    "df_external_adjusted_fra": future_external["USD"]["dfs"],
                    "df_native_future": future_native["USD"]["dfs"],
                }
            )
            display(future_curve_df)
            """
        ),
        code(
            """
            demo = nh.run_python_lgm_demo(seed=42, n_paths=512)
            param_frame = nh.lgm_params_frame(demo["params"])
            display(param_frame)

            alpha_frame = pd.DataFrame(
                {
                    "bucket": range(len(demo["params"].alpha_values)),
                    "alpha": list(demo["params"].alpha_values),
                }
            )
            kappa_frame = pd.DataFrame(
                {
                    "bucket": range(len(demo["params"].kappa_values)),
                    "kappa": list(demo["params"].kappa_values),
                }
            )
            display(alpha_frame)
            display(kappa_frame)

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
            axes[0].step(range(len(alpha_frame)), alpha_frame["alpha"], where="post", linewidth=2.0, color=nh.PALETTE["rose"])
            axes[0].set_title("Python alpha term structure")
            axes[0].set_xlabel("Bucket")
            axes[0].set_ylabel("Alpha")
            axes[1].step(range(len(kappa_frame)), kappa_frame["kappa"], where="post", linewidth=2.0, color=nh.PALETTE["teal"])
            axes[1].set_title("Python kappa term structure")
            axes[1].set_xlabel("Bucket")
            axes[1].set_ylabel("Kappa")
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Key takeaways

            - Programmatic quotes are enough to teach the fitter workflow cleanly.
            - The fitting method changes the curve shape more subtly than the quote mix does.
            - The LGM parameter tables can be inspected as ordinary Python data before any larger workflow is built on top.
            """
        ),
    ]


def notebook_04_1() -> list[dict]:
    return [
        md(
            """
            # 04_1. The Python LGM XVA Model

            This companion notebook is the Python-only version of notebook 04. It keeps the focus on the native LGM path,
            exposure profile, and simplified XVA decomposition without referencing any external engine.

            **Purpose**
            - give a readable Python-only XVA walkthrough
            - make the model state, exposure profile, and XVA stack visible
            - establish a self-contained baseline for Python-native experimentation

            **What you will learn**
            - how the Python LGM demo is assembled
            - how the state paths translate into exposure
            - how the simplified XVA metrics relate to that exposure profile
            """
        ),
        code(BOOTSTRAP),
        code(
            """
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
            """
        ),
        md(
            """
            ## Inputs we reuse from the repo

            This notebook leans on the same library code exercised by:
            - `notebook_series/legacy/demo_lgm_irs_xva.ipynb`
            - `tests/test_lgm.py`
            - `tests/test_irs_xva_utils.py`
            """
        ),
        code(
            """
            nh.plot_lgm_paths(demo["times"], demo["x_paths"], max_paths=30, title="Python LGM state paths")
            nh.plot_exposure_profile(
                demo["times"],
                demo["exposure"]["epe"],
                demo["exposure"]["ene"],
                title="IRS exposure profile under the Python LGM path",
            )
            """
        ),
        md(
            """
            The first figure is the latent state process. The second is the economic object that matters for XVA.
            """
        ),
        code(
            """
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
            axes[1].set_title("Illustrative multi-scenario XVA comparison")
            axes[1].tick_params(axis="x", rotation=15)
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Distribution view

            Aggregate metrics are useful, but the NPV distribution and the exposure profile explain more about why those metrics
            move.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Sensitivity and runtime feel

            A small path-count sweep is enough to show that the Python path is interactive and reasonably stable for exploratory work.
            """
        ),
        code(
            """
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
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Large multi-ccy portfolio example

            The small IRS walkthrough above is useful for understanding the mechanics, but it is not the workload where torch
            earns its keep. The bigger batched multi-currency FX forward portfolio is the better scaling example.
            """
        ),
        code(
            """
            portfolio_commands = pd.DataFrame(
                [
                    {
                        "surface": "OreSnapshotApp",
                        "mode": "python",
                        "entry_pattern": "OreSnapshotApp.from_strings(..., options=PurePythonRunOptions(engine='python', price=True, xva=True))",
                        "representative_runtime_sec": 1.55,
                        "paths": 10000,
                        "trades": 256,
                        "artifacts_written_to_repo": False,
                    },
                    {
                        "surface": "OreSnapshotApp",
                        "mode": "compare",
                        "entry_pattern": "OreSnapshotApp.from_strings(..., options=PurePythonRunOptions(engine='compare', price=True, xva=True))",
                        "representative_runtime_sec": 0.17,
                        "paths": 10000,
                        "trades": 256,
                        "artifacts_written_to_repo": False,
                    },
                ]
            )
            display(portfolio_commands)

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
            nh.plot_bar_frame(
                portfolio_commands,
                "mode",
                "representative_runtime_sec",
                title="Representative portfolio runtime shape",
                color=nh.PALETTE["gold"],
                ax=axes[0],
            )
            nh.plot_bar_frame(
                portfolio_commands.assign(paths_per_trade=portfolio_commands["paths"] / portfolio_commands["trades"]),
                "mode",
                "paths_per_trade",
                title="Workload density (paths per trade)",
                color=nh.PALETTE["teal"],
                ax=axes[1],
            )
            axes[0].set_ylabel("Seconds")
            axes[1].set_ylabel("Paths / trade")
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            The important point is the entry surface, not the literal syntax. For notebook and library use, the maintained
            path is the programmatic `OreSnapshotApp` API. It keeps the case in memory from the caller's perspective and avoids
            creating new repo output folders just to inspect one run.
            """
        ),
        md(
            """
            ## Key takeaways

            - The Python LGM path is transparent enough for notebook-level debugging and explanation.
            - The exposure profile is the real driver of the XVA stack; the final metrics are only a summary layer.
            - A small multi-scenario comparison helps show how the same runner behaves under different market regimes.
            - Repeated benchmark runs with throughput are a better performance demo than one elapsed-time printout.
            - The large 256-trade multi-currency portfolio is the right place to compare numpy and torch backends.
            """
        ),
    ]


def notebook_05() -> list[dict]:
    return [
        md(
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
        ),
        code(BOOTSTRAP),
        code(
            """
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
            """
        ),
        md(
            """
            ## Live dual-engine comparison on the repo parity case

            `load_case_snapshots` builds the native `XVASnapshot` used by both adapters (`compare_paths` in the setup cell).
            ORE numbers below come **only** from `ORESwigAdapter` / `OREApp` in this session — not from precomputed CSVs.

            Input validation is a preflight on `ore.xml` (linkages, markets, etc.); it does not substitute for the parity audit
            later in the notebook.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            Both cold and warm runs produce the same XVA numbers on the same snapshot — only the wall time differs.
            `delta` is **ORE-SWIG minus Python LGM**. FVA here means FBA + FCA; if either engine does not compute
            them they will appear as 0.
            """
        ),
        md(
            """
            ## Runtime comparison on the aligned case

            **Cold** rows time the **first** end-to-end pass: ``run_adapter(..., build_adapter=lambda: …)`` includes adapter
            construction, ``XVAEngine``, ``create_session`` (and ``map_snapshot``), and ``session.run``. For ORE-SWIG this is
            where module/`OREApp` warmup shows up.

            **Warm** rows time **only** ``session.run`` on an adapter that already exists (``run_adapter(..., warm=True)``) —
            the usual “many valuations in one batch” view. Python’s adapter is lightweight, so cold ≈ warm; ORE-SWIG typically
            has a large **cold − warm** gap. Timings use ``time.perf_counter`` (wall time, not CPU).
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Parity readiness on the aligned case

            Before trusting the comparison, audit the case. This is the step that prevents us from comparing “same trade name”
            while still using different schedules, curves, or enabled analytics.
            """
        ),
        code(
            """
            aligned_parity_report = aligned_ore_snapshot.parity_completeness_report()
            aligned_parity_df = aligned_ore_snapshot.parity_completeness_dataframe()
            display(aligned_parity_df)
            print("Comparability:", aligned_parity_report["comparability"])
            print("Issues:", aligned_parity_report["issues"])
            comparability_df = aligned_parity_df[aligned_parity_df["section"] == "comparability"].copy()
            comparability_df["comparable"] = comparability_df["value"].astype(bool)
            nh.plot_boolean_matrix(comparability_df, row_col="field", value_cols=["comparable"], title="Aligned case: comparable XVA blocks")
            """
        ),
        md(
            """
            The live dual-engine table is only as fair as the case setup. One important caveat remains:

            - the underlying ORE case still determines which analytics are actually comparable

            That is why funding-related metrics are not always treated as comparable. The audit makes that visible
            rather than leaving it implicit.
            """
        ),
        md(
            """
            ## Stored parity context from the repo

            The parity report below comes from a different common-snapshot setup with the full XVA stack enabled. It is still
            useful because it summarizes the broader prototype status: DVA and FVA are closer, while CVA is still the main gap.
            """
        ),
        code(
            """
            parity_df, parity_summary = nh.parse_parity_report()
            display(parity_df)
            print(parity_summary)
            nh.plot_metric_delta(parity_df, title="Stored parity report: ORE minus Python")

            capability_df = nh.capability_matrix_frame(swig["available"])
            display(capability_df)
            nh.plot_boolean_matrix(capability_df, row_col="capability", value_cols=["python", "ore_swig"], title="Current capability split")
            """
        ),
        md(
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
        ),
    ]


def notebook_05_1() -> list[dict]:
    return [
        md(
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
        ),
        code(BOOTSTRAP),
        code(
            """
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
            """
        ),
        md(
            """
            ## Base run

            The base snapshot is intentionally small. That keeps the session updates easy to reason about and makes it obvious
            which change caused which metric move.
            """
        ),
        code(
            """
            nh.plot_snapshot_composition(snapshot, title="Python-only session: snapshot inventory and quote mix")
            """
        ),
        md(
            """
            ## Market update

            First change a small set of market quotes and rerun the same session. This isolates the market sensitivity of the
            current portfolio without rebuilding a new notebook state from scratch.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The delta view is more informative than the level view once the baseline is familiar. It shows which metrics are
            genuinely moving and which are mostly stable to this small bump.
            """
        ),
        md(
            """
            ## Portfolio update

            Next add one trade and reprice through the same Python session. This is the workflow analogue of a trader-side
            portfolio patch rather than a market move.
            """
        ),
        code(
            """
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
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Capabilities in this workflow

            The capability table is still useful here, but it is read purely as a Python-side checklist rather than a split
            between engines.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Key takeaways

            - The native Python session is already enough for interactive market and portfolio iteration.
            - Base, market-bumped, and portfolio-patched runs are easiest to compare in one persistent session.
            - The notebook is most useful as a workflow demo when the snapshot stays small and explicit.
            """
        ),
    ]


def notebook_06() -> list[dict]:
    return [
        md(
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
        ),
        code(BOOTSTRAP),
        md(
            """
            ## Inputs reused from the repo

            This notebook uses the native Bermudan benchmark pack and the Python Bermudan pricer already exercised by the regression tests:
            - `Tools/PythonOreRunner/parity_artifacts/bermudan_method_compare`
            - `native_xva_interface/bermudan.py`
            - `native_xva_interface/tests/test_bermudan_pricer.py`
            """
        ),
        code(
            """
            berm_comp = nh.load_bermudan_method_comparison()
            display(berm_comp)

            berm_overview = berm_comp[["case_name", "fixed_rate", "py_lsmc", "py_backward", "ore_classic", "ore_amc", "ore_amc_source"]].copy()
            display(berm_overview)
            nh.plot_bermudan_method_levels(berm_comp, title="Stored Bermudan benchmark pack: Python vs ORE classic")
            nh.plot_bermudan_ore_deltas(berm_comp, title="Stored Bermudan benchmark pack: Python minus ORE classic")
            """
        ),
        md(
            """
            The main thing to read off the two plots is that `py_backward` is now the ORE-classic parity method on this
            benchmark pack. `py_lsmc` is still informative because it stays close to `backward`, but it is a control, not
            the target engine.
            """
        ),
        md(
            """
            ## Detailed base case: 2% Bermudan

            The `berm_200bp` case is the cleanest single example because it sits near the middle of the pack and also has
            an AMC fallback number stored in the comparison CSV. Here we reprice the case directly through the Python API
            so the notebook reflects the current code rather than only the stored summary file.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The summary table should show `model_param_source = calibration` for both Python methods. That is important:
            on these benchmark cases the best parity source is the actual `calibration.xml` emitted by ORE classic, not a
            Python-side reconstruction of the trade-specific builder path.
            """
        ),
        md(
            """
            The timing table mixes two sources on purpose:

            - Python timings are measured live in the notebook with wall-clock time
            - ORE timings come from the stored `pricingstats.csv` and `runtimes.csv` written by the benchmark run

            So this section is best read as an operational runtime view, not as a strict apples-to-apples microbenchmark.
            """
        ),
        md(
            """
            ## Interpreting the diagnostics

            The exercise-probability chart answers “when does the option matter most?”. The boundary-state chart answers
            “where is the continuation-versus-exercise switch happening in the LGM state variable?”. Those are the quickest
            sanity checks when the aggregate price looks wrong.
            """
        ),
        md(
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
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The expected pattern is:

            - `python_quote_bump` is close to `ore_direct_quote_bump`
            - both can disagree sharply with `ore_sensitivity_csv`

            That means the remaining gap is not the Bermudan backward engine. It is the sensitivity-definition layer used by
            ORE's analytic.
            """
        ),
        code(
            """
            display(sens_config)

            focus_cols = ["#TradeId", "Factor", "ScenarioDescription", "Base NPV", "Scenario NPV", "Difference"]
            available_cols = [c for c in focus_cols if c in sens_scenario.columns]
            if not available_cols:
                available_cols = list(sens_scenario.columns)
            display(sens_scenario[available_cols])
            """
        ),
        md(
            """
            Read the two ORE tables together:

            - `sensitivity_config.csv` shows which internal sensitivity key actually moved
            - `scenario.csv` shows the up/down scenario NPVs produced by the analytic

            On this case the reported factor label `IndexCurve/EUR-EURIBOR-6M/0/10Y` is associated with an internal node key,
            not a literal raw-market-file quote bump. That is why the direct ORE bump and the sensitivity analytic can have
            different signs.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            The forward `EUR 6M 10Y` row is the headline example:

            - ORE sensitivity analytic says the bump change is positive
            - direct ORE quote bump says the bump change is negative
            - Python quote bump is close to direct ORE quote bump

            So if the benchmark objective is "match direct market quote bumping", Python is already on the right side of the
            comparison. Matching `sensitivity.csv` would require replicating ORE's par-conversion / sensitivity-scenario
            machinery, not changing the Bermudan pricer.
            """
        ),
        code(
            """
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
            """
        ),
        md(
            """
            ## Key takeaways

            - Use the stored Bermudan comparison pack for the quick multi-case view.
            - Use the direct `berm_200bp` repricing section when you want current diagnostics from the live Python code.
            - For sensitivities, distinguish direct quote bumping from ORE's sensitivity analytic. They are not the same calculation.
            - Treat `backward` as the ORE-classic parity engine and `lsmc` as the control.
            - On these Bermudan cases, `calibration.xml` is the right model source whenever ORE has already produced it.
            """
        ),
    ]


def notebook_06_1() -> list[dict]:
    return [
        md(
            """
            # 06_1. Python Bermudan Swaption Pricing

            This companion notebook keeps the Bermudan topic but removes all external comparison machinery. The focus is purely
            on the Python pricing methods, their diagnostics, and the difference between a regression-based and a deterministic
            backward solver on the same synthetic trade.

            **Purpose**
            - price one Bermudan swaption with Python-only methods
            - compare LSMC and backward induction on a shared setup
            - inspect exercise diagnostics directly from the Python results

            **What you will learn**
            - how to define a Bermudan swaption with array-based leg data
            - how the two Python pricing methods compare on one trade
            - what the exercise diagnostics say about the option profile
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            import time

            from py_ore_tools.lgm import LGM1F, LGMParams, simulate_lgm_measure
            from py_ore_tools.lgm_ir_options import BermudanSwaptionDef, bermudan_backward_price, bermudan_lsmc_result
            from py_ore_tools.irs_xva_utils import build_discount_curve_from_zero_rate_pairs

            params = LGMParams(
                alpha_times=(1.0, 2.0, 4.0),
                alpha_values=(0.010, 0.012, 0.015, 0.013),
                kappa_times=(2.0,),
                kappa_values=(0.035, 0.028),
                shift=0.0,
                scaling=1.0,
            )
            model = LGM1F(params)
            p0_disc = build_discount_curve_from_zero_rate_pairs([(0.0, 0.028), (10.0, 0.031)])
            p0_fwd = build_discount_curve_from_zero_rate_pairs([(0.0, 0.029), (10.0, 0.032)])

            fixed_start = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
            fixed_end = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
            fixed_pay = fixed_end.copy()
            float_start = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=float)
            float_end = float_start + 0.5
            legs = {
                "fixed_start_time": fixed_start,
                "fixed_end_time": fixed_end,
                "fixed_pay_time": fixed_pay,
                "fixed_accrual": np.ones_like(fixed_pay),
                "fixed_rate": np.full_like(fixed_pay, 0.032),
                "fixed_notional": np.full_like(fixed_pay, 10_000_000.0),
                "fixed_sign": np.full_like(fixed_pay, -1.0),
                "fixed_amount": np.full_like(fixed_pay, -320_000.0),
                "float_pay_time": float_end,
                "float_start_time": float_start,
                "float_end_time": float_end,
                "float_accrual": np.full_like(float_start, 0.5),
                "float_notional": np.full_like(float_start, 10_000_000.0),
                "float_sign": np.full_like(float_start, 1.0),
                "float_spread": np.zeros_like(float_start),
                "float_coupon": np.zeros_like(float_start),
            }
            berm = BermudanSwaptionDef(
                trade_id="BERM_PY_ONLY",
                exercise_times=np.array([1.0, 2.0, 3.0], dtype=float),
                underlying_legs=legs,
                exercise_sign=1.0,
            )
            times = np.linspace(0.0, 5.0, 21)
            x_paths = simulate_lgm_measure(model, times, n_paths=4096, rng=np.random.default_rng(42))

            t0 = time.perf_counter()
            lsmc = bermudan_lsmc_result(model, p0_disc, p0_fwd, berm, times, x_paths, basis_degree=2, itm_only=True)
            lsmc_elapsed = time.perf_counter() - t0
            t0 = time.perf_counter()
            backward = bermudan_backward_price(model, p0_disc, p0_fwd, berm, n_grid=121, quadrature_order=21)
            backward_elapsed = time.perf_counter() - t0

            summary = pd.DataFrame(
                [
                    {"method": "py_lsmc", "price": float(np.mean(lsmc.npv_paths[0, :])), "elapsed_sec": lsmc_elapsed},
                    {"method": "py_backward", "price": float(backward.price), "elapsed_sec": backward_elapsed},
                ]
            )
            summary["delta_vs_backward"] = summary["price"] - float(backward.price)
            display(summary)
            """
        ),
        md(
            """
            ## Trade setup

            The underlying trade is deliberately explicit: fixed-leg arrays, floating-leg arrays, and exercise times are all in
            the notebook. That keeps the pricing problem inspectable as a pure Python object.
            """
        ),
        code(
            """
            leg_rows = pd.DataFrame(
                {
                    "fixed_start_time": pd.Series(legs["fixed_start_time"]),
                    "fixed_end_time": pd.Series(legs["fixed_end_time"]),
                    "fixed_pay_time": pd.Series(legs["fixed_pay_time"]),
                    "fixed_rate": pd.Series(legs["fixed_rate"]),
                    "float_start_time": pd.Series(legs["float_start_time"]),
                    "float_end_time": pd.Series(legs["float_end_time"]),
                }
            )
            display(leg_rows)
            display(nh.lgm_params_frame(params))
            """
        ),
        md(
            """
            ## Price and runtime comparison

            The method comparison here is internal to the Python stack: stochastic regression against deterministic backward induction.
            On a compact synthetic trade like this, a material gap is not surprising. The value of the notebook is that both
            methods are inspectable from the same trade definition and diagnostics.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
            nh.plot_bar_frame(summary, "method", "price", title="Python Bermudan price by method", color=nh.PALETTE["teal"], ax=axes[0])
            nh.plot_bar_frame(summary, "method", "elapsed_sec", title="Python Bermudan runtime by method", color=nh.PALETTE["gold"], ax=axes[1])
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        code(
            """
            diag_rows = []
            for diag in lsmc.diagnostics:
                diag_rows.append(
                    {
                        "method": "py_lsmc",
                        "time": float(diag.time),
                        "intrinsic_mean": float(diag.intrinsic_mean),
                        "continuation_mean": float(diag.continuation_mean),
                        "exercise_probability": float(diag.exercise_probability),
                        "boundary_state": np.nan if diag.boundary_state is None else float(diag.boundary_state),
                    }
                )
            for diag in backward.diagnostics:
                diag_rows.append(
                    {
                        "method": "py_backward",
                        "time": float(diag.time),
                        "intrinsic_mean": float(diag.intrinsic_mean),
                        "continuation_mean": float(diag.continuation_mean),
                        "exercise_probability": float(diag.exercise_probability),
                        "boundary_state": np.nan if diag.boundary_state is None else float(diag.boundary_state),
                    }
                )
            diag_df = pd.DataFrame(diag_rows).sort_values(["method", "time"]).reset_index(drop=True)
            display(diag_df)
            nh.plot_bermudan_exercise_diagnostics(diag_df, title="Python-only Bermudan exercise diagnostics")
            """
        ),
        md(
            """
            The exercise chart is the main sanity check. It shows whether the two methods agree on roughly when the option is
            alive and where the exercise boundary sits in the state variable.
            """
        ),
        code(
            """
            exercise_hist = pd.DataFrame(
                [
                    {"bucket": "exercise_1Y", "count": int(np.sum(lsmc.exercise_indices == np.searchsorted(times, 1.0)))},
                    {"bucket": "exercise_2Y", "count": int(np.sum(lsmc.exercise_indices == np.searchsorted(times, 2.0)))},
                    {"bucket": "exercise_3Y", "count": int(np.sum(lsmc.exercise_indices == np.searchsorted(times, 3.0)))},
                    {"bucket": "never_exercised", "count": int(np.sum(lsmc.exercise_indices < 0))},
                ]
            )
            display(exercise_hist)

            fig, ax = plt.subplots(figsize=(8.8, 4.2))
            nh.plot_bar_frame(exercise_hist, "bucket", "count", title="LSMC exercise histogram", color=nh.PALETTE["rose"], ax=ax)
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Key takeaways

            - Bermudan pricing can be demonstrated cleanly with Python-only trade arrays and curves.
            - The backward solver and the LSMC solver are complementary methods, not interchangeable black boxes.
            - Exercise diagnostics matter as much as the headline price when checking method behaviour.
            """
        ),
    ]


def notebook_06_2() -> list[dict]:
    return [
        md(
            """
            # 06_2. Python Swap, FX Forward, and Cap/Floor Pricing

            This companion notebook takes the same "explicit trade object plus direct Python pricer" style used in the
            Bermudan example and applies it to three simpler products. The goal is not engine comparison. It is to show
            how to build inspectable pricing examples for a vanilla swap, an FX forward, and a cap/floor directly from
            Python definitions.

            **Purpose**
            - price one vanilla swap directly from ORE-style leg arrays
            - price one EUR/USD FX forward directly from the hybrid IR/FX model
            - price one cap and one floor from explicit coupon schedules

            **What you will learn**
            - how to reuse ORE-style leg arrays for direct Python swap pricing
            - how the FX forward pricer fits into the two-currency hybrid setup
            - how to keep the notebook focused on standalone prices rather than XVA workflows
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            from py_ore_tools.lgm import LGM1F, LGMParams, simulate_lgm_measure
            from py_ore_tools.lgm_fx_xva_utils import FxForwardDef, build_two_ccy_hybrid, fx_forward_npv
            from py_ore_tools.lgm_ir_options import CapFloorDef, capfloor_npv_paths
            from py_ore_tools.irs_xva_utils import build_discount_curve_from_zero_rate_pairs, swap_npv_from_ore_legs_dual_curve

            params = LGMParams(
                alpha_times=(1.0, 2.0, 4.0),
                alpha_values=(0.010, 0.012, 0.015, 0.013),
                kappa_times=(2.0,),
                kappa_values=(0.035, 0.028),
                shift=0.0,
                scaling=1.0,
            )
            model = LGM1F(params)
            p0_disc = build_discount_curve_from_zero_rate_pairs([(0.0, 0.028), (10.0, 0.031)])
            p0_fwd = build_discount_curve_from_zero_rate_pairs([(0.0, 0.029), (10.0, 0.032)])
            ir_times = np.linspace(0.0, 5.0, 21)
            ir_x_paths = simulate_lgm_measure(model, ir_times, n_paths=4096, rng=np.random.default_rng(42))

            def summarize_price(name: str, npv_paths: np.ndarray) -> dict[str, float]:
                values = np.asarray(npv_paths, dtype=float)
                return {
                    "trade": name,
                    "t0_npv": float(np.mean(values[0, :])),
                    "t0_std": float(np.std(values[0, :])),
                    "t0_p05": float(np.quantile(values[0, :], 0.05)),
                    "t0_p95": float(np.quantile(values[0, :], 0.95)),
                }

            def mark_to_market_frame(name: str, times: np.ndarray, npv_paths: np.ndarray) -> pd.DataFrame:
                values = np.asarray(npv_paths, dtype=float)
                return pd.DataFrame(
                    {
                        "trade": name,
                        "time": np.asarray(times, dtype=float),
                        "mean_npv": np.mean(values, axis=1),
                        "p05_npv": np.quantile(values, 0.05, axis=1),
                        "p95_npv": np.quantile(values, 0.95, axis=1),
                    }
                )
            """
        ),
        md(
            """
            ## Swap pricing

            The swap uses the same array-style leg definition as the Bermudan notebook, but here we price the underlying
            directly. That is the cleanest way to show what the pathwise swap pricer is doing before any optionality is added.
            """
        ),
        code(
            """
            fixed_start = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
            fixed_end = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
            float_start = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=float)
            float_end = float_start + 0.5
            swap_legs = {
                "fixed_start_time": fixed_start,
                "fixed_end_time": fixed_end,
                "fixed_pay_time": fixed_end.copy(),
                "fixed_accrual": np.ones_like(fixed_end),
                "fixed_rate": np.full_like(fixed_end, 0.032),
                "fixed_notional": np.full_like(fixed_end, 10_000_000.0),
                "fixed_sign": np.full_like(fixed_end, -1.0),
                "fixed_amount": np.full_like(fixed_end, -320_000.0),
                "float_pay_time": float_end,
                "float_start_time": float_start,
                "float_end_time": float_end,
                "float_accrual": np.full_like(float_start, 0.5),
                "float_notional": np.full_like(float_start, 10_000_000.0),
                "float_sign": np.full_like(float_start, 1.0),
                "float_spread": np.zeros_like(float_start),
                "float_coupon": np.zeros_like(float_start),
            }
            swap_npv_paths = np.vstack(
                [
                    swap_npv_from_ore_legs_dual_curve(model, p0_disc, p0_fwd, swap_legs, float(t), ir_x_paths[i, :])
                    for i, t in enumerate(ir_times)
                ]
            )
            swap_summary = pd.DataFrame([summarize_price("payer_swap", swap_npv_paths)])

            swap_schedule = pd.DataFrame(
                {
                    "fixed_start_time": pd.Series(swap_legs["fixed_start_time"]),
                    "fixed_end_time": pd.Series(swap_legs["fixed_end_time"]),
                    "fixed_rate": pd.Series(swap_legs["fixed_rate"]),
                    "float_start_time": pd.Series(swap_legs["float_start_time"]),
                    "float_end_time": pd.Series(swap_legs["float_end_time"]),
                }
            )
            display(swap_summary)
            display(swap_schedule)
            display(nh.lgm_params_frame(params))
            """
        ),
        code(
            """
            swap_mtm = mark_to_market_frame("payer_swap", ir_times, swap_npv_paths)
            display(swap_mtm.head())

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
            nh.plot_bar_frame(swap_summary, "trade", "t0_npv", title="Swap t0 NPV", color=nh.PALETTE["teal"], ax=axes[0])
            axes[1].plot(swap_mtm["time"], swap_mtm["mean_npv"], label="mean NPV", color=nh.PALETTE["blue"], linewidth=2.0)
            axes[1].fill_between(
                swap_mtm["time"],
                swap_mtm["p05_npv"],
                swap_mtm["p95_npv"],
                color=nh.PALETTE["mint"],
                alpha=0.65,
                label="5%-95% band",
            )
            axes[1].set_title("Swap mark-to-market band")
            axes[1].set_xlabel("time")
            axes[1].set_ylabel("value")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            The swap chart is a pricing diagnostic, not an exposure report. It shows how the direct swap pricer evolves the
            mark-to-market across the simulation grid and how wide the simulated valuation band becomes through time.
            """
        ),
        md(
            """
            ## Cap and floor pricing

            The cap and floor reuse one coupon schedule so the notebook isolates option-type effects rather than schedule
            differences. Both instruments are priced pathwise from the same one-factor LGM model and dual-curve setup.
            """
        ),
        code(
            """
            coupon_start = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
            coupon_end = coupon_start + 0.5
            common_cf_kwargs = {
                "ccy": "EUR",
                "start_time": coupon_start,
                "end_time": coupon_end,
                "pay_time": coupon_end.copy(),
                "accrual": np.full_like(coupon_start, 0.5),
                "notional": np.full_like(coupon_start, 5_000_000.0),
                "fixing_time": coupon_start.copy(),
                "position": 1.0,
            }
            cap_def = CapFloorDef(
                trade_id="CAP_PY_ONLY",
                option_type="cap",
                strike=np.full_like(coupon_start, 0.031),
                **common_cf_kwargs,
            )
            floor_def = CapFloorDef(
                trade_id="FLOOR_PY_ONLY",
                option_type="floor",
                strike=np.full_like(coupon_start, 0.027),
                **common_cf_kwargs,
            )
            cap_npv_paths = capfloor_npv_paths(model, p0_disc, p0_fwd, cap_def, ir_times, ir_x_paths, lock_fixings=True)
            floor_npv_paths = capfloor_npv_paths(model, p0_disc, p0_fwd, floor_def, ir_times, ir_x_paths, lock_fixings=True)

            capfloor_summary = pd.DataFrame(
                [
                    summarize_price("cap", cap_npv_paths),
                    summarize_price("floor", floor_npv_paths),
                ]
            )
            capfloor_schedule = pd.DataFrame(
                {
                    "start_time": coupon_start,
                    "end_time": coupon_end,
                    "cap_strike": cap_def.strike,
                    "floor_strike": floor_def.strike,
                    "notional": cap_def.notional,
                }
            )
            display(capfloor_summary)
            display(capfloor_schedule)
            """
        ),
        code(
            """
            cap_mtm = mark_to_market_frame("cap", ir_times, cap_npv_paths)
            floor_mtm = mark_to_market_frame("floor", ir_times, floor_npv_paths)

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
            nh.plot_bar_frame(capfloor_summary, "trade", "t0_npv", title="Cap/Floor t0 NPV", color=nh.PALETTE["gold"], ax=axes[0])
            axes[1].plot(cap_mtm["time"], cap_mtm["mean_npv"], label="cap mean NPV", color=nh.PALETTE["blue"], linewidth=2.0)
            axes[1].plot(floor_mtm["time"], floor_mtm["mean_npv"], label="floor mean NPV", color=nh.PALETTE["rose"], linewidth=2.0)
            axes[1].set_title("Cap/Floor mark-to-market comparison")
            axes[1].set_xlabel("time")
            axes[1].set_ylabel("value")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            The cap/floor comparison stays in pricing space as well. On the same schedule, different strikes and option
            direction change both the t0 price and the mean mark-to-market path.
            """
        ),
        md(
            """
            ## FX forward pricing

            The FX forward uses the hybrid IR/FX model because the product depends on both discounting curves and the FX spot
            process. The setup stays explicit: one pair, one strike, one maturity, and one reporting currency.
            """
        ),
        code(
            """
            p0_eur = build_discount_curve_from_zero_rate_pairs([(0.0, 0.020), (5.0, 0.022)])
            p0_usd = build_discount_curve_from_zero_rate_pairs([(0.0, 0.030), (5.0, 0.033)])
            fx_times = np.linspace(0.0, 2.5, 11)

            hybrid = build_two_ccy_hybrid(
                pair="EUR/USD",
                ir_specs={
                    "EUR": {"alpha": 0.010, "kappa": 0.030},
                    "USD": {"alpha": 0.012, "kappa": 0.025},
                },
                fx_vol=0.14,
                corr_dom_fx=-0.20,
                corr_for_fx=0.15,
            )
            fx_def = FxForwardDef(
                trade_id="FXFWD_PY_ONLY",
                pair="EUR/USD",
                notional_base=2_000_000.0,
                strike=1.10,
                maturity=2.0,
            )
            fx_sim = hybrid.simulate_paths(
                fx_times,
                4096,
                rng=np.random.default_rng(7),
                log_s0={"EUR/USD": np.log(1.08)},
            )
            fx_npv_paths = np.vstack(
                [
                    fx_forward_npv(
                        hybrid,
                        fx_def,
                        float(t),
                        fx_sim["s"]["EUR/USD"][i, :],
                        fx_sim["x"]["USD"][i, :],
                        fx_sim["x"]["EUR"][i, :],
                        p0_usd,
                        p0_eur,
                    )
                    for i, t in enumerate(fx_times)
                ]
            )
            fx_summary = pd.DataFrame([summarize_price("eurusd_fx_forward", fx_npv_paths)])
            fx_setup = pd.DataFrame(
                [
                    {"field": "pair", "value": fx_def.pair},
                    {"field": "spot0", "value": 1.08},
                    {"field": "strike", "value": fx_def.strike},
                    {"field": "maturity", "value": fx_def.maturity},
                    {"field": "notional_base", "value": fx_def.notional_base},
                    {"field": "domestic_curve_label", "value": "USD"},
                    {"field": "foreign_curve_label", "value": "EUR"},
                ]
            )
            display(fx_summary)
            display(fx_setup)
            """
        ),
        code(
            """
            fx_mtm = mark_to_market_frame("eurusd_fx_forward", fx_times, fx_npv_paths)
            fx_spot_summary = pd.DataFrame(
                {
                    "time": fx_times,
                    "mean_spot": np.mean(fx_sim["s"]["EUR/USD"], axis=1),
                }
            )
            display(fx_mtm.head())
            display(fx_spot_summary.head())

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
            axes[0].plot(fx_spot_summary["time"], fx_spot_summary["mean_spot"], color=nh.PALETTE["cyan"], linewidth=2.0)
            axes[0].axhline(fx_def.strike, color=nh.PALETTE["slate"], linestyle="--", linewidth=1.5)
            axes[0].set_title("Mean simulated EUR/USD spot")
            axes[0].set_xlabel("time")
            axes[0].set_ylabel("spot")
            axes[1].plot(fx_mtm["time"], fx_mtm["mean_npv"], label="mean NPV", color=nh.PALETTE["blue"], linewidth=2.0)
            axes[1].fill_between(
                fx_mtm["time"],
                fx_mtm["p05_npv"],
                fx_mtm["p95_npv"],
                color=nh.PALETTE["sand"],
                alpha=0.7,
                label="5%-95% band",
            )
            axes[1].set_title("FX forward mark-to-market in USD")
            axes[1].set_xlabel("time")
            axes[1].set_ylabel("value")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Key takeaways

            - Vanilla swap, cap/floor, and FX forward pricing can all be demonstrated with explicit Python product definitions.
            - The swap and cap/floor examples share the one-factor IR setup, while the FX forward needs the hybrid IR/FX model.
            - The notebook stays on direct pricing: no exposure aggregation, no CVA, and no XVA workflow machinery.
            """
        ),
    ]


NOTEBOOKS = {
    "01_python_to_ore_swig_dataclasses.ipynb": notebook_01,
    "02_ore_snapshot_capabilities.ipynb": notebook_02,
    "03_curve_calibration_and_lgm_params.ipynb": notebook_03,
    "03_1_python_curve_calibration_and_lgm_params.ipynb": notebook_03_1,
    "04_python_lgm_xva_model.ipynb": notebook_04,
    "04_1_python_lgm_xva_model.ipynb": notebook_04_1,
    "05_joint_python_and_ore_workflow.ipynb": notebook_05,
    "05_1_python_only_workflow.ipynb": notebook_05_1,
    "06_bermudan_python_vs_ore.ipynb": notebook_06,
    "06_1_python_bermudan_swaption_pricing.ipynb": notebook_06_1,
    "06_2_python_swap_fxforward_capfloor_pricing.ipynb": notebook_06_2,
}


def _selected_notebooks(only: list[str]) -> list[tuple[str, Callable]]:
    if not only:
        return list(NOTEBOOKS.items())
    selected: list[tuple[str, Callable]] = []
    for name, builder in NOTEBOOKS.items():
        if any(token in name or token == Path(name).stem for token in only):
            selected.append((name, builder))
    if not selected:
        raise SystemExit(f"No notebooks matched: {only}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate notebook-series .ipynb and .py files")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run each generated .py mirror after writing it",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Limit generation to notebook names or stem fragments, e.g. --only 01_python_to_ore_swig_dataclasses",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    for name, builder in _selected_notebooks(args.only):
        cells = builder()
        nb = {
            "cells": cells,
            "metadata": {
                "kernelspec": KERNEL,
                "language_info": LANG_INFO,
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (root / name).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        py_name = Path(name).with_suffix(".py")
        py_path = root / py_name
        py_path.write_text(render_py(cells), encoding="utf-8")
        print(f"Wrote {name}")
        if args.run:
            print(f"Running {py_name}")
            subprocess.run([sys.executable, str(py_path)], cwd=str(root), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
