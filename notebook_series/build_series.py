from __future__ import annotations

import json
import textwrap
from pathlib import Path


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
            from native_xva_interface import XVASnapshot, map_snapshot, DeterministicToyAdapter, ORESwigAdapter

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
            ## Optional SWIG run

            This is intentionally a boundary check, not a parity claim. The point is that the same snapshot can be handed to
            an ORE-backed adapter without rewriting the notebook around engine-specific inputs.
            """
        ),
        code(
            """
            swig = nh.swig_status()
            print(swig["message"])

            if swig["available"] and RUN_ORE_SWIG:
                # Run the same snapshot through the ORE-backed adapter only when the user asks for it.
                swig_result, swig_elapsed = nh.run_adapter(snapshot, ORESwigAdapter())
                print(f"ORE-SWIG elapsed: {swig_elapsed:.4f}s")
                display(nh.result_metrics_frame(swig_result))
            elif swig["available"]:
                print("Skipping ORE-SWIG adapter execution. Set RUN_ORE_SWIG_DEMOS=1 to enable it.")
            else:
                print("Skipping ORE-SWIG adapter execution in this environment.")
            """
        ),
        md(
            """
            ## Key takeaways

            - The dataclass layer already carries enough structure to be the common hand-off point for both engines.
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
            from native_xva_interface import map_snapshot, DeterministicToyAdapter

            # Run ORE now, then load the fresh inputs and outputs into Python-facing snapshot objects.
            snapshot, ore_snapshot, fresh_meta = nh.load_fresh_case_snapshots("flat_EUR_5Y_A", label="notebook_02")
            mapped = map_snapshot(snapshot)

            print("Fresh run root:", fresh_meta["run_root"])
            print("Fresh ORE xml:", fresh_meta["ore_xml"])
            print("Fresh output dir:", fresh_meta["output_dir"])
            display(nh.snapshot_overview(snapshot))
            nh.plot_snapshot_composition(snapshot, title="Fresh loader-backed snapshot: inventory and market mix")
            """
        ),
        md(
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
            - Fresh ORE output generation removes the ambiguity of stale checked-in files.
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
            from py_ore_tools import (
                discount_factors_to_dataframe,
                extract_discount_factors_by_currency,
                extract_market_instruments_by_currency,
                extract_market_instruments_by_currency_from_quotes,
                fit_discount_curves_from_ore_market,
                fit_discount_curves_from_programmatic_quotes,
                fitted_curves_to_dataframe,
                quote_dicts_from_pairs,
            )
            from ore_curve_fit_parity import trace_discount_curve_from_ore, trace_index_curve_from_ore

            ORE_XML = nh.default_curve_case_ore_xml(repo)
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
            configured_lgm, calibrated_lgm, calibration_meta = nh.run_fresh_lgm_calibration_demo(ccy_key="EUR")

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
            """
        ),
        md(
            """
            The key visual is the alpha comparison. The template starts from a flat `1%` volatility line, while the calibrated
            output bends upward across maturities. Kappa stays flat because this case calibrates volatility but not reversion.
            """
        ),
        md(
            """
            ## Key takeaways

            - Curve fitting and LGM parameter extraction belong in the same story because they come from the same run setup.
            - The fitted grid is much denser than the market pillars, so visualizing both levels prevents false precision.
            - The direct extractor path and the native session bridge now explicitly show one shared curve-extraction contract.
            - The same fitter stack also works without XML, which is useful for small programmatic demos and tests.
            - The notebook now shows a real fresh ORE calibration run, not a flat parameter template masquerading as calibrated output.
            - In the chosen case, `alpha` is calibrated while `kappa` remains fixed, so that asymmetry is expected.
            - `load_from_ore_xml(...)` remains the clean bridge from ORE artifacts to Python-side model inputs.

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
            - `demo_lgm_irs_xva.ipynb`
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


def notebook_05() -> list[dict]:
    return [
        md(
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
        ),
        code(BOOTSTRAP),
        code(
            """
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
            """
        ),
        md(
            """
            ## Inputs we reuse from the repo

            The aligned comparison still starts from the repo's benchmark case definition:
            - `parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A`
            - `run_ore_snapshot_native_xva.py`
            - parity notes in `SKILL.md`

            The difference is that this notebook now regenerates the ORE outputs during execution, so the comparison is based on
            fresh run data rather than on the checked-in output folder.
            """
        ),
        code(
            """
            display(aligned_comparison)
            aligned_plot = aligned_comparison[aligned_comparison["metric"].isin(["PV", "CVA", "DVA"])].copy()
            nh.plot_metric_comparison(aligned_plot, "python_lgm", "ore_output", title="Aligned benchmark case: Python vs fresh ORE output")
            nh.plot_metric_delta(aligned_plot, title="Aligned benchmark case: Python minus ORE")
            """
        ),
        md(
            """
            The paired charts above answer two different questions. The grouped bars show level agreement. The delta chart shows
            whether the remaining gap is economically small or still material relative to the metric being compared.
            """
        ),
        md(
            """
            ## Runtime comparison on the aligned case

            The numerical comparison is only half of the story. The table below shows the wall-clock cost of the Python LGM run
            versus the fresh ORE executable run used to produce the benchmark outputs for this notebook execution.
            """
        ),
        code(
            """
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
            The aligned case is the main numerical comparison in this notebook. One important caveat remains:

            - `flat_EUR_5Y_A` only requests `CVA` in the underlying ORE case

            That is why funding-related metrics are not treated as comparable in the aligned section. The audit makes that visible
            rather than leaving it implicit.
            """
        ),
        md(
            """
            ## Live Native Parity Case

            The aligned benchmark above is file-based. This section uses a fresh live/native case from the current ORE example
            set, chosen because it already lines up well in the regression pack: `ore_measure_lgm_fixed.xml`.

            Unlike the old stress-classic demo, this is a real parity section. It is the right place to look at live `FBA/FCA`
            because both engines are materially closer on this case.
            """
        ),
        code(
            """
            display(nh.snapshot_overview(live_snapshot))
            display(nh.trade_frame(live_snapshot))
            display(live_perf_df)
            display(live_comparison)

            live_plot = live_comparison[live_comparison["metric"].isin(["CVA", "DVA", "FBA", "FCA"])].copy()
            nh.plot_metric_comparison(live_plot, "python_lgm", "ore_output", title="Live/native parity case: Python vs fresh ORE")
            nh.plot_metric_delta(live_plot, title="Live/native parity case: Python minus ORE")
            """
        ),
        md(
            """
            Read this section differently from the stress-classic workflow demo that appeared earlier in the series. Here the
            numbers are supposed to match. The point is to show that the current native Python path can in fact track a fresh ORE
            live run on a case that is set up for parity rather than for broad stress-style coverage.
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

            - Use the fresh aligned benchmark case when you want a clean Python-vs-ORE comparison.
            - Use the live/native parity case when you want to show that Python can also match a fresh ORE run on a case tuned for parity, including `FBA/FCA`.
            - Keep the parity audit in the notebook, because it tells you which comparisons are actually fair.

            ## End of series

            These five notebooks now form one continuous story: Python snapshot construction, fresh ORE-backed loading, market
            calibration, Python LGM XVA, and a final Python-and-ORE workflow that distinguishes aligned comparisons from live demos.
            """
        ),
    ]


NOTEBOOKS = {
    "01_python_to_ore_swig_dataclasses.ipynb": notebook_01,
    "02_ore_snapshot_capabilities.ipynb": notebook_02,
    "03_curve_calibration_and_lgm_params.ipynb": notebook_03,
    "04_python_lgm_xva_model.ipynb": notebook_04,
    "05_joint_python_and_ore_workflow.ipynb": notebook_05,
}


def main() -> int:
    root = Path(__file__).resolve().parent
    for name, builder in NOTEBOOKS.items():
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
        (root / py_name).write_text(render_py(cells), encoding="utf-8")
        print(f"Wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
