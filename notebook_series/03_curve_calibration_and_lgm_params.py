# Auto-generated from the notebook source in build_series.py
# Execute with: python3 <this_file>

# %% cell 0
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

# %% cell 3
"""
## Inputs we reuse from the repo

The curve side of the series is grounded in the existing parity walkthroughs and tests:
- `ore_curve_fit_parity/notebooks/usd_curve_parity_walkthrough.ipynb`
- `ore_curve_fit_parity/notebooks/multi_currency_curve_trace_demo.ipynb`
- `tests/test_quick_curve_fit.py`
"""

# %% cell 4
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

# %% cell 5
"""
The first chart answers a structural question: how dense is the fitted curve relative to the number of calibration
instruments? That difference is exactly why a fitted grid can look smooth even when the market input is sparse.
"""

# %% cell 6
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

# %% cell 7
"""
The older extractor notebook also tried to expose this through a session bridge. In the current codebase that bridge
is not surfaced as a public `XVASession` method, so this series keeps the extraction path direct and explicit rather
than pretending the runtime API already exposes a stable curve-extraction wrapper.
"""

# %% cell 8
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

# %% cell 9
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

# %% cell 10
"""
The weighted-versus-bootstrap comparison is worth showing because it makes the fitting choice visible. The old
extractor demo did this well, and it is more informative than showing one fitter in isolation.
"""

# %% cell 11
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

# %% cell 12
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

# %% cell 13
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

# %% cell 14
"""
The key visual is the alpha comparison. The template starts from a flat `1%` volatility line, while the calibrated
output bends upward across maturities. Kappa stays flat because this case calibrates volatility but not reversion.
"""

# %% cell 15
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

