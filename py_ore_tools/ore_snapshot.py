"""ore_snapshot.py — Single-entry-point loader for all ORE inputs needed by the Python LGM.

Reads the ORE XML input chain starting from a single ore.xml file and resolves
every path, curve-column name, LGM parameter set, swap leg, and ORE output
automatically.  No manual flags required.

Usage
-----
    from py_ore_tools.ore_snapshot import OreSnapshot, load_from_ore_xml

    snap = load_from_ore_xml("/path/to/ore_measure_lgm.xml")
    model = snap.build_model()
    x = simulate_lgm_measure(model, snap.exposure_times, n_paths=5000, rng=np.random.default_rng(42))

XML chain resolved automatically
---------------------------------
  ore.xml
    Setup/asofDate               → snap.asof_date
    Setup/outputPath             → resolves all output CSVs
    Setup/portfolioFile          → portfolio.xml
    Setup/marketConfigFile       → todaysmarket.xml
    Setup/marketDataFile         → market_data.txt
    Markets/simulation           → sim_config_id  (e.g. "libor")
    Analytics/simulation/
      simulationConfigFile       → simulation.xml

  simulation.xml
    DomesticCcy                  → snap.domestic_ccy
    Measure                      → snap.measure
    Seed / Samples               → snap.seed / snap.n_samples
    CrossAssetModel/LGM          → initial (pre-calibration) LGMParams
    Market/YieldCurves/Tenors    → snap.node_tenors (for interpolation)

  todaysmarket.xml
    Configuration[sim_config_id]
      /DiscountingCurvesId       → disc_curves_id
    DiscountingCurves[disc_curves_id]
      /DiscountingCurve[ccy]     → curve_handle   (e.g. "Yield/EUR/EUR6M")
    YieldCurves / IndexFwdCurves
      with text == curve_handle  → snap.discount_column (e.g. "EUR-EURIBOR-6M")

  portfolio.xml
    Trade[trade_id]
      /FloatingLegData/Index     → snap.forward_column
      /Envelope/CounterParty     → snap.counterparty
      /Envelope/NettingSetId     → snap.netting_set_id

  <outputPath>/
    calibration.xml   (if present) → calibrated LGMParams (preferred over simulation.xml)
    curves.csv        → snap.p0_disc, snap.p0_fwd
    exposure_trade_<id>.csv        → snap.exposure_times, ore_epe, ore_ene
    xva.csv                        → snap.ore_cva
    npv.csv                        → snap.ore_t0_npv
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .lgm import LGM1F, LGMParams
from .rate_futures import (
    RateFutureModelParams,
    build_rate_future_quote,
    future_forward_rate,
    future_to_fra_rate,
    year_fraction_365,
)
from .irs_xva_utils import (
    apply_parallel_float_spread_shift_to_match_npv,
    build_discount_curve_from_discount_pairs,
    calibrate_float_spreads_from_coupon,
    load_ore_default_curve_inputs,
    load_ore_discount_pairs_by_columns,
    load_ore_discount_pairs_from_curves,
    load_ore_exposure_profile,
    load_ore_legs_from_flows,
    load_simulation_yield_tenors,
    load_swap_legs_from_portfolio,
    parse_lgm_params_from_calibration_xml,
    parse_lgm_params_from_simulation_xml,
    survival_probability_from_hazard,
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OreSnapshot:
    """Structured container for every ORE input the Python LGM needs.

    Created by :func:`load_from_ore_xml`.  Treat as read-only.
    All file-path fields are absolute strings for auditability.
    """

    # --- Provenance ---------------------------------------------------------
    ore_xml_path: str           # absolute path of the ore.xml that was loaded
    asof_date: str              # ISO date string, e.g. "2016-02-05"

    # --- LGM parameters -----------------------------------------------------
    lgm_params: LGMParams       # ready-to-use, possibly from calibration.xml
    alpha_source: str           # "calibration" | "simulation"

    # --- Simulation config --------------------------------------------------
    measure: str                # "LGM" or "BA"
    seed: int
    n_samples: int              # number of ORE simulation paths
    domestic_ccy: str           # e.g. "EUR"
    node_tenors: np.ndarray     # yield-curve node tenors (years) for interp
    model_day_counter: str      # ORE model / curve day counter, e.g. "A365F"
    report_day_counter: str     # ORE exposure report day counter, always ActualActual(ISDA)

    # --- Trade / portfolio --------------------------------------------------
    trade_id: str
    counterparty: str
    netting_set_id: str
    legs: Dict                  # output of load_swap_legs_from_portfolio +
    #                              node_tenors injected + float spreads calibrated

    # --- Curve columns & raw data -------------------------------------------
    discount_column: str        # curves.csv column for discounting
    forward_column: str         # curves.csv column for forwarding (float index)
    xva_discount_column: str    # base-currency discount curve under XVA/simulation config
    curve_times_disc: np.ndarray
    curve_dfs_disc: np.ndarray
    curve_times_fwd: np.ndarray
    curve_dfs_fwd: np.ndarray
    curve_times_xva_disc: np.ndarray
    curve_dfs_xva_disc: np.ndarray

    # Callable discount-factor functions (built from the raw arrays above)
    p0_disc: object = dataclasses.field(repr=False)   # Callable[[float], float]
    p0_fwd: object = dataclasses.field(repr=False)    # Callable[[float], float]
    p0_xva_disc: object = dataclasses.field(repr=False)

    # --- ORE output: exposure profile ---------------------------------------
    exposure_times: np.ndarray
    exposure_dates: np.ndarray  # string dates matching exposure_times
    exposure_model_times: np.ndarray
    ore_epe: np.ndarray
    ore_ene: np.ndarray

    # --- ORE output: t0 NPV and XVA ------------------------------------------
    ore_t0_npv: float
    ore_cva: float

    # --- Credit inputs -------------------------------------------------------
    recovery: float
    hazard_times: np.ndarray
    hazard_rates: np.ndarray

    # --- ORE output: DVA / FVA (from xva.csv when present) -------------------
    ore_dva: float = 0.0
    ore_fba: float = 0.0
    ore_fca: float = 0.0

    # --- Optional funding curves for ORE-style FBA / FCA --------------------
    borrowing_curve_column: Optional[str] = None
    lending_curve_column: Optional[str] = None
    curve_times_borrow: Optional[np.ndarray] = None
    curve_dfs_borrow: Optional[np.ndarray] = None
    curve_times_lend: Optional[np.ndarray] = None
    curve_dfs_lend: Optional[np.ndarray] = None
    p0_borrow: Optional[object] = dataclasses.field(default=None, repr=False)
    p0_lend: Optional[object] = dataclasses.field(default=None, repr=False)

    # --- Own (bank) credit for DVA — from market when dvaName is in ore.xml ---
    own_hazard_times: Optional[np.ndarray] = None
    own_hazard_rates: Optional[np.ndarray] = None
    own_recovery: Optional[float] = None

    # --- Provenance / parity audit ------------------------------------------
    leg_source: str = "portfolio"
    requested_xva_metrics: tuple[str, ...] = ()
    own_name: Optional[str] = None
    portfolio_xml_path: Optional[str] = None
    todaysmarket_xml_path: Optional[str] = None
    market_data_path: Optional[str] = None
    simulation_xml_path: Optional[str] = None
    calibration_xml_path: Optional[str] = None
    curves_csv_path: Optional[str] = None
    exposure_csv_path: Optional[str] = None
    xva_csv_path: Optional[str] = None
    npv_csv_path: Optional[str] = None
    flows_csv_path: Optional[str] = None

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def build_model(self, alpha_scale: float = 1.0) -> LGM1F:
        """Construct and return a fresh :class:`LGM1F` from the snapshot parameters.

        Parameters
        ----------
        alpha_scale:
            Multiplicative scale applied to all ``alpha_values`` before building
            the model.  Use 1.0 (default) for the raw calibrated parameters.
        """
        if alpha_scale == 1.0:
            return LGM1F(self.lgm_params)
        params = dataclasses.replace(
            self.lgm_params,
            alpha_values=tuple(v * alpha_scale for v in self.lgm_params.alpha_values),
        )
        return LGM1F(params)

    def survival_probability(self, times: np.ndarray) -> np.ndarray:
        """Piecewise-flat hazard survival probabilities at the requested times."""
        return survival_probability_from_hazard(
            times, self.hazard_times, self.hazard_rates
        )

    def discount_factors(self, times: np.ndarray) -> np.ndarray:
        """Convenience wrapper: vectorised P^d(0, t) on the discount curve."""
        return np.asarray([self.p0_disc(float(t)) for t in times], dtype=float)

    def model_time_from_date(self, d: str | date) -> float:
        return _year_fraction_from_day_counter(self.asof_date, d, self.model_day_counter)

    def report_time_from_date(self, d: str | date) -> float:
        return _year_fraction_from_day_counter(self.asof_date, d, self.report_day_counter)

    def parity_completeness_report(self) -> Dict[str, object]:
        """Return a structured audit of parity-critical snapshot inputs.

        The report is intentionally explicit about what is present, what is
        missing, and which XVA components are reasonably comparable from this
        snapshot.
        """
        requested = tuple(self.requested_xva_metrics)
        file_paths = {
            "ore_xml": self.ore_xml_path,
            "portfolio_xml": self.portfolio_xml_path,
            "todaysmarket_xml": self.todaysmarket_xml_path,
            "market_data": self.market_data_path,
            "simulation_xml": self.simulation_xml_path,
            "calibration_xml": self.calibration_xml_path,
            "curves_csv": self.curves_csv_path,
            "exposure_csv": self.exposure_csv_path,
            "xva_csv": self.xva_csv_path,
            "npv_csv": self.npv_csv_path,
            "flows_csv": self.flows_csv_path,
        }
        files_present = {
            key: (bool(path) and Path(str(path)).exists())
            for key, path in file_paths.items()
        }
        trade_ok = bool(self.trade_id and self.counterparty and self.netting_set_id and self.legs)
        curve_ok = bool(
            self.discount_column
            and self.forward_column
            and self.curve_times_disc.size > 0
            and self.curve_times_fwd.size > 0
            and len(self.curve_times_disc) == len(self.curve_dfs_disc)
            and len(self.curve_times_fwd) == len(self.curve_dfs_fwd)
        )
        cpty_credit_ok = bool(
            self.hazard_times.size > 0
            and self.hazard_rates.size > 0
            and self.recovery is not None
        )
        own_credit_ok = bool(
            self.own_hazard_times is not None
            and self.own_hazard_rates is not None
            and np.size(self.own_hazard_times) > 0
            and np.size(self.own_hazard_rates) > 0
            and self.own_recovery is not None
        )
        funding_ok = bool(
            self.borrowing_curve_column
            and self.lending_curve_column
            and self.curve_times_borrow is not None
            and self.curve_dfs_borrow is not None
            and self.curve_times_lend is not None
            and self.curve_dfs_lend is not None
        )
        exposure_ok = bool(
            self.exposure_times.size > 1
            and np.all(np.diff(self.exposure_times) > 0.0)
            and len(self.exposure_times) == len(self.exposure_dates) == len(self.exposure_model_times)
        )
        issues: list[str] = []
        if not trade_ok:
            issues.append("trade/schedule inputs are incomplete")
        if self.leg_source != "flows":
            issues.append("legs are not sourced from flows.csv; portfolio-derived schedules may still differ from ORE cashflow signs")
        if not curve_ok:
            issues.append("discount/forward curve extraction is incomplete")
        if not cpty_credit_ok:
            issues.append("counterparty hazard/recovery inputs are incomplete")
        if ("DVA" in requested or abs(self.ore_dva) > 1.0e-12) and not own_credit_ok:
            issues.append("own-name hazard/recovery inputs are incomplete for DVA parity")
        if ("FVA" in requested or abs(self.ore_fba) > 1.0e-12 or abs(self.ore_fca) > 1.0e-12) and not funding_ok:
            issues.append("borrowing/lending curve inputs are incomplete for FVA parity")
        if not exposure_ok:
            issues.append("exposure grid is incomplete or inconsistent")

        comparable_metrics = {
            "CVA": trade_ok and curve_ok and cpty_credit_ok and exposure_ok,
            "DVA": trade_ok and curve_ok and cpty_credit_ok and own_credit_ok and exposure_ok,
            "FVA": trade_ok and curve_ok and funding_ok and exposure_ok,
            "MVA": trade_ok and curve_ok and exposure_ok and ("MVA" in requested or False),
        }

        return {
            "summary": {
                "trade_id": self.trade_id,
                "counterparty": self.counterparty,
                "netting_set_id": self.netting_set_id,
                "leg_source": self.leg_source,
                "requested_xva_metrics": list(requested),
                "own_name": self.own_name,
            },
            "conventions": {
                "measure": self.measure,
                "model_day_counter": self.model_day_counter,
                "report_day_counter": self.report_day_counter,
                "seed": self.seed,
                "samples": self.n_samples,
            },
            "trade_setup": {
                "complete": trade_ok,
                "fixed_leg_count": int(np.size(self.legs.get("fixed_pay_time", []))) if self.legs else 0,
                "float_leg_count": int(np.size(self.legs.get("float_pay_time", []))) if self.legs else 0,
            },
            "curve_setup": {
                "complete": curve_ok,
                "discount_column": self.discount_column,
                "forward_column": self.forward_column,
                "xva_discount_column": self.xva_discount_column,
                "borrowing_curve_column": self.borrowing_curve_column,
                "lending_curve_column": self.lending_curve_column,
                "discount_points": int(self.curve_times_disc.size),
                "forward_points": int(self.curve_times_fwd.size),
            },
            "credit_setup": {
                "counterparty_credit_complete": cpty_credit_ok,
                "own_credit_complete": own_credit_ok,
                "counterparty_hazard_points": int(self.hazard_times.size),
                "own_hazard_points": int(np.size(self.own_hazard_times)) if self.own_hazard_times is not None else 0,
                "counterparty_recovery": float(self.recovery),
                "own_recovery": None if self.own_recovery is None else float(self.own_recovery),
            },
            "funding_setup": {
                "complete": funding_ok,
                "ore_fba": float(self.ore_fba),
                "ore_fca": float(self.ore_fca),
            },
            "exposure_setup": {
                "complete": exposure_ok,
                "exposure_points": int(self.exposure_times.size),
                "ore_epe_points": int(self.ore_epe.size),
                "ore_ene_points": int(self.ore_ene.size),
            },
            "comparability": comparable_metrics,
            "files": {
                key: {"path": path, "exists": files_present[key]}
                for key, path in file_paths.items()
            },
            "issues": issues,
            "parity_ready": bool(comparable_metrics["CVA"] and not issues),
        }

    def parity_completeness_dataframe(self):
        import pandas as pd

        report = self.parity_completeness_report()
        rows = []
        for section in ("trade_setup", "curve_setup", "credit_setup", "funding_setup", "exposure_setup", "comparability"):
            payload = report.get(section, {})
            for key, value in payload.items():
                rows.append({"section": section, "field": key, "value": value})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # keep repr readable even with large arrays
        return (
            f"OreSnapshot("
            f"asof_date={self.asof_date!r}, "
            f"trade_id={self.trade_id!r}, "
            f"alpha_source={self.alpha_source!r}, "
            f"measure={self.measure!r}, "
            f"model_day_counter={self.model_day_counter!r}, "
            f"discount_column={self.discount_column!r}, "
            f"forward_column={self.forward_column!r}, "
            f"n_exposure_pts={self.exposure_times.size}, "
            f"ore_cva={self.ore_cva:,.2f}, "
            f"recovery={self.recovery:.2%})"
        )


# ---------------------------------------------------------------------------
# Per-currency discount-factor extraction API
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CurveDFPayload:
    """Programmatic payload for one currency discount curve."""

    ccy: str
    curve_id: str
    source_column: str
    asof_date: str
    day_counter: str
    times: np.ndarray
    dfs: np.ndarray
    calendar_dates: tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        """JSON-friendly representation."""
        return {
            "curve_id": self.curve_id,
            "source_column": self.source_column,
            "asof_date": self.asof_date,
            "day_counter": self.day_counter,
            "times": [float(x) for x in self.times],
            "dfs": [float(x) for x in self.dfs],
            "calendar_dates": list(self.calendar_dates),
        }


_TODAYSMARKET_OBJECT_META: tuple[tuple[str, str, str], ...] = (
    ("YieldCurvesId", "YieldCurves", "YieldCurve"),
    ("DiscountingCurvesId", "DiscountingCurves", "DiscountingCurve"),
    ("IndexForwardingCurvesId", "IndexForwardingCurves", "Index"),
    ("SwapIndexCurvesId", "SwapIndexCurves", "SwapIndex"),
    ("ZeroInflationIndexCurvesId", "ZeroInflationIndexCurves", "ZeroInflationIndexCurve"),
    ("YYInflationIndexCurvesId", "YYInflationIndexCurves", "YYInflationIndexCurve"),
    ("FxSpotsId", "FxSpots", "FxSpot"),
    ("FxVolatilitiesId", "FxVolatilities", "FxVolatility"),
    ("SwaptionVolatilitiesId", "SwaptionVolatilities", "SwaptionVolatility"),
    ("YieldVolatilitiesId", "YieldVolatilities", "YieldVolatility"),
    ("CapFloorVolatilitiesId", "CapFloorVolatilities", "CapFloorVolatility"),
    ("CDSVolatilitiesId", "CDSVolatilities", "CDSVolatility"),
    ("DefaultCurvesId", "DefaultCurves", "DefaultCurve"),
    ("YYInflationCapFloorVolatilitiesId", "YYInflationCapFloorVolatilities", "YYInflationCapFloorVolatility"),
    ("ZeroInflationCapFloorVolatilitiesId", "ZeroInflationCapFloorVolatilities", "ZeroInflationCapFloorVolatility"),
    ("EquityCurvesId", "EquityCurves", "EquityCurve"),
    ("EquityVolatilitiesId", "EquityVolatilities", "EquityVolatility"),
    ("SecuritiesId", "Securities", "Security"),
    ("BaseCorrelationsId", "BaseCorrelations", "BaseCorrelation"),
    ("CommodityCurvesId", "CommodityCurves", "CommodityCurve"),
    ("CommodityVolatilitiesId", "CommodityVolatilities", "CommodityVolatility"),
    ("CorrelationsId", "Correlations", "Correlation"),
)

_SPEC_PREFIX_TO_CURVECONFIG_TAGS: dict[str, tuple[str, ...]] = {
    "Yield": ("YieldCurves",),
    "FX": (),
    "FXVolatility": ("FXVolatilities",),
    "SwaptionVolatility": ("SwaptionVolatilities",),
    "YieldVolatility": ("YieldVolatilities",),
    "CapFloorVolatility": ("CapFloorVolatilities",),
    "CDSVolatility": ("CDSVolatilities",),
    "Default": ("DefaultCurves",),
    "Inflation": ("InflationCurves",),
    "InflationCapFloorVolatility": ("InflationCapFloorVolatilities",),
    "Equity": ("EquityCurves",),
    "EquityVolatility": ("EquityVolatilities",),
    "Security": ("Securities",),
    "BaseCorrelation": ("BaseCorrelations",),
    "Commodity": ("CommodityCurves",),
    "CommodityVolatility": ("CommodityVolatilities",),
    "Correlation": ("Correlations",),
}

_FX_DOMINANCE_ORDER: tuple[str, ...] = (
    "XAU", "XAG", "XPT", "XPD",
    "EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "ZAR",
    "MYR", "SGD",
    "DKK", "NOK", "SEK",
    "HKD", "THB", "TWD", "MXN",
    "CNY", "CNH",
    "JPY",
    "IDR", "KRW",
)


def validate_ore_input_snapshot(
    ore_xml_path: str | Path,
    *,
    include_all_market_configs: bool = False,
) -> Dict[str, object]:
    """Preflight validation of ORE input linkages for ore_snapshot-style runs.

    This codifies the common setup checklist:
    - conventions referenced by active curve configs exist
    - quotes referenced by active curve configs exist on asof date
    - curve specs in active todaysmarket sections resolve to real curve configs
    - yield and index curve aliases do not overlap inside the same todaysmarket id
    - requested market configurations in ore.xml exist in todaysmarket.xml
    - FX duplicate/dominance outcomes are surfaced explicitly
    - implyTodaysFixings and any asof-date fixings are surfaced together
    """
    ore_xml = Path(ore_xml_path).resolve()
    ore_root = ET.parse(ore_xml).getroot()

    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    markets_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Markets/Parameter")
    }
    analytics_params: dict[str, dict[str, str]] = {}
    for analytic in ore_root.findall("./Analytics/Analytic"):
        analytic_type = analytic.attrib.get("type", "").strip()
        analytics_params[analytic_type] = {
            n.attrib.get("name", ""): (n.text or "").strip()
            for n in analytic.findall("./Parameter")
        }

    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml}")

    base = ore_xml.parent
    curveconfig_xml = (base / setup_params.get("curveConfigFile", "curveconfig.xml")).resolve()
    conventions_xml = (base / setup_params.get("conventionsFile", "conventions.xml")).resolve()
    todaysmarket_xml = (base / setup_params.get("marketConfigFile", "todaysmarket.xml")).resolve()
    market_data_file = (base / setup_params.get("marketDataFile", "market.txt")).resolve()
    fixing_data_file = (base / setup_params.get("fixingDataFile", "fixings.txt")).resolve()
    portfolio_xml = (base / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    imply_todays_fixings = str(setup_params.get("implyTodaysFixings", "N")).strip().upper() in {"Y", "YES", "TRUE"}

    missing_files = [
        str(p)
        for p in (curveconfig_xml, conventions_xml, todaysmarket_xml, market_data_file)
        if not p.exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Required ORE input files not found: {missing_files}")

    tm_root = ET.parse(todaysmarket_xml).getroot()
    cc_root = ET.parse(curveconfig_xml).getroot()
    conv_root = ET.parse(conventions_xml).getroot()

    available_market_configs = sorted(
        {
            cfg.attrib.get("id", "").strip()
            for cfg in tm_root.findall("./Configuration")
            if cfg.attrib.get("id", "").strip()
        }
    )
    requested_market_configs = []
    for value in markets_params.values():
        value = value.strip()
        if value and value not in requested_market_configs:
            requested_market_configs.append(value)
    curves_cfg = analytics_params.get("curves", {}).get("configuration", "").strip()
    if curves_cfg and curves_cfg not in requested_market_configs:
        requested_market_configs.append(curves_cfg)
    if "default" not in requested_market_configs:
        requested_market_configs.append("default")

    missing_requested_market_configs = [
        cfg for cfg in requested_market_configs if cfg not in available_market_configs
    ]
    active_market_configs = (
        available_market_configs if include_all_market_configs else [cfg for cfg in requested_market_configs if cfg in available_market_configs]
    )

    curve_nodes_by_tag: dict[str, dict[str, ET.Element]] = defaultdict(dict)
    for section in list(cc_root):
        section_tag = section.tag
        for curve_node in list(section):
            curve_id = (curve_node.findtext("./CurveId") or "").strip()
            if curve_id:
                curve_nodes_by_tag[section_tag][curve_id] = curve_node

    available_conventions = sorted(
        {
            (node.findtext("./Id") or "").strip()
            for node in list(conv_root)
            if (node.findtext("./Id") or "").strip()
        }
    )
    available_convention_set = set(available_conventions)
    relevant_currencies, relevant_indices, relevant_counterparties = _collect_snapshot_relevance(
        portfolio_xml if portfolio_xml.exists() else None,
        analytics_params,
    )

    active_section_ids: dict[str, set[str]] = defaultdict(set)
    missing_section_refs: list[dict[str, str]] = []
    for config_id in active_market_configs:
        cfg = tm_root.find(f"./Configuration[@id='{config_id}']")
        if cfg is None:
            continue
        for config_child, section_tag, _ in _TODAYSMARKET_OBJECT_META:
            section_id = (cfg.findtext(f"./{config_child}") or "").strip()
            if not section_id:
                continue
            active_section_ids[section_tag].add(section_id)
            if tm_root.find(f"./{section_tag}[@id='{section_id}']") is None:
                missing_section_refs.append(
                    {
                        "configuration": config_id,
                        "section": section_tag,
                        "section_id": section_id,
                    }
                )

    active_curve_ids_by_tag: dict[str, set[str]] = defaultdict(set)
    invalid_curve_specs: list[dict[str, str]] = []
    for _, section_tag, item_tag in _TODAYSMARKET_OBJECT_META:
        for section_id in sorted(active_section_ids.get(section_tag, set())):
            section = tm_root.find(f"./{section_tag}[@id='{section_id}']")
            if section is None:
                continue
            for item in section.findall(f"./{item_tag}"):
                if section_tag == "SwapIndexCurves":
                    continue
                spec = (item.text or "").strip()
                if not spec:
                    continue
                parent_tags, curve_id = _curve_spec_target(spec)
                if parent_tags is None:
                    invalid_curve_specs.append(
                        {
                            "section": section_tag,
                            "section_id": section_id,
                            "item": item_tag,
                            "spec": spec,
                            "reason": "unrecognized curve spec prefix",
                        }
                    )
                    continue
                if not parent_tags:
                    continue
                if curve_id is None:
                    invalid_curve_specs.append(
                        {
                            "section": section_tag,
                            "section_id": section_id,
                            "item": item_tag,
                            "spec": spec,
                            "reason": "could not parse curve id from spec",
                        }
                    )
                    continue
                found = False
                for parent_tag in parent_tags:
                    if curve_id in curve_nodes_by_tag.get(parent_tag, {}):
                        active_curve_ids_by_tag[parent_tag].add(curve_id)
                        found = True
                if not found:
                    invalid_curve_specs.append(
                        {
                            "section": section_tag,
                            "section_id": section_id,
                            "item": item_tag,
                            "spec": spec,
                            "reason": "referenced curve config not found",
                        }
                    )

    quote_scope_curve_ids_by_tag: dict[str, set[str]] = defaultdict(set)
    for section_id in sorted(active_section_ids.get("DiscountingCurves", set())):
        section = tm_root.find(f"./DiscountingCurves[@id='{section_id}']")
        if section is None:
            continue
        selected = [
            node for node in section.findall("./DiscountingCurve")
            if not relevant_currencies or (node.attrib.get("currency", "").strip() in relevant_currencies)
        ]
        for node in selected or section.findall("./DiscountingCurve"):
            spec = (node.text or "").strip()
            parent_tags, curve_id = _curve_spec_target(spec)
            if not parent_tags or curve_id is None:
                continue
            for parent_tag in parent_tags:
                if curve_id in curve_nodes_by_tag.get(parent_tag, {}):
                    quote_scope_curve_ids_by_tag[parent_tag].add(curve_id)
    for section_id in sorted(active_section_ids.get("IndexForwardingCurves", set())):
        section = tm_root.find(f"./IndexForwardingCurves[@id='{section_id}']")
        if section is None:
            continue
        selected = [
            node for node in section.findall("./Index")
            if not relevant_indices or (node.attrib.get("name", "").strip() in relevant_indices)
        ]
        for node in selected or section.findall("./Index"):
            spec = (node.text or "").strip()
            parent_tags, curve_id = _curve_spec_target(spec)
            if not parent_tags or curve_id is None:
                continue
            for parent_tag in parent_tags:
                if curve_id in curve_nodes_by_tag.get(parent_tag, {}):
                    quote_scope_curve_ids_by_tag[parent_tag].add(curve_id)
    for section_id in sorted(active_section_ids.get("DefaultCurves", set())):
        section = tm_root.find(f"./DefaultCurves[@id='{section_id}']")
        if section is None:
            continue
        selected = [
            node for node in section.findall("./DefaultCurve")
            if not relevant_counterparties or (node.attrib.get("name", "").strip() in relevant_counterparties)
        ]
        for node in selected:
            spec = (node.text or "").strip()
            parent_tags, curve_id = _curve_spec_target(spec)
            if not parent_tags or curve_id is None:
                continue
            for parent_tag in parent_tags:
                if curve_id in curve_nodes_by_tag.get(parent_tag, {}):
                    quote_scope_curve_ids_by_tag[parent_tag].add(curve_id)
    if not quote_scope_curve_ids_by_tag:
        quote_scope_curve_ids_by_tag = active_curve_ids_by_tag

    used_conventions: set[str] = set()
    required_mandatory_quotes: set[str] = set()
    required_optional_quotes: set[str] = set()
    for section_tag, curve_ids in quote_scope_curve_ids_by_tag.items():
        for curve_id in sorted(curve_ids):
            node = curve_nodes_by_tag.get(section_tag, {}).get(curve_id)
            if node is None:
                continue
            for conv_node in node.findall(".//Conventions"):
                conv_id = (conv_node.text or "").strip()
                if conv_id:
                    used_conventions.add(conv_id)
            for quote_node in node.findall(".//Quotes/Quote"):
                quote = (quote_node.text or "").strip()
                if not quote:
                    continue
                optional = str(quote_node.attrib.get("optional", "")).strip().lower() in {"true", "y", "yes", "1"}
                if optional:
                    required_optional_quotes.add(quote)
                else:
                    required_mandatory_quotes.add(quote)
            for quote_node in node.findall(".//Quotes/CompositeQuote/RateQuote"):
                quote = (quote_node.text or "").strip()
                if quote:
                    required_mandatory_quotes.add(quote)
            for quote_node in node.findall(".//Quotes/CompositeQuote/SpreadQuote"):
                quote = (quote_node.text or "").strip()
                if quote:
                    required_mandatory_quotes.add(quote)

    missing_conventions = sorted(used_conventions - available_convention_set)

    market_quotes_by_date = _load_ore_csv_keys_by_date(market_data_file)
    asof_market_quotes = market_quotes_by_date.get(asof_date, set())
    missing_mandatory_quotes = sorted(required_mandatory_quotes - asof_market_quotes)
    missing_optional_quotes = sorted(required_optional_quotes - asof_market_quotes)

    overlaps: list[dict[str, str]] = []
    yield_ids = {
        section.attrib.get("id", "").strip()
        for section in tm_root.findall("./YieldCurves")
        if section.attrib.get("id", "").strip()
    }
    index_ids = {
        section.attrib.get("id", "").strip()
        for section in tm_root.findall("./IndexForwardingCurves")
        if section.attrib.get("id", "").strip()
    }
    for shared_id in sorted(yield_ids & index_ids):
        y_section = tm_root.find(f"./YieldCurves[@id='{shared_id}']")
        i_section = tm_root.find(f"./IndexForwardingCurves[@id='{shared_id}']")
        if y_section is None or i_section is None:
            continue
        yield_names = {
            node.attrib.get("name", "").strip()
            for node in y_section.findall("./YieldCurve")
            if node.attrib.get("name", "").strip()
        }
        index_names = {
            node.attrib.get("name", "").strip()
            for node in i_section.findall("./Index")
            if node.attrib.get("name", "").strip()
        }
        for name in sorted(yield_names & index_names):
            overlaps.append({"id": shared_id, "name": name})

    fx_pairs_with_both_directions = []
    fx_quote_pairs = sorted(q for q in asof_market_quotes if q.startswith("FX/RATE/"))
    fx_seen: set[tuple[str, str]] = set()
    for key in fx_quote_pairs:
        parts = key.split("/")
        if len(parts) != 4:
            continue
        c1, c2 = parts[2], parts[3]
        canonical = tuple(sorted((c1, c2)))
        if canonical in fx_seen:
            continue
        inverse = f"FX/RATE/{c2}/{c1}"
        if inverse in asof_market_quotes and c1 != c2:
            dominant = _fx_dominance(c1, c2)
            kept = f"FX/RATE/{dominant[:3]}/{dominant[3:]}"
            dropped = inverse if kept == key else key
            fx_pairs_with_both_directions.append(
                {
                    "pair": f"{canonical[0]}/{canonical[1]}",
                    "dominant": dominant,
                    "kept": kept,
                    "dropped": dropped,
                }
            )
            fx_seen.add(canonical)

    today_fixing_count = 0
    if fixing_data_file.exists():
        fixing_keys_by_date = _load_ore_csv_keys_by_date(fixing_data_file)
        today_fixing_count = len(fixing_keys_by_date.get(asof_date, set()))

    issues: list[str] = []
    action_items: list[dict[str, object]] = []
    if missing_requested_market_configs:
        issues.append(f"missing todaysmarket configurations: {missing_requested_market_configs}")
        action_items.append(
            {
                "code": "missing_market_configurations",
                "severity": "error",
                "what_failed": "Some market configurations requested by ore.xml are not defined in todaysmarket.xml.",
                "what_to_fix": "Add the missing <Configuration id=\"...\"> blocks to todaysmarket.xml or change the Markets/Parameter values in ore.xml to an existing configuration id.",
                "where_to_fix": [str(ore_xml), str(todaysmarket_xml)],
                "details": missing_requested_market_configs,
            }
        )
    if missing_section_refs:
        issues.append("some active todaysmarket section ids are referenced but not defined")
        action_items.append(
            {
                "code": "missing_todaysmarket_sections",
                "severity": "error",
                "what_failed": "An active configuration points to a todaysmarket section id that does not exist.",
                "what_to_fix": "Create the missing section block in todaysmarket.xml or change the corresponding *Id field in the active configuration to an existing section id.",
                "where_to_fix": [str(todaysmarket_xml)],
                "details": missing_section_refs,
            }
        )
    if invalid_curve_specs:
        issues.append("some active todaysmarket curve specs do not resolve to curve configurations")
        action_items.append(
            {
                "code": "unresolved_curve_specs",
                "severity": "error",
                "what_failed": "Some specs in todaysmarket.xml point to curve ids that are not present in curveconfig.xml.",
                "what_to_fix": "Either add the missing curve configuration to curveconfig.xml or change the spec in todaysmarket.xml so it points at a real CurveId.",
                "where_to_fix": [str(todaysmarket_xml), str(curveconfig_xml)],
                "details": invalid_curve_specs,
            }
        )
    if missing_conventions:
        issues.append("some active curve-config convention ids are missing")
        action_items.append(
            {
                "code": "missing_conventions",
                "severity": "error",
                "what_failed": "Active curve configs reference convention ids that do not exist in conventions.xml.",
                "what_to_fix": "Add these convention ids to conventions.xml or update the <Conventions> fields in curveconfig.xml to use ids that already exist.",
                "where_to_fix": [str(curveconfig_xml), str(conventions_xml)],
                "details": missing_conventions,
            }
        )
    if missing_mandatory_quotes:
        issues.append("some active mandatory curve-config quotes are missing on the asof date")
        action_items.append(
            {
                "code": "missing_mandatory_quotes",
                "severity": "error",
                "what_failed": "The active curve build requests mandatory quote ids that are not present in the market data on the asof date.",
                "what_to_fix": "Add these quote ids for the asof date to the market data file, or change the active curve configuration so it stops requesting them, or mark genuinely optional quotes as optional in curveconfig.xml.",
                "where_to_fix": [str(curveconfig_xml), str(market_data_file)],
                "details": missing_mandatory_quotes[:50],
                "detail_count": len(missing_mandatory_quotes),
            }
        )
    if overlaps:
        issues.append("yield and index curve aliases overlap inside an active todaysmarket id")
        action_items.append(
            {
                "code": "yield_index_alias_overlap",
                "severity": "error",
                "what_failed": "The same alias name is used in both YieldCurves and IndexForwardingCurves for the same todaysmarket id.",
                "what_to_fix": "Rename either the YieldCurve @name or the Index @name in todaysmarket.xml so the alias only appears once per id.",
                "where_to_fix": [str(todaysmarket_xml)],
                "details": overlaps,
            }
        )
    if today_fixing_count > 0 and not imply_todays_fixings:
        issues.append("asof-date fixings are present but implyTodaysFixings is disabled")
        action_items.append(
            {
                "code": "todays_fixings_ignored",
                "severity": "warning",
                "what_failed": "The fixing file contains asof-date fixings, but ORE is configured not to use today's fixings.",
                "what_to_fix": "Set Setup/implyTodaysFixings to Y in ore.xml if those asof-date fixings should be consumed, or remove them from the fixing file if they are not intentional.",
                "where_to_fix": [str(ore_xml), str(fixing_data_file) if fixing_data_file.exists() else str(ore_xml)],
                "details": {"asof_date": asof_date, "today_fixing_count": today_fixing_count},
            }
        )
    if fx_pairs_with_both_directions:
        action_items.append(
            {
                "code": "fx_dominance_notice",
                "severity": "info",
                "what_failed": "Both FX directions are present for some pairs, so ORE will keep only the dominant orientation internally.",
                "what_to_fix": "If you want a cleaner input set, keep only the dominant FX direction shown here in the market data file.",
                "where_to_fix": [str(market_data_file)],
                "details": fx_pairs_with_both_directions,
            }
        )

    checks = {
        "market_configurations_exist": not missing_requested_market_configs,
        "referenced_todaysmarket_sections_exist": not missing_section_refs,
        "curve_specs_resolve": not invalid_curve_specs,
        "conventions_exist": not missing_conventions,
        "quotes_exist_for_asof": not missing_mandatory_quotes,
        "yield_index_names_distinct": not overlaps,
    }

    return {
        "summary": {
            "ore_xml_path": str(ore_xml),
            "asof_date": asof_date,
            "selected_market_configs": active_market_configs,
            "market_contexts": dict(markets_params),
        },
        "files": {
            "curveconfig_xml": str(curveconfig_xml),
            "conventions_xml": str(conventions_xml),
            "todaysmarket_xml": str(todaysmarket_xml),
            "market_data_file": str(market_data_file),
            "fixing_data_file": str(fixing_data_file) if fixing_data_file.exists() else None,
            "portfolio_xml": str(portfolio_xml) if portfolio_xml.exists() else None,
        },
        "market_configurations": {
            "requested": requested_market_configs,
            "available": available_market_configs,
            "missing_requested": missing_requested_market_configs,
            "valid": not missing_requested_market_configs,
        },
        "todaysmarket_sections": {
            "missing_references": missing_section_refs,
            "valid": not missing_section_refs,
        },
        "curve_specs": {
            "active_curve_ids_by_section": {k: sorted(v) for k, v in sorted(active_curve_ids_by_tag.items())},
            "quote_scope_curve_ids_by_section": {k: sorted(v) for k, v in sorted(quote_scope_curve_ids_by_tag.items())},
            "invalid": invalid_curve_specs,
            "valid": not invalid_curve_specs,
        },
        "conventions": {
            "used_ids": sorted(used_conventions),
            "missing_ids": missing_conventions,
            "valid": not missing_conventions,
        },
        "quotes": {
            "mandatory_required_count": len(required_mandatory_quotes),
            "optional_required_count": len(required_optional_quotes),
            "asof_quote_count": len(asof_market_quotes),
            "missing_mandatory": missing_mandatory_quotes,
            "missing_optional": missing_optional_quotes,
            "valid": not missing_mandatory_quotes,
        },
        "todaysmarket_names": {
            "yield_index_overlaps": overlaps,
            "distinct": not overlaps,
        },
        "fx_dominance": {
            "pairs_with_both_directions": fx_pairs_with_both_directions,
            "pair_count_with_both_directions": len(fx_pairs_with_both_directions),
        },
        "fixings": {
            "imply_todays_fixings": imply_todays_fixings,
            "today_fixing_count": today_fixing_count,
            "potential_issue": bool(today_fixing_count > 0 and not imply_todays_fixings),
        },
        "checks": checks,
        "issues": issues,
        "action_items": action_items,
        "input_links_valid": bool(all(checks.values())),
    }


def ore_input_validation_dataframe(report: Dict[str, object]):
    """Convert ``validate_ore_input_snapshot()`` output into a flat DataFrame."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for ore_input_validation_dataframe()") from exc

    rows = []
    for section in (
        "market_configurations",
        "todaysmarket_sections",
        "curve_specs",
        "conventions",
        "quotes",
        "todaysmarket_names",
        "fixings",
        "checks",
    ):
        payload = report.get(section, {})
        if isinstance(payload, dict):
            for key, value in payload.items():
                rows.append({"section": section, "field": key, "value": value})
    for idx, item in enumerate(report.get("action_items", [])):
        rows.append({"section": "action_items", "field": f"{idx}:code", "value": item.get("code")})
        rows.append({"section": "action_items", "field": f"{idx}:severity", "value": item.get("severity")})
        rows.append({"section": "action_items", "field": f"{idx}:what_failed", "value": item.get("what_failed")})
        rows.append({"section": "action_items", "field": f"{idx}:what_to_fix", "value": item.get("what_to_fix")})
        rows.append({"section": "action_items", "field": f"{idx}:where_to_fix", "value": item.get("where_to_fix")})
    return pd.DataFrame(rows, columns=["section", "field", "value"])


def validate_xva_snapshot_dataclasses(snapshot) -> Dict[str, object]:
    """Validate an in-memory XVASnapshot-style dataclass object.

    This is intended for snapshots from ``native_xva_interface.dataclasses``
    or compatible programmatic objects with the same attribute layout.
    """
    market = snapshot.market
    fixings = snapshot.fixings
    portfolio = snapshot.portfolio
    config = snapshot.config
    netting = getattr(snapshot, "netting", None)
    collateral = getattr(snapshot, "collateral", None)

    quote_keys = [getattr(q, "key", "") for q in market.raw_quotes]
    quote_key_dates = [(getattr(q, "date", ""), getattr(q, "key", "")) for q in market.raw_quotes]
    quote_duplicate_count = len(quote_key_dates) - len(set(quote_key_dates))

    trade_ids = [t.trade_id for t in portfolio.trades]
    trade_counterparties = {t.counterparty for t in portfolio.trades}
    trade_netting_sets = {t.netting_set for t in portfolio.trades}
    defined_netting_sets = set((netting.netting_sets or {}).keys()) if netting is not None else set()
    collateral_netting_sets = {
        b.netting_set_id for b in (collateral.balances if collateral is not None else ())
    }
    supported_metrics = {"CVA", "DVA", "FVA", "MVA"}
    analytics = tuple(config.analytics)
    invalid_metrics = sorted(m for m in analytics if m not in supported_metrics)
    market_quote_dates = sorted({q.date for q in market.raw_quotes})
    fixing_dates = sorted({p.date for p in fixings.points})

    asof_consistent = market.asof == config.asof
    quote_dates_match_asof = all(q.date == market.asof for q in market.raw_quotes)
    fixing_dates_not_after_asof = all(p.date <= config.asof for p in fixings.points)
    analytics_supported = bool(analytics) and not invalid_metrics
    trade_ids_unique = len(trade_ids) == len(set(trade_ids))
    netting_sets_defined = not trade_netting_sets or trade_netting_sets.issubset(defined_netting_sets) or not defined_netting_sets
    collateral_matches_netting = not collateral_netting_sets or collateral_netting_sets.issubset(
        defined_netting_sets or trade_netting_sets
    )

    issues: list[str] = []
    action_items: list[dict[str, object]] = []
    if not asof_consistent:
        issues.append("market.asof and config.asof do not match")
        action_items.append(
            {
                "code": "asof_mismatch",
                "severity": "error",
                "what_failed": "The market asof date and config asof date do not match.",
                "what_to_fix": "Set snapshot.market.asof and snapshot.config.asof to the same date before running validation or pricing.",
                "where_to_fix": ["snapshot.market.asof", "snapshot.config.asof"],
                "details": {"market_asof": market.asof, "config_asof": config.asof},
            }
        )
    if not quote_dates_match_asof:
        issues.append("some market quotes are not dated on the market asof")
        action_items.append(
            {
                "code": "quote_date_mismatch",
                "severity": "error",
                "what_failed": "Some market quotes carry a date different from snapshot.market.asof.",
                "what_to_fix": "Either restamp those MarketQuote.date values to the snapshot asof or move them into a snapshot whose asof matches the quote date.",
                "where_to_fix": ["snapshot.market.raw_quotes[*].date"],
                "details": market_quote_dates,
            }
        )
    if quote_duplicate_count > 0:
        issues.append("duplicate market quote (date, key) entries are present")
        action_items.append(
            {
                "code": "duplicate_market_quotes",
                "severity": "error",
                "what_failed": "There are duplicate market quotes with the same (date, key) pair.",
                "what_to_fix": "Deduplicate snapshot.market.raw_quotes so each (date, key) appears once.",
                "where_to_fix": ["snapshot.market.raw_quotes"],
                "details": {"duplicate_count": quote_duplicate_count},
            }
        )
    if not fixing_dates_not_after_asof:
        issues.append("some fixings are dated after the snapshot asof")
        action_items.append(
            {
                "code": "fixings_after_asof",
                "severity": "error",
                "what_failed": "Some fixing dates are later than the snapshot asof date.",
                "what_to_fix": "Remove future fixings or move the snapshot asof forward so the fixings are historical or same-day.",
                "where_to_fix": ["snapshot.fixings.points[*].date", "snapshot.config.asof"],
                "details": fixing_dates,
            }
        )
    if not analytics_supported:
        issues.append("config.analytics contains unsupported metrics or is empty")
        action_items.append(
            {
                "code": "unsupported_analytics",
                "severity": "error",
                "what_failed": "The analytics tuple contains unsupported metric names or no metrics at all.",
                "what_to_fix": "Use only supported metrics: CVA, DVA, FVA, MVA.",
                "where_to_fix": ["snapshot.config.analytics"],
                "details": {"invalid": invalid_metrics, "requested": list(analytics)},
            }
        )
    if not trade_ids_unique:
        issues.append("portfolio trade ids are not unique")
        action_items.append(
            {
                "code": "duplicate_trade_ids",
                "severity": "error",
                "what_failed": "More than one trade uses the same trade_id.",
                "what_to_fix": "Give each trade a unique trade_id in snapshot.portfolio.trades.",
                "where_to_fix": ["snapshot.portfolio.trades[*].trade_id"],
                "details": trade_ids,
            }
        )
    if not netting_sets_defined and defined_netting_sets:
        issues.append("some trade netting sets are missing from netting config")
        action_items.append(
            {
                "code": "missing_netting_sets",
                "severity": "error",
                "what_failed": "Some trades reference netting sets that are not defined in snapshot.netting.netting_sets.",
                "what_to_fix": "Add the missing NettingSet objects to snapshot.netting.netting_sets or change the trade netting_set values to an existing key.",
                "where_to_fix": ["snapshot.portfolio.trades[*].netting_set", "snapshot.netting.netting_sets"],
                "details": sorted(trade_netting_sets - defined_netting_sets),
            }
        )
    if not collateral_matches_netting:
        issues.append("some collateral balances reference unknown netting sets")
        action_items.append(
            {
                "code": "unknown_collateral_netting_sets",
                "severity": "error",
                "what_failed": "Some collateral balances reference netting sets that do not exist in the snapshot netting/trade set.",
                "what_to_fix": "Change CollateralBalance.netting_set_id to a real trade/netting-set id or add the missing netting set definition.",
                "where_to_fix": ["snapshot.collateral.balances[*].netting_set_id", "snapshot.netting.netting_sets"],
                "details": sorted(collateral_netting_sets - (defined_netting_sets or trade_netting_sets)),
            }
        )

    checks = {
        "asof_consistent": asof_consistent,
        "quote_dates_match_asof": quote_dates_match_asof,
        "fixing_dates_not_after_asof": fixing_dates_not_after_asof,
        "analytics_supported": analytics_supported,
        "trade_ids_unique": trade_ids_unique,
        "netting_sets_defined": netting_sets_defined,
        "collateral_matches_netting": collateral_matches_netting,
    }

    return {
        "summary": {
            "asof": config.asof,
            "base_currency": config.base_currency,
            "analytics": list(analytics),
            "trade_count": len(portfolio.trades),
            "quote_count": len(market.raw_quotes),
            "fixing_count": len(fixings.points),
            "counterparties": sorted(trade_counterparties),
            "netting_sets": sorted(trade_netting_sets),
        },
        "market": {
            "market_asof": market.asof,
            "config_asof": config.asof,
            "quote_dates": market_quote_dates,
            "quote_duplicate_count": quote_duplicate_count,
            "quote_key_count": len(set(quote_keys)),
        },
        "fixings": {
            "dates": fixing_dates,
            "count": len(fixings.points),
        },
        "portfolio": {
            "trade_ids": trade_ids,
            "counterparties": sorted(trade_counterparties),
            "netting_sets": sorted(trade_netting_sets),
        },
        "netting": {
            "defined_netting_sets": sorted(defined_netting_sets),
            "missing_trade_netting_sets": sorted(trade_netting_sets - defined_netting_sets) if defined_netting_sets else [],
        },
        "collateral": {
            "balance_netting_sets": sorted(collateral_netting_sets),
            "unknown_balance_netting_sets": sorted(collateral_netting_sets - (defined_netting_sets or trade_netting_sets)),
        },
        "analytics": {
            "requested": list(analytics),
            "invalid": invalid_metrics,
        },
        "checks": checks,
        "issues": issues,
        "action_items": action_items,
        "snapshot_valid": bool(all(checks.values()) and not issues),
    }


def xva_snapshot_validation_dataframe(report: Dict[str, object]):
    """Convert ``validate_xva_snapshot_dataclasses()`` output into a flat DataFrame."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for xva_snapshot_validation_dataframe()") from exc

    rows = []
    for section in ("market", "fixings", "portfolio", "netting", "collateral", "analytics", "checks"):
        payload = report.get(section, {})
        if isinstance(payload, dict):
            for key, value in payload.items():
                rows.append({"section": section, "field": key, "value": value})
    for idx, item in enumerate(report.get("action_items", [])):
        rows.append({"section": "action_items", "field": f"{idx}:code", "value": item.get("code")})
        rows.append({"section": "action_items", "field": f"{idx}:severity", "value": item.get("severity")})
        rows.append({"section": "action_items", "field": f"{idx}:what_failed", "value": item.get("what_failed")})
        rows.append({"section": "action_items", "field": f"{idx}:what_to_fix", "value": item.get("what_to_fix")})
        rows.append({"section": "action_items", "field": f"{idx}:where_to_fix", "value": item.get("where_to_fix")})
    return pd.DataFrame(rows, columns=["section", "field", "value"])


def extract_discount_factors_by_currency(
    ore_xml_path: str | Path,
    configuration_id: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Extract explicit discount-factor pillars for all configured currencies.

    Returns a JSON-friendly dictionary keyed by currency:
      {
        "USD": {
          "curve_id": "Yield/USD/USD-OIS",
          "source_column": "USD-OIS",
          "asof_date": "2016-02-05",
          "times": [...],
          "dfs": [...],
          "calendar_dates": [...]
        },
        ...
      }
    """
    ore_xml = Path(ore_xml_path).resolve()
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }

    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml}")

    base = ore_xml.parent
    run_dir = base.parent
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    curves_csv = output_path / "curves.csv"
    if not curves_csv.exists():
        raise FileNotFoundError(f"ORE output file not found (run ORE first): {curves_csv}")

    simulation_analytic = ore_root.find("./Analytics/Analytic[@type='simulation']")
    if simulation_analytic is None:
        raise ValueError(f"Missing Analytics/Analytic[@type='simulation'] in {ore_xml}")
    simulation_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in simulation_analytic.findall("./Parameter")
    }
    simulation_xml = (base / simulation_params.get("simulationConfigFile", "simulation.xml")).resolve()
    if not simulation_xml.exists():
        raise FileNotFoundError(f"simulation xml not found: {simulation_xml}")
    simulation_root = ET.parse(simulation_xml).getroot()
    model_day_counter = _normalize_day_counter_name(
        (simulation_root.findtext("./DayCounter") or "A365F").strip()
    )

    todaysmarket_rel = setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")
    todaysmarket_xml = (base / todaysmarket_rel).resolve()
    if not todaysmarket_xml.exists():
        raise FileNotFoundError(f"todaysmarket.xml not found: {todaysmarket_xml}")
    tm_root = ET.parse(todaysmarket_xml).getroot()

    if configuration_id is None:
        curves_analytic = ore_root.find("./Analytics/Analytic[@type='curves']")
        if curves_analytic is not None:
            cfg_params = {
                n.attrib.get("name", ""): (n.text or "").strip()
                for n in curves_analytic.findall("./Parameter")
            }
            configuration_id = cfg_params.get("configuration", "default")
        else:
            configuration_id = "default"

    mappings = _resolve_discount_columns_by_currency(tm_root, configuration_id)
    unique_columns = [v["source_column"] for v in mappings.values()]
    curve_data = _load_ore_discount_pairs_by_columns_with_day_counter(
        str(curves_csv), unique_columns, asof_date=asof_date, day_counter=model_day_counter
    )

    result: Dict[str, Dict[str, object]] = {}
    for ccy, meta in sorted(mappings.items()):
        dates, times, dfs = curve_data[meta["source_column"]]
        payload = CurveDFPayload(
            ccy=ccy,
            curve_id=meta["curve_id"],
            source_column=meta["source_column"],
            asof_date=asof_date,
            day_counter=model_day_counter,
            times=times,
            dfs=dfs,
            calendar_dates=dates,
        )
        result[ccy] = payload.to_dict()
    return result


def discount_factors_to_dataframe(
    discount_factors_by_ccy: Dict[str, Dict[str, object]],
):
    """Convert dict payload into a long-form DataFrame.

    Output columns: ccy, curve_id, time, df, asof_date, source_column
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for discount_factors_to_dataframe()"
        ) from exc

    rows = []
    for ccy, payload in discount_factors_by_ccy.items():
        times = payload.get("times", [])
        dfs = payload.get("dfs", [])
        if len(times) != len(dfs):
            raise ValueError(f"times/dfs length mismatch for currency '{ccy}'")
        for t, df in zip(times, dfs):
            rows.append(
                {
                    "ccy": ccy,
                    "curve_id": payload.get("curve_id", ""),
                    "time": float(t),
                    "df": float(df),
                    "asof_date": payload.get("asof_date", ""),
                    "source_column": payload.get("source_column", ""),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["ccy", "curve_id", "time", "df", "asof_date", "source_column"]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["ccy", "time"]).reset_index(drop=True)


def dump_discount_factors_json(
    ore_xml_path: str | Path,
    configuration_id: Optional[str] = None,
    indent: int = 2,
) -> str:
    """Convenience helper for CLI/json workflows."""
    payload = extract_discount_factors_by_currency(
        ore_xml_path=ore_xml_path,
        configuration_id=configuration_id,
    )
    return json.dumps(payload, indent=indent, sort_keys=True)


def _curve_spec_target(spec: str) -> tuple[Optional[tuple[str, ...]], Optional[str]]:
    parts = [p.strip() for p in str(spec).split("/") if p.strip()]
    if not parts:
        return None, None
    prefix = parts[0]
    parent_tags = _SPEC_PREFIX_TO_CURVECONFIG_TAGS.get(prefix)
    if parent_tags is None:
        return None, None
    if not parent_tags:
        return parent_tags, None
    if len(parts) < 2:
        return parent_tags, None
    return parent_tags, parts[-1]


def _collect_snapshot_relevance(
    portfolio_xml: Optional[Path],
    analytics_params: dict[str, dict[str, str]],
) -> tuple[set[str], set[str], set[str]]:
    currencies: set[str] = set()
    indices: set[str] = set()
    counterparties: set[str] = set()
    for analytic in analytics_params.values():
        base_ccy = (analytic.get("baseCurrency") or "").strip()
        if base_ccy:
            currencies.add(base_ccy)
        for key in ("cvaName", "dvaName"):
            name = (analytic.get(key) or "").strip()
            if name:
                counterparties.add(name)
    if portfolio_xml is None or not portfolio_xml.exists():
        return currencies, indices, counterparties

    root = ET.parse(portfolio_xml).getroot()
    for node in root.findall(".//Envelope/CounterParty"):
        value = (node.text or "").strip()
        if value:
            counterparties.add(value)
    for node in root.findall(".//FloatingLegData/Index"):
        value = (node.text or "").strip()
        if value:
            indices.add(value)
    for xpath in (
        ".//Currency",
        ".//PayCurrency",
        ".//ReceiveCurrency",
        ".//BoughtCurrency",
        ".//SoldCurrency",
    ):
        for node in root.findall(xpath):
            value = (node.text or "").strip()
            if value and len(value) == 3 and value.isalpha():
                currencies.add(value.upper())
    return currencies, indices, counterparties


def _split_ore_loader_line(line: str) -> list[str]:
    return [tok for tok in re.split(r"[,;\t ]+", line.strip()) if tok]


def _load_ore_csv_keys_by_date(path: Path) -> dict[str, set[str]]:
    by_date: dict[str, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = _split_ore_loader_line(line)
            if len(tokens) < 3:
                continue
            by_date[tokens[0]].add(tokens[1])
    return by_date


def _fx_dominance(ccy1: str, ccy2: str) -> str:
    if ccy1 == ccy2:
        return ccy1 + ccy2
    dominance = list(_FX_DOMINANCE_ORDER)
    try:
        p1 = dominance.index(ccy1)
    except ValueError:
        p1 = None
    try:
        p2 = dominance.index(ccy2)
    except ValueError:
        p2 = None
    if p1 is not None and p2 is not None:
        return ccy1 + ccy2 if p1 < p2 else ccy2 + ccy1
    if p1 is None and p2 is None:
        return ccy1 + ccy2
    if ccy1 == "JPY":
        return ccy2 + ccy1
    if ccy2 == "JPY":
        return ccy1 + ccy2
    return ccy1 + ccy2 if p1 is not None else ccy2 + ccy1


def extract_market_instruments_by_currency(
    ore_xml_path: str | Path,
    instrument_types: tuple[str, ...] = ("ZERO", "MM", "IR_SWAP"),
) -> Dict[str, Dict[str, object]]:
    """Extract ORE market instruments grouped by currency for quick external fitting.

    The output is JSON-friendly and includes instrument-level details.
    """
    _, asof_date, market_data_file, _, _ = _resolve_ore_run_files(ore_xml_path)
    if not market_data_file.exists():
        raise FileNotFoundError(f"market data file not found: {market_data_file}")

    allowed = {x.upper() for x in instrument_types}
    grouped: Dict[str, list[Dict[str, object]]] = defaultdict(list)
    with open(market_data_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 3:
                continue
            key = toks[1].strip()
            try:
                quote = float(toks[2])
            except ValueError:
                continue
            parsed = _parse_market_instrument_key(key)
            if parsed is None:
                continue
            instrument_type = str(parsed["instrument_type"]).upper()
            if instrument_type not in allowed and not (instrument_type.endswith("_FUTURE") and "FUTURE" in allowed):
                continue
            grouped[parsed["ccy"]].append(_build_instrument_record(asof_date, parsed, key, quote))

    out: Dict[str, Dict[str, object]] = {}
    for ccy, instruments in sorted(grouped.items()):
        by_type: Dict[str, int] = {}
        for x in instruments:
            t = str(x["instrument_type"])
            by_type[t] = by_type.get(t, 0) + 1
        out[ccy] = {
            "asof_date": asof_date,
            "instrument_count": len(instruments),
            "instrument_counts_by_type": by_type,
            "instruments": sorted(instruments, key=lambda x: (float(x["maturity"]), str(x["instrument_type"]))),
        }
    return out


def extract_market_instruments_by_currency_from_quotes(
    asof_date: str,
    quotes: list[dict[str, object] | tuple[str, float]],
    instrument_types: tuple[str, ...] = ("ZERO", "MM", "IR_SWAP"),
) -> Dict[str, Dict[str, object]]:
    """Programmatic instrument extraction (no XML/files).

    Parameters
    ----------
    asof_date:
        Curve as-of date in YYYY-MM-DD format.
    quotes:
        Sequence of quote records with at least keys:
          - ``key`` (str, e.g. ``IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/5Y``)
          - ``value`` (float)
    """
    allowed = {x.upper() for x in instrument_types}
    grouped: Dict[str, list[Dict[str, object]]] = defaultdict(list)
    for q in _normalize_programmatic_quotes(quotes):
        key = str(q.get("key", "")).strip()
        if not key:
            continue
        try:
            quote = float(q.get("value"))
        except Exception:
            continue
        parsed = _parse_market_instrument_key(key)
        if parsed is None:
            continue
        instrument_type = str(parsed["instrument_type"]).upper()
        if instrument_type not in allowed and not (instrument_type.endswith("_FUTURE") and "FUTURE" in allowed):
            continue
        grouped[parsed["ccy"]].append(_build_instrument_record_from_quote(asof_date, parsed, q))

    out: Dict[str, Dict[str, object]] = {}
    for ccy, instruments in sorted(grouped.items()):
        by_type: Dict[str, int] = {}
        for x in instruments:
            t = str(x["instrument_type"])
            by_type[t] = by_type.get(t, 0) + 1
        out[ccy] = {
            "asof_date": asof_date,
            "instrument_count": len(instruments),
            "instrument_counts_by_type": by_type,
            "instruments": sorted(instruments, key=lambda x: (float(x["maturity"]), str(x["instrument_type"]))),
        }
    return out


def fit_discount_curves_from_ore_market(
    ore_xml_path: str | Path,
    instrument_types: tuple[str, ...] = ("ZERO", "MM", "IR_SWAP"),
    fit_method: str = "weighted_zero_logdf_v1",
    fit_grid_mode: str = "instrument",
    dense_step_years: float = 0.25,
    future_convexity_mode: str = "external_adjusted_fra",
    future_model_params: RateFutureModelParams | dict[str, object] | None = None,
) -> Dict[str, Dict[str, object]]:
    """Fast per-currency curve fitter from ORE market quotes.

    This is designed as a quick external fitter input path:
      instruments in per currency -> fitted curve points out.
    """
    inst = extract_market_instruments_by_currency(
        ore_xml_path=ore_xml_path,
        instrument_types=instrument_types,
    )
    output: Dict[str, Dict[str, object]] = {}
    for ccy, payload in sorted(inst.items()):
        instruments = payload["instruments"]
        if not instruments:
            continue
        fit = _fit_curve_from_instruments(
            payload["asof_date"],
            instruments,
            fit_method=fit_method,
            fit_grid_mode=fit_grid_mode,
            dense_step_years=dense_step_years,
            future_convexity_mode=future_convexity_mode,
            future_model_params=future_model_params,
        )
        output[ccy] = {
            "asof_date": payload["asof_date"],
            "curve_method": fit_method,
            "fit_grid_mode": fit_grid_mode,
            "dense_step_years": float(dense_step_years),
            "instrument_count": payload["instrument_count"],
            "instrument_counts_by_type": payload["instrument_counts_by_type"],
            "times": [float(x) for x in fit["times"]],
            "zero_rates": [float(x) for x in fit["zero_rates"]],
            "dfs": [float(x) for x in fit["dfs"]],
            "calendar_dates": list(fit["calendar_dates"]),
            "fit_points_count": int(len(fit["times"])),
            "instrument_times": [float(x) for x in fit["instrument_times"]],
            "instrument_zero_rates": [float(x) for x in fit["instrument_zero_rates"]],
            "bootstrap_diagnostics": list(fit.get("bootstrap_diagnostics", [])),
            "future_convexity_mode": str(future_convexity_mode),
            "input_instruments": instruments,
        }
    return output


def fit_discount_curves_from_programmatic_quotes(
    asof_date: str,
    quotes: list[dict[str, object] | tuple[str, float]],
    instrument_types: tuple[str, ...] = ("ZERO", "MM", "IR_SWAP"),
    fit_method: str = "weighted_zero_logdf_v1",
    fit_grid_mode: str = "instrument",
    dense_step_years: float = 0.25,
    future_convexity_mode: str = "external_adjusted_fra",
    future_model_params: RateFutureModelParams | dict[str, object] | None = None,
) -> Dict[str, Dict[str, object]]:
    """Programmatic curve fitting API (no XML/files).

    Designed for external systems that already hold ORE-style quote keys in memory.
    """
    inst = extract_market_instruments_by_currency_from_quotes(
        asof_date=asof_date,
        quotes=quotes,
        instrument_types=instrument_types,
    )
    output: Dict[str, Dict[str, object]] = {}
    for ccy, payload in sorted(inst.items()):
        instruments = payload["instruments"]
        if not instruments:
            continue
        fit = _fit_curve_from_instruments(
            payload["asof_date"],
            instruments,
            fit_method=fit_method,
            fit_grid_mode=fit_grid_mode,
            dense_step_years=dense_step_years,
            future_convexity_mode=future_convexity_mode,
            future_model_params=future_model_params,
        )
        output[ccy] = {
            "asof_date": payload["asof_date"],
            "curve_method": fit_method,
            "fit_grid_mode": fit_grid_mode,
            "dense_step_years": float(dense_step_years),
            "instrument_count": payload["instrument_count"],
            "instrument_counts_by_type": payload["instrument_counts_by_type"],
            "times": [float(x) for x in fit["times"]],
            "zero_rates": [float(x) for x in fit["zero_rates"]],
            "dfs": [float(x) for x in fit["dfs"]],
            "calendar_dates": list(fit["calendar_dates"]),
            "fit_points_count": int(len(fit["times"])),
            "instrument_times": [float(x) for x in fit["instrument_times"]],
            "instrument_zero_rates": [float(x) for x in fit["instrument_zero_rates"]],
            "bootstrap_diagnostics": list(fit.get("bootstrap_diagnostics", [])),
            "future_convexity_mode": str(future_convexity_mode),
            "input_instruments": instruments,
        }
    return output


def quote_dicts_from_pairs(
    quote_pairs: list[tuple[str, float]],
) -> list[dict[str, object]]:
    """Small helper to build programmatic quote payloads from tuples.

    Example
    -------
    ``quote_dicts_from_pairs([("ZERO/RATE/USD/1Y", 0.05)])``
    """
    return [{"key": str(k), "value": float(v)} for k, v in quote_pairs]


def fitted_curves_to_dataframe(
    fitted_curves_by_ccy: Dict[str, Dict[str, object]],
):
    """Convert fitted per-currency curves into a long-form DataFrame."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for fitted_curves_to_dataframe()") from exc

    rows = []
    for ccy, payload in fitted_curves_by_ccy.items():
        times = payload.get("times", [])
        dfs = payload.get("dfs", [])
        zeros = payload.get("zero_rates", [])
        dates = payload.get("calendar_dates", [])
        n = len(times)
        if not (len(dfs) == n and len(zeros) == n and len(dates) == n):
            raise ValueError(f"fitted curve arrays length mismatch for currency '{ccy}'")
        for t, df, z, d in zip(times, dfs, zeros, dates):
            rows.append(
                {
                    "ccy": ccy,
                    "asof_date": payload.get("asof_date", ""),
                    "time": float(t),
                    "calendar_date": str(d),
                    "df": float(df),
                    "zero_rate": float(z),
                    "curve_method": payload.get("curve_method", ""),
                    "instrument_count": int(payload.get("instrument_count", 0)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "ccy",
                "asof_date",
                "time",
                "calendar_date",
                "df",
                "zero_rate",
                "curve_method",
                "instrument_count",
            ]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["ccy", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_from_ore_xml(
    ore_xml_path: str | Path,
    trade_id: Optional[str] = None,
    cpty: Optional[str] = None,
    anchor_t0_npv: bool = False,
) -> OreSnapshot:
    """Load the complete ORE input set from a single ore.xml entry point.

    Parameters
    ----------
    ore_xml_path:
        Absolute or relative path to the ORE root configuration file
        (e.g. ``ore_measure_lgm.xml``).
    trade_id:
        Trade identifier to load.  If *None*, defaults to the first trade in
        the portfolio XML.
    cpty:
        Counterparty name used to look up CVA in ``xva.csv`` and credit data
        in ``todaysmarket.xml`` / ``market_data.txt``.  If *None*, inferred
        from the trade's ``Envelope/CounterParty`` field.
    anchor_t0_npv:
        If *True*, apply a parallel float-spread shift to the swap legs so
        that the Python t=0 NPV matches ORE's reported t=0 NPV (from
        ``npv.csv``).  This only works correctly when ``discount_column`` and
        ``forward_column`` resolve to the **same** curve (single-curve setup),
        or when the two curves are very close in value.  For the standard
        dual-curve setup (EUR-EONIA disc + EUR-EURIBOR-6M fwd) the anchoring
        can degrade parity.  Defaults to *False*; use
        :func:`~py_ore_tools.irs_xva_utils.apply_parallel_float_spread_shift_to_match_npv`
        directly for fine-grained control.

    Returns
    -------
    OreSnapshot
        Fully populated snapshot ready for use with the Python LGM.

    Raises
    ------
    FileNotFoundError
        If any required ORE output file is missing (run ORE first).
    ValueError
        If XML structure does not match expected schema or a required field
        is absent.
    """
    ore_xml_path = Path(ore_xml_path).resolve()
    base = ore_xml_path.parent  # directory containing ore.xml (e.g. .../Exposure/Input)

    # In ORE the run directory is the *parent* of the inputPath directory.
    # If inputPath is "Input" and the ore.xml lives inside that directory,
    # the run directory is base.parent.  The outputPath is relative to it.
    # ------------------------------------------------------------------
    # 1. Parse ore.xml
    # ------------------------------------------------------------------
    ore_root = ET.parse(ore_xml_path).getroot()

    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    markets_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Markets/Parameter")
    }

    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml_path}")

    # inputPath tells us the sub-directory name of the Input folder.
    # ore.xml lives inside that folder, so the ORE run-directory is base.parent.
    # outputPath is relative to that run-directory.
    run_dir = base.parent
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()

    todaysmarket_rel = setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")
    market_data_rel = setup_params.get("marketDataFile", "../../Input/market_20160205_flat.txt")
    portfolio_rel = setup_params.get("portfolioFile", "portfolio_singleswap.xml")

    todaysmarket_xml = (base / todaysmarket_rel).resolve()
    market_data_file = (base / market_data_rel).resolve()
    portfolio_xml = (base / portfolio_rel).resolve()

    sim_config_id = markets_params.get("simulation", "libor")

    # Simulation config file from Analytics section
    sim_analytic = ore_root.find("./Analytics/Analytic[@type='simulation']")
    if sim_analytic is None:
        raise ValueError(f"Missing Analytics/Analytic[@type='simulation'] in {ore_xml_path}")
    sim_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in sim_analytic.findall("./Parameter")
    }
    sim_cfg_rel = sim_params.get("simulationConfigFile", "simulation_lgm.xml")
    simulation_xml = (base / sim_cfg_rel).resolve()

    # The curves.csv analytic configuration determines what discount-curve
    # column appears in curves.csv.  Resolving from this configuration (usually
    # "default" → OIS/EUR-EONIA) gives the discount curve that ORE uses in its
    # CVA aggregation, and produces the best EPE/CVA parity with the Python LGM.
    # Using the simulation config (e.g. "libor" → EUR-EURIBOR-6M) instead would
    # match ORE's *simulation-measure* pricing but not the CVA discounting.
    curves_analytic = ore_root.find("./Analytics/Analytic[@type='curves']")
    if curves_analytic is not None:
        curves_cfg_params = {
            n.attrib.get("name", ""): (n.text or "").strip()
            for n in curves_analytic.findall("./Parameter")
        }
        curves_config_id = curves_cfg_params.get("configuration", "default")
    else:
        curves_config_id = "default"

    # ------------------------------------------------------------------
    # 2. Parse simulation.xml
    # ------------------------------------------------------------------
    if not simulation_xml.exists():
        raise FileNotFoundError(f"simulation xml not found: {simulation_xml}")

    sim_root = ET.parse(simulation_xml).getroot()
    domestic_ccy = (sim_root.findtext("./DomesticCcy") or "EUR").strip()
    measure = (sim_root.findtext("./Measure") or "LGM").strip()
    model_day_counter = _normalize_day_counter_name(
        (sim_root.findtext("./DayCounter") or "A365F").strip()
    )
    report_day_counter = "ActualActual(ISDA)"
    seed = int((sim_root.findtext("./Seed") or "42").strip())
    n_samples = int((sim_root.findtext("./Samples") or "1000").strip())
    node_tenors = load_simulation_yield_tenors(str(simulation_xml))

    # ------------------------------------------------------------------
    # 3. Parse portfolio.xml: trade_id, cpty, netting_set, float_index
    # ------------------------------------------------------------------
    if not portfolio_xml.exists():
        raise FileNotFoundError(f"portfolio xml not found: {portfolio_xml}")

    portfolio_root = ET.parse(portfolio_xml).getroot()

    if trade_id is None:
        trade_id = _get_first_trade_id(portfolio_root)
    if cpty is None:
        cpty = _get_cpty_from_portfolio(portfolio_root, trade_id)

    netting_set_id = _get_netting_set_from_portfolio(portfolio_root, trade_id)
    forward_column = _get_float_index(portfolio_root, trade_id)

    # ------------------------------------------------------------------
    # 4. Parse todaysmarket.xml → discount curve column
    # ------------------------------------------------------------------
    if not todaysmarket_xml.exists():
        raise FileNotFoundError(f"todaysmarket.xml not found: {todaysmarket_xml}")

    tm_root = ET.parse(todaysmarket_xml).getroot()
    discount_column = _resolve_discount_column(tm_root, curves_config_id, domestic_ccy)
    xva_discount_column = _resolve_discount_column(tm_root, sim_config_id, domestic_ccy)

    xva_analytic = ore_root.find("./Analytics/Analytic[@type='xva']")
    xva_params = (
        {n.attrib.get("name", ""): (n.text or "").strip() for n in xva_analytic.findall("./Parameter")}
        if xva_analytic is not None else {}
    )
    requested_xva_metrics = tuple(
        metric
        for metric, enabled in (
            ("CVA", xva_params.get("cva", "Y")),
            ("DVA", xva_params.get("dva", "N")),
            ("FVA", xva_params.get("fva", "N")),
            ("MVA", xva_params.get("mva", "N")),
        )
        if str(enabled).strip().upper() == "Y"
    )
    borrowing_curve_column = (xva_params.get("fvaBorrowingCurve") or "").strip() or None
    lending_curve_column = (xva_params.get("fvaLendingCurve") or "").strip() or None

    # ------------------------------------------------------------------
    # 5. Determine LGM parameters (calibration.xml preferred)
    # ------------------------------------------------------------------
    calibration_xml = output_path / "calibration.xml"
    if calibration_xml.exists():
        try:
            params_dict = parse_lgm_params_from_calibration_xml(
                str(calibration_xml), ccy_key=domestic_ccy
            )
            alpha_source = "calibration"
        except (ValueError, ET.ParseError):
            params_dict = parse_lgm_params_from_simulation_xml(
                str(simulation_xml), ccy_key=domestic_ccy
            )
            alpha_source = "simulation"
    else:
        params_dict = parse_lgm_params_from_simulation_xml(
            str(simulation_xml), ccy_key=domestic_ccy
        )
        alpha_source = "simulation"

    lgm_params = LGMParams(
        alpha_times=tuple(float(x) for x in params_dict["alpha_times"]),
        alpha_values=tuple(float(x) for x in params_dict["alpha_values"]),
        kappa_times=tuple(float(x) for x in params_dict["kappa_times"]),
        kappa_values=tuple(float(x) for x in params_dict["kappa_values"]),
        shift=float(params_dict["shift"]),
        scaling=float(params_dict["scaling"]),
    )

    # ------------------------------------------------------------------
    # 6. Load output CSVs
    # ------------------------------------------------------------------
    curves_csv = output_path / "curves.csv"
    exposure_csv = output_path / f"exposure_trade_{trade_id}.csv"
    xva_csv = output_path / "xva.csv"
    npv_csv = output_path / "npv.csv"

    for f in (curves_csv, exposure_csv, xva_csv, npv_csv):
        if not f.exists():
            raise FileNotFoundError(
                f"ORE output file not found (run ORE first): {f}"
            )

    # Discount curve
    curve_dates_by_col = _load_ore_discount_pairs_by_columns_with_day_counter(
        str(curves_csv), [discount_column], asof_date=asof_date, day_counter=model_day_counter
    )
    _, curve_times_disc, curve_dfs_disc = curve_dates_by_col[discount_column]
    p0_disc = build_discount_curve_from_discount_pairs(
        list(zip(curve_times_disc, curve_dfs_disc))
    )

    # Forward curve (may equal discount curve if columns are the same)
    if forward_column == discount_column:
        curve_times_fwd = curve_times_disc
        curve_dfs_fwd = curve_dfs_disc
        p0_fwd = p0_disc
    else:
        _, curve_times_fwd, curve_dfs_fwd = _load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv), [forward_column], asof_date=asof_date, day_counter=model_day_counter
        )[forward_column]
        p0_fwd = build_discount_curve_from_discount_pairs(
            list(zip(curve_times_fwd, curve_dfs_fwd))
        )

    if xva_discount_column == discount_column:
        curve_times_xva_disc = curve_times_disc
        curve_dfs_xva_disc = curve_dfs_disc
        p0_xva_disc = p0_disc
    elif xva_discount_column == forward_column:
        curve_times_xva_disc = curve_times_fwd
        curve_dfs_xva_disc = curve_dfs_fwd
        p0_xva_disc = p0_fwd
    else:
        _, curve_times_xva_disc, curve_dfs_xva_disc = _load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv), [xva_discount_column], asof_date=asof_date, day_counter=model_day_counter
        )[xva_discount_column]
        p0_xva_disc = build_discount_curve_from_discount_pairs(
            list(zip(curve_times_xva_disc, curve_dfs_xva_disc))
        )

    curve_times_borrow = None
    curve_dfs_borrow = None
    p0_borrow = None
    if borrowing_curve_column:
        _, curve_times_borrow, curve_dfs_borrow = _load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv), [borrowing_curve_column], asof_date=asof_date, day_counter=model_day_counter
        )[borrowing_curve_column]
        p0_borrow = build_discount_curve_from_discount_pairs(
            list(zip(curve_times_borrow, curve_dfs_borrow))
        )

    curve_times_lend = None
    curve_dfs_lend = None
    p0_lend = None
    if lending_curve_column:
        _, curve_times_lend, curve_dfs_lend = _load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv), [lending_curve_column], asof_date=asof_date, day_counter=model_day_counter
        )[lending_curve_column]
        p0_lend = build_discount_curve_from_discount_pairs(
            list(zip(curve_times_lend, curve_dfs_lend))
        )

    # Exposure profile
    exposure = load_ore_exposure_profile(str(exposure_csv))
    exposure_times = exposure["time"]
    exposure_dates = exposure["date"]
    exposure_model_times = np.asarray(
        [_year_fraction_from_day_counter(asof_date, d, model_day_counter) for d in exposure_dates],
        dtype=float,
    )
    ore_epe = exposure["epe"]
    ore_ene = exposure["ene"]

    # ORE XVA (aggregate row for netting set: CVA, DVA, FBA, FCA)
    xva_row = _load_ore_xva_aggregate(xva_csv, cpty_or_netting=netting_set_id or cpty)
    ore_cva = xva_row["cva"]
    ore_dva = xva_row["dva"]
    ore_fba = xva_row["fba"]
    ore_fca = xva_row["fca"]

    # ORE t0 NPV
    ore_t0_npv = _load_ore_npv(npv_csv, trade_id=trade_id)

    # ------------------------------------------------------------------
    # 7. Build swap legs: prefer flows.csv when present (ORE Amount signs = canonical
    #    for parity; see SKILL.md "Use ORE cashflow signs as canonical truth").
    # ------------------------------------------------------------------
    flows_csv = output_path / "flows.csv"
    legs = None
    leg_source = "portfolio"
    if flows_csv.exists():
        try:
            legs = load_ore_legs_from_flows(
                str(flows_csv), trade_id=trade_id, asof_date=asof_date, time_day_counter=model_day_counter
            )
            leg_source = "flows"
        except (ValueError, FileNotFoundError):
            pass
    if legs is None:
        legs = load_swap_legs_from_portfolio(
            str(portfolio_xml), trade_id=trade_id, asof_date=asof_date, time_day_counter=model_day_counter
        )
    legs["node_tenors"] = node_tenors
    legs = calibrate_float_spreads_from_coupon(legs, p0_fwd, t0=0.0)

    # Optional: anchor Python t=0 NPV to ORE's reported NPV.
    # This applies a parallel float-spread shift that absorbs residual
    # day-count and curve-convention differences between ORE and Python.
    if anchor_t0_npv:
        legs = apply_parallel_float_spread_shift_to_match_npv(
            legs, p0_disc, ore_t0_npv, t0=0.0
        )

    # ------------------------------------------------------------------
    # 8. Load credit inputs (hazard + recovery): counterparty and own (DVA)
    # ------------------------------------------------------------------
    if not market_data_file.exists():
        raise FileNotFoundError(f"market data file not found: {market_data_file}")

    credit = load_ore_default_curve_inputs(
        str(todaysmarket_xml), str(market_data_file), cpty_name=cpty
    )

    # Own (bank) credit for DVA: from ore.xml dvaName when present in market
    dva_name = (xva_params.get("dvaName") or "BANK").strip() or "BANK"
    own_credit = None
    if dva_name and dva_name != cpty:
        try:
            own_credit = load_ore_default_curve_inputs(
                str(todaysmarket_xml), str(market_data_file), cpty_name=dva_name
            )
        except (ValueError, FileNotFoundError):
            pass

    return OreSnapshot(
        ore_xml_path=str(ore_xml_path),
        asof_date=asof_date,
        lgm_params=lgm_params,
        alpha_source=alpha_source,
        measure=measure,
        seed=seed,
        n_samples=n_samples,
        domestic_ccy=domestic_ccy,
        node_tenors=node_tenors,
        model_day_counter=model_day_counter,
        report_day_counter=report_day_counter,
        trade_id=trade_id,
        counterparty=cpty,
        netting_set_id=netting_set_id,
        legs=legs,
        discount_column=discount_column,
        forward_column=forward_column,
        xva_discount_column=xva_discount_column,
        curve_times_disc=curve_times_disc,
        curve_dfs_disc=curve_dfs_disc,
        curve_times_fwd=curve_times_fwd,
        curve_dfs_fwd=curve_dfs_fwd,
        curve_times_xva_disc=curve_times_xva_disc,
        curve_dfs_xva_disc=curve_dfs_xva_disc,
        borrowing_curve_column=borrowing_curve_column,
        lending_curve_column=lending_curve_column,
        curve_times_borrow=curve_times_borrow,
        curve_dfs_borrow=curve_dfs_borrow,
        curve_times_lend=curve_times_lend,
        curve_dfs_lend=curve_dfs_lend,
        p0_disc=p0_disc,
        p0_fwd=p0_fwd,
        p0_xva_disc=p0_xva_disc,
        p0_borrow=p0_borrow,
        p0_lend=p0_lend,
        exposure_times=exposure_times,
        exposure_dates=exposure_dates,
        exposure_model_times=exposure_model_times,
        ore_epe=ore_epe,
        ore_ene=ore_ene,
        ore_t0_npv=ore_t0_npv,
        ore_cva=ore_cva,
        recovery=float(credit["recovery"]),
        hazard_times=credit["hazard_times"],
        hazard_rates=credit["hazard_rates"],
        ore_dva=ore_dva,
        ore_fba=ore_fba,
        ore_fca=ore_fca,
        own_hazard_times=own_credit["hazard_times"] if own_credit else None,
        own_hazard_rates=own_credit["hazard_rates"] if own_credit else None,
        own_recovery=float(own_credit["recovery"]) if own_credit else None,
        leg_source=leg_source,
        requested_xva_metrics=requested_xva_metrics,
        own_name=dva_name,
        portfolio_xml_path=str(portfolio_xml),
        todaysmarket_xml_path=str(todaysmarket_xml),
        market_data_path=str(market_data_file),
        simulation_xml_path=str(simulation_xml),
        calibration_xml_path=str(calibration_xml) if calibration_xml.exists() else None,
        curves_csv_path=str(curves_csv),
        exposure_csv_path=str(exposure_csv),
        xva_csv_path=str(xva_csv),
        npv_csv_path=str(npv_csv),
        flows_csv_path=str(flows_csv) if flows_csv.exists() else None,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_first_trade_id(portfolio_root: ET.Element) -> str:
    trade = portfolio_root.find("./Trade")
    if trade is not None:
        tid = trade.attrib.get("id", "").strip()
        if tid:
            return tid
    raise ValueError("No trades found in portfolio XML")


def _get_cpty_from_portfolio(portfolio_root: ET.Element, trade_id: str) -> str:
    for trade in portfolio_root.findall("./Trade"):
        if trade.attrib.get("id", "") == trade_id:
            cpty = (trade.findtext("./Envelope/CounterParty") or "").strip()
            if cpty:
                return cpty
    raise ValueError(f"CounterParty not found for trade '{trade_id}'")


def _get_netting_set_from_portfolio(portfolio_root: ET.Element, trade_id: str) -> str:
    for trade in portfolio_root.findall("./Trade"):
        if trade.attrib.get("id", "") == trade_id:
            ns = (trade.findtext("./Envelope/NettingSetId") or "").strip()
            return ns
    return ""


def _get_float_index(portfolio_root: ET.Element, trade_id: str) -> str:
    """Return the floating index name from the swap's FloatingLegData.

    This name is also the column header in curves.csv for the forwarding curve.
    """
    for trade in portfolio_root.findall("./Trade"):
        if trade.attrib.get("id", "") == trade_id:
            idx = trade.findtext("./SwapData/LegData/FloatingLegData/Index")
            if idx is not None:
                return idx.strip()
    raise ValueError(
        f"FloatingLegData/Index not found for trade '{trade_id}' in portfolio XML"
    )


def _handle_to_curve_name(tm_root: ET.Element, handle: str) -> str:
    """Resolve a curve handle (e.g. 'Yield/EUR/EUR6M') to its curves.csv column name.

    Searches both YieldCurves groups and IndexForwardingCurves groups because
    the curves.csv column name is the ``name`` attribute of either a YieldCurve
    or an Index element, whichever links to the handle.
    """
    # 1. Search YieldCurves groups
    for yc_group in tm_root.findall("./YieldCurves"):
        for yc in yc_group.findall("./YieldCurve"):
            if (yc.text or "").strip() == handle:
                name = yc.attrib.get("name", "").strip()
                if name:
                    return name

    # 2. Search IndexForwardingCurves groups (covers OIS-discount scenarios)
    for fc_group in tm_root.findall("./IndexForwardingCurves"):
        for idx_elem in fc_group.findall("./Index"):
            if (idx_elem.text or "").strip() == handle:
                name = idx_elem.attrib.get("name", "").strip()
                if name:
                    return name

    # Some benchmark curves.csv files collapse cross-currency discount handles
    # like ``Yield/USD/USD-IN-EUR`` down to the bare currency column ``USD``
    # instead of exposing a separate ``USD-IN-EUR`` column. When the explicit
    # XML indirection is absent, fall back to that convention so the Python
    # parity tools can still consume ORE output curves.
    parts = [p.strip() for p in handle.split("/") if p.strip()]
    if len(parts) == 3 and parts[0] == "Yield" and "-IN-" in parts[2]:
        ccy = parts[1].upper()
        if parts[2].upper().startswith(f"{ccy}-IN-"):
            return ccy

    raise ValueError(
        f"Could not resolve column name for curve handle '{handle}' in todaysmarket.xml. "
        f"Searched all YieldCurves and IndexForwardingCurves groups."
    )


def _resolve_discount_column(
    tm_root: ET.Element,
    sim_config_id: str,
    domestic_ccy: str,
) -> str:
    """Traverse the todaysmarket.xml XML chain to find the curves.csv column name
    for the discount curve of *domestic_ccy* under the *sim_config_id* market config.

    Chain:
      Configuration[@id=sim_config_id]
        → DiscountingCurvesId (e.g. "inccy_swap")
      DiscountingCurves[@id=disc_curves_id]
        /DiscountingCurve[@currency=domestic_ccy]
        → curve handle  (e.g. "Yield/EUR/EUR6M")
      YieldCurves or IndexForwardingCurves
        /[element with text == handle]
        → @name  (e.g. "EUR-EURIBOR-6M")  ← this is the curves.csv column
    """
    cfg = tm_root.find(f"./Configuration[@id='{sim_config_id}']")
    if cfg is None:
        raise ValueError(
            f"todaysmarket.xml has no Configuration[@id='{sim_config_id}']. "
            f"Check that Markets/simulation in ore.xml matches a Configuration id."
        )

    disc_curves_id = (cfg.findtext("./DiscountingCurvesId") or "").strip()
    if not disc_curves_id:
        raise ValueError(
            f"Configuration[@id='{sim_config_id}'] is missing DiscountingCurvesId"
        )

    disc_curves = tm_root.find(f"./DiscountingCurves[@id='{disc_curves_id}']")
    if disc_curves is None:
        raise ValueError(
            f"todaysmarket.xml has no DiscountingCurves[@id='{disc_curves_id}']"
        )

    handle_elem = disc_curves.find(f"./DiscountingCurve[@currency='{domestic_ccy}']")
    if handle_elem is None:
        raise ValueError(
            f"DiscountingCurves[@id='{disc_curves_id}'] has no "
            f"DiscountingCurve[@currency='{domestic_ccy}']"
        )

    handle = (handle_elem.text or "").strip()
    if not handle:
        raise ValueError(
            f"DiscountingCurve[@currency='{domestic_ccy}'] handle is empty"
        )

    return _handle_to_curve_name(tm_root, handle)


def _resolve_discount_columns_by_currency(
    tm_root: ET.Element,
    config_id: str,
) -> Dict[str, Dict[str, str]]:
    """Resolve discounting curve handles and curves.csv columns for all currencies."""
    cfg = tm_root.find(f"./Configuration[@id='{config_id}']")
    if cfg is None:
        raise ValueError(
            f"todaysmarket.xml has no Configuration[@id='{config_id}']. "
            f"Check analytics configuration id."
        )

    disc_curves_id = (cfg.findtext("./DiscountingCurvesId") or "").strip()
    if not disc_curves_id:
        raise ValueError(
            f"Configuration[@id='{config_id}'] is missing DiscountingCurvesId"
        )

    disc_curves = tm_root.find(f"./DiscountingCurves[@id='{disc_curves_id}']")
    if disc_curves is None:
        raise ValueError(
            f"todaysmarket.xml has no DiscountingCurves[@id='{disc_curves_id}']"
        )

    out: Dict[str, Dict[str, str]] = {}
    for node in disc_curves.findall("./DiscountingCurve"):
        ccy = (node.attrib.get("currency", "") or "").strip()
        handle = (node.text or "").strip()
        if not ccy or not handle:
            continue
        out[ccy] = {
            "curve_id": handle,
            "source_column": _handle_to_curve_name(tm_root, handle),
        }

    if not out:
        raise ValueError(
            f"DiscountingCurves[@id='{disc_curves_id}'] has no valid DiscountingCurve nodes"
        )
    return out


def _normalize_date_input(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _days_in_year(year: int) -> int:
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


def _year_fraction_actual_actual_isda(start: date, end: date) -> float:
    if end == start:
        return 0.0
    sign = 1.0
    if end < start:
        start, end = end, start
        sign = -1.0
    total = 0.0
    cur = start
    while cur.year < end.year:
        year_end = date(cur.year + 1, 1, 1)
        total += (year_end - cur).days / _days_in_year(cur.year)
        cur = year_end
    total += (end - cur).days / _days_in_year(cur.year)
    return sign * total


def _normalize_day_counter_name(day_counter: str) -> str:
    dc = str(day_counter).strip().upper().replace(" ", "")
    aliases = {
        "A365": "A365F",
        "ACTUAL/365(FIXED)": "A365F",
        "ACTUAL365FIXED": "A365F",
        "ACT/365(FIXED)": "A365F",
        "A365F": "A365F",
        "ACTUALACTUAL(ISDA)": "ActualActual(ISDA)",
        "ACT/ACT(ISDA)": "ActualActual(ISDA)",
        "ACTUALACTUALISDA": "ActualActual(ISDA)",
    }
    return aliases.get(dc, day_counter)


def _year_fraction_from_day_counter(
    start: str | date,
    end: str | date,
    day_counter: str,
) -> float:
    start_date = _normalize_date_input(start)
    end_date = _normalize_date_input(end)
    dc = _normalize_day_counter_name(day_counter)
    if dc == "A365F":
        return (end_date - start_date).days / 365.0
    if dc == "ActualActual(ISDA)":
        return _year_fraction_actual_actual_isda(start_date, end_date)
    raise ValueError(f"Unsupported day counter '{day_counter}'")


def _date_from_time_with_day_counter(asof_date: str, t: float, day_counter: str) -> str:
    base = _normalize_date_input(asof_date)
    dc = _normalize_day_counter_name(day_counter)
    if dc == "A365F":
        return (base + timedelta(days=int(round(float(t) * 365.0)))).isoformat()
    if dc == "ActualActual(ISDA)":
        target = float(t)
        lo = base
        hi = base + timedelta(days=max(int(np.ceil(abs(target) * 366.0)) + 3, 3))
        if target < 0.0:
            hi = base
            lo = base - timedelta(days=max(int(np.ceil(abs(target) * 366.0)) + 3, 3))
        best = base
        best_err = abs(_year_fraction_from_day_counter(base, base, dc) - target)
        cur = lo
        while cur <= hi:
            err = abs(_year_fraction_from_day_counter(base, cur, dc) - target)
            if err < best_err:
                best = cur
                best_err = err
            cur += timedelta(days=1)
        return best.isoformat()
    raise ValueError(f"Unsupported day counter '{day_counter}'")


def _load_ore_discount_pairs_by_columns_with_day_counter(
    curves_csv: str,
    discount_columns: list[str],
    *,
    asof_date: str,
    day_counter: str,
) -> Dict[str, tuple[tuple[str, ...], np.ndarray, np.ndarray]]:
    columns = [c for c in discount_columns if c]
    if not columns:
        raise ValueError("discount_columns must contain at least one column name")

    requested = list(dict.fromkeys(columns))
    rows = []
    with open(curves_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("curves.csv appears empty")
    if "Date" not in rows[0]:
        raise ValueError("curves.csv missing Date column")
    missing = [c for c in requested if c not in rows[0]]
    if missing:
        raise ValueError(f"curves.csv missing requested discount columns: {missing}")

    base_dates: list[str] = []
    times: list[float] = []
    by_col: Dict[str, list[float]] = {c: [] for c in requested}
    for r in rows:
        d = str(r["Date"])
        base_dates.append(d)
        times.append(_year_fraction_from_day_counter(asof_date, d, day_counter))
        for c in requested:
            by_col[c].append(float(r[c]))

    times_arr = np.asarray(times, dtype=float)
    uniq_t, idx = np.unique(times_arr, return_index=True)
    uniq_dates = tuple(base_dates[i] for i in idx)

    out: Dict[str, tuple[tuple[str, ...], np.ndarray, np.ndarray]] = {}
    for c in requested:
        dfs_arr = np.asarray(by_col[c], dtype=float)
        uniq_df = dfs_arr[idx].copy()
        dates = uniq_dates
        if uniq_t[0] > 1.0e-12:
            t = np.insert(uniq_t, 0, 0.0)
            df = np.insert(uniq_df, 0, 1.0)
            dates = (asof_date,) + dates
        else:
            t = uniq_t.copy()
            df = uniq_df
            if t[0] == 0.0:
                df[0] = 1.0
                dates = (asof_date,) + dates[1:]
        out[c] = (dates, t, df)
    return out


def _calendar_dates_from_times(
    asof_date: str,
    times: np.ndarray,
    day_counter: str = "A365F",
) -> tuple[str, ...]:
    return tuple(_date_from_time_with_day_counter(asof_date, float(t), day_counter) for t in times)


def _resolve_ore_run_files(
    ore_xml_path: str | Path,
) -> tuple[Path, str, Path, Path, Path]:
    """Resolve core ORE files from ore.xml."""
    ore_xml = Path(ore_xml_path).resolve()
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml}")
    base = ore_xml.parent
    run_dir = base.parent
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    todaysmarket_xml = (base / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    market_data_file = (base / setup_params.get("marketDataFile", "../../Input/market_20160205_flat.txt")).resolve()
    portfolio_xml = (base / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    return ore_xml, asof_date, market_data_file, todaysmarket_xml, output_path


def _parse_market_instrument_key(key: str) -> Optional[Dict[str, object]]:
    parts = key.strip().upper().split("/")
    if len(parts) < 4:
        return None

    # ZERO/RATE/CCY/TENOR
    if parts[0] == "ZERO" and parts[1] == "RATE":
        ccy = parts[2]
        tenor = parts[-1]
        maturity = _safe_tenor_years(tenor)
        if maturity is None:
            return None
        return {
            "instrument_type": "ZERO",
            "ccy": ccy,
            "tenor": tenor,
            "maturity": maturity,
            "index": "",
        }

    # MM/RATE/CCY/.../TENOR
    if parts[0] == "MM" and parts[1] == "RATE":
        ccy = parts[2]
        tenor = parts[-1]
        maturity = _safe_tenor_years(tenor)
        if maturity is None:
            return None
        idx = parts[3] if len(parts) > 4 else ""
        return {
            "instrument_type": "MM",
            "ccy": ccy,
            "tenor": tenor,
            "maturity": maturity,
            "index": idx,
        }

    # IR_SWAP/RATE/CCY/.../TENOR
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        ccy = parts[2]
        tenor = parts[-1]
        maturity = _safe_tenor_years(tenor)
        if maturity is None:
            return None
        idx = parts[4] if len(parts) > 5 else (parts[3] if len(parts) > 4 else "")
        return {
            "instrument_type": "IR_SWAP",
            "ccy": ccy,
            "tenor": tenor,
            "maturity": maturity,
            "index": idx,
        }
    if parts[0] in ("MM_FUTURE", "OI_FUTURE") and parts[1] == "PRICE" and len(parts) >= 6:
        ccy = parts[2]
        return {
            "instrument_type": parts[0],
            "ccy": ccy,
            "tenor": parts[-1],
            "maturity": None,
            "index": "",
            "contract": parts[3],
            "exchange": parts[4],
        }
    return None


def _build_instrument_record(
    asof_date: str,
    parsed: Dict[str, object],
    key: str,
    quote: float,
) -> Dict[str, object]:
    instrument_type = str(parsed["instrument_type"]).upper()
    record: Dict[str, object] = {
        "instrument_type": instrument_type,
        "quote_key": key,
        "quote_value": quote,
        "tenor": parsed["tenor"],
        "maturity": parsed["maturity"],
        "index": parsed["index"],
        "asof_date": asof_date,
    }
    if instrument_type in ("MM_FUTURE", "OI_FUTURE"):
        future_quote = build_rate_future_quote(
            asof_date=_normalize_date_input(asof_date),
            future_type=instrument_type,
            ccy=str(parsed["ccy"]),
            price=float(quote),
            contract_label=str(parsed.get("contract", "")),
            underlying_tenor=str(parsed["tenor"]),
            quote_key=key,
            exchange=str(parsed.get("exchange", "")),
        )
        record.update(
            {
                "maturity": future_quote.end_time_years,
                "contract": future_quote.contract_label,
                "exchange": future_quote.exchange,
                "contract_start": future_quote.contract_start.isoformat(),
                "contract_end": future_quote.contract_end.isoformat(),
                "start_time": future_quote.start_time_years,
                "end_time": future_quote.end_time_years,
                "accrual": future_quote.accrual_years,
            }
        )
    return record


def _build_instrument_record_from_quote(
    asof_date: str,
    parsed: Dict[str, object],
    quote_record: Dict[str, object],
) -> Dict[str, object]:
    key = str(quote_record.get("key", "")).strip()
    quote = float(quote_record["value"])
    record = _build_instrument_record(asof_date, parsed, key, quote)
    if str(parsed["instrument_type"]).upper() not in ("MM_FUTURE", "OI_FUTURE"):
        return record

    start_override = quote_record.get("contract_start")
    end_override = quote_record.get("contract_end")
    future_quote = build_rate_future_quote(
        asof_date=_normalize_date_input(asof_date),
        future_type=str(parsed["instrument_type"]),
        ccy=str(parsed["ccy"]),
        price=float(quote),
        contract_label=str(quote_record.get("contract", parsed.get("contract", ""))),
        underlying_tenor=str(quote_record.get("tenor", parsed["tenor"])),
        quote_key=key,
        exchange=str(quote_record.get("exchange", parsed.get("exchange", ""))),
        index=str(quote_record.get("index", parsed.get("index", ""))),
        contract_start=_normalize_date_input(start_override) if start_override else None,
        contract_end=_normalize_date_input(end_override) if end_override else None,
        convexity_adjustment=float(quote_record.get("convexity_adjustment", 0.0) or 0.0),
    )
    record.update(
        {
            "index": future_quote.index,
            "tenor": future_quote.underlying_tenor,
            "maturity": future_quote.end_time_years,
            "contract": future_quote.contract_label,
            "exchange": future_quote.exchange,
            "contract_start": future_quote.contract_start.isoformat(),
            "contract_end": future_quote.contract_end.isoformat(),
            "start_time": future_quote.start_time_years,
            "end_time": future_quote.end_time_years,
            "accrual": future_quote.accrual_years,
            "convexity_adjustment": future_quote.convexity_adjustment,
        }
    )
    return record


def _normalize_programmatic_quotes(
    quotes: list[dict[str, object] | tuple[str, float]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for q in quotes:
        if isinstance(q, tuple) and len(q) == 2:
            out.append({"key": str(q[0]), "value": float(q[1])})
            continue
        if isinstance(q, dict):
            out.append(q)
    return out


def _coerce_future_model_params(
    value: RateFutureModelParams | dict[str, object] | None,
) -> RateFutureModelParams | None:
    if value is None or isinstance(value, RateFutureModelParams):
        return value
    return RateFutureModelParams(
        model=str(value.get("model", "none")),
        mean_reversion=float(value.get("mean_reversion", 0.03)),
        volatility=float(value.get("volatility", 0.0)),
    )


def _ccy_from_quote_key(key: str) -> str:
    parts = str(key).strip().upper().split("/")
    return parts[2] if len(parts) >= 3 else ""


def _safe_tenor_years(tenor: str) -> Optional[float]:
    try:
        from .irs_xva_utils import parse_tenor_to_years

        t = float(parse_tenor_to_years(tenor))
        if t <= 0.0:
            return None
        return t
    except Exception:
        return None


def _fit_curve_from_instruments(
    asof_date: str,
    instruments: list[Dict[str, object]],
    fit_method: str = "weighted_zero_logdf_v1",
    fit_grid_mode: str = "instrument",
    dense_step_years: float = 0.25,
    future_convexity_mode: str = "external_adjusted_fra",
    future_model_params: RateFutureModelParams | dict[str, object] | None = None,
) -> Dict[str, object]:
    method = str(fit_method).strip().lower()
    diagnostics: list[Dict[str, object]] = []
    if method == "weighted_zero_logdf_v1":
        times, zeros = _fit_weighted_zero_nodes(instruments)
    elif method == "bootstrap_mm_irs_v1":
        times, zeros, diagnostics = _fit_bootstrap_mm_irs_nodes(
            asof_date,
            instruments,
            future_convexity_mode=future_convexity_mode,
            future_model_params=future_model_params,
        )
    elif method == "ql_helper_eur_v1":
        times, zeros = _fit_quantlib_helper_eur_nodes(asof_date, instruments)
    else:
        raise ValueError(
            "fit_method must be 'weighted_zero_logdf_v1', 'bootstrap_mm_irs_v1' or 'ql_helper_eur_v1'"
        )

    instr_t = np.asarray(times, dtype=float)
    instr_z = np.asarray(zeros, dtype=float)

    mode = str(fit_grid_mode).strip().lower()
    if mode not in ("instrument", "dense"):
        raise ValueError("fit_grid_mode must be 'instrument' or 'dense'")

    if mode == "dense":
        if dense_step_years <= 0.0:
            raise ValueError("dense_step_years must be > 0")
        t_max = float(instr_t[-1]) if instr_t.size > 0 else 0.0
        if t_max <= 0.0:
            t_arr = instr_t.copy()
            z_arr = instr_z.copy()
        else:
            n = int(np.floor(t_max / dense_step_years))
            dense = np.arange(0, n + 1, dtype=float) * float(dense_step_years)
            if dense.size == 0 or abs(dense[0]) > 1.0e-12:
                dense = np.insert(dense, 0, 0.0)
            if dense[-1] < t_max - 1.0e-12:
                dense = np.append(dense, t_max)
            z_dense = np.interp(dense, instr_t, instr_z, left=instr_z[0], right=instr_z[-1])
            t_arr = dense
            z_arr = z_dense
    else:
        t_arr = instr_t.copy()
        z_arr = instr_z.copy()

    dfs = np.exp(-z_arr * t_arr)
    # Basic monotonic clean-up for quick fitting workflows.
    for i in range(1, dfs.size):
        if dfs[i] > dfs[i - 1]:
            dfs[i] = dfs[i - 1]
            z_arr[i] = -np.log(max(dfs[i], 1.0e-12)) / max(t_arr[i], 1.0e-12)

    return {
        "times": t_arr,
        "zero_rates": z_arr,
        "dfs": dfs,
        "calendar_dates": _calendar_dates_from_times(asof_date, t_arr),
        "instrument_times": instr_t,
        "instrument_zero_rates": instr_z,
        "bootstrap_diagnostics": diagnostics,
    }


def _fit_weighted_zero_nodes(
    instruments: list[Dict[str, object]],
) -> tuple[list[float], list[float]]:
    # Aggregate quotes into a single zero-rate point per maturity.
    weight = {"ZERO": 1.0, "MM": 0.9, "IR_SWAP": 0.8}
    bucket: Dict[float, list[tuple[float, float]]] = defaultdict(list)
    for ins in instruments:
        t = float(ins["maturity"])
        r = float(ins["quote_value"])
        w = weight.get(str(ins["instrument_type"]), 0.5)
        bucket[t].append((r, w))

    times = [0.0]
    zeros = [0.0]
    for t in sorted(bucket.keys()):
        pairs = bucket[t]
        num = sum(r * w for r, w in pairs)
        den = sum(w for _, w in pairs)
        z = num / den if den > 0.0 else pairs[0][0]
        times.append(float(t))
        zeros.append(float(z))
    return times, zeros


def _fit_bootstrap_mm_irs_nodes(
    asof_date: str,
    instruments: list[Dict[str, object]],
    *,
    future_convexity_mode: str = "external_adjusted_fra",
    future_model_params: RateFutureModelParams | dict[str, object] | None = None,
) -> tuple[list[float], list[float], list[Dict[str, object]]]:
    def _avg_rate(items: list[Dict[str, object]]) -> float:
        return float(sum(float(x["quote_value"]) for x in items) / max(len(items), 1))

    by_type: Dict[str, list[Dict[str, object]]] = defaultdict(list)
    for ins in instruments:
        by_type[str(ins["instrument_type"]).upper()].append(ins)

    model_params = _coerce_future_model_params(future_model_params)
    future_mode = str(future_convexity_mode).strip().lower()
    if future_mode not in ("external_adjusted_fra", "native_future"):
        raise ValueError("future_convexity_mode must be 'external_adjusted_fra' or 'native_future'")

    known_df: Dict[float, float] = {0.0: 1.0}
    diagnostics: list[Dict[str, object]] = []

    # Seed with MM and ZERO quotes: treat as simple direct discount anchors.
    for tpe in ("MM", "ZERO"):
        buckets: Dict[float, list[Dict[str, object]]] = defaultdict(list)
        for ins in by_type.get(tpe, []):
            buckets[float(ins["maturity"])].append(ins)
        for t in sorted(buckets.keys()):
            r = _avg_rate(buckets[t])
            df = np.exp(-r * t) if t > 0.0 else 1.0
            known_df[t] = min(known_df.get(t, 1.0), float(df))

    def _interp_df(t: float) -> float:
        if t <= 0.0:
            return 1.0
        xs = sorted(known_df.keys())
        if t <= xs[0]:
            return known_df[xs[0]]
        if t >= xs[-1]:
            return known_df[xs[-1]]
        for i in range(1, len(xs)):
            if xs[i] >= t:
                t1, t2 = xs[i - 1], xs[i]
                p1, p2 = known_df[t1], known_df[t2]
                w = (t - t1) / max(t2 - t1, 1.0e-12)
                return float(np.exp((1.0 - w) * np.log(max(p1, 1.0e-12)) + w * np.log(max(p2, 1.0e-12))))
        return known_df[xs[-1]]

    future_instruments = by_type.get("MM_FUTURE", []) + by_type.get("OI_FUTURE", [])
    future_instruments = sorted(
        future_instruments,
        key=lambda x: (float(x.get("start_time", x.get("maturity", 0.0)) or 0.0), float(x.get("maturity", 0.0) or 0.0)),
    )

    for ins in future_instruments:
        future_quote = build_rate_future_quote(
            asof_date=_normalize_date_input(asof_date),
            future_type=str(ins["instrument_type"]),
            ccy=_ccy_from_quote_key(str(ins.get("quote_key", ""))),
            price=float(ins["quote_value"]),
            contract_label=str(ins.get("contract", "")),
            underlying_tenor=str(ins.get("tenor", "")),
            quote_key=str(ins.get("quote_key", "")),
            exchange=str(ins.get("exchange", "")),
            index=str(ins.get("index", "")),
            contract_start=_normalize_date_input(str(ins["contract_start"])) if ins.get("contract_start") else None,
            contract_end=_normalize_date_input(str(ins["contract_end"])) if ins.get("contract_end") else None,
            convexity_adjustment=float(ins.get("convexity_adjustment", 0.0) or 0.0),
        )
        forward_rate, futures_rate, convexity = future_forward_rate(future_quote, model_params)
        df_start = _interp_df(future_quote.start_time_years)
        df_end = df_start / max(1.0 + forward_rate * future_quote.accrual_years, 1.0e-12)
        df_end = float(np.clip(df_end, 1.0e-12, df_start))
        known_df[future_quote.end_time_years] = min(known_df.get(future_quote.end_time_years, 1.0), df_end)
        diagnostics.append(
            {
                "instrument_type": str(ins["instrument_type"]),
                "pricing_mode": future_mode,
                "quote_key": str(ins.get("quote_key", "")),
                "contract_start": future_quote.contract_start.isoformat(),
                "contract_end": future_quote.contract_end.isoformat(),
                "start_time": future_quote.start_time_years,
                "end_time": future_quote.end_time_years,
                "price": float(ins["quote_value"]),
                "futures_rate": futures_rate,
                "convexity_adjustment": convexity,
                "adjusted_forward_rate": future_to_fra_rate(future_quote, model_params),
            }
        )

    # Bootstrap swap maturities ascending with annual fixed leg schedule.
    swap_buckets: Dict[float, list[Dict[str, object]]] = defaultdict(list)
    for ins in by_type.get("IR_SWAP", []):
        swap_buckets[float(ins["maturity"])].append(ins)

    for T in sorted(swap_buckets.keys()):
        S = _avg_rate(swap_buckets[T])
        if T <= 0.0:
            continue
        # Annual schedule with final stub at T if needed.
        sched = []
        k = 1
        while float(k) < T - 1.0e-12:
            sched.append(float(k))
            k += 1
        sched.append(T)
        prev = 0.0
        accruals = []
        for x in sched:
            accruals.append(x - prev)
            prev = x
        # Par swap approximation: 1 - P(T) = S * sum(alpha_i * P(t_i))
        if len(sched) == 1:
            alpha_n = accruals[0]
            numer = 1.0
            denom = 1.0 + S * alpha_n
            pT = numer / max(denom, 1.0e-12)
        else:
            known_leg = 0.0
            for ti, ai in zip(sched[:-1], accruals[:-1]):
                known_leg += ai * _interp_df(ti)
            alpha_n = accruals[-1]
            pT = (1.0 - S * known_leg) / max(1.0 + S * alpha_n, 1.0e-12)
        pT = float(np.clip(pT, 1.0e-12, 1.0))
        if T in known_df:
            known_df[T] = min(known_df[T], pT)
        else:
            known_df[T] = pT

    times = sorted(known_df.keys())
    zeros = []
    for t in times:
        if t <= 0.0:
            zeros.append(0.0)
        else:
            zeros.append(float(-np.log(max(known_df[t], 1.0e-12)) / t))
    return times, zeros, diagnostics


def _fit_quantlib_helper_eur_nodes(
    asof_date: str,
    instruments: list[Dict[str, object]],
) -> tuple[list[float], list[float]]:
    try:
        import QuantLib as ql
    except Exception as exc:
        raise RuntimeError("QuantLib is required for fit_method='ql_helper_eur_v1'") from exc

    if not instruments:
        return [0.0], [0.0]

    ccy = ""
    first_key = str(instruments[0].get("quote_key", "")).upper()
    parts = first_key.split("/")
    if len(parts) >= 3:
        ccy = parts[2]
    if ccy != "EUR":
        raise ValueError("fit_method='ql_helper_eur_v1' currently supports EUR only")

    year, month, day = (int(x) for x in asof_date.split("-"))
    ref = ql.Date(day, month, year)
    ql.Settings.instance().evaluationDate = ref

    cal = ql.TARGET()
    fixed_dc = ql.Thirty360(ql.Thirty360.BondBasis)
    zero_dc = ql.Actual365Fixed()
    index_3m = ql.Euribor3M()
    index_6m = ql.Euribor6M()

    mm: list[Dict[str, object]] = []
    fra: list[Dict[str, object]] = []
    irs: list[Dict[str, object]] = []
    for ins in instruments:
        tpe = str(ins.get("instrument_type", "")).upper()
        key = str(ins.get("quote_key", "")).upper()
        if tpe == "MM":
            mm.append(ins)
        elif key.startswith("FRA/RATE/EUR/"):
            fra.append(ins)
        elif tpe == "IR_SWAP":
            irs.append(ins)

    def _quote(ins: Dict[str, object]) -> ql.QuoteHandle:
        return ql.QuoteHandle(ql.SimpleQuote(float(ins["quote_value"])))

    def _period(text: str) -> ql.Period:
        return ql.Period(str(text).upper())

    helpers: list[ql.RateHelper] = []
    discount_handle = ql.RelinkableYieldTermStructureHandle()

    has_ois = any("/1D/" in str(ins.get("quote_key", "")).upper() for ins in irs)
    if has_ois:
        overnight = ql.Eonia()
        # Only use the overnight deposit pillar to anchor the very front end.
        for ins in mm:
            key = str(ins.get("quote_key", "")).upper()
            parts = key.split("/")
            if len(parts) < 5:
                continue
            if parts[3] not in ("0D", "ON"):
                continue
            tenor = parts[-1]
            helpers.append(
                ql.DepositRateHelper(
                    _quote(ins),
                    _period(tenor),
                    0,
                    cal,
                    ql.Following,
                    False,
                    ql.Actual360(),
                )
            )
        for ins in sorted(irs, key=lambda x: float(x["maturity"])):
            key = str(ins.get("quote_key", "")).upper()
            if "/1D/" not in key:
                continue
            tenor = key.split("/")[-1]
            helpers.append(
                ql.OISRateHelper(
                    2,
                    _period(tenor),
                    _quote(ins),
                    overnight,
                )
            )
        curve = ql.PiecewiseLogLinearDiscount(ref, helpers, zero_dc)
    else:
        for ins in sorted(mm, key=lambda x: float(x["maturity"])):
            key = str(ins.get("quote_key", "")).upper()
            parts = key.split("/")
            if len(parts) < 5:
                continue
            tenor = parts[-1]
            fixing_days = int(parts[3][:-1]) if parts[3].endswith("D") and parts[3][:-1].isdigit() else 2
            helpers.append(
                ql.DepositRateHelper(
                    _quote(ins),
                    _period(tenor),
                    fixing_days,
                    cal,
                    ql.ModifiedFollowing,
                    False,
                    ql.Actual360(),
                )
            )
        for ins in sorted(fra, key=lambda x: float(x["maturity"])):
            key = str(ins.get("quote_key", "")).upper()
            parts = key.split("/")
            if len(parts) < 5:
                continue
            start_period = _period(parts[3])
            tenor = parts[4]
            if tenor == "3M":
                helpers.append(ql.FraRateHelper(_quote(ins), start_period, index_3m))
            elif tenor == "6M":
                helpers.append(ql.FraRateHelper(_quote(ins), start_period, index_6m))
        for ins in sorted(irs, key=lambda x: float(x["maturity"])):
            key = str(ins.get("quote_key", "")).upper()
            parts = key.split("/")
            if len(parts) < 6:
                continue
            float_tenor = parts[4]
            swap_tenor = parts[-1]
            if float_tenor == "3M":
                ibor = ql.Euribor3M(discount_handle)
            elif float_tenor == "6M":
                ibor = ql.Euribor6M(discount_handle)
            else:
                continue
            helpers.append(
                ql.SwapRateHelper(
                    _quote(ins),
                    _period(swap_tenor),
                    cal,
                    ql.Annual,
                    ql.ModifiedFollowing,
                    fixed_dc,
                    ibor,
                    ql.QuoteHandle(),
                    ql.Period(0, ql.Days),
                    discount_handle,
                )
            )
        curve = ql.PiecewiseLogLinearDiscount(ref, helpers, zero_dc)

    discount_handle.linkTo(curve)
    curve.enableExtrapolation()

    dates = list(curve.dates())
    times = [0.0]
    zeros = [0.0]
    seen = {0.0}
    for d in dates[1:]:
        t = float(curve.timeFromReference(d))
        if t <= 1.0e-12:
            continue
        if any(abs(t - s) < 1.0e-10 for s in seen):
            continue
        seen.add(t)
        times.append(t)
        zeros.append(float(curve.zeroRate(d, zero_dc, ql.Continuous).rate()))
    return times, zeros


def _load_ore_xva_aggregate(xva_csv: Path, cpty_or_netting: str) -> dict:
    """Read the aggregate XVA row (empty TradeId) for the given netting set.

    Returns dict with keys: cva, dva, fba, fca. Missing columns default to 0.0.
    """
    def _float(row: dict, key: str) -> float:
        val = row.get(key, "0") or "0"
        if val.strip() in ("", "#N/A"):
            return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0

    with open(xva_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tid_key = (
            "TradeId"
            if reader.fieldnames and "TradeId" in reader.fieldnames
            else "#TradeId"
        )
        for row in reader:
            if (
                row.get("NettingSetId", "") == cpty_or_netting
                and row.get(tid_key, "").strip() == ""
            ):
                return {
                    "cva": _float(row, "CVA"),
                    "dva": _float(row, "DVA"),
                    "fba": _float(row, "FBA"),
                    "fca": _float(row, "FCA"),
                }
    raise ValueError(
        f"Aggregate XVA row not found for netting set '{cpty_or_netting}' in {xva_csv}"
    )


def _load_ore_npv(npv_csv: Path, trade_id: str) -> float:
    """Read the t0 NPV for *trade_id* from ORE's npv.csv."""
    with open(npv_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tid_key = (
            "TradeId"
            if reader.fieldnames and "TradeId" in reader.fieldnames
            else "#TradeId"
        )
        for row in reader:
            if row.get(tid_key, "").strip() == trade_id:
                return float(row["NPV"])
    raise ValueError(f"Trade '{trade_id}' not found in {npv_csv}")
