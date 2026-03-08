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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .lgm import LGM1F, LGMParams
from .irs_xva_utils import (
    apply_parallel_float_spread_shift_to_match_npv,
    build_discount_curve_from_discount_pairs,
    calibrate_float_spreads_from_coupon,
    load_ore_default_curve_inputs,
    load_ore_discount_pairs_from_curves,
    load_ore_exposure_profile,
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

    # --- Trade / portfolio --------------------------------------------------
    trade_id: str
    counterparty: str
    netting_set_id: str
    legs: Dict                  # output of load_swap_legs_from_portfolio +
    #                              node_tenors injected + float spreads calibrated

    # --- Curve columns & raw data -------------------------------------------
    discount_column: str        # curves.csv column for discounting
    forward_column: str         # curves.csv column for forwarding (float index)
    curve_times_disc: np.ndarray
    curve_dfs_disc: np.ndarray
    curve_times_fwd: np.ndarray
    curve_dfs_fwd: np.ndarray

    # Callable discount-factor functions (built from the raw arrays above)
    p0_disc: object = dataclasses.field(repr=False)   # Callable[[float], float]
    p0_fwd: object = dataclasses.field(repr=False)    # Callable[[float], float]

    # --- ORE output: exposure profile ---------------------------------------
    exposure_times: np.ndarray
    exposure_dates: np.ndarray  # string dates matching exposure_times
    ore_epe: np.ndarray
    ore_ene: np.ndarray

    # --- ORE output: t0 NPV and CVA -----------------------------------------
    ore_t0_npv: float
    ore_cva: float

    # --- Credit inputs -------------------------------------------------------
    recovery: float
    hazard_times: np.ndarray
    hazard_rates: np.ndarray

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

    def __repr__(self) -> str:  # keep repr readable even with large arrays
        return (
            f"OreSnapshot("
            f"asof_date={self.asof_date!r}, "
            f"trade_id={self.trade_id!r}, "
            f"alpha_source={self.alpha_source!r}, "
            f"measure={self.measure!r}, "
            f"discount_column={self.discount_column!r}, "
            f"forward_column={self.forward_column!r}, "
            f"n_exposure_pts={self.exposure_times.size}, "
            f"ore_cva={self.ore_cva:,.2f}, "
            f"recovery={self.recovery:.2%})"
        )


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
    curve_times_disc, curve_dfs_disc = load_ore_discount_pairs_from_curves(
        str(curves_csv), discount_column=discount_column
    )
    p0_disc = build_discount_curve_from_discount_pairs(
        list(zip(curve_times_disc, curve_dfs_disc))
    )

    # Forward curve (may equal discount curve if columns are the same)
    if forward_column == discount_column:
        curve_times_fwd = curve_times_disc
        curve_dfs_fwd = curve_dfs_disc
        p0_fwd = p0_disc
    else:
        curve_times_fwd, curve_dfs_fwd = load_ore_discount_pairs_from_curves(
            str(curves_csv), discount_column=forward_column
        )
        p0_fwd = build_discount_curve_from_discount_pairs(
            list(zip(curve_times_fwd, curve_dfs_fwd))
        )

    # Exposure profile
    exposure = load_ore_exposure_profile(str(exposure_csv))
    exposure_times = exposure["time"]
    exposure_dates = exposure["date"]
    ore_epe = exposure["epe"]
    ore_ene = exposure["ene"]

    # ORE CVA (aggregate row for netting set)
    ore_cva = _load_ore_cva(xva_csv, cpty_or_netting=netting_set_id or cpty)

    # ORE t0 NPV
    ore_t0_npv = _load_ore_npv(npv_csv, trade_id=trade_id)

    # ------------------------------------------------------------------
    # 7. Build swap legs from portfolio (no flows.csv dependency)
    # ------------------------------------------------------------------
    legs = load_swap_legs_from_portfolio(
        str(portfolio_xml), trade_id=trade_id, asof_date=asof_date
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
    # 8. Load credit inputs (hazard + recovery)
    # ------------------------------------------------------------------
    if not market_data_file.exists():
        raise FileNotFoundError(f"market data file not found: {market_data_file}")

    credit = load_ore_default_curve_inputs(
        str(todaysmarket_xml), str(market_data_file), cpty_name=cpty
    )

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
        trade_id=trade_id,
        counterparty=cpty,
        netting_set_id=netting_set_id,
        legs=legs,
        discount_column=discount_column,
        forward_column=forward_column,
        curve_times_disc=curve_times_disc,
        curve_dfs_disc=curve_dfs_disc,
        curve_times_fwd=curve_times_fwd,
        curve_dfs_fwd=curve_dfs_fwd,
        p0_disc=p0_disc,
        p0_fwd=p0_fwd,
        exposure_times=exposure_times,
        exposure_dates=exposure_dates,
        ore_epe=ore_epe,
        ore_ene=ore_ene,
        ore_t0_npv=ore_t0_npv,
        ore_cva=ore_cva,
        recovery=float(credit["recovery"]),
        hazard_times=credit["hazard_times"],
        hazard_rates=credit["hazard_rates"],
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


def _load_ore_cva(xva_csv: Path, cpty_or_netting: str) -> float:
    """Read the aggregate CVA row (empty TradeId) for the given netting set."""
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
                return float(row["CVA"])
    raise ValueError(
        f"Aggregate CVA row not found for netting set '{cpty_or_netting}' in {xva_csv}"
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
