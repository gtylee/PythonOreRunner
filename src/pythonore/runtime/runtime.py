"""XVA runtime: orchestration layer for Python-native and ORE-SWIG XVA computation.

Architecture
------------
The module is organised in three layers:

* **Orchestration** – :class:`XVAEngine` is the top-level entry point.  It holds a
  :class:`RunnerAdapter` and creates :class:`XVASession` objects.  ``XVASession``
  owns the live computation state and exposes incremental-update helpers
  (``update_market``, ``update_portfolio``, ``update_config``).

* **Python-LGM path** – :class:`PythonLgmAdapter` implements the native Python LGM
  pricing pipeline.  It simulates LGM state-variable paths, prices IRS and
  FXForward trades, assembles XVA metrics, and optionally delegates unsupported
  trade types to the ORE SWIG fallback.

* **ORE SWIG fallback** – :class:`ORESwigAdapter` drives the ORE C++ engine through
  the Python SWIG bindings when they are available in the environment.

Input regimes
-------------
Two snapshot formats are supported:

1. **Full ORE-backed snapshot** – the snapshot was loaded from an ORE Input
   directory via ``XVALoader.from_files``.  ``snapshot.config.xml_buffers`` contains
   at minimum ``simulation.xml``, and the ORE output directory may hold
   ``curves.csv`` and ``calibration.xml`` artefacts that are consumed directly.

2. **Lightweight dataclass snapshot** – only ``snapshot.market.raw_quotes`` is
   populated.  Curves are bootstrapped on-the-fly from the market quotes without
   any ORE output artefacts.

Testing
-------
:class:`DeterministicToyAdapter` is the default in-memory adapter.  It applies
simple closed-form formulas to produce deterministic XVA numbers suitable for
unit-test assertions without requiring the full LGM or SWIG stack.
"""
from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import importlib
import math
import os
import re
from pathlib import Path
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from pythonore.domain.dataclasses import (
    FXForward,
    GenericProduct,
    InflationCapFloor,
    InflationSwap,
    IRS,
    MporConfig,
    Trade,
    XVASnapshot,
)
from pythonore.mapping.mapper import (
    MappedInputs,
    _default_index_for_ccy,
    _resolve_mpor_config,
    build_input_parameters,
    map_snapshot,
)
from pythonore.runtime.exposure_profiles import (
    build_ore_exposure_profile_from_paths,
    one_year_profile_value,
    ore_pfe_quantile,
)
from pythonore.runtime.dim import calculate_python_dim
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.results import CubeAccessor, XVAResult


class RunnerAdapter(Protocol):
    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult: ...


@dataclass
class SessionState:
    """Frozen snapshot of the input state tracked by an :class:`XVASession`.

    Holds the current :class:`XVASnapshot`, its stable hash key (used to detect
    redundant rebuilds), the pre-mapped ORE input lines, and per-component
    rebuild counters that record how many times each of market, portfolio, and
    config has been updated since the session was created.
    """

    snapshot: XVASnapshot
    snapshot_key: str
    mapped_inputs: MappedInputs
    rebuild_counts: Dict[str, int]


class XVAEngine:
    """Orchestration entry point for XVA computation.

    Holds a :class:`RunnerAdapter` that performs the actual pricing, and
    creates :class:`XVASession` objects that wrap a specific input state.
    Use :meth:`python_lgm_default` to obtain an engine backed by the native
    Python LGM stack, or pass a custom adapter to the constructor.
    """

    def __init__(self, adapter: Optional[RunnerAdapter] = None):
        self.adapter = adapter or DeterministicToyAdapter()

    @classmethod
    def python_lgm_default(cls, fallback_to_swig: bool = True) -> "XVAEngine":
        return cls(adapter=PythonLgmAdapter(fallback_to_swig=fallback_to_swig))

    def create_session(self, snapshot: XVASnapshot) -> "XVASession":
        mapped = map_snapshot(snapshot)
        return XVASession(
            engine=self,
            state=SessionState(
                snapshot=snapshot,
                snapshot_key=snapshot.stable_key(),
                mapped_inputs=mapped,
                rebuild_counts={"market": 1, "portfolio": 1, "config": 1},
            ),
        )

    def prepare_sensitivity_snapshot(
        self,
        snapshot: XVASnapshot,
        *,
        curve_node_shocks: Optional[Dict[str, object]] = None,
        curve_fit_mode: str = "ore_fit",
        use_ore_output_curves: bool = False,
        freeze_float_spreads: bool = False,
        frozen_float_spreads: Optional[Dict[str, List[float]]] = None,
    ) -> XVASnapshot:
        params = dict(snapshot.config.params)
        params["python.curve_fit_mode"] = str(curve_fit_mode)
        params["python.use_ore_output_curves"] = "Y" if use_ore_output_curves else "N"
        if curve_node_shocks is not None:
            params["python.curve_node_shocks"] = curve_node_shocks
        elif "python.curve_node_shocks" in params:
            params.pop("python.curve_node_shocks", None)
        if frozen_float_spreads is not None:
            params["python.frozen_float_spreads"] = frozen_float_spreads
        elif freeze_float_spreads and hasattr(self.adapter, "compute_frozen_float_spreads"):
            frozen = self.adapter.compute_frozen_float_spreads(
                replace(snapshot, config=replace(snapshot.config, params=params))
            )
            if frozen:
                params["python.frozen_float_spreads"] = frozen
        updated = replace(snapshot, config=replace(snapshot.config, params=params))
        if hasattr(self.adapter, "prepare_sensitivity_snapshot"):
            prepared = self.adapter.prepare_sensitivity_snapshot(
                updated,
                curve_node_shocks=curve_node_shocks,
                curve_fit_mode=curve_fit_mode,
                use_ore_output_curves=use_ore_output_curves,
                freeze_float_spreads=freeze_float_spreads,
                frozen_float_spreads=frozen_float_spreads,
            )
            if isinstance(prepared, XVASnapshot):
                return prepared
        return updated


class XVASession:
    """Live computation session wrapping a :class:`SessionState`.

    Supports incremental updates to market data, portfolio composition, and
    configuration through :meth:`update_market`, :meth:`update_portfolio`, and
    :meth:`update_config`.  Each update re-maps the snapshot only when the
    stable key has actually changed, avoiding redundant rebuilds.  Call
    :meth:`run` to execute the full XVA pipeline and obtain an
    :class:`XVAResult`.
    """

    def __init__(self, engine: XVAEngine, state: SessionState):
        self.engine = engine
        self.state = state

    def run(self, metrics: Optional[Sequence[str]] = None, return_cubes: bool = True) -> XVAResult:
        run_id = str(uuid.uuid4())
        if metrics:
            snapshot = replace(self.state.snapshot, config=replace(self.state.snapshot.config, analytics=tuple(metrics)))
            mapped = map_snapshot(snapshot)
        else:
            snapshot = self.state.snapshot
            mapped = self.state.mapped_inputs

        result = self.engine.adapter.run(snapshot=snapshot, mapped=mapped, run_id=run_id)
        result.metadata["rebuild_counts"] = dict(self.state.rebuild_counts)
        if not return_cubes:
            result.cubes = {}
        return result

    def run_incremental(self) -> XVAResult:
        return self.run()

    def update_market(self, market) -> None:
        updated = replace(self.state.snapshot, market=market)
        self._apply_snapshot(updated, changed="market")

    def update_config(self, config=None, **overrides) -> None:
        if config is not None and overrides:
            raise EngineRunError("Provide either config or keyword overrides, not both")
        if config is None:
            config = replace(self.state.snapshot.config, **overrides)
        updated = replace(self.state.snapshot, config=config)
        self._apply_snapshot(updated, changed="config")

    def update_portfolio(self, add: Iterable[Trade] = (), amend: Iterable[tuple[str, dict]] = (), remove: Iterable[str] = ()) -> None:
        trade_map = {t.trade_id: t for t in self.state.snapshot.portfolio.trades}

        for tid in remove:
            trade_map.pop(tid, None)
        for tid, updates in amend:
            if tid not in trade_map:
                raise EngineRunError(f"Cannot amend unknown trade {tid}")
            t = trade_map[tid]
            product_updates = dict(updates.get("product", {}))
            if isinstance(t.product, IRS):
                new_product = replace(t.product, **product_updates)
            elif isinstance(t.product, FXForward):
                new_product = replace(t.product, **product_updates)
            else:
                new_product = t.product
            trade_map[tid] = replace(t, product=new_product)
        for t in add:
            trade_map[t.trade_id] = t

        updated_pf = replace(self.state.snapshot.portfolio, trades=tuple(trade_map.values()))
        updated = replace(self.state.snapshot, portfolio=updated_pf)
        self._apply_snapshot(updated, changed="portfolio")

    def freeze(self) -> dict:
        return self.state.snapshot.to_dict()

    def _apply_snapshot(self, snapshot: XVASnapshot, changed: str) -> None:
        key = snapshot.stable_key()
        if key == self.state.snapshot_key:
            return
        mapped = map_snapshot(snapshot)
        self.state.snapshot = snapshot
        self.state.snapshot_key = key
        self.state.mapped_inputs = mapped
        self.state.rebuild_counts[changed] = self.state.rebuild_counts.get(changed, 0) + 1


class DeterministicToyAdapter:
    """In-memory deterministic adapter used for testable runtime behavior."""

    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        pv_total = 0.0
        epe_by_ns: Dict[str, float] = {}

        for t in snapshot.portfolio.trades:
            pv, epe = _toy_trade_numbers(t)
            pv_total += pv
            epe_by_ns[t.netting_set] = epe_by_ns.get(t.netting_set, 0.0) + epe

        total_epe = sum(epe_by_ns.values())
        metric_values: Dict[str, float] = {}
        for m in snapshot.config.analytics:
            if m == "CVA":
                metric_values[m] = 0.012 * total_epe
            elif m == "DVA":
                metric_values[m] = -0.006 * total_epe
            elif m == "FVA":
                metric_values[m] = 0.002 * abs(pv_total)
            elif m == "MVA":
                metric_values[m] = 0.0015 * total_epe

        xva_total = sum(metric_values.values())

        reports = {
            "xva": [
                {
                    "Metric": k,
                    "Value": v,
                }
                for k, v in metric_values.items()
            ],
            "exposure": [
                {"NettingSetId": ns, "EPE": v} for ns, v in epe_by_ns.items()
            ],
        }

        cubes = {
            "npv_cube": CubeAccessor(name="npv_cube", payload={"portfolio": {"t0": pv_total}}),
            "exposure_cube": CubeAccessor(name="exposure_cube", payload={ns: {"epe": v} for ns, v in epe_by_ns.items()}),
        }

        return XVAResult(
            run_id=run_id,
            pv_total=pv_total,
            xva_total=xva_total,
            xva_by_metric=metric_values,
            exposure_by_netting_set=epe_by_ns,
            exposure_profiles_by_netting_set={},
            exposure_profiles_by_trade={},
            reports=reports,
            cubes=cubes,
            metadata={
                "market_quotes": len(mapped.market_data_lines),
                "fixings": len(mapped.fixing_data_lines),
            },
        )


@dataclass
class _CurveBundle:
    """Curve artefacts produced by the market-curve building paths."""
    discount_curves: dict[str, Callable[[float], float]]
    forward_curves: dict[str, Callable[[float], float]]
    forward_curves_by_tenor: dict[str, dict[str, Callable[[float], float]]]
    forward_curves_by_name: dict[str, Callable[[float], float]]
    xva_discount_curve: Callable[[float], float] | None
    funding_borrow_curve: Callable[[float], float] | None
    funding_lend_curve: Callable[[float], float] | None


@dataclass(frozen=True)
class _TradeSpec:
    trade: Trade          # Original Trade dataclass instance.
    kind: str             # Pricing branch selector: "IRS" or "FXForward".
    notional: float       # Absolute notional in the trade currency.
    ccy: str              # ISO currency code for the trade (domestic leg for FX).
    legs: Dict[str, np.ndarray] | None = None  # Leg schedule arrays loaded from ORE flows output; None for FX trades.
    sticky_state: Dict[str, object] | None = None


@dataclass(frozen=True)
class _PythonLgmInputs:
    asof: str
    times: np.ndarray              # Full simulation grid (valuation grid plus sticky closeout grid).
    valuation_times: np.ndarray    # Augmented valuation grid (union of exposure grid and trade cash-flow dates).
    observation_times: np.ndarray  # Subset of times used as XVA observation points (from simulation.xml or fallback).
    observation_closeout_times: np.ndarray  # Sticky closeout times associated with observation_times.
    discount_curves: Dict[str, Callable[[float], float]]   # Risk-free discount curves keyed by ISO currency.
    forward_curves: Dict[str, Callable[[float], float]]    # Composite forward/index curves keyed by ISO currency.
    forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]]  # Forward curves keyed by (ccy, tenor string).
    forward_curves_by_name: Dict[str, Callable[[float], float]]  # Forward curves keyed by exact index name.
    swap_index_forward_tenors: Dict[str, str]  # CMS/swap index name -> underlying ibor tenor, from conventions.xml.
    inflation_curves: Dict[Tuple[str, str], Any]  # Inflation curves keyed by (index, curve_type) where curve_type is ZC or YY.
    xva_discount_curve: Optional[Callable[[float], float]]  # Simulation-market discount curve used for XVA deflation; falls back to discount_curves[base_ccy].
    funding_borrow_curve: Optional[Callable[[float], float]]  # Own-name borrowing curve used in FCA calculation; None when not configured.
    funding_lend_curve: Optional[Callable[[float], float]]    # Own-name lending curve used in FBA calculation; None when not configured.
    survival_curves: Dict[str, Callable[[float], float]]  # Optional ORE-style survival curves keyed by counterparty / own-name.
    hazard_times: Dict[str, np.ndarray]   # Piecewise-constant hazard rate node times keyed by counterparty / own-name.
    hazard_rates: Dict[str, np.ndarray]   # Piecewise-constant hazard rate values aligned with hazard_times.
    recovery_rates: Dict[str, float]
    lgm_params: Dict[str, object]         # Parsed LGM calibration parameters (alpha_times, alpha_values, kappa_times, kappa_values, shift, scaling).
    model_ccy: str
    seed: int
    fx_spots: Dict[str, float]
    fx_vols: Dict[str, List[Tuple[float, float]]]
    swaption_normal_vols: Dict[Tuple[str, str], List[Tuple[float, float]]]
    cms_correlations: Dict[Tuple[str, str], List[Tuple[float, float]]]
    stochastic_fx_pairs: Tuple[str, ...]
    trade_specs: Tuple[_TradeSpec, ...]
    unsupported: Tuple[Trade, ...]
    mpor: MporConfig
    input_provenance: Dict[str, str]  # Diagnostic dict recording which source was used for each input (model_params, market, grid, portfolio).
    input_fallbacks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _SharedFxSimulation:
    hybrid: Any
    sim: Dict[str, Any]
    pair_keys: Tuple[str, ...]


class PythonLgmAdapter:
    """Adapter that values supported trades using the Python LGM stack."""

    def __init__(self, fallback_to_swig: bool = True):
        self.fallback_to_swig = bool(fallback_to_swig)
        self._loaded = False
        self._lgm_mod = None
        self._irs_utils = None
        self._fx_utils = None
        self._inflation_mod = None
        self._ore_snapshot_mod = None
        self._asof_cache: Dict[int, str] = {}
        self._fixings_cache: Dict[int, Dict[tuple[str, str], float]] = {}
        self._time_date_cache: Dict[tuple[int, float], str] = {}
        self._portfolio_root_cache: Dict[int, ET.Element] = {}
        self._simulation_root_cache: Dict[int, ET.Element] = {}
        self._simulation_tenor_cache: Dict[int, np.ndarray] = {}
        self._generic_rate_swap_legs_cache: Dict[tuple[str, int], Optional[Dict[str, object]]] = {}
        self._irs_leg_cache: Dict[tuple[str, int], Dict[str, np.ndarray]] = {}

    def classify_portfolio_support(self, snapshot: XVASnapshot) -> Dict[str, Any]:
        """Classify a portfolio into Python-native and SWIG-only buckets."""
        self._ensure_py_lgm_imports()
        mapped = map_snapshot(snapshot)
        trade_specs, unsupported, _ = self._classify_portfolio_trades(snapshot, mapped)
        native_trade_ids = [spec.trade.trade_id for spec in trade_specs]
        native_trade_types = sorted({spec.trade.trade_type for spec in trade_specs})
        swig_trade_ids = [trade.trade_id for trade in unsupported]
        swig_trade_types = sorted({trade.trade_type for trade in unsupported})
        return {
            "mode": "hybrid" if self.fallback_to_swig else "native_only",
            "native_only": not self.fallback_to_swig,
            "python_supported": len(unsupported) == 0,
            "native_trade_ids": native_trade_ids,
            "native_trade_types": native_trade_types,
            "requires_swig_trade_ids": swig_trade_ids,
            "requires_swig_trade_types": swig_trade_types,
            "native_trade_count": len(native_trade_ids),
            "requires_swig_trade_count": len(swig_trade_ids),
        }

    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        """Execute the Python-LGM XVA computation pipeline.

        Simulates LGM state paths, prices each supported trade (IRS and
        FXForward), optionally falls back to the ORE SWIG adapter for
        unsupported trade types, then assembles and returns an
        :class:`XVAResult` with per-metric XVA figures and exposure cubes.
        """
        dim_mode = _active_dim_mode(snapshot)
        supported_python_dim_models = {
            "DynamicIM",
            "SimmAnalytic",
            "DeltaVaR",
            "DeltaGammaNormalVaR",
            "DeltaGammaVaR",
        }
        if dim_mode is not None and dim_mode not in supported_python_dim_models:
            raise EngineRunError(
                f"DIM mode '{dim_mode}' is not supported by the Python DIM port. "
                "Supported Python DIM models are DynamicIM, SimmAnalytic, DeltaVaR, DeltaGammaNormalVaR, DeltaGammaVaR."
            )
        if dim_mode in supported_python_dim_models and snapshot.config.params.get("python.dim_feeder") is None:
            raise EngineRunError(
                f"DIM mode '{dim_mode}' requires snapshot.config.params['python.dim_feeder'] for the Python DIM port."
            )
        self._ensure_py_lgm_imports()
        inputs = self._extract_inputs(snapshot, mapped)

        n_times = int(inputs.times.size)
        n_paths = int(snapshot.config.num_paths)
        npv_by_trade: Dict[str, np.ndarray] = {}
        fallback_trades: List[Trade] = []
        unsupported: List[Trade] = list(inputs.unsupported)
        preflight_support = self.classify_portfolio_support(snapshot)

        # Single-currency LGM model for IRS path simulation.
        model = self._lgm_mod.LGM1F(
            self._lgm_mod.LGMParams(
                alpha_times=tuple(float(x) for x in inputs.lgm_params["alpha_times"]),
                alpha_values=tuple(float(x) for x in inputs.lgm_params["alpha_values"]),
                kappa_times=tuple(float(x) for x in inputs.lgm_params["kappa_times"]),
                kappa_values=tuple(float(x) for x in inputs.lgm_params["kappa_values"]),
                shift=float(inputs.lgm_params["shift"]),
                scaling=float(inputs.lgm_params["scaling"]),
            )
        )
        rng_mode = self._python_lgm_rng_mode(snapshot)
        rng, draw_order = self._build_lgm_rng(inputs.seed, rng_mode)
        x_paths = self._lgm_mod.simulate_lgm_measure(
            model, inputs.times, n_paths=n_paths, rng=rng, x0=0.0, draw_order=draw_order
        )
        shared_fx_sim = self._build_shared_fx_simulation(snapshot, inputs, n_paths)

        for spec in inputs.trade_specs:
            if spec.kind == "IRS":
                p_disc = inputs.discount_curves[spec.ccy]
                legs = spec.legs or {}
                fwd_tenor = str(legs.get("float_index_tenor", "")).upper()
                p_fwd = inputs.forward_curves_by_tenor.get(spec.ccy, {}).get(
                    fwd_tenor,
                    inputs.forward_curves.get(spec.ccy, p_disc),
                )
                realized_coupon = self._compute_realized_float_coupons(
                    model=model,
                    p0_disc=p_disc,
                    p0_fwd=p_fwd,
                    legs=legs,
                    sim_times=inputs.times,
                    x_paths_on_sim_grid=x_paths,
                )
                vals = np.zeros((n_times, n_paths), dtype=float)
                for i, t in enumerate(inputs.times):
                    vals[i, :] = self._irs_utils.swap_npv_from_ore_legs_dual_curve(
                        model,
                        p_disc,
                        p_fwd,
                        legs,
                        float(t),
                        x_paths[i, :],
                        realized_float_coupon=realized_coupon,
                    )
                npv_by_trade[spec.trade.trade_id] = vals
            elif spec.kind == "RateSwap":
                vals = self._price_generic_rate_swap(spec, inputs, model, x_paths)
                npv_by_trade[spec.trade.trade_id] = vals
            elif spec.kind == "FXForward":
                vals = self._price_fx_forward(spec.trade, inputs, n_times, n_paths, shared_sim=shared_fx_sim)
                npv_by_trade[spec.trade.trade_id] = vals
            elif spec.kind == "InflationSwap":
                trade_product = spec.trade.product
                assert isinstance(trade_product, InflationSwap)
                curve_key = (
                    str(trade_product.index).upper(),
                    "YY" if str(trade_product.inflation_type).upper() == "YY" else "ZC",
                )
                inflation_curve = inputs.inflation_curves.get(curve_key)
                if inflation_curve is None or str(trade_product.pay_leg).lower() == "float":
                    unsupported.append(spec.trade)
                    continue
                p_disc = inputs.discount_curves[spec.ccy]
                vals = np.zeros((n_times, n_paths), dtype=float)
                if str(trade_product.inflation_type).upper() == "YY":
                    payment_times = self._inflation_mod.inflation_swap_payment_times(
                        float(trade_product.maturity_years),
                        str(trade_product.schedule_tenor or "1Y"),
                    )
                    profile = np.asarray(
                        [
                            self._inflation_mod.price_yoy_swap_at_time(
                                notional=float(trade_product.notional),
                                payment_times=payment_times,
                                fixed_rate=float(trade_product.fixed_rate),
                                inflation_curve=inflation_curve,
                                discount_curve=p_disc,
                                valuation_time=float(t),
                                receive_inflation=True,
                            )
                            for t in inputs.times
                        ],
                        dtype=float,
                    )
                else:
                    profile = np.asarray(
                        [
                            self._inflation_mod.price_zero_coupon_cpi_swap_at_time(
                                notional=float(trade_product.notional),
                                maturity_years=float(trade_product.maturity_years),
                                fixed_rate=float(trade_product.fixed_rate),
                                base_cpi=float(trade_product.base_cpi or 100.0),
                                inflation_curve=inflation_curve,
                                discount_curve=p_disc,
                                valuation_time=float(t),
                                receive_inflation=True,
                            )
                            for t in inputs.times
                        ],
                        dtype=float,
                    )
                vals[:, :] = profile[:, None]
                npv_by_trade[spec.trade.trade_id] = vals
            elif spec.kind == "InflationCapFloor":
                trade_product = spec.trade.product
                assert isinstance(trade_product, InflationCapFloor)
                curve_key = (
                    str(trade_product.index).upper(),
                    "YY" if str(trade_product.inflation_type).upper() == "YY" else "ZC",
                )
                inflation_curve = inputs.inflation_curves.get(curve_key)
                if inflation_curve is None:
                    unsupported.append(spec.trade)
                    continue
                p_disc = inputs.discount_curves[spec.ccy]
                definition = self._inflation_mod.InflationCapFloorDefinition(
                    trade_id=spec.trade.trade_id,
                    currency=spec.ccy,
                    inflation_type=str(trade_product.inflation_type),
                    option_type=str(trade_product.option_type),
                    index=str(trade_product.index),
                    strike=float(trade_product.strike),
                    notional=float(trade_product.notional),
                    maturity_years=float(trade_product.maturity_years),
                    base_cpi=float(trade_product.base_cpi) if trade_product.base_cpi is not None else None,
                    observation_lag=trade_product.observation_lag,
                    long_short=str(trade_product.long_short),
                )
                profile = np.asarray(
                    [
                        self._inflation_mod.price_inflation_capfloor_at_time(
                            definition=definition,
                            inflation_curve=inflation_curve,
                            discount_curve=p_disc,
                            valuation_time=float(t),
                        )
                        for t in inputs.times
                    ],
                    dtype=float,
                )
                vals = np.zeros((n_times, n_paths), dtype=float)
                vals[:, :] = profile[:, None]
                npv_by_trade[spec.trade.trade_id] = vals
            else:
                unsupported.append(spec.trade)

        fallback_result: XVAResult | None = None
        if unsupported:
            if not self.fallback_to_swig:
                bad = ", ".join(sorted({f"{t.trade_id}:{t.trade_type}" for t in unsupported}))
                raise EngineRunError(
                    "Unsupported by PythonLgmAdapter in native-only mode: "
                    f"{bad}. These trades are supported only through the ORE SWIG fallback."
                )
            try:
                swig_adapter = ORESwigAdapter()
            except Exception as exc:
                bad = ", ".join(sorted({f"{t.trade_id}:{t.trade_type}" for t in unsupported}))
                raise EngineRunError(
                    "Trades supported only through the ORE SWIG fallback were encountered: "
                    f"{bad}. SWIG adapter unavailable: {exc}"
                ) from exc
            fallback_trades = unsupported
            partial = replace(snapshot, portfolio=replace(snapshot.portfolio, trades=tuple(unsupported)))
            fallback_result = swig_adapter.run(partial, mapped=map_snapshot(partial), run_id=f"{run_id}-swig")

        result = self._assemble_result(
            run_id=run_id,
            snapshot=snapshot,
            inputs=inputs,
            model=model,
            x_paths=x_paths,
            npv_by_trade=npv_by_trade,
            fallback=fallback_result,
            fallback_trades=fallback_trades,
            unsupported=unsupported if fallback_result is None else [],
        )
        result.metadata["python_lgm_rng_mode"] = rng_mode
        if dim_mode in supported_python_dim_models:
            dim_result = calculate_python_dim(snapshot.config.params, dim_model=dim_mode)
            result.reports.update(dim_result.reports)
            result.cubes.update(dim_result.cubes)
            result.metadata["dim_mode"] = dim_mode
            result.metadata["dim_current"] = dict(dim_result.current_dim)
            result.metadata["dim_engine"] = dim_result.metadata.get("engine", "python-dim")
        result.metadata["support_classification"] = preflight_support
        return result

    def prepare_sensitivity_snapshot(
        self,
        snapshot: XVASnapshot,
        *,
        curve_node_shocks: Optional[Dict[str, object]] = None,
        curve_fit_mode: str = "ore_fit",
        use_ore_output_curves: bool = False,
        freeze_float_spreads: bool = False,
        frozen_float_spreads: Optional[Dict[str, List[float]]] = None,
    ) -> XVASnapshot:
        params = dict(snapshot.config.params)
        params["python.curve_fit_mode"] = str(curve_fit_mode)
        params["python.use_ore_output_curves"] = "Y" if use_ore_output_curves else "N"
        if curve_node_shocks is not None:
            params["python.curve_node_shocks"] = curve_node_shocks
        else:
            params.pop("python.curve_node_shocks", None)
        if frozen_float_spreads is not None:
            params["python.frozen_float_spreads"] = frozen_float_spreads
        elif freeze_float_spreads:
            frozen = self.compute_frozen_float_spreads(replace(snapshot, config=replace(snapshot.config, params=params)))
            if frozen:
                params["python.frozen_float_spreads"] = frozen
        return replace(snapshot, config=replace(snapshot.config, params=params))

    def compute_frozen_float_spreads(self, snapshot: XVASnapshot) -> Dict[str, List[float]]:
        self._ensure_py_lgm_imports()
        try:
            inputs = self._extract_inputs(snapshot, map_snapshot(snapshot))
        except Exception:
            return {}
        out: Dict[str, List[float]] = {}
        for spec in inputs.trade_specs:
            if spec.kind != "IRS" or spec.legs is None:
                continue
            spread = np.asarray(spec.legs.get("float_spread", []), dtype=float)
            coupon = np.asarray(spec.legs.get("float_coupon", []), dtype=float)
            if spread.size and coupon.size == spread.size:
                out[spec.trade.trade_id] = [float(x) for x in spread]
        return out

    def _ensure_py_lgm_imports(self) -> None:
        """Lazily load the py_ore_tools modules required by the LGM pipeline.

        Tries the installed ``py_ore_tools`` package first, then falls back to
        importing directly from the relative path within the Engine repository
        (``legacy/py_ore_tools``).  Raises :class:`EngineRunError`
        if neither path succeeds.
        """
        if self._loaded:
            return
        try:
            from pythonore.compute import inflation, irs_xva_utils, lgm, lgm_fx_xva_utils
            from pythonore.io import ore_snapshot

            self._inflation_mod = inflation
            self._lgm_mod = lgm
            self._irs_utils = irs_xva_utils
            self._fx_utils = lgm_fx_xva_utils
            self._ore_snapshot_mod = ore_snapshot
            self._loaded = True
            return
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[3]
        tools_dir = repo_root / "legacy" / "py_ore_tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

        try:
            import irs_xva_utils as irs_xva_utils_local
            import inflation as inflation_local
            import lgm as lgm_local
            import lgm_fx_xva_utils as lgm_fx_xva_utils_local
            import ore_snapshot as ore_snapshot_local

            self._inflation_mod = inflation_local
            self._lgm_mod = lgm_local
            self._irs_utils = irs_xva_utils_local
            self._fx_utils = lgm_fx_xva_utils_local
            self._ore_snapshot_mod = ore_snapshot_local
            self._loaded = True
        except Exception as exc:
            raise EngineRunError(f"Failed to import Python LGM toolchain: {exc}") from exc

    def _parse_model_params(
        self, xml: Dict[str, str], model_ccy: str, snapshot: "XVASnapshot"
    ) -> tuple[Dict[str, object], str]:
        """Return (lgm_params, param_source)."""
        param_source = "simulation"
        use_ore_output_lgm = str(snapshot.config.params.get("python.use_ore_output_lgm_params", "N")).strip().upper() not in ("N", "FALSE", "0", "")
        if use_ore_output_lgm and "calibration.xml" in xml:
            lgm_params = _parse_lgm_params_from_calibration_xml_text(xml["calibration.xml"], ccy_key=model_ccy)
            param_source = "calibration"
        elif self._is_ore_case_snapshot(snapshot):
            ore_path_txt = getattr(snapshot.config.source_meta, "path", "") or ""
            try:
                ore_path = Path(ore_path_txt).resolve()
                ore_root = ET.parse(ore_path).getroot()
                setup = {
                    n.attrib.get("name", ""): (n.text or "").strip()
                    for n in ore_root.findall("./Setup/Parameter")
                }
                sim_analytic = ore_root.find("./Analytics/Analytic[@type='simulation']")
                sim_cfg = "simulation.xml"
                if sim_analytic is not None:
                    sim_cfg = (
                        sim_analytic.findtext("./Parameter[@name='simulationConfigFile']")
                        or sim_cfg
                    ).strip()
                _, _, input_dir = self._ore_snapshot_mod._resolve_case_dirs(ore_path)
                simulation_xml_path = self._ore_snapshot_mod._resolve_ore_path(sim_cfg, input_dir)
                lgm_params = self._ore_snapshot_mod.calibrate_lgm_params_via_ore(
                    ore_xml_path=ore_path,
                    input_dir=input_dir,
                    simulation_xml_path=simulation_xml_path,
                    ccy_key=model_ccy,
                )
                if lgm_params is not None:
                    param_source = "calibration"
                elif "simulation.xml" in xml:
                    lgm_params = _parse_lgm_params_from_simulation_xml_text(xml["simulation.xml"], ccy_key=model_ccy)
                elif simulation_xml_path.exists():
                    lgm_params = self._ore_snapshot_mod.parse_lgm_params_from_simulation_xml(
                        str(simulation_xml_path),
                        ccy_key=model_ccy,
                    )
                else:
                    lgm_params = {
                        "alpha_times": np.array([], dtype=float),
                        "alpha_values": np.array([0.01], dtype=float),
                        "kappa_times": np.array([], dtype=float),
                        "kappa_values": np.array([0.03], dtype=float),
                        "shift": 0.0,
                        "scaling": 1.0,
                    }
            except Exception:
                if "simulation.xml" in xml:
                    lgm_params = _parse_lgm_params_from_simulation_xml_text(xml["simulation.xml"], ccy_key=model_ccy)
                else:
                    lgm_params = {
                        "alpha_times": np.array([], dtype=float),
                        "alpha_values": np.array([0.01], dtype=float),
                        "kappa_times": np.array([], dtype=float),
                        "kappa_values": np.array([0.03], dtype=float),
                        "shift": 0.0,
                        "scaling": 1.0,
                    }
        elif "simulation.xml" in xml:
            lgm_params = _parse_lgm_params_from_simulation_xml_text(xml["simulation.xml"], ccy_key=model_ccy)
        else:
            lgm_params = {
                "alpha_times": np.array([], dtype=float),
                "alpha_values": np.array([0.01], dtype=float),
                "kappa_times": np.array([], dtype=float),
                "kappa_values": np.array([0.03], dtype=float),
                "shift": 0.0,
                "scaling": 1.0,
            }
        if np.asarray(lgm_params["alpha_values"], dtype=float).size == 0:
            lgm_params["alpha_values"] = np.array([0.01], dtype=float)
        if np.asarray(lgm_params["kappa_values"], dtype=float).size == 0:
            lgm_params["kappa_values"] = np.array([0.03], dtype=float)
        return lgm_params, param_source

    def _build_exposure_grid(
        self, snapshot: "XVASnapshot", xml: Dict[str, str]
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Return (times, observation_times, grid_source)."""
        grid_source = "fallback"
        if "simulation.xml" in xml:
            times = _parse_exposure_times_from_simulation_xml_text(xml["simulation.xml"])
            if times.size > 1:
                grid_source = "xml"
            else:
                times = _fallback_exposure_grid(snapshot)
        else:
            times = _fallback_exposure_grid(snapshot)
        if times.size < 2:
            times = np.array([0.0, max(float(snapshot.config.horizon_years), 1.0)], dtype=float)
        observation_times = np.asarray(times, dtype=float)
        return times, observation_times, grid_source

    def _classify_portfolio_trades(
        self, snapshot: "XVASnapshot", mapped: "MappedInputs"
    ) -> tuple[list["_TradeSpec"], list["Trade"], set[str]]:
        """Return (trade_specs, unsupported, ccy_set)."""
        trade_specs: List[_TradeSpec] = []
        unsupported: List[Trade] = []
        generic_ccys: set[str] = set()
        for t in snapshot.portfolio.trades:
            if isinstance(t.product, IRS):
                trade_specs.append(
                    _TradeSpec(
                        trade=t,
                        kind="IRS",
                        notional=float(t.product.notional),
                        ccy=t.product.ccy.upper(),
                        legs=self._build_irs_legs(t, mapped, snapshot),
                    )
                )
            elif isinstance(t.product, FXForward):
                trade_specs.append(
                    _TradeSpec(
                        trade=t,
                        kind="FXForward",
                        notional=float(t.product.notional),
                        ccy=t.product.pair[3:].upper(),
                    )
                )
            elif isinstance(t.product, InflationSwap):
                trade_specs.append(
                    _TradeSpec(
                        trade=t,
                        kind="InflationSwap",
                        notional=float(t.product.notional),
                        ccy=t.product.ccy.upper(),
                    )
                )
            elif isinstance(t.product, InflationCapFloor):
                trade_specs.append(
                    _TradeSpec(
                        trade=t,
                        kind="InflationCapFloor",
                        notional=float(t.product.notional),
                        ccy=t.product.ccy.upper(),
                    )
                )
            elif isinstance(t.product, GenericProduct):
                generic_legs = self._build_generic_rate_swap_legs(t, snapshot)
                if generic_legs is not None:
                    notionals = [
                        abs(float(leg.get("notional", 0.0)))
                        for leg in generic_legs.get("rate_legs", [])
                    ]
                    generic_ccys.add(str(generic_legs.get("ccy", snapshot.config.base_currency)).upper())
                    trade_specs.append(
                        _TradeSpec(
                            trade=t,
                            kind="RateSwap",
                            notional=max(notionals) if notionals else 0.0,
                            ccy=str(generic_legs.get("ccy", snapshot.config.base_currency)).upper(),
                            legs=generic_legs,
                        )
                    )
                else:
                    unsupported.append(t)
            else:
                unsupported.append(t)
        ccy_set: set[str] = {snapshot.config.base_currency.upper()}
        for t in snapshot.portfolio.trades:
            if isinstance(t.product, IRS):
                ccy_set.add(t.product.ccy.upper())
            if isinstance(t.product, FXForward):
                ccy_set.add(t.product.pair[:3].upper())
                ccy_set.add(t.product.pair[3:].upper())
            if isinstance(t.product, InflationSwap):
                ccy_set.add(t.product.ccy.upper())
            if isinstance(t.product, InflationCapFloor):
                ccy_set.add(t.product.ccy.upper())
        ccy_set.update(generic_ccys)
        return trade_specs, unsupported, ccy_set

    def _normalized_asof(self, snapshot: XVASnapshot) -> str:
        key = id(snapshot)
        cached = self._asof_cache.get(key)
        if cached is None:
            cached = _normalize_asof_date(snapshot.config.asof)
            self._asof_cache[key] = cached
        return cached

    def _fixings_lookup(self, snapshot: XVASnapshot) -> Dict[tuple[str, str], float]:
        key = id(snapshot)
        cached = self._fixings_cache.get(key)
        if cached is None:
            cached = {
                (str(p.index).upper(), _normalize_asof_date(p.date)): float(p.value)
                for p in snapshot.fixings.points
            }
            self._fixings_cache[key] = cached
        return cached

    def _date_from_time_cached(self, snapshot: XVASnapshot, t: float) -> str:
        cache_key = (id(snapshot), float(t))
        cached = self._time_date_cache.get(cache_key)
        if cached is None:
            cached = _date_from_time(self._normalized_asof(snapshot), float(t))
            self._time_date_cache[cache_key] = cached
        return cached

    def _portfolio_root_from_xml(self, portfolio_xml: str) -> ET.Element:
        key = id(portfolio_xml)
        root = self._portfolio_root_cache.get(key)
        if root is None:
            root = ET.fromstring(portfolio_xml)
            self._portfolio_root_cache[key] = root
        return root

    def _simulation_root_from_xml(self, simulation_xml: str) -> ET.Element:
        key = id(simulation_xml)
        root = self._simulation_root_cache.get(key)
        if root is None:
            root = ET.fromstring(simulation_xml)
            self._simulation_root_cache[key] = root
        return root

    def _simulation_node_tenors_from_xml(self, simulation_xml: str) -> np.ndarray:
        key = id(simulation_xml)
        tenors = self._simulation_tenor_cache.get(key)
        if tenors is None:
            tenors = self._irs_utils.load_simulation_yield_tenors_from_root(
                self._simulation_root_from_xml(simulation_xml)
            )
            self._simulation_tenor_cache[key] = tenors
        return np.asarray(tenors, dtype=float)

    def _build_hazard_rates(
        self, snapshot: "XVASnapshot", hazards: Dict, recoveries: Dict
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
        """Return (hazard_times, hazard_rates, recovery_rates)."""
        hazard_times: Dict[str, np.ndarray] = {}
        hazard_rates: Dict[str, np.ndarray] = {}
        recovery_rates: Dict[str, float] = {}
        names = {t.counterparty.upper() for t in snapshot.portfolio.trades}
        own_name = _own_name_from_runtime(snapshot).upper()
        names.add(own_name)
        if snapshot.config.runtime is not None:
            names.update(str(x).upper() for x in snapshot.config.runtime.counterparties.ids)

        for ns in names:
            hz = hazards.get(ns)
            if hz is None and ns == own_name:
                # Own-name curve is sometimes not explicitly quoted in compact
                # stress datasets; use a conservative own-intensity fallback.
                hz = [(1.0, 0.017), (5.0, 0.017)]
            if hz is None:
                hz = [(1.0, 0.02), (5.0, 0.02)]
            hz = sorted((float(t), float(v)) for t, v in hz)
            hazard_times[ns] = np.asarray([x[0] for x in hz], dtype=float)
            hazard_rates[ns] = np.asarray([x[1] for x in hz], dtype=float)
            recovery_rates[ns] = float(recoveries.get(ns, 0.4))
        return hazard_times, hazard_rates, recovery_rates

    def _build_survival_curves(
        self,
        snapshot: "XVASnapshot",
        hazard_times: Dict[str, np.ndarray],
        hazard_rates: Dict[str, np.ndarray],
    ) -> Dict[str, Callable[[float], float]]:
        """Return survival curves, preferring explicit stressed survival nodes if present."""
        explicit = snapshot.config.params.get("python.credit_survival_curves", {})
        curves: Dict[str, Callable[[float], float]] = {}
        if isinstance(explicit, dict):
            for name, cfg in explicit.items():
                if not isinstance(cfg, dict):
                    continue
                node_times = np.asarray(cfg.get("node_times", []), dtype=float)
                node_surv = np.asarray(cfg.get("survival_probabilities", []), dtype=float)
                if node_times.size == 0 or node_surv.size == 0 or node_times.size != node_surv.size:
                    continue
                extrapolation = str(cfg.get("extrapolation", "flat_zero")).strip().lower()
                curves[str(name).upper()] = self._irs_utils.build_survival_probability_curve_from_nodes(
                    node_times,
                    node_surv,
                    extrapolation=extrapolation,
                )
        for name, times in hazard_times.items():
            uname = str(name).upper()
            if uname in curves:
                continue
            rates = hazard_rates.get(name)
            if rates is None:
                continue
            curves[uname] = lambda t, ht=times, hr=rates: float(
                self._irs_utils.survival_probability_from_hazard(
                    np.asarray([float(t)], dtype=float),
                    np.asarray(ht, dtype=float),
                    np.asarray(hr, dtype=float),
                )[0]
            )
        return curves

    def _parse_swap_index_forward_tenors(self, conventions_xml: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not conventions_xml.strip():
            return mapping
        try:
            root = ET.fromstring(conventions_xml)
        except Exception:
            return mapping
        swap_conv_to_index: Dict[str, str] = {}
        for node in root.findall("./Swap"):
            conv_id = (node.findtext("./Id") or "").strip()
            ibor_index = (node.findtext("./Index") or "").strip().upper()
            tenor_match = re.search(r"(\d+[YMWD])$", ibor_index)
            if conv_id and tenor_match is not None:
                swap_conv_to_index[conv_id] = tenor_match.group(1).upper()
        for node in root.findall("./SwapIndex"):
            index_id = (node.findtext("./Id") or "").strip().upper()
            conv_id = (node.findtext("./Conventions") or "").strip()
            tenor = swap_conv_to_index.get(conv_id, "")
            if index_id and tenor:
                mapping[index_id] = tenor
        return mapping

    def _load_inflation_curves(
        self,
        snapshot: XVASnapshot,
        trade_specs: Sequence[_TradeSpec],
    ) -> Dict[Tuple[str, str], Any]:
        needed = {
            (
                str(spec.trade.product.index).upper(),
                "YY" if str(spec.trade.product.inflation_type).upper() == "YY" else "ZC",
            )
            for spec in trade_specs
            if spec.kind in {"InflationSwap", "InflationCapFloor"}
            and isinstance(spec.trade.product, (InflationSwap, InflationCapFloor))
        }
        if not needed:
            return {}
        market_file: Path | None = None
        ore_path_txt = getattr(snapshot.config.source_meta, "path", "") or ""
        if ore_path_txt:
            try:
                ore_path = Path(ore_path_txt).resolve()
                _, _, market_data_file, _, _ = self._ore_snapshot_mod._resolve_ore_run_files(ore_path)
                market_file = Path(market_data_file)
            except Exception:
                market_file = None
        curves: Dict[Tuple[str, str], Any] = {}
        if market_file is not None and market_file.exists():
            for index_name, curve_type in sorted(needed):
                curves[(index_name, curve_type)] = self._inflation_mod.load_inflation_curve_from_market_data(
                    market_file,
                    snapshot.config.asof,
                    index_name,
                    curve_type=curve_type,
                )
        return curves

    def _extract_inputs(self, snapshot: XVASnapshot, mapped: MappedInputs) -> _PythonLgmInputs:
        """Resolve all model inputs from the snapshot and mapped inputs.

        Handles both the ORE-backed input regime (simulation.xml present, ORE
        output artefacts optionally available) and the lightweight-dataclass
        regime (raw_quotes only).  Classifies portfolio trades, builds or loads
        discount/forward curves, parses LGM calibration parameters, assembles
        the hazard-rate term structures, and returns a fully-populated
        :class:`_PythonLgmInputs` ready for simulation.
        """
        xml = snapshot.config.xml_buffers
        model_ccy = snapshot.config.base_currency.upper()

        # There are two supported input regimes here:
        # 1. An ORE-backed snapshot with simulation.xml / todaysmarket.xml and,
        #    ideally, output artifacts like curves.csv and calibration.xml.
        # 2. A lightweight dataclass snapshot with only raw quotes, where we
        #    intentionally fall back to a simplified market-overlay build.
        #
        # The fragile case is "almost ORE-backed": if an ore.xml case is supplied
        # but the key XML buffers are missing, silent fallback gives plausible but
        # misleading numbers. Treat that as an input incompatibility instead.
        if self._is_ore_case_snapshot(snapshot) and "simulation.xml" not in xml:
            raise EngineRunError(
                "PythonLgmAdapter requires 'simulation.xml' in snapshot.config.xml_buffers "
                "for ORE-backed snapshots. Fix: load the snapshot with XVALoader.from_files(...) "
                "from the full ORE Input directory, or include simulation.xml explicitly in xml_buffers."
            )

        lgm_params, param_source = self._parse_model_params(xml, model_ccy, snapshot)
        valuation_times, observation_times, grid_source = self._build_exposure_grid(snapshot, xml)
        mpor = _resolve_mpor_config(snapshot, xml)

        overlay = _parse_market_overlay(snapshot.market.raw_quotes)
        fx_spots = overlay["fx"]
        fx_vols = overlay.get("fx_vol", {})
        swaption_normal_vols = overlay.get("swaption_normal_vols", {})
        cms_correlations = overlay.get("cms_correlations", {})
        swap_index_forward_tenors = self._parse_swap_index_forward_tenors(xml.get("conventions.xml", ""))
        zero_curves = overlay["zero"]
        named_zero_curves = overlay.get("named_zero", {})
        fwd_curves_raw = overlay.get("fwd", {})
        hazards = overlay["hazard"]
        recoveries = overlay["recovery"]

        trade_specs, unsupported, ccy_set = self._classify_portfolio_trades(snapshot, mapped)
        inflation_curves = self._load_inflation_curves(snapshot, trade_specs)
        stochastic_fx_pairs = _parse_stochastic_fx_pairs_from_simulation_xml_text(
            xml.get("simulation.xml", ""),
            model_ccy=model_ccy,
            trade_specs=trade_specs,
        )
        needed_tenors: Dict[str, set[str]] = {}
        for spec in trade_specs:
            if spec.kind == "IRS" and spec.legs is not None:
                tenor = str(spec.legs.get("float_index_tenor", "")).upper()
                if tenor:
                    needed_tenors.setdefault(spec.ccy.upper(), set()).add(tenor)
            elif spec.kind == "RateSwap" and isinstance(spec.legs, dict):
                for leg in spec.legs.get("rate_legs", []):
                    for field in ("index_name", "index_name_1", "index_name_2"):
                        index_name = str(leg.get(field, "")).upper()
                        tenor = swap_index_forward_tenors.get(index_name, "")
                        if tenor:
                            needed_tenors.setdefault(spec.ccy.upper(), set()).add(tenor)
        input_fallbacks: List[str] = []
        strict_ore_inputs = self._is_ore_case_snapshot(snapshot)

        for spec in trade_specs:
            if spec.kind != "FXForward":
                continue
            pair6 = str(spec.trade.product.pair).upper()
            if pair6 not in fx_spots and pair6[3:] + pair6[:3] not in fx_spots:
                input_fallbacks.append(f"missing_fx_spot:{pair6}")
            if pair6 not in fx_vols and pair6[3:] + pair6[:3] not in fx_vols:
                input_fallbacks.append(f"missing_fx_vol:{pair6}")

        for c in list(ccy_set):
            if c not in zero_curves:
                input_fallbacks.append(f"missing_zero_curve:{c}")
                zero_curves.setdefault(c, [(0.0, 0.02), (max(float(snapshot.config.horizon_years), 1.0), 0.02)])

        curve_payload = self._load_ore_output_curves(snapshot, mapped, trade_specs)
        if curve_payload is not None:
            discount_curves = curve_payload.discount_curves
            forward_curves = curve_payload.forward_curves
            forward_curves_by_tenor = curve_payload.forward_curves_by_tenor
            forward_curves_by_name = curve_payload.forward_curves_by_name
            xva_discount_curve = curve_payload.xva_discount_curve
            funding_borrow_curve = curve_payload.funding_borrow_curve
            funding_lend_curve = curve_payload.funding_lend_curve
            curve_source = "ore_output_curves"
        else:
            discount_curves = {}
            forward_curves = {}
            forward_curves_by_tenor = {}
            forward_curves_by_name = {}
            xva_discount_curve = None
            funding_borrow_curve = None
            funding_lend_curve = None
            fitted_curve_mode = str(snapshot.config.params.get("python.curve_fit_mode", "")).strip().lower()
            fitted_curves = None
            if fitted_curve_mode == "ore_fit":
                fitted_curves = self._fit_curves_from_market_quotes(snapshot, trade_specs)
            if fitted_curves is not None:
                discount_curves = fitted_curves.discount_curves
                forward_curves = fitted_curves.forward_curves
                forward_curves_by_tenor = fitted_curves.forward_curves_by_tenor
                forward_curves_by_name = fitted_curves.forward_curves_by_name
                xva_discount_curve = fitted_curves.xva_discount_curve
                funding_borrow_curve = fitted_curves.funding_borrow_curve
                funding_lend_curve = fitted_curves.funding_lend_curve
                curve_source = "ore_quote_fit"
            else:
                ore_root = ET.fromstring(xml["ore.xml"]) if xml.get("ore.xml", "").strip() else None
                tm_root = ET.fromstring(xml["todaysmarket.xml"]) if xml.get("todaysmarket.xml", "").strip() else None
                pricing_config_id = (
                    (ore_root.findtext("./Markets/Parameter[@name='pricing']") if ore_root is not None else None)
                    or "default"
                ).strip() or "default"
                discount_source_columns: Dict[str, str] = {}
                if tm_root is not None:
                    for ccy in sorted(ccy_set):
                        try:
                            discount_source_columns[ccy] = self._ore_snapshot_mod._resolve_discount_column(
                                tm_root,
                                pricing_config_id,
                                ccy,
                            )
                        except Exception:
                            discount_source_columns[ccy] = ""
                for ccy, pts in zero_curves.items():
                    preferred_discount_family = _curve_family_from_source_column(
                        discount_source_columns.get(ccy, "")
                    )
                    if not preferred_discount_family:
                        tenors = needed_tenors.get(ccy, set())
                        if len(tenors) == 1:
                            preferred_discount_family = sorted(tenors)[0]
                    discount_pts = pts
                    if preferred_discount_family and preferred_discount_family not in ("", "1D", "ON", "O/N"):
                        discount_pts = fwd_curves_raw.get(ccy, {}).get(preferred_discount_family, pts)
                    by_time: Dict[float, List[float]] = {}
                    for t, r in discount_pts:
                        by_time.setdefault(float(t), []).append(float(r))
                    dedup: Dict[float, float] = {}
                    for t, vals in by_time.items():
                        dedup[t] = min(vals, key=lambda x: abs(x))
                    sorted_pts = sorted(dedup.items(), key=lambda x: x[0])
                    if len(sorted_pts) == 1:
                        sorted_pts = [(0.0, sorted_pts[0][1]), (max(float(snapshot.config.horizon_years), 1.0), sorted_pts[0][1])]
                    discount_curves[ccy] = self._irs_utils.build_discount_curve_from_zero_rate_pairs(sorted_pts)
                    fwd_pts = []
                    buckets: Mapping[str, List[Tuple[float, float]]] = fwd_curves_raw.get(ccy, {})
                    tenor_curves: Dict[str, Callable[[float], float]] = {}
                    for tenor_key, bucket_pts in buckets.items():
                        tenor_by_t: Dict[float, List[float]] = {}
                        for t, r in bucket_pts:
                            tenor_by_t.setdefault(float(t), []).append(float(r))
                        if tenor_by_t:
                            pts_tenor = sorted(
                                ((t, min(vals, key=lambda x: abs(x))) for t, vals in tenor_by_t.items()),
                                key=lambda x: x[0],
                            )
                            if len(pts_tenor) == 1:
                                pts_tenor = [(0.0, pts_tenor[0][1]), (max(float(snapshot.config.horizon_years), 1.0), pts_tenor[0][1])]
                            tenor_curves[tenor_key.upper()] = self._irs_utils.build_discount_curve_from_zero_rate_pairs(pts_tenor)
                    if buckets:
                        merged_by_t: Dict[float, List[float]] = {}
                        for bucket_pts in buckets.values():
                            for t, r in bucket_pts:
                                merged_by_t.setdefault(float(t), []).append(float(r))
                        if merged_by_t:
                            fwd_pts = [(t, min(vals, key=lambda x: abs(x))) for t, vals in merged_by_t.items()]
                    if fwd_pts:
                        fwd_by_t: Dict[float, float] = {}
                        for t, r in fwd_pts:
                            fwd_by_t[float(t)] = float(r)
                        sp = sorted(fwd_by_t.items(), key=lambda x: x[0])
                        if len(sp) == 1:
                            sp = [(0.0, sp[0][1]), (max(float(snapshot.config.horizon_years), 1.0), sp[0][1])]
                        forward_curves[ccy] = self._irs_utils.build_discount_curve_from_zero_rate_pairs(sp)
                    else:
                        forward_curves[ccy] = discount_curves[ccy]
                    forward_curves_by_tenor[ccy] = tenor_curves
                for index_name, tenor in swap_index_forward_tenors.items():
                    index_ccy = index_name.split("-", 1)[0].upper()
                    curve = forward_curves_by_tenor.get(index_ccy, {}).get(str(tenor).upper())
                    if curve is not None:
                        forward_curves_by_name.setdefault(index_name.upper(), curve)
                curve_source = "market_overlay"

        runtime = snapshot.config.runtime
        xva_cfg = runtime.xva_analytic if runtime is not None else None
        borrow_curve_name = (
            str(xva_cfg.fva_borrowing_curve)
            if xva_cfg is not None and xva_cfg.fva_borrowing_curve
            else str(snapshot.config.params.get("xva.fvaBorrowingCurve", "")).strip()
        )
        lend_curve_name = (
            str(xva_cfg.fva_lending_curve)
            if xva_cfg is not None and xva_cfg.fva_lending_curve
            else str(snapshot.config.params.get("xva.fvaLendingCurve", "")).strip()
        )
        if funding_borrow_curve is None and borrow_curve_name and borrow_curve_name in named_zero_curves:
            funding_borrow_curve = self._irs_utils.build_discount_curve_from_zero_rate_pairs(
                sorted(named_zero_curves[borrow_curve_name], key=lambda x: x[0])
            )
        if funding_lend_curve is None and lend_curve_name and lend_curve_name in named_zero_curves:
            funding_lend_curve = self._irs_utils.build_discount_curve_from_zero_rate_pairs(
                sorted(named_zero_curves[lend_curve_name], key=lambda x: x[0])
            )

        (
            discount_curves,
            forward_curves,
            forward_curves_by_tenor,
            forward_curves_by_name,
            xva_discount_curve,
        ) = _apply_curve_node_shocks(
            snapshot,
            discount_curves,
            forward_curves,
            forward_curves_by_tenor,
            forward_curves_by_name,
            xva_discount_curve,
        )

        calibrated_specs: List[_TradeSpec] = []
        frozen_spreads = snapshot.config.params.get("python.frozen_float_spreads", {})
        for spec in trade_specs:
            if spec.kind != "IRS" or spec.legs is None:
                calibrated_specs.append(spec)
                continue
            coupons = np.asarray(spec.legs.get("float_coupon", []), dtype=float)
            if coupons.size == 0 or np.allclose(coupons, 0.0):
                calibrated_specs.append(spec)
                continue
            frozen = None
            if isinstance(frozen_spreads, dict):
                frozen = frozen_spreads.get(spec.trade.trade_id)
            if frozen is not None:
                arr = np.asarray(frozen, dtype=float)
                if arr.size == coupons.size:
                    legs = {k: np.array(v, copy=True) for k, v in spec.legs.items()}
                    legs["float_spread"] = arr.copy()
                    calibrated_specs.append(
                        _TradeSpec(
                            trade=spec.trade,
                            kind=spec.kind,
                            notional=spec.notional,
                            ccy=spec.ccy,
                            legs=legs,
                        )
                    )
                    continue
            fwd_tenor = str(spec.legs.get("float_index_tenor", "")).upper()
            p_fwd = forward_curves_by_tenor.get(spec.ccy, {}).get(
                fwd_tenor,
                forward_curves.get(spec.ccy, discount_curves.get(spec.ccy)),
            )
            if p_fwd is None:
                calibrated_specs.append(spec)
                continue
            calibrated_specs.append(
                _TradeSpec(
                    trade=spec.trade,
                    kind=spec.kind,
                    notional=spec.notional,
                    ccy=spec.ccy,
                    legs=self._irs_utils.calibrate_float_spreads_from_coupon(spec.legs, p_fwd, t0=0.0),
                )
            )
        trade_specs = calibrated_specs

        valuation_times = self._augment_exposure_grid_with_trade_dates(valuation_times, trade_specs)
        if valuation_times.size < 2:
            valuation_times = np.array([0.0, max(float(snapshot.config.horizon_years), 1.0)], dtype=float)

        closeout_times = _build_sticky_closeout_times(valuation_times, mpor.mpor_years)
        observation_closeout_times = _build_sticky_closeout_times(observation_times, mpor.mpor_years)
        times = np.asarray(
            sorted(set(float(x) for x in np.concatenate((valuation_times, closeout_times)))),
            dtype=float,
        )

        hazard_times, hazard_rates, recovery_rates = self._build_hazard_rates(snapshot, hazards, recoveries)
        survival_curves = self._build_survival_curves(snapshot, hazard_times, hazard_rates)
        own_name = _own_name_from_runtime(snapshot).upper()
        if runtime := snapshot.config.runtime:
            xva = runtime.xva_analytic
            if xva.fva_borrowing_curve and _market_yield_spread(snapshot, model_ccy, xva.fva_borrowing_curve) is None:
                input_fallbacks.append(f"missing_funding_borrow_spread:{xva.fva_borrowing_curve}")
            if xva.fva_lending_curve and _market_yield_spread(snapshot, model_ccy, xva.fva_lending_curve) is None:
                input_fallbacks.append(f"missing_funding_lend_spread:{xva.fva_lending_curve}")
        if strict_ore_inputs and input_fallbacks:
            raise EngineRunError(
                "ORE-backed Python parity run is missing required market inputs and would fall back to synthetic assumptions: "
                + ", ".join(sorted(set(input_fallbacks)))
            )

        return _PythonLgmInputs(
            asof=_normalize_asof_date(snapshot.config.asof),
            times=times,
            valuation_times=valuation_times,
            observation_times=observation_times,
            observation_closeout_times=observation_closeout_times,
            discount_curves=discount_curves,
            forward_curves=forward_curves,
            forward_curves_by_tenor=forward_curves_by_tenor,
            forward_curves_by_name=forward_curves_by_name,
            swap_index_forward_tenors=swap_index_forward_tenors,
            inflation_curves=inflation_curves,
            xva_discount_curve=xva_discount_curve,
            funding_borrow_curve=funding_borrow_curve,
            funding_lend_curve=funding_lend_curve,
            survival_curves=survival_curves,
            hazard_times=hazard_times,
            hazard_rates=hazard_rates,
            recovery_rates=recovery_rates,
            lgm_params=lgm_params,
            model_ccy=model_ccy,
            seed=int(snapshot.config.runtime.simulation.seed) if snapshot.config.runtime else 42,
            fx_spots=fx_spots,
            fx_vols=fx_vols,
            swaption_normal_vols=swaption_normal_vols,
            cms_correlations=cms_correlations,
            stochastic_fx_pairs=stochastic_fx_pairs,
            trade_specs=tuple(trade_specs),
            unsupported=tuple(unsupported),
            mpor=mpor,
            input_provenance={
                "model_params": param_source,
                "market": curve_source,
                "grid": "xml+trade_dates" if grid_source == "xml" else grid_source,
                "portfolio": "dataclass",
                "mpor": mpor.source,
            },
            input_fallbacks=tuple(sorted(set(input_fallbacks))),
        )

    def _load_ore_output_curves(
        self,
        snapshot: XVASnapshot,
        mapped: MappedInputs,
        trade_specs: Sequence[_TradeSpec],
    ) -> Optional[_CurveBundle]:
        """Build discount, forward, and funding curves from ORE output artefacts.

        Reads ``curves.csv`` from the ORE output directory, resolves the
        relevant column names from ``todaysmarket.xml`` and ``ore.xml``, and
        constructs callable discount-factor curves for all required currencies.
        Returns ``None`` if the artefacts are absent, the feature is disabled,
        or the snapshot does not point to a full ORE directory.
        """
        if str(snapshot.config.params.get("python.use_ore_output_curves", "N")).strip().upper() in ("N", "FALSE", "0", ""):
            return None
        ore_path_txt = getattr(snapshot.config.source_meta, "path", "") or ""
        if not ore_path_txt:
            return None
        ore_path = Path(ore_path_txt).resolve()
        if ore_path.suffix.lower() != ".xml" or not ore_path.exists():
            return None

        curves_csv = (ore_path.parent.parent / snapshot.config.params.get("outputPath", "Output") / "curves.csv").resolve()
        if not curves_csv.exists():
            return None

        todaysmarket_xml = mapped.xml_buffers.get("todaysmarket.xml")
        simulation_xml = mapped.xml_buffers.get("simulation.xml")
        if not todaysmarket_xml or not simulation_xml:
            raise EngineRunError(
                "ORE output curves were found, but PythonLgmAdapter cannot consume them because "
                "'todaysmarket.xml' or 'simulation.xml' is missing from xml_buffers. "
                f"Fix: load the full ORE case rooted at '{ore_path.parent}' so both input XML files are available."
            )

        try:
            ore_root = ET.parse(ore_path).getroot()
            tm_root = ET.fromstring(todaysmarket_xml)
            sim_config_id = snapshot.config.params.get("market.simulation", "default")
            pricing_config_id = str(snapshot.config.params.get("market.pricing", "")).strip()
            if not pricing_config_id:
                curves_analytic = ore_root.find("./Analytics/Analytic[@type='curves']")
                if curves_analytic is not None:
                    curves_cfg_params = {
                        n.attrib.get("name", ""): (n.text or "").strip()
                        for n in curves_analytic.findall("./Parameter")
                    }
                    pricing_config_id = curves_cfg_params.get("configuration", "default")
                else:
                    pricing_config_id = "default"
            model_day_counter = self._day_counter_from_sim_xml(simulation_xml)
            ccy_set = {snapshot.config.base_currency.upper()}
            for spec in trade_specs:
                ccy_set.add(spec.ccy.upper())

            discount_meta = {
                ccy: {
                    "curve_id": "",
                    "source_column": self._ore_snapshot_mod._resolve_discount_column(tm_root, pricing_config_id, ccy),
                }
                for ccy in sorted(ccy_set)
            }
            xva_discount_meta = {
                ccy: {
                    "curve_id": "",
                    "source_column": self._ore_snapshot_mod._resolve_discount_column(tm_root, sim_config_id, ccy),
                }
                for ccy in sorted(ccy_set)
            }
            xva_params = {
                n.attrib.get("name", ""): (n.text or "").strip()
                for n in ore_root.findall("./Analytics/Analytic[@type='xva']/Parameter")
            }
            requested_columns = {meta["source_column"] for meta in discount_meta.values()}
            requested_columns.update(meta["source_column"] for meta in xva_discount_meta.values())
            forward_specs: Dict[str, Dict[str, str]] = {}
            forward_names: set[str] = set()
            for spec in trade_specs:
                if spec.kind != "IRS" or spec.legs is None:
                    if spec.kind != "RateSwap" or spec.legs is None:
                        continue
                    for leg in spec.legs.get("rate_legs", []):
                        for field in ("index_name", "index_name_1", "index_name_2"):
                            index_name = str(leg.get(field, "")).strip()
                            if index_name:
                                requested_columns.add(index_name)
                                forward_names.add(index_name.upper())
                    continue
                index_name = str(spec.legs.get("float_index", "")).strip()
                tenor_key = str(spec.legs.get("float_index_tenor", "")).upper()
                if index_name:
                    requested_columns.add(index_name)
                    forward_names.add(index_name.upper())
                if not index_name or not tenor_key:
                    continue
                forward_specs.setdefault(spec.ccy, {})[tenor_key] = index_name

            xva_ccy = snapshot.config.base_currency.upper()
            xva_curve_name = xva_discount_meta.get(xva_ccy, {}).get("source_column")
            runtime = snapshot.config.runtime
            if runtime is not None:
                lend_curve_name = runtime.xva_analytic.fva_lending_curve
                borrow_curve_name = runtime.xva_analytic.fva_borrowing_curve
            else:
                lend_curve_name = None
                borrow_curve_name = None
            if not borrow_curve_name:
                borrow_curve_name = xva_params.get("fvaBorrowingCurve")
            if not lend_curve_name:
                lend_curve_name = xva_params.get("fvaLendingCurve")
            if borrow_curve_name:
                requested_columns.add(str(borrow_curve_name))
            if lend_curve_name:
                requested_columns.add(str(lend_curve_name))

            curve_data = self._ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
                str(curves_csv),
                sorted(requested_columns),
                asof_date=snapshot.config.asof,
                day_counter=model_day_counter,
            )

            discount_curves: Dict[str, Callable[[float], float]] = {}
            for ccy, meta in discount_meta.items():
                discount_curves[ccy] = self._curve_from_column(curve_data, meta["source_column"])

            forward_curves: Dict[str, Callable[[float], float]] = {}
            forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]] = {}
            forward_curves_by_name: Dict[str, Callable[[float], float]] = {}
            for ccy, tenor_map in forward_specs.items():
                tenor_curves: Dict[str, Callable[[float], float]] = {}
                for tenor_key, column_name in tenor_map.items():
                    tenor_curves[tenor_key] = self._curve_from_column(curve_data, column_name)
                    forward_curves_by_name[column_name.upper()] = tenor_curves[tenor_key]
                forward_curves_by_tenor[ccy] = tenor_curves
                if tenor_curves:
                    preferred = "6M" if "6M" in tenor_curves else sorted(tenor_curves)[0]
                    forward_curves[ccy] = tenor_curves[preferred]
            for index_name in sorted(forward_names):
                if index_name in curve_data and index_name not in forward_curves_by_name:
                    forward_curves_by_name[index_name] = self._curve_from_column(curve_data, index_name)

            for ccy, disc in discount_curves.items():
                forward_curves.setdefault(ccy, disc)
                forward_curves_by_tenor.setdefault(ccy, {})
            xva_discount_curve = None
            if xva_curve_name and xva_curve_name in curve_data:
                xva_discount_curve = self._curve_from_column(curve_data, xva_curve_name)
            funding_borrow_curve = None
            if borrow_curve_name and str(borrow_curve_name) in curve_data:
                funding_borrow_curve = self._curve_from_column(curve_data, str(borrow_curve_name))
            funding_lend_curve = None
            if lend_curve_name and str(lend_curve_name) in curve_data:
                funding_lend_curve = self._curve_from_column(curve_data, str(lend_curve_name))

            return _CurveBundle(
                discount_curves=discount_curves,
                forward_curves=forward_curves,
                forward_curves_by_tenor=forward_curves_by_tenor,
                forward_curves_by_name=forward_curves_by_name,
                xva_discount_curve=xva_discount_curve,
                funding_borrow_curve=funding_borrow_curve,
                funding_lend_curve=funding_lend_curve,
            )
        except Exception as exc:
            raise EngineRunError(
                "Failed to build native curves from ORE output artifacts. "
                f"ore.xml='{ore_path}', curves.csv='{curves_csv}'. "
                "This usually means the case is only partially loaded, a required discount/forward curve "
                "handle cannot be resolved from todaysmarket.xml, or curves.csv does not contain the "
                "columns implied by the active ORE config. Fix: verify Input/todaysmarket.xml, "
                "Analytics/curves configuration in ore.xml, and the generated Output/curves.csv columns. "
                f"Original error: {exc}"
            ) from exc

    def _fit_curves_from_market_quotes(
        self,
        snapshot: XVASnapshot,
        trade_specs: Sequence[_TradeSpec],
    ) -> Optional[_CurveBundle]:
        """Bootstrap discount/forward curves from raw market quotes via the ORE SWIG curve-fitting path.

        Used as an alternative to reading pre-computed ORE output artefacts
        when ``python.curve_fit_mode`` is set to ``"ore_fit"``.  Returns
        ``None`` on any failure so the caller can fall back to the market
        overlay path.
        """
        try:
            quote_dicts = [{"key": str(q.key), "value": float(q.value)} for q in snapshot.market.raw_quotes]
            ccy_set = {snapshot.config.base_currency.upper()}
            for spec in trade_specs:
                ccy_set.add(spec.ccy.upper())
            ore_xml_text = snapshot.config.xml_buffers.get("ore.xml", "")
            ore_root = ET.fromstring(ore_xml_text) if ore_xml_text.strip() else None
            tm_root = ET.fromstring(snapshot.config.xml_buffers.get("todaysmarket.xml", ""))
            pricing_config_id = (
                (ore_root.findtext("./Markets/Parameter[@name='pricing']") if ore_root is not None else None)
                or "default"
            ).strip() or "default"
            sim_config_id = (
                (ore_root.findtext("./Markets/Parameter[@name='simulation']") if ore_root is not None else None)
                or pricing_config_id
            ).strip() or pricing_config_id
            discount_meta = {
                ccy: {
                    "source_column": self._ore_snapshot_mod._resolve_discount_column(tm_root, pricing_config_id, ccy),
                }
                for ccy in sorted(ccy_set)
            }
            xva_discount_meta = {
                ccy: {
                    "source_column": self._ore_snapshot_mod._resolve_discount_column(tm_root, sim_config_id, ccy),
                }
                for ccy in sorted(ccy_set)
            }
            swap_index_forward_tenors = self._parse_swap_index_forward_tenors(
                snapshot.config.xml_buffers.get("conventions.xml", "")
            )
            needed_tenors: Dict[str, set[str]] = {}
            for spec in trade_specs:
                if spec.kind == "IRS" and spec.legs is not None:
                    tenor = str(spec.legs.get("float_index_tenor", "")).upper()
                    if tenor:
                        needed_tenors.setdefault(spec.ccy.upper(), set()).add(tenor)
                elif spec.kind == "RateSwap" and isinstance(spec.legs, dict):
                    for leg in spec.legs.get("rate_legs", []):
                        for field in ("index_name", "index_name_1", "index_name_2"):
                            index_name = str(leg.get(field, "")).upper()
                            tenor = swap_index_forward_tenors.get(index_name, "")
                            if tenor:
                                needed_tenors.setdefault(spec.ccy.upper(), set()).add(tenor)
            discount_family_fallbacks = {
                ccy: (sorted(tenors)[0] if len(tenors) == 1 else "")
                for ccy, tenors in needed_tenors.items()
            }

            discount_curves: Dict[str, Callable[[float], float]] = {}
            for ccy in sorted(ccy_set):
                source_column = str(
                    discount_meta.get(ccy, {}).get("source_column")
                    or xva_discount_meta.get(ccy, {}).get("source_column")
                    or ""
                ).strip()
                discount_quotes = [
                    q
                    for q in quote_dicts
                    if _quote_matches_discount_curve(
                        str(q["key"]),
                        ccy,
                        source_column,
                        fallback_family=discount_family_fallbacks.get(ccy, ""),
                    )
                ]
                if not discount_quotes:
                    continue
                payload = self._ore_snapshot_mod.fit_discount_curves_from_programmatic_quotes(
                    snapshot.config.asof,
                    discount_quotes,
                    fit_method="bootstrap_mm_irs_v1",
                ).get(ccy)
                if not payload:
                    continue
                discount_curves[ccy] = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                    list(zip(payload["times"], payload["dfs"]))
                )
            if not discount_curves:
                return None

            forward_curves: Dict[str, Callable[[float], float]] = {}
            forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]] = {}
            forward_curves_by_name: Dict[str, Callable[[float], float]] = {}
            for ccy, tenors in needed_tenors.items():
                tenor_curves: Dict[str, Callable[[float], float]] = {}
                for tenor in sorted(tenors):
                    tenor_quotes = [
                        q for q in quote_dicts if _quote_matches_forward_curve(str(q["key"]), ccy, tenor)
                    ]
                    if not tenor_quotes:
                        continue
                    payload = self._ore_snapshot_mod.fit_discount_curves_from_programmatic_quotes(
                        snapshot.config.asof,
                        tenor_quotes,
                        fit_method="bootstrap_mm_irs_v1",
                    ).get(ccy)
                    if not payload:
                        continue
                    tenor_curves[tenor] = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                        list(zip(payload["times"], payload["dfs"]))
                    )
                    for spec in trade_specs:
                        if spec.kind == "IRS" and spec.ccy == ccy and spec.legs is not None and str(spec.legs.get("float_index_tenor", "")).upper() == tenor:
                            index_name = str(spec.legs.get("float_index", "")).upper()
                            if index_name:
                                forward_curves_by_name[index_name] = tenor_curves[tenor]
                    for index_name, mapped_tenor in swap_index_forward_tenors.items():
                        if index_name.startswith(ccy + "-") and mapped_tenor == tenor:
                            forward_curves_by_name.setdefault(index_name.upper(), tenor_curves[tenor])
                forward_curves_by_tenor[ccy] = tenor_curves
                if tenor_curves:
                    preferred = "6M" if "6M" in tenor_curves else sorted(tenor_curves)[0]
                    forward_curves[ccy] = tenor_curves[preferred]
                else:
                    forward_curves[ccy] = discount_curves.get(ccy)

            for ccy in ccy_set:
                forward_curves.setdefault(ccy, discount_curves.get(ccy))
                forward_curves_by_tenor.setdefault(ccy, {})

            base_curve = discount_curves.get(snapshot.config.base_currency.upper())
            return _CurveBundle(
                discount_curves=discount_curves,
                forward_curves=forward_curves,
                forward_curves_by_tenor=forward_curves_by_tenor,
                forward_curves_by_name=forward_curves_by_name,
                swap_index_forward_tenors=swap_index_forward_tenors,
                xva_discount_curve=base_curve,
                funding_borrow_curve=None,
                funding_lend_curve=None,
            )
        except Exception:
            return None

    def _python_lgm_rng_mode(self, snapshot: XVASnapshot) -> str:
        """Determine the RNG mode for LGM path simulation from config params.

        Reads ``python.lgm_rng_mode`` from the snapshot config and returns a
        normalised lower-case string (``"numpy"``, ``"ore_parity"``,
        ``"ore_parity_antithetic"``, ``"ore_sobol"``, or
        ``"ore_sobol_bridge"``).
        """
        return str(snapshot.config.params.get("python.lgm_rng_mode", "numpy")).strip().lower()

    def _is_ore_case_snapshot(self, snapshot: XVASnapshot) -> bool:
        """Return True if the snapshot was loaded from a full ORE directory.

        Checks whether ``source_meta.path`` points to an ``.xml`` file (i.e.,
        an ``ore.xml`` case file), which is the signature of a snapshot created
        by ``XVALoader.from_files``.
        """
        ore_path_txt = getattr(snapshot.config.source_meta, "path", "") or ""
        if not ore_path_txt:
            return False
        ore_path = Path(ore_path_txt)
        return ore_path.suffix.lower() == ".xml"

    def _augment_exposure_grid_with_trade_dates(self, times: np.ndarray, trade_specs: Sequence[_TradeSpec]) -> np.ndarray:
        out = np.asarray(times, dtype=float)
        extras: List[np.ndarray] = []
        for spec in trade_specs:
            if spec.kind == "IRS" and spec.legs is not None:
                legs = spec.legs
                for key in ("fixed_pay_time", "float_pay_time", "float_fixing_time"):
                    vals = np.asarray(legs.get(key, np.array([], dtype=float)), dtype=float)
                    if vals.size == 0:
                        continue
                    extras.append(vals[vals >= 0.0])
            elif spec.kind == "RateSwap" and spec.legs is not None:
                for leg in spec.legs.get("rate_legs", []):
                    for key in ("pay_time", "fixing_time"):
                        vals = np.asarray(leg.get(key, np.array([], dtype=float)), dtype=float)
                        if vals.size == 0:
                            continue
                        extras.append(vals[vals >= 0.0])
            elif spec.kind == "InflationSwap" and isinstance(spec.trade.product, InflationSwap):
                product = spec.trade.product
                if str(product.inflation_type).upper() == "YY":
                    pay_times = self._inflation_mod.inflation_swap_payment_times(
                        float(product.maturity_years),
                        str(product.schedule_tenor or "1Y"),
                    )
                    if pay_times:
                        extras.append(np.asarray(pay_times, dtype=float))
                else:
                    extras.append(np.asarray([max(float(product.maturity_years), 0.0)], dtype=float))
            elif spec.kind == "InflationCapFloor" and isinstance(spec.trade.product, InflationCapFloor):
                extras.append(np.asarray([max(float(spec.trade.product.maturity_years), 0.0)], dtype=float))
            elif spec.kind == "FXForward" and isinstance(spec.trade.product, FXForward):
                extras.append(np.asarray([max(float(spec.trade.product.maturity_years), 0.0)], dtype=float))
        if extras:
            out = np.unique(np.concatenate([out, *extras]))
        if out.size == 0 or out[0] > 0.0:
            out = np.unique(np.concatenate([np.array([0.0]), out]))
        return out

    def _build_lgm_rng(self, seed: int, rng_mode: str):
        """Construct the random number generator for LGM path simulation.

        Supports ``"numpy"`` (NumPy default RNG, time-major draw order),
        ``"ore_parity"`` (QuantLib MT Gaussian RNG in ORE path-major order),
        ``"ore_parity_antithetic"`` (QuantLib MT Gaussian RNG with antithetic
        path pairing in ORE path-major order),
        ``"ore_sobol"`` (QuantLib Sobol Gaussian RNG in ORE path-major order),
        and ``"ore_sobol_bridge"`` (QuantLib Sobol Gaussian RNG with
        Brownian-bridge rotation in ORE path-major order). Raises
        :class:`EngineRunError` for any other mode string.
        """
        if rng_mode == "numpy":
            return np.random.default_rng(seed), "time_major"
        if rng_mode == "ore_parity":
            return self._lgm_mod.make_ore_gaussian_rng(seed), "ore_path_major"
        if rng_mode == "ore_parity_antithetic":
            return self._lgm_mod.make_ore_gaussian_rng(seed, sequence_type="MersenneTwisterAntithetic"), "ore_path_major"
        if rng_mode == "ore_sobol":
            return self._lgm_mod.make_ore_gaussian_rng(seed, sequence_type="Sobol"), "ore_path_major"
        if rng_mode == "ore_sobol_bridge":
            return self._lgm_mod.make_ore_gaussian_rng(seed, sequence_type="SobolBrownianBridge"), "ore_path_major"
        raise EngineRunError(
            "Unsupported PythonLgmAdapter RNG mode "
            f"'{rng_mode}'. Use 'numpy', 'ore_parity', 'ore_parity_antithetic', 'ore_sobol' or "
            "'ore_sobol_bridge'."
        )

    def _day_counter_from_sim_xml(self, sim_xml: str) -> str:
        """Return the normalised day-counter string from simulation.xml text."""
        try:
            root = self._simulation_root_from_xml(sim_xml)
            raw = (root.findtext("./DayCounter") or "A365F").strip()
            return self._ore_snapshot_mod._normalize_day_counter_name(raw)
        except Exception:
            return "A365F"

    def _curve_from_column(
        self, curve_data: Dict[str, Any], col_name: str
    ) -> Callable[[float], float]:
        """Build a discount-factor curve callable from a *curve_data* column."""
        _, times, dfs = curve_data[col_name]
        return self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
            list(zip(times, dfs))
        )

    def _build_shared_fx_simulation(
        self,
        snapshot: XVASnapshot,
        inputs: _PythonLgmInputs,
        n_paths: int,
    ) -> _SharedFxSimulation | None:
        all_fx_pairs = sorted(
            {
                f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}"
                for spec in inputs.trade_specs
                if spec.kind == "FXForward" and isinstance(spec.trade.product, FXForward)
            }
        )
        fx_pairs = sorted(pair for pair in all_fx_pairs if pair in set(inputs.stochastic_fx_pairs))
        if not fx_pairs:
            return None

        ir_ccys = sorted(
            {
                snapshot.config.base_currency.upper(),
                *[pair.split("/")[0] for pair in fx_pairs],
                *[pair.split("/")[1] for pair in fx_pairs],
            }
        )
        ir_params = {
            ccy: self._fx_utils.build_lgm_params(
                alpha=(
                    tuple(float(x) for x in inputs.lgm_params["alpha_times"]),
                    tuple(float(x) for x in inputs.lgm_params["alpha_values"]),
                ),
                kappa=(
                    tuple(float(x) for x in inputs.lgm_params["kappa_times"]),
                    tuple(float(x) for x in inputs.lgm_params["kappa_values"]),
                ),
                shift=float(inputs.lgm_params["shift"]),
                scaling=float(inputs.lgm_params["scaling"]),
            )
            for ccy in ir_ccys
        }
        fx_vols = {}
        log_s0 = {}
        rd_minus_rf = {}
        for pair in fx_pairs:
            pair6 = pair.replace("/", "")
            max_maturity = max(
                float(spec.trade.product.maturity_years)
                for spec in inputs.trade_specs
                if spec.kind == "FXForward"
                and isinstance(spec.trade.product, FXForward)
                and f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}" == pair
            )
            fx_vol = _fx_vol_for_trade(inputs, pair6, max_maturity, default=0.15)
            fx_vols[pair] = (tuple(), (float(fx_vol),))
            spot = max(_spot_from_quotes(pair6, inputs, default=1.0), 1.0e-12)
            log_s0[pair] = float(np.log(spot))
            rd_minus_rf[pair] = _fx_carry_for_pair(pair, inputs, horizon=max_maturity)

        corr = self._build_shared_fx_correlation(snapshot, tuple(ir_ccys), tuple(fx_pairs))
        hybrid = self._fx_utils.LgmFxHybrid(
            self._fx_utils.MultiCcyLgmParams(
                ir_params=ir_params,
                fx_vols=fx_vols,
                corr=corr,
            )
        )
        sim = hybrid.simulate_paths(
            times=inputs.times,
            n_paths=n_paths,
            rng=np.random.default_rng(inputs.seed + 17),
            log_s0=log_s0,
            rd_minus_rf=rd_minus_rf,
        )
        return _SharedFxSimulation(hybrid=hybrid, sim=sim, pair_keys=tuple(fx_pairs))

    def _build_shared_fx_correlation(
        self,
        snapshot: XVASnapshot,
        ir_ccys: Tuple[str, ...],
        fx_pairs: Tuple[str, ...],
    ) -> np.ndarray:
        labels = [f"IR:{ccy}" for ccy in ir_ccys] + [f"FX:{pair}" for pair in fx_pairs]
        corr = np.eye(len(labels), dtype=float)
        runtime = snapshot.config.runtime
        if runtime is None:
            return corr
        entries = getattr(runtime.cross_asset_model, "correlations", ())
        if not entries:
            return corr
        for raw_left, raw_right, raw_value in entries:
            left = self._resolve_cam_factor(str(raw_left), ir_ccys, fx_pairs)
            right = self._resolve_cam_factor(str(raw_right), ir_ccys, fx_pairs)
            if left is None or right is None:
                continue
            li, ls = left
            ri, rs = right
            value = float(raw_value) * float(ls) * float(rs)
            corr[li, ri] = value
            corr[ri, li] = value
        np.fill_diagonal(corr, 1.0)
        return corr

    def _resolve_cam_factor(
        self,
        label: str,
        ir_ccys: Tuple[str, ...],
        fx_pairs: Tuple[str, ...],
    ) -> tuple[int, float] | None:
        txt = str(label).strip().upper()
        if txt.startswith("IR:"):
            ccy = txt.split(":", 1)[1]
            if ccy in ir_ccys:
                return ir_ccys.index(ccy), 1.0
            return None
        if not txt.startswith("FX:"):
            return None
        pair_txt = txt.split(":", 1)[1].replace("-", "/")
        if "/" not in pair_txt and len(pair_txt) == 6:
            pair_txt = pair_txt[:3] + "/" + pair_txt[3:]
        if pair_txt in fx_pairs:
            return len(ir_ccys) + fx_pairs.index(pair_txt), 1.0
        base, quote = pair_txt.split("/", 1)
        rev = f"{quote}/{base}"
        if rev in fx_pairs:
            return len(ir_ccys) + fx_pairs.index(rev), -1.0
        return None

    def _price_fx_forward(
        self,
        trade: Trade,
        inputs: _PythonLgmInputs,
        n_times: int,
        n_paths: int,
        shared_sim: _SharedFxSimulation | None = None,
    ) -> np.ndarray:
        """Compute the NPV of an FX forward across all simulation times and paths under the LGM measure.

        Builds a two-currency LGM-FX hybrid model, simulates joint spot and
        state-variable paths, and evaluates the forward NPV at each
        (time, path) grid point.  Returns a ``(n_times, n_paths)`` array.
        """
        p = trade.product
        assert isinstance(p, FXForward)
        pair = f"{p.pair[:3].upper()}/{p.pair[3:].upper()}"
        dom = p.pair[3:].upper()
        for_ccy = p.pair[:3].upper()
        p_dom = inputs.discount_curves[dom]
        p_for = inputs.discount_curves[for_ccy]
        if shared_sim is not None and pair in shared_sim.sim["s"]:
            hybrid = shared_sim.hybrid
            sim = shared_sim.sim
            s_t = sim["s"][pair]
            x_dom_t = sim["x"][dom]
            x_for_t = sim["x"][for_ccy]
        elif pair not in set(inputs.stochastic_fx_pairs):
            return self._price_fx_forward_deterministic(trade, inputs, n_times, n_paths)
        else:
            spot = _spot_from_quotes(p.pair, inputs, default=1.0)
            fx_vol = _fx_vol_for_trade(inputs, p.pair, float(p.maturity_years), default=0.15)
            ir_specs = {
                for_ccy: {
                    "alpha": (tuple(float(x) for x in inputs.lgm_params["alpha_times"]), tuple(float(x) for x in inputs.lgm_params["alpha_values"])),
                    "kappa": (tuple(float(x) for x in inputs.lgm_params["kappa_times"]), tuple(float(x) for x in inputs.lgm_params["kappa_values"])),
                    "shift": float(inputs.lgm_params["shift"]),
                    "scaling": float(inputs.lgm_params["scaling"]),
                },
                dom: {
                    "alpha": (tuple(float(x) for x in inputs.lgm_params["alpha_times"]), tuple(float(x) for x in inputs.lgm_params["alpha_values"])),
                    "kappa": (tuple(float(x) for x in inputs.lgm_params["kappa_times"]), tuple(float(x) for x in inputs.lgm_params["kappa_values"])),
                    "shift": float(inputs.lgm_params["shift"]),
                    "scaling": float(inputs.lgm_params["scaling"]),
                },
            }
            hybrid = self._fx_utils.build_two_ccy_hybrid(pair=pair, ir_specs=ir_specs, fx_vol=fx_vol)
            sim = hybrid.simulate_paths(
                times=inputs.times,
                n_paths=n_paths,
                log_s0={pair: float(np.log(max(spot, 1.0e-12)))},
                rd_minus_rf={pair: _fx_carry_for_pair(pair, inputs, horizon=float(p.maturity_years))},
                rng=np.random.default_rng(inputs.seed + 17),
            )
            s_t = sim["s"][pair]
            x_dom_t = sim["x"][dom]
            x_for_t = sim["x"][for_ccy]
        fx_def = self._fx_utils.FxForwardDef(
            trade_id=trade.trade_id,
            pair=pair,
            notional_base=float(p.notional),
            strike=float(p.strike),
            maturity=float(p.maturity_years),
        )
        direction = 1.0 if p.buy_base else -1.0
        vals = np.zeros((n_times, n_paths), dtype=float)
        report_ccy = inputs.model_ccy.upper()
        for i, t in enumerate(inputs.times):
            npv_quote = direction * self._fx_utils.fx_forward_npv(
                hybrid=hybrid,
                fx_def=fx_def,
                t=float(t),
                s_t=s_t[i, :],
                x_dom_t=x_dom_t[i, :],
                x_for_t=x_for_t[i, :],
                p0_dom=p_dom,
                p0_for=p_for,
            )
            vals[i, :] = _convert_fx_forward_npv_to_reporting_ccy(
                npv_quote=npv_quote,
                report_ccy=report_ccy,
                base_ccy=for_ccy,
                quote_ccy=dom,
                spot_path=s_t[i, :],
                inputs=inputs,
            )
        return vals

    def _price_fx_forward_deterministic(
        self,
        trade: Trade,
        inputs: _PythonLgmInputs,
        n_times: int,
        n_paths: int,
    ) -> np.ndarray:
        p = trade.product
        assert isinstance(p, FXForward)
        dom = p.pair[3:].upper()
        for_ccy = p.pair[:3].upper()
        direction = 1.0 if p.buy_base else -1.0
        spot0 = _spot_from_quotes(p.pair, inputs, default=1.0)
        p_dom = inputs.discount_curves[dom]
        p_for = inputs.discount_curves[for_ccy]
        report_ccy = inputs.model_ccy.upper()
        vals = np.zeros((n_times, n_paths), dtype=float)
        maturity = float(p.maturity_years)
        forward0 = float(spot0) * float(p_for(maturity)) / float(p_dom(maturity))
        for i, t in enumerate(inputs.times):
            if t > maturity + 1.0e-12:
                continue
            p_d_t_T = float(p_dom(maturity)) / max(float(p_dom(float(t))), 1.0e-18)
            npv_quote = direction * float(p.notional) * p_d_t_T * (forward0 - float(p.strike))
            vals[i, :] = _convert_fx_forward_npv_to_reporting_ccy(
                npv_quote=np.full(n_paths, npv_quote, dtype=float),
                report_ccy=report_ccy,
                base_ccy=for_ccy,
                quote_ccy=dom,
                spot_path=np.full(n_paths, max(float(spot0), 1.0e-12), dtype=float),
                inputs=inputs,
            )
        return vals

    def _fx_forward_closeout_paths(
        self,
        trade: Trade,
        npv_paths: np.ndarray,
        times: np.ndarray,
        observation_times: np.ndarray,
        closeout_times: np.ndarray,
    ) -> np.ndarray:
        p = trade.product
        assert isinstance(p, FXForward)
        maturity = float(p.maturity_years)
        out = np.zeros((observation_times.size, npv_paths.shape[1]), dtype=float)
        for i, (obs_t, co_t) in enumerate(zip(observation_times, closeout_times)):
            if obs_t >= maturity - 1.0e-12:
                continue
            target = float(co_t)
            if maturity <= co_t + 1.0e-12:
                target = maturity
            idx = int(np.searchsorted(times, target))
            if idx >= times.size:
                idx = times.size - 1
            if abs(float(times[idx]) - target) > 1.0e-10:
                if idx == 0:
                    idx_lo = idx_hi = 0
                elif idx >= times.size:
                    idx_lo = idx_hi = times.size - 1
                else:
                    idx_lo = idx - 1
                    idx_hi = idx
                t_lo = float(times[idx_lo])
                t_hi = float(times[idx_hi])
                if idx_lo == idx_hi or abs(t_hi - t_lo) <= 1.0e-12:
                    out[i, :] = npv_paths[idx_lo, :]
                else:
                    w = (target - t_lo) / (t_hi - t_lo)
                    out[i, :] = (1.0 - w) * npv_paths[idx_lo, :] + w * npv_paths[idx_hi, :]
            else:
                out[i, :] = npv_paths[idx, :]
        return out

    def _build_irs_legs(self, trade: Trade, mapped: MappedInputs, snapshot: XVASnapshot) -> Dict[str, np.ndarray]:
        cache_key = (trade.trade_id, id(snapshot))
        cached_legs = self._irs_leg_cache.get(cache_key)
        if cached_legs is not None:
            return cached_legs
        ore_path_txt = getattr(snapshot.config.source_meta, "path", "") or ""
        if ore_path_txt:
            try:
                ore_path = Path(ore_path_txt).resolve()
                output_dir = (ore_path.parent.parent / snapshot.config.params.get("outputPath", "Output")).resolve()
                flows_csv = output_dir / "flows.csv"
                if flows_csv.exists():
                    asof = self._normalized_asof(snapshot)
                    model_day_counter = "A365F"
                    sim_xml = mapped.xml_buffers.get("simulation.xml")
                    if sim_xml:
                        model_day_counter = self._day_counter_from_sim_xml(sim_xml)
                    legs = self._irs_utils.load_ore_legs_from_flows(
                        str(flows_csv),
                        trade_id=trade.trade_id,
                        asof_date=asof,
                        time_day_counter=model_day_counter,
                    )
                    if sim_xml:
                        try:
                            legs["node_tenors"] = self._simulation_node_tenors_from_xml(sim_xml)
                        except Exception:
                            pass
                    idx = str(trade.additional_fields.get("index", "")).upper()
                    if idx:
                        legs.setdefault("float_index", idx)
                        if "float_index_tenor" not in legs or not str(legs.get("float_index_tenor", "")).strip():
                            legs["float_index_tenor"] = idx.split("-")[-1].upper() if "-" in idx else ""
                    result = self._apply_historical_fixings_to_legs(snapshot, trade, legs)
                    self._irs_leg_cache[cache_key] = result
                    return result
            except Exception as exc:
                import warnings as _warnings
                _warnings.warn(
                    f"Failed to load legs from flows.csv for trade {trade.trade_id}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
        portfolio_xml = mapped.xml_buffers.get("portfolio.xml")
        if portfolio_xml:
            asof = self._normalized_asof(snapshot)
            try:
                legs = self._irs_utils.load_swap_legs_from_portfolio_root(
                    self._portfolio_root_from_xml(portfolio_xml),
                    trade.trade_id,
                    asof,
                )
                sim_xml = mapped.xml_buffers.get("simulation.xml")
                if sim_xml:
                    try:
                        legs["node_tenors"] = self._simulation_node_tenors_from_xml(sim_xml)
                    except Exception:
                        pass
                result = self._apply_historical_fixings_to_legs(snapshot, trade, legs)
                self._irs_leg_cache[cache_key] = result
                return result
            except Exception as exc:
                import warnings as _warnings
                _warnings.warn(
                    f"Failed to load legs from portfolio.xml for trade {trade.trade_id}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
        result = self._apply_historical_fixings_to_legs(
            snapshot,
            trade,
            _build_irs_legs_from_trade(trade, snapshot.config.asof),
        )
        self._irs_leg_cache[cache_key] = result
        return result

    def _build_generic_rate_swap_legs(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
    ) -> Optional[Dict[str, object]]:
        cache_key = (trade.trade_id, id(snapshot))
        if cache_key in self._generic_rate_swap_legs_cache:
            return self._generic_rate_swap_legs_cache[cache_key]
        product = trade.product
        if not isinstance(product, GenericProduct):
            return None
        if str(product.payload.get("trade_type", "")).strip() != "Swap":
            return None
        xml = str(product.payload.get("xml", "")).strip()
        if "<SwapData" not in xml:
            return None
        try:
            trade_root = ET.fromstring("<Trade>{}</Trade>".format(xml))
        except Exception:
            return None
        swap = trade_root.find("./SwapData")
        if swap is None:
            return None
        leg_nodes = swap.findall("./LegData")
        if not leg_nodes:
            return None

        asof = datetime.strptime(self._normalized_asof(snapshot), "%Y-%m-%d").date()
        parse_date = self._irs_utils._parse_yyyymmdd
        build_schedule = self._irs_utils._build_schedule
        time_from_dates = self._irs_utils._time_from_dates
        year_fraction = self._irs_utils._year_fraction
        advance_business_days = self._irs_utils._advance_business_days
        infer_index_day_counter = self._irs_utils._infer_index_day_counter

        rate_legs: List[Dict[str, object]] = []
        ccy = None
        for leg in leg_nodes:
            leg_type = (leg.findtext("./LegType") or "").strip()
            leg_type_upper = leg_type.upper()
            if leg_type_upper not in {"FIXED", "FLOATING", "CMS", "CMSSPREAD", "DIGITALCMSSPREAD"}:
                return None
            leg_ccy = (leg.findtext("./Currency") or "").strip().upper() or snapshot.config.base_currency.upper()
            ccy = ccy or leg_ccy
            payer = (leg.findtext("./Payer") or "").strip().lower() == "true"
            sign = -1.0 if payer else 1.0
            notional = float((leg.findtext("./Notionals/Notional") or "0").strip())
            dc = (leg.findtext("./DayCounter") or "A365").strip()
            pay_conv = (leg.findtext("./PaymentConvention") or "F").strip()
            rules = leg.find("./ScheduleData/Rules")
            if rules is None:
                return None
            start = parse_date((rules.findtext("./StartDate") or "").strip())
            end = parse_date((rules.findtext("./EndDate") or "").strip())
            tenor = (rules.findtext("./Tenor") or "").strip()
            cal = (rules.findtext("./Calendar") or "TARGET").strip()
            conv = (rules.findtext("./Convention") or pay_conv).strip()
            s_dates, e_dates, p_dates = build_schedule(start, end, tenor, cal, conv, pay_convention=pay_conv)
            s_t = np.asarray([time_from_dates(asof, d, "A365F") for d in s_dates], dtype=float)
            e_t = np.asarray([time_from_dates(asof, d, "A365F") for d in e_dates], dtype=float)
            p_t = np.asarray([time_from_dates(asof, d, "A365F") for d in p_dates], dtype=float)
            accr = np.asarray([year_fraction(sd, ed, dc) for sd, ed in zip(s_dates, e_dates)], dtype=float)
            leg_info: Dict[str, object] = {
                "kind": leg_type_upper,
                "ccy": leg_ccy,
                "notional": notional,
                "sign": sign,
                "start_time": s_t,
                "end_time": e_t,
                "pay_time": p_t,
                "accrual": accr,
                "schedule_tenor": tenor,
                "calendar": cal,
                "day_counter": dc,
            }
            if leg_type_upper == "FIXED":
                rate = float((leg.findtext("./FixedLegData/Rates/Rate") or "0").strip())
                amount = sign * notional * rate * accr
                leg_info["fixed_rate"] = np.full_like(accr, rate)
                leg_info["amount"] = amount
            elif leg_type_upper == "FLOATING":
                fld = leg.find("./FloatingLegData")
                if fld is None:
                    return None
                index_name = (fld.findtext("./Index") or "").strip().upper()
                spread = float((fld.findtext("./Spreads/Spread") or "0").strip() or 0.0)
                gearing = float((fld.findtext("./Gearings/Gearing") or "1").strip() or 1.0)
                fixing_days = int((fld.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
                index_dc = infer_index_day_counter(index_name, fallback=dc)
                fix_base_dates = e_dates if in_arrears else s_dates
                fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["index_name"] = index_name
                leg_info["spread"] = np.full_like(accr, spread)
                leg_info["gearing"] = np.full_like(accr, gearing)
                leg_info["is_in_arrears"] = in_arrears
                leg_info["fixing_days"] = fixing_days
                leg_info["fixing_time"] = np.asarray([time_from_dates(asof, fd, "A365F") for fd in fix_dates], dtype=float)
                leg_info["index_accrual"] = np.asarray([year_fraction(sd, ed, index_dc) for sd, ed in zip(s_dates, e_dates)], dtype=float)
                leg_info["quoted_coupon"] = np.zeros_like(accr)
            elif leg_type_upper == "CMS":
                fld = leg.find("./CMSLegData")
                if fld is None:
                    return None
                index_name = (fld.findtext("./Index") or "").strip().upper()
                spread = float((fld.findtext("./Spreads/Spread") or "0").strip() or 0.0)
                fixing_days = int((fld.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
                fix_base_dates = e_dates if in_arrears else s_dates
                fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["index_name"] = index_name
                leg_info["spread"] = np.full_like(accr, spread)
                leg_info["gearing"] = np.ones_like(accr)
                leg_info["is_in_arrears"] = in_arrears
                leg_info["fixing_days"] = fixing_days
                leg_info["fixing_time"] = np.asarray([time_from_dates(asof, fd, "A365F") for fd in fix_dates], dtype=float)
                leg_info["quoted_coupon"] = np.zeros_like(accr)
            elif leg_type_upper == "CMSSPREAD":
                fld = leg.find("./CMSSpreadLegData")
                if fld is None:
                    return None
                index1 = (fld.findtext("./Index1") or "").strip().upper()
                index2 = (fld.findtext("./Index2") or "").strip().upper()
                spread = float((fld.findtext("./Spreads/Spread") or "0").strip() or 0.0)
                gearing = float((fld.findtext("./Gearings/Gearing") or "1").strip() or 1.0)
                cap_txt = (fld.findtext("./Caps/Cap") or "").strip()
                floor_txt = (fld.findtext("./Floors/Floor") or "").strip()
                fixing_days = int((fld.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
                fix_base_dates = e_dates if in_arrears else s_dates
                fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["index_name_1"] = index1
                leg_info["index_name_2"] = index2
                leg_info["spread"] = np.full_like(accr, spread)
                leg_info["gearing"] = np.full_like(accr, gearing)
                leg_info["cap"] = float(cap_txt) if cap_txt else None
                leg_info["floor"] = float(floor_txt) if floor_txt else None
                leg_info["naked_option"] = (fld.findtext("./NakedOption") or "false").strip().lower() == "true"
                leg_info["is_in_arrears"] = in_arrears
                leg_info["fixing_days"] = fixing_days
                leg_info["fixing_time"] = np.asarray([time_from_dates(asof, fd, "A365F") for fd in fix_dates], dtype=float)
                leg_info["quoted_coupon"] = np.zeros_like(accr)
            else:
                fld = leg.find("./DigitalCMSSpreadLegData")
                if fld is None:
                    return None
                cms = fld.find("./CMSSpreadLegData")
                if cms is None:
                    return None
                index1 = (cms.findtext("./Index1") or "").strip().upper()
                index2 = (cms.findtext("./Index2") or "").strip().upper()
                spread = float((cms.findtext("./Spreads/Spread") or "0").strip() or 0.0)
                gearing = float((cms.findtext("./Gearings/Gearing") or "1").strip() or 1.0)
                fixing_days = int((cms.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (cms.findtext("./IsInArrears") or "false").strip().lower() == "true"
                fix_base_dates = e_dates if in_arrears else s_dates
                fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["index_name_1"] = index1
                leg_info["index_name_2"] = index2
                leg_info["spread"] = np.full_like(accr, spread)
                leg_info["gearing"] = np.full_like(accr, gearing)
                call_strike_txt = (fld.findtext("./CallStrikes/Strike") or "").strip()
                call_payoff_txt = (fld.findtext("./CallPayoffs/Payoff") or "").strip()
                put_strike_txt = (fld.findtext("./PutStrikes/Strike") or "").strip()
                put_payoff_txt = (fld.findtext("./PutPayoffs/Payoff") or "").strip()
                leg_info["call_strike"] = float(call_strike_txt) if call_strike_txt else math.nan
                leg_info["call_payoff"] = float(call_payoff_txt) if call_payoff_txt else math.nan
                leg_info["put_strike"] = float(put_strike_txt) if put_strike_txt else math.nan
                leg_info["put_payoff"] = float(put_payoff_txt) if put_payoff_txt else math.nan
                leg_info["call_position"] = -1.0 if (fld.findtext("./CallPosition") or "Long").strip().lower() == "short" else 1.0
                leg_info["put_position"] = -1.0 if (fld.findtext("./PutPosition") or "Long").strip().lower() == "short" else 1.0
                leg_info["is_call_atm_included"] = (fld.findtext("./IsCallATMIncluded") or "false").strip().lower() == "true"
                leg_info["is_put_atm_included"] = (fld.findtext("./IsPutATMIncluded") or "false").strip().lower() == "true"
                leg_info["naked_option"] = (cms.findtext("./NakedOption") or "false").strip().lower() == "true"
                leg_info["is_in_arrears"] = in_arrears
                leg_info["fixing_days"] = fixing_days
                leg_info["fixing_time"] = np.asarray([time_from_dates(asof, fd, "A365F") for fd in fix_dates], dtype=float)
                leg_info["quoted_coupon"] = np.zeros_like(accr)
            rate_legs.append(leg_info)

        result = self._apply_historical_fixings_to_generic_rate_legs(
            snapshot,
            {"ccy": ccy or snapshot.config.base_currency.upper(), "rate_legs": rate_legs},
        )
        self._generic_rate_swap_legs_cache[cache_key] = result
        return result

    def _apply_historical_fixings_to_generic_rate_legs(
        self,
        snapshot: XVASnapshot,
        legs: Dict[str, object],
    ) -> Dict[str, object]:
        asof = self._normalized_asof(snapshot)
        fixings = self._fixings_lookup(snapshot)
        for leg in legs.get("rate_legs", []):
            fixing_time = np.asarray(leg.get("fixing_time", []), dtype=float)
            if fixing_time.size == 0:
                continue
            fixed_mask = np.zeros(fixing_time.shape, dtype=bool)
            coupon = np.asarray(leg.get("quoted_coupon", np.zeros(fixing_time.shape)), dtype=float).copy()
            if leg.get("kind") in {"FLOATING", "CMS"}:
                index_name = str(leg.get("index_name", "")).upper()
                for i, ft in enumerate(fixing_time):
                    fixing_date = self._date_from_time_cached(snapshot, float(ft))
                    key = (index_name, fixing_date)
                    if fixing_date <= asof and key in fixings:
                        coupon[i] = fixings[key]
                        fixed_mask[i] = True
            leg["quoted_coupon"] = coupon
            leg["is_historically_fixed"] = fixed_mask
        return legs

    def _apply_historical_fixings_to_legs(
        self, snapshot: XVASnapshot, trade: Trade, legs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        fix_t = np.asarray(legs.get("float_fixing_time", []), dtype=float)
        if fix_t.size == 0:
            return legs
        asof = self._normalized_asof(snapshot)
        fixings = self._fixings_lookup(snapshot)
        p = trade.product
        assert isinstance(p, IRS)
        index_name = str(legs.get("float_index", trade.additional_fields.get("index", _default_index_for_ccy(p.ccy)))).upper()
        coupons = np.asarray(legs.get("float_coupon", np.zeros_like(fix_t)), dtype=float).copy()
        fixed_mask = np.asarray(legs.get("float_is_historically_fixed", np.zeros(fix_t.shape, dtype=bool)), dtype=bool).copy()
        for i, ft in enumerate(fix_t):
            fixing_date = self._date_from_time_cached(snapshot, float(ft))
            if fixing_date <= asof:
                key = (index_name, fixing_date)
                if key in fixings:
                    coupons[i] = fixings[key]
                    fixed_mask[i] = True
        legs["float_coupon"] = coupons
        legs["float_is_historically_fixed"] = fixed_mask
        return legs

    def _compute_realized_float_coupons(
        self,
        model: Any,
        p0_disc: Callable[[float], float],
        p0_fwd: Callable[[float], float],
        legs: Dict[str, np.ndarray],
        sim_times: np.ndarray,
        x_paths_on_sim_grid: np.ndarray,
    ) -> np.ndarray | None:
        s = np.asarray(legs.get("float_start_time", []), dtype=float)
        if s.size == 0:
            return None
        e = np.asarray(legs.get("float_end_time", []), dtype=float)
        tau = np.asarray(legs.get("float_accrual", []), dtype=float)
        spr = np.asarray(legs.get("float_spread", np.zeros_like(s)), dtype=float)
        fix_t = np.asarray(legs.get("float_fixing_time", s), dtype=float)
        pay_t = np.asarray(legs.get("float_pay_time", e), dtype=float)
        notional = np.asarray(legs.get("float_notional", np.ones_like(s)), dtype=float)
        sign = np.asarray(legs.get("float_sign", np.ones_like(s)), dtype=float)
        quoted_coupon = np.asarray(legs.get("float_coupon", np.zeros_like(s)), dtype=float)
        fixed_mask = np.asarray(legs.get("float_is_historically_fixed", np.zeros(s.shape, dtype=bool)), dtype=bool)

        n_cf = s.size
        n_paths = x_paths_on_sim_grid.shape[1]
        out = np.zeros((n_cf, n_paths), dtype=float)

        for i in range(n_cf):
            if fixed_mask[i]:
                # Historical fixings already overwrite quoted_coupon with the full
                # locked coupon, so adding spread again would double-count it.
                out[i, :] = quoted_coupon[i]
                continue
            if tau[i] <= 0.0:
                out[i, :] = quoted_coupon[i]
                continue
            ft = float(fix_t[i])
            if ft <= 1.0e-12:
                ps = float(p0_fwd(max(0.0, float(s[i]))))
                pe = float(p0_fwd(float(e[i])))
                fwd = (ps / pe - 1.0) / float(tau[i])
                out[i, :] = fwd + float(spr[i])
                continue
            j = int(np.searchsorted(sim_times, ft))
            if j >= sim_times.size or abs(float(sim_times[j]) - ft) > 1.0e-12:
                # If fixing time is not exactly on grid, keep deterministic coupon.
                ps = float(p0_fwd(max(0.0, float(s[i]))))
                pe = float(p0_fwd(float(e[i])))
                fwd = (ps / pe - 1.0) / float(tau[i])
                out[i, :] = fwd + float(spr[i])
                continue
            x_fix = x_paths_on_sim_grid[j, :]
            p_ft = float(p0_disc(ft))
            p_t_s_d = model.discount_bond(ft, float(s[i]), x_fix, p_ft, float(p0_disc(float(s[i]))))
            p_t_e_d = model.discount_bond(ft, float(e[i]), x_fix, p_ft, float(p0_disc(float(e[i]))))
            bt = float(p0_fwd(ft) / p0_disc(ft))
            bs = float(p0_fwd(float(s[i])) / p0_disc(float(s[i])))
            be = float(p0_fwd(float(e[i])) / p0_disc(float(e[i])))
            p_t_s_f = p_t_s_d * (bs / bt)
            p_t_e_f = p_t_e_d * (be / bt)
            fwd_path = (p_t_s_f / p_t_e_f - 1.0) / float(tau[i])
            coupon_path = fwd_path + float(spr[i])
            target_coupon = float(quoted_coupon[i])
            if abs(target_coupon) > 1.0e-14:
                p_fix_pay_d = model.discount_bond(ft, float(pay_t[i]), x_fix, p_ft, float(p0_disc(float(pay_t[i]))))
                numeraire = model.numeraire_lgm(ft, x_fix, p0_disc)
                current_mean = float(
                    np.mean(
                        float(sign[i]) * float(notional[i]) * float(tau[i]) * coupon_path * p_fix_pay_d / numeraire
                    )
                )
                target_mean = (
                    float(sign[i]) * float(notional[i]) * float(tau[i]) * target_coupon * float(p0_disc(float(pay_t[i])))
                )
                if abs(current_mean) > 1.0e-18:
                    coupon_path = coupon_path * (target_mean / current_mean)
            out[i, :] = coupon_path
        return out

    def _resolve_index_curve(
        self,
        inputs: _PythonLgmInputs,
        ccy: str,
        index_name: str,
    ) -> Callable[[float], float]:
        key = str(index_name).strip().upper()
        if key and key in inputs.forward_curves_by_name:
            return inputs.forward_curves_by_name[key]
        mapped_tenor = inputs.swap_index_forward_tenors.get(key, "")
        if mapped_tenor and mapped_tenor in inputs.forward_curves_by_tenor.get(ccy, {}):
            return inputs.forward_curves_by_tenor[ccy][mapped_tenor]
        tenor_match = re.search(r"(\d+[YMWD])$", key)
        if tenor_match:
            tenor = tenor_match.group(1).upper()
            if tenor in inputs.forward_curves_by_tenor.get(ccy, {}):
                return inputs.forward_curves_by_tenor[ccy][tenor]
        return inputs.forward_curves.get(ccy, inputs.discount_curves[ccy])

    def _par_swap_rate_paths(
        self,
        model: Any,
        curve: Callable[[float], float],
        t: float,
        x_t: np.ndarray,
        start: float,
        tenor_years: float,
    ) -> np.ndarray:
        effective_start = max(float(start), float(t))
        maturity = effective_start + max(float(tenor_years), 1.0e-8)
        pay_times = np.arange(effective_start + 1.0, maturity + 1.0e-10, 1.0)
        if pay_times.size == 0 or pay_times[-1] < maturity - 1.0e-10:
            pay_times = np.append(pay_times, maturity)
        p_t = float(curve(t))
        p_start = model.discount_bond(t, effective_start, x_t, p_t, float(curve(effective_start)))
        p_end = model.discount_bond(t, float(maturity), x_t, p_t, float(curve(float(maturity))))
        annuity = np.zeros_like(x_t, dtype=float)
        prev = effective_start
        for pay in pay_times:
            tau = max(float(pay) - prev, 1.0e-8)
            annuity += tau * model.discount_bond(t, float(pay), x_t, p_t, float(curve(float(pay))))
            prev = float(pay)
        annuity = np.where(np.abs(annuity) < 1.0e-12, 1.0e-12, annuity)
        return (p_start - p_end) / annuity

    def _interp_scalar_curve(self, points: Sequence[Tuple[float, float]], x: float) -> float:
        if not points:
            return 0.0
        pts = sorted((float(t), float(v)) for t, v in points)
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
            if t0 <= x <= t1:
                if abs(t1 - t0) < 1.0e-12:
                    return v1
                w = (x - t0) / (t1 - t0)
                return v0 + w * (v1 - v0)
        return pts[-1][1]

    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * np.square(x)) / math.sqrt(2.0 * math.pi)

    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

    def _normal_option_rate(self, forward: np.ndarray, strike: float, stddev: np.ndarray, *, is_call: bool) -> np.ndarray:
        stddev_safe = np.maximum(np.asarray(stddev, dtype=float), 1.0e-12)
        fwd = np.asarray(forward, dtype=float)
        k = float(strike)
        d = (fwd - k) / stddev_safe
        if is_call:
            value = (fwd - k) * self._normal_cdf(d) + stddev_safe * self._normal_pdf(d)
            intrinsic = np.maximum(fwd - k, 0.0)
        else:
            value = (k - fwd) * self._normal_cdf(-d) + stddev_safe * self._normal_pdf(d)
            intrinsic = np.maximum(k - fwd, 0.0)
        return np.where(np.asarray(stddev, dtype=float) <= 1.0e-12, intrinsic, value)

    def _cmsspread_vol_inputs(
        self,
        inputs: _PythonLgmInputs,
        ccy: str,
        idx1: str,
        idx2: str,
        fixing_time: float,
    ) -> Tuple[float, float, float]:
        tenor1 = re.search(r"(\d+[YMWD])$", idx1.upper())
        tenor2 = re.search(r"(\d+[YMWD])$", idx2.upper())
        vol1 = self._interp_scalar_curve(inputs.swaption_normal_vols.get((ccy, tenor1.group(1) if tenor1 else ""), []), fixing_time)
        vol2 = self._interp_scalar_curve(inputs.swaption_normal_vols.get((ccy, tenor2.group(1) if tenor2 else ""), []), fixing_time)
        corr = self._interp_scalar_curve(inputs.cms_correlations.get(tuple(sorted((idx1.upper(), idx2.upper()))), []), fixing_time)
        return float(vol1), float(vol2), float(max(min(corr, 0.9999), -0.9999))

    def _cmsspread_coupon_rate(
        self,
        inputs: _PythonLgmInputs,
        *,
        ccy: str,
        idx1: str,
        idx2: str,
        fixing_time: float,
        raw_coupon: np.ndarray,
        cap: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> np.ndarray:
        base = np.asarray(raw_coupon, dtype=float)
        if fixing_time <= 1.0e-12:
            return self._capped_floored_rate(base, cap=cap, floor=floor)
        vol1, vol2, corr = self._cmsspread_vol_inputs(inputs, ccy, idx1, idx2, fixing_time)
        spread_vol = math.sqrt(max(vol1 * vol1 + vol2 * vol2 - 2.0 * corr * vol1 * vol2, 0.0))
        stddev = np.full_like(base, spread_vol * math.sqrt(max(float(fixing_time), 0.0)))
        out = base.copy()
        if floor is not None:
            out = out + self._normal_option_rate(base, float(floor), stddev, is_call=False)
        if cap is not None:
            out = out - self._normal_option_rate(base, float(cap), stddev, is_call=True)
        return out

    def _static_ql_cms_rate(
        self,
        inputs: _PythonLgmInputs,
        *,
        asof: str,
        ccy: str,
        index_name: str,
        start_time: float,
        end_time: float,
        pay_time: float,
        fixing_days: int,
        day_counter_name: str,
        in_arrears: bool,
    ) -> Optional[float]:
        try:
            import QuantLib as ql
        except Exception:
            return None
        def _curve_handle(curve: Callable[[float], float]) -> Any:
            grid = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]
            dates = [eval_date]
            dfs = [1.0]
            for t in grid[1:]:
                dates.append(eval_date + int(round(365.25 * t)))
                dfs.append(max(float(curve(t)), 1.0e-10))
            return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))

        def _ql_day_counter(name: str) -> Any:
            norm = str(name).strip().upper()
            if norm in {"A360", "ACT/360"}:
                return ql.Actual360()
            if norm in {"30/360", "30E/360"}:
                return ql.Thirty360(ql.Thirty360.BondBasis)
            return ql.Actual365Fixed()
        tenor_match = re.search(r"(\d+[YMWD])$", index_name.upper())
        if ccy.upper() != "EUR" or tenor_match is None:
            return None
        tenor = tenor_match.group(1)
        vol_points = inputs.swaption_normal_vols.get((ccy.upper(), tenor), [])
        if not vol_points:
            return None
        eval_date = ql.DateParser.parseISO(_normalize_asof_date(asof))
        ql.Settings.instance().evaluationDate = eval_date
        def _ql_date_from_time(t: float) -> Any:
            return ql.DateParser.parseISO(_date_from_time(_normalize_asof_date(asof), float(t)))

        disc = _curve_handle(inputs.discount_curves[ccy.upper()])
        fwd = _curve_handle(self._resolve_index_curve(inputs, ccy.upper(), index_name))
        vol = max(self._interp_scalar_curve(vol_points, max(float(start_time), 1.0e-8)), 1.0e-8)
        vol_ts = ql.SwaptionVolatilityStructureHandle(
            ql.ConstantSwaptionVolatility(0, ql.TARGET(), ql.ModifiedFollowing, vol, ql.Actual365Fixed(), ql.Normal)
        )
        cms_pricer = ql.LinearTsrPricer(vol_ts, ql.QuoteHandle(ql.SimpleQuote(0.0)), disc)
        swap_index = ql.EuriborSwapIsdaFixA(ql.Period(tenor), fwd, disc)
        coupon = ql.CmsCoupon(
            _ql_date_from_time(pay_time),
            1.0,
            _ql_date_from_time(start_time),
            _ql_date_from_time(end_time),
            int(fixing_days),
            swap_index,
            1.0,
            0.0,
            _ql_date_from_time(start_time),
            _ql_date_from_time(end_time),
            _ql_day_counter(day_counter_name),
            bool(in_arrears),
        )
        coupon.setPricer(cms_pricer)
        return float(coupon.rate())

    def _static_ql_cmsspread_rate(
        self,
        inputs: _PythonLgmInputs,
        *,
        asof: str,
        ccy: str,
        idx1: str,
        idx2: str,
        start_time: float,
        end_time: float,
        pay_time: float,
        fixing_days: int,
        day_counter_name: str,
        in_arrears: bool,
        gearing: float,
        spread: float,
        cap: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> Optional[float]:
        try:
            import QuantLib as ql
        except Exception:
            return None

        def _curve_handle(curve: Callable[[float], float]) -> Any:
            grid = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]
            dates = [eval_date]
            dfs = [1.0]
            for t in grid[1:]:
                dates.append(eval_date + int(round(365.25 * t)))
                dfs.append(max(float(curve(t)), 1.0e-10))
            return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))

        def _ql_day_counter(name: str) -> Any:
            norm = str(name).strip().upper()
            if norm in {"A360", "ACT/360"}:
                return ql.Actual360()
            if norm in {"30/360", "30E/360"}:
                return ql.Thirty360(ql.Thirty360.BondBasis)
            return ql.Actual365Fixed()

        tenor1 = re.search(r"(\d+[YMWD])$", idx1.upper())
        tenor2 = re.search(r"(\d+[YMWD])$", idx2.upper())
        if ccy.upper() != "EUR" or tenor1 is None or tenor2 is None:
            return None
        vol1, vol2, corr = self._cmsspread_vol_inputs(inputs, ccy, idx1, idx2, max(float(start_time), 1.0e-8))
        avg_vol = max(0.5 * (abs(vol1) + abs(vol2)), 1.0e-8)
        eval_date = ql.DateParser.parseISO(_normalize_asof_date(asof))
        ql.Settings.instance().evaluationDate = eval_date

        def _ql_date_from_time(t: float) -> Any:
            return ql.DateParser.parseISO(_date_from_time(_normalize_asof_date(asof), float(t)))

        disc = _curve_handle(inputs.discount_curves[ccy.upper()])
        fwd1 = _curve_handle(self._resolve_index_curve(inputs, ccy.upper(), idx1))
        fwd2 = _curve_handle(self._resolve_index_curve(inputs, ccy.upper(), idx2))
        vol_ts = ql.SwaptionVolatilityStructureHandle(
            ql.ConstantSwaptionVolatility(0, ql.TARGET(), ql.ModifiedFollowing, avg_vol, ql.Actual365Fixed(), ql.Normal)
        )
        cms_pricer = ql.LinearTsrPricer(vol_ts, ql.QuoteHandle(ql.SimpleQuote(0.0)), disc)
        swap_index1 = ql.EuriborSwapIsdaFixA(ql.Period(tenor1.group(1)), fwd1, disc)
        swap_index2 = ql.EuriborSwapIsdaFixA(ql.Period(tenor2.group(1)), fwd2, disc)
        spread_index = ql.SwapSpreadIndex(f"{idx1}-{idx2}", swap_index1, swap_index2)
        spread_pricer = ql.LognormalCmsSpreadPricer(
            cms_pricer,
            ql.QuoteHandle(ql.SimpleQuote(float(corr))),
            disc,
            16,
        )
        coupon_args = (
            _ql_date_from_time(pay_time),
            1.0,
            _ql_date_from_time(start_time),
            _ql_date_from_time(end_time),
            int(fixing_days),
            spread_index,
            float(gearing),
            float(spread),
        )
        common_tail = (
            _ql_date_from_time(start_time),
            _ql_date_from_time(end_time),
            _ql_day_counter(day_counter_name),
            bool(in_arrears),
        )
        if cap is not None or floor is not None:
            coupon = ql.CappedFlooredCmsSpreadCoupon(
                *coupon_args,
                ql.nullDouble() if cap is None else float(cap),
                ql.nullDouble() if floor is None else float(floor),
                *common_tail,
            )
        else:
            coupon = ql.CmsSpreadCoupon(*coupon_args, *common_tail)
        coupon.setPricer(spread_pricer)
        return float(coupon.rate())

    def _capped_floored_rate(
        self,
        raw_rate: np.ndarray,
        cap: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> np.ndarray:
        out = np.asarray(raw_rate, dtype=float).copy()
        if floor is not None:
            out = np.maximum(out, float(floor))
        if cap is not None:
            out = np.minimum(out, float(cap))
        return out

    def _digital_option_rate(
        self,
        raw_rate: np.ndarray,
        strike: float,
        payoff: float,
        *,
        is_call: bool,
        long_short: float,
        fixed_mode: bool,
        atm_included: bool,
        capped_rate_fn: Optional[Callable[[float, float], np.ndarray]] = None,
    ) -> np.ndarray:
        if math.isnan(float(strike)):
            return np.zeros_like(raw_rate, dtype=float)
        strike_value = float(strike)
        eps = 1.0e-4
        if is_call and abs(strike_value) < eps / 2.0:
            strike_value = eps / 2.0

        if fixed_mode:
            if is_call:
                hit = raw_rate >= strike_value if atm_included else raw_rate > strike_value
            else:
                hit = raw_rate <= strike_value if atm_included else raw_rate < strike_value
            step = float(payoff) if not math.isnan(float(payoff)) else strike_value
            if math.isnan(float(payoff)):
                vanilla = np.maximum(raw_rate - strike_value, 0.0) if is_call else np.maximum(strike_value - raw_rate, 0.0)
                return float(long_short) * (step * hit.astype(float) + (vanilla if is_call else -vanilla))
            return float(long_short) * step * hit.astype(float)

        right = strike_value + eps / 2.0
        left = strike_value - eps / 2.0
        if capped_rate_fn is not None:
            next_rate = capped_rate_fn(right if is_call else math.nan, math.nan if is_call else right)
            prev_rate = capped_rate_fn(left if is_call else math.nan, math.nan if is_call else left)
        elif is_call:
            next_rate = self._capped_floored_rate(raw_rate, cap=right)
            prev_rate = self._capped_floored_rate(raw_rate, cap=left)
        else:
            next_rate = self._capped_floored_rate(raw_rate, floor=right)
            prev_rate = self._capped_floored_rate(raw_rate, floor=left)
        step = float(payoff) if not math.isnan(float(payoff)) else strike_value
        option_rate = step * (next_rate - prev_rate) / eps
        if math.isnan(float(payoff)):
            if capped_rate_fn is not None:
                at_strike = capped_rate_fn(strike_value if is_call else math.nan, math.nan if is_call else strike_value)
            else:
                at_strike = self._capped_floored_rate(raw_rate, cap=strike_value) if is_call else self._capped_floored_rate(raw_rate, floor=strike_value)
            vanilla = raw_rate - at_strike if is_call else -raw_rate + at_strike
            option_rate = option_rate + vanilla if is_call else option_rate - vanilla
        return float(long_short) * option_rate

    def _rate_leg_coupon_paths(
        self,
        model: Any,
        leg: Dict[str, object],
        ccy: str,
        inputs: _PythonLgmInputs,
        t: float,
        x_t: np.ndarray,
    ) -> np.ndarray:
        kind = str(leg.get("kind", "")).upper()
        start = np.asarray(leg.get("start_time", []), dtype=float)
        end = np.asarray(leg.get("end_time", []), dtype=float)
        fixing = np.asarray(leg.get("fixing_time", start), dtype=float)
        quoted = np.asarray(leg.get("quoted_coupon", np.zeros(start.shape)), dtype=float)
        fixed_mask = np.asarray(leg.get("is_historically_fixed", np.zeros(start.shape, dtype=bool)), dtype=bool)
        spread = np.asarray(leg.get("spread", np.zeros(start.shape)), dtype=float)
        gearing = np.asarray(leg.get("gearing", np.ones(start.shape)), dtype=float)
        coupons = np.zeros((start.size, x_t.size), dtype=float)
        for i in range(start.size):
            if fixed_mask[i] or fixing[i] <= t + 1.0e-12:
                base = np.full(x_t.shape, quoted[i], dtype=float)
            elif kind == "FLOATING":
                curve = self._resolve_index_curve(inputs, ccy, str(leg.get("index_name", "")))
                p_t = float(curve(t))
                effective_start = max(float(start[i]), float(t))
                p_s = model.discount_bond(t, effective_start, x_t, p_t, float(curve(effective_start)))
                p_e = model.discount_bond(t, float(end[i]), x_t, p_t, float(curve(float(end[i]))))
                tau = float(np.asarray(leg.get("index_accrual", leg["accrual"]), dtype=float)[i])
                base = (p_s / p_e - 1.0) / max(tau, 1.0e-8)
            elif kind == "CMS":
                index_name = str(leg.get("index_name", ""))
                curve = self._resolve_index_curve(inputs, ccy, index_name)
                tenor_match = re.search(r"(\d+[YMWD])$", index_name.upper())
                tenor_years = _parse_ore_tenor_to_years(tenor_match.group(1)) if tenor_match else 10.0
                ql_rate = None
                if t <= 1.0e-12:
                    ql_rate = self._static_ql_cms_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        index_name=index_name,
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=float(np.asarray(leg.get("pay_time", end), dtype=float)[i]),
                        fixing_days=int(leg.get("fixing_days", 2)),
                        day_counter_name=str(leg.get("day_counter", "A365")),
                        in_arrears=bool(leg.get("is_in_arrears", False)),
                    )
                base = np.full(x_t.shape, ql_rate, dtype=float) if ql_rate is not None else self._par_swap_rate_paths(model, curve, t, x_t, float(start[i]), tenor_years)
            elif kind in {"CMSSPREAD", "DIGITALCMSSPREAD"}:
                idx1 = str(leg.get("index_name_1", ""))
                idx2 = str(leg.get("index_name_2", ""))
                curve1 = self._resolve_index_curve(inputs, ccy, idx1)
                curve2 = self._resolve_index_curve(inputs, ccy, idx2)
                tenor1 = re.search(r"(\d+[YMWD])$", idx1.upper())
                tenor2 = re.search(r"(\d+[YMWD])$", idx2.upper())
                ql_rate1 = ql_rate2 = None
                if kind == "DIGITALCMSSPREAD" and t <= 1.0e-12:
                    pay_time = float(np.asarray(leg.get("pay_time", end), dtype=float)[i])
                    fixing_days = int(leg.get("fixing_days", 2))
                    dc_name = str(leg.get("day_counter", "A365"))
                    in_arrears = bool(leg.get("is_in_arrears", False))
                    ql_rate1 = self._static_ql_cms_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        index_name=idx1,
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=pay_time,
                        fixing_days=fixing_days,
                        day_counter_name=dc_name,
                        in_arrears=in_arrears,
                    )
                    ql_rate2 = self._static_ql_cms_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        index_name=idx2,
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=pay_time,
                        fixing_days=fixing_days,
                        day_counter_name=dc_name,
                        in_arrears=in_arrears,
                    )
                rate1 = np.full(x_t.shape, ql_rate1, dtype=float) if ql_rate1 is not None else self._par_swap_rate_paths(model, curve1, t, x_t, float(start[i]), _parse_ore_tenor_to_years(tenor1.group(1)) if tenor1 else 10.0)
                rate2 = np.full(x_t.shape, ql_rate2, dtype=float) if ql_rate2 is not None else self._par_swap_rate_paths(model, curve2, t, x_t, float(start[i]), _parse_ore_tenor_to_years(tenor2.group(1)) if tenor2 else 2.0)
                base = rate1 - rate2
            else:
                base = np.zeros_like(x_t, dtype=float)

            raw_coupon = gearing[i] * base + spread[i]
            coupon = raw_coupon
            if kind == "CMSSPREAD":
                cap = leg.get("cap")
                floor = leg.get("floor")
                ql_rate = None
                if not (fixed_mask[i] or fixing[i] <= t + 1.0e-12) and t <= 1.0e-12:
                    ql_rate = self._static_ql_cmsspread_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        idx1=str(leg.get("index_name_1", "")),
                        idx2=str(leg.get("index_name_2", "")),
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=float(np.asarray(leg.get("pay_time", end), dtype=float)[i]),
                        fixing_days=int(leg.get("fixing_days", 2)),
                        day_counter_name=str(leg.get("day_counter", "A365")),
                        in_arrears=bool(leg.get("is_in_arrears", False)),
                        gearing=float(gearing[i]),
                        spread=float(spread[i]),
                        cap=float(cap) if cap is not None else None,
                        floor=float(floor) if floor is not None else None,
                    )
                if ql_rate is not None:
                    coupon = np.full_like(raw_coupon, ql_rate, dtype=float)
                elif fixed_mask[i] or fixing[i] <= t + 1.0e-12:
                    coupon = self._capped_floored_rate(raw_coupon, cap=float(cap) if cap is not None else None, floor=float(floor) if floor is not None else None)
                else:
                    coupon = self._cmsspread_coupon_rate(
                        inputs,
                        ccy=ccy,
                        idx1=str(leg.get("index_name_1", "")),
                        idx2=str(leg.get("index_name_2", "")),
                        fixing_time=float(fixing[i]),
                        raw_coupon=raw_coupon,
                        cap=float(cap) if cap is not None else None,
                        floor=float(floor) if floor is not None else None,
                    )
            elif kind == "DIGITALCMSSPREAD":
                fixed_mode = bool(fixed_mask[i] or fixing[i] <= t + 1.0e-12)
                ql_raw = None
                ql_capped_rate_fn = None
                if not fixed_mode and t <= 1.0e-12:
                    idx1 = str(leg.get("index_name_1", ""))
                    idx2 = str(leg.get("index_name_2", ""))
                    pay_time = float(np.asarray(leg.get("pay_time", end), dtype=float)[i])
                    fixing_days = int(leg.get("fixing_days", 2))
                    dc_name = str(leg.get("day_counter", "A365"))
                    in_arrears = bool(leg.get("is_in_arrears", False))
                    ql_raw = self._static_ql_cmsspread_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        idx1=idx1,
                        idx2=idx2,
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=pay_time,
                        fixing_days=fixing_days,
                        day_counter_name=dc_name,
                        in_arrears=in_arrears,
                        gearing=float(gearing[i]),
                        spread=float(spread[i]),
                    )
                    ql_capped_rate_fn = lambda cap, floor, idx1_=idx1, idx2_=idx2, pt=pay_time, fd=fixing_days, dc_=dc_name, ia=in_arrears: np.full_like(
                        raw_coupon,
                        self._static_ql_cmsspread_rate(
                            inputs,
                            asof=inputs.asof,
                            ccy=ccy,
                            idx1=idx1_,
                            idx2=idx2_,
                            start_time=float(start[i]),
                            end_time=float(end[i]),
                            pay_time=pt,
                            fixing_days=fd,
                            day_counter_name=dc_,
                            in_arrears=ia,
                            gearing=float(gearing[i]),
                            spread=float(spread[i]),
                            cap=None if math.isnan(cap) else float(cap),
                            floor=None if math.isnan(floor) else float(floor),
                        ),
                        dtype=float,
                    )
                if bool(leg.get("naked_option", False)):
                    coupon = np.zeros_like(raw_coupon, dtype=float)
                elif ql_raw is not None:
                    coupon = np.full_like(raw_coupon, ql_raw, dtype=float)
                capped_rate_fn = None
                if ql_capped_rate_fn is not None:
                    capped_rate_fn = ql_capped_rate_fn
                elif not fixed_mode:
                    idx1 = str(leg.get("index_name_1", ""))
                    idx2 = str(leg.get("index_name_2", ""))
                    fixing_time = float(fixing[i])
                    capped_rate_fn = lambda cap, floor, raw=raw_coupon, ccy_=ccy, idx1_=idx1, idx2_=idx2, ft=fixing_time: self._cmsspread_coupon_rate(
                        inputs,
                        ccy=ccy_,
                        idx1=idx1_,
                        idx2=idx2_,
                        fixing_time=ft,
                        raw_coupon=raw,
                        cap=None if math.isnan(cap) else float(cap),
                        floor=None if math.isnan(floor) else float(floor),
                    )
                coupon = coupon + self._digital_option_rate(
                    raw_coupon,
                    float(leg.get("call_strike", float("nan"))),
                    float(leg.get("call_payoff", float("nan"))),
                    is_call=True,
                    long_short=float(leg.get("call_position", 1.0)),
                    fixed_mode=fixed_mode,
                    atm_included=bool(leg.get("is_call_atm_included", False)),
                    capped_rate_fn=capped_rate_fn,
                )
                coupon = coupon + self._digital_option_rate(
                    raw_coupon,
                    float(leg.get("put_strike", float("nan"))),
                    float(leg.get("put_payoff", float("nan"))),
                    is_call=False,
                    long_short=float(leg.get("put_position", 1.0)),
                    fixed_mode=fixed_mode,
                    atm_included=bool(leg.get("is_put_atm_included", False)),
                    capped_rate_fn=capped_rate_fn,
                )
            coupons[i, :] = coupon
        return coupons

    def _price_generic_rate_swap(
        self,
        spec: _TradeSpec,
        inputs: _PythonLgmInputs,
        model: Any,
        x_paths: np.ndarray,
    ) -> np.ndarray:
        legs_payload = spec.legs or {}
        rate_legs = list(legs_payload.get("rate_legs", []))
        n_times = int(inputs.times.size)
        n_paths = int(x_paths.shape[1])
        p_disc = inputs.discount_curves[spec.ccy]
        vals = np.zeros((n_times, n_paths), dtype=float)
        frozen_coupon_paths: Dict[int, np.ndarray] = {}
        for leg_idx, leg in enumerate(rate_legs):
            kind = str(leg.get("kind", "")).upper()
            if kind not in {"CMS", "CMSSPREAD", "DIGITALCMSSPREAD"}:
                continue
            frozen_coupon_paths[leg_idx] = self._rate_leg_coupon_paths(
                model,
                leg,
                spec.ccy,
                inputs,
                0.0,
                x_paths[0, :],
            )
        for i, t in enumerate(inputs.times):
            x_t = x_paths[i, :]
            p_t = float(p_disc(float(t)))
            pv = np.zeros((n_paths,), dtype=float)
            for leg_idx, leg in enumerate(rate_legs):
                kind = str(leg.get("kind", "")).upper()
                pay = np.asarray(leg.get("pay_time", []), dtype=float)
                start = np.asarray(leg.get("start_time", []), dtype=float)
                accr = np.asarray(leg.get("accrual", []), dtype=float)
                live = pay > float(t) + 1.0e-12
                if not np.any(live):
                    continue
                disc = model.discount_bond_paths(float(t), pay[live], x_t, p_t, np.asarray([float(p_disc(float(T))) for T in pay[live]], dtype=float))
                if kind == "FIXED":
                    amount = np.asarray(leg.get("amount", np.zeros(pay.shape)), dtype=float)[live]
                    pv += np.sum(amount[:, None] * disc, axis=0)
                    continue
                if float(t) > 1.0e-12 and leg_idx in frozen_coupon_paths:
                    coupons = frozen_coupon_paths[leg_idx][live, :]
                else:
                    coupons = self._rate_leg_coupon_paths(model, leg, spec.ccy, inputs, float(t), x_t)[live, :]
                notional = float(leg.get("notional", 0.0))
                sign = float(leg.get("sign", 1.0))
                amount = sign * notional * accr[live, None] * coupons
                pv += np.sum(amount * disc, axis=0)
            vals[i, :] = pv
        return vals

    def _assemble_result(
        self,
        run_id: str,
        snapshot: XVASnapshot,
        inputs: _PythonLgmInputs,
        model: Any,
        x_paths: np.ndarray,
        npv_by_trade: Dict[str, np.ndarray],
        fallback: XVAResult | None,
        fallback_trades: List[Trade],
        unsupported: List[Trade],
    ) -> XVAResult:
        times = inputs.times
        valuation_times = inputs.valuation_times
        obs_times = inputs.observation_times
        obs_closeout_times = inputs.observation_closeout_times
        obs_dates = [
            (datetime.fromisoformat(inputs.asof).date() + timedelta(days=int(round(float(t) * 365.0)))).isoformat()
            for t in obs_times
        ]
        valuation_idx = np.searchsorted(times, valuation_times)
        obs_idx = np.searchsorted(times, obs_times)
        obs_closeout_idx = np.searchsorted(times, obs_closeout_times)
        if (
            obs_idx.size != obs_times.size
            or np.any(obs_idx >= times.size)
            or np.any(np.abs(times[obs_idx] - obs_times) > 1.0e-10)
        ):
            raise EngineRunError("Observation times are not aligned with pricing grid after augmentation")
        if (
            valuation_idx.size != valuation_times.size
            or np.any(valuation_idx >= times.size)
            or np.any(np.abs(times[valuation_idx] - valuation_times) > 1.0e-10)
        ):
            raise EngineRunError("Valuation times are not aligned with pricing grid after augmentation")
        if (
            obs_closeout_idx.size != obs_closeout_times.size
            or np.any(obs_closeout_idx >= times.size)
            or np.any(np.abs(times[obs_closeout_idx] - obs_closeout_times) > 1.0e-10)
        ):
            raise EngineRunError("Sticky closeout times are not aligned with pricing grid after augmentation")
        df_base = inputs.discount_curves[snapshot.config.base_currency.upper()]
        df_vec = np.asarray([df_base(float(t)) for t in times], dtype=float)
        obs_df_vec = df_vec[obs_idx]
        ns_valuation_paths: Dict[str, np.ndarray] = {}
        ns_closeout_paths: Dict[str, np.ndarray] = {}
        xva_deflated_by_ns: Dict[str, bool] = {}
        pv_total = 0.0
        npv_cube_payload: Dict[str, Dict[str, object]] = {}
        exposure_cube_payload: Dict[str, Dict[str, object]] = {}
        exposure_profiles_by_trade: Dict[str, Dict[str, object]] = {}
        exposure_profiles_by_netting_set: Dict[str, Dict[str, object]] = {}
        pfe_quantile = _pfe_quantile(snapshot)

        for spec in inputs.trade_specs:
            v = npv_by_trade.get(spec.trade.trade_id)
            if v is None:
                continue
            pv_total += float(np.mean(v[valuation_idx[0], :]))
            if spec.kind == "IRS":
                p_disc = inputs.discount_curves[spec.ccy]
                v_xva = self._irs_utils.deflate_lgm_npv_paths(
                    model=model,
                    p0_disc=p_disc,
                    times=times,
                    x_paths=x_paths,
                    npv_paths=v,
                )
                xva_deflated = True
            else:
                v_xva = v
                xva_deflated = False
            ns = spec.trade.netting_set
            v_xva_val = v_xva[obs_idx, :]
            if spec.kind == "FXForward":
                v_xva_obs = self._fx_forward_closeout_paths(
                    spec.trade,
                    v_xva,
                    times,
                    obs_times,
                    obs_closeout_times,
                )
            else:
                v_xva_obs = v_xva[obs_closeout_idx, :]
            valuation_epe = np.mean(np.maximum(v_xva_val, 0.0), axis=1)
            valuation_ene = np.mean(np.maximum(-v_xva_val, 0.0), axis=1)
            epe = np.mean(np.maximum(v_xva_obs, 0.0), axis=1)
            ene = np.mean(np.maximum(-v_xva_obs, 0.0), axis=1)
            pfe = ore_pfe_quantile(v_xva_obs, pfe_quantile)
            trade_profile = build_ore_exposure_profile_from_paths(
                spec.trade.trade_id,
                obs_dates,
                obs_times.tolist(),
                v_xva_val,
                v_xva_obs,
                discount_factors=obs_df_vec.tolist(),
                closeout_times=obs_closeout_times.tolist(),
                pfe_quantile=pfe_quantile,
                asof_date=inputs.asof,
            )
            ns_valuation_paths[ns] = ns_valuation_paths.get(ns, np.zeros_like(v_xva_val)) + v_xva_val
            ns_closeout_paths[ns] = ns_closeout_paths.get(ns, np.zeros_like(v_xva_obs)) + v_xva_obs
            xva_deflated_by_ns[ns] = xva_deflated_by_ns.get(ns, True) and xva_deflated
            npv_cube_payload[spec.trade.trade_id] = {
                "times": valuation_times.tolist(),
                "npv_mean": np.mean(v[valuation_idx, :], axis=1).tolist(),
                "npv_xva_mean": np.mean(v_xva[valuation_idx, :], axis=1).tolist(),
                "closeout_times": _build_sticky_closeout_times(valuation_times, inputs.mpor.mpor_years).tolist(),
                "closeout_npv_mean": np.mean(v[np.searchsorted(times, _build_sticky_closeout_times(valuation_times, inputs.mpor.mpor_years)), :], axis=1).tolist(),
                "valuation_epe": valuation_epe.tolist(),
                "valuation_ene": valuation_ene.tolist(),
                "closeout_epe": epe.tolist(),
                "closeout_ene": ene.tolist(),
                "pfe": pfe.tolist(),
                "basel_ee": trade_profile["basel_ee"],
                "basel_eee": trade_profile["basel_eee"],
                "time_weighted_basel_epe": trade_profile["time_weighted_basel_epe"],
                "time_weighted_basel_eepe": trade_profile["time_weighted_basel_eepe"],
            }
            exposure_profiles_by_trade[spec.trade.trade_id] = trade_profile

        epe_by_ns_paths: Dict[str, np.ndarray] = {}
        ene_by_ns_paths: Dict[str, np.ndarray] = {}
        valuation_epe_by_ns_paths: Dict[str, np.ndarray] = {}
        valuation_ene_by_ns_paths: Dict[str, np.ndarray] = {}
        for ns, valuation_paths in ns_valuation_paths.items():
            closeout_paths = ns_closeout_paths[ns]
            collateral_paths = _estimate_vm_collateral_paths(snapshot, ns, valuation_paths)
            valuation_paths_net = valuation_paths - collateral_paths
            closeout_paths_net = closeout_paths - collateral_paths
            valuation_epe_by_ns_paths[ns] = np.mean(np.maximum(valuation_paths_net, 0.0), axis=1)
            valuation_ene_by_ns_paths[ns] = np.mean(np.maximum(-valuation_paths_net, 0.0), axis=1)
            epe_by_ns_paths[ns] = np.mean(np.maximum(closeout_paths_net, 0.0), axis=1)
            ene_by_ns_paths[ns] = np.mean(np.maximum(-closeout_paths_net, 0.0), axis=1)
            exposure_profiles_by_netting_set[ns] = build_ore_exposure_profile_from_paths(
                ns,
                obs_dates,
                obs_times.tolist(),
                valuation_paths_net,
                closeout_paths_net,
                discount_factors=obs_df_vec.tolist(),
                closeout_times=obs_closeout_times.tolist(),
                expected_collateral=np.mean(collateral_paths, axis=1).tolist(),
                pfe_quantile=pfe_quantile,
                asof_date=inputs.asof,
            )

        exposure_by_ns = {ns: float(np.max(vals)) for ns, vals in epe_by_ns_paths.items()}
        xva_by_metric: Dict[str, float] = {}
        metric_set = set(snapshot.config.analytics)

        cva_total = 0.0
        dva_total = 0.0
        fba_total = 0.0
        fca_total = 0.0
        mva_total = 0.0
        own_name = _own_name_from_runtime(snapshot).upper()
        borrow_spread, lend_spread = _effective_funding_spreads(snapshot, inputs, own_name)
        funding_ois_curve = inputs.xva_discount_curve or df_base
        funding_ois_df = np.asarray([funding_ois_curve(float(t)) for t in times], dtype=float)
        obs_funding_ois_df = funding_ois_df[obs_idx]
        funding_borrow_df = (
            np.asarray([inputs.funding_borrow_curve(float(t)) for t in times], dtype=float)
            if inputs.funding_borrow_curve is not None
            else None
        )
        obs_funding_borrow_df = funding_borrow_df[obs_idx] if funding_borrow_df is not None else None
        funding_lend_df = (
            np.asarray([inputs.funding_lend_curve(float(t)) for t in times], dtype=float)
            if inputs.funding_lend_curve is not None
            else None
        )
        obs_funding_lend_df = funding_lend_df[obs_idx] if funding_lend_df is not None else None
        for ns, epe_vec in epe_by_ns_paths.items():
            ene_vec = ene_by_ns_paths.get(ns, np.zeros_like(epe_vec))
            cpty = _counterparty_for_netting(snapshot, ns)
            discount_weight = np.ones_like(obs_df_vec) if xva_deflated_by_ns.get(ns, False) else obs_df_vec
            cpty_curve = inputs.survival_curves.get(cpty)
            if cpty_curve is not None:
                q_c = np.asarray([cpty_curve(float(t)) for t in obs_times], dtype=float)
            else:
                q_c = self._irs_utils.survival_probability_from_hazard(
                    obs_times,
                    inputs.hazard_times.get(cpty, np.array([1.0, 5.0])),
                    inputs.hazard_rates.get(cpty, np.array([0.02, 0.02])),
                )
            own_curve = inputs.survival_curves.get(own_name)
            if own_curve is not None:
                own_q = np.asarray([own_curve(float(t)) for t in obs_times], dtype=float)
            else:
                own_q = self._irs_utils.survival_probability_from_hazard(
                    obs_times,
                    inputs.hazard_times.get(own_name, np.array([1.0, 5.0])),
                    inputs.hazard_rates.get(own_name, np.array([0.015, 0.015])),
                )
            dpd_c = np.clip(np.concatenate(([0.0], q_c[:-1] - q_c[1:])), 0.0, None)
            dpd_b = np.clip(np.concatenate(([0.0], own_q[:-1] - own_q[1:])), 0.0, None)
            lgd = 1.0 - float(inputs.recovery_rates.get(cpty, 0.4))
            lgd_own = 1.0 - float(inputs.recovery_rates.get(own_name, 0.4))
            cva_total += float(np.sum(discount_weight * epe_vec * dpd_c * lgd))
            dva_total += float(np.sum(discount_weight * ene_vec * dpd_b * lgd_own))
            dt = np.diff(obs_times, prepend=0.0)
            if obs_funding_borrow_df is not None and obs_funding_lend_df is not None:
                surv_joint = q_c[:-1] * own_q[:-1]
                dcf_borrow = obs_funding_borrow_df[:-1] / obs_funding_borrow_df[1:] - obs_funding_ois_df[:-1] / obs_funding_ois_df[1:]
                dcf_lend = obs_funding_lend_df[:-1] / obs_funding_lend_df[1:] - obs_funding_ois_df[:-1] / obs_funding_ois_df[1:]
                fca_total += float(np.sum(surv_joint * epe_vec[1:] * dcf_borrow))
                fba_total += float(np.sum(surv_joint * ene_vec[1:] * dcf_lend))
            else:
                # Fallback spread-only approximation when explicit funding curves are absent.
                fba_total += float(np.sum(obs_df_vec * ene_vec * dt * lend_spread))
                fca_total += float(np.sum(obs_df_vec * epe_vec * dt * borrow_spread))
            # Use a liability-side IM funding proxy with negative sign to align
            # with ORE report convention direction.
            mva_total += float(-np.sum(obs_df_vec * ene_vec * dt * 0.00012))
            exposure_cube_payload[ns] = {
                "times": obs_times.tolist(),
                "closeout_times": obs_closeout_times.tolist(),
                "valuation_epe": valuation_epe_by_ns_paths.get(ns, np.zeros_like(epe_vec)).tolist(),
                "valuation_ene": valuation_ene_by_ns_paths.get(ns, np.zeros_like(ene_vec)).tolist(),
                "closeout_epe": epe_vec.tolist(),
                "closeout_ene": ene_vec.tolist(),
                "pfe": exposure_profiles_by_netting_set.get(ns, {}).get("pfe", []),
                "basel_ee": exposure_profiles_by_netting_set.get(ns, {}).get("basel_ee", []),
                "basel_eee": exposure_profiles_by_netting_set.get(ns, {}).get("basel_eee", []),
                "time_weighted_basel_epe": exposure_profiles_by_netting_set.get(ns, {}).get("time_weighted_basel_epe", []),
                "time_weighted_basel_eepe": exposure_profiles_by_netting_set.get(ns, {}).get("time_weighted_basel_eepe", []),
            }

        if "CVA" in metric_set:
            xva_by_metric["CVA"] = cva_total
        if "DVA" in metric_set:
            # ORE reports DVA as a positive adjustment component.
            xva_by_metric["DVA"] = dva_total
        if "FVA" in metric_set:
            xva_by_metric["FVA"] = fba_total + fca_total
            xva_by_metric["FBA"] = fba_total
            xva_by_metric["FCA"] = fca_total
        if "MVA" in metric_set:
            xva_by_metric["MVA"] = mva_total

        exposure_report_rows = []
        for ns, peak_epe in exposure_by_ns.items():
            profile = exposure_profiles_by_netting_set.get(ns, {})
            pfe_series = np.asarray(profile.get("pfe", []), dtype=float)
            closeout_epe_series = np.asarray(profile.get("closeout_epe", []), dtype=float)
            peak_idx = int(np.argmax(closeout_epe_series)) if closeout_epe_series.size else 0
            peak_pfe = float(pfe_series[peak_idx]) if pfe_series.size > peak_idx else 0.0
            exposure_report_rows.append(
                {
                    "NettingSetId": ns,
                    "EPE": peak_epe,
                    "PFE": peak_pfe,
                    "BaselEPE": one_year_profile_value(profile, "time_weighted_basel_epe", asof_date=inputs.asof),
                    "BaselEEPE": one_year_profile_value(profile, "time_weighted_basel_eepe", asof_date=inputs.asof),
                }
            )
        reports = {
            "xva": [{"Metric": k, "Value": v} for k, v in xva_by_metric.items()],
            "exposure": exposure_report_rows,
        }
        cubes = {
            "npv_cube": CubeAccessor(name="npv_cube", payload=npv_cube_payload),
            "exposure_cube": CubeAccessor(name="exposure_cube", payload=exposure_cube_payload),
        }

        if fallback is not None:
            pv_total += fallback.pv_total
            for k, v in fallback.exposure_by_netting_set.items():
                exposure_by_ns[k] = exposure_by_ns.get(k, 0.0) + float(v)
            for k, v in fallback.xva_by_metric.items():
                xva_by_metric[k] = xva_by_metric.get(k, 0.0) + float(v)
            reports.update(fallback.reports)
            cubes.update(fallback.cubes)

        total_notional = sum(_trade_notional(t) for t in snapshot.portfolio.trades) or 1.0
        py_notional = sum(_trade_notional(s.trade) for s in inputs.trade_specs if s.trade.trade_id in npv_by_trade)
        fallback_notional = sum(_trade_notional(t) for t in fallback_trades)

        metadata = {
            "engine": "python-lgm",
            "fallback_mode": "hybrid" if self.fallback_to_swig else "native_only",
            "using_fallback_swig": fallback is not None,
            "coverage": {
                "python_trades": len(npv_by_trade),
                "fallback_trades": len(fallback_trades),
                "unsupported": [f"{t.trade_id}:{t.trade_type}" for t in unsupported],
                "requires_swig": [f"{t.trade_id}:{t.trade_type}" for t in fallback_trades],
                "python_notional_pct": float(py_notional / total_notional),
                "fallback_notional_pct": float(fallback_notional / total_notional),
            },
            "input_provenance": dict(inputs.input_provenance),
            "input_fallbacks": list(inputs.input_fallbacks),
            "using_fallback_inputs": bool(inputs.input_fallbacks),
            "model_currency": inputs.model_ccy,
            "grid_size": int(times.size),
            "valuation_grid_size": int(valuation_times.size),
            "observation_grid_size": int(obs_times.size),
            "closeout_grid_size": int(np.unique(inputs.observation_closeout_times).size),
            "path_count": int(snapshot.config.num_paths),
            "pfe_quantile": float(pfe_quantile),
            "mpor_enabled": bool(inputs.mpor.enabled),
            "mpor_days": int(inputs.mpor.mpor_days),
            "mpor_mode": "sticky",
            "mpor_source": inputs.mpor.source,
        }
        cmsspread_trades = [
            spec.trade.trade_id
            for spec in inputs.trade_specs
            if spec.kind == "RateSwap"
            and spec.legs is not None
            and any(
                str(leg.get("kind", "")).upper() in {"CMS", "CMSSPREAD", "DIGITALCMSSPREAD"}
                for leg in spec.legs.get("rate_legs", [])
            )
        ]
        if cmsspread_trades:
            metadata["cmsspread_profile_mode"] = "frozen_input_native"
            metadata["cmsspread_profile_trades"] = cmsspread_trades
        if fallback is not None:
            metadata["fallback_adapter"] = "ore-swig"

        return XVAResult(
            run_id=run_id,
            pv_total=pv_total,
            xva_total=float(sum(xva_by_metric.values())),
            xva_by_metric=xva_by_metric,
            exposure_by_netting_set=exposure_by_ns,
            exposure_profiles_by_netting_set=exposure_profiles_by_netting_set,
            exposure_profiles_by_trade=exposure_profiles_by_trade,
            reports=reports,
            cubes=cubes,
            metadata=metadata,
        )


class ORESwigAdapter:
    """Optional adapter using ORE-SWIG when available in the environment."""

    def __init__(self, module: Optional[Any] = None):
        self._module = module or self._discover_module()
        self._input_parameters_cls = self._resolve_attr("InputParameters")
        self._ore_app_cls = self._resolve_attr("OREApp")

    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        ip = self._construct_input_parameters()
        build_input_parameters(snapshot, ip)

        app = self._construct_app(ip)
        self._invoke_run(app, mapped.market_data_lines, mapped.fixing_data_lines)

        reports = self._extract_reports(app)
        cubes = self._extract_cubes(app)
        dim_reports, dim_report_source = self._extract_dim_reports(snapshot, reports)
        reports.update(dim_reports)
        xva_by_metric = self._extract_xva_metrics(reports)
        exposure_by_netting_set = self._extract_exposures(reports)
        xva_total = sum(xva_by_metric.values())
        pv_total = self._extract_pv_total(reports)

        return XVAResult(
            run_id=run_id,
            pv_total=pv_total,
            xva_total=xva_total,
            xva_by_metric=xva_by_metric,
            exposure_by_netting_set=exposure_by_netting_set,
            reports=reports,
            cubes=cubes,
            metadata={
                "adapter": "ore-swig",
                "module": getattr(self._module, "__name__", type(self._module).__name__),
                "dim_mode": _active_dim_mode(snapshot),
                "dim_report_source": dim_report_source,
            },
        )

    def _discover_module(self) -> Any:
        errors = []
        for mod_name in ("ORE", "oreanalytics", "OREAnalytics"):
            try:
                module = importlib.import_module(mod_name)
                if self._contains_attr(module, "InputParameters") and self._contains_attr(module, "OREApp"):
                    return module
            except Exception as exc:
                errors.append(f"{mod_name}: {exc}")
        raise EngineRunError(
            "Could not load ORE-SWIG module with InputParameters/OREApp. "
            + "; ".join(errors)
        )

    def _contains_attr(self, obj: Any, name: str) -> bool:
        if hasattr(obj, name):
            return True
        nested = getattr(obj, "_ORE", None)
        return nested is not None and hasattr(nested, name)

    def _resolve_attr(self, name: str) -> Any:
        if hasattr(self._module, name):
            return getattr(self._module, name)
        nested = getattr(self._module, "_ORE", None)
        if nested is not None and hasattr(nested, name):
            return getattr(nested, name)
        raise EngineRunError(f"Missing required ORE-SWIG class: {name}")

    def _construct_input_parameters(self) -> Any:
        try:
            return self._input_parameters_cls()
        except Exception as exc:
            raise EngineRunError(f"Failed to construct InputParameters: {exc}") from exc

    def _construct_app(self, input_parameters: Any) -> Any:
        # The 5-arg form matches the OREApp(inputs, log_file, log_mask, console, file_log)
        # signature used by ORE >= 6.x.  Fall back to the 1-arg form for older builds.
        _ORE6_ARGS = ("", 31, False, True)  # log_file, log_mask, console_log, file_log
        candidates = [
            (input_parameters, *_ORE6_ARGS),
            (input_parameters,),
        ]
        for args in candidates:
            try:
                return self._ore_app_cls(*args)
            except Exception:
                continue
        raise EngineRunError("Failed to construct OREApp from InputParameters")

    def _invoke_run(self, app: Any, market_data: Sequence[str], fixing_data: Sequence[str]) -> None:
        import inspect as _inspect
        try:
            sig = _inspect.signature(app.run)
            sig.bind(list(market_data), list(fixing_data))
            accepts_args = True
        except TypeError:
            accepts_args = False

        if accepts_args:
            try:
                app.run(list(market_data), list(fixing_data))
                return
            except Exception as exc:
                raise EngineRunError(f"OREApp.run(marketData, fixingData) failed: {exc}") from exc

        try:
            app.run()
            return
        except Exception as exc:
            raise EngineRunError(f"OREApp.run() failed: {exc}") from exc

    def _extract_reports(self, app: Any) -> Dict[str, Any]:
        reports: Dict[str, Any] = {}
        if not hasattr(app, "getReportNames") or not hasattr(app, "getReport"):
            return reports
        try:
            for name in app.getReportNames():
                reports[str(name)] = app.getReport(str(name))
        except Exception:
            return reports
        return reports

    def _extract_cubes(self, app: Any) -> Dict[str, CubeAccessor]:
        cubes: Dict[str, CubeAccessor] = {}
        if hasattr(app, "getCubeNames") and hasattr(app, "getCube"):
            try:
                for name in app.getCubeNames():
                    cubes[str(name)] = CubeAccessor(name=str(name), payload={"raw": app.getCube(str(name))})
            except Exception:
                pass
        if hasattr(app, "getMarketCubeNames") and hasattr(app, "getMarketCube"):
            try:
                for name in app.getMarketCubeNames():
                    key = f"market::{name}"
                    cubes[key] = CubeAccessor(name=key, payload={"raw": app.getMarketCube(str(name))})
            except Exception:
                pass
        return cubes

    def _extract_dim_reports(self, snapshot: XVASnapshot, reports: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        dim_reports: Dict[str, Any] = {}
        report_source = "absent"

        report_aliases = {
            "dim_evolution": ("dim_evolution", "dimevolution", "dimEvolution"),
            "dim_regression": ("dim_regression", "dimregression", "dimRegression"),
        }
        for target, aliases in report_aliases.items():
            for alias in aliases:
                rows = self._rows_from_report(reports.get(alias))
                if rows:
                    dim_reports[target] = rows
                    report_source = "report"
                    break

        if dim_reports:
            return dim_reports, report_source

        file_reports = self._read_dim_reports_from_files(snapshot)
        if file_reports:
            return file_reports, "file"

        return {}, report_source

    def _read_dim_reports_from_files(self, snapshot: XVASnapshot) -> Dict[str, Any]:
        runtime = snapshot.config.runtime
        if runtime is None:
            return {}

        xva = runtime.xva_analytic
        configured = {
            "dim_evolution": xva.dim_evolution_file,
            "dim_regression": xva.dim_regression_files,
        }
        output_dir = _configured_output_dir(snapshot)
        if output_dir is None:
            return {}

        out: Dict[str, Any] = {}
        for key, rel_name in configured.items():
            if not rel_name:
                continue
            path = Path(rel_name)
            if not path.is_absolute():
                path = output_dir / path
            rows = _read_csv_report(path)
            if rows:
                out[key] = rows
        return out

    def _extract_xva_metrics(self, reports: Dict[str, Any]) -> Dict[str, float]:
        xva: Dict[str, float] = {}
        rows = self._rows_from_report(reports.get("xva"))
        if not rows:
            return xva

        # Case 1: already in metric/value format.
        for row in rows:
            metric = row.get("Metric") or row.get("metric") or row.get("Type")
            value = row.get("Value") or row.get("value")
            if metric in ("CVA", "DVA", "FVA", "MVA") and value is not None:
                xva[str(metric)] = float(value)
        if xva:
            return xva

        # Case 2: ORE xva report with one column per metric and rows by netting/trade.
        metric_cols = ("CVA", "DVA", "MVA", "FBA", "FCA")
        aggregates = {m: 0.0 for m in metric_cols}
        has_metric = {m: False for m in metric_cols}
        for row in rows:
            trade_id = str(row.get("TradeId", "")).strip()
            # Use summary rows where TradeId is empty, fallback to all rows.
            include = trade_id == ""
            for m in metric_cols:
                if m in row and row[m] is not None:
                    try:
                        v = float(row[m])
                    except Exception:
                        continue
                    if include:
                        aggregates[m] += v
                        has_metric[m] = True
        if not any(has_metric.values()):
            for row in rows:
                for m in metric_cols:
                    if m in row and row[m] is not None:
                        try:
                            aggregates[m] += float(row[m])
                            has_metric[m] = True
                        except Exception:
                            pass

        if has_metric["CVA"]:
            xva["CVA"] = aggregates["CVA"]
        if has_metric["DVA"]:
            xva["DVA"] = aggregates["DVA"]
        if has_metric["MVA"]:
            xva["MVA"] = aggregates["MVA"]
        if has_metric["FBA"] or has_metric["FCA"]:
            if has_metric["FBA"]:
                xva["FBA"] = aggregates["FBA"]
            if has_metric["FCA"]:
                xva["FCA"] = aggregates["FCA"]
            # Approximate combined funding adjustment from borrowing/lending components.
            xva["FVA"] = aggregates["FBA"] + aggregates["FCA"]
        return xva

    def _extract_exposures(self, reports: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        exposure_reports = [name for name in reports if str(name).lower().startswith("exposure")]
        if "exposure" not in exposure_reports and "exposure" in reports:
            exposure_reports.append("exposure")

        for report_name in exposure_reports:
            for row in self._rows_from_report(reports.get(report_name)):
                ns = row.get("NettingSetId") or row.get("nettingsetid")
                epe = self._first_present(row, ("EPE", "BaselEEPE", "ExpectedPositiveExposure", "expectedpositiveexposure"))
                if ns is not None and epe is not None:
                    out[str(ns)] = out.get(str(ns), 0.0) + epe
        if out:
            return out

        # Fallback to xva report columns when explicit exposure report is absent.
        rows = self._rows_from_report(reports.get("xva"))
        for row in rows:
            ns = row.get("NettingSetId")
            trade_id = str(row.get("TradeId", "")).strip()
            epe = row.get("BaselEEPE")
            if ns is not None and epe is not None and trade_id == "":
                v = self._to_float(epe)
                if v is not None:
                    out[str(ns)] = v
        return out

    def _extract_pv_total(self, reports: Dict[str, Any]) -> float:
        rows = self._rows_from_report(reports.get("npv"))
        total = 0.0
        for row in rows:
            value = self._first_present(row, ("NPV(Base)", "NPV", "PresentValue", "presentvalue"))
            if value is not None:
                total += value
        return total

    def _rows_from_report(self, report: Any) -> Sequence[Dict[str, Any]]:
        if report is None:
            return []
        if isinstance(report, list) and (not report or isinstance(report[0], dict)):
            return report
        if hasattr(report, "columns") and hasattr(report, "data"):
            cols = list(report.columns())
            data = report.data()
            out = []
            for row in data:
                out.append({str(c): row[i] for i, c in enumerate(cols) if i < len(row)})
            return out
        if hasattr(report, "rows") and hasattr(report, "columns") and hasattr(report, "header"):
            rows = int(report.rows())
            cols = int(report.columns())
            headers = [str(report.header(c)) for c in range(cols)]
            out = []
            for r in range(rows):
                row: Dict[str, Any] = {}
                for c, h in enumerate(headers):
                    v = None
                    for fn_name in ("dataAsString", "dataAsReal", "dataAsSize", "dataAsDate", "dataAsPeriod"):
                        fn = getattr(report, fn_name, None)
                        if not callable(fn):
                            continue
                        try:
                            candidate = fn(r, c)
                            v = candidate
                            break
                        except Exception:
                            continue
                    row[h] = v
                out.append(row)
            return out
        return []

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _first_present(self, row: Dict[str, Any], keys: Sequence[str]) -> float | None:
        for k in keys:
            if k in row and row[k] is not None:
                v = self._to_float(row[k])
                if v is not None:
                    return v
        return None


_TENOR_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([YMWD])$", re.IGNORECASE)
_PERIOD_PART_RE = re.compile(r"([0-9]+(?:\\.[0-9]+)?)([YMWD])", re.IGNORECASE)


@contextlib.contextmanager
def _tmp_xml_path(xml_text: str):
    """Write *xml_text* to a temp file, yield the file path, and delete on exit."""
    fd, path = tempfile.mkstemp(suffix=".xml")
    try:
        os.write(fd, xml_text.encode("utf-8"))
        os.close(fd)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _parse_tenor_to_years(value: str) -> float:
    txt = value.strip()
    m = _TENOR_RE.match(txt)
    if m is None:
        parts = _PERIOD_PART_RE.findall(txt)
        if not parts or "".join(a + b for a, b in parts).upper() != txt.upper():
            raise ValueError(f"unsupported tenor '{value}'")
        total = 0.0
        for n_txt, u_txt in parts:
            n = float(n_txt)
            u = u_txt.upper()
            if u == "Y":
                total += n
            elif u == "M":
                total += n / 12.0
            elif u == "W":
                total += n / 52.0
            else:
                total += n / 365.0
        return total
    n = float(m.group(1))
    u = m.group(2).upper()
    if u == "Y":
        return n
    if u == "M":
        return n / 12.0
    if u == "W":
        return n / 52.0
    return n / 365.0


def _parse_float_grid(text: str | None) -> np.ndarray:
    txt = (text or "").strip()
    if not txt:
        return np.array([], dtype=float)
    return np.asarray([float(x.strip()) for x in txt.split(",") if x.strip()], dtype=float)


def _active_dim_mode(snapshot: XVASnapshot) -> str | None:
    runtime = snapshot.config.runtime
    if runtime is None or runtime.xva_analytic is None:
        return None
    return runtime.xva_analytic.dim_model


def _pfe_quantile(snapshot: XVASnapshot) -> float:
    runtime = snapshot.config.runtime
    quantile = 0.95
    if runtime is not None and runtime.xva_analytic is not None:
        quantile = float(getattr(runtime.xva_analytic, "pfe_quantile", quantile))
    return min(max(quantile, 0.0), 1.0)


def _configured_output_dir(snapshot: XVASnapshot) -> Path | None:
    configured = str(snapshot.config.params.get("outputPath", "")).strip()
    if not configured:
        return None
    path = Path(configured)
    if path.is_absolute():
        return path
    ore_xml_path = snapshot.source_meta.get("config")
    if ore_xml_path and ore_xml_path.path and ore_xml_path.path.endswith(".xml"):
        ore_path = Path(ore_xml_path.path)
        return (ore_path.parent.parent / path).resolve()
    return Path.cwd() / path


def _read_csv_report(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _normalize_asof_date(asof: str) -> str:
    s = asof.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _date_from_time(asof: str, t: float) -> str:
    base = datetime.strptime(_normalize_asof_date(asof), "%Y-%m-%d").date()
    return (base + timedelta(days=int(round(float(t) * 365.25)))).isoformat()


def _parse_lgm_params_from_xml_node(node: ET.Element, source_label: str) -> dict:
    """Extract LGM parameters from an already-located LGM XML node."""
    vol_node = node.find("./Volatility")
    rev_node = node.find("./Reversion")
    trans_node = node.find("./ParameterTransformation")
    if vol_node is None or rev_node is None or trans_node is None:
        raise EngineRunError(
            f"{source_label} LGM node missing Volatility/Reversion/ParameterTransformation"
        )
    return {
        "alpha_times": _parse_float_grid(vol_node.findtext("./TimeGrid")),
        "alpha_values": _parse_float_grid(vol_node.findtext("./InitialValue")),
        "kappa_times": _parse_float_grid(rev_node.findtext("./TimeGrid")),
        "kappa_values": _parse_float_grid(rev_node.findtext("./InitialValue")),
        # ORE's ShiftHorizon is not the same object as the additive H(t) shift
        # used by the Python LGM kernel. Feeding it through materially breaks
        # parity on calibrated Bermudan cases.
        "shift": 0.0,
        "scaling": float((trans_node.findtext("./Scaling") or "1").strip()),
    }


def _parse_lgm_params_from_simulation_xml_text(xml_text: str, ccy_key: str = "EUR") -> Dict[str, object]:
    root = ET.fromstring(xml_text)
    models = root.find("./CrossAssetModel/InterestRateModels")
    if models is None:
        raise EngineRunError("simulation.xml missing CrossAssetModel/InterestRateModels")
    node = models.find(f"./LGM[@ccy='{ccy_key}']")
    if node is None:
        node = models.find("./LGM[@ccy='default']")
    if node is None:
        raise EngineRunError(f"simulation.xml missing LGM node for {ccy_key}")
    return _parse_lgm_params_from_xml_node(node, "simulation.xml")


def _parse_lgm_params_from_calibration_xml_text(xml_text: str, ccy_key: str = "EUR") -> Dict[str, object]:
    root = ET.fromstring(xml_text)
    models = root.find("./InterestRateModels")
    if models is None:
        raise EngineRunError("calibration.xml missing InterestRateModels")
    node = models.find(f"./LGM[@key='{ccy_key}']")
    if node is None:
        node = models.find(f"./LGM[@ccy='{ccy_key}']")
    if node is None:
        raise EngineRunError(f"calibration.xml missing LGM node for {ccy_key}")
    return _parse_lgm_params_from_xml_node(node, "calibration.xml")


def _parse_exposure_times_from_simulation_xml_text(xml_text: str) -> np.ndarray:
    root = ET.fromstring(xml_text)
    vals: List[float] = [0.0]
    grid_txt = root.findtext("./Parameters/Grid")
    if grid_txt:
        g = [x.strip() for x in grid_txt.split(",") if x.strip()]
        if len(g) == 2:
            try:
                n = int(float(g[0]))
                step = _parse_tenor_to_years(g[1])
                if n > 0 and step > 0:
                    vals.extend([i * step for i in range(1, n + 1)])
            except Exception:
                pass
    for path in (
        "./Market/YieldCurves/Configuration/Tenors",
        "./Market/DefaultCurves/Tenors",
        "./CrossAssetModel/InterestRateModels/LGM/CalibrationSwaptions/Expiries",
    ):
        txt = root.findtext(path)
        if not txt:
            continue
        for item in txt.split(","):
            s = item.strip()
            if not s:
                continue
            try:
                vals.append(_parse_tenor_to_years(s))
            except Exception:
                continue
    arr = np.asarray(sorted(set(float(x) for x in vals if x >= 0.0)), dtype=float)
    return arr


def _parse_stochastic_fx_pairs_from_simulation_xml_text(
    xml_text: str,
    *,
    model_ccy: str,
    trade_specs: Sequence[_TradeSpec],
) -> Tuple[str, ...]:
    if not xml_text.strip():
        return tuple(
            sorted(
                {
                    f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}"
                    for spec in trade_specs
                    if spec.kind == "FXForward" and isinstance(spec.trade.product, FXForward)
                }
            )
        )
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return ()
    market_pair_nodes = root.findall("./Market/FxVolatilities/CurrencyPairs/CurrencyPair")
    market_pairs = {
        str(node.text or "").strip().upper().replace("-", "/")
        for node in market_pair_nodes
        if str(node.text or "").strip()
    }
    normalized_market_pairs = set()
    for item in market_pairs:
        if "/" in item:
            normalized_market_pairs.add(item)
        elif len(item) == 6:
            normalized_market_pairs.add(f"{item[:3]}/{item[3:]}")
    if normalized_market_pairs:
        return tuple(
            sorted(
                {
                    f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}"
                    for spec in trade_specs
                    if spec.kind == "FXForward"
                    and isinstance(spec.trade.product, FXForward)
                    and f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}" in normalized_market_pairs
                }
            )
        )
    domestic = str(
        root.findtext("./CrossAssetModel/DomesticCcy")
        or model_ccy
    ).strip().upper()
    foreign_nodes = root.findall("./CrossAssetModel/ForeignExchangeModels/CrossCcyLGM")
    if not foreign_nodes:
        return tuple(
            sorted(
                {
                    f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}"
                    for spec in trade_specs
                    if spec.kind == "FXForward" and isinstance(spec.trade.product, FXForward)
                }
            )
        )
    supported_foreigns = {
        str(node.attrib.get("foreignCcy") or "").strip().upper()
        for node in foreign_nodes
        if str(node.attrib.get("foreignCcy") or "").strip()
    }
    has_default = "DEFAULT" in supported_foreigns
    explicit_foreigns = {ccy for ccy in supported_foreigns if ccy != "DEFAULT"}
    out = []
    for spec in trade_specs:
        if spec.kind != "FXForward" or not isinstance(spec.trade.product, FXForward):
            continue
        base = spec.trade.product.pair[:3].upper()
        quote = spec.trade.product.pair[3:].upper()
        if base == domestic and (quote in explicit_foreigns or (has_default and quote != domestic)):
            out.append(f"{base}/{quote}")
        elif quote == domestic and (base in explicit_foreigns or (has_default and base != domestic)):
            out.append(f"{base}/{quote}")
    return tuple(sorted(set(out)))


def _fallback_exposure_grid(snapshot: XVASnapshot) -> np.ndarray:
    max_mat = float(snapshot.config.horizon_years)
    for t in snapshot.portfolio.trades:
        p = t.product
        m = getattr(p, "maturity_years", None)
        if isinstance(m, (int, float)):
            max_mat = max(max_mat, float(m))
    if max_mat <= 0.0:
        max_mat = 1.0
    steps = max(4, int(np.ceil(max_mat * 4.0)))
    return np.linspace(0.0, max_mat, steps + 1)


def _build_sticky_closeout_times(times: np.ndarray, mpor_years: float) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return t.copy()
    mpor = float(mpor_years)
    if mpor <= 1.0e-15:
        return t.copy()
    return np.minimum(t + mpor, float(t[-1]))


def _parse_market_overlay(raw_quotes: Sequence[Any]) -> Dict[str, Any]:
    zero: Dict[str, List[Tuple[float, float]]] = {}
    named_zero: Dict[str, List[Tuple[float, float]]] = {}
    fwd: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    fx: Dict[str, float] = {}
    fx_vol: Dict[str, List[Tuple[float, float]]] = {}
    swaption_normal_vols: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    cms_correlations: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    hazard: Dict[str, List[Tuple[float, float]]] = {}
    recovery: Dict[str, float] = {}
    cds_spreads: Dict[str, List[Tuple[float, float]]] = {}
    for q in raw_quotes:
        key = str(q.key).strip()
        up = key.upper()
        val = float(q.value)
        parts = up.split("/")
        if len(parts) >= 4 and parts[0] == "FX" and parts[1] == "RATE":
            fx[parts[2] + parts[3]] = val
            continue
        if len(parts) >= 3 and parts[0] == "FX":
            fx[parts[1] + parts[2]] = val
            continue
        if len(parts) >= 6 and parts[0] == "FX_OPTION" and parts[1] == "RATE_LNVOL":
            ccy1 = parts[2]
            ccy2 = parts[3]
            tenor = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = _parse_tenor_to_years(tenor)
            except Exception:
                continue
            fx_vol.setdefault(ccy1 + ccy2, []).append((t, val))
            continue
        if len(parts) >= 6 and parts[0] == "SWAPTION" and parts[1] == "RATE_NVOL":
            ccy = parts[2]
            expiry = parts[3]
            swap_tenor = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = _parse_tenor_to_years(expiry)
            except Exception:
                continue
            swaption_normal_vols.setdefault((ccy, swap_tenor), []).append((t, val))
            continue
        if len(parts) >= 6 and parts[0] == "CORRELATION" and parts[1] == "RATE":
            idx1 = parts[2]
            idx2 = parts[3]
            expiry = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = _parse_tenor_to_years(expiry)
            except Exception:
                continue
            cms_correlations.setdefault(tuple(sorted((idx1, idx2))), []).append((t, val))
            continue
        if len(parts) >= 4 and parts[0] == "ZERO" and parts[1] == "RATE":
            ccy = parts[2]
            tenor = parts[3]
            curve_name = None
            if len(parts) >= 6:
                curve_name = parts[3]
                tenor = parts[-1]
            try:
                t = _parse_tenor_to_years(tenor)
            except Exception:
                continue
            if curve_name is not None:
                named_zero.setdefault(curve_name, []).append((t, val))
            else:
                zero.setdefault(ccy, []).append((t, val))
            continue
        if len(parts) >= 6 and parts[0] == "IR_SWAP" and parts[1] == "RATE":
            ccy = parts[2]
            idx_tenor = parts[4].upper()
            tenor = parts[-1]
            try:
                t = _parse_tenor_to_years(tenor)
            except Exception:
                continue
            if 0.0 < t <= 80.0:
                if idx_tenor in ("1D", "ON", "O/N"):
                    zero.setdefault(ccy, []).append((t, val))
                else:
                    fwd.setdefault(ccy, {}).setdefault(idx_tenor, []).append((t, val))
            continue
        if len(parts) >= 5 and parts[0] == "MM" and parts[1] == "RATE":
            ccy = parts[2]
            idx_tenor = parts[4].upper()
            tenor = parts[-1]
            try:
                t = _parse_tenor_to_years(tenor)
            except Exception:
                continue
            if 0.0 < t <= 10.0:
                if idx_tenor in ("1D", "ON", "O/N"):
                    zero.setdefault(ccy, []).append((t, val))
                else:
                    fwd.setdefault(ccy, {}).setdefault(idx_tenor, []).append((t, val))
            continue
        if len(parts) >= 6 and parts[0] == "HAZARD_RATE":
            cpty = parts[2]
            tenor = parts[-1]
            try:
                t = _parse_tenor_to_years(tenor)
            except Exception:
                continue
            hazard.setdefault(cpty, []).append((t, val))
            continue
        if len(parts) >= 6 and parts[0] == "CDS" and parts[1] == "CREDIT_SPREAD":
            cpty = parts[2]
            tenor = parts[-1]
            try:
                t = _parse_tenor_to_years(tenor)
            except Exception:
                continue
            cds_spreads.setdefault(cpty, []).append((t, val))
            continue
        if len(parts) >= 5 and parts[0] == "RECOVERY_RATE":
            cpty = parts[2]
            recovery[cpty] = val
            continue
    for cpty, spreads in cds_spreads.items():
        if cpty in hazard:
            continue
        rec = float(recovery.get(cpty, 0.4))
        lgd = max(1.0 - rec, 1.0e-6)
        hazard[cpty] = [(t, max(float(spread) / lgd, 0.0)) for t, spread in spreads]
    return {
        "zero": zero,
        "named_zero": named_zero,
        "fwd": fwd,
        "fx": fx,
        "fx_vol": fx_vol,
        "swaption_normal_vols": swaption_normal_vols,
        "cms_correlations": cms_correlations,
        "hazard": hazard,
        "cds_spreads": cds_spreads,
        "recovery": recovery,
    }


def _default_index_for_ccy(ccy: str) -> str:
    c = ccy.upper()
    if c == "USD":
        return "USD-LIBOR-3M"
    if c == "GBP":
        return "GBP-LIBOR-3M"
    if c == "CHF":
        return "CHF-LIBOR-3M"
    if c == "JPY":
        return "JPY-LIBOR-3M"
    return "EUR-EURIBOR-3M"


def _build_irs_legs_from_trade(trade: Trade, asof: str | None = None) -> Dict[str, np.ndarray]:
    """Last-resort IRS leg builder using the dataclass schedule fields, no historical fixings.

    This path is reached only when neither flows.csv nor portfolio.xml loading
    succeeded.  Results will be approximate — calendar, day-count and index
    conventions are hard-coded.  Always prefer loading legs from ORE output
    artifacts or a portfolio XML.
    """
    import warnings as _warnings
    _warnings.warn(
        f"Using fallback leg schedule for trade {trade.trade_id} "
        "(no flows.csv or portfolio.xml available). "
        "Schedule timing is generated from the IRS dataclass fields and remains approximate for calendars/day-count roll rules.",
        UserWarning,
        stacklevel=3,
    )
    p = trade.product
    if not isinstance(p, IRS):
        raise EngineRunError(f"Cannot build IRS legs for non-IRS trade {trade.trade_id}")
    start_offset, end_offset = _irs_schedule_bounds(p, asof)
    fixed_pay, fixed_start, fixed_end = _schedule_periods(
        start_offset, end_offset, p.fixed_leg_tenor, p.fixed_schedule_rule
    )
    float_pay, float_start, float_end = _schedule_periods(
        start_offset, end_offset, p.float_leg_tenor, p.float_schedule_rule
    )
    fixing = float_start - (float(p.fixing_days) / 365.25)
    fixed_sign = -1.0 if p.pay_fixed else 1.0
    float_sign = -fixed_sign
    float_index = str(p.float_index or trade.additional_fields.get("index", _default_index_for_ccy(p.ccy))).upper()
    float_index_tenor = float_index.split("-")[-1].upper() if "-" in float_index else ""
    return {
        "fixed_pay_time": fixed_pay,
        "fixed_accrual": np.maximum(fixed_end - fixed_start, 0.0),
        "fixed_rate": np.full(fixed_pay.shape, float(p.fixed_rate)),
        "fixed_notional": np.full(fixed_pay.shape, float(p.notional)),
        "fixed_sign": np.full(fixed_pay.shape, fixed_sign),
        "fixed_amount": fixed_sign * float(p.notional) * float(p.fixed_rate) * np.maximum(fixed_end - fixed_start, 0.0),
        "float_pay_time": float_pay,
        "float_start_time": float_start,
        "float_end_time": float_end,
        "float_accrual": np.maximum(float_end - float_start, 0.0),
        "float_notional": np.full(float_pay.shape, float(p.notional)),
        "float_sign": np.full(float_pay.shape, float_sign),
        "float_spread": np.full(float_pay.shape, float(p.float_spread)),
        "float_coupon": np.zeros(float_pay.shape),
        "float_amount": np.zeros(float_pay.shape),
        "float_fixing_time": fixing,
        "float_is_historically_fixed": np.zeros(float_pay.shape, dtype=bool),
        "float_index": float_index,
        "float_index_tenor": float_index_tenor,
    }


def _irs_schedule_bounds(product: IRS, asof: str | None) -> Tuple[float, float]:
    if product.start_date and asof:
        start = _normalize_asof_date(product.start_date)
        ref = _normalize_asof_date(asof)
        start_offset = (datetime.strptime(start, "%Y-%m-%d") - datetime.strptime(ref, "%Y-%m-%d")).days / 365.25
    else:
        start_offset = 0.0
    if product.end_date and asof:
        end = _normalize_asof_date(product.end_date)
        ref = _normalize_asof_date(asof)
        end_offset = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(ref, "%Y-%m-%d")).days / 365.25
    else:
        end_offset = start_offset + float(product.maturity_years)
    end_offset = max(end_offset, start_offset + 1.0 / 365.25)
    return float(start_offset), float(end_offset)


def _schedule_periods(start: float, end: float, tenor: str, rule: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = _tenor_to_years(tenor)
    if step <= 0.0:
        raise EngineRunError(f"Unsupported IRS tenor '{tenor}'")
    rule_name = str(rule or "Forward").strip().lower()
    starts: List[float] = []
    stops: List[float] = []
    if rule_name == "backward":
        current = float(end)
        while current > start + 1.0e-12:
            prev = max(start, current - step)
            starts.append(prev)
            stops.append(current)
            current = prev
        starts.reverse()
        stops.reverse()
    else:
        current = float(start)
        while current < end - 1.0e-12:
            nxt = min(end, current + step)
            starts.append(current)
            stops.append(nxt)
            current = nxt
    start_arr = np.asarray(starts, dtype=float)
    stop_arr = np.asarray(stops, dtype=float)
    return stop_arr.copy(), start_arr, stop_arr


def _tenor_to_years(tenor: str) -> float:
    text = str(tenor).strip().upper()
    m = re.fullmatch(r"(\d+)([DWMY])", text)
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "D":
        return n / 365.25
    if unit == "W":
        return (7.0 * n) / 365.25
    if unit == "M":
        return n / 12.0
    return n


def _counterparty_for_netting(snapshot: XVASnapshot, netting_set: str) -> str:
    for t in snapshot.portfolio.trades:
        if t.netting_set == netting_set:
            return t.counterparty
    return netting_set


def _estimate_vm_collateral_paths(snapshot: XVASnapshot, netting_set: str, valuation_paths: np.ndarray) -> np.ndarray:
    ns_cfg = snapshot.netting.netting_sets.get(netting_set)
    if ns_cfg is None or not bool(ns_cfg.active_csa):
        return np.zeros_like(valuation_paths)

    threshold_recv = float(ns_cfg.threshold_receive or 0.0)
    threshold_pay = float(ns_cfg.threshold_pay or 0.0)
    mta_recv = max(float(ns_cfg.mta_receive or 0.0), 0.0)
    mta_pay = max(float(ns_cfg.mta_pay or 0.0), 0.0)

    vm_balance = 0.0
    for balance in snapshot.collateral.balances:
        if balance.netting_set_id == netting_set:
            vm_balance = float(balance.variation_margin)
            break

    collateral = np.full_like(valuation_paths, vm_balance, dtype=float)
    collateral = np.where(valuation_paths > (threshold_recv + mta_recv), valuation_paths - threshold_recv, collateral)
    collateral = np.where(valuation_paths < -(threshold_pay + mta_pay), valuation_paths + threshold_pay, collateral)
    return collateral


def _own_name_from_runtime(snapshot: XVASnapshot) -> str:
    runtime = snapshot.config.runtime
    if runtime is None:
        return "BANK"
    xva = runtime.xva_analytic
    if xva.dva_name:
        return str(xva.dva_name)
    if runtime.counterparties.ids:
        return str(runtime.counterparties.ids[0])
    return "BANK"


def _effective_funding_spreads(snapshot: XVASnapshot, inputs: _PythonLgmInputs, own_name: str) -> Tuple[float, float]:
    own = own_name.upper()
    runtime = snapshot.config.runtime
    ccy = snapshot.config.base_currency.upper()
    borrow_curve = runtime.xva_analytic.fva_borrowing_curve if runtime is not None else None
    lend_curve = runtime.xva_analytic.fva_lending_curve if runtime is not None else None

    # Prefer explicit market funding spread quotes if present.
    m_borrow = _market_yield_spread(snapshot, ccy, str(borrow_curve) if borrow_curve else None)
    m_lend = _market_yield_spread(snapshot, ccy, str(lend_curve) if lend_curve else None)
    if m_borrow is not None or m_lend is not None:
        b = float(m_borrow if m_borrow is not None else 0.0)
        l = float(m_lend if m_lend is not None else 0.0)
        return b, l

    hz = inputs.hazard_rates.get(own)
    rec = float(inputs.recovery_rates.get(own, 0.4))
    hazard_avg = float(np.mean(hz)) if hz is not None and hz.size > 0 else 0.015
    # Approximate own funding spread from credit intensity and LGD.
    borrow = max(0.0005, hazard_avg * max(1.0 - rec, 0.05))
    # Lending benefit usually lower magnitude than borrowing cost in fallback mode.
    lend = 0.5 * borrow

    if runtime is not None:
        xva = runtime.xva_analytic
        # If dedicated funding curve handles are configured, widen spreads modestly.
        # The 1.5× scalar is a conservative liability-side funding premium:
        # own-name CDS spread alone understates the all-in funding cost because
        # it excludes bid/offer and liquidity components.
        if xva.fva_borrowing_curve:
            borrow *= 1.5
        if xva.fva_lending_curve:
            lend *= 1.5
    return borrow, lend


def _market_yield_spread(snapshot: XVASnapshot, ccy: str, curve_name: str | None) -> float | None:
    if not curve_name:
        return None
    target = f"ZERO/YIELD_SPREAD/{ccy}/{curve_name.upper()}/"
    vals: List[float] = []
    for q in snapshot.market.raw_quotes:
        k = str(q.key).upper()
        if k.startswith(target):
            try:
                vals.append(float(q.value))
            except Exception:
                pass
    if not vals:
        return None
    return float(np.mean(vals))


def _trade_notional(trade: Trade) -> float:
    p = trade.product
    n = getattr(p, "notional", None)
    if isinstance(n, (int, float)):
        return abs(float(n))
    return 0.0


def _spot_from_quotes(pair6: str, inputs: _PythonLgmInputs, default: float = 1.0) -> float:
    base = pair6[:3].upper()
    quote = pair6[3:].upper()
    fwd = base + quote
    inv = quote + base
    if fwd in inputs.fx_spots:
        return float(inputs.fx_spots[fwd])
    if inv in inputs.fx_spots:
        return 1.0 / max(float(inputs.fx_spots[inv]), 1.0e-12)
    return float(default)


def _convert_fx_forward_npv_to_reporting_ccy(
    *,
    npv_quote: np.ndarray,
    report_ccy: str,
    base_ccy: str,
    quote_ccy: str,
    spot_path: np.ndarray,
    inputs: _PythonLgmInputs,
) -> np.ndarray:
    report = report_ccy.upper()
    base = base_ccy.upper()
    quote = quote_ccy.upper()
    if report == quote:
        return np.asarray(npv_quote, dtype=float)
    if report == base:
        return np.asarray(npv_quote, dtype=float) / np.maximum(np.asarray(spot_path, dtype=float), 1.0e-12)
    # Fallback cross conversion using asof spots when report ccy is neither leg.
    cross = _spot_from_quotes(report + quote, inputs, default=0.0)
    if cross > 0.0:
        return np.asarray(npv_quote, dtype=float) / cross
    cross_inv = _spot_from_quotes(quote + report, inputs, default=0.0)
    if cross_inv > 0.0:
        return np.asarray(npv_quote, dtype=float) * cross_inv
    return np.asarray(npv_quote, dtype=float)


def _fx_vol_for_trade(inputs: _PythonLgmInputs, pair6: str, maturity: float, default: float = 0.15) -> float:
    pair = pair6.upper()
    points = list(inputs.fx_vols.get(pair, ()))
    if not points:
        inv = pair[3:] + pair[:3]
        points = list(inputs.fx_vols.get(inv, ()))
    if not points:
        return float(default)
    target = max(float(maturity), 0.0)
    best = min(points, key=lambda item: abs(float(item[0]) - target))
    return float(best[1])


def _fx_carry_for_pair(pair: str, inputs: _PythonLgmInputs, horizon: float) -> float:
    base, quote = pair.upper().replace("-", "/").split("/")
    h = max(min(float(horizon), 1.0), 1.0 / 12.0)
    p_quote = max(float(inputs.discount_curves[quote](h)), 1.0e-18)
    p_base = max(float(inputs.discount_curves[base](h)), 1.0e-18)
    r_quote = -math.log(p_quote) / h
    r_base = -math.log(p_base) / h
    return float(r_quote - r_base)


def _curve_family_from_source_column(source_column: str) -> str:
    txt = str(source_column).strip().upper()
    if not txt:
        return ""
    for tenor in ("ON", "O/N", "1D", "1W", "1M", "3M", "6M", "12M"):
        if txt.endswith(f"-{tenor}") or txt.endswith(tenor):
            return tenor
    if any(tag in txt for tag in ("EONIA", "FEDFUNDS", "SONIA", "TOIS", "TONAR")):
        return "1D"
    return ""


def _zero_rate_from_curve(curve: Callable[[float], float], t: float) -> float:
    tt = max(float(t), 1.0e-8)
    return float(-np.log(max(float(curve(tt)), 1.0e-18)) / tt)


def _build_zero_rate_shocked_curve(
    base_curve: Callable[[float], float],
    node_times: Sequence[float],
    node_shifts: Sequence[float],
) -> Callable[[float], float]:
    times = np.asarray([float(x) for x in node_times], dtype=float)
    shifts = np.asarray([float(x) for x in node_shifts], dtype=float)
    if times.size == 0 or times.size != shifts.size:
        return base_curve
    order = np.argsort(times)
    times = times[order]
    shifts = shifts[order]

    shocked_node_logs = np.log(
        np.clip(
            np.asarray([float(base_curve(float(t))) for t in times], dtype=float)
            * np.exp(-times * shifts),
            1.0e-18,
            None,
        )
    )

    def shocked_curve(t: float) -> float:
        tt = max(float(t), 0.0)
        if tt <= 1.0e-12:
            return 1.0

        # ORE curve configs in the benchmark cases use Discount + LogLinear.
        # Apply the node shocks at the bucket tenors, then interpolate the
        # shocked discount factors in log-discount space between nodes.
        if tt <= times[0]:
            base_df = float(base_curve(tt))
            return float(base_df * np.exp(-shifts[0] * tt))
        if tt >= times[-1]:
            base_df = float(base_curve(tt))
            return float(base_df * np.exp(-shifts[-1] * tt))

        shocked_log_df = float(np.interp(tt, times, shocked_node_logs))
        return float(np.exp(shocked_log_df))

    return shocked_curve


def _apply_curve_node_shocks(
    snapshot: XVASnapshot,
    discount_curves: Dict[str, Callable[[float], float]],
    forward_curves: Dict[str, Callable[[float], float]],
    forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]],
    forward_curves_by_name: Dict[str, Callable[[float], float]],
    xva_discount_curve: Optional[Callable[[float], float]],
) -> Tuple[
    Dict[str, Callable[[float], float]],
    Dict[str, Callable[[float], float]],
    Dict[str, Dict[str, Callable[[float], float]]],
    Dict[str, Callable[[float], float]],
    Optional[Callable[[float], float]],
]:
    specs = snapshot.config.params.get("python.curve_node_shocks")
    if not isinstance(specs, dict):
        return discount_curves, forward_curves, forward_curves_by_tenor, forward_curves_by_name, xva_discount_curve

    discount_specs = specs.get("discount", {}) if isinstance(specs.get("discount", {}), dict) else {}
    forward_specs = specs.get("forward", {}) if isinstance(specs.get("forward", {}), dict) else {}

    discount_curves = dict(discount_curves)
    forward_curves = dict(forward_curves)
    forward_curves_by_tenor = {ccy: dict(v) for ccy, v in forward_curves_by_tenor.items()}
    forward_curves_by_name = dict(forward_curves_by_name)

    for ccy, cfg in discount_specs.items():
        if ccy not in discount_curves or not isinstance(cfg, dict):
            continue
        node_times = cfg.get("node_times", ())
        node_shifts = cfg.get("node_shifts", ())
        bumped = _build_zero_rate_shocked_curve(discount_curves[ccy], node_times, node_shifts)
        discount_curves[ccy] = bumped
        if snapshot.config.base_currency.upper() == str(ccy).upper() and xva_discount_curve is not None:
            xva_discount_curve = bumped

    for ccy, tenor_map in forward_specs.items():
        if not isinstance(tenor_map, dict):
            continue
        tenor_curves = dict(forward_curves_by_tenor.get(ccy, {}))
        for tenor, cfg in tenor_map.items():
            if tenor not in tenor_curves or not isinstance(cfg, dict):
                continue
            node_times = cfg.get("node_times", ())
            node_shifts = cfg.get("node_shifts", ())
            bumped = _build_zero_rate_shocked_curve(tenor_curves[tenor], node_times, node_shifts)
            tenor_curves[tenor] = bumped
            for name, curve in list(forward_curves_by_name.items()):
                if curve is forward_curves_by_tenor.get(ccy, {}).get(tenor):
                    forward_curves_by_name[name] = bumped
            if ccy in forward_curves:
                preferred = "6M" if "6M" in tenor_curves else (sorted(tenor_curves)[0] if tenor_curves else "")
                if preferred and preferred == tenor:
                    forward_curves[ccy] = bumped
        forward_curves_by_tenor[ccy] = tenor_curves

    return discount_curves, forward_curves, forward_curves_by_tenor, forward_curves_by_name, xva_discount_curve


def _quote_matches_discount_curve(
    key: str,
    ccy: str,
    source_column: str,
    fallback_family: str = "",
) -> bool:
    parts = str(key).strip().upper().split("/")
    if len(parts) < 3 or parts[2] != ccy.upper():
        return False
    family = _curve_family_from_source_column(source_column)
    if not family:
        family = str(fallback_family).strip().upper()
    if parts[0] == "MM" and parts[1] == "RATE":
        return True
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        index_tenor = parts[4] if len(parts) > 5 else ""
        if family in ("", "1D", "ON", "O/N"):
            return index_tenor in ("1D", "ON", "O/N")
        return index_tenor == family
    if parts[0] == "FRA" and parts[1] == "RATE":
        return family not in ("", "1D", "ON", "O/N") and len(parts) > 4 and parts[-1] == family
    return False


def _quote_matches_forward_curve(key: str, ccy: str, tenor: str) -> bool:
    parts = str(key).strip().upper().split("/")
    if len(parts) < 3 or parts[2] != ccy.upper():
        return False
    if parts[0] == "MM" and parts[1] == "RATE":
        return True
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        return len(parts) > 5 and parts[4] == tenor.upper()
    if parts[0] == "FRA" and parts[1] == "RATE":
        return len(parts) > 4 and parts[-1] == tenor.upper()
    return False


def _parse_ore_tenor_to_years(tenor: str) -> float:
    txt = str(tenor).strip().upper()
    match = re.match(r"^(\d+)([YMWD])$", txt)
    if match is None:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "Y":
        return value
    if unit == "M":
        return value / 12.0
    if unit == "W":
        return value / 52.0
    return value / 365.0


def _toy_trade_numbers(trade: Trade):
    p = trade.product
    if isinstance(p, IRS):
        fair = 0.03
        direction = -1.0 if p.pay_fixed else 1.0
        pv = direction * p.notional * (fair - p.fixed_rate) * p.maturity_years
        epe = max(pv, 0.0) * 0.35 + abs(p.notional) * 0.001
        return pv, epe

    if isinstance(p, FXForward):
        spot = 1.1
        direction = 1.0 if p.buy_base else -1.0
        pv = direction * p.notional * (spot - p.strike)
        epe = max(pv, 0.0) * 0.5 + abs(p.notional) * 0.0008
        return pv, epe

    return 0.0, 1.0


def classify_portfolio_support(
    snapshot: XVASnapshot,
    *,
    fallback_to_swig: bool = True,
) -> Dict[str, Any]:
    """Public preflight helper for native-vs-SWIG support classification."""
    return PythonLgmAdapter(fallback_to_swig=fallback_to_swig).classify_portfolio_support(snapshot)
