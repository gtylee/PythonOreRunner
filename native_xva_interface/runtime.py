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
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import importlib
import os
import re
from pathlib import Path
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .dataclasses import (
    FXForward,
    IRS,
    MporConfig,
    Trade,
    XVASnapshot,
)
from .exceptions import EngineRunError
from .mapper import MappedInputs, _default_index_for_ccy, _resolve_mpor_config, build_input_parameters, map_snapshot
from .results import CubeAccessor, XVAResult


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
    times: np.ndarray              # Full simulation grid (valuation grid plus sticky closeout grid).
    valuation_times: np.ndarray    # Augmented valuation grid (union of exposure grid and trade cash-flow dates).
    observation_times: np.ndarray  # Subset of times used as XVA observation points (from simulation.xml or fallback).
    observation_closeout_times: np.ndarray  # Sticky closeout times associated with observation_times.
    discount_curves: Dict[str, Callable[[float], float]]   # Risk-free discount curves keyed by ISO currency.
    forward_curves: Dict[str, Callable[[float], float]]    # Composite forward/index curves keyed by ISO currency.
    forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]]  # Forward curves keyed by (ccy, tenor string).
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
    trade_specs: Tuple[_TradeSpec, ...]
    unsupported: Tuple[Trade, ...]
    mpor: MporConfig
    input_provenance: Dict[str, str]  # Diagnostic dict recording which source was used for each input (model_params, market, grid, portfolio).


class PythonLgmAdapter:
    """Adapter that values supported trades using the Python LGM stack."""

    def __init__(self, fallback_to_swig: bool = True):
        self.fallback_to_swig = bool(fallback_to_swig)
        self._loaded = False
        self._lgm_mod = None
        self._irs_utils = None
        self._fx_utils = None
        self._ore_snapshot_mod = None

    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        """Execute the Python-LGM XVA computation pipeline.

        Simulates LGM state paths, prices each supported trade (IRS and
        FXForward), optionally falls back to the ORE SWIG adapter for
        unsupported trade types, then assembles and returns an
        :class:`XVAResult` with per-metric XVA figures and exposure cubes.
        """
        self._ensure_py_lgm_imports()
        inputs = self._extract_inputs(snapshot, mapped)

        n_times = int(inputs.times.size)
        n_paths = int(snapshot.config.num_paths)
        npv_by_trade: Dict[str, np.ndarray] = {}
        fallback_trades: List[Trade] = []
        unsupported: List[Trade] = list(inputs.unsupported)

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
            elif spec.kind == "FXForward":
                vals = self._price_fx_forward(spec.trade, inputs, n_times, n_paths)
                npv_by_trade[spec.trade.trade_id] = vals
            else:
                unsupported.append(spec.trade)

        fallback_result: XVAResult | None = None
        if unsupported:
            if not self.fallback_to_swig:
                bad = ", ".join(sorted({f"{t.trade_id}:{t.trade_type}" for t in unsupported}))
                raise EngineRunError(f"Unsupported trade types for PythonLgmAdapter: {bad}")
            try:
                swig_adapter = ORESwigAdapter()
            except Exception as exc:
                bad = ", ".join(sorted({f"{t.trade_id}:{t.trade_type}" for t in unsupported}))
                raise EngineRunError(f"Unsupported trade types for PythonLgmAdapter and SWIG unavailable: {bad}; {exc}") from exc
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
        (``Tools/PythonOreRunner/py_ore_tools``).  Raises :class:`EngineRunError`
        if neither path succeeds.
        """
        if self._loaded:
            return
        try:
            from py_ore_tools import irs_xva_utils, lgm, lgm_fx_xva_utils, ore_snapshot

            self._lgm_mod = lgm
            self._irs_utils = irs_xva_utils
            self._fx_utils = lgm_fx_xva_utils
            self._ore_snapshot_mod = ore_snapshot
            self._loaded = True
            return
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[3]
        tools_dir = repo_root / "Tools" / "PythonOreRunner" / "py_ore_tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

        try:
            import irs_xva_utils as irs_xva_utils_local
            import lgm as lgm_local
            import lgm_fx_xva_utils as lgm_fx_xva_utils_local
            import ore_snapshot as ore_snapshot_local

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
        if "calibration.xml" in xml:
            lgm_params = _parse_lgm_params_from_calibration_xml_text(xml["calibration.xml"], ccy_key=model_ccy)
            param_source = "calibration"
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
            else:
                unsupported.append(t)
        ccy_set: set[str] = {snapshot.config.base_currency.upper()}
        for t in snapshot.portfolio.trades:
            if isinstance(t.product, IRS):
                ccy_set.add(t.product.ccy.upper())
            if isinstance(t.product, FXForward):
                ccy_set.add(t.product.pair[:3].upper())
                ccy_set.add(t.product.pair[3:].upper())
        return trade_specs, unsupported, ccy_set

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
        zero_curves = overlay["zero"]
        fwd_curves_raw = overlay.get("fwd", {})
        hazards = overlay["hazard"]
        recoveries = overlay["recovery"]

        trade_specs, unsupported, ccy_set = self._classify_portfolio_trades(snapshot, mapped)

        for c in list(ccy_set):
            zero_curves.setdefault(c, [(0.0, 0.02), (max(float(snapshot.config.horizon_years), 1.0), 0.02)])

        curve_payload = self._load_ore_output_curves(snapshot, mapped, trade_specs)
        if curve_payload is not None:
            discount_curves = curve_payload.discount_curves
            forward_curves = curve_payload.forward_curves
            forward_curves_by_tenor = curve_payload.forward_curves_by_tenor
            xva_discount_curve = curve_payload.xva_discount_curve
            funding_borrow_curve = curve_payload.funding_borrow_curve
            funding_lend_curve = curve_payload.funding_lend_curve
            curve_source = "ore_output_curves"
        else:
            discount_curves = {}
            forward_curves = {}
            forward_curves_by_tenor = {}
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
                xva_discount_curve = fitted_curves.xva_discount_curve
                funding_borrow_curve = fitted_curves.funding_borrow_curve
                funding_lend_curve = fitted_curves.funding_lend_curve
                curve_source = "ore_quote_fit"
            else:
                for ccy, pts in zero_curves.items():
                    by_time: Dict[float, List[float]] = {}
                    for t, r in pts:
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
                curve_source = "market_overlay"

        (
            discount_curves,
            forward_curves,
            forward_curves_by_tenor,
            xva_discount_curve,
        ) = _apply_curve_node_shocks(
            snapshot,
            discount_curves,
            forward_curves,
            forward_curves_by_tenor,
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

        return _PythonLgmInputs(
            times=times,
            valuation_times=valuation_times,
            observation_times=observation_times,
            observation_closeout_times=observation_closeout_times,
            discount_curves=discount_curves,
            forward_curves=forward_curves,
            forward_curves_by_tenor=forward_curves_by_tenor,
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
        if str(snapshot.config.params.get("python.use_ore_output_curves", "Y")).strip().upper() in ("N", "FALSE", "0"):
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
            for spec in trade_specs:
                if spec.kind != "IRS" or spec.legs is None:
                    continue
                index_name = str(spec.legs.get("float_index", "")).strip()
                tenor_key = str(spec.legs.get("float_index_tenor", "")).upper()
                if not index_name or not tenor_key:
                    continue
                requested_columns.add(index_name)
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
            for ccy, tenor_map in forward_specs.items():
                tenor_curves: Dict[str, Callable[[float], float]] = {}
                for tenor_key, column_name in tenor_map.items():
                    tenor_curves[tenor_key] = self._curve_from_column(curve_data, column_name)
                forward_curves_by_tenor[ccy] = tenor_curves
                if tenor_curves:
                    preferred = "6M" if "6M" in tenor_curves else sorted(tenor_curves)[0]
                    forward_curves[ccy] = tenor_curves[preferred]

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
                    if _quote_matches_discount_curve(str(q["key"]), ccy, source_column)
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
            needed_tenors: Dict[str, set[str]] = {}
            for spec in trade_specs:
                if spec.kind != "IRS" or spec.legs is None:
                    continue
                tenor = str(spec.legs.get("float_index_tenor", "")).upper()
                if tenor:
                    needed_tenors.setdefault(spec.ccy.upper(), set()).add(tenor)
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
                xva_discount_curve=base_curve,
                funding_borrow_curve=None,
                funding_lend_curve=None,
            )
        except Exception:
            return None

    def _python_lgm_rng_mode(self, snapshot: XVASnapshot) -> str:
        """Determine the RNG mode for LGM path simulation from config params.

        Reads ``python.lgm_rng_mode`` from the snapshot config and returns a
        normalised lower-case string (``"numpy"`` or ``"ore_parity"``).
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
            if spec.kind != "IRS" or spec.legs is None:
                continue
            legs = spec.legs
            for key in ("fixed_pay_time", "float_pay_time", "float_fixing_time"):
                vals = np.asarray(legs.get(key, np.array([], dtype=float)), dtype=float)
                if vals.size == 0:
                    continue
                extras.append(vals[vals >= 0.0])
        if extras:
            out = np.unique(np.concatenate([out, *extras]))
        if out.size == 0 or out[0] > 0.0:
            out = np.unique(np.concatenate([np.array([0.0]), out]))
        return out

    def _build_lgm_rng(self, seed: int, rng_mode: str):
        """Construct the random number generator for LGM path simulation.

        Supports ``"numpy"`` (NumPy default RNG, time-major draw order) and
        ``"ore_parity"`` (ORE Gaussian RNG producing path-major draws that
        match the ORE C++ simulation grid exactly).  Raises
        :class:`EngineRunError` for any other mode string.
        """
        if rng_mode == "numpy":
            return np.random.default_rng(seed), "time_major"
        if rng_mode == "ore_parity":
            return self._lgm_mod.make_ore_gaussian_rng(seed), "ore_path_major"
        raise EngineRunError(
            f"Unsupported PythonLgmAdapter RNG mode '{rng_mode}'. Use 'numpy' or 'ore_parity'."
        )

    def _day_counter_from_sim_xml(self, sim_xml: str) -> str:
        """Return the normalised day-counter string from simulation.xml text."""
        try:
            root = ET.fromstring(sim_xml)
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

    def _price_fx_forward(self, trade: Trade, inputs: _PythonLgmInputs, n_times: int, n_paths: int) -> np.ndarray:
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
        spot = _spot_from_quotes(p.pair, inputs, default=1.0)
        p_dom = inputs.discount_curves[dom]
        p_for = inputs.discount_curves[for_ccy]
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
        hybrid = self._fx_utils.build_two_ccy_hybrid(pair=pair, ir_specs=ir_specs, fx_vol=0.15)
        sim = hybrid.simulate_paths(
            times=inputs.times,
            n_paths=n_paths,
            log_s0={pair: float(np.log(max(spot, 1.0e-12)))},
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
        vals = np.zeros((n_times, n_paths), dtype=float)
        for i, t in enumerate(inputs.times):
            vals[i, :] = self._fx_utils.fx_forward_npv(
                hybrid=hybrid,
                fx_def=fx_def,
                t=float(t),
                s_t=s_t[i, :],
                x_dom_t=x_dom_t[i, :],
                x_for_t=x_for_t[i, :],
                p0_dom=p_dom,
                p0_for=p_for,
            )
        return vals

    def _build_irs_legs(self, trade: Trade, mapped: MappedInputs, snapshot: XVASnapshot) -> Dict[str, np.ndarray]:
        ore_path_txt = getattr(snapshot.config.source_meta, "path", "") or ""
        if ore_path_txt:
            try:
                ore_path = Path(ore_path_txt).resolve()
                output_dir = (ore_path.parent.parent / snapshot.config.params.get("outputPath", "Output")).resolve()
                flows_csv = output_dir / "flows.csv"
                if flows_csv.exists():
                    asof = _normalize_asof_date(snapshot.config.asof)
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
                            with _tmp_xml_path(sim_xml) as path:
                                legs["node_tenors"] = self._irs_utils.load_simulation_yield_tenors(path)
                        except Exception:
                            pass
                    idx = str(trade.additional_fields.get("index", "")).upper()
                    if idx:
                        legs.setdefault("float_index", idx)
                        if "float_index_tenor" not in legs or not str(legs.get("float_index_tenor", "")).strip():
                            legs["float_index_tenor"] = idx.split("-")[-1].upper() if "-" in idx else ""
                    return self._apply_historical_fixings_to_legs(snapshot, trade, legs)
            except Exception as exc:
                import warnings as _warnings
                _warnings.warn(
                    f"Failed to load legs from flows.csv for trade {trade.trade_id}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
        portfolio_xml = mapped.xml_buffers.get("portfolio.xml")
        if portfolio_xml:
            asof = _normalize_asof_date(snapshot.config.asof)
            try:
                with _tmp_xml_path(portfolio_xml) as path:
                    legs = self._irs_utils.load_swap_legs_from_portfolio(path, trade.trade_id, asof)
                sim_xml = mapped.xml_buffers.get("simulation.xml")
                if sim_xml:
                    try:
                        with _tmp_xml_path(sim_xml) as path:
                            legs["node_tenors"] = self._irs_utils.load_simulation_yield_tenors(path)
                    except Exception:
                        pass
                return self._apply_historical_fixings_to_legs(snapshot, trade, legs)
            except Exception as exc:
                import warnings as _warnings
                _warnings.warn(
                    f"Failed to load legs from portfolio.xml for trade {trade.trade_id}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
        return self._apply_historical_fixings_to_legs(snapshot, trade, _build_irs_legs_from_trade(trade))

    def _apply_historical_fixings_to_legs(
        self, snapshot: XVASnapshot, trade: Trade, legs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        fix_t = np.asarray(legs.get("float_fixing_time", []), dtype=float)
        if fix_t.size == 0:
            return legs
        asof = _normalize_asof_date(snapshot.config.asof)
        fixings = {
            (str(p.index).upper(), _normalize_asof_date(p.date)): float(p.value)
            for p in snapshot.fixings.points
        }
        p = trade.product
        assert isinstance(p, IRS)
        index_name = str(legs.get("float_index", trade.additional_fields.get("index", _default_index_for_ccy(p.ccy)))).upper()
        coupons = np.asarray(legs.get("float_coupon", np.zeros_like(fix_t)), dtype=float).copy()
        fixed_mask = np.asarray(legs.get("float_is_historically_fixed", np.zeros(fix_t.shape, dtype=bool)), dtype=bool).copy()
        for i, ft in enumerate(fix_t):
            fixing_date = _date_from_time(asof, float(ft))
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
        quoted_coupon = np.asarray(legs.get("float_coupon", np.zeros_like(s)), dtype=float)
        fixed_mask = np.asarray(legs.get("float_is_historically_fixed", np.zeros(s.shape, dtype=bool)), dtype=bool)

        n_cf = s.size
        n_paths = x_paths_on_sim_grid.shape[1]
        out = np.zeros((n_cf, n_paths), dtype=float)

        for i in range(n_cf):
            if fixed_mask[i]:
                out[i, :] = quoted_coupon[i] + float(spr[i])
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
            out[i, :] = fwd_path + float(spr[i])
        return out

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
        epe_by_ns_paths: Dict[str, np.ndarray] = {}
        ene_by_ns_paths: Dict[str, np.ndarray] = {}
        valuation_epe_by_ns_paths: Dict[str, np.ndarray] = {}
        valuation_ene_by_ns_paths: Dict[str, np.ndarray] = {}
        xva_deflated_by_ns: Dict[str, bool] = {}
        pv_total = 0.0
        npv_cube_payload: Dict[str, Dict[str, object]] = {}
        exposure_cube_payload: Dict[str, Dict[str, object]] = {}

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
            v_xva_obs = v_xva[obs_closeout_idx, :]
            valuation_epe = np.mean(np.maximum(v_xva_val, 0.0), axis=1)
            valuation_ene = np.mean(np.maximum(-v_xva_val, 0.0), axis=1)
            epe = np.mean(np.maximum(v_xva_obs, 0.0), axis=1)
            ene = np.mean(np.maximum(-v_xva_obs, 0.0), axis=1)
            valuation_epe_by_ns_paths[ns] = valuation_epe_by_ns_paths.get(ns, np.zeros_like(valuation_epe)) + valuation_epe
            valuation_ene_by_ns_paths[ns] = valuation_ene_by_ns_paths.get(ns, np.zeros_like(valuation_ene)) + valuation_ene
            epe_by_ns_paths[ns] = epe_by_ns_paths.get(ns, np.zeros_like(epe)) + epe
            ene_by_ns_paths[ns] = ene_by_ns_paths.get(ns, np.zeros_like(ene)) + ene
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
            }

        exposure_by_ns = {ns: float(vals[0]) for ns, vals in epe_by_ns_paths.items()}
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

        reports = {
            "xva": [{"Metric": k, "Value": v} for k, v in xva_by_metric.items()],
            "exposure": [{"NettingSetId": k, "EPE": v} for k, v in exposure_by_ns.items()],
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
            "coverage": {
                "python_trades": len(npv_by_trade),
                "fallback_trades": len(fallback_trades),
                "unsupported": [f"{t.trade_id}:{t.trade_type}" for t in unsupported],
                "python_notional_pct": float(py_notional / total_notional),
                "fallback_notional_pct": float(fallback_notional / total_notional),
            },
            "input_provenance": dict(inputs.input_provenance),
            "model_currency": inputs.model_ccy,
            "grid_size": int(times.size),
            "valuation_grid_size": int(valuation_times.size),
            "observation_grid_size": int(obs_times.size),
            "closeout_grid_size": int(np.unique(inputs.observation_closeout_times).size),
            "path_count": int(snapshot.config.num_paths),
            "mpor_enabled": bool(inputs.mpor.enabled),
            "mpor_days": int(inputs.mpor.mpor_days),
            "mpor_mode": "sticky",
            "mpor_source": inputs.mpor.source,
        }
        if fallback is not None:
            metadata["fallback_adapter"] = "ore-swig"

        return XVAResult(
            run_id=run_id,
            pv_total=pv_total,
            xva_total=float(sum(xva_by_metric.values())),
            xva_by_metric=xva_by_metric,
            exposure_by_netting_set=exposure_by_ns,
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
            metadata={"adapter": "ore-swig", "module": getattr(self._module, "__name__", type(self._module).__name__)},
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
    fwd: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    fx: Dict[str, float] = {}
    hazard: Dict[str, List[Tuple[float, float]]] = {}
    recovery: Dict[str, float] = {}
    for q in raw_quotes:
        key = str(q.key).strip()
        up = key.upper()
        val = float(q.value)
        parts = up.split("/")
        if len(parts) >= 3 and parts[0] == "FX":
            fx[parts[1] + parts[2]] = val
            continue
        if len(parts) >= 4 and parts[0] == "ZERO" and parts[1] == "RATE":
            ccy = parts[2]
            tenor = parts[3]
            try:
                t = _parse_tenor_to_years(tenor)
            except Exception:
                continue
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
        if len(parts) >= 5 and parts[0] == "RECOVERY_RATE":
            cpty = parts[2]
            recovery[cpty] = val
            continue
    return {"zero": zero, "fwd": fwd, "fx": fx, "hazard": hazard, "recovery": recovery}


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


def _build_irs_legs_from_trade(trade: Trade) -> Dict[str, np.ndarray]:
    """Last-resort skeleton leg builder: assumes 6M fixed / 3M float, no historical fixings.

    This path is reached only when neither flows.csv nor portfolio.xml loading
    succeeded.  Results will be approximate — calendar, day-count and index
    conventions are hard-coded.  Always prefer loading legs from ORE output
    artifacts or a portfolio XML.
    """
    import warnings as _warnings
    _warnings.warn(
        f"Using skeleton leg schedule for trade {trade.trade_id} "
        "(no flows.csv or portfolio.xml available). "
        "Leg conventions are approximate (6M fixed / 3M float, A365F).",
        UserWarning,
        stacklevel=3,
    )
    p = trade.product
    if not isinstance(p, IRS):
        raise EngineRunError(f"Cannot build IRS legs for non-IRS trade {trade.trade_id}")
    mat = max(float(p.maturity_years), 0.25)
    fixed_pay = np.arange(0.5, mat + 1.0e-12, 0.5, dtype=float)
    float_pay = np.arange(0.25, mat + 1.0e-12, 0.25, dtype=float)
    float_start = np.maximum(float_pay - 0.25, 0.0)
    float_end = float_pay.copy()
    fixing = float_start - (2.0 / 365.25)
    fixed_sign = -1.0 if p.pay_fixed else 1.0
    float_sign = -fixed_sign
    float_index = str(trade.additional_fields.get("index", _default_index_for_ccy(p.ccy))).upper()
    float_index_tenor = float_index.split("-")[-1].upper() if "-" in float_index else ""
    return {
        "fixed_pay_time": fixed_pay,
        "fixed_accrual": np.full(fixed_pay.shape, 0.5),
        "fixed_rate": np.full(fixed_pay.shape, float(p.fixed_rate)),
        "fixed_notional": np.full(fixed_pay.shape, float(p.notional)),
        "fixed_sign": np.full(fixed_pay.shape, fixed_sign),
        "fixed_amount": np.full(fixed_pay.shape, fixed_sign * float(p.notional) * float(p.fixed_rate) * 0.5),
        "float_pay_time": float_pay,
        "float_start_time": float_start,
        "float_end_time": float_end,
        "float_accrual": np.full(float_pay.shape, 0.25),
        "float_notional": np.full(float_pay.shape, float(p.notional)),
        "float_sign": np.full(float_pay.shape, float_sign),
        "float_spread": np.zeros(float_pay.shape),
        "float_coupon": np.zeros(float_pay.shape),
        "float_amount": np.zeros(float_pay.shape),
        "float_fixing_time": fixing,
        "float_is_historically_fixed": np.zeros(float_pay.shape, dtype=bool),
        "float_index": float_index,
        "float_index_tenor": float_index_tenor,
    }


def _counterparty_for_netting(snapshot: XVASnapshot, netting_set: str) -> str:
    for t in snapshot.portfolio.trades:
        if t.netting_set == netting_set:
            return t.counterparty
    return netting_set


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
    xva_discount_curve: Optional[Callable[[float], float]],
) -> Tuple[
    Dict[str, Callable[[float], float]],
    Dict[str, Callable[[float], float]],
    Dict[str, Dict[str, Callable[[float], float]]],
    Optional[Callable[[float], float]],
]:
    specs = snapshot.config.params.get("python.curve_node_shocks")
    if not isinstance(specs, dict):
        return discount_curves, forward_curves, forward_curves_by_tenor, xva_discount_curve

    discount_specs = specs.get("discount", {}) if isinstance(specs.get("discount", {}), dict) else {}
    forward_specs = specs.get("forward", {}) if isinstance(specs.get("forward", {}), dict) else {}

    discount_curves = dict(discount_curves)
    forward_curves = dict(forward_curves)
    forward_curves_by_tenor = {ccy: dict(v) for ccy, v in forward_curves_by_tenor.items()}

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
            if ccy in forward_curves:
                preferred = "6M" if "6M" in tenor_curves else (sorted(tenor_curves)[0] if tenor_curves else "")
                if preferred and preferred == tenor:
                    forward_curves[ccy] = bumped
        forward_curves_by_tenor[ccy] = tenor_curves

    return discount_curves, forward_curves, forward_curves_by_tenor, xva_discount_curve


def _quote_matches_discount_curve(key: str, ccy: str, source_column: str) -> bool:
    parts = str(key).strip().upper().split("/")
    if len(parts) < 3 or parts[2] != ccy.upper():
        return False
    family = _curve_family_from_source_column(source_column)
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
