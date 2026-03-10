from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import importlib
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
    Trade,
    XVASnapshot,
)
from .exceptions import EngineRunError
from .mapper import MappedInputs, _default_index_for_ccy, build_input_parameters, map_snapshot
from .results import CubeAccessor, XVAResult


class RunnerAdapter(Protocol):
    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult: ...


@dataclass
class SessionState:
    snapshot: XVASnapshot
    snapshot_key: str
    mapped_inputs: MappedInputs
    rebuild_counts: Dict[str, int]


class XVAEngine:
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


class XVASession:
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


@dataclass(frozen=True)
class _TradeSpec:
    trade: Trade
    kind: str
    notional: float
    ccy: str
    legs: Dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class _PythonLgmInputs:
    times: np.ndarray
    observation_times: np.ndarray
    discount_curves: Dict[str, Callable[[float], float]]
    forward_curves: Dict[str, Callable[[float], float]]
    forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]]
    xva_discount_curve: Optional[Callable[[float], float]]
    funding_borrow_curve: Optional[Callable[[float], float]]
    funding_lend_curve: Optional[Callable[[float], float]]
    hazard_times: Dict[str, np.ndarray]
    hazard_rates: Dict[str, np.ndarray]
    recovery_rates: Dict[str, float]
    lgm_params: Dict[str, object]
    model_ccy: str
    seed: int
    fx_spots: Dict[str, float]
    trade_specs: Tuple[_TradeSpec, ...]
    unsupported: Tuple[Trade, ...]
    input_provenance: Dict[str, str]


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

    def _ensure_py_lgm_imports(self) -> None:
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

    def _extract_inputs(self, snapshot: XVASnapshot, mapped: MappedInputs) -> _PythonLgmInputs:
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

        overlay = _parse_market_overlay(snapshot.market.raw_quotes)
        fx_spots = overlay["fx"]
        zero_curves = overlay["zero"]
        fwd_curves_raw = overlay.get("fwd", {})
        hazards = overlay["hazard"]
        recoveries = overlay["recovery"]

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

        ccy_set = {snapshot.config.base_currency.upper()}
        for t in snapshot.portfolio.trades:
            if isinstance(t.product, IRS):
                ccy_set.add(t.product.ccy.upper())
            if isinstance(t.product, FXForward):
                ccy_set.add(t.product.pair[:3].upper())
                ccy_set.add(t.product.pair[3:].upper())
        for c in list(ccy_set):
            zero_curves.setdefault(c, [(0.0, 0.02), (max(float(snapshot.config.horizon_years), 1.0), 0.02)])

        curve_payload = self._load_ore_output_curves(snapshot, mapped, trade_specs)
        if curve_payload is not None:
            (
                discount_curves,
                forward_curves,
                forward_curves_by_tenor,
                xva_discount_curve,
                funding_borrow_curve,
                funding_lend_curve,
            ) = curve_payload
            curve_source = "ore_output_curves"
        else:
            discount_curves = {}
            forward_curves = {}
            forward_curves_by_tenor = {}
            xva_discount_curve = None
            funding_borrow_curve = None
            funding_lend_curve = None
            for ccy, pts in zero_curves.items():
                by_time: Dict[float, List[float]] = {}
                for t, r in pts:
                    by_time.setdefault(float(t), []).append(float(r))
                dedup: Dict[float, float] = {}
                for t, vals in by_time.items():
                    # Prefer economically plausible nodes over flat placeholder quotes
                    # by taking the value closest to par (smallest absolute rate).
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

        calibrated_specs: List[_TradeSpec] = []
        for spec in trade_specs:
            if spec.kind != "IRS" or spec.legs is None:
                calibrated_specs.append(spec)
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

        times = self._augment_exposure_grid_with_trade_dates(times, trade_specs)
        if times.size < 2:
            times = np.array([0.0, max(float(snapshot.config.horizon_years), 1.0)], dtype=float)

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

        return _PythonLgmInputs(
            times=times,
            observation_times=observation_times,
            discount_curves=discount_curves,
            forward_curves=forward_curves,
            forward_curves_by_tenor=forward_curves_by_tenor,
            xva_discount_curve=xva_discount_curve,
            funding_borrow_curve=funding_borrow_curve,
            funding_lend_curve=funding_lend_curve,
            hazard_times=hazard_times,
            hazard_rates=hazard_rates,
            recovery_rates=recovery_rates,
            lgm_params=lgm_params,
            model_ccy=model_ccy,
            seed=int(snapshot.config.runtime.simulation.seed) if snapshot.config.runtime else 42,
            fx_spots=fx_spots,
            trade_specs=tuple(trade_specs),
            unsupported=tuple(unsupported),
            input_provenance={
                "model_params": param_source,
                "market": curve_source,
                "grid": "xml+trade_dates" if grid_source == "xml" else grid_source,
                "portfolio": "dataclass",
            },
        )

    def _load_ore_output_curves(
        self,
        snapshot: XVASnapshot,
        mapped: MappedInputs,
        trade_specs: Sequence[_TradeSpec],
    ) -> Optional[
        Tuple[
            Dict[str, Callable[[float], float]],
            Dict[str, Callable[[float], float]],
            Dict[str, Dict[str, Callable[[float], float]]],
            Optional[Callable[[float], float]],
            Optional[Callable[[float], float]],
            Optional[Callable[[float], float]],
        ]
    ]:
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
            sim_root = ET.fromstring(simulation_xml)
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
            model_day_counter = self._ore_snapshot_mod._normalize_day_counter_name(
                (sim_root.findtext("./DayCounter") or "A365F").strip()
            )
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

            xva_curve_name = None
            runtime = snapshot.config.runtime
            if runtime is not None:
                xva_ccy = snapshot.config.base_currency.upper()
                try:
                    xva_curve_name = xva_discount_meta.get(xva_ccy, {}).get("source_column")
                except Exception:
                    xva_curve_name = None
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
                _, times, dfs = curve_data[meta["source_column"]]
                discount_curves[ccy] = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                    list(zip(times, dfs))
                )

            forward_curves: Dict[str, Callable[[float], float]] = {}
            forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]] = {}
            for ccy, tenor_map in forward_specs.items():
                tenor_curves: Dict[str, Callable[[float], float]] = {}
                for tenor_key, column_name in tenor_map.items():
                    _, times, dfs = curve_data[column_name]
                    tenor_curves[tenor_key] = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                        list(zip(times, dfs))
                    )
                forward_curves_by_tenor[ccy] = tenor_curves
                if tenor_curves:
                    preferred = "6M" if "6M" in tenor_curves else sorted(tenor_curves)[0]
                    forward_curves[ccy] = tenor_curves[preferred]

            for ccy, disc in discount_curves.items():
                forward_curves.setdefault(ccy, disc)
                forward_curves_by_tenor.setdefault(ccy, {})
            xva_discount_curve = None
            if xva_curve_name and xva_curve_name in curve_data:
                _, times, dfs = curve_data[xva_curve_name]
                xva_discount_curve = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                    list(zip(times, dfs))
                )
            funding_borrow_curve = None
            if borrow_curve_name and str(borrow_curve_name) in curve_data:
                _, times, dfs = curve_data[str(borrow_curve_name)]
                funding_borrow_curve = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                    list(zip(times, dfs))
                )
            funding_lend_curve = None
            if lend_curve_name and str(lend_curve_name) in curve_data:
                _, times, dfs = curve_data[str(lend_curve_name)]
                funding_lend_curve = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                    list(zip(times, dfs))
                )

            return (
                discount_curves,
                forward_curves,
                forward_curves_by_tenor,
                xva_discount_curve,
                funding_borrow_curve,
                funding_lend_curve,
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

    def _python_lgm_rng_mode(self, snapshot: XVASnapshot) -> str:
        return str(snapshot.config.params.get("python.lgm_rng_mode", "numpy")).strip().lower()

    def _is_ore_case_snapshot(self, snapshot: XVASnapshot) -> bool:
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
        if rng_mode == "numpy":
            return np.random.default_rng(seed), "time_major"
        if rng_mode == "ore_parity":
            return self._lgm_mod.make_ore_gaussian_rng(seed), "ore_path_major"
        raise EngineRunError(
            f"Unsupported PythonLgmAdapter RNG mode '{rng_mode}'. Use 'numpy' or 'ore_parity'."
        )

    def _price_fx_forward(self, trade: Trade, inputs: _PythonLgmInputs, n_times: int, n_paths: int) -> np.ndarray:
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
                        try:
                            sim_root = ET.fromstring(sim_xml)
                            model_day_counter = self._ore_snapshot_mod._normalize_day_counter_name(
                                (sim_root.findtext("./DayCounter") or "A365F").strip()
                            )
                        except Exception:
                            pass
                    legs = self._irs_utils.load_ore_legs_from_flows(
                        str(flows_csv),
                        trade_id=trade.trade_id,
                        asof_date=asof,
                        time_day_counter=model_day_counter,
                    )
                    if sim_xml:
                        try:
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", encoding="utf-8", delete=True) as stmp:
                                stmp.write(sim_xml)
                                stmp.flush()
                                legs["node_tenors"] = self._irs_utils.load_simulation_yield_tenors(stmp.name)
                        except Exception:
                            pass
                    return self._apply_historical_fixings_to_legs(snapshot, trade, legs)
            except Exception:
                pass
        portfolio_xml = mapped.xml_buffers.get("portfolio.xml")
        if portfolio_xml:
            asof = _normalize_asof_date(snapshot.config.asof)
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", encoding="utf-8", delete=True) as tmp:
                    tmp.write(portfolio_xml)
                    tmp.flush()
                    legs = self._irs_utils.load_swap_legs_from_portfolio(tmp.name, trade.trade_id, asof)
                sim_xml = mapped.xml_buffers.get("simulation.xml")
                if sim_xml:
                    try:
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", encoding="utf-8", delete=True) as stmp:
                            stmp.write(sim_xml)
                            stmp.flush()
                            legs["node_tenors"] = self._irs_utils.load_simulation_yield_tenors(stmp.name)
                    except Exception:
                        pass
                return self._apply_historical_fixings_to_legs(snapshot, trade, legs)
            except Exception:
                pass
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
        obs_times = inputs.observation_times
        obs_idx = np.searchsorted(times, obs_times)
        if (
            obs_idx.size != obs_times.size
            or np.any(obs_idx >= times.size)
            or np.any(np.abs(times[obs_idx] - obs_times) > 1.0e-10)
        ):
            raise EngineRunError("Observation times are not aligned with pricing grid after augmentation")
        df_base = inputs.discount_curves[snapshot.config.base_currency.upper()]
        df_vec = np.asarray([df_base(float(t)) for t in times], dtype=float)
        obs_df_vec = df_vec[obs_idx]
        epe_by_ns_paths: Dict[str, np.ndarray] = {}
        ene_by_ns_paths: Dict[str, np.ndarray] = {}
        xva_deflated_by_ns: Dict[str, bool] = {}
        pv_total = 0.0
        npv_cube_payload: Dict[str, Dict[str, object]] = {}
        exposure_cube_payload: Dict[str, Dict[str, object]] = {}

        for spec in inputs.trade_specs:
            v = npv_by_trade.get(spec.trade.trade_id)
            if v is None:
                continue
            pv_total += float(np.mean(v[0, :]))
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
            v_xva_obs = v_xva[obs_idx, :]
            epe = np.mean(np.maximum(v_xva_obs, 0.0), axis=1)
            ene = np.mean(np.maximum(-v_xva_obs, 0.0), axis=1)
            epe_by_ns_paths[ns] = epe_by_ns_paths.get(ns, np.zeros_like(epe)) + epe
            ene_by_ns_paths[ns] = ene_by_ns_paths.get(ns, np.zeros_like(ene)) + ene
            xva_deflated_by_ns[ns] = xva_deflated_by_ns.get(ns, True) and xva_deflated
            npv_cube_payload[spec.trade.trade_id] = {
                "times": times.tolist(),
                "npv_mean": np.mean(v, axis=1).tolist(),
                "npv_xva_mean": np.mean(v_xva, axis=1).tolist(),
                "epe": epe.tolist(),
                "ene": ene.tolist(),
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
            q_c = self._irs_utils.survival_probability_from_hazard(
                obs_times,
                inputs.hazard_times.get(cpty, np.array([1.0, 5.0])),
                inputs.hazard_rates.get(cpty, np.array([0.02, 0.02])),
            )
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
            exposure_cube_payload[ns] = {"times": obs_times.tolist(), "epe": epe_vec.tolist(), "ene": ene_vec.tolist()}

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
        if reports["xva"]:
            reports["xva"].append({"Metric": "FBA", "Value": fba_total})
            reports["xva"].append({"Metric": "FCA", "Value": fca_total})
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
            "observation_grid_size": int(obs_times.size),
            "path_count": int(snapshot.config.num_paths),
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
        candidates = [
            (input_parameters, "", 31, False, True),
            (input_parameters,),
        ]
        for args in candidates:
            try:
                return self._ore_app_cls(*args)
            except Exception:
                continue
        raise EngineRunError("Failed to construct OREApp from InputParameters")

    def _invoke_run(self, app: Any, market_data: Sequence[str], fixing_data: Sequence[str]) -> None:
        try:
            app.run(list(market_data), list(fixing_data))
            return
        except TypeError:
            pass
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

    vol_node = node.find("./Volatility")
    rev_node = node.find("./Reversion")
    trans_node = node.find("./ParameterTransformation")
    if vol_node is None or rev_node is None or trans_node is None:
        raise EngineRunError("simulation.xml LGM node missing Volatility/Reversion/ParameterTransformation")

    return {
        "alpha_times": _parse_float_grid(vol_node.findtext("./TimeGrid")),
        "alpha_values": _parse_float_grid(vol_node.findtext("./InitialValue")),
        "kappa_times": _parse_float_grid(rev_node.findtext("./TimeGrid")),
        "kappa_values": _parse_float_grid(rev_node.findtext("./InitialValue")),
        "shift": float((trans_node.findtext("./ShiftHorizon") or "0").strip()),
        "scaling": float((trans_node.findtext("./Scaling") or "1").strip()),
    }


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
    vol_node = node.find("./Volatility")
    rev_node = node.find("./Reversion")
    trans_node = node.find("./ParameterTransformation")
    if vol_node is None or rev_node is None or trans_node is None:
        raise EngineRunError("calibration.xml LGM node missing Volatility/Reversion/ParameterTransformation")
    return {
        "alpha_times": _parse_float_grid(vol_node.findtext("./TimeGrid")),
        "alpha_values": _parse_float_grid(vol_node.findtext("./InitialValue")),
        "kappa_times": _parse_float_grid(rev_node.findtext("./TimeGrid")),
        "kappa_values": _parse_float_grid(rev_node.findtext("./InitialValue")),
        "shift": float((trans_node.findtext("./ShiftHorizon") or "0").strip()),
        "scaling": float((trans_node.findtext("./Scaling") or "1").strip()),
    }


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
