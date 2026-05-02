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
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
import importlib
import math
import os
import re
from pathlib import Path
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from pythonore.domain.dataclasses import (
    BermudanSwaption,
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
from pythonore.runtime.core import RunnerAdapter, SessionState, _ConsoleRunReporter, XVAEngine, XVASession
from pythonore.runtime.lgm import inputs as lgm_inputs
from pythonore.runtime.lgm import market as lgm_market
from pythonore.runtime.swig import ORESwigAdapter
from pythonore.runtime.toy import DeterministicToyAdapter, _toy_trade_numbers
from pythonore.runtime.results import CubeAccessor, XVAResult, xva_total_from_metrics
from pythonore.repo_paths import find_engine_repo_root


@dataclass
class _CurveBundle:
    """Curve artefacts produced by the market-curve building paths."""
    discount_curves: dict[str, Callable[[float], float]]
    discount_curve_dates: dict[str, tuple[str, ...]]
    discount_curve_dfs: dict[str, tuple[float, ...]]
    forward_curves: dict[str, Callable[[float], float]]
    forward_curves_by_tenor: dict[str, dict[str, Callable[[float], float]]]
    forward_curves_by_name: dict[str, Callable[[float], float]]
    xva_discount_curve: Callable[[float], float] | None
    funding_borrow_curve: Callable[[float], float] | None
    funding_lend_curve: Callable[[float], float] | None


def _normalize_forward_tenor_family(tenor: str) -> str:
    token = str(tenor or "").strip().upper()
    if token in {"", "0D", "1D", "ON", "O/N", "ESTR", "ESTER", "EONIA", "SOFR", "SONIA", "SARON", "TONAR"}:
        return "1D"
    return token


def _normalize_curve_lookup_key(name: str) -> str:
    token = str(name or "").strip().upper().replace("O/N", "ON")
    return token.replace("ESTER", "ESTR")


def _forward_index_family(index_name: str, swap_index_forward_tenors: Mapping[str, str] | None = None) -> str:
    key = str(index_name or "").strip().upper()
    if key in {"USD-SIFMA", "USD-BMA", "USD-SIFMA-1W", "USD-SIFMA-7D", "USD-BMA-1W", "USD-BMA-7D"}:
        return "1D"
    mapped = str((swap_index_forward_tenors or {}).get(key, "")).strip().upper()
    if mapped:
        return _normalize_forward_tenor_family(mapped)
    tenor_match = re.search(r"(\d+[YMWD])$", key)
    if tenor_match is not None:
        return _normalize_forward_tenor_family(tenor_match.group(1).upper())
    if any(tag in key for tag in ("EONIA", "ESTR", "ESTER", "FEDFUNDS", "SOFR", "SONIA", "SARON", "TOIS", "TONAR")):
        return "1D"
    return ""


def _is_bma_sifma_index(index_name: str) -> bool:
    key = str(index_name or "").strip().upper()
    return bool(re.search(r"(^|-)SIFMA($|-)|(^|-)BMA($|-)", key))


def _default_index_for_ccy_tenor(ccy: str, tenor: str) -> str:
    base = _default_index_for_ccy(ccy).upper()
    tenor_key = str(tenor or "").strip().upper()
    if not tenor_key:
        return base
    return re.sub(r"(\d+[YMWD])$", tenor_key, base)


def _trade_curve_need_signature(
    trade_specs: Sequence[Any],
    swap_index_forward_tenors: Mapping[str, str],
) -> tuple[tuple[object, ...], tuple[tuple[str, str], ...]]:
    spec_sig: list[tuple[object, ...]] = []
    for spec in trade_specs:
        kind = str(getattr(spec, "kind", "")).upper()
        ccy = str(getattr(spec, "ccy", "")).upper()
        legs = getattr(spec, "legs", None)
        if kind == "IRS" and isinstance(legs, dict):
            spec_sig.append(
                (
                    kind,
                    ccy,
                    str(legs.get("float_index", "")).upper(),
                    str(legs.get("float_index_tenor", "")).upper(),
                )
            )
            continue
        if kind == "RATESWAP" and isinstance(legs, dict):
            leg_sigs: list[tuple[str, str, str, str]] = []
            for leg in legs.get("rate_legs", []):
                leg_sigs.append(
                    (
                        str(leg.get("ccy", ccy)).upper(),
                        str(leg.get("index_name", "")).upper(),
                        str(leg.get("index_name_1", "")).upper(),
                        str(leg.get("index_name_2", "")).upper(),
                    )
                )
            spec_sig.append((kind, ccy, tuple(sorted(leg_sigs))))
    swap_sig = tuple(sorted((str(k).upper(), str(v).upper()) for k, v in swap_index_forward_tenors.items()))
    return tuple(spec_sig), swap_sig


@lru_cache(maxsize=256)
def _derive_curve_needs_from_signature(
    spec_sig: tuple[tuple[object, ...], ...],
    swap_sig: tuple[tuple[str, str], ...],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    swap_index_forward_tenors = {k: v for k, v in swap_sig}
    needed_tenors: Dict[str, set[str]] = {}
    needed_forward_names: Dict[str, set[str]] = {}
    for spec in spec_sig:
        kind = str(spec[0]).upper()
        ccy = str(spec[1]).upper()
        if kind == "IRS":
            index_name = str(spec[2]).upper()
            tenor = str(spec[3]).upper()
            if index_name:
                needed_forward_names.setdefault(ccy, set()).add(index_name)
            if tenor:
                needed_tenors.setdefault(ccy, set()).add(tenor)
        elif kind == "RATESWAP":
            for leg_ccy, *index_names in spec[2]:
                curve_ccy = str(leg_ccy or ccy).upper()
                for index_name in index_names:
                    if not index_name:
                        continue
                    needed_forward_names.setdefault(curve_ccy, set()).add(index_name)
                    tenor = swap_index_forward_tenors.get(index_name, "")
                    if tenor:
                        needed_tenors.setdefault(curve_ccy, set()).add(str(tenor).upper())
    tenors_sig = tuple((ccy, tuple(sorted(tenors))) for ccy, tenors in sorted(needed_tenors.items()))
    names_sig = tuple((ccy, tuple(sorted(names))) for ccy, names in sorted(needed_forward_names.items()))
    return tenors_sig, names_sig


def _index_name_matches_quote_token(token: str, index_name: str, ccy: str) -> bool:
    quote_token = _normalize_curve_lookup_key(token)
    exact_name = _normalize_curve_lookup_key(index_name)
    if not quote_token or not exact_name:
        return False
    if quote_token == exact_name:
        return True
    ccy_prefix = str(ccy or "").strip().upper() + "-"
    if exact_name.startswith(ccy_prefix) and quote_token == exact_name[len(ccy_prefix):]:
        return True
    if quote_token.startswith(ccy_prefix) and exact_name == quote_token[len(ccy_prefix):]:
        return True
    return False


def _filter_leg_arrays_by_mask(leg: Dict[str, object], mask: np.ndarray) -> Dict[str, object]:
    out = dict(leg)
    n = int(mask.size)
    for key, value in leg.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n:
            out[key] = value[mask].copy()
    return out


def _rate_leg_currencies(legs_payload: Dict[str, object] | None, fallback_ccy: str = "") -> tuple[str, ...]:
    if not isinstance(legs_payload, dict):
        return (str(fallback_ccy).upper(),) if fallback_ccy else ()
    values: list[str] = []
    for leg in legs_payload.get("rate_legs", []):
        ccy = str(leg.get("ccy", "")).strip().upper()
        if ccy and ccy not in values:
            values.append(ccy)
    if not values and fallback_ccy:
        values.append(str(fallback_ccy).upper())
    return tuple(values)


def _build_bma_proxy_curve(
    base_curve: Callable[[float], float],
    ratio_times: np.ndarray,
    ratio_vals: np.ndarray,
) -> Callable[[float], float]:
    times = np.asarray(ratio_times, dtype=float)
    ratios = np.asarray(ratio_vals, dtype=float)
    scalar_cache: Dict[float, float] = {}

    def curve(t: float) -> float:
        tt = float(t)
        cached = scalar_cache.get(tt)
        if cached is not None:
            return cached
        if times.size == 0:
            rr = 1.0
        elif times.size == 1:
            rr = float(ratios[0])
        else:
            rr = float(np.interp(tt, times, ratios, left=float(ratios[0]), right=float(ratios[-1])))
        rr = float(np.clip(rr, 0.01, 10.0))
        out = float(max(float(base_curve(tt)) ** rr, 1.0e-12))
        scalar_cache[tt] = out
        return out

    return curve


def _resolve_fx_pair_name(ccy1: str, ccy2: str, available_pairs: Sequence[str]) -> str | None:
    left = str(ccy1).upper()
    right = str(ccy2).upper()
    if not left or not right or left == right:
        return None
    direct = f"{left}/{right}"
    inverse = f"{right}/{left}"
    available = {str(pair).upper().replace("-", "/") for pair in available_pairs}
    if direct in available:
        return direct
    if inverse in available:
        return inverse
    return direct


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
    torch_device: str | None
    trade_specs: Tuple[_TradeSpec, ...]
    unsupported: Tuple[Trade, ...]
    mpor: MporConfig
    input_provenance: Dict[str, str]  # Diagnostic dict recording which source was used for each input (model_params, market, grid, portfolio).
    input_fallbacks: Tuple[str, ...] = ()
    fx_forwards: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    fx_spots_today: Dict[str, float] = field(default_factory=dict)
    discount_curve_dates: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    observation_dates: Tuple[str, ...] = ()  # ORE exposure report dates aligned with observation_times.
    discount_curve_dfs: Dict[str, Tuple[float, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _SharedFxSimulation:
    hybrid: Any
    sim: Dict[str, Any]
    pair_keys: Tuple[str, ...]


@dataclass
class _PricingContext:
    snapshot: XVASnapshot
    inputs: _PythonLgmInputs
    model: Any
    x_paths: np.ndarray
    irs_backend: Any
    shared_fx_sim: Any
    n_times: int
    n_paths: int
    torch_curve_cache: Dict[tuple[str, str], object] = field(default_factory=dict)
    torch_rate_leg_value_cache: Dict[tuple[object, ...], np.ndarray] = field(default_factory=dict)
    capfloor_value_cache: Dict[tuple[object, ...], np.ndarray] = field(default_factory=dict)
    swaption_value_cache: Dict[tuple[object, ...], np.ndarray] = field(default_factory=dict)
    trade_value_cache: Dict[tuple[object, ...], tuple[np.ndarray | None, bool]] = field(default_factory=dict)
    irs_curve_cache: Dict[tuple[str, str], Dict[str, object]] = field(default_factory=dict)
    last_trade_backend: str = ""
    last_trade_backend_detail: str = ""


class PythonLgmAdapter:
    """Adapter that values supported trades using the Python LGM stack."""

    def __init__(self, fallback_to_swig: bool = False):
        self.fallback_to_swig = bool(fallback_to_swig)
        self._loaded = False
        self._lgm_mod = None
        self._irs_utils = None
        self._fx_utils = None
        self._inflation_mod = None
        self._ir_options_mod = None
        self._ore_snapshot_mod = None
        self._asof_cache: Dict[int, str] = {}
        self._fixings_cache: Dict[int, Dict[tuple[str, str], float]] = {}
        self._time_date_cache: Dict[tuple[int, float], str] = {}
        self._portfolio_root_cache: Dict[int, ET.Element] = {}
        self._ore_root_cache: Dict[int, ET.Element] = {}
        self._todaysmarket_root_cache: Dict[int, ET.Element] = {}
        self._simulation_root_cache: Dict[int, ET.Element] = {}
        self._simulation_tenor_cache: Dict[int, np.ndarray] = {}
        self._swap_index_forward_tenor_cache: Dict[int, Dict[str, str]] = {}
        self._fx_index_convention_cache: Dict[int, Dict[str, Dict[str, object]]] = {}
        self._generic_rate_swap_legs_cache: Dict[tuple[str, int], Optional[Dict[str, object]]] = {}
        self._generic_capfloor_state_cache: Dict[tuple[str, int], Optional[Dict[str, object]]] = {}
        self._generic_swaption_state_cache: Dict[tuple[str, int], Optional[Dict[str, object]]] = {}
        self._generic_cashflow_state_cache: Dict[tuple[str, int], Optional[Dict[str, object]]] = {}
        self._generic_fra_state_cache: Dict[tuple[str, int], Optional[Dict[str, object]]] = {}
        self._generic_rate_swap_legs_xml_cache: Dict[tuple[object, ...], Optional[Dict[str, object]]] = {}
        self._generic_capfloor_state_xml_cache: Dict[tuple[object, ...], Optional[Dict[str, object]]] = {}
        self._generic_swaption_state_xml_cache: Dict[tuple[object, ...], Optional[Dict[str, object]]] = {}
        self._generic_cashflow_state_xml_cache: Dict[tuple[object, ...], Optional[Dict[str, object]]] = {}
        self._generic_fra_state_xml_cache: Dict[tuple[object, ...], Optional[Dict[str, object]]] = {}
        self._irs_leg_cache: Dict[tuple[str, int], Dict[str, np.ndarray]] = {}
        self._market_overlay_cache: Dict[int, Dict[str, Any]] = {}
        self._market_overlay_scan_cache: Dict[tuple[object, ...], Dict[str, Any]] = {}
        self._capfloor_vol_cache: Dict[tuple[int, str], Dict[str, Any]] = {}
        self._quote_dict_cache: Dict[int, Tuple[Dict[str, object], ...]] = {}
        self._extract_inputs_cache: Dict[str, _PythonLgmInputs] = {}
        self._portfolio_classification_cache: Dict[
            tuple[object, ...],
            tuple[list["_TradeSpec"], list["Trade"], set[str]],
        ] = {}
        self._curve_fit_cache: Dict[tuple[object, ...], Optional[_CurveBundle]] = {}
        self._swap_pricing_backend_cache: Dict[tuple[str, bool], tuple[object, ...] | None] = {}
        self._par_swap_deterministic_cache: Dict[tuple[int, float, float, float, int], np.ndarray] = {}
        self._coupon_path_cache: Dict[tuple[object, ...], np.ndarray] = {}
        self._lgm_path_cache: Dict[tuple[object, ...], tuple[np.ndarray, Any | None]] = {}

    def _progress_enabled(self, snapshot: XVASnapshot) -> bool:
        raw = str(snapshot.config.params.get("python.progress", "Y")).strip().upper()
        return raw not in {"N", "NO", "FALSE", "0", "OFF"}

    def _build_reporter(self, snapshot: XVASnapshot) -> _ConsoleRunReporter:
        raw_bar = str(snapshot.config.params.get("python.progress_bar", "Y")).strip().upper()
        use_bar = raw_bar not in {"N", "NO", "FALSE", "0", "OFF"}
        interval_raw = snapshot.config.params.get("python.progress_log_interval", 25)
        try:
            checkpoint_interval = int(interval_raw)
        except Exception:
            checkpoint_interval = 25
        return _ConsoleRunReporter(
            enabled=self._progress_enabled(snapshot),
            use_bar=use_bar,
            checkpoint_interval=checkpoint_interval,
        )

    def _resolve_irs_pricing_backend(self, inputs: _PythonLgmInputs):
        irs_count = sum(1 for spec in inputs.trade_specs if spec.kind == "IRS")
        rate_swap_count = sum(1 for spec in inputs.trade_specs if spec.kind == "RateSwap")
        if irs_count == 0 and rate_swap_count == 0:
            return None
        requested_device = str(inputs.torch_device or "").strip().lower()
        if requested_device in {"numpy", "none", "off", "disabled", "disable"}:
            return None
        try:
            import torch
            from pythonore.compute.lgm_torch_xva import (
                TorchDiscountCurve,
                capfloor_npv_paths_torch,
                deflate_lgm_npv_paths_torch_batched,
                par_swap_rate_paths_torch,
                price_plain_rate_leg_paths_torch,
                swap_npv_paths_from_ore_legs_dual_curve_torch,
            )
        except Exception:
            return None
        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        device = requested_device
        if device in {"", "auto"}:
            device = "mps" if mps_available else "cpu"
        elif device not in {"cpu", "mps"}:
            device = "mps" if mps_available else "cpu"
        elif device == "mps" and not mps_available:
            device = "cpu"

        cache_key = (device, mps_available)
        cached = self._swap_pricing_backend_cache.get(cache_key)
        if cached is not None or cache_key in self._swap_pricing_backend_cache:
            return cached
        backend = (
            TorchDiscountCurve,
            swap_npv_paths_from_ore_legs_dual_curve_torch,
            deflate_lgm_npv_paths_torch_batched,
            device,
            price_plain_rate_leg_paths_torch,
            par_swap_rate_paths_torch,
            capfloor_npv_paths_torch,
        )
        self._swap_pricing_backend_cache[cache_key] = backend
        return backend

    def _supports_torch_rate_swap(self, spec: _TradeSpec) -> bool:
        if spec.kind != "RateSwap" or not isinstance(spec.legs, dict):
            return False
        rate_legs = list(spec.legs.get("rate_legs", []))
        if not rate_legs:
            return False
        if any(isinstance(leg.get("fx_reset"), dict) for leg in rate_legs):
            return False
        if len(_rate_leg_currencies(spec.legs, spec.ccy)) > 1:
            return False
        floating_count = sum(str(leg.get("kind", "")).upper() == "FLOATING" for leg in rate_legs)
        fixed_count = sum(str(leg.get("kind", "")).upper() == "FIXED" for leg in rate_legs)
        if floating_count > 1 and fixed_count == 0:
            return False
        # The torch path supports plain vanilla fixed/floating coupons. We still
        # keep the generic builder for pure floating basis swaps, overnight-indexed
        # coupons, cap/floor features, naked-option legs, and other conventions
        # that need ORE's stripping, lookback, or rate-cutoff handling.
        for leg in rate_legs:
            if str(leg.get("kind", "")).upper() != "FLOATING":
                continue
            schedule_rule = str(leg.get("schedule_rule", "FORWARD")).upper()
            index_name = str(leg.get("index_name", "")).upper()
            if bool(leg.get("overnight_indexed", False)):
                return False
            if any(
                leg.get(key) is not None if key in {"cap", "floor"} else bool(leg.get(key, False))
                for key in ("cap", "floor", "naked_option", "local_cap_floor")
            ):
                return False
            if int(leg.get("lookback_days", 0) or 0) != 0:
                return False
            if int(leg.get("rate_cutoff", 0) or 0) != 0:
                return False
            if any(tag in index_name for tag in ("BMA", "SIFMA", "BASIS")) and schedule_rule != "FORWARD":
                return False
        return all(
            str(leg.get("kind", "")).upper() in {"FIXED", "FLOATING"}
            for leg in rate_legs
        )

    def _torch_curve_from_handle(
        self,
        curve_cache: Dict[tuple[str, str], object],
        torch_curve_ctor: object,
        torch_device: str,
        key: tuple[str, str],
        curve: Callable[[float], float],
        sample_times: np.ndarray,
    ):
        cached = curve_cache.get(key)
        if cached is not None:
            return cached
        pts = np.unique(np.asarray(sample_times, dtype=float))
        pts = pts[np.isfinite(pts)]
        pts.sort()
        cached = torch_curve_ctor(
            times=pts,
            dfs=np.asarray([float(curve(float(t))) for t in pts], dtype=float),
            device=torch_device,
        )
        curve_cache[key] = cached
        return cached

    def _torch_capped_floored_rate(
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

    def _torch_digital_option_rate(
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
        return self._digital_option_rate(
            raw_rate,
            strike,
            payoff,
            is_call=is_call,
            long_short=long_short,
            fixed_mode=fixed_mode,
            atm_included=atm_included,
            capped_rate_fn=capped_rate_fn,
        )

    def _rate_leg_pricing_cache_key(self, ccy: str, leg: Dict[str, object]) -> tuple[object, ...]:
        kind = str(leg.get("kind", "")).upper()
        key: list[object] = [ccy.upper(), kind]
        scalar_fields = (
            "notional",
            "sign",
            "index_name",
            "index_name_1",
            "index_name_2",
            "fixing_days",
            "is_in_arrears",
            "is_averaged",
            "has_sub_periods",
            "day_counter",
            "call_strike",
            "call_payoff",
            "put_strike",
            "put_payoff",
            "call_position",
            "put_position",
            "is_call_atm_included",
            "is_put_atm_included",
            "naked_option",
            "cap",
            "floor",
        )
        for field in scalar_fields:
            value = leg.get(field)
            if isinstance(value, np.ndarray):
                continue
            key.append((field, value))
        array_fields = (
            "pay_time",
            "start_time",
            "end_time",
            "pay_date",
            "start_date",
            "end_date",
            "fixing_date",
            "fixing_time",
            "amount",
            "spread",
            "gearing",
            "accrual",
            "index_accrual",
            "quoted_coupon",
            "is_historically_fixed",
        )
        for field in array_fields:
            if field not in leg:
                continue
            arr = np.asarray(leg.get(field))
            key.append((field, str(arr.dtype), tuple(arr.tolist())))
        return tuple(key)

    def _rate_leg_coupon_paths_torch(
        self,
        model: Any,
        leg: Dict[str, object],
        ccy: str,
        inputs: _PythonLgmInputs,
        t: float,
        x_t: np.ndarray,
        *,
        torch_backend: tuple[object, ...],
        curve_cache: Dict[tuple[str, str], object],
    ) -> np.ndarray:
        kind = str(leg.get("kind", "")).upper()
        torch_curve_ctor, _, _, torch_device, _, torch_par_swap = torch_backend
        start = np.asarray(leg.get("start_time", []), dtype=float)
        end = np.asarray(leg.get("end_time", []), dtype=float)
        pay = np.asarray(leg.get("pay_time", end), dtype=float)
        fixing = np.asarray(leg.get("fixing_time", start), dtype=float)
        quoted = np.asarray(leg.get("quoted_coupon", np.zeros(start.shape)), dtype=float)
        fixed_mask = np.asarray(leg.get("is_historically_fixed", np.zeros(start.shape, dtype=bool)), dtype=bool)
        spread = np.asarray(leg.get("spread", np.zeros(start.shape)), dtype=float)
        gearing = np.asarray(leg.get("gearing", np.ones(start.shape)), dtype=float)
        coupons = np.zeros((start.size, x_t.size), dtype=float)
        sample_base = np.concatenate((np.asarray([t], dtype=float), start, end, pay))
        for i in range(start.size):
            fixed_now = bool(fixed_mask[i] or fixing[i] <= t + 1.0e-12)
            if fixed_now:
                base = np.full(x_t.shape, quoted[i], dtype=float)
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
                        pay_time=float(pay[i]),
                        fixing_days=int(leg.get("fixing_days", 2)),
                        day_counter_name=str(leg.get("day_counter", "A365")),
                        in_arrears=bool(leg.get("is_in_arrears", False)),
                    )
                if ql_rate is not None:
                    base = np.full(x_t.shape, ql_rate, dtype=float)
                else:
                    cms_horizon = max(float(np.max(sample_base)) if sample_base.size else 0.0, float(start[i]) + float(tenor_years))
                    cms_dense_grid = np.arange(0.0, cms_horizon + 0.2500001, 0.25, dtype=float)
                    cms_sample = np.unique(np.concatenate((sample_base, cms_dense_grid, np.asarray([cms_horizon], dtype=float))))
                    torch_curve = self._torch_curve_from_handle(
                        curve_cache,
                        torch_curve_ctor,
                        torch_device,
                        ("cms", index_name.upper()),
                        curve,
                        cms_sample,
                    )
                    base = torch_par_swap(
                        model,
                        torch_curve,
                        float(t),
                        x_t,
                        float(start[i]),
                        float(tenor_years),
                        return_numpy=True,
                    )
            elif kind in {"CMSSPREAD", "DIGITALCMSSPREAD"}:
                idx1 = str(leg.get("index_name_1", ""))
                idx2 = str(leg.get("index_name_2", ""))
                curve1 = self._resolve_index_curve(inputs, ccy, idx1)
                curve2 = self._resolve_index_curve(inputs, ccy, idx2)
                tenor1 = re.search(r"(\d+[YMWD])$", idx1.upper())
                tenor2 = re.search(r"(\d+[YMWD])$", idx2.upper())
                ql_rate1 = ql_rate2 = None
                if kind == "DIGITALCMSSPREAD" and t <= 1.0e-12:
                    ql_rate1 = self._static_ql_cms_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        index_name=idx1,
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=float(pay[i]),
                        fixing_days=int(leg.get("fixing_days", 2)),
                        day_counter_name=str(leg.get("day_counter", "A365")),
                        in_arrears=bool(leg.get("is_in_arrears", False)),
                    )
                    ql_rate2 = self._static_ql_cms_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        index_name=idx2,
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=float(pay[i]),
                        fixing_days=int(leg.get("fixing_days", 2)),
                        day_counter_name=str(leg.get("day_counter", "A365")),
                        in_arrears=bool(leg.get("is_in_arrears", False)),
                    )
                if ql_rate1 is None:
                    tenor_years_1 = float(_parse_ore_tenor_to_years(tenor1.group(1)) if tenor1 else 10.0)
                    cms_horizon_1 = max(float(np.max(sample_base)) if sample_base.size else 0.0, float(start[i]) + tenor_years_1)
                    cms_sample_1 = np.unique(
                        np.concatenate(
                            (
                                sample_base,
                                np.arange(0.0, cms_horizon_1 + 0.2500001, 0.25, dtype=float),
                                np.asarray([cms_horizon_1], dtype=float),
                            )
                        )
                    )
                    torch_curve1 = self._torch_curve_from_handle(
                        curve_cache,
                        torch_curve_ctor,
                        torch_device,
                        ("cms", idx1.upper()),
                        curve1,
                        cms_sample_1,
                    )
                    rate1 = torch_par_swap(
                        model,
                        torch_curve1,
                        float(t),
                        x_t,
                        float(start[i]),
                        tenor_years_1,
                        return_numpy=True,
                    )
                else:
                    rate1 = np.full(x_t.shape, ql_rate1, dtype=float)
                if ql_rate2 is None:
                    tenor_years_2 = float(_parse_ore_tenor_to_years(tenor2.group(1)) if tenor2 else 2.0)
                    cms_horizon_2 = max(float(np.max(sample_base)) if sample_base.size else 0.0, float(start[i]) + tenor_years_2)
                    cms_sample_2 = np.unique(
                        np.concatenate(
                            (
                                sample_base,
                                np.arange(0.0, cms_horizon_2 + 0.2500001, 0.25, dtype=float),
                                np.asarray([cms_horizon_2], dtype=float),
                            )
                        )
                    )
                    torch_curve2 = self._torch_curve_from_handle(
                        curve_cache,
                        torch_curve_ctor,
                        torch_device,
                        ("cms", idx2.upper()),
                        curve2,
                        cms_sample_2,
                    )
                    rate2 = torch_par_swap(
                        model,
                        torch_curve2,
                        float(t),
                        x_t,
                        float(start[i]),
                        tenor_years_2,
                        return_numpy=True,
                    )
                else:
                    rate2 = np.full(x_t.shape, ql_rate2, dtype=float)
                base = rate1 - rate2
            elif kind == "FLOATING":
                return self._rate_leg_coupon_paths(model, leg, ccy, inputs, t, x_t, snapshot=snapshot)
            else:
                base = np.zeros_like(x_t, dtype=float)

            raw_coupon = gearing[i] * base + spread[i]
            coupon = raw_coupon
            if kind == "FLOATING":
                cap = leg.get("cap")
                floor = leg.get("floor")
                if cap is not None or floor is not None:
                    coupon = self._capped_floored_rate(
                        raw_coupon,
                        cap=float(cap) if cap is not None else None,
                        floor=float(floor) if floor is not None else None,
                    )
            if kind == "CMSSPREAD":
                cap = leg.get("cap")
                floor = leg.get("floor")
                ql_rate = None
                if t <= 1.0e-12:
                    ql_rate = self._static_ql_cmsspread_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        idx1=str(leg.get("index_name_1", "")),
                        idx2=str(leg.get("index_name_2", "")),
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=float(pay[i]),
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
                elif fixed_now:
                    coupon = self._torch_capped_floored_rate(
                        raw_coupon,
                        cap=float(cap) if cap is not None else None,
                        floor=float(floor) if floor is not None else None,
                    )
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
                ql_raw = None
                ql_capped_rate_fn = None
                if not fixed_now and t <= 1.0e-12:
                    idx1 = str(leg.get("index_name_1", ""))
                    idx2 = str(leg.get("index_name_2", ""))
                    ql_raw = self._static_ql_cmsspread_rate(
                        inputs,
                        asof=inputs.asof,
                        ccy=ccy,
                        idx1=idx1,
                        idx2=idx2,
                        start_time=float(start[i]),
                        end_time=float(end[i]),
                        pay_time=float(pay[i]),
                        fixing_days=int(leg.get("fixing_days", 2)),
                        day_counter_name=str(leg.get("day_counter", "A365")),
                        in_arrears=bool(leg.get("is_in_arrears", False)),
                        gearing=float(gearing[i]),
                        spread=float(spread[i]),
                    )
                    ql_capped_rate_fn = lambda cap, floor, idx1_=idx1, idx2_=idx2: np.full_like(
                        raw_coupon,
                        self._static_ql_cmsspread_rate(
                            inputs,
                            asof=inputs.asof,
                            ccy=ccy,
                            idx1=idx1_,
                            idx2=idx2_,
                            start_time=float(start[i]),
                            end_time=float(end[i]),
                            pay_time=float(pay[i]),
                            fixing_days=int(leg.get("fixing_days", 2)),
                            day_counter_name=str(leg.get("day_counter", "A365")),
                            in_arrears=bool(leg.get("is_in_arrears", False)),
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
                capped_rate_fn = ql_capped_rate_fn
                if capped_rate_fn is None and not fixed_now:
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
                coupon = coupon + self._torch_digital_option_rate(
                    raw_coupon,
                    float(leg.get("call_strike", float("nan"))),
                    float(leg.get("call_payoff", float("nan"))),
                    is_call=True,
                    long_short=float(leg.get("call_position", 1.0)),
                    fixed_mode=fixed_now,
                    atm_included=bool(leg.get("is_call_atm_included", False)),
                    capped_rate_fn=capped_rate_fn,
                )
                coupon = coupon + self._torch_digital_option_rate(
                    raw_coupon,
                    float(leg.get("put_strike", float("nan"))),
                    float(leg.get("put_payoff", float("nan"))),
                    is_call=False,
                    long_short=float(leg.get("put_position", 1.0)),
                    fixed_mode=fixed_now,
                    atm_included=bool(leg.get("is_put_atm_included", False)),
                    capped_rate_fn=capped_rate_fn,
                )
            coupons[i, :] = coupon
        return coupons

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
        self._par_swap_deterministic_cache.clear()
        self._coupon_path_cache.clear()
        reporter = self._build_reporter(snapshot)
        reporter.log(
            f"run {run_id}: start mode={'hybrid' if self.fallback_to_swig else 'native_only'} "
            f"trades={len(snapshot.portfolio.trades)} paths={int(snapshot.config.num_paths)} "
            f"analytics={','.join(snapshot.config.analytics)}"
        )
        reporter.log("extracting runtime inputs")
        inputs = self._extract_inputs(snapshot, mapped)
        reporter.log(
            "inputs ready: "
            f"grid={inputs.times.size} valuation_grid={inputs.valuation_times.size} "
            f"obs={inputs.observation_times.size} native_specs={len(inputs.trade_specs)} unsupported={len(inputs.unsupported)}"
        )

        n_times = int(inputs.times.size)
        n_paths = int(snapshot.config.num_paths)
        npv_by_trade: Dict[str, np.ndarray] = {}
        fallback_trades: List[Trade] = []
        unsupported: List[Trade] = list(inputs.unsupported)
        reporter.log(
            "support classification: "
            f"native={len(inputs.trade_specs)} "
            f"requires_swig={len(inputs.unsupported)}"
        )
        irs_backend = self._resolve_irs_pricing_backend(inputs)
        if irs_backend is None:
            reporter.log("irs pricing backend: numpy")
        else:
            reporter.log(f"irs pricing backend: torch ({irs_backend[3]})")

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
        reporter.log("simulating LGM state paths")
        x_paths, shared_fx_sim = self._simulate_lgm_paths_cached(snapshot, inputs, model, n_paths, rng_mode)
        reporter.log(f"simulation complete: x_paths_shape={tuple(x_paths.shape)}")
        valuation_idx, obs_idx, obs_closeout_idx, obs_dates = self._validated_grid_indices(inputs)
        df_base = inputs.discount_curves[snapshot.config.base_currency.upper()]
        obs_df_vec = np.asarray([df_base(float(t)) for t in inputs.times], dtype=float)[obs_idx]
        pfe_quantile = _pfe_quantile(snapshot)
        use_flow_amounts_t0 = str(snapshot.config.params.get("python.use_ore_flow_amounts_t0", "N")).strip().upper() in {"Y", "YES", "TRUE", "1"}
        store_npv_cube_paths = str(snapshot.config.params.get("python.store_npv_cube_paths", "N")).strip().upper() in {"Y", "YES", "TRUE", "1"}
        pv_total_native = 0.0
        ns_valuation_paths: Dict[str, np.ndarray] = {}
        ns_closeout_paths: Dict[str, np.ndarray] = {}
        xva_deflated_by_ns: Dict[str, bool] = {}
        npv_cube_payload: Dict[str, Dict[str, object]] = {}
        exposure_profiles_by_trade: Dict[str, Dict[str, object]] = {}
        reporter.start("native pricing", len(inputs.trade_specs))
        irs_curve_cache: Dict[tuple[str, str], Dict[str, object]] = {}
        torch_curve_cache: Dict[tuple[str, str], object] = {}
        torch_rate_leg_value_cache: Dict[tuple[object, ...], np.ndarray] = {}
        capfloor_value_cache: Dict[tuple[object, ...], np.ndarray] = {}
        swaption_value_cache: Dict[tuple[object, ...], np.ndarray] = {}
        trade_value_cache: Dict[tuple[object, ...], tuple[np.ndarray | None, bool]] = {}
        pricing_ctx = _PricingContext(
            snapshot=snapshot,
            inputs=inputs,
            model=model,
            x_paths=x_paths,
            irs_backend=irs_backend,
            shared_fx_sim=shared_fx_sim,
            n_times=n_times,
            n_paths=n_paths,
            torch_curve_cache=torch_curve_cache,
            torch_rate_leg_value_cache=torch_rate_leg_value_cache,
            capfloor_value_cache=capfloor_value_cache,
            swaption_value_cache=swaption_value_cache,
            trade_value_cache=trade_value_cache,
            irs_curve_cache=irs_curve_cache,
        )

        for trade_idx, spec in enumerate(inputs.trade_specs, start=1):
            vals, xva_deflated = self._price_trade_paths(spec, pricing_ctx)
            if vals is None:
                unsupported.append(spec.trade)
                continue
            if use_flow_amounts_t0 and spec.kind == "RateSwap":
                anchored = _load_ore_t0_npv_for_reporting_ccy(
                    snapshot,
                    trade_id=spec.trade.trade_id,
                    report_ccy=inputs.model_ccy,
                )
                if anchored is not None:
                    vals[valuation_idx[0], :] = anchored

            self._fail_on_trade_nan(spec, vals)
            npv_by_trade[spec.trade.trade_id] = vals
            if spec.kind == "RateSwap":
                reporter.log(
                    f"rateSwap trade {spec.trade.trade_id}: backend={pricing_ctx.last_trade_backend}"
                    + (f" detail={pricing_ctx.last_trade_backend_detail}" if pricing_ctx.last_trade_backend_detail else "")
                )
            if spec.kind == "CapFloor" and spec.sticky_state is not None:
                reporter.log(
                    f"capfloor trade {spec.trade.trade_id}: backend={pricing_ctx.last_trade_backend}"
                    + (f" detail={pricing_ctx.last_trade_backend_detail}" if pricing_ctx.last_trade_backend_detail else "")
                )
                definition = spec.sticky_state.get("definition")
                index_name = str(spec.sticky_state.get("index_name", ""))
                if definition is not None:
                    p_disc = inputs.discount_curves[spec.ccy]
                    p_fwd = self._resolve_index_curve(inputs, spec.ccy, index_name)
                    capfloor_t0 = float(inputs.times[valuation_idx[0]])
                    realized_forward = None
                    fixings = self._fixings_lookup(snapshot)
                    if fixings and getattr(definition, "fixing_date", None) is not None:
                        pay = np.asarray(definition.pay_time, dtype=float)
                        live = pay >= capfloor_t0 - 1.0e-12
                        if np.any(live):
                            fix_dates = np.asarray(definition.fixing_date, dtype=object)[live]
                            realized_forward = np.zeros((int(np.sum(live)), x_paths.shape[1]), dtype=float)
                            for k_local, fix_date in enumerate(fix_dates):
                                fixing_key = (index_name.upper(), str(fix_date))
                                if fixing_key in fixings:
                                    realized_forward[k_local, :] = float(fixings[fixing_key])
                    capfloor_pv = self._ir_options_mod.capfloor_npv(
                        model,
                        p_disc,
                        p_fwd,
                        definition,
                        capfloor_t0,
                        x_paths[valuation_idx[0], :],
                        realized_forward=realized_forward,
                    )
                    pv_total_native += float(np.mean(capfloor_pv))
                    continue
            if use_flow_amounts_t0 and spec.kind == "IRS" and spec.legs is not None:
                p_disc = inputs.discount_curves[spec.ccy]
                anchored = _price_irs_t0_from_flow_amounts(spec.legs, p_disc)
                pv_total_native += anchored if anchored is not None else float(np.mean(vals[valuation_idx[0], :]))
            else:
                pv_total_native += float(np.mean(vals[valuation_idx[0], :]))

            if xva_deflated:
                if spec.kind == "IRS" and irs_backend is not None and not bool((spec.legs or {}).get("float_is_averaged", False)):
                    _, _, torch_deflator, _, _, _, _ = irs_backend
                    curve_key = (
                        spec.ccy,
                        str(
                            (spec.legs or {}).get(
                                "float_index",
                                spec.trade.additional_fields.get("index", _default_index_for_ccy(spec.ccy)),
                            )
                        ),
                    )
                    curve_state = irs_curve_cache.get(curve_key)
                    if curve_state is None:
                        p_disc = inputs.discount_curves[spec.ccy]
                        vals_xva = self._irs_utils.deflate_lgm_npv_paths(
                            model=model,
                            p0_disc=p_disc,
                            times=inputs.times,
                            x_paths=x_paths,
                            npv_paths=vals,
                        )
                    else:
                        vals_xva = torch_deflator(
                            model=model,
                            disc_curve=curve_state["disc_curve"],
                            times=inputs.times,
                            x_paths=x_paths,
                            npv_paths=vals,
                            return_numpy=True,
                        )
                else:
                    p_disc = inputs.discount_curves[spec.ccy]
                    vals_xva = self._irs_utils.deflate_lgm_npv_paths(
                        model=model,
                        p0_disc=p_disc,
                        times=inputs.times,
                        x_paths=x_paths,
                        npv_paths=vals,
                    )
            else:
                vals_xva = vals
            vals_xva_val = vals_xva[obs_idx, :]
            if spec.kind == "FXForward":
                vals_xva_obs = self._fx_forward_closeout_paths(
                    spec.trade,
                    vals_xva,
                    inputs.times,
                    inputs.observation_times,
                    inputs.observation_closeout_times,
                )
            else:
                vals_xva_obs = vals_xva[obs_closeout_idx, :]
            self._fail_on_trade_nan(spec, vals_xva_obs)
            profile = build_ore_exposure_profile_from_paths(
                spec.trade.trade_id,
                obs_dates,
                inputs.observation_times.tolist(),
                vals_xva_val,
                vals_xva_obs,
                discount_factors=obs_df_vec.tolist(),
                closeout_times=inputs.observation_closeout_times.tolist(),
                pfe_quantile=pfe_quantile,
                asof_date=inputs.asof,
            )
            exposure_profiles_by_trade[spec.trade.trade_id] = profile
            cube_payload = {
                "times": inputs.valuation_times.tolist(),
                "npv_mean": np.mean(vals[valuation_idx, :], axis=1).tolist(),
                "npv_xva_mean": np.mean(vals_xva[valuation_idx, :], axis=1).tolist(),
                "closeout_times": _build_sticky_closeout_times(inputs.valuation_times, inputs.mpor.mpor_years).tolist(),
                "closeout_npv_mean": np.mean(vals[np.searchsorted(inputs.times, _build_sticky_closeout_times(inputs.valuation_times, inputs.mpor.mpor_years)), :], axis=1).tolist(),
                "valuation_epe": np.mean(np.maximum(vals_xva_val, 0.0), axis=1).tolist(),
                "valuation_ene": np.mean(np.maximum(-vals_xva_val, 0.0), axis=1).tolist(),
                "closeout_epe": np.mean(np.maximum(vals_xva_obs, 0.0), axis=1).tolist(),
                "closeout_ene": np.mean(np.maximum(-vals_xva_obs, 0.0), axis=1).tolist(),
                "pfe": ore_pfe_quantile(vals_xva_obs, pfe_quantile).tolist(),
                "basel_ee": profile["basel_ee"],
                "basel_eee": profile["basel_eee"],
                "time_weighted_basel_epe": profile["time_weighted_basel_epe"],
                "time_weighted_basel_eepe": profile["time_weighted_basel_eepe"],
            }
            if store_npv_cube_paths:
                cube_payload["npv_paths"] = vals[valuation_idx, :].tolist()
                cube_payload["npv_xva_paths"] = vals_xva[valuation_idx, :].tolist()
                cube_payload["dates"] = obs_dates
            npv_cube_payload[spec.trade.trade_id] = cube_payload
            ns = spec.trade.netting_set
            ns_valuation_paths[ns] = ns_valuation_paths.get(ns, np.zeros_like(vals_xva_val)) + vals_xva_val
            ns_closeout_paths[ns] = ns_closeout_paths.get(ns, np.zeros_like(vals_xva_obs)) + vals_xva_obs
            xva_deflated_by_ns[ns] = xva_deflated_by_ns.get(ns, True) and xva_deflated
            reporter.update(trade_idx, f"{spec.trade.trade_id} [{spec.kind}]")

        reporter.finish("native trade pricing complete")

        fallback_result: XVAResult | None = None
        if unsupported:
            if not self.fallback_to_swig:
                bad = ", ".join(sorted({f"{t.trade_id}:{t.trade_type}" for t in unsupported}))
                preview = ", ".join(sorted({f"{t.trade_id}:{t.trade_type}" for t in unsupported})[:10])
                reporter.log(
                    "native-only run blocked by unsupported trades: "
                    + preview
                    + (" ..." if len(unsupported) > 10 else "")
                )
                raise EngineRunError(
                    "Unsupported by PythonLgmAdapter in native-only mode: "
                    f"{bad}. These trades are supported only through the ORE SWIG fallback."
                )
            try:
                reporter.log(f"entering SWIG fallback for {len(unsupported)} trades")
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
            reporter.log("SWIG fallback completed")

        reporter.log("assembling result payload")
        result = self._assemble_result(
            run_id=run_id,
            snapshot=snapshot,
            inputs=inputs,
            model=model,
            x_paths=x_paths,
            npv_by_trade=npv_by_trade,
            pv_total_precomputed=pv_total_native,
            npv_cube_payload_precomputed=npv_cube_payload,
            exposure_profiles_by_trade_precomputed=exposure_profiles_by_trade,
            ns_valuation_paths_precomputed=ns_valuation_paths,
            ns_closeout_paths_precomputed=ns_closeout_paths,
            xva_deflated_by_ns_precomputed=xva_deflated_by_ns,
            fallback=fallback_result,
            fallback_trades=fallback_trades,
            unsupported=unsupported if fallback_result is None else [],
        )
        reporter.log(
            "run complete: "
            f"pv_total={float(result.pv_total):.6f} "
            f"metrics={','.join(sorted(result.xva_by_metric)) if result.xva_by_metric else 'none'}"
        )
        result.metadata["python_lgm_rng_mode"] = rng_mode
        result.metadata["irs_pricing_backend"] = f"torch:{irs_backend[3]}" if irs_backend is not None else "numpy"
        if dim_mode in supported_python_dim_models:
            dim_result = calculate_python_dim(snapshot.config.params, dim_model=dim_mode)
            result.reports.update(dim_result.reports)
            result.cubes.update(dim_result.cubes)
            result.metadata["dim_mode"] = dim_mode
            result.metadata["dim_current"] = dict(dim_result.current_dim)
            result.metadata["dim_engine"] = dim_result.metadata.get("engine", "python-dim")
        result.metadata["support_classification"] = {
            "mode": "hybrid" if self.fallback_to_swig else "native_only",
            "native_only": not self.fallback_to_swig,
            "python_supported": len(unsupported) == 0 and len(inputs.unsupported) == 0,
            "native_trade_ids": [spec.trade.trade_id for spec in inputs.trade_specs],
            "native_trade_types": sorted({spec.trade.trade_type for spec in inputs.trade_specs}),
            "requires_swig_trade_ids": [trade.trade_id for trade in inputs.unsupported],
            "requires_swig_trade_types": sorted({trade.trade_type for trade in inputs.unsupported}),
            "native_trade_count": len(inputs.trade_specs),
            "requires_swig_trade_count": len(inputs.unsupported),
        }
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
            mapped = map_snapshot(snapshot)
            trade_specs, _, _ = self._classify_portfolio_trades(snapshot, mapped)
        except Exception:
            return {}
        out: Dict[str, List[float]] = {}
        for spec in trade_specs:
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
            from pythonore.compute import inflation, irs_xva_utils, lgm, lgm_fx_xva_utils, lgm_ir_options
            from pythonore.io import ore_snapshot

            self._inflation_mod = inflation
            self._lgm_mod = lgm
            self._irs_utils = irs_xva_utils
            self._fx_utils = lgm_fx_xva_utils
            self._ir_options_mod = lgm_ir_options
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
            import lgm_ir_options as lgm_ir_options_local
            import ore_snapshot as ore_snapshot_local

            self._inflation_mod = inflation_local
            self._lgm_mod = lgm_local
            self._irs_utils = irs_xva_utils_local
            self._fx_utils = lgm_fx_xva_utils_local
            self._ir_options_mod = lgm_ir_options_local
            self._ore_snapshot_mod = ore_snapshot_local
            self._loaded = True
        except Exception as exc:
            raise EngineRunError(f"Failed to import Python LGM toolchain: {exc}") from exc

    def _parse_model_params(
        self, xml: Dict[str, str], model_ccy: str, snapshot: "XVASnapshot"
    ) -> tuple[Dict[str, object], str]:
        """Return (lgm_params, param_source)."""
        param_source = "simulation"
        requested_source = str(
            snapshot.config.params.get(
                "python.lgm_param_source",
                snapshot.config.params.get("lgm_param_source", "auto"),
            )
            or "auto"
        ).strip().lower()
        use_ore_output_lgm = str(snapshot.config.params.get("python.use_ore_output_lgm_params", "N")).strip().upper() not in ("N", "FALSE", "0", "")
        if use_ore_output_lgm and "calibration.xml" in xml:
            lgm_params = lgm_market._parse_lgm_params_from_calibration_xml_text(xml["calibration.xml"], ccy_key=model_ccy)
            param_source = "calibration"
        elif requested_source in {"simulation", "simulation_xml"} and "simulation.xml" in xml:
            lgm_params = lgm_market._parse_lgm_params_from_simulation_xml_text(xml["simulation.xml"], ccy_key=model_ccy)
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
                market_data_path = self._ore_snapshot_mod._resolve_ore_path(
                    setup.get("marketDataFile", ""),
                    input_dir,
                )
                curve_config_path = self._ore_snapshot_mod._resolve_ore_path(
                    setup.get("curveConfigFile", ""),
                    input_dir,
                )
                conventions_path = self._ore_snapshot_mod._resolve_ore_path(
                    setup.get("conventionsFile", ""),
                    input_dir,
                )
                todaysmarket_xml_path = self._ore_snapshot_mod._resolve_ore_path(
                    setup.get("marketConfigFile", "todaysmarket.xml"),
                    input_dir,
                )
                output_path = self._ore_snapshot_mod._resolve_output_ore_path(
                    setup.get("outputPath", "Output"),
                    ore_path.parent.parent,
                )
                lgm_params = None
                if requested_source in {"auto", "python", "ore", "runtime_calibration"}:
                    lgm_params = self._ore_snapshot_mod.calibrate_lgm_params_in_python(
                        ore_xml_path=ore_path,
                        input_dir=input_dir,
                        output_path=output_path,
                        market_data_path=market_data_path,
                        curve_config_path=curve_config_path,
                        conventions_path=conventions_path,
                        todaysmarket_xml_path=todaysmarket_xml_path,
                        simulation_xml_path=simulation_xml_path,
                        ccy_key=model_ccy,
                    )
                    if lgm_params is None and requested_source in {"ore", "runtime_calibration"}:
                        lgm_params = self._ore_snapshot_mod.calibrate_lgm_params_via_ore(
                            ore_xml_path=ore_path,
                            input_dir=input_dir,
                            output_path=output_path,
                            simulation_xml_path=simulation_xml_path,
                            ccy_key=model_ccy,
                        )
                if lgm_params is not None:
                    param_source = "calibration"
                elif use_ore_output_lgm and "calibration.xml" in xml:
                    lgm_params = lgm_market._parse_lgm_params_from_calibration_xml_text(xml["calibration.xml"], ccy_key=model_ccy)
                    param_source = "calibration"
                elif "simulation.xml" in xml:
                    try:
                        lgm_params = lgm_market._parse_lgm_params_from_simulation_xml_text(xml["simulation.xml"], ccy_key=model_ccy)
                    except Exception:
                        lgm_params = {
                            "alpha_times": np.array([1.0], dtype=float),
                            "alpha_values": np.array([0.01, 0.01], dtype=float),
                            "kappa_times": np.array([1.0], dtype=float),
                            "kappa_values": np.array([0.03, 0.03], dtype=float),
                            "shift": 0.0,
                            "scaling": 1.0,
                        }
                        param_source = "synthetic"
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
                if use_ore_output_lgm and "calibration.xml" in xml:
                    lgm_params = lgm_market._parse_lgm_params_from_calibration_xml_text(xml["calibration.xml"], ccy_key=model_ccy)
                elif "simulation.xml" in xml:
                    try:
                        lgm_params = lgm_market._parse_lgm_params_from_simulation_xml_text(xml["simulation.xml"], ccy_key=model_ccy)
                    except Exception:
                        lgm_params = {
                            "alpha_times": np.array([1.0], dtype=float),
                            "alpha_values": np.array([0.01, 0.01], dtype=float),
                            "kappa_times": np.array([1.0], dtype=float),
                            "kappa_values": np.array([0.03, 0.03], dtype=float),
                            "shift": 0.0,
                            "scaling": 1.0,
                        }
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
            lgm_params = lgm_market._parse_lgm_params_from_simulation_xml_text(xml["simulation.xml"], ccy_key=model_ccy)
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
    ) -> tuple[np.ndarray, np.ndarray, Tuple[str, ...], str]:
        """Return (times, observation_times, observation_dates, grid_source)."""
        grid_source = "fallback"
        observation_dates: Tuple[str, ...] = ()
        if "simulation.xml" in xml:
            exact_grid = _parse_ore_exposure_date_grid_from_simulation_xml_text(
                xml["simulation.xml"],
                self._normalized_asof(snapshot),
                self._irs_utils,
            )
            if exact_grid is not None and exact_grid[0].size > 1:
                times, observation_dates = exact_grid
            else:
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
        if len(observation_dates) != observation_times.size:
            observation_dates = tuple(self._date_from_time_cached(snapshot, float(t)) for t in observation_times)
        return times, observation_times, observation_dates, grid_source

    def _classify_portfolio_trades(
        self, snapshot: "XVASnapshot", mapped: "MappedInputs"
    ) -> tuple[list["_TradeSpec"], list["Trade"], set[str]]:
        """Return (trade_specs, unsupported, ccy_set)."""
        cache_key = (
            self._normalized_asof(snapshot),
            id(snapshot.portfolio.trades),
            id(snapshot.fixings.points),
            id(mapped.xml_buffers.get("portfolio.xml", "")),
            id(mapped.xml_buffers.get("simulation.xml", "")),
            snapshot.config.base_currency.upper(),
        )
        cached = self._portfolio_classification_cache.get(cache_key)
        if cached is not None:
            return cached
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
            elif isinstance(t.product, BermudanSwaption):
                bermudan_state = self._build_bermudan_swaption_state(t, snapshot, mapped)
                if bermudan_state is not None:
                    trade_specs.append(
                        _TradeSpec(
                            trade=t,
                            kind="Swaption",
                            notional=float(bermudan_state["notional"]),
                            ccy=str(bermudan_state["ccy"]).upper(),
                            legs=bermudan_state.get("underlying_legs"),
                            sticky_state=bermudan_state,
                        )
                    )
                else:
                    unsupported.append(t)
            elif isinstance(t.product, GenericProduct):
                generic_capfloor = self._build_generic_capfloor_state(t, snapshot)
                if generic_capfloor is not None:
                    notionals = np.asarray(generic_capfloor["definition"].notional, dtype=float)
                    trade_specs.append(
                        _TradeSpec(
                            trade=t,
                            kind="CapFloor",
                            notional=float(np.max(np.abs(notionals))) if notionals.size else 0.0,
                            ccy=str(generic_capfloor["ccy"]).upper(),
                            sticky_state=generic_capfloor,
                        )
                    )
                else:
                    generic_swaption = self._build_generic_swaption_state(t, snapshot)
                    if generic_swaption is not None:
                        trade_specs.append(
                            _TradeSpec(
                                trade=t,
                                kind="Swaption",
                                notional=float(generic_swaption["notional"]),
                                ccy=str(generic_swaption["ccy"]).upper(),
                                legs=generic_swaption.get("underlying_legs"),
                                sticky_state=generic_swaption,
                            )
                        )
                    else:
                        generic_cashflow = self._build_generic_cashflow_state(t, snapshot)
                        if generic_cashflow is not None:
                            generic_ccys.add(str(generic_cashflow.get("ccy", snapshot.config.base_currency)).upper())
                            trade_specs.append(
                                _TradeSpec(
                                    trade=t,
                                    kind="Cashflow",
                                    notional=float(np.max(np.abs(np.asarray(generic_cashflow["amount"], dtype=float)))) if np.asarray(generic_cashflow["amount"], dtype=float).size else 0.0,
                                    ccy=str(generic_cashflow.get("ccy", snapshot.config.base_currency)).upper(),
                                    sticky_state=generic_cashflow,
                                )
                            )
                        else:
                            generic_fra = self._build_generic_fra_state(t, snapshot)
                            if generic_fra is not None:
                                generic_ccys.add(str(generic_fra.get("ccy", snapshot.config.base_currency)).upper())
                                trade_specs.append(
                                    _TradeSpec(
                                        trade=t,
                                        kind="FRA",
                                        notional=float(generic_fra.get("notional", 0.0)),
                                        ccy=str(generic_fra.get("ccy", snapshot.config.base_currency)).upper(),
                                        sticky_state=generic_fra,
                                    )
                                )
                            else:
                                generic_legs = self._build_generic_rate_swap_legs(t, snapshot)
                                if generic_legs is not None:
                                    notionals = [
                                        float(np.max(np.abs(np.asarray(leg.get("notional", np.asarray([0.0], dtype=float)), dtype=float))))
                                        for leg in generic_legs.get("rate_legs", [])
                                    ]
                                    generic_ccys.update(_rate_leg_currencies(generic_legs, str(generic_legs.get("ccy", snapshot.config.base_currency))))
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
            if isinstance(t.product, BermudanSwaption):
                ccy_set.add(t.product.ccy.upper())
        ccy_set.update(generic_ccys)
        result = (trade_specs, unsupported, ccy_set)
        self._portfolio_classification_cache[cache_key] = result
        return result

    def _normalized_asof(self, snapshot: XVASnapshot) -> str:
        key = str(snapshot.config.asof)
        cached = self._asof_cache.get(key)
        if cached is None:
            cached = _normalize_asof_date(snapshot.config.asof)
            self._asof_cache[key] = cached
        return cached

    def _fixings_lookup(self, snapshot: XVASnapshot) -> Dict[tuple[str, str], float]:
        key = (self._normalized_asof(snapshot), id(snapshot.fixings.points), len(snapshot.fixings.points))
        cached = self._fixings_cache.get(key)
        if cached is None:
            cached = {
                (str(p.index).upper(), _normalize_asof_date(p.date)): float(p.value)
                for p in snapshot.fixings.points
            }
            self._fixings_cache[key] = cached
        return cached

    def _date_from_time_cached(self, snapshot: XVASnapshot, t: float) -> str:
        cache_key = (self._normalized_asof(snapshot), float(t))
        cached = self._time_date_cache.get(cache_key)
        if cached is None:
            cached = _date_from_time(self._normalized_asof(snapshot), float(t))
            self._time_date_cache[cache_key] = cached
        return cached

    def _portfolio_cache_key(self, trade: Trade, snapshot: XVASnapshot, portfolio_xml: str = "") -> tuple[object, ...]:
        return (
            trade.trade_id,
            self._normalized_asof(snapshot),
            id(snapshot.portfolio.trades),
            id(snapshot.fixings.points),
            id(portfolio_xml) if portfolio_xml else 0,
        )

    def _clone_cached_trade_state(
        self,
        cached_state: Optional[Dict[str, object]],
        trade_id: str,
    ) -> Optional[Dict[str, object]]:
        if cached_state is None:
            return None
        definition = cached_state.get("definition")
        if definition is None or getattr(definition, "trade_id", trade_id) == trade_id:
            return cached_state
        cloned = dict(cached_state)
        cloned["definition"] = replace(definition, trade_id=trade_id)
        return cloned

    def _generic_xml_cache_key(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
    ) -> tuple[object, ...]:
        payload = getattr(trade.product, "payload", {})
        return (
            str(payload.get("trade_type", "")).strip(),
            str(payload.get("xml", "")).strip(),
            self._normalized_asof(snapshot),
            id(snapshot.fixings.points),
            snapshot.config.base_currency.upper(),
        )

    def _portfolio_root_from_xml(self, portfolio_xml: str) -> ET.Element:
        key = id(portfolio_xml)
        root = self._portfolio_root_cache.get(key)
        if root is None:
            root = ET.fromstring(portfolio_xml)
            self._portfolio_root_cache[key] = root
        return root

    def _ore_root_from_xml(self, ore_xml: str) -> Optional[ET.Element]:
        if not ore_xml.strip():
            return None
        key = id(ore_xml)
        root = self._ore_root_cache.get(key)
        if root is None:
            root = ET.fromstring(ore_xml)
            self._ore_root_cache[key] = root
        return root

    def _todaysmarket_root_from_xml(self, todaysmarket_xml: str) -> Optional[ET.Element]:
        if not todaysmarket_xml.strip():
            return None
        key = id(todaysmarket_xml)
        root = self._todaysmarket_root_cache.get(key)
        if root is None:
            root = ET.fromstring(todaysmarket_xml)
            self._todaysmarket_root_cache[key] = root
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
        key = id(conventions_xml)
        cached = self._swap_index_forward_tenor_cache.get(key)
        if cached is not None:
            return cached
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
        self._swap_index_forward_tenor_cache[key] = mapping
        return mapping

    def _parse_fx_index_conventions(self, conventions_xml: str) -> Dict[str, Dict[str, object]]:
        key = id(conventions_xml)
        cached = self._fx_index_convention_cache.get(key)
        if cached is not None:
            return cached
        mapping: Dict[str, Dict[str, object]] = {}
        if not conventions_xml.strip():
            return mapping
        try:
            root = ET.fromstring(conventions_xml)
        except Exception:
            return mapping
        for node in root.findall("./FX"):
            conv_id = (node.findtext("./Id") or "").strip().upper()
            source = (node.findtext("./SourceCurrency") or "").strip().upper()
            target = (node.findtext("./TargetCurrency") or "").strip().upper()
            if not source or not target:
                continue
            try:
                spot_days = int((node.findtext("./SpotDays") or "2").strip() or 2)
            except Exception:
                spot_days = 2
            advance_calendar = (node.findtext("./AdvanceCalendar") or f"{source},{target}").strip()
            convention = {
                "spot_days": spot_days,
                "advance_calendar": advance_calendar,
                "source_currency": source,
                "target_currency": target,
                "points_factor": float((node.findtext("./PointsFactor") or "10000").strip() or 10000.0),
                "spot_relative": (node.findtext("./SpotRelative") or "true").strip().lower() == "true",
            }
            keys = {
                conv_id,
                f"{source}{target}",
                f"{target}{source}",
                f"{source}/{target}",
                f"{target}/{source}",
                f"FX-{source}-{target}",
                f"FX-{target}-{source}",
            }
            for k in keys:
                if k:
                    mapping[k.upper()] = convention
        self._fx_index_convention_cache[key] = mapping
        return mapping

    def _resolve_fx_index_convention(
        self,
        conventions_xml: str,
        fx_index: str,
        foreign_ccy: str,
        domestic_ccy: str,
    ) -> Dict[str, object]:
        mapping = self._parse_fx_index_conventions(conventions_xml)
        fx_index_u = str(fx_index or "").strip().upper()
        foreign = str(foreign_ccy or "").strip().upper()
        domestic = str(domestic_ccy or "").strip().upper()
        candidates = [fx_index_u]
        parts = fx_index_u.split("-")
        if len(parts) >= 4 and parts[0] == "FX":
            source = parts[-2]
            target = parts[-1]
            family = "-".join(parts[1:-2])
            candidates.extend(
                [
                    f"FX-{family}-{source}-{target}",
                    f"FX-{family}-{target}-{source}",
                    f"{source}{target}",
                    f"{target}{source}",
                    f"{source}/{target}",
                    f"{target}/{source}",
                    f"FX-{source}-{target}",
                    f"FX-{target}-{source}",
                ]
            )
        if foreign and domestic:
            candidates.extend(
                [
                    f"{foreign}{domestic}",
                    f"{domestic}{foreign}",
                    f"{foreign}/{domestic}",
                    f"{domestic}/{foreign}",
                    f"FX-{foreign}-{domestic}",
                    f"FX-{domestic}-{foreign}",
                ]
            )
        for candidate in candidates:
            conv = mapping.get(str(candidate).upper())
            if conv is not None:
                return conv
        return {
            "spot_days": 2,
            "advance_calendar": f"{foreign},{domestic}" if foreign and domestic else "USD",
            "source_currency": foreign,
            "target_currency": domestic,
            "points_factor": 10000.0,
            "spot_relative": True,
        }

    def _fx_today_spots_from_settlement_quotes(
        self,
        fx_spots: Mapping[str, float],
        fx_forwards: Mapping[str, Sequence[Tuple[float, float]]],
        conventions_xml: str,
    ) -> Dict[str, float]:
        out = {str(k).upper(): float(v) for k, v in fx_spots.items()}
        for pair, raw_spot in list(out.items()):
            if len(pair) != 6:
                continue
            base = pair[:3]
            quote = pair[3:]
            conv = self._resolve_fx_index_convention(conventions_xml, f"FX-{base}-{quote}", base, quote)
            if not bool(conv.get("spot_relative", True)):
                continue
            spot_days = max(int(conv.get("spot_days", 0) or 0), 0)
            if spot_days <= 0:
                continue
            points_factor = max(float(conv.get("points_factor", 10000.0) or 10000.0), 1.0e-12)
            nodes = sorted(
                (float(t), float(points))
                for t, points in fx_forwards.get(pair, ())
                if np.isfinite(float(t)) and np.isfinite(float(points))
            )
            if not nodes:
                continue
            short_points: list[float] = []
            seen_times: set[float] = set()
            for t, points in nodes:
                # ON/TN/SN nodes are the first distinct maturities.  To turn an
                # ORE spot-date quote into today's FX quote, roll the spot quote
                # backwards over the settlement lag by subtracting those points.
                key = round(float(t), 12)
                if key in seen_times:
                    continue
                seen_times.add(key)
                short_points.append(float(points))
                if len(short_points) >= spot_days:
                    break
            if len(short_points) >= spot_days:
                out[pair] = float(raw_spot) - sum(short_points[:spot_days]) / points_factor
        return out

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
        snapshot_key = snapshot.stable_key()
        cached_inputs = self._extract_inputs_cache.get(snapshot_key)
        if cached_inputs is not None:
            return cached_inputs
        xml = dict(snapshot.config.xml_buffers)
        xml.update(mapped.xml_buffers)
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
        valuation_times, observation_times, observation_dates, grid_source = self._build_exposure_grid(snapshot, xml)
        mpor = _resolve_mpor_config(snapshot, xml)

        market_cache_key = id(snapshot.market.raw_quotes)
        overlay = self._market_overlay_cache.get(market_cache_key)
        if overlay is None:
            overlay = lgm_market._parse_market_overlay(snapshot.market.raw_quotes, snapshot.config.asof)
            self._market_overlay_cache[market_cache_key] = overlay
        fx_spots = overlay["fx"]
        fx_forwards = overlay.get("fx_forward", {})
        fx_spots_today = self._fx_today_spots_from_settlement_quotes(
            fx_spots,
            fx_forwards,
            xml.get("conventions.xml", ""),
        )
        fx_vols = overlay.get("fx_vol", {})
        swaption_normal_vols = overlay.get("swaption_normal_vols", {})
        cms_correlations = overlay.get("cms_correlations", {})
        swap_index_forward_tenors = self._parse_swap_index_forward_tenors(xml.get("conventions.xml", ""))
        zero_curves = overlay["zero"]
        named_zero_curves = overlay.get("named_zero", {})
        fwd_curves_raw = overlay.get("fwd", {})
        fwd_curves_by_index = overlay.get("fwd_by_index", {})
        bma_ratio_curves = overlay.get("bma_ratio", {})
        hazards = overlay["hazard"]
        recoveries = overlay["recovery"]

        trade_specs, unsupported, ccy_set = self._classify_portfolio_trades(snapshot, mapped)
        inflation_curves = self._load_inflation_curves(snapshot, trade_specs)
        stochastic_fx_pairs = _parse_stochastic_fx_pairs_from_simulation_xml_text(
            xml.get("simulation.xml", ""),
            model_ccy=model_ccy,
            trade_specs=trade_specs,
        )
        need_sig = _trade_curve_need_signature(trade_specs, swap_index_forward_tenors)
        needed_tenors_sig, needed_forward_names_sig = _derive_curve_needs_from_signature(*need_sig)
        needed_tenors = {ccy: set(tenors) for ccy, tenors in needed_tenors_sig}
        needed_forward_names = {ccy: set(names) for ccy, names in needed_forward_names_sig}
        market_cache_key = id(snapshot.market.raw_quotes)
        quote_dicts = self._quote_dict_cache.get(market_cache_key)
        if quote_dicts is None:
            quote_dicts = _scan_market_quotes(snapshot.market.raw_quotes)
            self._quote_dict_cache[market_cache_key] = quote_dicts
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

        curve_payload = self._load_ore_output_curves(snapshot, mapped, trade_specs, bma_ratio_curves=bma_ratio_curves)
        if curve_payload is not None:
            discount_curves = curve_payload.discount_curves
            discount_curve_dates = curve_payload.discount_curve_dates
            discount_curve_dfs = curve_payload.discount_curve_dfs
            forward_curves = curve_payload.forward_curves
            forward_curves_by_tenor = curve_payload.forward_curves_by_tenor
            forward_curves_by_name = curve_payload.forward_curves_by_name
            xva_discount_curve = curve_payload.xva_discount_curve
            funding_borrow_curve = curve_payload.funding_borrow_curve
            funding_lend_curve = curve_payload.funding_lend_curve
            curve_source = "ore_output_curves"
        else:
            discount_curve_dates = {}
            discount_curve_dfs = {}
            for c in list(ccy_set):
                if c not in zero_curves:
                    input_fallbacks.append(f"missing_zero_curve:{c}")
                    zero_curves.setdefault(c, [(0.0, 0.02), (max(float(snapshot.config.horizon_years), 1.0), 0.02)])
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
                for ccy in sorted(ccy_set):
                    if ccy not in discount_curves and ccy in zero_curves:
                        pts = sorted(zero_curves[ccy], key=lambda x: x[0])
                        if pts:
                            discount_curves[ccy] = self._irs_utils.build_discount_curve_from_zero_rate_pairs(pts)
                    if ccy in discount_curves:
                        forward_curves.setdefault(ccy, discount_curves[ccy])
                        forward_curves_by_tenor.setdefault(ccy, {})
                curve_source = "ore_quote_fit"
            else:
                ore_root = self._ore_root_from_xml(xml.get("ore.xml", ""))
                tm_root = self._todaysmarket_root_from_xml(xml.get("todaysmarket.xml", ""))
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
                    preferred_discount_family = lgm_market._curve_family_from_source_column(
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
                    index_buckets: Mapping[str, List[Tuple[float, float]]] = fwd_curves_by_index.get(ccy, {})
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
                    for index_name, bucket_pts in index_buckets.items():
                        by_time: Dict[float, List[float]] = {}
                        for t, r in bucket_pts:
                            by_time.setdefault(float(t), []).append(float(r))
                        if by_time:
                            pts_index = sorted(
                                ((t, min(vals, key=lambda x: abs(x))) for t, vals in by_time.items()),
                                key=lambda x: x[0],
                            )
                            if len(pts_index) == 1:
                                pts_index = [
                                    (0.0, pts_index[0][1]),
                                    (max(float(snapshot.config.horizon_years), 1.0), pts_index[0][1]),
                                ]
                            forward_curves_by_name[index_name.upper()] = self._irs_utils.build_discount_curve_from_zero_rate_pairs(
                                pts_index
                            )
                for index_name, tenor in swap_index_forward_tenors.items():
                    index_ccy = index_name.split("-", 1)[0].upper()
                    curve = forward_curves_by_tenor.get(index_ccy, {}).get(str(tenor).upper())
                    if curve is not None:
                        forward_curves_by_name.setdefault(index_name.upper(), curve)
                for ccy, names in needed_forward_names.items():
                    for index_name in sorted(names):
                        family = _forward_index_family(index_name, swap_index_forward_tenors)
                        if not family:
                            continue
                        index_quotes = [
                            q
                            for q in quote_dicts
                            if lgm_market._quote_matches_forward_curve(str(q["key"]), ccy, family, index_name=index_name)
                        ]
                        if not index_quotes:
                            continue
                        payload = self._ore_snapshot_mod.fit_discount_curves_from_programmatic_quotes(
                            snapshot.config.asof,
                            index_quotes,
                            fit_method="bootstrap_mm_irs_v1",
                        ).get(ccy)
                        if not payload:
                            continue
                        forward_curves_by_name[index_name.upper()] = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                            list(zip(payload["times"], payload["dfs"]))
                        )
                for ccy, names in needed_forward_names.items():
                    for index_name in sorted(names):
                        payload = named_zero_curves.get(index_name)
                        if not payload:
                            continue
                        forward_curves_by_name[index_name] = self._irs_utils.build_discount_curve_from_zero_rate_pairs(
                            sorted(payload, key=lambda x: x[0])
                        )
                for ccy, pts in bma_ratio_curves.items():
                    if not pts:
                        continue
                    by_time: Dict[float, List[float]] = {}
                    for t, r in pts:
                        by_time.setdefault(float(t), []).append(float(r))
                    ratio_pts = sorted(
                        ((t, min(vals, key=lambda x: abs(x))) for t, vals in by_time.items()),
                        key=lambda x: x[0],
                    )
                    if len(ratio_pts) == 1:
                        ratio_pts = [
                            (0.0, ratio_pts[0][1]),
                            (max(float(snapshot.config.horizon_years), 1.0), ratio_pts[0][1]),
                        ]
                    ratio_times = np.asarray([float(t) for t, _ in ratio_pts], dtype=float)
                    ratio_vals = np.asarray([float(v) for _, v in ratio_pts], dtype=float)
                    libor_curve = (
                        forward_curves_by_name.get(f"{ccy.upper()}-LIBOR-3M")
                        or forward_curves_by_tenor.get(ccy.upper(), {}).get("3M")
                        or forward_curves.get(ccy.upper(), discount_curves.get(ccy.upper()))
                    )
                    if libor_curve is None:
                        continue

                    bma_curve = _build_bma_proxy_curve(libor_curve, ratio_times, ratio_vals)
                    forward_curves_by_name[f"{ccy.upper()}-SIFMA"] = bma_curve
                    forward_curves_by_name[f"{ccy.upper()}-BMA"] = bma_curve
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
        ) = lgm_market._apply_curve_node_shocks(
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
                _normalize_forward_tenor_family(fwd_tenor),
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

        result = _PythonLgmInputs(
            asof=_normalize_asof_date(snapshot.config.asof),
            times=times,
            valuation_times=valuation_times,
            observation_times=observation_times,
            observation_dates=observation_dates,
            observation_closeout_times=observation_closeout_times,
            discount_curves=discount_curves,
            discount_curve_dates=discount_curve_dates,
            discount_curve_dfs=discount_curve_dfs,
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
            fx_spots_today=fx_spots_today,
            fx_forwards=fx_forwards,
            fx_vols=fx_vols,
            swaption_normal_vols=swaption_normal_vols,
            cms_correlations=cms_correlations,
            stochastic_fx_pairs=stochastic_fx_pairs,
            torch_device=(
                str(snapshot.config.params.get("python.torch_device", snapshot.config.params.get("python.device", ""))).strip().lower()
                or None
            ),
            trade_specs=tuple(trade_specs),
            unsupported=tuple(unsupported),
            mpor=mpor,
            input_provenance={
                "model_params": param_source,
                "market": curve_source,
                "grid": grid_source,
                "portfolio": "dataclass",
                "mpor": mpor.source,
            },
            input_fallbacks=tuple(sorted(set(input_fallbacks))),
        )
        self._extract_inputs_cache[snapshot_key] = result
        if len(self._extract_inputs_cache) > 16:
            self._extract_inputs_cache.pop(next(iter(self._extract_inputs_cache)))
        return result

    def _load_ore_output_curves(
        self,
        snapshot: XVASnapshot,
        mapped: MappedInputs,
        trade_specs: Sequence[_TradeSpec],
        *,
        bma_ratio_curves: Mapping[str, Sequence[Tuple[float, float]]] | None = None,
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
            tm_root = self._todaysmarket_root_from_xml(todaysmarket_xml)
            if tm_root is None:
                raise EngineRunError("todaysmarket.xml is empty; cannot resolve ORE output curves.")
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
                if spec.kind == "RateSwap" and isinstance(spec.legs, dict):
                    ccy_set.update(_rate_leg_currencies(spec.legs, spec.ccy))

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
            swap_index_forward_tenors = self._parse_swap_index_forward_tenors(mapped.xml_buffers.get("conventions.xml", ""))
            for spec in trade_specs:
                if spec.kind != "IRS" or spec.legs is None:
                    if spec.kind != "RateSwap" or spec.legs is None:
                        continue
                    for leg in spec.legs.get("rate_legs", []):
                        for field in ("index_name", "index_name_1", "index_name_2"):
                            index_name = str(leg.get(field, "")).strip()
                            if index_name:
                                mapped_tenor = swap_index_forward_tenors.get(index_name.upper(), "")
                                if mapped_tenor:
                                    underlying_index = _default_index_for_ccy_tenor(str(leg.get("ccy", spec.ccy)), mapped_tenor)
                                    requested_columns.add(underlying_index)
                                    forward_specs.setdefault(str(leg.get("ccy", spec.ccy)).upper(), {})[
                                        _normalize_forward_tenor_family(mapped_tenor)
                                    ] = underlying_index
                                elif not _is_bma_sifma_index(index_name):
                                    requested_columns.add(index_name)
                                forward_names.add(index_name.upper())
                    continue
                index_name = str(spec.legs.get("float_index", "")).strip()
                tenor_key = str(spec.legs.get("float_index_tenor", "")).upper()
                if index_name:
                    if not _is_bma_sifma_index(index_name):
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
            discount_curve_dates: Dict[str, tuple[str, ...]] = {}
            discount_curve_dfs: Dict[str, tuple[float, ...]] = {}
            for ccy, meta in discount_meta.items():
                discount_curves[ccy] = self._curve_from_column(curve_data, meta["source_column"])
                dates, _, dfs = curve_data[meta["source_column"]]
                discount_curve_dates[ccy] = tuple(str(d) for d in dates)
                discount_curve_dfs[ccy] = tuple(float(df) for df in dfs)

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
                elif index_name not in forward_curves_by_name:
                    mapped_tenor = _normalize_forward_tenor_family(swap_index_forward_tenors.get(index_name, ""))
                    index_ccy = index_name.split("-", 1)[0].upper()
                    curve = forward_curves_by_tenor.get(index_ccy, {}).get(mapped_tenor)
                    if curve is not None:
                        forward_curves_by_name[index_name] = curve
            for ccy, pts in (bma_ratio_curves or {}).items():
                if not pts:
                    continue
                by_time: Dict[float, List[float]] = {}
                for t, r in pts:
                    by_time.setdefault(float(t), []).append(float(r))
                ratio_pts = sorted(
                    ((t, min(vals, key=lambda x: abs(x))) for t, vals in by_time.items()),
                    key=lambda x: x[0],
                )
                if len(ratio_pts) == 1:
                    ratio_pts = [
                        (0.0, ratio_pts[0][1]),
                        (max(float(snapshot.config.horizon_years), 1.0), ratio_pts[0][1]),
                    ]
                ratio_times = np.asarray([float(t) for t, _ in ratio_pts], dtype=float)
                ratio_vals = np.asarray([float(v) for _, v in ratio_pts], dtype=float)
                ccy_key = ccy.upper()
                libor_curve = (
                    forward_curves_by_name.get(f"{ccy_key}-LIBOR-3M")
                    or forward_curves_by_tenor.get(ccy_key, {}).get("3M")
                    or forward_curves.get(ccy_key, discount_curves.get(ccy_key))
                )
                if libor_curve is None:
                    continue
                bma_curve = _build_bma_proxy_curve(libor_curve, ratio_times, ratio_vals)
                forward_curves_by_name[f"{ccy_key}-SIFMA"] = bma_curve
                forward_curves_by_name[f"{ccy_key}-BMA"] = bma_curve

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
                discount_curve_dates=discount_curve_dates,
                discount_curve_dfs=discount_curve_dfs,
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
            ccy_set = {snapshot.config.base_currency.upper()}
            for spec in trade_specs:
                ccy_set.add(spec.ccy.upper())
                if spec.kind == "RateSwap" and isinstance(spec.legs, dict):
                    ccy_set.update(_rate_leg_currencies(spec.legs, spec.ccy))
            ore_xml_text = snapshot.config.xml_buffers.get("ore.xml", "")
            ore_root = self._ore_root_from_xml(ore_xml_text)
            tm_root = self._todaysmarket_root_from_xml(snapshot.config.xml_buffers.get("todaysmarket.xml", ""))
            pricing_config_id = (
                (ore_root.findtext("./Markets/Parameter[@name='pricing']") if ore_root is not None else None)
                or "default"
            ).strip() or "default"
            sim_config_id = (
                (ore_root.findtext("./Markets/Parameter[@name='simulation']") if ore_root is not None else None)
                or pricing_config_id
            ).strip() or pricing_config_id
            market_cache_key = id(snapshot.market.raw_quotes)
            quote_dicts = self._quote_dict_cache.get(market_cache_key)
            if quote_dicts is None:
                quote_dicts = _scan_market_quotes(snapshot.market.raw_quotes)
                self._quote_dict_cache[market_cache_key] = quote_dicts
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
            need_sig = _trade_curve_need_signature(trade_specs, swap_index_forward_tenors)
            needed_tenors_sig, needed_forward_names_sig = _derive_curve_needs_from_signature(*need_sig)
            needed_tenors = {ccy: set(tenors) for ccy, tenors in needed_tenors_sig}
            needed_forward_names = {ccy: set(names) for ccy, names in needed_forward_names_sig}
            cache_key = (
                market_cache_key,
                snapshot.config.asof,
                pricing_config_id,
                sim_config_id,
                tuple(sorted(ccy_set)),
                tuple((ccy, tuple(sorted(tenors))) for ccy, tenors in sorted(needed_tenors.items())),
                tuple((ccy, tuple(sorted(names))) for ccy, names in sorted(needed_forward_names.items())),
            )
            cached_bundle = self._curve_fit_cache.get(cache_key)
            if cached_bundle is not None:
                return cached_bundle
            discount_family_fallbacks = {
                ccy: (sorted(tenors)[0] if len(tenors) == 1 else "")
                for ccy, tenors in needed_tenors.items()
            }

            discount_curves: Dict[str, Callable[[float], float]] = {}
            discount_curve_dates: Dict[str, tuple[str, ...]] = {}
            discount_curve_dfs: Dict[str, tuple[float, ...]] = {}
            for ccy in sorted(ccy_set):
                source_column = str(
                    discount_meta.get(ccy, {}).get("source_column")
                    or xva_discount_meta.get(ccy, {}).get("source_column")
                    or ""
                ).strip()
                discount_quotes = [
                    q
                    for q in quote_dicts
                    if lgm_market._quote_matches_discount_curve(
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
                discount_curve_dates[ccy] = tuple(str(d) for d in payload["dates"])
                discount_curve_dfs[ccy] = tuple(float(df) for df in payload["dfs"])
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
                    normalized_tenor = _normalize_forward_tenor_family(tenor)
                    tenor_quotes = [
                    q for q in quote_dicts if lgm_market._quote_matches_forward_curve(str(q["key"]), ccy, tenor)
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
                    tenor_curves[normalized_tenor] = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
                        list(zip(payload["times"], payload["dfs"]))
                    )
                    for spec in trade_specs:
                        if (
                            spec.kind == "IRS"
                            and spec.ccy == ccy
                            and spec.legs is not None
                            and _normalize_forward_tenor_family(str(spec.legs.get("float_index_tenor", "")).upper()) == normalized_tenor
                        ):
                            index_name = str(spec.legs.get("float_index", "")).upper()
                            if index_name:
                                forward_curves_by_name[index_name] = tenor_curves[normalized_tenor]
                    for index_name, mapped_tenor in swap_index_forward_tenors.items():
                        if index_name.startswith(ccy + "-") and _normalize_forward_tenor_family(mapped_tenor) == normalized_tenor:
                            forward_curves_by_name.setdefault(index_name.upper(), tenor_curves[normalized_tenor])
                for index_name in sorted(needed_forward_names.get(ccy, ())):
                    family = _forward_index_family(index_name, swap_index_forward_tenors)
                    if not family:
                        continue
                    index_quotes = [
                        q
                        for q in quote_dicts
                        if lgm_market._quote_matches_forward_curve(str(q["key"]), ccy, family, index_name=index_name)
                    ]
                    if not index_quotes:
                        continue
                    payload = self._ore_snapshot_mod.fit_discount_curves_from_programmatic_quotes(
                        snapshot.config.asof,
                        index_quotes,
                        fit_method="bootstrap_mm_irs_v1",
                    ).get(ccy)
                    if not payload:
                        continue
                    forward_curves_by_name[index_name] = self._ore_snapshot_mod.build_discount_curve_from_discount_pairs(
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
            result = _CurveBundle(
                discount_curves=discount_curves,
                discount_curve_dates=discount_curve_dates,
                discount_curve_dfs=discount_curve_dfs,
                forward_curves=forward_curves,
                forward_curves_by_tenor=forward_curves_by_tenor,
                forward_curves_by_name=forward_curves_by_name,
                xva_discount_curve=base_curve,
                funding_borrow_curve=None,
                funding_lend_curve=None,
            )
            self._curve_fit_cache[cache_key] = result
            if len(self._curve_fit_cache) > 16:
                self._curve_fit_cache.pop(next(iter(self._curve_fit_cache)))
            return result
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
            elif spec.kind == "FRA" and isinstance(spec.sticky_state, dict):
                extras.append(
                    np.asarray(
                        [
                            max(float(spec.sticky_state.get("start_time", 0.0)), 0.0),
                            max(float(spec.sticky_state.get("end_time", 0.0)), 0.0),
                        ],
                        dtype=float,
                    )
                )
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

    def _simulate_lgm_paths_cached(
        self,
        snapshot: XVASnapshot,
        inputs: _PythonLgmInputs,
        model: Any,
        n_paths: int,
        rng_mode: str,
    ) -> tuple[np.ndarray, Any | None]:
        cache_key = (
            snapshot.stable_key(),
            int(n_paths),
            str(rng_mode),
        )
        cached = self._lgm_path_cache.get(cache_key)
        if cached is not None:
            return cached
        rng, draw_order = self._build_lgm_rng(inputs.seed, rng_mode)
        x_paths = self._lgm_mod.simulate_lgm_measure(
            model,
            inputs.times,
            n_paths=n_paths,
            rng=rng,
            x0=0.0,
            draw_order=draw_order,
        )
        shared_fx_sim = self._build_shared_fx_simulation(snapshot, inputs, n_paths)
        if len(self._lgm_path_cache) >= 8:
            self._lgm_path_cache.pop(next(iter(self._lgm_path_cache)))
        self._lgm_path_cache[cache_key] = (x_paths, shared_fx_sim)
        return x_paths, shared_fx_sim

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
        available_pairs = tuple(str(pair).upper().replace("-", "/") for pair in inputs.stochastic_fx_pairs)
        pair_maturities: Dict[str, float] = {}
        for spec in inputs.trade_specs:
            if spec.kind == "FXForward" and isinstance(spec.trade.product, FXForward):
                pair = f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}"
                pair_maturities[pair] = max(pair_maturities.get(pair, 0.0), float(spec.trade.product.maturity_years))
                continue
            if spec.kind == "Cashflow":
                local_ccy = str(spec.ccy).strip().upper()
                report_ccy = inputs.model_ccy.upper()
                pair = _resolve_fx_pair_name(local_ccy, report_ccy, available_pairs)
                if pair is None:
                    continue
                pay = np.asarray((spec.sticky_state or {}).get("pay_time", []), dtype=float)
                max_pay = float(np.max(pay)) if pay.size else 0.0
                pair_maturities[pair] = max(pair_maturities.get(pair, 0.0), max_pay)
                continue
            if spec.kind != "RateSwap":
                continue
            ccys = _rate_leg_currencies(spec.legs, spec.ccy)
            if len(ccys) != 2:
                continue
            pair = _resolve_fx_pair_name(ccys[0], ccys[1], available_pairs)
            if pair is None:
                continue
            max_pay = 0.0
            for leg in (spec.legs or {}).get("rate_legs", []):
                pay = np.asarray(leg.get("pay_time", []), dtype=float)
                if pay.size:
                    max_pay = max(max_pay, float(np.max(pay)))
            pair_maturities[pair] = max(pair_maturities.get(pair, 0.0), max_pay)
        all_fx_pairs = sorted(pair_maturities)
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
            max_maturity = max(float(pair_maturities.get(pair, 0.0)), 1.0 / 365.25)
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
        portfolio_xml = mapped.xml_buffers.get("portfolio.xml", "")
        cache_key = self._portfolio_cache_key(trade, snapshot, portfolio_xml)
        cached_legs = self._irs_leg_cache.get(cache_key)
        if cached_legs is not None:
            return cached_legs
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
                self._augment_irs_overnight_metadata_from_portfolio_xml(legs, portfolio_xml, trade.trade_id)
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
            lgm_inputs._build_irs_legs_from_trade(trade, snapshot.config.asof),
        )
        self._irs_leg_cache[cache_key] = result
        return result

    def _augment_irs_overnight_metadata_from_portfolio_xml(
        self,
        legs: Dict[str, np.ndarray],
        portfolio_xml: str,
        trade_id: str,
    ) -> None:
        if not portfolio_xml or "float_overnight_indexed" in legs:
            return
        try:
            root = self._portfolio_root_from_xml(portfolio_xml)
        except Exception:
            return
        trade_node = root.find(f"./Trade[@id='{trade_id}']")
        if trade_node is None:
            return
        float_leg = None
        for leg in trade_node.findall(".//SwapData/LegData"):
            if (leg.findtext("./LegType") or "").strip().upper() == "FLOATING":
                float_leg = leg
                break
        if float_leg is None:
            return
        fld = float_leg.find("./FloatingLegData")
        if fld is None:
            return
        index_name = (fld.findtext("./Index") or "").strip().upper()
        overnight_indexed = _forward_index_family(index_name) == "1D"
        legs["float_overnight_indexed"] = overnight_indexed
        legs["float_index"] = index_name
        legs["float_cap"] = float((fld.findtext("./Caps/Cap") or "").strip() or "nan") if (fld.findtext("./Caps/Cap") or "").strip() else None
        legs["float_floor"] = float((fld.findtext("./Floors/Floor") or "").strip() or "nan") if (fld.findtext("./Floors/Floor") or "").strip() else None
        lookback_txt = (fld.findtext("./Lookback") or "").strip()
        rate_cutoff_txt = (fld.findtext("./RateCutoff") or "").strip()
        legs["float_lookback_days"] = int(round(_parse_ore_tenor_to_years(lookback_txt) * 365.0)) if lookback_txt not in {"", "0D", "0d"} else 0
        legs["float_rate_cutoff"] = int(rate_cutoff_txt or "0")
        legs["float_naked_option"] = (fld.findtext("./NakedOption") or "false").strip().lower() == "true"
        legs["float_local_cap_floor"] = (fld.findtext("./LocalCapFloor") or "false").strip().lower() == "true"
        legs["float_apply_observation_shift"] = (fld.findtext("./ApplyObservationShift") or "false").strip().lower() == "true"
        legs["float_is_in_arrears"] = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
        legs["float_is_averaged"] = (fld.findtext("./IsAveraged") or "false").strip().lower() == "true"
        legs["float_has_sub_periods"] = (fld.findtext("./HasSubPeriods") or "false").strip().lower() == "true"
        legs["float_fixing_days"] = int((fld.findtext("./FixingDays") or "2").strip() or 2)
        spread = float((fld.findtext("./Spreads/Spread") or "0").strip() or 0.0)
        gearing = float((fld.findtext("./Gearings/Gearing") or "1").strip() or 1.0)
        if "float_spread" not in legs:
            legs["float_spread"] = np.full_like(np.asarray(legs.get("float_accrual", []), dtype=float), spread)
        if "float_gearing" not in legs:
            legs["float_gearing"] = np.full_like(np.asarray(legs.get("float_accrual", []), dtype=float), gearing)

    def _parse_generic_cashflow_schedule(
        self,
        cashflow_node: ET.Element,
        *,
        asof: Any,
        sign: float = 1.0,
    ) -> Optional[Dict[str, np.ndarray]]:
        parse_date = self._irs_utils._parse_yyyymmdd
        time_from_dates = self._irs_utils._time_from_dates
        entries: list[tuple[Any, float]] = []
        for flow_node in cashflow_node.findall("./Cashflow"):
            for amount_node in flow_node.findall("./Amount"):
                pay_date_txt = (amount_node.get("date") or amount_node.get("Date") or "").strip()
                amount_txt = (amount_node.text or "").strip()
                if not pay_date_txt or not amount_txt:
                    continue
                try:
                    entries.append((parse_date(pay_date_txt), float(amount_txt)))
                except Exception:
                    continue
        if not entries:
            payment_date = (
                (cashflow_node.findtext("./PaymentDate") or "").strip()
                or (cashflow_node.findtext("./Date") or "").strip()
            )
            amount_txt = (
                (cashflow_node.findtext("./Amount") or "").strip()
                or (cashflow_node.findtext("./CashflowAmount") or "").strip()
            )
            if payment_date and amount_txt:
                try:
                    entries.append((parse_date(payment_date), float(amount_txt)))
                except Exception:
                    return None
        if not entries:
            return None
        entries.sort(key=lambda item: item[0])
        pay_time = np.asarray([time_from_dates(asof, pay_date, "A365F") for pay_date, _ in entries], dtype=float)
        amount = float(sign) * np.asarray([cash_amount for _, cash_amount in entries], dtype=float)
        live_mask = pay_time >= -1.0e-12
        return {
            "pay_time": pay_time[live_mask].copy(),
            "amount": amount[live_mask].copy(),
        }

    def _build_generic_rate_swap_legs(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
    ) -> Optional[Dict[str, object]]:
        cache_key = self._portfolio_cache_key(trade, snapshot)
        if cache_key in self._generic_rate_swap_legs_cache:
            return self._generic_rate_swap_legs_cache[cache_key]
        xml_cache_key = self._generic_xml_cache_key(trade, snapshot)
        cached_xml_result = self._generic_rate_swap_legs_xml_cache.get(xml_cache_key)
        if cached_xml_result is not None or xml_cache_key in self._generic_rate_swap_legs_xml_cache:
            self._generic_rate_swap_legs_cache[cache_key] = cached_xml_result
            return cached_xml_result
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

        asof = self._irs_utils._parse_yyyymmdd(self._normalized_asof(snapshot))
        parse_date = self._irs_utils._parse_yyyymmdd
        build_schedule = self._irs_utils._build_schedule
        schedule_from_leg = getattr(self._irs_utils, "_schedule_from_leg", None)
        time_from_dates = self._irs_utils._time_from_dates
        year_fraction = self._irs_utils._year_fraction
        advance_business_days = self._irs_utils._advance_business_days
        infer_index_day_counter = self._irs_utils._infer_index_day_counter
        conventions_xml = snapshot.config.xml_buffers.get("conventions.xml", "")

        rate_legs: List[Dict[str, object]] = []
        ccy = None
        for leg in leg_nodes:
            leg_type = (leg.findtext("./LegType") or "").strip()
            leg_type_upper = leg_type.upper()
            if leg_type_upper not in {"FIXED", "FLOATING", "CMS", "CMSSPREAD", "DIGITALCMSSPREAD", "CASHFLOW"}:
                return None
            leg_ccy = (leg.findtext("./Currency") or "").strip().upper() or snapshot.config.base_currency.upper()
            ccy = ccy or leg_ccy
            payer = (leg.findtext("./Payer") or "").strip().lower() == "true"
            sign = -1.0 if payer else 1.0
            if leg_type_upper == "CASHFLOW":
                cashflow = leg.find("./CashflowData")
                if cashflow is None:
                    return None
                parsed_cashflows = self._parse_generic_cashflow_schedule(cashflow, asof=asof, sign=sign)
                if parsed_cashflows is None:
                    return None
                if np.asarray(parsed_cashflows.get("pay_time", []), dtype=float).size == 0:
                    continue
                rate_legs.append(
                    {
                        "kind": "CASHFLOW",
                        "ccy": leg_ccy,
                        "sign": sign,
                        "pay_time": np.asarray(parsed_cashflows["pay_time"], dtype=float),
                        "amount": np.asarray(parsed_cashflows["amount"], dtype=float),
                    }
                )
                continue
            dc = (leg.findtext("./DayCounter") or "A365").strip()
            pay_conv = (leg.findtext("./PaymentConvention") or "F").strip()
            notional_payment_lag = int((leg.findtext("./NotionalPaymentLag") or leg.findtext("./PaymentLag") or "0").strip() or 0)
            notional_initial_exchange = (leg.findtext("./Notionals/Exchanges/NotionalInitialExchange") or "false").strip().lower() == "true"
            notional_final_exchange = (leg.findtext("./Notionals/Exchanges/NotionalFinalExchange") or "false").strip().lower() == "true"
            notional_amort_exchange = (leg.findtext("./Notionals/Exchanges/NotionalAmortizingExchange") or "false").strip().lower() == "true"
            fx_reset_node = leg.find("./Notionals/FXReset")
            fx_reset: Dict[str, object] | None = None
            if fx_reset_node is not None:
                fx_index = (fx_reset_node.findtext("./FXIndex") or "").strip().upper()
                foreign_currency = (fx_reset_node.findtext("./ForeignCurrency") or "").strip().upper()
                fx_convention = self._resolve_fx_index_convention(
                    conventions_xml,
                    fx_index,
                    foreign_currency,
                    leg_ccy,
                )
                fx_reset = {
                    "foreign_currency": foreign_currency,
                    "foreign_amount": float((fx_reset_node.findtext("./ForeignAmount") or "0").strip() or 0.0),
                    "fx_index": fx_index,
                    "fixing_days": int(fx_convention.get("spot_days", 2) or 2),
                    "fixing_calendar": str(fx_convention.get("advance_calendar", "") or f"{foreign_currency},{leg_ccy}"),
                    "points_factor": float(fx_convention.get("points_factor", 10000.0) or 10000.0),
                    "spot_relative": bool(fx_convention.get("spot_relative", True)),
                    "reset_start_date": (fx_reset_node.findtext("./StartDate") or "").strip(),
                }
            rules = leg.find("./ScheduleData/Rules")
            tenor = (rules.findtext("./Tenor") if rules is not None else "") or ""
            cal = (
                (rules.findtext("./Calendar") if rules is not None else None)
                or leg.findtext("./Currency")
                or snapshot.config.base_currency
                or "TARGET"
            ).strip()
            if schedule_from_leg is not None:
                try:
                    s_dates, e_dates, p_dates = schedule_from_leg(leg, pay_convention=pay_conv)
                except Exception:
                    return None
                if notional_payment_lag:
                    adjust_date = getattr(self._irs_utils, "_adjust_date", None)
                    p_dates = np.asarray(
                        [
                            (adjust_date(advance_business_days(d, notional_payment_lag, leg_ccy), pay_conv, leg_ccy) if adjust_date is not None else advance_business_days(d, notional_payment_lag, leg_ccy))
                            for d in e_dates
                        ],
                        dtype=object,
                    )
            else:
                if rules is None:
                    return None
                start = parse_date((rules.findtext("./StartDate") or "").strip())
                end = parse_date((rules.findtext("./EndDate") or "").strip())
                tenor = (rules.findtext("./Tenor") or "").strip()
                conv = (rules.findtext("./Convention") or pay_conv).strip()
                s_dates, e_dates, p_dates = build_schedule(start, end, tenor, cal, conv, pay_convention=pay_conv)
            s_t = np.asarray([time_from_dates(asof, d, "A365F") for d in s_dates], dtype=float)
            e_t = np.asarray([time_from_dates(asof, d, "A365F") for d in e_dates], dtype=float)
            p_t = np.asarray([time_from_dates(asof, d, "A365F") for d in p_dates], dtype=float)
            accr = np.asarray([year_fraction(sd, ed, dc) for sd, ed in zip(s_dates, e_dates)], dtype=float)
            notionals = np.asarray(self._irs_utils.expand_leg_notionals(leg, s_dates, e_dates), dtype=float)
            leg_info: Dict[str, object] = {
                "kind": leg_type_upper,
                "ccy": leg_ccy,
                "notional": notionals,
                "sign": sign,
                "start_time": s_t,
                "end_time": e_t,
                "pay_time": p_t,
                "accrual": accr,
                "schedule_tenor": tenor,
                "calendar": cal,
                "day_counter": dc,
                "notional_payment_lag": notional_payment_lag,
                "notional_initial_exchange": notional_initial_exchange,
                "notional_final_exchange": notional_final_exchange,
                "notional_amortizing_exchange": notional_amort_exchange,
                "fx_reset": fx_reset,
            }
            if leg_type_upper == "FIXED":
                rate = float((leg.findtext("./FixedLegData/Rates/Rate") or "0").strip())
                amount = sign * notionals * rate * accr
                leg_info["fixed_rate"] = np.full_like(accr, rate)
                leg_info["amount"] = amount
            elif leg_type_upper == "FLOATING":
                fld = leg.find("./FloatingLegData")
                if fld is None:
                    return None
                index_name = (fld.findtext("./Index") or "").strip().upper()
                schedule_rule = (rules.findtext("./Rule") or "Forward").strip().upper() if rules is not None else "FORWARD"
                spread_nodes = [node for node in fld.findall("./Spreads/Spread") if (node.text or "").strip()]
                gearing_nodes = [node for node in fld.findall("./Gearings/Gearing") if (node.text or "").strip()]
                spread, _ = self._irs_utils._expand_notional_nodes(spread_nodes, s_dates, default_value=0.0)
                gearing, _ = self._irs_utils._expand_notional_nodes(gearing_nodes, s_dates, default_value=1.0)
                cap_txt = (fld.findtext("./Caps/Cap") or "").strip()
                floor_txt = (fld.findtext("./Floors/Floor") or "").strip()
                fixing_days = int((fld.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
                lookback_txt = (fld.findtext("./Lookback") or "").strip()
                rate_cutoff_txt = (fld.findtext("./RateCutoff") or "").strip()
                naked_option = (fld.findtext("./NakedOption") or "false").strip().lower() == "true"
                local_cap_floor = (fld.findtext("./LocalCapFloor") or "false").strip().lower() == "true"
                is_averaged = (fld.findtext("./IsAveraged") or "false").strip().lower() == "true"
                has_sub_periods = (fld.findtext("./HasSubPeriods") or "false").strip().lower() == "true"
                index_dc = infer_index_day_counter(index_name, fallback=dc)
                overnight_indexed = _forward_index_family(index_name) == "1D"
                lookback_days = 0
                if lookback_txt not in {"", "0D", "0d"}:
                    try:
                        lookback_days = int(round(_parse_ore_tenor_to_years(lookback_txt) * 365.0))
                    except Exception:
                        return None
                rate_cutoff = int(rate_cutoff_txt or "0")
                if (lookback_days or rate_cutoff or naked_option or local_cap_floor) and not overnight_indexed:
                    # ORE's overnight indexed cap/floor machinery supports these
                    # fields for overnight indices, but the generic floating-leg
                    # path does not implement them for ibor-style coupons.
                    return None
                leg_info["index_name"] = index_name
                leg_info["schedule_rule"] = schedule_rule
                leg_info["spread"] = np.asarray(spread, dtype=float)
                leg_info["gearing"] = np.asarray(gearing, dtype=float)
                leg_info["cap"] = float(cap_txt) if cap_txt else None
                leg_info["floor"] = float(floor_txt) if floor_txt else None
                leg_info["overnight_indexed"] = overnight_indexed
                leg_info["is_averaged"] = is_averaged
                leg_info["has_sub_periods"] = has_sub_periods
                leg_info["lookback_days"] = lookback_days
                leg_info["rate_cutoff"] = rate_cutoff
                leg_info["apply_observation_shift"] = (fld.findtext("./ApplyObservationShift") or "false").strip().lower() == "true"
                leg_info["local_cap_floor"] = local_cap_floor
                leg_info["naked_option"] = naked_option
                leg_info["is_in_arrears"] = in_arrears
                leg_info["fixing_days"] = fixing_days
                if is_averaged:
                    fix_dates = [
                        self._irs_utils._average_overnight_coupon_fixing_date(
                            sd,
                            ed,
                            calendar=cal,
                            fixing_days=fixing_days,
                            rate_cutoff=rate_cutoff,
                        )
                        for sd, ed in zip(s_dates, e_dates)
                    ]
                else:
                    fix_base_dates = e_dates if in_arrears else s_dates
                    fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["fixing_time"] = np.asarray([time_from_dates(asof, fd, "A365F") for fd in fix_dates], dtype=float)
                leg_info["start_date"] = np.asarray([d.isoformat() for d in s_dates], dtype=object)
                leg_info["end_date"] = np.asarray([d.isoformat() for d in e_dates], dtype=object)
                leg_info["pay_date"] = np.asarray([d.isoformat() for d in p_dates], dtype=object)
                leg_info["fixing_date"] = np.asarray([d.isoformat() for d in fix_dates], dtype=object)
                leg_info["index_accrual"] = np.asarray([year_fraction(sd, ed, index_dc) for sd, ed in zip(s_dates, e_dates)], dtype=float)
                leg_info["quoted_coupon"] = np.zeros_like(accr)
            elif leg_type_upper == "CMS":
                fld = leg.find("./CMSLegData")
                if fld is None:
                    return None
                index_name = (fld.findtext("./Index") or "").strip().upper()
                spread_nodes = [node for node in fld.findall("./Spreads/Spread") if (node.text or "").strip()]
                gearing_nodes = [node for node in fld.findall("./Gearings/Gearing") if (node.text or "").strip()]
                spread, _ = self._irs_utils._expand_notional_nodes(spread_nodes, s_dates, default_value=0.0)
                gearing, _ = self._irs_utils._expand_notional_nodes(gearing_nodes, s_dates, default_value=1.0)
                fixing_days = int((fld.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
                fix_base_dates = e_dates if in_arrears else s_dates
                fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["index_name"] = index_name
                leg_info["spread"] = np.asarray(spread, dtype=float)
                leg_info["gearing"] = np.asarray(gearing, dtype=float)
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
                spread_nodes = [node for node in fld.findall("./Spreads/Spread") if (node.text or "").strip()]
                gearing_nodes = [node for node in fld.findall("./Gearings/Gearing") if (node.text or "").strip()]
                spread, _ = self._irs_utils._expand_notional_nodes(spread_nodes, s_dates, default_value=0.0)
                gearing, _ = self._irs_utils._expand_notional_nodes(gearing_nodes, s_dates, default_value=1.0)
                cap_txt = (fld.findtext("./Caps/Cap") or "").strip()
                floor_txt = (fld.findtext("./Floors/Floor") or "").strip()
                fixing_days = int((fld.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
                fix_base_dates = e_dates if in_arrears else s_dates
                fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["index_name_1"] = index1
                leg_info["index_name_2"] = index2
                leg_info["spread"] = np.asarray(spread, dtype=float)
                leg_info["gearing"] = np.asarray(gearing, dtype=float)
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
                spread_nodes = [node for node in cms.findall("./Spreads/Spread") if (node.text or "").strip()]
                gearing_nodes = [node for node in cms.findall("./Gearings/Gearing") if (node.text or "").strip()]
                spread, _ = self._irs_utils._expand_notional_nodes(spread_nodes, s_dates, default_value=0.0)
                gearing, _ = self._irs_utils._expand_notional_nodes(gearing_nodes, s_dates, default_value=1.0)
                fixing_days = int((cms.findtext("./FixingDays") or "2").strip() or 2)
                in_arrears = (cms.findtext("./IsInArrears") or "false").strip().lower() == "true"
                fix_base_dates = e_dates if in_arrears else s_dates
                fix_dates = [advance_business_days(d, -fixing_days, cal) for d in fix_base_dates]
                leg_info["index_name_1"] = index1
                leg_info["index_name_2"] = index2
                leg_info["spread"] = np.asarray(spread, dtype=float)
                leg_info["gearing"] = np.asarray(gearing, dtype=float)
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
            live_mask = p_t >= -1.0e-12
            if not np.any(live_mask):
                continue
            leg_info = _filter_leg_arrays_by_mask(leg_info, live_mask)
            rate_legs.append(leg_info)

        result = self._apply_historical_fixings_to_generic_rate_legs(
            snapshot,
            {"ccy": ccy or snapshot.config.base_currency.upper(), "rate_legs": rate_legs},
        )
        self._generic_rate_swap_legs_cache[cache_key] = result
        self._generic_rate_swap_legs_xml_cache[xml_cache_key] = result
        return result

    def _build_generic_cashflow_state(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
    ) -> Optional[Dict[str, object]]:
        cache_key = self._portfolio_cache_key(trade, snapshot)
        if cache_key in self._generic_cashflow_state_cache:
            return self._generic_cashflow_state_cache[cache_key]
        xml_cache_key = self._generic_xml_cache_key(trade, snapshot)
        cached_xml_result = self._generic_cashflow_state_xml_cache.get(xml_cache_key)
        if cached_xml_result is not None or xml_cache_key in self._generic_cashflow_state_xml_cache:
            self._generic_cashflow_state_cache[cache_key] = cached_xml_result
            return cached_xml_result
        product = trade.product
        if not isinstance(product, GenericProduct):
            return None
        if str(product.payload.get("trade_type", "")).strip() != "Cashflow":
            return None
        xml = str(product.payload.get("xml", "")).strip()
        if "<CashflowData" not in xml:
            return None
        try:
            trade_root = ET.fromstring(f"<Trade>{xml}</Trade>")
        except Exception:
            return None
        cashflow = trade_root.find("./CashflowData")
        if cashflow is None:
            return None
        ccy = (cashflow.findtext("./Currency") or snapshot.config.base_currency).strip().upper()
        asof = self._irs_utils._parse_yyyymmdd(self._normalized_asof(snapshot))
        cashflow_state = self._parse_generic_cashflow_schedule(cashflow, asof=asof, sign=1.0)
        if cashflow_state is None:
            return None
        result = {
            "ccy": ccy,
            "pay_time": np.asarray(cashflow_state["pay_time"], dtype=float),
            "amount": np.asarray(cashflow_state["amount"], dtype=float),
        }
        self._generic_cashflow_state_cache[cache_key] = result
        self._generic_cashflow_state_xml_cache[xml_cache_key] = result
        return result

    def _build_generic_fra_state(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
    ) -> Optional[Dict[str, object]]:
        cache_key = self._portfolio_cache_key(trade, snapshot)
        if cache_key in self._generic_fra_state_cache:
            return self._generic_fra_state_cache[cache_key]
        xml_cache_key = self._generic_xml_cache_key(trade, snapshot)
        cached_xml_result = self._generic_fra_state_xml_cache.get(xml_cache_key)
        if cached_xml_result is not None or xml_cache_key in self._generic_fra_state_xml_cache:
            self._generic_fra_state_cache[cache_key] = cached_xml_result
            return cached_xml_result
        product = trade.product
        if not isinstance(product, GenericProduct):
            return None
        if str(product.payload.get("trade_type", "")).strip() != "ForwardRateAgreement":
            return None
        xml = str(product.payload.get("xml", "")).strip()
        if "<ForwardRateAgreementData" not in xml:
            return None
        try:
            trade_root = ET.fromstring(f"<Trade>{xml}</Trade>")
        except Exception:
            return None
        data = trade_root.find("./ForwardRateAgreementData")
        if data is None:
            return None
        try:
            asof = self._irs_utils._parse_yyyymmdd(self._normalized_asof(snapshot))
            start_date = self._irs_utils._parse_yyyymmdd((data.findtext("./StartDate") or "").strip())
            end_date = self._irs_utils._parse_yyyymmdd((data.findtext("./EndDate") or "").strip())
            index_name = (data.findtext("./Index") or "").strip().upper()
            day_counter = self._irs_utils._infer_index_day_counter(index_name, fallback="A360")
            result = {
                "ccy": (data.findtext("./Currency") or snapshot.config.base_currency).strip().upper(),
                "index_name": index_name,
                "notional": float((data.findtext("./Notional") or "0").strip() or 0.0),
                "strike": float((data.findtext("./Strike") or "0").strip() or 0.0),
                "position": -1.0 if (data.findtext("./LongShort") or "Long").strip().lower() == "short" else 1.0,
                "start_time": float(self._irs_utils._time_from_dates(asof, start_date, "A365F")),
                "end_time": float(self._irs_utils._time_from_dates(asof, end_date, "A365F")),
                "accrual": float(self._irs_utils._year_fraction(start_date, end_date, day_counter)),
                "day_counter": day_counter,
            }
        except Exception:
            result = None
        self._generic_fra_state_cache[cache_key] = result
        self._generic_fra_state_xml_cache[xml_cache_key] = result
        return result

    def _build_generic_capfloor_state(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
    ) -> Optional[Dict[str, object]]:
        cache_key = self._portfolio_cache_key(trade, snapshot)
        if cache_key in self._generic_capfloor_state_cache:
            return self._generic_capfloor_state_cache[cache_key]
        xml_cache_key = self._generic_xml_cache_key(trade, snapshot)
        cached_xml_result = self._generic_capfloor_state_xml_cache.get(xml_cache_key)
        if cached_xml_result is not None or xml_cache_key in self._generic_capfloor_state_xml_cache:
            result = self._clone_cached_trade_state(cached_xml_result, trade.trade_id)
            self._generic_capfloor_state_cache[cache_key] = result
            return result
        product = trade.product
        if not isinstance(product, GenericProduct):
            return None
        if str(product.payload.get("trade_type", "")).strip() != "CapFloor":
            return None
        xml = str(product.payload.get("xml", "")).strip()
        if "<CapFloorData" not in xml:
            return None
        try:
            trade_root = ET.fromstring(f"<Trade>{xml}</Trade>")
        except Exception:
            return None
        data = trade_root.find("./CapFloorData")
        leg = data.find("./LegData") if data is not None else None
        if data is None or leg is None:
            return None
        if (leg.findtext("./LegType") or "").strip().upper() != "FLOATING":
            return None
        asof = self._irs_utils._parse_yyyymmdd(self._normalized_asof(snapshot))
        parse_date = self._irs_utils._parse_yyyymmdd
        build_schedule = self._irs_utils._build_schedule
        schedule_from_leg = getattr(self._irs_utils, "_schedule_from_leg", None)
        time_from_dates = self._irs_utils._time_from_dates
        year_fraction = self._irs_utils._year_fraction
        advance_business_days = self._irs_utils._advance_business_days
        rules = leg.find("./ScheduleData/Rules")
        fld = leg.find("./FloatingLegData")
        if fld is None:
            return None
        pay_conv = (leg.findtext("./PaymentConvention") or "F").strip()
        tenor = (rules.findtext("./Tenor") if rules is not None else "") or ""
        cal = (
            (rules.findtext("./Calendar") if rules is not None else None)
            or leg.findtext("./Currency")
            or snapshot.config.base_currency
            or "TARGET"
        ).strip()
        if schedule_from_leg is not None:
            try:
                s_dates, e_dates, p_dates = schedule_from_leg(leg, pay_convention=pay_conv)
            except Exception:
                return None
        else:
            if rules is None:
                return None
            start = parse_date((rules.findtext("./StartDate") or "").strip())
            end = parse_date((rules.findtext("./EndDate") or "").strip())
            tenor = (rules.findtext("./Tenor") or "").strip()
            conv = (rules.findtext("./Convention") or pay_conv).strip()
            s_dates, e_dates, p_dates = build_schedule(start, end, tenor, cal, conv, pay_convention=pay_conv)
        dc = (leg.findtext("./DayCounter") or "A365").strip()
        start_t = np.asarray([time_from_dates(asof, d, "A365F") for d in s_dates], dtype=float)
        end_t = np.asarray([time_from_dates(asof, d, "A365F") for d in e_dates], dtype=float)
        pay_t = np.asarray([time_from_dates(asof, d, "A365F") for d in p_dates], dtype=float)
        accr = np.asarray([year_fraction(sd, ed, dc) for sd, ed in zip(s_dates, e_dates)], dtype=float)
        fixing_days = int((fld.findtext("./FixingDays") or "2").strip() or 2)
        in_arrears = (fld.findtext("./IsInArrears") or "false").strip().lower() == "true"
        fixing_base = e_dates if in_arrears else s_dates
        fixing_dates = [advance_business_days(d, -fixing_days, cal) for d in fixing_base]
        fixing_t = np.asarray([time_from_dates(asof, d, "A365F") for d in fixing_dates], dtype=float)
        live = (pay_t >= -1.0e-12) & (end_t >= start_t - 1.0e-12) & (pay_t >= end_t - 1.0e-12)
        if not np.any(live):
            self._generic_capfloor_state_cache[cache_key] = None
            self._generic_capfloor_state_xml_cache[xml_cache_key] = None
            return None
        start_t = start_t[live]
        end_t = end_t[live]
        pay_t = pay_t[live]
        accr = accr[live]
        fixing_t = fixing_t[live]
        notionals = np.asarray(self._irs_utils.expand_leg_notionals(leg, s_dates, e_dates), dtype=float)
        if notionals.size == live.size:
            notionals = notionals[live]
        elif notionals.size != start_t.size:
            if notionals.size >= live.size:
                notionals = notionals[live][: start_t.size]
            else:
                notionals = np.resize(notionals, start_t.size)
        spread_nodes = [node for node in fld.findall("./Spreads/Spread") if (node.text or "").strip()]
        gearing_nodes = [node for node in fld.findall("./Gearings/Gearing") if (node.text or "").strip()]
        spread, _ = self._irs_utils._expand_notional_nodes(spread_nodes, s_dates, default_value=0.0)
        gearing, _ = self._irs_utils._expand_notional_nodes(gearing_nodes, s_dates, default_value=1.0)
        spread = np.asarray(spread, dtype=float)
        gearing = np.asarray(gearing, dtype=float)
        if spread.size == live.size:
            spread = spread[live]
        if gearing.size == live.size:
            gearing = gearing[live]
        cap_text = (data.findtext("./Caps/Cap") or "").strip()
        floor_text = (data.findtext("./Floors/Floor") or "").strip()
        if cap_text:
            option_type = "cap"
            strike = float(cap_text)
        elif floor_text:
            option_type = "floor"
            strike = float(floor_text)
        else:
            return None
        long_short = (data.findtext("./LongShort") or "Long").strip().lower()
        position = -1.0 if long_short == "short" else 1.0
        ccy = (leg.findtext("./Currency") or snapshot.config.base_currency).strip().upper()
        index_name = (fld.findtext("./Index") or "").strip().upper()
        definition = self._ir_options_mod.CapFloorDef(
            trade_id=trade.trade_id,
            ccy=ccy,
            option_type=option_type,
            start_time=start_t,
            end_time=end_t,
            pay_time=pay_t,
            accrual=accr,
            notional=notionals,
            strike=np.full_like(accr, strike),
            gearing=np.asarray(gearing, dtype=float),
            spread=np.asarray(spread, dtype=float),
            fixing_time=fixing_t,
            fixing_date=np.asarray([d.isoformat() for d in fixing_dates], dtype=object)[live],
            position=position,
        )
        result = {"definition": definition, "ccy": ccy, "index_name": index_name}
        self._generic_capfloor_state_cache[cache_key] = result
        self._generic_capfloor_state_xml_cache[xml_cache_key] = result
        return result

    def _build_generic_swaption_state(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
    ) -> Optional[Dict[str, object]]:
        cache_key = self._portfolio_cache_key(trade, snapshot)
        if cache_key in self._generic_swaption_state_cache:
            return self._generic_swaption_state_cache[cache_key]
        xml_cache_key = self._generic_xml_cache_key(trade, snapshot)
        cached_xml_result = self._generic_swaption_state_xml_cache.get(xml_cache_key)
        if cached_xml_result is not None or xml_cache_key in self._generic_swaption_state_xml_cache:
            result = self._clone_cached_trade_state(cached_xml_result, trade.trade_id)
            self._generic_swaption_state_cache[cache_key] = result
            return result
        product = trade.product
        if not isinstance(product, GenericProduct):
            return None
        if str(product.payload.get("trade_type", "")).strip() != "Swaption":
            return None
        xml = str(product.payload.get("xml", "")).strip()
        if "<SwaptionData" not in xml:
            return None
        try:
            trade_root = ET.fromstring(f"<Trade id=\"{trade.trade_id}\"><TradeType>Swaption</TradeType>{xml}</Trade>")
        except Exception:
            return None
        style = (trade_root.findtext("./SwaptionData/OptionData/Style") or "").strip().lower()
        if style not in {"european", "bermudan"}:
            return None
        premium_records = lgm_inputs._parse_swaption_premium_records(trade_root)
        asof = self._normalized_asof(snapshot)
        fake_root = ET.fromstring(f"<Portfolio>{ET.tostring(trade_root, encoding='unicode')}</Portfolio>")
        try:
            underlying_legs = self._irs_utils.load_swap_legs_from_portfolio_root(fake_root, trade.trade_id, asof)
        except Exception:
            return None
        exercise_nodes = trade_root.findall("./SwaptionData/OptionData/ExerciseDates/ExerciseDate")
        exercise_dates = [
            self._irs_utils._parse_yyyymmdd((node.text or "").strip())
            for node in exercise_nodes
            if (node.text or "").strip()
        ]
        if style == "european" and len(exercise_dates) != 1:
            return None
        if len(exercise_dates) == 0:
            return None
        asof_date = self._irs_utils._parse_yyyymmdd(asof)
        live_exercise_dates = [dt for dt in exercise_dates if dt >= asof_date]
        if style == "european" and len(live_exercise_dates) != 1:
            return None
        if len(live_exercise_dates) == 0:
            return None
        exercise_times = np.asarray(
            [float(self._irs_utils._time_from_dates(asof_date, dt, "A365F")) for dt in live_exercise_dates],
            dtype=float,
        )
        fixed_leg = next((leg for leg in trade_root.findall("./SwaptionData/LegData") if (leg.findtext("./LegType") or "").strip().lower() == "fixed"), None)
        float_index = str(underlying_legs.get("float_index", "")).upper()
        ccy = (
            (fixed_leg.findtext("./Currency") if fixed_leg is not None else None)
            or trade_root.findtext("./SwaptionData/LegData/Currency")
            or snapshot.config.base_currency
        ).strip().upper()
        long_short = (trade_root.findtext("./SwaptionData/OptionData/LongShort") or "Long").strip().lower()
        option_type = (trade_root.findtext("./SwaptionData/OptionData/OptionType") or "Call").strip().lower()
        exercise_sign = 1.0 if option_type == "call" else -1.0
        premium_sign = 1.0 if long_short != "short" else -1.0
        if long_short == "short":
            exercise_sign *= -1.0
        bermudan = self._ir_options_mod.BermudanSwaptionDef(
            trade_id=trade.trade_id,
            exercise_times=exercise_times,
            underlying_legs=underlying_legs,
            exercise_sign=exercise_sign,
            settlement=(trade_root.findtext("./SwaptionData/OptionData/Settlement") or "Physical").strip().lower(),
        )
        notional = float(np.max(np.abs(np.asarray(underlying_legs.get("fixed_notional", [0.0]), dtype=float))))
        result = {
            "definition": bermudan,
            "ccy": ccy,
            "notional": notional,
            "underlying_legs": underlying_legs,
            "index_name": float_index,
            "style": style,
            "premium_records": premium_records,
            "premium_sign": premium_sign,
            "long_short": long_short,
        }
        self._generic_swaption_state_cache[cache_key] = result
        self._generic_swaption_state_xml_cache[xml_cache_key] = result
        return result

    def _build_bermudan_swaption_state(
        self,
        trade: Trade,
        snapshot: XVASnapshot,
        mapped: MappedInputs,
    ) -> Optional[Dict[str, object]]:
        product = trade.product
        if isinstance(product, GenericProduct):
            state = self._build_generic_swaption_state(trade, snapshot)
            if state is not None and str(state.get("style", "")).lower() == "bermudan":
                return state
            return None
        if not isinstance(product, BermudanSwaption):
            return None
        portfolio_xml = mapped.xml_buffers.get("portfolio.xml", "")
        if not portfolio_xml:
            return None
        asof = self._normalized_asof(snapshot)
        try:
            underlying_legs = self._irs_utils.load_swap_legs_from_portfolio_root(
                self._portfolio_root_from_xml(portfolio_xml),
                trade.trade_id,
                asof,
            )
        except Exception:
            return None
        asof_date = self._irs_utils._parse_yyyymmdd(asof)
        live_exercise_dates = []
        for d in product.exercise_dates:
            dt = self._irs_utils._parse_yyyymmdd(d)
            if dt >= asof_date:
                live_exercise_dates.append(dt)
        if len(live_exercise_dates) == 0:
            return None
        exercise_times = np.asarray(
            [float(self._irs_utils._time_from_dates(asof_date, dt, "A365F")) for dt in live_exercise_dates],
            dtype=float,
        )
        option_type = str(getattr(product, "option_type", "Call")).strip().lower()
        exercise_sign = 1.0 if option_type == "call" else -1.0
        if str(product.long_short).strip().lower() == "short":
            exercise_sign *= -1.0
        definition = self._ir_options_mod.BermudanSwaptionDef(
            trade_id=trade.trade_id,
            exercise_times=exercise_times,
            underlying_legs=underlying_legs,
            exercise_sign=exercise_sign,
            settlement=str(product.settlement).strip().lower(),
        )
        notional = float(np.max(np.abs(np.asarray(underlying_legs.get("fixed_notional", [0.0]), dtype=float))))
        return {
            "definition": definition,
            "ccy": str(product.ccy).upper(),
            "notional": notional,
            "underlying_legs": underlying_legs,
            "index_name": str(underlying_legs.get("float_index", "")).upper(),
            "style": "bermudan",
        }

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
                if bool(leg.get("overnight_indexed", False)) and not _is_bma_sifma_index(index_name):
                    leg["quoted_coupon"] = coupon
                    leg["is_historically_fixed"] = fixed_mask
                    continue
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
        quoted_coupon = np.asarray(legs.get("float_coupon", np.zeros_like(s)), dtype=float)
        fixed_mask = np.asarray(legs.get("float_is_historically_fixed", np.zeros(s.shape, dtype=bool)), dtype=bool)
        out = self._irs_utils.compute_realized_float_coupons(
            model=model,
            p0_disc=p0_disc,
            p0_fwd=p0_fwd,
            legs=legs,
            sim_times=sim_times,
            x_paths_on_sim_grid=x_paths_on_sim_grid,
        )
        if np.any(fixed_mask):
            out[fixed_mask, :] = quoted_coupon[fixed_mask, None]
        return out

    def _validated_grid_indices(
        self,
        inputs: _PythonLgmInputs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        times = inputs.times
        valuation_times = inputs.valuation_times
        obs_times = inputs.observation_times
        obs_closeout_times = inputs.observation_closeout_times
        valuation_idx = np.searchsorted(times, valuation_times)
        obs_idx = np.searchsorted(times, obs_times)
        obs_closeout_idx = np.searchsorted(times, obs_closeout_times)
        for name, idx, target in (
            ("Observation", obs_idx, obs_times),
            ("Valuation", valuation_idx, valuation_times),
            ("Sticky closeout", obs_closeout_idx, obs_closeout_times),
        ):
            if idx.size != target.size or np.any(idx >= times.size) or np.any(np.abs(times[idx] - target) > 1.0e-10):
                raise EngineRunError(f"{name} times are not aligned with the fixed pricing grid")
        if len(inputs.observation_dates) == obs_times.size:
            obs_dates = list(inputs.observation_dates)
        else:
            obs_dates = [
                (datetime.fromisoformat(inputs.asof).date() + timedelta(days=int(round(float(t) * 365.0)))).isoformat()
                for t in obs_times
            ]
        return valuation_idx, obs_idx, obs_closeout_idx, obs_dates

    def _fail_on_trade_nan(self, spec: _TradeSpec, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=float)
        if not np.isnan(arr).any():
            return
        idx = np.argwhere(np.isnan(arr))[0]
        raise EngineRunError(
            f"NaN detected in native trade pricing for {spec.trade.trade_id} ({spec.kind}) "
            f"at time_index={int(idx[0])}, path_index={int(idx[1])}"
        )

    def _price_trade_paths(
        self,
        spec: _TradeSpec,
        ctx: _PricingContext,
    ) -> tuple[np.ndarray | None, bool]:
        def _freeze_value(value: object) -> object:
            if isinstance(value, Mapping):
                return tuple(sorted((str(k), _freeze_value(v)) for k, v in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(_freeze_value(v) for v in value)
            try:
                arr = np.asarray(value)
                if arr.ndim > 0:
                    arr = np.ascontiguousarray(arr)
                    if arr.dtype.kind in {"f", "i", "u", "b"}:
                        return arr.shape, arr.dtype.str, arr.tobytes()
                    return arr.shape, arr.dtype.str, tuple(map(str, arr.reshape(-1).tolist()))
            except Exception:
                pass
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            return str(value)

        trade_cache_key = None
        if spec.kind in {"IRS", "RateSwap"}:
            trade_cache_key = (
                "trade_paths",
                spec.kind,
                spec.ccy,
                float(spec.notional),
                _freeze_value(spec.legs),
                _freeze_value(spec.sticky_state),
                ctx.irs_backend is not None,
                tuple(getattr(ctx.shared_fx_sim, "pair_keys", ()) or ()),
            )
            cached_trade = ctx.trade_value_cache.get(trade_cache_key)
            if cached_trade is not None:
                ctx.last_trade_backend = f"{spec.kind.lower()}-cache"
                ctx.last_trade_backend_detail = "economic-definition"
                return cached_trade
        handlers = {
            "IRS": self._price_trade_irs_paths,
            "RateSwap": self._price_trade_rate_swap_paths,
            "FXForward": self._price_trade_fx_forward_paths,
            "CapFloor": self._price_trade_capfloor_paths,
            "Swaption": self._price_trade_swaption_paths,
            "Cashflow": self._price_trade_cashflow_paths,
            "FRA": self._price_trade_fra_paths,
            "InflationSwap": self._price_trade_inflation_swap_paths,
            "InflationCapFloor": self._price_trade_inflation_capfloor_paths,
        }
        handler = handlers.get(spec.kind)
        if handler is None:
            return None, False
        result = handler(spec, ctx)
        if trade_cache_key is not None:
            ctx.trade_value_cache[trade_cache_key] = result
        return result

    def _price_trade_irs_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray, bool]:
        inputs = ctx.inputs
        model = ctx.model
        x_paths = ctx.x_paths
        p_disc = inputs.discount_curves[spec.ccy]
        legs = spec.legs or {}
        index_name = str(legs.get("float_index", spec.trade.additional_fields.get("index", _default_index_for_ccy(spec.ccy))))
        p_fwd = self._resolve_index_curve(inputs, spec.ccy, index_name)
        realized_coupon = self._compute_realized_float_coupons(
            model=model,
            p0_disc=p_disc,
            p0_fwd=p_fwd,
            legs=legs,
            sim_times=inputs.times,
            x_paths_on_sim_grid=x_paths,
        )
        use_exact_overnight = bool(legs.get("float_overnight_indexed", False)) and bool(legs.get("float_is_averaged", False))
        if use_exact_overnight:
            overnight_leg = {
                "kind": "FLOATING",
                "ccy": spec.ccy,
                "notional": np.asarray(legs.get("float_notional", []), dtype=float),
                "sign": np.asarray(legs.get("float_sign", []), dtype=float),
                "start_time": np.asarray(legs.get("float_start_time", []), dtype=float),
                "end_time": np.asarray(legs.get("float_end_time", []), dtype=float),
                "pay_time": np.asarray(legs.get("float_pay_time", []), dtype=float),
                "accrual": np.asarray(legs.get("float_accrual", []), dtype=float),
                "schedule_tenor": "",
                "calendar": "",
                "day_counter": "A365",
                "index_name": str(legs.get("float_index", "")),
                "spread": np.asarray(legs.get("float_spread", []), dtype=float),
                "gearing": np.asarray(legs.get("float_gearing", np.ones_like(np.asarray(legs.get("float_accrual", []), dtype=float))), dtype=float),
                "cap": legs.get("float_cap"),
                "floor": legs.get("float_floor"),
                "overnight_indexed": True,
                "is_averaged": True,
                "has_sub_periods": bool(legs.get("float_has_sub_periods", False)),
                "lookback_days": int(legs.get("float_lookback_days", 0) or 0),
                "rate_cutoff": int(legs.get("float_rate_cutoff", 0) or 0),
                "apply_observation_shift": bool(legs.get("float_apply_observation_shift", False)),
                "local_cap_floor": bool(legs.get("float_local_cap_floor", False)),
                "naked_option": bool(legs.get("float_naked_option", False)),
                "is_in_arrears": bool(legs.get("float_is_in_arrears", False)),
                "fixing_days": int(legs.get("float_fixing_days", 2) or 2),
                "fixing_time": np.asarray(legs.get("float_fixing_time", []), dtype=float),
                "index_accrual": np.asarray(legs.get("float_index_accrual", legs.get("float_accrual", [])), dtype=float),
                "quoted_coupon": np.asarray(legs.get("float_coupon", []), dtype=float),
            }
            vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
            for i, t in enumerate(inputs.times):
                pv = np.zeros((ctx.n_paths,), dtype=float)
                p_t_d = float(p_disc(float(t)))
                pay = np.asarray(legs.get("float_pay_time", []), dtype=float)
                if pay.size > 0:
                    live = (pay >= 0.0) & (pay > float(t) + 1.0e-12)
                    if np.any(live):
                        x_t = x_paths[i, :]
                        disc = model.discount_bond_paths(
                            float(t),
                            pay[live],
                            x_t,
                            p_t_d,
                            np.asarray(self._irs_utils.curve_values(p_disc, pay[live]), dtype=float),
                        )
                        coupons = self._rate_leg_coupon_paths(
                            model,
                            overnight_leg,
                            spec.ccy,
                            inputs,
                            float(t),
                            x_t,
                            snapshot=ctx.snapshot,
                        )[live, :]
                        notionals = np.asarray(legs.get("float_notional", []), dtype=float)[live]
                        accr = np.asarray(legs.get("float_accrual", []), dtype=float)[live]
                        sign = np.asarray(legs.get("float_sign", []), dtype=float)[live]
                        amount = sign[:, None] * notionals[:, None] * accr[:, None] * coupons
                        pv += np.sum(amount * disc, axis=0)
                fixed_pay = np.asarray(legs.get("fixed_pay_time", []), dtype=float)
                if fixed_pay.size > 0:
                    live_fixed = (fixed_pay >= 0.0) & (fixed_pay > float(t) + 1.0e-12)
                    if np.any(live_fixed):
                        x_t = x_paths[i, :]
                        disc = model.discount_bond_paths(
                            float(t),
                            fixed_pay[live_fixed],
                            x_t,
                            p_t_d,
                            np.asarray(self._irs_utils.curve_values(p_disc, fixed_pay[live_fixed]), dtype=float),
                        )
                        amount = np.asarray(legs.get("fixed_amount", []), dtype=float)[live_fixed]
                        pv += np.sum(amount[:, None] * disc, axis=0)
                vals[i, :] = pv
            return self._convert_path_grid_to_reporting_ccy(
                vals,
                local_ccy=spec.ccy,
                report_ccy=inputs.model_ccy,
                inputs=inputs,
                shared_fx_sim=ctx.shared_fx_sim,
            ), spec.ccy.upper() == inputs.model_ccy.upper()
        if ctx.irs_backend is None or bool(legs.get("float_overnight_indexed", False)):
            vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
            for i, t in enumerate(inputs.times):
                overnight_leg = None
                if float(t) <= 1.0e-12 and bool(legs.get("float_overnight_indexed", False)):
                    overnight_leg = {
                        "kind": "FLOATING",
                        "ccy": spec.ccy,
                        "notional": np.asarray(legs.get("float_notional", []), dtype=float),
                        "sign": np.asarray(legs.get("float_sign", []), dtype=float),
                        "start_time": np.asarray(legs.get("float_start_time", []), dtype=float),
                        "end_time": np.asarray(legs.get("float_end_time", []), dtype=float),
                        "pay_time": np.asarray(legs.get("float_pay_time", []), dtype=float),
                        "accrual": np.asarray(legs.get("float_accrual", []), dtype=float),
                        "schedule_tenor": "",
                        "calendar": "",
                        "day_counter": "A365",
                        "index_name": str(legs.get("float_index", "")),
                        "spread": np.asarray(legs.get("float_spread", []), dtype=float),
                        "gearing": np.asarray(legs.get("float_gearing", np.ones_like(np.asarray(legs.get("float_accrual", []), dtype=float))), dtype=float),
                        "cap": legs.get("float_cap"),
                        "floor": legs.get("float_floor"),
                        "overnight_indexed": True,
                        "is_averaged": bool(legs.get("float_is_averaged", False)),
                        "has_sub_periods": bool(legs.get("float_has_sub_periods", False)),
                        "lookback_days": int(legs.get("float_lookback_days", 0) or 0),
                        "rate_cutoff": int(legs.get("float_rate_cutoff", 0) or 0),
                        "apply_observation_shift": bool(legs.get("float_apply_observation_shift", False)),
                        "local_cap_floor": bool(legs.get("float_local_cap_floor", False)),
                        "naked_option": bool(legs.get("float_naked_option", False)),
                        "is_in_arrears": bool(legs.get("float_is_in_arrears", False)),
                        "fixing_days": int(legs.get("float_fixing_days", 2) or 2),
                        "fixing_time": np.zeros_like(np.asarray(legs.get("float_fixing_time", []), dtype=float)),
                        "index_accrual": np.asarray(legs.get("float_index_accrual", legs.get("float_accrual", [])), dtype=float),
                        "quoted_coupon": np.asarray(legs.get("float_coupon", []), dtype=float),
                    }
                    overnight_coupon_paths = None
                    if use_exact_overnight:
                        overnight_coupon_paths = self._rate_leg_coupon_paths(
                            model,
                            overnight_leg,
                            spec.ccy,
                            inputs,
                            float(t),
                            x_paths[i, :],
                            snapshot=ctx.snapshot,
                        )
                    overnight_legs = dict(legs)
                    if use_exact_overnight and overnight_coupon_paths is not None:
                        overnight_legs["float_fixing_time"] = np.asarray(legs.get("float_fixing_time", []), dtype=float)
                        realized_overnight_coupon = np.asarray(overnight_coupon_paths, dtype=float)
                    else:
                        overnight_legs["float_fixing_time"] = np.zeros_like(np.asarray(legs.get("float_fixing_time", []), dtype=float))
                        realized_overnight_coupon = np.asarray(
                            self._rate_leg_coupon_paths(
                                model,
                                overnight_leg,
                                spec.ccy,
                                inputs,
                                float(t),
                                x_paths[0, :],
                                snapshot=ctx.snapshot,
                            ),
                            dtype=float,
                        )
                    vals[i, :] = self._irs_utils.swap_npv_from_ore_legs_dual_curve(
                        model,
                        p_disc,
                        p_fwd,
                        overnight_legs,
                        float(t),
                        x_paths[i, :],
                        realized_float_coupon=realized_overnight_coupon,
                        live_float_coupon=realized_overnight_coupon if use_exact_overnight else None,
                    )
                    continue
                vals[i, :] = self._irs_utils.swap_npv_from_ore_legs_dual_curve(
                    model,
                    p_disc,
                    p_fwd,
                    legs,
                    float(t),
                    x_paths[i, :],
                    realized_float_coupon=realized_coupon,
                    live_float_coupon=realized_coupon if use_exact_overnight else None,
                )
            return self._convert_path_grid_to_reporting_ccy(
                vals,
                local_ccy=spec.ccy,
                report_ccy=inputs.model_ccy,
                inputs=inputs,
                shared_fx_sim=ctx.shared_fx_sim,
            ), spec.ccy.upper() == inputs.model_ccy.upper()
        torch_curve_ctor, torch_pricer, _, torch_device, _, _, _ = ctx.irs_backend
        curve_key = (spec.ccy, index_name)
        curve_state = ctx.irs_curve_cache.get(curve_key)
        if curve_state is None:
            sample_disc = np.unique(
                np.concatenate(
                    (
                        np.asarray([0.0], dtype=float),
                        np.asarray(inputs.times, dtype=float),
                        np.asarray(legs.get("fixed_pay_time", []), dtype=float),
                        np.asarray(legs.get("float_pay_time", []), dtype=float),
                    )
                )
            )
            sample_disc = sample_disc[np.isfinite(sample_disc)]
            sample_disc.sort()
            sample_fwd = np.unique(
                np.concatenate(
                    (
                        sample_disc,
                        np.asarray(legs.get("float_start_time", []), dtype=float),
                        np.asarray(legs.get("float_end_time", []), dtype=float),
                    )
                )
            )
            sample_fwd = sample_fwd[np.isfinite(sample_fwd)]
            sample_fwd.sort()
            curve_state = {
                "disc_curve": torch_curve_ctor(
                    times=sample_disc,
                    dfs=np.asarray([float(p_disc(float(t))) for t in sample_disc], dtype=float),
                    device=torch_device,
                ),
                "fwd_curve": torch_curve_ctor(
                    times=sample_fwd,
                    dfs=np.asarray([float(p_fwd(float(t))) for t in sample_fwd], dtype=float),
                    device=torch_device,
                ),
            }
            ctx.irs_curve_cache[curve_key] = curve_state
        vals = torch_pricer(
            model,
            curve_state["disc_curve"],
                curve_state["fwd_curve"],
                legs,
                np.asarray(inputs.times, dtype=float),
                x_paths,
                realized_float_coupon=realized_coupon,
                live_float_coupon=realized_coupon if use_exact_overnight else None,
                return_numpy=True,
            )
        return self._convert_path_grid_to_reporting_ccy(
            vals,
            local_ccy=spec.ccy,
            report_ccy=inputs.model_ccy,
            inputs=inputs,
            shared_fx_sim=ctx.shared_fx_sim,
        ), spec.ccy.upper() == inputs.model_ccy.upper()

    def _price_trade_rate_swap_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray, bool]:
        inputs = ctx.inputs
        if ctx.irs_backend is None or not self._supports_torch_rate_swap(spec):
            ctx.last_trade_backend = "rateSwap-generic"
            ctx.last_trade_backend_detail = "price_generic_rate_swap"
            return self._price_generic_rate_swap(spec, inputs, ctx.model, ctx.x_paths, ctx.shared_fx_sim, ctx.snapshot), False
        torch_curve_ctor, _, _, torch_device, torch_plain_leg_pricer, _, _ = ctx.irs_backend
        ctx.last_trade_backend = "rateSwap-torch"
        ctx.last_trade_backend_detail = "torch_plain_leg_pricer"
        rate_legs = list((spec.legs or {}).get("rate_legs", []))
        vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
        disc_key = ("disc", spec.ccy)
        disc_curve = ctx.torch_curve_cache.get(disc_key)
        if disc_curve is None:
            sample_disc = [np.asarray(inputs.times, dtype=float)]
            for leg in rate_legs:
                sample_disc.append(np.asarray(leg.get("pay_time", []), dtype=float))
            disc_times = np.unique(np.concatenate(sample_disc))
            disc_times = disc_times[np.isfinite(disc_times)]
            disc_times.sort()
            disc_curve = torch_curve_ctor(
                times=disc_times,
                dfs=np.asarray([float(inputs.discount_curves[spec.ccy](float(t))) for t in disc_times], dtype=float),
                device=torch_device,
            )
            ctx.torch_curve_cache[disc_key] = disc_curve
        p_disc = inputs.discount_curves[spec.ccy]
        report_ccy = inputs.model_ccy.upper()
        for leg in rate_legs:
            kind = str(leg.get("kind", "")).upper()
            leg_ccy = str(leg.get("ccy", spec.ccy)).upper()
            if kind in {"FIXED", "FLOATING"}:
                leg_cache_key = self._rate_leg_pricing_cache_key(spec.ccy, leg)
                cached_vals = ctx.torch_rate_leg_value_cache.get(leg_cache_key)
                if cached_vals is not None:
                    vals += cached_vals
                    continue
                fwd_curve = None
                if kind == "FLOATING":
                    index_name = str(leg.get("index_name", ""))
                    fwd_key = ("fwd", index_name.upper())
                    fwd_curve = ctx.torch_curve_cache.get(fwd_key)
                    if fwd_curve is None:
                        curve = self._resolve_index_curve(inputs, spec.ccy, index_name)
                        sample_fwd = np.unique(
                            np.concatenate(
                                (
                                    np.asarray(inputs.times, dtype=float),
                                    np.asarray(leg.get("start_time", []), dtype=float),
                                    np.asarray(leg.get("end_time", []), dtype=float),
                                )
                            )
                        )
                        sample_fwd = sample_fwd[np.isfinite(sample_fwd)]
                        sample_fwd.sort()
                        fwd_curve = torch_curve_ctor(
                            times=sample_fwd,
                            dfs=np.asarray([float(curve(float(t))) for t in sample_fwd], dtype=float),
                            device=torch_device,
                        )
                        ctx.torch_curve_cache[fwd_key] = fwd_curve
                leg_vals = torch_plain_leg_pricer(
                    ctx.model,
                    disc_curve,
                    leg,
                    np.asarray(inputs.times, dtype=float),
                    ctx.x_paths,
                    fwd_curve=fwd_curve,
                    return_numpy=True,
                )
                principal_pay = np.asarray([], dtype=float)
                principal_amount = np.asarray([], dtype=float)
                if bool(leg.get("notional_initial_exchange", False)) or bool(leg.get("notional_final_exchange", False)):
                    start = np.asarray(leg.get("start_time", []), dtype=float)
                    end = np.asarray(leg.get("end_time", []), dtype=float)
                    notionals = np.asarray(leg.get("notional", []), dtype=float)
                    if notionals.ndim == 1 and notionals.size:
                        pay_times: list[float] = []
                        amounts: list[float] = []
                        if bool(leg.get("notional_initial_exchange", False)):
                            pay_times.append(float(start[0]))
                            amounts.append(float(-notionals[0]))
                        for j in range(max(int(notionals.size) - 1, 0)):
                            pay_times.append(float(end[j]))
                            amounts.append(float(notionals[j + 1] - notionals[j]))
                        if bool(leg.get("notional_final_exchange", False)):
                            pay_times.append(float(end[-1]))
                            amounts.append(float(notionals[-1]))
                        if pay_times:
                            principal_pay = np.asarray(pay_times, dtype=float)
                            principal_amount = np.asarray(amounts, dtype=float)
                if principal_pay.size and principal_amount.size:
                    for i, t in enumerate(inputs.times):
                        live_principal = (principal_pay >= 0.0) & (principal_pay > float(t) + 1.0e-12)
                        if not np.any(live_principal):
                            continue
                        x_t = ctx.x_paths[i, :]
                        p_t = float(p_disc(float(t)))
                        disc_principal = ctx.model.discount_bond_paths(
                            float(t),
                            principal_pay[live_principal],
                            x_t,
                            p_t,
                            np.asarray([float(p_disc(float(T))) for T in principal_pay[live_principal]], dtype=float),
                        )
                        pv = np.sum(principal_amount[live_principal][:, None] * disc_principal, axis=0)
                        leg_vals[i, :] += pv
                leg_vals_report = self._convert_path_grid_to_reporting_ccy(
                    leg_vals,
                    local_ccy=leg_ccy,
                    report_ccy=report_ccy,
                    inputs=inputs,
                    shared_fx_sim=ctx.shared_fx_sim,
                )
                ctx.torch_rate_leg_value_cache[leg_cache_key] = leg_vals_report
                vals += leg_vals_report
                continue
            pay = np.asarray(leg.get("pay_time", []), dtype=float)
            accr = np.asarray(leg.get("accrual", []), dtype=float)
            notionals = np.asarray(leg.get("notional", np.zeros(pay.shape)), dtype=float)
            sign = float(leg.get("sign", 1.0))
            for i, t in enumerate(inputs.times):
                live = (pay >= 0.0) & (pay > float(t) + 1.0e-12)
                if not np.any(live):
                    continue
                x_t = ctx.x_paths[i, :]
                p_t = float(p_disc(float(t)))
                disc = ctx.model.discount_bond_paths(
                    float(t),
                    pay[live],
                    x_t,
                    p_t,
                    np.asarray([float(p_disc(float(T))) for T in pay[live]], dtype=float),
                )
                coupons = self._rate_leg_coupon_paths_torch(
                    ctx.model,
                    leg,
                    spec.ccy,
                    inputs,
                    float(t),
                    x_t,
                    torch_backend=ctx.irs_backend,
                    curve_cache=ctx.torch_curve_cache,
                )[live, :]
                amount = sign * notionals[live, None] * accr[live, None] * coupons
                vals[i, :] += np.sum(amount * disc, axis=0)
        return vals, False

    def _price_trade_fx_forward_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray, bool]:
        return self._price_fx_forward(spec.trade, ctx.inputs, ctx.n_times, ctx.n_paths, shared_sim=ctx.shared_fx_sim), False

    def _price_trade_capfloor_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray | None, bool]:
        if spec.sticky_state is None:
            ctx.last_trade_backend = "capfloor-missing-state"
            ctx.last_trade_backend_detail = "no-sticky-state"
            return None, False
        definition = spec.sticky_state.get("definition")
        index_name = str(spec.sticky_state.get("index_name", ""))
        if definition is None:
            ctx.last_trade_backend = "capfloor-missing-definition"
            ctx.last_trade_backend_detail = "no-definition"
            return None, False
        def _array_key(value: object, *, dtype: object = float) -> tuple[tuple[int, ...], str, bytes]:
            arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            return arr.shape, arr.dtype.str, arr.tobytes()

        capfloor_cache_key = (
            "capfloor_paths",
            spec.ccy,
            index_name.upper(),
            str(definition.option_type).strip().lower(),
            float(definition.position),
            _array_key(definition.start_time),
            _array_key(definition.end_time),
            _array_key(definition.pay_time),
            _array_key(definition.accrual),
            _array_key(definition.notional),
            _array_key(definition.strike),
            _array_key(definition.gearing if definition.gearing is not None else np.ones_like(definition.strike)),
            _array_key(definition.spread if definition.spread is not None else np.zeros_like(definition.strike)),
            _array_key(definition.fixing_time if definition.fixing_time is not None else definition.start_time),
            tuple(np.asarray(definition.fixing_date, dtype=object).tolist()) if definition.fixing_date is not None else None,
            ctx.irs_backend is not None,
        )
        cached_vals = ctx.capfloor_value_cache.get(capfloor_cache_key)
        if cached_vals is not None:
            ctx.last_trade_backend = "capfloor-cache"
            ctx.last_trade_backend_detail = "economic-definition"
            return cached_vals, False
        p_disc = ctx.inputs.discount_curves[spec.ccy]
        p_fwd = self._resolve_index_curve(ctx.inputs, spec.ccy, index_name)
        if ctx.irs_backend is None:
            ctx.last_trade_backend = "capfloor-numpy"
            ctx.last_trade_backend_detail = "capfloor_npv_paths"
            vals = self._ir_options_mod.capfloor_npv_paths(
                model=ctx.model,
                p0_disc=p_disc,
                p0_fwd=p_fwd,
                capfloor=definition,
                times=ctx.inputs.times,
                x_paths=ctx.x_paths,
                lock_fixings=True,
                fixings=self._fixings_lookup(ctx.snapshot),
                fixing_index=index_name,
            )
            ctx.capfloor_value_cache[capfloor_cache_key] = vals
            return vals, False
        torch_curve_ctor, _, _, torch_device, _, _, torch_capfloor_pricer = ctx.irs_backend
        ctx.last_trade_backend = "capfloor-torch"
        ctx.last_trade_backend_detail = "torch_capfloor_pricer"
        disc_key = ("capfloor_disc", spec.ccy)
        disc_curve = ctx.torch_curve_cache.get(disc_key)
        if disc_curve is None:
            disc_times = np.unique(
                np.concatenate(
                    (
                        np.asarray(ctx.inputs.times, dtype=float),
                        np.asarray(definition.start_time, dtype=float),
                        np.asarray(definition.end_time, dtype=float),
                        np.asarray(definition.pay_time, dtype=float),
                        np.asarray(definition.fixing_time if definition.fixing_time is not None else definition.start_time, dtype=float),
                    )
                )
            )
            disc_times = disc_times[np.isfinite(disc_times)]
            disc_times.sort()
            disc_curve = torch_curve_ctor(
                times=disc_times,
                dfs=np.asarray([float(p_disc(float(t))) for t in disc_times], dtype=float),
                device=torch_device,
            )
            ctx.torch_curve_cache[disc_key] = disc_curve
        fwd_key = ("capfloor_fwd", index_name.upper())
        fwd_curve = ctx.torch_curve_cache.get(fwd_key)
        if fwd_curve is None:
            fwd_times = np.unique(
                np.concatenate(
                    (
                        np.asarray(ctx.inputs.times, dtype=float),
                        np.asarray(definition.start_time, dtype=float),
                        np.asarray(definition.end_time, dtype=float),
                        np.asarray(definition.fixing_time if definition.fixing_time is not None else definition.start_time, dtype=float),
                    )
                )
            )
            fwd_times = fwd_times[np.isfinite(fwd_times)]
            fwd_times.sort()
            fwd_curve = torch_curve_ctor(
                times=fwd_times,
                dfs=np.asarray([float(p_fwd(float(t))) for t in fwd_times], dtype=float),
                device=torch_device,
            )
            ctx.torch_curve_cache[fwd_key] = fwd_curve
        vals = torch_capfloor_pricer(
            ctx.model,
            disc_curve,
            fwd_curve,
            definition,
            np.asarray(ctx.inputs.times, dtype=float),
            ctx.x_paths,
            lock_fixings=True,
            return_numpy=True,
        )
        ctx.capfloor_value_cache[capfloor_cache_key] = vals
        return vals, False

    def _price_swaption_premium_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> np.ndarray | None:
        if spec.sticky_state is None:
            return None
        premium_records = tuple(spec.sticky_state.get("premium_records", ()))
        if not premium_records:
            return None
        report_ccy = str(ctx.inputs.model_ccy or spec.ccy).upper()
        asof_date = self._irs_utils._parse_yyyymmdd(ctx.inputs.asof)
        premium_vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
        for i, t in enumerate(ctx.inputs.times):
            pv = np.zeros((ctx.n_paths,), dtype=float)
            for record in premium_records:
                pay_date_text = str(record.get("pay_date") or "").strip()
                if not pay_date_text:
                    continue
                try:
                    pay_date = self._irs_utils._parse_yyyymmdd(pay_date_text)
                except Exception:
                    continue
                pay_time = float(self._irs_utils._time_from_dates(asof_date, pay_date, "A365F"))
                if pay_time <= float(t) + 1.0e-12:
                    continue
                premium_ccy = str(record.get("currency") or spec.ccy).strip().upper() or spec.ccy
                p_disc = ctx.inputs.discount_curves.get(premium_ccy) or ctx.inputs.discount_curves.get(spec.ccy)
                if p_disc is None:
                    return None
                if ctx.shared_fx_sim is not None and premium_ccy in ctx.shared_fx_sim.sim.get("x", {}):
                    local_x_t = np.asarray(ctx.shared_fx_sim.sim["x"][premium_ccy][i, :], dtype=float)
                elif premium_ccy == report_ccy or premium_ccy == spec.ccy:
                    local_x_t = ctx.x_paths[i, :]
                else:
                    local_x_t = ctx.x_paths[i, :]
                p_t = float(p_disc(float(t)))
                p_T = float(p_disc(pay_time))
                disc = ctx.model.discount_bond_paths(
                    float(t),
                    np.asarray([pay_time], dtype=float),
                    local_x_t,
                    p_t,
                    np.asarray([p_T], dtype=float),
                )[0]
                local_pv = float(record.get("amount", 0.0)) * np.asarray(disc, dtype=float)
                pv += self._convert_amount_to_reporting_ccy(
                    local_pv,
                    local_ccy=premium_ccy,
                    report_ccy=report_ccy,
                    inputs=ctx.inputs,
                    shared_fx_sim=ctx.shared_fx_sim,
                    time_index=i,
                )
            premium_vals[i, :] = pv
        return premium_vals

    def _price_trade_swaption_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray | None, bool]:
        if spec.sticky_state is None:
            return None, False
        definition = spec.sticky_state.get("definition")
        if definition is None:
            return None, False
        premium_sign = float(spec.sticky_state.get("premium_sign", 1.0))
        def _array_key(value: object, *, dtype: object = float) -> tuple[tuple[int, ...], str, bytes]:
            arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            return arr.shape, arr.dtype.str, arr.tobytes()

        def _leg_key(legs: Mapping[str, object]) -> tuple[tuple[str, tuple[tuple[int, ...], str, bytes]], ...]:
            out = []
            for key in sorted(legs):
                value = legs[key]
                try:
                    out.append((str(key), _array_key(value)))
                except Exception:
                    out.append((str(key), _array_key(np.asarray(value, dtype=object), dtype=object)))
            return tuple(out)

        premium_records = tuple(spec.sticky_state.get("premium_records", ()))
        premium_key = tuple(
            tuple(sorted((str(k), str(v)) for k, v in dict(record).items()))
            for record in premium_records
        )
        swaption_cache_key = (
            "swaption_paths",
            spec.ccy,
            str(spec.sticky_state.get("index_name", "")).upper(),
            str(spec.sticky_state.get("style", "")).lower(),
            str(getattr(definition, "settlement", "")).lower(),
            float(getattr(definition, "exercise_sign", 0.0)),
            float(premium_sign),
            _array_key(getattr(definition, "exercise_times", np.asarray([], dtype=float))),
            _leg_key(getattr(definition, "underlying_legs", {})),
            premium_key,
            ctx.irs_backend is not None,
        )
        cached_vals = ctx.swaption_value_cache.get(swaption_cache_key)
        if cached_vals is not None:
            ctx.last_trade_backend = "swaption-cache"
            ctx.last_trade_backend_detail = "economic-definition"
            return cached_vals, False
        p_disc = ctx.inputs.discount_curves[spec.ccy]
        float_index = str(spec.sticky_state.get("index_name", ""))
        p_fwd = self._resolve_index_curve(ctx.inputs, spec.ccy, float_index)
        if ctx.irs_backend is not None:
            torch_curve_ctor, torch_pricer, _, torch_device, _, _, _ = ctx.irs_backend
            curve_key = (spec.ccy, float_index)
            curve_state = ctx.irs_curve_cache.get(curve_key)
            legs = definition.underlying_legs
            if curve_state is None:
                sample_disc = np.unique(
                    np.concatenate(
                        (
                            np.asarray([0.0], dtype=float),
                            np.asarray(ctx.inputs.times, dtype=float),
                            np.asarray(legs.get("fixed_pay_time", []), dtype=float),
                            np.asarray(legs.get("float_pay_time", []), dtype=float),
                        )
                    )
                )
                sample_disc = sample_disc[np.isfinite(sample_disc)]
                sample_disc.sort()
                sample_fwd = np.unique(
                    np.concatenate(
                        (
                            sample_disc,
                            np.asarray(legs.get("float_start_time", []), dtype=float),
                            np.asarray(legs.get("float_end_time", []), dtype=float),
                        )
                    )
                )
                sample_fwd = sample_fwd[np.isfinite(sample_fwd)]
                sample_fwd.sort()
                disc_curve = torch_curve_ctor(
                    times=sample_disc,
                    dfs=np.asarray([float(p_disc(float(t))) for t in sample_disc], dtype=float),
                    device=torch_device,
                )
                fwd_curve = torch_curve_ctor(
                    times=sample_fwd,
                    dfs=np.asarray([float(p_fwd(float(t))) for t in sample_fwd], dtype=float),
                    device=torch_device,
                )
                curve_state = {
                    "disc_curve": disc_curve,
                    "fwd_curve": fwd_curve,
                }
                ctx.irs_curve_cache[curve_key] = curve_state
            signed_swap = torch_pricer(
                ctx.model,
                curve_state["disc_curve"],
                curve_state["fwd_curve"],
                legs,
                np.asarray(ctx.inputs.times, dtype=float),
                ctx.x_paths,
                exercise_into_whole_periods=True,
                deterministic_fixings_cutoff=0.0,
                return_numpy=True,
            )
            vals = self._ir_options_mod.bermudan_npv_paths_from_underlying(
                model=ctx.model,
                p0_disc=p_disc,
                bermudan=definition,
                times=ctx.inputs.times,
                x_paths=ctx.x_paths,
                signed_swap=float(definition.exercise_sign) * np.asarray(signed_swap, dtype=float),
            )
            premium_vals = self._price_swaption_premium_paths(spec, ctx)
            if premium_vals is not None:
                vals = vals - premium_sign * premium_vals
            ctx.swaption_value_cache[swaption_cache_key] = vals
            return vals, False
        vals = self._ir_options_mod.bermudan_npv_paths(
            model=ctx.model,
            p0_disc=p_disc,
            p0_fwd=p_fwd,
            bermudan=definition,
            times=ctx.inputs.times,
            x_paths=ctx.x_paths,
        )
        premium_vals = self._price_swaption_premium_paths(spec, ctx)
        if premium_vals is not None:
            vals = vals - premium_sign * premium_vals
        ctx.swaption_value_cache[swaption_cache_key] = vals
        return vals, False

    def _price_trade_cashflow_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray | None, bool]:
        if spec.sticky_state is None:
            return None, False
        pay = np.asarray(spec.sticky_state.get("pay_time", []), dtype=float)
        amount = np.asarray(spec.sticky_state.get("amount", []), dtype=float)
        if pay.size == 0:
            return np.zeros((ctx.n_times, ctx.n_paths), dtype=float), False
        if amount.size != pay.size:
            return None, False
        p_disc = ctx.inputs.discount_curves[spec.ccy]
        report_ccy = ctx.inputs.model_ccy.upper()
        multi_ccy = spec.ccy.upper() != report_ccy
        vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
        for i, t in enumerate(ctx.inputs.times):
            live = (pay >= 0.0) & (pay > float(t) + 1.0e-12)
            if not np.any(live):
                continue
            if ctx.shared_fx_sim is not None and spec.ccy in ctx.shared_fx_sim.sim.get("x", {}):
                local_x_t = np.asarray(ctx.shared_fx_sim.sim["x"][spec.ccy][i, :], dtype=float)
            elif spec.ccy == report_ccy:
                local_x_t = ctx.x_paths[i, :]
            elif multi_ccy:
                return None, False
            else:
                local_x_t = ctx.x_paths[i, :]
            p_t = float(p_disc(float(t)))
            disc = ctx.model.discount_bond_paths(
                float(t),
                pay[live],
                local_x_t,
                p_t,
                np.asarray([float(p_disc(float(T))) for T in pay[live]], dtype=float),
            )
            local_pv = np.sum(amount[live][:, None] * disc, axis=0)
            vals[i, :] = self._convert_amount_to_reporting_ccy(
                local_pv,
                local_ccy=spec.ccy,
                report_ccy=report_ccy,
                inputs=ctx.inputs,
                shared_fx_sim=ctx.shared_fx_sim,
                time_index=i,
            )
        return vals, False

    def _price_trade_fra_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray | None, bool]:
        state = spec.sticky_state or {}
        try:
            start_t = float(state["start_time"])
            end_t = float(state["end_time"])
            accrual = float(state["accrual"])
            notional = float(state["notional"])
            strike = float(state["strike"])
            position = float(state.get("position", 1.0))
            index_name = str(state.get("index_name", ""))
        except Exception:
            return None, False
        if accrual <= 0.0 or end_t <= start_t:
            return None, False
        inputs = ctx.inputs
        model = ctx.model
        p_disc = inputs.discount_curves[spec.ccy]
        p_fwd = self._resolve_index_curve(inputs, spec.ccy, index_name)
        vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
        for i, t_raw in enumerate(inputs.times):
            t = float(t_raw)
            if t >= start_t - 1.0e-12:
                continue
            x_t = ctx.x_paths[i, :]
            p0_disc_t = float(p_disc(t))
            p0_fwd_t = float(p_fwd(t))
            p_t_start_disc = model.discount_bond(t, start_t, x_t, p0_disc_t, float(p_disc(start_t)))
            p_t_start_fwd = model.discount_bond(t, start_t, x_t, p0_fwd_t, float(p_fwd(start_t)))
            p_t_end_fwd = model.discount_bond(t, end_t, x_t, p0_fwd_t, float(p_fwd(end_t)))
            forward = (p_t_start_fwd / np.maximum(p_t_end_fwd, 1.0e-12) - 1.0) / accrual
            settlement = position * notional * accrual * (forward - strike) / np.maximum(1.0 + accrual * forward, 1.0e-12)
            vals[i, :] = settlement * p_t_start_disc
        return vals, True

    def _price_trade_inflation_swap_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray | None, bool]:
        trade_product = spec.trade.product
        assert isinstance(trade_product, InflationSwap)
        curve_key = (
            str(trade_product.index).upper(),
            "YY" if str(trade_product.inflation_type).upper() == "YY" else "ZC",
        )
        inflation_curve = ctx.inputs.inflation_curves.get(curve_key)
        if inflation_curve is None or str(trade_product.pay_leg).lower() == "float":
            return None, False
        p_disc = ctx.inputs.discount_curves[spec.ccy]
        vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
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
                    for t in ctx.inputs.times
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
                    for t in ctx.inputs.times
                ],
                dtype=float,
            )
        vals[:, :] = profile[:, None]
        return vals, False

    def _price_trade_inflation_capfloor_paths(self, spec: _TradeSpec, ctx: _PricingContext) -> tuple[np.ndarray | None, bool]:
        trade_product = spec.trade.product
        assert isinstance(trade_product, InflationCapFloor)
        curve_key = (
            str(trade_product.index).upper(),
            "YY" if str(trade_product.inflation_type).upper() == "YY" else "ZC",
        )
        inflation_curve = ctx.inputs.inflation_curves.get(curve_key)
        if inflation_curve is None:
            return None, False
        p_disc = ctx.inputs.discount_curves[spec.ccy]
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
                for t in ctx.inputs.times
            ],
            dtype=float,
        )
        vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
        vals[:, :] = profile[:, None]
        return vals, False

    def _resolve_index_curve(
        self,
        inputs: _PythonLgmInputs,
        ccy: str,
        index_name: str,
    ) -> Callable[[float], float]:
        key = str(index_name).strip().upper()
        if key in {"USD-SIFMA-1W", "USD-SIFMA-7D"}:
            key = "USD-SIFMA"
        elif key in {"USD-BMA-1W", "USD-BMA-7D"}:
            key = "USD-BMA"
        family = _forward_index_family(key, inputs.swap_index_forward_tenors)
        if key and key in inputs.forward_curves_by_name:
            return inputs.forward_curves_by_name[key]
        normalized_key = _normalize_curve_lookup_key(key)
        if normalized_key and normalized_key != key:
            for candidate_name, curve in inputs.forward_curves_by_name.items():
                if _normalize_curve_lookup_key(candidate_name) == normalized_key:
                    return curve
        if _is_bma_sifma_index(key):
            raise EngineRunError(
                f"{key} curve is required but was not built from BMA_SWAP/RATIO market quotes"
            )
        mapped_tenor = inputs.swap_index_forward_tenors.get(key, "")
        if not mapped_tenor and normalized_key:
            for candidate_name, candidate_tenor in inputs.swap_index_forward_tenors.items():
                if _normalize_curve_lookup_key(candidate_name) == normalized_key:
                    mapped_tenor = candidate_tenor
                    break
        mapped_tenor = _normalize_forward_tenor_family(mapped_tenor)
        if mapped_tenor and mapped_tenor in inputs.forward_curves_by_tenor.get(ccy, {}):
            return inputs.forward_curves_by_tenor[ccy][mapped_tenor]
        tenor_match = re.search(r"(\d+[YMWD])$", key)
        if tenor_match:
            tenor = _normalize_forward_tenor_family(tenor_match.group(1).upper())
            if tenor in inputs.forward_curves_by_tenor.get(ccy, {}):
                return inputs.forward_curves_by_tenor[ccy][tenor]
        if family == "1D":
            overnight_curves = inputs.forward_curves_by_tenor.get(ccy, {})
            overnight_curve = overnight_curves.get("1D") or overnight_curves.get("0D")
            if overnight_curve is not None:
                return overnight_curve
            return inputs.discount_curves[ccy]
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
        curve_values = self._irs_utils.curve_values
        if float(t) <= 1.0e-12 and np.ptp(np.asarray(x_t, dtype=float)) <= 1.0e-14:
            cache_key = (
                id(curve),
                round(float(t), 12),
                round(float(effective_start), 12),
                round(float(tenor_years), 12),
                int(np.asarray(x_t, dtype=float).size),
            )
            cached = self._par_swap_deterministic_cache.get(cache_key)
            if cached is not None:
                return cached
            pay_dfs = np.asarray(curve_values(curve, pay_times), dtype=float)
            p_start = float(curve(effective_start))
            p_end = float(curve(float(maturity)))
            annuity = 0.0
            prev = effective_start
            for pay, pay_df in zip(pay_times, pay_dfs):
                tau = max(float(pay) - prev, 1.0e-8)
                annuity += tau * float(pay_df)
                prev = float(pay)
            annuity = annuity if abs(annuity) >= 1.0e-12 else 1.0e-12
            par_rate = (p_start - p_end) / annuity
            out = np.full_like(np.asarray(x_t, dtype=float), par_rate, dtype=float)
            self._par_swap_deterministic_cache[cache_key] = out
            return out
        p_t = float(curve(t))
        terminal_times = np.asarray([effective_start, float(maturity)], dtype=float)
        terminal_dfs = np.asarray(curve_values(curve, terminal_times), dtype=float)
        p_start = model.discount_bond(t, effective_start, x_t, p_t, float(terminal_dfs[0]))
        p_end = model.discount_bond(t, float(maturity), x_t, p_t, float(terminal_dfs[1]))
        annuity = np.zeros_like(x_t, dtype=float)
        prev = effective_start
        pay_dfs = np.asarray(curve_values(curve, pay_times), dtype=float)
        for pay, pay_df in zip(pay_times, pay_dfs):
            tau = max(float(pay) - prev, 1.0e-8)
            annuity += tau * model.discount_bond(t, float(pay), x_t, p_t, float(pay_df))
            prev = float(pay)
        annuity = np.where(np.abs(annuity) < 1.0e-12, 1.0e-12, annuity)
        return (p_start - p_end) / annuity

    def _interp_scalar_curve(self, points: Sequence[Tuple[float, float]], x: float) -> float:
        if not points:
            return 0.0
        pts = sorted((float(t), float(v)) for t, v in points)
        return float(
            self._irs_utils.interpolate_linear_flat(
                float(x),
                np.asarray([t for t, _ in pts], dtype=float),
                np.asarray([v for _, v in pts], dtype=float),
            )
        )

    def _finite_optional_rate(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            rate = float(value)
        except Exception:
            return None
        return rate if math.isfinite(rate) else None

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

    def _capfloor_normal_vol(self, snapshot: XVASnapshot, *, ccy: str, expiry_time: float, strike: float) -> float:
        """Interpolate a normal cap/floor vol from market quotes.

        The legacy ORE examples store cap/floor normal vols in ``marketdata.csv``
        using keys of the form ``CAPFLOOR/RATE_NVOL/<CCY>/<EXPIRY>/.../<STRIKE>``.
        This helper extracts that grid and uses a simple bilinear interpolation
        with flat extrapolation.
        """

        key = (id(snapshot.market.raw_quotes), ccy.upper())
        grid = self._capfloor_vol_cache.get(key)
        if grid is None:
            points: dict[float, dict[float, float]] = {}
            for quote in snapshot.market.raw_quotes:
                raw_key = str(getattr(quote, "key", "")).strip().upper()
                if not raw_key.startswith(f"CAPFLOOR/RATE_NVOL/{ccy.upper()}/"):
                    continue
                parts = raw_key.split("/")
                if len(parts) < 8:
                    continue
                expiry_txt = parts[3]
                strike_txt = parts[-1]
                try:
                    expiry = float(_parse_ore_tenor_to_years(expiry_txt))
                    strike_value = float(strike_txt)
                    vol = float(getattr(quote, "value", 0.0))
                except Exception:
                    continue
                points.setdefault(expiry, {})[strike_value] = vol
            expiries = sorted(points)
            strikes = sorted({s for row in points.values() for s in row})
            matrix = np.full((len(expiries), len(strikes)), np.nan, dtype=float)
            for i, expiry in enumerate(expiries):
                row = points.get(expiry, {})
                for j, strike_value in enumerate(strikes):
                    if strike_value in row:
                        matrix[i, j] = row[strike_value]
            grid = {"expiries": np.asarray(expiries, dtype=float), "strikes": np.asarray(strikes, dtype=float), "matrix": matrix}
            self._capfloor_vol_cache[key] = grid

        expiries = np.asarray(grid["expiries"], dtype=float)
        strikes = np.asarray(grid["strikes"], dtype=float)
        matrix = np.asarray(grid["matrix"], dtype=float)
        if expiries.size == 0 or strikes.size == 0:
            return 0.01

        t = float(expiry_time)
        k = float(strike)
        t = float(np.clip(t, expiries[0], expiries[-1]))
        k = float(np.clip(k, strikes[0], strikes[-1]))

        def _interp_row(i: int) -> float:
            row = matrix[i, :]
            mask = np.isfinite(row)
            if not np.any(mask):
                return float(np.nanmean(matrix[np.isfinite(matrix)])) if np.any(np.isfinite(matrix)) else 0.01
            return float(np.interp(k, strikes[mask], row[mask], left=row[mask][0], right=row[mask][-1]))

        if expiries.size == 1:
            return _interp_row(0)
        hi = int(np.searchsorted(expiries, t, side="right"))
        hi = min(max(hi, 1), expiries.size - 1)
        lo = hi - 1
        v_lo = _interp_row(lo)
        v_hi = _interp_row(hi)
        t_lo = float(expiries[lo])
        t_hi = float(expiries[hi])
        if abs(t_hi - t_lo) <= 1.0e-12:
            return v_lo
        w = (t - t_lo) / (t_hi - t_lo)
        return float((1.0 - w) * v_lo + w * v_hi)

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
        snapshot: XVASnapshot | None = None,
    ) -> np.ndarray:
        x_arr = np.asarray(x_t, dtype=float)
        cache_key = (
            self._rate_leg_pricing_cache_key(ccy, leg),
            round(float(t), 12),
            int(x_arr.ctypes.data),
            int(x_arr.size),
        )
        cached = self._coupon_path_cache.get(cache_key)
        if cached is not None:
            return cached
        kind = str(leg.get("kind", "")).upper()
        start = np.asarray(leg.get("start_time", []), dtype=float)
        end = np.asarray(leg.get("end_time", []), dtype=float)
        fixing = np.asarray(leg.get("fixing_time", start), dtype=float)
        quoted = np.asarray(leg.get("quoted_coupon", np.zeros(start.shape)), dtype=float)
        fixed_mask = np.asarray(leg.get("is_historically_fixed", np.zeros(start.shape, dtype=bool)), dtype=bool)
        spread = np.asarray(leg.get("spread", np.zeros(start.shape)), dtype=float)
        gearing = np.asarray(leg.get("gearing", np.ones(start.shape)), dtype=float)
        coupons = np.zeros((start.size, x_arr.size), dtype=float)

        if (
            bool(leg.get("overnight_indexed", False))
            and not _is_bma_sifma_index(str(leg.get("index_name", "")))
            and snapshot is not None
        ):
            from pythonore.compute.overnight_capfloor_shim import price_average_overnight_coupon_paths, price_overnight_capfloor_coupon_paths

            if bool(leg.get("is_averaged", False)):
                coupons = price_average_overnight_coupon_paths(
                    self,
                    model=model,
                    inputs=inputs,
                    leg=leg,
                    ccy=ccy,
                    t=t,
                    x_t=x_arr,
                    snapshot=snapshot,
                )
            else:
                coupons = price_overnight_capfloor_coupon_paths(
                    self, inputs=inputs, leg=leg, ccy=ccy, t=t, x_t=x_arr, snapshot=snapshot
                )
            fixed_rows = fixed_mask
            if np.any(fixed_rows):
                coupons[fixed_rows, :] = quoted[fixed_rows, None]
            self._coupon_path_cache[cache_key] = coupons
            return coupons
        for i in range(start.size):
            if fixed_mask[i] or fixing[i] <= t + 1.0e-12:
                base = np.full(x_arr.shape, quoted[i], dtype=float)
            elif kind == "FLOATING":
                curve = self._resolve_index_curve(inputs, ccy, str(leg.get("index_name", "")))
                p_t = float(curve(t))
                effective_start = max(float(start[i]), float(t))
                leg_dfs = np.asarray(self._irs_utils.curve_values(curve, np.asarray([effective_start, float(end[i])], dtype=float)), dtype=float)
                p_s = model.discount_bond(t, effective_start, x_arr, p_t, float(leg_dfs[0]))
                p_e = model.discount_bond(t, float(end[i]), x_arr, p_t, float(leg_dfs[1]))
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
                base = np.full(x_arr.shape, ql_rate, dtype=float) if ql_rate is not None else self._par_swap_rate_paths(model, curve, t, x_arr, float(start[i]), tenor_years)
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
                rate1 = np.full(x_arr.shape, ql_rate1, dtype=float) if ql_rate1 is not None else self._par_swap_rate_paths(model, curve1, t, x_arr, float(start[i]), _parse_ore_tenor_to_years(tenor1.group(1)) if tenor1 else 10.0)
                rate2 = np.full(x_arr.shape, ql_rate2, dtype=float) if ql_rate2 is not None else self._par_swap_rate_paths(model, curve2, t, x_arr, float(start[i]), _parse_ore_tenor_to_years(tenor2.group(1)) if tenor2 else 2.0)
                base = rate1 - rate2
            else:
                base = np.zeros_like(x_arr, dtype=float)

            raw_coupon = gearing[i] * base + spread[i]
            coupon = raw_coupon
            if kind == "CMSSPREAD":
                cap = leg.get("cap")
                floor = leg.get("floor")
                ql_rate = None
                if not (fixed_mask[i] or fixing[i] <= t + 1.0e-12) and t <= 1.0e-12:
                    ql_rate = self._finite_optional_rate(
                        self._static_ql_cmsspread_rate(
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
                if not np.all(np.isfinite(coupon)):
                    coupon = self._capped_floored_rate(
                        raw_coupon,
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
                    ql_raw = self._finite_optional_rate(
                        self._static_ql_cmsspread_rate(
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
                    )
                    if ql_raw is not None:
                        ql_capped_rate_fn = lambda cap, floor, idx1_=idx1, idx2_=idx2, pt=pay_time, fd=fixing_days, dc_=dc_name, ia=in_arrears, raw=raw_coupon: np.full_like(
                            raw,
                            self._finite_optional_rate(
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
                                )
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
                if not np.all(np.isfinite(coupon)):
                    coupon = np.zeros_like(raw_coupon, dtype=float) if bool(leg.get("naked_option", False)) else raw_coupon.copy()
                    coupon = coupon + self._digital_option_rate(
                        raw_coupon,
                        float(leg.get("call_strike", float("nan"))),
                        float(leg.get("call_payoff", float("nan"))),
                        is_call=True,
                        long_short=float(leg.get("call_position", 1.0)),
                        fixed_mode=True,
                        atm_included=bool(leg.get("is_call_atm_included", False)),
                    )
                    coupon = coupon + self._digital_option_rate(
                        raw_coupon,
                        float(leg.get("put_strike", float("nan"))),
                        float(leg.get("put_payoff", float("nan"))),
                        is_call=False,
                        long_short=float(leg.get("put_position", 1.0)),
                        fixed_mode=True,
                        atm_included=bool(leg.get("is_put_atm_included", False)),
                    )
            if not np.all(np.isfinite(coupon)):
                fallback = float(quoted[i]) if np.isfinite(quoted[i]) else 0.0
                coupon = np.where(np.isfinite(coupon), coupon, fallback)
            coupons[i, :] = coupon
        self._coupon_path_cache[cache_key] = coupons
        return coupons

    def _price_generic_rate_swap(
        self,
        spec: _TradeSpec,
        inputs: _PythonLgmInputs,
        model: Any,
        x_paths: np.ndarray,
        shared_fx_sim: _SharedFxSimulation | None = None,
        snapshot: XVASnapshot | None = None,
    ) -> np.ndarray:
        legs_payload = spec.legs or {}
        rate_legs = list(legs_payload.get("rate_legs", []))
        n_times = int(inputs.times.size)
        n_paths = int(x_paths.shape[1])
        report_ccy = inputs.model_ccy.upper()
        multi_ccy = len(_rate_leg_currencies(legs_payload, spec.ccy)) > 1
        asof_dt = self._irs_utils._parse_yyyymmdd(inputs.asof)
        fixings = self._fixings_lookup(snapshot) if snapshot is not None else {}
        vals = np.zeros((n_times, n_paths), dtype=float)
        frozen_coupon_paths: Dict[int, np.ndarray] = {}

        def _fx_pair_from_index(fx_index: str, foreign_ccy: str, leg_ccy: str) -> str | None:
            txt = str(fx_index or "").strip().upper().replace("-", "/")
            parts = [p for p in txt.split("/") if p]
            if len(parts) >= 3 and parts[0] == "FX":
                return f"{parts[-2]}/{parts[-1]}"
            if foreign_ccy and leg_ccy:
                return f"{foreign_ccy.upper()}/{leg_ccy.upper()}"
            return None

        def _spot_at_time(paths: np.ndarray, t_fix: float) -> np.ndarray:
            times = np.asarray(inputs.times, dtype=float)
            if paths.ndim != 2 or paths.shape[0] == 0:
                return np.zeros((n_paths,), dtype=float)
            if t_fix <= times[0]:
                out = np.asarray(paths[0, :], dtype=float)
            elif t_fix >= times[-1]:
                out = np.asarray(paths[-1, :], dtype=float)
            else:
                hi = int(np.searchsorted(times, float(t_fix), side="right"))
                hi = min(max(hi, 1), times.size - 1)
                lo = hi - 1
                w = (float(t_fix) - float(times[lo])) / max(float(times[hi] - times[lo]), 1.0e-12)
                out = (1.0 - w) * np.asarray(paths[lo, :], dtype=float) + w * np.asarray(paths[hi, :], dtype=float)
            if np.all(np.isfinite(out)):
                return out
            return np.full((n_paths,), 1.0, dtype=float)

        for leg_idx, leg in enumerate(rate_legs):
            kind = str(leg.get("kind", "")).upper()
            if kind not in {"CMS", "CMSSPREAD", "DIGITALCMSSPREAD"}:
                continue
            leg_ccy = str(leg.get("ccy", spec.ccy)).strip().upper() or spec.ccy
            if shared_fx_sim is not None and leg_ccy in shared_fx_sim.sim.get("x", {}):
                leg_x_paths = np.asarray(shared_fx_sim.sim["x"][leg_ccy], dtype=float)
            else:
                leg_x_paths = x_paths
            frozen_coupon_paths[leg_idx] = self._rate_leg_coupon_paths(
                model,
                leg,
                leg_ccy,
                inputs,
                0.0,
                leg_x_paths[0, :],
                snapshot=snapshot,
            )
        for i, t in enumerate(inputs.times):
            pv = np.zeros((n_paths,), dtype=float)
            for leg_idx, leg in enumerate(rate_legs):
                kind = str(leg.get("kind", "")).upper()
                leg_ccy = str(leg.get("ccy", spec.ccy)).strip().upper() or spec.ccy
                p_disc = inputs.discount_curves[leg_ccy]
                if shared_fx_sim is not None and leg_ccy in shared_fx_sim.sim.get("x", {}):
                    leg_x_t = np.asarray(shared_fx_sim.sim["x"][leg_ccy][i, :], dtype=float)
                else:
                    leg_x_t = x_paths[i, :]
                pay = np.asarray(leg.get("pay_time", []), dtype=float)
                start = np.asarray(leg.get("start_time", []), dtype=float)
                end = np.asarray(leg.get("end_time", []), dtype=float)
                accr = np.asarray(leg.get("accrual", []), dtype=float)
                live = (pay >= 0.0) & (pay > float(t) + 1.0e-12)
                if not np.any(live):
                    continue
                pay_live = pay[live]
                p_t = float(p_disc(float(t)))
                disc = model.discount_bond_paths(
                    float(t),
                    pay_live,
                    leg_x_t,
                    p_t,
                    np.asarray(self._irs_utils.curve_values(p_disc, pay_live), dtype=float),
                )
                if kind == "CASHFLOW":
                    amount = np.asarray(leg.get("amount", np.zeros(pay.shape)), dtype=float)[live]
                    if float(t) <= 1.0e-12 and report_ccy == inputs.model_ccy.upper() and "pv_base" in leg:
                        pv_base = np.asarray(leg.get("pv_base", np.zeros(pay.shape)), dtype=float)[live]
                        pv += np.full((n_paths,), float(np.sum(pv_base)), dtype=float)
                        continue
                    leg_pv = np.sum(amount[:, None] * disc, axis=0)
                    pv += self._convert_amount_to_reporting_ccy(
                        leg_pv,
                        local_ccy=leg_ccy,
                        report_ccy=report_ccy,
                        inputs=inputs,
                        shared_fx_sim=shared_fx_sim,
                        time_index=i,
                    )
                    continue
                if kind == "FIXED":
                    amount = np.asarray(leg.get("amount", np.zeros(pay.shape)), dtype=float)[live]
                    leg_pv = np.sum(amount[:, None] * disc, axis=0)
                    pv += self._convert_amount_to_reporting_ccy(
                        leg_pv,
                        local_ccy=leg_ccy,
                        report_ccy=report_ccy,
                        inputs=inputs,
                        shared_fx_sim=shared_fx_sim,
                        time_index=i,
                    )
                    continue
                notionals = np.asarray(leg.get("notional", np.zeros(pay.shape)), dtype=float)
                fx_reset = leg.get("fx_reset")
                sign = float(leg.get("sign", 1.0))
                notional_initial_exchange = bool(leg.get("notional_initial_exchange", False))
                notional_final_exchange = bool(leg.get("notional_final_exchange", False))
                notional_amort_exchange = bool(
                    leg.get("notional_amortizing_exchange", leg.get("notional_amort_exchange", False))
                )
                principal_pay = np.asarray([], dtype=float)
                principal_amount = np.asarray([], dtype=float)
                if isinstance(fx_reset, dict):
                    foreign_amount = float(fx_reset.get("foreign_amount", 0.0))
                    fx_index = str(fx_reset.get("fx_index", "")).strip().upper()
                    foreign_ccy = str(fx_reset.get("foreign_currency", "")).strip().upper()
                    fx_pair = _fx_pair_from_index(fx_index, foreign_ccy, leg_ccy)
                    spot0 = _spot_from_quotes(foreign_ccy + leg_ccy, inputs, default=1.0)
                    p_dom_fx = inputs.discount_curves.get(leg_ccy)
                    p_for_fx = inputs.discount_curves.get(foreign_ccy)
                    fx_forward_nodes = []
                    if fx_pair is not None:
                        direct_pair = fx_pair.replace("/", "").upper()
                        inverse_pair = "".join(reversed([direct_pair[:3], direct_pair[3:]])) if len(direct_pair) == 6 else ""
                        fx_forward_nodes = list(inputs.fx_forwards.get(direct_pair, []))
                        if not fx_forward_nodes and inverse_pair:
                            fx_forward_nodes = list(inputs.fx_forwards.get(inverse_pair, []))
                    fx_paths = None
                    inverted = False
                    if shared_fx_sim is not None and fx_pair is not None:
                        sim_s = shared_fx_sim.sim.get("s", {})
                        if fx_pair in sim_s:
                            fx_paths = np.asarray(sim_s[fx_pair], dtype=float)
                        else:
                            base, quote = fx_pair.split("/", 1)
                            inv = f"{quote}/{base}"
                            if inv in sim_s:
                                fx_paths = np.asarray(sim_s[inv], dtype=float)
                                inverted = True
                    if foreign_amount != 0.0:
                        coupon_notionals = np.empty((pay.size, n_paths), dtype=float)
                        fx_fixing_days = int(fx_reset.get("fixing_days", 2) or 2)
                        fx_fixing_calendar = str(fx_reset.get("fixing_calendar", "") or leg.get("calendar", leg_ccy))
                        fx_points_factor = float(fx_reset.get("points_factor", 10000.0) or 10000.0)
                        for j in range(pay.size):
                            if j == 0 and notionals.ndim == 1 and notionals.size:
                                coupon_notionals[j, :] = float(notionals[0])
                                continue
                            start_date = self._irs_utils._parse_yyyymmdd(self._date_from_time_cached(snapshot, float(start[j]))) if snapshot is not None else asof_dt
                            fix_date = self._irs_utils._advance_business_days(start_date, -fx_fixing_days, fx_fixing_calendar)
                            key = (fx_index, fix_date.isoformat())
                            if key in fixings:
                                fx_value = float(fixings[key])
                                if inverted:
                                    fx_value = 1.0 / max(fx_value, 1.0e-12)
                                coupon_notionals[j, :] = foreign_amount * fx_value
                                continue
                            if fix_date <= asof_dt and fx_paths is None:
                                pair6 = f"{foreign_ccy}{leg_ccy}"
                                fx_value = _spot_from_quotes(pair6, inputs, default=1.0)
                                if inverted:
                                    fx_value = 1.0 / max(float(fx_value), 1.0e-12)
                                coupon_notionals[j, :] = foreign_amount * float(fx_value)
                            else:
                                if fx_paths is not None:
                                    fix_t = float(self._irs_utils._time_from_dates(asof_dt, fix_date, "A365F"))
                                    fx_value = _spot_at_time(fx_paths, fix_t)
                                    fx_value = np.asarray(fx_value, dtype=float)
                                    if not np.all(np.isfinite(fx_value)):
                                        fx_value = np.full_like(fx_value, spot0)
                                    if inverted:
                                        fx_value = 1.0 / np.maximum(fx_value, 1.0e-12)
                                    coupon_notionals[j, :] = foreign_amount * fx_value
                                elif fx_forward_nodes:
                                    fix_t = float(self._irs_utils._time_from_dates(asof_dt, fix_date, "A365F"))
                                    nodes = sorted((float(tn), float(points)) for tn, points in fx_forward_nodes if np.isfinite(float(tn)) and np.isfinite(float(points)))
                                    node_t = np.asarray([tn for tn, _ in nodes], dtype=float)
                                    node_p = np.asarray([points for _, points in nodes], dtype=float)
                                    points = float(np.interp(max(fix_t, 0.0), node_t, node_p))
                                    fx_value = spot0 + points / max(fx_points_factor, 1.0e-12)
                                    if inverted:
                                        fx_value = 1.0 / max(float(fx_value), 1.0e-12)
                                    coupon_notionals[j, :] = foreign_amount * float(fx_value)
                                else:
                                    if p_dom_fx is not None and p_for_fx is not None:
                                        fix_t = float(self._irs_utils._time_from_dates(asof_dt, fix_date, "A365F"))
                                        df_dom = max(float(p_dom_fx(fix_t)), 1.0e-12)
                                        df_for = max(float(p_for_fx(fix_t)), 1.0e-12)
                                        fx_value = spot0 * df_for / df_dom
                                    else:
                                        fx_value = spot0
                                    if inverted:
                                        fx_value = 1.0 / max(float(fx_value), 1.0e-12)
                                    coupon_notionals[j, :] = foreign_amount * float(fx_value)
                        if pay.size:
                            principal_times: list[float] = []
                            principal_amounts: list[np.ndarray] = []
                            include_initial_period = True
                            if snapshot is not None and start.size:
                                include_initial_period = self._irs_utils._parse_yyyymmdd(
                                    self._date_from_time_cached(snapshot, float(start[0]))
                                ) >= asof_dt
                            for j in range(pay.size):
                                amount_j = np.asarray(coupon_notionals[j], dtype=float)
                                if j == 0:
                                    if include_initial_period and notional_initial_exchange:
                                        principal_times.append(float(start[0]))
                                        principal_amounts.append(-sign * amount_j)
                                    if pay.size > 1:
                                        amount_next = np.asarray(coupon_notionals[j + 1], dtype=float)
                                        principal_times.append(float(end[0]))
                                        principal_amounts.append(sign * amount_j)
                                        principal_times.append(float(end[0]))
                                        principal_amounts.append(-sign * amount_next)
                                    elif notional_final_exchange:
                                        principal_times.append(float(end[0]))
                                        principal_amounts.append(sign * amount_j)
                                else:
                                    if j < pay.size - 1:
                                        amount_next = np.asarray(coupon_notionals[j + 1], dtype=float)
                                        principal_times.append(float(end[j]))
                                        principal_amounts.append(sign * amount_j)
                                        principal_times.append(float(end[j]))
                                        principal_amounts.append(-sign * amount_next)
                                    elif notional_final_exchange:
                                        principal_times.append(float(end[j]))
                                        principal_amounts.append(sign * amount_j)
                            if principal_times:
                                principal_pay = np.asarray(principal_times, dtype=float)
                                principal_amount = np.asarray(principal_amounts, dtype=float)
                        notionals = coupon_notionals
                elif pay.size and notionals.size:
                    principal_times = []
                    principal_amounts = []
                    include_initial_period = True
                    if snapshot is not None and start.size:
                        include_initial_period = self._irs_utils._parse_yyyymmdd(
                            self._date_from_time_cached(snapshot, float(start[0]))
                        ) >= asof_dt
                    if include_initial_period and notional_initial_exchange:
                        principal_times.append(float(start[0]))
                        principal_amounts.append(-sign * float(notionals[0]))
                    if notional_amort_exchange and notionals.size > 1:
                        for j in range(notionals.size - 1):
                            principal_times.append(float(end[j]))
                            principal_amounts.append(sign * float(notionals[j]))
                            principal_times.append(float(end[j]))
                            principal_amounts.append(-sign * float(notionals[j + 1]))
                    if notional_final_exchange:
                        principal_times.append(float(end[min(end.size - 1, notionals.size - 1)]))
                        principal_amounts.append(sign * float(notionals[-1]))
                    if principal_times:
                        principal_pay = np.asarray(principal_times, dtype=float)
                        principal_amount = np.asarray(principal_amounts, dtype=float)
                if float(t) > 1.0e-12 and leg_idx in frozen_coupon_paths:
                    coupons = frozen_coupon_paths[leg_idx][live, :]
                else:
                    coupons = self._rate_leg_coupon_paths(model, leg, leg_ccy, inputs, float(t), leg_x_t, snapshot=snapshot)[live, :]
                if not np.all(np.isfinite(coupons)):
                    quoted = np.asarray(leg.get("quoted_coupon", np.zeros(pay.shape)), dtype=float)[live]
                    coupons = np.where(np.isfinite(coupons), coupons, quoted[:, None])
                if notionals.ndim == 1:
                    amount = sign * notionals[live, None] * accr[live, None] * coupons
                else:
                    amount = sign * notionals[live, :] * accr[live, None] * coupons
                leg_pv = np.sum(amount * disc, axis=0)
                pv += self._convert_amount_to_reporting_ccy(
                    leg_pv,
                    local_ccy=leg_ccy,
                    report_ccy=report_ccy,
                    inputs=inputs,
                    shared_fx_sim=shared_fx_sim,
                    time_index=i,
                )
                if principal_pay.size and principal_amount.size:
                    live_principal = (principal_pay >= 0.0) & (principal_pay > float(t) + 1.0e-12)
                    if np.any(live_principal):
                        disc_principal = model.discount_bond_paths(
                            float(t),
                            principal_pay[live_principal],
                            leg_x_t,
                            p_t,
                            np.asarray(self._irs_utils.curve_values(p_disc, principal_pay[live_principal]), dtype=float),
                        )
                        principal_vals = principal_amount[live_principal]
                        if principal_vals.ndim == 1:
                            leg_pv = np.sum(principal_vals[:, None] * disc_principal, axis=0)
                        else:
                            leg_pv = np.sum(principal_vals * disc_principal, axis=0)
                        pv += self._convert_amount_to_reporting_ccy(
                            leg_pv,
                            local_ccy=leg_ccy,
                            report_ccy=report_ccy,
                            inputs=inputs,
                            shared_fx_sim=shared_fx_sim,
                            time_index=i,
                        )
            vals[i, :] = pv
        return vals

    def _convert_amount_to_reporting_ccy(
        self,
        amount: np.ndarray,
        *,
        local_ccy: str,
        report_ccy: str,
        inputs: _PythonLgmInputs,
        shared_fx_sim: _SharedFxSimulation | None,
        time_index: int,
    ) -> np.ndarray:
        local = str(local_ccy).upper()
        report = str(report_ccy).upper()
        values = np.asarray(amount, dtype=float)
        if local == report:
            return values
        fallback_spot = _today_spot_from_quotes(local + report, inputs, default=0.0)
        if shared_fx_sim is not None:
            direct = f"{local}/{report}"
            inverse = f"{report}/{local}"
            if direct in shared_fx_sim.sim.get("s", {}):
                spot_path = np.asarray(shared_fx_sim.sim["s"][direct][time_index, :], dtype=float)
                if np.all(np.isfinite(spot_path)):
                    return values * spot_path
                if fallback_spot > 0.0:
                    return values * float(fallback_spot)
                return values
            if inverse in shared_fx_sim.sim.get("s", {}):
                spot_path = np.asarray(shared_fx_sim.sim["s"][inverse][time_index, :], dtype=float)
                if np.all(np.isfinite(spot_path)):
                    return values / np.maximum(spot_path, 1.0e-12)
                if fallback_spot > 0.0:
                    return values * float(fallback_spot)
                return values
        if fallback_spot > 0.0:
            return values * fallback_spot
        inverse = _today_spot_from_quotes(report + local, inputs, default=0.0)
        if inverse > 0.0:
            return values / max(inverse, 1.0e-12)
        return values

    def _convert_path_grid_to_reporting_ccy(
        self,
        values: np.ndarray,
        *,
        local_ccy: str,
        report_ccy: str,
        inputs: _PythonLgmInputs,
        shared_fx_sim: _SharedFxSimulation | None,
    ) -> np.ndarray:
        local = str(local_ccy).upper()
        report = str(report_ccy).upper()
        out = np.asarray(values, dtype=float)
        if local == report or out.size == 0:
            return out
        converted = np.empty_like(out)
        for i in range(out.shape[0]):
            converted[i, :] = self._convert_amount_to_reporting_ccy(
                out[i, :],
                local_ccy=local,
                report_ccy=report,
                inputs=inputs,
                shared_fx_sim=shared_fx_sim,
                time_index=i,
            )
        return converted

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
        pv_total_precomputed: float | None = None,
        npv_cube_payload_precomputed: Dict[str, Dict[str, object]] | None = None,
        exposure_profiles_by_trade_precomputed: Dict[str, Dict[str, object]] | None = None,
        ns_valuation_paths_precomputed: Dict[str, np.ndarray] | None = None,
        ns_closeout_paths_precomputed: Dict[str, np.ndarray] | None = None,
        xva_deflated_by_ns_precomputed: Dict[str, bool] | None = None,
    ) -> XVAResult:
        times = inputs.times
        valuation_times = inputs.valuation_times
        obs_times = inputs.observation_times
        obs_closeout_times = inputs.observation_closeout_times
        valuation_idx, obs_idx, obs_closeout_idx, obs_dates = self._validated_grid_indices(inputs)
        df_base = inputs.discount_curves[snapshot.config.base_currency.upper()]
        df_vec = np.asarray([df_base(float(t)) for t in times], dtype=float)
        obs_df_vec = df_vec[obs_idx]
        ns_valuation_paths: Dict[str, np.ndarray] = dict(ns_valuation_paths_precomputed or {})
        ns_closeout_paths: Dict[str, np.ndarray] = dict(ns_closeout_paths_precomputed or {})
        xva_deflated_by_ns: Dict[str, bool] = dict(xva_deflated_by_ns_precomputed or {})
        pv_total = float(pv_total_precomputed or 0.0)
        npv_cube_payload: Dict[str, Dict[str, object]] = dict(npv_cube_payload_precomputed or {})
        exposure_cube_payload: Dict[str, Dict[str, object]] = {}
        exposure_profiles_by_trade: Dict[str, Dict[str, object]] = dict(exposure_profiles_by_trade_precomputed or {})
        exposure_profiles_by_netting_set: Dict[str, Dict[str, object]] = {}
        pfe_quantile = _pfe_quantile(snapshot)
        use_flow_amounts_t0 = str(snapshot.config.params.get("python.use_ore_flow_amounts_t0", "N")).strip().upper() in {"Y", "YES", "TRUE", "1"}

        if not ns_valuation_paths_precomputed or not ns_closeout_paths_precomputed:
            for spec in inputs.trade_specs:
                v = npv_by_trade.get(spec.trade.trade_id)
                if v is None:
                    continue
                if use_flow_amounts_t0 and spec.kind == "IRS" and spec.legs is not None:
                    p_disc = inputs.discount_curves[spec.ccy]
                    anchored = _price_irs_t0_from_flow_amounts(spec.legs, p_disc)
                    if anchored is not None:
                        pv_total += anchored
                    else:
                        pv_total += float(np.mean(v[valuation_idx[0], :]))
                else:
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
            xva_total=xva_total_from_metrics(xva_by_metric),
            xva_by_metric=xva_by_metric,
            exposure_by_netting_set=exposure_by_ns,
            exposure_profiles_by_netting_set=exposure_profiles_by_netting_set,
            exposure_profiles_by_trade=exposure_profiles_by_trade,
            reports=reports,
            cubes=cubes,
            metadata=metadata,
        )


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
    txt = str(value or "").strip().upper()
    return _parse_tenor_to_years_cached(txt)


@lru_cache(maxsize=16384)
def _parse_tenor_to_years_cached(txt: str) -> float:
    m = _TENOR_RE.match(txt)
    if m is None:
        parts = _PERIOD_PART_RE.findall(txt)
        if not parts or "".join(a + b for a, b in parts).upper() != txt.upper():
            raise ValueError(f"unsupported tenor '{txt}'")
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
    arr = np.asarray(sorted(set(float(x) for x in vals if x >= 0.0)), dtype=float)
    return arr


def _parse_ore_exposure_date_grid_from_simulation_xml_text(
    xml_text: str,
    asof: str,
    irs_utils: Any,
) -> tuple[np.ndarray, Tuple[str, ...]] | None:
    root = ET.fromstring(xml_text)
    grid_txt = root.findtext("./Parameters/Grid")
    if not grid_txt:
        return None
    grid_parts = [x.strip() for x in grid_txt.split(",") if x.strip()]
    if len(grid_parts) != 2:
        return None
    try:
        n = int(float(grid_parts[0]))
        amount, unit = irs_utils._parse_tenor_value_unit(grid_parts[1])
    except Exception:
        return None
    if n <= 0 or amount <= 0:
        return None

    calendar = (root.findtext("./Parameters/Calendar") or "TARGET").strip() or "TARGET"
    asof_dt = irs_utils._parse_yyyymmdd(_normalize_asof_date(asof))
    dates: list[date] = [asof_dt]
    for i in range(1, n + 1):
        unadjusted = irs_utils._shift_date_by_tenor(asof_dt, amount * i, unit)
        dates.append(irs_utils._adjust_date(unadjusted, "F", calendar))

    unique_dates: list[date] = []
    seen: set[date] = set()
    for d in dates:
        if d not in seen:
            unique_dates.append(d)
            seen.add(d)
    times = np.asarray(
        [float(irs_utils._time_from_dates(asof_dt, d, "ActualActual(ISDA)")) for d in unique_dates],
        dtype=float,
    )
    return times, tuple(d.isoformat() for d in unique_dates)


def _parse_stochastic_fx_pairs_from_simulation_xml_text(
    xml_text: str,
    *,
    model_ccy: str,
    trade_specs: Sequence[_TradeSpec],
) -> Tuple[str, ...]:
    def trade_fx_pairs() -> set[str]:
        pairs: set[str] = {
            f"{spec.trade.product.pair[:3].upper()}/{spec.trade.product.pair[3:].upper()}"
            for spec in trade_specs
            if spec.kind == "FXForward" and isinstance(spec.trade.product, FXForward)
        }
        for spec in trade_specs:
            if spec.kind != "RateSwap":
                continue
            ccys = _rate_leg_currencies(spec.legs, spec.ccy)
            if len(ccys) == 2:
                pairs.add(f"{ccys[0]}/{ccys[1]}")
        return pairs

    if not xml_text.strip():
        return tuple(sorted(trade_fx_pairs()))
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
        out = []
        for pair in trade_fx_pairs():
            base, quote = pair.split("/")
            resolved = _resolve_fx_pair_name(base, quote, normalized_market_pairs)
            if resolved in normalized_market_pairs:
                out.append(resolved)
        return tuple(sorted(set(out)))
    domestic = str(
        root.findtext("./CrossAssetModel/DomesticCcy")
        or model_ccy
    ).strip().upper()
    foreign_nodes = root.findall("./CrossAssetModel/ForeignExchangeModels/CrossCcyLGM")
    if not foreign_nodes:
        return tuple(sorted(trade_fx_pairs()))
    supported_foreigns = {
        str(node.attrib.get("foreignCcy") or "").strip().upper()
        for node in foreign_nodes
        if str(node.attrib.get("foreignCcy") or "").strip()
    }
    has_default = "DEFAULT" in supported_foreigns
    explicit_foreigns = {ccy for ccy in supported_foreigns if ccy != "DEFAULT"}
    out = []
    for pair in trade_fx_pairs():
        base, quote = pair.split("/")
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


def _parse_zero_quote_time(token: str, asof_date: str | None, day_counter: str = "A365F") -> float | None:
    return _parse_zero_quote_time_cached(str(token).strip(), _normalize_asof_date(asof_date or ""), day_counter)


@lru_cache(maxsize=16384)
def _parse_zero_quote_time_cached(token: str, asof_date: str, day_counter: str) -> float | None:
    if not asof_date:
        return None
    try:
        from pythonore.io import ore_snapshot as ore_snapshot_io

        anchor = datetime.fromisoformat(asof_date).date()
        if re.fullmatch(r"\d{8}", token):
            pillar = datetime.strptime(token, "%Y%m%d").date()
        else:
            pillar = datetime.fromisoformat(token).date()
    except Exception:
        return None
    yf = ore_snapshot_io._year_fraction_from_day_counter(anchor, pillar, day_counter)
    if yf <= 0.0:
        return None
    return float(yf)


def _scan_market_quotes(raw_quotes: Sequence[Any]) -> Tuple[Dict[str, object], ...]:
    key = tuple((str(q.key), float(q.value)) for q in raw_quotes)
    cached = getattr(_scan_market_quotes, "_cache", None)
    if cached is None:
        cached = {}
        setattr(_scan_market_quotes, "_cache", cached)
    result = cached.get(key)
    if result is not None:
        return result
    result = tuple({"key": str(q.key), "value": float(q.value)} for q in raw_quotes)
    cached[key] = result
    return result


def _parse_market_overlay(raw_quotes: Sequence[Any], asof_date: str | None = None) -> Dict[str, Any]:
    cache_key = (_normalize_asof_date(asof_date or ""), tuple((str(q.key).strip().upper(), float(q.value)) for q in raw_quotes))
    cached = getattr(_parse_market_overlay, "_cache", None)
    if cached is None:
        cached = {}
        setattr(_parse_market_overlay, "_cache", cached)
    overlay = cached.get(cache_key)
    if overlay is not None:
        return overlay
    zero: Dict[str, List[Tuple[float, float]]] = {}
    named_zero: Dict[str, List[Tuple[float, float]]] = {}
    fwd: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    fwd_by_index: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    bma_ratio: Dict[str, List[Tuple[float, float]]] = {}
    fx: Dict[str, float] = {}
    fx_forward: Dict[str, List[Tuple[float, float]]] = {}
    fx_vol: Dict[str, List[Tuple[float, float]]] = {}
    swaption_normal_vols: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    cms_correlations: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    hazard: Dict[str, List[Tuple[float, float]]] = {}
    recovery: Dict[str, float] = {}
    cds_spreads: Dict[str, List[Tuple[float, float]]] = {}
    parse_tenor = _parse_tenor_to_years
    parse_zero_time = _parse_zero_quote_time
    for q in raw_quotes:
        key = str(q.key).strip()
        up = key.upper()
        val = float(q.value)
        parts = up.split("/")
        p0 = parts[0] if len(parts) > 0 else ""
        p1 = parts[1] if len(parts) > 1 else ""
        if len(parts) >= 4 and p0 == "FX" and p1 == "RATE":
            fx[parts[2] + parts[3]] = val
            continue
        if len(parts) >= 5 and p0 == "FXFWD" and p1 == "RATE":
            pair = parts[2] + parts[3]
            tenor = parts[4]
            try:
                if tenor in {"ON", "TN", "SN"}:
                    t = {"ON": 1.0, "TN": 2.0, "SN": 2.0}[tenor] / 365.0
                else:
                    t = parse_tenor(tenor)
            except Exception:
                continue
            fx_forward.setdefault(pair, []).append((t, val))
            continue
        if len(parts) >= 3 and p0 == "FX":
            fx[parts[1] + parts[2]] = val
            continue
        if len(parts) >= 6 and p0 == "FX_OPTION" and p1 == "RATE_LNVOL":
            ccy1 = parts[2]
            ccy2 = parts[3]
            tenor = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            fx_vol.setdefault(ccy1 + ccy2, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "SWAPTION" and p1 == "RATE_NVOL":
            ccy = parts[2]
            expiry = parts[3]
            swap_tenor = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = parse_tenor(expiry)
            except Exception:
                continue
            swaption_normal_vols.setdefault((ccy, swap_tenor), []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "CORRELATION" and p1 == "RATE":
            idx1 = parts[2]
            idx2 = parts[3]
            expiry = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = parse_tenor(expiry)
            except Exception:
                continue
            cms_correlations.setdefault(tuple(sorted((idx1, idx2))), []).append((t, val))
            continue
        if len(parts) >= 4 and p0 == "ZERO" and p1 == "RATE":
            ccy = parts[2]
            tenor = parts[3]
            curve_name = None
            day_counter = "A365F"
            if len(parts) >= 6:
                curve_name = parts[3]
                if len(parts) >= 7:
                    day_counter = parts[-2]
                tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                t = parse_zero_time(tenor, asof_date, day_counter=day_counter)
                if t is None:
                    continue
            if curve_name is not None:
                named_zero.setdefault(curve_name, []).append((t, val))
            else:
                zero.setdefault(ccy, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "IR_SWAP" and p1 == "RATE":
            ccy = parts[2]
            idx_name = parts[4].upper()
            if _is_bma_sifma_index(idx_name) or (len(parts) > 3 and _is_bma_sifma_index(parts[3])):
                continue
            idx_tenor = idx_name.split("-")[-1].upper()
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            if 0.0 < t <= 80.0:
                if idx_tenor in ("1D", "ON", "O/N"):
                    zero.setdefault(ccy, []).append((t, val))
                else:
                    fwd_by_index.setdefault(ccy, {}).setdefault(idx_name, []).append((t, val))
                    fwd.setdefault(ccy, {}).setdefault(idx_tenor, []).append((t, val))
            continue
        if len(parts) >= 5 and p0 == "BMA_SWAP" and p1 == "RATIO":
            ccy = parts[2]
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            if 0.0 < t <= 80.0:
                bma_ratio.setdefault(ccy, []).append((t, val))
            continue
        if len(parts) >= 5 and p0 == "MM" and p1 == "RATE":
            ccy = parts[2]
            idx_name = parts[4].upper()
            if _is_bma_sifma_index(idx_name) or (len(parts) > 3 and _is_bma_sifma_index(parts[3])):
                continue
            idx_tenor = idx_name.split("-")[-1].upper()
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            if 0.0 < t <= 10.0:
                if idx_tenor in ("1D", "ON", "O/N"):
                    zero.setdefault(ccy, []).append((t, val))
                else:
                    fwd_by_index.setdefault(ccy, {}).setdefault(idx_name, []).append((t, val))
                    fwd.setdefault(ccy, {}).setdefault(idx_tenor, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "HAZARD_RATE":
            cpty = parts[2]
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            hazard.setdefault(cpty, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "CDS" and p1 == "CREDIT_SPREAD":
            cpty = parts[2]
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            cds_spreads.setdefault(cpty, []).append((t, val))
            continue
        if len(parts) >= 5 and p0 == "RECOVERY_RATE":
            cpty = parts[2]
            recovery[cpty] = val
            continue
    for cpty, spreads in cds_spreads.items():
        if cpty in hazard:
            continue
        rec = float(recovery.get(cpty, 0.4))
        lgd = max(1.0 - rec, 1.0e-6)
        hazard[cpty] = [(t, max(float(spread) / lgd, 0.0)) for t, spread in spreads]
    overlay = {
        "zero": zero,
        "named_zero": named_zero,
        "fwd": fwd,
        "fwd_by_index": fwd_by_index,
        "bma_ratio": bma_ratio,
        "fx": fx,
        "fx_forward": fx_forward,
        "fx_vol": fx_vol,
        "swaption_normal_vols": swaption_normal_vols,
        "cms_correlations": cms_correlations,
        "hazard": hazard,
        "cds_spreads": cds_spreads,
        "recovery": recovery,
    }
    cached[cache_key] = overlay
    return overlay


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


def _price_irs_t0_from_flow_amounts(
    legs: Mapping[str, np.ndarray],
    p_disc: Callable[[float], float],
) -> float | None:
    fixed_pay = np.asarray(legs.get("fixed_pay_time", []), dtype=float)
    fixed_amount = np.asarray(legs.get("fixed_amount", []), dtype=float)
    float_pay = np.asarray(legs.get("float_pay_time", []), dtype=float)
    float_amount = np.asarray(legs.get("float_amount", []), dtype=float)
    if fixed_pay.size != fixed_amount.size or float_pay.size != float_amount.size:
        return None
    if fixed_pay.size == 0 and float_pay.size == 0:
        return None
    pv = 0.0
    if fixed_pay.size:
        pv += float(np.sum(fixed_amount * np.fromiter((float(p_disc(float(t))) for t in fixed_pay), dtype=float, count=fixed_pay.size)))
    if float_pay.size:
        pv += float(np.sum(float_amount * np.fromiter((float(p_disc(float(t))) for t in float_pay), dtype=float, count=float_pay.size)))
    return pv


def _load_ore_t0_npv_for_reporting_ccy(
    snapshot: XVASnapshot,
    *,
    trade_id: str,
    report_ccy: str,
) -> float | None:
    ore_path_txt = getattr(snapshot.config.source_meta, "path", "") or ""
    if not ore_path_txt:
        return None
    ore_path = Path(ore_path_txt).resolve()
    if ore_path.suffix.lower() != ".xml" or not ore_path.exists():
        return None
    npv_csv = (ore_path.parent.parent / snapshot.config.params.get("outputPath", "Output") / "npv.csv").resolve()
    if not npv_csv.exists():
        return None
    report = str(report_ccy).strip().upper()
    try:
        with npv_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            trade_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
            for row in reader:
                if (row.get(trade_key) or "").strip() != trade_id:
                    continue
                npv_ccy = (row.get("NpvCurrency") or row.get("NPVCurrency") or "").strip().upper()
                base_ccy = (row.get("BaseCurrency") or "").strip().upper()
                if report and base_ccy == report:
                    value = (row.get("NPV(Base)") or row.get("NPV Base") or "").strip()
                    if value:
                        return float(value)
                if report and npv_ccy == report:
                    value = (row.get("NPV") or row.get("npv") or "").strip()
                    if value:
                        return float(value)
                value = (row.get("NPV(Base)") or row.get("NPV Base") or row.get("NPV") or row.get("npv") or "").strip()
                return float(value) if value else None
    except Exception:
        return None
    return None


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
    overnight_indexed = _forward_index_family(float_index) == "1D"
    lookback_days = 0
    rate_cutoff = 0
    naked_option = False
    local_cap_floor = False
    cap = None
    floor = None
    apply_observation_shift = False
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
        "float_overnight_indexed": overnight_indexed,
        "float_lookback_days": lookback_days,
        "float_rate_cutoff": rate_cutoff,
        "float_naked_option": naked_option,
        "float_local_cap_floor": local_cap_floor,
        "float_cap": cap,
        "float_floor": floor,
        "float_apply_observation_shift": apply_observation_shift,
        "float_gearing": np.ones(float_pay.shape),
        "float_is_in_arrears": True,
        "float_fixing_days": int(p.fixing_days),
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


def _today_spot_from_quotes(pair6: str, inputs: _PythonLgmInputs, default: float = 1.0) -> float:
    base = pair6[:3].upper()
    quote = pair6[3:].upper()
    fwd = base + quote
    inv = quote + base
    spots = inputs.fx_spots_today or inputs.fx_spots
    if fwd in spots:
        return float(spots[fwd])
    if inv in spots:
        return 1.0 / max(float(spots[inv]), 1.0e-12)
    return _spot_from_quotes(pair6, inputs, default=default)


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
    from pythonore.compute import irs_xva_utils as irs_utils

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
    scalar_cache: Dict[float, float] = {}

    def shocked_curve(t: float) -> float:
        tt = max(float(t), 0.0)
        cached = scalar_cache.get(tt)
        if cached is not None:
            return cached
        if tt <= 1.0e-12:
            return 1.0

        # ORE curve configs in the benchmark cases use Discount + LogLinear.
        # Apply the node shocks at the bucket tenors, then interpolate the
        # shocked discount factors in log-discount space between nodes.
        if tt <= times[0]:
            base_df = float(base_curve(tt))
            out = float(base_df * np.exp(-shifts[0] * tt))
            scalar_cache[tt] = out
            return out
        if tt >= times[-1]:
            base_df = float(base_curve(tt))
            out = float(base_df * np.exp(-shifts[-1] * tt))
            scalar_cache[tt] = out
            return out

        shocked_log_df = float(irs_utils.interpolate_linear_flat(tt, times, shocked_node_logs))
        out = float(np.exp(shocked_log_df))
        scalar_cache[tt] = out
        return out

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
    if parts[0] == "ZERO" and parts[1] == "RATE":
        if len(parts) == 4:
            return True
        if len(parts) >= 6:
            return parts[3] == family or family in {"", "1D", "ON", "O/N"}
        return False
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


def _quote_matches_forward_curve(key: str, ccy: str, tenor: str, index_name: str = "") -> bool:
    parts = str(key).strip().upper().split("/")
    if len(parts) < 3 or parts[2] != ccy.upper():
        return False
    family = _normalize_forward_tenor_family(tenor)
    exact_index = _normalize_curve_lookup_key(index_name)
    if _is_bma_sifma_index(exact_index):
        return False
    if parts[0] == "ZERO" and parts[1] == "RATE":
        if exact_index:
            return len(parts) >= 6 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        if family == "1D":
            return True
        if len(parts) == 4:
            return False
        if len(parts) >= 6:
            return _normalize_forward_tenor_family(parts[3]) == family
        return False
    if parts[0] == "MM" and parts[1] == "RATE":
        if exact_index:
            return len(parts) >= 6 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        return True
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        if exact_index:
            return len(parts) > 5 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        return len(parts) > 5 and _normalize_forward_tenor_family(parts[4]) == family
    if parts[0] == "FRA" and parts[1] == "RATE":
        if exact_index:
            return len(parts) > 4 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        return len(parts) > 4 and _normalize_forward_tenor_family(parts[-1]) == family
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


def classify_portfolio_support(
    snapshot: XVASnapshot,
    *,
    fallback_to_swig: bool = False,
) -> Dict[str, Any]:
    """Public preflight helper for native-vs-SWIG support classification."""
    return PythonLgmAdapter(fallback_to_swig=fallback_to_swig).classify_portfolio_support(snapshot)
