from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from pythonore.domain.dataclasses import MporConfig, Trade, XVASnapshot


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


@dataclass(frozen=True)
class _TradeSpec:
    trade: Trade
    kind: str
    notional: float
    ccy: str
    legs: Dict[str, np.ndarray] | None = None
    sticky_state: Dict[str, object] | None = None


@dataclass(frozen=True)
class _PythonLgmInputs:
    asof: str
    times: np.ndarray
    valuation_times: np.ndarray
    observation_times: np.ndarray
    observation_closeout_times: np.ndarray
    discount_curves: Dict[str, Callable[[float], float]]
    forward_curves: Dict[str, Callable[[float], float]]
    forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]]
    forward_curves_by_name: Dict[str, Callable[[float], float]]
    swap_index_forward_tenors: Dict[str, str]
    inflation_curves: Dict[Tuple[str, str], Any]
    xva_discount_curve: Callable[[float], float] | None
    funding_borrow_curve: Callable[[float], float] | None
    funding_lend_curve: Callable[[float], float] | None
    survival_curves: Dict[str, Callable[[float], float]]
    hazard_times: Dict[str, np.ndarray]
    hazard_rates: Dict[str, np.ndarray]
    recovery_rates: Dict[str, float]
    lgm_params: Dict[str, object]
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
    input_provenance: Dict[str, str]
    input_fallbacks: Tuple[str, ...] = ()
    fx_forwards: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    fx_spots_today: Dict[str, float] = field(default_factory=dict)
    discount_curve_dates: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    observation_dates: Tuple[str, ...] = ()
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


__all__ = [
    "_CurveBundle",
    "_PricingContext",
    "_PythonLgmInputs",
    "_SharedFxSimulation",
    "_TradeSpec",
]
