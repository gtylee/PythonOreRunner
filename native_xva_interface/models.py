from __future__ import annotations

"""Compatibility-only prototype models.

The maintained snapshot model now lives behind the shared dataclass surface used by
`native_xva_interface.dataclasses` and the canonical `pythonore.domain` package.
This module remains importable for older toy-engine callers, but it is no longer
the source of truth for new work.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Mapping, Optional


Metric = Literal["CVA", "DVA", "FVA", "MVA"]


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Immutable market input snapshot used for reproducible runs."""

    asof: str
    curves: Dict[str, Dict[str, float]]
    fx: Dict[str, float]
    credit_spreads_bps: Dict[str, Dict[str, float]]

    def bumped_curve(self, curve_id: str, tenor: str, shift_bp: float) -> "MarketDataSnapshot":
        curves = {k: dict(v) for k, v in self.curves.items()}
        if curve_id not in curves:
            raise KeyError(f"Unknown curve: {curve_id}")
        if tenor not in curves[curve_id]:
            raise KeyError(f"Unknown tenor {tenor} on curve {curve_id}")
        curves[curve_id][tenor] += shift_bp / 10_000.0
        return replace(self, curves=curves)


@dataclass(frozen=True)
class XVAConfig:
    num_paths: int = 5000
    horizon_years: int = 5
    include: List[Metric] = field(default_factory=lambda: ["CVA", "DVA", "FVA"])


@dataclass(frozen=True)
class Product:
    product_type: str = field(init=False)


@dataclass(frozen=True)
class IRS(Product):
    product_type: str = field(init=False, default="IRS")
    ccy: str
    notional: float
    fixed_rate: float
    maturity_years: float
    pay_fixed: bool = True


@dataclass(frozen=True)
class FXForward(Product):
    product_type: str = field(init=False, default="FXForward")
    pair: str
    notional: float
    strike: float
    maturity_years: float
    buy_base: bool = True


@dataclass(frozen=True)
class EuropeanOption(Product):
    product_type: str = field(init=False, default="EuropeanOption")
    underlying: str
    kind: Literal["call", "put"]
    strike: float
    notional: float
    maturity_years: float


@dataclass(frozen=True)
class Trade:
    trade_id: str
    counterparty: str
    netting_set: str
    product: Product


@dataclass(frozen=True)
class PortfolioSnapshot:
    trades: List[Trade]


@dataclass
class XVAResult:
    pv_total: float
    xva_total: float
    xva_by_metric: Dict[str, float]
    epe_by_netting_set: Dict[str, float]
    rebuilt_components: List[str]


@dataclass
class EngineState:
    market: MarketDataSnapshot
    portfolio: PortfolioSnapshot
    config: XVAConfig
    market_key: str
    portfolio_key: str
    compiled_market: Mapping[str, object]
    trade_pv_cache: Dict[str, float]
    trade_epe_cache: Dict[str, float]
