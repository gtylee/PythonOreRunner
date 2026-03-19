from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .models import (
    EngineState,
    EuropeanOption,
    FXForward,
    IRS,
    MarketDataSnapshot,
    PortfolioSnapshot,
    Trade,
    XVAConfig,
    XVAResult,
)


class XVAEngine:
    """
    Demonstration engine for a Python-first XVA interface.

    This is intentionally lightweight: numbers are toy approximations to
    demonstrate API shape, incremental rebuild behavior, and object lifecycle.
    """

    def build_state(
        self,
        market: MarketDataSnapshot,
        portfolio: PortfolioSnapshot,
        config: XVAConfig,
    ) -> EngineState:
        compiled_market = self._compile_market(market)
        trade_pv_cache, trade_epe_cache = self._build_trade_caches(portfolio, compiled_market)
        return EngineState(
            market=market,
            portfolio=portfolio,
            config=config,
            market_key=self._market_key(market),
            portfolio_key=self._portfolio_key(portfolio),
            compiled_market=compiled_market,
            trade_pv_cache=trade_pv_cache,
            trade_epe_cache=trade_epe_cache,
        )

    def run_stateless(
        self,
        market: MarketDataSnapshot,
        portfolio: PortfolioSnapshot,
        config: XVAConfig,
    ) -> XVAResult:
        state = self.build_state(market, portfolio, config)
        return self.xva(state, rebuilt_components=["market", "portfolio"])

    def xva(self, state: EngineState, rebuilt_components: Sequence[str] | None = None) -> XVAResult:
        rebuilt = list(rebuilt_components or [])
        pv_total = sum(state.trade_pv_cache.values())

        epe_by_netting_set: Dict[str, float] = {}
        for t in state.portfolio.trades:
            epe_by_netting_set[t.netting_set] = epe_by_netting_set.get(t.netting_set, 0.0) + state.trade_epe_cache[t.trade_id]

        xva_by_metric: Dict[str, float] = {}
        total_epe = sum(epe_by_netting_set.values())
        cpty_weight = self._avg_credit_weight(state.market)

        for metric in state.config.include:
            if metric == "CVA":
                xva_by_metric[metric] = 0.012 * total_epe * cpty_weight
            elif metric == "DVA":
                xva_by_metric[metric] = -0.006 * total_epe
            elif metric == "FVA":
                xva_by_metric[metric] = 0.002 * abs(pv_total)
            elif metric == "MVA":
                xva_by_metric[metric] = 0.0015 * total_epe

        xva_total = sum(xva_by_metric.values())
        return XVAResult(
            pv_total=pv_total,
            xva_total=xva_total,
            xva_by_metric=xva_by_metric,
            epe_by_netting_set=epe_by_netting_set,
            rebuilt_components=rebuilt,
        )

    def apply_market_patch(
        self,
        state: EngineState,
        patch: Mapping[str, float],
    ) -> EngineState:
        """
        Patch format example: {"USD_OIS.2Y": 0.0415}
        """
        curves = {k: dict(v) for k, v in state.market.curves.items()}
        for key, value in patch.items():
            curve_id, tenor = key.split(".", 1)
            if curve_id not in curves:
                raise KeyError(f"Unknown curve in patch: {curve_id}")
            curves[curve_id][tenor] = float(value)

        new_market = replace(state.market, curves=curves)
        new_market_key = self._market_key(new_market)
        if new_market_key == state.market_key:
            return state

        compiled_market = self._compile_market(new_market)
        trade_pv_cache, trade_epe_cache = self._build_trade_caches(state.portfolio, compiled_market)
        return replace(
            state,
            market=new_market,
            market_key=new_market_key,
            compiled_market=compiled_market,
            trade_pv_cache=trade_pv_cache,
            trade_epe_cache=trade_epe_cache,
        )

    def apply_portfolio_patch(
        self,
        state: EngineState,
        add: Iterable[Trade] = (),
        amend: Iterable[Tuple[str, Dict[str, object]]] = (),
        remove: Iterable[str] = (),
    ) -> EngineState:
        old_trade_map = {t.trade_id: t for t in state.portfolio.trades}
        trade_map = {t.trade_id: t for t in state.portfolio.trades}

        for tid in remove:
            trade_map.pop(tid, None)

        for tid, updates in amend:
            if tid not in trade_map:
                raise KeyError(f"Cannot amend unknown trade {tid}")
            existing = trade_map[tid]
            updates_copy = dict(updates)
            product = existing.product
            product_updates = updates_copy.pop("product", None)
            if product_updates:
                product = replace(product, **product_updates)
            trade_map[tid] = replace(existing, product=product, **updates_copy)

        for t in add:
            trade_map[t.trade_id] = t

        new_portfolio = PortfolioSnapshot(trades=list(trade_map.values()))
        new_portfolio_key = self._portfolio_key(new_portfolio)
        if new_portfolio_key == state.portfolio_key:
            return state

        new_trade_pv_cache = dict(state.trade_pv_cache)
        new_trade_epe_cache = dict(state.trade_epe_cache)

        changed_trade_ids = set(new_trade_pv_cache) ^ set(trade_map)
        for tid in set(new_trade_pv_cache).intersection(trade_map):
            if asdict(old_trade_map[tid]) != asdict(trade_map[tid]):
                changed_trade_ids.add(tid)

        for tid in changed_trade_ids:
            if tid in trade_map:
                pv, epe = self._price_trade(trade_map[tid], state.compiled_market)
                new_trade_pv_cache[tid] = pv
                new_trade_epe_cache[tid] = epe
            else:
                new_trade_pv_cache.pop(tid, None)
                new_trade_epe_cache.pop(tid, None)

        return replace(
            state,
            portfolio=new_portfolio,
            portfolio_key=new_portfolio_key,
            trade_pv_cache=new_trade_pv_cache,
            trade_epe_cache=new_trade_epe_cache,
        )

    def _compile_market(self, market: MarketDataSnapshot) -> Mapping[str, object]:
        usd_ois_2y = market.curves.get("USD_OIS", {}).get("2Y", 0.03)
        fx = dict(market.fx)
        return {
            "discount_rate": usd_ois_2y,
            "fx": fx,
            "credit": market.credit_spreads_bps,
        }

    def _build_trade_caches(
        self,
        portfolio: PortfolioSnapshot,
        compiled_market: Mapping[str, object],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        pv_cache: Dict[str, float] = {}
        epe_cache: Dict[str, float] = {}
        for t in portfolio.trades:
            pv, epe = self._price_trade(t, compiled_market)
            pv_cache[t.trade_id] = pv
            epe_cache[t.trade_id] = epe
        return pv_cache, epe_cache

    def _price_trade(self, trade: Trade, compiled_market: Mapping[str, object]) -> Tuple[float, float]:
        r = float(compiled_market["discount_rate"])
        fx = compiled_market["fx"]
        p = trade.product

        if isinstance(p, IRS):
            fair = r
            direction = -1.0 if p.pay_fixed else 1.0
            pv = direction * p.notional * (fair - p.fixed_rate) * p.maturity_years
            epe = max(pv, 0.0) * 0.35 + abs(p.notional) * 0.001
            return pv, epe

        if isinstance(p, FXForward):
            spot = float(fx[p.pair])
            direction = 1.0 if p.buy_base else -1.0
            pv = direction * p.notional * (spot - p.strike)
            epe = max(pv, 0.0) * 0.50 + abs(p.notional) * 0.0008
            return pv, epe

        if isinstance(p, EuropeanOption):
            spot = float(fx[p.underlying])
            intrinsic = max(spot - p.strike, 0.0) if p.kind == "call" else max(p.strike - spot, 0.0)
            time_value = 0.01 * p.maturity_years
            pv = p.notional * (intrinsic + time_value)
            epe = max(pv, 0.0) * 0.65 + abs(p.notional) * 0.0012
            return pv, epe

        raise TypeError(f"Unsupported product type for trade {trade.trade_id}: {type(p)}")

    def _avg_credit_weight(self, market: MarketDataSnapshot) -> float:
        spreads = [x for cpty in market.credit_spreads_bps.values() for x in cpty.values()]
        if not spreads:
            return 1.0
        avg_spread_bps = sum(spreads) / len(spreads)
        return max(0.7, min(1.8, avg_spread_bps / 100.0))

    def _market_key(self, market: MarketDataSnapshot) -> str:
        return self._stable_hash(asdict(market))

    def _portfolio_key(self, portfolio: PortfolioSnapshot) -> str:
        return self._stable_hash(asdict(portfolio))

    @staticmethod
    def _stable_hash(obj: object) -> str:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
