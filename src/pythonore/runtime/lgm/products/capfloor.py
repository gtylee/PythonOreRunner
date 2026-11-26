from __future__ import annotations

from typing import Any

import numpy as np

from pythonore.runtime.lgm.products.capfloor_market import (
    market_capfloor_npv_paths,
    market_capfloor_t0_value,
)
from pythonore.runtime.lgm.types import _PricingContext, _TradeSpec


def price_trade_capfloor_paths(
    adapter: Any,
    spec: _TradeSpec,
    ctx: _PricingContext,
) -> tuple[np.ndarray | None, bool]:
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
        _anchor_market_t0(adapter, spec, ctx, cached_vals)
        return cached_vals, False

    ctx.last_trade_backend = "capfloor-market-vol"
    ctx.last_trade_backend_detail = "market-optionlet-vol-paths"
    try:
        market_vals = market_capfloor_npv_paths(adapter, spec, ctx)
    except Exception:
        market_vals = None
    if market_vals is not None:
        _anchor_market_t0(adapter, spec, ctx, market_vals)
        ctx.capfloor_value_cache[capfloor_cache_key] = market_vals
        return market_vals, False

    p_disc = ctx.inputs.discount_curves[spec.ccy]
    p_fwd = adapter._resolve_index_curve(ctx.inputs, spec.ccy, index_name)
    if ctx.irs_backend is None:
        ctx.last_trade_backend = "capfloor-numpy"
        ctx.last_trade_backend_detail = "capfloor_npv_paths"
        vals = adapter._ir_options_mod.capfloor_npv_paths(
            model=ctx.model,
            p0_disc=p_disc,
            p0_fwd=p_fwd,
            capfloor=definition,
            times=ctx.inputs.times,
            x_paths=ctx.x_paths,
            lock_fixings=True,
            fixings=adapter._fixings_lookup(ctx.snapshot),
            fixing_index=index_name,
        )
        _anchor_market_t0(adapter, spec, ctx, vals)
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
        fixings=adapter._fixings_lookup(ctx.snapshot),
        fixing_index=index_name,
        return_numpy=True,
    )
    _anchor_market_t0(adapter, spec, ctx, vals)
    ctx.capfloor_value_cache[capfloor_cache_key] = vals
    return vals, False


def _anchor_market_t0(adapter: Any, spec: _TradeSpec, ctx: _PricingContext, values: np.ndarray) -> None:
    valuation_times = np.asarray(ctx.inputs.valuation_times, dtype=float)
    times = np.asarray(ctx.inputs.times, dtype=float)
    if valuation_times.size == 0 or times.size == 0 or abs(float(valuation_times[0])) > 1.0e-10:
        return
    t0_idx = int(np.argmin(np.abs(times)))
    if abs(float(times[t0_idx])) > 1.0e-10:
        return
    try:
        market_value = market_capfloor_t0_value(adapter, spec, ctx)
    except Exception:
        market_value = None
    if market_value is None or not np.isfinite(market_value):
        return
    values[t0_idx, :] = float(market_value)
    ctx.last_trade_backend_detail = (
        f"{ctx.last_trade_backend_detail};t0-market-capfloor-anchor"
        if ctx.last_trade_backend_detail
        else "t0-market-capfloor-anchor"
    )


__all__ = ["price_trade_capfloor_paths"]
