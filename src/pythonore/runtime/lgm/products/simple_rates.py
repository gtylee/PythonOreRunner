from __future__ import annotations

from typing import Any

import numpy as np

from pythonore.runtime.lgm.types import _PricingContext, _TradeSpec


def price_trade_cashflow_paths(
    adapter: Any,
    spec: _TradeSpec,
    ctx: _PricingContext,
) -> tuple[np.ndarray | None, bool]:
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
        vals[i, :] = adapter._convert_amount_to_reporting_ccy(
            local_pv,
            local_ccy=spec.ccy,
            report_ccy=report_ccy,
            inputs=ctx.inputs,
            shared_fx_sim=ctx.shared_fx_sim,
            time_index=i,
        )
    return vals, False


def price_trade_fra_paths(
    adapter: Any,
    spec: _TradeSpec,
    ctx: _PricingContext,
) -> tuple[np.ndarray | None, bool]:
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
    p_fwd = adapter._resolve_index_curve(inputs, spec.ccy, index_name)
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


__all__ = ["price_trade_cashflow_paths", "price_trade_fra_paths"]
