from __future__ import annotations

import math
from typing import Any

import numpy as np

from pythonore.domain.dataclasses import FXForward, Trade
from pythonore.runtime.lgm.types import _PythonLgmInputs, _SharedFxSimulation


def price_fx_forward(
    adapter: Any,
    trade: Trade,
    inputs: _PythonLgmInputs,
    n_times: int,
    n_paths: int,
    shared_sim: _SharedFxSimulation | None = None,
) -> np.ndarray:
    """Compute the NPV of an FX forward across all simulation times and paths."""
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
        return price_fx_forward_deterministic(trade, inputs, n_times, n_paths)
    else:
        spot = spot_from_quotes(p.pair, inputs, default=1.0)
        fx_vol = fx_vol_for_trade(inputs, p.pair, float(p.maturity_years), default=0.15)
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
        hybrid = adapter._fx_utils.build_two_ccy_hybrid(pair=pair, ir_specs=ir_specs, fx_vol=fx_vol)
        sim = hybrid.simulate_paths(
            times=inputs.times,
            n_paths=n_paths,
            log_s0={pair: float(np.log(max(spot, 1.0e-12)))},
            rd_minus_rf={pair: fx_carry_for_pair(pair, inputs, horizon=float(p.maturity_years))},
            rng=np.random.default_rng(inputs.seed + 17),
        )
        s_t = sim["s"][pair]
        x_dom_t = sim["x"][dom]
        x_for_t = sim["x"][for_ccy]
    fx_def = adapter._fx_utils.FxForwardDef(
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
        npv_quote = direction * adapter._fx_utils.fx_forward_npv(
            hybrid=hybrid,
            fx_def=fx_def,
            t=float(t),
            s_t=s_t[i, :],
            x_dom_t=x_dom_t[i, :],
            x_for_t=x_for_t[i, :],
            p0_dom=p_dom,
            p0_for=p_for,
        )
        vals[i, :] = convert_fx_forward_npv_to_reporting_ccy(
            npv_quote=npv_quote,
            report_ccy=report_ccy,
            base_ccy=for_ccy,
            quote_ccy=dom,
            spot_path=s_t[i, :],
            inputs=inputs,
        )
    return vals


def price_fx_forward_deterministic(
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
    spot0 = spot_from_quotes(p.pair, inputs, default=1.0)
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
        vals[i, :] = convert_fx_forward_npv_to_reporting_ccy(
            npv_quote=np.full(n_paths, npv_quote, dtype=float),
            report_ccy=report_ccy,
            base_ccy=for_ccy,
            quote_ccy=dom,
            spot_path=np.full(n_paths, max(float(spot0), 1.0e-12), dtype=float),
            inputs=inputs,
        )
    return vals


def fx_forward_closeout_paths(
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
        if float(co_t) <= float(obs_t) + 1.0e-12:
            target = float(obs_t)
            idx = int(np.searchsorted(times, target))
            if idx >= times.size:
                idx = times.size - 1
            out[i, :] = npv_paths[idx, :]
            continue
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


def spot_from_quotes(pair6: str, inputs: _PythonLgmInputs, default: float = 1.0) -> float:
    base = pair6[:3].upper()
    quote = pair6[3:].upper()
    fwd = base + quote
    inv = quote + base
    if fwd in inputs.fx_spots:
        return float(inputs.fx_spots[fwd])
    if inv in inputs.fx_spots:
        return 1.0 / max(float(inputs.fx_spots[inv]), 1.0e-12)
    return float(default)


def today_spot_from_quotes(pair6: str, inputs: _PythonLgmInputs, default: float = 1.0) -> float:
    base = pair6[:3].upper()
    quote = pair6[3:].upper()
    fwd = base + quote
    inv = quote + base
    spots = inputs.fx_spots_today or inputs.fx_spots
    if fwd in spots:
        return float(spots[fwd])
    if inv in spots:
        return 1.0 / max(float(spots[inv]), 1.0e-12)
    return spot_from_quotes(pair6, inputs, default=default)


def convert_fx_forward_npv_to_reporting_ccy(
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
    cross = spot_from_quotes(report + quote, inputs, default=0.0)
    if cross > 0.0:
        return np.asarray(npv_quote, dtype=float) / cross
    cross_inv = spot_from_quotes(quote + report, inputs, default=0.0)
    if cross_inv > 0.0:
        return np.asarray(npv_quote, dtype=float) * cross_inv
    return np.asarray(npv_quote, dtype=float)


def fx_vol_for_trade(inputs: _PythonLgmInputs, pair6: str, maturity: float, default: float = 0.15) -> float:
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


def fx_carry_for_pair(pair: str, inputs: _PythonLgmInputs, horizon: float) -> float:
    base, quote = pair.upper().replace("-", "/").split("/")
    h = max(min(float(horizon), 1.0), 1.0 / 12.0)
    p_quote = max(float(inputs.discount_curves[quote](h)), 1.0e-18)
    p_base = max(float(inputs.discount_curves[base](h)), 1.0e-18)
    r_quote = -math.log(p_quote) / h
    r_base = -math.log(p_base) / h
    return float(r_quote - r_base)


__all__ = [
    "convert_fx_forward_npv_to_reporting_ccy",
    "fx_carry_for_pair",
    "fx_forward_closeout_paths",
    "fx_vol_for_trade",
    "price_fx_forward",
    "price_fx_forward_deterministic",
    "spot_from_quotes",
    "today_spot_from_quotes",
]
