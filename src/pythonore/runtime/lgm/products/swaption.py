from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from pythonore.runtime.lgm.types import _PricingContext, _TradeSpec


def price_swaption_premium_paths(
    adapter: Any,
    spec: _TradeSpec,
    ctx: _PricingContext,
) -> np.ndarray | None:
    if spec.sticky_state is None:
        return None
    premium_records = tuple(spec.sticky_state.get("premium_records", ()))
    if not premium_records:
        return None
    report_ccy = str(ctx.inputs.model_ccy or spec.ccy).upper()
    asof_date = adapter._irs_utils._parse_yyyymmdd(ctx.inputs.asof)
    premium_vals = np.zeros((ctx.n_times, ctx.n_paths), dtype=float)
    for i, t in enumerate(ctx.inputs.times):
        pv = np.zeros((ctx.n_paths,), dtype=float)
        for record in premium_records:
            pay_date_text = str(record.get("pay_date") or "").strip()
            if not pay_date_text:
                continue
            try:
                pay_date = adapter._irs_utils._parse_yyyymmdd(pay_date_text)
            except Exception:
                continue
            pay_time = float(adapter._irs_utils._time_from_dates(asof_date, pay_date, "A365F"))
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
            pv += adapter._convert_amount_to_reporting_ccy(
                local_pv,
                local_ccy=premium_ccy,
                report_ccy=report_ccy,
                inputs=ctx.inputs,
                shared_fx_sim=ctx.shared_fx_sim,
                time_index=i,
            )
        premium_vals[i, :] = pv
    return premium_vals


def price_trade_swaption_paths(
    adapter: Any,
    spec: _TradeSpec,
    ctx: _PricingContext,
) -> tuple[np.ndarray | None, bool]:
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
    p_fwd = adapter._resolve_index_curve(ctx.inputs, spec.ccy, float_index)
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
        vals = adapter._ir_options_mod.bermudan_npv_paths_from_underlying(
            model=ctx.model,
            p0_disc=p_disc,
            bermudan=definition,
            times=ctx.inputs.times,
            x_paths=ctx.x_paths,
            signed_swap=float(definition.exercise_sign) * np.asarray(signed_swap, dtype=float),
        )
        premium_vals = price_swaption_premium_paths(adapter, spec, ctx)
        if premium_vals is not None:
            vals = vals - premium_sign * premium_vals
        ctx.swaption_value_cache[swaption_cache_key] = vals
        return vals, False

    vals = adapter._ir_options_mod.bermudan_npv_paths(
        model=ctx.model,
        p0_disc=p_disc,
        p0_fwd=p_fwd,
        bermudan=definition,
        times=ctx.inputs.times,
        x_paths=ctx.x_paths,
    )
    premium_vals = price_swaption_premium_paths(adapter, spec, ctx)
    if premium_vals is not None:
        vals = vals - premium_sign * premium_vals
    ctx.swaption_value_cache[swaption_cache_key] = vals
    return vals, False


__all__ = ["price_swaption_premium_paths", "price_trade_swaption_paths"]
