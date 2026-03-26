"""LGM+FX product valuation and XVA helper functions.

This module sits one level above the model kernels:

- ``lgm.py`` supplies the single-currency Gaussian rate identities
- ``lgm_fx_hybrid.py`` supplies the correlated IR/FX path generator
- this module turns those states into ORE-style trade NPVs and exposure profiles

In practice the goal is to let Python-side examples consume trade or curve data in
an ORE-shaped form and produce profiles that are easy to compare with ORE exposure
reports, even though the implementation is intentionally narrower than the full ORE
analytics stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import numpy as np

try:
    from .lgm import LGMParams
    from .irs_xva_utils import swap_npv_from_ore_legs_dual_curve
    from .irs_xva_utils import build_discount_curve_from_zero_rate_pairs, survival_probability_from_hazard
    from .lgm_fx_hybrid import LgmFxHybrid
    from .lgm_fx_hybrid import MultiCcyLgmParams
    from .lgm_fx_hybrid_torch import TorchLgmFxHybrid, simulate_hybrid_paths_torch
except ImportError:  # pragma: no cover - script-mode fallback
    from lgm import LGMParams
    from irs_xva_utils import swap_npv_from_ore_legs_dual_curve
    from irs_xva_utils import build_discount_curve_from_zero_rate_pairs, survival_probability_from_hazard
    from lgm_fx_hybrid import LgmFxHybrid
    from lgm_fx_hybrid import MultiCcyLgmParams
    from lgm_fx_hybrid_torch import TorchLgmFxHybrid, simulate_hybrid_paths_torch


@dataclass(frozen=True)
class FxForwardDef:
    """Compact ORE-like FX forward trade definition used by the helper pricers."""

    trade_id: str
    pair: str  # BASE/QUOTE
    notional_base: float
    strike: float
    maturity: float
    pay_ccy: str | None = None


@dataclass(frozen=True)
class FxOptionDef:
    """Compact ORE-like European FX option definition used by hybrid pricers."""

    trade_id: str
    pair: str  # BASE/QUOTE
    notional_base: float
    strike: float
    maturity: float
    option_type: str  # CALL or PUT on BASE vs QUOTE
    long_short: str = "LONG"
    report_ccy: str | None = None


@dataclass(frozen=True)
class XccyFloatLegDef:
    """ORE-like floating leg for XCCY IRS.

    Arrays are leg-local and should be aligned by coupon index.
    Sign convention follows cashflow amount sign from perspective of the netting set.
    The layout is intentionally close to the arrays produced from ORE flow exports so
    that parsed trades can be re-used with minimal remapping.
    """

    ccy: str
    pay_time: np.ndarray
    start_time: np.ndarray
    end_time: np.ndarray
    accrual: np.ndarray
    notional: np.ndarray
    spread: np.ndarray
    sign: np.ndarray


def build_lgm_params(
    alpha,
    kappa,
    shift: float = 0.0,
    scaling: float = 1.0,
) -> LGMParams:
    """Convenience builder around ``LGMParams.from_spec``.

    This keeps the higher-level examples tolerant of both scalar inputs and the
    piecewise term-structure format commonly extracted from ORE calibration files.
    """
    if hasattr(LGMParams, "from_spec"):
        return LGMParams.from_spec(alpha=alpha, kappa=kappa, shift=shift, scaling=scaling)

    # Backward compatibility for kernels holding an older LGMParams class.
    def _coerce(spec):
        if np.isscalar(spec):
            return (), (float(spec),)
        if isinstance(spec, Mapping):
            if "times" not in spec or "values" not in spec:
                raise ValueError("mapping spec must include 'times' and 'values'")
            return tuple(float(x) for x in spec["times"]), tuple(float(x) for x in spec["values"])
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            return tuple(float(x) for x in spec[0]), tuple(float(x) for x in spec[1])
        raise ValueError("invalid piecewise spec")

    alpha_times, alpha_values = _coerce(alpha)
    kappa_times, kappa_values = _coerce(kappa)
    return LGMParams(
        alpha_times=alpha_times,
        alpha_values=alpha_values,
        kappa_times=kappa_times,
        kappa_values=kappa_values,
        shift=float(shift),
        scaling=float(scaling),
    )


def build_two_ccy_hybrid(
    pair: str,
    ir_specs: Mapping[str, Mapping[str, object]],
    fx_vol,
    corr_dom_fx: float = 0.0,
    corr_for_fx: float = 0.0,
) -> LgmFxHybrid:
    """Create a two-IR-factor + one-FX-factor hybrid for a currency pair.

    This is the smallest useful cross-asset configuration for ORE-style FX products:
    one foreign IR factor, one domestic IR factor, one FX spot factor.
    """
    base, quote = pair.upper().replace("-", "/").split("/")
    if base not in ir_specs or quote not in ir_specs:
        raise ValueError(f"ir_specs must include both currencies: {base}, {quote}")

    def _mk(spec: Mapping[str, object]) -> LGMParams:
        return build_lgm_params(
            alpha=spec["alpha"],
            kappa=spec["kappa"],
            shift=float(spec.get("shift", 0.0)),
            scaling=float(spec.get("scaling", 1.0)),
        )

    for_params = _mk(ir_specs[base])
    dom_params = _mk(ir_specs[quote])

    corr = np.array(
        [
            [1.0, 0.0, float(corr_for_fx)],
            [0.0, 1.0, float(corr_dom_fx)],
            [float(corr_for_fx), float(corr_dom_fx), 1.0],
        ],
        dtype=float,
    )
    return LgmFxHybrid(
        MultiCcyLgmParams(
            ir_params={base: for_params, quote: dom_params},
            fx_vols={pair: LGMParams._coerce_piecewise("fx_vol", fx_vol)},
            corr=corr,
        )
    )


def _as_curve_map(curves: Mapping[str, Callable[[float], float]]) -> Dict[str, Callable[[float], float]]:
    out: Dict[str, Callable[[float], float]] = {}
    for k, v in curves.items():
        kk = k.upper()
        if len(kk) != 3:
            raise ValueError(f"invalid currency key '{k}'")
        out[kk] = v
    return out


def _normalize_zero_rate_curve_input(zero_rate, horizon: float) -> List[Tuple[float, float]]:
    if np.isscalar(zero_rate):
        r = float(zero_rate)
        return [(0.0, r), (max(float(horizon), 1.0), r)]
    pts = sorted((float(t), float(v)) for t, v in zero_rate)
    if not pts:
        raise ValueError("zero rate input must not be empty")
    return pts


TensorBackend = Literal["auto", "numpy", "torch-cpu", "torch-mps"]


def _normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    vec = np.vectorize(lambda y: 0.5 * (1.0 + math.erf(float(y) / math.sqrt(2.0))))
    return vec(x)


def _torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch optional
        return False
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _torch_cpu_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch optional
        return False
    return torch is not None


def resolve_tensor_backend(
    tensor_backend: TensorBackend = "auto",
    *,
    case_family: Literal["fx_forward_profile_xva", "fx_forward_portfolio"] = "fx_forward_profile_xva",
) -> str:
    """Resolve a backend preference to an executable backend.

    ``auto`` is intentionally case-family aware rather than globally "prefer torch":
    the single-trade FX profile path is still faster and simpler on NumPy, while the
    batched portfolio path is the workflow where torch currently earns its keep.
    """
    backend = str(tensor_backend).lower()
    if backend not in {"auto", "numpy", "torch-cpu", "torch-mps"}:
        raise ValueError(f"unsupported tensor backend '{tensor_backend}'")
    if backend == "numpy":
        return "numpy"
    if backend == "torch-cpu":
        if not _torch_cpu_available():
            raise RuntimeError("torch-cpu requested but torch is not installed")
        return "torch-cpu"
    if backend == "torch-mps":
        if not _torch_cpu_available():
            raise RuntimeError("torch-mps requested but torch is not installed")
        if not _torch_mps_available():
            raise RuntimeError("torch-mps requested but MPS is not available in this process")
        return "torch-mps"
    if case_family == "fx_forward_profile_xva":
        return "numpy"
    if _torch_mps_available():
        return "torch-mps"
    if _torch_cpu_available():
        return "torch-cpu"
    return "numpy"


def fx_forward_npv(
    hybrid: LgmFxHybrid,
    fx_def: FxForwardDef,
    t: float,
    s_t: np.ndarray,
    x_dom_t: np.ndarray,
    x_for_t: np.ndarray,
    p0_dom: Callable[[float], float],
    p0_for: Callable[[float], float],
) -> np.ndarray:
    """Pathwise FX forward NPV in quote currency for trade paying at maturity.

    NPV_t = N_base * P_d(t,T) * (F_t(T) - K)
    where F_t(T) = S_t * P_f(t,T)/P_d(t,T)

    This matches the standard ORE reporting convention for an FX forward whose MtM
    is expressed in the quote/domestic currency.
    """
    base, quote = fx_def.pair.upper().replace("-", "/").split("/")
    if quote not in hybrid.ir_models or base not in hybrid.ir_models:
        raise ValueError("hybrid model missing required currencies for FX forward")

    # Keep the maturity-date payoff on-grid because ORE exposure reports usually show
    # the exposure that still exists on the payment/exercise date itself.
    if t > fx_def.maturity + 1.0e-12:
        return np.zeros_like(np.asarray(s_t, dtype=float))

    p_d = hybrid.zc_bond(quote, t, fx_def.maturity, x_dom_t, float(p0_dom(t)), float(p0_dom(fx_def.maturity)))
    p_f = hybrid.zc_bond(base, t, fx_def.maturity, x_for_t, float(p0_for(t)), float(p0_for(fx_def.maturity)))
    fwd = hybrid.fx_forward(fx_def.pair, t, fx_def.maturity, s_t, p_d, p_f)
    return fx_def.notional_base * p_d * (fwd - fx_def.strike)


def fx_option_npv(
    hybrid: LgmFxHybrid,
    fx_def: FxOptionDef,
    t: float,
    s_t: np.ndarray,
    x_dom_t: np.ndarray,
    x_for_t: np.ndarray,
    p0_dom: Callable[[float], float],
    p0_for: Callable[[float], float],
    vol: float | Callable[[float, float], float],
) -> np.ndarray:
    """Pathwise European FX option NPV in the requested reporting currency.

    The payoff is an option on the BASE currency expressed against QUOTE:
      N_base * max(S_T - K, 0) for a call, or max(K - S_T, 0) for a put.
    Black/Garman-Kohlhagen pricing is applied on each path using the hybrid
    pathwise forward and deterministic residual volatility input.
    """
    base, quote = fx_def.pair.upper().replace("-", "/").split("/")
    if quote not in hybrid.ir_models or base not in hybrid.ir_models:
        raise ValueError("hybrid model missing required currencies for FX option")

    s = np.asarray(s_t, dtype=float)
    if t > fx_def.maturity + 1.0e-12:
        return np.zeros_like(s)

    p_d = hybrid.zc_bond(quote, t, fx_def.maturity, x_dom_t, float(p0_dom(t)), float(p0_dom(fx_def.maturity)))
    p_f = hybrid.zc_bond(base, t, fx_def.maturity, x_for_t, float(p0_for(t)), float(p0_for(fx_def.maturity)))
    fwd = hybrid.fx_forward(fx_def.pair, t, fx_def.maturity, s, p_d, p_f)
    tau = max(float(fx_def.maturity) - float(t), 0.0)
    sigma = float(vol(float(t), float(fx_def.maturity)) if callable(vol) else vol)
    sigma = max(sigma, 0.0)
    stddev = sigma * math.sqrt(max(tau, 0.0))

    strike = float(fx_def.strike)
    if stddev > 0.0 and strike > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(np.clip(fwd, 1.0e-18, None) / strike) + 0.5 * stddev * stddev) / stddev
        d2 = d1 - stddev
        if str(fx_def.option_type).upper() == "CALL":
            undiscounted = fwd * _normal_cdf(d1) - strike * _normal_cdf(d2)
        else:
            undiscounted = strike * _normal_cdf(-d2) - fwd * _normal_cdf(-d1)
    else:
        if str(fx_def.option_type).upper() == "CALL":
            undiscounted = np.maximum(fwd - strike, 0.0)
        else:
            undiscounted = np.maximum(strike - fwd, 0.0)

    npv_quote = float(fx_def.notional_base) * np.asarray(p_d, dtype=float) * np.asarray(undiscounted, dtype=float)
    if str(fx_def.long_short).upper() == "SHORT":
        npv_quote = -npv_quote

    report_ccy = (fx_def.report_ccy or quote).upper()
    if report_ccy == quote:
        return npv_quote
    if report_ccy == base:
        return npv_quote / np.clip(s, 1.0e-18, None)
    raise ValueError(f"unsupported report currency '{report_ccy}' for pair {fx_def.pair}")


def fx_forward_npv_torch(
    hybrid: LgmFxHybrid,
    fx_def: FxForwardDef,
    t: float,
    s_t,
    x_dom_t,
    x_for_t,
    p0_dom: Callable[[float], float],
    p0_for: Callable[[float], float],
):
    import torch

    s = torch.as_tensor(s_t)
    if t > fx_def.maturity + 1.0e-12:
        return torch.zeros_like(s)
    base, quote = fx_def.pair.upper().replace("-", "/").split("/")
    p_d = torch.as_tensor(
        hybrid.zc_bond(quote, t, fx_def.maturity, torch.as_tensor(x_dom_t).detach().cpu().numpy(), float(p0_dom(t)), float(p0_dom(fx_def.maturity))),
        dtype=s.dtype,
        device=s.device,
    )
    p_f = torch.as_tensor(
        hybrid.zc_bond(base, t, fx_def.maturity, torch.as_tensor(x_for_t).detach().cpu().numpy(), float(p0_for(t)), float(p0_for(fx_def.maturity))),
        dtype=s.dtype,
        device=s.device,
    )
    fwd = s * p_f / p_d
    return float(fx_def.notional_base) * p_d * (fwd - float(fx_def.strike))


def fx_forward_portfolio_npv_paths_torch(
    hybrid: LgmFxHybrid,
    fx_defs: Sequence[FxForwardDef],
    times: np.ndarray,
    sim: Mapping[str, object],
    disc_curves: Mapping[str, Callable[[float], float]],
    fwd_curves: Mapping[str, Callable[[float], float]],
    *,
    return_numpy: bool = True,
):
    import torch

    if not fx_defs:
        raise ValueError("fx_defs must not be empty")
    t = np.asarray(times, dtype=float)
    curves_d = _as_curve_map(disc_curves)
    curves_f = _as_curve_map(fwd_curves)

    pair_groups: dict[str, list[FxForwardDef]] = {}
    for fx in fx_defs:
        pair_groups.setdefault(fx.pair.upper().replace("-", "/"), []).append(fx)

    sample_pair = next(iter(pair_groups))
    sample_base, sample_quote = sample_pair.split("/")
    s_any = sim["s"][sample_pair]
    device = s_any.device if hasattr(s_any, "device") else None
    dtype = s_any.dtype if hasattr(s_any, "dtype") else None
    out = torch.zeros((t.size, s_any.shape[1]), dtype=dtype, device=device)

    with torch.inference_mode():
        for pair, defs in pair_groups.items():
            base, quote = pair.split("/")
            s = torch.as_tensor(sim["s"][pair], dtype=dtype, device=device)
            x_dom = torch.as_tensor(sim["x"][quote], dtype=dtype, device=device)
            x_for = torch.as_tensor(sim["x"][base], dtype=dtype, device=device)
            maturities = np.asarray([float(d.maturity) for d in defs], dtype=float)
            strikes = torch.as_tensor([float(d.strike) for d in defs], dtype=dtype, device=device)
            notionals = torch.as_tensor([float(d.notional_base) for d in defs], dtype=dtype, device=device)
            h_t = torch.as_tensor(np.asarray(hybrid.ir_models[quote].H(t), dtype=float), dtype=dtype, device=device)
            z_t = torch.as_tensor(np.asarray(hybrid.ir_models[quote].zeta(t), dtype=float), dtype=dtype, device=device)
            h_td = torch.as_tensor(np.asarray(hybrid.ir_models[quote].H(maturities), dtype=float), dtype=dtype, device=device)
            h_tf = torch.as_tensor(np.asarray(hybrid.ir_models[base].H(maturities), dtype=float), dtype=dtype, device=device)
            p0_t_d = torch.as_tensor([float(curves_d[quote](float(tt))) for tt in t], dtype=dtype, device=device)
            p0_t_f = torch.as_tensor([float(curves_f[base](float(tt))) for tt in t], dtype=dtype, device=device)
            p0_T_d = torch.as_tensor([float(curves_d[quote](float(T))) for T in maturities], dtype=dtype, device=device)
            p0_T_f = torch.as_tensor([float(curves_f[base](float(T))) for T in maturities], dtype=dtype, device=device)
            d_hd = h_td[None, :] - h_t[:, None]
            d_h2d = h_td[None, :] * h_td[None, :] - h_t[:, None] * h_t[:, None]
            d_hf = h_tf[None, :] - torch.as_tensor(np.asarray(hybrid.ir_models[base].H(t), dtype=float), dtype=dtype, device=device)[:, None]
            d_h2f = h_tf[None, :] * h_tf[None, :] - torch.as_tensor(np.asarray(hybrid.ir_models[base].H(t), dtype=float), dtype=dtype, device=device)[:, None] ** 2
            z_f = torch.as_tensor(np.asarray(hybrid.ir_models[base].zeta(t), dtype=float), dtype=dtype, device=device)

            p_d = (p0_T_d[None, :] / p0_t_d[:, None])[:, :, None] * torch.exp(
                -d_hd[:, :, None] * x_dom[:, None, :] - 0.5 * d_h2d[:, :, None] * z_t[:, None, None]
            )
            p_f = (p0_T_f[None, :] / p0_t_f[:, None])[:, :, None] * torch.exp(
                -d_hf[:, :, None] * x_for[:, None, :] - 0.5 * d_h2f[:, :, None] * z_f[:, None, None]
            )
            fwd = s[:, None, :] * p_f / p_d
            mtm = notionals[None, :, None] * p_d * (fwd - strikes[None, :, None])
            live = torch.as_tensor((t[:, None] <= maturities[None, :] + 1.0e-12), dtype=dtype, device=device)[:, :, None]
            out = out + torch.sum(mtm * live, dim=1)

    return out.detach().cpu().numpy() if return_numpy else out


def fx_forward_portfolio_npv_paths(
    hybrid: LgmFxHybrid,
    fx_defs: Sequence[FxForwardDef],
    times: np.ndarray,
    sim: Mapping[str, object],
    disc_curves: Mapping[str, Callable[[float], float]],
    fwd_curves: Mapping[str, Callable[[float], float]],
    *,
    tensor_backend: TensorBackend = "auto",
    return_numpy: bool = True,
):
    """Unified portfolio pricer that dispatches to NumPy or torch underneath."""
    backend = resolve_tensor_backend(tensor_backend, case_family="fx_forward_portfolio")
    if backend == "numpy":
        t = np.asarray(times, dtype=float)
        if not fx_defs:
            raise ValueError("fx_defs must not be empty")
        out = np.zeros((t.size, np.asarray(sim["s"][fx_defs[0].pair.upper().replace("-", "/")], dtype=float).shape[1]), dtype=float)
        curves_d = _as_curve_map(disc_curves)
        curves_f = _as_curve_map(fwd_curves)
        for fx in fx_defs:
            base, quote = fx.pair.upper().replace("-", "/").split("/")
            pair = f"{base}/{quote}"
            s = np.asarray(sim["s"][pair], dtype=float)
            x_dom = np.asarray(sim["x"][quote], dtype=float)
            x_for = np.asarray(sim["x"][base], dtype=float)
            for i, ti in enumerate(t):
                out[i, :] += fx_forward_npv(hybrid, fx, float(ti), s[i, :], x_dom[i, :], x_for[i, :], curves_d[quote], curves_f[base])
        return out
    device = "mps" if backend == "torch-mps" else "cpu"
    return fx_forward_portfolio_npv_paths_torch(
        hybrid,
        fx_defs,
        times,
        sim,
        disc_curves,
        fwd_curves,
        return_numpy=return_numpy,
    )


def xccy_float_float_swap_npv(
    hybrid: LgmFxHybrid,
    domestic_ccy: str,
    leg1: XccyFloatLegDef,
    leg2: XccyFloatLegDef,
    t: float,
    x_by_ccy: Mapping[str, np.ndarray],
    s_fx_by_pair: Mapping[str, np.ndarray],
    disc_curves: Mapping[str, Callable[[float], float]],
    fwd_curves: Mapping[str, Callable[[float], float]],
) -> np.ndarray:
    """Pathwise NPV of two floating legs converted to domestic currency.

    Each floating coupon i is projected by simple forward from the leg's forwarding
    curve and discounted on the leg currency discount curve, then converted at S_t
    if leg ccy != domestic_ccy.

    The leg representation is intentionally simpler than ORE's full XCCY swap model,
    but it preserves the core ORE idea of separate discounting / forwarding curves
    and explicit FX conversion into a chosen reporting currency.
    """
    dccy = domestic_ccy.upper()
    curves_d = _as_curve_map(disc_curves)
    curves_f = _as_curve_map(fwd_curves)

    def leg_npv_dom(leg: XccyFloatLegDef) -> np.ndarray:
        c = leg.ccy.upper()
        x_t = np.asarray(x_by_ccy[c], dtype=float)
        p0_d = curves_d[c]
        p0_f = curves_f[c]
        n_paths = x_t.size

        live = leg.pay_time > t + 1.0e-12
        if not np.any(live):
            return np.zeros(n_paths, dtype=float)

        s = leg.start_time[live]
        e = leg.end_time[live]
        pay = leg.pay_time[live]
        tau = leg.accrual[live]
        n = leg.notional[live]
        spr = np.nan_to_num(leg.spread[live], nan=0.0, posinf=0.0, neginf=0.0)
        sign = leg.sign[live]
        effective_s = np.maximum(s, float(t))

        p_t_s_d = np.array([
            hybrid.zc_bond(c, t, float(si), x_t, float(p0_d(t)), float(p0_d(float(si)))) for si in effective_s
        ])
        p_t_e_d = np.array([
            hybrid.zc_bond(c, t, float(ei), x_t, float(p0_d(t)), float(p0_d(float(ei)))) for ei in e
        ])
        p_t_pay_d = np.array([
            hybrid.zc_bond(c, t, float(pi), x_t, float(p0_d(t)), float(p0_d(float(pi)))) for pi in pay
        ])

        # Deterministic basis ratio maps the simulated discounting bond into the
        # forwarding bond.  This is the same approximation used elsewhere in the
        # package to keep one LGM state while still honouring dual-curve inputs.
        b_t = float(p0_f(t) / max(p0_d(t), 1.0e-18))
        b_s = np.array([float(p0_f(float(si)) / max(p0_d(float(si)), 1.0e-18)) for si in effective_s], dtype=float)
        b_e = np.array([float(p0_f(float(ei)) / p0_d(float(ei))) for ei in e], dtype=float)
        p_t_s_f = p_t_s_d * (b_s / b_t)[:, None]
        p_t_e_f = p_t_e_d * (b_e / b_t)[:, None]

        fwd = np.nan_to_num((p_t_s_f / p_t_e_f - 1.0) / np.clip(tau[:, None], 1.0e-8, None), nan=0.0, posinf=0.0, neginf=0.0)
        cash = sign[:, None] * n[:, None] * (fwd + spr[:, None]) * tau[:, None]
        pv_ccy = np.sum(np.nan_to_num(cash * p_t_pay_d, nan=0.0, posinf=0.0, neginf=0.0), axis=0)

        if c == dccy:
            return pv_ccy

        pair = f"{c}/{dccy}"
        pair_inv = f"{dccy}/{c}"
        if pair in s_fx_by_pair:
            spot = np.asarray(s_fx_by_pair[pair], dtype=float)
            return pv_ccy * spot
        if pair_inv in s_fx_by_pair:
            spot_inv = np.asarray(s_fx_by_pair[pair_inv], dtype=float)
            return pv_ccy / np.clip(spot_inv, 1.0e-18, None)
        raise ValueError(f"missing FX spot state for conversion: need {pair} or {pair_inv}")

    return leg_npv_dom(leg1) + leg_npv_dom(leg2)


def single_ccy_irs_npv(
    model,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    legs: Dict[str, np.ndarray],
    t: float,
    x_t: np.ndarray,
) -> np.ndarray:
    """Thin wrapper to keep valuation entrypoints centralized."""
    return swap_npv_from_ore_legs_dual_curve(model, p0_disc, p0_fwd, legs, t, x_t)


def aggregate_exposure_profile(npv_paths: np.ndarray) -> Dict[str, np.ndarray]:
    """Return EE/EPE/ENE time profiles from pathwise NPV matrix [n_times, n_paths].

    These are the same exposure statistics that appear in ORE exposure outputs:
    expected exposure, expected positive exposure, and expected negative exposure.
    """
    if npv_paths.ndim != 2:
        raise ValueError("npv_paths must be 2D [n_times, n_paths]")
    ee = np.mean(npv_paths, axis=1)
    epe = np.mean(np.maximum(npv_paths, 0.0), axis=1)
    ene = np.mean(np.minimum(npv_paths, 0.0), axis=1)
    return {"ee": ee, "epe": epe, "ene": ene}


def apply_mpor_closeout(
    npv_paths: np.ndarray,
    times: np.ndarray,
    mpor_years: float = 0.0,
    sticky: bool = True,
) -> np.ndarray:
    """Map valuation NPV paths to closeout NPVs using a margin period of risk.

    A sticky closeout maps V(t) -> V(t + mpor) along the same simulated path.
    For non-sticky mode in this lightweight implementation, we currently apply the
    same mapping because no additional state re-simulation is available at closeout.

    This helper remains useful for benchmarks and notebooks, but the authoritative
    sticky-MPOR logic used by ``PythonLgmAdapter`` now revalues trades directly on
    a union valuation/closeout grid in ``native_xva_interface.runtime``.
    """
    v = np.asarray(npv_paths, dtype=float)
    t = np.asarray(times, dtype=float)
    if v.ndim != 2:
        raise ValueError("npv_paths must be 2D [n_times, n_paths]")
    if t.ndim != 1 or t.size != v.shape[0]:
        raise ValueError("times must be 1D and match npv_paths first dimension")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    mpor = float(mpor_years)
    if mpor < 0.0:
        raise ValueError("mpor_years must be non-negative")
    if mpor <= 1.0e-15:
        return v.copy()

    # Non-sticky fallback currently uses the same pathwise interpolation map.
    _ = bool(sticky)
    tc = np.minimum(t + mpor, t[-1])
    out = np.empty_like(v)
    for p in range(v.shape[1]):
        out[:, p] = np.interp(tc, t, v[:, p])
    return out


def cva_terms_from_profile(
    times: np.ndarray,
    epe: np.ndarray,
    discount: np.ndarray,
    survival: np.ndarray,
    recovery: float,
) -> Dict[str, np.ndarray]:
    """Unilateral CVA decomposition on exposure grid.

    term_i = LGD * EPE_i * DF_i * dPD_i
    where dPD_i = Q(t_{i-1}) - Q(t_i), i>=1.

    The output is intentionally decomposed into incremental terms because that is the
    most useful format when reconciling a Python profile against ORE-style XVA runs.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(epe, dtype=float)
    df = np.asarray(discount, dtype=float)
    q = np.asarray(survival, dtype=float)
    if not (t.shape == e.shape == df.shape == q.shape):
        raise ValueError("times, epe, discount, survival must have identical shape")
    if t.size < 2:
        raise ValueError("need at least two grid points for CVA terms")
    lgd = 1.0 - float(recovery)

    dpd = np.zeros_like(t)
    dpd[1:] = np.clip(q[:-1] - q[1:], 0.0, None)
    terms = lgd * e * df * dpd
    return {"dpd": dpd, "terms": terms, "cva": np.array([float(np.sum(terms))], dtype=float)}


def run_fx_forward_profile_xva(
    *,
    name: str,
    pair: str,
    maturity: float,
    spot0: float,
    strike: float,
    notional_base: float,
    dom_zero_rate,
    for_zero_rate,
    fx_vol=0.12,
    alpha_dom=0.010,
    alpha_for=0.010,
    kappa_dom=0.03,
    kappa_for=0.03,
    corr_dom_fx: float = 0.0,
    corr_for_fx: float = 0.0,
    n_paths: int = 30000,
    seed: int = 2026,
    cpty_hazard: float = 0.015,
    own_hazard: float = 0.010,
    recovery_cpty: float = 0.40,
    recovery_own: float = 0.40,
    funding_spread: float = 0.0015,
) -> Dict[str, object]:
    """Simulate profile and XVA for one FX forward (quote currency reporting).

    This is an end-to-end convenience entrypoint:
    curves -> hybrid model -> pathwise NPV -> EE/EPE/ENE -> unilateral CVA/DVA/FVA.
    It is mainly intended for notebook demonstrations and coarse comparisons to ORE
    profile exports rather than as a production risk engine.
    """
    base, quote = pair.upper().replace("-", "/").split("/")

    # Use a compact grid suitable for example exposure profiles.  ORE usually has a
    # richer simulation grid configured in XML; callers needing parity can pass / use
    # those grids in lower-level routines instead of this convenience function.
    times = np.unique(np.concatenate([np.array([0.0, float(maturity)]), np.linspace(0.0, float(maturity), 25)]))
    dom_curve = _normalize_zero_rate_curve_input(dom_zero_rate, float(maturity) + 10.0)
    for_curve = _normalize_zero_rate_curve_input(for_zero_rate, float(maturity) + 10.0)
    p0_dom = build_discount_curve_from_zero_rate_pairs(dom_curve)
    p0_for = build_discount_curve_from_zero_rate_pairs(for_curve)

    hybrid = build_two_ccy_hybrid(
        pair=pair,
        ir_specs={
            quote: {"alpha": alpha_dom, "kappa": kappa_dom},
            base: {"alpha": alpha_for, "kappa": kappa_for},
        },
        fx_vol=fx_vol,
        corr_dom_fx=corr_dom_fx,
        corr_for_fx=corr_for_fx,
    )

    fx_def = FxForwardDef(
        trade_id=name,
        pair=f"{base}/{quote}",
        notional_base=float(notional_base),
        strike=float(strike),
        maturity=float(maturity),
    )
    rd_rf = float(dom_curve[0][1]) - float(for_curve[0][1])
    rng = np.random.default_rng(int(seed))
    sim = hybrid.simulate_paths(
        times=times,
        n_paths=int(n_paths),
        rng=rng,
        log_s0={f"{base}/{quote}": float(np.log(spot0))},
        rd_minus_rf={f"{base}/{quote}": rd_rf},
    )

    x_dom = sim["x"][quote]
    x_for = sim["x"][base]
    s = sim["s"][f"{base}/{quote}"]

    npv_paths = np.empty((times.size, int(n_paths)), dtype=float)
    for i, t in enumerate(times):
        npv_paths[i, :] = fx_forward_npv(hybrid, fx_def, float(t), s[i, :], x_dom[i, :], x_for[i, :], p0_dom, p0_for)

    exp = aggregate_exposure_profile(npv_paths)
    ee = exp["ee"]
    epe = exp["epe"]
    ene = exp["ene"]

    df = np.asarray([p0_dom(float(t)) for t in times], dtype=float)
    hz_times = np.array([float(maturity)], dtype=float)
    q_cpty = survival_probability_from_hazard(times, hz_times, np.array([float(cpty_hazard)], dtype=float))
    q_own = survival_probability_from_hazard(times, hz_times, np.array([float(own_hazard)], dtype=float))
    cva_pack = cva_terms_from_profile(times, epe, df, q_cpty, recovery_cpty)
    dva_pack = cva_terms_from_profile(times, -ene, df, q_own, recovery_own)

    dpd_own = np.zeros_like(times)
    dpd_own[1:] = np.clip(q_own[:-1] - q_own[1:], 0.0, None)
    fca_terms = funding_spread * epe * df * dpd_own

    cva = float(cva_pack["cva"][0])
    dva = float(dva_pack["cva"][0])
    fva = float(np.sum(fca_terms))
    return {
        "name": name,
        "pair": f"{base}/{quote}",
        "times": times,
        "ee": ee,
        "epe": epe,
        "ene": ene,
        "npv_paths": npv_paths,
        "cva": cva,
        "dva": dva,
        "fva": fva,
        "xva_total": cva - dva + fva,
    }


def run_fx_forward_profile_xva_torch(
    *,
    name: str,
    pair: str,
    maturity: float,
    spot0: float,
    strike: float,
    notional_base: float,
    dom_zero_rate,
    for_zero_rate,
    fx_vol=0.12,
    alpha_dom=0.010,
    alpha_for=0.010,
    kappa_dom=0.03,
    kappa_for=0.03,
    corr_dom_fx: float = 0.0,
    corr_for_fx: float = 0.0,
    n_paths: int = 30000,
    seed: int = 2026,
    cpty_hazard: float = 0.015,
    own_hazard: float = 0.010,
    recovery_cpty: float = 0.40,
    recovery_own: float = 0.40,
    funding_spread: float = 0.0015,
    device: str = "cpu",
) -> Dict[str, object]:
    import torch

    base, quote = pair.upper().replace("-", "/").split("/")
    times = np.unique(np.concatenate([np.array([0.0, float(maturity)]), np.linspace(0.0, float(maturity), 25)]))
    dom_curve = _normalize_zero_rate_curve_input(dom_zero_rate, float(maturity) + 10.0)
    for_curve = _normalize_zero_rate_curve_input(for_zero_rate, float(maturity) + 10.0)
    p0_dom = build_discount_curve_from_zero_rate_pairs(dom_curve)
    p0_for = build_discount_curve_from_zero_rate_pairs(for_curve)

    hybrid_np = build_two_ccy_hybrid(
        pair=pair,
        ir_specs={
            quote: {"alpha": alpha_dom, "kappa": kappa_dom},
            base: {"alpha": alpha_for, "kappa": kappa_for},
        },
        fx_vol=fx_vol,
        corr_dom_fx=corr_dom_fx,
        corr_for_fx=corr_for_fx,
    )
    hybrid = TorchLgmFxHybrid(hybrid_np.params)

    fx_def = FxForwardDef(
        trade_id=name,
        pair=f"{base}/{quote}",
        notional_base=float(notional_base),
        strike=float(strike),
        maturity=float(maturity),
    )
    rd_rf = float(dom_curve[0][1]) - float(for_curve[0][1])
    sim = simulate_hybrid_paths_torch(
        hybrid,
        times=times,
        n_paths=int(n_paths),
        rng=np.random.default_rng(int(seed)),
        log_s0={f"{base}/{quote}": float(np.log(spot0))},
        rd_minus_rf={f"{base}/{quote}": rd_rf},
        device=device,
        return_numpy=False,
    )

    x_dom = sim["x"][quote]
    x_for = sim["x"][base]
    s = sim["s"][f"{base}/{quote}"]

    with torch.inference_mode():
        npv_paths = torch.empty((times.size, int(n_paths)), dtype=s.dtype, device=s.device)
        for i, t in enumerate(times):
            npv_paths[i, :] = fx_forward_npv_torch(hybrid_np, fx_def, float(t), s[i, :], x_dom[i, :], x_for[i, :], p0_dom, p0_for)
        npv_np = npv_paths.detach().cpu().numpy()

    exp = aggregate_exposure_profile(npv_np)
    ee = exp["ee"]
    epe = exp["epe"]
    ene = exp["ene"]
    df = np.asarray([p0_dom(float(t)) for t in times], dtype=float)
    hz_times = np.array([float(maturity)], dtype=float)
    q_cpty = survival_probability_from_hazard(times, hz_times, np.array([float(cpty_hazard)], dtype=float))
    q_own = survival_probability_from_hazard(times, hz_times, np.array([float(own_hazard)], dtype=float))
    cva_pack = cva_terms_from_profile(times, epe, df, q_cpty, recovery_cpty)
    dva_pack = cva_terms_from_profile(times, -ene, df, q_own, recovery_own)
    dpd_own = np.zeros_like(times)
    dpd_own[1:] = np.clip(q_own[:-1] - q_own[1:], 0.0, None)
    fca_terms = funding_spread * epe * df * dpd_own

    cva = float(cva_pack["cva"][0])
    dva = float(dva_pack["cva"][0])
    fva = float(np.sum(fca_terms))
    return {
        "name": name,
        "pair": f"{base}/{quote}",
        "times": times,
        "ee": ee,
        "epe": epe,
        "ene": ene,
        "npv_paths": npv_np,
        "cva": cva,
        "dva": dva,
        "fva": fva,
        "xva_total": cva - dva + fva,
    }


def run_fx_forward_profile_xva_backend(
    *,
    tensor_backend: TensorBackend = "auto",
    **kwargs,
) -> Dict[str, object]:
    """Unified FX forward profile/XVA entrypoint with backend dispatch."""
    backend = resolve_tensor_backend(tensor_backend, case_family="fx_forward_profile_xva")
    if backend == "numpy":
        out = run_fx_forward_profile_xva(**kwargs)
        out["backend_used"] = "numpy"
        return out
    device = "mps" if backend == "torch-mps" else "cpu"
    out = run_fx_forward_profile_xva_torch(**kwargs, device=device)
    out["backend_used"] = backend
    return out


def run_fx_forward_example(
    *,
    name: str,
    pair: str,
    maturity: float,
    spot0: float,
    strike: float,
    notional_base: float,
    dom_zero_rate,
    for_zero_rate,
    fx_vol=0.12,
    alpha_dom=0.010,
    alpha_for=0.010,
    kappa_dom=0.03,
    kappa_for=0.03,
    corr_dom_fx: float = 0.0,
    corr_for_fx: float = 0.0,
    n_paths: int = 20000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Quick FX forward sanity run with deterministic t0 and terminal path stats.

    Useful as a smoke test before comparing the richer profile output against ORE.
    """
    out = run_fx_forward_profile_xva(
        name=name,
        pair=pair,
        maturity=maturity,
        spot0=spot0,
        strike=strike,
        notional_base=notional_base,
        dom_zero_rate=dom_zero_rate,
        for_zero_rate=for_zero_rate,
        fx_vol=fx_vol,
        alpha_dom=alpha_dom,
        alpha_for=alpha_for,
        kappa_dom=kappa_dom,
        kappa_for=kappa_for,
        corr_dom_fx=corr_dom_fx,
        corr_for_fx=corr_for_fx,
        n_paths=n_paths,
        seed=seed,
    )
    base, quote = pair.upper().replace("-", "/").split("/")
    dom_curve = _normalize_zero_rate_curve_input(dom_zero_rate, float(maturity) + 10.0)
    for_curve = _normalize_zero_rate_curve_input(for_zero_rate, float(maturity) + 10.0)
    p0_dom = build_discount_curve_from_zero_rate_pairs(dom_curve)
    p0_for = build_discount_curve_from_zero_rate_pairs(for_curve)
    fwd0 = float(spot0) * float(p0_for(float(maturity))) / float(p0_dom(float(maturity)))
    npv0 = float(notional_base) * float(p0_dom(float(maturity))) * (fwd0 - float(strike))
    terminal = out["npv_paths"][-1, :]
    return {
        "name": name,
        "pair": f"{base}/{quote}",
        "maturity": float(maturity),
        "spot0": float(spot0),
        "strike": float(strike),
        "fwd0": fwd0,
        "npv0": npv0,
        "mtm_at_maturity_mean": float(np.mean(terminal)),
        "mtm_at_maturity_p05": float(np.percentile(terminal, 5.0)),
        "mtm_at_maturity_p95": float(np.percentile(terminal, 95.0)),
    }


__all__ = [
    "TensorBackend",
    "resolve_tensor_backend",
    "FxForwardDef",
    "FxOptionDef",
    "XccyFloatLegDef",
    "build_lgm_params",
    "build_two_ccy_hybrid",
    "fx_forward_npv",
    "fx_option_npv",
    "fx_forward_npv_torch",
    "fx_forward_portfolio_npv_paths",
    "fx_forward_portfolio_npv_paths_torch",
    "xccy_float_float_swap_npv",
    "single_ccy_irs_npv",
    "aggregate_exposure_profile",
    "apply_mpor_closeout",
    "cva_terms_from_profile",
    "run_fx_forward_profile_xva",
    "run_fx_forward_profile_xva_backend",
    "run_fx_forward_profile_xva_torch",
    "run_fx_forward_example",
]
