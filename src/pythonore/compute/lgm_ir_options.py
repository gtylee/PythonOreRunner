"""LGM IR option helpers (cap/floor and Bermudan swaption via LSMC).

These routines extend the ORE-style linear IR trade helpers with optionality while
staying on the same modelling stack:

- the underlying rate dynamics still come from ``lgm.py``
- swap cashflow handling still comes from ``irs_xva_utils.py``
- the output is still pathwise NPV profiles suitable for exposure/XVA analysis

The focus is educational / workflow-oriented rather than feature-complete ORE parity.
The comments below therefore emphasise how each approximation maps back to concepts
that ORE users will recognise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np

try:
    from .irs_xva_utils import curve_values, interpolate_path_grid
    from .irs_xva_utils import swap_npv_from_ore_legs_dual_curve
    from .lgm import LGM1F
except ImportError:  # pragma: no cover - script-mode fallback
    from irs_xva_utils import curve_values, interpolate_path_grid
    from irs_xva_utils import swap_npv_from_ore_legs_dual_curve
    from lgm import LGM1F


@dataclass(frozen=True)
class CapFloorDef:
    """ORE-like cap/floor coupon definition arrays.

    Each array is coupon-indexed so a cap/floor can be valued pathwise without
    carrying a heavyweight trade object.  This mirrors how ORE cashflow reports are
    often flattened into arrays for analysis.
    """

    trade_id: str
    ccy: str
    option_type: str  # "cap" or "floor"
    start_time: np.ndarray
    end_time: np.ndarray
    pay_time: np.ndarray
    accrual: np.ndarray
    notional: np.ndarray
    strike: np.ndarray
    gearing: np.ndarray | None = None
    spread: np.ndarray | None = None
    fixing_time: np.ndarray | None = None
    fixing_date: np.ndarray | None = None
    position: float = 1.0  # +1 long, -1 short

    def __post_init__(self) -> None:
        t0 = np.asarray(self.start_time, dtype=float)
        t1 = np.asarray(self.end_time, dtype=float)
        tp = np.asarray(self.pay_time, dtype=float)
        tau = np.asarray(self.accrual, dtype=float)
        n = np.asarray(self.notional, dtype=float)
        k = np.asarray(self.strike, dtype=float)
        g = np.ones_like(k) if self.gearing is None else np.asarray(self.gearing, dtype=float)
        s = np.zeros_like(k) if self.spread is None else np.asarray(self.spread, dtype=float)
        if not (t0.ndim == t1.ndim == tp.ndim == tau.ndim == n.ndim == k.ndim == 1):
            raise ValueError("cap/floor arrays must be one-dimensional")
        m = t0.size
        if not (t1.size == tp.size == tau.size == n.size == k.size == g.size == s.size == m):
            raise ValueError("cap/floor arrays must have equal length")
        if m == 0:
            raise ValueError("cap/floor must contain at least one coupon")
        if np.any(t1 < t0) or np.any(tp < t1):
            raise ValueError("invalid cap/floor times ordering")
        if np.any(tau <= 0.0):
            raise ValueError("accrual factors must be positive")
        o = self.option_type.strip().lower()
        if o not in ("cap", "floor"):
            raise ValueError("option_type must be 'cap' or 'floor'")
        if self.fixing_time is not None:
            tf = np.asarray(self.fixing_time, dtype=float)
            if tf.shape != t0.shape:
                raise ValueError("fixing_time must match coupon array shape")
        if self.fixing_date is not None:
            fd = np.asarray(self.fixing_date, dtype=object)
            if fd.shape != t0.shape:
                raise ValueError("fixing_date must match coupon array shape")


@dataclass(frozen=True)
class BermudanSwaptionDef:
    """Bermudan option on a swap represented by ORE-like leg arrays.

    ``underlying_legs`` is expected to have the same shape and semantics as the swap
    leg dictionaries produced / consumed by ``irs_xva_utils``.
    """

    trade_id: str
    exercise_times: np.ndarray
    underlying_legs: Dict[str, np.ndarray]
    exercise_sign: float = 1.0  # +1 for max(swap,0), -1 for max(-swap,0)
    settlement: str = "physical"  # "physical" or "cash"

    def __post_init__(self) -> None:
        ex = np.asarray(self.exercise_times, dtype=float)
        if ex.ndim != 1 or ex.size == 0:
            raise ValueError("exercise_times must be a non-empty one-dimensional array")
        if np.any(ex < 0.0) or np.any(np.diff(ex) <= 0.0):
            raise ValueError("exercise_times must be strictly increasing and non-negative")
        if abs(float(self.exercise_sign)) <= 0.0:
            raise ValueError("exercise_sign must be non-zero")
        s = self.settlement.strip().lower()
        if s not in ("physical", "cash"):
            raise ValueError("settlement must be 'physical' or 'cash'")


@dataclass(frozen=True)
class BermudanExerciseDiagnostic:
    time: float
    intrinsic_mean: float
    continuation_mean: float
    exercise_probability: float
    active_paths: int
    boundary_state: float | None = None


@dataclass(frozen=True)
class BermudanLsmcResult:
    npv_paths: np.ndarray
    option_npv_paths: np.ndarray
    signed_underlying_paths: np.ndarray
    exercise_indices: np.ndarray
    diagnostics: tuple[BermudanExerciseDiagnostic, ...]


@dataclass(frozen=True)
class BermudanBackwardResult:
    price: float
    exercise_values: tuple[tuple[float, np.ndarray], ...]
    diagnostics: tuple[BermudanExerciseDiagnostic, ...]


def _to_1d(arr: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return out


def _fixing_times(cf: CapFloorDef) -> np.ndarray:
    if cf.fixing_time is None:
        return np.asarray(cf.start_time, dtype=float)
    return np.asarray(cf.fixing_time, dtype=float)


def _basis_polynomial_1d(x: np.ndarray, degree: int) -> np.ndarray:
    x1 = np.asarray(x, dtype=float)
    cols = [np.ones_like(x1)]
    for d in range(1, degree + 1):
        cols.append(np.power(x1, d))
    return np.column_stack(cols)


def _basis_cache_key(x: np.ndarray, degree: int) -> tuple[int, tuple[int, ...], bytes]:
    x1 = np.ascontiguousarray(np.asarray(x, dtype=float))
    return int(degree), x1.shape, x1.tobytes()


def _basis_polynomial_cached(
    cache: dict[tuple[int, tuple[int, ...], bytes], np.ndarray],
    x: np.ndarray,
    degree: int,
) -> np.ndarray:
    key = _basis_cache_key(x, degree)
    cached = cache.get(key)
    if cached is not None:
        return cached
    out = _basis_polynomial_1d(x, degree)
    cache[key] = out
    return out


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float) / np.sqrt(2.0)
    erf_vec = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(z))


def forward_rate_from_bonds(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    t: float,
    x_t: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    accrual: np.ndarray,
    fixing_time: float | None = None,
) -> np.ndarray:
    """Pathwise simple forward rates for a coupon set at valuation time t.

    The forwarding curve is reconstructed from the discounting-state bond via the
    deterministic basis ratio, matching the dual-curve approximation used elsewhere
    in these ORE-style helpers.
    """
    x = _to_1d(x_t, "x_t")
    s = _to_1d(start, "start")
    e = _to_1d(end, "end")
    tau = _to_1d(accrual, "accrual")
    if not (s.size == e.size == tau.size):
        raise ValueError("start/end/accrual size mismatch")
    if fixing_time is not None:
        s = np.maximum(s, float(fixing_time))
    p_t = float(p0_disc(t))
    p_ts_d = model.discount_bond_paths(t, s, x, p_t, lambda u: float(p0_disc(float(u))))
    p_te_d = model.discount_bond_paths(t, e, x, p_t, lambda u: float(p0_disc(float(u))))

    bt = float(p0_fwd(t) / p0_disc(t))
    bs = np.array([float(p0_fwd(float(u)) / p0_disc(float(u))) for u in s], dtype=float)
    be = np.array([float(p0_fwd(float(u)) / p0_disc(float(u))) for u in e], dtype=float)
    p_ts_f = p_ts_d * (bs / bt)[:, None]
    p_te_f = p_te_d * (be / bt)[:, None]
    return (p_ts_f / p_te_f - 1.0) / tau[:, None]


def capfloor_npv(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    capfloor: CapFloorDef,
    t: float,
    x_t: np.ndarray,
    realized_forward: np.ndarray | None = None,
) -> np.ndarray:
    """Pathwise cap/floor NPV at valuation time t.

    Coupon handling is split the same way an ORE user would reason about it:
    fixed coupons are deterministic discounted cashflows, while future coupons are
    option payoffs on still-unfixed forward rates.
    """
    x = _to_1d(x_t, "x_t")
    if t < 0.0:
        raise ValueError("t must be non-negative")

    start = np.asarray(capfloor.start_time, dtype=float)
    end = np.asarray(capfloor.end_time, dtype=float)
    pay = np.asarray(capfloor.pay_time, dtype=float)
    tau = np.asarray(capfloor.accrual, dtype=float)
    notional = np.asarray(capfloor.notional, dtype=float)
    strike = np.asarray(capfloor.strike, dtype=float)
    fixing = _fixing_times(capfloor)

    # ORE includes cashflows that pay on the valuation date when today events are
    # enabled, so keep coupons with pay_time == t in scope.
    live = pay >= t - 1.0e-12
    if not np.any(live):
        return np.zeros_like(x)

    s = start[live]
    e = end[live]
    p = pay[live]
    a = tau[live]
    n = notional[live]
    k = strike[live]
    g = np.asarray(capfloor.gearing if capfloor.gearing is not None else np.ones_like(strike), dtype=float)[live]
    sread = np.asarray(capfloor.spread if capfloor.spread is not None else np.zeros_like(strike), dtype=float)[live]
    f = fixing[live]
    fixed_mask = f <= t + 1.0e-12

    l = np.zeros((p.size, x.size), dtype=float)
    if np.any(fixed_mask):
        if realized_forward is None:
            # Fallback to deterministic forward from curve if fixings are unavailable.
            ps = np.array([float(p0_fwd(float(u))) for u in s[fixed_mask]], dtype=float)
            pe = np.array([float(p0_fwd(float(u))) for u in e[fixed_mask]], dtype=float)
            l[fixed_mask, :] = ((ps / pe - 1.0) / a[fixed_mask])[:, None]
        else:
            rf = np.asarray(realized_forward, dtype=float)
            if rf.shape != (p.size, x.size):
                raise ValueError("realized_forward shape must be (n_live_coupons, n_paths)")
            l[fixed_mask, :] = rf[fixed_mask, :]
    # Once the fixing date has passed, the caplet/floorlet is just a known coupon
    # amount that still needs discounting to the valuation time.
    if np.any(fixed_mask):
        effective_rate = g[fixed_mask, None] * l[fixed_mask, :] + sread[fixed_mask, None]
        if capfloor.option_type.strip().lower() == "cap":
            payoff_fix = np.maximum(effective_rate - k[fixed_mask, None], 0.0)
        else:
            payoff_fix = np.maximum(k[fixed_mask, None] - effective_rate, 0.0)
        amount_fix = float(capfloor.position) * n[fixed_mask, None] * a[fixed_mask, None] * payoff_fix
        p_t = float(p0_disc(t))
        disc_fix = model.discount_bond_paths(t, p[fixed_mask], x, p_t, lambda u: float(p0_disc(float(u))))
        pv_fix = np.sum(amount_fix * disc_fix, axis=0)
    else:
        pv_fix = np.zeros_like(x)

    # For still-unfixed coupons, use the Gaussian-LGM bond-option representation.
    # This is the compact analytical analogue of the cap/floor pricing logic used in
    # ORE's LGM-based interest-rate option machinery.
    pv_unfixed = np.zeros_like(x)
    if np.any(~fixed_mask):
        s2 = s[~fixed_mask]
        e2 = e[~fixed_mask]
        s2_eff = np.maximum(s2, float(t))
        a2 = a[~fixed_mask]
        n2 = n[~fixed_mask]
        k2 = k[~fixed_mask]
        g2 = g[~fixed_mask]
        s2_spread = sread[~fixed_mask]

        p_t = float(p0_disc(t))
        p_ts_d = model.discount_bond_paths(t, s2_eff, x, p_t, lambda u: float(p0_disc(float(u))))
        p_te_d = model.discount_bond_paths(t, e2, x, p_t, lambda u: float(p0_disc(float(u))))

        # Dual-curve mapping: derive forwarding bonds from discounting bonds using
        # the deterministic t=0 basis ratio.
        bt = float(p0_fwd(t) / p0_disc(t))
        bs = np.array([float(p0_fwd(float(u)) / p0_disc(float(u))) for u in s2_eff], dtype=float)
        be = np.array([float(p0_fwd(float(u)) / p0_disc(float(u))) for u in e2], dtype=float)
        c = be / bs  # P_f(T0,T1) = c * P_d(T0,T1)

        # Rewrite g*L+s-K as an option on the forwarding bond.
        # Using L = (1/Pf(T0,T1)-1)/tau, the payoff becomes linear in 1/Pf(T0,T1).
        strike_adj = k2 - s2_spread
        kbar_d = (1.0 + ((strike_adj * a2) / np.clip(g2, 1.0e-18, None))) * c
        strike_bond = 1.0 / np.clip(kbar_d, 1.0e-18, None)

        h_s = np.asarray(model.H(s2_eff), dtype=float)
        h_e = np.asarray(model.H(e2), dtype=float)
        z_t = float(model.zeta(t))
        z_s = np.asarray(model.zeta(s2_eff), dtype=float)
        sigma = np.abs(h_e - h_s) * np.sqrt(np.clip(z_s - z_t, 0.0, None))

        fwd_bond = p_te_d / np.clip(p_ts_d, 1.0e-18, None)
        k_mat = strike_bond[:, None]
        sig_mat = sigma[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(np.clip(fwd_bond, 1.0e-18, None) / np.clip(k_mat, 1.0e-18, None)) + 0.5 * sig_mat * sig_mat) / np.clip(sig_mat, 1.0e-18, None)
        d2 = d1 - sig_mat

        call = p_te_d * _norm_cdf(d1) - k_mat * p_ts_d * _norm_cdf(d2)
        put = k_mat * p_ts_d * _norm_cdf(-d2) - p_te_d * _norm_cdf(-d1)

        # Deterministic sigma=0 fallback to intrinsic.
        tiny = sig_mat[:, 0] < 1.0e-14
        if np.any(tiny):
            call[tiny, :] = np.maximum(p_te_d[tiny, :] - k_mat[tiny, :] * p_ts_d[tiny, :], 0.0)
            put[tiny, :] = np.maximum(k_mat[tiny, :] * p_ts_d[tiny, :] - p_te_d[tiny, :], 0.0)

        scale = float(capfloor.position) * (n2 * g2 * kbar_d)[:, None]
        if capfloor.option_type.strip().lower() == "cap":
            pv_unfixed = np.sum(scale * put, axis=0)
        else:
            pv_unfixed = np.sum(scale * call, axis=0)

    return pv_fix + pv_unfixed


def capfloor_npv_paths(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    capfloor: CapFloorDef,
    times: Iterable[float],
    x_paths: np.ndarray,
    lock_fixings: bool = True,
    fixings: Mapping[tuple[str, str], float] | None = None,
    fixing_index: str | None = None,
) -> np.ndarray:
    """Pathwise cap/floor NPV profile on a simulation grid.

    If ``lock_fixings`` is enabled, once a fixing date lands on the simulation grid
    we freeze that coupon's realised forward for all later exposure dates, which is
    closer to how ORE exposure profiles treat already-fixed coupons.
    """
    t = _to_1d(np.asarray(list(times), dtype=float), "times")
    if x_paths.shape[0] != t.size:
        raise ValueError("x_paths first dimension must match times size")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")

    fixing = _fixing_times(capfloor)
    start = np.asarray(capfloor.start_time, dtype=float)
    end = np.asarray(capfloor.end_time, dtype=float)
    tau = np.asarray(capfloor.accrual, dtype=float)

    fix_to_idx: Dict[int, int] = {}
    if lock_fixings:
        for j, tf in enumerate(fixing):
            if tf <= 1.0e-12:
                continue
            k = int(np.searchsorted(t, tf, side="right"))
            fix_to_idx[j] = max(min(k, t.size - 1), 1)

    out = np.empty_like(x_paths)
    for i, ti in enumerate(t):
        live = np.asarray(capfloor.pay_time, dtype=float) >= ti - 1.0e-12
        rf_live = None
        if lock_fixings and np.any(live):
            idx_live = np.where(live)[0]
            rf_live = np.zeros((idx_live.size, x_paths.shape[1]), dtype=float)
            for k_local, j in enumerate(idx_live):
                tf = float(fixing[j])
                if tf > ti + 1.0e-12:
                    continue
                if fixings is not None and fixing_index is not None and capfloor.fixing_date is not None:
                    fixing_date = str(np.asarray(capfloor.fixing_date, dtype=object)[j])
                    fixing_key = (fixing_index.upper(), fixing_date)
                    if fixing_key in fixings:
                        rf_live[k_local, :] = float(fixings[fixing_key])
                        continue
                if tf <= 1.0e-12:
                    ps = float(p0_fwd(max(0.0, float(start[j]))))
                    pe = float(p0_fwd(float(end[j])))
                    rf_live[k_local, :] = (ps / pe - 1.0) / float(tau[j])
                    continue
                if tf >= float(end[j]) - 1.0e-12:
                    ps = float(p0_fwd(max(0.0, float(start[j]))))
                    pe = float(p0_fwd(float(end[j])))
                    rf_live[k_local, :] = (ps / pe - 1.0) / float(tau[j])
                    continue
                if j in fix_to_idx:
                    kf = fix_to_idx[j]
                    x_fix = interpolate_path_grid(t, x_paths, tf)[0, :]
                else:
                    x_fix = interpolate_path_grid(t, x_paths, tf)[0, :]
                fwd = forward_rate_from_bonds(
                    model,
                    p0_disc,
                    p0_fwd,
                    tf,
                    x_fix,
                    np.array([float(start[j])], dtype=float),
                    np.array([float(end[j])], dtype=float),
                    np.array([float(tau[j])], dtype=float),
                    fixing_time=tf,
                )[0, :]
                rf_live[k_local, :] = fwd
        out[i, :] = capfloor_npv(model, p0_disc, p0_fwd, capfloor, float(ti), x_paths[i, :], realized_forward=rf_live)
    return out


def bermudan_npv_paths(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    bermudan: BermudanSwaptionDef,
    times: Iterable[float],
    x_paths: np.ndarray,
    basis_degree: int = 2,
    itm_only: bool = True,
) -> np.ndarray:
    signed_swap = bermudan_signed_underlying_paths(
        model=model,
        p0_disc=p0_disc,
        p0_fwd=p0_fwd,
        bermudan=bermudan,
        times=times,
        x_paths=x_paths,
    )
    return bermudan_npv_paths_from_underlying(
        model=model,
        p0_disc=p0_disc,
        bermudan=bermudan,
        times=times,
        x_paths=x_paths,
        signed_swap=signed_swap,
        basis_degree=basis_degree,
        itm_only=itm_only,
    )


def bermudan_signed_underlying_paths(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    bermudan: BermudanSwaptionDef,
    times: Iterable[float],
    x_paths: np.ndarray,
 ) -> np.ndarray:
    t = _to_1d(np.asarray(list(times), dtype=float), "times")
    if x_paths.shape[0] != t.size:
        raise ValueError("x_paths first dimension must match times size")
    signed_swap = np.zeros_like(x_paths)
    for i in range(t.size):
        swap_npv = swap_npv_from_ore_legs_dual_curve(
            model,
            p0_disc,
            p0_fwd,
            bermudan.underlying_legs,
            float(t[i]),
            x_paths[i, :],
            exercise_into_whole_periods=True,
            deterministic_fixings_cutoff=0.0,
        )
        signed_swap[i, :] = float(bermudan.exercise_sign) * swap_npv
    return signed_swap


def bermudan_npv_paths_from_underlying(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    bermudan: BermudanSwaptionDef,
    times: Iterable[float],
    x_paths: np.ndarray,
    signed_swap: np.ndarray,
    basis_degree: int = 2,
    itm_only: bool = True,
) -> np.ndarray:
    """Bermudan swaption pathwise NPV profile via least-squares Monte Carlo.

    For physical settlement, this follows ORE-style wrapper behavior:
    - calibrate continuation with backward LSMC
    - decide exercise pathwise on exercise dates
    - once exercised, switch profile to underlying swap NPV on later dates

    The implementation is intentionally compact, but the structure mirrors the same
    conceptual decomposition ORE users expect for Bermudan exercise logic.
    """
    t = _to_1d(np.asarray(list(times), dtype=float), "times")
    if x_paths.shape[0] != t.size:
        raise ValueError("x_paths first dimension must match times size")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if basis_degree < 0:
        raise ValueError("basis_degree must be non-negative")

    ex = np.asarray(bermudan.exercise_times, dtype=float)
    ex = ex[ex >= 0.0]
    if ex.size == 0:
        return np.zeros_like(x_paths)
    ex_idx = np.searchsorted(t, ex)
    if np.any(ex_idx >= t.size):
        raise ValueError("exercise_times must be <= last simulation time")
    ex_flags = np.zeros(t.size, dtype=bool)
    ex_flags[ex_idx] = True
    signed_swap = np.asarray(signed_swap, dtype=float)
    if signed_swap.shape != x_paths.shape:
        raise ValueError("signed_swap must match x_paths shape")
    disc_curve_vals = np.asarray(curve_values(p0_disc, t), dtype=float)
    # Standard LSMC backward induction: regress discounted continuation on basis
    # functions of the current state x(t), then compare against immediate exercise.
    v = np.zeros_like(x_paths)
    v[-1, :] = np.maximum(signed_swap[-1, :], 0.0) if ex_flags[-1] else 0.0
    betas: dict[int, np.ndarray] = {}
    basis_cache: dict[tuple[int, tuple[int, ...], bytes], np.ndarray] = {}

    for i in range(t.size - 2, -1, -1):
        cont_realized = model.discount_bond(
            float(t[i]),
            float(t[i + 1]),
            x_paths[i, :],
            float(disc_curve_vals[i]),
            float(disc_curve_vals[i + 1]),
        ) * v[i + 1, :]
        if not ex_flags[i]:
            v[i, :] = cont_realized
            continue

        x_i = x_paths[i, :]
        y_i = cont_realized
        exer_raw = signed_swap[i, :]

        reg_mask = np.ones_like(x_i, dtype=bool)
        if itm_only:
            reg_mask = exer_raw > 1.0e-14
            if np.count_nonzero(reg_mask) < max(8, basis_degree + 2):
                reg_mask = np.ones_like(x_i, dtype=bool)

        basis_all = _basis_polynomial_cached(basis_cache, x_i, basis_degree)
        a = basis_all[reg_mask]
        b = y_i[reg_mask]
        try:
            beta, *_ = np.linalg.lstsq(a, b, rcond=None)
            betas[i] = beta
            cont_hat = basis_all @ beta
        except np.linalg.LinAlgError:
            cont_hat = np.full_like(y_i, float(np.mean(b)))
        cont_hat = np.maximum(cont_hat, 0.0)

        # Exercise policy: take the underlying if it dominates continuation value.
        # This is the core Bermudan decision rule also used conceptually in ORE.
        exercise_now = exer_raw > cont_hat
        v[i, :] = np.where(exercise_now, exer_raw, cont_realized)

    # For cash-settled options, keep the option-value surface (legacy behavior).
    if bermudan.settlement.strip().lower() == "cash":
        return np.maximum(v, 0.0)

    # Forward pass turns the regression surface into an actual exercised profile:
    # before exercise we keep option value, afterwards we hold the exercised swap.
    out = np.zeros_like(x_paths)
    exercised = np.zeros(x_paths.shape[1], dtype=bool)
    for i in range(t.size):
        active = ~exercised
        if np.any(active):
            out[i, active] = v[i, active]
        if np.any(exercised):
            out[i, exercised] = signed_swap[i, exercised]

        if not ex_flags[i]:
            continue
        active_idx = np.where(~exercised)[0]
        if active_idx.size == 0:
            continue

        x_i = x_paths[i, active_idx]
        if i in betas:
            cont_hat = _basis_polynomial_cached(basis_cache, x_i, basis_degree) @ betas[i]
        else:
            cont_hat = np.zeros_like(x_i)
        cont_hat = np.maximum(cont_hat, 0.0)
        exer_now_val = signed_swap[i, active_idx]
        do_ex = exer_now_val > cont_hat
        if np.any(do_ex):
            ex_idx_global = active_idx[do_ex]
            exercised[ex_idx_global] = True
            out[i, ex_idx_global] = exer_now_val[do_ex]
        if np.any(~do_ex):
            keep_idx_global = active_idx[~do_ex]
            out[i, keep_idx_global] = cont_hat[~do_ex]

    return out


def bermudan_lsmc_result(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    bermudan: BermudanSwaptionDef,
    times: Iterable[float],
    x_paths: np.ndarray,
    basis_degree: int = 2,
    itm_only: bool = True,
) -> BermudanLsmcResult:
    """Full Bermudan LSMC result including continuation diagnostics.

    This is a more implementation-facing variant of :func:`bermudan_npv_paths`
    used by higher-level benchmarking and parity tooling.
    """
    t = _to_1d(np.asarray(list(times), dtype=float), "times")
    if x_paths.shape[0] != t.size:
        raise ValueError("x_paths first dimension must match times size")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if basis_degree < 0:
        raise ValueError("basis_degree must be non-negative")

    ex = np.asarray(bermudan.exercise_times, dtype=float)
    ex = ex[ex >= 0.0]
    if ex.size == 0:
        return BermudanLsmcResult(
            npv_paths=np.zeros_like(x_paths),
            option_npv_paths=np.zeros_like(x_paths),
            signed_underlying_paths=np.zeros_like(x_paths),
            exercise_indices=np.full(x_paths.shape[1], -1, dtype=int),
            diagnostics=(),
        )
    ex_idx = np.searchsorted(t, ex)
    if np.any(ex_idx >= t.size):
        raise ValueError("exercise_times must be <= last simulation time")
    ex_flags = np.zeros(t.size, dtype=bool)
    ex_flags[ex_idx] = True
    disc_curve_vals = np.asarray(curve_values(p0_disc, t), dtype=float)

    signed_swap = np.zeros_like(x_paths)
    for i in range(t.size):
        swap_npv = swap_npv_from_ore_legs_dual_curve(
            model,
            p0_disc,
            p0_fwd,
            bermudan.underlying_legs,
            float(t[i]),
            x_paths[i, :],
            exercise_into_whole_periods=True,
            deterministic_fixings_cutoff=0.0,
        )
        signed_swap[i, :] = float(bermudan.exercise_sign) * swap_npv

    v = np.zeros_like(x_paths)
    v[-1, :] = np.maximum(signed_swap[-1, :], 0.0) if ex_flags[-1] else 0.0
    betas: dict[int, np.ndarray] = {}
    cont_hat_by_idx: dict[int, np.ndarray] = {}
    intrinsic_by_idx: dict[int, np.ndarray] = {}
    basis_cache: dict[tuple[int, tuple[int, ...], bytes], np.ndarray] = {}

    for i in range(t.size - 2, -1, -1):
        cont_realized = model.discount_bond(
            float(t[i]),
            float(t[i + 1]),
            x_paths[i, :],
            float(disc_curve_vals[i]),
            float(disc_curve_vals[i + 1]),
        ) * v[i + 1, :]
        if not ex_flags[i]:
            v[i, :] = cont_realized
            continue

        x_i = x_paths[i, :]
        y_i = cont_realized
        exer_raw = signed_swap[i, :]
        intrinsic = np.maximum(exer_raw, 0.0)

        reg_mask = np.ones_like(x_i, dtype=bool)
        if itm_only:
            reg_mask = intrinsic > 1.0e-14
            if np.count_nonzero(reg_mask) < max(8, basis_degree + 2):
                reg_mask = np.ones_like(x_i, dtype=bool)

        basis_all = _basis_polynomial_cached(basis_cache, x_i, basis_degree)
        a = basis_all[reg_mask]
        b = y_i[reg_mask]
        try:
            beta, *_ = np.linalg.lstsq(a, b, rcond=None)
            betas[i] = beta
            cont_hat = basis_all @ beta
        except np.linalg.LinAlgError:
            cont_hat = np.full_like(y_i, float(np.mean(b)))
        cont_hat = np.maximum(cont_hat, 0.0)

        exercise_now = exer_raw > cont_hat
        cont_hat_by_idx[i] = cont_hat
        intrinsic_by_idx[i] = intrinsic
        v[i, :] = np.where(exercise_now, exer_raw, cont_realized)

    if bermudan.settlement.strip().lower() == "cash":
        exercise_indices = np.full(x_paths.shape[1], -1, dtype=int)
        diagnostics = []
        for i in ex_idx:
            intrinsic = intrinsic_by_idx.get(int(i), np.maximum(signed_swap[i, :], 0.0))
            cont_hat = cont_hat_by_idx.get(int(i), np.zeros(x_paths.shape[1], dtype=float))
            ex_mask = intrinsic > cont_hat
            diagnostics.append(
                BermudanExerciseDiagnostic(
                    time=float(t[i]),
                    intrinsic_mean=float(np.mean(intrinsic)),
                    continuation_mean=float(np.mean(cont_hat)),
                    exercise_probability=float(np.mean(ex_mask)),
                    active_paths=int(x_paths.shape[1]),
                    boundary_state=_exercise_boundary_state(x_paths[i, :], ex_mask),
                )
            )
        return BermudanLsmcResult(
            npv_paths=np.maximum(v, 0.0),
            option_npv_paths=np.maximum(v, 0.0),
            signed_underlying_paths=signed_swap,
            exercise_indices=exercise_indices,
            diagnostics=tuple(diagnostics),
        )

    out = np.zeros_like(x_paths)
    exercised = np.zeros(x_paths.shape[1], dtype=bool)
    exercise_indices = np.full(x_paths.shape[1], -1, dtype=int)
    diagnostics = []
    for i in range(t.size):
        active = ~exercised
        if np.any(active):
            out[i, active] = v[i, active]
        if np.any(exercised):
            out[i, exercised] = signed_swap[i, exercised]

        if not ex_flags[i]:
            continue
        active_idx = np.where(~exercised)[0]
        if active_idx.size == 0:
            diagnostics.append(
                BermudanExerciseDiagnostic(
                    time=float(t[i]),
                    intrinsic_mean=0.0,
                    continuation_mean=0.0,
                    exercise_probability=0.0,
                    active_paths=0,
                    boundary_state=None,
                )
            )
            continue

        x_i = x_paths[i, active_idx]
        if i in betas:
            cont_hat = _basis_polynomial_cached(basis_cache, x_i, basis_degree) @ betas[i]
        else:
            cont_hat = np.zeros_like(x_i)
        cont_hat = np.maximum(cont_hat, 0.0)
        exer_now_val = signed_swap[i, active_idx]
        intrinsic = np.maximum(exer_now_val, 0.0)
        do_ex = exer_now_val > cont_hat
        diagnostics.append(
            BermudanExerciseDiagnostic(
                time=float(t[i]),
                intrinsic_mean=float(np.mean(intrinsic)),
                continuation_mean=float(np.mean(cont_hat)),
                exercise_probability=float(np.mean(do_ex)),
                active_paths=int(active_idx.size),
                boundary_state=_exercise_boundary_state(x_i, do_ex),
            )
        )
        if np.any(do_ex):
            ex_idx_global = active_idx[do_ex]
            exercised[ex_idx_global] = True
            exercise_indices[ex_idx_global] = i
            out[i, ex_idx_global] = exer_now_val[do_ex]
        if np.any(~do_ex):
            keep_idx_global = active_idx[~do_ex]
            out[i, keep_idx_global] = cont_hat[~do_ex]

    return BermudanLsmcResult(
        npv_paths=out,
        option_npv_paths=np.maximum(v, 0.0),
        signed_underlying_paths=signed_swap,
        exercise_indices=exercise_indices,
        diagnostics=tuple(diagnostics),
    )


def bermudan_price(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    bermudan: BermudanSwaptionDef,
    times: Iterable[float],
    x_paths: np.ndarray,
    basis_degree: int = 2,
    itm_only: bool = True,
) -> float:
    """Convenience scalar price from pathwise Bermudan values."""
    v = bermudan_npv_paths(model, p0_disc, p0_fwd, bermudan, times, x_paths, basis_degree=basis_degree, itm_only=itm_only)
    return float(np.mean(v[0, :]))


def bermudan_backward_price(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    bermudan: BermudanSwaptionDef,
    *,
    n_grid: int = 121,
    stddevs: float = 6.0,
    quadrature_order: int = 21,
    convolution_sx: float | None = None,
    convolution_nx: int | None = None,
    convolution_sy: float = 3.0,
    convolution_ny: int = 10,
) -> BermudanBackwardResult:
    """Deterministic backward induction on a one-dimensional LGM state grid."""
    if n_grid < 3 or n_grid % 2 == 0:
        raise ValueError("n_grid must be an odd integer >= 3")
    if quadrature_order < 3:
        raise ValueError("quadrature_order must be >= 3")
    if convolution_ny < 1:
        raise ValueError("convolution_ny must be >= 1")

    ex = np.asarray(bermudan.exercise_times, dtype=float)
    ex = ex[ex >= 0.0]
    if ex.size == 0:
        return BermudanBackwardResult(price=0.0, exercise_values=(), diagnostics=())
    times = np.concatenate(([0.0], ex))
    zeta = np.asarray(model.zeta(times), dtype=float)
    mx = (int(n_grid) - 1) // 2
    if convolution_nx is None:
        convolution_nx = max(1, int(round(mx / max(float(stddevs), 1.0e-12))))
    if convolution_sx is None:
        convolution_sx = mx / float(convolution_nx)
    y_nodes, y_weights = _convolution_nodes_and_weights(float(convolution_sy), int(convolution_ny))

    x_grids: list[np.ndarray] = [_convolution_state_grid(float(v), mx, int(convolution_nx)) for v in zeta]
    values: list[np.ndarray] = [np.zeros_like(g) for g in x_grids]

    diagnostics: list[BermudanExerciseDiagnostic] = []
    exercise_values: list[tuple[float, np.ndarray]] = []

    last = len(times) - 1
    last_grid = x_grids[last]
    intrinsic_last_pv = np.maximum(
        float(bermudan.exercise_sign)
        * swap_npv_from_ore_legs_dual_curve(
            model,
            p0_disc,
            p0_fwd,
            bermudan.underlying_legs,
            float(times[last]),
            last_grid,
            exercise_into_whole_periods=True,
            deterministic_fixings_cutoff=0.0,
        ),
        0.0,
    )
    num_last = np.asarray(model.numeraire_lgm(float(times[last]), last_grid, p0_disc), dtype=float)
    intrinsic_last = intrinsic_last_pv / num_last
    values[last] = intrinsic_last
    exercise_values.append((float(times[last]), intrinsic_last_pv.copy()))
    diagnostics.append(
        BermudanExerciseDiagnostic(
            time=float(times[last]),
            intrinsic_mean=float(np.mean(intrinsic_last_pv)),
            continuation_mean=0.0,
            exercise_probability=float(np.mean(intrinsic_last_pv > 0.0)),
            active_paths=int(last_grid.size),
            boundary_state=_exercise_boundary_state(last_grid, intrinsic_last_pv > 0.0),
        )
    )

    for i in range(len(times) - 2, -1, -1):
        t_i = float(times[i])
        t_n = float(times[i + 1])
        grid_i = x_grids[i]
        grid_n = x_grids[i + 1]
        next_v = values[i + 1]
        cont = _convolution_rollback(
            next_v,
            zeta_t1=float(zeta[i + 1]),
            zeta_t0=float(zeta[i]),
            mx=mx,
            nx=int(convolution_nx),
            y_nodes=y_nodes,
            y_weights=y_weights,
        )

        if i == 0:
            values[i] = cont
            continue

        intrinsic_pv = np.maximum(
            float(bermudan.exercise_sign)
            * swap_npv_from_ore_legs_dual_curve(
                model,
                p0_disc,
                p0_fwd,
                bermudan.underlying_legs,
                t_i,
                grid_i,
                exercise_into_whole_periods=True,
                deterministic_fixings_cutoff=0.0,
            ),
            0.0,
        )
        intrinsic = intrinsic_pv / np.asarray(model.numeraire_lgm(t_i, grid_i, p0_disc), dtype=float)
        values[i] = np.maximum(intrinsic, cont)
        exercise_values.append((t_i, intrinsic_pv.copy()))
        diagnostics.append(
            _backward_diagnostic(
                model,
                p0_disc,
                p0_fwd,
                bermudan,
                t_i,
                grid_i,
                cont,
                reduced_values=True,
            )
        )

    diagnostics = sorted(diagnostics, key=lambda d: d.time)
    exercise_values = sorted(exercise_values, key=lambda x: x[0])
    return BermudanBackwardResult(
        price=float(values[0][0]),
        exercise_values=tuple(exercise_values),
        diagnostics=tuple(diagnostics),
    )


def _backward_diagnostic(
    model: LGM1F,
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
    bermudan: BermudanSwaptionDef,
    t: float,
    grid: np.ndarray,
    cont: np.ndarray,
    reduced_values: bool = False,
) -> BermudanExerciseDiagnostic:
    x_eval = np.asarray(grid, dtype=float)
    probs = None
    intrinsic_pv = np.maximum(
        float(bermudan.exercise_sign)
        * swap_npv_from_ore_legs_dual_curve(
            model,
            p0_disc,
            p0_fwd,
            bermudan.underlying_legs,
            float(t),
            x_eval,
            exercise_into_whole_periods=True,
            deterministic_fixings_cutoff=0.0,
        ),
        0.0,
    )
    if reduced_values:
        intrinsic = intrinsic_pv / np.asarray(model.numeraire_lgm(float(t), x_eval, p0_disc), dtype=float)
    else:
        intrinsic = intrinsic_pv
    cont_eval = np.interp(x_eval, grid, cont, left=cont[0], right=cont[-1])
    ex_mask = intrinsic > cont_eval
    if probs is None:
        probs = _state_grid_probabilities(model, float(t), x_eval)
    return BermudanExerciseDiagnostic(
        time=float(t),
        intrinsic_mean=float(np.sum(probs * intrinsic_pv)),
        continuation_mean=float(np.sum(probs * cont_eval)),
        exercise_probability=float(np.sum(probs * ex_mask.astype(float))),
        active_paths=int(x_eval.size),
        boundary_state=_exercise_boundary_state(x_eval, ex_mask),
    )


def _convolution_nodes_and_weights(sy: float, ny: int) -> tuple[np.ndarray, np.ndarray]:
    h = 1.0 / float(ny)
    my = int(np.floor(float(sy) * float(ny) + 0.5))
    y = h * (np.arange(2 * my + 1, dtype=float) - my)
    n = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(y / np.sqrt(2.0)))
    g = np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)
    w = np.empty_like(y)
    for i in range(y.size):
        if i == 0 or i == y.size - 1:
            y0 = y[0]
            w[i] = (1.0 + y0 / h) * (0.5 * (1.0 + math.erf((y0 + h) / np.sqrt(2.0)))) - (y0 / h) * (
                0.5 * (1.0 + math.erf(y0 / np.sqrt(2.0)))
            ) + (math.exp(-0.5 * (y0 + h) * (y0 + h)) / np.sqrt(2.0 * np.pi) - math.exp(-0.5 * y0 * y0) / np.sqrt(2.0 * np.pi)) / h
        else:
            w[i] = (
                (1.0 + y[i] / h) * n[i + 1]
                - 2.0 * y[i] / h * n[i]
                - (1.0 - y[i] / h) * n[i - 1]
                + (g[i + 1] - 2.0 * g[i] + g[i - 1]) / h
            )
        if w[i] < 0.0 and w[i] > -1.0e-10:
            w[i] = 0.0
    return y, w


def _convolution_state_grid(zeta_t: float, mx: int, nx: int) -> np.ndarray:
    if mx < 0 or nx < 1:
        raise ValueError("invalid convolution grid parameters")
    if abs(zeta_t) <= 1.0e-18:
        return np.zeros(2 * mx + 1, dtype=float)
    dx = np.sqrt(max(float(zeta_t), 0.0)) / float(nx)
    return dx * (np.arange(2 * mx + 1, dtype=float) - mx)


def _convolution_rollback_python(
    values: np.ndarray,
    *,
    zeta_t1: float,
    zeta_t0: float,
    mx: int,
    nx: int,
    y_nodes: np.ndarray,
    y_weights: np.ndarray,
) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if abs(zeta_t1 - zeta_t0) <= 1.0e-18:
        return v.copy()
    sigma = np.sqrt(max(float(zeta_t1), 0.0))
    dx = sigma / float(nx)
    out = np.zeros(2 * mx + 1, dtype=float)
    if abs(zeta_t0) <= 1.0e-18:
        acc = 0.0
        for y_i, w_i in zip(y_nodes, y_weights):
            kp = y_i * sigma / dx + mx
            kk = int(np.floor(kp))
            alpha = kp - kk
            beta = 1.0 - alpha
            interp = v[0] if kk < 0 else (v[-1] if kk + 1 > 2 * mx else alpha * v[kk + 1] + beta * v[kk])
            acc += w_i * interp
        out.fill(acc)
        return out
    std = np.sqrt(max(float(zeta_t1 - zeta_t0), 0.0))
    dx2 = np.sqrt(max(float(zeta_t0), 0.0)) / float(nx)
    for k in range(2 * mx + 1):
        acc = 0.0
        for y_i, w_i in zip(y_nodes, y_weights):
            kp = (dx2 * (k - mx) + y_i * std) / dx + mx
            kk = int(np.floor(kp))
            alpha = kp - kk
            beta = 1.0 - alpha
            interp = v[0] if kk < 0 else (v[-1] if kk + 1 > 2 * mx else alpha * v[kk + 1] + beta * v[kk])
            acc += w_i * interp
        out[k] = acc
    return out


def _convolution_rollback_vectorized(
    values: np.ndarray,
    *,
    zeta_t1: float,
    zeta_t0: float,
    mx: int,
    nx: int,
    y_nodes: np.ndarray,
    y_weights: np.ndarray,
) -> np.ndarray:
    v = np.ascontiguousarray(np.asarray(values, dtype=float))
    if abs(zeta_t1 - zeta_t0) <= 1.0e-18:
        return v.copy()

    sigma = np.sqrt(max(float(zeta_t1), 0.0))
    dx = sigma / float(nx)
    last = 2 * mx

    if abs(zeta_t0) <= 1.0e-18:
        kp = y_nodes * sigma / dx + mx
        kk = np.floor(kp).astype(np.int64, copy=False)
        alpha = kp - kk
        beta = 1.0 - alpha
        left = np.clip(kk, 0, last)
        right = np.clip(kk + 1, 0, last)
        interp = alpha * v[right] + beta * v[left]
        acc = float(np.dot(y_weights, interp))
        out = np.empty(last + 1, dtype=float)
        out.fill(acc)
        return out

    std = np.sqrt(max(float(zeta_t1 - zeta_t0), 0.0))
    dx2 = np.sqrt(max(float(zeta_t0), 0.0)) / float(nx)
    k_grid = np.arange(last + 1, dtype=float) - mx
    kp = ((dx2 * k_grid)[:, None] + (y_nodes * std)[None, :]) / dx + mx
    kk = np.floor(kp).astype(np.int64, copy=False)
    alpha = kp - kk
    beta = 1.0 - alpha
    left = np.clip(kk, 0, last)
    right = np.clip(kk + 1, 0, last)
    interp = alpha * v[right] + beta * v[left]
    return interp @ y_weights


def _convolution_rollback(
    values: np.ndarray,
    *,
    zeta_t1: float,
    zeta_t0: float,
    mx: int,
    nx: int,
    y_nodes: np.ndarray,
    y_weights: np.ndarray,
) -> np.ndarray:
    return _convolution_rollback_vectorized(
        values,
        zeta_t1=zeta_t1,
        zeta_t0=zeta_t0,
        mx=mx,
        nx=nx,
        y_nodes=np.asarray(y_nodes, dtype=float),
        y_weights=np.asarray(y_weights, dtype=float),
    )


def _state_grid_probabilities(model: LGM1F, t: float, x_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x_grid, dtype=float)
    if x.size == 1:
        return np.array([1.0], dtype=float)
    z = max(float(model.zeta(t)), 0.0)
    if z <= 1.0e-18:
        probs = np.zeros_like(x)
        probs[np.argmin(np.abs(x))] = 1.0
        return probs
    sigma = np.sqrt(z)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    cdf = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(edges / (sigma * np.sqrt(2.0))))
    probs = np.diff(cdf)
    s = float(np.sum(probs))
    if s <= 0.0:
        return np.full(x.size, 1.0 / x.size, dtype=float)
    return probs / s


def _exercise_boundary_state(x: np.ndarray, exercise_mask: np.ndarray) -> float | None:
    x1 = np.asarray(x, dtype=float)
    mask = np.asarray(exercise_mask, dtype=bool)
    if x1.size == 0 or not np.any(mask) or np.all(mask):
        return None
    order = np.argsort(x1)
    xs = x1[order]
    ms = mask[order]
    switch = np.where(ms[:-1] != ms[1:])[0]
    if switch.size == 0:
        return None
    k = int(switch[-1])
    return float(0.5 * (xs[k] + xs[k + 1]))


__all__ = [
    "CapFloorDef",
    "BermudanSwaptionDef",
    "BermudanBackwardResult",
    "BermudanExerciseDiagnostic",
    "BermudanLsmcResult",
    "forward_rate_from_bonds",
    "capfloor_npv",
    "capfloor_npv_paths",
    "bermudan_backward_price",
    "bermudan_lsmc_result",
    "bermudan_npv_paths",
    "bermudan_price",
]
