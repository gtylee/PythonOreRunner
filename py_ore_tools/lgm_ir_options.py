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
    from .irs_xva_utils import swap_npv_from_ore_legs_dual_curve
    from .lgm import LGM1F
except ImportError:  # pragma: no cover - script-mode fallback
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
    fixing_time: np.ndarray | None = None
    position: float = 1.0  # +1 long, -1 short

    def __post_init__(self) -> None:
        t0 = np.asarray(self.start_time, dtype=float)
        t1 = np.asarray(self.end_time, dtype=float)
        tp = np.asarray(self.pay_time, dtype=float)
        tau = np.asarray(self.accrual, dtype=float)
        n = np.asarray(self.notional, dtype=float)
        k = np.asarray(self.strike, dtype=float)
        if not (t0.ndim == t1.ndim == tp.ndim == tau.ndim == n.ndim == k.ndim == 1):
            raise ValueError("cap/floor arrays must be one-dimensional")
        m = t0.size
        if not (t1.size == tp.size == tau.size == n.size == k.size == m):
            raise ValueError("cap/floor arrays must have equal length")
        if m == 0:
            raise ValueError("cap/floor must contain at least one coupon")
        if np.any(t0 < 0.0) or np.any(t1 < t0) or np.any(tp < t1):
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

    live = pay > t + 1.0e-12
    if not np.any(live):
        return np.zeros_like(x)

    s = start[live]
    e = end[live]
    p = pay[live]
    a = tau[live]
    n = notional[live]
    k = strike[live]
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
        if capfloor.option_type.strip().lower() == "cap":
            payoff_fix = np.maximum(l[fixed_mask, :] - k[fixed_mask, None], 0.0)
        else:
            payoff_fix = np.maximum(k[fixed_mask, None] - l[fixed_mask, :], 0.0)
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
        a2 = a[~fixed_mask]
        n2 = n[~fixed_mask]
        k2 = k[~fixed_mask]

        p_t = float(p0_disc(t))
        p_ts_d = model.discount_bond_paths(t, s2, x, p_t, lambda u: float(p0_disc(float(u))))
        p_te_d = model.discount_bond_paths(t, e2, x, p_t, lambda u: float(p0_disc(float(u))))

        # Dual-curve mapping: derive forwarding bonds from discounting bonds using
        # the deterministic t=0 basis ratio.
        bt = float(p0_fwd(t) / p0_disc(t))
        bs = np.array([float(p0_fwd(float(u)) / p0_disc(float(u))) for u in s2], dtype=float)
        be = np.array([float(p0_fwd(float(u)) / p0_disc(float(u))) for u in e2], dtype=float)
        c = be / bs  # P_f(T0,T1) = c * P_d(T0,T1)

        # Rewrite the caplet/floorlet in terms of a bond option under LGM.  This is
        # the key step that turns the coupon option into a Gaussian closed form.
        kbar_d = (1.0 + k2 * a2) * c
        strike_bond = 1.0 / np.clip(kbar_d, 1.0e-18, None)

        h_s = np.asarray(model.H(s2), dtype=float)
        h_e = np.asarray(model.H(e2), dtype=float)
        z_t = float(model.zeta(t))
        z_s = np.asarray(model.zeta(s2), dtype=float)
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

        scale = float(capfloor.position) * (n2 * kbar_d)[:, None]
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
            k = int(np.searchsorted(t, tf))
            if k >= t.size or abs(float(t[k]) - float(tf)) > 1.0e-10:
                continue
            fix_to_idx[j] = k

    out = np.empty_like(x_paths)
    for i, ti in enumerate(t):
        live = np.asarray(capfloor.pay_time, dtype=float) > ti + 1.0e-12
        rf_live = None
        if lock_fixings and np.any(live):
            idx_live = np.where(live)[0]
            rf_live = np.zeros((idx_live.size, x_paths.shape[1]), dtype=float)
            for k_local, j in enumerate(idx_live):
                tf = float(fixing[j])
                if tf > ti + 1.0e-12:
                    continue
                if tf <= 1.0e-12:
                    ps = float(p0_fwd(max(0.0, float(start[j]))))
                    pe = float(p0_fwd(float(end[j])))
                    rf_live[k_local, :] = (ps / pe - 1.0) / float(tau[j])
                    continue
                if j not in fix_to_idx:
                    continue
                kf = fix_to_idx[j]
                x_fix = x_paths[kf, :]
                fwd = forward_rate_from_bonds(
                    model,
                    p0_disc,
                    p0_fwd,
                    tf,
                    x_fix,
                    np.array([float(start[j])], dtype=float),
                    np.array([float(end[j])], dtype=float),
                    np.array([float(tau[j])], dtype=float),
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
    ex_idx = np.searchsorted(t, ex)
    if np.any(ex_idx >= t.size):
        raise ValueError("exercise_times must be <= last simulation time")
    ex_flags = np.zeros(t.size, dtype=bool)
    ex_flags[ex_idx] = True

    # Keep signed underlying values on every date so the forward pass can switch from
    # option value to exercised swap value once exercise has occurred.
    signed_swap = np.zeros_like(x_paths)
    for i in range(t.size):
        swap_npv = swap_npv_from_ore_legs_dual_curve(
            model,
            p0_disc,
            p0_fwd,
            bermudan.underlying_legs,
            float(t[i]),
            x_paths[i, :],
        )
        signed_swap[i, :] = float(bermudan.exercise_sign) * swap_npv

    # Standard LSMC backward induction: regress discounted continuation on basis
    # functions of the current state x(t), then compare against immediate exercise.
    v = np.zeros_like(x_paths)
    v[-1, :] = np.maximum(signed_swap[-1, :], 0.0) if ex_flags[-1] else 0.0
    betas: dict[int, np.ndarray] = {}

    for i in range(t.size - 2, -1, -1):
        p_i = float(p0_disc(float(t[i])))
        cont_realized = model.discount_bond(float(t[i]), float(t[i + 1]), x_paths[i, :], p_i, float(p0_disc(float(t[i + 1])))) * v[i + 1, :]
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

        a = _basis_polynomial_1d(x_i[reg_mask], basis_degree)
        b = y_i[reg_mask]
        try:
            beta, *_ = np.linalg.lstsq(a, b, rcond=None)
            betas[i] = beta
            cont_hat = _basis_polynomial_1d(x_i, basis_degree) @ beta
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
            cont_hat = _basis_polynomial_1d(x_i, basis_degree) @ betas[i]
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


__all__ = [
    "CapFloorDef",
    "BermudanSwaptionDef",
    "forward_rate_from_bonds",
    "capfloor_npv",
    "capfloor_npv_paths",
    "bermudan_npv_paths",
    "bermudan_price",
]
