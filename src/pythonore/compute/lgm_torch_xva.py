"""Torch-backed LGM swap/XVA helpers.

This module keeps the pathwise simulation and a representative downstream pricing
kernel on torch so CPU vs GPU comparisons include more of the real workflow than
state evolution alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .lgm import LGM1F
from .lgm_torch import _require_torch
from .lgm_ir_options import CapFloorDef, _fixing_times, forward_rate_from_bonds
from .irs_xva_utils import interpolate_path_grid


@dataclass
class TorchDiscountCurve:
    times: np.ndarray
    dfs: np.ndarray
    device: Optional[str] = None
    dtype: object = None

    def __post_init__(self) -> None:
        torch = _require_torch()
        times = np.asarray(self.times, dtype=float)
        dfs = np.asarray(self.dfs, dtype=float)
        if times.ndim != 1 or dfs.ndim != 1 or times.size != dfs.size:
            raise ValueError("times and dfs must be one-dimensional with matching size")
        if times.size < 2:
            raise ValueError("curve requires at least two points")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("curve times must be strictly increasing")
        if np.any(dfs <= 0.0):
            raise ValueError("curve discount factors must be positive")
        self.times = times
        self.dfs = dfs
        device_obj = torch.device(self.device) if self.device is not None else torch.device("cpu")
        self.device = str(device_obj)
        if self.dtype is None:
            self.dtype = torch.float32 if device_obj.type == "mps" else torch.float64
        self._times_t = torch.as_tensor(times, dtype=self.dtype, device=device_obj)
        self._dfs_t = torch.as_tensor(dfs, dtype=self.dtype, device=device_obj)

    @property
    def device_obj(self):
        torch = _require_torch()
        return torch.device(self.device)

    def discount(self, t):
        torch = _require_torch()
        t_t = torch.as_tensor(t, dtype=self.dtype, device=self.device_obj)
        if t_t.ndim == 0:
            return self._discount_scalar(t_t)
        flat = t_t.reshape(-1)
        out = torch.empty_like(flat)
        left = flat <= self._times_t[0]
        right = flat >= self._times_t[-1]
        interior = ~(left | right)
        out[left] = self._dfs_t[0]
        out[right] = self._dfs_t[-1]
        if torch.any(interior):
            x = flat[interior]
            idx = torch.searchsorted(self._times_t, x, right=False)
            idx = torch.clamp(idx, 1, self._times_t.numel() - 1)
            t0 = self._times_t[idx - 1]
            t1 = self._times_t[idx]
            y0 = self._dfs_t[idx - 1]
            y1 = self._dfs_t[idx]
            w = (x - t0) / torch.clamp(t1 - t0, min=torch.as_tensor(1.0e-12, dtype=self.dtype, device=self.device_obj))
            out[interior] = y0 + w * (y1 - y0)
        return out.reshape(t_t.shape)

    def _discount_scalar(self, t_t):
        torch = _require_torch()
        if t_t <= self._times_t[0]:
            return self._dfs_t[0]
        if t_t >= self._times_t[-1]:
            return self._dfs_t[-1]
        idx = int(torch.searchsorted(self._times_t, t_t, right=False).item())
        idx = max(1, min(idx, self._times_t.numel() - 1))
        t0 = self._times_t[idx - 1]
        t1 = self._times_t[idx]
        y0 = self._dfs_t[idx - 1]
        y1 = self._dfs_t[idx]
        w = (t_t - t0) / max(float((t1 - t0).item()), 1.0e-12)
        return y0 + w * (y1 - y0)


def _tensorize_legs(legs: Mapping[str, np.ndarray], *, device: str, dtype):
    torch = _require_torch()
    out = {}
    for k, v in legs.items():
        arr = np.asarray(v)
        if arr.dtype.kind in ("f", "i", "u"):
            out[k] = torch.as_tensor(arr, dtype=dtype, device=torch.device(device))
    return out


def discount_bond_paths_torch(
    model: LGM1F,
    t: float,
    maturities,
    x_t,
    p0_t,
    p0_T,
):
    torch = _require_torch()
    x = torch.as_tensor(x_t)
    T_np = np.asarray(maturities, dtype=float)
    h_t = float(model.H(t))
    z_t = float(model.zeta(t))
    h_T = torch.as_tensor(np.asarray(model.H(T_np), dtype=float), dtype=x.dtype, device=x.device)
    p0_T_t = torch.as_tensor(p0_T, dtype=x.dtype, device=x.device)
    p0_t_t = torch.as_tensor(p0_t, dtype=x.dtype, device=x.device)
    p_ratio = p0_T_t / p0_t_t
    d_h = h_T - h_t
    d_h2 = h_T * h_T - h_t * h_t
    return p_ratio[:, None] * torch.exp(-d_h[:, None] * x[None, :] - 0.5 * d_h2[:, None] * z_t)


def discount_bond_path_grid_torch(
    model: LGM1F,
    eval_times,
    maturities,
    x_paths,
    p0_t,
    p0_T,
):
    torch = _require_torch()
    x = torch.as_tensor(x_paths)
    t_np = np.asarray(eval_times, dtype=float)
    T_np = np.asarray(maturities, dtype=float)
    h_t = torch.as_tensor(np.asarray(model.H(t_np), dtype=float), dtype=x.dtype, device=x.device)
    z_t = torch.as_tensor(np.asarray(model.zeta(t_np), dtype=float), dtype=x.dtype, device=x.device)
    h_T = torch.as_tensor(np.asarray(model.H(T_np), dtype=float), dtype=x.dtype, device=x.device)
    p0_t_t = torch.as_tensor(p0_t, dtype=x.dtype, device=x.device)
    p0_T_t = torch.as_tensor(p0_T, dtype=x.dtype, device=x.device)
    d_h = h_T[None, :] - h_t[:, None]
    d_h2 = h_T[None, :] * h_T[None, :] - h_t[:, None] * h_t[:, None]
    return (p0_T_t[None, :] / p0_t_t[:, None])[:, :, None] * torch.exp(
        -d_h[:, :, None] * x[:, None, :] - 0.5 * d_h2[:, :, None] * z_t[:, None, None]
    )


def numeraire_lgm_torch(model: LGM1F, t: float, x_t, p0_t):
    torch = _require_torch()
    x = torch.as_tensor(x_t)
    h_t = float(model.H(t))
    z_t = float(model.zeta(t))
    p_t = torch.as_tensor(p0_t, dtype=x.dtype, device=x.device)
    return torch.exp(h_t * x + 0.5 * h_t * h_t * z_t) / p_t


def swap_npv_from_ore_legs_dual_curve_torch(
    model: LGM1F,
    disc_curve: TorchDiscountCurve,
    fwd_curve: TorchDiscountCurve,
    legs: Mapping[str, np.ndarray],
    t: float,
    x_t,
    *,
    realized_float_coupon: Optional[np.ndarray] = None,
    return_numpy: bool = True,
):
    torch = _require_torch()
    with torch.inference_mode():
        x = torch.as_tensor(x_t, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        legs_t = _tensorize_legs(legs, device=disc_curve.device, dtype=disc_curve.dtype)
        pv = torch.zeros_like(x)
        p_t_d = disc_curve.discount(float(t))
        p_t_f = fwd_curve.discount(float(t))

    mask_f = legs_t["fixed_pay_time"] > float(t) + 1.0e-12
    if torch.any(mask_f):
        pay = legs_t["fixed_pay_time"][mask_f]
        disc = discount_bond_paths_torch(
            model,
            float(t),
            pay.detach().cpu().numpy(),
            x,
            p_t_d,
            disc_curve.discount(pay).detach().cpu().numpy(),
        )
        cash = legs_t["fixed_amount"][mask_f]
        pv = pv + torch.sum(cash[:, None] * disc, dim=0)

    fix_t = legs_t.get("float_fixing_time", legs_t["float_start_time"])
    pay_all = legs_t["float_pay_time"]
    live = pay_all > float(t) + 1.0e-12
    if torch.any(live):
        s = legs_t["float_start_time"][live]
        e = legs_t["float_end_time"][live]
        pay = pay_all[live]
        tau = legs_t["float_accrual"][live]
        index_tau = legs_t.get("float_index_accrual", legs_t["float_accrual"])[live]
        n = legs_t["float_notional"][live]
        sign = legs_t["float_sign"][live]
        spread = legs_t["float_spread"][live]
        fixed = fix_t[live] <= float(t) + 1.0e-12

        p_tp_d = discount_bond_paths_torch(
            model,
            float(t),
            pay.detach().cpu().numpy(),
            x,
            p_t_d,
            disc_curve.discount(pay).detach().cpu().numpy(),
        )
        amount = torch.zeros((pay.numel(), x.numel()), dtype=x.dtype, device=x.device)

        if torch.any(fixed):
            if realized_float_coupon is not None:
                coupon_fix = torch.as_tensor(np.asarray(realized_float_coupon, dtype=float), dtype=x.dtype, device=x.device)[live][fixed]
            else:
                coupon_fix = legs_t["float_coupon"][live][fixed][:, None].expand(-1, x.numel())
            amount[fixed, :] = sign[fixed, None] * n[fixed, None] * coupon_fix * tau[fixed, None]

        if torch.any(~fixed):
            s2 = s[~fixed]
            e2 = e[~fixed]
            tau2 = tau[~fixed]
            index_tau2 = index_tau[~fixed]
            n2 = n[~fixed]
            sign2 = sign[~fixed]
            spread2 = spread[~fixed]
            p_ts_f2 = discount_bond_paths_torch(
                model,
                float(t),
                s2.detach().cpu().numpy(),
                x,
                p_t_f,
                fwd_curve.discount(s2).detach().cpu().numpy(),
            )
            p_te_f2 = discount_bond_paths_torch(
                model,
                float(t),
                e2.detach().cpu().numpy(),
                x,
                p_t_f,
                fwd_curve.discount(e2).detach().cpu().numpy(),
            )
            fwd2 = (p_ts_f2 / p_te_f2 - 1.0) / index_tau2[:, None]
            amount[~fixed, :] = sign2[:, None] * n2[:, None] * (fwd2 + spread2[:, None]) * tau2[:, None]

        pv = pv + torch.sum(amount * p_tp_d, dim=0)

        return pv.detach().cpu().numpy() if return_numpy else pv


def deflate_lgm_npv_paths_torch(
    model: LGM1F,
    disc_curve: TorchDiscountCurve,
    times: np.ndarray,
    x_paths,
    npv_paths,
    *,
    return_numpy: bool = True,
):
    torch = _require_torch()
    with torch.inference_mode():
        t = np.asarray(times, dtype=float)
        x = torch.as_tensor(x_paths, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        v = torch.as_tensor(npv_paths, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        out = torch.empty_like(v)
        for i, ti in enumerate(t):
            out[i, :] = v[i, :] / numeraire_lgm_torch(model, float(ti), x[i, :], disc_curve.discount(float(ti)))
        return out.detach().cpu().numpy() if return_numpy else out


def deflate_lgm_npv_paths_torch_batched(
    model: LGM1F,
    disc_curve: TorchDiscountCurve,
    times: np.ndarray,
    x_paths,
    npv_paths,
    *,
    return_numpy: bool = True,
):
    torch = _require_torch()
    with torch.inference_mode():
        t = np.asarray(times, dtype=float)
        x = torch.as_tensor(x_paths, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        v = torch.as_tensor(npv_paths, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        h_t = torch.as_tensor(np.asarray(model.H(t), dtype=float), dtype=x.dtype, device=x.device)
        z_t = torch.as_tensor(np.asarray(model.zeta(t), dtype=float), dtype=x.dtype, device=x.device)
        p_t = disc_curve.discount(torch.as_tensor(t, dtype=x.dtype, device=x.device))
        numeraires = torch.exp(h_t[:, None] * x + 0.5 * (h_t * h_t * z_t)[:, None]) / p_t[:, None]
        out = v / numeraires
        return out.detach().cpu().numpy() if return_numpy else out


def swap_npv_paths_from_ore_legs_dual_curve_torch(
    model: LGM1F,
    disc_curve: TorchDiscountCurve,
    fwd_curve: TorchDiscountCurve,
    legs: Mapping[str, np.ndarray],
    times: np.ndarray,
    x_paths,
    *,
    realized_float_coupon: Optional[np.ndarray] = None,
    return_numpy: bool = True,
):
    torch = _require_torch()
    with torch.inference_mode():
        eval_times_np = np.asarray(times, dtype=float)
        x = torch.as_tensor(x_paths, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        if x.ndim != 2 or x.shape[0] != eval_times_np.size:
            raise ValueError("x_paths must have shape [n_times, n_paths]")
        legs_t = _tensorize_legs(legs, device=disc_curve.device, dtype=disc_curve.dtype)
        eval_times = torch.as_tensor(eval_times_np, dtype=x.dtype, device=x.device)
        pv = torch.zeros_like(x)

    fixed_pay = legs_t["fixed_pay_time"]
    fixed_amount = legs_t["fixed_amount"]
    if fixed_pay.numel() > 0:
        fixed_disc = discount_bond_path_grid_torch(
            model,
            eval_times_np,
            fixed_pay.detach().cpu().numpy(),
            x,
            disc_curve.discount(eval_times),
            disc_curve.discount(fixed_pay),
        )
        fixed_live = fixed_pay[None, :] > eval_times[:, None] + 1.0e-12
        pv = pv + torch.sum(fixed_disc * fixed_amount[None, :, None] * fixed_live[:, :, None], dim=1)

    pay = legs_t["float_pay_time"]
    if pay.numel() > 0:
        start = legs_t["float_start_time"]
        end = legs_t["float_end_time"]
        fix_t = legs_t.get("float_fixing_time", start)
        tau = legs_t["float_accrual"]
        index_tau = legs_t.get("float_index_accrual", tau)
        notionals = legs_t["float_notional"]
        sign = legs_t["float_sign"]
        spread = legs_t["float_spread"]
        p_tp_d = discount_bond_path_grid_torch(
            model,
            eval_times_np,
            pay.detach().cpu().numpy(),
            x,
            disc_curve.discount(eval_times),
            disc_curve.discount(pay),
        )
        live = pay[None, :] > eval_times[:, None] + 1.0e-12
        fixed = fix_t[None, :] <= eval_times[:, None] + 1.0e-12
        amount = torch.zeros((eval_times.numel(), pay.numel(), x.shape[1]), dtype=x.dtype, device=x.device)

        if torch.any(live & fixed):
            if realized_float_coupon is not None:
                coupons = torch.as_tensor(np.asarray(realized_float_coupon, dtype=float), dtype=x.dtype, device=x.device)
            else:
                coupons = legs_t["float_coupon"][:, None].expand(-1, x.shape[1])
            fixed_amounts = sign[:, None] * notionals[:, None] * coupons * tau[:, None]
            amount = amount + fixed_amounts[None, :, :] * (live & fixed)[:, :, None]

        unfixed = live & ~fixed
        if torch.any(unfixed):
            p_ts_f = discount_bond_path_grid_torch(
                model,
                eval_times_np,
                start.detach().cpu().numpy(),
                x,
                fwd_curve.discount(eval_times),
                fwd_curve.discount(start),
            )
            p_te_f = discount_bond_path_grid_torch(
                model,
                eval_times_np,
                end.detach().cpu().numpy(),
                x,
                fwd_curve.discount(eval_times),
                fwd_curve.discount(end),
            )
            fwd = (p_ts_f / p_te_f - 1.0) / index_tau[None, :, None]
            float_amounts = sign[None, :, None] * notionals[None, :, None] * (fwd + spread[None, :, None]) * tau[None, :, None]
            amount = amount + float_amounts * unfixed[:, :, None]

        pv = pv + torch.sum(amount * p_tp_d * live[:, :, None], dim=1)

        return pv.detach().cpu().numpy() if return_numpy else pv


def price_plain_rate_leg_paths_torch(
    model: LGM1F,
    disc_curve: TorchDiscountCurve,
    leg: Mapping[str, np.ndarray | float | str],
    times: np.ndarray,
    x_paths,
    *,
    fwd_curve: Optional[TorchDiscountCurve] = None,
    return_numpy: bool = True,
):
    torch = _require_torch()
    kind = str(leg.get("kind", "")).upper()
    if kind not in {"FIXED", "FLOATING"}:
        raise ValueError(f"unsupported plain rate leg kind '{kind}'")
    with torch.inference_mode():
        eval_times_np = np.asarray(times, dtype=float)
        x = torch.as_tensor(x_paths, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        if x.ndim != 2 or x.shape[0] != eval_times_np.size:
            raise ValueError("x_paths must have shape [n_times, n_paths]")
        eval_times = torch.as_tensor(eval_times_np, dtype=x.dtype, device=x.device)
        pay_np = np.asarray(leg.get("pay_time", []), dtype=float)
        if pay_np.size == 0:
            out = torch.zeros_like(x)
            return out.detach().cpu().numpy() if return_numpy else out
        pay = torch.as_tensor(pay_np, dtype=x.dtype, device=x.device)
        live = (pay[None, :] >= 0.0) & (pay[None, :] > eval_times[:, None] + 1.0e-12)
        p_tp_d = discount_bond_path_grid_torch(
            model,
            eval_times_np,
            pay_np,
            x,
            disc_curve.discount(eval_times),
            disc_curve.discount(pay),
        )

        if kind == "FIXED":
            amount = torch.as_tensor(np.asarray(leg.get("amount", np.zeros(pay_np.shape)), dtype=float), dtype=x.dtype, device=x.device)
            pv = torch.sum(p_tp_d * amount[None, :, None] * live[:, :, None], dim=1)
            return pv.detach().cpu().numpy() if return_numpy else pv

        if fwd_curve is None:
            raise ValueError("floating rate leg requires fwd_curve")

        start_np = np.asarray(leg.get("start_time", []), dtype=float)
        end_np = np.asarray(leg.get("end_time", []), dtype=float)
        fixing_np = np.asarray(leg.get("fixing_time", start_np), dtype=float)
        quoted_np = np.asarray(leg.get("quoted_coupon", np.zeros(start_np.shape)), dtype=float)
        fixed_mask_np = np.asarray(
            leg.get("is_historically_fixed", np.zeros(start_np.shape, dtype=bool)),
            dtype=bool,
        )
        spread_np = np.asarray(leg.get("spread", np.zeros(start_np.shape)), dtype=float)
        gearing_np = np.asarray(leg.get("gearing", np.ones(start_np.shape)), dtype=float)
        accr_np = np.asarray(leg.get("accrual", np.zeros(start_np.shape)), dtype=float)
        index_accr_np = np.asarray(leg.get("index_accrual", accr_np), dtype=float)
        start = torch.as_tensor(start_np, dtype=x.dtype, device=x.device)
        end = torch.as_tensor(end_np, dtype=x.dtype, device=x.device)
        fixing = torch.as_tensor(fixing_np, dtype=x.dtype, device=x.device)
        quoted = torch.as_tensor(quoted_np, dtype=x.dtype, device=x.device)
        fixed_mask = torch.as_tensor(fixed_mask_np, dtype=torch.bool, device=x.device)
        spread = torch.as_tensor(spread_np, dtype=x.dtype, device=x.device)
        gearing = torch.as_tensor(gearing_np, dtype=x.dtype, device=x.device)
        accr = torch.as_tensor(accr_np, dtype=x.dtype, device=x.device)
        index_accr = torch.as_tensor(index_accr_np, dtype=x.dtype, device=x.device)

        p_t_f = fwd_curve.discount(eval_times)
        p_te_f = discount_bond_path_grid_torch(
            model,
            eval_times_np,
            end_np,
            x,
            p_t_f,
            fwd_curve.discount(end),
        )
        p_ts_f = discount_bond_path_grid_torch(
            model,
            eval_times_np,
            start_np,
            x,
            p_t_f,
            fwd_curve.discount(start),
        )
        after_start = eval_times[:, None] >= start[None, :] - 1.0e-12
        p_ts_f = torch.where(after_start[:, :, None], torch.ones_like(p_ts_f), p_ts_f)
        tau = torch.clamp(index_accr[None, :, None], min=torch.as_tensor(1.0e-8, dtype=x.dtype, device=x.device))
        floating_base = (p_ts_f / p_te_f - 1.0) / tau
        fixed_now = fixed_mask[None, :] | (fixing[None, :] <= eval_times[:, None] + 1.0e-12)
        base = torch.where(fixed_now[:, :, None], quoted[None, :, None], floating_base)
        coupon = gearing[None, :, None] * base + spread[None, :, None]
        notional = float(leg.get("notional", 0.0))
        sign = float(leg.get("sign", 1.0))
        amount = sign * notional * accr[None, :, None] * coupon
        pv = torch.sum(amount * p_tp_d * live[:, :, None], dim=1)
        return pv.detach().cpu().numpy() if return_numpy else pv


def par_swap_rate_paths_torch(
    model: LGM1F,
    curve: TorchDiscountCurve,
    t: float,
    x_t,
    start: float,
    tenor_years: float,
    *,
    return_numpy: bool = True,
):
    torch = _require_torch()
    with torch.inference_mode():
        x = torch.as_tensor(x_t, dtype=curve.dtype, device=curve.device_obj)
        effective_start = max(float(start), float(t))
        maturity = effective_start + max(float(tenor_years), 1.0e-8)
        pay_times = np.arange(effective_start + 1.0, maturity + 1.0e-10, 1.0)
        if pay_times.size == 0 or pay_times[-1] < maturity - 1.0e-10:
            pay_times = np.append(pay_times, maturity)
        p_t = curve.discount(float(t))
        p_start = discount_bond_paths_torch(
            model,
            float(t),
            np.asarray([effective_start], dtype=float),
            x,
            p_t,
            curve.discount(torch.as_tensor([effective_start], dtype=x.dtype, device=x.device)).detach().cpu().numpy(),
        )[0]
        p_end = discount_bond_paths_torch(
            model,
            float(t),
            np.asarray([maturity], dtype=float),
            x,
            p_t,
            curve.discount(torch.as_tensor([maturity], dtype=x.dtype, device=x.device)).detach().cpu().numpy(),
        )[0]
        annuity = torch.zeros_like(x)
        prev = effective_start
        for pay in pay_times:
            tau = max(float(pay) - prev, 1.0e-8)
            disc = discount_bond_paths_torch(
                model,
                float(t),
                np.asarray([float(pay)], dtype=float),
                x,
                p_t,
                curve.discount(torch.as_tensor([float(pay)], dtype=x.dtype, device=x.device)).detach().cpu().numpy(),
            )[0]
            annuity = annuity + float(tau) * disc
            prev = float(pay)
        eps = torch.as_tensor(1.0e-12, dtype=x.dtype, device=x.device)
        annuity = torch.where(torch.abs(annuity) < eps, eps, annuity)
        out = (p_start - p_end) / annuity
        return out.detach().cpu().numpy() if return_numpy else out


def _norm_cdf_torch(x):
    torch = _require_torch()
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def capfloor_npv_torch(
    model: LGM1F,
    disc_curve: TorchDiscountCurve,
    fwd_curve: TorchDiscountCurve,
    capfloor: CapFloorDef,
    t: float,
    x_t,
    *,
    realized_forward: Optional[np.ndarray] = None,
    return_numpy: bool = True,
):
    torch = _require_torch()
    with torch.inference_mode():
        x = torch.as_tensor(x_t, dtype=disc_curve.dtype, device=disc_curve.device_obj)
        if x.ndim != 1:
            raise ValueError("x_t must be one-dimensional")
        start = np.asarray(capfloor.start_time, dtype=float)
        end = np.asarray(capfloor.end_time, dtype=float)
        pay = np.asarray(capfloor.pay_time, dtype=float)
        tau = np.asarray(capfloor.accrual, dtype=float)
        notional = np.asarray(capfloor.notional, dtype=float)
        strike = np.asarray(capfloor.strike, dtype=float)
        fixing = _fixing_times(capfloor)
        live = pay > float(t) + 1.0e-12
        if not np.any(live):
            out = torch.zeros_like(x)
            return out.detach().cpu().numpy() if return_numpy else out
        s = start[live]
        e = end[live]
        p = pay[live]
        a = tau[live]
        n = notional[live]
        k = strike[live]
        g = np.asarray(capfloor.gearing if capfloor.gearing is not None else np.ones_like(strike), dtype=float)[live]
        spread = np.asarray(capfloor.spread if capfloor.spread is not None else np.zeros_like(strike), dtype=float)[live]
        f = fixing[live]
        fixed_mask = f <= float(t) + 1.0e-12
        option_is_cap = capfloor.option_type.strip().lower() == "cap"

        pv_fix = torch.zeros_like(x)
        if np.any(fixed_mask):
            if realized_forward is None:
                ps = fwd_curve.discount(torch.as_tensor(s[fixed_mask], dtype=x.dtype, device=x.device))
                pe = fwd_curve.discount(torch.as_tensor(e[fixed_mask], dtype=x.dtype, device=x.device))
                l_fix = ((ps / pe) - 1.0)[:, None] / torch.as_tensor(a[fixed_mask], dtype=x.dtype, device=x.device)[:, None]
                l_fix = l_fix.expand(-1, x.numel())
            else:
                rf = np.asarray(realized_forward, dtype=float)
                if rf.shape != (p.size, x.numel()):
                    raise ValueError("realized_forward shape must be (n_live_coupons, n_paths)")
                l_fix = torch.as_tensor(rf[fixed_mask, :], dtype=x.dtype, device=x.device)
            effective_rate = (
                torch.as_tensor(g[fixed_mask], dtype=x.dtype, device=x.device)[:, None] * l_fix
                + torch.as_tensor(spread[fixed_mask], dtype=x.dtype, device=x.device)[:, None]
            )
            k_fix = torch.as_tensor(k[fixed_mask], dtype=x.dtype, device=x.device)[:, None]
            payoff_fix = torch.maximum(effective_rate - k_fix, torch.zeros_like(effective_rate)) if option_is_cap else torch.maximum(k_fix - effective_rate, torch.zeros_like(effective_rate))
            amount_fix = (
                float(capfloor.position)
                * torch.as_tensor(n[fixed_mask], dtype=x.dtype, device=x.device)[:, None]
                * torch.as_tensor(a[fixed_mask], dtype=x.dtype, device=x.device)[:, None]
                * payoff_fix
            )
            p_t = disc_curve.discount(float(t))
            disc_fix = discount_bond_paths_torch(
                model,
                float(t),
                p[fixed_mask],
                x,
                p_t,
                disc_curve.discount(torch.as_tensor(p[fixed_mask], dtype=x.dtype, device=x.device)).detach().cpu().numpy(),
            )
            pv_fix = torch.sum(amount_fix * disc_fix, dim=0)

        pv_unfixed = torch.zeros_like(x)
        if np.any(~fixed_mask):
            s2 = s[~fixed_mask]
            e2 = e[~fixed_mask]
            p2 = p[~fixed_mask]
            a2 = a[~fixed_mask]
            n2 = n[~fixed_mask]
            k2 = k[~fixed_mask]
            g2 = g[~fixed_mask]
            spread2 = spread[~fixed_mask]

            p_t = disc_curve.discount(float(t))
            p_ts_d = discount_bond_paths_torch(
                model,
                float(t),
                s2,
                x,
                p_t,
                disc_curve.discount(torch.as_tensor(s2, dtype=x.dtype, device=x.device)).detach().cpu().numpy(),
            )
            p_te_d = discount_bond_paths_torch(
                model,
                float(t),
                e2,
                x,
                p_t,
                disc_curve.discount(torch.as_tensor(e2, dtype=x.dtype, device=x.device)).detach().cpu().numpy(),
            )
            bt = float(fwd_curve.discount(float(t)) / disc_curve.discount(float(t)))
            bs = (
                fwd_curve.discount(torch.as_tensor(s2, dtype=x.dtype, device=x.device))
                / disc_curve.discount(torch.as_tensor(s2, dtype=x.dtype, device=x.device))
            )
            be = (
                fwd_curve.discount(torch.as_tensor(e2, dtype=x.dtype, device=x.device))
                / disc_curve.discount(torch.as_tensor(e2, dtype=x.dtype, device=x.device))
            )
            c = be / bs
            strike_adj = torch.as_tensor(k2 - spread2, dtype=x.dtype, device=x.device)
            g2_t = torch.as_tensor(g2, dtype=x.dtype, device=x.device)
            a2_t = torch.as_tensor(a2, dtype=x.dtype, device=x.device)
            kbar_d = (1.0 + (strike_adj * a2_t) / torch.clamp(g2_t, min=torch.as_tensor(1.0e-18, dtype=x.dtype, device=x.device))) * c
            strike_bond = 1.0 / torch.clamp(kbar_d, min=torch.as_tensor(1.0e-18, dtype=x.dtype, device=x.device))
            h_s = torch.as_tensor(np.asarray(model.H(s2), dtype=float), dtype=x.dtype, device=x.device)
            h_e = torch.as_tensor(np.asarray(model.H(e2), dtype=float), dtype=x.dtype, device=x.device)
            z_t = float(model.zeta(float(t)))
            z_s = torch.as_tensor(np.asarray(model.zeta(s2), dtype=float), dtype=x.dtype, device=x.device)
            sigma = torch.abs(h_e - h_s) * torch.sqrt(torch.clamp(z_s - z_t, min=0.0))
            fwd_bond = p_te_d / torch.clamp(p_ts_d, min=torch.as_tensor(1.0e-18, dtype=x.dtype, device=x.device))
            k_mat = strike_bond[:, None]
            sig_mat = sigma[:, None]
            d1 = (
                torch.log(torch.clamp(fwd_bond, min=torch.as_tensor(1.0e-18, dtype=x.dtype, device=x.device)) / torch.clamp(k_mat, min=torch.as_tensor(1.0e-18, dtype=x.dtype, device=x.device)))
                + 0.5 * sig_mat * sig_mat
            ) / torch.clamp(sig_mat, min=torch.as_tensor(1.0e-18, dtype=x.dtype, device=x.device))
            d2 = d1 - sig_mat
            call = p_te_d * _norm_cdf_torch(d1) - k_mat * p_ts_d * _norm_cdf_torch(d2)
            put = k_mat * p_ts_d * _norm_cdf_torch(-d2) - p_te_d * _norm_cdf_torch(-d1)
            tiny = sigma < 1.0e-14
            if torch.any(tiny):
                idx = torch.nonzero(tiny, as_tuple=False).reshape(-1)
                call[idx, :] = torch.maximum(p_te_d[idx, :] - k_mat[idx, :] * p_ts_d[idx, :], torch.zeros_like(call[idx, :]))
                put[idx, :] = torch.maximum(k_mat[idx, :] * p_ts_d[idx, :] - p_te_d[idx, :], torch.zeros_like(put[idx, :]))
            scale = float(capfloor.position) * (
                torch.as_tensor(n2, dtype=x.dtype, device=x.device)
                * g2_t
                * kbar_d
            )[:, None]
            pv_unfixed = torch.sum(scale * (put if option_is_cap else call), dim=0)

        out = pv_fix + pv_unfixed
        return out.detach().cpu().numpy() if return_numpy else out


def capfloor_npv_paths_torch(
    model: LGM1F,
    disc_curve: TorchDiscountCurve,
    fwd_curve: TorchDiscountCurve,
    capfloor: CapFloorDef,
    times: np.ndarray,
    x_paths,
    *,
    lock_fixings: bool = True,
    return_numpy: bool = True,
):
    t = np.asarray(times, dtype=float)
    x = np.asarray(x_paths, dtype=float)
    if x.shape[0] != t.size:
        raise ValueError("x_paths first dimension must match times size")
    fixing = _fixing_times(capfloor)
    start = np.asarray(capfloor.start_time, dtype=float)
    end = np.asarray(capfloor.end_time, dtype=float)
    tau = np.asarray(capfloor.accrual, dtype=float)
    out = np.empty_like(x)
    fix_to_idx: dict[int, int] = {}
    if lock_fixings:
        for j, tf in enumerate(fixing):
            if tf <= 1.0e-12:
                continue
            k = int(np.searchsorted(t, tf, side="right"))
            fix_to_idx[j] = max(min(k, t.size - 1), 1)
    for i, ti in enumerate(t):
        live = np.asarray(capfloor.pay_time, dtype=float) > ti + 1.0e-12
        rf_live = None
        if lock_fixings and np.any(live):
            idx_live = np.where(live)[0]
            rf_live = np.zeros((idx_live.size, x.shape[1]), dtype=float)
            for k_local, j in enumerate(idx_live):
                tf = float(fixing[j])
                if tf > ti + 1.0e-12:
                    continue
                if tf <= 1.0e-12:
                    ps = float(fwd_curve.discount(max(0.0, float(start[j]))))
                    pe = float(fwd_curve.discount(float(end[j])))
                    rf_live[k_local, :] = (ps / pe - 1.0) / float(tau[j])
                    continue
                x_fix = interpolate_path_grid(t, x, tf)[0, :]
                fwd = forward_rate_from_bonds(
                    model,
                    lambda u: float(disc_curve.discount(float(u))),
                    lambda u: float(fwd_curve.discount(float(u))),
                    tf,
                    x_fix,
                    np.array([float(start[j])], dtype=float),
                    np.array([float(end[j])], dtype=float),
                    np.array([float(tau[j])], dtype=float),
                )[0, :]
                rf_live[k_local, :] = fwd
        out[i, :] = capfloor_npv_torch(
            model,
            disc_curve,
            fwd_curve,
            capfloor,
            float(ti),
            x[i, :],
            realized_forward=rf_live,
            return_numpy=True,
        )
    return out if return_numpy else _require_torch().as_tensor(out, dtype=disc_curve.dtype, device=disc_curve.device_obj)


__all__ = [
    "TorchDiscountCurve",
    "discount_bond_paths_torch",
    "discount_bond_path_grid_torch",
    "numeraire_lgm_torch",
    "swap_npv_from_ore_legs_dual_curve_torch",
    "deflate_lgm_npv_paths_torch",
    "swap_npv_paths_from_ore_legs_dual_curve_torch",
    "deflate_lgm_npv_paths_torch_batched",
    "price_plain_rate_leg_paths_torch",
    "par_swap_rate_paths_torch",
    "capfloor_npv_torch",
    "capfloor_npv_paths_torch",
]
