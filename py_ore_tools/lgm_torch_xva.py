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


__all__ = [
    "TorchDiscountCurve",
    "discount_bond_paths_torch",
    "discount_bond_path_grid_torch",
    "numeraire_lgm_torch",
    "swap_npv_from_ore_legs_dual_curve_torch",
    "deflate_lgm_npv_paths_torch",
    "swap_npv_paths_from_ore_legs_dual_curve_torch",
    "deflate_lgm_npv_paths_torch_batched",
]
