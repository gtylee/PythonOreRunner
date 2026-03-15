"""Torch-backed multi-currency LGM + FX hybrid simulation."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .lgm_fx_hybrid import LgmFxHybrid, MultiCcyLgmParams, _to_pair_key
from .lgm_torch import TorchLGM1F, _require_torch


def _as_sorted_unique_times_torch(values: Sequence[float], *, torch_mod, dtype, device):
    times_t = torch_mod.as_tensor(values, dtype=dtype, device=device)
    if times_t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if times_t.numel() == 0:
        raise ValueError("times must be non-empty")
    if torch_mod.any(~torch_mod.isfinite(times_t)).item() or torch_mod.any(times_t < 0.0).item():
        raise ValueError("times must be finite and non-negative")
    if times_t.numel() > 1 and torch_mod.any(torch_mod.diff(times_t) <= 0.0).item():
        raise ValueError("times must be strictly increasing")
    return times_t


class TorchLgmFxHybrid(LgmFxHybrid):
    """Torch-native correlated multi-ccy LGM + FX helper."""

    def __init__(
        self,
        params: MultiCcyLgmParams,
        *,
        zero_eig_clip: float = 1.0e-12,
        device: Optional[str] = None,
        dtype=None,
    ) -> None:
        torch_mod = _require_torch()
        self.params = params
        self.zero_eig_clip = float(zero_eig_clip)
        if self.zero_eig_clip <= 0.0:
            raise ValueError("zero_eig_clip must be positive")

        device_obj = torch_mod.device(device) if device is not None else torch_mod.device("cpu")
        self.device = str(device_obj)
        self.dtype = dtype if dtype is not None else (torch_mod.float32 if device_obj.type == "mps" else torch_mod.float64)

        self.ir_ccys = tuple(sorted(ccy.upper() for ccy in params.ir_params.keys()))
        self.ir_models: Dict[str, TorchLGM1F] = {
            c: TorchLGM1F(params.ir_params[c], device=self.device, dtype=self.dtype) for c in self.ir_ccys
        }

        self.fx_pairs = tuple(sorted(p.upper().replace("-", "/") for p in params.fx_vols.keys()))
        self._fx_times: Dict[str, np.ndarray] = {}
        self._fx_vols: Dict[str, np.ndarray] = {}
        for p in self.fx_pairs:
            _ = _to_pair_key(p)
            t, v = params.fx_vols[p]
            self._fx_times[p] = np.asarray(t, dtype=float)
            self._fx_vols[p] = np.asarray(v, dtype=float)

        self.factor_labels = [f"IR:{c}" for c in self.ir_ccys] + [f"FX:{p}" for p in self.fx_pairs]
        self.n_factors = len(self.factor_labels)
        corr = np.asarray(params.corr, dtype=float)
        if corr.shape != (self.n_factors, self.n_factors):
            raise ValueError(
                f"corr shape {corr.shape} must match {(self.n_factors, self.n_factors)} "
                f"for factor ordering {self.factor_labels}"
            )
        if not np.allclose(corr, corr.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("corr must be symmetric")
        if np.max(np.abs(np.diag(corr) - 1.0)) > 1.0e-10:
            raise ValueError("corr must have unit diagonal")

        self.corr = corr
        self.corr_psd, self._chol = self._make_psd_cholesky(corr)

    def _make_psd_cholesky(self, corr: np.ndarray):
        eigvals, eigvecs = np.linalg.eigh(corr)
        clipped = np.maximum(eigvals, self.zero_eig_clip)
        corr_psd = eigvecs @ np.diag(clipped) @ eigvecs.T
        d = np.sqrt(np.clip(np.diag(corr_psd), self.zero_eig_clip, None))
        corr_psd = corr_psd / d[:, None] / d[None, :]
        corr_psd = 0.5 * (corr_psd + corr_psd.T)
        chol = np.linalg.cholesky(corr_psd)
        return corr_psd, chol

    @staticmethod
    def _piecewise_value(times: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(times, t, side="right")
        return values[idx]

    def fx_vol(self, pair: str, t: np.ndarray) -> np.ndarray:
        p = pair.upper().replace("-", "/")
        if p not in self._fx_times:
            raise ValueError(f"unknown FX pair '{pair}'")
        tt = np.asarray(t, dtype=float)
        if np.any(tt < 0.0):
            raise ValueError("time must be non-negative")
        return self._piecewise_value(self._fx_times[p], self._fx_vols[p], tt)

    def factor_ordering(self):
        return tuple(self.factor_labels)

    def zc_bond(
        self,
        ccy: str,
        t: float,
        T: float,
        x_t,
        p0_t: float,
        p0_T: float,
    ):
        c = ccy.upper()
        if c not in self.ir_models:
            raise ValueError(f"unknown currency '{ccy}'")
        return self.ir_models[c].discount_bond(t, T, x_t, p0_t, p0_T)

    def fx_forward(
        self,
        pair: str,
        t: float,
        T: float,
        s_t,
        p_d_t_T,
        p_f_t_T,
    ):
        torch_mod = _require_torch()
        _ = _to_pair_key(pair)
        s = torch_mod.as_tensor(s_t)
        p_d = torch_mod.as_tensor(p_d_t_T, dtype=s.dtype, device=s.device)
        p_f = torch_mod.as_tensor(p_f_t_T, dtype=s.dtype, device=s.device)
        if s.shape != p_d.shape or s.shape != p_f.shape:
            raise ValueError("s_t, p_d_t_T, p_f_t_T must have identical shape")
        if torch_mod.any(p_d <= 0.0).item() or torch_mod.any(p_f <= 0.0).item():
            raise ValueError("bond prices must be positive")
        out = s * p_f / p_d
        if isinstance(s_t, torch_mod.Tensor):
            return out
        arr = out.detach().cpu().numpy()
        if np.isscalar(s_t):
            return float(arr.reshape(()))
        return arr


def simulate_hybrid_paths_torch(
    model: TorchLgmFxHybrid,
    times: Sequence[float],
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    *,
    x0: Mapping[str, float] | None = None,
    log_s0: Mapping[str, float] | None = None,
    rd_minus_rf: Mapping[str, float] | None = None,
    device: Optional[str] = None,
    dtype=None,
    normal_draws: Optional[np.ndarray] = None,
    return_numpy: bool = True,
    antithetic: bool = False,
) -> Dict[str, object]:
    torch_mod = _require_torch()
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    device_obj = torch_mod.device(device) if device is not None else torch_mod.device(model.device)
    if dtype is None:
        dtype = torch_mod.float32 if device_obj.type == "mps" else model.dtype

    times_t = _as_sorted_unique_times_torch(times, torch_mod=torch_mod, dtype=dtype, device=device_obj)
    if abs(float(times_t[0].item())) > 1.0e-14:
        raise ValueError("times must start at 0.0")

    x0 = {k.upper(): float(v) for k, v in (x0 or {}).items()}
    log_s0 = {k.upper().replace("-", "/"): float(v) for k, v in (log_s0 or {}).items()}
    rd_minus_rf = {k.upper().replace("-", "/"): float(v) for k, v in (rd_minus_rf or {}).items()}
    if antithetic and normal_draws is not None:
        raise ValueError("antithetic=True cannot be combined with explicit normal_draws")

    if normal_draws is None:
        half_paths = (n_paths + 1) // 2 if antithetic else n_paths
        if rng is None:
            draws_t = torch_mod.randn((times_t.numel() - 1, model.n_factors, half_paths), dtype=dtype, device=device_obj)
            if antithetic:
                draws_t = torch_mod.cat((draws_t, -draws_t), dim=2)[:, :, :n_paths]
        else:
            draws = rng.standard_normal(size=(times_t.numel() - 1, model.n_factors, half_paths))
            if antithetic:
                draws = np.concatenate((draws, -draws), axis=2)[:, :, :n_paths]
            draws_t = torch_mod.as_tensor(draws, dtype=dtype, device=device_obj)
    else:
        draws = np.asarray(normal_draws, dtype=float)
        if draws.shape != (times_t.numel() - 1, model.n_factors, n_paths):
            raise ValueError("normal_draws must have shape (n_steps, n_factors, n_paths)")
        draws_t = torch_mod.as_tensor(draws, dtype=dtype, device=device_obj)

    with torch_mod.inference_mode():
        chol_t = torch_mod.as_tensor(model._chol, dtype=dtype, device=device_obj)
        dt_t = torch_mod.diff(times_t)
        mid_t = 0.5 * (times_t[:-1] + times_t[1:])

        x_out: Dict[str, object] = {
            c: torch_mod.zeros((times_t.numel(), n_paths), dtype=dtype, device=device_obj) for c in model.ir_ccys
        }
        ls_out: Dict[str, object] = {
            p: torch_mod.zeros((times_t.numel(), n_paths), dtype=dtype, device=device_obj) for p in model.fx_pairs
        }

        for c in model.ir_ccys:
            x_out[c][0, :].fill_(x0.get(c, 0.0))
        for p in model.fx_pairs:
            ls_out[p][0, :].fill_(log_s0.get(p, 0.0))

        ir_count = len(model.ir_ccys)
        fx_sigma_t = {
            p: torch_mod.as_tensor(model.fx_vol(p, mid_t.detach().cpu().numpy()), dtype=dtype, device=device_obj)
            for p in model.fx_pairs
        }
        ir_step_scale_t = {}
        for c in model.ir_ccys:
            zeta_grid = torch_mod.as_tensor(model.ir_models[c].zeta(times_t), dtype=dtype, device=device_obj)
            dz = torch_mod.diff(zeta_grid)
            if torch_mod.any(dz < -1.0e-14).item():
                raise ValueError(f"non-monotone zeta interval for {c}")
            ir_step_scale_t[c] = torch_mod.sqrt(torch_mod.clamp_min(dz, 0.0))

        for i in range(times_t.numel() - 1):
            zc = chol_t @ draws_t[i]

            for j, c in enumerate(model.ir_ccys):
                x_out[c][i + 1, :] = x_out[c][i, :] + ir_step_scale_t[c][i] * zc[j, :]

            for k, p in enumerate(model.fx_pairs):
                idx = ir_count + k
                sigma = fx_sigma_t[p][i]
                mu = rd_minus_rf.get(p, 0.0)
                dlog = (mu - 0.5 * sigma * sigma) * dt_t[i] + sigma * torch_mod.sqrt(dt_t[i]) * zc[idx, :]
                ls_out[p][i + 1, :] = ls_out[p][i, :] + dlog

        s_out = {p: torch_mod.exp(ls_out[p]) for p in model.fx_pairs}

        if return_numpy:
            return {
                "times": times_t.detach().cpu().numpy(),
                "x": {k: v.detach().cpu().numpy() for k, v in x_out.items()},
                "log_s": {k: v.detach().cpu().numpy() for k, v in ls_out.items()},
                "s": {k: v.detach().cpu().numpy() for k, v in s_out.items()},
            }
        return {"times": times_t, "x": x_out, "log_s": ls_out, "s": s_out}


__all__ = [
    "TorchLgmFxHybrid",
    "simulate_hybrid_paths_torch",
]
