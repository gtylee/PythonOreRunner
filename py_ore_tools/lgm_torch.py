"""Torch-backed sibling of the NumPy LGM simulator.

This keeps the analytical model identical to ``py_ore_tools.lgm`` and moves only
the main state evolution onto torch tensors so kernels can be compared
side-by-side.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .lgm import LGM1F, LGMParams, _as_1d_float_array, _validate_piecewise

try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None


class TorchLGM1F:
    """Torch-native 1-factor LGM representation."""

    def __init__(
        self,
        params: LGMParams,
        *,
        zero_cutoff: float = 1.0e-10,
        device: Optional[str] = None,
        dtype=None,
    ) -> None:
        torch_mod = _require_torch()
        self.params = params
        self.zero_cutoff = float(zero_cutoff)
        if self.zero_cutoff <= 0.0:
            raise ValueError("zero_cutoff must be positive")

        self.alpha_times = _as_1d_float_array(params.alpha_times, "alpha_times")
        self.alpha_values = _as_1d_float_array(params.alpha_values, "alpha_values")
        self.kappa_times = _as_1d_float_array(params.kappa_times, "kappa_times")
        self.kappa_values = _as_1d_float_array(params.kappa_values, "kappa_values")
        _validate_piecewise(self.alpha_times, self.alpha_values, "alpha")
        _validate_piecewise(self.kappa_times, self.kappa_values, "kappa")
        if np.any(self.alpha_values < 0.0):
            raise ValueError("alpha_values must be non-negative")

        self.shift = float(params.shift)
        self.scaling = float(params.scaling)
        if not np.isfinite(self.shift):
            raise ValueError("shift must be finite")
        if not np.isfinite(self.scaling) or self.scaling <= 0.0:
            raise ValueError("scaling must be finite and positive")

        device_obj = torch_mod.device(device) if device is not None else torch_mod.device("cpu")
        self.device = str(device_obj)
        self.dtype = dtype if dtype is not None else (torch_mod.float32 if device_obj.type == "mps" else torch_mod.float64)

        self._alpha_prefix_int = self._prefix_integral_square(self.alpha_times, self.alpha_values)
        self._kappa_prefix_int = self._prefix_integral_linear(self.kappa_times, self.kappa_values)
        self._h_prefix_int = self._build_h_prefix_integral()

    @staticmethod
    def _prefix_integral_square(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.zeros(times.size, dtype=float)
        running = 0.0
        for i in range(times.size):
            t0 = 0.0 if i == 0 else times[i - 1]
            running += values[i] * values[i] * (times[i] - t0)
            out[i] = running
        return out

    @staticmethod
    def _prefix_integral_linear(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.zeros(times.size, dtype=float)
        running = 0.0
        for i in range(times.size):
            t0 = 0.0 if i == 0 else times[i - 1]
            running += values[i] * (times[i] - t0)
            out[i] = running
        return out

    def _int_kappa_scalar(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        i = int(np.searchsorted(self.kappa_times, t, side="right"))
        res = 0.0
        if i >= 1:
            res += self._kappa_prefix_int[min(i - 1, self._kappa_prefix_int.size - 1)]
        k = self.kappa_values[min(i, self.kappa_values.size - 1)]
        t0 = 0.0 if i == 0 else self.kappa_times[i - 1]
        return res + k * (t - t0)

    def _build_h_prefix_integral(self) -> np.ndarray:
        out = np.zeros(self.kappa_times.size, dtype=float)
        running = 0.0
        for i in range(self.kappa_times.size):
            s = 0.0 if i == 0 else self.kappa_times[i - 1]
            e = self.kappa_times[i]
            k = self.kappa_values[i]
            base = np.exp(-self._int_kappa_scalar(s))
            if abs(k) < self.zero_cutoff:
                contrib = base * (e - s)
            else:
                contrib = base * (1.0 - np.exp(-k * (e - s))) / k
            running += contrib
            out[i] = running
        return out

    def _resolve_device_dtype(self, t):
        torch_mod = _require_torch()
        if isinstance(t, torch_mod.Tensor):
            return t.device, t.dtype
        return torch_mod.device(self.device), self.dtype

    def _as_time_tensor(self, t, *, device, dtype):
        torch_mod = _require_torch()
        t_t = torch_mod.as_tensor(t, dtype=dtype, device=device)
        if torch_mod.any(~torch_mod.isfinite(t_t)).item() or torch_mod.any(t_t < 0.0).item():
            raise ValueError("time input must be finite and non-negative")
        return t_t

    def _cached_coeffs(self, *, device, dtype):
        torch_mod = _require_torch()
        return {
            "alpha_times": torch_mod.as_tensor(self.alpha_times, dtype=dtype, device=device),
            "alpha_values": torch_mod.as_tensor(self.alpha_values, dtype=dtype, device=device),
            "alpha_prefix": torch_mod.as_tensor(self._alpha_prefix_int, dtype=dtype, device=device),
            "kappa_times": torch_mod.as_tensor(self.kappa_times, dtype=dtype, device=device),
            "kappa_values": torch_mod.as_tensor(self.kappa_values, dtype=dtype, device=device),
            "kappa_prefix": torch_mod.as_tensor(self._kappa_prefix_int, dtype=dtype, device=device),
            "h_prefix": torch_mod.as_tensor(self._h_prefix_int, dtype=dtype, device=device),
        }

    def _restore_time_output(self, ref, out_t):
        torch_mod = _require_torch()
        if isinstance(ref, torch_mod.Tensor):
            return out_t.reshape(ref.shape)
        arr = out_t.detach().cpu().numpy()
        if np.isscalar(ref):
            return float(arr.reshape(()))
        return arr

    def alpha(self, t):
        torch_mod = _require_torch()
        device, dtype = self._resolve_device_dtype(t)
        t_t = self._as_time_tensor(t, device=device, dtype=dtype)
        coeffs = self._cached_coeffs(device=device, dtype=dtype)
        idx = torch_mod.searchsorted(coeffs["alpha_times"], t_t.reshape(-1), right=True)
        out = coeffs["alpha_values"][idx] / self.scaling
        return self._restore_time_output(t, out)

    def kappa(self, t):
        torch_mod = _require_torch()
        device, dtype = self._resolve_device_dtype(t)
        t_t = self._as_time_tensor(t, device=device, dtype=dtype)
        coeffs = self._cached_coeffs(device=device, dtype=dtype)
        idx = torch_mod.searchsorted(coeffs["kappa_times"], t_t.reshape(-1), right=True)
        out = coeffs["kappa_values"][idx]
        return self._restore_time_output(t, out)

    def zeta(self, t):
        torch_mod = _require_torch()
        device, dtype = self._resolve_device_dtype(t)
        t_t = self._as_time_tensor(t, device=device, dtype=dtype)
        coeffs = self._cached_coeffs(device=device, dtype=dtype)
        flat = t_t.reshape(-1)
        idx = torch_mod.searchsorted(coeffs["alpha_times"], flat, right=True)
        out = torch_mod.zeros_like(flat)
        mask = idx > 0
        if mask.any().item():
            out[mask] = coeffs["alpha_prefix"][idx[mask] - 1]
        t0 = torch_mod.zeros_like(flat)
        if mask.any().item():
            t0[mask] = coeffs["alpha_times"][idx[mask] - 1]
        out = out + coeffs["alpha_values"][idx].square() * (flat - t0)
        out = out / (self.scaling * self.scaling)
        return self._restore_time_output(t, out)

    def Hprime(self, t):
        torch_mod = _require_torch()
        device, dtype = self._resolve_device_dtype(t)
        t_t = self._as_time_tensor(t, device=device, dtype=dtype)
        coeffs = self._cached_coeffs(device=device, dtype=dtype)
        flat = t_t.reshape(-1)
        idx = torch_mod.searchsorted(coeffs["kappa_times"], flat, right=True)
        delta = flat.clone()
        mask = idx > 0
        if mask.any().item():
            delta[mask] = delta[mask] - coeffs["kappa_times"][idx[mask] - 1]
        base = torch_mod.ones_like(flat)
        if mask.any().item():
            base[mask] = torch_mod.exp(-coeffs["kappa_prefix"][idx[mask] - 1])
        out = self.scaling * base * torch_mod.exp(-coeffs["kappa_values"][idx] * delta)
        return self._restore_time_output(t, out)

    def H(self, t):
        torch_mod = _require_torch()
        device, dtype = self._resolve_device_dtype(t)
        t_t = self._as_time_tensor(t, device=device, dtype=dtype)
        coeffs = self._cached_coeffs(device=device, dtype=dtype)
        flat = t_t.reshape(-1)
        idx = torch_mod.searchsorted(coeffs["kappa_times"], flat, right=True)
        out = torch_mod.zeros_like(flat)
        mask = idx > 0
        if mask.any().item():
            out[mask] = coeffs["h_prefix"][idx[mask] - 1]
        delta = flat.clone()
        if mask.any().item():
            delta[mask] = delta[mask] - coeffs["kappa_times"][idx[mask] - 1]
        base = torch_mod.ones_like(flat)
        if mask.any().item():
            base[mask] = torch_mod.exp(-coeffs["kappa_prefix"][idx[mask] - 1])
        k = coeffs["kappa_values"][idx]
        small = torch_mod.abs(k) < self.zero_cutoff
        tail = torch_mod.empty_like(flat)
        tail[small] = base[small] * delta[small]
        tail[~small] = base[~small] * (1.0 - torch_mod.exp(-k[~small] * delta[~small])) / k[~small]
        out = self.scaling * (out + tail) + self.shift
        return self._restore_time_output(t, out)

    def discount_bond(self, t: float, T: float, x_t, p0_t: float, p0_T: float):
        torch_mod = _require_torch()
        if T < t or t < 0.0:
            raise ValueError("require T >= t >= 0")
        x = torch_mod.as_tensor(x_t)
        if abs(T - t) <= 1.0e-14:
            out = torch_mod.ones_like(x)
            if isinstance(x_t, torch_mod.Tensor):
                return out
            arr = out.detach().cpu().numpy()
            if np.isscalar(x_t):
                return float(arr.reshape(()))
            return arr

        h_t = torch_mod.as_tensor(self.H(t), dtype=x.dtype, device=x.device)
        h_T = torch_mod.as_tensor(self.H(T), dtype=x.dtype, device=x.device)
        z_t = torch_mod.as_tensor(self.zeta(t), dtype=x.dtype, device=x.device)
        p_ratio = torch_mod.as_tensor(float(p0_T) / float(p0_t), dtype=x.dtype, device=x.device)
        out = p_ratio * torch_mod.exp(-(h_T - h_t) * x - 0.5 * (h_T * h_T - h_t * h_t) * z_t)
        if isinstance(x_t, torch_mod.Tensor):
            return out
        arr = out.detach().cpu().numpy()
        if np.isscalar(x_t):
            return float(arr.reshape(()))
        return arr


def _require_torch():
    if torch is None:
        raise ImportError("torch-backed LGM simulation requires the torch package")
    return torch


def _validate_time_grid_torch(times: Iterable[float], *, torch_mod, dtype, device):
    times_t = torch_mod.as_tensor(times, dtype=dtype, device=device)
    if times_t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if times_t.numel() == 0:
        raise ValueError("times must be non-empty")
    if torch_mod.any(~torch_mod.isfinite(times_t)).item():
        raise ValueError("times contains non-finite values")
    if torch_mod.any(times_t < 0.0).item():
        raise ValueError("times must be non-negative")
    if times_t.numel() > 1 and torch_mod.any(torch_mod.diff(times_t) <= 0.0).item():
        raise ValueError("times must be strictly increasing")
    return times_t


def _zeta_grid_torch(model: LGM1F, times_t, *, torch_mod):
    alpha_times_t = torch_mod.as_tensor(model.alpha_times, dtype=times_t.dtype, device=times_t.device)
    alpha_values_t = torch_mod.as_tensor(model.alpha_values, dtype=times_t.dtype, device=times_t.device)
    alpha_prefix_t = torch_mod.as_tensor(model._alpha_prefix_int, dtype=times_t.dtype, device=times_t.device)

    flat = times_t.reshape(-1)
    idx = torch_mod.searchsorted(alpha_times_t, flat, right=True)
    out = torch_mod.zeros_like(flat)
    mask = idx > 0
    if mask.any().item():
        out[mask] = alpha_prefix_t[idx[mask] - 1]
    t0 = torch_mod.zeros_like(flat)
    if mask.any().item():
        t0[mask] = alpha_times_t[idx[mask] - 1]
    a = alpha_values_t[idx]
    out = out + a.square() * (flat - t0)
    out = out / (model.scaling * model.scaling)
    return out.reshape_as(times_t)


def simulate_lgm_measure_torch(
    model: LGM1F,
    times: Iterable[float],
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    x0: float = 0.0,
    draw_order: str = "time_major",
    *,
    device: Optional[str] = None,
    dtype=None,
    normal_draws: Optional[np.ndarray] = None,
    return_numpy: bool = True,
):
    """Simulate the LGM state with torch as the accumulation backend.

    ``normal_draws`` is meant for deterministic NumPy-vs-torch comparisons. For
    ``time_major`` it must have shape ``(n_times - 1, n_paths)``. For
    ``ore_path_major`` it must have shape ``(n_paths, n_times - 1)``.
    """

    torch_mod = _require_torch()
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if draw_order not in ("time_major", "ore_path_major"):
        raise ValueError("draw_order must be 'time_major' or 'ore_path_major'")
    if rng is None and normal_draws is None and draw_order == "ore_path_major":
        raise ValueError("draw_order='ore_path_major' requires an explicit rng or normal_draws")
    if rng is None and normal_draws is None:
        rng = np.random.default_rng()

    device_obj = torch_mod.device(device) if device is not None else torch_mod.device("cpu")
    if dtype is None:
        dtype = torch_mod.float32 if device_obj.type == "mps" else torch_mod.float64

    with torch_mod.inference_mode():
        times_t = _validate_time_grid_torch(times, torch_mod=torch_mod, dtype=dtype, device=device_obj)
        zeta_grid_t = _zeta_grid_torch(model, times_t, torch_mod=torch_mod)
        var_increments_t = torch_mod.diff(zeta_grid_t)
        if torch_mod.any(var_increments_t < -1.0e-14).item():
            raise ValueError("encountered negative variance increment")
        step_scales_t = torch_mod.sqrt(torch_mod.clamp_min(var_increments_t, 0.0))

        x = torch_mod.empty((times_t.numel(), n_paths), dtype=dtype, device=device_obj)
        x[0].fill_(float(x0))

        if draw_order == "time_major":
            if normal_draws is None:
                draws = rng.standard_normal((step_scales_t.numel(), n_paths))
            else:
                draws = np.asarray(normal_draws, dtype=float)
                if draws.shape != (step_scales_t.numel(), n_paths):
                    raise ValueError("normal_draws must have shape (n_steps, n_paths) for time_major")
            draws_t = torch_mod.as_tensor(draws, dtype=dtype, device=device_obj)
            increments = step_scales_t[:, None] * draws_t
            x[1:] = float(x0) + torch_mod.cumsum(increments, dim=0)
            return x.cpu().numpy() if return_numpy else x

        if normal_draws is None:
            if not hasattr(rng, "next_sequence"):
                raise TypeError("draw_order='ore_path_major' requires an rng with a next_sequence(size) method")
            draws = np.vstack([np.asarray(rng.next_sequence(step_scales_t.numel()), dtype=float) for _ in range(n_paths)])
        else:
            draws = np.asarray(normal_draws, dtype=float)
            if draws.shape != (n_paths, step_scales_t.numel()):
                raise ValueError("normal_draws must have shape (n_paths, n_steps) for ore_path_major")
        draws_t = torch_mod.as_tensor(draws, dtype=dtype, device=device_obj)
        increments = draws_t * step_scales_t[None, :]
        x[1:] = float(x0) + torch_mod.cumsum(increments, dim=1).transpose(0, 1)
        return x.cpu().numpy() if return_numpy else x


__all__ = [
    "TorchLGM1F",
    "simulate_lgm_measure_torch",
]
