"""Torch-backed sibling of the NumPy LGM simulator.

This keeps the analytical model identical to ``py_ore_tools.lgm`` and moves only
the main state evolution onto torch tensors so kernels can be compared
side-by-side.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .lgm import LGM1F

try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None


class TorchLGM1F(LGM1F):
    """Torch-flavoured LGM model.

    The analytical helpers are inherited unchanged from ``LGM1F``.
    """


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
