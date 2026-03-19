"""Torch-backed sibling of the NumPy LGM simulator.

This keeps the analytical model identical to ``py_ore_tools.lgm`` and moves only
the main state evolution onto torch tensors so kernels can be compared
side-by-side.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .lgm import LGM1F, _validate_time_grid

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
    times_arr = _validate_time_grid(times)
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
        zeta_grid = np.asarray(model.zeta(times_arr), dtype=float)
        var_increments = np.diff(zeta_grid)
        if np.any(var_increments < -1.0e-14):
            raise ValueError("encountered negative variance increment")
        step_scales = np.sqrt(np.maximum(var_increments, 0.0))
        step_scales_t = torch_mod.as_tensor(step_scales, dtype=dtype, device=device_obj)

        x = torch_mod.empty((times_arr.size, n_paths), dtype=dtype, device=device_obj)
        x[0].fill_(float(x0))

        if draw_order == "time_major":
            if normal_draws is None:
                draws = rng.standard_normal((step_scales.size, n_paths))
            else:
                draws = np.asarray(normal_draws, dtype=float)
                if draws.shape != (step_scales.size, n_paths):
                    raise ValueError("normal_draws must have shape (n_steps, n_paths) for time_major")
            draws_t = torch_mod.as_tensor(draws, dtype=dtype, device=device_obj)
            increments = step_scales_t[:, None] * draws_t
            x[1:] = float(x0) + torch_mod.cumsum(increments, dim=0)
            return x.cpu().numpy() if return_numpy else x

        if normal_draws is None:
            if not hasattr(rng, "next_sequence"):
                raise TypeError("draw_order='ore_path_major' requires an rng with a next_sequence(size) method")
            draws = np.vstack([np.asarray(rng.next_sequence(step_scales.size), dtype=float) for _ in range(n_paths)])
        else:
            draws = np.asarray(normal_draws, dtype=float)
            if draws.shape != (n_paths, step_scales.size):
                raise ValueError("normal_draws must have shape (n_paths, n_steps) for ore_path_major")
        draws_t = torch_mod.as_tensor(draws, dtype=dtype, device=device_obj)
        increments = draws_t * step_scales_t[None, :]
        x[1:] = float(x0) + torch_mod.cumsum(increments, dim=1).transpose(0, 1)
        return x.cpu().numpy() if return_numpy else x


__all__ = [
    "TorchLGM1F",
    "simulate_lgm_measure_torch",
]
