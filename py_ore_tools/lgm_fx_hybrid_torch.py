"""Torch-backed multi-currency LGM + FX hybrid simulation."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .lgm_fx_hybrid import LgmFxHybrid, _as_sorted_unique_times
from .lgm_torch import _require_torch


class TorchLgmFxHybrid(LgmFxHybrid):
    """Torch-flavoured hybrid model.

    Analytical identities stay in the NumPy base class; only path simulation is
    moved onto torch tensors.
    """


def simulate_hybrid_paths_torch(
    model: LgmFxHybrid,
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
) -> Dict[str, object]:
    torch = _require_torch()
    t = _as_sorted_unique_times(times, "times")
    if t[0] != 0.0:
        raise ValueError("times must start at 0.0")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if rng is None and normal_draws is None:
        rng = np.random.default_rng()

    x0 = {k.upper(): float(v) for k, v in (x0 or {}).items()}
    log_s0 = {k.upper().replace("-", "/"): float(v) for k, v in (log_s0 or {}).items()}
    rd_minus_rf = {k.upper().replace("-", "/"): float(v) for k, v in (rd_minus_rf or {}).items()}

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    if dtype is None:
        dtype = torch.float32 if device_obj.type == "mps" else torch.float64

    if normal_draws is None:
        draws = rng.standard_normal(size=(t.size - 1, model.n_factors, n_paths))
    else:
        draws = np.asarray(normal_draws, dtype=float)
        if draws.shape != (t.size - 1, model.n_factors, n_paths):
            raise ValueError("normal_draws must have shape (n_steps, n_factors, n_paths)")

    chol_t = torch.as_tensor(model._chol, dtype=dtype, device=device_obj)

    with torch.inference_mode():
        x_out: Dict[str, object] = {
            c: torch.zeros((t.size, n_paths), dtype=dtype, device=device_obj) for c in model.ir_ccys
        }
        ls_out: Dict[str, object] = {
            p: torch.zeros((t.size, n_paths), dtype=dtype, device=device_obj) for p in model.fx_pairs
        }

        for c in model.ir_ccys:
            x_out[c][0, :].fill_(x0.get(c, 0.0))
        for p in model.fx_pairs:
            ls_out[p][0, :].fill_(log_s0.get(p, 0.0))

        draws_t = torch.as_tensor(draws, dtype=dtype, device=device_obj)
        ir_count = len(model.ir_ccys)
        for i in range(t.size - 1):
            t0 = float(t[i])
            t1 = float(t[i + 1])
            dt = t1 - t0
            zc = chol_t @ draws_t[i]

            for j, c in enumerate(model.ir_ccys):
                m = model.ir_models[c]
                dz = float(m.zeta(t1) - m.zeta(t0))
                if dz < -1.0e-14:
                    raise ValueError(f"non-monotone zeta interval for {c} at step {i}")
                vol = np.sqrt(max(dz, 0.0))
                x_out[c][i + 1, :] = x_out[c][i, :] + vol * zc[j, :]

            for k, p in enumerate(model.fx_pairs):
                idx = ir_count + k
                sigma = float(model.fx_vol(p, np.array([0.5 * (t0 + t1)]))[0])
                mu = rd_minus_rf.get(p, 0.0)
                dlog = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * zc[idx, :]
                ls_out[p][i + 1, :] = ls_out[p][i, :] + dlog

        s_out = {p: torch.exp(ls_out[p]) for p in model.fx_pairs}

        if return_numpy:
            return {
                "times": t,
                "x": {k: v.detach().cpu().numpy() for k, v in x_out.items()},
                "log_s": {k: v.detach().cpu().numpy() for k, v in ls_out.items()},
                "s": {k: v.detach().cpu().numpy() for k, v in s_out.items()},
            }
        return {"times": t, "x": x_out, "log_s": ls_out, "s": s_out}


__all__ = [
    "TorchLgmFxHybrid",
    "simulate_hybrid_paths_torch",
]
