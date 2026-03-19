"""Multi-currency LGM + FX hybrid utilities anchored to ORE-style conventions.

ORE's cross-asset model couples one LGM factor per interest-rate currency with FX
factors.  This module provides a deliberately small Python analogue of that idea:

- each currency keeps the same 1F LGM state and bond formula as in ``lgm.py``
- FX pairs are simulated as correlated lognormal spots on the same path grid
- a user-supplied factor correlation matrix glues the IR and FX factors together

The purpose is not full ORE CAM parity; it is to give the surrounding exposure/XVA
utilities enough structure to price multi-currency cashflows using ORE-like inputs
and factor naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

try:
    from .lgm import LGM1F, LGMParams
except ImportError:  # pragma: no cover - script-mode fallback
    from lgm import LGM1F, LGMParams


def _as_sorted_unique_times(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _to_pair_key(pair: str) -> Tuple[str, str]:
    p = pair.strip().upper().replace("-", "/")
    if "/" not in p:
        raise ValueError(f"invalid FX pair '{pair}', expected BASE/QUOTE")
    base, quote = p.split("/", 1)
    if len(base) != 3 or len(quote) != 3:
        raise ValueError(f"invalid FX pair '{pair}', expected 3-letter currencies")
    return base, quote


@dataclass(frozen=True)
class MultiCcyLgmParams:
    """Container for per-ccy LGM and FX model inputs.

    `ir_params` maps currency code -> LGMParams.
    `fx_vols` maps FX pair BASE/QUOTE -> piecewise-constant vol term structure.
    `corr` is a full factor correlation matrix matching ordering:
      [IR(EUR), IR(USD), ..., FX(EUR/USD), FX(GBP/USD), ...]

    This mirrors how ORE users typically think about the cross-asset setup: one
    block of IR factors, one block of FX factors, then a correlation specification
    across the combined factor set.
    """

    ir_params: Mapping[str, LGMParams]
    fx_vols: Mapping[str, Tuple[Tuple[float, ...], Tuple[float, ...]]]
    corr: np.ndarray

    def __post_init__(self) -> None:
        if not self.ir_params:
            raise ValueError("ir_params must not be empty")
        for ccy in self.ir_params:
            if len(ccy) != 3:
                raise ValueError(f"invalid currency key '{ccy}'")
        for pair, tv in self.fx_vols.items():
            _ = _to_pair_key(pair)
            if len(tv) != 2:
                raise ValueError(f"fx_vols['{pair}'] must be (times, values)")
            times = np.asarray(tv[0], dtype=float)
            vols = np.asarray(tv[1], dtype=float)
            if times.ndim != 1 or vols.ndim != 1:
                raise ValueError(f"fx_vols['{pair}'] arrays must be one-dimensional")
            if vols.size != times.size + 1:
                raise ValueError(f"fx_vols['{pair}'] values size must equal times size + 1")
            if np.any(vols < 0.0):
                raise ValueError(f"fx_vols['{pair}'] contains negative vol")
            if times.size > 0 and (np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0)):
                raise ValueError(f"fx_vols['{pair}'] times must be strictly increasing and positive")


class LgmFxHybrid:
    """Correlated multi-ccy LGM + FX simulation helper.

    This is the bridge between single-currency ORE-style rate dynamics and
    multi-currency products.  Trade pricers elsewhere in the package treat it as a
    path generator plus a small library of pathwise pricing identities.
    """

    def __init__(self, params: MultiCcyLgmParams, zero_eig_clip: float = 1.0e-12) -> None:
        self.params = params
        self.zero_eig_clip = float(zero_eig_clip)
        if self.zero_eig_clip <= 0.0:
            raise ValueError("zero_eig_clip must be positive")

        self.ir_ccys = tuple(sorted(ccy.upper() for ccy in params.ir_params.keys()))
        self.ir_models: Dict[str, LGM1F] = {
            c: LGM1F(params.ir_params[c]) for c in self.ir_ccys
        }

        self.fx_pairs = tuple(sorted(p.upper().replace("-", "/") for p in params.fx_vols.keys()))
        self._fx_times: Dict[str, np.ndarray] = {}
        self._fx_vols: Dict[str, np.ndarray] = {}
        for p in self.fx_pairs:
            t, v = params.fx_vols[p]
            self._fx_times[p] = np.asarray(t, dtype=float)
            self._fx_vols[p] = np.asarray(v, dtype=float)

        # Keep an explicit factor ordering because the correlation matrix must match
        # this exact sequence, just as ORE CAM diagnostics refer to named factors.
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

    def _make_psd_cholesky(self, corr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Empirical / hand-authored correlation matrices are often only "nearly" PSD.
        # Clip tiny negative eigenvalues so path generation remains robust while
        # preserving the intended factor structure as closely as possible.
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

    def factor_ordering(self) -> Sequence[str]:
        return tuple(self.factor_labels)

    def simulate_paths(
        self,
        times: Sequence[float],
        n_paths: int,
        rng: np.random.Generator,
        x0: Mapping[str, float] | None = None,
        log_s0: Mapping[str, float] | None = None,
        rd_minus_rf: Mapping[str, float] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate correlated IR factors and FX log-spot states.

        Returns dict:
          - `times`: shape (n_times,)
          - `x`: dict cc y-> shape (n_times, n_paths)
          - `log_s`: dict pair-> shape (n_times, n_paths)
          - `s`: dict pair-> shape (n_times, n_paths)

        ORE linkage:
        - IR states use the exact LGM variance clock from the single-currency model
        - FX states are intentionally simpler than full ORE CAM drift handling, but
          are sufficient for exposure examples where deterministic carry
          ``rd_minus_rf`` is supplied externally
        """
        t = _as_sorted_unique_times(times, "times")
        if t[0] != 0.0:
            raise ValueError("times must start at 0.0")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        x0 = {k.upper(): float(v) for k, v in (x0 or {}).items()}
        log_s0 = {k.upper().replace("-", "/"): float(v) for k, v in (log_s0 or {}).items()}
        rd_minus_rf = {k.upper().replace("-", "/"): float(v) for k, v in (rd_minus_rf or {}).items()}

        x_out: Dict[str, np.ndarray] = {c: np.zeros((t.size, n_paths), dtype=float) for c in self.ir_ccys}
        ls_out: Dict[str, np.ndarray] = {p: np.zeros((t.size, n_paths), dtype=float) for p in self.fx_pairs}

        for c in self.ir_ccys:
            x_out[c][0, :] = x0.get(c, 0.0)
        for p in self.fx_pairs:
            ls_out[p][0, :] = log_s0.get(p, 0.0)

        ir_count = len(self.ir_ccys)
        for i in range(t.size - 1):
            t0 = t[i]
            t1 = t[i + 1]
            dt = t1 - t0

            # Draw all factor shocks together so the supplied CAM-style correlation
            # is respected pathwise across currencies and FX pairs.
            z = rng.standard_normal(size=(self.n_factors, n_paths))
            zc = self._chol @ z

            for j, c in enumerate(self.ir_ccys):
                m = self.ir_models[c]
                # In the LGM measure the increment variance is exactly Delta zeta.
                dz = float(m.zeta(t1) - m.zeta(t0))
                if dz < -1.0e-14:
                    raise ValueError(f"non-monotone zeta interval for {c} at step {i}")
                vol = np.sqrt(max(dz, 0.0))
                x_out[c][i + 1, :] = x_out[c][i, :] + vol * zc[j, :]

            for k, p in enumerate(self.fx_pairs):
                idx = ir_count + k
                # FX is evolved as a lognormal factor using the mid-interval vol
                # bucket.  That is a lightweight proxy for the richer ORE CAM
                # treatment, but it keeps all trades on one shared simulation grid.
                sigma = float(self.fx_vol(p, np.array([0.5 * (t0 + t1)]))[0])
                mu = rd_minus_rf.get(p, 0.0)
                dlog = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * zc[idx, :]
                ls_out[p][i + 1, :] = ls_out[p][i, :] + dlog

        s_out = {p: np.exp(ls_out[p]) for p in self.fx_pairs}
        return {"times": t, "x": x_out, "log_s": ls_out, "s": s_out}

    def zc_bond(
        self,
        ccy: str,
        t: float,
        T: float,
        x_t: np.ndarray,
        p0_t: float,
        p0_T: float,
    ) -> np.ndarray:
        """Pathwise zero-coupon bond in currency `ccy` using the underlying LGM1F.

        This is the same single-currency ORE/LGM bond identity, just routed through
        the currency-specific model chosen from the hybrid container.
        """
        c = ccy.upper()
        if c not in self.ir_models:
            raise ValueError(f"unknown currency '{ccy}'")
        return self.ir_models[c].discount_bond(t, T, x_t, p0_t, p0_T)

    def fx_forward(
        self,
        pair: str,
        t: float,
        T: float,
        s_t: np.ndarray,
        p_d_t_T: np.ndarray,
        p_f_t_T: np.ndarray,
    ) -> np.ndarray:
        """Pathwise FX forward F_t(T) = S_t * P_f(t,T)/P_d(t,T).

        The formula matches the usual covered-interest-parity relationship used by
        ORE when valuing FX forwards and FX-linked cashflows.
        """
        _ = _to_pair_key(pair)
        s = np.asarray(s_t, dtype=float)
        p_d = np.asarray(p_d_t_T, dtype=float)
        p_f = np.asarray(p_f_t_T, dtype=float)
        if s.shape != p_d.shape or s.shape != p_f.shape:
            raise ValueError("s_t, p_d_t_T, p_f_t_T must have identical shape")
        if np.any(p_d <= 0.0) or np.any(p_f <= 0.0):
            raise ValueError("bond prices must be positive")
        return s * p_f / p_d


__all__ = ["MultiCcyLgmParams", "LgmFxHybrid"]
