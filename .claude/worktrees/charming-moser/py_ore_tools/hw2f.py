"""Pure Python Hull-White n-factor kernel extracted from QuantExt formulas.

This module mirrors the small subset of the QuantExt HW implementation needed for
Python-side pricing and exposure diagnostics:

- piecewise-constant sigma_x(t) and kappa(t)
- deterministic ``y(t)`` and ``g(t,T)`` transforms
- discount-bond pricing
- BA-measure Euler simulation of the state and auxiliary bank-account integral

The implementation follows the formulas in:
- ``QuantExt/qle/models/hwpiecewiseparametrization.hpp``
- ``QuantExt/qle/models/hwmodel.cpp``
- ``QuantExt/qle/processes/irhwstateprocess.cpp``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import xml.etree.ElementTree as ET

import numpy as np


ArrayLike = np.ndarray | Sequence[float]


def _as_1d_float_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _piecewise_index(times: np.ndarray, t: float) -> int:
    if t < 0.0:
        raise ValueError("time input must be non-negative")
    return int(np.searchsorted(times, t, side="right"))


def _resolve_discount_scalar(value, t: float) -> float:
    if callable(value):
        out = float(value(float(t)))
    else:
        out = float(np.asarray(value, dtype=float))
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError("discount factor must be finite and positive")
    return out


def _resolve_discount_vector(value, t: np.ndarray) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    if callable(value):
        out = np.fromiter((float(value(float(x))) for x in t_arr), dtype=float, count=t_arr.size)
    else:
        out = np.asarray(value, dtype=float)
        if out.ndim == 0:
            out = np.full(t_arr.size, float(out), dtype=float)
        elif out.shape != t_arr.shape:
            raise ValueError("discount vector shape does not match maturity vector")
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("discount factor must be finite and positive")
    return out


@dataclass(frozen=True)
class HW2FParams:
    times: tuple[float, ...]
    sigma: tuple[tuple[tuple[float, ...], ...], ...]
    kappa: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        times = _as_1d_float_array(self.times, "times")
        if times.size and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")

        if len(self.sigma) == 0:
            raise ValueError("sigma must contain at least one matrix")
        if len(self.sigma) != len(self.kappa):
            raise ValueError("sigma and kappa must have the same number of buckets")
        if len(self.sigma) != len(self.times) + 1:
            raise ValueError("sigma/kappa bucket count must equal len(times) + 1")

        sigma_shapes = set()
        kappa_sizes = set()
        for i, matrix in enumerate(self.sigma):
            sigma_arr = np.asarray(matrix, dtype=float)
            if sigma_arr.ndim != 2:
                raise ValueError(f"sigma bucket {i} must be two-dimensional")
            if np.any(~np.isfinite(sigma_arr)):
                raise ValueError(f"sigma bucket {i} contains non-finite values")
            sigma_shapes.add(sigma_arr.shape)

        for i, bucket in enumerate(self.kappa):
            kappa_arr = np.asarray(bucket, dtype=float)
            if kappa_arr.ndim != 1:
                raise ValueError(f"kappa bucket {i} must be one-dimensional")
            if np.any(~np.isfinite(kappa_arr)):
                raise ValueError(f"kappa bucket {i} contains non-finite values")
            kappa_sizes.add(kappa_arr.size)

        if len(sigma_shapes) != 1:
            raise ValueError("all sigma matrices must have the same shape")
        if len(kappa_sizes) != 1:
            raise ValueError("all kappa vectors must have the same size")

        sigma_shape = next(iter(sigma_shapes))
        kappa_size = next(iter(kappa_sizes))
        if sigma_shape[1] != kappa_size:
            raise ValueError("sigma column count must match kappa size")


class HW2FModel:
    def __init__(self, params: HW2FParams, zero_kappa_cutoff: float = 1.0e-6) -> None:
        self.params = params
        self.times = _as_1d_float_array(params.times, "times")
        self.sigma_buckets = [np.asarray(m, dtype=float) for m in params.sigma]
        self.kappa_buckets = [np.asarray(v, dtype=float) for v in params.kappa]
        self.m = self.sigma_buckets[0].shape[0]
        self.n = self.sigma_buckets[0].shape[1]
        self.zero_kappa_cutoff = float(zero_kappa_cutoff)
        if self.n != 2:
            raise ValueError(f"HW2FModel expects exactly 2 factors; got {self.n}")

    def sigma_x(self, t: float) -> np.ndarray:
        return self.sigma_buckets[_piecewise_index(self.times, float(t))]

    def kappa(self, t: float) -> np.ndarray:
        return self.kappa_buckets[_piecewise_index(self.times, float(t))]

    def y(self, t: float) -> np.ndarray:
        t = float(t)
        if t < 0.0:
            raise ValueError("t must be non-negative")
        y = np.zeros((self.n, self.n), dtype=float)
        k0 = _piecewise_index(self.times, t)

        for k in range(k0):
            a = 0.0 if k == 0 else float(self.times[k - 1])
            b = float(self.times[k])
            y += self._y_block(a, b, t, self.kappa_buckets[k], self.sigma_buckets[k])

        a = 0.0 if k0 == 0 else float(self.times[k0 - 1])
        y += self._y_block(a, t, t, self.kappa_buckets[k0], self.sigma_buckets[k0])
        return y

    def g(self, t: float, T: float) -> np.ndarray:
        t = float(t)
        T = float(T)
        if not (0.0 <= t <= T):
            raise ValueError(f"expected 0 <= t <= T, got t={t}, T={T}")

        g = np.zeros(self.n, dtype=float)
        k0 = _piecewise_index(self.times, t)
        k1 = _piecewise_index(self.times, T)

        for k in range(k0, k1):
            a = t if k == k0 else float(self.times[k - 1])
            b = float(self.times[k])
            g += self._g_block(t, a, b, self.kappa_buckets[k])

        a = t if k1 == k0 else float(self.times[k1 - 1])
        g += self._g_block(t, a, T, self.kappa_buckets[k1])
        return g

    def discount_bond(self, t: float, T: float, x_t: ArrayLike, p0_t, p0_T) -> np.ndarray:
        if np.isclose(t, T):
            x = self._coerce_state(x_t)
            return np.ones(x.shape[0], dtype=float)
        if T < t or t < 0.0:
            raise ValueError("require T >= t >= 0")
        x = self._coerce_state(x_t)
        gt = self.g(float(t), float(T))
        yt = self.y(float(t))
        p_ratio = _resolve_discount_scalar(p0_T, float(T)) / _resolve_discount_scalar(p0_t, float(t))
        expo = -x @ gt - 0.5 * float(gt @ yt @ gt)
        return p_ratio * np.exp(expo)

    def discount_bond_paths(self, t: float, T: Iterable[float], x_t: ArrayLike, p0_t, p0_T) -> np.ndarray:
        T_arr = _as_1d_float_array(T, "T")
        if np.any(T_arr < t) or t < 0.0:
            raise ValueError("require T >= t >= 0")
        x = self._coerce_state(x_t)
        if T_arr.size == 0:
            return np.empty((0, x.shape[0]), dtype=float)

        p_t = _resolve_discount_scalar(p0_t, float(t))
        p_T = _resolve_discount_vector(p0_T, T_arr)
        out = np.empty((T_arr.size, x.shape[0]), dtype=float)
        for i, maturity in enumerate(T_arr):
            if np.isclose(float(maturity), float(t)):
                out[i, :] = 1.0
                continue
            gt = self.g(float(t), float(maturity))
            yt = self.y(float(t))
            expo = -x @ gt - 0.5 * float(gt @ yt @ gt)
            out[i, :] = (p_T[i] / p_t) * np.exp(expo)
        return out

    def numeraire_ba(self, t: float, aux_t: ArrayLike, p0_t) -> np.ndarray:
        aux = self._coerce_state(aux_t)
        return np.exp(np.sum(aux, axis=1)) / _resolve_discount_scalar(p0_t, float(t))

    def short_rate(self, t: float, x_t: ArrayLike, forward_rate_t0: float) -> np.ndarray:
        x = self._coerce_state(x_t)
        return np.sum(x, axis=1) + float(forward_rate_t0)

    def _coerce_state(self, x_t: ArrayLike) -> np.ndarray:
        x = np.asarray(x_t, dtype=float)
        if x.ndim == 1:
            if x.size != self.n:
                raise ValueError(f"state vector must have size {self.n}")
            x = x.reshape(1, self.n)
        elif x.ndim == 2:
            if x.shape[1] != self.n:
                raise ValueError(f"state matrix must have shape (n_paths, {self.n})")
        else:
            raise ValueError("state must be one- or two-dimensional")
        return x

    def _y_block(self, a: float, b: float, t: float, kappa: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                denom = float(kappa[i] + kappa[j])
                if abs(denom) < self.zero_kappa_cutoff:
                    tmp = b - a
                else:
                    tmp = (np.exp(-denom * (t - b)) - np.exp(-denom * (t - a))) / denom
                out[i, j] = float(np.dot(sigma[:, i], sigma[:, j]) * tmp)
        return out

    def _g_block(self, t: float, a: float, b: float, kappa: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            if abs(float(kappa[i])) < self.zero_kappa_cutoff:
                out[i] = b - a
            else:
                out[i] = (np.exp(-float(kappa[i]) * (a - t)) - np.exp(-float(kappa[i]) * (b - t))) / float(kappa[i])
        return out


def simulate_hw_ba_euler(
    model: HW2FModel,
    times: Iterable[float],
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    x0: Optional[ArrayLike] = None,
    aux0: Optional[ArrayLike] = None,
) -> tuple[np.ndarray, np.ndarray]:
    times_arr = _as_1d_float_array(times, "times")
    if times_arr.size == 0:
        raise ValueError("times must be non-empty")
    if np.any(times_arr < 0.0) or np.any(np.diff(times_arr) <= 0.0):
        raise ValueError("times must be non-negative and strictly increasing")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if rng is None:
        rng = np.random.default_rng()

    x_paths = np.zeros((times_arr.size, n_paths, model.n), dtype=float)
    aux_paths = np.zeros((times_arr.size, n_paths, model.n), dtype=float)

    if x0 is not None:
        x_paths[0, :, :] = model._coerce_state(x0)
    if aux0 is not None:
        aux_paths[0, :, :] = model._coerce_state(aux0)

    ones = np.ones(model.n, dtype=float)
    for i in range(times_arr.size - 1):
        t = float(times_arr[i])
        dt = float(times_arr[i + 1] - times_arr[i])
        x_curr = x_paths[i, :, :]
        aux_curr = aux_paths[i, :, :]
        sigma = model.sigma_x(t)
        kappa = model.kappa(t)
        y_t = model.y(t)
        drift_x = (y_t @ ones)[None, :] - x_curr * kappa[None, :]
        dW = rng.standard_normal((n_paths, model.m)) * np.sqrt(dt)
        x_paths[i + 1, :, :] = x_curr + drift_x * dt + dW @ sigma
        aux_paths[i + 1, :, :] = aux_curr + x_curr * dt

    return x_paths, aux_paths


def parse_hw2f_params_from_simulation_xml(simulation_xml: str | Path, key: str = "default") -> HW2FParams:
    root = ET.parse(simulation_xml).getroot()
    models = root.find("./CrossAssetModel/InterestRateModels")
    if models is None:
        raise ValueError("simulation xml missing CrossAssetModel/InterestRateModels")

    node = models.find(f"./HWModel[@key='{key}']")
    if node is None:
        node = models.find(f"./HWModel[@ccy='{key}']")
    if node is None:
        node = models.find("./HWModel[@key='default']")
    if node is None:
        node = models.find("./HWModel[@ccy='default']")
    if node is None:
        raise ValueError(f"no HWModel node found for key '{key}' or 'default'")

    vol_node = node.find("./Volatility")
    rev_node = node.find("./Reversion")
    if vol_node is None or rev_node is None:
        raise ValueError("HWModel node missing Volatility or Reversion block")

    times_text = (vol_node.findtext("./TimeGrid") or rev_node.findtext("./TimeGrid") or "").strip()
    times = tuple(float(x.strip()) for x in times_text.split(",") if x.strip())

    sigma_values: list[tuple[tuple[float, ...], ...]] = []
    initial_sigma = vol_node.find("./InitialValue")
    if initial_sigma is None:
        raise ValueError("Volatility block missing InitialValue")
    for sigma_node in initial_sigma.findall("./Sigma"):
        rows = []
        for row_node in sigma_node.findall("./Row"):
            rows.append(tuple(float(x.strip()) for x in (row_node.text or "").split(",") if x.strip()))
        sigma_values.append(tuple(rows))

    kappa_values: list[tuple[float, ...]] = []
    initial_kappa = rev_node.find("./InitialValue")
    if initial_kappa is None:
        raise ValueError("Reversion block missing InitialValue")
    for kappa_node in initial_kappa.findall("./Kappa"):
        kappa_values.append(tuple(float(x.strip()) for x in (kappa_node.text or "").split(",") if x.strip()))

    return HW2FParams(times=times, sigma=tuple(sigma_values), kappa=tuple(kappa_values))


__all__ = [
    "HW2FParams",
    "HW2FModel",
    "parse_hw2f_params_from_simulation_xml",
    "simulate_hw_ba_euler",
]
