from __future__ import annotations

import bisect
from math import exp, log
from typing import Callable

import numpy as np


def _natural_cubic_spline_coefficients(x: list[float], y: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("at least three points are required for cubic interpolation")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.any(np.diff(x_arr) <= 0.0):
        raise ValueError("times must be strictly increasing")

    h = np.diff(x_arr)
    alpha = np.zeros_like(x_arr)
    for i in range(1, len(x_arr) - 1):
        alpha[i] = 3.0 * (y_arr[i + 1] - y_arr[i]) / h[i] - 3.0 * (y_arr[i] - y_arr[i - 1]) / h[i - 1]

    l = np.ones_like(x_arr)
    mu = np.zeros_like(x_arr)
    z = np.zeros_like(x_arr)
    for i in range(1, len(x_arr) - 1):
        l[i] = 2.0 * (x_arr[i + 1] - x_arr[i - 1]) - h[i - 1] * mu[i - 1]
        if abs(l[i]) <= 1.0e-14:
            l[i] = 1.0e-14
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    a = y_arr[:-1].copy()
    b = np.zeros(len(x_arr) - 1, dtype=float)
    c = np.zeros(len(x_arr), dtype=float)
    d = np.zeros(len(x_arr) - 1, dtype=float)

    for j in range(len(x_arr) - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_arr[j + 1] - y_arr[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    return a, b, c[:-1], d


def _evaluate_cubic_piecewise(
    x: list[float],
    y: list[float],
    coeffs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    point: float,
    *,
    left_flat: bool,
    right_flat: bool,
) -> float:
    a, b, c, d = coeffs
    xp = float(point)
    if xp <= x[0]:
        if left_flat:
            return float(y[0])
        idx = 0
    elif xp >= x[-1]:
        if right_flat:
            return float(y[-1])
        idx = len(x) - 2
    else:
        idx = bisect.bisect_right(x, xp) - 1
    s = xp - x[idx]
    return float(a[idx] + b[idx] * s + c[idx] * s * s + d[idx] * s * s * s)


def build_log_linear_discount_interpolator(
    times: list[float],
    dfs: list[float],
    *,
    left_flat: bool = True,
    right_flat: bool = False,
) -> Callable[[float], float]:
    """Build a log-linear discount interpolator.

    By default this follows the ORE discount-curve assumption on the right tail:
    it extrapolates using the last log-discount slope instead of flattening the
    discount factor at the final pillar.
    """
    if len(times) != len(dfs):
        raise ValueError("times and dfs must have the same length")
    if len(times) < 2:
        raise ValueError("at least two points are required")

    x = [float(t) for t in times]
    y = [float(df) for df in dfs]
    if any(df <= 0.0 for df in y):
        raise ValueError("discount factors must be strictly positive for log-linear interpolation")
    if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("times must be strictly increasing")

    log_y = [log(df) for df in y]

    def interpolate(t: float) -> float:
        point = float(t)
        if point <= x[0]:
            if left_flat:
                return y[0]
            i = 1
        elif point >= x[-1]:
            if right_flat:
                return y[-1]
            i = len(x) - 1
        else:
            i = bisect.bisect_right(x, point)

        x0, x1 = x[i - 1], x[i]
        y0, y1 = log_y[i - 1], log_y[i]
        w = (point - x0) / (x1 - x0)
        return exp((1.0 - w) * y0 + w * y1)

    return interpolate


def build_cubic_discount_interpolator(
    times: list[float],
    dfs: list[float],
    *,
    left_flat: bool = True,
    right_flat: bool = False,
) -> Callable[[float], float]:
    """Build a natural cubic discount interpolator in discount-factor space."""
    if len(times) != len(dfs):
        raise ValueError("times and dfs must have the same length")
    x = [float(t) for t in times]
    y = [float(df) for df in dfs]
    if any(df <= 0.0 for df in y):
        raise ValueError("discount factors must be strictly positive for cubic interpolation")
    coeffs = _natural_cubic_spline_coefficients(x, y)

    def interpolate(t: float) -> float:
        return _evaluate_cubic_piecewise(x, y, coeffs, float(t), left_flat=left_flat, right_flat=right_flat)

    return interpolate


def build_log_cubic_discount_interpolator(
    times: list[float],
    dfs: list[float],
    *,
    left_flat: bool = True,
    right_flat: bool = False,
) -> Callable[[float], float]:
    """Build a natural cubic interpolator in log-discount space."""
    if len(times) != len(dfs):
        raise ValueError("times and dfs must have the same length")
    x = [float(t) for t in times]
    y = [float(df) for df in dfs]
    if any(df <= 0.0 for df in y):
        raise ValueError("discount factors must be strictly positive for cubic interpolation")
    log_y = [log(df) for df in y]
    coeffs = _natural_cubic_spline_coefficients(x, log_y)

    def interpolate(t: float) -> float:
        return exp(_evaluate_cubic_piecewise(x, log_y, coeffs, float(t), left_flat=left_flat, right_flat=right_flat))

    return interpolate
