from __future__ import annotations

import bisect
from math import exp, log
from typing import Callable


def build_log_linear_discount_interpolator(
    times: list[float],
    dfs: list[float],
    *,
    left_flat: bool = True,
    right_flat: bool = True,
) -> Callable[[float], float]:
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
