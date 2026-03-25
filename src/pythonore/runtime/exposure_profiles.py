from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

import numpy as np


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return date.fromisoformat(text)


def _weekends_only_adjust(value: date) -> date:
    if value.weekday() == 5:
        return value + timedelta(days=2)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def ore_pfe_quantile(samples: np.ndarray, quantile: float) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("samples must be 1D or 2D")
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    q = min(max(float(quantile), 0.0), 1.0)
    index = int(math.floor(q * (values.shape[1] - 1) + 0.5))
    kth = np.partition(values, index, axis=1)[:, index]
    return np.maximum(kth, 0.0)


def _mean_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.maximum(arr, 0.0)
    return np.mean(np.maximum(arr, 0.0), axis=1)


def _mean_negative(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.maximum(-arr, 0.0)
    return np.mean(np.maximum(-arr, 0.0), axis=1)


def ore_one_year_index(
    exposure_dates: Sequence[Any],
    exposure_times: Sequence[float],
    *,
    asof_date: Any | None = None,
) -> int:
    times = np.asarray(exposure_times, dtype=float)
    if times.size == 0:
        return 0
    if exposure_dates:
        asof = _coerce_date(asof_date) if asof_date is not None else _coerce_date(exposure_dates[0])
        cutoff = _weekends_only_adjust(asof + timedelta(days=369))
        idx = 0
        for i, d in enumerate(exposure_dates):
            if _coerce_date(d) <= cutoff:
                idx = i
            else:
                break
        return idx
    idx = int(np.searchsorted(times, 1.0, side="left"))
    return min(idx, times.size - 1)


def build_ore_basel_profile(
    exposure_dates: Sequence[Any],
    exposure_times: Sequence[float],
    closeout_epe: Sequence[float],
    discount_factors: Sequence[float] | None,
    *,
    asof_date: Any | None = None,
) -> dict[str, list[float] | float | int]:
    times = np.asarray(exposure_times, dtype=float)
    epe = np.asarray(closeout_epe, dtype=float)
    if times.shape != epe.shape:
        raise ValueError("exposure_times and closeout_epe must have the same shape")
    if discount_factors is None:
        dfs = np.ones_like(times)
    else:
        dfs = np.asarray(discount_factors, dtype=float)
        if dfs.shape != times.shape:
            raise ValueError("discount_factors must match exposure_times shape")
    safe_dfs = np.where(np.abs(dfs) > 1.0e-12, dfs, 1.0e-12)
    basel_ee = epe / safe_dfs
    basel_eee = np.maximum.accumulate(basel_ee)
    tw_epe = np.zeros_like(basel_ee)
    tw_eepe = np.zeros_like(basel_eee)
    acc_epe = 0.0
    acc_eepe = 0.0
    prev_t = 0.0
    for i, t in enumerate(times):
        dt = max(float(t) - prev_t, 0.0)
        prev_t = float(t)
        if i == 0 and float(t) == 0.0:
            tw_epe[i] = float(basel_ee[i])
            tw_eepe[i] = float(basel_eee[i])
            continue
        acc_epe += float(basel_ee[i]) * dt
        acc_eepe += float(basel_eee[i]) * dt
        denom = max(float(t), 1.0e-12)
        tw_epe[i] = acc_epe / denom
        tw_eepe[i] = acc_eepe / denom
    one_year_idx = ore_one_year_index(exposure_dates, exposure_times, asof_date=asof_date)
    return {
        "basel_ee": basel_ee.tolist(),
        "basel_eee": basel_eee.tolist(),
        "time_weighted_basel_epe": tw_epe.tolist(),
        "time_weighted_basel_eepe": tw_eepe.tolist(),
        "one_year_index": int(one_year_idx),
        "basel_epe": float(tw_epe[one_year_idx]) if tw_epe.size else 0.0,
        "basel_eepe": float(tw_eepe[one_year_idx]) if tw_eepe.size else 0.0,
    }


def build_ore_exposure_profile_from_paths(
    entity_id: str,
    exposure_dates: Sequence[Any],
    exposure_times: Sequence[float],
    valuation_paths: np.ndarray,
    closeout_paths: np.ndarray,
    *,
    discount_factors: Sequence[float] | None = None,
    closeout_times: Sequence[float] | None = None,
    expected_collateral: Sequence[float] | None = None,
    pfe_quantile: float = 0.95,
    asof_date: Any | None = None,
) -> dict[str, Any]:
    val = np.asarray(valuation_paths, dtype=float)
    close = np.asarray(closeout_paths, dtype=float)
    valuation_epe = _mean_positive(val)
    valuation_ene = _mean_negative(val)
    closeout_epe = _mean_positive(close)
    closeout_ene = _mean_negative(close)
    pfe = ore_pfe_quantile(close, pfe_quantile)
    return build_ore_exposure_profile_from_series(
        entity_id,
        exposure_dates,
        exposure_times,
        valuation_epe=valuation_epe,
        valuation_ene=valuation_ene,
        closeout_epe=closeout_epe,
        closeout_ene=closeout_ene,
        pfe=pfe,
        discount_factors=discount_factors,
        closeout_times=closeout_times,
        expected_collateral=expected_collateral,
        asof_date=asof_date,
    )


def build_ore_exposure_profile_from_series(
    entity_id: str,
    exposure_dates: Sequence[Any],
    exposure_times: Sequence[float],
    *,
    valuation_epe: Sequence[float],
    valuation_ene: Sequence[float],
    closeout_epe: Sequence[float],
    closeout_ene: Sequence[float],
    pfe: Sequence[float],
    discount_factors: Sequence[float] | None = None,
    closeout_times: Sequence[float] | None = None,
    expected_collateral: Sequence[float] | None = None,
    asof_date: Any | None = None,
) -> dict[str, Any]:
    dates = [str(x) for x in exposure_dates]
    times = np.asarray(exposure_times, dtype=float)
    valuation_epe_arr = np.asarray(valuation_epe, dtype=float)
    valuation_ene_arr = np.asarray(valuation_ene, dtype=float)
    closeout_epe_arr = np.asarray(closeout_epe, dtype=float)
    closeout_ene_arr = np.asarray(closeout_ene, dtype=float)
    pfe_arr = np.asarray(pfe, dtype=float)
    shapes = {
        valuation_epe_arr.shape,
        valuation_ene_arr.shape,
        closeout_epe_arr.shape,
        closeout_ene_arr.shape,
        pfe_arr.shape,
        times.shape,
    }
    if len(shapes) != 1:
        raise ValueError("exposure profile series must all share the same shape")
    closeout = np.asarray(closeout_times if closeout_times is not None else exposure_times, dtype=float)
    if closeout.shape != times.shape:
        raise ValueError("closeout_times must match exposure_times shape")
    basel = build_ore_basel_profile(
        exposure_dates=dates,
        exposure_times=times,
        closeout_epe=closeout_epe_arr,
        discount_factors=discount_factors,
        asof_date=asof_date,
    )
    if expected_collateral is None:
        exp_coll = np.zeros_like(closeout_epe_arr)
        if exp_coll.size:
            exp_coll[0] = closeout_epe_arr[0]
    else:
        exp_coll = np.asarray(expected_collateral, dtype=float)
        if exp_coll.shape != times.shape:
            raise ValueError("expected_collateral must match exposure_times shape")
    return {
        "entity_id": entity_id,
        "dates": dates,
        "times": times.tolist(),
        "closeout_times": closeout.tolist(),
        "valuation_epe": valuation_epe_arr.tolist(),
        "valuation_ene": valuation_ene_arr.tolist(),
        "closeout_epe": closeout_epe_arr.tolist(),
        "closeout_ene": closeout_ene_arr.tolist(),
        "pfe": pfe_arr.tolist(),
        "expected_collateral": exp_coll.tolist(),
        "basel_ee": basel["basel_ee"],
        "basel_eee": basel["basel_eee"],
        "time_weighted_basel_epe": basel["time_weighted_basel_epe"],
        "time_weighted_basel_eepe": basel["time_weighted_basel_eepe"],
    }


def one_year_profile_value(
    profile: Mapping[str, Any],
    key: str,
    *,
    asof_date: Any | None = None,
) -> float:
    dates = profile.get("dates", [])
    times = profile.get("times", [])
    values = np.asarray(profile.get(key, []), dtype=float)
    if values.size == 0:
        return 0.0
    idx = ore_one_year_index(dates, times, asof_date=asof_date)
    idx = min(idx, values.size - 1)
    return float(values[idx])
