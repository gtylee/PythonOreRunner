from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import erf, exp, log, sqrt
from pathlib import Path
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from pythonore.payoff_ir.exec_numpy import NumpyExecutionEnv


def _to_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date value {value!r}")


def _yearfrac(start: date, end: date) -> float:
    return (end - start).days / 365.0


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _continuous_zero_rate_from_df(df: float, t: float) -> float:
    tt = max(float(t), 0.0)
    if tt <= 1.0e-12:
        return 0.0
    return -log(max(float(df), 1.0e-16)) / tt


def _black_forward_option_npv(*, forward: float, strike: float, maturity_time: float, vol: float, discount: float, call: bool) -> float:
    tt = max(float(maturity_time), 0.0)
    if tt <= 1.0e-12 or vol <= 1.0e-12:
        intrinsic = max(float(forward) - float(strike), 0.0) if call else max(float(strike) - float(forward), 0.0)
        return float(discount) * intrinsic
    std_dev = max(float(vol) * sqrt(tt), 1.0e-12)
    d1 = (log(max(float(forward), 1.0e-12) / max(float(strike), 1.0e-12)) + 0.5 * std_dev * std_dev) / std_dev
    d2 = d1 - std_dev
    if call:
        return float(discount) * (float(forward) * _normal_cdf(d1) - float(strike) * _normal_cdf(d2))
    return float(discount) * (float(strike) * _normal_cdf(-d2) - float(forward) * _normal_cdf(-d1))


def _implied_vol_from_premium(*, premium: float, forward: float, strike: float, maturity_time: float, discount: float, call: bool) -> float:
    target = max(float(premium), 0.0)
    intrinsic = _black_forward_option_npv(
        forward=forward, strike=strike, maturity_time=maturity_time, vol=0.0, discount=discount, call=call
    )
    if target <= intrinsic + 1.0e-10:
        return 1.0e-8
    lo, hi = 1.0e-8, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        price = _black_forward_option_npv(
            forward=forward, strike=strike, maturity_time=maturity_time, vol=mid, discount=discount, call=call
        )
        if price < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _interp_linear(points: list[tuple[float, float]], x: float) -> float:
    ordered = sorted((float(px), float(py)) for px, py in points)
    if not ordered:
        raise ValueError("no interpolation points")
    xx = float(x)
    if xx <= ordered[0][0]:
        return ordered[0][1]
    if xx >= ordered[-1][0]:
        return ordered[-1][1]
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= xx <= x1:
            if abs(x1 - x0) <= 1.0e-16:
                return y1
            w = (xx - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return ordered[-1][1]


def _interp_total_variance(points: list[tuple[float, float]], maturity: float) -> float:
    tt = max(float(maturity), 1.0e-12)
    ordered = sorted((max(float(t), 1.0e-12), max(float(v), 1.0e-12)) for t, v in points)
    if not ordered:
        raise ValueError("no total variance interpolation points")
    if tt <= ordered[0][0]:
        return ordered[0][1]
    if tt >= ordered[-1][0]:
        if len(ordered) == 1:
            return ordered[0][1]
        t0, v0 = ordered[-2]
        t1, v1 = ordered[-1]
    else:
        for (t0, v0), (t1, v1) in zip(ordered, ordered[1:]):
            if t0 <= tt <= t1:
                break
    tv0 = v0 * v0 * t0
    tv1 = v1 * v1 * t1
    if abs(t1 - t0) <= 1.0e-16:
        return v1
    tv = tv0 + (tt - t0) / (t1 - t0) * (tv1 - tv0)
    return sqrt(max(tv / tt, 0.0))


@dataclass
class _SparseOptionSurface:
    asof: date
    rows: dict[date, dict[float, float]]

    def expiries(self) -> list[date]:
        return sorted(self.rows)

    def strikes_at(self, expiry: date) -> list[float]:
        row = self.rows.get(expiry, {})
        return sorted(float(k) for k in row)

    def get_value(self, expiry: date, strike: float) -> float:
        expiries = self.expiries()
        if not expiries:
            raise ValueError("empty sparse surface")
        if expiry in self.rows:
            return self._value_for_strike(expiry, strike)
        target_t = max(_yearfrac(self.asof, expiry), 0.0)
        pts = [(max(_yearfrac(self.asof, d), 0.0), self._value_for_strike(d, strike)) for d in expiries]
        return float(_interp_linear(pts, target_t))

    def _value_for_strike(self, expiry: date, strike: float) -> float:
        row = self.rows.get(expiry, {})
        if not row:
            raise ValueError(f"no row for expiry {expiry}")
        points = sorted((float(k), float(v)) for k, v in row.items())
        return float(_interp_linear(points, float(strike)))


def _build_sparse_option_surface(
    *,
    asof: date,
    quotes: Mapping[str, float],
    prefix: str,
) -> _SparseOptionSurface | None:
    rows: dict[date, dict[float, float]] = {}
    for quote_id, quote_value in quotes.items():
        if not quote_id.startswith(prefix):
            continue
        parts = quote_id.split("/")
        if len(parts) < 7:
            continue
        try:
            expiry_date = _to_date(parts[4])
            strike_value = float(parts[5])
            value = float(quote_value)
        except Exception:
            continue
        rows.setdefault(expiry_date, {})[strike_value] = value
    if not rows:
        return None
    return _SparseOptionSurface(asof=asof, rows=rows)


def _create_relevant_strikes(forward: float, call_strikes: list[float], put_strikes: list[float], prefer_out_of_the_money: bool) -> dict[float, str]:
    relevant: dict[float, str] = {}
    restricted_calls = [k for k in call_strikes if (k >= forward) == prefer_out_of_the_money]
    restricted_puts = [k for k in put_strikes if (k <= forward) == prefer_out_of_the_money]
    if restricted_calls and restricted_puts:
        for k in restricted_puts:
            relevant[float(k)] = "P"
        for k in restricted_calls:
            relevant[float(k)] = "C"
    elif restricted_calls:
        for k in call_strikes:
            relevant[float(k)] = "C"
    elif restricted_puts:
        for k in put_strikes:
            relevant[float(k)] = "P"
    else:
        for k in call_strikes:
            relevant[float(k)] = "C"
    return relevant


def _build_premium_surface_vol_curve(
    *,
    quotes: Mapping[str, float],
    asof: date,
    forward_fn: Callable[[float], float],
    discount_curve: Callable[[float], float],
    equity_name: str,
    currency: str,
    strike: float,
    option_type: str,
    prefer_out_of_the_money: bool = True,
) -> Callable[[float], float] | None:
    prefix = f"EQUITY_OPTION/PRICE/{equity_name}/{currency}/"
    suffix = "C" if str(option_type).strip().upper().startswith("C") else "P"
    premium_points: list[tuple[float, float]] = []
    for quote_id, quote_value in quotes.items():
        if not quote_id.startswith(prefix) or not quote_id.endswith(f"/{suffix}"):
            continue
        parts = quote_id.split("/")
        if len(parts) < 7:
            continue
        try:
            expiry_date = _to_date(parts[4])
            strike_val = float(parts[5])
        except Exception:
            continue
        if abs(strike_val - float(strike)) > 1.0e-10:
            continue
        maturity = max(_yearfrac(asof, expiry_date), 1.0e-12)
        premium_points.append((maturity, float(quote_value)))
    if not premium_points:
        return None

    def _curve(t: float) -> float:
        maturity = max(float(t), 1.0e-12)
        premium = float(_interp_linear(premium_points, maturity))
        forward = float(forward_fn(maturity))
        discount = float(discount_curve(maturity))
        return float(
            _implied_vol_from_premium(
                premium=premium,
                forward=forward,
                strike=float(strike),
                maturity_time=maturity,
                discount=discount,
                call=suffix == "C",
            )
        )

    return _curve


def _build_lnvol_surface_vol_curve(
    *,
    quotes: Mapping[str, float],
    asof: date,
    forward_fn: Callable[[float], float],
    equity_name: str,
    currency: str,
    strike: float,
    option_type: str,
    prefer_out_of_the_money: bool = True,
) -> Callable[[float], float] | None:
    prefix = f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/"
    fallback_suffix = "C" if str(option_type).strip().upper().startswith("C") else "P"
    rows: dict[date, dict[str, float]] = {}
    target_strike = float(strike)
    tol = max(abs(target_strike), 1.0) * 1.0e-10
    for quote_id, quote_value in quotes.items():
        if not quote_id.startswith(prefix):
            continue
        parts = quote_id.split("/")
        if len(parts) < 7:
            continue
        try:
            expiry_date = _to_date(parts[4])
            strike_val = float(parts[5])
            cp_flag = str(parts[6]).strip().upper()
        except Exception:
            continue
        if cp_flag not in {"C", "P"}:
            continue
        if abs(strike_val - target_strike) > tol:
            continue
        rows.setdefault(expiry_date, {})[cp_flag] = float(quote_value)

    vol_points: list[tuple[float, float]] = []
    for expiry_date, vol_quotes in sorted(rows.items()):
        maturity = max(_yearfrac(asof, expiry_date), 1.0e-12)
        forward = float(forward_fn(maturity))
        preferred_suffix = "C" if ((prefer_out_of_the_money and target_strike >= forward) or (not prefer_out_of_the_money and target_strike <= forward)) else "P"
        chosen_suffix = preferred_suffix if preferred_suffix in vol_quotes else (fallback_suffix if fallback_suffix in vol_quotes else next(iter(vol_quotes), None))
        if chosen_suffix is None:
            continue
        vol_points.append((maturity, float(vol_quotes[chosen_suffix])))
    if not vol_points:
        return None

    def _curve(t: float) -> float:
        return float(_interp_total_variance(vol_points, float(t)))

    return _curve


def _build_stripped_premium_surface_vol_curve(
    *,
    quotes: Mapping[str, float],
    asof: date,
    forward_fn: Callable[[float], float],
    discount_curve: Callable[[float], float],
    equity_name: str,
    currency: str,
    strike: float,
    option_type: str,
) -> Callable[[float], float] | None:
    suffix = "C" if str(option_type).strip().upper().startswith("C") else "P"
    call_surface = _build_sparse_option_surface(
        asof=asof,
        quotes={k: v for k, v in quotes.items() if k.startswith(f"EQUITY_OPTION/PRICE/{equity_name}/{currency}/") and k.endswith("/C")},
        prefix=f"EQUITY_OPTION/PRICE/{equity_name}/{currency}/",
    )
    put_surface = _build_sparse_option_surface(
        asof=asof,
        quotes={k: v for k, v in quotes.items() if k.startswith(f"EQUITY_OPTION/PRICE/{equity_name}/{currency}/") and k.endswith("/P")},
        prefix=f"EQUITY_OPTION/PRICE/{equity_name}/{currency}/",
    )
    if call_surface is None or put_surface is None:
        return None
    premium_points: list[tuple[float, float]] = []
    for expiry_date in sorted(set(call_surface.expiries()) | set(put_surface.expiries())):
        maturity = max(_yearfrac(asof, expiry_date), 1.0e-12)
        forward = float(forward_fn(maturity))
        discount = float(discount_curve(maturity))
        call_strikes = call_surface.strikes_at(expiry_date)
        put_strikes = put_surface.strikes_at(expiry_date)
        if not call_strikes and not put_strikes:
            continue
        if not call_strikes or not put_strikes:
            active_surface = call_surface if call_strikes else put_surface
            premium = active_surface.get_value(expiry_date, float(strike))
            premium_points.append(
                (
                    maturity,
                    _implied_vol_from_premium(
                        premium=premium,
                        forward=forward,
                        strike=float(strike),
                        maturity_time=maturity,
                        discount=discount,
                        call=suffix == "C",
                    ),
                )
            )
            continue
        relevant = _create_relevant_strikes(forward, call_strikes, put_strikes, True)
        stripped_row: list[tuple[float, float]] = []
        for k in sorted(relevant):
            flag = relevant[k]
            active_surface = call_surface if flag == "C" else put_surface
            premium = active_surface.get_value(expiry_date, k)
            implied = _implied_vol_from_premium(
                premium=float(premium),
                forward=forward,
                strike=float(k),
                maturity_time=maturity,
                discount=discount,
                call=flag == "C",
            )
            stripped_row.append((float(k), float(implied)))

        if not stripped_row:
            continue
        if len(stripped_row) == 1:
            implied_at_strike = stripped_row[0][1]
        else:
            strike_tv = [(k, v * v * maturity) for k, v in stripped_row]
            implied_at_strike = sqrt(max(_interp_linear(strike_tv, float(strike)) / maturity, 0.0))
        premium_points.append(
            (
                maturity,
                implied_at_strike,
            )
        )
    if not premium_points:
        return None

    def _curve(t: float) -> float:
        return float(_interp_total_variance(premium_points, float(t)))

    return _curve


def _build_stripped_lnvol_surface_vol_curve(
    *,
    quotes: Mapping[str, float],
    asof: date,
    forward_fn: Callable[[float], float],
    equity_name: str,
    currency: str,
    strike: float,
    prefer_out_of_the_money: bool = True,
) -> Callable[[float], float] | None:
    call_surface = _build_sparse_option_surface(
        asof=asof,
        quotes={k: v for k, v in quotes.items() if k.startswith(f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/") and k.endswith("/C")},
        prefix=f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/",
    )
    put_surface = _build_sparse_option_surface(
        asof=asof,
        quotes={k: v for k, v in quotes.items() if k.startswith(f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/") and k.endswith("/P")},
        prefix=f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/",
    )
    if call_surface is None or put_surface is None:
        return None

    vol_points: list[tuple[float, float]] = []
    for expiry_date in sorted(set(call_surface.expiries()) | set(put_surface.expiries())):
        maturity = max(_yearfrac(asof, expiry_date), 1.0e-12)
        forward = float(forward_fn(maturity))
        call_strikes = call_surface.strikes_at(expiry_date)
        put_strikes = put_surface.strikes_at(expiry_date)
        if not call_strikes and not put_strikes:
            continue
        relevant = _create_relevant_strikes(forward, call_strikes, put_strikes, prefer_out_of_the_money)

        stripped_row: list[tuple[float, float]] = []
        for k in sorted(relevant):
            flag = relevant[k]
            active_surface = call_surface if flag == "C" else put_surface
            stripped_row.append((float(k), float(active_surface.get_value(expiry_date, k))))
        if not stripped_row:
            continue
        if len(stripped_row) == 1:
            implied_at_strike = stripped_row[0][1]
        else:
            strike_tv = [(k, v * v * maturity) for k, v in stripped_row]
            implied_at_strike = sqrt(max(_interp_linear(strike_tv, float(strike)) / maturity, 0.0))
        vol_points.append((maturity, implied_at_strike))

    if not vol_points:
        return None

    def _curve(t: float) -> float:
        return float(_interp_total_variance(vol_points, float(t)))

    return _curve


@dataclass
class BlackScholesMonteCarloModel:
    reference_date: date
    spot: float
    volatility: float | None = None
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    n_paths: int = 10000
    seed: int = 0
    index_name: str = "Underlying"
    currency: str = "EUR"
    observation_dates: tuple[Any, ...] = ()
    time_steps_per_year: int = 0
    discount_curve: Callable[[float], float] | None = None
    dividend_curve: Callable[[float], float] | None = None
    vol_curve: Callable[[float], float] | None = None
    paths_by_date: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    cumulative_variance_by_date: dict[str, float] = field(init=False, default_factory=dict)
    effective_simulation_dates: tuple[date, ...] = field(init=False, default_factory=tuple)

    def __post_init__(self):
        dates = sorted({_to_date(d) for d in self.observation_dates})
        if not dates:
            self.paths_by_date = {self.reference_date.isoformat(): np.full(self.n_paths, self.spot)}
            self.cumulative_variance_by_date = {self.reference_date.isoformat(): 0.0}
            self.effective_simulation_dates = (self.reference_date,)
            return
        self.effective_simulation_dates = self._build_effective_simulation_dates(dates)
        rng = np.random.default_rng(self.seed)
        prev_t = 0.0
        prev = np.full(self.n_paths, self.spot)
        prev_discount_ratio = 1.0
        cumulative_variance = 0.0
        self.paths_by_date[self.reference_date.isoformat()] = prev.copy()
        self.cumulative_variance_by_date[self.reference_date.isoformat()] = 0.0
        observation_set = {d.isoformat() for d in dates}
        for d in self.effective_simulation_dates[1:]:
            t = max(_yearfrac(self.reference_date, d), 0.0)
            dt = max(t - prev_t, 0.0)
            z = rng.standard_normal(self.n_paths)
            variance_increment = max(self._variance_increment(prev_t, t), 0.0)
            sigma = sqrt(max(variance_increment / max(dt, 1.0e-16), 0.0)) if dt > 0.0 else 0.0
            discount_ratio = self._discount_ratio(t)
            drift = -np.log(max(discount_ratio / max(prev_discount_ratio, 1.0e-16), 1.0e-16)) - 0.5 * variance_increment
            diffusion = sigma * sqrt(dt) * z
            prev = prev * np.exp(drift + diffusion)
            cumulative_variance += variance_increment
            if d.isoformat() in observation_set:
                self.paths_by_date[d.isoformat()] = prev.copy()
                self.cumulative_variance_by_date[d.isoformat()] = cumulative_variance
            prev_discount_ratio = discount_ratio
            prev_t = t
        for d in dates:
            key = d.isoformat()
            if key not in self.paths_by_date:
                raise KeyError(f"missing simulated observation date {key}")

    def _build_effective_simulation_dates(self, observation_dates: list[date]) -> tuple[date, ...]:
        if self.time_steps_per_year <= 0:
            return tuple([self.reference_date] + observation_dates)
        effective = {self.reference_date, *observation_dates}
        for start, end in zip([self.reference_date] + observation_dates[:-1], observation_dates):
            dt_years = max(_yearfrac(start, end), 0.0)
            n_steps = max(int(np.ceil(dt_years * float(self.time_steps_per_year))), 1)
            if n_steps <= 1:
                continue
            total_days = max((end - start).days, 0)
            for step in range(1, n_steps):
                offset_days = int(round(total_days * step / n_steps))
                if 0 < offset_days < total_days:
                    effective.add(start + timedelta(days=offset_days))
        return tuple(sorted(effective))

    def _variance_increment(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        if self.vol_curve is not None:
            vol1 = float(self.vol_curve(t1))
            vol0 = float(self.vol_curve(t0)) if t0 > 0.0 else vol1
            var1 = vol1 * vol1 * max(float(t1), 0.0)
            var0 = vol0 * vol0 * max(float(t0), 0.0)
            return max(var1 - var0, 0.0)
        vol = float(self.volatility or 0.0)
        return vol * vol * (t1 - t0)

    def _discount_df(self, t: float) -> float:
        if self.discount_curve is not None:
            return float(self.discount_curve(max(t, 0.0)))
        return float(np.exp(-self.risk_free_rate * max(t, 0.0)))

    def _dividend_df(self, t: float) -> float:
        if self.dividend_curve is not None:
            return float(self.dividend_curve(max(t, 0.0)))
        return float(np.exp(-self.dividend_yield * max(t, 0.0)))

    def _discount_ratio(self, t: float) -> float:
        return self._discount_df(t) / max(self._dividend_df(t), 1.0e-16)

    def index_at(self, index: str, obs_date: Any, n_paths: int):
        if index != self.index_name:
            raise KeyError(f"Unsupported index '{index}', expected '{self.index_name}'")
        key = _to_date(obs_date).isoformat()
        return self.paths_by_date[key]

    def discount(self, obs_date: Any, pay_date: Any, currency: str, n_paths: int):
        if currency != self.currency:
            raise KeyError(f"Unsupported currency '{currency}', expected '{self.currency}'")
        t_obs = _yearfrac(self.reference_date, _to_date(obs_date))
        t_pay = _yearfrac(self.reference_date, _to_date(pay_date))
        df = self._discount_df(t_pay) / max(self._discount_df(t_obs), 1.0e-16)
        return np.full(n_paths, df)

    def discount_t0(self, pay_date: Any, currency: str):
        if currency != self.currency:
            raise KeyError(f"Unsupported currency '{currency}', expected '{self.currency}'")
        t = _yearfrac(self.reference_date, _to_date(pay_date))
        return self._discount_df(t)

    def above_prob(self, index: str, obs_date1: Any, obs_date2: Any, level: float, n_paths: int):
        return self._barrier_prob(index, obs_date1, obs_date2, level, above=True)

    def below_prob(self, index: str, obs_date1: Any, obs_date2: Any, level: float, n_paths: int):
        return self._barrier_prob(index, obs_date1, obs_date2, level, above=False)

    def _barrier_prob(self, index: str, obs_date1: Any, obs_date2: Any, level: float, above: bool):
        v1 = self.index_at(index, obs_date1, self.n_paths)
        v2 = self.index_at(index, obs_date2, self.n_paths)
        d1 = _to_date(obs_date1).isoformat()
        d2 = _to_date(obs_date2).isoformat()
        variance = max(self.cumulative_variance_by_date[d2] - self.cumulative_variance_by_date[d1], 0.0)
        if above:
            barrier_hit = (v1 >= level) | (v2 >= level)
        else:
            barrier_hit = (v1 <= level) | (v2 <= level)
        result = np.where(barrier_hit, 1.0, 0.0)
        if variance <= 0.0:
            return result
        eps = 1.0e-14
        hit_prob = np.exp((-2.0 / variance) * np.log(v1 / max(level, eps)) * np.log(v2 / max(level, eps)))
        return result + np.where(barrier_hit, 0.0, hit_prob)

    def make_env(self, parameters: Mapping[str, Any]) -> NumpyExecutionEnv:
        def _pay(amount: Any, obs_date: Any, pay_date: Any, currency: str, n_paths: int):
            if currency != self.currency:
                raise KeyError(f"Unsupported currency '{currency}', expected '{self.currency}'")
            pay_date_cmp = _to_date(pay_date)
            if pay_date_cmp <= self.reference_date:
                return np.zeros(n_paths)
            df = self.discount_t0(pay_date, currency)
            return np.asarray(amount, dtype=float) * float(df)

        return NumpyExecutionEnv(
            parameters=parameters,
            n_paths=self.n_paths,
            index_at=self.index_at,
            discount=self.discount,
            pay=_pay,
            above_prob=self.above_prob,
            below_prob=self.below_prob,
            reference_date=self.reference_date,
            discount_t0=self.discount_t0,
        )

    def fd_single_asset_option_price(
        self,
        *,
        strike: float,
        expiry: Any,
        settlement: Any,
        put_call: float = 1.0,
        long_short: float = 1.0,
        quantity: float = 1.0,
        time_grid: int = 200,
        x_grid: int = 200,
        damping_steps: int = 0,
    ) -> float:
        import QuantLib as ql

        expiry_date = _to_date(expiry)
        settlement_date = _to_date(settlement)
        maturity_time = max(_yearfrac(self.reference_date, expiry_date), 0.0)
        settlement_time = max(_yearfrac(self.reference_date, settlement_date), 0.0)
        rate = _continuous_zero_rate_from_df(self._discount_df(maturity_time), maturity_time)
        dividend = _continuous_zero_rate_from_df(self._dividend_df(maturity_time), maturity_time)
        vol = float(self.vol_curve(maturity_time) if self.vol_curve is not None else (self.volatility or 0.0))

        ql.Settings.instance().evaluationDate = ql.Date(
            self.reference_date.day, self.reference_date.month, self.reference_date.year
        )
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(self.spot)))
        rf_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(ql.Settings.instance().evaluationDate, float(rate), ql.Actual365Fixed())
        )
        div_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(ql.Settings.instance().evaluationDate, float(dividend), ql.Actual365Fixed())
        )
        vol_curve = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                ql.Settings.instance().evaluationDate, ql.NullCalendar(), float(vol), ql.Actual365Fixed()
            )
        )
        process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rf_curve, vol_curve)
        option_type = ql.Option.Call if float(put_call) >= 0.0 else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(option_type, float(strike))
        exercise = ql.EuropeanExercise(ql.Date(expiry_date.day, expiry_date.month, expiry_date.year))
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, int(time_grid), int(x_grid), int(damping_steps)))
        npv = float(option.NPV())
        if settlement_date > expiry_date:
            npv *= float(self._discount_df(settlement_time) / max(self._discount_df(maturity_time), 1.0e-16))
        return float(long_short) * float(quantity) * npv


def build_equity_ore_black_scholes_model(
    *,
    ore_xml: Path,
    todaysmarket_xml: Path,
    curveconfig_path: Path,
    market_data_path: Path,
    pricing_config_id: str,
    asof: date,
    equity_name: str,
    currency: str,
    strike: float,
    option_type: str = "Call",
    vol_surface_mode: str = "auto",
    observation_dates: tuple[Any, ...],
    n_paths: int,
    seed: int,
) -> BlackScholesMonteCarloModel:
    from pythonore.io.ore_snapshot import fit_discount_curves_from_ore_market
    from pythonore.workflows import ore_snapshot_cli as osc

    pricing_root = ET.parse(ore_xml.parent / "pricingengine.xml").getroot()
    scripted_product = next((p for p in pricing_root.findall("./Product") if p.get("type") == "ScriptedTrade"), None)
    time_steps_per_year = 0
    if scripted_product is not None:
        ts_node = scripted_product.find("./EngineParameters/Parameter[@name='TimeStepsPerYear']")
        if ts_node is not None and (ts_node.text or "").strip():
            try:
                time_steps_per_year = int((ts_node.text or "").strip())
            except Exception:
                time_steps_per_year = 0
    # Mirror ORE's MC builder: ignore timestep refinement when there are no correlations.
    # The current Python helper is single-asset / uncorrelated, so large timesteps are equivalent.
    time_steps_per_year = 0

    quotes = osc._parse_market_quotes(market_data_path, asof.isoformat())
    curve_spec = osc._load_equity_curve_spec(
        todaysmarket_xml,
        curveconfig_path,
        pricing_config_id=pricing_config_id,
        equity_name=equity_name,
    )
    vol_spec = osc._load_equity_vol_spec(
        todaysmarket_xml,
        curveconfig_path,
        pricing_config_id=pricing_config_id,
        equity_name=equity_name,
    )
    spot = float(quotes[curve_spec["spot_quote"]])

    fitted = fit_discount_curves_from_ore_market(ore_xml_path=ore_xml)
    curve_payload = fitted.get(str(currency).upper())
    if curve_payload and curve_payload.get("times") and curve_payload.get("dfs"):
        discount_curve = osc.ore_snapshot_mod.build_discount_curve_from_discount_pairs(
            list(zip(curve_payload["times"], curve_payload["dfs"]))
        )
    else:
        discount_curve = osc._build_discount_curve_from_market_fit(ore_xml, currency)

    forecast_curve = osc._build_discount_curve_from_market_fit(ore_xml, currency)
    forecasting_curve_name = str(curve_spec.get("forecasting_curve") or "").strip()
    if forecasting_curve_name:
        try:
            handle = f"Yield/{currency}/{forecasting_curve_name}"
            tm_root = ET.parse(todaysmarket_xml).getroot()
            forecast_column = osc.ore_snapshot_mod._handle_to_curve_name(tm_root, handle)
            curves_csv = (ore_xml.parent.parent / "Output" / "curves.csv").resolve()
            curve_dates = osc.ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
                str(curves_csv),
                [forecast_column],
                asof_date=asof.isoformat(),
                day_counter="A365F",
            )
            _, curve_times, curve_dfs = curve_dates[forecast_column]
            forecast_curve = osc.ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times, curve_dfs)))
        except Exception:
            forecast_curve = osc._build_discount_curve_from_market_fit(ore_xml, currency)

    curve_type = str(curve_spec.get("curve_type") or "").strip().lower()
    if curve_type == "dividendyield":
        dividend_curve = osc._load_equity_dividend_curve(curve_spec, quotes=quotes, asof=asof)
    elif curve_type == "forwardprice":
        forward_curve = osc._load_equity_forward_curve(curve_spec, quotes=quotes, asof=asof, spot=spot)

        def dividend_curve(t: float) -> float:
            return float(forward_curve(t)) * float(forecast_curve(t)) / max(float(spot), 1.0e-16)

    else:
        dividend_curve = lambda t: 1.0

    direct_lnvol_curve = _build_lnvol_surface_vol_curve(
        quotes=quotes,
        asof=asof,
        forward_fn=lambda maturity: osc._equity_forward_from_market_inputs(
            curve_spec=curve_spec,
            spot=spot,
            maturity_time=maturity,
            forecast_curve=forecast_curve,
            quotes=quotes,
            asof=asof,
        ),
        equity_name=equity_name,
        currency=currency,
        strike=strike,
        option_type=option_type,
    )
    stripped_lnvol_curve = _build_stripped_lnvol_surface_vol_curve(
        quotes=quotes,
        asof=asof,
        forward_fn=lambda maturity: osc._equity_forward_from_market_inputs(
            curve_spec=curve_spec,
            spot=spot,
            maturity_time=maturity,
            forecast_curve=forecast_curve,
            quotes=quotes,
            asof=asof,
        ),
        equity_name=equity_name,
        currency=currency,
        strike=strike,
    )
    direct_premium_curve = _build_premium_surface_vol_curve(
        quotes=quotes,
        asof=asof,
        forward_fn=lambda maturity: osc._equity_forward_from_market_inputs(
            curve_spec=curve_spec,
            spot=spot,
            maturity_time=maturity,
            forecast_curve=forecast_curve,
            quotes=quotes,
            asof=asof,
        ),
        discount_curve=discount_curve,
        equity_name=equity_name,
        currency=currency,
        strike=strike,
        option_type=option_type,
    )
    stripped_premium_curve = _build_stripped_premium_surface_vol_curve(
        quotes=quotes,
        asof=asof,
        forward_fn=lambda maturity: osc._equity_forward_from_market_inputs(
            curve_spec=curve_spec,
            spot=spot,
            maturity_time=maturity,
            forecast_curve=forecast_curve,
            quotes=quotes,
            asof=asof,
        ),
        discount_curve=discount_curve,
        equity_name=equity_name,
        currency=currency,
        strike=strike,
        option_type=option_type,
    )
    unique_obs = sorted({_to_date(d) for d in observation_dates if _to_date(d) >= asof})
    selected_mode = str(vol_surface_mode).strip().lower()
    if selected_mode == "auto":
        # Dense monitoring schedules benefit from the stripped OTM surface, while
        # sparse long-dated fixings behave better with direct same-strike inversion.
        selected_mode = "stripped" if len(unique_obs) > 20 else "direct"
    max_obs_t = max((_yearfrac(asof, d) for d in unique_obs), default=0.0)
    if selected_mode == "stripped":
        premium_curve = stripped_premium_curve or stripped_lnvol_curve
    else:
        premium_curve = (direct_lnvol_curve or direct_premium_curve) if max_obs_t <= 1.0 else (direct_premium_curve or direct_lnvol_curve)

    def vol_curve(t: float) -> float:
        maturity = max(float(t), 1.0e-12)
        if premium_curve is not None:
            return float(premium_curve(maturity))
        try:
            return float(
                osc._load_equity_smile_vol(
                    quotes,
                    asof=asof,
                    equity_name=equity_name,
                    currency=currency,
                    maturity_time=maturity,
                    strike=strike,
                    spec=vol_spec,
                )
            )
        except Exception:
            atm_points = []
            for expiry_label in vol_spec.get("expiries", []):
                expiry_time = osc._market_label_to_time(asof, expiry_label)
                for atm_label in ("ATMF", "ATM"):
                    quote_id = f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/{expiry_label}/{atm_label}"
                    if quote_id in quotes:
                        atm_points.append((expiry_time, float(quotes[quote_id])))
                        break
            if not atm_points:
                prefix = f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/"
                for quote_id, quote_value in quotes.items():
                    if not quote_id.startswith(prefix):
                        continue
                    parts = quote_id.split("/")
                    if len(parts) >= 7 and parts[5] == "ATM":
                        try:
                            expiry_time = osc._market_label_to_time(asof, parts[4])
                            atm_points.append((expiry_time, float(quote_value)))
                        except Exception:
                            continue
            if not atm_points:
                raise
            return float(osc._interp_total_variance(atm_points, maturity))

    return BlackScholesMonteCarloModel(
        reference_date=asof,
        spot=spot,
        volatility=None,
        n_paths=n_paths,
        seed=seed,
        index_name=equity_name,
        currency=currency,
        observation_dates=observation_dates,
        time_steps_per_year=time_steps_per_year,
        discount_curve=discount_curve,
        dividend_curve=dividend_curve,
        vol_curve=vol_curve,
    )
