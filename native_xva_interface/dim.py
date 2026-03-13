from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .exceptions import EngineRunError
from .results import CubeAccessor, DIMMarginComponents, DIMResult


def _as_np(value: Any, *, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise EngineRunError(f"Expected array with ndim={ndim}, got shape {arr.shape}")
    return arr


def _period_labels(value: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(x) for x in value)


def _max_abs(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(np.max(np.abs(value)))


def _is_close_zero(value: float, tol: float = 1.0e-14) -> bool:
    return abs(float(value)) <= tol


def _zero_components(total: np.ndarray) -> DIMMarginComponents:
    total = np.asarray(total, dtype=float)
    zeros = np.zeros_like(total, dtype=float)
    return DIMMarginComponents(
        total=total,
        delta_margin_ir=zeros.copy(),
        vega_margin_ir=zeros.copy(),
        curvature_margin_ir=zeros.copy(),
        delta_margin_fx=zeros.copy(),
        vega_margin_fx=zeros.copy(),
        curvature_margin_fx=zeros.copy(),
        delta_margin=zeros.copy(),
        vega_margin=zeros.copy(),
        curvature_margin=zeros.copy(),
        ir_delta_margin=zeros.copy(),
        fx_delta_margin=zeros.copy(),
    )


@dataclass(frozen=True)
class PythonSimpleDynamicSimmConfig:
    corr_ir_fx: float
    ir_delta_rw: np.ndarray
    ir_vega_rw: float
    ir_gamma: float
    ir_curvature_scaling: float
    ir_delta_correlations: np.ndarray
    ir_vega_correlations: np.ndarray
    ir_curvature_weights: np.ndarray
    fx_delta_rw: float
    fx_vega_rw: float
    fx_sigma: float
    fx_hvr: float
    fx_corr: float
    fx_vega_correlations: np.ndarray
    fx_curvature_weights: np.ndarray

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PythonSimpleDynamicSimmConfig":
        return cls(
            corr_ir_fx=float(data["corr_ir_fx"]),
            ir_delta_rw=_as_np(data["ir_delta_rw"], ndim=1),
            ir_vega_rw=float(data["ir_vega_rw"]),
            ir_gamma=float(data["ir_gamma"]),
            ir_curvature_scaling=float(data["ir_curvature_scaling"]),
            ir_delta_correlations=_as_np(data["ir_delta_correlations"], ndim=2),
            ir_vega_correlations=_as_np(data["ir_vega_correlations"], ndim=2),
            ir_curvature_weights=_as_np(data["ir_curvature_weights"], ndim=1),
            fx_delta_rw=float(data["fx_delta_rw"]),
            fx_vega_rw=float(data["fx_vega_rw"]),
            fx_sigma=float(data["fx_sigma"]),
            fx_hvr=float(data["fx_hvr"]),
            fx_corr=float(data["fx_corr"]),
            fx_vega_correlations=_as_np(data["fx_vega_correlations"], ndim=2),
            fx_curvature_weights=_as_np(data["fx_curvature_weights"], ndim=1),
        )


@dataclass(frozen=True)
class PythonVarDimConfig:
    quantile: float
    horizon_calendar_days: float = 0.0

    @property
    def theta_factor(self) -> float:
        return float(self.horizon_calendar_days) / 365.25

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PythonVarDimConfig":
        return cls(
            quantile=float(data["quantile"]),
            horizon_calendar_days=float(data.get("horizon_calendar_days", 0.0)),
        )


@dataclass(frozen=True)
class PythonDimTimeSlice:
    time: float
    date: str
    days_in_period: int
    numeraire: np.ndarray
    ir_delta: np.ndarray
    ir_vega: np.ndarray
    fx_delta: np.ndarray
    fx_vega: np.ndarray
    flow: np.ndarray | None = None

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        currencies: int,
        ir_delta_terms: int,
        ir_vega_terms: int,
        fx_risk_factors: int,
        fx_vega_terms: int,
    ) -> "PythonDimTimeSlice":
        numeraire = _as_np(data["numeraire"], ndim=1)
        samples = numeraire.shape[0]
        ir_delta = _as_np(data["ir_delta"], ndim=3)
        ir_vega = _as_np(data["ir_vega"], ndim=3)
        fx_delta = _as_np(data["fx_delta"], ndim=2)
        fx_vega = _as_np(data["fx_vega"], ndim=3)
        if ir_delta.shape != (currencies, ir_delta_terms, samples):
            raise EngineRunError(f"ir_delta shape {ir_delta.shape} does not match {(currencies, ir_delta_terms, samples)}")
        if ir_vega.shape != (currencies, ir_vega_terms, samples):
            raise EngineRunError(f"ir_vega shape {ir_vega.shape} does not match {(currencies, ir_vega_terms, samples)}")
        if fx_delta.shape != (fx_risk_factors, samples):
            raise EngineRunError(f"fx_delta shape {fx_delta.shape} does not match {(fx_risk_factors, samples)}")
        if fx_vega.shape != (fx_risk_factors, fx_vega_terms, samples):
            raise EngineRunError(f"fx_vega shape {fx_vega.shape} does not match {(fx_risk_factors, fx_vega_terms, samples)}")
        flow = None
        if data.get("flow") is not None:
            flow = _as_np(data["flow"], ndim=1)
            if flow.shape[0] != samples:
                raise EngineRunError(f"flow shape {flow.shape} does not match sample size {samples}")
        return cls(
            time=float(data.get("time", 0.0)),
            date=str(data.get("date", "")),
            days_in_period=int(data.get("days_in_period", 0)),
            numeraire=numeraire,
            ir_delta=ir_delta,
            ir_vega=ir_vega,
            fx_delta=fx_delta,
            fx_vega=fx_vega,
            flow=flow,
        )


@dataclass(frozen=True)
class PythonVarDimCurrentSlice:
    covariance: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    theta: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PythonVarDimCurrentSlice":
        covariance = _as_np(data["covariance"], ndim=2)
        delta = _as_np(data["delta"], ndim=1)
        gamma = _as_np(data["gamma"], ndim=2)
        if covariance.shape[0] != covariance.shape[1]:
            raise EngineRunError(f"covariance must be square, got {covariance.shape}")
        if gamma.shape != covariance.shape:
            raise EngineRunError(f"gamma shape {gamma.shape} does not match covariance {covariance.shape}")
        if delta.shape[0] != covariance.shape[0]:
            raise EngineRunError(f"delta shape {delta.shape} does not match covariance {covariance.shape}")
        return cls(
            covariance=covariance,
            delta=delta,
            gamma=gamma,
            theta=float(data.get("theta", 0.0)),
        )


@dataclass(frozen=True)
class PythonVarDimTimeSlice:
    time: float
    date: str
    days_in_period: int
    numeraire: np.ndarray
    covariance: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    flow: np.ndarray | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PythonVarDimTimeSlice":
        numeraire = _as_np(data["numeraire"], ndim=1)
        covariance = _as_np(data["covariance"], ndim=2)
        delta = _as_np(data["delta"], ndim=2)
        gamma = _as_np(data["gamma"], ndim=3)
        theta = _as_np(data.get("theta", np.zeros(numeraire.shape[0], dtype=float)), ndim=1)
        samples = numeraire.shape[0]
        if covariance.shape[0] != covariance.shape[1]:
            raise EngineRunError(f"covariance must be square, got {covariance.shape}")
        factors = covariance.shape[0]
        if delta.shape != (factors, samples):
            raise EngineRunError(f"delta shape {delta.shape} does not match {(factors, samples)}")
        if gamma.shape != (factors, factors, samples):
            raise EngineRunError(f"gamma shape {gamma.shape} does not match {(factors, factors, samples)}")
        if theta.shape[0] != samples:
            raise EngineRunError(f"theta shape {theta.shape} does not match sample size {samples}")
        flow = None
        if data.get("flow") is not None:
            flow = _as_np(data["flow"], ndim=1)
            if flow.shape[0] != samples:
                raise EngineRunError(f"flow shape {flow.shape} does not match sample size {samples}")
        return cls(
            time=float(data.get("time", 0.0)),
            date=str(data.get("date", "")),
            days_in_period=int(data.get("days_in_period", 0)),
            numeraire=numeraire,
            covariance=covariance,
            delta=delta,
            gamma=gamma,
            theta=theta,
            flow=flow,
        )


@dataclass(frozen=True)
class PythonDimNettingSetInput:
    netting_set_id: str
    current_slice: PythonDimTimeSlice
    time_slices: tuple[PythonDimTimeSlice, ...]
    regression: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PythonVarDimNettingSetInput:
    netting_set_id: str
    current_slice: PythonVarDimCurrentSlice
    time_slices: tuple[PythonVarDimTimeSlice, ...]
    current_im: float | None = None
    regression: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PythonDimInput:
    currencies: tuple[str, ...]
    ir_delta_terms: tuple[str, ...]
    ir_vega_terms: tuple[str, ...]
    fx_vega_terms: tuple[str, ...]
    simm_config: PythonSimpleDynamicSimmConfig
    netting_sets: tuple[PythonDimNettingSetInput, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PythonDimInput":
        currencies = _period_labels(data["currencies"])
        ir_delta_terms = _period_labels(data["ir_delta_terms"])
        ir_vega_terms = _period_labels(data["ir_vega_terms"])
        fx_vega_terms = _period_labels(data["fx_vega_terms"])
        config = PythonSimpleDynamicSimmConfig.from_dict(data["simm_config"])
        fx_risk_factors = max(len(currencies) - 1, 0)
        netting_sets: List[PythonDimNettingSetInput] = []
        for netting_set_id, payload in dict(data["netting_sets"]).items():
            current_slice = PythonDimTimeSlice.from_dict(
                payload["current_slice"],
                currencies=len(currencies),
                ir_delta_terms=len(ir_delta_terms),
                ir_vega_terms=len(ir_vega_terms),
                fx_risk_factors=fx_risk_factors,
                fx_vega_terms=len(fx_vega_terms),
            )
            time_slices = tuple(
                PythonDimTimeSlice.from_dict(
                    item,
                    currencies=len(currencies),
                    ir_delta_terms=len(ir_delta_terms),
                    ir_vega_terms=len(ir_vega_terms),
                    fx_risk_factors=fx_risk_factors,
                    fx_vega_terms=len(fx_vega_terms),
                )
                for item in payload.get("time_slices", ())
            )
            netting_sets.append(
                PythonDimNettingSetInput(
                    netting_set_id=str(netting_set_id),
                    current_slice=current_slice,
                    time_slices=time_slices,
                    regression=dict(payload.get("regression", {})),
                )
            )
        return cls(
            currencies=currencies,
            ir_delta_terms=ir_delta_terms,
            ir_vega_terms=ir_vega_terms,
            fx_vega_terms=fx_vega_terms,
            simm_config=config,
            netting_sets=tuple(netting_sets),
        )


@dataclass(frozen=True)
class PythonVarDimInput:
    var_config: PythonVarDimConfig
    netting_sets: tuple[PythonVarDimNettingSetInput, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PythonVarDimInput":
        config = PythonVarDimConfig.from_dict(data["var_config"])
        netting_sets: List[PythonVarDimNettingSetInput] = []
        for netting_set_id, payload in dict(data["netting_sets"]).items():
            netting_sets.append(
                PythonVarDimNettingSetInput(
                    netting_set_id=str(netting_set_id),
                    current_slice=PythonVarDimCurrentSlice.from_dict(payload["current_slice"]),
                    time_slices=tuple(PythonVarDimTimeSlice.from_dict(item) for item in payload.get("time_slices", ())),
                    current_im=float(payload["current_im"]) if payload.get("current_im") is not None else None,
                    regression=dict(payload.get("regression", {})),
                )
            )
        return cls(var_config=config, netting_sets=tuple(netting_sets))


class PythonSimpleDynamicSimm:
    def __init__(self, config: PythonSimpleDynamicSimmConfig):
        self.config = config

    def value(
        self,
        ir_delta: np.ndarray,
        ir_vega: np.ndarray,
        fx_delta: np.ndarray,
        fx_vega: np.ndarray,
    ) -> DIMMarginComponents:
        delta_margin_ir = self._delta_margin_ir(ir_delta)
        vega_margin_ir = self._vega_margin_ir(ir_vega)
        curvature_margin_ir = self._curvature_margin_ir(ir_vega)
        im_ir = delta_margin_ir + vega_margin_ir + curvature_margin_ir

        delta_margin_fx = self._delta_margin_fx(fx_delta)
        vega_margin_fx = self._vega_margin_fx(fx_vega)
        curvature_margin_fx = self._curvature_margin_fx(fx_vega)
        im_fx = delta_margin_fx + vega_margin_fx + curvature_margin_fx

        total = np.sqrt(im_ir * im_ir + im_fx * im_fx + 2.0 * self.config.corr_ir_fx * im_ir * im_fx)
        return DIMMarginComponents(
            total=total,
            delta_margin_ir=delta_margin_ir,
            vega_margin_ir=vega_margin_ir,
            curvature_margin_ir=curvature_margin_ir,
            delta_margin_fx=delta_margin_fx,
            vega_margin_fx=vega_margin_fx,
            curvature_margin_fx=curvature_margin_fx,
            delta_margin=delta_margin_ir + delta_margin_fx,
            vega_margin=vega_margin_ir + vega_margin_fx,
            curvature_margin=curvature_margin_ir + curvature_margin_fx,
            ir_delta_margin=delta_margin_ir,
            fx_delta_margin=delta_margin_fx,
        )

    def _delta_margin_ir(self, ir_delta: np.ndarray) -> np.ndarray:
        kbs = []
        sbs = []
        for ccy in range(ir_delta.shape[0]):
            kb = np.zeros(ir_delta.shape[2], dtype=float)
            sb = np.zeros(ir_delta.shape[2], dtype=float)
            for i in range(ir_delta.shape[1]):
                tmp = self.config.ir_delta_rw[i] * ir_delta[ccy, i]
                kb += tmp * tmp
                sb += tmp
                for j in range(i):
                    tmp2 = self.config.ir_delta_rw[j] * ir_delta[ccy, j]
                    kb += 2.0 * self.config.ir_delta_correlations[i, j] * tmp * tmp2
            kb = np.sqrt(np.maximum(kb, 0.0))
            sb = np.clip(sb, -kb, kb)
            kbs.append(kb)
            sbs.append(sb)
        return self._aggregate_buckets(kbs, sbs, self.config.ir_gamma)

    def _vega_margin_ir(self, ir_vega: np.ndarray) -> np.ndarray:
        kbs = []
        sbs = []
        for ccy in range(ir_vega.shape[0]):
            kb = np.zeros(ir_vega.shape[2], dtype=float)
            sb = np.zeros(ir_vega.shape[2], dtype=float)
            for i in range(ir_vega.shape[1]):
                tmp = self.config.ir_vega_rw * ir_vega[ccy, i]
                kb += tmp * tmp
                sb += tmp
                for j in range(i):
                    tmp2 = self.config.ir_vega_rw * ir_vega[ccy, j]
                    kb += 2.0 * self.config.ir_vega_correlations[i, j] * tmp * tmp2
            kb = np.sqrt(np.maximum(kb, 0.0))
            sb = np.clip(sb, -kb, kb)
            kbs.append(kb)
            sbs.append(sb)
        return self._aggregate_buckets(kbs, sbs, self.config.ir_gamma)

    def _curvature_margin_ir(self, ir_vega: np.ndarray) -> np.ndarray:
        return self._curvature_margin(
            values=ir_vega,
            weights=self.config.ir_curvature_weights,
            correlations=self.config.ir_vega_correlations,
            inter_bucket_corr=self.config.ir_gamma,
            scaling=self.config.ir_curvature_scaling,
            extra_scale=1.0,
        )

    def _delta_margin_fx(self, fx_delta: np.ndarray) -> np.ndarray:
        kbs = [self.config.fx_delta_rw * fx_delta[i] for i in range(fx_delta.shape[0])]
        total = np.zeros(fx_delta.shape[1], dtype=float)
        for i in range(len(kbs)):
            total += kbs[i] * kbs[i]
            for j in range(i):
                total += 2.0 * self.config.fx_corr * kbs[i] * kbs[j]
        return np.sqrt(np.maximum(total, 0.0))

    def _vega_margin_fx(self, fx_vega: np.ndarray) -> np.ndarray:
        kbs = []
        sbs = []
        scale = self.config.fx_vega_rw * self.config.fx_sigma * self.config.fx_hvr
        for ccy in range(fx_vega.shape[0]):
            kb = np.zeros(fx_vega.shape[2], dtype=float)
            sb = np.zeros(fx_vega.shape[2], dtype=float)
            for i in range(fx_vega.shape[1]):
                tmp = scale * fx_vega[ccy, i]
                kb += tmp * tmp
                sb += tmp
                for j in range(i):
                    tmp2 = scale * fx_vega[ccy, j]
                    kb += 2.0 * self.config.fx_vega_correlations[i, j] * tmp * tmp2
            kb = np.sqrt(np.maximum(kb, 0.0))
            sb = np.clip(sb, -kb, kb)
            kbs.append(kb)
            sbs.append(sb)
        return self._aggregate_buckets(kbs, sbs, self.config.fx_corr)

    def _curvature_margin_fx(self, fx_vega: np.ndarray) -> np.ndarray:
        return self._curvature_margin(
            values=fx_vega,
            weights=self.config.fx_curvature_weights,
            correlations=self.config.fx_vega_correlations,
            inter_bucket_corr=self.config.fx_corr,
            scaling=1.0,
            extra_scale=self.config.fx_sigma,
        )

    def _curvature_margin(
        self,
        *,
        values: np.ndarray,
        weights: np.ndarray,
        correlations: np.ndarray,
        inter_bucket_corr: float,
        scaling: float,
        extra_scale: float,
    ) -> np.ndarray:
        kbs = []
        sbs = []
        s = np.zeros(values.shape[2], dtype=float)
        sabs = np.zeros(values.shape[2], dtype=float)
        for bucket in range(values.shape[0]):
            kb = np.zeros(values.shape[2], dtype=float)
            sb = np.zeros(values.shape[2], dtype=float)
            for i in range(values.shape[1]):
                tmp = weights[i] * extra_scale * values[bucket, i]
                kb += tmp * tmp
                sb += tmp
                s += tmp
                sabs += np.abs(tmp)
                for j in range(i):
                    tmp2 = weights[j] * extra_scale * values[bucket, j]
                    kb += 2.0 * correlations[i, j] * correlations[i, j] * tmp * tmp2
            kb = np.sqrt(np.maximum(kb, 0.0))
            sb = np.clip(sb, -kb, kb)
            kbs.append(kb)
            sbs.append(sb)
        curvature = np.zeros(values.shape[2], dtype=float)
        for i in range(len(kbs)):
            curvature += kbs[i] * kbs[i]
            for j in range(i):
                curvature += 2.0 * inter_bucket_corr * inter_bucket_corr * sbs[i] * sbs[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(s, sabs, out=np.zeros_like(s), where=sabs != 0.0)
        theta = np.minimum(0.0, ratio)
        lamb = 5.634896601 * (1.0 + theta) - theta
        return np.maximum(0.0, s + lamb * np.sqrt(np.maximum(curvature, 0.0))) * scaling

    def _aggregate_buckets(self, kbs: list[np.ndarray], sbs: list[np.ndarray], gamma: float) -> np.ndarray:
        total = np.zeros_like(kbs[0]) if kbs else np.array([], dtype=float)
        for i in range(len(kbs)):
            total += kbs[i] * kbs[i]
            for j in range(i):
                total += 2.0 * gamma * sbs[i] * sbs[j]
        return np.sqrt(np.maximum(total, 0.0))


class PythonSimmHelper:
    def __init__(self, dim_input: PythonDimInput):
        self.dim_input = dim_input
        self.im_calculator = PythonSimpleDynamicSimm(dim_input.simm_config)
        self._components: DIMMarginComponents | None = None

    def initial_margin(self, time_slice: PythonDimTimeSlice) -> np.ndarray:
        self._components = self.im_calculator.value(
            time_slice.ir_delta,
            time_slice.ir_vega,
            time_slice.fx_delta,
            time_slice.fx_vega,
        )
        return self._components.total

    @property
    def components(self) -> DIMMarginComponents:
        if self._components is None:
            raise EngineRunError("DIM components are not available before initial_margin()")
        return self._components


class PythonDeltaGammaVarHelper:
    def __init__(self, config: PythonVarDimConfig, order: int):
        if order not in (1, 2, 3):
            raise EngineRunError(f"Unsupported delta/gamma VaR order {order}")
        self.config = config
        self.order = int(order)
        self._normal = NormalDist()

    def value(self, covariance: np.ndarray, delta: np.ndarray, gamma: np.ndarray, theta: float = 0.0) -> float:
        gamma_is_zero = bool(np.allclose(gamma, 0.0))
        if self.order == 1 or gamma_is_zero:
            res = self._delta_var(covariance, delta, self.config.quantile)
        elif self.order == 2:
            res = self._delta_gamma_var_normal(covariance, delta, gamma, self.config.quantile)
        else:
            res = self._delta_gamma_var_cornish_fisher(covariance, delta, gamma, self.config.quantile)
        return float(res + float(theta) * self.config.theta_factor)

    def _delta_var(self, omega: np.ndarray, delta: np.ndarray, p: float) -> float:
        num = _max_abs(delta)
        if _is_close_zero(num):
            return 0.0
        tmp_delta = delta / num
        variance = float(tmp_delta @ omega @ tmp_delta)
        if variance <= 0.0:
            return 0.0
        return float(np.sqrt(variance) * self._normal.inv_cdf(p) * num)

    def _delta_gamma_var_normal(self, omega: np.ndarray, delta: np.ndarray, gamma: np.ndarray, p: float) -> float:
        num, mu, variance, _, _ = self._moments(omega, delta, gamma, with_higher=False)
        if _is_close_zero(num) or variance <= 0.0:
            return 0.0
        return float((np.sqrt(variance) * self._normal.inv_cdf(p) + mu) * num)

    def _delta_gamma_var_cornish_fisher(
        self,
        omega: np.ndarray,
        delta: np.ndarray,
        gamma: np.ndarray,
        p: float,
    ) -> float:
        num, mu, variance, tau, kappa = self._moments(omega, delta, gamma, with_higher=True)
        if _is_close_zero(num) or variance <= 0.0:
            return 0.0
        s = self._normal.inv_cdf(p)
        x_tilde = (
            s
            + tau / 6.0 * (s * s - 1.0)
            + kappa / 24.0 * s * (s * s - 3.0)
            - tau * tau / 36.0 * s * (2.0 * s * s - 5.0)
        )
        return float((x_tilde * np.sqrt(variance) + mu) * num)

    def _moments(
        self,
        omega: np.ndarray,
        delta: np.ndarray,
        gamma: np.ndarray,
        *,
        with_higher: bool,
    ) -> tuple[float, float, float, float, float]:
        num = max(_max_abs(delta), _max_abs(gamma))
        if _is_close_zero(num):
            return 0.0, 0.0, 0.0, 0.0, 0.0

        tmp_delta = delta / num
        tmp_gamma = gamma / num

        d_od = float(tmp_delta @ omega @ tmp_delta)
        go = tmp_gamma @ omega
        go2 = go @ go
        mu = 0.5 * float(np.trace(go))
        variance = d_od + 0.5 * float(np.trace(go2))
        if not with_higher or variance <= 0.0:
            return num, mu, variance, 0.0, 0.0

        go3 = go2 @ go
        go4 = go2 @ go2
        ogo = omega @ go
        o_go2 = omega @ go2
        tau = (float(np.trace(go3)) + 3.0 * float(tmp_delta @ ogo @ tmp_delta)) / (variance ** 1.5)
        kappa = (3.0 * float(np.trace(go4)) + 12.0 * float(tmp_delta @ o_go2 @ tmp_delta)) / (variance * variance)
        return num, mu, variance, tau, kappa


class PythonDynamicSimmCalculator:
    def __init__(self, dim_input: PythonDimInput):
        self.dim_input = dim_input
        self.simm_helper = PythonSimmHelper(dim_input)

    def build(self) -> DIMResult:
        dim_cube_payload: Dict[str, Dict[str, Any]] = {}
        reports: Dict[str, Any] = {}
        current_dim: Dict[str, float] = {}
        dim_evolution_rows: List[Dict[str, Any]] = []
        dim_distribution_rows: List[Dict[str, Any]] = []
        dim_regression_rows: Dict[str, Any] = {}

        for netting_set in self.dim_input.netting_sets:
            current = self.simm_helper.initial_margin(netting_set.current_slice)
            current_components = self.simm_helper.components
            current_dim[netting_set.netting_set_id] = float(np.mean(current))

            times = []
            average_dim = []
            average_flow = []
            cube_rows: List[Dict[str, Any]] = []
            distributions: List[Dict[str, Any]] = []

            cube_rows.extend(
                _depth_rows(
                    netting_set=netting_set.netting_set_id,
                    sample_count=netting_set.current_slice.numeraire.shape[0],
                    date=netting_set.current_slice.date,
                    time=0.0,
                    components=current_components,
                )
            )

            for time_step, slice_ in enumerate(netting_set.time_slices):
                margin = self.simm_helper.initial_margin(slice_)
                components = self.simm_helper.components
                discounted_margin = np.divide(
                    margin,
                    slice_.numeraire,
                    out=np.zeros_like(margin),
                    where=slice_.numeraire != 0.0,
                )
                flow = np.zeros_like(discounted_margin) if slice_.flow is None else slice_.flow
                avg_dim = float(np.mean(discounted_margin))
                avg_flow = float(np.mean(flow))
                times.append(float(slice_.time))
                average_dim.append(avg_dim)
                average_flow.append(avg_flow)
                dim_evolution_rows.append(
                    {
                        "TimeStep": time_step,
                        "Date": slice_.date,
                        "DaysInPeriod": int(slice_.days_in_period),
                        "AverageDIM": avg_dim,
                        "AverageFLOW": avg_flow,
                        "NettingSet": netting_set.netting_set_id,
                        "Time": float(slice_.time),
                    }
                )
                distributions.extend(
                    _distribution_rows(
                        netting_set=netting_set.netting_set_id,
                        time_step=time_step,
                        date=slice_.date,
                        values=discounted_margin,
                    )
                )
                cube_rows.extend(
                    _depth_rows(
                        netting_set=netting_set.netting_set_id,
                        sample_count=slice_.numeraire.shape[0],
                        date=slice_.date,
                        time=float(slice_.time),
                        components=components,
                        numeraires=slice_.numeraire,
                    )
                )

            if netting_set.regression:
                dim_regression_rows[netting_set.netting_set_id] = list(netting_set.regression.get("rows", []))

            dim_cube_payload[netting_set.netting_set_id] = {
                "times": times,
                "average_dim": average_dim,
                "average_flow": average_flow,
                "rows": cube_rows,
            }
            dim_distribution_rows.extend(distributions)

        reports["dim_evolution"] = dim_evolution_rows
        reports["dim_distribution"] = dim_distribution_rows
        reports["dim_cube"] = [row for payload in dim_cube_payload.values() for row in payload["rows"]]
        if dim_regression_rows:
            reports["dim_regression"] = dim_regression_rows

        cubes = {"dim_cube": CubeAccessor(name="dim_cube", payload=dim_cube_payload)}
        return DIMResult(current_dim=current_dim, reports=reports, cubes=cubes, metadata={"engine": "python-dim"})


class PythonDynamicDeltaVarCalculator:
    def __init__(self, dim_input: PythonVarDimInput, order: int):
        self.dim_input = dim_input
        self.var_helper = PythonDeltaGammaVarHelper(dim_input.var_config, order=order)

    def build(self) -> DIMResult:
        dim_cube_payload: Dict[str, Dict[str, Any]] = {}
        reports: Dict[str, Any] = {}
        current_dim: Dict[str, float] = {}
        dim_evolution_rows: List[Dict[str, Any]] = []
        dim_distribution_rows: List[Dict[str, Any]] = []
        dim_regression_rows: Dict[str, Any] = {}

        for netting_set in self.dim_input.netting_sets:
            current = self.var_helper.value(
                netting_set.current_slice.covariance,
                netting_set.current_slice.delta,
                netting_set.current_slice.gamma,
                netting_set.current_slice.theta,
            )
            current_dim[netting_set.netting_set_id] = current
            scaling = 1.0
            if netting_set.current_im is not None:
                if _is_close_zero(current):
                    raise EngineRunError(
                        f"Netting set {netting_set.netting_set_id} has current_im but zero current DIM; cannot scale."
                    )
                scaling = float(netting_set.current_im) / current

            times = []
            average_dim = []
            average_flow = []
            cube_rows: List[Dict[str, Any]] = []
            distributions: List[Dict[str, Any]] = []

            cube_rows.extend(
                _depth_rows(
                    netting_set=netting_set.netting_set_id,
                    sample_count=1,
                    date=netting_set.time_slices[0].date if netting_set.time_slices else "",
                    time=0.0,
                    components=_zero_components(np.asarray([current], dtype=float)),
                )
            )

            for time_step, slice_ in enumerate(netting_set.time_slices):
                margin = np.zeros(slice_.numeraire.shape[0], dtype=float)
                for sample in range(slice_.numeraire.shape[0]):
                    margin[sample] = self.var_helper.value(
                        slice_.covariance,
                        slice_.delta[:, sample],
                        slice_.gamma[:, :, sample],
                        slice_.theta[sample],
                    )
                margin *= scaling
                discounted_margin = np.divide(
                    margin,
                    slice_.numeraire,
                    out=np.zeros_like(margin),
                    where=slice_.numeraire != 0.0,
                )
                flow = np.zeros_like(discounted_margin) if slice_.flow is None else slice_.flow
                avg_dim = float(np.mean(discounted_margin))
                avg_flow = float(np.mean(flow))
                times.append(float(slice_.time))
                average_dim.append(avg_dim)
                average_flow.append(avg_flow)
                dim_evolution_rows.append(
                    {
                        "TimeStep": time_step,
                        "Date": slice_.date,
                        "DaysInPeriod": int(slice_.days_in_period),
                        "AverageDIM": avg_dim,
                        "AverageFLOW": avg_flow,
                        "NettingSet": netting_set.netting_set_id,
                        "Time": float(slice_.time),
                    }
                )
                distributions.extend(
                    _distribution_rows(
                        netting_set=netting_set.netting_set_id,
                        time_step=time_step,
                        date=slice_.date,
                        values=discounted_margin,
                    )
                )
                cube_rows.extend(
                    _depth_rows(
                        netting_set=netting_set.netting_set_id,
                        sample_count=slice_.numeraire.shape[0],
                        date=slice_.date,
                        time=float(slice_.time),
                        components=_zero_components(margin),
                        numeraires=slice_.numeraire,
                    )
                )

            if netting_set.regression:
                dim_regression_rows[netting_set.netting_set_id] = list(netting_set.regression.get("rows", []))

            dim_cube_payload[netting_set.netting_set_id] = {
                "times": times,
                "average_dim": average_dim,
                "average_flow": average_flow,
                "rows": cube_rows,
            }
            dim_distribution_rows.extend(distributions)

        reports["dim_evolution"] = dim_evolution_rows
        reports["dim_distribution"] = dim_distribution_rows
        reports["dim_cube"] = [row for payload in dim_cube_payload.values() for row in payload["rows"]]
        if dim_regression_rows:
            reports["dim_regression"] = dim_regression_rows

        cubes = {"dim_cube": CubeAccessor(name="dim_cube", payload=dim_cube_payload)}
        return DIMResult(current_dim=current_dim, reports=reports, cubes=cubes, metadata={"engine": "python-dim"})


def calculate_python_dim(snapshot_params: Mapping[str, Any], dim_model: str | None = None) -> DIMResult:
    raw = snapshot_params.get("python.dim_feeder")
    if raw is None:
        raise EngineRunError("DIM mode is enabled but snapshot.config.params['python.dim_feeder'] is missing")

    model = str(dim_model or raw.get("dim_model") or "").strip()
    if model in {"DynamicIM", "SimmAnalytic"} or (not model and "simm_config" in raw):
        return PythonDynamicSimmCalculator(PythonDimInput.from_dict(raw)).build()
    if model in {"DeltaVaR", "DeltaGammaNormalVaR", "DeltaGammaVaR"} or (not model and "var_config" in raw):
        order = {"DeltaVaR": 1, "DeltaGammaNormalVaR": 2, "DeltaGammaVaR": 3, "": 1}[model]
        return PythonDynamicDeltaVarCalculator(PythonVarDimInput.from_dict(raw), order=order).build()

    raise EngineRunError(
        f"Unsupported Python DIM model '{model or '<unspecified>'}'. "
        "Supported models are DynamicIM, SimmAnalytic, DeltaVaR, DeltaGammaNormalVaR, DeltaGammaVaR."
    )


def _distribution_rows(
    *,
    netting_set: str,
    time_step: int,
    date: str,
    values: np.ndarray,
    grid_size: int = 50,
    covered_std_devs: float = 5.0,
) -> list[dict[str, Any]]:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return []
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if covered_std_devs > 0.0 and std > 0.0:
        lower = mean - covered_std_devs * std
        upper = mean + covered_std_devs * std
    else:
        lower = float(np.min(vals))
        upper = float(np.max(vals))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.min(vals))
        upper = float(np.max(vals)) + 1.0e-12
    edges = np.linspace(lower, upper, grid_size + 1)
    counts, _ = np.histogram(vals, bins=edges)
    bounds = edges[1:]
    rows = []
    for bound, count in zip(bounds, counts):
        rows.append(
            {
                "NettingSet": netting_set,
                "TimeStep": time_step,
                "Date": date,
                "Bound": float(bound),
                "Count": int(count),
            }
        )
    return rows


def _depth_rows(
    *,
    netting_set: str,
    sample_count: int,
    date: str,
    time: float,
    components: DIMMarginComponents,
    numeraires: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    numeraires = np.ones(sample_count, dtype=float) if numeraires is None else np.asarray(numeraires, dtype=float)
    if (
        np.allclose(components.delta_margin, 0.0)
        and np.allclose(components.vega_margin, 0.0)
        and np.allclose(components.curvature_margin, 0.0)
    ):
        depth_values = [components.total]
    else:
        depth_values = [
            components.total,
            components.delta_margin,
            components.vega_margin,
            components.curvature_margin,
            components.ir_delta_margin,
            components.fx_delta_margin,
        ]

    rows: list[dict[str, Any]] = []
    for sample in range(sample_count):
        for depth, values in enumerate(depth_values):
            value = float(values[sample] / numeraires[sample]) if numeraires[sample] != 0.0 else 0.0
            rows.append(
                {
                    "Portfolio": netting_set,
                    "Sample": sample,
                    "AsOfDate": date,
                    "Time": float(time),
                    "InitialMargin": value,
                    "Currency": "",
                    "SimmSide": "Call",
                    "Depth": depth,
                }
            )
    return rows
