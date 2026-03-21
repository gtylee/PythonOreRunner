"""Minimal ORE-style 1F LGM simulation utilities.

This module is the lowest-level "rates model kernel" used by the other helpers in
 ``py_ore_tools``.  The intent is not to re-implement all of ORE, but to mirror the
 small subset of ORE's single-factor LGM machinery that is needed by the example
 notebooks and parity checks:

- piecewise-constant alpha / kappa parametrisations compatible with ORE LGM setup
- the state variable ``x(t)`` used in ORE's Gaussian rates representation
- the deterministic transforms ``H(t)`` and ``zeta(t)`` that drive bond pricing
- simulation under both the LGM measure and the BA measure
- pathwise zero-bond / numeraire identities reused by swap, FX and option helpers

The broader relationship to ORE is:
- ORE provides the trade definitions, market data conventions and calibrated model
  parameters
- this module provides a lightweight NumPy implementation of the same analytical
  identities so the surrounding Python tooling can simulate and price using inputs
  extracted from ORE XML / CSV artefacts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]
DiscountInput = Union[float, np.ndarray, Callable[[float], float]]
PiecewiseSpec = Union[float, Tuple[Iterable[float], Iterable[float]], Mapping[str, Iterable[float]]]

ORE_PARITY_SEQUENCE_TYPE = "MersenneTwister"
ORE_PARITY_ANTITHETIC_SEQUENCE_TYPE = "MersenneTwisterAntithetic"
ORE_SOBOL_SEQUENCE_TYPE = "Sobol"
ORE_SOBOL_BROWNIAN_BRIDGE_SEQUENCE_TYPE = "SobolBrownianBridge"


def _as_1d_float_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_piecewise(times: np.ndarray, values: np.ndarray, name: str) -> None:
    if times.size > 0:
        if np.any(times <= 0.0):
            raise ValueError(f"{name}_times must be strictly positive")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{name}_times must be strictly increasing")
    if values.size != times.size + 1:
        raise ValueError(
            f"{name}_values size must equal {name}_times size + 1, got "
            f"{values.size} vs {times.size + 1}"
        )


def _piecewise_value(times: np.ndarray, values: np.ndarray, t: ArrayLike) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    if np.any(~np.isfinite(t_arr)):
        raise ValueError("time input contains non-finite values")
    if np.any(t_arr < 0.0):
        raise ValueError("time input must be non-negative")
    idx = np.searchsorted(times, t_arr, side="right")
    return values[idx]


def _resolve_discount(value: DiscountInput, t: float) -> float:
    if callable(value):
        out = float(value(t))
    else:
        out = float(np.asarray(value, dtype=float))
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError("discount factor must be finite and positive")
    return out


def _resolve_discount_vector(value: DiscountInput, t: np.ndarray) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim != 1:
        raise ValueError("discount times must be one-dimensional")
    if callable(value):
        out = np.fromiter((float(value(float(x))) for x in t_arr), dtype=float, count=t_arr.size)
    else:
        out = np.asarray(value, dtype=float)
        if out.ndim == 0:
            out = np.full(t_arr.size, float(out), dtype=float)
        elif out.shape != t_arr.shape:
            raise ValueError("discount factors must match the maturity vector shape")
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("discount factor must be finite and positive")
    return out


def _validate_time_input(t: ArrayLike) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    if np.any(~np.isfinite(t_arr)) or np.any(t_arr < 0.0):
        raise ValueError("time input must be finite and non-negative")
    return t_arr


def _coerce_seed(seed: int) -> int:
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be non-negative")
    return seed_int


def _load_quantlib():
    try:
        import QuantLib as ql
    except ImportError as exc:
        raise ImportError(
            "Ore parity mode requires the QuantLib Python bindings to be installed"
        ) from exc
    return ql


class OreMersenneTwisterGaussianRng:
    """QuantLib-backed Gaussian generator matching Ore's pseudo-random stream."""

    def __init__(self, seed: int) -> None:
        self.seed = _coerce_seed(seed)
        self._dimension: Optional[int] = None
        self._generator = None

    def _ensure_dimension(self, size: int) -> None:
        size = int(size)
        if size <= 0:
            raise ValueError("size must be positive")
        if self._dimension is None:
            ql = _load_quantlib()
            uniform = ql.MersenneTwisterUniformRsg(size, self.seed)
            self._generator = ql.InvCumulativeMersenneTwisterGaussianRsg(uniform)
            self._dimension = size
            return
        if size != self._dimension:
            raise ValueError(
                f"OreMersenneTwisterGaussianRng was initialised with dimension {self._dimension}, got {size}"
            )

    def next_sequence(self, size: int) -> np.ndarray:
        self._ensure_dimension(size)
        assert self._generator is not None
        return np.asarray(self._generator.nextSequence().value(), dtype=float)

    def standard_normal(self, size: Union[int, tuple[int, ...]]) -> np.ndarray:
        if isinstance(size, tuple):
            if len(size) != 1:
                raise ValueError("OreMersenneTwisterGaussianRng only supports one-dimensional draws")
            size = size[0]
        return self.next_sequence(int(size))


class OreMersenneTwisterAntitheticGaussianRng:
    """Path-major MT Gaussian generator with simple antithetic pairing."""

    def __init__(self, seed: int) -> None:
        self._base = OreMersenneTwisterGaussianRng(seed)
        self._pending_negative: Optional[np.ndarray] = None

    def next_sequence(self, size: int) -> np.ndarray:
        if self._pending_negative is not None:
            out = self._pending_negative
            self._pending_negative = None
            return out.copy()
        draws = self._base.next_sequence(size)
        self._pending_negative = -draws
        return draws

    def standard_normal(self, size: Union[int, tuple[int, ...]]) -> np.ndarray:
        if isinstance(size, tuple):
            if len(size) != 1:
                raise ValueError(
                    "OreMersenneTwisterAntitheticGaussianRng only supports one-dimensional draws"
                )
            size = size[0]
        return self.next_sequence(int(size))


class OreSobolGaussianRng:
    """QuantLib-backed Sobol Gaussian generator for path-major LGM simulation.

    This uses ``SobolRsg`` followed by inverse-Gaussian transformation. It matches
    the sequence family ORE uses more closely than pseudo-random MT draws, but it
    does not apply Brownian-bridge reordering yet.
    """

    def __init__(self, seed: int) -> None:
        self.seed = _coerce_seed(seed)
        self._dimension: Optional[int] = None
        self._generator = None

    def _ensure_dimension(self, size: int) -> None:
        size = int(size)
        if size <= 0:
            raise ValueError("size must be positive")
        if self._dimension is None:
            ql = _load_quantlib()
            uniform = ql.SobolRsg(size, self.seed)
            self._generator = ql.InvCumulativeSobolGaussianRsg(uniform)
            self._dimension = size
            return
        if size != self._dimension:
            raise ValueError(
                f"OreSobolGaussianRng was initialised with dimension {self._dimension}, got {size}"
            )

    def next_sequence(self, size: int) -> np.ndarray:
        self._ensure_dimension(size)
        assert self._generator is not None
        return np.asarray(self._generator.nextSequence().value(), dtype=float)

    def standard_normal(self, size: Union[int, tuple[int, ...]]) -> np.ndarray:
        if isinstance(size, tuple):
            if len(size) != 1:
                raise ValueError("OreSobolGaussianRng only supports one-dimensional draws")
            size = size[0]
        return self.next_sequence(int(size))


class OreSobolBrownianBridgeGaussianRng:
    """QuantLib-backed Sobol Gaussian generator with Brownian-bridge rotation."""

    def __init__(self, seed: int) -> None:
        self.seed = _coerce_seed(seed)
        self._dimension: Optional[int] = None
        self._generator = None
        self._bridge = None
        self._bridge_times: Optional[tuple[float, ...]] = None

    def _ensure_dimension(self, size: int) -> None:
        size = int(size)
        if size <= 0:
            raise ValueError("size must be positive")
        if self._dimension is None:
            ql = _load_quantlib()
            uniform = ql.SobolRsg(size, self.seed)
            self._generator = ql.InvCumulativeSobolGaussianRsg(uniform)
            self._bridge = ql.BrownianBridge(list(self._bridge_times)) if self._bridge_times is not None else ql.BrownianBridge(size)
            self._dimension = size
            return
        if size != self._dimension:
            raise ValueError(
                "OreSobolBrownianBridgeGaussianRng was initialised with "
                f"dimension {self._dimension}, got {size}"
            )

    def configure_time_grid(self, times: Iterable[float]) -> None:
        times_arr = tuple(float(x) for x in times)
        if len(times_arr) == 0:
            self._bridge_times = None
        else:
            self._bridge_times = times_arr
        if self._dimension is not None and self._generator is not None:
            if self._bridge_times is not None and len(self._bridge_times) != self._dimension:
                raise ValueError(
                    "configured bridge time grid length must match existing RNG dimension "
                    f"{self._dimension}, got {len(self._bridge_times)}"
                )
            ql = _load_quantlib()
            self._bridge = (
                ql.BrownianBridge(list(self._bridge_times))
                if self._bridge_times is not None
                else ql.BrownianBridge(self._dimension)
            )

    def next_sequence(self, size: int) -> np.ndarray:
        self._ensure_dimension(size)
        assert self._generator is not None
        assert self._bridge is not None
        seq = np.asarray(self._generator.nextSequence().value(), dtype=float)
        return np.asarray(self._bridge.transform(seq.tolist()), dtype=float)

    def standard_normal(self, size: Union[int, tuple[int, ...]]) -> np.ndarray:
        if isinstance(size, tuple):
            if len(size) != 1:
                raise ValueError(
                    "OreSobolBrownianBridgeGaussianRng only supports one-dimensional draws"
                )
            size = size[0]
        return self.next_sequence(int(size))


def make_ore_gaussian_rng(
    seed: int, sequence_type: str = ORE_PARITY_SEQUENCE_TYPE
) -> Union[
    OreMersenneTwisterGaussianRng,
    OreMersenneTwisterAntitheticGaussianRng,
    OreSobolGaussianRng,
    OreSobolBrownianBridgeGaussianRng,
]:
    """Build a QuantLib-backed Gaussian RNG for Ore-style path-major simulation."""
    if sequence_type == ORE_PARITY_SEQUENCE_TYPE:
        return OreMersenneTwisterGaussianRng(seed)
    if sequence_type == ORE_PARITY_ANTITHETIC_SEQUENCE_TYPE:
        return OreMersenneTwisterAntitheticGaussianRng(seed)
    if sequence_type == ORE_SOBOL_SEQUENCE_TYPE:
        return OreSobolGaussianRng(seed)
    if sequence_type == ORE_SOBOL_BROWNIAN_BRIDGE_SEQUENCE_TYPE:
        return OreSobolBrownianBridgeGaussianRng(seed)
    raise ValueError(
        f"unsupported sequence_type '{sequence_type}', expected one of "
        f"'{ORE_PARITY_SEQUENCE_TYPE}', '{ORE_PARITY_ANTITHETIC_SEQUENCE_TYPE}', "
        f"'{ORE_SOBOL_SEQUENCE_TYPE}' or "
        f"'{ORE_SOBOL_BROWNIAN_BRIDGE_SEQUENCE_TYPE}'"
    )


@dataclass(frozen=True)
class LGMParams:
    """Container matching the core ORE LGM parameter blocks.

    ``alpha_*`` corresponds to the piecewise Hagan volatility parametrisation and
    ``kappa_*`` to the piecewise mean-reversion / transformation driving ``H``.
    ``shift`` and ``scaling`` mirror ORE's parameter transformation settings.
    """

    alpha_times: Tuple[float, ...]
    alpha_values: Tuple[float, ...]
    kappa_times: Tuple[float, ...]
    kappa_values: Tuple[float, ...]
    shift: float = 0.0
    scaling: float = 1.0

    @staticmethod
    def _coerce_piecewise(name: str, spec: PiecewiseSpec) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        if np.isscalar(spec):
            value = float(spec)
            return (), (value,)

        if isinstance(spec, Mapping):
            if "times" not in spec or "values" not in spec:
                raise ValueError(f"{name} mapping spec must include 'times' and 'values'")
            times = tuple(float(x) for x in spec["times"])
            values = tuple(float(x) for x in spec["values"])
            return times, values

        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            times = tuple(float(x) for x in spec[0])
            values = tuple(float(x) for x in spec[1])
            return times, values

        raise ValueError(
            f"invalid {name} spec; use scalar, (times, values), or "
            "{'times': [...], 'values': [...]}"
        )

    @classmethod
    def constant(
        cls,
        alpha: float,
        kappa: float,
        shift: float = 0.0,
        scaling: float = 1.0,
    ) -> "LGMParams":
        return cls(
            alpha_times=(),
            alpha_values=(float(alpha),),
            kappa_times=(),
            kappa_values=(float(kappa),),
            shift=float(shift),
            scaling=float(scaling),
        )

    @classmethod
    def from_spec(
        cls,
        alpha: PiecewiseSpec,
        kappa: PiecewiseSpec,
        shift: float = 0.0,
        scaling: float = 1.0,
    ) -> "LGMParams":
        alpha_times, alpha_values = cls._coerce_piecewise("alpha", alpha)
        kappa_times, kappa_values = cls._coerce_piecewise("kappa", kappa)
        return cls(
            alpha_times=alpha_times,
            alpha_values=alpha_values,
            kappa_times=kappa_times,
            kappa_values=kappa_values,
            shift=float(shift),
            scaling=float(scaling),
        )

    def __post_init__(self) -> None:
        alpha_times = _as_1d_float_array(self.alpha_times, "alpha_times")
        alpha_values = _as_1d_float_array(self.alpha_values, "alpha_values")
        kappa_times = _as_1d_float_array(self.kappa_times, "kappa_times")
        kappa_values = _as_1d_float_array(self.kappa_values, "kappa_values")

        _validate_piecewise(alpha_times, alpha_values, "alpha")
        _validate_piecewise(kappa_times, kappa_values, "kappa")

        if np.any(alpha_values < 0.0):
            raise ValueError("alpha_values must be non-negative")
        if not np.isfinite(self.shift):
            raise ValueError("shift must be finite")
        if not np.isfinite(self.scaling) or self.scaling <= 0.0:
            raise ValueError("scaling must be finite and positive")


class LGM1F:
    """Minimal 1-factor LGM representation for simulation and identities.

    The model exposes exactly the objects the higher-level ORE-style helpers need:
    bond prices, numeraires and time-integrated variance quantities.  It deliberately
    stays close to ORE naming so that formulas in notebooks can be compared against
    ORE logs / XML with minimal translation.
    """

    def __init__(self, params: LGMParams, zero_cutoff: float = 1.0e-10) -> None:
        self.params = params
        self.zero_cutoff = float(zero_cutoff)
        if self.zero_cutoff <= 0.0:
            raise ValueError("zero_cutoff must be positive")

        self.alpha_times = _as_1d_float_array(params.alpha_times, "alpha_times")
        self.alpha_values = _as_1d_float_array(params.alpha_values, "alpha_values")
        self.kappa_times = _as_1d_float_array(params.kappa_times, "kappa_times")
        self.kappa_values = _as_1d_float_array(params.kappa_values, "kappa_values")
        self.shift = float(params.shift)
        self.scaling = float(params.scaling)

        # Cache prefix integrals because most downstream routines repeatedly query
        # H(t), zeta(t) and related interval moments on the same piecewise grid.
        self._alpha_prefix_int = self._prefix_integral_square(self.alpha_times, self.alpha_values)
        self._kappa_prefix_int = self._prefix_integral_linear(self.kappa_times, self.kappa_values)
        self._h_prefix_int = self._build_h_prefix_integral()

    @staticmethod
    def _prefix_integral_square(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.zeros(times.size, dtype=float)
        running = 0.0
        for i in range(times.size):
            t0 = 0.0 if i == 0 else times[i - 1]
            dt = times[i] - t0
            a = values[i]
            running += a * a * dt
            out[i] = running
        return out

    @staticmethod
    def _prefix_integral_linear(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.zeros(times.size, dtype=float)
        running = 0.0
        for i in range(times.size):
            t0 = 0.0 if i == 0 else times[i - 1]
            dt = times[i] - t0
            k = values[i]
            running += k * dt
            out[i] = running
        return out

    def _build_h_prefix_integral(self) -> np.ndarray:
        # ``H`` is the ORE transformation that turns the Gaussian state ``x`` into
        # bond price loadings.  Precomputing the integral piece over each constant
        # kappa bucket keeps later bond evaluations cheap.
        out = np.zeros(self.kappa_times.size, dtype=float)
        running = 0.0
        for i in range(self.kappa_times.size):
            s = 0.0 if i == 0 else self.kappa_times[i - 1]
            e = self.kappa_times[i]
            k = self.kappa_values[i]
            base = np.exp(-self._int_kappa_scalar(s))
            if abs(k) < self.zero_cutoff:
                contrib = base * (e - s)
            else:
                contrib = base * (1.0 - np.exp(-k * (e - s))) / k
            running += contrib
            out[i] = running
        return out

    def alpha(self, t: ArrayLike) -> np.ndarray:
        return _piecewise_value(self.alpha_times, self.alpha_values, t) / self.scaling

    def kappa(self, t: ArrayLike) -> np.ndarray:
        return _piecewise_value(self.kappa_times, self.kappa_values, t)

    def _int_piecewise_square(self, times: np.ndarray, values: np.ndarray, prefix: np.ndarray, t: float) -> float:
        if t <= 0.0:
            return 0.0
        i = int(np.searchsorted(times, t, side="right"))
        res = 0.0
        if i >= 1:
            res += prefix[min(i - 1, prefix.size - 1)]
        v = values[min(i, values.size - 1)]
        t0 = 0.0 if i == 0 else times[i - 1]
        res += v * v * (t - t0)
        return res

    def _int_kappa_scalar(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        i = int(np.searchsorted(self.kappa_times, t, side="right"))
        res = 0.0
        if i >= 1:
            res += self._kappa_prefix_int[min(i - 1, self._kappa_prefix_int.size - 1)]
        k = self.kappa_values[min(i, self.kappa_values.size - 1)]
        t0 = 0.0 if i == 0 else self.kappa_times[i - 1]
        res += k * (t - t0)
        return res

    def _int_kappa(self, t: ArrayLike) -> np.ndarray:
        t_arr = _validate_time_input(t)
        flat = t_arr.ravel()
        idx = np.searchsorted(self.kappa_times, flat, side="right")
        out = np.zeros_like(flat)
        mask = idx > 0
        if np.any(mask):
            out[mask] = self._kappa_prefix_int[idx[mask] - 1]
        t0 = np.zeros_like(flat)
        if np.any(mask):
            t0[mask] = self.kappa_times[idx[mask] - 1]
        k = self.kappa_values[idx]
        out += k * (flat - t0)
        return out.reshape(t_arr.shape)

    def zeta(self, t: ArrayLike) -> np.ndarray:
        # ``zeta(t) = int_0^t alpha(u)^2 du`` in this scaled representation.  Under
        # the LGM measure it is also the variance clock for the state ``x(t)``.
        t_arr = _validate_time_input(t)
        flat = t_arr.ravel()
        idx = np.searchsorted(self.alpha_times, flat, side="right")
        out = np.zeros_like(flat)
        mask = idx > 0
        if np.any(mask):
            out[mask] = self._alpha_prefix_int[idx[mask] - 1]
        t0 = np.zeros_like(flat)
        if np.any(mask):
            t0[mask] = self.alpha_times[idx[mask] - 1]
        a = self.alpha_values[idx]
        out += a * a * (flat - t0)
        out /= self.scaling * self.scaling
        return out.reshape(t_arr.shape)

    def Hprime(self, t: ArrayLike) -> np.ndarray:
        t_arr = _validate_time_input(t)
        flat = t_arr.ravel()
        idx = np.searchsorted(self.kappa_times, flat, side="right")
        delta = flat.copy()
        mask = idx > 0
        if np.any(mask):
            delta[mask] -= self.kappa_times[idx[mask] - 1]
        base = np.ones_like(flat)
        if np.any(mask):
            base[mask] = np.exp(-self._kappa_prefix_int[idx[mask] - 1])
        k = self.kappa_values[idx]
        out = self.scaling * base * np.exp(-k * delta)
        return out.reshape(t_arr.shape)

    def H(self, t: ArrayLike) -> np.ndarray:
        # ``H`` is the deterministic loading appearing in the ORE/LGM bond formula
        # P(t,T) = P(0,T)/P(0,t) * exp(-(H(T)-H(t)) x_t - ...).
        t_arr = _validate_time_input(t)
        flat = t_arr.ravel()
        idx = np.searchsorted(self.kappa_times, flat, side="right")
        out = np.zeros_like(flat)
        mask = idx > 0
        if np.any(mask):
            out[mask] = self._h_prefix_int[idx[mask] - 1]
        delta = flat.copy()
        if np.any(mask):
            delta[mask] -= self.kappa_times[idx[mask] - 1]
        base = np.ones_like(flat)
        if np.any(mask):
            base[mask] = np.exp(-self._kappa_prefix_int[idx[mask] - 1])
        k = self.kappa_values[idx]
        small = np.abs(k) < self.zero_cutoff
        tail = np.empty_like(flat)
        tail[small] = base[small] * delta[small]
        tail[~small] = base[~small] * (1.0 - np.exp(-k[~small] * delta[~small])) / k[~small]
        out = self.scaling * (out + tail) + self.shift
        return out.reshape(t_arr.shape)

    def _int_exp_minus_int_kappa_scalar(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        i = int(np.searchsorted(self.kappa_times, t, side="right"))
        res = 0.0
        if i >= 1:
            res += self._h_prefix_int[min(i - 1, self._h_prefix_int.size - 1)]

        s = 0.0 if i == 0 else self.kappa_times[i - 1]
        k = self.kappa_values[min(i, self.kappa_values.size - 1)]
        base = np.exp(-self._int_kappa_scalar(s))
        if abs(k) < self.zero_cutoff:
            res += base * (t - s)
        else:
            res += base * (1.0 - np.exp(-k * (t - s))) / k
        return res

    def _integration_knots(self, t: float, t0: float = 0.0) -> np.ndarray:
        if t < t0:
            raise ValueError("integration end time must be >= start time")
        knots = [t0, t]
        for arr in (self.alpha_times, self.kappa_times):
            if arr.size:
                inner = arr[(arr > t0) & (arr < t)]
                if inner.size:
                    knots.extend(inner.tolist())
        return np.array(sorted(set(knots)), dtype=float)

    def _zetan_interval_numeric(self, n: int, t0: float, t1: float, max_substeps_per_year: int = 64) -> float:
        knots = self._integration_knots(t1, t0)
        total = 0.0
        for i in range(len(knots) - 1):
            a = knots[i]
            b = knots[i + 1]
            seg = b - a
            n_steps = max(2, int(np.ceil(seg * max_substeps_per_year)) + 1)
            grid = np.linspace(a, b, n_steps)
            integrand = np.square(self.alpha(grid)) * np.power(self.H(grid), n)
            total += np.trapezoid(integrand, grid)
        return float(total)

    def _h_increment(self, delta: float, kappa: float) -> float:
        if abs(kappa) < self.zero_cutoff:
            return delta
        return (1.0 - np.exp(-kappa * delta)) / kappa

    def _h_increment_integral(self, delta: float, kappa: float) -> float:
        if abs(kappa) < self.zero_cutoff:
            return 0.5 * delta * delta
        return delta / kappa - (1.0 - np.exp(-kappa * delta)) / (kappa * kappa)

    def _h_increment_square_integral(self, delta: float, kappa: float) -> float:
        if abs(kappa) < self.zero_cutoff:
            return delta * delta * delta / 3.0
        exp_k = np.exp(-kappa * delta)
        exp_2k = exp_k * exp_k
        return (
            delta / (kappa * kappa)
            - 2.0 * (1.0 - exp_k) / (kappa * kappa * kappa)
            + (1.0 - exp_2k) / (2.0 * kappa * kappa * kappa)
        )

    def _zetan_interval_exact(self, n: int, t0: float, t1: float) -> float:
        # The ORE analytics used by BA-measure simulation need integrals of
        # alpha^2 * H^n.  Closed forms are available for n in {0,1,2}; fall back to
        # numerical integration for higher moments that are only used rarely.
        if n not in (0, 1, 2):
            return self._zetan_interval_numeric(n, t0, t1)
        if t1 < t0:
            raise ValueError("integration end time must be >= start time")
        if t1 == t0:
            return 0.0

        knots = self._integration_knots(t1, t0)
        h_curr = float(self.H(t0))
        base = float(self.Hprime(t0)) / self.scaling
        total = 0.0

        for i in range(len(knots) - 1):
            a = knots[i]
            b = knots[i + 1]
            delta = b - a
            alpha_idx = int(np.searchsorted(self.alpha_times, a, side="right"))
            kappa_idx = int(np.searchsorted(self.kappa_times, a, side="right"))
            alpha = self.alpha_values[alpha_idx] / self.scaling
            kappa = self.kappa_values[kappa_idx]
            alpha2 = alpha * alpha
            c = self.scaling * base

            if n == 0:
                total += alpha2 * delta
            else:
                j1 = self._h_increment_integral(delta, kappa)
                if n == 1:
                    total += alpha2 * (h_curr * delta + c * j1)
                else:
                    j2 = self._h_increment_square_integral(delta, kappa)
                    total += alpha2 * (h_curr * h_curr * delta + 2.0 * h_curr * c * j1 + c * c * j2)

            h_curr += c * self._h_increment(delta, kappa)
            if abs(kappa) >= self.zero_cutoff:
                base *= np.exp(-kappa * delta)

        return float(total)

    def zetan(self, n: int, t: float, max_substeps_per_year: int = 64) -> float:
        if n < 0:
            raise ValueError("n must be non-negative")
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("t must be finite and non-negative")
        if t == 0.0:
            return 0.0
        if n in (0, 1, 2):
            return self._zetan_interval_exact(n, 0.0, t)
        return self._zetan_interval_numeric(n, 0.0, t, max_substeps_per_year=max_substeps_per_year)

    def zetan_grid(self, n: int, times: Iterable[float]) -> np.ndarray:
        times_arr = _validate_time_grid(times)
        out = np.empty(times_arr.size, dtype=float)
        out[0] = self.zetan(n, float(times_arr[0]))
        for i in range(1, times_arr.size):
            out[i] = out[i - 1] + self._zetan_interval_exact(n, float(times_arr[i - 1]), float(times_arr[i]))
        return out

    def integral_alpha2(self, t0: float, t1: float) -> float:
        return float(self.zeta(t1) - self.zeta(t0))

    def integral_alpha2_H(self, t0: float, t1: float) -> float:
        return self._zetan_interval_exact(1, t0, t1)

    def integral_alpha2_H2(self, t0: float, t1: float) -> float:
        return self._zetan_interval_exact(2, t0, t1)

    def ba_interval_moments(self, t0: float, t1: float) -> Tuple[float, float, float, float]:
        # Under the BA measure, ORE evolves an auxiliary state y(t) jointly with x(t).
        # These interval moments are the exact Gaussian moments needed to simulate
        # that pair without Euler discretisation bias.
        if t1 < t0:
            raise ValueError("t1 must be >= t0")
        if t1 == t0:
            return 0.0, 0.0, 0.0, 0.0
        var_x = self.integral_alpha2(t0, t1)
        cov_xy = self.integral_alpha2_H(t0, t1)
        var_y = self.integral_alpha2_H2(t0, t1)
        mean_dx = -cov_xy
        return mean_dx, var_x, cov_xy, var_y

    def discount_bond(self, t: float, T: float, x_t: ArrayLike, p0_t: DiscountInput, p0_T: DiscountInput) -> np.ndarray:
        # Core ORE/LGM identity reused everywhere else in this package.  Higher-level
        # trade pricers feed in deterministic t=0 discount factors from curves
        # exported by ORE and stochastic state x_t from the simulated model.
        if T < t or t < 0.0:
            raise ValueError("require T >= t >= 0")
        if np.isclose(T, t):
            return np.ones_like(np.asarray(x_t, dtype=float))
        h_t = float(self.H(t))
        h_T = float(self.H(T))
        z_t = float(self.zeta(t))
        p_ratio = _resolve_discount(p0_T, T) / _resolve_discount(p0_t, t)
        x = np.asarray(x_t, dtype=float)
        return p_ratio * np.exp(-(h_T - h_t) * x - 0.5 * (h_T * h_T - h_t * h_t) * z_t)

    def discount_bond_paths(
        self,
        t: float,
        T: Iterable[float],
        x_t: ArrayLike,
        p0_t: DiscountInput,
        p0_T: DiscountInput,
    ) -> np.ndarray:
        if t < 0.0:
            raise ValueError("require T >= t >= 0")
        T_arr = _as_1d_float_array(T, "T")
        if np.any(T_arr < t):
            raise ValueError("require T >= t >= 0")
        x = np.asarray(x_t, dtype=float)
        if T_arr.size == 0:
            return np.empty((0,) + x.shape, dtype=float)

        h_t = float(self.H(t))
        z_t = float(self.zeta(t))
        h_T = np.asarray(self.H(T_arr), dtype=float)
        p_ratio = _resolve_discount_vector(p0_T, T_arr) / _resolve_discount(p0_t, t)
        d_h = h_T - h_t
        d_h2 = h_T * h_T - h_t * h_t
        return p_ratio[:, None] * np.exp(-d_h[:, None] * x - 0.5 * d_h2[:, None] * z_t)

    def numeraire_lgm(self, t: float, x_t: ArrayLike, p0_t: DiscountInput) -> np.ndarray:
        if t < 0.0:
            raise ValueError("t must be non-negative")
        h_t = float(self.H(t))
        z_t = float(self.zeta(t))
        p_t = _resolve_discount(p0_t, t)
        x = np.asarray(x_t, dtype=float)
        return np.exp(h_t * x + 0.5 * h_t * h_t * z_t) / p_t

    def numeraire_ba(self, t: float, x_t: ArrayLike, y_t: ArrayLike, p0_t: DiscountInput) -> np.ndarray:
        if t < 0.0:
            raise ValueError("t must be non-negative")
        h_t = float(self.H(t))
        z_t = float(self.zeta(t))
        v_t = 0.5 * (h_t * h_t * z_t + self.zetan(2, t))
        p_t = _resolve_discount(p0_t, t)
        x = np.asarray(x_t, dtype=float)
        y = np.asarray(y_t, dtype=float)
        return np.exp(h_t * x - y + v_t) / p_t


def _validate_time_grid(times: Iterable[float]) -> np.ndarray:
    t = _as_1d_float_array(times, "times")
    if t.size == 0:
        raise ValueError("times must be non-empty")
    if np.any(t < 0.0):
        raise ValueError("times must be non-negative")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return t


def simulate_lgm_measure(
    model: LGM1F,
    times: Iterable[float],
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    x0: float = 0.0,
    draw_order: str = "time_major",
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate the ORE LGM state ``x(t)`` on a time grid.

    Because ``x`` is Gaussian with variance increments given by ``zeta``, the exact
    transition is just a Brownian increment with no time-stepping approximation.
    """
    times_arr = _validate_time_grid(times)
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if draw_order not in ("time_major", "ore_path_major"):
        raise ValueError("draw_order must be 'time_major' or 'ore_path_major'")
    if rng is None and draw_order == "ore_path_major":
        raise ValueError("draw_order='ore_path_major' requires an explicit rng")
    if antithetic and draw_order != "time_major":
        raise ValueError("antithetic=True is only supported for draw_order='time_major'")
    if rng is None:
        rng = np.random.default_rng()

    x = np.empty((times_arr.size, n_paths), dtype=float)
    x[0, :] = float(x0)
    # ``zeta`` is the exact cumulative variance under the LGM measure, so its
    # increments are the path simulation variances on each interval.
    zeta_grid = model.zeta(times_arr)
    var_increments = np.diff(zeta_grid)
    if draw_order == "time_major":
        half_paths = (n_paths + 1) // 2 if antithetic else n_paths
        for i, var in enumerate(var_increments):
            if var < -1.0e-14:
                raise ValueError("encountered negative variance increment")
            var = max(var, 0.0)
            draws = rng.standard_normal(half_paths)
            if antithetic:
                draws = np.concatenate((draws, -draws), axis=0)[:n_paths]
            x[i + 1, :] = x[i, :] + np.sqrt(var) * draws
        return x

    if not hasattr(rng, "next_sequence"):
        raise TypeError("draw_order='ore_path_major' requires an rng with a next_sequence(size) method")
    step_scales = np.empty_like(var_increments)
    for i, var in enumerate(var_increments):
        if var < -1.0e-14:
            raise ValueError("encountered negative variance increment")
        step_scales[i] = np.sqrt(max(var, 0.0))
    if hasattr(rng, "configure_time_grid"):
        rng.configure_time_grid(times_arr[1:])
    for p in range(n_paths):
        draws = np.asarray(rng.next_sequence(step_scales.size), dtype=float)
        if draws.shape != step_scales.shape:
            raise ValueError("rng.next_sequence returned an unexpected shape")
        x_curr = float(x0)
        for i, scale in enumerate(step_scales):
            x_curr += scale * draws[i]
            x[i + 1, p] = x_curr
    return x


def _sample_correlated_2d(
    mean_dx: float,
    var_x: float,
    cov_xy: float,
    var_y: float,
    n_paths: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    var_x = max(var_x, 0.0)
    var_y = max(var_y, 0.0)
    z1 = rng.standard_normal(n_paths)
    if var_x <= 0.0:
        dx = np.full(n_paths, mean_dx, dtype=float)
        if var_y <= 0.0:
            return dx, np.zeros(n_paths, dtype=float)
        return dx, np.sqrt(var_y) * rng.standard_normal(n_paths)

    l11 = np.sqrt(var_x)
    l21 = cov_xy / l11
    l22_sq = max(var_y - l21 * l21, 0.0)
    z2 = rng.standard_normal(n_paths)
    dx = mean_dx + l11 * z1
    dy = l21 * z1 + np.sqrt(l22_sq) * z2
    return dx, dy


def simulate_ba_measure(
    model: LGM1F,
    times: Iterable[float],
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    x0: float = 0.0,
    y0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the BA-measure state pair used by some ORE analytical formulas."""
    times_arr = _validate_time_grid(times)
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if rng is None:
        rng = np.random.default_rng()

    x = np.empty((times_arr.size, n_paths), dtype=float)
    y = np.empty((times_arr.size, n_paths), dtype=float)
    x[0, :] = float(x0)
    y[0, :] = float(y0)

    zeta_grid = model.zeta(times_arr)
    zetan_1 = model.zetan_grid(1, times_arr)
    zetan_2 = model.zetan_grid(2, times_arr)
    mean_dx = -np.diff(zetan_1)
    var_x = np.diff(zeta_grid)
    cov_xy = np.diff(zetan_1)
    var_y = np.diff(zetan_2)

    for i in range(times_arr.size - 1):
        dx, dy = _sample_correlated_2d(mean_dx[i], var_x[i], cov_xy[i], var_y[i], n_paths, rng)
        x[i + 1, :] = x[i, :] + dx
        y[i + 1, :] = y[i, :] + dy

    return x, y


__all__ = [
    "LGMParams",
    "LGM1F",
    "ORE_PARITY_SEQUENCE_TYPE",
    "OreMersenneTwisterGaussianRng",
    "make_ore_gaussian_rng",
    "simulate_lgm_measure",
    "simulate_ba_measure",
]
