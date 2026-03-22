#!/usr/bin/env python3
"""Diagnose ORE vs Python Sobol-Brownian scenario coupling on a benchmark case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pythonore.io.ore_snapshot import load_from_ore_xml
from pythonore.compute.lgm import make_ore_gaussian_rng
from pythonore.compute.lgm import _load_quantlib
from pythonore.workflows.ore_snapshot_cli import _simulate_with_fixing_grid


def _load_ore_numeraire_surface(output_dir: Path, n_paths: int, n_dates: int) -> np.ndarray:
    out = np.zeros((n_dates, n_paths), dtype=float)
    path = output_dir / "scenariodata.csv.gz"
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(row for row in handle if not row.startswith("#"))
        next(reader)
        for date_idx, sample, key, value in reader:
            if key != "1":
                continue
            sample_idx = int(sample)
            if sample_idx >= n_paths:
                continue
            out[int(date_idx) - 1, sample_idx] = float(value)
    return out


def _load_ore_index_surface(output_dir: Path, n_paths: int, n_dates: int) -> np.ndarray:
    out = np.full((n_dates, n_paths), np.nan, dtype=float)
    path = output_dir / "scenariodata.csv.gz"
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(row for row in handle if not row.startswith("#"))
        next(reader)
        for date_idx, sample, key, value in reader:
            if key != "0":
                continue
            sample_idx = int(sample)
            if sample_idx >= n_paths:
                continue
            out[int(date_idx) - 1, sample_idx] = float(value)
    return out


def _rank_positions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    return order, pos


def _recover_state_from_numeraire(model, p0_disc, times: np.ndarray, numeraire: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numeraire)
    for i, t in enumerate(times):
        h_t = float(model.H(float(t)))
        if abs(h_t) <= 1.0e-15:
            continue
        z_t = float(model.zeta(float(t)))
        p0_t = float(p0_disc(float(t)))
        out[i, :] = (np.log(numeraire[i, :] * p0_t) - 0.5 * h_t * h_t * z_t) / h_t
    return out


def _probe_indices(n_dates: int) -> list[int]:
    candidates = [0, 1, 11, 23, 59, 119]
    return [i for i in candidates if i < n_dates]


def _discount_bond_block(model, p0, t: float, maturities: np.ndarray, x_t: np.ndarray, p_t: float) -> np.ndarray:
    p_T = np.asarray([float(p0(float(T))) for T in maturities], dtype=float)
    return model.discount_bond_paths(float(t), maturities, x_t, p_t, p_T)


def _compute_index_proxies(snap, model, sim_times: np.ndarray, x_paths: np.ndarray) -> dict[str, np.ndarray]:
    exposure_times = np.asarray(snap.exposure_model_times[1:], dtype=float)
    n_dates = exposure_times.size
    n_paths = x_paths.shape[1]
    out = {
        "next_live_coupon": np.full((n_dates, n_paths), np.nan, dtype=float),
        "spot_forward": np.full((n_dates, n_paths), np.nan, dtype=float),
        "spot_forward_plus_spread": np.full((n_dates, n_paths), np.nan, dtype=float),
        "ibor_fixing_forward": np.full((n_dates, n_paths), np.nan, dtype=float),
    }
    fix_t = np.asarray(snap.legs.get("float_fixing_time", []), dtype=float)
    s = np.asarray(snap.legs.get("float_start_time", []), dtype=float)
    e = np.asarray(snap.legs.get("float_end_time", []), dtype=float)
    index_tau = np.asarray(snap.legs.get("float_index_accrual", snap.legs.get("float_accrual", [])), dtype=float)
    spread = np.asarray(snap.legs.get("float_spread", np.zeros_like(index_tau)), dtype=float)
    if index_tau.size == 0:
        return out

    tenor = float(np.median(index_tau))
    spread_level = float(np.median(spread)) if spread.size else 0.0
    ql = _load_quantlib()
    dummy_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.TARGET(), 0.0, ql.Actual365Fixed())
    )
    index_name = str(getattr(snap, "forward_column", "") or "")
    if index_name == "EUR-EURIBOR-6M":
        ibor = ql.Euribor6M(dummy_curve)
    elif index_name == "EUR-EURIBOR-3M":
        ibor = ql.Euribor3M(dummy_curve)
    elif index_name == "USD-LIBOR-3M":
        ibor = ql.USDLibor(ql.Period("3M"), dummy_curve)
    elif index_name == "USD-LIBOR-6M":
        ibor = ql.USDLibor(ql.Period("6M"), dummy_curve)
    elif index_name == "GBP-LIBOR-3M":
        ibor = ql.GBPLibor(ql.Period("3M"), dummy_curve)
    elif index_name == "GBP-LIBOR-6M":
        ibor = ql.GBPLibor(ql.Period("6M"), dummy_curve)
    else:
        ibor = None
    for i, t in enumerate(exposure_times):
        x_t = x_paths[i + 1, :]
        j = int(np.searchsorted(fix_t, t, side="left"))
        if j < index_tau.size:
            out["next_live_coupon"][i, :] = float(snap.legs["float_coupon"][j])
        p_t_f = float(snap.p0_fwd(float(t)))
        start = np.full(1, float(t), dtype=float)
        end = np.full(1, float(t + tenor), dtype=float)
        p_ts = _discount_bond_block(model, snap.p0_fwd, float(t), start, x_t, p_t_f)[0, :]
        p_te = _discount_bond_block(model, snap.p0_fwd, float(t), end, x_t, p_t_f)[0, :]
        fwd = (p_ts / p_te - 1.0) / tenor
        out["spot_forward"][i, :] = fwd
        out["spot_forward_plus_spread"][i, :] = fwd + spread_level
        if ibor is not None:
            asof = ql.DateParser.parseISO(str(snap.exposure_dates[i + 1]))
            value_date = ibor.valueDate(asof)
            maturity_date = ibor.maturityDate(value_date)
            start_t = float(snap.model_time_from_date(value_date.ISO()))
            end_t = float(snap.model_time_from_date(maturity_date.ISO()))
            tau_q = float(ibor.dayCounter().yearFraction(value_date, maturity_date))
            p_start = _discount_bond_block(model, snap.p0_fwd, float(t), np.full(1, start_t, dtype=float), x_t, p_t_f)[0, :]
            p_end = _discount_bond_block(model, snap.p0_fwd, float(t), np.full(1, end_t, dtype=float), x_t, p_t_f)[0, :]
            out["ibor_fixing_forward"][i, :] = (p_start / p_end - 1.0) / tau_q
    return out


def _surface_stats(ref: np.ndarray, py: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ref) & np.isfinite(py)
    if not np.any(mask):
        return {"mean_abs": float("nan"), "p95_abs": float("nan")}
    absd = np.abs(py[mask] - ref[mask])
    return {
        "mean_abs": float(np.mean(absd)),
        "p95_abs": float(np.percentile(absd, 95.0)),
    }


def _composite_score(result: dict[str, object]) -> dict[str, float]:
    numeraire = result["numeraire"]
    increments = result["increments"]
    index_surface = result["index_surface"]["spot_forward"]
    components = {
        "numeraire_direct_p95_abs": float(numeraire["direct_p95_abs"]),
        "numeraire_sorted_p95_abs": float(numeraire["sorted_p95_abs"]),
        "index_spot_forward_p95_abs": float(index_surface["p95_abs"]),
        "increments_p95_abs": float(increments["p95_abs"]),
    }
    return {
        "total": float(sum(components.values())),
        "components": components,
    }


class _SobolBridgeVariantRng:
    def __init__(self, mode: str, seed: int) -> None:
        self.mode = str(mode)
        self.seed = int(seed)
        self._dimension: int | None = None
        self._generator = None
        self._bridge = None
        self._bridge_times: tuple[float, ...] | None = None

    def configure_time_grid(self, times: Iterable[float]) -> None:
        self._bridge_times = tuple(float(x) for x in times)

    def _ensure_dimension(self, size: int) -> None:
        size = int(size)
        if self._dimension is not None:
            if size != self._dimension:
                raise ValueError(f"dimension mismatch: expected {self._dimension}, got {size}")
            return
        ql = _load_quantlib()
        if self.mode == "joekuod7_steps":
            sobol = ql.SobolRsg(size, 0, ql.SobolRsg.JoeKuoD7)
            self._generator = ql.InvCumulativeSobolGaussianRsg(sobol)
            self._bridge = ql.BrownianBridge(size)
        elif self.mode == "jaeckel_steps":
            sobol = ql.SobolRsg(size, 0, ql.SobolRsg.Jaeckel)
            self._generator = ql.InvCumulativeSobolGaussianRsg(sobol)
            self._bridge = ql.BrownianBridge(size)
        elif self.mode == "joekuod7_times":
            sobol = ql.SobolRsg(size, 0, ql.SobolRsg.JoeKuoD7)
            self._generator = ql.InvCumulativeSobolGaussianRsg(sobol)
            bridge_times = self._bridge_times if self._bridge_times is not None else tuple(float(i + 1) for i in range(size))
            self._bridge = ql.BrownianBridge(list(bridge_times))
        elif self.mode == "direct_bridge_rsg":
            self._generator = ql.SobolBrownianBridgeRsg(1, size)
            self._bridge = None
        else:
            raise ValueError(f"unsupported mode '{self.mode}'")
        self._dimension = size

    def next_sequence(self, size: int) -> np.ndarray:
        self._ensure_dimension(size)
        if self.mode == "direct_bridge_rsg":
            return np.asarray(self._generator.nextSequence().value(), dtype=float)
        return np.asarray(self._bridge.transform(self._generator.nextSequence().value()), dtype=float)


def _build_rng(mode: str, seed: int):
    if mode == "current":
        return make_ore_gaussian_rng(seed, sequence_type="SobolBrownianBridge")
    return _SobolBridgeVariantRng(mode, seed)


def _evaluate_mode(snap, model, ore_num: np.ndarray, ore_idx: np.ndarray, n_paths: int, seed: int, mode: str) -> dict[str, object]:
    rng = _build_rng(mode, seed)
    x, _, sim_times, _, _ = _simulate_with_fixing_grid(
        model=model,
        exposure_times=snap.exposure_model_times,
        fixing_times=np.asarray(snap.legs.get("float_fixing_time", []), dtype=float),
        n_paths=n_paths,
        rng=rng,
        draw_order="ore_path_major",
    )
    py_num = np.vstack(
        [
            model.numeraire_lgm(float(t), x[i, :], snap.p0_disc)
            for i, t in enumerate(snap.exposure_model_times[1:], start=1)
        ]
    )
    py_x = x[1:, :]
    index_proxies = _compute_index_proxies(snap, model, sim_times, x)
    ore_x = _recover_state_from_numeraire(model, snap.p0_disc, snap.exposure_model_times[1:], ore_num)

    ore_order0 = np.argsort(ore_num[0])
    py_order0 = np.argsort(py_num[0])
    perm = np.empty(n_paths, dtype=int)
    perm[ore_order0] = py_order0
    py_num_aligned = py_num[:, perm]
    py_x_aligned = py_x[:, perm]

    probe_rows = []
    _, base_pos_ore = _rank_positions(ore_num[0])
    _, base_pos_py = _rank_positions(py_num[0])
    for di in _probe_indices(py_num.shape[0]):
        order_ore, pos_ore = _rank_positions(ore_num[di])
        order_py, pos_py = _rank_positions(py_num[di])
        probe_rows.append(
            {
                "date": str(snap.exposure_dates[di + 1]),
                "time": float(snap.exposure_times[di + 1]),
                "direct_rank_corr": float(np.corrcoef(pos_ore, pos_py)[0, 1]),
                "ore_rank_corr_vs_first_date": float(np.corrcoef(base_pos_ore, pos_ore)[0, 1]),
                "py_rank_corr_vs_first_date": float(np.corrcoef(base_pos_py, pos_py)[0, 1]),
                "top10_overlap": int(np.intersect1d(order_ore[-10:], order_py[-10:]).size),
                "bottom10_overlap": int(np.intersect1d(order_ore[:10], order_py[:10]).size),
                "aligned_x_corr": float(np.corrcoef(py_x_aligned[di], ore_x[di])[0, 1]),
                "aligned_x_mean_abs": float(np.mean(np.abs(py_x_aligned[di] - ore_x[di]))),
                "aligned_num_mean_abs": float(np.mean(np.abs(py_num_aligned[di] - ore_num[di]))),
            }
        )

    sorted_num_abs = np.abs(np.sort(py_num, axis=1) - np.sort(ore_num, axis=1))
    direct_num_abs = np.abs(py_num - ore_num)
    py_dx = np.diff(py_x_aligned, axis=0)
    ore_dx = np.diff(ore_x, axis=0)
    dx_abs = np.abs(py_dx - ore_dx)
    order_ore_all = np.argsort(ore_num, axis=1)
    order_py_all = np.argsort(py_num, axis=1)
    ore_adj = [
        float(np.corrcoef(order_ore_all[i], order_ore_all[i + 1])[0, 1])
        for i in range(order_ore_all.shape[0] - 1)
    ]
    py_adj = [
        float(np.corrcoef(order_py_all[i], order_py_all[i + 1])[0, 1])
        for i in range(order_py_all.shape[0] - 1)
    ]
    return {
        "mode": mode,
        "numeraire": {
            "direct_mean_abs": float(direct_num_abs.mean()),
            "direct_p95_abs": float(np.percentile(direct_num_abs, 95.0)),
            "sorted_mean_abs": float(sorted_num_abs.mean()),
            "sorted_p95_abs": float(np.percentile(sorted_num_abs, 95.0)),
        },
        "increments": {
            "mean_abs": float(dx_abs.mean()),
            "p95_abs": float(np.percentile(dx_abs, 95.0)),
            "corr_first_interval": float(np.corrcoef(py_dx[0], ore_dx[0])[0, 1]),
            "corr_interval_12": float(np.corrcoef(py_dx[min(11, py_dx.shape[0] - 1)], ore_dx[min(11, ore_dx.shape[0] - 1)])[0, 1]),
        },
        "rank_dynamics": {
            "adjacent_rank_corr_mean_ore": float(np.mean(ore_adj)),
            "adjacent_rank_corr_mean_py": float(np.mean(py_adj)),
            "adjacent_rank_corr_p5_ore": float(np.percentile(ore_adj, 5.0)),
            "adjacent_rank_corr_p5_py": float(np.percentile(py_adj, 5.0)),
        },
        "index_surface": {
            name: _surface_stats(ore_idx, values) for name, values in index_proxies.items()
        },
        "composite": {},
        "probe_rows": probe_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ore_xml", type=Path)
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        default=None,
        help="RNG construction mode to test. Repeatable. Defaults to current, joekuod7_steps, jaeckel_steps, joekuod7_times, direct_bridge_rsg.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    snap = load_from_ore_xml(args.ore_xml, anchor_t0_npv=False)
    model = snap.build_model()
    setattr(model, "_measure", str(getattr(snap, "measure", "LGM")).upper())

    ore_num = _load_ore_numeraire_surface(
        args.ore_xml.parent.parent / "Output",
        n_paths=args.paths,
        n_dates=max(snap.exposure_model_times.size - 1, 0),
    )
    ore_idx = _load_ore_index_surface(
        args.ore_xml.parent.parent / "Output",
        n_paths=args.paths,
        n_dates=max(snap.exposure_model_times.size - 1, 0),
    )
    modes = args.modes or ["current", "joekuod7_steps", "jaeckel_steps", "joekuod7_times", "direct_bridge_rsg"]
    results = [_evaluate_mode(snap, model, ore_num, ore_idx, args.paths, args.seed, mode) for mode in modes]
    for row in results:
        row["composite"] = _composite_score(row)
    ranking = [
        {"mode": row["mode"], "total": row["composite"]["total"]}
        for row in sorted(results, key=lambda item: item["composite"]["total"])
    ]

    payload = {
        "ore_xml": str(args.ore_xml.resolve()),
        "paths": int(args.paths),
        "seed": int(args.seed),
        "ranking": ranking,
        "results": results,
    }

    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
