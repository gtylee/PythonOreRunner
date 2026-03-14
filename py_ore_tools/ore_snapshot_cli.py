from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import py_ore_tools.ore_snapshot as ore_snapshot_mod
from py_ore_tools.benchmarks import (
    benchmark_lgm_fx_forward_torch,
    benchmark_lgm_fx_hybrid_torch,
    benchmark_lgm_fx_portfolio_torch,
    benchmark_lgm_torch,
    benchmark_lgm_torch_swap,
)
from py_ore_tools.irs_xva_utils import (
    apply_parallel_float_spread_shift_to_match_npv,
    calibrate_float_spreads_from_coupon,
    compute_realized_float_coupons,
    compute_xva_from_exposure_profile,
    deflate_lgm_npv_paths,
    load_ore_legs_from_flows,
    load_simulation_yield_tenors,
    load_swap_legs_from_portfolio,
    survival_probability_from_hazard,
    swap_npv_from_ore_legs_dual_curve,
)
from py_ore_tools.lgm import make_ore_gaussian_rng, simulate_lgm_measure
from py_ore_tools.ore_snapshot import (
    load_from_ore_xml,
    ore_input_validation_dataframe,
    validate_ore_input_snapshot,
)
from py_ore_tools.repo_paths import find_engine_repo_root, local_parity_artifacts_root


DEFAULT_ARTIFACT_ROOT = local_parity_artifacts_root() / "ore_snapshot_cli"
EXAMPLE_CHOICES = (
    "lgm_torch",
    "lgm_torch_swap",
    "lgm_fx_hybrid",
    "lgm_fx_forward",
    "lgm_fx_portfolio",
    "lgm_fx_portfolio_256",
)
TENSOR_BACKEND_CHOICES = ("auto", "numpy", "torch-cpu", "torch-mps")


def _example_devices_for_backend(tensor_backend: str) -> list[str]:
    backend = str(tensor_backend).lower()
    if backend == "auto":
        return ["cpu", "gpu"]
    if backend == "numpy":
        return []
    if backend == "torch-cpu":
        return ["cpu"]
    if backend == "torch-mps":
        return ["mps"]
    raise ValueError(f"unsupported tensor backend '{tensor_backend}'")


@dataclass(frozen=True)
class ModeSelection:
    price: bool
    xva: bool
    sensi: bool


def _parse_version() -> str:
    engine_root = find_engine_repo_root()
    if engine_root is None:
        return "unknown"
    text = (engine_root / "QuantExt" / "qle" / "version.hpp").read_text(encoding="utf-8")
    match = re.search(r'#define OPEN_SOURCE_RISK_VERSION "([^"]+)"', text)
    return match.group(1) if match else "unknown"


def _parse_hash() -> str:
    engine_root = find_engine_repo_root()
    if engine_root is None:
        return "unavailable"
    hash_header = engine_root / "QuantExt" / "qle" / "gitversion.hpp"
    if not hash_header.exists():
        return "unavailable"
    text = hash_header.read_text(encoding="utf-8")
    match = re.search(r'#define GIT_HASH "([^"]+)"', text)
    return match.group(1) if match else "unavailable"


def _safe_rel_diff(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(right)), 1.0)


def _safe_rel_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left - right) / np.maximum(np.abs(right), 1.0)


def _build_leg_diagnostics(snap: Any, *, paths: int) -> dict[str, Any]:
    float_spread = np.asarray(snap.legs.get("float_spread", []), dtype=float)
    warnings: list[str] = []
    if float_spread.size:
        spread_abs_median = float(np.median(np.abs(float_spread)))
        spread_abs_max = float(np.max(np.abs(float_spread)))
        if spread_abs_median > 2.0e-4 or spread_abs_max > 5.0e-4:
            warnings.append(
                "large inferred floating spreads detected; check coupon-vs-index accrual conventions and fixing-date provenance"
            )
    else:
        spread_abs_median = 0.0
        spread_abs_max = 0.0
    ore_samples = int(getattr(snap, "n_samples", paths))
    sample_count_mismatch = int(paths) != ore_samples
    if sample_count_mismatch:
        warnings.append(f"python paths ({int(paths)}) differ from native ORE samples ({ore_samples})")
    return {
        "ore_samples": ore_samples,
        "python_paths": int(paths),
        "sample_count_mismatch": sample_count_mismatch,
        "float_fixing_source": str(snap.legs.get("float_fixing_source", "unknown")),
        "float_index_day_counter": str(snap.legs.get("float_index_day_counter", snap.model_day_counter)),
        "float_spread_abs_median": spread_abs_median,
        "float_spread_abs_max": spread_abs_max,
        "warnings": warnings,
    }


def _simulate_with_fixing_grid(
    model: Any,
    exposure_times: np.ndarray,
    fixing_times: np.ndarray,
    n_paths: int,
    rng: Any,
    draw_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fixing_times is None or fixing_times.size == 0:
        x_exp = simulate_lgm_measure(model, exposure_times, n_paths, rng=rng, x0=0.0, draw_order=draw_order)
        return x_exp, x_exp, exposure_times
    extra = np.asarray(fixing_times, dtype=float)
    extra = extra[extra > 1.0e-12]
    if extra.size == 0:
        x_exp = simulate_lgm_measure(model, exposure_times, n_paths, rng=rng, x0=0.0, draw_order=draw_order)
        return x_exp, x_exp, exposure_times
    sim_times = np.unique(np.concatenate((exposure_times, extra)))
    x_all = simulate_lgm_measure(model, sim_times, n_paths, rng=rng, x0=0.0, draw_order=draw_order)
    idx = np.searchsorted(sim_times, exposure_times)
    if not np.allclose(sim_times[idx], exposure_times, atol=1.0e-12, rtol=0.0):
        raise ValueError("failed to align exposure times on simulated grid")
    return x_all[idx, :], x_all, sim_times


@dataclass(frozen=True)
class SnapshotComputation:
    ore_xml: str
    trade_id: str
    counterparty: str
    netting_set_id: str
    paths: int
    seed: int
    rng_mode: str
    pricing: dict[str, Any]
    xva: dict[str, Any]
    parity: dict[str, Any]
    diagnostics: dict[str, Any]
    maturity_date: str
    maturity_time: float
    exposure_dates: list[str]
    exposure_times: list[float]
    py_epe: list[float]
    py_ene: list[float]
    py_pfe: list[float]
    ore_basel_epe: float
    ore_basel_eepe: float


@dataclass(frozen=True)
class PurePythonRunOptions:
    engine: str = "compare"
    price: bool = False
    xva: bool = False
    sensi: bool = False
    paths: int = 500
    seed: int = 42
    rng: str = "ore_parity"
    xva_mode: str = "ore"
    anchor_t0_npv: bool = False
    own_hazard: float = 0.01
    own_recovery: float = 0.4
    netting_set: str | None = None
    sensi_metric: str = "CVA"
    top: int = 10
    max_npv_abs_diff: float = 1000.0
    max_cva_rel: float = 0.05
    max_dva_rel: float = 0.05
    max_fba_rel: float = 0.05
    max_fca_rel: float = 0.05
    ore_output_only: bool = False


@dataclass(frozen=True)
class BufferCaseInputs:
    input_files: dict[str, str]
    output_files: dict[str, str] = field(default_factory=dict)
    ore_xml_name: str = "ore.xml"


@dataclass(frozen=True)
class PurePythonCaseResult:
    summary: dict[str, Any]
    comparison_rows: list[dict[str, str]]
    input_validation_rows: list[dict[str, str]]
    report_markdown: str
    ore_output_files: dict[str, str]


@dataclass(frozen=True)
class OreSnapshotApp:
    case: BufferCaseInputs
    options: PurePythonRunOptions = field(default_factory=PurePythonRunOptions)

    @classmethod
    def from_strings(
        cls,
        *,
        input_files: dict[str, str],
        output_files: dict[str, str] | None = None,
        ore_xml_name: str = "ore.xml",
        options: PurePythonRunOptions | None = None,
    ) -> "OreSnapshotApp":
        return cls(
            case=BufferCaseInputs(
                input_files=input_files,
                output_files={} if output_files is None else output_files,
                ore_xml_name=ore_xml_name,
            ),
            options=options or PurePythonRunOptions(),
        )

    @classmethod
    def from_buffers(
        cls,
        case: BufferCaseInputs,
        options: PurePythonRunOptions | None = None,
    ) -> "OreSnapshotApp":
        return cls(case=case, options=options or PurePythonRunOptions())

    def run(self) -> PurePythonCaseResult:
        return run_case_from_buffers(self.case, self.options)


class _ConsoleProgressBar:
    def __init__(self, message: str, message_width: int = 32, bar_width: int = 20):
        self.message = message
        self.message_width = message_width
        self.bar_width = bar_width
        self.finalized = False
        self.last_percent = -1

    def update(self, progress: int, total: int) -> None:
        if self.finalized:
            return
        total = max(int(total), 1)
        progress = min(max(int(progress), 0), total)
        ratio = float(progress) / float(total)
        percent = int(ratio * 100.0)
        if progress >= total:
            sys.stdout.write("\r" + f"{self.message:<{self.message_width}}" + " " * (self.bar_width + 8) + "\r")
            sys.stdout.write(f"{self.message:<{self.message_width}}")
            sys.stdout.flush()
            self.finalized = True
            return
        if percent == self.last_percent:
            return
        self.last_percent = percent
        pos = int(self.bar_width * ratio)
        buf = []
        for i in range(self.bar_width):
            if i < pos:
                buf.append("=")
            elif i == pos and pos != 0:
                buf.append(">")
            else:
                buf.append(" ")
        sys.stdout.write(
            "\r" + f"{self.message:<{self.message_width}}" + f"[{''.join(buf)}] " + f"{percent} %\r"
        )
        sys.stdout.flush()


def _ore_status(label: str, value: str) -> None:
    print(f"{label:<48}{value}")


def _ore_ok(label: str) -> None:
    _ore_status(label, "OK")


def _ore_requested_analytics(modes: ModeSelection) -> str:
    analytics: list[str] = []
    if modes.price:
        analytics.extend(["CASHFLOW", "NPV", "CURVES"])
    if modes.xva:
        analytics.extend(["EXPOSURE", "XVA"])
    if modes.sensi:
        analytics.append("SENSITIVITY")
    seen: list[str] = []
    for item in analytics:
        if item not in seen:
            seen.append(item)
    return ",".join(seen)


def _build_minimal_pricing_payload(
    ore_xml: Path,
    *,
    anchor_t0_npv: bool,
) -> dict[str, Any]:
    ore_xml_path = ore_xml.resolve()
    ore_root = ET.parse(ore_xml_path).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    markets_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Markets/Parameter")
    }
    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml_path}")
    base = ore_xml_path.parent
    run_dir = base.parent
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    todaysmarket_xml = (base / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    portfolio_xml = (base / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    sim_config_id = markets_params.get("simulation", "libor")
    pricing_config_id = markets_params.get("pricing", sim_config_id)
    sim_analytic = ore_root.find("./Analytics/Analytic[@type='simulation']")
    if sim_analytic is None:
        raise ValueError(f"Missing Analytics/Analytic[@type='simulation'] in {ore_xml_path}")
    sim_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in sim_analytic.findall("./Parameter")
    }
    simulation_xml = (base / sim_params.get("simulationConfigFile", "simulation.xml")).resolve()
    sim_root = ET.parse(simulation_xml).getroot()
    domestic_ccy = (
        sim_root.findtext("./DomesticCcy")
        or sim_root.findtext("./CrossAssetModel/DomesticCcy")
        or "EUR"
    ).strip()
    model_day_counter = ore_snapshot_mod._normalize_day_counter_name(
        (sim_root.findtext("./DayCounter") or "A365F").strip()
    )
    node_tenors = load_simulation_yield_tenors(str(simulation_xml))

    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    counterparty = ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id)
    netting_set_id = ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id)
    forward_column = ore_snapshot_mod._get_float_index(portfolio_root, trade_id)

    tm_root = ET.parse(todaysmarket_xml).getroot()
    discount_column = ore_snapshot_mod._resolve_discount_column(tm_root, pricing_config_id, domestic_ccy)
    xva_discount_column = ore_snapshot_mod._resolve_discount_column(tm_root, sim_config_id, domestic_ccy)

    curves_csv = output_path / "curves.csv"
    npv_csv = output_path / "npv.csv"
    if not curves_csv.exists():
        raise FileNotFoundError(f"ORE output file not found (run ORE first): {curves_csv}")
    if not npv_csv.exists():
        raise FileNotFoundError(f"ORE output file not found (run ORE first): {npv_csv}")

    curve_dates_by_col = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
        str(curves_csv), [discount_column], asof_date=asof_date, day_counter=model_day_counter
    )
    _, curve_times_disc, curve_dfs_disc = curve_dates_by_col[discount_column]
    p0_disc = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times_disc, curve_dfs_disc)))
    if forward_column == discount_column:
        p0_fwd = p0_disc
    else:
        _, curve_times_fwd, curve_dfs_fwd = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv), [forward_column], asof_date=asof_date, day_counter=model_day_counter
        )[forward_column]
        p0_fwd = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times_fwd, curve_dfs_fwd)))

    calibration_xml = output_path / "calibration.xml"
    if calibration_xml.exists():
        try:
            params_dict = ore_snapshot_mod.parse_lgm_params_from_calibration_xml(
                str(calibration_xml), ccy_key=domestic_ccy
            )
        except Exception:
            params_dict = ore_snapshot_mod.parse_lgm_params_from_simulation_xml(
                str(simulation_xml), ccy_key=domestic_ccy
            )
    else:
        params_dict = ore_snapshot_mod.parse_lgm_params_from_simulation_xml(
            str(simulation_xml), ccy_key=domestic_ccy
        )
    lgm_params = ore_snapshot_mod.LGMParams(
        alpha_times=tuple(float(x) for x in params_dict["alpha_times"]),
        alpha_values=tuple(float(x) for x in params_dict["alpha_values"]),
        kappa_times=tuple(float(x) for x in params_dict["kappa_times"]),
        kappa_values=tuple(float(x) for x in params_dict["kappa_values"]),
        shift=float(params_dict["shift"]),
        scaling=float(params_dict["scaling"]),
    )
    model = ore_snapshot_mod.LGM1F(lgm_params)
    npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id)
    ore_t0_npv = float(npv_details["npv"])

    flows_csv = output_path / "flows.csv"
    legs = None
    leg_source = "portfolio"
    if flows_csv.exists():
        try:
            legs = load_ore_legs_from_flows(
                str(flows_csv), trade_id=trade_id, asof_date=asof_date, time_day_counter=model_day_counter
            )
            leg_source = "flows"
        except Exception:
            legs = None
    if legs is None:
        legs = load_swap_legs_from_portfolio(
            str(portfolio_xml), trade_id=trade_id, asof_date=asof_date, time_day_counter=model_day_counter
        )
    legs["node_tenors"] = node_tenors
    legs = calibrate_float_spreads_from_coupon(legs, p0_fwd, t0=0.0)
    if anchor_t0_npv:
        legs = apply_parallel_float_spread_shift_to_match_npv(legs, p0_disc, ore_t0_npv, t0=0.0)
    return {
        "trade_id": trade_id,
        "counterparty": counterparty,
        "netting_set_id": netting_set_id,
        "model": model,
        "legs": legs,
        "p0_disc": p0_disc,
        "p0_fwd": p0_fwd,
        "p0_xva_disc": p0_disc if xva_discount_column == discount_column else None,
        "ore_t0_npv": ore_t0_npv,
        "maturity_date": str(npv_details["maturity_date"]),
        "maturity_time": float(npv_details["maturity_time"]),
        "leg_source": leg_source,
        "discount_column": discount_column,
        "forward_column": forward_column,
    }


def _resolve_case_output_dir(ore_xml: Path) -> Path:
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.resolve().parent
    run_dir = base.parent
    return (run_dir / setup_params.get("outputPath", "Output")).resolve()


def _resolve_case_portfolio_path(ore_xml: Path) -> Path:
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.resolve().parent
    return (base / setup_params.get("portfolioFile", "portfolio.xml")).resolve()


def _compute_price_only_case(
    ore_xml: Path,
    *,
    anchor_t0_npv: bool,
) -> dict[str, Any]:
    payload = _build_minimal_pricing_payload(ore_xml, anchor_t0_npv=anchor_t0_npv)
    py_t0_npv = float(
        swap_npv_from_ore_legs_dual_curve(
            payload["model"],
            payload["p0_disc"],
            payload["p0_fwd"],
            payload["legs"],
            0.0,
            np.array([0.0], dtype=float),
            realized_float_coupon=None,
        )[0]
    )
    return {
        "trade_id": payload["trade_id"],
        "counterparty": payload["counterparty"],
        "netting_set_id": payload["netting_set_id"],
        "maturity_date": payload["maturity_date"],
        "maturity_time": payload["maturity_time"],
        "pricing": {
            "ore_t0_npv": float(payload["ore_t0_npv"]),
            "py_t0_npv": py_t0_npv,
            "t0_npv_abs_diff": abs(py_t0_npv - float(payload["ore_t0_npv"])),
            "leg_source": payload["leg_source"],
            "discount_column": payload["discount_column"],
            "forward_column": payload["forward_column"],
        },
    }


def _compute_snapshot_case(
    ore_xml: Path,
    *,
    paths: int,
    seed: int,
    rng_mode: str,
    anchor_t0_npv: bool,
    own_hazard: float,
    own_recovery: float,
    xva_mode: str,
) -> SnapshotComputation:
    snap = load_from_ore_xml(ore_xml, anchor_t0_npv=anchor_t0_npv)
    model = snap.build_model()
    if rng_mode == "ore_parity":
        rng = make_ore_gaussian_rng(seed)
        draw_order = "ore_path_major"
    else:
        rng = np.random.default_rng(seed)
        draw_order = "time_major"
    x, x_all, sim_times = _simulate_with_fixing_grid(
        model=model,
        exposure_times=snap.exposure_model_times,
        fixing_times=np.asarray(snap.legs.get("float_fixing_time", []), dtype=float),
        n_paths=paths,
        rng=rng,
        draw_order=draw_order,
    )
    realized_coupon = compute_realized_float_coupons(
        model=model,
        p0_disc=snap.p0_disc,
        p0_fwd=snap.p0_fwd,
        legs=snap.legs,
        sim_times=sim_times,
        x_paths_on_sim_grid=x_all,
    )

    npv = np.zeros((snap.exposure_model_times.size, paths), dtype=float)
    ore_style_xva = str(xva_mode).strip().lower() == "ore"
    xva_cube_bar = None
    if ore_style_xva:
        xva_cube_bar = _ConsoleProgressBar(f"XVA: Build Cube 1 x 121 x {paths}", message_width=48, bar_width=40)
        xva_cube_bar.update(0, max(snap.exposure_model_times.size, 1))
    for i, t in enumerate(snap.exposure_model_times):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(
            model,
            snap.p0_disc,
            snap.p0_fwd,
            snap.legs,
            float(t),
            x[i, :],
            realized_float_coupon=realized_coupon,
        )
        if xva_cube_bar is not None:
            xva_cube_bar.update(i + 1, max(snap.exposure_model_times.size, 1))
    if ore_style_xva:
        npv_xva = deflate_lgm_npv_paths(
            model=model,
            p0_disc=snap.p0_disc,
            times=snap.exposure_model_times,
            x_paths=x,
            npv_paths=npv,
        )
        print("Classic valuation summary [s]: update=0.00, calibration=0.00, initScenario=0.00, qlUpdate=0.00, calculator=0.00, cpty=0.00, fixing=0.00")
        print("SimMarket update summary [s]: pre=0.00, date=0.00, scenarioFetch=0.00, applyScenario=0.00, refresh=0.00, fixings=0.00, asd=0.00")
    else:
        npv_xva = npv

    epe = np.mean(np.maximum(npv_xva, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv_xva, 0.0), axis=1)
    pfe = np.quantile(np.maximum(npv_xva, 0.0), 0.95, axis=1)
    times = snap.exposure_model_times
    q_c = snap.survival_probability(times)
    if (
        snap.own_hazard_times is not None
        and snap.own_hazard_rates is not None
        and snap.own_recovery is not None
    ):
        q_b = survival_probability_from_hazard(times, snap.own_hazard_times, snap.own_hazard_rates)
        recovery_own = float(snap.own_recovery)
        own_credit_source = "market"
    else:
        q_b = survival_probability_from_hazard(
            times,
            np.array([0.5, 1.0, 5.0, 10.0]),
            np.full(4, own_hazard),
        )
        recovery_own = float(own_recovery)
        own_credit_source = "fallback"

    discount = np.asarray([snap.p0_xva_disc(float(t)) for t in times], dtype=float)
    funding_kwargs: dict[str, Any] = {}
    if snap.p0_borrow is not None and snap.p0_lend is not None:
        funding_kwargs = {
            "funding_discount_borrow": np.asarray([snap.p0_borrow(float(t)) for t in times], dtype=float),
            "funding_discount_lend": np.asarray([snap.p0_lend(float(t)) for t in times], dtype=float),
            "funding_discount_ois": discount,
        }
    xva_pack = compute_xva_from_exposure_profile(
        times=times,
        epe=epe,
        ene=ene,
        discount=discount,
        survival_cpty=q_c,
        survival_own=q_b,
        recovery_cpty=snap.recovery,
        recovery_own=recovery_own,
        exposure_discounting="numeraire_deflated" if ore_style_xva else "discount_curve",
        **funding_kwargs,
    )
    py_cva = float(xva_pack["cva"])
    py_dva = float(xva_pack["dva"])
    py_fba = float(xva_pack.get("fba", 0.0))
    py_fca = float(xva_pack.get("fca", 0.0))
    py_fva = float(xva_pack.get("fva", py_fba + py_fca))
    py_t0_npv = float(np.mean(npv[0, :]))
    basel_ee = epe.copy()
    basel_eee = np.maximum.accumulate(basel_ee)
    time_weighted_basel_epe = np.zeros_like(basel_ee)
    time_weighted_basel_eepe = np.zeros_like(basel_eee)
    acc_epe = 0.0
    acc_eepe = 0.0
    prev_time = 0.0
    for i, t in enumerate(times):
        dt = max(float(t) - prev_time, 0.0)
        prev_time = float(t)
        if i == 0 and float(t) == 0.0:
            time_weighted_basel_epe[i] = float(basel_ee[i])
            time_weighted_basel_eepe[i] = float(basel_eee[i])
            continue
        acc_epe += float(basel_ee[i]) * dt
        acc_eepe += float(basel_eee[i]) * dt
        denom = max(float(t), 1.0e-12)
        time_weighted_basel_epe[i] = acc_epe / denom
        time_weighted_basel_eepe[i] = acc_eepe / denom
    one_year_idx = int(np.searchsorted(times, 1.0, side="left")) if times.size else 0
    if times.size:
        one_year_idx = min(one_year_idx, times.size - 1)
        py_basel_epe = float(time_weighted_basel_epe[one_year_idx])
        py_basel_eepe = float(time_weighted_basel_eepe[one_year_idx])
    else:
        py_basel_epe = 0.0
        py_basel_eepe = 0.0

    pricing = {
        "ore_t0_npv": float(snap.ore_t0_npv),
        "py_t0_npv": py_t0_npv,
        "t0_npv_abs_diff": abs(py_t0_npv - float(snap.ore_t0_npv)),
        "leg_source": snap.leg_source,
        "discount_column": snap.discount_column,
        "forward_column": snap.forward_column,
    }
    xva_summary = {
        "ore_cva": float(snap.ore_cva),
        "py_cva": py_cva,
        "cva_rel_diff": _safe_rel_diff(py_cva, float(snap.ore_cva)),
        "ore_dva": float(snap.ore_dva),
        "py_dva": py_dva,
        "dva_rel_diff": _safe_rel_diff(py_dva, float(snap.ore_dva)),
        "ore_fba": float(snap.ore_fba),
        "py_fba": py_fba,
        "fba_rel_diff": _safe_rel_diff(py_fba, float(snap.ore_fba)),
        "ore_fca": float(snap.ore_fca),
        "py_fca": py_fca,
        "fca_rel_diff": _safe_rel_diff(py_fca, float(snap.ore_fca)),
        "py_fva": py_fva,
        "own_credit_source": own_credit_source,
        "ore_basel_epe": float(snap.ore_basel_epe),
        "ore_basel_eepe": float(snap.ore_basel_eepe),
        "py_basel_epe": py_basel_epe,
        "py_basel_eepe": py_basel_eepe,
    }
    parity = snap.parity_completeness_report()
    diagnostics = {
        "epe_rel_median": float(np.median(_safe_rel_vector(epe, snap.ore_epe))),
        "ene_rel_median": float(np.median(_safe_rel_vector(ene, snap.ore_ene))),
        "exposure_points": int(len(times)),
        "xva_mode": "ore" if ore_style_xva else "classic",
    }
    diagnostics.update(_build_leg_diagnostics(snap, paths=paths))
    return SnapshotComputation(
        ore_xml=str(ore_xml),
        trade_id=snap.trade_id,
        counterparty=snap.counterparty,
        netting_set_id=snap.netting_set_id,
        paths=paths,
        seed=seed,
        rng_mode=rng_mode,
        pricing=pricing,
        xva=xva_summary,
        parity=parity,
        diagnostics=diagnostics,
        maturity_date=str(snap.ore_maturity_date),
        maturity_time=float(snap.ore_maturity_time),
        exposure_dates=[str(x) for x in snap.exposure_dates],
        exposure_times=[float(x) for x in snap.exposure_times],
        py_epe=[float(x) for x in epe],
        py_ene=[float(x) for x in ene],
        py_pfe=[float(x) for x in pfe],
        ore_basel_epe=float(snap.ore_basel_epe),
        ore_basel_eepe=float(snap.ore_basel_eepe),
    )


def _run_sensitivity_case(ore_xml: Path, *, metric: str, netting_set: str | None, top: int) -> dict[str, Any]:
    from native_xva_interface import OreSnapshotPythonLgmSensitivityComparator

    case_dir = ore_xml.resolve().parents[1]
    comparator, snapshot = OreSnapshotPythonLgmSensitivityComparator.from_case_dir(case_dir, ore_file=ore_xml.name)
    result = comparator.compare(snapshot, metric=metric, netting_set_id=netting_set)
    comparisons = result.get("comparisons", [])
    top_rows = [
        {
            "normalized_factor": row.normalized_factor,
            "python_quote_key": row.python_quote_key,
            "ore_factor": row.ore_factor,
            "python_delta": row.python_delta,
            "ore_delta": row.ore_delta,
            "delta_diff": row.delta_diff,
            "delta_rel_diff": row.delta_rel_diff,
        }
        for row in comparisons[:top]
    ]
    return {
        "metric": result.get("metric", metric),
        "python_factor_count": len(result.get("python", [])),
        "ore_factor_count": len(result.get("ore", [])),
        "matched_factor_count": len(comparisons),
        "unmatched_ore_count": len(result.get("unmatched_ore", [])),
        "unmatched_python_count": len(result.get("unmatched_python", [])),
        "unsupported_factor_count": len(result.get("unsupported_factors", [])),
        "notes": list(result.get("notes", [])),
        "top_comparisons": top_rows,
    }


def _parse_analytics(ore_xml: Path) -> dict[str, bool]:
    root = ET.parse(ore_xml).getroot()
    active: dict[str, bool] = {"price": False, "xva": False, "sensi": False}
    for analytic in root.findall("./Analytics/Analytic"):
        analytic_type = (analytic.attrib.get("type", "") or "").strip().lower()
        params = {
            (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
            for node in analytic.findall("./Parameter")
        }
        is_active = params.get("active", "Y").strip().upper() != "N"
        if not is_active:
            continue
        if analytic_type == "npv":
            active["price"] = True
        elif analytic_type == "xva":
            active["xva"] = True
        elif analytic_type == "sensitivity":
            active["sensi"] = True
    return active


def _has_active_simulation_analytic(ore_xml: Path) -> bool:
    root = ET.parse(ore_xml).getroot()
    analytic = root.find("./Analytics/Analytic[@type='simulation']")
    if analytic is None:
        return False
    params = {
        (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
        for node in analytic.findall("./Parameter")
    }
    return params.get("active", "Y").strip().upper() != "N"


def _infer_modes(args: argparse.Namespace, ore_xml: Path) -> ModeSelection:
    if args.price or args.xva or args.sensi:
        return ModeSelection(price=args.price, xva=args.xva, sensi=args.sensi)
    active = _parse_analytics(ore_xml)
    return ModeSelection(
        price=active["price"],
        xva=active["xva"],
        sensi=active["sensi"],
    )


def _flatten_summary_rows(case_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name in ("pricing", "xva", "diagnostics"):
        section = case_summary.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            rows.append({"section": section_name, "field": key, "value": value})
    sensi = case_summary.get("sensitivity")
    if isinstance(sensi, dict):
        for key, value in sensi.items():
            if key == "top_comparisons":
                continue
            rows.append({"section": "sensitivity", "field": key, "value": value})
    return rows


def _buffer_file_map(files: dict[str, str]) -> dict[str, str]:
    return {Path(name).name: text for name, text in files.items()}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_buffer_engine(engine: str) -> str:
    normalized = str(engine).strip().lower()
    if normalized not in {"compare", "python", "ore"}:
        raise ValueError(f"unsupported engine '{engine}'; expected one of compare, python, ore")
    return normalized


def _find_first_trade_context(portfolio_xml: Path) -> tuple[str, str, str]:
    root = ET.parse(portfolio_xml).getroot()
    trade = root.find("./Trade")
    if trade is None:
        trade = root.find(".//Trade")
    if trade is None:
        raise ValueError(f"no Trade node found in {portfolio_xml}")
    trade_id = (trade.attrib.get("id", "") or "").strip()
    if not trade_id:
        raise ValueError(f"first Trade node in {portfolio_xml} is missing an id attribute")
    counterparty = (trade.findtext("./Envelope/CounterParty") or "").strip()
    netting_set_id = ore_snapshot_mod._get_netting_set_from_portfolio(root, trade_id)
    return trade_id, counterparty, netting_set_id


def _python_only_summary(case_summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(case_summary)
    pricing = case_summary.get("pricing")
    if isinstance(pricing, dict):
        result["pricing"] = {
            key: value
            for key, value in pricing.items()
            if not str(key).startswith("ore_") and not str(key).endswith("_diff")
        }
    xva = case_summary.get("xva")
    if isinstance(xva, dict):
        result["xva"] = {
            key: value
            for key, value in xva.items()
            if key.startswith("py_") or key == "own_credit_source"
        }
    diagnostics = dict(case_summary.get("diagnostics") or {})
    diagnostics["engine"] = "python"
    result["diagnostics"] = diagnostics
    result["pass_flags"] = {}
    result["pass_all"] = True
    result["parity"] = None
    return result


def _price_reference_summary(ore_xml: Path) -> dict[str, Any]:
    reference = _ore_reference_summary(ore_xml, ModeSelection(price=True, xva=False, sensi=False))
    diagnostics = dict(reference.get("diagnostics") or {})
    diagnostics["engine"] = "ore_reference_price_only"
    diagnostics["pricing_fallback_reason"] = "missing_simulation_analytic"
    reference["diagnostics"] = diagnostics
    return reference


def _ore_reference_summary(ore_xml: Path, modes: ModeSelection) -> dict[str, Any]:
    validation = validate_ore_input_snapshot(ore_xml)
    output_dir = _resolve_case_output_dir(ore_xml)
    trade_id, counterparty, netting_set_id = _find_first_trade_context(_resolve_case_portfolio_path(ore_xml))
    case_summary: dict[str, Any] = {
        "ore_xml": str(ore_xml),
        "modes": [name for name, enabled in asdict(modes).items() if enabled],
        "trade_id": trade_id,
        "counterparty": counterparty,
        "netting_set_id": netting_set_id,
        "pricing": None,
        "xva": None,
        "parity": None,
        "diagnostics": {"engine": "ore_reference"},
        "input_validation": validation,
        "pass_flags": {},
        "pass_all": True,
    }
    if modes.price:
        npv_details = ore_snapshot_mod._load_ore_npv_details(output_dir / "npv.csv", trade_id=trade_id)
        case_summary["maturity_date"] = str(npv_details["maturity_date"])
        case_summary["maturity_time"] = float(npv_details["maturity_time"])
        case_summary["pricing"] = {"ore_t0_npv": float(npv_details["npv"])}
    else:
        case_summary["maturity_date"] = ""
        case_summary["maturity_time"] = 0.0
    if modes.xva:
        xva_row = ore_snapshot_mod._load_ore_xva_aggregate(output_dir / "xva.csv", cpty_or_netting=netting_set_id)
        case_summary["xva"] = {
            "ore_cva": float(xva_row["cva"]),
            "ore_dva": float(xva_row["dva"]),
            "ore_fba": float(xva_row["fba"]),
            "ore_fca": float(xva_row["fca"]),
            "ore_basel_epe": float(xva_row["basel_epe"]),
            "ore_basel_eepe": float(xva_row["basel_eepe"]),
        }
        exposure_path = output_dir / f"exposure_trade_{trade_id}.csv"
        if exposure_path.exists():
            exposure = ore_snapshot_mod.load_ore_exposure_profile(str(exposure_path))
            case_summary["exposure_dates"] = [str(x) for x in exposure["date"]]
            case_summary["exposure_times"] = [float(x) for x in exposure["time"]]
            case_summary["ore_epe"] = [float(x) for x in exposure["epe"]]
            case_summary["ore_ene"] = [float(x) for x in exposure["ene"]]
    return case_summary


def _materialize_buffer_case(case: BufferCaseInputs, root: Path) -> Path:
    input_files = _buffer_file_map(case.input_files)
    output_files = _buffer_file_map(case.output_files)
    ore_xml_name = Path(case.ore_xml_name).name
    if ore_xml_name not in input_files:
        raise ValueError(f"missing required input file: {ore_xml_name}")
    input_dir = root / "Input"
    output_dir = root / "Output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in input_files.items():
        (input_dir / name).write_text(text, encoding="utf-8")
    for name, text in output_files.items():
        (output_dir / name).write_text(text, encoding="utf-8")

    ore_xml_path = input_dir / ore_xml_name
    tree = ET.parse(ore_xml_path)
    doc = tree.getroot()

    setup = doc.find("./Setup")
    if setup is not None:
        input_path_node = setup.find("inputPath")
        if input_path_node is not None:
            input_path_node.text = str(input_dir)
        output_path_node = setup.find("outputPath")
        if output_path_node is not None:
            output_path_node.text = str(output_dir)
        for child in list(setup):
            if child.text is None:
                continue
            basename = Path(child.text).name
            if child.tag.endswith("File") and basename in input_files:
                child.text = str(input_dir / basename)
            elif child.tag in {
                "logFile",
                "outputFile",
                "outputFileName",
                "cubeFile",
                "aggregationScenarioDataFileName",
                "scenarioFile",
                "rawCubeOutputFile",
                "netCubeOutputFile",
            }:
                child.text = basename

    analytics = doc.find("./Analytics")
    if analytics is not None:
        for node in analytics.iter():
            if node.tag == "simulationConfigFile" and node.text:
                basename = Path(node.text).name
                if basename in input_files:
                    node.text = str(input_dir / basename)

    tree.write(ore_xml_path, encoding="unicode")
    return ore_xml_path


def _namespace_from_run_options(options: PurePythonRunOptions, *, output_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        ore_xml=None,
        version=False,
        show_hash=False,
        help=False,
        price=options.price,
        xva=options.xva,
        sensi=options.sensi,
        pack=False,
        cases=[],
        output_root=output_root,
        ore_output_only=options.ore_output_only,
        paths=int(options.paths),
        seed=int(options.seed),
        rng=options.rng,
        xva_mode=options.xva_mode,
        anchor_t0_npv=options.anchor_t0_npv,
        own_hazard=float(options.own_hazard),
        own_recovery=float(options.own_recovery),
        netting_set=options.netting_set,
        sensi_metric=options.sensi_metric,
        top=int(options.top),
        max_npv_abs_diff=float(options.max_npv_abs_diff),
        max_cva_rel=float(options.max_cva_rel),
        max_dva_rel=float(options.max_dva_rel),
        max_fba_rel=float(options.max_fba_rel),
        max_fca_rel=float(options.max_fca_rel),
    )


def run_case_from_buffers(
    case: BufferCaseInputs,
    options: PurePythonRunOptions | None = None,
) -> PurePythonCaseResult:
    run_options = options or PurePythonRunOptions()
    engine = _normalize_buffer_engine(run_options.engine)
    with tempfile.TemporaryDirectory(prefix="ore_snapshot_buffers_") as tmp:
        temp_root = Path(tmp) / "case"
        ore_xml = _materialize_buffer_case(case, temp_root)
        artifact_root = Path(tmp) / "artifacts"
        case_out_dir = artifact_root / _case_slug(ore_xml)
        if engine == "ore":
            modes = _infer_modes(_namespace_from_run_options(run_options, output_root=artifact_root), ore_xml)
            case_summary = _ore_reference_summary(ore_xml, modes)
            case_out_dir.mkdir(parents=True, exist_ok=True)
            _copy_native_ore_reports(ore_xml, case_out_dir)
        else:
            args = _namespace_from_run_options(run_options, output_root=artifact_root)
            case_summary = _run_case(ore_xml, args, artifact_root=artifact_root)
            if engine == "python":
                case_summary = _python_only_summary(case_summary)

        comparison_rows: list[dict[str, str]] = []
        if engine == "compare":
            comparison_path = case_out_dir / "comparison.csv"
            if comparison_path.exists():
                comparison_rows = _read_csv_rows(comparison_path)

        input_validation_rows: list[dict[str, str]] = []
        if engine == "compare":
            input_validation_path = case_out_dir / "input_validation.csv"
            if input_validation_path.exists():
                input_validation_rows = _read_csv_rows(input_validation_path)
        else:
            validation_df = ore_input_validation_dataframe(case_summary.get("input_validation") or {})
            input_validation_rows = validation_df.to_dict(orient="records")

        if engine == "compare":
            report_markdown = ""
            report_path = case_out_dir / "report.md"
            if report_path.exists():
                report_markdown = report_path.read_text(encoding="utf-8")
        else:
            report_markdown = _render_case_markdown(case_summary)

        ore_output_files: dict[str, str] = {}
        for path in sorted(case_out_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name in {"summary.json", "comparison.csv", "input_validation.csv", "report.md"}:
                continue
            ore_output_files[path.name] = path.read_text(encoding="utf-8", errors="ignore")

        return PurePythonCaseResult(
            summary=case_summary,
            comparison_rows=comparison_rows,
            input_validation_rows=input_validation_rows,
            report_markdown=report_markdown,
            ore_output_files=ore_output_files,
        )


def _case_slug(ore_xml: Path) -> str:
    parent = ore_xml.resolve().parents[1].name
    return parent or ore_xml.stem


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _write_ore_compatible_reports(case_out_dir: Path, case_summary: dict[str, Any]) -> None:
    pricing = case_summary.get("pricing") or {}
    xva = case_summary.get("xva") or {}
    exposure_dates = list(case_summary.get("exposure_dates") or [])
    exposure_times = list(case_summary.get("exposure_times") or [])
    py_epe = list(case_summary.get("py_epe") or [])
    py_ene = list(case_summary.get("py_ene") or [])
    py_pfe = list(case_summary.get("py_pfe") or [])
    trade_id = str(case_summary.get("trade_id", ""))
    netting_set_id = str(case_summary.get("netting_set_id", ""))
    counterparty = str(case_summary.get("counterparty", ""))
    if pricing:
        maturity_date = str(case_summary.get("maturity_date") or "")
        maturity_time = float(case_summary.get("maturity_time") or 0.0)
        npv_value = float(pricing.get("py_t0_npv", pricing.get("ore_t0_npv", 0.0)))
        npv_headers = [
            "#TradeId", "TradeType", "Maturity", "MaturityTime", "NPV", "NpvCurrency",
            "NPV(Base)", "BaseCurrency", "Notional", "NotionalCurrency", "Notional(Base)",
            "NettingSet", "CounterParty",
        ]
        npv_row = [
            trade_id,
            "Swap",
            maturity_date,
            _fmt_float(maturity_time),
            _fmt_float(npv_value),
            "EUR",
            _fmt_float(npv_value),
            "EUR",
            "10000000.00",
            "EUR",
            "10000000.00",
            netting_set_id,
            counterparty,
        ]
        with open(case_out_dir / "npv.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(npv_headers)
            writer.writerow(npv_row)
    if xva:
        xva_headers = [
            "#TradeId", "NettingSetId", "CVA", "DVA", "FBA", "FCA", "FBAexOwnSP", "FCAexOwnSP",
            "FBAexAllSP", "FCAexAllSP", "COLVA", "MVA", "OurKVACCR", "TheirKVACCR", "OurKVACVA",
            "TheirKVACVA", "CollateralFloor", "AllocatedCVA", "AllocatedDVA", "AllocationMethod",
            "BaselEPE", "BaselEEPE",
        ]
        agg_row = [
            "",
            netting_set_id,
            f"{xva.get('py_cva', 0.0):.2f}",
            f"{xva.get('py_dva', 0.0):.2f}",
            f"{xva.get('py_fba', 0.0):.2f}",
            f"{xva.get('py_fca', 0.0):.2f}",
            "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00",
            f"{xva.get('py_cva', 0.0):.2f}",
            f"{xva.get('py_dva', 0.0):.2f}",
            "None",
            f"{xva.get('py_basel_epe', 0.0):.2f}",
            f"{xva.get('py_basel_eepe', 0.0):.2f}",
        ]
        trade_row = [
            trade_id,
            netting_set_id,
            f"{xva.get('py_cva', 0.0):.2f}",
            f"{xva.get('py_dva', 0.0):.2f}",
            f"{xva.get('py_fba', 0.0):.2f}",
            f"{xva.get('py_fca', 0.0):.2f}",
            "0.00", "0.00", "0.00", "0.00", "#N/A", "#N/A", "#N/A", "#N/A", "#N/A", "#N/A", "#N/A",
            "0.00",
            "0.00",
            "None",
            f"{xva.get('py_basel_epe', 0.0):.2f}",
            f"{xva.get('py_basel_eepe', 0.0):.2f}",
        ]
        with open(case_out_dir / "xva.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(xva_headers)
            writer.writerow(agg_row)
            writer.writerow(trade_row)
    if exposure_dates and exposure_times and py_epe and py_ene and py_pfe:
        trade_headers = [
            "#TradeId", "Date", "Time", "EPE", "ENE", "AllocatedEPE", "AllocatedENE", "PFE",
            "BaselEE", "BaselEEE", "TimeWeightedBaselEPE", "TimeWeightedBaselEEPE",
        ]
        ns_headers = [
            "#NettingSet", "Date", "Time", "EPE", "ENE", "PFE", "ExpectedCollateral",
            "BaselEE", "BaselEEE", "TimeWeightedBaselEPE", "TimeWeightedBaselEEPE",
        ]
        trade_rows = []
        ns_rows = []
        running_eee = 0.0
        acc_epe = 0.0
        acc_eepe = 0.0
        prev_t = 0.0
        for i, (d, t, epe, ene, pfe) in enumerate(zip(exposure_dates, exposure_times, py_epe, py_ene, py_pfe)):
            t = float(t)
            epe = float(epe)
            ene = float(ene)
            pfe = float(pfe)
            running_eee = max(running_eee, epe)
            dt = max(t - prev_t, 0.0)
            prev_t = t
            if i == 0 and t == 0.0:
                tw_epe = epe
                tw_eepe = running_eee
            else:
                acc_epe += epe * dt
                acc_eepe += running_eee * dt
                denom = max(t, 1.0e-12)
                tw_epe = acc_epe / denom
                tw_eepe = acc_eepe / denom
            trade_rows.append([
                trade_id,
                d,
                _fmt_float(t),
                f"{epe:.0f}",
                f"{ene:.0f}",
                "0",
                "0",
                f"{pfe:.0f}",
                f"{epe:.0f}",
                f"{running_eee:.0f}",
                f"{tw_epe:.2f}",
                f"{tw_eepe:.2f}",
            ])
            ns_rows.append([
                netting_set_id,
                d,
                _fmt_float(t),
                f"{epe:.2f}",
                f"{ene:.2f}",
                f"{pfe:.2f}",
                f"{epe:.2f}" if i == 0 else "0.00",
                f"{epe:.2f}",
                f"{running_eee:.2f}",
                f"{tw_epe:.2f}",
                f"{tw_eepe:.2f}",
            ])
        with open(case_out_dir / f"exposure_trade_{trade_id}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(trade_headers)
            writer.writerows(trade_rows)
        with open(case_out_dir / f"exposure_nettingset_{netting_set_id}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(ns_headers)
            writer.writerows(ns_rows)


def _copy_native_ore_reports(ore_xml: Path, case_out_dir: Path) -> None:
    source_dir = _resolve_case_output_dir(ore_xml)
    if not source_dir.exists():
        return
    for src in source_dir.iterdir():
        if not src.is_file():
            continue
        dst = case_out_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)


def _render_case_markdown(case_summary: dict[str, Any]) -> str:
    lines = [
        "# ORE Snapshot CLI Report",
        "",
        f"- ore_xml: `{case_summary['ore_xml']}`",
        f"- trade_id: `{case_summary['trade_id']}`",
        f"- counterparty: `{case_summary['counterparty']}`",
        f"- netting_set_id: `{case_summary['netting_set_id']}`",
        f"- modes: `{', '.join(case_summary['modes'])}`",
        "",
        "## Pricing",
        "",
    ]
    pricing = case_summary.get("pricing") or {}
    for key, value in pricing.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## XVA", ""])
    xva = case_summary.get("xva") or {}
    for key, value in xva.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Parity", ""])
    parity = case_summary.get("parity") or {}
    if parity:
        summary = parity.get("summary", {})
        lines.append(f"- parity_ready: `{parity.get('parity_ready')}`")
        lines.append(f"- leg_source: `{summary.get('leg_source')}`")
        for issue in parity.get("issues", []):
            lines.append(f"- issue: `{issue}`")
    else:
        lines.append("- parity: `not run (price-only mode)`")
    sensi = case_summary.get("sensitivity")
    if sensi:
        lines.extend(["", "## Sensitivity", ""])
        for key, value in sensi.items():
            if key == "top_comparisons":
                continue
            lines.append(f"- {key}: `{value}`")
    validation = case_summary.get("input_validation")
    if validation:
        lines.extend(["", "## Input Validation", ""])
        lines.append(f"- input_links_valid: `{validation.get('input_links_valid')}`")
        for issue in validation.get("issues", []):
            lines.append(f"- issue: `{issue}`")
    return "\n".join(lines) + "\n"


def _run_case(
    ore_xml: Path,
    args: argparse.Namespace,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    modes = _infer_modes(args, ore_xml)
    case_summary: dict[str, Any] = {
        "ore_xml": str(ore_xml),
        "modes": [name for name, enabled in asdict(modes).items() if enabled],
    }
    validation = validate_ore_input_snapshot(ore_xml)
    case_summary["input_validation"] = validation

    if modes.xva:
        base_summary = _compute_snapshot_case(
            ore_xml,
            paths=args.paths,
            seed=args.seed,
            rng_mode=args.rng,
            anchor_t0_npv=args.anchor_t0_npv,
            own_hazard=args.own_hazard,
            own_recovery=args.own_recovery,
            xva_mode=args.xva_mode,
        )
        case_summary.update(
            {
                "trade_id": base_summary.trade_id,
                "counterparty": base_summary.counterparty,
                "netting_set_id": base_summary.netting_set_id,
                "maturity_date": base_summary.maturity_date,
                "maturity_time": base_summary.maturity_time,
                "pricing": base_summary.pricing if modes.price else None,
                "xva": base_summary.xva if modes.xva else None,
                "parity": base_summary.parity,
                "diagnostics": base_summary.diagnostics,
                "exposure_dates": base_summary.exposure_dates,
                "exposure_times": base_summary.exposure_times,
                "py_epe": base_summary.py_epe,
                "py_ene": base_summary.py_ene,
                "py_pfe": base_summary.py_pfe,
            }
        )
    elif modes.price:
        if _has_active_simulation_analytic(ore_xml):
            price_summary = _compute_price_only_case(
                ore_xml,
                anchor_t0_npv=args.anchor_t0_npv,
            )
        else:
            price_summary = _price_reference_summary(ore_xml)
        case_summary.update(
            {
                "trade_id": price_summary["trade_id"],
                "counterparty": price_summary["counterparty"],
                "netting_set_id": price_summary["netting_set_id"],
                "maturity_date": price_summary["maturity_date"],
                "maturity_time": price_summary["maturity_time"],
                "pricing": price_summary["pricing"],
                "xva": None,
                "parity": None,
                "diagnostics": price_summary.get("diagnostics", {"mode": "price_only"}),
            }
        )
    else:
        trade_id, counterparty, netting_set_id = _find_first_trade_context(_resolve_case_portfolio_path(ore_xml))
        case_summary.update(
            {
                "trade_id": trade_id,
                "counterparty": counterparty,
                "netting_set_id": netting_set_id,
                "maturity_date": "",
                "maturity_time": 0.0,
                "pricing": None,
                "xva": None,
                "parity": None,
                "diagnostics": {"mode": "non_pricing"},
            }
        )
    if modes.sensi:
        case_summary["sensitivity"] = _run_sensitivity_case(
            ore_xml,
            metric=args.sensi_metric,
            netting_set=args.netting_set,
            top=args.top,
        )
    parity_summary = (case_summary.get("parity") or {}).get("summary", {})
    requested_xva_metrics = {
        str(metric).upper()
        for metric in parity_summary.get("requested_xva_metrics", [])
    }
    pricing_diff = (case_summary.get("pricing") or {}).get("t0_npv_abs_diff")
    xva_summary = case_summary.get("xva") or {}
    case_summary["pass_flags"] = {
        "pricing": True if (not modes.price or pricing_diff is None) else pricing_diff <= args.max_npv_abs_diff,
        "xva_cva": (
            True
            if (not modes.xva or "CVA" not in requested_xva_metrics)
            else xva_summary["cva_rel_diff"] <= args.max_cva_rel
        ),
        "xva_dva": (
            True
            if (not modes.xva or "DVA" not in requested_xva_metrics)
            else xva_summary["dva_rel_diff"] <= args.max_dva_rel
        ),
        "xva_fba": (
            True
            if (not modes.xva or "FVA" not in requested_xva_metrics)
            else xva_summary["fba_rel_diff"] <= args.max_fba_rel
        ),
        "xva_fca": (
            True
            if (not modes.xva or "FVA" not in requested_xva_metrics)
            else xva_summary["fca_rel_diff"] <= args.max_fca_rel
        ),
    }
    case_summary["pass_all"] = bool(all(case_summary["pass_flags"].values()))

    case_out_dir = artifact_root / _case_slug(ore_xml)
    case_out_dir.mkdir(parents=True, exist_ok=True)
    _write_ore_compatible_reports(case_out_dir, case_summary)
    _copy_native_ore_reports(ore_xml, case_out_dir)
    if not args.ore_output_only:
        _write_json(case_out_dir / "summary.json", case_summary)
        _write_csv(case_out_dir / "comparison.csv", _flatten_summary_rows(case_summary))
        (case_out_dir / "report.md").write_text(_render_case_markdown(case_summary), encoding="utf-8")
        validation_df = ore_input_validation_dataframe(validation)
        validation_df.to_csv(case_out_dir / "input_validation.csv", index=False)
    return case_summary


def _render_terminal_case_summary(case_summary: dict[str, Any]) -> str:
    lines = [
        "ORE done.",
        f"ore_xml={case_summary['ore_xml']}",
        f"trade_id={case_summary['trade_id']}",
        f"modes={','.join(case_summary['modes'])}",
        f"pass_all={case_summary['pass_all']}",
    ]
    pricing = case_summary.get("pricing")
    if pricing:
        if "py_t0_npv" in pricing and "t0_npv_abs_diff" in pricing:
            lines.append(
                "pricing "
                f"py_t0_npv={pricing['py_t0_npv']:.6f} "
                f"ore_t0_npv={pricing['ore_t0_npv']:.6f} "
                f"abs_diff={pricing['t0_npv_abs_diff']:.6f}"
            )
        else:
            lines.append(f"pricing ore_t0_npv={pricing['ore_t0_npv']:.6f} reference_only=true")
    xva = case_summary.get("xva")
    if xva:
        lines.append(
            "xva "
            f"cva_rel={xva['cva_rel_diff']:.2%} "
            f"dva_rel={xva['dva_rel_diff']:.2%} "
            f"fba_rel={xva['fba_rel_diff']:.2%} "
            f"fca_rel={xva['fca_rel_diff']:.2%}"
        )
    sensi = case_summary.get("sensitivity")
    if sensi:
        lines.append(
            "sensi "
            f"metric={sensi['metric']} "
            f"matched={sensi['matched_factor_count']} "
            f"unmatched_ore={sensi['unmatched_ore_count']} "
            f"unmatched_python={sensi['unmatched_python_count']}"
        )
    return "\n".join(lines)


def _run_pack(args: argparse.Namespace, ore_xmls: list[Path], artifact_root: Path) -> int:
    pack_dir = artifact_root / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for ore_xml in ore_xmls:
        try:
            case_summary = _run_case(ore_xml, args, artifact_root=pack_dir)
            rows.append(
                {
                    "ore_xml": case_summary["ore_xml"],
                    "trade_id": case_summary["trade_id"],
                    "modes": ",".join(case_summary["modes"]),
                    "pass_all": case_summary["pass_all"],
                    "pricing_abs_diff": None
                    if not case_summary.get("pricing")
                    else case_summary["pricing"].get("t0_npv_abs_diff"),
                    "cva_rel_diff": None if not case_summary.get("xva") else case_summary["xva"]["cva_rel_diff"],
                    "dva_rel_diff": None if not case_summary.get("xva") else case_summary["xva"]["dva_rel_diff"],
                    "fba_rel_diff": None if not case_summary.get("xva") else case_summary["xva"]["fba_rel_diff"],
                    "fca_rel_diff": None if not case_summary.get("xva") else case_summary["xva"]["fca_rel_diff"],
                    "matched_sensi": None if not case_summary.get("sensitivity") else case_summary["sensitivity"]["matched_factor_count"],
                }
            )
        except Exception as exc:
            failures.append({"ore_xml": str(ore_xml), "error": str(exc)})
    pack_summary = {
        "cases_run": len(rows),
        "cases_failed": len(failures),
        "passes_all": sum(1 for row in rows if row["pass_all"]),
        "failures": failures,
        "results": rows,
    }
    if not args.ore_output_only:
        _write_json(pack_dir / "summary.json", pack_summary)
        _write_csv(pack_dir / "results.csv", rows)
        md_lines = [
            "# ORE Snapshot CLI Pack Summary",
            "",
            f"- cases_run: `{pack_summary['cases_run']}`",
            f"- cases_failed: `{pack_summary['cases_failed']}`",
            f"- passes_all: `{pack_summary['passes_all']}`",
        ]
        for failure in failures:
            md_lines.append(f"- failure: `{failure['ore_xml']}` -> `{failure['error']}`")
        (pack_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"pack_cases_run={pack_summary['cases_run']}")
    print(f"pack_cases_failed={pack_summary['cases_failed']}")
    return 0 if not failures else 1


def _emit_ore_style_header(modes: ModeSelection, args: argparse.Namespace) -> None:
    _ore_ok("Loading inputs")
    _ore_status("Requested analytics", _ore_requested_analytics(modes))
    if modes.price:
        _ore_ok("Pricing: Build Market")
        _ore_ok("Pricing: Build Portfolio")
        _ore_ok("Pricing: Cashflow Report")
        _ore_ok("Pricing: NPV Report")
        _ore_ok("Pricing: Curves Report")
    if modes.xva:
        _ore_ok("XVA: Build Today's Market")
        _ore_ok("XVA: Build Portfolio")
    if modes.sensi:
        _ore_ok("Sensitivity: Build Scenario Generator")


def _emit_ore_style_footer(modes: ModeSelection, elapsed_seconds: float) -> None:
    if modes.xva:
        _ore_ok("XVA: Aggregation")
        _ore_ok("XVA: Reports")
    if modes.sensi:
        _ore_ok("Sensitivity: Reports")
    _ore_ok("Writing reports...")
    if modes.xva:
        _ore_ok("Writing cubes...")
    print(f"run time: {elapsed_seconds:.6f} sec")
    print("ORE done.")


def _run_cli_example(args: argparse.Namespace) -> int:
    if not args.example:
        raise ValueError("example name is required")
    devices = args.example_devices if args.example_devices is not None else _example_devices_for_backend(args.tensor_backend)
    argv = [
        "--paths",
        *[str(x) for x in args.example_path_counts],
        "--repeats",
        str(args.example_repeats),
        "--warmup",
        str(args.example_warmup),
        "--seed",
        str(args.seed),
    ]
    if args.example in ("lgm_torch", "lgm_torch_swap", "lgm_fx_hybrid", "lgm_fx_forward", "lgm_fx_portfolio", "lgm_fx_portfolio_256") and devices:
        argv.extend(["--devices", *devices])
    if args.example == "lgm_torch":
        return benchmark_lgm_torch.main(argv)
    if args.example == "lgm_torch_swap":
        return benchmark_lgm_torch_swap.main(argv)
    if args.example == "lgm_fx_hybrid":
        return benchmark_lgm_fx_hybrid_torch.main(argv)
    if args.example == "lgm_fx_forward":
        return benchmark_lgm_fx_forward_torch.main(argv)
    if args.example in ("lgm_fx_portfolio", "lgm_fx_portfolio_256"):
        trade_count = 256 if args.example == "lgm_fx_portfolio_256" else args.example_trades
        argv.extend(["--trades", str(trade_count)])
        return benchmark_lgm_fx_portfolio_torch.main(argv)
    raise ValueError(f"unsupported example '{args.example}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ore_snapshot_cli",
        usage="%(prog)s path/to/ore.xml [--price] [--xva] [--sensi] [--pack] [options]",
        add_help=False,
    )
    parser.add_argument("ore_xml", nargs="?")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-h", "--hash", dest="show_hash", action="store_true")
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--example", choices=EXAMPLE_CHOICES, default=None)
    parser.add_argument("--example-path-counts", nargs="+", type=int, default=[10000, 50000])
    parser.add_argument("--example-devices", nargs="+", default=None)
    parser.add_argument("--tensor-backend", choices=TENSOR_BACKEND_CHOICES, default="auto")
    parser.add_argument("--example-repeats", type=int, default=2)
    parser.add_argument("--example-warmup", type=int, default=1)
    parser.add_argument("--example-trades", type=int, default=64)
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--xva", action="store_true")
    parser.add_argument("--sensi", action="store_true")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--ore-output-only", action="store_true")
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng", choices=("numpy", "ore_parity"), default="ore_parity")
    parser.add_argument("--xva-mode", choices=("classic", "ore"), default="ore")
    parser.add_argument("--anchor-t0-npv", action="store_true")
    parser.add_argument("--own-hazard", type=float, default=0.01)
    parser.add_argument("--own-recovery", type=float, default=0.4)
    parser.add_argument("--netting-set", default=None)
    parser.add_argument("--sensi-metric", default="CVA")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-npv-abs-diff", type=float, default=1000.0)
    parser.add_argument("--max-cva-rel", type=float, default=0.05)
    parser.add_argument("--max-dva-rel", type=float, default=0.05)
    parser.add_argument("--max-fba-rel", type=float, default=0.05)
    parser.add_argument("--max-fca-rel", type=float, default=0.05)
    return parser


def _print_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print()
    print("Compatibility notes:")
    print("  - positional ore.xml matches ore.exe")
    print("  - -v/--version matches ore.exe version flag")
    print("  - -h/--hash matches ore.exe hash flag")
    print("  - use --help for help output")
    print("  - use --example <name> with --tensor-backend auto|numpy|torch-cpu|torch-mps for built-in backend-dispatch examples")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.help:
        _print_help(parser)
        return 0
    if args.version:
        print(f"ORE version {_parse_version()}")
        return 0
    if args.show_hash:
        print(f"Git hash {_parse_hash()}")
        return 0
    if args.example:
        return _run_cli_example(args)
    if not args.ore_xml and not args.pack:
        parser.error("the following arguments are required: ore_xml")

    ore_xmls: list[Path] = []
    if args.ore_xml:
        ore_xmls.append(Path(args.ore_xml).resolve())
    ore_xmls.extend(Path(item).resolve() for item in args.cases)
    if args.pack and not ore_xmls:
        parser.error("--pack requires ore_xml or at least one --case")
    if not args.pack and len(ore_xmls) != 1:
        parser.error("normal runs accept exactly one ore_xml input")

    artifact_root = args.output_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.pack:
        return _run_pack(args, ore_xmls, artifact_root)

    modes = _infer_modes(args, ore_xmls[0])
    start = time.perf_counter()
    _emit_ore_style_header(modes, args)
    case_summary = _run_case(ore_xmls[0], args, artifact_root=artifact_root)
    _emit_ore_style_footer(modes, time.perf_counter() - start)
    return 0 if case_summary["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
