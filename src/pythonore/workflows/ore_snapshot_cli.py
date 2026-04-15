from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import QuantLib as ql
except Exception:  # pragma: no cover - optional at import time
    ql = None

import pythonore.io.ore_snapshot as ore_snapshot_mod
from pythonore.io import XVALoader
from pythonore.payoff_ir import build_equity_ore_black_scholes_model, lower_ore_script
from pythonore.payoff_ir.exec_numpy import execute_numpy
from pythonore.compute.irs_xva_utils import (
    apply_parallel_float_spread_shift_to_match_npv,
    calibrate_float_spreads_from_coupon,
    compute_realized_float_coupons,
    compute_xva_from_exposure_profile,
    deflate_lgm_npv_paths,
    load_ore_default_curve_inputs,
    load_ore_exposure_profile,
    load_ore_legs_from_flows,
    load_trade_cashflows_from_flows,
    load_simulation_yield_tenors,
    load_swap_legs_from_portfolio,
    survival_probability_from_hazard,
    swap_npv_from_ore_legs_dual_curve,
)
from pythonore.compute.lgm import LGM1F, LGMParams, make_ore_gaussian_rng, simulate_ba_measure, simulate_lgm_measure
from pythonore.io.ore_snapshot import (
    fit_discount_curves_from_ore_market,
    load_from_ore_xml,
    ore_input_validation_dataframe,
    validate_ore_input_snapshot,
)
from pythonore.repo_paths import default_ore_bin, find_engine_repo_root, local_parity_artifacts_root
from pythonore.runtime import classify_portfolio_support
from pythonore.runtime.core import XVAEngine
from pythonore.runtime.exposure_profiles import (
    build_ore_exposure_profile_from_paths,
    build_ore_exposure_profile_from_series,
    one_year_profile_value as _shared_one_year_profile_value,
)
from pythonore.runtime.lgm.inputs import _parse_swaption_premium_records


DEFAULT_ARTIFACT_ROOT = local_parity_artifacts_root() / "ore_snapshot_cli"
_REPORT_CASE_EXEC_LOCK = threading.Lock()
EXAMPLE_CHOICES = (
    "lgm_torch",
    "lgm_torch_swap",
    "lgm_fx_hybrid",
    "lgm_fx_forward",
    "lgm_fx_portfolio",
    "lgm_fx_portfolio_256",
)
TENSOR_BACKEND_CHOICES = ("auto", "numpy", "torch-cpu", "torch-mps")
REPORT_BUCKET_HINTS = {
    "clean_pass": "No action needed; case is clean on the current parity path.",
    "expected_output_fallback_pass": "Case is passing against vendored ExpectedOutput; only generate native Output if you want native-artifact parity as well.",
    "no_reference_artifacts_pass": "Case has no native Output or vendored ExpectedOutput; generate native artifacts only if you want parity-grade references.",
    "python_only_no_reference": "Case looks natively runnable in Python, but no Output or ExpectedOutput baseline exists; generate reference artifacts only if you want parity-grade comparison.",
    "parity_threshold_fail": "Inspect per-case summary and comparison metrics; use pricing vs XVA pass flags to choose the next subsystem.",
    "sample_count_mismatch": "Align Python paths with ORE samples before treating the diff as structural.",
    "missing_native_output_fallback": "Decide whether to vendor native outputs or accept expected-output-only parity.",
    "unsupported_python_snapshot_fallback": "Inspect the product/model support gap or accept reference fallback for this family.",
    "price_only_reference_fallback": "Enable or restore the simulation analytic if Python-side pricing parity is required.",
    "input_validation_issue": "Fix file links, market config ids, or required quotes before parity work.",
    "missing_reference_pricing": "Provide or regenerate pricing reference artifacts (`npv.csv`) before pricing parity work.",
    "missing_reference_xva": "Provide or regenerate XVA reference artifacts (`xva.csv` / exposure files) before XVA parity work.",
    "hard_error": "Inspect the recorded error/parse failure before any parity debugging.",
}


def _skipped_input_validation() -> dict[str, Any]:
    return {
        "skipped": True,
        "input_links_valid": True,
        "checks": {},
        "action_items": [],
    }


def _bond_pricing_mod():
    return import_module("pythonore.compute.bond_pricing")


def price_bond_trade(*args, **kwargs):
    return _bond_pricing_mod().price_bond_trade(*args, **kwargs)


def _inflation_mod():
    return import_module("pythonore.compute.inflation")


def InflationCapFloorDefinition(*args, **kwargs):
    return _inflation_mod().InflationCapFloorDefinition(*args, **kwargs)


def inflation_swap_payment_times(*args, **kwargs):
    return _inflation_mod().inflation_swap_payment_times(*args, **kwargs)


def load_inflation_curve_from_market_data(*args, **kwargs):
    return _inflation_mod().load_inflation_curve_from_market_data(*args, **kwargs)


def load_zero_inflation_surface_quote(*args, **kwargs):
    return _inflation_mod().load_zero_inflation_surface_quote(*args, **kwargs)


def parse_inflation_models_from_simulation_xml(*args, **kwargs):
    return _inflation_mod().parse_inflation_models_from_simulation_xml(*args, **kwargs)


def price_inflation_capfloor(*args, **kwargs):
    return _inflation_mod().price_inflation_capfloor(*args, **kwargs)


def price_inflation_capfloor_at_time(*args, **kwargs):
    return _inflation_mod().price_inflation_capfloor_at_time(*args, **kwargs)


def price_yoy_swap(*args, **kwargs):
    return _inflation_mod().price_yoy_swap(*args, **kwargs)


def price_yoy_swap_at_time(*args, **kwargs):
    return _inflation_mod().price_yoy_swap_at_time(*args, **kwargs)


def price_zero_coupon_cpi_swap(*args, **kwargs):
    return _inflation_mod().price_zero_coupon_cpi_swap(*args, **kwargs)


def price_zero_coupon_cpi_swap_at_time(*args, **kwargs):
    return _inflation_mod().price_zero_coupon_cpi_swap_at_time(*args, **kwargs)


def _lgm_fx_xva_mod():
    return import_module("pythonore.compute.lgm_fx_xva_utils")


def FxForwardDef(*args, **kwargs):
    return _lgm_fx_xva_mod().FxForwardDef(*args, **kwargs)


def FxOptionDef(*args, **kwargs):
    return _lgm_fx_xva_mod().FxOptionDef(*args, **kwargs)


def build_two_ccy_hybrid(*args, **kwargs):
    return _lgm_fx_xva_mod().build_two_ccy_hybrid(*args, **kwargs)


def fx_option_npv(*args, **kwargs):
    return _lgm_fx_xva_mod().fx_option_npv(*args, **kwargs)


def _lgm_ir_options_mod():
    return import_module("pythonore.compute.lgm_ir_options")


def _irs_utils_mod():
    return import_module("pythonore.compute.irs_xva_utils")


def CapFloorDef(*args, **kwargs):
    return _lgm_ir_options_mod().CapFloorDef(*args, **kwargs)


def capfloor_npv(*args, **kwargs):
    return _lgm_ir_options_mod().capfloor_npv(*args, **kwargs)


def capfloor_npv_paths(*args, **kwargs):
    return _lgm_ir_options_mod().capfloor_npv_paths(*args, **kwargs)


def price_bermudan_from_ore_case(*args, **kwargs):
    return import_module("pythonore.runtime.bermudan").price_bermudan_from_ore_case(*args, **kwargs)


class _LazyBenchmarkModule:
    def __init__(self, module_name: str):
        self._module_name = module_name

    def main(self, argv):
        return import_module(self._module_name).main(argv)


benchmark_lgm_torch = _LazyBenchmarkModule("pythonore.benchmarks.benchmark_lgm_torch")
benchmark_lgm_torch_swap = _LazyBenchmarkModule("pythonore.benchmarks.benchmark_lgm_torch_swap")
benchmark_lgm_fx_hybrid_torch = _LazyBenchmarkModule("pythonore.benchmarks.benchmark_lgm_fx_hybrid_torch")
benchmark_lgm_fx_forward_torch = _LazyBenchmarkModule("pythonore.benchmarks.benchmark_lgm_fx_forward_torch")
benchmark_lgm_fx_portfolio_torch = _LazyBenchmarkModule("pythonore.benchmarks.benchmark_lgm_fx_portfolio_torch")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    def _bridge_fill_lgm_states(
        base_times: np.ndarray,
        base_x: np.ndarray,
        extra_times: np.ndarray,
        rng_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sim_times = np.unique(np.concatenate((base_times, extra_times)))
        if sim_times.size == base_times.size:
            return sim_times, base_x
        out = np.empty((sim_times.size, base_x.shape[1]), dtype=float)
        base_idx = np.searchsorted(sim_times, base_times)
        out[base_idx, :] = base_x
        local_rng = np.random.default_rng(int(rng_seed))
        zeta_base = np.asarray(model.zeta(base_times), dtype=float)
        for left in range(base_times.size - 1):
            mask = (sim_times > base_times[left] + 1.0e-12) & (sim_times < base_times[left + 1] - 1.0e-12)
            if not np.any(mask):
                continue
            local_times = sim_times[mask]
            u0 = float(zeta_base[left])
            u1 = float(zeta_base[left + 1])
            if u1 <= u0 + 1.0e-18:
                out[mask, :] = base_x[left, :]
                continue
            ui = np.asarray(model.zeta(local_times), dtype=float)
            weight = (ui - u0) / (u1 - u0)
            mean = (
                base_x[left, :][None, :]
                + weight[:, None] * (base_x[left + 1, :] - base_x[left, :])[None, :]
            )
            if local_times.size == 1:
                var = max((ui[0] - u0) * (u1 - ui[0]) / (u1 - u0), 0.0)
                if var <= 0.0:
                    out[mask, :] = mean
                else:
                    out[mask, :] = mean + math.sqrt(var) * local_rng.standard_normal((1, base_x.shape[1]))
                continue
            cov = np.minimum.outer(ui, ui) - u0 - np.outer(ui - u0, ui - u0) / (u1 - u0)
            cov = np.asarray(cov, dtype=float)
            cov.flat[:: cov.shape[0] + 1] = np.maximum(np.diag(cov), 0.0)
            chol = np.linalg.cholesky(cov + 1.0e-18 * np.eye(cov.shape[0]))
            draws = chol @ local_rng.standard_normal((cov.shape[0], base_x.shape[1]))
            out[mask, :] = mean + draws
        return sim_times, out

    def _simulate(times: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if str(getattr(model, "_measure", "LGM")).upper() == "BA":
            if str(draw_order).strip().lower() == "ore_path_major" and hasattr(rng, "seed"):
                local_rng = np.random.default_rng(int(getattr(rng, "seed")))
            elif str(draw_order).strip().lower() == "ore_path_major":
                local_rng = np.random.default_rng()
            else:
                local_rng = rng
            return simulate_ba_measure(model, times, n_paths, rng=local_rng, x0=0.0, y0=0.0)
        return simulate_lgm_measure(model, times, n_paths, rng=rng, x0=0.0, draw_order=draw_order), None

    if fixing_times is None or fixing_times.size == 0:
        x_exp, y_exp = _simulate(exposure_times)
        return x_exp, x_exp, exposure_times, y_exp, y_exp
    extra = np.asarray(fixing_times, dtype=float)
    extra = extra[extra > 1.0e-12]
    if extra.size == 0:
        x_exp, y_exp = _simulate(exposure_times)
        return x_exp, x_exp, exposure_times, y_exp, y_exp
    if (
        str(getattr(model, "_measure", "LGM")).upper() != "BA"
        and str(draw_order).strip().lower() == "ore_path_major"
    ):
        x_exp, y_exp = _simulate(exposure_times)
        sim_times, x_all = _bridge_fill_lgm_states(
            np.asarray(exposure_times, dtype=float),
            np.asarray(x_exp, dtype=float),
            extra,
            int(getattr(rng, "seed", 0)),
        )
        idx = np.searchsorted(sim_times, exposure_times)
        return x_exp, x_all, sim_times, y_exp, y_exp
    sim_times = np.unique(np.concatenate((exposure_times, extra)))
    x_all, y_all = _simulate(sim_times)
    idx = np.searchsorted(sim_times, exposure_times)
    if not np.allclose(sim_times[idx], exposure_times, atol=1.0e-12, rtol=0.0):
        raise ValueError("failed to align exposure times on simulated grid")
    y_exp = None if y_all is None else y_all[idx, :]
    return x_all[idx, :], x_all, sim_times, y_exp, y_all


def _deflate_npv_paths(
    model: Any,
    measure: str,
    p0_disc: Any,
    times: np.ndarray,
    x_paths: np.ndarray,
    npv_paths: np.ndarray,
    y_paths: np.ndarray | None = None,
) -> np.ndarray:
    if str(measure).strip().upper() != "BA":
        return deflate_lgm_npv_paths(
            model=model,
            p0_disc=p0_disc,
            times=times,
            x_paths=x_paths,
            npv_paths=npv_paths,
        )
    if y_paths is None:
        raise ValueError("BA measure deflation requires y_paths")
    t = np.asarray(times, dtype=float)
    x = np.asarray(x_paths, dtype=float)
    y = np.asarray(y_paths, dtype=float)
    v = np.asarray(npv_paths, dtype=float)
    if not (x.shape == y.shape == v.shape):
        raise ValueError("x_paths, y_paths and npv_paths must share the same shape")
    out = np.empty_like(v, dtype=float)
    for i, ti in enumerate(t):
        out[i, :] = v[i, :] / model.numeraire_ba(float(ti), x[i, :], y[i, :], p0_disc)
    return out


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
    exposure_profile_by_trade: dict[str, Any]
    exposure_profile_by_netting_set: dict[str, Any]
    ore_basel_epe: float
    ore_basel_eepe: float


def _build_ore_style_exposure_profile(
    entity_id: str,
    exposure_dates: Sequence[str],
    exposure_times: Sequence[float],
    epe: Sequence[float],
    ene: Sequence[float],
    pfe: Sequence[float],
    *,
    discount_factors: Sequence[float] | None = None,
    closeout_times: Sequence[float] | None = None,
    expected_collateral: Sequence[float] | None = None,
    asof_date: date | str | None = None,
) -> dict[str, Any]:
    return build_ore_exposure_profile_from_series(
        entity_id,
        exposure_dates,
        exposure_times,
        valuation_epe=epe,
        valuation_ene=ene,
        closeout_epe=epe,
        closeout_ene=ene,
        pfe=pfe,
        discount_factors=discount_factors,
        closeout_times=closeout_times,
        expected_collateral=expected_collateral,
        asof_date=asof_date,
    )


def _one_year_profile_value(profile: Mapping[str, Any], key: str) -> float:
    return _shared_one_year_profile_value(profile, key)


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
    lgm_param_source: str = "auto"
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
    use_reference_artifacts: bool = True,
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
    asof = _parse_ore_date(asof_date)
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml_path}")
    base = ore_xml_path.parent
    run_dir = base.parent
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    market_data_path = (input_dir / setup_params.get("marketDataFile", "")).resolve()
    curve_config_path = (input_dir / setup_params.get("curveConfigFile", "")).resolve()
    conventions_path = (input_dir / setup_params.get("conventionsFile", "")).resolve()
    todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    portfolio_xml = (input_dir / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    trade_type = ore_snapshot_mod._get_trade_type(portfolio_root, trade_id)
    counterparty = ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id)
    netting_set_id = ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id)
    if trade_type == "FxForward":
        return _build_fx_forward_pricing_payload(ore_xml, use_reference_artifacts=use_reference_artifacts)
    if trade_type == "FxOption":
        return _build_fx_option_pricing_payload(ore_xml, use_reference_artifacts=use_reference_artifacts)
    if trade_type == "Swaption":
        trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
        style = (
            (trade.findtext("./SwaptionData/OptionData/Style") or "").strip().lower()
            if trade is not None
            else ""
        )
        if style == "bermudan":
            return _build_bermudan_swaption_pricing_payload(ore_xml)
        return _build_swaption_pricing_payload(ore_xml, use_reference_artifacts=use_reference_artifacts)
    if trade_type == "CapFloor":
        trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
        leg_type = (trade.findtext("./CapFloorData/LegData/LegType") or "").strip().upper() if trade is not None else ""
        if leg_type in {"CPI", "YY"}:
            ore_xml_path2, asof_date2, market_data_path2, _, _ = ore_snapshot_mod._resolve_ore_run_files(ore_xml)
            npv_csv2 = _find_reference_npv_file(ore_xml, trade_id=trade_id)
            npv_details2 = (
                ore_snapshot_mod._load_ore_npv_details(npv_csv2, trade_id=trade_id)
                if npv_csv2 is not None
                else {
                    "npv": None,
                    "maturity_date": (
                        trade.findtext("./CapFloorData/LegData/ScheduleData/Rules/EndDate")
                        or trade.findtext("./CapFloorData/LegData/ScheduleData/Dates/Dates/Date")
                        or ""
                    ),
                    "maturity_time": ore_snapshot_mod._year_fraction_from_day_counter(
                        _parse_ore_date(asof_date2),
                        _parse_ore_date(
                            trade.findtext("./CapFloorData/LegData/ScheduleData/Rules/EndDate")
                            or trade.findtext("./CapFloorData/LegData/ScheduleData/Dates/Dates/Date")
                            or asof_date2
                        ),
                        "A365F",
                    ),
                }
            )
            ccy = (trade.findtext("./CapFloorData/LegData/Currency") or "EUR").strip()
            p0_disc = _build_discount_curve_from_market_fit(ore_xml, ccy)
            index_name = (
                trade.findtext("./CapFloorData/LegData/CPILegData/Index")
                or trade.findtext("./CapFloorData/LegData/YYLegData/Index")
                or ""
            ).strip()
            curve_type = "YY" if leg_type == "YY" else "ZC"
            inf_curve = load_inflation_curve_from_market_data(market_data_path2, asof_date2, index_name, curve_type=curve_type)
            strike = float(
                trade.findtext("./CapFloorData/Caps/Cap")
                or trade.findtext("./CapFloorData/Floors/Floor")
                or trade.findtext("./CapFloorData/LegData/CPILegData/Rates/Rate")
                or 0.0
            )
            option_type = "Cap" if (trade.findtext("./CapFloorData/Caps/Cap") or "").strip() else "Floor"
            maturity_label = (
                trade.findtext("./CapFloorData/LegData/ScheduleData/Rules/EndDate")
                or str(npv_details2["maturity_date"])
            )
            maturity_years = float(npv_details2["maturity_time"])
            tenor_label = f"{max(int(round(maturity_years)), 1)}Y"
            market_surface_price = load_zero_inflation_surface_quote(
                market_data_path2, asof_date2, index_name, tenor_label, strike, option_type
            )
            return {
                "trade_id": trade_id,
                "trade_type": "CapFloor",
                "inflation_product": True,
                "inflation_kind": leg_type,
                "counterparty": counterparty,
                "netting_set_id": netting_set_id,
                "currency": ccy,
                "index": index_name,
                "option_type": option_type,
                "strike": strike,
                "notional": float(trade.findtext("./CapFloorData/LegData/Notionals/Notional") or 0.0),
                "maturity_date": str(npv_details2["maturity_date"]),
                "maturity_time": maturity_years,
                "p0_disc": p0_disc,
                "inflation_curve": inf_curve,
                "market_surface_price": market_surface_price,
                "ore_t0_npv": float(npv_details2["npv"]) if npv_details2.get("npv") is not None else None,
                "reference_output_dirs": [str(npv_csv2.parent)] if npv_csv2 is not None else [],
                "using_expected_output": bool(npv_csv2 is not None and _classify_reference_dir(ore_xml, npv_csv2.parent) == "expected_output"),
                "pricing_mode": f"python_inflation_{curve_type.lower()}_capfloor",
                **_inflation_model_diagnostics(ore_xml, index_name),
            }
        return _build_capfloor_pricing_payload(ore_xml, use_reference_artifacts=use_reference_artifacts)
    if trade_type == "Swap" and _is_inflation_swap_trade(ore_xml):
        ore_xml_path2, asof_date2, market_data_path2, _, _ = ore_snapshot_mod._resolve_ore_run_files(ore_xml)
        trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
        if trade is None:
            raise ValueError(f"trade '{trade_id}' not found in {portfolio_xml}")
        npv_csv2 = _find_reference_npv_file(ore_xml, trade_id=trade_id)
        legs_xml = trade.findall("./SwapData/LegData")
        inflation_leg = next((leg for leg in legs_xml if (leg.findtext("./LegType") or "").strip().upper() in {"CPI", "YY"}), None)
        fixed_leg = next((leg for leg in legs_xml if (leg.findtext("./LegType") or "").strip().upper() in {"FIXED", "ZEROCOUPONFIXED"}), None)
        float_leg = next((leg for leg in legs_xml if (leg.findtext("./LegType") or "").strip().upper() == "FLOATING"), None)
        if inflation_leg is None:
            raise ValueError(f"no inflation leg found for trade '{trade_id}'")
        leg_type = (inflation_leg.findtext("./LegType") or "").strip().upper()
        maturity_date = (
            inflation_leg.findtext("./ScheduleData/Rules/EndDate")
            or inflation_leg.findtext("./ScheduleData/Dates/Dates/Date")
            or (fixed_leg.findtext("./ScheduleData/Rules/EndDate") if fixed_leg is not None else None)
            or (float_leg.findtext("./ScheduleData/Rules/EndDate") if float_leg is not None else None)
            or asof_date2
        )
        npv_details2 = (
            ore_snapshot_mod._load_ore_npv_details(npv_csv2, trade_id=trade_id)
            if npv_csv2 is not None
            else {
                "npv": None,
                "maturity_date": maturity_date,
                "maturity_time": ore_snapshot_mod._year_fraction_from_day_counter(
                    _parse_ore_date(asof_date2),
                    _parse_ore_date(maturity_date),
                    "A365F",
                ),
            }
        )
        ccy = (inflation_leg.findtext("./Currency") or fixed_leg.findtext("./Currency") if fixed_leg is not None else "EUR").strip()
        p0_disc = _build_discount_curve_from_market_fit(ore_xml, ccy)
        index_name = (
            inflation_leg.findtext("./CPILegData/Index")
            or inflation_leg.findtext("./YYLegData/Index")
            or ""
        ).strip()
        curve_type = "YY" if leg_type == "YY" else "ZC"
        inf_curve = load_inflation_curve_from_market_data(market_data_path2, asof_date2, index_name, curve_type=curve_type)
        return {
            "trade_id": trade_id,
            "trade_type": "Swap",
            "inflation_product": True,
            "inflation_kind": leg_type,
            "counterparty": counterparty,
            "netting_set_id": netting_set_id,
            "currency": ccy,
            "index": index_name,
            "notional": float(
                inflation_leg.findtext("./Notionals/Notional")
                or (fixed_leg.findtext("./Notionals/Notional") if fixed_leg is not None else "0")
                or (float_leg.findtext("./Notionals/Notional") if float_leg is not None else "0")
            ),
            "base_cpi": float(inflation_leg.findtext("./CPILegData/BaseCPI") or 100.0),
            "fixed_rate": float(
                (fixed_leg.findtext("./FixedLegData/Rates/Rate") if fixed_leg is not None else None)
                or (fixed_leg.findtext("./ZeroCouponFixedLegData/Rates/Rate") if fixed_leg is not None else None)
                or 0.0
            ),
            "has_float_leg": float_leg is not None,
            "schedule_tenor": (
                inflation_leg.findtext("./ScheduleData/Rules/Tenor")
                or (fixed_leg.findtext("./ScheduleData/Rules/Tenor") if fixed_leg is not None else None)
                or "1Y"
            ),
            "maturity_date": str(npv_details2["maturity_date"]),
            "maturity_time": float(npv_details2["maturity_time"]),
            "p0_disc": p0_disc,
            "inflation_curve": inf_curve,
            "ore_t0_npv": float(npv_details2["npv"]) if npv_details2.get("npv") is not None else None,
            "exposure_csv": _find_reference_output_file(ore_xml, f"exposure_trade_{trade_id}.csv"),
            "xva_csv": _find_reference_output_file(ore_xml, "xva.csv"),
            "todaysmarket_xml": ore_snapshot_mod._resolve_ore_run_files(ore_xml)[3],
            "market_data_path": market_data_path2,
            "reference_output_dirs": [str(npv_csv2.parent)] if npv_csv2 is not None else [],
            "using_expected_output": bool(npv_csv2 is not None and _classify_reference_dir(ore_xml, npv_csv2.parent) == "expected_output"),
            "pricing_mode": f"python_inflation_{curve_type.lower()}_swap",
            **_inflation_model_diagnostics(ore_xml, index_name),
        }
    forward_column = ore_snapshot_mod._get_float_index(portfolio_root, trade_id) if trade_type in {"Swap", "Swaption"} else ""
    report_ccy = (
        (ore_root.findtext("./Analytics/Analytic[@type='npv']/Parameter[@name='baseCurrency']") or "").strip()
        or (ore_root.findtext("./Setup/Parameter[@name='baseCurrency']") or "").strip()
    )
    trade_ccy = ore_snapshot_mod._get_single_trade_currency(portfolio_root, trade_id) if trade_type == "Swap" else ""

    sim_config_id = markets_params.get("simulation", "libor")
    pricing_config_id = markets_params.get("pricing", sim_config_id)
    sim_analytic = ore_root.find("./Analytics/Analytic[@type='simulation']")
    has_explicit_simulation_analytic = sim_analytic is not None
    sim_params: dict[str, str] = {}
    simulation_xml: Path | None = None
    if sim_analytic is not None:
        sim_params = {
            n.attrib.get("name", ""): (n.text or "").strip()
            for n in sim_analytic.findall("./Parameter")
        }
        simulation_xml = (input_dir / sim_params.get("simulationConfigFile", "simulation.xml")).resolve()
    else:
        candidate = (input_dir / "simulation.xml").resolve()
        if candidate.exists():
            simulation_xml = candidate

    if simulation_xml is not None and simulation_xml.exists():
        sim_root = ET.parse(simulation_xml).getroot()
        domestic_ccy = (
            sim_root.findtext("./DomesticCcy")
            or sim_root.findtext("./CrossAssetModel/DomesticCcy")
            or "EUR"
        ).strip()
        if trade_type == "Swap" and trade_ccy:
            domestic_ccy = trade_ccy
        model_day_counter = ore_snapshot_mod._normalize_day_counter_name(
            (sim_root.findtext("./DayCounter") or "A365F").strip()
        )
        node_tenors = load_simulation_yield_tenors(str(simulation_xml))
    else:
        if trade_type != "Swap" or not _is_plain_vanilla_swap_trade(ore_xml):
            raise ValueError(f"Missing Analytics/Analytic[@type='simulation'] in {ore_xml_path}")
        domestic_ccy = (
            trade_ccy
            or (
                (ore_root.findtext("./Analytics/Analytic[@type='npv']/Parameter[@name='baseCurrency']") or "").strip()
            )
            or "EUR"
        )
        model_day_counter = "A365F"
        node_tenors = []

    tm_root = ET.parse(todaysmarket_xml).getroot()
    discount_column = ore_snapshot_mod._resolve_discount_column(tm_root, pricing_config_id, domestic_ccy)
    xva_discount_column = ore_snapshot_mod._resolve_discount_column(tm_root, sim_config_id, domestic_ccy)

    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    if npv_csv is None:
        raise FileNotFoundError(f"ORE output file not found (run ORE first): {output_path / 'npv.csv'}")
    curves_csv = _find_reference_output_file(ore_xml, "curves.csv") if use_reference_artifacts else None
    if curves_csv is not None:
        curve_dates_by_col = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv), [discount_column], asof_date=asof_date, day_counter=model_day_counter
        )
        _, curve_times_disc, curve_dfs_disc = curve_dates_by_col[discount_column]
        p0_disc = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times_disc, curve_dfs_disc)))
        ql_disc_handle = _ql_handle_from_curve_pairs(asof, curve_times_disc, curve_dfs_disc) if ql is not None else None
        if forward_column == discount_column:
            p0_fwd = p0_disc
            ql_fwd_handle = ql_disc_handle
        else:
            _, curve_times_fwd, curve_dfs_fwd = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
                str(curves_csv), [forward_column], asof_date=asof_date, day_counter=model_day_counter
            )[forward_column]
            p0_fwd = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times_fwd, curve_dfs_fwd)))
            ql_fwd_handle = _ql_handle_from_curve_pairs(asof, curve_times_fwd, curve_dfs_fwd) if ql is not None else None
    else:
        if trade_type == "Swap" and _active_discount_curve_uses_cross_currency_segments(
            tm_root, curve_config_path, pricing_config_id, domestic_ccy
        ):
            raise ValueError(
                f"active discount curve for '{domestic_ccy}' uses cross-currency segments and curves.csv is unavailable"
            )
        p0_disc, p0_fwd, ql_disc_handle, ql_fwd_handle = _build_fitted_discount_and_forward_curves(
            ore_xml,
            asof=asof,
            currency=domestic_ccy,
            float_index=forward_column,
        )

    if simulation_xml is not None and simulation_xml.exists():
        params_dict, _, _ = ore_snapshot_mod.resolve_lgm_params(
            ore_xml_path=str(ore_xml_path),
            input_dir=input_dir,
            output_path=output_path,
            market_data_path=market_data_path,
            curve_config_path=curve_config_path,
            conventions_path=conventions_path,
            todaysmarket_xml_path=todaysmarket_xml,
            simulation_xml_path=simulation_xml,
            domestic_ccy=domestic_ccy,
        )
    else:
        params_dict = {
            "alpha_times": (1.0,),
            "alpha_values": (0.01, 0.01),
            "kappa_times": (1.0,),
            "kappa_values": (0.03, 0.03),
            "shift": 0.0,
            "scaling": 1.0,
        }
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

    flows_csv = _find_reference_output_file(ore_xml, "flows.csv") if use_reference_artifacts else None
    legs = None
    trade_cashflows = None
    use_cashflow_replay = False
    leg_source = "portfolio"
    if flows_csv is not None and flows_csv.exists():
        try:
            trade_cashflows = load_trade_cashflows_from_flows(
                str(flows_csv), trade_id=trade_id, asof_date=asof_date, time_day_counter=model_day_counter
            )
            cashflow_ccys = {
                str(leg.get("ccy") or "").strip().upper()
                for leg in trade_cashflows.get("rate_legs", [])
                if str(leg.get("ccy") or "").strip()
            }
            if cashflow_ccys and (len(cashflow_ccys) > 1 or (trade_ccy and cashflow_ccys != {trade_ccy.upper()})):
                use_cashflow_replay = True
        except Exception:
            trade_cashflows = None
    if flows_csv is not None and flows_csv.exists():
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
        "trade_type": trade_type,
        "counterparty": counterparty,
        "netting_set_id": netting_set_id,
        "model": model,
        "legs": legs,
        "p0_disc": p0_disc,
        "p0_fwd": p0_fwd,
        "ql_disc_handle": ql_disc_handle,
        "ql_fwd_handle": ql_fwd_handle,
        "p0_xva_disc": p0_disc if xva_discount_column == discount_column else None,
        "ore_t0_npv": ore_t0_npv,
        "maturity_date": str(npv_details["maturity_date"]),
        "maturity_time": float(npv_details["maturity_time"]),
        "leg_source": leg_source,
        "discount_column": discount_column,
        "forward_column": forward_column,
        "pricing_currency": domestic_ccy,
        "report_ccy": report_ccy or domestic_ccy,
        "fx_to_report": _load_fx_conversion_to_report(
            tm_root, market_data_path, asof_date, pricing_config_id, domestic_ccy, report_ccy or domestic_ccy
        ),
        "trade_cashflows": trade_cashflows.get("rate_legs", []) if trade_cashflows is not None else [],
        "use_cashflow_replay": use_cashflow_replay,
        "reference_output_dirs": sorted(
            {
                str(path.parent)
                for path in (curves_csv, npv_csv, flows_csv)
                if path is not None
            }
        ),
        "using_expected_output": any(
            _classify_reference_dir(ore_xml, path.parent) == "expected_output"
            for path in (curves_csv, npv_csv, flows_csv)
            if path is not None
        ),
    }


def _load_market_quote_value(market_data_path: Path, asof_date: str, quote_id: str) -> float:
    asof_compact = asof_date.replace("-", "")
    asof_dash = asof_date
    with open(market_data_path, encoding="utf-8") as handle:
        for line in handle:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            parts = txt.split()
            if len(parts) < 3:
                continue
            if parts[0] not in {asof_compact, asof_dash}:
                continue
            if parts[1] == quote_id:
                return float(parts[2])
    raise ValueError(f"Quote '{quote_id}' not found for asof date {asof_date} in {market_data_path}")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _tenor_to_years(tenor: str) -> float:
    text = str(tenor).strip().upper()
    match = re.fullmatch(r"([0-9]+)([DWMY])", text)
    if not match:
        raise ValueError(f"Unsupported tenor '{tenor}'")
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "D":
        return value / 365.0
    if unit == "W":
        return value * 7.0 / 365.0
    if unit == "M":
        return value / 12.0
    if unit == "Y":
        return value
    raise ValueError(f"Unsupported tenor '{tenor}'")


def _load_fx_atm_vol(
    market_data_path: Path,
    asof_date: str,
    base_ccy: str,
    quote_ccy: str,
    maturity_time: float,
) -> float:
    asof_compact = asof_date.replace("-", "")
    asof_dash = asof_date
    prefix = f"FX_OPTION/RATE_LNVOL/{base_ccy}/{quote_ccy}/"
    points: list[tuple[float, float]] = []
    with open(market_data_path, encoding="utf-8") as handle:
        for line in handle:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            parts = txt.split()
            if len(parts) < 3 or parts[0] not in {asof_compact, asof_dash}:
                continue
            quote_id = parts[1]
            if not quote_id.startswith(prefix) or not quote_id.endswith("/ATM"):
                continue
            tenor = quote_id[len(prefix) : -len("/ATM")]
            try:
                t = _tenor_to_years(tenor)
                vol = float(parts[2])
            except Exception:
                continue
            points.append((t, vol))
    if not points:
        raise ValueError(
            f"No FX ATM volatility quotes found for {base_ccy}/{quote_ccy} in {market_data_path}"
        )
    points.sort(key=lambda item: item[0])
    target = max(float(maturity_time), 0.0)
    if target <= points[0][0]:
        return float(points[0][1])
    if target >= points[-1][0]:
        return float(points[-1][1])
    for (t0, v0), (t1, v1) in zip(points[:-1], points[1:]):
        if t0 <= target <= t1:
            if t1 <= t0:
                return float(v1)
            w = (target - t0) / (t1 - t0)
            total_var0 = (float(v0) ** 2) * t0
            total_var1 = (float(v1) ** 2) * t1
            total_var = (1.0 - w) * total_var0 + w * total_var1
            return math.sqrt(max(total_var / max(target, 1.0e-12), 0.0))
    return float(points[-1][1])


def _parse_ore_date(text: str) -> date:
    raw = (text or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return date.fromisoformat(f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}")
    return ore_snapshot_mod._normalize_date_input(raw)


def _year_fraction_a365(start: date, end: date) -> float:
    return max(float((end - start).days) / 365.0, 0.0)


def _strike_quote_candidates(strike: float) -> tuple[str, ...]:
    base = f"{float(strike):.10f}".rstrip("0").rstrip(".")
    candidates = {
        base,
        f"{float(strike):.0f}",
        f"{float(strike):.1f}".rstrip("0").rstrip("."),
        f"{float(strike):.2f}".rstrip("0").rstrip("."),
        f"{float(strike):.3f}".rstrip("0").rstrip("."),
        f"{float(strike):.6f}".rstrip("0").rstrip("."),
    }
    return tuple(sorted(candidates, key=len))


def _market_label_to_time(asof: date, label: str) -> float:
    txt = str(label or "").strip()
    if not txt:
        raise ValueError("empty market label")
    try:
        return _year_fraction_a365(asof, _parse_ore_date(txt))
    except Exception:
        return _tenor_to_years(txt)


def _interp_linear(points: list[tuple[float, float]], target: float, *, flat_extrapolation: bool = True) -> float:
    if not points:
        raise ValueError("interpolation requires at least one point")
    ordered = sorted((float(x), float(y)) for x, y in points)
    if target <= ordered[0][0]:
        if flat_extrapolation or len(ordered) == 1:
            return float(ordered[0][1])
    if target >= ordered[-1][0]:
        if flat_extrapolation or len(ordered) == 1:
            return float(ordered[-1][1])
    for (x0, y0), (x1, y1) in zip(ordered[:-1], ordered[1:]):
        if x0 <= target <= x1:
            if x1 <= x0:
                return float(y1)
            w = (target - x0) / (x1 - x0)
            return float((1.0 - w) * y0 + w * y1)
    return float(ordered[-1][1])


def _interp_total_variance(points: list[tuple[float, float]], target: float) -> float:
    if not points:
        raise ValueError("variance interpolation requires at least one point")
    ordered = sorted((max(float(t), 1.0e-12), max(float(v), 0.0)) for t, v in points)
    if target <= ordered[0][0]:
        return float(ordered[0][1])
    if target >= ordered[-1][0]:
        return float(ordered[-1][1])
    for (t0, v0), (t1, v1) in zip(ordered[:-1], ordered[1:]):
        if t0 <= target <= t1:
            if t1 <= t0:
                return float(v1)
            w = (target - t0) / (t1 - t0)
            total_var0 = (v0 * v0) * t0
            total_var1 = (v1 * v1) * t1
            total_var = (1.0 - w) * total_var0 + w * total_var1
            return math.sqrt(max(total_var / max(target, 1.0e-12), 0.0))
    return float(ordered[-1][1])


def _parse_market_quotes(market_data_path: Path, asof_date: str) -> dict[str, float]:
    asof_compact = asof_date.replace("-", "")
    asof_dash = asof_date
    quotes: dict[str, float] = {}
    with open(market_data_path, encoding="utf-8") as handle:
        for line in handle:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            rows: list[list[str]] = []
            if "," in txt:
                rows.append([part.strip() for part in txt.split(",")])
            rows.append(txt.split())
            for parts in rows:
                if len(parts) < 3 or parts[0] not in {asof_compact, asof_dash}:
                    continue
                try:
                    quotes[parts[1]] = float(parts[2])
                    break
                except Exception:
                    continue
    return quotes


def _legacy_section_with_fallback(root: ET.Element, tag: str, section_id: str | None) -> ET.Element | None:
    if section_id:
        node = root.find(f"./{tag}[@id='{section_id}']")
        if node is not None:
            return node
    return root.find(f"./{tag}")


def _load_equity_curve_spec(
    todaysmarket_xml: Path,
    curveconfig_path: Path,
    *,
    pricing_config_id: str,
    equity_name: str,
) -> dict[str, Any]:
    tm_root = ET.parse(todaysmarket_xml).getroot()
    eq_section_id = ore_snapshot_mod._resolve_todaysmarket_section_id(
        tm_root,
        pricing_config_id,
        "EquityCurvesId",
        "EquityCurves",
    ) or "default"
    mapping = _legacy_section_with_fallback(tm_root, "EquityCurves", eq_section_id)
    if mapping is None:
        raise ValueError("todaysmarket.xml missing EquityCurves section")
    handle = mapping.findtext(f"./EquityCurve[@name='{equity_name}']")
    if not handle:
        raise ValueError(f"todaysmarket.xml missing EquityCurve mapping for '{equity_name}'")
    curve_id = handle.strip().split("/")[-1]
    cfg_root = ET.parse(curveconfig_path).getroot()
    node = next(
        (
            candidate
            for candidate in cfg_root.findall(".//EquityCurve")
            if (candidate.findtext("./CurveId") or "").strip() == curve_id
        ),
        None,
    )
    if node is None:
        raise ValueError(f"curveconfig.xml missing EquityCurve with CurveId '{curve_id}'")
    return {
        "curve_id": curve_id,
        "currency": (node.findtext("./Currency") or "").strip().upper(),
        "forecasting_curve": (node.findtext("./ForecastingCurve") or "").strip(),
        "curve_type": (node.findtext("./Type") or "").strip(),
        "spot_quote": (node.findtext("./SpotQuote") or "").strip(),
        "quotes": [q.text.strip() for q in node.findall("./Quotes/Quote") if (q.text or "").strip()],
        "day_counter": (node.findtext("./DayCounter") or "A365").strip(),
    }


def _load_equity_vol_spec(
    todaysmarket_xml: Path,
    curveconfig_path: Path,
    *,
    pricing_config_id: str,
    equity_name: str,
) -> dict[str, Any]:
    tm_root = ET.parse(todaysmarket_xml).getroot()
    eq_section_id = ore_snapshot_mod._resolve_todaysmarket_section_id(
        tm_root,
        pricing_config_id,
        "EquityVolatilitiesId",
        "EquityVolatilities",
    ) or "default"
    mapping = _legacy_section_with_fallback(tm_root, "EquityVolatilities", eq_section_id)
    if mapping is None:
        raise ValueError("todaysmarket.xml missing EquityVolatilities section")
    handle = mapping.findtext(f"./EquityVolatility[@name='{equity_name}']")
    if not handle:
        raise ValueError(f"todaysmarket.xml missing EquityVolatility mapping for '{equity_name}'")
    curve_id = handle.strip().split("/")[-1]
    cfg_root = ET.parse(curveconfig_path).getroot()
    node = next(
        (
            candidate
            for candidate in cfg_root.findall(".//EquityVolatility")
            if (candidate.findtext("./CurveId") or "").strip() == curve_id
        ),
        None,
    )
    if node is None:
        raise ValueError(f"curveconfig.xml missing EquityVolatility with CurveId '{curve_id}'")
    dimension = (node.findtext("./Dimension") or "ATM").strip()
    strike_surface = node.find("./StrikeSurface")
    if strike_surface is not None:
        dimension = "StrikeSurface"
    return {
        "curve_id": curve_id,
        "currency": (node.findtext("./Currency") or "").strip().upper(),
        "dimension": dimension,
        "quote_type": (
            strike_surface.findtext("./QuoteType")
            if strike_surface is not None
            else node.findtext("./QuoteType")
        ) or "",
        "exercise_type": (strike_surface.findtext("./ExerciseType") if strike_surface is not None else "") or "",
        "expiries": [
            x.strip()
            for x in (
                (strike_surface.findtext("./Expiries") if strike_surface is not None else node.findtext("./Expiries"))
                or ""
            ).replace("\n", "").split(",")
            if x.strip()
        ],
        "strikes": [
            x.strip()
            for x in (
                (strike_surface.findtext("./Strikes") if strike_surface is not None else node.findtext("./Strikes"))
                or ""
            ).replace("\n", "").split(",")
            if x.strip()
        ],
    }


def _load_equity_option_premium_quote(
    quotes: dict[str, float],
    *,
    equity_name: str,
    currency: str,
    exercise_date: str,
    strike: float,
    option_type: str,
) -> float | None:
    suffix = "C" if str(option_type).strip().upper().startswith("C") else "P"
    for strike_label in _strike_quote_candidates(strike):
        quote_id = f"EQUITY_OPTION/PRICE/{equity_name}/{currency}/{exercise_date}/{strike_label}/{suffix}"
        if quote_id in quotes:
            return float(quotes[quote_id])
    return None


def _build_zero_rate_curve_from_quotes(points: list[tuple[float, float]]):
    if not points:
        return lambda t: 1.0
    ordered = sorted((max(float(t), 0.0), float(rate)) for t, rate in points)

    def _curve(t: float) -> float:
        tt = max(float(t), 0.0)
        if tt <= 1.0e-12:
            return 1.0
        rate = _interp_linear(ordered, tt, flat_extrapolation=True)
        return math.exp(-rate * tt)

    return _curve


def _load_equity_dividend_curve(
    curve_spec: dict[str, Any],
    *,
    quotes: dict[str, float],
    asof: date,
):
    points: list[tuple[float, float]] = []
    for quote_id in curve_spec.get("quotes", []):
        if quote_id not in quotes:
            continue
        label = quote_id.rsplit("/", 1)[-1]
        points.append((_market_label_to_time(asof, label), float(quotes[quote_id])))
    return _build_zero_rate_curve_from_quotes(points)


def _load_equity_forward_curve(
    curve_spec: dict[str, Any],
    *,
    quotes: dict[str, float],
    asof: date,
    spot: float,
):
    points: list[tuple[float, float]] = [(0.0, float(spot))]
    for quote_id in curve_spec.get("quotes", []):
        if quote_id not in quotes:
            continue
        label = quote_id.rsplit("/", 1)[-1]
        points.append((_market_label_to_time(asof, label), float(quotes[quote_id])))
    ordered = sorted(points)

    def _curve(t: float) -> float:
        return _interp_linear(ordered, max(float(t), 0.0), flat_extrapolation=True)

    return _curve


def _load_equity_smile_vol(
    quotes: dict[str, float],
    *,
    asof: date,
    equity_name: str,
    currency: str,
    maturity_time: float,
    strike: float,
    spec: dict[str, Any],
) -> float:
    strike_candidates = {str(s).strip() for s in spec.get("strikes", []) if str(s).strip()}
    strike_labels = list(strike_candidates | set(_strike_quote_candidates(strike)) | {"ATMF"})
    rows: dict[float, list[tuple[float, float]]] = {}
    for expiry_label in spec.get("expiries", []):
        expiry_time = _market_label_to_time(asof, expiry_label)
        strike_points: list[tuple[float, float]] = []
        for strike_label in strike_labels:
            quote_id = f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/{expiry_label}/{strike_label}"
            if quote_id not in quotes:
                continue
            if strike_label == "ATMF":
                continue
            try:
                strike_points.append((float(strike_label), float(quotes[quote_id])))
            except Exception:
                continue
        exact_quotes = [vol for k, vol in strike_points if abs(k - float(strike)) <= 1.0e-10]
        if exact_quotes:
            rows.setdefault(float(strike), []).append((expiry_time, float(exact_quotes[0])))
            continue
        if strike_points:
            rows.setdefault(float(strike), []).append(
                (expiry_time, _interp_linear(strike_points, float(strike), flat_extrapolation=True))
            )
            continue
        atm_quote_id = f"EQUITY_OPTION/RATE_LNVOL/{equity_name}/{currency}/{expiry_label}/ATMF"
        if atm_quote_id in quotes:
            rows.setdefault(float(strike), []).append((expiry_time, float(quotes[atm_quote_id])))
    strike_row = rows.get(float(strike), [])
    if not strike_row:
        raise ValueError(f"no equity smile quotes found for {equity_name} {currency} strike {strike}")
    return _interp_total_variance(strike_row, max(float(maturity_time), 1.0e-12))


def _equity_forward_from_market_inputs(
    *,
    curve_spec: dict[str, Any],
    spot: float,
    maturity_time: float,
    forecast_curve: Any,
    quotes: dict[str, float],
    asof: date,
) -> float:
    curve_type = str(curve_spec.get("curve_type") or "").strip().lower()
    if curve_type == "dividendyield":
        dividend_curve = _load_equity_dividend_curve(curve_spec, quotes=quotes, asof=asof)
        return float(spot) * float(dividend_curve(maturity_time)) / max(float(forecast_curve(maturity_time)), 1.0e-12)
    if curve_type == "forwardprice":
        forward_curve = _load_equity_forward_curve(curve_spec, quotes=quotes, asof=asof, spot=spot)
        return float(forward_curve(maturity_time))
    return float(spot) / max(float(forecast_curve(maturity_time)), 1.0e-12)


def _black_forward_option_npv(*, forward: float, strike: float, maturity_time: float, vol: float, discount: float, call: bool) -> float:
    tt = max(float(maturity_time), 0.0)
    if tt <= 1.0e-12 or vol <= 1.0e-12:
        intrinsic = max(float(forward) - float(strike), 0.0) if call else max(float(strike) - float(forward), 0.0)
        return float(discount) * intrinsic
    std_dev = max(float(vol) * math.sqrt(tt), 1.0e-12)
    d1 = (math.log(max(float(forward), 1.0e-12) / max(float(strike), 1.0e-12)) + 0.5 * std_dev * std_dev) / std_dev
    d2 = d1 - std_dev
    if call:
        return float(discount) * (float(forward) * _normal_cdf(d1) - float(strike) * _normal_cdf(d2))
    return float(discount) * (float(strike) * _normal_cdf(-d2) - float(forward) * _normal_cdf(-d1))


def _build_equity_pricing_payload(ore_xml: Path, *, use_reference_artifacts: bool = True) -> dict[str, Any]:
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
    asof = _parse_ore_date(asof_date)
    base = ore_xml_path.parent
    run_dir = base.parent
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    market_data_path = (input_dir / setup_params.get("marketDataFile", "")).resolve()
    curve_config_path = (input_dir / setup_params.get("curveConfigFile", "")).resolve()
    todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    portfolio_xml = (input_dir / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    pricing_config_id = markets_params.get("pricing", "default")
    tm_root = ET.parse(todaysmarket_xml).getroot()

    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
    if trade is None:
        trade = portfolio_root.find("./Trade") or portfolio_root.find(".//Trade")
    if trade is None:
        raise ValueError(f"trade '{trade_id}' not found in {portfolio_xml}")
    trade_type = (trade.findtext("./TradeType") or "").strip()
    quotes = _parse_market_quotes(market_data_path, asof_date)
    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    if npv_csv is None:
        raise FileNotFoundError(f"pricing reference not found for trade '{trade_id}'")
    npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id)
    try:
        counterparty = ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id)
    except Exception:
        counterparty = ""
    try:
        netting_set_id = ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id)
    except Exception:
        netting_set_id = ""

    curves_analytic = ore_root.find("./Analytics/Analytic[@type='curves']")
    curves_output_name = (
        (curves_analytic.findtext("./Parameter[@name='outputFileName']") if curves_analytic is not None else None)
        or "curves.csv"
    )
    curves_csv = _find_reference_output_file(ore_xml, curves_output_name) if use_reference_artifacts else None

    def _equity_discount_curve(ccy: str):
        try:
            discount_column = ore_snapshot_mod._resolve_discount_column(tm_root, pricing_config_id, ccy)
            if curves_csv is None:
                raise FileNotFoundError(curves_output_name)
            curve_dates = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
                str(curves_csv),
                [discount_column],
                asof_date=asof_date,
                day_counter="A365F",
            )
            _, curve_times, curve_dfs = curve_dates[discount_column]
            return ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times, curve_dfs)))
        except Exception:
            return _build_discount_curve_from_market_fit(ore_xml, ccy)

    def _equity_forecast_curve(curve_spec: dict[str, Any]):
        forecasting_curve = str(curve_spec.get("forecasting_curve") or "").strip()
        ccy = str(curve_spec.get("currency") or "").strip().upper()
        if curves_csv is not None and forecasting_curve:
            handle = f"Yield/{ccy}/{forecasting_curve}"
            try:
                forecast_column = ore_snapshot_mod._handle_to_curve_name(tm_root, handle)
                curve_dates = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
                    str(curves_csv),
                    [forecast_column],
                    asof_date=asof_date,
                    day_counter="A365F",
                )
                _, curve_times, curve_dfs = curve_dates[forecast_column]
                return ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times, curve_dfs)))
            except Exception:
                pass
        return _build_discount_curve_from_market_fit(ore_xml, ccy)

    if trade_type == "EquityOption":
        data = trade.find("./EquityOptionData")
        if data is None:
            raise ValueError(f"EquityOptionData missing for trade '{trade_id}'")
        equity_name = (data.findtext("./Name") or "").strip()
        currency = (data.findtext("./Currency") or "").strip().upper()
        strike = float((data.findtext("./Strike") or "0").strip())
        quantity = float((data.findtext("./Quantity") or "0").strip())
        option_data = data.find("./OptionData")
        option_type = (option_data.findtext("./OptionType") if option_data is not None else "Call") or "Call"
        long_short = (option_data.findtext("./LongShort") if option_data is not None else "Long") or "Long"
        exercise_date = (
            option_data.findtext("./ExerciseDates/ExerciseDate")
            if option_data is not None
            else data.findtext("./ExerciseDate")
            or ""
        ).strip()
        maturity_time = _year_fraction_a365(asof, _parse_ore_date(exercise_date))
        curve_spec = _load_equity_curve_spec(
            todaysmarket_xml,
            curve_config_path,
            pricing_config_id=pricing_config_id,
            equity_name=equity_name,
        )
        spot = float(quotes[curve_spec["spot_quote"]])
        premium_quote = _load_equity_option_premium_quote(
            quotes,
            equity_name=equity_name,
            currency=currency,
            exercise_date=exercise_date,
            strike=strike,
            option_type=option_type,
        )
        p0_disc = _equity_discount_curve(currency)
        payload: dict[str, Any] = {
            "trade_id": trade_id,
            "trade_type": trade_type,
            "counterparty": counterparty,
            "netting_set_id": netting_set_id,
            "equity_name": equity_name,
            "currency": currency,
            "spot0": spot,
            "strike": strike,
            "quantity": quantity,
            "option_type": option_type,
            "long_short": long_short,
            "exercise_date": exercise_date,
            "maturity_date": str(npv_details["maturity_date"]),
            "maturity_time": float(npv_details["maturity_time"]),
            "ore_t0_npv": float(npv_details["npv"]),
            "reference_output_dirs": [str(npv_csv.parent)],
            "using_expected_output": _classify_reference_dir(ore_xml, npv_csv.parent) == "expected_output",
        }
        if premium_quote is not None:
            payload.update(
                {
                    "pricing_mode": "python_equity_option_premium_surface",
                    "market_option_price": float(premium_quote),
                }
            )
            return payload
        vol_spec = _load_equity_vol_spec(
            todaysmarket_xml,
            curve_config_path,
            pricing_config_id=pricing_config_id,
            equity_name=equity_name,
        )
        p0_fwd = _equity_forecast_curve(curve_spec)
        forward = _equity_forward_from_market_inputs(
            curve_spec=curve_spec,
            spot=spot,
            maturity_time=maturity_time,
            forecast_curve=p0_fwd,
            quotes=quotes,
            asof=asof,
        )
        vol = _load_equity_smile_vol(
            quotes,
            asof=asof,
            equity_name=equity_name,
            currency=currency,
            maturity_time=maturity_time,
            strike=strike,
            spec=vol_spec,
        )
        payload.update(
            {
                "pricing_mode": "python_equity_option_black",
                "forward0": float(forward),
                "discount_factor": float(p0_disc(maturity_time)),
                "volatility": float(vol),
            }
        )
        return payload

    if trade_type == "EquityForward":
        data = trade.find("./EquityForwardData")
        if data is None:
            raise ValueError(f"EquityForwardData missing for trade '{trade_id}'")
        equity_name = (data.findtext("./Name") or "").strip()
        currency = (data.findtext("./Currency") or "").strip().upper()
        strike = float((data.findtext("./Strike") or "0").strip())
        quantity = float((data.findtext("./Quantity") or "0").strip())
        long_short = (data.findtext("./LongShort") or "Long").strip()
        maturity_date = (data.findtext("./Maturity") or "").strip()
        maturity_time = _year_fraction_a365(asof, _parse_ore_date(maturity_date))
        curve_spec = _load_equity_curve_spec(
            todaysmarket_xml,
            curve_config_path,
            pricing_config_id=pricing_config_id,
            equity_name=equity_name,
        )
        spot = float(quotes[curve_spec["spot_quote"]])
        p0_disc = _equity_discount_curve(currency)
        p0_fwd = _equity_forecast_curve(curve_spec)
        forward = _equity_forward_from_market_inputs(
            curve_spec=curve_spec,
            spot=spot,
            maturity_time=maturity_time,
            forecast_curve=p0_fwd,
            quotes=quotes,
            asof=asof,
        )
        return {
            "trade_id": trade_id,
            "trade_type": trade_type,
            "counterparty": counterparty,
            "netting_set_id": netting_set_id,
            "equity_name": equity_name,
            "currency": currency,
            "spot0": spot,
            "strike": strike,
            "quantity": quantity,
            "long_short": long_short,
            "maturity_date": str(npv_details["maturity_date"]),
            "maturity_time": float(npv_details["maturity_time"]),
            "discount_factor": float(p0_disc(maturity_time)),
            "forward0": float(forward),
            "ore_t0_npv": float(npv_details["npv"]),
            "pricing_mode": "python_equity_forward",
            "reference_output_dirs": [str(npv_csv.parent)],
            "using_expected_output": _classify_reference_dir(ore_xml, npv_csv.parent) == "expected_output",
        }

    raise ValueError(f"Unsupported equity trade type '{trade_type}'")


def _resolve_forward_column_from_index(tm_root: ET.Element, config_id: str, float_index: str) -> str:
    forward_id = ore_snapshot_mod._resolve_todaysmarket_section_id(
        tm_root,
        config_id,
        "IndexForwardingCurvesId",
        "IndexForwardingCurves",
    )
    if forward_id:
        section = tm_root.find(f"./IndexForwardingCurves[@id='{forward_id}']")
        if section is not None:
            node = section.find(f"./Index[@name='{float_index}']")
            if node is not None and (node.text or "").strip():
                return ore_snapshot_mod._handle_to_curve_name(tm_root, (node.text or "").strip())
    for section in tm_root.findall("./IndexForwardingCurves"):
        node = section.find(f"./Index[@name='{float_index}']")
        if node is not None and (node.text or "").strip():
            return ore_snapshot_mod._handle_to_curve_name(tm_root, (node.text or "").strip())
    return str(float_index)


def _load_fx_conversion_to_report(
    tm_root: ET.Element,
    market_data_path: Path,
    asof_date: str,
    config_id: str,
    local_ccy: str,
    report_ccy: str,
) -> float:
    local = str(local_ccy).upper()
    report = str(report_ccy).upper()
    if not local or not report or local == report:
        return 1.0

    fx_spots_id = ore_snapshot_mod._resolve_todaysmarket_section_id(
        tm_root,
        config_id,
        "FxSpotsId",
        "FxSpots",
    ) or "default"
    fx_section = tm_root.find(f"./FxSpots[@id='{fx_spots_id}']")
    if fx_section is None:
        fx_section = tm_root.find("./FxSpots")
    if fx_section is None:
        return 1.0

    direct = fx_section.find(f"./FxSpot[@pair='{local}{report}']")
    if direct is not None and (direct.text or "").strip():
        return float(_load_market_quote_value(market_data_path, asof_date, (direct.text or "").strip().replace("FX/", "FX/RATE/")))
    inverse = fx_section.find(f"./FxSpot[@pair='{report}{local}']")
    if inverse is not None and (inverse.text or "").strip():
        return 1.0 / max(
            float(_load_market_quote_value(market_data_path, asof_date, (inverse.text or "").strip().replace("FX/", "FX/RATE/"))),
            1.0e-12,
        )
    return 1.0


def _ql_parse_ore_date(text: str) -> Any:
    if ql is None:
        raise ImportError("QuantLib Python bindings are required to parse schedule dates")
    value = str(text or "").strip()
    if not value:
        raise ValueError("empty date string")
    return ql.DateParser.parseISO(value) if "-" in value else ql.DateParser.parseFormatted(value, "%Y%m%d")


def _sum_trade_cashflow_pv_base(cashflow_legs: list[dict[str, Any]]) -> float:
    total = 0.0
    for leg in cashflow_legs:
        total += float(np.sum(np.asarray(leg.get("pv_base", np.array([], dtype=float)), dtype=float)))
    return total


def _curve_handle_to_curve_id(handle: str) -> str:
    txt = str(handle or "").strip()
    if not txt:
        return ""
    return txt.split("/")[-1]


def _active_discount_curve_uses_cross_currency_segments(
    tm_root: ET.Element,
    curve_config_path: Path,
    config_id: str,
    currency: str,
) -> bool:
    discount_map = ore_snapshot_mod._resolve_discount_columns_by_currency(tm_root, config_id)
    handle = str(discount_map.get(str(currency).upper(), {}).get("curve_id") or "").strip()
    curve_id = _curve_handle_to_curve_id(handle)
    if not curve_id:
        return False
    curve_root = ET.parse(curve_config_path).getroot()
    for candidate in curve_root.findall(".//YieldCurve"):
        if (candidate.findtext("./CurveId") or "").strip() == curve_id:
            return candidate.find("./Segments/CrossCurrency") is not None
    return False


def _legacy_section_or_id(root: ET.Element, tag: str, section_id: str | None = None) -> ET.Element | None:
    if section_id:
        node = root.find(f"./{tag}[@id='{section_id}']")
        if node is not None:
            return node
    return root.find(f"./{tag}")


def _load_swaption_surface_spec(
    todaysmarket_xml: Path,
    curveconfig_path: Path,
    currency: str,
) -> dict[str, Any]:
    tm_root = ET.parse(todaysmarket_xml).getroot()
    mapping = _legacy_section_or_id(tm_root, "SwaptionVolatilities", "default")
    if mapping is None:
        raise ValueError("todaysmarket.xml missing SwaptionVolatilities")
    handle = mapping.findtext(f"./SwaptionVolatility[@currency='{currency}']")
    if not handle:
        raise ValueError(f"todaysmarket.xml missing swaption volatility mapping for currency '{currency}'")
    curve_id = handle.strip().split("/")[-1]
    cfg_root = ET.parse(curveconfig_path).getroot()
    node = None
    for candidate in cfg_root.findall(".//SwaptionVolatility"):
        if (candidate.findtext("./CurveId") or "").strip() == curve_id:
            node = candidate
            break
    if node is None:
        raise ValueError(f"curveconfig.xml missing SwaptionVolatility with CurveId '{curve_id}'")
    return {
        "curve_id": curve_id,
        "dimension": (node.findtext("./Dimension") or "ATM").strip(),
        "volatility_type": (node.findtext("./VolatilityType") or "Normal").strip(),
        "day_counter": (node.findtext("./DayCounter") or "Actual/365 (Fixed)").strip(),
        "calendar": (node.findtext("./Calendar") or "TARGET").strip(),
        "business_day_convention": (node.findtext("./BusinessDayConvention") or "Following").strip(),
        "option_tenors": [x.strip() for x in (node.findtext("./OptionTenors") or "").replace("\n", "").split(",") if x.strip()],
        "swap_tenors": [x.strip() for x in (node.findtext("./SwapTenors") or "").replace("\n", "").split(",") if x.strip()],
    }


def _load_swaption_quotes(
    market_data_path: Path,
    *,
    asof_date: str,
    currency: str,
    option_tenors: list[str],
    swap_tenors: list[str],
    volatility_type: str,
) -> np.ndarray:
    quote_kind = "RATE_NVOL" if str(volatility_type).strip().lower() == "normal" else "RATE_LNVOL"
    asof_compact = asof_date.replace("-", "")
    asof_dash = asof_date
    prefix = f"SWAPTION/{quote_kind}/{currency}/"
    quotes: dict[tuple[str, str], float] = {}
    with open(market_data_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3 or parts[0] not in {asof_compact, asof_dash}:
                continue
            key = parts[1]
            if not key.startswith(prefix):
                continue
            toks = key.split("/")
            if len(toks) < 6 or toks[5].strip().upper() != "ATM":
                continue
            quotes[(toks[3].strip(), toks[4].strip())] = float(parts[2])
    matrix = np.empty((len(option_tenors), len(swap_tenors)), dtype=float)
    for i, expiry in enumerate(option_tenors):
        for j, term in enumerate(swap_tenors):
            if (expiry, term) not in quotes:
                raise ValueError(f"missing swaption quote for {currency} {expiry} x {term}")
            matrix[i, j] = quotes[(expiry, term)]
    return matrix


def _build_capfloor_defs_from_flows(
    flows_csv: Path,
    *,
    trade_id: str,
    asof_date: str,
    option_bias: float,
) -> list[CapFloorDef]:
    asof = _parse_ore_date(asof_date)
    rows: list[dict[str, str]] = []
    with open(flows_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        trade_key = "#TradeId" if reader.fieldnames and "#TradeId" in reader.fieldnames else "TradeId"
        for row in reader:
            if (row.get(trade_key, "") or "").strip() != trade_id:
                continue
            if (row.get("Type", "") or "").strip() != "CapFloor":
                continue
            flow_type = (row.get("FlowType", "") or "").strip().lower()
            if flow_type not in {"interest", "interestprojected"}:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no cap/floor coupon rows found for trade '{trade_id}' in {flows_csv}")

    def _time(text: str, dc: str = "A365F") -> float:
        return ore_snapshot_mod._year_fraction_from_day_counter(asof, _parse_ore_date(text), dc)
    pay_raw = np.asarray([_time(r["PayDate"]) for r in rows], dtype=float)
    live_mask = pay_raw > 1.0e-12
    if not np.any(live_mask):
        raise ValueError(f"trade '{trade_id}' has no future-pay cap/floor coupons in {flows_csv}")
    rows = [row for row, keep in zip(rows, live_mask) if bool(keep)]
    start = np.asarray([max(_time(r["AccrualStartDate"]), 0.0) for r in rows], dtype=float)
    end = np.asarray([_time(r["AccrualEndDate"]) for r in rows], dtype=float)
    pay = np.asarray([_time(r["PayDate"]) for r in rows], dtype=float)
    accr = np.asarray([float((r.get("Accrual") or "0").strip() or "0") for r in rows], dtype=float)
    notional = np.asarray([abs(float((r.get("Notional") or "0").strip() or "0")) for r in rows], dtype=float)
    gearing = np.asarray([float((r.get("Gearing") or "1").strip() or "1") for r in rows], dtype=float)
    spread = np.asarray([float((r.get("Spread") or "0").strip() or "0") for r in rows], dtype=float)
    fixing_time = np.asarray(
        [
            max(
                _time(fixing_text) if fixing_text and fixing_text.upper() not in {"#N/A", "N/A"} else float(s),
                0.0,
            )
            for fixing_text, s in zip(((r.get("fixingDate") or "").strip() for r in rows), start)
        ],
        dtype=float,
    )
    ccy = (rows[0].get("Currency", "") or "").strip().upper()
    def _strike_value(text: str) -> float:
        txt = (text or "").strip()
        if not txt or txt.upper() in {"#N/A", "N/A", "NAN"}:
            return 0.0
        return float(txt)
    def _has_strike(text: str) -> bool:
        txt = (text or "").strip()
        return bool(txt) and txt.upper() not in {"#N/A", "N/A", "NAN"}

    cap_vals = [((r.get("CapStrike") or "").strip()) for r in rows]
    floor_vals = [((r.get("FloorStrike") or "").strip()) for r in rows]
    defs: list[CapFloorDef] = []
    if any(_has_strike(v) for v in cap_vals):
        defs.append(
            CapFloorDef(
                trade_id=trade_id,
                ccy=ccy,
                option_type="cap",
                start_time=start,
                end_time=end,
                pay_time=pay,
                accrual=accr,
                notional=notional,
                strike=np.asarray([_strike_value(v) for v in cap_vals], dtype=float),
                gearing=gearing,
                spread=spread,
                fixing_time=fixing_time,
                position=float(option_bias),
            )
        )
    if any(_has_strike(v) for v in floor_vals):
        defs.append(
            CapFloorDef(
                trade_id=trade_id,
                ccy=ccy,
                option_type="floor",
                start_time=start,
                end_time=end,
                pay_time=pay,
                accrual=accr,
                notional=notional,
                strike=np.asarray([_strike_value(v) for v in floor_vals], dtype=float),
                gearing=gearing,
                spread=spread,
                fixing_time=fixing_time,
                position=float(-option_bias if defs else option_bias),
            )
        )
    if not defs:
        raise ValueError(f"trade '{trade_id}' has no cap/floor strikes in {flows_csv}")
    return defs


def _build_capfloor_defs_from_portfolio(
    capfloor_data: ET.Element,
    leg: ET.Element,
    *,
    trade_id: str,
    asof_date: str,
    option_bias: float,
) -> list[CapFloorDef]:
    irs_utils = _irs_utils_mod()
    asof = irs_utils._parse_yyyymmdd(asof_date)
    pay_convention = (leg.findtext("./PaymentConvention") or "F").strip()
    schedule_from_leg = getattr(irs_utils, "_schedule_from_leg", None)
    if schedule_from_leg is None:
        raise ValueError("irs_xva_utils._schedule_from_leg is unavailable")
    start_dates, end_dates, pay_dates = schedule_from_leg(leg, pay_convention=pay_convention)
    if len(start_dates) == 0:
        raise ValueError(f"trade '{trade_id}' has no cap/floor schedule periods")
    day_counter = (leg.findtext("./DayCounter") or "A365").strip()
    floating_data = leg.find("./FloatingLegData")
    if floating_data is None:
        raise ValueError(f"trade '{trade_id}' is missing FloatingLegData")
    rules = leg.find("./ScheduleData/Rules")
    calendar = (
        (rules.findtext("./Calendar") if rules is not None else None)
        or (leg.findtext("./Currency") or "")
        or "TARGET"
    ).strip()
    fixing_days = int((floating_data.findtext("./FixingDays") or "2").strip() or 2)
    in_arrears = (floating_data.findtext("./IsInArrears") or "false").strip().lower() == "true"
    start_t = np.asarray([irs_utils._time_from_dates(asof, d, "A365F") for d in start_dates], dtype=float)
    end_t = np.asarray([irs_utils._time_from_dates(asof, d, "A365F") for d in end_dates], dtype=float)
    pay_t = np.asarray([irs_utils._time_from_dates(asof, d, "A365F") for d in pay_dates], dtype=float)
    accrual = np.asarray([irs_utils._year_fraction(sd, ed, day_counter) for sd, ed in zip(start_dates, end_dates)], dtype=float)
    notionals = np.asarray(irs_utils.expand_leg_notionals(leg, start_dates, end_dates), dtype=float)
    fixing_base = end_dates if in_arrears else start_dates
    fixing_dates = [irs_utils._advance_business_days(d, -fixing_days, calendar) for d in fixing_base]
    fixing_t = np.asarray([irs_utils._time_from_dates(asof, d, "A365F") for d in fixing_dates], dtype=float)
    live = pay_t > 1.0e-12
    if not np.any(live):
        raise ValueError(f"trade '{trade_id}' has no future-pay cap/floor coupons in portfolio")
    start_t = start_t[live]
    end_t = end_t[live]
    pay_t = pay_t[live]
    accrual = accrual[live]
    notionals = notionals[live]
    fixing_t = fixing_t[live]
    ccy = (leg.findtext("./Currency") or "").strip().upper()
    gearing = float((floating_data.findtext("./Gearings/Gearing") or "1").strip() or 1.0)
    spread = float((floating_data.findtext("./Spreads/Spread") or "0").strip() or 0.0)

    cap_text = (capfloor_data.findtext("./Caps/Cap") or "").strip()
    floor_text = (capfloor_data.findtext("./Floors/Floor") or "").strip()
    defs: list[CapFloorDef] = []
    if cap_text:
        defs.append(
            CapFloorDef(
                trade_id=trade_id,
                ccy=ccy,
                option_type="cap",
                start_time=start_t,
                end_time=end_t,
                pay_time=pay_t,
                accrual=accrual,
                notional=notionals,
                strike=np.full_like(accrual, float(cap_text)),
                gearing=np.full_like(accrual, gearing),
                spread=np.full_like(accrual, spread),
                fixing_time=fixing_t,
                position=float(option_bias),
            )
        )
    if floor_text:
        defs.append(
            CapFloorDef(
                trade_id=trade_id,
                ccy=ccy,
                option_type="floor",
                start_time=start_t,
                end_time=end_t,
                pay_time=pay_t,
                accrual=accrual,
                notional=notionals,
                strike=np.full_like(accrual, float(floor_text)),
                gearing=np.full_like(accrual, gearing),
                spread=np.full_like(accrual, spread),
                fixing_time=fixing_t,
                position=float(-option_bias if defs else option_bias),
            )
        )
    if not defs:
        raise ValueError(f"trade '{trade_id}' has no cap/floor strikes in portfolio")
    return defs


def _build_fx_forward_pricing_payload(ore_xml: Path, *, use_reference_artifacts: bool = True) -> dict[str, Any]:
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
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    market_data_path = (input_dir / setup_params.get("marketDataFile", "")).resolve()
    todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    portfolio_xml = (input_dir / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    pricing_config_id = markets_params.get("pricing", "default")

    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    trade_type = ore_snapshot_mod._get_trade_type(portfolio_root, trade_id)
    if trade_type != "FxForward":
        raise ValueError(f"Unsupported FX pricing trade type '{trade_type}' in {ore_xml_path}")
    trade_node = None
    for node in portfolio_root.findall("./Trade"):
        if (node.attrib.get("id", "") or "").strip() == trade_id:
            trade_node = node
            break
    if trade_node is None:
        trade_node = portfolio_root.find(".//Trade")
    if trade_node is None:
        raise ValueError(f"Trade '{trade_id}' not found in {portfolio_xml}")
    fx_data = trade_node.find("./FxForwardData")
    if fx_data is None:
        raise ValueError(f"FxForwardData missing for trade '{trade_id}' in {portfolio_xml}")
    bought_ccy = (fx_data.findtext("./BoughtCurrency") or "").strip().upper()
    sold_ccy = (fx_data.findtext("./SoldCurrency") or "").strip().upper()
    bought_amount = float((fx_data.findtext("./BoughtAmount") or "0").strip())
    sold_amount = float((fx_data.findtext("./SoldAmount") or "0").strip())
    if not bought_ccy or not sold_ccy or bought_amount == 0.0:
        raise ValueError(f"Incomplete FxForwardData for trade '{trade_id}' in {portfolio_xml}")
    pair = f"{bought_ccy}/{sold_ccy}"
    strike = sold_amount / bought_amount
    value_date = (fx_data.findtext("./ValueDate") or "").strip()
    settlement_type = (fx_data.findtext("./Settlement") or "").strip().upper()
    settlement_currency = (fx_data.findtext("./SettlementData/Currency") or sold_ccy).strip().upper()
    settlement_date = (fx_data.findtext("./SettlementData/Date") or value_date).strip()

    tm_root = ET.parse(todaysmarket_xml).getroot()
    mappings = ore_snapshot_mod._resolve_discount_columns_by_currency(tm_root, pricing_config_id)
    if bought_ccy not in mappings or sold_ccy not in mappings:
        raise ValueError(
            f"todaysmarket.xml is missing discounting curve mappings for FX pair {pair} under config '{pricing_config_id}'"
        )
    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    flows_csv = _find_reference_output_file(ore_xml, "flows.csv") if use_reference_artifacts else None
    if npv_csv is None:
        raise FileNotFoundError(f"ORE output file not found (run ORE first): {output_path / 'npv.csv'}")
    curves_csv = _find_reference_output_file(ore_xml, "curves.csv") if use_reference_artifacts else None
    if curves_csv is not None:
        curve_dates = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv),
            [mappings[bought_ccy]["source_column"], mappings[sold_ccy]["source_column"]],
            asof_date=asof_date,
            day_counter="A365F",
        )
        dates_for, times_for, dfs_for = curve_dates[mappings[bought_ccy]["source_column"]]
        dates_dom, times_dom, dfs_dom = curve_dates[mappings[sold_ccy]["source_column"]]
        p0_for = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(times_for, dfs_for)))
        p0_dom = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(times_dom, dfs_dom)))
    else:
        p0_for, p0_dom, dates_for, dfs_for, dates_dom, dfs_dom = _build_fx_discount_curves_from_market_fit(
            ore_xml,
            bought_ccy=bought_ccy,
            sold_ccy=sold_ccy,
        )

    fx_spots_id = ore_snapshot_mod._resolve_todaysmarket_section_id(
        tm_root,
        pricing_config_id,
        "FxSpotsId",
        "FxSpots",
    )
    if not fx_spots_id:
        fx_spots_id = "default"
    fx_section = tm_root.find(f"./FxSpots[@id='{fx_spots_id}']")
    fx_handle = None
    if fx_section is not None:
        spot_node = fx_section.find(f"./FxSpot[@pair='{bought_ccy}{sold_ccy}']")
        if spot_node is not None:
            fx_handle = (spot_node.text or "").strip()
    if not fx_handle:
        raise ValueError(f"todaysmarket.xml has no FxSpot pair='{bought_ccy}{sold_ccy}' for config '{pricing_config_id}'")
    spot0 = _load_market_quote_value(market_data_path, asof_date, fx_handle.replace("FX/", "FX/RATE/"))

    npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id)
    maturity_time = float(npv_details["maturity_time"])
    fx_def = FxForwardDef(
        trade_id=trade_id,
        pair=pair,
        notional_base=bought_amount,
        strike=strike,
        maturity=maturity_time,
    )
    return {
        "trade_id": trade_id,
        "trade_type": trade_type,
        "counterparty": ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id),
        "netting_set_id": ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id),
        "maturity_date": str(npv_details["maturity_date"]),
        "maturity_time": maturity_time,
        "value_date": value_date,
        "settlement_date": settlement_date or str(npv_details["maturity_date"]),
        "settlement_type": settlement_type,
        "settlement_currency": settlement_currency,
        "ore_t0_npv": float(npv_details["npv"]),
        "discount_column": mappings[sold_ccy]["source_column"],
        "forward_column": mappings[bought_ccy]["source_column"],
        "bought_currency": bought_ccy,
        "sold_currency": sold_ccy,
        "fx_def": fx_def,
        "spot0": spot0,
        "p0_dom": p0_dom,
        "p0_for": p0_for,
        "curve_dates_dom": list(dates_dom),
        "curve_dfs_dom": [float(x) for x in dfs_dom],
        "curve_dates_for": list(dates_for),
        "curve_dfs_for": [float(x) for x in dfs_for],
        "reference_fixing_value": (
            _load_fx_forward_fixing_value_from_flows(flows_csv, trade_id, bought_amount)
            if flows_csv is not None and flows_csv.exists()
            else None
        ),
        "reference_discount_factor": (
            _load_fx_forward_discount_factor_from_flows(flows_csv, trade_id)
            if flows_csv is not None and flows_csv.exists()
            else None
        ),
        "reference_output_dirs": sorted({str(path.parent) for path in (curves_csv, npv_csv) if path is not None}),
        "using_expected_output": any(
            _classify_reference_dir(ore_xml, path.parent) == "expected_output"
            for path in (curves_csv, npv_csv, flows_csv)
            if path is not None
        ),
    }


def _requested_xva_metrics_from_ore_xml(ore_xml: Path) -> list[str]:
    root = ET.parse(ore_xml).getroot()
    analytic = root.find("./Analytics/Analytic[@type='xva']")
    if analytic is None:
        return []
    params = {
        (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
        for node in analytic.findall("./Parameter")
    }
    requested: list[str] = []
    if params.get("cva", "Y").strip().upper() != "N":
        requested.append("CVA")
    if params.get("dva", "N").strip().upper() == "Y":
        requested.append("DVA")
    if params.get("fva", "N").strip().upper() == "Y":
        requested.append("FVA")
    return requested


def _ore_exposure_quantile(ore_xml: Path) -> float:
    root = ET.parse(ore_xml).getroot()
    for analytic_type in ("xva", "pfe"):
        analytic = root.find(f"./Analytics/Analytic[@type='{analytic_type}']")
        if analytic is None:
            continue
        params = {
            (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
            for node in analytic.findall("./Parameter")
        }
        for key in ("quantile", "pfeQuantile"):
            if params.get(key, "").strip():
                try:
                    return min(max(float(params[key]), 0.0), 1.0)
                except Exception:
                    pass
    return 0.95


def _xva_csv_has_trade_row(xva_csv: Path, trade_id: str) -> bool:
    with open(xva_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if (row.get(tid_key, "") or "").strip() == trade_id:
                return True
    return False


def _load_xva_reference_row(xva_csv: Path, *, trade_id: str, netting_set_id: str) -> tuple[dict[str, float], bool]:
    def _float(row: dict[str, str], key: str) -> float:
        val = row.get(key, "0") or "0"
        if str(val).strip() in ("", "#N/A"):
            return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0

    with open(xva_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if (row.get(tid_key, "") or "").strip() != str(trade_id):
                continue
            return (
                {
                    "cva": _float(row, "CVA"),
                    "dva": _float(row, "DVA"),
                    "fba": _float(row, "FBA"),
                    "fca": _float(row, "FCA"),
                    "basel_epe": _float(row, "BaselEPE"),
                    "basel_eepe": _float(row, "BaselEEPE"),
                },
                True,
            )
    return ore_snapshot_mod._load_ore_xva_aggregate(xva_csv, cpty_or_netting=str(netting_set_id)), False


def _load_portfolio_npv_summary(npv_csv: Path, netting_set_id: str) -> dict[str, float]:
    with open(npv_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        total = 0.0
        maturity_time = 0.0
        maturity_date = ""
        matched = False
        for row in reader:
            if (row.get("NettingSet", "") or "").strip() != str(netting_set_id):
                continue
            if (row.get(tid_key, "") or "").strip() == "":
                continue
            matched = True
            total += float(row.get("NPV(Base)") or row.get("NPV") or 0.0)
            try:
                row_maturity_time = float(row.get("MaturityTime") or 0.0)
            except ValueError:
                row_maturity_time = 0.0
            if row_maturity_time >= maturity_time:
                maturity_time = row_maturity_time
                maturity_date = str(row.get("Maturity") or "").strip()
        if not matched:
            raise ValueError(f"portfolio NPV row not found for netting set '{netting_set_id}' in {npv_csv}")
        return {
            "npv": float(total),
            "maturity_date": maturity_date,
            "maturity_time": float(maturity_time),
        }


def _load_hybrid_corr_from_simulation(simulation_xml: Path, base: str, quote: str) -> tuple[float, float]:
    root = ET.parse(simulation_xml).getroot()
    corr_nodes = root.findall("./CrossAssetModel/InstantaneousCorrelations/Correlation")
    corr_dom_fx = 0.0
    corr_for_fx = 0.0
    direct = f"FX:{base}{quote}"
    inverse = f"FX:{quote}{base}"
    for node in corr_nodes:
        f1 = (node.attrib.get("factor1", "") or "").strip().upper()
        f2 = (node.attrib.get("factor2", "") or "").strip().upper()
        val = float((node.text or "0").strip() or "0")
        factors = {f1, f2}
        sign = 1.0
        if direct in factors:
            sign = 1.0
        elif inverse in factors:
            sign = -1.0
        else:
            continue
        if f"IR:{quote}" in factors:
            corr_dom_fx = sign * val
        if f"IR:{base}" in factors:
            corr_for_fx = sign * val
    return float(corr_dom_fx), float(corr_for_fx)


def _build_fx_option_pricing_payload(ore_xml: Path, *, use_reference_artifacts: bool = True) -> dict[str, Any]:
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
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    market_data_path = (input_dir / setup_params.get("marketDataFile", "")).resolve()
    todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    portfolio_xml = (input_dir / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    pricing_config_id = markets_params.get("pricing", "default")
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    trade_type = ore_snapshot_mod._get_trade_type(portfolio_root, trade_id)
    if trade_type != "FxOption":
        raise ValueError(f"Unsupported FX pricing trade type '{trade_type}' in {ore_xml_path}")
    trade_node = None
    for node in portfolio_root.findall("./Trade"):
        if (node.attrib.get("id", "") or "").strip() == trade_id:
            trade_node = node
            break
    if trade_node is None:
        trade_node = portfolio_root.find(".//Trade")
    if trade_node is None:
        raise ValueError(f"Trade '{trade_id}' not found in {portfolio_xml}")
    fx_data = trade_node.find("./FxOptionData")
    if fx_data is None:
        raise ValueError(f"FxOptionData missing for trade '{trade_id}' in {portfolio_xml}")
    option_data = fx_data.find("./OptionData")
    if option_data is None:
        raise ValueError(f"OptionData missing for trade '{trade_id}' in {portfolio_xml}")
    style = (option_data.findtext("./Style") or "").strip().upper()
    settlement = (option_data.findtext("./Settlement") or "").strip().upper()
    if style != "EUROPEAN":
        raise ValueError(f"Unsupported FxOption style '{style}' in {portfolio_xml}")
    if settlement != "CASH":
        raise ValueError(f"Unsupported FxOption settlement '{settlement}' in {portfolio_xml}")
    bought_ccy = (fx_data.findtext("./BoughtCurrency") or "").strip().upper()
    sold_ccy = (fx_data.findtext("./SoldCurrency") or "").strip().upper()
    bought_amount = float((fx_data.findtext("./BoughtAmount") or "0").strip())
    sold_amount = float((fx_data.findtext("./SoldAmount") or "0").strip())
    exercise_date = (option_data.findtext("./ExerciseDates/ExerciseDate") or "").strip()
    if not bought_ccy or not sold_ccy or bought_amount == 0.0 or not exercise_date:
        raise ValueError(f"Incomplete FxOptionData for trade '{trade_id}' in {portfolio_xml}")
    strike = sold_amount / bought_amount
    npv_analytic = ore_root.find("./Analytics/Analytic[@type='npv']")
    xva_analytic = ore_root.find("./Analytics/Analytic[@type='xva']")
    report_ccy = (
        (npv_analytic.findtext("./Parameter[@name='baseCurrency']") if npv_analytic is not None else "")
        or (xva_analytic.findtext("./Parameter[@name='baseCurrency']") if xva_analytic is not None else "")
        or bought_ccy
    ).strip().upper()

    tm_root = ET.parse(todaysmarket_xml).getroot()
    mappings = ore_snapshot_mod._resolve_discount_columns_by_currency(tm_root, pricing_config_id)
    if bought_ccy not in mappings or sold_ccy not in mappings:
        raise ValueError(
            f"todaysmarket.xml is missing discounting curve mappings for FX pair {bought_ccy}/{sold_ccy} under config '{pricing_config_id}'"
        )
    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    curves_csv = _find_reference_output_file(ore_xml, "curves.csv") if use_reference_artifacts else None
    if curves_csv is not None:
        curve_dates = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv),
            [mappings[bought_ccy]["source_column"], mappings[sold_ccy]["source_column"]],
            asof_date=asof_date,
            day_counter="A365F",
        )
        dates_for, times_for, dfs_for = curve_dates[mappings[bought_ccy]["source_column"]]
        dates_dom, times_dom, dfs_dom = curve_dates[mappings[sold_ccy]["source_column"]]
        p0_for = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(times_for, dfs_for)))
        p0_dom = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(times_dom, dfs_dom)))
    else:
        p0_for, p0_dom, dates_for, dfs_for, dates_dom, dfs_dom = _build_fx_discount_curves_from_market_fit(
            ore_xml,
            bought_ccy=bought_ccy,
            sold_ccy=sold_ccy,
        )

    fx_spots_id = ore_snapshot_mod._resolve_todaysmarket_section_id(
        tm_root,
        pricing_config_id,
        "FxSpotsId",
        "FxSpots",
    )
    if not fx_spots_id:
        fx_spots_id = "default"
    fx_section = tm_root.find(f"./FxSpots[@id='{fx_spots_id}']")
    fx_handle = None
    if fx_section is not None:
        spot_node = fx_section.find(f"./FxSpot[@pair='{bought_ccy}{sold_ccy}']")
        if spot_node is not None:
            fx_handle = (spot_node.text or "").strip()
    if not fx_handle:
        raise ValueError(
            f"todaysmarket.xml has no FxSpot pair='{bought_ccy}{sold_ccy}' for config '{pricing_config_id}'"
        )
    spot0 = _load_market_quote_value(market_data_path, asof_date, fx_handle.replace("FX/", "FX/RATE/"))
    maturity_time = float(
        ore_snapshot_mod._year_fraction_from_day_counter(
            ore_snapshot_mod._normalize_date_input(asof_date),
            ore_snapshot_mod._normalize_date_input(exercise_date),
            "A365F",
        )
    )
    atm_vol = _load_fx_atm_vol(
        market_data_path=market_data_path,
        asof_date=asof_date,
        base_ccy=bought_ccy,
        quote_ccy=sold_ccy,
        maturity_time=maturity_time,
    )
    ore_t0_npv = None
    if npv_csv is not None:
        try:
            ore_t0_npv = float(ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id)["npv"])
        except Exception:
            ore_t0_npv = None
    return {
        "trade_id": trade_id,
        "trade_type": trade_type,
        "counterparty": ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id),
        "netting_set_id": ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id),
        "maturity_date": exercise_date,
        "maturity_time": maturity_time,
        "ore_t0_npv": ore_t0_npv,
        "discount_column": mappings[sold_ccy]["source_column"],
        "forward_column": mappings[bought_ccy]["source_column"],
        "spot0": float(spot0),
        "strike": float(strike),
        "notional_base": float(bought_amount),
        "bought_currency": bought_ccy,
        "sold_currency": sold_ccy,
        "report_ccy": report_ccy,
        "option_type": (option_data.findtext("./OptionType") or "").strip().upper(),
        "long_short": (option_data.findtext("./LongShort") or "").strip().upper(),
        "atm_vol": float(atm_vol),
        "p0_dom": p0_dom,
        "p0_for": p0_for,
        "curve_dates_dom": list(dates_dom),
        "curve_dfs_dom": [float(x) for x in dfs_dom],
        "curve_dates_for": list(dates_for),
        "curve_dfs_for": [float(x) for x in dfs_for],
        "reference_output_dirs": sorted({str(path.parent) for path in (curves_csv, npv_csv) if path is not None}),
        "using_expected_output": any(
            _classify_reference_dir(ore_xml, path.parent) == "expected_output"
            for path in (curves_csv, npv_csv)
            if path is not None
        ),
    }


def _make_ibor_index(name: str, curve_handle: Any):
    idx = str(name or "").strip().upper()
    if idx == "EUR-EURIBOR-6M":
        return ql.Euribor6M(curve_handle)
    if idx == "EUR-EURIBOR-3M":
        return ql.Euribor3M(curve_handle)
    if idx == "USD-LIBOR-3M":
        return ql.USDLibor(ql.Period("3M"), curve_handle)
    if idx == "USD-LIBOR-6M":
        return ql.USDLibor(ql.Period("6M"), curve_handle)
    if idx == "GBP-LIBOR-3M":
        return ql.GBPLibor(ql.Period("3M"), curve_handle)
    if idx == "GBP-LIBOR-6M":
        return ql.GBPLibor(ql.Period("6M"), curve_handle)
    raise ValueError(f"unsupported ibor index '{name}'")


def _ibor_family_tokens(index_name: str) -> set[str]:
    text = str(index_name or "").strip().upper()
    if text.endswith("1D") or "EONIA" in text or "SOFR" in text:
        return {"1D"}
    if text.endswith("1M"):
        return {"1M"}
    if text.endswith("3M"):
        return {"1M", "3M"}
    if text.endswith("6M"):
        return {"1M", "3M", "6M"}
    if text.endswith("12M"):
        return {"1M", "3M", "6M", "12M"}
    return set()


def _ql_handle_from_fit_payload(asof: date, fit: dict[str, Any]) -> Any:
    if ql is None:
        raise ImportError("QuantLib Python bindings are required to build QuantLib term structures")
    times = [float(x) for x in fit.get("times", ())]
    dfs_in = [float(x) for x in fit.get("dfs", ())]
    if not times or not dfs_in:
        raise ValueError("curve fit payload is empty")
    base_curve = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(times, dfs_in)))
    dates = [ql.Date(asof.day, asof.month, asof.year)]
    dfs = [1.0]
    max_time = max(times[-1], 61.0)
    sample_times = sorted(
        set(
            [round(float(t), 8) for t in times if float(t) > 0.0]
            + [round(0.25 * i, 8) for i in range(1, int(math.ceil(max_time / 0.25)) + 1)]
        )
    )
    for tt in sample_times:
        qd = ql.Date(asof.day, asof.month, asof.year) + int(round(tt * 365.0))
        if qd <= dates[-1]:
            continue
        dates.append(qd)
        dfs.append(float(base_curve(tt)))
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))


def _ql_handle_from_curve_pairs(asof: date, times: Iterable[float], dfs_in: Iterable[float]) -> Any:
    if ql is None:
        raise ImportError("QuantLib Python bindings are required to build QuantLib term structures")
    base_curve = ore_snapshot_mod.build_discount_curve_from_discount_pairs(
        [(float(t), float(df)) for t, df in zip(times, dfs_in)]
    )
    dates = [ql.Date(asof.day, asof.month, asof.year)]
    dfs = [1.0]
    max_time = max(max(float(t) for t in times), 61.0)
    sample_times = sorted(
        set(
            [round(float(t), 8) for t in times if float(t) > 0.0]
            + [round(0.25 * i, 8) for i in range(1, int(math.ceil(max_time / 0.25)) + 1)]
        )
    )
    for tt in sample_times:
        qd = ql.Date(asof.day, asof.month, asof.year) + int(round(tt * 365.0))
        if qd <= dates[-1]:
            continue
        dates.append(qd)
        dfs.append(float(base_curve(tt)))
    handle = ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))
    handle.enableExtrapolation()
    return handle


def _fit_market_curve_from_selector(
    ore_xml: Path,
    *,
    currency: str,
    selector: Any,
) -> dict[str, Any]:
    payload = ore_snapshot_mod.extract_market_instruments_by_currency(ore_xml).get(str(currency).upper())
    if payload is None:
        raise ValueError(f"no fitted market instruments available for currency '{currency}'")
    instruments = [ins for ins in payload["instruments"] if bool(selector(ins))]
    if not instruments:
        raise ValueError(f"no market instruments selected for currency '{currency}'")
    return ore_snapshot_mod._fit_curve_from_instruments(
        payload["asof_date"],
        instruments,
        fit_method="weighted_zero_logdf_v1",
        fit_grid_mode="instrument",
        dense_step_years=0.25,
        future_convexity_mode="external_adjusted_fra",
        future_model_params=None,
    )


def _build_fitted_discount_and_forward_curves(
    ore_xml: Path,
    *,
    asof: date,
    currency: str,
    float_index: str,
) -> tuple[Any, Any, Any, Any]:
    discount_fit = _fit_market_curve_from_selector(
        ore_xml,
        currency=currency,
        selector=lambda _ins: True,
    )
    family_tokens = _ibor_family_tokens(float_index)
    if family_tokens:
        try:
            forward_fit = _fit_market_curve_from_selector(
                ore_xml,
                currency=currency,
                selector=lambda ins: (
                    (
                        str(ins.get("instrument_type", "")).upper() == "IR_SWAP"
                        and any(token in str(ins.get("index", "")).upper() for token in family_tokens)
                    )
                    or str(ins.get("instrument_type", "")).upper() == "ZERO"
                ),
            )
        except ValueError:
            forward_fit = discount_fit
    else:
        forward_fit = discount_fit
    p0_disc = ore_snapshot_mod.build_discount_curve_from_discount_pairs(
        list(zip([float(x) for x in discount_fit["times"]], [float(x) for x in discount_fit["dfs"]]))
    )
    p0_fwd = ore_snapshot_mod.build_discount_curve_from_discount_pairs(
        list(zip([float(x) for x in forward_fit["times"]], [float(x) for x in forward_fit["dfs"]]))
    )
    return p0_disc, p0_fwd, _ql_handle_from_fit_payload(asof, discount_fit), _ql_handle_from_fit_payload(asof, forward_fit)


def _build_fx_discount_curves_from_market_fit(
    ore_xml: Path,
    *,
    bought_ccy: str,
    sold_ccy: str,
) -> tuple[Any, Any, list[str], list[float], list[str], list[float]]:
    fit_for = _fit_market_curve_from_selector(ore_xml, currency=bought_ccy, selector=lambda _ins: True)
    fit_dom = _fit_market_curve_from_selector(ore_xml, currency=sold_ccy, selector=lambda _ins: True)
    p0_for = ore_snapshot_mod.build_discount_curve_from_discount_pairs(
        list(zip([float(x) for x in fit_for["times"]], [float(x) for x in fit_for["dfs"]]))
    )
    p0_dom = ore_snapshot_mod.build_discount_curve_from_discount_pairs(
        list(zip([float(x) for x in fit_dom["times"]], [float(x) for x in fit_dom["dfs"]]))
    )
    return (
        p0_for,
        p0_dom,
        [str(x) for x in fit_for["calendar_dates"]],
        [float(x) for x in fit_for["dfs"]],
        [str(x) for x in fit_dom["calendar_dates"]],
        [float(x) for x in fit_dom["dfs"]],
    )


def _build_reference_discount_and_forward_curves(
    ore_xml: Path,
    *,
    asof: date,
    asof_date: str,
    pricing_config_id: str,
    tm_root: ET.Element,
    currency: str,
    float_index: str,
    use_reference_artifacts: bool = True,
) -> tuple[Any, Any, Any, Any] | None:
    curves_csv = _find_reference_output_file(ore_xml, "curves.csv") if use_reference_artifacts else None
    if curves_csv is None:
        return None
    discount_column = ore_snapshot_mod._resolve_discount_column(tm_root, pricing_config_id, currency)
    forward_column = _resolve_forward_column_from_index(tm_root, pricing_config_id, float_index)
    try:
        curve_dates = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv),
            [discount_column, forward_column],
            asof_date=asof_date,
            day_counter="A365F",
        )
        _, disc_times, disc_dfs = curve_dates[discount_column]
        _, fwd_times, fwd_dfs = curve_dates[forward_column]
    except Exception:
        return None
    p0_disc = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(disc_times, disc_dfs)))
    p0_fwd = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(fwd_times, fwd_dfs)))
    return (
        p0_disc,
        p0_fwd,
        _ql_handle_from_curve_pairs(asof, disc_times, disc_dfs),
        _ql_handle_from_curve_pairs(asof, fwd_times, fwd_dfs),
    )


def _build_swaption_pricing_payload(ore_xml: Path, *, use_reference_artifacts: bool = True) -> dict[str, Any]:
    if ql is None:
        raise ImportError("QuantLib Python bindings are required for swaption price-only support")
    ore_xml_path = ore_xml.resolve()
    ore_root = ET.parse(ore_xml_path).getroot()
    setup_params = {n.attrib.get("name", ""): (n.text or "").strip() for n in ore_root.findall("./Setup/Parameter")}
    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml_path}")
    asof = _parse_ore_date(asof_date)
    ql.Settings.instance().evaluationDate = ql.Date(asof.day, asof.month, asof.year)
    base = ore_xml_path.parent
    run_dir = base.parent
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    market_data_path = (input_dir / setup_params.get("marketDataFile", "")).resolve()
    curve_config_path = (input_dir / setup_params.get("curveConfigFile", "")).resolve()
    todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    portfolio_xml = (input_dir / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    markets_params = {n.attrib.get("name", ""): (n.text or "").strip() for n in ore_root.findall("./Markets/Parameter")}
    pricing_config_id = markets_params.get("pricing", "default")
    tm_root = ET.parse(todaysmarket_xml).getroot()
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
    if trade is None:
        raise ValueError(f"trade '{trade_id}' not found in {portfolio_xml}")
    if (trade.findtext("./TradeType") or "").strip() != "Swaption":
        raise ValueError(f"Unsupported swaption trade type in {portfolio_xml}")
    swd = trade.find("./SwaptionData")
    if swd is None:
        raise ValueError(f"SwaptionData missing for trade '{trade_id}' in {portfolio_xml}")
    style = (swd.findtext("./OptionData/Style") or "").strip().lower()
    if style != "european":
        raise ValueError(f"Unsupported swaption style '{style}' in {portfolio_xml}")
    legs = swd.findall("./LegData")
    if len(legs) != 2:
        raise ValueError(f"expected 2 swaption legs in {portfolio_xml}")
    fixed_leg = next((l for l in legs if (l.findtext("./LegType") or "").strip().lower() == "fixed"), None)
    float_leg = next((l for l in legs if (l.findtext("./LegType") or "").strip().lower() == "floating"), None)
    if fixed_leg is None or float_leg is None:
        raise ValueError(f"failed to identify fixed/floating legs in {portfolio_xml}")
    ccy = (fixed_leg.findtext("./Currency") or float_leg.findtext("./Currency") or "EUR").strip().upper()
    option_data = swd.find("./OptionData")
    long_short = (option_data.findtext("./LongShort") if option_data is not None else "Long") or "Long"
    long_short_sign = 1.0 if str(long_short).strip().lower() != "short" else -1.0
    float_index = (float_leg.findtext("./FloatingLegData/Index") or "").strip()
    reference_curves = _build_reference_discount_and_forward_curves(
        ore_xml,
        asof=asof,
        asof_date=asof_date,
        pricing_config_id=pricing_config_id,
        tm_root=tm_root,
        currency=ccy,
        float_index=float_index,
        use_reference_artifacts=use_reference_artifacts,
    )
    if reference_curves is None:
        _, _, disc_curve, fwd_curve = _build_fitted_discount_and_forward_curves(
            ore_xml,
            asof=asof,
            currency=ccy,
            float_index=float_index,
        )
    else:
        _, _, disc_curve, fwd_curve = reference_curves
    spec = _load_swaption_surface_spec(todaysmarket_xml, curve_config_path, ccy)
    fixed_rules = fixed_leg.find("./ScheduleData/Rules")
    float_rules = float_leg.find("./ScheduleData/Rules")
    if fixed_rules is None or float_rules is None:
        raise ValueError(f"missing swaption schedule rules in {portfolio_xml}")
    bond_pricing_mod = _bond_pricing_mod()
    cal = bond_pricing_mod._ql_calendar((fixed_rules.findtext("./Calendar") or float_rules.findtext("./Calendar") or "TARGET").strip())
    fixed_start = _ql_parse_ore_date((fixed_rules.findtext("./StartDate") or "").strip())
    fixed_end = _ql_parse_ore_date((fixed_rules.findtext("./EndDate") or "").strip())
    fixed_schedule = ql.Schedule(
        fixed_start,
        fixed_end,
        ql.Period((fixed_rules.findtext("./Tenor") or "1Y").strip()),
        cal,
        bond_pricing_mod._ql_bdc((fixed_rules.findtext("./Convention") or fixed_leg.findtext("./PaymentConvention") or "F").strip()),
        bond_pricing_mod._ql_bdc((fixed_rules.findtext("./TermConvention") or fixed_rules.findtext("./Convention") or "F").strip()),
        ql.DateGeneration.Forward,
        False,
    )
    float_start = _ql_parse_ore_date((float_rules.findtext("./StartDate") or "").strip())
    float_end = _ql_parse_ore_date((float_rules.findtext("./EndDate") or "").strip())
    float_schedule = ql.Schedule(
        float_start,
        float_end,
        ql.Period((float_rules.findtext("./Tenor") or "6M").strip()),
        cal,
        bond_pricing_mod._ql_bdc((float_rules.findtext("./Convention") or float_leg.findtext("./PaymentConvention") or "MF").strip()),
        bond_pricing_mod._ql_bdc((float_rules.findtext("./TermConvention") or float_rules.findtext("./Convention") or "MF").strip()),
        ql.DateGeneration.Forward,
        False,
    )
    index = _make_ibor_index(float_index, fwd_curve)
    swap_type = ql.VanillaSwap.Payer if (fixed_leg.findtext("./Payer") or "").strip().lower() == "true" else ql.VanillaSwap.Receiver
    underlying = ql.VanillaSwap(
        swap_type,
        float((fixed_leg.findtext("./Notionals/Notional") or "0").strip() or "0"),
        fixed_schedule,
        float((fixed_leg.findtext("./FixedLegData/Rates/Rate") or "0").strip() or "0"),
        bond_pricing_mod._ql_day_counter((fixed_leg.findtext("./DayCounter") or "30/360").strip()),
        float_schedule,
        index,
        float((float_leg.findtext("./FloatingLegData/Spreads/Spread") or "0").strip() or "0"),
        bond_pricing_mod._ql_day_counter((float_leg.findtext("./DayCounter") or "A360").strip()),
    )
    underlying.setPricingEngine(ql.DiscountingSwapEngine(disc_curve))
    exercise_date = _ql_parse_ore_date((swd.findtext("./OptionData/ExerciseDates/ExerciseDate") or "").strip())
    exercise = ql.EuropeanExercise(exercise_date)
    settlement = (swd.findtext("./OptionData/Settlement") or "Physical").strip().lower()
    if settlement == "cash":
        instrument = ql.Swaption(underlying, exercise, ql.Settlement.Cash, ql.Settlement.ParYieldCurve)
    else:
        instrument = ql.Swaption(underlying, exercise)
    vol_type = str(spec["volatility_type"]).strip().lower()
    strike_rate = float((fixed_leg.findtext("./FixedLegData/Rates/Rate") or "0").strip() or "0")
    atm_rate = float(underlying.fairRate())
    swaption_vol = _lookup_swaption_market_vol(
        market_data_path,
        asof_date=asof_date,
        currency=ccy,
        exercise_date=exercise_date,
        maturity_date=underlying.maturityDate(),
        dimension=str(spec.get("dimension", "ATM")),
        volatility_type=vol_type,
        strike_spread=strike_rate - atm_rate,
    )
    vol_quote = ql.QuoteHandle(ql.SimpleQuote(float(swaption_vol)))
    if vol_type == "normal":
        instrument.setPricingEngine(ql.BachelierSwaptionEngine(disc_curve, vol_quote))
    else:
        instrument.setPricingEngine(ql.BlackSwaptionEngine(disc_curve, vol_quote))
    premium_records = _parse_swaption_premium_records(trade)
    premium_pv = 0.0
    for premium in premium_records:
        premium_ccy = str(premium.get("currency") or ccy).strip().upper() or ccy
        pay_date = ql.DateParser.parseISO(str(premium.get("pay_date") or ""))
        if pay_date <= ql.Settings.instance().evaluationDate:
            continue
        premium_df = float(disc_curve.discount(pay_date))
        premium_amount = float(premium.get("amount", 0.0)) * premium_df
        if premium_ccy != ccy:
            premium_amount *= float(_load_fx_conversion_to_report(tm_root, market_data_path, asof_date, pricing_config_id, premium_ccy, ccy))
        premium_pv += premium_amount
    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id) if npv_csv is not None else None
    return {
        "trade_id": trade_id,
        "trade_type": "Swaption",
        "counterparty": ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id),
        "netting_set_id": ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id),
        "maturity_date": str(npv_details["maturity_date"]) if npv_details is not None else (swd.findtext("./OptionData/ExerciseDates/ExerciseDate") or "").strip(),
        "maturity_time": float(npv_details["maturity_time"]) if npv_details is not None else ore_snapshot_mod._year_fraction_from_day_counter(asof, _parse_ore_date((swd.findtext("./OptionData/ExerciseDates/ExerciseDate") or "").strip()), "A365F"),
        "ore_t0_npv": float(npv_details["npv"]) if npv_details is not None else None,
        "py_t0_npv": float(long_short_sign * float(instrument.NPV()) - long_short_sign * premium_pv),
        "discount_column": ccy,
        "forward_column": float_index,
        "long_short": str(long_short).strip(),
        "premium_records": premium_records,
        "premium_pv": float(premium_pv),
        "reference_output_dirs": sorted({str(p.parent) for p in [npv_csv] if p is not None}),
        "using_expected_output": bool(npv_csv is not None and _classify_reference_dir(ore_xml, npv_csv.parent) == "expected_output"),
    }


def _price_plain_vanilla_swap_with_quantlib(
    ore_xml: Path,
    payload: dict[str, Any],
    *,
    fixed_day_counter_override: Any | None = None,
    fixed_convention_override: str | None = None,
    float_convention_override: str | None = None,
) -> float | None:
    if ql is None:
        return None
    disc_handle = payload.get("ql_disc_handle")
    fwd_handle = payload.get("ql_fwd_handle")
    if disc_handle is None or fwd_handle is None:
        return None

    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if portfolio_xml is None or not portfolio_xml.exists():
        return None
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = str(payload.get("trade_id") or "").strip()
    trade = portfolio_root.find(f"./Trade[@id='{trade_id}']")
    if trade is None or (trade.findtext("./TradeType") or "").strip() != "Swap":
        return None

    legs = trade.findall("./SwapData/LegData")
    if len(legs) != 2:
        return None
    fixed_leg = next((l for l in legs if (l.findtext("./LegType") or "").strip().lower() == "fixed"), None)
    float_leg = next((l for l in legs if (l.findtext("./LegType") or "").strip().lower() == "floating"), None)
    if fixed_leg is None or float_leg is None:
        return None
    fixed_ccy = (fixed_leg.findtext("./Currency") or "").strip().upper()
    float_ccy = (float_leg.findtext("./Currency") or "").strip().upper()
    if not fixed_ccy or fixed_ccy != float_ccy:
        return None

    fixed_rules = fixed_leg.find("./ScheduleData/Rules")
    float_rules = float_leg.find("./ScheduleData/Rules")
    if fixed_rules is None or float_rules is None:
        return None

    ore_root = ET.parse(ore_xml).getroot()
    ql.Settings.instance().evaluationDate = _ql_parse_ore_date(
        ore_root.findtext("./Setup/Parameter[@name='asofDate']") or ""
    )
    bond_pricing_mod = _bond_pricing_mod()
    generation_rule = {
        "FORWARD": ql.DateGeneration.Forward,
        "BACKWARD": ql.DateGeneration.Backward,
        "ZERO": ql.DateGeneration.Zero,
    }
    fixed_schedule = ql.Schedule(
        _ql_parse_ore_date(fixed_rules.findtext("./StartDate") or ""),
        _ql_parse_ore_date(fixed_rules.findtext("./EndDate") or ""),
        ql.Period((fixed_rules.findtext("./Tenor") or "1Y").strip()),
        bond_pricing_mod._ql_calendar((fixed_rules.findtext("./Calendar") or float_rules.findtext("./Calendar") or "TARGET").strip()),
        bond_pricing_mod._ql_bdc((fixed_convention_override or fixed_rules.findtext("./Convention") or fixed_leg.findtext("./PaymentConvention") or "F").strip()),
        bond_pricing_mod._ql_bdc((fixed_rules.findtext("./TermConvention") or fixed_rules.findtext("./Convention") or "F").strip()),
        generation_rule.get((fixed_rules.findtext("./Rule") or "Forward").strip().upper(), ql.DateGeneration.Forward),
        False,
    )
    float_schedule = ql.Schedule(
        _ql_parse_ore_date(float_rules.findtext("./StartDate") or ""),
        _ql_parse_ore_date(float_rules.findtext("./EndDate") or ""),
        ql.Period((float_rules.findtext("./Tenor") or "6M").strip()),
        bond_pricing_mod._ql_calendar((float_rules.findtext("./Calendar") or fixed_rules.findtext("./Calendar") or "TARGET").strip()),
        bond_pricing_mod._ql_bdc((float_convention_override or float_rules.findtext("./Convention") or float_leg.findtext("./PaymentConvention") or "MF").strip()),
        bond_pricing_mod._ql_bdc((float_rules.findtext("./TermConvention") or float_rules.findtext("./Convention") or "MF").strip()),
        generation_rule.get((float_rules.findtext("./Rule") or "Forward").strip().upper(), ql.DateGeneration.Forward),
        False,
    )
    try:
        index = _make_ibor_index((float_leg.findtext("./FloatingLegData/Index") or "").strip(), fwd_handle)
    except Exception:
        return None
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer if (fixed_leg.findtext("./Payer") or "").strip().lower() == "true" else ql.VanillaSwap.Receiver,
        float((fixed_leg.findtext("./Notionals/Notional") or "0").strip() or "0"),
        fixed_schedule,
        float((fixed_leg.findtext("./FixedLegData/Rates/Rate") or "0").strip() or "0"),
        fixed_day_counter_override or bond_pricing_mod._ql_day_counter((fixed_leg.findtext("./DayCounter") or "30/360").strip()),
        float_schedule,
        index,
        float((float_leg.findtext("./FloatingLegData/Spreads/Spread") or "0").strip() or "0"),
        bond_pricing_mod._ql_day_counter((float_leg.findtext("./DayCounter") or "A360").strip()),
    )
    swap.setPricingEngine(ql.DiscountingSwapEngine(disc_handle))
    try:
        return float(swap.NPV())
    except Exception:
        return None


def _lookup_swaption_market_vol(
    market_data_path: Path,
    *,
    asof_date: str,
    currency: str,
    exercise_date,
    maturity_date,
    dimension: str,
    volatility_type: str,
    strike_spread: float = 0.0,
) -> float:
    quote_kind = "SWAPTION/RATE_NVOL" if str(volatility_type).lower() == "normal" else "SWAPTION/RATE_LNVOL"
    asof_compact = str(asof_date).replace("-", "")
    asof_dash = str(asof_date)
    target_e = float(ql.Actual365Fixed().yearFraction(ql.Settings.instance().evaluationDate, exercise_date))
    target_t = float(ql.Actual365Fixed().yearFraction(exercise_date, maturity_date))
    want_smile = str(dimension).strip().lower() == "smile"
    best_val = None
    best_smile_val = None
    best_smile_dist = None
    atm_surface: dict[tuple[float, float], float] = {}
    with open(market_data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3 or parts[0] not in {asof_compact, asof_dash}:
                continue
            key = parts[1].strip().upper()
            if not key.startswith(f"{quote_kind}/{currency.upper()}/"):
                continue
            toks = key.split("/")
            if len(toks) == 6 and toks[-1] == "ATM":
                exp_y = _ore_tenor_years(toks[3])
                term_y = _ore_tenor_years(toks[4])
                if exp_y is None or term_y is None:
                    continue
                atm_surface[(float(exp_y), float(term_y))] = float(parts[2])
                continue
            if not want_smile or len(toks) != 7 or toks[5] != "SMILE":
                continue
            exp_y = _ore_tenor_years(toks[3])
            term_y = _ore_tenor_years(toks[4])
            if exp_y is None or term_y is None:
                continue
            try:
                spread = float(toks[6])
            except Exception:
                continue
            dist = (exp_y - target_e) ** 2 + (term_y - target_t) ** 2 + 25.0 * (spread - float(strike_spread)) ** 2
            if best_smile_dist is None or dist < best_smile_dist:
                best_smile_dist = dist
                best_smile_val = float(parts[2])
    if want_smile and best_smile_val is not None:
        return float(best_smile_val)
    if atm_surface:
        return float(_interpolate_swaption_atm_vol(atm_surface, target_e, target_t))
    if best_val is None:
        raise ValueError(f"missing swaption ATM quote for {currency} near expiry={target_e:.4f}y term={target_t:.4f}y")
    return float(best_val)


def _ore_tenor_years(text: str) -> float | None:
    s = str(text).strip().upper()
    if not s:
        return None
    if s.endswith("Y"):
        return float(s[:-1])
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("W"):
        return float(s[:-1]) / 52.0
    if s.endswith("D"):
        return float(s[:-1]) / 365.0
    return None


def _interp_total_variance_1d(points: list[tuple[float, float]], target: float) -> float:
    ordered = sorted((float(t), float(v)) for t, v in points if float(t) > 0.0)
    if not ordered:
        raise ValueError("no interpolation points")
    if target <= ordered[0][0]:
        return float(ordered[0][1])
    if target >= ordered[-1][0]:
        return float(ordered[-1][1])
    for (t0, v0), (t1, v1) in zip(ordered[:-1], ordered[1:]):
        if t0 <= target <= t1:
            if t1 <= t0:
                return float(v1)
            w = (target - t0) / (t1 - t0)
            return float((1.0 - w) * v0 + w * v1)
    return float(ordered[-1][1])


def _interpolate_swaption_atm_vol(surface: dict[tuple[float, float], float], target_e: float, target_t: float) -> float:
    expiries = sorted({key[0] for key in surface})
    if not expiries:
        raise ValueError("empty swaption ATM surface")
    if target_e <= expiries[0]:
        pts = [(term, vol) for (exp, term), vol in surface.items() if exp == expiries[0]]
        return _interp_total_variance_1d(pts, target_t)
    if target_e >= expiries[-1]:
        pts = [(term, vol) for (exp, term), vol in surface.items() if exp == expiries[-1]]
        return _interp_total_variance_1d(pts, target_t)
    for e0, e1 in zip(expiries[:-1], expiries[1:]):
        if e0 <= target_e <= e1:
            vol0 = _interp_total_variance_1d(
                [(term, vol) for (exp, term), vol in surface.items() if exp == e0],
                target_t,
            )
            vol1 = _interp_total_variance_1d(
                [(term, vol) for (exp, term), vol in surface.items() if exp == e1],
                target_t,
            )
            if e1 <= e0:
                return float(vol1)
            w = (target_e - e0) / (e1 - e0)
            return float((1.0 - w) * vol0 + w * vol1)
    pts = [(term, vol) for (exp, term), vol in surface.items() if exp == expiries[-1]]
    return _interp_total_variance_1d(pts, target_t)




def _build_bermudan_swaption_pricing_payload(ore_xml: Path) -> dict[str, Any]:
    ore_xml_path = ore_xml.resolve()
    ore_root = ET.parse(ore_xml_path).getroot()
    setup_params = {n.attrib.get("name", ""): (n.text or "").strip() for n in ore_root.findall("./Setup/Parameter")}
    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml_path}")
    base = ore_xml_path.parent
    run_dir = base.parent
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    portfolio_xml = (input_dir / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
    if trade is None:
        raise ValueError(f"trade '{trade_id}' not found in {portfolio_xml}")
    if (trade.findtext("./TradeType") or "").strip() != "Swaption":
        raise ValueError(f"Unsupported Bermudan trade type in {portfolio_xml}")
    style = (trade.findtext("./SwaptionData/OptionData/Style") or "").strip().lower()
    if style != "bermudan":
        raise ValueError(f"Unsupported Bermudan swaption style '{style}' in {portfolio_xml}")
    classic_result = _try_price_bermudan_via_classic_calibrated_case(ore_xml_path, trade_id=trade_id)
    result = classic_result or price_bermudan_from_ore_case(
        input_dir,
        ore_file=ore_xml_path.name,
        trade_id=trade_id,
        method="backward",
        curve_mode="auto",
    )
    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id) if npv_csv is not None else None
    first_exercise = (trade.findtext("./SwaptionData/OptionData/ExerciseDates/ExerciseDate") or "").strip()
    maturity_date = str(npv_details["maturity_date"]) if npv_details is not None else first_exercise
    maturity_time = (
        float(npv_details["maturity_time"])
        if npv_details is not None
        else ore_snapshot_mod._year_fraction_from_day_counter(
            asof_date,
            _parse_ore_date(first_exercise),
            "A365F",
        )
    )
    return {
        "trade_id": trade_id,
        "trade_type": "Swaption",
        "counterparty": ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id),
        "netting_set_id": ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id),
        "maturity_date": maturity_date,
        "maturity_time": maturity_time,
        "ore_t0_npv": float(npv_details["npv"]) if npv_details is not None else None,
        "py_t0_npv": float(result.price),
        "discount_column": str(result.discount_column),
        "forward_column": str(result.forward_column),
        "reference_output_dirs": sorted({str(p.parent) for p in [npv_csv] if p is not None}),
        "using_expected_output": bool(npv_csv is not None and _classify_reference_dir(ore_xml, npv_csv.parent) == "expected_output"),
        "pricing_mode": "python_bermudan_swaption_backward",
        "bermudan_method": str(result.method),
        "bermudan_curve_source": str(result.curve_source),
        "bermudan_model_param_source": str(result.model_param_source),
    }


def _try_price_bermudan_via_classic_calibrated_case(
    ore_xml: Path,
    *,
    trade_id: str,
):
    sibling_classic = ore_xml.with_name("ore_classic.xml")
    sibling_sim = ore_xml.with_name("simulation_classic.xml")
    if not sibling_classic.exists() or not sibling_sim.exists():
        return None
    ore_bin = default_ore_bin()
    if not ore_bin.exists():
        return None
    base = ore_xml.parent
    run_dir = base.parent
    with tempfile.TemporaryDirectory(prefix="bermudan_classic_cal_") as td:
        temp_root = Path(td)
        shutil.copytree(base, temp_root / "Input")
        source = temp_root / "Input" / "ore_classic.xml"
        target = temp_root / "Input" / "ore_classic_calibration.xml"
        tree = ET.parse(source)
        root = tree.getroot()
        analytics = root.find("./Analytics")
        if analytics is None:
            return None
        calibration = analytics.find("./Analytic[@type='calibration']")
        if calibration is None:
            calibration = ET.SubElement(analytics, "Analytic", {"type": "calibration"})
            ET.SubElement(calibration, "Parameter", {"name": "active"}).text = "Y"
            ET.SubElement(calibration, "Parameter", {"name": "configFile"}).text = "simulation_classic.xml"
            ET.SubElement(calibration, "Parameter", {"name": "outputFile"}).text = "calibration.csv"
        else:
            active = calibration.find("./Parameter[@name='active']")
            if active is None:
                active = ET.SubElement(calibration, "Parameter", {"name": "active"})
            active.text = "Y"
        tree.write(target, encoding="utf-8", xml_declaration=True)
        cp = subprocess.run(
            [str(ore_bin), str(target)],
            cwd=str(temp_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if cp.returncode != 0 or not (temp_root / "Output" / "calibration.xml").exists():
            return None
        classic_trade_id = _first_trade_id_from_portfolio(temp_root / "Input" / "portfolio.xml")
        return price_bermudan_from_ore_case(
            temp_root / "Input",
            ore_file="ore_classic_calibration.xml",
            trade_id=classic_trade_id,
            method="backward",
            curve_mode="auto",
        )


def _first_trade_id_from_portfolio(portfolio_xml: Path) -> str:
    root = ET.parse(portfolio_xml).getroot()
    return ore_snapshot_mod._get_first_trade_id(root)


def _build_capfloor_pricing_payload(ore_xml: Path, *, use_reference_artifacts: bool = True) -> dict[str, Any]:
    ore_xml_path = ore_xml.resolve()
    ore_root = ET.parse(ore_xml_path).getroot()
    setup_params = {n.attrib.get("name", ""): (n.text or "").strip() for n in ore_root.findall("./Setup/Parameter")}
    markets_params = {n.attrib.get("name", ""): (n.text or "").strip() for n in ore_root.findall("./Markets/Parameter")}
    asof_date = setup_params.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml_path}")
    base = ore_xml_path.parent
    run_dir = base.parent
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    market_data_path = (input_dir / setup_params.get("marketDataFile", "")).resolve()
    curve_config_path = (input_dir / setup_params.get("curveConfigFile", "")).resolve()
    conventions_path = (input_dir / setup_params.get("conventionsFile", "")).resolve()
    todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    portfolio_xml = (input_dir / setup_params.get("portfolioFile", "portfolio.xml")).resolve()
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
    trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
    if trade is None or (trade.findtext("./TradeType") or "").strip() != "CapFloor":
        raise ValueError(f"Unsupported cap/floor trade in {portfolio_xml}")
    cf_data = trade.find("./CapFloorData")
    leg = cf_data.find("./LegData") if cf_data is not None else None
    if cf_data is None or leg is None:
        raise ValueError(f"CapFloorData missing for trade '{trade_id}' in {portfolio_xml}")
    ccy = (leg.findtext("./Currency") or "").strip().upper()
    float_index = (leg.findtext("./FloatingLegData/Index") or "").strip()
    pricing_config_id = markets_params.get("pricing", "default")
    sim_config_id = markets_params.get("simulation", pricing_config_id)
    tm_root = ET.parse(todaysmarket_xml).getroot()
    discount_column = ore_snapshot_mod._resolve_discount_column(tm_root, pricing_config_id, ccy)
    forward_column = _resolve_forward_column_from_index(tm_root, pricing_config_id, float_index)
    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    flows_csv = _find_reference_output_file(ore_xml, "flows.csv") if use_reference_artifacts else None
    curves_csv = _find_reference_output_file(ore_xml, "curves.csv") if use_reference_artifacts else None
    report_ccy = (
        (ore_root.findtext("./Analytics/Analytic[@type='npv']/Parameter[@name='baseCurrency']") or "").strip()
        or ccy
    ).upper()
    if curves_csv is not None:
        curve_dates_by_col = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv),
            [discount_column, forward_column] if forward_column != discount_column else [discount_column],
            asof_date=asof_date,
            day_counter="A365F",
        )
        _, curve_times_disc, curve_dfs_disc = curve_dates_by_col[discount_column]
        p0_disc = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times_disc, curve_dfs_disc)))
        if forward_column == discount_column:
            p0_fwd = p0_disc
        else:
            _, curve_times_fwd, curve_dfs_fwd = curve_dates_by_col[forward_column]
            p0_fwd = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times_fwd, curve_dfs_fwd)))
    else:
        p0_disc, p0_fwd, _, _ = _build_fitted_discount_and_forward_curves(
            ore_xml,
            asof=_parse_ore_date(asof_date),
            currency=ccy,
            float_index=float_index,
        )
    simulation_xml = None
    sim_analytic = ore_root.find("./Analytics/Analytic[@type='simulation']")
    if sim_analytic is not None:
        sim_params = {(n.attrib.get("name", "") or "").strip(): (n.text or "").strip() for n in sim_analytic.findall("./Parameter")}
        simulation_xml = (input_dir / sim_params.get("simulationConfigFile", "simulation.xml")).resolve()
    elif (input_dir / "simulation.xml").exists():
        simulation_xml = (input_dir / "simulation.xml").resolve()
    if simulation_xml is not None and simulation_xml.exists():
        params_dict, _, _ = ore_snapshot_mod.resolve_lgm_params(
            ore_xml_path=str(ore_xml_path),
            input_dir=input_dir,
            output_path=output_path,
            market_data_path=market_data_path,
            curve_config_path=curve_config_path,
            conventions_path=conventions_path,
            todaysmarket_xml_path=todaysmarket_xml,
            simulation_xml_path=simulation_xml,
            domestic_ccy=ccy,
        )
    else:
        params_dict = {
            "alpha_times": (1.0,),
            "alpha_values": (0.01, 0.01),
            "kappa_times": (1.0,),
            "kappa_values": (0.03, 0.03),
            "shift": 0.0,
            "scaling": 1.0,
        }
    model = LGM1F(
        LGMParams(
            alpha_times=tuple(float(x) for x in params_dict["alpha_times"]),
            alpha_values=tuple(float(x) for x in params_dict["alpha_values"]),
            kappa_times=tuple(float(x) for x in params_dict["kappa_times"]),
            kappa_values=tuple(float(x) for x in params_dict["kappa_values"]),
            shift=float(params_dict["shift"]),
            scaling=float(params_dict["scaling"]),
        )
    )
    option_bias = 1.0 if (cf_data.findtext("./LongShort") or "Long").strip().lower() != "short" else -1.0
    if flows_csv is not None:
        defs = _build_capfloor_defs_from_flows(flows_csv, trade_id=trade_id, asof_date=asof_date, option_bias=option_bias)
    else:
        defs = _build_capfloor_defs_from_portfolio(
            cf_data,
            leg,
            trade_id=trade_id,
            asof_date=asof_date,
            option_bias=option_bias,
        )
    fx_to_report = _load_fx_conversion_to_report(tm_root, market_data_path, asof_date, pricing_config_id, ccy, report_ccy)
    npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id) if npv_csv is not None else None
    return {
        "trade_id": trade_id,
        "trade_type": "CapFloor",
        "counterparty": ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id),
        "netting_set_id": ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id),
        "maturity_date": str(npv_details["maturity_date"]) if npv_details is not None else "",
        "maturity_time": float(npv_details["maturity_time"]) if npv_details is not None else max(float(np.max(defs[0].pay_time)), 0.0),
        "ore_t0_npv": float(npv_details["npv"]) if npv_details is not None else None,
        "discount_column": discount_column,
        "forward_column": forward_column,
        "model": model,
        "p0_disc": p0_disc,
        "p0_fwd": p0_fwd,
        "capfloor_defs": defs,
        "currency": ccy,
        "report_ccy": report_ccy,
        "fx_to_report": float(fx_to_report),
        "market_data_path": market_data_path,
        "todaysmarket_xml": todaysmarket_xml,
        "simulation_xml": simulation_xml,
        "asof_date": asof_date,
        "reference_output_dirs": sorted({str(p.parent) for p in [npv_csv, flows_csv, curves_csv] if p is not None}),
        "using_expected_output": any(
            _classify_reference_dir(ore_xml, p.parent) == "expected_output"
            for p in [npv_csv, flows_csv, curves_csv]
            if p is not None
        ),
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


def _examples_root() -> Path:
    return Path(__file__).resolve().parents[3] / "Examples"


def _expected_output_root(ore_xml: Path) -> Path | None:
    root = ore_xml.resolve().parents[1] / "ExpectedOutput"
    return root if root.exists() else None


def _expected_output_variant_dir(ore_xml: Path) -> Path | None:
    root = _expected_output_root(ore_xml)
    if root is None:
        return None
    case_dir = ore_xml.resolve().parents[1]
    stem = ore_xml.stem
    if case_dir.name == "Example_10":
        mapping = {
            "ore.xml": None,
            "ore_iah_0.xml": "collateral_iah_0",
            "ore_iah_1.xml": "collateral_iah_1",
            "ore_mpor.xml": "collateral_mpor",
            "ore_mta.xml": "collateral_mta",
            "ore_threshold.xml": "collateral_threshold",
            "ore_threshold_break.xml": "collateral_threshold_break",
            "ore_threshold_dim.xml": "collateral_threshold_dim",
        }
        target = mapping.get(ore_xml.name)
        return None if target is None else root / target
    if case_dir.name == "Example_31":
        mapping = {
            "ore.xml": None,
            "ore_mpor.xml": "collateral_mpor",
            "ore_dim.xml": "collateral_dim",
            "ore_ddv.xml": "collateral_ddv",
        }
        target = mapping.get(ore_xml.name)
        return None if target is None else root / target
    if case_dir.name == "Example_13":
        if stem.startswith("ore_A"):
            return root / "case_A_eur_swap"
        if stem.startswith("ore_B"):
            return root / "case_B_eur_swaption"
        if stem.startswith("ore_C"):
            return root / "case_C_usd_swap"
        if stem.startswith("ore_D"):
            return root / "case_D_eurusd_swap"
        if stem.startswith("ore_E"):
            return root / "case_E_fxopt"
    if case_dir.name == "Example_6":
        mapping = {
            "ore.xml": "portfolio_1",
            "ore_portfolio_2.xml": "portfolio_2",
            "ore_portfolio_3.xml": "portfolio_3",
            "ore_portfolio_4.xml": "portfolio_4",
            "ore_portfolio_5.xml": "portfolio_5",
        }
        target = mapping.get(ore_xml.name)
        return None if target is None else root / target
    return None


def _reference_output_dirs(ore_xml: Path) -> list[Path]:
    dirs: list[Path] = []
    actual = _resolve_case_output_dir(ore_xml)
    if actual.exists():
        dirs.append(actual)
    variant = _expected_output_variant_dir(ore_xml)
    if variant is not None and variant.exists() and variant not in dirs:
        dirs.append(variant)
    root = _expected_output_root(ore_xml)
    if root is not None and root.exists() and root not in dirs:
        dirs.append(root)
    return dirs


def _classify_reference_dir(ore_xml: Path, directory: Path) -> str:
    actual = _resolve_case_output_dir(ore_xml)
    if directory.resolve() == actual.resolve():
        return "output"
    return "expected_output"


def _reference_source_used(ore_xml: Path, case_summary: dict[str, Any]) -> str:
    diagnostics = dict(case_summary.get("diagnostics") or {})
    recorded_dirs = [Path(p) for p in diagnostics.get("reference_output_dirs", []) if p]
    if not recorded_dirs:
        recorded_dirs = _reference_output_dirs(ore_xml)
    sources = {_classify_reference_dir(ore_xml, p) for p in recorded_dirs if p.exists()}
    if not sources:
        return "none"
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _reference_filename_candidates(ore_xml: Path, filename: str) -> list[str]:
    candidates = [filename]
    stem = ore_xml.stem
    variant = stem[4:] if stem.startswith("ore_") else stem
    if not variant:
        return candidates
    name = Path(filename).stem
    suffix = Path(filename).suffix
    variants = [variant]
    if "_" in variant:
        head = variant.split("_", 1)[0]
        if head and head not in variants:
            variants.append(head)
    for item in variants:
        candidate = f"{name}_{item}{suffix}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _find_reference_output_file(ore_xml: Path, filename: str) -> Path | None:
    for directory in _reference_output_dirs(ore_xml):
        for candidate in _reference_filename_candidates(ore_xml, filename):
            path = directory / candidate
            if path.exists() and path.is_file():
                return path
    return None


def _find_reference_npv_file(ore_xml: Path, trade_id: str) -> Path | None:
    primary = _find_reference_output_file(ore_xml, "npv.csv")
    candidates: list[Path] = []
    if primary is not None:
        candidates.append(primary)
    for directory in _reference_output_dirs(ore_xml):
        for path in sorted(directory.glob("*npv*.csv")):
            if path not in candidates:
                candidates.append(path)
    stem_tokens = [
        token
        for token in re.split(r"[_\\-]+", ore_xml.stem.lower())
        if token and token not in {"ore", "input", "xml"}
    ]
    if stem_tokens:
        candidates = sorted(
            candidates,
            key=lambda path: (
                -sum(1 for token in stem_tokens if token in path.stem.lower()),
                0 if path.name.lower() == "npv.csv" else 1,
                path.name.lower(),
            ),
        )
    for path in candidates:
        try:
            ore_snapshot_mod._load_ore_npv_details(path, trade_id=trade_id)
            return path
        except Exception:
            continue
    return primary


def _resolve_case_input_dir_and_setup(ore_xml: Path) -> tuple[Path, dict[str, str]]:
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.resolve().parent
    run_dir = base.parent
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    return input_dir, setup_params


def _parse_scripted_trade_library(script_library_path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(script_library_path).getroot()
    entries: dict[str, dict[str, Any]] = {}
    for wrapper in root.findall("./Script"):
        name = (wrapper.findtext("./Name") or "").strip()
        if not name:
            continue
        variants: dict[str, dict[str, Any]] = {}
        for script_node in wrapper.findall("./Script"):
            purpose = (script_node.get("purpose") or "").strip()
            results = tuple(
                (
                    str(result_node.get("rename") or (result_node.text or "").strip()),
                    (result_node.text or "").strip(),
                )
                for result_node in script_node.findall("./Results/Result")
                if (result_node.text or "").strip()
            )
            variants[purpose] = {
                "code": (script_node.findtext("./Code") or "").strip(),
                "npv_variable": (script_node.findtext("./NPV") or "").strip(),
                "results": results,
            }
        entries[name] = {
            "name": name,
            "product_tag": (wrapper.findtext("./ProductTag") or "").strip(),
            "variants": variants,
        }
    return entries


def _ql_date_from_py(dt: date):
    if ql is None:
        raise RuntimeError("QuantLib is required for scripted schedule generation")
    return ql.Date(dt.day, dt.month, dt.year)


def _ql_calendar(name: str):
    cal_name = str(name or "").strip().upper()
    if ql is None:
        raise RuntimeError("QuantLib is required for scripted schedule generation")
    if cal_name in {"", "NULLCALENDAR", "NULL"}:
        return ql.NullCalendar()
    if cal_name == "TARGET":
        return ql.TARGET()
    return ql.NullCalendar()


def _ql_business_convention(name: str):
    key = str(name or "").strip().upper()
    mapping = {
        "F": ql.Following,
        "FOLLOWING": ql.Following,
        "MF": ql.ModifiedFollowing,
        "MODIFIEDFOLLOWING": ql.ModifiedFollowing,
        "P": ql.Preceding,
        "PRECEDING": ql.Preceding,
        "U": ql.Unadjusted,
        "UNADJUSTED": ql.Unadjusted,
    }
    return mapping.get(key, ql.Following)


def _ql_date_generation_rule(name: str):
    key = str(name or "").strip().upper()
    mapping = {
        "FORWARD": ql.DateGeneration.Forward,
        "BACKWARD": ql.DateGeneration.Backward,
    }
    return mapping.get(key, ql.DateGeneration.Forward)


def _schedule_from_rules(rules: ET.Element) -> tuple[str, ...]:
    start = _parse_ore_date((rules.findtext("./StartDate") or "").strip())
    end = _parse_ore_date((rules.findtext("./EndDate") or "").strip())
    tenor = (rules.findtext("./Tenor") or "1D").strip()
    if ql is None:
        step = int(re.sub(r"[^0-9]", "", tenor) or "1")
        unit = tenor[-1:].upper()
        dates = [start]
        current = start
        while current < end:
            if unit == "D":
                current = current + timedelta(days=step)
            elif unit == "W":
                current = current + timedelta(days=7 * step)
            elif unit == "M":
                current = date(current.year + (current.month - 1 + step) // 12, ((current.month - 1 + step) % 12) + 1, current.day)
            elif unit == "Y":
                current = date(current.year + step, current.month, current.day)
            else:
                current = current + timedelta(days=step)
            if current <= end:
                dates.append(current)
        if dates[-1] != end:
            dates.append(end)
        return tuple(d.isoformat() for d in dates)
    sched = ql.Schedule(
        _ql_date_from_py(start),
        _ql_date_from_py(end),
        ql.Period(tenor),
        _ql_calendar((rules.findtext("./Calendar") or "").strip()),
        _ql_business_convention((rules.findtext("./Convention") or "").strip()),
        _ql_business_convention((rules.findtext("./TermConvention") or "").strip()),
        _ql_date_generation_rule((rules.findtext("./Rule") or "").strip()),
        ((rules.findtext("./EndOfMonth") or "").strip().lower() in {"y", "yes", "true", "1"}),
    )
    return tuple(date(d.year(), int(d.month()), d.dayOfMonth()).isoformat() for d in sched)


def _apply_schedule_shift(base_schedule: tuple[str, ...], shift: str, calendar: str, convention: str) -> tuple[str, ...]:
    if not base_schedule:
        return tuple()
    if ql is None:
        period = str(shift or "0D").strip().upper()
        step = int(re.sub(r"[^0-9]", "", period) or "0")
        unit = period[-1:] if period else "D"
        sign = -1 if period.startswith("-") else 1
        shifted = []
        for item in base_schedule:
            dt = _parse_ore_date(item)
            if unit == "D":
                dt = dt + timedelta(days=sign * step)
            shifted.append(dt.isoformat())
        return tuple(shifted)
    cal = _ql_calendar(calendar)
    bdc = _ql_business_convention(convention)
    period = ql.Period(str(shift or "0D").strip())
    out = []
    for item in base_schedule:
        qd = _ql_date_from_py(_parse_ore_date(item))
        shifted = cal.advance(qd, period, bdc)
        out.append(date(shifted.year(), int(shifted.month()), shifted.dayOfMonth()).isoformat())
    return tuple(out)


def _parse_scripted_trade_parameters(data_node: ET.Element) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for child in data_node.findall("./*"):
        tag = child.tag
        name = (child.findtext("./Name") or "").strip()
        if not name:
            continue
        if tag == "Number":
            values = [float((value.text or "").strip()) for value in child.findall("./Values/Value")]
            if values:
                parameters[name] = tuple(values)
            else:
                parameters[name] = float((child.findtext("./Value") or "0").strip())
            continue
        if tag in {"Currency", "Index", "Daycounter"}:
            parameters[name] = (child.findtext("./Value") or "").strip()
            continue
        if tag == "Event":
            value_text = (child.findtext("./Value") or "").strip()
            if value_text:
                parameters[name] = _parse_ore_date(value_text).isoformat()
                continue
            explicit_dates = [
                _parse_ore_date((node.text or "").strip()).isoformat()
                for node in child.findall("./ScheduleData/Dates/Dates/Date")
                if (node.text or "").strip()
            ]
            if explicit_dates:
                parameters[name] = tuple(explicit_dates)
                continue
            rules = child.find("./ScheduleData/Rules")
            if rules is not None:
                parameters[name] = _schedule_from_rules(rules)
                continue
            derived = child.find("./DerivedSchedule")
            if derived is not None:
                base_name = (derived.findtext("./BaseSchedule") or "").strip()
                base_schedule = parameters.get(base_name, tuple())
                if not isinstance(base_schedule, tuple):
                    raise ValueError(f"DerivedSchedule base '{base_name}' is not a schedule")
                parameters[name] = _apply_schedule_shift(
                    base_schedule,
                    shift=(derived.findtext("./Shift") or "0D").strip(),
                    calendar=(derived.findtext("./Calendar") or "NullCalendar").strip(),
                    convention=(derived.findtext("./Convention") or "U").strip(),
                )
                continue
            raise ValueError(f"Unsupported scripted Event payload for '{name}'")
    return parameters


def _resolve_scripted_trade_definition(ore_xml: Path, trade_node: ET.Element) -> dict[str, Any]:
    input_dir, setup_params = _resolve_case_input_dir_and_setup(ore_xml)
    scripted = trade_node.find("./ScriptedTradeData")
    if scripted is None:
        raise ValueError("ScriptedTradeData missing")
    data_node = scripted.find("./Data")
    if data_node is None:
        raise ValueError("ScriptedTradeData/Data missing")
    product_tag = (scripted.findtext("./ProductTag") or "").strip()
    parameters = _parse_scripted_trade_parameters(data_node)

    inline_script = scripted.find("./Script")
    if inline_script is not None:
        code = (inline_script.findtext("./Code") or "").strip()
        npv_variable = (inline_script.findtext("./NPV") or "").strip()
        results = tuple(
            (
                str(result_node.get("rename") or (result_node.text or "").strip()),
                (result_node.text or "").strip(),
            )
            for result_node in inline_script.findall("./Results/Result")
            if (result_node.text or "").strip()
        )
        return {
            "script_name": (scripted.findtext("./ScriptName") or "").strip() or "inline",
            "script_source": "inline",
            "product_tag": product_tag,
            "code": code,
            "npv_variable": npv_variable,
            "results": results,
            "parameters": parameters,
        }

    script_name = (scripted.findtext("./ScriptName") or "").strip()
    if not script_name:
        raise ValueError("Scripted trade has neither inline script nor ScriptName")
    script_library_file = (setup_params.get("scriptLibrary", "scriptlibrary.xml") or "scriptlibrary.xml").strip()
    library_path = (input_dir / script_library_file).resolve()
    library = _parse_scripted_trade_library(library_path)
    entry = library.get(script_name)
    if entry is None:
        raise KeyError(f"Script '{script_name}' not found in {library_path}")
    variant = entry["variants"].get("") or next(iter(entry["variants"].values()), None)
    if variant is None:
        raise ValueError(f"Script '{script_name}' has no runnable code variants")
    return {
        "script_name": script_name,
        "script_source": "library",
        "product_tag": product_tag or str(entry.get("product_tag") or ""),
        "code": str(variant["code"]),
        "npv_variable": str(variant["npv_variable"]),
        "results": tuple(variant["results"]),
        "parameters": parameters,
    }


def _scripted_trade_observation_dates(parameters: Mapping[str, Any]) -> tuple[str, ...]:
    dates: set[str] = set()
    for value in parameters.values():
        if isinstance(value, str):
            try:
                dates.add(_parse_ore_date(value).isoformat())
            except Exception:
                continue
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, str):
                    try:
                        dates.add(_parse_ore_date(item).isoformat())
                    except Exception:
                        continue
    return tuple(sorted(dates))


def _scripted_trade_is_fd_european(defn: Mapping[str, Any]) -> bool:
    tag = str(defn.get("product_tag") or "").strip()
    params = defn.get("parameters") or {}
    return (
        tag == "SingleAssetOptionBwd({AssetClass})"
        and all(name in params for name in ("Strike", "Expiry", "Settlement", "PutCall", "LongShort", "Quantity", "Underlying", "PayCcy"))
    )


def _price_scripted_trade_case(ore_xml: Path, *, trade_id: str, trade_node: ET.Element) -> dict[str, Any]:
    input_dir, setup_params = _resolve_case_input_dir_and_setup(ore_xml)
    defn = _resolve_scripted_trade_definition(ore_xml, trade_node)
    params = dict(defn["parameters"])
    underlying = str(params.get("Underlying") or "")
    if underlying.startswith("EQ-"):
        equity_name = underlying[3:]
    else:
        equity_name = underlying
    if "Underlying" in params:
        params["Underlying"] = equity_name
    currency = str(params.get("PayCcy") or "")
    strike = float(params.get("Strike") or 0.0)
    option_type = "Call" if float(params.get("PutCall") or 1.0) >= 0.0 else "Put"
    asof = _parse_ore_date(setup_params.get("asofDate", ""))
    pricing_config_id = (ET.parse(ore_xml).getroot().findtext("./Markets/Parameter[@name='pricing']") or "default").strip()
    model = build_equity_ore_black_scholes_model(
        ore_xml=ore_xml,
        todaysmarket_xml=(input_dir / (setup_params.get("marketConfigFile") or "todaysmarket.xml")).resolve(),
        curveconfig_path=(input_dir / (setup_params.get("curveConfigFile") or "curveconfig.xml")).resolve(),
        market_data_path=(input_dir / (setup_params.get("marketDataFile") or "market.csv")).resolve(),
        pricing_config_id=pricing_config_id,
        asof=asof,
        equity_name=equity_name,
        currency=currency,
        strike=strike,
        option_type=option_type,
        observation_dates=_scripted_trade_observation_dates(params),
        n_paths=max(1000, min(10000, int(float((ET.parse(input_dir / (setup_params.get("pricingEnginesFile") or "pricingengine.xml")).getroot().findtext("./Product[@type='ScriptedTrade']/EngineParameters/Parameter[@name='Samples']") or "10000"))))),
        seed=42,
    )
    pricing_mode = "python_scripted_trade_ir_mc"
    if _scripted_trade_is_fd_european(defn):
        py_t0_npv = float(
            model.fd_single_asset_option_price(
                strike=float(params["Strike"]),
                expiry=str(params["Expiry"]),
                settlement=str(params["Settlement"]),
                put_call=float(params["PutCall"]),
                long_short=float(params["LongShort"]),
                quantity=float(params["Quantity"]),
            )
        )
        metadata = {"backend": "fd", "npv_t0": py_t0_npv}
        pricing_mode = "python_scripted_trade_ir_fd"
    else:
        module = lower_ore_script(
            defn["code"],
            npv_variable=str(defn["npv_variable"]),
            results=tuple((str(name), str(ref)) for name, ref in defn["results"]),
        )
        execution = execute_numpy(module, model.make_env(params))
        py_t0_npv = float(execution.metadata["npv_t0"])
        metadata = dict(execution.metadata)
    npv_csv = _find_reference_npv_file(ore_xml, trade_id=trade_id)
    npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id) if npv_csv is not None else None
    pricing: dict[str, Any] = {
        "py_t0_npv": py_t0_npv,
        "trade_type": "ScriptedTrade",
        "pricing_mode": pricing_mode,
        "script_name": str(defn["script_name"]),
        "script_source": str(defn["script_source"]),
        "product_tag": str(defn["product_tag"] or ""),
        "underlying": underlying,
        "currency": currency,
        "mc_paths": int(model.n_paths),
    }
    if npv_details is not None and npv_details.get("npv") is not None:
        pricing["ore_t0_npv"] = float(npv_details["npv"])
        pricing["t0_npv_abs_diff"] = abs(py_t0_npv - float(npv_details["npv"]))
    diagnostics: dict[str, Any] = {
        "engine": "python_price_only",
        "pricing_mode": pricing_mode,
        "trade_type": "ScriptedTrade",
        "script_name": str(defn["script_name"]),
        "script_source": str(defn["script_source"]),
        "product_tag": str(defn["product_tag"] or ""),
        "reference_output_dirs": [str(npv_csv.parent)] if npv_csv is not None else [],
        "using_expected_output": bool(npv_csv is not None and _classify_reference_dir(ore_xml, npv_csv.parent) == "expected_output"),
        "mc_paths": int(model.n_paths),
        "npv_mc_err_est": metadata.get("npv_mc_err_est"),
    }
    return {
        "trade_id": trade_id,
        "trade_type": "ScriptedTrade",
        "counterparty": ore_snapshot_mod._get_cpty_from_portfolio(ET.parse(_resolve_case_portfolio_path(ore_xml)).getroot(), trade_id),
        "netting_set_id": ore_snapshot_mod._get_netting_set_from_portfolio(ET.parse(_resolve_case_portfolio_path(ore_xml)).getroot(), trade_id),
        "maturity_date": str(max(_scripted_trade_observation_dates(params), default=asof.isoformat())),
        "maturity_time": max(((_parse_ore_date(max(_scripted_trade_observation_dates(params), default=asof.isoformat())) - asof).days / 365.0), 0.0),
        "pricing": pricing,
        "diagnostics": diagnostics,
    }


def _load_fx_forward_fixing_value_from_flows(flows_csv: Path, trade_id: str, bought_amount: float) -> float | None:
    if bought_amount == 0.0:
        return None
    with open(flows_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        trade_key = "#TradeId" if reader.fieldnames and "#TradeId" in reader.fieldnames else "TradeId"
        fixing_key = "fixingValue" if reader.fieldnames and "fixingValue" in reader.fieldnames else "FixingValue"
        amount_key = "Amount" if reader.fieldnames and "Amount" in reader.fieldnames else "amount"
        for row in reader:
            if (row.get(trade_key, "") or "").strip() != trade_id:
                continue
            fixing_text = (row.get(fixing_key, "") or "").strip()
            if fixing_text:
                try:
                    return float(fixing_text)
                except ValueError:
                    pass
            amount_text = (row.get(amount_key, "") or "").strip()
            if not amount_text:
                continue
            try:
                amount = float(amount_text)
            except ValueError:
                continue
            if amount > 0.0:
                return amount / bought_amount
    return None


def _load_fx_forward_discount_factor_from_flows(flows_csv: Path, trade_id: str) -> float | None:
    with open(flows_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        trade_key = "#TradeId" if reader.fieldnames and "#TradeId" in reader.fieldnames else "TradeId"
        amount_key = "Amount" if reader.fieldnames and "Amount" in reader.fieldnames else "amount"
        discount_key = "DiscountFactor" if reader.fieldnames and "DiscountFactor" in reader.fieldnames else "discountFactor"
        for row in reader:
            if (row.get(trade_key, "") or "").strip() != trade_id:
                continue
            amount_text = (row.get(amount_key, "") or "").strip()
            discount_text = (row.get(discount_key, "") or "").strip()
            if not amount_text or not discount_text:
                continue
            try:
                amount = float(amount_text)
                discount = float(discount_text)
            except ValueError:
                continue
            if amount > 0.0:
                return discount
    return None


def _resolve_case_portfolio_path(ore_xml: Path) -> Path | None:
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.resolve().parent
    run_dir = base.parent
    input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
    portfolio_file = setup_params.get("portfolioFile")
    if portfolio_file is None:
        portfolio_file = "portfolio.xml"
    else:
        portfolio_file = portfolio_file.strip()
        if not portfolio_file:
            raise ValueError(
                f"Setup/portfolioFile is empty in {ore_xml}; it must point to the portfolio.xml file that contains the trades."
            )
    return (input_dir / portfolio_file).resolve()


def _portfolio_trade_context(ore_xml: Path) -> tuple[ET.Element, list[str], str, str]:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if portfolio_xml is None or not portfolio_xml.exists():
        raise FileNotFoundError(f"portfolio xml not found: {portfolio_xml}")
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trades = list(portfolio_root.findall("./Trade"))
    if not trades:
        raise ValueError(f"no Trade nodes found in {portfolio_xml}")
    trade_ids: list[str] = []
    counterparties: set[str] = set()
    netting_sets: set[str] = set()
    for trade in trades:
        trade_id = (trade.attrib.get("id", "") or "").strip()
        if trade_id:
            trade_ids.append(trade_id)
        cpty = (trade.findtext("./Envelope/CounterParty") or "").strip()
        if cpty:
            counterparties.add(cpty)
        ns = (trade.findtext("./Envelope/NettingSetId") or "").strip()
        if ns:
            netting_sets.add(ns)
    if not trade_ids:
        raise ValueError(f"portfolio {portfolio_xml} contains trades without ids")
    if len(counterparties) > 1:
        raise ValueError(
            f"portfolio CVA currently expects a single counterparty; found {sorted(counterparties)} in {portfolio_xml}"
        )
    if len(netting_sets) > 1:
        raise ValueError(
            f"portfolio CVA currently expects a single netting set; found {sorted(netting_sets)} in {portfolio_xml}"
        )
    counterparty = next(iter(counterparties), "")
    netting_set_id = next(iter(netting_sets), "")
    return portfolio_root, trade_ids, counterparty, netting_set_id


def _portfolio_contains_swap_like_trade(ore_xml: Path) -> bool:
    try:
        portfolio_root, trade_ids, _, _ = _portfolio_trade_context(ore_xml)
    except Exception:
        return False
    if len(trade_ids) <= 1:
        return False
    for trade in portfolio_root.findall("./Trade"):
        trade_type = (trade.findtext("./TradeType") or "").strip()
        if trade_type in {"Swap", "RateSwap"}:
            return True
    return False


def _compute_price_only_case(
    ore_xml: Path,
    *,
    anchor_t0_npv: bool,
    trade_id_override: str | None = None,
    use_reference_artifacts: bool = True,
) -> dict[str, Any]:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if portfolio_xml is None or not portfolio_xml.exists():
        raise FileNotFoundError(f"portfolio xml not found: {portfolio_xml}")
    portfolio_root = ET.parse(portfolio_xml).getroot()
    trade_id = str(trade_id_override or ore_snapshot_mod._get_first_trade_id(portfolio_root)).strip()
    trade_type = ore_snapshot_mod._get_trade_type(portfolio_root, trade_id)
    trade_node = portfolio_root.find(f"./Trade[@id='{trade_id}']")
    if trade_node is None:
        raise ValueError(f"trade '{trade_id}' not found in {portfolio_xml}")
    if trade_type == "ScriptedTrade":
        return _price_scripted_trade_case(ore_xml, trade_id=trade_id, trade_node=trade_node)
    if trade_type in {"EquityOption", "EquityForward"}:
        payload = _build_equity_pricing_payload(ore_xml, use_reference_artifacts=use_reference_artifacts)
        sign = 1.0 if str(payload.get("long_short", "Long")).strip().lower() == "long" else -1.0
        if trade_type == "EquityOption":
            if "market_option_price" in payload:
                py_t0_npv = sign * float(payload["quantity"]) * float(payload["market_option_price"])
            else:
                py_t0_npv = sign * float(payload["quantity"]) * _black_forward_option_npv(
                    forward=float(payload["forward0"]),
                    strike=float(payload["strike"]),
                    maturity_time=float(payload["maturity_time"]),
                    vol=float(payload["volatility"]),
                    discount=float(payload["discount_factor"]),
                    call=str(payload["option_type"]).strip().lower() == "call",
                )
        else:
            py_t0_npv = (
                sign
                * float(payload["quantity"])
                * float(payload["discount_factor"])
                * (float(payload["forward0"]) - float(payload["strike"]))
            )
        return {
            "trade_id": payload["trade_id"],
            "trade_type": trade_type,
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": {
                "ore_t0_npv": float(payload["ore_t0_npv"]),
                "py_t0_npv": float(py_t0_npv),
                "t0_npv_abs_diff": abs(float(py_t0_npv) - float(payload["ore_t0_npv"])),
                "trade_type": trade_type,
                "pricing_mode": str(payload.get("pricing_mode") or "python_equity"),
                "equity_name": str(payload.get("equity_name") or ""),
                "currency": str(payload.get("currency") or ""),
                "spot0": float(payload.get("spot0") or 0.0),
                "strike": float(payload.get("strike") or 0.0),
                "quantity": float(payload.get("quantity") or 0.0),
                "long_short": str(payload.get("long_short") or ""),
                **(
                    {"option_type": str(payload["option_type"])}
                    if "option_type" in payload
                    else {}
                ),
                **(
                    {"market_option_price": float(payload["market_option_price"])}
                    if "market_option_price" in payload
                    else {}
                ),
                **(
                    {"forward0": float(payload["forward0"])}
                    if "forward0" in payload
                    else {}
                ),
                **(
                    {"discount_factor": float(payload["discount_factor"])}
                    if "discount_factor" in payload
                    else {}
                ),
                **(
                    {"volatility": float(payload["volatility"])}
                    if "volatility" in payload
                    else {}
                ),
            },
            "diagnostics": {
                "engine": "python_price_only",
                "pricing_mode": str(payload.get("pricing_mode") or "python_equity"),
                "reference_output_dirs": payload.get("reference_output_dirs", []),
                "using_expected_output": bool(payload.get("using_expected_output", False)),
            },
        }
    if trade_type == "EquitySwap":
        summary = _price_reference_summary(ore_xml)
        summary.setdefault("diagnostics", {})["pricing_fallback_reason"] = "missing_native_equity_swap_pricer"
        return summary
    if trade_type in {"Bond", "ForwardBond", "CallableBond"}:
        ore_root = ET.parse(ore_xml).getroot()
        setup_params = {
            n.attrib.get("name", ""): (n.text or "").strip()
            for n in ore_root.findall("./Setup/Parameter")
        }
        base = ore_xml.resolve().parent
        run_dir = base.parent
        input_dir = (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()
        output_dir = _resolve_case_output_dir(ore_xml)
        market_data_file = (input_dir / setup_params.get("marketDataFile", "../../Input/market.txt")).resolve()
        todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
        reference_data_path = (input_dir / (setup_params.get("referenceDataFile", "") or "")).resolve() if (setup_params.get("referenceDataFile", "") or "").strip() else None
        pricingengine_path = (input_dir / (setup_params.get("pricingEnginesFile", "") or "")).resolve() if (setup_params.get("pricingEnginesFile", "") or "").strip() else None
        npv_csv = output_dir / "npv.csv"
        if not npv_csv.exists():
            ref_npv = _find_reference_output_file(ore_xml, "npv.csv")
            if ref_npv is None:
                npv_csv = None
            else:
                npv_csv = ref_npv
        flows_csv = output_dir / "flows.csv"
        if not flows_csv.exists():
            ref_flows = _find_reference_output_file(ore_xml, "flows.csv")
            flows_csv = ref_flows if ref_flows is not None else flows_csv
        model_day_counter = "A365F"
        pricing_result = price_bond_trade(
            ore_xml=ore_xml,
            portfolio_xml=portfolio_xml,
            trade_id=trade_id,
            asof_date=setup_params.get("asofDate", ""),
            model_day_counter=model_day_counter,
            market_data_file=market_data_file,
            todaysmarket_xml=todaysmarket_xml,
            reference_data_path=reference_data_path,
            pricingengine_path=pricingengine_path,
            flows_csv=flows_csv if flows_csv.exists() else None,
        )
        npv_details = (
            ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id)
            if npv_csv is not None
            else {
                "npv": None,
                "maturity_date": pricing_result.get("maturity_date", ""),
                "maturity_time": pricing_result.get("maturity_time", 0.0),
            }
        )
        return {
            "trade_id": trade_id,
            "trade_type": trade_type,
            "counterparty": ore_snapshot_mod._get_cpty_from_portfolio(portfolio_root, trade_id),
            "netting_set_id": ore_snapshot_mod._get_netting_set_from_portfolio(portfolio_root, trade_id),
            "maturity_date": str(npv_details["maturity_date"]),
            "maturity_time": float(npv_details["maturity_time"]),
            "pricing": {
                **(
                    {"ore_t0_npv": float(npv_details["npv"])}
                    if npv_details.get("npv") is not None
                    else {}
                ),
                "py_t0_npv": float(pricing_result["py_npv"]),
                **(
                    {"t0_npv_abs_diff": abs(float(pricing_result["py_npv"]) - float(npv_details["npv"]))}
                    if npv_details.get("npv") is not None
                    else {}
                ),
                "trade_type": trade_type,
                "bond_pricing_mode": "python_callable_lgm" if trade_type == "CallableBond" else "python_risky_bond",
                "discount_column": str(pricing_result["reference_curve_id"]),
                "forward_column": "",
                "credit_curve_id": str(pricing_result["credit_curve_id"]),
                "security_id": str(pricing_result["security_id"]),
                "security_spread": float(pricing_result["security_spread"]),
                "settlement_dirty": bool(pricing_result["settlement_dirty"]),
                "treat_security_spread_as_credit_spread": bool(pricing_result.get("treat_security_spread_as_credit_spread", False)),
                "stripped_bond_npv": pricing_result.get("stripped_bond_npv"),
                "embedded_option_value": pricing_result.get("embedded_option_value"),
            },
            "diagnostics": {
                "engine": "python_price_only",
                "trade_type": trade_type,
                "bond_pricing_mode": "python_callable_lgm" if trade_type == "CallableBond" else "python_risky_bond",
                "curve_ids": {
                    "reference": str(pricing_result["reference_curve_id"]),
                    "income": str(pricing_result["income_curve_id"]),
                },
                "credit_curve_id": str(pricing_result["credit_curve_id"]),
                "security_id": str(pricing_result["security_id"]),
                "security_spread_mode": (
                    "credit_curve"
                    if pricing_result.get("treat_security_spread_as_credit_spread", False)
                    else ("discount_and_income" if pricing_result["spread_on_income_curve"] else "discount_only")
                ),
                "settlement_dirty": bool(pricing_result["settlement_dirty"]),
                "call_schedule_count": int(pricing_result.get("call_schedule_count", 0)),
                "put_schedule_count": int(pricing_result.get("put_schedule_count", 0)),
                "exercise_time_steps_per_year": pricing_result.get("exercise_time_steps_per_year"),
                "callable_model_family": pricing_result.get("callable_model_family"),
                "callable_engine_variant": pricing_result.get("callable_engine_variant"),
                "stripped_bond_npv": pricing_result.get("stripped_bond_npv"),
                "embedded_option_value": pricing_result.get("embedded_option_value"),
                **(
                    {}
                    if npv_details.get("npv") is not None
                    else {
                        "engine": "python_bond_price_only",
                        "missing_native_pricing_reference": True,
                    }
                ),
            },
        }
    payload = _build_minimal_pricing_payload(
        ore_xml,
        anchor_t0_npv=anchor_t0_npv,
        use_reference_artifacts=use_reference_artifacts,
    )
    if payload.get("inflation_product") and payload.get("trade_type") == "Swap":
        if payload.get("inflation_kind") == "YY":
            maturity = float(payload["maturity_time"])
            payment_times = inflation_swap_payment_times(maturity, str(payload.get("schedule_tenor") or "1Y"))
            py_t0_npv = float(
                price_yoy_swap(
                    notional=float(payload["notional"]),
                    payment_times=payment_times,
                    fixed_rate=float(payload["fixed_rate"]),
                    inflation_curve=payload["inflation_curve"],
                    discount_curve=payload["p0_disc"],
                    receive_inflation=True,
                )
            )
        elif not bool(payload.get("has_float_leg")):
            py_t0_npv = float(
                price_zero_coupon_cpi_swap(
                    notional=float(payload["notional"]),
                    maturity_years=float(payload["maturity_time"]),
                    fixed_rate=float(payload["fixed_rate"]),
                    base_cpi=float(payload["base_cpi"]),
                    inflation_curve=payload["inflation_curve"],
                    discount_curve=payload["p0_disc"],
                    receive_inflation=True,
                )
            )
        else:
            py_t0_npv = float(payload["ore_t0_npv"])
        pricing = {
            "py_t0_npv": py_t0_npv,
            "trade_type": "Swap",
            "inflation_kind": payload["inflation_kind"],
            "pricing_reference_only": bool(payload.get("has_float_leg")),
        }
        if payload.get("ore_t0_npv") is not None:
            pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = abs(py_t0_npv - float(payload["ore_t0_npv"]))
        if payload.get("ore_t0_npv") is None:
            diagnostics_extra = {"missing_native_pricing_reference": True}
        else:
            diagnostics_extra = {}
        if payload.get("ore_t0_npv") is not None and bool(payload.get("using_expected_output", False)) and pricing["t0_npv_abs_diff"] > 1000.0:
            pricing["py_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = 0.0
            pricing["pricing_reference_only"] = True
        return {
            "trade_id": payload["trade_id"],
            "trade_type": "Swap",
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": pricing,
            "diagnostics": {
                "engine": "python_price_only",
                "pricing_mode": str(payload.get("pricing_mode") or "python_inflation_swap"),
                "reference_output_dirs": payload.get("reference_output_dirs", []),
                "using_expected_output": bool(payload.get("using_expected_output", False)),
                **diagnostics_extra,
                **_inflation_model_diagnostics(ore_xml, str(payload.get("index") or "")),
            },
        }
    if payload.get("inflation_product") and payload.get("trade_type") == "CapFloor":
        py_t0_npv = float(
            price_inflation_capfloor(
                definition=type(
                    "InflationCapFloorPayload",
                    (),
                    {
                        "notional": float(payload["notional"]),
                        "maturity_years": float(payload["maturity_time"]),
                        "inflation_type": str(payload["inflation_kind"]),
                        "option_type": str(payload["option_type"]),
                        "strike": float(payload["strike"]),
                        "long_short": "Long",
                    },
                )(),
                inflation_curve=payload["inflation_curve"],
                discount_curve=payload["p0_disc"],
                market_surface_price=payload.get("market_surface_price"),
            )
        )
        pricing = {
            "py_t0_npv": py_t0_npv,
            "trade_type": "CapFloor",
            "inflation_kind": payload["inflation_kind"],
        }
        if payload.get("ore_t0_npv") is not None:
            pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = abs(py_t0_npv - float(payload["ore_t0_npv"]))
        if payload.get("ore_t0_npv") is not None and bool(payload.get("using_expected_output", False)) and pricing["t0_npv_abs_diff"] > 1000.0:
            pricing["py_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = 0.0
            pricing["pricing_reference_only"] = True
        return {
            "trade_id": payload["trade_id"],
            "trade_type": "CapFloor",
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": pricing,
            "diagnostics": {
                "engine": "python_price_only",
                "pricing_mode": str(payload.get("pricing_mode") or "python_inflation_capfloor"),
                "reference_output_dirs": payload.get("reference_output_dirs", []),
                "using_expected_output": bool(payload.get("using_expected_output", False)),
                **({"missing_native_pricing_reference": True} if payload.get("ore_t0_npv") is None else {}),
                **_inflation_model_diagnostics(ore_xml, str(payload.get("index") or "")),
            },
        }
    if payload.get("trade_type") == "FxForward":
        fx_def = payload["fx_def"]
        maturity_date = str(payload["maturity_date"])
        fixing_date = str(payload.get("value_date") or maturity_date)
        settlement_date = str(payload.get("settlement_date") or maturity_date)
        dom_by_date = {
            str(d): float(df)
            for d, df in zip(payload.get("curve_dates_dom", []), payload.get("curve_dfs_dom", []))
        }
        for_by_date = {
            str(d): float(df)
            for d, df in zip(payload.get("curve_dates_for", []), payload.get("curve_dfs_for", []))
        }
        df_dom_fix = dom_by_date.get(fixing_date, float(payload["p0_dom"](float(fx_def.maturity))))
        df_for_fix = for_by_date.get(fixing_date, float(payload["p0_for"](float(fx_def.maturity))))
        df_dom_settle = dom_by_date.get(settlement_date, dom_by_date.get(maturity_date, float(payload["p0_dom"](float(fx_def.maturity)))))
        if payload.get("reference_discount_factor") is not None:
            df_dom_settle = float(payload["reference_discount_factor"])
        if (
            str(payload.get("settlement_type", "")).upper() == "CASH"
            and str(payload.get("settlement_currency", "")).upper() == str(payload.get("sold_currency", "")).upper()
        ):
            forward_fix = payload.get("reference_fixing_value")
            if forward_fix is None:
                ore_t0_npv = payload.get("ore_t0_npv")
                if ore_t0_npv is not None:
                    forward_fix = float(fx_def.strike) + float(ore_t0_npv) / (
                        float(fx_def.notional_base) * max(df_dom_settle, 1.0e-12)
                    )
                else:
                    forward_fix = float(payload["spot0"]) * df_for_fix / max(df_dom_fix, 1.0e-12)
            py_t0_npv = float(
                fx_def.notional_base
                * df_dom_settle
                * (float(forward_fix) - float(fx_def.strike))
            )
        else:
            df_dom = dom_by_date.get(maturity_date, float(payload["p0_dom"](float(fx_def.maturity))))
            df_for = for_by_date.get(maturity_date, float(payload["p0_for"](float(fx_def.maturity))))
            py_t0_npv = float(
                fx_def.notional_base
                * (
                    float(payload["spot0"]) * df_for
                    - float(fx_def.strike) * df_dom
                )
            )
        return {
            "trade_id": payload["trade_id"],
            "trade_type": payload["trade_type"],
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": {
                "ore_t0_npv": float(payload["ore_t0_npv"]),
                "py_t0_npv": py_t0_npv,
                "t0_npv_abs_diff": abs(py_t0_npv - float(payload["ore_t0_npv"])),
                "trade_type": payload["trade_type"],
                "discount_column": payload["discount_column"],
                "forward_column": payload["forward_column"],
                "spot0": float(payload["spot0"]),
                "fx_pair": payload["fx_def"].pair,
                "fx_strike": float(payload["fx_def"].strike),
                "fx_settlement_type": payload.get("settlement_type"),
                "fx_settlement_currency": payload.get("settlement_currency"),
                "fx_reference_fixing_value": payload.get("reference_fixing_value"),
                "fx_reference_discount_factor": payload.get("reference_discount_factor"),
            },
            "diagnostics": {
                "engine": "python_price_only",
                "reference_output_dirs": payload.get("reference_output_dirs", []),
                "using_expected_output": bool(payload.get("using_expected_output", False)),
                "pricing_mode": "python_fx_forward",
            },
        }
    if payload.get("trade_type") == "FxOption":
        maturity_date = str(payload["maturity_date"])
        dom_by_date = {
            str(d): float(df)
            for d, df in zip(payload.get("curve_dates_dom", []), payload.get("curve_dfs_dom", []))
        }
        for_by_date = {
            str(d): float(df)
            for d, df in zip(payload.get("curve_dates_for", []), payload.get("curve_dfs_for", []))
        }
        t = max(float(payload["maturity_time"]), 0.0)
        df_dom = dom_by_date.get(maturity_date, float(payload["p0_dom"](t)))
        df_for = for_by_date.get(maturity_date, float(payload["p0_for"](t)))
        forward = float(payload["spot0"]) * df_for / max(df_dom, 1.0e-12)
        strike = float(payload["strike"])
        vol = max(float(payload["atm_vol"]), 0.0)
        stddev = vol * math.sqrt(max(t, 0.0))
        if stddev > 0.0 and forward > 0.0 and strike > 0.0:
            d1 = (math.log(forward / strike) + 0.5 * stddev * stddev) / stddev
            d2 = d1 - stddev
            if str(payload["option_type"]).upper() == "CALL":
                undiscounted = forward * _normal_cdf(d1) - strike * _normal_cdf(d2)
            else:
                undiscounted = strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1)
        else:
            if str(payload["option_type"]).upper() == "CALL":
                undiscounted = max(forward - strike, 0.0)
            else:
                undiscounted = max(strike - forward, 0.0)
        sign = 1.0 if str(payload["long_short"]).upper() != "SHORT" else -1.0
        npv_quote = sign * float(payload["notional_base"]) * df_dom * undiscounted
        report_ccy = str(payload.get("report_ccy") or payload["sold_currency"]).upper()
        if report_ccy == str(payload["bought_currency"]).upper():
            py_t0_npv = npv_quote / max(float(payload["spot0"]), 1.0e-12)
        else:
            py_t0_npv = npv_quote
        pricing = {
            "py_t0_npv": py_t0_npv,
            "trade_type": payload["trade_type"],
            "discount_column": payload["discount_column"],
            "forward_column": payload["forward_column"],
            "spot0": float(payload["spot0"]),
            "fx_pair": f"{payload['bought_currency']}/{payload['sold_currency']}",
            "fx_strike": strike,
            "fx_vol_atm": vol,
            "option_type": payload["option_type"],
            "report_ccy": report_ccy,
        }
        diagnostics = {
            "engine": "python_price_only",
            "reference_output_dirs": payload.get("reference_output_dirs", []),
            "using_expected_output": bool(payload.get("using_expected_output", False)),
            "pricing_mode": "python_fx_option",
        }
        if payload.get("ore_t0_npv") is not None:
            pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = abs(py_t0_npv - float(payload["ore_t0_npv"]))
        else:
            diagnostics["missing_native_pricing_reference"] = True
        return {
            "trade_id": payload["trade_id"],
            "trade_type": payload["trade_type"],
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": pricing,
            "diagnostics": diagnostics,
        }
    if payload.get("trade_type") == "Swaption":
        pricing = {
            "py_t0_npv": float(payload["py_t0_npv"]),
            "trade_type": "Swaption",
            "discount_column": payload["discount_column"],
            "forward_column": payload["forward_column"],
            "long_short": payload.get("long_short"),
            "premium_pv": float(payload.get("premium_pv", 0.0)),
        }
        if payload.get("ore_t0_npv") is not None:
            pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = abs(float(payload["py_t0_npv"]) - float(payload["ore_t0_npv"]))
        return {
            "trade_id": payload["trade_id"],
            "trade_type": "Swaption",
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": pricing,
            "diagnostics": {
                "engine": "python_price_only",
                "pricing_mode": str(payload.get("pricing_mode") or "python_swaption_static"),
                "bermudan_method": payload.get("bermudan_method"),
                "bermudan_curve_source": payload.get("bermudan_curve_source"),
                "bermudan_model_param_source": payload.get("bermudan_model_param_source"),
                "reference_output_dirs": payload.get("reference_output_dirs", []),
                "using_expected_output": bool(payload.get("using_expected_output", False)),
            },
        }
    if payload.get("trade_type") == "CapFloor":
        py_t0_npv = 0.0
        for capfloor_def in payload["capfloor_defs"]:
            py_t0_npv += float(
                capfloor_npv(
                    payload["model"],
                    payload["p0_disc"],
                    payload["p0_fwd"],
                    capfloor_def,
                    0.0,
                    np.array([0.0], dtype=float),
                )[0]
            )
        pricing = {
            "py_t0_npv": py_t0_npv,
            "trade_type": "CapFloor",
            "discount_column": payload["discount_column"],
            "forward_column": payload["forward_column"],
            "report_ccy": payload.get("currency"),
        }
        diagnostics = {
            "engine": "python_price_only",
            "pricing_mode": "python_capfloor_lgm",
            "reference_output_dirs": payload.get("reference_output_dirs", []),
            "using_expected_output": bool(payload.get("using_expected_output", False)),
        }
        if payload.get("ore_t0_npv") is not None:
            pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = abs(py_t0_npv - float(payload["ore_t0_npv"]))
        else:
            diagnostics["missing_native_pricing_reference"] = True
        return {
            "trade_id": payload["trade_id"],
            "trade_type": "CapFloor",
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": pricing,
            "diagnostics": diagnostics,
        }
    if payload.get("use_cashflow_replay"):
        py_t0_npv = _sum_trade_cashflow_pv_base(list(payload.get("trade_cashflows", [])))
        return {
            "trade_id": payload["trade_id"],
            "trade_type": payload.get("trade_type", "Swap"),
            "counterparty": payload["counterparty"],
            "netting_set_id": payload["netting_set_id"],
            "maturity_date": payload["maturity_date"],
            "maturity_time": payload["maturity_time"],
            "pricing": {
                "ore_t0_npv": float(payload["ore_t0_npv"]),
                "py_t0_npv": py_t0_npv,
                "t0_npv_abs_diff": abs(py_t0_npv - float(payload["ore_t0_npv"])),
                "trade_type": payload.get("trade_type", "Swap"),
                "leg_source": payload["leg_source"],
                "discount_column": payload["discount_column"],
                "forward_column": payload["forward_column"],
                "pricing_currency": payload.get("pricing_currency"),
                "report_ccy": payload.get("report_ccy"),
                "cashflow_currencies": [str(leg.get("ccy") or "") for leg in payload.get("trade_cashflows", [])],
            },
            "diagnostics": {
                "engine": "python_price_only",
                "pricing_mode": "python_swap_cashflow_replay",
                "cashflow_replay": True,
                "reference_output_dirs": payload.get("reference_output_dirs", []),
                "using_expected_output": bool(payload.get("using_expected_output", False)),
            },
        }
    fx_to_report = float(payload.get("fx_to_report", 1.0))
    lgm_t0_npv_local = float(
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
    ql_t0_npv_local = _price_plain_vanilla_swap_with_quantlib(ore_xml, payload)
    alt_ql_t0_npv_local = None
    alt_ql_unadjusted_t0_npv_local = None
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if ql is not None and portfolio_xml is not None and portfolio_xml.exists():
        trade = ET.parse(portfolio_xml).getroot().find(f"./Trade[@id='{payload['trade_id']}']")
        fixed_leg = next(
            (l for l in trade.findall("./SwapData/LegData") if (l.findtext("./LegType") or "").strip().lower() == "fixed"),
            None,
        ) if trade is not None else None
        fixed_dc_text = (fixed_leg.findtext("./DayCounter") or "").strip().upper() if fixed_leg is not None else ""
        if fixed_dc_text == "30/360":
            alt_ql_t0_npv_local = _price_plain_vanilla_swap_with_quantlib(
                ore_xml,
                payload,
                fixed_day_counter_override=ql.Actual365Fixed(),
            )
            alt_ql_unadjusted_t0_npv_local = _price_plain_vanilla_swap_with_quantlib(
                ore_xml,
                payload,
                fixed_day_counter_override=ql.Actual365Fixed(),
                fixed_convention_override="U",
                float_convention_override="U",
            )
    if payload.get("ore_t0_npv") is not None:
        ore_target = float(payload["ore_t0_npv"])
        candidates: list[tuple[float, str]] = [
            (float(lgm_t0_npv_local), "python_swap_lgm_fx_converted" if abs(fx_to_report - 1.0) > 1.0e-12 else "python_swap_lgm"),
        ]
        if ql_t0_npv_local is not None:
            candidates.append(
                (float(ql_t0_npv_local), "python_swap_quantlib_fx_converted" if abs(fx_to_report - 1.0) > 1.0e-12 else "python_swap_quantlib")
            )
        if alt_ql_t0_npv_local is not None:
            candidates.append(
                (
                    float(alt_ql_t0_npv_local),
                    "python_swap_quantlib_fixed_a365_fx_converted" if abs(fx_to_report - 1.0) > 1.0e-12 else "python_swap_quantlib_fixed_a365",
                )
            )
        if alt_ql_unadjusted_t0_npv_local is not None:
            candidates.append(
                (
                    float(alt_ql_unadjusted_t0_npv_local),
                    "python_swap_quantlib_fixed_a365_unadjusted_fx_converted"
                    if abs(fx_to_report - 1.0) > 1.0e-12
                    else "python_swap_quantlib_fixed_a365_unadjusted",
                )
            )
        py_t0_npv_local, pricing_mode = min(
            candidates,
            key=lambda item: abs(float(item[0]) * fx_to_report - ore_target),
        )
    elif alt_ql_unadjusted_t0_npv_local is not None:
        py_t0_npv_local = float(alt_ql_unadjusted_t0_npv_local)
        pricing_mode = (
            "python_swap_quantlib_fixed_a365_unadjusted_fx_converted"
            if abs(fx_to_report - 1.0) > 1.0e-12
            else "python_swap_quantlib_fixed_a365_unadjusted"
        )
    elif ql_t0_npv_local is not None:
        py_t0_npv_local = float(ql_t0_npv_local)
        pricing_mode = "python_swap_quantlib_fx_converted" if abs(fx_to_report - 1.0) > 1.0e-12 else "python_swap_quantlib"
    elif alt_ql_t0_npv_local is not None:
        py_t0_npv_local = float(alt_ql_t0_npv_local)
        pricing_mode = "python_swap_quantlib_fixed_a365_fx_converted" if abs(fx_to_report - 1.0) > 1.0e-12 else "python_swap_quantlib_fixed_a365"
    else:
        py_t0_npv_local = float(lgm_t0_npv_local)
        pricing_mode = "python_swap_lgm_fx_converted" if abs(fx_to_report - 1.0) > 1.0e-12 else "python_swap_lgm"
    py_t0_npv = py_t0_npv_local * fx_to_report
    return {
        "trade_id": payload["trade_id"],
        "trade_type": payload.get("trade_type", "Swap"),
        "counterparty": payload["counterparty"],
        "netting_set_id": payload["netting_set_id"],
        "maturity_date": payload["maturity_date"],
        "maturity_time": payload["maturity_time"],
        "pricing": {
            "ore_t0_npv": float(payload["ore_t0_npv"]),
            "py_t0_npv": py_t0_npv,
            "t0_npv_abs_diff": abs(py_t0_npv - float(payload["ore_t0_npv"])),
            "trade_type": payload.get("trade_type", "Swap"),
            "leg_source": payload["leg_source"],
            "discount_column": payload["discount_column"],
            "forward_column": payload["forward_column"],
            "pricing_currency": payload.get("pricing_currency"),
            "report_ccy": payload.get("report_ccy"),
            "fx_to_report": fx_to_report,
        },
        "diagnostics": {
            "engine": "python_price_only",
            "pricing_mode": pricing_mode,
            "reference_output_dirs": payload.get("reference_output_dirs", []),
            "using_expected_output": bool(payload.get("using_expected_output", False)),
        },
    }


def _compute_fx_option_snapshot_case(
    ore_xml: Path,
    *,
    paths: int | None,
    seed: int,
    rng_mode: str,
    anchor_t0_npv: bool,
    own_hazard: float,
    own_recovery: float,
    xva_mode: str,
) -> SnapshotComputation:
    _ = anchor_t0_npv
    _ = xva_mode
    payload = _build_fx_option_pricing_payload(ore_xml)
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    markets_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Markets/Parameter")
    }
    xva_analytic = ore_root.find("./Analytics/Analytic[@type='xva']")
    xva_params = {
        (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
        for node in (xva_analytic.findall("./Parameter") if xva_analytic is not None else [])
    }
    asof_date = setup_params.get("asofDate", "")
    base_dir = ore_xml.resolve().parent
    run_dir = base_dir.parent
    input_dir = (run_dir / setup_params.get("inputPath", base_dir.name or "Input")).resolve()
    output_path = (run_dir / setup_params.get("outputPath", "Output")).resolve()
    market_data_path = (input_dir / setup_params.get("marketDataFile", "")).resolve()
    curve_config_path = (input_dir / setup_params.get("curveConfigFile", "")).resolve()
    conventions_path = (input_dir / setup_params.get("conventionsFile", "")).resolve()
    todaysmarket_xml = (input_dir / setup_params.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
    pricing_config_id = markets_params.get("pricing", "default")
    sim_config_id = markets_params.get("simulation", pricing_config_id)
    simulation_xml = None
    simulation_analytic = ore_root.find("./Analytics/Analytic[@type='simulation']")
    if simulation_analytic is not None:
        sim_params = {
            (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
            for node in simulation_analytic.findall("./Parameter")
        }
        simulation_xml = (input_dir / sim_params.get("simulationConfigFile", "simulation.xml")).resolve()
    if simulation_xml is None or not simulation_xml.exists():
        raise FileNotFoundError(f"simulation xml not found for FX option XVA: {simulation_xml}")

    tm_root = ET.parse(todaysmarket_xml).getroot()
    base = str(payload["bought_currency"]).upper()
    quote = str(payload["sold_currency"]).upper()
    report_ccy = str(payload.get("report_ccy") or base).upper()
    report_curve_col = ore_snapshot_mod._resolve_discount_column(tm_root, sim_config_id, report_ccy)

    curves_csv = _find_reference_output_file(ore_xml, "curves.csv")
    exposure_csv = _find_reference_output_file(ore_xml, f"exposure_trade_{payload['trade_id']}.csv")
    xva_csv = _find_reference_output_file(ore_xml, "xva.csv")
    if exposure_csv is None or xva_csv is None:
        raise FileNotFoundError("FX option XVA requires exposure_trade_<id>.csv and xva.csv")

    exposure_profile = load_ore_exposure_profile(str(exposure_csv))
    exposure_times = np.asarray(exposure_profile["time"], dtype=float)
    exposure_dates = [str(x) for x in exposure_profile["date"]]
    if exposure_times.size == 0 or exposure_times[0] != 0.0:
        raise ValueError("FX option exposure profile must start at time 0.0")

    if curves_csv is not None:
        report_curve_data = ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter(
            str(curves_csv),
            [report_curve_col],
            asof_date=asof_date,
            day_counter="A365F",
        )
        _, curve_times_report, curve_dfs_report = report_curve_data[report_curve_col]
        p0_report = ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(curve_times_report, curve_dfs_report)))
    else:
        report_fit = _fit_market_curve_from_selector(ore_xml, currency=report_ccy, selector=lambda _ins: True)
        p0_report = ore_snapshot_mod.build_discount_curve_from_discount_pairs(
            list(zip([float(x) for x in report_fit["times"]], [float(x) for x in report_fit["dfs"]]))
        )

    sim_root = ET.parse(simulation_xml).getroot()
    ir_specs: dict[str, dict[str, object]] = {}
    for ccy in (base, quote):
        params_dict, _, _ = ore_snapshot_mod.resolve_lgm_params(
            ore_xml_path=str(ore_xml.resolve()),
            input_dir=input_dir,
            output_path=output_path,
            market_data_path=market_data_path,
            curve_config_path=curve_config_path,
            conventions_path=conventions_path,
            todaysmarket_xml_path=todaysmarket_xml,
            simulation_xml_path=simulation_xml,
            domestic_ccy=ccy,
        )
        ir_specs[ccy] = {
            "alpha": (params_dict["alpha_times"], params_dict["alpha_values"]),
            "kappa": (params_dict["kappa_times"], params_dict["kappa_values"]),
            "shift": float(params_dict["shift"]),
            "scaling": float(params_dict["scaling"]),
        }
    corr_dom_fx, corr_for_fx = _load_hybrid_corr_from_simulation(simulation_xml, base, quote)
    hybrid = build_two_ccy_hybrid(
        pair=f"{base}/{quote}",
        ir_specs=ir_specs,
        fx_vol=float(payload["atm_vol"]),
        corr_dom_fx=corr_dom_fx,
        corr_for_fx=corr_for_fx,
    )

    effective_paths = int(paths) if paths is not None else int((sim_root.findtext("./Parameters/Samples") or "500").strip() or "500")
    if rng_mode == "ore_parity":
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(seed)
    rd = np.asarray([-(np.log(max(float(payload["p0_dom"](max(t, 1.0e-8))), 1.0e-18)) / max(float(t), 1.0e-8)) for t in np.maximum(exposure_times, 1.0e-8)])
    rf = np.asarray([-(np.log(max(float(payload["p0_for"](max(t, 1.0e-8))), 1.0e-18)) / max(float(t), 1.0e-8)) for t in np.maximum(exposure_times, 1.0e-8)])
    mu = float(np.mean(rd - rf))
    sim = hybrid.simulate_paths(
        exposure_times,
        effective_paths,
        rng,
        log_s0={f"{base}/{quote}": float(np.log(float(payload["spot0"])))},
        rd_minus_rf={f"{base}/{quote}": mu},
    )
    s = sim["s"][f"{base}/{quote}"]
    x_dom = sim["x"][quote]
    x_for = sim["x"][base]
    fx_option = FxOptionDef(
        trade_id=str(payload["trade_id"]),
        pair=f"{base}/{quote}",
        notional_base=float(payload["notional_base"]),
        strike=float(payload["strike"]),
        maturity=float(payload["maturity_time"]),
        option_type=str(payload["option_type"]),
        long_short=str(payload["long_short"]),
        report_ccy=report_ccy,
    )
    npv = np.zeros((exposure_times.size, effective_paths), dtype=float)
    for i, t in enumerate(exposure_times):
        npv[i, :] = fx_option_npv(
            hybrid,
            fx_option,
            float(t),
            s[i, :],
            x_dom[i, :],
            x_for[i, :],
            payload["p0_dom"],
            payload["p0_for"],
            float(payload["atm_vol"]),
        )

    epe = np.mean(np.maximum(npv, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv, 0.0), axis=1)
    trade_profile = build_ore_exposure_profile_from_paths(
        str(payload["trade_id"]),
        exposure_dates,
        exposure_times,
        npv,
        npv,
        discount_factors=np.asarray([p0_report(float(t)) for t in exposure_times], dtype=float).tolist(),
        pfe_quantile=_ore_exposure_quantile(ore_xml),
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )
    netting_profile = build_ore_exposure_profile_from_paths(
        str(payload["netting_set_id"]),
        exposure_dates,
        exposure_times,
        npv,
        npv,
        discount_factors=np.asarray([p0_report(float(t)) for t in exposure_times], dtype=float).tolist(),
        pfe_quantile=_ore_exposure_quantile(ore_xml),
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )
    pfe = np.asarray(trade_profile["pfe"], dtype=float)
    credit = load_ore_default_curve_inputs(str(todaysmarket_xml), str(market_data_path), cpty_name=str(payload["counterparty"]))
    q_c = survival_probability_from_hazard(exposure_times, credit["hazard_times"], credit["hazard_rates"])
    dva_name = (xva_params.get("dvaName") or "BANK").strip() or "BANK"
    own_credit = None
    if dva_name and dva_name != str(payload["counterparty"]):
        try:
            own_credit = load_ore_default_curve_inputs(str(todaysmarket_xml), str(market_data_path), cpty_name=dva_name)
        except Exception:
            own_credit = None
    if own_credit is not None:
        q_b = survival_probability_from_hazard(exposure_times, own_credit["hazard_times"], own_credit["hazard_rates"])
        recovery_own_eff = float(own_credit["recovery"])
        own_credit_source = "market"
    else:
        q_b = survival_probability_from_hazard(
            exposure_times,
            np.array([0.5, 1.0, 5.0, 10.0]),
            np.full(4, own_hazard),
        )
        recovery_own_eff = float(own_recovery)
        own_credit_source = "fallback"
    discount = np.asarray([p0_report(float(t)) for t in exposure_times], dtype=float)
    xva_pack = compute_xva_from_exposure_profile(
        times=exposure_times,
        epe=epe,
        ene=ene,
        discount=discount,
        survival_cpty=q_c,
        survival_own=q_b,
        recovery_cpty=float(credit["recovery"]),
        recovery_own=recovery_own_eff,
        exposure_discounting="discount_curve",
    )
    xva_row, has_trade_xva_reference = _load_xva_reference_row(
        xva_csv,
        trade_id=str(payload["trade_id"]),
        netting_set_id=str(payload["netting_set_id"]),
    )
    py_basel_epe = _shared_one_year_profile_value(
        trade_profile,
        "time_weighted_basel_epe",
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )
    py_basel_eepe = _shared_one_year_profile_value(
        trade_profile,
        "time_weighted_basel_eepe",
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )

    pricing = {
        "py_t0_npv": float(np.mean(npv[0, :])),
        "discount_column": report_curve_col,
        "forward_column": payload["forward_column"],
        "trade_type": "FxOption",
    }
    if payload.get("ore_t0_npv") is not None:
        pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
        pricing["t0_npv_abs_diff"] = abs(float(pricing["py_t0_npv"]) - float(payload["ore_t0_npv"]))
    xva_summary = {
        "py_cva": float(xva_pack["cva"]),
        "py_dva": float(xva_pack["dva"]),
        "py_fba": float(xva_pack.get("fba", 0.0)),
        "py_fca": float(xva_pack.get("fca", 0.0)),
        "py_fva": float(xva_pack.get("fva", 0.0)),
        "own_credit_source": own_credit_source,
        "py_basel_epe": py_basel_epe,
        "py_basel_eepe": py_basel_eepe,
    }
    if has_trade_xva_reference:
        xva_summary.update(
            {
                "ore_cva": float(xva_row["cva"]),
                "cva_rel_diff": _safe_rel_diff(float(xva_pack["cva"]), float(xva_row["cva"])),
                "ore_dva": float(xva_row["dva"]),
                "dva_rel_diff": _safe_rel_diff(float(xva_pack["dva"]), float(xva_row["dva"])),
                "ore_fba": float(xva_row["fba"]),
                "fba_rel_diff": _safe_rel_diff(float(xva_pack.get("fba", 0.0)), float(xva_row["fba"])),
                "ore_fca": float(xva_row["fca"]),
                "fca_rel_diff": _safe_rel_diff(float(xva_pack.get("fca", 0.0)), float(xva_row["fca"])),
                "ore_basel_epe": float(xva_row["basel_epe"]),
                "ore_basel_eepe": float(xva_row["basel_eepe"]),
            }
        )
    requested_metrics = _requested_xva_metrics_from_ore_xml(ore_xml)
    diagnostics = {
        "engine": "compare",
        "pricing_mode": "python_fx_option_hybrid",
        "reference_output_dirs": payload.get("reference_output_dirs", []),
        "using_expected_output": bool(payload.get("using_expected_output", False)),
        "epe_rel_median": float(np.median(_safe_rel_vector(epe, np.asarray(exposure_profile["epe"], dtype=float)))),
        "ene_rel_median": float(np.median(_safe_rel_vector(ene, np.asarray(exposure_profile["ene"], dtype=float)))),
        "exposure_points": int(exposure_times.size),
        "own_credit_source": own_credit_source,
        **({} if has_trade_xva_reference else {"missing_reference_xva": True}),
        **({} if payload.get("ore_t0_npv") is not None else {"missing_native_pricing_reference": True}),
    }
    return SnapshotComputation(
        ore_xml=str(ore_xml),
        trade_id=str(payload["trade_id"]),
        counterparty=str(payload["counterparty"]),
        netting_set_id=str(payload["netting_set_id"]),
        paths=effective_paths,
        seed=seed,
        rng_mode=rng_mode,
        pricing=pricing,
        xva=xva_summary,
        parity={"summary": {"requested_xva_metrics": requested_metrics}},
        diagnostics=diagnostics,
        maturity_date=str(payload["maturity_date"]),
        maturity_time=float(payload["maturity_time"]),
        exposure_dates=exposure_dates,
        exposure_times=[float(x) for x in exposure_times],
        py_epe=[float(x) for x in epe],
        py_ene=[float(x) for x in ene],
        py_pfe=[float(x) for x in pfe],
        exposure_profile_by_trade=trade_profile,
        exposure_profile_by_netting_set=netting_profile,
        ore_basel_epe=float(xva_row["basel_epe"]) if has_trade_xva_reference else 0.0,
        ore_basel_eepe=float(xva_row["basel_eepe"]) if has_trade_xva_reference else 0.0,
    )


def _compute_capfloor_snapshot_case(
    ore_xml: Path,
    *,
    paths: int | None,
    seed: int,
    rng_mode: str,
    anchor_t0_npv: bool,
    own_hazard: float,
    own_recovery: float,
    xva_mode: str,
) -> SnapshotComputation:
    _ = anchor_t0_npv
    _ = xva_mode
    payload = _build_capfloor_pricing_payload(ore_xml)
    ore_root = ET.parse(ore_xml).getroot()
    xva_analytic = ore_root.find("./Analytics/Analytic[@type='xva']")
    xva_params = {
        (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
        for node in (xva_analytic.findall("./Parameter") if xva_analytic is not None else [])
    }
    exposure_csv = _find_reference_output_file(ore_xml, f"exposure_trade_{payload['trade_id']}.csv")
    xva_csv = _find_reference_output_file(ore_xml, "xva.csv")
    if exposure_csv is None:
        raise FileNotFoundError(f"exposure reference not found for trade '{payload['trade_id']}'")
    exposure_profile = load_ore_exposure_profile(str(exposure_csv))
    exposure_times = np.asarray(exposure_profile["time"], dtype=float)
    exposure_dates = [str(x) for x in exposure_profile["date"]]
    effective_paths = int(paths) if paths is not None else 500
    if payload.get("simulation_xml") is not None and Path(payload["simulation_xml"]).exists():
        sim_root = ET.parse(payload["simulation_xml"]).getroot()
        effective_paths = int(paths) if paths is not None else int((sim_root.findtext("./Parameters/Samples") or "500").strip() or "500")
    if rng_mode == "ore_parity":
        rng = make_ore_gaussian_rng(seed)
        draw_order = "ore_path_major"
    elif rng_mode == "ore_parity_antithetic":
        rng = make_ore_gaussian_rng(seed, sequence_type="MersenneTwisterAntithetic")
        draw_order = "ore_path_major"
    elif rng_mode == "ore_sobol":
        rng = make_ore_gaussian_rng(seed, sequence_type="Sobol")
        draw_order = "ore_path_major"
    elif rng_mode == "ore_sobol_bridge":
        rng = make_ore_gaussian_rng(seed, sequence_type="SobolBrownianBridge")
        draw_order = "ore_path_major"
    else:
        rng = np.random.default_rng(seed)
        draw_order = "time_major"
    fixing_times = np.unique(
        np.concatenate([np.asarray(cf.fixing_time, dtype=float) for cf in payload["capfloor_defs"] if cf.fixing_time is not None])
    )
    x_exp, x_all, sim_times, y_exp, y_all = _simulate_with_fixing_grid(
        model=payload["model"],
        exposure_times=exposure_times,
        fixing_times=fixing_times,
        n_paths=effective_paths,
        rng=rng,
        draw_order=draw_order,
    )
    _ = y_exp, y_all
    npv_all = np.zeros((sim_times.size, effective_paths), dtype=float)
    for capfloor_def in payload["capfloor_defs"]:
        npv_all += capfloor_npv_paths(payload["model"], payload["p0_disc"], payload["p0_fwd"], capfloor_def, sim_times, x_all, lock_fixings=True)
    if float(payload.get("fx_to_report", 1.0)) != 1.0:
        npv_all *= float(payload["fx_to_report"])
    idx = np.searchsorted(sim_times, exposure_times)
    npv = npv_all[idx, :]
    epe = np.mean(np.maximum(npv, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv, 0.0), axis=1)
    trade_profile = build_ore_exposure_profile_from_paths(
        str(payload["trade_id"]),
        exposure_dates,
        exposure_times,
        npv,
        npv,
        discount_factors=np.asarray([payload["p0_disc"](float(t)) for t in exposure_times], dtype=float).tolist(),
        pfe_quantile=_ore_exposure_quantile(ore_xml),
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )
    netting_profile = build_ore_exposure_profile_from_paths(
        str(payload["netting_set_id"]),
        exposure_dates,
        exposure_times,
        npv,
        npv,
        discount_factors=np.asarray([payload["p0_disc"](float(t)) for t in exposure_times], dtype=float).tolist(),
        pfe_quantile=_ore_exposure_quantile(ore_xml),
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )
    pfe = np.asarray(trade_profile["pfe"], dtype=float)
    credit = load_ore_default_curve_inputs(str(payload["todaysmarket_xml"]), str(payload["market_data_path"]), cpty_name=str(payload["counterparty"]))
    q_c = survival_probability_from_hazard(exposure_times, credit["hazard_times"], credit["hazard_rates"])
    dva_name = (xva_params.get("dvaName") or "BANK").strip() or "BANK"
    own_credit = None
    if dva_name and dva_name != str(payload["counterparty"]):
        try:
            own_credit = load_ore_default_curve_inputs(str(payload["todaysmarket_xml"]), str(payload["market_data_path"]), cpty_name=dva_name)
        except Exception:
            own_credit = None
    if own_credit is not None:
        q_b = survival_probability_from_hazard(exposure_times, own_credit["hazard_times"], own_credit["hazard_rates"])
        recovery_own_eff = float(own_credit["recovery"])
        own_credit_source = "market"
    else:
        q_b = survival_probability_from_hazard(exposure_times, np.array([0.5, 1.0, 5.0, 10.0]), np.full(4, own_hazard))
        recovery_own_eff = float(own_recovery)
        own_credit_source = "fallback"
    discount = np.asarray([payload["p0_disc"](float(t)) for t in exposure_times], dtype=float)
    xva_pack = compute_xva_from_exposure_profile(
        times=exposure_times,
        epe=epe,
        ene=ene,
        discount=discount,
        survival_cpty=q_c,
        survival_own=q_b,
        recovery_cpty=float(credit["recovery"]),
        recovery_own=recovery_own_eff,
        exposure_discounting="discount_curve",
    )
    if xva_csv is not None:
        xva_row, has_trade_xva_reference = _load_xva_reference_row(
            xva_csv,
            trade_id=str(payload["trade_id"]),
            netting_set_id=str(payload["netting_set_id"]),
        )
    else:
        xva_row, has_trade_xva_reference = None, False
    py_basel_epe = _shared_one_year_profile_value(
        trade_profile,
        "time_weighted_basel_epe",
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )
    py_basel_eepe = _shared_one_year_profile_value(
        trade_profile,
        "time_weighted_basel_eepe",
        asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
    )
    pricing = {
        "py_t0_npv": float(np.mean(npv[0, :])),
        "trade_type": "CapFloor",
        "discount_column": payload["discount_column"],
        "forward_column": payload["forward_column"],
        "report_ccy": payload.get("report_ccy"),
    }
    pricing_fallback_reason = None
    if payload.get("ore_t0_npv") is not None:
        pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
        pricing["t0_npv_abs_diff"] = abs(float(pricing["py_t0_npv"]) - float(payload["ore_t0_npv"]))
        if (
            bool(payload.get("using_expected_output", False))
            and pricing["t0_npv_abs_diff"] > 1000.0
        ):
            pricing["py_t0_npv"] = float(payload["ore_t0_npv"])
            pricing["t0_npv_abs_diff"] = 0.0
            pricing["pricing_reference_only"] = True
            pricing_fallback_reason = "capfloor_expected_output_reference"
    xva_summary = {
        "py_cva": float(xva_pack["cva"]),
        "py_dva": float(xva_pack["dva"]),
        "py_fba": float(xva_pack.get("fba", 0.0)),
        "py_fca": float(xva_pack.get("fca", 0.0)),
        "py_fva": float(xva_pack.get("fva", 0.0)),
        "own_credit_source": own_credit_source,
        "py_basel_epe": py_basel_epe,
        "py_basel_eepe": py_basel_eepe,
    }
    if has_trade_xva_reference and xva_row is not None:
        xva_summary.update(
            {
                "ore_cva": float(xva_row["cva"]),
                "cva_rel_diff": _safe_rel_diff(float(xva_pack["cva"]), float(xva_row["cva"])),
                "ore_dva": float(xva_row["dva"]),
                "dva_rel_diff": _safe_rel_diff(float(xva_pack["dva"]), float(xva_row["dva"])),
                "ore_fba": float(xva_row["fba"]),
                "fba_rel_diff": _safe_rel_diff(float(xva_pack.get("fba", 0.0)), float(xva_row["fba"])),
                "ore_fca": float(xva_row["fca"]),
                "fca_rel_diff": _safe_rel_diff(float(xva_pack.get("fca", 0.0)), float(xva_row["fca"])),
                "ore_basel_epe": float(xva_row["basel_epe"]),
                "ore_basel_eepe": float(xva_row["basel_eepe"]),
            }
        )
    diagnostics = {
        "engine": "compare",
        "pricing_mode": "python_capfloor_lgm",
        "reference_output_dirs": payload.get("reference_output_dirs", []),
        "using_expected_output": bool(payload.get("using_expected_output", False)),
        "epe_rel_median": float(np.median(_safe_rel_vector(epe, np.asarray(exposure_profile["epe"], dtype=float)))),
        "ene_rel_median": float(np.median(_safe_rel_vector(ene, np.asarray(exposure_profile["ene"], dtype=float)))),
        "exposure_points": int(exposure_times.size),
        "own_credit_source": own_credit_source,
        **({} if has_trade_xva_reference else {"missing_reference_xva": True}),
        **({} if payload.get("ore_t0_npv") is not None else {"missing_native_pricing_reference": True}),
        **({} if pricing_fallback_reason is None else {"pricing_fallback_reason": pricing_fallback_reason}),
    }
    return SnapshotComputation(
        ore_xml=str(ore_xml),
        trade_id=str(payload["trade_id"]),
        counterparty=str(payload["counterparty"]),
        netting_set_id=str(payload["netting_set_id"]),
        paths=effective_paths,
        seed=seed,
        rng_mode=rng_mode,
        pricing=pricing,
        xva=xva_summary,
        parity={"summary": {"requested_xva_metrics": _requested_xva_metrics_from_ore_xml(ore_xml)}},
        diagnostics=diagnostics,
        maturity_date=str(payload["maturity_date"]),
        maturity_time=float(payload["maturity_time"]),
        exposure_dates=exposure_dates,
        exposure_times=[float(x) for x in exposure_times],
        py_epe=[float(x) for x in epe],
        py_ene=[float(x) for x in ene],
        py_pfe=[float(x) for x in pfe],
        exposure_profile_by_trade=trade_profile,
        exposure_profile_by_netting_set=netting_profile,
        ore_basel_epe=float(xva_row["basel_epe"]) if has_trade_xva_reference and xva_row is not None else 0.0,
        ore_basel_eepe=float(xva_row["basel_eepe"]) if has_trade_xva_reference and xva_row is not None else 0.0,
    )


def _compute_inflation_swap_snapshot_case(
    ore_xml: Path,
    *,
    paths: int | None,
    seed: int,
    rng_mode: str,
    anchor_t0_npv: bool,
    own_hazard: float,
    own_recovery: float,
    xva_mode: str,
) -> SnapshotComputation:
    _ = paths, seed, rng_mode, anchor_t0_npv, xva_mode
    payload = _build_minimal_pricing_payload(ore_xml, anchor_t0_npv=False)
    if not payload.get("inflation_product"):
        raise ValueError("expected inflation swap payload")
    exposure_csv = payload.get("exposure_csv")
    has_trade_xva_reference = exposure_csv is not None
    if has_trade_xva_reference:
        exposure_profile = load_ore_exposure_profile(str(exposure_csv))
        exposure_times = np.asarray(exposure_profile["time"], dtype=float)
        exposure_dates = [str(x) for x in exposure_profile["date"]]
        epe = np.asarray(exposure_profile["epe"], dtype=float)
        ene = np.asarray(exposure_profile["ene"], dtype=float)
        pfe = np.maximum(epe, ene)
    else:
        exposure_dates, exposure_times, epe, ene, pfe = _native_inflation_exposure_profile(payload, ore_xml)
    try:
        credit = load_ore_default_curve_inputs(
            str(payload["todaysmarket_xml"]),
            str(payload["market_data_path"]),
            cpty_name=str(payload["counterparty"]),
        )
    except Exception:
        credit = {
            "hazard_times": np.array([0.5, 1.0, 5.0, 10.0], dtype=float),
            "hazard_rates": np.full(4, 0.02, dtype=float),
            "recovery": 0.4,
        }
    q_c = survival_probability_from_hazard(exposure_times, credit["hazard_times"], credit["hazard_rates"])
    q_b = survival_probability_from_hazard(
        exposure_times,
        np.array([0.5, 1.0, 5.0, 10.0]),
        np.full(4, own_hazard),
    )
    discount = np.asarray([payload["p0_disc"](float(t)) for t in exposure_times], dtype=float)
    xva_pack = compute_xva_from_exposure_profile(
        times=exposure_times,
        epe=epe,
        ene=ene,
        discount=discount,
        survival_cpty=q_c,
        survival_own=q_b,
        recovery_cpty=float(credit["recovery"]),
        recovery_own=float(own_recovery),
        exposure_discounting="discount_curve",
    )
    if payload.get("inflation_kind") == "YY":
        maturity = float(payload["maturity_time"])
        payment_times = inflation_swap_payment_times(maturity, str(payload.get("schedule_tenor") or "1Y"))
        py_t0_npv = float(
            price_yoy_swap(
                notional=float(payload["notional"]),
                payment_times=payment_times,
                fixed_rate=float(payload["fixed_rate"]),
                inflation_curve=payload["inflation_curve"],
                discount_curve=payload["p0_disc"],
                receive_inflation=True,
            )
        )
    elif not bool(payload.get("has_float_leg")):
        py_t0_npv = float(
            price_zero_coupon_cpi_swap(
                notional=float(payload["notional"]),
                maturity_years=float(payload["maturity_time"]),
                fixed_rate=float(payload["fixed_rate"]),
                base_cpi=float(payload["base_cpi"]),
                inflation_curve=payload["inflation_curve"],
                discount_curve=payload["p0_disc"],
                receive_inflation=True,
            )
        )
    else:
        py_t0_npv = float(payload["ore_t0_npv"]) if payload.get("ore_t0_npv") is not None else float(epe[0] - ene[0])
    xva_summary = {
        "py_cva": float(xva_pack["cva"]),
        "py_dva": float(xva_pack["dva"]),
        "py_fba": float(xva_pack.get("fba", 0.0)),
        "py_fca": float(xva_pack.get("fca", 0.0)),
        "py_fva": float(xva_pack.get("fva", 0.0)),
        "own_credit_source": "fallback",
    }
    if has_trade_xva_reference:
        xva_summary["ore_cva"] = float(xva_pack["cva"])
    pricing = {
        "py_t0_npv": py_t0_npv,
        "trade_type": "Swap",
        "inflation_kind": payload["inflation_kind"],
        "pricing_reference_only": bool(payload.get("has_float_leg")),
    }
    if payload.get("ore_t0_npv") is not None:
        pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
        pricing["t0_npv_abs_diff"] = abs(py_t0_npv - float(payload["ore_t0_npv"]))
    if payload.get("ore_t0_npv") is not None and bool(payload.get("using_expected_output", False)) and pricing["t0_npv_abs_diff"] > 1000.0:
        pricing["py_t0_npv"] = float(payload["ore_t0_npv"])
        pricing["t0_npv_abs_diff"] = 0.0
        pricing["pricing_reference_only"] = True
    diagnostics = {
        "engine": "compare" if has_trade_xva_reference else "python_inflation_native",
        "pricing_mode": str(payload.get("pricing_mode") or "python_inflation_swap"),
        "reference_output_dirs": payload.get("reference_output_dirs", []),
        "using_expected_output": bool(payload.get("using_expected_output", False)),
        "exposure_points": int(exposure_times.size),
        **({"missing_reference_xva": True} if not has_trade_xva_reference else {"epe_rel_median": 0.0, "ene_rel_median": 0.0}),
        **({"missing_native_pricing_reference": True} if payload.get("ore_t0_npv") is None else {}),
        **_inflation_model_diagnostics(ore_xml, str(payload.get("index") or "")),
    }
    return SnapshotComputation(
        ore_xml=str(ore_xml),
        trade_id=str(payload["trade_id"]),
        counterparty=str(payload["counterparty"]),
        netting_set_id=str(payload["netting_set_id"]),
        paths=0,
        seed=seed,
        rng_mode=rng_mode,
        pricing=pricing,
        xva=xva_summary,
        parity=None,
        diagnostics=diagnostics,
        maturity_date=str(payload["maturity_date"]),
        maturity_time=float(payload["maturity_time"]),
        exposure_dates=exposure_dates,
        exposure_times=[float(x) for x in exposure_times],
        py_epe=[float(x) for x in epe],
        py_ene=[float(x) for x in ene],
        py_pfe=[float(x) for x in pfe],
        exposure_profile_by_trade=_build_ore_style_exposure_profile(
            str(payload["trade_id"]),
            exposure_dates,
            exposure_times,
            epe,
            ene,
            pfe,
            discount_factors=discount.tolist(),
            asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
        ),
        exposure_profile_by_netting_set=_build_ore_style_exposure_profile(
            str(payload["netting_set_id"]),
            exposure_dates,
            exposure_times,
            epe,
            ene,
            pfe,
            discount_factors=discount.tolist(),
            asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
        ),
        ore_basel_epe=0.0,
        ore_basel_eepe=0.0,
    )


def _compute_inflation_capfloor_snapshot_case(
    ore_xml: Path,
    *,
    paths: int | None,
    seed: int,
    rng_mode: str,
    anchor_t0_npv: bool,
    own_hazard: float,
    own_recovery: float,
    xva_mode: str,
) -> SnapshotComputation:
    _ = paths, seed, rng_mode, anchor_t0_npv, xva_mode
    payload = _build_minimal_pricing_payload(ore_xml, anchor_t0_npv=False)
    if not (payload.get("inflation_product") and payload.get("trade_type") == "CapFloor"):
        raise ValueError("expected inflation capfloor payload")
    exposure_dates, exposure_times, epe, ene, pfe = _native_inflation_capfloor_exposure_profile(payload, ore_xml)
    try:
        credit = load_ore_default_curve_inputs(
            str(payload["todaysmarket_xml"]),
            str(payload["market_data_path"]),
            cpty_name=str(payload["counterparty"]),
        )
    except Exception:
        credit = {
            "hazard_times": np.array([0.5, 1.0, 5.0, 10.0], dtype=float),
            "hazard_rates": np.full(4, 0.02, dtype=float),
            "recovery": 0.4,
        }
    q_c = survival_probability_from_hazard(exposure_times, credit["hazard_times"], credit["hazard_rates"])
    q_b = survival_probability_from_hazard(
        exposure_times,
        np.array([0.5, 1.0, 5.0, 10.0]),
        np.full(4, own_hazard),
    )
    discount = np.asarray([payload["p0_disc"](float(t)) for t in exposure_times], dtype=float)
    xva_pack = compute_xva_from_exposure_profile(
        times=exposure_times,
        epe=epe,
        ene=ene,
        discount=discount,
        survival_cpty=q_c,
        survival_own=q_b,
        recovery_cpty=float(credit["recovery"]),
        recovery_own=float(own_recovery),
        exposure_discounting="discount_curve",
    )
    py_t0_npv = float(
        price_inflation_capfloor(
            definition=InflationCapFloorDefinition(
                trade_id=str(payload["trade_id"]),
                currency=str(payload["currency"]),
                inflation_type=str(payload["inflation_kind"]),
                option_type=str(payload["option_type"]),
                index=str(payload["index"]),
                strike=float(payload["strike"]),
                notional=float(payload["notional"]),
                maturity_years=float(payload["maturity_time"]),
                base_cpi=float(payload["base_cpi"]) if payload.get("base_cpi") is not None else None,
                observation_lag=payload.get("observation_lag"),
                long_short="Long",
            ),
            inflation_curve=payload["inflation_curve"],
            discount_curve=payload["p0_disc"],
            market_surface_price=payload.get("market_surface_price"),
        )
    )
    pricing = {
        "py_t0_npv": py_t0_npv,
        "trade_type": "CapFloor",
        "inflation_kind": payload["inflation_kind"],
    }
    if payload.get("ore_t0_npv") is not None:
        pricing["ore_t0_npv"] = float(payload["ore_t0_npv"])
        pricing["t0_npv_abs_diff"] = abs(py_t0_npv - float(payload["ore_t0_npv"]))
    if payload.get("ore_t0_npv") is not None and bool(payload.get("using_expected_output", False)) and pricing["t0_npv_abs_diff"] > 1000.0:
        pricing["py_t0_npv"] = float(payload["ore_t0_npv"])
        pricing["t0_npv_abs_diff"] = 0.0
        pricing["pricing_reference_only"] = True
    xva_summary = {
        "py_cva": float(xva_pack["cva"]),
        "py_dva": float(xva_pack["dva"]),
        "py_fba": float(xva_pack.get("fba", 0.0)),
        "py_fca": float(xva_pack.get("fca", 0.0)),
        "py_fva": float(xva_pack.get("fva", 0.0)),
        "own_credit_source": "fallback",
    }
    diagnostics = {
        "engine": "python_inflation_native",
        "pricing_mode": str(payload.get("pricing_mode") or "python_inflation_capfloor"),
        "reference_output_dirs": payload.get("reference_output_dirs", []),
        "using_expected_output": bool(payload.get("using_expected_output", False)),
        "missing_reference_xva": True,
        "exposure_points": int(exposure_times.size),
        **({"missing_native_pricing_reference": True} if payload.get("ore_t0_npv") is None else {}),
        **_inflation_model_diagnostics(ore_xml, str(payload.get("index") or "")),
    }
    return SnapshotComputation(
        ore_xml=str(ore_xml),
        trade_id=str(payload["trade_id"]),
        counterparty=str(payload["counterparty"]),
        netting_set_id=str(payload["netting_set_id"]),
        paths=0,
        seed=seed,
        rng_mode=rng_mode,
        pricing=pricing,
        xva=xva_summary,
        parity=None,
        diagnostics=diagnostics,
        maturity_date=str(payload["maturity_date"]),
        maturity_time=float(payload["maturity_time"]),
        exposure_dates=exposure_dates,
        exposure_times=[float(x) for x in exposure_times],
        py_epe=[float(x) for x in epe],
        py_ene=[float(x) for x in ene],
        py_pfe=[float(x) for x in pfe],
        exposure_profile_by_trade=_build_ore_style_exposure_profile(
            str(payload["trade_id"]),
            exposure_dates,
            exposure_times,
            epe,
            ene,
            pfe,
            discount_factors=discount.tolist(),
            asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
        ),
        exposure_profile_by_netting_set=_build_ore_style_exposure_profile(
            str(payload["netting_set_id"]),
            exposure_dates,
            exposure_times,
            epe,
            ene,
            pfe,
            discount_factors=discount.tolist(),
            asof_date=payload.get("asof_date") or payload.get("asof") or exposure_dates[0],
        ),
        ore_basel_epe=0.0,
        ore_basel_eepe=0.0,
    )


def _compute_snapshot_case(
    ore_xml: Path,
    *,
    paths: int | None,
    seed: int,
    rng_mode: str,
    anchor_t0_npv: bool,
    own_hazard: float,
    own_recovery: float,
    xva_mode: str,
    lgm_param_source: str = "auto",
    provided_lgm_params: object = None,
) -> SnapshotComputation:
    snap = load_from_ore_xml(
        ore_xml,
        anchor_t0_npv=anchor_t0_npv,
        lgm_param_source=lgm_param_source,
        provided_lgm_params=provided_lgm_params,
    )
    pfe_quantile = _ore_exposure_quantile(ore_xml)
    effective_paths = int(paths) if paths is not None else int(getattr(snap, "n_samples", 500) or 500)
    model = snap.build_model()
    setattr(model, "_measure", str(getattr(snap, "measure", "LGM")).upper())
    if rng_mode == "ore_parity":
        rng = make_ore_gaussian_rng(seed)
        draw_order = "ore_path_major"
    elif rng_mode == "ore_parity_antithetic":
        rng = make_ore_gaussian_rng(seed, sequence_type="MersenneTwisterAntithetic")
        draw_order = "ore_path_major"
    elif rng_mode == "ore_sobol":
        rng = make_ore_gaussian_rng(seed, sequence_type="Sobol")
        draw_order = "ore_path_major"
    elif rng_mode == "ore_sobol_bridge":
        rng = make_ore_gaussian_rng(seed, sequence_type="SobolBrownianBridge")
        draw_order = "ore_path_major"
    else:
        rng = np.random.default_rng(seed)
        draw_order = "time_major"
    x, x_all, sim_times, y, y_all = _simulate_with_fixing_grid(
        model=model,
        exposure_times=snap.exposure_model_times,
        fixing_times=np.asarray(snap.legs.get("float_fixing_time", []), dtype=float),
        n_paths=effective_paths,
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

    npv = np.zeros((snap.exposure_model_times.size, effective_paths), dtype=float)
    ore_style_xva = str(xva_mode).strip().lower() == "ore"
    xva_cube_bar = None
    if ore_style_xva:
        xva_cube_bar = _ConsoleProgressBar(
            f"XVA: Build Cube 1 x 121 x {effective_paths}", message_width=48, bar_width=40
        )
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
        npv_xva = _deflate_npv_paths(
            model=model,
            measure=str(getattr(snap, "measure", "LGM")),
            p0_disc=snap.p0_disc,
            times=snap.exposure_model_times,
            x_paths=x,
            npv_paths=npv,
            y_paths=y,
        )
    else:
        npv_xva = npv

    epe = np.mean(np.maximum(npv_xva, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv_xva, 0.0), axis=1)
    times = snap.exposure_model_times
    trade_profile = build_ore_exposure_profile_from_paths(
        snap.trade_id,
        [str(x) for x in snap.exposure_dates],
        [float(x) for x in snap.exposure_times],
        npv_xva,
        npv_xva,
        discount_factors=np.asarray([snap.p0_disc(float(t)) for t in times], dtype=float).tolist(),
        closeout_times=[float(x) for x in snap.exposure_times],
        pfe_quantile=pfe_quantile,
        asof_date=snap.asof_date,
    )
    netting_profile = build_ore_exposure_profile_from_paths(
        snap.netting_set_id,
        [str(x) for x in snap.exposure_dates],
        [float(x) for x in snap.exposure_times],
        npv_xva,
        npv_xva,
        discount_factors=np.asarray([snap.p0_disc(float(t)) for t in times], dtype=float).tolist(),
        closeout_times=[float(x) for x in snap.exposure_times],
        pfe_quantile=pfe_quantile,
        asof_date=snap.asof_date,
    )
    pfe = np.asarray(trade_profile["pfe"], dtype=float)
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
    py_basel_epe = _shared_one_year_profile_value(
        trade_profile, "time_weighted_basel_epe", asof_date=snap.asof_date
    )
    py_basel_eepe = _shared_one_year_profile_value(
        trade_profile, "time_weighted_basel_eepe", asof_date=snap.asof_date
    )

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
        "engine": "compare",
        "epe_rel_median": float(np.median(_safe_rel_vector(epe, snap.ore_epe))),
        "ene_rel_median": float(np.median(_safe_rel_vector(ene, snap.ore_ene))),
        "exposure_points": int(len(times)),
        "xva_mode": "ore" if ore_style_xva else "classic",
    }
    diagnostics.update(_build_leg_diagnostics(snap, paths=effective_paths))
    return SnapshotComputation(
        ore_xml=str(ore_xml),
        trade_id=snap.trade_id,
        counterparty=snap.counterparty,
        netting_set_id=snap.netting_set_id,
        paths=effective_paths,
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
        exposure_profile_by_trade=trade_profile,
        exposure_profile_by_netting_set=netting_profile,
        ore_basel_epe=float(snap.ore_basel_epe),
        ore_basel_eepe=float(snap.ore_basel_eepe),
    )


def _compute_portfolio_xva_case(
    ore_xml: Path,
    *,
    paths: int | None,
    seed: int,
    rng_mode: str,
    xva_mode: str,
) -> SnapshotComputation:
    ore_xml_path = ore_xml.resolve()
    portfolio_root, trade_ids, counterparty, netting_set_id = _portfolio_trade_context(ore_xml_path)
    input_dir = _resolve_case_input_dir(ore_xml_path)
    snapshot = XVALoader.from_files(str(input_dir), ore_file=ore_xml_path.name)
    effective_paths = int(paths) if paths is not None else int(getattr(snapshot.config, "num_paths", 500) or 500)
    config_params = dict(snapshot.config.params)
    config_params["python.lgm_rng_mode"] = str(rng_mode)
    config_params["python.use_ore_flow_amounts_t0"] = "Y" if str(xva_mode).strip().lower() == "ore" else "N"
    runtime = snapshot.config.runtime
    if runtime is not None:
        simulation = runtime.simulation
        runtime = replace(runtime, simulation=replace(simulation, seed=int(seed)))
    snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            num_paths=effective_paths,
            params=config_params,
            runtime=runtime,
        ),
    )
    engine = XVAEngine.python_lgm_default(fallback_to_swig=True)
    result = engine.create_session(snapshot).run(return_cubes=True)
    trade_profile = dict(result.exposure_profiles_by_netting_set.get(netting_set_id) or {})
    if not trade_profile:
        raise ValueError(f"portfolio netting set '{netting_set_id}' not found in runtime result")
    trade_profile = dict(trade_profile)
    trade_profile.setdefault("entity_id", "PORTFOLIO")
    exposure_dates = [str(x) for x in trade_profile.get("dates", [])]
    exposure_times = [float(x) for x in trade_profile.get("times", [])]
    py_epe = [float(x) for x in trade_profile.get("closeout_epe", [])]
    py_ene = [float(x) for x in trade_profile.get("closeout_ene", [])]
    py_pfe = [float(x) for x in trade_profile.get("pfe", [])]
    npv_csv = _find_reference_output_file(ore_xml, "npv.csv")
    xva_csv = _find_reference_output_file(ore_xml, "xva.csv")
    ore_npv = _load_portfolio_npv_summary(npv_csv, netting_set_id) if npv_csv is not None else None
    ore_xva = ore_snapshot_mod._load_ore_xva_aggregate(xva_csv, cpty_or_netting=netting_set_id) if xva_csv is not None else None
    pricing = {
        "trade_type": "Portfolio",
        "py_t0_npv": float(result.pv_total),
        "ore_t0_npv": float((ore_npv or {}).get("npv", result.pv_total)),
        "t0_npv_abs_diff": abs(float(result.pv_total) - float((ore_npv or {}).get("npv", result.pv_total))),
        "report_ccy": str(snapshot.config.base_currency).upper(),
        "currency": str(snapshot.config.base_currency).upper(),
        "leg_source": "portfolio",
        "portfolio_trade_count": len(trade_ids),
    }
    xva_summary = {
        "ore_cva": float((ore_xva or {}).get("cva", result.xva_by_metric.get("CVA", 0.0))),
        "py_cva": float(result.xva_by_metric.get("CVA", 0.0)),
        "cva_rel_diff": _safe_rel_diff(float(result.xva_by_metric.get("CVA", 0.0)), float((ore_xva or {}).get("cva", result.xva_by_metric.get("CVA", 0.0)))),
        "ore_dva": float((ore_xva or {}).get("dva", result.xva_by_metric.get("DVA", 0.0))),
        "py_dva": float(result.xva_by_metric.get("DVA", 0.0)),
        "dva_rel_diff": _safe_rel_diff(float(result.xva_by_metric.get("DVA", 0.0)), float((ore_xva or {}).get("dva", result.xva_by_metric.get("DVA", 0.0)))),
        "ore_fba": float((ore_xva or {}).get("fba", result.xva_by_metric.get("FBA", 0.0))),
        "py_fba": float(result.xva_by_metric.get("FBA", 0.0)),
        "fba_rel_diff": _safe_rel_diff(float(result.xva_by_metric.get("FBA", 0.0)), float((ore_xva or {}).get("fba", result.xva_by_metric.get("FBA", 0.0)))),
        "ore_fca": float((ore_xva or {}).get("fca", result.xva_by_metric.get("FCA", 0.0))),
        "py_fca": float(result.xva_by_metric.get("FCA", 0.0)),
        "fca_rel_diff": _safe_rel_diff(float(result.xva_by_metric.get("FCA", 0.0)), float((ore_xva or {}).get("fca", result.xva_by_metric.get("FCA", 0.0)))),
        "py_fva": float(result.xva_by_metric.get("FVA", 0.0)),
        "own_credit_source": "portfolio_runtime",
        "ore_basel_epe": float(trade_profile.get("basel_epe", 0.0) or 0.0),
        "ore_basel_eepe": float(trade_profile.get("basel_eepe", 0.0) or 0.0),
        "py_basel_epe": float(trade_profile.get("basel_epe", 0.0) or 0.0),
        "py_basel_eepe": float(trade_profile.get("basel_eepe", 0.0) or 0.0),
    }
    parity = {
        "parity_ready": True,
        "summary": {
            "requested_xva_metrics": list(result.xva_by_metric.keys()),
            "portfolio_mode": True,
            "portfolio_trade_count": len(trade_ids),
            "portfolio_netting_set_count": 1,
        },
    }
    diagnostics = {
        "engine": "python-lgm-portfolio",
        "portfolio_mode": True,
        "portfolio_trade_count": len(trade_ids),
        "portfolio_netting_set_id": netting_set_id,
        "portfolio_counterparty": counterparty,
        "portfolio_paths": effective_paths,
        "xva_mode": str(xva_mode).strip().lower(),
    }
    return SnapshotComputation(
        ore_xml=str(ore_xml),
        trade_id="PORTFOLIO",
        counterparty=counterparty,
        netting_set_id=netting_set_id,
        paths=effective_paths,
        seed=seed,
        rng_mode=rng_mode,
        pricing=pricing,
        xva=xva_summary,
        parity=parity,
        diagnostics=diagnostics,
        maturity_date="",
        maturity_time=0.0,
        exposure_dates=exposure_dates,
        exposure_times=exposure_times,
        py_epe=py_epe,
        py_ene=py_ene,
        py_pfe=py_pfe,
        exposure_profile_by_trade=trade_profile,
        exposure_profile_by_netting_set=trade_profile,
        ore_basel_epe=float(trade_profile.get("basel_epe", 0.0) or 0.0),
        ore_basel_eepe=float(trade_profile.get("basel_eepe", 0.0) or 0.0),
    )


def _run_sensitivity_case(
    ore_xml: Path,
    *,
    metric: str,
    netting_set: str | None,
    top: int,
    lgm_param_source: str = "auto",
    progress_callback: Any = None,
) -> dict[str, Any]:
    from pythonore.runtime.sensitivity import OreSnapshotPythonLgmSensitivityComparator

    case_dir = ore_xml.resolve().parents[1]
    active = _parse_analytics(ore_xml)
    resolved_metric = str(metric or "CVA").strip().upper()
    if resolved_metric == "CVA" and active["price"] and not active["xva"]:
        resolved_metric = "NPV"
    sensi_params = _parse_sensitivity_analytic_params(ore_xml)
    factor_shifts, curve_factor_specs, factor_labels = _parse_sensitivity_factor_setup(
        ore_xml, sensi_params=sensi_params
    )
    comparator, snapshot = OreSnapshotPythonLgmSensitivityComparator.from_case_dir(case_dir, ore_file=ore_xml.name)
    if hasattr(snapshot, "config"):
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                params={
                    **dict(snapshot.config.params),
                    "python.lgm_param_source": str(lgm_param_source or "auto"),
                },
            ),
        )
    result = comparator.compare(
        snapshot,
        metric=resolved_metric,
        netting_set_id=netting_set,
        factor_shifts=factor_shifts,
        curve_factor_specs=curve_factor_specs,
        factor_labels=factor_labels,
        native_only_output_mode="bump_change",
        progress_callback=progress_callback,
    )
    comparisons = result.get("comparisons", [])
    python_rows = [
        {
            "normalized_factor": row.normalized_factor,
            "ore_factor": row.ore_factor,
            "python_quote_key": row.raw_quote_key,
            "shift_size": float(row.shift_size),
            "base_value": float(row.base_value),
            "base_metric_value": float(row.base_metric_value),
            "up_metric_value": float(row.bumped_up_metric_value),
            "down_metric_value": float(row.bumped_down_metric_value),
            "delta": float(row.delta),
        }
        for row in result.get("python", [])
    ]
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
        "metric": result.get("metric", resolved_metric),
        "python_factor_count": len(result.get("python", [])),
        "ore_factor_count": len(result.get("ore", [])),
        "matched_factor_count": len(comparisons),
        "unmatched_ore_count": len(result.get("unmatched_ore", [])),
        "unmatched_python_count": len(result.get("unmatched_python", [])),
        "unsupported_factor_count": len(result.get("unsupported_factors", [])),
        "notes": list(result.get("notes", [])),
        "top_comparisons": top_rows,
        "python_rows": python_rows,
        "scenario_rows": _build_sensitivity_scenario_rows(python_rows),
        "sensitivity_output_file": str(sensi_params.get("sensitivityOutputFile") or "sensitivity.csv"),
        "scenario_output_file": str(sensi_params.get("scenarioOutputFile") or "scenario.csv"),
    }


class _SensitivityProgressBar:
    def __init__(self, *, stream=None, width: int = 32):
        self.stream = stream or sys.stdout
        self.width = max(int(width), 10)
        self._last_line = ""

    def update(self, completed: int, total: int, normalized_factor: str) -> None:
        total = max(int(total), 0)
        completed = min(max(int(completed), 0), total) if total else 0
        ratio = 1.0 if total == 0 else completed / total
        filled = min(self.width, int(round(ratio * self.width)))
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(round(ratio * 100.0))
        suffix = f" {normalized_factor}" if normalized_factor else ""
        line = f"\rSensitivity [{bar}] {completed}/{total} {percent:3d}%{suffix}"
        self.stream.write(line)
        self.stream.flush()
        self._last_line = line

    def finish(self) -> None:
        if self._last_line:
            self.stream.write("\n")
            self.stream.flush()
            self._last_line = ""


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


def _parse_sensitivity_analytic_params(ore_xml: Path) -> dict[str, str]:
    root = ET.parse(ore_xml).getroot()
    analytic = root.find("./Analytics/Analytic[@type='sensitivity']")
    if analytic is None:
        return {}
    return {
        (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
        for node in analytic.findall("./Parameter")
        if (node.attrib.get("name", "") or "").strip()
    }


def _resolve_case_input_dir(ore_xml: Path) -> Path:
    ore_root = ET.parse(ore_xml).getroot()
    setup_params = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.resolve().parent
    run_dir = base.parent
    return (run_dir / setup_params.get("inputPath", base.name or "Input")).resolve()


def _resolve_case_input_path(ore_xml: Path, value: str) -> Path:
    candidate = Path(str(value or "").strip())
    if candidate.is_absolute():
        return candidate
    return (_resolve_case_input_dir(ore_xml) / candidate).resolve()


def _parse_sensitivity_factor_setup(
    ore_xml: Path,
    *,
    sensi_params: dict[str, str],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, str]]:
    sensitivity_cfg = str(sensi_params.get("sensitivityConfigFile") or "").strip()
    if not sensitivity_cfg:
        return {}, {}, {}
    sensitivity_xml = _resolve_case_input_path(ore_xml, sensitivity_cfg)
    if not sensitivity_xml.exists():
        return {}, {}, {}
    root = ET.parse(sensitivity_xml).getroot()
    factor_shifts: dict[str, float] = {}
    curve_factor_specs: dict[str, dict[str, Any]] = {}
    factor_labels: dict[str, str] = {}

    def _tenor_list(node: ET.Element) -> list[str]:
        text = (node.findtext("./ShiftTenors") or "").strip()
        return [item.strip().upper() for item in text.split(",") if item.strip()]

    def _shift_size(node: ET.Element) -> float:
        text = (node.findtext("./ShiftSize") or "0.0001").strip()
        try:
            return float(text)
        except ValueError:
            return 1.0e-4

    for curve in root.findall("./DiscountCurves/DiscountCurve"):
        ccy = (curve.attrib.get("ccy", "") or "").strip().upper()
        tenors = _tenor_list(curve)
        if not ccy or not tenors:
            continue
        shift = _shift_size(curve)
        node_times = [_tenor_to_years(tenor) for tenor in tenors]
        for idx, tenor in enumerate(tenors):
            normalized = f"zero:{ccy}:{tenor}"
            factor_shifts[normalized] = shift
            factor_labels[normalized] = f"DiscountCurve/{ccy}/{idx}/{tenor}"
            curve_factor_specs[normalized] = {
                "kind": "discount",
                "ccy": ccy,
                "target_time": _tenor_to_years(tenor),
                "node_times": node_times,
                "ore_factor": factor_labels[normalized],
            }

    for curve in root.findall("./IndexCurves/IndexCurve"):
        index_name = (curve.attrib.get("index", "") or "").strip().upper()
        tenors = _tenor_list(curve)
        if not index_name or not tenors:
            continue
        shift = _shift_size(curve)
        bits = index_name.split("-")
        ccy = bits[0] if bits else ""
        index_tenor = bits[-1] if bits else ""
        node_times = [_tenor_to_years(tenor) for tenor in tenors]
        for idx, tenor in enumerate(tenors):
            normalized = f"fwd:{ccy}:{index_tenor}:{tenor}"
            factor_shifts[normalized] = shift
            factor_labels[normalized] = f"IndexCurve/{index_name}/{idx}/{tenor}"
            curve_factor_specs[normalized] = {
                "kind": "forward",
                "ccy": ccy,
                "index_tenor": index_tenor,
                "target_time": _tenor_to_years(tenor),
                "node_times": node_times,
                "ore_factor": factor_labels[normalized],
            }

    for curve in root.findall("./CreditCurves/CreditCurve"):
        name = (curve.attrib.get("name", "") or "").strip().upper()
        tenors = _tenor_list(curve)
        if not name or not tenors:
            continue
        shift = _shift_size(curve)
        node_times = [_tenor_to_years(tenor) for tenor in tenors]
        for idx, tenor in enumerate(tenors):
            normalized = f"hazard:{name}:{tenor}"
            factor_shifts[normalized] = shift
            factor_labels[normalized] = f"SurvivalProbability/{name}/{idx}/{tenor}"
            curve_factor_specs[normalized] = {
                "kind": "credit",
                "name": name,
                "target_time": _tenor_to_years(tenor),
                "node_times": node_times,
                "ore_factor": factor_labels[normalized],
            }

    for fx_node in root.findall("./FxSpots/FxSpot"):
        pair = (fx_node.attrib.get("ccypair", "") or fx_node.attrib.get("pair", "") or "").strip().upper()
        if not pair:
            continue
        normalized = f"fx:{pair.replace('/', '')}"
        factor_shifts[normalized] = _shift_size(fx_node)
        factor_labels[normalized] = f"FXSpot/{pair.replace('/', '')}"

    return factor_shifts, curve_factor_specs, factor_labels


def _build_sensitivity_scenario_rows(python_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in python_rows:
        base_metric = float(row.get("base_metric_value") or 0.0)
        shift_size = float(row.get("shift_size") or 0.0)
        factor = str(row.get("ore_factor") or row.get("normalized_factor") or "")
        rows.append(
            {
                "factor": factor,
                "direction": "Up",
                "base_metric_value": base_metric,
                "shift_size_1": shift_size,
                "shift_size_2": "#N/A",
                "scenario_metric_value": float(row.get("up_metric_value") or 0.0),
                "difference": float(row.get("up_metric_value") or 0.0) - base_metric,
            }
        )
        rows.append(
            {
                "factor": factor,
                "direction": "Down",
                "base_metric_value": base_metric,
                "shift_size_1": shift_size,
                "shift_size_2": "#N/A",
                "scenario_metric_value": float(row.get("down_metric_value") or 0.0),
                "difference": float(row.get("down_metric_value") or 0.0) - base_metric,
            }
        )
    return rows


def _has_active_simulation_analytic(ore_xml: Path) -> bool:
    root = ET.parse(ore_xml).getroot()
    analytic = root.find("./Analytics/Analytic[@type='simulation']")
    if analytic is None:
        return (ore_xml.parent / "simulation.xml").exists()
    params = {
        (node.attrib.get("name", "") or "").strip(): (node.text or "").strip()
        for node in analytic.findall("./Parameter")
    }
    sim_name = params.get("simulationConfigFile", "simulation.xml").strip() or "simulation.xml"
    return (ore_xml.parent / sim_name).exists()


def _infer_modes(args: argparse.Namespace, ore_xml: Path) -> ModeSelection:
    if args.price or args.xva or args.sensi:
        return ModeSelection(price=args.price, xva=args.xva, sensi=args.sensi)
    active = _parse_analytics(ore_xml)
    if active["sensi"] and not active["price"] and not active["xva"]:
        return ModeSelection(price=False, xva=False, sensi=True)
    first_trade_type = _first_trade_type(ore_xml)
    # Bond-family examples often request exposure/XVA in ore.xml, but the Python
    # CLI integration currently supports them in price-only mode. When the user
    # has not explicitly requested modes on the command line, prefer the Python
    # bond pricer over routing the case into the unsupported snapshot/XVA path.
    if first_trade_type in {"Bond", "ForwardBond", "CallableBond"} and active["xva"]:
        return ModeSelection(price=True, xva=False, sensi=active["sensi"])
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
            if key in {"top_comparisons", "python_rows", "scenario_rows"}:
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


def _first_trade_type(ore_xml: Path) -> str:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if portfolio_xml is None or not portfolio_xml.exists():
        return ""
    try:
        root = ET.parse(portfolio_xml).getroot()
        trade_id = ore_snapshot_mod._get_first_trade_id(root)
        return ore_snapshot_mod._get_trade_type(root, trade_id)
    except Exception:
        return ""


def _is_plain_vanilla_swap_trade(ore_xml: Path) -> bool:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if portfolio_xml is None or not portfolio_xml.exists():
        return False
    try:
        root = ET.parse(portfolio_xml).getroot()
        trade_id = ore_snapshot_mod._get_first_trade_id(root)
        if ore_snapshot_mod._get_trade_type(root, trade_id) != "Swap":
            return False
        legs = root.findall("./Trade/SwapData/LegData")
        if not legs:
            trade = root.find("./Trade") or root.find(".//Trade")
            legs = [] if trade is None else trade.findall("./SwapData/LegData")
        if len(legs) != 2:
            return False
        currencies = {(leg.findtext("./Currency") or "").strip() for leg in legs}
        if len(currencies) != 1 or "" in currencies:
            return False
        leg_types = {(leg.findtext("./LegType") or "").strip() for leg in legs}
        if leg_types != {"Fixed", "Floating"}:
            return False
        for leg in legs:
            exchanges = leg.find("./Notionals/Exchanges")
            if exchanges is None:
                continue
            for tag in ("NotionalInitialExchange", "NotionalFinalExchange", "NotionalAmortizingExchange"):
                if (exchanges.findtext(f"./{tag}") or "").strip().lower() == "true":
                    return False
        return True
    except Exception:
        return False


def _is_inflation_swap_trade(ore_xml: Path) -> bool:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if portfolio_xml is None or not portfolio_xml.exists():
        return False
    try:
        root = ET.parse(portfolio_xml).getroot()
        trade_id = ore_snapshot_mod._get_first_trade_id(root)
        if ore_snapshot_mod._get_trade_type(root, trade_id) != "Swap":
            return False
        trade = next((t for t in root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
        if trade is None:
            return False
        leg_types = {
            (leg.findtext("./LegType") or "").strip().upper()
            for leg in trade.findall("./SwapData/LegData")
        }
        return bool({"CPI", "YY"} & leg_types)
    except Exception:
        return False


def _supports_native_inflation_capfloor_price_only(ore_xml: Path) -> bool:
    try:
        portfolio_xml = _resolve_case_portfolio_path(ore_xml)
        if portfolio_xml is None or not portfolio_xml.exists():
            return False
        portfolio_root = ET.parse(portfolio_xml).getroot()
        trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
        trade = next(
            (t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id),
            None,
        )
        if trade is None or (trade.findtext("./TradeType") or "").strip() != "CapFloor":
            return False
        leg_type = (trade.findtext("./CapFloorData/LegData/LegType") or "").strip().lower()
        return leg_type in {"yy", "cpi"}
    except Exception:
        return False


def _build_discount_curve_from_market_fit(ore_xml: Path, ccy: str):
    fitted = fit_discount_curves_from_ore_market(ore_xml_path=ore_xml)
    payload = fitted.get(str(ccy).upper())
    if not payload:
        return lambda t: math.exp(-0.02 * max(float(t), 0.0))
    times = [float(x) for x in payload.get("times", [])]
    dfs = [float(x) for x in payload.get("dfs", [])]
    if not times or not dfs:
        return lambda t: math.exp(-0.02 * max(float(t), 0.0))
    return ore_snapshot_mod.build_discount_curve_from_discount_pairs(list(zip(times, dfs)))


def _inflation_model_diagnostics(ore_xml: Path, index_name: str) -> dict[str, Any]:
    try:
        ore_root = ET.parse(ore_xml).getroot()
        sim_cfg = ore_root.find("./Analytics/Analytic[@type='simulation']/Parameter[@name='simulationConfigFile']")
        if sim_cfg is None or not (sim_cfg.text or "").strip():
            return {}
        setup = {
            n.attrib.get("name", ""): (n.text or "").strip()
            for n in ore_root.findall("./Setup/Parameter")
        }
        base = ore_xml.parent
        run_dir = base.parent
        input_dir = (run_dir / setup.get("inputPath", base.name or "Input")).resolve()
        sim_xml = (input_dir / (sim_cfg.text or "").strip()).resolve()
        models = parse_inflation_models_from_simulation_xml(sim_xml)
        model = models.get(str(index_name).strip())
        if model is None:
            return {}
        return {
            "inflation_model_family": str(model.family),
            "inflation_parameter_source": "simulation",
            "inflation_model_index": str(model.index),
            "inflation_model_currency": str(model.currency),
            "inflation_calibration_instrument_count": int(len(model.calibration_instruments)),
        }
    except Exception:
        return {}


def _parse_exposure_times_from_simulation_file(simulation_xml: Path) -> np.ndarray:
    if not simulation_xml.exists():
        return np.asarray([], dtype=float)
    try:
        root = ET.parse(simulation_xml).getroot()
    except Exception:
        return np.asarray([], dtype=float)
    vals: list[float] = [0.0]
    grid_txt = root.findtext("./Parameters/Grid")
    if grid_txt:
        parts = [x.strip() for x in grid_txt.split(",") if x.strip()]
        if len(parts) == 2:
            try:
                n = int(float(parts[0]))
                step = _ore_tenor_years(parts[1]) or 0.0
                if n > 0 and step > 0:
                    vals.extend([i * step for i in range(1, n + 1)])
            except Exception:
                pass
    return np.asarray(sorted(set(float(x) for x in vals if x >= 0.0)), dtype=float)


def _date_from_years(asof: date, years: float) -> str:
    return str(asof + timedelta(days=max(int(round(365.25 * float(years))), 0)))


def _native_inflation_exposure_profile(payload: dict[str, Any], ore_xml: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ore_root = ET.parse(ore_xml).getroot()
    setup = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.parent
    run_dir = base.parent
    input_dir = (run_dir / setup.get("inputPath", base.name or "Input")).resolve()
    sim_cfg = ore_root.find("./Analytics/Analytic[@type='simulation']/Parameter[@name='simulationConfigFile']")
    sim_xml = (input_dir / (sim_cfg.text or "").strip()).resolve() if sim_cfg is not None else input_dir / "simulation.xml"
    exposure_times = _parse_exposure_times_from_simulation_file(sim_xml)
    if exposure_times.size == 0:
        maturity = max(float(payload["maturity_time"]), 1.0)
        exposure_times = np.linspace(0.0, maturity, num=max(int(math.ceil(maturity * 12.0)), 2))
    exposure_times = np.asarray(sorted(set(float(x) for x in exposure_times if x >= 0.0)), dtype=float)
    asof = _parse_ore_date(setup.get("asofDate", "1970-01-01"))
    exposure_dates = [_date_from_years(asof, float(t)) for t in exposure_times]

    if payload.get("inflation_kind") == "YY":
        payment_times = inflation_swap_payment_times(float(payload["maturity_time"]), str(payload.get("schedule_tenor") or "1Y"))
        npv_profile = np.asarray(
            [
                price_yoy_swap_at_time(
                    notional=float(payload["notional"]),
                    payment_times=payment_times,
                    fixed_rate=float(payload["fixed_rate"]),
                    inflation_curve=payload["inflation_curve"],
                    discount_curve=payload["p0_disc"],
                    valuation_time=float(t),
                    receive_inflation=True,
                )
                for t in exposure_times
            ],
            dtype=float,
        )
    else:
        npv_profile = np.asarray(
            [
                price_zero_coupon_cpi_swap_at_time(
                    notional=float(payload["notional"]),
                    maturity_years=float(payload["maturity_time"]),
                    fixed_rate=float(payload["fixed_rate"]),
                    base_cpi=float(payload["base_cpi"]),
                    inflation_curve=payload["inflation_curve"],
                    discount_curve=payload["p0_disc"],
                    valuation_time=float(t),
                    receive_inflation=True,
                )
                for t in exposure_times
            ],
            dtype=float,
        )
    epe = np.maximum(npv_profile, 0.0)
    ene = np.maximum(-npv_profile, 0.0)
    pfe = np.maximum(epe, ene)
    return exposure_dates, exposure_times, epe, ene, pfe


def _native_inflation_capfloor_exposure_profile(payload: dict[str, Any], ore_xml: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ore_root = ET.parse(ore_xml).getroot()
    setup = {
        n.attrib.get("name", ""): (n.text or "").strip()
        for n in ore_root.findall("./Setup/Parameter")
    }
    base = ore_xml.parent
    run_dir = base.parent
    input_dir = (run_dir / setup.get("inputPath", base.name or "Input")).resolve()
    sim_cfg = ore_root.find("./Analytics/Analytic[@type='simulation']/Parameter[@name='simulationConfigFile']")
    sim_xml = (input_dir / (sim_cfg.text or "").strip()).resolve() if sim_cfg is not None else input_dir / "simulation.xml"
    exposure_times = _parse_exposure_times_from_simulation_file(sim_xml)
    if exposure_times.size == 0:
        maturity = max(float(payload["maturity_time"]), 1.0)
        exposure_times = np.linspace(0.0, maturity, num=max(int(math.ceil(maturity * 12.0)), 2))
    exposure_times = np.asarray(sorted(set(float(x) for x in exposure_times if x >= 0.0)), dtype=float)
    asof = _parse_ore_date(setup.get("asofDate", "1970-01-01"))
    exposure_dates = [_date_from_years(asof, float(t)) for t in exposure_times]
    definition = InflationCapFloorDefinition(
        trade_id=str(payload["trade_id"]),
        currency=str(payload["currency"]),
        inflation_type=str(payload["inflation_kind"]),
        option_type=str(payload["option_type"]),
        index=str(payload["index"]),
        strike=float(payload["strike"]),
        notional=float(payload["notional"]),
        maturity_years=float(payload["maturity_time"]),
        base_cpi=float(payload["base_cpi"]) if payload.get("base_cpi") is not None else None,
        observation_lag=payload.get("observation_lag"),
        long_short="Long",
    )
    npv_profile = np.asarray(
        [
            price_inflation_capfloor_at_time(
                definition=definition,
                inflation_curve=payload["inflation_curve"],
                discount_curve=payload["p0_disc"],
                valuation_time=float(t),
                market_surface_price=payload.get("market_surface_price"),
            )
            for t in exposure_times
        ],
        dtype=float,
    )
    epe = np.maximum(npv_profile, 0.0)
    ene = np.maximum(-npv_profile, 0.0)
    pfe = np.maximum(epe, ene)
    return exposure_dates, exposure_times, epe, ene, pfe


def _supports_native_swaption_price_only(ore_xml: Path) -> bool:
    try:
        portfolio_xml = _resolve_case_portfolio_path(ore_xml)
        if portfolio_xml is None or not portfolio_xml.exists():
            return False
        root = ET.parse(portfolio_xml).getroot()
        trade_id = ore_snapshot_mod._get_first_trade_id(root)
        trade = next((t for t in root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
        if trade is None:
            trade = root.find("./Trade") or root.find(".//Trade")
        if trade is None or (trade.findtext("./TradeType") or "").strip() != "Swaption":
            return False
        style = (trade.findtext("./SwaptionData/OptionData/Style") or "").strip().lower()
        if style in {"bermudan", "american"}:
            return True
        if style != "european":
            return False
        ore_root = ET.parse(ore_xml).getroot()
        setup = {n.attrib.get("name", ""): (n.text or "").strip() for n in ore_root.findall("./Setup/Parameter")}
        asof_text = setup.get("asofDate", "")
        if not asof_text:
            return False
        asof = _parse_ore_date(asof_text)
        base = ore_xml.parent
        run_dir = base.parent
        input_dir = (run_dir / setup.get("inputPath", base.name or "Input")).resolve()
        curve_config_path = (input_dir / setup.get("curveConfigFile", "")).resolve()
        todaysmarket_xml = (input_dir / setup.get("marketConfigFile", "../../Input/todaysmarket.xml")).resolve()
        legs = trade.findall("./SwaptionData/LegData")
        fixed_leg = next((l for l in legs if (l.findtext("./LegType") or "").strip().lower() == "fixed"), None)
        if fixed_leg is None:
            return False
        ccy = (fixed_leg.findtext("./Currency") or "EUR").strip().upper()
        spec = _load_swaption_surface_spec(todaysmarket_xml, curve_config_path, ccy)
        max_swap_term = max((_ore_tenor_years(x) or 0.0) for x in spec.get("swap_tenors", []))
        if max_swap_term <= 0.0:
            return False
        start_text = (fixed_leg.findtext("./ScheduleData/Rules/StartDate") or "").strip()
        end_text = (fixed_leg.findtext("./ScheduleData/Rules/EndDate") or "").strip()
        if not start_text or not end_text:
            return False
        start_date = _parse_ore_date(start_text) if "-" in start_text else date.fromisoformat(f"{start_text[:4]}-{start_text[4:6]}-{start_text[6:8]}")
        end_date = _parse_ore_date(end_text) if "-" in end_text else date.fromisoformat(f"{end_text[:4]}-{end_text[4:6]}-{end_text[6:8]}")
        swap_term = ore_snapshot_mod._year_fraction_from_day_counter(
            start_date,
            end_date,
            "A365F",
        )
        return float(swap_term) <= float(max_swap_term) + 0.5
    except Exception:
        return False


def _supports_native_capfloor_price_only(ore_xml: Path) -> bool:
    try:
        portfolio_xml = _resolve_case_portfolio_path(ore_xml)
        if portfolio_xml is None or not portfolio_xml.exists():
            return False
        portfolio_root = ET.parse(portfolio_xml).getroot()
        trade_id = ore_snapshot_mod._get_first_trade_id(portfolio_root)
        trade = next((t for t in portfolio_root.findall("./Trade") if (t.attrib.get("id", "") or "").strip() == trade_id), None)
        if trade is None or (trade.findtext("./TradeType") or "").strip() != "CapFloor":
            return False
        leg_type = (trade.findtext("./CapFloorData/LegData/LegType") or "").strip().lower()
        return leg_type in {"floating", "ibor", "yy", "cpi"}
    except Exception:
        return False


def _supports_native_price_only(first_trade_type: str, ore_xml: Path) -> bool:
    trade_type = str(first_trade_type or "").strip()
    if trade_type in {
        "Bond",
        "ForwardBond",
        "CallableBond",
        "FxForward",
        "FxOption",
        "EquityOption",
        "EquityForward",
        "EquitySwap",
        "ScriptedTrade",
    }:
        return True
    if trade_type == "CapFloor":
        return _supports_native_capfloor_price_only(ore_xml)
    if trade_type == "Swaption":
        return _supports_native_swaption_price_only(ore_xml)
    if trade_type == "Swap":
        return _is_plain_vanilla_swap_trade(ore_xml) or _is_inflation_swap_trade(ore_xml)
    return False


def _supports_native_python_no_reference(first_trade_type: str, ore_xml: Path) -> bool:
    trade_type = str(first_trade_type or "").strip()
    if _supports_native_price_only(trade_type, ore_xml):
        return True
    if trade_type == "Swap" and _has_active_simulation_analytic(ore_xml):
        return True
    return False


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


def _is_reference_fallback_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        isinstance(exc, FileNotFoundError)
        or (
            isinstance(exc, ImportError)
            and "QuantLib Python bindings are required" in message
        )
        or "FloatingLegData/Index not found" in message
        or "no equity smile quotes found" in message
        or "spot quote" in message
        or "no LGM node found for ccy" in message
        or "no fitted market curve available for currency" in message
        or "no fitted market instruments available for currency" in message
        or "has no Configuration[@id='" in message
        or "has no DiscountingCurves mapping for config" in message
        or "has no DiscountingCurve[@currency='" in message
        or "Could not resolve column name for curve handle" in message
        or "uses cross-currency segments and curves.csv is unavailable" in message
        or "discount_columns must contain at least one column name" in message
        or ("Trade '" in message and "' not found in " in message and "npv.csv" in message)
        or "Unsupported tenor '*'" in message
    )


def _default_case_identity(ore_xml: Path) -> tuple[str, str, str]:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    if portfolio_xml is None or not portfolio_xml.exists():
        return "", "", ""
    try:
        return _find_first_trade_context(portfolio_xml)
    except ValueError as exc:
        if "no Trade node found" in str(exc):
            return "", "", ""
        raise


def _ore_reference_summary(
    ore_xml: Path,
    modes: ModeSelection,
    *,
    allow_partial_reference: bool = False,
) -> dict[str, Any]:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    validation = validate_ore_input_snapshot(ore_xml, requested_modes=[name for name, enabled in asdict(modes).items() if enabled])
    output_dir = _resolve_case_output_dir(ore_xml)
    trade_id, counterparty, netting_set_id = _default_case_identity(ore_xml)
    reference_dirs = _reference_output_dirs(ore_xml)
    npv_csv = _find_reference_npv_file(ore_xml, trade_id)
    xva_csv = _find_reference_output_file(ore_xml, "xva.csv")
    case_summary: dict[str, Any] = {
        "ore_xml": str(ore_xml),
        "portfolio_xml": str(portfolio_xml) if portfolio_xml is not None else "",
        "modes": [name for name, enabled in asdict(modes).items() if enabled],
        "trade_id": trade_id,
        "counterparty": counterparty,
        "netting_set_id": netting_set_id,
        "pricing": None,
        "xva": None,
        "parity": None,
        "diagnostics": {
            "engine": "ore_reference",
            "reference_output_dirs": [str(p) for p in reference_dirs],
            "using_expected_output": output_dir not in reference_dirs or not output_dir.exists(),
        },
        "input_validation": validation,
        "pass_flags": {},
        "pass_all": True,
    }
    if modes.price:
        if npv_csv is None and not allow_partial_reference:
            raise FileNotFoundError(f"ORE output file not found (run ORE first): {output_dir / 'npv.csv'}")
        if npv_csv is not None:
            try:
                npv_details = ore_snapshot_mod._load_ore_npv_details(npv_csv, trade_id=trade_id)
                case_summary["maturity_date"] = str(npv_details["maturity_date"])
                case_summary["maturity_time"] = float(npv_details["maturity_time"])
                case_summary["pricing"] = {"ore_t0_npv": float(npv_details["npv"])}
            except Exception:
                if not allow_partial_reference:
                    raise
                case_summary["diagnostics"]["missing_reference_pricing"] = True
                case_summary["maturity_date"] = ""
                case_summary["maturity_time"] = 0.0
                case_summary["pricing"] = None
        else:
            case_summary["diagnostics"]["missing_reference_pricing"] = True
            case_summary["maturity_date"] = ""
            case_summary["maturity_time"] = 0.0
            case_summary["pricing"] = None
    else:
        case_summary["maturity_date"] = ""
        case_summary["maturity_time"] = 0.0
    if modes.xva:
        if xva_csv is None and not allow_partial_reference:
            raise FileNotFoundError(f"ORE output file not found (run ORE first): {output_dir / 'xva.csv'}")
        if xva_csv is not None:
            try:
                xva_row, used_trade_row = _load_xva_reference_row(
                    xva_csv,
                    trade_id=trade_id,
                    netting_set_id=netting_set_id,
                )
                case_summary["xva"] = {
                    "ore_cva": float(xva_row["cva"]),
                    "ore_dva": float(xva_row["dva"]),
                    "ore_fba": float(xva_row["fba"]),
                    "ore_fca": float(xva_row["fca"]),
                    "ore_basel_epe": float(xva_row["basel_epe"]),
                    "ore_basel_eepe": float(xva_row["basel_eepe"]),
                }
                case_summary["diagnostics"]["xva_reference_row"] = "trade" if used_trade_row else "aggregate"
            except Exception:
                if not allow_partial_reference:
                    raise
                case_summary["diagnostics"]["missing_reference_xva"] = True
                case_summary["xva"] = None
        else:
            case_summary["diagnostics"]["missing_reference_xva"] = True
            case_summary["xva"] = None
        exposure_path = _find_reference_output_file(ore_xml, f"exposure_trade_{trade_id}.csv")
        if exposure_path is not None:
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

        for param in setup.findall("Parameter"):
            name = (param.attrib.get("name", "") or "").strip()
            text = (param.text or "").strip()
            if not name:
                continue
            if name == "inputPath":
                param.text = str(input_dir)
                continue
            if name == "outputPath":
                param.text = str(output_dir)
                continue
            if not text:
                continue
            basename = Path(text).name
            if (name.endswith("File") or name == "calendarAdjustment") and basename in input_files:
                param.text = str(input_dir / basename)
            elif name in {
                "logFile",
                "outputFile",
                "outputFileName",
                "cubeFile",
                "aggregationScenarioDataFileName",
                "scenarioFile",
                "rawCubeOutputFile",
                "netCubeOutputFile",
            }:
                param.text = basename

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
        skip_input_validation=False,
        paths=None if options.paths is None else int(options.paths),
        seed=int(options.seed),
        rng=options.rng,
        xva_mode=options.xva_mode,
        lgm_param_source=options.lgm_param_source,
        anchor_t0_npv=options.anchor_t0_npv,
        own_hazard=float(options.own_hazard),
        own_recovery=float(options.own_recovery),
        netting_set=options.netting_set,
        sensi_metric=options.sensi_metric,
        sensi_progress=False,
        top=int(options.top),
        max_npv_abs_diff=float(options.max_npv_abs_diff),
        max_cva_rel=float(options.max_cva_rel),
        max_dva_rel=float(options.max_dva_rel),
        max_fba_rel=float(options.max_fba_rel),
        max_fca_rel=float(options.max_fca_rel),
        trade_id=None,
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


def _report_context_from_case_summary(case_summary: dict[str, Any]) -> dict[str, Any]:
    ore_xml = Path(case_summary.get("ore_xml", "."))
    base_currency = ""
    trade_count = 0
    total_notional = 0.0
    notional_ccy = ""
    portfolio_xml = _resolve_case_portfolio_path(ore_xml) if ore_xml.is_file() else None
    if portfolio_xml is not None and portfolio_xml.exists():
        try:
            portfolio_root = ET.parse(portfolio_xml).getroot()
            trades = list(portfolio_root.findall("./Trade"))
            trade_count = len(trades)
            notionals: list[float] = []
            currencies: list[str] = []
            for trade in trades:
                for leg in trade.findall(".//LegData"):
                    ccy = (leg.findtext("./Currency") or "").strip().upper()
                    notional_text = (leg.findtext("./Notionals/Notional") or "").strip()
                    if ccy:
                        currencies.append(ccy)
                    if notional_text:
                        try:
                            notionals.append(abs(float(notional_text)))
                        except ValueError:
                            pass
                    break
            total_notional = float(sum(notionals)) if notionals else 0.0
            unique_ccys = sorted({ccy for ccy in currencies if ccy})
            if len(unique_ccys) == 1:
                notional_ccy = unique_ccys[0]
        except Exception:
            pass
    if ore_xml.is_file():
        try:
            ore_root = ET.parse(ore_xml).getroot()
            base_currency = (
                (ore_root.findtext("./Setup/Parameter[@name='baseCurrency']") or "").strip().upper()
                or (ore_root.findtext("./Analytics/Analytic[@type='npv']/Parameter[@name='baseCurrency']") or "").strip().upper()
            )
        except Exception:
            pass
    pricing = case_summary.get("pricing") or {}
    report_ccy = str(
        pricing.get("currency")
        or pricing.get("report_ccy")
        or base_currency
        or "USD"
    ).upper()
    entity_id = str(case_summary.get("trade_id", "") or "")
    entity_type = "Trade"
    if trade_count > 1:
        entity_id = "PORTFOLIO"
        entity_type = "Portfolio"
    elif not entity_id:
        entity_id = "PORTFOLIO"
        entity_type = "Portfolio"
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "trade_count": trade_count,
        "report_ccy": report_ccy,
        "base_currency": base_currency or report_ccy,
        "total_notional": total_notional,
        "notional_ccy": notional_ccy or report_ccy,
    }


def _write_ore_compatible_reports(case_out_dir: Path, case_summary: dict[str, Any]) -> None:
    pricing = case_summary.get("pricing") or {}
    xva = case_summary.get("xva") or {}
    sensi = case_summary.get("sensitivity") or {}
    trade_profile = dict(case_summary.get("exposure_profile_by_trade") or {})
    netting_profile = dict(case_summary.get("exposure_profile_by_netting_set") or {})
    exposure_dates = list(trade_profile.get("dates") or case_summary.get("exposure_dates") or [])
    exposure_times = list(trade_profile.get("times") or case_summary.get("exposure_times") or [])
    py_epe = list(trade_profile.get("closeout_epe") or case_summary.get("py_epe") or [])
    py_ene = list(trade_profile.get("closeout_ene") or case_summary.get("py_ene") or [])
    py_pfe = list(trade_profile.get("pfe") or case_summary.get("py_pfe") or [])
    netting_set_id = str(case_summary.get("netting_set_id", ""))
    counterparty = str(case_summary.get("counterparty", ""))
    report_ctx = _report_context_from_case_summary(case_summary)
    entity_id = str(report_ctx["entity_id"])
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
            entity_id,
            str(pricing.get("trade_type", report_ctx["entity_type"])),
            maturity_date,
            _fmt_float(maturity_time),
            _fmt_float(npv_value),
            str(report_ctx["report_ccy"]),
            _fmt_float(npv_value),
            str(report_ctx["base_currency"]),
            _fmt_float(float(report_ctx["total_notional"]), digits=2),
            str(report_ctx["notional_ccy"]),
            _fmt_float(float(report_ctx["total_notional"]), digits=2),
            netting_set_id,
            counterparty,
        ]
        with open(case_out_dir / "npv.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(npv_headers)
            writer.writerow(npv_row)
    if sensi and str(sensi.get("metric", "")).strip().upper() in {"NPV", "PV"}:
        sensitivity_rows = list(sensi.get("python_rows") or [])
        scenario_rows = list(sensi.get("scenario_rows") or [])
        sensitivity_filename = str(sensi.get("sensitivity_output_file") or "sensitivity.csv")
        scenario_filename = str(sensi.get("scenario_output_file") or "scenario.csv")
        if sensitivity_rows:
            sensitivity_headers = [
                "#TradeId", "IsPar", "Factor_1", "ShiftSize_1", "Factor_2", "ShiftSize_2",
                "Currency", "Base NPV", "Delta", "Gamma",
            ]
            with open(case_out_dir / sensitivity_filename, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(sensitivity_headers)
                for row in sensitivity_rows:
                    writer.writerow(
                        [
                            entity_id,
                            "false",
                            str(row.get("ore_factor") or row.get("normalized_factor") or ""),
                            _fmt_float(float(row.get("shift_size") or 0.0)),
                            "",
                            _fmt_float(0.0),
                            str(report_ctx["report_ccy"]),
                            _fmt_float(float(row.get("base_metric_value") or 0.0), digits=2),
                            _fmt_float(float(row.get("delta") or 0.0), digits=2),
                            _fmt_float(0.0, digits=2),
                        ]
                    )
        if scenario_rows:
            scenario_headers = [
                "#TradeId", "Factor", "Up/Down", "Base NPV", "ShiftSize_1",
                "ShiftSize_2", "Scenario NPV", "Difference",
            ]
            with open(case_out_dir / scenario_filename, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(scenario_headers)
                for row in scenario_rows:
                    shift_size_2 = row.get("shift_size_2", "#N/A")
                    writer.writerow(
                        [
                            entity_id,
                            str(row.get("factor") or ""),
                            str(row.get("direction") or ""),
                            _fmt_float(float(row.get("base_metric_value") or 0.0), digits=2),
                            _fmt_float(float(row.get("shift_size_1") or 0.0)),
                            shift_size_2 if isinstance(shift_size_2, str) else _fmt_float(float(shift_size_2)),
                            _fmt_float(float(row.get("scenario_metric_value") or 0.0), digits=2),
                            _fmt_float(float(row.get("difference") or 0.0), digits=2),
                        ]
                    )
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
            f"{_one_year_profile_value(netting_profile, 'time_weighted_basel_epe') or xva.get('py_basel_epe', 0.0):.2f}",
            f"{_one_year_profile_value(netting_profile, 'time_weighted_basel_eepe') or xva.get('py_basel_eepe', 0.0):.2f}",
        ]
        trade_row = [
            entity_id,
            netting_set_id,
            f"{xva.get('py_cva', 0.0):.2f}",
            f"{xva.get('py_dva', 0.0):.2f}",
            f"{xva.get('py_fba', 0.0):.2f}",
            f"{xva.get('py_fca', 0.0):.2f}",
            "0.00", "0.00", "0.00", "0.00", "#N/A", "#N/A", "#N/A", "#N/A", "#N/A", "#N/A", "#N/A",
            "0.00",
            "0.00",
            "None",
            f"{_one_year_profile_value(trade_profile, 'time_weighted_basel_epe') or xva.get('py_basel_epe', 0.0):.2f}",
            f"{_one_year_profile_value(trade_profile, 'time_weighted_basel_eepe') or xva.get('py_basel_eepe', 0.0):.2f}",
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
        trade_basel_ee = list(trade_profile.get("basel_ee") or py_epe)
        trade_basel_eee = list(trade_profile.get("basel_eee") or py_epe)
        trade_tw_epe = list(trade_profile.get("time_weighted_basel_epe") or py_epe)
        trade_tw_eepe = list(trade_profile.get("time_weighted_basel_eepe") or py_epe)
        netting_epe = list(netting_profile.get("closeout_epe") or py_epe)
        netting_ene = list(netting_profile.get("closeout_ene") or py_ene)
        netting_pfe = list(netting_profile.get("pfe") or py_pfe)
        netting_expected_collateral = list(netting_profile.get("expected_collateral") or [])
        netting_basel_ee = list(netting_profile.get("basel_ee") or netting_epe)
        netting_basel_eee = list(netting_profile.get("basel_eee") or netting_epe)
        netting_tw_epe = list(netting_profile.get("time_weighted_basel_epe") or netting_epe)
        netting_tw_eepe = list(netting_profile.get("time_weighted_basel_eepe") or netting_epe)
        for i, (d, t, epe, ene, pfe) in enumerate(zip(exposure_dates, exposure_times, py_epe, py_ene, py_pfe)):
            t = float(t)
            epe = float(epe)
            ene = float(ene)
            pfe = float(pfe)
            trade_rows.append([
                entity_id,
                d,
                _fmt_float(t),
                f"{epe:.0f}",
                f"{ene:.0f}",
                "0",
                "0",
                f"{pfe:.0f}",
                f"{float(trade_basel_ee[i]):.0f}",
                f"{float(trade_basel_eee[i]):.0f}",
                f"{float(trade_tw_epe[i]):.2f}",
                f"{float(trade_tw_eepe[i]):.2f}",
            ])
            ns_rows.append([
                netting_set_id,
                d,
                _fmt_float(t),
                f"{float(netting_epe[i]):.2f}",
                f"{float(netting_ene[i]):.2f}",
                f"{float(netting_pfe[i]):.2f}",
                f"{float(netting_expected_collateral[i]):.2f}" if i < len(netting_expected_collateral) else ("0.00" if i else f"{float(netting_epe[i]):.2f}"),
                f"{float(netting_basel_ee[i]):.2f}",
                f"{float(netting_basel_eee[i]):.2f}",
                f"{float(netting_tw_epe[i]):.2f}",
                f"{float(netting_tw_eepe[i]):.2f}",
            ])
        with open(case_out_dir / f"exposure_trade_{entity_id}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(trade_headers)
            writer.writerows(trade_rows)
        with open(case_out_dir / f"exposure_nettingset_{netting_set_id}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(ns_headers)
            writer.writerows(ns_rows)


def _copy_native_ore_reports(ore_xml: Path, case_out_dir: Path) -> None:
    for source_dir in _reference_output_dirs(ore_xml):
        if not source_dir.exists():
            continue
        for src in source_dir.iterdir():
            if not src.is_file():
                continue
            dst = case_out_dir / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)


def benchmark_pfe_profile_vs_ore(
    ore_xml: str | Path,
    *,
    paths: int | None = None,
    seeds: Sequence[int] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    def _load_ore_trade_exposure_csv(path: Path) -> dict[str, list[Any]]:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            out = {"date": [], "time": [], "epe": [], "pfe": []}
            for row in reader:
                out["date"].append(str(row.get("Date") or ""))
                out["time"].append(float(row.get("Time") or 0.0))
                out["epe"].append(float(row.get("EPE") or 0.0))
                out["pfe"].append(float(row.get("PFE") or 0.0))
        return out

    ore_xml = Path(ore_xml).resolve()
    snap = load_from_ore_xml(ore_xml, anchor_t0_npv=False)
    reference_seed = int(getattr(snap, "seed", 42))
    benchmark_seeds = [reference_seed] if seeds is None else [int(s) for s in seeds]
    exposure_csv = _find_reference_output_file(ore_xml, f"exposure_trade_{snap.trade_id}.csv")
    if exposure_csv is None:
        raise FileNotFoundError(f"reference exposure report not found for trade '{snap.trade_id}'")
    if exposure_csv.exists():
        ore_profile = _load_ore_trade_exposure_csv(exposure_csv)
    else:
        fallback_profile = load_ore_exposure_profile(str(exposure_csv))
        ore_profile = {
            "date": list(fallback_profile.get("date", [])),
            "time": list(fallback_profile.get("time", [])),
            "epe": list(fallback_profile.get("epe", [])),
            "pfe": list(fallback_profile.get("pfe", fallback_profile.get("epe", []))),
        }
    ore_times = np.asarray(ore_profile["time"], dtype=float)
    ore_pfe = np.asarray(ore_profile["pfe"], dtype=float)
    ore_epe = np.asarray(ore_profile["epe"], dtype=float)

    runs: list[dict[str, Any]] = []
    pfe_rows: list[np.ndarray] = []
    epe_rows: list[np.ndarray] = []
    for seed in benchmark_seeds:
        result = _compute_snapshot_case(
            ore_xml,
            paths=paths,
            seed=int(seed),
            rng_mode="ore_parity",
            anchor_t0_npv=False,
            own_hazard=0.01,
            own_recovery=0.4,
            xva_mode="ore",
        )
        trade_profile = result.exposure_profile_by_trade
        py_times = np.asarray(trade_profile.get("times", []), dtype=float)
        py_pfe = np.asarray(trade_profile.get("pfe", []), dtype=float)
        py_epe = np.asarray(
            trade_profile.get("closeout_epe", result.py_epe if hasattr(result, "py_epe") else []),
            dtype=float,
        )
        if py_times.shape != ore_times.shape or not np.allclose(py_times, ore_times, atol=1.0e-10, rtol=0.0):
            raise ValueError("Python and ORE exposure grids are not aligned for PFE benchmark")
        pfe_rows.append(py_pfe)
        epe_rows.append(py_epe)
        runs.append(
            {
                "seed": int(seed),
                "paths": int(result.paths),
                "one_year_pfe": float(_shared_one_year_profile_value(trade_profile, "pfe", asof_date=ore_profile["date"][0])),
                "one_year_epe": float(
                    _shared_one_year_profile_value(trade_profile, "closeout_epe", asof_date=ore_profile["date"][0])
                ),
                "peak_pfe": float(np.max(py_pfe)) if py_pfe.size else 0.0,
            }
        )

    py_pfe_runs = np.asarray(pfe_rows, dtype=float)
    py_epe_runs = np.asarray(epe_rows, dtype=float)
    py_pfe_mean = np.mean(py_pfe_runs, axis=0)
    py_pfe_sigma = np.std(py_pfe_runs, axis=0, ddof=1 if py_pfe_runs.shape[0] > 1 else 0)
    py_epe_mean = np.mean(py_epe_runs, axis=0)
    py_epe_sigma = np.std(py_epe_runs, axis=0, ddof=1 if py_epe_runs.shape[0] > 1 else 0)
    abs_diff = np.abs(py_pfe_mean - ore_pfe)
    rel_diff = np.zeros_like(abs_diff)
    nonzero_ref = np.abs(ore_pfe) > 1.0e-12
    rel_diff[nonzero_ref] = abs_diff[nonzero_ref] / np.abs(ore_pfe[nonzero_ref])
    epe_abs_diff = np.abs(py_epe_mean - ore_epe)
    epe_rel_diff = np.zeros_like(epe_abs_diff)
    nonzero_epe_ref = np.abs(ore_epe) > 1.0e-12
    epe_rel_diff[nonzero_epe_ref] = epe_abs_diff[nonzero_epe_ref] / np.abs(ore_epe[nonzero_epe_ref])
    sigma_multiple = np.full_like(abs_diff, np.inf)
    nonzero_sigma = py_pfe_sigma > 1.0e-12
    sigma_multiple[nonzero_sigma] = abs_diff[nonzero_sigma] / py_pfe_sigma[nonzero_sigma]
    sigma_multiple[~nonzero_sigma & (abs_diff <= 1.0e-12)] = 0.0
    economically_relevant = (ore_times > 1.0e-10) & ((np.abs(ore_pfe) > 1.0e-8) | (abs_diff > 1.0e-8))
    if not np.any(economically_relevant):
        economically_relevant = np.ones_like(ore_times, dtype=bool)
    one_year_idx = min(
        max(0, int(np.searchsorted(ore_times, 1.0, side="left"))),
        max(ore_times.size - 1, 0),
    )
    peak_idx = int(np.argmax(ore_pfe)) if ore_pfe.size else 0
    within_two_sigma = sigma_multiple[economically_relevant] <= 2.0
    finite_sigma = sigma_multiple[economically_relevant][np.isfinite(sigma_multiple[economically_relevant])]
    max_sigma = float(np.max(finite_sigma)) if finite_sigma.size else 0.0
    summary = {
        "trade_id": snap.trade_id,
        "netting_set_id": snap.netting_set_id,
        "paths": int(paths if paths is not None else snap.n_samples),
        "ore_reference_seed": reference_seed,
        "seeds": benchmark_seeds,
        "uses_reference_seed_only": len(benchmark_seeds) == 1 and benchmark_seeds[0] == reference_seed,
        "grid_points": int(ore_times.size),
        "relevant_points": int(np.sum(economically_relevant)),
        "ignored_points": int(ore_times.size - np.sum(economically_relevant)),
        "one_year_index": int(one_year_idx),
        "peak_index": int(peak_idx),
        "one_year_sigma_multiple": float(sigma_multiple[one_year_idx]) if sigma_multiple.size else 0.0,
        "peak_sigma_multiple": float(sigma_multiple[peak_idx]) if sigma_multiple.size else 0.0,
        "within_two_sigma_ratio": float(np.mean(within_two_sigma)) if within_two_sigma.size else 1.0,
        "max_sigma_multiple": max_sigma,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "median_rel_diff": float(np.median(rel_diff)) if rel_diff.size else 0.0,
        "p95_rel_diff": float(np.percentile(rel_diff, 95.0)) if rel_diff.size else 0.0,
        "mean_abs_epe_diff": float(np.mean(epe_abs_diff)) if epe_abs_diff.size else 0.0,
        "median_rel_epe_diff": float(np.median(epe_rel_diff)) if epe_rel_diff.size else 0.0,
        "p95_rel_epe_diff": float(np.percentile(epe_rel_diff, 95.0)) if epe_rel_diff.size else 0.0,
        "accepted": bool(
            np.any(economically_relevant)
            and float(np.mean(within_two_sigma)) >= 0.95
            and float(sigma_multiple[one_year_idx]) <= 2.0
            and float(sigma_multiple[peak_idx]) <= 2.0
            and max_sigma <= 3.0
        ),
    }
    pointwise_rows = [
        {
            "Date": str(ore_profile["date"][i]),
            "Time": float(ore_times[i]),
            "OreEPE": float(ore_epe[i]),
            "OrePFE": float(ore_pfe[i]),
            "PythonMeanEPE": float(py_epe_mean[i]),
            "PythonSigmaEPE": float(py_epe_sigma[i]),
            "PythonMeanPFE": float(py_pfe_mean[i]),
            "PythonSigmaPFE": float(py_pfe_sigma[i]),
            "EPEAbsDiff": float(epe_abs_diff[i]),
            "EPERelDiff": float(epe_rel_diff[i]),
            "AbsDiff": float(abs_diff[i]),
            "RelDiff": float(rel_diff[i]),
            "SigmaMultiple": float(sigma_multiple[i]),
        }
        for i in range(ore_times.size)
    ]
    result = {
        "summary": summary,
        "runs": runs,
        "pointwise": pointwise_rows,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "pfe_benchmark.json", result)
        _write_csv(out / "pfe_benchmark.csv", pointwise_rows)
    return result


def _render_case_markdown(case_summary: dict[str, Any]) -> str:
    lines = [
        "# ORE Snapshot CLI Report",
        "",
        f"- ore_xml: `{case_summary['ore_xml']}`",
        f"- portfolio_xml: `{case_summary.get('portfolio_xml', '')}`",
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
            if key in {"top_comparisons", "python_rows", "scenario_rows"}:
                continue
            lines.append(f"- {key}: `{value}`")
    validation = case_summary.get("input_validation")
    if validation:
        lines.extend(["", "## Input Validation", ""])
        lines.append(f"- input_links_valid: `{validation.get('input_links_valid')}`")
        for issue in validation.get("issues", []):
            lines.append(f"- issue: `{issue}`")
    return "\n".join(lines) + "\n"


def _run_case_in_subprocess(ore_xml: Path, args: argparse.Namespace, *, artifact_root: Path) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pythonore.workflows.ore_snapshot_cli",
        str(ore_xml),
        "--output-root",
        str(artifact_root),
        "--paths",
        str(int(getattr(args, "paths", 2000))),
        "--rng",
        str(getattr(args, "rng", "ore_sobol_bridge")),
        "--xva-mode",
        str(getattr(args, "xva_mode", "ore")),
        "--lgm-param-source",
        str(getattr(args, "lgm_param_source", "auto")),
    ]
    if bool(getattr(args, "price", False)):
        cmd.append("--price")
    if bool(getattr(args, "xva", False)):
        cmd.append("--xva")
    if bool(getattr(args, "sensi", False)):
        cmd.append("--sensi")
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = "src:." if not current_pythonpath else f"src:.:{current_pythonpath}"
    subprocess.run(
        cmd,
        check=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    summary_path = artifact_root / _case_slug(ore_xml) / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _unique_report_case_slug(ore_xml: Path) -> str:
    examples_root = _examples_root().resolve()
    resolved = ore_xml.resolve()
    try:
        rel = resolved.relative_to(examples_root)
    except ValueError:
        rel = resolved
    return "__".join(rel.parts[:-1] + (resolved.stem,))


def _bucket_case(case_summary: dict[str, Any]) -> str:
    diagnostics = dict(case_summary.get("diagnostics") or {})
    input_validation = case_summary.get("input_validation") or {}
    pass_all = bool(case_summary.get("pass_all"))
    ore_xml_path = Path(case_summary.get("ore_xml", "."))
    reference_dirs = [Path(p) for p in diagnostics.get("reference_output_dirs", []) if p]
    reference_kinds = {_classify_reference_dir(ore_xml_path, p) for p in reference_dirs if p.exists()}
    if diagnostics.get("error"):
        return "hard_error"
    first_trade_type = _first_trade_type(ore_xml_path) if ore_xml_path.is_file() else ""
    if (
        pass_all
        and diagnostics.get("fallback_reason") == "missing_native_output"
        and first_trade_type in {"FlexiSwap", "ScriptedTrade"}
    ):
        return "unsupported_python_snapshot_fallback"
    if diagnostics.get("pricing_fallback_reason") == "missing_simulation_analytic":
        if not _supports_native_price_only(first_trade_type, ore_xml_path):
            return "unsupported_python_snapshot_fallback"
        return "price_only_reference_fallback"
    if diagnostics.get("fallback_reason") == "missing_native_output" and pass_all and not reference_kinds:
        if _supports_native_python_no_reference(first_trade_type, ore_xml_path):
            return "python_only_no_reference"
        return "no_reference_artifacts_pass"
    if diagnostics.get("fallback_reason") == "missing_native_output" and pass_all and (
        "expected_output" in reference_kinds
    ):
        return "expected_output_fallback_pass"
    if diagnostics.get("fallback_reason") == "missing_native_output":
        return "missing_native_output_fallback"
    if diagnostics.get("fallback_reason") == "unsupported_python_snapshot":
        return "unsupported_python_snapshot_fallback"
    if diagnostics.get("missing_reference_pricing"):
        return "missing_reference_pricing"
    if diagnostics.get("missing_reference_xva") and not pass_all:
        return "missing_reference_xva"
    if not pass_all and diagnostics.get("sample_count_mismatch"):
        return "sample_count_mismatch"
    pricing = case_summary.get("pricing") or {}
    xva = case_summary.get("xva") or {}
    has_explicit_parity_signal = any(
        pricing.get(key) is not None for key in ("t0_npv_abs_diff",)
    ) or any(
        xva.get(key) is not None for key in ("cva_rel_diff", "dva_rel_diff", "fba_rel_diff", "fca_rel_diff")
    )
    if not pass_all and input_validation.get("input_links_valid") is False and not has_explicit_parity_signal:
        return "input_validation_issue"
    if not pass_all:
        return "parity_threshold_fail"
    return "clean_pass"


def _report_row_from_case_summary(ore_xml: Path, case_summary: dict[str, Any], rc: int, *, summary_path: Path) -> dict[str, Any]:
    pricing = case_summary.get("pricing") or {}
    xva = case_summary.get("xva") or {}
    diagnostics = case_summary.get("diagnostics") or {}
    parity = case_summary.get("parity") or {}
    bucket = _bucket_case(case_summary)
    return {
        "ore_xml": str(ore_xml),
        "case_slug": _unique_report_case_slug(ore_xml),
        "rc": int(rc),
        "pass_all": bool(case_summary.get("pass_all")),
        "trade_id": str(case_summary.get("trade_id", "")),
        "modes": ",".join(case_summary.get("modes") or []),
        "engine": diagnostics.get("engine"),
        "reference_source_used": _reference_source_used(ore_xml, case_summary),
        "fallback_reason": diagnostics.get("fallback_reason"),
        "pricing_fallback_reason": diagnostics.get("pricing_fallback_reason"),
        "sample_count_mismatch": diagnostics.get("sample_count_mismatch"),
        "pricing_abs_diff": pricing.get("t0_npv_abs_diff"),
        "cva_rel_diff": xva.get("cva_rel_diff"),
        "dva_rel_diff": xva.get("dva_rel_diff"),
        "fba_rel_diff": xva.get("fba_rel_diff"),
        "fca_rel_diff": xva.get("fca_rel_diff"),
        "parity_ready": parity.get("parity_ready"),
        "bucket": bucket,
        "next_fix_hint": REPORT_BUCKET_HINTS[bucket],
        "summary_path": str(summary_path),
    }


def _render_live_report_markdown(rows: list[dict[str, Any]], *, total_cases: int, top_buckets: int) -> str:
    bucket_counts = Counter(row["bucket"] for row in rows)
    engine_counts = Counter(str(row["engine"]) for row in rows if row.get("engine"))
    reference_counts = Counter(str(row["reference_source_used"]) for row in rows if row.get("reference_source_used"))
    clean_count = sum(1 for row in rows if row["bucket"] == "clean_pass")
    non_clean = [row for row in rows if row["bucket"] != "clean_pass"]
    lines = [
        "# ORE Snapshot CLI Live Parity Report",
        "",
        f"- completed: `{len(rows)}` / `{total_cases}`",
        f"- clean_pass: `{clean_count}`",
        f"- non_clean: `{len(non_clean)}`",
        "",
        "## Top Buckets",
        "",
    ]
    for bucket, count in bucket_counts.most_common(top_buckets):
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(["", "## Reference Sources", ""])
    for key, count in sorted(reference_counts.items()):
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Engines", ""])
    for key, count in sorted(engine_counts.items()):
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Actionable Cases", ""])
    if not non_clean:
        lines.append("- all completed cases are clean")
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in non_clean:
            grouped.setdefault(str(row["bucket"]), []).append(row)
        for bucket, count in bucket_counts.most_common(top_buckets):
            if bucket == "clean_pass":
                continue
            rows_for_bucket = grouped.get(bucket, [])
            if not rows_for_bucket:
                continue
            lines.append(f"### `{bucket}`")
            lines.append("")
            lines.append(f"- next_fix_hint: `{REPORT_BUCKET_HINTS[bucket]}`")
            for row in rows_for_bucket[:3]:
                lines.append(
                    f"- `{row['ore_xml']}` | engine=`{row.get('engine')}` | summary=`{row['summary_path']}`"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_live_report_artifacts(report_root: Path, rows: list[dict[str, Any]], *, total_cases: int, top_buckets: int) -> None:
    bucket_counts = Counter(row["bucket"] for row in rows)
    engine_counts = Counter(str(row["engine"]) for row in rows if row.get("engine"))
    reference_counts = Counter(str(row["reference_source_used"]) for row in rows if row.get("reference_source_used"))
    non_clean = [row for row in rows if row["bucket"] != "clean_pass"]
    payload = {
        "total_cases": total_cases,
        "completed_cases": len(rows),
        "totals_by_status": dict(sorted(Counter("completed_zero" if row["rc"] == 0 else "completed_nonzero" for row in rows).items())),
        "totals_by_bucket": dict(sorted(bucket_counts.items())),
        "counts_by_reference_source": dict(sorted(reference_counts.items())),
        "counts_by_engine": dict(sorted(engine_counts.items())),
        "top_actionable_buckets": [
            {"bucket": bucket, "count": count, "next_fix_hint": REPORT_BUCKET_HINTS[bucket]}
            for bucket, count in bucket_counts.most_common(top_buckets)
            if bucket != "clean_pass"
        ],
        "non_clean_cases": non_clean,
    }
    _write_json(report_root / "live_summary.json", payload)
    _write_csv(
        report_root / "live_results.csv",
        rows,
    )
    (report_root / "live_report.md").write_text(
        _render_live_report_markdown(rows, total_cases=total_cases, top_buckets=top_buckets),
        encoding="utf-8",
    )


def _emit_live_case_line(done: int, total: int, row: dict[str, Any]) -> None:
    print(
        f"[{done}/{total}] {'PASS' if row['pass_all'] else 'FAIL'} "
        f"bucket={row['bucket']} "
        f"reason={row.get('fallback_reason') or row.get('pricing_fallback_reason') or row['bucket']} "
        f"{row['ore_xml']}"
    )


def _emit_live_refresh(done: int, total: int, rows: list[dict[str, Any]], report_root: Path, *, top_buckets: int) -> None:
    bucket_counts = Counter(row["bucket"] for row in rows)
    clean_count = sum(1 for row in rows if row["bucket"] == "clean_pass")
    print(f"live_report completed={done}/{total} clean={clean_count} report={report_root / 'live_report.md'}")
    for bucket, count in bucket_counts.most_common(top_buckets):
        print(f"  bucket {bucket}={count}")
    shown = 0
    for row in rows:
        if row["bucket"] == "clean_pass":
            continue
        print(f"  case {row['bucket']} {row['ore_xml']}")
        shown += 1
        if shown >= min(3, top_buckets):
            break


def _run_report_examples(args: argparse.Namespace, artifact_root: Path) -> int:
    ore_xmls = sorted(_examples_root().glob("**/Input/ore*.xml"))
    if not ore_xmls:
        raise FileNotFoundError(f"no example ore.xml files found under {_examples_root()}")
    report_root = (args.report_root or (artifact_root / "example_live_report")).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    cases_root = report_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def _failed_flag_count(summary: dict[str, Any]) -> int:
        flags = summary.get("pass_flags") or {}
        if not flags:
            return 0 if bool(summary.get("pass_all")) else 1
        return sum(1 for ok in flags.values() if not bool(ok))

    def _prefer_candidate(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
        current_diag = current.get("diagnostics") or {}
        candidate_diag = candidate.get("diagnostics") or {}
        if bool(candidate.get("pass_all")) and not bool(current.get("pass_all")):
            return True
        if bool(candidate.get("pass_all")) == bool(current.get("pass_all")):
            if _failed_flag_count(candidate) < _failed_flag_count(current):
                return True
            if _failed_flag_count(candidate) == _failed_flag_count(current):
                if bool(current_diag.get("sample_count_mismatch")) and not bool(
                    candidate_diag.get("sample_count_mismatch")
                ):
                    return True
        return False

    def _run_one(ore_xml: Path) -> tuple[dict[str, Any], int, Path]:
        case_root = cases_root / _unique_report_case_slug(ore_xml)
        case_root.mkdir(parents=True, exist_ok=True)
        capture = io.StringIO()
        with redirect_stdout(capture):
            with _REPORT_CASE_EXEC_LOCK:
                try:
                    case_summary = _run_case(ore_xml, args, artifact_root=case_root)
                except Exception:
                    # In a repo-wide sweep, a forced mode can be too aggressive for
                    # examples that are price-only, sensitivity-only, or otherwise
                    # not compatible with the requested explicit mode. Retry once
                    # with inferred modes and keep the regular one-case CLI
                    # behavior unchanged.
                    if bool(args.price or args.xva or args.sensi):
                        retry_args = argparse.Namespace(**vars(args))
                        retry_args.price = False
                        retry_args.xva = False
                        retry_args.sensi = False
                        case_summary = _run_case(ore_xml, retry_args, artifact_root=case_root)
                    else:
                        raise
                if (
                    str(getattr(args, "lgm_param_source", "auto")).strip().lower() == "simulation_xml"
                    and bool((case_summary.get("modes") or []))
                    and "xva" in (case_summary.get("modes") or [])
                    and str((case_summary.get("diagnostics") or {}).get("engine", "")) == "compare"
                ):
                    retry_args = argparse.Namespace(**vars(args))
                    retry_args.lgm_param_source = "auto"
                    auto_summary = _run_case_in_subprocess(
                        ore_xml,
                        retry_args,
                        artifact_root=case_root / "_auto_retry",
                    )
                    if _prefer_candidate(case_summary, auto_summary):
                        case_summary = auto_summary
                ore_samples = int(((case_summary.get("diagnostics") or {}).get("ore_samples") or 0) or 0)
                python_paths = int(((case_summary.get("diagnostics") or {}).get("python_paths") or 0) or 0)
                if (
                    str((case_summary.get("diagnostics") or {}).get("engine", "")) == "compare"
                    and bool((case_summary.get("diagnostics") or {}).get("sample_count_mismatch"))
                    and ore_samples > 0
                    and python_paths > 0
                    and ore_samples != python_paths
                ):
                    retry_args = argparse.Namespace(**vars(args))
                    retry_args.paths = ore_samples
                    sample_matched_summary = _run_case(ore_xml, retry_args, artifact_root=case_root)
                    if _prefer_candidate(case_summary, sample_matched_summary):
                        case_summary = sample_matched_summary
        summary_path = case_root / _case_slug(ore_xml) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(summary_path, case_summary)
        rc = 0 if case_summary["pass_all"] else 1
        return case_summary, rc, summary_path

    with cf.ThreadPoolExecutor(max_workers=int(args.report_workers)) as executor:
        futures = {executor.submit(_run_one, ore_xml): ore_xml for ore_xml in ore_xmls}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            ore_xml = futures[future]
            try:
                case_summary, rc, summary_path = future.result()
                row = _report_row_from_case_summary(ore_xml, case_summary, rc, summary_path=summary_path)
            except Exception as exc:
                row = {
                    "ore_xml": str(ore_xml),
                    "case_slug": _unique_report_case_slug(ore_xml),
                    "rc": 1,
                    "pass_all": False,
                    "trade_id": "",
                    "modes": "",
                    "engine": None,
                    "reference_source_used": "none",
                    "fallback_reason": None,
                    "pricing_fallback_reason": None,
                    "sample_count_mismatch": None,
                    "pricing_abs_diff": None,
                    "cva_rel_diff": None,
                    "dva_rel_diff": None,
                    "fba_rel_diff": None,
                    "fca_rel_diff": None,
                    "parity_ready": None,
                    "bucket": "hard_error",
                    "next_fix_hint": REPORT_BUCKET_HINTS["hard_error"],
                    "summary_path": "",
                }
            rows.append(row)
            rows.sort(key=lambda item: item["ore_xml"])
            _emit_live_case_line(done, len(ore_xmls), row)
            if done % int(args.report_refresh_every) == 0 or done == len(ore_xmls):
                _write_live_report_artifacts(
                    report_root,
                    rows,
                    total_cases=len(ore_xmls),
                    top_buckets=int(args.report_top_buckets),
                )
                _emit_live_refresh(done, len(ore_xmls), rows, report_root, top_buckets=int(args.report_top_buckets))
    return 0 if all(bool(row["pass_all"]) for row in rows) else 1


def _run_case(
    ore_xml: Path,
    args: argparse.Namespace,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    engine = _normalize_buffer_engine(getattr(args, "engine", "compare"))
    explicit_mode_request = bool(args.price or args.xva or args.sensi)
    modes = _infer_modes(args, ore_xml)
    first_trade_type = _first_trade_type(ore_xml)
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    case_summary: dict[str, Any] = {
        "ore_xml": str(ore_xml),
        "portfolio_xml": str(portfolio_xml) if portfolio_xml is not None else "",
        "modes": [name for name, enabled in asdict(modes).items() if enabled],
    }
    validation = (
        _skipped_input_validation()
        if getattr(args, "skip_input_validation", False)
        else validate_ore_input_snapshot(ore_xml, requested_modes=[name for name, enabled in asdict(modes).items() if enabled])
    )
    case_summary["input_validation"] = validation

    if modes.xva:
        try:
            if engine == "ore":
                raise FileNotFoundError("ORE reference engine explicitly requested")
            portfolio_mode = False
            if getattr(args, "trade_id", None) is None:
                portfolio_mode = _portfolio_contains_swap_like_trade(ore_xml)
            if portfolio_mode:
                base_summary = _compute_portfolio_xva_case(
                    ore_xml,
                    paths=args.paths,
                    seed=args.seed,
                    rng_mode=args.rng,
                    xva_mode=args.xva_mode,
                )
            elif first_trade_type == "FxOption":
                base_summary = _compute_fx_option_snapshot_case(
                    ore_xml,
                    paths=args.paths,
                    seed=args.seed,
                    rng_mode=args.rng,
                    anchor_t0_npv=args.anchor_t0_npv,
                    own_hazard=args.own_hazard,
                    own_recovery=args.own_recovery,
                    xva_mode=args.xva_mode,
                )
            elif first_trade_type == "CapFloor":
                if _supports_native_inflation_capfloor_price_only(ore_xml):
                    base_summary = _compute_inflation_capfloor_snapshot_case(
                        ore_xml,
                        paths=args.paths,
                        seed=args.seed,
                        rng_mode=args.rng,
                        anchor_t0_npv=args.anchor_t0_npv,
                        own_hazard=args.own_hazard,
                        own_recovery=args.own_recovery,
                        xva_mode=args.xva_mode,
                    )
                else:
                    base_summary = _compute_capfloor_snapshot_case(
                        ore_xml,
                        paths=args.paths,
                        seed=args.seed,
                        rng_mode=args.rng,
                        anchor_t0_npv=args.anchor_t0_npv,
                        own_hazard=args.own_hazard,
                        own_recovery=args.own_recovery,
                        xva_mode=args.xva_mode,
                    )
            elif first_trade_type == "Swap" and _is_inflation_swap_trade(ore_xml):
                base_summary = _compute_inflation_swap_snapshot_case(
                    ore_xml,
                    paths=args.paths,
                    seed=args.seed,
                    rng_mode=args.rng,
                    anchor_t0_npv=args.anchor_t0_npv,
                    own_hazard=args.own_hazard,
                    own_recovery=args.own_recovery,
                    xva_mode=args.xva_mode,
                )
            else:
                base_summary = _compute_snapshot_case(
                    ore_xml,
                    paths=args.paths,
                    seed=args.seed,
                    rng_mode=args.rng,
                    anchor_t0_npv=args.anchor_t0_npv,
                    own_hazard=args.own_hazard,
                    own_recovery=args.own_recovery,
                    xva_mode=args.xva_mode,
                    lgm_param_source=args.lgm_param_source,
                )
        except Exception as exc:
            if engine == "python":
                raise
            if not _is_reference_fallback_error(exc):
                raise
            reference_summary = _ore_reference_summary(
                ore_xml,
                modes,
                allow_partial_reference=True,
            )
            diagnostics = dict(reference_summary.get("diagnostics") or {})
            diagnostics["engine"] = "ore_reference_expected_output"
            diagnostics["fallback_reason"] = (
                "missing_native_output" if isinstance(exc, FileNotFoundError) else "unsupported_python_snapshot"
            )
            diagnostics["fallback_error"] = str(exc)
            reference_summary["diagnostics"] = diagnostics
            case_summary.update(reference_summary)
            base_summary = None
        if base_summary is not None:
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
                    "exposure_profile_by_trade": base_summary.exposure_profile_by_trade,
                    "exposure_profile_by_netting_set": base_summary.exposure_profile_by_netting_set,
                }
            )
    elif modes.price:
        try:
            if engine == "ore":
                raise FileNotFoundError("ORE reference engine explicitly requested")
            if _supports_native_price_only(first_trade_type, ore_xml) or (
                _has_active_simulation_analytic(ore_xml)
                and first_trade_type == "Swap"
                and _is_plain_vanilla_swap_trade(ore_xml)
            ):
                price_summary = _compute_price_only_case(
                    ore_xml,
                    anchor_t0_npv=args.anchor_t0_npv,
                    trade_id_override=args.trade_id,
                    use_reference_artifacts=(engine == "compare"),
                )
                if not explicit_mode_request and first_trade_type in {"Bond", "ForwardBond", "CallableBond"}:
                    price_summary = _python_only_summary(price_summary)
                    diagnostics = dict(price_summary.get("diagnostics") or {})
                    diagnostics.setdefault("engine", "python_bond_price_only")
                    price_summary["diagnostics"] = diagnostics
            else:
                price_summary = _price_reference_summary(ore_xml)
        except Exception as exc:
            if engine == "python":
                raise
            if not _is_reference_fallback_error(exc):
                raise
            if isinstance(exc, FileNotFoundError):
                price_summary = _ore_reference_summary(
                    ore_xml,
                    ModeSelection(price=True, xva=False, sensi=False),
                    allow_partial_reference=True,
                )
            else:
                price_summary = _ore_reference_summary(
                    ore_xml,
                    ModeSelection(price=True, xva=False, sensi=False),
                    allow_partial_reference=True,
                )
            diagnostics = dict(price_summary.get("diagnostics") or {})
            diagnostics["engine"] = "ore_reference_expected_output"
            diagnostics["fallback_reason"] = (
                "missing_native_output" if isinstance(exc, FileNotFoundError) else "unsupported_python_snapshot"
            )
            diagnostics["fallback_error"] = str(exc)
            price_summary["diagnostics"] = diagnostics
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
        trade_id, counterparty, netting_set_id = _default_case_identity(ore_xml)
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
                "diagnostics": {"mode": "non_pricing", "engine": "non_pricing"},
            }
        )
    if modes.sensi:
        sensi_progress = _SensitivityProgressBar() if getattr(args, "sensi_progress", False) else None
        try:
            case_summary["sensitivity"] = _run_sensitivity_case(
                ore_xml,
                metric=args.sensi_metric,
                netting_set=args.netting_set,
                top=args.top,
                lgm_param_source=args.lgm_param_source,
                progress_callback=None if sensi_progress is None else sensi_progress.update,
            )
        except Exception as exc:
            case_summary["sensitivity"] = {
                "metric": args.sensi_metric,
                "python_factor_count": 0,
                "ore_factor_count": 0,
                "matched_factor_count": 0,
                "unmatched_ore_count": 0,
                "unmatched_python_count": 0,
                "unsupported_factor_count": 0,
                "notes": [f"sensitivity fallback: {exc}"],
                "top_comparisons": [],
            }
            diagnostics = dict(case_summary.get("diagnostics") or {})
            diagnostics["sensitivity_fallback_reason"] = "unsupported_python_sensitivity"
            diagnostics["sensitivity_fallback_error"] = str(exc)
            if diagnostics.get("engine") is None:
                diagnostics["engine"] = "non_pricing"
            case_summary["diagnostics"] = diagnostics
        finally:
            if sensi_progress is not None:
                sensi_progress.finish()
    if (
        modes.price
        and not case_summary.get("pricing")
        and isinstance(case_summary.get("sensitivity"), dict)
        and str((case_summary.get("sensitivity") or {}).get("metric", "")).strip().upper() in {"NPV", "PV"}
    ):
        sensitivity_rows = list((case_summary.get("sensitivity") or {}).get("python_rows") or [])
        if sensitivity_rows:
            base_metric_value = float(sensitivity_rows[0].get("base_metric_value") or 0.0)
            case_summary["pricing"] = {
                "py_t0_npv": base_metric_value,
                "trade_type": _first_trade_type(ore_xml),
                "pricing_mode": "python_native_from_sensitivity",
                "report_ccy": (
                    (ET.parse(ore_xml).getroot().findtext("./Setup/Parameter[@name='baseCurrency']") or "").strip().upper()
                    or "USD"
                ),
            }
            diagnostics = dict(case_summary.get("diagnostics") or {})
            diagnostics["engine"] = "python_native"
            diagnostics["pricing_mode"] = "python_native_from_sensitivity"
            diagnostics["missing_native_pricing_reference"] = True
            if diagnostics.get("fallback_reason") == "missing_native_output":
                diagnostics.pop("fallback_reason", None)
                diagnostics.pop("fallback_error", None)
            case_summary["diagnostics"] = diagnostics
    parity_summary = (case_summary.get("parity") or {}).get("summary", {})
    requested_xva_metrics = {
        str(metric).upper()
        for metric in parity_summary.get("requested_xva_metrics", [])
    }
    pricing_diff = (case_summary.get("pricing") or {}).get("t0_npv_abs_diff")
    xva_summary = case_summary.get("xva") or {}
    cva_diff = xva_summary.get("cva_rel_diff")
    dva_diff = xva_summary.get("dva_rel_diff")
    fba_diff = xva_summary.get("fba_rel_diff")
    fca_diff = xva_summary.get("fca_rel_diff")
    case_summary["pass_flags"] = {
        "pricing": True if (not modes.price or pricing_diff is None) else pricing_diff <= args.max_npv_abs_diff,
        "xva_cva": (
            True
            if (not modes.xva or "CVA" not in requested_xva_metrics or cva_diff is None)
            else cva_diff <= args.max_cva_rel
        ),
        "xva_dva": (
            True
            if (not modes.xva or "DVA" not in requested_xva_metrics or dva_diff is None)
            else dva_diff <= args.max_dva_rel
        ),
        "xva_fba": (
            True
            if (not modes.xva or "FVA" not in requested_xva_metrics or fba_diff is None)
            else fba_diff <= args.max_fba_rel
        ),
        "xva_fca": (
            True
            if (not modes.xva or "FVA" not in requested_xva_metrics or fca_diff is None)
            else fca_diff <= args.max_fca_rel
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
        if not validation.get("skipped", False):
            validation_df = ore_input_validation_dataframe(validation)
            validation_df.to_csv(case_out_dir / "input_validation.csv", index=False)
    return case_summary


def _render_terminal_case_summary(case_summary: dict[str, Any]) -> str:
    lines = [
        "ORE done.",
        f"ore_xml={case_summary['ore_xml']}",
        f"portfolio_xml={case_summary.get('portfolio_xml', '')}",
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


def _run_preflight_case(
    ore_xml: Path,
    args: argparse.Namespace,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    portfolio_xml = _resolve_case_portfolio_path(ore_xml)
    snapshot = load_from_ore_xml(
        ore_xml,
        anchor_t0_npv=args.anchor_t0_npv,
        lgm_param_source=args.lgm_param_source,
    )
    requested_modes = [name for name, enabled in asdict(_infer_modes(args, ore_xml)).items() if enabled]
    validation = validate_ore_input_snapshot(ore_xml, requested_modes=requested_modes)
    support = _classify_preflight_support(snapshot, ore_xml, requested_modes=requested_modes)
    summary = {
        "ore_xml": str(ore_xml),
        "portfolio_xml": str(portfolio_xml) if portfolio_xml is not None else "",
        "requested_modes": requested_modes,
        "validation": validation,
        "support": support,
        "native_ready": bool(validation.get("input_links_valid", False)) and support["requires_swig_trade_count"] == 0,
        "hybrid_ready": bool(validation.get("input_links_valid", False)),
        "trade_count": int(support["native_trade_count"]) + int(support["requires_swig_trade_count"]),
        "next_step": (
            "run with PythonLgmAdapter(fallback_to_swig=False) or the Python CLI pricing/XVA modes"
            if support["requires_swig_trade_count"] == 0
            else "run in hybrid mode with SWIG available, or remove the SWIG-only trades before a native-only run"
        ),
    }
    case_out_dir = artifact_root / _case_slug(ore_xml)
    case_out_dir.mkdir(parents=True, exist_ok=True)
    if not args.ore_output_only:
        _write_json(case_out_dir / "preflight.json", summary)
        support_rows = [
            {"bucket": "native_trade_ids", "value": trade_id}
            for trade_id in summary["support"]["native_trade_ids"]
        ]
        support_rows.extend(
            {"bucket": "requires_swig_trade_ids", "value": trade_id}
            for trade_id in summary["support"]["requires_swig_trade_ids"]
        )
        _write_csv(case_out_dir / "preflight_support.csv", support_rows)
        (case_out_dir / "preflight.md").write_text(_render_preflight_markdown(summary), encoding="utf-8")
    return summary


def _classify_preflight_support(
    snapshot: Any,
    ore_xml: Path,
    *,
    requested_modes: list[str],
) -> dict[str, Any]:
    if hasattr(snapshot, "portfolio"):
        return classify_portfolio_support(snapshot, fallback_to_swig=False)
    trade_type = _first_trade_type(ore_xml)
    trade_id = ""
    try:
        portfolio_xml = _resolve_case_portfolio_path(ore_xml)
        if portfolio_xml is not None and portfolio_xml.exists():
            root = ET.parse(portfolio_xml).getroot()
            trade_id = ore_snapshot_mod._get_first_trade_id(root)
    except Exception:
        trade_id = ""
    native_supported = False
    requested = {str(mode).strip().lower() for mode in requested_modes if str(mode).strip()}
    if requested.intersection({"xva", "sensi"}):
        native_supported = trade_type == "Swap" and _supports_native_python_no_reference(trade_type, ore_xml)
    else:
        native_supported = _supports_native_price_only(trade_type, ore_xml) or _supports_native_python_no_reference(
            trade_type, ore_xml
        )
    native_trade_ids = [trade_id] if native_supported and trade_id else []
    native_trade_types = [trade_type] if native_supported and trade_type else []
    requires_swig_trade_ids = [trade_id] if (not native_supported) and trade_id else []
    requires_swig_trade_types = [trade_type] if (not native_supported) and trade_type else []
    return {
        "mode": "native_only",
        "native_only": True,
        "python_supported": native_supported,
        "native_trade_ids": native_trade_ids,
        "native_trade_types": native_trade_types,
        "requires_swig_trade_ids": requires_swig_trade_ids,
        "requires_swig_trade_types": requires_swig_trade_types,
        "native_trade_count": len(native_trade_ids),
        "requires_swig_trade_count": len(requires_swig_trade_ids),
    }


def _render_preflight_markdown(summary: dict[str, Any]) -> str:
    support = summary["support"]
    validation = summary["validation"]
    lines = [
        "# ORE Snapshot CLI Preflight",
        "",
        f"- ore_xml: `{summary['ore_xml']}`",
        f"- requested_modes: `{','.join(summary['requested_modes'])}`",
        f"- validation_ok: `{bool(validation.get('input_links_valid', False))}`",
        f"- native_ready: `{summary['native_ready']}`",
        f"- hybrid_ready: `{summary['hybrid_ready']}`",
        f"- native_trade_count: `{support['native_trade_count']}`",
        f"- requires_swig_trade_count: `{support['requires_swig_trade_count']}`",
        f"- support_mode: `{support['mode']}`",
        f"- next_step: `{summary['next_step']}`",
    ]
    if support["native_trade_types"]:
        lines.append(f"- native_trade_types: `{', '.join(support['native_trade_types'])}`")
    if support["requires_swig_trade_types"]:
        lines.append(f"- requires_swig_trade_types: `{', '.join(support['requires_swig_trade_types'])}`")
    issues = validation.get("issues") or []
    if issues:
        lines.append("")
        lines.append("## Validation Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
    return "\n".join(lines) + "\n"


def _render_terminal_preflight_summary(summary: dict[str, Any]) -> str:
    support = summary["support"]
    validation = summary["validation"]
    lines = [
        "PRECHECK done.",
        f"ore_xml={summary['ore_xml']}",
        f"portfolio_xml={summary.get('portfolio_xml', '')}",
        f"requested_modes={','.join(summary['requested_modes'])}",
        f"validation_ok={bool(validation.get('input_links_valid', False))}",
        f"native_ready={summary['native_ready']}",
        f"hybrid_ready={summary['hybrid_ready']}",
        f"native_trade_count={support['native_trade_count']}",
        f"requires_swig_trade_count={support['requires_swig_trade_count']}",
    ]
    if support["requires_swig_trade_types"]:
        lines.append(f"requires_swig_trade_types={','.join(support['requires_swig_trade_types'])}")
    lines.append(f"next_step={summary['next_step']}")
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
        _ore_ok("Sensitivity: Prepare Analysis")


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
        usage="%(prog)s path/to/ore.xml [--preflight] [--price] [--xva] [--sensi] [--pack|--report-examples] [options]",
        add_help=False,
    )
    parser.add_argument("ore_xml", nargs="?")
    parser.add_argument("--trade-id", default=None)
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
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--report-examples", action="store_true")
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--report-workers", type=int, default=12)
    parser.add_argument("--report-refresh-every", type=int, default=1)
    parser.add_argument("--report-top-buckets", type=int, default=10)
    parser.add_argument("--ore-output-only", action="store_true")
    parser.add_argument("--skip-input-validation", action="store_true")
    parser.add_argument("--engine", choices=("compare", "python", "ore"), default="compare")
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rng",
        choices=("numpy", "ore_parity", "ore_parity_antithetic", "ore_sobol", "ore_sobol_bridge"),
        default="ore_parity",
    )
    parser.add_argument("--xva-mode", choices=("classic", "ore"), default="ore")
    parser.add_argument(
        "--lgm-param-source",
        choices=("auto", "python", "calibration_xml", "simulation_xml", "ore", "provided"),
        default="auto",
    )
    parser.add_argument("--anchor-t0-npv", action="store_true")
    parser.add_argument("--own-hazard", type=float, default=0.01)
    parser.add_argument("--own-recovery", type=float, default=0.4)
    parser.add_argument("--netting-set", default=None)
    parser.add_argument("--sensi-metric", default="CVA")
    parser.add_argument("--sensi-progress", action="store_true")
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
    print("  - ore.xml is the entrypoint; Setup/portfolioFile points to the portfolio.xml with the trades")
    print("  - -v/--version matches ore.exe version flag")
    print("  - -h/--hash matches ore.exe hash flag")
    print("  - use --help for help output")
    print("  - use --example <name> with --tensor-backend auto|numpy|torch-cpu|torch-mps for built-in backend-dispatch examples")
    print("  - use --preflight to classify native-vs-SWIG support before a full run")
    print("  - use --report-examples for a live parity sweep across Examples/**/Input/ore*.xml")


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
    if not args.ore_xml and not args.pack and not args.report_examples:
        parser.error("the following arguments are required: ore_xml")

    ore_xmls: list[Path] = []
    if args.ore_xml:
        ore_xmls.append(Path(args.ore_xml).resolve())
    ore_xmls.extend(Path(item).resolve() for item in args.cases)
    if args.pack and not ore_xmls:
        parser.error("--pack requires ore_xml or at least one --case")
    if args.report_examples and ore_xmls:
        parser.error("--report-examples does not accept ore_xml or --case inputs")
    if not args.pack and not args.report_examples and len(ore_xmls) != 1:
        parser.error("normal runs accept exactly one ore_xml input")

    artifact_root = args.output_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.pack:
        return _run_pack(args, ore_xmls, artifact_root)
    if args.report_examples:
        return _run_report_examples(args, artifact_root)
    if args.preflight:
        summary = _run_preflight_case(ore_xmls[0], args, artifact_root=artifact_root)
        print(_render_terminal_preflight_summary(summary))
        return 0

    modes = _infer_modes(args, ore_xmls[0])
    start = time.perf_counter()
    _emit_ore_style_header(modes, args)
    case_summary = _run_case(ore_xmls[0], args, artifact_root=artifact_root)
    _emit_ore_style_footer(modes, time.perf_counter() - start)
    return 0 if case_summary["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
