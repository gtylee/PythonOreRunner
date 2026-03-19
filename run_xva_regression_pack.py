#!/usr/bin/env python3
"""Run a small cross-case Python-vs-ORE XVA regression pack.

This is a lightweight sanity pack for the current LGM/XVA parity path. It reuses
the OreSnapshot loader and the same pathwise revaluation logic as
``example_ore_snapshot.py``, but drives multiple preselected ORE cases and writes
compact machine-readable results.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(THIS_DIR))

from example_ore_snapshot import _simulate_with_fixing_grid
from native_xva_interface import XVALoader, XVAEngine, PythonLgmAdapter
from py_ore_tools.irs_xva_utils import (
    compute_realized_float_coupons,
    compute_xva_from_exposure_profile,
    deflate_lgm_npv_paths,
    survival_probability_from_hazard,
    swap_npv_from_ore_legs_dual_curve,
)
from py_ore_tools.lgm import make_ore_gaussian_rng
from py_ore_tools.ore_snapshot import load_from_ore_xml


REPO_ROOT = THIS_DIR.parents[1]


@dataclass(frozen=True)
class CaseDef:
    name: str
    ore_xml: Optional[str] = None
    case_type: str = "irs_snapshot"
    summary_json: Optional[str] = None


DEFAULT_CASES: tuple[CaseDef, ...] = (
    CaseDef("measure_lgm", "Examples/Exposure/Input/ore_measure_lgm.xml"),
    CaseDef("measure_lgm_fixed", "Examples/Exposure/Input/ore_measure_lgm_fixed.xml"),
    CaseDef("measure_lgm_with_calibration", "Examples/Exposure/Input/ore_measure_lgm_with_calibration.xml"),
    CaseDef(
        "flat_eur_5y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A/Input/ore.xml",
    ),
    CaseDef(
        "flat_eur_5y_b",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_B/Input/ore.xml",
    ),
    CaseDef(
        "flat_eur_10y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_10Y_A/Input/ore.xml",
    ),
    CaseDef(
        "flat_eur_10y_b",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_10Y_B/Input/ore.xml",
    ),
    CaseDef(
        "full_eur_5y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/full_EUR_5Y_A/Input/ore.xml",
    ),
    CaseDef(
        "full_eur_10y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/full_EUR_10Y_A/Input/ore.xml",
    ),
    CaseDef(
        "flat_usd_5y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_USD_5Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "flat_usd_5y_b",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_USD_5Y_B/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "flat_usd_10y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_USD_10Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "flat_usd_10y_b",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_USD_10Y_B/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "full_usd_5y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/full_USD_5Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "full_usd_10y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/full_USD_10Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "flat_gbp_5y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_GBP_5Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "flat_gbp_5y_b",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_GBP_5Y_B/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "flat_gbp_10y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_GBP_10Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "flat_gbp_10y_b",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_GBP_10Y_B/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "full_gbp_5y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/full_GBP_5Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "full_gbp_10y_a",
        "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/full_GBP_10Y_A/Input/ore.xml",
        case_type="native_ore_case",
    ),
    CaseDef(
        "fxfwd_gbpusd_1y",
        case_type="artifact_summary",
        summary_json="Tools/PythonOreRunner/parity_artifacts/fxfwd_ore_xva_benchmark/FXFWD_GBPUSD_1Y/python_compare/FXFWD_GBPUSD_1Y_summary.json",
    ),
    CaseDef(
        "fxfwd_usdcad_2y",
        case_type="artifact_summary",
        summary_json="Tools/PythonOreRunner/parity_artifacts/fxfwd_ore_xva_benchmark/FXFWD_USDCAD_2Y/python_compare/FXFWD_USDCAD_2Y_summary.json",
    ),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a small cross-case XVA parity regression pack.")
    p.add_argument("--paths", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "xva_regression_pack_latest",
    )
    p.add_argument(
        "--cases",
        nargs="*",
        default=[c.name for c in DEFAULT_CASES],
        help="Subset of case names to run. Defaults to the built-in pack.",
    )
    p.add_argument("--max-cva-rel", type=float, default=0.05)
    p.add_argument("--max-dva-rel", type=float, default=0.05)
    p.add_argument("--max-fba-rel", type=float, default=0.05)
    p.add_argument("--max-fca-rel", type=float, default=0.05)
    return p.parse_args()


def _case_map() -> dict[str, CaseDef]:
    return {c.name: c for c in DEFAULT_CASES}


def _safe_rel(py: float, ore: float) -> Optional[float]:
    denom = max(abs(float(ore)), 1.0)
    return abs(float(py) - float(ore)) / denom


def _clean_inactive_metric_fields(row: dict[str, object]) -> dict[str, object]:
    metric_map = {
        "cva": bool(row.get("metric_active_cva", True)),
        "dva": bool(row.get("metric_active_dva", True)),
        "fba": bool(row.get("metric_active_fba", True)),
        "fca": bool(row.get("metric_active_fca", True)),
    }
    for metric, active in metric_map.items():
        if active:
            continue
        row[f"ore_{metric}"] = None
        row[f"py_{metric}"] = None
        row[f"{metric}_rel_diff"] = None
    return row


def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _write_markdown_summary(path: Path, summary: dict[str, object]) -> None:
    rows = list(summary.get("rows", []))
    lines: list[str] = []
    lines.append("# XVA Regression Pack Summary")
    lines.append("")
    lines.append(f"- Paths: `{summary['paths']}`")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append(f"- Cases run: `{summary['cases_ok']}`")
    lines.append(f"- Passes: `{summary['passes_all']}`")
    lines.append(f"- Failures: `{summary['cases_failed']}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    thresholds = summary.get("thresholds", {})
    lines.append(f"- CVA: `{_fmt_pct(thresholds.get('max_cva_rel'))}`")
    lines.append(f"- DVA: `{_fmt_pct(thresholds.get('max_dva_rel'))}`")
    lines.append(f"- FBA: `{_fmt_pct(thresholds.get('max_fba_rel'))}`")
    lines.append(f"- FCA: `{_fmt_pct(thresholds.get('max_fca_rel'))}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Case | Type | CVA | DVA | FBA | FCA | Pass |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| `{row['case']}` | `{row['case_type']}` | {_fmt_pct(row.get('cva_rel_diff'))} | "
            f"{_fmt_pct(row.get('dva_rel_diff'))} | {_fmt_pct(row.get('fba_rel_diff'))} | "
            f"{_fmt_pct(row.get('fca_rel_diff'))} | "
            f"{'Yes' if row.get('pass_all') else 'No'} |"
        )
    failures = list(summary.get("failures", []))
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(f"- `{failure.get('case')}`: {failure.get('error')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_irs_snapshot_case(case: CaseDef, paths: int, seed: int) -> dict[str, object]:
    snap = load_from_ore_xml(case.ore_xml)
    model = snap.build_model()
    use_ore_rng = case.name.startswith("measure_lgm")
    rng = make_ore_gaussian_rng(seed) if use_ore_rng else np.random.default_rng(seed)
    x, x_all, sim_times = _simulate_with_fixing_grid(
        model=model,
        exposure_times=snap.exposure_model_times,
        fixing_times=np.asarray(snap.legs.get("float_fixing_time", []), dtype=float),
        n_paths=paths,
        rng=rng,
        draw_order="ore_path_major" if use_ore_rng else "time_major",
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

    npv_xva = deflate_lgm_npv_paths(
        model=model,
        p0_disc=snap.p0_disc,
        times=snap.exposure_model_times,
        x_paths=x,
        npv_paths=npv,
    )
    epe = np.mean(np.maximum(npv_xva, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv_xva, 0.0), axis=1)
    times = snap.exposure_model_times
    q_c = snap.survival_probability(times)
    q_b = survival_probability_from_hazard(times, snap.own_hazard_times, snap.own_hazard_rates)
    df_ois = np.asarray([snap.p0_xva_disc(float(t)) for t in times], dtype=float)
    funding_kwargs = {}
    if snap.p0_borrow is not None and snap.p0_lend is not None:
        funding_kwargs = {
            "funding_discount_borrow": np.asarray([snap.p0_borrow(float(t)) for t in times], dtype=float),
            "funding_discount_lend": np.asarray([snap.p0_lend(float(t)) for t in times], dtype=float),
            "funding_discount_ois": df_ois,
        }

    pack = compute_xva_from_exposure_profile(
        times=times,
        epe=epe,
        ene=ene,
        discount=df_ois,
        survival_cpty=q_c,
        survival_own=q_b,
        recovery_cpty=snap.recovery,
        recovery_own=float(snap.own_recovery),
        exposure_discounting="numeraire_deflated",
        **funding_kwargs,
    )

    py_cva = float(pack["cva"])
    py_dva = float(pack["dva"])
    py_fba = float(pack.get("fba", 0.0))
    py_fca = float(pack.get("fca", 0.0))
    row = {
        "case": case.name,
        "case_type": case.case_type,
        "ore_xml": str(Path(case.ore_xml).resolve()),
        "trade_id": snap.trade_id,
        "domestic_ccy": snap.domestic_ccy,
        "paths": paths,
        "seed": seed,
        "ore_cva": snap.ore_cva,
        "py_cva": py_cva,
        "cva_rel_diff": _safe_rel(py_cva, snap.ore_cva),
        "ore_dva": snap.ore_dva,
        "py_dva": py_dva,
        "dva_rel_diff": _safe_rel(py_dva, snap.ore_dva),
        "ore_fba": snap.ore_fba,
        "py_fba": py_fba,
        "fba_rel_diff": _safe_rel(py_fba, snap.ore_fba),
        "ore_fca": snap.ore_fca,
        "py_fca": py_fca,
        "fca_rel_diff": _safe_rel(py_fca, snap.ore_fca),
        "ore_t0_npv": snap.ore_t0_npv,
        "py_t0_npv": float(np.mean(npv[0, :])),
        "t0_npv_abs_diff": abs(float(np.mean(npv[0, :])) - snap.ore_t0_npv),
        "epe_rel_median": float(np.median(np.abs(epe - snap.ore_epe) / np.maximum(np.abs(snap.ore_epe), 1.0))),
        "ene_rel_median": float(np.median(np.abs(ene - snap.ore_ene) / np.maximum(np.abs(snap.ore_ene), 1.0))),
        "metric_active_cva": True,
        "metric_active_dva": abs(float(snap.ore_dva)) > 1.0e-12,
        "metric_active_fba": abs(float(snap.ore_fba)) > 1.0e-12,
        "metric_active_fca": abs(float(snap.ore_fca)) > 1.0e-12,
    }
    return _clean_inactive_metric_fields(row)


def _run_artifact_summary_case(case: CaseDef, paths: int, seed: int) -> dict[str, object]:
    summary_path = Path(case.summary_json).resolve()
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    ore_cva = float(data["ore_cva"])
    py_cva = float(data["py_cva"])
    ore_dva = float(data.get("ore_dva", 0.0))
    py_dva = float(data.get("py_dva", 0.0))
    ore_fba = float(data.get("ore_fba", 0.0))
    py_fba = float(data.get("py_fba", 0.0))
    ore_fca = float(data.get("ore_fca", 0.0))
    py_fca = float(data.get("py_fca", 0.0))
    ore_t0_npv = float(data.get("ore_t0_npv", data.get("ore_npv0", 0.0)))
    py_t0_npv = float(data.get("py_t0_npv", data.get("py_npv0", 0.0)))

    row = {
        "case": case.name,
        "case_type": case.case_type,
        "ore_xml": str(Path(data["ore_input_xml"]).resolve()) if "ore_input_xml" in data else None,
        "trade_id": data.get("trade_id", case.name),
        "domestic_ccy": data.get("domestic_ccy", ""),
        "paths": int(data.get("n_paths", paths)),
        "seed": int(data.get("seed", seed)),
        "ore_cva": ore_cva,
        "py_cva": py_cva,
        "cva_rel_diff": _safe_rel(py_cva, ore_cva),
        "ore_dva": ore_dva,
        "py_dva": py_dva,
        "dva_rel_diff": _safe_rel(py_dva, ore_dva),
        "ore_fba": ore_fba,
        "py_fba": py_fba,
        "fba_rel_diff": _safe_rel(py_fba, ore_fba),
        "ore_fca": ore_fca,
        "py_fca": py_fca,
        "fca_rel_diff": _safe_rel(py_fca, ore_fca),
        "ore_t0_npv": ore_t0_npv,
        "py_t0_npv": py_t0_npv,
        "t0_npv_abs_diff": abs(py_t0_npv - ore_t0_npv),
        "epe_rel_median": float(data.get("epe_rel_diff_median", 0.0)),
        "ene_rel_median": float(data.get("ene_rel_diff_median", 0.0)),
        "source_summary_json": str(summary_path),
        "metric_active_cva": True,
        "metric_active_dva": "ore_dva" in data or "py_dva" in data,
        "metric_active_fba": "ore_fba" in data or "py_fba" in data,
        "metric_active_fca": "ore_fca" in data or "py_fca" in data,
    }
    return _clean_inactive_metric_fields(row)


def _read_ore_xva_row(output_dir: Path) -> dict[str, float]:
    path = output_dir / "xva.csv"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader((line.lstrip("#") for line in f if line.strip())))
    for row in rows:
        if not row.get("TradeId", "").strip():
            return {
                "CVA": float(row.get("CVA", 0.0) or 0.0),
                "DVA": float(row.get("DVA", 0.0) or 0.0),
                "FBA": float(row.get("FBA", 0.0) or 0.0),
                "FCA": float(row.get("FCA", 0.0) or 0.0),
            }
    return {}


def _read_ore_npv0(output_dir: Path) -> float:
    path = output_dir / "npv.csv"
    if not path.exists():
        return 0.0
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader((line.lstrip("#") for line in f if line.strip())))
    if not rows:
        return 0.0
    row = rows[0]
    for key in ("NPV(Base)", "NPV"):
        if key in row:
            return float(row[key] or 0.0)
    return 0.0


def _run_native_ore_case(case: CaseDef, paths: int, seed: int) -> dict[str, object]:
    ore_xml_path = Path(case.ore_xml).resolve()
    input_dir = ore_xml_path.parent
    snap = XVALoader.from_files(str(input_dir), ore_file=ore_xml_path.name)
    output_dir = (input_dir.parent / snap.config.params.get("outputPath", "Output")).resolve()
    requested_metrics = tuple(snap.config.analytics)
    snap = replace(
        snap,
        config=replace(
            snap.config,
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=paths,
            params={**snap.config.params, "python.lgm_rng_mode": "ore_parity"},
        ),
    )
    result = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snap).run(return_cubes=False)
    ore_xva = _read_ore_xva_row(output_dir)
    ore_pv = _read_ore_npv0(output_dir)
    row = {
        "case": case.name,
        "case_type": case.case_type,
        "ore_xml": str(ore_xml_path),
        "trade_id": snap.portfolio.trades[0].trade_id if snap.portfolio.trades else case.name,
        "domestic_ccy": snap.config.base_currency,
        "paths": paths,
        "seed": seed,
        "ore_cva": float(ore_xva.get("CVA", 0.0)),
        "py_cva": float(result.xva_by_metric.get("CVA", 0.0)),
        "cva_rel_diff": _safe_rel(float(result.xva_by_metric.get("CVA", 0.0)), float(ore_xva.get("CVA", 0.0))),
        "ore_dva": float(ore_xva.get("DVA", 0.0)),
        "py_dva": float(result.xva_by_metric.get("DVA", 0.0)),
        "dva_rel_diff": _safe_rel(float(result.xva_by_metric.get("DVA", 0.0)), float(ore_xva.get("DVA", 0.0))),
        "ore_fba": float(ore_xva.get("FBA", 0.0)),
        "py_fba": float(result.xva_by_metric.get("FBA", 0.0)),
        "fba_rel_diff": _safe_rel(float(result.xva_by_metric.get("FBA", 0.0)), float(ore_xva.get("FBA", 0.0))),
        "ore_fca": float(ore_xva.get("FCA", 0.0)),
        "py_fca": float(result.xva_by_metric.get("FCA", 0.0)),
        "fca_rel_diff": _safe_rel(float(result.xva_by_metric.get("FCA", 0.0)), float(ore_xva.get("FCA", 0.0))),
        "ore_t0_npv": ore_pv,
        "py_t0_npv": float(result.pv_total),
        "t0_npv_abs_diff": abs(float(result.pv_total) - ore_pv),
        "input_provenance": dict(result.metadata.get("input_provenance", {})),
        "metric_active_cva": True,
        "metric_active_dva": ("DVA" in requested_metrics),
        "metric_active_fba": ("FVA" in requested_metrics) or abs(float(ore_xva.get("FBA", 0.0))) > 1.0e-12,
        "metric_active_fca": ("FVA" in requested_metrics) or abs(float(ore_xva.get("FCA", 0.0))) > 1.0e-12,
    }
    return _clean_inactive_metric_fields(row)


def _run_case(case: CaseDef, paths: int, seed: int) -> dict[str, object]:
    if case.case_type == "irs_snapshot":
        return _run_irs_snapshot_case(case, paths=paths, seed=seed)
    if case.case_type == "native_ore_case":
        return _run_native_ore_case(case, paths=paths, seed=seed)
    if case.case_type == "artifact_summary":
        return _run_artifact_summary_case(case, paths=paths, seed=seed)
    raise ValueError(f"unsupported case_type '{case.case_type}' for case '{case.name}'")


def main() -> None:
    args = _parse_args()
    case_map = _case_map()
    selected = []
    for name in args.cases:
        if name not in case_map:
            raise ValueError(f"unknown case '{name}', choose from: {', '.join(sorted(case_map))}")
        selected.append(case_map[name])

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for case in selected:
        try:
            row = _run_case(case, paths=args.paths, seed=args.seed)
            row["pass_cva"] = bool((not row.get("metric_active_cva", True)) or float(row["cva_rel_diff"]) <= args.max_cva_rel)
            row["pass_dva"] = bool((not row.get("metric_active_dva", True)) or float(row["dva_rel_diff"]) <= args.max_dva_rel)
            row["pass_fba"] = bool((not row.get("metric_active_fba", True)) or float(row["fba_rel_diff"]) <= args.max_fba_rel)
            row["pass_fca"] = bool((not row.get("metric_active_fca", True)) or float(row["fca_rel_diff"]) <= args.max_fca_rel)
            row["pass_all"] = bool(row["pass_cva"] and row["pass_dva"] and row["pass_fba"] and row["pass_fca"])
            rows.append(row)
            def _fmt(metric: str) -> str:
                val = row.get(f"{metric}_rel_diff")
                return "n/a" if val is None else f"{float(val):.2%}"
            print(
                f"{case.name}: "
                f"CVA={_fmt('cva')} "
                f"DVA={_fmt('dva')} "
                f"FBA={_fmt('fba')} "
                f"FCA={_fmt('fca')} "
                f"pass={row['pass_all']}"
            )
        except Exception as exc:
            failures.append({"case": case.name, "ore_xml": case.ore_xml, "error": str(exc)})
            print(f"{case.name}: ERROR: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "paths": args.paths,
        "seed": args.seed,
        "thresholds": {
            "max_cva_rel": args.max_cva_rel,
            "max_dva_rel": args.max_dva_rel,
            "max_fba_rel": args.max_fba_rel,
            "max_fca_rel": args.max_fca_rel,
        },
        "cases_requested": [c.name for c in selected],
        "cases_ok": len(rows),
        "cases_failed": len(failures),
        "passes_all": sum(1 for r in rows if r["pass_all"]),
        "rows": rows,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_summary(args.output_dir / "summary_pretty.md", summary)

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(args.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
