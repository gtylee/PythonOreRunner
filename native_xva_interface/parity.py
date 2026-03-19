from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .results import XVAResult


@dataclass(frozen=True)
class ParityTolerance:
    abs_tol: float = 1e-6
    rel_tol: float = 1e-4


@dataclass
class ParityCheckResult:
    ok: bool
    diffs: List[str]


def compare_results(
    legacy: XVAResult,
    native: XVAResult,
    metric_tolerances: Dict[str, ParityTolerance] | None = None,
) -> ParityCheckResult:
    diffs: List[str] = []
    metric_tolerances = metric_tolerances or {}

    _compare_scalar("pv_total", legacy.pv_total, native.pv_total, metric_tolerances.get("pv_total", ParityTolerance()), diffs)
    _compare_scalar("xva_total", legacy.xva_total, native.xva_total, metric_tolerances.get("xva_total", ParityTolerance()), diffs)

    all_metrics = set(legacy.xva_by_metric).union(native.xva_by_metric)
    for m in sorted(all_metrics):
        _compare_scalar(
            f"xva_by_metric[{m}]",
            legacy.xva_by_metric.get(m, 0.0),
            native.xva_by_metric.get(m, 0.0),
            metric_tolerances.get(m, ParityTolerance()),
            diffs,
        )

    all_ns = set(legacy.exposure_by_netting_set).union(native.exposure_by_netting_set)
    for ns in sorted(all_ns):
        _compare_scalar(
            f"exposure_by_netting_set[{ns}]",
            legacy.exposure_by_netting_set.get(ns, 0.0),
            native.exposure_by_netting_set.get(ns, 0.0),
            metric_tolerances.get(f"ns:{ns}", ParityTolerance()),
            diffs,
        )

    return ParityCheckResult(ok=not diffs, diffs=diffs)


def _compare_scalar(name: str, left: float, right: float, tol: ParityTolerance, diffs: List[str]) -> None:
    diff = abs(left - right)
    rel = diff / max(abs(left), abs(right), 1.0)
    if diff > tol.abs_tol and rel > tol.rel_tol:
        diffs.append(f"{name}: left={left}, right={right}, abs={diff}, rel={rel}")
