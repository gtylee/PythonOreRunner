"""Helpers for reproducible ORE parity artifact layout and metadata manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping


MANDATORY_SUBDIRS = (
    "times",
    "curves",
    "trades",
    "calibration",
    "exposure",
    "xva",
    "perf",
)


@dataclass(frozen=True)
class CaseMetadata:
    case_id: str
    run_mode: str  # fixed or calibrated
    asof_date: str
    base_ccy: str
    model_ccys: tuple[str, ...]
    fx_pairs: tuple[str, ...]
    indices: tuple[str, ...]
    products: tuple[str, ...]
    convention_profile: str
    ore_samples: int
    python_paths: int
    seed: int
    notes: str = ""


@dataclass(frozen=True)
class CasePaths:
    root: Path
    times: Path
    curves: Path
    trades: Path
    calibration: Path
    exposure: Path
    xva: Path
    perf: Path


def build_case_layout(root: Path, case_id: str, run_mode: str) -> CasePaths:
    if run_mode not in ("fixed", "calibrated"):
        raise ValueError("run_mode must be 'fixed' or 'calibrated'")
    case_root = root / run_mode / case_id
    case_root.mkdir(parents=True, exist_ok=True)

    sub = {}
    for d in MANDATORY_SUBDIRS:
        p = case_root / d
        p.mkdir(parents=True, exist_ok=True)
        sub[d] = p

    return CasePaths(
        root=case_root,
        times=sub["times"],
        curves=sub["curves"],
        trades=sub["trades"],
        calibration=sub["calibration"],
        exposure=sub["exposure"],
        xva=sub["xva"],
        perf=sub["perf"],
    )


def write_case_manifest(paths: CasePaths, metadata: CaseMetadata, extra: Mapping[str, Any] | None = None) -> Path:
    payload: Dict[str, Any] = asdict(metadata)
    payload["artifact_layout"] = {k: str(paths.root / k) for k in MANDATORY_SUBDIRS}
    if extra:
        payload["extra"] = dict(extra)
    out = paths.root / "manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_command_log(paths: CasePaths, ore_cmd: list[str], python_cmd: list[str] | None = None) -> Path:
    payload = {"ore_command": ore_cmd}
    if python_cmd is not None:
        payload["python_command"] = python_cmd
    out = paths.root / "commands.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


__all__ = [
    "MANDATORY_SUBDIRS",
    "CaseMetadata",
    "CasePaths",
    "build_case_layout",
    "write_case_manifest",
    "write_command_log",
]
