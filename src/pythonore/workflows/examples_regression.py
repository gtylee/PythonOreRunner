from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import csv
import difflib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pythonore.workflows.ore_snapshot_cli as ore_snapshot_cli


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "regression_artifacts" / "examples_python" / "manifest.json"
DEFAULT_BASELINE_ROOT = DEFAULT_MANIFEST.parent / "cases"
_REPO_ROOT_TOKEN = "$REPO_ROOT"


@dataclass(frozen=True)
class ExampleRegressionCase:
    case_id: str
    ore_xml: str
    cli_args: tuple[str, ...] = ()
    description: str = ""
    workflow: str = "python"
    baseline_mode: str = "python"
    tags: tuple[str, ...] = ()
    secondary_parity: bool = True
    files_to_compare: tuple[str, ...] = ()
    tolerances: dict[str, float] = field(default_factory=dict)

    @property
    def ore_xml_path(self) -> Path:
        return REPO_ROOT / self.ore_xml

    @property
    def baseline_dir(self) -> Path:
        return DEFAULT_BASELINE_ROOT / self.case_id

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExampleRegressionCase":
        return cls(
            case_id=str(payload["case_id"]),
            ore_xml=str(payload["ore_xml"]),
            cli_args=tuple(payload.get("cli_args", ())),
            description=str(payload.get("description", "")),
            workflow=str(payload.get("workflow", "python")),
            baseline_mode=str(payload.get("baseline_mode", "python")),
            tags=tuple(payload.get("tags", ())),
            secondary_parity=bool(payload.get("secondary_parity", True)),
            files_to_compare=tuple(payload.get("files_to_compare", ())),
            tolerances=dict(payload.get("tolerances", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ore_xml": self.ore_xml,
            "cli_args": list(self.cli_args),
            "description": self.description,
            "workflow": self.workflow,
            "baseline_mode": self.baseline_mode,
            "tags": list(self.tags),
            "secondary_parity": self.secondary_parity,
            "files_to_compare": list(self.files_to_compare),
            "tolerances": self.tolerances,
        }


@dataclass(frozen=True)
class ExampleRegressionResult:
    case: ExampleRegressionCase
    exit_code: int
    summary: dict[str, Any]
    python_summary: dict[str, Any]
    report_markdown: str
    input_validation_rows: list[dict[str, str]]
    output_files: dict[str, str]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[ExampleRegressionCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(ExampleRegressionCase.from_dict(item) for item in payload["cases"])


def refresh_baselines(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    case_ids: set[str] | None = None,
) -> list[Path]:
    written: list[Path] = []
    for case in load_manifest(manifest_path):
        if case_ids and case.case_id not in case_ids:
            continue
        result = _run_example_case(case)
        case_dir = baseline_root / case.case_id
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_baseline_case(case_dir, result)
        written.append(case_dir)
    return written


def compare_baselines(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    case_ids: set[str] | None = None,
) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for case in load_manifest(manifest_path):
        if case_ids and case.case_id not in case_ids:
            continue
        diffs = _compare_case(case, baseline_root / case.case_id)
        if diffs:
            failures[case.case_id] = diffs
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh or compare Python-first regression baselines from Examples.")
    parser.add_argument("command", choices=("list", "refresh", "compare"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    baseline_root = Path(args.baseline_root)
    case_ids = set(args.case)

    if args.command == "list":
        for case in load_manifest(manifest):
            if case_ids and case.case_id not in case_ids:
                continue
            print(f"{case.case_id}: {case.ore_xml} {' '.join(case.cli_args)}")
        return 0

    if args.command == "refresh":
        written = refresh_baselines(manifest_path=manifest, baseline_root=baseline_root, case_ids=case_ids or None)
        for path in written:
            print(path)
        return 0

    failures = compare_baselines(manifest_path=manifest, baseline_root=baseline_root, case_ids=case_ids or None)
    if not failures:
        print("All Python example baselines match.")
        return 0
    for case_id, diffs in failures.items():
        print(case_id)
        for diff in diffs:
            print(diff)
    return 1


def _run_example_case(case: ExampleRegressionCase) -> ExampleRegressionResult:
    with tempfile.TemporaryDirectory(prefix=f"pythonore_example_{case.case_id}_") as tmp:
        artifact_root = Path(tmp) / "artifacts"
        argv = [str(case.ore_xml_path), *case.cli_args, "--output-root", str(artifact_root)]
        exit_code = ore_snapshot_cli.main(argv)
        if exit_code not in (0, 1):
            raise RuntimeError(f"Case {case.case_id} failed with exit code {exit_code}")
        case_out_dir = artifact_root / ore_snapshot_cli._case_slug(case.ore_xml_path)
        summary = json.loads((case_out_dir / "summary.json").read_text(encoding="utf-8"))
        python_summary = _normalize_payload(ore_snapshot_cli._python_only_summary(summary))
        report_markdown = ore_snapshot_cli._render_case_markdown(python_summary)
        input_validation_rows = _read_csv_rows(case_out_dir / "input_validation.csv")
        output_files = {}
        selected_files = set(case.files_to_compare)
        for path in sorted(case_out_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name in {"summary.json", "comparison.csv", "input_validation.csv", "report.md"}:
                continue
            if selected_files and path.name not in selected_files:
                continue
            output_files[path.name] = _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        return ExampleRegressionResult(
            case=case,
            exit_code=exit_code,
            summary=_normalize_payload(summary),
            python_summary=python_summary,
            report_markdown=_normalize_text(report_markdown),
            input_validation_rows=_normalize_payload(input_validation_rows),
            output_files=output_files,
        )


def _write_baseline_case(case_dir: Path, result: ExampleRegressionResult) -> None:
    (case_dir / "summary.json").write_text(json.dumps(result.python_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (case_dir / "parity_reference.json").write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(case_dir / "input_validation.csv", result.input_validation_rows)
    (case_dir / "report.md").write_text(result.report_markdown, encoding="utf-8")
    metadata = {
        "case": result.case.to_dict(),
        "exit_code": result.exit_code,
        "baseline_mode": result.case.baseline_mode,
        "files": sorted(["summary.json", "input_validation.csv", "report.md", *result.output_files.keys()]),
    }
    (case_dir / "case.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, content in result.output_files.items():
        (case_dir / name).write_text(content, encoding="utf-8")


def _compare_case(case: ExampleRegressionCase, baseline_dir: Path) -> list[str]:
    if not baseline_dir.exists():
        return [f"missing baseline dir: {baseline_dir}"]
    current = _run_example_case(case)
    diffs: list[str] = []
    expected_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    if expected_summary != current.python_summary:
        diffs.append(_render_json_diff(baseline_dir / "summary.json", expected_summary, current.python_summary))
    expected_rows = _read_csv_rows(baseline_dir / "input_validation.csv")
    if expected_rows != current.input_validation_rows:
        diffs.append(_render_json_diff(baseline_dir / "input_validation.csv", expected_rows, current.input_validation_rows))
    expected_report = (baseline_dir / "report.md").read_text(encoding="utf-8")
    if expected_report != current.report_markdown:
        diffs.append(_render_text_diff(baseline_dir / "report.md", expected_report, current.report_markdown))
    expected_files = sorted(path.name for path in baseline_dir.iterdir() if path.is_file() and path.name not in {"summary.json", "parity_reference.json", "input_validation.csv", "report.md", "case.json"})
    current_files = sorted(current.output_files)
    if expected_files != current_files:
        diffs.append(f"output file set mismatch: expected {expected_files}, got {current_files}")
    for name in sorted(set(expected_files).intersection(current.output_files)):
        expected_text = (baseline_dir / name).read_text(encoding="utf-8", errors="ignore")
        actual_text = current.output_files[name]
        if expected_text != actual_text:
            diffs.append(_render_text_diff(baseline_dir / name, expected_text, actual_text))
    return diffs


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return [_normalize_payload(value) for value in payload]
    if isinstance(payload, str):
        return _normalize_text(payload)
    return payload


def _normalize_text(text: str) -> str:
    return text.replace(str(REPO_ROOT), _REPO_ROOT_TOKEN)


def _render_json_diff(path: Path, expected: Any, actual: Any) -> str:
    expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    return "\n".join(
        difflib.unified_diff(expected_text, actual_text, fromfile=f"{path}:expected", tofile=f"{path}:actual", lineterm="")
    )


def _render_text_diff(path: Path, expected: str, actual: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"{path}:expected",
            tofile=f"{path}:actual",
            lineterm="",
        )
    )
