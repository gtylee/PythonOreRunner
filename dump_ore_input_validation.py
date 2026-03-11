#!/usr/bin/env python3
"""Dump ore_snapshot input-link validation for an ORE ore.xml file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from py_ore_tools.ore_snapshot import (
    ore_input_validation_dataframe,
    validate_ore_input_snapshot,
)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_xml = (
        repo_root
        / "Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A/Input/ore.xml"
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ore-xml",
        type=Path,
        default=default_xml,
        help="Path to ORE root XML (default: %(default)s)",
    )
    p.add_argument(
        "--format",
        choices=("json", "csv", "md", "text"),
        default="text",
        help="Output format (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file path",
    )
    p.add_argument(
        "--all-market-configs",
        action="store_true",
        help="Validate all todaysmarket configurations, not just those referenced by ore.xml",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: %(default)s)",
    )
    return p.parse_args()


def _to_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    checks = report["checks"]
    issues = report["issues"]
    action_items = report.get("action_items", [])
    quotes = report["quotes"]
    conventions = report["conventions"]
    curve_specs = report["curve_specs"]
    fixings = report["fixings"]
    fx = report["fx_dominance"]
    lines = [
        "# ORE Input Validation",
        "",
        f"- ore_xml: `{summary['ore_xml_path']}`",
        f"- asof_date: `{summary['asof_date']}`",
        f"- selected_market_configs: `{', '.join(summary['selected_market_configs'])}`",
        f"- input_links_valid: `{report['input_links_valid']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Gaps",
            "",
            f"- missing_conventions: `{len(conventions['missing_ids'])}`",
            f"- missing_mandatory_quotes: `{len(quotes['missing_mandatory'])}`",
            f"- invalid_curve_specs: `{len(curve_specs['invalid'])}`",
            f"- today_fixing_count: `{fixings['today_fixing_count']}` with implyTodaysFixings=`{fixings['imply_todays_fixings']}`",
            f"- fx_pairs_with_both_directions: `{fx['pair_count_with_both_directions']}`",
            "",
            "## Issues",
            "",
        ]
    )
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- none")
    lines.extend(["", "## What To Fix", ""])
    if action_items:
        for item in action_items:
            lines.append(f"- `{item['code']}` [{item['severity']}]")
            lines.append(f"  failed: {item['what_failed']}")
            lines.append(f"  fix: {item['what_to_fix']}")
            where = item.get("where_to_fix", [])
            if where:
                lines.append(f"  where: {', '.join(str(x) for x in where)}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _to_text(report: dict[str, object]) -> str:
    summary = report["summary"]
    checks = report["checks"]
    issues = report["issues"]
    action_items = report.get("action_items", [])
    lines = [
        f"ore_xml: {summary['ore_xml_path']}",
        f"asof_date: {summary['asof_date']}",
        f"market_configs: {', '.join(summary['selected_market_configs'])}",
        f"input_links_valid: {report['input_links_valid']}",
        "checks:",
    ]
    for key, value in checks.items():
        lines.append(f"  - {key}: {value}")
    lines.append("issues:")
    if issues:
        lines.extend(f"  - {issue}" for issue in issues)
    else:
        lines.append("  - none")
    lines.append("what_to_fix:")
    if action_items:
        for item in action_items:
            lines.append(f"  - {item['code']} [{item['severity']}]")
            lines.append(f"    failed: {item['what_failed']}")
            lines.append(f"    fix: {item['what_to_fix']}")
            where = item.get("where_to_fix", [])
            if where:
                lines.append(f"    where: {', '.join(str(x) for x in where)}")
    else:
        lines.append("  - none")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    report = validate_ore_input_snapshot(
        args.ore_xml,
        include_all_market_configs=args.all_market_configs,
    )

    if args.format == "json":
        payload = json.dumps(report, indent=args.indent, sort_keys=True) + "\n"
    elif args.format == "csv":
        df = ore_input_validation_dataframe(report)
        payload = df.to_csv(index=False)
    elif args.format == "md":
        payload = _to_markdown(report) + "\n"
    else:
        payload = _to_text(report) + "\n"

    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.format} validation report: {args.out}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
