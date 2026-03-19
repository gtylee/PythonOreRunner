#!/usr/bin/env python3
"""Run a convention sweep for Python-vs-ORE LGM IRS CVA parity."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_bool_list(s: str) -> list[bool]:
    out: list[bool] = []
    for tok in _parse_list(s):
        low = tok.lower()
        if low in ("1", "true", "t", "y", "yes", "on"):
            out.append(True)
        elif low in ("0", "false", "f", "n", "no", "off"):
            out.append(False)
        else:
            raise ValueError(f"invalid bool token '{tok}'")
    if not out:
        raise ValueError("boolean list is empty")
    return out


def _parse_float_list(s: str) -> list[float]:
    vals = [float(x) for x in _parse_list(s)]
    if not vals:
        raise ValueError("float list is empty")
    return vals


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["fixed", "calibrated"], default="calibrated")
    p.add_argument("--trade-id", default="Swap_20")
    p.add_argument("--cpty", default="CPTY_A")
    p.add_argument("--paths", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--discount-column", default="EUR-EONIA")
    p.add_argument("--forward-columns", default="EUR-EURIBOR-6M,EUR-EONIA")
    p.add_argument("--swap-sources", default="trade,flows")
    p.add_argument("--pathwise-fixing-lock-options", default="false,true")
    p.add_argument("--node-interp-options", default="false,true")
    p.add_argument("--spread-calibration-options", default="false,true")
    p.add_argument("--alpha-scales", default="1.0,1.05")
    p.add_argument("--alpha-source", choices=["auto", "simulation", "calibration"], default="calibration")
    p.add_argument("--artifact-root", type=Path, default=repo_root / "Tools/PythonOreRunner/parity_artifacts")
    p.add_argument("--out-prefix", default="sweep")
    p.add_argument("--max-cases", type=int, default=0, help="0 means no limit")
    return p.parse_args()


def _run_case(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def main() -> None:
    args = _parse_args()
    script = Path(__file__).resolve().parent / "compare_ore_python_lgm.py"
    if not script.exists():
        raise FileNotFoundError(script)

    swap_sources = _parse_list(args.swap_sources)
    forward_columns = _parse_list(args.forward_columns)
    pathwise_options = _parse_bool_list(args.pathwise_fixing_lock_options)
    node_interp_options = _parse_bool_list(args.node_interp_options)
    spread_calib_options = _parse_bool_list(args.spread_calibration_options)
    alpha_scales = _parse_float_list(args.alpha_scales)

    combos = list(
        itertools.product(
            swap_sources,
            forward_columns,
            pathwise_options,
            node_interp_options,
            spread_calib_options,
            alpha_scales,
        )
    )
    if args.max_cases > 0:
        combos = combos[: args.max_cases]
    if not combos:
        raise ValueError("no sweep combinations produced")

    scenario_dir = args.artifact_root / args.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []

    t0 = perf_counter()
    n_total = len(combos)
    for i, (swap_source, fwd_col, use_fix_lock, use_node_interp, use_spread_calib, alpha_scale) in enumerate(combos, start=1):
        case_id = (
            f"{args.out_prefix}_c{i:03d}_"
            f"{swap_source}_"
            f"{fwd_col.replace('-', '_')}_"
            f"fix{int(use_fix_lock)}_"
            f"node{int(use_node_interp)}_"
            f"spr{int(use_spread_calib)}_"
            f"a{alpha_scale:.4f}"
        )
        cmd = [
            sys.executable,
            str(script),
            "--scenario",
            args.scenario,
            "--trade-id",
            args.trade_id,
            "--cpty",
            args.cpty,
            "--paths",
            str(args.paths),
            "--seed",
            str(args.seed),
            "--discount-column",
            args.discount_column,
            "--forward-column",
            fwd_col,
            "--swap-source",
            swap_source,
            "--alpha-source",
            args.alpha_source,
            "--alpha-scale",
            str(alpha_scale),
            "--out-prefix",
            case_id,
            "--artifact-root",
            str(args.artifact_root),
        ]
        if use_fix_lock:
            cmd.append("--pathwise-fixing-lock")
        if not use_node_interp:
            cmd.append("--no-node-tenor-interp")
        if not use_spread_calib:
            cmd.append("--no-coupon-spread-calibration")

        res = _run_case(cmd)
        summary_json = scenario_dir / f"{case_id}_summary.json"
        row = {
            "case_id": case_id,
            "status": "ok" if res.returncode == 0 and summary_json.exists() else "error",
            "returncode": res.returncode,
            "swap_source": swap_source,
            "forward_column": fwd_col,
            "pathwise_fixing_lock": use_fix_lock,
            "node_tenor_interp": use_node_interp,
            "coupon_spread_calibration": use_spread_calib,
            "alpha_scale": alpha_scale,
            "ore_cva": None,
            "py_cva": None,
            "cva_rel_diff": None,
            "stdout_tail": res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "",
            "stderr_tail": res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "",
        }
        if row["status"] == "ok":
            with open(summary_json, "r", encoding="utf-8") as f:
                s = json.load(f)
            row["ore_cva"] = float(s["ore_cva"])
            row["py_cva"] = float(s["py_cva"])
            row["cva_rel_diff"] = float(s["cva_rel_diff"])
        run_rows.append(row)
        print(f"[{i}/{n_total}] {case_id} -> {row['status']} rel={row['cva_rel_diff']}")

    elapsed = perf_counter() - t0
    out_csv = scenario_dir / f"{args.out_prefix}_results.csv"
    out_json = scenario_dir / f"{args.out_prefix}_results.json"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "status",
                "returncode",
                "swap_source",
                "forward_column",
                "pathwise_fixing_lock",
                "node_tenor_interp",
                "coupon_spread_calibration",
                "alpha_scale",
                "ore_cva",
                "py_cva",
                "cva_rel_diff",
                "stdout_tail",
                "stderr_tail",
            ],
        )
        writer.writeheader()
        writer.writerows(run_rows)

    ok_rows = [r for r in run_rows if r["status"] == "ok" and r["cva_rel_diff"] is not None]
    ok_rows.sort(key=lambda r: abs(float(r["cva_rel_diff"])))
    best = ok_rows[0] if ok_rows else None

    payload = {
        "scenario": args.scenario,
        "paths": args.paths,
        "seed": args.seed,
        "cases_total": n_total,
        "cases_ok": len(ok_rows),
        "elapsed_seconds": elapsed,
        "best_case": best,
        "top5": ok_rows[:5],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Results CSV: {out_csv}")
    print(f"Results JSON: {out_json}")
    if best is not None:
        print(
            "Best case: "
            f"{best['case_id']} rel={best['cva_rel_diff']:.4%} "
            f"(PY={best['py_cva']:,.2f}, ORE={best['ore_cva']:,.2f})"
        )


if __name__ == "__main__":
    main()
