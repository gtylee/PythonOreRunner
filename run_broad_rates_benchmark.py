#!/usr/bin/env python3
"""Generate and run the broad USD rates native benchmark from the repo root.

Examples:
  python3 run_broad_rates_benchmark.py
  python3 run_broad_rates_benchmark.py --count-per-type 134 --paths 2000
  python3 run_broad_rates_benchmark.py --count-per-type 134 --paths 10000
  python3 run_broad_rates_benchmark.py --preflight-only
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_COUNT_PER_TYPE = 134

for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the broad USD rates case and run a native Python XVA benchmark."
    )
    parser.add_argument("--count-per-type", type=int, default=DEFAULT_COUNT_PER_TYPE)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument(
        "--case-root",
        type=Path,
        default=None,
        help="Output case directory. Defaults to Examples/Generated/USD_AllRatesProductsSnapshot_<count>PerType",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="cpu",
        help="Torch pricing device override for native IR torch paths.",
    )
    parser.add_argument(
        "--lgm-param-source",
        choices=("auto", "calibration_xml", "simulation_xml", "ore"),
        default="simulation_xml",
    )
    return parser.parse_args()


def _default_case_root(count_per_type: int) -> Path:
    return REPO_ROOT / "Examples" / "Generated" / f"USD_AllRatesProductsSnapshot_{count_per_type}PerType"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(part) for part in cmd))
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), str(SRC_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _safe_rate(units: float, seconds: float) -> float:
    if seconds <= 0.0:
        return float("inf")
    return float(units) / float(seconds)


def main() -> None:
    args = _parse_args()
    case_root = (args.case_root or _default_case_root(args.count_per_type)).resolve()
    ore_xml = case_root / "Input" / "ore.xml"

    if not args.no_generate:
        if ore_xml.exists() and not args.overwrite:
            print(f"Reusing existing generated case: {ore_xml}")
        else:
            generate_cmd = [
                sys.executable,
                str(REPO_ROOT / "example_ore_snapshot_usd_all_rates_products.py"),
                "--count-per-type",
                str(args.count_per_type),
                "--case-root",
                str(case_root),
                "--no-run",
            ]
            if args.overwrite:
                generate_cmd.append("--overwrite")
            _run(generate_cmd)
    elif not ore_xml.exists():
        raise FileNotFoundError(f"Expected generated case at {ore_xml}")

    from pythonore.io.loader import XVALoader
    from pythonore.runtime.runtime import XVAEngine, classify_portfolio_support
    try:
        import torch
    except Exception:
        torch = None

    snapshot = XVALoader.from_ore_xml(ore_xml)
    preflight = classify_portfolio_support(snapshot, fallback_to_swig=False)
    print("Environment")
    print(f"  python                  : {sys.executable}")
    print(f"  cwd                     : {REPO_ROOT}")
    print(f"  PYTHONPATH              : {os.environ.get('PYTHONPATH', '') or '<unset>'}")
    print(f"  OMP_NUM_THREADS         : {os.environ.get('OMP_NUM_THREADS', '') or '<unset>'}")
    print(f"  MKL_NUM_THREADS         : {os.environ.get('MKL_NUM_THREADS', '') or '<unset>'}")
    print(f"  VECLIB_MAXIMUM_THREADS  : {os.environ.get('VECLIB_MAXIMUM_THREADS', '') or '<unset>'}")
    print(f"  PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '') or '<unset>'}")
    if torch is None:
        print("  torch                   : <unavailable>")
    else:
        print(f"  torch                   : {getattr(torch, '__version__', '<unknown>')}")
        print(f"  torch num threads       : {torch.get_num_threads()}")
        print(f"  torch interop threads   : {torch.get_num_interop_threads()}")
        has_mps = bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        print(f"  torch mps available     : {has_mps}")
        print(f"  torch cuda available    : {torch.cuda.is_available()}")
        config_mod = getattr(torch, "__config__", None)
        if config_mod is not None and hasattr(config_mod, "parallel_info"):
            parallel = str(config_mod.parallel_info()).strip().splitlines()
            if parallel:
                print("  torch parallel info     :")
                for line in parallel[:8]:
                    print(f"    {line}")
    print("Preflight summary")
    print(f"  ore_xml                  : {ore_xml}")
    print(f"  total_trades             : {len(snapshot.portfolio.trades)}")
    print(f"  native_trade_count       : {preflight['native_trade_count']}")
    print(f"  requires_swig_trade_count: {preflight['requires_swig_trade_count']}")
    if preflight["requires_swig_trade_ids"]:
        preview = ", ".join(preflight["requires_swig_trade_ids"][:10])
        more = " ..." if len(preflight["requires_swig_trade_ids"]) > 10 else ""
        print(f"  requires_swig_trade_ids  : {preview}{more}")

    if args.preflight_only:
        return

    run_snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            num_paths=args.paths,
            analytics=("CVA",),
            params={
                **dict(snapshot.config.params or {}),
                "python.lgm_rng_mode": "ore_sobol_bridge",
                "python.progress": "Y",
                "python.progress_bar": "Y",
                "python.lgm_param_source": args.lgm_param_source,
                **({} if args.device == "auto" else {"python.torch_device": args.device}),
            },
        ),
    )
    print("Running native XVA benchmark")
    print(f"  paths                    : {args.paths}")
    print(f"  lgm_param_source         : {args.lgm_param_source}")
    print(f"  torch_device             : {args.device}")
    engine = XVAEngine.python_lgm_default(fallback_to_swig=False)
    session_build_t0 = perf_counter()
    session = engine.create_session(run_snapshot)
    session_build_sec = perf_counter() - session_build_t0
    run_t0 = perf_counter()
    result = session.run(return_cubes=False)
    run_sec = perf_counter() - run_t0
    trade_count = len(snapshot.portfolio.trades)
    path_trade_units = float(trade_count) * float(args.paths)
    print("Run summary")
    print(f"  pv_total                 : {float(result.pv_total):.6f}")
    print(f"  cva                      : {float(result.xva_by_metric.get('CVA', 0.0)):.6f}")
    print(f"  xva_total                : {float(result.xva_total):.6f}")
    print("Performance summary")
    print(f"  session_build_sec        : {session_build_sec:.6f}")
    print(f"  run_sec                  : {run_sec:.6f}")
    print(f"  trades_per_sec           : {_safe_rate(trade_count, run_sec):.2f}")
    print(f"  paths_per_sec            : {_safe_rate(args.paths, run_sec):.2f}")
    print(f"  path_trades_per_sec      : {_safe_rate(path_trade_units, run_sec):.2f}")


if __name__ == "__main__":
    main()
