#!/usr/bin/env python3
"""Explicit CPU vs GPU benchmark for 1F LGM state evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))

from pythonore.compute.lgm import LGM1F, LGMParams, simulate_lgm_measure
from pythonore.compute.lgm_torch import TorchLGM1F, simulate_lgm_measure_torch

try:
    import torch
except ImportError:  # pragma: no cover - benchmark is unusable without torch
    torch = None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, nargs="+", default=[2000, 10000, 50000, 100000])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--horizon", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "gpu"],
        help="Explicit device groups to benchmark: cpu, gpu. 'gpu' expands to cuda and/or mps when available.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON payload in addition to the human-readable report.",
    )
    return parser.parse_args(argv)


def _build_times(steps: int, horizon: float) -> np.ndarray:
    return np.linspace(0.0, float(horizon), int(steps) + 1, dtype=float)


def _build_models() -> tuple[LGM1F, TorchLGM1F]:
    params = LGMParams(
        alpha_times=(1.0, 5.0, 10.0),
        alpha_values=(0.012, 0.018, 0.015, 0.010),
        kappa_times=(3.0, 12.0),
        kappa_values=(0.025, 0.018, 0.012),
        shift=0.0,
        scaling=1.0,
    )
    return LGM1F(params), TorchLGM1F(params)


def _torch_env() -> dict[str, object]:
    if torch is None:
        return {"torch_available": False, "cpu": True, "cuda": False, "mps": False, "expanded_devices": ["cpu"]}
    has_cuda = bool(torch.cuda.is_available())
    has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return {
        "torch_available": True,
        "cpu": True,
        "cuda": has_cuda,
        "mps": has_mps,
        "expanded_devices": ["cpu"] + (["cuda"] if has_cuda else []) + (["mps"] if has_mps else []),
    }


def _expand_devices(requested: list[str]) -> list[str]:
    env = _torch_env()
    expanded: list[str] = []
    for item in requested:
        key = item.lower()
        if key == "cpu":
            expanded.append("cpu")
        elif key == "gpu":
            if env["cuda"]:
                expanded.append("cuda")
            if env["mps"]:
                expanded.append("mps")
        elif key in ("cuda", "mps"):
            expanded.append(key)
        else:
            raise ValueError(f"unsupported device spec '{item}', use cpu, gpu, cuda, or mps")
    deduped: list[str] = []
    for item in expanded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _sync_device(device: str) -> None:
    if torch is None:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _bench(fn, *, repeats: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        durations.append(perf_counter() - t0)
    arr = np.asarray(durations, dtype=float)
    return {
        "mean_sec": float(arr.mean()),
        "min_sec": float(arr.min()),
        "std_sec": float(arr.std(ddof=0)),
    }


def _numpy_from_shared_draws(model: LGM1F, times: np.ndarray, draws: np.ndarray, x0: float = 0.0) -> np.ndarray:
    zeta_grid = np.asarray(model.zeta(times), dtype=float)
    step_scales = np.sqrt(np.maximum(np.diff(zeta_grid), 0.0))
    out = np.empty((times.size, draws.shape[1]), dtype=float)
    out[0, :] = float(x0)
    out[1:, :] = float(x0) + np.cumsum(step_scales[:, None] * draws, axis=0)
    return out


def _bench_torch_mode(
    model: TorchLGM1F,
    times: np.ndarray,
    n_paths: int,
    seed: int,
    *,
    repeats: int,
    warmup: int,
    device: str,
    return_numpy: bool,
) -> dict[str, float]:
    def _run():
        _sync_device(device)
        t0 = perf_counter()
        out = simulate_lgm_measure_torch(
            model,
            times,
            n_paths,
            rng=np.random.default_rng(seed),
            device=device,
            return_numpy=return_numpy,
        )
        _sync_device(device)
        return perf_counter() - t0, out

    for _ in range(warmup):
        _run()

    durations = []
    for _ in range(repeats):
        elapsed, _ = _run()
        durations.append(elapsed)
    arr = np.asarray(durations, dtype=float)
    return {
        "mean_sec": float(arr.mean()),
        "min_sec": float(arr.min()),
        "std_sec": float(arr.std(ddof=0)),
    }


def _fmt_sec(value: float) -> str:
    return f"{value:.4f}s"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_rate(value: float) -> str:
    return f"{value:,.0f}"


def _fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _make_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([line, header_line, line, *body, line])


def _render_report(*, payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    rows = payload["results"]
    assert isinstance(metadata, dict)
    assert isinstance(rows, list)

    lines: list[str] = []
    lines.append("LGM 1F Torch Benchmark Report")
    lines.append("=" * 29)
    lines.append("Configuration:")
    lines.append(f"  - Paths tested      : {', '.join(_fmt_int(int(p)) for p in metadata['paths'])}")
    lines.append(f"  - Time grid         : {metadata['steps']} steps over {float(metadata['horizon_years']):.2f} years")
    lines.append(f"  - Repeats / warmup  : {metadata['repeats']} / {metadata['warmup']}")
    lines.append(f"  - Seed              : {metadata['seed']}")
    lines.append(f"  - Requested devices : {', '.join(str(d) for d in metadata['requested_devices'])}")
    lines.append(f"  - Expanded devices  : {', '.join(str(d) for d in metadata['expanded_devices'])}")
    lines.append("")
    lines.append("How to read this report:")
    lines.append("  - speedup_vs_numpy compares mean runtime against NumPy/CPU for the same path count.")
    lines.append("  - path_steps_per_sec is throughput (higher is better).")
    lines.append("  - parity_max_abs is max absolute difference vs NumPy using shared random draws.")
    lines.append("  - mode '*to_numpy' includes device-to-host transfer; '*device_only' excludes it.")
    lines.append("")

    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        if isinstance(row, dict):
            grouped.setdefault(int(row["n_paths"]), []).append(row)

    for n_paths in sorted(grouped):
        group = grouped[n_paths]
        numpy_cpu = next((r for r in group if r.get("mode") == "numpy/cpu"), None)
        numpy_mean = float(numpy_cpu["mean_sec"]) if isinstance(numpy_cpu, dict) else None
        table_rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        for row in sorted(group, key=lambda r: str(r["mode"])):
            mean_sec = float(row["mean_sec"])
            speedup = (numpy_mean / mean_sec) if numpy_mean is not None and mean_sec > 0.0 else None
            table_rows.append(
                (
                    str(row["mode"]),
                    str(row["device"]),
                    _fmt_sec(mean_sec),
                    _fmt_sec(float(row["min_sec"])),
                    _fmt_sec(float(row["std_sec"])),
                    _fmt_speedup(speedup),
                    _fmt_rate(float(row["path_steps_per_sec"])),
                    _fmt_sci(float(row["parity_max_abs"])),
                    "yes" if bool(row["includes_host_transfer"]) else "no",
                )
            )
        lines.append(f"Results for n_paths = {_fmt_int(n_paths)}")
        lines.append(
            _make_table(
                [
                    "mode",
                    "device",
                    "mean",
                    "min",
                    "std",
                    "speedup_vs_numpy",
                    "path_steps_per_sec",
                    "parity_max_abs",
                    "host_transfer",
                ],
                table_rows,
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if torch is None:
        raise SystemExit("torch is required for this benchmark")
    np_model, torch_model = _build_models()
    times = _build_times(args.steps, args.horizon)
    devices = _expand_devices(args.devices)
    rows: list[dict[str, float | int | str]] = []

    for n_paths in args.paths:
        shared_draws = np.random.default_rng(args.seed).standard_normal((times.size - 1, n_paths))
        numpy_shared = _numpy_from_shared_draws(np_model, times, shared_draws)
        path_steps = n_paths * max(times.size - 1, 1)

        if "cpu" in devices:
            numpy_stats = _bench(
                lambda n=n_paths: simulate_lgm_measure(np_model, times, n, rng=np.random.default_rng(args.seed)),
                repeats=args.repeats,
                warmup=args.warmup,
            )
            torch_cpu_shared = simulate_lgm_measure_torch(
                torch_model,
                times,
                n_paths,
                normal_draws=shared_draws,
                device="cpu",
            )
            rows.append(
                {
                    "engine": "numpy",
                    "mode": "numpy/cpu",
                    "device": "cpu",
                    "n_paths": n_paths,
                    "n_steps": times.size - 1,
                    "mean_sec": numpy_stats["mean_sec"],
                    "min_sec": numpy_stats["min_sec"],
                    "std_sec": numpy_stats["std_sec"],
                    "path_steps_per_sec": path_steps / max(numpy_stats["mean_sec"], 1.0e-12),
                    "parity_max_abs": 0.0,
                    "includes_host_transfer": True,
                }
            )
            torch_cpu_stats = _bench_torch_mode(
                torch_model,
                times,
                n_paths,
                args.seed,
                repeats=args.repeats,
                warmup=args.warmup,
                device="cpu",
                return_numpy=True,
            )
            rows.append(
                {
                    "engine": "torch",
                    "mode": "torch/cpu",
                    "device": "cpu",
                    "n_paths": n_paths,
                    "n_steps": times.size - 1,
                    "mean_sec": torch_cpu_stats["mean_sec"],
                    "min_sec": torch_cpu_stats["min_sec"],
                    "std_sec": torch_cpu_stats["std_sec"],
                    "path_steps_per_sec": path_steps / max(torch_cpu_stats["mean_sec"], 1.0e-12),
                    "parity_max_abs": float(np.max(np.abs(torch_cpu_shared - numpy_shared))),
                    "includes_host_transfer": True,
                }
            )

        for device in [d for d in devices if d in ("cuda", "mps")]:
            torch_gpu_shared = simulate_lgm_measure_torch(
                torch_model,
                times,
                n_paths,
                normal_draws=shared_draws,
                device=device,
                return_numpy=True,
            )
            torch_gpu_device_stats = _bench_torch_mode(
                torch_model,
                times,
                n_paths,
                args.seed,
                repeats=args.repeats,
                warmup=args.warmup,
                device=device,
                return_numpy=False,
            )
            torch_gpu_host_stats = _bench_torch_mode(
                torch_model,
                times,
                n_paths,
                args.seed,
                repeats=args.repeats,
                warmup=args.warmup,
                device=device,
                return_numpy=True,
            )
            rows.append(
                {
                    "engine": "torch",
                    "mode": f"torch/{device}/device_only",
                    "device": device,
                    "n_paths": n_paths,
                    "n_steps": times.size - 1,
                    "mean_sec": torch_gpu_device_stats["mean_sec"],
                    "min_sec": torch_gpu_device_stats["min_sec"],
                    "std_sec": torch_gpu_device_stats["std_sec"],
                    "path_steps_per_sec": path_steps / max(torch_gpu_device_stats["mean_sec"], 1.0e-12),
                    "parity_max_abs": float(np.max(np.abs(torch_gpu_shared - numpy_shared))),
                    "includes_host_transfer": False,
                }
            )
            rows.append(
                {
                    "engine": "torch",
                    "mode": f"torch/{device}/to_numpy",
                    "device": device,
                    "n_paths": n_paths,
                    "n_steps": times.size - 1,
                    "mean_sec": torch_gpu_host_stats["mean_sec"],
                    "min_sec": torch_gpu_host_stats["min_sec"],
                    "std_sec": torch_gpu_host_stats["std_sec"],
                    "path_steps_per_sec": path_steps / max(torch_gpu_host_stats["mean_sec"], 1.0e-12),
                    "parity_max_abs": float(np.max(np.abs(torch_gpu_shared - numpy_shared))),
                    "includes_host_transfer": True,
                }
            )

    payload = {
        "metadata": {
            "seed": args.seed,
            "paths": [int(p) for p in args.paths],
            "steps": times.size - 1,
            "horizon_years": float(args.horizon),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "requested_devices": args.devices,
            "expanded_devices": devices,
            "torch_env": _torch_env(),
        },
        "results": rows,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(_render_report(payload=payload))
    if args.json:
        print("Machine-readable JSON:")
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


sys.modules.setdefault("pythonore.benchmarks.benchmark_lgm_torch", sys.modules[__name__])
sys.modules.setdefault("py_ore_tools.benchmarks.benchmark_lgm_torch", sys.modules[__name__])
