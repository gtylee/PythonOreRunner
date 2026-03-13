#!/usr/bin/env python3
"""Dump a small Ore-compatible LGM RNG parity artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_ore_tools.lgm import (
    LGM1F,
    LGMParams,
    ORE_PARITY_SEQUENCE_TYPE,
    make_ore_gaussian_rng,
    simulate_lgm_measure,
)
from py_ore_tools.repo_paths import local_parity_artifacts_root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=local_parity_artifacts_root() / "lgm_rng_alignment" / "mt_seed_42_constant.json",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paths", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    params = LGMParams.constant(alpha=0.02, kappa=0.03, shift=0.0, scaling=1.0)
    times = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=float)
    model = LGM1F(params)

    n_steps = times.size - 1
    z_rng = make_ore_gaussian_rng(args.seed)
    z = np.vstack([z_rng.next_sequence(n_steps) for _ in range(args.paths)])

    x_rng = make_ore_gaussian_rng(args.seed)
    x_paths = simulate_lgm_measure(
        model,
        times,
        args.paths,
        rng=x_rng,
        x0=0.0,
        draw_order="ore_path_major",
    )

    payload = {
        "metadata": {
            "source": "QuantLib InvCumulativeMersenneTwisterGaussianRsg",
            "sequence_type": ORE_PARITY_SEQUENCE_TYPE,
            "seed": int(args.seed),
            "n_paths": int(args.paths),
            "draw_order": "ore_path_major",
        },
        "params": {
            "alpha_times": list(params.alpha_times),
            "alpha_values": list(params.alpha_values),
            "kappa_times": list(params.kappa_times),
            "kappa_values": list(params.kappa_values),
            "shift": params.shift,
            "scaling": params.scaling,
        },
        "times": times.tolist(),
        "z": z.tolist(),
        "x_paths": x_paths.tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
