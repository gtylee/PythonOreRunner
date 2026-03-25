#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from py_ore_tools.lgm import simulate_lgm_measure
from py_ore_tools.irs_xva_utils import swap_npv_from_ore_legs_dual_curve
from py_ore_tools.ore_snapshot import load_from_ore_xml
from py_ore_tools.repo_paths import local_parity_artifacts_root


def _parse_args() -> argparse.Namespace:
    artifact_root = local_parity_artifacts_root()
    default_xml = artifact_root / "multiccy_benchmark_final/cases/flat_EUR_5Y_A/Input/ore.xml"
    default_out = artifact_root / "multiccy_benchmark_final/cases/flat_EUR_5Y_A/Output/epe_ene_py_vs_ore_vs_lgm1d.png"

    p = argparse.ArgumentParser(description="Plot ORE vs Python MC vs LGM 1D semi-analytic EPE/ENE")
    p.add_argument("--ore-xml", type=Path, default=default_xml)
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gh-order", type=int, default=128, help="Gauss-Hermite order for 1D integration")
    p.add_argument("--out", type=Path, default=default_out)
    return p.parse_args()


def _mc_profile(snap, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = snap.build_model()
    rng = np.random.default_rng(seed)
    x = simulate_lgm_measure(model, snap.exposure_model_times, n_paths=n_paths, rng=None)
    npv = np.zeros((snap.exposure_model_times.size, n_paths), dtype=float)
    for i, t in enumerate(snap.exposure_model_times):
        npv[i, :] = swap_npv_from_ore_legs_dual_curve(
            model, snap.p0_disc, snap.p0_fwd, snap.legs, float(t), x[i, :]
        )
    epe = np.mean(np.maximum(npv, 0.0), axis=1)
    ene = np.mean(np.maximum(-npv, 0.0), axis=1)
    return epe, ene


def _semi_analytic_profile(snap, gh_order: int) -> tuple[np.ndarray, np.ndarray]:
    model = snap.build_model()
    nodes, weights = np.polynomial.hermite.hermgauss(int(gh_order))
    norm = 1.0 / np.sqrt(np.pi)

    epe = np.zeros_like(snap.exposure_model_times, dtype=float)
    ene = np.zeros_like(snap.exposure_model_times, dtype=float)

    for i, t in enumerate(snap.exposure_model_times):
        var_t = float(np.asarray(model.zeta(float(t))).reshape(-1)[0])
        if var_t <= 1.0e-16:
            v = swap_npv_from_ore_legs_dual_curve(
                model, snap.p0_disc, snap.p0_fwd, snap.legs, float(t), np.array([0.0])
            )[0]
            epe[i] = max(v, 0.0)
            ene[i] = max(-v, 0.0)
            continue

        x_nodes = np.sqrt(2.0 * var_t) * nodes
        v_nodes = swap_npv_from_ore_legs_dual_curve(
            model, snap.p0_disc, snap.p0_fwd, snap.legs, float(t), x_nodes
        )
        epe[i] = norm * float(np.sum(weights * np.maximum(v_nodes, 0.0)))
        ene[i] = norm * float(np.sum(weights * np.maximum(-v_nodes, 0.0)))

    return epe, ene


def main() -> None:
    args = _parse_args()
    snap = load_from_ore_xml(args.ore_xml)

    py_epe_mc, py_ene_mc = _mc_profile(snap, n_paths=args.paths, seed=args.seed)
    py_epe_1d, py_ene_1d = _semi_analytic_profile(snap, gh_order=args.gh_order)

    ore_epe = snap.ore_epe
    ore_ene = snap.ore_ene
    t = snap.exposure_times

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, ore_epe, label="ORE EPE", linewidth=2)
    axes[0].plot(t, py_epe_mc, label=f"Python MC EPE ({args.paths} paths)", linewidth=1.8, linestyle="--")
    axes[0].plot(t, py_epe_1d, label=f"LGM 1D EPE (GH-{args.gh_order})", linewidth=1.8, linestyle=":")
    axes[0].set_title("EPE: ORE vs Python MC vs LGM 1D")
    axes[0].set_ylabel("Exposure")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, ore_ene, label="ORE ENE", linewidth=2)
    axes[1].plot(t, py_ene_mc, label=f"Python MC ENE ({args.paths} paths)", linewidth=1.8, linestyle="--")
    axes[1].plot(t, py_ene_1d, label=f"LGM 1D ENE (GH-{args.gh_order})", linewidth=1.8, linestyle=":")
    axes[1].set_title("ENE: ORE vs Python MC vs LGM 1D")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("Exposure")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{snap.trade_id}: exposure profile parity")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    print(f"Saved plot: {args.out}")
    print(f"EPE MAE vs ORE: MC={mae(py_epe_mc, ore_epe):.2f}, LGM1D={mae(py_epe_1d, ore_epe):.2f}")
    print(f"ENE MAE vs ORE: MC={mae(py_ene_mc, ore_ene):.2f}, LGM1D={mae(py_ene_1d, ore_ene):.2f}")


if __name__ == "__main__":
    main()
