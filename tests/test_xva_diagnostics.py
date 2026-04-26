from pathlib import Path

from pythonore.runtime.xva_diagnostics import compare_native_exposure_to_ore


def test_xva_exposure_diagnostic_splits_formula_from_exposure_shape():
    case_dir = (
        Path(__file__).resolve().parents[1]
        / "parity_artifacts"
        / "multiccy_benchmark_final"
        / "cases"
        / "flat_EUR_5Y_A"
    )

    result = compare_native_exposure_to_ore(case_dir, paths=8, rng_mode="ore_parity")
    summary = result["summary"]

    assert summary["trade_id"] == "SWAP_EUR_5Y_A_flat"
    assert summary["grid_points"] == 122
    assert summary["grid_mismatch"] is False
    assert summary["native_only_grid_points"] == 0
    assert summary["market_source"] == "ore_output_curves"
    assert summary["cube_comparison_points"] == 122
    assert summary["native_raw_cube_paths"] == 8
    assert summary["backend_comparison"]["numpy_backend"] == "numpy"
    assert summary["backend_comparison"]["torch_backend"].startswith("torch:")
    assert summary["backend_comparison"]["raw_cube_max_abs_diff"] == 0.0
    assert summary["first_material_epe_divergence"]["Date"] == "2016-03-07"
    assert summary["worst_epe_divergence"]["Date"] == "2016-10-05"
    assert abs(summary["python_formula_on_ore_epe_cva"] - summary["ore_report_cva"]) < 5.0
    assert result["pointwise"][0]["Date"] == "2016-02-05"
