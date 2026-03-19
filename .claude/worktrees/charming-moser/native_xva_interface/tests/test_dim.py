from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from native_xva_interface import PythonDeltaGammaVarHelper, PythonSimpleDynamicSimm, PythonSimpleDynamicSimmConfig
from native_xva_interface.dim import calculate_python_dim


def _config() -> PythonSimpleDynamicSimmConfig:
    return PythonSimpleDynamicSimmConfig(
        corr_ir_fx=0.25,
        ir_delta_rw=np.array([1.0]),
        ir_vega_rw=2.0,
        ir_gamma=0.0,
        ir_curvature_scaling=1.0,
        ir_delta_correlations=np.array([[1.0]]),
        ir_vega_correlations=np.array([[1.0]]),
        ir_curvature_weights=np.array([1.5]),
        fx_delta_rw=3.0,
        fx_vega_rw=4.0,
        fx_sigma=0.5,
        fx_hvr=2.0,
        fx_corr=0.0,
        fx_vega_correlations=np.array([[1.0]]),
        fx_curvature_weights=np.array([0.7]),
    )


def test_python_simple_dynamic_simm_matches_manual_formula():
    calc = PythonSimpleDynamicSimm(_config())
    ir_delta = np.array([[[1.0]], [[2.0]]], dtype=float)
    ir_vega = np.array([[[0.5]], [[0.25]]], dtype=float)
    fx_delta = np.array([[0.2]], dtype=float)
    fx_vega = np.array([[[0.3]]], dtype=float)

    result = calc.value(ir_delta, ir_vega, fx_delta, fx_vega)

    delta_ir = math.sqrt(1.0**2 + 2.0**2)
    vega_ir = math.sqrt((2.0 * 0.5) ** 2 + (2.0 * 0.25) ** 2)
    s_ir = 1.5 * 0.5 + 1.5 * 0.25
    kb_ir = math.sqrt((1.5 * 0.5) ** 2 + (1.5 * 0.25) ** 2)
    curvature_ir = s_ir + 5.634896601 * kb_ir
    delta_fx = abs(3.0 * 0.2)
    vega_fx = abs(4.0 * 0.5 * 2.0 * 0.3)
    s_fx = 0.7 * 0.5 * 0.3
    curvature_fx = s_fx + 5.634896601 * abs(s_fx)
    im_ir = delta_ir + vega_ir + curvature_ir
    im_fx = delta_fx + vega_fx + curvature_fx
    total = math.sqrt(im_ir * im_ir + im_fx * im_fx + 2.0 * 0.25 * im_ir * im_fx)

    assert result.delta_margin_ir[0] == delta_ir
    assert result.vega_margin_ir[0] == vega_ir
    assert result.curvature_margin_ir[0] == curvature_ir
    assert result.delta_margin_fx[0] == delta_fx
    assert result.vega_margin_fx[0] == vega_fx
    assert result.curvature_margin_fx[0] == curvature_fx
    assert result.total[0] == total


def test_calculate_python_dim_emits_reports_and_cube():
    feeder = {
        "currencies": ["EUR", "USD"],
        "ir_delta_terms": ["1Y"],
        "ir_vega_terms": ["1Y"],
        "fx_vega_terms": ["1Y"],
        "simm_config": {
            "corr_ir_fx": 0.25,
            "ir_delta_rw": [1.0],
            "ir_vega_rw": 2.0,
            "ir_gamma": 0.0,
            "ir_curvature_scaling": 1.0,
            "ir_delta_correlations": [[1.0]],
            "ir_vega_correlations": [[1.0]],
            "ir_curvature_weights": [1.5],
            "fx_delta_rw": 3.0,
            "fx_vega_rw": 4.0,
            "fx_sigma": 0.5,
            "fx_hvr": 2.0,
            "fx_corr": 0.0,
            "fx_vega_correlations": [[1.0]],
            "fx_curvature_weights": [0.7],
        },
        "netting_sets": {
            "CPTY_A": {
                "current_slice": {
                    "time": 0.0,
                    "date": "2025-02-10",
                    "days_in_period": 14,
                    "numeraire": [1.0, 1.0],
                    "ir_delta": [[[1.0, 1.0]], [[2.0, 2.0]]],
                    "ir_vega": [[[0.5, 0.5]], [[0.25, 0.25]]],
                    "fx_delta": [[0.2, 0.2]],
                    "fx_vega": [[[0.3, 0.3]]],
                },
                "time_slices": [
                    {
                        "time": 0.5,
                        "date": "2025-08-10",
                        "days_in_period": 14,
                        "numeraire": [1.0, 2.0],
                        "ir_delta": [[[1.0, 1.0]], [[2.0, 2.0]]],
                        "ir_vega": [[[0.5, 0.5]], [[0.25, 0.25]]],
                        "fx_delta": [[0.2, 0.2]],
                        "fx_vega": [[[0.3, 0.3]]],
                        "flow": [10.0, 20.0],
                    }
                ],
                "regression": {"rows": [{"Sample": 0, "RegressionDIM": 1.23}]},
            }
        },
    }

    result = calculate_python_dim({"python.dim_feeder": feeder})

    assert result.current_dim["CPTY_A"] > 0.0
    assert "dim_evolution" in result.reports
    assert "dim_distribution" in result.reports
    assert "dim_cube" in result.reports
    assert "dim_regression" in result.reports
    assert "dim_cube" in result.cubes
    assert result.reports["dim_evolution"][0]["NettingSet"] == "CPTY_A"


def test_python_delta_var_matches_manual_formula():
    helper = PythonDeltaGammaVarHelper(config=type("Cfg", (), {"quantile": 0.99, "theta_factor": 14.0 / 365.25})(), order=1)
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]], dtype=float)
    delta = np.array([2.0, -1.0], dtype=float)
    gamma = np.zeros((2, 2), dtype=float)
    theta = 3.0

    result = helper.value(covariance, delta, gamma, theta)

    expected = math.sqrt(float(delta @ covariance @ delta)) * NormalDist().inv_cdf(0.99) + theta * (14.0 / 365.25)
    assert result == expected


def test_python_delta_gamma_normal_var_matches_manual_formula():
    helper = PythonDeltaGammaVarHelper(config=type("Cfg", (), {"quantile": 0.99, "theta_factor": 0.0})(), order=2)
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=float)
    delta = np.array([0.4, -0.3], dtype=float)
    gamma = np.array([[0.5, 0.1], [0.1, -0.2]], dtype=float)

    result = helper.value(covariance, delta, gamma, 0.0)

    num = max(np.max(np.abs(delta)), np.max(np.abs(gamma)))
    tmp_delta = delta / num
    tmp_gamma = gamma / num
    d_od = float(tmp_delta @ covariance @ tmp_delta)
    go = tmp_gamma @ covariance
    mu = 0.5 * float(np.trace(go))
    variance = d_od + 0.5 * float(np.trace(go @ go))
    expected = (math.sqrt(variance) * NormalDist().inv_cdf(0.99) + mu) * num
    assert result == expected


def test_calculate_python_delta_var_dim_emits_reports_and_cube():
    feeder = {
        "var_config": {"quantile": 0.99, "horizon_calendar_days": 14},
        "netting_sets": {
            "CPTY_A": {
                "current_slice": {
                    "covariance": [[4.0, 1.0], [1.0, 9.0]],
                    "delta": [2.0, -1.0],
                    "gamma": [[0.0, 0.0], [0.0, 0.0]],
                    "theta": 3.0,
                },
                "time_slices": [
                    {
                        "time": 0.5,
                        "date": "2025-08-10",
                        "days_in_period": 14,
                        "numeraire": [1.0, 2.0],
                        "covariance": [[4.0, 1.0], [1.0, 9.0]],
                        "delta": [[2.0, 1.0], [-1.0, -0.5]],
                        "gamma": [
                            [[0.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 0.0]],
                        ],
                        "theta": [3.0, 1.0],
                        "flow": [10.0, 20.0],
                    }
                ],
            }
        },
    }

    result = calculate_python_dim({"python.dim_feeder": feeder}, dim_model="DeltaVaR")

    assert result.current_dim["CPTY_A"] > 0.0
    assert "dim_evolution" in result.reports
    assert "dim_distribution" in result.reports
    assert "dim_cube" in result.reports
    assert "dim_cube" in result.cubes
    assert result.reports["dim_evolution"][0]["NettingSet"] == "CPTY_A"
