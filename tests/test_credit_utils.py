import numpy as np

from py_ore_tools.irs_xva_utils import (
    average_hazard_from_survival_probabilities,
    build_survival_probability_curve_from_nodes,
    survival_probabilities_from_average_hazard,
    survival_probability_from_hazard,
)


def test_survival_probability_uses_later_hazard_pillars():
    times = np.array([0.0, 1.0, 2.0, 5.0], dtype=float)
    base = survival_probability_from_hazard(
        times,
        np.array([1.0, 5.0], dtype=float),
        np.array([0.02, 0.02], dtype=float),
    )
    bumped_5y = survival_probability_from_hazard(
        times,
        np.array([1.0, 5.0], dtype=float),
        np.array([0.02, 0.03], dtype=float),
    )

    assert bumped_5y[1] == base[1]
    assert bumped_5y[2] < base[2]
    assert bumped_5y[3] < base[3]


def test_loglinear_survival_curve_matches_average_hazard_nodes():
    times = np.array([1.0, 5.0], dtype=float)
    surv = np.array([np.exp(-0.02 * 1.0), np.exp(-0.03 * 5.0)], dtype=float)
    curve = build_survival_probability_curve_from_nodes(times, surv, extrapolation="flat_zero")

    np.testing.assert_allclose(np.array([curve(1.0), curve(5.0)]), surv, rtol=0, atol=1e-12)
    expected_mid = np.exp(0.5 * np.log(surv[0]) + 0.5 * np.log(surv[1]))
    np.testing.assert_allclose(np.array([curve(3.0)]), np.array([expected_mid]), rtol=0, atol=1e-12)


def test_average_hazard_round_trip_survival_nodes():
    times = np.array([1.0, 5.0], dtype=float)
    surv = np.array([0.98, 0.86], dtype=float)
    avg_h = average_hazard_from_survival_probabilities(times, surv)
    rebuilt = survival_probabilities_from_average_hazard(times, avg_h)
    np.testing.assert_allclose(rebuilt, surv, rtol=0, atol=1e-12)


def test_loglinear_survival_curve_flat_zero_extrapolation_uses_last_average_hazard():
    times = np.array([1.0, 5.0], dtype=float)
    surv = np.array([np.exp(-0.02 * 1.0), np.exp(-0.03 * 5.0)], dtype=float)
    curve = build_survival_probability_curve_from_nodes(times, surv, extrapolation="flat_zero")
    np.testing.assert_allclose(np.array([curve(7.0)]), np.array([np.exp(-0.03 * 7.0)]), rtol=0, atol=1e-12)


def test_loglinear_survival_curve_flat_fwd_extrapolation_uses_last_forward_hazard():
    times = np.array([1.0, 5.0], dtype=float)
    surv = np.array([0.98, 0.86], dtype=float)
    curve = build_survival_probability_curve_from_nodes(times, surv, extrapolation="flat_fwd")
    last_fwd_hazard = -(np.log(surv[1]) - np.log(surv[0])) / (5.0 - 1.0)
    expected = np.exp(np.log(surv[1]) - last_fwd_hazard * (7.0 - 5.0))
    np.testing.assert_allclose(np.array([curve(7.0)]), np.array([expected]), rtol=0, atol=1e-12)
