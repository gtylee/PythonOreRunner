import numpy as np

from py_ore_tools.irs_xva_utils import survival_probability_from_hazard


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
