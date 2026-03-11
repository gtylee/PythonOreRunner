import unittest

import numpy as np

from py_ore_tools.irs_xva_utils import build_discount_curve_from_zero_rate_pairs, swap_npv_from_ore_legs_dual_curve
from py_ore_tools.lgm import LGM1F, LGMParams, simulate_lgm_measure
from py_ore_tools.lgm_ir_options import (
    BermudanSwaptionDef,
    CapFloorDef,
    bermudan_backward_price,
    bermudan_npv_paths,
    bermudan_price,
    capfloor_npv,
    capfloor_npv_paths,
)


class TestLgmIrOptions(unittest.TestCase):
    def setUp(self):
        self.model = LGM1F(LGMParams.constant(alpha=0.0, kappa=0.03))
        self.p0 = build_discount_curve_from_zero_rate_pairs([(0.0, 0.02), (30.0, 0.02)])

    def test_capfloor_deterministic_npv(self):
        cf = CapFloorDef(
            trade_id="CAP1",
            ccy="EUR",
            option_type="cap",
            start_time=np.array([1.0]),
            end_time=np.array([2.0]),
            pay_time=np.array([2.0]),
            accrual=np.array([1.0]),
            notional=np.array([1_000_000.0]),
            strike=np.array([0.01]),
            fixing_time=np.array([1.0]),
            position=1.0,
        )
        x = np.zeros(2048)
        v = capfloor_npv(self.model, self.p0, self.p0, cf, 0.0, x)
        fwd = self.p0(1.0) / self.p0(2.0) - 1.0
        expected = 1_000_000.0 * self.p0(2.0) * max(fwd - 0.01, 0.0)
        self.assertAlmostEqual(float(np.mean(v)), expected, places=10)

    def test_capfloor_profile_shape_and_terminal_zero(self):
        cf = CapFloorDef(
            trade_id="FLOOR1",
            ccy="EUR",
            option_type="floor",
            start_time=np.array([1.0]),
            end_time=np.array([2.0]),
            pay_time=np.array([2.0]),
            accrual=np.array([1.0]),
            notional=np.array([1_000_000.0]),
            strike=np.array([0.03]),
            fixing_time=np.array([1.0]),
            position=1.0,
        )
        times = np.array([0.0, 1.0, 2.0], dtype=float)
        x = simulate_lgm_measure(self.model, times, 1024, rng=np.random.default_rng(8))
        v = capfloor_npv_paths(self.model, self.p0, self.p0, cf, times, x, lock_fixings=True)
        self.assertEqual(v.shape, x.shape)
        self.assertTrue(np.allclose(v[-1, :], 0.0))

    def test_bermudan_single_exercise_deterministic(self):
        legs = {
            "fixed_pay_time": np.array([2.0]),
            "fixed_accrual": np.array([1.0]),
            "fixed_rate": np.array([0.03]),
            "fixed_notional": np.array([1_000_000.0]),
            "fixed_sign": np.array([-1.0]),
            "fixed_amount": np.array([-30_000.0]),
            "float_pay_time": np.array([2.0]),
            "float_start_time": np.array([1.0]),
            "float_end_time": np.array([2.0]),
            "float_accrual": np.array([1.0]),
            "float_notional": np.array([1_000_000.0]),
            "float_sign": np.array([1.0]),
            "float_spread": np.array([0.0]),
            "float_coupon": np.array([0.0]),
        }
        berm = BermudanSwaptionDef(
            trade_id="BERM1",
            exercise_times=np.array([1.0]),
            underlying_legs=legs,
            exercise_sign=-1.0,
        )
        times = np.array([0.0, 1.0, 2.0], dtype=float)
        x = simulate_lgm_measure(self.model, times, 4096, rng=np.random.default_rng(11))

        v_paths = bermudan_npv_paths(self.model, self.p0, self.p0, berm, times, x)
        p = bermudan_price(self.model, self.p0, self.p0, berm, times, x)
        self.assertEqual(v_paths.shape, x.shape)
        self.assertAlmostEqual(float(np.mean(v_paths[0, :])), p, places=12)

        swap_at_ex = swap_npv_from_ore_legs_dual_curve(self.model, self.p0, self.p0, legs, 1.0, np.zeros(1))[0]
        expected = self.p0(1.0) * max(-swap_at_ex, 0.0)
        self.assertAlmostEqual(p, expected, places=9)

    def test_bermudan_backward_matches_lsmc_reasonably_on_multi_exercise_case(self):
        model = LGM1F(LGMParams.constant(alpha=0.01, kappa=0.03))
        legs = {
            "fixed_pay_time": np.array([2.0, 3.0]),
            "fixed_start_time": np.array([1.0, 2.0]),
            "fixed_end_time": np.array([2.0, 3.0]),
            "fixed_accrual": np.array([1.0, 1.0]),
            "fixed_rate": np.array([0.03, 0.03]),
            "fixed_notional": np.array([1_000_000.0, 1_000_000.0]),
            "fixed_sign": np.array([-1.0, -1.0]),
            "fixed_amount": np.array([-30_000.0, -30_000.0]),
            "float_pay_time": np.array([2.0, 3.0]),
            "float_start_time": np.array([1.0, 2.0]),
            "float_end_time": np.array([2.0, 3.0]),
            "float_accrual": np.array([1.0, 1.0]),
            "float_notional": np.array([1_000_000.0, 1_000_000.0]),
            "float_sign": np.array([1.0, 1.0]),
            "float_spread": np.array([0.0, 0.0]),
            "float_coupon": np.array([0.0, 0.0]),
        }
        berm = BermudanSwaptionDef(
            trade_id="BERM2",
            exercise_times=np.array([1.0, 2.0]),
            underlying_legs=legs,
            exercise_sign=-1.0,
        )
        times = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        x = simulate_lgm_measure(model, times, 4096, rng=np.random.default_rng(23))

        lsmc_price = bermudan_price(model, self.p0, self.p0, berm, times, x)
        backward = bermudan_backward_price(model, self.p0, self.p0, berm, n_grid=81)

        self.assertGreater(lsmc_price, 0.0)
        self.assertGreater(backward.price, 0.0)
        self.assertAlmostEqual(backward.price, lsmc_price, delta=0.08 * lsmc_price)
        self.assertEqual(len(backward.diagnostics), 2)


if __name__ == "__main__":
    unittest.main()
