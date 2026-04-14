import unittest

import numpy as np

from py_ore_tools.lgm import LGM1F, LGMParams
from ore_curve_fit_parity.interpolation import build_cubic_discount_interpolator, build_log_linear_discount_interpolator

try:
    import torch
except ImportError:  # pragma: no cover - torch-specific tests skip without torch
    torch = None

if torch is not None:
    from py_ore_tools.lgm_torch import TorchLGM1F, simulate_lgm_measure_torch
    from py_ore_tools.lgm_torch_xva import TorchDiscountCurve, swap_npv_from_ore_legs_dual_curve_torch


@unittest.skipIf(torch is None, "torch is required for torch LGM tests")
class TestTorchLGM(unittest.TestCase):
    def setUp(self):
        self.params = LGMParams(
            alpha_times=(1.0, 3.0),
            alpha_values=(0.015, 0.02, 0.012),
            kappa_times=(2.0,),
            kappa_values=(0.03, 0.025),
            shift=0.0,
            scaling=1.0,
        )
        self.np_model = LGM1F(self.params)
        self.torch_model = TorchLGM1F(self.params)

    def test_torch_model_matches_numpy_analytics(self):
        times = np.array([0.0, 0.25, 1.0, 2.5, 5.0], dtype=float)
        self.assertTrue(np.allclose(self.torch_model.zeta(times), self.np_model.zeta(times)))
        self.assertTrue(np.allclose(self.torch_model.H(times), self.np_model.H(times)))
        self.assertTrue(np.allclose(self.torch_model.Hprime(times), self.np_model.Hprime(times)))

    def test_torch_model_tensor_outputs(self):
        times_t = torch.tensor([0.0, 0.25, 1.0, 2.5, 5.0], dtype=torch.float64)
        zeta_t = self.torch_model.zeta(times_t)
        h_t = self.torch_model.H(times_t)
        hp_t = self.torch_model.Hprime(times_t)
        self.assertIsInstance(zeta_t, torch.Tensor)
        self.assertTrue(torch.allclose(zeta_t, torch.as_tensor(self.np_model.zeta(times_t.numpy()), dtype=times_t.dtype)))
        self.assertTrue(torch.allclose(h_t, torch.as_tensor(self.np_model.H(times_t.numpy()), dtype=times_t.dtype)))
        self.assertTrue(torch.allclose(hp_t, torch.as_tensor(self.np_model.Hprime(times_t.numpy()), dtype=times_t.dtype)))

    def test_simulate_lgm_measure_torch_matches_numpy_for_shared_draws(self):
        times = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=float)
        draws = np.random.default_rng(7).standard_normal((times.size - 1, 16))

        zeta_grid = np.asarray(self.np_model.zeta(times), dtype=float)
        step_scales = np.sqrt(np.maximum(np.diff(zeta_grid), 0.0))
        expected = np.empty((times.size, draws.shape[1]), dtype=float)
        expected[0, :] = 0.0
        expected[1:, :] = np.cumsum(step_scales[:, None] * draws, axis=0)

        actual = simulate_lgm_measure_torch(
            self.torch_model,
            times,
            draws.shape[1],
            normal_draws=draws,
            return_numpy=True,
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_simulate_lgm_measure_torch_antithetic_pairs_cancel(self):
        times = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
        actual = simulate_lgm_measure_torch(
            self.torch_model,
            times,
            8,
            rng=np.random.default_rng(7),
            return_numpy=True,
            antithetic=True,
        )
        self.assertTrue(np.allclose(actual[:, :4] + actual[:, 4:], 0.0, atol=1.0e-12))

    def test_torch_discount_curve_matches_numpy_loglinear_and_cubic(self):
        times = np.array([0.0, 5.0, 10.0, 20.0], dtype=float)
        dfs = np.array([1.0, 0.93, 0.84, 0.68], dtype=float)

        loglinear_np = build_log_linear_discount_interpolator(times.tolist(), dfs.tolist())
        cubic_np = build_cubic_discount_interpolator(times.tolist(), dfs.tolist())
        loglinear_torch = TorchDiscountCurve(times, dfs, interpolation="loglinear")
        cubic_torch = TorchDiscountCurve(times, dfs, interpolation="cubic")

        for t in [0.0, 2.5, 12.0, 30.0]:
            self.assertAlmostEqual(float(loglinear_torch.discount(t)), float(loglinear_np(t)), places=12)
            self.assertAlmostEqual(float(cubic_torch.discount(t)), float(cubic_np(t)), places=12)

        self.assertNotAlmostEqual(float(loglinear_torch.discount(30.0)), float(cubic_torch.discount(30.0)), places=8)

    def test_torch_swap_npv_uses_explicit_live_coupons_for_averaged_overnight_legs(self):
        disc_curve = TorchDiscountCurve(np.array([0.0, 1.0], dtype=float), np.array([1.0, 0.98], dtype=float))
        fwd_curve = TorchDiscountCurve(np.array([0.0, 1.0], dtype=float), np.array([1.0, 0.99], dtype=float))
        legs = {
            "fixed_pay_time": np.array([], dtype=float),
            "fixed_start_time": np.array([], dtype=float),
            "fixed_end_time": np.array([], dtype=float),
            "fixed_amount": np.array([], dtype=float),
            "float_start_time": np.array([0.10], dtype=float),
            "float_end_time": np.array([0.40], dtype=float),
            "float_pay_time": np.array([0.50], dtype=float),
            "float_accrual": np.array([0.30], dtype=float),
            "float_index_accrual": np.array([0.30], dtype=float),
            "float_notional": np.array([100.0], dtype=float),
            "float_sign": np.array([1.0], dtype=float),
            "float_spread": np.array([0.0], dtype=float),
            "float_coupon": np.array([0.0], dtype=float),
            "float_fixing_time": np.array([0.60], dtype=float),
            "float_is_averaged": np.array([True], dtype=bool),
        }
        live_coupon = np.array([[0.123]], dtype=float)
        pv = swap_npv_from_ore_legs_dual_curve_torch(
            self.torch_model,
            disc_curve,
            fwd_curve,
            legs,
            0.0,
            np.zeros(1, dtype=float),
            realized_float_coupon=live_coupon,
            live_float_coupon=live_coupon,
            return_numpy=True,
        )
        expected = 100.0 * 0.30 * 0.123 * float(disc_curve.discount(0.50))
        self.assertEqual(pv.shape, (1,))
        self.assertAlmostEqual(float(pv[0]), expected, places=12)


if __name__ == "__main__":
    unittest.main()
