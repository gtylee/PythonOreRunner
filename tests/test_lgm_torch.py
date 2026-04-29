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
    from py_ore_tools.lgm_torch_xva import TorchDiscountCurve, capfloor_npv_paths_torch, swap_npv_from_ore_legs_dual_curve_torch
    from py_ore_tools.lgm_ir_options import CapFloorDef, capfloor_npv_paths


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

    def test_capfloor_paths_precompute_unique_fixings_once(self):
        disc_curve = TorchDiscountCurve(np.array([0.0, 0.5, 1.0], dtype=float), np.array([1.0, 0.99, 0.98], dtype=float))
        fwd_curve = TorchDiscountCurve(np.array([0.0, 0.5, 1.0], dtype=float), np.array([1.0, 0.992, 0.985], dtype=float))
        capfloor = CapFloorDef(
            trade_id="CF1",
            ccy="USD",
            option_type="cap",
            start_time=np.array([0.10, 0.10], dtype=float),
            end_time=np.array([0.40, 0.50], dtype=float),
            pay_time=np.array([0.55, 0.60], dtype=float),
            accrual=np.array([0.30, 0.40], dtype=float),
            notional=np.array([100.0, 100.0], dtype=float),
            strike=np.array([0.02, 0.02], dtype=float),
            gearing=np.array([1.0, 1.0], dtype=float),
            spread=np.array([0.0, 0.0], dtype=float),
            fixing_time=np.array([0.25, 0.25], dtype=float),
            fixing_date=np.array(["2025-01-15", "2025-01-15"], dtype=object),
        )
        times = np.array([0.0, 0.5, 1.0], dtype=float)
        x_paths = np.zeros((times.size, 2), dtype=float)
        call_counter_np = {"count": 0}
        call_counter_torch = {"count": 0}

        from py_ore_tools import lgm_ir_options as lgm_ir_options_mod
        from py_ore_tools import lgm_torch_xva as lgm_torch_xva_mod

        original_np_interp = lgm_ir_options_mod.interpolate_path_grid
        original_torch_interp = lgm_torch_xva_mod.interpolate_path_grid

        def _counting_interp_np(*args, **kwargs):
            call_counter_np["count"] += 1
            return original_np_interp(*args, **kwargs)

        def _counting_interp_torch(*args, **kwargs):
            call_counter_torch["count"] += 1
            return original_np_interp(*args, **kwargs)

        try:
            lgm_ir_options_mod.interpolate_path_grid = _counting_interp_np
            pv_np = capfloor_npv_paths(
                self.np_model,
                disc_curve.discount,
                fwd_curve.discount,
                capfloor,
                times,
                x_paths,
                lock_fixings=True,
            )
            lgm_torch_xva_mod.interpolate_path_grid = _counting_interp_torch
            pv_torch = capfloor_npv_paths_torch(
                self.torch_model,
                disc_curve,
                fwd_curve,
                capfloor,
                times,
                x_paths,
                lock_fixings=True,
                return_numpy=True,
            )
        finally:
            lgm_ir_options_mod.interpolate_path_grid = original_np_interp
            lgm_torch_xva_mod.interpolate_path_grid = original_torch_interp

        self.assertEqual(call_counter_np["count"], 1)
        self.assertEqual(call_counter_torch["count"], 1)
        self.assertTrue(np.allclose(pv_np, pv_torch))

    def test_capfloor_paths_accept_mps_tensor_inputs(self):
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            self.skipTest("mps backend is not available")
        model = TorchLGM1F(self.params, device="mps")
        disc_curve = TorchDiscountCurve(
            np.array([0.0, 0.5, 1.0], dtype=float),
            np.array([1.0, 0.99, 0.98], dtype=float),
            device="mps",
        )
        fwd_curve = TorchDiscountCurve(
            np.array([0.0, 0.5, 1.0], dtype=float),
            np.array([1.0, 0.992, 0.985], dtype=float),
            device="mps",
        )
        capfloor = CapFloorDef(
            trade_id="CF_MPS",
            ccy="USD",
            option_type="cap",
            start_time=np.array([0.10], dtype=float),
            end_time=np.array([0.50], dtype=float),
            pay_time=np.array([0.60], dtype=float),
            accrual=np.array([0.40], dtype=float),
            notional=np.array([100.0], dtype=float),
            strike=np.array([0.02], dtype=float),
            gearing=np.array([1.0], dtype=float),
            spread=np.array([0.0], dtype=float),
            fixing_time=np.array([0.0], dtype=float),
        )
        times = np.array([0.0, 0.5, 1.0], dtype=float)
        x_paths = torch.zeros((times.size, 2), dtype=disc_curve.dtype, device=disc_curve.device_obj)

        pv = capfloor_npv_paths_torch(
            model,
            disc_curve,
            fwd_curve,
            capfloor,
            times,
            x_paths,
            lock_fixings=True,
            return_numpy=True,
        )

        self.assertIsInstance(pv, np.ndarray)
        self.assertEqual(pv.shape, (times.size, 2))
        self.assertTrue(np.all(np.isfinite(pv)))


if __name__ == "__main__":
    unittest.main()
