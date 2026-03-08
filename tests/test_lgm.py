import unittest

import numpy as np

from py_ore_tools.lgm import LGM1F, LGMParams, simulate_ba_measure, simulate_lgm_measure


class TestLGM(unittest.TestCase):
    def setUp(self):
        self.params = LGMParams(
            alpha_times=(1.0, 3.0),
            alpha_values=(0.015, 0.02, 0.012),
            kappa_times=(2.0,),
            kappa_values=(0.03, 0.025),
            shift=0.0,
            scaling=1.0,
        )
        self.model = LGM1F(self.params)

    def test_lgm_measure_moments(self):
        rng = np.random.default_rng(7)
        times = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=float)
        n_paths = 120000
        x = simulate_lgm_measure(self.model, times, n_paths, rng=rng)

        for i, t in enumerate(times):
            emp_mean = np.mean(x[i])
            emp_var = np.var(x[i], ddof=0)
            tgt_var = float(self.model.zeta(t))
            se_mean = np.sqrt(max(emp_var, 1.0e-16) / n_paths)
            self.assertLess(abs(emp_mean), 4.5 * se_mean + 1.0e-4)
            if tgt_var > 1.0e-8:
                self.assertLess(abs(emp_var - tgt_var) / tgt_var, 0.05)

    def test_ba_measure_interval_moments(self):
        rng = np.random.default_rng(11)
        times = np.array([0.0, 0.4, 1.0], dtype=float)
        n_paths = 180000
        x, y = simulate_ba_measure(self.model, times, n_paths, rng=rng)

        for i in range(len(times) - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            mean_dx, var_x, cov_xy, var_y = self.model.ba_interval_moments(times[i], times[i + 1])

            emp_mean_dx = np.mean(dx)
            emp_var_x = np.var(dx, ddof=0)
            emp_var_y = np.var(dy, ddof=0)
            emp_cov_xy = np.mean((dx - emp_mean_dx) * (dy - np.mean(dy)))

            self.assertLess(abs(emp_mean_dx - mean_dx), 3.5e-4)
            self.assertLess(abs(emp_var_x - var_x), 2.5e-4)
            self.assertLess(abs(emp_var_y - var_y), 2.5e-4)
            self.assertLess(abs(emp_cov_xy - cov_xy), 2.5e-4)

    def test_pricing_identities(self):
        rng = np.random.default_rng(3)
        t = 1.0
        T = 4.0
        n_paths = 50000
        x_t = rng.normal(0.0, np.sqrt(float(self.model.zeta(t))), size=n_paths)

        p0 = lambda u: float(np.exp(-0.02 * u))

        p_tt = self.model.discount_bond(t, t, x_t, p0, p0)
        self.assertTrue(np.allclose(p_tt, 1.0))

        lhs = self.model.discount_bond(t, T, x_t, p0, p0) / self.model.numeraire_lgm(t, x_t, p0)
        rhs = p0(T) * np.exp(-float(self.model.H(T)) * x_t - 0.5 * float(self.model.H(T)) ** 2 * float(self.model.zeta(t)))
        self.assertTrue(np.allclose(lhs, rhs, rtol=1.0e-11, atol=1.0e-12))

    def test_constant_parametrization_closed_forms(self):
        alpha = 0.02
        kappa = 0.03
        shift = -0.1
        scaling = 1.3
        params = LGMParams(
            alpha_times=(),
            alpha_values=(alpha,),
            kappa_times=(),
            kappa_values=(kappa,),
            shift=shift,
            scaling=scaling,
        )
        model = LGM1F(params)

        times = np.array([0.25, 1.0, 2.5, 7.0])
        zeta_expected = alpha * alpha * times / (scaling * scaling)
        h_expected = scaling * (1.0 - np.exp(-kappa * times)) / kappa + shift

        self.assertTrue(np.allclose(model.zeta(times), zeta_expected, rtol=1.0e-12, atol=1.0e-14))
        self.assertTrue(np.allclose(model.H(times), h_expected, rtol=1.0e-12, atol=1.0e-14))

    def test_params_convenience_api(self):
        p_const = LGMParams.constant(alpha=0.02, kappa=0.03, shift=0.1, scaling=1.2)
        self.assertEqual(p_const.alpha_times, ())
        self.assertEqual(p_const.kappa_times, ())
        self.assertEqual(p_const.alpha_values, (0.02,))
        self.assertEqual(p_const.kappa_values, (0.03,))

        p_spec = LGMParams.from_spec(
            alpha={"times": (1.0, 3.0), "values": (0.01, 0.02, 0.03)},
            kappa=((2.0,), (0.03, 0.025)),
            shift=0.0,
            scaling=1.0,
        )
        self.assertEqual(p_spec.alpha_times, (1.0, 3.0))
        self.assertEqual(p_spec.alpha_values, (0.01, 0.02, 0.03))
        self.assertEqual(p_spec.kappa_times, (2.0,))
        self.assertEqual(p_spec.kappa_values, (0.03, 0.025))

    def test_vectorized_helpers_match_scalar_evaluations(self):
        times = np.array([0.0, 0.1, 0.5, 1.0, 1.7, 2.0, 4.0, 6.5], dtype=float)

        int_kappa_vec = self.model._int_kappa(times)
        int_kappa_scalar = np.array([self.model._int_kappa_scalar(float(t)) for t in times])
        self.assertTrue(np.allclose(int_kappa_vec, int_kappa_scalar, rtol=1.0e-13, atol=1.0e-14))

        h_vec = self.model.H(times)
        h_scalar = np.array([float(self.model.H(float(t))) for t in times])
        self.assertTrue(np.allclose(h_vec, h_scalar, rtol=1.0e-13, atol=1.0e-14))

        zeta_vec = self.model.zeta(times)
        zeta_scalar = np.array([float(self.model.zeta(float(t))) for t in times])
        self.assertTrue(np.allclose(zeta_vec, zeta_scalar, rtol=1.0e-13, atol=1.0e-14))

    def test_exact_zetan_matches_dense_numeric_integration(self):
        intervals = [(0.0, 0.7), (0.2, 2.3), (1.0, 5.0)]
        for n in (0, 1, 2):
            for t0, t1 in intervals:
                grid = np.linspace(t0, t1, 20001)
                numeric = np.trapezoid(np.square(self.model.alpha(grid)) * np.power(self.model.H(grid), n), grid)
                exact = self.model._zetan_interval_exact(n, t0, t1)
                self.assertTrue(np.isclose(exact, numeric, rtol=5.0e-5, atol=3.0e-7))

    def test_zetan_grid_matches_pointwise_zetan(self):
        times = np.array([0.05, 0.25, 0.9, 1.8, 3.2, 5.0], dtype=float)
        grid_vals = self.model.zetan_grid(1, times)
        pointwise_vals = np.array([self.model.zetan(1, float(t)) for t in times])
        self.assertTrue(np.allclose(grid_vals, pointwise_vals, rtol=1.0e-12, atol=1.0e-13))

        grid_vals = self.model.zetan_grid(2, times)
        pointwise_vals = np.array([self.model.zetan(2, float(t)) for t in times])
        self.assertTrue(np.allclose(grid_vals, pointwise_vals, rtol=1.0e-12, atol=1.0e-13))


if __name__ == "__main__":
    unittest.main()
