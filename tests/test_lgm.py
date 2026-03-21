import unittest
import json
from pathlib import Path

import numpy as np

from py_ore_tools.repo_paths import pythonorerunner_root, require_engine_repo_root
from py_ore_tools.lgm import (
    LGM1F,
    LGMParams,
    ORE_PARITY_SEQUENCE_TYPE,
    ORE_SOBOL_SEQUENCE_TYPE,
    ORE_SOBOL_BROWNIAN_BRIDGE_SEQUENCE_TYPE,
    make_ore_gaussian_rng,
    simulate_ba_measure,
    simulate_lgm_measure,
)
from py_ore_tools.irs_xva_utils import (
    compute_xva_from_exposure_profile,
    deflate_lgm_npv_paths,
)
from py_ore_tools import ore_snapshot as ore_snapshot_module

try:
    import QuantLib as ql
except ImportError:  # pragma: no cover - parity mode requires QuantLib in the test env
    ql = None


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
        self.parity_fixture = (
            pythonorerunner_root()
            / "parity_artifacts"
            / "lgm_rng_alignment"
            / "mt_seed_42_constant.json"
        )

    def _quantlib_sequences(self, seed: int, n_paths: int, dimension: int) -> np.ndarray:
        if ql is None:
            self.skipTest("QuantLib Python bindings are required for Ore parity tests")
        gen = ql.InvCumulativeMersenneTwisterGaussianRsg(ql.MersenneTwisterUniformRsg(dimension, seed))
        return np.vstack([np.asarray(gen.nextSequence().value(), dtype=float) for _ in range(n_paths)])

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

    def test_lgm_measure_antithetic_pairs_cancel(self):
        times = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
        x = simulate_lgm_measure(self.model, times, 8, rng=np.random.default_rng(7), antithetic=True)
        self.assertTrue(np.allclose(x[:, :4] + x[:, 4:], 0.0, atol=1.0e-12))

    def test_ore_mt_rng_matches_oracle_fixture(self):
        payload = json.loads(self.parity_fixture.read_text(encoding="utf-8"))
        expected = np.asarray(payload["z"], dtype=float)
        seed = int(payload["metadata"]["seed"])

        rng = make_ore_gaussian_rng(seed)
        actual = np.vstack([rng.next_sequence(expected.shape[1]) for _ in range(expected.shape[0])])
        self.assertTrue(np.array_equal(actual, expected))

    def test_lgm_measure_ore_path_major_matches_oracle_fixture(self):
        payload = json.loads(self.parity_fixture.read_text(encoding="utf-8"))
        params_payload = payload["params"]
        params = LGMParams(
            alpha_times=tuple(params_payload["alpha_times"]),
            alpha_values=tuple(params_payload["alpha_values"]),
            kappa_times=tuple(params_payload["kappa_times"]),
            kappa_values=tuple(params_payload["kappa_values"]),
            shift=float(params_payload["shift"]),
            scaling=float(params_payload["scaling"]),
        )
        model = LGM1F(params)
        times = np.asarray(payload["times"], dtype=float)
        expected = np.asarray(payload["x_paths"], dtype=float)
        seed = int(payload["metadata"]["seed"])

        actual = simulate_lgm_measure(
            model,
            times,
            expected.shape[1],
            rng=make_ore_gaussian_rng(seed),
            draw_order="ore_path_major",
        )
        self.assertTrue(np.array_equal(actual, expected))

    def test_piecewise_lgm_measure_matches_quantlib_path_order_oracle(self):
        times = np.array([0.0, 0.3, 0.9, 1.7, 3.25], dtype=float)
        n_paths = 5
        seed = 17
        z = self._quantlib_sequences(seed, n_paths=n_paths, dimension=times.size - 1)

        expected = np.empty((times.size, n_paths), dtype=float)
        expected[0, :] = 0.0
        step_scales = np.sqrt(np.maximum(np.diff(self.model.zeta(times)), 0.0))
        for p in range(n_paths):
            x_curr = 0.0
            for i, scale in enumerate(step_scales):
                x_curr += scale * z[p, i]
                expected[i + 1, p] = x_curr

        actual = simulate_lgm_measure(
            self.model,
            times,
            n_paths,
            rng=make_ore_gaussian_rng(seed),
            draw_order="ore_path_major",
        )
        self.assertTrue(np.array_equal(actual, expected))

    def test_exact_parity_rejects_unknown_sequence_type(self):
        with self.assertRaisesRegex(ValueError, "unsupported sequence_type"):
            make_ore_gaussian_rng(42, sequence_type="NotARealSequence")

    def test_sobol_rng_matches_quantlib_sequence(self):
        if ql is None:
            self.skipTest("QuantLib Python bindings are required for Ore parity tests")
        seed = 42
        dimension = 5
        rng = make_ore_gaussian_rng(seed, sequence_type=ORE_SOBOL_SEQUENCE_TYPE)
        actual = np.vstack([rng.next_sequence(dimension) for _ in range(4)])
        expected_gen = ql.InvCumulativeSobolGaussianRsg(ql.SobolRsg(dimension, seed))
        expected = np.vstack([np.asarray(expected_gen.nextSequence().value(), dtype=float) for _ in range(4)])
        self.assertTrue(np.array_equal(actual, expected))

    def test_sobol_brownian_bridge_rng_matches_quantlib_transform(self):
        if ql is None:
            self.skipTest("QuantLib Python bindings are required for Ore parity tests")
        seed = 42
        dimension = 5
        rng = make_ore_gaussian_rng(seed, sequence_type=ORE_SOBOL_BROWNIAN_BRIDGE_SEQUENCE_TYPE)
        actual = np.vstack([rng.next_sequence(dimension) for _ in range(4)])
        base_gen = ql.InvCumulativeSobolGaussianRsg(ql.SobolRsg(dimension, seed))
        bridge = ql.BrownianBridge(dimension)
        expected = np.vstack(
            [
                np.asarray(bridge.transform(list(base_gen.nextSequence().value())), dtype=float)
                for _ in range(4)
            ]
        )
        self.assertTrue(np.array_equal(actual, expected))

    def test_discount_bond_profile_matches_when_paths_are_shared(self):
        payload = json.loads(self.parity_fixture.read_text(encoding="utf-8"))
        times = np.asarray(payload["times"], dtype=float)
        x_paths = np.asarray(payload["x_paths"], dtype=float)

        params_payload = payload["params"]
        model = LGM1F(
            LGMParams(
                alpha_times=tuple(params_payload["alpha_times"]),
                alpha_values=tuple(params_payload["alpha_values"]),
                kappa_times=tuple(params_payload["kappa_times"]),
                kappa_values=tuple(params_payload["kappa_values"]),
                shift=float(params_payload["shift"]),
                scaling=float(params_payload["scaling"]),
            )
        )
        p0 = lambda t: float(np.exp(-0.02 * t))

        regenerated = simulate_lgm_measure(
            model,
            times,
            x_paths.shape[1],
            rng=make_ore_gaussian_rng(int(payload["metadata"]["seed"])),
            draw_order="ore_path_major",
        )
        t = float(times[2])
        maturity = 4.0
        lhs = model.discount_bond(t, maturity, x_paths[2], p0, p0)
        rhs = model.discount_bond(t, maturity, regenerated[2], p0, p0)
        self.assertTrue(np.array_equal(lhs, rhs))

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

    def test_ore_snapshot_day_counter_conventions(self):
        self.assertAlmostEqual(
            ore_snapshot_module._year_fraction_from_day_counter("2016-02-05", "2016-05-05", "A365F"),
            90.0 / 365.0,
        )
        self.assertAlmostEqual(
            ore_snapshot_module._year_fraction_from_day_counter(
                "2016-02-05", "2016-05-05", "ActualActual(ISDA)"
            ),
            90.0 / 366.0,
        )

    def test_ore_snapshot_date_roundtrip_for_report_time(self):
        t = ore_snapshot_module._year_fraction_from_day_counter(
            "2016-02-05", "2016-05-05", "ActualActual(ISDA)"
        )
        self.assertEqual(
            ore_snapshot_module._date_from_time_with_day_counter("2016-02-05", t, "ActualActual(ISDA)"),
            "2016-05-05",
        )

    def test_curve_loader_uses_model_day_counter(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "curves.csv"
            csv_path.write_text(
                "Date,EUR-EONIA\n"
                "2016-02-05,1.0\n"
                "2016-05-05,0.99\n",
                encoding="utf-8",
            )
            payload = ore_snapshot_module._load_ore_discount_pairs_by_columns_with_day_counter(
                str(csv_path), ["EUR-EONIA"], asof_date="2016-02-05", day_counter="A365F"
            )
            dates, times, dfs = payload["EUR-EONIA"]
            self.assertEqual(dates, ("2016-02-05", "2016-05-05"))
            self.assertTrue(np.allclose(times, np.array([0.0, 90.0 / 365.0])))
            self.assertTrue(np.allclose(dfs, np.array([1.0, 0.99])))

    def test_portfolio_leg_times_can_use_a365f(self):
        from py_ore_tools.irs_xva_utils import load_swap_legs_from_portfolio

        portfolio = (
            require_engine_repo_root()
            / "Examples"
            / "Exposure"
            / "Input"
            / "portfolio_singleswap.xml"
        )
        legs = load_swap_legs_from_portfolio(
            str(portfolio), "Swap_20", "2016-02-05", time_day_counter="A365F"
        )
        self.assertAlmostEqual(float(legs["fixed_pay_time"][0]), 390.0 / 365.0)

    def test_lgm_npv_deflation_matches_numeraire_identity(self):
        times = np.array([0.0, 0.5, 1.0], dtype=float)
        x_paths = np.array(
            [
                [0.0, 0.0],
                [0.01, -0.02],
                [0.03, 0.01],
            ],
            dtype=float,
        )
        npv_paths = np.array(
            [
                [100.0, 100.0],
                [110.0, 90.0],
                [80.0, 120.0],
            ],
            dtype=float,
        )
        p0 = lambda t: float(np.exp(-0.02 * t))
        deflated = deflate_lgm_npv_paths(self.model, p0, times, x_paths, npv_paths)
        for i, t in enumerate(times):
            expected = npv_paths[i, :] / self.model.numeraire_lgm(float(t), x_paths[i, :], p0)
            self.assertTrue(np.allclose(deflated[i, :], expected))

    def test_xva_profile_ore_discounting_mode_skips_extra_df(self):
        times = np.array([0.0, 1.0, 2.0], dtype=float)
        epe = np.array([10.0, 20.0, 30.0], dtype=float)
        ene = np.array([5.0, 4.0, 3.0], dtype=float)
        df = np.array([1.0, 0.95, 0.90], dtype=float)
        q_c = np.array([1.0, 0.98, 0.95], dtype=float)
        q_b = np.array([1.0, 0.99, 0.97], dtype=float)

        classic = compute_xva_from_exposure_profile(
            times, epe, ene, df, q_c, q_b, funding_spread=0.01, exposure_discounting="discount_curve"
        )
        ore_mode = compute_xva_from_exposure_profile(
            times, epe, ene, df, q_c, q_b, funding_spread=0.01, exposure_discounting="numeraire_deflated"
        )

        self.assertGreater(classic["cva"], 0.0)
        self.assertGreater(ore_mode["cva"], classic["cva"])
        self.assertGreater(ore_mode["fva"], classic["fva"])

    def test_xva_profile_ore_funding_curves_match_increment_formula(self):
        times = np.array([0.0, 1.0, 2.0], dtype=float)
        epe = np.array([0.0, 20.0, 30.0], dtype=float)
        ene = np.array([0.0, 4.0, 3.0], dtype=float)
        df_ois = np.array([1.0, 0.99, 0.97], dtype=float)
        df_borrow = np.array([1.0, 0.985, 0.96], dtype=float)
        df_lend = np.array([1.0, 0.992, 0.975], dtype=float)
        q_c = np.array([1.0, 0.98, 0.95], dtype=float)
        q_b = np.array([1.0, 0.97, 0.94], dtype=float)

        out = compute_xva_from_exposure_profile(
            times,
            epe,
            ene,
            df_ois,
            q_c,
            q_b,
            exposure_discounting="numeraire_deflated",
            funding_discount_borrow=df_borrow,
            funding_discount_lend=df_lend,
            funding_discount_ois=df_ois,
        )
        expected_fca = (
            q_c[0] * q_b[0] * epe[1] * (df_borrow[0] / df_borrow[1] - df_ois[0] / df_ois[1])
            + q_c[1] * q_b[1] * epe[2] * (df_borrow[1] / df_borrow[2] - df_ois[1] / df_ois[2])
        )
        expected_fba = (
            q_c[0] * q_b[0] * ene[1] * (df_lend[0] / df_lend[1] - df_ois[0] / df_ois[1])
            + q_c[1] * q_b[1] * ene[2] * (df_lend[1] / df_lend[2] - df_ois[1] / df_ois[2])
        )
        self.assertAlmostEqual(float(out["fca"]), float(expected_fca), places=12)
        self.assertAlmostEqual(float(out["fba"]), float(expected_fba), places=12)
        self.assertAlmostEqual(float(out["fva"]), float(expected_fba + expected_fca), places=12)

    def test_dual_curve_swap_uses_stored_fixed_float_amounts(self):
        from py_ore_tools.irs_xva_utils import swap_npv_from_ore_legs_dual_curve

        model = LGM1F(LGMParams.constant(alpha=0.01, kappa=0.03))
        p0 = lambda t: float(np.exp(-0.02 * t))
        legs = {
            "fixed_pay_time": np.array([], dtype=float),
            "fixed_amount": np.array([], dtype=float),
            "float_pay_time": np.array([1.0], dtype=float),
            "float_start_time": np.array([0.5], dtype=float),
            "float_end_time": np.array([1.0], dtype=float),
            "float_accrual": np.array([0.5], dtype=float),
            "float_notional": np.array([100.0], dtype=float),
            "float_sign": np.array([-1.0], dtype=float),
            "float_spread": np.array([0.0], dtype=float),
            "float_coupon": np.array([0.03], dtype=float),
            "float_amount": np.array([-2.0], dtype=float),
            "float_fixing_time": np.array([0.25], dtype=float),
        }
        x_t = np.array([0.0, 0.1], dtype=float)
        pv = swap_npv_from_ore_legs_dual_curve(model, p0, p0, legs, 0.75, x_t)
        expected_df = model.discount_bond(0.75, 1.0, x_t, p0, p0)
        self.assertTrue(np.allclose(pv, -2.0 * expected_df))

    def test_dual_curve_swap_t0_ignores_node_interpolation(self):
        from py_ore_tools.irs_xva_utils import swap_npv_from_ore_legs_dual_curve

        model = LGM1F(LGMParams.constant(alpha=0.01, kappa=0.03))
        p0 = lambda t: float(np.exp(-0.02 * t))
        legs = {
            "fixed_pay_time": np.array([1.0, 2.0], dtype=float),
            "fixed_amount": np.array([10.0, 10.0], dtype=float),
            "float_pay_time": np.array([1.0], dtype=float),
            "float_start_time": np.array([0.5], dtype=float),
            "float_end_time": np.array([1.0], dtype=float),
            "float_accrual": np.array([0.5], dtype=float),
            "float_notional": np.array([100.0], dtype=float),
            "float_sign": np.array([-1.0], dtype=float),
            "float_spread": np.array([0.0], dtype=float),
            "float_coupon": np.array([0.03], dtype=float),
            "float_amount": np.array([-1.5], dtype=float),
            "float_fixing_time": np.array([0.25], dtype=float),
            "node_tenors": np.array([0.25, 0.5, 1.0], dtype=float),
        }
        x0 = np.array([0.0], dtype=float)
        pv = swap_npv_from_ore_legs_dual_curve(model, p0, p0, legs, 0.0, x0)[0]
        legs_no_nodes = dict(legs)
        legs_no_nodes.pop("node_tenors")
        expected = swap_npv_from_ore_legs_dual_curve(model, p0, p0, legs_no_nodes, 0.0, x0)[0]
        self.assertAlmostEqual(float(pv), float(expected), places=12)

    def test_dual_curve_swap_defaults_to_exact_curve_valuation(self):
        from py_ore_tools.irs_xva_utils import swap_npv_from_ore_legs_dual_curve

        model = LGM1F(LGMParams.constant(alpha=0.01, kappa=0.03))
        p0 = lambda t: float(np.exp(-0.02 * t))
        legs = {
            "fixed_pay_time": np.array([1.0, 2.0], dtype=float),
            "fixed_amount": np.array([10.0, 10.0], dtype=float),
            "float_pay_time": np.array([1.0], dtype=float),
            "float_start_time": np.array([0.5], dtype=float),
            "float_end_time": np.array([1.0], dtype=float),
            "float_accrual": np.array([0.5], dtype=float),
            "float_notional": np.array([100.0], dtype=float),
            "float_sign": np.array([-1.0], dtype=float),
            "float_spread": np.array([0.0], dtype=float),
            "float_coupon": np.array([0.03], dtype=float),
            "float_amount": np.array([-1.5], dtype=float),
            "float_fixing_time": np.array([0.25], dtype=float),
            "node_tenors": np.array([0.25, 0.5, 1.0], dtype=float),
        }
        x_t = np.array([0.0, 0.1], dtype=float)
        pv_default = swap_npv_from_ore_legs_dual_curve(model, p0, p0, legs, 0.75, x_t)
        legs_no_nodes = dict(legs)
        legs_no_nodes.pop("node_tenors")
        pv_exact = swap_npv_from_ore_legs_dual_curve(model, p0, p0, legs_no_nodes, 0.75, x_t)
        pv_nodes = swap_npv_from_ore_legs_dual_curve(
            model, p0, p0, legs, 0.75, x_t, use_node_interpolation=True
        )
        self.assertTrue(np.allclose(pv_default, pv_exact, rtol=1.0e-12, atol=1.0e-12))
        self.assertFalse(np.allclose(pv_default, pv_nodes, rtol=1.0e-12, atol=1.0e-12))

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
