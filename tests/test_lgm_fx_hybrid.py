import unittest

import numpy as np

from py_ore_tools.lgm import LGMParams
from py_ore_tools.lgm_fx_hybrid import LgmFxHybrid, MultiCcyLgmParams
from py_ore_tools.lgm_fx_xva_utils import (
    FxForwardDef,
    aggregate_exposure_profile,
    build_lgm_params,
    build_two_ccy_hybrid,
    cva_terms_from_profile,
    fx_forward_npv,
    run_fx_forward_profile_xva,
)


class TestLgmFxHybrid(unittest.TestCase):
    def _make_model(self, corr_12: float = 0.0):
        eur = LGMParams(alpha_times=(1.0,), alpha_values=(0.01, 0.01), kappa_times=(), kappa_values=(0.03,))
        usd = LGMParams(alpha_times=(1.0,), alpha_values=(0.012, 0.012), kappa_times=(), kappa_values=(0.02,))
        corr = np.array(
            [
                [1.0, 0.0, corr_12],
                [0.0, 1.0, corr_12],
                [corr_12, corr_12, 1.0],
            ],
            dtype=float,
        )
        return LgmFxHybrid(
            MultiCcyLgmParams(
                ir_params={"EUR": eur, "USD": usd},
                fx_vols={"EUR/USD": (tuple(), (0.15,))},
                corr=corr,
            )
        )

    def test_simulation_shapes(self):
        m = self._make_model(0.1)
        t = np.array([0.0, 0.25, 1.0, 2.0])
        out = m.simulate_paths(t, 256, rng=np.random.default_rng(7), log_s0={"EUR/USD": np.log(1.1)})
        self.assertEqual(out["x"]["EUR"].shape, (4, 256))
        self.assertEqual(out["x"]["USD"].shape, (4, 256))
        self.assertEqual(out["s"]["EUR/USD"].shape, (4, 256))
        self.assertTrue(np.all(out["s"]["EUR/USD"] > 0.0))

    def test_simulation_antithetic_ir_pairs_cancel(self):
        m = self._make_model(0.1)
        t = np.array([0.0, 0.25, 1.0, 2.0])
        out = m.simulate_paths(t, 8, rng=np.random.default_rng(7), log_s0={"EUR/USD": np.log(1.1)}, antithetic=True)
        self.assertTrue(np.allclose(out["x"]["EUR"][:, :4] + out["x"]["EUR"][:, 4:], 0.0, atol=1.0e-12))
        self.assertTrue(np.allclose(out["x"]["USD"][:, :4] + out["x"]["USD"][:, 4:], 0.0, atol=1.0e-12))

    def test_fx_forward_identity(self):
        m = self._make_model(0.0)
        s = np.array([1.20, 1.30])
        p_d = np.array([0.95, 0.93])
        p_f = np.array([0.97, 0.94])
        f = m.fx_forward("EUR/USD", 0.5, 2.0, s, p_d, p_f)
        self.assertTrue(np.allclose(f, s * p_f / p_d))

    def test_fx_forward_npv_at_strike(self):
        m = self._make_model(0.0)
        fx = FxForwardDef("T1", "EUR/USD", notional_base=1_000_000, strike=1.10, maturity=1.0)
        x = np.zeros(1000)
        s = np.full(1000, 1.10)
        p0 = lambda t: float(np.exp(-0.02 * t))
        npv = fx_forward_npv(m, fx, 0.0, s, x, x, p0, p0)
        self.assertLess(abs(float(np.mean(npv))), 1.0e-10)

    def test_cva_terms(self):
        t = np.array([0.0, 1.0, 2.0])
        epe = np.array([0.0, 100.0, 100.0])
        df = np.array([1.0, 0.98, 0.95])
        q = np.array([1.0, 0.99, 0.97])
        out = cva_terms_from_profile(t, epe, df, q, recovery=0.4)
        expected = 0.6 * (100.0 * 0.98 * 0.01 + 100.0 * 0.95 * 0.02)
        self.assertAlmostEqual(float(out["cva"][0]), expected, places=12)

    def test_exposure_aggregation(self):
        x = np.array([[1.0, -2.0, 3.0], [-1.0, -2.0, 1.0]], dtype=float)
        out = aggregate_exposure_profile(x)
        self.assertTrue(np.allclose(out["ee"], np.array([2.0 / 3.0, -2.0 / 3.0])))
        self.assertTrue(np.allclose(out["epe"], np.array([4.0 / 3.0, 1.0 / 3.0])))
        self.assertTrue(np.allclose(out["ene"], np.array([-2.0 / 3.0, -1.0])))

    def test_build_param_and_hybrid_convenience(self):
        p = build_lgm_params(alpha={"times": (1.0,), "values": (0.01, 0.015)}, kappa=0.03)
        self.assertEqual(p.alpha_values, (0.01, 0.015))
        m = build_two_ccy_hybrid(
            pair="EUR/USD",
            ir_specs={"EUR": {"alpha": 0.01, "kappa": 0.03}, "USD": {"alpha": 0.012, "kappa": 0.02}},
            fx_vol=0.15,
        )
        self.assertEqual(set(m.ir_models.keys()), {"EUR", "USD"})

    def test_fx_profile_xva_runner(self):
        out = run_fx_forward_profile_xva(
            name="FXFWD_TEST",
            pair="EUR/USD",
            maturity=1.0,
            spot0=1.1,
            strike=1.12,
            notional_base=1_000_000,
            dom_zero_rate=[(0.0, 0.03), (5.0, 0.03)],
            for_zero_rate=[(0.0, 0.02), (5.0, 0.02)],
            fx_vol=0.12,
            n_paths=256,
            seed=17,
        )
        self.assertEqual(out["npv_paths"].shape[0], out["times"].shape[0])
        self.assertEqual(out["npv_paths"].shape[1], 256)
        self.assertTrue(np.isfinite(out["cva"]))


if __name__ == "__main__":
    unittest.main()
