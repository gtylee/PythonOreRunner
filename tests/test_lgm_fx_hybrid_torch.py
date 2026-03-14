import unittest

import numpy as np

from py_ore_tools.lgm import LGMParams
from py_ore_tools.lgm_fx_hybrid import LgmFxHybrid, MultiCcyLgmParams

try:
    import torch
except ImportError:  # pragma: no cover - torch-specific tests skip without torch
    torch = None

if torch is not None:
    from py_ore_tools.lgm_fx_hybrid_torch import TorchLgmFxHybrid, simulate_hybrid_paths_torch


@unittest.skipIf(torch is None, "torch is required for torch hybrid tests")
class TestTorchLgmFxHybrid(unittest.TestCase):
    def _make_params(self):
        eur = LGMParams(alpha_times=(1.0,), alpha_values=(0.01, 0.01), kappa_times=(), kappa_values=(0.03,))
        usd = LGMParams(alpha_times=(1.0,), alpha_values=(0.012, 0.012), kappa_times=(), kappa_values=(0.02,))
        gbp = LGMParams(alpha_times=(1.0,), alpha_values=(0.011, 0.011), kappa_times=(), kappa_values=(0.025,))
        corr = np.array(
            [
                [1.0, 0.20, 0.10, 0.15, 0.05],
                [0.20, 1.0, 0.25, 0.10, 0.08],
                [0.10, 0.25, 1.0, 0.06, 0.12],
                [0.15, 0.10, 0.06, 1.0, 0.30],
                [0.05, 0.08, 0.12, 0.30, 1.0],
            ],
            dtype=float,
        )
        return MultiCcyLgmParams(
            ir_params={"EUR": eur, "USD": usd, "GBP": gbp},
            fx_vols={"EUR/USD": (tuple(), (0.15,)), "GBP/USD": (tuple(), (0.13,))},
            corr=corr,
        )

    def setUp(self):
        params = self._make_params()
        self.np_model = LgmFxHybrid(params)
        self.torch_model = TorchLgmFxHybrid(params)

    def test_factor_structure_matches_numpy_hybrid(self):
        self.assertEqual(self.torch_model.ir_ccys, self.np_model.ir_ccys)
        self.assertEqual(self.torch_model.fx_pairs, self.np_model.fx_pairs)
        self.assertEqual(self.torch_model.factor_ordering(), tuple(self.np_model.factor_ordering()))
        self.assertTrue(np.allclose(self.torch_model.corr_psd, self.np_model.corr_psd))

    def test_zc_bond_matches_numpy_hybrid(self):
        x = np.linspace(-0.02, 0.02, 11)
        got = self.torch_model.zc_bond("USD", 0.5, 2.0, x, 0.99, 0.94)
        ref = self.np_model.zc_bond("USD", 0.5, 2.0, x, 0.99, 0.94)
        self.assertTrue(np.allclose(got, ref))

    def test_simulation_matches_numpy_for_shared_draws(self):
        times = np.array([0.0, 0.25, 1.0, 2.0], dtype=float)
        n_paths = 64
        ref = self.np_model.simulate_paths(
            times,
            n_paths,
            rng=np.random.default_rng(7),
            log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)},
        )
        got = simulate_hybrid_paths_torch(
            self.torch_model,
            times,
            n_paths,
            rng=np.random.default_rng(7),
            log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)},
            return_numpy=True,
        )
        self.assertTrue(np.allclose(got["x"]["EUR"], ref["x"]["EUR"]))
        self.assertTrue(np.allclose(got["x"]["USD"], ref["x"]["USD"]))
        self.assertTrue(np.allclose(got["x"]["GBP"], ref["x"]["GBP"]))
        self.assertTrue(np.allclose(got["s"]["EUR/USD"], ref["s"]["EUR/USD"]))
        self.assertTrue(np.allclose(got["s"]["GBP/USD"], ref["s"]["GBP/USD"]))

    def test_torch_output_shapes(self):
        times = np.array([0.0, 0.25, 1.0, 2.0], dtype=float)
        out = simulate_hybrid_paths_torch(
            self.torch_model,
            times,
            32,
            rng=np.random.default_rng(5),
            log_s0={"EUR/USD": np.log(1.1), "GBP/USD": np.log(1.27)},
            return_numpy=False,
        )
        self.assertIsInstance(out["times"], torch.Tensor)
        self.assertIsInstance(out["x"]["EUR"], torch.Tensor)
        self.assertEqual(tuple(out["x"]["EUR"].shape), (4, 32))
        self.assertEqual(tuple(out["s"]["EUR/USD"].shape), (4, 32))


if __name__ == "__main__":
    unittest.main()
