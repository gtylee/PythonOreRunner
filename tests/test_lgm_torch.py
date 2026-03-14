import unittest

import numpy as np

from py_ore_tools.lgm import LGM1F, LGMParams

try:
    import torch
except ImportError:  # pragma: no cover - torch-specific tests skip without torch
    torch = None

if torch is not None:
    from py_ore_tools.lgm_torch import TorchLGM1F, simulate_lgm_measure_torch


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


if __name__ == "__main__":
    unittest.main()
