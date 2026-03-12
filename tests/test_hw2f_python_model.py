import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.hw2f import HW2FModel, HW2FParams, parse_hw2f_params_from_simulation_xml, simulate_hw_ba_euler
from py_ore_tools.hw2f_integration import compare_hw2f_python_vs_ore, price_hw2f_swap_t0
from py_ore_tools.hw2f_ore_runner import build_hw2f_case, run_ore_case


class TestHw2FPythonModel(unittest.TestCase):
    def test_discount_bond_identity_and_shapes(self):
        model = HW2FModel(
            HW2FParams(
                times=(),
                sigma=((((0.002, 0.008), (0.009, 0.001))),),
                kappa=((0.01, 0.2),),
            )
        )
        x0 = np.zeros((3, 2), dtype=float)
        p = model.discount_bond(2.0, 2.0, x0, 0.97, 0.97)
        self.assertTrue(np.allclose(p, 1.0))
        self.assertEqual(model.y(1.0).shape, (2, 2))
        self.assertEqual(model.g(1.0, 4.0).shape, (2,))

    def test_parse_and_simulate_generated_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = build_hw2f_case(
                Path(tmp) / "case" / "Input",
                sigma=[[[0.002, 0.008], [0.009, 0.001]]],
                kappa=[[0.01, 0.2]],
                times=[],
                samples=8,
            )
            params = parse_hw2f_params_from_simulation_xml(case_paths.simulation_xml)
            model = HW2FModel(params)
            times = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=float)
            x_paths, aux_paths = simulate_hw_ba_euler(model, times, n_paths=16, rng=np.random.default_rng(7))
            self.assertEqual(x_paths.shape, (4, 16, 2))
            self.assertEqual(aux_paths.shape, (4, 16, 2))
            self.assertTrue(np.all(np.isfinite(x_paths)))
            self.assertTrue(np.all(np.isfinite(aux_paths)))

    def test_python_t0_npv_tracks_ore_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_root = Path(tmp) / "case"
            build_hw2f_case(
                case_root / "Input",
                sigma=[[[0.002, 0.008], [0.009, 0.001]]],
                kappa=[[0.01, 0.2]],
                times=[],
                samples=16,
                grid="12,1Y",
            )
            run_ore_case(case_root)
            py_npv = price_hw2f_swap_t0(case_root)
            cmp = compare_hw2f_python_vs_ore(case_root)
            self.assertTrue(math.isfinite(py_npv))
            self.assertTrue(math.isfinite(cmp["ore_t0_npv"]))
            self.assertLess(abs(cmp["rel_diff"]), 0.15)


if __name__ == "__main__":
    unittest.main()
