import math
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.hw2f_ore_runner import build_hw2f_case, read_ore_results, run_ore_case


class TestHw2FOreRunner(unittest.TestCase):
    def test_build_constant_case_writes_hwmodel(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = build_hw2f_case(
                Path(tmp) / "case" / "Input",
                sigma=[[[0.002, 0.008], [0.009, 0.001]]],
                kappa=[[0.01, 0.2]],
                times=[],
                samples=8,
            )

            root = ET.parse(case_paths.simulation_xml).getroot()
            self.assertIsNotNone(root.find("./CrossAssetModel/InterestRateModels/HWModel"))
            self.assertIsNone(root.find("./CrossAssetModel/InterestRateModels/LGM"))

            sigma_rows = root.findall("./CrossAssetModel/InterestRateModels/HWModel/Volatility/InitialValue/Sigma/Row")
            self.assertEqual(len(sigma_rows), 2)
            self.assertEqual([len(row.text.split(",")) for row in sigma_rows], [2, 2])

            kappas = root.findall("./CrossAssetModel/InterestRateModels/HWModel/Reversion/InitialValue/Kappa")
            self.assertEqual(len(kappas), 1)
            self.assertEqual(len(kappas[0].text.split(",")), 2)

    def test_build_piecewise_case_writes_matching_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = build_hw2f_case(
                Path(tmp) / "case" / "Input",
                sigma=[
                    [[0.002, 0.008], [0.009, 0.001]],
                    [[0.003, 0.007], [0.008, 0.002]],
                ],
                kappa=[[0.01, 0.2], [0.015, 0.18]],
                times=[5.0],
                samples=8,
            )

            root = ET.parse(case_paths.simulation_xml).getroot()
            hw_model = root.find("./CrossAssetModel/InterestRateModels/HWModel")
            self.assertEqual(hw_model.findtext("./Volatility/ParamType"), "Piecewise")
            self.assertEqual(hw_model.findtext("./Reversion/ParamType"), "Piecewise")
            self.assertEqual(hw_model.findtext("./Volatility/TimeGrid"), "5")
            self.assertEqual(len(hw_model.findall("./Volatility/InitialValue/Sigma")), 2)
            self.assertEqual(len(hw_model.findall("./Reversion/InitialValue/Kappa")), 2)

    def test_run_case_produces_finite_npv_and_stable_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_root = Path(tmp) / "case"
            build_hw2f_case(
                case_root / "Input",
                sigma=[[[0.002, 0.008], [0.009, 0.001]]],
                kappa=[[0.01, 0.2]],
                times=[],
                samples=8,
                grid="12,1Y",
            )

            run_ore_case(case_root)
            first_results = read_ore_results(case_root)
            first_npv = first_results["npv"]["base_npv"]
            self.assertIsNotNone(first_npv)
            self.assertTrue(math.isfinite(first_npv))
            self.assertTrue((case_root / "Output" / "npv.csv").exists())
            self.assertTrue((case_root / "Output" / "xva.csv").exists())

            first_snapshot = shutil.copyfile(
                case_root / "Output" / "npv.csv",
                case_root / "Output" / "npv_first.csv",
            )
            self.assertTrue(Path(first_snapshot).exists())

            run_ore_case(case_root)
            second_results = read_ore_results(case_root)
            second_npv = second_results["npv"]["base_npv"]
            self.assertIsNotNone(second_npv)
            self.assertAlmostEqual(first_npv, second_npv, places=10)
            self.assertGreaterEqual(len(second_results["trade_exposure_files"]), 1)


if __name__ == "__main__":
    unittest.main()
