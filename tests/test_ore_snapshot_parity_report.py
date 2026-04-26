import unittest
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd

from py_ore_tools.repo_paths import require_engine_repo_root

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.ore_snapshot import load_from_ore_xml
from test_ore_snapshot_cli import _clone_example28_eur_base_case


class TestOreSnapshotParityReport(unittest.TestCase):
    def test_parity_completeness_report_on_real_case(self):
        ore_xml = (
            Path(__file__).resolve().parents[1]
            / "parity_artifacts"
            / "multiccy_benchmark_final"
            / "cases"
            / "flat_EUR_5Y_A"
            / "Input"
            / "ore.xml"
        )
        snap = load_from_ore_xml(ore_xml)
        report = snap.parity_completeness_report()

        self.assertEqual(report["summary"]["trade_id"], "SWAP_EUR_5Y_A_flat")
        self.assertIn(report["summary"]["leg_source"], {"portfolio", "flows"})
        self.assertIn("CVA", report["summary"]["requested_xva_metrics"])
        self.assertIn("DVA", report["summary"]["requested_xva_metrics"])
        self.assertTrue(report["curve_setup"]["complete"])
        self.assertTrue(report["credit_setup"]["counterparty_credit_complete"])
        self.assertTrue(report["comparability"]["CVA"])
        self.assertTrue(report["comparability"]["DVA"])
        self.assertTrue(report["files"]["flows_csv"]["exists"])
        self.assertIn(snap.leg_source, {"portfolio", "flows"})

    def test_parity_completeness_dataframe_shape(self):
        ore_xml = (
            Path(__file__).resolve().parents[1]
            / "parity_artifacts"
            / "multiccy_benchmark_final"
            / "cases"
            / "flat_EUR_5Y_A"
            / "Input"
            / "ore.xml"
        )
        snap = load_from_ore_xml(ore_xml)
        df = snap.parity_completeness_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["section", "field", "value"])
        self.assertTrue(((df["section"] == "comparability") & (df["field"] == "CVA")).any())

    def test_load_from_ore_xml_prefers_calibration_output_when_available(self):
        ore_xml = (
            require_engine_repo_root()
            / "Examples"
            / "Exposure"
            / "Input"
            / "ore_measure_lgm_with_calibration.xml"
        )
        snap = load_from_ore_xml(ore_xml)

        self.assertEqual(snap.alpha_source, "calibration")
        self.assertIsNotNone(snap.calibration_xml_path)
        self.assertAlmostEqual(snap.lgm_params.shift, 0.0, places=12)

    def test_load_from_ore_xml_reads_cross_asset_measure_and_parameter_counts(self):
        ore_xml = (
            require_engine_repo_root()
            / "Examples"
            / "Exposure"
            / "Input"
            / "ore_measure_ba.xml"
        )
        snap = load_from_ore_xml(ore_xml)

        self.assertEqual(snap.measure, "BA")
        self.assertEqual(snap.seed, 42)
        self.assertEqual(snap.n_samples, 1000)

    def test_load_from_ore_xml_falls_back_to_matching_example_calibration(self):
        ore_xml = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_measure_ba.xml"
        snap = load_from_ore_xml(ore_xml)

        self.assertEqual(snap.alpha_source, "calibration")
        self.assertIsNotNone(snap.calibration_xml_path)
        self.assertTrue(str(snap.calibration_xml_path).endswith("Examples/Exposure/Output/measure_lgm/calibration.xml"))

    def test_load_from_ore_xml_falls_back_to_matching_cross_example_calibration(self):
        ore_xml = TOOLS_DIR / "Examples" / "Legacy" / "Example_1" / "Input" / "ore.xml"
        snap = load_from_ore_xml(ore_xml)

        self.assertEqual(snap.alpha_source, "calibration")
        self.assertIsNotNone(snap.calibration_xml_path)
        self.assertTrue(str(snap.calibration_xml_path).endswith("Examples/Exposure/Output/measure_lgm/calibration.xml"))

    def test_load_from_ore_xml_falls_back_to_matching_calibration_for_cloned_case_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copytree(TOOLS_DIR / "Examples" / "Exposure", tmp / "Examples" / "Exposure")
            shutil.copytree(TOOLS_DIR / "Examples" / "Input", tmp / "Examples" / "Input")
            ore_xml = tmp / "Examples" / "Exposure" / "Input" / "ore_measure_ba.xml"

            snap = load_from_ore_xml(ore_xml)

            self.assertEqual(snap.alpha_source, "calibration")
            self.assertIsNotNone(snap.calibration_xml_path)
            self.assertTrue(str(snap.calibration_xml_path).endswith("Examples/Exposure/Output/measure_lgm/calibration.xml"))

    def test_load_from_ore_xml_cloned_case_with_rewritten_samples_keeps_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copytree(TOOLS_DIR / "Examples" / "Exposure", tmp / "Examples" / "Exposure")
            shutil.copytree(TOOLS_DIR / "Examples" / "Input", tmp / "Examples" / "Input")
            simulation_xml = tmp / "Examples" / "Exposure" / "Input" / "simulation_ba.xml"
            simulation_xml.write_text(
                simulation_xml.read_text(encoding="utf-8").replace("<Samples>1000</Samples>", "<Samples>2000</Samples>"),
                encoding="utf-8",
            )
            ore_xml = tmp / "Examples" / "Exposure" / "Input" / "ore_measure_ba.xml"

            snap = load_from_ore_xml(ore_xml)

            self.assertEqual(snap.n_samples, 2000)
            self.assertEqual(snap.alpha_source, "calibration")
            self.assertIsNotNone(snap.calibration_xml_path)
            self.assertTrue(str(snap.calibration_xml_path).endswith("Examples/Exposure/Output/measure_lgm/calibration.xml"))

    def test_load_from_ore_xml_cloned_cross_example_with_rewritten_samples_keeps_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copytree(TOOLS_DIR / "Examples" / "Legacy" / "Example_1", tmp / "Examples" / "Legacy" / "Example_1")
            shutil.copytree(TOOLS_DIR / "Examples" / "Input", tmp / "Examples" / "Input")
            simulation_xml = tmp / "Examples" / "Legacy" / "Example_1" / "Input" / "simulation.xml"
            simulation_xml.write_text(
                simulation_xml.read_text(encoding="utf-8").replace("<Samples>1000</Samples>", "<Samples>2000</Samples>"),
                encoding="utf-8",
            )
            ore_xml = tmp / "Examples" / "Legacy" / "Example_1" / "Input" / "ore.xml"

            snap = load_from_ore_xml(ore_xml)

            self.assertEqual(snap.n_samples, 2000)
            self.assertEqual(snap.alpha_source, "calibration")
            self.assertIsNotNone(snap.calibration_xml_path)
            self.assertTrue(str(snap.calibration_xml_path).endswith("Examples/Exposure/Output/measure_lgm/calibration.xml"))

    def test_parity_completeness_report_supports_generic_price_only_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ore_xml = _clone_example28_eur_base_case(Path(tmpdir))

            snap = load_from_ore_xml(ore_xml, trade_id="FXFWD_1Y")
            report = snap.parity_completeness_report()

            self.assertEqual(report["summary"]["trade_id"], "FXFWD_1Y")
            self.assertEqual(report["summary"]["trade_type"], "FxForward")
            self.assertTrue(report["comparability"]["NPV"])
            self.assertFalse(report["comparability"]["CVA"])
            self.assertTrue(report["parity_ready"])


if __name__ == "__main__":
    unittest.main()
