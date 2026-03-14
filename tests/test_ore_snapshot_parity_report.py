import unittest
from pathlib import Path
import sys

import pandas as pd

from py_ore_tools.repo_paths import require_engine_repo_root

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.ore_snapshot import load_from_ore_xml


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
        self.assertEqual(report["summary"]["leg_source"], "flows")
        self.assertIn("CVA", report["summary"]["requested_xva_metrics"])
        self.assertTrue(report["curve_setup"]["complete"])
        self.assertTrue(report["credit_setup"]["counterparty_credit_complete"])
        self.assertTrue(report["comparability"]["CVA"])
        self.assertTrue(report["comparability"]["DVA"])
        self.assertTrue(report["files"]["flows_csv"]["exists"])
        self.assertEqual(snap.leg_source, "flows")
        self.assertIsNotNone(snap.flows_csv_path)

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


if __name__ == "__main__":
    unittest.main()
