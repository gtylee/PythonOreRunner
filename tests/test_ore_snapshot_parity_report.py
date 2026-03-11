import unittest
from pathlib import Path
import sys

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
