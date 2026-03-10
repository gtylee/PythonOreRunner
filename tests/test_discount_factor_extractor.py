import tempfile
import unittest
from pathlib import Path

import numpy as np

from py_ore_tools.irs_xva_utils import load_ore_discount_pairs_from_curves
from py_ore_tools.ore_snapshot import (
    discount_factors_to_dataframe,
    extract_discount_factors_by_currency,
)


class TestDiscountFactorExtractor(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        input_dir = root / "Input"
        output_dir = root / "Output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        ore_xml = """<ORE>
  <Setup>
    <Parameter name="asofDate">2024-01-01</Parameter>
    <Parameter name="outputPath">Output</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
  </Setup>
  <Analytics>
    <Analytic type="curves">
      <Parameter name="configuration">default</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""
        todaysmarket_xml = """<TodaysMarket>
  <Configuration id="default">
    <DiscountingCurvesId>disc_set</DiscountingCurvesId>
  </Configuration>
  <DiscountingCurves id="disc_set">
    <DiscountingCurve currency="USD">Yield/USD/USD-OIS</DiscountingCurve>
    <DiscountingCurve currency="EUR">Yield/EUR/EONIA</DiscountingCurve>
    <DiscountingCurve currency="GBP">Yield/USD/USD-OIS</DiscountingCurve>
  </DiscountingCurves>
  <YieldCurves>
    <YieldCurve name="USD-OIS">Yield/USD/USD-OIS</YieldCurve>
    <YieldCurve name="EUR-EONIA">Yield/EUR/EONIA</YieldCurve>
  </YieldCurves>
</TodaysMarket>
"""
        curves_csv = """Date,USD-OIS,EUR-EONIA
2024-01-01,0.999,0.998
2024-01-01,0.997,0.996
2024-07-01,0.980,0.970
2025-01-01,0.960,0.940
"""
        (input_dir / "ore_measure_lgm.xml").write_text(ore_xml, encoding="utf-8")
        (input_dir / "todaysmarket.xml").write_text(todaysmarket_xml, encoding="utf-8")
        (output_dir / "curves.csv").write_text(curves_csv, encoding="utf-8")
        return input_dir / "ore_measure_lgm.xml"

    def test_extract_discount_factors_by_currency(self):
        with tempfile.TemporaryDirectory() as td:
            ore_xml = self._write_fixture(Path(td))
            payload = extract_discount_factors_by_currency(ore_xml)

            self.assertEqual(set(payload.keys()), {"USD", "EUR", "GBP"})
            self.assertEqual(payload["USD"]["source_column"], "USD-OIS")
            self.assertEqual(payload["EUR"]["source_column"], "EUR-EONIA")
            self.assertEqual(payload["GBP"]["source_column"], "USD-OIS")

            for ccy in ("USD", "EUR", "GBP"):
                times = np.asarray(payload[ccy]["times"], dtype=float)
                dfs = np.asarray(payload[ccy]["dfs"], dtype=float)
                dates = payload[ccy]["calendar_dates"]
                self.assertTrue(np.all(np.diff(times) > 0.0))
                self.assertTrue(np.all(dfs > 0.0))
                self.assertAlmostEqual(float(times[0]), 0.0)
                self.assertAlmostEqual(float(dfs[0]), 1.0)
                self.assertEqual(len(dates), len(times))
                self.assertEqual(dates[0], "2024-01-01")

            self.assertEqual(payload["USD"]["times"], payload["GBP"]["times"])
            self.assertEqual(payload["USD"]["dfs"], payload["GBP"]["dfs"])

    def test_dataframe_adapter_and_single_curve_parity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ore_xml = self._write_fixture(root)
            payload = extract_discount_factors_by_currency(ore_xml)
            df = discount_factors_to_dataframe(payload)

            self.assertEqual(
                list(df.columns),
                ["ccy", "curve_id", "time", "df", "asof_date", "source_column"],
            )
            self.assertGreater(len(df), 0)

            curves_csv = root / "Output" / "curves.csv"
            times_usd, dfs_usd = load_ore_discount_pairs_from_curves(
                str(curves_csv),
                discount_column="USD-OIS",
            )
            self.assertTrue(np.allclose(payload["USD"]["times"], times_usd))
            self.assertTrue(np.allclose(payload["USD"]["dfs"], dfs_usd))


if __name__ == "__main__":
    unittest.main()
