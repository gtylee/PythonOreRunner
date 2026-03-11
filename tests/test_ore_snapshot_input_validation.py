import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.ore_snapshot import (
    ore_input_validation_dataframe,
    validate_ore_input_snapshot,
)


class TestOreSnapshotInputValidation(unittest.TestCase):
    def test_input_validation_on_real_case(self):
        ore_xml = (
            Path(__file__).resolve().parents[1]
            / "parity_artifacts"
            / "multiccy_benchmark_final"
            / "cases"
            / "flat_EUR_5Y_A"
            / "Input"
            / "ore.xml"
        )
        report = validate_ore_input_snapshot(ore_xml)

        self.assertTrue(report["market_configurations"]["valid"])
        self.assertTrue(report["todaysmarket_sections"]["valid"])
        self.assertTrue(report["curve_specs"]["valid"])
        self.assertTrue(report["conventions"]["valid"])
        self.assertTrue(report["todaysmarket_names"]["distinct"])
        self.assertEqual(
            report["curve_specs"]["quote_scope_curve_ids_by_section"]["YieldCurves"],
            ["EUR1D", "EUR6M"],
        )
        self.assertGreater(len(report["quotes"]["missing_mandatory"]), 0)
        self.assertFalse(report["input_links_valid"])

    def test_validation_dataframe_shape(self):
        ore_xml = (
            Path(__file__).resolve().parents[1]
            / "parity_artifacts"
            / "multiccy_benchmark_final"
            / "cases"
            / "flat_EUR_5Y_A"
            / "Input"
            / "ore.xml"
        )
        report = validate_ore_input_snapshot(ore_xml)
        df = ore_input_validation_dataframe(report)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["section", "field", "value"])
        self.assertTrue(((df["section"] == "checks") & (df["field"] == "curve_specs_resolve")).any())

    def test_input_validation_flags_missing_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()

            (input_dir / "ore.xml").write_text(
                """<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">2025-01-01</Parameter>
    <Parameter name="marketDataFile">market.txt</Parameter>
    <Parameter name="fixingDataFile">fixings.txt</Parameter>
    <Parameter name="curveConfigFile">curveconfig.xml</Parameter>
    <Parameter name="conventionsFile">conventions.xml</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
    <Parameter name="implyTodaysFixings">N</Parameter>
  </Setup>
  <Markets>
    <Parameter name="simulation">simcfg</Parameter>
    <Parameter name="sensitivity">missing_cfg</Parameter>
  </Markets>
  <Analytics>
    <Analytic type="curves">
      <Parameter name="configuration">default</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            (input_dir / "curveconfig.xml").write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<CurveConfiguration>
  <YieldCurves>
    <YieldCurve>
      <CurveId>USD3M</CurveId>
      <CurveDescription>USD 3M</CurveDescription>
      <Currency>USD</Currency>
      <DiscountCurve>USD3M</DiscountCurve>
      <Segments>
        <Simple>
          <Type>Deposit</Type>
          <Quotes>
            <Quote>MM/RATE/USD/USD-LIBOR/3M</Quote>
          </Quotes>
          <Conventions>USD-LIBOR-CONVENTIONS-MISSING</Conventions>
        </Simple>
      </Segments>
    </YieldCurve>
  </YieldCurves>
</CurveConfiguration>
""",
                encoding="utf-8",
            )
            (input_dir / "conventions.xml").write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<Conventions>
  <Deposit>
    <Id>USD-LIBOR-CONVENTIONS</Id>
    <IndexBased>true</IndexBased>
    <Index>USD-LIBOR</Index>
  </Deposit>
</Conventions>
""",
                encoding="utf-8",
            )
            (input_dir / "todaysmarket.xml").write_text(
                """<?xml version="1.0"?>
<TodaysMarket>
  <Configuration id="default">
    <DiscountingCurvesId>default</DiscountingCurvesId>
    <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
  </Configuration>
  <Configuration id="simcfg">
    <DiscountingCurvesId>default</DiscountingCurvesId>
    <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
  </Configuration>
  <YieldCurves id="default">
    <YieldCurve name="USD-LIBOR-3M">Yield/USD/USD3M</YieldCurve>
  </YieldCurves>
  <IndexForwardingCurves id="default">
    <Index name="USD-LIBOR-3M">Yield/USD/MISSING</Index>
  </IndexForwardingCurves>
  <DiscountingCurves id="default">
    <DiscountingCurve currency="USD">Yield/USD/USD3M</DiscountingCurve>
  </DiscountingCurves>
  <FxSpots id="default">
    <FxSpot pair="EURUSD">FX/EUR/USD</FxSpot>
  </FxSpots>
</TodaysMarket>
""",
                encoding="utf-8",
            )
            (input_dir / "market.txt").write_text(
                "\n".join(
                    [
                        "2025-01-01 FX/RATE/EUR/USD 1.10",
                        "2025-01-01 FX/RATE/USD/EUR 0.91",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "fixings.txt").write_text(
                "2025-01-01 USD-LIBOR-3M 0.05\n",
                encoding="utf-8",
            )

            report = validate_ore_input_snapshot(input_dir / "ore.xml")

            self.assertFalse(report["market_configurations"]["valid"])
            self.assertIn("missing_cfg", report["market_configurations"]["missing_requested"])
            self.assertTrue(any(item["code"] == "missing_market_configurations" for item in report["action_items"]))
            self.assertFalse(report["curve_specs"]["valid"])
            self.assertEqual(report["curve_specs"]["invalid"][0]["spec"], "Yield/USD/MISSING")
            self.assertFalse(report["conventions"]["valid"])
            self.assertIn("USD-LIBOR-CONVENTIONS-MISSING", report["conventions"]["missing_ids"])
            self.assertFalse(report["quotes"]["valid"])
            self.assertIn("MM/RATE/USD/USD-LIBOR/3M", report["quotes"]["missing_mandatory"])
            quote_item = next(item for item in report["action_items"] if item["code"] == "missing_mandatory_quotes")
            self.assertIn("Add these quote ids", quote_item["what_to_fix"])
            self.assertFalse(report["todaysmarket_names"]["distinct"])
            self.assertEqual(report["todaysmarket_names"]["yield_index_overlaps"][0]["name"], "USD-LIBOR-3M")
            self.assertTrue(report["fixings"]["potential_issue"])
            self.assertEqual(report["fx_dominance"]["pair_count_with_both_directions"], 1)
            self.assertEqual(
                report["fx_dominance"]["pairs_with_both_directions"][0]["kept"],
                "FX/RATE/EUR/USD",
            )
            self.assertFalse(report["input_links_valid"])


if __name__ == "__main__":
    unittest.main()
