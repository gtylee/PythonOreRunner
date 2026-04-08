import tempfile
import unittest
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.ore_snapshot import (
    _load_ore_discount_pairs_by_columns_with_day_counter,
    _resolve_discount_column,
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

    def test_legacy_todaysmarket_schema_is_accepted(self):
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
    <Parameter name="simulation">default</Parameter>
    <Parameter name="pricing">default</Parameter>
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
      <CurveId>EUR1D</CurveId>
      <CurveDescription>EUR OIS</CurveDescription>
      <Currency>EUR</Currency>
      <DiscountCurve>EUR1D</DiscountCurve>
      <Segments>
        <Simple>
          <Type>Deposit</Type>
          <Quotes>
            <Quote>MM/RATE/EUR/EUR-EONIA/1D</Quote>
          </Quotes>
          <Conventions>EUR-CONV</Conventions>
        </Simple>
      </Segments>
    </YieldCurve>
    <YieldCurve>
      <CurveId>EUR6M</CurveId>
      <CurveDescription>EUR 6M</CurveDescription>
      <Currency>EUR</Currency>
      <DiscountCurve>EUR1D</DiscountCurve>
      <Segments>
        <Simple>
          <Type>Deposit</Type>
          <Quotes>
            <Quote>MM/RATE/EUR/EUR-EURIBOR-6M/6M</Quote>
          </Quotes>
          <Conventions>EUR-CONV</Conventions>
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
    <Id>EUR-CONV</Id>
    <IndexBased>true</IndexBased>
    <Index>EUR-EONIA</Index>
  </Deposit>
</Conventions>
""",
                encoding="utf-8",
            )
            (input_dir / "todaysmarket.xml").write_text(
                """<?xml version="1.0"?>
<TodaysMarket>
  <YieldCurves id="default">
    <YieldCurve name="EUR">Yield/EUR/EUR1D</YieldCurve>
  </YieldCurves>
  <DiscountingCurves id="default">
    <DiscountingCurve currency="EUR">Yield/EUR/EUR1D</DiscountingCurve>
  </DiscountingCurves>
  <IndexForwardingCurves id="default">
    <Index name="EUR-EURIBOR-6M">Yield/EUR/EUR6M</Index>
  </IndexForwardingCurves>
</TodaysMarket>
""",
                encoding="utf-8",
            )
            (input_dir / "market.txt").write_text(
                "\n".join(
                    [
                        "2025-01-01 MM/RATE/EUR/EUR-EONIA/1D 0.01",
                        "2025-01-01 MM/RATE/EUR/EUR-EURIBOR-6M/6M 0.015",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "fixings.txt").write_text("", encoding="utf-8")

            report = validate_ore_input_snapshot(input_dir / "ore.xml")

            self.assertTrue(report["market_configurations"]["valid"])
            self.assertIn("default", report["market_configurations"]["available"])
            self.assertTrue(report["todaysmarket_sections"]["valid"])
            self.assertTrue(report["curve_specs"]["valid"])
            self.assertEqual(report["summary"]["selected_market_configs"], ["default"])

    def test_resolve_discount_column_supports_legacy_todaysmarket_schema(self):
        tm_root = ET.fromstring(
            """<?xml version="1.0"?>
<TodaysMarket>
  <YieldCurves id="default">
    <YieldCurve name="EUR">Yield/EUR/EUR1D</YieldCurve>
  </YieldCurves>
  <DiscountingCurves id="default">
    <DiscountingCurve currency="EUR">Yield/EUR/EUR1D</DiscountingCurve>
  </DiscountingCurves>
  <IndexForwardingCurves id="default">
    <Index name="EUR-EURIBOR-6M">Yield/EUR/EUR6M</Index>
  </IndexForwardingCurves>
</TodaysMarket>
"""
        )

        self.assertEqual(_resolve_discount_column(tm_root, "default", "EUR"), "EUR")

    def test_load_discount_pairs_with_day_counter_matches_ester_alias_case_insensitively(self):
        with tempfile.TemporaryDirectory() as tmp:
            curves_csv = Path(tmp) / "curves.csv"
            curves_csv.write_text(
                "Date,EUR-ESTER\n"
                "2026-03-08,1.0\n"
                "2027-03-08,0.98\n",
                encoding="utf-8",
            )

            payload = _load_ore_discount_pairs_by_columns_with_day_counter(
                str(curves_csv),
                ["eur-estr"],
                asof_date="2026-03-08",
                day_counter="A365F",
            )

        dates, times, dfs = payload["eur-estr"]
        self.assertEqual(dates, ("2026-03-08", "2027-03-08"))
        self.assertEqual(times.tolist(), [0.0, 1.0])
        self.assertEqual(dfs.tolist(), [1.0, 0.98])

    def test_validation_does_not_force_default_or_irrelevant_index_quotes(self):
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
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
    <Parameter name="implyTodaysFixings">N</Parameter>
  </Setup>
  <Markets>
    <Parameter name="pricing">xois</Parameter>
    <Parameter name="simulation">xois</Parameter>
  </Markets>
  <Analytics>
    <Analytic type="npv">
      <Parameter name="active">Y</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
    </Analytic>
    <Analytic type="curves">
      <Parameter name="active">Y</Parameter>
      <Parameter name="configuration">xois</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            (input_dir / "portfolio.xml").write_text(
                """<?xml version="1.0"?>
<Portfolio>
  <Trade id="FXNDF">
    <TradeType>FxForward</TradeType>
    <Envelope>
      <CounterParty>CP</CounterParty>
      <NettingSetId>NS</NettingSetId>
    </Envelope>
    <FxForwardData>
      <ValueDate>2026-01-01</ValueDate>
      <BoughtCurrency>GBP</BoughtCurrency>
      <BoughtAmount>1000000</BoughtAmount>
      <SoldCurrency>USD</SoldCurrency>
      <SoldAmount>1200000</SoldAmount>
      <Settlement>Cash</Settlement>
      <SettlementData>
        <Currency>USD</Currency>
        <FXIndex>FX-ECB-GBP-USD</FXIndex>
        <Date>2026-01-05</Date>
      </SettlementData>
    </FxForwardData>
  </Trade>
</Portfolio>
""",
                encoding="utf-8",
            )
            (input_dir / "curveconfig.xml").write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<CurveConfiguration>
  <YieldCurves>
    <YieldCurve>
      <CurveId>USD1D</CurveId>
      <CurveDescription>USD OIS</CurveDescription>
      <Currency>USD</Currency>
      <DiscountCurve>USD1D</DiscountCurve>
      <Segments>
        <Simple>
          <Type>Deposit</Type>
          <Quotes>
            <Quote>MM/RATE/USD/USD-FED-FUNDS/1D</Quote>
          </Quotes>
          <Conventions>USD-CONV</Conventions>
        </Simple>
      </Segments>
    </YieldCurve>
    <YieldCurve>
      <CurveId>GBP-IN-USD</CurveId>
      <CurveDescription>GBP in USD</CurveDescription>
      <Currency>GBP</Currency>
      <DiscountCurve>USD1D</DiscountCurve>
      <Segments>
        <Simple>
          <Type>FX Forward</Type>
          <Quotes>
            <Quote>FXFWD/RATE/GBP/USD/1Y</Quote>
          </Quotes>
          <Conventions>USD-CONV</Conventions>
        </Simple>
      </Segments>
    </YieldCurve>
    <YieldCurve>
      <CurveId>USD3M</CurveId>
      <CurveDescription>USD 3M</CurveDescription>
      <Currency>USD</Currency>
      <DiscountCurve>USD1D</DiscountCurve>
      <Segments>
        <Simple>
          <Type>Deposit</Type>
          <Quotes>
            <Quote>MM/RATE/USD/USD-LIBOR-3M/3M</Quote>
          </Quotes>
          <Conventions>USD-CONV</Conventions>
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
    <Id>USD-CONV</Id>
    <IndexBased>false</IndexBased>
    <Calendar>TARGET</Calendar>
    <DayCounter>A360</DayCounter>
    <BusinessDayConvention>F</BusinessDayConvention>
    <SettlementDays>2</SettlementDays>
    <EOM>false</EOM>
  </Deposit>
</Conventions>
""",
                encoding="utf-8",
            )
            (input_dir / "todaysmarket.xml").write_text(
                """<?xml version="1.0"?>
<TodaysMarket>
  <Configuration id="xois">
    <DiscountingCurvesId>xois</DiscountingCurvesId>
    <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
  </Configuration>
  <DiscountingCurves id="xois">
    <DiscountingCurve currency="USD">Yield/USD/USD1D</DiscountingCurve>
    <DiscountingCurve currency="GBP">Yield/GBP/GBP-IN-USD</DiscountingCurve>
  </DiscountingCurves>
  <IndexForwardingCurves id="default">
    <Index name="USD-LIBOR-3M">Yield/USD/USD3M</Index>
  </IndexForwardingCurves>
  <FxSpots id="default">
    <FxSpot pair="GBPUSD">FX/GBP/USD</FxSpot>
  </FxSpots>
</TodaysMarket>
""",
                encoding="utf-8",
            )
            (input_dir / "market.txt").write_text(
                "\n".join(
                    [
                        "2025-01-01 MM/RATE/USD/USD-FED-FUNDS/1D 0.01",
                        "2025-01-01 FXFWD/RATE/GBP/USD/1Y 1.25",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "fixings.txt").write_text("", encoding="utf-8")

            report = validate_ore_input_snapshot(input_dir / "ore.xml")

            self.assertTrue(report["market_configurations"]["valid"])
            self.assertEqual(report["market_configurations"]["requested"], ["xois"])
            self.assertTrue(report["quotes"]["valid"])
            self.assertEqual(report["quotes"]["missing_mandatory"], [])


if __name__ == "__main__":
    unittest.main()
