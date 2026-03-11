import tempfile
import unittest
from pathlib import Path

import numpy as np

from py_ore_tools.ore_snapshot import (
    extract_market_instruments_by_currency,
    extract_market_instruments_by_currency_from_quotes,
    fit_discount_curves_from_ore_market,
    fit_discount_curves_from_programmatic_quotes,
    fitted_curves_to_dataframe,
    quote_dicts_from_pairs,
)


class TestQuickCurveFit(unittest.TestCase):
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
    <Parameter name="marketDataFile">market_data.txt</Parameter>
  </Setup>
</ORE>
"""
        todaysmarket_xml = "<TodaysMarket/>"
        market_data_txt = """20240101 ZERO/RATE/USD/1Y 0.0500
20240101 ZERO/RATE/USD/2Y 0.0480
20240101 MM/RATE/USD/USD-LIBOR/3M 0.0520
20240101 IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/5Y 0.0470
20240101 ZERO/RATE/EUR/1Y 0.0300
20240101 IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/5Y 0.0320
"""
        (input_dir / "ore_measure_lgm.xml").write_text(ore_xml, encoding="utf-8")
        (input_dir / "todaysmarket.xml").write_text(todaysmarket_xml, encoding="utf-8")
        (input_dir / "market_data.txt").write_text(market_data_txt, encoding="utf-8")
        return input_dir / "ore_measure_lgm.xml"

    def test_extract_and_fit(self):
        with tempfile.TemporaryDirectory() as td:
            ore_xml = self._write_fixture(Path(td))

            instruments = extract_market_instruments_by_currency(ore_xml)
            self.assertEqual(set(instruments.keys()), {"USD", "EUR"})
            self.assertEqual(instruments["USD"]["instrument_count"], 4)
            self.assertEqual(instruments["EUR"]["instrument_count"], 2)

            fitted = fit_discount_curves_from_ore_market(ore_xml)
            self.assertEqual(set(fitted.keys()), {"USD", "EUR"})
            for ccy in ("USD", "EUR"):
                p = fitted[ccy]
                times = np.asarray(p["times"], dtype=float)
                dfs = np.asarray(p["dfs"], dtype=float)
                zeros = np.asarray(p["zero_rates"], dtype=float)
                dates = p["calendar_dates"]
                self.assertEqual(len(times), len(dfs))
                self.assertEqual(len(times), len(zeros))
                self.assertEqual(len(times), len(dates))
                self.assertTrue(np.all(np.diff(times) > 0.0))
                self.assertTrue(np.all(dfs > 0.0))
                self.assertTrue(np.all(np.diff(dfs) <= 1.0e-12))
                self.assertEqual(p["instrument_count"], instruments[ccy]["instrument_count"])
                self.assertEqual(len(p["instrument_times"]), len(p["instrument_zero_rates"]))

            df = fitted_curves_to_dataframe(fitted)
            self.assertGreater(len(df), 0)
            self.assertIn("calendar_date", df.columns)
            self.assertIn("instrument_count", df.columns)

            fitted_dense = fit_discount_curves_from_ore_market(
                ore_xml,
                fit_grid_mode="dense",
                dense_step_years=0.25,
            )
            self.assertEqual(set(fitted_dense.keys()), {"USD", "EUR"})
            for ccy in ("USD", "EUR"):
                self.assertGreaterEqual(
                    fitted_dense[ccy]["fit_points_count"],
                    fitted[ccy]["fit_points_count"],
                )

            fitted_bootstrap = fit_discount_curves_from_ore_market(
                ore_xml,
                fit_method="bootstrap_mm_irs_v1",
                fit_grid_mode="instrument",
            )
            self.assertEqual(set(fitted_bootstrap.keys()), {"USD", "EUR"})
            for ccy in ("USD", "EUR"):
                p = fitted_bootstrap[ccy]
                self.assertEqual(p["curve_method"], "bootstrap_mm_irs_v1")
                self.assertGreater(p["fit_points_count"], 1)
                self.assertTrue(np.all(np.asarray(p["dfs"], dtype=float) > 0.0))

    def test_programmatic_curve_fit_no_xml(self):
        quotes = [
            {"key": "ZERO/RATE/USD/1Y", "value": 0.05},
            {"key": "MM/RATE/USD/USD-LIBOR/3M", "value": 0.051},
            {"key": "IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/5Y", "value": 0.047},
            {"key": "ZERO/RATE/EUR/1Y", "value": 0.03},
            {"key": "IR_SWAP/RATE/EUR/EUR-EURIBOR-6M/6M/5Y", "value": 0.032},
        ]

        inst = extract_market_instruments_by_currency_from_quotes(
            asof_date="2024-01-01",
            quotes=quotes,
        )
        self.assertEqual(set(inst.keys()), {"USD", "EUR"})
        self.assertEqual(inst["USD"]["instrument_count"], 3)

        fitted = fit_discount_curves_from_programmatic_quotes(
            asof_date="2024-01-01",
            quotes=quotes,
            fit_method="bootstrap_mm_irs_v1",
            fit_grid_mode="dense",
            dense_step_years=0.5,
        )
        self.assertEqual(set(fitted.keys()), {"USD", "EUR"})
        for ccy in ("USD", "EUR"):
            p = fitted[ccy]
            self.assertGreater(p["fit_points_count"], 1)
            self.assertEqual(len(p["times"]), len(p["calendar_dates"]))

    def test_programmatic_quote_tuple_helper(self):
        pairs = [
            ("ZERO/RATE/USD/1Y", 0.05),
            ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/5Y", 0.047),
        ]
        quote_dicts = quote_dicts_from_pairs(pairs)
        self.assertEqual(len(quote_dicts), 2)
        fitted = fit_discount_curves_from_programmatic_quotes(
            asof_date="2024-01-01",
            quotes=pairs,  # tuples accepted directly
            fit_method="weighted_zero_logdf_v1",
            fit_grid_mode="instrument",
        )
        self.assertIn("USD", fitted)

    def test_quantlib_helper_eur_curve_bump_keeps_10y_forward_direction(self):
        quotes = [
            ("MM/RATE/EUR/0D/1D", 0.02),
            ("IR_SWAP/RATE/EUR/2D/1D/1Y", 0.02),
            ("IR_SWAP/RATE/EUR/2D/1D/2Y", 0.02),
            ("IR_SWAP/RATE/EUR/2D/1D/5Y", 0.02),
            ("IR_SWAP/RATE/EUR/2D/1D/10Y", 0.02),
            ("MM/RATE/EUR/2D/6M", 0.02),
            ("FRA/RATE/EUR/1M/6M", 0.02),
            ("FRA/RATE/EUR/3M/6M", 0.02),
            ("FRA/RATE/EUR/6M/6M", 0.02),
            ("IR_SWAP/RATE/EUR/2D/6M/2Y", 0.02),
            ("IR_SWAP/RATE/EUR/2D/6M/5Y", 0.02),
            ("IR_SWAP/RATE/EUR/2D/6M/10Y", 0.02),
            ("IR_SWAP/RATE/EUR/2D/6M/11Y", 0.02),
        ]

        base = fit_discount_curves_from_programmatic_quotes(
            asof_date="2016-02-05",
            quotes=quotes,
            fit_method="ql_helper_eur_v1",
            fit_grid_mode="instrument",
        )["EUR"]
        bumped_quotes = [
            (k, v + 1.0e-4 if k == "IR_SWAP/RATE/EUR/2D/6M/10Y" else v)
            for k, v in quotes
        ]
        bumped = fit_discount_curves_from_programmatic_quotes(
            asof_date="2016-02-05",
            quotes=bumped_quotes,
            fit_method="ql_helper_eur_v1",
            fit_grid_mode="instrument",
        )["EUR"]

        base_dfs = np.asarray(base["dfs"], dtype=float)
        bumped_dfs = np.asarray(bumped["dfs"], dtype=float)

        self.assertTrue(np.all(np.diff(base_dfs) <= 1.0e-12))
        self.assertTrue(np.all(np.diff(bumped_dfs) <= 1.0e-12))
        self.assertGreater(len(base["times"]), 1)
        self.assertEqual(len(base["times"]), len(bumped["times"]))


if __name__ == "__main__":
    unittest.main()
