import unittest

import numpy as np

from py_ore_tools import (
    RateFutureModelParams,
    extract_market_instruments_by_currency_from_quotes,
    fit_discount_curves_from_programmatic_quotes,
)


class TestRateFutureCurveFit(unittest.TestCase):
    def test_extract_programmatic_mm_future_metadata(self):
        quotes = [
            {
                "key": "MM_FUTURE/PRICE/USD/2020-08/ED/3M",
                "value": 99.73,
                "contract_start": "2020-08-19",
                "contract_end": "2020-11-19",
                "convexity_adjustment": 0.0010,
            }
        ]

        extracted = extract_market_instruments_by_currency_from_quotes(
            asof_date="2020-05-15",
            quotes=quotes,
            instrument_types=("FUTURE",),
        )

        self.assertIn("USD", extracted)
        inst = extracted["USD"]["instruments"][0]
        self.assertEqual(inst["instrument_type"], "MM_FUTURE")
        self.assertEqual(inst["contract_start"], "2020-08-19")
        self.assertEqual(inst["contract_end"], "2020-11-19")
        self.assertAlmostEqual(float(inst["convexity_adjustment"]), 0.0010)
        self.assertGreater(float(inst["maturity"]), float(inst["start_time"]))

    def test_bootstrap_future_modes_share_curve_shape(self):
        quotes = [
            ("MM/RATE/USD/USD-LIBOR/1M", 0.0200),
            ("MM/RATE/USD/USD-LIBOR/3M", 0.0210),
            {
                "key": "MM_FUTURE/PRICE/USD/2020-08/ED/3M",
                "value": 99.73,
                "contract_start": "2020-08-19",
                "contract_end": "2020-11-19",
                "convexity_adjustment": 0.0010,
            },
            ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/2Y", 0.0240),
        ]

        external = fit_discount_curves_from_programmatic_quotes(
            asof_date="2020-05-15",
            quotes=quotes,
            instrument_types=("MM", "IR_SWAP", "FUTURE"),
            fit_method="bootstrap_mm_irs_v1",
            fit_grid_mode="instrument",
            future_convexity_mode="external_adjusted_fra",
        )["USD"]
        native = fit_discount_curves_from_programmatic_quotes(
            asof_date="2020-05-15",
            quotes=quotes,
            instrument_types=("MM", "IR_SWAP", "FUTURE"),
            fit_method="bootstrap_mm_irs_v1",
            fit_grid_mode="instrument",
            future_convexity_mode="native_future",
        )["USD"]

        self.assertTrue(np.allclose(external["times"], native["times"]))
        self.assertTrue(np.allclose(external["dfs"], native["dfs"]))
        self.assertEqual(external["bootstrap_diagnostics"][0]["pricing_mode"], "external_adjusted_fra")
        self.assertEqual(native["bootstrap_diagnostics"][0]["pricing_mode"], "native_future")
        self.assertAlmostEqual(
            external["bootstrap_diagnostics"][0]["adjusted_forward_rate"],
            native["bootstrap_diagnostics"][0]["adjusted_forward_rate"],
        )

    def test_bootstrap_future_model_convexity_is_positive(self):
        quotes = [
            ("MM/RATE/USD/USD-LIBOR/1M", 0.0200),
            ("MM/RATE/USD/USD-LIBOR/3M", 0.0210),
            ("MM_FUTURE/PRICE/USD/2020-08/ED/3M", 99.73),
            ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/2Y", 0.0240),
        ]

        fitted = fit_discount_curves_from_programmatic_quotes(
            asof_date="2020-05-15",
            quotes=quotes,
            instrument_types=("MM", "IR_SWAP", "FUTURE"),
            fit_method="bootstrap_mm_irs_v1",
            fit_grid_mode="instrument",
            future_convexity_mode="native_future",
            future_model_params=RateFutureModelParams(model="hw", mean_reversion=0.03, volatility=0.01),
        )["USD"]

        diag = fitted["bootstrap_diagnostics"][0]
        self.assertGreater(diag["convexity_adjustment"], 0.0)
        self.assertLess(diag["adjusted_forward_rate"], diag["futures_rate"])


if __name__ == "__main__":
    unittest.main()
