import unittest
from pathlib import Path

import numpy as np

from ore_curve_fit_parity.curve_trace import (
    trace_curve_graph_from_ore,
    trace_discount_curve_from_ore,
    trace_index_curve_from_ore,
    trace_usd_curve_from_ore,
)
from ore_curve_fit_parity.interpolation import build_cubic_discount_interpolator, build_log_linear_discount_interpolator


class TestUsdCurveTrace(unittest.TestCase):
    ORE_XML = Path(
        "/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/"
        "multiccy_benchmark_final/cases/flat_USD_5Y_B/Input/ore.xml"
    )

    def test_log_linear_discount_interpolator(self):
        curve = build_log_linear_discount_interpolator(
            [0.0, 1.0, 2.0],
            [1.0, 0.95, 0.90],
        )

        self.assertAlmostEqual(curve(0.0), 1.0)
        self.assertAlmostEqual(curve(1.0), 0.95)
        self.assertAlmostEqual(curve(2.0), 0.90)

        expected_mid = (1.0 * 0.95) ** 0.5
        expected_right = float(np.exp(np.log(0.90) + (np.log(0.90) - np.log(0.95)) * 3.0))
        self.assertAlmostEqual(curve(0.5), expected_mid, places=12)
        self.assertAlmostEqual(curve(-1.0), 1.0)
        self.assertAlmostEqual(curve(5.0), expected_right, places=12)

    def test_cubic_discount_interpolator_differs_from_loglinear_on_long_tail(self):
        loglinear = build_log_linear_discount_interpolator(
            [0.0, 5.0, 10.0, 20.0],
            [1.0, 0.93, 0.84, 0.68],
        )
        cubic = build_cubic_discount_interpolator(
            [0.0, 5.0, 10.0, 20.0],
            [1.0, 0.93, 0.84, 0.68],
        )

        ll_30 = loglinear(30.0)
        cubic_30 = cubic(30.0)
        self.assertNotAlmostEqual(ll_30, cubic_30, places=8)
        self.assertGreater(abs(cubic_30 - ll_30), 1.0e-4)

    def test_trace_existing_usd_case(self):
        payload = trace_usd_curve_from_ore(self.ORE_XML)

        self.assertEqual(payload["currency"], "USD")
        self.assertEqual(payload["configuration_id"], "libor")
        self.assertEqual(payload["curve_handle"], "Yield/USD/USD3M")
        self.assertEqual(payload["curve_name"], "USD-LIBOR-3M")
        self.assertEqual(payload["curve_config"]["curve_id"], "USD3M")
        self.assertEqual(payload["curve_config"]["discount_curve"], "USD1D")
        self.assertEqual(len(payload["curve_config"]["segments"]), 3)
        self.assertEqual(
            [segment["type"] for segment in payload["curve_config"]["segments"]],
            ["Deposit", "FRA", "Swap"],
        )
        self.assertEqual(payload["discount_curve_dependency"]["curve_id"], "USD1D")
        self.assertIn("USD-3M-SWAP-CONVENTIONS", payload["conventions"])
        self.assertEqual(
            payload["conventions"]["USD-3M-SWAP-CONVENTIONS"]["convention_type"],
            "Swap",
        )
        self.assertGreater(len(payload["ore_curve_points"]["times"]), 10)
        self.assertGreater(len(payload["ore_calibration_trace"]["pillars"]), 10)
        self.assertEqual(len(payload["segment_alignment"]), 3)
        self.assertEqual(
            payload["segment_alignment"][0]["matched_calibration_rows"],
            payload["segment_alignment"][0]["quote_count"],
        )
        self.assertFalse(
            any(
                quote["missing"]
                for segment in payload["curve_config"]["segments"]
                for quote in segment["quotes"]
            )
        )
        self.assertIn("USD1D", payload["dependency_graph"])
        self.assertIn("USD3M", payload["dependency_graph"])

    def test_trace_discount_curve_generic_api_matches_usd_wrapper(self):
        wrapper_payload = trace_usd_curve_from_ore(self.ORE_XML)
        generic_payload = trace_discount_curve_from_ore(self.ORE_XML, currency="USD")
        self.assertEqual(generic_payload["curve_handle"], wrapper_payload["curve_handle"])
        self.assertEqual(generic_payload["trace_type"], "discount_curve")
        self.assertEqual(generic_payload["dependency_graph"]["USD3M"]["dependencies"], ["USD1D"])

    def test_trace_index_curve_and_graph_api(self):
        payload = trace_index_curve_from_ore(self.ORE_XML, index_name="USD-LIBOR-3M")
        self.assertEqual(payload["curve_handle"], "Yield/USD/USD3M")
        self.assertEqual(payload["curve_name"], "USD-LIBOR-3M")
        self.assertEqual(payload["trace_type"], "index_curve")

        graph_payload = trace_curve_graph_from_ore(self.ORE_XML, index_name="USD-LIBOR-3M")
        self.assertEqual(graph_payload["curve_handle"], payload["curve_handle"])
        self.assertIn("USD1D", graph_payload["dependency_graph"])

    def test_trace_other_currencies_discount_curves(self):
        expected = {
            "EUR": ("Yield/EUR/EUR6M", {"EUR1D", "EUR6M"}),
            "GBP": ("Yield/GBP/GBP6M", {"GBP1D", "GBP6M"}),
            "CHF": ("Yield/CHF/CHF6M", {"CHF1D", "CHF6M"}),
            "JPY": ("Yield/JPY/JPY6M", {"JPY1D", "JPY6M"}),
        }
        for ccy, (handle, deps) in expected.items():
            with self.subTest(currency=ccy):
                payload = trace_discount_curve_from_ore(self.ORE_XML, currency=ccy)
                self.assertEqual(payload["curve_handle"], handle)
                self.assertEqual(set(payload["dependency_graph"].keys()), deps)
                self.assertGreater(len(payload["ore_calibration_trace"]["pillars"]), 10)

    def test_trace_usd_6m_dual_curve_graph(self):
        payload = trace_index_curve_from_ore(self.ORE_XML, index_name="USD-LIBOR-6M")
        self.assertEqual(payload["curve_handle"], "Yield/USD/USD6M")
        self.assertEqual(set(payload["dependency_graph"].keys()), {"USD1D", "USD3M", "USD6M"})
        self.assertEqual(payload["dependency_graph"]["USD6M"]["dependencies"], ["USD1D", "USD3M"])

    def test_cli_can_write_json(self):
        payload = trace_usd_curve_from_ore(self.ORE_XML)
        self.assertIn("ore_calibration_trace", payload)


if __name__ == "__main__":
    unittest.main()
