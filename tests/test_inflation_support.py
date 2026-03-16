from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from py_ore_tools import ore_snapshot_cli
from pythonore.compute.inflation import (
    load_inflation_curve_from_market_data,
    load_zero_inflation_surface_quote,
    parse_inflation_models_from_simulation_xml,
    price_inflation_capfloor,
    price_yoy_swap,
    price_zero_coupon_cpi_swap,
)
from pythonore.io.loader import XVALoader
from pythonore.runtime.runtime import XVAEngine


TOOLS_DIR = Path(__file__).resolve().parents[1]


class TestInflationSupport(unittest.TestCase):
    def test_parse_inflation_models_from_simulation_xml_supports_both_families(self):
        lgm_models = parse_inflation_models_from_simulation_xml(
            TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input" / "simulation.xml"
        )
        self.assertIn("EUHICPXT", lgm_models)
        self.assertEqual(lgm_models["EUHICPXT"].family, "LGM")
        self.assertTrue(lgm_models["EUHICPXT"].calibration_instruments)

        jy_models = parse_inflation_models_from_simulation_xml(
            TOOLS_DIR / "Examples" / "Exposure" / "Input" / "simulation_inflation_jy.xml"
        )
        self.assertIn("UKRPI", jy_models)
        self.assertEqual(jy_models["UKRPI"].family, "JarrowYildirim")
        self.assertTrue(any(x.family == "YoYSwap" for x in jy_models["UKRPI"].calibration_instruments))

    def test_load_inflation_curves_and_surface_quotes(self):
        market_path = TOOLS_DIR / "Examples" / "Input" / "market_20160205.txt"
        zc = load_inflation_curve_from_market_data(market_path, "2016-02-05", "UKRPI", curve_type="ZC")
        yy = load_inflation_curve_from_market_data(market_path, "2016-02-05", "EUHICPXT", curve_type="YY")
        self.assertGreater(zc.rate(10.0), 0.0)
        self.assertGreater(yy.rate(5.0), 0.0)
        price = load_zero_inflation_surface_quote(market_path, "2016-02-05", "EUHICPXT", "10Y", 0.015, "Floor")
        self.assertIsNotNone(price)
        self.assertGreater(float(price), 0.0)

    def test_basic_inflation_pricers_return_finite_values(self):
        market_path = TOOLS_DIR / "Examples" / "Input" / "market_20160205.txt"
        zc = load_inflation_curve_from_market_data(market_path, "2016-02-05", "UKRPI", curve_type="ZC")
        yy = load_inflation_curve_from_market_data(market_path, "2016-02-05", "EUHICPXT", curve_type="YY")
        disc = lambda t: math.exp(-0.02 * float(t))

        zc_npv = price_zero_coupon_cpi_swap(
            notional=10_000_000.0,
            maturity_years=20.0,
            fixed_rate=0.056467309,
            base_cpi=210.0,
            inflation_curve=zc,
            discount_curve=disc,
            receive_inflation=True,
        )
        yy_npv = price_yoy_swap(
            notional=10_000_000.0,
            payment_times=(1.0, 2.0, 3.0, 4.0, 5.0),
            fixed_rate=0.008375,
            inflation_curve=yy,
            discount_curve=disc,
            receive_inflation=True,
        )
        cap_npv = price_inflation_capfloor(
            definition=type(
                "Def",
                (),
                {
                    "notional": 10_000_000.0,
                    "maturity_years": 10.0,
                    "inflation_type": "YY",
                    "option_type": "Floor",
                    "strike": 0.015,
                    "long_short": "Long",
                },
            )(),
            inflation_curve=yy,
            discount_curve=disc,
            market_surface_price=0.064862,
        )
        self.assertTrue(math.isfinite(zc_npv))
        self.assertTrue(math.isfinite(yy_npv))
        self.assertTrue(math.isfinite(cap_npv))
        self.assertGreater(cap_npv, 0.0)

    def test_loader_parses_inflation_products(self):
        snap = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input"), ore_file="ore.xml")
        product_types = {t.trade_id: t.product.product_type for t in snap.portfolio.trades}
        self.assertEqual(product_types["CPI_Swap_1"], "InflationSwap")
        self.assertEqual(product_types["CPI_Swap_2"], "InflationSwap")
        self.assertEqual(product_types["YearOnYear_Swap"], "InflationSwap")
        self.assertEqual(next(t for t in snap.portfolio.trades if t.trade_id == "CPI_Swap_1").product.pay_leg, "fixed")

    def test_loader_parses_inflation_capfloor_products(self):
        snap = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input"), ore_file="ore_capfloor.xml")
        self.assertTrue(all(t.product.product_type == "InflationCapFloor" for t in snap.portfolio.trades))

    def test_cli_marks_inflation_examples_as_native_price_supported(self):
        swap_case = TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input" / "ore.xml"
        capfloor_case = TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input" / "ore_capfloor.xml"
        self.assertTrue(ore_snapshot_cli._supports_native_price_only("Swap", swap_case))
        self.assertTrue(ore_snapshot_cli._supports_native_price_only("CapFloor", capfloor_case))

    def test_cli_price_only_handles_legacy_inflation_cases(self):
        swap_case = TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input" / "ore.xml"
        result = ore_snapshot_cli._compute_price_only_case(swap_case, anchor_t0_npv=False)
        self.assertEqual(result["diagnostics"]["engine"], "python_price_only")
        self.assertEqual(result["pricing"]["trade_type"], "Swap")
        self.assertIn("inflation_kind", result["pricing"])

    def test_cli_price_only_handles_inflation_cases_without_pricing_reference(self):
        swap_case = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_inflation_jy.xml"
        result = ore_snapshot_cli._compute_price_only_case(swap_case, anchor_t0_npv=False)
        self.assertEqual(result["diagnostics"]["engine"], "python_price_only")
        self.assertTrue(result["diagnostics"].get("missing_native_pricing_reference"))
        self.assertEqual(result["diagnostics"].get("inflation_model_family"), "JarrowYildirim")
        self.assertIn("py_t0_npv", result["pricing"])
        self.assertNotIn("ore_t0_npv", result["pricing"])

    def test_cli_xva_handles_inflation_cases_without_reference_output(self):
        swap_case = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_inflation_dk.xml"
        result = ore_snapshot_cli._compute_inflation_swap_snapshot_case(
            swap_case,
            paths=None,
            seed=42,
            rng_mode="numpy",
            anchor_t0_npv=False,
            own_hazard=0.02,
            own_recovery=0.4,
            xva_mode="ore",
        )
        self.assertEqual(result.diagnostics["engine"], "python_inflation_native")
        self.assertTrue(result.diagnostics.get("missing_reference_xva"))
        self.assertEqual(result.diagnostics.get("inflation_model_family"), "LGM")
        self.assertGreater(len(result.exposure_times), 1)
        self.assertGreaterEqual(result.xva["py_cva"], 0.0)
        self.assertNotIn("ore_t0_npv", result.pricing)

    def test_python_lgm_runtime_handles_inflation_swap_portfolio(self):
        snap = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Exposure" / "Input"), ore_file="ore_inflation_jy.xml")
        result = XVAEngine.python_lgm_default(fallback_to_swig=False).create_session(snap).run(return_cubes=False)
        self.assertIn("CVA", result.xva_by_metric)
        self.assertGreaterEqual(result.xva_by_metric["CVA"], 0.0)
        self.assertIn("CPTY_A", result.exposure_by_netting_set)


if __name__ == "__main__":
    unittest.main()
