from pathlib import Path
from datetime import date

import unittest

from pythonore.benchmarks.benchmark_ore_ir_grid import (
    _berm_portfolio_xml,
    _effective_curve_pairs,
    _ore_market_for_curve_pair,
    _ore_pricing_only_xml,
    _parse_args,
)
from pythonore.benchmarks.benchmark_ore_ir_options import _build_berm_from_ore_flows, _build_model_from_ore_calibration
from pythonore.compute.irs_xva_utils import build_discount_curve_from_discount_pairs, load_ore_discount_pairs_from_curves
from pythonore.compute.lgm_ir_options import bermudan_backward_price


class TestBenchmarkOreIrGrid(unittest.TestCase):
    def test_portfolio_contains_only_bermudan_trades(self):
        portfolio_xml = _berm_portfolio_xml()
        self.assertIn('<Trade id="BERM_EUR_5Y">', portfolio_xml)
        self.assertNotIn("CapFloor", portfolio_xml)

    def test_pricing_only_ore_xml_omits_simulation_and_xva(self):
        xml_text = _ore_pricing_only_xml(Path("/tmp/grid_out"), Path("/tmp/grid_in"), pricing_market="default")
        self.assertIn('<Analytic type="npv">', xml_text)
        self.assertIn('<Analytic type="cashflow">', xml_text)
        self.assertIn('<Analytic type="curves">', xml_text)
        self.assertNotIn('<Analytic type="simulation">', xml_text)
        self.assertNotIn('<Analytic type="xva">', xml_text)
        self.assertIn('<Parameter name="pricing">default</Parameter>', xml_text)

    def test_fast_grid_benchmark_supports_explicit_model_sources(self):
        args = _parse_args(["--model-source", "simulation_stub"])
        self.assertEqual(args.model_source, "simulation_stub")

    def test_fast_grid_benchmark_supports_multiple_curve_pairs(self):
        args = _parse_args(["--curve-pair", "EUR-EURIBOR-6M:EUR-EURIBOR-6M"])
        self.assertEqual(
            _effective_curve_pairs(args),
            [
                ("EUR-EONIA", "EUR-EURIBOR-6M"),
                ("EUR-EURIBOR-6M", "EUR-EURIBOR-6M"),
            ],
        )

    def test_ore_market_is_inferred_from_supported_curve_pair(self):
        self.assertEqual(_ore_market_for_curve_pair("EUR-EONIA", "EUR-EURIBOR-6M"), "default")
        self.assertEqual(_ore_market_for_curve_pair("EUR-EURIBOR-6M", "EUR-EURIBOR-6M"), "libor")

    def test_dual_curve_forwarding_changes_bermudan_backward_price(self):
        out = Path("/Users/gordonlee/Documents/PythonOreRunner/parity_artifacts/ir_grid_ore_benchmark/market_default/Output")
        model = _build_model_from_ore_calibration(out / "classic" / "calibration.xml", ccy_key="EUR")
        berm = _build_berm_from_ore_flows(out / "flows.csv", asof=date(2016, 2, 5), trade_id="BERM_EUR_5Y")

        disc_t, disc_df = load_ore_discount_pairs_from_curves((out / "curves.csv").as_posix(), discount_column="EUR-EONIA")
        p0_disc = build_discount_curve_from_discount_pairs(list(zip(disc_t.tolist(), disc_df.tolist())))
        fwd_t, fwd_df = load_ore_discount_pairs_from_curves((out / "curves.csv").as_posix(), discount_column="EUR-EURIBOR-6M")
        p0_fwd = build_discount_curve_from_discount_pairs(list(zip(fwd_t.tolist(), fwd_df.tolist())))

        single = bermudan_backward_price(model, p0_disc, p0_disc, berm, n_grid=41).price
        dual = bermudan_backward_price(model, p0_disc, p0_fwd, berm, n_grid=41).price
        self.assertNotAlmostEqual(single, dual, places=6)

    def test_libor_grid_benchmark_stays_close_to_ore_with_a360_float(self):
        out = Path("/Users/gordonlee/Documents/PythonOreRunner/parity_artifacts/ir_grid_ore_benchmark/market_libor/Output")
        model = _build_model_from_ore_calibration(out / "classic" / "calibration.xml", ccy_key="EUR")
        disc_t, disc_df = load_ore_discount_pairs_from_curves((out / "curves.csv").as_posix(), discount_column="EUR-EURIBOR-6M")
        p0 = build_discount_curve_from_discount_pairs(list(zip(disc_t.tolist(), disc_df.tolist())))

        expected = {
            "BERM_EUR_5Y": 0.003,
            "BERM_EUR_5Y_LOWK": 0.003,
            "BERM_EUR_5Y_HIGHK": 0.003,
        }
        for trade_id, threshold in expected.items():
            berm = _build_berm_from_ore_flows(out / "flows.csv", asof=date(2016, 2, 5), trade_id=trade_id)
            py = bermudan_backward_price(model, p0, p0, berm, n_grid=41).price

            ore = None
            import csv

            with open(out / "npv.csv", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
                for row in reader:
                    if row.get(tid_key, "") == trade_id:
                        ore = float(row["NPV"])
                        break
            self.assertIsNotNone(ore)
            rel = abs(py - ore) / max(abs(float(ore)), 1.0)
            self.assertLess(rel, threshold, msg=f"{trade_id} rel diff {rel:.6%} exceeded {threshold:.2%}")


if __name__ == "__main__":
    unittest.main()
