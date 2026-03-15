from pathlib import Path

import unittest

from pythonore.benchmarks.benchmark_ore_ir_grid import _berm_portfolio_xml, _ore_pricing_only_xml, _parse_args


class TestBenchmarkOreIrGrid(unittest.TestCase):
    def test_portfolio_contains_only_bermudan_trades(self):
        portfolio_xml = _berm_portfolio_xml()
        self.assertIn('<Trade id="BERM_EUR_5Y">', portfolio_xml)
        self.assertNotIn("CapFloor", portfolio_xml)

    def test_pricing_only_ore_xml_omits_simulation_and_xva(self):
        xml_text = _ore_pricing_only_xml(Path("/tmp/grid_out"), Path("/tmp/grid_in"))
        self.assertIn('<Analytic type="npv">', xml_text)
        self.assertIn('<Analytic type="cashflow">', xml_text)
        self.assertIn('<Analytic type="curves">', xml_text)
        self.assertNotIn('<Analytic type="simulation">', xml_text)
        self.assertNotIn('<Analytic type="xva">', xml_text)

    def test_fast_grid_benchmark_supports_explicit_model_sources(self):
        args = _parse_args(["--model-source", "simulation_stub"])
        self.assertEqual(args.model_source, "simulation_stub")


if __name__ == "__main__":
    unittest.main()
