import unittest
from pathlib import Path

from pythonore.benchmarks.benchmark_lgm_ore_multiccy import (
    _load_ore_trade_npv,
    _load_pricing_market,
    _pricing_discount_column,
    _python_t0_npv,
)
from pythonore.repo_paths import local_parity_artifacts_root


class TestBenchmarkLgmOreMulticcy(unittest.TestCase):
    CASES_ROOT = local_parity_artifacts_root() / "multiccy_benchmark_final" / "cases"

    def test_cad_npv_only_uses_flow_amount_discounting_under_libor_market(self):
        cases = [
            ("flat_CAD_5Y_A_CAD", "SWAP_CAD_5Y_A_CAD_flat"),
            ("flat_CAD_5Y_B", "SWAP_CAD_5Y_B_flat"),
        ]
        for case_name, trade_id in cases:
            with self.subTest(case=case_name):
                case_root = self.CASES_ROOT / case_name
                ore_xml = case_root / "Input" / "ore.xml"
                pricing_market = _load_pricing_market(ore_xml)
                self.assertEqual(pricing_market, "libor")

                pricing_disc_col = _pricing_discount_column(
                    "CAD",
                    pricing_market,
                    fallback_disc_col="CAD-CORRA",
                    forward_col="CAD-CDOR-3M",
                )
                self.assertEqual(pricing_disc_col, "CAD-CDOR-3M")

                ore_npv = _load_ore_trade_npv(case_root / "Output" / "npv.csv", trade_id)
                py_npv, source = _python_t0_npv(
                    simulation_xml=case_root / "Input" / "simulation.xml",
                    curves_csv=case_root / "Output" / "curves.csv",
                    portfolio_xml=case_root / "Input" / "portfolio.xml",
                    trade_id=trade_id,
                    model_ccy="CAD",
                    disc_col="CAD-CORRA",
                    fwd_col="CAD-CDOR-3M",
                    flows_csv=case_root / "Output" / "flows.csv",
                    pricing_disc_col=pricing_disc_col,
                    ore_npv=ore_npv,
                )

                self.assertEqual(source, "flows_amount_discounted")
                rel_diff = abs(py_npv - ore_npv) / max(abs(ore_npv), 1.0)
                self.assertLess(rel_diff, 1.0e-4)


if __name__ == "__main__":
    unittest.main()
