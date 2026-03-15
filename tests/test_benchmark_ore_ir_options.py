from pathlib import Path

import unittest

import numpy as np

from pythonore.benchmarks.benchmark_ore_ir_options import (
    BERM_TRADE_SPECS,
    CAP_INDEX,
    CAP_TENOR,
    _bachelier_caplet_pv,
    _cap_schedule_arrays,
    _cap_t0_bachelier_npv,
    _cap_trade_xml,
    _load_cap_vol_quotes,
    _portfolio_xml,
)


class TestBenchmarkOreIrOptions(unittest.TestCase):
    def test_cap_trade_matches_available_eur_surface_tenor(self):
        trade_xml = _cap_trade_xml("CAP_TEST", 0.03)
        self.assertIn(f"<Tenor>{CAP_TENOR}</Tenor>", trade_xml)
        self.assertIn(f"<Index>{CAP_INDEX}</Index>", trade_xml)

    def test_python_cap_schedule_uses_three_6m_periods(self):
        start, end, accrual = _cap_schedule_arrays()
        np.testing.assert_allclose(start, np.array([0.5, 1.0, 1.5]))
        np.testing.assert_allclose(end, np.array([1.0, 1.5, 2.0]))
        np.testing.assert_allclose(accrual, np.full(3, 0.5))

    def test_portfolio_contains_multiple_bermudan_examples(self):
        portfolio_xml = _portfolio_xml()
        self.assertGreaterEqual(len(BERM_TRADE_SPECS), 3)
        for spec in BERM_TRADE_SPECS:
            self.assertIn(f'<Trade id="{spec["trade_id"]}">', portfolio_xml)

    def test_bachelier_caplet_remains_positive_for_deep_otm_normal_cap(self):
        pv = _bachelier_caplet_pv(
            forward=0.01,
            strike=0.08,
            normal_vol=0.01789382,
            expiry=1.5,
            pay_df=0.99,
            accrual=0.5,
            notional=10_000_000.0,
        )
        self.assertGreater(pv, 1.0)

    def test_market_quote_cap_t0_price_stays_material_for_8pct_strike(self):
        quotes = _load_cap_vol_quotes(
            Path("/Users/gordonlee/Documents/PythonOreRunner/Examples/Input/market_20160205_flat.txt")
        )
        self.assertAlmostEqual(quotes[0.08], 0.01789382)

        def disc_curve(t: float) -> float:
            return float(np.exp(-0.01 * t))

        fwd_rate = 2.0 * float(np.log(1.0 + 0.03 * 0.5)) / 0.5

        def fwd_curve(t: float) -> float:
            return float(np.exp(-fwd_rate * t))

        pv = _cap_t0_bachelier_npv(disc_curve, fwd_curve, strike=0.08)
        self.assertGreater(pv, 100.0)


if __name__ == "__main__":
    unittest.main()
