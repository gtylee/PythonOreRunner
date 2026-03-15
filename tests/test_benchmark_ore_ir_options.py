from pathlib import Path
from datetime import date

import unittest

import numpy as np

from pythonore.benchmarks.benchmark_ore_ir_options import (
    BERM_BACKWARD_N_GRID,
    BERM_TRADE_SPECS,
    CAP_INDEX,
    CAP_TENOR,
    _build_berm_from_ore_flows,
    _build_model_from_ore_calibration,
    _build_simulation_stub_model,
    _bachelier_caplet_pv,
    _cap_schedule_arrays,
    _cap_t0_bachelier_npv,
    _cap_trade_xml,
    _load_cap_vol_quotes,
    _ore_classic_calibration_xml,
    _portfolio_xml,
    _simulation_classic_xml,
)


class TestBenchmarkOreIrOptions(unittest.TestCase):
    def test_bermudan_backward_grid_is_tuned_explicitly(self):
        self.assertEqual(BERM_BACKWARD_N_GRID, 11)

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

    def test_simulation_classic_xml_keeps_calibration_live(self):
        xml_text = _simulation_classic_xml(256, 42)
        self.assertIn("<Samples>256</Samples>", xml_text)
        self.assertIn("<Seed>42</Seed>", xml_text)
        self.assertIn("<Calibrate>Y</Calibrate>", xml_text)
        self.assertIn("<InitialValue>0.0</InitialValue>", xml_text)

    def test_classic_calibration_ore_xml_targets_output_classic(self):
        xml_text = _ore_classic_calibration_xml(
            Path("/tmp/ir_options_bench"),
            Path("/tmp/ir_options_bench/Input"),
            Path("/tmp/ir_options_bench/Input/simulation_classic.xml"),
        )
        self.assertIn("<Parameter name=\"outputPath\">/tmp/ir_options_bench/classic</Parameter>", xml_text)
        self.assertIn("<Analytic type=\"calibration\">", xml_text)
        self.assertIn("<Parameter name=\"configFile\">/tmp/ir_options_bench/Input/simulation_classic.xml</Parameter>", xml_text)

    def test_build_model_from_classic_calibration_uses_ore_emitted_params(self):
        calibration_xml = Path(
            "/Users/gordonlee/Documents/PythonOreRunner/parity_artifacts/bermudan_method_compare/berm_200bp/Output/classic/calibration.xml"
        )
        model = _build_model_from_ore_calibration(calibration_xml, ccy_key="EUR")
        np.testing.assert_allclose(
            np.asarray(model.params.alpha_values, dtype=float)[:3],
            np.array([0.00490028, 0.00490182, 0.00490381]),
            rtol=0.0,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(np.asarray(model.params.kappa_values, dtype=float), np.array([0.0]), rtol=0.0, atol=1.0e-12)

    def test_build_simulation_stub_model_is_explicit(self):
        model = _build_simulation_stub_model()
        np.testing.assert_allclose(np.asarray(model.params.alpha_values, dtype=float), np.full(8, 0.01))
        np.testing.assert_allclose(np.asarray(model.params.kappa_values, dtype=float), np.array([0.03]))

    def test_reconstructed_bermudan_uses_flow_fixing_dates(self):
        berm = _build_berm_from_ore_flows(
            Path("/Users/gordonlee/Documents/PythonOreRunner/parity_artifacts/ir_grid_ore_benchmark/Output/flows.csv"),
            asof=date(2016, 2, 5),
            trade_id="BERM_EUR_5Y",
        )
        self.assertIn("float_fixing_time", berm.underlying_legs)
        self.assertLess(float(berm.underlying_legs["float_fixing_time"][0]), float(berm.underlying_legs["float_start_time"][0]))


if __name__ == "__main__":
    unittest.main()
