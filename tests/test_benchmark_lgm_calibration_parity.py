import json
import tempfile
import unittest
from pathlib import Path

from py_ore_tools.benchmarks.benchmark_lgm_calibration_parity import (
    _load_ore_currency_config,
    _resolve_vol_type,
)


class TestBenchmarkLgmCalibrationParity(unittest.TestCase):
    def test_load_ore_currency_config_reads_swaptions(self):
        calibration_xml = """<Root>
  <InterestRateModels>
    <LGM key="EUR">
      <CalibrationType>Bootstrap</CalibrationType>
      <CalibrationSwaptions>
        <Expiries>1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,11Y</Expiries>
        <Terms>10Y,9Y,8Y,7Y,6Y,5Y,4Y,3Y,2Y,18M,1Y</Terms>
        <Strikes>ATM,ATM,ATM,ATM,ATM,ATM,ATM,ATM,ATM,ATM,ATM</Strikes>
      </CalibrationSwaptions>
      <Volatility>
        <TimeGrid>0.5,1,2,3,4,5,6,7,8,9</TimeGrid>
        <InitialValue>0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02</InitialValue>
      </Volatility>
    </LGM>
  </InterestRateModels>
</Root>
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.xml"
            path.write_text(calibration_xml, encoding="utf-8")
            config, ore_times, ore_sigmas, expiries, terms, strikes = _load_ore_currency_config(path, "EUR")

        self.assertEqual(config.currency, "EUR")
        self.assertEqual(config.calibration_type.value, "Bootstrap")
        self.assertEqual(len(config.calibration_swaptions), 11)
        self.assertEqual(config.calibration_swaptions[0].expiry, "1Y")
        self.assertEqual(config.calibration_swaptions[-1].term, "1Y")
        self.assertEqual(len(ore_times), 10)
        self.assertEqual(len(ore_sigmas), 11)
        self.assertEqual(expiries[0], "1Y")
        self.assertEqual(terms[-1], "1Y")
        self.assertEqual(strikes[0], "ATM")

    def test_resolve_vol_type_falls_back_to_normal(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "curveconfig.xml"
            path.write_text("<Root />", encoding="utf-8")
            self.assertEqual(_resolve_vol_type(path, "auto", "EUR"), "normal")

    def test_resolve_vol_type_reads_ore_curve_config(self):
        curve_config = Path("/Users/gordonlee/Documents/Engine/Examples/Input/curveconfig.xml")
        self.assertEqual(_resolve_vol_type(curve_config, "auto", "EUR"), "normal")


if __name__ == "__main__":
    unittest.main()
