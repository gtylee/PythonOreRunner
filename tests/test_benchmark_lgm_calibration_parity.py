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
        calibration_xml = Path("/tmp/lgm_hw_parity_case/Output/calibration.xml")
        config, ore_times, ore_sigmas, expiries, terms, strikes = _load_ore_currency_config(calibration_xml, "EUR")
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
