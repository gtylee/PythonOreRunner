import unittest

import QuantLib as ql

import py_ore_tools.lgm_calibration as lgm_calibration_module
from py_ore_tools.lgm_calibration import (
    CalibrationType,
    CurrencyLgmConfig,
    FallbackType,
    LgmCalibrationError,
    LgmMarketInputs,
    ParamType,
    QuantLibGsrCalibrationBackend,
    VolatilityType,
    build_lgm_swaption_basket,
    calibrate_lgm_currency,
)


def _build_market(volatility=0.20):
    ref = ql.Date(11, 3, 2026)
    ql.Settings.instance().evaluationDate = ref
    curve = ql.YieldTermStructureHandle(ql.FlatForward(ref, 0.02, ql.Actual365Fixed()))
    long_index = ql.SwapIndex(
        "EUR-LONG",
        ql.Period("30Y"),
        2,
        ql.EURCurrency(),
        ql.TARGET(),
        ql.Period("1Y"),
        ql.ModifiedFollowing,
        ql.Thirty360(ql.Thirty360.BondBasis),
        ql.Euribor6M(curve),
        curve,
    )
    short_index = ql.SwapIndex(
        "EUR-SHORT",
        ql.Period("1Y"),
        2,
        ql.EURCurrency(),
        ql.TARGET(),
        ql.Period("1Y"),
        ql.ModifiedFollowing,
        ql.Thirty360(ql.Thirty360.BondBasis),
        ql.Euribor3M(curve),
        curve,
    )
    vol = ql.ConstantSwaptionVolatility(
        ref,
        ql.TARGET(),
        ql.ModifiedFollowing,
        volatility,
        ql.Actual365Fixed(),
        ql.Normal,
    )
    return LgmMarketInputs(
        swaption_vol_surface=ql.SwaptionVolatilityStructureHandle(vol),
        swap_index=long_index,
        short_swap_index=short_index,
        calibration_discount_curve=curve,
        model_discount_curve=curve,
    )


class _DummyHelper:
    def __init__(self):
        self.market = 1.0
        self.model = 1.0

    def marketValue(self):
        return self.market

    def modelValue(self):
        return self.model


class _DummyModel:
    def __init__(self, n_reversions=1, n_vols=3):
        self.calls = []
        self._reversions = [0.03] * n_reversions
        self._vols = [0.01] * n_vols
        self._params = self._reversions + self._vols

    def reversion(self):
        return self._reversions

    def volatility(self):
        return self._vols

    def params(self):
        return self._params

    def calibrateVolatilitiesIterative(self, *args):
        self.calls.append(("iter_vol", len(args[0])))

    def calibrate(self, helpers, method, end_criteria, constraint, weights, mask):
        self.calls.append(("calibrate", len(helpers), list(mask)))


class _DummyBasketInstrument:
    def __init__(self):
        self.helper = _DummyHelper()


class TestLgmCalibration(unittest.TestCase):
    def test_bootstrap_grid_override_uses_swaption_expiries(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "volatility": {
                    "calibrate": True,
                    "type": "HullWhite",
                    "param_type": "Piecewise",
                    "time_grid": [1.0, 2.0, 3.0],
                    "initial_values": [0.01, 0.02, 0.03, 0.04],
                },
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
                "calibration_swaptions": {
                    "expiries": ["1Y", "2Y", "3Y"],
                    "terms": ["5Y", "4Y", "3Y"],
                    "strikes": ["ATM", "ATM", "ATM"],
                },
            },
        )
        market = _build_market()
        basket = build_lgm_swaption_basket(config, market)
        backend = QuantLibGsrCalibrationBackend(config, market, basket)
        _, _, effective_vol_times, _, _, _ = backend._build_gsr()
        self.assertEqual(tuple(round(x, 6) for x in effective_vol_times), tuple(round(x, 6) for x in [basket[0].expiry_time, basket[1].expiry_time]))

    def test_basket_handles_date_and_period_inputs(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "volatility": {"calibrate": True, "type": "HullWhite", "param_type": "Piecewise", "time_grid": [1.0], "initial_values": [0.01, 0.01]},
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
                "calibration_swaptions": [
                    {"expiry": "2027-03-11", "term": "5Y", "strike": "ATM"},
                    {"expiry": "1Y", "term": "2031-03-11", "strike": "ATM"},
                ],
            },
        )
        basket = build_lgm_swaption_basket(config, _build_market())
        self.assertEqual(len(basket), 2)
        self.assertGreater(basket[0].swap_length, 0.0)
        self.assertGreater(basket[1].swap_length, 0.0)

    def test_short_and_long_swap_index_selection(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "volatility": {"calibrate": True, "type": "HullWhite", "param_type": "Piecewise", "time_grid": [1.0], "initial_values": [0.01, 0.01]},
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
                "calibration_swaptions": [
                    {"expiry": "1Y", "term": "6M", "strike": "ATM"},
                    {"expiry": "1Y", "term": "5Y", "strike": "ATM"},
                ],
            },
        )
        basket = build_lgm_swaption_basket(config, _build_market())
        self.assertEqual(basket[0].ibor_index_name, "Euribor3M Actual/360")
        self.assertEqual(basket[1].ibor_index_name, "Euribor6M Actual/360")

    def test_strike_fallback_rule_caps_far_otm_strike(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "volatility": {"calibrate": True, "type": "HullWhite", "param_type": "Piecewise", "time_grid": [1.0], "initial_values": [0.01, 0.01]},
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
                "calibration_swaptions": [{"expiry": "1Y", "term": "5Y", "strike": "1.00"}],
            },
        )
        basket = build_lgm_swaption_basket(config, _build_market(volatility=0.01))
        self.assertEqual(basket[0].fallback_type, FallbackType.FALLBACK_RULE_1)
        self.assertLess(basket[0].strike_used, 1.0)

    def test_reference_calibration_grid_filters_basket(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "reference_calibration_grid": "18M,30M",
                "volatility": {"calibrate": True, "type": "HullWhite", "param_type": "Piecewise", "time_grid": [1.0, 2.0], "initial_values": [0.01, 0.01, 0.01]},
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
                "calibration_swaptions": {
                    "expiries": ["1Y", "2Y", "3Y"],
                    "terms": ["5Y", "4Y", "3Y"],
                    "strikes": ["ATM", "ATM", "ATM"],
                },
            },
        )
        basket = build_lgm_swaption_basket(config, _build_market())
        self.assertEqual(len(basket), 2)

    def test_bootstrap_vol_branch_uses_iterative_method(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "volatility": {"calibrate": True, "type": "HullWhite", "param_type": "Piecewise", "time_grid": [1.0], "initial_values": [0.01, 0.01]},
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
            },
        )
        backend = QuantLibGsrCalibrationBackend(config, _build_market(), [_DummyBasketInstrument(), _DummyBasketInstrument()])
        model = _DummyModel(n_reversions=1, n_vols=2)
        original = lgm_calibration_module._build_black_helper_vector
        try:
            lgm_calibration_module._build_black_helper_vector = lambda xs: xs
            backend._precheck_bootstrap_vol = lambda *args: None
            backend._run_calibration(model)
        finally:
            lgm_calibration_module._build_black_helper_vector = original
        self.assertEqual(model.calls, [("iter_vol", 2)])

    def test_bootstrap_reversion_branch_runs_one_helper_at_a_time(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "volatility": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.01]},
                "reversion": {"calibrate": True, "type": "HullWhite", "param_type": "Piecewise", "time_grid": [1.0], "initial_values": [0.03, 0.03]},
            },
        )
        backend = QuantLibGsrCalibrationBackend(config, _build_market(), [_DummyBasketInstrument(), _DummyBasketInstrument()])
        model = _DummyModel(n_reversions=2, n_vols=1)
        original = lgm_calibration_module._build_black_helper_vector
        try:
            lgm_calibration_module._build_black_helper_vector = lambda xs: xs
            backend._precheck_bootstrap_vol = lambda *args: None
            backend._run_calibration(model)
        finally:
            lgm_calibration_module._build_black_helper_vector = original
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call[0] == "calibrate" and call[1] == 1 for call in model.calls))

    def test_global_vol_only_branch_fixes_reversion_parameters(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "BestFit",
                "volatility": {"calibrate": True, "type": "HullWhite", "param_type": "Piecewise", "time_grid": [1.0], "initial_values": [0.01, 0.01]},
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
            },
        )
        backend = QuantLibGsrCalibrationBackend(config, _build_market(), [_DummyBasketInstrument(), _DummyBasketInstrument()])
        model = _DummyModel(n_reversions=1, n_vols=2)
        original = lgm_calibration_module._build_black_helper_vector
        try:
            lgm_calibration_module._build_black_helper_vector = lambda xs: xs
            backend._precheck_bootstrap_vol = lambda *args: None
            backend._run_calibration(model)
        finally:
            lgm_calibration_module._build_black_helper_vector = original
        self.assertEqual(model.calls[0][0], "calibrate")
        self.assertEqual(model.calls[0][2], [True, False, False])

    def test_hagan_volatility_fails_fast_without_quantext_bindings(self):
        config = CurrencyLgmConfig.from_dict(
            "EUR",
            {
                "calibration_type": "Bootstrap",
                "volatility": {"calibrate": True, "type": "Hagan", "param_type": "Piecewise", "time_grid": [1.0], "initial_values": [0.01, 0.01]},
                "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
                "calibration_swaptions": [{"expiry": "1Y", "term": "5Y", "strike": "ATM"}],
            },
        )
        with self.assertRaises(NotImplementedError):
            calibrate_lgm_currency(config, _build_market())

    def test_continue_on_error_controls_invalid_result_vs_exception(self):
        base = {
            "calibration_type": "None",
            "volatility": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.0001]},
            "reversion": {"calibrate": False, "type": "HullWhite", "param_type": "Constant", "initial_values": [0.03]},
            "calibration_swaptions": [{"expiry": "1Y", "term": "5Y", "strike": "ATM"}],
            "bootstrap_tolerance": 1.0e-12,
        }
        strict = CurrencyLgmConfig.from_dict("EUR", base)
        with self.assertRaises(LgmCalibrationError):
            calibrate_lgm_currency(strict, _build_market())

        permissive = CurrencyLgmConfig.from_dict("EUR", {**base, "continue_on_error": True})
        result = calibrate_lgm_currency(permissive, _build_market())
        self.assertFalse(result.valid)
        self.assertEqual(len(result.points), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
