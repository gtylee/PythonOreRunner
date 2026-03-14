import sys
import unittest
from dataclasses import replace
from datetime import date
import numpy as np
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.bond_pricing import (
    BondCashflow,
    BondEngineSpec,
    BondScenarioGrid,
    BondTradeSpec,
    CallableExerciseSpec,
    CompiledBondTrade,
    _callable_price_amount,
    _build_lgm_model_for_callable,
    _fit_curve_for_currency,
    _load_security_spread,
    _price_callable_bond_lgm,
    build_bond_scenario_grid_numpy,
    compile_bond_trade,
    _bond_npv,
    _curve_from_flow_discounts,
    _load_bond_cashflows_from_flows,
    load_bond_trade_spec,
    load_callable_bond_trade_spec,
    price_bond_scenarios_numpy,
    price_bond_scenarios_torch,
    price_bond_single_numpy,
    price_bond_trade,
    torch,
)
from py_ore_tools.ore_snapshot import _load_ore_npv_details
from py_ore_tools.irs_xva_utils import load_ore_default_curve_inputs


EXAMPLE_18_ORE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "Input" / "ore.xml"
EXAMPLE_18_PORTFOLIO = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "Input" / "portfolio.xml"
EXAMPLE_18_REFERENCE = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "Input" / "referencedata.xml"
EXAMPLE_18_FLOWS = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "ExpectedOutput" / "flows.csv"
EXAMPLE_18_NPV = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "ExpectedOutput" / "npv.csv"
EXAMPLE_78_ORE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_78" / "Input" / "ore.xml"
EXAMPLE_78_PORTFOLIO = TOOLS_DIR / "Examples" / "Legacy" / "Example_78" / "Input" / "portfolio.xml"
EXAMPLE_78_REFERENCE = TOOLS_DIR / "Examples" / "Legacy" / "Example_78" / "Input" / "referencedata.xml"
EXAMPLE_78_FLOWS = TOOLS_DIR / "Examples" / "Legacy" / "Example_78" / "ExpectedOutput" / "flows.csv"
EXAMPLE_78_NPV = TOOLS_DIR / "Examples" / "Legacy" / "Example_78" / "ExpectedOutput" / "npv.csv"
EXAMPLE_SHARED_MARKET = TOOLS_DIR / "Examples" / "Input" / "market_20160205_flat.txt"
EXAMPLE_SHARED_TM = TOOLS_DIR / "Examples" / "Input" / "todaysmarket.xml"
EXAMPLE_SHARED_PE = TOOLS_DIR / "Examples" / "Input" / "pricingengine.xml"
CALLABLE_ORE_XML = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_callable_bond.xml"
CALLABLE_PORTFOLIO = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "portfolio_callablebond.xml"
CALLABLE_REFERENCE = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "reference_data_callablebond.xml"
CALLABLE_PE = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "pricingengine_callablebond.xml"
CALLABLE_SIM = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "simulation_callablebond.xml"
CALLABLE_LGM_GRID_ORE_XML = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_callable_bond_lgm_grid_npv_only.xml"
CALLABLE_LGM_GRID_PE = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "pricingengine_callablebond_lgm_grid.xml"
CALLABLE_LGM_GRID_NPV = TOOLS_DIR / "Examples" / "Exposure" / "Output" / "callable_bond_lgm_grid_npv_only" / "npv.csv"
EXAMPLE_SHARED_MARKET_FULL = TOOLS_DIR / "Examples" / "Input" / "market_20160205.txt"


def _flat_curve(rate: float):
    return lambda t: 1.0 if float(t) <= 0.0 else float(pow(2.718281828459045, -rate * float(t)))


class TestBondPricing(unittest.TestCase):
    def _price_callable_variant(self, trade_id: str, *, spec_override=None, engine_override=None) -> dict:
        spec, engine = load_callable_bond_trade_spec(
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id=trade_id,
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_LGM_GRID_PE,
        )
        if spec_override is not None:
            spec = spec_override(spec)
        if engine_override is not None:
            engine = engine_override(engine)
        credit = load_ore_default_curve_inputs(str(EXAMPLE_SHARED_TM), str(EXAMPLE_SHARED_MARKET_FULL), cpty_name=spec.credit_curve_id)
        hazard_times = np.asarray(credit["hazard_times"], dtype=float)
        hazard_rates = np.asarray(credit["hazard_rates"], dtype=float)
        recovery_rate = float(credit["recovery"])
        security_spread = _load_security_spread(EXAMPLE_SHARED_MARKET_FULL, spec.security_id)
        base_curve = _fit_curve_for_currency(CALLABLE_LGM_GRID_ORE_XML, spec.currency)
        model = _build_lgm_model_for_callable(
            ore_xml=CALLABLE_LGM_GRID_ORE_XML,
            pricingengine_path=CALLABLE_LGM_GRID_PE,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            currency=spec.currency,
            maturity_date=max(cf.pay_date for cf in spec.bond.cashflows),
            asof_date=date(2016, 2, 5),
        )
        return _price_callable_bond_lgm(
            spec,
            engine,
            asof_date=date(2016, 2, 5),
            day_counter="A365F",
            reference_curve=base_curve,
            income_curve=base_curve,
            hazard_times=hazard_times,
            hazard_rates=hazard_rates,
            recovery_rate=recovery_rate,
            security_spread=security_spread,
            model=model,
        )

    def _price_example18(self, trade_id: str) -> tuple[dict, float]:
        out = price_bond_trade(
            ore_xml=EXAMPLE_18_ORE_XML,
            portfolio_xml=EXAMPLE_18_PORTFOLIO,
            trade_id=trade_id,
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=EXAMPLE_18_REFERENCE,
            pricingengine_path=EXAMPLE_SHARED_PE,
            flows_csv=EXAMPLE_18_FLOWS,
        )
        ore_npv = _load_ore_npv_details(EXAMPLE_18_NPV, trade_id=trade_id)["npv"]
        return out, ore_npv

    def test_load_ore_flows_for_forward_bond_filters_forward_value_row(self):
        flows = _load_bond_cashflows_from_flows(
            EXAMPLE_18_FLOWS,
            "FwdBond_Fixed",
            forward_underlying_only=True,
        )
        self.assertTrue(flows)
        self.assertTrue(all(not cf.flow_type.startswith("Bond_") for cf in flows))
        self.assertTrue(all(cf.flow_type != "ForwardValue" for cf in flows))
        self.assertEqual(flows[-1].flow_type, "Notional")

    def test_curve_from_flow_discounts_uses_flow_discount_factors(self):
        curve = _curve_from_flow_discounts(
            EXAMPLE_18_FLOWS,
            "Bond_Fixed",
            asof_date=date(2016, 2, 5),
            day_counter="A365F",
        )
        self.assertAlmostEqual(curve(0.0), 1.0, places=12)
        self.assertLess(abs(curve(0.9945205479452055) - 0.9803373492), 1.0e-4)
        self.assertLess(curve(1.0), 1.0)
        self.assertGreater(curve(1.0), curve(5.0))

    def test_bond_npv_without_credit_matches_discounted_cashflows(self):
        spec = BondTradeSpec(
            trade_id="B1",
            trade_type="Bond",
            currency="EUR",
            payer=False,
            security_id="SEC",
            credit_curve_id="CPTY",
            reference_curve_id="EUR-EURIBOR-6M",
            income_curve_id="",
            settlement_days=2,
            calendar="TARGET",
            issue_date=date(2024, 1, 1),
            bond_notional=1.0,
            cashflows=(
                BondCashflow(pay_date=date(2025, 1, 1), amount=5.0, flow_type="Interest", accrual_start=date(2024, 1, 1), accrual_end=date(2025, 1, 1), nominal=100.0),
                BondCashflow(pay_date=date(2025, 1, 1), amount=100.0, flow_type="Notional", nominal=100.0),
            ),
        )
        curve = _flat_curve(0.05)
        value, _ = _bond_npv(
            spec,
            asof_date=date(2024, 1, 1),
            day_counter="A365F",
            discount_curve=curve,
            income_curve=curve,
            hazard_times=np.asarray([10.0], dtype=float),
            hazard_rates=np.asarray([0.0], dtype=float),
            recovery_rate=0.0,
            security_spread=0.0,
            engine_spec=BondEngineSpec(),
        )
        expected = 105.0 * curve(366.0 / 365.0)
        self.assertAlmostEqual(value, expected, places=8)

    def test_load_bond_trade_spec_covers_construction_variants(self):
        cases = [
            ("Bond_Fixed", {"contains": {"Interest", "Notional"}}),
            ("Bond_Floating", {"contains": {"Interest", "InterestProjected"}}),
            ("Bond_Fixed_Then_Floating", {"prefix": ["Interest"] * 5, "contains": {"InterestProjected"}}),
            ("Bond_Amortizing_FixedAmount", {"declining_nominal": True}),
            ("FwdBond_Fixed", {"forward": True}),
        ]
        for trade_id, checks in cases:
            with self.subTest(trade_id=trade_id):
                spec, _ = load_bond_trade_spec(
                    portfolio_xml=EXAMPLE_18_PORTFOLIO,
                    trade_id=trade_id,
                    reference_data_path=EXAMPLE_18_REFERENCE,
                    pricingengine_path=EXAMPLE_SHARED_PE,
                    flows_csv=EXAMPLE_18_FLOWS,
                )
                types = [cf.flow_type for cf in spec.cashflows]
                if "contains" in checks:
                    self.assertTrue(checks["contains"].issubset(set(types)))
                if "prefix" in checks:
                    self.assertEqual(types[: len(checks["prefix"])], checks["prefix"])
                if checks.get("declining_nominal"):
                    notionals = [cf.nominal for cf in spec.cashflows if cf.nominal is not None]
                    self.assertGreater(len(notionals), 2)
                    self.assertTrue(all(lhs >= rhs for lhs, rhs in zip(notionals, notionals[1:])))
                    self.assertLess(min(notionals), max(notionals))
                if checks.get("forward"):
                    self.assertEqual(spec.trade_type, "ForwardBond")
                    self.assertEqual(spec.forward_maturity_date, date(2016, 8, 8))
                    self.assertEqual(spec.forward_amount, 6300000.0)
                    self.assertTrue(spec.settlement_dirty)
                    self.assertTrue(spec.long_in_forward)

    def test_compiled_trade_matches_example18_fixed_shape(self):
        spec, engine = load_bond_trade_spec(
            portfolio_xml=EXAMPLE_18_PORTFOLIO,
            trade_id="Bond_Fixed",
            reference_data_path=EXAMPLE_18_REFERENCE,
            pricingengine_path=EXAMPLE_SHARED_PE,
            flows_csv=EXAMPLE_18_FLOWS,
        )
        compiled = compile_bond_trade(spec, asof_date="2016-02-05", day_counter="A365F", engine_spec=engine)
        self.assertEqual(compiled.trade_type, "Bond")
        self.assertEqual(compiled.pay_times.shape, (6,))
        self.assertEqual(compiled.amounts.shape, (6,))
        self.assertEqual(compiled.recovery_nominals.shape, (5,))
        self.assertEqual(compiled.recovery_start_times.shape, (5,))
        self.assertEqual(compiled.recovery_end_times.shape, (5,))

    def test_single_numpy_pricing_is_scenario_pricing_with_n1(self):
        spec = BondTradeSpec(
            trade_id="B1",
            trade_type="Bond",
            currency="EUR",
            payer=False,
            security_id="SEC",
            credit_curve_id="CPTY",
            reference_curve_id="EUR-EURIBOR-6M",
            income_curve_id="",
            settlement_days=2,
            calendar="TARGET",
            issue_date=date(2024, 1, 1),
            bond_notional=1.0,
            cashflows=(
                BondCashflow(pay_date=date(2025, 1, 1), amount=5.0, flow_type="Interest", accrual_start=date(2024, 1, 1), accrual_end=date(2025, 1, 1), nominal=100.0),
                BondCashflow(pay_date=date(2025, 1, 1), amount=100.0, flow_type="Notional", nominal=None),
            ),
        )
        compiled = compile_bond_trade(spec, asof_date="2024-01-01", day_counter="A365F", engine_spec=BondEngineSpec())
        curve = _flat_curve(0.05)
        grid = build_bond_scenario_grid_numpy(
            compiled,
            discount_curve=curve,
            income_curve=curve,
            hazard_times=np.asarray([10.0], dtype=float),
            hazard_rates=np.asarray([0.0], dtype=float),
            recovery_rate=0.0,
        )
        scalar = price_bond_single_numpy(compiled, grid)
        vector = price_bond_scenarios_numpy(compiled, grid)[0]
        self.assertAlmostEqual(scalar, vector, places=12)

    def test_numpy_and_torch_scenario_pricing_match(self):
        compiled = CompiledBondTrade(
            trade_id="B1",
            trade_type="Bond",
            currency="EUR",
            security_id="SEC",
            credit_curve_id="CPTY",
            reference_curve_id="EUR-EURIBOR-6M",
            income_curve_id="",
            bond_notional=1.0,
            pay_times=np.asarray([1.0, 2.0], dtype=float),
            amounts=np.asarray([5.0, 100.0], dtype=float),
            is_coupon=np.asarray([True, False], dtype=bool),
            recovery_nominals=np.asarray([100.0], dtype=float),
            recovery_start_times=np.asarray([0.0], dtype=float),
            recovery_end_times=np.asarray([1.0], dtype=float),
            maturity_time=2.0,
            engine_spec=BondEngineSpec(),
        )
        grid = BondScenarioGrid(
            discount_to_pay=np.asarray([[0.97, 0.94], [0.96, 0.92]], dtype=float),
            income_to_npv=np.asarray([1.0, 1.0], dtype=float),
            income_to_settlement=np.asarray([1.0, 1.0], dtype=float),
            survival_to_pay=np.asarray([[0.99, 0.97], [0.98, 0.95]], dtype=float),
            recovery_discount_mid=np.asarray([[0.985], [0.975]], dtype=float),
            recovery_default_prob=np.asarray([[0.01], [0.02]], dtype=float),
            recovery_rate=np.asarray([0.4, 0.4], dtype=float),
        )
        np_out = price_bond_scenarios_numpy(compiled, grid)
        if torch is None:
            self.skipTest("torch is not available in this environment")
        th_out = price_bond_scenarios_torch(compiled, grid).detach().cpu().numpy()
        np.testing.assert_allclose(np_out, th_out, rtol=0.0, atol=1.0e-12)

    def test_load_callable_bond_trade_spec_merges_reference_and_expands_american(self):
        spec, engine = load_callable_bond_trade_spec(
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondTrade",
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
        )
        self.assertEqual(spec.security_id, "SECURITY_CALL")
        self.assertEqual(spec.credit_curve_id, "CPTY_A")
        self.assertEqual(spec.reference_curve_id, "EUR-EURIBOR-3M")
        self.assertEqual(spec.bond_notional, 100000000.0)
        self.assertEqual(len(spec.call_data), 3)
        self.assertEqual([x.exercise_type for x in spec.call_data], ["FromThisDateOn", "FromThisDateOn", "OnThisDate"])
        self.assertEqual([x.price_type for x in spec.call_data], ["Clean", "Clean", "Clean"])
        self.assertTrue(all(x.include_accrual for x in spec.call_data))
        self.assertEqual(engine.model_family, "LGM")
        self.assertEqual(engine.exercise_time_steps_per_year, 24)

    def test_callable_price_amount_matches_clean_dirty_include_accrual_rules(self):
        clean_with_accrual = CallableExerciseSpec(date(2020, 1, 1), "OnThisDate", 1.0, "Clean", True)
        clean_without_accrual = CallableExerciseSpec(date(2020, 1, 1), "OnThisDate", 1.0, "Clean", False)
        dirty_with_accrual = CallableExerciseSpec(date(2020, 1, 1), "OnThisDate", 1.0, "Dirty", True)
        self.assertAlmostEqual(_callable_price_amount(clean_with_accrual, 100.0, 3.5), 103.5, places=12)
        self.assertAlmostEqual(_callable_price_amount(clean_without_accrual, 100.0, 3.5), 100.0, places=12)
        self.assertAlmostEqual(_callable_price_amount(dirty_with_accrual, 100.0, 3.5), 100.0, places=12)

    def test_callable_no_call_price_matches_stripped_bond_within_tolerance(self):
        out = price_bond_trade(
            ore_xml=CALLABLE_ORE_XML,
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondNoCall",
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
            flows_csv=None,
        )
        self.assertEqual(out["trade_type"], "CallableBond")
        self.assertAlmostEqual(float(out["py_npv"]), float(out["stripped_bond_npv"]), delta=5.0e5)
        self.assertAlmostEqual(float(out["embedded_option_value"]), 0.0, delta=5.0e5)

    def test_callable_certain_call_is_no_more_valuable_than_no_call(self):
        no_call = price_bond_trade(
            ore_xml=CALLABLE_ORE_XML,
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondNoCall",
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
            flows_csv=None,
        )
        certain_call = price_bond_trade(
            ore_xml=CALLABLE_ORE_XML,
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondCertainCall",
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
            flows_csv=None,
        )
        self.assertLess(float(certain_call["py_npv"]), float(no_call["py_npv"]))
        self.assertGreater(float(certain_call["embedded_option_value"]), 0.0)

    def test_callable_put_overrides_call_and_increases_value(self):
        call_only = price_bond_trade(
            ore_xml=CALLABLE_ORE_XML,
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondTrade",
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
            flows_csv=None,
        )
        put_call = price_bond_trade(
            ore_xml=CALLABLE_ORE_XML,
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="PutCallBondTrade",
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
            flows_csv=None,
        )
        self.assertGreater(float(put_call["py_npv"]), float(call_only["py_npv"]))
        self.assertEqual(int(put_call["put_schedule_count"]), 3)
        self.assertEqual(int(put_call["call_schedule_count"]), 3)

    def test_put_call_variant_put_only_is_more_valuable_than_call_only(self):
        call_only = self._price_callable_variant(
            "PutCallBondTrade",
            spec_override=lambda s: replace(s, put_data=()),
        )
        put_only = self._price_callable_variant(
            "PutCallBondTrade",
            spec_override=lambda s: replace(s, call_data=()),
        )
        both = self._price_callable_variant("PutCallBondTrade")
        self.assertGreater(float(put_only["py_npv"]), float(both["py_npv"]))
        self.assertGreater(float(both["py_npv"]), float(call_only["py_npv"]))

    def test_put_call_variant_american_and_bermudan_currently_coincide(self):
        def _to_bermudan(spec):
            calls = tuple(replace(x, exercise_type="OnThisDate") for x in spec.call_data)
            puts = tuple(replace(x, exercise_type="OnThisDate") for x in spec.put_data)
            return replace(spec, call_data=calls, put_data=puts)

        american = self._price_callable_variant("PutCallBondTrade")
        bermudan = self._price_callable_variant("PutCallBondTrade", spec_override=_to_bermudan)
        # This is intentionally a diagnostic regression test rather than a
        # finance-theory assertion. In the current Python callable engine, the
        # additional American grid points in this mixed put/call case do not
        # change the optimal exercise result versus Bermudan-only dates. If
        # that changes in a future parity pass, this test should be updated
        # deliberately rather than silently drifting.
        self.assertAlmostEqual(float(american["py_npv"]), float(bermudan["py_npv"]), delta=1.0)

    def test_put_call_variant_include_accrual_changes_value(self):
        with_accrual = self._price_callable_variant("PutCallBondTrade")
        without_accrual = self._price_callable_variant(
            "PutCallBondTrade",
            spec_override=lambda s: replace(
                s,
                call_data=tuple(replace(x, include_accrual=False) for x in s.call_data),
                put_data=tuple(replace(x, include_accrual=False) for x in s.put_data),
            ),
        )
        self.assertNotAlmostEqual(float(with_accrual["py_npv"]), float(without_accrual["py_npv"]), delta=1.0)

    def test_put_call_variant_dirty_prices_change_value(self):
        clean = self._price_callable_variant("PutCallBondTrade")
        dirty = self._price_callable_variant(
            "PutCallBondTrade",
            spec_override=lambda s: replace(
                s,
                call_data=tuple(replace(x, price_type="Dirty") for x in s.call_data),
                put_data=tuple(replace(x, price_type="Dirty") for x in s.put_data),
            ),
        )
        self.assertNotAlmostEqual(float(clean["py_npv"]), float(dirty["py_npv"]), delta=1.0)

    def test_put_call_variant_exercise_step_density_is_stable(self):
        coarse = self._price_callable_variant(
            "PutCallBondTrade",
            engine_override=lambda e: replace(e, exercise_time_steps_per_year=12),
        )
        base = self._price_callable_variant("PutCallBondTrade")
        dense = self._price_callable_variant(
            "PutCallBondTrade",
            engine_override=lambda e: replace(e, exercise_time_steps_per_year=48),
        )
        self.assertGreater(float(coarse["py_npv"]), float(base["py_npv"]))
        self.assertGreater(float(base["py_npv"]), float(dense["py_npv"]))
        self.assertLess(abs(float(coarse["py_npv"]) - float(dense["py_npv"])), 5.0e4)

    def test_callable_lgm_grid_parity_against_native_ore_npv(self):
        tolerances = {
            "CallableBondTrade": 3.2e6,
            "CallableBondNoCall": 5.1e5,
            "CallableBondCertainCall": 1.4e6,
            "PutCallBondTrade": 2.15e6,
        }
        for trade_id, tol in tolerances.items():
            with self.subTest(trade_id=trade_id):
                out = price_bond_trade(
                    ore_xml=CALLABLE_LGM_GRID_ORE_XML,
                    portfolio_xml=CALLABLE_PORTFOLIO,
                    trade_id=trade_id,
                    asof_date="2016-02-05",
                    model_day_counter="A365F",
                    market_data_file=EXAMPLE_SHARED_MARKET_FULL,
                    todaysmarket_xml=EXAMPLE_SHARED_TM,
                    reference_data_path=CALLABLE_REFERENCE,
                    pricingengine_path=CALLABLE_LGM_GRID_PE,
                    flows_csv=None,
                )
                ore_npv = _load_ore_npv_details(CALLABLE_LGM_GRID_NPV, trade_id=trade_id)["npv"]
                self.assertLess(abs(float(out["py_npv"]) - float(ore_npv)), tol)

    def test_positive_hazard_reduces_bond_value(self):
        spec = BondTradeSpec(
            trade_id="B1",
            trade_type="Bond",
            currency="EUR",
            payer=False,
            security_id="SEC",
            credit_curve_id="CPTY",
            reference_curve_id="EUR-EURIBOR-6M",
            income_curve_id="",
            settlement_days=2,
            calendar="TARGET",
            issue_date=date(2024, 1, 1),
            bond_notional=1.0,
            cashflows=(BondCashflow(pay_date=date(2026, 1, 1), amount=100.0, flow_type="Notional", nominal=100.0),),
        )
        curve = _flat_curve(0.03)
        low_hazard, _ = _bond_npv(
            spec,
            asof_date=date(2024, 1, 1),
            day_counter="A365F",
            discount_curve=curve,
            income_curve=curve,
            hazard_times=np.asarray([10.0], dtype=float),
            hazard_rates=np.asarray([0.0], dtype=float),
            recovery_rate=0.0,
            security_spread=0.0,
            engine_spec=BondEngineSpec(),
        )
        high_hazard, _ = _bond_npv(
            spec,
            asof_date=date(2024, 1, 1),
            day_counter="A365F",
            discount_curve=curve,
            income_curve=curve,
            hazard_times=np.asarray([10.0], dtype=float),
            hazard_rates=np.asarray([0.05], dtype=float),
            recovery_rate=0.0,
            security_spread=0.0,
            engine_spec=BondEngineSpec(),
        )
        self.assertLess(high_hazard, low_hazard)

    def test_recovery_increases_bond_value_under_credit_risk(self):
        spec = BondTradeSpec(
            trade_id="B1",
            trade_type="Bond",
            currency="EUR",
            payer=False,
            security_id="SEC",
            credit_curve_id="CPTY",
            reference_curve_id="EUR-EURIBOR-6M",
            income_curve_id="",
            settlement_days=2,
            calendar="TARGET",
            issue_date=date(2024, 1, 1),
            bond_notional=1.0,
            cashflows=(BondCashflow(pay_date=date(2026, 1, 1), amount=100.0, flow_type="Notional", nominal=100.0),),
        )
        curve = _flat_curve(0.03)
        no_rec, _ = _bond_npv(
            spec,
            asof_date=date(2024, 1, 1),
            day_counter="A365F",
            discount_curve=curve,
            income_curve=curve,
            hazard_times=np.asarray([10.0], dtype=float),
            hazard_rates=np.asarray([0.05], dtype=float),
            recovery_rate=0.0,
            security_spread=0.0,
            engine_spec=BondEngineSpec(),
        )
        with_rec, _ = _bond_npv(
            spec,
            asof_date=date(2024, 1, 1),
            day_counter="A365F",
            discount_curve=curve,
            income_curve=curve,
            hazard_times=np.asarray([10.0], dtype=float),
            hazard_rates=np.asarray([0.05], dtype=float),
            recovery_rate=0.4,
            security_spread=0.0,
            engine_spec=BondEngineSpec(),
        )
        self.assertGreater(with_rec, no_rec)

    def test_example18_fixed_bond_parity_stays_within_regression_band(self):
        out, ore_npv = self._price_example18("Bond_Fixed")
        self.assertLess(abs(out["py_npv"] - ore_npv), 260000.0)

    def test_example18_forward_bond_parity_stays_within_regression_band(self):
        out, ore_npv = self._price_example18("FwdBond_Fixed")
        self.assertLess(abs(out["py_npv"] - ore_npv), 70000.0)

    def test_example18_bond_matrix_has_near_exact_parity(self):
        trade_ids = [
            "Bond_Fixed",
            "Bond_Floating",
            "Bond_Fixed_Then_Floating",
            "Bond_Amortizing_FixedAmount",
            "Bond_Amortizing_Percentage_Initial",
            "Bond_Amortizing_Percentage_Previous",
            "Bond_Amortizing_Fixed_Annuity",
            "Bond_Amortizing_Floating_Annuity",
            "Bond_Amortizing_FixedAmount_PercentagePrevious",
            "Bond_Fixed_using_BMBond_curve_Pricing",
        ]
        for trade_id in trade_ids:
            with self.subTest(trade_id=trade_id):
                out, ore_npv = self._price_example18(trade_id)
                self.assertEqual(out["trade_type"], "Bond")
                self.assertLess(abs(out["py_npv"] - ore_npv), 1.0e-2)

    def test_example18_forward_bond_matrix_stays_in_tight_band(self):
        trade_ids = [
            "FwdBond_Fixed",
            "FwdBond_Floating",
            "FwdBond_Fixed_Then_Floating",
            "FwdBond_Amortizing_FixedAmount",
            "FwdBond_Amortizing_Percentage_Initial",
            "FwdBond_Amortizing_Percentage_Previous",
            "FwdBond_Amortizing_Fixed_Annuity",
            "FwdBond_Amortizing_Floating_Annuity",
            "FwdBond_Amortizing_FixedAmount_PercentagePrevious",
        ]
        for trade_id in trade_ids:
            with self.subTest(trade_id=trade_id):
                out, ore_npv = self._price_example18(trade_id)
                self.assertEqual(out["trade_type"], "ForwardBond")
                self.assertLess(abs(out["py_npv"] - ore_npv), 2500.0)

    def test_example78_fixed_rate_bond_parity_is_near_exact(self):
        out = price_bond_trade(
            ore_xml=EXAMPLE_78_ORE_XML,
            portfolio_xml=EXAMPLE_78_PORTFOLIO,
            trade_id="FixedRateBond",
            asof_date="2022-03-01",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_78_ORE_XML.parent / "MD_2022-03-01.csv",
            todaysmarket_xml=EXAMPLE_78_ORE_XML.parent / "todaysmarket.xml",
            reference_data_path=EXAMPLE_78_REFERENCE,
            pricingengine_path=EXAMPLE_78_ORE_XML.parent / "pricingengine.xml",
            flows_csv=EXAMPLE_78_FLOWS,
        )
        ore_npv = _load_ore_npv_details(EXAMPLE_78_NPV, trade_id="FixedRateBond")["npv"]
        self.assertLess(abs(out["py_npv"] - ore_npv), 1.0e-2)


if __name__ == "__main__":
    unittest.main()
