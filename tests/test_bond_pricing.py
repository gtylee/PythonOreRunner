import sys
import unittest
import tempfile
import shutil
import subprocess
from dataclasses import replace
from datetime import date, timedelta
import csv
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
    CallableBondScenarioPack,
    CallableExerciseSpec,
    CompiledBondTrade,
    CompiledCallableBondTrade,
    _callable_price_amount,
    _build_lgm_model_for_callable,
    _fit_curve_for_currency,
    _load_callable_option_curve_from_reference_output,
    _load_curve_from_reference_output,
    _accrued_amount,
    _load_security_spread,
    _parse_reference_grid_dates,
    _price_callable_bond_lgm,
    _time_from_dates,
    build_bond_scenario_grid_numpy,
    build_bond_scenario_grid_from_scenarios,
    build_callable_bond_scenario_pack,
    build_callable_bond_scenario_pack_from_arrays,
    compile_bond_trade,
    compile_callable_bond_trade,
    _bond_npv,
    _curve_from_flow_discounts,
    _adjust_date,
    _load_bond_cashflows_from_flows,
    load_bond_trade_spec,
    load_callable_bond_trade_spec,
    price_bond_scenarios_numpy,
    price_bond_scenarios_torch,
    price_callable_bond_scenarios_numpy,
    price_callable_bond_scenarios_torch,
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
CALLABLE_LGM_GRID_CURVES_ORE_XML = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_callable_bond_lgm_grid_curves.xml"
CALLABLE_LGM_GRID_PE = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "pricingengine_callablebond_lgm_grid.xml"
CALLABLE_LGM_GRID_NPV = TOOLS_DIR / "Examples" / "Exposure" / "Output" / "callable_bond_lgm_grid_npv_only" / "npv.csv"
CALLABLE_LGM_GRID_ADDITIONAL = TOOLS_DIR / "Examples" / "Exposure" / "Output" / "callable_bond_lgm_grid_npv_additional" / "additional_results.csv"
EXAMPLE_SHARED_MARKET_FULL = TOOLS_DIR / "Examples" / "Input" / "market_20160205.txt"
AMC_FORWARDBOND_ORE_XML = TOOLS_DIR / "Examples" / "AmericanMonteCarlo" / "Input" / "ore_forwardbond.xml"
AMC_FORWARDBOND_PORTFOLIO = TOOLS_DIR / "Examples" / "AmericanMonteCarlo" / "Input" / "portfolio_forwardbond.xml"
AMC_FORWARDBOND_REFERENCE = TOOLS_DIR / "Examples" / "AmericanMonteCarlo" / "Input" / "referencedata.xml"
AMC_FORWARDBOND_PE = TOOLS_DIR / "Examples" / "AmericanMonteCarlo" / "Input" / "pricingengine.xml"
AMC_FORWARDBOND_TM = TOOLS_DIR / "Examples" / "AmericanMonteCarlo" / "Input" / "todaysmarket.xml"
AMC_FORWARDBOND_MARKET = TOOLS_DIR / "Examples" / "AmericanMonteCarlo" / "Input" / "market.txt"
LOCAL_ORE_BINARY = Path("/Users/gordonlee/Documents/Engine/build/App/ore")


def _flat_curve(rate: float):
    return lambda t: 1.0 if float(t) <= 0.0 else float(pow(2.718281828459045, -rate * float(t)))


def _shift_curve(curve, shift: float):
    return lambda t, curve=curve, shift=shift: float(curve(float(t))) * float(np.exp(-float(shift) * max(float(t), 0.0)))


class TestBondPricing(unittest.TestCase):
    def _callable_portfolio_with_underlying_bond(self, tmp_root: Path) -> tuple[Path, Path]:
        input_dir = tmp_root / "Input"
        shutil.copytree(CALLABLE_PORTFOLIO.parent, input_dir)
        shared_input = tmp_root.parent / "Input"
        if shared_input.exists():
            shutil.rmtree(shared_input)
        shutil.copytree(TOOLS_DIR / "Examples" / "Input", shared_input)

        portfolio = input_dir / "portfolio_callablebond.xml"
        text = portfolio.read_text(encoding="utf-8")
        text = text.replace('  <!--Trade id="UnderlyingBondTrade">', '  <Trade id="UnderlyingBondTrade">')
        text = text.replace("  </Trade-->", "  </Trade>")
        portfolio.write_text(text, encoding="utf-8")

        ore_xml = input_dir / "ore_callable_bond_lgm_grid_npv_only.xml"
        text = ore_xml.read_text(encoding="utf-8")
        text = text.replace("Output/callable_bond_lgm_grid_npv_only", "Output/callable_bond_compare")
        ore_xml.write_text(text, encoding="utf-8")
        return ore_xml, portfolio

    def _american_forwardbond_fixture(self, trade_id: str):
        spec, engine = load_bond_trade_spec(
            portfolio_xml=AMC_FORWARDBOND_PORTFOLIO,
            trade_id=trade_id,
            reference_data_path=AMC_FORWARDBOND_REFERENCE,
            pricingengine_path=AMC_FORWARDBOND_PE,
            flows_csv=None,
        )
        asof = date(2022, 1, 31)
        curve = _fit_curve_for_currency(AMC_FORWARDBOND_ORE_XML, spec.currency)
        if spec.credit_curve_id:
            credit = load_ore_default_curve_inputs(
                str(AMC_FORWARDBOND_TM),
                str(AMC_FORWARDBOND_MARKET),
                cpty_name=spec.credit_curve_id,
            )
            hazard_times = np.asarray(credit["hazard_times"], dtype=float)
            hazard_rates = np.asarray(credit["hazard_rates"], dtype=float)
            recovery_rate = float(credit["recovery"])
        else:
            hazard_times = np.asarray([50.0], dtype=float)
            hazard_rates = np.asarray([0.0], dtype=float)
            recovery_rate = 0.0
        return {
            "spec": spec,
            "engine": engine,
            "compiled": compile_bond_trade(spec, asof_date=asof, day_counter="A365F", engine_spec=engine),
            "curve": curve,
            "hazard_times": hazard_times,
            "hazard_rates": hazard_rates,
            "recovery_rate": recovery_rate,
            "security_spread": _load_security_spread(AMC_FORWARDBOND_MARKET, spec.security_id),
            "asof": asof,
        }

    def _load_native_callable_event_rows(self, trade_id: str):
        rows = []
        with open(CALLABLE_LGM_GRID_ADDITIONAL, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("#TradeId") != trade_id:
                    continue
                result_id = row.get("ResultId", "")
                if not result_id.startswith("event_") or result_id.endswith("!"):
                    continue
                payload = row.get("ResultValue", "")
                cols = [c.strip() for c in payload.split("|")[1:-1]]
                if len(cols) != 11:
                    continue
                rows.append(
                    {
                        "time": float(cols[0]) if cols[0] else 0.0,
                        "date": cols[1],
                        "notional": float(cols[2]) if cols[2] else 0.0,
                        "accrual": float(cols[3]) if cols[3] else 0.0,
                        "flow": float(cols[4]) if cols[4] else 0.0,
                        "call": float(cols[5].replace("@", "")) if cols[5] else None,
                        "put": float(cols[6].replace("@", "")) if cols[6] else None,
                    }
                )
        return rows

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
            include_trace=True,
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

    def test_multi_scenario_grid_matches_scalar_for_american_bond_example(self):
        fx = self._american_forwardbond_fixture("NoFwdContract")
        shifts = np.asarray([-0.0040, -0.0015, 0.0, 0.0020, 0.0050], dtype=float)
        hazard_bumps = np.asarray([0.0000, 0.0010, 0.0025, 0.0040, 0.0060], dtype=float)
        recovery_rates = np.clip(fx["recovery_rate"] + np.asarray([0.0, -0.02, 0.01, -0.01, 0.015], dtype=float), 0.0, 0.95)
        security_spreads = fx["security_spread"] + np.asarray([0.0, 0.0005, 0.0010, -0.0003, 0.0015], dtype=float)
        discount_curves = [_shift_curve(fx["curve"], s) for s in shifts]
        hazard_rates = np.clip(fx["hazard_rates"][None, :] + hazard_bumps[:, None], 0.0, None)

        vector_grid = build_bond_scenario_grid_from_scenarios(
            fx["compiled"],
            discount_curves=discount_curves,
            income_curves=discount_curves,
            hazard_times=fx["hazard_times"],
            hazard_rates=hazard_rates,
            recovery_rate=recovery_rates,
            security_spread=security_spreads,
            engine_spec=fx["engine"],
        )
        vector_out = price_bond_scenarios_numpy(fx["compiled"], vector_grid)

        scalar_out = []
        for i in range(len(discount_curves)):
            scalar_grid = build_bond_scenario_grid_numpy(
                fx["compiled"],
                discount_curve=discount_curves[i],
                income_curve=discount_curves[i],
                hazard_times=fx["hazard_times"],
                hazard_rates=hazard_rates[i],
                recovery_rate=float(recovery_rates[i]),
                security_spread=float(security_spreads[i]),
                engine_spec=fx["engine"],
            )
            scalar_out.append(price_bond_single_numpy(fx["compiled"], scalar_grid))
        np.testing.assert_allclose(vector_out, np.asarray(scalar_out, dtype=float), rtol=0.0, atol=1.0e-10)

    def test_multi_scenario_grid_matches_scalar_for_american_forwardbond_example(self):
        fx = self._american_forwardbond_fixture("FwdBond")
        shifts = np.asarray([-0.0035, -0.0010, 0.0, 0.0015, 0.0045], dtype=float)
        hazard_bumps = np.asarray([0.0000, 0.0010, 0.0020, 0.0035, 0.0050], dtype=float)
        recovery_rates = np.clip(fx["recovery_rate"] + np.asarray([0.0, -0.015, 0.01, -0.005, 0.02], dtype=float), 0.0, 0.95)
        security_spreads = fx["security_spread"] + np.asarray([0.0, 0.0004, 0.0008, -0.0002, 0.0012], dtype=float)
        discount_curves = [_shift_curve(fx["curve"], s) for s in shifts]
        hazard_rates = np.clip(fx["hazard_rates"][None, :] + hazard_bumps[:, None], 0.0, None)

        bond_settlement_date = _adjust_date(
            fx["spec"].forward_maturity_date + timedelta(days=fx["spec"].settlement_days),
            "F",
            fx["spec"].calendar,
        )
        accrued_at_bond_settlement = np.full(
            len(discount_curves),
            float(_accrued_amount(fx["spec"], bond_settlement_date)),
            dtype=float,
        )
        payoff_time = max(_time_from_dates(fx["asof"], bond_settlement_date, "A365F"), 0.0)
        payoff_discount = np.asarray([float(curve(payoff_time)) for curve in discount_curves], dtype=float)
        premium_discount = None
        if fx["spec"].compensation_payment and fx["spec"].compensation_payment_date and fx["spec"].compensation_payment_date > fx["asof"]:
            premium_time = max(_time_from_dates(fx["asof"], fx["spec"].compensation_payment_date, "A365F"), 0.0)
            premium_discount = np.asarray([float(curve(premium_time)) for curve in discount_curves], dtype=float)

        forward_dirty_value = []
        scalar_out = []
        for i in range(len(discount_curves)):
            _, forward_value = _bond_npv(
                fx["spec"],
                asof_date=fx["asof"],
                day_counter="A365F",
                discount_curve=discount_curves[i],
                income_curve=discount_curves[i],
                hazard_times=fx["hazard_times"],
                hazard_rates=hazard_rates[i],
                recovery_rate=float(recovery_rates[i]),
                security_spread=float(security_spreads[i]),
                engine_spec=fx["engine"],
                npv_date=fx["spec"].forward_maturity_date,
                settlement_date=fx["spec"].forward_maturity_date,
                conditional_on_survival=True,
            )
            forward_dirty_value.append(forward_value)
            scalar_grid = build_bond_scenario_grid_numpy(
                fx["compiled"],
                discount_curve=discount_curves[i],
                income_curve=discount_curves[i],
                hazard_times=fx["hazard_times"],
                hazard_rates=hazard_rates[i],
                recovery_rate=float(recovery_rates[i]),
                security_spread=float(security_spreads[i]),
                engine_spec=fx["engine"],
                forward_dirty_value=float(forward_value),
                accrued_at_bond_settlement=float(accrued_at_bond_settlement[i]),
                payoff_discount=float(payoff_discount[i]),
                premium_discount=None if premium_discount is None else float(premium_discount[i]),
            )
            scalar_out.append(price_bond_single_numpy(fx["compiled"], scalar_grid))

        vector_grid = build_bond_scenario_grid_from_scenarios(
            fx["compiled"],
            discount_curves=discount_curves,
            income_curves=discount_curves,
            hazard_times=fx["hazard_times"],
            hazard_rates=hazard_rates,
            recovery_rate=recovery_rates,
            security_spread=security_spreads,
            engine_spec=fx["engine"],
            forward_dirty_value=np.asarray(forward_dirty_value, dtype=float),
            accrued_at_bond_settlement=accrued_at_bond_settlement,
            payoff_discount=payoff_discount,
            premium_discount=premium_discount,
        )
        vector_out = price_bond_scenarios_numpy(fx["compiled"], vector_grid)
        np.testing.assert_allclose(vector_out, np.asarray(scalar_out, dtype=float), rtol=0.0, atol=1.0e-10)
        if torch is not None:
            torch_out = price_bond_scenarios_torch(fx["compiled"], vector_grid).detach().cpu().numpy()
            np.testing.assert_allclose(vector_out, torch_out, rtol=0.0, atol=1.0e-10)

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

    def test_callable_batch_scenario_numpy_matches_scalar_price(self):
        spec, engine = load_callable_bond_trade_spec(
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondTrade",
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
        )
        compiled = compile_callable_bond_trade(
            spec,
            engine,
            asof_date="2016-02-05",
            day_counter="A365F",
        )
        model = _build_lgm_model_for_callable(
            ore_xml=CALLABLE_ORE_XML,
            pricingengine_path=CALLABLE_PE,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            currency=spec.currency,
            maturity_date=max(cf.pay_date for cf in spec.bond.cashflows),
            asof_date=date(2016, 2, 5),
        )
        native_curve = _load_callable_option_curve_from_reference_output(
            CALLABLE_ORE_XML,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            curve_id=spec.reference_curve_id,
            asof_date="2016-02-05",
            day_counter="A365F",
        )
        rollback_curve = native_curve[0] if native_curve is not None else _fit_curve_for_currency(CALLABLE_ORE_XML, spec.currency)
        stripped_curve = _fit_curve_for_currency(CALLABLE_ORE_XML, spec.currency)
        credit = load_ore_default_curve_inputs(
            str(EXAMPLE_SHARED_TM),
            str(EXAMPLE_SHARED_MARKET_FULL),
            cpty_name=spec.credit_curve_id,
        )
        security_spread = _load_security_spread(EXAMPLE_SHARED_MARKET_FULL, spec.security_id)
        pack = build_callable_bond_scenario_pack(
            compiled,
            reference_curves=[rollback_curve, rollback_curve],
            models=[model, model],
            hazard_times=np.asarray(credit["hazard_times"], dtype=float),
            hazard_rates=np.repeat(np.asarray(credit["hazard_rates"], dtype=float).reshape(1, -1), 2, axis=0),
            recovery_rate=np.asarray([float(credit["recovery"]), float(credit["recovery"])], dtype=float),
            security_spread=np.asarray([security_spread, security_spread], dtype=float),
            stripped_discount_curves=[stripped_curve, stripped_curve],
            stripped_income_curves=[stripped_curve, stripped_curve],
        )
        batch = price_callable_bond_scenarios_numpy(compiled, pack, chunk_size=1)
        scalar = price_bond_trade(
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
        np.testing.assert_allclose(batch, np.asarray([scalar["py_npv"], scalar["py_npv"]], dtype=float), rtol=0.0, atol=2.0e-8)

    @unittest.skipIf(torch is None, "torch not available")
    def test_callable_batch_scenario_torch_matches_numpy(self):
        spec, engine = load_callable_bond_trade_spec(
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondTrade",
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
        )
        compiled = compile_callable_bond_trade(
            spec,
            engine,
            asof_date="2016-02-05",
            day_counter="A365F",
        )
        model = _build_lgm_model_for_callable(
            ore_xml=CALLABLE_ORE_XML,
            pricingengine_path=CALLABLE_PE,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            currency=spec.currency,
            maturity_date=max(cf.pay_date for cf in spec.bond.cashflows),
            asof_date=date(2016, 2, 5),
        )
        native_curve = _load_callable_option_curve_from_reference_output(
            CALLABLE_ORE_XML,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            curve_id=spec.reference_curve_id,
            asof_date="2016-02-05",
            day_counter="A365F",
        )
        rollback_curve = native_curve[0] if native_curve is not None else _fit_curve_for_currency(CALLABLE_ORE_XML, spec.currency)
        stripped_curve = _fit_curve_for_currency(CALLABLE_ORE_XML, spec.currency)
        credit = load_ore_default_curve_inputs(
            str(EXAMPLE_SHARED_TM),
            str(EXAMPLE_SHARED_MARKET_FULL),
            cpty_name=spec.credit_curve_id,
        )
        security_spread = _load_security_spread(EXAMPLE_SHARED_MARKET_FULL, spec.security_id)
        pack = build_callable_bond_scenario_pack(
            compiled,
            reference_curves=[rollback_curve, rollback_curve],
            models=[model, model],
            hazard_times=np.asarray(credit["hazard_times"], dtype=float),
            hazard_rates=np.repeat(np.asarray(credit["hazard_rates"], dtype=float).reshape(1, -1), 2, axis=0),
            recovery_rate=np.asarray([float(credit["recovery"]), float(credit["recovery"])], dtype=float),
            security_spread=np.asarray([security_spread, security_spread], dtype=float),
            stripped_discount_curves=[stripped_curve, stripped_curve],
            stripped_income_curves=[stripped_curve, stripped_curve],
        )
        npv_numpy = price_callable_bond_scenarios_numpy(compiled, pack, chunk_size=2)
        npv_torch = price_callable_bond_scenarios_torch(compiled, pack, chunk_size=2, device="cpu").detach().cpu().numpy()
        np.testing.assert_allclose(npv_torch, npv_numpy, rtol=0.0, atol=2.0e-8)

    def test_callable_batch_scenario_pack_from_arrays_matches_builder(self):
        spec, engine = load_callable_bond_trade_spec(
            portfolio_xml=CALLABLE_PORTFOLIO,
            trade_id="CallableBondTrade",
            reference_data_path=CALLABLE_REFERENCE,
            pricingengine_path=CALLABLE_PE,
        )
        compiled = compile_callable_bond_trade(spec, engine, asof_date="2016-02-05", day_counter="A365F")
        model = _build_lgm_model_for_callable(
            ore_xml=CALLABLE_ORE_XML,
            pricingengine_path=CALLABLE_PE,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            market_data_file=EXAMPLE_SHARED_MARKET_FULL,
            currency=spec.currency,
            maturity_date=max(cf.pay_date for cf in spec.bond.cashflows),
            asof_date=date(2016, 2, 5),
        )
        native_curve = _load_callable_option_curve_from_reference_output(
            CALLABLE_ORE_XML,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            curve_id=spec.reference_curve_id,
            asof_date="2016-02-05",
            day_counter="A365F",
        )
        rollback_curve = native_curve[0] if native_curve is not None else _fit_curve_for_currency(CALLABLE_ORE_XML, spec.currency)
        stripped_curve = _fit_curve_for_currency(CALLABLE_ORE_XML, spec.currency)
        credit = load_ore_default_curve_inputs(
            str(EXAMPLE_SHARED_TM),
            str(EXAMPLE_SHARED_MARKET_FULL),
            cpty_name=spec.credit_curve_id,
        )
        security_spread = _load_security_spread(EXAMPLE_SHARED_MARKET_FULL, spec.security_id)
        built = build_callable_bond_scenario_pack(
            compiled,
            reference_curves=[rollback_curve, rollback_curve],
            models=[model, model],
            hazard_times=np.asarray(credit["hazard_times"], dtype=float),
            hazard_rates=np.repeat(np.asarray(credit["hazard_rates"], dtype=float).reshape(1, -1), 2, axis=0),
            recovery_rate=np.asarray([float(credit["recovery"]), float(credit["recovery"])], dtype=float),
            security_spread=np.asarray([security_spread, security_spread], dtype=float),
            stripped_discount_curves=[stripped_curve, stripped_curve],
            stripped_income_curves=[stripped_curve, stripped_curve],
        )
        packed = build_callable_bond_scenario_pack_from_arrays(
            compiled,
            p0_grid=built.p0_grid,
            h_grid=built.h_grid,
            zeta_grid=built.zeta_grid,
            stripped_grid=built.stripped_grid,
        )
        np.testing.assert_allclose(packed.p0_grid, built.p0_grid, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(packed.h_grid, built.h_grid, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(packed.zeta_grid, built.zeta_grid, rtol=0.0, atol=0.0)

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

    def test_put_call_trace_aligns_with_native_event_schedule(self):
        out = self._price_callable_variant("PutCallBondTrade")
        trace_rows = list(out["trace_rows"])
        native_rows = self._load_native_callable_event_rows("PutCallBondTrade")
        native_ex_rows = [r for r in native_rows if r["call"] is not None or r["put"] is not None]
        py_rows = []
        for ore_row in native_ex_rows:
            match = min(trace_rows, key=lambda r: abs(r.time - ore_row["time"]))
            py_rows.append(match)
        self.assertEqual(len(py_rows), len(native_ex_rows))
        for py_row, ore_row in zip(py_rows, native_ex_rows):
            with self.subTest(time=ore_row["time"]):
                self.assertAlmostEqual(py_row.time, ore_row["time"], delta=2.0e-3)
                self.assertAlmostEqual(py_row.notional / 1.0e8, ore_row["notional"], delta=1.0e-10)
                self.assertAlmostEqual(py_row.accrual / 1.0e8, ore_row["accrual"], delta=1.0e-3)
                if ore_row["call"] is None:
                    self.assertIsNone(py_row.call_price)
                else:
                    self.assertAlmostEqual(float(py_row.call_price), ore_row["call"], delta=1.0e-12)
                if ore_row["put"] is None:
                    self.assertIsNone(py_row.put_price)
                else:
                    self.assertAlmostEqual(float(py_row.put_price), ore_row["put"], delta=1.0e-12)

    def test_reference_calibration_grid_uses_ore_dategrid_following_adjustment(self):
        dates = _parse_reference_grid_dates(date(2016, 2, 5), date(2024, 2, 26), "400,3M")
        self.assertGreaterEqual(len(dates), 4)
        self.assertEqual(dates[:4], [date(2016, 5, 5), date(2016, 8, 5), date(2016, 11, 7), date(2017, 2, 6)])

    def test_callable_reference_curve_loader_matches_native_event_refdsc(self):
        curve_info = _load_curve_from_reference_output(
            CALLABLE_LGM_GRID_CURVES_ORE_XML,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            curve_id="EUR-EURIBOR-3M",
            asof_date="2016-02-05",
            day_counter="A365F",
        )
        self.assertIsNotNone(curve_info)
        curve, column, curves_csv = curve_info
        self.assertEqual(column, "EUR-EURIBOR-3M")
        self.assertTrue(Path(curves_csv).exists())

        native_rows = self._load_native_callable_event_rows("PutCallBondTrade")
        native_ex_rows = [r for r in native_rows if r["call"] is not None or r["put"] is not None]
        expected = {
            4.72329: 1.00058,
            5.06027: 0.99924,
            6.72329: 0.987045,
        }
        for row in native_ex_rows:
            key = round(float(row["time"]), 5)
            if key not in expected:
                continue
            with self.subTest(time=key):
                self.assertAlmostEqual(float(curve(float(row["time"]))), expected[key], delta=2.0e-3)

    def test_callable_option_curve_loader_finds_sibling_curves_output(self):
        curve_info = _load_callable_option_curve_from_reference_output(
            CALLABLE_LGM_GRID_ORE_XML,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            curve_id="EUR-EURIBOR-3M",
            asof_date="2016-02-05",
            day_counter="A365F",
        )
        self.assertIsNotNone(curve_info)
        _, column, curves_csv = curve_info
        self.assertEqual(column, "EUR-EURIBOR-3M")
        self.assertIn("callable_bond_lgm_grid_curves", str(curves_csv))

    def test_callable_lgm_grid_parity_against_native_ore_npv(self):
        tolerances = {
            "CallableBondTrade": 4.0e5,
            "CallableBondNoCall": 5.1e5,
            "CallableBondCertainCall": 3.0e5,
            "PutCallBondTrade": 3.0e5,
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
                self.assertEqual(out.get("callable_option_reference_curve_column"), "EUR-EURIBOR-3M")
                ore_npv = _load_ore_npv_details(CALLABLE_LGM_GRID_NPV, trade_id=trade_id)["npv"]
                self.assertLess(abs(float(out["py_npv"]) - float(ore_npv)), tol)

    @unittest.skipUnless(LOCAL_ORE_BINARY.exists(), "local ORE binary not available")
    def test_native_ore_callable_no_call_differs_from_equivalent_plain_bond(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR / "tmp") as tmpdir:
            tmp_root = Path(tmpdir)
            ore_xml, _ = self._callable_portfolio_with_underlying_bond(tmp_root)
            proc = subprocess.run(
                [str(LOCAL_ORE_BINARY), "Input/ore_callable_bond_lgm_grid_npv_only.xml"],
                cwd=tmp_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            npv_csv = tmp_root / "Output" / "callable_bond_compare" / "npv.csv"
            self.assertTrue(npv_csv.exists())
            rows: dict[str, float] = {}
            with npv_csv.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows[row["#TradeId"]] = float(row["NPV"])
            self.assertIn("CallableBondNoCall", rows)
            self.assertIn("UnderlyingBondTrade", rows)
            self.assertGreater(rows["UnderlyingBondTrade"], rows["CallableBondNoCall"])
            self.assertGreater(rows["UnderlyingBondTrade"] - rows["CallableBondNoCall"], 2.0e6)

    @unittest.expectedFailure
    def test_python_callable_no_call_differs_from_equivalent_plain_bond(self):
        with tempfile.TemporaryDirectory(dir=TOOLS_DIR / "tmp") as tmpdir:
            tmp_root = Path(tmpdir)
            ore_xml, portfolio = self._callable_portfolio_with_underlying_bond(tmp_root)
            callable_out = price_bond_trade(
                ore_xml=ore_xml,
                portfolio_xml=portfolio,
                trade_id="CallableBondNoCall",
                asof_date="2016-02-05",
                model_day_counter="A365F",
                market_data_file=EXAMPLE_SHARED_MARKET_FULL,
                todaysmarket_xml=EXAMPLE_SHARED_TM,
                reference_data_path=tmp_root / "Input" / "reference_data_callablebond.xml",
                pricingengine_path=tmp_root / "Input" / "pricingengine_callablebond_lgm_grid.xml",
                flows_csv=None,
            )
            bond_out = price_bond_trade(
                ore_xml=ore_xml,
                portfolio_xml=portfolio,
                trade_id="UnderlyingBondTrade",
                asof_date="2016-02-05",
                model_day_counter="A365F",
                market_data_file=EXAMPLE_SHARED_MARKET_FULL,
                todaysmarket_xml=EXAMPLE_SHARED_TM,
                reference_data_path=tmp_root / "Input" / "reference_data_callablebond.xml",
                pricingengine_path=tmp_root / "Input" / "pricingengine_callablebond_lgm_grid.xml",
                flows_csv=None,
            )
            # Native ORE prices these two trades differently; Python currently
            # still reuses the standalone risky-bond layer for the no-call
            # callable, so this is an explicit gap marker rather than a hidden
            # documentation caveat.
            self.assertGreater(float(bond_out["py_npv"]), float(callable_out["py_npv"]))
            self.assertGreater(float(bond_out["py_npv"]) - float(callable_out["py_npv"]), 1.0e6)

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
