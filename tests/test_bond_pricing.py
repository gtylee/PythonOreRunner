import sys
import unittest
from datetime import date
import numpy as np
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools.bond_pricing import (
    BondCashflow,
    BondEngineSpec,
    BondTradeSpec,
    _bond_npv,
    _curve_from_flow_discounts,
    _load_bond_cashflows_from_flows,
    price_bond_trade,
)
from py_ore_tools.ore_snapshot import _load_ore_npv_details


EXAMPLE_18_ORE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "Input" / "ore.xml"
EXAMPLE_18_PORTFOLIO = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "Input" / "portfolio.xml"
EXAMPLE_18_REFERENCE = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "Input" / "referencedata.xml"
EXAMPLE_18_FLOWS = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "ExpectedOutput" / "flows.csv"
EXAMPLE_18_NPV = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "ExpectedOutput" / "npv.csv"
EXAMPLE_SHARED_MARKET = TOOLS_DIR / "Examples" / "Input" / "market_20160205_flat.txt"
EXAMPLE_SHARED_TM = TOOLS_DIR / "Examples" / "Input" / "todaysmarket.xml"
EXAMPLE_SHARED_PE = TOOLS_DIR / "Examples" / "Input" / "pricingengine.xml"


def _flat_curve(rate: float):
    return lambda t: 1.0 if float(t) <= 0.0 else float(pow(2.718281828459045, -rate * float(t)))


class TestBondPricing(unittest.TestCase):
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
        self.assertAlmostEqual(curve(0.9945205479452055), 0.9803373492, places=10)
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
        out = price_bond_trade(
            ore_xml=EXAMPLE_18_ORE_XML,
            portfolio_xml=EXAMPLE_18_PORTFOLIO,
            trade_id="Bond_Fixed",
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=EXAMPLE_18_REFERENCE,
            pricingengine_path=EXAMPLE_SHARED_PE,
            flows_csv=EXAMPLE_18_FLOWS,
        )
        ore_npv = _load_ore_npv_details(EXAMPLE_18_NPV, trade_id="Bond_Fixed")["npv"]
        self.assertLess(abs(out["py_npv"] - ore_npv), 260000.0)

    def test_example18_forward_bond_parity_stays_within_regression_band(self):
        out = price_bond_trade(
            ore_xml=EXAMPLE_18_ORE_XML,
            portfolio_xml=EXAMPLE_18_PORTFOLIO,
            trade_id="FwdBond_Fixed",
            asof_date="2016-02-05",
            model_day_counter="A365F",
            market_data_file=EXAMPLE_SHARED_MARKET,
            todaysmarket_xml=EXAMPLE_SHARED_TM,
            reference_data_path=EXAMPLE_18_REFERENCE,
            pricingengine_path=EXAMPLE_SHARED_PE,
            flows_csv=EXAMPLE_18_FLOWS,
        )
        ore_npv = _load_ore_npv_details(EXAMPLE_18_NPV, trade_id="FwdBond_Fixed")["npv"]
        self.assertLess(abs(out["py_npv"] - ore_npv), 70000.0)


if __name__ == "__main__":
    unittest.main()
