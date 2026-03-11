import unittest
from pathlib import Path
import sys

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOLS_DIR.parents[1]
PY_NATIVE_DIR = REPO_ROOT / "PythonIntegration"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(PY_NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(PY_NATIVE_DIR))

from native_xva_interface import (
    CollateralBalance,
    CollateralConfig,
    FixingPoint,
    FixingsData,
    IRS,
    MarketData,
    MarketQuote,
    NettingConfig,
    NettingSet,
    Portfolio,
    Trade,
    XVAConfig,
    XVASnapshot,
)
from py_ore_tools.ore_snapshot import (
    validate_xva_snapshot_dataclasses,
    xva_snapshot_validation_dataframe,
)


class TestOreSnapshotDataclassValidation(unittest.TestCase):
    def test_validate_xva_snapshot_dataclasses_happy_path(self):
        snap = XVASnapshot(
            market=MarketData(
                asof="2026-03-08",
                raw_quotes=(
                    MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),
                ),
            ),
            fixings=FixingsData(
                points=(FixingPoint(date="2026-03-07", index="USD-LIBOR-3M", value=0.05),)
            ),
            portfolio=Portfolio(
                trades=(
                    Trade(
                        trade_id="T1",
                        counterparty="CP_A",
                        netting_set="NS1",
                        trade_type="Swap",
                        product=IRS(ccy="USD", notional=1_000_000, fixed_rate=0.03, maturity_years=5.0),
                    ),
                )
            ),
            config=XVAConfig(asof="2026-03-08", base_currency="USD", analytics=("CVA", "DVA")),
            netting=NettingConfig(netting_sets={"NS1": NettingSet(netting_set_id="NS1", counterparty="CP_A")}),
            collateral=CollateralConfig(balances=(CollateralBalance(netting_set_id="NS1", currency="USD"),)),
        )

        report = validate_xva_snapshot_dataclasses(snap)

        self.assertTrue(report["checks"]["asof_consistent"])
        self.assertTrue(report["checks"]["quote_dates_match_asof"])
        self.assertTrue(report["checks"]["fixing_dates_not_after_asof"])
        self.assertTrue(report["checks"]["analytics_supported"])
        self.assertTrue(report["checks"]["netting_sets_defined"])
        self.assertTrue(report["checks"]["collateral_matches_netting"])
        self.assertTrue(report["snapshot_valid"])

    def test_validate_xva_snapshot_dataclasses_flags_gaps(self):
        snap = XVASnapshot(
            market=MarketData(
                asof="2026-03-08",
                raw_quotes=(
                    MarketQuote(date="2026-03-07", key="FX/EUR/USD", value=1.1),
                    MarketQuote(date="2026-03-07", key="FX/EUR/USD", value=1.2),
                ),
            ),
            fixings=FixingsData(
                points=(FixingPoint(date="2026-03-09", index="USD-LIBOR-3M", value=0.05),)
            ),
            portfolio=Portfolio(
                trades=(
                    Trade(
                        trade_id="T1",
                        counterparty="CP_A",
                        netting_set="NS_MISSING",
                        trade_type="Swap",
                        product=IRS(ccy="USD", notional=1_000_000, fixed_rate=0.03, maturity_years=5.0),
                    ),
                )
            ),
            config=XVAConfig(asof="2026-03-08", base_currency="USD", analytics=("CVA", "BAD")),
            netting=NettingConfig(netting_sets={"NS1": NettingSet(netting_set_id="NS1", counterparty="CP_A")}),
            collateral=CollateralConfig(balances=(CollateralBalance(netting_set_id="NS2", currency="USD"),)),
        )

        report = validate_xva_snapshot_dataclasses(snap)

        self.assertFalse(report["checks"]["quote_dates_match_asof"])
        self.assertFalse(report["checks"]["fixing_dates_not_after_asof"])
        self.assertFalse(report["checks"]["analytics_supported"])
        self.assertFalse(report["checks"]["netting_sets_defined"])
        self.assertFalse(report["checks"]["collateral_matches_netting"])
        self.assertEqual(report["market"]["quote_duplicate_count"], 1)
        self.assertIn("BAD", report["analytics"]["invalid"])
        self.assertTrue(any(item["code"] == "unsupported_analytics" for item in report["action_items"]))
        self.assertIn("NS_MISSING", report["netting"]["missing_trade_netting_sets"])
        self.assertIn("NS2", report["collateral"]["unknown_balance_netting_sets"])
        self.assertFalse(report["snapshot_valid"])

    def test_xva_snapshot_validation_dataframe_shape(self):
        snap = XVASnapshot(
            market=MarketData(asof="2026-03-08"),
            fixings=FixingsData(),
            portfolio=Portfolio(
                trades=(
                    Trade(
                        trade_id="T1",
                        counterparty="CP_A",
                        netting_set="NS1",
                        trade_type="Swap",
                        product=IRS(ccy="USD", notional=1_000_000, fixed_rate=0.03, maturity_years=5.0),
                    ),
                )
            ),
            config=XVAConfig(asof="2026-03-08", base_currency="USD"),
        )

        report = validate_xva_snapshot_dataclasses(snap)
        df = xva_snapshot_validation_dataframe(report)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["section", "field", "value"])
        self.assertTrue(((df["section"] == "checks") & (df["field"] == "asof_consistent")).any())


if __name__ == "__main__":
    unittest.main()
