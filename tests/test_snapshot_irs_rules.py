from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from pythonore.domain.dataclasses import (
    CollateralConfig,
    FixingsData,
    IRS,
    MarketData,
    MarketQuote,
    NettingConfig,
    Portfolio,
    Trade,
    XVAConfig,
    XVASnapshot,
)
from pythonore.mapping.mapper import map_snapshot
from pythonore.runtime.runtime import _build_irs_legs_from_trade


def _snapshot_with_trade(product: IRS) -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swap",
                    product=product,
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )


def test_extended_irs_fields_round_trip_through_snapshot_dict():
    product = IRS(
        ccy="EUR",
        notional=1_000_000,
        fixed_rate=0.03,
        maturity_years=2.0,
        pay_fixed=False,
        start_date="2026-06-08",
        end_date="2028-06-08",
        fixed_leg_tenor="1Y",
        float_leg_tenor="6M",
        fixed_day_counter="30E/360",
        float_day_counter="A360",
        calendar="TARGET",
        fixed_payment_convention="Following",
        float_payment_convention="ModifiedFollowing",
        fixed_schedule_convention="Unadjusted",
        float_schedule_convention="ModifiedFollowing",
        fixed_term_convention="Unadjusted",
        float_term_convention="ModifiedFollowing",
        fixed_schedule_rule="Backward",
        float_schedule_rule="Backward",
        end_of_month=True,
        float_index="EUR-EURIBOR-6M",
        fixing_days=4,
        float_spread=0.0015,
    )
    snapshot = _snapshot_with_trade(product)

    restored = XVASnapshot.from_dict(snapshot.to_dict())
    restored_product = restored.portfolio.trades[0].product

    assert isinstance(restored_product, IRS)
    assert restored_product.start_date == "2026-06-08"
    assert restored_product.end_date == "2028-06-08"
    assert restored_product.fixed_leg_tenor == "1Y"
    assert restored_product.float_leg_tenor == "6M"
    assert restored_product.fixed_schedule_rule == "Backward"
    assert restored_product.float_schedule_rule == "Backward"
    assert restored_product.float_index == "EUR-EURIBOR-6M"
    assert restored_product.fixing_days == 4
    assert restored_product.float_spread == 0.0015


def test_map_snapshot_exposes_extended_irs_schedule_rules_in_portfolio_xml():
    snapshot = _snapshot_with_trade(
        IRS(
            ccy="EUR",
            notional=1_000_000,
            fixed_rate=0.03,
            maturity_years=2.0,
            start_date="2026-06-08",
            end_date="2028-06-08",
            fixed_leg_tenor="1Y",
            float_leg_tenor="6M",
            fixed_payment_convention="Following",
            float_payment_convention="ModifiedFollowing",
            fixed_schedule_convention="Unadjusted",
            float_schedule_convention="ModifiedFollowing",
            fixed_term_convention="Unadjusted",
            float_term_convention="ModifiedFollowing",
            fixed_schedule_rule="Backward",
            float_schedule_rule="Forward",
            end_of_month=True,
            float_index="EUR-EURIBOR-6M",
            fixing_days=4,
            float_spread=0.0015,
        )
    )

    mapped = map_snapshot(snapshot)
    root = ET.fromstring(mapped.xml_buffers["portfolio.xml"])
    rules = root.findall(".//SwapData/LegData/ScheduleData/Rules")

    assert len(rules) == 2
    assert rules[0].findtext("./StartDate") == "20260608"
    assert rules[0].findtext("./EndDate") == "20280608"
    assert rules[0].findtext("./Tenor") == "1Y"
    assert rules[0].findtext("./Convention") == "Unadjusted"
    assert rules[0].findtext("./Rule") == "Backward"
    assert rules[0].findtext("./EndOfMonth") == "true"
    assert rules[1].findtext("./Tenor") == "6M"
    assert rules[1].findtext("./Convention") == "ModifiedFollowing"
    assert rules[1].findtext("./Rule") == "Forward"
    assert root.findtext(".//FloatingLegData/Index") == "EUR-EURIBOR-6M"
    assert root.findtext(".//FloatingLegData/FixingDays") == "4"
    assert root.findtext(".//FloatingLegData/Spreads/Spread") == "0.0015"


def test_fallback_irs_leg_builder_uses_custom_tenors_rules_and_fixing_days():
    trade = Trade(
        trade_id="T1",
        counterparty="CP_A",
        netting_set="NS1",
        trade_type="Swap",
        product=IRS(
            ccy="EUR",
            notional=1_000_000,
            fixed_rate=0.03,
            maturity_years=2.0,
            start_date="2026-06-08",
            end_date="2028-06-08",
            fixed_leg_tenor="1Y",
            float_leg_tenor="6M",
            fixed_schedule_rule="Backward",
            float_schedule_rule="Backward",
            float_index="EUR-EURIBOR-6M",
            fixing_days=4,
            float_spread=0.0015,
        ),
    )

    legs = _build_irs_legs_from_trade(trade, "2026-03-08")

    assert len(legs["fixed_accrual"]) == 3
    assert np.isclose(float(np.sum(legs["fixed_accrual"])), float(np.sum(legs["float_accrual"])), atol=1.0e-8)
    assert np.allclose(legs["fixed_accrual"][1:], np.array([1.0, 1.0]), atol=1.0e-8)
    assert np.allclose(legs["float_accrual"][1:], np.full(4, 0.5), atol=1.0e-8)
    assert 0.0 < float(legs["fixed_accrual"][0]) < 0.01
    assert 0.0 < float(legs["float_accrual"][0]) < 0.01
    assert np.allclose(legs["float_spread"], np.full(5, 0.0015), atol=1.0e-12)
    assert legs["float_index"] == "EUR-EURIBOR-6M"
    assert np.allclose(legs["float_fixing_time"], legs["float_start_time"] - (4.0 / 365.25), atol=1.0e-12)
