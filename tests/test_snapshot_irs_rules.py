from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from pythonore.domain.dataclasses import (
    BermudanSwaption,
    CollateralConfig,
    ConventionsConfig,
    CrossAssetModelConfig,
    EquityFactorConfig,
    EquityForward,
    EquityOption,
    EquitySwap,
    FXForward,
    FixingsData,
    GenericProduct,
    IndependentAmount,
    IRS,
    MarginingFrequency,
    MarketData,
    MarketQuote,
    CollateralBalance,
    NettingConfig,
    NettingSet,
    Portfolio,
    PricingEngineConfig,
    RuntimeConfig,
    SimulationConfig,
    SimulationMarketConfig,
    Trade,
    TodaysMarketConfig,
    XVAConfig,
    XVASnapshot,
)
from pythonore.io.loader import _parse_product_from_trade_xml
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


def test_xva_snapshot_stable_key_is_content_based():
    snapshot = XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),
                MarketQuote(date="2026-03-08", key="EUR-EURIBOR-6M", value=0.025),
            ),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swap",
                    product=IRS(
                        ccy="EUR",
                        notional=1_000_000,
                        fixed_rate=0.03,
                        maturity_years=2.0,
                    ),
                ),
            )
        ),
        config=XVAConfig(
            asof="2026-03-08",
            base_currency="EUR",
            xml_buffers={"simulation.xml": "<Simulation><Samples>10</Samples></Simulation>"},
        ),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    cloned = XVASnapshot.from_dict(snapshot.to_dict())
    quote_changed = XVASnapshot.from_dict(
        {
            **snapshot.to_dict(),
            "market": {
                **snapshot.to_dict()["market"],
                "raw_quotes": [
                    {"date": "2026-03-08", "key": "FX/EUR/USD", "value": 1.11},
                    {"date": "2026-03-08", "key": "EUR-EURIBOR-6M", "value": 0.025},
                ],
            },
        }
    )
    xml_changed = XVASnapshot.from_dict(
        {
            **snapshot.to_dict(),
            "config": {
                **snapshot.to_dict()["config"],
                "xml_buffers": {"simulation.xml": "<Simulation><Samples>11</Samples></Simulation>"},
            },
        }
    )

    assert snapshot.stable_key() == cloned.stable_key()
    assert snapshot.stable_key() != quote_changed.stable_key()
    assert snapshot.stable_key() != xml_changed.stable_key()


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


def test_fx_forward_and_bermudan_rules_round_trip_via_generated_portfolio_xml():
    runtime = RuntimeConfig(conventions=ConventionsConfig(day_counter="A360", calendar="TARGET"))
    snapshot = XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="FX1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.11, maturity_years=1.0, value_date="2027-03-10"),
                ),
                Trade(
                    trade_id="BERM1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swaption",
                    product=BermudanSwaption(
                        ccy="EUR",
                        notional=1_000_000,
                        fixed_rate=0.025,
                        maturity_years=5.0,
                        pay_fixed=True,
                        exercise_dates=("2027-03-08", "2028-03-08"),
                        start_date="2026-06-08",
                        end_date="2031-06-08",
                        fixed_leg_tenor="1Y",
                        float_leg_tenor="3M",
                        fixed_payment_convention="Unadjusted",
                        float_payment_convention="ModifiedFollowing",
                        fixed_schedule_convention="Unadjusted",
                        float_schedule_convention="ModifiedFollowing",
                        fixed_schedule_rule="Backward",
                        float_schedule_rule="Forward",
                        end_of_month=True,
                        float_index="EUR-EURIBOR-3M",
                        fixing_days=5,
                        float_spread=0.002,
                        payoff_at_expiry=True,
                        is_in_arrears=True,
                    ),
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", runtime=runtime),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    mapped = map_snapshot(snapshot)
    root = ET.fromstring(mapped.xml_buffers["portfolio.xml"])

    fx_trade = root.find("./Trade[@id='FX1']")
    fx_product = _parse_product_from_trade_xml(fx_trade, "FxForward")
    assert isinstance(fx_product, FXForward)
    assert fx_product.value_date == "2027-03-10"

    berm_trade = root.find("./Trade[@id='BERM1']")
    berm_product = _parse_product_from_trade_xml(berm_trade, "Swaption")
    assert isinstance(berm_product, BermudanSwaption)
    assert berm_product.start_date == "2026-06-08"
    assert berm_product.end_date == "2031-06-08"
    assert berm_product.float_leg_tenor == "3M"
    assert berm_product.fixed_schedule_rule == "Backward"
    assert berm_product.float_index == "EUR-EURIBOR-3M"
    assert berm_product.fixing_days == 5
    assert berm_product.float_spread == 0.002
    assert berm_product.payoff_at_expiry is True
    assert berm_product.is_in_arrears is True


def test_runtime_config_fields_are_exposed_in_generated_xml():
    runtime = RuntimeConfig(
        pricing_engine=PricingEngineConfig(
            model="BlackScholes",
            npv_engine="DiscountedCashflows",
            fx_model="DiscountedCashflows",
            fx_engine="DiscountingFxForwardEngine",
            swap_model="DiscountedCashflows",
            swap_engine="DiscountingSwapEngine",
            bermudan_model="LGM",
            bermudan_engine="Gaussian1dNonstandardSwaptionEngine",
            bermudan_reversion=0.05,
            bermudan_volatility=0.02,
            bermudan_shift_horizon=5.0,
            bermudan_sx=4.0,
            bermudan_nx=12,
            bermudan_sy=2.5,
            bermudan_ny=8,
        ),
        todays_market=TodaysMarketConfig(
            market_id="base",
            discount_curve="EUR-ESTR",
            fx_pairs=("EURUSD", "GBPUSD"),
            yield_curves_id="yield-set",
            discounting_curves_id="disc-set",
            index_forwarding_curves_id="index-set",
            fx_spots_id="spot-set",
            fx_volatilities_id="fxvol-set",
            swaption_volatilities_id="swaption-set",
            default_curves_id="credit-set",
        ),
        simulation=SimulationConfig(
            samples=128,
            seed=7,
            dates=("6M", "1Y", "2Y"),
            discretization="Euler",
            sequence="MersenneTwister",
            scenario="Stress",
            closeout_lag="3W",
            mpor_mode="StickyDate",
            day_counter="A360",
            calendar="TARGET,USD",
        ),
        simulation_market=SimulationMarketConfig(
            base_currency="EUR",
            currencies=("EUR", "USD"),
            indices=("EUR-ESTR", "USD-SOFR"),
            default_curve_names=("BANK", "CPTY_A"),
            fx_pairs=("EURUSD",),
            yield_curve_tenors=("1M", "3M", "1Y"),
            yield_curve_interpolation="Linear",
            yield_curve_extrapolation=False,
            default_curve_tenors=("6M", "1Y"),
            default_simulate_survival_probabilities=False,
            default_simulate_recovery_rates=False,
            default_curve_calendar="TARGET",
            default_curve_extrapolation="FlatFwd",
            swaption_simulate=True,
            swaption_reaction_to_time_decay="ConstantVariance",
            swaption_expiries=("1Y", "2Y"),
            swaption_terms=("5Y", "10Y"),
            fxvol_simulate=True,
            fxvol_reaction_to_time_decay="StickyStrike",
            fxvol_expiries=("6M", "1Y"),
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy="EUR",
            currencies=("EUR", "USD"),
            ir_model_ccys=("EUR", "USD"),
            fx_model_ccys=("USD",),
            bootstrap_tolerance=0.001,
            ir_calibration_type="BestFit",
            ir_volatility=0.015,
            ir_reversion=0.01,
            ir_shift_horizon=10.0,
            ir_scaling=0.8,
            ir_calibration_expiries=("2Y",),
            ir_calibration_terms=("10Y",),
            fx_calibration_type="Analytic",
            fx_sigma=0.2,
            fx_calibration_expiries=("2Y",),
        ),
    )
    snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(trades=(Trade(trade_id="T1", counterparty="CP_A", netting_set="NS1", trade_type="Swap", product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.03, maturity_years=2.0)),)),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", runtime=runtime),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    mapped = map_snapshot(snapshot)
    pricing = ET.fromstring(mapped.xml_buffers["pricingengine.xml"])
    todays_market = ET.fromstring(mapped.xml_buffers["todaysmarket.xml"])
    simulation = ET.fromstring(mapped.xml_buffers["simulation.xml"])

    assert pricing.findtext("./Product[@type='FxForward']/Engine") == "DiscountingFxForwardEngine"
    assert pricing.findtext("./Product[@type='BermudanSwaption']/ModelParameters/Parameter[@name='Reversion']") == "0.05"
    assert pricing.findtext("./Product[@type='BermudanSwaption']/EngineParameters/Parameter[@name='nx']") == "12"

    assert todays_market.findtext("./Configuration/YieldCurvesId") == "yield-set"
    assert todays_market.find("./FxSpots[@id='spot-set']") is not None
    assert todays_market.find("./DefaultCurves[@id='credit-set']") is not None

    assert simulation.findtext("./Parameters/Discretization") == "Euler"
    assert simulation.findtext("./Parameters/Sequence") == "MersenneTwister"
    assert simulation.findtext("./Parameters/Calendar") == "TARGET,USD"
    assert simulation.findtext("./CrossAssetModel/BootstrapTolerance") == "0.001"
    assert simulation.findtext("./CrossAssetModel/InterestRateModels/LGM/CalibrationType") == "BestFit"
    assert simulation.findtext("./Market/YieldCurves/Configuration/Interpolation") == "Linear"
    assert simulation.findtext("./Market/DefaultCurves/SimulateSurvivalProbabilities") == "false"
    assert simulation.findtext("./Market/SwaptionVolatilities/Simulate") == "true"
    assert simulation.findtext("./Market/FxVolatilities/ReactionToTimeDecay") == "StickyStrike"


def test_custom_product_fields_change_generated_portfolio_xml_and_fallback_legs():
    default_snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.03, maturity_years=2.0),
                ),
                Trade(
                    trade_id="FX1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.1, maturity_years=1.0),
                ),
                Trade(
                    trade_id="BERM1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swaption",
                    product=BermudanSwaption(
                        ccy="EUR",
                        notional=1_000_000,
                        fixed_rate=0.025,
                        maturity_years=5.0,
                        pay_fixed=True,
                        exercise_dates=("2027-03-08", "2028-03-08"),
                    ),
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", runtime=RuntimeConfig()),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )
    custom_snapshot = XVASnapshot(
        market=default_snapshot.market,
        fixings=default_snapshot.fixings,
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
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
                ),
                Trade(
                    trade_id="FX1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.1, maturity_years=1.0, value_date="2027-03-10"),
                ),
                Trade(
                    trade_id="BERM1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swaption",
                    product=BermudanSwaption(
                        ccy="EUR",
                        notional=1_000_000,
                        fixed_rate=0.025,
                        maturity_years=5.0,
                        pay_fixed=True,
                        exercise_dates=("2027-03-08", "2028-03-08"),
                        start_date="2026-06-08",
                        end_date="2031-06-08",
                        fixed_leg_tenor="1Y",
                        float_leg_tenor="3M",
                        fixed_schedule_rule="Backward",
                        float_schedule_rule="Forward",
                        float_index="EUR-EURIBOR-3M",
                        fixing_days=5,
                        float_spread=0.002,
                        payoff_at_expiry=True,
                        is_in_arrears=True,
                    ),
                ),
            )
        ),
        config=default_snapshot.config,
        netting=default_snapshot.netting,
        collateral=default_snapshot.collateral,
    )

    default_xml = map_snapshot(default_snapshot).xml_buffers["portfolio.xml"]
    custom_xml = map_snapshot(custom_snapshot).xml_buffers["portfolio.xml"]

    assert default_xml != custom_xml

    default_root = ET.fromstring(default_xml)
    custom_root = ET.fromstring(custom_xml)

    assert default_root.findtext(".//Trade[@id='FX1']//ValueDate") != custom_root.findtext(".//Trade[@id='FX1']//ValueDate")
    assert default_root.findtext(".//Trade[@id='BERM1']//PayOffAtExpiry") != custom_root.findtext(".//Trade[@id='BERM1']//PayOffAtExpiry")
    assert default_root.findtext(".//Trade[@id='BERM1']//FloatingLegData/FixingDays") != custom_root.findtext(".//Trade[@id='BERM1']//FloatingLegData/FixingDays")
    assert default_root.findtext(".//Trade[@id='IRS1']//FloatingLegData/Spreads/Spread") != custom_root.findtext(".//Trade[@id='IRS1']//FloatingLegData/Spreads/Spread")

    default_legs = _build_irs_legs_from_trade(default_snapshot.portfolio.trades[0], "2026-03-08")
    custom_legs = _build_irs_legs_from_trade(custom_snapshot.portfolio.trades[0], "2026-03-08")

    assert not np.array_equal(default_legs["fixed_pay_time"], custom_legs["fixed_pay_time"])
    assert not np.array_equal(default_legs["float_fixing_time"], custom_legs["float_fixing_time"])
    assert not np.array_equal(default_legs["float_spread"], custom_legs["float_spread"])


def test_custom_runtime_config_changes_generated_xml_payloads():
    base_runtime = RuntimeConfig()
    custom_runtime = RuntimeConfig(
        pricing_engine=PricingEngineConfig(
            fx_engine="DiscountingFxForwardEngine",
            bermudan_reversion=0.05,
            bermudan_nx=12,
        ),
        todays_market=TodaysMarketConfig(
            market_id="base",
            discount_curve="EUR-ESTR",
            fx_pairs=("EURUSD", "GBPUSD"),
            yield_curves_id="yield-set",
            discounting_curves_id="disc-set",
            index_forwarding_curves_id="index-set",
            fx_spots_id="spot-set",
            fx_volatilities_id="fxvol-set",
            swaption_volatilities_id="swaption-set",
            default_curves_id="credit-set",
        ),
        simulation=SimulationConfig(
            dates=("6M", "1Y", "2Y"),
            discretization="Euler",
            sequence="MersenneTwister",
            closeout_lag="3W",
            day_counter="A360",
            calendar="TARGET,USD",
        ),
        simulation_market=SimulationMarketConfig(
            yield_curve_tenors=("1M", "3M", "1Y"),
            yield_curve_interpolation="Linear",
            yield_curve_extrapolation=False,
            default_curve_tenors=("6M", "1Y"),
            default_simulate_survival_probabilities=False,
            swaption_simulate=True,
            fxvol_reaction_to_time_decay="StickyStrike",
            fxvol_expiries=("6M", "1Y"),
        ),
        cross_asset_model=CrossAssetModelConfig(
            bootstrap_tolerance=0.001,
            ir_calibration_type="BestFit",
            ir_volatility=0.015,
            fx_sigma=0.2,
        ),
    )
    base_snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(trades=(Trade(trade_id="T1", counterparty="CP_A", netting_set="NS1", trade_type="Swap", product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.03, maturity_years=2.0)),)),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", runtime=base_runtime),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )
    custom_snapshot = XVASnapshot(
        market=base_snapshot.market,
        fixings=base_snapshot.fixings,
        portfolio=base_snapshot.portfolio,
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", runtime=custom_runtime),
        netting=base_snapshot.netting,
        collateral=base_snapshot.collateral,
    )

    base_mapped = map_snapshot(base_snapshot)
    custom_mapped = map_snapshot(custom_snapshot)

    assert base_mapped.xml_buffers["pricingengine.xml"] != custom_mapped.xml_buffers["pricingengine.xml"]
    assert base_mapped.xml_buffers["todaysmarket.xml"] != custom_mapped.xml_buffers["todaysmarket.xml"]
    assert base_mapped.xml_buffers["simulation.xml"] != custom_mapped.xml_buffers["simulation.xml"]

    base_pricing = ET.fromstring(base_mapped.xml_buffers["pricingengine.xml"])
    custom_pricing = ET.fromstring(custom_mapped.xml_buffers["pricingengine.xml"])
    assert base_pricing.findtext("./Product[@type='FxForward']/Engine") != custom_pricing.findtext("./Product[@type='FxForward']/Engine")
    assert base_pricing.findtext("./Product[@type='BermudanSwaption']/EngineParameters/Parameter[@name='nx']") != custom_pricing.findtext("./Product[@type='BermudanSwaption']/EngineParameters/Parameter[@name='nx']")

    base_tm = ET.fromstring(base_mapped.xml_buffers["todaysmarket.xml"])
    custom_tm = ET.fromstring(custom_mapped.xml_buffers["todaysmarket.xml"])
    assert base_tm.findtext("./Configuration/YieldCurvesId") != custom_tm.findtext("./Configuration/YieldCurvesId")
    assert base_tm.find("./FxSpots[@id='default']") is not None
    assert custom_tm.find("./FxSpots[@id='spot-set']") is not None

    base_sim = ET.fromstring(base_mapped.xml_buffers["simulation.xml"])
    custom_sim = ET.fromstring(custom_mapped.xml_buffers["simulation.xml"])
    assert base_sim.findtext("./Parameters/Discretization") != custom_sim.findtext("./Parameters/Discretization")
    assert base_sim.findtext("./Parameters/Calendar") != custom_sim.findtext("./Parameters/Calendar")
    assert base_sim.findtext("./CrossAssetModel/BootstrapTolerance") != custom_sim.findtext("./CrossAssetModel/BootstrapTolerance")
    assert base_sim.findtext("./Market/FxVolatilities/ReactionToTimeDecay") != custom_sim.findtext("./Market/FxVolatilities/ReactionToTimeDecay")


def test_trade_additional_fields_are_preserved_in_generated_portfolio_xml():
    snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.03, maturity_years=2.0),
                    additional_fields={"Desk": "Rates", "Strategy-Id": "alpha<1>&beta"},
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    root = ET.fromstring(map_snapshot(snapshot).xml_buffers["portfolio.xml"])
    assert root.findtext("./Trade/Envelope/AdditionalFields/Desk") == "Rates"
    assert root.findtext("./Trade/Envelope/AdditionalFields/Strategy-Id") == "alpha<1>&beta"


def test_generic_product_xml_is_preserved_across_loader_and_mapper():
    portfolio_xml = """
<Portfolio>
  <Trade id="GEN1">
    <TradeType>CommodityForward</TradeType>
    <Envelope>
      <CounterParty>CP_A</CounterParty>
      <NettingSetId>NS1</NettingSetId>
      <AdditionalFields>
        <Desk>Commodities</Desk>
      </AdditionalFields>
    </Envelope>
    <CommodityForwardData>
      <Name>POWER_Q1</Name>
      <Currency>EUR</Currency>
      <Quantity>100</Quantity>
    </CommodityForwardData>
  </Trade>
</Portfolio>
""".strip()
    trade = ET.fromstring(portfolio_xml).find("./Trade")
    product = _parse_product_from_trade_xml(trade, "CommodityForward")

    assert isinstance(product, GenericProduct)
    assert "CommodityForwardData" in product.payload["xml"]

    snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="GEN1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="CommodityForward",
                    product=product,
                    additional_fields={"Desk": "Commodities"},
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    mapped = map_snapshot(snapshot)
    remapped_root = ET.fromstring(mapped.xml_buffers["portfolio.xml"])
    assert remapped_root.find("./Trade/CommodityForwardData") is not None
    assert remapped_root.findtext("./Trade/CommodityForwardData/Name") == "POWER_Q1"
    assert remapped_root.findtext("./Trade/Envelope/AdditionalFields/Desk") == "Commodities"


def test_rich_netting_and_collateral_fields_round_trip_and_change_generated_xml():
    default_snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(trades=(Trade(trade_id="T1", counterparty="CP_A", netting_set="NS1", trade_type="Swap", product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.03, maturity_years=2.0)),)),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
        netting=NettingConfig(netting_sets={"NS1": NettingSet(netting_set_id="NS1", active_csa=True, csa_currency="EUR")}),
        collateral=CollateralConfig(balances=(CollateralBalance(netting_set_id="NS1", currency="EUR", initial_margin=0.0, variation_margin=0.0),)),
    )
    custom_snapshot = XVASnapshot(
        market=default_snapshot.market,
        fixings=default_snapshot.fixings,
        portfolio=default_snapshot.portfolio,
        config=default_snapshot.config,
        netting=NettingConfig(
            netting_sets={
                "NS1": NettingSet(
                    netting_set_id="NS1",
                    counterparty="CP_A",
                    active_csa=True,
                    bilateral="Bilateral",
                    csa_currency="EUR",
                    index="EUR-EONIA",
                    margin_period_of_risk="2W",
                    threshold_pay=100000.0,
                    threshold_receive=200000.0,
                    mta_pay=5000.0,
                    mta_receive=6000.0,
                    independent_amount=IndependentAmount(held=7000.0, posted=3000.0, amount_type="FIXED"),
                    margining_frequency=MarginingFrequency(call_frequency="1D", post_frequency="2D"),
                    collateral_compounding_spread_receive=0.01,
                    collateral_compounding_spread_pay=0.02,
                    eligible_collateral_currencies=("EUR", "USD"),
                    raw_csa_fields={"OpeningBalance": "123.45"},
                )
            }
        ),
        collateral=CollateralConfig(
            balances=(
                CollateralBalance(
                    netting_set_id="NS1",
                    currency="EUR",
                    initial_margin=1500000.0,
                    variation_margin=75000.0,
                    initial_margin_type="SIMM",
                    variation_margin_type="VM",
                    raw_fields={"Custodian": "CUST_A"},
                ),
            )
        ),
    )

    restored = XVASnapshot.from_dict(custom_snapshot.to_dict())
    restored_ns = restored.netting.netting_sets["NS1"]
    restored_bal = restored.collateral.balances[0]
    assert restored_ns.index == "EUR-EONIA"
    assert restored_ns.independent_amount.amount_type == "FIXED"
    assert restored_ns.margining_frequency.post_frequency == "2D"
    assert restored_ns.eligible_collateral_currencies == ("EUR", "USD")
    assert restored_ns.raw_csa_fields["OpeningBalance"] == "123.45"
    assert restored_bal.initial_margin_type == "SIMM"
    assert restored_bal.raw_fields["Custodian"] == "CUST_A"

    default_mapped = map_snapshot(default_snapshot)
    custom_mapped = map_snapshot(custom_snapshot)
    assert default_mapped.xml_buffers["netting.xml"] != custom_mapped.xml_buffers["netting.xml"]
    assert default_mapped.xml_buffers["collateralbalances.xml"] != custom_mapped.xml_buffers["collateralbalances.xml"]

    netting_root = ET.fromstring(custom_mapped.xml_buffers["netting.xml"])
    assert netting_root.findtext("./NettingSet/CSADetails/Index") == "EUR-EONIA"
    assert netting_root.findtext("./NettingSet/CSADetails/IndependentAmount/IndependentAmountPosted") == "3000.0"
    assert netting_root.findtext("./NettingSet/CSADetails/MarginingFrequency/PostFrequency") == "2D"
    assert [n.text for n in netting_root.findall("./NettingSet/CSADetails/EligibleCollaterals/Currencies/Currency")] == ["EUR", "USD"]
    assert netting_root.findtext("./NettingSet/CSADetails/OpeningBalance") == "123.45"

    collateral_root = ET.fromstring(custom_mapped.xml_buffers["collateralbalances.xml"])
    assert collateral_root.findtext("./CollateralBalance/InitialMarginType") == "SIMM"
    assert collateral_root.findtext("./CollateralBalance/VariationMarginType") == "VM"
    assert collateral_root.findtext("./CollateralBalance/Custodian") == "CUST_A"


def test_equity_products_round_trip_through_snapshot_dict():
    snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="EQO1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="EquityOption",
                    product=EquityOption(
                        name="RIC:.STOXX50E",
                        currency="EUR",
                        strike=4000.0,
                        quantity=1.0,
                        option_type="Call",
                        exercise_date="2021-12-17",
                    ),
                ),
                Trade(
                    trade_id="EQF1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="EquityForward",
                    product=EquityForward(
                        name="SP5",
                        currency="USD",
                        strike=2147.56,
                        quantity=775.0,
                        maturity_date="2018-02-05",
                        long_short="Short",
                        strike_currency="USD",
                        maturity_years=2.0,
                    ),
                ),
                Trade(
                    trade_id="EQS1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="EquitySwap",
                    product=EquitySwap(
                        name="SP5",
                        currency="USD",
                        notional=100_000_000.0,
                        initial_price=2196.5,
                        quantity=100_000_000.0 / 2196.5,
                        start_date="2016-02-05",
                        end_date="2026-02-05",
                        maturity_years=10.0,
                        float_index="USD-LIBOR-3M",
                    ),
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    restored = XVASnapshot.from_dict(snapshot.to_dict())

    assert isinstance(restored.portfolio.trades[0].product, EquityOption)
    assert restored.portfolio.trades[0].product.exercise_date == "2021-12-17"
    assert isinstance(restored.portfolio.trades[1].product, EquityForward)
    assert restored.portfolio.trades[1].product.long_short == "Short"
    assert isinstance(restored.portfolio.trades[2].product, EquitySwap)
    assert restored.portfolio.trades[2].product.float_index == "USD-LIBOR-3M"


def test_loader_and_mapper_round_trip_example_equity_products():
    portfolio_xml = """
<Portfolio>
  <Trade id="EQO1">
    <TradeType>EquityOption</TradeType>
    <Envelope><CounterParty>CP_A</CounterParty><NettingSetId>NS1</NettingSetId></Envelope>
    <EquityOptionData>
      <OptionData>
        <LongShort>Long</LongShort>
        <OptionType>Call</OptionType>
        <Style>European</Style>
        <Settlement>Cash</Settlement>
        <PayOffAtExpiry>false</PayOffAtExpiry>
        <ExerciseDates><ExerciseDate>2021-12-17</ExerciseDate></ExerciseDates>
      </OptionData>
      <Name>RIC:.STOXX50E</Name>
      <Currency>EUR</Currency>
      <Strike>4000</Strike>
      <Quantity>1</Quantity>
    </EquityOptionData>
  </Trade>
  <Trade id="EQF1">
    <TradeType>EquityForward</TradeType>
    <Envelope><CounterParty>CP_A</CounterParty><NettingSetId>NS1</NettingSetId></Envelope>
    <EquityForwardData>
      <LongShort>Short</LongShort>
      <Maturity>2018-02-05</Maturity>
      <Name>SP5</Name>
      <Currency>USD</Currency>
      <Strike>2147.56</Strike>
      <StrikeCurrency>USD</StrikeCurrency>
      <Quantity>775</Quantity>
    </EquityForwardData>
  </Trade>
  <Trade id="EQS1">
    <TradeType>EquitySwap</TradeType>
    <Envelope><CounterParty>CP_A</CounterParty><NettingSetId>NS1</NettingSetId></Envelope>
    <SwapData>
      <LegData>
        <LegType>Equity</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>100000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <EquityLegData><ReturnType>Price</ReturnType><Name>SP5</Name><InitialPrice>2196.5</InitialPrice></EquityLegData>
        <ScheduleData><Rules><StartDate>20160205</StartDate><EndDate>20260205</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>F</Convention><TermConvention>F</TermConvention><Rule>Forward</Rule><EndOfMonth>false</EndOfMonth></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>100000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData><Index>USD-LIBOR-3M</Index><Spreads><Spread>0.0</Spread></Spreads><FixingDays>2</FixingDays></FloatingLegData>
        <ScheduleData><Rules><StartDate>20160205</StartDate><EndDate>20260205</EndDate><Tenor>3M</Tenor><Calendar>USD</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule><EndOfMonth>false</EndOfMonth></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>
</Portfolio>
""".strip()
    root = ET.fromstring(portfolio_xml)

    eq_option = _parse_product_from_trade_xml(root.find("./Trade[@id='EQO1']"), "EquityOption")
    eq_forward = _parse_product_from_trade_xml(root.find("./Trade[@id='EQF1']"), "EquityForward")
    eq_swap = _parse_product_from_trade_xml(root.find("./Trade[@id='EQS1']"), "EquitySwap")

    assert isinstance(eq_option, EquityOption)
    assert isinstance(eq_forward, EquityForward)
    assert isinstance(eq_swap, EquitySwap)

    snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade("EQO1", "CP_A", "NS1", "EquityOption", eq_option),
                Trade("EQF1", "CP_A", "NS1", "EquityForward", eq_forward),
                Trade("EQS1", "CP_A", "NS1", "EquitySwap", eq_swap),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    mapped = map_snapshot(snapshot)
    remapped_root = ET.fromstring(mapped.xml_buffers["portfolio.xml"])
    assert remapped_root.findtext("./Trade[@id='EQO1']/EquityOptionData/Name") == "RIC:.STOXX50E"
    assert remapped_root.findtext("./Trade[@id='EQF1']/EquityForwardData/LongShort") == "Short"
    assert remapped_root.findtext("./Trade[@id='EQS1']//EquityLegData/Name") == "SP5"
    assert remapped_root.findtext("./Trade[@id='EQS1']//FloatingLegData/Index") == "USD-LIBOR-3M"


def test_runtime_config_generates_equity_pricing_and_simulation_model_xml():
    runtime = RuntimeConfig(
        pricing_engine=PricingEngineConfig(
            equity_option_model="BlackScholesMerton",
            equity_option_engine="AnalyticEuropeanEngine",
            equity_forward_model="DiscountedCashflows",
            equity_forward_engine="DiscountingEquityForwardEngine",
            equity_swap_model="DiscountedCashflows",
            equity_swap_engine="DiscountingSwapEngine",
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy="EUR",
            currencies=("EUR", "USD"),
            ir_model_ccys=("EUR", "USD"),
            fx_model_ccys=("USD",),
            equity_factors=(
                EquityFactorConfig(
                    name="default",
                    currency="EUR",
                    sigma=0.1,
                    time_grid=("1.0", "2.0", "3.0"),
                    expiries=("1Y", "2Y"),
                    strikes=(),
                ),
                EquityFactorConfig(
                    name="SP5",
                    currency="USD",
                    sigma=0.2,
                    time_grid=("1.0", "2.0"),
                    expiries=("1Y",),
                    strikes=(),
                ),
            ),
        ),
    )
    snapshot = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(Trade(trade_id="T1", counterparty="CP_A", netting_set="NS1", trade_type="EquityOption", product=EquityOption(name="RIC:.STOXX50E", currency="EUR", strike=4000.0, quantity=1.0, option_type="Call", exercise_date="2021-12-17")),)
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", runtime=runtime),
        netting=NettingConfig(),
        collateral=CollateralConfig(),
    )

    mapped = map_snapshot(snapshot)
    pricing = ET.fromstring(mapped.xml_buffers["pricingengine.xml"])
    simulation = ET.fromstring(mapped.xml_buffers["simulation.xml"])

    assert pricing.findtext("./Product[@type='EquityOption']/Engine") == "AnalyticEuropeanEngine"
    assert pricing.findtext("./Product[@type='EquityForward']/Engine") == "DiscountingEquityForwardEngine"
    assert pricing.findtext("./Product[@type='EquitySwap']/Engine") == "DiscountingSwapEngine"
    assert simulation.findtext("./CrossAssetModel/EquityModels/CrossAssetLGM[@name='default']/Currency") == "EUR"
    assert simulation.findtext("./CrossAssetModel/EquityModels/CrossAssetLGM[@name='SP5']/Currency") == "USD"
