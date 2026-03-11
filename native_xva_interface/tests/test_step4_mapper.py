from native_xva_interface import (
    BermudanSwaption,
    ConventionsConfig,
    CounterpartyConfig,
    CrossAssetModelConfig,
    CreditEntityConfig,
    CreditSimulationConfig,
    CurveConfig,
    FXForward,
    IRS,
    MarketData,
    MarketQuote,
    Portfolio,
    PricingEngineConfig,
    RuntimeConfig,
    SimulationConfig,
    SimulationMarketConfig,
    Trade,
    TodaysMarketConfig,
    XVAAnalyticConfig,
    XVAConfig,
    XVASnapshot,
    FixingsData,
    build_input_parameters,
    map_snapshot,
)
from native_xva_interface.mapper import _default_index_for_ccy


class FakeInputParameters:
    def __init__(self):
        self.calls = []

    def _store(self, name, *args):
        self.calls.append((name, args))

    def setAsOfDate(self, s):
        self._store("setAsOfDate", s)

    def setBaseCurrency(self, s):
        self._store("setBaseCurrency", s)

    def setAnalytics(self, s):
        self._store("setAnalytics", s)

    def setPortfolio(self, xml):
        self._store("setPortfolio", xml)

    def setNettingSetManager(self, xml):
        self._store("setNettingSetManager", xml)

    def setCollateralBalances(self, xml):
        self._store("setCollateralBalances", xml)

    def setPricingEngine(self, xml):
        self._store("setPricingEngine", xml)

    def setTodaysMarketParams(self, xml):
        self._store("setTodaysMarketParams", xml)

    def setCurveConfigs(self, xml, id=""):
        self._store("setCurveConfigs", xml, id)

    def setExposureAllocationMethod(self, v):
        self._store("setExposureAllocationMethod", v)

    def setExposureObservationModel(self, v):
        self._store("setExposureObservationModel", v)

    def setPfeQuantile(self, v):
        self._store("setPfeQuantile", v)

    def setCollateralCalculationType(self, v):
        self._store("setCollateralCalculationType", v)

    def setMarginalAllocationLimit(self, v):
        self._store("setMarginalAllocationLimit", v)

    def setScenarioGenType(self, v):
        self._store("setScenarioGenType", v)

    def setNettingSetId(self, v):
        self._store("setNettingSetId", v)

    def setCreditSimulationParametersFromBuffer(self, xml):
        self._store("setCreditSimulationParametersFromBuffer", xml)

    def setDimQuantile(self, v):
        self._store("setDimQuantile", v)

    def setDimHorizonCalendarDays(self, v):
        self._store("setDimHorizonCalendarDays", v)

    def setDimRegressionOrder(self, v):
        self._store("setDimRegressionOrder", v)

    def setDimRegressors(self, v):
        self._store("setDimRegressors", v)

    def setDimOutputNettingSet(self, v):
        self._store("setDimOutputNettingSet", v)

    def setDimOutputGridPoints(self, v):
        self._store("setDimOutputGridPoints", v)


def _snapshot() -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),)),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.1, maturity_years=1.0),
                ),
            )
        ),
        config=XVAConfig(
            asof="2026-03-08",
            base_currency="EUR",
            num_paths=123,
            runtime=RuntimeConfig(
                pricing_engine=PricingEngineConfig(model="LGM", npv_engine="Analytic"),
                todays_market=TodaysMarketConfig(market_id="default", discount_curve="EUR-EONIA", fx_pairs=("EURUSD",)),
                curve_configs=(CurveConfig(curve_id="EUR-EONIA", currency="EUR", tenors=("1Y", "5Y")),),
                simulation=SimulationConfig(samples=999, seed=7, dates=("1Y", "3Y")),
                xva_analytic=XVAAnalyticConfig(
                    scenario_gen_type="Simple",
                    dim_quantile=0.99,
                    dim_horizon_calendar_days=14,
                    dim_regression_order=2,
                    dim_regressors="",
                    dim_output_netting_set="NS1",
                    dim_output_grid_points="0",
                ),
                conventions=ConventionsConfig(day_counter="A360", calendar="TARGET"),
                counterparties=CounterpartyConfig(ids=("CP_A",)),
            ),
            xml_buffers={
                "simulation.xml": "<Simulation><Parameters><Samples>5000</Samples></Parameters></Simulation>",
            },
        ),
    )


def test_mapper_outputs_expected_sections():
    mapped = map_snapshot(_snapshot())
    assert mapped.base_currency == "EUR"
    assert len(mapped.market_data_lines) == 1
    assert "portfolio.xml" in mapped.xml_buffers
    assert "<Samples>123</Samples>" in mapped.xml_buffers["simulation.xml"]


def test_mapper_builds_runtime_xml_buffers_without_templates():
    snap = _snapshot()
    snap = XVASnapshot(
        market=snap.market,
        fixings=snap.fixings,
        portfolio=snap.portfolio,
        config=XVAConfig(
            asof=snap.config.asof,
            base_currency=snap.config.base_currency,
            analytics=snap.config.analytics,
            num_paths=10,
            runtime=snap.config.runtime,
        ),
        netting=snap.netting,
        collateral=snap.collateral,
        source_meta=snap.source_meta,
    )
    mapped = map_snapshot(snap)
    assert "pricingengine.xml" in mapped.xml_buffers
    assert "<PricingEngines>" in mapped.xml_buffers["pricingengine.xml"]
    assert "<TodaysMarket>" in mapped.xml_buffers["todaysmarket.xml"]
    assert "<CurveConfiguration>" in mapped.xml_buffers["curveconfig.xml"]
    assert "<Conventions>" in mapped.xml_buffers["conventions.xml"]
    assert "<CounterpartyInformation>" in mapped.xml_buffers["counterparty.xml"]
    assert '<Product type="BermudanSwaption">' in mapped.xml_buffers["pricingengine.xml"]
    assert "<Samples>10</Samples>" in mapped.xml_buffers["simulation.xml"]
    assert "<CrossAssetModel>" in mapped.xml_buffers["simulation.xml"]
    assert "<Market>" in mapped.xml_buffers["simulation.xml"]


def test_mapper_replaces_generated_placeholder_xml_with_runtime_bundle():
    snap = _snapshot()
    snap = XVASnapshot(
        market=snap.market,
        fixings=snap.fixings,
        portfolio=snap.portfolio,
        config=XVAConfig(
            asof=snap.config.asof,
            base_currency=snap.config.base_currency,
            analytics=snap.config.analytics,
            num_paths=10,
            xml_buffers={
                "pricingengine.xml": "<pricingengine/>",
                "todaysmarket.xml": "<todaysmarket/>",
                "curveconfig.xml": "<curveconfig/>",
                "simulation.xml": "<simulation/>",
            },
        ),
        netting=snap.netting,
        collateral=snap.collateral,
        source_meta=snap.source_meta,
    )

    mapped = map_snapshot(snap)

    assert "<PricingEngines>" in mapped.xml_buffers["pricingengine.xml"]
    assert "<TodaysMarket>" in mapped.xml_buffers["todaysmarket.xml"]
    assert "<CurveConfiguration>" in mapped.xml_buffers["curveconfig.xml"]
    assert "<Simulation>" in mapped.xml_buffers["simulation.xml"]
    assert "<Samples>10</Samples>" in mapped.xml_buffers["simulation.xml"]


def test_mapper_populates_input_parameters_like_object():
    fake = FakeInputParameters()
    build_input_parameters(_snapshot(), fake)
    called = {x[0] for x in fake.calls}
    assert "setPortfolio" in called
    assert "setPricingEngine" in called
    assert "setExposureAllocationMethod" in called
    assert "setExposureObservationModel" in called
    assert "setPfeQuantile" in called
    assert "setScenarioGenType" in called
    assert "setDimQuantile" in called
    assert "setDimHorizonCalendarDays" in called
    assert "setDimRegressionOrder" in called
    assert "setDimOutputNettingSet" in called


def test_mapper_builds_credit_simulation_xml_when_enabled():
    snap = _snapshot()
    runtime = RuntimeConfig(
        pricing_engine=snap.config.runtime.pricing_engine,
        todays_market=snap.config.runtime.todays_market,
        curve_configs=snap.config.runtime.curve_configs,
        simulation=snap.config.runtime.simulation,
        simulation_market=snap.config.runtime.simulation_market,
        cross_asset_model=snap.config.runtime.cross_asset_model,
        xva_analytic=snap.config.runtime.xva_analytic,
        conventions=snap.config.runtime.conventions,
        counterparties=snap.config.runtime.counterparties,
        credit_simulation=CreditSimulationConfig(
            enabled=True,
            netting_set_ids=("NS1",),
            entities=(CreditEntityConfig(name="CP_A"),),
            paths=123,
        ),
    )
    snap = XVASnapshot(
        market=snap.market,
        fixings=snap.fixings,
        portfolio=snap.portfolio,
        config=XVAConfig(
            asof=snap.config.asof,
            base_currency=snap.config.base_currency,
            analytics=snap.config.analytics,
            num_paths=snap.config.num_paths,
            runtime=runtime,
        ),
        netting=snap.netting,
        collateral=snap.collateral,
        source_meta=snap.source_meta,
    )
    mapped = map_snapshot(snap)
    assert "creditsimulation.xml" in mapped.xml_buffers
    assert "<CreditSimulation>" in mapped.xml_buffers["creditsimulation.xml"]
    assert "<Paths>123</Paths>" in mapped.xml_buffers["creditsimulation.xml"]


def test_mapper_fx_forward_value_date_uses_trade_maturity():
    snap = _snapshot()
    snap = XVASnapshot(
        market=snap.market,
        fixings=snap.fixings,
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T2",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.1, maturity_years=1.5),
                ),
            )
        ),
        config=XVAConfig(
            asof=snap.config.asof,
            base_currency=snap.config.base_currency,
            analytics=snap.config.analytics,
            num_paths=snap.config.num_paths,
            runtime=snap.config.runtime,
        ),
        netting=snap.netting,
        collateral=snap.collateral,
        source_meta=snap.source_meta,
    )

    mapped = map_snapshot(snap)

    assert "<ValueDate>2027-09-08</ValueDate>" in mapped.xml_buffers["portfolio.xml"]
    assert "2030-01-01" not in mapped.xml_buffers["portfolio.xml"]


def test_default_index_fallbacks_use_rfr_names():
    assert _default_index_for_ccy("USD") == "USD-SOFR"
    assert _default_index_for_ccy("GBP") == "GBP-SONIA"
    assert _default_index_for_ccy("CHF") == "CHF-SARON"
    assert _default_index_for_ccy("JPY") == "JPY-TONAR"
    assert _default_index_for_ccy("EUR") == "EUR-ESTER"


def test_mapper_generated_xml_is_multi_currency_and_strict_mode_is_config_driven():
    snap = _snapshot()
    runtime = RuntimeConfig(
        pricing_engine=snap.config.runtime.pricing_engine,
        todays_market=TodaysMarketConfig(market_id="default", discount_curve="USD-SOFR", fx_pairs=("GBPUSD",)),
        curve_configs=(
            CurveConfig(curve_id="USD-SOFR", currency="USD", tenors=("1Y", "5Y")),
            CurveConfig(curve_id="GBP-SONIA", currency="GBP", tenors=("1Y", "5Y")),
        ),
        simulation=SimulationConfig(samples=32, seed=9, dates=("6M", "1Y"), strict_template=True),
        simulation_market=SimulationMarketConfig(
            base_currency="USD",
            currencies=("USD", "GBP"),
            indices=("USD-SOFR", "GBP-SONIA"),
            default_curve_names=("BANK",),
            fx_pairs=("GBPUSD",),
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy="USD",
            currencies=("USD", "GBP"),
            ir_model_ccys=("USD", "GBP"),
            fx_model_ccys=("GBP",),
        ),
        xva_analytic=snap.config.runtime.xva_analytic,
        conventions=ConventionsConfig(day_counter="A365", calendar="UK"),
        counterparties=snap.config.runtime.counterparties,
    )
    portfolio = Portfolio(
        trades=(
            Trade(
                trade_id="IRS_GBP",
                counterparty="CP_A",
                netting_set="NS1",
                trade_type="Swap",
                product=IRS(ccy="GBP", notional=1_000_000, fixed_rate=0.03, maturity_years=5.0, pay_fixed=True),
            ),
        )
    )
    snap = XVASnapshot(
        market=snap.market,
        fixings=snap.fixings,
        portfolio=portfolio,
        config=XVAConfig(
            asof=snap.config.asof,
            base_currency="USD",
            analytics=snap.config.analytics,
            num_paths=16,
            runtime=runtime,
        ),
        netting=snap.netting,
        collateral=snap.collateral,
        source_meta=snap.source_meta,
    )

    mapped = map_snapshot(snap)
    todays_market_xml = mapped.xml_buffers["todaysmarket.xml"]
    simulation_xml = mapped.xml_buffers["simulation.xml"]
    curve_config_xml = mapped.xml_buffers["curveconfig.xml"]
    portfolio_xml = mapped.xml_buffers["portfolio.xml"]

    assert 'DiscountingCurve currency="USD"' in todays_market_xml
    assert 'DiscountingCurve currency="GBP"' in todays_market_xml
    assert '<Index name="USD-SOFR">' in todays_market_xml
    assert '<Index name="GBP-SONIA">' in todays_market_xml
    assert "EUR-ESTER" not in todays_market_xml
    assert "Default/USD/BANK" in todays_market_xml

    assert "<Calendar>USD,GBP</Calendar>" in simulation_xml
    assert "<Currency>USD</Currency>" in simulation_xml
    assert "<Currency>GBP</Currency>" in simulation_xml
    assert "<Name>USD-CMS-1Y</Name>" in simulation_xml
    assert "<Name>GBP-CMS-30Y</Name>" in simulation_xml
    assert "<Index>USD-SOFR</Index>" in simulation_xml
    assert "<Index>GBP-SONIA</Index>" in simulation_xml
    assert "USD-LIBOR" not in simulation_xml
    assert "EUR,USD" not in simulation_xml

    assert "<Currency>USD</Currency>" in curve_config_xml
    assert "<Currency>GBP</Currency>" in curve_config_xml
    assert "USD-ON-DEPOSIT-SOFR" in curve_config_xml
    assert "GBP-ON-DEPOSIT-SONIA" in curve_config_xml
    assert "CDS/CREDIT_SPREAD/BANK/SNRFOR/USD/*" in curve_config_xml

    assert "<Index>GBP-SONIA</Index>" in portfolio_xml
    assert "<Calendar>UK</Calendar>" in portfolio_xml


def test_mapper_counterparty_xml_uses_configured_risk_fields_and_curve_currency():
    snap = _snapshot()
    runtime = RuntimeConfig(
        pricing_engine=snap.config.runtime.pricing_engine,
        todays_market=snap.config.runtime.todays_market,
        curve_configs=snap.config.runtime.curve_configs,
        simulation=snap.config.runtime.simulation,
        simulation_market=SimulationMarketConfig(
            base_currency="EUR",
            currencies=("EUR", "CHF"),
            indices=("EUR-ESTER", "CHF-SARON"),
            default_curve_names=("CP_A",),
            fx_pairs=("CHFEUR",),
        ),
        cross_asset_model=snap.config.runtime.cross_asset_model,
        xva_analytic=snap.config.runtime.xva_analytic,
        conventions=snap.config.runtime.conventions,
        counterparties=CounterpartyConfig(
            ids=("CP_A",),
            curve_currencies={"CP_A": "CHF"},
            credit_qualities={"CP_A": "HY"},
            bacva_risk_weights={"CP_A": 0.11},
            saccr_risk_weights={"CP_A": 0.7},
            sacva_risk_buckets={"CP_A": 5},
        ),
    )
    portfolio = Portfolio(
        trades=(
            Trade(
                trade_id="BERM1",
                counterparty="CP_A",
                netting_set="NS1",
                trade_type="Swaption",
                product=BermudanSwaption(
                    ccy="EUR",
                    notional=1_000_000,
                    fixed_rate=0.02,
                    maturity_years=5.0,
                    pay_fixed=True,
                    exercise_dates=("2027-03-08", "2028-03-08"),
                ),
            ),
        )
    )
    snap = XVASnapshot(
        market=snap.market,
        fixings=snap.fixings,
        portfolio=portfolio,
        config=XVAConfig(
            asof=snap.config.asof,
            base_currency="EUR",
            analytics=snap.config.analytics,
            num_paths=snap.config.num_paths,
            runtime=runtime,
        ),
        netting=snap.netting,
        collateral=snap.collateral,
        source_meta=snap.source_meta,
    )

    mapped = map_snapshot(snap)

    assert "<CreditQuality>HY</CreditQuality>" in mapped.xml_buffers["counterparty.xml"]
    assert "<BaCvaRiskWeight>0.11</BaCvaRiskWeight>" in mapped.xml_buffers["counterparty.xml"]
    assert "<SaCcrRiskWeight>0.7</SaCcrRiskWeight>" in mapped.xml_buffers["counterparty.xml"]
    assert "<SaCvaRiskBucket>5</SaCvaRiskBucket>" in mapped.xml_buffers["counterparty.xml"]
    assert "Default/CHF/CP_A" in mapped.xml_buffers["todaysmarket.xml"]
    assert "CDS/CREDIT_SPREAD/CP_A/SNRFOR/CHF/*" in mapped.xml_buffers["curveconfig.xml"]
    assert '<Product type="BermudanSwaption">' in mapped.xml_buffers["pricingengine.xml"]
    assert "Gaussian1dNonstandardSwaptionEngine" in mapped.xml_buffers["pricingengine.xml"]
