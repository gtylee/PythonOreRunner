from native_xva_interface import (
    ConventionsConfig,
    CounterpartyConfig,
    CreditEntityConfig,
    CreditSimulationConfig,
    CurveConfig,
    FXForward,
    MarketData,
    MarketQuote,
    Portfolio,
    PricingEngineConfig,
    RuntimeConfig,
    SimulationConfig,
    Trade,
    TodaysMarketConfig,
    XVAAnalyticConfig,
    XVAConfig,
    XVASnapshot,
    FixingsData,
    build_input_parameters,
    map_snapshot,
)


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
    assert "<Samples>10</Samples>" in mapped.xml_buffers["simulation.xml"]
    assert "<CrossAssetModel>" in mapped.xml_buffers["simulation.xml"]
    assert "<Market>" in mapped.xml_buffers["simulation.xml"]


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
