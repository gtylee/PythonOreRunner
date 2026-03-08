from native_xva_interface import (
    ConventionsConfig,
    CounterpartyConfig,
    CreditEntityConfig,
    CreditSimulationConfig,
    CrossAssetModelConfig,
    CurveConfig,
    FXForward,
    FixingsData,
    MarketData,
    MarketQuote,
    ORESwigAdapter,
    Portfolio,
    PricingEngineConfig,
    RuntimeConfig,
    SimulationConfig,
    SimulationMarketConfig,
    TodaysMarketConfig,
    Trade,
    XVAAnalyticConfig,
    XVAConfig,
    XVASnapshot,
    build_input_parameters,
    map_snapshot,
)


class CaptureInputParameters:
    def __init__(self):
        self.calls = {}

    def _record(self, name, *args):
        self.calls[name] = args

    def __getattr__(self, name):
        if name.startswith("set"):
            return lambda *args: self._record(name, *args)
        raise AttributeError(name)


class FakeOREAppComprehensive:
    def __init__(self, inputs, *args):
        self.inputs = inputs
        self.market_data = None
        self.fixing_data = None

    def run(self, market_data=None, fixing_data=None):
        self.market_data = market_data or []
        self.fixing_data = fixing_data or []

    def getReportNames(self):
        return ["xva", "exposure_trade", "npv"]

    def getReport(self, name):
        if name == "xva":
            return [{"NettingSetId": "CPTY_A", "TradeId": "", "CVA": "100.0", "DVA": "10.0", "FBA": "4.0", "FCA": "1.0", "MVA": "2.0"}]
        if name == "exposure_trade":
            return [{"NettingSetId": "CPTY_A", "ExpectedPositiveExposure": "25.0"}]
        if name == "npv":
            return [{"NPV(Base)": "12.0"}, {"NPV(Base)": "8.0"}]
        return []

    def getCubeNames(self):
        return ["cube"]

    def getCube(self, name):
        return {"name": name}

    def getMarketCubeNames(self):
        return ["mkt"]

    def getMarketCube(self, name):
        return {"name": name}


class FakeModuleComprehensive:
    InputParameters = CaptureInputParameters
    OREApp = FakeOREAppComprehensive


def _snapshot() -> XVASnapshot:
    runtime = RuntimeConfig(
        pricing_engine=PricingEngineConfig(model="DiscountedCashflows", npv_engine="DiscountingFxForwardEngine"),
        todays_market=TodaysMarketConfig(market_id="default", discount_curve="EUR-EONIA", fx_pairs=("EURUSD", "EURGBP")),
        curve_configs=(
            CurveConfig(curve_id="EUR-EONIA", currency="EUR", tenors=("1Y", "2Y", "5Y")),
            CurveConfig(curve_id="USD-SOFR", currency="USD", tenors=("1Y", "2Y", "5Y")),
        ),
        simulation=SimulationConfig(samples=77, seed=13, dates=("1Y", "2Y", "5Y")),
        simulation_market=SimulationMarketConfig(
            base_currency="EUR",
            currencies=("EUR", "USD"),
            indices=("EUR-ESTER", "USD-SOFR"),
            default_curve_names=("BANK", "CPTY_A"),
            fx_pairs=("USDEUR",),
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy="EUR",
            currencies=("EUR", "USD"),
            ir_model_ccys=("EUR", "USD"),
            fx_model_ccys=("USD",),
        ),
        xva_analytic=XVAAnalyticConfig(
            exposure_allocation_method="None",
            exposure_observation_model="Disable",
            pfe_quantile=0.95,
            collateral_calculation_type="Symmetric",
            marginal_allocation_limit=1.0,
            exercise_next_break=False,
            scenario_gen_type="SIMPLE",
            netting_set_ids=("CPTY_A",),
            cva_enabled=True,
            dva_enabled=True,
            fva_enabled=True,
            mva_enabled=True,
            colva_enabled=False,
            dim_enabled=True,
            full_initial_collateralisation=False,
            flip_view_xva=False,
            dim_quantile=0.99,
            dim_horizon_calendar_days=14,
            dim_regression_order=2,
            dim_regressors="",
            dim_output_netting_set="CPTY_A",
            dim_output_grid_points="0",
            dim_local_regression_evaluations=0,
            dim_local_regression_bandwidth=1.0,
            dva_name="BANK",
            fva_borrowing_curve="BANK_EUR_BORROW",
            fva_lending_curve="BANK_EUR_LEND",
        ),
        credit_simulation=CreditSimulationConfig(
            enabled=True,
            netting_set_ids=("CPTY_A",),
            entities=(CreditEntityConfig(name="CPTY_A"), CreditEntityConfig(name="BANK")),
            paths=200,
            seed=42,
        ),
        conventions=ConventionsConfig(day_counter="A365", calendar="TARGET"),
        counterparties=CounterpartyConfig(ids=("CPTY_A",)),
    )
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/1Y", value=0.02),
            ),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.09, maturity_years=1.0),
                ),
            )
        ),
        config=XVAConfig(
            asof="2026-03-08",
            base_currency="EUR",
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=10,
            runtime=runtime,
            params={"market.default": "default"},
        ),
    )


def test_comprehensive_mapping_sets_expected_swig_inputs():
    snap = _snapshot()
    mapped = map_snapshot(snap)

    assert "<Simulation>" in mapped.xml_buffers["simulation.xml"]
    assert "<CrossAssetModel>" in mapped.xml_buffers["simulation.xml"]
    assert "<Market>" in mapped.xml_buffers["simulation.xml"]
    assert "<CreditSimulation>" in mapped.xml_buffers["creditsimulation.xml"]
    assert "<DefaultCurves>" in mapped.xml_buffers["curveconfig.xml"]

    ip = CaptureInputParameters()
    build_input_parameters(snap, ip)

    # Core buffers and analytics wiring
    assert ip.calls["setAnalytics"][0] == "PRICING,EXPOSURE,XVA"
    assert "setPortfolio" in ip.calls
    assert "setPricingEngine" in ip.calls
    assert "setScenarioGeneratorData" in ip.calls
    assert "setCreditSimulationParametersFromBuffer" in ip.calls

    # XVA-specific setter coverage
    assert ip.calls["setExposureAllocationMethod"][0] == "None"
    assert ip.calls["setExposureObservationModel"][0] == "Disable"
    assert ip.calls["setPfeQuantile"][0] == 0.95
    assert ip.calls["setCollateralCalculationType"][0] == "Symmetric"
    assert ip.calls["setMarginalAllocationLimit"][0] == 1.0
    assert ip.calls["setScenarioGenType"][0] == "SIMPLE"
    assert ip.calls["setNettingSetId"][0] == "CPTY_A"
    assert ip.calls["setDvaName"][0] == "BANK"
    assert ip.calls["setFvaBorrowingCurve"][0] == "BANK_EUR_BORROW"
    assert ip.calls["setFvaLendingCurve"][0] == "BANK_EUR_LEND"
    assert ip.calls["setDimQuantile"][0] == 0.99
    assert ip.calls["setDimHorizonCalendarDays"][0] == 14
    assert ip.calls["setDimRegressionOrder"][0] == 2
    assert ip.calls["setDimOutputNettingSet"][0] == "CPTY_A"
    assert ip.calls["setDimOutputGridPoints"][0] == "0"


def test_comprehensive_fake_swig_run_parses_metrics_and_reports():
    snap = _snapshot()
    adapter = ORESwigAdapter(module=FakeModuleComprehensive)
    result = adapter.run(snap, mapped=map_snapshot(snap), run_id="r-comprehensive")

    assert result.pv_total == 20.0
    assert result.xva_by_metric["CVA"] == 100.0
    assert result.xva_by_metric["DVA"] == 10.0
    assert result.xva_by_metric["MVA"] == 2.0
    assert result.xva_by_metric["FVA"] == 5.0
    assert result.exposure_by_netting_set["CPTY_A"] == 25.0
    assert "cube" in result.cubes
    assert "market::mkt" in result.cubes
