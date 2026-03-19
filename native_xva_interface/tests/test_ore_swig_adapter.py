from native_xva_interface import (
    FXForward,
    FixingsData,
    MarketData,
    MarketQuote,
    ORESwigAdapter,
    Portfolio,
    Trade,
    XVAConfig,
    XVASnapshot,
    map_snapshot,
)


class FakeInputParameters:
    def __init__(self):
        self.values = {}

    def setAsOfDate(self, s):
        self.values["asof"] = s

    def setBaseCurrency(self, s):
        self.values["base"] = s

    def setAnalytics(self, s):
        self.values["analytics"] = s

    def setPortfolio(self, xml):
        self.values["portfolio"] = xml

    def setNettingSetManager(self, xml):
        self.values["netting"] = xml

    def setCollateralBalances(self, xml):
        self.values["collateral"] = xml

    def setPricingEngine(self, xml):
        self.values["pricing"] = xml

    def setTodaysMarketParams(self, xml):
        self.values["market"] = xml

    def setCurveConfigs(self, xml, id=""):
        self.values["curve"] = (xml, id)


class FakeOREApp:
    def __init__(self, inputs, *args):
        self.inputs = inputs
        self.market_data = None
        self.fixing_data = None

    def run(self, market_data=None, fixing_data=None):
        self.market_data = market_data or []
        self.fixing_data = fixing_data or []

    def getReportNames(self):
        return ["xva", "exposure", "npv"]

    def getReport(self, name):
        if name == "xva":
            return [{"Metric": "CVA", "Value": 12.5}, {"Metric": "FVA", "Value": 2.5}]
        if name == "exposure":
            return [{"NettingSetId": "NS1", "EPE": 1000.0}]
        return [{"NPV(Base)": 50.0}]

    def getCubeNames(self):
        return ["cube"]

    def getCube(self, name):
        return {"name": name}

    def getMarketCubeNames(self):
        return []


class FakeModule:
    InputParameters = FakeInputParameters
    OREApp = FakeOREApp


class FakeOREAppAdvanced(FakeOREApp):
    def getReportNames(self):
        return ["xva", "exposure_trade", "npv"]

    def getReport(self, name):
        if name == "xva":
            return [{"NettingSetId": "NS1", "TradeId": "", "CVA": "10.0", "FBA": "2.0", "FCA": "1.0"}]
        if name == "exposure_trade":
            return [
                {"NettingSetId": "NS1", "ExpectedPositiveExposure": "3.0"},
                {"NettingSetId": "NS1", "EPE": "2.0"},
            ]
        if name == "npv":
            return [{"NPV(Base)": "20.0"}, {"NPV(Base)": "5.0"}]
        return []


class FakeModuleAdvanced:
    InputParameters = FakeInputParameters
    OREApp = FakeOREAppAdvanced


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
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", analytics=("CVA", "FVA")),
    )


def test_ore_swig_adapter_with_fake_module_runs_and_extracts_outputs():
    adapter = ORESwigAdapter(module=FakeModule)
    snap = _snapshot()
    result = adapter.run(snap, mapped=map_snapshot(snap), run_id="r1")

    assert result.xva_by_metric["CVA"] == 12.5
    assert result.xva_by_metric["FVA"] == 2.5
    assert result.exposure_by_netting_set["NS1"] == 1000.0
    assert result.pv_total == 50.0
    assert "cube" in result.cubes


def test_ore_swig_adapter_extracts_advanced_report_layouts():
    adapter = ORESwigAdapter(module=FakeModuleAdvanced)
    snap = _snapshot()
    result = adapter.run(snap, mapped=map_snapshot(snap), run_id="r2")

    assert result.xva_by_metric["CVA"] == 10.0
    assert result.xva_by_metric["FVA"] == 3.0
    assert result.exposure_by_netting_set["NS1"] == 5.0
    assert result.pv_total == 25.0
