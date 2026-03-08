from __future__ import annotations

from dataclasses import replace

import pytest

from native_xva_interface import (
    FXForward,
    FixingsData,
    GenericProduct,
    IRS,
    MarketData,
    MarketQuote,
    Portfolio,
    PythonLgmAdapter,
    Trade,
    XVAConfig,
    XVAEngine,
    XVAResult,
    XVASnapshot,
)


def _base_snapshot() -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/1Y", value=0.02),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/5Y", value=0.025),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/USD/1Y", value=0.03),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/USD/5Y", value=0.032),
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/1Y", value=0.02),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.02),
                MarketQuote(date="2026-03-08", key="RECOVERY_RATE/RATE/CPTY_A/SR/USD", value=0.4),
            ),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=5_000_000, fixed_rate=0.021, maturity_years=3.0, pay_fixed=True),
                ),
                Trade(
                    trade_id="FX1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=2_000_000, strike=1.09, maturity_years=1.0),
                ),
            )
        ),
        config=XVAConfig(
            asof="2026-03-08",
            base_currency="EUR",
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=64,
            xml_buffers={
                "simulation.xml": """
                <Simulation>
                  <Parameters><Seed>7</Seed><Samples>64</Samples></Parameters>
                  <CrossAssetModel>
                    <InterestRateModels>
                      <LGM ccy='EUR'>
                        <Volatility><TimeGrid>1.0,3.0</TimeGrid><InitialValue>0.01,0.012,0.013</InitialValue></Volatility>
                        <Reversion><TimeGrid/><InitialValue>0.03</InitialValue></Reversion>
                        <ParameterTransformation><ShiftHorizon>0.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
                      </LGM>
                    </InterestRateModels>
                  </CrossAssetModel>
                  <Market>
                    <YieldCurves><Configuration><Tenors>6M,1Y,2Y,5Y</Tenors></Configuration></YieldCurves>
                  </Market>
                </Simulation>
                """,
            },
        ),
    )


def test_python_lgm_adapter_supported_run_and_metadata():
    snap = _base_snapshot()
    result = XVAEngine(adapter=PythonLgmAdapter()).create_session(snap).run(return_cubes=False)

    assert result.metadata["engine"] == "python-lgm"
    assert result.metadata["coverage"]["python_trades"] == 2
    assert result.metadata["coverage"]["fallback_trades"] == 0
    assert set(result.xva_by_metric).issubset({"CVA", "DVA", "FVA", "MVA"})
    assert "CPTY_A" in result.exposure_by_netting_set


def test_python_lgm_adapter_market_overlay_impacts_results():
    low = _base_snapshot()
    high_market = tuple(
        replace(q, value=(0.06 if q.key == "ZERO/RATE/EUR/5Y" else q.value)) for q in low.market.raw_quotes
    )
    high = replace(low, market=replace(low.market, raw_quotes=high_market))

    adapter = PythonLgmAdapter()
    r_low = XVAEngine(adapter=adapter).create_session(low).run(return_cubes=False)
    r_high = XVAEngine(adapter=adapter).create_session(high).run(return_cubes=False)

    assert r_low.pv_total != r_high.pv_total


def test_python_lgm_adapter_unsupported_without_fallback_raises():
    snap = _base_snapshot()
    bad_trade = Trade(
        trade_id="GEN1",
        counterparty="CPTY_A",
        netting_set="CPTY_A",
        trade_type="Generic",
        product=GenericProduct(payload={"foo": "bar"}),
    )
    snap = replace(snap, portfolio=replace(snap.portfolio, trades=snap.portfolio.trades + (bad_trade,)))

    with pytest.raises(Exception, match="Unsupported trade types"):
        XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snap).run(return_cubes=False)


def test_python_lgm_adapter_unsupported_with_fallback_uses_swig(monkeypatch):
    snap = _base_snapshot()
    bad_trade = Trade(
        trade_id="GEN1",
        counterparty="CPTY_A",
        netting_set="CPTY_A",
        trade_type="Generic",
        product=GenericProduct(payload={"foo": "bar"}),
    )
    snap = replace(snap, portfolio=replace(snap.portfolio, trades=snap.portfolio.trades + (bad_trade,)))

    class _FakeSwig:
        def run(self, snapshot, mapped, run_id):
            return XVAResult(
                run_id=run_id,
                pv_total=10.0,
                xva_total=1.0,
                xva_by_metric={"CVA": 1.0},
                exposure_by_netting_set={"CPTY_A": 3.0},
                reports={"xva": [{"Metric": "CVA", "Value": 1.0}]},
                cubes={},
                metadata={"adapter": "fake"},
            )

    monkeypatch.setattr("native_xva_interface.runtime.ORESwigAdapter", _FakeSwig)

    result = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=True)).create_session(snap).run(return_cubes=False)

    assert result.metadata["coverage"]["fallback_trades"] == 1
    assert result.xva_by_metric["CVA"] >= 1.0


def test_engine_python_lgm_default_factory():
    engine = XVAEngine.python_lgm_default()
    assert isinstance(engine.adapter, PythonLgmAdapter)
