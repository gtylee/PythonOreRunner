from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from native_xva_interface import (
    FXForward,
    FixingsData,
    FixingPoint,
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
from native_xva_interface.mapper import map_snapshot


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
    assert set(result.xva_by_metric).issubset({"CVA", "DVA", "FVA", "FBA", "FCA", "MVA"})
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


def test_python_lgm_adapter_injects_historical_fixings_into_fallback_irs_legs():
    snap = replace(
        _base_snapshot(),
        fixings=FixingsData(
            points=(
                FixingPoint(date="2026-03-06", index="EUR-EURIBOR-3M", value=0.018),
            )
        ),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=5_000_000, fixed_rate=0.021, maturity_years=1.0, pay_fixed=True),
                ),
            )
        ),
        config=replace(_base_snapshot().config, xml_buffers={}),
    )

    adapter = PythonLgmAdapter()
    legs = adapter._build_irs_legs(snap.portfolio.trades[0], map_snapshot(snap), snap)

    assert float(legs["float_fixing_time"][0]) < 0.0
    assert bool(legs["float_is_historically_fixed"][0]) is True
    assert legs["float_coupon"][0] == pytest.approx(0.018)
    assert bool(legs["float_is_historically_fixed"][1]) is False


def test_python_lgm_adapter_augments_grid_with_irs_pay_dates():
    snap = replace(
        _base_snapshot(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=5_000_000, fixed_rate=0.021, maturity_years=3.0, pay_fixed=True),
                ),
            )
        ),
    )

    adapter = PythonLgmAdapter()
    adapter._ensure_py_lgm_imports()
    inputs = adapter._extract_inputs(snap, map_snapshot(snap))
    legs = inputs.trade_specs[0].legs
    assert legs is not None

    final_fixed_pay = float(legs["fixed_pay_time"][-1])
    final_float_pay = float(legs["float_pay_time"][-1])
    assert np.any(np.isclose(inputs.times, final_fixed_pay))
    assert np.any(np.isclose(inputs.times, final_float_pay))


def test_python_lgm_adapter_rejects_partial_ore_snapshot_missing_simulation_xml():
    base = _base_snapshot()
    snap = replace(
        base,
        config=replace(
            base.config,
            xml_buffers={},
            source_meta=replace(base.config.source_meta, path="/tmp/ore_case.xml"),
        ),
    )

    with pytest.raises(Exception, match="requires 'simulation.xml'"):
        XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snap).run(return_cubes=False)


def test_python_lgm_adapter_explains_ore_curve_loading_failure(tmp_path):
    base = _base_snapshot()
    input_dir = tmp_path / "Input"
    output_dir = tmp_path / "Output"
    input_dir.mkdir()
    output_dir.mkdir()
    ore_xml = input_dir / "ore_case.xml"
    ore_xml.write_text(
        """
        <ORE>
          <Analytics>
            <Analytic type="curves">
              <Parameter name="configuration">default</Parameter>
            </Analytic>
          </Analytics>
        </ORE>
        """,
        encoding="utf-8",
    )
    (output_dir / "curves.csv").write_text("date,curveId,discountFactor\n", encoding="utf-8")

    snap = replace(
        base,
        config=replace(
            base.config,
            source_meta=replace(base.config.source_meta, path=str(ore_xml)),
            xml_buffers={
                **base.config.xml_buffers,
                "todaysmarket.xml": "<TodaysMarket/>",
            },
            params={"outputPath": "Output"},
        ),
    )

    adapter = PythonLgmAdapter(fallback_to_swig=False)
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snap)

    with pytest.raises(Exception, match="Failed to build native curves from ORE output artifacts"):
        adapter._load_ore_output_curves(snap, mapped, ())
