from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
import pytest

from native_xva_interface import (
    ConventionsConfig,
    CrossAssetModelConfig,
    CurveConfig,
    FXForward,
    FixingsData,
    FixingPoint,
    GenericProduct,
    IRS,
    MarketData,
    MarketQuote,
    MporConfig,
    NettingConfig,
    NettingSet,
    Portfolio,
    PricingEngineConfig,
    PythonLgmAdapter,
    RuntimeConfig,
    SimulationConfig,
    SimulationMarketConfig,
    TodaysMarketConfig,
    Trade,
    XVAConfig,
    XVAEngine,
    XVAResult,
    XVALoader,
    XVASnapshot,
)
from native_xva_interface.mapper import _resolve_mpor_config, build_input_parameters, map_snapshot
from native_xva_interface import runtime as runtime_module
from native_xva_interface.runtime import _build_zero_rate_shocked_curve, _quote_matches_discount_curve


_ORE_CASE_DIR = (
    Path(__file__).resolve().parents[2]
    / "parity_artifacts"
    / "multiccy_benchmark_final"
    / "cases"
    / "flat_EUR_5Y_A"
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


def test_python_lgm_adapter_zero_mpor_is_noop():
    snap = _base_snapshot()
    base = XVAEngine(adapter=PythonLgmAdapter()).create_session(snap).run(return_cubes=False)
    zero_mpor = replace(
        snap,
        config=replace(snap.config, params={**snap.config.params, "python.mpor_days": "0"}),
    )
    bumped = XVAEngine(adapter=PythonLgmAdapter()).create_session(zero_mpor).run(return_cubes=False)

    assert bumped.pv_total == pytest.approx(base.pv_total)
    assert bumped.xva_total == pytest.approx(base.xva_total)
    assert bumped.metadata["mpor_enabled"] is False


def test_python_lgm_adapter_sticky_mpor_changes_results_and_metadata():
    snap = replace(
        _base_snapshot(),
        config=replace(
            _base_snapshot().config,
            params={"python.mpor_days": "14"},
            xml_buffers={
                "simulation.xml": """
                <Simulation>
                  <Parameters><Grid>4,1Y</Grid><Seed>7</Seed><Samples>64</Samples></Parameters>
                  <CrossAssetModel>
                    <InterestRateModels>
                      <LGM ccy='EUR'>
                        <Volatility><TimeGrid>1.0,3.0</TimeGrid><InitialValue>0.01,0.012,0.013</InitialValue></Volatility>
                        <Reversion><TimeGrid/><InitialValue>0.03</InitialValue></Reversion>
                        <ParameterTransformation><ShiftHorizon>0.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
                      </LGM>
                    </InterestRateModels>
                  </CrossAssetModel>
                </Simulation>
                """,
            },
        ),
    )
    base = XVAEngine(adapter=PythonLgmAdapter()).create_session(_base_snapshot()).run(return_cubes=False)
    result = XVAEngine(adapter=PythonLgmAdapter()).create_session(snap).run(return_cubes=False)

    assert result.metadata["mpor_enabled"] is True
    assert result.metadata["mpor_days"] == 14
    assert result.metadata["mpor_mode"] == "sticky"
    assert result.metadata["mpor_source"] == "python.mpor_days"
    assert result.metadata["closeout_grid_size"] >= 1
    assert result.metadata["grid_size"] > result.metadata["valuation_grid_size"]
    assert result.xva_total != pytest.approx(base.xva_total)


def test_python_lgm_adapter_union_grid_closeout_not_valuation_interp_only():
    snap = replace(
        _base_snapshot(),
        config=replace(
            _base_snapshot().config,
            params={"python.mpor_days": "14"},
            xml_buffers={
                "simulation.xml": """
                <Simulation>
                  <Parameters><Grid>2,2Y</Grid><Seed>7</Seed><Samples>64</Samples></Parameters>
                  <CrossAssetModel>
                    <InterestRateModels>
                      <LGM ccy='EUR'>
                        <Volatility><TimeGrid>1.0,3.0</TimeGrid><InitialValue>0.01,0.012,0.013</InitialValue></Volatility>
                        <Reversion><TimeGrid/><InitialValue>0.03</InitialValue></Reversion>
                        <ParameterTransformation><ShiftHorizon>0.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
                      </LGM>
                    </InterestRateModels>
                  </CrossAssetModel>
                </Simulation>
                """,
            },
        ),
    )
    result = XVAEngine(adapter=PythonLgmAdapter()).create_session(snap).run(return_cubes=True)
    cube = result.cube("npv_cube").payload["IRS1"]

    assert "closeout_times" in cube
    assert "closeout_npv_mean" in cube
    assert len(cube["closeout_times"]) == len(cube["times"])
    assert cube["closeout_times"] != cube["times"]


def test_python_lgm_adapter_actual_mode_rejected():
    snap = replace(
        _base_snapshot(),
        config=replace(_base_snapshot().config, params={"python.mpor_mode": "actual", "python.mpor_days": "10"}),
    )

    with pytest.raises(Exception, match="python.mpor_mode"):
        XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snap).run(return_cubes=False)


def test_mpor_resolution_precedence_simulation_then_netting_then_python_override():
    snap = replace(
        _base_snapshot(),
        netting=NettingConfig(
            netting_sets={"CPTY_A": NettingSet(netting_set_id="CPTY_A", margin_period_of_risk="2W")}
        ),
        config=replace(
            _base_snapshot().config,
            params={"python.mpor_days": "7"},
            xml_buffers={
                "simulation.xml": """
                <Simulation><Parameters><CloseOutLag>1W</CloseOutLag><MporMode>StickyDate</MporMode></Parameters></Simulation>
                """,
            },
        ),
    )

    mpor = _resolve_mpor_config(snap, map_snapshot(snap).xml_buffers)
    assert mpor.mpor_days == 7
    assert mpor.source == "python.mpor_days"


def test_mpor_resolution_python_override_beats_pre_resolved_config():
    snap = replace(
        _base_snapshot(),
        config=replace(
            _base_snapshot().config,
            params={"python.mpor_days": "10"},
            mpor=MporConfig(
                enabled=False,
                mpor_years=0.0,
                mpor_days=0,
                closeout_lag_period="",
                sticky=True,
                cashflow_mode="NonePay",
                source="disabled",
            ),
        ),
    )

    mpor = _resolve_mpor_config(snap, map_snapshot(snap).xml_buffers)
    assert mpor.mpor_days == 10
    assert mpor.source == "python.mpor_days"


def test_mpor_resolution_warns_on_malformed_simulation_xml_and_keeps_precedence():
    snap = replace(
        _base_snapshot(),
        netting=NettingConfig(
            netting_sets={"CPTY_A": NettingSet(netting_set_id="CPTY_A", margin_period_of_risk="2W")}
        ),
        config=replace(
            _base_snapshot().config,
            params={"python.mpor_days": "7"},
            xml_buffers={"simulation.xml": "<<not-xml>>"},
        ),
    )

    with pytest.warns(UserWarning, match="Failed to parse simulation.xml"):
        mpor = _resolve_mpor_config(snap, map_snapshot(snap).xml_buffers)

    assert mpor.mpor_days == 7
    assert mpor.source == "python.mpor_days"


def test_mpor_resolution_warns_on_invalid_python_days_and_falls_back():
    snap = replace(
        _base_snapshot(),
        netting=NettingConfig(
            netting_sets={"CPTY_A": NettingSet(netting_set_id="CPTY_A", margin_period_of_risk="2W")}
        ),
        config=replace(
            _base_snapshot().config,
            params={"python.mpor_days": "abc"},
            xml_buffers={
                "simulation.xml": """
                <Simulation><Parameters><CloseOutLag>1W</CloseOutLag><MporMode>StickyDate</MporMode></Parameters></Simulation>
                """,
            },
        ),
    )

    with pytest.warns(UserWarning, match="Failed to parse python.mpor_days"):
        mpor = _resolve_mpor_config(snap, map_snapshot(snap).xml_buffers)

    assert mpor.closeout_lag_period == "2W"
    assert mpor.source == "netting:CPTY_A"


def test_python_lgm_generated_runtime_snapshot_keeps_stable_numbers():
    runtime = RuntimeConfig(
        pricing_engine=PricingEngineConfig(model="DiscountedCashflows", npv_engine="DiscountingFxForwardEngine"),
        todays_market=TodaysMarketConfig(market_id="default", discount_curve="EUR-EONIA", fx_pairs=("EURUSD",)),
        curve_configs=(
            CurveConfig(curve_id="EUR-EONIA", currency="EUR", tenors=("1Y", "5Y")),
            CurveConfig(curve_id="USD-SOFR", currency="USD", tenors=("1Y", "5Y")),
        ),
        simulation=SimulationConfig(samples=64, seed=7, dates=("1Y", "3Y")),
        simulation_market=SimulationMarketConfig(
            base_currency="EUR",
            currencies=("EUR", "USD"),
            indices=("EUR-ESTER", "USD-SOFR"),
            default_curve_names=("CPTY_A",),
            fx_pairs=("USDEUR",),
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy="EUR",
            currencies=("EUR", "USD"),
            ir_model_ccys=("EUR", "USD"),
            fx_model_ccys=("USD",),
        ),
        conventions=ConventionsConfig(day_counter="A360", calendar="TARGET"),
    )
    snap = replace(_base_snapshot(), config=replace(_base_snapshot().config, xml_buffers={}, runtime=runtime))

    result = XVAEngine(adapter=PythonLgmAdapter()).create_session(snap).run(return_cubes=False)

    assert np.isfinite(result.pv_total)
    assert np.isfinite(result.xva_total)
    assert result.metadata["coverage"]["python_trades"] == 2
    assert result.pv_total == pytest.approx(59318.445308124, rel=1.0e-10, abs=1.0e-8)
    assert result.xva_total == pytest.approx(9001.085926240548, rel=1.0e-10, abs=1.0e-8)


def test_flow_loaded_fixed_leg_dates_preserve_t0_pv_parity():
    snap = XVALoader.from_files(str(_ORE_CASE_DIR / "Input"), ore_file="ore.xml")
    snap = replace(snap, config=replace(snap.config, analytics=("CVA",), num_paths=256))

    adapter = PythonLgmAdapter(fallback_to_swig=False)
    mapped = map_snapshot(snap)
    adapter._ensure_py_lgm_imports()
    legs = adapter._build_irs_legs(snap.portfolio.trades[0], mapped, snap)

    assert "fixed_start_time" in legs
    assert "fixed_end_time" in legs
    assert legs["fixed_start_time"].size == legs["fixed_pay_time"].size
    assert np.all(legs["fixed_start_time"] >= -1.0e-12)

    result = XVAEngine(adapter=adapter).create_session(snap).run(return_cubes=False)
    assert result.pv_total == pytest.approx(823.56, abs=2.0)


def test_dual_curve_pricer_keeps_started_unpaid_coupons_alive():
    adapter = PythonLgmAdapter(fallback_to_swig=False)
    adapter._ensure_py_lgm_imports()

    model = adapter._lgm_mod.LGM1F(
        adapter._lgm_mod.LGMParams(
            alpha_times=(),
            alpha_values=(0.0,),
            kappa_times=(),
            kappa_values=(0.03,),
            shift=0.0,
            scaling=1.0,
        )
    )

    def p0(t: float) -> float:
        return float(np.exp(-0.02 * t))

    legs = {
        "fixed_start_time": np.asarray([0.25], dtype=float),
        "fixed_end_time": np.asarray([0.50], dtype=float),
        "fixed_pay_time": np.asarray([0.50], dtype=float),
        "fixed_amount": np.asarray([100.0], dtype=float),
        "float_pay_time": np.asarray([], dtype=float),
        "float_start_time": np.asarray([], dtype=float),
        "float_end_time": np.asarray([], dtype=float),
        "float_accrual": np.asarray([], dtype=float),
        "float_notional": np.asarray([], dtype=float),
        "float_sign": np.asarray([], dtype=float),
        "float_spread": np.asarray([], dtype=float),
        "float_coupon": np.asarray([], dtype=float),
        "float_fixing_time": np.asarray([], dtype=float),
    }

    pv = adapter._irs_utils.swap_npv_from_ore_legs_dual_curve(
        model,
        p0,
        p0,
        legs,
        0.30,
        np.asarray([0.0], dtype=float),
    )

    assert pv[0] > 0.0


def test_ore_backed_snapshot_exposes_xva_discount_curve_without_runtime_config():
    snap = XVALoader.from_files(str(_ORE_CASE_DIR / "Input"), ore_file="ore.xml")
    adapter = PythonLgmAdapter(fallback_to_swig=False)
    adapter._ensure_py_lgm_imports()

    inputs = adapter._extract_inputs(snap, map_snapshot(snap))

    assert inputs.xva_discount_curve is not None


def test_build_input_parameters_sets_native_mpor_fields():
    snap = replace(
        _base_snapshot(),
        config=replace(_base_snapshot().config, params={"python.mpor_days": "10"}),
    )

    class _FakeInputParameters:
        def __init__(self):
            self.calls = {}

        def __getattr__(self, name):
            def _recorder(*args):
                self.calls[name] = args
            return _recorder

    fake = _FakeInputParameters()
    build_input_parameters(snap, fake)

    assert fake.calls["setMporDays"] == (10,)
    assert fake.calls["setMporCalendar"] == ("EUR",)
    assert fake.calls["setMporForward"] == (True,)


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


def test_discount_quote_family_filter_uses_curve_family_not_all_zero_quotes():
    assert _quote_matches_discount_curve("MM/RATE/EUR/2D/6M", "EUR", "EUR-EURIBOR-6M") is True
    assert _quote_matches_discount_curve("IR_SWAP/RATE/EUR/2D/6M/5Y", "EUR", "EUR-EURIBOR-6M") is True
    assert _quote_matches_discount_curve("FRA/RATE/EUR/3M/6M", "EUR", "EUR-EURIBOR-6M") is True
    assert _quote_matches_discount_curve("IR_SWAP/RATE/EUR/2D/1D/5Y", "EUR", "EUR-EURIBOR-6M") is False
    assert _quote_matches_discount_curve("ZERO/RATE/EUR/BANK_EUR_BORROW/A365/5Y", "EUR", "EUR-EURIBOR-6M") is False


def test_zero_rate_shocked_curve_applies_local_node_shift():
    base = lambda t: float(np.exp(-0.02 * float(t)))
    shocked = _build_zero_rate_shocked_curve(base, [1.0, 5.0], [0.0, 0.0001])
    assert shocked(1.0) == pytest.approx(base(1.0))
    assert shocked(5.0) == pytest.approx(base(5.0) * np.exp(-0.0001 * 5.0))
    expected_mid = np.exp(0.5 * np.log(base(1.0)) + 0.5 * np.log(base(5.0) * np.exp(-0.0001 * 5.0)))
    assert shocked(3.0) == pytest.approx(expected_mid)


def test_runtime_lgm_param_parsers_ignore_shift_horizon():
    simulation_xml = """
    <Simulation>
      <CrossAssetModel>
        <InterestRateModels>
          <LGM ccy="EUR">
            <Volatility><TimeGrid>1.0</TimeGrid><InitialValue>0.01,0.02</InitialValue></Volatility>
            <Reversion><TimeGrid/><InitialValue>0.03</InitialValue></Reversion>
            <ParameterTransformation><ShiftHorizon>15.0</ShiftHorizon><Scaling>1.25</Scaling></ParameterTransformation>
          </LGM>
        </InterestRateModels>
      </CrossAssetModel>
    </Simulation>
    """
    calibration_xml = """
    <CrossAssetModelData>
      <InterestRateModels>
        <LGM key="EUR">
          <Volatility><TimeGrid>1.0</TimeGrid><InitialValue>0.011,0.021</InitialValue></Volatility>
          <Reversion><TimeGrid/><InitialValue>0.031</InitialValue></Reversion>
          <ParameterTransformation><ShiftHorizon>25.0</ShiftHorizon><Scaling>1.5</Scaling></ParameterTransformation>
        </LGM>
      </InterestRateModels>
    </CrossAssetModelData>
    """

    sim = runtime_module._parse_lgm_params_from_simulation_xml_text(simulation_xml, ccy_key="EUR")
    cal = runtime_module._parse_lgm_params_from_calibration_xml_text(calibration_xml, ccy_key="EUR")

    assert sim["shift"] == pytest.approx(0.0)
    assert cal["shift"] == pytest.approx(0.0)
    assert sim["scaling"] == pytest.approx(1.25)
    assert cal["scaling"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Correctness regression tests for runtime.py cleanup (#1 and #2)
# ---------------------------------------------------------------------------


def test_assemble_result_does_not_duplicate_fba_fca_rows():
    """FBA and FCA must each appear exactly once in reports['xva'] when FVA is in analytics.

    Bug: _assemble_result builds reports["xva"] from xva_by_metric (which already
    contains FBA/FCA keys when FVA is active), then unconditionally appends FBA and
    FCA again — producing two rows for each.
    """
    snap = _base_snapshot()  # analytics=("CVA", "DVA", "FVA", "MVA")
    result = XVAEngine(adapter=PythonLgmAdapter()).create_session(snap).run(return_cubes=False)

    metrics = [row["Metric"] for row in result.reports.get("xva", [])]
    assert metrics.count("FBA") == 1, f"FBA duplicated in xva report: {metrics}"
    assert metrics.count("FCA") == 1, f"FCA duplicated in xva report: {metrics}"


def test_build_irs_legs_warns_on_flows_csv_parse_failure(tmp_path, monkeypatch):
    """A failure while parsing flows.csv must surface as a UserWarning, not be silently
    swallowed.

    Bug: the outer except-block in _build_irs_legs is a bare ``except: pass`` that
    discards any exception from the flows.csv loading path without any diagnostic.
    """
    # Layout ORE expects: ore.xml lives in Input/, Output/ is a sibling of Input/.
    ore_xml = tmp_path / "Input" / "ore.xml"
    ore_xml.parent.mkdir()
    ore_xml.write_text("<ORE/>", encoding="utf-8")
    output_dir = tmp_path / "Output"
    output_dir.mkdir()
    # A minimal flows.csv that exists so the code enters the loading path.
    (output_dir / "flows.csv").write_text("TradeId,FlowType\n", encoding="utf-8")

    snap = replace(
        _base_snapshot(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.021, maturity_years=1.0, pay_fixed=True),
                ),
            )
        ),
        config=replace(
            _base_snapshot().config,
            source_meta=replace(_base_snapshot().config.source_meta, path=str(ore_xml)),
            xml_buffers={},
            params={"outputPath": "Output"},
        ),
    )

    adapter = PythonLgmAdapter()
    adapter._ensure_py_lgm_imports()

    def _bad_load(*args, **kwargs):
        raise ValueError("corrupt flows.csv data")

    monkeypatch.setattr(adapter._irs_utils, "load_ore_legs_from_flows", _bad_load)
    mapped = map_snapshot(snap)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legs = adapter._build_irs_legs(snap.portfolio.trades[0], mapped, snap)

    # The fallback must still produce usable legs even after the parse failure.
    assert legs is not None and len(legs) > 0, "Expected fallback legs even after flows.csv failure"

    # A diagnostic warning must have been emitted — not silently swallowed.
    warning_messages = [str(w.message) for w in caught]
    assert any("flows.csv" in m.lower() or "IRS1" in m for m in warning_messages), (
        f"Expected a UserWarning about flows.csv parse failure, but got: {warning_messages}"
    )
