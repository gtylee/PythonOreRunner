from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from pythonore.domain.dataclasses import (
    CollateralBalance,
    CollateralConfig,
    FXForward,
    FixingsData,
    GenericProduct,
    IRS,
    MarketData,
    MarketQuote,
    NettingConfig,
    NettingSet,
    Portfolio,
    RuntimeConfig,
    Trade,
    XVAAnalyticConfig,
    XVAConfig,
    XVASnapshot,
)
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.runtime import (
    DeterministicToyAdapter,
    PythonLgmAdapter,
    XVAEngine,
    _quote_matches_discount_curve,
    classify_portfolio_support,
)
from pythonore.mapping.mapper import map_snapshot


def _make_snapshot(*, include_unsupported: bool = False, runtime: RuntimeConfig | None = None, params=None) -> XVASnapshot:
    trades = [
        Trade(
            trade_id="IRS_DEMO_1",
            counterparty="CP_A",
            netting_set="NS_EUR",
            trade_type="Swap",
            product=IRS(ccy="EUR", notional=5_000_000, fixed_rate=0.024, maturity_years=5.0, pay_fixed=True),
        ),
        Trade(
            trade_id="FXFWD_DEMO_1",
            counterparty="CP_A",
            netting_set="NS_EUR",
            trade_type="FxForward",
            product=FXForward(pair="EURUSD", notional=2_000_000, strike=1.11, maturity_years=1.0, buy_base=True),
        ),
    ]
    if include_unsupported:
        trades.append(
            Trade(
                trade_id="UNSUPPORTED_1",
                counterparty="CP_A",
                netting_set="NS_EUR",
                trade_type="EquityOption",
                product=GenericProduct(payload={"trade_type": "EquityOption"}),
            )
        )
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/1Y", value=0.0210),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/USD/1Y", value=0.0315),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/EUR-ESTR/1Y/5Y", value=0.0230),
            ),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(trades=tuple(trades)),
        netting=NettingConfig(
            netting_sets={
                "NS_EUR": NettingSet(
                    netting_set_id="NS_EUR",
                    counterparty="CP_A",
                    active_csa=True,
                    csa_currency="EUR",
                )
            }
        ),
        collateral=CollateralConfig(
            balances=(CollateralBalance(netting_set_id="NS_EUR", currency="EUR"),)
        ),
        config=XVAConfig(
            asof="2026-03-08",
            base_currency="EUR",
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=8,
            horizon_years=5,
            runtime=runtime,
            params=dict(params or {}),
        ),
    )


def _cashflow_trade_xml(*, payment_date: str, amount: float, currency: str = "EUR") -> str:
    return f"""
<CashflowData>
  <PaymentDate>{payment_date}</PaymentDate>
  <Amount>{amount}</Amount>
  <Currency>{currency}</Currency>
</CashflowData>
""".strip()


def _generic_capfloor_trade_xml() -> str:
    return """
<CapFloorData>
  <LongShort>Long</LongShort>
  <Caps><Cap>0.03</Cap></Caps>
  <LegData>
    <LegType>Floating</LegType>
    <Currency>EUR</Currency>
    <PaymentConvention>F</PaymentConvention>
    <DayCounter>A360</DayCounter>
    <Notionals><Notional>1000000</Notional></Notionals>
    <ScheduleData>
      <Rules>
        <StartDate>2026-03-08</StartDate>
        <EndDate>2027-09-08</EndDate>
        <Tenor>6M</Tenor>
        <Calendar>TARGET</Calendar>
        <Convention>F</Convention>
      </Rules>
    </ScheduleData>
    <FloatingLegData>
      <Index>EUR-EURIBOR-6M</Index>
      <FixingDays>2</FixingDays>
      <IsInArrears>false</IsInArrears>
      <Spreads><Spread>0.0</Spread></Spreads>
      <Gearings><Gearing>1.0</Gearing></Gearings>
    </FloatingLegData>
  </LegData>
</CapFloorData>
""".strip()


def _generic_capfloor_in_arrears_trade_xml() -> str:
    return """
<CapFloorData>
  <LongShort>Long</LongShort>
  <Caps><Cap>0.03</Cap></Caps>
  <LegData>
    <LegType>Floating</LegType>
    <Currency>USD</Currency>
    <PaymentConvention>F</PaymentConvention>
    <DayCounter>A360</DayCounter>
    <Notionals><Notional>1000000</Notional></Notionals>
    <ScheduleData>
      <Rules>
        <StartDate>2026-03-08</StartDate>
        <EndDate>2027-09-08</EndDate>
        <Tenor>3M</Tenor>
        <Calendar>US</Calendar>
        <Convention>F</Convention>
      </Rules>
    </ScheduleData>
    <FloatingLegData>
      <Index>USD-SOFR-3M</Index>
      <FixingDays>0</FixingDays>
      <IsInArrears>true</IsInArrears>
      <Spreads><Spread>0.0</Spread></Spreads>
      <Gearings><Gearing>1.0</Gearing></Gearings>
    </FloatingLegData>
  </LegData>
</CapFloorData>
""".strip()


def _generic_swaption_trade_xml() -> str:
    return """
<SwaptionData>
  <OptionData>
    <Style>European</Style>
    <Settlement>Physical</Settlement>
    <LongShort>Long</LongShort>
    <ExerciseDates><ExerciseDate>2026-09-08</ExerciseDate></ExerciseDates>
  </OptionData>
  <LegData>
    <LegType>Fixed</LegType>
    <Currency>EUR</Currency>
    <Payer>true</Payer>
    <PaymentConvention>F</PaymentConvention>
    <DayCounter>30/360</DayCounter>
    <Notionals><Notional>1000000</Notional></Notionals>
    <ScheduleData>
      <Rules>
        <StartDate>2026-09-08</StartDate>
        <EndDate>2028-09-08</EndDate>
        <Tenor>1Y</Tenor>
        <Calendar>TARGET</Calendar>
        <Convention>F</Convention>
      </Rules>
    </ScheduleData>
    <FixedLegData><Rates><Rate>0.025</Rate></Rates></FixedLegData>
  </LegData>
  <LegData>
    <LegType>Floating</LegType>
    <Currency>EUR</Currency>
    <Payer>false</Payer>
    <PaymentConvention>F</PaymentConvention>
    <DayCounter>A360</DayCounter>
    <Notionals><Notional>1000000</Notional></Notionals>
    <ScheduleData>
      <Rules>
        <StartDate>2026-09-08</StartDate>
        <EndDate>2028-09-08</EndDate>
        <Tenor>6M</Tenor>
        <Calendar>TARGET</Calendar>
        <Convention>F</Convention>
      </Rules>
    </ScheduleData>
    <FloatingLegData>
      <Index>EUR-EURIBOR-6M</Index>
      <FixingDays>2</FixingDays>
      <Spreads><Spread>0.0</Spread></Spreads>
    </FloatingLegData>
  </LegData>
</SwaptionData>
""".strip()


def _simulation_xml_with_grid(grid: str) -> str:
    return f"""
<Simulation>
  <Parameters><Grid>{grid}</Grid></Parameters>
  <CrossAssetModel>
    <InterestRateModels>
      <LGM ccy="EUR">
        <Volatility><InitialValue>0.01</InitialValue></Volatility>
        <Reversion><InitialValue>0.03</InitialValue></Reversion>
        <ParameterTransformation><ShiftHorizon>0</ShiftHorizon><Scaling>1</Scaling></ParameterTransformation>
      </LGM>
    </InterestRateModels>
  </CrossAssetModel>
</Simulation>
""".strip()


def _dynamic_im_feeder():
    fx_delta = [[2.0, 2.1, 2.2]]
    fx_vega = [[[0.2, 0.22, 0.24]]]
    return {
        "dim_model": "DynamicIM",
        "currencies": ["EUR", "USD"],
        "ir_delta_terms": ["1Y", "5Y"],
        "ir_vega_terms": ["1Y"],
        "fx_vega_terms": ["1Y"],
        "simm_config": {
            "corr_ir_fx": 0.25,
            "ir_delta_rw": [0.01, 0.02],
            "ir_vega_rw": 0.30,
            "ir_gamma": 0.50,
            "ir_curvature_scaling": 0.75,
            "ir_delta_correlations": [[1.0, 0.2], [0.2, 1.0]],
            "ir_vega_correlations": [[1.0]],
            "ir_curvature_weights": [1.0],
            "fx_delta_rw": 0.15,
            "fx_vega_rw": 0.35,
            "fx_sigma": 0.20,
            "fx_hvr": 0.50,
            "fx_corr": 0.30,
            "fx_vega_correlations": [[1.0]],
            "fx_curvature_weights": [1.0],
        },
        "netting_sets": {
            "NS_EUR": {
                "current_slice": {
                    "time": 0.0,
                    "date": "2026-03-08",
                    "days_in_period": 0,
                    "numeraire": [1.0, 1.0, 1.0],
                    "ir_delta": [
                        [[10.0, 12.0, 14.0], [8.0, 9.0, 10.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    "ir_vega": [
                        [[1.0, 1.1, 1.2]],
                        [[0.0, 0.0, 0.0]],
                    ],
                    "fx_delta": fx_delta,
                    "fx_vega": fx_vega,
                },
                "time_slices": [
                    {
                        "time": 0.5,
                        "date": "2026-09-08",
                        "days_in_period": 182,
                        "numeraire": [1.0, 1.01, 1.02],
                        "ir_delta": [
                            [[9.0, 10.0, 11.0], [7.5, 8.0, 8.5]],
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        ],
                        "ir_vega": [
                            [[0.9, 1.0, 1.1]],
                            [[0.0, 0.0, 0.0]],
                        ],
                        "fx_delta": fx_delta,
                        "fx_vega": fx_vega,
                        "flow": [0.1, 0.1, 0.1],
                    }
                ],
            }
        },
    }


def test_classify_portfolio_support_flags_swig_only_trades():
    snapshot = _make_snapshot(include_unsupported=True)
    support = classify_portfolio_support(snapshot, fallback_to_swig=False)

    assert support["mode"] == "native_only"
    assert support["native_trade_count"] == 2
    assert support["requires_swig_trade_count"] == 1
    assert support["requires_swig_trade_ids"] == ["UNSUPPORTED_1"]
    assert support["requires_swig_trade_types"] == ["EquityOption"]


def test_xva_session_incremental_updates_track_rebuild_counts():
    snapshot = _make_snapshot()
    session = XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot)

    base = session.run(return_cubes=False)
    assert base.metadata["rebuild_counts"] == {"market": 1, "portfolio": 1, "config": 1}

    session.update_market(snapshot.market)
    no_op = session.run(return_cubes=False)
    assert no_op.metadata["rebuild_counts"] == {"market": 1, "portfolio": 1, "config": 1}

    bumped_market = replace(
        snapshot.market,
        raw_quotes=tuple(
            replace(q, value=q.value + 0.001) if q.key == "ZERO/RATE/EUR/1Y" else q
            for q in snapshot.market.raw_quotes
        ),
    )
    session.update_market(bumped_market)
    after_market = session.run(return_cubes=False)
    assert after_market.metadata["rebuild_counts"] == {"market": 2, "portfolio": 1, "config": 1}

    session.update_config(num_paths=16)
    after_config = session.run(return_cubes=False)
    assert after_config.metadata["rebuild_counts"] == {"market": 2, "portfolio": 1, "config": 2}

    session.update_portfolio(
        add=[
            Trade(
                trade_id="FXFWD_PATCH_2",
                counterparty="CP_A",
                netting_set="NS_EUR",
                trade_type="FxForward",
                product=FXForward(pair="EURUSD", notional=1_500_000, strike=1.12, maturity_years=1.5, buy_base=False),
            )
        ]
    )
    after_portfolio = session.run(return_cubes=False)
    assert after_portfolio.metadata["rebuild_counts"] == {"market": 2, "portfolio": 2, "config": 2}


def test_python_lgm_adapter_native_only_error_is_explicit():
    snapshot = _make_snapshot(include_unsupported=True)
    adapter = PythonLgmAdapter(fallback_to_swig=False)

    try:
        adapter.run(snapshot, mapped=XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs, run_id="native-only")
    except EngineRunError as exc:
        text = str(exc)
        assert "native-only mode" in text
        assert "supported only through the ORE SWIG fallback" in text
        assert "UNSUPPORTED_1:EquityOption" in text
    else:  # pragma: no cover
        raise AssertionError("Expected EngineRunError")


def test_python_lgm_adapter_swig_unavailable_error_is_explicit():
    snapshot = _make_snapshot(include_unsupported=True)
    adapter = PythonLgmAdapter(fallback_to_swig=True)
    mapped = XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs

    with patch("pythonore.runtime.runtime.ORESwigAdapter", side_effect=RuntimeError("swig missing")):
        try:
            adapter.run(snapshot, mapped=mapped, run_id="hybrid")
        except EngineRunError as exc:
            text = str(exc)
            assert "ORE SWIG fallback" in text
            assert "SWIG adapter unavailable" in text
            assert "UNSUPPORTED_1:EquityOption" in text
        else:  # pragma: no cover
            raise AssertionError("Expected EngineRunError")


def test_native_runtime_keeps_fixed_ore_grid_without_trade_date_augmentation():
    snapshot = _make_snapshot(
        params={},
    )
    cashflow_trade = Trade(
        trade_id="CF_OFFGRID",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Cashflow",
        product=GenericProduct(payload={"trade_type": "Cashflow", "xml": _cashflow_trade_xml(payment_date="2027-09-08", amount=1000.0)}),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(cashflow_trade,)),
        config=replace(
            snapshot.config,
            analytics=("CVA",),
            xml_buffers={
                "simulation.xml": _simulation_xml_with_grid("2,1Y"),
            },
        ),
    )
    result = XVAEngine.python_lgm_default(fallback_to_swig=False).create_session(snapshot).run(return_cubes=False)
    assert result.metadata["valuation_grid_size"] == 3
    assert result.metadata["grid_size"] >= 3


def test_python_lgm_default_is_native_only_by_default():
    engine = XVAEngine.python_lgm_default()
    assert engine.adapter.fallback_to_swig is False


def test_native_runtime_filters_past_cashflow_payments():
    snapshot = _make_snapshot()
    past_cashflow_trade = Trade(
        trade_id="CF_PAST",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Cashflow",
        product=GenericProduct(payload={"trade_type": "Cashflow", "xml": _cashflow_trade_xml(payment_date="2025-03-08", amount=1000.0)}),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(past_cashflow_trade,)),
        config=replace(snapshot.config, analytics=("CVA",)),
    )
    result = XVAEngine.python_lgm_default(fallback_to_swig=False).create_session(snapshot).run(return_cubes=False)
    assert abs(float(result.pv_total)) <= 1.0e-12


def test_native_runtime_emits_progress_logs(capsys):
    snapshot = _make_snapshot(params={"python.progress": "Y", "python.progress_bar": "N", "python.progress_log_interval": 1})
    XVAEngine.python_lgm_default(fallback_to_swig=False).create_session(snapshot).run(return_cubes=False)
    err = capsys.readouterr().err
    assert "extracting runtime inputs" in err
    assert "support classification:" in err
    assert "native pricing:" in err
    assert "run complete:" in err


def test_native_runtime_auto_lgm_calibration_does_not_fall_back_to_ore_subprocess():
    snapshot = _make_snapshot(params={"python.lgm_param_source": "auto"})
    adapter = PythonLgmAdapter(fallback_to_swig=False)
    adapter._ensure_py_lgm_imports()
    runtime_params = {
        "alpha_times": np.array([], dtype=float),
        "alpha_values": np.array([0.01], dtype=float),
        "kappa_times": np.array([], dtype=float),
        "kappa_values": np.array([0.03], dtype=float),
        "shift": 0.0,
        "scaling": 1.0,
    }
    snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            source_meta=SimpleNamespace(path="/tmp/fake/ore.xml"),
        ),
    )
    xml = {"simulation.xml": _simulation_xml_with_grid("4,6M")}
    fake_root = ET.fromstring(
        """
<ORE>
  <Setup>
    <Parameter name="marketDataFile">market.txt</Parameter>
    <Parameter name="curveConfigFile">curve.xml</Parameter>
    <Parameter name="conventionsFile">conv.xml</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
    <Parameter name="outputPath">Output</Parameter>
  </Setup>
  <Analytics>
    <Analytic type="simulation">
      <Parameter name="simulationConfigFile">simulation.xml</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""".strip()
    )
    with patch.object(adapter, "_is_ore_case_snapshot", return_value=True), patch(
        "pythonore.runtime.runtime.ET.parse",
        return_value=SimpleNamespace(getroot=lambda: fake_root),
    ), patch.object(
        adapter._ore_snapshot_mod,
        "_resolve_case_dirs",
        return_value=(Path("/tmp/fake/ore.xml"), fake_root, Path("/tmp/fake/Input")),
    ), patch.object(
        adapter._ore_snapshot_mod,
        "_resolve_ore_path",
        side_effect=lambda rel, base: Path(base) / str(rel or "missing"),
    ), patch.object(
        adapter._ore_snapshot_mod,
        "_resolve_output_ore_path",
        return_value=Path("/tmp/fake/Output"),
    ), patch.object(
        adapter._ore_snapshot_mod,
        "calibrate_lgm_params_in_python",
        return_value=runtime_params,
    ) as py_cal, patch.object(
        adapter._ore_snapshot_mod,
        "calibrate_lgm_params_via_ore",
        side_effect=AssertionError("ORE calibration subprocess should not run in auto mode"),
    ):
        params, source = adapter._parse_model_params(xml, "EUR", snapshot)
    assert source == "calibration"
    assert float(np.asarray(params["alpha_values"], dtype=float)[0]) == 0.01
    py_cal.assert_called()


def test_native_runtime_fails_fast_on_trade_nan():
    snapshot = _make_snapshot()
    adapter = PythonLgmAdapter(fallback_to_swig=False)
    mapped = XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs
    with patch.object(PythonLgmAdapter, "_price_fx_forward", return_value=np.full((2, 8), np.nan)):
        try:
            adapter.run(snapshot, mapped=mapped, run_id="nan-fast-fail")
        except EngineRunError as exc:
            assert "NaN detected in native trade pricing" in str(exc)
            assert "FXFWD_DEMO_1" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected EngineRunError")


def test_native_runtime_supports_generic_cashflow_capfloor_and_swaption():
    snapshot = _make_snapshot()
    trades = (
        Trade(
            trade_id="CF_NATIVE",
            counterparty="CP_A",
            netting_set="NS_EUR",
            trade_type="Cashflow",
            product=GenericProduct(payload={"trade_type": "Cashflow", "xml": _cashflow_trade_xml(payment_date="2027-03-08", amount=5000.0)}),
        ),
        Trade(
            trade_id="CAP_NATIVE",
            counterparty="CP_A",
            netting_set="NS_EUR",
            trade_type="CapFloor",
            product=GenericProduct(payload={"trade_type": "CapFloor", "xml": _generic_capfloor_trade_xml()}),
        ),
        Trade(
            trade_id="SWO_NATIVE",
            counterparty="CP_A",
            netting_set="NS_EUR",
            trade_type="Swaption",
            product=GenericProduct(payload={"trade_type": "Swaption", "xml": _generic_swaption_trade_xml()}),
        ),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=trades),
        config=replace(
            snapshot.config,
            analytics=("CVA",),
            xml_buffers={"simulation.xml": _simulation_xml_with_grid("4,6M")},
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    with patch.object(adapter._ir_options_mod, "capfloor_npv_paths", return_value=np.zeros((5, snapshot.config.num_paths))), patch.object(
        adapter._ir_options_mod,
        "bermudan_npv_paths",
        return_value=np.zeros((5, snapshot.config.num_paths)),
    ):
        result = adapter.run(snapshot, mapped=XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs, run_id="generic-native")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []


def test_native_runtime_keeps_in_arrears_capfloors_off_native_path():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="CAP_IN_ARREARS",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="CapFloor",
        product=GenericProduct(payload={"trade_type": "CapFloor", "xml": _generic_capfloor_in_arrears_trade_xml()}),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, analytics=("CVA",)),
    )
    support = classify_portfolio_support(snapshot, fallback_to_swig=False)
    assert support["native_trade_count"] == 0
    assert support["requires_swig_trade_ids"] == ["CAP_IN_ARREARS"]


def test_python_lgm_runtime_attaches_dim_reports_and_metadata():
    runtime = RuntimeConfig(
        xva_analytic=XVAAnalyticConfig(
            dim_model="DynamicIM",
            mva_enabled=True,
            dim_enabled=True,
        )
    )
    snapshot = _make_snapshot(
        runtime=runtime,
        params={"python.dim_feeder": _dynamic_im_feeder()},
    )

    result = XVAEngine.python_lgm_default(fallback_to_swig=False).create_session(snapshot).run()

    assert result.metadata["dim_mode"] == "DynamicIM"
    assert result.metadata["dim_engine"] == "python-dim"
    assert "NS_EUR" in result.metadata["dim_current"]
    assert "dim_evolution" in result.reports
    assert "dim_cube" in result.cubes
    assert result.metadata["support_classification"]["mode"] == "native_only"
    assert result.metadata["fallback_mode"] == "native_only"


def test_quote_matches_discount_curve_uses_unique_trade_family_as_fallback():
    quote_6m = "IR_SWAP/RATE/GBP/0D/6M/10Y"
    quote_1d = "IR_SWAP/RATE/GBP/0D/1D/10Y"

    assert _quote_matches_discount_curve(quote_6m, "GBP", "GBP", fallback_family="6M") is True
    assert _quote_matches_discount_curve(quote_1d, "GBP", "GBP", fallback_family="6M") is False
    assert _quote_matches_discount_curve(quote_1d, "GBP", "GBP") is True


def test_parse_swap_index_forward_tenors_caches_conventions_root():
    adapter = PythonLgmAdapter(fallback_to_swig=False)
    conventions_xml = """
    <Conventions>
      <Swap>
        <Id>EUR_SWAP_6M</Id>
        <Index>EUR-EURIBOR-6M</Index>
      </Swap>
      <SwapIndex>
        <Id>EUR-CMS-10Y</Id>
        <Conventions>EUR_SWAP_6M</Conventions>
      </SwapIndex>
    </Conventions>
    """.strip()

    with patch("pythonore.runtime.runtime.ET.fromstring", wraps=ET.fromstring) as fromstring:
        first = adapter._parse_swap_index_forward_tenors(conventions_xml)
        second = adapter._parse_swap_index_forward_tenors(conventions_xml)

    assert first == {"EUR-CMS-10Y": "6M"}
    assert second == first
    assert fromstring.call_count == 1
