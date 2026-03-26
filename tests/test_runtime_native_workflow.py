from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from pythonore.domain.dataclasses import (
    BermudanSwaption,
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
from pythonore.io.loader import XVALoader
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.runtime import (
    DeterministicToyAdapter,
    PythonLgmAdapter,
    XVAEngine,
    _parse_market_overlay,
    _quote_matches_discount_curve,
    _quote_matches_forward_curve,
    classify_portfolio_support,
)
from pythonore.mapping.mapper import map_snapshot

TOOLS_DIR = Path(__file__).resolve().parents[1]


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


def _generic_digital_cmsspread_trade_xml() -> str:
    return """
<SwapData>
  <LegData>
    <LegType>DigitalCMSSpread</LegType>
    <Payer>false</Payer>
    <Currency>USD</Currency>
    <PaymentConvention>F</PaymentConvention>
    <DayCounter>30/360</DayCounter>
    <Notionals><Notional>1000000</Notional></Notionals>
    <DigitalCMSSpreadLegData>
      <CMSSpreadLegData>
        <Index1>USD-CMS-30Y</Index1>
        <Index2>USD-CMS-2Y</Index2>
        <IsInArrears>false</IsInArrears>
        <FixingDays>2</FixingDays>
        <Gearings><Gearing>2.0</Gearing></Gearings>
        <Spreads><Spread>0.001</Spread></Spreads>
      </CMSSpreadLegData>
      <CallPosition>Long</CallPosition>
      <IsCallATMIncluded>false</IsCallATMIncluded>
      <CallStrikes><Strike>0.002</Strike></CallStrikes>
      <CallPayoffs><Payoff>0.002</Payoff></CallPayoffs>
      <PutPosition>Long</PutPosition>
      <IsPutATMIncluded>false</IsPutATMIncluded>
      <PutStrikes><Strike>0.0005</Strike></PutStrikes>
      <PutPayoffs><Payoff>0.001</Payoff></PutPayoffs>
    </DigitalCMSSpreadLegData>
    <ScheduleData>
      <Rules>
        <StartDate>2026-03-08</StartDate>
        <EndDate>2031-03-08</EndDate>
        <Tenor>6M</Tenor>
        <Calendar>US</Calendar>
        <Convention>F</Convention>
      </Rules>
    </ScheduleData>
  </LegData>
  <LegData>
    <LegType>Floating</LegType>
    <Payer>true</Payer>
    <Currency>USD</Currency>
    <PaymentConvention>F</PaymentConvention>
    <DayCounter>A360</DayCounter>
    <Notionals><Notional>1000000</Notional></Notionals>
    <FloatingLegData>
      <Index>USD-LIBOR-6M</Index>
      <FixingDays>2</FixingDays>
      <IsInArrears>false</IsInArrears>
      <Spreads><Spread>0.002</Spread></Spreads>
    </FloatingLegData>
    <ScheduleData>
      <Rules>
        <StartDate>2026-03-08</StartDate>
        <EndDate>2031-03-08</EndDate>
        <Tenor>6M</Tenor>
        <Calendar>US</Calendar>
        <Convention>F</Convention>
      </Rules>
    </ScheduleData>
  </LegData>
</SwapData>
""".strip()


def _generic_rate_swap_trade_xml(*, start_date: str = "2024-03-08", end_date: str = "2028-03-08") -> str:
    return f"""
<SwapData>
  <LegData>
    <LegType>Fixed</LegType>
    <Currency>EUR</Currency>
    <Payer>true</Payer>
    <PaymentConvention>F</PaymentConvention>
    <DayCounter>30/360</DayCounter>
    <Notionals><Notional>1000000</Notional></Notionals>
    <ScheduleData>
      <Rules>
        <StartDate>{start_date}</StartDate>
        <EndDate>{end_date}</EndDate>
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
        <StartDate>{start_date}</StartDate>
        <EndDate>{end_date}</EndDate>
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
</SwapData>
""".strip()


def _simulation_xml_with_usd_grid(spec: str) -> str:
    return f"""
<Simulation>
  <Parameters><Grid>{spec}</Grid></Parameters>
  <CrossAssetModel>
    <InterestRateModels>
      <LGM ccy="USD">
        <Reversion><Value>0.03</Value></Reversion>
        <Volatility><Value>0.01</Value></Volatility>
        <ParameterTransformation><ShiftHorizon>0</ShiftHorizon><Scaling>1</Scaling></ParameterTransformation>
      </LGM>
    </InterestRateModels>
  </CrossAssetModel>
</Simulation>
""".strip()


def _generic_bermudan_swaption_trade_xml() -> str:
    return """
<SwaptionData>
  <OptionData>
    <Style>Bermudan</Style>
    <Settlement>Physical</Settlement>
    <LongShort>Long</LongShort>
    <ExerciseDates>
      <ExerciseDate>2026-09-08</ExerciseDate>
      <ExerciseDate>2027-03-08</ExerciseDate>
    </ExerciseDates>
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
        <EndDate>2029-09-08</EndDate>
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
        <EndDate>2029-09-08</EndDate>
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


def _generic_live_capfloor_with_past_coupon_xml() -> str:
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


def _generic_amortizing_capfloor_trade_xml() -> str:
    return """
<CapFloorData>
  <LongShort>Long</LongShort>
  <Caps><Cap>0.03</Cap></Caps>
  <LegData>
    <LegType>Floating</LegType>
    <Currency>GBP</Currency>
    <PaymentConvention>MF</PaymentConvention>
    <DayCounter>ACT/365</DayCounter>
    <Notionals>
      <Notional>3000000</Notional>
      <Notional>2900000</Notional>
      <Notional>2800000</Notional>
    </Notionals>
    <ScheduleData>
      <Rules>
        <StartDate>2026-03-08</StartDate>
        <EndDate>2027-09-08</EndDate>
        <Tenor>6M</Tenor>
        <Calendar>UK</Calendar>
        <Convention>MF</Convention>
      </Rules>
    </ScheduleData>
    <FloatingLegData>
      <Index>GBP-LIBOR-6M</Index>
      <FixingDays>0</FixingDays>
      <IsInArrears>false</IsInArrears>
      <Spreads><Spread>0.0</Spread></Spreads>
      <Gearings><Gearing>1.0</Gearing></Gearings>
    </FloatingLegData>
  </LegData>
</CapFloorData>
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


def test_generic_cashflow_state_returns_empty_live_flows_for_fully_past_payment():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="CF_PAST_DIRECT",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Cashflow",
        product=GenericProduct(payload={"trade_type": "Cashflow", "xml": _cashflow_trade_xml(payment_date="2025-03-08", amount=1000.0)}),
    )
    adapter = PythonLgmAdapter(fallback_to_swig=False)
    adapter._ensure_py_lgm_imports()

    state = adapter._build_generic_cashflow_state(trade, snapshot)

    assert state is not None
    assert np.asarray(state["pay_time"], dtype=float).size == 0
    assert np.asarray(state["amount"], dtype=float).size == 0


def test_generic_rate_swap_builder_drops_fully_past_coupons():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="GENERIC_SWAP_FILTER",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Swap",
        product=GenericProduct(payload={"trade_type": "Swap", "xml": _generic_rate_swap_trade_xml()}),
    )
    adapter = PythonLgmAdapter(fallback_to_swig=False)
    adapter._ensure_py_lgm_imports()

    state = adapter._build_generic_rate_swap_legs(trade, snapshot)

    assert state is not None
    assert state["rate_legs"]
    for leg in state["rate_legs"]:
        pay = np.asarray(leg["pay_time"], dtype=float)
        assert pay.size > 0
        assert np.all(pay >= -1.0e-12)


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


def test_native_runtime_handles_usd_digital_cmsspread_without_nan():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="DIGITAL_CMSSPREAD_NATIVE",
        counterparty="CP_A",
        netting_set="NS_USD",
        trade_type="Swap",
        product=GenericProduct(payload={"trade_type": "Swap", "xml": _generic_digital_cmsspread_trade_xml()}),
    )
    snapshot = replace(
        snapshot,
        market=replace(
            snapshot.market,
            raw_quotes=tuple(snapshot.market.raw_quotes)
            + (
                MarketQuote(date="2026-03-08", key="ZERO/RATE/USD/30Y", value=0.0315),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/USD/USD-LIBOR-6M/1Y/30Y", value=0.0330),
            ),
        ),
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        netting=replace(
            snapshot.netting,
            netting_sets={
                "NS_USD": NettingSet(
                    netting_set_id="NS_USD",
                    counterparty="CP_A",
                    active_csa=True,
                    csa_currency="USD",
                )
            },
        ),
        collateral=replace(
            snapshot.collateral,
            balances=(CollateralBalance(netting_set_id="NS_USD", currency="USD"),),
        ),
        config=replace(
            snapshot.config,
            base_currency="USD",
            analytics=("CVA",),
            xml_buffers={"simulation.xml": _simulation_xml_with_usd_grid("4,6M")},
        ),
    )
    result = XVAEngine.python_lgm_default(fallback_to_swig=False).create_session(snapshot).run(return_cubes=False)
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert np.isfinite(float(result.pv_total))


def test_torch_generic_capfloor_matches_numpy_runtime():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="CAP_TORCH_PARITY",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="CapFloor",
        product=GenericProduct(payload={"trade_type": "CapFloor", "xml": _generic_capfloor_trade_xml()}),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            analytics=("CVA",),
            xml_buffers={"simulation.xml": _simulation_xml_with_grid("4,6M")},
        ),
    )
    mapped = XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs

    numpy_adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    with patch.object(numpy_adapter, "_resolve_irs_pricing_backend", return_value=None):
        numpy_result = numpy_adapter.run(snapshot, mapped=mapped, run_id="capfloor-numpy")

    torch_adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    from pythonore.compute.lgm_torch_xva import (
        TorchDiscountCurve,
        capfloor_npv_paths_torch,
        deflate_lgm_npv_paths_torch_batched,
        par_swap_rate_paths_torch,
        price_plain_rate_leg_paths_torch,
        swap_npv_paths_from_ore_legs_dual_curve_torch,
    )

    backend = (
        TorchDiscountCurve,
        swap_npv_paths_from_ore_legs_dual_curve_torch,
        deflate_lgm_npv_paths_torch_batched,
        "cpu",
        price_plain_rate_leg_paths_torch,
        par_swap_rate_paths_torch,
        capfloor_npv_paths_torch,
    )
    with patch.object(torch_adapter, "_resolve_irs_pricing_backend", return_value=backend):
        torch_result = torch_adapter.run(snapshot, mapped=mapped, run_id="capfloor-torch")

    assert abs(float(torch_result.pv_total) - float(numpy_result.pv_total)) < 1.0e-8
    assert abs(float(torch_result.xva_by_metric.get("CVA", 0.0)) - float(numpy_result.xva_by_metric.get("CVA", 0.0))) < 1.0e-8


def test_native_runtime_supports_in_arrears_capfloors():
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
    assert support["native_trade_count"] == 1
    assert support["requires_swig_trade_ids"] == []

    result = XVAEngine.python_lgm_default(fallback_to_swig=False).create_session(snapshot).run(return_cubes=False)
    assert np.isfinite(float(result.pv_total))


def test_native_runtime_torch_bermudan_parity():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="BERM_TORCH_PARITY",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Swaption",
        product=GenericProduct(payload={"trade_type": "Swaption", "xml": _generic_bermudan_swaption_trade_xml()}),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            analytics=("CVA",),
            xml_buffers={"simulation.xml": _simulation_xml_with_grid("4,6M")},
        ),
    )
    mapped = XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs

    numpy_adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    with patch.object(numpy_adapter, "_resolve_irs_pricing_backend", return_value=None):
        numpy_result = numpy_adapter.run(snapshot, mapped=mapped, run_id="bermudan-numpy")

    torch_adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    from pythonore.compute.lgm_torch_xva import (
        TorchDiscountCurve,
        capfloor_npv_paths_torch,
        deflate_lgm_npv_paths_torch_batched,
        par_swap_rate_paths_torch,
        price_plain_rate_leg_paths_torch,
        swap_npv_paths_from_ore_legs_dual_curve_torch,
    )

    backend = (
        TorchDiscountCurve,
        swap_npv_paths_from_ore_legs_dual_curve_torch,
        deflate_lgm_npv_paths_torch_batched,
        "cpu",
        price_plain_rate_leg_paths_torch,
        par_swap_rate_paths_torch,
        capfloor_npv_paths_torch,
    )
    with patch.object(torch_adapter, "_resolve_irs_pricing_backend", return_value=backend):
        torch_result = torch_adapter.run(snapshot, mapped=mapped, run_id="bermudan-torch")

    assert abs(float(torch_result.pv_total) - float(numpy_result.pv_total)) < 1.0e-8
    assert abs(float(torch_result.xva_by_metric.get("CVA", 0.0)) - float(numpy_result.xva_by_metric.get("CVA", 0.0))) < 1.0e-8


def test_parse_market_overlay_accepts_named_dated_zero_quotes():
    overlay = _parse_market_overlay(
        (
            MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/EUR-EURIBOR-6M/A365/2037-06-15", value=0.02),
            MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/2037-06-15", value=0.021),
        ),
        asof_date="2026-03-08",
    )

    named = overlay["named_zero"]["EUR-EURIBOR-6M"]
    unnamed = overlay["zero"]["EUR"]
    assert len(named) == 1
    assert len(unnamed) == 1
    assert named[0][0] > 11.0
    assert abs(named[0][1] - 0.02) < 1.0e-12
    assert unnamed[0][0] > 11.0


def test_loader_from_ore_xml_matches_from_files():
    ore_xml = TOOLS_DIR / "Examples" / "Legacy" / "Example_6" / "Input" / "ore_portfolio_2.xml"
    from_xml = XVALoader.from_ore_xml(ore_xml)
    from_dir = XVALoader.from_files(str(ore_xml.parent), ore_file=ore_xml.name)

    assert from_xml.config.source_meta.path == str(ore_xml)
    assert from_dir.config.source_meta.path == str(ore_xml)
    assert "<ORE>" in from_xml.config.xml_buffers["ore.xml"]
    assert "<ORE>" in from_dir.config.xml_buffers["ore.xml"]
    assert len(from_xml.portfolio.trades) == len(from_dir.portfolio.trades)
    assert tuple(t.trade_id for t in from_xml.portfolio.trades) == tuple(t.trade_id for t in from_dir.portfolio.trades)


def test_live_capfloor_with_past_coupon_stays_native_and_filters_paid_cashflows():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="CAP_LIVE_PAST_COUPON",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="CapFloor",
        product=GenericProduct(payload={"trade_type": "CapFloor", "xml": _generic_live_capfloor_with_past_coupon_xml()}),
    )
    snapshot = replace(
        snapshot,
        market=replace(snapshot.market, asof="2026-04-15"),
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, asof="2026-04-15", analytics=("CVA",)),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    state = adapter._build_generic_capfloor_state(trade, snapshot)

    assert state is not None
    definition = state["definition"]
    assert np.all(np.asarray(definition.pay_time, dtype=float) >= -1.0e-12)
    assert np.min(np.asarray(definition.start_time, dtype=float)) < 0.0

    support = classify_portfolio_support(snapshot, fallback_to_swig=False)
    assert support["native_trade_count"] == 1
    assert support["requires_swig_trade_ids"] == []


def test_generic_capfloor_uses_period_notionals_from_xml_schedule():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="CAP_AMORT_XML",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="CapFloor",
        product=GenericProduct(payload={"trade_type": "CapFloor", "xml": _generic_amortizing_capfloor_trade_xml()}),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    state = adapter._build_generic_capfloor_state(trade, snapshot)

    assert state is not None
    np.testing.assert_allclose(np.asarray(state["definition"].notional, dtype=float), np.array([3_000_000.0, 2_900_000.0, 2_800_000.0]))


def test_native_runtime_supports_generic_bermudan_swaption():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="BERM_GENERIC",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Swaption",
        product=GenericProduct(payload={"trade_type": "Swaption", "xml": _generic_bermudan_swaption_trade_xml()}),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, analytics=("CVA",), xml_buffers={"simulation.xml": _simulation_xml_with_grid("4,6M")}),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    with patch.object(adapter._ir_options_mod, "bermudan_npv_paths", return_value=np.zeros((5, snapshot.config.num_paths))):
        result = adapter.run(snapshot, mapped=XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs, run_id="berm-generic")

    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []


def test_native_runtime_supports_dataclass_bermudan_swaption():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="BERM_DATACLASS",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Swaption",
        product=BermudanSwaption(
            ccy="EUR",
            notional=1_000_000.0,
            fixed_rate=0.025,
            maturity_years=3.0,
            pay_fixed=True,
            exercise_dates=("2026-09-08", "2027-03-08"),
            start_date="2026-09-08",
            end_date="2029-09-08",
            float_index="EUR-EURIBOR-6M",
        ),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, analytics=("CVA",), xml_buffers={"simulation.xml": _simulation_xml_with_grid("4,6M")}),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    with patch.object(adapter._ir_options_mod, "bermudan_npv_paths", return_value=np.zeros((5, snapshot.config.num_paths))):
        result = adapter.run(snapshot, mapped=XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).state.mapped_inputs, run_id="berm-dataclass")

    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []


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


def test_quote_matches_forward_curve_accepts_zero_quotes_for_zero_tenor_family():
    assert _quote_matches_forward_curve("ZERO/RATE/EUR/1Y", "EUR", "1D") is True
    assert _quote_matches_forward_curve("ZERO/RATE/EUR/EUR-ESTR/A365/2027-03-08", "EUR", "ON") is True
    assert _quote_matches_forward_curve("IR_SWAP/RATE/EUR/2D/1D/10Y", "EUR", "0D") is True
    assert _quote_matches_forward_curve("IR_SWAP/RATE/EUR/2D/6M/10Y", "EUR", "ON") is False


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


def test_generic_bermudan_runtime_drops_past_exercise_dates():
    snapshot = _make_snapshot()
    trade = Trade(
        trade_id="BERM_PAST_EX",
        counterparty="CP_A",
        netting_set="NS_EUR",
        trade_type="Swaption",
        product=GenericProduct(
            payload={
                "trade_type": "Swaption",
                "xml": _generic_bermudan_swaption_trade_xml().replace(
                    "<ExerciseDate>2026-09-08</ExerciseDate>",
                    "<ExerciseDate>2025-09-08</ExerciseDate>",
                    1,
                ),
            }
        ),
    )
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, analytics=("CVA",), xml_buffers={"simulation.xml": _simulation_xml_with_grid("4,6M")}),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    fake_legs = {
        "float_index": "EUR-EURIBOR-6M",
        "fixed_notional": np.array([1_000_000.0], dtype=float),
    }
    with patch.object(adapter._irs_utils, "load_swap_legs_from_portfolio_root", return_value=fake_legs):
        state = adapter._build_generic_swaption_state(trade, snapshot)
    assert state is not None
    ex = np.asarray(state["definition"].exercise_times, dtype=float)
    assert ex.size == 1
    assert np.all(ex >= 0.0)


def test_torch_swap_pricer_handles_negative_schedule_times():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from pythonore.compute.lgm import LGM1F, LGMParams
    from pythonore.compute.lgm_torch_xva import (
        TorchDiscountCurve,
        deflate_lgm_npv_paths_torch_batched,
        swap_npv_paths_from_ore_legs_dual_curve_torch,
    )

    model = LGM1F(LGMParams.constant(alpha=0.01, kappa=0.03))
    curve = TorchDiscountCurve(
        times=np.array([0.0, 1.0, 5.0], dtype=float),
        dfs=np.array([1.0, 0.98, 0.90], dtype=float),
        device="cpu",
        dtype=torch.float64,
    )
    legs = {
        "fixed_pay_time": np.array([-0.25, 1.0], dtype=float),
        "fixed_amount": np.array([100.0, 100.0], dtype=float),
        "fixed_start_time": np.array([-0.75, 0.5], dtype=float),
        "float_pay_time": np.array([0.5, 1.0], dtype=float),
        "float_start_time": np.array([-0.5, 0.5], dtype=float),
        "float_end_time": np.array([0.5, 1.0], dtype=float),
        "float_fixing_time": np.array([-0.5, 0.5], dtype=float),
        "float_accrual": np.array([0.5, 0.5], dtype=float),
        "float_index_accrual": np.array([0.5, 0.5], dtype=float),
        "float_notional": np.array([1_000_000.0, 1_000_000.0], dtype=float),
        "float_sign": np.array([1.0, 1.0], dtype=float),
        "float_spread": np.array([np.nan, 0.001], dtype=float),
        "float_coupon": np.array([0.0, 0.0], dtype=float),
    }
    times = np.array([-0.25, 0.0, 0.5, 1.0], dtype=float)
    x_paths = np.zeros((times.size, 4), dtype=float)
    npv = swap_npv_paths_from_ore_legs_dual_curve_torch(
        model,
        curve,
        curve,
        legs,
        times,
        x_paths,
        return_numpy=True,
    )
    deflated = deflate_lgm_npv_paths_torch_batched(
        model,
        curve,
        times,
        x_paths,
        npv,
        return_numpy=True,
    )
    assert np.all(np.isfinite(npv))
    assert np.all(np.isfinite(deflated))
