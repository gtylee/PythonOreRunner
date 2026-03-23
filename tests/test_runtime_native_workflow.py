from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

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
