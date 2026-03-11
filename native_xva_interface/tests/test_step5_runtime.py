from native_xva_interface import FXForward, MarketData, MarketQuote, Portfolio, Trade, XVAConfig, XVASnapshot, FixingsData, XVAEngine


def _snapshot() -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(asof="2026-03-08"),
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


def test_runtime_baseline_and_incremental_updates():
    engine = XVAEngine()
    session = engine.create_session(_snapshot())
    r0 = session.run()
    assert "CVA" in r0.xva_by_metric

    bumped_market = MarketData(
        asof="2026-03-08",
        raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.12),),
    )
    session.update_market(bumped_market)
    r1 = session.run_incremental()
    assert r1.metadata["rebuild_counts"]["market"] >= 2

    session.update_portfolio(add=[])
    r2 = session.run_incremental()
    assert r2.metadata["rebuild_counts"]["portfolio"] >= 1


def test_runtime_update_config_invalidates_and_rebuilds():
    engine = XVAEngine()
    session = engine.create_session(_snapshot())
    before = session.state.snapshot_key

    session.update_config(num_paths=10, analytics=("CVA",))
    after = session.state.snapshot_key

    assert before != after
    assert session.state.snapshot.config.num_paths == 10
    assert session.state.snapshot.config.analytics == ("CVA",)
    assert session.state.rebuild_counts["config"] >= 2


def test_prepare_sensitivity_snapshot_applies_runtime_knobs():
    engine = XVAEngine()
    prepared = engine.prepare_sensitivity_snapshot(
        _snapshot(),
        curve_fit_mode="ore_fit",
        use_ore_output_curves=False,
        curve_node_shocks={"discount": {"EUR": {"node_times": [5.0], "node_shifts": [1.0e-4]}}},
    )

    assert prepared.config.params["python.curve_fit_mode"] == "ore_fit"
    assert prepared.config.params["python.use_ore_output_curves"] == "N"
    assert prepared.config.params["python.curve_node_shocks"]["discount"]["EUR"]["node_times"] == [5.0]


def test_prepare_sensitivity_snapshot_keeps_explicit_credit_survival_curves():
    engine = XVAEngine()
    prepared = engine.prepare_sensitivity_snapshot(
        _snapshot(),
        curve_fit_mode="ore_fit",
        use_ore_output_curves=False,
    )
    prepared.config.params["python.credit_survival_curves"] = {
        "CP_A": {
            "node_times": [1.0, 5.0],
            "survival_probabilities": [0.98, 0.90],
            "extrapolation": "flat_zero",
        }
    }

    again = engine.prepare_sensitivity_snapshot(
        prepared,
        curve_fit_mode="ore_fit",
        use_ore_output_curves=False,
    )

    assert again.config.params["python.credit_survival_curves"]["CP_A"]["node_times"] == [1.0, 5.0]
