from native_xva_interface import FXForward, IRS, MarketData, MarketQuote, Portfolio, Trade, XVAConfig, XVASnapshot, FixingsData


def test_dataclass_construction_and_validation():
    market = MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.1),))
    portfolio = Portfolio(
        trades=(
            Trade(
                trade_id="T1",
                counterparty="CP_A",
                netting_set="NS1",
                trade_type="Swap",
                product=IRS(ccy="USD", notional=1_000_000, fixed_rate=0.03, maturity_years=5.0),
            ),
        )
    )
    cfg = XVAConfig(asof="2026-03-08", base_currency="USD")
    snap = XVASnapshot(market=market, fixings=FixingsData(), portfolio=portfolio, config=cfg)
    assert snap.portfolio.trades[0].trade_id == "T1"


def test_round_trip_and_stable_key():
    snap = XVASnapshot(
        market=MarketData(asof="2026-03-08"),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T2",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.1, maturity_years=1.0),
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
    )
    round_trip = XVASnapshot.from_dict(snap.to_dict())
    assert round_trip.to_dict() == snap.to_dict()
    assert round_trip.stable_key() == snap.stable_key()
