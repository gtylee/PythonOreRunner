import pytest

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


# ---------------------------------------------------------------------------
# RED: from_dict must raise ValueError (not bare KeyError) on missing top-level keys
# ---------------------------------------------------------------------------


def test_snapshot_from_dict_raises_value_error_not_bare_key_error_when_config_missing():
    """from_dict must raise ValueError with a helpful message when 'config' is absent.

    Currently: ``config = data["config"]`` at line ~466 of dataclasses.py
    raises a bare ``KeyError: 'config'`` with no context about what went wrong
    or how to fix it.
    Fix: add a pre-check at the top of ``_snapshot_from_dict`` that raises
    ``ValueError(f"Snapshot dict missing required key: 'config'")`` for each
    of the four required top-level keys.
    """
    with pytest.raises(ValueError, match="(?i)config|missing|required"):
        XVASnapshot.from_dict(
            {
                "market": {"asof": "2026-01-01"},
                "fixings": {},
                "portfolio": {"trades": []},
                # "config" key is deliberately absent
            }
        )


def test_snapshot_from_dict_raises_value_error_not_bare_key_error_when_market_missing():
    """from_dict must raise ValueError with a helpful message when 'market' is absent."""
    with pytest.raises(ValueError, match="(?i)market|missing|required"):
        XVASnapshot.from_dict(
            {
                # "market" key is deliberately absent
                "fixings": {},
                "portfolio": {"trades": []},
                "config": {"asof": "2026-01-01", "base_currency": "EUR"},
            }
        )
