import pytest

from native_xva_interface import (
    FXForward,
    MarketData,
    MarketQuote,
    Portfolio,
    Trade,
    XVAConfig,
    XVASnapshot,
    FixingsData,
    ConflictError,
    merge_snapshots,
)


def _base() -> XVASnapshot:
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
        config=XVAConfig(asof="2026-03-08", base_currency="EUR"),
    )


def test_merge_override_conflict_resolution():
    base = _base()
    ov = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.2),)),
        fixings=FixingsData(),
        portfolio=base.portfolio,
        config=base.config,
    )
    merged = merge_snapshots(base, ov, on_conflict="override")
    assert merged.market.raw_quotes[0].value == 1.2


def test_merge_strict_conflict_error():
    base = _base()
    ov = XVASnapshot(
        market=MarketData(asof="2026-03-08", raw_quotes=(MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.2),)),
        fixings=FixingsData(),
        portfolio=base.portfolio,
        config=base.config,
    )
    with pytest.raises(ConflictError):
        merge_snapshots(base, ov, on_conflict="error")
