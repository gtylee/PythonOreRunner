"""Regression tests for silent-failure and edge-case paths in loader.py / mapper.py.

All tests marked RED below assert behaviour that does NOT yet exist and will fail
until the corresponding fix is applied.  Tests marked GREEN document existing
behaviour that must not regress.
"""
from __future__ import annotations

import warnings

import pytest

from native_xva_interface import (
    FixingsData,
    MarketData,
    MappingError,
    MporConfig,
    NettingConfig,
    Portfolio,
    XVAConfig,
    XVASnapshot,
    map_snapshot,
)
from native_xva_interface.loader import _resolve_snapshot_mpor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_snapshot(**overrides) -> XVASnapshot:
    """Return the smallest valid snapshot (empty portfolio, no market data)."""
    defaults: dict = dict(
        market=MarketData(asof="2026-01-01"),
        fixings=FixingsData(),
        portfolio=Portfolio(trades=()),
        config=XVAConfig(asof="2026-01-01", base_currency="EUR"),
    )
    defaults.update(overrides)
    return XVASnapshot(**defaults)


# ---------------------------------------------------------------------------
# RED: _resolve_snapshot_mpor must warn when simulation.xml is malformed XML
# ---------------------------------------------------------------------------


def test_resolve_snapshot_mpor_warns_on_malformed_simulation_xml():
    """A corrupt simulation.xml buffer must emit a UserWarning, not silently fall through.

    Currently: ``except Exception: pass`` at line ~676 of loader.py swallows
    the ET.fromstring error with no diagnostic output.
    Fix: replace with ``except Exception as exc: warnings.warn(...)``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_snapshot_mpor(
            base_currency="EUR",
            params={},
            xml_buffers={"simulation.xml": "<<THIS IS NOT VALID XML AT ALL>>"},
            netting=NettingConfig(),
        )

    # Must still return a valid disabled MporConfig — no crash
    assert isinstance(result, MporConfig)
    assert result.enabled is False

    warning_messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "simulation.xml" in m.lower() or "parse" in m.lower() or "malformed" in m.lower()
        for m in warning_messages
    ), (
        "Expected a UserWarning mentioning 'simulation.xml' or 'parse', "
        f"got warnings: {warning_messages}"
    )


# ---------------------------------------------------------------------------
# RED: _resolve_snapshot_mpor must warn when python.mpor_days is not numeric
# ---------------------------------------------------------------------------


def test_resolve_snapshot_mpor_warns_on_unparseable_mpor_days():
    """An unparseable python.mpor_days param must emit a UserWarning, not silently do nothing.

    Currently: ``except Exception: pass`` at line ~695 of loader.py swallows
    the int(float(...)) conversion error so the caller has no way to detect
    that their override was ignored.
    Fix: replace with ``except Exception as exc: warnings.warn(...)``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_snapshot_mpor(
            base_currency="EUR",
            params={"python.mpor_days": "not-a-number"},
            xml_buffers={},
            netting=NettingConfig(),
        )

    # A malformed override must not silently enable MPOR
    assert isinstance(result, MporConfig)
    assert result.enabled is False, (
        "Malformed python.mpor_days must not enable MPOR; "
        f"got mpor_years={result.mpor_years}, enabled={result.enabled}"
    )

    warning_messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "mpor_days" in m.lower() or "mpor" in m.lower() or "parse" in m.lower()
        for m in warning_messages
    ), (
        "Expected a UserWarning mentioning 'mpor_days' or 'parse', "
        f"got warnings: {warning_messages}"
    )


# ---------------------------------------------------------------------------
# GREEN: map_snapshot must raise MappingError immediately for an empty portfolio
# ---------------------------------------------------------------------------


def test_map_snapshot_raises_mapping_error_on_empty_portfolio():
    """map_snapshot must raise MappingError immediately when the portfolio has no trades.

    This is existing behaviour (mapper.py line ~58-63) that must not regress.
    """
    snap = _minimal_snapshot()
    with pytest.raises(MappingError, match="(?i)empty"):
        map_snapshot(snap)


# ---------------------------------------------------------------------------
# RED: map_snapshot must warn when no market data is present in the snapshot
# ---------------------------------------------------------------------------


def test_map_snapshot_warns_when_snapshot_has_no_market_quotes():
    """A snapshot with an empty raw_quotes tuple should emit a UserWarning.

    Without any market data the XVA engine will produce meaningless zeros with
    no indication to the caller that something is wrong.

    Currently: the empty-quotes path is silently accepted with no diagnostic.
    Fix: emit ``warnings.warn("Snapshot contains no market quotes ...")`` in
    map_snapshot (or _extract_inputs) when raw_quotes is empty.
    """
    from native_xva_interface import IRS, Trade

    snap = XVASnapshot(
        market=MarketData(asof="2026-01-01", raw_quotes=()),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="T1",
                    counterparty="CP_A",
                    netting_set="NS1",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.03, maturity_years=5.0),
                ),
            )
        ),
        config=XVAConfig(asof="2026-01-01", base_currency="EUR"),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        map_snapshot(snap)  # must not raise

    warning_messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "market" in m.lower() or "quote" in m.lower() or "empty" in m.lower()
        for m in warning_messages
    ), (
        "Expected a UserWarning about missing market data; "
        f"got warnings: {warning_messages}"
    )
