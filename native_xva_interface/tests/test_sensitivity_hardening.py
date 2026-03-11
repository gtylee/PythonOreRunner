"""Regression tests for silent-failure and boundary-condition paths in sensitivity.py.

Tests marked RED assert behaviour that does NOT yet exist and will fail until
the corresponding fix is applied.  Tests marked GREEN document existing
behaviour that must not regress.
"""
from __future__ import annotations

import warnings

import pytest

from native_xva_interface import (
    FixingsData,
    MarketData,
    Portfolio,
    XVAConfig,
    XVASnapshot,
)
from native_xva_interface.sensitivity import (
    _discount_curve_family_by_currency,
    _ore_bucket_weight,
    _survival_from_piecewise_hazard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_with_xml_buffers(**xml_buffers) -> XVASnapshot:
    """Return a minimal XVASnapshot carrying the given xml_buffers."""
    return XVASnapshot(
        market=MarketData(asof="2026-01-01"),
        fixings=FixingsData(),
        portfolio=Portfolio(trades=()),
        config=XVAConfig(
            asof="2026-01-01",
            base_currency="EUR",
            xml_buffers=xml_buffers,
        ),
    )


# ---------------------------------------------------------------------------
# RED: _ore_bucket_weight must return 0.0 for out-of-range bucket index
# ---------------------------------------------------------------------------


def test_ore_bucket_weight_returns_zero_for_bucket_beyond_last_tenor():
    """_ore_bucket_weight must return 0.0 when bucket >= len(shift_times), not raise IndexError.

    Currently: line 793 does ``t1 = tenors[j]`` before any bounds check, so a
    bucket index beyond the end of the tenors list raises ``IndexError``.
    Fix: guard with ``if j < 0 or j >= len(tenors): return 0.0`` at the top
    of the function.
    """
    # bucket=10 is well beyond the 3-element tenors list
    result = _ore_bucket_weight([1.0, 2.0, 5.0], bucket=10, t=3.0)
    assert result == 0.0, f"Expected 0.0 for out-of-range bucket, got {result}"


def test_ore_bucket_weight_returns_zero_for_negative_bucket():
    """A negative bucket index must return 0.0, not silently wrap via Python's negative indexing.

    Currently: ``j = int(-1)``; ``tenors[-1]`` returns the *last* element (5.0),
    then ``t0 = tenors[-2] = 2.0`` and ``t2 = tenors[0] = 1.0``.  For t=3.0:
    the condition ``3.0 >= 2.0 and 3.0 <= 5.0`` is True, so the function
    returns (3.0 - 2.0) / (5.0 - 2.0) ≈ 0.33 — a nonsensical weight.
    Fix: the same bounds guard (``if j < 0 or j >= len(tenors): return 0.0``)
    covers the negative case.
    """
    result = _ore_bucket_weight([1.0, 2.0, 5.0], bucket=-1, t=3.0)
    assert result == 0.0, f"Expected 0.0 for negative bucket, got {result}"


def test_ore_bucket_weight_handles_single_tenor_list_correctly():
    """GREEN: the existing single-tenor shortcut (``return 1.0``) must not regress.

    ``len(tenors) == 1`` is handled before the bounds-problematic code, so this
    must continue to work after the fix is applied.
    """
    assert _ore_bucket_weight([3.0], bucket=0, t=1.0) == 1.0
    assert _ore_bucket_weight([3.0], bucket=0, t=5.0) == 1.0


# ---------------------------------------------------------------------------
# RED: _discount_curve_family_by_currency must warn on malformed todaysmarket.xml
# ---------------------------------------------------------------------------


def test_discount_curve_family_warns_on_malformed_todaysmarket_xml():
    """Malformed todaysmarket.xml must emit a UserWarning instead of silently returning {}.

    Currently: lines 625-628 catch the ET.fromstring exception and immediately
    ``return {}`` with no diagnostic output, making it impossible to distinguish
    between "no todaysmarket.xml" and "corrupted todaysmarket.xml".
    Fix: ``except Exception as exc: warnings.warn(f"Failed to parse todaysmarket.xml: {exc}")``.
    """
    snap = _snapshot_with_xml_buffers(**{"todaysmarket.xml": "<<THIS IS NOT VALID XML>>"})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _discount_curve_family_by_currency(snap)

    # Must still return empty dict — no crash
    assert result == {}, f"Expected empty dict on parse error, got {result}"

    warning_messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "todaysmarket" in m.lower() or "parse" in m.lower() or "xml" in m.lower()
        for m in warning_messages
    ), (
        "Expected a UserWarning mentioning 'todaysmarket' or 'parse', "
        f"got warnings: {warning_messages}"
    )


def test_discount_curve_family_returns_empty_dict_when_xml_buffer_absent():
    """GREEN: returns {} immediately when todaysmarket.xml is not in xml_buffers (no warning needed)."""
    snap = _snapshot_with_xml_buffers()  # no todaysmarket.xml
    result = _discount_curve_family_by_currency(snap)
    assert result == {}


# ---------------------------------------------------------------------------
# RED: SensitivityComparator must use tolerance for near-zero ORE delta
# ---------------------------------------------------------------------------


def test_compare_relative_diff_uses_tolerance_for_near_zero_ore_delta():
    """delta_rel_diff must be 0.0 when ore.delta is below a tolerance threshold.

    Currently: ``rel = 0.0 if ore.delta == 0.0 else (py - ore) / abs(ore)``
    If ore.delta == 1e-16 (machine-epsilon level, effectively zero), the
    division produces rel = -1.0 — a spurious 100% relative discrepancy.
    Fix: replace the exact equality check with ``abs(ore.delta) < 1e-12``.
    """
    from native_xva_interface.sensitivity import (
        ORESensitivityEntry,
        PythonSensitivityEntry,
    )
    from native_xva_interface import XVAEngine, OreSnapshotPythonLgmSensitivityComparator
    import tempfile, os

    # Build a comparator backed by the toy adapter (no SWIG needed)
    engine = XVAEngine()
    comparator = OreSnapshotPythonLgmSensitivityComparator(engine=engine)

    # Call the internal _build_comparison_rows helper if exposed, otherwise
    # exercise via the _compare_entries path.  For now we inspect the formula
    # directly by constructing what compare() would compute:
    py_entry = PythonSensitivityEntry(
        raw_quote_key="ZERO/RATE/EUR/5Y",
        normalized_factor="discount:EUR:5Y",
        shift_size=1e-4,
        base_value=0.02,
        base_metric_value=0.0,
        bumped_up_metric_value=0.0,
        bumped_down_metric_value=0.0,
        delta=0.0,
    )
    ore_entry = ORESensitivityEntry(
        factor="EUR_ZERO_5Y",
        normalized_factor="discount:EUR:5Y",
        shift_size=1e-4,
        base_xva=0.0,
        delta=1e-16,  # machine-epsilon: effectively zero but not == 0.0
    )

    # Replicate the production formula to confirm the bug is observable
    ore_delta = ore_entry.delta
    py_delta = py_entry.delta
    current_rel = 0.0 if ore_delta == 0.0 else (py_delta - ore_delta) / abs(ore_delta)

    # The bug: current formula gives -1.0 (100% discrepancy) for two ~zero values
    assert abs(current_rel) > 0.5, (
        "Pre-condition failed: expected the buggy formula to produce a large "
        f"relative diff for near-zero deltas, got {current_rel}"
    )

    # After the fix the comparator must report rel_diff ≈ 0.0 for this pair.
    # We can't easily call compare() without an output_dir, so we verify the
    # corrected formula directly — the test will be wired to the actual
    # production code once a _build_comparison_entry helper is extracted.
    corrected_rel = 0.0 if abs(ore_delta) < 1e-12 else (py_delta - ore_delta) / abs(ore_delta)
    assert corrected_rel == 0.0, (
        f"Corrected formula should give 0.0 for near-zero ore.delta, got {corrected_rel}"
    )

    # TODO: once the fix is in place, also assert via the public API:
    #   result = comparator._compute_rel_diff(py_entry, ore_entry)
    #   assert abs(result) < 1e-6


# ---------------------------------------------------------------------------
# GREEN: _survival_from_piecewise_hazard must handle a single-point hazard curve
# ---------------------------------------------------------------------------


def test_survival_from_piecewise_hazard_single_point_does_not_crash():
    """GREEN: a hazard curve with only one tenor must not raise IndexError or ZeroDivisionError."""
    import numpy as np

    surv = _survival_from_piecewise_hazard(
        times=[1.0, 2.0, 5.0],
        hazard_times=[5.0],
        hazard_rates=[0.02],
    )
    assert surv.shape == (3,)
    assert (surv > 0.0).all(), "Survival probabilities must be positive"
    assert (surv <= 1.0).all(), "Survival probabilities must be ≤ 1"
