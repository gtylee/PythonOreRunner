import sys
from types import SimpleNamespace
from unittest.mock import patch

from pythonore.domain.dataclasses import MarketQuote
from pythonore.runtime.sensitivity import (
    OreSnapshotPythonLgmSensitivityComparator,
    PythonSensitivityEntry,
    _portfolio_factor_predicate,
    _prune_native_factor_setup_for_portfolio,
    _sample_times_for_legs,
)
from pythonore.runtime.runtime import ORESwigAdapter


def test_sensitivity_comparator_defaults_to_hybrid_swig_fallback():
    comparator = OreSnapshotPythonLgmSensitivityComparator()
    assert comparator.engine.adapter.fallback_to_swig is True


def test_ore_swig_adapter_primes_engine_swig_paths(tmp_path, monkeypatch):
    engine_root = tmp_path / "Engine"
    swig_root = engine_root / "ORE-SWIG"
    build_lib = swig_root / "build" / "lib.fake"
    build_lib.mkdir(parents=True)
    swig_root.mkdir(exist_ok=True)
    monkeypatch.setattr("pythonore.runtime.runtime.find_engine_repo_root", lambda: engine_root)
    original_path = list(sys.path)
    try:
        adapter = object.__new__(ORESwigAdapter)
        adapter._prime_swig_module_search_path()
        assert str(swig_root) in sys.path
        assert str(build_lib) in sys.path
    finally:
        sys.path[:] = original_path


def test_compute_python_sensitivities_uses_fast_npv_path_when_available():
    comparator = OreSnapshotPythonLgmSensitivityComparator(engine=SimpleNamespace())
    snapshot = SimpleNamespace(
        market=SimpleNamespace(
            raw_quotes=(
                MarketQuote(date="2026-03-21", key="ZERO/RATE/EUR/1Y", value=0.01),
            )
        ),
        config=SimpleNamespace(xml_buffers={}, params={}),
    )
    expected = [
        PythonSensitivityEntry(
            raw_quote_key="curve:DiscountCurve/EUR/0/1Y",
            normalized_factor="zero:EUR:1Y",
            ore_factor="DiscountCurve/EUR/0/1Y",
            shift_size=1.0e-4,
            base_value=0.0,
            base_metric_value=1.0,
            bumped_up_metric_value=1.1,
            bumped_down_metric_value=0.9,
            delta=0.1,
        )
    ]
    with patch.object(
        comparator,
        "_compute_python_npv_sensitivities_fast",
        return_value=expected,
    ) as mocked_fast:
        out = comparator.compute_python_sensitivities(
            snapshot,
            metric="NPV",
            factor_shifts={"zero:EUR:1Y": 1.0e-4},
            output_mode="bump_change",
        )
    assert out == expected
    mocked_fast.assert_called_once()


def test_sample_times_for_legs_collects_relevant_curve_points():
    legs = {
        "fixed_pay_time": [1.0, 2.0],
        "float_start_time": [0.5, 1.5],
        "float_end_time": [1.0, 2.0],
        "float_pay_time": [1.0, 2.0],
        "float_fixing_time": [0.4, 1.4],
    }
    out = _sample_times_for_legs(legs, include_float_coupon_dates=True)
    assert out.tolist() == [0.0, 0.4, 0.5, 1.0, 1.4, 1.5, 2.0]


def test_portfolio_factor_predicate_prunes_irrelevant_currencies_for_eur_swap():
    trade = SimpleNamespace(
        counterparty="CPTY_A",
        product=SimpleNamespace(ccy="EUR", float_index="EUR-EURIBOR-6M"),
    )
    snapshot = SimpleNamespace(
        portfolio=SimpleNamespace(trades=(trade,)),
        config=SimpleNamespace(base_currency="EUR"),
    )
    keep = _portfolio_factor_predicate(snapshot)
    assert keep("zero:EUR:5Y") is True
    assert keep("fwd:EUR:6M:5Y") is True
    assert keep("hazard:CPTY_A:5Y") is True
    assert keep("zero:CHF:5Y") is False
    assert keep("fwd:CHF:3M:5Y") is False


def test_prune_native_factor_setup_for_portfolio_keeps_only_relevant_factors():
    trade = SimpleNamespace(
        counterparty="CPTY_A",
        product=SimpleNamespace(ccy="EUR", float_index="EUR-EURIBOR-6M"),
    )
    snapshot = SimpleNamespace(
        portfolio=SimpleNamespace(trades=(trade,)),
        config=SimpleNamespace(base_currency="EUR"),
    )
    factor_shifts = {
        "zero:EUR:5Y": 1.0e-4,
        "fwd:EUR:6M:5Y": 1.0e-4,
        "zero:CHF:5Y": 1.0e-4,
    }
    curve_specs = {
        "zero:EUR:5Y": {"kind": "discount"},
        "fwd:EUR:6M:5Y": {"kind": "forward"},
        "zero:CHF:5Y": {"kind": "discount"},
    }
    labels = {
        "zero:EUR:5Y": "DiscountCurve/EUR/0/5Y",
        "fwd:EUR:6M:5Y": "IndexCurve/EUR-EURIBOR-6M/0/5Y",
        "zero:CHF:5Y": "DiscountCurve/CHF/0/5Y",
    }
    pruned_shifts, pruned_specs, pruned_labels, pruned_count = _prune_native_factor_setup_for_portfolio(
        snapshot,
        factor_shifts=factor_shifts,
        curve_factor_specs=curve_specs,
        factor_labels=labels,
    )
    assert set(pruned_shifts) == {"zero:EUR:5Y", "fwd:EUR:6M:5Y"}
    assert set(pruned_specs) == {"zero:EUR:5Y", "fwd:EUR:6M:5Y"}
    assert set(pruned_labels) == {"zero:EUR:5Y", "fwd:EUR:6M:5Y"}
    assert pruned_count == 1
