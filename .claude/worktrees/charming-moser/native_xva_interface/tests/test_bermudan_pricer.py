from pathlib import Path

import pytest
from py_ore_tools.repo_paths import require_engine_repo_root, pythonorerunner_root

from native_xva_interface import (
    benchmark_bermudan_from_ore_case,
    benchmark_bermudan_sensitivities_from_ore_case,
    price_bermudan_from_ore_case,
    price_bermudan_with_sensis_from_ore_case,
)


def _amc_input_dir() -> Path:
    return require_engine_repo_root() / "Examples" / "AmericanMonteCarlo" / "Input"


def _bermudan_compare_case(case_name: str) -> Path:
    return (
        pythonorerunner_root()
        / "parity_artifacts"
        / "bermudan_method_compare"
        / case_name
        / "Input"
    )


@pytest.mark.parametrize("method", ["lsmc", "backward"])
def test_price_bermudan_from_ore_case_is_deterministic_and_positive(method: str):
    case_dir = _amc_input_dir()

    r1 = price_bermudan_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method=method,
        num_paths=1024,
        seed=7,
        curve_mode="market_fit",
    )
    r2 = price_bermudan_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method=method,
        num_paths=1024,
        seed=7,
        curve_mode="market_fit",
    )

    assert r1.price > 0.0
    assert r1.method == method
    assert r1.price == pytest.approx(r2.price)
    assert len(r1.exercise_diagnostics) == 10
    assert r1.exercise_diagnostics[-1].time > r1.exercise_diagnostics[-2].time


def test_lsmc_and_backward_are_both_finite_and_reasonably_close():
    case_dir = _amc_input_dir()

    lsmc = price_bermudan_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method="lsmc",
        num_paths=2048,
        seed=13,
        curve_mode="market_fit",
    )
    backward = price_bermudan_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method="backward",
        num_paths=2048,
        seed=13,
        curve_mode="market_fit",
    )

    assert lsmc.price > 0.0
    assert backward.price > 0.0
    assert abs(lsmc.price - backward.price) / backward.price < 0.35


@pytest.mark.parametrize("case_name", ["single_first", "single_last", "berm_200bp"])
def test_backward_bermudan_prefers_ore_calibration_when_available(case_name: str):
    result = price_bermudan_from_ore_case(
        _bermudan_compare_case(case_name),
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method="backward",
        num_paths=256,
        seed=17,
        curve_mode="market_fit",
    )

    assert result.model_param_source == "calibration"


@pytest.mark.parametrize(
    ("case_name", "ore_price", "rel_tol"),
    [
        ("single_first", 46378.716109, 0.01),
        ("single_last", 5858.731775, 0.01),
        ("berm_200bp", 52289.126872, 5.0e-4),
    ],
)
def test_backward_bermudan_stays_close_to_ore_classic(case_name: str, ore_price: float, rel_tol: float):
    result = price_bermudan_from_ore_case(
        _bermudan_compare_case(case_name),
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method="backward",
        num_paths=256,
        seed=17,
        curve_mode="market_fit",
    )

    assert result.price == pytest.approx(ore_price, rel=rel_tol)


def test_price_bermudan_with_sensis_returns_stable_nonzero_delta():
    case_dir = _amc_input_dir()
    factor = "IR_SWAP/RATE/EUR/2D/1D/10Y"

    result = price_bermudan_with_sensis_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        num_paths=512,
        seed=9,
        factors=[factor],
        shift_size=1.0e-3,
    )

    assert len(result.sensitivities) == 1
    sensi = result.sensitivities[0]
    assert sensi.factor == factor
    assert abs(sensi.delta) > 1.0
    assert sensi.bumped_up_price != pytest.approx(sensi.bumped_down_price)


def test_fast_curve_jacobian_bermudan_sensis_track_full_reprice():
    case_dir = _bermudan_compare_case("berm_200bp")
    factors = [
        "IR_SWAP/RATE/EUR/2D/1D/10Y",
        "IR_SWAP/RATE/EUR/2D/6M/10Y",
    ]

    full = price_bermudan_with_sensis_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method="backward",
        num_paths=256,
        seed=17,
        factors=factors,
        shift_size=1.0e-4,
        sensitivity_mode="full_reprice",
    )
    fast = price_bermudan_with_sensis_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method="backward",
        num_paths=256,
        seed=17,
        factors=factors,
        shift_size=1.0e-4,
        sensitivity_mode="fast_curve_jacobian",
    )

    full_map = {s.factor: s for s in full.sensitivities}
    fast_map = {s.factor: s for s in fast.sensitivities}
    assert set(full_map) == set(fast_map) == set(factors)
    for factor in factors:
        f = full_map[factor]
        g = fast_map[factor]
        assert abs(g.delta) > 1.0
        assert g.delta == pytest.approx(f.delta, rel=0.15, abs=50.0)


def test_benchmark_bermudan_sensitivities_compares_fast_and_full():
    case_dir = _bermudan_compare_case("berm_200bp")
    factor = "IR_SWAP/RATE/EUR/2D/1D/10Y"

    result = benchmark_bermudan_sensitivities_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        method="backward",
        num_paths=256,
        seed=17,
        factors=[factor],
        shift_size=1.0e-4,
    )

    assert result.trade_id == "BermSwp"
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.factor == factor
    assert abs(row.python_full_reprice_delta) > 1.0
    assert abs(row.python_fast_delta) > 1.0
    assert row.fast_minus_full == pytest.approx(row.python_fast_delta - row.python_full_reprice_delta)


def test_benchmark_bermudan_infers_amc_expected_output():
    case_dir = _amc_input_dir()
    result = benchmark_bermudan_from_ore_case(
        case_dir,
        ore_file="ore_classic.xml",
        trade_id="BermSwp",
        num_paths=512,
        seed=11,
    )

    assert result.pricing.trade_id == "BermSwp"
    assert result.amc_exposure_rows > 0
    assert result.amc_npv is not None
