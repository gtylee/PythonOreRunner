from pathlib import Path

from native_xva_interface import BermudanSwaption, XVALoader
from py_ore_tools.repo_paths import pythonorerunner_root, require_engine_repo_root


def _xvarisk_input_dir() -> Path:
    return pythonorerunner_root() / "Examples" / "XvaRisk" / "Input"


def test_loader_parses_known_example_folder():
    snap = XVALoader.from_files(str(_xvarisk_input_dir()), ore_file="ore_stress_classic.xml")
    assert snap.config.asof == "2016-02-05"
    assert snap.config.base_currency == "EUR"
    assert len(snap.portfolio.trades) >= 3
    assert "CPTY_A" in snap.netting.netting_sets
    assert len(snap.market.raw_quotes) > 100


def test_loader_extracts_expected_trade_and_netting_fields():
    snap = XVALoader.from_files(str(_xvarisk_input_dir()), ore_file="ore_stress_classic.xml")
    ids = {t.trade_id for t in snap.portfolio.trades}
    assert "Swap_EUR" in ids
    assert any(t.netting_set == "CPTY_A" for t in snap.portfolio.trades)


def test_loader_extracts_swap_float_index_from_trade_body():
    case_dir = (
        Path(__file__).resolve().parents[2]
        / "parity_artifacts"
        / "multiccy_benchmark_final"
        / "cases"
        / "flat_EUR_5Y_A"
        / "Input"
    )
    snap = XVALoader.from_files(str(case_dir), ore_file="ore.xml")
    trade = next(t for t in snap.portfolio.trades if t.trade_id == "SWAP_EUR_5Y_A_flat")
    assert trade.additional_fields["index"] == "EUR-EURIBOR-6M"


def test_loader_parses_native_bermudan_swaption():
    case_dir = require_engine_repo_root() / "Examples" / "AmericanMonteCarlo" / "Input"
    snap = XVALoader.from_files(str(case_dir), ore_file="ore_classic.xml")
    trade = next(t for t in snap.portfolio.trades if t.trade_id == "BermSwp")

    assert isinstance(trade.product, BermudanSwaption)
    assert trade.product.float_index == "EUR-EURIBOR-6M"
    assert trade.product.pay_fixed is True
    assert trade.product.exercise_dates[0] == "2026-02-25"
    assert trade.product.settlement == "Physical"
