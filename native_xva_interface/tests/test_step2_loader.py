from pathlib import Path

from native_xva_interface import XVALoader
from native_xva_interface.tests.conftest import require_examples_repo_root


def _xvarisk_input_dir() -> Path:
    return require_examples_repo_root() / "Examples" / "XvaRisk" / "Input"


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
