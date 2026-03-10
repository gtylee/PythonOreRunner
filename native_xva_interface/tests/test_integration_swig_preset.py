import os
from pathlib import Path

import pytest

from native_xva_interface import (
    FXForward,
    MarketData,
    ORESwigAdapter,
    Portfolio,
    Trade,
    XVAEngine,
    XVALoader,
    XVASnapshot,
    stress_classic_native_preset,
)
from native_xva_interface.tests.conftest import require_examples_repo_root


@pytest.mark.skipif(os.getenv("RUN_ORE_SWIG_INTEGRATION") != "1", reason="Set RUN_ORE_SWIG_INTEGRATION=1 to run SWIG integration")
def test_swig_stress_classic_preset_produces_non_zero_xva():
    try:
        import ORE  # noqa: F401
    except Exception as exc:
        pytest.skip(f"ORE-SWIG unavailable: {exc}")

    repo_root = require_examples_repo_root()
    input_dir = repo_root / "Examples" / "XvaRisk" / "Input"
    base = XVALoader.from_files(str(input_dir), ore_file="ore_stress_classic.xml")

    snapshot = XVASnapshot(
        market=MarketData(asof=base.market.asof, raw_quotes=base.market.raw_quotes),
        fixings=base.fixings,
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="PFX1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.09, maturity_years=1.0),
                ),
            )
        ),
        config=stress_classic_native_preset(repo_root, num_paths=10),
        netting=base.netting,
        collateral=base.collateral,
        source_meta=base.source_meta,
    )

    result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
    assert result.xva_total != 0.0
    assert "xva" in result.reports
    assert any(name.startswith("exposure_nettingset_") for name in result.reports)
