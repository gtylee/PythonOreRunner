from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    script_dir = str(here.parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)

    repo_root = here.parents[4]
    py_root = repo_root / "Tools" / "PythonOreRunner"
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))


_bootstrap()

from native_xva_interface import (
    DeterministicToyAdapter,
    FXForward,
    FixingsData,
    MarketData,
    MarketQuote,
    NettingConfig,
    NettingSet,
    Portfolio,
    Trade,
    XVAConfig,
    XVAEngine,
    XVASnapshot,
)


def main() -> None:
    snapshot = XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/5Y", value=0.025),
            ),
        ),
        fixings=FixingsData(),
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
        netting=NettingConfig(
            netting_sets={
                "CPTY_A": NettingSet(netting_set_id="CPTY_A", counterparty="CPTY_A", active_csa=True, csa_currency="EUR")
            }
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", analytics=("CVA", "DVA", "FVA"), num_paths=64),
    )

    result = XVAEngine(adapter=DeterministicToyAdapter()).create_session(snapshot).run(return_cubes=False)
    print("run_id:", result.run_id)
    print("xva_total:", result.xva_total)
    print("xva_by_metric:", result.xva_by_metric)
    print("report_count:", len(result.reports))
    print("reports_head:", sorted(result.reports.keys())[:20])


if __name__ == "__main__":
    main()
