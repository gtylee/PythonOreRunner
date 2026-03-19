from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    script_dir = str(here.parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)

    repo_root = here.parents[4]
    py_root = repo_root / "Tools" / "PythonOreRunner"
    swig_build = repo_root / "ORE-SWIG" / "build" / "lib.macosx-10.13-universal2-cpython-313"
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))
    if swig_build.exists() and str(swig_build) not in sys.path:
        sys.path.insert(0, str(swig_build))
    return repo_root


def main() -> None:
    repo_root = _bootstrap()

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
        config=stress_classic_native_preset(repo_root, num_paths=64),
        netting=base.netting,
        collateral=base.collateral,
        source_meta=base.source_meta,
    )

    try:
        result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
    except Exception as exc:
        print(f"ORE-SWIG unavailable/failed: {exc}")
        return
    print("run_id:", result.run_id)
    print("num_paths:", snapshot.config.num_paths)
    print("xva_total:", result.xva_total)
    print("xva_by_metric:", result.xva_by_metric)
    print("report_count:", len(result.reports))
    print("reports_head:", sorted(result.reports.keys())[:20])


if __name__ == "__main__":
    main()
