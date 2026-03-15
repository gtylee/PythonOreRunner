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
    swig_build = repo_root / "ORE-SWIG" / "build" / "lib.macosx-10.13-universal2-cpython-313"
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))
    if swig_build.exists() and str(swig_build) not in sys.path:
        sys.path.insert(0, str(swig_build))


def main() -> None:
    _bootstrap()

    from native_xva_interface import (
        FXForward,
        FixingPoint,
        FixingsData,
        MarketData,
        MarketQuote,
        NettingConfig,
        NettingSet,
        ORESwigAdapter,
        Portfolio,
        Trade,
        XVAConfig,
        XVAEngine,
        XVASnapshot,
        stress_classic_native_runtime,
        stress_classic_fixing_lines,
        stress_classic_market_lines,
        stress_classic_xml_buffers,
    )

    asof = "2016-02-05"

    market_lines = stress_classic_market_lines()
    fixing_lines = stress_classic_fixing_lines()
    quotes = []
    for ln in market_lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        d = parts[0]
        if len(d) == 8 and d.isdigit():
            d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        quotes.append(MarketQuote(date=d, key=parts[1], value=float(parts[2])))

    fixings = []
    for ln in fixing_lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        d = parts[0]
        if len(d) == 8 and d.isdigit():
            d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        fixings.append(FixingPoint(date=d, index=parts[1], value=float(parts[2])))

    runtime = stress_classic_native_runtime()
    xml_buffers = stress_classic_xml_buffers(num_paths=64)

    snapshot = XVASnapshot(
        market=MarketData(asof=asof, raw_quotes=tuple(quotes)),
        fixings=FixingsData(points=tuple(fixings)),
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
                "CPTY_A": NettingSet(
                    netting_set_id="CPTY_A",
                    counterparty="CPTY_A",
                    active_csa=False,
                    csa_currency="EUR",
                )
            }
        ),
        config=XVAConfig(
            asof=asof,
            base_currency="EUR",
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=64,
            params={
                "market.lgmcalibration": "collateral_inccy",
                "market.fxcalibration": "xois_eur",
                "market.pricing": "xois_eur",
                "market.simulation": "xois_eur",
                "market.sensitivity": "xois_eur",
            },
            runtime=runtime,
            xml_buffers=xml_buffers,
        ),
    )

    try:
        result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
    except Exception as exc:
        print(f"ORE-SWIG unavailable/failed: {exc}")
        return
    print("run_id:", result.run_id)
    print("xva_total:", result.xva_total)
    print("xva_by_metric:", result.xva_by_metric)
    print("report_count:", len(result.reports))
    print("reports_head:", sorted(result.reports.keys())[:20])


if __name__ == "__main__":
    main()
