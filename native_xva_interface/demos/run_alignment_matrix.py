from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable


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


def _run_case(name: str, build_snapshot: Callable):
    try:
        snapshot = build_snapshot()
        from native_xva_interface import ORESwigAdapter, XVAEngine

        result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
        reports = sorted(result.reports.keys())
        print(f"[{name}]")
        print(f"  xva_total    : {result.xva_total}")
        print(f"  xva_by_metric: {result.xva_by_metric}")
        print(f"  report_count : {len(reports)}")
        print(f"  has_xva      : {'xva' in result.reports}")
        print(f"  reports_head : {reports[:16]}")
        print()
        return result
    except Exception as exc:
        print(f"[{name}] FAILED: {exc}")
        print()
        return None


def main() -> None:
    repo_root = _bootstrap()

    try:
        import ORE  # noqa: F401
    except Exception as exc:
        print(f"ORE-SWIG unavailable/failed: {exc}")
        return
    from dataclasses import replace
    from native_xva_interface import (
        FXForward,
        FixingPoint,
        FixingsData,
        MarketData,
        MarketQuote,
        NettingConfig,
        NettingSet,
        Portfolio,
        Trade,
        XVAConfig,
        XVAEngine,
        XVALoader,
        XVASnapshot,
        stress_classic_fixing_lines,
        stress_classic_market_lines,
        stress_classic_native_preset,
        stress_classic_native_runtime,
        stress_classic_xml_buffers,
    )

    def strict_pure_native_snapshot() -> XVASnapshot:
        asof = "2016-02-05"
        quotes = []
        for ln in stress_classic_market_lines():
            parts = ln.split()
            if len(parts) < 3:
                continue
            d = parts[0]
            if len(d) == 8 and d.isdigit():
                d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            quotes.append(MarketQuote(date=d, key=parts[1], value=float(parts[2])))
        fixings = []
        for ln in stress_classic_fixing_lines():
            parts = ln.split()
            if len(parts) < 3:
                continue
            d = parts[0]
            if len(d) == 8 and d.isdigit():
                d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            fixings.append(FixingPoint(date=d, index=parts[1], value=float(parts[2])))
        runtime = stress_classic_native_runtime()
        runtime = replace(runtime, simulation=replace(runtime.simulation, samples=64, seed=42, strict_template=True))
        return XVASnapshot(
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
                        active_csa=True,
                        csa_currency="EUR",
                        threshold_pay=100000.0,
                        threshold_receive=100000.0,
                        mta_pay=0.0,
                        mta_receive=0.0,
                    )
                }
            ),
            config=XVAConfig(
                asof=asof,
                base_currency="EUR",
                analytics=("CVA", "DVA", "FVA", "MVA"),
                num_paths=64,
                runtime=runtime,
                params={
                    "market.lgmcalibration": "collateral_inccy",
                    "market.fxcalibration": "xois_eur",
                    "market.pricing": "xois_eur",
                    "market.simulation": "xois_eur",
                    "market.sensitivity": "xois_eur",
                },
            ),
        )

    def embedded_programmatic_snapshot() -> XVASnapshot:
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

        return XVASnapshot(
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
                runtime=stress_classic_native_runtime(),
                xml_buffers=stress_classic_xml_buffers(num_paths=64),
            ),
        )

    def hybrid_preset_snapshot() -> XVASnapshot:
        input_dir = repo_root / "Examples" / "XvaRisk" / "Input"
        base = XVALoader.from_files(str(input_dir), ore_file="ore_stress_classic.xml")
        return XVASnapshot(
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
            netting=base.netting,
            collateral=base.collateral,
            config=stress_classic_native_preset(repo_root, num_paths=64),
        )

    strict_res = _run_case("strict-pure-native", strict_pure_native_snapshot)
    embedded_res = _run_case("pure-programmatic-embedded-templates", embedded_programmatic_snapshot)
    hybrid_res = _run_case("hybrid-preset", hybrid_preset_snapshot)

    print("Alignment Verdict")
    print(f"  strict_non_zero  : {bool(strict_res and strict_res.xva_total != 0.0)}")
    print(f"  embedded_non_zero: {bool(embedded_res and embedded_res.xva_total != 0.0)}")
    print(f"  hybrid_non_zero  : {bool(hybrid_res and hybrid_res.xva_total != 0.0)}")


if __name__ == "__main__":
    main()
