from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap_paths() -> Path:
    here = Path(__file__).resolve()
    script_dir = str(here.parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)

    repo_root = here.parents[4]
    py_integration = repo_root / "Tools" / "PythonOreRunner"
    ore_swig_build = repo_root / "ORE-SWIG" / "build" / "lib.macosx-10.13-universal2-cpython-313"

    if str(py_integration) not in sys.path:
        sys.path.insert(0, str(py_integration))
    if ore_swig_build.exists() and str(ore_swig_build) not in sys.path:
        sys.path.insert(0, str(ore_swig_build))

    return repo_root


def _default_ore_inputs(repo_root: Path) -> tuple[Path, str]:
    minimal_real_bundle = (
        repo_root / "Examples" / "Exposure" / "Input",
        "ore_swap.xml",
    )
    if (minimal_real_bundle[0] / minimal_real_bundle[1]).exists():
        return minimal_real_bundle

    fallback_bundle = (
        repo_root / "Examples" / "XvaRisk" / "Input",
        "ore_stress_classic.xml",
    )
    return fallback_bundle


def _print_result(label: str, result) -> None:
    print(f"\n{label}")
    print(f"  run_id: {result.run_id}")
    print(f"  xva_total: {result.xva_total}")
    print(f"  xva_by_metric: {result.xva_by_metric}")
    print(f"  exposure_by_netting_set: {result.exposure_by_netting_set}")
    print(f"  report_count: {len(result.reports)}")
    print(f"  reports: {sorted(result.reports.keys())[:20]}")


def main() -> None:
    repo_root = _bootstrap_paths()
    default_input_dir, default_ore_file = _default_ore_inputs(repo_root)

    from native_xva_interface import (
        CollateralBalance,
        CollateralConfig,
        ConventionsConfig,
        CounterpartyConfig,
        DeterministicToyAdapter,
        FixingPoint,
        FixingsData,
        FXForward,
        IRS,
        MarketData,
        MarketQuote,
        NettingConfig,
        NettingSet,
        ORESwigAdapter,
        Portfolio,
        PricingEngineConfig,
        RuntimeConfig,
        SimulationConfig,
        Trade,
        TodaysMarketConfig,
        CurveConfig,
        XVAConfig,
        XVAEngine,
        XVALoader,
        XVASnapshot,
    )

    parser = argparse.ArgumentParser(description="Demonstrate current native_xva_interface achievements")
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help="ORE input directory (defaults to a small real ORE example bundle)",
    )
    parser.add_argument(
        "--ore-file",
        default=default_ore_file,
        help="ORE config file inside input dir",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=64,
        help="Override simulation Samples in-memory for runtime",
    )
    args = parser.parse_args()

    print("Native XVA Interface Demo")
    print(f"  input_dir: {args.input_dir}")
    print(f"  ore_file: {args.ore_file}")
    print("  bundle: minimal real ORE files")

    snapshot = XVALoader.from_files(args.input_dir, ore_file=args.ore_file)
    cfg = snapshot.config
    cfg = XVAConfig(
        asof=cfg.asof,
        base_currency=cfg.base_currency,
        analytics=cfg.analytics,
        num_paths=args.num_paths,
        horizon_years=cfg.horizon_years,
        params=dict(cfg.params),
        xml_buffers=dict(cfg.xml_buffers),
        source_meta=cfg.source_meta,
    )
    snapshot = XVASnapshot(
        market=snapshot.market,
        fixings=snapshot.fixings,
        portfolio=snapshot.portfolio,
        config=cfg,
        netting=snapshot.netting,
        collateral=snapshot.collateral,
        source_meta=dict(snapshot.source_meta),
    )

    print("\nSnapshot loaded")
    print(f"  asof: {snapshot.config.asof}")
    print(f"  base_currency: {snapshot.config.base_currency}")
    print(f"  trades: {len(snapshot.portfolio.trades)}")
    print(f"  netting_sets: {sorted(snapshot.netting.netting_sets.keys())}")
    print(f"  market_quotes: {len(snapshot.market.raw_quotes)}")
    print(f"  fixings: {len(snapshot.fixings.points)}")
    print(f"  num_paths override: {snapshot.config.num_paths}")

    toy_engine = XVAEngine(adapter=DeterministicToyAdapter())
    toy_result = toy_engine.create_session(snapshot).run(return_cubes=False)
    _print_result("Toy Adapter Result", toy_result)

    try:
        swig_engine = XVAEngine(adapter=ORESwigAdapter())
        swig_result = swig_engine.create_session(snapshot).run(return_cubes=False)
        _print_result("ORE-SWIG Adapter Result", swig_result)
    except Exception as exc:
        print("\nORE-SWIG Adapter Result")
        print(f"  unavailable/failed: {exc}")

    # Programmatic, no-file demonstration.
    prog_snapshot = XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/5Y", value=0.025),
            ),
        ),
        fixings=FixingsData(
            points=(
                FixingPoint(date="2026-03-06", index="EUR-ESTR", value=0.019),
                FixingPoint(date="2026-03-07", index="EUR-ESTR", value=0.0191),
            )
        ),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="P_T1",
                    counterparty="CPY_P",
                    netting_set="NS_P",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=5_000_000, fixed_rate=0.024, maturity_years=5.0, pay_fixed=True),
                ),
                Trade(
                    trade_id="P_T2",
                    counterparty="CPY_P",
                    netting_set="NS_P",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=2_000_000, strike=1.11, maturity_years=1.0, buy_base=True),
                ),
            )
        ),
        netting=NettingConfig(
            netting_sets={
                "NS_P": NettingSet(
                    netting_set_id="NS_P",
                    counterparty="CPY_P",
                    active_csa=True,
                    csa_currency="EUR",
                    threshold_pay=0.0,
                    threshold_receive=0.0,
                    mta_pay=100000.0,
                    mta_receive=100000.0,
                )
            }
        ),
        collateral=CollateralConfig(
            balances=(
                CollateralBalance(
                    netting_set_id="NS_P",
                    currency="EUR",
                    initial_margin=250000.0,
                    variation_margin=50000.0,
                ),
            )
        ),
        config=XVAConfig(
            asof="2026-03-08",
            base_currency="EUR",
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=args.num_paths,
            horizon_years=5,
            runtime=RuntimeConfig(
                pricing_engine=PricingEngineConfig(model="LGM", npv_engine="Analytic"),
                todays_market=TodaysMarketConfig(market_id="default", discount_curve="EUR-EONIA", fx_pairs=("EURUSD",)),
                curve_configs=(CurveConfig(curve_id="EUR-EONIA", currency="EUR", tenors=("1Y", "5Y", "10Y")),),
                simulation=SimulationConfig(samples=args.num_paths, seed=42, dates=("1Y", "2Y", "5Y")),
                conventions=ConventionsConfig(day_counter="A365", calendar="TARGET"),
                counterparties=CounterpartyConfig(ids=("CPY_P",)),
            ),
        ),
    )

    print("\nProgrammatic Snapshot (no files)")
    print(f"  asof: {prog_snapshot.config.asof}")
    print(f"  trades: {len(prog_snapshot.portfolio.trades)}")
    print(f"  market_quotes: {len(prog_snapshot.market.raw_quotes)}")
    print(f"  fixings: {len(prog_snapshot.fixings.points)}")
    print(f"  num_paths: {prog_snapshot.config.num_paths}")

    prog_result = toy_engine.create_session(prog_snapshot).run(return_cubes=False)
    _print_result("Programmatic Snapshot (Toy Adapter)", prog_result)


if __name__ == "__main__":
    main()
