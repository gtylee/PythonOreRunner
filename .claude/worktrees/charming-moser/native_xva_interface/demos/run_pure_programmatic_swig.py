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


def _programmatic_quotes(asof: str):
    from native_xva_interface import MarketQuote, stress_classic_market_lines

    quotes = []
    for ln in stress_classic_market_lines():
        parts = ln.split()
        if len(parts) < 3:
            continue
        d = parts[0]
        if len(d) == 8 and d.isdigit():
            d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        quotes.append(MarketQuote(date=d, key=parts[1], value=float(parts[2])))
    return tuple(quotes)


def main() -> None:
    _bootstrap()

    from native_xva_interface import (
        CollateralConfig,
        ConventionsConfig,
        CounterpartyConfig,
        CreditEntityConfig,
        CreditSimulationConfig,
        CrossAssetModelConfig,
        CurveConfig,
        FXForward,
        FixingsData,
        MarketData,
        NettingConfig,
        NettingSet,
        ORESwigAdapter,
        Portfolio,
        PricingEngineConfig,
        RuntimeConfig,
        SimulationConfig,
        SimulationMarketConfig,
        TodaysMarketConfig,
        Trade,
        XVAAnalyticConfig,
        XVAConfig,
        XVAEngine,
        XVASnapshot,
        FixingPoint,
        stress_classic_fixing_lines,
    )

    asof = "2016-02-05"
    runtime = RuntimeConfig(
        pricing_engine=PricingEngineConfig(model="DiscountedCashflows", npv_engine="DiscountingFxForwardEngine"),
        todays_market=TodaysMarketConfig(market_id="default", discount_curve="EUR-EONIA", fx_pairs=("EURUSD",)),
        curve_configs=(
            CurveConfig(curve_id="EUR-EONIA", currency="EUR", tenors=("1Y", "2Y", "5Y", "10Y")),
            CurveConfig(curve_id="USD-SOFR", currency="USD", tenors=("1Y", "2Y", "5Y", "10Y")),
        ),
        simulation=SimulationConfig(samples=64, seed=42, dates=("88", "3M"), strict_template=True),
        simulation_market=SimulationMarketConfig(
            base_currency="EUR",
            currencies=("EUR", "USD"),
            indices=("EUR-EONIA", "USD-LIBOR-3M", "USD-LIBOR-6M"),
            default_curve_names=("BANK", "CPTY_A"),
            fx_pairs=("USDEUR",),
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy="EUR",
            currencies=("EUR", "USD"),
            ir_model_ccys=("EUR", "USD"),
            fx_model_ccys=("USD",),
        ),
        xva_analytic=XVAAnalyticConfig(
            exposure_allocation_method="None",
            exposure_observation_model="Disable",
            pfe_quantile=0.95,
            collateral_calculation_type="Symmetric",
            marginal_allocation_limit=1.0,
            netting_set_ids=("CPTY_A",),
            cva_enabled=True,
            dva_enabled=False,
            fva_enabled=False,
            mva_enabled=False,
            colva_enabled=False,
            collateral_floor_enabled=False,
            dim_enabled=True,
            dim_quantile=0.99,
            dim_horizon_calendar_days=14,
            dim_regression_order=2,
            dim_regressors="",
            dim_output_netting_set="CPTY_A",
            dim_output_grid_points="0",
            dim_local_regression_evaluations=0,
            dim_local_regression_bandwidth=1.0,
            dva_name="BANK",
            fva_borrowing_curve="BANK_EUR_BORROW",
            fva_lending_curve="BANK_EUR_LEND",
        ),
        credit_simulation=CreditSimulationConfig(
            enabled=True,
            netting_set_ids=("CPTY_A",),
            entities=(CreditEntityConfig(name="CPTY_A"), CreditEntityConfig(name="BANK")),
            paths=64,
            seed=42,
        ),
        conventions=ConventionsConfig(day_counter="A365", calendar="TARGET"),
        counterparties=CounterpartyConfig(ids=("CPTY_A",)),
    )

    fixing_points = []
    for ln in stress_classic_fixing_lines():
        parts = ln.split()
        if len(parts) < 3:
            continue
        d = parts[0]
        if len(d) == 8 and d.isdigit():
            d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        fixing_points.append(FixingPoint(date=d, index=parts[1], value=float(parts[2])))

    snapshot = XVASnapshot(
        market=MarketData(asof=asof, raw_quotes=_programmatic_quotes(asof)),
        fixings=FixingsData(points=tuple(fixing_points)),
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
        collateral=CollateralConfig(),
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
