from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .dataclasses import (
    FXForward,
    FixingsData,
    MarketData,
    Portfolio,
    ConventionsConfig,
    CounterpartyConfig,
    CreditEntityConfig,
    CreditSimulationConfig,
    CrossAssetModelConfig,
    Trade,
    CurveConfig,
    PricingEngineConfig,
    RuntimeConfig,
    SimulationConfig,
    SimulationMarketConfig,
    XVASnapshot,
    SourceMeta,
    TodaysMarketConfig,
    XVAAnalyticConfig,
    XVAConfig,
)
from .loader import XVALoader
from .mapper import map_snapshot


def stress_classic_native_runtime() -> RuntimeConfig:
    """Native runtime defaults aligned to stress-classic style assumptions."""
    return RuntimeConfig(
        pricing_engine=PricingEngineConfig(model="DiscountedCashflows", npv_engine="DiscountingFxForwardEngine"),
        todays_market=TodaysMarketConfig(
            market_id="default",
            discount_curve="EUR-EONIA",
            fx_pairs=("EURUSD",),
        ),
        curve_configs=(
            CurveConfig(curve_id="EUR-EONIA", currency="EUR", tenors=("1Y", "2Y", "5Y", "10Y")),
            CurveConfig(curve_id="USD-SOFR", currency="USD", tenors=("1Y", "2Y", "5Y", "10Y")),
        ),
        simulation=SimulationConfig(samples=64, seed=42, dates=("88", "3M"), strict_template=True),
        simulation_market=SimulationMarketConfig(
            base_currency="EUR",
            currencies=("EUR", "USD"),
            indices=("EUR-ESTER", "USD-SOFR"),
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


def stress_classic_native_preset(repo_root: str | Path, num_paths: int = 64) -> XVAConfig:
    """
    Return a known-good stress-classic XVA config.

    This preset keeps runtime dataclasses fully native while injecting
    known-good XML buffers from the XvaRisk example as a compatibility
    bridge to get non-zero XVA today.
    """

    root = Path(repo_root)
    input_dir = root / "Examples" / "XvaRisk" / "Input"
    snap = XVALoader.from_files(str(input_dir), ore_file="ore_stress_classic.xml")

    runtime = stress_classic_native_runtime()
    native_buffers = _native_runtime_buffers(
        runtime=runtime,
        asof=snap.config.asof,
        base_currency=snap.config.base_currency,
        num_paths=num_paths,
    )
    xml_buffers = dict(snap.config.xml_buffers)
    # Stepwise replacement: use native pricing engine block first.
    if "pricingengine.xml" in native_buffers:
        xml_buffers["pricingengine.xml"] = native_buffers["pricingengine.xml"]
    if "simulation.xml" in native_buffers:
        xml_buffers["simulation.xml"] = native_buffers["simulation.xml"]

    cfg = replace(
        snap.config,
        num_paths=num_paths,
        runtime=runtime,
        xml_buffers=xml_buffers,
        source_meta=SourceMeta(origin="preset", path=str(input_dir / "ore_stress_classic.xml")),
    )
    return cfg


def _native_runtime_buffers(runtime: RuntimeConfig, asof: str, base_currency: str, num_paths: int) -> dict[str, str]:
    dummy = XVASnapshot(
        market=MarketData(asof=asof),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="PRESET_DUMMY_FXFWD",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=1_000_000, strike=1.1, maturity_years=1.0),
                ),
            )
        ),
        config=XVAConfig(
            asof=asof,
            base_currency=base_currency,
            analytics=("CVA",),
            num_paths=num_paths,
            runtime=runtime,
            params={"market.default": "default"},
        ),
    )
    return map_snapshot(dummy).xml_buffers
