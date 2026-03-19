from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import date
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from py_ore_tools.repo_paths import local_parity_artifacts_root, require_engine_repo_root

from . import (
    CollateralBalance,
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
    MarketQuote,
    NettingConfig,
    NettingSet,
    ORESwigAdapter,
    Portfolio,
    PricingEngineConfig,
    PythonLgmAdapter,
    RuntimeConfig,
    SimulationConfig,
    SimulationMarketConfig,
    SourceMeta,
    TodaysMarketConfig,
    Trade,
    XVAAnalyticConfig,
    XVAConfig,
    XVAEngine,
    XVASnapshot,
)
from .loader import XVALoader
from .mapper import _default_index_for_ccy
from .mapper import _curve_convention_for_currency
from .mapper import map_snapshot


DEFAULT_CURRENCIES: tuple[str, ...] = (
    "EUR",
    "USD",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "SEK",
    "NOK",
    "DKK",
    "CZK",
    "PLN",
    "HUF",
    "RON",
    "TRY",
    "ZAR",
    "MXN",
    "BRL",
    "CLP",
    "COP",
    "PEN",
    "CNY",
    "CNH",
    "KRW",
    "SGD",
    "HKD",
    "INR",
    "IDR",
    "PHP",
)
DEFAULT_MATURITY_MONTHS: tuple[int, ...] = (1, 2, 3, 5)
DEFAULT_CURVE_TENORS: tuple[str, ...] = ("1M", "2M", "3M", "5M", "1Y", "2Y", "5Y")
DEFAULT_CREDIT_TENORS: tuple[str, ...] = ("1Y", "2Y", "5Y")
DEFAULT_COUNTERPARTY = "CPTY_A"
DEFAULT_NETTING_SET = "NS1"
DEFAULT_OWN_NAME = "BANK"
DEFAULT_SMALL_MTA = 10000.0
_TEMPLATE_CASE_BY_BASE: dict[str, str] = {
    "EUR": "exposure_with_collateral_fx_xva",
    "USD": "flat_USD_5Y_A",
    "GBP": "flat_GBP_5Y_A",
    "CAD": "flat_CAD_10Y_B",
}

_SPOT_LEVELS: dict[str, float] = {
    "USD": 1.09,
    "GBP": 0.86,
    "JPY": 161.0,
    "CHF": 0.97,
    "CAD": 1.47,
    "AUD": 1.64,
    "NZD": 1.78,
    "SEK": 11.25,
    "NOK": 11.70,
    "DKK": 7.46,
    "CZK": 25.10,
    "PLN": 4.31,
    "HUF": 392.0,
    "RON": 4.97,
    "TRY": 35.4,
    "ZAR": 20.1,
    "MXN": 18.4,
    "BRL": 5.45,
    "CLP": 1030.0,
    "COP": 4340.0,
    "PEN": 4.05,
    "CNY": 7.82,
    "CNH": 7.84,
    "KRW": 1460.0,
    "SGD": 1.46,
    "HKD": 8.49,
    "INR": 90.3,
    "IDR": 17050.0,
    "PHP": 61.1,
}
_ZERO_RATE_LEVELS: dict[str, float] = {
    "EUR": 0.021,
    "USD": 0.041,
    "GBP": 0.038,
    "JPY": 0.009,
    "CHF": 0.013,
    "CAD": 0.034,
    "AUD": 0.041,
    "NZD": 0.044,
    "SEK": 0.028,
    "NOK": 0.037,
    "DKK": 0.022,
    "CZK": 0.036,
    "PLN": 0.049,
    "HUF": 0.061,
    "RON": 0.054,
    "TRY": 0.165,
    "ZAR": 0.082,
    "MXN": 0.091,
    "BRL": 0.107,
    "CLP": 0.058,
    "COP": 0.096,
    "PEN": 0.053,
    "CNY": 0.028,
    "CNH": 0.029,
    "KRW": 0.031,
    "SGD": 0.025,
    "HKD": 0.034,
    "INR": 0.068,
    "IDR": 0.062,
    "PHP": 0.057,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a synthetic large FX-forward MPOR benchmark via native_xva_interface.",
    )
    parser.add_argument("--engine", choices=("python", "ore", "both"), default="both")
    parser.add_argument("--base-ccy", default="EUR")
    parser.add_argument(
        "--currencies",
        default=",".join(DEFAULT_CURRENCIES),
        help="Comma-separated currency list. Base currency will be included and placed first.",
    )
    parser.add_argument("--trade-count", type=int, default=None)
    parser.add_argument("--trades-per-ccy", type=int, default=len(DEFAULT_MATURITY_MONTHS))
    parser.add_argument("--max-maturity-months", type=int, default=5)
    parser.add_argument("--mpor-days", type=int, default=10)
    parser.add_argument("--ore-paths", type=int, default=250)
    parser.add_argument("--py-paths", type=int, default=2000)
    parser.add_argument(
        "--python-market-source",
        choices=("synthetic", "ore-template"),
        default="synthetic",
        help="Python market source. Use ore-template for a like-for-like run against the prepared ORE case.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("Tools/PythonOreRunner/parity_artifacts/large_fx_universe_benchmark_latest.json"),
    )
    return parser


def parse_currency_list(base_ccy: str, currencies: str | Sequence[str]) -> tuple[str, ...]:
    base = str(base_ccy).strip().upper()
    if isinstance(currencies, str):
        raw = [item.strip().upper() for item in currencies.split(",") if item.strip()]
    else:
        raw = [str(item).strip().upper() for item in currencies if str(item).strip()]
    seen: set[str] = set()
    ordered = [base]
    seen.add(base)
    for ccy in raw:
        if len(ccy) != 3:
            raise ValueError(f"Currency code must be 3 letters: {ccy!r}")
        if ccy not in seen:
            seen.add(ccy)
            ordered.append(ccy)
    if len(ordered) < 2:
        raise ValueError("At least one non-base currency is required")
    return tuple(ordered)


def maturity_months(max_maturity_months: int, trades_per_ccy: int) -> tuple[int, ...]:
    if max_maturity_months <= 0:
        raise ValueError("max_maturity_months must be > 0")
    if trades_per_ccy <= 0:
        raise ValueError("trades_per_ccy must be > 0")
    allowed = [month for month in DEFAULT_MATURITY_MONTHS if month <= max_maturity_months]
    if not allowed:
        raise ValueError("No supported short-dated maturities remain under max_maturity_months")
    out = [allowed[i % len(allowed)] for i in range(trades_per_ccy)]
    return tuple(out)


def build_fx_forward_trades(
    base_ccy: str,
    currencies: Sequence[str],
    trades_per_ccy: int = len(DEFAULT_MATURITY_MONTHS),
    max_maturity_months: int = 5,
    trade_count: int | None = None,
    asof: str = "2026-03-08",
) -> tuple[Trade, ...]:
    base = str(base_ccy).strip().upper()
    ordered = parse_currency_list(base, currencies)
    maturities = maturity_months(max_maturity_months, trades_per_ccy)
    foreign_ccys = [ccy for ccy in ordered if ccy != base]
    all_trades: list[Trade] = []
    notional_cycle = (1_250_000.0, 1_750_000.0, 2_250_000.0, 2_750_000.0)
    strike_bump_pattern = (0.0005, 0.0005, -0.0005, -0.0005)

    for ccy_index, foreign in enumerate(foreign_ccys):
        spot = spot_level(base, foreign)
        base_notional = notional_cycle[ccy_index % len(notional_cycle)] * (1.0 + 0.02 * (ccy_index % 3))
        for maturity_index, month in enumerate(maturities):
            cycle_index = ccy_index * len(maturities) + maturity_index
            buy_base = cycle_index % 2 == 0
            notional = base_notional
            maturity_years = _year_fraction_from_months(asof, month)
            fair_forward = fair_forward_strike(base, foreign, maturity_years, spot=spot)
            strike_bump = strike_bump_pattern[maturity_index % len(strike_bump_pattern)]
            strike = fair_forward * (1.0 + strike_bump)
            trade_id = f"FXFWD_{base}{foreign}_{month:02d}M_{cycle_index + 1:03d}"
            all_trades.append(
                Trade(
                    trade_id=trade_id,
                    counterparty=DEFAULT_COUNTERPARTY,
                    netting_set=DEFAULT_NETTING_SET,
                    trade_type="FxForward",
                    product=FXForward(
                        pair=f"{base}{foreign}",
                        notional=float(round(notional, 2)),
                        strike=float(round(strike, 8)),
                        maturity_years=float(maturity_years),
                        buy_base=buy_base,
                    ),
                )
            )

    if trade_count is not None:
        if trade_count <= 0:
            raise ValueError("trade_count must be > 0 when provided")
        if trade_count > len(all_trades):
            raise ValueError(
                f"Requested trade_count={trade_count} exceeds generated universe size={len(all_trades)}"
            )
        all_trades = all_trades[:trade_count]

    return tuple(all_trades)


def build_large_fx_universe_snapshot(
    *,
    base_ccy: str = "EUR",
    currencies: Sequence[str] = DEFAULT_CURRENCIES,
    trades_per_ccy: int = len(DEFAULT_MATURITY_MONTHS),
    trade_count: int | None = None,
    max_maturity_months: int = 5,
    mpor_days: int = 10,
    num_paths: int = 2000,
    seed: int = 42,
    asof: str = "2026-03-08",
) -> XVASnapshot:
    ordered = parse_currency_list(base_ccy, currencies)
    base = ordered[0]
    trades = build_fx_forward_trades(
        base,
        ordered,
        trades_per_ccy=trades_per_ccy,
        max_maturity_months=max_maturity_months,
        trade_count=trade_count,
        asof=asof,
    )
    runtime = build_runtime_config(ordered, base_currency=base, num_paths=num_paths, seed=seed)
    curve_ids = {ccy: curve_id_for_currency(ccy) for ccy in ordered}
    indices = {ccy: _default_index_for_ccy(ccy) for ccy in ordered}
    market = MarketData(
        asof=asof,
        raw_quotes=build_market_quotes(ordered, base_currency=base, asof=asof),
        source_meta=SourceMeta(origin="synthetic-large-fx-universe"),
    )
    return XVASnapshot(
        market=market,
        fixings=FixingsData(),
        portfolio=Portfolio(trades=trades, source_meta=SourceMeta(origin="synthetic-large-fx-universe")),
        netting=NettingConfig(
            netting_sets={
                DEFAULT_NETTING_SET: NettingSet(
                    netting_set_id=DEFAULT_NETTING_SET,
                    counterparty=DEFAULT_COUNTERPARTY,
                    active_csa=True,
                    csa_currency=base,
                    margin_period_of_risk=f"{max(int(mpor_days), 0)}D",
                    threshold_pay=0.0,
                    threshold_receive=0.0,
                    mta_pay=DEFAULT_SMALL_MTA,
                    mta_receive=DEFAULT_SMALL_MTA,
                )
            }
        ),
        collateral=CollateralConfig(
            balances=(
                CollateralBalance(
                    netting_set_id=DEFAULT_NETTING_SET,
                    currency=base,
                    initial_margin=0.0,
                    variation_margin=0.0,
                ),
            )
        ),
        config=XVAConfig(
            asof=asof,
            base_currency=base,
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=num_paths,
            params={
                "python.mpor_days": str(max(int(mpor_days), 0)),
                "python.mpor_mode": "sticky",
                "python.lgm_rng_mode": "ore_parity",
            },
            xml_buffers=build_benchmark_xml_buffers(ordered, base_currency=base, curve_ids=curve_ids, indices=indices),
            runtime=runtime,
            source_meta=SourceMeta(origin="synthetic-large-fx-universe"),
        ),
        source_meta={"builder": SourceMeta(origin="synthetic-large-fx-universe")},
    )


def build_runtime_config(
    currencies: Sequence[str],
    *,
    base_currency: str,
    num_paths: int,
    seed: int,
) -> RuntimeConfig:
    ordered = parse_currency_list(base_currency, currencies)
    base = ordered[0]
    foreign = tuple(ccy for ccy in ordered if ccy != base)
    curve_ids = {ccy: curve_id_for_currency(ccy) for ccy in ordered}
    indices = {ccy: _default_index_for_ccy(ccy) for ccy in ordered}
    curve_configs = tuple(
        CurveConfig(curve_id=curve_ids[ccy], currency=ccy, tenors=DEFAULT_CURVE_TENORS)
        for ccy in ordered
    )


def _year_fraction_from_months(asof: str, months: int) -> float:
    start = date.fromisoformat(str(asof))
    y = start.year + (start.month - 1 + int(months)) // 12
    m = (start.month - 1 + int(months)) % 12 + 1
    d = min(start.day, _month_end_day(y, m))
    end = date(y, m, d)
    return max((end - start).days / 365.0, 0.0)


def _month_end_day(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days
    correlations: list[tuple[str, str, float]] = []
    for ccy in foreign:
        correlations.extend(
            [
                (f"IR:{base}", f"IR:{ccy}", 0.25),
                (f"IR:{base}", f"FX:{ccy}{base}", -0.10),
                (f"IR:{ccy}", f"FX:{ccy}{base}", 0.15),
            ]
        )
    return RuntimeConfig(
        pricing_engine=PricingEngineConfig(model="DiscountedCashflows", npv_engine="DiscountingFxForwardEngine"),
        todays_market=TodaysMarketConfig(
            market_id="default",
            discount_curve=curve_ids[base],
            fx_pairs=tuple(f"{base}{ccy}" for ccy in foreign),
        ),
        curve_configs=curve_configs,
        simulation=SimulationConfig(samples=num_paths, seed=seed, dates=("1M", "2M", "3M", "5M"), strict_template=True),
        simulation_market=SimulationMarketConfig(
            base_currency=base,
            currencies=tuple(ordered),
            indices=tuple(indices[ccy] for ccy in ordered),
            default_curve_names=(DEFAULT_OWN_NAME, DEFAULT_COUNTERPARTY),
            fx_pairs=tuple(f"{ccy}{base}" for ccy in foreign),
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy=base,
            currencies=tuple(ordered),
            ir_model_ccys=tuple(ordered),
            fx_model_ccys=foreign,
            correlations=tuple(correlations),
        ),
        xva_analytic=XVAAnalyticConfig(
            netting_set_ids=(DEFAULT_NETTING_SET,),
            cva_enabled=True,
            dva_enabled=True,
            fva_enabled=True,
            mva_enabled=False,
            colva_enabled=False,
            collateral_floor_enabled=False,
            dim_enabled=False,
            dva_name=DEFAULT_OWN_NAME,
            fva_borrowing_curve=f"{DEFAULT_OWN_NAME}_{base}_BORROW",
            fva_lending_curve=f"{DEFAULT_OWN_NAME}_{base}_LEND",
        ),
        credit_simulation=CreditSimulationConfig(
            enabled=True,
            netting_set_ids=(DEFAULT_NETTING_SET,),
            entities=(CreditEntityConfig(name=DEFAULT_COUNTERPARTY), CreditEntityConfig(name=DEFAULT_OWN_NAME)),
            paths=num_paths,
            seed=seed,
        ),
        conventions=ConventionsConfig(day_counter="A365", calendar="TARGET"),
        counterparties=CounterpartyConfig(
            ids=(DEFAULT_COUNTERPARTY, DEFAULT_OWN_NAME),
            curve_currencies={DEFAULT_COUNTERPARTY: base, DEFAULT_OWN_NAME: base},
        ),
    )


def build_market_quotes(currencies: Sequence[str], *, base_currency: str, asof: str = "2026-03-08") -> tuple[MarketQuote, ...]:
    ordered = parse_currency_list(base_currency, currencies)
    base = ordered[0]
    quotes: list[MarketQuote] = []

    for ccy in ordered:
        base_rate = zero_rate_level(ccy)
        for tenor_index, tenor in enumerate(DEFAULT_CURVE_TENORS):
            quotes.append(
                MarketQuote(
                    date=asof,
                    key=f"ZERO/RATE/{ccy}/{tenor}",
                    value=round(base_rate + 0.0008 * tenor_index, 8),
                )
            )

    for ccy in ordered:
        if ccy == base:
            continue
        quotes.append(MarketQuote(date=asof, key=f"FX/{base}/{ccy}", value=round(spot_level(base, ccy), 8)))

    for name, spread in ((DEFAULT_COUNTERPARTY, 0.0125), (DEFAULT_OWN_NAME, 0.0100)):
        for tenor_index, tenor in enumerate(DEFAULT_CREDIT_TENORS):
            quotes.append(
                MarketQuote(
                    date=asof,
                    key=f"CDS/CREDIT_SPREAD/{name}/SNRFOR/{base}/{tenor}",
                    value=round(spread + 0.0009 * tenor_index, 8),
                )
            )
        quotes.append(
            MarketQuote(
                date=asof,
                key=f"RECOVERY_RATE/RATE/{name}/SNRFOR/{base}",
                value=0.40 if name == DEFAULT_OWN_NAME else 0.35,
            )
        )

    for curve_name, level in ((f"{DEFAULT_OWN_NAME}_{base}_BORROW", 0.0015), (f"{DEFAULT_OWN_NAME}_{base}_LEND", 0.0007)):
        for tenor_index, tenor in enumerate(("1Y", "2Y", "5Y")):
            quotes.append(
                MarketQuote(
                    date=asof,
                    key=f"ZERO/YIELD_SPREAD/{base}/{curve_name}/A365/{tenor}",
                    value=round(level + 0.0003 * tenor_index, 8),
                )
            )

    return tuple(quotes)


def spot_level(base_currency: str, foreign_currency: str) -> float:
    if foreign_currency == base_currency:
        return 1.0
    if foreign_currency in _SPOT_LEVELS:
        return _SPOT_LEVELS[foreign_currency]
    seed = sum(ord(c) for c in (base_currency + foreign_currency))
    return round(0.75 + (seed % 300) / 100.0, 8)


def zero_rate_level(currency: str) -> float:
    if currency in _ZERO_RATE_LEVELS:
        return _ZERO_RATE_LEVELS[currency]
    seed = sum(ord(c) for c in currency)
    return round(0.015 + (seed % 20) / 1000.0, 8)


def curve_id_for_currency(currency: str) -> str:
    return f"{currency.upper()}-DISC"


def fair_forward_strike(base_currency: str, foreign_currency: str, maturity_years: float, *, spot: float | None = None) -> float:
    s0 = spot if spot is not None else spot_level(base_currency, foreign_currency)
    r_base = zero_rate_level(base_currency)
    r_foreign = zero_rate_level(foreign_currency)
    return float(s0 * math.exp((r_foreign - r_base) * float(maturity_years)))


def build_benchmark_xml_buffers(
    currencies: Sequence[str],
    *,
    base_currency: str,
    curve_ids: dict[str, str],
    indices: dict[str, str],
) -> dict[str, str]:
    ordered = parse_currency_list(base_currency, currencies)
    return {
        "todaysmarket.xml": build_benchmark_todaysmarket_xml(ordered, base_currency=ordered[0], curve_ids=curve_ids, indices=indices),
        "curveconfig.xml": build_benchmark_curveconfig_xml(ordered, base_currency=ordered[0], curve_ids=curve_ids),
    }


def build_benchmark_todaysmarket_xml(
    currencies: Sequence[str],
    *,
    base_currency: str,
    curve_ids: dict[str, str],
    indices: dict[str, str],
) -> str:
    ordered = parse_currency_list(base_currency, currencies)
    base = ordered[0]
    yield_curves = "\n".join(
        f'    <YieldCurve name="{curve_ids[ccy]}">Yield/{ccy}/{curve_ids[ccy]}</YieldCurve>'
        for ccy in ordered
    )
    discounting = "\n".join(
        f'    <DiscountingCurve currency="{ccy}">Yield/{ccy}/{curve_ids[ccy]}</DiscountingCurve>'
        for ccy in ordered
    )
    forwarding = "\n".join(
        f'    <Index name="{indices[ccy]}">Yield/{ccy}/{curve_ids[ccy]}</Index>'
        for ccy in ordered
    )
    fx_spots = "\n".join(
        f'    <FxSpot pair="{base}{ccy}">FX/{base}/{ccy}</FxSpot>'
        for ccy in ordered
        if ccy != base
    )
    return "\n".join(
        [
            "<TodaysMarket>",
            '  <Configuration id="default">',
            "    <YieldCurvesId>default</YieldCurvesId>",
            "    <DiscountingCurvesId>default</DiscountingCurvesId>",
            "    <IndexForwardingCurvesId>default</IndexForwardingCurvesId>",
            "    <FxSpotsId>default</FxSpotsId>",
            "    <FxVolatilitiesId>default</FxVolatilitiesId>",
            "    <SwaptionVolatilitiesId>default</SwaptionVolatilitiesId>",
            "    <DefaultCurvesId>default</DefaultCurvesId>",
            "  </Configuration>",
            '  <YieldCurves id="default">',
            yield_curves,
            "  </YieldCurves>",
            '  <DiscountingCurves id="default">',
            discounting,
            "  </DiscountingCurves>",
            '  <IndexForwardingCurves id="default">',
            forwarding,
            "  </IndexForwardingCurves>",
            '  <FxSpots id="default">',
            fx_spots,
            "  </FxSpots>",
            '  <FxVolatilities id="default"/>',
            '  <SwaptionVolatilities id="default"/>',
            '  <DefaultCurves id="default">',
            f'    <DefaultCurve name="{DEFAULT_OWN_NAME}">Default/{base}/{DEFAULT_OWN_NAME}</DefaultCurve>',
            f'    <DefaultCurve name="{DEFAULT_COUNTERPARTY}">Default/{base}/{DEFAULT_COUNTERPARTY}</DefaultCurve>',
            "  </DefaultCurves>",
            "</TodaysMarket>",
        ]
    )


def build_benchmark_curveconfig_xml(
    currencies: Sequence[str],
    *,
    base_currency: str,
    curve_ids: dict[str, str],
) -> str:
    ordered = parse_currency_list(base_currency, currencies)
    yield_curves: list[str] = ["<CurveConfiguration>", "  <YieldCurves>"]
    for ccy in ordered:
        quotes = "".join(f"<Quote>ZERO/RATE/{ccy}/{tenor}</Quote>" for tenor in DEFAULT_CURVE_TENORS)
        yield_curves.extend(
            [
                "    <YieldCurve>",
                f"      <CurveId>{curve_ids[ccy]}</CurveId>",
                "      <CurveDescription>Large FX benchmark curve</CurveDescription>",
                f"      <Currency>{ccy}</Currency>",
                f"      <DiscountCurve>{curve_ids[ccy]}</DiscountCurve>",
                "      <Segments>",
                "        <Simple>",
                "          <Type>Deposit</Type>",
                f"          <Quotes>{quotes}</Quotes>",
                f"          <Conventions>{_curve_convention_for_currency(ccy)}</Conventions>",
                "        </Simple>",
                "      </Segments>",
                "      <InterpolationVariable>Discount</InterpolationVariable>",
                "      <InterpolationMethod>LogLinear</InterpolationMethod>",
                "      <YieldCurveDayCounter>A365</YieldCurveDayCounter>",
                "      <Extrapolation>true</Extrapolation>",
                "    </YieldCurve>",
            ]
        )
    yield_curves.extend(["  </YieldCurves>", "  <DefaultCurves>"])
    for name in (DEFAULT_OWN_NAME, DEFAULT_COUNTERPARTY):
        yield_curves.extend(
            [
                "    <DefaultCurve>",
                f"      <CurveId>{name}</CurveId>",
                "      <CurveDescription/>",
                f"      <Currency>{base_currency}</Currency>",
                "      <Configurations>",
                '        <Configuration priority="0">',
                "          <Type>SpreadCDS</Type>",
                f"          <DiscountCurve>Yield/{base_currency}/{curve_ids[base_currency]}</DiscountCurve>",
                "          <DayCounter>Actual/365 (Fixed)</DayCounter>",
                f"          <RecoveryRate>RECOVERY_RATE/RATE/{name}/SNRFOR/{base_currency}</RecoveryRate>",
                "          <Quotes>",
                f"            <Quote>CDS/CREDIT_SPREAD/{name}/SNRFOR/{base_currency}/*</Quote>",
                "          </Quotes>",
                "          <Conventions>CDS-STANDARD-CONVENTIONS</Conventions>",
                "          <Extrapolation>true</Extrapolation>",
                "          <AllowNegativeRates>false</AllowNegativeRates>",
                "        </Configuration>",
                "      </Configurations>",
                "    </DefaultCurve>",
            ]
        )
    yield_curves.extend(["  </DefaultCurves>", "</CurveConfiguration>"])
    return "\n".join(yield_curves)


def ore_swig_available() -> bool:
    try:
        ORESwigAdapter()
    except Exception:
        return False
    return True


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    requested_currencies = parse_currency_list(args.base_ccy, args.currencies)
    effective_currencies = requested_currencies
    effective_python_market_source = "ore-template" if args.engine == "both" else str(args.python_market_source)
    use_template_subset = args.engine in ("ore", "both") or effective_python_market_source == "ore-template"
    if use_template_subset:
        template_case = _template_case_path(requested_currencies[0])
        supported = _template_supported_currencies(template_case)
        effective_currencies = tuple(ccy for ccy in requested_currencies if ccy in supported)
        dropped = tuple(ccy for ccy in requested_currencies if ccy not in supported)
        if len(effective_currencies) < 2:
            raise RuntimeError(
                f"ORE template {template_case.name} does not support enough requested currencies: {requested_currencies}"
            )
    else:
        template_case = None
        dropped = ()
    benchmark_asof = _template_asof_date(template_case) if template_case is not None else "2026-03-08"
    effective_mpor_days = int(args.mpor_days)
    mpor_warning = None
    if template_case is not None and template_case.name == "Example_9" and effective_mpor_days > 0:
        effective_mpor_days = 0
        mpor_warning = (
            f"Template {template_case.name} does not produce usable ORE XVA with sticky MPOR; "
            f"using effective MPOR 0D for parity instead of requested {int(args.mpor_days)}D"
        )

    report: dict[str, Any] = {
        "config": {
            "engine": args.engine,
            "asof": benchmark_asof,
            "base_ccy": effective_currencies[0],
            "currencies": list(effective_currencies),
            "requested_currencies": list(requested_currencies),
            "trades_per_ccy": int(args.trades_per_ccy),
            "trade_count_request": None if args.trade_count is None else int(args.trade_count),
            "max_maturity_months": int(args.max_maturity_months),
            "mpor_days": effective_mpor_days,
            "requested_mpor_days": int(args.mpor_days),
            "ore_paths": int(args.ore_paths),
            "py_paths": int(args.py_paths),
            "python_market_source": effective_python_market_source,
            "seed": int(args.seed),
        },
        "currencies": list(effective_currencies),
        "requested_currencies": list(requested_currencies),
        "mpor": {"days": effective_mpor_days, "requested_days": int(args.mpor_days), "mode": "sticky"},
        "engines": {},
        "warnings": [],
    }
    if dropped:
        report["warnings"].append(
            f"Using ORE-supported subset from template {template_case.name}: dropped {', '.join(dropped)}"
        )
    if mpor_warning:
        report["warnings"].append(mpor_warning)

    full_trades = build_fx_forward_trades(
        effective_currencies[0],
        effective_currencies,
        trades_per_ccy=args.trades_per_ccy,
        max_maturity_months=args.max_maturity_months,
        trade_count=args.trade_count,
    )
    report["trade_count"] = len(full_trades)
    report["trade_ids_sample"] = [trade.trade_id for trade in full_trades[:8]]

    requested_engines = ("python", "ore") if args.engine == "both" else (args.engine,)
    requested_paths = {"python": int(args.py_paths), "ore": int(args.ore_paths)}

    if args.engine == "both":
        for engine_name in requested_engines:
            report["engines"][engine_name] = _run_engine_subprocess(
                args,
                engine_name,
                currencies=effective_currencies,
            )
            if report["engines"][engine_name]["status"] != "ok":
                report["warnings"].append(f"{engine_name} run failed: {report['engines'][engine_name]['error']}")
        return report

    for engine_name in requested_engines:
        snapshot = build_large_fx_universe_snapshot(
            base_ccy=effective_currencies[0],
            currencies=effective_currencies,
            trades_per_ccy=args.trades_per_ccy,
            trade_count=args.trade_count,
            max_maturity_months=args.max_maturity_months,
            mpor_days=effective_mpor_days,
            num_paths=requested_paths[engine_name],
            seed=args.seed,
            asof=benchmark_asof,
        )
        report["engines"][engine_name] = run_engine(
            engine_name,
            snapshot,
            python_market_source=effective_python_market_source,
        )
        if report["engines"][engine_name]["status"] != "ok":
            report["warnings"].append(f"{engine_name} run failed: {report['engines'][engine_name]['error']}")

    return report


def run_engine(
    engine_name: str,
    snapshot: XVASnapshot,
    *,
    python_market_source: str = "synthetic",
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if engine_name == "python":
            if python_market_source == "ore-template":
                return _run_python_with_ore_template(snapshot, started=start)
            engine = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False))
            result = engine.create_session(snapshot).run(return_cubes=False)
            return {
                "status": "ok",
                "runtime_seconds": round(time.perf_counter() - start, 6),
                "path_count": int(snapshot.config.num_paths),
                "pv_total": float(result.pv_total),
                "xva_total": float(result.xva_total),
                "metrics": {
                    "CVA": float(result.xva_by_metric.get("CVA", 0.0)),
                    "DVA": float(result.xva_by_metric.get("DVA", 0.0)),
                    "FBA": float(result.xva_by_metric.get("FBA", 0.0)),
                    "FCA": float(result.xva_by_metric.get("FCA", 0.0)),
                    "FVA": float(result.xva_by_metric.get("FVA", 0.0)),
                    "MVA": float(result.xva_by_metric.get("MVA", 0.0)),
                },
                "exposure_by_netting_set": {key: float(value) for key, value in result.exposure_by_netting_set.items()},
                "metadata": _json_safe(result.metadata),
            }
        elif engine_name == "ore":
            return _run_ore_file_driven(snapshot, started=start)
        else:
            raise ValueError(f"Unsupported engine {engine_name!r}")
    except Exception as exc:
        return {
            "status": "failed",
            "runtime_seconds": round(time.perf_counter() - start, 6),
            "path_count": int(snapshot.config.num_paths),
            "error": str(exc),
        }


def write_report(report: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_report(report: dict[str, Any], output_json: Path) -> None:
    cfg = report["config"]
    print(
        "config "
        f"base={cfg['base_ccy']} "
        f"currencies={len(report['currencies'])} "
        f"trades={report['trade_count']} "
        f"mpor_days={cfg['mpor_days']}"
    )
    for engine_name, payload in report["engines"].items():
        if payload["status"] == "ok":
            metrics = payload["metrics"]
            print(
                f"[{engine_name}] ok "
                f"paths={payload['path_count']} "
                f"runtime_s={payload['runtime_seconds']:.3f} "
                f"pv={payload['pv_total']:.2f} "
                f"cva={metrics['CVA']:.2f} "
                f"dva={metrics['DVA']:.2f} "
                f"fba={metrics['FBA']:.2f} "
                f"fca={metrics['FCA']:.2f}"
            )
        else:
            print(
                f"[{engine_name}] failed "
                f"paths={payload['path_count']} "
                f"runtime_s={payload['runtime_seconds']:.3f} "
                f"error={payload['error']}"
            )
    print(f"output_json={output_json}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    output_json = args.output_json.resolve()
    write_report(report, output_json)
    print_report(report, output_json)
    return 0 if all(item["status"] == "ok" for item in report["engines"].values()) else 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _run_python_with_ore_template(snapshot: XVASnapshot, *, started: float) -> dict[str, Any]:
    prepared = _prepare_ore_case(snapshot)
    try:
        ore_reports_generated = False
        try:
            ore_module = _discover_ore_module()
            params = ore_module.Parameters()
            params.fromFile(str(prepared["ore_xml"]))
            app = ore_module.OREApp(params)
            app.run()
            ore_reports_generated = True
        except Exception:
            ore_reports_generated = False
        loaded = XVALoader.from_files(str(prepared["input_dir"]), ore_file="ore.xml")
        portfolio = snapshot.portfolio
        if prepared["template_case"].name in ("Example_9", "ExposureWithCollateral"):
            portfolio = replace(
                snapshot.portfolio,
                trades=tuple(replace(trade, netting_set=DEFAULT_COUNTERPARTY) for trade in snapshot.portfolio.trades),
            )
        aligned = replace(
            loaded,
            portfolio=portfolio,
            config=replace(
                loaded.config,
                analytics=snapshot.config.analytics,
                num_paths=int(snapshot.config.num_paths),
                params={**loaded.config.params, **snapshot.config.params},
                source_meta=SourceMeta(origin="ore-template-loaded", path=str(prepared["ore_xml"])),
            ),
            source_meta={**loaded.source_meta, "builder": SourceMeta(origin="ore-template-loaded", path=str(prepared["ore_xml"]))},
        )
        engine = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False))
        result = engine.create_session(aligned).run(return_cubes=False)
        metadata = _json_safe(result.metadata)
        metadata.update(
            {
                "adapter": "python-lgm-ore-template-market",
                "template_case": str(prepared["template_case"]),
                "case_root": str(prepared["case_root"]),
                "ore_reports_generated": ore_reports_generated,
                "market_mode": prepared["market_mode"],
                "ore_supported_currencies": list(prepared["supported_currencies"]),
                "ore_requested_currencies": list(prepared["requested_currencies"]),
                "ore_dropped_currencies": list(prepared["dropped_currencies"]),
                "ore_input_trade_ids_sample": list(prepared["trade_ids_sample"]),
            }
        )
        return {
            "status": "ok",
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "path_count": int(aligned.config.num_paths),
            "pv_total": float(result.pv_total),
            "xva_total": float(result.xva_total),
            "metrics": {
                "CVA": float(result.xva_by_metric.get("CVA", 0.0)),
                "DVA": float(result.xva_by_metric.get("DVA", 0.0)),
                "FBA": float(result.xva_by_metric.get("FBA", 0.0)),
                "FCA": float(result.xva_by_metric.get("FCA", 0.0)),
                "FVA": float(result.xva_by_metric.get("FVA", 0.0)),
                "MVA": float(result.xva_by_metric.get("MVA", 0.0)),
            },
            "exposure_by_netting_set": {key: float(value) for key, value in result.exposure_by_netting_set.items()},
            "metadata": metadata,
        }
    finally:
        shutil.rmtree(prepared["temp_root"], ignore_errors=True)


def _run_ore_file_driven(snapshot: XVASnapshot, *, started: float) -> dict[str, Any]:
    ore_module = _discover_ore_module()
    prepared = _prepare_ore_case(snapshot)
    try:
        params = ore_module.Parameters()
        params.fromFile(str(prepared["ore_xml"]))
        app = ore_module.OREApp(params)
        app.run()
        reports = list(app.getReportNames()) if hasattr(app, "getReportNames") else []
        errors = [str(item) for item in list(app.getErrors())] if hasattr(app, "getErrors") else []
        parsed = _parse_ore_output(prepared["output_dir"])
        metadata = {
            "adapter": "ore-file-driven-template",
            "template_case": str(prepared["template_case"]),
            "case_root": str(prepared["case_root"]),
            "reported_analytics": reports,
            "reported_errors": errors,
            "ore_supported_currencies": list(prepared["supported_currencies"]),
            "ore_requested_currencies": list(prepared["requested_currencies"]),
            "ore_dropped_currencies": list(prepared["dropped_currencies"]),
            "ore_trade_count": int(prepared["trade_count"]),
            "ore_input_trade_ids_sample": list(prepared["trade_ids_sample"]),
            "market_mode": prepared["market_mode"],
            "warnings": list(prepared["warnings"]),
        }
        return {
            "status": "ok",
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "path_count": int(snapshot.config.num_paths),
            "pv_total": float(parsed["pv_total"]),
            "xva_total": float(parsed["xva_total"]),
            "metrics": parsed["metrics"],
            "exposure_by_netting_set": parsed["exposure_by_netting_set"],
            "metadata": _json_safe(metadata),
        }
    finally:
        shutil.rmtree(prepared["temp_root"], ignore_errors=True)


def _discover_ore_module() -> Any:
    adapter = ORESwigAdapter()
    return adapter._module


def _prepare_ore_case(snapshot: XVASnapshot) -> dict[str, Any]:
    requested_currencies = _snapshot_currencies(snapshot)
    base = snapshot.config.base_currency.upper()
    template_case = _template_case_path(base)
    supported_currencies = _template_supported_currencies(template_case)
    retained_currencies = tuple(ccy for ccy in requested_currencies if ccy in supported_currencies)
    dropped_currencies = tuple(ccy for ccy in requested_currencies if ccy not in supported_currencies)
    if len(retained_currencies) < 2:
        raise RuntimeError(
            f"ORE template {template_case.name} does not support enough currencies for base {base}: "
            f"supported={supported_currencies}"
        )

    filtered_snapshot = _filter_snapshot_for_ore(snapshot, retained_currencies)
    mapped = map_snapshot(filtered_snapshot)

    temp_root = Path(tempfile.mkdtemp(prefix="large_fx_ore_case_"))
    case_root = temp_root / template_case.name
    input_dir = case_root / "Input"
    output_dir = case_root / "Output"
    shutil.copytree(template_case / "Input", input_dir)
    if template_case.name == "ExposureWithCollateral":
        shared_input = require_engine_repo_root() / "Examples" / "Input"
        for name in (
            "market_20160205.txt",
            "fixings_20160205.txt",
            "curveconfig.xml",
            "conventions.xml",
            "todaysmarket.xml",
            "pricingengine.xml",
        ):
            shutil.copy2(shared_input / name, input_dir / name)
    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio_xml = mapped.xml_buffers["portfolio.xml"]
    if template_case.name in ("Example_9", "ExposureWithCollateral"):
        portfolio_xml = portfolio_xml.replace(
            f"<NettingSetId>{DEFAULT_NETTING_SET}</NettingSetId>",
            f"<NettingSetId>{DEFAULT_COUNTERPARTY}</NettingSetId>",
        )
    (input_dir / "portfolio.xml").write_text(portfolio_xml, encoding="utf-8")

    netting_xml = (
        _build_ore_template_netting_xml(
            netting_set_id=DEFAULT_COUNTERPARTY,
            csa_currency=snapshot.config.base_currency,
            mpor_days=int(snapshot.config.params.get("python.mpor_days", "0") or "0"),
        )
        if template_case.name in ("Example_9", "ExposureWithCollateral")
        else mapped.xml_buffers["netting.xml"]
    )
    (input_dir / "netting.xml").write_text(netting_xml, encoding="utf-8")

    ore_xml_path = input_dir / "ore.xml"
    _rewrite_ore_case_files(
        ore_xml_path=ore_xml_path,
        template_case=template_case,
        input_dir=input_dir,
        output_dir=output_dir,
        base_currency=base,
        num_paths=int(snapshot.config.num_paths),
        analytics=snapshot.config.analytics,
        mpor_days=int(snapshot.config.params.get("python.mpor_days", "0") or "0"),
    )

    return {
        "temp_root": temp_root,
        "case_root": case_root,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "ore_xml": ore_xml_path,
        "template_case": template_case,
        "supported_currencies": retained_currencies,
        "requested_currencies": requested_currencies,
        "dropped_currencies": dropped_currencies,
        "trade_count": len(filtered_snapshot.portfolio.trades),
        "trade_ids_sample": tuple(trade.trade_id for trade in filtered_snapshot.portfolio.trades[:8]),
        "warnings": (
            [f"ORE run filtered unsupported currencies: {', '.join(dropped_currencies)}"] if dropped_currencies else []
        ),
        "market_mode": (
            "example9_fx_xva_market"
            if template_case.name == "Example_9"
            else "exposure_with_collateral_market"
            if template_case.name == "ExposureWithCollateral"
            else "template_case_market"
        ),
    }


def _snapshot_currencies(snapshot: XVASnapshot) -> tuple[str, ...]:
    currencies = {snapshot.config.base_currency.upper()}
    for trade in snapshot.portfolio.trades:
        product = getattr(trade, "product", None)
        pair = getattr(product, "pair", "")
        if isinstance(pair, str) and len(pair) >= 6:
            currencies.add(pair[:3].upper())
            currencies.add(pair[3:6].upper())
    return tuple(sorted(currencies, key=lambda ccy: (ccy != snapshot.config.base_currency.upper(), ccy)))


def _template_case_path(base_currency: str) -> Path:
    case_name = _TEMPLATE_CASE_BY_BASE.get(base_currency.upper(), "flat_EUR_5Y_A")
    engine_root = require_engine_repo_root()
    if case_name == "example9_fx_xva":
        path = engine_root / "Examples" / "ORE-Python" / "Notebooks" / "Example_9"
    elif case_name == "exposure_with_collateral_fx_xva":
        path = engine_root / "Examples" / "ExposureWithCollateral"
    else:
        path = local_parity_artifacts_root() / "multiccy_benchmark_final" / "cases" / case_name
    if not path.exists():
        raise RuntimeError(f"ORE template case not found: {path}")
    return path


def _template_supported_currencies(template_case: Path) -> tuple[str, ...]:
    ore_xml = ET.parse(template_case / "Input" / "ore.xml").getroot()
    market_cfg = ""
    for node in ore_xml.findall("./Setup/Parameter"):
        if node.attrib.get("name") == "marketConfigFile":
            market_cfg = (node.text or "").strip()
            break
    if not market_cfg:
        raise RuntimeError(f"Template case {template_case} has no marketConfigFile in ore.xml")
    market_path = Path(market_cfg)
    if not market_path.is_absolute():
        market_path = (template_case / "Input" / market_path).resolve()
    tm_root = ET.parse(market_path).getroot()
    currencies = sorted(
        {
            node.attrib["currency"].upper()
            for node in tm_root.findall(".//DiscountingCurve")
            if node.attrib.get("currency")
        }
    )
    return tuple(currencies)


def _template_asof_date(template_case: Path | None) -> str:
    if template_case is None:
        return "2026-03-08"
    ore_xml = ET.parse(template_case / "Input" / "ore.xml").getroot()
    for node in ore_xml.findall("./Setup/Parameter"):
        if node.attrib.get("name") == "asofDate":
            text = (node.text or "").strip()
            if text:
                return text
    raise RuntimeError(f"Template case {template_case} has no asofDate in ore.xml")


def _filter_snapshot_for_ore(snapshot: XVASnapshot, currencies: Sequence[str]) -> XVASnapshot:
    allowed = {ccy.upper() for ccy in currencies}
    filtered_trades = tuple(
        trade
        for trade in snapshot.portfolio.trades
        if getattr(getattr(trade, "product", None), "pair", "")[:3].upper() in allowed
        and getattr(getattr(trade, "product", None), "pair", "")[3:6].upper() in allowed
    )
    if not filtered_trades:
        raise RuntimeError("ORE filtering removed all trades from the benchmark portfolio")
    return replace(snapshot, portfolio=replace(snapshot.portfolio, trades=filtered_trades))


def _rewrite_ore_case_files(
    *,
    ore_xml_path: Path,
    template_case: Path,
    input_dir: Path,
    output_dir: Path,
    base_currency: str,
    num_paths: int,
    analytics: Sequence[str],
    mpor_days: int,
) -> None:
    text = ore_xml_path.read_text(encoding="utf-8")
    text = text.replace(str(template_case / "Input"), str(input_dir))
    text = text.replace(str(template_case / "Output"), str(output_dir))
    for name in (
        "market_20160205.txt",
        "fixings_20160205.txt",
        "curveconfig.xml",
        "conventions.xml",
        "todaysmarket.xml",
        "pricingengine.xml",
    ):
        text = text.replace(f"../../Input/{name}", str(input_dir / name))
    text = re.sub(r'(<Parameter name="inputPath">)(.*?)(</Parameter>)', rf"\1{input_dir}\3", text)
    text = re.sub(r'(<Parameter name="outputPath">)(.*?)(</Parameter>)', rf"\1{output_dir}\3", text)
    text = re.sub(
        r'(<Parameter name="portfolioFile">)(.*?)(</Parameter>)',
        rf"\1{input_dir / 'portfolio.xml'}\3",
        text,
    )
    text = re.sub(
        r'(<Parameter name="csaFile">)(.*?)(</Parameter>)',
        rf"\1{input_dir / 'netting.xml'}\3",
        text,
    )
    text = re.sub(
        r'(<Parameter name="simulationConfigFile">)(.*?)(</Parameter>)',
        rf"\1{input_dir / 'simulation.xml'}\3",
        text,
    )
    text = re.sub(
        r'(<Parameter name="baseCurrency">)(.*?)(</Parameter>)',
        rf"\1{base_currency}\3",
        text,
    )
    enabled = {str(item).upper() for item in analytics}
    for name, metric in (("cva", "CVA"), ("dva", "DVA"), ("fva", "FVA"), ("mva", "MVA"), ("colva", "COLVA")):
        text = re.sub(
            rf'(<Parameter name="{name}">)(.*?)(</Parameter>)',
            rf"\1{'Y' if metric in enabled else 'N'}\3",
            text,
        )
    text = re.sub(r'(<Parameter name="cubeFile">)(.*?)(</Parameter>)', r"\1cube.csv.gz\3", text)
    text = re.sub(
        r'(<Parameter name="aggregationScenarioDataFileName">)(.*?)(</Parameter>)',
        r"\1scenariodata.csv.gz\3",
        text,
    )
    text = re.sub(r'(<Parameter name="scenarioFile">)(.*?)(</Parameter>)', r"\1scenariodata.csv.gz\3", text)
    text = re.sub(r'(<Parameter name="rawCubeOutputFile">)(.*?)(</Parameter>)', r"\1rawcube.csv\3", text)
    text = re.sub(r'(<Parameter name="netCubeOutputFile">)(.*?)(</Parameter>)', r"\1netcube.csv\3", text)
    ore_xml_path.write_text(text, encoding="utf-8")

    simulation_xml_path = input_dir / "simulation.xml"
    sim_text = simulation_xml_path.read_text(encoding="utf-8")
    sim_text = re.sub(r"<Samples>\d+</Samples>", f"<Samples>{num_paths}</Samples>", sim_text)
    simulation_xml_path.write_text(sim_text, encoding="utf-8")


def _build_ore_template_netting_xml(*, netting_set_id: str, csa_currency: str, mpor_days: int) -> str:
    return "\n".join(
        [
            '<?xml version="1.0"?>',
            "<NettingSetDefinitions>",
            "  <NettingSet>",
            f"    <NettingSetId>{netting_set_id}</NettingSetId>",
            "    <ActiveCSAFlag>true</ActiveCSAFlag>",
            "    <CSADetails>",
            "      <Bilateral>Bilateral</Bilateral>",
            f"      <CSACurrency>{csa_currency}</CSACurrency>",
            "      <Index>EUR-EONIA</Index>",
            "      <ThresholdPay>0</ThresholdPay>",
            "      <ThresholdReceive>0</ThresholdReceive>",
            f"      <MinimumTransferAmountPay>{int(DEFAULT_SMALL_MTA)}</MinimumTransferAmountPay>",
            f"      <MinimumTransferAmountReceive>{int(DEFAULT_SMALL_MTA)}</MinimumTransferAmountReceive>",
            "      <IndependentAmount>",
            "        <IndependentAmountHeld>0</IndependentAmountHeld>",
            "        <IndependentAmountType>FIXED</IndependentAmountType>",
            "      </IndependentAmount>",
            "      <MarginingFrequency>",
            "        <CallFrequency>1D</CallFrequency>",
            "        <PostFrequency>1D</PostFrequency>",
            "      </MarginingFrequency>",
            f"      <MarginPeriodOfRisk>{max(mpor_days, 0)}D</MarginPeriodOfRisk>",
            "      <CollateralCompoundingSpreadReceive>0.00</CollateralCompoundingSpreadReceive>",
            "      <CollateralCompoundingSpreadPay>0.00</CollateralCompoundingSpreadPay>",
            "      <EligibleCollaterals>",
            "        <Currencies>",
            f"          <Currency>{csa_currency}</Currency>",
            "        </Currencies>",
            "      </EligibleCollaterals>",
            "    </CSADetails>",
            "  </NettingSet>",
            "</NettingSetDefinitions>",
        ]
    )


def _parse_ore_output(output_dir: Path) -> dict[str, Any]:
    npv_rows = _read_csv_rows(output_dir / "npv.csv")
    xva_rows = _read_csv_rows(output_dir / "xva.csv")
    pv_total = 0.0
    for row in npv_rows:
        for key in ("NPV(Base)", "NPV", "PresentValue"):
            value = _csv_float(row.get(key))
            if value is not None:
                pv_total += value
                break

    metrics = {"CVA": 0.0, "DVA": 0.0, "FBA": 0.0, "FCA": 0.0, "FVA": 0.0, "MVA": 0.0}
    for row in xva_rows:
        trade_id = str(row.get("TradeId") or row.get("#TradeId") or "").strip()
        if trade_id:
            continue
        for key in ("CVA", "DVA", "FBA", "FCA", "MVA"):
            value = _csv_float(row.get(key))
            if value is not None:
                metrics[key] = value
    if metrics["FBA"] or metrics["FCA"]:
        metrics["FVA"] = metrics["FBA"] + metrics["FCA"]
    xva_total = sum(metrics.values())

    exposure_by_netting_set: dict[str, float] = {}
    for path in sorted(output_dir.glob("exposure_nettingset_*.csv")):
        rows = _read_csv_rows(path)
        if not rows:
            continue
        netting_set = path.stem.replace("exposure_nettingset_", "")
        epe_values = [
            value
            for row in rows
            for value in (_csv_float(row.get("EPE") or row.get("ExpectedPositiveExposure")),)
            if value is not None
        ]
        basel_values = [
            value
            for row in rows
            for value in (_csv_float(row.get("TimeWeightedBaselEPE") or row.get("BaselEEPE")),)
            if value is not None
        ]
        if basel_values:
            exposure_by_netting_set[netting_set] = max(basel_values)
        elif epe_values:
            exposure_by_netting_set[netting_set] = max(epe_values)

    return {
        "pv_total": pv_total,
        "xva_total": xva_total,
        "metrics": metrics,
        "exposure_by_netting_set": exposure_by_netting_set,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _csv_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "#N/A":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _run_engine_subprocess(
    args: argparse.Namespace,
    engine_name: str,
    *,
    currencies: Sequence[str],
) -> dict[str, Any]:
    wrapper = Path(__file__).resolve().parent / "demos" / "run_large_fx_universe_benchmark.py"
    fd, temp_name = tempfile.mkstemp(prefix=f"large_fx_{engine_name}_", suffix=".json")
    Path(temp_name).unlink(missing_ok=True)
    try:
        import os

        os.close(fd)
    except Exception:
        pass
    temp_output = Path(temp_name)
    cmd = [
        sys.executable,
        str(wrapper),
        "--engine",
        engine_name,
        "--base-ccy",
        str(args.base_ccy),
        "--currencies",
        ",".join(currencies),
        "--trades-per-ccy",
        str(args.trades_per_ccy),
        "--max-maturity-months",
        str(args.max_maturity_months),
        "--mpor-days",
        str(args.mpor_days),
        "--ore-paths",
        str(args.ore_paths),
        "--py-paths",
        str(args.py_paths),
        "--python-market-source",
        str("ore-template" if engine_name == "python" and args.engine == "both" else args.python_market_source),
        "--seed",
        str(args.seed),
        "--output-json",
        str(temp_output),
    ]
    if args.trade_count is not None:
        cmd.extend(["--trade-count", str(args.trade_count)])

    completed = subprocess.run(cmd, capture_output=True, text=True)
    try:
        if completed.returncode != 0 and not temp_output.exists():
            stderr = completed.stderr.strip() or completed.stdout.strip() or f"subprocess failed with code {completed.returncode}"
            return {
                "status": "failed",
                "runtime_seconds": 0.0,
                "path_count": int(args.py_paths if engine_name == "python" else args.ore_paths),
                "error": stderr,
            }
        payload = json.loads(temp_output.read_text(encoding="utf-8"))
        return payload["engines"][engine_name]
    finally:
        temp_output.unlink(missing_ok=True)
