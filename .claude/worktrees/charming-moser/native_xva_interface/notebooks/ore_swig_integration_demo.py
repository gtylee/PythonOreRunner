# Auto-generated from ore_swig_integration_demo.ipynb

# %% [code cell 1]
from pathlib import Path
import os
import sys

repo = Path(__file__).resolve().parents[4]
py_integration = repo / 'Tools' / 'PythonOreRunner'
ore_swig_build = repo / 'ORE-SWIG' / 'build' / 'lib.macosx-10.13-universal2-cpython-313'

# avoid local dataclasses.py shadowing
sys.path = [p for p in sys.path if not p.endswith('native_xva_interface')]
if str(py_integration) not in sys.path:
    sys.path.insert(0, str(py_integration))
if ore_swig_build.exists() and str(ore_swig_build) not in sys.path:
    sys.path.insert(0, str(ore_swig_build))

dyld_parts = [
    str(repo / 'build' / 'OREAnalytics' / 'orea'),
    str(repo / 'build' / 'OREData' / 'ored'),
    str(repo / 'build' / 'QuantExt' / 'qle'),
    str(repo / 'build' / 'QuantLib' / 'ql'),
]
existing = os.environ.get('DYLD_LIBRARY_PATH', '')
os.environ['DYLD_LIBRARY_PATH'] = ':'.join([*dyld_parts, existing]) if existing else ':'.join(dyld_parts)

print('Configured Python + DYLD paths')

# %% [code cell 2]
from native_xva_interface import (
    ConventionsConfig,
    CounterpartyConfig,
    CreditEntityConfig,
    CreditSimulationConfig,
    CrossAssetModelConfig,
    CurveConfig,
    DeterministicToyAdapter,
    FXForward,
    FixingsData,
    MarketData,
    MarketQuote,
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
    XVALoader,
    XVASnapshot,
    map_snapshot,
)

swig_available = False
try:
    import ORE
    swig_available = True
    print('ORE-SWIG module:', ORE.__file__)
except Exception as exc:
    print('ORE-SWIG unavailable in this kernel:', exc)

if not swig_available:
    print('Skipping heavy SWIG notebook flow because ORE-SWIG is unavailable.')
    raise SystemExit(0)

if os.getenv("RUN_ORE_SWIG_DEMOS") != "1":
    print('Skipping heavy SWIG notebook flow (set RUN_ORE_SWIG_DEMOS=1 to enable).')
    raise SystemExit(0)

# %% [code cell 3]
input_dir = repo / 'Examples' / 'XvaRisk' / 'Input'
base = XVALoader.from_files(str(input_dir), ore_file='ore_stress_classic.xml')

# Programmatic portfolio + market, but keep known-good config buffers
market = MarketData(asof=base.market.asof, raw_quotes=base.market.raw_quotes)
portfolio = Portfolio(
    trades=(
        Trade(
            trade_id='PFX1',
            counterparty='CPTY_A',
            netting_set='CPTY_A',
            trade_type='FxForward',
            product=FXForward(pair='EURUSD', notional=1_000_000, strike=1.09, maturity_years=1.0),
        ),
    )
)

hybrid_snapshot = XVASnapshot(
    market=market,
    fixings=base.fixings,
    portfolio=portfolio,
    config=base.config,
    netting=base.netting,
    collateral=base.collateral,
    source_meta=base.source_meta,
)

toy_result = XVAEngine(adapter=DeterministicToyAdapter()).create_session(hybrid_snapshot).run(return_cubes=False)
print('toy xva_total :', toy_result.xva_total)

if swig_available:
    swig_result = XVAEngine(adapter=ORESwigAdapter()).create_session(hybrid_snapshot).run(return_cubes=False)
    print('swig xva_total:', swig_result.xva_total)
    print('swig by_metric:', swig_result.xva_by_metric)
    print('reports      :', sorted(swig_result.reports.keys())[:20])

# %% [code cell 4]
runtime = RuntimeConfig(
    pricing_engine=PricingEngineConfig(model='DiscountedCashflows', npv_engine='DiscountingFxForwardEngine'),
    todays_market=TodaysMarketConfig(market_id='default', discount_curve='EUR-EONIA', fx_pairs=('EURUSD',)),
    curve_configs=(
        CurveConfig(curve_id='EUR-EONIA', currency='EUR', tenors=('1Y', '2Y', '5Y', '10Y')),
        CurveConfig(curve_id='USD-SOFR', currency='USD', tenors=('1Y', '2Y', '5Y', '10Y')),
    ),
    simulation=SimulationConfig(samples=64, seed=42, dates=('1Y', '2Y', '5Y')),
    simulation_market=SimulationMarketConfig(
        base_currency='EUR',
        currencies=('EUR', 'USD'),
        indices=('EUR-ESTER', 'USD-SOFR'),
        default_curve_names=('BANK', 'CPTY_A'),
        fx_pairs=('USDEUR',),
    ),
    cross_asset_model=CrossAssetModelConfig(domestic_ccy='EUR', currencies=('EUR', 'USD'), ir_model_ccys=('EUR', 'USD'), fx_model_ccys=('USD',)),
    xva_analytic=XVAAnalyticConfig(
        exposure_allocation_method='None',
        exposure_observation_model='Disable',
        pfe_quantile=0.95,
        collateral_calculation_type='Symmetric',
        marginal_allocation_limit=1.0,
        netting_set_ids=('CPTY_A',),
        cva_enabled=True,
        dim_enabled=True,
        dim_quantile=0.99,
        dim_horizon_calendar_days=14,
        dim_regression_order=2,
        dim_output_netting_set='CPTY_A',
        dim_output_grid_points='0',
    ),
    credit_simulation=CreditSimulationConfig(
        enabled=True,
        netting_set_ids=('CPTY_A',),
        entities=(CreditEntityConfig(name='CPTY_A'), CreditEntityConfig(name='BANK')),
        paths=64,
        seed=42,
    ),
    conventions=ConventionsConfig(day_counter='A365', calendar='TARGET'),
    counterparties=CounterpartyConfig(ids=('CPTY_A',)),
)

quotes = (
    MarketQuote(date='2015-02-05', key='FX/EUR/USD', value=1.1),
    MarketQuote(date='2015-02-05', key='ZERO/RATE/EUR/1Y', value=0.02),
    MarketQuote(date='2015-02-05', key='ZERO/RATE/USD/1Y', value=0.015),
)

pure_snapshot = XVASnapshot(
    market=MarketData(asof='2015-02-05', raw_quotes=quotes),
    fixings=FixingsData(),
    portfolio=portfolio,
    netting=NettingConfig(netting_sets={'CPTY_A': NettingSet(netting_set_id='CPTY_A', counterparty='CPTY_A', active_csa=True, csa_currency='EUR')}),
    config=XVAConfig(asof='2015-02-05', base_currency='EUR', analytics=('CVA', 'DVA', 'FVA', 'MVA'), num_paths=64, runtime=runtime, params={'market.default': 'default'}),
)

mapped = map_snapshot(pure_snapshot)
print('has simulation.xml     :', 'simulation.xml' in mapped.xml_buffers)
print('has creditsimulation.xml:', 'creditsimulation.xml' in mapped.xml_buffers)
print('simulation contains CAM:', '<CrossAssetModel>' in mapped.xml_buffers['simulation.xml'])
print('simulation contains Mkt:', '<Market>' in mapped.xml_buffers['simulation.xml'])

if swig_available:
    res = XVAEngine(adapter=ORESwigAdapter()).create_session(pure_snapshot).run(return_cubes=False)
    print('pure-native xva_total:', res.xva_total)
    print('pure-native reports  :', sorted(res.reports.keys()))

# %% [code cell 5]
if not swig_available:
    print('SWIG unavailable: parity summary skipped')
else:
    hybrid = locals().get('hybrid_result') or locals().get('swig_result')
    pure = locals().get('pure_result') or locals().get('res')
    if hybrid is None:
        print('Hybrid result not available: run hybrid cell first')
    elif pure is None:
        print('Pure-native result not available: run pure-native cell first')
    else:
        hx = float(hybrid.xva_total)
        px = float(pure.xva_total)
        diff = px - hx
        rel = (abs(diff) / abs(hx)) if hx != 0 else None
        print('hybrid xva_total   :', hx)
        print('pure-native xva_total:', px)
        print('absolute diff      :', diff)
        if rel is None:
            print('relative diff      : n/a (hybrid baseline is zero)')
        else:
            print('relative diff      :', rel)

        has_h_xva = 'xva' in hybrid.reports
        has_p_xva = 'xva' in pure.reports
        print('hybrid has xva report   :', has_h_xva)
        print('pure-native has xva report:', has_p_xva)

        if px != 0.0 and has_p_xva:
            print('PARITY STATUS: pure-native has reached non-zero XVA output')
        else:
            print('PARITY STATUS: pure-native still below hybrid capability (expected currently)')
