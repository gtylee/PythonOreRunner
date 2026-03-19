import numpy as np
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent / "py_ore_tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from lgm import LGMParams
from lgm_fx_hybrid import MultiCcyLgmParams, LgmFxHybrid
from lgm_fx_xva_utils import FxForwardDef, fx_forward_npv
from irs_xva_utils import build_discount_curve_from_zero_rate_pairs


def run_fx_forward_example(name, pair, maturity, spot0, strike, notional_base,
                           dom_zero_rate, for_zero_rate, fx_vol=0.12,
                           ir_vol_dom=0.010, ir_vol_for=0.010,
                           corr_dom_fx=0.0, corr_for_fx=0.0,
                           n_paths=20000, seed=2026):
    base, quote = pair.split('/')

    dom_params = LGMParams(
        alpha_times=(1.0, 3.0, 7.0),
        alpha_values=(ir_vol_dom, ir_vol_dom, ir_vol_dom, ir_vol_dom),
        kappa_times=(),
        kappa_values=(0.03,),
        shift=0.0,
        scaling=1.0,
    )
    for_params = LGMParams(
        alpha_times=(1.0, 3.0, 7.0),
        alpha_values=(ir_vol_for, ir_vol_for, ir_vol_for, ir_vol_for),
        kappa_times=(),
        kappa_values=(0.03,),
        shift=0.0,
        scaling=1.0,
    )

    corr = np.array([
        [1.0, 0.0, corr_for_fx],
        [0.0, 1.0, corr_dom_fx],
        [corr_for_fx, corr_dom_fx, 1.0],
    ], dtype=float)

    hybrid = LgmFxHybrid(
        MultiCcyLgmParams(
            ir_params={base: for_params, quote: dom_params},
            fx_vols={pair: ((), (fx_vol,))},
            corr=corr,
        )
    )

    p0_dom = build_discount_curve_from_zero_rate_pairs([(0.0, dom_zero_rate), (maturity + 5.0, dom_zero_rate)])
    p0_for = build_discount_curve_from_zero_rate_pairs([(0.0, for_zero_rate), (maturity + 5.0, for_zero_rate)])

    mu = dom_zero_rate - for_zero_rate

    times = np.linspace(0.0, maturity, int(max(12 * maturity, 12)) + 1)
    paths = hybrid.simulate_paths(
        times,
        n_paths=n_paths,
        rng=np.random.default_rng(seed),
        log_s0={pair: np.log(spot0)},
        rd_minus_rf={pair: mu},
    )

    x_dom_t = paths['x'][quote][-1, :]
    x_for_t = paths['x'][base][-1, :]
    s_t = paths['s'][pair][-1, :]

    fx_def = FxForwardDef(
        trade_id=name,
        pair=pair,
        notional_base=notional_base,
        strike=strike,
        maturity=maturity,
    )

    npv_t = fx_forward_npv(hybrid, fx_def, maturity - 1e-8, s_t, x_dom_t, x_for_t, p0_dom, p0_for)

    fwd0 = spot0 * p0_for(maturity) / p0_dom(maturity)
    npv0 = notional_base * p0_dom(maturity) * (fwd0 - strike)

    return {
        'name': name,
        'pair': pair,
        'maturity': maturity,
        'spot0': spot0,
        'strike': strike,
        'fwd0': float(fwd0),
        'npv0': float(npv0),
        'mtm_at_maturity_mean': float(np.mean(npv_t)),
        'mtm_at_maturity_p05': float(np.percentile(npv_t, 5.0)),
        'mtm_at_maturity_p95': float(np.percentile(npv_t, 95.0)),
    }


fx_ex_1 = run_fx_forward_example(
    name='FXFWD_GBPUSD_1Y',
    pair='GBP/USD',
    maturity=1.0,
    spot0=1.2700,
    strike=1.2850,
    notional_base=10_000_000,
    dom_zero_rate=0.0475,
    for_zero_rate=0.0400,
    fx_vol=0.115,
    corr_dom_fx=-0.15,
    corr_for_fx=0.10,
    n_paths=20000,
    seed=1234,
)

fx_ex_2 = run_fx_forward_example(
    name='FXFWD_USDCAD_2Y',
    pair='USD/CAD',
    maturity=2.0,
    spot0=1.3400,
    strike=1.3600,
    notional_base=10_000_000,
    dom_zero_rate=0.0360,
    for_zero_rate=0.0475,
    fx_vol=0.105,
    corr_dom_fx=-0.05,
    corr_for_fx=0.05,
    n_paths=20000,
    seed=5678,
)

for r in (fx_ex_1, fx_ex_2):
    print(f"{r['name']} ({r['pair']}, {r['maturity']}Y)")
    print(f"  spot0={r['spot0']:.6f} strike={r['strike']:.6f} fwd0={r['fwd0']:.6f}")
    print(f"  t0 NPV (deterministic): {r['npv0']:,.2f}")
    print(f"  terminal MTM mean:      {r['mtm_at_maturity_mean']:,.2f}")
    print(f"  terminal MTM P05/P95:   {r['mtm_at_maturity_p05']:,.2f} / {r['mtm_at_maturity_p95']:,.2f}")
