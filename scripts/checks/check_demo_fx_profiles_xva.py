import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "py_ore_tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from lgm import LGMParams
from lgm_fx_hybrid import MultiCcyLgmParams, LgmFxHybrid
from lgm_fx_xva_utils import FxForwardDef, fx_forward_npv, aggregate_exposure_profile
from irs_xva_utils import build_discount_curve_from_zero_rate_pairs, survival_probability_from_hazard


def run_fx_forward_profile_xva(
    name,
    pair,
    maturity,
    spot0,
    strike,
    notional_base,
    dom_zero_rate,
    for_zero_rate,
    fx_vol=0.12,
    ir_vol_dom=0.010,
    ir_vol_for=0.010,
    corr_dom_fx=0.0,
    corr_for_fx=0.0,
    n_paths=30000,
    seed=2026,
    cpty_hazard=0.015,
    own_hazard=0.010,
    recovery_cpty=0.40,
    recovery_own=0.40,
    funding_spread=0.0015,
):
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

    corr = np.array(
        [
            [1.0, 0.0, corr_for_fx],
            [0.0, 1.0, corr_dom_fx],
            [corr_for_fx, corr_dom_fx, 1.0],
        ],
        dtype=float,
    )

    hybrid = LgmFxHybrid(
        MultiCcyLgmParams(
            ir_params={base: for_params, quote: dom_params},
            fx_vols={pair: ((), (fx_vol,))},
            corr=corr,
        )
    )

    p0_dom = build_discount_curve_from_zero_rate_pairs([(0.0, dom_zero_rate), (maturity + 10.0, dom_zero_rate)])
    p0_for = build_discount_curve_from_zero_rate_pairs([(0.0, for_zero_rate), (maturity + 10.0, for_zero_rate)])

    mu = dom_zero_rate - for_zero_rate
    times = np.linspace(0.0, maturity, int(max(12 * maturity, 12)) + 1)

    paths = hybrid.simulate_paths(
        times,
        n_paths=n_paths,
        rng=np.random.default_rng(seed),
        log_s0={pair: np.log(spot0)},
        rd_minus_rf={pair: mu},
    )

    fx_def = FxForwardDef(
        trade_id=name,
        pair=pair,
        notional_base=notional_base,
        strike=strike,
        maturity=maturity,
    )

    npv_paths = np.zeros((times.size, n_paths), dtype=float)
    for i, t in enumerate(times):
        x_dom_t = paths['x'][quote][i, :]
        x_for_t = paths['x'][base][i, :]
        s_t = paths['s'][pair][i, :]
        if t >= maturity - 1.0e-12:
            npv_paths[i, :] = 0.0
        else:
            npv_paths[i, :] = fx_forward_npv(hybrid, fx_def, float(t), s_t, x_dom_t, x_for_t, p0_dom, p0_for)

    exp = aggregate_exposure_profile(npv_paths)
    ee = exp['ee']
    epe = exp['epe']
    ene = exp['ene']

    df = np.asarray([p0_dom(float(t)) for t in times], dtype=float)

    hz_t = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0], dtype=float)
    c_hz = np.full(hz_t.shape, float(cpty_hazard), dtype=float)
    o_hz = np.full(hz_t.shape, float(own_hazard), dtype=float)
    q_cpty = survival_probability_from_hazard(times, hz_t, c_hz)
    q_own = survival_probability_from_hazard(times, hz_t, o_hz)
    dpd_cpty = np.zeros_like(times)
    dpd_own = np.zeros_like(times)
    dpd_cpty[1:] = np.maximum(q_cpty[:-1] - q_cpty[1:], 0.0)
    dpd_own[1:] = np.maximum(q_own[:-1] - q_own[1:], 0.0)

    lgd_c = 1.0 - float(recovery_cpty)
    lgd_o = 1.0 - float(recovery_own)

    cva_terms = lgd_c * epe * df * dpd_cpty
    dva_terms = lgd_o * (-ene) * df * dpd_own

    dt = np.zeros_like(times)
    dt[1:] = np.diff(times)
    fva_terms = funding_spread * epe * df * dt

    cva = float(np.sum(cva_terms))
    dva = float(np.sum(dva_terms))
    fva = float(np.sum(fva_terms))
    xva_total = cva - dva + fva

    fwd0 = spot0 * p0_for(maturity) / p0_dom(maturity)
    npv0 = notional_base * p0_dom(maturity) * (fwd0 - strike)

    return {
        'name': name,
        'pair': pair,
        'times': times,
        'ee': ee,
        'epe': epe,
        'ene': ene,
        'npv_paths': npv_paths,
        'cva_terms': cva_terms,
        'dva_terms': dva_terms,
        'fva_terms': fva_terms,
        'cva': cva,
        'dva': dva,
        'fva': fva,
        'xva_total': xva_total,
        'fwd0': float(fwd0),
        'npv0': float(npv0),
    }


def main():
    cfgs = [
        dict(
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
            n_paths=25000,
            seed=1234,
        ),
        dict(
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
            n_paths=25000,
            seed=5678,
        ),
    ]

    for cfg in cfgs:
        r = run_fx_forward_profile_xva(**cfg)
        print(f"{r['name']} ({r['pair']})")
        print(f"  t0 fwd={r['fwd0']:.6f}  t0 npv={r['npv0']:,.2f}")
        print(f"  CVA={r['cva']:,.2f}  DVA={r['dva']:,.2f}  FVA={r['fva']:,.2f}  Total={r['xva_total']:,.2f}")
        print(f"  Peak EPE={np.max(r['epe']):,.2f}  Peak ENE={np.min(r['ene']):,.2f}")


if __name__ == '__main__':
    main()
