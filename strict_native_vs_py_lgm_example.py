from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    _src_s = str(_SRC_ROOT)
    if _src_s not in sys.path:
        sys.path.insert(0, _src_s)

from native_xva_interface import EngineRunError


def run_ore_strict(num_paths: int = 100) -> dict:
    from dataclasses import replace

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
        stress_classic_fixing_lines,
        stress_classic_market_lines,
        stress_classic_native_runtime,
    )

    asof = "2016-02-05"
    quotes = []
    for line in stress_classic_market_lines():
        parts = line.split()
        if len(parts) < 3:
            continue
        date = parts[0]
        if len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        quotes.append(MarketQuote(date=date, key=parts[1], value=float(parts[2])))

    fixings = []
    for line in stress_classic_fixing_lines():
        parts = line.split()
        if len(parts) < 3:
            continue
        date = parts[0]
        if len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        fixings.append(FixingPoint(date=date, index=parts[1], value=float(parts[2])))

    runtime = stress_classic_native_runtime()
    runtime = replace(runtime, simulation=replace(runtime.simulation, samples=num_paths, seed=42, strict_template=True))

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
            num_paths=num_paths,
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

    result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
    return {
        "xva_total": float(result.xva_total),
        "xva_by_metric": dict(result.xva_by_metric),
        "report_count": len(result.reports),
        "has_xva_report": "xva" in result.reports,
    }


def run_py_lgm_fx(num_paths: int = 5000, seed: int = 1234) -> dict:
    from py_ore_tools.lgm_fx_xva_utils import (
        FxForwardDef,
        aggregate_exposure_profile,
        build_discount_curve_from_zero_rate_pairs,
        build_two_ccy_hybrid,
        cva_terms_from_profile,
        fx_forward_npv,
        survival_probability_from_hazard,
    )

    pair = "EUR/USD"
    maturity = 13.9
    spot0 = 1.10
    strike = 1.09
    notional = 1_000_000.0

    hybrid = build_two_ccy_hybrid(
        pair=pair,
        ir_specs={
            "EUR": {"alpha": ((), (0.010,)), "kappa": ((), (0.03,))},
            "USD": {"alpha": ((), (0.010,)), "kappa": ((), (0.03,))},
        },
        fx_vol=((), (0.27,)),
        corr_dom_fx=-0.05,
        corr_for_fx=0.05,
    )

    r_usd = 0.015
    r_eur = 0.010
    p0_dom = build_discount_curve_from_zero_rate_pairs([(0.0, r_usd), (10.0, r_usd)])
    p0_for = build_discount_curve_from_zero_rate_pairs([(0.0, r_eur), (10.0, r_eur)])

    times = np.linspace(0.0, maturity, int(maturity * 12) + 1)
    mu = r_usd - r_eur
    paths = hybrid.simulate_paths(
        times,
        n_paths=num_paths,
        rng=np.random.default_rng(seed),
        log_s0={pair: np.log(spot0)},
        rd_minus_rf={pair: mu},
    )

    fx_def = FxForwardDef(
        trade_id="PFX1",
        pair=pair,
        notional_base=notional,
        strike=strike,
        maturity=maturity,
    )
    npv_paths = np.zeros((times.size, num_paths), dtype=float)
    for idx, time in enumerate(times):
        if time >= maturity - 1.0e-12:
            continue
        npv_paths[idx, :] = fx_forward_npv(
            hybrid=hybrid,
            fx_def=fx_def,
            t=float(time),
            s_t=paths["s"][pair][idx, :],
            x_dom_t=paths["x"]["USD"][idx, :],
            x_for_t=paths["x"]["EUR"][idx, :],
            p0_dom=p0_dom,
            p0_for=p0_for,
        )

    exp = aggregate_exposure_profile(npv_paths)
    epe = exp["epe"]
    ene = exp["ene"]
    df = np.array([p0_dom(float(time)) for time in times], dtype=float)

    hz_t = np.array([0.5, 1.0, 2.0, 5.0], dtype=float)
    q_cpty = survival_probability_from_hazard(times, hz_t, np.full(hz_t.shape, 0.010))
    q_own = survival_probability_from_hazard(times, hz_t, np.full(hz_t.shape, 0.004))

    cva = float(cva_terms_from_profile(times, epe, df, q_cpty, recovery=0.40)["cva"][0])
    dva = float(cva_terms_from_profile(times, -ene, df, q_own, recovery=0.40)["cva"][0])
    dt = np.zeros_like(times)
    dt[1:] = np.diff(times)
    borrowing_spread = 0.0010
    lending_spread = 0.0040
    fva = float(np.sum(borrowing_spread * epe * df * dt) - np.sum(lending_spread * (-ene) * df * dt))
    xva_total = cva + dva + fva

    return {
        "xva_total": xva_total,
        "xva_by_metric": {"CVA": cva, "DVA": dva, "FVA": fva, "MVA": 0.0},
        "report_count": None,
        "has_xva_report": None,
    }


def main() -> int:
    py = run_py_lgm_fx(num_paths=5000, seed=1234)
    try:
        ore = run_ore_strict(num_paths=100)
    except EngineRunError as exc:
        print("ORE-SWIG strict run unavailable:", exc)
        print()
        print("Python LGM Approx")
        print("  xva_total     :", py["xva_total"])
        print("  xva_by_metric :", py["xva_by_metric"])
        return 0

    print("Strict Pure Native ORE-SWIG")
    print("  xva_total     :", ore["xva_total"])
    print("  xva_by_metric :", ore["xva_by_metric"])
    print("  report_count  :", ore["report_count"])
    print("  has_xva_report:", ore["has_xva_report"])
    print()
    print("Python LGM Approx")
    print("  xva_total     :", py["xva_total"])
    print("  xva_by_metric :", py["xva_by_metric"])
    print()
    print("Comparison (PY - ORE)")
    print("  xva_total_diff:", py["xva_total"] - ore["xva_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
