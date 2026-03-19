"""Independent checker for demo_lgm_irs_xva notebook calculations.

It reproduces the notebook setup and prints both:
1) BUGGY par-rate/XVA path (missing fixed-leg notional in PV01 denominator)
2) CORRECTED par-rate/XVA path
"""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_ore_tools.lgm import LGM1F, LGMParams, simulate_lgm_measure


def build_model():
    params = LGMParams(
        alpha_times=(0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
        alpha_values=(0.010, 0.013, 0.017, 0.021, 0.019, 0.016, 0.014, 0.011),
        kappa_times=(1.0, 4.0, 8.0),
        kappa_values=(0.040, 0.030, 0.022, 0.018),
        shift=0.0,
        scaling=1.0,
    )
    return LGM1F(params)


def trade_definition():
    return {
        "TradeType": "Swap",
        "Envelope": {"CounterParty": "CPTY_A", "NettingSetId": "CPTY_A"},
        "SwapData": {
            "Start": 0.0,
            "End": 12.0,
            "LegData": [
                {
                    "LegType": "Fixed",
                    "Payer": True,
                    "Currency": "USD",
                    "DayCounter": "ACT/360",
                    "Frequency": "Semiannual",
                    "FrontStub": 0.25,
                    "Notional": 10_000_000.0,
                    "FixedRate": None,
                },
                {
                    "LegType": "Floating",
                    "Payer": False,
                    "Currency": "USD",
                    "Index": "USD-LIBOR-3M",
                    "DayCounter": "ACT/360",
                    "Frequency": "Quarterly",
                    "Notional": 10_000_000.0,
                },
            ],
        },
    }


def remaining_schedule(dates, t):
    return dates[dates > t + 1e-12]


def accruals_from_dates(rem_dates, t):
    if rem_dates.size == 0:
        return rem_dates
    prev = np.concatenate(([t], rem_dates[:-1]))
    return rem_dates - prev


def forward_simple_from_discounts(p_start, p_end, tau):
    return (p_start / p_end - 1.0) / tau


def payer_swap_npv_at_time(model, p0, fixed_dates, float_dates, t, x_t, fixed_rate, trade_def):
    notional_fixed = trade_def["SwapData"]["LegData"][0]["Notional"]
    notional_float = trade_def["SwapData"]["LegData"][1]["Notional"]
    x = np.asarray(x_t, dtype=float)

    rem_fixed = remaining_schedule(fixed_dates, t)
    rem_float = remaining_schedule(float_dates, t)

    if rem_fixed.size == 0 and rem_float.size == 0:
        return np.zeros_like(x, dtype=float)

    p_t = p0(t)

    fixed_leg_pv = np.zeros_like(x, dtype=float)
    if rem_fixed.size > 0:
        p_t_T_fixed = model.discount_bond_paths(
            t,
            rem_fixed,
            x,
            p_t,
            np.fromiter((float(p0(float(T))) for T in rem_fixed), dtype=float, count=rem_fixed.size),
        )
        tau_fixed = accruals_from_dates(rem_fixed, t)
        fixed_leg_pv = notional_fixed * fixed_rate * (tau_fixed @ p_t_T_fixed)

    float_leg_pv = np.zeros_like(x, dtype=float)
    if rem_float.size > 0:
        p_t_T_float = model.discount_bond_paths(
            t,
            rem_float,
            x,
            p_t,
            np.fromiter((float(p0(float(T))) for T in rem_float), dtype=float, count=rem_float.size),
        )
        p_start = np.empty_like(p_t_T_float)
        p_start[0, :] = 1.0
        if p_t_T_float.shape[0] > 1:
            p_start[1:, :] = p_t_T_float[:-1, :]
        float_leg_pv = notional_float * np.sum(p_start - p_t_T_float, axis=0)

    return float_leg_pv - fixed_leg_pv


def main():
    model = build_model()
    r0 = 0.022
    p0 = lambda t: np.exp(-r0 * t)

    trade = trade_definition()
    maturity = trade["SwapData"]["End"]
    fixed_dates = np.concatenate((np.array([trade["SwapData"]["LegData"][0]["FrontStub"]]), np.arange(0.75, maturity + 1e-12, 0.5)))
    float_dates = np.arange(0.25, maturity + 1e-12, 0.25)

    grid_1 = np.arange(0.0, 2.0 + 1e-12, 1.0 / 12.0)
    grid_2 = np.arange(2.25, 5.0 + 1e-12, 0.25)
    grid_3 = np.arange(5.5, maturity + 1e-12, 0.5)
    expo_times = np.unique(np.concatenate((grid_1, grid_2, grid_3, fixed_dates, float_dates)))

    x0_vec = np.array([0.0])
    float_pv_0 = payer_swap_npv_at_time(model, p0, fixed_dates, float_dates, 0.0, x0_vec, 0.0, trade)[0]

    # BUGGY denominator (as in notebook before fix)
    pv01_buggy = np.sum(accruals_from_dates(fixed_dates, 0.0) * np.array([p0(T) for T in fixed_dates]))
    k_par_buggy = float_pv_0 / pv01_buggy

    # CORRECT denominator
    notional_fixed = trade["SwapData"]["LegData"][0]["Notional"]
    pv01_correct = notional_fixed * np.sum(
        accruals_from_dates(fixed_dates, 0.0) * np.array([p0(T) for T in fixed_dates])
    )
    k_par_correct = float_pv_0 / pv01_correct

    print("K_par_buggy:  ", k_par_buggy)
    print("K_par_correct:", k_par_correct)

    n_paths = 30000
    x_paths = simulate_lgm_measure(model, expo_times, n_paths=n_paths, rng=np.random.default_rng(42), x0=0.0)

    def xva_for_rate(k_rate):
        npv_paths = np.zeros_like(x_paths)
        for i, t in enumerate(expo_times):
            npv_paths[i] = payer_swap_npv_at_time(model, p0, fixed_dates, float_dates, t, x_paths[i], k_rate, trade)

        ee = np.mean(np.maximum(npv_paths, 0.0), axis=1)
        ene = np.mean(np.maximum(-npv_paths, 0.0), axis=1)

        lambda_cpty = 0.020
        lambda_own = 0.015
        lgd = 0.60
        fspread = 0.005

        dt = np.diff(expo_times)
        df = np.exp(-r0 * expo_times)
        d_pd_c = np.exp(-lambda_cpty * expo_times[:-1]) - np.exp(-lambda_cpty * expo_times[1:])
        d_pd_b = np.exp(-lambda_own * expo_times[:-1]) - np.exp(-lambda_own * expo_times[1:])

        cva = lgd * np.sum(df[1:] * ee[1:] * d_pd_c)
        dva = lgd * np.sum(df[1:] * ene[1:] * d_pd_b)
        fva = fspread * np.sum(df[1:] * ee[1:] * dt)
        return cva, dva, fva, cva - dva + fva, np.max(np.abs(npv_paths))

    cva_b, dva_b, fva_b, xva_b, max_npv_b = xva_for_rate(k_par_buggy + 0.0015)
    cva_c, dva_c, fva_c, xva_c, max_npv_c = xva_for_rate(k_par_correct + 0.0015)

    print("\nBUGGY RATE RESULTS")
    print("  max |NPV|:", max_npv_b)
    print("  CVA/DVA/FVA/XVA:", cva_b, dva_b, fva_b, xva_b)

    print("\nCORRECT RATE RESULTS")
    print("  max |NPV|:", max_npv_c)
    print("  CVA/DVA/FVA/XVA:", cva_c, dva_c, fva_c, xva_c)


if __name__ == "__main__":
    main()
