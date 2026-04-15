from datetime import date
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from pythonore.compute.irs_xva_utils import _discount_bond_block
from pythonore.compute.irs_xva_utils import expand_leg_notionals
from pythonore.compute.irs_xva_utils import interpolate_linear_flat
from ore_curve_fit_parity.interpolation import build_cubic_discount_interpolator, build_log_linear_discount_interpolator
from py_ore_tools.irs_xva_utils import (
    build_discount_curve_from_discount_pairs,
    calibrate_float_spreads_from_coupon,
    compute_realized_float_coupons,
    load_ore_default_curve_inputs,
    load_ore_discount_pairs_by_columns,
    load_swap_legs_from_portfolio_root,
    payer_swap_npv_at_time,
    swap_npv_from_ore_legs,
    swap_npv_from_ore_legs_dual_curve,
)
from py_ore_tools.lgm import LGM1F, LGMParams


def _payer_swap_npv_reference(model, p0, fixed_dates, float_dates, t, x_t, fixed_rate, trade_def):
    notional_fixed = trade_def["SwapData"]["LegData"][0]["Notional"]
    notional_float = trade_def["SwapData"]["LegData"][1]["Notional"]

    rem_fixed = fixed_dates[fixed_dates > t + 1.0e-12]
    rem_float = float_dates[float_dates > t + 1.0e-12]

    if rem_fixed.size == 0 and rem_float.size == 0:
        return np.zeros_like(x_t, dtype=float)

    p_t = p0(t)

    fixed_leg_pv = np.zeros_like(x_t, dtype=float)
    if rem_fixed.size > 0:
        p_t_T_fixed = np.array([model.discount_bond(t, T, x_t, p_t, p0(T)) for T in rem_fixed])
        prev = np.concatenate(([t], rem_fixed[:-1]))
        tau_fixed = rem_fixed - prev
        fixed_leg_pv = notional_fixed * fixed_rate * np.sum(tau_fixed[:, None] * p_t_T_fixed, axis=0)

    float_leg_pv = np.zeros_like(x_t, dtype=float)
    if rem_float.size > 0:
        p_t_T_float = np.array([model.discount_bond(t, T, x_t, p_t, p0(T)) for T in rem_float])
        prev = np.concatenate(([t], rem_float[:-1]))
        tau_float = rem_float - prev
        p_start = np.concatenate((np.ones((1, x_t.size)), p_t_T_float[:-1]), axis=0)
        forwards = (p_start / p_t_T_float - 1.0) / tau_float[:, None]
        float_leg_pv = notional_float * np.sum(tau_float[:, None] * forwards * p_t_T_float, axis=0)

    return float_leg_pv - fixed_leg_pv


def _swap_npv_from_ore_legs_reference(model, p0, legs, t, x_t):
    pv = np.zeros_like(x_t, dtype=float)
    p_t = p0(t)

    mask_f = legs["fixed_pay_time"] > t + 1.0e-12
    if np.any(mask_f):
        pay = legs["fixed_pay_time"][mask_f]
        disc = np.array([model.discount_bond(t, T, x_t, p_t, p0(T)) for T in pay])
        cash = legs["fixed_amount"][mask_f]
        pv += np.sum(cash[:, None] * disc, axis=0)

    mask_future = (legs["float_pay_time"] > t + 1.0e-12) & (legs["float_start_time"] >= t - 1.0e-12)
    if np.any(mask_future):
        s = legs["float_start_time"][mask_future]
        e = legs["float_end_time"][mask_future]
        pay = legs["float_pay_time"][mask_future]
        tau = legs["float_accrual"][mask_future]
        n = legs["float_notional"][mask_future]
        sign = legs["float_sign"][mask_future]
        spread = legs["float_spread"][mask_future]

        p_ts = np.array([model.discount_bond(t, T, x_t, p_t, p0(T)) for T in s])
        p_te = np.array([model.discount_bond(t, T, x_t, p_t, p0(T)) for T in e])
        p_tp = np.array([model.discount_bond(t, T, x_t, p_t, p0(T)) for T in pay])
        fwd = (p_ts / p_te - 1.0) / tau[:, None]
        cash = sign[:, None] * n[:, None] * (fwd + spread[:, None]) * tau[:, None]
        pv += np.sum(cash * p_tp, axis=0)

    mask_in_period = (
        (legs["float_pay_time"] > t + 1.0e-12)
        & (legs["float_start_time"] < t - 1.0e-12)
        & (legs["float_end_time"] >= t - 1.0e-12)
    )
    if np.any(mask_in_period):
        pay = legs["float_pay_time"][mask_in_period]
        amount = legs["float_amount"][mask_in_period]
        p_tp = np.array([model.discount_bond(t, T, x_t, p_t, p0(T)) for T in pay])
        pv += np.sum(amount[:, None] * p_tp, axis=0)

    return pv


def _swap_npv_from_ore_legs_dual_curve_reference(
    model,
    p0_disc,
    p0_fwd,
    legs,
    t,
    x_t,
    realized_float_coupon=None,
):
    pv = np.zeros_like(x_t, dtype=float)
    p_t_d = p0_disc(t)

    node_tenors = np.asarray(legs.get("node_tenors", np.array([], dtype=float)), dtype=float)
    use_nodes = node_tenors.size > 0
    p_nodes_d = None
    if use_nodes:
        p_nodes_d = np.array([model.discount_bond(t, t + tau, x_t, p_t_d, p0_disc(t + tau)) for tau in node_tenors])

    def interp_from_nodes(T: float) -> np.ndarray:
        if not use_nodes:
            return model.discount_bond(t, T, x_t, p_t_d, p0_disc(T))
        if T <= t + 1.0e-14:
            return np.ones_like(x_t, dtype=float)
        grid = t + node_tenors
        logp = np.log(np.clip(p_nodes_d, 1.0e-18, None))
        if T <= grid[0]:
            return model.discount_bond(t, T, x_t, p_t_d, p0_disc(T))
        if T >= grid[-1]:
            t1, t2 = grid[-2], grid[-1]
            slope = (logp[-1] - logp[-2]) / max(t2 - t1, 1.0e-12)
            return np.exp(logp[-1] + slope * (T - t2))
        j = int(np.searchsorted(grid, T, side="right"))
        t1, t2 = grid[j - 1], grid[j]
        w = (T - t1) / max(t2 - t1, 1.0e-12)
        return np.exp((1.0 - w) * logp[j - 1] + w * logp[j])

    mask_f = legs["fixed_pay_time"] > t + 1.0e-12
    if np.any(mask_f):
        pay = legs["fixed_pay_time"][mask_f]
        disc = np.array([interp_from_nodes(T) for T in pay])
        cash = legs["fixed_amount"][mask_f]
        pv += np.sum(cash[:, None] * disc, axis=0)

    fix_t = np.asarray(legs.get("float_fixing_time", legs["float_start_time"]), dtype=float)
    pay_all = legs["float_pay_time"]
    live = pay_all > t + 1.0e-12
    if np.any(live):
        s = legs["float_start_time"][live]
        e = legs["float_end_time"][live]
        pay = pay_all[live]
        tau = legs["float_accrual"][live]
        n = legs["float_notional"][live]
        sign = legs["float_sign"][live]
        gearing = np.asarray(legs.get("float_gearing", np.ones_like(legs["float_accrual"])), dtype=float)[live]
        spread = legs["float_spread"][live]
        fixed = fix_t[live] <= t + 1.0e-12

        p_tp_d = np.array([interp_from_nodes(T) for T in pay])
        amount = np.zeros((pay.size, x_t.size), dtype=float)

        if np.any(fixed):
            if realized_float_coupon is not None:
                coupon_fix = realized_float_coupon[live][fixed]
            else:
                coupon_fix = np.tile(legs["float_coupon"][live][fixed][:, None], (1, x_t.size))
            amount[fixed, :] = sign[fixed, None] * n[fixed, None] * coupon_fix * tau[fixed, None]

        if np.any(~fixed):
            s2 = s[~fixed]
            e2 = e[~fixed]
            tau2 = tau[~fixed]
            n2 = n[~fixed]
            sign2 = sign[~fixed]
            gearing2 = gearing[~fixed]
            spread2 = spread[~fixed]
            p_t_f = p0_fwd(t)
            p_ts_f2 = np.array([model.discount_bond(t, T, x_t, p_t_f, p0_fwd(T)) for T in s2])
            p_te_f2 = np.array([model.discount_bond(t, T, x_t, p_t_f, p0_fwd(T)) for T in e2])
            fwd2 = (p_ts_f2 / p_te_f2 - 1.0) / tau2[:, None]
            amount[~fixed, :] = sign2[:, None] * n2[:, None] * (gearing2[:, None] * fwd2 + spread2[:, None]) * tau2[:, None]

        pv += np.sum(amount * p_tp_d, axis=0)

    return pv


class TestIrsXvaUtils(unittest.TestCase):
    def setUp(self):
        self.model = LGM1F(
            LGMParams(
                alpha_times=(1.0, 3.0),
                alpha_values=(0.015, 0.02, 0.012),
                kappa_times=(2.0,),
                kappa_values=(0.03, 0.025),
                shift=0.0,
                scaling=1.0,
            )
        )
        self.p0 = lambda t: float(np.exp(-0.02 * t))
        self.p0_fwd = lambda t: float(np.exp(-0.0175 * t))
        self.x_t = np.linspace(-0.08, 0.08, 17)

    def test_discount_bond_paths_matches_scalar_stack(self):
        t = 0.75
        maturities = np.array([1.0, 1.5, 2.0, 4.0, 5.5], dtype=float)
        scalar = np.array([self.model.discount_bond(t, float(T), self.x_t, self.p0(t), self.p0(float(T))) for T in maturities])
        vectorized = self.model.discount_bond_paths(t, maturities, self.x_t, self.p0(t), self.p0)
        self.assertTrue(np.allclose(vectorized, scalar, rtol=1.0e-12, atol=1.0e-13))

    def test_discount_bond_block_returns_identity_for_matured_points(self):
        t = 0.75
        maturities = np.array([0.25, 0.75, 1.25], dtype=float)
        got = _discount_bond_block(self.model, self.p0, t, maturities, self.x_t, self.p0(t))

        self.assertEqual(got.shape, (3, self.x_t.size))
        np.testing.assert_allclose(got[0], np.ones_like(self.x_t), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(got[1], np.ones_like(self.x_t), rtol=0.0, atol=1.0e-12)
        self.assertTrue(np.all(np.isfinite(got[2])))

    def test_build_discount_curve_from_discount_pairs_extrapolates_in_log_discount_space(self):
        p0 = build_discount_curve_from_discount_pairs([(0.0, 1.0), (1.0, np.exp(-0.02)), (2.0, np.exp(-0.04))])

        self.assertAlmostEqual(p0(0.5), float(np.exp(-0.01)), places=12)
        self.assertAlmostEqual(p0(2.5), float(np.exp(-0.05)), places=12)

    def test_log_linear_discount_helpers_match_ore_tail_extrapolation(self):
        try:
            import QuantLib as ql
        except Exception:  # pragma: no cover
            self.skipTest("QuantLib bindings are required for curve extrapolation parity")

        times = [0.0, 10.0, 20.0]
        dfs = [1.0, float(np.exp(-0.02 * 10.0)), float(np.exp(-0.03 * 20.0))]
        python_interp = build_log_linear_discount_interpolator(times, dfs)
        python_curve = build_discount_curve_from_discount_pairs(list(zip(times, dfs)))

        ref = ql.Date(1, 1, 2020)
        ql.Settings.instance().evaluationDate = ref
        dates = [ref + int(round(365.0 * t)) for t in times]
        q_curve = ql.DiscountCurve(dates, dfs, ql.Actual365Fixed())
        q_curve.enableExtrapolation()

        query_date = ref + int(round(365.0 * 30.0))
        ore_like = float(q_curve.discount(query_date))
        self.assertAlmostEqual(float(python_interp(30.0)), ore_like, places=12)
        self.assertAlmostEqual(float(python_curve(30.0)), ore_like, places=12)

    def test_cubic_discount_interpolator_matches_quantlib_inside_range(self):
        try:
            import QuantLib as ql
        except Exception:  # pragma: no cover
            self.skipTest("QuantLib bindings are required for cubic interpolation parity")

        times = [0.0, 1.0, 2.0, 4.0]
        dfs = [1.0, 0.985, 0.965, 0.925]
        python_cubic = build_cubic_discount_interpolator(times, dfs)
        q_cubic = ql.CubicNaturalSpline(times, dfs)

        for t in [0.25, 0.5, 1.5, 3.0]:
            self.assertAlmostEqual(float(python_cubic(t)), float(q_cubic(t)), places=12)

    def test_interpolate_linear_flat_handles_scalar_vector_and_flat_extrapolation(self):
        x = np.array([1.0, 2.0, 4.0], dtype=float)
        y = np.array([10.0, 20.0, 40.0], dtype=float)

        self.assertAlmostEqual(float(interpolate_linear_flat(0.5, x, y)), 10.0, places=12)
        self.assertAlmostEqual(float(interpolate_linear_flat(1.5, x, y)), 15.0, places=12)
        self.assertAlmostEqual(float(interpolate_linear_flat(5.0, x, y)), 40.0, places=12)

        got = np.asarray(interpolate_linear_flat(np.array([0.5, 1.5, 5.0], dtype=float), x, y), dtype=float)
        np.testing.assert_allclose(got, np.array([10.0, 15.0, 40.0], dtype=float), rtol=0.0, atol=1.0e-12)

    def test_interpolate_linear_flat_supports_log_space_curve_values(self):
        x = np.array([0.0, 1.0, 2.0], dtype=float)
        log_df = np.log(np.array([1.0, np.exp(-0.02), np.exp(-0.04)], dtype=float))

        mid = float(np.exp(interpolate_linear_flat(1.5, x, log_df)))
        left = float(np.exp(interpolate_linear_flat(-0.5, x, log_df)))
        right = float(np.exp(interpolate_linear_flat(3.0, x, log_df)))

        self.assertAlmostEqual(mid, float(np.exp(-0.03)), places=12)
        self.assertAlmostEqual(left, 1.0, places=12)
        self.assertAlmostEqual(right, float(np.exp(-0.04)), places=12)

    def test_load_swap_legs_from_portfolio_root_expands_amortization_schedule(self):
        root = ET.fromstring(
            """
            <Portfolio>
              <Trade id="AMORT_SWAP">
                <TradeType>Swap</TradeType>
                <SwapData>
                  <LegData>
                    <LegType>Fixed</LegType>
                    <Payer>false</Payer>
                    <Currency>GBP</Currency>
                    <Notionals><Notional>3000000</Notional></Notionals>
                    <Amortizations>
                      <AmortizationData>
                        <Type>RelativeToPreviousNotional</Type>
                        <StartDate>2024-10-17</StartDate>
                        <EndDate>2025-10-17</EndDate>
                        <Value>0.0666666666666667</Value>
                      </AmortizationData>
                      <AmortizationData>
                        <Type>RelativeToPreviousNotional</Type>
                        <StartDate>2025-10-17</StartDate>
                        <EndDate>2026-10-17</EndDate>
                        <Value>0.0357142857142857</Value>
                      </AmortizationData>
                    </Amortizations>
                    <DayCounter>ACT/365</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <FixedLegData><Rates><Rate>0.04</Rate></Rates></FixedLegData>
                    <ScheduleData>
                      <Rules>
                        <StartDate>2023-10-18</StartDate>
                        <EndDate>2026-10-17</EndDate>
                        <Tenor>1Y</Tenor>
                        <Calendar>UK</Calendar>
                        <Convention>MF</Convention>
                      </Rules>
                    </ScheduleData>
                  </LegData>
                  <LegData>
                    <LegType>Floating</LegType>
                    <Payer>true</Payer>
                    <Currency>GBP</Currency>
                    <Notionals><Notional>3000000</Notional></Notionals>
                    <Amortizations>
                      <AmortizationData>
                        <Type>RelativeToPreviousNotional</Type>
                        <StartDate>2024-10-17</StartDate>
                        <EndDate>2025-10-17</EndDate>
                        <Value>0.0666666666666667</Value>
                      </AmortizationData>
                      <AmortizationData>
                        <Type>RelativeToPreviousNotional</Type>
                        <StartDate>2025-10-17</StartDate>
                        <EndDate>2026-10-17</EndDate>
                        <Value>0.0357142857142857</Value>
                      </AmortizationData>
                    </Amortizations>
                    <DayCounter>ACT/365</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <ScheduleData>
                      <Rules>
                        <StartDate>2023-10-18</StartDate>
                        <EndDate>2026-10-17</EndDate>
                        <Tenor>1Y</Tenor>
                        <Calendar>UK</Calendar>
                        <Convention>MF</Convention>
                      </Rules>
                    </ScheduleData>
                    <FloatingLegData>
                      <Index>GBP-LIBOR-1Y</Index>
                      <FixingDays>2</FixingDays>
                      <Spreads><Spread>0.0</Spread></Spreads>
                    </FloatingLegData>
                  </LegData>
                </SwapData>
              </Trade>
            </Portfolio>
            """
        )

        legs = load_swap_legs_from_portfolio_root(root, "AMORT_SWAP", "2023-10-18")
        np.testing.assert_allclose(legs["fixed_notional"], np.array([3_000_000.0, 2_800_000.0, 2_700_000.0]))
        np.testing.assert_allclose(legs["float_notional"], np.array([3_000_000.0, 2_800_000.0, 2_700_000.0]))

    def test_load_swap_legs_from_portfolio_root_expands_recurring_amortization_frequency(self):
        root = ET.fromstring(
            """
            <Portfolio>
              <Trade id="AMORT_FREQ_SWAP">
                <TradeType>Swap</TradeType>
                <SwapData>
                  <LegData>
                    <LegType>Fixed</LegType>
                    <Payer>false</Payer>
                    <Currency>EUR</Currency>
                    <Notionals><Notional>3000000</Notional></Notionals>
                    <Amortizations>
                      <AmortizationData>
                        <Type>RelativeToPreviousNotional</Type>
                        <StartDate>2024-10-17</StartDate>
                        <Frequency>1Y</Frequency>
                        <Value>0.10</Value>
                      </AmortizationData>
                    </Amortizations>
                    <DayCounter>ACT/365</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <FixedLegData><Rates><Rate>0.04</Rate></Rates></FixedLegData>
                    <ScheduleData>
                      <Rules>
                        <StartDate>20231018</StartDate>
                        <EndDate>2026-10-17</EndDate>
                        <Tenor>1Y</Tenor>
                        <Calendar>TARGET</Calendar>
                        <Convention>MF</Convention>
                      </Rules>
                    </ScheduleData>
                  </LegData>
                  <LegData>
                    <LegType>Floating</LegType>
                    <Payer>true</Payer>
                    <Currency>EUR</Currency>
                    <Notionals><Notional>3000000</Notional></Notionals>
                    <Amortizations>
                      <AmortizationData>
                        <Type>RelativeToPreviousNotional</Type>
                        <StartDate>2024-10-17</StartDate>
                        <Frequency>1Y</Frequency>
                        <Value>0.10</Value>
                      </AmortizationData>
                    </Amortizations>
                    <DayCounter>ACT/365</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <ScheduleData>
                      <Rules>
                        <StartDate>20231018</StartDate>
                        <EndDate>2026-10-17</EndDate>
                        <Tenor>1Y</Tenor>
                        <Calendar>TARGET</Calendar>
                        <Convention>MF</Convention>
                      </Rules>
                    </ScheduleData>
                    <FloatingLegData>
                      <Index>EUR-ESTR</Index>
                      <FixingDays>2</FixingDays>
                      <Spreads><Spread>0.0</Spread></Spreads>
                    </FloatingLegData>
                  </LegData>
                </SwapData>
              </Trade>
            </Portfolio>
            """
        )

        legs = load_swap_legs_from_portfolio_root(root, "AMORT_FREQ_SWAP", "2023-10-18")
        expected = np.array([3_000_000.0, 2_700_000.0, 2_430_000.0])
        np.testing.assert_allclose(legs["fixed_notional"], expected)
        np.testing.assert_allclose(legs["float_notional"], expected)

    def test_expand_leg_notionals_supports_dated_nodes_with_undated_base(self):
        leg = ET.fromstring(
            """
            <LegData>
              <Notionals>
                <Notional>3000000</Notional>
                <Notional startDate="2025-10-17">2800000</Notional>
                <Notional startDate="20261017">2600000</Notional>
              </Notionals>
            </LegData>
            """
        )

        notionals = expand_leg_notionals(
            leg,
            [date(2024, 10, 18), date(2025, 10, 17), date(2026, 10, 17)],
            [date(2025, 10, 17), date(2026, 10, 17), date(2027, 10, 17)],
        )

        np.testing.assert_allclose(notionals, np.array([3_000_000.0, 2_800_000.0, 2_600_000.0]))

    def test_expand_leg_notionals_keeps_active_periods_on_base_notional_until_future_start_date(self):
        leg = ET.fromstring(
            """
            <LegData>
              <Notionals>
                <Notional>1000000</Notional>
                <Notional startDate="2025-07-01">900000</Notional>
              </Notionals>
            </LegData>
            """
        )

        notionals = expand_leg_notionals(
            leg,
            [date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)],
            [date(2025, 7, 1), date(2026, 1, 1), date(2026, 7, 1)],
        )

        np.testing.assert_allclose(notionals, np.array([1_000_000.0, 900_000.0, 900_000.0]))

    def test_expand_leg_notionals_aligns_nearby_start_dates_to_generated_schedule(self):
        leg = ET.fromstring(
            """
            <LegData>
              <Notionals>
                <Notional>1000000</Notional>
                <Notional startDate="2025-07-04">900000</Notional>
              </Notionals>
            </LegData>
            """
        )

        notionals = expand_leg_notionals(
            leg,
            [date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)],
            [date(2025, 7, 1), date(2026, 1, 1), date(2026, 7, 1)],
        )

        np.testing.assert_allclose(notionals, np.array([1_000_000.0, 900_000.0, 900_000.0]))

    def test_expand_leg_notionals_aligns_nearby_dates_before_boundary(self):
        leg = ET.fromstring(
            """
            <LegData>
              <Notionals>
                <Notional>1000000</Notional>
                <Notional startDate="2025-06-28">900000</Notional>
              </Notionals>
            </LegData>
            """
        )

        notionals = expand_leg_notionals(
            leg,
            [date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)],
            [date(2025, 7, 1), date(2026, 1, 1), date(2026, 7, 1)],
        )

        np.testing.assert_allclose(notionals, np.array([1_000_000.0, 900_000.0, 900_000.0]))

    def test_expand_leg_notionals_large_mid_period_dates_still_roll_forward(self):
        leg = ET.fromstring(
            """
            <LegData>
              <Notionals>
                <Notional>1000000</Notional>
                <Notional startDate="2025-09-15">900000</Notional>
              </Notionals>
            </LegData>
            """
        )

        notionals = expand_leg_notionals(
            leg,
            [date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)],
            [date(2025, 7, 1), date(2026, 1, 1), date(2026, 7, 1)],
        )

        np.testing.assert_allclose(notionals, np.array([1_000_000.0, 1_000_000.0, 900_000.0]))

    def test_load_swap_legs_from_portfolio_root_prefers_explicit_schedule_dates(self):
        root = ET.fromstring(
            """
            <Portfolio>
              <Trade id="DATES_SWAP">
                <TradeType>Swap</TradeType>
                <SwapData>
                  <LegData>
                    <LegType>Fixed</LegType>
                    <Payer>false</Payer>
                    <Currency>USD</Currency>
                    <Notionals>
                      <Notional>1000000</Notional>
                      <Notional startDate="2025-07-04">900000</Notional>
                    </Notionals>
                    <DayCounter>A360</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <FixedLegData><Rates><Rate>0.04</Rate></Rates></FixedLegData>
                    <ScheduleData>
                      <Dates>
                        <Dates>
                          <Date>2025-01-03</Date>
                          <Date>2025-07-03</Date>
                          <Date>2026-01-05</Date>
                        </Dates>
                      </Dates>
                    </ScheduleData>
                  </LegData>
                  <LegData>
                    <LegType>Floating</LegType>
                    <Payer>true</Payer>
                    <Currency>USD</Currency>
                    <Notionals>
                      <Notional>1000000</Notional>
                      <Notional startDate="2025-07-04">900000</Notional>
                    </Notionals>
                    <DayCounter>A360</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <ScheduleData>
                      <Dates>
                        <Dates>
                          <Date>2025-01-03</Date>
                          <Date>2025-07-03</Date>
                          <Date>2026-01-05</Date>
                        </Dates>
                      </Dates>
                    </ScheduleData>
                    <FloatingLegData>
                      <Index>USD-FedFunds</Index>
                      <FixingDays>2</FixingDays>
                      <Spreads><Spread>0.0</Spread></Spreads>
                    </FloatingLegData>
                  </LegData>
                </SwapData>
              </Trade>
            </Portfolio>
            """
        )

        legs = load_swap_legs_from_portfolio_root(root, "DATES_SWAP", "2025-01-03")

        np.testing.assert_allclose(legs["fixed_notional"], np.array([1_000_000.0, 900_000.0]))
        np.testing.assert_allclose(legs["float_notional"], np.array([1_000_000.0, 900_000.0]))

    def test_load_swap_legs_from_portfolio_root_allows_float_float_swaps(self):
        root = ET.fromstring(
            """
            <Portfolio>
              <Trade id="FLOAT_FLOAT">
                <TradeType>Swap</TradeType>
                <SwapData>
                  <LegData>
                    <LegType>Floating</LegType>
                    <Payer>true</Payer>
                    <Currency>USD</Currency>
                    <Notionals>
                      <Notional>1000000</Notional>
                    </Notionals>
                    <DayCounter>ACT/360</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <FloatingLegData>
                      <Index>USD-LIBOR-3M</Index>
                      <FixingDays>2</FixingDays>
                      <Spreads><Spread>0.001</Spread></Spreads>
                    </FloatingLegData>
                    <ScheduleData>
                      <Dates>
                        <Dates>
                          <Date>2024-01-02</Date>
                          <Date>2024-04-02</Date>
                        </Dates>
                      </Dates>
                    </ScheduleData>
                  </LegData>
                  <LegData>
                    <LegType>Floating</LegType>
                    <Payer>false</Payer>
                    <Currency>USD</Currency>
                    <Notionals>
                      <Notional>1000000</Notional>
                    </Notionals>
                    <DayCounter>ACT/360</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <FloatingLegData>
                      <Index>USD-LIBOR-3M</Index>
                      <FixingDays>2</FixingDays>
                      <Spreads><Spread>-0.0005</Spread></Spreads>
                    </FloatingLegData>
                    <ScheduleData>
                      <Dates>
                        <Dates>
                          <Date>2024-01-02</Date>
                          <Date>2024-04-02</Date>
                        </Dates>
                      </Dates>
                    </ScheduleData>
                  </LegData>
                </SwapData>
              </Trade>
            </Portfolio>
            """
        )

        legs = load_swap_legs_from_portfolio_root(root, "FLOAT_FLOAT", "2023-12-29")

        self.assertEqual(np.asarray(legs["fixed_pay_time"], dtype=float).size, 0)
        self.assertEqual(np.asarray(legs["fixed_amount"], dtype=float).size, 0)
        self.assertEqual(np.asarray(legs["float_pay_time"], dtype=float).size, 2)
        self.assertEqual(list(np.asarray(legs["float_index_by_leg"], dtype=object)), ["USD-LIBOR-3M", "USD-LIBOR-3M"])
        np.testing.assert_allclose(np.asarray(legs["float_sign"], dtype=float), np.array([-1.0, 1.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(legs["float_fixing_time"], dtype=float))))

    def test_load_swap_legs_from_portfolio_root_keeps_float_leg_indices_and_drops_live_accrual(self):
        root = ET.fromstring(
            """
            <Portfolio>
              <Trade id="FLOAT_FLOAT_SPLIT">
                <TradeType>Swap</TradeType>
                <SwapData>
                  <LegData>
                    <LegType>Floating</LegType>
                    <Payer>true</Payer>
                    <Currency>USD</Currency>
                    <Notionals>
                      <Notional>1000000</Notional>
                    </Notionals>
                    <DayCounter>ACT/360</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <FloatingLegData>
                      <Index>USD-LIBOR-3M</Index>
                      <FixingDays>2</FixingDays>
                      <Spreads><Spread>0.001</Spread></Spreads>
                    </FloatingLegData>
                    <ScheduleData>
                      <Dates>
                        <Dates>
                          <Date>2024-01-02</Date>
                          <Date>2024-04-02</Date>
                          <Date>2024-07-02</Date>
                        </Dates>
                      </Dates>
                    </ScheduleData>
                  </LegData>
                  <LegData>
                    <LegType>Floating</LegType>
                    <Payer>false</Payer>
                    <Currency>USD</Currency>
                    <Notionals>
                      <Notional>1000000</Notional>
                    </Notionals>
                    <DayCounter>ACT/360</DayCounter>
                    <PaymentConvention>MF</PaymentConvention>
                    <FloatingLegData>
                      <Index>USD-SOFR-3M</Index>
                      <FixingDays>2</FixingDays>
                      <Spreads><Spread>-0.0005</Spread></Spreads>
                    </FloatingLegData>
                    <ScheduleData>
                      <Dates>
                        <Dates>
                          <Date>2024-01-02</Date>
                          <Date>2024-04-02</Date>
                          <Date>2024-07-02</Date>
                        </Dates>
                      </Dates>
                    </ScheduleData>
                  </LegData>
                </SwapData>
              </Trade>
            </Portfolio>
            """
        )

        legs = load_swap_legs_from_portfolio_root(root, "FLOAT_FLOAT_SPLIT", "2024-03-15")

        self.assertEqual(list(np.asarray(legs["float_index_by_leg"], dtype=object)), ["USD-LIBOR-3M", "USD-SOFR-3M"])
        self.assertTrue(np.all(np.asarray(legs["float_start_time"], dtype=float) >= 0.0))
        self.assertEqual(list(np.asarray(legs["float_leg_index"], dtype=int)), [0, 1])
        self.assertEqual(int(np.asarray(legs["float_leg0_count"], dtype=int)[0]), 1)
        self.assertEqual(np.asarray(legs["float_pay_time"], dtype=float).size, 2)

    def test_build_schedule_backward_with_first_date_advances_past_stub(self):
        from pythonore.compute.irs_xva_utils import _build_schedule

        starts, ends, pays = _build_schedule(
            date(2024, 1, 2),
            date(2024, 7, 2),
            "3M",
            "TARGET",
            "F",
            rule="Backward",
            first_date=date(2024, 2, 15),
        )

        self.assertGreaterEqual(len(starts), 2)
        self.assertEqual(starts[0], date(2024, 1, 2))
        self.assertEqual(ends[-1], date(2024, 7, 2))
        self.assertIn(date(2024, 2, 15), list(starts))
        self.assertIn(date(2024, 2, 15), list(ends))
        self.assertEqual(len(starts), len(ends))
        self.assertEqual(len(ends), len(pays))

    def test_build_schedule_forward_with_last_date_advances_past_stub(self):
        from pythonore.compute.irs_xva_utils import _build_schedule

        starts, ends, pays = _build_schedule(
            date(2024, 1, 2),
            date(2024, 7, 2),
            "3M",
            "TARGET",
            "F",
            rule="Forward",
            last_date=date(2024, 5, 15),
        )

        self.assertGreaterEqual(len(starts), 2)
        self.assertEqual(starts[0], date(2024, 1, 2))
        self.assertEqual(ends[-1], date(2024, 7, 2))
        self.assertIn(date(2024, 5, 15), list(ends))
        self.assertEqual(len(starts), len(ends))
        self.assertEqual(len(ends), len(pays))

    def test_day_counter_a365_alias_matches_a365f(self):
        from pythonore.compute.irs_xva_utils import _time_from_dates, _year_fraction

        start = date(2024, 1, 1)
        end = date(2025, 1, 1)
        self.assertAlmostEqual(_year_fraction(start, end, "A365"), _year_fraction(start, end, "A365F"))
        self.assertAlmostEqual(_time_from_dates(start, end, "A365"), _time_from_dates(start, end, "A365F"))

    def test_build_schedule_respects_term_convention_and_end_of_month(self):
        import QuantLib as ql
        from pythonore.compute.irs_xva_utils import _build_schedule

        starts, ends, pays = _build_schedule(
            date(2024, 1, 31),
            date(2024, 5, 31),
            "1M",
            "TARGET",
            "MF",
            term_convention="P",
            end_of_month=True,
            rule="Forward",
        )
        expected = ql.Schedule(
            ql.Date(31, 1, 2024),
            ql.Date(31, 5, 2024),
            ql.Period(1, ql.Months),
            ql.TARGET(),
            ql.ModifiedFollowing,
            ql.Preceding,
            ql.DateGeneration.Forward,
            True,
            ql.Date(),
            ql.Date(),
        )
        expected_dates = [date(d.year(), int(d.month()), d.dayOfMonth()) for d in expected]
        actual_dates = list(starts) + [ends[-1]]
        self.assertEqual(actual_dates, expected_dates)
        self.assertEqual(len(starts), len(ends))
        self.assertEqual(len(ends), len(pays))

    def test_schedule_from_leg_uses_joint_calendar_and_term_convention(self):
        import QuantLib as ql
        from pythonore.compute.irs_xva_utils import _schedule_from_leg

        xml = """\
<LegData>
  <LegType>Floating</LegType>
  <Payer>false</Payer>
  <Currency>USD</Currency>
  <Notionals><Notional>1000000</Notional></Notionals>
  <DayCounter>A360</DayCounter>
  <PaymentConvention>MF</PaymentConvention>
  <FloatingLegData>
    <Index>USD-SOFR</Index>
    <FixingDays>0</FixingDays>
    <Spreads><Spread>0.0</Spread></Spreads>
    <Gearings><Gearing>1.0</Gearing></Gearings>
  </FloatingLegData>
  <ScheduleData>
    <Rules>
      <StartDate>2024-01-31</StartDate>
      <EndDate>2024-05-31</EndDate>
      <Tenor>1M</Tenor>
      <Calendar>US,TARGET</Calendar>
      <Convention>MF</Convention>
      <TermConvention>P</TermConvention>
      <EndOfMonth>true</EndOfMonth>
      <Rule>Forward</Rule>
    </Rules>
  </ScheduleData>
</LegData>
"""
        leg = ET.fromstring(xml)
        starts, ends, pays = _schedule_from_leg(leg, pay_convention="MF")
        joint_calendar = ql.JointCalendar(ql.UnitedStates(ql.UnitedStates.GovernmentBond), ql.TARGET(), ql.JoinHolidays)
        expected = ql.Schedule(
            ql.Date(31, 1, 2024),
            ql.Date(31, 5, 2024),
            ql.Period(1, ql.Months),
            joint_calendar,
            ql.ModifiedFollowing,
            ql.Preceding,
            ql.DateGeneration.Forward,
            True,
            ql.Date(),
            ql.Date(),
        )
        expected_dates = [date(d.year(), int(d.month()), d.dayOfMonth()) for d in expected]
        actual_dates = list(starts) + [ends[-1]]
        self.assertEqual(actual_dates, expected_dates)
        self.assertEqual(len(starts), len(ends))
        self.assertEqual(len(ends), len(pays))

    def test_load_ore_discount_pairs_by_columns_is_case_insensitive_and_supports_ester_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            curves_csv = f"{tmp}/curves.csv"
            with open(curves_csv, "w", encoding="utf-8", newline="") as handle:
                handle.write("Date,EUR-ESTER,usd-sofr\n")
                handle.write("2026-03-08,1.0,1.0\n")
                handle.write("2027-03-08,0.98,0.97\n")

            data = load_ore_discount_pairs_by_columns(curves_csv, ["eur-estr", "USD-SOFR"])

        times_estr, dfs_estr = data["eur-estr"]
        times_sofr, dfs_sofr = data["USD-SOFR"]
        np.testing.assert_allclose(times_estr, np.array([0.0, 1.0]))
        np.testing.assert_allclose(dfs_estr, np.array([1.0, 0.98]))
        np.testing.assert_allclose(times_sofr, np.array([0.0, 1.0]))
        np.testing.assert_allclose(dfs_sofr, np.array([1.0, 0.97]))

    def test_payer_swap_npv_matches_reference_implementation(self):
        trade_def = {
            "SwapData": {
                "LegData": [
                    {"Notional": 5_000_000.0},
                    {"Notional": 5_000_000.0},
                ]
            }
        }
        fixed_dates = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
        float_dates = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], dtype=float)
        fixed_rate = 0.0275

        for t in (0.0, 0.6, 1.3):
            ref = _payer_swap_npv_reference(self.model, self.p0, fixed_dates, float_dates, t, self.x_t, fixed_rate, trade_def)
            got = payer_swap_npv_at_time(self.model, self.p0, fixed_dates, float_dates, t, self.x_t, fixed_rate, trade_def)
            self.assertTrue(np.allclose(got, ref, rtol=1.0e-12, atol=1.0e-11))

    def test_swap_npv_from_ore_legs_matches_reference_implementation(self):
        legs = {
            "fixed_pay_time": np.array([0.5, 1.0, 1.5], dtype=float),
            "fixed_amount": np.array([-12_500.0, -12_500.0, -12_500.0], dtype=float),
            "float_pay_time": np.array([0.5, 1.0, 1.5], dtype=float),
            "float_start_time": np.array([0.0, 0.5, 1.0], dtype=float),
            "float_end_time": np.array([0.5, 1.0, 1.5], dtype=float),
            "float_accrual": np.array([0.5, 0.5, 0.5], dtype=float),
            "float_notional": np.array([1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
            "float_sign": np.array([1.0, 1.0, 1.0], dtype=float),
            "float_spread": np.array([0.0010, 0.0012, 0.0014], dtype=float),
            "float_amount": np.array([0.0, 13_333.0, 0.0], dtype=float),
        }

        for t in (0.0, 0.75):
            ref = _swap_npv_from_ore_legs_reference(self.model, self.p0, legs, t, self.x_t)
            got = swap_npv_from_ore_legs(self.model, self.p0, legs, t, self.x_t)
            self.assertTrue(np.allclose(got, ref, rtol=1.0e-12, atol=1.0e-11))

    def test_swap_npv_from_ore_legs_dual_curve_matches_reference_without_nodes(self):
        legs = {
            "fixed_pay_time": np.array([0.5, 1.0, 1.5, 2.0], dtype=float),
            "fixed_amount": np.array([-11_000.0, -11_200.0, -11_400.0, -11_600.0], dtype=float),
            "float_pay_time": np.array([0.5, 1.0, 1.5, 2.0], dtype=float),
            "float_start_time": np.array([0.0, 0.5, 1.0, 1.5], dtype=float),
            "float_end_time": np.array([0.5, 1.0, 1.5, 2.0], dtype=float),
            "float_accrual": np.array([0.5, 0.5, 0.5, 0.5], dtype=float),
            "float_notional": np.array([1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
            "float_sign": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
            "float_spread": np.array([0.0010, 0.0012, 0.0014, 0.0016], dtype=float),
            "float_coupon": np.array([0.0200, 0.0205, 0.0210, 0.0215], dtype=float),
        }

        for t in (0.0, 0.6, 1.1):
            ref = _swap_npv_from_ore_legs_dual_curve_reference(self.model, self.p0, self.p0_fwd, legs, t, self.x_t)
            got = swap_npv_from_ore_legs_dual_curve(self.model, self.p0, self.p0_fwd, legs, t, self.x_t)
            self.assertTrue(np.allclose(got, ref, rtol=1.0e-12, atol=1.0e-11))

    def test_swap_npv_from_ore_legs_dual_curve_matches_reference_with_nodes_and_fixings(self):
        legs = {
            "fixed_pay_time": np.array([0.5, 1.0, 1.5, 2.5], dtype=float),
            "fixed_amount": np.array([-11_000.0, -11_200.0, -11_400.0, -11_800.0], dtype=float),
            "float_pay_time": np.array([0.5, 1.0, 1.5, 2.5], dtype=float),
            "float_start_time": np.array([0.0, 0.5, 1.0, 1.5], dtype=float),
            "float_end_time": np.array([0.5, 1.0, 1.5, 2.5], dtype=float),
            "float_fixing_time": np.array([0.0, 0.45, 0.95, 1.45], dtype=float),
            "float_accrual": np.array([0.5, 0.5, 0.5, 1.0], dtype=float),
            "float_notional": np.array([1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
            "float_sign": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
            "float_spread": np.array([0.0010, 0.0012, 0.0014, 0.0016], dtype=float),
            "float_coupon": np.array([0.0200, 0.0205, 0.0210, 0.0215], dtype=float),
            "node_tenors": np.array([0.25, 0.5, 1.0, 1.5, 2.5], dtype=float),
        }
        realized_float_coupon = np.vstack(
            [
                np.full(self.x_t.size, 0.0200),
                np.full(self.x_t.size, 0.0205),
                np.linspace(0.0208, 0.0212, self.x_t.size),
                np.linspace(0.0213, 0.0217, self.x_t.size),
            ]
        )

        for t in (0.0, 1.1):
            ref = _swap_npv_from_ore_legs_dual_curve_reference(
                self.model,
                self.p0,
                self.p0_fwd,
                legs,
                t,
                self.x_t,
                realized_float_coupon=realized_float_coupon,
            )
            got = swap_npv_from_ore_legs_dual_curve(
                self.model,
                self.p0,
                self.p0_fwd,
                legs,
                t,
                self.x_t,
                realized_float_coupon=realized_float_coupon,
                use_node_interpolation=True,
            )
            self.assertTrue(np.allclose(got, ref, rtol=1.0e-12, atol=1.0e-11))

    def test_compute_realized_float_coupons_matches_t0_coupon_pv_in_expectation(self):
        legs = {
            "float_start_time": np.array([0.5, 1.0], dtype=float),
            "float_end_time": np.array([1.0, 1.5], dtype=float),
            "float_fixing_time": np.array([0.45, 0.95], dtype=float),
            "float_pay_time": np.array([1.0, 1.5], dtype=float),
            "float_accrual": np.array([0.5, 0.5], dtype=float),
            "float_index_accrual": np.array([0.5, 0.5], dtype=float),
            "float_notional": np.array([1_000_000.0, 1_000_000.0], dtype=float),
            "float_sign": np.array([1.0, 1.0], dtype=float),
            "float_spread": np.array([0.0010, 0.0012], dtype=float),
            "float_coupon": np.array([0.0200, 0.0215], dtype=float),
        }
        sim_times = np.array([0.0, 0.45, 0.95, 1.5], dtype=float)
        x_paths = np.vstack(
            [
                np.zeros(self.x_t.size),
                np.linspace(-0.04, 0.04, self.x_t.size),
                np.linspace(-0.06, 0.06, self.x_t.size),
                np.linspace(-0.08, 0.08, self.x_t.size),
            ]
        )

        coupons = compute_realized_float_coupons(
            self.model,
            self.p0,
            self.p0_fwd,
            legs,
            sim_times,
            x_paths,
        )

        for i, ft in enumerate(legs["float_fixing_time"]):
            j = int(np.searchsorted(sim_times, ft))
            x_fix = x_paths[j, :]
            p_ft_f = self.p0_fwd(float(ft))
            p_t_s_f = self.model.discount_bond(
                float(ft), float(legs["float_start_time"][i]), x_fix, p_ft_f, self.p0_fwd(float(legs["float_start_time"][i]))
            )
            p_t_e_f = self.model.discount_bond(
                float(ft), float(legs["float_end_time"][i]), x_fix, p_ft_f, self.p0_fwd(float(legs["float_end_time"][i]))
            )
            expected = (p_t_s_f / p_t_e_f - 1.0) / float(legs["float_index_accrual"][i])
            expected = expected + float(legs["float_spread"][i])
            np.testing.assert_allclose(coupons[i, :], expected, atol=1.0e-12, rtol=0.0)


    def test_load_ore_default_curve_inputs_converts_cds_spreads_to_hazard(self):
        todaysmarket_xml = """\
<TodaysMarket>
  <DefaultCurves id="default">
    <DefaultCurve name="BANK">Default/USD/BANK_SR_USD</DefaultCurve>
  </DefaultCurves>
</TodaysMarket>
"""
        market_data = """\
20260101 RECOVERY_RATE/RATE/BANK/SR/USD 0.4
20260101 CDS/CREDIT_SPREAD/BANK/SR/USD/1Y 0.012
20260101 CDS/CREDIT_SPREAD/BANK/SR/USD/5Y 0.018
"""
        with tempfile.TemporaryDirectory() as tmp:
            tm = f"{tmp}/todaysmarket.xml"
            md = f"{tmp}/market.txt"
            with open(tm, "w", encoding="utf-8") as f:
                f.write(todaysmarket_xml)
            with open(md, "w", encoding="utf-8") as f:
                f.write(market_data)
            out = load_ore_default_curve_inputs(tm, md, cpty_name="BANK")

        self.assertAlmostEqual(float(out["recovery"]), 0.4)
        self.assertTrue(np.allclose(out["hazard_times"], np.array([1.0, 5.0])))
        self.assertTrue(
            np.allclose(out["hazard_rates"], np.array([0.012 / 0.6, 0.018 / 0.6]))
        )

    def test_load_ore_default_curve_inputs_accepts_production_style_handle_without_legacy_suffix(self):
        todaysmarket_xml = """\
<TodaysMarket>
  <DefaultCurves id="default">
    <DefaultCurve name="BANK">Default/USD/financials-North-Anerica-8</DefaultCurve>
  </DefaultCurves>
</TodaysMarket>
"""
        market_data = """\
20260101 RECOVERY_RATE/RATE/financials-North-Anerica-8/USD 0.4
20260101 HAZARD_RATE/RATE/financials-North-Anerica-8/USD/1Y 0.012
20260101 HAZARD_RATE/RATE/financials-North-Anerica-8/USD/5Y 0.018
"""
        with tempfile.TemporaryDirectory() as tmp:
            tm = f"{tmp}/todaysmarket.xml"
            md = f"{tmp}/market.txt"
            with open(tm, "w", encoding="utf-8") as f:
                f.write(todaysmarket_xml)
            with open(md, "w", encoding="utf-8") as f:
                f.write(market_data)
            out = load_ore_default_curve_inputs(tm, md, cpty_name="BANK")

        self.assertAlmostEqual(float(out["recovery"]), 0.4)
        self.assertEqual(out["reference_name"], "financials-North-Anerica-8")
        self.assertIsNone(out["seniority"])
        self.assertEqual(out["ccy"], "USD")
        self.assertTrue(np.allclose(out["hazard_times"], np.array([1.0, 5.0])))
        self.assertTrue(np.allclose(out["hazard_rates"], np.array([0.012, 0.018])))

    def test_load_ore_default_curve_inputs_accepts_red_pipe_handle(self):
        todaysmarket_xml = """\
<TodaysMarket>
  <DefaultCurves id="default">
    <DefaultCurve name="BANK">Default/USD/RED:007G93|SNRFOR|USD|XR14</DefaultCurve>
  </DefaultCurves>
</TodaysMarket>
"""
        market_data = """\
20260101 RECOVERY_RATE/RATE/007G93/SNRFOR/USD/XR14 0.4
20260101 CDS/CREDIT_SPREAD/007G93/SNRFOR/USD/XR14/1Y 0.012
20260101 CDS/CREDIT_SPREAD/007G93/SNRFOR/USD/XR14/5Y 0.018
"""
        with tempfile.TemporaryDirectory() as tmp:
            tm = f"{tmp}/todaysmarket.xml"
            md = f"{tmp}/market.txt"
            with open(tm, "w", encoding="utf-8") as f:
                f.write(todaysmarket_xml)
            with open(md, "w", encoding="utf-8") as f:
                f.write(market_data)
            out = load_ore_default_curve_inputs(tm, md, cpty_name="BANK")

        self.assertAlmostEqual(float(out["recovery"]), 0.4)
        self.assertEqual(out["reference_name"], "007G93")
        self.assertEqual(out["seniority"], "SNRFOR")
        self.assertEqual(out["ccy"], "USD")
        self.assertEqual(out["tier"], "XR14")
        self.assertTrue(np.allclose(out["hazard_times"], np.array([1.0, 5.0])))
        self.assertTrue(
            np.allclose(out["hazard_rates"], np.array([0.012 / 0.6, 0.018 / 0.6]))
        )

    def test_load_ore_legs_from_flows_keeps_constant_leg_sign_when_coupons_turn_negative(self):
        from py_ore_tools.irs_xva_utils import load_ore_legs_from_flows

        flows_csv = """\
#TradeId,Type,CashflowNo,LegNo,PayDate,FlowType,Amount,Currency,Coupon,Accrual,AccrualStartDate,AccrualEndDate,AccruedAmount,fixingDate,fixingValue,Notional,DiscountFactor,PresentValue,FXRate(Local-Base),PresentValue(Base),BaseCurrency
T1,Swap,1,0,2016-09-01,Interest,100.0,EUR,0.02,0.5,2016-03-01,2016-09-01,0.0,#N/A,#N/A,10000.0,1.0,100.0,1.0,100.0,EUR
T1,Swap,1,1,2016-06-01,InterestProjected,-10.0,EUR,0.004,0.25,2016-03-01,2016-06-01,0.0,2016-02-26,0.004,10000.0,1.0,-10.0,1.0,-10.0,EUR
T1,Swap,2,1,2016-09-01,InterestProjected,5.0,EUR,-0.002,0.25,2016-06-01,2016-09-01,0.0,2016-05-30,-0.002,10000.0,1.0,5.0,1.0,5.0,EUR
"""
        with tempfile.TemporaryDirectory() as tmp:
            fp = f"{tmp}/flows.csv"
            with open(fp, "w", encoding="utf-8") as f:
                f.write(flows_csv)
            legs = load_ore_legs_from_flows(fp, trade_id="T1", asof_date="2016-02-05", time_day_counter="A365F")

        self.assertTrue(np.allclose(legs["float_sign"], np.array([-1.0, -1.0])))

    def test_load_ore_legs_from_flows_uses_exported_accrual_for_index_basis_by_default(self):
        from py_ore_tools.irs_xva_utils import load_ore_legs_from_flows

        flows_csv = """\
#TradeId,Type,CashflowNo,LegNo,PayDate,FlowType,Amount,Currency,Coupon,Accrual,AccrualStartDate,AccrualEndDate,AccruedAmount,fixingDate,fixingValue,Notional,DiscountFactor,PresentValue,FXRate(Local-Base),PresentValue(Base),BaseCurrency
T1,Swap,1,0,2016-09-01,Interest,100.0,EUR,0.02,0.5,2016-03-01,2016-09-01,0.0,#N/A,#N/A,10000.0,1.0,100.0,1.0,100.0,EUR
T1,Swap,1,1,2016-09-01,InterestProjected,50.0,EUR,0.02,0.5000000000,2016-03-01,2016-09-01,0.0,2016-02-26,0.02,10000.0,1.0,50.0,1.0,50.0,EUR
"""
        with tempfile.TemporaryDirectory() as tmp:
            fp = f"{tmp}/flows.csv"
            with open(fp, "w", encoding="utf-8") as f:
                f.write(flows_csv)
            legs = load_ore_legs_from_flows(fp, trade_id="T1", asof_date="2016-02-05", time_day_counter="A365F")

        self.assertAlmostEqual(float(legs["float_accrual"][0]), 0.5)
        self.assertAlmostEqual(float(legs["float_index_accrual"][0]), 0.5)
        self.assertEqual(legs["float_index_day_counter"], "flows_accrual")

    def test_load_ore_legs_from_flows_ignores_placeholder_fixing_dates(self):
        from py_ore_tools.irs_xva_utils import load_ore_legs_from_flows

        flows_csv = """\
#TradeId,Type,CashflowNo,LegNo,PayDate,FlowType,Amount,Currency,Coupon,Accrual,AccrualStartDate,AccrualEndDate,AccruedAmount,fixingDate,fixingValue,Notional,DiscountFactor,PresentValue,FXRate(Local-Base),PresentValue(Base),BaseCurrency
T1,Swap,1,0,2016-09-01,Interest,100.0,EUR,0.02,0.5,2016-03-01,2016-09-01,0.0,#N/A,#N/A,10000.0,1.0,100.0,1.0,100.0,EUR
T1,Swap,1,1,2016-06-01,InterestProjected,-10.0,EUR,0.004,0.25,2016-03-01,2016-06-01,0.0,#N/A,#N/A,10000.0,1.0,-10.0,1.0,-10.0,EUR
T1,Swap,2,1,2016-09-01,InterestProjected,-11.0,EUR,0.0044,0.25,2016-06-01,2016-09-01,0.0,#N/A,#N/A,10000.0,1.0,-11.0,1.0,-11.0,EUR
"""
        with tempfile.TemporaryDirectory() as tmp:
            fp = f"{tmp}/flows.csv"
            with open(fp, "w", encoding="utf-8") as f:
                f.write(flows_csv)
            legs = load_ore_legs_from_flows(fp, trade_id="T1", asof_date="2016-02-05", time_day_counter="A365F")

        self.assertEqual(legs["float_fixing_source"], "accrual_start_fallback")

    def test_load_ore_legs_from_flows_detects_reversed_leg_numbers_via_fixings(self):
        from py_ore_tools.irs_xva_utils import load_ore_legs_from_flows

        flows_csv = """\
#TradeId,Type,CashflowNo,LegNo,PayDate,FlowType,Amount,Currency,Coupon,Accrual,AccrualStartDate,AccrualEndDate,AccruedAmount,fixingDate,fixingValue,Notional,DiscountFactor,PresentValue,FXRate(Local-Base),PresentValue(Base),BaseCurrency
T1,Swap,1,0,2016-06-01,InterestProjected,-10.0,USD,0.004,0.25,2016-03-01,2016-06-01,0.0,2016-02-26,0.004,10000.0,1.0,-10.0,1.0,-10.0,USD
T1,Swap,2,0,2016-09-01,InterestProjected,-11.0,USD,0.0044,0.25,2016-06-01,2016-09-01,0.0,2016-05-30,0.0044,10000.0,1.0,-11.0,1.0,-11.0,USD
T1,Swap,1,1,2016-09-01,Interest,100.0,TRY,0.10,0.5,2016-03-01,2016-09-01,0.0,#N/A,#N/A,20000.0,1.0,100.0,1.0,100.0,USD
"""
        with tempfile.TemporaryDirectory() as tmp:
            fp = f"{tmp}/flows.csv"
            with open(fp, "w", encoding="utf-8") as f:
                f.write(flows_csv)
            legs = load_ore_legs_from_flows(fp, trade_id="T1", asof_date="2016-02-05", time_day_counter="A365F")

        self.assertAlmostEqual(float(legs["fixed_notional"][0]), 20000.0)
        self.assertAlmostEqual(float(legs["float_notional"][0]), 10000.0)
        self.assertEqual(legs["float_fixing_source"], "flows_fixing_date")

    def test_calibrate_float_spreads_from_coupon_keeps_spread_finite_when_coupon_missing_or_nan(self):
        legs = {
            "float_start_time": np.array([0.0, 1.0], dtype=float),
            "float_end_time": np.array([1.0, 2.0], dtype=float),
            "float_accrual": np.array([1.0, 1.0], dtype=float),
            "float_index_accrual": np.array([1.0, 1.0], dtype=float),
            "float_gearing": np.array([2.0, 1.0], dtype=float),
            "float_spread": np.array([0.001, np.nan], dtype=float),
            "float_coupon": np.array([0.0, np.nan], dtype=float),
        }
        p0 = build_discount_curve_from_discount_pairs([(0.0, 1.0), (1.0, 0.98), (2.0, 0.95)])

        out = calibrate_float_spreads_from_coupon(legs, p0, t0=0.0)

        self.assertTrue(np.isfinite(out["float_spread"][0]))
        self.assertAlmostEqual(float(out["float_spread"][0]), 0.001, places=12)
        self.assertEqual(float(out["float_spread"][1]), 0.0)
        self.assertTrue(np.all(np.isfinite(out["float_coupon"])))

    def test_load_swap_legs_from_portfolio_root_carries_float_gearing(self):
        xml = """\
<Portfolio>
  <Trade id="GEAR_SWAP">
    <TradeType>Swap</TradeType>
    <SwapData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>1000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FixedLegData><Rates><Rate>0.02</Rate></Rates></FixedLegData>
        <ScheduleData><Rules><StartDate>2024-01-02</StartDate><EndDate>2024-04-02</EndDate><Tenor>1M</Tenor><Calendar>TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>1000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR</Index>
          <Gearings><Gearing>1.75</Gearing></Gearings>
          <Spreads><Spread>0.0015</Spread></Spreads>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>2024-01-02</StartDate><EndDate>2024-04-02</EndDate><Tenor>1M</Tenor><Calendar>TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>
</Portfolio>
"""
        root = ET.fromstring(xml)
        legs = load_swap_legs_from_portfolio_root(root, "GEAR_SWAP", "2024-01-02")
        self.assertIn("float_gearing", legs)
        self.assertTrue(np.allclose(legs["float_gearing"], np.full_like(legs["float_gearing"], 1.75)))

    def test_load_swap_legs_from_portfolio_root_carries_fx_reset_by_leg(self):
        xml = """\
<Portfolio>
  <Trade id="XCCY_RESET">
    <TradeType>Swap</TradeType>
    <SwapData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>102120250</Notional>
          <FXReset>
            <ForeignCurrency>EUR</ForeignCurrency>
            <ForeignAmount>95000000</ForeignAmount>
            <FXIndex>FX-ECB-EUR-USD</FXIndex>
            <FixingDays>2</FixingDays>
          </FXReset>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR</Index>
          <FixingDays>2</FixingDays>
          <Spreads><Spread>0.0</Spread></Spreads>
          <Gearings><Gearing>1.0</Gearing></Gearings>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>2024-01-02</StartDate><EndDate>2024-07-02</EndDate><Tenor>3M</Tenor><Calendar>US,TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>EUR</Currency>
        <Notionals><Notional>95000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>EUR-EURIBOR-6M</Index>
          <FixingDays>2</FixingDays>
          <Spreads><Spread>0.0</Spread></Spreads>
          <Gearings><Gearing>1.0</Gearing></Gearings>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>2024-01-02</StartDate><EndDate>2024-07-02</EndDate><Tenor>3M</Tenor><Calendar>US,TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>
</Portfolio>
"""
        root = ET.fromstring(xml)
        legs = load_swap_legs_from_portfolio_root(root, "XCCY_RESET", "2024-01-02")
        self.assertIn("float_fx_reset_by_leg", legs)
        self.assertEqual(len(legs["float_fx_reset_by_leg"]), 2)
        self.assertIsNotNone(legs["float_fx_reset_by_leg"][0])
        self.assertIsNone(legs["float_fx_reset_by_leg"][1])
        self.assertEqual(legs["float_fx_reset_by_leg"][0]["foreign_currency"], "EUR")
        self.assertEqual(legs["float_fx_reset_by_leg"][0]["foreign_amount"], 95000000.0)

    def test_load_swap_legs_from_portfolio_root_expands_piecewise_spread_gearing_and_notional(self):
        xml = """\
<Portfolio>
  <Trade id="PIECEWISE_SWAP">
    <TradeType>Swap</TradeType>
    <SwapData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>1000000</Notional>
          <Notional>900000</Notional>
          <Notional>800000</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FixedLegData><Rates><Rate>0.02</Rate></Rates></FixedLegData>
        <ScheduleData><Rules><StartDate>2024-01-02</StartDate><EndDate>2024-04-02</EndDate><Tenor>1M</Tenor><Calendar>TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals>
          <Notional>1000000</Notional>
          <Notional>900000</Notional>
          <Notional>800000</Notional>
        </Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-SOFR</Index>
          <Gearings>
            <Gearing>1.0</Gearing>
            <Gearing>1.5</Gearing>
            <Gearing>2.0</Gearing>
          </Gearings>
          <Spreads>
            <Spread>0.0010</Spread>
            <Spread>0.0020</Spread>
            <Spread>0.0030</Spread>
          </Spreads>
          <FixingDays>0</FixingDays>
        </FloatingLegData>
        <ScheduleData><Rules><StartDate>2024-01-02</StartDate><EndDate>2024-04-02</EndDate><Tenor>1M</Tenor><Calendar>TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule></Rules></ScheduleData>
      </LegData>
    </SwapData>
  </Trade>
</Portfolio>
"""
        root = ET.fromstring(xml)
        legs = load_swap_legs_from_portfolio_root(root, "PIECEWISE_SWAP", "2024-01-02")
        np.testing.assert_allclose(legs["fixed_notional"], np.array([1_000_000.0, 900_000.0, 800_000.0]))
        np.testing.assert_allclose(legs["float_notional"], np.array([1_000_000.0, 900_000.0, 800_000.0]))
        np.testing.assert_allclose(legs["float_gearing"], np.array([1.0, 1.5, 2.0]))
        np.testing.assert_allclose(legs["float_spread"], np.array([0.0010, 0.0020, 0.0030]))

    def test_swap_npv_from_ore_legs_dual_curve_applies_gearing_to_forward_only(self):
        legs = {
            "fixed_pay_time": np.array([], dtype=float),
            "fixed_start_time": np.array([], dtype=float),
            "fixed_end_time": np.array([], dtype=float),
            "fixed_amount": np.array([], dtype=float),
            "float_pay_time": np.array([1.0], dtype=float),
            "float_start_time": np.array([0.0], dtype=float),
            "float_end_time": np.array([1.0], dtype=float),
            "float_accrual": np.array([1.0], dtype=float),
            "float_index_accrual": np.array([1.0], dtype=float),
            "float_notional": np.array([100.0], dtype=float),
            "float_sign": np.array([1.0], dtype=float),
            "float_gearing": np.array([1.5], dtype=float),
            "float_spread": np.array([0.01], dtype=float),
            "float_coupon": np.array([0.0], dtype=float),
            "float_fixing_time": np.array([2.0], dtype=float),
            "float_is_averaged": np.array([False], dtype=bool),
        }

        class _Model:
            def discount_bond(self, t, maturity, x, p_t, p0_maturity):
                return np.full_like(np.asarray(x, dtype=float), float(p0_maturity), dtype=float)

            def discount_bond_paths(self, t, maturities, x_t, p_t, p0_maturities):
                p0_maturities = np.asarray(p0_maturities, dtype=float)
                x_t = np.asarray(x_t, dtype=float)
                return np.tile((p0_maturities / float(p_t))[:, None], (1, x_t.size))

        p0_disc = lambda t: float(np.exp(-0.02 * t))
        p0_fwd = lambda t: float(np.exp(-0.01 * t))
        pv = swap_npv_from_ore_legs_dual_curve(_Model(), p0_disc, p0_fwd, legs, 0.0, np.zeros(4, dtype=float))
        expected_coupon = 1.5 * ((p0_fwd(0.0) / p0_fwd(1.0) - 1.0) / 1.0) + 0.01
        expected = 100.0 * expected_coupon * p0_disc(1.0)
        self.assertTrue(np.allclose(pv, np.full_like(pv, expected), atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
