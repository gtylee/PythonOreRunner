from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from pythonore.payoff_ir import (
    BinaryExpr,
    ForEachDateStmt,
    LetStmt,
    LocalRefExpr,
    ParamRefExpr,
    PayoffModuleIR,
    SetNpvStmt,
    diff_modules,
    emit_ore_script,
    lower_ore_script,
    lower_python_payoff,
    normalize_module,
    validate_module,
)
from pythonore.payoff_ir.black_scholes import (
    BlackScholesMonteCarloModel,
    _black_forward_option_npv,
    _build_lnvol_surface_vol_curve,
)
from pythonore.payoff_ir.exec_numpy import NumpyExecutionEnv, execute_numpy
from pythonore.payoff_ir.types import DateScheduleParam, NumberParam
from pythonore.payoff_surface import RestrictedPayoffError


def test_lower_python_payoff_builds_effectful_ir():
    source = """
def payoff(ctx):
    expiry = ctx.event("Expiry")
    settlement = ctx.event("Settlement")
    strike = ctx.number("Strike")
    quantity = ctx.number("Quantity")
    underlying = ctx.index("Underlying")
    payoff_value = ctx.pay(
        ctx.max(underlying.at(expiry) - strike, 0.0) * quantity,
        expiry,
        settlement,
        ctx.currency("PayCcy"),
        leg=1,
        flow_type="OptionFlow",
    )
    ctx.record_result("CurrentNotional", strike * quantity)
    return ctx.set_npv(payoff_value)
"""
    module = lower_python_payoff(source)

    validate_module(module)

    parameter_names = {param.name for param in module.parameters}
    assert {"Expiry", "Settlement", "Strike", "Quantity", "Underlying", "PayCcy"} <= parameter_names
    assert any(isinstance(stmt, LetStmt) and stmt.name.startswith("_cfv_") for stmt in module.regions)
    assert any(type(stmt).__name__ == "EmitCashflowStmt" for stmt in module.regions)
    assert any(type(stmt).__name__ == "RecordResultStmt" and stmt.name == "CurrentNotional" for stmt in module.regions)
    assert isinstance(module.regions[-1], SetNpvStmt)


def test_restricted_loader_rejects_imports():
    source = """
import math

def payoff(ctx):
    return ctx.set_npv(1.0)
"""
    with pytest.raises(RestrictedPayoffError):
        lower_python_payoff(source)


def test_lower_ore_and_emit_roundtrip_simple_pay_script():
    script = """
NUMBER Strike, Quantity;
EVENT Expiry, Settlement;
INDEX Underlying;
CURRENCY PayCcy;
Option = PAY(max(Underlying(Expiry) - Strike, 0), Expiry, Settlement, PayCcy, 1, OptionFlow);
"""
    module = lower_ore_script(
        script,
        npv_variable="Option",
        results=(("CurrentNotional", "Strike"),),
    )

    emitted = emit_ore_script(module)
    assert "PAY(" in emitted
    assert "FOR " not in emitted
    assert "-- NPV" in emitted

    roundtripped = lower_ore_script(emitted, npv_variable="v2")
    validate_module(roundtripped)
    assert "PAY(" in emit_ore_script(roundtripped)


def test_lower_ore_builds_foreach_loop_for_schedule_iteration():
    script = """
NUMBER KnockedOut;
EVENT StartDate;
EVENT BarrierMonitoringDates;
INDEX Underlying;
NUMBER BarrierLevel;
KnockedOut = 0;
FOR d IN (1, SIZE(BarrierMonitoringDates), 1) DO
  IF Underlying(BarrierMonitoringDates[d]) > BarrierLevel THEN
    KnockedOut = 1;
  END;
END;
"""
    module = lower_ore_script(script, npv_variable="KnockedOut")

    loops = [stmt for stmt in module.regions if isinstance(stmt, ForEachDateStmt)]
    assert len(loops) == 1
    assert isinstance(loops[0].schedule, ParamRefExpr)
    assert loops[0].schedule.name == "BarrierMonitoringDates"


def test_lower_ore_hoists_branch_defined_npv_local():
    script = """
NUMBER Flag;
IF Flag == 1 THEN
  value = 10;
ELSE
  value = 20;
END;
"""
    module = lower_ore_script(script, npv_variable="value")

    emitted = emit_ore_script(module)
    assert "IF " in emitted
    assert "-- NPV" in emitted


def test_lower_ore_supports_indexed_assignment_results():
    script = """
NUMBER x, Fixing[SIZE(FixingDates)];
FOR d IN (1, SIZE(FixingDates), 1) DO
  x = 10 * d;
  Fixing[d] = x;
END;
"""
    module = lower_ore_script(script, npv_variable="x", results=(("Fixing", "Fixing"),))
    env = NumpyExecutionEnv(
        parameters={"FixingDates": ("2026-01-01", "2026-02-01")},
        n_paths=1,
    )
    result = execute_numpy(module, env)
    fixing = result.results["Fixing"]
    assert isinstance(fixing, tuple)
    assert len(fixing) == 2
    assert float(fixing[0][0]) == pytest.approx(10.0)
    assert float(fixing[1][0]) == pytest.approx(20.0)
    assert result.metadata["results_t0"]["Fixing"] == pytest.approx((10.0, 20.0))


def test_execute_numpy_preserves_tuple_locals_across_mixed_if_branches():
    script = """
NUMBER x, Fixing[SIZE(FixingDates)];
FOR d IN (1, SIZE(FixingDates), 1) DO
  IF Flag == 1 THEN
    x = 10 * d;
  ELSE
    x = 20 * d;
  END;
  Fixing[d] = x;
END;
"""
    module = lower_ore_script(script, npv_variable="x", results=(("Fixing", "Fixing"),))
    env = NumpyExecutionEnv(
        parameters={
            "FixingDates": ("2026-01-01", "2026-02-01"),
            "Flag": np.array([1.0, 0.0]),
        },
        n_paths=2,
    )
    result = execute_numpy(module, env)
    fixing = result.results["Fixing"]
    assert isinstance(fixing, tuple)
    assert len(fixing) == 2
    assert np.allclose(fixing[0], np.array([10.0, 20.0]))
    assert np.allclose(fixing[1], np.array([20.0, 40.0]))
    assert result.metadata["results_t0"]["Fixing"] == pytest.approx((15.0, 30.0))


def test_lnvol_surface_prefers_same_strike_otm_quote():
    quotes = {
        "EQUITY_OPTION/RATE_LNVOL/IDX/EUR/2024-01-01/100/C": 0.10,
        "EQUITY_OPTION/RATE_LNVOL/IDX/EUR/2024-01-01/100/P": 0.20,
    }
    curve = _build_lnvol_surface_vol_curve(
        quotes=quotes,
        asof=date(2023, 1, 1),
        forward_fn=lambda _: 105.0,
        equity_name="IDX",
        currency="EUR",
        strike=100.0,
        option_type="Call",
    )
    assert curve is not None

    call_curve = _build_lnvol_surface_vol_curve(
        quotes={"EQUITY_OPTION/RATE_LNVOL/IDX/EUR/2024-01-01/100/C": 0.10},
        asof=date(2023, 1, 1),
        forward_fn=lambda _: 105.0,
        equity_name="IDX",
        currency="EUR",
        strike=100.0,
        option_type="Call",
    )
    assert call_curve is not None
    # With K < F, ORE's preferOutOfTheMoney logic selects the put premium at the same strike.
    assert curve(1.0) != pytest.approx(call_curve(1.0))


def test_fd_single_asset_option_price_is_close_to_analytic_for_european():
    model = BlackScholesMonteCarloModel(
        reference_date=date(2023, 1, 1),
        spot=100.0,
        volatility=0.2,
        risk_free_rate=0.01,
        dividend_yield=0.0,
        n_paths=8,
        seed=1,
        observation_dates=("2024-01-01",),
    )
    fd_price = model.fd_single_asset_option_price(
        strike=100.0,
        expiry="2024-01-01",
        settlement="2024-01-01",
        put_call=1.0,
        long_short=1.0,
        quantity=1.0,
        time_grid=100,
        x_grid=100,
    )
    analytic = _black_forward_option_npv(
        forward=model.spot / model._discount_ratio(1.0),
        strike=100.0,
        maturity_time=1.0,
        vol=0.2,
        discount=model._discount_df(1.0),
        call=True,
    )
    assert fd_price == pytest.approx(analytic, rel=2.0e-3)


def test_emit_ore_rejects_non_ore_where_expression():
    module = lower_python_payoff(
        """
def payoff(ctx):
    value = ctx.where(ctx.number("Flag") > 0.0, 1.0, 0.0)
    return ctx.set_npv(value)
"""
    )

    with pytest.raises(ValueError, match="WhereExpr"):
        emit_ore_script(module)


def test_execute_numpy_prices_simple_european_payoff():
    source = """
def payoff(ctx):
    expiry = ctx.event("Expiry")
    settlement = ctx.event("Settlement")
    strike = ctx.number("Strike")
    quantity = ctx.number("Quantity")
    payoff_value = ctx.pay(
        ctx.max(ctx.index("Underlying").at(expiry) - strike, 0.0) * quantity,
        expiry,
        settlement,
        ctx.currency("PayCcy"),
    )
    return ctx.set_npv(payoff_value)
"""
    module = lower_python_payoff(source)
    env = NumpyExecutionEnv(
        parameters={
            "Expiry": "2026-06-01",
            "Settlement": "2026-06-03",
            "Strike": 100.0,
            "Quantity": 2.0,
            "Underlying": "SP5",
            "PayCcy": "USD",
        },
        n_paths=3,
        index_at=lambda index, date, n: np.array([95.0, 105.0, 130.0]),
        discount=lambda obs, pay, ccy, n: np.array([0.99, 0.99, 0.99]),
    )

    result = execute_numpy(module, env)

    assert np.allclose(result.npv, np.array([0.0, 9.9, 59.4]))
    assert len(result.cashflows) == 1
    assert result.cashflows[0].currency == "USD"
    assert result.metadata["npv_t0"] == pytest.approx(23.1)
    assert result.metadata["npv_mc_err_est"] is not None


def test_execute_numpy_uses_model_pay_semantics_for_value_and_cashflows():
    source = """
def payoff(ctx):
    expiry = ctx.event("Expiry")
    amount = ctx.pay(10.0, expiry, expiry, ctx.currency("PayCcy"))
    logged = ctx.logpay(10.0, expiry, expiry, ctx.currency("PayCcy"), leg=1, flow_type="Coupon")
    return ctx.set_npv(amount + logged)
"""
    module = lower_python_payoff(source)
    env = NumpyExecutionEnv(
        parameters={
            "Expiry": "2026-06-01",
            "PayCcy": "USD",
        },
        n_paths=2,
        pay=lambda amount, obs, pay, ccy, n: np.full(n, 7.5),
    )

    result = execute_numpy(module, env)

    assert np.allclose(result.npv, np.array([15.0, 15.0]))
    assert len(result.cashflows) == 2
    assert np.allclose(result.cashflows[0].amount, np.array([7.5, 7.5]))
    assert np.allclose(result.cashflows[1].amount, np.array([7.5, 7.5]))


def test_black_scholes_model_barrier_prob_matches_endpoints():
    model = BlackScholesMonteCarloModel(
        reference_date=date(2023, 6, 5),
        spot=100.0,
        volatility=0.2,
        risk_free_rate=0.0,
        dividend_yield=0.0,
        n_paths=8,
        seed=7,
        observation_dates=("2023-06-06", "2023-06-07"),
    )
    probs = model.above_prob("Underlying", "2023-06-06", "2023-06-07", 90.0, 8)
    assert np.all(probs >= 1.0)


def test_black_scholes_model_builds_numpy_env():
    model = BlackScholesMonteCarloModel(
        reference_date=date(2023, 6, 5),
        spot=100.0,
        volatility=0.2,
        n_paths=16,
        seed=11,
        observation_dates=("2023-06-06", "2023-06-07"),
    )
    source = """
def payoff(ctx):
    expiry = ctx.event("Expiry")
    payoff_value = ctx.pay(
        ctx.max(ctx.index("Underlying").at(expiry) - ctx.number("Strike"), 0.0),
        expiry,
        expiry,
        ctx.currency("PayCcy"),
    )
    return ctx.set_npv(payoff_value)
"""
    module = lower_python_payoff(source)
    result = execute_numpy(
        module,
        model.make_env({"Expiry": "2023-06-07", "Strike": 100.0, "Underlying": "Underlying", "PayCcy": "EUR"}),
    )
    assert result.metadata["npv_t0"] >= 0.0
    assert result.metadata["npv_mc_err_est"] is not None


def test_black_scholes_uses_cumulative_black_variance_increments():
    model = BlackScholesMonteCarloModel(
        reference_date=date(2023, 6, 5),
        spot=100.0,
        n_paths=4,
        seed=1,
        observation_dates=("2023-06-06", "2023-06-07"),
        vol_curve=lambda t: 0.30 if t <= (1.0 / 365.0) else 0.20,
    )
    t1 = 1.0 / 365.0
    t2 = 2.0 / 365.0
    first = model._variance_increment(0.0, t1)
    second = model._variance_increment(t1, t2)
    assert first == pytest.approx(0.30 * 0.30 * t1)
    assert second == pytest.approx(max(0.20 * 0.20 * t2 - 0.30 * 0.30 * t1, 0.0))


def test_validate_module_requires_single_npv():
    module = PayoffModuleIR(
        parameters=(DateScheduleParam(name="FixingDates"),),
        regions=(
            LetStmt("x", BinaryExpr("add", LocalRefExpr("y"), ParamRefExpr("FixingDates"))),
        ),
    )
    with pytest.raises(ValueError, match="Unknown local reference 'y'"):
        validate_module(module)


def test_normalize_module_eliminates_dead_let_and_stabilizes_names():
    module = PayoffModuleIR(
        parameters=(NumberParam(name="A"), NumberParam(name="B")),
        regions=(
            LetStmt("unused", BinaryExpr("add", ParamRefExpr("A"), ParamRefExpr("B"))),
            LetStmt("live_value", BinaryExpr("add", ParamRefExpr("B"), ParamRefExpr("A"))),
            SetNpvStmt(LocalRefExpr("live_value")),
        ),
    )
    normalized = normalize_module(module)

    assert len(normalized.regions) == 2
    assert isinstance(normalized.regions[0], LetStmt)
    assert normalized.regions[0].name == "v1"
    assert normalized.regions[0].expr == BinaryExpr("add", ParamRefExpr("A"), ParamRefExpr("B"))
    assert normalized.regions[1] == SetNpvStmt(LocalRefExpr("v1"))


def test_diff_modules_empty_for_identical_modules():
    module = lower_python_payoff(
        """
def payoff(ctx):
    return ctx.set_npv(ctx.number("X") + 1.0)
"""
    )

    assert diff_modules(module, module) == ""
