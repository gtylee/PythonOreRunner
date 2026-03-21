from __future__ import annotations

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
