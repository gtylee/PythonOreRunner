from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Set

from pythonore.payoff_ir.ir import (
    AbsExpr,
    AboveProbExpr,
    AssignStateStmt,
    BelowProbExpr,
    BinaryExpr,
    BooleanExpr,
    CashflowValueExpr,
    CompareExpr,
    ConstantExpr,
    ContinuationValueExpr,
    DiscountFactorExpr,
    EmitCashflowStmt,
    EmitLoggedCashflowStmt,
    ExpExpr,
    Expr,
    ForEachDateStmt,
    IfStmt,
    IndexAtDateExpr,
    LetStmt,
    LocalRefExpr,
    LogExpr,
    MaxExpr,
    MinExpr,
    ParamRefExpr,
    PayoffModuleIR,
    RecordResultStmt,
    RequireStmt,
    ScheduleItemExpr,
    ScheduleSizeExpr,
    SetNpvStmt,
    SqrtExpr,
    Statement,
    UnaryExpr,
    WhereExpr,
)


def _walk_expr(expr: Expr) -> Iterable[Expr]:
    yield expr
    if isinstance(expr, (ConstantExpr, ParamRefExpr, LocalRefExpr)):
        return
    if isinstance(expr, (UnaryExpr, AbsExpr, ExpExpr, LogExpr, SqrtExpr)):
        child = expr.operand
        yield from _walk_expr(child)
        return
    if isinstance(expr, (BinaryExpr, CompareExpr)):
        yield from _walk_expr(expr.left)
        yield from _walk_expr(expr.right)
        return
    if isinstance(expr, (BooleanExpr, MinExpr, MaxExpr)):
        for item in expr.operands:
            yield from _walk_expr(item)
        return
    if isinstance(expr, WhereExpr):
        yield from _walk_expr(expr.condition)
        yield from _walk_expr(expr.if_true)
        yield from _walk_expr(expr.if_false)
        return
    if isinstance(expr, IndexAtDateExpr):
        yield from _walk_expr(expr.index)
        yield from _walk_expr(expr.date)
        return
    if isinstance(expr, DiscountFactorExpr):
        yield from _walk_expr(expr.obs_date)
        yield from _walk_expr(expr.pay_date)
        yield from _walk_expr(expr.currency)
        return
    if isinstance(expr, CashflowValueExpr):
        yield from _walk_expr(expr.amount)
        yield from _walk_expr(expr.obs_date)
        yield from _walk_expr(expr.pay_date)
        yield from _walk_expr(expr.currency)
        return
    if isinstance(expr, (AboveProbExpr, BelowProbExpr)):
        yield from _walk_expr(expr.index)
        yield from _walk_expr(expr.start)
        yield from _walk_expr(expr.end)
        yield from _walk_expr(expr.level)
        return
    if isinstance(expr, ScheduleSizeExpr):
        yield from _walk_expr(expr.schedule)
        return
    if isinstance(expr, ScheduleItemExpr):
        yield from _walk_expr(expr.schedule)
        yield from _walk_expr(expr.index)
        return
    if isinstance(expr, ContinuationValueExpr):
        yield from _walk_expr(expr.value)
        for item in expr.features:
            yield from _walk_expr(item)
        if expr.when is not None:
            yield from _walk_expr(expr.when)
        return
    raise TypeError(f"Unhandled expression type {type(expr)!r}")


def _validate_expr(expr: Expr, defined: Set[str], params: Set[str]) -> None:
    for node in _walk_expr(expr):
        if isinstance(node, ParamRefExpr) and node.name not in params:
            raise ValueError(f"Unknown parameter reference '{node.name}'")
        if isinstance(node, LocalRefExpr) and node.name not in defined:
            raise ValueError(f"Unknown local reference '{node.name}'")
        if isinstance(node, ScheduleItemExpr):
            sched = node.schedule
            if not isinstance(sched, (ParamRefExpr, LocalRefExpr)):
                raise ValueError("ScheduleItemExpr.schedule must be a schedule reference")
        if isinstance(node, ScheduleSizeExpr):
            sched = node.schedule
            if not isinstance(sched, (ParamRefExpr, LocalRefExpr)):
                raise ValueError("ScheduleSizeExpr.schedule must be a schedule reference")


def _validate_block(stmts: tuple[Statement, ...], defined: Set[str], params: Set[str], results: Set[str], npv_count: list[int]) -> None:
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            _validate_expr(stmt.expr, defined, params)
            defined.add(stmt.name)
        elif isinstance(stmt, AssignStateStmt):
            if stmt.name not in defined:
                raise ValueError(f"AssignState target '{stmt.name}' is not defined")
            _validate_expr(stmt.expr, defined, params)
        elif isinstance(stmt, IfStmt):
            _validate_expr(stmt.condition, defined, params)
            _validate_block(stmt.then_body, set(defined), params, results, npv_count)
            _validate_block(stmt.else_body, set(defined), params, results, npv_count)
        elif isinstance(stmt, ForEachDateStmt):
            if not isinstance(stmt.schedule, (ParamRefExpr, LocalRefExpr)):
                raise ValueError("ForEachDate schedule must be a deterministic schedule reference")
            if isinstance(stmt.schedule, ParamRefExpr) and stmt.schedule.name not in params:
                raise ValueError(f"Unknown schedule parameter '{stmt.schedule.name}'")
            nested = set(defined)
            nested.add(stmt.loop_var)
            if stmt.index_var:
                nested.add(stmt.index_var)
            for state_name in stmt.state_in:
                if state_name not in defined:
                    raise ValueError(f"ForEachDate state_in '{state_name}' is not defined")
            _validate_block(stmt.body, nested, params, results, npv_count)
        elif isinstance(stmt, EmitCashflowStmt):
            _validate_expr(stmt.amount, defined, params)
            _validate_expr(stmt.obs_date, defined, params)
            _validate_expr(stmt.pay_date, defined, params)
            _validate_expr(stmt.currency, defined, params)
        elif isinstance(stmt, EmitLoggedCashflowStmt):
            _validate_expr(stmt.amount, defined, params)
            _validate_expr(stmt.obs_date, defined, params)
            _validate_expr(stmt.pay_date, defined, params)
            _validate_expr(stmt.currency, defined, params)
        elif isinstance(stmt, RecordResultStmt):
            _validate_expr(stmt.expr, defined, params)
            results.add(stmt.name)
        elif isinstance(stmt, SetNpvStmt):
            _validate_expr(stmt.expr, defined, params)
            npv_count[0] += 1
        elif isinstance(stmt, RequireStmt):
            _validate_expr(stmt.condition, defined, params)
        else:
            raise TypeError(f"Unhandled statement type {type(stmt)!r}")


def validate_module(module: PayoffModuleIR) -> PayoffModuleIR:
    params = {p.name for p in module.parameters}
    results: Set[str] = set()
    npv_count = [0]
    _validate_block(module.regions, set(), params, results, npv_count)
    if npv_count[0] != 1:
        raise ValueError(f"Expected exactly one SetNpvStmt, got {npv_count[0]}")
    declared = {r.name for r in module.results_schema}
    if declared and not results.issubset(declared | {"npv"}):
        missing = sorted(results - declared)
        raise ValueError(f"Results recorded but missing from schema: {missing}")
    return module
