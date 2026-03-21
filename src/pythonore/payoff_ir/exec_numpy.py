from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

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
    UnaryExpr,
    WhereExpr,
)
from pythonore.payoff_ir.normalize import normalize_module
from pythonore.payoff_ir.types import CashflowEvent, ExecutionResult
from pythonore.payoff_ir.validate import validate_module


@dataclass
class NumpyExecutionEnv:
    parameters: Mapping[str, Any]
    n_paths: int = 1
    index_at: Any = None
    discount: Any = None
    above_prob: Any = None
    below_prob: Any = None
    continuation: Any = None


def _broadcast(value: Any, n_paths: int):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full((n_paths,), float(arr.item()) if arr.dtype.kind in {"i", "f", "b"} else arr.item(), dtype=object if arr.dtype.kind not in {"i", "f", "b"} else float)
    if arr.shape == (n_paths,):
        return arr
    if arr.dtype.kind in {"U", "S", "O"} and arr.shape == ():
        return np.full((n_paths,), arr.item(), dtype=object)
    raise ValueError(f"Cannot broadcast value with shape {arr.shape} to n_paths={n_paths}")


def _as_scalar(value: Any):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 1 and arr.size:
        return arr[0].item() if hasattr(arr[0], "item") else arr[0]
    return value


def _eval_expr(expr: Expr, locals_: Dict[str, Any], env: NumpyExecutionEnv):
    n = int(env.n_paths)
    if isinstance(expr, ConstantExpr):
        return _broadcast(expr.value, n)
    if isinstance(expr, ParamRefExpr):
        return _broadcast(env.parameters[expr.name], n)
    if isinstance(expr, LocalRefExpr):
        return locals_[expr.name]
    if isinstance(expr, UnaryExpr):
        operand = _eval_expr(expr.operand, locals_, env)
        if expr.op == "neg":
            return -operand
        if expr.op == "not":
            return ~np.asarray(operand, dtype=bool)
    if isinstance(expr, (AbsExpr, ExpExpr, LogExpr, SqrtExpr)):
        operand = _eval_expr(expr.operand, locals_, env)
        fn = {AbsExpr: np.abs, ExpExpr: np.exp, LogExpr: np.log, SqrtExpr: np.sqrt}[type(expr)]
        return fn(operand)
    if isinstance(expr, BinaryExpr):
        left = _eval_expr(expr.left, locals_, env)
        right = _eval_expr(expr.right, locals_, env)
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "div":
            return left / right
    if isinstance(expr, CompareExpr):
        left = _eval_expr(expr.left, locals_, env)
        right = _eval_expr(expr.right, locals_, env)
        if expr.op == "eq":
            return left == right
        if expr.op == "ne":
            return left != right
        if expr.op == "lt":
            return left < right
        if expr.op == "le":
            return left <= right
        if expr.op == "gt":
            return left > right
        if expr.op == "ge":
            return left >= right
    if isinstance(expr, BooleanExpr):
        vals = [_eval_expr(x, locals_, env) for x in expr.operands]
        if expr.op == "and":
            out = np.asarray(vals[0], dtype=bool)
            for item in vals[1:]:
                out = out & np.asarray(item, dtype=bool)
            return out
        if expr.op == "or":
            out = np.asarray(vals[0], dtype=bool)
            for item in vals[1:]:
                out = out | np.asarray(item, dtype=bool)
            return out
    if isinstance(expr, MaxExpr):
        vals = [_eval_expr(x, locals_, env) for x in expr.operands]
        out = vals[0]
        for item in vals[1:]:
            out = np.maximum(out, item)
        return out
    if isinstance(expr, MinExpr):
        vals = [_eval_expr(x, locals_, env) for x in expr.operands]
        out = vals[0]
        for item in vals[1:]:
            out = np.minimum(out, item)
        return out
    if isinstance(expr, WhereExpr):
        cond = np.asarray(_eval_expr(expr.condition, locals_, env), dtype=bool)
        a = _eval_expr(expr.if_true, locals_, env)
        b = _eval_expr(expr.if_false, locals_, env)
        return np.where(cond, a, b)
    if isinstance(expr, IndexAtDateExpr):
        if env.index_at is None:
            raise ValueError("index_at callback is required for IndexAtDateExpr")
        index = _as_scalar(_eval_expr(expr.index, locals_, env))
        date = _as_scalar(_eval_expr(expr.date, locals_, env))
        return _broadcast(env.index_at(index, date, env.n_paths), n)
    if isinstance(expr, DiscountFactorExpr):
        if env.discount is None:
            return _broadcast(1.0, n)
        obs_date = _as_scalar(_eval_expr(expr.obs_date, locals_, env))
        pay_date = _as_scalar(_eval_expr(expr.pay_date, locals_, env))
        ccy = _as_scalar(_eval_expr(expr.currency, locals_, env))
        return _broadcast(env.discount(obs_date, pay_date, ccy, env.n_paths), n)
    if isinstance(expr, CashflowValueExpr):
        amount = _eval_expr(expr.amount, locals_, env)
        df = _eval_expr(DiscountFactorExpr(expr.obs_date, expr.pay_date, expr.currency), locals_, env)
        return amount * df
    if isinstance(expr, AboveProbExpr):
        if env.above_prob is None:
            raise ValueError("above_prob callback is required for AboveProbExpr")
        index = _as_scalar(_eval_expr(expr.index, locals_, env))
        start = _as_scalar(_eval_expr(expr.start, locals_, env))
        end = _as_scalar(_eval_expr(expr.end, locals_, env))
        level = _as_scalar(_eval_expr(expr.level, locals_, env))
        return _broadcast(env.above_prob(index, start, end, level, env.n_paths), n)
    if isinstance(expr, BelowProbExpr):
        if env.below_prob is None:
            raise ValueError("below_prob callback is required for BelowProbExpr")
        index = _as_scalar(_eval_expr(expr.index, locals_, env))
        start = _as_scalar(_eval_expr(expr.start, locals_, env))
        end = _as_scalar(_eval_expr(expr.end, locals_, env))
        level = _as_scalar(_eval_expr(expr.level, locals_, env))
        return _broadcast(env.below_prob(index, start, end, level, env.n_paths), n)
    if isinstance(expr, ScheduleSizeExpr):
        sched = _eval_expr(expr.schedule, locals_, env)
        if isinstance(sched, tuple):
            return _broadcast(float(len(sched)), n)
        raise ValueError("ScheduleSizeExpr expects a schedule tuple")
    if isinstance(expr, ScheduleItemExpr):
        sched = _eval_expr(expr.schedule, locals_, env)
        idx = int(_as_scalar(_eval_expr(expr.index, locals_, env)))
        if isinstance(sched, tuple):
            return _broadcast(sched[idx - 1], n)
        raise ValueError("ScheduleItemExpr expects a schedule tuple")
    if isinstance(expr, ContinuationValueExpr):
        if env.continuation is None:
            return _eval_expr(expr.value, locals_, env)
        value = _eval_expr(expr.value, locals_, env)
        features = tuple(_eval_expr(x, locals_, env) for x in expr.features)
        when = None if expr.when is None else _as_scalar(_eval_expr(expr.when, locals_, env))
        return _broadcast(env.continuation(expr.label, value, features, when, env.n_paths), n)
    raise TypeError(f"Unsupported expression {type(expr)!r}")


def _execute_block(stmts, locals_: Dict[str, Any], env: NumpyExecutionEnv, cashflows: list, results: Dict[str, Any]):
    npv = None
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            locals_[stmt.name] = _eval_expr(stmt.expr, locals_, env)
        elif isinstance(stmt, AssignStateStmt):
            locals_[stmt.name] = _eval_expr(stmt.expr, locals_, env)
        elif isinstance(stmt, RequireStmt):
            cond = np.asarray(_eval_expr(stmt.condition, locals_, env), dtype=bool)
            if not bool(np.all(cond)):
                raise ValueError(stmt.message or "RequireStmt failed")
        elif isinstance(stmt, IfStmt):
            cond = np.asarray(_eval_expr(stmt.condition, locals_, env), dtype=bool)
            if bool(np.all(cond)):
                npv = _execute_block(stmt.then_body, locals_, env, cashflows, results)
            elif bool(np.all(~cond)) and stmt.else_body:
                npv = _execute_block(stmt.else_body, locals_, env, cashflows, results)
            else:
                then_locals = dict(locals_)
                else_locals = dict(locals_)
                _execute_block(stmt.then_body, then_locals, env, cashflows, results)
                _execute_block(stmt.else_body, else_locals, env, cashflows, results)
                for key in set(then_locals) & set(else_locals):
                    locals_[key] = np.where(cond, then_locals[key], else_locals[key])
        elif isinstance(stmt, ForEachDateStmt):
            if isinstance(stmt.schedule, ParamRefExpr):
                schedule = env.parameters[stmt.schedule.name]
            elif isinstance(stmt.schedule, LocalRefExpr):
                schedule = locals_[stmt.schedule.name]
            else:
                schedule = _eval_expr(stmt.schedule, locals_, env)
            for i, item in enumerate(schedule, start=1):
                locals_[stmt.loop_var] = _broadcast(item, env.n_paths)
                if stmt.index_var:
                    locals_[stmt.index_var] = _broadcast(i, env.n_paths)
                _execute_block(stmt.body, locals_, env, cashflows, results)
        elif isinstance(stmt, EmitCashflowStmt):
            cashflows.append(CashflowEvent(_eval_expr(stmt.amount, locals_, env), _as_scalar(_eval_expr(stmt.obs_date, locals_, env)), _as_scalar(_eval_expr(stmt.pay_date, locals_, env)), _as_scalar(_eval_expr(stmt.currency, locals_, env)), logged=False, flow_type=stmt.flow_type, leg=stmt.leg, metadata=stmt.metadata))
        elif isinstance(stmt, EmitLoggedCashflowStmt):
            cashflows.append(CashflowEvent(_eval_expr(stmt.amount, locals_, env), _as_scalar(_eval_expr(stmt.obs_date, locals_, env)), _as_scalar(_eval_expr(stmt.pay_date, locals_, env)), _as_scalar(_eval_expr(stmt.currency, locals_, env)), logged=True, flow_type=stmt.flow_type, leg=stmt.leg, metadata=stmt.metadata))
        elif isinstance(stmt, RecordResultStmt):
            results[stmt.name] = _eval_expr(stmt.expr, locals_, env)
        elif isinstance(stmt, SetNpvStmt):
            npv = _eval_expr(stmt.expr, locals_, env)
    return npv


def execute_numpy(module: PayoffModuleIR, env: NumpyExecutionEnv) -> ExecutionResult:
    mod = normalize_module(module)
    validate_module(mod)
    locals_: Dict[str, Any] = {}
    cashflows = []
    results: Dict[str, Any] = {}
    npv = _execute_block(mod.regions, locals_, env, cashflows, results)
    if npv is None:
        raise ValueError("Module execution did not produce an NPV")
    return ExecutionResult(npv=npv, results=results, cashflows=tuple(cashflows), metadata={"backend": "numpy"})
