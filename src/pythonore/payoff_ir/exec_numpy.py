from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from pythonore.payoff_ir.ir import (
    AbsExpr,
    AboveProbExpr,
    AssignItemStmt,
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
    pay: Any = None
    above_prob: Any = None
    below_prob: Any = None
    continuation: Any = None
    reference_date: Any = None
    discount_t0: Any = None
    fx_spot_t0: Any = None
    extract_t0_result: Any = None


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


def _empty_like(value: Any, n_paths: int):
    arr = np.asarray(value)
    if arr.ndim == 0:
        if arr.dtype.kind in {"i", "f", "b"}:
            return np.zeros(n_paths, dtype=float)
        return np.full((n_paths,), None, dtype=object)
    return np.zeros_like(arr)


def _merge_branch_value(cond: Any, left: Any, right: Any):
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("Cannot merge tuple locals with different lengths")
        return tuple(_merge_branch_value(cond, l, r) for l, r in zip(left, right))
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Cannot merge list locals with different lengths")
        return [_merge_branch_value(cond, l, r) for l, r in zip(left, right)]
    return np.where(np.asarray(cond, dtype=bool), left, right)


def _extract_t0(value: Any, env: NumpyExecutionEnv):
    if isinstance(value, tuple):
        return tuple(_extract_t0(v, env) for v in value)
    if isinstance(value, list):
        return tuple(_extract_t0(v, env) for v in value)
    if env.extract_t0_result is not None:
        return env.extract_t0_result(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.dtype.kind not in {"i", "f", "b"}:
        return _as_scalar(arr)
    return float(np.mean(arr))


def _mc_error(value: Any, n_paths: int):
    if isinstance(value, tuple):
        errs = tuple(_mc_error(v, n_paths) for v in value)
        return errs
    if isinstance(value, list):
        return tuple(_mc_error(v, n_paths) for v in value)
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.dtype.kind not in {"i", "f", "b"}:
        return None
    if arr.size <= 1:
        return 0.0
    return float(np.sqrt(np.var(arr, ddof=1) / float(n_paths)))


def _discount_t0(env: NumpyExecutionEnv, pay_date: Any, currency: Any):
    if env.discount_t0 is not None:
        return float(env.discount_t0(pay_date, currency))
    if env.reference_date is None or env.discount is None:
        return 1.0
    return float(_as_scalar(env.discount(env.reference_date, pay_date, currency, 1)))


def _pay_value(env: NumpyExecutionEnv, amount: Any, obs_date: Any, pay_date: Any, currency: Any):
    if env.pay is not None:
        return _broadcast(env.pay(amount, obs_date, pay_date, currency, env.n_paths), int(env.n_paths))
    df = _broadcast(
        env.discount(obs_date, pay_date, currency, env.n_paths) if env.discount is not None else 1.0,
        int(env.n_paths),
    )
    return amount * df


def _fx_spot_t0(env: NumpyExecutionEnv, currency: Any):
    if env.fx_spot_t0 is not None:
        return float(env.fx_spot_t0(currency))
    return 1.0


def _to_date(value: Any):
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return value
    return value


def _cashflow_results(cashflows: list[CashflowEvent], env: NumpyExecutionEnv):
    out = []
    for cf in cashflows:
        amount_t0 = _extract_t0(cf.amount, env)
        discount = 1.0
        fx = 1.0
        normalized_amount = amount_t0
        pay_date = cf.pay_date
        pay_date_cmp = _to_date(pay_date)
        ref_date_cmp = _to_date(env.reference_date)
        if ref_date_cmp is not None and isinstance(pay_date_cmp, date) and pay_date_cmp > ref_date_cmp:
            fx = _fx_spot_t0(env, cf.currency)
            discount = _discount_t0(env, pay_date, cf.currency)
            if fx != 0.0 and discount != 0.0:
                normalized_amount = amount_t0 / (fx * discount)
        entry = {
            "amount": normalized_amount,
            "amount_t0": amount_t0,
            "obs_date": cf.obs_date,
            "pay_date": cf.pay_date,
            "currency": cf.currency,
            "logged": cf.logged,
            "flow_type": cf.flow_type,
            "leg": cf.leg,
            "discount_factor": discount,
            "fx_spot_t0": fx,
        }
        out.append(entry)
    return tuple(out)


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
        obs_date = _as_scalar(_eval_expr(expr.obs_date, locals_, env))
        pay_date = _as_scalar(_eval_expr(expr.pay_date, locals_, env))
        currency = _as_scalar(_eval_expr(expr.currency, locals_, env))
        return _pay_value(env, amount, obs_date, pay_date, currency)
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
        elif isinstance(stmt, AssignItemStmt):
            idx = int(_as_scalar(_eval_expr(stmt.index, locals_, env))) - 1
            value = _eval_expr(stmt.expr, locals_, env)
            current = locals_.get(stmt.target, tuple())
            if not isinstance(current, list):
                current = list(current) if isinstance(current, tuple) else []
            while len(current) <= idx:
                current.append(_empty_like(value, int(env.n_paths)))
            current[idx] = value
            locals_[stmt.target] = tuple(current)
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
                    locals_[key] = _merge_branch_value(cond, then_locals[key], else_locals[key])
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
            amount = _eval_expr(stmt.amount, locals_, env)
            obs_date = _as_scalar(_eval_expr(stmt.obs_date, locals_, env))
            pay_date = _as_scalar(_eval_expr(stmt.pay_date, locals_, env))
            currency = _as_scalar(_eval_expr(stmt.currency, locals_, env))
            cashflows.append(CashflowEvent(_pay_value(env, amount, obs_date, pay_date, currency), obs_date, pay_date, currency, logged=False, flow_type=stmt.flow_type, leg=stmt.leg, metadata=stmt.metadata))
        elif isinstance(stmt, EmitLoggedCashflowStmt):
            amount = _eval_expr(stmt.amount, locals_, env)
            obs_date = _as_scalar(_eval_expr(stmt.obs_date, locals_, env))
            pay_date = _as_scalar(_eval_expr(stmt.pay_date, locals_, env))
            currency = _as_scalar(_eval_expr(stmt.currency, locals_, env))
            cashflows.append(CashflowEvent(_pay_value(env, amount, obs_date, pay_date, currency), obs_date, pay_date, currency, logged=True, flow_type=stmt.flow_type, leg=stmt.leg, metadata=stmt.metadata))
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
    result_t0 = {k: _extract_t0(v, env) for k, v in results.items()}
    result_err = {k: _mc_error(v, int(env.n_paths)) for k, v in results.items() if _mc_error(v, int(env.n_paths)) is not None}
    metadata = {
        "backend": "numpy",
        "npv_t0": _extract_t0(npv, env),
        "npv_mc_err_est": _mc_error(npv, int(env.n_paths)),
        "results_t0": result_t0,
        "results_mc_err_est": result_err,
        "cashflow_results": _cashflow_results(cashflows, env),
    }
    return ExecutionResult(npv=npv, results=results, cashflows=tuple(cashflows), metadata=metadata)
