from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

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
    Statement,
    UnaryExpr,
    WhereExpr,
)
from pythonore.payoff_ir.validate import validate_module


_COMMUTATIVE_OPS = {"add", "mul", "and", "or", "eq", "ne"}


def _const_eval_binary(op: str, left, right):
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        return left / right
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    raise ValueError(op)


def _normalize_expr(expr: Expr, cache: Dict[Expr, Expr]) -> Expr:
    if expr in cache:
        return cache[expr]
    result = expr
    if isinstance(expr, UnaryExpr):
        operand = _normalize_expr(expr.operand, cache)
        if isinstance(operand, ConstantExpr) and expr.op == "neg":
            result = ConstantExpr(-operand.value)
        else:
            result = replace(expr, operand=operand)
    elif isinstance(expr, (AbsExpr, ExpExpr, LogExpr, SqrtExpr)):
        result = replace(expr, operand=_normalize_expr(expr.operand, cache))
    elif isinstance(expr, (BinaryExpr, CompareExpr)):
        left = _normalize_expr(expr.left, cache)
        right = _normalize_expr(expr.right, cache)
        op = expr.op
        if op in _COMMUTATIVE_OPS and repr(left) > repr(right):
            left, right = right, left
        if isinstance(left, ConstantExpr) and isinstance(right, ConstantExpr):
            result = ConstantExpr(_const_eval_binary(op, left.value, right.value))
        else:
            result = replace(expr, left=left, right=right)
    elif isinstance(expr, (BooleanExpr, MinExpr, MaxExpr)):
        operands = tuple(_normalize_expr(x, cache) for x in expr.operands)
        if isinstance(expr, BooleanExpr) and expr.op in {"and", "or"}:
            operands = tuple(sorted(operands, key=repr))
        result = replace(expr, operands=operands)
    elif isinstance(expr, WhereExpr):
        result = replace(
            expr,
            condition=_normalize_expr(expr.condition, cache),
            if_true=_normalize_expr(expr.if_true, cache),
            if_false=_normalize_expr(expr.if_false, cache),
        )
    elif isinstance(expr, IndexAtDateExpr):
        result = replace(expr, index=_normalize_expr(expr.index, cache), date=_normalize_expr(expr.date, cache))
    elif isinstance(expr, DiscountFactorExpr):
        result = replace(
            expr,
            obs_date=_normalize_expr(expr.obs_date, cache),
            pay_date=_normalize_expr(expr.pay_date, cache),
            currency=_normalize_expr(expr.currency, cache),
        )
    elif isinstance(expr, CashflowValueExpr):
        result = replace(
            expr,
            amount=_normalize_expr(expr.amount, cache),
            obs_date=_normalize_expr(expr.obs_date, cache),
            pay_date=_normalize_expr(expr.pay_date, cache),
            currency=_normalize_expr(expr.currency, cache),
        )
    elif isinstance(expr, (AboveProbExpr, BelowProbExpr)):
        result = replace(
            expr,
            index=_normalize_expr(expr.index, cache),
            start=_normalize_expr(expr.start, cache),
            end=_normalize_expr(expr.end, cache),
            level=_normalize_expr(expr.level, cache),
        )
    elif isinstance(expr, ScheduleSizeExpr):
        result = replace(expr, schedule=_normalize_expr(expr.schedule, cache))
    elif isinstance(expr, ScheduleItemExpr):
        result = replace(expr, schedule=_normalize_expr(expr.schedule, cache), index=_normalize_expr(expr.index, cache))
    elif isinstance(expr, ContinuationValueExpr):
        result = replace(
            expr,
            value=_normalize_expr(expr.value, cache),
            features=tuple(_normalize_expr(x, cache) for x in expr.features),
            when=None if expr.when is None else _normalize_expr(expr.when, cache),
        )
    cache[result] = result
    return result


def _collect_used_locals(stmts: Tuple[Statement, ...]) -> set[str]:
    used: set[str] = set()

    def walk_expr(expr: Expr):
        if isinstance(expr, LocalRefExpr):
            used.add(expr.name)
        for attr in getattr(expr, "__dataclass_fields__", {}):
            value = getattr(expr, attr)
            if isinstance(value, Expr):
                walk_expr(value)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, Expr):
                        walk_expr(item)

    def walk_block(block: Tuple[Statement, ...]):
        for stmt in block:
            if isinstance(stmt, AssignStateStmt):
                used.add(stmt.name)
            elif isinstance(stmt, AssignItemStmt):
                used.add(stmt.target)
            elif isinstance(stmt, ForEachDateStmt):
                used.update(stmt.state_in)
                used.update(stmt.state_out)
            for attr in getattr(stmt, "__dataclass_fields__", {}):
                value = getattr(stmt, attr)
                if isinstance(value, Expr):
                    walk_expr(value)
                elif isinstance(value, tuple):
                    for item in value:
                        if isinstance(item, Expr):
                            walk_expr(item)
                        elif isinstance(item, Statement):
                            walk_block((item,))
                elif isinstance(value, Statement):
                    walk_block((value,))

    walk_block(stmts)
    return used


def _rename_block(stmts: Tuple[Statement, ...], mapping: Dict[str, str], counter: list[int], used: set[str]) -> Tuple[Statement, ...]:
    def remap(expr: Expr) -> Expr:
        if isinstance(expr, LocalRefExpr) and expr.name in mapping:
            return LocalRefExpr(mapping[expr.name])
        if isinstance(expr, (ConstantExpr, ParamRefExpr)):
            return expr
        cache: Dict[Expr, Expr] = {}
        replaced = _normalize_expr(expr, cache)
        if isinstance(replaced, LocalRefExpr) and replaced.name in mapping:
            return LocalRefExpr(mapping[replaced.name])
        fields = getattr(replaced, "__dataclass_fields__", {})
        kwargs = {}
        for name in fields:
            value = getattr(replaced, name)
            if isinstance(value, Expr):
                kwargs[name] = remap(value)
            elif isinstance(value, tuple):
                kwargs[name] = tuple(remap(item) if isinstance(item, Expr) else item for item in value)
            else:
                kwargs[name] = value
        return replace(replaced, **kwargs) if kwargs else replaced

    out = []
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            if stmt.name not in used:
                continue
            counter[0] += 1
            new_name = f"v{counter[0]}"
            mapping = dict(mapping)
            mapping[stmt.name] = new_name
            out.append(LetStmt(new_name, remap(stmt.expr)))
        elif isinstance(stmt, AssignStateStmt):
            name = mapping.get(stmt.name, stmt.name)
            out.append(AssignStateStmt(name, remap(stmt.expr)))
        elif isinstance(stmt, AssignItemStmt):
            target = mapping.get(stmt.target, stmt.target)
            out.append(AssignItemStmt(target, remap(stmt.index), remap(stmt.expr)))
        elif isinstance(stmt, IfStmt):
            out.append(
                IfStmt(
                    remap(stmt.condition),
                    _rename_block(stmt.then_body, dict(mapping), counter, used),
                    _rename_block(stmt.else_body, dict(mapping), counter, used),
                )
            )
        elif isinstance(stmt, ForEachDateStmt):
            counter[0] += 1
            loop_name = f"d{counter[0]}"
            nested = dict(mapping)
            nested[stmt.loop_var] = loop_name
            index_name = stmt.index_var
            if stmt.index_var:
                counter[0] += 1
                idx_name = f"i{counter[0]}"
                nested[stmt.index_var] = idx_name
                index_name = idx_name
            out.append(
                ForEachDateStmt(
                    schedule=remap(stmt.schedule),
                    loop_var=loop_name,
                    body=_rename_block(stmt.body, nested, counter, used),
                    state_in=tuple(mapping.get(x, x) for x in stmt.state_in),
                    state_out=tuple(mapping.get(x, x) for x in stmt.state_out),
                    index_var=index_name,
                )
            )
        elif isinstance(stmt, EmitCashflowStmt):
            out.append(replace(stmt, amount=remap(stmt.amount), obs_date=remap(stmt.obs_date), pay_date=remap(stmt.pay_date), currency=remap(stmt.currency)))
        elif isinstance(stmt, EmitLoggedCashflowStmt):
            out.append(replace(stmt, amount=remap(stmt.amount), obs_date=remap(stmt.obs_date), pay_date=remap(stmt.pay_date), currency=remap(stmt.currency)))
        elif isinstance(stmt, RecordResultStmt):
            out.append(replace(stmt, expr=remap(stmt.expr)))
        elif isinstance(stmt, SetNpvStmt):
            out.append(replace(stmt, expr=remap(stmt.expr)))
        elif isinstance(stmt, RequireStmt):
            out.append(replace(stmt, condition=remap(stmt.condition)))
        else:
            out.append(stmt)
    return tuple(out)


def normalize_module(module: PayoffModuleIR) -> PayoffModuleIR:
    validate_module(module)
    cache: Dict[Expr, Expr] = {}

    def norm_block(stmts: Tuple[Statement, ...]) -> Tuple[Statement, ...]:
        out = []
        for stmt in stmts:
            if isinstance(stmt, LetStmt):
                out.append(LetStmt(stmt.name, _normalize_expr(stmt.expr, cache)))
            elif isinstance(stmt, AssignStateStmt):
                out.append(AssignStateStmt(stmt.name, _normalize_expr(stmt.expr, cache)))
            elif isinstance(stmt, AssignItemStmt):
                out.append(AssignItemStmt(stmt.target, _normalize_expr(stmt.index, cache), _normalize_expr(stmt.expr, cache)))
            elif isinstance(stmt, IfStmt):
                out.append(IfStmt(_normalize_expr(stmt.condition, cache), norm_block(stmt.then_body), norm_block(stmt.else_body)))
            elif isinstance(stmt, ForEachDateStmt):
                out.append(
                    ForEachDateStmt(
                        _normalize_expr(stmt.schedule, cache),
                        stmt.loop_var,
                        norm_block(stmt.body),
                        stmt.state_in,
                        stmt.state_out,
                        stmt.index_var,
                    )
                )
            elif isinstance(stmt, EmitCashflowStmt):
                out.append(replace(stmt, amount=_normalize_expr(stmt.amount, cache), obs_date=_normalize_expr(stmt.obs_date, cache), pay_date=_normalize_expr(stmt.pay_date, cache), currency=_normalize_expr(stmt.currency, cache)))
            elif isinstance(stmt, EmitLoggedCashflowStmt):
                out.append(replace(stmt, amount=_normalize_expr(stmt.amount, cache), obs_date=_normalize_expr(stmt.obs_date, cache), pay_date=_normalize_expr(stmt.pay_date, cache), currency=_normalize_expr(stmt.currency, cache)))
            elif isinstance(stmt, RecordResultStmt):
                out.append(replace(stmt, expr=_normalize_expr(stmt.expr, cache)))
            elif isinstance(stmt, SetNpvStmt):
                out.append(replace(stmt, expr=_normalize_expr(stmt.expr, cache)))
            elif isinstance(stmt, RequireStmt):
                out.append(replace(stmt, condition=_normalize_expr(stmt.condition, cache)))
            else:
                out.append(stmt)
        return tuple(out)

    normalized = replace(module, regions=norm_block(module.regions))
    used = _collect_used_locals(normalized.regions)
    renamed = replace(normalized, regions=_rename_block(normalized.regions, {}, [0], used))
    validate_module(renamed)
    return renamed
