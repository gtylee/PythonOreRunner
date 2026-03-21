from __future__ import annotations

from typing import Dict, List, Tuple

from pythonore.payoff_ir.ir import (
    AbsExpr,
    AssignStateStmt,
    AboveProbExpr,
    BinaryExpr,
    BelowProbExpr,
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


def _render_expr(expr: Expr) -> str:
    if isinstance(expr, ConstantExpr):
        if isinstance(expr.value, str):
            return expr.value
        return repr(expr.value)
    if isinstance(expr, (ParamRefExpr, LocalRefExpr)):
        return expr.name
    if isinstance(expr, UnaryExpr):
        if expr.op == "neg":
            return f"-({_render_expr(expr.operand)})"
        if expr.op == "not":
            return f"NOT ({_render_expr(expr.operand)})"
    if isinstance(expr, AbsExpr):
        return f"abs({_render_expr(expr.operand)})"
    if isinstance(expr, ExpExpr):
        return f"exp({_render_expr(expr.operand)})"
    if isinstance(expr, LogExpr):
        return f"log({_render_expr(expr.operand)})"
    if isinstance(expr, SqrtExpr):
        return f"sqrt({_render_expr(expr.operand)})"
    if isinstance(expr, BinaryExpr):
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[expr.op]
        return f"({_render_expr(expr.left)} {op} {_render_expr(expr.right)})"
    if isinstance(expr, CompareExpr):
        op = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[expr.op]
        return f"({_render_expr(expr.left)} {op} {_render_expr(expr.right)})"
    if isinstance(expr, BooleanExpr):
        joiner = " AND " if expr.op == "and" else " OR "
        return "(" + joiner.join(_render_expr(x) for x in expr.operands) + ")"
    if isinstance(expr, MaxExpr):
        return f"max({', '.join(_render_expr(x) for x in expr.operands)})"
    if isinstance(expr, MinExpr):
        return f"min({', '.join(_render_expr(x) for x in expr.operands)})"
    if isinstance(expr, WhereExpr):
        raise ValueError("WhereExpr cannot be emitted directly to ORE syntax")
    if isinstance(expr, IndexAtDateExpr):
        return f"{_render_expr(expr.index)}({_render_expr(expr.date)})"
    if isinstance(expr, ScheduleSizeExpr):
        return f"SIZE({_render_expr(expr.schedule)})"
    if isinstance(expr, ScheduleItemExpr):
        return f"{_render_expr(expr.schedule)}[{_render_expr(expr.index)}]"
    if isinstance(expr, AboveProbExpr):
        return f"ABOVEPROB({_render_expr(expr.index)}, {_render_expr(expr.start)}, {_render_expr(expr.end)}, {_render_expr(expr.level)})"
    if isinstance(expr, BelowProbExpr):
        return f"BELOWPROB({_render_expr(expr.index)}, {_render_expr(expr.start)}, {_render_expr(expr.end)}, {_render_expr(expr.level)})"
    if isinstance(expr, DiscountFactorExpr):
        return f"DISCOUNT({_render_expr(expr.obs_date)}, {_render_expr(expr.pay_date)}, {_render_expr(expr.currency)})"
    if isinstance(expr, CashflowValueExpr):
        fname = "LOGPAY" if expr.logged else "PAY"
        args = [
            _render_expr(expr.amount),
            _render_expr(expr.obs_date),
            _render_expr(expr.pay_date),
            _render_expr(expr.currency),
        ]
        if expr.leg is not None:
            args.append(str(expr.leg))
        if expr.flow_type:
            args.append(str(expr.flow_type))
        return f"{fname}({', '.join(args)})"
    if isinstance(expr, ContinuationValueExpr):
        return f"NPV({_render_expr(expr.value)})"
    raise TypeError(f"Unsupported expr for emission {type(expr)!r}")


def _emit_block(stmts: Tuple, lines: List[str], indent: int = 0):
    prefix = " " * indent
    skip_effect_for: set[str] = set()
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, LetStmt):
            lines.append(f"{prefix}{stmt.name} = {_render_expr(stmt.expr)};")
            if isinstance(stmt.expr, CashflowValueExpr):
                skip_effect_for.add(stmt.name)
        elif isinstance(stmt, AssignStateStmt):
            lines.append(f"{prefix}{stmt.name} = {_render_expr(stmt.expr)};")
        elif isinstance(stmt, RequireStmt):
            lines.append(f"{prefix}REQUIRE {_render_expr(stmt.condition)};")
        elif isinstance(stmt, IfStmt):
            lines.append(f"{prefix}IF {_render_expr(stmt.condition)} THEN")
            _emit_block(stmt.then_body, lines, indent + 2)
            if stmt.else_body:
                lines.append(f"{prefix}ELSE")
                _emit_block(stmt.else_body, lines, indent + 2)
            lines.append(f"{prefix}END;")
        elif isinstance(stmt, ForEachDateStmt):
            idx = stmt.index_var or "i"
            schedule_name = _render_expr(stmt.schedule)
            lines.append(f"{prefix}FOR {idx} IN (1, SIZE({schedule_name}), 1) DO")
            lines.append(f"{prefix}  {stmt.loop_var} = {schedule_name}[{idx}];")
            _emit_block(stmt.body, lines, indent + 2)
            lines.append(f"{prefix}END;")
        elif isinstance(stmt, (EmitCashflowStmt, EmitLoggedCashflowStmt)):
            continue
        elif isinstance(stmt, RecordResultStmt):
            lines.append(f"{prefix}-- RESULT {stmt.name} = {_render_expr(stmt.expr)}")
        elif isinstance(stmt, SetNpvStmt):
            lines.append(f"{prefix}-- NPV {_render_expr(stmt.expr)}")
        else:
            raise TypeError(f"Unsupported statement for emission {type(stmt)!r}")


def emit_ore_script(module: PayoffModuleIR) -> str:
    mod = normalize_module(module)
    lines: List[str] = []
    _emit_block(mod.regions, lines)
    return "\n".join(lines).strip() + "\n"
