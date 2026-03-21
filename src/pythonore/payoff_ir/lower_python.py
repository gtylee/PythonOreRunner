from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from pythonore.payoff_ir.ir import (
    AssignStateStmt,
    BinaryExpr,
    BooleanExpr,
    CashflowValueExpr,
    CompareExpr,
    ConstantExpr,
    EmitCashflowStmt,
    EmitLoggedCashflowStmt,
    Expr,
    ForEachDateStmt,
    IfStmt,
    IndexAtDateExpr,
    LetStmt,
    LocalRefExpr,
    MaxExpr,
    MinExpr,
    ParamRefExpr,
    PayoffModuleIR,
    RecordResultStmt,
    RequireStmt,
    SetNpvStmt,
    UnaryExpr,
    WhereExpr,
)
from pythonore.payoff_ir.types import (
    CurrencyParam,
    DateParam,
    DateScheduleParam,
    DaycountParam,
    ExternalSpec,
    IndexParam,
    NumberParam,
    ResultFieldSpec,
)
from pythonore.payoff_surface.restricted_loader import parse_restricted_payoff


class _State:
    def __init__(self):
        self.parameters: Dict[str, object] = {}
        self.seen_locals: set[str] = set()
        self.results_schema: Dict[str, ResultFieldSpec] = {}
        self.cashflow_counter = 0

    def param(self, name: str, kind: str):
        if name in self.parameters:
            return
        cls = {
            "number": NumberParam,
            "date": DateParam,
            "date_schedule": DateScheduleParam,
            "currency": CurrencyParam,
            "daycount": DaycountParam,
            "index": IndexParam,
        }[kind]
        self.parameters[name] = cls(name=name)


def _const(node: ast.AST) -> ConstantExpr:
    if isinstance(node, ast.Constant):
        return ConstantExpr(node.value)
    raise ValueError(f"Unsupported constant node {ast.dump(node)}")


def _lower_expr(node: ast.AST, state: _State, loop_schedule: Optional[str] = None, loop_var: Optional[str] = None) -> Tuple[Expr, Tuple]:
    if isinstance(node, ast.Constant):
        return ConstantExpr(node.value), ()
    if isinstance(node, ast.Name):
        if loop_var and node.id == loop_var:
            return LocalRefExpr(node.id), ()
        return LocalRefExpr(node.id), ()
    if isinstance(node, ast.UnaryOp):
        operand, prep = _lower_expr(node.operand, state, loop_schedule, loop_var)
        if isinstance(node.op, ast.USub):
            return UnaryExpr("neg", operand), prep
        if isinstance(node.op, ast.Not):
            return UnaryExpr("not", operand), prep
        raise ValueError(f"Unsupported unary op {ast.dump(node.op)}")
    if isinstance(node, ast.BoolOp):
        values: List[Expr] = []
        prep = []
        for value in node.values:
            expr, sub = _lower_expr(value, state, loop_schedule, loop_var)
            values.append(expr)
            prep.extend(sub)
        op = "and" if isinstance(node.op, ast.And) else "or"
        return BooleanExpr(op, tuple(values)), tuple(prep)
    if isinstance(node, ast.BinOp):
        left, prep_l = _lower_expr(node.left, state, loop_schedule, loop_var)
        right, prep_r = _lower_expr(node.right, state, loop_schedule, loop_var)
        op = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
        }.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binop {ast.dump(node.op)}")
        return BinaryExpr(op, left, right), prep_l + prep_r
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only simple binary comparisons are supported")
        left, prep_l = _lower_expr(node.left, state, loop_schedule, loop_var)
        right, prep_r = _lower_expr(node.comparators[0], state, loop_schedule, loop_var)
        op = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }.get(type(node.ops[0]))
        if op is None:
            raise ValueError(f"Unsupported compare op {ast.dump(node.ops[0])}")
        return CompareExpr(op, left, right), prep_l + prep_r
    if isinstance(node, ast.Subscript):
        schedule, prep_s = _lower_expr(node.value, state, loop_schedule, loop_var)
        if isinstance(node.slice, ast.Name) and loop_schedule and loop_var and node.slice.id == loop_var:
            if isinstance(schedule, ParamRefExpr) and schedule.name == loop_schedule:
                return LocalRefExpr(loop_var), prep_s
        index, prep_i = _lower_expr(node.slice, state, loop_schedule, loop_var)
        from pythonore.payoff_ir.ir import ScheduleItemExpr

        return ScheduleItemExpr(schedule, index), prep_s + prep_i
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "ctx":
            attr = node.func.attr
            if attr in {"number", "event", "events", "index", "currency", "daycount"}:
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise ValueError(f"ctx.{attr}() expects a single string literal")
                name = str(node.args[0].value)
                kind = {
                    "number": "number",
                    "event": "date",
                    "events": "date_schedule",
                    "index": "index",
                    "currency": "currency",
                    "daycount": "daycount",
                }[attr]
                state.param(name, kind)
                return ParamRefExpr(name), ()
            if attr in {"max", "min"}:
                exprs: List[Expr] = []
                prep = []
                for arg in node.args:
                    sub_expr, sub_prep = _lower_expr(arg, state, loop_schedule, loop_var)
                    exprs.append(sub_expr)
                    prep.extend(sub_prep)
                cls = MaxExpr if attr == "max" else MinExpr
                return cls(tuple(exprs)), tuple(prep)
            if attr == "where":
                cond, p0 = _lower_expr(node.args[0], state, loop_schedule, loop_var)
                a, p1 = _lower_expr(node.args[1], state, loop_schedule, loop_var)
                b, p2 = _lower_expr(node.args[2], state, loop_schedule, loop_var)
                return WhereExpr(cond, a, b), p0 + p1 + p2
            if attr in {"pay", "logpay"}:
                amount, p0 = _lower_expr(node.args[0], state, loop_schedule, loop_var)
                obs_date, p1 = _lower_expr(node.args[1], state, loop_schedule, loop_var)
                pay_date, p2 = _lower_expr(node.args[2], state, loop_schedule, loop_var)
                currency, p3 = _lower_expr(node.args[3], state, loop_schedule, loop_var)
                logged = attr == "logpay"
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                flow_type = None
                leg = None
                if "flow_type" in kwargs and isinstance(kwargs["flow_type"], ast.Constant):
                    flow_type = kwargs["flow_type"].value
                if "leg" in kwargs and isinstance(kwargs["leg"], ast.Constant):
                    leg = kwargs["leg"].value
                if len(node.args) >= 5 and isinstance(node.args[4], ast.Constant):
                    leg = node.args[4].value
                if len(node.args) >= 6 and isinstance(node.args[5], ast.Name):
                    flow_type = node.args[5].id
                state.cashflow_counter += 1
                tmp_name = f"_cfv_{state.cashflow_counter}"
                value = CashflowValueExpr(amount, obs_date, pay_date, currency, logged=logged, flow_type=flow_type, leg=leg)
                effect = EmitLoggedCashflowStmt(amount, obs_date, pay_date, currency, flow_type=flow_type, leg=leg) if logged else EmitCashflowStmt(amount, obs_date, pay_date, currency, flow_type=flow_type, leg=leg)
                return LocalRefExpr(tmp_name), p0 + p1 + p2 + p3 + (LetStmt(tmp_name, value), effect)
            raise ValueError(f"Unsupported ctx method '{attr}' in expression")
        if isinstance(node.func, ast.Attribute):
            receiver, prep = _lower_expr(node.func.value, state, loop_schedule, loop_var)
            if node.func.attr == "at":
                date, p2 = _lower_expr(node.args[0], state, loop_schedule, loop_var)
                return IndexAtDateExpr(receiver, date), prep + p2
        raise ValueError(f"Unsupported call expression {ast.dump(node)}")
    raise ValueError(f"Unsupported expression {ast.dump(node)}")


def _lower_stmt(node: ast.stmt, state: _State, loop_schedule: Optional[str] = None, loop_var: Optional[str] = None):
    out = []
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("Only simple name assignments are supported")
        name = node.targets[0].id
        expr, prep = _lower_expr(node.value, state, loop_schedule, loop_var)
        out.extend(prep)
        if name in state.seen_locals:
            out.append(AssignStateStmt(name, expr))
        else:
            state.seen_locals.add(name)
            out.append(LetStmt(name, expr))
        return out
    if isinstance(node, ast.If):
        cond, prep = _lower_expr(node.test, state, loop_schedule, loop_var)
        then_body = []
        else_body = []
        for item in node.body:
            then_body.extend(_lower_stmt(item, state, loop_schedule, loop_var))
        for item in node.orelse:
            else_body.extend(_lower_stmt(item, state, loop_schedule, loop_var))
        out.extend(prep)
        out.append(IfStmt(cond, tuple(then_body), tuple(else_body)))
        return out
    if isinstance(node, ast.For):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Loop target must be a simple name")
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and isinstance(node.iter.func.value, ast.Name)
            and node.iter.func.value.id == "ctx"
            and node.iter.func.attr == "events"
            and len(node.iter.args) == 1
            and isinstance(node.iter.args[0], ast.Constant)
            and isinstance(node.iter.args[0].value, str)
        ):
            raise ValueError("Only loops over ctx.events('...') are supported")
        schedule_name = str(node.iter.args[0].value)
        state.param(schedule_name, "date_schedule")
        nested = _State()
        nested.parameters = dict(state.parameters)
        nested.seen_locals = set(state.seen_locals)
        nested.results_schema = dict(state.results_schema)
        nested.cashflow_counter = state.cashflow_counter
        body = []
        for item in node.body:
            body.extend(_lower_stmt(item, nested, schedule_name, node.target.id))
        state.cashflow_counter = nested.cashflow_counter
        out.append(ForEachDateStmt(ParamRefExpr(schedule_name), node.target.id, tuple(body)))
        return out
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "ctx":
            attr = call.func.attr
            if attr == "record_result":
                name_node = call.args[0]
                if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
                    raise ValueError("record_result name must be a string literal")
                expr, prep = _lower_expr(call.args[1], state, loop_schedule, loop_var)
                out.extend(prep)
                state.results_schema[str(name_node.value)] = ResultFieldSpec(name=str(name_node.value))
                out.append(RecordResultStmt(str(name_node.value), expr))
                return out
            if attr == "require":
                cond, prep = _lower_expr(call.args[0], state, loop_schedule, loop_var)
                msg = ""
                if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
                    msg = str(call.args[1].value)
                out.extend(prep)
                out.append(RequireStmt(cond, msg))
                return out
        raise ValueError(f"Unsupported expression statement {ast.dump(node)}")
    if isinstance(node, ast.Return):
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "ctx"
            and call.func.attr == "set_npv"
            and len(call.args) == 1
        ):
            raise ValueError("Return must be ctx.set_npv(expr)")
        expr, prep = _lower_expr(call.args[0], state, loop_schedule, loop_var)
        out.extend(prep)
        out.append(SetNpvStmt(expr))
        return out
    raise ValueError(f"Unsupported statement {ast.dump(node)}")


def lower_python_payoff(source: str) -> PayoffModuleIR:
    func = parse_restricted_payoff(source)
    state = _State()
    region = []
    for stmt in func.body:
        region.extend(_lower_stmt(stmt, state))
    return PayoffModuleIR(
        parameters=tuple(state.parameters.values()),
        externals=(
            ExternalSpec("index_at", "index_lookup", required=False),
            ExternalSpec("discount", "discount_factor", required=False),
            ExternalSpec("above_prob", "probability", required=False),
            ExternalSpec("below_prob", "probability", required=False),
        ),
        regions=tuple(region),
        results_schema=tuple(state.results_schema.values()),
        metadata={"source": "python"},
    )
