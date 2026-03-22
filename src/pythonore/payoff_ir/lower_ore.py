from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Tuple

from pythonore.payoff_ir.ir import (
    AssignItemStmt,
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
    ScheduleItemExpr,
    ScheduleSizeExpr,
    SetNpvStmt,
    UnaryExpr,
    WhereExpr,
    AboveProbExpr,
    BelowProbExpr,
)
from pythonore.payoff_ir.types import (
    CurrencyParam,
    DateParam,
    DateScheduleParam,
    DaycountParam,
    IndexParam,
    NumberParam,
    ResultFieldSpec,
)


_DECL_RE = re.compile(r"^(NUMBER|EVENT|INDEX|CURRENCY|DAYCOUNTER)\s+(.+)$", re.IGNORECASE)
_FOR_RE = re.compile(r"^FOR\s+([A-Za-z_]\w*)\s+IN\s+\(\s*1\s*,\s*SIZE\(\s*([A-Za-z_]\w*)\s*\)\s*,\s*1\s*\)\s+DO$", re.IGNORECASE)


class _State:
    def __init__(self):
        self.parameters: Dict[str, object] = {}
        self.seen: set[str] = set()
        self.declared_locals: set[str] = set()
        self.defined_locals: set[str] = set()
        self.cashflow_counter = 0

    def clone(self) -> "_State":
        other = _State()
        other.parameters = dict(self.parameters)
        other.seen = set(self.seen)
        other.declared_locals = set(self.declared_locals)
        other.defined_locals = set(self.defined_locals)
        other.cashflow_counter = self.cashflow_counter
        return other

    def merge_metadata(self, other: "_State") -> None:
        self.parameters.update(other.parameters)
        self.seen |= other.seen
        self.declared_locals |= other.declared_locals
        self.cashflow_counter = max(self.cashflow_counter, other.cashflow_counter)

    def add_param(self, name: str, kind: str):
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


def _logical_lines(script: str) -> List[str]:
    cleaned = []
    for raw in script.splitlines():
        line = raw.split("--", 1)[0].rstrip()
        if line.strip():
            cleaned.append(line)
    out: List[str] = []
    buffer = ""
    for line in cleaned:
        piece = line.strip()
        if not buffer:
            buffer = piece
        else:
            buffer += " " + piece
        upper = buffer.upper()
        if upper == "ELSE" or upper.startswith("FOR ") and upper.endswith(" DO") or upper.startswith("IF ") and upper.endswith(" THEN"):
            out.append(buffer)
            buffer = ""
            continue
        while ";" in buffer:
            head, tail = buffer.split(";", 1)
            if head.strip():
                out.append(head.strip())
            buffer = tail.strip()
    if buffer.strip():
        out.append(buffer.strip())
    return out


def _preprocess_expr(expr: str) -> str:
    txt = expr.replace("{", "(").replace("}", ")")
    txt = re.sub(r"\bAND\b", " and ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bOR\b", " or ", txt, flags=re.IGNORECASE)
    return txt


def _lower_expr_from_str(expr: str, state: _State, loop_schedule: Optional[str] = None, loop_index: Optional[str] = None, loop_date: Optional[str] = None) -> Tuple[Expr, Tuple]:
    node = ast.parse(_preprocess_expr(expr), mode="eval").body
    return _lower_ast_expr(node, state, loop_schedule=loop_schedule, loop_index=loop_index, loop_date=loop_date)


def _lower_ast_expr(node: ast.AST, state: _State, loop_schedule: Optional[str] = None, loop_index: Optional[str] = None, loop_date: Optional[str] = None) -> Tuple[Expr, Tuple]:
    if isinstance(node, ast.Constant):
        return ConstantExpr(node.value), ()
    if isinstance(node, ast.Name):
        if loop_index and node.id == loop_index:
            return LocalRefExpr(node.id), ()
        if loop_date and node.id == loop_date:
            return LocalRefExpr(node.id), ()
        if node.id in state.seen or node.id in state.declared_locals or node.id in state.defined_locals:
            return LocalRefExpr(node.id), ()
        if node.id and node.id[0].isupper():
            kind = "number"
            if "Date" in node.id or node.id in {"Expiry", "Settlement", "StartDate"}:
                kind = "date"
            elif node.id.endswith("Dates"):
                kind = "date_schedule"
            elif node.id.endswith("Ccy") or "Currency" in node.id:
                kind = "currency"
            elif node.id in {"Underlying"}:
                kind = "index"
            state.add_param(node.id, kind)
            return ParamRefExpr(node.id), ()
        return LocalRefExpr(node.id), ()
    if isinstance(node, ast.UnaryOp):
        operand, prep = _lower_ast_expr(node.operand, state, loop_schedule, loop_index, loop_date)
        if isinstance(node.op, ast.USub):
            return UnaryExpr("neg", operand), prep
        if isinstance(node.op, ast.Not):
            return UnaryExpr("not", operand), prep
        raise ValueError(f"Unsupported unary op {ast.dump(node.op)}")
    if isinstance(node, ast.BoolOp):
        exprs: List[Expr] = []
        prep = []
        for value in node.values:
            e, p = _lower_ast_expr(value, state, loop_schedule, loop_index, loop_date)
            exprs.append(e)
            prep.extend(p)
        return BooleanExpr("and" if isinstance(node.op, ast.And) else "or", tuple(exprs)), tuple(prep)
    if isinstance(node, ast.BinOp):
        left, p0 = _lower_ast_expr(node.left, state, loop_schedule, loop_index, loop_date)
        right, p1 = _lower_ast_expr(node.right, state, loop_schedule, loop_index, loop_date)
        op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary op {ast.dump(node.op)}")
        return BinaryExpr(op, left, right), p0 + p1
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only simple comparisons are supported")
        left, p0 = _lower_ast_expr(node.left, state, loop_schedule, loop_index, loop_date)
        right, p1 = _lower_ast_expr(node.comparators[0], state, loop_schedule, loop_index, loop_date)
        op = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}[type(node.ops[0])]
        return CompareExpr(op, left, right), p0 + p1
    if isinstance(node, ast.Subscript):
        schedule, p0 = _lower_ast_expr(node.value, state, loop_schedule, loop_index, loop_date)
        if isinstance(node.slice, ast.Name) and loop_schedule and loop_index and loop_date and node.slice.id == loop_index:
            if isinstance(schedule, ParamRefExpr) and schedule.name == loop_schedule:
                return LocalRefExpr(loop_date), p0
        idx, p1 = _lower_ast_expr(node.slice, state, loop_schedule, loop_index, loop_date)
        return ScheduleItemExpr(schedule, idx), p0 + p1
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname == "SIZE":
                expr, prep = _lower_ast_expr(node.args[0], state, loop_schedule, loop_index, loop_date)
                args = [expr]
                return ScheduleSizeExpr(args[0]), tuple(prep)
            args: List[Expr] = []
            prep = []
            for arg in node.args:
                e, p = _lower_ast_expr(arg, state, loop_schedule, loop_index, loop_date)
                args.append(e)
                prep.extend(p)
            if fname == "max":
                return MaxExpr(tuple(args)), tuple(prep)
            if fname == "min":
                return MinExpr(tuple(args)), tuple(prep)
            if fname == "PAY" or fname == "LOGPAY":
                logged = fname == "LOGPAY"
                args = []
                prep = []
                for arg in node.args[:4]:
                    e, p = _lower_ast_expr(arg, state, loop_schedule, loop_index, loop_date)
                    args.append(e)
                    prep.extend(p)
                flow_type = None
                leg = None
                if len(node.args) >= 5 and isinstance(node.args[4], ast.Constant):
                    leg = int(node.args[4].value)
                if len(node.args) >= 6:
                    if isinstance(node.args[5], ast.Name):
                        flow_type = node.args[5].id
                    elif isinstance(node.args[5], ast.Constant):
                        flow_type = str(node.args[5].value)
                state.cashflow_counter += 1
                tmp = f"_cfv_{state.cashflow_counter}"
                value = CashflowValueExpr(args[0], args[1], args[2], args[3], logged=logged, flow_type=flow_type, leg=leg)
                effect = EmitLoggedCashflowStmt(args[0], args[1], args[2], args[3], flow_type=flow_type, leg=leg) if logged else EmitCashflowStmt(args[0], args[1], args[2], args[3], flow_type=flow_type, leg=leg)
                return LocalRefExpr(tmp), tuple(prep) + (LetStmt(tmp, value), effect)
            if fname == "ABOVEPROB":
                return AboveProbExpr(args[0], args[1], args[2], args[3]), tuple(prep)
            if fname == "BELOWPROB":
                return BelowProbExpr(args[0], args[1], args[2], args[3]), tuple(prep)
            if fname and fname[0].isupper():
                if len(args) == 1:
                    state.add_param(fname, "index")
                    return IndexAtDateExpr(ParamRefExpr(fname), args[0]), tuple(prep)
            raise ValueError(f"Unsupported call '{fname}'")
        raise ValueError(f"Unsupported call expression {ast.dump(node)}")
    raise ValueError(f"Unsupported ORE expression {ast.dump(node)}")


def _parse_block(lines: List[str], pos: int, state: _State, loop_schedule: Optional[str] = None, loop_index: Optional[str] = None, loop_date: Optional[str] = None):
    out = []
    while pos < len(lines):
        line = lines[pos].strip()
        upper = line.upper()
        if upper == "ELSE" or upper == "END":
            break
        decl = _DECL_RE.match(line)
        if decl:
            decl_kind = decl.group(1).upper()
            names = [x.strip() for x in decl.group(2).split(",") if x.strip()]
            for name in names:
                local_name = name.split("[", 1)[0].strip()
                if local_name not in state.seen:
                    state.seen.add(local_name)
                    state.declared_locals.add(local_name)
                    state.defined_locals.add(local_name)
                    out.append(LetStmt(local_name, ConstantExpr(0.0)))
            pos += 1
            continue
        loop = _FOR_RE.match(line)
        if loop:
            idx_name = loop.group(1)
            sched_name = loop.group(2)
            state.add_param(sched_name, "date_schedule")
            body, pos = _parse_block(lines, pos + 1, state, sched_name, idx_name, f"{idx_name}_date")
            if pos >= len(lines) or lines[pos].strip().upper() != "END":
                raise ValueError("FOR without matching END")
            out.append(ForEachDateStmt(ParamRefExpr(sched_name), f"{idx_name}_date", tuple(body), index_var=idx_name))
            pos += 1
            continue
        if upper.startswith("IF ") and upper.endswith(" THEN"):
            cond_text = line[3:-5].strip()
            cond, prep = _lower_expr_from_str(cond_text, state, loop_schedule, loop_index, loop_date)
            outer_defined = set(state.defined_locals)

            then_state = state.clone()
            then_body, pos = _parse_block(lines, pos + 1, then_state, loop_schedule, loop_index, loop_date)
            else_body = []
            else_state = state.clone()
            else_state.cashflow_counter = then_state.cashflow_counter
            if pos < len(lines) and lines[pos].strip().upper() == "ELSE":
                else_body, pos = _parse_block(lines, pos + 1, else_state, loop_schedule, loop_index, loop_date)
            if pos >= len(lines) or lines[pos].strip().upper() != "END":
                raise ValueError("IF without matching END")

            state.merge_metadata(then_state)
            state.merge_metadata(else_state)

            common_new_defs = (then_state.defined_locals - outer_defined) & (else_state.defined_locals - outer_defined) if else_body else set()
            for name in sorted(common_new_defs):
                if name not in state.defined_locals:
                    state.seen.add(name)
                    state.defined_locals.add(name)
                    out.append(LetStmt(name, ConstantExpr(0.0)))
            if common_new_defs:
                then_body = _rewrite_branch_defs(tuple(then_body), common_new_defs)
                else_body = _rewrite_branch_defs(tuple(else_body), common_new_defs)

            out.extend(prep)
            out.append(IfStmt(cond, tuple(then_body), tuple(else_body)))
            pos += 1
            continue
        if upper.startswith("REQUIRE "):
            cond, prep = _lower_expr_from_str(line[8:].strip(), state, loop_schedule, loop_index, loop_date)
            out.extend(prep)
            out.append(RequireStmt(cond, line))
            pos += 1
            continue
        if "=" in line:
            name, expr_txt = [x.strip() for x in line.split("=", 1)]
            expr, prep = _lower_expr_from_str(expr_txt, state, loop_schedule, loop_index, loop_date)
            out.extend(prep)
            item_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]$", name)
            if item_match:
                target = item_match.group(1).strip()
                idx_expr, idx_prep = _lower_expr_from_str(item_match.group(2).strip(), state, loop_schedule, loop_index, loop_date)
                out.extend(idx_prep)
                if target not in state.seen:
                    state.seen.add(target)
                    state.defined_locals.add(target)
                    out.append(LetStmt(target, ConstantExpr(tuple())))
                out.append(AssignItemStmt(target, idx_expr, expr))
                pos += 1
                continue
            if name in state.defined_locals:
                out.append(AssignStateStmt(name, expr))
            else:
                state.seen.add(name)
                state.defined_locals.add(name)
                out.append(LetStmt(name, expr))
            pos += 1
            continue
        raise ValueError(f"Unsupported ORE statement '{line}'")
    return out, pos


def _rewrite_branch_defs(stmts: Tuple, names: set[str]) -> Tuple:
    out = []
    for stmt in stmts:
        if isinstance(stmt, LetStmt) and stmt.name in names:
            out.append(AssignStateStmt(stmt.name, stmt.expr))
        elif isinstance(stmt, IfStmt):
            out.append(
                IfStmt(
                    stmt.condition,
                    _rewrite_branch_defs(stmt.then_body, names),
                    _rewrite_branch_defs(stmt.else_body, names),
                )
            )
        elif isinstance(stmt, ForEachDateStmt):
            out.append(
                ForEachDateStmt(
                    stmt.schedule,
                    stmt.loop_var,
                    _rewrite_branch_defs(stmt.body, names),
                    stmt.state_in,
                    stmt.state_out,
                    stmt.index_var,
                )
            )
        else:
            out.append(stmt)
    return tuple(out)


def lower_ore_script(
    code: str,
    *,
    npv_variable: str,
    results: Optional[Tuple[Tuple[str, str], ...]] = None,
) -> PayoffModuleIR:
    state = _State()
    lines = _logical_lines(code)
    stmts, pos = _parse_block(lines, 0, state)
    if pos != len(lines):
        raise ValueError(f"Unparsed ORE script tail starting at line {pos + 1}")
    if npv_variable not in state.seen:
        raise ValueError(f"NPV variable '{npv_variable}' was not assigned in script")
    results_schema = []
    for name, ref in results or ():
        results_schema.append(ResultFieldSpec(name=name))
        stmts.append(RecordResultStmt(name, ParamRefExpr(ref) if ref in state.parameters else LocalRefExpr(ref)))
    stmts.append(SetNpvStmt(LocalRefExpr(npv_variable)))
    return PayoffModuleIR(
        parameters=tuple(state.parameters.values()),
        regions=tuple(stmts),
        results_schema=tuple(results_schema),
        metadata={"source": "ore", "npv_variable": npv_variable},
    )
