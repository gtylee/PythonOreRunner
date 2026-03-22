from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from pythonore.payoff_ir.types import ExternalSpec, ParameterSpec, ResultFieldSpec


class Expr:
    """Marker base for pure payoff expressions."""


@dataclass(frozen=True)
class ConstantExpr(Expr):
    value: Any


@dataclass(frozen=True)
class ParamRefExpr(Expr):
    name: str


@dataclass(frozen=True)
class LocalRefExpr(Expr):
    name: str


@dataclass(frozen=True)
class UnaryExpr(Expr):
    op: str
    operand: Expr


@dataclass(frozen=True)
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class CompareExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class BooleanExpr(Expr):
    op: str
    operands: Tuple[Expr, ...]


@dataclass(frozen=True)
class MinExpr(Expr):
    operands: Tuple[Expr, ...]


@dataclass(frozen=True)
class MaxExpr(Expr):
    operands: Tuple[Expr, ...]


@dataclass(frozen=True)
class AbsExpr(Expr):
    operand: Expr


@dataclass(frozen=True)
class ExpExpr(Expr):
    operand: Expr


@dataclass(frozen=True)
class LogExpr(Expr):
    operand: Expr


@dataclass(frozen=True)
class SqrtExpr(Expr):
    operand: Expr


@dataclass(frozen=True)
class WhereExpr(Expr):
    condition: Expr
    if_true: Expr
    if_false: Expr


@dataclass(frozen=True)
class IndexAtDateExpr(Expr):
    index: Expr
    date: Expr


@dataclass(frozen=True)
class DiscountFactorExpr(Expr):
    obs_date: Expr
    pay_date: Expr
    currency: Expr


@dataclass(frozen=True)
class CashflowValueExpr(Expr):
    amount: Expr
    obs_date: Expr
    pay_date: Expr
    currency: Expr
    logged: bool = False
    flow_type: Optional[str] = None
    leg: Optional[int] = None


@dataclass(frozen=True)
class AboveProbExpr(Expr):
    index: Expr
    start: Expr
    end: Expr
    level: Expr


@dataclass(frozen=True)
class BelowProbExpr(Expr):
    index: Expr
    start: Expr
    end: Expr
    level: Expr


@dataclass(frozen=True)
class ScheduleSizeExpr(Expr):
    schedule: Expr


@dataclass(frozen=True)
class ScheduleItemExpr(Expr):
    schedule: Expr
    index: Expr


@dataclass(frozen=True)
class ContinuationValueExpr(Expr):
    label: str
    value: Expr
    features: Tuple[Expr, ...] = ()
    when: Optional[Expr] = None


class Statement:
    """Marker base for ordered payoff statements."""


@dataclass(frozen=True)
class LetStmt(Statement):
    name: str
    expr: Expr


@dataclass(frozen=True)
class AssignStateStmt(Statement):
    name: str
    expr: Expr


@dataclass(frozen=True)
class AssignItemStmt(Statement):
    target: str
    index: Expr
    expr: Expr


@dataclass(frozen=True)
class IfStmt(Statement):
    condition: Expr
    then_body: Tuple[Statement, ...]
    else_body: Tuple[Statement, ...] = ()


@dataclass(frozen=True)
class ForEachDateStmt(Statement):
    schedule: Expr
    loop_var: str
    body: Tuple[Statement, ...]
    state_in: Tuple[str, ...] = ()
    state_out: Tuple[str, ...] = ()
    index_var: Optional[str] = None


@dataclass(frozen=True)
class EmitCashflowStmt(Statement):
    amount: Expr
    obs_date: Expr
    pay_date: Expr
    currency: Expr
    flow_type: Optional[str] = None
    leg: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmitLoggedCashflowStmt(Statement):
    amount: Expr
    obs_date: Expr
    pay_date: Expr
    currency: Expr
    flow_type: Optional[str] = None
    leg: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordResultStmt(Statement):
    name: str
    expr: Expr


@dataclass(frozen=True)
class SetNpvStmt(Statement):
    expr: Expr


@dataclass(frozen=True)
class RequireStmt(Statement):
    condition: Expr
    message: str = ""


@dataclass(frozen=True)
class PayoffModuleIR:
    parameters: Tuple[ParameterSpec, ...] = ()
    externals: Tuple[ExternalSpec, ...] = ()
    regions: Tuple[Statement, ...] = ()
    results_schema: Tuple[ResultFieldSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
