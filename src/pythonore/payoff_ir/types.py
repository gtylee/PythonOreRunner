from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


ValueKind = str


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: ValueKind
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NumberParam(ParameterSpec):
    kind: ValueKind = field(init=False, default="number")


@dataclass(frozen=True)
class DateParam(ParameterSpec):
    kind: ValueKind = field(init=False, default="date")


@dataclass(frozen=True)
class DateScheduleParam(ParameterSpec):
    kind: ValueKind = field(init=False, default="date_schedule")


@dataclass(frozen=True)
class CurrencyParam(ParameterSpec):
    kind: ValueKind = field(init=False, default="currency")


@dataclass(frozen=True)
class IndexParam(ParameterSpec):
    kind: ValueKind = field(init=False, default="index")


@dataclass(frozen=True)
class DaycountParam(ParameterSpec):
    kind: ValueKind = field(init=False, default="daycount")


@dataclass(frozen=True)
class ExternalSpec:
    name: str
    kind: str
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultFieldSpec:
    name: str
    kind: str = "number"
    optional: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CashflowEvent:
    amount: Any
    obs_date: Any
    pay_date: Any
    currency: Any
    logged: bool = False
    flow_type: Optional[str] = None
    leg: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    npv: Any
    results: Mapping[str, Any] = field(default_factory=dict)
    cashflows: Tuple[CashflowEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
