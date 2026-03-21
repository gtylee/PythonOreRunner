from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SymbolicValue:
    name: str


@dataclass(frozen=True)
class SymbolicSchedule:
    name: str


class PayoffContext:
    """Restricted authoring surface for symbolic payoff definitions.

    The context is meant to be lowered from source code, not executed directly.
    """

    def _unsupported(self, name: str):
        raise RuntimeError(f"{name}() is only available when lowering a payoff source to IR")

    def number(self, name: str) -> SymbolicValue:
        self._unsupported("number")

    def event(self, name: str) -> SymbolicValue:
        self._unsupported("event")

    def events(self, name: str) -> SymbolicSchedule:
        self._unsupported("events")

    def index(self, name: str) -> SymbolicValue:
        self._unsupported("index")

    def currency(self, name: str) -> SymbolicValue:
        self._unsupported("currency")

    def daycount(self, name: str) -> SymbolicValue:
        self._unsupported("daycount")

    def where(self, condition: Any, if_true: Any, if_false: Any):
        self._unsupported("where")

    def max(self, *args: Any):
        self._unsupported("max")

    def min(self, *args: Any):
        self._unsupported("min")

    def pay(self, *args: Any, **kwargs: Any):
        self._unsupported("pay")

    def logpay(self, *args: Any, **kwargs: Any):
        self._unsupported("logpay")

    def record_result(self, *args: Any, **kwargs: Any):
        self._unsupported("record_result")

    def set_npv(self, *args: Any, **kwargs: Any):
        self._unsupported("set_npv")

    def require(self, *args: Any, **kwargs: Any):
        self._unsupported("require")
