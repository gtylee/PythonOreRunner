from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class CubeAccessor:
    name: str
    payload: Dict[str, Any]

    def value(self, trade_id: str, key: str, default: Optional[float] = None) -> Optional[float]:
        table = self.payload.get(trade_id, {})
        return table.get(key, default)


@dataclass
class XVAResult:
    run_id: str
    pv_total: float
    xva_total: float
    xva_by_metric: Dict[str, float]
    exposure_by_netting_set: Dict[str, float]
    reports: Dict[str, Any] = field(default_factory=dict)
    cubes: Dict[str, CubeAccessor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def report(self, name: str) -> Any:
        return self.reports[name]

    def cube(self, name: str) -> CubeAccessor:
        return self.cubes[name]

    def reports_as_dataframe(self, name: str):
        import pandas as pd

        data = self.reports.get(name)
        if data is None:
            raise KeyError(name)
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, Iterable):
            return pd.DataFrame(list(data))
        raise TypeError(f"Unsupported report type for {name}: {type(data)}")
