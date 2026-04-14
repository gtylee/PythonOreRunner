from __future__ import annotations

from typing import Dict

import numpy as np

from pythonore.domain.dataclasses import FXForward, IRS, Trade, XVASnapshot
from pythonore.mapping.mapper import MappedInputs
from pythonore.runtime.results import CubeAccessor, XVAResult


def _toy_trade_numbers(trade: Trade):
    p = trade.product
    if isinstance(p, IRS):
        fair = 0.03
        direction = -1.0 if p.pay_fixed else 1.0
        pv = direction * p.notional * (fair - p.fixed_rate) * p.maturity_years
        epe = max(pv, 0.0) * 0.35 + abs(p.notional) * 0.001
        return pv, epe

    if isinstance(p, FXForward):
        spot = 1.1
        direction = 1.0 if p.buy_base else -1.0
        pv = direction * p.notional * (spot - p.strike)
        epe = max(pv, 0.0) * 0.5 + abs(p.notional) * 0.0008
        return pv, epe

    return 0.0, 1.0


class DeterministicToyAdapter:
    """In-memory deterministic adapter used for testable runtime behavior."""

    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        pv_total = 0.0
        epe_by_ns: Dict[str, float] = {}

        for t in snapshot.portfolio.trades:
            pv, epe = _toy_trade_numbers(t)
            pv_total += pv
            epe_by_ns[t.netting_set] = epe_by_ns.get(t.netting_set, 0.0) + epe

        total_epe = sum(epe_by_ns.values())
        metric_values: Dict[str, float] = {}
        for m in snapshot.config.analytics:
            if m == "CVA":
                metric_values[m] = 0.012 * total_epe
            elif m == "DVA":
                metric_values[m] = -0.006 * total_epe
            elif m == "FVA":
                metric_values[m] = 0.002 * abs(pv_total)
            elif m == "MVA":
                metric_values[m] = 0.0015 * total_epe

        xva_total = sum(metric_values.values())

        reports = {
            "xva": [
                {
                    "Metric": k,
                    "Value": v,
                }
                for k, v in metric_values.items()
            ],
            "exposure": [
                {"NettingSetId": ns, "EPE": v} for ns, v in epe_by_ns.items()
            ],
        }

        cubes = {
            "npv_cube": CubeAccessor(name="npv_cube", payload={"portfolio": {"t0": pv_total}}),
            "exposure_cube": CubeAccessor(name="exposure_cube", payload={ns: {"epe": v} for ns, v in epe_by_ns.items()}),
        }

        return XVAResult(
            run_id=run_id,
            pv_total=pv_total,
            xva_total=xva_total,
            xva_by_metric=metric_values,
            exposure_by_netting_set=epe_by_ns,
            exposure_profiles_by_netting_set={},
            exposure_profiles_by_trade={},
            reports=reports,
            cubes=cubes,
            metadata={
                "market_quotes": len(mapped.market_data_lines),
                "fixings": len(mapped.fixing_data_lines),
            },
        )


__all__ = ["DeterministicToyAdapter", "_toy_trade_numbers"]
