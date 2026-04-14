from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence

from pythonore.domain.dataclasses import FXForward, IRS, Trade, XVASnapshot
from pythonore.mapping.mapper import MappedInputs, map_snapshot
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.results import XVAResult
from pythonore.runtime.toy import DeterministicToyAdapter


class RunnerAdapter(Protocol):
    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult: ...


@dataclass
class SessionState:
    """Frozen snapshot of the input state tracked by an :class:`XVASession`."""

    snapshot: XVASnapshot
    snapshot_key: str
    mapped_inputs: MappedInputs
    rebuild_counts: Dict[str, int]


class _ConsoleRunReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        stream: Any = None,
        use_bar: Optional[bool] = None,
        checkpoint_interval: int = 25,
        bar_width: int = 24,
    ):
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stderr
        self.use_bar = bool(use_bar) if use_bar is not None else bool(getattr(self.stream, "isatty", lambda: False)())
        self.checkpoint_interval = max(int(checkpoint_interval), 1)
        self.bar_width = max(int(bar_width), 10)
        self.start_time = time.monotonic()
        self._active_label = ""
        self._active_total = 0
        self._active_progress = 0
        self._last_percent = -1
        self._last_checkpoint = -1
        self._bar_open = False

    def _elapsed(self) -> str:
        elapsed = max(time.monotonic() - self.start_time, 0.0)
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def _write_line(self, text: str) -> None:
        if not self.enabled:
            return
        if self._bar_open:
            self.stream.write("\r" + " " * 160 + "\r")
            self._bar_open = False
        self.stream.write(f"[python-lgm {self._elapsed()}] {text}\n")
        self.stream.flush()

    def log(self, text: str) -> None:
        self._write_line(text)

    def start(self, label: str, total: int) -> None:
        if not self.enabled:
            return
        self._active_label = str(label)
        self._active_total = max(int(total), 0)
        self._active_progress = 0
        self._last_percent = -1
        self._last_checkpoint = -1
        if self._active_total <= 0:
            self.log(f"{self._active_label}: nothing to do")
            return
        if not self.use_bar:
            self.log(f"{self._active_label}: 0/{self._active_total}")

    def update(self, progress: int, detail: str = "") -> None:
        if not self.enabled or self._active_total <= 0:
            return
        current = min(max(int(progress), 0), self._active_total)
        self._active_progress = current
        suffix = f" {detail}" if detail else ""
        if self.use_bar:
            ratio = float(current) / float(self._active_total)
            percent = int(ratio * 100.0)
            if percent == self._last_percent and current < self._active_total:
                return
            self._last_percent = percent
            filled = min(int(self.bar_width * ratio), self.bar_width)
            if filled >= self.bar_width:
                bar = "=" * self.bar_width
            else:
                head = "=" * max(filled - 1, 0)
                mid = ">" if filled > 0 else ""
                tail = " " * (self.bar_width - len(head) - len(mid))
                bar = head + mid + tail
            self.stream.write(
                "\r"
                + f"[python-lgm {self._elapsed()}] {self._active_label:<20} "
                + f"[{bar}] {current}/{self._active_total}{suffix}"[:220]
            )
            self.stream.flush()
            self._bar_open = current < self._active_total
            if current >= self._active_total:
                self.stream.write("\n")
                self.stream.flush()
                self._bar_open = False
            return
        if current == self._active_total or current == 0 or current - self._last_checkpoint >= self.checkpoint_interval:
            self._last_checkpoint = current
            self.log(f"{self._active_label}: {current}/{self._active_total}{suffix}")

    def finish(self, detail: str = "") -> None:
        if not self.enabled or self._active_total <= 0:
            return
        self.update(self._active_total, detail)
        self._active_label = ""
        self._active_total = 0
        self._active_progress = 0


class XVAEngine:
    """Orchestration entry point for XVA computation."""

    def __init__(self, adapter: Optional[RunnerAdapter] = None):
        from .runtime_impl import PythonLgmAdapter

        self.adapter = adapter or DeterministicToyAdapter()
        self._python_lgm_adapter_type = PythonLgmAdapter

    @classmethod
    def python_lgm_default(cls, fallback_to_swig: bool = False) -> "XVAEngine":
        from .runtime_impl import PythonLgmAdapter

        return cls(adapter=PythonLgmAdapter(fallback_to_swig=fallback_to_swig))

    def create_session(self, snapshot: XVASnapshot) -> "XVASession":
        mapped = map_snapshot(snapshot)
        return XVASession(
            engine=self,
            state=SessionState(
                snapshot=snapshot,
                snapshot_key=snapshot.stable_key(),
                mapped_inputs=mapped,
                rebuild_counts={"market": 1, "portfolio": 1, "config": 1},
            ),
        )

    def prepare_sensitivity_snapshot(
        self,
        snapshot: XVASnapshot,
        *,
        curve_node_shocks: Optional[Dict[str, object]] = None,
        curve_fit_mode: str = "ore_fit",
        use_ore_output_curves: bool = False,
        freeze_float_spreads: bool = False,
        frozen_float_spreads: Optional[Dict[str, Sequence[float]]] = None,
    ) -> XVASnapshot:
        params = dict(snapshot.config.params)
        params["python.curve_fit_mode"] = str(curve_fit_mode)
        params["python.use_ore_output_curves"] = "Y" if use_ore_output_curves else "N"
        if curve_node_shocks is not None:
            params["python.curve_node_shocks"] = curve_node_shocks
        elif "python.curve_node_shocks" in params:
            params.pop("python.curve_node_shocks", None)
        if frozen_float_spreads is not None:
            params["python.frozen_float_spreads"] = frozen_float_spreads
        elif freeze_float_spreads and hasattr(self.adapter, "compute_frozen_float_spreads"):
            frozen = self.adapter.compute_frozen_float_spreads(
                replace(snapshot, config=replace(snapshot.config, params=params))
            )
            if frozen:
                params["python.frozen_float_spreads"] = frozen
        updated = replace(snapshot, config=replace(snapshot.config, params=params))
        if hasattr(self.adapter, "prepare_sensitivity_snapshot"):
            prepared = self.adapter.prepare_sensitivity_snapshot(
                updated,
                curve_node_shocks=curve_node_shocks,
                curve_fit_mode=curve_fit_mode,
                use_ore_output_curves=use_ore_output_curves,
                freeze_float_spreads=freeze_float_spreads,
                frozen_float_spreads=frozen_float_spreads,
            )
            if isinstance(prepared, XVASnapshot):
                return prepared
        return updated


class XVASession:
    """Live computation session wrapping a :class:`SessionState`."""

    def __init__(self, engine: XVAEngine, state: SessionState):
        self.engine = engine
        self.state = state

    def run(self, metrics: Optional[Sequence[str]] = None, return_cubes: bool = True) -> XVAResult:
        run_id = str(uuid.uuid4())
        if metrics:
            snapshot = replace(self.state.snapshot, config=replace(self.state.snapshot.config, analytics=tuple(metrics)))
            mapped = map_snapshot(snapshot)
        else:
            snapshot = self.state.snapshot
            mapped = self.state.mapped_inputs

        result = self.engine.adapter.run(snapshot=snapshot, mapped=mapped, run_id=run_id)
        result.metadata["rebuild_counts"] = dict(self.state.rebuild_counts)
        if not return_cubes:
            result.cubes = {}
        return result

    def run_incremental(self) -> XVAResult:
        return self.run()

    def update_market(self, market) -> None:
        updated = replace(self.state.snapshot, market=market)
        self._apply_snapshot(updated, changed="market")

    def update_config(self, config=None, **overrides) -> None:
        if config is not None and overrides:
            raise EngineRunError("Provide either config or keyword overrides, not both")
        if config is None:
            config = replace(self.state.snapshot.config, **overrides)
        updated = replace(self.state.snapshot, config=config)
        self._apply_snapshot(updated, changed="config")

    def update_portfolio(self, add: Iterable[Trade] = (), amend: Iterable[tuple[str, dict]] = (), remove: Iterable[str] = ()) -> None:
        trade_map = {t.trade_id: t for t in self.state.snapshot.portfolio.trades}

        for tid in remove:
            trade_map.pop(tid, None)
        for tid, updates in amend:
            if tid not in trade_map:
                raise EngineRunError(f"Cannot amend unknown trade {tid}")
            t = trade_map[tid]
            product_updates = dict(updates.get("product", {}))
            if isinstance(t.product, IRS):
                new_product = replace(t.product, **product_updates)
            elif isinstance(t.product, FXForward):
                new_product = replace(t.product, **product_updates)
            else:
                new_product = t.product
            trade_map[tid] = replace(t, product=new_product)
        for t in add:
            trade_map[t.trade_id] = t

        updated_pf = replace(self.state.snapshot.portfolio, trades=tuple(trade_map.values()))
        updated = replace(self.state.snapshot, portfolio=updated_pf)
        self._apply_snapshot(updated, changed="portfolio")

    def freeze(self) -> dict:
        return self.state.snapshot.to_dict()

    def _apply_snapshot(self, snapshot: XVASnapshot, changed: str) -> None:
        key = snapshot.stable_key()
        if key == self.state.snapshot_key:
            return
        mapped = map_snapshot(snapshot)
        self.state.snapshot = snapshot
        self.state.snapshot_key = key
        self.state.mapped_inputs = mapped
        self.state.rebuild_counts[changed] = self.state.rebuild_counts.get(changed, 0) + 1


__all__ = [
    "RunnerAdapter",
    "SessionState",
    "_ConsoleRunReporter",
    "XVAEngine",
    "XVASession",
]
