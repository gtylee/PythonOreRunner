from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import csv
import math
import numpy as np
import warnings
import xml.etree.ElementTree as ET

from pythonore.domain.dataclasses import MarketData, MarketQuote, XVASnapshot
from pythonore.mapping.mapper import map_snapshot
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.runtime import XVAEngine


@dataclass(frozen=True)
class PythonSensitivityEntry:
    raw_quote_key: str
    normalized_factor: str
    ore_factor: str
    shift_size: float
    base_value: float
    base_metric_value: float
    bumped_up_metric_value: float
    bumped_down_metric_value: float
    delta: float


@dataclass(frozen=True)
class ORESensitivityEntry:
    factor: str
    normalized_factor: str
    shift_size: float
    base_xva: float
    delta: float
    netting_set_id: str = ""
    trade_id: str = ""


@dataclass(frozen=True)
class SensitivityComparisonEntry:
    normalized_factor: str
    python_quote_key: str
    ore_factor: str
    shift_size: float
    python_delta: float
    ore_delta: float
    delta_diff: float
    delta_rel_diff: float


class OreSnapshotPythonLgmSensitivityComparator:
    """Finite-difference XVA sensitivities for ore_snapshot plus ORE comparison."""

    def __init__(self, engine: Optional[XVAEngine] = None):
        self.engine = engine or XVAEngine.python_lgm_default(fallback_to_swig=True)

    @classmethod
    def from_case_dir(
        cls,
        case_dir: str | Path,
        ore_file: str = "ore.xml",
        engine: Optional[XVAEngine] = None,
    ) -> tuple["OreSnapshotPythonLgmSensitivityComparator", XVASnapshot]:
        from pythonore.io.loader import XVALoader

        case_dir = Path(case_dir)
        snapshot = XVALoader.from_files(str(case_dir / "Input"), ore_file=ore_file)
        return cls(engine=engine), snapshot

    def compute_python_sensitivities(
        self,
        snapshot: XVASnapshot,
        metric: str = "CVA",
        factor_shifts: Optional[Dict[str, float]] = None,
        bump_modes: Optional[Dict[str, str]] = None,
        curve_factor_specs: Optional[Dict[str, Dict[str, object]]] = None,
        factor_labels: Optional[Dict[str, str]] = None,
        apply_portfolio_pruning: bool = False,
        output_mode: str = "derivative",
    ) -> List[PythonSensitivityEntry]:
        factor_shifts = factor_shifts or {}
        bump_modes = bump_modes or {}
        curve_factor_specs = curve_factor_specs or {}
        factor_labels = factor_labels or {}
        quote_map = self._discover_supported_quotes(snapshot)
        predicate = _portfolio_factor_predicate(snapshot) if apply_portfolio_pruning else (lambda _: True)
        if factor_shifts:
            requested = [
                f for f in factor_shifts
                if (f in quote_map or f in curve_factor_specs) and predicate(f)
            ]
        else:
            requested = [f for f in sorted(set(quote_map).union(curve_factor_specs)) if predicate(f)]
        if not requested:
            return []

        fast_entries = self._compute_python_npv_sensitivities_fast(
            snapshot,
            metric=metric,
            requested=requested,
            factor_shifts=factor_shifts,
            bump_modes=bump_modes,
            curve_factor_specs=curve_factor_specs,
            factor_labels=factor_labels,
            output_mode=output_mode,
        )
        if fast_entries is not None:
            return fast_entries

        native_snapshot = self.engine.prepare_sensitivity_snapshot(
            snapshot,
            curve_fit_mode="ore_fit",
            use_ore_output_curves=False,
            freeze_float_spreads=True,
        )
        frozen_float_spreads = native_snapshot.config.params.get("python.frozen_float_spreads")
        base_result = self.engine.create_session(native_snapshot).run(return_cubes=False)
        base_metric_value = self._result_metric_value(base_result, metric)

        entries: List[PythonSensitivityEntry] = []
        for normalized_factor in requested:
            curve_spec = curve_factor_specs.get(normalized_factor)
            quote_entries = quote_map.get(normalized_factor, [])
            bump_mode = bump_modes.get(normalized_factor, "quote_value")
            if bump_mode == "survival_probability":
                quote_entries = self._hazard_curve_quote_entries(native_snapshot, normalized_factor) or quote_entries
            quote = quote_entries[0][1] if quote_entries else None
            default_shift = 1.0e-4 if quote is None else self._default_shift_for_quote(quote)
            shift_size = float(factor_shifts.get(normalized_factor, default_shift))
            if shift_size == 0.0:
                continue
            if curve_spec is not None:
                up_snapshot = self._bump_snapshot_curve(native_snapshot, curve_spec, shift_size, frozen_float_spreads)
                down_snapshot = self._bump_snapshot_curve(native_snapshot, curve_spec, -shift_size, frozen_float_spreads)
                raw_label = f"curve:{curve_spec.get('ore_factor', normalized_factor)}"
                base_value = 0.0
            else:
                if quote is None:
                    continue
                up_values, down_values = self._bumped_quote_values(
                    quotes=[q for _, q in quote_entries],
                    normalized_factor=normalized_factor,
                    shift_size=shift_size,
                    bump_mode=bump_mode,
                )
                up_snapshot = self.engine.prepare_sensitivity_snapshot(
                    self._bump_snapshot_quotes(native_snapshot, quote_entries, up_values),
                    curve_fit_mode="ore_fit",
                    use_ore_output_curves=False,
                    frozen_float_spreads=frozen_float_spreads,
                )
                down_snapshot = self.engine.prepare_sensitivity_snapshot(
                    self._bump_snapshot_quotes(native_snapshot, quote_entries, down_values),
                    curve_fit_mode="ore_fit",
                    use_ore_output_curves=False,
                    frozen_float_spreads=frozen_float_spreads,
                )
                raw_label = _quote_label([q for _, q in quote_entries])
                base_value = float(quote.value)
            up_result = self.engine.create_session(up_snapshot).run(return_cubes=False)
            down_result = self.engine.create_session(down_snapshot).run(return_cubes=False)
            up_metric = self._result_metric_value(up_result, metric)
            down_metric = self._result_metric_value(down_result, metric)
            if output_mode == "bump_change":
                delta = up_metric - base_metric_value
            else:
                delta = (up_metric - down_metric) / (2.0 * shift_size)
            entries.append(
                PythonSensitivityEntry(
                    raw_quote_key=raw_label,
                    normalized_factor=normalized_factor,
                    ore_factor=str(
                        factor_labels.get(normalized_factor)
                        or (
                            curve_spec.get("ore_factor")
                            if curve_spec is not None
                            else _normalized_factor_to_ore_factor(normalized_factor)
                        )
                    ),
                    shift_size=shift_size,
                    base_value=base_value,
                    base_metric_value=base_metric_value,
                    bumped_up_metric_value=up_metric,
                    bumped_down_metric_value=down_metric,
                    delta=delta,
                )
            )
        return entries

    def load_ore_zero_sensitivities(
        self,
        output_dir: str | Path,
        metric: str = "CVA",
        netting_set_id: Optional[str] = None,
    ) -> List[ORESensitivityEntry]:
        output_dir = Path(output_dir)
        file_name = f"xva_zero_sensitivity_{metric.lower()}.csv"
        csv_path = self._resolve_ore_sensitivity_file(output_dir, file_name)
        if not csv_path.exists():
            return []

        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        entries: List[ORESensitivityEntry] = []
        for row in rows:
            trade_id = (row.get("TradeId") or "").strip()
            ns = (row.get("NettingSetId") or "").strip()
            if trade_id:
                continue
            if netting_set_id is not None and ns != netting_set_id:
                continue
            factor = (row.get("Factor_1") or "").strip()
            normalized = normalize_ore_factor(factor)
            if normalized is None:
                continue
            entries.append(
                ORESensitivityEntry(
                    factor=factor,
                    normalized_factor=normalized,
                    shift_size=_float_or_zero(row.get("ShiftSize_1")),
                    base_xva=_float_or_zero(row.get("Base XVA")),
                    delta=_float_or_zero(row.get("Delta")),
                    netting_set_id=ns,
                    trade_id=trade_id,
                )
            )
        return entries

    def compare(
        self,
        snapshot: XVASnapshot,
        metric: str = "CVA",
        output_dir: str | Path | None = None,
        netting_set_id: Optional[str] = None,
        factor_shifts: Optional[Dict[str, float]] = None,
        curve_factor_specs: Optional[Dict[str, Dict[str, object]]] = None,
        factor_labels: Optional[Dict[str, str]] = None,
        native_only_output_mode: str = "bump_change",
    ) -> Dict[str, object]:
        if output_dir is None:
            output_dir = snapshot.config.params.get("outputPath", "")
        if not output_dir:
            raise EngineRunError("No output_dir provided and snapshot.config.params['outputPath'] is empty")

        ore_entries = self.load_ore_zero_sensitivities(output_dir, metric=metric, netting_set_id=netting_set_id)
        factor_shifts = dict(factor_shifts or {})
        curve_factor_specs = dict(curve_factor_specs or {})
        factor_labels = dict(factor_labels or {})
        bump_modes: Dict[str, str] = {}
        pruned_count = 0
        if ore_entries:
            ore_curve_factor_specs = _curve_factor_specs_from_ore_entries(ore_entries)
            curve_factor_specs = {**curve_factor_specs, **ore_curve_factor_specs}
            for entry in ore_entries:
                if entry.shift_size != 0.0:
                    factor_shifts.setdefault(entry.normalized_factor, entry.shift_size)
                if entry.normalized_factor.startswith("hazard:"):
                    bump_modes.setdefault(entry.normalized_factor, "survival_probability")
                factor_labels.setdefault(entry.normalized_factor, entry.factor)
        else:
            (
                factor_shifts,
                curve_factor_specs,
                factor_labels,
                pruned_count,
            ) = _prune_native_factor_setup_for_portfolio(
                snapshot,
                factor_shifts=factor_shifts,
                curve_factor_specs=curve_factor_specs,
                factor_labels=factor_labels,
            )
        if not factor_shifts:
            notes = [
                "No ORE zero-sensitivity rows were found; running native finite-difference sensitivities only."
            ]
        else:
            notes = []
        if not ore_entries and pruned_count:
            notes.append(
                f"Pruned {int(pruned_count)} native sensitivity factors that are outside the portfolio currencies/index families."
            )
        python_entries = self.compute_python_sensitivities(
            snapshot,
            metric=metric,
            factor_shifts=factor_shifts,
            bump_modes=bump_modes,
            curve_factor_specs=curve_factor_specs,
            factor_labels=factor_labels,
            apply_portfolio_pruning=not ore_entries,
            output_mode="bump_change" if ore_entries else native_only_output_mode,
        )
        unsupported_prefixes = ("recovery:",)
        unsupported_factors = sorted(
            {
                e.normalized_factor
                for e in python_entries
                if e.normalized_factor.startswith(unsupported_prefixes)
            }
            | {
                e.normalized_factor
                for e in ore_entries
                if e.normalized_factor.startswith(unsupported_prefixes)
            }
        )
        if unsupported_factors:
            notes.append(
                "Recovery sensitivity parity is not implemented in the Python snapshot path and is excluded "
                "from comparison output."
            )
        python_supported = [e for e in python_entries if e.normalized_factor not in unsupported_factors]
        ore_supported = [e for e in ore_entries if e.normalized_factor not in unsupported_factors]
        python_map = {entry.normalized_factor: entry for entry in python_supported}
        ore_map = {entry.normalized_factor: entry for entry in ore_supported}

        comparisons: List[SensitivityComparisonEntry] = []
        for normalized_factor in sorted(set(python_map).intersection(ore_map)):
            py = python_map[normalized_factor]
            ore = ore_map[normalized_factor]
            rel = 0.0 if ore.delta == 0.0 else (py.delta - ore.delta) / abs(ore.delta)
            comparisons.append(
                SensitivityComparisonEntry(
                    normalized_factor=normalized_factor,
                    python_quote_key=py.raw_quote_key,
                    ore_factor=ore.factor,
                    shift_size=py.shift_size,
                    python_delta=py.delta,
                    ore_delta=ore.delta,
                    delta_diff=py.delta - ore.delta,
                    delta_rel_diff=rel,
                )
            )

        return {
            "metric": metric,
            "python": python_supported,
            "ore": ore_supported,
            "comparisons": comparisons,
            "unmatched_python": sorted(set(python_map).difference(ore_map)),
            "unmatched_ore": sorted(set(ore_map).difference(python_map)),
            "unsupported_python": [e for e in python_entries if e.normalized_factor in unsupported_factors],
            "unsupported_ore": [e for e in ore_entries if e.normalized_factor in unsupported_factors],
            "unsupported_factors": unsupported_factors,
            "notes": notes,
        }

    def _result_metric_value(self, result, metric: str) -> float:
        metric_name = str(metric or "").strip().upper()
        if metric_name in {"NPV", "PV"}:
            return float(getattr(result, "pv_total", 0.0))
        return float(getattr(result, "xva_by_metric", {}).get(metric_name, 0.0))

    def _compute_python_npv_sensitivities_fast(
        self,
        snapshot: XVASnapshot,
        *,
        metric: str,
        requested: Sequence[str],
        factor_shifts: Dict[str, float],
        bump_modes: Dict[str, str],
        curve_factor_specs: Dict[str, Dict[str, object]],
        factor_labels: Dict[str, str],
        output_mode: str,
    ) -> Optional[List[PythonSensitivityEntry]]:
        metric_name = str(metric or "").strip().upper()
        if metric_name not in {"NPV", "PV"}:
            return None
        if not self._supports_fast_npv_sensitivity(snapshot):
            return None

        native_snapshot = self.engine.prepare_sensitivity_snapshot(
            snapshot,
            curve_fit_mode="ore_fit",
            use_ore_output_curves=False,
            freeze_float_spreads=True,
        )
        frozen_float_spreads = native_snapshot.config.params.get("python.frozen_float_spreads")
        base_metric_value = self._price_snapshot_t0_npv(native_snapshot)

        entries: List[PythonSensitivityEntry] = []
        for normalized_factor in requested:
            curve_spec = curve_factor_specs.get(normalized_factor)
            quote_map = self._discover_supported_quotes(native_snapshot)
            quote_entries = quote_map.get(normalized_factor, [])
            bump_mode = bump_modes.get(normalized_factor, "quote_value")
            if bump_mode == "survival_probability":
                quote_entries = self._hazard_curve_quote_entries(native_snapshot, normalized_factor) or quote_entries
            quote = quote_entries[0][1] if quote_entries else None
            default_shift = 1.0e-4 if quote is None else self._default_shift_for_quote(quote)
            shift_size = float(factor_shifts.get(normalized_factor, default_shift))
            if shift_size == 0.0:
                continue
            if curve_spec is not None:
                up_snapshot = self._bump_snapshot_curve(native_snapshot, curve_spec, shift_size, frozen_float_spreads)
                down_snapshot = self._bump_snapshot_curve(native_snapshot, curve_spec, -shift_size, frozen_float_spreads)
                raw_label = f"curve:{curve_spec.get('ore_factor', normalized_factor)}"
                base_value = 0.0
            else:
                if quote is None:
                    continue
                up_values, down_values = self._bumped_quote_values(
                    quotes=[q for _, q in quote_entries],
                    normalized_factor=normalized_factor,
                    shift_size=shift_size,
                    bump_mode=bump_mode,
                )
                up_snapshot = self.engine.prepare_sensitivity_snapshot(
                    self._bump_snapshot_quotes(native_snapshot, quote_entries, up_values),
                    curve_fit_mode="ore_fit",
                    use_ore_output_curves=False,
                    frozen_float_spreads=frozen_float_spreads,
                )
                down_snapshot = self.engine.prepare_sensitivity_snapshot(
                    self._bump_snapshot_quotes(native_snapshot, quote_entries, down_values),
                    curve_fit_mode="ore_fit",
                    use_ore_output_curves=False,
                    frozen_float_spreads=frozen_float_spreads,
                )
                raw_label = _quote_label([q for _, q in quote_entries])
                base_value = float(quote.value)
            up_metric = self._price_snapshot_t0_npv(up_snapshot)
            down_metric = self._price_snapshot_t0_npv(down_snapshot)
            if output_mode == "bump_change":
                delta = up_metric - base_metric_value
            else:
                delta = (up_metric - down_metric) / (2.0 * shift_size)
            entries.append(
                PythonSensitivityEntry(
                    raw_quote_key=raw_label,
                    normalized_factor=normalized_factor,
                    ore_factor=str(
                        factor_labels.get(normalized_factor)
                        or (
                            curve_spec.get("ore_factor")
                            if curve_spec is not None
                            else _normalized_factor_to_ore_factor(normalized_factor)
                        )
                    ),
                    shift_size=shift_size,
                    base_value=base_value,
                    base_metric_value=base_metric_value,
                    bumped_up_metric_value=up_metric,
                    bumped_down_metric_value=down_metric,
                    delta=delta,
                )
            )
        return entries

    def _supports_fast_npv_sensitivity(self, snapshot: XVASnapshot) -> bool:
        adapter = getattr(self.engine, "adapter", None)
        if adapter is None or not hasattr(adapter, "_extract_inputs") or not hasattr(adapter, "_ensure_py_lgm_imports"):
            return False
        try:
            adapter._ensure_py_lgm_imports()
            mapped = map_snapshot(snapshot)
            inputs = adapter._extract_inputs(snapshot, mapped)
        except Exception:
            return False
        if inputs.unsupported:
            return False
        if not inputs.trade_specs:
            return False
        return all(spec.kind == "IRS" and spec.legs is not None for spec in inputs.trade_specs)

    def _price_snapshot_t0_npv(self, snapshot: XVASnapshot) -> float:
        adapter = self.engine.adapter
        adapter._ensure_py_lgm_imports()
        mapped = map_snapshot(snapshot)
        inputs = adapter._extract_inputs(snapshot, mapped)
        if inputs.unsupported:
            bad = ", ".join(sorted({f"{t.trade_id}:{t.trade_type}" for t in inputs.unsupported}))
            raise EngineRunError(f"Fast NPV sensitivity path only supports IRS portfolios, got unsupported trades: {bad}")
        model = adapter._lgm_mod.LGM1F(
            adapter._lgm_mod.LGMParams(
                alpha_times=tuple(float(x) for x in inputs.lgm_params["alpha_times"]),
                alpha_values=tuple(float(x) for x in inputs.lgm_params["alpha_values"]),
                kappa_times=tuple(float(x) for x in inputs.lgm_params["kappa_times"]),
                kappa_values=tuple(float(x) for x in inputs.lgm_params["kappa_values"]),
                shift=float(inputs.lgm_params["shift"]),
                scaling=float(inputs.lgm_params["scaling"]),
            )
        )
        pricing_backend = self._resolve_swap_npv_backend()
        total = 0.0
        for spec in inputs.trade_specs:
            if spec.kind != "IRS" or spec.legs is None:
                raise EngineRunError(f"Fast NPV sensitivity path does not support trade kind '{spec.kind}'")
            p_disc = inputs.discount_curves[spec.ccy]
            fwd_tenor = str(spec.legs.get("float_index_tenor", "")).upper()
            p_fwd = inputs.forward_curves_by_tenor.get(spec.ccy, {}).get(
                fwd_tenor,
                inputs.forward_curves.get(spec.ccy, p_disc),
            )
            realized_coupon = adapter._compute_realized_float_coupons(
                model=model,
                p0_disc=p_disc,
                p0_fwd=p_fwd,
                legs=spec.legs,
                sim_times=np.asarray([0.0], dtype=float),
                x_paths_on_sim_grid=np.asarray([[0.0]], dtype=float),
            )
            if pricing_backend is None:
                value = float(
                    adapter._irs_utils.swap_npv_from_ore_legs_dual_curve(
                        model,
                        p_disc,
                        p_fwd,
                        spec.legs,
                        0.0,
                        np.asarray([0.0], dtype=float),
                        realized_float_coupon=realized_coupon,
                    )[0]
                )
            else:
                torch_curve_ctor, torch_pricer, device = pricing_backend
                sample_disc = _sample_times_for_legs(spec.legs, include_float_coupon_dates=False)
                sample_fwd = _sample_times_for_legs(spec.legs, include_float_coupon_dates=True)
                disc_curve = torch_curve_ctor(
                    times=sample_disc,
                    dfs=np.asarray([float(p_disc(float(t))) for t in sample_disc], dtype=float),
                    device=device,
                )
                fwd_curve = torch_curve_ctor(
                    times=sample_fwd,
                    dfs=np.asarray([float(p_fwd(float(t))) for t in sample_fwd], dtype=float),
                    device=device,
                )
                value = float(
                    torch_pricer(
                        model,
                        disc_curve,
                        fwd_curve,
                        spec.legs,
                        0.0,
                        np.asarray([0.0], dtype=float),
                        realized_float_coupon=realized_coupon,
                        return_numpy=True,
                    )[0]
                )
            total += value
        return float(total)

    def _resolve_swap_npv_backend(self):
        try:
            import torch
            from pythonore.compute.lgm_torch_xva import TorchDiscountCurve, swap_npv_from_ore_legs_dual_curve_torch
        except Exception:
            return None
        device = "mps" if bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        return TorchDiscountCurve, swap_npv_from_ore_legs_dual_curve_torch, device

    def _resolve_ore_sensitivity_file(self, output_dir: str | Path, file_name: str) -> Path:
        output_dir = Path(output_dir)
        candidates = [
            output_dir / file_name,
            output_dir / "xva_sensitivity" / file_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _discover_supported_quotes(self, snapshot: XVASnapshot) -> Dict[str, List[Tuple[int, MarketQuote]]]:
        out: Dict[str, List[Tuple[int, MarketQuote]]] = {}
        discount_family_by_ccy = _discount_curve_family_by_currency(snapshot)
        for idx, quote in enumerate(snapshot.market.raw_quotes):
            seen: set[str] = set()
            key = str(quote.key)
            normalized = normalize_raw_quote_key(key)
            if _should_skip_generic_normalization(key, normalized, discount_family_by_ccy):
                normalized = None
            if normalized is not None:
                seen.add(normalized)
                out.setdefault(normalized, []).append((idx, quote))
            discount_normalized = normalize_raw_discount_quote_key(key, discount_family_by_ccy)
            if discount_normalized is not None and discount_normalized not in seen:
                out.setdefault(discount_normalized, []).append((idx, quote))
        return out

    def _bump_snapshot_quotes(
        self,
        snapshot: XVASnapshot,
        quote_entries: Sequence[Tuple[int, MarketQuote]],
        new_values: Sequence[float],
    ) -> XVASnapshot:
        quotes = list(snapshot.market.raw_quotes)
        for (quote_index, orig), new_value in zip(quote_entries, new_values):
            quotes[quote_index] = replace(orig, value=float(new_value))
        market = replace(snapshot.market, raw_quotes=tuple(quotes))
        return replace(snapshot, market=market)

    def _bump_snapshot_curve(
        self,
        snapshot: XVASnapshot,
        curve_spec: Dict[str, object],
        shift_size: float,
        frozen_float_spreads: Optional[Dict[str, List[float]]] = None,
    ) -> XVASnapshot:
        curve_node_shocks = dict(snapshot.config.params.get("python.curve_node_shocks", {}))
        kind = str(curve_spec.get("kind", "")).lower()
        ccy = str(curve_spec.get("ccy", "")).upper()
        node_times = list(curve_spec.get("node_times", []))
        target_time = float(curve_spec.get("target_time", 0.0))
        node_shifts = [shift_size if abs(float(t) - target_time) < 1.0e-12 else 0.0 for t in node_times]
        if kind == "discount":
            discount = dict(curve_node_shocks.get("discount", {}))
            discount[ccy] = {"node_times": node_times, "node_shifts": node_shifts}
            curve_node_shocks["discount"] = discount
        elif kind == "forward":
            tenor = str(curve_spec.get("index_tenor", "")).upper()
            forward = dict(curve_node_shocks.get("forward", {}))
            ccy_map = dict(forward.get(ccy, {}))
            ccy_map[tenor] = {"node_times": node_times, "node_shifts": node_shifts}
            forward[ccy] = ccy_map
            curve_node_shocks["forward"] = forward
        elif kind == "credit":
            return self._bump_snapshot_credit_curve(snapshot, curve_spec, shift_size)
        return self.engine.prepare_sensitivity_snapshot(
            snapshot,
            curve_node_shocks=curve_node_shocks,
            curve_fit_mode="ore_fit",
            use_ore_output_curves=False,
            frozen_float_spreads=frozen_float_spreads,
        )

    def _default_shift_for_quote(self, quote: MarketQuote) -> float:
        key = str(quote.key).upper()
        if key.startswith("FX/"):
            return max(abs(float(quote.value)) * 0.01, 1.0e-6)
        if key.startswith("RECOVERY_RATE/"):
            return 0.01
        return 1.0e-4

    def _hazard_curve_quote_entries(
        self,
        snapshot: XVASnapshot,
        normalized_factor: str,
    ) -> List[Tuple[int, MarketQuote]]:
        parts = normalized_factor.split(":")
        if len(parts) < 3 or parts[0] != "hazard":
            return []
        cpty = parts[1].upper()
        out: List[Tuple[int, MarketQuote]] = []
        for idx, quote in enumerate(snapshot.market.raw_quotes):
            key = str(quote.key).strip().upper()
            if key.startswith(f"HAZARD_RATE/RATE/{cpty}/"):
                tenor = _parse_tenor_to_years(key.split("/")[-1])
                if tenor is not None and tenor > 0.0:
                    out.append((idx, quote))
        out.sort(key=lambda x: _parse_tenor_to_years(str(x[1].key).split("/")[-1]) or 0.0)
        return out

    def _bump_snapshot_credit_curve(
        self,
        snapshot: XVASnapshot,
        curve_spec: Dict[str, object],
        shift_size: float,
    ) -> XVASnapshot:
        name = str(curve_spec.get("name", "")).upper()
        coarse_nodes = [float(x) for x in curve_spec.get("node_times", [])]
        target_time = float(curve_spec.get("target_time", 0.0))
        if not name or not coarse_nodes or target_time <= 0.0:
            return snapshot
        quote_entries = self._hazard_curve_quote_entries(snapshot, f"hazard:{name}:{int(target_time)}Y")
        if not quote_entries:
            return snapshot
        credit_curves = dict(snapshot.config.params.get("python.credit_survival_curves", {}))
        credit_curves[name] = _build_credit_survival_curve_shock(
            [q for _, q in quote_entries],
            coarse_node_times=coarse_nodes,
            target_time=target_time,
            shift_size=shift_size,
        )
        params = dict(snapshot.config.params)
        params["python.credit_survival_curves"] = credit_curves
        return self.engine.prepare_sensitivity_snapshot(
            replace(snapshot, config=replace(snapshot.config, params=params)),
            curve_fit_mode="ore_fit",
            use_ore_output_curves=False,
            frozen_float_spreads=snapshot.config.params.get("python.frozen_float_spreads"),
        )

    def _bumped_quote_values(
        self,
        quotes: Sequence[MarketQuote],
        normalized_factor: str,
        shift_size: float,
        bump_mode: str,
    ) -> Tuple[List[float], List[float]]:
        base_values = [float(q.value) for q in quotes]
        if bump_mode != "survival_probability":
            return (
                [base_value + shift_size for base_value in base_values],
                [base_value - shift_size for base_value in base_values],
            )

        tenor_years = _tenor_years_from_normalized_factor(normalized_factor)
        if tenor_years is None or tenor_years <= 0.0:
            return (
                [base_value + shift_size for base_value in base_values],
                [base_value - shift_size for base_value in base_values],
            )
        return _bump_hazard_curve_quotes_by_survival_node(quotes, tenor_years, shift_size)


def normalize_raw_quote_key(raw_key: str) -> Optional[str]:
    parts = str(raw_key).strip().upper().split("/")
    if len(parts) >= 3 and parts[0] == "FX":
        return f"fx:{parts[1]}{parts[2]}"
    if len(parts) >= 4 and parts[0] == "ZERO" and parts[1] == "RATE":
        return f"zero:{parts[2]}:{parts[3]}"
    if len(parts) >= 5 and parts[0] == "MM" and parts[1] == "RATE":
        ccy = parts[2]
        index_tenor = parts[4]
        tenor = parts[-1]
        if index_tenor in ("1D", "ON", "O/N"):
            return f"zero:{ccy}:{tenor}"
        return f"fwd:{ccy}:{index_tenor}:{tenor}"
    if len(parts) >= 6 and parts[0] == "IR_SWAP" and parts[1] == "RATE":
        ccy = parts[2]
        index_tenor = parts[4]
        tenor = parts[-1]
        if index_tenor in ("1D", "ON", "O/N"):
            return f"zero:{ccy}:{tenor}"
        return f"fwd:{ccy}:{index_tenor}:{tenor}"
    if len(parts) >= 6 and parts[0] == "HAZARD_RATE":
        return f"hazard:{parts[2]}:{parts[-1]}"
    if len(parts) >= 5 and parts[0] == "RECOVERY_RATE":
        return f"recovery:{parts[2]}"
    return None


def normalize_ore_factor(factor: str) -> Optional[str]:
    parts = str(factor).strip().split("/")
    if len(parts) < 2:
        return None
    head = parts[0]
    if head == "DiscountCurve" and len(parts) >= 4:
        return f"zero:{parts[1].upper()}:{parts[-1].upper()}"
    if head == "IndexCurve" and len(parts) >= 4:
        index_name = parts[1].upper()
        ccy = index_name.split("-")[0]
        index_tenor = index_name.split("-")[-1]
        return f"fwd:{ccy}:{index_tenor}:{parts[-1].upper()}"
    if head == "SurvivalProbability" and len(parts) >= 4:
        return f"hazard:{parts[1].upper()}:{parts[-1].upper()}"
    if head == "FXSpot" and len(parts) >= 2:
        return f"fx:{parts[1].upper()}"
    return None


def _normalized_factor_to_ore_factor(normalized_factor: str) -> str:
    parts = str(normalized_factor).strip().split(":")
    if len(parts) < 2:
        return normalized_factor
    head = parts[0].lower()
    if head == "zero" and len(parts) >= 3:
        return f"DiscountCurve/{parts[1].upper()}/0/{parts[2].upper()}"
    if head == "fwd" and len(parts) >= 4:
        return f"IndexCurve/{parts[1].upper()}-{parts[2].upper()}/0/{parts[3].upper()}"
    if head == "hazard" and len(parts) >= 3:
        return f"SurvivalProbability/{parts[1].upper()}/0/{parts[2].upper()}"
    if head == "recovery" and len(parts) >= 2:
        return f"RecoveryRate/{parts[1].upper()}"
    if head == "fx" and len(parts) >= 2:
        return f"FXSpot/{parts[1].upper()}"
    return normalized_factor


def _sample_times_for_legs(
    legs: Mapping[str, np.ndarray],
    *,
    include_float_coupon_dates: bool,
) -> np.ndarray:
    samples = [0.0]
    for key in ("fixed_pay_time", "float_start_time", "float_end_time", "float_pay_time"):
        values = np.asarray(legs.get(key, []), dtype=float)
        if values.size:
            samples.extend(float(x) for x in values if float(x) >= 0.0)
    if include_float_coupon_dates:
        values = np.asarray(legs.get("float_fixing_time", []), dtype=float)
        if values.size:
            samples.extend(float(x) for x in values if float(x) >= 0.0)
    unique = sorted(set(samples))
    if len(unique) < 2:
        unique.append(max(unique[0] + 1.0, 1.0))
    return np.asarray(unique, dtype=float)


def _prune_native_factor_setup_for_portfolio(
    snapshot: XVASnapshot,
    *,
    factor_shifts: Dict[str, float],
    curve_factor_specs: Dict[str, Dict[str, object]],
    factor_labels: Dict[str, str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, object]], Dict[str, str], int]:
    predicate = _portfolio_factor_predicate(snapshot)
    keep = {factor for factor in factor_shifts if predicate(factor)}
    pruned_count = max(len(factor_shifts) - len(keep), 0)
    return (
        {k: v for k, v in factor_shifts.items() if k in keep},
        {k: v for k, v in curve_factor_specs.items() if k in keep},
        {k: v for k, v in factor_labels.items() if k in keep},
        pruned_count,
    )


def _portfolio_factor_predicate(snapshot: XVASnapshot):
    relevant_ccys: set[str] = set()
    relevant_forward_keys: set[Tuple[str, str]] = set()
    relevant_fx_pairs: set[str] = set()
    relevant_credit_names: set[str] = set()

    for trade in getattr(snapshot.portfolio, "trades", ()):
        relevant_credit_names.add(str(getattr(trade, "counterparty", "")).strip().upper())
        product = getattr(trade, "product", None)
        ccy = str(getattr(product, "ccy", "") or getattr(product, "currency", "")).strip().upper()
        if ccy:
            relevant_ccys.add(ccy)
        float_index = str(getattr(product, "float_index", "")).strip().upper()
        if ccy and float_index:
            tenor = float_index.split("-")[-1]
            if tenor:
                relevant_forward_keys.add((ccy, tenor))
        if product is not None and product.__class__.__name__ == "FXForward":
            bought = str(getattr(product, "base_ccy", "")).strip().upper()
            sold = str(getattr(product, "quote_ccy", "")).strip().upper()
            if bought:
                relevant_ccys.add(bought)
            if sold:
                relevant_ccys.add(sold)
            if bought and sold:
                relevant_fx_pairs.add(f"{bought}{sold}")
                relevant_fx_pairs.add(f"{sold}{bought}")

    base_ccy = str(getattr(snapshot.config, "base_currency", "")).strip().upper()
    if base_ccy:
        relevant_ccys.add(base_ccy)

    def _keep(normalized_factor: str) -> bool:
        parts = str(normalized_factor).strip().split(":")
        if not parts:
            return True
        head = parts[0].lower()
        if head == "zero" and len(parts) >= 3:
            return parts[1].upper() in relevant_ccys
        if head == "fwd" and len(parts) >= 4:
            key = (parts[1].upper(), parts[2].upper())
            return key in relevant_forward_keys or parts[1].upper() in relevant_ccys
        if head == "fx" and len(parts) >= 2:
            pair = parts[1].upper()
            return not relevant_fx_pairs or pair in relevant_fx_pairs
        if head in {"hazard", "recovery"} and len(parts) >= 2:
            return not relevant_credit_names or parts[1].upper() in relevant_credit_names
        return True

    return _keep


def _curve_factor_specs_from_ore_entries(
    ore_entries: Sequence[ORESensitivityEntry],
) -> Dict[str, Dict[str, object]]:
    grouped_times: Dict[Tuple[str, str, str], List[float]] = {}
    parsed_rows: List[Tuple[ORESensitivityEntry, Optional[Dict[str, object]]]] = []
    for entry in ore_entries:
        spec = _parse_ore_curve_factor(entry.factor)
        parsed_rows.append((entry, spec))
        if spec is None:
            continue
        key = _curve_spec_group_key(spec)
        grouped_times.setdefault(key, []).append(float(spec["target_time"]))

    out: Dict[str, Dict[str, object]] = {}
    for entry, spec in parsed_rows:
        if spec is None:
            continue
        key = _curve_spec_group_key(spec)
        spec = dict(spec)
        spec["node_times"] = sorted(set(grouped_times.get(key, [float(spec["target_time"])])))
        spec["ore_factor"] = entry.factor
        out[entry.normalized_factor] = spec
    return out


def _parse_ore_curve_factor(factor: str) -> Optional[Dict[str, object]]:
    parts = str(factor).strip().split("/")
    if len(parts) < 4:
        return None
    head = parts[0]
    if head == "DiscountCurve":
        return {
            "kind": "discount",
            "ccy": parts[1].upper(),
            "target_time": _parse_tenor_to_years(parts[-1]),
        }
    if head == "IndexCurve":
        index_name = parts[1].upper()
        return {
            "kind": "forward",
            "ccy": index_name.split("-")[0],
            "index_tenor": index_name.split("-")[-1],
            "target_time": _parse_tenor_to_years(parts[-1]),
        }
    if head == "SurvivalProbability":
        return {
            "kind": "credit",
            "name": parts[1].upper(),
            "target_time": _parse_tenor_to_years(parts[-1]),
        }
    return None


def _curve_spec_group_key(spec: Dict[str, object]) -> Tuple[str, str, str]:
    return (
        str(spec.get("kind", "")),
        str(spec.get("ccy", spec.get("name", ""))).upper(),
        str(spec.get("index_tenor", "")).upper(),
    )


def normalize_raw_discount_quote_key(
    raw_key: str,
    discount_family_by_ccy: Dict[str, str],
) -> Optional[str]:
    parts = str(raw_key).strip().upper().split("/")
    if len(parts) < 3:
        return None
    ccy = parts[2]
    family = str(discount_family_by_ccy.get(ccy, "")).upper()
    if not family:
        return None
    if parts[0] == "MM" and parts[1] == "RATE":
        return f"zero:{ccy}:{parts[-1]}"
    if parts[0] == "FRA" and parts[1] == "RATE":
        if family in ("1D", "ON", "O/N"):
            return None
        if len(parts) > 4 and parts[-1] == family:
            return f"zero:{ccy}:{parts[-2]}"
        return None
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        index_tenor = parts[4] if len(parts) > 5 else ""
        if family in ("1D", "ON", "O/N"):
            if index_tenor in ("1D", "ON", "O/N"):
                return f"zero:{ccy}:{parts[-1]}"
            return None
        if index_tenor == family:
            return f"zero:{ccy}:{parts[-1]}"
    return None


def _should_skip_generic_normalization(
    raw_key: str,
    normalized_factor: Optional[str],
    discount_family_by_ccy: Dict[str, str],
) -> bool:
    if normalized_factor is None or not normalized_factor.startswith("zero:"):
        return False
    parts = str(raw_key).strip().upper().split("/")
    if len(parts) < 3 or parts[2] not in discount_family_by_ccy:
        return False
    family = discount_family_by_ccy[parts[2]]
    if family in ("", "1D", "ON", "O/N"):
        return False
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        index_tenor = parts[4] if len(parts) > 5 else ""
        return index_tenor in ("1D", "ON", "O/N")
    return False


def _discount_curve_family_by_currency(snapshot: XVASnapshot) -> Dict[str, str]:
    todaysmarket_xml = snapshot.config.xml_buffers.get("todaysmarket.xml")
    if not todaysmarket_xml:
        return {}
    try:
        tm_root = ET.fromstring(todaysmarket_xml)
    except Exception as exc:
        warnings.warn(
            f"Failed to parse todaysmarket.xml when resolving discount curve families: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return {}
    config_id = (
        str(snapshot.config.params.get("market.simulation", "")).strip()
        or str(snapshot.config.params.get("market.pricing", "")).strip()
        or "default"
    )
    cfg = tm_root.find(f"./Configuration[@id='{config_id}']")
    if cfg is None:
        return {}
    disc_curves_id = (cfg.findtext("./DiscountingCurvesId") or "").strip()
    if not disc_curves_id:
        return {}
    disc_curves = tm_root.find(f"./DiscountingCurves[@id='{disc_curves_id}']")
    if disc_curves is None:
        return {}
    out: Dict[str, str] = {}
    for node in disc_curves.findall("./DiscountingCurve"):
        ccy = (node.attrib.get("currency", "") or "").strip().upper()
        handle = (node.text or "").strip()
        if not ccy or not handle:
            continue
        curve_name = _handle_to_curve_name(tm_root, handle).upper()
        family = _curve_family_from_source_column(curve_name)
        if family:
            out[ccy] = family
    return out


def _handle_to_curve_name(tm_root: ET.Element, handle: str) -> str:
    for yc_group in tm_root.findall("./YieldCurves"):
        for yc in yc_group.findall("./YieldCurve"):
            if (yc.text or "").strip() == handle:
                name = yc.attrib.get("name", "").strip()
                if name:
                    return name
    for fc_group in tm_root.findall("./IndexForwardingCurves"):
        for idx_elem in fc_group.findall("./Index"):
            if (idx_elem.text or "").strip() == handle:
                name = idx_elem.attrib.get("name", "").strip()
                if name:
                    return name
    return ""


def _curve_family_from_source_column(source_column: str) -> str:
    txt = str(source_column).strip().upper()
    if not txt:
        return ""
    for tenor in ("ON", "O/N", "1D", "1W", "1M", "3M", "6M", "12M"):
        if txt.endswith(f"-{tenor}") or txt.endswith(tenor):
            return tenor
    if any(tag in txt for tag in ("EONIA", "FEDFUNDS", "SONIA", "TOIS", "TONAR")):
        return "1D"
    return ""


def _float_or_zero(value: object) -> float:
    if value in (None, "", "#N/A"):
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _quote_label(quotes: Sequence[MarketQuote]) -> str:
    if not quotes:
        return ""
    first = str(quotes[0].key)
    if len(quotes) == 1:
        return first
    return f"{first} (+{len(quotes) - 1} more)"


def _tenor_years_from_normalized_factor(normalized_factor: str) -> Optional[float]:
    parts = normalized_factor.split(":")
    if not parts:
        return None
    tenor = parts[-1]
    return _parse_tenor_to_years(tenor)


def _parse_tenor_to_years(tenor: str) -> Optional[float]:
    s = str(tenor).strip().upper()
    if not s:
        return None
    if s.endswith("Y"):
        return float(s[:-1])
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("W"):
        return float(s[:-1]) * 7.0 / 365.25
    if s.endswith("D"):
        return float(s[:-1]) / 365.25
    return None


def _bump_hazard_curve_quotes_by_survival_node(
    quotes: Sequence[MarketQuote],
    target_tenor_years: float,
    shift_size: float,
) -> Tuple[List[float], List[float]]:
    times = []
    values = []
    for q in quotes:
        tenor = _parse_tenor_to_years(str(q.key).split("/")[-1])
        if tenor is None or tenor <= 0.0:
            continue
        times.append(float(tenor))
        values.append(float(q.value))
    if not times:
        base_values = [float(q.value) for q in quotes]
        return (
            [x + shift_size for x in base_values],
            [x - shift_size for x in base_values],
        )

    times_arr = np.asarray(times, dtype=float)
    lambdas = np.asarray(values, dtype=float)
    target_idx = int(np.argmin(np.abs(times_arr - float(target_tenor_years))))

    def survival_nodes(vals: np.ndarray) -> np.ndarray:
        out = np.empty_like(vals)
        prev_t = 0.0
        acc = 0.0
        for i, (t, lam) in enumerate(zip(times_arr, vals)):
            acc += float(lam) * max(float(t) - prev_t, 0.0)
            out[i] = math.exp(-acc)
            prev_t = float(t)
        return out

    base_surv = survival_nodes(lambdas)
    prev_surv = 1.0 if target_idx == 0 else float(base_surv[target_idx - 1])
    prev_t = 0.0 if target_idx == 0 else float(times_arr[target_idx - 1])
    this_t = float(times_arr[target_idx])
    dt = max(this_t - prev_t, 1.0e-12)
    eps = 1.0e-10

    def solve(direction: float) -> List[float]:
        desired_surv = min(max(float(base_surv[target_idx]) + direction * shift_size, eps), 1.0 - eps)
        new_vals = np.array(lambdas, copy=True)
        new_vals[target_idx] = -(math.log(desired_surv) - math.log(prev_surv)) / dt
        if target_idx + 1 < new_vals.size:
            next_t = float(times_arr[target_idx + 1])
            next_surv = float(base_surv[target_idx + 1])
            dt_next = max(next_t - this_t, 1.0e-12)
            new_vals[target_idx + 1] = -(math.log(next_surv) - math.log(desired_surv)) / dt_next
        out: List[float] = []
        cursor = 0
        for q in quotes:
            tenor = _parse_tenor_to_years(str(q.key).split("/")[-1])
            if tenor is None or tenor <= 0.0:
                out.append(float(q.value))
            else:
                out.append(float(new_vals[cursor]))
                cursor += 1
        return out

    return solve(+1.0), solve(-1.0)


def _ore_bucket_weight(shift_times: Sequence[float], bucket: int, t: float) -> float:
    tenors = [float(x) for x in shift_times]
    tt = float(t)
    j = int(bucket)
    if j < 0 or j >= len(tenors):
        return 0.0
    t1 = tenors[j]
    if len(tenors) == 1:
        return 1.0
    if j == 0:
        t2 = tenors[j + 1]
        if tt <= t1:
            return 1.0
        if tt <= t2:
            return (t2 - tt) / max(t2 - t1, 1.0e-12)
        return 0.0
    if j == len(tenors) - 1:
        t0 = tenors[j - 1]
        if tt >= t0 and tt <= t1:
            return (tt - t0) / max(t1 - t0, 1.0e-12)
        if tt > t1:
            return 1.0
        return 0.0
    t0 = tenors[j - 1]
    t2 = tenors[j + 1]
    if tt >= t0 and tt <= t1:
        return (tt - t0) / max(t1 - t0, 1.0e-12)
    if tt > t1 and tt <= t2:
        return (t2 - tt) / max(t2 - t1, 1.0e-12)
    return 0.0


def _survival_from_piecewise_hazard(
    times: Sequence[float],
    hazard_times: Sequence[float],
    hazard_rates: Sequence[float],
) -> np.ndarray:
    out = np.empty(len(times), dtype=float)
    prev_t = 0.0
    acc = 0.0
    h_times = [float(x) for x in hazard_times]
    h_rates = [float(x) for x in hazard_rates]
    i = 0
    for j, t in enumerate(times):
        tt = float(t)
        while i < len(h_times) and h_times[i] <= tt + 1.0e-12:
            acc += h_rates[i] * max(h_times[i] - prev_t, 0.0)
            prev_t = h_times[i]
            i += 1
        if tt > prev_t and i > 0:
            acc_now = acc + h_rates[i - 1] * (tt - prev_t)
        elif tt > prev_t and h_rates:
            acc_now = acc + h_rates[0] * (tt - prev_t)
        else:
            acc_now = acc
        out[j] = math.exp(-acc_now)
    return out


def _rebuild_hazard_quotes_from_survival_node_shock(
    quotes: Sequence[MarketQuote],
    coarse_node_times: Sequence[float],
    target_time: float,
    shift_size: float,
) -> List[float]:
    raw_times = []
    raw_rates = []
    for q in quotes:
        tenor = _parse_tenor_to_years(str(q.key).split("/")[-1])
        if tenor is None or tenor <= 0.0:
            continue
        raw_times.append(float(tenor))
        raw_rates.append(float(q.value))
    if not raw_times:
        return [float(q.value) for q in quotes]

    raw_times_arr = np.asarray(raw_times, dtype=float)
    raw_rates_arr = np.asarray(raw_rates, dtype=float)
    coarse = np.asarray(sorted(float(x) for x in coarse_node_times if float(x) > 0.0), dtype=float)
    if coarse.size == 0:
        return [float(q.value) for q in quotes]
    base_surv = _survival_from_piecewise_hazard(coarse, raw_times_arr, raw_rates_arr)
    target_idx = int(np.argmin(np.abs(coarse - float(target_time))))
    desired = np.array(base_surv, copy=True)
    eps = 1.0e-10
    desired[target_idx] = min(max(desired[target_idx] + float(shift_size), eps), 1.0 - eps)
    for i in range(1, desired.size):
        desired[i] = min(desired[i], desired[i - 1] - eps)
        desired[i] = max(desired[i], eps)

    coarse_lambdas = np.empty_like(coarse)
    prev_surv = 1.0
    prev_t = 0.0
    for i, t in enumerate(coarse):
        coarse_lambdas[i] = -(math.log(desired[i]) - math.log(prev_surv)) / max(float(t) - prev_t, 1.0e-12)
        prev_surv = float(desired[i])
        prev_t = float(t)

    rebuilt: List[float] = []
    for q in quotes:
        tenor = _parse_tenor_to_years(str(q.key).split("/")[-1])
        if tenor is None or tenor <= 0.0:
            rebuilt.append(float(q.value))
            continue
        idx = int(np.searchsorted(coarse, float(tenor), side="left"))
        if idx >= coarse_lambdas.size:
            rebuilt.append(float(coarse_lambdas[-1]))
        else:
            rebuilt.append(float(coarse_lambdas[idx]))
    return rebuilt


def _build_credit_survival_curve_shock(
    quotes: Sequence[MarketQuote],
    coarse_node_times: Sequence[float],
    target_time: float,
    shift_size: float,
    extrapolation: str = "flat_zero",
) -> Dict[str, object]:
    raw_times = []
    raw_rates = []
    for q in quotes:
        tenor = _parse_tenor_to_years(str(q.key).split("/")[-1])
        if tenor is None or tenor <= 0.0:
            continue
        raw_times.append(float(tenor))
        raw_rates.append(float(q.value))
    coarse = np.asarray(sorted(float(x) for x in coarse_node_times if float(x) > 0.0), dtype=float)
    if not raw_times or coarse.size == 0:
        return {"node_times": list(coarse_node_times), "survival_probabilities": [], "extrapolation": extrapolation}

    raw_times_arr = np.asarray(raw_times, dtype=float)
    raw_rates_arr = np.asarray(raw_rates, dtype=float)
    base_surv = _survival_from_piecewise_hazard(coarse, raw_times_arr, raw_rates_arr)
    avg_haz = -np.log(np.clip(base_surv, 1.0e-18, 1.0)) / coarse
    target_idx = int(np.argmin(np.abs(coarse - float(target_time))))
    shifted_avg_haz = np.array(avg_haz, copy=True)
    for i, t in enumerate(coarse):
        shifted_avg_haz[i] = avg_haz[i] + shift_size * _ore_bucket_weight(coarse, target_idx, float(t))
    shifted_surv = np.exp(-shifted_avg_haz * coarse)
    return {
        "node_times": [float(x) for x in coarse],
        "survival_probabilities": [float(x) for x in shifted_surv],
        "extrapolation": str(extrapolation).strip().lower(),
    }
