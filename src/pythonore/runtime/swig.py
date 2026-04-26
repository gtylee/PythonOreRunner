from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from pythonore.mapping.mapper import MappedInputs, build_input_parameters
from pythonore.repo_paths import find_engine_repo_root
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.lgm.market import _active_dim_mode, _configured_output_dir, _read_csv_report
from pythonore.runtime.results import CubeAccessor, XVAResult, xva_total_from_metrics
from pythonore.domain.dataclasses import XVASnapshot


class ORESwigAdapter:
    """Optional adapter using ORE-SWIG when available in the environment."""

    def __init__(self, module: Optional[Any] = None, process_isolation: bool = True):
        self._module = module or self._discover_module()
        self._process_isolation = bool(process_isolation)
        self._input_parameters_cls = self._resolve_attr("InputParameters")
        self._ore_app_cls = self._resolve_attr("OREApp")

    def run(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        if self._process_isolation and not _IN_CHILD_PROCESS:
            return self._run_via_subprocess(snapshot, mapped, run_id)
        return self._run_direct(snapshot, mapped, run_id)

    def _run_direct(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        ip = self._construct_input_parameters()
        build_input_parameters(snapshot, ip)

        app = self._construct_app(ip)
        self._invoke_run(app, mapped.market_data_lines, mapped.fixing_data_lines)

        reports = self._extract_reports(app)
        cubes = self._extract_cubes(app)
        dim_reports, dim_report_source = self._extract_dim_reports(snapshot, reports)
        reports.update(dim_reports)
        xva_by_metric = self._extract_xva_metrics(reports)
        exposure_by_netting_set = self._extract_exposures(reports)
        xva_total = xva_total_from_metrics(xva_by_metric)
        pv_total = self._extract_pv_total(reports)

        return XVAResult(
            run_id=run_id,
            pv_total=pv_total,
            xva_total=xva_total,
            xva_by_metric=xva_by_metric,
            exposure_by_netting_set=exposure_by_netting_set,
            reports=reports,
            cubes=cubes,
            metadata={
                "adapter": "ore-swig",
                "module": getattr(self._module, "__name__", type(self._module).__name__),
                "dim_mode": _active_dim_mode(snapshot),
                "dim_report_source": dim_report_source,
            },
        )

    def _run_via_subprocess(self, snapshot: XVASnapshot, mapped: MappedInputs, run_id: str) -> XVAResult:
        with tempfile.TemporaryDirectory(prefix="pythonore_ore_swig_") as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "input.pkl"
            output_path = tmp / "output.pkl"
            input_path.write_bytes(
                pickle.dumps(
                    {"snapshot": snapshot, "mapped": mapped, "run_id": run_id},
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            )
            env = os.environ.copy()
            src_root = Path(__file__).resolve().parents[2]
            pythonpath_parts = [str(src_root), env.get("PYTHONPATH", "")]
            env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
            cmd = [
                sys.executable,
                "-m",
                "pythonore.runtime.swig",
                "--swig-child",
                str(input_path),
                str(output_path),
            ]
            proc = subprocess.run(cmd, cwd=src_root.parent, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                raise EngineRunError(
                    "ORE-SWIG subprocess run failed "
                    f"(exit={proc.returncode}).\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            try:
                return pickle.loads(output_path.read_bytes())
            except Exception as exc:
                raise EngineRunError(f"Failed to load ORE-SWIG subprocess result: {exc}") from exc

    def _discover_module(self) -> Any:
        self._prime_swig_module_search_path()
        errors = []
        for mod_name in ("ORE", "oreanalytics", "OREAnalytics"):
            try:
                module = importlib.import_module(mod_name)
                if self._contains_attr(module, "InputParameters") and self._contains_attr(module, "OREApp"):
                    return module
            except Exception as exc:
                errors.append(f"{mod_name}: {exc}")
        raise EngineRunError(
            "Could not load ORE-SWIG module with InputParameters/OREApp. "
            + "; ".join(errors)
        )

    def _prime_swig_module_search_path(self) -> None:
        candidates: list[Path] = []
        env_root = os.getenv("ORE_SWIG_ROOT", "").strip()
        if env_root:
            root = Path(env_root).expanduser().resolve()
            candidates.append(root)
            candidates.extend(sorted(root.glob("build/lib.*")))
        engine_root = find_engine_repo_root()
        if engine_root is not None:
            swig_root = engine_root / "ORE-SWIG"
            candidates.append(swig_root)
            candidates.extend(sorted(swig_root.glob("build/lib.*")))
        for candidate in candidates:
            if candidate.exists():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)

    def _contains_attr(self, obj: Any, name: str) -> bool:
        if hasattr(obj, name):
            return True
        nested = getattr(obj, "_ORE", None)
        return nested is not None and hasattr(nested, name)

    def _resolve_attr(self, name: str) -> Any:
        if hasattr(self._module, name):
            return getattr(self._module, name)
        nested = getattr(self._module, "_ORE", None)
        if nested is not None and hasattr(nested, name):
            return getattr(nested, name)
        raise EngineRunError(f"Missing required ORE-SWIG class: {name}")

    def _construct_input_parameters(self) -> Any:
        try:
            return self._input_parameters_cls()
        except Exception as exc:
            raise EngineRunError(f"Failed to construct InputParameters: {exc}") from exc

    def _construct_app(self, input_parameters: Any) -> Any:
        _ORE6_ARGS = ("", 31, False, True)
        candidates = [
            (input_parameters, *_ORE6_ARGS),
            (input_parameters,),
        ]
        for args in candidates:
            try:
                return self._ore_app_cls(*args)
            except Exception:
                continue
        raise EngineRunError("Failed to construct OREApp from InputParameters")

    def _invoke_run(self, app: Any, market_data: Sequence[str], fixing_data: Sequence[str]) -> None:
        try:
            sig = inspect.signature(app.run)
            sig.bind(list(market_data), list(fixing_data))
            accepts_args = True
        except TypeError:
            accepts_args = False

        if accepts_args:
            try:
                app.run(list(market_data), list(fixing_data))
                return
            except Exception as exc:
                raise EngineRunError(f"OREApp.run(marketData, fixingData) failed: {exc}") from exc

        try:
            app.run()
            return
        except Exception as exc:
            raise EngineRunError(f"OREApp.run() failed: {exc}") from exc

    def _extract_reports(self, app: Any) -> Dict[str, Any]:
        reports: Dict[str, Any] = {}
        if not hasattr(app, "getReportNames") or not hasattr(app, "getReport"):
            return reports
        try:
            for name in app.getReportNames():
                reports[str(name)] = self._normalize_transport_value(app.getReport(str(name)))
        except Exception:
            return reports
        return reports

    def _extract_cubes(self, app: Any) -> Dict[str, CubeAccessor]:
        cubes: Dict[str, CubeAccessor] = {}
        if hasattr(app, "getCubeNames") and hasattr(app, "getCube"):
            try:
                for name in app.getCubeNames():
                    cubes[str(name)] = CubeAccessor(
                        name=str(name),
                        payload={"raw": self._normalize_transport_value(app.getCube(str(name)))},
                    )
            except Exception:
                pass
        if hasattr(app, "getMarketCubeNames") and hasattr(app, "getMarketCube"):
            try:
                for name in app.getMarketCubeNames():
                    key = f"market::{name}"
                    cubes[key] = CubeAccessor(
                        name=key,
                        payload={"raw": self._normalize_transport_value(app.getMarketCube(str(name)))},
                    )
            except Exception:
                pass
        return cubes

    def _extract_dim_reports(self, snapshot: XVASnapshot, reports: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        dim_reports: Dict[str, Any] = {}
        report_source = "absent"

        report_aliases = {
            "dim_evolution": ("dim_evolution", "dimevolution", "dimEvolution"),
            "dim_regression": ("dim_regression", "dimregression", "dimRegression"),
        }
        for target, aliases in report_aliases.items():
            for alias in aliases:
                rows = self._rows_from_report(reports.get(alias))
                if rows:
                    dim_reports[target] = rows
                    report_source = "report"
                    break

        if dim_reports:
            return dim_reports, report_source

        file_reports = self._read_dim_reports_from_files(snapshot)
        if file_reports:
            return file_reports, "file"

        return {}, report_source

    def _read_dim_reports_from_files(self, snapshot: XVASnapshot) -> Dict[str, Any]:
        runtime = snapshot.config.runtime
        if runtime is None:
            return {}

        xva = runtime.xva_analytic
        configured = {
            "dim_evolution": xva.dim_evolution_file,
            "dim_regression": xva.dim_regression_files,
        }
        output_dir = _configured_output_dir(snapshot)
        if output_dir is None:
            return {}

        out: Dict[str, Any] = {}
        for key, rel_name in configured.items():
            if not rel_name:
                continue
            path = Path(rel_name)
            if not path.is_absolute():
                path = output_dir / path
            rows = _read_csv_report(path)
            if rows:
                out[key] = rows
        return out

    def _extract_xva_metrics(self, reports: Dict[str, Any]) -> Dict[str, float]:
        xva: Dict[str, float] = {}
        rows = self._rows_from_report(reports.get("xva"))
        if not rows:
            return xva

        for row in rows:
            metric = row.get("Metric") or row.get("metric") or row.get("Type")
            value = row.get("Value") or row.get("value")
            if metric in ("CVA", "DVA", "FVA", "MVA") and value is not None:
                xva[str(metric)] = float(value)
        if xva:
            return xva

        metric_cols = ("CVA", "DVA", "MVA", "FBA", "FCA")
        aggregates = {m: 0.0 for m in metric_cols}
        has_metric = {m: False for m in metric_cols}
        for row in rows:
            trade_id = str(row.get("TradeId", "")).strip()
            include = trade_id == ""
            for m in metric_cols:
                if m in row and row[m] is not None:
                    try:
                        v = float(row[m])
                    except Exception:
                        continue
                    if include:
                        aggregates[m] += v
                        has_metric[m] = True
        if not any(has_metric.values()):
            for row in rows:
                for m in metric_cols:
                    if m in row and row[m] is not None:
                        try:
                            aggregates[m] += float(row[m])
                            has_metric[m] = True
                        except Exception:
                            pass

        if has_metric["CVA"]:
            xva["CVA"] = aggregates["CVA"]
        if has_metric["DVA"]:
            xva["DVA"] = aggregates["DVA"]
        if has_metric["MVA"]:
            xva["MVA"] = aggregates["MVA"]
        if has_metric["FBA"] or has_metric["FCA"]:
            if has_metric["FBA"]:
                xva["FBA"] = aggregates["FBA"]
            if has_metric["FCA"]:
                xva["FCA"] = aggregates["FCA"]
            xva["FVA"] = aggregates["FBA"] + aggregates["FCA"]
        return xva

    def _extract_exposures(self, reports: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        exposure_reports = [name for name in reports if str(name).lower().startswith("exposure")]
        if "exposure" not in exposure_reports and "exposure" in reports:
            exposure_reports.append("exposure")

        for report_name in exposure_reports:
            for row in self._rows_from_report(reports.get(report_name)):
                ns = row.get("NettingSetId") or row.get("nettingsetid")
                epe = self._first_present(row, ("EPE", "BaselEEPE", "ExpectedPositiveExposure", "expectedpositiveexposure"))
                if ns is not None and epe is not None:
                    out[str(ns)] = out.get(str(ns), 0.0) + epe
        if out:
            return out

        rows = self._rows_from_report(reports.get("xva"))
        for row in rows:
            ns = row.get("NettingSetId")
            trade_id = str(row.get("TradeId", "")).strip()
            epe = row.get("BaselEEPE")
            if ns is not None and epe is not None and trade_id == "":
                v = self._to_float(epe)
                if v is not None:
                    out[str(ns)] = v
        return out

    def _extract_pv_total(self, reports: Dict[str, Any]) -> float:
        rows = self._rows_from_report(reports.get("npv"))
        total = 0.0
        for row in rows:
            value = self._first_present(row, ("NPV(Base)", "NPV", "PresentValue", "presentvalue"))
            if value is not None:
                total += value
        return total

    def _rows_from_report(self, report: Any) -> Sequence[Dict[str, Any]]:
        if report is None:
            return []
        if isinstance(report, list) and (not report or isinstance(report[0], dict)):
            return report
        if hasattr(report, "columns") and hasattr(report, "data"):
            cols = list(report.columns())
            data = report.data()
            out = []
            for row in data:
                out.append({str(c): row[i] for i, c in enumerate(cols) if i < len(row)})
            return out
        if hasattr(report, "rows") and hasattr(report, "columns") and hasattr(report, "header"):
            rows = int(report.rows())
            cols = int(report.columns())
            headers = [str(report.header(c)) for c in range(cols)]
            out = []
            for r in range(rows):
                row: Dict[str, Any] = {}
                for c, h in enumerate(headers):
                    v = None
                    for fn_name in ("dataAsString", "dataAsReal", "dataAsSize", "dataAsDate", "dataAsPeriod"):
                        fn = getattr(report, fn_name, None)
                        if not callable(fn):
                            continue
                        try:
                            candidate = fn(r, c)
                            v = candidate
                            break
                        except Exception:
                            continue
                    row[h] = v
                out.append(row)
            return out
        return []

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _first_present(self, row: Dict[str, Any], keys: Sequence[str]) -> float | None:
        for k in keys:
            if k in row and row[k] is not None:
                v = self._to_float(row[k])
                if v is not None:
                    return v
        return None

    def _normalize_transport_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._normalize_transport_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize_transport_value(v) for v in value]
        if self._is_pickleable(value):
            return value
        rows = self._rows_from_report(value)
        if rows:
            return [self._normalize_transport_value(row) for row in rows]
        return str(value)

    def _is_pickleable(self, value: Any) -> bool:
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception:
            return False


def _swig_child_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--swig-child", nargs=2, metavar=("INPUT", "OUTPUT"))
    parsed, _ = parser.parse_known_args(list(argv))
    if parsed.swig_child is None:
        return 0
    input_path = Path(parsed.swig_child[0])
    output_path = Path(parsed.swig_child[1])
    payload = pickle.loads(input_path.read_bytes())
    adapter = ORESwigAdapter(process_isolation=False)
    result = adapter.run(payload["snapshot"], payload["mapped"], str(payload["run_id"]))
    output_path.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return 0


_IN_CHILD_PROCESS = False


__all__ = ["ORESwigAdapter"]


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    _IN_CHILD_PROCESS = True
    raise SystemExit(_swig_child_main(sys.argv[1:]))
