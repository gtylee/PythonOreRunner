from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from .curve_trace import (
    list_curve_handles_from_todaysmarket,
    trace_curve_handle_from_ore,
    trace_discount_curve_from_ore,
    trace_index_curve_from_ore,
)
from .interpolation import build_log_linear_discount_interpolator


@dataclass(frozen=True)
class CurveBuildRequest:
    ore_xml_path: str
    engine: str = "swig"
    configuration_id: str | None = None
    selected_curve_handles: tuple[str, ...] = ()
    currencies: tuple[str, ...] = ()
    index_names: tuple[str, ...] = ()
    persist_case: bool = False
    output_root: str | None = None


@dataclass(frozen=True)
class CurveTrace:
    curve_handle: str
    curve_name: str
    curve_id: str
    family: str
    status: str
    source_engine: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class CurveBuildResult:
    request: CurveBuildRequest
    source_engine: str
    run_root: str
    ore_xml_path: str
    traces: tuple[CurveTrace, ...]
    unsupported_sections: dict[str, list[str]]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    reports: tuple[str, ...]
    runtime_seconds: float | None
    artifact_snapshot: dict[str, Any]


@dataclass(frozen=True)
class CurveComparisonPoint:
    time: float
    ore_value: float
    engine_value: float
    abs_error: float
    rel_error: float


@dataclass(frozen=True)
class CurveComparison:
    curve_handle: str
    curve_id: str
    engine: str
    status: str
    max_abs_error: float | None
    max_rel_error: float | None
    points: tuple[CurveComparisonPoint, ...]
    tolerances: dict[str, float]
    message: str = ""


def build_curves_from_ore_inputs(request: CurveBuildRequest) -> CurveBuildResult:
    engine = request.engine.strip().lower()
    if engine != "swig":
        raise ValueError(f"Unsupported build engine '{request.engine}': use 'swig' for ORE-native builds")

    if not swig_module_available():
        raise RuntimeError("ORE-SWIG is not available in this environment")

    with _prepared_case(request) as case:
        runtime_seconds, reports, errors = _run_ore_case(case.ore_xml_path)
        section_handles = list_curve_handles_from_todaysmarket(
            case.ore_xml_path,
            configuration_id=request.configuration_id,
        )
        selected_yield_handles = _select_yield_handles(section_handles, request, case.ore_xml_path)
        traces = tuple(
            CurveTrace(
                curve_handle=payload["curve_handle"],
                curve_name=payload["curve_name"],
                curve_id=payload["curve_config"]["curve_id"],
                family="yield_curve",
                status="ok",
                source_engine="swig",
                payload=payload,
            )
            for payload in (
                trace_curve_handle_from_ore(case.ore_xml_path, curve_handle)
                for curve_handle in selected_yield_handles
            )
        )

        unsupported_sections = {
            key: value
            for key, value in section_handles.items()
            if key not in ("yield_curves", "discounting_curves", "index_forwarding_curves") and value
        }
        warnings = []
        for section_name, values in unsupported_sections.items():
            warnings.append(f"{section_name} not_implemented: {len(values)} object(s)")
        warnings.extend(errors)

        artifact_snapshot = {
            "input": {
                "ore_xml": case.ore_xml_path,
                "configuration_id": request.configuration_id,
            },
            "output": {
                "curves_csv": str(Path(case.run_root) / "Output" / "curves.csv"),
                "todaysmarketcalibration_csv": str(Path(case.run_root) / "Output" / "todaysmarketcalibration.csv"),
                "marketdata_csv": str(Path(case.run_root) / "Output" / "marketdata.csv"),
            },
            "reports": list(reports),
            "errors": list(errors),
            "selected_curve_handles": list(selected_yield_handles),
        }

        return CurveBuildResult(
            request=request,
            source_engine="swig",
            run_root=case.run_root,
            ore_xml_path=case.ore_xml_path,
            traces=traces,
            unsupported_sections=unsupported_sections,
            warnings=tuple(warnings),
            errors=tuple(errors),
            reports=tuple(reports),
            runtime_seconds=runtime_seconds,
            artifact_snapshot=artifact_snapshot,
        )


def trace_curve(
    ore_xml_path: str | Path,
    *,
    curve_handle: str | None = None,
    currency: str | None = None,
    index_name: str | None = None,
) -> CurveTrace:
    provided = [curve_handle is not None, currency is not None, index_name is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of curve_handle, currency, or index_name")

    if curve_handle is not None:
        payload = trace_curve_handle_from_ore(ore_xml_path, curve_handle)
    elif currency is not None:
        payload = trace_discount_curve_from_ore(ore_xml_path, currency=currency)
    else:
        payload = trace_index_curve_from_ore(ore_xml_path, index_name=str(index_name))

    return CurveTrace(
        curve_handle=payload["curve_handle"],
        curve_name=payload["curve_name"],
        curve_id=payload["curve_config"]["curve_id"],
        family="yield_curve",
        status="ok",
        source_engine="swig",
        payload=payload,
    )


def compare_python_vs_ore(
    ore_xml_path: str | Path,
    *,
    curve_handle: str | None = None,
    currency: str | None = None,
    index_name: str | None = None,
    engine: str = "python",
    abs_tolerance: float = 1.0e-8,
    rel_tolerance: float = 1.0e-8,
) -> CurveComparison:
    selected_engine = engine.strip().lower()
    if selected_engine != "python":
        raise ValueError(f"Unsupported comparison engine '{engine}': use 'python'")

    curve = trace_curve(
        ore_xml_path,
        curve_handle=curve_handle,
        currency=currency,
        index_name=index_name,
    )
    payload = curve.payload
    curve_config = payload["curve_config"]
    nodes = payload.get("native_curve_nodes", {})
    points = payload["ore_curve_points"]
    supported = (
        curve.family == "yield_curve"
        and curve_config.get("interpolation_variable") == "Discount"
        and curve_config.get("interpolation_method") == "LogLinear"
        and len(nodes.get("times", [])) >= 2
    )

    if not supported:
        return CurveComparison(
            curve_handle=curve.curve_handle,
            curve_id=curve.curve_id,
            engine=selected_engine,
            status="not_implemented",
            max_abs_error=None,
            max_rel_error=None,
            points=(),
            tolerances={"abs": abs_tolerance, "rel": rel_tolerance},
            message="Python comparator currently supports only yield curves with Discount/LogLinear interpolation",
        )

    interpolator = build_log_linear_discount_interpolator(
        list(nodes["times"]),
        list(nodes["discount_factors"]),
    )
    comparison_points = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    for time_value, ore_value in zip(points["times"], points["dfs"]):
        engine_value = float(interpolator(float(time_value)))
        abs_error = abs(engine_value - float(ore_value))
        rel_error = abs_error / max(abs(float(ore_value)), 1.0e-16)
        comparison_points.append(
            CurveComparisonPoint(
                time=float(time_value),
                ore_value=float(ore_value),
                engine_value=engine_value,
                abs_error=abs_error,
                rel_error=rel_error,
            )
        )
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)

    status = "ok"
    if max_abs_error > abs_tolerance or max_rel_error > rel_tolerance:
        status = "tolerance_breach"

    return CurveComparison(
        curve_handle=curve.curve_handle,
        curve_id=curve.curve_id,
        engine=selected_engine,
        status=status,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        points=tuple(comparison_points),
        tolerances={"abs": abs_tolerance, "rel": rel_tolerance},
    )


def swig_module_available() -> bool:
    try:
        _load_ore_module()
        return True
    except Exception:
        return False


def result_to_json(result: CurveBuildResult | CurveComparison | CurveTrace) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True)


@dataclass(frozen=True)
class _PreparedCase:
    run_root: str
    ore_xml_path: str


@contextmanager
def _prepared_case(request: CurveBuildRequest) -> Iterable[_PreparedCase]:
    source_ore_xml = Path(request.ore_xml_path).resolve()
    source_run_root = _default_run_dir(source_ore_xml)
    if request.persist_case:
        if request.output_root:
            base_root = Path(request.output_root).resolve()
            base_root.mkdir(parents=True, exist_ok=True)
            case_root = base_root / source_run_root.name
            if case_root.exists():
                shutil.rmtree(case_root)
            shutil.copytree(source_run_root, case_root)
        else:
            case_root = source_run_root
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="ore_curve_fit_parity_"))
        case_root = temp_root / source_run_root.name
        shutil.copytree(source_run_root, case_root)

    cloned_ore_xml = case_root / source_ore_xml.relative_to(source_run_root)
    _rewrite_case_paths(source_run_root, case_root, cloned_ore_xml)

    yield _PreparedCase(
        run_root=str(case_root),
        ore_xml_path=str(cloned_ore_xml),
    )


def _rewrite_case_paths(source_run_root: Path, cloned_run_root: Path, ore_xml_path: Path) -> None:
    tree = ET.parse(ore_xml_path)
    root = tree.getroot()
    cloned_input = cloned_run_root / "Input"
    cloned_output = cloned_run_root / "Output"

    def _remap(value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            try:
                rel = path.relative_to(source_run_root)
                return str(cloned_run_root / rel)
            except ValueError:
                return value
        return value

    for node in root.findall("./Setup/Parameter"):
        name = node.attrib.get("name", "")
        value = (node.text or "").strip()
        if not value:
            continue
        if name == "inputPath":
            node.text = str(cloned_input)
        elif name == "outputPath":
            node.text = str(cloned_output)
        elif name.endswith("File") or name == "calendarAdjustment":
            node.text = _remap(value)

    for analytic in root.findall("./Analytics/Analytic"):
        for node in analytic.findall("./Parameter"):
            name = node.attrib.get("name", "")
            value = (node.text or "").strip()
            if not value:
                continue
            if name.endswith("File") or name.endswith("FileName"):
                path = Path(value)
                if not path.is_absolute():
                    continue
                try:
                    rel = path.relative_to(source_run_root)
                    node.text = str(cloned_run_root / rel)
                except ValueError:
                    pass

    tree.write(ore_xml_path, encoding="utf-8", xml_declaration=True)


def _run_ore_case(ore_xml_path: str) -> tuple[float | None, tuple[str, ...], tuple[str, ...]]:
    ore_xml = Path(ore_xml_path).resolve()
    ore_module = _load_ore_module()
    started = time.perf_counter()
    with _pushd(_default_run_dir(ore_xml)):
        params = ore_module.Parameters()
        params.fromFile(str(ore_xml))
        app = _construct_ore_app(ore_module, params)
        app.run()
        runtime_seconds = _extract_runtime_seconds(app, started)
        reports = tuple(sorted(_safe_iterable_call(app, "getReportNames")))
        errors = tuple(str(x) for x in _safe_iterable_call(app, "getErrors"))
    return runtime_seconds, reports, errors


def _extract_runtime_seconds(app: Any, started: float) -> float:
    runtime = getattr(app, "getRunTime", None)
    if callable(runtime):
        try:
            return float(runtime())
        except Exception:
            pass
    return time.perf_counter() - started


def _safe_iterable_call(obj: Any, method_name: str) -> list[Any]:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return []
    try:
        return list(method())
    except Exception:
        return []


def _construct_ore_app(ore_module: Any, params: Any) -> Any:
    candidates = [
        (params, False),
        (params, "", 31, False, True),
        (params,),
    ]
    last_error: Exception | None = None
    for args in candidates:
        try:
            return ore_module.OREApp(*args)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to construct OREApp: {last_error}")


def _select_yield_handles(
    section_handles: dict[str, list[str]],
    request: CurveBuildRequest,
    ore_xml_path: str,
) -> tuple[str, ...]:
    handles: list[str] = []
    if request.selected_curve_handles:
        handles.extend(request.selected_curve_handles)

    for currency in request.currencies:
        discount_trace = trace_discount_curve_from_ore(ore_xml_path, currency=currency)
        handles.append(str(discount_trace["curve_handle"]))

    for index_name in request.index_names:
        index_trace = trace_index_curve_from_ore(ore_xml_path, index_name=index_name)
        handles.append(str(index_trace["curve_handle"]))

    if not handles:
        handles.extend(section_handles.get("yield_curves", []))
        handles.extend(section_handles.get("discounting_curves", []))
        handles.extend(section_handles.get("index_forwarding_curves", []))

    ordered: list[str] = []
    seen: set[str] = set()
    for handle in handles:
        if not handle.startswith("Yield/"):
            continue
        if handle in seen:
            continue
        seen.add(handle)
        ordered.append(handle)
    return tuple(ordered)


def _default_run_dir(ore_xml: Path) -> Path:
    if ore_xml.parent.name.lower() == "input":
        return ore_xml.parent.parent
    return ore_xml.parent


@contextmanager
def _pushd(path: Path) -> Iterable[None]:
    previous = Path.cwd()
    os_path = str(path)
    try:
        Path(os_path)
        import os

        os.chdir(os_path)
        yield
    finally:
        import os

        os.chdir(str(previous))


def _load_ore_module() -> Any:
    candidate_paths = []
    repo_root = Path(__file__).resolve().parents[3]
    swig_root = repo_root / "ORE-SWIG"
    if swig_root.exists():
        direct_module = swig_root
        if str(direct_module) not in sys.path:
            candidate_paths.append(direct_module)
        build_root = swig_root / "build"
        if build_root.exists():
            for child in sorted(build_root.glob("lib.*")):
                candidate_paths.append(child)

    for path in candidate_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    errors = []
    for module_name in ("ORE", "oreanalytics", "OREAnalytics"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "OREApp") and hasattr(module, "Parameters"):
                return module
            nested = getattr(module, "_ORE", None)
            if nested is not None and hasattr(nested, "OREApp") and hasattr(nested, "Parameters"):
                return nested
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError("Could not import a usable ORE-SWIG module. " + "; ".join(errors))
