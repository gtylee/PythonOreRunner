from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict


def _setup_params(ore_xml: Path) -> dict[str, str]:
    root = ET.parse(ore_xml).getroot()
    return {
        node.attrib.get("name", ""): (node.text or "").strip()
        for node in root.findall("./Setup/Parameter")
    }


def _simulation_config_id(ore_xml: Path) -> str:
    root = ET.parse(ore_xml).getroot()
    node = root.find("./Markets/Parameter[@name='simulation']")
    if node is None or not (node.text or "").strip():
        return "default"
    return (node.text or "").strip()


def _resolve_curveconfig_path(ore_xml: Path) -> Path:
    setup = _setup_params(ore_xml)
    curveconfig = setup.get("curveConfigFile", "")
    if not curveconfig:
        raise ValueError(f"Missing Setup/curveConfigFile in {ore_xml}")
    return (ore_xml.parent / curveconfig).resolve()


def _resolve_conventions_path(ore_xml: Path) -> Path:
    setup = _setup_params(ore_xml)
    conventions = setup.get("conventionsFile", "")
    if not conventions:
        raise ValueError(f"Missing Setup/conventionsFile in {ore_xml}")
    return (ore_xml.parent / conventions).resolve()


def _normalize_date_input(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _normalize_day_counter_name(day_counter: str) -> str:
    dc = str(day_counter).strip().upper().replace(" ", "")
    aliases = {
        "A365": "A365F",
        "ACTUAL/365(FIXED)": "A365F",
        "ACTUAL365FIXED": "A365F",
        "ACT/365(FIXED)": "A365F",
        "A365F": "A365F",
    }
    return aliases.get(dc, day_counter)


def _year_fraction_from_day_counter(start: str | date, end: str | date, day_counter: str) -> float:
    start_date = _normalize_date_input(start)
    end_date = _normalize_date_input(end)
    dc = _normalize_day_counter_name(day_counter)
    if dc != "A365F":
        raise ValueError(f"Unsupported day counter '{day_counter}'")
    return (end_date - start_date).days / 365.0


def _load_ore_discount_pairs_by_columns_with_day_counter(
    curves_csv: str,
    discount_columns: list[str],
    *,
    asof_date: str,
    day_counter: str,
) -> Dict[str, tuple[tuple[str, ...], list[float], list[float]]]:
    columns = [column for column in discount_columns if column]
    if not columns:
        raise ValueError("discount_columns must contain at least one column name")

    requested = list(dict.fromkeys(columns))
    rows: list[dict[str, str]] = []
    with open(curves_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("curves.csv appears empty")
    if "Date" not in rows[0]:
        raise ValueError("curves.csv missing Date column")
    missing = [column for column in requested if column not in rows[0]]
    if missing:
        raise ValueError(f"curves.csv missing requested discount columns: {missing}")

    base_dates: list[str] = []
    times: list[float] = []
    by_col: Dict[str, list[float]] = {column: [] for column in requested}
    for row in rows:
        row_date = str(row["Date"])
        base_dates.append(row_date)
        times.append(_year_fraction_from_day_counter(asof_date, row_date, day_counter))
        for column in requested:
            by_col[column].append(float(row[column]))

    unique_index: list[int] = []
    seen: set[float] = set()
    for idx, time in enumerate(times):
        if time in seen:
            continue
        seen.add(time)
        unique_index.append(idx)

    unique_dates = tuple(base_dates[idx] for idx in unique_index)
    unique_times = [times[idx] for idx in unique_index]
    out: Dict[str, tuple[tuple[str, ...], list[float], list[float]]] = {}
    for column in requested:
        unique_dfs = [by_col[column][idx] for idx in unique_index]
        dates = unique_dates
        curve_times = unique_times[:]
        dfs = unique_dfs[:]
        if curve_times[0] > 1.0e-12:
            curve_times.insert(0, 0.0)
            dfs.insert(0, 1.0)
            dates = (asof_date,) + dates
        else:
            dfs[0] = 1.0
            dates = (asof_date,) + dates[1:]
        out[column] = (dates, curve_times, dfs)
    return out


def _resolve_ore_run_files(ore_xml_path: str | Path) -> tuple[Path, str, Path, Path, Path]:
    ore_xml = Path(ore_xml_path).resolve()
    setup = _setup_params(ore_xml)
    asof_date = setup.get("asofDate", "")
    if not asof_date:
        raise ValueError(f"Missing Setup/asofDate in {ore_xml}")
    base = ore_xml.parent
    run_dir = base.parent
    output_path = Path(setup.get("outputPath", "Output"))
    if not output_path.is_absolute():
        output_path = (run_dir / output_path).resolve()
    market_config = Path(setup.get("marketConfigFile", "../../Input/todaysmarket.xml"))
    if not market_config.is_absolute():
        market_config = (base / market_config).resolve()
    market_data = Path(setup.get("marketDataFile", "../../Input/market_20160205_flat.txt"))
    if not market_data.is_absolute():
        market_data = (base / market_data).resolve()
    return ore_xml, asof_date, market_data, market_config, output_path


def _parse_market_quotes(market_data_file: Path) -> dict[str, float]:
    quotes: dict[str, float] = {}
    with market_data_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < 3:
                continue
            try:
                quotes[tokens[1]] = float(tokens[2])
            except ValueError:
                continue
    return quotes


def _load_todaysmarket_root(todaysmarket_xml: Path) -> ET.Element:
    return ET.parse(todaysmarket_xml).getroot()


def _load_configuration_node(todaysmarket_root: ET.Element, configuration_id: str) -> ET.Element:
    config = todaysmarket_root.find(f"./Configuration[@id='{configuration_id}']")
    if config is None:
        raise ValueError(f"Configuration '{configuration_id}' not found")
    return config


def _find_discount_curve_handle(todaysmarket_xml: Path, configuration_id: str, currency: str) -> str:
    root = _load_todaysmarket_root(todaysmarket_xml)
    config = _load_configuration_node(root, configuration_id)
    discounting_id = (config.findtext("./DiscountingCurvesId") or "").strip()
    if not discounting_id:
        raise ValueError(f"Configuration '{configuration_id}' has no DiscountingCurvesId")
    node = root.find(
        f"./DiscountingCurves[@id='{discounting_id}']/DiscountingCurve[@currency='{currency}']"
    )
    if node is None or not (node.text or "").strip():
        raise ValueError(
            f"Discounting curve for currency '{currency}' not found in DiscountingCurves '{discounting_id}'"
        )
    return (node.text or "").strip()


def _find_index_curve_handle(todaysmarket_xml: Path, configuration_id: str, index_name: str) -> str:
    root = _load_todaysmarket_root(todaysmarket_xml)
    config = _load_configuration_node(root, configuration_id)
    forwarding_id = (config.findtext("./IndexForwardingCurvesId") or "").strip()
    if not forwarding_id:
        raise ValueError(f"Configuration '{configuration_id}' has no IndexForwardingCurvesId")
    node = root.find(
        f"./IndexForwardingCurves[@id='{forwarding_id}']/Index[@name='{index_name}']"
    )
    if node is None or not (node.text or "").strip():
        raise ValueError(
            f"Forwarding curve for index '{index_name}' not found in IndexForwardingCurves '{forwarding_id}'"
        )
    return (node.text or "").strip()


def _find_curve_name(todaysmarket_xml: Path, curve_handle: str) -> str:
    root = _load_todaysmarket_root(todaysmarket_xml)
    for group in root.findall("./YieldCurves"):
        for curve in group.findall("./YieldCurve"):
            if (curve.text or "").strip() == curve_handle:
                name = curve.attrib.get("name", "").strip()
                if name:
                    return name
    for group in root.findall("./IndexForwardingCurves"):
        for index in group.findall("./Index"):
            if (index.text or "").strip() == curve_handle:
                name = index.attrib.get("name", "").strip()
                if name:
                    return name
    raise ValueError(f"Could not resolve curves.csv column for handle '{curve_handle}'")


def _curve_id_from_handle(curve_handle: str) -> str:
    parts = curve_handle.split("/")
    if len(parts) < 3:
        raise ValueError(f"Unexpected curve handle '{curve_handle}'")
    return parts[-1]


def _child_text(node: ET.Element, name: str, default: str = "") -> str:
    child = node.find(name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _element_to_dict(node: ET.Element) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if node.attrib:
        payload["attributes"] = dict(node.attrib)
    children = list(node)
    if not children:
        text = (node.text or "").strip()
        if text:
            payload["value"] = text
        return payload

    grouped: dict[str, list[Any]] = {}
    for child in children:
        grouped.setdefault(child.tag, []).append(_element_to_dict(child))

    for key, values in grouped.items():
        payload[key] = values if len(values) > 1 else values[0]
    text = (node.text or "").strip()
    if text:
        payload["text"] = text
    return payload


def _parse_conventions(conventions_xml: Path, convention_ids: list[str]) -> dict[str, dict[str, Any]]:
    requested = {convention_id for convention_id in convention_ids if convention_id}
    if not requested:
        return {}
    root = ET.parse(conventions_xml).getroot()
    out: dict[str, dict[str, Any]] = {}
    for child in root:
        convention_id = _child_text(child, "Id")
        if convention_id and convention_id in requested:
            out[convention_id] = {
                "convention_type": child.tag,
                "details": _element_to_dict(child),
            }
    missing = sorted(requested.difference(out))
    for convention_id in missing:
        out[convention_id] = {
            "convention_type": "Missing",
            "details": {},
        }
    return out


def _parse_segments(yield_curve: ET.Element, market_quotes: dict[str, float]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    segments_node = yield_curve.find("./Segments")
    if segments_node is None:
        return segments

    for segment in list(segments_node):
        quote_keys = [
            (quote.text or "").strip()
            for quote in segment.findall("./Quotes/Quote")
            if (quote.text or "").strip()
        ]
        segments.append(
            {
                "segment_tag": segment.tag,
                "type": _child_text(segment, "Type"),
                "conventions": _child_text(segment, "Conventions"),
                "pillar_choice": _child_text(segment, "PillarChoice"),
                "projection_curve": _child_text(segment, "ProjectionCurve"),
                "projection_curve_receive": _child_text(segment, "ProjectionCurveReceive"),
                "projection_curve_pay": _child_text(segment, "ProjectionCurvePay"),
                "quotes": [
                    {
                        "quote_key": key,
                        "quote_value": market_quotes.get(key),
                        "missing": key not in market_quotes,
                    }
                    for key in quote_keys
                ],
            }
        )
    return segments


def _parse_bootstrap_config(yield_curve: ET.Element) -> dict[str, Any]:
    node = yield_curve.find("./BootstrapConfig")
    if node is None:
        return {}
    payload: dict[str, Any] = {}
    for child in list(node):
        key = child.tag
        value = (child.text or "").strip()
        if not value:
            continue
        if value.lower() in ("true", "false"):
            payload[key] = value.lower() == "true"
            continue
        try:
            payload[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            payload[key] = float(value)
            continue
        except ValueError:
            pass
        payload[key] = value
    return payload


def _parse_yield_curve(curveconfig_xml: Path, curve_id: str, market_quotes: dict[str, float]) -> dict[str, Any]:
    root = ET.parse(curveconfig_xml).getroot()
    for yield_curve in root.findall("./YieldCurves/YieldCurve"):
        if _child_text(yield_curve, "CurveId") != curve_id:
            continue
        discount_curve = _child_text(yield_curve, "DiscountCurve", curve_id)
        return {
            "curve_id": curve_id,
            "currency": _child_text(yield_curve, "Currency"),
            "discount_curve": discount_curve,
            "interpolation_variable": _child_text(yield_curve, "InterpolationVariable", "Discount"),
            "interpolation_method": _child_text(yield_curve, "InterpolationMethod", "LogLinear"),
            "yield_curve_day_counter": _child_text(yield_curve, "YieldCurveDayCounter", "A365"),
            "pillar_choice": _child_text(yield_curve, "PillarChoice", "LastRelevantDate"),
            "extrapolation": _child_text(yield_curve, "Extrapolation"),
            "tolerance": _child_text(yield_curve, "Tolerance"),
            "bootstrap_config": _parse_bootstrap_config(yield_curve),
            "segments": _parse_segments(yield_curve, market_quotes),
        }
    raise ValueError(f"Yield curve '{curve_id}' not found in {curveconfig_xml}")


def _curve_dependency_ids(curve_config: dict[str, Any]) -> list[str]:
    deps: list[str] = []
    discount_curve = str(curve_config.get("discount_curve", "")).strip()
    curve_id = str(curve_config.get("curve_id", "")).strip()
    if discount_curve and discount_curve != curve_id:
        deps.append(discount_curve)
    for segment in curve_config.get("segments", []):
        for field in ("projection_curve", "projection_curve_receive", "projection_curve_pay"):
            dep = str(segment.get(field, "")).strip()
            if dep and dep != curve_id:
                deps.append(dep)
    deduped: list[str] = []
    seen: set[str] = set()
    for dep in deps:
        if dep in seen:
            continue
        seen.add(dep)
        deduped.append(dep)
    return deduped


def _load_calibration_rows(calibration_csv: Path, curve_id: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    day_counter = ""
    currency = ""

    with calibration_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("MarketObjectId", "") != curve_id:
                continue
            result_id = row.get("ResultId", "")
            if result_id == "dayCounter":
                day_counter = row.get("ResultValue", "")
                continue
            if result_id == "currency":
                currency = row.get("ResultValue", "")
                continue
            quote_key = row.get("ResultKey2", "")
            if not quote_key:
                continue
            entry = grouped.setdefault(
                quote_key,
                {
                    "date": row.get("ResultKey1", ""),
                    "quote_key": quote_key,
                },
            )
            entry[result_id] = row.get("ResultValue", "")

    ordered = [
        grouped[key]
        for key in sorted(grouped.keys(), key=lambda item: (float(grouped[item].get("time", "0") or 0.0), item))
    ]
    return {
        "day_counter": day_counter,
        "currency": currency,
        "pillars": ordered,
    }


def _native_curve_nodes(calibration: dict[str, Any]) -> dict[str, Any]:
    dates = []
    times = []
    discount_factors = []
    zero_rates = []
    forward_rates = []

    for pillar in calibration.get("pillars", []):
        time_text = pillar.get("time", "")
        df_text = pillar.get("discountFactor", "")
        if not time_text or not df_text:
            continue
        dates.append(str(pillar.get("date", "")))
        times.append(float(time_text))
        discount_factors.append(float(df_text))
        zero_rates.append(float(pillar["zeroRate"]) if pillar.get("zeroRate", "") else None)
        forward_rates.append(float(pillar["forwardRate"]) if pillar.get("forwardRate", "") else None)

    if times and abs(times[0]) > 1.0e-12:
        dates.insert(0, "")
        times.insert(0, 0.0)
        discount_factors.insert(0, 1.0)
        zero_rates.insert(0, 0.0)
        forward_rates.insert(0, None)
    elif discount_factors:
        discount_factors[0] = 1.0

    return {
        "calendar_dates": dates,
        "times": times,
        "discount_factors": discount_factors,
        "zero_rates": zero_rates,
        "forward_rates": forward_rates,
    }


def _align_segments_with_calibration(
    segments: list[dict[str, Any]],
    calibration_pillars: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_quote = {pillar["quote_key"]: pillar for pillar in calibration_pillars}
    aligned: list[dict[str, Any]] = []
    for segment in segments:
        enriched_quotes = []
        matched = 0
        for quote in segment["quotes"]:
            ore_pillar = by_quote.get(quote["quote_key"])
            if ore_pillar is not None:
                matched += 1
            enriched_quotes.append({**quote, "ore_pillar": ore_pillar})
        aligned.append(
            {
                "segment_tag": segment["segment_tag"],
                "type": segment["type"],
                "conventions": segment["conventions"],
                "quote_count": len(segment["quotes"]),
                "matched_calibration_rows": matched,
                "quotes": enriched_quotes,
            }
        )
    return aligned


def _curve_points(output_path: Path, asof_date: str, column_name: str, day_counter: str) -> dict[str, Any]:
    curves_csv = output_path / "curves.csv"
    dates, times, dfs = _load_ore_discount_pairs_by_columns_with_day_counter(
        str(curves_csv),
        [column_name],
        asof_date=asof_date,
        day_counter=day_counter,
    )[column_name]
    return {
        "column_name": column_name,
        "day_counter": day_counter,
        "calendar_dates": list(dates),
        "times": [float(x) for x in times],
        "dfs": [float(x) for x in dfs],
    }


def _collect_convention_ids(curve_configs: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for curve_config in curve_configs:
        ids.append(str(curve_config.get("curve_id", "")))
        for segment in curve_config.get("segments", []):
            ids.append(str(segment.get("conventions", "")))
    deduped: list[str] = []
    seen: set[str] = set()
    for convention_id in ids:
        if not convention_id or convention_id in seen:
            continue
        seen.add(convention_id)
        deduped.append(convention_id)
    return deduped


def _trace_curve_by_handle(
    *,
    ore_xml: Path,
    asof_date: str,
    market_data_file: Path,
    todaysmarket_xml: Path,
    output_path: Path,
    curveconfig_xml: Path,
    conventions_xml: Path,
    configuration_id: str,
    curve_handle: str,
    currency: str,
    trace_type: str,
) -> dict[str, Any]:
    market_quotes = _parse_market_quotes(market_data_file)
    curve_name = _find_curve_name(todaysmarket_xml, curve_handle)
    curve_id = _curve_id_from_handle(curve_handle)

    root_curve_config = _parse_yield_curve(curveconfig_xml, curve_id, market_quotes)
    graph_configs: dict[str, dict[str, Any]] = {}
    pending = [curve_id]
    while pending:
        current = pending.pop()
        if current in graph_configs:
            continue
        current_cfg = _parse_yield_curve(curveconfig_xml, current, market_quotes)
        graph_configs[current] = current_cfg
        for dep in _curve_dependency_ids(current_cfg):
            if dep not in graph_configs:
                pending.append(dep)

    calibration = _load_calibration_rows(output_path / "todaysmarketcalibration.csv", curve_id)
    curve_points = _curve_points(output_path, asof_date, curve_name, calibration["day_counter"] or "A365F")
    native_nodes = _native_curve_nodes(calibration)
    conventions = _parse_conventions(conventions_xml, _collect_convention_ids(list(graph_configs.values())))
    segment_alignment = _align_segments_with_calibration(
        root_curve_config["segments"],
        calibration["pillars"],
    )
    dependency_graph = {
        cid: {
            "curve_config": cfg,
            "dependencies": _curve_dependency_ids(cfg),
        }
        for cid, cfg in graph_configs.items()
    }

    return {
        "asof_date": asof_date,
        "currency": currency,
        "configuration_id": configuration_id,
        "trace_type": trace_type,
        "ore_xml": str(ore_xml),
        "market_data_file": str(market_data_file),
        "todaysmarket_xml": str(todaysmarket_xml),
        "curveconfig_xml": str(curveconfig_xml),
        "conventions_xml": str(conventions_xml),
        "output_path": str(output_path),
        "curve_handle": curve_handle,
        "curve_name": curve_name,
        "curve_config": root_curve_config,
        "discount_curve_dependency": (
            graph_configs.get(str(root_curve_config.get("discount_curve", "")).strip())
            if str(root_curve_config.get("discount_curve", "")).strip()
            and str(root_curve_config.get("discount_curve", "")).strip() != curve_id
            else None
        ),
        "conventions": conventions,
        "segment_alignment": segment_alignment,
        "native_curve_nodes": native_nodes,
        "ore_curve_points": curve_points,
        "ore_calibration_trace": calibration,
        "dependency_graph": dependency_graph,
    }


def trace_curve_handle_from_ore(ore_xml_path: str | Path, curve_handle: str) -> dict[str, Any]:
    ore_xml, asof_date, market_data_file, todaysmarket_xml, output_path = _resolve_ore_run_files(ore_xml_path)
    configuration_id = _simulation_config_id(ore_xml)
    curveconfig_xml = _resolve_curveconfig_path(ore_xml)
    conventions_xml = _resolve_conventions_path(ore_xml)
    parts = curve_handle.split("/")
    if len(parts) < 3 or parts[0] != "Yield":
        raise ValueError(f"Unsupported curve handle '{curve_handle}': only Yield handles are currently supported")
    currency = parts[1]
    return _trace_curve_by_handle(
        ore_xml=ore_xml,
        asof_date=asof_date,
        market_data_file=market_data_file,
        todaysmarket_xml=todaysmarket_xml,
        output_path=output_path,
        curveconfig_xml=curveconfig_xml,
        conventions_xml=conventions_xml,
        configuration_id=configuration_id,
        curve_handle=curve_handle,
        currency=currency,
        trace_type="curve_handle",
    )


def list_curve_handles_from_todaysmarket(
    ore_xml_path: str | Path,
    *,
    configuration_id: str | None = None,
) -> dict[str, list[str]]:
    ore_xml, _, _, todaysmarket_xml, _ = _resolve_ore_run_files(ore_xml_path)
    root = _load_todaysmarket_root(todaysmarket_xml)
    selected_configuration = configuration_id or _simulation_config_id(ore_xml)
    config = _load_configuration_node(root, selected_configuration)

    def _handles(section_tag: str, node_tag: str, id_tag: str) -> list[str]:
        section_id = (config.findtext(f"./{id_tag}") or "").strip()
        if not section_id:
            return []
        section = root.find(f"./{section_tag}[@id='{section_id}']")
        if section is None:
            return []
        handles = []
        for node in section.findall(f"./{node_tag}"):
            text = (node.text or "").strip()
            if text:
                handles.append(text)
        return handles

    return {
        "yield_curves": _handles("YieldCurves", "YieldCurve", "YieldCurvesId"),
        "discounting_curves": _handles("DiscountingCurves", "DiscountingCurve", "DiscountingCurvesId"),
        "index_forwarding_curves": _handles("IndexForwardingCurves", "Index", "IndexForwardingCurvesId"),
        "default_curves": _handles("DefaultCurves", "DefaultCurve", "DefaultCurvesId"),
        "swap_indices": _handles("SwapIndices", "SwapIndex", "SwapIndicesId"),
        "fx_volatilities": _handles("FxVolatilities", "FxVolatility", "FxVolatilitiesId"),
        "swaption_volatilities": _handles("SwaptionVolatilities", "SwaptionVolatility", "SwaptionVolatilitiesId"),
        "cap_floor_volatilities": _handles("CapFloorVolatilities", "CapFloorVolatility", "CapFloorVolatilitiesId"),
    }


def trace_discount_curve_from_ore(ore_xml_path: str | Path, currency: str = "USD") -> dict[str, Any]:
    ore_xml, asof_date, market_data_file, todaysmarket_xml, output_path = _resolve_ore_run_files(ore_xml_path)
    configuration_id = _simulation_config_id(ore_xml)
    curveconfig_xml = _resolve_curveconfig_path(ore_xml)
    conventions_xml = _resolve_conventions_path(ore_xml)
    curve_handle = _find_discount_curve_handle(todaysmarket_xml, configuration_id, currency)
    return _trace_curve_by_handle(
        ore_xml=ore_xml,
        asof_date=asof_date,
        market_data_file=market_data_file,
        todaysmarket_xml=todaysmarket_xml,
        output_path=output_path,
        curveconfig_xml=curveconfig_xml,
        conventions_xml=conventions_xml,
        configuration_id=configuration_id,
        curve_handle=curve_handle,
        currency=currency,
        trace_type="discount_curve",
    )


def trace_index_curve_from_ore(ore_xml_path: str | Path, index_name: str) -> dict[str, Any]:
    ore_xml, asof_date, market_data_file, todaysmarket_xml, output_path = _resolve_ore_run_files(ore_xml_path)
    configuration_id = _simulation_config_id(ore_xml)
    curveconfig_xml = _resolve_curveconfig_path(ore_xml)
    conventions_xml = _resolve_conventions_path(ore_xml)
    curve_handle = _find_index_curve_handle(todaysmarket_xml, configuration_id, index_name)
    currency = _curve_id_from_handle(curve_handle)[:3]
    return _trace_curve_by_handle(
        ore_xml=ore_xml,
        asof_date=asof_date,
        market_data_file=market_data_file,
        todaysmarket_xml=todaysmarket_xml,
        output_path=output_path,
        curveconfig_xml=curveconfig_xml,
        conventions_xml=conventions_xml,
        configuration_id=configuration_id,
        curve_handle=curve_handle,
        currency=currency,
        trace_type="index_curve",
    )


def trace_curve_graph_from_ore(
    ore_xml_path: str | Path,
    *,
    currency: str | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    if bool(currency) == bool(index_name):
        raise ValueError("Provide exactly one of currency or index_name")
    if currency is not None:
        return trace_discount_curve_from_ore(ore_xml_path, currency=currency)
    return trace_index_curve_from_ore(ore_xml_path, index_name=str(index_name))


def trace_usd_curve_from_ore(ore_xml_path: str | Path, currency: str = "USD") -> dict[str, Any]:
    return trace_discount_curve_from_ore(ore_xml_path, currency=currency)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace ORE yield curve setup for an ORE run.")
    parser.add_argument("ore_xml", help="Path to the ORE input XML")
    parser.add_argument(
        "--currency",
        default="USD",
        help="Discounting currency to trace from the selected market configuration (default: USD)",
    )
    parser.add_argument(
        "--index-name",
        default="",
        help="Forwarding index name to trace from the selected market configuration",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    payload = trace_curve_graph_from_ore(
        args.ore_xml,
        currency=args.currency or None,
        index_name=args.index_name or None,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
