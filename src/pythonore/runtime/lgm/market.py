from __future__ import annotations

import csv
import os
import re
import tempfile
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pythonore.runtime.exceptions import EngineRunError

_TENOR_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([YMWD])$", re.IGNORECASE)
_PERIOD_PART_RE = re.compile(r"([0-9]+(?:\\.[0-9]+)?)([YMWD])", re.IGNORECASE)


def _normalize_asof_date(asof: str) -> str:
    s = asof.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _date_from_time(asof: str, t: float) -> str:
    base = datetime.strptime(_normalize_asof_date(asof), "%Y-%m-%d").date()
    return (base + timedelta(days=int(round(float(t) * 365.25)))).isoformat()


def _parse_tenor_to_years(value: str) -> float:
    txt = str(value or "").strip().upper()
    return _parse_tenor_to_years_cached(txt)


@lru_cache(maxsize=16384)
def _parse_tenor_to_years_cached(txt: str) -> float:
    m = _TENOR_RE.match(txt)
    if m is None:
        parts = _PERIOD_PART_RE.findall(txt)
        if not parts or "".join(a + b for a, b in parts).upper() != txt.upper():
            raise ValueError(f"unsupported tenor '{txt}'")
        total = 0.0
        for n_txt, u_txt in parts:
            n = float(n_txt)
            u = u_txt.upper()
            if u == "Y":
                total += n
            elif u == "M":
                total += n / 12.0
            elif u == "W":
                total += n / 52.0
            else:
                total += n / 365.0
        return total
    n = float(m.group(1))
    u = m.group(2).upper()
    if u == "Y":
        return n
    if u == "M":
        return n / 12.0
    if u == "W":
        return n / 52.0
    return n / 365.0


def _parse_float_grid(text: str | None) -> np.ndarray:
    txt = (text or "").strip()
    if not txt:
        return np.array([], dtype=float)
    return np.asarray([float(x.strip()) for x in txt.split(",") if x.strip()], dtype=float)


def _active_dim_mode(snapshot) -> str | None:
    runtime = snapshot.config.runtime
    if runtime is None or runtime.xva_analytic is None:
        return None
    return runtime.xva_analytic.dim_model


def _pfe_quantile(snapshot) -> float:
    runtime = snapshot.config.runtime
    quantile = 0.95
    if runtime is not None and runtime.xva_analytic is not None:
        quantile = float(getattr(runtime.xva_analytic, "pfe_quantile", quantile))
    return min(max(quantile, 0.0), 1.0)


def _configured_output_dir(snapshot) -> Path | None:
    configured = str(snapshot.config.params.get("outputPath", "")).strip()
    if not configured:
        return None
    path = Path(configured)
    if path.is_absolute():
        return path
    ore_xml_path = snapshot.source_meta.get("config")
    if ore_xml_path and ore_xml_path.path and ore_xml_path.path.endswith(".xml"):
        ore_path = Path(ore_xml_path.path)
        return (ore_path.parent.parent / path).resolve()
    return Path.cwd() / path


def _read_csv_report(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _parse_lgm_params_from_xml_node(node: ET.Element, source_label: str) -> dict:
    vol_node = node.find("./Volatility")
    rev_node = node.find("./Reversion")
    trans_node = node.find("./ParameterTransformation")
    if vol_node is None or rev_node is None or trans_node is None:
        raise EngineRunError(
            f"{source_label} LGM node missing Volatility/Reversion/ParameterTransformation"
        )
    return {
        "alpha_times": _parse_float_grid(vol_node.findtext("./TimeGrid")),
        "alpha_values": _parse_float_grid(vol_node.findtext("./InitialValue")),
        "kappa_times": _parse_float_grid(rev_node.findtext("./TimeGrid")),
        "kappa_values": _parse_float_grid(rev_node.findtext("./InitialValue")),
        "shift": 0.0,
        "scaling": float((trans_node.findtext("./Scaling") or "1").strip()),
    }


def _parse_lgm_params_from_simulation_xml_text(xml_text: str, ccy_key: str = "EUR") -> Dict[str, object]:
    root = ET.fromstring(xml_text)
    models = root.find("./CrossAssetModel/InterestRateModels")
    if models is None:
        raise EngineRunError("simulation.xml missing CrossAssetModel/InterestRateModels")
    node = models.find(f"./LGM[@ccy='{ccy_key}']")
    if node is None:
        node = models.find("./LGM[@ccy='default']")
    if node is None:
        raise EngineRunError(f"simulation.xml missing LGM node for {ccy_key}")
    return _parse_lgm_params_from_xml_node(node, "simulation.xml")


def _parse_lgm_params_from_calibration_xml_text(xml_text: str, ccy_key: str = "EUR") -> Dict[str, object]:
    root = ET.fromstring(xml_text)
    models = root.find("./InterestRateModels")
    if models is None:
        raise EngineRunError("calibration.xml missing InterestRateModels")
    node = models.find(f"./LGM[@key='{ccy_key}']")
    if node is None:
        node = models.find(f"./LGM[@ccy='{ccy_key}']")
    if node is None:
        raise EngineRunError(f"calibration.xml missing LGM node for {ccy_key}")
    return _parse_lgm_params_from_xml_node(node, "calibration.xml")


def _parse_zero_quote_time(token: str, asof_date: str | None, day_counter: str = "A365F") -> float | None:
    return _parse_zero_quote_time_cached(str(token).strip(), _normalize_asof_date(asof_date or ""), day_counter)


@lru_cache(maxsize=16384)
def _parse_zero_quote_time_cached(token: str, asof_date: str, day_counter: str) -> float | None:
    if not asof_date:
        return None
    try:
        from pythonore.io import ore_snapshot as ore_snapshot_io

        anchor = datetime.fromisoformat(asof_date).date()
        if re.fullmatch(r"\d{8}", token):
            pillar = datetime.strptime(token, "%Y%m%d").date()
        else:
            pillar = datetime.fromisoformat(token).date()
    except Exception:
        return None
    yf = ore_snapshot_io._year_fraction_from_day_counter(anchor, pillar, day_counter)
    if yf <= 0.0:
        return None
    return float(yf)


def _market_overlay_cache_key(raw_quotes: Sequence[Any], asof_date: str | None) -> tuple[object, ...]:
    return (
        _normalize_asof_date(asof_date or ""),
        tuple((str(q.key).strip().upper(), float(q.value)) for q in raw_quotes),
    )


def _parse_market_overlay(raw_quotes: Sequence[Any], asof_date: str | None = None) -> Dict[str, Any]:
    cache_key = _market_overlay_cache_key(raw_quotes, asof_date)
    cached = getattr(_parse_market_overlay, "_cache", None)
    if cached is None:
        cached = {}
        setattr(_parse_market_overlay, "_cache", cached)
    overlay = cached.get(cache_key)
    if overlay is not None:
        return overlay
    zero: Dict[str, List[Tuple[float, float]]] = {}
    named_zero: Dict[str, List[Tuple[float, float]]] = {}
    fwd: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    fwd_by_index: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    bma_ratio: Dict[str, List[Tuple[float, float]]] = {}
    fx: Dict[str, float] = {}
    fx_vol: Dict[str, List[Tuple[float, float]]] = {}
    swaption_normal_vols: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    cms_correlations: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    hazard: Dict[str, List[Tuple[float, float]]] = {}
    recovery: Dict[str, float] = {}
    cds_spreads: Dict[str, List[Tuple[float, float]]] = {}
    parse_tenor = _parse_tenor_to_years
    parse_zero_time = _parse_zero_quote_time
    for q in raw_quotes:
        key = str(q.key).strip()
        up = key.upper()
        val = float(q.value)
        parts = up.split("/")
        p0 = parts[0] if len(parts) > 0 else ""
        p1 = parts[1] if len(parts) > 1 else ""
        if len(parts) >= 4 and p0 == "FX" and p1 == "RATE":
            fx[parts[2] + parts[3]] = val
            continue
        if len(parts) >= 3 and p0 == "FX":
            fx[parts[1] + parts[2]] = val
            continue
        if len(parts) >= 6 and p0 == "FX_OPTION" and p1 == "RATE_LNVOL":
            ccy1 = parts[2]
            ccy2 = parts[3]
            tenor = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            fx_vol.setdefault(ccy1 + ccy2, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "SWAPTION" and p1 == "RATE_NVOL":
            ccy = parts[2]
            expiry = parts[3]
            swap_tenor = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = parse_tenor(expiry)
            except Exception:
                continue
            swaption_normal_vols.setdefault((ccy, swap_tenor), []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "CORRELATION" and p1 == "RATE":
            idx1 = parts[2]
            idx2 = parts[3]
            expiry = parts[4]
            strike = parts[5]
            if strike != "ATM":
                continue
            try:
                t = parse_tenor(expiry)
            except Exception:
                continue
            cms_correlations.setdefault(tuple(sorted((idx1, idx2))), []).append((t, val))
            continue
        if len(parts) >= 4 and p0 == "ZERO" and p1 == "RATE":
            ccy = parts[2]
            tenor = parts[3]
            curve_name = None
            day_counter = "A365F"
            if len(parts) >= 6:
                curve_name = parts[3]
                if len(parts) >= 7:
                    day_counter = parts[-2]
                tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                t = parse_zero_time(tenor, asof_date, day_counter=day_counter)
                if t is None:
                    continue
            if curve_name is not None:
                named_zero.setdefault(curve_name, []).append((t, val))
            else:
                zero.setdefault(ccy, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "IR_SWAP" and p1 == "RATE":
            ccy = parts[2]
            idx_name = parts[4].upper()
            idx_tenor = idx_name.split("-")[-1].upper()
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            if 0.0 < t <= 80.0:
                if idx_tenor in ("1D", "ON", "O/N"):
                    zero.setdefault(ccy, []).append((t, val))
                else:
                    fwd_by_index.setdefault(ccy, {}).setdefault(idx_name, []).append((t, val))
                    fwd.setdefault(ccy, {}).setdefault(idx_tenor, []).append((t, val))
            continue
        if len(parts) >= 5 and p0 == "BMA_SWAP" and p1 == "RATIO":
            ccy = parts[2]
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            if 0.0 < t <= 80.0:
                bma_ratio.setdefault(ccy, []).append((t, val))
            continue
        if len(parts) >= 5 and p0 == "MM" and p1 == "RATE":
            ccy = parts[2]
            idx_name = parts[4].upper()
            idx_tenor = idx_name.split("-")[-1].upper()
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            if 0.0 < t <= 10.0:
                if idx_tenor in ("1D", "ON", "O/N"):
                    zero.setdefault(ccy, []).append((t, val))
                else:
                    fwd_by_index.setdefault(ccy, {}).setdefault(idx_name, []).append((t, val))
                    fwd.setdefault(ccy, {}).setdefault(idx_tenor, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "HAZARD_RATE":
            cpty = parts[2]
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            hazard.setdefault(cpty, []).append((t, val))
            continue
        if len(parts) >= 6 and p0 == "CDS" and p1 == "CREDIT_SPREAD":
            cpty = parts[2]
            tenor = parts[-1]
            try:
                t = parse_tenor(tenor)
            except Exception:
                continue
            cds_spreads.setdefault(cpty, []).append((t, val))
            continue
        if len(parts) >= 5 and p0 == "RECOVERY_RATE":
            cpty = parts[2]
            recovery[cpty] = val
            continue
    for cpty, spreads in cds_spreads.items():
        if cpty in hazard:
            continue
        rec = float(recovery.get(cpty, 0.4))
        lgd = max(1.0 - rec, 1.0e-6)
        hazard[cpty] = [(t, max(float(spread) / lgd, 0.0)) for t, spread in spreads]
    overlay = {
        "zero": zero,
        "named_zero": named_zero,
        "fwd": fwd,
        "fwd_by_index": fwd_by_index,
        "bma_ratio": bma_ratio,
        "fx": fx,
        "fx_vol": fx_vol,
        "swaption_normal_vols": swaption_normal_vols,
        "cms_correlations": cms_correlations,
        "hazard": hazard,
        "cds_spreads": cds_spreads,
        "recovery": recovery,
    }
    cached[cache_key] = overlay
    return overlay


def _default_index_for_ccy(ccy: str) -> str:
    c = ccy.upper()
    if c == "USD":
        return "USD-LIBOR-3M"
    if c == "GBP":
        return "GBP-LIBOR-3M"
    if c == "CHF":
        return "CHF-LIBOR-3M"
    if c == "JPY":
        return "JPY-LIBOR-3M"
    return "EUR-EURIBOR-3M"


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


def _zero_rate_from_curve(curve: Callable[[float], float], t: float) -> float:
    tt = max(float(t), 1.0e-8)
    return float(-np.log(max(float(curve(tt)), 1.0e-18)) / tt)


def _build_zero_rate_shocked_curve(
    base_curve: Callable[[float], float],
    node_times: Sequence[float],
    node_shifts: Sequence[float],
) -> Callable[[float], float]:
    from pythonore.compute import irs_xva_utils as irs_utils

    times = np.asarray([float(x) for x in node_times], dtype=float)
    shifts = np.asarray([float(x) for x in node_shifts], dtype=float)
    if times.size == 0 or times.size != shifts.size:
        return base_curve
    order = np.argsort(times)
    times = times[order]
    shifts = shifts[order]

    shocked_node_logs = np.log(
        np.clip(
            np.asarray([float(base_curve(float(t))) for t in times], dtype=float)
            * np.exp(-times * shifts),
            1.0e-18,
            None,
        )
    )
    scalar_cache: Dict[float, float] = {}

    def shocked_curve(t: float) -> float:
        tt = max(float(t), 0.0)
        cached = scalar_cache.get(tt)
        if cached is not None:
            return cached
        if tt <= 1.0e-12:
            return 1.0
        if tt <= times[0]:
            base_df = float(base_curve(tt))
            out = float(base_df * np.exp(-shifts[0] * tt))
            scalar_cache[tt] = out
            return out
        if tt >= times[-1]:
            base_df = float(base_curve(tt))
            out = float(base_df * np.exp(-shifts[-1] * tt))
            scalar_cache[tt] = out
            return out

        shocked_log_df = float(irs_utils.interpolate_linear_flat(tt, times, shocked_node_logs))
        out = float(np.exp(shocked_log_df))
        scalar_cache[tt] = out
        return out

    return shocked_curve


def _apply_curve_node_shocks(
    snapshot,
    discount_curves: Dict[str, Callable[[float], float]],
    forward_curves: Dict[str, Callable[[float], float]],
    forward_curves_by_tenor: Dict[str, Dict[str, Callable[[float], float]]],
    forward_curves_by_name: Dict[str, Callable[[float], float]],
    xva_discount_curve: Optional[Callable[[float], float]],
) -> Tuple[
    Dict[str, Callable[[float], float]],
    Dict[str, Callable[[float], float]],
    Dict[str, Dict[str, Callable[[float], float]]],
    Dict[str, Callable[[float], float]],
    Optional[Callable[[float], float]],
]:
    specs = snapshot.config.params.get("python.curve_node_shocks")
    if not isinstance(specs, dict):
        return discount_curves, forward_curves, forward_curves_by_tenor, forward_curves_by_name, xva_discount_curve

    discount_specs = specs.get("discount", {}) if isinstance(specs.get("discount", {}), dict) else {}
    forward_specs = specs.get("forward", {}) if isinstance(specs.get("forward", {}), dict) else {}

    discount_curves = dict(discount_curves)
    forward_curves = dict(forward_curves)
    forward_curves_by_tenor = {ccy: dict(v) for ccy, v in forward_curves_by_tenor.items()}
    forward_curves_by_name = dict(forward_curves_by_name)

    for ccy, cfg in discount_specs.items():
        if ccy not in discount_curves or not isinstance(cfg, dict):
            continue
        node_times = cfg.get("node_times", ())
        node_shifts = cfg.get("node_shifts", ())
        bumped = _build_zero_rate_shocked_curve(discount_curves[ccy], node_times, node_shifts)
        discount_curves[ccy] = bumped
        if snapshot.config.base_currency.upper() == str(ccy).upper() and xva_discount_curve is not None:
            xva_discount_curve = bumped

    for ccy, tenor_map in forward_specs.items():
        if not isinstance(tenor_map, dict):
            continue
        tenor_curves = dict(forward_curves_by_tenor.get(ccy, {}))
        for tenor, cfg in tenor_map.items():
            if tenor not in tenor_curves or not isinstance(cfg, dict):
                continue
            node_times = cfg.get("node_times", ())
            node_shifts = cfg.get("node_shifts", ())
            bumped = _build_zero_rate_shocked_curve(tenor_curves[tenor], node_times, node_shifts)
            tenor_curves[tenor] = bumped
            for name, curve in list(forward_curves_by_name.items()):
                if curve is forward_curves_by_tenor.get(ccy, {}).get(tenor):
                    forward_curves_by_name[name] = bumped
            if ccy in forward_curves:
                preferred = "6M" if "6M" in tenor_curves else (sorted(tenor_curves)[0] if tenor_curves else "")
                if preferred and preferred == tenor:
                    forward_curves[ccy] = bumped
        forward_curves_by_tenor[ccy] = tenor_curves

    return discount_curves, forward_curves, forward_curves_by_tenor, forward_curves_by_name, xva_discount_curve


def _quote_matches_discount_curve(
    key: str,
    ccy: str,
    source_column: str,
    fallback_family: str = "",
) -> bool:
    parts = str(key).strip().upper().split("/")
    if len(parts) < 3 or parts[2] != ccy.upper():
        return False
    family = _curve_family_from_source_column(source_column)
    if not family:
        family = str(fallback_family).strip().upper()
    if parts[0] == "ZERO" and parts[1] == "RATE":
        if len(parts) == 4:
            return True
        if len(parts) >= 6:
            return parts[3] == family or family in {"", "1D", "ON", "O/N"}
        return False
    if parts[0] == "MM" and parts[1] == "RATE":
        return True
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        index_tenor = parts[4] if len(parts) > 5 else ""
        if family in ("", "1D", "ON", "O/N"):
            return index_tenor in ("1D", "ON", "O/N")
        return index_tenor == family
    if parts[0] == "FRA" and parts[1] == "RATE":
        return family not in ("", "1D", "ON", "O/N") and len(parts) > 4 and parts[-1] == family
    return False


def _normalize_forward_tenor_family(tenor: str) -> str:
    txt = str(tenor).strip().upper()
    if not txt:
        return ""
    if txt in {"O/N", "ON", "1D", "0D"}:
        return "1D"
    if txt.endswith("D") and txt[:-1].isdigit():
        return "1D"
    if txt.endswith("W") and txt[:-1].isdigit():
        return "1W"
    if txt.endswith("M") and txt[:-1].isdigit():
        return f"{int(txt[:-1])}M" if txt[:-1].isdigit() else txt
    if txt.endswith("Y") and txt[:-1].isdigit():
        return f"{int(txt[:-1])}Y" if txt[:-1].isdigit() else txt
    return txt


def _normalize_curve_lookup_key(name: str) -> str:
    return str(name).strip().upper()


def _forward_index_family(index_name: str, swap_index_forward_tenors: Mapping[str, str] | None = None) -> str:
    key = _normalize_curve_lookup_key(index_name)
    if not key:
        return ""
    if key in {"USD-SIFMA", "USD-BMA", "USD-SIFMA-1W", "USD-SIFMA-7D", "USD-BMA-1W", "USD-BMA-7D"}:
        return "1D"
    if swap_index_forward_tenors:
        direct = swap_index_forward_tenors.get(key)
        if direct:
            return _normalize_forward_tenor_family(direct)
    if key.endswith("ON") or key.endswith("O/N") or key.endswith("1D"):
        return "1D"
    m = re.search(r"(\d+[DWMY])$", key)
    if m:
        return _normalize_forward_tenor_family(m.group(1))
    if "ESTR" in key or "SOFR" in key or "SONIA" in key or "TONAR" in key:
        return "1D"
    return ""


def _index_name_matches_quote_token(token: str, index_name: str, ccy: str) -> bool:
    token_key = _normalize_curve_lookup_key(token)
    index_key = _normalize_curve_lookup_key(index_name)
    if token_key == index_key:
        return True
    if token_key == _normalize_curve_lookup_key(ccy):
        return False
    token_family = _forward_index_family(token_key)
    index_family = _forward_index_family(index_key)
    return bool(token_family and token_family == index_family)


def _quote_matches_forward_curve(key: str, ccy: str, tenor: str, index_name: str = "") -> bool:
    parts = str(key).strip().upper().split("/")
    if len(parts) < 3 or parts[2] != ccy.upper():
        return False
    family = _normalize_forward_tenor_family(tenor)
    exact_index = _normalize_curve_lookup_key(index_name)
    if parts[0] == "ZERO" and parts[1] == "RATE":
        if exact_index:
            return len(parts) >= 6 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        if family == "1D":
            return True
        if len(parts) == 4:
            return False
        if len(parts) >= 6:
            return _normalize_forward_tenor_family(parts[3]) == family
        return False
    if parts[0] == "MM" and parts[1] == "RATE":
        if exact_index:
            return len(parts) >= 6 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        return True
    if parts[0] == "IR_SWAP" and parts[1] == "RATE":
        if exact_index:
            return len(parts) > 5 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        return len(parts) > 5 and _normalize_forward_tenor_family(parts[4]) == family
    if parts[0] == "FRA" and parts[1] == "RATE":
        if exact_index:
            return len(parts) > 4 and _index_name_matches_quote_token(parts[3], exact_index, ccy)
        return len(parts) > 4 and _normalize_forward_tenor_family(parts[-1]) == family
    return False


__all__ = [
    "_active_dim_mode",
    "_apply_curve_node_shocks",
    "_build_zero_rate_shocked_curve",
    "_configured_output_dir",
    "_curve_family_from_source_column",
    "_date_from_time",
    "_default_index_for_ccy",
    "_forward_index_family",
    "_index_name_matches_quote_token",
    "_normalize_asof_date",
    "_normalize_curve_lookup_key",
    "_normalize_forward_tenor_family",
    "_parse_float_grid",
    "_parse_lgm_params_from_calibration_xml_text",
    "_parse_lgm_params_from_simulation_xml_text",
    "_parse_market_overlay",
    "_parse_ore_tenor_to_years",
    "_parse_tenor_to_years",
    "_parse_zero_quote_time",
    "_pfe_quantile",
    "_quote_matches_discount_curve",
    "_quote_matches_forward_curve",
    "_read_csv_report",
    "_zero_rate_from_curve",
]


def _parse_ore_tenor_to_years(tenor: str) -> float:
    txt = str(tenor).strip().upper()
    match = re.match(r"^(\d+)([YMWD])$", txt)
    if match is None:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "Y":
        return value
    if unit == "M":
        return value / 12.0
    if unit == "W":
        return value / 52.0
    return value / 365.0
