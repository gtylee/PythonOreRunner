from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import csv
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from pythonore.domain.dataclasses import (
    BermudanSwaption,
    CollateralBalance,
    CollateralConfig,
    EquityForward,
    EquityOption,
    EquitySwap,
    FXForward,
    IndependentAmount,
    FixingPoint,
    FixingsData,
    GenericProduct,
    InflationCapFloor,
    InflationSwap,
    IRS,
    MarginingFrequency,
    MarketData,
    MarketQuote,
    MporConfig,
    NettingConfig,
    NettingSet,
    Portfolio,
    SourceMeta,
    Trade,
    XVAConfig,
    XVASnapshot,
)
from pythonore.runtime.exceptions import ConflictError, ValidationError


ConflictPolicy = Literal["override", "error"]


def _market_quote_fast(date: str, key: str, value: float) -> MarketQuote:
    quote = object.__new__(MarketQuote)
    object.__setattr__(quote, "date", date)
    object.__setattr__(quote, "key", key)
    object.__setattr__(quote, "value", float(value))
    return quote


def _fixing_point_fast(date: str, index: str, value: float) -> FixingPoint:
    point = object.__new__(FixingPoint)
    object.__setattr__(point, "date", date)
    object.__setattr__(point, "index", index)
    object.__setattr__(point, "value", float(value))
    return point


def _first_samples_count_from_simulation_xml(xml: str) -> Optional[int]:
    """Return the first ``<Samples>N</Samples>`` from simulation config XML, if present."""
    if not xml:
        return None
    match = re.search(r"<Samples>\s*(\d+)\s*</Samples>", xml, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        n = int(match.group(1))
    except ValueError:
        return None
    return n if n > 0 else None


class XVALoader:
    @staticmethod
    def from_files(input_dir: str, ore_file: Optional[str] = None) -> XVASnapshot:
        base_dir = Path(input_dir)
        if not base_dir.exists():
            raise ValidationError(
                f"Input directory does not exist: {input_dir}. "
                "Fix: point XVALoader.from_files(...) at the ORE Input directory containing "
                "ore.xml, portfolio.xml, todaysmarket.xml, and related input files."
            )

        ore_path = _pick_ore_file(base_dir, ore_file)
        return XVALoader.from_ore_xml(ore_path)

    @staticmethod
    def from_ore_xml(path: str | Path) -> XVASnapshot:
        ore_path = Path(path)
        if not ore_path.exists():
            raise ValidationError(
                f"ORE file not found: {ore_path}. "
                "Fix: pass a valid ore.xml path, or use XVALoader.from_files(...) with the Input directory."
            )
        if ore_path.is_dir():
            raise ValidationError(
                f"Expected an ore.xml file but received a directory: {ore_path}. "
                "Fix: pass the ore.xml file path or use XVALoader.from_files(...) instead."
            )
        ore_tree = ET.parse(ore_path)
        ore_root = ore_tree.getroot()
        setup = _parse_ore_setup(ore_root)
        markets = _parse_markets(ore_root)
        analytics = _parse_analytics(ore_root)

        asof = setup.get("asofDate")
        if not asof:
            raise ValidationError(
                "Missing Setup/asofDate in ore file. "
                "Fix: add <Parameter name=\"asofDate\">YYYY-MM-DD</Parameter> under <Setup>."
            )

        input_path = setup.get("inputPath", ".")

        portfolio_file = setup.get("portfolioFile")
        if not portfolio_file:
            raise ValidationError(
                "Missing Setup/portfolioFile in ore file. "
                "Fix: add <Parameter name=\"portfolioFile\">...</Parameter> under <Setup>, "
                "or do not use XVALoader.from_files(...) for this snapshot."
            )
        portfolio_path = _resolve_ref(portfolio_file, ore_path, input_path)

        netting_file = analytics.get("xva", {}).get("csaFile") or setup.get("nettingSetFile") or "netting.xml"
        netting_path = _resolve_ref(netting_file, ore_path, input_path)

        market_file = setup.get("marketDataFile")
        fixing_file = setup.get("fixingDataFile")

        if market_file:
            market_path = _resolve_ref(market_file, ore_path, input_path)
            market_quotes = _load_market_quotes(market_path)
        else:
            market_quotes = ()

        if fixing_file:
            fixing_path = _resolve_ref(fixing_file, ore_path, input_path)
            fixings = _load_fixings(fixing_path)
        else:
            fixings = ()

        portfolio = _parse_portfolio(portfolio_path)
        netting = _parse_netting(netting_path)
        collateral = _parse_collateral(_resolve_ref("collateralbalances.xml", ore_path, input_path), required=False)

        base_currency = (
            setup.get("baseCurrency")
            or analytics.get("xva", {}).get("baseCurrency")
            or analytics.get("simulation", {}).get("baseCurrency")
            or analytics.get("npv", {}).get("baseCurrency")
            or "EUR"
        )

        metrics = tuple(
            m
            for m, flag in (
                ("CVA", analytics.get("xva", {}).get("cva", "Y")),
                ("DVA", analytics.get("xva", {}).get("dva", "N")),
                ("FVA", analytics.get("xva", {}).get("fva", "N")),
                ("MVA", analytics.get("xva", {}).get("mva", "N")),
            )
            if str(flag).upper() == "Y"
        ) or ("CVA",)

        xml_buffers = _load_known_xml_buffers(ore_path, setup, analytics, input_path)

        sim_xml = xml_buffers.get("simulation.xml") or ""
        parsed_samples = _first_samples_count_from_simulation_xml(sim_xml)
        num_paths = parsed_samples if parsed_samples is not None else 5000

        config = XVAConfig(
            asof=_normalize_date(asof),
            base_currency=base_currency,
            analytics=metrics,
            num_paths=num_paths,
            params={**setup, **{f"market.{k}": v for k, v in markets.items()}},
            xml_buffers=xml_buffers,
            mpor=_resolve_snapshot_mpor(
                base_currency=base_currency,
                params={**setup, **analytics.get("xva", {})},
                xml_buffers=xml_buffers,
                netting=netting,
            ),
            source_meta=SourceMeta(origin="file", path=str(ore_path)),
        )

        market = MarketData(
            asof=_normalize_date(asof),
            raw_quotes=market_quotes,
            source_meta=SourceMeta(origin="file", path=str(market_file) if market_file else str(ore_path)),
        )
        fixing_data = FixingsData(points=fixings, source_meta=SourceMeta(origin="file", path=str(fixing_file) if fixing_file else str(ore_path)))

        source_meta = {
            "market": market.source_meta,
            "fixings": fixing_data.source_meta,
            "portfolio": portfolio.source_meta,
            "config": config.source_meta,
            "netting": netting.source_meta,
            "collateral": collateral.source_meta,
        }

        return XVASnapshot(
            market=market,
            fixings=fixing_data,
            portfolio=portfolio,
            config=config,
            netting=netting,
            collateral=collateral,
            source_meta=source_meta,
        )

    @staticmethod
    def from_mixed(input_dir: str, overrides: XVASnapshot, ore_file: Optional[str] = None, on_conflict: ConflictPolicy = "override") -> XVASnapshot:
        base = XVALoader.from_files(input_dir=input_dir, ore_file=ore_file)
        return merge_snapshots(base, overrides, on_conflict=on_conflict)


def merge_snapshots(base: XVASnapshot, overrides: XVASnapshot, on_conflict: ConflictPolicy = "override") -> XVASnapshot:
    if on_conflict not in ("override", "error"):
        raise ValidationError(f"Unknown conflict policy: {on_conflict}")

    market = _merge_market(base.market, overrides.market, on_conflict)
    fixings = _merge_fixings(base.fixings, overrides.fixings, on_conflict)
    portfolio = _merge_portfolio(base.portfolio, overrides.portfolio, on_conflict)
    netting = _merge_netting(base.netting, overrides.netting, on_conflict)
    collateral = _merge_collateral(base.collateral, overrides.collateral, on_conflict)
    config = _merge_config(base.config, overrides.config, on_conflict)

    source_meta = dict(base.source_meta)
    source_meta.update(overrides.source_meta)

    return XVASnapshot(
        market=market,
        fixings=fixings,
        portfolio=portfolio,
        netting=netting,
        collateral=collateral,
        config=config,
        source_meta=source_meta,
    )


def _merge_market(base: MarketData, override: MarketData, on_conflict: ConflictPolicy) -> MarketData:
    q = {(x.date, x.key): x for x in base.raw_quotes}
    for x in override.raw_quotes:
        key = (x.date, x.key)
        if on_conflict == "error" and key in q and q[key].value != x.value:
            raise ConflictError(f"Market quote conflict at {key}")
        q[key] = x
    return replace(override if override.raw_quotes else base, raw_quotes=tuple(q.values()), source_meta=SourceMeta(origin="merged"))


def _merge_fixings(base: FixingsData, override: FixingsData, on_conflict: ConflictPolicy) -> FixingsData:
    p = {(x.date, x.index): x for x in base.points}
    for x in override.points:
        key = (x.date, x.index)
        if on_conflict == "error" and key in p and p[key].value != x.value:
            raise ConflictError(f"Fixing conflict at {key}")
        p[key] = x
    return FixingsData(points=tuple(p.values()), source_meta=SourceMeta(origin="merged"))


def _merge_portfolio(base: Portfolio, override: Portfolio, on_conflict: ConflictPolicy) -> Portfolio:
    t = {x.trade_id: x for x in base.trades}
    for x in override.trades:
        if on_conflict == "error" and x.trade_id in t and asdict(t[x.trade_id]) != asdict(x):
            raise ConflictError(f"Trade conflict for {x.trade_id}")
        t[x.trade_id] = x
    return Portfolio(trades=tuple(t.values()), source_meta=SourceMeta(origin="merged"))


def _merge_netting(base: NettingConfig, override: NettingConfig, on_conflict: ConflictPolicy) -> NettingConfig:
    merged = dict(base.netting_sets)
    for k, v in override.netting_sets.items():
        if on_conflict == "error" and k in merged and asdict(merged[k]) != asdict(v):
            raise ConflictError(f"Netting config conflict for {k}")
        merged[k] = v
    return NettingConfig(netting_sets=merged, source_meta=SourceMeta(origin="merged"))


def _merge_collateral(base: CollateralConfig, override: CollateralConfig, on_conflict: ConflictPolicy) -> CollateralConfig:
    merged = {x.netting_set_id: x for x in base.balances}
    for x in override.balances:
        if on_conflict == "error" and x.netting_set_id in merged and asdict(merged[x.netting_set_id]) != asdict(x):
            raise ConflictError(f"Collateral conflict for {x.netting_set_id}")
        merged[x.netting_set_id] = x
    return CollateralConfig(balances=tuple(merged.values()), source_meta=SourceMeta(origin="merged"))


def _merge_config(base: XVAConfig, override: XVAConfig, on_conflict: ConflictPolicy) -> XVAConfig:
    params = dict(base.params)
    for k, v in override.params.items():
        if on_conflict == "error" and k in params and params[k] != v:
            raise ConflictError(f"Config conflict for parameter {k}")
        params[k] = v

    xml_buffers = dict(base.xml_buffers)
    xml_buffers.update(override.xml_buffers)

    return XVAConfig(
        asof=override.asof or base.asof,
        base_currency=override.base_currency or base.base_currency,
        analytics=override.analytics or base.analytics,
        num_paths=override.num_paths or base.num_paths,
        horizon_years=override.horizon_years or base.horizon_years,
        params=params,
        xml_buffers=xml_buffers,
        runtime=override.runtime or base.runtime,
        mpor=override.mpor if override.mpor != MporConfig() else base.mpor,
        source_meta=SourceMeta(origin="merged"),
    )


def _pick_ore_file(base_dir: Path, ore_file: Optional[str]) -> Path:
    if ore_file:
        p = base_dir / ore_file if not Path(ore_file).is_absolute() else Path(ore_file)
        if not p.exists():
            raise ValidationError(
                f"ORE file not found: {p}. "
                "Fix: pass the ore xml filename relative to the Input directory, "
                "for example ore_file='ore_stress_classic.xml', or use an absolute path."
            )
        return p
    explicit = base_dir / "ore.xml"
    if explicit.exists():
        return explicit
    cands = sorted(base_dir.glob("ore*.xml"))
    if not cands:
        raise ValidationError(
            f"No ore xml files found in {base_dir}. "
            "Fix: point the loader at the ORE Input directory, not the run root or Output directory."
        )
    return cands[0]


def _parse_ore_setup(root: ET.Element) -> Dict[str, str]:
    setup: Dict[str, str] = {}
    for p in root.findall("./Setup/Parameter"):
        name = p.attrib.get("name")
        if name:
            setup[name] = (p.text or "").strip()
    return setup


def _parse_analytics(root: ET.Element) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for node in root.findall("./Analytics/Analytic"):
        analytic_type = node.attrib.get("type", "unknown")
        out[analytic_type] = {}
        for p in node.findall("./Parameter"):
            name = p.attrib.get("name")
            if name:
                out[analytic_type][name] = (p.text or "").strip()
    return out


def _parse_markets(root: ET.Element) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in root.findall("./Markets/Parameter"):
        name = p.attrib.get("name")
        if name:
            out[name] = (p.text or "").strip()
    return out


def _resolve_ref(ref: str, ore_file: Path, input_path: str) -> Path:
    p = Path(ref)
    if p.is_absolute() and p.exists():
        return p

    run_dir = ore_file.parent.parent
    candidates = [
        (ore_file.parent / p).resolve(),
        (run_dir / input_path / p).resolve(),
        (ore_file.parent / p.name).resolve(),
        (run_dir / input_path / p.name).resolve(),
        (ore_file.parents[2] / "Examples" / "Input" / p.name).resolve() if len(ore_file.parents) > 2 else None,
        (Path(__file__).resolve().parents[3] / "Examples" / "Input" / p.name).resolve(),
    ]

    for c in candidates:
        if c is not None and c.exists():
            return c

    # Keep deterministic fallback for optional files. Required files should be
    # validated by the caller so that their error messages can explain how the
    # ORE case should be wired, instead of failing here with an opaque path miss.
    return candidates[0]


def _load_market_quotes(path: Path) -> Tuple[MarketQuote, ...]:
    if path.suffix.lower() == ".csv":
        return _load_market_csv_cached(str(path.resolve()))
    quotes: List[MarketQuote] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            quotes.append(MarketQuote(date=_normalize_date(parts[0]), key=parts[1], value=float(parts[2])))
    return tuple(quotes)


def _load_market_csv(path: Path) -> Tuple[MarketQuote, ...]:
    return _load_market_csv_cached(str(path.resolve()))


@lru_cache(maxsize=16)
def _load_market_csv_cached(path_text: str) -> Tuple[MarketQuote, ...]:
    path = Path(path_text)
    quotes: List[MarketQuote] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return ()
        cols = {c.strip().lower(): i for i, c in enumerate(header)}
        date_i = cols.get("date", cols.get("asofdate", 0))
        key_i = cols.get("quote", cols.get("key", 1))
        val_i = cols.get("value", cols.get("quotevalue", 2))
        for row in reader:
            if len(row) <= max(date_i, key_i, val_i):
                continue
            date_text = str(row[date_i]).strip()
            if not date_text or date_text.startswith("#"):
                continue
            quotes.append(
                _market_quote_fast(
                    _normalize_date(date_text),
                    row[key_i].strip(),
                    float(row[val_i]),
                )
            )
    return tuple(quotes)


def _load_fixings(path: Path) -> Tuple[FixingPoint, ...]:
    if path.suffix.lower() == ".csv":
        return _load_fixings_csv_cached(str(path.resolve()))

    out: List[FixingPoint] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            out.append(FixingPoint(date=_normalize_date(parts[0]), index=parts[1], value=float(parts[2])))
    return tuple(out)


def _load_fixings_csv(path: Path) -> Tuple[FixingPoint, ...]:
    return _load_fixings_csv_cached(str(path.resolve()))


@lru_cache(maxsize=16)
def _load_fixings_csv_cached(path_text: str) -> Tuple[FixingPoint, ...]:
    path = Path(path_text)
    out: List[FixingPoint] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return ()
        cols = {c.strip().lower(): i for i, c in enumerate(header)}
        date_i = cols.get("date", 0)
        idx_i = cols.get("index", cols.get("name", 1))
        val_i = cols.get("value", 2)
        for row in reader:
            if len(row) <= max(date_i, idx_i, val_i):
                continue
            out.append(
                _fixing_point_fast(
                    _normalize_date(str(row[date_i]).strip()),
                    row[idx_i].strip(),
                    float(row[val_i]),
                )
            )
    return tuple(out)


def _parse_portfolio(path: Path) -> Portfolio:
    tree = ET.parse(path)
    root = tree.getroot()
    trades: List[Trade] = []
    for t in root.findall("./Trade"):
        trade_id = t.attrib.get("id", "")
        trade_type = _text(t, "./TradeType") or "Unknown"
        cpty = _text(t, "./Envelope/CounterParty") or "UNKNOWN"
        ns = _text(t, "./Envelope/NettingSetId") or cpty

        product = _parse_product_from_trade_xml(t, trade_type)
        additional: Dict[str, str] = {}
        for af in t.findall("./Envelope/AdditionalFields/*"):
            additional[af.tag] = (af.text or "").strip()
        if trade_type == "Swap":
            for leg in t.findall(".//SwapData/LegData"):
                if (_text(leg, "./LegType") or "").lower() != "floating":
                    continue
                index = (_text(leg, "./FloatingLegData/Index") or "").strip()
                if index and "index" not in additional:
                    additional["index"] = index
                fixing_days = (_text(leg, "./FloatingLegData/FixingDays") or "").strip()
                if fixing_days and "fixingDays" not in additional:
                    additional["fixingDays"] = fixing_days
                break

        trades.append(
            Trade(
                trade_id=trade_id,
                counterparty=cpty,
                netting_set=ns,
                trade_type=trade_type,
                product=product,
                additional_fields=additional,
            )
        )
    return Portfolio(trades=tuple(trades), source_meta=SourceMeta(origin="file", path=str(path)))


def _parse_product_from_trade_xml(trade: ET.Element, trade_type: str):
    if trade_type == "Swap":
        fixed_leg = None
        float_leg = None
        inflation_leg = None
        generic_rate_leg = None
        for leg in trade.findall(".//SwapData/LegData"):
            leg_type = (_text(leg, "./LegType") or "").lower()
            if leg_type in {"fixed", "zerocouponfixed"}:
                fixed_leg = leg
            elif leg_type == "floating":
                float_leg = leg
            elif leg_type in {"cpi", "yy"}:
                inflation_leg = leg
            elif leg_type in {"cms", "cmsspread", "digitalcmsspread"}:
                generic_rate_leg = leg
        if inflation_leg is not None:
            ccy = _text(inflation_leg, "./Currency") or _text(fixed_leg, "./Currency") or _text(float_leg, "./Currency") or "EUR"
            notional = float(
                _text(inflation_leg, "./Notionals/Notional")
                or _text(fixed_leg, "./Notionals/Notional")
                or _text(float_leg, "./Notionals/Notional")
                or 0.0
            )
            start = (
                _text(inflation_leg, "./CPILegData/StartDate")
                or _text(inflation_leg, "./ScheduleData/Rules/StartDate")
                or _text(fixed_leg, "./ScheduleData/Rules/StartDate")
                or _text(float_leg, "./ScheduleData/Rules/StartDate")
            )
            end = (
                _text(inflation_leg, "./ScheduleData/Rules/EndDate")
                or _text(inflation_leg, "./ScheduleData/Dates/Dates/Date")
                or _text(fixed_leg, "./ScheduleData/Rules/EndDate")
                or _text(float_leg, "./ScheduleData/Rules/EndDate")
            )
            maturity = _maturity_years(start, end)
            infl_type = (_text(inflation_leg, "./LegType") or "CPI").upper()
            pay_leg = "inflation" if (_text(inflation_leg, "./Payer") or "false").lower() == "true" else (
                "fixed" if fixed_leg is not None and (_text(fixed_leg, "./Payer") or "false").lower() == "true" else "float"
            )
            fixed_rate = float(
                (_text(fixed_leg, "./FixedLegData/Rates/Rate") if fixed_leg is not None else None)
                or (_text(fixed_leg, "./ZeroCouponFixedLegData/Rates/Rate") if fixed_leg is not None else None)
                or (_text(float_leg, "./FloatingLegData/Spreads/Spread") if float_leg is not None else None)
                or 0.0
            )
            return InflationSwap(
                ccy=ccy,
                notional=notional,
                maturity_years=max(maturity, 1.0 / 365.25),
                index=_text(inflation_leg, "./CPILegData/Index") or _text(inflation_leg, "./YYLegData/Index") or "",
                inflation_type="CPI" if infl_type == "CPI" else "YY",
                pay_leg=pay_leg,
                fixed_rate=fixed_rate,
                base_cpi=(
                    float(_text(inflation_leg, "./CPILegData/BaseCPI"))
                    if _text(inflation_leg, "./CPILegData/BaseCPI")
                    else None
                ),
                observation_lag=_text(inflation_leg, "./CPILegData/ObservationLag") or _text(inflation_leg, "./YYLegData/ObservationLag"),
                start_date=_normalize_date(start) if start else None,
                end_date=_normalize_date(end) if end else None,
                schedule_tenor=_text(inflation_leg, "./ScheduleData/Rules/Tenor") or "1Y",
            )
        if generic_rate_leg is not None:
            return GenericProduct(payload={"trade_type": trade_type, "subtype": "GenericRateSwap", "xml": _product_inner_xml(trade)})
        if fixed_leg is not None and float_leg is not None:
            ccy = _text(fixed_leg, "./Currency") or "EUR"
            notional = float(_text(fixed_leg, "./Notionals/Notional") or 0.0)
            fixed_rate = float(_text(fixed_leg, "./FixedLegData/Rates/Rate") or 0.0)
            pay_fixed = (_text(fixed_leg, "./Payer") or "true").lower() == "true"
            start = _text(fixed_leg, "./ScheduleData/Rules/StartDate")
            end = _text(fixed_leg, "./ScheduleData/Rules/EndDate")
            maturity = _maturity_years(start, end)
            if notional > 0 and maturity > 0:
                return IRS(
                    ccy=ccy,
                    notional=notional,
                    fixed_rate=fixed_rate,
                    maturity_years=maturity,
                    pay_fixed=pay_fixed,
                    start_date=_normalize_date(start) if start else None,
                    end_date=_normalize_date(end) if end else None,
                    fixed_leg_tenor=_text(fixed_leg, "./ScheduleData/Rules/Tenor") or "6M",
                    float_leg_tenor=_text(float_leg, "./ScheduleData/Rules/Tenor") or "3M",
                    fixed_day_counter=_text(fixed_leg, "./DayCounter") or None,
                    float_day_counter=_text(float_leg, "./DayCounter") or None,
                    calendar=_text(fixed_leg, "./ScheduleData/Rules/Calendar")
                    or _text(float_leg, "./ScheduleData/Rules/Calendar")
                    or None,
                    fixed_payment_convention=_text(fixed_leg, "./PaymentConvention") or "MF",
                    float_payment_convention=_text(float_leg, "./PaymentConvention") or "MF",
                    fixed_schedule_convention=_text(fixed_leg, "./ScheduleData/Rules/Convention") or None,
                    float_schedule_convention=_text(float_leg, "./ScheduleData/Rules/Convention") or None,
                    fixed_term_convention=_text(fixed_leg, "./ScheduleData/Rules/TermConvention") or None,
                    float_term_convention=_text(float_leg, "./ScheduleData/Rules/TermConvention") or None,
                    fixed_schedule_rule=_text(fixed_leg, "./ScheduleData/Rules/Rule") or "Forward",
                    float_schedule_rule=_text(float_leg, "./ScheduleData/Rules/Rule") or "Forward",
                    end_of_month=(
                        (_text(fixed_leg, "./ScheduleData/Rules/EndOfMonth") or "").strip().lower()
                        in {"true", "1", "y", "yes"}
                        or (_text(float_leg, "./ScheduleData/Rules/EndOfMonth") or "").strip().lower()
                        in {"true", "1", "y", "yes"}
                    ),
                    float_index=_text(float_leg, "./FloatingLegData/Index") or "",
                    fixing_days=int(_text(float_leg, "./FloatingLegData/FixingDays") or 2),
                    float_spread=float(_text(float_leg, "./FloatingLegData/Spreads/Spread") or 0.0),
                )
        if fixed_leg is not None or float_leg is not None:
            return GenericProduct(payload={"trade_type": trade_type, "subtype": "SingleLegSwap", "xml": _product_inner_xml(trade)})

    if trade_type == "FxForward":
        bought_ccy = _text(trade, ".//FxForwardData/BoughtCurrency") or "EUR"
        sold_ccy = _text(trade, ".//FxForwardData/SoldCurrency") or "USD"
        bought_amt = float(_text(trade, ".//FxForwardData/BoughtAmount") or 0.0)
        sold_amt = float(_text(trade, ".//FxForwardData/SoldAmount") or 0.0)
        pair = f"{bought_ccy}{sold_ccy}"
        strike = sold_amt / bought_amt if bought_amt else 1.0
        maturity = _maturity_years(None, _text(trade, ".//FxForwardData/ValueDate"))
        return FXForward(
            pair=pair,
            notional=abs(bought_amt),
            strike=strike,
            maturity_years=max(maturity, 1.0),
            buy_base=True,
            value_date=_normalize_date(_text(trade, ".//FxForwardData/ValueDate") or "") if _text(trade, ".//FxForwardData/ValueDate") else None,
        )

    if trade_type == "CapFloor":
        leg = trade.find("./CapFloorData/LegData")
        if leg is not None:
            leg_type = (_text(leg, "./LegType") or "").strip().upper()
            if leg_type in {"CPI", "YY"}:
                ccy = _text(leg, "./Currency") or "EUR"
                notional = float(_text(leg, "./Notionals/Notional") or 0.0)
                start = (
                    _text(leg, "./CPILegData/StartDate")
                    or _text(leg, "./ScheduleData/Rules/StartDate")
                )
                end = (
                    _text(leg, "./ScheduleData/Rules/EndDate")
                    or _text(leg, "./ScheduleData/Dates/Dates/Date")
                )
                maturity = _maturity_years(start, end)
                option_type = (_text(trade, "./CapFloorData/OptionData/Type") or "Cap").strip().title()
                strike = float(
                    _text(leg, "./CPILegData/Rates/Rate")
                    or _text(leg, "./YYLegData/Caps/Cap")
                    or _text(leg, "./YYLegData/Floors/Floor")
                    or 0.0
                )
                return InflationCapFloor(
                    ccy=ccy,
                    notional=notional,
                    maturity_years=max(maturity, 1.0 / 365.25),
                    index=_text(leg, "./CPILegData/Index") or _text(leg, "./YYLegData/Index") or "",
                    inflation_type="CPI" if leg_type == "CPI" else "YY",
                    option_type="Cap" if option_type != "Floor" else "Floor",
                    strike=strike,
                    long_short=(_text(trade, "./CapFloorData/LongShort") or "Long").strip().title(),
                    base_cpi=(
                        float(_text(leg, "./CPILegData/BaseCPI"))
                        if _text(leg, "./CPILegData/BaseCPI")
                        else None
                    ),
                    observation_lag=_text(leg, "./CPILegData/ObservationLag") or _text(leg, "./YYLegData/ObservationLag"),
                    start_date=_normalize_date(start) if start else None,
                    end_date=_normalize_date(end) if end else None,
                    schedule_tenor=_text(leg, "./ScheduleData/Rules/Tenor") or "1Y",
                )

    if trade_type == "FxOption":
        bought_ccy = _text(trade, ".//FxOptionData/BoughtCurrency") or "EUR"
        sold_ccy = _text(trade, ".//FxOptionData/SoldCurrency") or "USD"
        bought_amt = float(_text(trade, ".//FxOptionData/BoughtAmount") or 0.0)
        sold_amt = float(_text(trade, ".//FxOptionData/SoldAmount") or 0.0)
        strike = sold_amt / bought_amt if bought_amt else 1.0
        kind = ((_text(trade, ".//FxOptionData/OptionData/OptionType") or "Call").strip().lower())
        maturity = _maturity_years(None, _text(trade, ".//FxOptionData/OptionData/ExerciseDates/ExerciseDate"))
        from pythonore.domain.dataclasses import EuropeanOption

        return EuropeanOption(
            underlying=f"{bought_ccy}{sold_ccy}",
            kind="call" if kind.startswith("call") else "put",
            strike=strike,
            notional=abs(bought_amt),
            maturity_years=max(maturity, 1.0),
        )

    if trade_type == "EquityOption":
        option_data = trade.find("./EquityOptionData")
        if option_data is not None:
            exercise_date = _text(option_data, "./OptionData/ExerciseDates/ExerciseDate")
            maturity = _maturity_years(None, exercise_date)
            return EquityOption(
                name=_text(option_data, "./Name") or "",
                currency=_text(option_data, "./Currency") or "EUR",
                strike=float(_text(option_data, "./Strike") or 0.0),
                quantity=abs(float(_text(option_data, "./Quantity") or 0.0)),
                option_type=_text(option_data, "./OptionData/OptionType") or "Call",
                long_short=_text(option_data, "./OptionData/LongShort") or "Long",
                style=_text(option_data, "./OptionData/Style") or "European",
                settlement=_text(option_data, "./OptionData/Settlement") or "Cash",
                exercise_date=_normalize_date(exercise_date) if exercise_date else None,
                payoff_at_expiry=(_text(option_data, "./OptionData/PayOffAtExpiry") or "false").strip().lower() in {"true", "1", "y", "yes"},
            )

    if trade_type == "EquityForward":
        forward_data = trade.find("./EquityForwardData")
        if forward_data is not None:
            maturity_date = _text(forward_data, "./Maturity")
            maturity = _maturity_years(None, maturity_date)
            return EquityForward(
                name=_text(forward_data, "./Name") or "",
                currency=_text(forward_data, "./Currency") or "EUR",
                strike=float(_text(forward_data, "./Strike") or 0.0),
                quantity=abs(float(_text(forward_data, "./Quantity") or 0.0)),
                maturity_date=_normalize_date(maturity_date) if maturity_date else None,
                long_short=_text(forward_data, "./LongShort") or "Long",
                strike_currency=_text(forward_data, "./StrikeCurrency") or None,
                maturity_years=max(maturity, 1.0),
            )

    if trade_type == "EquitySwap":
        equity_leg = None
        float_leg = None
        for leg in trade.findall(".//SwapData/LegData"):
            leg_type = (_text(leg, "./LegType") or "").strip().lower()
            if leg_type == "equity":
                equity_leg = leg
            elif leg_type == "floating":
                float_leg = leg
        if equity_leg is not None and float_leg is not None:
            start = _text(equity_leg, "./ScheduleData/Rules/StartDate") or _text(float_leg, "./ScheduleData/Rules/StartDate")
            end = _text(equity_leg, "./ScheduleData/Rules/EndDate") or _text(float_leg, "./ScheduleData/Rules/EndDate")
            maturity = _maturity_years(start, end)
            notional = float(_text(equity_leg, "./Notionals/Notional") or _text(float_leg, "./Notionals/Notional") or 0.0)
            initial_price = float(_text(equity_leg, "./EquityLegData/InitialPrice") or 0.0)
            quantity = abs(notional / initial_price) if initial_price else 0.0
            return EquitySwap(
                name=_text(equity_leg, "./EquityLegData/Name") or "",
                currency=_text(equity_leg, "./Currency") or _text(float_leg, "./Currency") or "EUR",
                notional=abs(notional),
                initial_price=initial_price,
                quantity=quantity,
                start_date=_normalize_date(start) if start else None,
                end_date=_normalize_date(end) if end else None,
                maturity_years=max(maturity, 1.0),
                equity_leg_tenor=_text(equity_leg, "./ScheduleData/Rules/Tenor") or "3M",
                equity_day_counter=_text(equity_leg, "./DayCounter") or "A360",
                equity_payment_convention=_text(equity_leg, "./PaymentConvention") or "F",
                equity_schedule_convention=_text(equity_leg, "./ScheduleData/Rules/Convention") or "F",
                equity_term_convention=_text(equity_leg, "./ScheduleData/Rules/TermConvention") or "F",
                equity_schedule_rule=_text(equity_leg, "./ScheduleData/Rules/Rule") or "Forward",
                float_leg_tenor=_text(float_leg, "./ScheduleData/Rules/Tenor") or "3M",
                float_day_counter=_text(float_leg, "./DayCounter") or "A360",
                float_payment_convention=_text(float_leg, "./PaymentConvention") or "MF",
                float_schedule_convention=_text(float_leg, "./ScheduleData/Rules/Convention") or "MF",
                float_term_convention=_text(float_leg, "./ScheduleData/Rules/TermConvention") or "MF",
                float_schedule_rule=_text(float_leg, "./ScheduleData/Rules/Rule") or "Forward",
                float_index=_text(float_leg, "./FloatingLegData/Index") or "",
                fixing_days=int(_text(float_leg, "./FloatingLegData/FixingDays") or 2),
                float_spread=float(_text(float_leg, "./FloatingLegData/Spreads/Spread") or 0.0),
                equity_payer=(_text(equity_leg, "./Payer") or "true").strip().lower() == "true",
                return_type=_text(equity_leg, "./EquityLegData/ReturnType") or "Price",
                calendar=_text(equity_leg, "./ScheduleData/Rules/Calendar")
                or _text(float_leg, "./ScheduleData/Rules/Calendar")
                or None,
                end_of_month=(
                    (_text(equity_leg, "./ScheduleData/Rules/EndOfMonth") or "").strip().lower() in {"true", "1", "y", "yes"}
                    or (_text(float_leg, "./ScheduleData/Rules/EndOfMonth") or "").strip().lower() in {"true", "1", "y", "yes"}
                ),
            )

    if trade_type == "Swaption":
        style = (_text(trade, ".//SwaptionData/OptionData/Style") or "").strip().lower()
        if style == "bermudan":
            fixed_leg = None
            float_leg = None
            for leg in trade.findall(".//SwaptionData/LegData"):
                leg_type = (_text(leg, "./LegType") or "").strip().lower()
                if leg_type == "fixed":
                    fixed_leg = leg
                elif leg_type == "floating":
                    float_leg = leg
            if fixed_leg is not None and float_leg is not None:
                ccy = _text(fixed_leg, "./Currency") or _text(float_leg, "./Currency") or "EUR"
                notional = float(_text(fixed_leg, "./Notionals/Notional") or _text(float_leg, "./Notionals/Notional") or 0.0)
                fixed_rate = float(_text(fixed_leg, "./FixedLegData/Rates/Rate") or 0.0)
                pay_fixed = (_text(fixed_leg, "./Payer") or "true").lower() == "true"
                start = _text(fixed_leg, "./ScheduleData/Rules/StartDate") or _text(float_leg, "./ScheduleData/Rules/StartDate")
                end = _text(fixed_leg, "./ScheduleData/Rules/EndDate") or _text(float_leg, "./ScheduleData/Rules/EndDate")
                maturity = _maturity_years(start, end)
                exercise_dates = tuple(
                    _normalize_date((n.text or "").strip())
                    for n in trade.findall(".//SwaptionData/OptionData/ExerciseDates/ExerciseDate")
                    if (n.text or "").strip()
                )
                if not exercise_dates:
                    return GenericProduct(payload={"trade_type": trade_type, "style": "Bermudan", "xml": _product_inner_xml(trade)})
                return BermudanSwaption(
                    ccy=ccy,
                    notional=notional,
                    fixed_rate=fixed_rate,
                    maturity_years=max(maturity, 1.0),
                    pay_fixed=pay_fixed,
                    exercise_dates=exercise_dates,
                    settlement=_text(trade, ".//SwaptionData/OptionData/Settlement") or "Physical",
                    option_type=_text(trade, ".//SwaptionData/OptionData/OptionType") or "Call",
                    long_short=_text(trade, ".//SwaptionData/OptionData/LongShort") or "Long",
                    start_date=_normalize_date(start) if start else None,
                    end_date=_normalize_date(end) if end else None,
                    fixed_leg_tenor=_text(fixed_leg, "./ScheduleData/Rules/Tenor") or "1Y",
                    float_leg_tenor=_text(float_leg, "./ScheduleData/Rules/Tenor") or "6M",
                    fixed_day_counter=_text(fixed_leg, "./DayCounter") or None,
                    float_day_counter=_text(float_leg, "./DayCounter") or None,
                    calendar=_text(fixed_leg, "./ScheduleData/Rules/Calendar")
                    or _text(float_leg, "./ScheduleData/Rules/Calendar")
                    or None,
                    fixed_payment_convention=_text(fixed_leg, "./PaymentConvention") or "Following",
                    float_payment_convention=_text(float_leg, "./PaymentConvention") or "ModifiedFollowing",
                    fixed_schedule_convention=_text(fixed_leg, "./ScheduleData/Rules/Convention") or None,
                    float_schedule_convention=_text(float_leg, "./ScheduleData/Rules/Convention") or None,
                    fixed_term_convention=_text(fixed_leg, "./ScheduleData/Rules/TermConvention") or None,
                    float_term_convention=_text(float_leg, "./ScheduleData/Rules/TermConvention") or None,
                    fixed_schedule_rule=_text(fixed_leg, "./ScheduleData/Rules/Rule") or "Forward",
                    float_schedule_rule=_text(float_leg, "./ScheduleData/Rules/Rule") or "Forward",
                    end_of_month=(
                        (_text(fixed_leg, "./ScheduleData/Rules/EndOfMonth") or "").strip().lower()
                        in {"true", "1", "y", "yes"}
                        or (_text(float_leg, "./ScheduleData/Rules/EndOfMonth") or "").strip().lower()
                        in {"true", "1", "y", "yes"}
                    ),
                    float_index=_text(float_leg, "./FloatingLegData/Index") or "",
                    fixing_days=int(_text(float_leg, "./FloatingLegData/FixingDays") or 2),
                    float_spread=float(_text(float_leg, "./FloatingLegData/Spreads/Spread") or 0.0),
                    payoff_at_expiry=(_text(trade, ".//SwaptionData/OptionData/PayOffAtExpiry") or "false").strip().lower() in {"true", "1", "y", "yes"},
                    is_in_arrears=(_text(float_leg, "./FloatingLegData/IsInArrears") or "false").strip().lower() in {"true", "1", "y", "yes"},
                )

    return GenericProduct(payload={"trade_type": trade_type, "xml": _product_inner_xml(trade)})


def _product_inner_xml(trade: ET.Element) -> str:
    parts: List[str] = []
    for child in list(trade):
        if child.tag in {"TradeType", "Envelope"}:
            continue
        parts.append(ET.tostring(child, encoding="unicode"))
    return "\n".join(parts)


def _parse_netting(path: Path) -> NettingConfig:
    if not path.exists():
        return NettingConfig(source_meta=SourceMeta(origin="file", path=str(path)))

    tree = ET.parse(path)
    root = tree.getroot()
    out: Dict[str, NettingSet] = {}
    for n in root.findall("./NettingSet"):
        ns_id = _text(n, "./NettingSetId")
        if not ns_id:
            continue
        csa_flag = (_text(n, "./ActiveCSAFlag") or "false").lower() == "true"
        out[ns_id] = NettingSet(
            netting_set_id=ns_id,
            active_csa=csa_flag,
            bilateral=_text(n, "./CSADetails/Bilateral"),
            csa_currency=_text(n, "./CSADetails/CSACurrency"),
            index=_text(n, "./CSADetails/Index"),
            margin_period_of_risk=_text(n, "./CSADetails/MarginPeriodOfRisk"),
            threshold_pay=_to_float(_text(n, "./CSADetails/ThresholdPay")),
            threshold_receive=_to_float(_text(n, "./CSADetails/ThresholdReceive")),
            mta_pay=_to_float(_text(n, "./CSADetails/MinimumTransferAmountPay")),
            mta_receive=_to_float(_text(n, "./CSADetails/MinimumTransferAmountReceive")),
            independent_amount=IndependentAmount(
                held=_to_float(_text(n, "./CSADetails/IndependentAmount/IndependentAmountHeld")),
                posted=_to_float(_text(n, "./CSADetails/IndependentAmount/IndependentAmountPosted")),
                amount_type=_text(n, "./CSADetails/IndependentAmount/IndependentAmountType"),
            ),
            margining_frequency=MarginingFrequency(
                call_frequency=_text(n, "./CSADetails/MarginingFrequency/CallFrequency"),
                post_frequency=_text(n, "./CSADetails/MarginingFrequency/PostFrequency"),
            ),
            collateral_compounding_spread_receive=_to_float(_text(n, "./CSADetails/CollateralCompoundingSpreadReceive")),
            collateral_compounding_spread_pay=_to_float(_text(n, "./CSADetails/CollateralCompoundingSpreadPay")),
            eligible_collateral_currencies=tuple(
                (c.text or "").strip()
                for c in n.findall("./CSADetails/EligibleCollaterals/Currencies/Currency")
                if (c.text or "").strip()
            ),
            raw_csa_fields=_child_text_map(n.find("./CSADetails"), exclude={
                "Bilateral",
                "CSACurrency",
                "Index",
                "ThresholdPay",
                "ThresholdReceive",
                "MinimumTransferAmountPay",
                "MinimumTransferAmountReceive",
                "IndependentAmount",
                "MarginingFrequency",
                "MarginPeriodOfRisk",
                "CollateralCompoundingSpreadReceive",
                "CollateralCompoundingSpreadPay",
                "EligibleCollaterals",
            }),
        )
    return NettingConfig(netting_sets=out, source_meta=SourceMeta(origin="file", path=str(path)))


def _parse_collateral(path: Path, required: bool) -> CollateralConfig:
    if not path.exists():
        if required:
            raise ValidationError(
                f"Missing collateral balances file: {path}. "
                "Fix: provide collateralbalances.xml at the expected inputPath-relative location, "
                "or load a case that does not require collateral balances."
            )
        return CollateralConfig(source_meta=SourceMeta(origin="file", path=str(path)))

    tree = ET.parse(path)
    root = tree.getroot()
    out: List[CollateralBalance] = []
    for b in root.findall("./CollateralBalance"):
        ns = _text(b, "./NettingSetId")
        ccy = _text(b, "./Currency")
        if not ns or not ccy:
            continue
        out.append(
            CollateralBalance(
                netting_set_id=ns,
                currency=ccy,
                initial_margin=_to_float(_text(b, "./InitialMargin")) or 0.0,
                variation_margin=_to_float(_text(b, "./VariationMargin")) or 0.0,
                initial_margin_type=_text(b, "./InitialMarginType"),
                variation_margin_type=_text(b, "./VariationMarginType"),
                raw_fields=_child_text_map(b, exclude={
                    "NettingSetId",
                    "Currency",
                    "InitialMargin",
                    "VariationMargin",
                    "InitialMarginType",
                    "VariationMarginType",
                }),
            )
        )
    return CollateralConfig(balances=tuple(out), source_meta=SourceMeta(origin="file", path=str(path)))


def _load_known_xml_buffers(ore_path: Path, setup: Dict[str, str], analytics: Dict[str, Dict[str, str]], input_path: str) -> Dict[str, str]:
    # These XML buffers are the minimum set the Python/native runtime knows how
    # to consume directly. Missing files here do not always make the snapshot
    # invalid, but they usually force mapper/runtime fallbacks and therefore
    # weaken parity with a real ORE run.
    fields = {
        "curveconfig.xml": setup.get("curveConfigFile"),
        "conventions.xml": setup.get("conventionsFile"),
        "todaysmarket.xml": setup.get("marketConfigFile"),
        "pricingengine.xml": setup.get("pricingEnginesFile"),
        "portfolio.xml": setup.get("portfolioFile"),
        "counterparty.xml": setup.get("counterpartyFile"),
        "simulation.xml": analytics.get("simulation", {}).get("simulationConfigFile") or "simulation.xml",
        "netting.xml": analytics.get("xva", {}).get("csaFile") or setup.get("nettingSetFile") or "netting.xml",
        "collateralbalances.xml": "collateralbalances.xml",
    }
    buffers: Dict[str, str] = {}
    buffers["ore.xml"] = ore_path.read_text(encoding="utf-8")
    for key, ref in fields.items():
        if not ref:
            continue
        p = _resolve_ref(ref, ore_path, input_path)
        if p.exists():
            buffers[key] = p.read_text(encoding="utf-8")
    output_path = (ore_path.parent.parent / setup.get("outputPath", "Output")).resolve()
    calibration_xml = output_path / "calibration.xml"
    if calibration_xml.exists():
        buffers["calibration.xml"] = calibration_xml.read_text(encoding="utf-8")
    return buffers


def _maturity_years(start: Optional[str], end: Optional[str]) -> float:
    if end is None:
        return 1.0
    end_dt = _to_dt(end)
    if start is None:
        today = datetime.now(timezone.utc).replace(tzinfo=None)
        return max((end_dt - today).days / 365.25, 0.0)
    start_dt = _to_dt(start)
    return max((end_dt - start_dt).days / 365.25, 0.0)


def _to_dt(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValidationError(
        f"Unsupported date format: {s}. Supported formats are YYYY-MM-DD and YYYYMMDD."
    )


def _normalize_date(s: str) -> str:
    return _normalize_date_cached(str(s).strip())


@lru_cache(maxsize=131072)
def _normalize_date_cached(s: str) -> str:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValidationError(
        f"Unsupported date format: {s}. Supported formats are YYYY-MM-DD and YYYYMMDD."
    )


def _text(node: ET.Element, path: str) -> Optional[str]:
    v = node.findtext(path)
    if v is None:
        return None
    return v.strip()


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


def _child_text_map(node: ET.Element | None, exclude: set[str] | None = None) -> Dict[str, str]:
    if node is None:
        return {}
    excluded = exclude or set()
    out: Dict[str, str] = {}
    for child in list(node):
        if child.tag in excluded or list(child):
            continue
        text = (child.text or "").strip()
        if text:
            out[child.tag] = text
    return out


def _resolve_snapshot_mpor(
    *,
    base_currency: str,
    params: Dict[str, str],
    xml_buffers: Dict[str, str],
    netting: NettingConfig,
) -> MporConfig:
    period = ""
    source = "disabled"

    simulation_xml = xml_buffers.get("simulation.xml", "")
    if simulation_xml:
        try:
            root = ET.fromstring(simulation_xml)
            period = (root.findtext("./Parameters/CloseOutLag") or "").strip()
            if period:
                source = "simulation.xml"
        except Exception as exc:
            warnings.warn(
                f"Failed to parse simulation.xml when resolving MPOR CloseOutLag: {exc}. "
                "Falling back to netting set or param overrides.",
                UserWarning,
                stacklevel=2,
            )

    for ns in netting.netting_sets.values():
        if ns.margin_period_of_risk:
            period = str(ns.margin_period_of_risk).strip()
            source = f"netting:{ns.netting_set_id}"
            break

    override_days = str(params.get("python.mpor_days", "")).strip()
    override_period = str(params.get("python.mpor_source_override", "")).strip()
    if override_period:
        period = override_period
        source = "python.mpor_source_override"
    elif override_days:
        try:
            days = max(int(float(override_days)), 0)
            period = f"{days}D"
            source = "python.mpor_days"
        except Exception as exc:
            warnings.warn(
                f"Failed to parse python.mpor_days={override_days!r} as a number: {exc}. "
                "The override will be ignored and MPOR remains disabled.",
                UserWarning,
                stacklevel=2,
            )

    mode = str(params.get("python.mpor_mode", "sticky")).strip().lower()
    if mode and mode != "sticky":
        raise ValidationError(
            f"Unsupported python.mpor_mode '{mode}'. Only 'sticky' is supported."
        )

    years = _period_to_years(period)
    days = _period_to_days(period)
    enabled = years > 0.0 or days > 0
    return MporConfig(
        enabled=enabled,
        mpor_years=years,
        mpor_days=days,
        closeout_lag_period=period,
        sticky=True,
        cashflow_mode="NonePay",
        source=source if enabled else "disabled",
    )


def _period_to_years(period: str) -> float:
    p = str(period).strip().upper()
    if not p:
        return 0.0
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([DWMY])", p)
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "D":
        return n / 365.25
    if unit == "W":
        return (7.0 * n) / 365.25
    if unit == "M":
        return n / 12.0
    return n


def _period_to_days(period: str) -> int:
    p = str(period).strip().upper()
    if not p:
        return 0
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([DWMY])", p)
    if not m:
        return 0
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "D":
        return int(round(n))
    if unit == "W":
        return int(round(5.0 * n))
    if unit == "M":
        return int(round(21.0 * n))
    return int(round(252.0 * n))
