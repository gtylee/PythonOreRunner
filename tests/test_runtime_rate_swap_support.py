import csv
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import pytest

import example_ore_snapshot_usd_all_rates_products as broad_rates_example
import pythonore.compute.irs_xva_utils as irs_utils_mod
import pythonore.compute.overnight_capfloor_shim as overnight_shim_mod
import pythonore.runtime.runtime_impl as runtime_impl_mod
from pythonore.domain.dataclasses import (
    CollateralBalance,
    CollateralConfig,
    GenericProduct,
    NettingConfig,
    NettingSet,
    RuntimeConfig,
    XVAAnalyticConfig,
)
from pythonore.compute.irs_xva_utils import (
    _average_overnight_coupon_fixing_date,
    _infer_index_day_counter,
    _schedule_from_leg,
    _year_fraction,
    compute_realized_float_coupons,
    load_swap_legs_from_portfolio_root,
    swap_npv_from_ore_legs_dual_curve,
)
from pythonore.compute.overnight_capfloor_shim import price_overnight_capfloor_coupon_paths
from pythonore.io.loader import XVALoader
from pythonore.mapping.mapper import map_snapshot
from py_ore_tools import ore_snapshot_cli
from pythonore.runtime.bermudan import _exercise_sign, price_bermudan_from_ore_case
from pythonore.runtime.exceptions import EngineRunError
from pythonore.runtime.runtime import XVAEngine
from pythonore.runtime.runtime_impl import _today_spot_from_quotes


TOOLS_DIR = Path(__file__).resolve().parents[1]
LOCAL_ORE_BINARY = Path("/Users/gordonlee/Documents/Engine/build/App/ore")


def _load_case(case_dir: str, ore_file: str = "ore.xml"):
    snapshot = XVALoader.from_files(str(TOOLS_DIR / case_dir), ore_file=ore_file)
    return replace(snapshot, config=replace(snapshot.config, num_paths=4))


def _run_python(snapshot, run_id: str):
    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        return XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id=run_id,
        )


def _clone_pricing_only_case(case_name: str, *, trade_ids, ore_file: str = "ore.xml"):
    tmp = tempfile.TemporaryDirectory()
    tmp_root = Path(tmp.name)
    case_root = tmp_root / "Examples" / "Legacy" / case_name
    shutil.copytree(TOOLS_DIR / "Examples" / "Legacy" / case_name / "Input", case_root / "Input")
    shutil.copytree(TOOLS_DIR / "Examples" / "Input", tmp_root / "Examples" / "Input")

    portfolio = ET.parse(case_root / "Input" / "portfolio.xml")
    root = portfolio.getroot()
    keep = set(trade_ids)
    for trade in list(root.findall("./Trade")):
        if trade.get("id") not in keep:
            root.remove(trade)
    portfolio.write(case_root / "Input" / "portfolio.xml", encoding="utf-8", xml_declaration=True)

    ore_tree = ET.parse(case_root / "Input" / ore_file)
    ore_root = ore_tree.getroot()
    for analytic in ore_root.findall(".//Analytic"):
        kind = analytic.get("type")
        active = analytic.find("./Parameter[@name='active']")
        if active is not None:
            active.text = "Y" if kind in {"npv", "cashflow", "curves"} else "N"
    analytics_root = ore_root.find("./Analytics")
    if analytics_root is not None and not any(node.get("type") == "cashflow" for node in analytics_root.findall("./Analytic")):
        npv_analytic = next((node for node in analytics_root.findall("./Analytic") if node.get("type") == "npv"), None)
        if npv_analytic is not None:
            cashflow_analytic = ET.fromstring(ET.tostring(npv_analytic, encoding="unicode"))
            cashflow_analytic.set("type", "cashflow")
            active = cashflow_analytic.find("./Parameter[@name='active']")
            if active is not None:
                active.text = "Y"
            analytics_root.append(cashflow_analytic)
    ore_tree.write(case_root / "Input" / ore_file, encoding="utf-8", xml_declaration=True)
    return tmp, case_root


def _clone_sofr_averaging_case(
    case_name: str,
    *,
    gearing: float,
    spread: float,
):
    tmp = tempfile.TemporaryDirectory()
    tmp_root = Path(tmp.name)
    case_root = tmp_root / "Examples" / "Generated" / case_name
    shutil.copytree(TOOLS_DIR / "Examples" / "Generated" / "USD_SOFRAveragingTrue", case_root)

    portfolio = ET.parse(case_root / "Input" / "portfolio.xml")
    root = portfolio.getroot()
    trade = root.find("./Trade[@id='USD_SOFR_OIS_AVG_TRUE']")
    if trade is None:
        raise AssertionError(f"geared SOFR trade not found in {case_root / 'Input' / 'portfolio.xml'}")
    floating_leg = trade.findall("./SwapData/LegData")[1]
    floating_data = floating_leg.find("./FloatingLegData")
    if floating_data is None:
        raise AssertionError(f"floating leg data missing in {case_root / 'Input' / 'portfolio.xml'}")
    gearings = floating_data.find("./Gearings")
    if gearings is None:
        gearings = ET.SubElement(floating_data, "Gearings")
    gearing_node = gearings.find("./Gearing")
    if gearing_node is None:
        gearing_node = ET.SubElement(gearings, "Gearing")
    gearing_node.text = f"{float(gearing):.16g}"
    spread_node = floating_data.find("./Spreads/Spread")
    if spread_node is None:
        spreads = floating_data.find("./Spreads")
        if spreads is None:
            spreads = ET.SubElement(floating_data, "Spreads")
        spread_node = ET.SubElement(spreads, "Spread")
    spread_node.text = f"{float(spread):.16g}"
    portfolio.write(case_root / "Input" / "portfolio.xml", encoding="utf-8", xml_declaration=True)
    return tmp, case_root


def _set_xml_child(parent: ET.Element, tag: str, text: str) -> ET.Element:
    node = parent.find(f"./{tag}")
    if node is None:
        node = ET.SubElement(parent, tag)
    node.text = text
    return node


def _build_xccy_basis_trade(
    *,
    trade_id: str,
    counterparty: str,
    netting_set_id: str,
    asof_date: str,
    start_date: str,
    end_date: str,
    tenor: str,
    calendar: str,
    leg0: dict[str, object],
    leg1: dict[str, object],
) -> ET.Element:
    root = ET.Element("Portfolio")
    trade = ET.SubElement(root, "Trade", id=trade_id)
    ET.SubElement(trade, "TradeType").text = "Swap"
    envelope = ET.SubElement(trade, "Envelope")
    ET.SubElement(envelope, "CounterParty").text = counterparty
    ET.SubElement(envelope, "NettingSetId").text = netting_set_id
    additional_fields = ET.SubElement(envelope, "AdditionalFields")
    ET.SubElement(additional_fields, "valuation_date").text = asof_date
    swap = ET.SubElement(trade, "SwapData")

    for spec in (leg0, leg1):
        leg = ET.SubElement(swap, "LegData")
        ET.SubElement(leg, "LegType").text = "Floating"
        ET.SubElement(leg, "Payer").text = "true" if bool(spec["payer"]) else "false"
        ET.SubElement(leg, "Currency").text = str(spec["currency"])
        notionals = ET.SubElement(leg, "Notionals")
        ET.SubElement(notionals, "Notional").text = f"{float(spec["notional"]):.16g}"
        fx_reset = spec.get("fx_reset")
        if isinstance(fx_reset, dict):
            fx_reset_node = ET.SubElement(notionals, "FXReset")
            ET.SubElement(fx_reset_node, "ForeignCurrency").text = str(fx_reset["foreign_currency"])
            ET.SubElement(fx_reset_node, "ForeignAmount").text = f"{float(fx_reset['foreign_amount']):.16g}"
            ET.SubElement(fx_reset_node, "FXIndex").text = str(fx_reset["fx_index"])
            ET.SubElement(fx_reset_node, "FixingDays").text = str(int(fx_reset.get("fixing_days", 2)))
        exchanges = ET.SubElement(notionals, "Exchanges")
        ET.SubElement(exchanges, "NotionalInitialExchange").text = "true" if bool(spec.get("notional_initial_exchange", True)) else "false"
        ET.SubElement(exchanges, "NotionalFinalExchange").text = "true" if bool(spec.get("notional_final_exchange", True)) else "false"
        ET.SubElement(leg, "DayCounter").text = str(spec.get("day_counter", "ACT/360"))
        ET.SubElement(leg, "PaymentConvention").text = str(spec.get("payment_convention", "ModifiedFollowing"))
        floating = ET.SubElement(leg, "FloatingLegData")
        ET.SubElement(floating, "Index").text = str(spec["index"])
        spreads = ET.SubElement(floating, "Spreads")
        ET.SubElement(spreads, "Spread").text = f"{float(spec.get('spread', 0.0)):.16g}"
        gearings = ET.SubElement(floating, "Gearings")
        ET.SubElement(gearings, "Gearing").text = f"{float(spec.get('gearing', 1.0)):.16g}"
        ET.SubElement(floating, "IsInArrears").text = "true" if bool(spec.get("is_in_arrears", False)) else "false"
        ET.SubElement(floating, "FixingDays").text = str(int(spec.get("fixing_days", 2)))
        schedule = ET.SubElement(leg, "ScheduleData")
        rules = ET.SubElement(schedule, "Rules")
        ET.SubElement(rules, "StartDate").text = start_date
        ET.SubElement(rules, "EndDate").text = end_date
        ET.SubElement(rules, "Tenor").text = str(spec.get("tenor", tenor))
        ET.SubElement(rules, "Calendar").text = str(spec.get("calendar", calendar))
        ET.SubElement(rules, "Convention").text = str(spec.get("schedule_convention", "ModifiedFollowing"))
        ET.SubElement(rules, "TermConvention").text = str(spec.get("term_convention", "ModifiedFollowing"))
        ET.SubElement(rules, "Rule").text = str(spec.get("rule", "Forward"))
        ET.SubElement(rules, "EndOfMonth")
        ET.SubElement(rules, "FirstDate")
        ET.SubElement(rules, "LastDate")
    return root


def _clone_xccy_basis_case(
    source_input_dir: Path,
    case_name: str,
    *,
    trade_id: str,
    counterparty: str,
    netting_set_id: str,
    asof_date: str,
    start_date: str,
    end_date: str,
    tenor: str,
    calendar: str,
    leg0: dict[str, object],
    leg1: dict[str, object],
    ore_file: str = "ore.xml",
):
    tmp = tempfile.TemporaryDirectory()
    tmp_root = Path(tmp.name)
    case_root = tmp_root / "Examples" / "Generated" / case_name
    shutil.copytree(source_input_dir, case_root / "Input")
    shutil.copytree(TOOLS_DIR / "Examples" / "Input", tmp_root / "Examples" / "Input")

    portfolio_root = _build_xccy_basis_trade(
        trade_id=trade_id,
        counterparty=counterparty,
        netting_set_id=netting_set_id,
        asof_date=asof_date,
        start_date=start_date,
        end_date=end_date,
        tenor=tenor,
        calendar=calendar,
        leg0=leg0,
        leg1=leg1,
    )
    ET.ElementTree(portfolio_root).write(case_root / "Input" / "portfolio.xml", encoding="utf-8", xml_declaration=True)

    ore_tree = ET.parse(case_root / "Input" / ore_file)
    ore_root = ore_tree.getroot()
    for analytic in ore_root.findall(".//Analytic"):
        kind = analytic.get("type")
        active = analytic.find("./Parameter[@name='active']")
        if active is not None:
            active.text = "Y" if kind in {"npv", "cashflow"} else "N"
    analytics_root = ore_root.find("./Analytics")
    if analytics_root is not None and not any(node.get("type") == "cashflow" for node in analytics_root.findall("./Analytic")):
        npv_analytic = next((node for node in analytics_root.findall("./Analytic") if node.get("type") == "npv"), None)
        if npv_analytic is not None:
            cashflow_analytic = ET.fromstring(ET.tostring(npv_analytic, encoding="unicode"))
            cashflow_analytic.set("type", "cashflow")
            active = cashflow_analytic.find("./Parameter[@name='active']")
            if active is not None:
                active.text = "Y"
            analytics_root.append(cashflow_analytic)
    ore_tree.write(case_root / "Input" / ore_file, encoding="utf-8", xml_declaration=True)
    return tmp, case_root


def _clone_basis_swap_case(
    case_name: str,
    *,
    trade_id: str,
    counterparty: str,
    netting_set_id: str,
    asof_date: str,
    start_date: str,
    end_date: str,
    tenor: str,
    calendar: str,
    leg0: dict[str, object],
    leg1: dict[str, object],
    ore_file: str = "ore.xml",
):
    tmp = tempfile.TemporaryDirectory()
    tmp_root = Path(tmp.name)
    case_root = tmp_root / "Examples" / "Generated" / case_name
    shutil.copytree(TOOLS_DIR / "Examples" / "Products" / "Input", case_root / "Input")

    portfolio_root = _build_xccy_basis_trade(
        trade_id=trade_id,
        counterparty=counterparty,
        netting_set_id=netting_set_id,
        asof_date=asof_date,
        start_date=start_date,
        end_date=end_date,
        tenor=tenor,
        calendar=calendar,
        leg0=leg0,
        leg1=leg1,
    )
    ET.ElementTree(portfolio_root).write(case_root / "Input" / "portfolio.xml", encoding="utf-8", xml_declaration=True)

    ore_tree = ET.parse(case_root / "Input" / ore_file)
    ore_root = ore_tree.getroot()
    setup = ore_root.find("./Setup")
    if setup is not None:
        for param in setup.findall("./Parameter"):
            if param.attrib.get("name") == "portfolioFile":
                param.text = "portfolio.xml"
            elif param.attrib.get("name") == "asofDate":
                param.text = asof_date
    ore_tree.write(case_root / "Input" / ore_file, encoding="utf-8", xml_declaration=True)
    return tmp, case_root


def _find_report_csv(case_root: Path, filename: str) -> Path:
    matches = sorted(case_root.rglob(filename))
    if not matches:
        raise AssertionError(f"could not find '{filename}' under {case_root}")
    return matches[0]


def _append_fixings_from_marketdata(
    case_root: Path,
    *,
    fixing_id: str,
    quote_id: str,
    start_date: date,
    end_date: date,
) -> None:
    marketdata_csv = case_root / "Input" / "marketdata.csv"
    fixings_csv = case_root / "Input" / "fixings.csv"
    quote_value = None
    with marketdata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = (
                (row.get("datumId") or row.get("#datumId") or row.get("QuoteId") or row.get("quoteId") or row.get("Quote") or "")
                .strip()
            )
            if row_id == quote_id:
                quote_value = float(row.get("datumValue") or row.get("Value") or row.get("value") or row.get("QuoteValue"))
                break
    if quote_value is None:
        with marketdata_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) >= 3 and row[1].strip() == quote_id:
                    quote_value = float(row[2])
                    break
    if quote_value is None:
        raise AssertionError(f"could not find market quote '{quote_id}' in {marketdata_csv}")

    with fixings_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        rows = [["#fixingDate", "fixingId", "fixingValue"]]
    header = rows[0]
    existing = {
        (row[0].strip(), row[1].strip())
        for row in rows[1:]
        if len(row) >= 2 and row[0].strip() and row[1].strip()
    }
    new_rows = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            key = (cur.isoformat(), fixing_id)
            if key not in existing:
                new_rows.append([cur.isoformat(), fixing_id, f"{quote_value:.10f}"])
        cur += timedelta(days=1)
    if not new_rows:
        return
    with fixings_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows[1:])
        writer.writerows(new_rows)


def _backfill_fixings_from_existing_series(
    case_root: Path,
    *,
    fixing_id: str,
    start_date: date,
    end_date: date,
) -> None:
    fixings_csv = case_root / "Input" / "fixings.csv"
    with fixings_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise AssertionError(f"cannot backfill from empty fixings file: {fixings_csv}")
    header = rows[0]
    matching = [row for row in rows[1:] if len(row) >= 3 and row[1].strip() == fixing_id]
    if not matching:
        raise AssertionError(f"could not find fixing series '{fixing_id}' in {fixings_csv}")
    source_value = matching[0][2].strip()
    existing = {
        (row[0].strip(), row[1].strip())
        for row in rows[1:]
        if len(row) >= 2 and row[0].strip() and row[1].strip()
    }
    new_rows = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            key = (cur.isoformat(), fixing_id)
            if key not in existing:
                new_rows.append([cur.isoformat(), fixing_id, source_value])
        cur += timedelta(days=1)
    if not new_rows:
        return
    with fixings_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows[1:])
        writer.writerows(new_rows)


def _read_first_floating_coupon(case_root: Path, trade_id: str) -> float:
    flows_csv = _find_report_csv(case_root, "flows.csv")
    with flows_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row
            for row in reader
            if (row.get("TradeId") or row.get("#TradeId")) == trade_id and row.get("LegNo") == "1"
        ]
    if not rows:
        raise AssertionError(f"no floating rows found for trade '{trade_id}' in {flows_csv}")
    return float(rows[0]["Coupon"])


def _write_poisoned_output_artifacts(case_root: Path):
    output_dir = case_root / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "curves.csv").write_text(
        "Date,EUR-EONIA\n2016-02-05,-99.0\n2017-02-05,-99.0\n",
        encoding="utf-8",
    )
    (output_dir / "flows.csv").write_text(
        "#TradeId,Type,CashflowNo,LegNo,PayDate,FlowType,Amount,Currency,Coupon,Accrual,AccrualStartDate,AccrualEndDate,AccruedAmount,fixingDate,fixingValue,Notional,DiscountFactor,PresentValue,FXRate(Local-Base),PresentValue(Base),BaseCurrency\n"
        "CMS_Spread_Swap,Swap,1,0,1900-01-01,InterestProjected,999999999,EUR,9.99,1.0,1900-01-01,1901-01-01,0.0,1900-01-01,9.99,1.0,1.0,1.0,1.0,1.0,EUR\n",
        encoding="utf-8",
    )
    poisoned_calibration = """<?xml version="1.0" encoding="UTF-8"?>
<Root><InterestRateModels><LGM currency="EUR"><CalibrationType>Bootstrap</CalibrationType>
<Reversion><TimeGrid>1.0</TimeGrid><Value>9.99</Value></Reversion>
<Volatility><TimeGrid>1.0</TimeGrid><Value>9.99</Value></Volatility>
<ParameterTransformation><ShiftHorizon>0</ShiftHorizon><Scaling>1</Scaling></ParameterTransformation>
</LGM></InterestRateModels></Root>"""
    (output_dir / "calibration.xml").write_text(poisoned_calibration, encoding="utf-8")


def test_loader_treats_cms_swap_as_generic_rate_swap():
    snapshot = _load_case("Examples/Legacy/Example_21/Input")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "CMS_Swap")
    assert isinstance(trade.product, GenericProduct)
    assert trade.product.payload.get("subtype") == "GenericRateSwap"


def test_python_runtime_supports_cms_swap_without_fallback():
    snapshot = _load_case("Examples/Legacy/Example_21/Input")
    cms_trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "CMS_Swap")
    snapshot = replace(snapshot, portfolio=replace(snapshot.portfolio, trades=(cms_trade,)))
    result = _run_python(snapshot, "cms-swap-test")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


def test_python_runtime_supports_real_cms_spread_case_without_fallback():
    snapshot = _load_case("Examples/Legacy/Example_25/Input")
    result = _run_python(snapshot, "cms-spread-test")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


def test_schedule_reconstruction_preserves_stubbed_swap_dates():
    root = ET.parse(TOOLS_DIR / "Examples" / "Legacy" / "Example_10" / "Input" / "portfolio.xml").getroot()
    leg = root.find("./Trade[@id='Swap_1']/SwapData/LegData")
    assert leg is not None
    starts, ends, pays = _schedule_from_leg(leg, pay_convention=(leg.findtext("./PaymentConvention") or "F").strip())
    iso_ends = [d.isoformat() for d in ends]
    idx = iso_ends.index("2016-03-07")
    assert starts[idx].isoformat() == "2015-09-07"
    assert ends[idx].isoformat() == "2016-03-07"
    assert pays[idx].isoformat() == "2016-03-07"


def test_schedule_reconstruction_applies_payment_lag_to_xccy_leg():
    root = ET.parse(TOOLS_DIR / "Examples" / "Legacy" / "Example_63" / "Input" / "portfolio.xml").getroot()
    leg = root.find("./Trade[@id='XccySwap']/SwapData/LegData")
    assert leg is not None
    starts, ends, pays = _schedule_from_leg(leg, pay_convention=(leg.findtext("./PaymentConvention") or "F").strip())
    assert starts[0].isoformat() == "2024-01-02"
    assert ends[0].isoformat() == "2024-04-02"
    assert pays[0].isoformat() == "2024-04-04"
    assert pays[1].isoformat() == "2024-07-05"


def test_generic_rate_swap_parses_xccy_fx_reset_metadata():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
    eur_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "EUR")
    fx_reset = usd_leg["fx_reset"]
    assert fx_reset is not None
    assert fx_reset["foreign_currency"] == "EUR"
    assert fx_reset["foreign_amount"] == 95000000.0
    assert fx_reset["fx_index"] == "FX-ECB-EUR-USD"
    assert fx_reset["fixing_calendar"] == "TARGET,US"
    assert usd_leg["notional_initial_exchange"] is True
    assert usd_leg["notional_final_exchange"] is True
    assert list(usd_leg["pay_date"][:2]) == ["2024-04-04", "2024-07-05"]
    assert list(eur_leg["pay_date"][:2]) == ["2024-04-04", "2024-07-04"]
    second_start = adapter._irs_utils._parse_yyyymmdd(str(usd_leg["start_date"][1]))
    fx_fixing_date = adapter._irs_utils._advance_business_days(
        second_start,
        -int(fx_reset["fixing_days"]),
        str(fx_reset["fixing_calendar"]),
    )
    assert fx_fixing_date.isoformat() == "2024-03-27"


def test_example59_cap_usd_sofr_matches_ore_npv_when_replaying_flows_csv():
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_pricing_only_case("Example_59", trade_ids=("Cap_USD_SOFR",))
    try:
        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        with (case_root / "Output" / "npv.csv").open(newline="", encoding="utf-8") as handle:
            ore_row = next(row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == "Cap_USD_SOFR")
        ore_npv_base = float(ore_row["NPV(Base)"])

        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        adapter._ensure_py_lgm_imports()

        pure_snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=8,
                params={
                    **dict(snapshot.config.params),
                    "python.use_flows_csv": "Y",
                    "python.use_ore_output_curves": "Y",
                    "python.use_ore_flow_amounts_t0": "N",
                },
            ),
        )
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            pure_result = adapter.run(pure_snapshot, mapped=map_snapshot(pure_snapshot), run_id="example59-cap-pure-parity")

        assert math.isclose(float(pure_result.pv_total), ore_npv_base, rel_tol=0.0, abs_tol=4.0)

        replay_snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=8,
                params={
                    **dict(snapshot.config.params),
                    "python.use_flows_csv": "Y",
                    "python.use_ore_output_curves": "Y",
                    "python.use_ore_flow_amounts_t0": "Y",
                },
            ),
        )
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            result = adapter.run(replay_snapshot, mapped=map_snapshot(replay_snapshot), run_id="example59-cap-ore-flows-parity")

        assert math.isclose(float(result.pv_total), ore_npv_base, rel_tol=1.0e-12, abs_tol=1.0e-9)
        assert math.isclose(
            float(result.cubes["npv_cube"].payload["Cap_USD_SOFR"]["npv_mean"][0]),
            ore_npv_base,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
    finally:
        tmp.cleanup()


@pytest.mark.parametrize(
    ("local_cap_floor", "naked_option", "abs_tol"),
    (
        (False, False, 700.0),
        (True, False, 1000.0),
        (True, True, 35.0),
    ),
)
def test_example59_overnight_floor_local_and_global_coupon_parity(local_cap_floor, naked_option, abs_tol):
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_pricing_only_case("Example_59", trade_ids=("Cap_USD_SOFR",))
    try:
        portfolio_path = case_root / "Input" / "portfolio.xml"
        portfolio = ET.parse(portfolio_path)
        floating_data = portfolio.getroot().find("./Trade[@id='Cap_USD_SOFR']/SwapData/LegData/FloatingLegData")
        assert floating_data is not None
        _set_xml_child(floating_data, "NakedOption", "true" if naked_option else "false")
        _set_xml_child(floating_data, "LocalCapFloor", "true" if local_cap_floor else "false")
        portfolio.write(portfolio_path, encoding="utf-8", xml_declaration=True)

        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        with (case_root / "Output" / "npv.csv").open(newline="", encoding="utf-8") as handle:
            ore_row = next(row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == "Cap_USD_SOFR")
        ore_npv_base = float(ore_row["NPV(Base)"])

        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=8,
                params={
                    **dict(snapshot.config.params),
                    "python.use_flows_csv": "Y",
                    "python.use_ore_output_curves": "Y",
                    "python.use_ore_flow_amounts_t0": "N",
                    "python.progress": "N",
                },
            ),
        )
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
                snapshot,
                mapped=map_snapshot(snapshot),
                run_id=f"example59-overnight-floor-local-{local_cap_floor}",
            )

        assert math.isclose(float(result.pv_total), ore_npv_base, rel_tol=0.0, abs_tol=abs_tol)
    finally:
        tmp.cleanup()


def test_capfloor_surface_rate_computation_period_uses_curve_config_not_quote_tenor():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_59" / "Input"), ore_file="ore.xml")
    overnight_shim_mod._CAPFLOOR_SURFACE_PERIOD_CACHE.clear()

    assert overnight_shim_mod.capfloor_surface_rate_computation_period(snapshot, ccy="USD") == "3M"


def test_example63_xccy_notional_rows_follow_fx_reset_ladder():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
    root = ET.parse(TOOLS_DIR / "Examples" / "Legacy" / "Example_63" / "Input" / "portfolio.xml").getroot()
    leg = root.find("./Trade[@id='XccySwap']/SwapData/LegData")
    assert leg is not None
    _, ends, _ = _schedule_from_leg(leg, pay_convention=(leg.findtext("./PaymentConvention") or "F").strip())
    flows_csv = TOOLS_DIR / "Examples" / "Legacy" / "Example_63" / "Output" / "valid_xccy" / "flows.csv"
    with flows_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if (row.get("TradeId") or row.get("#TradeId")) == "XccySwap" and row["LegNo"] in {"2", "3"}]
    usd_rows = [row for row in rows if row["LegNo"] == "2"]
    eur_rows = [row for row in rows if row["LegNo"] == "3"]
    assert len(usd_rows) == 15
    assert len(eur_rows) == 1

    boundary_dates = [d.isoformat() for d in ends]
    assert [usd_rows[i]["PayDate"] for i in range(0, len(usd_rows), 2)] == boundary_dates
    assert [usd_rows[i]["PayDate"] for i in range(1, len(usd_rows), 2)] == boundary_dates[:-1]

    assert float(usd_rows[0]["Amount"]) == -usd_leg["notional"][0]
    assert float(eur_rows[0]["Amount"]) == 95000000.0
    assert eur_rows[0]["PayDate"] == ends[-1].isoformat()

    for row in usd_rows[1:]:
        fixing_value = row.get("fixingValue", "").strip()
        if fixing_value and fixing_value not in {"#N/A", "N/A"}:
            expected = 95000000.0 * float(fixing_value)
            assert math.isclose(abs(float(row["Amount"])), expected, rel_tol=1.0e-10, abs_tol=5.0e-4)


def test_overnight_static_state_uses_exact_helper(monkeypatch):
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_59" / "Input"), ore_file="ore.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "Cap_USD_SOFR")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, num_paths=4),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    inputs = adapter._extract_inputs(snapshot, mapped)
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    leg = next(leg for leg in state["rate_legs"] if leg.get("overnight_indexed", False))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("exact helper called")

    monkeypatch.setattr("pythonore.compute.overnight_capfloor_shim._quant_ext_overnight_coupon_rate", _boom)
    with pytest.raises(RuntimeError, match="exact helper called"):
        price_overnight_capfloor_coupon_paths(
            adapter,
            inputs=inputs,
            leg=leg,
            ccy=str(leg["ccy"]),
            t=0.0,
            x_t=np.zeros(1, dtype=float),
            snapshot=snapshot,
        )


def test_overnight_global_coupon_rate_handles_missing_floor():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_59" / "Input"), ore_file="ore.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "Cap_USD_SOFR")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, num_paths=4),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    inputs = adapter._extract_inputs(snapshot, mapped)
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    leg = next(leg for leg in state["rate_legs"] if leg.get("overnight_indexed", False))
    surface_period = overnight_shim_mod.capfloor_surface_rate_computation_period(snapshot, ccy=str(leg["ccy"])) or ""
    static_state = overnight_shim_mod._build_overnight_static_state(
        adapter,
        inputs,
        snapshot,
        ccy=str(leg["ccy"]),
        index_name=str(leg["index_name"]),
        surface_period=surface_period,
        target_period=str(leg.get("schedule_tenor", "")).strip() or "3M",
        leg=leg,
    )
    assert static_state is not None
    pricer = static_state["pricer"]
    raw_rate = np.full(1, 0.01, dtype=float)
    out = pricer.global_coupon_rate(
        raw_rate=raw_rate,
        cap=0.02,
        floor=None,
        naked_option=bool(leg.get("naked_option", False)),
        expiry_time=float(static_state["expiry_times"][0]),
        fixing_date=static_state["fixing_dates"][0],
        fixing_dates=list(static_state["fixing_dates"]),
        asof_date=static_state["asof_date"],
    )
    assert out.shape == raw_rate.shape
    assert np.all(np.isfinite(out))


def test_example63_xccy_trade_runs_natively_without_swig():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.use_ore_output_curves": "Y"},
        ),
    )
    result = _run_python(snapshot, "xccy-native-only")
    assert math.isfinite(float(result.pv_total))


def test_example63_xccy_trade_ignores_flows_csv_for_native_runtime():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={
                **dict(snapshot.config.params),
                "python.use_ore_output_curves": "Y",
                "python.use_flows_csv": "Y",
            },
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        result = adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id="xccy-ore-flows-parity",
        )

    assert math.isfinite(float(result.pv_total))
    specs, unsupported, _ = adapter._classify_portfolio_trades(snapshot, map_snapshot(snapshot))
    assert unsupported == []
    assert len(specs) == 1
    assert specs[0].kind == "RateSwap"
    assert specs[0].legs is not None
    assert specs[0].legs.get("source") != "flows.csv"
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []


def test_example63_overnight_xccy_first_coupon_uses_daily_compounding():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.use_ore_output_curves": "Y"},
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    inputs = adapter._extract_inputs(snapshot, mapped)
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
    quoted_coupon = np.asarray(usd_leg["quoted_coupon"], dtype=float)
    fixed_mask = np.asarray(usd_leg["is_historically_fixed"], dtype=bool)
    assert quoted_coupon[0] == 0.0
    assert not bool(fixed_mask[0])

    p = inputs.lgm_params
    model = adapter._lgm_mod.LGM1F(
        adapter._lgm_mod.LGMParams(
            alpha_times=tuple(p["alpha_times"]),
            alpha_values=tuple(p["alpha_values"]),
            kappa_times=tuple(p["kappa_times"]),
            kappa_values=tuple(p["kappa_values"]),
            shift=p["shift"],
            scaling=p["scaling"],
        )
    )
    coupons = adapter._rate_leg_coupon_paths(
        model,
        usd_leg,
        "USD",
        inputs,
        0.0,
        np.zeros(1, dtype=float),
        snapshot=snapshot,
    )
    assert coupons.shape[0] == quoted_coupon.size
    assert coupons[0, 0] > 0.01


def test_example79_usd_jpy_xccy_keeps_leg_currency_and_base_currency_separate():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_79" / "Input"), ore_file="ore.xml")
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap_USD_JPY")
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
    jpy_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "JPY")
    assert usd_leg["fx_reset"] is not None
    assert usd_leg["fx_reset"]["foreign_currency"] == "JPY"
    assert usd_leg["fx_reset"]["foreign_amount"] == 10887500000.0
    assert math.isclose(float(usd_leg["notional"][0]), 100000000.0, rel_tol=0.0, abs_tol=1.0e-6)
    assert math.isclose(float(jpy_leg["notional"][0]), 10887500000.0, rel_tol=0.0, abs_tol=1.0e-6)

    result = _run_python(snapshot, "usd-jpy-xccy-example")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


def test_generated_usd_jpy_overnight_xccy_uses_tonar_fx_reset_without_fx_blowup():
    tmp, case_root = _clone_xccy_basis_case(
        TOOLS_DIR / "Examples" / "Products" / "Input",
        "USD_SOFR_TONAR_XCCY_NO_ORE",
        trade_id="SOFR_TONAR_XCCY_NO_ORE",
        counterparty="CPTY_A",
        netting_set_id="CPTY_A",
        asof_date="2025-02-10",
        start_date="2025-04-10",
        end_date="2026-04-10",
        tenor="3M",
        calendar="US,JP",
        leg0={
            "payer": True,
            "currency": "USD",
            "notional": 100000000.0,
            "index": "USD-SOFR",
            "spread": 0.0,
        },
        leg1={
            "payer": False,
            "currency": "JPY",
            "notional": 15222818282.0,
            "index": "JPY-TONAR",
            "spread": 0.0,
            "fx_reset": {
                "foreign_currency": "USD",
                "foreign_amount": 100000000.0,
                "fx_index": "FX-BOE-USD-JPY",
                "fixing_days": 2,
            },
        },
        ore_file="ore.xml",
    )
    try:
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "SOFR_TONAR_XCCY_NO_ORE")
        adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        adapter._ensure_py_lgm_imports()
        mapped = map_snapshot(snapshot)
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            inputs = adapter._extract_inputs(snapshot, mapped)
        state = adapter._build_generic_rate_swap_legs(trade, snapshot)
        assert state is not None
        usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
        jpy_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "JPY")
        assert usd_leg["overnight_indexed"] is True
        assert jpy_leg["overnight_indexed"] is True
        assert usd_leg["fx_reset"] is None
        assert jpy_leg["fx_reset"] is not None
        assert jpy_leg["fx_reset"]["foreign_currency"] == "USD"
        assert jpy_leg["fx_reset"]["foreign_amount"] == 100000000.0

        usd_jpy = _today_spot_from_quotes("USDJPY", inputs)
        assert usd_jpy > 100.0
        p = inputs.lgm_params
        model = adapter._lgm_mod.LGM1F(
            adapter._lgm_mod.LGMParams(
                alpha_times=tuple(p["alpha_times"]),
                alpha_values=tuple(p["alpha_values"]),
                kappa_times=tuple(p["kappa_times"]),
                kappa_values=tuple(p["kappa_values"]),
                shift=p["shift"],
                scaling=p["scaling"],
            )
        )
        jpy_coupons = adapter._rate_leg_coupon_paths(
            model,
            jpy_leg,
            "JPY",
            inputs,
            0.0,
            np.zeros(1, dtype=float),
            snapshot=snapshot,
        )
        assert float(np.max(jpy_coupons[:, 0])) < 0.02
        one_usd = adapter._convert_amount_to_reporting_ccy(
            np.asarray([usd_jpy], dtype=float),
            local_ccy="JPY",
            report_ccy="USD",
            inputs=inputs,
            shared_fx_sim=None,
            time_index=0,
        )
        assert np.allclose(one_usd, np.asarray([1.0]), rtol=0.0, atol=1.0e-10)

        result = _run_python(snapshot, "usd-jpy-overnight-xccy-no-ore")
        coverage = result.metadata["coverage"]
        assert coverage["fallback_trades"] == 0
        assert coverage["unsupported"] == []
        assert math.isfinite(float(result.pv_total))
    finally:
        tmp.cleanup()


def test_generated_usd_jpy_overnight_xccy_zero_threshold_csa_mpor_has_finite_closeout():
    tmp, case_root = _clone_xccy_basis_case(
        TOOLS_DIR / "Examples" / "Products" / "Input",
        "USD_SOFR_TONAR_XCCY_MPOR_NO_ORE",
        trade_id="SOFR_TONAR_XCCY_MPOR_NO_ORE",
        counterparty="CPTY_A",
        netting_set_id="CPTY_A",
        asof_date="2025-02-10",
        start_date="2025-04-10",
        end_date="2026-04-10",
        tenor="3M",
        calendar="US,JP",
        leg0={
            "payer": True,
            "currency": "USD",
            "notional": 100000000.0,
            "index": "USD-SOFR",
            "spread": 0.0,
        },
        leg1={
            "payer": False,
            "currency": "JPY",
            "notional": 15222818282.0,
            "index": "JPY-TONAR",
            "spread": 0.0,
            "fx_reset": {
                "foreign_currency": "USD",
                "foreign_amount": 100000000.0,
                "fx_index": "FX-BOE-USD-JPY",
                "fixing_days": 2,
            },
        },
        ore_file="ore.xml",
    )
    try:
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(
            snapshot,
            netting=NettingConfig(
                netting_sets={
                    "CPTY_A": NettingSet(
                        netting_set_id="CPTY_A",
                        counterparty="CPTY_A",
                        active_csa=True,
                        csa_currency="USD",
                        threshold_pay=0.0,
                        threshold_receive=0.0,
                        mta_pay=0.0,
                        mta_receive=0.0,
                    )
                }
            ),
            collateral=CollateralConfig(
                balances=(CollateralBalance(netting_set_id="CPTY_A", currency="USD"),)
            ),
            config=replace(
                snapshot.config,
                analytics=("CVA",),
                num_paths=4,
                params={**snapshot.config.params, "python.mpor_source_override": "1Y"},
            ),
        )
        result = _run_python(snapshot, "usd-jpy-overnight-xccy-zero-threshold-mpor")
        profile = result.exposure_profiles_by_netting_set["CPTY_A"]
        coverage = result.metadata["coverage"]

        assert coverage["fallback_trades"] == 0
        assert result.metadata["mpor_enabled"] is True
        assert result.metadata["mpor_source"] == "python.mpor_source_override"
        assert max(abs(float(x)) for x in profile["valuation_epe"]) <= 1.0e-8
        assert max(float(x) for x in profile["closeout_epe"]) > 0.0
        assert max(float(x) for x in profile["closeout_epe"]) < 1.0e9
        assert all(math.isfinite(float(x)) for x in profile["closeout_epe"])
        assert all(math.isfinite(float(x)) for x in profile["closeout_ene"])
    finally:
        tmp.cleanup()


def test_sofr_euribor_xccy_fx_reset_coupon_ladder_matches_ore_and_python_runtime():
    if not LOCAL_ORE_BINARY.exists():
        return
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={
                **dict(snapshot.config.params),
                "python.use_ore_output_curves": "Y",
                "python.use_ore_flow_amounts_t0": "Y",
            },
        ),
    )

    npv_csv = TOOLS_DIR / "Examples" / "Legacy" / "Example_63" / "Output" / "valid_xccy" / "npv.csv"
    with npv_csv.open(newline="", encoding="utf-8") as handle:
        ore_row = next(row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == "XccySwap")
    ore_npv = float(ore_row.get("NPV(Base)") or ore_row.get("NPV Base") or ore_row.get("NPV") or ore_row.get("npv"))

    flows_csv = TOOLS_DIR / "Examples" / "Legacy" / "Example_63" / "Output" / "valid_xccy" / "flows.csv"
    with flows_csv.open(newline="", encoding="utf-8") as handle:
        flows = [row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == "XccySwap"]
    usd_interest_rows = [row for row in flows if row.get("Currency") == "USD" and row.get("LegNo") == "0" and "Interest" in row.get("FlowType", "")]
    eur_interest_rows = [row for row in flows if row.get("Currency") == "EUR" and row.get("LegNo") == "1" and "Interest" in row.get("FlowType", "")]
    assert usd_interest_rows
    assert eur_interest_rows

    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
    eur_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "EUR")
    assert usd_leg["fx_reset"] is not None
    assert usd_leg["fx_reset"]["foreign_currency"] == "EUR"
    assert usd_leg["fx_reset"]["foreign_amount"] == 95000000.0
    assert eur_leg["fx_reset"] is None
    ore_usd_notionals = np.asarray([float(row["Notional"]) for row in usd_interest_rows], dtype=float)
    ore_eur_notionals = np.asarray([float(row["Notional"]) for row in eur_interest_rows], dtype=float)
    assert not np.allclose(ore_usd_notionals, ore_usd_notionals[0])
    assert np.allclose(ore_eur_notionals, 95000000.0)
    mapped = map_snapshot(snapshot)
    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        inputs = adapter._extract_inputs(snapshot, mapped)
    assert math.isclose(_today_spot_from_quotes("USDEUR", inputs), 0.9066219663, rel_tol=0.0, abs_tol=1.0e-10)

    result = _run_python(snapshot, "sofr-euribor-xccy")
    assert math.isfinite(float(result.pv_total))
    assert math.isclose(float(result.pv_total), ore_npv, rel_tol=1.0e-12, abs_tol=1.0e-8)


def test_sofr_tonar_xccy_fx_reset_coupon_ladder_matches_ore_and_python_runtime():
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_xccy_basis_case(
        TOOLS_DIR / "Examples" / "Products" / "Input",
        "USD_SOFR_TONAR_XCCY",
        trade_id="SOFR_TONAR_XCCY",
        counterparty="CPTY_A",
        netting_set_id="CPTY_A",
        asof_date="2025-02-10",
        start_date="2024-01-02",
        end_date="2026-01-02",
        tenor="3M",
        calendar="US,JP",
        leg0={
            "payer": True,
            "currency": "USD",
            "notional": 100000000.0,
            "index": "USD-SOFR",
            "spread": 0.0,
        },
        leg1={
            "payer": False,
            "currency": "JPY",
            "notional": 15222818282.0,
            "index": "JPY-TONAR",
            "spread": 0.0,
            "fx_reset": {
                "foreign_currency": "USD",
                "foreign_amount": 100000000.0,
                "fx_index": "FX-BOE-USD-JPY",
                "fixing_days": 2,
            },
        },
        ore_file="ore.xml",
    )
    _append_fixings_from_marketdata(
        case_root,
        fixing_id="FX-BOE-USD-JPY",
        quote_id="FX/RATE/USD/JPY",
        start_date=date(2023, 12, 27),
        end_date=date(2025, 2, 10),
    )
    _backfill_fixings_from_existing_series(
        case_root,
        fixing_id="JPY-TONAR",
        start_date=date(2023, 12, 27),
        end_date=date(2024, 10, 9),
    )
    try:
        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )

        npv_csv = _find_report_csv(case_root, "npv.csv")
        with npv_csv.open(newline="", encoding="utf-8") as handle:
            ore_row = next(row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == "SOFR_TONAR_XCCY")
        ore_npv = float(ore_row.get("NPV") or ore_row.get("npv") or ore_row.get("NPV(Base)") or ore_row.get("NPV Base"))

        flows_csv = _find_report_csv(case_root, "flows.csv")
        with flows_csv.open(newline="", encoding="utf-8") as handle:
            flows = [row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == "SOFR_TONAR_XCCY"]
        jpy_interest_rows = [row for row in flows if row.get("Currency") == "JPY" and row.get("LegNo") == "1" and "Interest" in row.get("FlowType", "")]
        usd_interest_rows = [row for row in flows if row.get("Currency") == "USD" and row.get("LegNo") == "0" and "Interest" in row.get("FlowType", "")]
        assert jpy_interest_rows
        assert usd_interest_rows

        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                params={**dict(snapshot.config.params), "python.use_ore_flow_amounts_t0": "Y"},
            ),
        )
        trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "SOFR_TONAR_XCCY")
        adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        adapter._ensure_py_lgm_imports()
        state = adapter._build_generic_rate_swap_legs(trade, snapshot)
        assert state is not None
        usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
        jpy_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "JPY")
        assert usd_leg["fx_reset"] is None
        assert jpy_leg["fx_reset"] is not None
        assert jpy_leg["fx_reset"]["foreign_currency"] == "USD"
        assert jpy_leg["fx_reset"]["foreign_amount"] == 100000000.0
        ore_jpy_notionals = np.asarray([float(row["Notional"]) for row in jpy_interest_rows], dtype=float)
        ore_usd_notionals = np.asarray([float(row["Notional"]) for row in usd_interest_rows], dtype=float)
        assert not np.allclose(ore_jpy_notionals, ore_jpy_notionals[0])
        assert np.allclose(ore_usd_notionals, 100000000.0)

        result = _run_python(snapshot, "sofr-tonar-xccy")
        assert math.isfinite(float(result.pv_total))
        assert math.isclose(float(result.pv_total), ore_npv, rel_tol=1.0e-12, abs_tol=1.0e-8)
    finally:
        tmp.cleanup()


def test_generic_rate_swap_missing_fx_fixing_falls_back_to_spot():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.use_ore_output_curves": "Y"},
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    with patch.object(adapter, "_supports_torch_rate_swap", return_value=False), patch.object(
        adapter, "_fixings_lookup", return_value={}
    ), patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        result = adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id="xccy-missing-fixings",
        )
    assert math.isfinite(float(result.pv_total))


def test_default_python_path_does_not_use_ore_flow_t0_anchors(monkeypatch):
    snapshot = XVALoader.from_files(
        str(TOOLS_DIR / "Examples" / "Generated" / "USD_AllRatesProductsSnapshot_20PerType" / "Input"),
        ore_file="ore.xml",
    )
    keep = {"IRS_USD_0001", "BASIS_USD_LIB3M_LIB6M_0001"}
    trades = tuple(t for t in snapshot.portfolio.trades if t.trade_id in keep)
    assert {t.trade_id for t in trades} == keep
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=trades),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.use_ore_flow_amounts_t0": "N"},
        ),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("pure Python path must not read ORE flow/NPV t0 anchors")

    monkeypatch.setattr(runtime_impl_mod, "_load_ore_t0_npv_for_reporting_ccy", fail_if_called)
    monkeypatch.setattr(runtime_impl_mod, "_price_irs_t0_from_flow_amounts", fail_if_called)

    result = _run_python(snapshot, "pure-python-no-ore-flow-anchor")
    assert result.metadata["coverage"]["fallback_trades"] == 0
    assert result.metadata["coverage"]["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


def test_geared_usd_sofr_ois_matches_ore_and_scales_coupon_forward_only():
    if not LOCAL_ORE_BINARY.exists():
        return
    spread = 0.00125
    control_tmp, control_case = _clone_sofr_averaging_case(
        "USD_SOFRAveragingControl",
        gearing=1.0,
        spread=spread,
    )
    geared_tmp, geared_case = _clone_sofr_averaging_case(
        "USD_SOFRAveragingGeared",
        gearing=1.75,
        spread=spread,
    )
    trade_id = "USD_SOFR_OIS_AVG_TRUE"
    try:
        for case_root in (control_case, geared_case):
            subprocess.run(
                [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
                cwd=case_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=240,
            )

        control_payload = ore_snapshot_cli._compute_price_only_case(
            control_case / "Input" / "ore.xml",
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        geared_payload = ore_snapshot_cli._compute_price_only_case(
            geared_case / "Input" / "ore.xml",
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )

        control_root = ET.parse(control_case / "Input" / "portfolio.xml").getroot()
        geared_root = ET.parse(geared_case / "Input" / "portfolio.xml").getroot()
        control_legs = load_swap_legs_from_portfolio_root(control_root, trade_id, "2025-02-10")
        geared_legs = load_swap_legs_from_portfolio_root(geared_root, trade_id, "2025-02-10")
        assert np.allclose(np.asarray(control_legs["float_gearing"], dtype=float), 1.0)
        assert np.allclose(np.asarray(geared_legs["float_gearing"], dtype=float), 1.75)

        control_coupon = _read_first_floating_coupon(control_case, trade_id)
        geared_coupon = _read_first_floating_coupon(geared_case, trade_id)
        assert math.isclose(
            geared_coupon,
            1.75 * (control_coupon - spread) + spread,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )

        assert math.isfinite(float(control_payload["pricing"]["ore_t0_npv"]))
        assert math.isfinite(float(geared_payload["pricing"]["ore_t0_npv"]))
        assert math.isfinite(float(control_payload["pricing"]["py_t0_npv"]))
        assert math.isfinite(float(geared_payload["pricing"]["py_t0_npv"]))
        assert control_payload["pricing"]["t0_npv_abs_diff"] < 0.5
        assert geared_payload["pricing"]["t0_npv_abs_diff"] < 0.5
    finally:
        control_tmp.cleanup()
        geared_tmp.cleanup()


def test_averaged_sofr_floor_price_only_matches_ore_cashflow_parity():
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_sofr_averaging_case(
        "USD_SOFRAveragingFloored",
        gearing=1.0,
        spread=0.0,
    )
    try:
        portfolio_path = case_root / "Input" / "portfolio.xml"
        portfolio = ET.parse(portfolio_path)
        floating_data = portfolio.getroot().find("./Trade[@id='USD_SOFR_OIS_AVG_TRUE']/SwapData/LegData[2]/FloatingLegData")
        assert floating_data is not None
        floors = floating_data.find("./Floors")
        if floors is None:
            floors = ET.SubElement(floating_data, "Floors")
        _set_xml_child(floors, "Floor", "0.052")
        _set_xml_child(floating_data, "NakedOption", "false")
        _set_xml_child(floating_data, "LocalCapFloor", "false")
        portfolio.write(portfolio_path, encoding="utf-8", xml_declaration=True)

        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        payload = ore_snapshot_cli._compute_price_only_case(
            case_root / "Input" / "ore.xml",
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )

        assert math.isfinite(float(payload["pricing"]["ore_t0_npv"]))
        assert math.isfinite(float(payload["pricing"]["py_t0_npv"]))
        assert payload["pricing"]["leg_source"] == "flows"
        assert payload["pricing"]["t0_npv_abs_diff"] < 0.05
    finally:
        tmp.cleanup()


def test_bermudan_swaption_exercise_sign_follows_option_type_not_fixed_leg_orientation():
    snapshot = XVALoader.from_files(
        str(TOOLS_DIR / "Examples" / "Generated" / "USD_AllRatesProductsSnapshot_134PerType" / "Input"),
        ore_file="ore.xml",
    )
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "BERMUDAN_SWAPTION_USD_0001")
    assert trade.product.option_type == "Call"
    assert trade.product.pay_fixed is False
    assert _exercise_sign(trade.product) == 1.0


def test_generated_all_rates_smoke_includes_bermudan_variants(tmp_path):
    case_root = tmp_path / "USD_AllRatesProductsSnapshot_BermudanVariants"
    broad_rates_example._write_files(case_root, count_per_type=1)

    snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    trades = {t.trade_id: t for t in snapshot.portfolio.trades}
    expected_ids = {
        "BERMUDAN_SWAPTION_USD_0001",
        "BERMUDAN_SWAPTION_USD_RECEIVER_PUT_0001",
        "BERMUDAN_SWAPTION_USD_SHORT_CALL_0001",
        "BERMUDAN_SWAPTION_USD_PAST_EXERCISE_0001",
    }
    assert expected_ids <= set(trades)

    base = trades["BERMUDAN_SWAPTION_USD_0001"].product
    receiver_put = trades["BERMUDAN_SWAPTION_USD_RECEIVER_PUT_0001"].product
    short_call = trades["BERMUDAN_SWAPTION_USD_SHORT_CALL_0001"].product
    past_exercise = trades["BERMUDAN_SWAPTION_USD_PAST_EXERCISE_0001"].product

    assert base.option_type == "Call"
    assert base.long_short == "Long"
    assert base.pay_fixed is False
    assert _exercise_sign(base) == 1.0

    assert receiver_put.option_type == "Put"
    assert receiver_put.long_short == "Long"
    assert receiver_put.pay_fixed is True
    assert _exercise_sign(receiver_put) == -1.0

    assert short_call.option_type == "Call"
    assert short_call.long_short == "Short"
    assert short_call.pay_fixed is False
    assert _exercise_sign(short_call) == -1.0

    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    state = adapter._build_bermudan_swaption_state(
        trades["BERMUDAN_SWAPTION_USD_PAST_EXERCISE_0001"],
        snapshot,
        mapped,
    )
    assert state is not None
    exercise_times = np.asarray(state["definition"].exercise_times, dtype=float)
    assert len(past_exercise.exercise_dates) == 4
    assert exercise_times.size == 3
    assert np.all(exercise_times > 0.0)


def test_generated_all_rates_smoke_includes_sifma_tonar_xccy_variants(tmp_path):
    case_root = tmp_path / "USD_AllRatesProductsSnapshot_Variants"
    broad_rates_example._write_files(case_root, count_per_type=1)

    snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    trade_ids = {t.trade_id for t in snapshot.portfolio.trades}
    assert "CAP_USD_SOFR3M_0001" in trade_ids
    assert "FLOOR_USD_LIB3M_0001" in trade_ids
    assert "FLOOR_USD_SOFR3M_FORWARD_0001" in trade_ids
    assert "CAP_USD_LIB3M_SHORT_GEARED_0001" in trade_ids
    assert "BASIS_USD_LIB3M_SIFMA_0001" in trade_ids
    assert "BASIS_USD_SOFR3M_SIFMA_0001" in trade_ids
    assert "XCCY_USD_SOFR_JPY_TONAR_0001" in trade_ids

    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()

    forward_floor = next(t for t in snapshot.portfolio.trades if t.trade_id == "FLOOR_USD_SOFR3M_FORWARD_0001")
    forward_floor_state = adapter._build_generic_capfloor_state(forward_floor, snapshot)
    assert forward_floor_state is not None
    assert forward_floor_state["definition"].option_type == "floor"
    assert np.allclose(forward_floor_state["definition"].gearing, 1.25)
    assert np.allclose(forward_floor_state["definition"].spread, 0.0010)

    short_cap = next(t for t in snapshot.portfolio.trades if t.trade_id == "CAP_USD_LIB3M_SHORT_GEARED_0001")
    short_cap_state = adapter._build_generic_capfloor_state(short_cap, snapshot)
    assert short_cap_state is not None
    assert short_cap_state["definition"].option_type == "cap"
    assert short_cap_state["definition"].position == -1.0
    assert np.allclose(short_cap_state["definition"].gearing, 1.50)
    assert np.allclose(short_cap_state["definition"].spread, 0.0010)

    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XCCY_USD_SOFR_JPY_TONAR_0001")
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
    jpy_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "JPY")
    assert usd_leg["overnight_indexed"] is True
    assert jpy_leg["overnight_indexed"] is True
    assert jpy_leg["fx_reset"] is not None
    assert jpy_leg["fx_reset"]["foreign_currency"] == "USD"
    assert jpy_leg["fx_reset"]["foreign_amount"] == 100000000.0

    mapped = map_snapshot(snapshot)
    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        inputs = adapter._extract_inputs(snapshot, mapped)
        assert "USD/JPY" in inputs.stochastic_fx_pairs
        assert _today_spot_from_quotes("USDJPY", inputs) > 100.0
        result = adapter.run(
            replace(snapshot, config=replace(snapshot.config, num_paths=4)),
            mapped=mapped,
            run_id="generated-all-rates-variants",
        )
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


def test_generated_all_rates_strict_ore_inputs_cover_required_market_families(tmp_path):
    case_root = tmp_path / "USD_AllRatesProductsSnapshot_StrictInputs"
    broad_rates_example._write_files(case_root, count_per_type=1)
    snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    wanted = {
        "BASIS_USD_LIB3M_SIFMA_0001",
        "CMS_SWAP_USD_0001",
        "CAP_USD_SOFR3M_0001",
        "XCCY_USD_SOFR_JPY_TONAR_0001",
    }
    trades = tuple(t for t in snapshot.portfolio.trades if t.trade_id in wanted)
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=trades),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.strict_ore_inputs": "Y"},
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()

    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        inputs = adapter._extract_inputs(snapshot, map_snapshot(snapshot))

    assert not inputs.input_fallbacks
    assert "USD-SIFMA" in inputs.forward_curves_by_name
    assert math.isfinite(float(adapter._resolve_index_curve(inputs, "USD", "USD-CMS-30Y")(5.0)))
    assert "USD-SOFR" in inputs.forward_curves_by_name or "1D" in inputs.forward_curves_by_tenor.get("USD", {})
    assert "JPY-TONAR" in inputs.forward_curves_by_name or "1D" in inputs.forward_curves_by_tenor.get("JPY", {})
    assert _today_spot_from_quotes("USDJPY", inputs) > 100.0


@pytest.mark.parametrize(
    ("trade_id", "drop_prefix", "expected"),
    (
        ("BASIS_USD_LIB3M_SIFMA_0001", "BMA_SWAP/RATIO/USD", "missing_bma_ratio_curve:USD-SIFMA"),
        ("CAP_USD_SOFR3M_0001", "SOFR", "missing_forward_curve:USD-SOFR-3M"),
        ("XCCY_USD_SOFR_JPY_TONAR_0001", "FX/RATE/USD/JPY", "missing_fx_spot:USDJPY"),
        ("XCCY_USD_SOFR_JPY_TONAR_0001", "IR_SWAP/RATE/JPY/2D/1D", "missing_forward_curve:JPY-TONAR"),
    ),
)
def test_generated_all_rates_strict_ore_inputs_fail_fast_when_market_family_missing(tmp_path, trade_id, drop_prefix, expected):
    case_root = tmp_path / f"USD_AllRatesProductsSnapshot_StrictMissing_{trade_id}"
    broad_rates_example._write_files(case_root, count_per_type=1)
    snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == trade_id)
    snapshot = replace(
        snapshot,
        market=replace(
            snapshot.market,
            raw_quotes=tuple(
                q
                for q in snapshot.market.raw_quotes
                if drop_prefix.upper() not in str(q.key).upper()
                and not (drop_prefix.upper() == "IR_SWAP/RATE/JPY/2D/1D" and str(q.key).upper().startswith("MM/RATE/JPY/0D/1D"))
            ),
        ),
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.strict_ore_inputs": "Y"},
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()

    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ), pytest.raises(EngineRunError, match=expected):
        adapter._extract_inputs(snapshot, map_snapshot(snapshot))


def test_generated_sofr_capfloor_variants_have_ore_sign_and_scale_parity(tmp_path):
    if not LOCAL_ORE_BINARY.exists():
        return
    case_root = tmp_path / "USD_AllRatesProductsSnapshot_CapFloorOreParity"
    broad_rates_example._write_files(case_root, count_per_type=1)

    subprocess.run(
        [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
        cwd=case_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert "did not find capfloor curve for key" not in (case_root / "Output" / "log.txt").read_text(errors="ignore")
    with (case_root / "Output" / "npv.csv").open(newline="", encoding="utf-8") as handle:
        ore_npvs = {
            row.get("TradeId") or row.get("#TradeId"): float(row["NPV(Base)"])
            for row in csv.DictReader(handle)
            if row.get("NPV(Base)") not in {"", "#N/A", None}
        }

    base_snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    for trade_id, abs_tol in (
        ("CAP_USD_SOFR3M_0001", 6000.0),
        ("FLOOR_USD_SOFR3M_0001", 1500.0),
        ("FLOOR_USD_SOFR3M_FORWARD_0001", 1500.0),
        ("CAP_USD_LIB3M_0001", 5000.0),
        ("FLOOR_USD_LIB3M_0001", 500.0),
        ("CAP_USD_LIB3M_SHORT_GEARED_0001", 75000.0),
    ):
        trade = next(t for t in base_snapshot.portfolio.trades if t.trade_id == trade_id)
        snapshot = replace(
            base_snapshot,
            portfolio=replace(base_snapshot.portfolio, trades=(trade,)),
            config=replace(
                base_snapshot.config,
                num_paths=8,
                params={
                    **dict(base_snapshot.config.params),
                    "python.use_ore_output_curves": "Y",
                    "python.progress": "N",
                    "python.use_ore_flow_amounts_t0": "N",
                },
            ),
        )
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
                snapshot,
                mapped=map_snapshot(snapshot),
                run_id=f"generated-sofr-capfloor-ore-parity-{trade_id.lower()}",
            )
        ore_npv = ore_npvs[trade_id]
        py_npv = float(result.pv_total)
        assert math.copysign(1.0, py_npv) == math.copysign(1.0, ore_npv)
        assert abs(py_npv - ore_npv) <= abs_tol


@pytest.mark.parametrize("count_per_type", (1, 5))
def test_generated_all_rates_numpy_and_torch_t0_prices_align(tmp_path, count_per_type):
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch is required for backend parity regression")

    case_root = tmp_path / "USD_AllRatesProductsSnapshot_BackendParity"
    broad_rates_example._write_files(case_root, count_per_type=count_per_type)
    base_snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")

    def run_backend(device: str):
        snapshot = replace(
            base_snapshot,
            config=replace(
                base_snapshot.config,
                num_paths=16,
                params={
                    **dict(base_snapshot.config.params),
                    "python.torch_device": device,
                    "python.lgm_rng_mode": "numpy",
                    "python.progress": "N",
                    "python.store_npv_cube_paths": "Y",
                },
            ),
        )
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            return XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
                snapshot,
                mapped=map_snapshot(snapshot),
                run_id=f"generated-all-rates-{device}",
            )

    numpy_result = run_backend("numpy")
    torch_result = run_backend("cpu")
    assert numpy_result.metadata["irs_pricing_backend"] == "numpy"
    assert torch_result.metadata["irs_pricing_backend"] == "torch:cpu"
    assert math.isclose(float(torch_result.pv_total), float(numpy_result.pv_total), rel_tol=0.0, abs_tol=1.0e-7)
    assert math.isclose(
        float(torch_result.xva_by_metric.get("CVA", 0.0)),
        float(numpy_result.xva_by_metric.get("CVA", 0.0)),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )

    numpy_cube = numpy_result.cubes["npv_cube"].payload
    torch_cube = torch_result.cubes["npv_cube"].payload
    assert set(torch_cube) == set(numpy_cube)
    for trade_id, numpy_payload in numpy_cube.items():
        numpy_t0 = float(numpy_payload["npv_mean"][0])
        torch_t0 = float(torch_cube[trade_id]["npv_mean"][0])
        assert math.isclose(torch_t0, numpy_t0, rel_tol=0.0, abs_tol=1.0e-7), trade_id

    for path_key in ("npv_paths", "npv_xva_paths"):
        for trade_id, numpy_payload in numpy_cube.items():
            max_abs_diff = float(
                np.max(
                    np.abs(
                        np.asarray(torch_cube[trade_id][path_key])
                        - np.asarray(numpy_payload[path_key])
                    )
                )
            )
            assert max_abs_diff <= 1.0e-6, (path_key, trade_id, max_abs_diff)

    mapped = map_snapshot(base_snapshot)
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    specs, unsupported, _ = adapter._classify_portfolio_trades(base_snapshot, mapped)
    assert unsupported == []
    basis_specs = [
        spec
        for spec in specs
        if spec.trade.trade_id.startswith(("BASIS_USD_LIB3M_LIB6M", "BASIS_USD_SOFR3M_LIB3M"))
    ]
    assert basis_specs
    assert all(not adapter._supports_torch_rate_swap(spec) for spec in basis_specs)


def test_bermudan_swaption_uses_trade_specific_gsr_calibration_when_available():
    case_dir = TOOLS_DIR / "Examples" / "Generated" / "USD_AllRatesProductsSnapshot_134PerType" / "Input"
    result = price_bermudan_from_ore_case(
        case_dir,
        ore_file="ore.xml",
        trade_id="BERMUDAN_SWAPTION_USD_0001",
        method="backward",
        num_paths=256,
        seed=42,
        basis_degree=2,
        curve_mode="auto",
    )
    assert result.model_param_source == "trade_specific_gsr"
    assert result.curve_source == "ore_quote_fit"


def test_example76_floor_usd_maturing_matches_ore_trade_npv():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_76"), ore_file="Input/ore_maturing.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "FLOOR_USD_Maturing")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.use_ore_output_curves": "Y"},
        ),
    )
    with (TOOLS_DIR / "Examples" / "Legacy" / "Example_76" / "ExpectedOutput" / "npv.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        ore_rows = csv.DictReader(handle)
        ore_npv_row = next(row for row in ore_rows if (row.get("TradeId") or row.get("#TradeId")) == "FLOOR_USD_Maturing")
    ore_npv = float(ore_npv_row["NPV"])

    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        result = adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id="example76-floor-usd-maturing-ore-trade-npv",
        )

    assert math.isclose(float(result.pv_total), ore_npv, rel_tol=1.0e-12, abs_tol=1.0e-6)


def test_example63_cap_trade_matches_ore_npv_when_replaying_flows_csv():
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_pricing_only_case("Example_63", trade_ids=("Cap",), ore_file="ore_valid_cap.xml")
    try:
        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore_valid_cap.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore_valid_cap.xml")
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=8,
                params={
                    **dict(snapshot.config.params),
                    "python.use_flows_csv": "Y",
                    "python.use_ore_output_curves": "Y",
                },
            ),
        )
        adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        adapter._ensure_py_lgm_imports()
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            result = adapter.run(snapshot, mapped=map_snapshot(snapshot), run_id="example63-cap-ore-flows-parity")

        assert math.isfinite(float(result.pv_total))
        coverage = result.metadata["coverage"]
        assert coverage["fallback_trades"] == 0
        assert coverage["unsupported"] == []
    finally:
        tmp.cleanup()


def test_runtime_exposure_profiles_include_pfe_and_basel_fields():
    snapshot = _load_case("Examples/Legacy/Example_21/Input")
    cms_trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "CMS_Swap")
    snapshot = replace(snapshot, portfolio=replace(snapshot.portfolio, trades=(cms_trade,)))
    runtime = snapshot.config.runtime or RuntimeConfig()
    snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            num_paths=32,
            runtime=replace(
                runtime,
                xva_analytic=replace(runtime.xva_analytic, pfe_quantile=0.95),
            ),
        ),
    )
    result = _run_python(snapshot, "exposure-profile-shape")
    assert result.exposure_profiles_by_netting_set
    assert result.exposure_profiles_by_trade
    ns_profile = next(iter(result.exposure_profiles_by_netting_set.values()))
    trade_profile = next(iter(result.exposure_profiles_by_trade.values()))
    for profile in (ns_profile, trade_profile):
        for key in (
            "times",
            "closeout_times",
            "valuation_epe",
            "valuation_ene",
            "closeout_epe",
            "closeout_ene",
            "pfe",
            "basel_ee",
            "basel_eee",
            "time_weighted_basel_epe",
            "time_weighted_basel_eepe",
        ):
            assert key in profile
        assert len(profile["times"]) == len(profile["pfe"])
        assert len(profile["times"]) == len(profile["basel_ee"])
    basel_eee = np.asarray(ns_profile["basel_eee"], dtype=float)
    assert np.all(basel_eee[1:] >= basel_eee[:-1])
    assert math.isclose(
        result.reports["exposure"][0]["BaselEEPE"],
        result.netting_exposure_profile(next(iter(result.exposure_profiles_by_netting_set))).series("time_weighted_basel_eepe")[
            min(
                np.searchsorted(np.asarray(ns_profile["times"], dtype=float), 1.0, side="left"),
                len(ns_profile["times"]) - 1,
            )
        ],
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )


def test_runtime_pfe_quantile_is_monotonic():
    snapshot = _load_case("Examples/Legacy/Example_21/Input")
    cms_trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "CMS_Swap")
    snapshot = replace(snapshot, portfolio=replace(snapshot.portfolio, trades=(cms_trade,)))
    runtime = snapshot.config.runtime or RuntimeConfig()
    low_snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            num_paths=64,
            runtime=replace(
                runtime,
                xva_analytic=replace(getattr(runtime, "xva_analytic", XVAAnalyticConfig()), pfe_quantile=0.80),
            ),
        ),
    )
    high_snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            num_paths=64,
            runtime=replace(
                runtime,
                xva_analytic=replace(getattr(runtime, "xva_analytic", XVAAnalyticConfig()), pfe_quantile=0.99),
            ),
        ),
    )
    low = _run_python(low_snapshot, "pfe-q80")
    high = _run_python(high_snapshot, "pfe-q99")
    netting_id = next(iter(low.exposure_profiles_by_netting_set))
    low_pfe = np.asarray(low.netting_exposure_profile(netting_id).series("pfe"), dtype=float)
    high_pfe = np.asarray(high.netting_exposure_profile(netting_id).series("pfe"), dtype=float)
    assert np.all(high_pfe >= low_pfe - 1.0e-12)


def test_python_runtime_reprices_example25_cmsspread_from_input_market():
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=("CMS_Spread_Swap",))
    try:
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=4))
        result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id="cmsspread-native",
        )
        assert math.isclose(float(result.pv_total), 309150.0484271799, rel_tol=1.0e-10, abs_tol=1.0e-8)
        assert result.metadata["cmsspread_profile_mode"] == "frozen_input_native"
    finally:
        tmp.cleanup()


@pytest.mark.parametrize("trade_id", ("CMS_Spread_Swap", "Digital_CMS_Spread"))
def test_example25_cmsspread_replay_mode_anchors_rate_swap_t0_to_ore_npv(trade_id):
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=(trade_id,))
    try:
        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        with (case_root / "Output" / "npv.csv").open(newline="", encoding="utf-8") as handle:
            ore_npv = float(next(csv.DictReader(handle))["NPV(Base)"])

        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=4,
                params={
                    **dict(snapshot.config.params),
                    "python.use_ore_flow_amounts_t0": "Y",
                    "python.progress": "N",
                    "python.store_npv_cube_paths": "Y",
                },
            ),
        )
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
                snapshot,
                mapped=map_snapshot(snapshot),
                run_id=f"{trade_id.lower()}-ore-replay",
            )

        assert math.isclose(float(result.pv_total), ore_npv, rel_tol=1.0e-12, abs_tol=1.0e-8)
        cube_t0 = float(result.cubes["npv_cube"].payload[trade_id]["npv_mean"][0])
        assert math.isclose(cube_t0, ore_npv, rel_tol=1.0e-12, abs_tol=1.0e-8)
    finally:
        tmp.cleanup()


def test_example25_ore_output_curves_alias_cms_indices_to_underlying_forward_curve():
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=("CMS_Spread_Swap",))
    try:
        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=4,
                params={**dict(snapshot.config.params), "python.use_ore_output_curves": "Y"},
            ),
        )
        adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        adapter._ensure_py_lgm_imports()
        with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
        ):
            inputs = adapter._extract_inputs(snapshot, map_snapshot(snapshot))

        assert inputs.forward_curves_by_name["EUR-CMS-10Y"] is inputs.forward_curves_by_name["EUR-EURIBOR-6M"]
        assert inputs.forward_curves_by_name["EUR-CMS-2Y"] is inputs.forward_curves_by_name["EUR-EURIBOR-6M"]
    finally:
        tmp.cleanup()


def test_python_runtime_supports_real_bma_basis_case_without_fallback():
    snapshot = _load_case("Examples/Legacy/Example_27/Input")
    result = _run_python(snapshot, "bma-test")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


def test_python_runtime_supports_real_fra_without_fallback():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Exposure" / "Input"), ore_file="ore_fra.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "fra1")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            analytics=("CVA",),
            num_paths=8,
            params={**dict(snapshot.config.params), "python.progress": "N"},
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    specs, unsupported, _ = adapter._classify_portfolio_trades(snapshot, mapped)
    assert unsupported == []
    assert [(spec.trade.trade_id, spec.kind, spec.ccy) for spec in specs] == [("fra1", "FRA", "EUR")]

    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        result = adapter.run(snapshot, mapped=mapped, run_id="fra-native")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


@pytest.mark.parametrize(
    "trade_id",
    ("BASIS_USD_LIB3M_SIFMA_0001", "BASIS_USD_SOFR3M_SIFMA_0001"),
)
def test_python_runtime_builds_real_sifma_basis_curves_from_bma_ratio_only(trade_id):
    snapshot = _load_case("Examples/Generated/USD_AllRatesProductsSnapshot_20PerType/Input")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == trade_id)
    snapshot = replace(snapshot, portfolio=replace(snapshot.portfolio, trades=(trade,)))
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    inputs = adapter._extract_inputs(snapshot, mapped)
    assert "USD-SIFMA" in inputs.forward_curves_by_name
    assert "USD-BMA" in inputs.forward_curves_by_name
    assert inputs.forward_curves_by_name["USD-SIFMA"] is inputs.forward_curves_by_name["USD-BMA"]
    assert math.isfinite(float(inputs.forward_curves_by_name["USD-SIFMA"](5.0)))
    assert (
        adapter._resolve_index_curve(inputs, "USD", "USD-SIFMA-1W")
        is inputs.forward_curves_by_name["USD-SIFMA"]
    )
    assert (
        adapter._resolve_index_curve(inputs, "USD", "USD-SIFMA-7D")
        is inputs.forward_curves_by_name["USD-SIFMA"]
    )
    p = inputs.lgm_params
    model = adapter._lgm_mod.LGM1F(
        adapter._lgm_mod.LGMParams(
            alpha_times=tuple(p["alpha_times"]),
            alpha_values=tuple(p["alpha_values"]),
            kappa_times=tuple(p["kappa_times"]),
            kappa_values=tuple(p["kappa_values"]),
            shift=p["shift"],
            scaling=p["scaling"],
        )
    )
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    sifma_leg = next(leg for leg in state["rate_legs"] if "SIFMA" in str(leg.get("index_name", "")).upper())
    assert bool(sifma_leg.get("overnight_indexed", False))
    sifma_coupons = adapter._rate_leg_coupon_paths(
        model,
        sifma_leg,
        "USD",
        inputs,
        0.0,
        np.zeros(2, dtype=float),
        snapshot=snapshot,
    )
    assert np.all(np.isfinite(sifma_coupons))
    assert float(np.max(np.abs(sifma_coupons))) > 1.0e-6

    specs, unsupported, _ = adapter._classify_portfolio_trades(snapshot, mapped)
    assert unsupported == []
    spec = next(s for s in specs if s.trade.trade_id == trade_id)
    assert spec.kind == "RateSwap"
    assert not adapter._supports_torch_rate_swap(spec)

    with patch.object(adapter, "_resolve_irs_pricing_backend", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None
    ), patch("pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None):
        numpy_result = adapter.run(snapshot, mapped=mapped, run_id=f"sifma-{trade_id.lower()}-numpy")

    assert math.isfinite(float(numpy_result.pv_total))
    exclusions = numpy_result.metadata["torch_rate_swap_exclusions"][trade_id]
    assert "floating_basis_swap" in exclusions
    assert "overnight_indexed" in exclusions
    coverage = numpy_result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []


def test_ore_output_curves_builds_sifma_from_bma_ratio_when_curves_csv_has_no_sifma(tmp_path, monkeypatch):
    source_input = TOOLS_DIR / "Examples" / "Generated" / "USD_AllRatesProductsSnapshot_20PerType" / "Input"
    case_root = tmp_path / "sifma_output_curves_case"
    shutil.copytree(source_input, case_root / "Input")
    output_dir = case_root / "Output"
    output_dir.mkdir()
    (output_dir / "curves.csv").write_text(
        "Date,USD,USD-LIBOR-3M\n2026-03-08,1.0,1.0\n2027-03-08,0.96,0.97\n",
        encoding="utf-8",
    )

    snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "BASIS_USD_LIB3M_SIFMA_0001")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(
            snapshot.config,
            num_paths=4,
            params={**dict(snapshot.config.params), "python.use_ore_output_curves": "Y"},
        ),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    specs, unsupported, _ = adapter._classify_portfolio_trades(snapshot, mapped)
    assert unsupported == []

    def fake_curve_loader(_curves_csv, requested_columns, *, asof_date, day_counter):
        del _curves_csv, asof_date, day_counter
        dates = ("2026-03-08", "2027-03-08", "2031-03-08")
        times = np.asarray([0.0, 1.0, 5.0], dtype=float)
        return {
            str(col): (dates, times, np.asarray([1.0, 0.96, 0.82], dtype=float))
            for col in requested_columns
            if str(col).upper() != "USD-SIFMA"
        }

    monkeypatch.setattr(
        adapter._ore_snapshot_mod,
        "_load_ore_discount_pairs_by_columns_with_day_counter",
        fake_curve_loader,
    )
    bundle = adapter._load_ore_output_curves(
        snapshot,
        mapped,
        specs,
        bma_ratio_curves={"USD": [(1.0, 0.72), (5.0, 0.75)]},
    )
    assert bundle is not None
    assert "USD-SIFMA" in bundle.forward_curves_by_name
    assert "USD-BMA" in bundle.forward_curves_by_name
    assert bundle.forward_curves_by_name["USD-SIFMA"] is bundle.forward_curves_by_name["USD-BMA"]
    assert math.isfinite(float(bundle.forward_curves_by_name["USD-SIFMA"](5.0)))


@pytest.mark.parametrize(
    "case_name,trade_id,leg0,leg1,expected_indices",
    (
        (
            "USD_SOFR_LIBOR_Basis",
            "USD_SOFR_LIBOR_BASIS",
            {
                "payer": True,
                "currency": "USD",
                "notional": 10_000_000.0,
                "index": "USD-SOFR-3M",
                "spread": 0.0005,
                "gearing": 1.0,
                "tenor": "3M",
                "calendar": "TARGET",
                "day_counter": "ACT",
                    "payment_convention": "MF",
                    "schedule_convention": "MF",
                    "term_convention": "MF",
                    "rule": "Forward",
                    "notional_initial_exchange": False,
                    "notional_final_exchange": False,
                    "fixing_days": 0,
                    "is_in_arrears": True,
                },
                {
                "payer": False,
                "currency": "USD",
                "notional": 10_000_000.0,
                "index": "USD-LIBOR-3M",
                "spread": 0.0015,
                "gearing": 1.0,
                "tenor": "3M",
                "calendar": "TARGET",
                "day_counter": "A360",
                    "payment_convention": "MF",
                    "schedule_convention": "MF",
                    "term_convention": "MF",
                    "rule": "Forward",
                    "notional_initial_exchange": False,
                    "notional_final_exchange": False,
                    "fixing_days": 2,
                    "is_in_arrears": False,
                },
            {"USD-SOFR-3M", "USD-LIBOR-3M"},
        ),
        (
            "USD_LIBOR1M_LIBOR3M_Basis",
            "USD_LIBOR1M_LIBOR3M_BASIS",
            {
                "payer": True,
                "currency": "USD",
                "notional": 10_000_000.0,
                "index": "USD-LIBOR-1M",
                "spread": 0.0005,
                "gearing": 1.0,
                "tenor": "1M",
                "calendar": "US",
                "day_counter": "A360",
                    "payment_convention": "MF",
                    "schedule_convention": "MF",
                    "term_convention": "MF",
                    "rule": "Forward",
                    "notional_initial_exchange": False,
                    "notional_final_exchange": False,
                    "fixing_days": 2,
                    "is_in_arrears": False,
                },
            {
                "payer": False,
                "currency": "USD",
                "notional": 10_000_000.0,
                "index": "USD-LIBOR-3M",
                "spread": 0.0015,
                "gearing": 1.0,
                "tenor": "3M",
                "calendar": "US",
                "day_counter": "A360",
                    "payment_convention": "MF",
                    "schedule_convention": "MF",
                    "term_convention": "MF",
                    "rule": "Forward",
                    "notional_initial_exchange": False,
                    "notional_final_exchange": False,
                    "fixing_days": 2,
                    "is_in_arrears": False,
                },
            {"USD-LIBOR-1M", "USD-LIBOR-3M"},
        ),
    ),
)
def test_python_runtime_supports_float_float_basis_swaps_via_generic_rate_swap(case_name, trade_id, leg0, leg1, expected_indices):
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_basis_swap_case(
        case_name,
        trade_id=trade_id,
        counterparty="CPTY_A",
        netting_set_id="CPTY_A",
        asof_date="2025-02-10",
        start_date="2019-12-04",
        end_date="2045-04-05",
        tenor=str(leg0["tenor"]),
        calendar=str(leg0["calendar"]),
        leg0=leg0,
        leg1=leg1,
    )
    try:
        subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=256))
        trade = next(t for t in snapshot.portfolio.trades if t.trade_id == trade_id)
        assert isinstance(trade.product, GenericProduct)
        assert trade.product.payload.get("subtype") == "GenericRateSwap"
        adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        adapter._ensure_py_lgm_imports()
        state = adapter._build_generic_rate_swap_legs(trade, snapshot)
        assert state is not None
        assert len(state["rate_legs"]) == 2
        assert all(leg["kind"] == "FLOATING" for leg in state["rate_legs"])
        assert {str(leg["index_name"]).upper() for leg in state["rate_legs"]} == {idx.upper() for idx in expected_indices}

        result = _run_python(snapshot, case_name.lower())
        assert math.isfinite(float(result.pv_total))

        npv_csv = _find_report_csv(case_root, "npv.csv")
        with npv_csv.open(newline="", encoding="utf-8") as handle:
            ore_row = next(row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == trade_id)
        ore_npv = float(ore_row.get("NPV") or ore_row.get("npv") or ore_row.get("NPV(Base)") or ore_row.get("NPV Base"))
        assert math.copysign(1.0, float(result.pv_total)) == math.copysign(1.0, ore_npv)
        assert abs(float(result.pv_total) - ore_npv) < 1.0e6
    finally:
        tmp.cleanup()


def test_torch_plain_rate_swap_matches_numpy_runtime_on_bma_basis_case():
    snapshot = _load_case("Examples/Legacy/Example_27/Input")
    snapshot = replace(
        snapshot,
        config=replace(
            snapshot.config,
            num_paths=16,
            params={**snapshot.config.params, "python.progress": "N", "python.progress_bar": "N"},
        ),
    )
    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_in_python", return_value=None), patch(
        "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None
    ):
        numpy_adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        with patch.object(numpy_adapter, "_resolve_irs_pricing_backend", return_value=None):
            numpy_result = numpy_adapter.run(
                snapshot,
                mapped=map_snapshot(snapshot),
                run_id="bma-numpy",
            )

        torch_adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
        from pythonore.compute.lgm_torch_xva import (
            TorchDiscountCurve,
            capfloor_npv_paths_torch,
            deflate_lgm_npv_paths_torch_batched,
            par_swap_rate_paths_torch,
            price_plain_rate_leg_paths_torch,
            swap_npv_paths_from_ore_legs_dual_curve_torch,
        )

        backend = (
            TorchDiscountCurve,
            swap_npv_paths_from_ore_legs_dual_curve_torch,
            deflate_lgm_npv_paths_torch_batched,
            "cpu",
            price_plain_rate_leg_paths_torch,
            par_swap_rate_paths_torch,
            capfloor_npv_paths_torch,
        )
        with patch.object(torch_adapter, "_resolve_irs_pricing_backend", return_value=backend):
            torch_result = torch_adapter.run(
                snapshot,
                mapped=map_snapshot(snapshot),
                run_id="bma-torch",
            )

    assert math.isclose(float(torch_result.pv_total), float(numpy_result.pv_total), rel_tol=2.0e-5, abs_tol=1.0e-2)
    assert math.isclose(
        float(torch_result.xva_by_metric.get("CVA", 0.0)),
        float(numpy_result.xva_by_metric.get("CVA", 0.0)),
        rel_tol=2.0e-5,
        abs_tol=1.0e-2,
    )


def test_native_runtime_ignores_residual_output_curves_and_calibration_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        case_root = tmp_root / "case"
        input_dir = case_root / "Input"
        output_dir = case_root / "Output"
        source_input = TOOLS_DIR / "Examples/Legacy/Example_25/Input"
        source_output = TOOLS_DIR / "Examples/Legacy/Example_25/Output"
        shutil.copytree(source_input, input_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("npv.csv", "flows.csv"):
            src = source_output / name
            if src.exists():
                shutil.copy2(src, output_dir / name)
        (output_dir / "curves.csv").write_text(
            "Date,EUR-EONIA\n2016-02-05,-99.0\n2017-02-05,-99.0\n",
            encoding="utf-8",
        )
        poisoned_calibration = """<?xml version="1.0" encoding="UTF-8"?>
<Root><InterestRateModels><LGM currency="EUR"><CalibrationType>Bootstrap</CalibrationType>
<Reversion><TimeGrid>1.0</TimeGrid><Value>9.99</Value></Reversion>
<Volatility><TimeGrid>1.0</TimeGrid><Value>9.99</Value></Volatility>
<ParameterTransformation><ShiftHorizon>0</ShiftHorizon><Scaling>1</Scaling></ParameterTransformation>
</LGM></InterestRateModels></Root>"""
        (output_dir / "calibration.xml").write_text(poisoned_calibration, encoding="utf-8")

        snapshot = XVALoader.from_files(str(case_root), ore_file="Input/ore.xml")
        snapshot = replace(
            snapshot,
            config=replace(
                snapshot.config,
                num_paths=4,
                xml_buffers={**snapshot.config.xml_buffers, "calibration.xml": poisoned_calibration},
            ),
        )
        runtime_params = {
            "alpha_times": (1.0,),
            "alpha_values": (0.015, 0.015),
            "kappa_times": (1.0,),
            "kappa_values": (0.03, 0.03),
            "shift": 0.0,
            "scaling": 1.0,
        }
        with patch(
            "pythonore.runtime.lgm.market._parse_lgm_params_from_calibration_xml_text",
            side_effect=AssertionError("residual calibration.xml should not be used"),
        ), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_in_python",
            return_value=runtime_params,
        ), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore",
            side_effect=AssertionError("ORE calibration should not be used when Python calibration succeeds"),
        ):
            result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
                snapshot,
                mapped=map_snapshot(snapshot),
                run_id="ignore-output-artifacts",
            )
        provenance = result.metadata["input_provenance"]
        assert provenance["market"] != "ore_output_curves"
        assert provenance["model_params"] == "calibration"


def test_digital_cmsspread_replication_matches_ore_digital_coupon_identities():
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    raw = np.asarray([-0.0020, -0.0001, 0.00005, 0.0002, 0.0015], dtype=float)
    strike = 0.0001
    payoff = 0.0001
    eps = 1.0e-4

    call_cash = adapter._digital_option_rate(
        raw,
        strike,
        payoff,
        is_call=True,
        long_short=1.0,
        fixed_mode=False,
        atm_included=False,
    )
    expected_call_cash = payoff * (
        adapter._capped_floored_rate(raw, cap=strike + eps / 2.0)
        - adapter._capped_floored_rate(raw, cap=strike - eps / 2.0)
    ) / eps
    assert np.allclose(call_cash, expected_call_cash)

    put_cash = adapter._digital_option_rate(
        raw,
        strike,
        payoff,
        is_call=False,
        long_short=1.0,
        fixed_mode=False,
        atm_included=False,
    )
    expected_put_cash = payoff * (
        adapter._capped_floored_rate(raw, floor=strike + eps / 2.0)
        - adapter._capped_floored_rate(raw, floor=strike - eps / 2.0)
    ) / eps
    assert np.allclose(put_cash, expected_put_cash)

    call_asset = adapter._digital_option_rate(
        raw,
        strike,
        float("nan"),
        is_call=True,
        long_short=1.0,
        fixed_mode=False,
        atm_included=False,
    )
    expected_call_asset = strike * (
        adapter._capped_floored_rate(raw, cap=strike + eps / 2.0)
        - adapter._capped_floored_rate(raw, cap=strike - eps / 2.0)
    ) / eps + (raw - adapter._capped_floored_rate(raw, cap=strike))
    assert np.allclose(call_asset, expected_call_asset)

    put_asset = adapter._digital_option_rate(
        raw,
        strike,
        float("nan"),
        is_call=False,
        long_short=1.0,
        fixed_mode=False,
        atm_included=False,
    )
    expected_put_asset = strike * (
        adapter._capped_floored_rate(raw, floor=strike + eps / 2.0)
        - adapter._capped_floored_rate(raw, floor=strike - eps / 2.0)
    ) / eps - (-raw + adapter._capped_floored_rate(raw, floor=strike))
    assert np.allclose(put_asset, expected_put_asset)


def test_python_runtime_reprices_example25_digital_cmsspread_from_input_market():
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=("Digital_CMS_Spread",))
    try:
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=4))
        result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id="digital-cmsspread-native",
        )
        assert math.isclose(float(result.pv_total), 284672.65625257127, rel_tol=1.0e-10, abs_tol=1.0e-8)
        provenance = result.metadata["input_provenance"]
        assert provenance["market"] == "market_overlay"
        assert provenance["model_params"] == "calibration"
        assert result.metadata["cmsspread_profile_mode"] == "frozen_input_native"
    finally:
        tmp.cleanup()


def test_example25_cmsspread_ignores_poisoned_output_artifacts():
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=("CMS_Spread_Swap", "Digital_CMS_Spread"))
    try:
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=4))
        baseline = _run_python(snapshot, "cmsspread-baseline")
        _write_poisoned_output_artifacts(case_root)
        poisoned_snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        poisoned_snapshot = replace(poisoned_snapshot, config=replace(poisoned_snapshot.config, num_paths=4))
        poisoned = _run_python(poisoned_snapshot, "cmsspread-poisoned")
        assert math.isclose(float(baseline.pv_total), float(poisoned.pv_total), rel_tol=1.0e-12, abs_tol=1.0e-12)
    finally:
        tmp.cleanup()


def test_example25_cmsspread_prices_without_flows_csv():
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=("CMS_Spread_Swap", "Digital_CMS_Spread"))
    try:
        output_dir = case_root / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=4))
        result = _run_python(snapshot, "cmsspread-no-flows")
        assert math.isfinite(float(result.pv_total))
        assert result.metadata["cmsspread_profile_mode"] == "frozen_input_native"
    finally:
        tmp.cleanup()


def test_build_irs_legs_ignores_flows_csv_by_default():
    case_root = TOOLS_DIR / "parity_artifacts" / "multiccy_benchmark_final" / "cases" / "flat_EUR_5Y_A"
    snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=4))
    trade = snapshot.portfolio.trades[0]
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()

    with patch.object(adapter._irs_utils, "load_ore_legs_from_flows", side_effect=AssertionError("flows.csv should not be used by default")):
        legs = adapter._build_irs_legs(trade, map_snapshot(snapshot), snapshot)

    assert "fixed_notional" in legs
    assert np.asarray(legs["fixed_notional"], dtype=float).size > 0


def test_build_irs_legs_ignores_flows_csv_when_explicitly_requested():
    case_root = TOOLS_DIR / "parity_artifacts" / "multiccy_benchmark_final" / "cases" / "flat_EUR_5Y_A"
    snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
    snapshot = replace(
        snapshot,
        config=replace(snapshot.config, num_paths=4, params={**dict(snapshot.config.params), "python.use_flows_csv": "Y"}),
    )
    trade = snapshot.portfolio.trades[0]
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    flow_legs = {
        "fixed_pay_time": np.array([0.5], dtype=float),
        "fixed_start_time": np.array([0.0], dtype=float),
        "fixed_end_time": np.array([0.5], dtype=float),
        "fixed_accrual": np.array([0.5], dtype=float),
        "fixed_rate": np.array([0.02], dtype=float),
        "fixed_notional": np.array([1_234_567.0], dtype=float),
        "fixed_sign": np.array([-1.0], dtype=float),
        "fixed_amount": np.array([-12_345.67], dtype=float),
        "float_pay_time": np.array([0.5], dtype=float),
        "float_start_time": np.array([0.0], dtype=float),
        "float_end_time": np.array([0.5], dtype=float),
        "float_accrual": np.array([0.5], dtype=float),
        "float_index_accrual": np.array([0.5], dtype=float),
        "float_notional": np.array([1_234_567.0], dtype=float),
        "float_sign": np.array([1.0], dtype=float),
        "float_coupon": np.array([0.0], dtype=float),
        "float_amount": np.array([0.0], dtype=float),
        "float_spread": np.array([0.0], dtype=float),
        "float_fixing_time": np.array([0.0], dtype=float),
    }

    with patch.object(adapter._irs_utils, "load_ore_legs_from_flows", side_effect=AssertionError("flows.csv should not be used")) as flow_loader, patch.object(
        adapter._irs_utils, "load_swap_legs_from_portfolio_root", return_value=flow_legs
    ) as portfolio_loader:
        legs = adapter._build_irs_legs(trade, map_snapshot(snapshot), snapshot)

    flow_loader.assert_not_called()
    portfolio_loader.assert_called()
    np.testing.assert_allclose(np.asarray(legs["fixed_notional"], dtype=float), np.array([1_234_567.0]))


def test_compute_realized_float_coupons_clamps_start_to_fixing_time():
    class _DummyModel:
        def __init__(self):
            self.calls = []

        def discount_bond(self, t, maturity, x, p_t, p0_maturity):
            self.calls.append((float(t), float(maturity), float(p_t), float(p0_maturity)))
            return np.full_like(np.asarray(x, dtype=float), float(p0_maturity), dtype=float)

    model = _DummyModel()
    p0_disc = lambda t: 1.0 / (1.0 + 0.02 * float(t))
    p0_fwd = lambda t: 1.0 / (1.0 + 0.015 * float(t))
    legs = {
        "float_start_time": np.array([0.10], dtype=float),
        "float_end_time": np.array([0.40], dtype=float),
        "float_accrual": np.array([0.30], dtype=float),
        "float_index_accrual": np.array([0.30], dtype=float),
        "float_spread": np.array([0.0], dtype=float),
        "float_fixing_time": np.array([0.25], dtype=float),
    }
    sim_times = np.array([0.0, 0.25, 0.5], dtype=float)
    x_paths = np.zeros((sim_times.size, 1), dtype=float)

    realized = compute_realized_float_coupons(model, p0_disc, p0_fwd, legs, sim_times, x_paths)

    assert realized.shape == (1, 1)
    assert model.calls, "expected the coupon helper to call the discount bond helper"
    # The helper must clamp the effective start to the fixing time, otherwise this
    # call would still use the accrual start at 0.10.
    assert math.isclose(model.calls[0][1], 0.25, rel_tol=0.0, abs_tol=1.0e-12)


def test_swap_npv_uses_explicit_live_coupons_for_averaged_overnight_legs():
    class _DummyModel:
        def discount_bond(self, t, maturity, x, p_t, p0_maturity):
            return np.full_like(np.asarray(x, dtype=float), float(p0_maturity), dtype=float)

        def discount_bond_paths(self, t, maturities, x_paths, p_t, p0_T):
            p0_T = np.asarray(p0_T, dtype=float)
            x_paths = np.asarray(x_paths, dtype=float)
            return np.tile((p0_T / float(p_t))[:, None], (1, x_paths.size))

    model = _DummyModel()
    p0_disc = lambda t: 1.0 / (1.0 + 0.02 * float(t))
    p0_fwd = lambda t: 1.0 / (1.0 + 0.01 * float(t))
    legs = {
        "fixed_pay_time": np.array([], dtype=float),
        "fixed_start_time": np.array([], dtype=float),
        "fixed_end_time": np.array([], dtype=float),
        "fixed_amount": np.array([], dtype=float),
        "float_start_time": np.array([0.10], dtype=float),
        "float_end_time": np.array([0.40], dtype=float),
        "float_pay_time": np.array([0.50], dtype=float),
        "float_accrual": np.array([0.30], dtype=float),
        "float_index_accrual": np.array([0.30], dtype=float),
        "float_notional": np.array([100.0], dtype=float),
        "float_sign": np.array([1.0], dtype=float),
        "float_spread": np.array([0.0], dtype=float),
        "float_coupon": np.array([0.0], dtype=float),
        "float_fixing_time": np.array([0.60], dtype=float),
        "float_is_averaged": np.array([True], dtype=bool),
    }
    live_coupon = np.array([[0.123]], dtype=float)
    pv = swap_npv_from_ore_legs_dual_curve(
        model,
        p0_disc,
        p0_fwd,
        legs,
        t=0.0,
        x_t=np.zeros(1, dtype=float),
        realized_float_coupon=live_coupon,
        live_float_coupon=live_coupon,
    )
    expected = 100.0 * 0.30 * 0.123 * p0_disc(0.50)
    assert pv.shape == (1,)
    assert math.isclose(float(pv[0]), expected, rel_tol=0.0, abs_tol=1.0e-12)


def test_year_fraction_supports_30e_360_and_bare_act():
    assert math.isclose(_year_fraction(date(2024, 1, 31), date(2024, 2, 28), "30E/360"), 28.0 / 360.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(_year_fraction(date(2024, 2, 28), date(2024, 3, 1), "ACT"), 2.0 / 366.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_sofr_index_day_counter_defaults_to_act_act():
    assert _infer_index_day_counter("USD-SOFR", fallback="A365F") == "ACT/ACT"


def test_average_ois_uses_end_lagged_fixings():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_23" / "Input"), ore_file="ore.xml")
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "averageOIS")
    snapshot = replace(
        snapshot,
        portfolio=replace(snapshot.portfolio, trades=(trade,)),
        config=replace(snapshot.config, analytics=(), num_paths=4),
    )
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    mapped = map_snapshot(snapshot)
    legs = adapter._build_irs_legs(trade, mapped, snapshot)
    assert bool(legs.get("float_is_averaged", False)) is True
    assert np.asarray(legs["float_fixing_time"], dtype=float)[0] > np.asarray(legs["float_start_time"], dtype=float)[0]
    assert np.asarray(legs["float_fixing_time"], dtype=float)[0] < np.asarray(legs["float_end_time"], dtype=float)[0]


def test_average_overnight_fixing_date_moves_toward_period_end_with_cutoff():
    start = date(2024, 2, 5)
    end = date(2024, 5, 6)
    late = _average_overnight_coupon_fixing_date(start, end, calendar="TARGET", fixing_days=0, rate_cutoff=0)
    early = _average_overnight_coupon_fixing_date(start, end, calendar="TARGET", fixing_days=0, rate_cutoff=5)
    assert late > start
    assert late < end
    assert early < late


def test_capfloor_surface_rate_computation_period_caches_raw_quote_scan():
    class RawQuotes:
        def __init__(self, quotes):
            self.quotes = tuple(quotes)
            self.iter_count = 0

        def __iter__(self):
            self.iter_count += 1
            return iter(self.quotes)

    class Snapshot:
        def __init__(self, raw_quotes):
            self.market = type("Market", (), {"raw_quotes": raw_quotes})()

    cache_before = dict(overnight_shim_mod._CAPFLOOR_SURFACE_PERIOD_CACHE)
    try:
        raw_quotes = RawQuotes(
            [
                type("Quote", (), {"key": "CAPFLOOR/RATE_NVOL/USD/1Y/3M/0.0200", "value": 0.01})(),
                type("Quote", (), {"key": "OTHER/QUOTE", "value": 0.02})(),
            ]
        )
        snapshot = Snapshot(raw_quotes)
        period1 = overnight_shim_mod.capfloor_surface_rate_computation_period(snapshot, ccy="USD")
        period2 = overnight_shim_mod.capfloor_surface_rate_computation_period(snapshot, ccy="USD")
        assert period1 == "3M"
        assert period2 == "3M"
        assert raw_quotes.iter_count == 1
    finally:
        overnight_shim_mod._CAPFLOOR_SURFACE_PERIOD_CACHE.clear()
        overnight_shim_mod._CAPFLOOR_SURFACE_PERIOD_CACHE.update(cache_before)


def test_overnight_qlextras_cache_curve_handle_and_index_objects():
    try:
        import QuantLib as ql
    except Exception:
        pytest.skip("QuantLib is required for this cache regression")

    eval_date = ql.Date(15, 1, 2025)
    curve = type("Curve", (), {"__call__": lambda self, t: 1.0 / (1.0 + float(t))})()
    handle1 = overnight_shim_mod._curve_handle_from_curve(eval_date, curve, extra_times=[0.0, 1.0, 2.0])
    handle2 = overnight_shim_mod._curve_handle_from_curve(eval_date, curve, extra_times=[0.0, 1.0, 2.0])
    index1 = overnight_shim_mod._build_ql_overnight_index("SOFR", handle1)
    index2 = overnight_shim_mod._build_ql_overnight_index("SOFR", handle2)
    assert handle1 is handle2
    assert index1 is index2


def test_overnight_curve_handle_deduplicates_dates():
    try:
        import QuantLib as ql
    except Exception:
        pytest.skip("QuantLib is required for this cache regression")

    eval_date = ql.Date(15, 1, 2025)
    curve = type("Curve", (), {"__call__": lambda self, t: 1.0 / (1.0 + float(t))})()
    handle = overnight_shim_mod._curve_handle_from_curve(
        eval_date,
        curve,
        extra_times=[0.0, 1.0, 1.0001, 2.0],
        dates=["2025-01-15", "2026-01-15", "2026-01-15"],
        dfs=[1.0, 0.97, 0.96],
    )
    assert handle.discount(eval_date) == 1.0
    assert math.isfinite(float(handle.discount(ql.Date(15, 1, 2026))))


def test_load_swap_legs_from_portfolio_root_reuses_trade_lookup_cache():
    portfolio_root = ET.fromstring(
        """
        <Portfolio>
          <Trade id="T1">
            <TradeType>Swap</TradeType>
            <SwapData>
              <LegData>
                <LegType>Fixed</LegType>
                <Currency>EUR</Currency>
                <PaymentConvention>F</PaymentConvention>
                <DayCounter>A360</DayCounter>
                <Notionals><Notional>1000000</Notional></Notionals>
                <ScheduleData>
                  <Rules>
                    <StartDate>2025-02-10</StartDate>
                    <EndDate>2026-02-10</EndDate>
                    <Tenor>6M</Tenor>
                    <Calendar>TARGET</Calendar>
                    <Convention>F</Convention>
                  </Rules>
                </ScheduleData>
                <FixedLegData><Rates><Rate>0.02</Rate></Rates></FixedLegData>
              </LegData>
              <LegData>
                <LegType>Floating</LegType>
                <Currency>EUR</Currency>
                <PaymentConvention>F</PaymentConvention>
                <DayCounter>A360</DayCounter>
                <Notionals><Notional>1000000</Notional></Notionals>
                <ScheduleData>
                  <Rules>
                    <StartDate>2025-02-10</StartDate>
                    <EndDate>2026-02-10</EndDate>
                    <Tenor>6M</Tenor>
                    <Calendar>TARGET</Calendar>
                    <Convention>F</Convention>
                  </Rules>
                </ScheduleData>
                <FloatingLegData>
                  <Index>EUR-EURIBOR-6M</Index>
                  <FixingDays>2</FixingDays>
                  <IsInArrears>false</IsInArrears>
                  <Spreads><Spread>0.0</Spread></Spreads>
                  <Gearings><Gearing>1.0</Gearing></Gearings>
                </FloatingLegData>
              </LegData>
            </SwapData>
          </Trade>
        </Portfolio>
        """
    )

    cache_before = dict(irs_utils_mod._PORTFOLIO_TRADE_LOOKUP_CACHE)
    try:
        first = irs_utils_mod._portfolio_trade_lookup(portfolio_root)
        second = irs_utils_mod._portfolio_trade_lookup(portfolio_root)
        assert first is second
        cache_key = ("T1", "2025-02-10", "ActualActual(ISDA)")
        irs_utils_mod._SWAP_LEG_CACHE.pop(id(portfolio_root), None)
        legs = load_swap_legs_from_portfolio_root(portfolio_root, "T1", "2025-02-10")
        cached = load_swap_legs_from_portfolio_root(portfolio_root, "T1", "2025-02-10")
        assert legs["fixed_pay_time"].size > 0
        assert cached is legs
        assert cache_key in irs_utils_mod._SWAP_LEG_CACHE[id(portfolio_root)][1]
        assert len(irs_utils_mod._PORTFOLIO_TRADE_LOOKUP_CACHE) >= 1
    finally:
        irs_utils_mod._PORTFOLIO_TRADE_LOOKUP_CACHE.clear()
        irs_utils_mod._SWAP_LEG_CACHE.clear()
        irs_utils_mod._PORTFOLIO_TRADE_LOOKUP_CACHE.update(cache_before)


def test_python_lgm_adapter_reuses_cached_lgm_paths_for_same_snapshot():
    snapshot = _load_case("Examples/Legacy/Example_25/Input")
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    inputs = adapter._extract_inputs(snapshot, map_snapshot(snapshot))
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
    call_counter = {"count": 0}
    original = adapter._lgm_mod.simulate_lgm_measure

    def _counting_simulate(*args, **kwargs):
        call_counter["count"] += 1
        return original(*args, **kwargs)

    try:
        adapter._lgm_mod.simulate_lgm_measure = _counting_simulate
        first_x, first_fx = adapter._simulate_lgm_paths_cached(snapshot, inputs, model, 4, "numpy")
        second_x, second_fx = adapter._simulate_lgm_paths_cached(snapshot, inputs, model, 4, "numpy")
    finally:
        adapter._lgm_mod.simulate_lgm_measure = original

    assert call_counter["count"] == 1
    assert first_x is second_x
    assert first_fx is second_fx


def test_curve_values_caches_repeated_callable_vector_evaluation():
    class CountingCurve:
        def __init__(self):
            self.count = 0

        def __call__(self, t):
            self.count += 1
            return 1.0 / (1.0 + float(t))

    curve = CountingCurve()
    times = np.array([0.0, 1.0, 2.0, 5.0], dtype=float)
    first = irs_utils_mod.curve_values(curve, times)
    second = irs_utils_mod.curve_values(curve, times)
    assert np.allclose(first, second)
    assert curve.count == times.size


def test_aggregate_portfolio_npv_paths_reuses_cached_aggregation():
    trade_paths = {
        "T1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "T2": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
    }
    trade_to_ns = {"T1": "NS1", "T2": "NS1"}
    first = irs_utils_mod.aggregate_portfolio_npv_paths(trade_paths, trade_to_netting_set=trade_to_ns)
    second = irs_utils_mod.aggregate_portfolio_npv_paths(trade_paths, trade_to_netting_set=trade_to_ns)
    assert first is second
    assert np.allclose(first["portfolio"], np.array([[11.0, 22.0], [33.0, 44.0]], dtype=float))
    assert np.allclose(first["by_netting_set"]["NS1"], first["portfolio"])


def test_resolve_irs_pricing_backend_respects_requested_device_per_snapshot():
    import torch

    if not hasattr(torch.backends, "mps"):
        pytest.skip("mps backend is not available in this torch build")

    snapshot = _load_case("Examples/Legacy/Example_25/Input")
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    inputs = adapter._extract_inputs(snapshot, map_snapshot(snapshot))
    original_mps_available = torch.backends.mps.is_available
    try:
        torch.backends.mps.is_available = lambda: True  # type: ignore[assignment]
        cpu_backend = adapter._resolve_irs_pricing_backend(replace(inputs, torch_device="cpu"))
        mps_backend = adapter._resolve_irs_pricing_backend(replace(inputs, torch_device="mps"))
    finally:
        torch.backends.mps.is_available = original_mps_available  # type: ignore[assignment]

    assert cpu_backend is not None
    assert mps_backend is not None
    assert cpu_backend[3] == "cpu"
    assert mps_backend[3] == "mps"
    assert cpu_backend is not mps_backend
    assert len(adapter._swap_pricing_backend_cache) >= 2


def test_python_runtime_compares_example25_digital_cmsspread_with_local_ore_run():
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=("Digital_CMS_Spread",))
    try:
        proc = subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        with (case_root / "Output" / "npv.csv").open(newline="", encoding="utf-8") as handle:
            ore_npv = float(next(csv.DictReader(handle))["NPV"])

        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=4))
        py_result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id="digital-cmsspread-parity",
        )

        assert math.isclose(ore_npv, 279806.656601, rel_tol=1.0e-10, abs_tol=1.0e-8)
        assert math.isclose(float(py_result.pv_total), 284672.65625257127, rel_tol=1.0e-10, abs_tol=1.0e-8)
        assert math.isclose(float(py_result.pv_total - ore_npv), 4865.99965157127, rel_tol=1.0e-10, abs_tol=1.0e-8)
    finally:
        tmp.cleanup()


def test_python_runtime_compares_example25_cmsspread_with_local_ore_run():
    if not LOCAL_ORE_BINARY.exists():
        return
    tmp, case_root = _clone_pricing_only_case("Example_25", trade_ids=("CMS_Spread_Swap",))
    try:
        proc = subprocess.run(
            [str(LOCAL_ORE_BINARY), "Input/ore.xml"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        with (case_root / "Output" / "npv.csv").open(newline="", encoding="utf-8") as handle:
            ore_npv = float(next(csv.DictReader(handle))["NPV"])

        snapshot = XVALoader.from_files(str(case_root / "Input"), ore_file="ore.xml")
        snapshot = replace(snapshot, config=replace(snapshot.config, num_paths=4))
        py_result = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id="cmsspread-parity",
        )

        assert math.isclose(ore_npv, 328271.965698, rel_tol=1.0e-10, abs_tol=1.0e-8)
        assert math.isclose(float(py_result.pv_total), 309150.0484271799, rel_tol=1.0e-10, abs_tol=1.0e-8)
        assert math.isclose(float(py_result.pv_total - ore_npv), -19121.917270820122, rel_tol=1.0e-10, abs_tol=1.0e-8)
    finally:
        tmp.cleanup()
