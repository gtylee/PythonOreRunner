#!/usr/bin/env python3
"""Generate and run a one-trade USD snapshot case with a tiny sensitivity grid."""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pythonore.apps import ore_snapshot_cli
from pythonore.io.loader import XVALoader
from pythonore.repo_paths import find_ore_bin
from pythonore.runtime.runtime import XVAEngine, PythonLgmAdapter
from example_ore_snapshot_usd_all_rates_products import (
    _basis_fedfunds_libor_3m_trade_xml,
    _basis_libor_3m_6m_trade_xml,
    _basis_libor_3m_sifma_trade_xml,
    _basis_sofr_3m_libor_3m_trade_xml,
    _basis_sofr_3m_sifma_trade_xml,
    _simulation_xml as _full_simulation_xml,
)


PRODUCTS_INPUT = REPO_ROOT / "Examples" / "Products" / "Input"
DEFAULT_CASE_ROOT = REPO_ROOT / "Examples" / "Generated" / "USD_OneTradeLimitedSensi"
ASOF_DATE = "2025-02-10"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a one-trade USD case with a tiny sensitivity grid.")
    parser.add_argument("--trade-count", type=int, default=300)
    parser.add_argument("--trade-mix", choices=("irs", "mixed"), default="irs")
    parser.add_argument("--sensi-tenors", default="6M,1Y,18M,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y,25Y,30Y,35Y,40Y")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--paths", type=int, default=96)
    parser.add_argument(
        "--lgm-param-source",
        choices=("auto", "calibration_xml", "simulation_xml", "ore"),
        default="simulation_xml",
    )
    parser.add_argument("--compare-ore", action="store_true")
    return parser.parse_args()


def _ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace the generated case.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _irs_trade_xml(idx: int) -> str:
    suffix = f"{idx:04d}"
    fixed_rate = 0.04 + 0.0005 * (idx - 1)
    return f"""  <Trade id="IRS_USD_{suffix}">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>30/360</DayCounter>
        <PaymentConvention>F</PaymentConvention>
        <FixedLegData><Rates><Rate>{fixed_rate:.6f}</Rate></Rates></FixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>6M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <Notionals><Notional>10000000</Notional></Notionals>
        <DayCounter>A360</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>USD-LIBOR-3M</Index>
          <Spreads><Spread>0.0000</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{ASOF_DATE}</StartDate>
            <EndDate>2035-02-10</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>USD</Calendar>
            <Convention>MF</Convention>
            <TermConvention>MF</TermConvention>
            <Rule>Forward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>"""


def _portfolio_xml(trade_count: int) -> str:
    trades = "\n".join(_irs_trade_xml(i) for i in range(1, trade_count + 1))
    return f"<Portfolio>\n{trades}\n</Portfolio>\n"


def _mixed_portfolio_xml(trade_count: int) -> str:
    builders = (
        ("IRS", lambda i: _irs_trade_xml(i)),
        ("BASIS_L36", lambda i: _basis_libor_3m_6m_trade_xml(f"{i:04d}")),
        ("BASIS_L3S", lambda i: _basis_libor_3m_sifma_trade_xml(f"{i:04d}")),
        ("BASIS_FF3", lambda i: _basis_fedfunds_libor_3m_trade_xml(f"{i:04d}")),
        ("BASIS_S3L", lambda i: _basis_sofr_3m_libor_3m_trade_xml(f"{i:04d}")),
        ("BASIS_S3S", lambda i: _basis_sofr_3m_sifma_trade_xml(f"{i:04d}")),
    )
    trades: list[str] = []
    for idx in range(1, trade_count + 1):
        _, builder = builders[(idx - 1) % len(builders)]
        trades.append(builder(idx))
    return f"<Portfolio>\n{'\n'.join(trades)}\n</Portfolio>\n"


def _ore_xml() -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">{ASOF_DATE}</Parameter>
    <Parameter name="baseCurrency">USD</Parameter>
    <Parameter name="marketDataFile">{PRODUCTS_INPUT / "marketdata.csv"}</Parameter>
    <Parameter name="fixingDataFile">{PRODUCTS_INPUT / "fixings.csv"}</Parameter>
    <Parameter name="curveConfigFile">{PRODUCTS_INPUT / "curveconfig.xml"}</Parameter>
    <Parameter name="conventionsFile">{PRODUCTS_INPUT / "conventions.xml"}</Parameter>
    <Parameter name="marketConfigFile">{PRODUCTS_INPUT / "todaysmarket.xml"}</Parameter>
    <Parameter name="pricingEnginesFile">{PRODUCTS_INPUT / "pricingengine.xml"}</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
    <Parameter name="referenceDataFile">{PRODUCTS_INPUT / "referencedata.xml"}</Parameter>
    <Parameter name="scriptLibrary">{PRODUCTS_INPUT / "scriptlibrary.xml"}</Parameter>
    <Parameter name="outputPath">Output</Parameter>
    <Parameter name="observationModel">None</Parameter>
    <Parameter name="continueOnError">false</Parameter>
    <Parameter name="buildFailedTrades">true</Parameter>
  </Setup>
  <Markets>
    <Parameter name="lgmcalibration">default</Parameter>
    <Parameter name="fxcalibration">default</Parameter>
    <Parameter name="pricing">default</Parameter>
    <Parameter name="simulation">default</Parameter>
  </Markets>
  <Analytics>
    <Analytic type="npv">
      <Parameter name="active">Y</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
      <Parameter name="outputFileName">npv.csv</Parameter>
    </Analytic>
    <Analytic type="cashflow">
      <Parameter name="active">Y</Parameter>
      <Parameter name="outputFileName">flows.csv</Parameter>
    </Analytic>
    <Analytic type="curves">
      <Parameter name="active">Y</Parameter>
      <Parameter name="configuration">default</Parameter>
      <Parameter name="grid">240,1M</Parameter>
      <Parameter name="outputFileName">curves.csv</Parameter>
    </Analytic>
    <Analytic type="simulation">
      <Parameter name="active">N</Parameter>
      <Parameter name="simulationConfigFile">simulation.xml</Parameter>
      <Parameter name="pricingEnginesFile">{PRODUCTS_INPUT / "pricingengine.xml"}</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
    </Analytic>
    <Analytic type="sensitivity">
      <Parameter name="active">Y</Parameter>
      <Parameter name="sensitivityConfigFile">sensitivity.xml</Parameter>
      <Parameter name="sensitivityOutputFile">sensitivity.csv</Parameter>
      <Parameter name="scenarioOutputFile">scenario.csv</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def _sensitivity_xml(sensi_tenors: str, *, trade_mix: str) -> str:
    index_blocks = [
        """  <IndexCurve index="USD-LIBOR-3M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </IndexCurve>""",
    ]
    if trade_mix == "mixed":
        index_blocks.extend(
            [
                """  <IndexCurve index="USD-LIBOR-6M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </IndexCurve>""",
                """  <IndexCurve index="USD-FedFunds">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </IndexCurve>""",
                """  <IndexCurve index="USD-SIFMA">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </IndexCurve>""",
                """  <IndexCurve index="USD-SOFR-3M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </IndexCurve>""",
            ]
        )
    return """<?xml version="1.0"?>
<SensitivityAnalysis>
  <DiscountCurves>
    <DiscountCurve ccy="USD">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{tenors}</ShiftTenors>
    </DiscountCurve>
  </DiscountCurves>
  <IndexCurves>
{index_curves}
  </IndexCurves>
</SensitivityAnalysis>
""".format(tenors=sensi_tenors, index_curves="\n".join(block.format(tenors=sensi_tenors) for block in index_blocks))


def _write_case(case_root: Path, *, trade_count: int, trade_mix: str, sensi_tenors: str) -> Path:
    input_dir = case_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "ore.xml").write_text(_ore_xml(), encoding="utf-8")
    portfolio_xml = _portfolio_xml(trade_count) if trade_mix == "irs" else _mixed_portfolio_xml(trade_count)
    (input_dir / "portfolio.xml").write_text(portfolio_xml, encoding="utf-8")
    (input_dir / "simulation.xml").write_text(_full_simulation_xml(), encoding="utf-8")
    (input_dir / "netting.xml").write_text("<NettingSetDefinitions/>\n", encoding="utf-8")
    (input_dir / "sensitivity.xml").write_text(_sensitivity_xml(sensi_tenors, trade_mix=trade_mix), encoding="utf-8")
    return input_dir / "ore.xml"


def _artifact_case_dir(artifact_root: Path, case_root: Path) -> Path:
    return artifact_root / case_root.name


def _read_portfolio_npv(npv_csv: Path) -> float:
    with npv_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        total = 0.0
        for row in reader:
            trade_id = str(row.get("#TradeId", "")).strip()
            if trade_id == "PORTFOLIO":
                return float(row["NPV(Base)"])
            total += float(row["NPV(Base)"])
    return total


def _prepare_ore_compare_case(case_root: Path) -> Path:
    compare_root = Path(tempfile.mkdtemp(prefix="ore_compare_", dir=str(case_root)))
    input_dir = compare_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    src_input = case_root / "Input"
    for name in ("portfolio.xml", "simulation.xml", "sensitivity.xml", "netting.xml"):
        src = src_input / name
        if src.exists():
            shutil.copy2(src, input_dir / name)

    for name in (
        "marketdata.csv",
        "fixings.csv",
        "curveconfig.xml",
        "conventions.xml",
        "todaysmarket.xml",
        "pricingengine.xml",
        "referencedata.xml",
        "scriptlibrary.xml",
    ):
        shutil.copy2(PRODUCTS_INPUT / name, input_dir / name)

    tree = ET.parse(src_input / "ore.xml")
    root = tree.getroot()
    setup = root.find("Setup")
    if setup is None:
        raise ValueError("Generated ore.xml is missing <Setup>")
    params = {
        p.get("name"): p
        for p in setup.findall("Parameter")
        if p.get("name")
    }

    def set_setup(name: str, value: str) -> None:
        node = params.get(name)
        if node is None:
            node = ET.SubElement(setup, "Parameter", {"name": name})
            params[name] = node
        node.text = value

    set_setup("inputPath", "Input")
    set_setup("outputPath", "Output")
    set_setup("logFile", "log.txt")
    set_setup("logMask", "255")
    set_setup("marketDataFile", "marketdata.csv")
    set_setup("fixingDataFile", "fixings.csv")
    set_setup("implyTodaysFixings", "N")
    set_setup("curveConfigFile", "curveconfig.xml")
    set_setup("conventionsFile", "conventions.xml")
    set_setup("marketConfigFile", "todaysmarket.xml")
    set_setup("pricingEnginesFile", "pricingengine.xml")
    set_setup("portfolioFile", "portfolio.xml")
    set_setup("referenceDataFile", "referencedata.xml")
    set_setup("scriptLibrary", "scriptlibrary.xml")

    markets = root.find("Markets")
    if markets is not None and not any(p.get("name") == "infcalibration" for p in markets.findall("Parameter")):
        ET.SubElement(markets, "Parameter", {"name": "infcalibration"}).text = "default"

    analytics = root.find("Analytics")
    if analytics is not None:
        for analytic in analytics.findall("Analytic"):
            analytic_type = str(analytic.get("type", "")).lower()
            active = analytic.find("./Parameter[@name='active']")
            if active is None:
                active = ET.SubElement(analytic, "Parameter", {"name": "active"})
            active.text = "Y" if analytic_type in {"npv", "cashflow", "curves"} else "N"
            if analytic_type == "npv":
                extra = analytic.find("./Parameter[@name='additionalResults']")
                if extra is None:
                    extra = ET.SubElement(analytic, "Parameter", {"name": "additionalResults"})
                extra.text = "Y"
                prec = analytic.find("./Parameter[@name='additionalResultsReportPrecision']")
                if prec is None:
                    prec = ET.SubElement(analytic, "Parameter", {"name": "additionalResultsReportPrecision"})
                prec.text = "12"

    ET.indent(tree, space="  ")
    tree.write(input_dir / "ore.xml", encoding="utf-8", xml_declaration=True)
    return compare_root


def _run_ore_compare(case_root: Path, artifact_root: Path) -> int:
    ore_bin = find_ore_bin()
    if ore_bin is None:
        print("ore_compare    : skipped (ORE binary not found)")
        return 0

    compare_root = _prepare_ore_compare_case(case_root)
    ore_xml = compare_root / "Input" / "ore.xml"
    cp = subprocess.run(
        [str(ore_bin), str(ore_xml)],
        cwd=str(compare_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        stderr = (cp.stderr or "").strip()
        stdout = (cp.stdout or "").strip()
        raise RuntimeError(
            "ORE compare run failed: "
            + (stderr.splitlines()[-1] if stderr else stdout.splitlines()[-1] if stdout else f"exit {cp.returncode}")
        )

    ore_npv_csv = compare_root / "Output" / "npv.csv"
    if not ore_npv_csv.exists():
        raise FileNotFoundError(f"ORE compare npv.csv not found: {ore_npv_csv}")
    ore_npv = _read_portfolio_npv(ore_npv_csv)

    snap = XVALoader.from_files(str(compare_root / "Input"), ore_file="ore.xml")
    params = dict(snap.config.params)
    params["python.use_ore_output_curves"] = "Y"
    params["python.use_ore_flow_amounts_t0"] = "Y"
    snap = replace(snap, config=replace(snap.config, analytics=("NPV",), params=params))
    py_result = XVAEngine(adapter=PythonLgmAdapter(fallback_to_swig=False)).create_session(snap).run(return_cubes=False)
    py_npv = float(py_result.pv_total)
    py_provenance = dict(py_result.metadata.get("input_provenance", {}))

    abs_diff = abs(py_npv - ore_npv)
    rel_diff = abs_diff / max(abs(ore_npv), 1.0)

    print(f"ore_compare    : {compare_root}")
    print(f"python_npv     : {py_npv:.12f}")
    print(f"ore_npv        : {ore_npv:.12f}")
    print(f"npv_abs_diff   : {abs_diff:.12f}")
    print(f"npv_rel_diff   : {rel_diff:.6%}")
    print(f"py_provenance  : {py_provenance}")
    return 0


def main() -> int:
    args = _parse_args()
    case_root = args.case_root.resolve()
    artifact_root = (args.artifact_root or (case_root / "artifacts")).resolve()
    _ensure_clean_dir(case_root, overwrite=args.overwrite)
    if args.trade_count <= 0:
        raise ValueError("--trade-count must be > 0")
    ore_xml = _write_case(case_root, trade_count=args.trade_count, trade_mix=args.trade_mix, sensi_tenors=args.sensi_tenors)
    print(f"case_root      : {case_root}")
    print(f"ore_xml        : {ore_xml}")
    print(f"artifact_root  : {artifact_root}")
    print(f"trade_count    : {args.trade_count}")
    print(f"trade_mix      : {args.trade_mix}")
    tenor_count = len([t for t in str(args.sensi_tenors).split(",") if t.strip()])
    curve_count = 2 if args.trade_mix == "irs" else 5
    print(f"sensi_factors  : {curve_count * tenor_count} ({args.trade_mix} benchmark across {args.sensi_tenors})")
    argv = [
        str(ore_xml),
        "--price",
        "--sensi",
        "--sensi-progress",
        "--sensi-metric",
        "NPV",
        "--paths",
        str(args.paths),
        "--lgm-param-source",
        str(args.lgm_param_source),
        "--skip-input-validation",
        "--output-root",
        str(artifact_root),
    ]
    rc = ore_snapshot_cli.main(argv)
    if rc != 0:
        return rc
    if args.compare_ore:
        return _run_ore_compare(case_root, artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
