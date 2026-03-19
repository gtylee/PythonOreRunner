#!/usr/bin/env python3
"""Run ORE XVA benchmark for demo FX forwards (GBP/USD 1Y, USD/CAD 2Y)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from py_ore_tools.repo_paths import default_ore_bin, local_parity_artifacts_root, require_engine_repo_root

REPO_ROOT = require_engine_repo_root()
EXAMPLES_INPUT = REPO_ROOT / "Examples" / "Input"
EXPOSURE_INPUT = REPO_ROOT / "Examples" / "Exposure" / "Input"
ORE_BIN_DEFAULT = default_ore_bin()


@dataclass(frozen=True)
class FxFwdCase:
    case_id: str
    pair: str
    bought_ccy: str
    sold_ccy: str
    bought_amount: float
    sold_amount: float
    value_date: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ore-bin", type=Path, default=ORE_BIN_DEFAULT)
    p.add_argument(
        "--output-root",
        type=Path,
        default=local_parity_artifacts_root() / "fxfwd_ore_xva_benchmark",
    )
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _portfolio_xml(case: FxFwdCase) -> str:
    return f"""<?xml version=\"1.0\"?>
<Portfolio>
  <Trade id=\"{case.case_id}\">
    <TradeType>FxForward</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>CPTY_A</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <FxForwardData>
      <ValueDate>{case.value_date}</ValueDate>
      <BoughtCurrency>{case.bought_ccy}</BoughtCurrency>
      <BoughtAmount>{case.bought_amount:.6f}</BoughtAmount>
      <SoldCurrency>{case.sold_ccy}</SoldCurrency>
      <SoldAmount>{case.sold_amount:.6f}</SoldAmount>
    </FxForwardData>
  </Trade>
</Portfolio>
"""


def _simulation_xml(domestic: str, foreign: str, index_dom: str, index_for: str, samples: int, seed: int) -> str:
    return f"""<?xml version=\"1.0\"?>
<Simulation>
  <Parameters>
    <Discretization>Exact</Discretization>
    <Grid>81,3M</Grid>
    <Calendar>{domestic},{foreign}</Calendar>
    <Sequence>SobolBrownianBridge</Sequence>
    <Scenario>Simple</Scenario>
    <Seed>{seed}</Seed>
    <Samples>{samples}</Samples>
    <Ordering>Steps</Ordering>
    <DirectionIntegers>JoeKuoD7</DirectionIntegers>
  </Parameters>
  <CrossAssetModel>
    <DomesticCcy>{domestic}</DomesticCcy>
    <Currencies>
      <Currency>{domestic}</Currency>
      <Currency>{foreign}</Currency>
    </Currencies>
    <BootstrapTolerance>0.0001</BootstrapTolerance>
    <Measure>LGM</Measure>
    <InterestRateModels>
      <LGM ccy=\"{domestic}\">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility>
          <Calibrate>N</Calibrate>
          <VolatilityType>Hagan</VolatilityType>
          <ParamType>Piecewise</ParamType>
          <TimeGrid>1.0,2.0,3.0,5.0,7.0,10.0</TimeGrid>
          <InitialValue>0.01,0.01,0.01,0.01,0.01,0.01,0.01</InitialValue>
        </Volatility>
        <Reversion>
          <Calibrate>N</Calibrate>
          <ReversionType>HullWhite</ReversionType>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>0.03</InitialValue>
        </Reversion>
        <ParameterTransformation><ShiftHorizon>0.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
      </LGM>
      <LGM ccy=\"{foreign}\">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility>
          <Calibrate>N</Calibrate>
          <VolatilityType>Hagan</VolatilityType>
          <ParamType>Piecewise</ParamType>
          <TimeGrid>1.0,2.0,3.0,5.0,7.0,10.0</TimeGrid>
          <InitialValue>0.01,0.01,0.01,0.01,0.01,0.01,0.01</InitialValue>
        </Volatility>
        <Reversion>
          <Calibrate>N</Calibrate>
          <ReversionType>HullWhite</ReversionType>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>0.03</InitialValue>
        </Reversion>
        <ParameterTransformation><ShiftHorizon>0.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
      </LGM>
    </InterestRateModels>
    <ForeignExchangeModels>
      <CrossCcyLGM foreignCcy=\"{foreign}\">
        <DomesticCcy>{domestic}</DomesticCcy>
        <CalibrationType>Bootstrap</CalibrationType>
        <Sigma>
          <Calibrate>N</Calibrate>
          <ParamType>Piecewise</ParamType>
          <TimeGrid>1.0,2.0,3.0,5.0,7.0,10.0</TimeGrid>
          <InitialValue>0.10,0.10,0.10,0.10,0.10,0.10,0.10</InitialValue>
        </Sigma>
        <CalibrationOptions><Expiries>1Y,2Y,3Y,5Y,10Y</Expiries><Strikes/></CalibrationOptions>
      </CrossCcyLGM>
    </ForeignExchangeModels>
    <InstantaneousCorrelations>
      <Correlation factor1=\"IR:{domestic}\" factor2=\"IR:{foreign}\">0.3</Correlation>
      <Correlation factor1=\"IR:{domestic}\" factor2=\"FX:{foreign}{domestic}\">0.0</Correlation>
      <Correlation factor1=\"IR:{foreign}\" factor2=\"FX:{foreign}{domestic}\">0.0</Correlation>
    </InstantaneousCorrelations>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>{domestic}</BaseCurrency>
    <Currencies><Currency>{domestic}</Currency><Currency>{foreign}</Currency></Currencies>
    <YieldCurves>
      <Configuration>
        <Tenors>3M,6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y</Tenors>
        <Interpolation>LogLinear</Interpolation>
        <Extrapolation>Y</Extrapolation>
      </Configuration>
    </YieldCurves>
    <Indices><Index>{index_dom}</Index><Index>{index_for}</Index></Indices>
    <DefaultCurves><Names/><Tenors>6M,1Y,2Y,5Y,10Y</Tenors></DefaultCurves>
  </Market>
</Simulation>
"""


def _ore_xml(
    output_dir: Path,
    market_file: Path,
    todaysmarket_file: Path,
    portfolio_file: Path,
    simulation_file: Path,
    base_ccy: str,
    curveconfig_file: Path,
    conventions_file: Path,
) -> str:
    return f"""<?xml version=\"1.0\"?>
<ORE>
  <Setup>
    <Parameter name=\"asofDate\">2016-02-05</Parameter>
    <Parameter name=\"inputPath\">{portfolio_file.parent.as_posix()}</Parameter>
    <Parameter name=\"outputPath\">{output_dir.as_posix()}</Parameter>
    <Parameter name=\"logFile\">log.txt</Parameter>
    <Parameter name=\"logMask\">31</Parameter>
    <Parameter name=\"marketDataFile\">{market_file.as_posix()}</Parameter>
    <Parameter name=\"fixingDataFile\">{(EXAMPLES_INPUT / 'fixings_20160205.txt').as_posix()}</Parameter>
    <Parameter name=\"implyTodaysFixings\">Y</Parameter>
    <Parameter name=\"curveConfigFile\">{curveconfig_file.as_posix()}</Parameter>
    <Parameter name=\"conventionsFile\">{conventions_file.as_posix()}</Parameter>
    <Parameter name=\"marketConfigFile\">{todaysmarket_file.as_posix()}</Parameter>
    <Parameter name=\"pricingEnginesFile\">{(EXAMPLES_INPUT / 'pricingengine.xml').as_posix()}</Parameter>
    <Parameter name=\"portfolioFile\">{portfolio_file.as_posix()}</Parameter>
    <Parameter name=\"observationModel\">None</Parameter>
    <Parameter name=\"continueOnError\">false</Parameter>
    <Parameter name=\"calendarAdjustment\">{(EXAMPLES_INPUT / 'calendaradjustment.xml').as_posix()}</Parameter>
  </Setup>
  <Markets>
    <Parameter name=\"pricing\">libor</Parameter>
    <Parameter name=\"simulation\">libor</Parameter>
  </Markets>
  <Analytics>
    <Analytic type=\"npv\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"baseCurrency\">{base_ccy}</Parameter><Parameter name=\"outputFileName\">npv.csv</Parameter></Analytic>
    <Analytic type=\"cashflow\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"outputFileName\">flows.csv</Parameter></Analytic>
    <Analytic type=\"curves\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"configuration\">default</Parameter><Parameter name=\"grid\">240,1M</Parameter><Parameter name=\"outputFileName\">curves.csv</Parameter></Analytic>
    <Analytic type=\"simulation\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"simulationConfigFile\">{simulation_file.as_posix()}</Parameter><Parameter name=\"pricingEnginesFile\">{(EXAMPLES_INPUT / 'pricingengine.xml').as_posix()}</Parameter><Parameter name=\"baseCurrency\">{base_ccy}</Parameter><Parameter name=\"cubeFile\">cube.csv.gz</Parameter><Parameter name=\"aggregationScenarioDataFileName\">scenariodata.csv.gz</Parameter></Analytic>
    <Analytic type=\"xva\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"useXvaRunner\">N</Parameter><Parameter name=\"csaFile\">{(EXPOSURE_INPUT / 'netting.xml').as_posix()}</Parameter><Parameter name=\"cubeFile\">cube.csv.gz</Parameter><Parameter name=\"scenarioFile\">scenariodata.csv.gz</Parameter><Parameter name=\"baseCurrency\">{base_ccy}</Parameter><Parameter name=\"exposureProfiles\">Y</Parameter><Parameter name=\"exposureProfilesByTrade\">Y</Parameter><Parameter name=\"cva\">Y</Parameter><Parameter name=\"dva\">N</Parameter><Parameter name=\"fva\">N</Parameter><Parameter name=\"rawCubeOutputFile\">rawcube.csv</Parameter><Parameter name=\"netCubeOutputFile\">netcube.csv</Parameter></Analytic>
  </Analytics>
</ORE>
"""


def _augment_todaysmarket_for_usdcad(src: Path, dst: Path) -> None:
    root = ET.parse(src).getroot()
    changed = False

    fxspots = root.find("./FxSpots[@id='default']")
    if fxspots is not None and not any(n.attrib.get("pair", "") == "USDCAD" for n in fxspots.findall("./FxSpot")):
        n = ET.SubElement(fxspots, "FxSpot")
        n.attrib["pair"] = "USDCAD"
        n.text = "FX/USD/CAD"
        changed = True

    swv = root.find("./SwaptionVolatilities[@id='default']")
    if swv is not None and not any(n.attrib.get("currency", "") == "CAD" for n in swv.findall("./SwaptionVolatility")):
        n = ET.SubElement(swv, "SwaptionVolatility")
        n.attrib["currency"] = "CAD"
        # Reuse USD swaption surface so CAM builder has a valid short/long swap index source.
        n.text = "SwaptionVolatility/USD/USD_SW_N"
        changed = True

    if changed:
        dst.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    else:
        shutil.copy2(src, dst)


def _augment_conventions_for_cad(src: Path, dst: Path) -> None:
    root = ET.parse(src).getroot()
    ids = {n.findtext("./Id", default="").strip() for n in root.findall("./SwapIndex")}
    changed = False
    if "CAD-CMS-1Y" not in ids:
        n = ET.SubElement(root, "SwapIndex")
        i = ET.SubElement(n, "Id")
        i.text = "CAD-CMS-1Y"
        c = ET.SubElement(n, "Conventions")
        c.text = "CAD-3M-SWAP-CONVENTIONS-1Y"
        changed = True
    if "CAD-CMS-30Y" not in ids:
        n = ET.SubElement(root, "SwapIndex")
        i = ET.SubElement(n, "Id")
        i.text = "CAD-CMS-30Y"
        c = ET.SubElement(n, "Conventions")
        c.text = "CAD-3M-SWAP-CONVENTIONS"
        changed = True
    if changed:
        dst.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    else:
        shutil.copy2(src, dst)


def _augment_curveconfig_for_cad(src: Path, dst: Path) -> None:
    root = ET.parse(src).getroot()
    swv_parent = root.find("./SwaptionVolatilities")
    changed = False
    if swv_parent is None:
        swv_parent = ET.SubElement(root, "SwaptionVolatilities")
        changed = True

    curve_ids = {n.findtext("./CurveId", default="").strip() for n in swv_parent.findall("./SwaptionVolatility")}
    if "CAD_SW_N" not in curve_ids:
        # Clone USD as template and override CAD-specific identifiers.
        usd = None
        for n in swv_parent.findall("./SwaptionVolatility"):
            if (n.findtext("./CurveId") or "").strip() == "USD_SW_N":
                usd = n
                break
        if usd is not None:
            cad = ET.fromstring(ET.tostring(usd, encoding="unicode"))
            cad.find("./CurveId").text = "CAD_SW_N"
            cad.find("./CurveDescription").text = "CAD normal swaption volatilities"
            cad.find("./ShortSwapIndexBase").text = "CAD-CMS-1Y"
            cad.find("./SwapIndexBase").text = "CAD-CMS-30Y"
            swv_parent.append(cad)
            changed = True

    if changed:
        dst.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    else:
        shutil.copy2(src, dst)


def _load_ore_cva(xva_csv: Path, cpty: str = "CPTY_A") -> float:
    with open(xva_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get("NettingSetId", "") == cpty and row.get(tid_key, "") == "":
                return float(row["CVA"])
    raise ValueError(f"aggregate CVA row not found in {xva_csv}")


def _load_trade_epe_stats(exposure_csv: Path) -> tuple[float, float]:
    epe = []
    with open(exposure_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            epe.append(float(r["EPE"]))
    if not epe:
        return 0.0, 0.0
    return max(epe), sum(epe) / len(epe)


def main() -> None:
    args = _parse_args()
    if not args.ore_bin.exists():
        raise FileNotFoundError(args.ore_bin)

    out_root = args.output_root
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cad_market = REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "multiccy_benchmark_final" / "shared" / "market_20160205_flat_with_cad.txt"
    if not cad_market.exists():
        raise FileNotFoundError(f"missing CAD market file: {cad_market}")

    cad_tm_base = REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "multiccy_benchmark_final" / "shared" / "todaysmarket_with_cad.xml"
    if not cad_tm_base.exists():
        raise FileNotFoundError(f"missing CAD todaysmarket file: {cad_tm_base}")
    tm_usdcad = out_root / "shared" / "todaysmarket_usdcad.xml"
    _augment_todaysmarket_for_usdcad(cad_tm_base, tm_usdcad)

    cases = [
        FxFwdCase("FXFWD_GBPUSD_1Y", "GBP/USD", "GBP", "USD", 10_000_000.0, 12_850_000.0, "2017-03-01"),
        FxFwdCase("FXFWD_USDCAD_2Y", "USD/CAD", "USD", "CAD", 10_000_000.0, 13_600_000.0, "2018-03-01"),
    ]

    rows = []
    for c in cases:
        case_dir = out_root / c.case_id
        inp = case_dir / "Input"
        out = case_dir / "Output"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)

        conventions_file = EXAMPLES_INPUT / "conventions.xml"
        curveconfig_file = EXAMPLES_INPUT / "curveconfig.xml"

        if c.pair == "GBP/USD":
            market_file = EXAMPLES_INPUT / "market_20160205.txt"
            tm_file = EXAMPLES_INPUT / "todaysmarket.xml"
            dom, foreign = "USD", "GBP"
            idx_dom, idx_for = "USD-LIBOR-3M", "GBP-LIBOR-6M"
        else:
            market_file = cad_market
            tm_file = tm_usdcad
            dom, foreign = "CAD", "USD"
            idx_dom, idx_for = "CAD-CDOR-3M", "USD-LIBOR-3M"
            conventions_file = inp / "conventions_cad.xml"
            curveconfig_file = inp / "curveconfig_cad.xml"
            _augment_conventions_for_cad(EXAMPLES_INPUT / "conventions.xml", conventions_file)
            _augment_curveconfig_for_cad(EXAMPLES_INPUT / "curveconfig.xml", curveconfig_file)

        portfolio_xml = inp / "portfolio.xml"
        simulation_xml = inp / "simulation.xml"
        ore_xml = inp / "ore.xml"
        _write(portfolio_xml, _portfolio_xml(c))
        _write(simulation_xml, _simulation_xml(dom, foreign, idx_dom, idx_for, args.samples, args.seed))
        _write(
            ore_xml,
            _ore_xml(
                out,
                market_file,
                tm_file,
                portfolio_xml,
                simulation_xml,
                base_ccy=dom,
                curveconfig_file=curveconfig_file,
                conventions_file=conventions_file,
            ),
        )

        t0 = perf_counter()
        cp = subprocess.run([str(args.ore_bin), str(ore_xml)], cwd=str(case_dir), capture_output=True, text=True, check=False)
        runtime = perf_counter() - t0

        (case_dir / "ore_stdout.log").write_text(cp.stdout, encoding="utf-8")
        (case_dir / "ore_stderr.log").write_text(cp.stderr, encoding="utf-8")

        status = "ok" if cp.returncode == 0 else "ore_error"
        ore_cva = float("nan")
        peak_epe = float("nan")
        avg_epe = float("nan")

        xva_csv = out / "xva.csv"
        exp_csv = out / f"exposure_trade_{c.case_id}.csv"
        if status == "ok" and xva_csv.exists() and exp_csv.exists():
            ore_cva = _load_ore_cva(xva_csv, cpty="CPTY_A")
            peak_epe, avg_epe = _load_trade_epe_stats(exp_csv)
        else:
            status = "missing_xva_output" if status == "ok" else status

        rows.append(
            {
                "case_id": c.case_id,
                "pair": c.pair,
                "status": status,
                "ore_seconds": runtime,
                "ore_cva": ore_cva,
                "peak_epe": peak_epe,
                "avg_epe": avg_epe,
                "output_dir": str(out),
            }
        )

    (out_root / "fxfwd_xva_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with open(out_root / "fxfwd_xva_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "pair", "status", "ore_seconds", "ore_cva", "peak_epe", "avg_epe", "output_dir"])
        for r in rows:
            w.writerow([r[k] for k in ["case_id", "pair", "status", "ore_seconds", "ore_cva", "peak_epe", "avg_epe", "output_dir"]])

    for r in rows:
        print(
            f"{r['case_id']} ({r['pair']}): status={r['status']}, "
            f"CVA={r['ore_cva']}, peakEPE={r['peak_epe']}, avgEPE={r['avg_epe']}, t={r['ore_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
