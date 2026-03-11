#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SingleCurveComparisonResult:
    case_name: str
    trade_id: str
    shift_size: float
    base_npv_single_curve: float
    up_npv_single_curve: float
    direct_quote_bump_change: float
    ore_sensitivity_rows: list[dict[str, str]]
    ore_par_sensitivity_rows: list[dict[str, str]]
    ore_jacobi_rows: list[dict[str, str]]
    ore_jacobi_inverse_rows: list[dict[str, str]]
    run_root: str

    def to_dict(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "trade_id": self.trade_id,
            "shift_size": self.shift_size,
            "base_npv_single_curve": self.base_npv_single_curve,
            "up_npv_single_curve": self.up_npv_single_curve,
            "direct_quote_bump_change": self.direct_quote_bump_change,
            "ore_sensitivity_rows": self.ore_sensitivity_rows,
            "ore_par_sensitivity_rows": self.ore_par_sensitivity_rows,
            "ore_jacobi_rows": self.ore_jacobi_rows,
            "ore_jacobi_inverse_rows": self.ore_jacobi_inverse_rows,
            "run_root": self.run_root,
        }


class SingleCurveBermudanOREComparison:
    def __init__(
        self,
        case_name: str = "berm_200bp",
        trade_id: str = "BermSwp",
        shift_size: float = 1.0e-4,
        output_root: Path | None = None,
    ) -> None:
        self.case_name = case_name
        self.trade_id = trade_id
        self.shift_size = float(shift_size)
        self.output_root = output_root or (TOOLS_DIR / "parity_artifacts" / "bermudan_singlecurve_compare")
        self.case_input = TOOLS_DIR / "parity_artifacts" / "bermudan_method_compare" / case_name / "Input"
        self.run_root = self.output_root / case_name

    def run(self) -> SingleCurveComparisonResult:
        ore_bin = self._locate_ore_exe()
        self._prepare_run_root()
        self._write_market_files()

        self._run_ore(
            ore_bin,
            self._write_ore_xml("ore_sensi.xml", "market_base.txt", "sensi", include_sensi=True),
        )
        self._run_ore(
            ore_bin,
            self._write_ore_xml("ore_base.xml", "market_base.txt", "base", include_sensi=False),
        )
        self._run_ore(
            ore_bin,
            self._write_ore_xml("ore_up.xml", "market_up.txt", "up", include_sensi=False),
        )

        base_npv = self._read_trade_npv(self.run_root / "Output" / "base" / "npv.csv")
        up_npv = self._read_trade_npv(self.run_root / "Output" / "up" / "npv.csv")

        result = SingleCurveComparisonResult(
            case_name=self.case_name,
            trade_id=self.trade_id,
            shift_size=self.shift_size,
            base_npv_single_curve=base_npv,
            up_npv_single_curve=up_npv,
            direct_quote_bump_change=up_npv - base_npv,
            ore_sensitivity_rows=self._read_csv_dicts(self.run_root / "Output" / "sensi" / "sensitivity.csv"),
            ore_par_sensitivity_rows=self._read_csv_dicts(self.run_root / "Output" / "sensi" / "parsensitivity.csv"),
            ore_jacobi_rows=self._read_csv_dicts(self.run_root / "Output" / "sensi" / "jacobi.csv"),
            ore_jacobi_inverse_rows=self._read_csv_dicts(self.run_root / "Output" / "sensi" / "jacobi_inverse.csv"),
            run_root=str(self.run_root),
        )
        summary_path = self.run_root / "summary.json"
        summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    def _locate_ore_exe(self) -> Path:
        for candidate in (
            REPO_ROOT / "build" / "App" / "ore",
            REPO_ROOT / "build" / "ore" / "App" / "ore",
            REPO_ROOT / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore",
        ):
            if candidate.exists():
                return candidate
        raise FileNotFoundError("ORE executable not found under the local build tree")

    def _prepare_run_root(self) -> None:
        if self.run_root.exists():
            shutil.rmtree(self.run_root)
        (self.run_root / "Input").mkdir(parents=True, exist_ok=True)
        (self.run_root / "Output").mkdir(parents=True, exist_ok=True)
        self._write_single_curve_curveconfig(self.run_root / "Input" / "curveconfig_single.xml")
        self._write_text(self.run_root / "Input" / "todaysmarket_single.xml", self._todaysmarket_xml())
        self._write_text(self.run_root / "Input" / "simulation_sensitivity.xml", self._simulation_sensitivity_xml())
        self._write_text(self.run_root / "Input" / "sensitivity.xml", self._sensitivity_xml())

    def _write_single_curve_curveconfig(self, dest: Path) -> None:
        src = REPO_ROOT / "Examples" / "Input" / "curveconfig.xml"
        root = ET.parse(src).getroot()
        for yc in root.findall(".//YieldCurve"):
            if (yc.findtext("CurveId", "") or "").strip() == "EUR6M":
                dc = yc.find("DiscountCurve")
                if dc is not None:
                    dc.text = "EUR6M"
                break
        ET.ElementTree(root).write(dest, encoding="utf-8", xml_declaration=True)

    def _write_market_files(self) -> None:
        lines = (self.case_input / "market_20160205_flat_fixed_fxfwd.txt").read_text(encoding="utf-8").splitlines()
        (self.run_root / "Input" / "market_base.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        bumped: list[str] = []
        for line in lines:
            if line.strip().startswith("20160205 IR_SWAP/RATE/EUR/2D/6M/10Y "):
                parts = line.split()
                parts[-1] = f"{float(parts[-1]) + self.shift_size:.10f}"
                line = " ".join(parts)
            bumped.append(line)
        (self.run_root / "Input" / "market_up.txt").write_text("\n".join(bumped) + "\n", encoding="utf-8")

    def _write_ore_xml(self, xml_name: str, market_file: str, output_subdir: str, include_sensi: bool) -> Path:
        root = ET.parse(self.case_input / "ore_classic.xml").getroot()
        setup = root.find("Setup")
        assert setup is not None
        for node in setup.findall("Parameter"):
            name = node.attrib.get("name", "")
            if name == "inputPath":
                node.text = str(self.run_root / "Input")
            elif name == "outputPath":
                node.text = str(self.run_root / "Output" / output_subdir)
            elif name == "portfolioFile":
                node.text = str(self.case_input / "portfolio.xml")
            elif name == "pricingEnginesFile":
                node.text = str(self.case_input / "pricingengine.xml")
            elif name == "marketDataFile":
                node.text = str(self.run_root / "Input" / market_file)
            elif name == "fixingDataFile":
                node.text = str(REPO_ROOT / "Examples" / "Input" / "fixings_20160205.txt")
            elif name == "curveConfigFile":
                node.text = str(self.run_root / "Input" / "curveconfig_single.xml")
            elif name == "conventionsFile":
                node.text = str(REPO_ROOT / "Examples" / "Input" / "conventions.xml")
            elif name == "marketConfigFile":
                node.text = str(self.run_root / "Input" / "todaysmarket_single.xml")
        analytics = root.find("Analytics")
        assert analytics is not None
        for child in list(analytics):
            analytics.remove(child)
        analytic = ET.SubElement(analytics, "Analytic", {"type": "npv"})
        for key, value in (("active", "Y"), ("baseCurrency", "EUR"), ("outputFileName", "npv.csv")):
            ET.SubElement(analytic, "Parameter", {"name": key}).text = value
        if include_sensi:
            analytic = ET.SubElement(analytics, "Analytic", {"type": "sensitivity"})
            for key, value in {
                "active": "Y",
                "marketConfigFile": str(self.run_root / "Input" / "simulation_sensitivity.xml"),
                "sensitivityConfigFile": str(self.run_root / "Input" / "sensitivity.xml"),
                "pricingEnginesFile": str(self.case_input / "pricingengine.xml"),
                "scenarioOutputFile": "scenario.csv",
                "sensitivityOutputFile": "sensitivity.csv",
                "parSensitivity": "Y",
                "parSensitivityOutputFile": "parsensitivity.csv",
                "outputJacobi": "Y",
                "jacobiOutputFile": "jacobi.csv",
                "jacobiInverseOutputFile": "jacobi_inverse.csv",
                "outputSensitivityThreshold": "0.0",
                "recalibrateModels": "Y",
            }.items():
                ET.SubElement(analytic, "Parameter", {"name": key}).text = value
        xml_path = self.run_root / "Input" / xml_name
        ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path

    def _run_ore(self, ore_bin: Path, xml_path: Path) -> None:
        cp = subprocess.run([str(ore_bin), str(xml_path)], cwd=str(self.run_root), capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            raise RuntimeError(f"ORE run failed with return code {cp.returncode}\n{cp.stdout}\n{cp.stderr}")

    def _read_trade_npv(self, path: Path) -> float:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            tid_key = "#TradeId" if "#TradeId" in (reader.fieldnames or []) else "TradeId"
            for row in reader:
                if row.get(tid_key, "").strip() == self.trade_id:
                    return float(row["NPV"])
        raise ValueError(f"Trade {self.trade_id} not found in {path}")

    def _read_csv_dicts(self, path: Path) -> list[dict[str, str]]:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _todaysmarket_xml(self) -> str:
        return """<?xml version="1.0"?>
<TodaysMarket>
  <Configuration id="default"><DiscountingCurvesId>single</DiscountingCurvesId><IndexForwardingCurvesId>single</IndexForwardingCurvesId><SwaptionVolatilitiesId>default</SwaptionVolatilitiesId><SwapIndexCurvesId>single</SwapIndexCurvesId></Configuration>
  <Configuration id="xois_eur"><DiscountingCurvesId>single</DiscountingCurvesId><IndexForwardingCurvesId>single</IndexForwardingCurvesId><SwaptionVolatilitiesId>default</SwaptionVolatilitiesId><SwapIndexCurvesId>single</SwapIndexCurvesId></Configuration>
  <Configuration id="collateral_inccy"><DiscountingCurvesId>single</DiscountingCurvesId><IndexForwardingCurvesId>single</IndexForwardingCurvesId><SwaptionVolatilitiesId>default</SwaptionVolatilitiesId><SwapIndexCurvesId>single</SwapIndexCurvesId></Configuration>
  <DiscountingCurves id="single"><DiscountingCurve currency="EUR">Yield/EUR/EUR6M</DiscountingCurve></DiscountingCurves>
  <IndexForwardingCurves id="single"><Index name="EUR-EONIA">Yield/EUR/EUR6M</Index><Index name="EUR-EURIBOR-3M">Yield/EUR/EUR6M</Index><Index name="EUR-EURIBOR-6M">Yield/EUR/EUR6M</Index></IndexForwardingCurves>
  <SwapIndexCurves id="single"><SwapIndex name="EUR-CMS-1Y"><Discounting>EUR-EURIBOR-6M</Discounting></SwapIndex><SwapIndex name="EUR-CMS-30Y"><Discounting>EUR-EURIBOR-6M</Discounting></SwapIndex></SwapIndexCurves>
  <SwaptionVolatilities id="default"><SwaptionVolatility currency="EUR">SwaptionVolatility/EUR/EUR_SW_N</SwaptionVolatility></SwaptionVolatilities>
</TodaysMarket>
"""

    def _simulation_sensitivity_xml(self) -> str:
        return """<Simulation>
  <Market>
    <BaseCurrency>EUR</BaseCurrency>
    <Currencies><Currency>EUR</Currency><Currency>USD</Currency></Currencies>
    <YieldCurves><Configuration curve=""><Tenors>6M,1Y,2Y,3Y,5Y,7Y,10Y,15Y,20Y,30Y</Tenors><Interpolation>LogLinear</Interpolation><Extrapolation>true</Extrapolation></Configuration></YieldCurves>
    <Indices><Index>EUR-EONIA</Index><Index>EUR-EURIBOR-3M</Index><Index>EUR-EURIBOR-6M</Index></Indices>
    <SwapIndices><SwapIndex><Name>EUR-CMS-1Y</Name><DiscountingIndex>EUR-EURIBOR-6M</DiscountingIndex></SwapIndex><SwapIndex><Name>EUR-CMS-30Y</Name><DiscountingIndex>EUR-EURIBOR-6M</DiscountingIndex></SwapIndex></SwapIndices>
    <SwaptionVolatilities><Simulate>true</Simulate><ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay><Currencies><Currency>EUR</Currency></Currencies><Expiries>1Y,2Y,3Y,5Y,10Y,15Y,20Y,30Y</Expiries><Terms>1Y,2Y,3Y,5Y,10Y,15Y,20Y,30Y</Terms><DayCounters><DayCounter ccy="">A365</DayCounter></DayCounters></SwaptionVolatilities>
  </Market>
</Simulation>
"""

    def _sensitivity_xml(self) -> str:
        return """<?xml version="1.0"?>
<SensitivityAnalysis>
  <DiscountCurves/>
  <IndexCurves>
    <IndexCurve index="EUR-EURIBOR-6M">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>0.0001000000</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>10Y</ShiftTenors>
      <ParConversion>
        <Instruments>IRS</Instruments>
        <SingleCurve>true</SingleCurve>
        <Conventions>
          <Convention id="DEP">EUR-DEPOSIT</Convention>
          <Convention id="IRS">EUR-6M-SWAP-CONVENTIONS</Convention>
        </Conventions>
      </ParConversion>
    </IndexCurve>
  </IndexCurves>
  <YieldCurves/>
  <FxSpots/>
  <FxVolatilities/>
  <SwaptionVolatilities/>
  <CapFloorVolatilities/>
  <CDSVolatilities/>
  <CreditCurves/>
  <EquitySpots/>
  <EquityVolatilities/>
  <ZeroInflationIndexCurves/>
  <YYInflationIndexCurves/>
  <BaseCorrelations/>
  <SecuritySpreads/>
  <Correlations/>
  <CrossGammaFilter/>
</SensitivityAnalysis>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real single-curve Bermudan ORE direct-bump vs sensitivity comparison.")
    parser.add_argument("--case-name", default="berm_200bp")
    parser.add_argument("--trade-id", default="BermSwp")
    parser.add_argument("--shift-size", type=float, default=1.0e-4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TOOLS_DIR / "parity_artifacts" / "bermudan_singlecurve_compare",
    )
    args = parser.parse_args()

    runner = SingleCurveBermudanOREComparison(
        case_name=args.case_name,
        trade_id=args.trade_id,
        shift_size=args.shift_size,
        output_root=args.output_root,
    )
    result = runner.run()

    print(f"single-curve base npv={result.base_npv_single_curve:.6f}")
    print(f"single-curve direct quote bump change={result.direct_quote_bump_change:.6f}")
    if result.ore_sensitivity_rows:
        print(f"ore sensitivity delta={float(result.ore_sensitivity_rows[0]['Delta']):.6f}")
    if result.ore_par_sensitivity_rows:
        print(f"ore par sensitivity delta={float(result.ore_par_sensitivity_rows[0]['Delta']):.6f}")
    if result.ore_jacobi_inverse_rows:
        print(f"ore jacobi inverse dz/dc={float(result.ore_jacobi_inverse_rows[0]['dz/dc']):.12f}")
    print(f"wrote {Path(result.run_root) / 'summary.json'}")


if __name__ == "__main__":
    main()
