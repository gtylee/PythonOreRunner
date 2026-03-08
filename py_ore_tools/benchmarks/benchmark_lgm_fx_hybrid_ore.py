#!/usr/bin/env python3
"""Build and optionally run ORE-based LGM+FX hybrid parity benchmark cases."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ore_parity_artifacts import (
    CaseMetadata,
    build_case_layout,
    write_case_manifest,
    write_command_log,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLES_INPUT = REPO_ROOT / "Examples" / "Input"
EXPOSURE_INPUT = REPO_ROOT / "Examples" / "Exposure" / "Input"
ORE_BIN_DEFAULT = REPO_ROOT / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore"
COMPARE_SCRIPT = Path(__file__).resolve().parents[1] / "demos" / "compare_ore_python_lgm_fx.py"


@dataclass(frozen=True)
class BenchmarkCase:
    product: str
    market: str
    ccy_or_pair: str
    maturity: str
    conv: str

    @property
    def case_id(self) -> str:
        return f"{self.market}_{self.product}_{self.ccy_or_pair}_{self.maturity}_{self.conv}".replace("/", "_")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "Tools" / "PythonOreRunner" / "parity_artifacts" / "lgm_fx_hybrid_benchmark")
    p.add_argument("--ore-bin", type=Path, default=ORE_BIN_DEFAULT)
    p.add_argument("--ore-samples", type=int, default=2000)
    p.add_argument("--python-paths", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-modes", default="fixed,calibrated")
    p.add_argument("--execute", action="store_true", default=False)
    p.add_argument("--max-cases", type=int, default=0)
    return p.parse_args()


def _cases() -> list[BenchmarkCase]:
    out: list[BenchmarkCase] = []
    markets = ["flat", "full"]
    maturities = ["1Y", "5Y", "10Y"]
    convs = ["A", "B"]

    for m in markets:
        for ccy in ["EUR", "USD", "GBP", "CAD"]:
            for ten in maturities:
                for conv in convs:
                    out.append(BenchmarkCase("IRS", m, ccy, ten, conv))
        for pair in ["GBP/USD", "USD/CAD"]:
            for ten in maturities:
                for conv in convs:
                    out.append(BenchmarkCase("FXFWD", m, pair, ten, conv))
                    out.append(BenchmarkCase("XCCY", m, pair, ten, conv))
    return out


def _tenor_years(tenor: str) -> int:
    if not tenor.endswith("Y"):
        raise ValueError(f"unsupported tenor '{tenor}'")
    return int(tenor[:-1])


def _start_end(tenor: str) -> tuple[str, str]:
    # Keep as-of aligned with existing ORE examples.
    y = 2016 + _tenor_years(tenor)
    return "20160301", f"{y}0301"


def _market_file(market: str) -> Path:
    if market == "flat":
        return EXAMPLES_INPUT / "market_20160205_flat.txt"
    if market == "full":
        return EXAMPLES_INPUT / "market_20160205.txt"
    raise ValueError(market)


def _tm_file() -> Path:
    return EXAMPLES_INPUT / "todaysmarket.xml"


def _ore_xml(output_dir: Path, market_file: Path, tm_file: Path, portfolio_xml: Path, sim_xml: Path, base_ccy: str, xva_active: bool) -> str:
    return f"""<?xml version=\"1.0\"?>
<ORE>
  <Setup>
    <Parameter name=\"asofDate\">2016-02-05</Parameter>
    <Parameter name=\"inputPath\">{portfolio_xml.parent.as_posix()}</Parameter>
    <Parameter name=\"outputPath\">{output_dir.as_posix()}</Parameter>
    <Parameter name=\"marketDataFile\">{market_file.as_posix()}</Parameter>
    <Parameter name=\"fixingDataFile\">{(EXAMPLES_INPUT / 'fixings_20160205.txt').as_posix()}</Parameter>
    <Parameter name=\"curveConfigFile\">{(EXAMPLES_INPUT / 'curveconfig.xml').as_posix()}</Parameter>
    <Parameter name=\"conventionsFile\">{(EXAMPLES_INPUT / 'conventions.xml').as_posix()}</Parameter>
    <Parameter name=\"marketConfigFile\">{tm_file.as_posix()}</Parameter>
    <Parameter name=\"pricingEnginesFile\">{(EXAMPLES_INPUT / 'pricingengine.xml').as_posix()}</Parameter>
    <Parameter name=\"portfolioFile\">{portfolio_xml.as_posix()}</Parameter>
    <Parameter name=\"continueOnError\">true</Parameter>
  </Setup>
  <Markets>
    <Parameter name=\"lgmcalibration\">libor</Parameter>
    <Parameter name=\"pricing\">libor</Parameter>
    <Parameter name=\"simulation\">libor</Parameter>
  </Markets>
  <Analytics>
    <Analytic type=\"npv\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"baseCurrency\">{base_ccy}</Parameter><Parameter name=\"outputFileName\">npv.csv</Parameter></Analytic>
    <Analytic type=\"cashflow\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"outputFileName\">flows.csv</Parameter></Analytic>
    <Analytic type=\"curves\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"configuration\">default</Parameter><Parameter name=\"grid\">240,1M</Parameter><Parameter name=\"outputFileName\">curves.csv</Parameter></Analytic>
    <Analytic type=\"simulation\"><Parameter name=\"active\">Y</Parameter><Parameter name=\"simulationConfigFile\">{sim_xml.as_posix()}</Parameter><Parameter name=\"pricingEnginesFile\">{(EXAMPLES_INPUT / 'pricingengine.xml').as_posix()}</Parameter><Parameter name=\"baseCurrency\">{base_ccy}</Parameter><Parameter name=\"cubeFile\">cube.csv.gz</Parameter><Parameter name=\"aggregationScenarioDataFileName\">scenariodata.csv.gz</Parameter></Analytic>
    <Analytic type=\"xva\"><Parameter name=\"active\">{'Y' if xva_active else 'N'}</Parameter><Parameter name=\"useXvaRunner\">N</Parameter><Parameter name=\"csaFile\">{(EXPOSURE_INPUT / 'netting.xml').as_posix()}</Parameter><Parameter name=\"cubeFile\">cube.csv.gz</Parameter><Parameter name=\"scenarioFile\">scenariodata.csv.gz</Parameter><Parameter name=\"baseCurrency\">{base_ccy}</Parameter><Parameter name=\"exposureProfiles\">Y</Parameter><Parameter name=\"cva\">Y</Parameter><Parameter name=\"dva\">N</Parameter><Parameter name=\"fva\">N</Parameter></Analytic>
  </Analytics>
</ORE>
"""


def _sim_xml_single_ccy(ccy: str, index: str, samples: int, seed: int, calibrated: bool) -> str:
    cal = "Y" if calibrated else "N"
    return f"""<?xml version=\"1.0\"?>
<Simulation>
  <Parameters>
    <Discretization>Exact</Discretization>
    <Grid>121,1M</Grid>
    <Calendar>TARGET</Calendar>
    <Sequence>SobolBrownianBridge</Sequence>
    <Scenario>Simple</Scenario>
    <Seed>{seed}</Seed>
    <Samples>{samples}</Samples>
    <Ordering>Steps</Ordering>
    <DirectionIntegers>JoeKuoD7</DirectionIntegers>
  </Parameters>
  <CrossAssetModel>
    <DomesticCcy>{ccy}</DomesticCcy>
    <Currencies><Currency>{ccy}</Currency></Currencies>
    <BootstrapTolerance>0.0001</BootstrapTolerance>
    <Measure>LGM</Measure>
    <InterestRateModels>
      <LGM ccy=\"{ccy}\"><CalibrationType>Bootstrap</CalibrationType>
        <Volatility><Calibrate>{cal}</Calibrate><VolatilityType>Hagan</VolatilityType><ParamType>Piecewise</ParamType><TimeGrid>1,2,3,5,7,10</TimeGrid><InitialValue>0.01,0.01,0.01,0.01,0.01,0.01,0.01</InitialValue></Volatility>
        <Reversion><Calibrate>{cal}</Calibrate><ReversionType>HullWhite</ReversionType><ParamType>Constant</ParamType><TimeGrid/><InitialValue>0.03</InitialValue></Reversion>
        <ParameterTransformation><ShiftHorizon>0.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
      </LGM>
    </InterestRateModels>
    <InstantaneousCorrelations/>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>{ccy}</BaseCurrency>
    <Currencies><Currency>{ccy}</Currency></Currencies>
    <YieldCurves><Configuration><Tenors>3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,15Y,20Y</Tenors><Interpolation>LogLinear</Interpolation><Extrapolation>Y</Extrapolation></Configuration></YieldCurves>
    <Indices><Index>{index}</Index></Indices>
    <DefaultCurves><Names/><Tenors>6M,1Y,2Y,5Y,10Y</Tenors></DefaultCurves>
  </Market>
</Simulation>
"""


def _portfolio_irs(trade_id: str, ccy: str, idx: str, tenor: str, conv: str) -> str:
    start, end = _start_end(tenor)
    fixed_tenor = "1Y" if conv == "A" else "6M"
    float_tenor = "6M" if conv == "A" else "3M"
    fixed_dc = "30/360" if conv == "A" else "A365"
    float_dc = "A360" if conv == "A" else "A365"
    fixed_pc = "MF" if conv == "A" else "F"
    float_pc = "MF" if conv == "A" else "F"
    return f"""<?xml version=\"1.0\"?>
<Portfolio>
  <Trade id=\"{trade_id}\"><TradeType>Swap</TradeType>
    <Envelope><CounterParty>CPTY_A</CounterParty><NettingSetId>CPTY_A</NettingSetId><AdditionalFields/></Envelope>
    <SwapData>
      <LegData><LegType>Fixed</LegType><Payer>false</Payer><Currency>{ccy}</Currency><Notionals><Notional>10000000</Notional></Notionals><DayCounter>{fixed_dc}</DayCounter><PaymentConvention>{fixed_pc}</PaymentConvention><FixedLegData><Rates><Rate>0.02</Rate></Rates></FixedLegData><ScheduleData><Rules><StartDate>{start}</StartDate><EndDate>{end}</EndDate><Tenor>{fixed_tenor}</Tenor><Calendar>TARGET</Calendar><Convention>{fixed_pc}</Convention><TermConvention>{fixed_pc}</TermConvention><Rule>Forward</Rule><EndOfMonth/></Rules></ScheduleData></LegData>
      <LegData><LegType>Floating</LegType><Payer>true</Payer><Currency>{ccy}</Currency><Notionals><Notional>10000000</Notional></Notionals><DayCounter>{float_dc}</DayCounter><PaymentConvention>{float_pc}</PaymentConvention><FloatingLegData><Index>{idx}</Index><Spreads><Spread>0.0</Spread></Spreads><IsInArrears>false</IsInArrears><FixingDays>2</FixingDays></FloatingLegData><ScheduleData><Rules><StartDate>{start}</StartDate><EndDate>{end}</EndDate><Tenor>{float_tenor}</Tenor><Calendar>TARGET</Calendar><Convention>{float_pc}</Convention><TermConvention>{float_pc}</TermConvention><Rule>Forward</Rule><EndOfMonth/></Rules></ScheduleData></LegData>
    </SwapData>
  </Trade>
</Portfolio>
"""


def _portfolio_fxfwd(trade_id: str, pair: str, tenor: str) -> str:
    start, end = _start_end(tenor)
    base, quote = pair.split("/")
    return f"""<?xml version=\"1.0\"?>
<Portfolio>
  <Trade id=\"{trade_id}\"><TradeType>FxForward</TradeType>
    <Envelope><CounterParty>CPTY_A</CounterParty><NettingSetId>CPTY_A</NettingSetId><AdditionalFields/></Envelope>
    <FxForwardData>
      <ValueDate>{start}</ValueDate>
      <Maturity>{end}</Maturity>
      <BoughtCurrency>{base}</BoughtCurrency>
      <BoughtAmount>10000000</BoughtAmount>
      <SoldCurrency>{quote}</SoldCurrency>
      <SoldAmount>13000000</SoldAmount>
    </FxForwardData>
  </Trade>
</Portfolio>
"""


def _portfolio_xccy(trade_id: str, pair: str, tenor: str, conv: str) -> str:
    start, end = _start_end(tenor)
    c1, c2 = pair.split("/")
    tnr = "6M" if conv == "A" else "3M"
    idx1 = "GBP-LIBOR-6M" if c1 == "GBP" else "USD-LIBOR-6M"
    idx2 = "USD-LIBOR-6M" if c2 == "USD" else "CAD-CDOR-3M"
    return f"""<?xml version=\"1.0\"?>
<Portfolio>
  <Trade id=\"{trade_id}\"><TradeType>Swap</TradeType>
    <Envelope><CounterParty>CPTY_A</CounterParty><NettingSetId>CPTY_A</NettingSetId><AdditionalFields/></Envelope>
    <SwapData>
      <LegData><LegType>Floating</LegType><Payer>false</Payer><Currency>{c1}</Currency><Notionals><Notional>10000000</Notional></Notionals><DayCounter>A360</DayCounter><PaymentConvention>MF</PaymentConvention><FloatingLegData><Index>{idx1}</Index><Spreads><Spread>0.0</Spread></Spreads><FixingDays>2</FixingDays></FloatingLegData><ScheduleData><Rules><StartDate>{start}</StartDate><EndDate>{end}</EndDate><Tenor>{tnr}</Tenor><Calendar>TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule><EndOfMonth/></Rules></ScheduleData></LegData>
      <LegData><LegType>Floating</LegType><Payer>true</Payer><Currency>{c2}</Currency><Notionals><Notional>13000000</Notional></Notionals><DayCounter>A360</DayCounter><PaymentConvention>MF</PaymentConvention><FloatingLegData><Index>{idx2}</Index><Spreads><Spread>0.0</Spread></Spreads><FixingDays>2</FixingDays></FloatingLegData><ScheduleData><Rules><StartDate>{start}</StartDate><EndDate>{end}</EndDate><Tenor>{tnr}</Tenor><Calendar>TARGET</Calendar><Convention>MF</Convention><TermConvention>MF</TermConvention><Rule>Forward</Rule><EndOfMonth/></Rules></ScheduleData></LegData>
    </SwapData>
  </Trade>
</Portfolio>
"""


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
    t0 = perf_counter()
    cp = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    return cp.returncode, cp.stdout, cp.stderr, perf_counter() - t0


def main() -> None:
    args = _parse_args()
    modes = [x.strip() for x in args.run_modes.split(",") if x.strip()]
    if not modes:
        raise ValueError("--run-modes must not be empty")

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases = _cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    results = []
    for mode in modes:
        calibrated = mode == "calibrated"
        for case in cases:
            layout = build_case_layout(args.output_root, case.case_id, mode)
            product = case.product

            if product == "IRS":
                ccy = case.ccy_or_pair
                base_ccy = ccy
                idx = {
                    "EUR": "EUR-EURIBOR-6M",
                    "USD": "USD-LIBOR-6M",
                    "GBP": "GBP-LIBOR-6M",
                    "CAD": "CAD-CDOR-3M",
                }[ccy]
                portfolio_txt = _portfolio_irs(case.case_id, ccy, idx, case.maturity, case.conv)
                sim_txt = _sim_xml_single_ccy(ccy, idx, args.ore_samples, args.seed, calibrated)
                product_json = None
                model_ccys = (ccy,)
                fx_pairs = tuple()
                indices = (idx,)
            elif product == "FXFWD":
                pair = case.ccy_or_pair
                base, quote = pair.split("/")
                base_ccy = quote
                portfolio_txt = _portfolio_fxfwd(case.case_id, pair, case.maturity)
                sim_txt = _sim_xml_single_ccy(base_ccy, f"{base_ccy}-LIBOR-6M" if base_ccy != "CAD" else "CAD-CDOR-3M", args.ore_samples, args.seed, calibrated)
                product_json = {
                    "trade_id": case.case_id,
                    "pair": pair,
                    "notional_base": 10_000_000,
                    "strike": 1.3,
                    "spot0": 1.3,
                    "maturity": float(_tenor_years(case.maturity)),
                    "discount_columns": {"USD": "USD-LIBOR-3M", "GBP": "GBP-LIBOR-6M", "CAD": "CAD-CDOR-3M", "EUR": "EUR-EURIBOR-6M"},
                }
                model_ccys = (base, quote)
                fx_pairs = (pair,)
                indices = tuple()
            else:
                pair = case.ccy_or_pair
                c1, c2 = pair.split("/")
                base_ccy = c2
                portfolio_txt = _portfolio_xccy(case.case_id, pair, case.maturity, case.conv)
                sim_txt = _sim_xml_single_ccy(base_ccy, f"{base_ccy}-LIBOR-6M" if base_ccy != "CAD" else "CAD-CDOR-3M", args.ore_samples, args.seed, calibrated)
                # Simple generated leg spec for Python side; ORE trade remains source of cashflow truth in parity runs.
                y = float(_tenor_years(case.maturity))
                n = int(y * 2)
                pays = [(i + 1) * 0.5 for i in range(n)]
                starts = [i * 0.5 for i in range(n)]
                product_json = {
                    "domestic_ccy": c2,
                    "spot0": 1.3,
                    "discount_columns": {"USD": "USD-LIBOR-3M", "GBP": "GBP-LIBOR-6M", "CAD": "CAD-CDOR-3M", "EUR": "EUR-EURIBOR-6M"},
                    "forward_columns": {"USD": "USD-LIBOR-6M", "GBP": "GBP-LIBOR-6M", "CAD": "CAD-CDOR-3M", "EUR": "EUR-EURIBOR-6M"},
                    "leg1": {"ccy": c1, "pay_time": pays, "start_time": starts, "end_time": pays, "accrual": [0.5] * n, "notional": [10_000_000] * n, "spread": [0.0] * n, "sign": [1.0] * n},
                    "leg2": {"ccy": c2, "pay_time": pays, "start_time": starts, "end_time": pays, "accrual": [0.5] * n, "notional": [13_000_000] * n, "spread": [0.0] * n, "sign": [-1.0] * n},
                }
                model_ccys = (c1, c2)
                fx_pairs = (pair,)
                indices = tuple()

            ore_xml = layout.trades / "ore.xml"
            sim_xml = layout.trades / "simulation.xml"
            portfolio_xml = layout.trades / "portfolio.xml"
            market_file = _market_file(case.market)
            tm_file = _tm_file()
            ore_txt = _ore_xml(layout.root / "Output", market_file, tm_file, portfolio_xml, sim_xml, base_ccy, xva_active=True)

            portfolio_xml.write_text(portfolio_txt, encoding="utf-8")
            sim_xml.write_text(sim_txt, encoding="utf-8")
            ore_xml.write_text(ore_txt, encoding="utf-8")
            if product_json is not None:
                (layout.trades / "product.json").write_text(json.dumps(product_json, indent=2, sort_keys=True), encoding="utf-8")

            meta = CaseMetadata(
                case_id=case.case_id,
                run_mode=mode,
                asof_date="2016-02-05",
                base_ccy=base_ccy,
                model_ccys=tuple(model_ccys),
                fx_pairs=tuple(fx_pairs),
                indices=tuple(indices),
                products=(product,),
                convention_profile=case.conv,
                ore_samples=args.ore_samples,
                python_paths=args.python_paths,
                seed=args.seed,
                notes=f"market={case.market}; tenor={case.maturity}",
            )
            write_case_manifest(layout, meta)

            ore_cmd = [str(args.ore_bin), str(ore_xml)]
            py_cmd = [
                "python3",
                str(COMPARE_SCRIPT),
                "--product",
                {"IRS": "irs_single", "FXFWD": "fx_forward", "XCCY": "xccy_float_float"}[product],
                "--ore-output-dir",
                str(layout.root / "Output"),
                "--simulation-xml",
                str(sim_xml),
                "--portfolio-xml",
                str(portfolio_xml),
                "--trade-id",
                case.case_id,
                "--artifact-dir",
                str(layout.exposure),
                "--paths",
                str(args.python_paths),
                "--seed",
                str(args.seed),
                "--todaysmarket-xml",
                str(tm_file),
                "--market-data-file",
                str(market_file),
                "--discount-column",
                "USD-LIBOR-3M" if base_ccy == "USD" else ("EUR-EURIBOR-6M" if base_ccy == "EUR" else ("GBP-LIBOR-6M" if base_ccy == "GBP" else "CAD-CDOR-3M")),
                "--model-ccy",
                base_ccy,
            ]
            if (layout.trades / "product.json").exists():
                py_cmd.extend(["--product-json", str(layout.trades / "product.json")])

            write_command_log(layout, ore_cmd, py_cmd)

            status = "prepared"
            ore_seconds = 0.0
            py_seconds = 0.0
            if args.execute:
                rc, out, err, ore_seconds = _run(ore_cmd, layout.root)
                (layout.perf / "ore_stdout.log").write_text(out, encoding="utf-8")
                (layout.perf / "ore_stderr.log").write_text(err, encoding="utf-8")
                if rc == 0:
                    rc2, out2, err2, py_seconds = _run(py_cmd, layout.root)
                    (layout.perf / "py_stdout.log").write_text(out2, encoding="utf-8")
                    (layout.perf / "py_stderr.log").write_text(err2, encoding="utf-8")
                    status = "ok" if rc2 == 0 else "python_error"
                else:
                    status = "ore_error"

            perf_payload = {
                "status": status,
                "ore_seconds": ore_seconds,
                "python_seconds": py_seconds,
                "total_seconds": ore_seconds + py_seconds,
            }
            (layout.perf / "timing.json").write_text(json.dumps(perf_payload, indent=2, sort_keys=True), encoding="utf-8")

            results.append(
                {
                    "case_id": case.case_id,
                    "mode": mode,
                    "status": status,
                    "product": product,
                    "market": case.market,
                    "ccy_or_pair": case.ccy_or_pair,
                    "maturity": case.maturity,
                    "conv": case.conv,
                    "ore_seconds": ore_seconds,
                    "python_seconds": py_seconds,
                }
            )

    (args.output_root / "benchmark_results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    # Basic performance summary for quick reporting.
    ores = [r["ore_seconds"] for r in results if r["ore_seconds"] > 0.0]
    pys = [r["python_seconds"] for r in results if r["python_seconds"] > 0.0]
    summary = {
        "cases": len(results),
        "executed": args.execute,
        "ore_seconds": {
            "mean": float(sum(ores) / len(ores)) if ores else 0.0,
            "max": float(max(ores)) if ores else 0.0,
        },
        "python_seconds": {
            "mean": float(sum(pys) / len(pys)) if pys else 0.0,
            "max": float(max(pys)) if pys else 0.0,
        },
    }
    (args.output_root / "benchmark_perf_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Prepared {len(results)} cases under {args.output_root}")
    if args.execute:
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"Executed: {ok}/{len(results)} successful")


if __name__ == "__main__":
    main()
