import io
import json
import shutil
import sys
import tempfile
import unittest
import csv
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from py_ore_tools import ore_snapshot_cli

REAL_CASE_XML = (
    TOOLS_DIR
    / "parity_artifacts"
    / "multiccy_benchmark_final"
    / "cases"
    / "flat_EUR_5Y_A"
    / "Input"
    / "ore.xml"
)
BOND_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_18" / "Input" / "ore.xml"
CALLABLE_CASE_XML = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_callable_bond.xml"


class TestOreSnapshotCli(unittest.TestCase):
    @staticmethod
    def _real_case_buffers() -> tuple[dict[str, str], dict[str, str]]:
        input_files = {
            path.name: path.read_text(encoding="utf-8")
            for path in REAL_CASE_XML.parent.iterdir()
            if path.is_file()
        }
        output_dir = REAL_CASE_XML.parents[1] / "Output"
        output_files = {}
        for path in output_dir.iterdir():
            if path.is_file() and path.suffix in {".xml", ".csv", ".txt"}:
                output_files[path.name] = path.read_text(encoding="utf-8")
        return input_files, output_files

    def test_run_case_from_buffers_returns_object_result(self):
        input_files, output_files = self._real_case_buffers()
        result = ore_snapshot_cli.run_case_from_buffers(
            ore_snapshot_cli.BufferCaseInputs(input_files=input_files, output_files=output_files),
            ore_snapshot_cli.PurePythonRunOptions(price=True, xva=True, paths=32),
        )
        self.assertEqual(result.summary["trade_id"], "SWAP_EUR_5Y_A_flat")
        self.assertIn("pricing", result.summary)
        self.assertIn("xva", result.summary)
        self.assertTrue(result.summary["diagnostics"]["sample_count_mismatch"])
        self.assertIn("npv.csv", result.ore_output_files)
        self.assertIn("xva.csv", result.ore_output_files)
        self.assertTrue(result.report_markdown)
        self.assertTrue(result.comparison_rows)
        self.assertTrue(result.input_validation_rows)

    def test_run_case_from_buffers_python_engine_returns_python_only_summary(self):
        input_files, output_files = self._real_case_buffers()
        result = ore_snapshot_cli.run_case_from_buffers(
            ore_snapshot_cli.BufferCaseInputs(input_files=input_files, output_files=output_files),
            ore_snapshot_cli.PurePythonRunOptions(engine="python", price=True, xva=True, paths=32),
        )
        self.assertEqual(result.summary["diagnostics"]["engine"], "python")
        self.assertIn("py_t0_npv", result.summary["pricing"])
        self.assertNotIn("ore_t0_npv", result.summary["pricing"])
        self.assertIn("py_cva", result.summary["xva"])
        self.assertNotIn("ore_cva", result.summary["xva"])
        self.assertEqual(result.comparison_rows, [])
        self.assertIsNone(result.summary["parity"])

    def test_run_case_from_buffers_ore_engine_returns_reference_summary(self):
        input_files, output_files = self._real_case_buffers()
        result = ore_snapshot_cli.run_case_from_buffers(
            ore_snapshot_cli.BufferCaseInputs(input_files=input_files, output_files=output_files),
            ore_snapshot_cli.PurePythonRunOptions(engine="ore", price=True, xva=True),
        )
        self.assertEqual(result.summary["diagnostics"]["engine"], "ore_reference")
        self.assertIn("ore_t0_npv", result.summary["pricing"])
        self.assertNotIn("py_t0_npv", result.summary["pricing"])
        self.assertIn("ore_cva", result.summary["xva"])
        self.assertNotIn("py_cva", result.summary["xva"])
        self.assertEqual(result.comparison_rows, [])
        self.assertIn("npv.csv", result.ore_output_files)

    def test_ore_snapshot_app_wrapper_runs_from_strings(self):
        input_files, output_files = self._real_case_buffers()
        app = ore_snapshot_cli.OreSnapshotApp.from_strings(
            input_files=input_files,
            output_files=output_files,
            options=ore_snapshot_cli.PurePythonRunOptions(engine="python", price=True, xva=True, paths=16),
        )
        result = app.run()
        self.assertEqual(result.summary["trade_id"], "SWAP_EUR_5Y_A_flat")
        self.assertEqual(result.summary["diagnostics"]["engine"], "python")

    def test_run_case_from_buffers_requires_ore_xml(self):
        with self.assertRaises(ValueError):
            ore_snapshot_cli.run_case_from_buffers(
                ore_snapshot_cli.BufferCaseInputs(input_files={"portfolio.xml": "<Portfolio />"})
            )

    def test_version_flag_matches_ore_shape(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ore_snapshot_cli.main(["-v"])
        self.assertEqual(rc, 0)
        self.assertIn("ORE version", out.getvalue())

    def test_hash_flag_matches_ore_shape(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ore_snapshot_cli.main(["-h"])
        self.assertEqual(rc, 0)
        self.assertIn("Git hash", out.getvalue())

    def test_requires_positional_ore_xml_for_normal_run(self):
        with self.assertRaises(SystemExit):
            ore_snapshot_cli.main([])

    def test_example_mode_runs_without_ore_xml(self):
        with patch("py_ore_tools.ore_snapshot_cli.benchmark_lgm_torch.main", return_value=0) as bench:
            rc = ore_snapshot_cli.main(["--example", "lgm_torch", "--example-path-counts", "2000", "4000"])
        self.assertEqual(rc, 0)
        bench.assert_called_once_with(
            [
                "--paths",
                "2000",
                "4000",
                "--repeats",
                "2",
                "--warmup",
                "1",
                "--seed",
                "42",
                "--devices",
                "cpu",
                "gpu",
            ]
        )

    def test_example_numpy_backend_omits_torch_devices(self):
        with patch("py_ore_tools.ore_snapshot_cli.benchmark_lgm_torch.main", return_value=0) as bench:
            rc = ore_snapshot_cli.main(["--example", "lgm_torch", "--tensor-backend", "numpy", "--example-path-counts", "2000"])
        self.assertEqual(rc, 0)
        bench.assert_called_once_with(
            [
                "--paths",
                "2000",
                "--repeats",
                "2",
                "--warmup",
                "1",
                "--seed",
                "42",
            ]
        )

    def test_example_swap_mode_dispatches_to_full_pipeline_benchmark(self):
        with patch("py_ore_tools.ore_snapshot_cli.benchmark_lgm_torch_swap.main", return_value=0) as bench:
            rc = ore_snapshot_cli.main(
                [
                    "--example",
                    "lgm_torch_swap",
                    "--example-path-counts",
                    "10000",
                    "--example-devices",
                    "cpu",
                    "mps",
                    "--example-repeats",
                    "3",
                    "--example-warmup",
                    "2",
                    "--seed",
                    "7",
                ]
            )
        self.assertEqual(rc, 0)
        bench.assert_called_once_with(
            [
                "--paths",
                "10000",
                "--repeats",
                "3",
                "--warmup",
                "2",
                "--seed",
                "7",
                "--devices",
                "cpu",
                "mps",
            ]
        )

    def test_example_fx_portfolio_mode_dispatches_with_trade_count(self):
        with patch("py_ore_tools.ore_snapshot_cli.benchmark_lgm_fx_portfolio_torch.main", return_value=0) as bench:
            rc = ore_snapshot_cli.main(
                [
                    "--example",
                    "lgm_fx_portfolio",
                    "--tensor-backend",
                    "torch-cpu",
                    "--example-path-counts",
                    "50000",
                    "--example-trades",
                    "96",
                ]
            )
        self.assertEqual(rc, 0)
        bench.assert_called_once_with(
            [
                "--paths",
                "50000",
                "--repeats",
                "2",
                "--warmup",
                "1",
                "--seed",
                "42",
                "--devices",
                "cpu",
                "--trades",
                "96",
            ]
        )

    def test_example_fx_portfolio_256_mode_uses_canned_trade_count(self):
        with patch("py_ore_tools.ore_snapshot_cli.benchmark_lgm_fx_portfolio_torch.main", return_value=0) as bench:
            rc = ore_snapshot_cli.main(
                [
                    "--example",
                    "lgm_fx_portfolio_256",
                    "--tensor-backend",
                    "torch-cpu",
                    "--example-path-counts",
                    "10000",
                ]
            )
        self.assertEqual(rc, 0)
        bench.assert_called_once_with(
            [
                "--paths",
                "10000",
                "--repeats",
                "2",
                "--warmup",
                "1",
                "--seed",
                "42",
                "--devices",
                "cpu",
                "--trades",
                "256",
            ]
        )

    def test_infers_modes_from_ore_xml(self):
        modes = ore_snapshot_cli._infer_modes(
            ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML)]),
            REAL_CASE_XML,
        )
        self.assertTrue(modes.price)
        self.assertTrue(modes.xva)

    def test_infer_modes_does_not_force_price_when_npv_analytic_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ore_xml = Path(tmp) / "ore.xml"
            ore_xml.write_text(
                """<ORE>
  <Setup><Parameter name="asofDate">2016-02-05</Parameter></Setup>
  <Analytics>
    <Analytic type="sensitivity">
      <Parameter name="active">Y</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            modes = ore_snapshot_cli._infer_modes(
                ore_snapshot_cli.build_parser().parse_args([str(ore_xml)]),
                ore_xml,
            )
        self.assertFalse(modes.price)
        self.assertFalse(modes.xva)
        self.assertTrue(modes.sensi)

    def test_validate_snapshot_allows_blank_portfolio_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()
            (input_dir / "ore.xml").write_text(
                """<ORE>
  <Setup>
    <Parameter name="asofDate">2020-12-28</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output/SIMM</Parameter>
    <Parameter name="marketDataFile">market.txt</Parameter>
    <Parameter name="curveConfigFile">curveconfig.xml</Parameter>
    <Parameter name="conventionsFile">conventions.xml</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
    <Parameter name="portfolioFile"></Parameter>
  </Setup>
  <Markets />
  <Analytics>
    <Analytic type="simm">
      <Parameter name="active">Y</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            for name in ("market.txt", "curveconfig.xml", "conventions.xml", "todaysmarket.xml"):
                (input_dir / name).write_text("<root />" if name.endswith(".xml") else "", encoding="utf-8")
            result = ore_snapshot_cli.validate_ore_input_snapshot(input_dir / "ore.xml")
        self.assertIsInstance(result, dict)

    def test_default_case_identity_allows_empty_portfolio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()
            (input_dir / "portfolio.xml").write_text("<Portfolio />", encoding="utf-8")
            (input_dir / "ore.xml").write_text(
                """<ORE>
  <Setup>
    <Parameter name="asofDate">2020-12-28</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
  </Setup>
</ORE>
""",
                encoding="utf-8",
            )
            self.assertEqual(
                ore_snapshot_cli._default_case_identity(input_dir / "ore.xml"),
                ("", "", ""),
            )

    def test_non_pricing_case_with_empty_portfolio_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()
            (input_dir / "portfolio.xml").write_text("<Portfolio />", encoding="utf-8")
            (input_dir / "market.txt").write_text("", encoding="utf-8")
            for name in ("curveconfig.xml", "conventions.xml", "todaysmarket.xml"):
                (input_dir / name).write_text("<root />", encoding="utf-8")
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<ORE>
  <Setup>
    <Parameter name="asofDate">2020-12-28</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output</Parameter>
    <Parameter name="marketDataFile">market.txt</Parameter>
    <Parameter name="curveConfigFile">curveconfig.xml</Parameter>
    <Parameter name="conventionsFile">conventions.xml</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
  </Setup>
  <Analytics>
    <Analytic type="zeroToParShift">
      <Parameter name="active">Y</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            summary = ore_snapshot_cli._run_case(
                ore_xml,
                ore_snapshot_cli.build_parser().parse_args([str(ore_xml), "--output-root", str(root / "artifacts")]),
                artifact_root=root / "artifacts",
            )
        self.assertEqual(summary["trade_id"], "")
        self.assertEqual(summary["diagnostics"]["mode"], "non_pricing")
        self.assertTrue(summary["pass_all"])

    def test_xva_case_falls_back_to_reference_on_unsupported_product_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<ORE>
  <Setup>
    <Parameter name="asofDate">2020-12-28</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output</Parameter>
  </Setup>
  <Analytics>
    <Analytic type="xva">
      <Parameter name="active">Y</Parameter>
      <Parameter name="cva">Y</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            args = ore_snapshot_cli.build_parser().parse_args([str(ore_xml), "--output-root", str(root / "artifacts")])
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}), patch(
                "py_ore_tools.ore_snapshot_cli._compute_snapshot_case",
                side_effect=ValueError("FloatingLegData/Index not found for trade 'X' in portfolio XML"),
            ), patch(
                "py_ore_tools.ore_snapshot_cli._ore_reference_summary",
                return_value={
                    "ore_xml": str(ore_xml),
                    "modes": ["xva"],
                    "trade_id": "X",
                    "counterparty": "CPTY",
                    "netting_set_id": "CPTY",
                    "pricing": None,
                    "xva": {"ore_cva": 1.0},
                    "parity": None,
                    "diagnostics": {"engine": "ore_reference"},
                    "input_validation": {},
                    "pass_flags": {},
                    "pass_all": True,
                },
            ), patch("py_ore_tools.ore_snapshot_cli._copy_native_ore_reports"), patch(
                "py_ore_tools.ore_snapshot_cli._write_ore_compatible_reports"
            ):
                summary = ore_snapshot_cli._run_case(ore_xml, args, artifact_root=root / "artifacts")
        self.assertEqual(summary["diagnostics"]["fallback_reason"], "unsupported_python_snapshot")
        self.assertIn("FloatingLegData/Index not found", summary["diagnostics"]["fallback_error"])

    def test_price_case_falls_back_to_reference_on_unsupported_product_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<ORE>
  <Setup>
    <Parameter name="asofDate">2020-12-28</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output</Parameter>
  </Setup>
  <Analytics>
    <Analytic type="npv">
      <Parameter name="active">Y</Parameter>
    </Analytic>
    <Analytic type="simulation">
      <Parameter name="active">Y</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            args = ore_snapshot_cli.build_parser().parse_args([str(ore_xml), "--output-root", str(root / "artifacts")])
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}), patch(
                "py_ore_tools.ore_snapshot_cli._compute_price_only_case",
                side_effect=ValueError("FloatingLegData/Index not found for trade 'X' in portfolio XML"),
            ), patch(
                "py_ore_tools.ore_snapshot_cli._ore_reference_summary",
                return_value={
                    "ore_xml": str(ore_xml),
                    "modes": ["price"],
                    "trade_id": "X",
                    "counterparty": "CPTY",
                    "netting_set_id": "CPTY",
                    "maturity_date": "",
                    "maturity_time": 0.0,
                    "pricing": {"ore_t0_npv": 1.0},
                    "diagnostics": {"engine": "ore_reference_price_only"},
                },
            ), patch("py_ore_tools.ore_snapshot_cli._copy_native_ore_reports"), patch(
                "py_ore_tools.ore_snapshot_cli._write_ore_compatible_reports"
            ):
                summary = ore_snapshot_cli._run_case(ore_xml, args, artifact_root=root / "artifacts")
        self.assertEqual(summary["diagnostics"]["fallback_reason"], "unsupported_python_snapshot")
        self.assertIn("FloatingLegData/Index not found", summary["diagnostics"]["fallback_error"])

    def test_case_run_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            argv = [
                str(REAL_CASE_XML),
                "--price",
                "--xva",
                "--paths",
                "32",
                "--output-root",
                tmp,
            ]
            with redirect_stdout(out):
                rc = ore_snapshot_cli.main(argv)
            self.assertIn(rc, (0, 1))
            case_dir = Path(tmp) / REAL_CASE_XML.parents[1].name
            self.assertTrue((case_dir / "summary.json").exists())
            self.assertTrue((case_dir / "comparison.csv").exists())
            self.assertTrue((case_dir / "report.md").exists())
            self.assertTrue((case_dir / "npv.csv").exists())
            self.assertTrue((case_dir / "xva.csv").exists())
            self.assertTrue((case_dir / "flows.csv").exists())
            self.assertTrue((case_dir / "curves.csv").exists())
            self.assertTrue((case_dir / "cube.csv.gz").exists())
            self.assertTrue((case_dir / "exposure_trade_SWAP_EUR_5Y_A_flat.csv").exists())
            self.assertTrue((case_dir / "exposure_nettingset_CPTY_A.csv").exists())
            payload = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["trade_id"], "SWAP_EUR_5Y_A_flat")
            self.assertIn("pricing", payload)
            self.assertIn("xva", payload)
            with open(case_dir / "npv.csv", newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.reader(f))), 2)
            with open(case_dir / "xva.csv", newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.reader(f))), 3)
            with open(case_dir / "exposure_trade_SWAP_EUR_5Y_A_flat.csv", newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.reader(f))), 123)
            with open(case_dir / "npv.csv", newline="", encoding="utf-8") as f:
                npv_row = next(csv.DictReader(f))
            self.assertEqual(npv_row["Maturity"], "2021-03-01")
            self.assertEqual(npv_row["MaturityTime"], "5.066015")

    def test_pack_mode_writes_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                str(REAL_CASE_XML),
                "--pack",
                "--price",
                "--paths",
                "16",
                "--output-root",
                tmp,
            ]
            rc = ore_snapshot_cli.main(argv)
            self.assertIn(rc, (0, 1))
            pack_dir = Path(tmp) / "pack"
            self.assertTrue((pack_dir / "summary.json").exists())
            self.assertTrue((pack_dir / "results.csv").exists())
            self.assertTrue((pack_dir / "summary.md").exists())

    def test_ore_output_only_suppresses_python_side_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                str(REAL_CASE_XML),
                "--price",
                "--xva",
                "--paths",
                "16",
                "--ore-output-only",
                "--output-root",
                tmp,
            ]
            rc = ore_snapshot_cli.main(argv)
            self.assertIn(rc, (0, 1))
            case_dir = Path(tmp) / REAL_CASE_XML.parents[1].name
            self.assertTrue((case_dir / "npv.csv").exists())
            self.assertTrue((case_dir / "xva.csv").exists())
            self.assertFalse((case_dir / "summary.json").exists())
            self.assertFalse((case_dir / "comparison.csv").exists())
            self.assertFalse((case_dir / "report.md").exists())
            self.assertFalse((case_dir / "input_validation.csv").exists())

    def test_generated_xva_basel_fields_follow_generated_exposure_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = ore_snapshot_cli.main(
                [
                    str(REAL_CASE_XML),
                    "--price",
                    "--xva",
                    "--paths",
                    "32",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertIn(rc, (0, 1))
            case_dir = Path(tmp) / REAL_CASE_XML.parents[1].name
            with open(case_dir / "exposure_nettingset_CPTY_A.csv", newline="", encoding="utf-8") as f:
                exposure_rows = list(csv.DictReader(f))
            with open(case_dir / "xva.csv", newline="", encoding="utf-8") as f:
                xva_rows = list(csv.DictReader(f))
            one_year_row = next(row for row in exposure_rows if float(row["Time"]) >= 1.0)
            agg_row = next(row for row in xva_rows if row["#TradeId"] == "")
            self.assertLess(abs(float(agg_row["BaselEPE"]) - float(one_year_row["TimeWeightedBaselEPE"])), 50.0)
            self.assertLess(abs(float(agg_row["BaselEEPE"]) - float(one_year_row["TimeWeightedBaselEEPE"])), 50.0)
            payload = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
            diagnostics = payload["diagnostics"]
            self.assertIn("float_fixing_source", diagnostics)
            self.assertIn("float_index_day_counter", diagnostics)
            self.assertIn("ore_samples", diagnostics)
            self.assertIn("python_paths", diagnostics)
            self.assertTrue(diagnostics["sample_count_mismatch"])

    def test_price_only_run_does_not_require_xva_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "price_only_case"
            input_dir = case_root / "Input"
            output_dir = case_root / "Output"
            shutil.copytree(REAL_CASE_XML.parent, input_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            real_output = REAL_CASE_XML.parents[1] / "Output"
            for name in ("curves.csv", "npv.csv", "flows.csv", "calibration.xml"):
                src = real_output / name
                if src.exists():
                    shutil.copy2(src, output_dir / name)
            rc = ore_snapshot_cli.main(
                [
                    str(input_dir / "ore.xml"),
                    "--price",
                    "--paths",
                    "8",
                    "--output-root",
                    str(tmp_root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads(
                (tmp_root / "artifacts" / "price_only_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("pricing", payload)
            self.assertIsNone(payload["xva"])

    def test_price_only_run_falls_back_to_reference_without_simulation_analytic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "price_only_case"
            input_dir = case_root / "Input"
            output_dir = case_root / "Output"
            shutil.copytree(REAL_CASE_XML.parent, input_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            real_output = REAL_CASE_XML.parents[1] / "Output"
            for name in ("curves.csv", "npv.csv", "flows.csv"):
                src = real_output / name
                if src.exists():
                    shutil.copy2(src, output_dir / name)
            ore_xml_path = input_dir / "ore.xml"
            text = ore_xml_path.read_text(encoding="utf-8")
            start = text.index('<Analytic type="simulation">')
            end = text.index("</Analytic>", start) + len("</Analytic>")
            ore_xml_path.write_text(text[:start] + text[end:], encoding="utf-8")
            rc = ore_snapshot_cli.main(
                [
                    str(ore_xml_path),
                    "--price",
                    "--output-root",
                    str(tmp_root / "artifacts"),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(
                (tmp_root / "artifacts" / "price_only_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["diagnostics"]["engine"], "ore_reference_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_fallback_reason"], "missing_simulation_analytic")
            self.assertNotIn("py_t0_npv", payload["pricing"])
            self.assertIn("ore_t0_npv", payload["pricing"])
            self.assertTrue(payload["pass_all"])

    def test_run_case_without_supported_analytics_and_without_portfolio_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<ORE>
  <Setup>
    <Parameter name="asofDate">2020-12-28</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output/SIMM</Parameter>
    <Parameter name="marketDataFile">market.txt</Parameter>
    <Parameter name="curveConfigFile">curveconfig.xml</Parameter>
    <Parameter name="conventionsFile">conventions.xml</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
    <Parameter name="portfolioFile"></Parameter>
  </Setup>
  <Markets />
  <Analytics>
    <Analytic type="simm">
      <Parameter name="active">Y</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            for name in ("market.txt", "curveconfig.xml", "conventions.xml", "todaysmarket.xml"):
                (input_dir / name).write_text("<root />" if name.endswith(".xml") else "", encoding="utf-8")
            rc = ore_snapshot_cli.main([str(ore_xml), "--output-root", str(root / "artifacts")])
            self.assertEqual(rc, 0)
            payload = json.loads((root / "artifacts" / root.name / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["modes"], [])
        self.assertEqual(payload["trade_id"], "")
        self.assertTrue(payload["pass_all"])

    def test_sensi_flag_uses_comparator(self):
        fake_result = {
            "metric": "CVA",
            "python": [1],
            "ore": [1],
            "comparisons": [],
            "unmatched_ore": [],
            "unmatched_python": [],
            "unsupported_factors": [],
            "notes": [],
        }

        class _FakeComparator:
            def compare(self, snapshot, metric="CVA", netting_set_id=None):
                return fake_result

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "native_xva_interface.OreSnapshotPythonLgmSensitivityComparator.from_case_dir",
                return_value=(_FakeComparator(), object()),
            ):
                rc = ore_snapshot_cli.main(
                    [
                        str(REAL_CASE_XML),
                        "--sensi",
                        "--price",
                        "--paths",
                        "8",
                        "--output-root",
                        tmp,
                    ]
                )
            self.assertIn(rc, (0, 1))
            payload = json.loads(
                (Path(tmp) / REAL_CASE_XML.parents[1].name / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["sensitivity"]["metric"], "CVA")

    def test_flatten_summary_rows_returns_csv_shape(self):
        rows = ore_snapshot_cli._flatten_summary_rows(
            {
                "pricing": {"ore_t0_npv": 1.0},
                "xva": {"ore_cva": 2.0},
                "diagnostics": {"epe_rel_median": 0.1},
                "sensitivity": {"metric": "CVA", "top_comparisons": []},
            }
        )
        df = pd.DataFrame(rows)
        self.assertEqual(list(df.columns), ["section", "field", "value"])

    def test_build_leg_diagnostics_warns_on_large_spreads(self):
        snap = SimpleNamespace(
            n_samples=500,
            model_day_counter="A365F",
            legs={
                "float_spread": [0.0, 3.5e-4, -3.2e-4],
                "float_fixing_source": "flows_fixing_date",
                "float_index_day_counter": "A360",
            },
        )
        diag = ore_snapshot_cli._build_leg_diagnostics(snap, paths=5000)
        self.assertTrue(diag["sample_count_mismatch"])
        self.assertEqual(diag["float_fixing_source"], "flows_fixing_date")
        self.assertEqual(diag["float_index_day_counter"], "A360")
        self.assertGreater(diag["float_spread_abs_median"], 2.0e-4)
        self.assertEqual(len(diag["warnings"]), 2)

    def test_run_case_only_gates_requested_xva_metrics(self):
        args = ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML), "--price", "--xva"])
        fake_summary = ore_snapshot_cli.SnapshotComputation(
            ore_xml=str(REAL_CASE_XML),
            trade_id="T1",
            counterparty="CPTY_A",
            netting_set_id="CPTY_A",
            paths=500,
            seed=42,
            rng_mode="ore_parity",
            pricing={"t0_npv_abs_diff": 0.1},
            xva={
                "cva_rel_diff": 0.01,
                "dva_rel_diff": 0.50,
                "fba_rel_diff": 0.50,
                "fca_rel_diff": 0.50,
            },
            parity={"summary": {"requested_xva_metrics": ["CVA"]}},
            diagnostics={},
            maturity_date="2021-03-01",
            maturity_time=5.0,
            exposure_dates=[],
            exposure_times=[],
            py_epe=[],
            py_ene=[],
            py_pfe=[],
            ore_basel_epe=0.0,
            ore_basel_eepe=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}):
                with patch("py_ore_tools.ore_snapshot_cli._compute_snapshot_case", return_value=fake_summary):
                    with patch("py_ore_tools.ore_snapshot_cli._write_ore_compatible_reports"):
                        with patch("py_ore_tools.ore_snapshot_cli._copy_native_ore_reports"):
                            result = ore_snapshot_cli._run_case(REAL_CASE_XML, args, artifact_root=Path(tmp))
        self.assertTrue(result["pass_flags"]["xva_cva"])
        self.assertTrue(result["pass_flags"]["xva_dva"])
        self.assertTrue(result["pass_flags"]["xva_fba"])
        self.assertTrue(result["pass_flags"]["xva_fca"])
        self.assertTrue(result["pass_all"])

    def test_bond_price_only_case_uses_python_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = ore_snapshot_cli.main([str(BOND_CASE_XML), "--price", "--output-root", tmp])
            self.assertIn(rc, (0, 1))
            payload = json.loads((Path(tmp) / BOND_CASE_XML.parents[1].name / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["trade_id"], "Bond_Fixed")
        self.assertEqual(payload["pricing"]["trade_type"], "Bond")
        self.assertEqual(payload["diagnostics"]["trade_type"], "Bond")
        self.assertEqual(payload["diagnostics"]["bond_pricing_mode"], "python_risky_bond")
        self.assertIn("py_t0_npv", payload["pricing"])
        self.assertIn("ore_t0_npv", payload["pricing"])

    def test_write_ore_reports_preserves_bond_trade_type(self):
        case_summary = {
            "trade_id": "Bond_Fixed",
            "netting_set_id": "",
            "counterparty": "CPTY_C",
            "maturity_date": "2021-02-03",
            "maturity_time": 5.0,
            "pricing": {"trade_type": "Bond", "py_t0_npv": 12.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            ore_snapshot_cli._write_ore_compatible_reports(Path(tmp), case_summary)
            with open(Path(tmp) / "npv.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["TradeType"], "Bond")

    def test_callable_bond_price_only_case_uses_python_dispatch(self):
        fake_result = {
            "trade_type": "CallableBond",
            "py_npv": 101.25,
            "reference_curve_id": "EUR-EURIBOR-3M",
            "income_curve_id": "EUR-EURIBOR-3M",
            "credit_curve_id": "CPTY_A",
            "security_id": "SECURITY_CALL",
            "security_spread": 0.0,
            "settlement_dirty": True,
            "spread_on_income_curve": True,
            "call_schedule_count": 3,
            "put_schedule_count": 0,
            "exercise_time_steps_per_year": 24,
            "callable_model_family": "LGM",
            "callable_engine_variant": "Grid",
            "stripped_bond_npv": 103.0,
            "embedded_option_value": -1.75,
        }
        fake_npv = {"npv": 100.0, "maturity_date": "2024-02-26", "maturity_time": 8.0}
        with patch("py_ore_tools.ore_snapshot_cli.price_bond_trade", return_value=fake_result):
            with patch("py_ore_tools.ore_snapshot._load_ore_npv_details", return_value=fake_npv):
                with patch("py_ore_tools.ore_snapshot_cli._find_reference_output_file", return_value=CALLABLE_CASE_XML):
                    payload = ore_snapshot_cli._compute_price_only_case(CALLABLE_CASE_XML, anchor_t0_npv=False)
        self.assertEqual(payload["trade_type"], "CallableBond")
        self.assertEqual(payload["pricing"]["bond_pricing_mode"], "python_callable_lgm")
        self.assertEqual(payload["diagnostics"]["bond_pricing_mode"], "python_callable_lgm")
        self.assertEqual(payload["diagnostics"]["call_schedule_count"], 3)
        self.assertEqual(payload["diagnostics"]["callable_model_family"], "LGM")


if __name__ == "__main__":
    unittest.main()
