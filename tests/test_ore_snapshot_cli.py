import ast
import io
import json
import os
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
import pythonore.io.ore_snapshot as ore_snapshot_io
from pythonore.runtime import bermudan as bermudan_runtime

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
FX_FORWARD_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_28" / "Input" / "ore_eur_base.xml"
FX_OPTION_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_13" / "Input" / "ore_E0.xml"
FX_NDF_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_71" / "Input" / "ore.xml"
SWAPTION_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_19" / "Input" / "ore_flat.xml"
SWAPTION_SMILE_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_19" / "Input" / "ore_smile.xml"
SWAPTION_LONG_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_12" / "Input" / "ore_swaption.xml"
SWAPTION_MIXED_CASE_XML = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_swap_swaptions.xml"
BERMUDAN_CASE_XML = TOOLS_DIR / "Examples" / "ORE-Python" / "Notebooks" / "Example_3" / "Input" / "ore_bermudans.xml"
BERMUDAN_SENSI_CASE_XML = TOOLS_DIR / "parity_artifacts" / "bermudan_sensitivity_compare" / "berm_200bp" / "Input" / "ore.xml"
CAPFLOOR_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_6" / "Input" / "ore_portfolio_2.xml"
INFLATION_CAPFLOOR_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input" / "ore_capfloor.xml"
TA001_EQUITY_CASE_XML = TOOLS_DIR / "Examples" / "Academy" / "TA001_Equity_Option" / "Input" / "ore.xml"
EXAMPLE22_EQUITY_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_22" / "Input" / "ore_atmOnly.xml"
SCRIPTED_EQUITY_CASE_XML = TOOLS_DIR / "Examples" / "ScriptedTrade" / "Input" / "ore.xml"


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

    def test_is_plain_vanilla_swap_trade_distinguishes_cross_currency_swap(self):
        self.assertFalse(
            ore_snapshot_cli._is_plain_vanilla_swap_trade(
                TOOLS_DIR / "Examples" / "Legacy" / "Example_29" / "Input" / "ore.xml"
            )
        )
        self.assertTrue(
            ore_snapshot_cli._is_plain_vanilla_swap_trade(
                TOOLS_DIR / "Examples" / "ORE-Python" / "Notebooks" / "Example_6" / "Input" / "ore.xml"
            )
        )

    def test_supports_native_price_only_excludes_unsupported_non_swap_products(self):
        self.assertFalse(
            ore_snapshot_cli._supports_native_price_only(
                "CommodityForward",
                TOOLS_DIR / "Examples" / "Legacy" / "Example_24" / "Input" / "ore.xml",
            )
        )
        self.assertTrue(
            ore_snapshot_cli._supports_native_price_only(
                "EquityOption",
                TOOLS_DIR / "Examples" / "Legacy" / "Example_22" / "Input" / "ore_atmOnly.xml",
            )
        )
        self.assertTrue(
            ore_snapshot_cli._supports_native_price_only(
                "EquityForward",
                TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_equity.xml",
            )
        )
        self.assertTrue(
            ore_snapshot_cli._supports_native_price_only(
                "EquitySwap",
                TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_equity.xml",
            )
        )
        self.assertTrue(
            ore_snapshot_cli._supports_native_price_only(
                "FxOption",
                FX_OPTION_CASE_XML,
            )
        )
        self.assertFalse(
            ore_snapshot_cli._supports_native_price_only(
                "Swaption",
                SWAPTION_LONG_CASE_XML,
            )
        )
        self.assertTrue(
            ore_snapshot_cli._supports_native_price_only(
                "CapFloor",
                INFLATION_CAPFLOOR_CASE_XML,
            )
        )
        self.assertTrue(
            ore_snapshot_cli._supports_native_price_only(
                "CapFloor",
                CAPFLOOR_CASE_XML,
            )
        )

    def test_parse_ore_date_accepts_compact_format(self):
        parsed = ore_snapshot_cli._parse_ore_date("20170301")
        self.assertEqual(parsed.isoformat(), "2017-03-01")

    def test_parse_market_instrument_key_accepts_zero_tenor(self):
        parsed = ore_snapshot_io._parse_market_instrument_key("ZERO/RATE/USD/5Y", asof_date="2020-01-01")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["instrument_type"], "ZERO")
        self.assertEqual(parsed["tenor"], "5Y")

    def test_parse_market_instrument_key_accepts_dated_zero_iso(self):
        parsed = ore_snapshot_io._parse_market_instrument_key("ZERO/RATE/USD/2027-03-20", asof_date="2026-03-20")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["instrument_type"], "ZERO")
        self.assertEqual(parsed["tenor"], "2027-03-20")
        self.assertAlmostEqual(float(parsed["maturity"]), 1.0, places=6)

    def test_parse_market_instrument_key_accepts_dated_zero_compact(self):
        parsed = ore_snapshot_io._parse_market_instrument_key("ZERO/RATE/USD/20270320", asof_date="2026-03-20")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["instrument_type"], "ZERO")
        self.assertEqual(parsed["tenor"], "20270320")
        self.assertAlmostEqual(float(parsed["maturity"]), 1.0, places=6)

    def test_parse_market_instrument_key_rejects_invalid_dated_zero(self):
        parsed = ore_snapshot_io._parse_market_instrument_key("ZERO/RATE/USD/2027-13-40", asof_date="2026-03-20")
        self.assertIsNone(parsed)

    def test_extract_market_instruments_from_quotes_accepts_dated_zeroes(self):
        payload = ore_snapshot_io.extract_market_instruments_by_currency_from_quotes(
            "2026-03-20",
            [
                ("ZERO/RATE/USD/2027-03-20", 0.02),
                ("ZERO/RATE/USD/20280320", 0.025),
                ("IR_SWAP/RATE/USD/USD-LIBOR-3M/3M/5Y", 0.03),
            ],
        )
        self.assertIn("USD", payload)
        instruments = payload["USD"]["instruments"]
        zeros = [ins for ins in instruments if ins["instrument_type"] == "ZERO"]
        self.assertEqual(len(zeros), 2)
        self.assertTrue(all(float(ins["maturity"]) > 0.0 for ins in zeros))

    def test_fit_curve_from_instruments_accepts_dated_zeroes(self):
        payload = ore_snapshot_io.extract_market_instruments_by_currency_from_quotes(
            "2026-03-20",
            [
                ("ZERO/RATE/USD/2027-03-20", 0.02),
                ("ZERO/RATE/USD/20280320", 0.025),
                ("ZERO/RATE/USD/5Y", 0.03),
            ],
        )
        fit = ore_snapshot_io._fit_curve_from_instruments("2026-03-20", payload["USD"]["instruments"])
        self.assertGreaterEqual(len(fit["times"]), 4)
        self.assertAlmostEqual(float(fit["times"][0]), 0.0, places=12)

    def test_forward_curve_fit_selector_accepts_zeroes_and_matching_swaps(self):
        calls = []

        def fake_fit(ore_xml, *, currency, selector):
            instruments = [
                {"instrument_type": "ZERO", "index": "", "maturity": 1.0},
                {"instrument_type": "IR_SWAP", "index": "USD-LIBOR-3M", "maturity": 5.0},
                {"instrument_type": "IR_SWAP", "index": "USD-LIBOR-6M", "maturity": 7.0},
            ]
            chosen = [ins for ins in instruments if selector(ins)]
            calls.append([str(ins["instrument_type"]) + ":" + str(ins["index"]) for ins in chosen])
            if not chosen:
                raise ValueError("no market instruments selected")
            base = 0.99 if len(calls) == 1 else 0.985
            return {"times": [0.0, 1.0], "dfs": [1.0, base], "calendar_dates": ["2026-03-20", "2027-03-20"]}

        with patch("py_ore_tools.ore_snapshot_cli._fit_market_curve_from_selector", side_effect=fake_fit):
            ore_snapshot_cli._build_fitted_discount_and_forward_curves(
                FX_FORWARD_CASE_XML,
                asof=ore_snapshot_cli._parse_ore_date("2026-03-20"),
                currency="USD",
                float_index="USD-LIBOR-3M",
            )
        self.assertEqual(len(calls), 2)
        self.assertIn("ZERO:", calls[1])
        self.assertIn("IR_SWAP:USD-LIBOR-3M", calls[1])
        self.assertNotIn("IR_SWAP:USD-LIBOR-6M", calls[1])

    def test_forward_curve_fit_selector_falls_back_to_discount_fit_when_family_subset_missing(self):
        calls = []

        def fake_fit(ore_xml, *, currency, selector):
            instruments = [{"instrument_type": "MM", "index": "", "maturity": 0.5}]
            chosen = [ins for ins in instruments if selector(ins)]
            calls.append(len(chosen))
            if not chosen:
                raise ValueError("no market instruments selected")
            return {"times": [0.0, 1.0], "dfs": [1.0, 0.99], "calendar_dates": ["2026-03-20", "2027-03-20"]}

        with patch("py_ore_tools.ore_snapshot_cli._fit_market_curve_from_selector", side_effect=fake_fit):
            p0_disc, p0_fwd, _, _ = ore_snapshot_cli._build_fitted_discount_and_forward_curves(
                FX_FORWARD_CASE_XML,
                asof=ore_snapshot_cli._parse_ore_date("2026-03-20"),
                currency="USD",
                float_index="USD-LIBOR-3M",
            )
        self.assertEqual(calls, [1, 0])
        self.assertAlmostEqual(float(p0_disc(1.0)), float(p0_fwd(1.0)), places=12)

    def test_parse_market_quotes_accepts_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            market = Path(tmp) / "market.csv"
            market.write_text(
                "2023-06-05,EQUITY/PRICE/RIC:.STOXX50E/EUR,4337.5\n",
                encoding="utf-8",
            )
            quotes = ore_snapshot_cli._parse_market_quotes(market, "2023-06-05")
        self.assertEqual(quotes["EQUITY/PRICE/RIC:.STOXX50E/EUR"], 4337.5)

    def test_bermudan_invalid_grid_text_degrades_to_no_grid(self):
        grid = bermudan_runtime._simulation_grid_times_from_xml_text(
            "<Simulation><Parameters><Grid>1Y,1Y</Grid></Parameters></Simulation>"
        )
        self.assertIsNone(grid)

    def test_price_only_fx_forward_runs_python_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(FX_FORWARD_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_28" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_forward")
            self.assertEqual(payload["pricing"]["trade_type"], "FxForward")
            self.assertIn("py_t0_npv", payload["pricing"])

    def test_price_only_fx_ndf_uses_cash_settlement_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(FX_NDF_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_71" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_forward")
            self.assertEqual(payload["pricing"]["trade_type"], "FxForward")
            self.assertEqual(payload["pricing"]["fx_settlement_type"], "CASH")
            self.assertEqual(payload["pricing"]["fx_settlement_currency"], "USD")
            self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 1.0)

    def test_sensitivity_failure_falls_back_instead_of_crashing_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("py_ore_tools.ore_snapshot_cli._run_sensitivity_case", side_effect=RuntimeError("boom")):
                rc = ore_snapshot_cli.main(
                    [
                        str(FX_NDF_CASE_XML),
                        "--output-root",
                        str(root / "artifacts"),
                    ]
                )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_71" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["sensitivity_fallback_reason"], "unsupported_python_sensitivity")
            self.assertEqual(payload["sensitivity"]["top_comparisons"], [])
            self.assertIn("boom", payload["sensitivity"]["notes"][0])

    def test_price_only_fx_option_runs_python_path_without_native_npv_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(FX_OPTION_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads((root / "artifacts" / "Example_13" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_option")
            self.assertTrue(payload["diagnostics"]["missing_native_pricing_reference"])
            self.assertEqual(payload["pricing"]["trade_type"], "FxOption")
            self.assertIn("py_t0_npv", payload["pricing"])
            self.assertNotIn("ore_t0_npv", payload["pricing"])

    def test_price_only_scripted_equity_option_runs_python_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(SCRIPTED_EQUITY_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "ScriptedTrade" / "summary.json").read_text(encoding="utf-8"))
            self.assertIn(payload["diagnostics"]["engine"], {"python_price_only", "ore_reference_expected_output"})
            self.assertTrue(payload["pricing"])

    def test_price_only_swaption_runs_python_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(SWAPTION_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_19" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_swaption_static")
            self.assertEqual(payload["pricing"]["trade_type"], "Swaption")
            self.assertIn("py_t0_npv", payload["pricing"])
            self.assertGreater(float(payload["pricing"]["py_t0_npv"]), 0.0)
            self.assertTrue(payload["pass_all"])
            self.assertLess(float(payload["pricing"]["t0_npv_abs_diff"]), 500.0)

    def test_price_only_mixed_swaption_case_accepts_compact_exercise_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(SWAPTION_MIXED_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Exposure" / "summary.json").read_text(encoding="utf-8"))
            self.assertNotEqual(payload.get("bucket"), "hard_error")
            self.assertIn(payload["diagnostics"]["engine"], {"python_price_only", "ore_reference_expected_output"})

    def test_price_only_swaption_smile_runs_python_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(SWAPTION_SMILE_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_19" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_swaption_static")
            self.assertTrue(payload["pass_all"])
            self.assertLess(float(payload["pricing"]["t0_npv_abs_diff"]), 2000.0)

    def test_price_only_bermudan_swaption_runs_python_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(BERMUDAN_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_3" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_bermudan_swaption_backward")
            self.assertEqual(payload["diagnostics"]["bermudan_method"], "backward")
            self.assertEqual(payload["pricing"]["trade_type"], "Swaption")
            self.assertIn("py_t0_npv", payload["pricing"])
            self.assertGreater(float(payload["pricing"]["py_t0_npv"]), 0.0)
            self.assertLess(float(payload["pricing"]["t0_npv_abs_diff"]), 1000.0)
            self.assertTrue(payload["pass_all"])

    def test_capfloor_xva_runs_native_compare_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(CAPFLOOR_CASE_XML),
                    "--xva",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_6" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "compare")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_capfloor_lgm")
            self.assertEqual(payload["trade_id"], "cap_01")
            self.assertIn("py_cva", payload["xva"])

    def test_fx_option_xva_runs_native_compare_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(FX_OPTION_CASE_XML),
                    "--price",
                    "--xva",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_13" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "compare")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_option_hybrid")
            self.assertEqual(payload["pricing"]["trade_type"], "FxOption")
            self.assertIn("py_cva", payload["xva"])
            self.assertTrue(payload["diagnostics"]["missing_reference_xva"])
            self.assertNotIn("ore_cva", payload["xva"])
            self.assertGreater(len(payload["py_epe"]), 1)

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
            (input_dir / "simulation.xml").write_text("<Simulation />", encoding="utf-8")
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
      <Parameter name="simulationConfigFile">simulation.xml</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            args = ore_snapshot_cli.build_parser().parse_args([str(ore_xml), "--output-root", str(root / "artifacts")])
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}), patch(
                "py_ore_tools.ore_snapshot_cli._supports_native_price_only",
                return_value=True,
            ), patch(
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

    def test_price_case_falls_back_gracefully_when_quantlib_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir()
            ore_xml = input_dir / "ore.xml"
            (input_dir / "simulation.xml").write_text("<Simulation />", encoding="utf-8")
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
      <Parameter name="simulationConfigFile">simulation.xml</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            args = ore_snapshot_cli.build_parser().parse_args([str(ore_xml), "--output-root", str(root / "artifacts")])
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}), patch(
                "py_ore_tools.ore_snapshot_cli._supports_native_price_only",
                return_value=True,
            ), patch(
                "py_ore_tools.ore_snapshot_cli._compute_price_only_case",
                side_effect=ImportError("QuantLib Python bindings are required for swaption price-only support"),
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
        self.assertIn("QuantLib Python bindings are required", summary["diagnostics"]["fallback_error"])

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

    def test_default_xva_run_uses_snapshot_sample_count_when_paths_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = ore_snapshot_cli.main(
                [
                    str(REAL_CASE_XML),
                    "--price",
                    "--xva",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((Path(tmp) / REAL_CASE_XML.parents[1].name / "summary.json").read_text(encoding="utf-8"))
            diagnostics = payload["diagnostics"]
            self.assertEqual(diagnostics["python_paths"], diagnostics["ore_samples"])
            self.assertFalse(diagnostics["sample_count_mismatch"])

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

    def test_price_only_swap_run_works_without_curves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "swap_no_curves_case"
            input_dir = case_root / "Input"
            output_dir = case_root / "Output"
            shutil.copytree(REAL_CASE_XML.parent, input_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            real_output = REAL_CASE_XML.parents[1] / "Output"
            for name in ("npv.csv", "flows.csv", "calibration.xml"):
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
                (tmp_root / "artifacts" / "swap_no_curves_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertIn("py_t0_npv", payload["pricing"])

    def test_price_only_swap_run_calibrates_lgm_params_when_reference_calibration_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "swap_calibration_case"
            input_dir = case_root / "Input"
            output_dir = case_root / "Output"
            shutil.copytree(REAL_CASE_XML.parent, input_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            real_output = REAL_CASE_XML.parents[1] / "Output"
            for name in ("npv.csv", "flows.csv"):
                src = real_output / name
                if src.exists():
                    shutil.copy2(src, output_dir / name)
            calibrated = {
                "alpha_times": (1.0,),
                "alpha_values": (0.015, 0.015),
                "kappa_times": (1.0,),
                "kappa_values": (0.03, 0.03),
                "shift": 0.0,
                "scaling": 1.0,
            }
            with patch("py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.resolve_calibration_xml_path", return_value=None), patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.calibrate_lgm_params_via_ore",
                return_value=calibrated,
            ) as calibrate_mock, patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.parse_lgm_params_from_simulation_xml",
                side_effect=AssertionError("simulation fallback should not be used"),
            ):
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
            self.assertEqual(calibrate_mock.call_count, 1)
            payload = json.loads(
                (tmp_root / "artifacts" / "swap_calibration_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertIn("py_t0_npv", payload["pricing"])

    def test_price_only_swap_run_falls_back_to_simulation_when_runtime_calibration_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "swap_simulation_fallback_case"
            input_dir = case_root / "Input"
            output_dir = case_root / "Output"
            shutil.copytree(REAL_CASE_XML.parent, input_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            real_output = REAL_CASE_XML.parents[1] / "Output"
            for name in ("npv.csv", "flows.csv"):
                src = real_output / name
                if src.exists():
                    shutil.copy2(src, output_dir / name)
            sim_params = {
                "alpha_times": (1.0,),
                "alpha_values": (0.01, 0.01),
                "kappa_times": (1.0,),
                "kappa_values": (0.03, 0.03),
                "shift": 0.0,
                "scaling": 1.0,
            }
            with patch("py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.resolve_calibration_xml_path", return_value=None), patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.calibrate_lgm_params_via_ore",
                return_value=None,
            ) as calibrate_mock, patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.parse_lgm_params_from_simulation_xml",
                return_value=sim_params,
            ) as simulation_mock:
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
            self.assertEqual(calibrate_mock.call_count, 1)
            self.assertEqual(simulation_mock.call_count, 1)
            payload = json.loads(
                (tmp_root / "artifacts" / "swap_simulation_fallback_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertIn("py_t0_npv", payload["pricing"])

    def test_price_only_fx_forward_runs_without_curves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "fx_forward_no_curves_case"
            input_dir = case_root / "Input"
            expected_dir = case_root / "ExpectedOutput"
            shutil.copytree(FX_FORWARD_CASE_XML.parent, input_dir)
            common_input = FX_FORWARD_CASE_XML.parents[3] / "Input"
            if common_input.exists():
                shared_input = tmp_root.parent / "Input"
                if shared_input.exists():
                    shutil.rmtree(shared_input)
                shutil.copytree(common_input, shared_input)
            expected_dir.mkdir(parents=True, exist_ok=True)
            source_dir = FX_FORWARD_CASE_XML.parents[1] / "ExpectedOutput"
            for name in ("npv_eur_base.csv", "flows_eur_base.csv"):
                shutil.copy2(source_dir / name, expected_dir / name)
            rc = ore_snapshot_cli.main(
                [
                    str(input_dir / "ore_eur_base.xml"),
                    "--price",
                    "--output-root",
                    str(tmp_root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads(
                (tmp_root / "artifacts" / "fx_forward_no_curves_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_forward")
            self.assertIn("py_t0_npv", payload["pricing"])

    def test_price_only_fx_option_runs_without_curves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "fx_option_no_curves_case"
            input_dir = case_root / "Input"
            expected_dir = case_root / "ExpectedOutput"
            shutil.copytree(FX_OPTION_CASE_XML.parent, input_dir)
            common_input = FX_OPTION_CASE_XML.parents[3] / "Input"
            if common_input.exists():
                shared_input = tmp_root.parent / "Input"
                if shared_input.exists():
                    shutil.rmtree(shared_input)
                shutil.copytree(common_input, shared_input)
            shutil.copytree(FX_OPTION_CASE_XML.parents[1] / "ExpectedOutput", expected_dir)
            curves_csv = expected_dir / "curves.csv"
            if curves_csv.exists():
                curves_csv.unlink()
            rc = ore_snapshot_cli.main(
                [
                    str(input_dir / "ore_E0.xml"),
                    "--price",
                    "--output-root",
                    str(tmp_root / "artifacts"),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(
                (tmp_root / "artifacts" / "fx_option_no_curves_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_option")
            self.assertIn("py_t0_npv", payload["pricing"])

    def test_fx_option_xva_runs_without_curves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "fx_option_xva_no_curves_case"
            input_dir = case_root / "Input"
            expected_dir = case_root / "ExpectedOutput"
            shutil.copytree(FX_OPTION_CASE_XML.parent, input_dir)
            common_input = FX_OPTION_CASE_XML.parents[3] / "Input"
            if common_input.exists():
                shared_input = tmp_root.parent / "Input"
                if shared_input.exists():
                    shutil.rmtree(shared_input)
                shutil.copytree(common_input, shared_input)
            shutil.copytree(FX_OPTION_CASE_XML.parents[1] / "ExpectedOutput", expected_dir)
            curves_csv = expected_dir / "curves.csv"
            if curves_csv.exists():
                curves_csv.unlink()
            rc = ore_snapshot_cli.main(
                [
                    str(input_dir / "ore_E0.xml"),
                    "--price",
                    "--xva",
                    "--output-root",
                    str(tmp_root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads(
                (tmp_root / "artifacts" / "fx_option_xva_no_curves_case" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("py_cva", payload["xva"])
            self.assertTrue(payload["diagnostics"]["missing_reference_xva"])

    def test_price_only_run_uses_synthetic_swap_setup_without_simulation_analytic(self):
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
            sim_xml = input_dir / "simulation.xml"
            if sim_xml.exists():
                sim_xml.unlink()
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
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertIn("py_t0_npv", payload["pricing"])
            self.assertIn("ore_t0_npv", payload["pricing"])
            self.assertTrue(payload["pass_all"])

    def test_reference_curve_builder_does_not_fall_back_without_curves_csv(self):
        tm_path = ore_snapshot_io._resolve_ore_run_files(REAL_CASE_XML)[3]
        tm_root = ore_snapshot_io.ET.parse(tm_path).getroot()
        with patch("py_ore_tools.ore_snapshot_cli._find_reference_output_file", return_value=None):
            result = ore_snapshot_cli._build_reference_discount_and_forward_curves(
                REAL_CASE_XML,
                asof=ore_snapshot_cli._parse_ore_date("2016-02-05"),
                asof_date="2016-02-05",
                pricing_config_id="default",
                tm_root=tm_root,
                currency="EUR",
                float_index="EUR-EURIBOR-6M",
            )
        self.assertIsNone(result)

    def test_load_from_ore_xml_runs_runtime_calibration_when_reference_calibration_missing(self):
        calibrated = {
            "alpha_times": (1.0,),
            "alpha_values": (0.02, 0.02),
            "kappa_times": (1.0,),
            "kappa_values": (0.03, 0.03),
            "shift": 0.0,
            "scaling": 1.0,
        }
        with patch("pythonore.io.ore_snapshot.resolve_calibration_xml_path", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore",
            return_value=calibrated,
        ) as calibrate_mock, patch(
            "pythonore.io.ore_snapshot.parse_lgm_params_from_simulation_xml",
            side_effect=AssertionError("simulation fallback should not be used"),
        ):
            snap = ore_snapshot_io.load_from_ore_xml(REAL_CASE_XML)
        self.assertEqual(calibrate_mock.call_count, 1)
        self.assertEqual(snap.alpha_source, "calibration")

    def test_cli_surface_parses_under_python38_grammar(self):
        files = [
            TOOLS_DIR / "src" / "pythonore" / "apps" / "ore_snapshot_cli.py",
            TOOLS_DIR / "src" / "pythonore" / "workflows" / "ore_snapshot_cli.py",
            TOOLS_DIR / "src" / "pythonore" / "repo_paths.py",
            TOOLS_DIR / "src" / "py_ore_tools" / "__init__.py",
        ]
        for path in files:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))

    def test_price_only_run_uses_sibling_simulation_xml_without_simulation_analytic(self):
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
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertNotIn("pricing_fallback_reason", payload["diagnostics"])
            self.assertIn("py_t0_npv", payload["pricing"])
            self.assertIn("ore_t0_npv", payload["pricing"])
            self.assertTrue(payload["pass_all"])

    def test_has_active_simulation_analytic_accepts_inactive_analytic_with_existing_simulation_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            input_dir = tmp_root / "Input"
            input_dir.mkdir()
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<ORE>
  <Analytics>
    <Analytic type="simulation">
      <Parameter name="active">N</Parameter>
      <Parameter name="simulationConfigFile">simulation.xml</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            (input_dir / "simulation.xml").write_text("<Simulation />", encoding="utf-8")
            self.assertTrue(ore_snapshot_cli._has_active_simulation_analytic(ore_xml))

    def test_reference_fallback_classifier_accepts_todaysmarket_resolution_errors(self):
        self.assertTrue(
            ore_snapshot_cli._is_reference_fallback_error(
                ValueError("DiscountingCurves[@id='default'] has no DiscountingCurve[@currency='EUR']")
            )
        )

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

    def test_forward_bond_default_mode_prefers_python_price_only_dispatch(self):
        forward_case = TOOLS_DIR / "Examples" / "Legacy" / "Example_73" / "Input" / "ore.xml"
        with tempfile.TemporaryDirectory() as tmp:
            rc = ore_snapshot_cli.main([str(forward_case), "--output-root", tmp])
            self.assertEqual(rc, 0)
            payload = json.loads((Path(tmp) / forward_case.parents[1].name / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["trade_id"], "FwdBond")
        self.assertEqual(payload["pricing"]["trade_type"], "ForwardBond")
        self.assertEqual(payload["diagnostics"]["bond_pricing_mode"], "python_risky_bond")
        self.assertEqual(payload["diagnostics"].get("engine"), "python")
        self.assertNotIn("ore_t0_npv", payload["pricing"])
        self.assertIsNone(payload.get("xva"))

    def test_callable_bond_default_mode_prefers_python_price_only_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = ore_snapshot_cli.main([str(CALLABLE_CASE_XML), "--output-root", tmp])
            self.assertIn(rc, (0, 1))
            payload = json.loads((Path(tmp) / CALLABLE_CASE_XML.parents[1].name / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["trade_id"], "CallableBondTrade")
        self.assertEqual(payload["pricing"]["trade_type"], "CallableBond")
        self.assertEqual(payload["diagnostics"]["bond_pricing_mode"], "python_callable_lgm")
        self.assertEqual(payload["diagnostics"].get("engine"), "python")
        self.assertNotIn("ore_t0_npv", payload["pricing"])
        self.assertIsNone(payload.get("xva"))

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

    def test_parse_sensitivity_factor_setup_reads_curve_nodes(self):
        params = ore_snapshot_cli._parse_sensitivity_analytic_params(BERMUDAN_SENSI_CASE_XML)
        factor_shifts, curve_specs, factor_labels = ore_snapshot_cli._parse_sensitivity_factor_setup(
            BERMUDAN_SENSI_CASE_XML,
            sensi_params=params,
        )
        self.assertEqual(factor_shifts["zero:EUR:10Y"], 1.0e-4)
        self.assertEqual(factor_shifts["fwd:EUR:6M:10Y"], 1.0e-4)
        self.assertEqual(factor_labels["zero:EUR:10Y"], "DiscountCurve/EUR/0/10Y")
        self.assertEqual(factor_labels["fwd:EUR:6M:10Y"], "IndexCurve/EUR-EURIBOR-6M/0/10Y")
        self.assertEqual(curve_specs["zero:EUR:10Y"]["kind"], "discount")
        self.assertEqual(curve_specs["fwd:EUR:6M:10Y"]["kind"], "forward")

    def test_run_sensitivity_case_uses_native_npv_mode_for_pricing_sensitivity(self):
        calls = {}

        class _FakeComparator:
            def compare(self, snapshot, **kwargs):
                calls.update(kwargs)
                row = SimpleNamespace(
                    normalized_factor="zero:EUR:10Y",
                    raw_quote_key="curve:DiscountCurve/EUR/0/10Y",
                    ore_factor="DiscountCurve/EUR/0/10Y",
                    shift_size=1.0e-4,
                    base_value=0.0,
                    base_metric_value=12.0,
                    bumped_up_metric_value=12.25,
                    bumped_down_metric_value=11.75,
                    delta=0.25,
                )
                return {
                    "metric": kwargs["metric"],
                    "python": [row],
                    "ore": [],
                    "comparisons": [],
                    "unmatched_ore": [],
                    "unmatched_python": ["zero:EUR:10Y"],
                    "unsupported_factors": [],
                    "notes": ["native only"],
                }

        with patch(
            "pythonore.runtime.sensitivity.OreSnapshotPythonLgmSensitivityComparator.from_case_dir",
            return_value=(_FakeComparator(), object()),
        ):
            result = ore_snapshot_cli._run_sensitivity_case(
                BERMUDAN_SENSI_CASE_XML,
                metric="CVA",
                netting_set=None,
                top=5,
            )
        self.assertEqual(calls["metric"], "NPV")
        self.assertIn("zero:EUR:10Y", calls["factor_shifts"])
        self.assertEqual(result["metric"], "NPV")
        self.assertEqual(result["python_factor_count"], 1)
        self.assertEqual(result["scenario_rows"][0]["direction"], "Up")
        self.assertEqual(result["sensitivity_output_file"], "sensitivity.csv")

    def test_write_ore_reports_emits_native_sensitivity_and_scenario_csv(self):
        case_summary = {
            "trade_id": "Trade_1",
            "netting_set_id": "CPTY_A",
            "counterparty": "CPTY_A",
            "maturity_date": "2030-01-01",
            "maturity_time": 5.0,
            "pricing": {"trade_type": "Swap", "py_t0_npv": 10.0, "currency": "EUR"},
            "sensitivity": {
                "metric": "NPV",
                "python_rows": [
                    {
                        "normalized_factor": "zero:EUR:10Y",
                        "ore_factor": "DiscountCurve/EUR/0/10Y",
                        "shift_size": 1.0e-4,
                        "base_metric_value": 10.0,
                        "delta": 0.25,
                    }
                ],
                "scenario_rows": [
                    {
                        "factor": "DiscountCurve/EUR/0/10Y",
                        "direction": "Up",
                        "base_metric_value": 10.0,
                        "shift_size_1": 1.0e-4,
                        "shift_size_2": "#N/A",
                        "scenario_metric_value": 10.25,
                        "difference": 0.25,
                    }
                ],
                "sensitivity_output_file": "sensitivity.csv",
                "scenario_output_file": "scenario.csv",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            ore_snapshot_cli._write_ore_compatible_reports(Path(tmp), case_summary)
            with open(Path(tmp) / "sensitivity.csv", newline="", encoding="utf-8") as handle:
                sensitivity_rows = list(csv.DictReader(handle))
            with open(Path(tmp) / "scenario.csv", newline="", encoding="utf-8") as handle:
                scenario_rows = list(csv.DictReader(handle))
        self.assertEqual(sensitivity_rows[0]["Factor_1"], "DiscountCurve/EUR/0/10Y")
        self.assertEqual(scenario_rows[0]["Up/Down"], "Up")
        self.assertEqual(scenario_rows[0]["Difference"], "0.25")

    @unittest.skipUnless(
        os.getenv("PY_ORE_RUN_SLOW_CLI_INTEGRATION") == "1",
        "set PY_ORE_RUN_SLOW_CLI_INTEGRATION=1 to run slow real-case CLI sensitivity integration",
    )
    def test_real_swap_case_writes_native_npv_sensitivity_and_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = ore_snapshot_cli.main(
                [
                    str(REAL_CASE_XML),
                    "--price",
                    "--sensi",
                    "--sensi-metric",
                    "NPV",
                    "--paths",
                    "32",
                    "--output-root",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            case_dir = Path(tmp) / REAL_CASE_XML.parents[1].name
            summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["trade_id"], "SWAP_EUR_5Y_A_flat")
            self.assertEqual(summary["sensitivity"]["metric"], "NPV")
            self.assertGreaterEqual(summary["sensitivity"]["python_factor_count"], 100)
            self.assertTrue((case_dir / "sensitivity.csv").exists())
            self.assertTrue((case_dir / "scenario.csv").exists())

            with open(case_dir / "sensitivity.csv", newline="", encoding="utf-8") as handle:
                sensi_rows = list(csv.DictReader(handle))
            sensi_rows.sort(key=lambda row: abs(float(row["Delta"])), reverse=True)
            self.assertEqual(sensi_rows[0]["Factor_1"], "IndexCurve/EUR-6M/0/5Y")
            self.assertEqual(sensi_rows[0]["Delta"], "-4325.57")
            self.assertEqual(sensi_rows[1]["Factor_1"], "DiscountCurve/EUR/0/5Y")
            self.assertEqual(sensi_rows[1]["Delta"], "-4325.57")

            with open(case_dir / "scenario.csv", newline="", encoding="utf-8") as handle:
                scenario_rows = list(csv.DictReader(handle))
            scenario_rows.sort(key=lambda row: abs(float(row["Difference"])), reverse=True)
            self.assertEqual(scenario_rows[0]["Factor"], "IndexCurve/EUR-6M/0/5Y")
            self.assertEqual(scenario_rows[0]["Up/Down"], "Up")
            self.assertEqual(scenario_rows[0]["Difference"], "-4325.57")
            self.assertEqual(scenario_rows[1]["Factor"], "DiscountCurve/EUR/0/5Y")
            self.assertEqual(scenario_rows[1]["Up/Down"], "Up")
            self.assertEqual(scenario_rows[1]["Difference"], "-4325.57")

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
            "embedded_option_value": 1.75,
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

    def test_ta001_equity_option_price_only_case_uses_native_premium_surface(self):
        payload = ore_snapshot_cli._compute_price_only_case(TA001_EQUITY_CASE_XML, anchor_t0_npv=False)
        self.assertEqual(payload["trade_id"], "EQ_CALL_STOXX50E")
        self.assertEqual(payload["pricing"]["trade_type"], "EquityOption")
        self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
        self.assertEqual(payload["pricing"]["pricing_mode"], "python_equity_option_premium_surface")
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 0.1)

    def test_example22_equity_option_price_only_case_uses_native_black_path(self):
        payload = ore_snapshot_cli._compute_price_only_case(EXAMPLE22_EQUITY_CASE_XML, anchor_t0_npv=False)
        self.assertEqual(payload["trade_id"], "EQ_CALL_SP5")
        self.assertEqual(payload["pricing"]["trade_type"], "EquityOption")
        self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
        self.assertEqual(payload["pricing"]["pricing_mode"], "python_equity_option_black")
        self.assertIn("py_t0_npv", payload["pricing"])
        self.assertGreater(payload["pricing"]["py_t0_npv"], 0.0)
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 100.0)

    def test_unique_report_case_slug_avoids_collisions(self):
        first = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_measure_ba.xml"
        second = TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore_measure_lgm.xml"
        self.assertNotEqual(
            ore_snapshot_cli._unique_report_case_slug(first),
            ore_snapshot_cli._unique_report_case_slug(second),
        )

    def test_reference_source_used_detects_output_expected_and_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_root = root / "Examples" / "Demo"
            input_dir = case_root / "Input"
            output_dir = case_root / "Output"
            expected_dir = case_root / "ExpectedOutput"
            input_dir.mkdir(parents=True)
            output_dir.mkdir()
            expected_dir.mkdir()
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<ORE><Setup>
<Parameter name="outputPath">Output</Parameter>
</Setup></ORE>""",
                encoding="utf-8",
            )
            output_case = {"diagnostics": {"reference_output_dirs": [str(output_dir)]}}
            expected_case = {"diagnostics": {"reference_output_dirs": [str(expected_dir)]}}
            mixed_case = {"diagnostics": {"reference_output_dirs": [str(output_dir), str(expected_dir)]}}
            self.assertEqual(ore_snapshot_cli._reference_source_used(ore_xml, output_case), "output")
            self.assertEqual(ore_snapshot_cli._reference_source_used(ore_xml, expected_case), "expected_output")
            self.assertEqual(ore_snapshot_cli._reference_source_used(ore_xml, mixed_case), "mixed")

    def test_bucket_case_precedence_is_deterministic(self):
        case_summary = {
            "pass_all": False,
            "diagnostics": {
                "missing_reference_xva": True,
                "fallback_reason": "missing_native_output",
                "sample_count_mismatch": True,
            },
            "input_validation": {"input_links_valid": False},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "missing_native_output_fallback")

    def test_bucket_case_prefers_fallback_reason_over_missing_reference_pricing(self):
        case_summary = {
            "pass_all": False,
            "ore_xml": str(TOOLS_DIR / "Examples" / "Exposure" / "Input" / "ore.xml"),
            "diagnostics": {
                "fallback_reason": "missing_native_output",
                "missing_reference_pricing": True,
                "missing_reference_xva": True,
            },
            "input_validation": {"input_links_valid": True},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "missing_native_output_fallback")

    def test_bucket_case_splits_expected_output_fallback_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_root = root / "Examples" / "Demo"
            input_dir = case_root / "Input"
            expected_dir = case_root / "ExpectedOutput"
            input_dir.mkdir(parents=True)
            expected_dir.mkdir()
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text("<ORE><Setup><Parameter name='outputPath'>Output</Parameter></Setup></ORE>", encoding="utf-8")
            case_summary = {
                "pass_all": True,
                "ore_xml": str(ore_xml),
                "diagnostics": {
                    "fallback_reason": "missing_native_output",
                    "reference_output_dirs": [str(expected_dir)],
                },
                "input_validation": {"input_links_valid": True},
            }
            self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "expected_output_fallback_pass")

    def test_bucket_case_reclassifies_unsupported_price_only_fallbacks(self):
        case_summary = {
            "pass_all": True,
            "ore_xml": str(TOOLS_DIR / "Examples" / "Legacy" / "Example_22" / "Input" / "ore_atmOnly.xml"),
            "diagnostics": {
                "pricing_fallback_reason": "missing_simulation_analytic",
            },
            "input_validation": {"input_links_valid": True},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "price_only_reference_fallback")

    def test_bucket_case_reclassifies_unsupported_missing_output_passes(self):
        case_summary = {
            "pass_all": True,
            "ore_xml": str(TOOLS_DIR / "Examples" / "Legacy" / "Example_65" / "Input" / "ore_0.xml"),
            "diagnostics": {
                "fallback_reason": "missing_native_output",
            },
            "input_validation": {"input_links_valid": True},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "unsupported_python_snapshot_fallback")

    def test_bucket_case_marks_native_swap_without_references_as_python_only_no_reference(self):
        case_summary = {
            "pass_all": True,
            "ore_xml": str(TOOLS_DIR / "Examples" / "Legacy" / "Example_35" / "Input" / "ore_Normal.xml"),
            "diagnostics": {
                "fallback_reason": "missing_native_output",
                "reference_output_dirs": [],
            },
            "input_validation": {"input_links_valid": True},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "python_only_no_reference")

    def test_bucket_case_splits_no_reference_artifacts_passes(self):
        case_summary = {
            "pass_all": True,
            "ore_xml": str(TOOLS_DIR / "Examples" / "Legacy" / "Example_35" / "Input" / "ore_Normal.xml"),
            "diagnostics": {
                "fallback_reason": "missing_native_output",
                "reference_output_dirs": [],
            },
            "input_validation": {"input_links_valid": True},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "python_only_no_reference")

    def test_bucket_case_keeps_missing_reference_xva_passes_clean(self):
        case_summary = {
            "pass_all": True,
            "ore_xml": str(FX_OPTION_CASE_XML),
            "diagnostics": {
                "engine": "compare",
                "missing_reference_xva": True,
                "reference_output_dirs": [str(FX_OPTION_CASE_XML.parents[1] / "ExpectedOutput")],
            },
            "input_validation": {"input_links_valid": True},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "clean_pass")

    def test_bucket_case_prefers_parity_fail_over_validation_noise(self):
        case_summary = {
            "pass_all": False,
            "ore_xml": str(SWAPTION_CASE_XML),
            "pricing": {"t0_npv_abs_diff": 123.0},
            "diagnostics": {"engine": "python_price_only"},
            "input_validation": {"input_links_valid": False},
        }
        self.assertEqual(ore_snapshot_cli._bucket_case(case_summary), "parity_threshold_fail")

    def test_write_live_report_artifacts_includes_next_fix_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "ore_xml": "/tmp/a.xml",
                    "case_slug": "case_a",
                    "rc": 1,
                    "pass_all": False,
                    "trade_id": "T1",
                    "modes": "price,xva",
                    "engine": "ore_reference_expected_output",
                    "reference_source_used": "expected_output",
                    "fallback_reason": "missing_native_output",
                    "pricing_fallback_reason": None,
                    "sample_count_mismatch": False,
                    "pricing_abs_diff": None,
                    "cva_rel_diff": None,
                    "dva_rel_diff": None,
                    "fba_rel_diff": None,
                    "fca_rel_diff": None,
                    "parity_ready": None,
                    "bucket": "missing_native_output_fallback",
                    "next_fix_hint": ore_snapshot_cli.REPORT_BUCKET_HINTS["missing_native_output_fallback"],
                    "summary_path": "/tmp/a/summary.json",
                }
            ]
            ore_snapshot_cli._write_live_report_artifacts(root, rows, total_cases=2, top_buckets=5)
            summary = json.loads((root / "live_summary.json").read_text(encoding="utf-8"))
            report = (root / "live_report.md").read_text(encoding="utf-8")
            self.assertEqual(summary["totals_by_bucket"]["missing_native_output_fallback"], 1)
            self.assertIn("next_fix_hint", report)
            self.assertTrue((root / "live_results.csv").exists())

    def test_report_examples_mode_runs_and_writes_live_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples_root = root / "Examples"
            case_dir = examples_root / "FamilyA" / "Input"
            case_dir.mkdir(parents=True)
            ore_a = case_dir / "ore.xml"
            ore_b = case_dir / "ore_alt.xml"
            ore_a.write_text("<ORE />", encoding="utf-8")
            ore_b.write_text("<ORE />", encoding="utf-8")

            def fake_run_case(ore_xml, args, *, artifact_root):
                out_dir = artifact_root / ore_snapshot_cli._case_slug(ore_xml)
                out_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "ore_xml": str(ore_xml),
                    "trade_id": ore_xml.stem,
                    "counterparty": "CPTY",
                    "netting_set_id": "NET",
                    "modes": ["price"],
                    "pricing": {"ore_t0_npv": 1.0, "py_t0_npv": 1.0, "t0_npv_abs_diff": 0.0},
                    "xva": None,
                    "parity": {"parity_ready": True, "summary": {"requested_xva_metrics": []}},
                    "diagnostics": {"engine": "python"},
                    "input_validation": {"input_links_valid": True, "issues": []},
                    "pass_flags": {"pricing": True},
                    "pass_all": True,
                }
                (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with patch("py_ore_tools.ore_snapshot_cli._examples_root", return_value=examples_root):
                with patch("py_ore_tools.ore_snapshot_cli._run_case", side_effect=fake_run_case):
                    rc = ore_snapshot_cli.main(
                        [
                            "--report-examples",
                            "--report-root",
                            str(root / "report"),
                            "--report-workers",
                            "2",
                            "--report-refresh-every",
                            "1",
                        ]
                    )
            self.assertEqual(rc, 0)
            summary = json.loads((root / "report" / "live_summary.json").read_text(encoding="utf-8"))
            rows = list(csv.DictReader((root / "report" / "live_results.csv").open(encoding="utf-8")))
            self.assertEqual(summary["completed_cases"], 2)
            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]["case_slug"], rows[1]["case_slug"])
            self.assertTrue(all(row["summary_path"] for row in rows))

    def test_report_examples_mode_does_not_change_normal_single_case_behavior(self):
        args = ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML), "--price"])
        self.assertFalse(args.report_examples)
        self.assertEqual(args.report_workers, 12)


if __name__ == "__main__":
    unittest.main()
