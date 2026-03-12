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


class TestOreSnapshotCli(unittest.TestCase):
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

    def test_infers_modes_from_ore_xml(self):
        modes = ore_snapshot_cli._infer_modes(
            ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML)]),
            REAL_CASE_XML,
        )
        self.assertTrue(modes.price)
        self.assertTrue(modes.xva)

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


if __name__ == "__main__":
    unittest.main()
