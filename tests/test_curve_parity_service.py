import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import xml.etree.ElementTree as ET

from ore_curve_fit_parity import (
    CurveBuildRequest,
    build_curves_from_ore_inputs,
    compare_python_vs_ore,
    swig_module_available,
    trace_curve,
)
from ore_curve_fit_parity.curve_trace import list_curve_handles_from_todaysmarket
from ore_curve_fit_parity.service import _rewrite_case_paths


class TestCurveParityService(unittest.TestCase):
    ORE_XML = Path(
        "/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/"
        "multiccy_benchmark_final/cases/flat_USD_5Y_B/Input/ore.xml"
    )

    @classmethod
    def setUpClass(cls) -> None:
        if not swig_module_available():
            raise unittest.SkipTest("ORE-SWIG is not available")

    def test_build_curves_from_inputs_runs_swig_and_extracts_traces(self):
        request = CurveBuildRequest(
            ore_xml_path=str(self.ORE_XML),
            currencies=("USD",),
            index_names=("USD-LIBOR-6M",),
        )
        result = build_curves_from_ore_inputs(request)

        self.assertEqual(result.source_engine, "swig")
        self.assertGreaterEqual(len(result.traces), 2)
        self.assertTrue(Path(result.run_root).exists())
        self.assertTrue(Path(result.artifact_snapshot["output"]["curves_csv"]).exists())
        self.assertTrue(Path(result.artifact_snapshot["output"]["todaysmarketcalibration_csv"]).exists())
        self.assertEqual(
            result.artifact_snapshot["selected_curve_handles"],
            ["Yield/USD/USD3M", "Yield/USD/USD6M"],
        )

        by_handle = {trace.curve_handle: trace for trace in result.traces}
        self.assertIn("Yield/USD/USD3M", by_handle)
        self.assertIn("Yield/USD/USD6M", by_handle)
        self.assertEqual(
            by_handle["Yield/USD/USD6M"].payload["dependency_graph"]["USD6M"]["dependencies"],
            ["USD1D", "USD3M"],
        )
        self.assertIsNotNone(result.runtime_seconds)
        self.assertGreater(result.runtime_seconds, 0.0)

    def test_trace_curve_supports_generic_handle(self):
        trace = trace_curve(self.ORE_XML, curve_handle="Yield/USD/USD3M")
        self.assertEqual(trace.curve_id, "USD3M")
        self.assertEqual(trace.curve_name, "USD-LIBOR-3M")
        self.assertEqual(trace.payload["curve_config"]["bootstrap_config"], {})
        self.assertEqual(trace.payload["curve_config"]["interpolation_method"], "LogLinear")

    def test_compare_python_vs_ore_matches_loglinear_discount_grid(self):
        comparison = compare_python_vs_ore(self.ORE_XML, currency="USD")
        self.assertEqual(comparison.curve_handle, "Yield/USD/USD3M")
        self.assertEqual(comparison.status, "ok")
        self.assertGreater(len(comparison.points), 10)
        self.assertLess(comparison.max_abs_error or 0.0, 1.0e-8)
        self.assertLess(comparison.max_rel_error or 0.0, 1.0e-8)

    def test_list_curve_handles_from_todaysmarket_for_libor_config(self):
        handles = list_curve_handles_from_todaysmarket(self.ORE_XML)
        self.assertIn("Yield/USD/USD3M", handles["discounting_curves"])
        self.assertIn("Yield/USD/USD3M", handles["index_forwarding_curves"])
        self.assertIn("Yield/USD/USD6M", handles["index_forwarding_curves"])
        self.assertIn("Yield/EUR/EUR6M", handles["yield_curves"])
        self.assertGreater(len(handles["yield_curves"]), 0)

    def test_rewrite_case_paths_keeps_relative_logfile(self):
        with TemporaryDirectory() as td:
            source_root = Path(td) / "source_case"
            cloned_root = Path(td) / "cloned_case"
            (source_root / "Input").mkdir(parents=True)
            (source_root / "Output").mkdir(parents=True)
            (cloned_root / "Input").mkdir(parents=True)
            (cloned_root / "Output").mkdir(parents=True)

            ore_xml = source_root / "Input" / "ore.xml"
            ore_xml.write_text(
                f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="inputPath">{source_root / "Input"}</Parameter>
    <Parameter name="outputPath">{source_root / "Output"}</Parameter>
    <Parameter name="logFile">log.txt</Parameter>
    <Parameter name="portfolioFile">{source_root / "Input" / "portfolio.xml"}</Parameter>
  </Setup>
  <Analytics>
    <Analytic type="simulation">
      <Parameter name="simulationConfigFile">{source_root / "Input" / "simulation.xml"}</Parameter>
    </Analytic>
  </Analytics>
</ORE>
""",
                encoding="utf-8",
            )
            cloned_xml = cloned_root / "Input" / "ore.xml"
            cloned_xml.write_text(ore_xml.read_text(encoding="utf-8"), encoding="utf-8")

            _rewrite_case_paths(source_root, cloned_root, cloned_xml)
            root = ET.parse(cloned_xml).getroot()
            params = {
                node.attrib["name"]: (node.text or "").strip()
                for node in root.findall("./Setup/Parameter")
            }

            self.assertEqual(params["inputPath"], str(cloned_root / "Input"))
            self.assertEqual(params["outputPath"], str(cloned_root / "Output"))
            self.assertEqual(params["logFile"], "log.txt")
            self.assertEqual(params["portfolioFile"], str(cloned_root / "Input" / "portfolio.xml"))


if __name__ == "__main__":
    unittest.main()
