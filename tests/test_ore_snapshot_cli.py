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
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pythonore.io.ore_snapshot as ore_snapshot_io
from pythonore.compute.inflation import load_inflation_curve_from_market_data, load_zero_inflation_surface_quote
from pythonore.runtime import bermudan as bermudan_runtime
from pythonore.runtime.exposure_profiles import ore_pfe_quantile

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
PORTFOLIO_SWAP_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_6" / "Input" / "ore_portfolio_4.xml"
INFLATION_CAPFLOOR_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input" / "ore_capfloor.xml"
TA001_EQUITY_CASE_XML = TOOLS_DIR / "Examples" / "Academy" / "TA001_Equity_Option" / "Input" / "ore.xml"
EXAMPLE22_EQUITY_CASE_XML = TOOLS_DIR / "Examples" / "Legacy" / "Example_22" / "Input" / "ore_atmOnly.xml"
SCRIPTED_EQUITY_CASE_XML = TOOLS_DIR / "Examples" / "ScriptedTrade" / "Input" / "ore.xml"
SOFR_BASIS_SIMULATION_XML = TOOLS_DIR / "Examples" / "Generated" / "USD_SOFRBasisSnapshot" / "Input" / "simulation.xml"


def _clone_example28_eur_base_case(tmp_root: Path) -> Path:
    case_root = tmp_root / "Examples" / "Legacy" / "Example_28"
    shutil.copytree(TOOLS_DIR / "Examples" / "Legacy" / "Example_28", case_root)
    shutil.copytree(TOOLS_DIR / "Examples" / "Input", tmp_root / "Examples" / "Input")
    output_dir = case_root / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_dir = case_root / "ExpectedOutput"
    for name in ("npv_eur_base.csv", "curves_eur_base.csv", "flows_eur_base.csv"):
        shutil.copy2(expected_dir / name, output_dir / name)
    return case_root / "Input" / "ore_eur_base.xml"


def _clone_example3_case(tmp_root: Path) -> Path:
    case_root = tmp_root / "Examples" / "Legacy" / "Example_3"
    shutil.copytree(TOOLS_DIR / "Examples" / "Legacy" / "Example_3", case_root)
    shutil.copytree(TOOLS_DIR / "Examples" / "Input", tmp_root / "Examples" / "Input")
    return case_root / "Input" / "ore.xml"


def _clone_basis_case_with_simulation(tmp_root: Path, case_name: str) -> Path:
    case_root = tmp_root / "Examples" / "Generated" / case_name
    shutil.copytree(TOOLS_DIR / "Examples" / "Generated" / case_name, case_root)
    shutil.copy2(SOFR_BASIS_SIMULATION_XML, case_root / "Input" / "simulation.xml")

    ore_xml = case_root / "Input" / "ore.xml"
    root = ET.parse(ore_xml).getroot()
    analytics = root.find("./Analytics")
    if analytics is not None:
        for analytic in list(analytics.findall("./Analytic[@type='xva']")):
            analytics.remove(analytic)
    ET.ElementTree(root).write(ore_xml, encoding="utf-8", xml_declaration=True)
    return ore_xml


def test_fingerprint_path_or_resource_changes_when_file_content_changes(tmp_path):
    path = tmp_path / "market.xml"
    path.write_text("<Market><Quote>1</Quote></Market>", encoding="utf-8")
    first = ore_snapshot_io._fingerprint_path_or_resource(path)
    path.write_text("<Market><Quote>2</Quote></Market>", encoding="utf-8")
    second = ore_snapshot_io._fingerprint_path_or_resource(path)

    assert first != second


def _promote_trade_to_first(portfolio_xml: Path, trade_id: str, *, long_short: str | None = None) -> None:
    tree = ET.parse(portfolio_xml)
    root = tree.getroot()
    trade = next((node for node in root.findall("./Trade") if (node.attrib.get("id", "") or "").strip() == trade_id), None)
    if trade is None:
        raise AssertionError(f"trade '{trade_id}' not found in {portfolio_xml}")
    root.remove(trade)
    root.insert(0, trade)
    if long_short is not None:
        option_data = trade.find("./SwaptionData/OptionData")
        if option_data is None:
            raise AssertionError(f"trade '{trade_id}' has no SwaptionData/OptionData in {portfolio_xml}")
        long_short_node = option_data.find("./LongShort")
        if long_short_node is None:
            long_short_node = ET.SubElement(option_data, "LongShort")
        long_short_node.text = long_short
    tree.write(portfolio_xml, encoding="utf-8", xml_declaration=True)


def _patch_swap_floating_tenor(xml_text: str, tenor: str) -> str:
    root = ET.fromstring(xml_text)
    for leg in root.findall(".//LegData"):
        if (leg.findtext("./LegType") or "").strip().upper() != "FLOATING":
            continue
        tenor_node = leg.find("./ScheduleData/Rules/Tenor")
        if tenor_node is not None:
            tenor_node.text = tenor
    return ET.tostring(root, encoding="unicode")


class TestOreSnapshotCli(unittest.TestCase):
    def _write_runtime_artifact_case(self, root: Path, *, marketdata: str = "") -> Path:
        input_dir = root / "Input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "ore.xml").write_text(
            """<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">2026-03-08</Parameter>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="outputPath">Output</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
    <Parameter name="marketDataFile">marketdata.csv</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
    <Parameter name="curveConfigFile">curveconfig.xml</Parameter>
    <Parameter name="conventionsFile">conventions.xml</Parameter>
  </Setup>
  <Markets><Parameter name="pricing">default</Parameter><Parameter name="simulation">default</Parameter></Markets>
  <Analytics>
    <Analytic type="simulation"><Parameter name="active">Y</Parameter><Parameter name="simulationConfigFile">simulation.xml</Parameter></Analytic>
  </Analytics>
</ORE>
""",
            encoding="utf-8",
        )
        (input_dir / "portfolio.xml").write_text(
            """<Portfolio>
  <Trade id="SIFMA_SWAP">
    <TradeType>Swap</TradeType>
    <Envelope><CounterParty>CPTY_A</CounterParty><NettingSetId>NS_A</NettingSetId></Envelope>
    <SwapData>
      <LegData><LegType>Floating</LegType><Currency>USD</Currency><FloatingLegData><Index>USD-SIFMA</Index></FloatingLegData></LegData>
    </SwapData>
  </Trade>
</Portfolio>
""",
            encoding="utf-8",
        )
        (input_dir / "marketdata.csv").write_text(marketdata, encoding="utf-8")
        (input_dir / "simulation.xml").write_text(
            """<Simulation><CrossAssetModel><DomesticCcy>USD</DomesticCcy><InterestRateModels><LGM ccy="USD"><Volatility><TimeGrid>1</TimeGrid><InitialValue>0.01,0.01</InitialValue></Volatility><Reversion><TimeGrid>1</TimeGrid><InitialValue>0.03,0.03</InitialValue></Reversion><ParameterTransformation><Scaling>1</Scaling></ParameterTransformation></LGM></InterestRateModels></CrossAssetModel></Simulation>""",
            encoding="utf-8",
        )
        for name in ("todaysmarket.xml", "curveconfig.xml", "conventions.xml"):
            (input_dir / name).write_text("<Root/>", encoding="utf-8")
        return input_dir / "ore.xml"

    def test_runtime_artifact_preflight_builds_missing_curves_and_lgm_params(self):
        with tempfile.TemporaryDirectory() as td:
            ore_xml = self._write_runtime_artifact_case(
                Path(td),
                marketdata="2026-03-08,BMA_SWAP/RATIO/USD/3M/2Y,0.72\n",
            )
            fake_ore = Path(td) / "ore"
            fake_ore.write_text("#!/bin/sh\n", encoding="utf-8")

            def fake_run(_cmd, **_kwargs):
                output = Path(td) / "Output"
                output.mkdir(exist_ok=True)
                (output / "curves.csv").write_text("Date,USD\n2026-03-08,1.0\n2027-03-08,0.97\n", encoding="utf-8")
                (output / "calibration.xml").write_text(
                    """<Calibration><InterestRateModels><LGM key="USD"><Volatility><TimeGrid>1</TimeGrid><InitialValue>0.01,0.01</InitialValue></Volatility><Reversion><TimeGrid>1</TimeGrid><InitialValue>0.03,0.03</InitialValue></Reversion><ParameterTransformation><Scaling>1</Scaling></ParameterTransformation></LGM></InterestRateModels></Calibration>""",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(ore_snapshot_cli, "default_ore_bin", return_value=fake_ore), patch.object(
                ore_snapshot_cli.subprocess, "run", side_effect=fake_run
            ):
                status = ore_snapshot_cli._ensure_ore_snapshot_runtime_artifacts(
                    ore_xml,
                    model_ccy="USD",
                    require_lgm_calibration=True,
                )

            self.assertTrue(status["constructed"])
            self.assertTrue((Path(td) / "Output" / "curves.csv").exists())
            self.assertTrue((Path(td) / "Output" / "calibration.xml").exists())

    def test_runtime_artifact_preflight_fails_sifma_without_bma_ratio_quote(self):
        with tempfile.TemporaryDirectory() as td:
            ore_xml = self._write_runtime_artifact_case(Path(td), marketdata="")
            with self.assertRaisesRegex(RuntimeError, "missing_bma_ratio_curve:USD-SIFMA"):
                ore_snapshot_cli._ensure_ore_snapshot_runtime_artifacts(
                    ore_xml,
                    model_ccy="USD",
                    require_lgm_calibration=True,
                )

    def test_runtime_artifact_preflight_errors_if_ore_build_still_leaves_missing_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            ore_xml = self._write_runtime_artifact_case(
                Path(td),
                marketdata="2026-03-08,BMA_SWAP/RATIO/USD/3M/2Y,0.72\n",
            )
            fake_ore = Path(td) / "ore"
            fake_ore.write_text("#!/bin/sh\n", encoding="utf-8")
            with patch.object(ore_snapshot_cli, "default_ore_bin", return_value=fake_ore), patch.object(
                ore_snapshot_cli.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ):
                with self.assertRaisesRegex(RuntimeError, "still unavailable"):
                    ore_snapshot_cli._ensure_ore_snapshot_runtime_artifacts(
                        ore_xml,
                        model_ccy="USD",
                        require_lgm_calibration=True,
                    )

    def test_resolve_case_portfolio_path_uses_setup_portfolio_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir(parents=True)
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<ORE>
  <Setup>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="portfolioFile">portfolio_a.xml</Parameter>
  </Setup>
</ORE>
""",
                encoding="utf-8",
            )
            (input_dir / "portfolio_a.xml").write_text("<Portfolio />", encoding="utf-8")

            resolved = ore_snapshot_cli._resolve_case_portfolio_path(ore_xml)

            self.assertEqual(resolved, (input_dir / "portfolio_a.xml").resolve())

    def test_resolve_case_portfolio_path_rejects_blank_portfolio_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir(parents=True)
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<ORE>
  <Setup>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="portfolioFile"></Parameter>
  </Setup>
</ORE>
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Setup/portfolioFile is empty"):
                ore_snapshot_cli._resolve_case_portfolio_path(ore_xml)

    def test_terminal_preflight_summary_includes_resolved_portfolio_xml(self):
        summary = {
            "ore_xml": "/tmp/case/Input/ore.xml",
            "portfolio_xml": "/tmp/case/Input/portfolio.xml",
            "requested_modes": ["price", "xva"],
            "validation": {"input_links_valid": True},
            "native_ready": True,
            "hybrid_ready": True,
            "support": {
                "native_trade_count": 1,
                "requires_swig_trade_count": 0,
                "requires_swig_trade_types": [],
            },
            "next_step": "run native",
        }

        text = ore_snapshot_cli._render_terminal_preflight_summary(summary)

        self.assertIn("portfolio_xml=/tmp/case/Input/portfolio.xml", text)

    def test_simulate_with_fixing_grid_uses_exposure_grid_then_bridges_fixings_for_ore_path_major(self):
        class _DummyModel:
            _measure = "LGM"

            @staticmethod
            def zeta(t):
                return np.asarray(t, dtype=float)

        class _DummyRng:
            seed = 42

        exposure_times = np.array([0.0, 0.5, 1.0], dtype=float)
        fixing_times = np.array([0.25, 0.75], dtype=float)
        base_x = np.array(
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [2.0, -2.0],
            ],
            dtype=float,
        )

        calls = []

        def _fake_simulate(model, times, n_paths, rng=None, x0=0.0, draw_order="time_major"):
            calls.append(np.asarray(times, dtype=float).copy())
            return base_x.copy()

        with patch("py_ore_tools.ore_snapshot_cli.simulate_lgm_measure", side_effect=_fake_simulate):
            x_exp, x_all, sim_times, y_exp, y_all = ore_snapshot_cli._simulate_with_fixing_grid(
                model=_DummyModel(),
                exposure_times=exposure_times,
                fixing_times=fixing_times,
                n_paths=2,
                rng=_DummyRng(),
                draw_order="ore_path_major",
            )

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0], exposure_times)
        np.testing.assert_allclose(x_exp, base_x)
        np.testing.assert_allclose(sim_times, np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float))
        exp_idx = np.searchsorted(sim_times, exposure_times)
        np.testing.assert_allclose(x_all[exp_idx, :], base_x)
        self.assertEqual(y_exp is None, True)
        self.assertEqual(y_all is None, True)

    def test_compute_snapshot_case_splits_float_legs_by_index(self):
        class _DummyModel:
            _measure = "LGM"

            @staticmethod
            def zeta(t):
                return np.asarray(t, dtype=float)

        class _DummySnapshot:
            trade_id = "TEST_SWAP"
            counterparty = "CP_1"
            netting_set_id = "NET_1"
            asof_date = "2024-01-02"
            measure = "LGM"
            discount_column = "USD-OIS"
            forward_column = "USD-LIBOR-3M"
            xva_discount_column = "USD-OIS"
            p0_disc = staticmethod(lambda t: 1.0 - 0.01 * float(t))
            p0_fwd = staticmethod(lambda t: 1.0 - 0.02 * float(t))
            p0_xva_disc = staticmethod(lambda t: 1.0 - 0.01 * float(t))
            recovery = 0.4
            ore_t0_npv = 0.0
            ore_cva = 0.0
            ore_dva = 0.0
            ore_fba = 0.0
            ore_fca = 0.0
            ore_basel_epe = 0.0
            ore_basel_eepe = 0.0
            ore_maturity_date = "2025-01-02"
            ore_maturity_time = 1.0
            ore_epe = np.array([0.0, 0.0], dtype=float)
            ore_ene = np.array([0.0, 0.0], dtype=float)
            hazard_times = np.array([1.0], dtype=float)
            hazard_rates = np.array([0.01], dtype=float)
            own_hazard_times = None
            own_hazard_rates = None
            own_recovery = None
            borrowing_curve_column = None
            lending_curve_column = None
            p0_borrow = None
            p0_lend = None
            model_day_counter = "A365F"
            report_day_counter = "ActualActual(ISDA)"
            exposure_model_times = np.array([0.0, 1.0], dtype=float)
            exposure_dates = np.array(["2024-01-02", "2025-01-02"], dtype=object)
            exposure_times = np.array([0.0, 1.0], dtype=float)
            n_samples = 4
            curves_csv_path = ""
            todaysmarket_xml_path = ""
            leg_source = "portfolio"
            requested_xva_metrics = ()
            portfolio_xml_path = None
            trade_float_index2 = "USD-SOFR-3M"

            def build_model(self):
                return _DummyModel()

            def survival_probability(self, times):
                return np.ones_like(np.asarray(times, dtype=float))

            def parity_completeness_report(self):
                return {"summary": {"trade_id": self.trade_id}}

            legs = {
                "fixed_pay_time": np.array([], dtype=float),
                "fixed_start_time": np.array([], dtype=float),
                "fixed_end_time": np.array([], dtype=float),
                "fixed_accrual": np.array([], dtype=float),
                "fixed_rate": np.array([], dtype=float),
                "fixed_notional": np.array([], dtype=float),
                "fixed_sign": np.array([], dtype=float),
                "fixed_amount": np.array([], dtype=float),
                "float_pay_time": np.array([0.5, 1.0, 0.5, 1.0], dtype=float),
                "float_start_time": np.array([0.1, 0.6, 0.1, 0.6], dtype=float),
                "float_end_time": np.array([0.5, 1.0, 0.5, 1.0], dtype=float),
                "float_accrual": np.array([0.4, 0.4, 0.4, 0.4], dtype=float),
                "float_notional": np.array([1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
                "float_sign": np.array([-1.0, -1.0, 1.0, 1.0], dtype=float),
                "float_gearing": np.array([1.25, 1.25, 1.75, 1.75], dtype=float),
                "float_spread": np.array([0.001, 0.001, -0.0005, -0.0005], dtype=float),
                "float_coupon": np.zeros(4, dtype=float),
                "float_amount": np.zeros(4, dtype=float),
                "float_fixing_time": np.array([0.0, 0.5, 0.0, 0.5], dtype=float),
                "float_index_accrual": np.array([0.4, 0.4, 0.4, 0.4], dtype=float),
                "float_is_averaged": np.array([False, False, False, False], dtype=bool),
                "float_leg_index": np.array([0, 0, 1, 1], dtype=int),
                "float_leg0_count": np.array([2], dtype=int),
                "float_index": "USD-LIBOR-3M",
                "float_index_tenor": "3M",
                "float_index_day_counter": "A365F",
                "float_fixing_source": "portfolio_fixing_days",
                "float_index_by_leg": np.array(["USD-LIBOR-3M", "USD-SOFR-3M"], dtype=object),
                "node_tenors": np.array([], dtype=float),
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ore_xml = root / "ore.xml"
            ore_xml.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">2024-01-02</Parameter>
  </Setup>
  <Markets>
    <Parameter name="pricing">libor</Parameter>
  </Markets>
</ORE>
""",
                encoding="utf-8",
            )
            curves_csv = root / "curves.csv"
            curves_csv.write_text("Date,USD-LIBOR-3M,USD-SOFR-3M\n2024-01-02,1.0,1.0\n2025-01-02,0.98,0.97\n", encoding="utf-8")

            fake_snap = _DummySnapshot()
            fake_snap.curves_csv_path = str(curves_csv)
            fake_snap.todaysmarket_xml_path = str(root / "missing_todaysmarket.xml")

            curve_load_calls: list[str] = []
            realized_calls: list[tuple[str, float]] = []
            price_calls: list[tuple[str, float]] = []
            gearing_calls: list[np.ndarray] = []

            def _fake_curve_pairs(path, columns, **kwargs):
                column = str(columns[0])
                curve_load_calls.append(column)
                if column == "USD-SOFR-3M":
                    dfs = np.array([1.0, 0.97], dtype=float)
                else:
                    dfs = np.array([1.0, 0.98], dtype=float)
                return {column: (np.array(["2024-01-02", "2025-01-02"], dtype=object), np.array([0.0, 1.0], dtype=float), dfs)}

            def _fake_realized(*, p0_fwd, legs, **kwargs):
                realized_calls.append((str(legs.get("float_index", "")), float(p0_fwd(1.0))))
                count = int(np.asarray(legs.get("float_pay_time", []), dtype=float).size)
                return np.zeros((count, 2), dtype=float)

            def _fake_swap_npv(model, p0_disc, p0_fwd, legs, t, x_t, realized_float_coupon=None, **kwargs):
                size = int(np.asarray(legs.get("float_pay_time", []), dtype=float).size)
                price_calls.append((size, str(legs.get("float_index", "")), float(p0_fwd(1.0))))
                gearing_calls.append(np.asarray(legs.get("float_gearing", []), dtype=float).copy())
                if size == 0:
                    return np.zeros_like(np.asarray(x_t, dtype=float), dtype=float)
                return np.full_like(np.asarray(x_t, dtype=float), float(p0_fwd(1.0)), dtype=float)

            with patch("pythonore.workflows.ore_snapshot_cli.load_from_ore_xml", return_value=fake_snap), patch(
                "pythonore.workflows.ore_snapshot_cli._simulate_with_fixing_grid",
                return_value=(
                    np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
                    np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
                    np.array([0.0, 1.0], dtype=float),
                    None,
                    None,
                ),
            ), patch(
                "pythonore.workflows.ore_snapshot_cli._ore_exposure_quantile",
                return_value=0.95,
                ), patch(
                    "pythonore.workflows.ore_snapshot_cli.build_ore_exposure_profile_from_paths",
                    side_effect=lambda entity_id, *args, **kwargs: {
                        "entity_id": entity_id,
                        "dates": ["2024-01-02", "2025-01-02"],
                        "times": [0.0, 1.0],
                        "pfe": [0.0, 0.0],
                        "time_weighted_basel_epe": [0.0, 0.0],
                        "time_weighted_basel_eepe": [0.0, 0.0],
                    },
                ), patch(
                "pythonore.workflows.ore_snapshot_cli.compute_realized_float_coupons",
                side_effect=_fake_realized,
            ), patch(
                "pythonore.workflows.ore_snapshot_cli.swap_npv_from_ore_legs_dual_curve",
                side_effect=_fake_swap_npv,
            ), patch(
                "pythonore.workflows.ore_snapshot_cli.ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter",
                side_effect=_fake_curve_pairs,
            ):
                result = ore_snapshot_cli._compute_snapshot_case(
                    ore_xml,
                    paths=2,
                    seed=7,
                    rng_mode="numpy",
                    anchor_t0_npv=False,
                    own_hazard=0.01,
                    own_recovery=0.4,
                    xva_mode="numpy",
                )

        self.assertEqual(curve_load_calls.count("USD-SOFR-3M"), 1)
        self.assertEqual(realized_calls, [("USD-LIBOR-3M", 0.98), ("USD-SOFR-3M", 0.97)])
        expected_price_calls = [
            (0, "USD-LIBOR-3M", 0.98),
            (2, "USD-LIBOR-3M", 0.98),
            (2, "USD-SOFR-3M", 0.97),
        ]
        self.assertEqual(price_calls, expected_price_calls * 2)
        self.assertTrue(np.allclose(gearing_calls[0], np.array([], dtype=float)))
        self.assertTrue(np.allclose(gearing_calls[1], np.array([1.25, 1.25], dtype=float)))
        self.assertTrue(np.allclose(gearing_calls[2], np.array([1.75, 1.75], dtype=float)))
        self.assertTrue(np.allclose(gearing_calls[3], np.array([], dtype=float)))
        self.assertTrue(np.allclose(gearing_calls[4], np.array([1.25, 1.25], dtype=float)))
        self.assertTrue(np.allclose(gearing_calls[5], np.array([1.75, 1.75], dtype=float)))
        self.assertAlmostEqual(float(result.pricing["py_t0_npv"]), 1.95, places=12)

    def test_multi_curve_basis_examples_keep_analytics_and_simulation_markets_separate(self):
        for case_name, expected_ore_npv in (
            ("USD_SIFMA_SOFRBasisLong", -120463.206785),
            ("USD_SIFMA_SOFRBasisShort", 120463.206785),
        ):
            with self.subTest(case=case_name):
                with tempfile.TemporaryDirectory() as tmp:
                    ore_xml = _clone_basis_case_with_simulation(Path(tmp), case_name)
                    snap = ore_snapshot_io.load_from_ore_xml(ore_xml)

                self.assertEqual(snap.domestic_ccy, "USD")
                self.assertEqual(snap.discount_column, "USD-IN-EUR")
                self.assertEqual(snap.forward_column, "USD-SOFR-3M")
                self.assertEqual(snap.trade_float_index2, "USD-SIFMA")
                self.assertEqual(
                    Path(snap.simulation_xml_path).resolve(),
                    SOFR_BASIS_SIMULATION_XML.resolve(),
                )
                self.assertAlmostEqual(float(snap.ore_t0_npv), expected_ore_npv, places=6)

                sim_root = ET.parse(snap.simulation_xml_path).getroot()
                sim_indices = {
                    (node.text or "").strip()
                    for node in sim_root.findall("./Market/Indices/Index")
                    if (node.text or "").strip()
                }
                self.assertTrue(
                    {"USD-SOFR", "USD-SOFR-3M", "USD-SIFMA", "USD-LIBOR-3M"}.issubset(sim_indices)
                )

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

    def test_is_plain_vanilla_swap_trade_handles_multi_swap_portfolio_trade_override(self):
        ore_xml = TOOLS_DIR / "Examples" / "Legacy" / "Example_51" / "Input" / "ore.xml"

        self.assertTrue(ore_snapshot_cli._is_plain_vanilla_swap_trade(ore_xml, trade_id="Swap_Euribor"))
        self.assertTrue(ore_snapshot_cli._is_plain_vanilla_swap_trade(ore_xml, trade_id="Swap_OIS"))

    def test_example51_multi_swap_no_simulation_replays_reference_flows(self):
        ore_xml = TOOLS_DIR / "Examples" / "Legacy" / "Example_51" / "Input" / "ore.xml"

        summary = ore_snapshot_cli._compute_portfolio_price_case(
            ore_xml,
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )

        self.assertEqual(summary["pricing"]["trade_type"], "Portfolio")
        self.assertLess(summary["pricing"]["t0_npv_abs_diff"], 1.0e-6)
        self.assertTrue(summary["diagnostics"]["using_expected_output"])
        self.assertTrue(all(row["diagnostics"].get("cashflow_replay") for row in summary["portfolio_trade_rows"]))

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
                "ScriptedTrade",
                SCRIPTED_EQUITY_CASE_XML,
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

    def test_capfloor_validator_keeps_pay_date_on_valuation_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flows_csv = root / "flows.csv"
            flows_csv.write_text(
                "TradeId,Type,FlowType,PayDate,AccrualStartDate,AccrualEndDate,Accrual,Notional,Gearing,Spread,fixingDate,Currency,CapStrike,FloorStrike\n"
                "CF1,CapFloor,interest,2026-09-08,2026-03-08,2026-09-08,0.500000,1000000,1.0,0.0,2026-03-06,EUR,0.03,\n",
                encoding="utf-8",
            )
            defs_from_flows = ore_snapshot_cli._build_capfloor_defs_from_flows(
                flows_csv,
                trade_id="CF1",
                asof_date="2026-09-08",
                option_bias=1.0,
            )
            self.assertEqual(len(defs_from_flows), 1)
            self.assertEqual(defs_from_flows[0].pay_time.size, 1)
            self.assertAlmostEqual(float(defs_from_flows[0].pay_time[0]), 0.0, places=12)

            portfolio_xml = root / "portfolio.xml"
            portfolio_xml.write_text(
                """
<Portfolio>
  <Trade id="CF1">
    <TradeType>CapFloor</TradeType>
    <CapFloorData>
      <LongShort>Long</LongShort>
      <Caps><Cap>0.03</Cap></Caps>
      <LegData>
        <LegType>Floating</LegType>
        <Currency>EUR</Currency>
        <PaymentConvention>F</PaymentConvention>
        <DayCounter>A360</DayCounter>
        <Notionals><Notional>1000000</Notional></Notionals>
        <ScheduleData>
          <Dates>
            <Date>2026-03-08</Date>
            <Date>2026-09-08</Date>
          </Dates>
        </ScheduleData>
        <FloatingLegData>
          <Index>EUR-EURIBOR-6M</Index>
          <FixingDays>2</FixingDays>
          <IsInArrears>false</IsInArrears>
          <Spreads><Spread>0.0</Spread></Spreads>
          <Gearings><Gearing>1.0</Gearing></Gearings>
        </FloatingLegData>
      </LegData>
    </CapFloorData>
  </Trade>
</Portfolio>
""".strip(),
                encoding="utf-8",
            )
            tree = ET.parse(portfolio_xml)
            trade = tree.getroot().find("./Trade")
            self.assertIsNotNone(trade)
            defs_from_portfolio = ore_snapshot_cli._build_capfloor_defs_from_portfolio(
                trade.find("./CapFloorData"),  # type: ignore[union-attr]
                trade.find("./CapFloorData/LegData"),  # type: ignore[union-attr]
                trade_id="CF1",
                asof_date="2026-09-08",
                option_bias=1.0,
            )
            self.assertEqual(len(defs_from_portfolio), 1)
            self.assertEqual(defs_from_portfolio[0].pay_time.size, 1)
            self.assertAlmostEqual(float(defs_from_portfolio[0].pay_time[0]), 0.0, places=12)

    def test_market_data_read_accepts_compact_and_dashed_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market = root / "market.txt"
            market.write_text(
                "\n".join(
                    [
                        "20260102 QUOTE/COMPACT 1.5",
                        "2026-01-02 QUOTE/DASHED 2.5",
                        "20260102 SWAPTION/RATE_NVOL/USD/1Y/5Y/ATM 0.02",
                        "2026-01-02 ZC_INFLATIONSWAP/RATE/USD-CPI/5Y 0.03",
                        "20260102 ZC_INFLATIONCAPFLOOR/PRICE/USD-CPI/5Y/C/0.04 7.89",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(ore_snapshot_io._normalize_date_input("20260102").isoformat(), "2026-01-02")
            self.assertEqual(ore_snapshot_cli._load_market_quote_value(market, "2026-01-02", "QUOTE/COMPACT"), 1.5)
            self.assertEqual(ore_snapshot_cli._load_market_quote_value(market, "20260102", "QUOTE/DASHED"), 2.5)

            quotes_dash = ore_snapshot_cli._parse_market_quotes(market, "2026-01-02")
            quotes_compact = ore_snapshot_cli._parse_market_quotes(market, "20260102")
            self.assertEqual(quotes_dash["QUOTE/COMPACT"], 1.5)
            self.assertEqual(quotes_compact["QUOTE/DASHED"], 2.5)

            vols = ore_snapshot_io._load_lgm_swaption_quotes(
                market_data_path=market,
                asof_date="20260102",
                currency="USD",
                option_tenors=["1Y"],
                swap_tenors=["5Y"],
                volatility_type="Normal",
            )
            self.assertAlmostEqual(float(vols[0, 0]), 0.02)

            infl_curve = load_inflation_curve_from_market_data(market, "2026-01-02", "USD-CPI", curve_type="ZC")
            self.assertEqual(infl_curve.times, (5.0,))
            self.assertAlmostEqual(infl_curve.rates[0], 0.03)

            infl_quote = load_zero_inflation_surface_quote(market, "20260102", "USD-CPI", "5Y", 0.04, "Call")
            self.assertAlmostEqual(float(infl_quote), 7.89)

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

    def test_portfolio_trade_lookup_caches_trade_index(self):
        portfolio_root = ore_snapshot_io.ET.fromstring(
            """
            <Portfolio>
              <Trade id="T1">
                <TradeType>Swap</TradeType>
                <Envelope>
                  <CounterParty>CP_A</CounterParty>
                  <NettingSetId>NS_A</NettingSetId>
                </Envelope>
                <SwapData>
                  <LegData>
                    <FloatingLegData>
                      <Index>EUR-EURIBOR-6M</Index>
                    </FloatingLegData>
                  </LegData>
                </SwapData>
              </Trade>
            </Portfolio>
            """
        )

        first = ore_snapshot_io._portfolio_trade_lookup(portfolio_root)
        second = ore_snapshot_io._portfolio_trade_lookup(portfolio_root)

        self.assertIs(first, second)
        self.assertEqual(ore_snapshot_io._get_trade_type(portfolio_root, "T1"), "Swap")
        self.assertEqual(ore_snapshot_io._get_cpty_from_portfolio(portfolio_root, "T1"), "CP_A")
        self.assertEqual(ore_snapshot_io._get_netting_set_from_portfolio(portfolio_root, "T1"), "NS_A")
        self.assertEqual(ore_snapshot_io._get_float_index(portfolio_root, "T1"), "EUR-EURIBOR-6M")

    def test_portfolio_trade_lookup_cache_is_guarded_by_root_identity(self):
        portfolio_root_a = ore_snapshot_io.ET.fromstring(
            """
            <Portfolio>
              <Trade id="T1"><TradeType>Swap</TradeType></Trade>
            </Portfolio>
            """
        )
        portfolio_root_b = ore_snapshot_io.ET.fromstring(
            """
            <Portfolio>
              <Trade id="FXNDF"><TradeType>FxForward</TradeType></Trade>
            </Portfolio>
            """
        )
        cache_key = id(portfolio_root_b)
        original = dict(ore_snapshot_io._PORTFOLIO_TRADE_LOOKUP_CACHE)
        try:
            ore_snapshot_io._PORTFOLIO_TRADE_LOOKUP_CACHE[cache_key] = (
                portfolio_root_a,
                {"T1": portfolio_root_a.find("./Trade")},
            )
            self.assertEqual(ore_snapshot_io._get_trade_type(portfolio_root_b, "FXNDF"), "FxForward")
        finally:
            ore_snapshot_io._PORTFOLIO_TRADE_LOOKUP_CACHE.clear()
            ore_snapshot_io._PORTFOLIO_TRADE_LOOKUP_CACHE.update(original)

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

    def test_price_only_fx_forward_matches_explicit_ore_npv(self):
        with tempfile.TemporaryDirectory() as tmp:
            ore_xml = _clone_example28_eur_base_case(Path(tmp))
            payload = ore_snapshot_cli._compute_price_only_case(
                ore_xml,
                anchor_t0_npv=False,
                use_reference_artifacts=True,
            )
        self.assertEqual(payload["trade_id"], "FXFWD_1Y")
        self.assertEqual(payload["pricing"]["trade_type"], "FxForward")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_forward")
        self.assertFalse(payload["diagnostics"]["using_expected_output"])
        self.assertAlmostEqual(float(payload["pricing"]["ore_t0_npv"]), 1094.870785, places=6)
        self.assertAlmostEqual(
            float(payload["pricing"]["py_t0_npv"]),
            float(payload["pricing"]["ore_t0_npv"]),
            places=6,
        )
        self.assertLess(float(payload["pricing"]["t0_npv_abs_diff"]), 1.0e-4)

    def test_preflight_mode_writes_support_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout = io.StringIO()
            support = {
                "mode": "native_only",
                "native_only": True,
                "python_supported": False,
                "native_trade_ids": ["IRS_A"],
                "native_trade_types": ["Swap"],
                "requires_swig_trade_ids": ["EQ_1"],
                "requires_swig_trade_types": ["EquityOption"],
                "native_trade_count": 1,
                "requires_swig_trade_count": 1,
            }
            validation = {
                "input_links_valid": True,
                "issues": [],
                "summary": {"source_mode": "ore_xml"},
            }
            with patch("py_ore_tools.ore_snapshot_cli.load_from_ore_xml", return_value=SimpleNamespace()):
                with patch("py_ore_tools.ore_snapshot_cli._classify_preflight_support", return_value=support):
                    with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value=validation):
                        with redirect_stdout(stdout):
                            rc = ore_snapshot_cli.main(
                                [
                                    str(REAL_CASE_XML),
                                    "--preflight",
                                    "--output-root",
                                    str(root / "artifacts"),
                                ]
                            )
            self.assertEqual(rc, 0)
            text = stdout.getvalue()
            self.assertIn("PRECHECK done.", text)
            self.assertIn("requires_swig_trade_types=EquityOption", text)
            payload = json.loads((root / "artifacts" / "flat_EUR_5Y_A" / "preflight.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["native_ready"])
            self.assertTrue(payload["hybrid_ready"])
            self.assertEqual(payload["support"]["requires_swig_trade_ids"], ["EQ_1"])
            self.assertTrue((root / "artifacts" / "flat_EUR_5Y_A" / "preflight.md").exists())
            self.assertTrue((root / "artifacts" / "flat_EUR_5Y_A" / "preflight_support.csv").exists())

    def test_preflight_mode_lists_multiple_trades_and_netting_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_root = root / "case"
            input_dir = case_root / "Input"
            input_dir.mkdir(parents=True)
            ore_xml = input_dir / "ore.xml"
            portfolio_xml = input_dir / "portfolio.xml"
            ore_xml.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<ORE>
  <Setup>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
  </Setup>
</ORE>
""",
                encoding="utf-8",
            )
            portfolio_xml.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<Portfolio>
  <Trade id="SWAP_A">
    <TradeType>Swap</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>NS_A</NettingSetId>
    </Envelope>
  </Trade>
  <Trade id="FXFWD_B">
    <TradeType>FxForward</TradeType>
    <Envelope>
      <CounterParty>CPTY_A</CounterParty>
      <NettingSetId>NS_B</NettingSetId>
    </Envelope>
  </Trade>
</Portfolio>
""",
                encoding="utf-8",
            )
            snapshot = SimpleNamespace(portfolio=SimpleNamespace(trades=[object(), object()]))
            support = {
                "mode": "native_only",
                "native_only": True,
                "python_supported": True,
                "native_trade_ids": ["SWAP_A", "FXFWD_B"],
                "native_trade_types": ["FxForward", "Swap"],
                "requires_swig_trade_ids": [],
                "requires_swig_trade_types": [],
                "native_trade_count": 2,
                "requires_swig_trade_count": 0,
            }
            validation = {
                "input_links_valid": True,
                "issues": [],
                "summary": {"source_mode": "ore_xml"},
            }
            with patch("py_ore_tools.ore_snapshot_cli.load_from_ore_xml", return_value=snapshot):
                with patch("py_ore_tools.ore_snapshot_cli.classify_portfolio_support", return_value=support):
                    with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value=validation):
                        with redirect_stdout(io.StringIO()):
                            rc = ore_snapshot_cli.main(
                                [
                                    str(ore_xml),
                                    "--preflight",
                                    "--output-root",
                                    str(root / "artifacts"),
                                ]
                            )
            self.assertEqual(rc, 0)
            payload = json.loads((root / "artifacts" / "case" / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["portfolio_trade_count"], 2)
            self.assertEqual(payload["portfolio_netting_set_count"], 2)
            self.assertEqual(payload["portfolio_netting_sets"], ["NS_A", "NS_B"])
            self.assertEqual(payload["support"]["native_trade_count"], 2)
            with open(root / "artifacts" / "case" / "preflight_support.csv", encoding="utf-8") as handle:
                support_rows = list(csv.DictReader(handle))
            self.assertIn("portfolio_netting_set_ids", {row["bucket"] for row in support_rows})
            self.assertIn("SWAP_A", {row["entity_id"] for row in support_rows})
            self.assertIn("FXFWD_B", {row["entity_id"] for row in support_rows})

    def test_portfolio_contains_swap_like_trade_handles_multiple_netting_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "Input"
            input_dir.mkdir(parents=True)
            ore_xml = input_dir / "ore.xml"
            ore_xml.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<ORE>
  <Setup>
    <Parameter name="inputPath">Input</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
  </Setup>
</ORE>
""",
                encoding="utf-8",
            )
            (input_dir / "portfolio.xml").write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<Portfolio>
  <Trade id="BOND_A"><TradeType>Bond</TradeType></Trade>
  <Trade id="SWAP_A"><TradeType>Swap</TradeType></Trade>
</Portfolio>
""",
                encoding="utf-8",
            )
            self.assertTrue(ore_snapshot_cli._portfolio_contains_swap_like_trade(ore_xml))

    def test_price_only_fx_ndf_uses_cash_settlement_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(FX_NDF_CASE_XML),
                    "--price",
                    "--engine",
                    "python",
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

    def test_price_only_fx_ndf_matches_explicit_ore_npv(self):
        payload = ore_snapshot_cli._compute_price_only_case(FX_NDF_CASE_XML, anchor_t0_npv=False)
        self.assertEqual(payload["trade_id"], "FXNDF")
        self.assertEqual(payload["pricing"]["trade_type"], "FxForward")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_fx_forward")
        self.assertTrue(payload["diagnostics"]["using_expected_output"])
        self.assertAlmostEqual(float(payload["pricing"]["ore_t0_npv"]), 227467.592353, places=6)
        self.assertAlmostEqual(
            float(payload["pricing"]["py_t0_npv"]),
            float(payload["pricing"]["ore_t0_npv"]),
            places=4,
        )
        self.assertLess(float(payload["pricing"]["t0_npv_abs_diff"]), 1.0e-3)

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
                    "--trade-id",
                    "2:EquityOption",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "ScriptedTrade" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["trade_id"], "2:EquityOption")
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["pricing"]["trade_type"], "ScriptedTrade")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_scripted_trade_ir_mc")
            self.assertEqual(payload["diagnostics"]["script_source"], "inline")
            self.assertIn("py_t0_npv", payload["pricing"])

    def test_price_only_scripted_trade_default_trade_falls_back_for_wildcard_tenor(self):
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
            self.assertEqual(rc, 0)
            payload = json.loads((root / "artifacts" / "ScriptedTrade" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["trade_id"], "1:EquityOption:AN")
            self.assertEqual(payload["diagnostics"]["engine"], "ore_reference_expected_output")
            self.assertEqual(payload["diagnostics"]["fallback_reason"], "unsupported_python_snapshot")
            self.assertIn("Unsupported tenor '*'", payload["diagnostics"]["fallback_error"])
            self.assertEqual(set(payload["pricing"].keys()), {"ore_t0_npv"})

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

    def test_price_only_swaption_premium_trades_match_ore_expected_output(self):
        for base_trade_id, premium_trade_id in (("SwaptionCash", "SwaptionCashPremium"), ("SwaptionPhysical", "SwaptionPhysicalPremium")):
            with self.subTest(base_trade_id=base_trade_id, premium_trade_id=premium_trade_id):
                with tempfile.TemporaryDirectory() as tmp_base, tempfile.TemporaryDirectory() as tmp_premium:
                    root_base = Path(tmp_base)
                    root_premium = Path(tmp_premium)
                    ore_xml_base = _clone_example3_case(root_base)
                    ore_xml_premium = _clone_example3_case(root_premium)
                    _promote_trade_to_first(ore_xml_base.parent / "portfolio.xml", base_trade_id)
                    _promote_trade_to_first(ore_xml_premium.parent / "portfolio.xml", premium_trade_id)
                    rc_base = ore_snapshot_cli.main(
                        [
                            str(ore_xml_base),
                            "--price",
                            "--output-root",
                            str(root_base / "artifacts"),
                        ]
                    )
                    rc_premium = ore_snapshot_cli.main(
                        [
                            str(ore_xml_premium),
                            "--price",
                            "--output-root",
                            str(root_premium / "artifacts"),
                        ]
                    )
                    self.assertIn(rc_base, (0, 1))
                    self.assertIn(rc_premium, (0, 1))
                    payload_base = json.loads((root_base / "artifacts" / "Example_3" / "summary.json").read_text(encoding="utf-8"))
                    payload_premium = json.loads((root_premium / "artifacts" / "Example_3" / "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(payload_base["trade_id"], base_trade_id)
                    self.assertEqual(payload_premium["trade_id"], premium_trade_id)
                    self.assertEqual(payload_premium["diagnostics"]["engine"], "python_price_only")
                    self.assertEqual(payload_premium["diagnostics"]["pricing_mode"], "python_swaption_static")
                    self.assertEqual(payload_premium["pricing"]["trade_type"], "Swaption")
                    self.assertEqual(payload_premium["pricing"]["long_short"], "Long")
                    self.assertGreater(float(payload_premium["pricing"]["premium_pv"]), 0.0)
                    delta_py = float(payload_base["pricing"]["py_t0_npv"]) - float(payload_premium["pricing"]["py_t0_npv"])
                    delta_ore = float(payload_base["pricing"]["ore_t0_npv"]) - float(payload_premium["pricing"]["ore_t0_npv"])
                    self.assertLess(abs(delta_py - delta_ore), 1.0)

    def test_price_only_swaption_short_sign_flips_against_long_case(self):
        for trade_id in ("SwaptionCashPremium", "SwaptionPhysicalPremium"):
            with self.subTest(trade_id=trade_id), tempfile.TemporaryDirectory() as tmp_long, tempfile.TemporaryDirectory() as tmp_short:
                root_long = Path(tmp_long)
                root_short = Path(tmp_short)
                ore_xml_long = _clone_example3_case(root_long)
                ore_xml_short = _clone_example3_case(root_short)
                _promote_trade_to_first(ore_xml_long.parent / "portfolio.xml", trade_id)
                _promote_trade_to_first(ore_xml_short.parent / "portfolio.xml", trade_id, long_short="Short")
                rc_long = ore_snapshot_cli.main(
                    [
                        str(ore_xml_long),
                        "--price",
                        "--output-root",
                        str(root_long / "artifacts"),
                    ]
                )
                rc_short = ore_snapshot_cli.main(
                    [
                        str(ore_xml_short),
                        "--price",
                        "--output-root",
                        str(root_short / "artifacts"),
                    ]
                )
                self.assertIn(rc_long, (0, 1))
                self.assertIn(rc_short, (0, 1))
                payload_long = json.loads((root_long / "artifacts" / "Example_3" / "summary.json").read_text(encoding="utf-8"))
                payload_short = json.loads((root_short / "artifacts" / "Example_3" / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(payload_short["pricing"]["long_short"], "Short")
                self.assertAlmostEqual(
                    float(payload_short["pricing"]["py_t0_npv"]) + float(payload_long["pricing"]["py_t0_npv"]),
                    0.0,
                    delta=1.0e-3,
                )

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

    def test_swap_portfolio_xva_runs_across_multiple_netting_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(PORTFOLIO_SWAP_CASE_XML),
                    "--xva",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_6" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python-lgm-portfolio")
            self.assertEqual(payload["diagnostics"]["portfolio_mode"], True)
            self.assertGreater(payload["diagnostics"]["portfolio_netting_set_count"], 1)
            self.assertGreater(payload["diagnostics"]["portfolio_counterparty_count"], 1)
            self.assertEqual(payload["trade_id"], "swap_01")
            self.assertIn("py_cva", payload["xva"])

    def test_capfloor_price_only_surfaces_large_parity_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(CAPFLOOR_CASE_XML),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertEqual(rc, 1)
            payload = json.loads((root / "artifacts" / "Example_6" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
            self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_capfloor_lgm")
            self.assertFalse(payload["pass_all"])
            self.assertGreater(float(payload["pricing"]["t0_npv_abs_diff"]), 1000.0)

    def test_capfloor_price_only_ignores_reference_artifacts_outside_parity_mode(self):
        with patch(
            "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter",
            side_effect=AssertionError("curves.csv should not be used"),
        ), patch(
            "py_ore_tools.ore_snapshot_cli._build_capfloor_defs_from_flows",
            side_effect=AssertionError("flows.csv should not be used"),
        ):
            payload = ore_snapshot_cli._compute_price_only_case(
                CAPFLOOR_CASE_XML,
                anchor_t0_npv=False,
                use_reference_artifacts=False,
            )
        self.assertEqual(payload["trade_id"], "cap_01")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_capfloor_lgm")
        self.assertIn("py_t0_npv", payload["pricing"])

    def test_capfloor_price_only_python_mode_disables_reference_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "py_ore_tools.ore_snapshot_cli._compute_price_only_case",
                wraps=ore_snapshot_cli._compute_price_only_case,
            ) as compute_mock:
                rc = ore_snapshot_cli.main(
                    [
                        str(CAPFLOOR_CASE_XML),
                        "--price",
                        "--engine",
                        "python",
                        "--output-root",
                        str(root / "artifacts"),
                    ]
                )
            self.assertIn(rc, (0, 1))
            self.assertTrue(compute_mock.called)
            self.assertFalse(compute_mock.call_args.kwargs["use_reference_artifacts"])

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

    def test_run_case_from_buffers_compare_path_handles_day_based_swap_schedule(self):
        input_files = {
            path.name: path.read_text(encoding="utf-8")
            for path in (TOOLS_DIR / "Examples" / "Legacy" / "Example_1" / "Input").iterdir()
            if path.is_file()
        }
        input_files["portfolio_swap.xml"] = _patch_swap_floating_tenor(input_files["portfolio_swap.xml"], "365D")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_root = root / "case"
            ore_xml = ore_snapshot_cli._materialize_buffer_case(
                ore_snapshot_cli.BufferCaseInputs(input_files=input_files),
                case_root,
            )
            ore_root = ET.parse(ore_xml).getroot()
            asof = ore_snapshot_cli._parse_ore_date(
                ore_root.findtext("./Setup/Parameter[@name='asofDate']") or ""
            )
            p0_disc, p0_fwd, ql_disc_handle, ql_fwd_handle = ore_snapshot_cli._build_fitted_discount_and_forward_curves(
                ore_xml,
                asof=asof,
                currency="EUR",
                float_index="EUR-EURIBOR-6M",
            )
            payload = {
                "trade_id": "Swap_20y",
                "ql_disc_handle": ql_disc_handle,
                "ql_fwd_handle": ql_fwd_handle,
            }
            py_npv = ore_snapshot_cli._price_plain_vanilla_swap_with_quantlib(ore_xml, payload)
            self.assertIsNotNone(py_npv)

            artifact_root = root / "artifacts"
            case_out_dir = artifact_root / ore_snapshot_cli._case_slug(ore_xml)
            case_out_dir.mkdir(parents=True, exist_ok=True)
            ore_snapshot_cli._write_ore_compatible_reports(
                case_out_dir,
                {
                    "ore_xml": str(ore_xml),
                    "trade_id": "Swap_20y",
                    "counterparty": "CPTY_A",
                    "netting_set_id": "CPTY_A",
                    "maturity_date": "20360301",
                    "maturity_time": 20.0,
                    "pricing": {
                        "py_t0_npv": float(py_npv),
                        "trade_type": "Swap",
                        "report_ccy": "EUR",
                    },
                },
            )
            output_files = {
                path.name: path.read_text(encoding="utf-8")
                for path in case_out_dir.iterdir()
                if path.is_file()
            }

            compare_result = ore_snapshot_cli.run_case_from_buffers(
                ore_snapshot_cli.BufferCaseInputs(input_files=input_files, output_files=output_files),
                ore_snapshot_cli.PurePythonRunOptions(engine="compare", price=True, xva=False, paths=16),
            )

        self.assertTrue(compare_result.comparison_rows)
        self.assertTrue(compare_result.summary["pass_all"])
        self.assertIn("pricing", compare_result.summary)
        self.assertLessEqual(float(compare_result.summary["pricing"]["t0_npv_abs_diff"]), 1.0e-6)

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

    def test_report_writer_prefers_exposure_profile_payload_over_legacy_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            summary = {
                "trade_id": "T1",
                "counterparty": "CPTY_A",
                "netting_set_id": "CPTY_A",
                "maturity_date": "2021-03-01",
                "maturity_time": 5.0,
                "pricing": {"py_t0_npv": 10.0, "trade_type": "Swap"},
                "xva": {"py_cva": 1.0, "py_dva": 2.0, "py_fba": 3.0, "py_fca": 4.0, "py_basel_epe": 999.0, "py_basel_eepe": 999.0},
                "exposure_dates": ["2020-01-01", "2021-01-01"],
                "exposure_times": [0.0, 1.0],
                "py_epe": [999.0, 999.0],
                "py_ene": [999.0, 999.0],
                "py_pfe": [999.0, 999.0],
                "exposure_profile_by_trade": {
                    "dates": ["2020-01-01", "2021-01-01"],
                    "times": [0.0, 1.0],
                    "closeout_times": [0.0, 1.0],
                    "closeout_epe": [10.0, 20.0],
                    "closeout_ene": [1.0, 2.0],
                    "pfe": [30.0, 40.0],
                    "basel_ee": [10.0, 20.0],
                    "basel_eee": [10.0, 20.0],
                    "time_weighted_basel_epe": [10.0, 15.0],
                    "time_weighted_basel_eepe": [10.0, 15.0],
                },
                "exposure_profile_by_netting_set": {
                    "dates": ["2020-01-01", "2021-01-01"],
                    "times": [0.0, 1.0],
                    "closeout_times": [0.0, 1.0],
                    "closeout_epe": [100.0, 200.0],
                    "closeout_ene": [11.0, 22.0],
                    "pfe": [300.0, 400.0],
                    "expected_collateral": [100.0, 0.0],
                    "basel_ee": [100.0, 200.0],
                    "basel_eee": [100.0, 200.0],
                    "time_weighted_basel_epe": [100.0, 150.0],
                    "time_weighted_basel_eepe": [100.0, 150.0],
                },
            }
            ore_snapshot_cli._write_ore_compatible_reports(out_dir, summary)
            with open(out_dir / "exposure_trade_T1.csv", newline="", encoding="utf-8") as f:
                trade_rows = list(csv.DictReader(f))
            with open(out_dir / "exposure_nettingset_CPTY_A.csv", newline="", encoding="utf-8") as f:
                netting_rows = list(csv.DictReader(f))
            with open(out_dir / "xva.csv", newline="", encoding="utf-8") as f:
                xva_rows = list(csv.DictReader(f))
            self.assertEqual(trade_rows[1]["PFE"], "40")
            self.assertEqual(trade_rows[1]["TimeWeightedBaselEEPE"], "15.00")
            self.assertEqual(netting_rows[1]["PFE"], "400.00")
            self.assertEqual(netting_rows[0]["ExpectedCollateral"], "100.00")
            self.assertEqual(xva_rows[0]["BaselEPE"], "150.00")
            self.assertEqual(xva_rows[0]["BaselEEPE"], "150.00")

    def test_profile_builder_uses_ore_quantile_and_discounted_basel_fields(self):
        profile = ore_snapshot_cli._build_ore_style_exposure_profile(
            "T1",
            ["2020-01-03", "2021-01-04", "2021-01-08"],
            [0.0, 1.005, 1.02],
            [100.0, 100.0, 100.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            discount_factors=[1.0, 0.9, 0.8],
        )
        self.assertAlmostEqual(profile["basel_ee"][1], 100.0 / 0.9, places=12)
        self.assertAlmostEqual(profile["basel_ee"][2], 100.0 / 0.8, places=12)
        self.assertAlmostEqual(
            ore_snapshot_cli._one_year_profile_value(profile, "time_weighted_basel_epe"),
            profile["time_weighted_basel_epe"][1],
            places=12,
        )

    def test_ore_pfe_quantile_matches_ore_rounding_and_floors_after_selection(self):
        samples = np.asarray([[-10.0, -5.0, -1.0, 2.0, 100.0]], dtype=float)
        self.assertEqual(float(ore_pfe_quantile(samples, 0.80)[0]), 2.0)
        self.assertEqual(float(ore_pfe_quantile(np.asarray([[-10.0, -5.0, -1.0]], dtype=float), 0.95)[0]), 0.0)

    def test_benchmark_pfe_profile_vs_ore_aggregates_seed_statistics(self):
        ore_profile = {
            "date": ["2020-01-01", "2021-01-02"],
            "time": [0.0, 1.0],
            "epe": [10.0, 20.0],
            "pfe": [12.0, 22.0],
        }

        def fake_compute(*args, **kwargs):
            seed = int(kwargs["seed"])
            pfe = [12.0 + seed, 22.0 + seed]
            return ore_snapshot_cli.SnapshotComputation(
                ore_xml=str(REAL_CASE_XML),
                trade_id="T1",
                counterparty="CPTY_A",
                netting_set_id="NS1",
                paths=2000,
                seed=seed,
                rng_mode="ore_parity",
                pricing={},
                xva={},
                parity={},
                diagnostics={},
                maturity_date="2025-01-01",
                maturity_time=5.0,
                exposure_dates=["2020-01-01", "2021-01-02"],
                exposure_times=[0.0, 1.0],
                py_epe=[10.0, 20.0],
                py_ene=[0.0, 0.0],
                py_pfe=pfe,
                exposure_profile_by_trade={
                    "dates": ["2020-01-01", "2021-01-02"],
                    "times": [0.0, 1.0],
                    "pfe": pfe,
                },
                exposure_profile_by_netting_set={},
                ore_basel_epe=0.0,
                ore_basel_eepe=0.0,
            )

        with patch("py_ore_tools.ore_snapshot_cli.load_from_ore_xml", return_value=SimpleNamespace(trade_id="T1", netting_set_id="NS1", n_samples=2000)):
            with patch("py_ore_tools.ore_snapshot_cli._find_reference_output_file", return_value=Path("/tmp/exposure_trade_T1.csv")):
                with patch("py_ore_tools.ore_snapshot_cli.load_ore_exposure_profile", return_value=ore_profile):
                    with patch("py_ore_tools.ore_snapshot_cli._compute_snapshot_case", side_effect=fake_compute):
                        result = ore_snapshot_cli.benchmark_pfe_profile_vs_ore(REAL_CASE_XML, paths=2000, seeds=[1, 3])
        self.assertEqual(result["summary"]["grid_points"], 2)
        self.assertEqual(len(result["runs"]), 2)
        self.assertIn("SigmaMultiple", result["pointwise"][0])
        self.assertGreaterEqual(result["summary"]["within_two_sigma_ratio"], 0.0)

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
            self.assertIn(
                payload["diagnostics"]["pricing_mode"],
                {
                    "python_swap_quantlib",
                    "python_swap_lgm",
                    "python_swap_quantlib_fixed_a365",
                    "python_swap_quantlib_fixed_a365_unadjusted",
                },
            )

    def test_quantlib_plain_swap_pricer_builds_supported_vanilla_swap(self):
        payload = ore_snapshot_cli._build_minimal_pricing_payload(REAL_CASE_XML, anchor_t0_npv=False)
        npv_local = ore_snapshot_cli._price_plain_vanilla_swap_with_quantlib(REAL_CASE_XML, payload)
        self.assertIsNotNone(npv_local)
        self.assertTrue(np.isfinite(float(npv_local)))

    def test_price_only_swap_example2_uses_closer_fixed_daycount_candidate(self):
        payload = ore_snapshot_cli._compute_price_only_case(
            TOOLS_DIR / "Examples" / "Legacy" / "Example_2" / "Input" / "ore.xml",
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "Swap_20")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_swap_quantlib_fixed_a365_unadjusted")
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 80.0)

    def test_price_only_swap_example9_stays_on_lgm_when_quantlib_conventions_are_worse(self):
        payload = ore_snapshot_cli._compute_price_only_case(
            TOOLS_DIR / "Examples" / "ORE-Python" / "Notebooks" / "Example_9" / "Input" / "ore.xml",
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "Swap_20y")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_swap_lgm")
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 20.0)

    def test_price_only_swap_example12_stays_on_lgm_when_quantlib_conventions_are_worse(self):
        payload = ore_snapshot_cli._compute_price_only_case(
            TOOLS_DIR / "Examples" / "Legacy" / "Example_12" / "Input" / "ore2.xml",
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "Swap_50y_2")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_swap_lgm")
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 5.0)

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
            with patch.dict(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod._RUNTIME_LGM_CALIBRATION_CACHE",
                {},
                clear=True,
            ), patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.resolve_calibration_xml_path", return_value=None
            ), patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.calibrate_lgm_params_in_python",
                return_value=calibrated,
            ) as python_calibrate_mock, patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.calibrate_lgm_params_via_ore",
                side_effect=AssertionError("ORE calibration should not be used when Python calibration succeeds"),
            ) as ore_calibrate_mock, patch(
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
            self.assertEqual(python_calibrate_mock.call_count, 1)
            self.assertEqual(ore_calibrate_mock.call_count, 0)
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
            with patch.dict(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod._RUNTIME_LGM_CALIBRATION_CACHE",
                {},
                clear=True,
            ), patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.resolve_calibration_xml_path", return_value=None
            ), patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.calibrate_lgm_params_in_python",
                return_value=None,
            ) as python_calibrate_mock, patch(
                "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod.calibrate_lgm_params_via_ore",
                return_value=None,
            ) as ore_calibrate_mock, patch(
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
            self.assertEqual(python_calibrate_mock.call_count, 1)
            self.assertEqual(ore_calibrate_mock.call_count, 1)
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
                    "--engine",
                    "python",
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
            text = text.replace(str(REAL_CASE_XML.parent), str(input_dir))
            text = text.replace(str(REAL_CASE_XML.parents[1] / "Output"), str(output_dir))
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

    def test_price_only_swap_without_simulation_analytic_rejects_cross_currency_discount_curve_without_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            case_root = tmp_root / "Example_31"
            shutil.copytree(TOOLS_DIR / "Examples" / "Legacy" / "Example_31", case_root)
            shared_input_dir = tmp_root.parent / "Input"
            if shared_input_dir.exists():
                shutil.rmtree(shared_input_dir)
            shutil.copytree(TOOLS_DIR / "Examples" / "Input", shared_input_dir)
            input_dir = case_root / "Input"
            ore_xml_path = input_dir / "ore.xml"
            text = ore_xml_path.read_text(encoding="utf-8")
            start = text.index('<Analytic type="simulation">')
            end = text.index("</Analytic>", start) + len("</Analytic>")
            ore_xml_path.write_text(text[:start] + text[end:], encoding="utf-8")
            sim_xml = input_dir / "simulation.xml"
            if sim_xml.exists():
                sim_xml.unlink()
            with self.assertRaisesRegex(ValueError, "cross-currency segments"):
                ore_snapshot_cli._build_minimal_pricing_payload(ore_xml_path, anchor_t0_npv=False)

    def test_price_only_swap_with_cross_currency_discount_curve_falls_back_to_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(TOOLS_DIR / "Examples" / "Legacy" / "Example_31" / "Input" / "ore.xml"),
                    "--price",
                    "--output-root",
                    str(tmp_root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((tmp_root / "artifacts" / "Example_31" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["trade_id"], "Swap_1")
            self.assertEqual(payload["diagnostics"]["engine"], "ore_reference_expected_output")
            self.assertEqual(payload["diagnostics"]["fallback_reason"], "unsupported_python_snapshot")
            self.assertIn("cross-currency segments", payload["diagnostics"]["fallback_error"])

    def test_price_only_swap_replays_multicurrency_reference_flows_when_leg_reconstruction_is_inconsistent(self):
        payload = ore_snapshot_cli._compute_price_only_case(
            TOOLS_DIR / "Examples" / "Legacy" / "Example_13" / "Input" / "ore_A0.xml",
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "Swap_2")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_swap_cashflow_replay")
        self.assertTrue(payload["diagnostics"]["cashflow_replay"])
        self.assertEqual(payload["pricing"]["cashflow_currencies"], ["EUR", "USD"])
        self.assertAlmostEqual(payload["pricing"]["ore_t0_npv"], -19221.933964, places=6)
        self.assertAlmostEqual(payload["pricing"]["py_t0_npv"], -19221.93396382034, places=6)
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 1.0e-4)

    def test_price_only_swap_ignores_reference_flows_and_curves_outside_parity_mode(self):
        with patch(
            "py_ore_tools.ore_snapshot_cli.ore_snapshot_mod._load_ore_discount_pairs_by_columns_with_day_counter",
            side_effect=AssertionError("curves.csv should not be used"),
        ), patch(
            "py_ore_tools.ore_snapshot_cli.load_trade_cashflows_from_flows",
            side_effect=AssertionError("flows.csv trade replay should not be used"),
        ), patch(
            "py_ore_tools.ore_snapshot_cli.load_ore_legs_from_flows",
            side_effect=AssertionError("flows.csv legs should not be used"),
        ):
            payload = ore_snapshot_cli._compute_price_only_case(
                TOOLS_DIR / "Examples" / "Legacy" / "Example_13" / "Input" / "ore_A0.xml",
                anchor_t0_npv=False,
                use_reference_artifacts=False,
            )
        self.assertEqual(payload["trade_id"], "Swap_2")
        self.assertNotEqual(payload["diagnostics"]["pricing_mode"], "python_swap_cashflow_replay")
        self.assertFalse(payload["diagnostics"].get("cashflow_replay", False))

    def test_price_only_swap_compare_mode_explicitly_enables_reference_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "py_ore_tools.ore_snapshot_cli._compute_price_only_case",
                wraps=ore_snapshot_cli._compute_price_only_case,
            ) as compute_mock:
                rc = ore_snapshot_cli.main(
                    [
                        str(TOOLS_DIR / "Examples" / "Legacy" / "Example_13" / "Input" / "ore_A0.xml"),
                        "--price",
                        "--engine",
                        "compare",
                        "--output-root",
                        str(root / "artifacts"),
                    ]
                )
            self.assertIn(rc, (0, 1))
            self.assertTrue(compute_mock.called)
            self.assertTrue(compute_mock.call_args.kwargs["use_reference_artifacts"])

    def test_price_only_swap_python_mode_disables_reference_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "py_ore_tools.ore_snapshot_cli._compute_price_only_case",
                wraps=ore_snapshot_cli._compute_price_only_case,
            ) as compute_mock:
                rc = ore_snapshot_cli.main(
                    [
                        str(TOOLS_DIR / "Examples" / "Legacy" / "Example_13" / "Input" / "ore_A0.xml"),
                        "--price",
                        "--engine",
                        "python",
                        "--output-root",
                        str(root / "artifacts"),
                    ]
                )
            self.assertIn(rc, (0, 1))
            self.assertTrue(compute_mock.called)
            self.assertFalse(compute_mock.call_args.kwargs["use_reference_artifacts"])

    def test_price_only_swap_missing_trade_reference_row_falls_back_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ore_snapshot_cli.main(
                [
                    str(TOOLS_DIR / "Examples" / "Legacy" / "Example_12" / "Input" / "ore.xml"),
                    "--price",
                    "--output-root",
                    str(root / "artifacts"),
                ]
            )
            self.assertIn(rc, (0, 1))
            payload = json.loads((root / "artifacts" / "Example_12" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["trade_id"], "Swap_50y")
            self.assertEqual(payload["diagnostics"]["engine"], "ore_reference_expected_output")
            self.assertEqual(payload["diagnostics"]["fallback_reason"], "unsupported_python_snapshot")
            self.assertIn("Trade 'Swap_50y' not found", payload["diagnostics"]["fallback_error"])

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
        with patch.dict(
            "pythonore.io.ore_snapshot._RUNTIME_LGM_CALIBRATION_CACHE",
            {},
            clear=True,
        ), patch("pythonore.io.ore_snapshot.resolve_calibration_xml_path", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_in_python",
            return_value=calibrated,
        ) as python_calibrate_mock, patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore",
            side_effect=AssertionError("ORE calibration should not be used when Python calibration succeeds"),
        ) as ore_calibrate_mock, patch(
            "pythonore.io.ore_snapshot.parse_lgm_params_from_simulation_xml",
            side_effect=AssertionError("simulation fallback should not be used"),
        ):
            snap = ore_snapshot_io.load_from_ore_xml(REAL_CASE_XML)
        self.assertEqual(python_calibrate_mock.call_count, 1)
        self.assertEqual(ore_calibrate_mock.call_count, 0)
        self.assertEqual(snap.alpha_source, "calibration")

    def test_load_from_ore_xml_can_force_simulation_xml_params(self):
        simulated = {
            "alpha_times": (1.0,),
            "alpha_values": (0.015, 0.015),
            "kappa_times": (1.0,),
            "kappa_values": (0.025, 0.025),
            "shift": 0.0,
            "scaling": 1.0,
        }
        with patch("pythonore.io.ore_snapshot.resolve_calibration_xml_path", return_value=None), patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_in_python",
            side_effect=AssertionError("Python calibration should not be used"),
        ) as python_calibrate_mock, patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore",
            side_effect=AssertionError("ORE calibration should not be used"),
        ) as calibrate_mock, patch(
            "pythonore.io.ore_snapshot.parse_lgm_params_from_simulation_xml",
            return_value=simulated,
        ) as simulation_mock:
            snap = ore_snapshot_io.load_from_ore_xml(REAL_CASE_XML, lgm_param_source="simulation_xml")
        self.assertEqual(python_calibrate_mock.call_count, 0)
        self.assertEqual(calibrate_mock.call_count, 0)
        self.assertEqual(simulation_mock.call_count, 1)
        self.assertEqual(snap.alpha_source, "simulation")
        self.assertAlmostEqual(snap.lgm_params.alpha_values[0], 0.015, places=12)

    def test_load_from_ore_xml_can_use_provided_lgm_params(self):
        provided = ore_snapshot_io.LGMParams(
            alpha_times=(1.0,),
            alpha_values=(0.014, 0.014),
            kappa_times=(1.0,),
            kappa_values=(0.031, 0.031),
            shift=0.0,
            scaling=1.0,
        )
        with patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_in_python",
            side_effect=AssertionError("Python calibration should not be used"),
        ) as python_calibrate_mock, patch(
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore",
            side_effect=AssertionError("ORE calibration should not be used"),
        ) as calibrate_mock, patch(
            "pythonore.io.ore_snapshot.parse_lgm_params_from_simulation_xml",
            side_effect=AssertionError("simulation params should not be used"),
        ) as simulation_mock:
            snap = ore_snapshot_io.load_from_ore_xml(
                REAL_CASE_XML,
                lgm_param_source="provided",
                provided_lgm_params=provided,
            )
        self.assertEqual(python_calibrate_mock.call_count, 0)
        self.assertEqual(calibrate_mock.call_count, 0)
        self.assertEqual(simulation_mock.call_count, 0)
        self.assertEqual(snap.alpha_source, "provided")
        self.assertEqual(tuple(snap.lgm_params.alpha_values), (0.014, 0.014))

    def test_load_ore_xva_reference_row_prefers_trade_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            xva_csv = Path(tmp) / "xva.csv"
            xva_csv.write_text(
                "\n".join(
                    [
                        "#TradeId,NettingSetId,CVA,DVA,FBA,FCA,BaselEPE,BaselEEPE",
                        ",CPTY_A,100.0,200.0,300.0,400.0,500.0,600.0",
                        "Swap_EUR,CPTY_A,10.0,20.0,30.0,40.0,50.0,60.0",
                    ]
                ),
                encoding="utf-8",
            )
            row, used_trade_row = ore_snapshot_io._load_ore_xva_reference_row(
                xva_csv, trade_id="Swap_EUR", netting_set_id="CPTY_A"
            )
        self.assertTrue(used_trade_row)
        self.assertEqual(row["cva"], 10.0)
        self.assertEqual(row["dva"], 20.0)
        self.assertEqual(row["fba"], 30.0)
        self.assertEqual(row["fca"], 40.0)
        self.assertEqual(row["basel_epe"], 50.0)
        self.assertEqual(row["basel_eepe"], 60.0)

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
            text = text.replace(str(REAL_CASE_XML.parent), str(input_dir))
            text = text.replace(str(REAL_CASE_XML.parents[1] / "Output"), str(output_dir))
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

    def test_load_from_ore_xml_supports_price_only_case_without_xva_outputs(self):
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
            root = ET.parse(ore_xml_path).getroot()
            for node in root.findall("./Setup/Parameter"):
                name = node.attrib.get("name", "")
                if name == "inputPath":
                    node.text = str(input_dir)
                elif name == "outputPath":
                    node.text = str(output_dir)
                elif name == "portfolioFile":
                    node.text = str(input_dir / "portfolio.xml")
            analytics = root.find("./Analytics")
            if analytics is not None:
                for analytic in list(analytics.findall("./Analytic")):
                    if analytic.attrib.get("type", "") == "simulation":
                        analytics.remove(analytic)
            ET.ElementTree(root).write(ore_xml_path, encoding="utf-8")

            snap = ore_snapshot_io.load_from_ore_xml(ore_xml_path)

            self.assertTrue(np.isfinite(snap.ore_t0_npv))
            self.assertEqual(snap.ore_cva, 0.0)
            self.assertEqual(snap.ore_dva, 0.0)
            self.assertEqual(snap.ore_fba, 0.0)
            self.assertEqual(snap.ore_fca, 0.0)
            self.assertEqual(snap.exposure_times.size, 0)
            self.assertEqual(snap.ore_epe.size, 0)
            self.assertEqual(snap.ore_ene.size, 0)
            self.assertTrue(str(snap.simulation_xml_path).endswith("simulation.xml"))
            self.assertIsNone(snap.exposure_csv_path)
            self.assertIsNone(snap.xva_csv_path)

    def test_load_from_ore_xml_supports_generic_fx_forward_price_only_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ore_xml = _clone_example28_eur_base_case(Path(tmpdir))

            snap = ore_snapshot_io.load_from_ore_xml(ore_xml, trade_id="FXFWD_1Y")

            self.assertEqual(snap.trade_id, "FXFWD_1Y")
            self.assertEqual(snap.trade_type, "FxForward")
            self.assertTrue(np.isfinite(snap.ore_t0_npv))
            self.assertEqual(snap.ore_cva, 0.0)
            self.assertEqual(snap.ore_dva, 0.0)
            self.assertEqual(snap.ore_fba, 0.0)
            self.assertEqual(snap.ore_fca, 0.0)
            self.assertEqual(snap.exposure_times.size, 0)
            self.assertEqual(snap.ore_epe.size, 0)
            self.assertEqual(snap.ore_ene.size, 0)
            self.assertTrue(str(snap.npv_csv_path).endswith("npv_eur_base.csv"))
            self.assertTrue(str(snap.curves_csv_path).endswith("curves_eur_base.csv"))
            self.assertTrue(str(snap.flows_csv_path).endswith("flows_eur_base.csv"))
            self.assertIsNone(snap.xva_csv_path)
            self.assertIsNone(snap.exposure_csv_path)

    def test_load_todaysmarket_calibration_csv_parses_multi_curve_rows(self):
        calibration_csv = (
            TOOLS_DIR
            / "Examples"
            / "Legacy"
            / "Example_63"
            / "Output"
            / "valid_xccy"
            / "todaysmarketcalibration.csv"
        )
        simulation_xml = TOOLS_DIR / "Examples" / "Legacy" / "Example_63" / "Input" / "simulation.xml"

        rows = ore_snapshot_io.load_todaysmarket_calibration_csv(calibration_csv)
        grouped = ore_snapshot_io.group_todaysmarket_calibration_rows(rows)
        yield_curve_ids = {
            market_object_id
            for (market_object_type, market_object_id) in grouped
            if market_object_type == "yieldCurve"
        }
        sim_root = ET.parse(simulation_xml).getroot()
        sim_indices = {
            (node.text or "").strip()
            for node in sim_root.findall("./Market/Indices/Index")
            if (node.text or "").strip()
        }

        self.assertGreater(len(rows), 0)
        self.assertTrue(sim_indices.issubset(yield_curve_ids))
        self.assertIn("USD-IN-EUR", yield_curve_ids)
        self.assertNotIn("USD-IN-EUR", sim_indices)
        usd_sofr_rows = grouped[("yieldCurve", "USD-SOFR")]
        self.assertIn("zeroRate", {row.result_id for row in usd_sofr_rows})
        self.assertIn("discountFactor", {row.result_id for row in usd_sofr_rows})
        self.assertGreater(
            sum(1 for row in usd_sofr_rows if row.result_id == "discountFactor" and row.as_float() > 0.0),
            0,
        )

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
            exposure_profile_by_trade={},
            exposure_profile_by_netting_set={},
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

    def test_run_case_uses_portfolio_xva_branch_for_multi_trade_portfolios(self):
        args = ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML), "--xva"])
        fake_summary = ore_snapshot_cli.SnapshotComputation(
            ore_xml=str(REAL_CASE_XML),
            trade_id="PORTFOLIO",
            counterparty="CPTY_A",
            netting_set_id="NS_A",
            paths=500,
            seed=42,
            rng_mode="ore_parity",
            pricing={
                "trade_type": "Portfolio",
                "py_t0_npv": 12.0,
                "ore_t0_npv": 12.0,
                "t0_npv_abs_diff": 0.0,
                "report_ccy": "EUR",
                "currency": "EUR",
                "leg_source": "portfolio",
            },
            xva={
                "ore_cva": 4.0,
                "py_cva": 4.0,
                "cva_rel_diff": 0.0,
                "ore_dva": 1.0,
                "py_dva": 1.0,
                "dva_rel_diff": 0.0,
                "ore_fba": 0.0,
                "py_fba": 0.0,
                "fba_rel_diff": 0.0,
                "ore_fca": 0.0,
                "py_fca": 0.0,
                "fca_rel_diff": 0.0,
                "py_fva": 0.0,
                "own_credit_source": "portfolio_runtime",
                "ore_basel_epe": 0.0,
                "ore_basel_eepe": 0.0,
                "py_basel_epe": 0.0,
                "py_basel_eepe": 0.0,
            },
            parity={"parity_ready": True, "summary": {"requested_xva_metrics": ["CVA", "DVA"], "portfolio_mode": True}},
            diagnostics={"engine": "python-lgm-portfolio"},
            maturity_date="",
            maturity_time=0.0,
            exposure_dates=["2025-01-01"],
            exposure_times=[0.0],
            py_epe=[0.0],
            py_ene=[0.0],
            py_pfe=[0.0],
            exposure_profile_by_trade={"dates": ["2025-01-01"], "times": [0.0], "closeout_epe": [0.0], "closeout_ene": [0.0], "pfe": [0.0]},
            exposure_profile_by_netting_set={"dates": ["2025-01-01"], "times": [0.0], "closeout_epe": [0.0], "closeout_ene": [0.0], "pfe": [0.0]},
            ore_basel_epe=0.0,
            ore_basel_eepe=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}):
                with patch("py_ore_tools.ore_snapshot_cli._portfolio_contains_swap_like_trade", return_value=True):
                    with patch("py_ore_tools.ore_snapshot_cli._portfolio_trade_context", return_value=(ET.Element("Portfolio"), ["T1", "T2"], "CPTY_A", "NS_A")):
                        with patch("py_ore_tools.ore_snapshot_cli._compute_portfolio_xva_case", return_value=fake_summary) as portfolio_case:
                            with patch("py_ore_tools.ore_snapshot_cli._compute_snapshot_case", side_effect=AssertionError("single-trade path should not be used")):
                                with patch("py_ore_tools.ore_snapshot_cli._write_ore_compatible_reports"):
                                    with patch("py_ore_tools.ore_snapshot_cli._copy_native_ore_reports"):
                                        result = ore_snapshot_cli._run_case(REAL_CASE_XML, args, artifact_root=Path(tmp))
        portfolio_case.assert_called_once()
        self.assertEqual(result["trade_id"], "PORTFOLIO")
        self.assertIsNone(result.get("pricing"))
        self.assertEqual(result["xva"]["ore_cva"], 4.0)

    def test_run_case_uses_portfolio_price_branch_for_multi_trade_portfolios(self):
        args = ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML), "--price"])
        fake_summary = {
            "trade_id": "PORTFOLIO",
            "counterparty": "CPTY_A",
            "netting_set_id": "NS_A",
            "maturity_date": "2026-01-01",
            "maturity_time": 1.0,
            "pricing": {
                "trade_type": "Portfolio",
                "py_t0_npv": 12.0,
                "ore_t0_npv": 12.0,
                "t0_npv_abs_diff": 0.0,
                "report_ccy": "EUR",
                "currency": "EUR",
                "leg_source": "portfolio",
            },
            "diagnostics": {"engine": "python_price_only", "pricing_mode": "python_portfolio_price_only"},
            "portfolio_trade_rows": [
                {
                    "trade_id": "T1",
                    "trade_type": "Swap",
                    "counterparty": "CPTY_A",
                    "netting_set_id": "NS_A",
                    "maturity_date": "2025-01-01",
                    "maturity_time": 0.5,
                    "pricing": {"py_t0_npv": 5.0, "ore_t0_npv": 5.0, "report_ccy": "EUR"},
                    "diagnostics": {"engine": "python_price_only"},
                    "notional": 100.0,
                    "notional_currency": "EUR",
                },
                {
                    "trade_id": "T2",
                    "trade_type": "Swap",
                    "counterparty": "CPTY_A",
                    "netting_set_id": "NS_A",
                    "maturity_date": "2026-01-01",
                    "maturity_time": 1.0,
                    "pricing": {"py_t0_npv": 7.0, "ore_t0_npv": 7.0, "report_ccy": "EUR"},
                    "diagnostics": {"engine": "python_price_only"},
                    "notional": 120.0,
                    "notional_currency": "EUR",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}):
                with patch(
                    "py_ore_tools.ore_snapshot_cli._portfolio_trade_summary",
                    return_value={"trade_count": 2, "trade_rows": [{"trade_id": "T1"}, {"trade_id": "T2"}], "trade_ids_by_netting_set": {"NS_A": ["T1", "T2"]}},
                ):
                    with patch("py_ore_tools.ore_snapshot_cli._compute_portfolio_price_case", return_value=fake_summary) as portfolio_case:
                        with patch("py_ore_tools.ore_snapshot_cli._compute_price_only_case", side_effect=AssertionError("single-trade path should not be used")):
                            with patch("py_ore_tools.ore_snapshot_cli._price_reference_summary", side_effect=AssertionError("reference fallback should not be used")):
                                with patch("py_ore_tools.ore_snapshot_cli._write_ore_compatible_reports"):
                                    with patch("py_ore_tools.ore_snapshot_cli._copy_native_ore_reports"):
                                        result = ore_snapshot_cli._run_case(REAL_CASE_XML, args, artifact_root=Path(tmp))
        portfolio_case.assert_called_once()
        self.assertEqual(result["trade_id"], "PORTFOLIO")
        self.assertEqual(result["pricing"]["py_t0_npv"], 12.0)
        self.assertEqual(result["pricing"]["ore_t0_npv"], 12.0)
        self.assertEqual(result["diagnostics"]["pricing_mode"], "python_portfolio_price_only")

    def test_run_case_uses_portfolio_xva_branch_when_swap_is_not_anchor_trade(self):
        args = ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML), "--xva"])
        fake_summary = ore_snapshot_cli.SnapshotComputation(
            ore_xml=str(REAL_CASE_XML),
            trade_id="PORTFOLIO",
            counterparty="CPTY_A",
            netting_set_id="NS_A",
            paths=500,
            seed=42,
            rng_mode="ore_parity",
            pricing={
                "trade_type": "Portfolio",
                "py_t0_npv": 12.0,
                "ore_t0_npv": 12.0,
                "t0_npv_abs_diff": 0.0,
                "report_ccy": "EUR",
                "currency": "EUR",
                "leg_source": "portfolio",
            },
            xva={
                "ore_cva": 4.0,
                "py_cva": 4.0,
                "cva_rel_diff": 0.0,
                "ore_dva": 1.0,
                "py_dva": 1.0,
                "dva_rel_diff": 0.0,
                "ore_fba": 0.0,
                "py_fba": 0.0,
                "fba_rel_diff": 0.0,
                "ore_fca": 0.0,
                "py_fca": 0.0,
                "fca_rel_diff": 0.0,
                "py_fva": 0.0,
                "own_credit_source": "portfolio_runtime",
                "ore_basel_epe": 0.0,
                "ore_basel_eepe": 0.0,
                "py_basel_epe": 0.0,
                "py_basel_eepe": 0.0,
            },
            parity={"parity_ready": True, "summary": {"requested_xva_metrics": ["CVA", "DVA"], "portfolio_mode": True}},
            diagnostics={"engine": "python-lgm-portfolio"},
            maturity_date="",
            maturity_time=0.0,
            exposure_dates=["2025-01-01"],
            exposure_times=[0.0],
            py_epe=[0.0],
            py_ene=[0.0],
            py_pfe=[0.0],
            exposure_profile_by_trade={"dates": ["2025-01-01"], "times": [0.0], "closeout_epe": [0.0], "closeout_ene": [0.0], "pfe": [0.0]},
            exposure_profile_by_netting_set={"dates": ["2025-01-01"], "times": [0.0], "closeout_epe": [0.0], "closeout_ene": [0.0], "pfe": [0.0]},
            ore_basel_epe=0.0,
            ore_basel_eepe=0.0,
        )
        portfolio_root = ET.fromstring(
            """
            <Portfolio>
              <Trade id="AnchorBond"><TradeType>Bond</TradeType></Trade>
              <Trade id="SoFrOisSwap"><TradeType>Swap</TradeType></Trade>
            </Portfolio>
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("py_ore_tools.ore_snapshot_cli.validate_ore_input_snapshot", return_value={}):
                with patch("py_ore_tools.ore_snapshot_cli._portfolio_contains_swap_like_trade", return_value=True):
                    with patch(
                        "py_ore_tools.ore_snapshot_cli._portfolio_trade_context",
                        return_value=(portfolio_root, ["AnchorBond", "SoFrOisSwap"], "CPTY_A", "NS_A"),
                    ):
                        with patch("py_ore_tools.ore_snapshot_cli._compute_portfolio_xva_case", return_value=fake_summary) as portfolio_case:
                            with patch("py_ore_tools.ore_snapshot_cli._compute_snapshot_case", side_effect=AssertionError("single-trade path should not be used")):
                                with patch("py_ore_tools.ore_snapshot_cli._write_ore_compatible_reports"):
                                    with patch("py_ore_tools.ore_snapshot_cli._copy_native_ore_reports"):
                                        result = ore_snapshot_cli._run_case(REAL_CASE_XML, args, artifact_root=Path(tmp))
        portfolio_case.assert_called_once()
        self.assertEqual(result["trade_id"], "PORTFOLIO")
        self.assertEqual(result["xva"]["ore_cva"], 4.0)

    def test_write_reports_emits_per_trade_npv_rows_for_portfolio_price_only(self):
        case_summary = {
            "trade_id": "PORTFOLIO",
            "counterparty": "CPTY_A",
            "netting_set_id": "NS_A",
            "maturity_date": "2026-01-01",
            "maturity_time": 1.0,
            "pricing": {
                "trade_type": "Portfolio",
                "py_t0_npv": 12.0,
                "ore_t0_npv": 12.0,
                "t0_npv_abs_diff": 0.0,
                "report_ccy": "EUR",
                "currency": "EUR",
                "leg_source": "portfolio",
            },
            "portfolio_trade_rows": [
                {
                    "trade_id": "T1",
                    "trade_type": "Swap",
                    "counterparty": "CPTY_A",
                    "netting_set_id": "NS_A",
                    "maturity_date": "2025-01-01",
                    "maturity_time": 0.5,
                    "pricing": {"py_t0_npv": 5.0, "ore_t0_npv": 5.0, "report_ccy": "EUR"},
                    "notional": 100.0,
                    "notional_currency": "EUR",
                },
                {
                    "trade_id": "T2",
                    "trade_type": "Swap",
                    "counterparty": "CPTY_A",
                    "netting_set_id": "NS_A",
                    "maturity_date": "2026-01-01",
                    "maturity_time": 1.0,
                    "pricing": {"py_t0_npv": 7.0, "ore_t0_npv": 7.0, "report_ccy": "EUR"},
                    "notional": 120.0,
                    "notional_currency": "EUR",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            ore_snapshot_cli._write_ore_compatible_reports(out_dir, case_summary)
            with open(out_dir / "npv.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["#TradeId"] for row in rows], ["T1", "T2"])
        self.assertEqual([float(row["NPV(Base)"]) for row in rows], [5.0, 7.0])
        self.assertEqual([float(row["Notional"]) for row in rows], [100.0, 120.0])

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
        payload = ore_snapshot_cli._compute_price_only_case(
            TA001_EQUITY_CASE_XML,
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "EQ_CALL_STOXX50E")
        self.assertEqual(payload["pricing"]["trade_type"], "EquityOption")
        self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
        self.assertEqual(payload["pricing"]["pricing_mode"], "python_equity_option_premium_surface")
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 0.1)

    def test_ta001_equity_option_matches_explicit_ore_npv(self):
        payload = ore_snapshot_cli._compute_price_only_case(
            TA001_EQUITY_CASE_XML,
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "EQ_CALL_STOXX50E")
        self.assertEqual(payload["pricing"]["trade_type"], "EquityOption")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_equity_option_premium_surface")
        self.assertTrue(payload["diagnostics"]["using_expected_output"])
        self.assertAlmostEqual(float(payload["pricing"]["ore_t0_npv"]), 236.649138, places=6)
        self.assertAlmostEqual(float(payload["pricing"]["py_t0_npv"]), 236.7, places=6)
        self.assertLess(float(payload["pricing"]["t0_npv_abs_diff"]), 0.1)

    def test_example22_equity_option_price_only_case_uses_native_black_path(self):
        payload = ore_snapshot_cli._compute_price_only_case(
            EXAMPLE22_EQUITY_CASE_XML,
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "EQ_CALL_SP5")
        self.assertEqual(payload["pricing"]["trade_type"], "EquityOption")
        self.assertEqual(payload["diagnostics"]["engine"], "python_price_only")
        self.assertEqual(payload["pricing"]["pricing_mode"], "python_equity_option_black")
        self.assertIn("py_t0_npv", payload["pricing"])
        self.assertGreater(payload["pricing"]["py_t0_npv"], 0.0)
        self.assertLess(payload["pricing"]["t0_npv_abs_diff"], 100.0)

    def test_example22_equity_option_matches_explicit_ore_npv_within_black_tolerance(self):
        payload = ore_snapshot_cli._compute_price_only_case(
            EXAMPLE22_EQUITY_CASE_XML,
            anchor_t0_npv=False,
            use_reference_artifacts=True,
        )
        self.assertEqual(payload["trade_id"], "EQ_CALL_SP5")
        self.assertEqual(payload["pricing"]["trade_type"], "EquityOption")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_equity_option_black")
        self.assertTrue(payload["diagnostics"]["using_expected_output"])
        self.assertAlmostEqual(float(payload["pricing"]["ore_t0_npv"]), 132179.29212, places=5)
        self.assertAlmostEqual(
            float(payload["pricing"]["py_t0_npv"]),
            132103.23843732217,
            places=6,
        )
        self.assertLess(float(payload["pricing"]["t0_npv_abs_diff"]), 100.0)

    def test_inflation_capfloor_matches_explicit_ore_npv(self):
        payload = ore_snapshot_cli._compute_price_only_case(INFLATION_CAPFLOOR_CASE_XML, anchor_t0_npv=False)
        self.assertEqual(payload["trade_id"], "YearOnYear_Cap")
        self.assertEqual(payload["pricing"]["trade_type"], "CapFloor")
        self.assertEqual(payload["diagnostics"]["pricing_mode"], "python_inflation_yy_capfloor")
        self.assertTrue(payload["diagnostics"]["using_expected_output"])
        self.assertAlmostEqual(float(payload["pricing"]["ore_t0_npv"]), 25065.914792, places=6)
        self.assertAlmostEqual(
            float(payload["pricing"]["py_t0_npv"]),
            float(payload["pricing"]["ore_t0_npv"]),
            places=6,
        )
        self.assertEqual(float(payload["pricing"]["t0_npv_abs_diff"]), 0.0)

    def test_inflation_swap_trade_id_override_prices_selected_trade(self):
        ore_xml = TOOLS_DIR / "Examples" / "Legacy" / "Example_17" / "Input" / "ore.xml"
        expected = {
            "CPI_Swap_1": (2167085.146541, "CPI"),
            "CPI_Swap_2": (829612.548929, "CPI"),
            "YearOnYear_Swap": (267955.312135, "YY"),
        }

        for trade_id, (ore_npv, inflation_kind) in expected.items():
            with self.subTest(trade_id=trade_id):
                payload = ore_snapshot_cli._compute_price_only_case(
                    ore_xml,
                    anchor_t0_npv=False,
                    trade_id_override=trade_id,
                )
                self.assertEqual(payload["trade_id"], trade_id)
                self.assertEqual(payload["pricing"]["inflation_kind"], inflation_kind)
                self.assertAlmostEqual(float(payload["pricing"]["ore_t0_npv"]), ore_npv, places=6)
                self.assertAlmostEqual(float(payload["pricing"]["py_t0_npv"]), ore_npv, places=6)
                self.assertEqual(float(payload["pricing"]["t0_npv_abs_diff"]), 0.0)

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

    def test_report_examples_mode_retries_compare_xva_cases_with_auto_lgm_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples_root = root / "Examples"
            case_dir = examples_root / "FamilyA" / "Input"
            case_dir.mkdir(parents=True)
            ore_xml = case_dir / "ore.xml"
            ore_xml.write_text("<ORE />", encoding="utf-8")

            def fake_run_case(case_ore_xml, run_args, *, artifact_root):
                out_dir = artifact_root / ore_snapshot_cli._case_slug(case_ore_xml)
                out_dir.mkdir(parents=True, exist_ok=True)
                if getattr(run_args, "lgm_param_source", "auto") == "simulation_xml":
                    payload = {
                        "ore_xml": str(case_ore_xml),
                        "trade_id": case_ore_xml.stem,
                        "counterparty": "CPTY",
                        "netting_set_id": "NET",
                        "modes": ["xva"],
                        "xva": {"cva_rel_diff": 0.50, "dva_rel_diff": 0.25, "fba_rel_diff": 0.0, "fca_rel_diff": 0.0},
                        "pricing": None,
                        "parity": {"parity_ready": True, "summary": {"requested_xva_metrics": ["CVA", "DVA"]}},
                        "diagnostics": {"engine": "compare"},
                        "input_validation": {"input_links_valid": True, "issues": []},
                        "pass_flags": {"xva_cva": False, "xva_dva": False},
                        "pass_all": False,
                    }
                else:
                    payload = {
                        "ore_xml": str(case_ore_xml),
                        "trade_id": case_ore_xml.stem,
                        "counterparty": "CPTY",
                        "netting_set_id": "NET",
                        "modes": ["xva"],
                        "xva": {"cva_rel_diff": 0.02, "dva_rel_diff": 0.01, "fba_rel_diff": 0.0, "fca_rel_diff": 0.0},
                        "pricing": None,
                        "parity": {"parity_ready": True, "summary": {"requested_xva_metrics": ["CVA", "DVA"]}},
                        "diagnostics": {"engine": "compare"},
                        "input_validation": {"input_links_valid": True, "issues": []},
                        "pass_flags": {"xva_cva": True, "xva_dva": True},
                        "pass_all": True,
                    }
                (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with patch("py_ore_tools.ore_snapshot_cli._examples_root", return_value=examples_root):
                with patch("py_ore_tools.ore_snapshot_cli._run_case", side_effect=fake_run_case):
                    with patch("py_ore_tools.ore_snapshot_cli._run_case_in_subprocess", side_effect=fake_run_case):
                        rc = ore_snapshot_cli.main(
                            [
                                "--report-examples",
                                "--xva",
                                "--lgm-param-source",
                                "simulation_xml",
                                "--report-root",
                                str(root / "report"),
                                "--report-workers",
                                "1",
                                "--report-refresh-every",
                                "1",
                            ]
                        )
            self.assertEqual(rc, 0)
            rows = list(csv.DictReader((root / "report" / "live_results.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bucket"], "clean_pass")
            payload = json.loads(Path(rows[0]["summary_path"]).read_text(encoding="utf-8"))
            self.assertTrue(payload["pass_all"])
            self.assertAlmostEqual(payload["xva"]["cva_rel_diff"], 0.02)

    def test_report_examples_mode_retries_sample_mismatch_cases_at_ore_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples_root = root / "Examples"
            case_dir = examples_root / "FamilyA" / "Input"
            case_dir.mkdir(parents=True)
            ore_xml = case_dir / "ore.xml"
            ore_xml.write_text("<ORE />", encoding="utf-8")

            def fake_run_case(case_ore_xml, run_args, *, artifact_root):
                out_dir = artifact_root / ore_snapshot_cli._case_slug(case_ore_xml)
                out_dir.mkdir(parents=True, exist_ok=True)
                paths = int(getattr(run_args, "paths", 0))
                if paths == 2000:
                    payload = {
                        "ore_xml": str(case_ore_xml),
                        "trade_id": case_ore_xml.stem,
                        "counterparty": "CPTY",
                        "netting_set_id": "NET",
                        "modes": ["xva"],
                        "xva": {"cva_rel_diff": 0.40, "dva_rel_diff": 0.30, "fba_rel_diff": 0.0, "fca_rel_diff": 0.0},
                        "pricing": None,
                        "parity": {"parity_ready": True, "summary": {"requested_xva_metrics": ["CVA", "DVA"]}},
                        "diagnostics": {
                            "engine": "compare",
                            "sample_count_mismatch": True,
                            "ore_samples": 200,
                            "python_paths": 2000,
                        },
                        "input_validation": {"input_links_valid": True, "issues": []},
                        "pass_flags": {"xva_cva": False, "xva_dva": False},
                        "pass_all": False,
                    }
                else:
                    payload = {
                        "ore_xml": str(case_ore_xml),
                        "trade_id": case_ore_xml.stem,
                        "counterparty": "CPTY",
                        "netting_set_id": "NET",
                        "modes": ["xva"],
                        "xva": {"cva_rel_diff": 0.03, "dva_rel_diff": 0.02, "fba_rel_diff": 0.0, "fca_rel_diff": 0.0},
                        "pricing": None,
                        "parity": {"parity_ready": True, "summary": {"requested_xva_metrics": ["CVA", "DVA"]}},
                        "diagnostics": {
                            "engine": "compare",
                            "sample_count_mismatch": False,
                            "ore_samples": 200,
                            "python_paths": 200,
                        },
                        "input_validation": {"input_links_valid": True, "issues": []},
                        "pass_flags": {"xva_cva": True, "xva_dva": True},
                        "pass_all": True,
                    }
                (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with patch("py_ore_tools.ore_snapshot_cli._examples_root", return_value=examples_root):
                with patch("py_ore_tools.ore_snapshot_cli._run_case", side_effect=fake_run_case):
                    rc = ore_snapshot_cli.main(
                        [
                            "--report-examples",
                            "--xva",
                            "--paths",
                            "2000",
                            "--report-root",
                            str(root / "report"),
                            "--report-workers",
                            "1",
                            "--report-refresh-every",
                            "1",
                        ]
                    )
            self.assertEqual(rc, 0)
            rows = list(csv.DictReader((root / "report" / "live_results.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bucket"], "clean_pass")
            payload = json.loads(Path(rows[0]["summary_path"]).read_text(encoding="utf-8"))
            self.assertTrue(payload["pass_all"])
            self.assertFalse(payload["diagnostics"]["sample_count_mismatch"])
            self.assertEqual(payload["diagnostics"]["python_paths"], 200)

    def test_report_examples_mode_does_not_change_normal_single_case_behavior(self):
        args = ore_snapshot_cli.build_parser().parse_args([str(REAL_CASE_XML), "--price"])
        self.assertFalse(args.report_examples)
        self.assertEqual(args.report_workers, 12)


if __name__ == "__main__":
    unittest.main()
