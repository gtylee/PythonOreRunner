import csv
from dataclasses import replace
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import pytest

from pythonore.domain.dataclasses import GenericProduct, RuntimeConfig, XVAAnalyticConfig
from pythonore.compute.irs_xva_utils import _schedule_from_leg
from pythonore.io.loader import XVALoader
from pythonore.mapping.mapper import map_snapshot
from pythonore.runtime.runtime import XVAEngine


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
    assert pays[1].isoformat() == "2024-07-04"


def test_generic_rate_swap_parses_xccy_fx_reset_metadata():
    snapshot = XVALoader.from_files(str(TOOLS_DIR / "Examples" / "Legacy" / "Example_63"), ore_file="Input/ore_valid_xccy.xml")
    adapter = XVAEngine.python_lgm_default(fallback_to_swig=False).adapter
    adapter._ensure_py_lgm_imports()
    trade = next(t for t in snapshot.portfolio.trades if t.trade_id == "XccySwap")
    state = adapter._build_generic_rate_swap_legs(trade, snapshot)
    assert state is not None
    usd_leg = next(leg for leg in state["rate_legs"] if leg["ccy"] == "USD")
    fx_reset = usd_leg["fx_reset"]
    assert fx_reset is not None
    assert fx_reset["foreign_currency"] == "EUR"
    assert fx_reset["foreign_amount"] == 95000000.0
    assert fx_reset["fx_index"] == "FX-ECB-EUR-USD"
    assert usd_leg["notional_initial_exchange"] is True
    assert usd_leg["notional_final_exchange"] is True


def test_example59_cap_usd_sofr_matches_ore_npv_when_replaying_flows_csv():
    if not LOCAL_ORE_BINARY.exists():
        return
    pytest.xfail("Example_59 Cap_USD_SOFR still uses the local overnight cap/floor shim; exact QuantExt parity is not wired in")
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
            result = adapter.run(snapshot, mapped=map_snapshot(snapshot), run_id="example59-cap-ore-flows-parity")

        assert math.isclose(float(result.pv_total), ore_npv_base, rel_tol=1.0e-12, abs_tol=1.0e-9)
        assert math.isclose(
            float(result.cubes["npv_cube"].payload["Cap_USD_SOFR"]["npv_mean"][0]),
            ore_npv_base,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
    finally:
        tmp.cleanup()


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
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


def test_example63_xccy_trade_matches_ore_npv_when_replaying_flows_csv():
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
    with (TOOLS_DIR / "Examples" / "Legacy" / "Example_63" / "Output" / "valid_xccy" / "npv.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        ore_rows = csv.DictReader(handle)
        ore_npv_row = next(row for row in ore_rows if (row.get("TradeId") or row.get("#TradeId")) == "XccySwap")
    ore_npv_base = float(ore_npv_row["NPV(Base)"])

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

    assert math.isclose(float(result.pv_total), ore_npv_base, rel_tol=1.0e-12, abs_tol=1.0e-9)
    assert math.isclose(
        float(result.cubes["npv_cube"].payload["XccySwap"]["npv_mean"][0]),
        ore_npv_base,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    )


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
        with (case_root / "Output" / "valid_cap" / "npv.csv").open(newline="", encoding="utf-8") as handle:
            ore_row = next(row for row in csv.DictReader(handle) if (row.get("TradeId") or row.get("#TradeId")) == "Cap")
        ore_npv_base = float(ore_row["NPV(Base)"])

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

        assert math.isclose(float(result.pv_total), ore_npv_base, rel_tol=1.0e-12, abs_tol=1.0e-9)
        assert math.isclose(
            float(result.cubes["npv_cube"].payload["Cap"]["npv_mean"][0]),
            ore_npv_base,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
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


def test_python_runtime_supports_real_bma_basis_case_without_fallback():
    snapshot = _load_case("Examples/Legacy/Example_27/Input")
    result = _run_python(snapshot, "bma-test")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


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
            "pythonore.runtime.runtime._parse_lgm_params_from_calibration_xml_text",
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


def test_build_irs_legs_can_use_flows_csv_when_explicitly_requested():
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

    with patch.object(adapter._irs_utils, "load_ore_legs_from_flows", return_value=flow_legs) as flow_loader:
        legs = adapter._build_irs_legs(trade, map_snapshot(snapshot), snapshot)

    flow_loader.assert_called_once()
    np.testing.assert_allclose(np.asarray(legs["fixed_notional"], dtype=float), np.array([1_234_567.0]))


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
