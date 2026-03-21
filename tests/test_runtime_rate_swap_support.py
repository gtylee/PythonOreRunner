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

from pythonore.domain.dataclasses import GenericProduct
from pythonore.io.loader import XVALoader
from pythonore.mapping.mapper import map_snapshot
from pythonore.runtime.runtime import XVAEngine


TOOLS_DIR = Path(__file__).resolve().parents[1]
LOCAL_ORE_BINARY = Path("/Users/gordonlee/Documents/Engine/build/App/ore")


def _load_case(case_dir: str, ore_file: str = "ore.xml"):
    snapshot = XVALoader.from_files(str(TOOLS_DIR / case_dir), ore_file=ore_file)
    return replace(snapshot, config=replace(snapshot.config, num_paths=4))


def _run_python(snapshot, run_id: str):
    with patch("pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore", return_value=None):
        return XVAEngine.python_lgm_default(fallback_to_swig=False).adapter.run(
            snapshot,
            mapped=map_snapshot(snapshot),
            run_id=run_id,
        )


def _clone_pricing_only_case(case_name: str, *, trade_ids):
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

    ore_tree = ET.parse(case_root / "Input" / "ore.xml")
    ore_root = ore_tree.getroot()
    for analytic in ore_root.findall(".//Analytic"):
        kind = analytic.get("type")
        active = analytic.find("./Parameter[@name='active']")
        if active is not None:
            active.text = "Y" if kind in {"npv", "cashflow", "curves"} else "N"
    ore_tree.write(case_root / "Input" / "ore.xml", encoding="utf-8", xml_declaration=True)
    return tmp, case_root


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
        assert math.isclose(float(result.pv_total), 406186.32950821426, rel_tol=1.0e-10, abs_tol=1.0e-8)
    finally:
        tmp.cleanup()


def test_python_runtime_supports_real_bma_basis_case_without_fallback():
    snapshot = _load_case("Examples/Legacy/Example_27/Input")
    result = _run_python(snapshot, "bma-test")
    coverage = result.metadata["coverage"]
    assert coverage["fallback_trades"] == 0
    assert coverage["unsupported"] == []
    assert math.isfinite(float(result.pv_total))


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

        snapshot = XVALoader.from_files(str(input_dir), ore_file="ore.xml")
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
            "pythonore.io.ore_snapshot.calibrate_lgm_params_via_ore",
            return_value=runtime_params,
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
        assert math.isclose(float(result.pv_total), 342727.856174012, rel_tol=1.0e-10, abs_tol=1.0e-8)
        provenance = result.metadata["input_provenance"]
        assert provenance["market"] == "market_overlay"
        assert provenance["model_params"] == "calibration"
    finally:
        tmp.cleanup()


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
        assert math.isclose(float(py_result.pv_total), 342727.856174012, rel_tol=1.0e-10, abs_tol=1.0e-8)
        assert math.isclose(float(py_result.pv_total - ore_npv), 62921.199573012015, rel_tol=1.0e-10, abs_tol=1.0e-8)
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
        assert math.isclose(float(py_result.pv_total), 406186.32950821426, rel_tol=1.0e-10, abs_tol=1.0e-8)
        assert math.isclose(float(py_result.pv_total - ore_npv), 77914.36381021427, rel_tol=1.0e-10, abs_tol=1.0e-8)
    finally:
        tmp.cleanup()
