from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import example_ore_snapshot_usd_all_rates_products as example


def _portfolio_fragment(xml: str) -> ET.Element:
    return ET.fromstring(f"<Portfolio>{xml}</Portfolio>")


def test_cms_examples_avoid_missing_asof_fixings_and_keep_regular_irs_start() -> None:
    cms_fragments = (
        example._cms_trade_xml("0001"),
        example._cmsspread_trade_xml("0001"),
        example._digital_cmsspread_trade_xml("0001"),
    )
    for fragment in cms_fragments:
        trade = _portfolio_fragment(fragment)
        starts = [node.text for node in trade.findall(".//ScheduleData/Rules/StartDate")]
        assert starts
        assert set(starts) == {example.CMS_START_DATE}

    irs = _portfolio_fragment(example._irs_trade_xml("0001"))
    assert {node.text for node in irs.findall(".//ScheduleData/Rules/StartDate")} == {example.ASOF_DATE}


def test_simulation_requests_optionlet_keys_and_jpy_swap_indices() -> None:
    simulation = ET.fromstring(example._simulation_xml())
    market = simulation.find("Market")
    assert {node.text for node in market.findall("./CapFloorVolatilities/Keys/Key")} == {
        "USD-LIBOR-3M",
        "USD-SOFR-3M",
    }
    assert {node.findtext("Name") for node in market.findall("./SwapIndices/SwapIndex")} >= {
        "JPY-CMS-1Y",
        "JPY-CMS-30Y",
    }
    assert "JPY" in {node.text for node in market.findall("./SwaptionVolatilities/Currencies/Currency")}


def test_generated_market_inputs_include_curves_and_jpy_proxy(tmp_path) -> None:
    case_root = tmp_path / "all-rates"
    ore_path, portfolio_path = example._write_files(case_root, count_per_type=1)
    input_dir = case_root / "Input"
    assert portfolio_path.exists()

    ore = ET.parse(ore_path)
    parameters = {node.get("name"): node.text for node in ore.findall("./Setup/Parameter")}
    assert parameters["curveConfigFile"] == "curveconfig.xml"
    assert parameters["marketConfigFile"] == "todaysmarket.xml"
    assert parameters["conventionsFile"] == "conventions.xml"

    curve_config = ET.parse(input_dir / "curveconfig.xml")
    usd_surface = next(
        node
        for node in curve_config.findall("./CapFloorVolatilities/CapFloorVolatility")
        if node.findtext("CurveId") == "USD"
    )
    assert usd_surface.findtext("InputType") == "OptionletVolatilities"
    assert usd_surface.findtext("RateComputationPeriod") == "1D"

    for curve_id in ("USD-LIBOR-3M", "USD-SOFR-3M"):
        proxy = next(
            node
            for node in curve_config.findall("./CapFloorVolatilities/CapFloorVolatility")
            if node.findtext("CurveId") == curve_id
        )
        assert proxy.findtext("./ProxyConfig/Source/CurveId") == "CapFloorVolatility/USD/USD"
        assert proxy.findtext("./ProxyConfig/Source/RateComputationPeriod") == "1D"
        assert proxy.findtext("./ProxyConfig/Target/RateComputationPeriod") == "3M"

    jpy_vol = next(
        node
        for node in curve_config.findall("./SwaptionVolatilities/SwaptionVolatility")
        if node.findtext("CurveId") == "JPY"
    )
    assert jpy_vol.findtext("./ProxyConfig/Target/ShortSwapIndexBase") == "JPY-CMS-1Y"
    assert jpy_vol.findtext("./ProxyConfig/Target/SwapIndexBase") == "JPY-CMS-30Y"
    assert "not calibrated" in jpy_vol.findtext("CurveDescription")

    market_config = ET.parse(input_dir / "todaysmarket.xml")
    assert market_config.findtext('./SwaptionVolatilities[@id="default"]/SwaptionVolatility[@key="JPY"]') == (
        "SwaptionVolatility/JPY/JPY"
    )
    assert market_config.findtext('./CapFloorVolatilities[@id="default"]/CapFloorVolatility[@key="USD-LIBOR-3M"]') == (
        "CapFloorVolatility/USD-LIBOR-3M/USD-LIBOR-3M"
    )
    assert {
        node.get("name") for node in market_config.findall('./SwapIndexCurves[@id="default"]/SwapIndex')
    } >= {"JPY-CMS-1Y", "JPY-CMS-30Y"}

    conventions = ET.parse(input_dir / "conventions.xml")
    assert {node.findtext("Id") for node in conventions.findall("SwapIndex")} >= {
        "JPY-CMS-1Y",
        "JPY-CMS-30Y",
    }


def test_checked_in_all_rates_samples_use_repaired_inputs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    case_names = (
        "USD_AllRatesProductsSnapshot",
        "USD_AllRatesProductsSnapshot_Check",
        "USD_AllRatesProductsSnapshot_20PerType",
        "USD_AllRatesProductsSnapshot_MoreTypes",
        "USD_AllRatesProductsSnapshot_134PerType",
    )
    for case_name in case_names:
        input_dir = repo_root / "Examples" / "Generated" / case_name / "Input"
        ore = ET.parse(input_dir / "ore.xml")
        parameters = {node.get("name"): node.text for node in ore.findall("./Setup/Parameter")}
        assert parameters["curveConfigFile"] == "curveconfig.xml"
        assert parameters["conventionsFile"] == "conventions.xml"
        assert parameters["marketConfigFile"] == "todaysmarket.xml"
        assert all((input_dir / filename).is_file() for filename in (
            "curveconfig.xml",
            "conventions.xml",
            "todaysmarket.xml",
        ))

        simulation = ET.parse(input_dir / "simulation.xml")
        capfloor_keys = {
            node.text for node in simulation.findall("./Market/CapFloorVolatilities/Keys/Key")
        }
        assert capfloor_keys == {"USD-LIBOR-3M", "USD-SOFR-3M"}

        portfolio = ET.parse(input_dir / "portfolio_usd_all_rates_products.xml")
        for trade in portfolio.findall("./Trade"):
            if trade.find(".//CMSLegData") is None:
                continue
            starts = trade.findall(".//ScheduleData/Rules/StartDate")
            assert starts
            assert {node.text for node in starts} == {example.CMS_START_DATE}
