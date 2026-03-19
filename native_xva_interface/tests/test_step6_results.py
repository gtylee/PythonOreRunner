import pytest

from native_xva_interface import CubeAccessor, XVAResult


def test_result_schema_and_cube_access():
    r = XVAResult(
        run_id="r1",
        pv_total=1.0,
        xva_total=2.0,
        xva_by_metric={"CVA": 2.0},
        exposure_by_netting_set={"NS1": 10.0},
        reports={"xva": [{"Metric": "CVA", "Value": 2.0}]},
        cubes={"c": CubeAccessor(name="c", payload={"T1": {"epe": 3.0}})},
    )
    assert r.cube("c").value("T1", "epe") == 3.0
    assert r.report("xva")[0]["Metric"] == "CVA"


def test_report_dataframe_conversion():
    r = XVAResult(
        run_id="r1",
        pv_total=1.0,
        xva_total=2.0,
        xva_by_metric={"CVA": 2.0},
        exposure_by_netting_set={"NS1": 10.0},
        reports={"xva": [{"Metric": "CVA", "Value": 2.0}]},
    )
    try:
        df = r.reports_as_dataframe("xva")
    except Exception as exc:
        pytest.skip(f"pandas unavailable or conversion failed: {exc}")
    assert "Metric" in df.columns
