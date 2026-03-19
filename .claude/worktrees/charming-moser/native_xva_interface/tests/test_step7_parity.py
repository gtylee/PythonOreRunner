from native_xva_interface import XVAResult, compare_results, ParityTolerance


def test_parity_passes_within_tolerance():
    a = XVAResult(
        run_id="a",
        pv_total=100.0,
        xva_total=10.0,
        xva_by_metric={"CVA": 9.0, "FVA": 1.0},
        exposure_by_netting_set={"NS1": 1000.0},
    )
    b = XVAResult(
        run_id="b",
        pv_total=100.001,
        xva_total=10.0005,
        xva_by_metric={"CVA": 9.0001, "FVA": 1.0004},
        exposure_by_netting_set={"NS1": 1000.2},
    )
    out = compare_results(
        a,
        b,
        metric_tolerances={
            "pv_total": ParityTolerance(abs_tol=0.01, rel_tol=1e-3),
            "FVA": ParityTolerance(abs_tol=0.001, rel_tol=1e-3),
            "ns:NS1": ParityTolerance(abs_tol=1.0, rel_tol=1e-3),
        },
    )
    assert out.ok


def test_parity_flags_large_diff():
    a = XVAResult(
        run_id="a",
        pv_total=100.0,
        xva_total=10.0,
        xva_by_metric={"CVA": 9.0},
        exposure_by_netting_set={"NS1": 1000.0},
    )
    b = XVAResult(
        run_id="b",
        pv_total=130.0,
        xva_total=12.0,
        xva_by_metric={"CVA": 11.0},
        exposure_by_netting_set={"NS1": 1400.0},
    )
    out = compare_results(a, b)
    assert not out.ok
    assert out.diffs
