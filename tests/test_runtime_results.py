from pythonore.runtime.results import xva_total_from_metrics


def test_xva_total_uses_signed_ore_adjustment_convention():
    metrics = {"CVA": 100.0, "DVA": 40.0, "FVA": 7.0, "FBA": 3.0, "FCA": 4.0, "MVA": -2.0}

    assert xva_total_from_metrics(metrics) == 65.0


def test_xva_total_builds_fva_from_fba_fca_when_total_fva_absent():
    metrics = {"CVA": 100.0, "DVA": 40.0, "FBA": 3.0, "FCA": 4.0}

    assert xva_total_from_metrics(metrics) == 67.0
