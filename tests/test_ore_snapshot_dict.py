from __future__ import annotations

import numpy as np

from py_ore_tools.lgm import LGMParams
from py_ore_tools.ore_snapshot import OreSnapshot


def test_ore_snapshot_to_dict_exposes_lgm_params():
    snap = OreSnapshot(
        ore_xml_path="/tmp/ore.xml",
        asof_date="2024-01-01",
        lgm_params=LGMParams(
            alpha_times=(1.0, 2.0),
            alpha_values=(0.01, 0.02, 0.03),
            kappa_times=(3.0,),
            kappa_values=(0.04, 0.05),
            shift=0.1,
            scaling=1.2,
        ),
        alpha_source="simulation",
        measure="LGM",
        seed=42,
        n_samples=128,
        domestic_ccy="EUR",
        node_tenors=np.array([1.0, 2.0]),
        model_day_counter="A365F",
        report_day_counter="ActualActual(ISDA)",
        trade_id="T1",
        counterparty="CPTY_A",
        netting_set_id="CPTY_A",
        legs={
            "fixed_pay_time": np.array([1.0, 2.0]),
            "meta": {"index": "EUR-EURIBOR-6M"},
        },
        discount_column="EUR-EONIA",
        forward_column="EUR-EURIBOR-6M",
        xva_discount_column="EUR-EONIA",
        curve_times_disc=np.array([0.0, 1.0]),
        curve_dfs_disc=np.array([1.0, 0.99]),
        curve_times_fwd=np.array([0.0, 1.0]),
        curve_dfs_fwd=np.array([1.0, 0.98]),
        curve_times_xva_disc=np.array([0.0, 1.0]),
        curve_dfs_xva_disc=np.array([1.0, 0.97]),
        p0_disc=lambda t: 1.0,
        p0_fwd=lambda t: 1.0,
        p0_xva_disc=lambda t: 1.0,
        exposure_times=np.array([0.0, 1.0]),
        exposure_dates=np.array(["2024-01-01", "2025-01-01"]),
        exposure_model_times=np.array([0.0, 1.0]),
        ore_epe=np.array([0.0, 10.0]),
        ore_ene=np.array([0.0, 5.0]),
        ore_t0_npv=12.5,
        ore_cva=3.5,
        recovery=0.4,
        hazard_times=np.array([1.0, 2.0]),
        hazard_rates=np.array([0.01, 0.02]),
    )

    payload = snap.to_dict()

    assert payload["lgm_params"]["alpha_times"] == [1.0, 2.0]
    assert payload["lgm_params"]["alpha_values"] == [0.01, 0.02, 0.03]
    assert payload["lgm_params"]["kappa_times"] == [3.0]
    assert payload["lgm_params"]["kappa_values"] == [0.04, 0.05]
    assert payload["lgm_params"]["shift"] == 0.1
    assert payload["lgm_params"]["scaling"] == 1.2
    assert payload["node_tenors"] == [1.0, 2.0]
    assert payload["legs"]["fixed_pay_time"] == [1.0, 2.0]
    assert payload["ore_xml_path"] == "/tmp/ore.xml"
    assert "p0_disc" not in payload
