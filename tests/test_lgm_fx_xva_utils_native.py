from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from pythonore.compute.lgm_fx_xva_utils import (
    FxOptionDef,
    XccyFloatLegDef,
    build_two_ccy_hybrid,
    fx_option_npv,
    xccy_float_float_swap_npv,
)
from pythonore.hw2f_integration import simulate_hw2f_exposure_paths
from pythonore.hw2f_ore_runner import build_hw2f_case
from pythonore.runtime.stress_classic_templates import (
    stress_classic_fixing_lines,
    stress_classic_market_lines,
    stress_classic_xml_buffers,
)


def _curve(rate: float):
    return lambda t: math.exp(-float(rate) * float(t))


def _hybrid():
    return build_two_ccy_hybrid(
        pair="EUR/USD",
        ir_specs={
            "EUR": {"alpha": 0.01, "kappa": 0.03},
            "USD": {"alpha": 0.01, "kappa": 0.03},
        },
        fx_vol=0.2,
    )


def test_fx_option_npv_report_currency_conversion_and_finite():
    hybrid = _hybrid()
    s_t = np.array([1.10, 1.15, 1.20], dtype=float)
    x_dom_t = np.zeros(3, dtype=float)
    x_for_t = np.zeros(3, dtype=float)
    quote_ccy = fx_option_npv(
        hybrid,
        FxOptionDef(
            trade_id="FXOPT_1",
            pair="EUR/USD",
            notional_base=1_000_000,
            strike=1.12,
            maturity=2.0,
            option_type="CALL",
            report_ccy="USD",
        ),
        0.5,
        s_t,
        x_dom_t,
        x_for_t,
        _curve(0.03),
        _curve(0.02),
        0.18,
    )
    base_ccy = fx_option_npv(
        hybrid,
        FxOptionDef(
            trade_id="FXOPT_1",
            pair="EUR/USD",
            notional_base=1_000_000,
            strike=1.12,
            maturity=2.0,
            option_type="CALL",
            report_ccy="EUR",
        ),
        0.5,
        s_t,
        x_dom_t,
        x_for_t,
        _curve(0.03),
        _curve(0.02),
        0.18,
    )

    assert np.all(np.isfinite(quote_ccy))
    assert np.all(np.isfinite(base_ccy))
    assert np.allclose(base_ccy, quote_ccy / s_t)


def test_xccy_float_float_swap_npv_regression():
    hybrid = _hybrid()
    result = xccy_float_float_swap_npv(
        hybrid=hybrid,
        domestic_ccy="USD",
        leg1=XccyFloatLegDef(
            ccy="EUR",
            pay_time=np.array([1.0, 2.0]),
            start_time=np.array([0.0, 1.0]),
            end_time=np.array([1.0, 2.0]),
            accrual=np.array([1.0, 1.0]),
            notional=np.array([1_000_000.0, 1_000_000.0]),
            spread=np.array([0.001, 0.001]),
            sign=np.array([1.0, 1.0]),
        ),
        leg2=XccyFloatLegDef(
            ccy="USD",
            pay_time=np.array([1.0, 2.0]),
            start_time=np.array([0.0, 1.0]),
            end_time=np.array([1.0, 2.0]),
            accrual=np.array([1.0, 1.0]),
            notional=np.array([1_100_000.0, 1_100_000.0]),
            spread=np.array([0.002, 0.002]),
            sign=np.array([-1.0, -1.0]),
        ),
        t=0.0,
        x_by_ccy={"EUR": np.zeros(3), "USD": np.zeros(3)},
        s_fx_by_pair={"EUR/USD": np.full(3, 1.1)},
        disc_curves={"EUR": _curve(0.02), "USD": _curve(0.03)},
        fwd_curves={"EUR": _curve(0.025), "USD": _curve(0.035)},
    )

    assert np.allclose(result, np.full(3, -22945.39475074), rtol=1.0e-10, atol=1.0e-8)


def test_xccy_float_float_swap_npv_handles_negative_start_times_and_nan_spread():
    hybrid = _hybrid()
    result = xccy_float_float_swap_npv(
        hybrid=hybrid,
        domestic_ccy="USD",
        leg1=XccyFloatLegDef(
            ccy="EUR",
            pay_time=np.array([1.0, 1.5]),
            start_time=np.array([0.0, 0.5]),
            end_time=np.array([1.0, 1.5]),
            accrual=np.array([1.0, 1.0]),
            notional=np.array([1_000_000.0, 1_000_000.0]),
            spread=np.array([np.nan, 0.001]),
            sign=np.array([1.0, 1.0]),
        ),
        leg2=XccyFloatLegDef(
            ccy="USD",
            pay_time=np.array([1.0, 1.5]),
            start_time=np.array([0.0, 0.5]),
            end_time=np.array([1.0, 1.5]),
            accrual=np.array([1.0, 1.0]),
            notional=np.array([1_050_000.0, 1_050_000.0]),
            spread=np.array([0.002, np.nan]),
            sign=np.array([-1.0, -1.0]),
        ),
        t=0.5,
        x_by_ccy={"EUR": np.zeros(4), "USD": np.zeros(4)},
        s_fx_by_pair={"EUR/USD": np.full(4, 1.1)},
        disc_curves={"EUR": _curve(0.02), "USD": _curve(0.03)},
        fwd_curves={"EUR": _curve(0.025), "USD": _curve(0.035)},
    )

    assert np.all(np.isfinite(result))


def test_stress_classic_templates_smoke():
    xml_buffers = stress_classic_xml_buffers(num_paths=64)

    assert {"pricingengine.xml", "simulation.xml", "netting.xml", "counterparty.xml", "todaysmarket.xml", "curveconfig.xml"}.issubset(xml_buffers)
    assert len(stress_classic_market_lines()) > 0
    assert len(stress_classic_fixing_lines()) > 0

    for name, text in xml_buffers.items():
        root = ET.fromstring(text)
        assert root.tag
        assert name.endswith(".xml")


def test_simulate_hw2f_exposure_paths_returns_shapes_from_generated_case():
    with tempfile.TemporaryDirectory() as tmp:
        case = build_hw2f_case(
            Path(tmp) / "case" / "Input",
            sigma=[[[0.002, 0.008], [0.009, 0.001]]],
            kappa=[[0.01, 0.2]],
            times=[],
            samples=8,
            grid="12,1Y",
        )
        curves_csv = case.output_dir / "curves.csv"
        curves_csv.write_text(
            "\n".join(
                [
                    "Date,EUR-EONIA",
                    "2016-02-05,1.0",
                    "2017-02-05,0.99",
                    "2018-02-05,0.975",
                    "2020-02-05,0.94",
                    "2028-02-05,0.80",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        times, x_paths, npv_paths = simulate_hw2f_exposure_paths(case.case_dir, n_paths=6, seed=7)

    assert times.ndim == 1
    assert x_paths.shape == (times.size, 6, 2)
    assert npv_paths.shape == (times.size, 6)
    assert np.all(np.isfinite(x_paths))
    assert np.all(np.isfinite(npv_paths))
