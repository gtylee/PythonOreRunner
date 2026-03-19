from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import pytest

from native_xva_interface import (
    FixingsData,
    IRS,
    MarketData,
    MarketQuote,
    OreSnapshotPythonLgmSensitivityComparator,
    Portfolio,
    Trade,
    XVAConfig,
    XVAEngine,
    XVAResult,
    XVASnapshot,
)
from native_xva_interface.sensitivity import (
    _bump_hazard_curve_quotes_by_survival_node,
    _should_skip_generic_normalization,
    _curve_factor_specs_from_ore_entries,
    normalize_raw_discount_quote_key,
)
from native_xva_interface.sensitivity import ORESensitivityEntry


class _LinearQuoteAdapter:
    def run(self, snapshot, mapped, run_id):
        qmap = {q.key: q.value for q in snapshot.market.raw_quotes}
        curve_shocks = snapshot.config.params.get("python.curve_node_shocks", {})
        discount = curve_shocks.get("discount", {}).get("EUR", {}) if isinstance(curve_shocks, dict) else {}
        node_times = list(discount.get("node_times", []))
        node_shifts = list(discount.get("node_shifts", []))
        eur_zero_shift = 0.0
        for t, s in zip(node_times, node_shifts):
            if abs(float(t) - 5.0) < 1.0e-12:
                eur_zero_shift = float(s)
                break
        cva = 1000.0 * (float(qmap.get("ZERO/RATE/EUR/5Y", 0.0)) + eur_zero_shift) + 200.0 * float(
            qmap.get("HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", 0.0)
        )
        return XVAResult(
            run_id=run_id,
            pv_total=0.0,
            xva_total=cva,
            xva_by_metric={"CVA": cva},
            exposure_by_netting_set={"CPTY_A": 1.0},
            reports={},
            cubes={},
            metadata={"adapter": "linear-quote"},
        )


def _snapshot() -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/5Y", value=0.025),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.02),
            ),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.02, maturity_years=5.0, pay_fixed=True),
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", analytics=("CVA",), num_paths=8),
    )


def test_compare_matches_ore_zero_sensitivity_rows(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
                [
                    "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                    "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
                    f"CPTY_A,,false,SurvivalProbability/CPTY_A/1/5Y,0.0001,,0.0,EUR,0.0,{200.0 * (-1.0 / (5.0 * math.exp(-0.02 * 5.0))):.12f},#N/A",
                ]
            ),
        encoding="utf-8",
    )

    snap = replace(_snapshot(), config=replace(_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))
    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert len(result["comparisons"]) == 1
    assert result["unmatched_ore"] == []
    assert result["unmatched_python"] == []
    deltas = {row.normalized_factor: row.python_delta for row in result["comparisons"]}
    assert deltas["zero:EUR:5Y"] == pytest.approx(0.1)
    assert len(result["unsupported_ore"]) == 1
    assert result["unsupported_ore"][0].normalized_factor == "hazard:CPTY_A:5Y"
    assert len(result["unsupported_python"]) == 1
    assert result["unsupported_python"][0].normalized_factor == "hazard:CPTY_A:5Y"
    assert result["unsupported_factors"] == ["hazard:CPTY_A:5Y"]


def test_compare_excludes_credit_factors_from_parity_output(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
                "CPTY_A,,false,SurvivalProbability/CPTY_A/1/5Y,0.0001,,0.0,EUR,0.0,0.2,#N/A",
            ]
        ),
        encoding="utf-8",
    )
    snap = replace(_snapshot(), config=replace(_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))
    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert [c.normalized_factor for c in result["comparisons"]] == ["zero:EUR:5Y"]
    assert len(result["unsupported_ore"]) == 1
    assert result["unsupported_ore"][0].normalized_factor == "hazard:CPTY_A:5Y"
    assert result["unsupported_factors"] == ["hazard:CPTY_A:5Y"]
    assert result["notes"]


def test_compute_python_sensitivities_uses_requested_shift_sizes():
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))
    result = cmp.compute_python_sensitivities(
        _snapshot(),
        metric="CVA",
        factor_shifts={"zero:EUR:5Y": 0.0002},
    )
    assert len(result) == 1
    assert result[0].shift_size == 0.0002
    assert result[0].delta == pytest.approx(1000.0)


def test_compute_python_sensitivities_can_bump_hazard_via_survival_probability():
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))
    result = cmp.compute_python_sensitivities(
        _snapshot(),
        metric="CVA",
        factor_shifts={"hazard:CPTY_A:5Y": 0.0001},
        bump_modes={"hazard:CPTY_A:5Y": "survival_probability"},
    )
    assert len(result) == 1
    expected = 200.0 * (-1.0 / (5.0 * math.exp(-0.02 * 5.0)))
    assert result[0].delta == pytest.approx(expected, rel=1e-4)


def test_compute_python_sensitivities_can_emit_bump_change():
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))
    result = cmp.compute_python_sensitivities(
        _snapshot(),
        metric="CVA",
        factor_shifts={"zero:EUR:5Y": 0.0002},
        curve_factor_specs={
            "zero:EUR:5Y": {
                "kind": "discount",
                "ccy": "EUR",
                "target_time": 5.0,
                "node_times": [5.0],
            }
        },
        output_mode="bump_change",
    )
    assert len(result) == 1
    assert result[0].delta == pytest.approx(0.2)


def test_normalize_raw_discount_quote_key_uses_configured_discount_family():
    discount_family_by_ccy = {"EUR": "6M"}
    assert (
        normalize_raw_discount_quote_key("IR_SWAP/RATE/EUR/2D/6M/5Y", discount_family_by_ccy)
        == "zero:EUR:5Y"
    )
    assert normalize_raw_discount_quote_key("IR_SWAP/RATE/EUR/2D/1D/5Y", discount_family_by_ccy) is None


def test_skip_generic_zero_mapping_when_discount_family_is_not_ois():
    discount_family_by_ccy = {"EUR": "6M"}
    assert (
        _should_skip_generic_normalization(
            "IR_SWAP/RATE/EUR/2D/1D/5Y",
            "zero:EUR:5Y",
            discount_family_by_ccy,
        )
        is True
    )


def test_curve_factor_specs_from_ore_entries_groups_node_times():
    ore_entries = [
        ORESensitivityEntry(
            factor="DiscountCurve/EUR/0/1Y",
            normalized_factor="zero:EUR:1Y",
            shift_size=1.0e-4,
            base_xva=0.0,
            delta=0.0,
        ),
        ORESensitivityEntry(
            factor="DiscountCurve/EUR/1/5Y",
            normalized_factor="zero:EUR:5Y",
            shift_size=1.0e-4,
            base_xva=0.0,
            delta=0.0,
        ),
        ORESensitivityEntry(
            factor="IndexCurve/EUR-EURIBOR-6M/1/5Y",
            normalized_factor="fwd:EUR:6M:5Y",
            shift_size=1.0e-4,
            base_xva=0.0,
            delta=0.0,
        ),
    ]
    specs = _curve_factor_specs_from_ore_entries(ore_entries)
    assert specs["zero:EUR:5Y"]["kind"] == "discount"
    assert specs["zero:EUR:5Y"]["node_times"] == [1.0, 5.0]
    assert specs["fwd:EUR:6M:5Y"]["kind"] == "forward"
    assert specs["fwd:EUR:6M:5Y"]["index_tenor"] == "6M"


def test_curve_factor_specs_group_credit_nodes_by_counterparty():
    ore_entries = [
        ORESensitivityEntry(
            factor="SurvivalProbability/CPTY_A/0/1Y",
            normalized_factor="hazard:CPTY_A:1Y",
            shift_size=1.0e-4,
            base_xva=0.0,
            delta=0.0,
        ),
        ORESensitivityEntry(
            factor="SurvivalProbability/CPTY_A/1/5Y",
            normalized_factor="hazard:CPTY_A:5Y",
            shift_size=1.0e-4,
            base_xva=0.0,
            delta=0.0,
        ),
    ]
    specs = _curve_factor_specs_from_ore_entries(ore_entries)
    assert specs["hazard:CPTY_A:1Y"]["kind"] == "credit"
    assert specs["hazard:CPTY_A:1Y"]["node_times"] == [1.0, 5.0]
    assert specs["hazard:CPTY_A:5Y"]["node_times"] == [1.0, 5.0]


def test_bump_hazard_curve_quotes_by_survival_node_moves_adjacent_interval():
    quotes = [
        MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/1Y", value=0.02),
        MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.02),
        MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/7Y", value=0.02),
    ]
    up, down = _bump_hazard_curve_quotes_by_survival_node(quotes, 5.0, 1.0e-4)
    assert up[0] == pytest.approx(0.02)
    assert up[1] != pytest.approx(0.02)
    assert up[2] != pytest.approx(0.02)
    assert down[1] != pytest.approx(0.02)
