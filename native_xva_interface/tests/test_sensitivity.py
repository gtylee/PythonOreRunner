from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import pytest

from native_xva_interface import (
    FXForward,
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
    _build_credit_survival_curve_shock,
    _bump_hazard_curve_quotes_by_survival_node,
    _survival_from_piecewise_hazard,
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
        credit_curves = snapshot.config.params.get("python.credit_survival_curves", {})
        if isinstance(credit_curves, dict) and "CPTY_A" in credit_curves:
            hazard_term = 200.0 * _curve_survival_at(credit_curves["CPTY_A"], 5.0)
        else:
            hazard_term = 200.0 * math.exp(-5.0 * float(qmap.get("HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", 0.0)))
        cva = 1000.0 * (float(qmap.get("ZERO/RATE/EUR/5Y", 0.0)) + eur_zero_shift) + hazard_term
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


class _MultiFactorAdapter:
    def run(self, snapshot, mapped, run_id):
        qmap = {q.key: q.value for q in snapshot.market.raw_quotes}
        curve_shocks = snapshot.config.params.get("python.curve_node_shocks", {})
        discount = curve_shocks.get("discount", {}).get("EUR", {}) if isinstance(curve_shocks, dict) else {}
        forward = (
            curve_shocks.get("forward", {}).get("EUR", {}).get("6M", {}) if isinstance(curve_shocks, dict) else {}
        )
        eur_zero_5y_shift = 0.0
        eur_fwd_5y_shift = 0.0
        for t, s in zip(discount.get("node_times", []), discount.get("node_shifts", [])):
            if abs(float(t) - 5.0) < 1.0e-12:
                eur_zero_5y_shift = float(s)
                break
        for t, s in zip(forward.get("node_times", []), forward.get("node_shifts", [])):
            if abs(float(t) - 5.0) < 1.0e-12:
                eur_fwd_5y_shift = float(s)
                break
        cva = (
            1000.0 * (float(qmap.get("ZERO/RATE/EUR/5Y", 0.0)) + eur_zero_5y_shift)
            + 500.0 * float(qmap.get("FX/EUR/USD", 0.0))
            + 700.0 * eur_fwd_5y_shift
            + 200.0 * float(qmap.get("HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", 0.0))
            + 100.0 * float(qmap.get("RECOVERY_RATE/RATE/CPTY_A/SR/USD", 0.0))
        )
        return XVAResult(
            run_id=run_id,
            pv_total=0.0,
            xva_total=cva,
            xva_by_metric={"CVA": cva},
            exposure_by_netting_set={"CPTY_A": 1.0, "CPTY_B": 1.0},
            reports={},
            cubes={},
            metadata={"adapter": "multi-factor"},
        )


class _MatrixAdapter:
    def run(self, snapshot, mapped, run_id):
        qmap = {q.key: q.value for q in snapshot.market.raw_quotes}
        curve_shocks = snapshot.config.params.get("python.curve_node_shocks", {})

        def discount_shift(ccy: str, target_time: float) -> float:
            cfg = curve_shocks.get("discount", {}).get(ccy, {}) if isinstance(curve_shocks, dict) else {}
            for t, s in zip(cfg.get("node_times", []), cfg.get("node_shifts", [])):
                if abs(float(t) - target_time) < 1.0e-12:
                    return float(s)
            return 0.0

        def forward_shift(ccy: str, tenor: str, target_time: float) -> float:
            cfg = (
                curve_shocks.get("forward", {}).get(ccy, {}).get(tenor, {})
                if isinstance(curve_shocks, dict)
                else {}
            )
            for t, s in zip(cfg.get("node_times", []), cfg.get("node_shifts", [])):
                if abs(float(t) - target_time) < 1.0e-12:
                    return float(s)
            return 0.0

        cva = 0.0
        cva += 1000.0 * (float(qmap.get("ZERO/RATE/EUR/1Y", 0.0)) + discount_shift("EUR", 1.0))
        cva += 1500.0 * (float(qmap.get("ZERO/RATE/EUR/5Y", 0.0)) + discount_shift("EUR", 5.0))
        cva += 1100.0 * (float(qmap.get("ZERO/RATE/USD/2Y", 0.0)) + discount_shift("USD", 2.0))
        cva += 700.0 * forward_shift("EUR", "6M", 1.0)
        cva += 900.0 * forward_shift("EUR", "6M", 5.0)
        cva += 800.0 * forward_shift("USD", "3M", 2.0)
        cva += 500.0 * float(qmap.get("FX/EUR/USD", 0.0))
        cva += 600.0 * float(qmap.get("FX/GBP/USD", 0.0))
        cva += 200.0 * float(qmap.get("HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", 0.0))
        cva += 100.0 * float(qmap.get("RECOVERY_RATE/RATE/CPTY_A/SR/USD", 0.0))
        fva = 2.0 * cva
        return XVAResult(
            run_id=run_id,
            pv_total=0.0,
            xva_total=cva + fva,
            xva_by_metric={"CVA": cva, "FVA": fva},
            exposure_by_netting_set={"CPTY_A": 1.0, "CPTY_B": 1.0},
            reports={},
            cubes={},
            metadata={"adapter": "matrix"},
        )


class _LargePortfolioAdapter:
    def run(self, snapshot, mapped, run_id):
        qmap = {q.key: q.value for q in snapshot.market.raw_quotes}
        curve_shocks = snapshot.config.params.get("python.curve_node_shocks", {})
        credit_curves = snapshot.config.params.get("python.credit_survival_curves", {})

        def discount_shift(ccy: str, target_time: float) -> float:
            cfg = curve_shocks.get("discount", {}).get(ccy, {}) if isinstance(curve_shocks, dict) else {}
            for t, s in zip(cfg.get("node_times", []), cfg.get("node_shifts", [])):
                if abs(float(t) - target_time) < 1.0e-12:
                    return float(s)
            return 0.0

        def forward_shift(ccy: str, tenor: str, target_time: float) -> float:
            cfg = (
                curve_shocks.get("forward", {}).get(ccy, {}).get(tenor, {})
                if isinstance(curve_shocks, dict)
                else {}
            )
            for t, s in zip(cfg.get("node_times", []), cfg.get("node_shifts", [])):
                if abs(float(t) - target_time) < 1.0e-12:
                    return float(s)
            return 0.0

        cva = 0.0
        cva += 1300.0 * (float(qmap.get("ZERO/RATE/EUR/1Y", 0.0)) + discount_shift("EUR", 1.0))
        cva += 1700.0 * (float(qmap.get("ZERO/RATE/EUR/5Y", 0.0)) + discount_shift("EUR", 5.0))
        cva += 1600.0 * (float(qmap.get("ZERO/RATE/USD/2Y", 0.0)) + discount_shift("USD", 2.0))
        cva += 1400.0 * (float(qmap.get("ZERO/RATE/GBP/5Y", 0.0)) + discount_shift("GBP", 5.0))
        cva += 800.0 * forward_shift("EUR", "6M", 1.0)
        cva += 1100.0 * forward_shift("EUR", "6M", 5.0)
        cva += 900.0 * forward_shift("USD", "3M", 2.0)
        cva += 700.0 * forward_shift("GBP", "6M", 5.0)
        cva += 550.0 * float(qmap.get("FX/EUR/USD", 0.0))
        cva += 650.0 * float(qmap.get("FX/GBP/USD", 0.0))
        cva += 750.0 * float(qmap.get("FX/EUR/GBP", 0.0))
        cva += 220.0 * (
            _curve_survival_at(credit_curves["CPTY_A"], 5.0)
            if isinstance(credit_curves, dict) and "CPTY_A" in credit_curves
            else math.exp(-5.0 * float(qmap.get("HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", 0.0)))
        )
        cva += 180.0 * (
            _curve_survival_at(credit_curves["CPTY_B"], 5.0)
            if isinstance(credit_curves, dict) and "CPTY_B" in credit_curves
            else math.exp(-5.0 * float(qmap.get("HAZARD_RATE/RATE/CPTY_B/SR/USD/5Y", 0.0)))
        )
        cva += 90.0 * float(qmap.get("RECOVERY_RATE/RATE/CPTY_A/SR/USD", 0.0))
        cva += 70.0 * float(qmap.get("RECOVERY_RATE/RATE/CPTY_B/SR/USD", 0.0))
        return XVAResult(
            run_id=run_id,
            pv_total=0.0,
            xva_total=cva,
            xva_by_metric={"CVA": cva},
            exposure_by_netting_set={"CPTY_A": 1.0, "CPTY_B": 1.0, "CPTY_C": 1.0},
            reports={},
            cubes={},
            metadata={"adapter": "large-portfolio"},
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


def _curve_survival_at(cfg: Dict[str, object], t: float) -> float:
    times = [float(x) for x in cfg.get("node_times", [])]
    surv = [float(x) for x in cfg.get("survival_probabilities", [])]
    if not times or not surv or len(times) != len(surv):
        return 1.0
    tt = max(float(t), 0.0)
    if tt <= 1.0e-12:
        return 1.0
    logs = [math.log(max(x, 1.0e-18)) for x in surv]
    avg0 = -logs[0] / times[0]
    if tt <= times[0]:
        return math.exp(-avg0 * tt)
    if tt < times[-1]:
        return math.exp(float(np.interp(tt, np.asarray(times, dtype=float), np.asarray(logs, dtype=float))))
    avg_last = -logs[-1] / times[-1]
    return math.exp(-avg_last * tt)


def _multifactor_snapshot() -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/5Y", value=0.025),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/2D/6M/5Y", value=0.021),
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/1Y", value=0.02),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.02),
                MarketQuote(date="2026-03-08", key="RECOVERY_RATE/RATE/CPTY_A/SR/USD", value=0.40),
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


def _matrix_snapshot() -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/1Y", value=0.020),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/5Y", value=0.025),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/USD/2Y", value=0.030),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/2D/6M/1Y", value=0.015),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/2D/6M/5Y", value=0.021),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/USD/2D/3M/2Y", value=0.019),
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="FX/GBP/USD", value=1.25),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/1Y", value=0.02),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.02),
                MarketQuote(date="2026-03-08", key="RECOVERY_RATE/RATE/CPTY_A/SR/USD", value=0.40),
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
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", analytics=("CVA", "FVA"), num_paths=8),
    )


def _large_portfolio_snapshot() -> XVASnapshot:
    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/1Y", value=0.020),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/5Y", value=0.025),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/USD/2Y", value=0.030),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/GBP/5Y", value=0.028),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/2D/6M/1Y", value=0.015),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/2D/6M/5Y", value=0.021),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/USD/2D/3M/2Y", value=0.019),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/GBP/2D/6M/5Y", value=0.022),
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="FX/GBP/USD", value=1.25),
                MarketQuote(date="2026-03-08", key="FX/EUR/GBP", value=0.88),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/1Y", value=0.02),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.02),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_B/SR/USD/1Y", value=0.03),
                MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_B/SR/USD/5Y", value=0.03),
                MarketQuote(date="2026-03-08", key="RECOVERY_RATE/RATE/CPTY_A/SR/USD", value=0.40),
                MarketQuote(date="2026-03-08", key="RECOVERY_RATE/RATE/CPTY_B/SR/USD", value=0.35),
            ),
        ),
        fixings=FixingsData(),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS_EUR_1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=1_000_000, fixed_rate=0.020, maturity_years=5.0, pay_fixed=True),
                ),
                Trade(
                    trade_id="IRS_EUR_2",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=2_000_000, fixed_rate=0.018, maturity_years=7.0, pay_fixed=False),
                ),
                Trade(
                    trade_id="IRS_USD_1",
                    counterparty="CPTY_B",
                    netting_set="CPTY_B",
                    trade_type="Swap",
                    product=IRS(ccy="USD", notional=1_500_000, fixed_rate=0.024, maturity_years=2.0, pay_fixed=True),
                ),
                Trade(
                    trade_id="IRS_GBP_1",
                    counterparty="CPTY_C",
                    netting_set="CPTY_C",
                    trade_type="Swap",
                    product=IRS(ccy="GBP", notional=900_000, fixed_rate=0.023, maturity_years=5.0, pay_fixed=True),
                ),
                Trade(
                    trade_id="FXFWD_EURUSD_1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=500_000, strike=1.09, maturity_years=1.0),
                ),
                Trade(
                    trade_id="FXFWD_GBPUSD_1",
                    counterparty="CPTY_B",
                    netting_set="CPTY_B",
                    trade_type="FxForward",
                    product=FXForward(pair="GBPUSD", notional=400_000, strike=1.24, maturity_years=1.5),
                ),
            )
        ),
        config=XVAConfig(asof="2026-03-08", base_currency="EUR", analytics=("CVA",), num_paths=16),
    )


def test_compare_matches_ore_zero_sensitivity_rows(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
                [
                    "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                    "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
                    "CPTY_A,,false,SurvivalProbability/CPTY_A/1/5Y,0.0001,,0.0,EUR,0.0,-0.090461,#N/A",
                ]
            ),
        encoding="utf-8",
    )

    snap = replace(_snapshot(), config=replace(_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))
    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert len(result["comparisons"]) == 2
    assert result["unmatched_ore"] == []
    assert result["unmatched_python"] == []
    deltas = {row.normalized_factor: row.python_delta for row in result["comparisons"]}
    assert deltas["zero:EUR:5Y"] == pytest.approx(0.1)
    assert deltas["hazard:CPTY_A:5Y"] == pytest.approx(-0.090461, abs=1e-6)
    assert result["unsupported_factors"] == []


def test_compare_includes_credit_factors_in_parity_output(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
                "CPTY_A,,false,SurvivalProbability/CPTY_A/1/5Y,0.0001,,0.0,EUR,0.0,-0.090461,#N/A",
            ]
        ),
        encoding="utf-8",
    )
    snap = replace(_snapshot(), config=replace(_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))
    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert [c.normalized_factor for c in result["comparisons"]] == ["hazard:CPTY_A:5Y", "zero:EUR:5Y"]
    assert result["unsupported_factors"] == []
    assert result["notes"] == []


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
    assert result[0].delta == pytest.approx(200.0)


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


def test_build_credit_survival_curve_shock_matches_front_bucket_shape():
    quotes = [
        MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/1Y", value=0.02),
        MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.03),
    ]
    shocked = _build_credit_survival_curve_shock(
        quotes,
        coarse_node_times=[1.0, 3.0, 5.0],
        target_time=1.0,
        shift_size=1.0e-4,
    )

    assert shocked["node_times"] == [1.0, 3.0, 5.0]
    surv = shocked["survival_probabilities"]
    base_1y, base_3y, base_5y = _survival_from_piecewise_hazard(
        [1.0, 3.0, 5.0],
        [1.0, 5.0],
        [0.02, 0.03],
    )
    expected = [
        math.exp(-(-math.log(base_1y) / 1.0 + 1.0e-4) * 1.0),
        base_3y,
        base_5y,
    ]

    assert surv == pytest.approx(expected, abs=1e-12)


def test_build_credit_survival_curve_shock_matches_back_bucket_shape():
    quotes = [
        MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/1Y", value=0.02),
        MarketQuote(date="2026-03-08", key="HAZARD_RATE/RATE/CPTY_A/SR/USD/5Y", value=0.03),
    ]
    shocked = _build_credit_survival_curve_shock(
        quotes,
        coarse_node_times=[1.0, 3.0, 5.0],
        target_time=5.0,
        shift_size=1.0e-4,
    )

    surv = shocked["survival_probabilities"]
    base_1y, base_3y, base_5y = _survival_from_piecewise_hazard(
        [1.0, 3.0, 5.0],
        [1.0, 5.0],
        [0.02, 0.03],
    )
    expected = [
        base_1y,
        base_3y,
        math.exp(-(-math.log(base_5y) / 5.0 + 1.0e-4) * 5.0),
    ]

    assert surv == pytest.approx(expected, abs=1e-12)
    assert all(surv[i] >= surv[i + 1] for i in range(len(surv) - 1))


def test_compare_skips_unmapped_recovery_rows(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
                "CPTY_A,,false,RecoveryRate/CPTY_A,0.0001,,0.0,EUR,0.0,0.05,#N/A",
            ]
        ),
        encoding="utf-8",
    )
    snap = replace(_multifactor_snapshot(), config=replace(_multifactor_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_MultiFactorAdapter()))

    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert [c.normalized_factor for c in result["comparisons"]] == ["zero:EUR:5Y"]
    assert result["unsupported_factors"] == []
    assert result["unsupported_ore"] == []
    assert result["notes"] == []


def test_compare_reads_nested_xva_sensitivity_output_and_matches_fx_and_forward(tmp_path: Path):
    out_dir = tmp_path / "Output"
    nested = out_dir / "xva_sensitivity"
    nested.mkdir(parents=True)
    (nested / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
                "CPTY_A,,false,IndexCurve/EUR-EURIBOR-6M/1/5Y,0.0001,,0.0,EUR,0.0,0.07,#N/A",
                "CPTY_A,,false,FXSpot/EURUSD,0.01,,0.0,EUR,0.0,5.0,#N/A",
            ]
        ),
        encoding="utf-8",
    )

    snap = replace(_multifactor_snapshot(), config=replace(_multifactor_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_MultiFactorAdapter()))
    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert [c.normalized_factor for c in result["comparisons"]] == ["fwd:EUR:6M:5Y", "fx:EURUSD", "zero:EUR:5Y"]
    rows = {c.normalized_factor: c for c in result["comparisons"]}
    assert rows["zero:EUR:5Y"].python_delta == pytest.approx(0.1)
    assert rows["fwd:EUR:6M:5Y"].python_delta == pytest.approx(0.07)
    assert rows["fx:EURUSD"].python_delta == pytest.approx(5.0)


def test_load_ore_zero_sensitivities_filters_trade_rows_and_netting_set(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                "CPTY_A,IRS1,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,9.9,#N/A",
                "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
                "CPTY_B,,false,FXSpot/EURUSD,0.01,,0.0,EUR,0.0,5.5,#N/A",
            ]
        ),
        encoding="utf-8",
    )
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_MultiFactorAdapter()))

    rows_a = cmp.load_ore_zero_sensitivities(out_dir, metric="CVA", netting_set_id="CPTY_A")
    rows_b = cmp.load_ore_zero_sensitivities(out_dir, metric="CVA", netting_set_id="CPTY_B")

    assert [(r.netting_set_id, r.trade_id, r.normalized_factor, r.delta) for r in rows_a] == [
        ("CPTY_A", "", "zero:EUR:5Y", 0.1)
    ]
    assert [(r.netting_set_id, r.trade_id, r.normalized_factor, r.delta) for r in rows_b] == [
        ("CPTY_B", "", "fx:EURUSD", 5.5)
    ]


def test_compare_reports_multiple_unsupported_credit_factors_sorted(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                "CPTY_A,,false,SurvivalProbability/CPTY_A/0/1Y,0.0001,,0.0,EUR,0.0,-0.036193,#N/A",
                "CPTY_A,,false,SurvivalProbability/CPTY_A/1/5Y,0.0001,,0.0,EUR,0.0,-0.090461,#N/A",
                "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.1,#N/A",
            ]
        ),
        encoding="utf-8",
    )
    snap = replace(_multifactor_snapshot(), config=replace(_multifactor_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_MultiFactorAdapter()))
    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert [c.normalized_factor for c in result["comparisons"]] == ["hazard:CPTY_A:1Y", "hazard:CPTY_A:5Y", "zero:EUR:5Y"]
    assert result["unsupported_factors"] == []


def test_compare_without_ore_rows_returns_note_for_legacy_layout(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "cva_sensitivity_nettingset_CPTY_A.csv").write_text("header_only\n", encoding="utf-8")
    snap = replace(_snapshot(), config=replace(_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LinearQuoteAdapter()))

    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert result["python"] == []
    assert result["comparisons"] == []
    assert "Legacy file 'cva_sensitivity_nettingset_CPTY_A.csv' exists" in result["notes"][0]


@pytest.mark.parametrize(
    ("factor", "shift", "expected_delta"),
    [
        ("DiscountCurve/EUR/0/1Y", 1.0e-4, 0.1),
        ("DiscountCurve/EUR/1/5Y", 1.0e-4, 0.15),
        ("DiscountCurve/USD/0/2Y", 1.0e-4, 0.11),
        ("IndexCurve/EUR-EURIBOR-6M/0/1Y", 1.0e-4, 0.07),
        ("IndexCurve/EUR-EURIBOR-6M/1/5Y", 1.0e-4, 0.09),
        ("IndexCurve/USD-LIBOR-3M/0/2Y", 1.0e-4, 0.08),
        ("FXSpot/EURUSD", 0.01, 5.0),
        ("FXSpot/GBPUSD", 0.01, 6.0),
    ],
)
def test_compare_factor_matrix_cva(tmp_path: Path, factor: str, shift: float, expected_delta: float):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                f"CPTY_A,,false,{factor},{shift},,0.0,EUR,0.0,{expected_delta},#N/A",
            ]
        ),
        encoding="utf-8",
    )
    snap = replace(_matrix_snapshot(), config=replace(_matrix_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_MatrixAdapter()))

    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0].python_delta == pytest.approx(expected_delta)
    assert result["comparisons"][0].ore_delta == pytest.approx(expected_delta)
    assert result["unsupported_factors"] == []


@pytest.mark.parametrize(
    ("metric", "factor", "shift", "expected_delta"),
    [
        ("FVA", "DiscountCurve/EUR/1/5Y", 1.0e-4, 0.30),
        ("FVA", "IndexCurve/EUR-EURIBOR-6M/1/5Y", 1.0e-4, 0.18),
        ("FVA", "FXSpot/EURUSD", 0.01, 10.0),
    ],
)
def test_compare_metric_file_matrix(tmp_path: Path, metric: str, factor: str, shift: float, expected_delta: float):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / f"xva_zero_sensitivity_{metric.lower()}.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                f"CPTY_A,,false,{factor},{shift},,0.0,EUR,0.0,{expected_delta},#N/A",
            ]
        ),
        encoding="utf-8",
    )
    snap = replace(_matrix_snapshot(), config=replace(_matrix_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_MatrixAdapter()))

    result = cmp.compare(snap, metric=metric, output_dir=out_dir, netting_set_id="CPTY_A")

    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0].python_delta == pytest.approx(expected_delta)
    assert result["metric"] == metric


@pytest.mark.parametrize(
    ("netting_set", "expected_factor"),
    [
        ("CPTY_A", "zero:EUR:5Y"),
        ("CPTY_B", "fx:GBPUSD"),
    ],
)
def test_compare_netting_set_matrix(tmp_path: Path, netting_set: str, expected_factor: str):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            [
                "NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma",
                "CPTY_A,,false,DiscountCurve/EUR/1/5Y,0.0001,,0.0,EUR,0.0,0.15,#N/A",
                "CPTY_B,,false,FXSpot/GBPUSD,0.01,,0.0,EUR,0.0,6.0,#N/A",
            ]
        ),
        encoding="utf-8",
    )
    snap = replace(_matrix_snapshot(), config=replace(_matrix_snapshot().config, params={"outputPath": str(out_dir)}))
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_MatrixAdapter()))

    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id=netting_set)

    assert [c.normalized_factor for c in result["comparisons"]] == [expected_factor]


def test_compare_large_mixed_portfolio_matrix(tmp_path: Path):
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    rows = [
        ("DiscountCurve/EUR/0/1Y", 1.0e-4, 0.13),
        ("DiscountCurve/EUR/1/5Y", 1.0e-4, 0.17),
        ("DiscountCurve/USD/0/2Y", 1.0e-4, 0.16),
        ("DiscountCurve/GBP/1/5Y", 1.0e-4, 0.14),
        ("IndexCurve/EUR-EURIBOR-6M/0/1Y", 1.0e-4, 0.08),
        ("IndexCurve/EUR-EURIBOR-6M/1/5Y", 1.0e-4, 0.11),
        ("IndexCurve/USD-LIBOR-3M/0/2Y", 1.0e-4, 0.09),
        ("IndexCurve/GBP-LIBOR-6M/1/5Y", 1.0e-4, 0.07),
        ("FXSpot/EURUSD", 0.01, 5.5),
        ("FXSpot/GBPUSD", 0.01, 6.5),
        ("FXSpot/EURGBP", 0.01, 7.5),
            ("SurvivalProbability/CPTY_A/1/5Y", 1.0e-4, -0.099507),
            ("SurvivalProbability/CPTY_B/1/5Y", 1.0e-4, -0.078826),
    ]
    (out_dir / "xva_zero_sensitivity_cva.csv").write_text(
        "\n".join(
            ["NettingSetId,TradeId,IsPar,Factor_1,ShiftSize_1,Factor_2,ShiftSize_2,Currency,Base XVA,Delta,Gamma"]
            + [f"CPTY_A,,false,{factor},{shift},,0.0,EUR,0.0,{delta},#N/A" for factor, shift, delta in rows]
        ),
        encoding="utf-8",
    )

    snap = replace(
        _large_portfolio_snapshot(),
        config=replace(_large_portfolio_snapshot().config, params={"outputPath": str(out_dir)}),
    )
    cmp = OreSnapshotPythonLgmSensitivityComparator(engine=XVAEngine(adapter=_LargePortfolioAdapter()))

    result = cmp.compare(snap, metric="CVA", output_dir=out_dir, netting_set_id="CPTY_A")

    matched = {c.normalized_factor: c for c in result["comparisons"]}
    assert sorted(matched) == [
        "fwd:EUR:6M:1Y",
        "fwd:EUR:6M:5Y",
        "fwd:GBP:6M:5Y",
        "fwd:USD:3M:2Y",
        "fx:EURGBP",
        "fx:EURUSD",
        "fx:GBPUSD",
        "hazard:CPTY_A:5Y",
        "hazard:CPTY_B:5Y",
        "zero:EUR:1Y",
        "zero:EUR:5Y",
        "zero:GBP:5Y",
        "zero:USD:2Y",
    ]
    assert result["unsupported_factors"] == []
    assert matched["zero:EUR:1Y"].python_delta == pytest.approx(0.13)
    assert matched["zero:EUR:5Y"].python_delta == pytest.approx(0.17)
    assert matched["zero:USD:2Y"].python_delta == pytest.approx(0.16)
    assert matched["zero:GBP:5Y"].python_delta == pytest.approx(0.14)
    assert matched["fwd:EUR:6M:1Y"].python_delta == pytest.approx(0.08)
    assert matched["fwd:EUR:6M:5Y"].python_delta == pytest.approx(0.11)
    assert matched["fwd:USD:3M:2Y"].python_delta == pytest.approx(0.09)
    assert matched["fwd:GBP:6M:5Y"].python_delta == pytest.approx(0.07)
    assert matched["fx:EURUSD"].python_delta == pytest.approx(5.5)
    assert matched["fx:GBPUSD"].python_delta == pytest.approx(6.5)
    assert matched["fx:EURGBP"].python_delta == pytest.approx(7.5)
    assert matched["hazard:CPTY_A:5Y"].python_delta == pytest.approx(-0.099507, abs=1e-6)
    assert matched["hazard:CPTY_B:5Y"].python_delta == pytest.approx(-0.077444355175885, abs=1e-6)
    assert sum(c.python_delta for c in result["comparisons"]) == pytest.approx(20.273048407723763, abs=1e-6)
    assert sum(c.ore_delta for c in result["comparisons"]) == pytest.approx(20.271667000000004, abs=1e-6)
