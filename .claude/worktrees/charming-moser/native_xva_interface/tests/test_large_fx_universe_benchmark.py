from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from native_xva_interface.large_fx_universe_benchmark import (
    DEFAULT_CURRENCIES,
    build_fx_forward_trades,
    build_large_fx_universe_snapshot,
    build_parser,
    fair_forward_strike,
    ore_swig_available,
    run_benchmark,
)


def test_default_universe_returns_30_currencies_and_116_trades():
    snap = build_large_fx_universe_snapshot(num_paths=32, seed=7)

    assert len(DEFAULT_CURRENCIES) == 30
    assert snap.config.base_currency == "EUR"
    assert len(snap.portfolio.trades) == 116
    assert len({trade.trade_id for trade in snap.portfolio.trades}) == 116


def test_generated_maturities_are_short_dated():
    trades = build_fx_forward_trades("EUR", DEFAULT_CURRENCIES)

    maturities = sorted({trade.product.maturity_years for trade in trades})
    assert maturities == [
        pytest.approx(x)
        for x in (31.0 / 365.0, 61.0 / 365.0, 92.0 / 365.0, 153.0 / 365.0)
    ]
    assert all(trade.product.maturity_years < 0.5 for trade in trades)


def test_default_snapshot_is_collateralised_with_zero_threshold_and_small_mta():
    snap = build_large_fx_universe_snapshot(num_paths=32, seed=7)
    netting = snap.netting.netting_sets["NS1"]

    assert netting.active_csa is True
    assert netting.threshold_pay == 0.0
    assert netting.threshold_receive == 0.0
    assert netting.mta_pay == pytest.approx(10000.0)
    assert netting.mta_receive == pytest.approx(10000.0)
    assert snap.collateral.balances[0].netting_set_id == "NS1"


def test_generated_strikes_are_close_to_fair_forwards():
    trades = build_fx_forward_trades("EUR", ("EUR", "USD"), trades_per_ccy=4)

    for trade in trades:
        fair = fair_forward_strike("EUR", "USD", trade.product.maturity_years)
        rel_diff = abs(trade.product.strike - fair) / fair
        assert rel_diff <= 0.00051


def test_python_mpor_metadata_is_sticky_and_enabled():
    snap = build_large_fx_universe_snapshot(
        currencies=("EUR", "USD", "GBP"),
        trades_per_ccy=2,
        mpor_days=10,
        num_paths=24,
        seed=5,
    )
    result = run_benchmark(
        build_parser().parse_args(
            [
                "--engine",
                "python",
                "--currencies",
                "EUR,USD,GBP",
                "--trades-per-ccy",
                "2",
                "--py-paths",
                "24",
                "--mpor-days",
                "10",
                "--output-json",
                "unused.json",
            ]
        )
    )

    engine = result["engines"]["python"]
    assert snap.config.params["python.mpor_days"] == "10"
    assert engine["status"] == "ok"
    assert engine["metadata"]["mpor_enabled"] is True
    assert engine["metadata"]["mpor_days"] == 10
    assert engine["metadata"]["mpor_mode"] == "sticky"


def test_cli_parser_and_json_payload_shape(tmp_path: Path):
    output_json = tmp_path / "large_fx_report.json"
    args = build_parser().parse_args(
        [
            "--engine",
            "python",
            "--currencies",
            "EUR,USD,GBP",
            "--trades-per-ccy",
            "2",
            "--py-paths",
            "16",
            "--seed",
            "9",
            "--output-json",
            str(output_json),
        ]
    )

    assert args.engine == "python"
    assert args.py_paths == 16
    assert args.python_market_source == "synthetic"
    report = run_benchmark(args)
    output_json.write_text(json.dumps(report), encoding="utf-8")
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["config"]["base_ccy"] == "EUR"
    assert payload["trade_count"] == 4
    assert "python" in payload["engines"]
    assert "currencies" in payload
    assert "mpor" in payload


def test_both_engine_uses_ore_supported_currency_subset():
    args = build_parser().parse_args(
        [
            "--engine",
            "both",
            "--currencies",
            ",".join(DEFAULT_CURRENCIES),
            "--trades-per-ccy",
            "1",
            "--py-paths",
            "8",
            "--ore-paths",
            "8",
            "--output-json",
            "unused.json",
        ]
    )

    report = run_benchmark(args)

    assert report["currencies"] == ["EUR", "USD", "GBP", "JPY", "CHF"]
    assert report["trade_count"] == 4
    assert report["requested_currencies"][0] == "EUR"
    assert any("dropped" in warning for warning in report["warnings"])


def test_python_only_smoke_run_returns_non_empty_metrics():
    args = build_parser().parse_args(
        [
            "--engine",
            "python",
            "--currencies",
            "EUR,USD,GBP,JPY",
            "--trades-per-ccy",
            "1",
            "--py-paths",
            "12",
            "--seed",
            "4",
            "--output-json",
            "unused.json",
        ]
    )
    report = run_benchmark(args)
    engine = report["engines"]["python"]

    assert engine["status"] == "ok"
    assert np.isfinite(engine["pv_total"])
    assert np.isfinite(engine["xva_total"])
    assert set(engine["metrics"]) == {"CVA", "DVA", "FBA", "FCA", "FVA", "MVA"}
    assert all(np.isfinite(value) for value in engine["metrics"].values())


def test_python_ore_template_market_smoke_run_returns_non_empty_metrics():
    args = build_parser().parse_args(
        [
            "--engine",
            "python",
            "--currencies",
            "EUR,USD,GBP,JPY",
            "--trades-per-ccy",
            "1",
            "--py-paths",
            "12",
            "--python-market-source",
            "ore-template",
            "--seed",
            "4",
            "--output-json",
            "unused.json",
        ]
    )
    report = run_benchmark(args)
    engine = report["engines"]["python"]

    assert report["config"]["python_market_source"] == "ore-template"
    assert report["currencies"] == ["EUR", "USD", "GBP", "JPY"]
    assert engine["status"] == "ok"
    assert engine["metadata"]["adapter"] == "python-lgm-ore-template-market"
    assert np.isfinite(engine["pv_total"])
    assert np.isfinite(engine["xva_total"])


def test_ore_only_smoke_is_skipped_when_swig_unavailable():
    if not ore_swig_available():
        pytest.skip("ORE-SWIG unavailable in this environment")

    args = build_parser().parse_args(
        [
            "--engine",
            "ore",
            "--currencies",
            "EUR,USD,GBP",
            "--trades-per-ccy",
            "1",
            "--ore-paths",
            "8",
            "--seed",
            "3",
            "--output-json",
            "unused.json",
        ]
    )
    report = run_benchmark(args)

    assert "ore" in report["engines"]
    assert report["engines"]["ore"]["path_count"] == 8
