from __future__ import annotations

import json
import os
import re
import sys
import time
import csv
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_DEFAULT_MPLCONFIGDIR = Path("/tmp/codex-mplconfig")
_DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#1f2933",
    "blue": "#1d4ed8",
    "cyan": "#0891b2",
    "teal": "#0f766e",
    "gold": "#d97706",
    "rose": "#be123c",
    "slate": "#64748b",
    "cloud": "#e2e8f0",
    "mint": "#d1fae5",
    "sand": "#fef3c7",
}


def _is_pythonorerunner_root(path: Path) -> bool:
    return (
        (path / "notebook_series" / "series_helpers.py").exists()
        and ((path / "pythonore").exists() or (path / "src" / "pythonore").exists())
    )


def _is_engine_root(path: Path) -> bool:
    return (path / "Tools" / "PythonOreRunner" / "notebook_series" / "series_helpers.py").exists()


def _pythonorerunner_root(repo_root: Path) -> Path:
    if _is_pythonorerunner_root(repo_root):
        return repo_root
    if _is_engine_root(repo_root):
        return repo_root / "Tools" / "PythonOreRunner"
    raise RuntimeError(f"Could not resolve PythonOreRunner root from {repo_root}")


def _examples_root(repo_root: Path) -> Path:
    local_examples = _pythonorerunner_root(repo_root) / "Examples"
    if local_examples.exists():
        return local_examples
    engine_repo = _engine_repo_root(repo_root)
    return engine_repo / "Examples"


def _engine_repo_root(start: Path | None = None) -> Path:
    candidates: list[Path] = []
    env_root = os.getenv("ENGINE_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    if start is not None:
        current = Path(start).resolve()
        candidates.extend([current, *current.parents])
    candidates.extend(
        [
            Path("/Users/gordonlee/Documents/Engine"),
            Path(__file__).resolve().parents[2] / "Engine",
        ]
    )
    for candidate in candidates:
        if _is_engine_root(candidate) and (candidate / "Examples").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate an Engine checkout. Set ENGINE_REPO_ROOT to a local Engine repository."
    )


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if _is_pythonorerunner_root(candidate) or _is_engine_root(candidate):
            return candidate
    raise RuntimeError(
        "Could not locate a PythonOreRunner or Engine repo root from the current notebook path"
    )


def bootstrap_notebook_env(start: Path | None = None) -> Path:
    repo = find_repo_root(start)
    py_runner = _pythonorerunner_root(repo)
    notebook_dir = Path(__file__).resolve().parent
    src_root = py_runner / "src"
    engine_repo = _engine_repo_root(repo)
    swig_build = engine_repo / "ORE-SWIG" / "build" / "lib.macosx-10.13-universal2-cpython-313"
    os.environ.setdefault("MPLCONFIGDIR", str((py_runner / ".mplconfig").resolve()))
    os.chdir(repo)

    sys.path = [p for p in sys.path if not p.endswith("native_xva_interface")]
    for path in (str(src_root), str(notebook_dir), str(py_runner), str(swig_build)):
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)

    dyld_parts = [
        repo / "build" / "OREAnalytics" / "orea",
        repo / "build" / "OREData" / "ored",
        repo / "build" / "QuantExt" / "qle",
        repo / "build" / "QuantLib" / "ql",
    ]
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    merged = [str(p) for p in dyld_parts if p.exists()]
    if existing:
        merged.append(existing)
    if merged:
        os.environ["DYLD_LIBRARY_PATH"] = ":".join(merged)
    return repo


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfdff",
            "axes.edgecolor": PALETTE["cloud"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.grid": True,
            "grid.color": "#dbe4ee",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def swig_status() -> dict[str, Any]:
    bootstrap_notebook_env()
    try:
        import ORE  # type: ignore

        return {
            "available": True,
            "module_path": getattr(ORE, "__file__", "<builtin>"),
            "message": "ORE-SWIG is available in this kernel.",
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": False,
            "module_path": "",
            "message": f"ORE-SWIG unavailable: {exc}",
        }


def default_input_bundle(repo: Path | None = None) -> tuple[Path, str]:
    repo = repo or find_repo_root()
    examples = _examples_root(repo)
    xva = examples / "XvaRisk" / "Input"
    if (xva / "ore_stress_classic.xml").exists():
        return xva, "ore_stress_classic.xml"
    exposure = examples / "Exposure" / "Input"
    return exposure, "ore_swap.xml"


def default_curve_case_ore_xml(repo: Path | None = None) -> Path:
    repo = repo or find_repo_root()
    py_runner = _pythonorerunner_root(repo)
    return (
        py_runner
        / "parity_artifacts"
        / "multiccy_benchmark_final"
        / "cases"
        / "flat_USD_5Y_B"
        / "Input"
        / "ore.xml"
    )


def default_live_parity_ore_xml(repo: Path | None = None) -> Path:
    repo = repo or find_repo_root()
    return _examples_root(repo) / "Exposure" / "Input" / "ore_measure_lgm_fixed.xml"


def default_lgm_calibration_ore_xml(repo: Path | None = None) -> Path:
    repo = repo or find_repo_root()
    return _examples_root(repo) / "Exposure" / "Input" / "ore_measure_lgm_with_calibration.xml"


def locate_ore_exe(repo: Path | None = None) -> Path:
    repo = repo or find_repo_root()
    engine_repo = _engine_repo_root(repo)
    for candidate in (
        engine_repo / "build" / "App" / "ore",
        engine_repo / "build" / "ore" / "App" / "ore",
        engine_repo / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ORE executable not found under the local build tree")


def _normalize_ore_parameters(
    root: ET.Element,
    *,
    base_dir: Path,
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_relative_names = {
        "logFile",
        "outputFile",
        "outputFileName",
        "cubeFile",
        "aggregationScenarioDataFileName",
        "scenarioFile",
        "rawCubeOutputFile",
        "netCubeOutputFile",
    }
    extra_file_names = {"calendarAdjustment", "configFile"}
    for node in root.findall(".//Parameter"):
        name = node.attrib.get("name", "")
        text = (node.text or "").strip()
        if name == "inputPath":
            node.text = str(input_dir)
            continue
        if name == "outputPath":
            node.text = str(output_dir)
            continue
        if name in output_relative_names and text:
            node.text = Path(text).name
            continue
        if (name.endswith("File") or name in extra_file_names) and text:
            resolved = (base_dir / text).resolve() if not Path(text).is_absolute() else Path(text)
            node.text = str(resolved)
        elif name.endswith("Path") and name != "inputPath" and text:
            resolved = (base_dir / text).resolve() if not Path(text).is_absolute() else Path(text)
            node.text = str(resolved)


def run_ore_case_fresh(source_ore_xml: Path, *, label: str) -> tuple[Path, Path, dict[str, Any]]:
    repo = find_repo_root()
    ore_exe = locate_ore_exe(repo)
    source_ore_xml = Path(source_ore_xml).resolve()
    run_root = (
        repo
        / "Tools"
        / "PythonOreRunner"
        / "notebook_series"
        / "_fresh_runs"
        / f"{label}_{int(time.time() * 1000)}"
    )
    input_dir = run_root / "Input"
    output_dir = run_root / "Output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = ET.parse(source_ore_xml).getroot()
    _normalize_ore_parameters(root, base_dir=source_ore_xml.parent, input_dir=source_ore_xml.parent, output_dir=output_dir)

    fresh_ore_xml = input_dir / "ore.xml"
    ET.ElementTree(root).write(fresh_ore_xml, encoding="utf-8", xml_declaration=True)
    start = time.perf_counter()
    cp = subprocess.run(
        [str(ore_exe), str(fresh_ore_xml)],
        cwd=str(run_root),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    meta = {
        "run_root": str(run_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "ore_xml": str(fresh_ore_xml),
        "ore_exe": str(ore_exe),
        "elapsed_sec": elapsed,
        "returncode": int(cp.returncode),
        "stdout_tail": cp.stdout[-2000:],
        "stderr_tail": cp.stderr[-2000:],
    }
    if cp.returncode != 0:
        raise RuntimeError(
            f"Fresh ORE run failed for {source_ore_xml} with return code {cp.returncode}: "
            f"{meta['stderr_tail'] or meta['stdout_tail']}"
        )
    return fresh_ore_xml, output_dir, meta


def load_fresh_case_snapshots(case_name: str, *, label: str):
    bootstrap_notebook_env()
    from native_xva_interface import XVALoader
    from py_ore_tools.ore_snapshot import load_from_ore_xml

    source_ore_xml = aligned_case_dir(case_name) / "Input" / "ore.xml"
    fresh_ore_xml, output_dir, meta = run_ore_case_fresh(source_ore_xml, label=label)
    runtime_snapshot = XVALoader.from_files(str(fresh_ore_xml.parent), ore_file=fresh_ore_xml.name)
    ore_snapshot = load_from_ore_xml(fresh_ore_xml)
    meta["output_dir"] = str(output_dir)
    return runtime_snapshot, ore_snapshot, meta


def _case_setup_params(ore_xml: Path) -> dict[str, str]:
    root = ET.parse(ore_xml).getroot()
    return {
        node.attrib.get("name", ""): (node.text or "").strip()
        for node in root.findall("./Setup/Parameter")
        if node.attrib.get("name")
    }


def resolve_case_dirs(ore_xml: str | Path) -> tuple[Path, Path]:
    ore_xml = Path(ore_xml).resolve()
    setup_params = _case_setup_params(ore_xml)
    run_root = ore_xml.parent.parent
    input_dir = run_root / setup_params.get("inputPath", ore_xml.parent.name or "Input")
    output_dir = run_root / setup_params.get("outputPath", "Output")
    return input_dir.resolve(), output_dir.resolve()


def load_case_buffers(ore_xml: str | Path, *, include_output: bool = True) -> tuple[dict[str, str], dict[str, str], dict[str, object]]:
    ore_xml = Path(ore_xml).resolve()
    setup_params = _case_setup_params(ore_xml)
    input_dir, output_dir = resolve_case_dirs(ore_xml)
    input_files = {
        path.name: path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(input_dir.iterdir())
        if path.is_file()
    }
    file_ref_root = ore_xml.parent
    for key, text in setup_params.items():
        if not text:
            continue
        if not (key.endswith("File") or key in {"calendarAdjustment"}):
            continue
        ref_path = Path(text)
        if not ref_path.is_absolute():
            ref_path = (file_ref_root / ref_path).resolve()
        if ref_path.exists() and ref_path.is_file():
            input_files.setdefault(ref_path.name, ref_path.read_text(encoding="utf-8", errors="ignore"))
    if ore_xml.name not in input_files:
        input_files[ore_xml.name] = ore_xml.read_text(encoding="utf-8", errors="ignore")
    output_files: dict[str, str] = {}
    if include_output and output_dir.exists():
        output_files = {
            path.name: path.read_text(encoding="utf-8", errors="ignore")
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
    meta = {
        "ore_xml": str(ore_xml),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_file_count": len(input_files),
        "output_file_count": len(output_files),
    }
    return input_files, output_files, meta


def load_case_snapshots(ore_xml: str | Path):
    bootstrap_notebook_env()
    from native_xva_interface import XVALoader
    from py_ore_tools.ore_snapshot import load_from_ore_xml

    ore_xml = Path(ore_xml).resolve()
    input_dir, output_dir = resolve_case_dirs(ore_xml)
    runtime_snapshot = XVALoader.from_files(str(input_dir), ore_file=ore_xml.name)
    ore_snapshot = load_from_ore_xml(ore_xml)
    meta = {
        "ore_xml": str(ore_xml),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "trade_ids": [t.trade_id for t in runtime_snapshot.portfolio.trades],
        "base_currency": runtime_snapshot.config.base_currency,
    }
    return runtime_snapshot, ore_snapshot, meta


def run_ore_snapshot_app_case(
    ore_xml: str | Path,
    *,
    engine: str = "compare",
    price: bool = True,
    xva: bool = True,
    sensi: bool = False,
    paths: int = 500,
    seed: int = 42,
    rng: str = "ore_parity",
    ore_output_only: bool = False,
):
    bootstrap_notebook_env()
    from pythonore.workflows import OreSnapshotApp, PurePythonRunOptions

    ore_xml = Path(ore_xml).resolve()
    input_files, output_files, case_meta = load_case_buffers(ore_xml, include_output=engine in {"compare", "ore"})
    options = PurePythonRunOptions(
        engine=engine,
        price=price,
        xva=xva,
        sensi=sensi,
        paths=paths,
        seed=seed,
        rng=rng,
        ore_output_only=ore_output_only,
    )
    app = OreSnapshotApp.from_strings(
        input_files=input_files,
        output_files=output_files,
        ore_xml_name=ore_xml.name,
        options=options,
    )
    start = time.perf_counter()
    result = app.run()
    elapsed = time.perf_counter() - start
    meta = {
        **case_meta,
        "engine": engine,
        "paths": paths,
        "seed": seed,
        "rng": rng,
        "elapsed_sec": elapsed,
        "comparison_rows": len(result.comparison_rows),
        "input_validation_rows": len(result.input_validation_rows),
        "ore_output_files": sorted(result.ore_output_files.keys()),
    }
    return result, meta


def ore_snapshot_app_summary_frame(result) -> pd.DataFrame:
    summary = dict(result.summary)
    pricing = summary.get("pricing") or {}
    xva = summary.get("xva") or {}
    diagnostics = summary.get("diagnostics") or {}
    rows = [
        {"field": "trade_id", "value": summary.get("trade_id", "")},
        {"field": "counterparty", "value": summary.get("counterparty", "")},
        {"field": "netting_set_id", "value": summary.get("netting_set_id", "")},
        {"field": "paths", "value": summary.get("paths", "")},
        {"field": "seed", "value": summary.get("seed", "")},
        {"field": "rng_mode", "value": summary.get("rng_mode", "")},
        {"field": "py_t0_npv", "value": pricing.get("py_t0_npv", np.nan)},
        {"field": "ore_t0_npv", "value": pricing.get("ore_t0_npv", np.nan)},
        {"field": "py_cva", "value": xva.get("py_cva", np.nan)},
        {"field": "ore_cva", "value": xva.get("ore_cva", np.nan)},
        {"field": "report_bucket", "value": diagnostics.get("report_bucket", "")},
    ]
    return pd.DataFrame(rows)


def ore_snapshot_app_metric_frame(result) -> pd.DataFrame:
    summary = dict(result.summary)
    pricing = summary.get("pricing") or {}
    xva = summary.get("xva") or {}
    rows = [
        {
            "metric": "PV",
            "python_lgm": float(pricing.get("py_t0_npv", np.nan)),
            "ore_output": float(pricing.get("ore_t0_npv", np.nan)),
        },
        {
            "metric": "CVA",
            "python_lgm": float(xva.get("py_cva", np.nan)),
            "ore_output": float(xva.get("ore_cva", np.nan)),
        },
        {
            "metric": "DVA",
            "python_lgm": float(xva.get("py_dva", np.nan)),
            "ore_output": float(xva.get("ore_dva", np.nan)),
        },
        {
            "metric": "FBA",
            "python_lgm": float(xva.get("py_fba", np.nan)),
            "ore_output": float(xva.get("ore_fba", np.nan)),
        },
        {
            "metric": "FCA",
            "python_lgm": float(xva.get("py_fca", np.nan)),
            "ore_output": float(xva.get("ore_fca", np.nan)),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["delta"] = frame["python_lgm"] - frame["ore_output"]
    frame["abs_rel_diff"] = frame.apply(
        lambda row: np.nan if pd.isna(row["ore_output"]) else abs(float(row["delta"])) / max(abs(float(row["ore_output"])), 1.0),
        axis=1,
    )
    return frame


def make_programmatic_snapshot(num_paths: int = 128):
    bootstrap_notebook_env()
    from native_xva_interface import (
        CollateralBalance,
        CollateralConfig,
        FixingPoint,
        FixingsData,
        FXForward,
        IRS,
        MarketData,
        MarketQuote,
        NettingConfig,
        NettingSet,
        Portfolio,
        Trade,
        XVAConfig,
        XVASnapshot,
    )

    return XVASnapshot(
        market=MarketData(
            asof="2026-03-08",
            raw_quotes=(
                MarketQuote(date="2026-03-08", key="FX/EUR/USD", value=1.10),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/EUR/1Y", value=0.0210),
                MarketQuote(date="2026-03-08", key="ZERO/RATE/USD/1Y", value=0.0315),
                MarketQuote(date="2026-03-08", key="IR_SWAP/RATE/EUR/EUR-ESTR/1Y/5Y", value=0.0230),
            ),
        ),
        fixings=FixingsData(
            points=(
                FixingPoint(date="2026-03-06", index="EUR-ESTR", value=0.0190),
                FixingPoint(date="2026-03-07", index="EUR-ESTR", value=0.0191),
            )
        ),
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS_DEMO_1",
                    counterparty="CP_A",
                    netting_set="NS_EUR",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=5_000_000, fixed_rate=0.024, maturity_years=5.0, pay_fixed=True),
                ),
                Trade(
                    trade_id="FXFWD_DEMO_1",
                    counterparty="CP_A",
                    netting_set="NS_EUR",
                    trade_type="FxForward",
                    product=FXForward(pair="EURUSD", notional=2_000_000, strike=1.11, maturity_years=1.0, buy_base=True),
                ),
            )
        ),
        netting=NettingConfig(
            netting_sets={
                "NS_EUR": NettingSet(
                    netting_set_id="NS_EUR",
                    counterparty="CP_A",
                    active_csa=True,
                    csa_currency="EUR",
                    threshold_pay=0.0,
                    threshold_receive=0.0,
                    mta_pay=100_000.0,
                    mta_receive=100_000.0,
                )
            }
        ),
        collateral=CollateralConfig(
            balances=(
                CollateralBalance(
                    netting_set_id="NS_EUR",
                    currency="EUR",
                    initial_margin=250_000.0,
                    variation_margin=50_000.0,
                ),
            )
        ),
        config=XVAConfig(
            asof="2026-03-08",
            base_currency="EUR",
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=num_paths,
            horizon_years=5,
        ),
    )


def load_base_snapshot(num_paths: int = 256):
    bootstrap_notebook_env()
    from native_xva_interface import XVAConfig, XVALoader, XVASnapshot

    input_dir, ore_file = default_input_bundle()
    snap = XVALoader.from_files(str(input_dir), ore_file=ore_file)
    cfg = XVAConfig(
        asof=snap.config.asof,
        base_currency=snap.config.base_currency,
        analytics=snap.config.analytics,
        num_paths=num_paths,
        horizon_years=snap.config.horizon_years,
        params=dict(snap.config.params),
        xml_buffers=dict(snap.config.xml_buffers),
        runtime=snap.config.runtime,
        source_meta=snap.config.source_meta,
    )
    return XVASnapshot(
        market=snap.market,
        fixings=snap.fixings,
        portfolio=snap.portfolio,
        config=cfg,
        netting=snap.netting,
        collateral=snap.collateral,
        source_meta=dict(snap.source_meta),
    )


def make_joint_demo_snapshot(num_paths: int = 256):
    bootstrap_notebook_env()
    from native_xva_interface import IRS, Portfolio, Trade, XVASnapshot

    base = load_base_snapshot(num_paths=num_paths)
    portfolio = Portfolio(
        trades=(
            Trade(
                trade_id="JOINT_IRS_1",
                counterparty="CPY_P",
                netting_set="CPY_P",
                trade_type="Swap",
                product=IRS(ccy=base.config.base_currency, notional=10_000_000, fixed_rate=0.025, maturity_years=5.0),
            ),
        )
    )
    netting_sets = dict(base.netting.netting_sets)
    if "CPY_P" not in netting_sets:
        sample = next(iter(netting_sets.values()), None)
        if sample is not None:
            netting_sets["CPY_P"] = type(sample)(
                netting_set_id="CPY_P",
                counterparty="CPY_P",
                active_csa=sample.active_csa,
                csa_currency=sample.csa_currency,
                threshold_pay=sample.threshold_pay,
                threshold_receive=sample.threshold_receive,
                mta_pay=sample.mta_pay,
                mta_receive=sample.mta_receive,
            )
    return XVASnapshot(
        market=base.market,
        fixings=base.fixings,
        portfolio=portfolio,
        config=base.config,
        netting=type(base.netting)(netting_sets=netting_sets, source_meta=base.netting.source_meta),
        collateral=base.collateral,
        source_meta=dict(base.source_meta),
    )


def snapshot_overview(snapshot) -> pd.DataFrame:
    rows = [
        {"component": "market quotes", "count": len(snapshot.market.raw_quotes), "notes": snapshot.market.asof},
        {"component": "fixings", "count": len(snapshot.fixings.points), "notes": snapshot.fixings.source_meta.origin},
        {"component": "trades", "count": len(snapshot.portfolio.trades), "notes": snapshot.portfolio.source_meta.origin},
        {"component": "netting sets", "count": len(snapshot.netting.netting_sets), "notes": snapshot.config.base_currency},
        {"component": "collateral balances", "count": len(snapshot.collateral.balances), "notes": snapshot.config.asof},
        {"component": "analytics", "count": len(snapshot.config.analytics), "notes": ",".join(snapshot.config.analytics)},
    ]
    return pd.DataFrame(rows)


def trade_frame(snapshot) -> pd.DataFrame:
    rows = []
    for trade in snapshot.portfolio.trades:
        product = trade.product
        rows.append(
            {
                "trade_id": trade.trade_id,
                "trade_type": trade.trade_type,
                "counterparty": trade.counterparty,
                "netting_set": trade.netting_set,
                "product_type": getattr(product, "product_type", type(product).__name__),
                "currency_or_pair": getattr(product, "ccy", getattr(product, "pair", "")),
                "notional": getattr(product, "notional", np.nan),
                "maturity_years": getattr(product, "maturity_years", np.nan),
            }
        )
    return pd.DataFrame(rows)


def quote_family_frame(snapshot) -> pd.DataFrame:
    families: dict[str, int] = {}
    for quote in snapshot.market.raw_quotes:
        family = quote.key.split("/", 1)[0]
        families[family] = families.get(family, 0) + 1
    return pd.DataFrame(
        [{"family": family, "count": count} for family, count in sorted(families.items())]
    )


def netting_frame(snapshot) -> pd.DataFrame:
    rows = []
    for ns_id, ns in snapshot.netting.netting_sets.items():
        rows.append(
            {
                "netting_set_id": ns_id,
                "counterparty": ns.counterparty,
                "active_csa": ns.active_csa,
                "csa_currency": ns.csa_currency,
                "threshold_pay": ns.threshold_pay,
                "threshold_receive": ns.threshold_receive,
            }
        )
    return pd.DataFrame(rows)


def collateral_frame(snapshot) -> pd.DataFrame:
    rows = []
    for bal in snapshot.collateral.balances:
        rows.append(
            {
                "netting_set_id": bal.netting_set_id,
                "currency": bal.currency,
                "initial_margin": bal.initial_margin,
                "variation_margin": bal.variation_margin,
            }
        )
    return pd.DataFrame(rows)


def mapped_input_summary(mapped) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "asof", "value": mapped.asof},
            {"field": "base_currency", "value": mapped.base_currency},
            {"field": "analytics", "value": ",".join(mapped.analytics)},
            {"field": "market_data_lines", "value": len(mapped.market_data_lines)},
            {"field": "fixing_data_lines", "value": len(mapped.fixing_data_lines)},
            {"field": "xml_buffers", "value": len(mapped.xml_buffers)},
        ]
    )


def xml_buffer_summary(mapped) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "xml_name": name,
                "chars": len(text),
                "root_tag": _root_tag(text),
            }
            for name, text in sorted(mapped.xml_buffers.items())
        ]
    )


def result_metrics_frame(result) -> pd.DataFrame:
    rows = [{"metric": "PV", "value": float(result.pv_total)}]
    rows.extend({"metric": metric, "value": float(value)} for metric, value in sorted(result.xva_by_metric.items()))
    rows.append({"metric": "XVA_TOTAL", "value": float(result.xva_total)})
    return pd.DataFrame(rows)


def compare_results_frame(lhs_name: str, lhs_result, rhs_name: str, rhs_result) -> pd.DataFrame:
    metrics = sorted(set(lhs_result.xva_by_metric) | set(rhs_result.xva_by_metric) | {"PV", "XVA_TOTAL"})
    rows = []
    for metric in metrics:
        if metric == "PV":
            lhs = float(lhs_result.pv_total)
            rhs = float(rhs_result.pv_total)
        elif metric == "XVA_TOTAL":
            lhs = float(lhs_result.xva_total)
            rhs = float(rhs_result.xva_total)
        else:
            lhs = float(lhs_result.xva_by_metric.get(metric, 0.0))
            rhs = float(rhs_result.xva_by_metric.get(metric, 0.0))
        rows.append(
            {
                "metric": metric,
                lhs_name: lhs,
                rhs_name: rhs,
                "delta": rhs - lhs,
                "abs_delta": abs(rhs - lhs),
            }
        )
    return pd.DataFrame(rows)


def capability_matrix_frame(swig_available: bool) -> pd.DataFrame:
    rows = [
        {"capability": "Programmatic dataclass snapshot", "python": True, "ore_swig": swig_available},
        {"capability": "Loader from real ORE input bundle", "python": True, "ore_swig": swig_available},
        {"capability": "Curve extraction and fit visuals", "python": True, "ore_swig": False},
        {"capability": "Python LGM path simulation", "python": True, "ore_swig": False},
        {"capability": "Non-zero ORE XVA on known-good configs", "python": False, "ore_swig": swig_available},
    ]
    return pd.DataFrame(rows)


def aligned_case_dir(case_name: str = "flat_EUR_10Y_B") -> Path:
    repo = find_repo_root()
    return _pythonorerunner_root(repo) / "parity_artifacts" / "multiccy_benchmark_final" / "cases" / case_name


def bermudan_compare_root(repo: Path | None = None) -> Path:
    repo = repo or find_repo_root()
    return _pythonorerunner_root(repo) / "parity_artifacts" / "bermudan_method_compare"


def load_bermudan_method_comparison() -> pd.DataFrame:
    path = bermudan_compare_root() / "comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Bermudan comparison csv not found: {path}")
    frame = pd.read_csv(path)
    for col in ("py_lsmc", "py_backward", "ore_classic", "ore_amc"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if {"py_lsmc", "ore_classic"}.issubset(frame.columns):
        frame["py_lsmc_minus_ore_classic"] = frame["py_lsmc"] - frame["ore_classic"]
    if {"py_backward", "ore_classic"}.issubset(frame.columns):
        frame["py_backward_minus_ore_classic"] = frame["py_backward"] - frame["ore_classic"]
        frame["py_backward_abs_rel_diff"] = (
            frame["py_backward_minus_ore_classic"].abs() / frame["ore_classic"].abs().clip(lower=1.0)
        )
    return frame.sort_values("fixed_rate").reset_index(drop=True)


def _load_trade_npv_csv(npv_csv: Path, trade_id: str) -> float:
    with npv_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("TradeId", row.get("#TradeId", ""))
            if str(key).strip() == str(trade_id):
                for candidate in ("NPV(Base)", "NPV", "Base NPV"):
                    if candidate in row and str(row[candidate]).strip():
                        return float(row[candidate])
    raise ValueError(f"Trade '{trade_id}' not found in {npv_csv}")


def _load_trade_pricing_stats(pricingstats_csv: Path, trade_id: str) -> dict[str, float]:
    with pricingstats_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("TradeId", row.get("#TradeId", ""))
            if str(key).strip() == str(trade_id):
                return {
                    "number_of_pricings": float(row.get("NumberOfPricings", 0.0) or 0.0),
                    "cumulative_timing_us": float(row.get("CumulativeTiming", 0.0) or 0.0),
                    "average_timing_us": float(row.get("AverageTiming", 0.0) or 0.0),
                }
    raise ValueError(f"Trade '{trade_id}' not found in {pricingstats_csv}")


def _load_runtime_totals(runtimes_csv: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    with runtimes_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get("#Key", row.get("Key", ""))).strip()
            if key:
                totals[key] = float(row.get("Total", 0.0) or 0.0) / 1.0e6
    return totals


def run_bermudan_case_summary(case_name: str = "berm_200bp", *, trade_id: str = "BermSwp", num_paths: int = 4096, seed: int = 42):
    bootstrap_notebook_env()
    from native_xva_interface import price_bermudan_from_ore_case

    case_root = bermudan_compare_root() / case_name
    input_dir = case_root / "Input"
    output_dir = case_root / "Output" / "classic"
    if not input_dir.exists():
        raise FileNotFoundError(f"Bermudan case input not found: {input_dir}")
    ore_price = _load_trade_npv_csv(output_dir / "npv.csv", trade_id)
    ore_trade_stats = _load_trade_pricing_stats(output_dir / "pricingstats.csv", trade_id)
    ore_runtime_totals = _load_runtime_totals(output_dir / "runtimes.csv")

    t0 = time.perf_counter()
    lsmc = price_bermudan_from_ore_case(
        input_dir,
        ore_file="ore_classic.xml",
        trade_id=trade_id,
        method="lsmc",
        num_paths=num_paths,
        seed=seed,
        curve_mode="market_fit",
    )
    lsmc_elapsed = time.perf_counter() - t0
    t0 = time.perf_counter()
    backward = price_bermudan_from_ore_case(
        input_dir,
        ore_file="ore_classic.xml",
        trade_id=trade_id,
        method="backward",
        num_paths=num_paths,
        seed=seed,
        curve_mode="market_fit",
    )
    backward_elapsed = time.perf_counter() - t0

    summary = pd.DataFrame(
        [
            {
                "method": "py_lsmc",
                "price": float(lsmc.price),
                "ore_classic": float(ore_price),
                "delta_vs_ore": float(lsmc.price - ore_price),
                "abs_rel_diff": abs(float(lsmc.price - ore_price)) / max(abs(float(ore_price)), 1.0),
                "model_param_source": str(lsmc.model_param_source),
                "curve_source": str(lsmc.curve_source),
            },
            {
                "method": "py_backward",
                "price": float(backward.price),
                "ore_classic": float(ore_price),
                "delta_vs_ore": float(backward.price - ore_price),
                "abs_rel_diff": abs(float(backward.price - ore_price)) / max(abs(float(ore_price)), 1.0),
                "model_param_source": str(backward.model_param_source),
                "curve_source": str(backward.curve_source),
            },
        ]
    )

    speed = pd.DataFrame(
        [
            {
                "engine": "py_lsmc",
                "elapsed_sec": float(lsmc_elapsed),
                "timing_source": "wall_clock_notebook",
            },
            {
                "engine": "py_backward",
                "elapsed_sec": float(backward_elapsed),
                "timing_source": "wall_clock_notebook",
            },
            {
                "engine": "ore_classic_trade_avg",
                "elapsed_sec": float(ore_trade_stats["average_timing_us"]) / 1.0e6,
                "timing_source": "pricingstats_average",
            },
            {
                "engine": "ore_classic_trade_cumulative",
                "elapsed_sec": float(ore_trade_stats["cumulative_timing_us"]) / 1.0e6,
                "timing_source": "pricingstats_cumulative",
            },
            {
                "engine": "ore_classic_xva_total",
                "elapsed_sec": float(ore_runtime_totals.get("XVA|Run XVAAnalytic", np.nan)),
                "timing_source": "runtimes_total",
            },
        ]
    )
    speed["speed_ratio_vs_py_backward"] = speed["elapsed_sec"] / max(float(backward_elapsed), 1.0e-12)

    diag_rows: list[dict[str, Any]] = []
    for method_name, result in (("py_lsmc", lsmc), ("py_backward", backward)):
        for diag in result.exercise_diagnostics:
            diag_rows.append(
                {
                    "method": method_name,
                    "time": float(diag.time),
                    "intrinsic_mean": float(diag.intrinsic_mean),
                    "continuation_mean": float(diag.continuation_mean),
                    "exercise_probability": float(diag.exercise_probability),
                    "boundary_state": np.nan if diag.boundary_state is None else float(diag.boundary_state),
                }
            )
    diagnostics = pd.DataFrame(diag_rows).sort_values(["method", "time"]).reset_index(drop=True)
    meta = {
        "case_name": case_name,
        "trade_id": trade_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "ore_npv_csv": str(output_dir / "npv.csv"),
        "num_paths": int(num_paths),
        "seed": int(seed),
    }
    return summary, diagnostics, speed, meta


def run_bermudan_sensitivity_comparison(
    case_name: str = "berm_200bp",
    *,
    method: str = "backward",
    num_paths: int = 256,
    seed: int = 17,
    shift_size: float = 1.0e-4,
):
    repo = bootstrap_notebook_env()
    py_runner = _pythonorerunner_root(repo)
    compare_script = py_runner / "scripts" / "compare" / "compare_bermudan_ore_sensitivities.py"
    output_root = py_runner / "parity_artifacts" / "bermudan_sensitivity_compare"
    cmd = [
        sys.executable,
        str(compare_script),
        "--case-name",
        case_name,
        "--method",
        method,
        "--num-paths",
        str(num_paths),
        "--seed",
        str(seed),
        "--shift-size",
        str(shift_size),
        "--output-root",
        str(output_root),
    ]
    start = time.perf_counter()
    cp = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if cp.returncode != 0:
        raise RuntimeError(
            f"Bermudan sensitivity comparison run failed with return code {cp.returncode}: "
            f"{cp.stderr[-2000:] or cp.stdout[-2000:]}"
        )
    json_path = output_root / f"{case_name}_{method}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload.get("rows", []))
    meta = pd.DataFrame(
        [
            {
                "case_name": payload.get("case_name", case_name),
                "trade_id": payload.get("trade_id", "BermSwp"),
                "method": payload.get("method", method),
                "shift_size": payload.get("shift_size", shift_size),
                "python_price": payload.get("python_price"),
                "ore_price": payload.get("ore_price"),
                "price_diff": payload.get("price_diff"),
                "ore_run_root": payload.get("ore_run_root"),
                "elapsed_sec": elapsed,
            }
        ]
    )
    ore_run_root = Path(payload["ore_run_root"])
    sensitivity_config = pd.read_csv(ore_run_root / "Output" / "sensitivity_config.csv")
    scenario = pd.read_csv(ore_run_root / "Output" / "scenario.csv")
    return payload, meta, rows, sensitivity_config, scenario


def run_aligned_case_compare(case_name: str = "flat_EUR_10Y_B", *, paths: int = 2000):
    bootstrap_notebook_env()
    from dataclasses import replace
    from native_xva_interface import XVAEngine, PythonLgmAdapter

    snap, ore_snapshot, run_meta = load_fresh_case_snapshots(case_name, label=f"notebook_05_{case_name}")
    fresh_ore_xml = Path(run_meta["ore_xml"])
    output_dir = Path(run_meta["output_dir"])
    source_case_dir = aligned_case_dir(case_name)
    fresh_run_root = fresh_ore_xml.parent.parent
    requested_metrics = tuple(snap.config.analytics)
    snap = replace(
        snap,
        config=replace(
            snap.config,
            num_paths=paths,
            analytics=("CVA", "DVA", "FVA", "MVA"),
            params={**snap.config.params, "python.lgm_rng_mode": "ore_parity"},
        ),
    )
    py_result, py_elapsed = run_adapter(snap, PythonLgmAdapter(fallback_to_swig=False))
    ore_xva = _read_ore_xva_row(output_dir / "xva.csv")
    ore_pv = _read_ore_npv_row(output_dir / "npv.csv")
    comparison = pd.DataFrame(
        [
            {"metric": "PV", "python_lgm": float(py_result.pv_total), "ore_output": float(ore_pv)},
            {"metric": "CVA", "python_lgm": float(py_result.xva_by_metric.get("CVA", 0.0)), "ore_output": float(ore_xva.get("CVA", 0.0))},
            {"metric": "DVA", "python_lgm": float(py_result.xva_by_metric.get("DVA", 0.0)), "ore_output": float(ore_xva.get("DVA", 0.0))},
            {"metric": "FBA", "python_lgm": float(py_result.xva_by_metric.get("FBA", 0.0)), "ore_output": float(ore_xva.get("FBA", 0.0))},
            {"metric": "FCA", "python_lgm": float(py_result.xva_by_metric.get("FCA", 0.0)), "ore_output": float(ore_xva.get("FCA", 0.0))},
        ]
    )
    if "FVA" not in requested_metrics:
        comparison.loc[comparison["metric"].isin(["FBA", "FCA"]), "ore_output"] = np.nan
    comparison["delta"] = comparison["python_lgm"] - comparison["ore_output"]
    comparison["abs_rel_diff"] = comparison.apply(
        lambda row: np.nan if pd.isna(row["ore_output"]) else abs(float(row["delta"])) / max(abs(float(row["ore_output"])), 1.0),
        axis=1,
    )
    meta = {
        "case_name": case_name,
        "source_case_dir": str(source_case_dir),
        "fresh_run_root": str(fresh_run_root),
        "fresh_ore_xml": str(fresh_ore_xml),
        "fresh_output_dir": str(output_dir),
        "requested_metrics_in_case": requested_metrics,
        "python_elapsed_sec": py_elapsed,
        "ore_elapsed_sec": float(run_meta.get("elapsed_sec", 0.0)),
        "trade_count": len(snap.portfolio.trades),
        "trade_ids": [t.trade_id for t in snap.portfolio.trades],
        "base_currency": snap.config.base_currency,
    }
    meta.update({f"run_{k}": v for k, v in run_meta.items() if k not in ("output_dir", "ore_xml")})
    return py_result, comparison, meta, ore_snapshot


def run_live_measure_lgm_compare(*, paths: int = 1000):
    bootstrap_notebook_env()
    from dataclasses import replace
    from native_xva_interface import XVALoader, PythonLgmAdapter

    source_ore_xml = default_live_parity_ore_xml()
    fresh_ore_xml, output_dir, run_meta = run_ore_case_fresh(source_ore_xml, label="notebook_05_measure_lgm_fixed")
    snap = XVALoader.from_files(str(fresh_ore_xml.parent), ore_file=fresh_ore_xml.name)
    requested_metrics = tuple(snap.config.analytics)
    snap = replace(
        snap,
        config=replace(
            snap.config,
            analytics=("CVA", "DVA", "FVA", "MVA"),
            num_paths=paths,
            params={**snap.config.params, "python.lgm_rng_mode": "ore_parity"},
        ),
    )
    py_result, py_elapsed = run_adapter(snap, PythonLgmAdapter(fallback_to_swig=False))
    ore_xva = _read_ore_xva_row(output_dir / "xva.csv")
    ore_pv = _read_ore_npv_row(output_dir / "npv.csv")
    comparison = pd.DataFrame(
        [
            {"metric": "PV", "python_lgm": float(py_result.pv_total), "ore_output": float(ore_pv)},
            {"metric": "CVA", "python_lgm": float(py_result.xva_by_metric.get("CVA", 0.0)), "ore_output": float(ore_xva.get("CVA", 0.0))},
            {"metric": "DVA", "python_lgm": float(py_result.xva_by_metric.get("DVA", 0.0)), "ore_output": float(ore_xva.get("DVA", 0.0))},
            {"metric": "FBA", "python_lgm": float(py_result.xva_by_metric.get("FBA", 0.0)), "ore_output": float(ore_xva.get("FBA", 0.0))},
            {"metric": "FCA", "python_lgm": float(py_result.xva_by_metric.get("FCA", 0.0)), "ore_output": float(ore_xva.get("FCA", 0.0))},
            {"metric": "MVA", "python_lgm": float(py_result.xva_by_metric.get("MVA", 0.0)), "ore_output": float(ore_xva.get("MVA", 0.0))},
        ]
    )
    comparison["delta"] = comparison["python_lgm"] - comparison["ore_output"]
    comparison["abs_rel_diff"] = comparison.apply(
        lambda row: np.nan if pd.isna(row["ore_output"]) else abs(float(row["delta"])) / max(abs(float(row["ore_output"])), 1.0),
        axis=1,
    )
    perf = pd.DataFrame(
        [
            {"engine": "python_lgm", "elapsed_sec": py_elapsed},
            {"engine": "ore_fresh_run", "elapsed_sec": float(run_meta.get("elapsed_sec", 0.0))},
        ]
    )
    perf["speed_ratio_vs_python"] = perf["elapsed_sec"] / max(py_elapsed, 1.0e-12)
    meta = {
        "source_ore_xml": str(source_ore_xml),
        "fresh_ore_xml": str(fresh_ore_xml),
        "fresh_output_dir": str(output_dir),
        "fresh_run_root": str(fresh_ore_xml.parent.parent),
        "requested_metrics_in_case": requested_metrics,
        "python_elapsed_sec": py_elapsed,
        "ore_elapsed_sec": float(run_meta.get("elapsed_sec", 0.0)),
        "trade_ids": [t.trade_id for t in snap.portfolio.trades],
        "base_currency": snap.config.base_currency,
    }
    return snap, py_result, comparison, perf, meta


def run_fresh_lgm_calibration_demo(*, ccy_key: str = "EUR"):
    bootstrap_notebook_env()
    from py_ore_tools.irs_xva_utils import parse_lgm_params_from_calibration_xml, parse_lgm_params_from_simulation_xml

    source_ore_xml = default_lgm_calibration_ore_xml()
    fresh_ore_xml, output_dir, run_meta = run_ore_case_fresh(source_ore_xml, label="notebook_03_lgm_calibration")

    ore_root = ET.parse(fresh_ore_xml).getroot()
    sim_node = ore_root.find("./Analytics/Analytic[@type='simulation']/Parameter[@name='simulationConfigFile']")
    if sim_node is None or not (sim_node.text or "").strip():
        raise ValueError(f"simulationConfigFile missing in {fresh_ore_xml}")
    sim_text = (sim_node.text or "").strip()
    sim_xml = (source_ore_xml.parent / sim_text).resolve() if not Path(sim_text).is_absolute() else Path(sim_text)
    calibration_xml = output_dir / "calibration.xml"
    if not calibration_xml.exists():
        raise FileNotFoundError(f"Fresh calibration output not found: {calibration_xml}")

    configured = parse_lgm_params_from_simulation_xml(str(sim_xml), ccy_key=ccy_key)
    calibrated = parse_lgm_params_from_calibration_xml(str(calibration_xml), ccy_key=ccy_key)
    meta = {
        "source_ore_xml": str(source_ore_xml),
        "fresh_ore_xml": str(fresh_ore_xml),
        "simulation_xml": str(sim_xml),
        "calibration_xml": str(calibration_xml),
        "fresh_output_dir": str(output_dir),
        "elapsed_sec": float(run_meta.get("elapsed_sec", 0.0)),
        "calibrate_vol": bool(configured.get("calibrate_vol", False)),
        "calibrate_kappa": bool(configured.get("calibrate_kappa", False)),
        "ccy_key": ccy_key,
    }
    return configured, calibrated, meta


def run_fresh_lgm_calibration_hullwhite_parity_demo(*, base_fresh_ore_xml: str | Path):
    bootstrap_notebook_env()
    repo = find_repo_root()
    ore_exe = locate_ore_exe(repo)
    base_fresh_ore_xml = Path(base_fresh_ore_xml).resolve()
    if not base_fresh_ore_xml.exists():
        raise FileNotFoundError(f"base fresh ore xml not found: {base_fresh_ore_xml}")

    ore_root = ET.parse(base_fresh_ore_xml).getroot()
    sim_node = ore_root.find("./Analytics/Analytic[@type='simulation']/Parameter[@name='simulationConfigFile']")
    if sim_node is None or not (sim_node.text or "").strip():
        raise ValueError(f"simulationConfigFile missing in {base_fresh_ore_xml}")

    source_sim_xml = Path((sim_node.text or "").strip()).resolve()
    if not source_sim_xml.exists():
        raise FileNotFoundError(f"simulation xml not found: {source_sim_xml}")

    run_root = (
        repo
        / "Tools"
        / "PythonOreRunner"
        / "notebook_series"
        / "_fresh_runs"
        / f"notebook_03_lgm_calibration_hw_parity_{int(time.time() * 1000)}"
    )
    input_dir = run_root / "Input"
    output_dir = run_root / "Output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_root = ET.parse(source_sim_xml).getroot()
    for node in sim_root.findall("./CrossAssetModel/InterestRateModels/LGM/Volatility/VolatilityType"):
        node.text = "HullWhite"
    hw_sim_xml = input_dir / "simulation_hw.xml"
    ET.ElementTree(sim_root).write(hw_sim_xml, encoding="utf-8", xml_declaration=True)

    ore_copy_root = ET.parse(base_fresh_ore_xml).getroot()
    _normalize_ore_parameters(ore_copy_root, base_dir=base_fresh_ore_xml.parent, input_dir=input_dir, output_dir=output_dir)
    sim_copy_node = ore_copy_root.find("./Analytics/Analytic[@type='simulation']/Parameter[@name='simulationConfigFile']")
    if sim_copy_node is None:
        raise ValueError(f"simulationConfigFile missing in {base_fresh_ore_xml}")
    sim_copy_node.text = str(hw_sim_xml)
    calibration_cfg_node = ore_copy_root.find("./Analytics/Analytic[@type='calibration']/Parameter[@name='configFile']")
    if calibration_cfg_node is not None:
        calibration_cfg_node.text = str(hw_sim_xml)
    hw_source_ore_xml = input_dir / "ore.xml"
    ET.ElementTree(ore_copy_root).write(hw_source_ore_xml, encoding="utf-8", xml_declaration=True)

    start = time.perf_counter()
    cp = subprocess.run(
        [str(ore_exe), str(hw_source_ore_xml)],
        cwd=str(run_root),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if cp.returncode != 0:
        raise RuntimeError(
            f"Fresh HullWhite parity ORE run failed for {hw_source_ore_xml} with return code {cp.returncode}: "
            f"{cp.stderr[-2000:] or cp.stdout[-2000:]}"
        )
    meta = {
        "source_ore_xml": str(base_fresh_ore_xml),
        "derived_source_ore_xml": str(hw_source_ore_xml),
        "derived_simulation_xml": str(hw_sim_xml),
        "fresh_ore_xml": str(hw_source_ore_xml),
        "fresh_output_dir": str(output_dir),
        "elapsed_sec": elapsed,
        "returncode": int(cp.returncode),
        "run_root": str(run_root),
    }
    return meta


def run_live_swig_stress_classic_irs(*, num_paths: int = 128):
    bootstrap_notebook_env()
    from dataclasses import replace
    from native_xva_interface import (
        IRS,
        MarketData,
        Portfolio,
        Trade,
        XVASnapshot,
        XVALoader,
        ORESwigAdapter,
        stress_classic_native_preset,
    )

    repo = find_repo_root()
    input_dir = repo / "Examples" / "XvaRisk" / "Input"
    base = XVALoader.from_files(str(input_dir), ore_file="ore_stress_classic.xml")
    cfg = replace(stress_classic_native_preset(repo, num_paths=num_paths), analytics=("CVA", "DVA", "FVA", "MVA"))
    snap = XVASnapshot(
        market=MarketData(asof=base.market.asof, raw_quotes=base.market.raw_quotes),
        fixings=base.fixings,
        portfolio=Portfolio(
            trades=(
                Trade(
                    trade_id="IRS1",
                    counterparty="CPTY_A",
                    netting_set="CPTY_A",
                    trade_type="Swap",
                    product=IRS(ccy="EUR", notional=10_000_000, fixed_rate=0.025, maturity_years=5.0, pay_fixed=True),
                ),
            )
        ),
        netting=base.netting,
        collateral=base.collateral,
        config=cfg,
        source_meta=base.source_meta,
    )
    ore_result, ore_elapsed = run_adapter(snap, ORESwigAdapter())
    return snap, ore_result, ore_elapsed


def plot_bar_frame(frame: pd.DataFrame, x: str, y: str, *, title: str, color: str = PALETTE["blue"], ax=None) -> None:
    ax = ax or plt.gca()
    plot_frame = frame.sort_values(y, ascending=False).copy()
    ax.bar(plot_frame[x].astype(str), plot_frame[y].astype(float), color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)


def plot_ranked_bars(frame: pd.DataFrame, label_col: str, value_col: str, *, title: str, color: str = PALETTE["blue"], top_n: int | None = None, ax=None) -> None:
    ax = ax or plt.gca()
    plot_frame = frame[[label_col, value_col]].copy()
    plot_frame = plot_frame.sort_values(value_col, ascending=True)
    if top_n is not None:
        plot_frame = plot_frame.tail(top_n)
    ax.barh(plot_frame[label_col].astype(str), plot_frame[value_col].astype(float), color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_ylabel("")


def plot_snapshot_composition(snapshot, *, title: str = "Snapshot composition") -> None:
    frame = snapshot_overview(snapshot)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    plot_ranked_bars(frame, "component", "count", title=title, color=PALETTE["blue"], ax=axes[0])
    quote_frame = quote_family_frame(snapshot)
    if not quote_frame.empty:
        plot_ranked_bars(quote_frame, "family", "count", title="Quote mix by family", color=PALETTE["teal"], top_n=10, ax=axes[1])
    else:
        axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def plot_xml_buffer_sizes(mapped, *, title: str = "Generated XML payload sizes") -> None:
    frame = xml_buffer_summary(mapped).sort_values("chars", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.barh(frame["xml_name"], frame["chars"], color=PALETTE["cyan"], edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Characters")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def plot_mapping_pipeline(title: str = "Dataclass snapshot to ORE runtime mapping") -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    steps = [
        ("Python dataclasses", "XVASnapshot"),
        ("Mapping layer", "map_snapshot"),
        ("InputParameters", "XML + flat files"),
        ("Runtime adapter", "Python or ORE-SWIG"),
    ]
    x_positions = [0.10, 0.35, 0.62, 0.88]
    for xpos, (head, tail) in zip(x_positions, steps):
        ax.text(
            xpos,
            0.55,
            f"{head}\n{tail}",
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef6ff", ec=PALETTE["blue"], lw=1.5),
            transform=ax.transAxes,
        )
    for start, end in ((0.17, 0.28), (0.42, 0.54), (0.69, 0.81)):
        ax.annotate(
            "",
            xy=(end, 0.55),
            xytext=(start, 0.55),
            arrowprops=dict(arrowstyle="->", lw=2, color=PALETTE["teal"]),
            xycoords=ax.transAxes,
        )
    ax.set_title(title)
    plt.show()
    plt.close(fig)


def plot_fitted_curves(fitted: dict[str, dict[str, Any]], *, title: str = "Discount factor fit") -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["rose"], PALETTE["cyan"]]
    for idx, (ccy, payload) in enumerate(sorted(fitted.items())):
        color = colors[idx % len(colors)]
        ax.plot(payload["times"], payload["dfs"], label=f"{ccy} fit", color=color, lw=2.0)
        ax.scatter(
            payload["instrument_times"],
            np.exp(-np.asarray(payload["instrument_zero_rates"]) * np.asarray(payload["instrument_times"])),
            color=color,
            s=24,
            alpha=0.8,
        )
    ax.set_title(title)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Discount factor")
    ax.legend()
    plt.show()
    plt.close(fig)


def plot_lgm_paths(times: np.ndarray, x_paths: np.ndarray, *, max_paths: int = 25, title: str = "LGM state paths") -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    take = min(max_paths, x_paths.shape[1])
    for idx in range(take):
        ax.plot(times, x_paths[:, idx], color=PALETTE["blue"], alpha=0.18, lw=1.0)
    ax.plot(times, np.mean(x_paths, axis=1), color=PALETTE["rose"], lw=2.2, label="path mean")
    ax.set_title(title)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("x(t)")
    ax.legend()
    plt.show()
    plt.close(fig)


def plot_exposure_profile(times: np.ndarray, epe: np.ndarray, ene: np.ndarray, *, title: str = "Exposure profile") -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(times, epe, label="EPE", color=PALETTE["blue"], lw=2.2)
    ax.plot(times, ene, label="ENE", color=PALETTE["gold"], lw=2.2)
    ax.fill_between(times, epe, color=PALETTE["blue"], alpha=0.12)
    ax.fill_between(times, ene, color=PALETTE["gold"], alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure")
    ax.legend()
    plt.show()
    plt.close(fig)


def plot_metric_comparison(frame: pd.DataFrame, lhs_col: str, rhs_col: str, *, title: str = "Metric comparison") -> None:
    plot_frame = frame.copy()
    x = np.arange(len(plot_frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - width / 2, plot_frame[lhs_col], width=width, label=lhs_col, color=PALETTE["blue"])
    ax.bar(x + width / 2, plot_frame[rhs_col], width=width, label=rhs_col, color=PALETTE["gold"])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["metric"])
    ax.set_title(title)
    ax.legend()
    plt.show()
    plt.close(fig)


def plot_metric_delta(frame: pd.DataFrame, *, metric_col: str = "metric", delta_col: str = "delta", title: str = "Difference by metric") -> None:
    plot_frame = frame.dropna(subset=[delta_col]).copy()
    colors = [PALETTE["rose"] if val < 0 else PALETTE["teal"] for val in plot_frame[delta_col]]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axvline(0.0, color=PALETTE["slate"], lw=1.2)
    ax.barh(plot_frame[metric_col].astype(str), plot_frame[delta_col].astype(float), color=colors, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(delta_col.replace("_", " ").title())
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_bermudan_method_levels(frame: pd.DataFrame, *, title: str = "Bermudan pricing methods vs ORE") -> None:
    required = ["case_name", "py_lsmc", "py_backward", "ore_classic"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing Bermudan comparison columns: {missing}")
    plot_frame = frame.copy()
    x = np.arange(len(plot_frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(x - width, plot_frame["py_lsmc"], width=width, label="py_lsmc", color=PALETTE["blue"])
    ax.bar(x, plot_frame["py_backward"], width=width, label="py_backward", color=PALETTE["teal"])
    ax.bar(x + width, plot_frame["ore_classic"], width=width, label="ore_classic", color=PALETTE["gold"])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["case_name"].astype(str))
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_bermudan_ore_deltas(frame: pd.DataFrame, *, title: str = "Python minus ORE classic for Bermudan cases") -> None:
    required = ["case_name", "py_lsmc_minus_ore_classic", "py_backward_minus_ore_classic"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing Bermudan delta columns: {missing}")
    plot_frame = frame.copy()
    x = np.arange(len(plot_frame))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.axhline(0.0, color=PALETTE["slate"], lw=1.2)
    ax.bar(x - width / 2, plot_frame["py_lsmc_minus_ore_classic"], width=width, label="py_lsmc - ore", color=PALETTE["rose"])
    ax.bar(x + width / 2, plot_frame["py_backward_minus_ore_classic"], width=width, label="py_backward - ore", color=PALETTE["teal"])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["case_name"].astype(str))
    ax.set_title(title)
    ax.set_ylabel("Delta")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_bermudan_exercise_diagnostics(frame: pd.DataFrame, *, title: str = "Bermudan exercise diagnostics") -> None:
    required = ["method", "time", "exercise_probability", "boundary_state"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing Bermudan diagnostic columns: {missing}")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for method, grp in frame.groupby("method"):
        g = grp.sort_values("time")
        axes[0].plot(g["time"], g["exercise_probability"], marker="o", lw=2.0, label=method)
        axes[1].plot(g["time"], g["boundary_state"], marker="o", lw=2.0, label=method)
    axes[0].set_title("Exercise probability by date")
    axes[0].set_xlabel("Exercise time (years)")
    axes[0].set_ylabel("Probability")
    axes[1].set_title("Boundary state by date")
    axes[1].set_xlabel("Exercise time (years)")
    axes[1].set_ylabel("Boundary state")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_bermudan_sensitivity_triplet(frame: pd.DataFrame, *, title: str = "Bermudan sensitivity comparison") -> None:
    required = [
        "normalized_factor",
        "ore_bump_change",
        "ore_direct_quote_bump_change",
        "python_quote_full_bump_change",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing Bermudan sensitivity columns: {missing}")
    plot_frame = frame.copy()
    x = np.arange(len(plot_frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axhline(0.0, color=PALETTE["slate"], lw=1.2)
    ax.bar(x - width, plot_frame["ore_bump_change"], width=width, label="ore_sensitivity_csv", color=PALETTE["gold"])
    ax.bar(
        x,
        plot_frame["ore_direct_quote_bump_change"],
        width=width,
        label="ore_direct_quote_bump",
        color=PALETTE["rose"],
    )
    ax.bar(
        x + width,
        plot_frame["python_quote_full_bump_change"],
        width=width,
        label="python_quote_bump",
        color=PALETTE["teal"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["normalized_factor"].astype(str))
    ax.set_title(title)
    ax.set_ylabel("Bump change")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_boolean_matrix(frame: pd.DataFrame, *, row_col: str, value_cols: list[str], title: str) -> None:
    plot_frame = frame[[row_col, *value_cols]].copy()
    matrix = plot_frame[value_cols].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(value_cols), 0.7 + 0.55 * len(plot_frame)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols)
    ax.set_yticks(np.arange(len(plot_frame)))
    ax.set_yticklabels(plot_frame[row_col].astype(str))
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "Y" if matrix[i, j] >= 0.5 else "N", ha="center", va="center", color=PALETTE["ink"], fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_curve_diagnostics(curve_frame: pd.DataFrame, *, ccy: str, title: str = "Curve diagnostics") -> None:
    plot_frame = curve_frame[curve_frame["ccy"] == ccy].sort_values("time").copy()
    if plot_frame.empty:
        raise ValueError(f"No fitted curve rows for currency {ccy}")
    time = plot_frame["time"].astype(float).to_numpy()
    dfs = plot_frame["df"].astype(float).to_numpy()
    zero = -np.log(np.clip(dfs, 1e-12, None)) / np.where(time == 0.0, np.nan, time)
    zero[0] = zero[1] if len(zero) > 1 else 0.0
    forward = np.gradient(-np.log(np.clip(dfs, 1e-12, None)), time, edge_order=1)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axes[0].plot(time, dfs, color=PALETTE["blue"], lw=2.2)
    axes[0].set_title(f"{ccy} discount factor")
    axes[0].set_xlabel("Time (years)")

    axes[1].plot(time, zero, color=PALETTE["teal"], lw=2.2)
    axes[1].set_title(f"{ccy} zero rate")
    axes[1].set_xlabel("Time (years)")

    axes[2].plot(time, forward, color=PALETTE["gold"], lw=2.2)
    axes[2].set_title(f"{ccy} instantaneous forward")
    axes[2].set_xlabel("Time (years)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def zero_rate_heatmap_frame(df: pd.DataFrame, *, maturities: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)) -> pd.DataFrame:
    rows = []
    for ccy, grp in df.groupby("ccy"):
        g = grp.sort_values("time")
        t = g["time"].to_numpy(dtype=float)
        p = np.clip(g["df"].to_numpy(dtype=float), 1.0e-12, None)
        asof = pd.to_datetime(g["asof_date"].iloc[0])
        for m in maturities:
            p_m = float(np.interp(m, t, p, left=p[0], right=p[-1]))
            z_m = -np.log(p_m) / max(m, 1.0e-12)
            cal_date = asof + pd.to_timedelta(int(round(m * 365.0)), unit="D")
            rows.append(
                {
                    "ccy": ccy,
                    "maturity": m,
                    "calendar_date": cal_date.date().isoformat(),
                    "zero_rate": z_m,
                }
            )
    return pd.DataFrame(rows)


def plot_zero_rate_heatmap(frame: pd.DataFrame, *, title: str = "Approx zero rates by currency / maturity") -> None:
    pivot = frame.pivot(index="ccy", columns="maturity", values="zero_rate").sort_index()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Currency")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=list(pivot.index))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3%}", ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Zero Rate")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_log_discount_factors(df: pd.DataFrame, *, title: str = "log(DF) by currency") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for ccy, grp in df.groupby("ccy"):
        g = grp.sort_values("time")
        safe_df = np.clip(g["df"].to_numpy(dtype=float), 1.0e-12, None)
        ax.plot(g["time"], np.log(safe_df), linewidth=1.6, label=ccy)
    ax.set_title(title)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("log(Discount Factor)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def lgm_param_frame(params_dict: dict[str, Any], *, param: str) -> pd.DataFrame:
    if param not in {"alpha", "kappa"}:
        raise ValueError(f"Unsupported LGM parameter '{param}'")
    times = np.asarray(params_dict[f"{param}_times"], dtype=float)
    values = np.asarray(params_dict[f"{param}_values"], dtype=float)
    labels = [f"{t:.3g}Y" for t in times] + ["terminal"]
    horizons = np.concatenate(([0.0], times[: max(len(values) - 1, 0)]))
    if len(horizons) != len(values):
        horizons = np.arange(len(values), dtype=float)
    return pd.DataFrame(
        {
            "bucket_index": np.arange(len(values), dtype=int),
            "bucket_label": labels[: len(values)],
            "horizon_years": horizons,
            "value": values,
        }
    )


def plot_lgm_calibration_summary(configured: dict[str, Any], calibrated: dict[str, Any], *, title: str = "LGM calibration summary") -> None:
    alpha_cfg = lgm_param_frame(configured, param="alpha")
    alpha_cal = lgm_param_frame(calibrated, param="alpha")
    kappa_cfg = lgm_param_frame(configured, param="kappa")
    kappa_cal = lgm_param_frame(calibrated, param="kappa")

    alpha_merge = alpha_cal[["bucket_index", "bucket_label", "horizon_years", "value"]].rename(columns={"value": "calibrated_alpha"})
    alpha_merge = alpha_merge.merge(
        alpha_cfg[["bucket_index", "value"]].rename(columns={"value": "configured_alpha"}),
        on="bucket_index",
        how="left",
    )
    alpha_merge["configured_alpha"] = alpha_merge["configured_alpha"].fillna(alpha_cfg["value"].iloc[-1])
    alpha_merge["alpha_shift_bp"] = 1.0e4 * (alpha_merge["calibrated_alpha"] - alpha_merge["configured_alpha"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    axes[0].step(alpha_cfg["horizon_years"], alpha_cfg["value"], where="post", lw=2.0, color=PALETTE["slate"], label="configured")
    axes[0].step(alpha_cal["horizon_years"], alpha_cal["value"], where="post", lw=2.2, color=PALETTE["rose"], label="calibrated")
    axes[0].set_title("Alpha term structure")
    axes[0].set_xlabel("Horizon (years)")
    axes[0].legend()

    colors = [PALETTE["teal"] if x >= 0.0 else PALETTE["rose"] for x in alpha_merge["alpha_shift_bp"]]
    axes[1].bar(alpha_merge["bucket_label"], alpha_merge["alpha_shift_bp"], color=colors, edgecolor="white")
    axes[1].axhline(0.0, color=PALETTE["slate"], lw=1.0)
    axes[1].set_title("Alpha shift vs template")
    axes[1].set_ylabel("Basis points")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].step(kappa_cfg["horizon_years"], kappa_cfg["value"], where="post", lw=2.0, color=PALETTE["gold"], label="configured")
    axes[2].step(kappa_cal["horizon_years"], kappa_cal["value"], where="post", lw=2.2, color=PALETTE["blue"], linestyle="--", label="calibrated")
    axes[2].set_title("Kappa term structure")
    axes[2].set_xlabel("Horizon (years)")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def lgm_benchmark_frame(model, times: np.ndarray, *, path_counts: tuple[int, ...] = (2000, 10000, 50000), repeats: int = 5, warmup: int = 1) -> pd.DataFrame:
    from py_ore_tools import simulate_ba_measure, simulate_lgm_measure

    rows = []

    def _bench(fn):
        for _ in range(warmup):
            fn()
        durations = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            durations.append(time.perf_counter() - t0)
        arr = np.asarray(durations, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min())

    for n_paths in path_counts:
        path_steps = n_paths * max(len(times) - 1, 1)
        lgm_mean, lgm_std, lgm_min = _bench(lambda n=n_paths: simulate_lgm_measure(model, times, n_paths=n, rng=np.random.default_rng(42), x0=0.0))
        ba_mean, ba_std, ba_min = _bench(lambda n=n_paths: simulate_ba_measure(model, times, n_paths=n, rng=np.random.default_rng(42), x0=0.0, y0=0.0))
        rows.extend(
            [
                {
                    "measure": "lgm",
                    "n_paths": n_paths,
                    "mean_sec": lgm_mean,
                    "std_sec": lgm_std,
                    "min_sec": lgm_min,
                    "path_steps_per_sec": path_steps / max(lgm_mean, 1.0e-12),
                },
                {
                    "measure": "ba",
                    "n_paths": n_paths,
                    "mean_sec": ba_mean,
                    "std_sec": ba_std,
                    "min_sec": ba_min,
                    "path_steps_per_sec": path_steps / max(ba_mean, 1.0e-12),
                },
            ]
        )
    return pd.DataFrame(rows)


def parse_parity_report() -> tuple[pd.DataFrame, str]:
    repo = find_repo_root()
    report_path = _pythonorerunner_root(repo) / "legacy" / "native_xva_interface" / "docs" / "PY_LGM_ORE_PARITY_REPORT.md"
    text = report_path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        match = re.search(r"- `([A-Z]+)`: `([-0-9.]+)` vs `([-0-9.]+)`", line)
        if match:
            metric, py_val, ore_val = match.groups()
            rows.append(
                {
                    "metric": metric,
                    "python_lgm": float(py_val),
                    "ore_swig": float(ore_val),
                    "delta": float(ore_val) - float(py_val),
                }
            )
    summary = "CVA remains the main residual gap; DVA and FVA are close enough to frame as parity-ready for prototype use."
    return pd.DataFrame(rows), summary


def _read_ore_xva_row(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader((line.lstrip("#") for line in handle if line.strip())))
    for row in rows:
        trade_id = (row.get("TradeId") or row.get("#TradeId") or "").strip()
        if trade_id:
            continue
        out = {}
        for key in ("CVA", "DVA", "FBA", "FCA", "MVA", "BaselEPE", "BaselEEPE"):
            if key in row:
                try:
                    out[key] = float(row[key] or 0.0)
                except ValueError:
                    out[key] = 0.0
        return out
    return {}


def _read_ore_npv_row(path: Path) -> float:
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader((line.lstrip("#") for line in handle if line.strip())))
    if not rows:
        return 0.0
    row = rows[0]
    for key in ("NPV(Base)", "NPV"):
        if key in row:
            try:
                return float(row[key] or 0.0)
            except ValueError:
                return 0.0
    return 0.0


def run_adapter(snapshot, adapter) -> tuple[Any, float]:
    bootstrap_notebook_env()
    from native_xva_interface import XVAEngine

    start = time.perf_counter()
    result = XVAEngine(adapter=adapter).create_session(snapshot).run(return_cubes=False)
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_python_lgm_demo(seed: int = 42, n_paths: int = 4000) -> dict[str, Any]:
    bootstrap_notebook_env()
    from py_ore_tools.irs_xva_utils import (
        aggregate_exposure_profile_from_npv_paths,
        build_discount_curve_from_zero_rate_pairs,
        build_irregular_exposure_grid,
        build_swap_schedules,
        compute_xva_from_exposure_profile,
        par_rate_from_trade,
        payer_swap_npv_at_time,
    )
    from py_ore_tools.lgm import LGM1F, LGMParams, simulate_lgm_measure

    trade_def = {
        "TradeType": "Swap",
        "Envelope": {"CounterParty": "CPY_P", "NettingSetId": "CPY_P"},
        "SwapData": {
            "Start": 0.0,
            "End": 7.0,
            "LegData": [
                {
                    "LegType": "Fixed",
                    "Payer": True,
                    "Currency": "USD",
                    "DayCounter": "ACT/360",
                    "Frequency": "Semiannual",
                    "Notional": 10_000_000.0,
                    "FrontStub": 0.5,
                    "FixedRate": None,
                },
                {
                    "LegType": "Floating",
                    "Payer": False,
                    "Currency": "USD",
                    "Index": "USD-SOFR",
                    "DayCounter": "ACT/360",
                    "Frequency": "Quarterly",
                    "Notional": 10_000_000.0,
                },
            ],
        },
    }

    params = LGMParams(
        alpha_times=(0.5, 1.0, 2.0, 4.0),
        alpha_values=(0.010, 0.013, 0.017, 0.016, 0.013),
        kappa_times=(1.0, 3.0),
        kappa_values=(0.040, 0.030, 0.022),
        shift=0.0,
        scaling=1.0,
    )
    model = LGM1F(params)
    p0 = build_discount_curve_from_zero_rate_pairs([(0.0, 0.031), (7.0, 0.031)])
    fixed_dates, float_dates, maturity = build_swap_schedules(trade_def)
    times = build_irregular_exposure_grid(maturity)
    par_rate = par_rate_from_trade(model, p0, fixed_dates, float_dates, trade_def)
    fixed_rate = par_rate + 0.0015

    x_paths = simulate_lgm_measure(model, times, n_paths=n_paths, rng=np.random.default_rng(seed), x0=0.0)
    npv_paths = np.zeros_like(x_paths)
    for idx, tval in enumerate(times):
        npv_paths[idx] = payer_swap_npv_at_time(model, p0, fixed_dates, float_dates, tval, x_paths[idx], fixed_rate, trade_def)

    exposure = aggregate_exposure_profile_from_npv_paths(npv_paths)
    discount = np.asarray([p0(float(t)) for t in times], dtype=float)
    survival_cpty = np.exp(-0.02 * times)
    survival_own = np.exp(-0.015 * times)
    xva = compute_xva_from_exposure_profile(
        times,
        exposure["epe"],
        exposure["ene"],
        discount,
        survival_cpty,
        survival_own,
        recovery_cpty=0.4,
        recovery_own=0.4,
        funding_spread=0.005,
    )

    metrics = {
        "CVA": float(xva["cva"]),
        "DVA": float(xva["dva"]),
        "FVA": float(xva["fva"]),
        "XVA_TOTAL": float(xva["cva"] - xva["dva"] + xva["fva"]),
    }
    return {
        "model": model,
        "params": params,
        "trade_def": trade_def,
        "times": times,
        "fixed_dates": fixed_dates,
        "float_dates": float_dates,
        "par_rate": float(par_rate),
        "fixed_rate": float(fixed_rate),
        "x_paths": x_paths,
        "npv_paths": npv_paths,
        "exposure": exposure,
        "discount": discount,
        "metrics": metrics,
    }


def lgm_params_frame(params) -> pd.DataFrame:
    alpha_times = list(params.alpha_times) + ["terminal"]
    kappa_times = list(params.kappa_times) + ["terminal"]
    return pd.DataFrame(
        {
            "alpha_time": alpha_times,
            "alpha_value": list(params.alpha_values),
            "kappa_time": kappa_times + [np.nan] * max(0, len(params.alpha_values) - len(kappa_times)),
            "kappa_value": list(params.kappa_values) + [np.nan] * max(0, len(params.alpha_values) - len(params.kappa_values)),
        }
    )


def to_pretty_json(data: Any, *, limit: int = 1200) -> str:
    text = json.dumps(data, indent=2, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + "\n..."


def _root_tag(xml_text: str) -> str:
    match = re.search(r"<([A-Za-z0-9_:-]+)", xml_text)
    return match.group(1) if match else ""
