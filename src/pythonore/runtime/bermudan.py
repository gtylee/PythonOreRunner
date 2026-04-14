from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from datetime import datetime
import os
from pathlib import Path
import tempfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from pythonore.domain.dataclasses import BermudanSwaption, MarketQuote, Trade, XVASnapshot
from pythonore.io.loader import XVALoader
from pythonore.mapping.mapper import map_snapshot
from pythonore.runtime.exceptions import EngineRunError, ValidationError
from pythonore.runtime.lgm import market as lgm_market


@dataclass(frozen=True)
class BermudanPvSensitivity:
    factor: str
    shift_size: float
    base_quote_value: float
    base_price: float
    bumped_up_price: float
    bumped_down_price: float
    delta: float


@dataclass(frozen=True)
class BermudanPricingResult:
    trade_id: str
    price: float
    method: str
    exercise_diagnostics: tuple[object, ...]
    curve_source: str
    model_param_source: str
    num_paths: int
    seed: int
    discount_column: str
    forward_column: str
    sensitivities: tuple[BermudanPvSensitivity, ...] = ()


@dataclass(frozen=True)
class BermudanBenchmarkResult:
    pricing: BermudanPricingResult
    ore_price: Optional[float]
    ore_price_diff: Optional[float]
    ore_sensitivity_rows: tuple[dict, ...]
    amc_npv: Optional[float]
    amc_exposure_rows: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class BermudanSensitivityBenchmarkRow:
    factor: str
    base_quote_value: float
    python_full_reprice_delta: float
    python_fast_delta: float
    fast_minus_full: float
    ore_delta: Optional[float] = None
    fast_minus_ore: Optional[float] = None
    full_minus_ore: Optional[float] = None


@dataclass(frozen=True)
class BermudanSensitivityBenchmarkResult:
    trade_id: str
    method: str
    shift_size: float
    rows: tuple[BermudanSensitivityBenchmarkRow, ...]
    ore_sensitivity_rows: tuple[dict, ...]
    notes: tuple[str, ...]


def price_bermudan_from_ore_case(
    input_dir: str | Path,
    ore_file: str = "ore.xml",
    trade_id: Optional[str] = None,
    *,
    method: str = "lsmc",
    num_paths: int = 8192,
    seed: int = 42,
    basis_degree: int = 2,
    curve_mode: str = "auto",
) -> BermudanPricingResult:
    snapshot = XVALoader.from_files(str(input_dir), ore_file=ore_file)
    return _price_bermudan_snapshot(
        snapshot,
        trade_id=trade_id,
        method=method,
        num_paths=num_paths,
        seed=seed,
        basis_degree=basis_degree,
        curve_mode=curve_mode,
    )


def price_bermudan_with_sensis_from_ore_case(
    input_dir: str | Path,
    ore_file: str = "ore.xml",
    trade_id: Optional[str] = None,
    *,
    method: str = "lsmc",
    num_paths: int = 8192,
    seed: int = 42,
    basis_degree: int = 2,
    factors: Optional[Sequence[str]] = None,
    shift_size: float = 1.0e-4,
    sensitivity_mode: str = "full_reprice",
) -> BermudanPricingResult:
    snapshot = XVALoader.from_files(str(input_dir), ore_file=ore_file)
    ctx = _build_bermudan_context(
        snapshot,
        trade_id=trade_id,
        method=method,
        num_paths=num_paths,
        seed=seed,
        basis_degree=basis_degree,
        curve_mode="market_fit",
    )
    base = ctx["result"]
    trade = ctx["trade"]
    relevant = _relevant_market_factors(snapshot, trade)
    selected = [f for f in relevant if factors is None or f in set(factors)]

    mode = str(sensitivity_mode).strip().lower()
    if mode == "fast_curve_jacobian":
        entries = _fast_curve_jacobian_sensitivities(snapshot, ctx, selected, shift_size)
        return replace(base, sensitivities=tuple(entries))
    if mode != "full_reprice":
        raise ValidationError(f"Unsupported Bermudan sensitivity mode '{sensitivity_mode}'")

    entries: List[BermudanPvSensitivity] = []
    for factor in selected:
        base_quote = next(q for q in snapshot.market.raw_quotes if q.key == factor)
        up_snap = replace(snapshot, market=replace(snapshot.market, raw_quotes=_bump_quotes(snapshot.market.raw_quotes, factor, shift_size)))
        dn_snap = replace(snapshot, market=replace(snapshot.market, raw_quotes=_bump_quotes(snapshot.market.raw_quotes, factor, -shift_size)))
        up = _price_bermudan_snapshot(
            up_snap,
            trade_id=trade.trade_id,
            method=method,
            num_paths=num_paths,
            seed=seed,
            basis_degree=basis_degree,
            curve_mode="market_fit",
        )
        dn = _price_bermudan_snapshot(
            dn_snap,
            trade_id=trade.trade_id,
            method=method,
            num_paths=num_paths,
            seed=seed,
            basis_degree=basis_degree,
            curve_mode="market_fit",
        )
        entries.append(
            BermudanPvSensitivity(
                factor=factor,
                shift_size=float(shift_size),
                base_quote_value=float(base_quote.value),
                base_price=float(base.price),
                bumped_up_price=float(up.price),
                bumped_down_price=float(dn.price),
                delta=float((up.price - dn.price) / (2.0 * shift_size)),
            )
        )

    return replace(base, sensitivities=tuple(entries))


def benchmark_bermudan_sensitivities_from_ore_case(
    input_dir: str | Path,
    ore_file: str = "ore.xml",
    trade_id: Optional[str] = None,
    *,
    method: str = "backward",
    num_paths: int = 8192,
    seed: int = 42,
    basis_degree: int = 2,
    factors: Optional[Sequence[str]] = None,
    shift_size: float = 1.0e-4,
    ore_sensitivity_csv: str | Path | None = None,
) -> BermudanSensitivityBenchmarkResult:
    full = price_bermudan_with_sensis_from_ore_case(
        input_dir,
        ore_file=ore_file,
        trade_id=trade_id,
        method=method,
        num_paths=num_paths,
        seed=seed,
        basis_degree=basis_degree,
        factors=factors,
        shift_size=shift_size,
        sensitivity_mode="full_reprice",
    )
    fast = price_bermudan_with_sensis_from_ore_case(
        input_dir,
        ore_file=ore_file,
        trade_id=trade_id,
        method=method,
        num_paths=num_paths,
        seed=seed,
        basis_degree=basis_degree,
        factors=factors,
        shift_size=shift_size,
        sensitivity_mode="fast_curve_jacobian",
    )

    ore_rows: tuple[dict, ...] = ()
    notes: List[str] = []
    ore_map: Dict[str, float] = {}
    if ore_sensitivity_csv is not None:
        path = Path(ore_sensitivity_csv)
        if path.exists():
            ore_rows = tuple(_load_generic_sensitivity_rows(path))
            for row in ore_rows:
                factor = str(row.get("Factor", row.get("factor", ""))).strip()
                for val_key in ("Delta", "delta", "Sensitivity", "sensitivity"):
                    if factor and val_key in row and str(row[val_key]).strip():
                        try:
                            ore_map[factor] = float(row[val_key])
                            break
                        except ValueError:
                            pass
        else:
            notes.append(f"ORE sensitivity benchmark unavailable: missing {path}")
    else:
        notes.append("ORE PV sensitivity CSV not provided; benchmark compares fast vs full Python only")

    full_map = {s.factor: s for s in full.sensitivities}
    fast_map = {s.factor: s for s in fast.sensitivities}
    factors_all = sorted(set(full_map) & set(fast_map))
    rows = []
    for factor in factors_all:
        full_s = full_map[factor]
        fast_s = fast_map[factor]
        ore_delta = ore_map.get(factor)
        rows.append(
            BermudanSensitivityBenchmarkRow(
                factor=factor,
                base_quote_value=float(full_s.base_quote_value),
                python_full_reprice_delta=float(full_s.delta),
                python_fast_delta=float(fast_s.delta),
                fast_minus_full=float(fast_s.delta - full_s.delta),
                ore_delta=ore_delta,
                fast_minus_ore=None if ore_delta is None else float(fast_s.delta - ore_delta),
                full_minus_ore=None if ore_delta is None else float(full_s.delta - ore_delta),
            )
        )
    return BermudanSensitivityBenchmarkResult(
        trade_id=str(full.trade_id),
        method=str(method).strip().lower(),
        shift_size=float(shift_size),
        rows=tuple(rows),
        ore_sensitivity_rows=ore_rows,
        notes=tuple(notes),
    )


def benchmark_bermudan_from_ore_case(
    input_dir: str | Path,
    ore_file: str = "ore.xml",
    trade_id: Optional[str] = None,
    *,
    method: str = "lsmc",
    num_paths: int = 8192,
    seed: int = 42,
    basis_degree: int = 2,
    ore_sensitivity_csv: str | Path | None = None,
    amc_output_dir: str | Path | None = None,
) -> BermudanBenchmarkResult:
    pricing = price_bermudan_from_ore_case(
        input_dir=input_dir,
        ore_file=ore_file,
        trade_id=trade_id,
        method=method,
        num_paths=num_paths,
        seed=seed,
        basis_degree=basis_degree,
    )
    snapshot = XVALoader.from_files(str(input_dir), ore_file=ore_file)
    trade = _resolve_bermudan_trade(snapshot, trade_id)
    output_dir = _ore_output_dir(snapshot)
    notes: List[str] = []

    ore_price = None
    ore_price_diff = None
    npv_csv = output_dir / "npv.csv"
    if npv_csv.exists():
        ore_price = _load_trade_npv(npv_csv, trade.trade_id)
        ore_price_diff = pricing.price - ore_price
    else:
        notes.append(f"ORE direct price benchmark unavailable: missing {npv_csv}")

    ore_rows: tuple[dict, ...] = ()
    if ore_sensitivity_csv is not None:
        sensi_path = Path(ore_sensitivity_csv)
        if sensi_path.exists():
            ore_rows = tuple(_load_generic_sensitivity_rows(sensi_path))
        else:
            notes.append(f"ORE sensitivity benchmark unavailable: missing {sensi_path}")

    amc_npv = None
    amc_exposure_rows = 0
    amc_dir = Path(amc_output_dir) if amc_output_dir is not None else _default_amc_output_dir(snapshot)
    if amc_dir is not None:
        amc_npv_csv = amc_dir / "npv.csv"
        amc_expo_csv = amc_dir / f"exposure_trade_{trade.trade_id}.csv"
        if amc_npv_csv.exists():
            try:
                amc_npv = _load_trade_npv(amc_npv_csv, trade.trade_id)
            except ValueError:
                notes.append(f"AMC npv benchmark present but trade {trade.trade_id} not found in {amc_npv_csv}")
        if amc_expo_csv.exists():
            amc_exposure_rows = _count_csv_rows(amc_expo_csv)
        else:
            notes.append(f"AMC exposure benchmark unavailable: missing {amc_expo_csv}")
    else:
        notes.append("AMC benchmark directory could not be inferred from this case")

    return BermudanBenchmarkResult(
        pricing=pricing,
        ore_price=ore_price,
        ore_price_diff=ore_price_diff,
        ore_sensitivity_rows=ore_rows,
        amc_npv=amc_npv,
        amc_exposure_rows=amc_exposure_rows,
        notes=tuple(notes),
    )


def _price_bermudan_snapshot(
    snapshot: XVASnapshot,
    *,
    trade_id: Optional[str],
    method: str,
    num_paths: int,
    seed: int,
    basis_degree: int,
    curve_mode: str,
) -> BermudanPricingResult:
    ctx = _build_bermudan_context(
        snapshot,
        trade_id=trade_id,
        method=method,
        num_paths=num_paths,
        seed=seed,
        basis_degree=basis_degree,
        curve_mode=curve_mode,
    )
    return ctx["result"]


def _build_bermudan_context(
    snapshot: XVASnapshot,
    *,
    trade_id: Optional[str],
    method: str,
    num_paths: int,
    seed: int,
    basis_degree: int,
    curve_mode: str,
) -> Dict[str, object]:
    trade = _resolve_bermudan_trade(snapshot, trade_id)
    pricing_method = str(method).strip().lower()
    if pricing_method not in ("lsmc", "backward"):
        raise ValidationError(f"Unsupported Bermudan pricing method '{method}'")
    mapped = map_snapshot(snapshot)
    curve_bundle = _build_curves(snapshot, mapped, trade, curve_mode)
    model, model_source = _build_lgm_model(snapshot, mapped, trade, curve_bundle)
    exercise_times = _exercise_times(
        snapshot.config.asof,
        trade.product.exercise_dates,
        curve_bundle["model_day_counter"],
        curve_bundle["ore_snapshot_mod"],
        simulation_xml_text=mapped.xml_buffers.get("simulation.xml", ""),
    )
    legs = curve_bundle["irs_utils"].load_swap_legs_from_portfolio(
        curve_bundle["portfolio_xml_path"],
        trade.trade_id,
        asof_date=snapshot.config.asof,
        time_day_counter=curve_bundle["model_day_counter"],
    )
    bermudan_def = curve_bundle["lgm_ir_options"].BermudanSwaptionDef(
        trade_id=trade.trade_id,
        exercise_times=exercise_times,
        underlying_legs=legs,
        exercise_sign=_exercise_sign(trade.product),
        settlement=str(trade.product.settlement).strip().lower(),
    )
    times = _pricing_grid(legs, exercise_times)
    if pricing_method == "lsmc":
        x_paths = curve_bundle["lgm_mod"].simulate_lgm_measure(
            model,
            times,
            n_paths=int(num_paths),
            rng=np.random.default_rng(int(seed)),
            x0=0.0,
        )
        result = curve_bundle["lgm_ir_options"].bermudan_lsmc_result(
            model,
            curve_bundle["p0_disc"],
            curve_bundle["p0_fwd"],
            bermudan_def,
            times,
            x_paths,
            basis_degree=basis_degree,
            itm_only=True,
        )
        price = float(np.mean(result.npv_paths[0, :]))
        diagnostics = tuple(result.diagnostics)
    else:
        engine_kwargs = _backward_engine_kwargs(snapshot)
        result = curve_bundle["lgm_ir_options"].bermudan_backward_price(
            model,
            curve_bundle["p0_disc"],
            curve_bundle["p0_fwd"],
            bermudan_def,
            n_grid=max(61, basis_degree * 40 + 1),
            convolution_sx=float(engine_kwargs["sx"]),
            convolution_nx=int(engine_kwargs["nx"]),
            convolution_sy=float(engine_kwargs["sy"]),
            convolution_ny=int(engine_kwargs["ny"]),
        )
        price = float(result.price)
        diagnostics = tuple(result.diagnostics)
    result_obj = BermudanPricingResult(
        trade_id=trade.trade_id,
        price=price,
        method=pricing_method,
        exercise_diagnostics=diagnostics,
        curve_source=str(curve_bundle["curve_source"]),
        model_param_source=str(model_source),
        num_paths=int(num_paths),
        seed=int(seed),
        discount_column=str(curve_bundle["discount_column"]),
        forward_column=str(curve_bundle["forward_column"]),
    )
    return {
        "result": result_obj,
        "snapshot": snapshot,
        "trade": trade,
        "curve_bundle": curve_bundle,
        "model": model,
        "mapped": mapped,
        "times": times,
        "legs": legs,
        "bermudan_def": bermudan_def,
        "pricing_method": pricing_method,
        "basis_degree": int(basis_degree),
        "num_paths": int(num_paths),
        "seed": int(seed),
    }


def _resolve_bermudan_trade(snapshot: XVASnapshot, trade_id: Optional[str]) -> Trade:
    candidates = [t for t in snapshot.portfolio.trades if isinstance(t.product, BermudanSwaption)]
    if trade_id is not None:
        candidates = [t for t in candidates if t.trade_id == trade_id]
    if not candidates:
        raise ValidationError("No native BermudanSwaption trade found in snapshot portfolio")
    return candidates[0]


def _build_lgm_model(snapshot: XVASnapshot, mapped, trade: Trade, curve_bundle: Dict[str, object]):
    from pythonore.compute.lgm import LGM1F, LGMParams
    from pythonore.compute.irs_xva_utils import (
        parse_lgm_params_from_calibration_xml,
        parse_lgm_params_from_simulation_xml,
    )

    output_dir = _ore_output_dir(snapshot)
    calibration_xml = output_dir / "calibration.xml"
    if calibration_xml.exists():
        params = parse_lgm_params_from_calibration_xml(str(calibration_xml), ccy_key=trade.product.ccy)
        source = "calibration"
        return (
            LGM1F(
                LGMParams(
                    alpha_times=tuple(float(x) for x in params["alpha_times"]),
                    alpha_values=tuple(float(x) for x in params["alpha_values"]),
                    kappa_times=tuple(float(x) for x in params["kappa_times"]),
                    kappa_values=tuple(float(x) for x in params["kappa_values"]),
                    shift=float(params["shift"]),
                    scaling=float(params["scaling"]),
                )
            ),
            source,
        )

    trade_specific = _build_trade_specific_lgm_params(snapshot, trade, curve_bundle)
    if trade_specific is not None:
        params, source = trade_specific
    else:
        source = "simulation"
        with _temp_xml_file(mapped.xml_buffers["simulation.xml"]) as path:
            params = parse_lgm_params_from_simulation_xml(path, ccy_key=trade.product.ccy)
    return (
        LGM1F(
            LGMParams(
                alpha_times=tuple(float(x) for x in params["alpha_times"]),
                alpha_values=tuple(float(x) for x in params["alpha_values"]),
                kappa_times=tuple(float(x) for x in params["kappa_times"]),
                kappa_values=tuple(float(x) for x in params["kappa_values"]),
                shift=float(params["shift"]),
                scaling=float(params["scaling"]),
            )
        ),
        source,
    )


def _build_trade_specific_lgm_params(snapshot: XVASnapshot, trade: Trade, curve_bundle: Dict[str, object]):
    from pythonore.compute.lgm import LGM1F, LGMParams

    try:
        import QuantLib as ql
    except Exception:
        return None
    ql.Settings.instance().evaluationDate = ql.DateParser.parseISO(snapshot.config.asof)

    if not isinstance(trade.product, BermudanSwaption):
        return None

    pricingengine_xml = snapshot.config.xml_buffers.get("pricingengine.xml", "")
    if not pricingengine_xml:
        return None

    try:
        engine_spec = _parse_bermudan_engine_spec(pricingengine_xml)
        basket = _build_trade_specific_basket(snapshot, trade, curve_bundle, ql)
        if not basket:
            return None
        yts = _build_quantlib_discount_curve(snapshot, curve_bundle["p0_disc"], ql)
        index = _build_quantlib_ibor_index(trade.product.float_index, yts, ql)
        if index is None:
            return None

        fixed_leg_dc = _infer_fixed_leg_day_counter(curve_bundle["portfolio_xml_path"], trade.trade_id, ql)
        fixed_leg_tenor = _infer_fixed_leg_tenor(curve_bundle["portfolio_xml_path"], trade.trade_id, ql)
        helpers = []
        for expiry_period, term_period, market_vol, _, vol_kind in basket:
            vol_handle = ql.QuoteHandle(ql.SimpleQuote(float(market_vol)))
            ql_vol_type = ql.Normal if str(vol_kind).strip().lower() == "normal" else ql.ShiftedLognormal
            helper = ql.SwaptionHelper(
                expiry_period,
                term_period,
                vol_handle,
                index,
                fixed_leg_tenor,
                fixed_leg_dc,
                index.dayCounter(),
                yts,
                ql.BlackCalibrationHelper.RelativePriceError,
                ql.nullDouble(),
                1.0,
                ql_vol_type,
                0.0,
            )
            helpers.append(helper)

        step_dates = [yts.referenceDate() + p for p, _, _, _, _ in basket[:-1]]
        sigma_quotes = [ql.QuoteHandle(ql.SimpleQuote(float(engine_spec["volatility"]))) for _ in range(len(step_dates) + 1)]
        rev_quotes = [ql.QuoteHandle(ql.SimpleQuote(float(engine_spec["reversion"])))]
        model = ql.Gsr(yts, step_dates, sigma_quotes, rev_quotes)
        engine = ql.Gaussian1dSwaptionEngine(model, 64, 7.0, True, False, yts)
        for helper in helpers:
            helper.setPricingEngine(engine)

        opt = ql.LevenbergMarquardt()
        ec = ql.EndCriteria(1000, 100, 1.0e-8, 1.0e-8, 1.0e-8)
        model.calibrateVolatilitiesIterative(helpers, opt, ec)

        params = list(model.params())
        sigmas = np.asarray(params[1:], dtype=float)
        alpha_times = np.asarray(
            [ql.ActualActual(ql.ActualActual.ISDA).yearFraction(yts.referenceDate(), d) for d in step_dates],
            dtype=float,
        )
        kappa_values = np.array([float(engine_spec["reversion"])], dtype=float)
        kappa_times = np.array([], dtype=float)
        shift_horizon_time = float(engine_spec["shift_horizon_ratio"]) * float(basket[-1][3])
        temp = LGM1F(
            LGMParams(
                alpha_times=tuple(float(x) for x in alpha_times),
                alpha_values=tuple(float(x) for x in sigmas),
                kappa_times=(),
                kappa_values=(float(engine_spec["reversion"]),),
                shift=0.0,
                scaling=1.0,
            )
        )
        shift = -float(temp.H(shift_horizon_time))
        return (
            {
                "alpha_times": alpha_times,
                "alpha_values": sigmas,
                "kappa_times": kappa_times,
                "kappa_values": kappa_values,
                "shift": shift,
                "scaling": 1.0,
            },
            "trade_specific_gsr",
        )
    except Exception:
        return None


def _parse_bermudan_engine_spec(pricingengine_xml_text: str) -> Dict[str, float]:
    root = ET.fromstring(pricingengine_xml_text)
    node = root.find("./Product[@type='BermudanSwaption']")
    if node is None:
        raise ValidationError("pricingengine.xml is missing a BermudanSwaption engine specification")
    model_params = {p.get("name", ""): (p.text or "").strip() for p in node.findall("./ModelParameters/Parameter")}
    return {
        "reversion": float(model_params.get("Reversion", "0.0") or "0.0"),
        "volatility": float(model_params.get("Volatility", "0.01") or "0.01"),
        "shift_horizon_ratio": float(model_params.get("ShiftHorizon", "0.0") or "0.0"),
    }


def _backward_engine_kwargs(snapshot: XVASnapshot) -> Dict[str, float]:
    pricingengine_xml = snapshot.config.xml_buffers.get("pricingengine.xml", "")
    if not pricingengine_xml:
        return {"sx": 3.0, "nx": 10.0, "sy": 3.0, "ny": 10.0}
    root = ET.fromstring(pricingengine_xml)
    node = root.find("./Product[@type='BermudanSwaption']")
    if node is None:
        return {"sx": 3.0, "nx": 10.0, "sy": 3.0, "ny": 10.0}
    engine_params = {p.get("name", ""): (p.text or "").strip() for p in node.findall("./EngineParameters/Parameter")}
    return {
        "sx": float(engine_params.get("sx", "3.0") or "3.0"),
        "nx": float(engine_params.get("nx", "10") or "10"),
        "sy": float(engine_params.get("sy", "3.0") or "3.0"),
        "ny": float(engine_params.get("ny", "10") or "10"),
    }


def _build_trade_specific_basket(snapshot: XVASnapshot, trade: Trade, curve_bundle: Dict[str, object], ql):
    if not isinstance(trade.product, BermudanSwaption):
        return []
    asof = ql.DateParser.parseISO(snapshot.config.asof)
    exercise_dates = [ql.DateParser.parseISO(d) for d in trade.product.exercise_dates]
    maturity_date = _infer_underlying_maturity_date(curve_bundle["portfolio_xml_path"], trade.trade_id, ql)
    if maturity_date is None:
        return []

    basket = []
    for ex in exercise_dates:
        if ex <= asof or ex >= maturity_date:
            continue
        expiry_years = ql.ActualActual(ql.ActualActual.ISDA).yearFraction(asof, ex)
        term_years = ql.ActualActual(ql.ActualActual.ISDA).yearFraction(ex, maturity_date)
        expiry_period = _year_fraction_to_period(expiry_years, ql)
        term_period = _year_fraction_to_period(term_years, ql)
        market_vol = _lookup_atm_swaption_vol(snapshot.market.raw_quotes, trade.product.ccy, expiry_period, term_period)
        if market_vol is None:
            return []
        basket.append((expiry_period, term_period, market_vol[0], expiry_years, market_vol[1]))
    return basket


def _lookup_atm_swaption_vol(
    quotes: Iterable[MarketQuote],
    ccy: str,
    expiry_period,
    term_period,
) -> tuple[float, str] | None:
    exp = _period_to_ore_tenor(expiry_period)
    term = _period_to_ore_tenor(term_period)
    target_e = _ore_tenor_to_years(exp)
    target_t = _ore_tenor_to_years(term)
    if target_e is None or target_t is None:
        return None
    for vol_kind in ("RATE_NVOL", "RATE_LNVOL"):
        exact_key = f"SWAPTION/{vol_kind}/{str(ccy).upper()}/{exp}/{term}/ATM"
        surface: list[tuple[float, float, float]] = []
        for q in quotes:
            key = str(q.key).strip().upper()
            if key == exact_key:
                return float(q.value), ("normal" if vol_kind == "RATE_NVOL" else "lognormal")
            parts = key.split("/")
            if len(parts) not in (6, 7):
                continue
            if parts[0] != "SWAPTION" or parts[1] != vol_kind or parts[2] != str(ccy).upper():
                continue
            if len(parts) == 6:
                exp_token, term_token, atm_token = parts[3], parts[4], parts[5]
            else:
                exp_token, term_token, atm_token = parts[4], parts[5], parts[6]
            if atm_token != "ATM":
                continue
            exp_y = _ore_tenor_to_years(exp_token)
            term_y = _ore_tenor_to_years(term_token)
            if exp_y is None or term_y is None:
                continue
            surface.append((exp_y, term_y, float(q.value)))
        if surface:
            best = min(surface, key=lambda x: (x[0] - target_e) ** 2 + (x[1] - target_t) ** 2)
            return float(best[2]), ("normal" if vol_kind == "RATE_NVOL" else "lognormal")
    return None


def _build_quantlib_discount_curve(snapshot: XVASnapshot, p0_disc: Callable[[float], float], ql):
    asof = ql.DateParser.parseISO(snapshot.config.asof)
    cal = ql.TARGET()
    dc = ql.Actual365Fixed()
    dates = [asof]
    dfs = [1.0]
    for months in range(1, 12 * 61 + 1):
        d = cal.advance(asof, ql.Period(int(months), ql.Months))
        t = dc.yearFraction(asof, d)
        dates.append(d)
        dfs.append(float(p0_disc(float(t))))
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, dc, cal))


def _build_quantlib_ibor_index(index_name: str, yts, ql):
    name = str(index_name).strip().upper()
    if name == "EUR-EURIBOR-6M":
        return ql.Euribor6M(yts)
    if name == "EUR-EURIBOR-3M":
        return ql.Euribor3M(yts)
    if name == "USD-LIBOR-3M":
        return ql.USDLibor(ql.Period("3M"), yts)
    if name == "USD-LIBOR-6M":
        return ql.USDLibor(ql.Period("6M"), yts)
    return None


def _infer_underlying_maturity_date(portfolio_xml_path: str, trade_id: str, ql):
    root = ET.parse(portfolio_xml_path).getroot()
    trade = root.find(f"./Trade[@id='{trade_id}']")
    if trade is None:
        return None
    end_dates = []
    for leg in trade.findall("./SwaptionData/LegData"):
        txt = (leg.findtext("./ScheduleData/Rules/EndDate") or "").strip()
        if txt:
            end_dates.append(ql.DateParser.parseISO(_normalize_date(txt)))
    return max(end_dates) if end_dates else None


def _infer_fixed_leg_day_counter(portfolio_xml_path: str, trade_id: str, ql):
    root = ET.parse(portfolio_xml_path).getroot()
    trade = root.find(f"./Trade[@id='{trade_id}']")
    if trade is None:
        return ql.ActualActual(ql.ActualActual.ISDA)
    for leg in trade.findall("./SwaptionData/LegData"):
        if (leg.findtext("./LegType") or "").strip().lower() == "fixed":
            dc = (leg.findtext("./DayCounter") or "").strip().upper().replace(" ", "")
            if dc in ("ACT/ACT", "ACT/ACT(ISDA)", "AAISDA", "ACTUAL/ACTUAL", "ACTUALACTUAL", "ACTUALACTUAL(ISDA)"):
                return ql.ActualActual(ql.ActualActual.ISDA)
            if dc in ("A365", "A365F", "ACT/365", "ACT/365(FIXED)"):
                return ql.Actual365Fixed()
            if dc in ("A360", "ACT/360"):
                return ql.Actual360()
    return ql.ActualActual(ql.ActualActual.ISDA)


def _infer_fixed_leg_tenor(portfolio_xml_path: str, trade_id: str, ql):
    root = ET.parse(portfolio_xml_path).getroot()
    trade = root.find(f"./Trade[@id='{trade_id}']")
    if trade is None:
        return ql.Period("1Y")
    for leg in trade.findall("./SwaptionData/LegData"):
        if (leg.findtext("./LegType") or "").strip().lower() == "fixed":
            tenor = (leg.findtext("./ScheduleData/Rules/Tenor") or "1Y").strip()
            return ql.Period(tenor)
    return ql.Period("1Y")


def _period_to_ore_tenor(period) -> str:
    units = int(period.units())
    length = int(period.length())
    if units == 0:
        return f"{length}D"
    if units == 1:
        return f"{length}W"
    if units == 2:
        return f"{length}M"
    if units == 3:
        return f"{length}Y"
    return str(period)


def _year_fraction_to_period(years: float, ql):
    years_f = float(years)
    nearest_year = int(round(years_f))
    if nearest_year >= 1 and abs(years_f - nearest_year) <= 0.125:
        return ql.Period(nearest_year, ql.Years)
    months = int(round(years_f * 12.0))
    months = max(months, 1)
    if months % 12 == 0:
        return ql.Period(months // 12, ql.Years)
    return ql.Period(months, ql.Months)


def _ore_tenor_to_years(tenor: str) -> float | None:
    txt = str(tenor).strip().upper()
    if not txt:
        return None
    if txt.endswith("Y"):
        return float(txt[:-1])
    if txt.endswith("M"):
        return float(txt[:-1]) / 12.0
    if txt.endswith("W"):
        return float(txt[:-1]) / 52.0
    if txt.endswith("D"):
        return float(txt[:-1]) / 365.0
    return None


def _build_curves(snapshot: XVASnapshot, mapped, trade: Trade, curve_mode: str) -> Dict[str, object]:
    from pythonore.compute import irs_xva_utils, lgm as lgm_mod
    from pythonore.compute import lgm_ir_options
    from pythonore.io import ore_snapshot

    if not isinstance(trade.product, BermudanSwaption):
        raise ValidationError(f"Trade {trade.trade_id} is not a BermudanSwaption")
    ore_xml_path = Path(snapshot.config.source_meta.path or "")
    if not ore_xml_path.exists():
        raise ValidationError("Snapshot is missing the source ore.xml path required for Bermudan pricing")

    tm_root = ET.fromstring(mapped.xml_buffers["todaysmarket.xml"])
    pricing_config = (
        snapshot.config.params.get("market.pricing")
        or snapshot.config.params.get("market.simulation")
        or "default"
    )
    discount_column = ore_snapshot._resolve_discount_column(tm_root, pricing_config, trade.product.ccy.upper())
    forward_column = trade.product.float_index or ""
    if not forward_column:
        raise ValidationError(f"Bermudan trade {trade.trade_id} is missing float index information")

    model_day_counter = _model_day_counter(mapped.xml_buffers["simulation.xml"])
    output_dir = _ore_output_dir(snapshot)
    curves_csv = output_dir / "curves.csv"
    if curve_mode not in ("auto", "ore_output", "market_fit"):
        raise ValidationError(f"Unsupported curve_mode '{curve_mode}'")

    p0_disc = None
    p0_fwd = None
    curve_source = "market_fit"
    disc_fit = None
    fwd_fit = None
    disc_quotes = None
    fwd_quotes = None
    fit_method = "bootstrap_mm_irs_v1"
    if curve_mode in ("auto", "ore_output") and curves_csv.exists():
        try:
            curve_payload = ore_snapshot._load_ore_discount_pairs_by_columns_with_day_counter(
                str(curves_csv),
                [discount_column, forward_column],
                asof_date=snapshot.config.asof,
                day_counter=model_day_counter,
            )
            _, t_disc, df_disc = curve_payload[discount_column]
            _, t_fwd, df_fwd = curve_payload[forward_column]
            p0_disc = irs_xva_utils.build_discount_curve_from_discount_pairs(list(zip(t_disc, df_disc)))
            p0_fwd = irs_xva_utils.build_discount_curve_from_discount_pairs(list(zip(t_fwd, df_fwd)))
            curve_source = "ore_output_curves"
        except Exception as exc:
            if curve_mode == "ore_output":
                raise EngineRunError(f"Failed to load Bermudan curves from ORE output: {exc}") from exc

    if p0_disc is None or p0_fwd is None:
        quote_dicts = [{"key": str(q.key), "value": float(q.value)} for q in snapshot.market.raw_quotes]
        disc_quotes = [
            q for q in quote_dicts if lgm_market._quote_matches_discount_curve(str(q["key"]), trade.product.ccy.upper(), discount_column)
        ]
        fwd_quotes = [
            q
            for q in quote_dicts
            if lgm_market._quote_matches_forward_curve(str(q["key"]), trade.product.ccy.upper(), _float_index_tenor(forward_column))
        ]
        if not disc_quotes:
            raise EngineRunError(f"No discount quotes found for {trade.product.ccy} using source column '{discount_column}'")
        if not fwd_quotes:
            raise EngineRunError(f"No forward quotes found for {trade.product.ccy} tenor '{_float_index_tenor(forward_column)}'")
        disc_fit = ore_snapshot.fit_discount_curves_from_programmatic_quotes(
            snapshot.config.asof,
            disc_quotes,
            fit_method=fit_method,
        ).get(trade.product.ccy.upper())
        fwd_fit = ore_snapshot.fit_discount_curves_from_programmatic_quotes(
            snapshot.config.asof,
            fwd_quotes,
            fit_method=fit_method,
        ).get(trade.product.ccy.upper())
        if not disc_fit or not fwd_fit:
            raise EngineRunError("Failed to bootstrap Bermudan discount/forward curves from ORE market quotes")
        p0_disc = ore_snapshot.build_discount_curve_from_discount_pairs(list(zip(disc_fit["times"], disc_fit["dfs"])))
        p0_fwd = ore_snapshot.build_discount_curve_from_discount_pairs(list(zip(fwd_fit["times"], fwd_fit["dfs"])))
        curve_source = "ore_quote_fit"

    return {
        "p0_disc": p0_disc,
        "p0_fwd": p0_fwd,
        "curve_source": curve_source,
        "discount_column": discount_column,
        "forward_column": forward_column,
        "model_day_counter": model_day_counter,
        "portfolio_xml_path": _portfolio_xml_path(snapshot),
        "irs_utils": irs_xva_utils,
        "ore_snapshot_mod": ore_snapshot,
        "lgm_mod": lgm_mod,
        "lgm_ir_options": lgm_ir_options,
        "disc_fit": disc_fit,
        "fwd_fit": fwd_fit,
        "disc_quotes": disc_quotes,
        "fwd_quotes": fwd_quotes,
        "fit_method": fit_method,
    }


def _pricing_grid(legs: Dict[str, np.ndarray], exercise_times: np.ndarray) -> np.ndarray:
    parts = [np.array([0.0], dtype=float), np.asarray(exercise_times, dtype=float)]
    for key in ("fixed_pay_time", "float_pay_time", "float_fixing_time"):
        vals = np.asarray(legs.get(key, np.array([], dtype=float)), dtype=float)
        if vals.size:
            parts.append(vals[vals >= 0.0])
    return np.unique(np.concatenate(parts))


def _exercise_times(
    asof: str,
    exercise_dates: Sequence[str],
    model_day_counter: str,
    ore_snapshot_mod,
    *,
    simulation_xml_text: str = "",
) -> np.ndarray:
    raw = np.asarray(
        [
            ore_snapshot_mod._year_fraction_from_day_counter(asof, _normalize_date(d), model_day_counter)
            for d in exercise_dates
        ],
        dtype=float,
    )
    sim_grid = _simulation_grid_times_from_xml_text(simulation_xml_text)
    if sim_grid is None or sim_grid.size == 0:
        return raw
    idx = np.searchsorted(sim_grid, raw, side="left")
    idx = np.clip(idx, 0, sim_grid.size - 1)
    return np.asarray(sim_grid[idx], dtype=float)


def _simulation_grid_times_from_xml_text(simulation_xml_text: str) -> np.ndarray | None:
    text = str(simulation_xml_text or "").strip()
    if not text:
        return None
    root = ET.fromstring(text)
    grid_text = (root.findtext("./Parameters/Grid") or "").strip()
    if "," not in grid_text:
        return None
    count_text, tenor_text = [x.strip() for x in grid_text.split(",", 1)]
    try:
        count = int(count_text)
    except (TypeError, ValueError):
        return None
    tenor = tenor_text.upper()
    if tenor.endswith("Y"):
        step = float(tenor[:-1])
    elif tenor.endswith("M"):
        step = float(tenor[:-1]) / 12.0
    elif tenor.endswith("W"):
        step = float(tenor[:-1]) / 52.0
    elif tenor.endswith("D"):
        step = float(tenor[:-1]) / 365.0
    else:
        return None
    return np.asarray([i * step for i in range(count + 1)], dtype=float)


def _exercise_sign(product: BermudanSwaption) -> float:
    option_type = str(getattr(product, "option_type", "Call")).strip().lower()
    long_short = str(product.long_short).strip().lower()
    # The underlying swap cashflows already encode payer/receiver orientation.
    # Bermudan payoff direction should therefore follow the option type:
    # Call -> max(swap, 0), Put -> max(-swap, 0).
    sign = 1.0 if option_type == "call" else -1.0
    if long_short == "short":
        sign *= -1.0
    return sign


def _relevant_market_factors(snapshot: XVASnapshot, trade: Trade) -> List[str]:
    if not isinstance(trade.product, BermudanSwaption):
        return []
    mapped = map_snapshot(snapshot)
    tm_root = ET.fromstring(mapped.xml_buffers["todaysmarket.xml"])
    pricing_config = (
        snapshot.config.params.get("market.pricing")
        or snapshot.config.params.get("market.simulation")
        or "default"
    )
    from pythonore.io import ore_snapshot

    discount_column = ore_snapshot._resolve_discount_column(tm_root, pricing_config, trade.product.ccy.upper())
    tenor = _float_index_tenor(trade.product.float_index)
    out = []
    for q in snapshot.market.raw_quotes:
        key = str(q.key)
        if lgm_market._quote_matches_discount_curve(key, trade.product.ccy.upper(), discount_column):
            out.append(key)
            continue
        if lgm_market._quote_matches_forward_curve(key, trade.product.ccy.upper(), tenor):
            out.append(key)
    return sorted(dict.fromkeys(out))


def _price_bermudan_frozen(
    ctx: Dict[str, object],
    p0_disc: Callable[[float], float],
    p0_fwd: Callable[[float], float],
) -> float:
    curve_bundle = ctx["curve_bundle"]
    pricing_method = str(ctx["pricing_method"])
    model = ctx["model"]
    bermudan_def = ctx["bermudan_def"]
    basis_degree = int(ctx["basis_degree"])
    if pricing_method == "lsmc":
        lgm_mod = curve_bundle["lgm_mod"]
        lgm_ir_options = curve_bundle["lgm_ir_options"]
        times = ctx["times"]
        x_paths = lgm_mod.simulate_lgm_measure(
            model,
            times,
            n_paths=int(ctx["num_paths"]),
            rng=np.random.default_rng(int(ctx["seed"])),
            x0=0.0,
        )
        result = lgm_ir_options.bermudan_lsmc_result(
            model,
            p0_disc,
            p0_fwd,
            bermudan_def,
            times,
            x_paths,
            basis_degree=basis_degree,
            itm_only=True,
        )
        return float(np.mean(result.npv_paths[0, :]))

    lgm_ir_options = curve_bundle["lgm_ir_options"]
    engine_kwargs = _backward_engine_kwargs(ctx["snapshot"])
    result = lgm_ir_options.bermudan_backward_price(
        model,
        p0_disc,
        p0_fwd,
        bermudan_def,
        n_grid=max(61, basis_degree * 40 + 1),
        convolution_sx=float(engine_kwargs["sx"]),
        convolution_nx=int(engine_kwargs["nx"]),
        convolution_sy=float(engine_kwargs["sy"]),
        convolution_ny=int(engine_kwargs["ny"]),
    )
    return float(result.price)


def _fast_curve_jacobian_sensitivities(
    snapshot: XVASnapshot,
    ctx: Dict[str, object],
    selected: Sequence[str],
    shift_size: float,
) -> List[BermudanPvSensitivity]:
    curve_bundle = ctx["curve_bundle"]
    if str(curve_bundle["curve_source"]) != "ore_quote_fit":
        raise ValidationError("fast_curve_jacobian sensitivities require curve_mode='market_fit' with fitted quote curves")

    disc_fit = curve_bundle.get("disc_fit")
    fwd_fit = curve_bundle.get("fwd_fit")
    disc_quotes = curve_bundle.get("disc_quotes") or []
    fwd_quotes = curve_bundle.get("fwd_quotes") or []
    if not disc_fit or not fwd_fit:
        raise ValidationError("fast_curve_jacobian sensitivities require fitted discount and forward curves")

    disc_node_times = _nonzero_curve_nodes(disc_fit["times"])
    fwd_node_times = _nonzero_curve_nodes(fwd_fit["times"])
    base_disc = curve_bundle["p0_disc"]
    base_fwd = curve_bundle["p0_fwd"]

    disc_node_delta = _curve_node_deltas(ctx, base_disc, base_fwd, disc_node_times, curve_side="discount", shift_size=shift_size)
    fwd_node_delta = _curve_node_deltas(ctx, base_disc, base_fwd, fwd_node_times, curve_side="forward", shift_size=shift_size)
    disc_jac = _quote_to_zero_node_jacobian(
        snapshot.config.asof,
        disc_quotes,
        disc_fit,
        ccy=str(ctx["trade"].product.ccy).upper(),
        shift_size=shift_size,
        fit_method=str(curve_bundle.get("fit_method", "bootstrap_mm_irs_v1")),
    )
    fwd_jac = _quote_to_zero_node_jacobian(
        snapshot.config.asof,
        fwd_quotes,
        fwd_fit,
        ccy=str(ctx["trade"].product.ccy).upper(),
        shift_size=shift_size,
        fit_method=str(curve_bundle.get("fit_method", "bootstrap_mm_irs_v1")),
    )

    base_price = float(ctx["result"].price)
    selected_set = set(str(f) for f in selected)
    entries: List[BermudanPvSensitivity] = []

    disc_quote_keys = [str(q["key"]) for q in disc_quotes]
    disc_base_values = [float(q["value"]) for q in disc_quotes]
    for j, factor in enumerate(disc_quote_keys):
        if factor not in selected_set:
            continue
        delta = float(np.dot(disc_node_delta, disc_jac[:, j]))
        entries.append(
            BermudanPvSensitivity(
                factor=factor,
                shift_size=float(shift_size),
                base_quote_value=disc_base_values[j],
                base_price=base_price,
                bumped_up_price=float("nan"),
                bumped_down_price=float("nan"),
                delta=delta,
            )
        )

    fwd_quote_keys = [str(q["key"]) for q in fwd_quotes]
    fwd_base_values = [float(q["value"]) for q in fwd_quotes]
    for j, factor in enumerate(fwd_quote_keys):
        if factor not in selected_set:
            continue
        delta = float(np.dot(fwd_node_delta, fwd_jac[:, j]))
        entries.append(
            BermudanPvSensitivity(
                factor=factor,
                shift_size=float(shift_size),
                base_quote_value=fwd_base_values[j],
                base_price=base_price,
                bumped_up_price=float("nan"),
                bumped_down_price=float("nan"),
                delta=delta,
            )
        )

    entries.sort(key=lambda x: x.factor)
    return entries


def _curve_node_deltas(
    ctx: Dict[str, object],
    base_disc: Callable[[float], float],
    base_fwd: Callable[[float], float],
    node_times: np.ndarray,
    *,
    curve_side: str,
    shift_size: float,
) -> np.ndarray:
    deltas = np.zeros(node_times.size, dtype=float)
    for i, node_t in enumerate(node_times):
        shifts = np.zeros(node_times.size, dtype=float)
        shifts[i] = float(shift_size)
        if curve_side == "discount":
            up_disc = lgm_market._build_zero_rate_shocked_curve(base_disc, list(node_times), list(shifts))
            dn_disc = lgm_market._build_zero_rate_shocked_curve(base_disc, list(node_times), list(-shifts))
            up = _price_bermudan_frozen(ctx, up_disc, base_fwd)
            dn = _price_bermudan_frozen(ctx, dn_disc, base_fwd)
        else:
            up_fwd = lgm_market._build_zero_rate_shocked_curve(base_fwd, list(node_times), list(shifts))
            dn_fwd = lgm_market._build_zero_rate_shocked_curve(base_fwd, list(node_times), list(-shifts))
            up = _price_bermudan_frozen(ctx, base_disc, up_fwd)
            dn = _price_bermudan_frozen(ctx, base_disc, dn_fwd)
        deltas[i] = (up - dn) / (2.0 * float(shift_size))
    return deltas


def _nonzero_curve_nodes(times: Sequence[float]) -> np.ndarray:
    arr = np.asarray(times, dtype=float)
    return arr[arr > 1.0e-12]


def _quote_to_zero_node_jacobian(
    asof_date: str,
    quote_dicts: Sequence[dict],
    base_fit: dict,
    *,
    ccy: str,
    shift_size: float,
    fit_method: str,
) -> np.ndarray:
    from pythonore.io import ore_snapshot

    base_nodes = np.asarray(base_fit["zero_rates"], dtype=float)[1:]
    jac = np.zeros((base_nodes.size, len(quote_dicts)), dtype=float)
    if base_nodes.size == 0 or not quote_dicts:
        return jac

    for j, q in enumerate(quote_dicts):
        key = str(q["key"])
        up_quotes = [{**qq, "value": float(qq["value"]) + (float(shift_size) if str(qq["key"]) == key else 0.0)} for qq in quote_dicts]
        dn_quotes = [{**qq, "value": float(qq["value"]) - (float(shift_size) if str(qq["key"]) == key else 0.0)} for qq in quote_dicts]
        up_fit = ore_snapshot.fit_discount_curves_from_programmatic_quotes(
            asof_date,
            up_quotes,
            fit_method=fit_method,
        )
        dn_fit = ore_snapshot.fit_discount_curves_from_programmatic_quotes(
            asof_date,
            dn_quotes,
            fit_method=fit_method,
        )
        if ccy not in up_fit or ccy not in dn_fit:
            continue
        up_nodes = np.asarray(up_fit[ccy]["zero_rates"], dtype=float)[1:]
        dn_nodes = np.asarray(dn_fit[ccy]["zero_rates"], dtype=float)[1:]
        jac[:, j] = (up_nodes - dn_nodes) / (2.0 * float(shift_size))
    return jac


def _bump_quotes(quotes: Iterable[MarketQuote], factor: str, shift: float) -> tuple[MarketQuote, ...]:
    bumped = []
    for q in quotes:
        if q.key == factor:
            bumped.append(replace(q, value=float(q.value) + float(shift)))
        else:
            bumped.append(q)
    return tuple(bumped)


def _ore_output_dir(snapshot: XVASnapshot) -> Path:
    ore_xml_path = Path(snapshot.config.source_meta.path or "")
    output_path = snapshot.config.params.get("outputPath", "Output")
    return (ore_xml_path.parent.parent / output_path).resolve()


def _portfolio_xml_path(snapshot: XVASnapshot) -> str:
    path = snapshot.source_meta.get("portfolio")
    if path is None or not path.path:
        raise ValidationError("Snapshot is missing source path metadata for portfolio.xml")
    return str(Path(path.path).resolve())


def _model_day_counter(simulation_xml_text: str) -> str:
    from pythonore.io import ore_snapshot

    try:
        root = ET.fromstring(simulation_xml_text)
        raw = (root.findtext("./DayCounter") or "A365F").strip()
        return ore_snapshot._normalize_day_counter_name(raw)
    except Exception:
        return "A365F"


def _float_index_tenor(index_name: str) -> str:
    idx = str(index_name).strip().upper()
    return idx.split("-")[-1].upper() if "-" in idx else ""


def _normalize_date(value: str) -> str:
    s = str(value).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
    return s


def _load_trade_npv(npv_csv: Path, trade_id: str) -> float:
    with open(npv_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tid_key = "TradeId" if reader.fieldnames and "TradeId" in reader.fieldnames else "#TradeId"
        for row in reader:
            if row.get(tid_key, "").strip() == trade_id:
                return float(row["NPV"])
    raise ValueError(f"Trade '{trade_id}' not found in {npv_csv}")


def _load_generic_sensitivity_rows(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count_csv_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _default_amc_output_dir(snapshot: XVASnapshot) -> Optional[Path]:
    portfolio_path = snapshot.source_meta.get("portfolio")
    if portfolio_path is None or not portfolio_path.path:
        return None
    p = Path(portfolio_path.path).resolve()
    if "Examples/AmericanMonteCarlo/Input" in p.as_posix():
        return p.parents[1] / "ExpectedOutput" / "classic"
    return None


class _temp_xml_file:
    def __init__(self, xml_text: str):
        self.xml_text = xml_text
        self.path: Optional[Path] = None

    def __enter__(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        self.path = Path(path)
        Path(path).write_text(self.xml_text, encoding="utf-8")
        return str(path)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.path is not None:
                self.path.unlink()
        except OSError:
            pass
