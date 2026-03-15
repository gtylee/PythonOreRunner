from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union


Metric = Literal["CVA", "DVA", "FVA", "MVA"]
DimModel = Literal["Regression", "DeltaVaR", "DeltaGammaNormalVaR", "DeltaGammaVaR", "DynamicIM", "SimmAnalytic"]
ProductType = Literal["IRS", "FXForward", "EuropeanOption", "BermudanSwaption", "Generic"]


@dataclass(frozen=True)
class SourceMeta:
    origin: str
    path: Optional[str] = None


@dataclass(frozen=True)
class CurvePoint:
    tenor: str
    value: float


@dataclass(frozen=True)
class Curve:
    curve_id: str
    points: Tuple[CurvePoint, ...] = ()


@dataclass(frozen=True)
class FXQuote:
    pair: str
    value: float


@dataclass(frozen=True)
class FXQuotes:
    quotes: Tuple[FXQuote, ...] = ()


@dataclass(frozen=True)
class CreditCurve:
    name: str
    points: Tuple[CurvePoint, ...] = ()


@dataclass(frozen=True)
class MarketQuote:
    date: str
    key: str
    value: float

    def __post_init__(self) -> None:
        _validate_date_like(self.date, "MarketQuote.date")


@dataclass(frozen=True)
class FixingPoint:
    date: str
    index: str
    value: float

    def __post_init__(self) -> None:
        _validate_date_like(self.date, "FixingPoint.date")


@dataclass(frozen=True)
class MarketData:
    asof: str
    raw_quotes: Tuple[MarketQuote, ...] = ()
    yield_curves: Dict[str, Curve] = field(default_factory=dict)
    fx_quotes: FXQuotes = field(default_factory=FXQuotes)
    credit_curves: Dict[str, CreditCurve] = field(default_factory=dict)
    source_meta: SourceMeta = field(default_factory=lambda: SourceMeta(origin="dataclass"))

    def __post_init__(self) -> None:
        _validate_date_like(self.asof, "MarketData.asof")


@dataclass(frozen=True)
class FixingsData:
    points: Tuple[FixingPoint, ...] = ()
    source_meta: SourceMeta = field(default_factory=lambda: SourceMeta(origin="dataclass"))


@dataclass(frozen=True)
class Product:
    product_type: ProductType


@dataclass(frozen=True)
class IRS(Product):
    ccy: str
    notional: float
    fixed_rate: float
    maturity_years: float
    pay_fixed: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fixed_leg_tenor: str = "6M"
    float_leg_tenor: str = "3M"
    fixed_day_counter: Optional[str] = None
    float_day_counter: Optional[str] = None
    calendar: Optional[str] = None
    fixed_payment_convention: str = "MF"
    float_payment_convention: str = "MF"
    fixed_schedule_convention: Optional[str] = None
    float_schedule_convention: Optional[str] = None
    fixed_term_convention: Optional[str] = None
    float_term_convention: Optional[str] = None
    fixed_schedule_rule: str = "Forward"
    float_schedule_rule: str = "Forward"
    end_of_month: bool = False
    float_index: str = ""
    fixing_days: int = 2
    float_spread: float = 0.0
    product_type: ProductType = field(init=False, default="IRS")

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError("IRS.notional must be > 0")
        if self.maturity_years <= 0:
            raise ValueError("IRS.maturity_years must be > 0")
        if self.start_date is not None:
            _validate_date_like(self.start_date, "IRS.start_date")
        if self.end_date is not None:
            _validate_date_like(self.end_date, "IRS.end_date")
        if self.fixing_days < 0:
            raise ValueError("IRS.fixing_days must be >= 0")


@dataclass(frozen=True)
class FXForward(Product):
    pair: str
    notional: float
    strike: float
    maturity_years: float
    buy_base: bool = True
    value_date: Optional[str] = None
    product_type: ProductType = field(init=False, default="FXForward")

    def __post_init__(self) -> None:
        if len(self.pair) != 6:
            raise ValueError("FXForward.pair must be 6 chars, e.g. EURUSD")
        if self.notional <= 0:
            raise ValueError("FXForward.notional must be > 0")
        if self.maturity_years <= 0:
            raise ValueError("FXForward.maturity_years must be > 0")
        if self.value_date is not None:
            _validate_date_like(self.value_date, "FXForward.value_date")


@dataclass(frozen=True)
class EuropeanOption(Product):
    underlying: str
    kind: Literal["call", "put"]
    strike: float
    notional: float
    maturity_years: float
    product_type: ProductType = field(init=False, default="EuropeanOption")

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError("EuropeanOption.notional must be > 0")
        if self.maturity_years <= 0:
            raise ValueError("EuropeanOption.maturity_years must be > 0")


@dataclass(frozen=True)
class GenericProduct(Product):
    payload: Dict[str, Any] = field(default_factory=dict)
    product_type: ProductType = field(init=False, default="Generic")


@dataclass(frozen=True)
class BermudanSwaption(Product):
    ccy: str
    notional: float
    fixed_rate: float
    maturity_years: float
    pay_fixed: bool
    exercise_dates: Tuple[str, ...]
    settlement: str = "Physical"
    option_type: str = "Call"
    long_short: str = "Long"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fixed_leg_tenor: str = "1Y"
    float_leg_tenor: str = "6M"
    fixed_day_counter: Optional[str] = None
    float_day_counter: Optional[str] = None
    calendar: Optional[str] = None
    fixed_payment_convention: str = "Following"
    float_payment_convention: str = "ModifiedFollowing"
    fixed_schedule_convention: Optional[str] = None
    float_schedule_convention: Optional[str] = None
    fixed_term_convention: Optional[str] = None
    float_term_convention: Optional[str] = None
    fixed_schedule_rule: str = "Forward"
    float_schedule_rule: str = "Forward"
    end_of_month: bool = False
    float_index: str = ""
    fixing_days: int = 2
    float_spread: float = 0.0
    payoff_at_expiry: bool = False
    is_in_arrears: bool = False
    product_type: ProductType = field(init=False, default="BermudanSwaption")

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError("BermudanSwaption.notional must be > 0")
        if self.maturity_years <= 0:
            raise ValueError("BermudanSwaption.maturity_years must be > 0")
        if not self.exercise_dates:
            raise ValueError("BermudanSwaption.exercise_dates must not be empty")
        for d in self.exercise_dates:
            _validate_date_like(d, "BermudanSwaption.exercise_dates")
        if self.start_date is not None:
            _validate_date_like(self.start_date, "BermudanSwaption.start_date")
        if self.end_date is not None:
            _validate_date_like(self.end_date, "BermudanSwaption.end_date")
        if self.fixing_days < 0:
            raise ValueError("BermudanSwaption.fixing_days must be >= 0")


ProductSpec = Union[IRS, FXForward, EuropeanOption, BermudanSwaption, GenericProduct]


@dataclass(frozen=True)
class Trade:
    trade_id: str
    counterparty: str
    netting_set: str
    trade_type: str
    product: ProductSpec
    additional_fields: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trade_id:
            raise ValueError("Trade.trade_id is required")
        if not self.counterparty:
            raise ValueError("Trade.counterparty is required")
        if not self.netting_set:
            raise ValueError("Trade.netting_set is required")


@dataclass(frozen=True)
class Portfolio:
    trades: Tuple[Trade, ...]
    source_meta: SourceMeta = field(default_factory=lambda: SourceMeta(origin="dataclass"))

    def __post_init__(self) -> None:
        trade_ids = [t.trade_id for t in self.trades]
        if len(set(trade_ids)) != len(trade_ids):
            raise ValueError("Portfolio.trades contains duplicate trade_id")


@dataclass(frozen=True)
class NettingSet:
    netting_set_id: str
    counterparty: Optional[str] = None
    active_csa: bool = False
    csa_currency: Optional[str] = None
    margin_period_of_risk: Optional[str] = None
    threshold_pay: Optional[float] = None
    threshold_receive: Optional[float] = None
    mta_pay: Optional[float] = None
    mta_receive: Optional[float] = None


@dataclass(frozen=True)
class MporConfig:
    enabled: bool = False
    mpor_years: float = 0.0
    mpor_days: int = 0
    closeout_lag_period: str = ""
    sticky: bool = True
    cashflow_mode: str = "NonePay"
    source: str = "disabled"


@dataclass(frozen=True)
class NettingConfig:
    netting_sets: Dict[str, NettingSet] = field(default_factory=dict)
    source_meta: SourceMeta = field(default_factory=lambda: SourceMeta(origin="dataclass"))


@dataclass(frozen=True)
class CollateralBalance:
    netting_set_id: str
    currency: str
    initial_margin: float = 0.0
    variation_margin: float = 0.0


@dataclass(frozen=True)
class CollateralConfig:
    balances: Tuple[CollateralBalance, ...] = ()
    source_meta: SourceMeta = field(default_factory=lambda: SourceMeta(origin="dataclass"))


@dataclass(frozen=True)
class PricingEngineConfig:
    model: str = "BlackScholes"
    npv_engine: str = "DiscountedCashflows"
    fx_model: Optional[str] = None
    fx_engine: Optional[str] = None
    swap_model: str = "DiscountedCashflows"
    swap_engine: str = "DiscountingSwapEngine"
    bermudan_model: str = "LGM"
    bermudan_engine: str = "Gaussian1dNonstandardSwaptionEngine"
    bermudan_reversion: float = 0.03
    bermudan_volatility: float = 0.01
    bermudan_shift_horizon: float = 0.0
    bermudan_sx: float = 3.0
    bermudan_nx: int = 10
    bermudan_sy: float = 3.0
    bermudan_ny: int = 10


@dataclass(frozen=True)
class TodaysMarketConfig:
    market_id: str = "default"
    discount_curve: str = "EUR-EONIA"
    fx_pairs: Tuple[str, ...] = ("EURUSD",)
    yield_curves_id: Optional[str] = None
    discounting_curves_id: Optional[str] = None
    index_forwarding_curves_id: Optional[str] = None
    fx_spots_id: Optional[str] = None
    fx_volatilities_id: Optional[str] = None
    swaption_volatilities_id: Optional[str] = None
    default_curves_id: Optional[str] = None


@dataclass(frozen=True)
class CurveConfig:
    curve_id: str
    currency: str
    tenors: Tuple[str, ...] = ("1Y", "2Y", "5Y", "10Y")


@dataclass(frozen=True)
class SimulationConfig:
    samples: int = 5000
    seed: int = 42
    dates: Tuple[str, ...] = ()
    strict_template: bool = False
    discretization: str = "Exact"
    sequence: str = "SobolBrownianBridge"
    scenario: str = "Simple"
    closeout_lag: str = "2W"
    mpor_mode: str = "StickyDate"
    day_counter: str = "A365F"
    calendar: Optional[str] = None
    xva_cg_dynamic_im: bool = False
    xva_cg_dynamic_im_step_size: int = 1
    xva_cg_regression_order_dynamic_im: Optional[int] = None
    xva_cg_regression_report_time_steps_dynamic_im: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("SimulationConfig.samples must be > 0")
        if self.xva_cg_dynamic_im_step_size <= 0:
            raise ValueError("SimulationConfig.xva_cg_dynamic_im_step_size must be > 0")


@dataclass(frozen=True)
class SimulationMarketConfig:
    base_currency: str = "EUR"
    currencies: Tuple[str, ...] = ("EUR", "USD")
    indices: Tuple[str, ...] = ("EUR-ESTER", "USD-SOFR")
    default_curve_names: Tuple[str, ...] = ("BANK", "CPTY_A")
    fx_pairs: Tuple[str, ...] = ("USDEUR",)
    yield_curve_tenors: Tuple[str, ...] = ("3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y")
    yield_curve_interpolation: str = "LogLinear"
    yield_curve_extrapolation: bool = True
    default_curve_tenors: Tuple[str, ...] = ("1Y", "2Y", "5Y", "10Y")
    default_simulate_survival_probabilities: bool = True
    default_simulate_recovery_rates: bool = True
    default_curve_calendar: str = "TARGET"
    default_curve_extrapolation: str = "FlatZero"
    swaption_simulate: bool = False
    swaption_reaction_to_time_decay: str = "ForwardVariance"
    swaption_expiries: Tuple[str, ...] = ("6M", "1Y", "2Y", "3Y", "5Y", "10Y", "12Y", "15Y", "20Y")
    swaption_terms: Tuple[str, ...] = ("1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
    fxvol_simulate: bool = False
    fxvol_reaction_to_time_decay: str = "ForwardVariance"
    fxvol_expiries: Tuple[str, ...] = ("1Y", "2Y", "5Y")


@dataclass(frozen=True)
class CrossAssetModelConfig:
    domestic_ccy: str = "EUR"
    currencies: Tuple[str, ...] = ("EUR", "USD")
    ir_model_ccys: Tuple[str, ...] = ("EUR", "USD")
    fx_model_ccys: Tuple[str, ...] = ("USD",)
    bootstrap_tolerance: float = 0.0001
    ir_calibration_type: str = "Bootstrap"
    ir_volatility: float = 0.01
    ir_reversion: float = 0.0
    ir_shift_horizon: float = 20.0
    ir_scaling: float = 1.0
    ir_calibration_expiries: Tuple[str, ...] = ("1Y",)
    ir_calibration_terms: Tuple[str, ...] = ("5Y",)
    fx_calibration_type: str = "Bootstrap"
    fx_sigma: float = 0.1
    fx_calibration_expiries: Tuple[str, ...] = ("1Y",)
    correlations: Tuple[Tuple[str, str, float], ...] = (
        ("IR:EUR", "IR:USD", 0.0),
        ("IR:EUR", "FX:USDEUR", 0.0),
        ("IR:USD", "FX:USDEUR", 0.0),
    )


@dataclass(frozen=True)
class XVAAnalyticConfig:
    exposure_allocation_method: str = "None"
    exposure_observation_model: str = "Disable"
    pfe_quantile: float = 0.95
    collateral_calculation_type: str = "Symmetric"
    marginal_allocation_limit: float = 1.0
    exercise_next_break: bool = False
    scenario_gen_type: Optional[str] = None
    netting_set_ids: Tuple[str, ...] = ()
    full_initial_collateralisation: Optional[bool] = None
    flip_view_xva: Optional[bool] = None
    collateral_floor_enabled: Optional[bool] = None
    dim_model: Optional[DimModel] = None
    dim_quantile: Optional[float] = None
    dim_horizon_calendar_days: Optional[int] = None
    dim_regression_order: Optional[int] = None
    dim_regressors: Optional[str] = None
    dim_evolution_file: Optional[str] = None
    dim_regression_files: Optional[str] = None
    dim_output_grid_points: Optional[str] = None
    dim_output_netting_set: Optional[str] = None
    dim_local_regression_evaluations: Optional[int] = None
    dim_local_regression_bandwidth: Optional[float] = None
    flip_view_borrowing_curve_postfix: Optional[str] = None
    flip_view_lending_curve_postfix: Optional[str] = None
    cva_enabled: Optional[bool] = None
    dva_enabled: Optional[bool] = None
    fva_enabled: Optional[bool] = None
    mva_enabled: Optional[bool] = None
    colva_enabled: Optional[bool] = None
    dim_enabled: Optional[bool] = None
    dynamic_credit_enabled: Optional[bool] = None
    dva_name: Optional[str] = None
    fva_borrowing_curve: Optional[str] = None
    fva_lending_curve: Optional[str] = None


@dataclass(frozen=True)
class CreditEntityConfig:
    name: str
    factor_loading: float = 0.4898979485566356
    transition_matrix: str = "TransitionMatrix_1"
    initial_state: int = 4


@dataclass(frozen=True)
class CreditSimulationConfig:
    enabled: bool = False
    transition_matrix_name: str = "TransitionMatrix_1"
    netting_set_ids: Tuple[str, ...] = ()
    entities: Tuple[CreditEntityConfig, ...] = ()
    paths: int = 1000
    seed: int = 42
    evaluation: str = "Analytic"
    credit_mode: str = "Migration"
    loan_exposure_mode: str = "Value"
    market_risk: bool = False
    credit_risk: bool = True
    double_default: bool = True
    zero_market_pnl: bool = False


@dataclass(frozen=True)
class ConventionsConfig:
    day_counter: str = "A365"
    calendar: str = "TARGET"
    yield_curve_day_counter: str = "A365"
    cds_day_counter: str = "Actual/365 (Fixed)"
    cds_conventions: str = "CDS-STANDARD-CONVENTIONS"


@dataclass(frozen=True)
class CounterpartyConfig:
    ids: Tuple[str, ...] = ()
    curve_currencies: Dict[str, str] = field(default_factory=dict)
    credit_qualities: Dict[str, str] = field(default_factory=dict)
    bacva_risk_weights: Dict[str, float] = field(default_factory=dict)
    saccr_risk_weights: Dict[str, float] = field(default_factory=dict)
    sacva_risk_buckets: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    pricing_engine: PricingEngineConfig = field(default_factory=PricingEngineConfig)
    todays_market: TodaysMarketConfig = field(default_factory=TodaysMarketConfig)
    curve_configs: Tuple[CurveConfig, ...] = ()
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    simulation_market: SimulationMarketConfig = field(default_factory=SimulationMarketConfig)
    cross_asset_model: CrossAssetModelConfig = field(default_factory=CrossAssetModelConfig)
    xva_analytic: XVAAnalyticConfig = field(default_factory=XVAAnalyticConfig)
    credit_simulation: CreditSimulationConfig = field(default_factory=CreditSimulationConfig)
    conventions: ConventionsConfig = field(default_factory=ConventionsConfig)
    counterparties: CounterpartyConfig = field(default_factory=CounterpartyConfig)
    store_sensis: bool = False
    curve_sensi_grid: Tuple[float, ...] = ()
    vega_sensi_grid: Tuple[float, ...] = ()


@dataclass(frozen=True)
class XVAConfig:
    asof: str
    base_currency: str
    analytics: Tuple[Metric, ...] = ("CVA", "DVA", "FVA", "MVA")
    num_paths: int = 5000
    horizon_years: int = 5
    params: Dict[str, str] = field(default_factory=dict)
    xml_buffers: Dict[str, str] = field(default_factory=dict)
    runtime: Optional[RuntimeConfig] = None
    mpor: MporConfig = field(default_factory=MporConfig)
    source_meta: SourceMeta = field(default_factory=lambda: SourceMeta(origin="dataclass"))

    def __post_init__(self) -> None:
        _validate_date_like(self.asof, "XVAConfig.asof")
        if self.num_paths <= 0:
            raise ValueError("XVAConfig.num_paths must be > 0")
        if self.horizon_years <= 0:
            raise ValueError("XVAConfig.horizon_years must be > 0")


@dataclass(frozen=True)
class XVASnapshot:
    market: MarketData
    fixings: FixingsData
    portfolio: Portfolio
    config: XVAConfig
    netting: NettingConfig = field(default_factory=NettingConfig)
    collateral: CollateralConfig = field(default_factory=CollateralConfig)
    source_meta: Dict[str, SourceMeta] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XVASnapshot":
        return _snapshot_from_dict(data)

    def stable_key(self) -> str:
        payload = self.to_dict()
        stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return str(abs(hash(stable)))


def _parse_product(data: Dict[str, Any]) -> ProductSpec:
    t = data.get("product_type")
    if t == "IRS":
        return IRS(
            **{
                k: data[k]
                for k in (
                    "ccy",
                    "notional",
                    "fixed_rate",
                    "maturity_years",
                    "pay_fixed",
                    "start_date",
                    "end_date",
                    "fixed_leg_tenor",
                    "float_leg_tenor",
                    "fixed_day_counter",
                    "float_day_counter",
                    "calendar",
                    "fixed_payment_convention",
                    "float_payment_convention",
                    "fixed_schedule_convention",
                    "float_schedule_convention",
                    "fixed_term_convention",
                    "float_term_convention",
                    "fixed_schedule_rule",
                    "float_schedule_rule",
                    "end_of_month",
                    "float_index",
                    "fixing_days",
                    "float_spread",
                )
                if k in data
            }
        )
    if t == "FXForward":
        return FXForward(**{k: data[k] for k in ("pair", "notional", "strike", "maturity_years", "buy_base", "value_date") if k in data})
    if t == "EuropeanOption":
        return EuropeanOption(**{k: data[k] for k in ("underlying", "kind", "strike", "notional", "maturity_years") if k in data})
    if t == "BermudanSwaption":
        return BermudanSwaption(
            **{
                k: data[k]
                for k in (
                    "ccy",
                    "notional",
                    "fixed_rate",
                    "maturity_years",
                    "pay_fixed",
                    "exercise_dates",
                    "settlement",
                    "option_type",
                    "long_short",
                    "start_date",
                    "end_date",
                    "fixed_leg_tenor",
                    "float_leg_tenor",
                    "fixed_day_counter",
                    "float_day_counter",
                    "calendar",
                    "fixed_payment_convention",
                    "float_payment_convention",
                    "fixed_schedule_convention",
                    "float_schedule_convention",
                    "fixed_term_convention",
                    "float_term_convention",
                    "fixed_schedule_rule",
                    "float_schedule_rule",
                    "end_of_month",
                    "float_index",
                    "fixing_days",
                    "float_spread",
                    "payoff_at_expiry",
                    "is_in_arrears",
                )
                if k in data
            }
        )
    return GenericProduct(payload=data.get("payload", {}))


def _snapshot_from_dict(data: Dict[str, Any]) -> XVASnapshot:
    for required_key in ("market", "fixings", "portfolio", "config"):
        if required_key not in data:
            raise ValueError(
                f"Snapshot dict is missing required key: '{required_key}'. "
                f"Keys present: {sorted(data.keys())}. "
                "Fix: ensure the dict was produced by XVASnapshot.to_dict() or "
                "provide all four required top-level keys."
            )
    market = data["market"]
    fixings = data["fixings"]
    portfolio = data["portfolio"]
    config = data["config"]
    netting = data.get("netting", {})
    collateral = data.get("collateral", {})

    mk = MarketData(
        asof=market["asof"],
        raw_quotes=tuple(MarketQuote(**q) for q in market.get("raw_quotes", [])),
        source_meta=SourceMeta(**market.get("source_meta", {"origin": "dataclass"})),
    )
    fx = FixingsData(
        points=tuple(FixingPoint(**p) for p in fixings.get("points", [])),
        source_meta=SourceMeta(**fixings.get("source_meta", {"origin": "dataclass"})),
    )
    trades: List[Trade] = []
    for t in portfolio.get("trades", []):
        trades.append(
            Trade(
                trade_id=t["trade_id"],
                counterparty=t["counterparty"],
                netting_set=t["netting_set"],
                trade_type=t["trade_type"],
                product=_parse_product(t["product"]),
                additional_fields=t.get("additional_fields", {}),
            )
        )

    pf = Portfolio(trades=tuple(trades), source_meta=SourceMeta(**portfolio.get("source_meta", {"origin": "dataclass"})))
    cfg = XVAConfig(
        asof=config["asof"],
        base_currency=config["base_currency"],
        analytics=tuple(config.get("analytics", ("CVA", "DVA", "FVA", "MVA"))),
        num_paths=config.get("num_paths", 5000),
        horizon_years=config.get("horizon_years", 5),
        params=config.get("params", {}),
        xml_buffers=config.get("xml_buffers", {}),
        runtime=_runtime_from_dict(config.get("runtime")) if config.get("runtime") else None,
        mpor=MporConfig(**config.get("mpor", {})),
        source_meta=SourceMeta(**config.get("source_meta", {"origin": "dataclass"})),
    )

    ns = NettingConfig(
        netting_sets={k: NettingSet(**v) for k, v in netting.get("netting_sets", {}).items()},
        source_meta=SourceMeta(**netting.get("source_meta", {"origin": "dataclass"})),
    )

    cc = CollateralConfig(
        balances=tuple(CollateralBalance(**v) for v in collateral.get("balances", [])),
        source_meta=SourceMeta(**collateral.get("source_meta", {"origin": "dataclass"})),
    )

    source_meta = {k: SourceMeta(**v) for k, v in data.get("source_meta", {}).items()}
    return XVASnapshot(market=mk, fixings=fx, portfolio=pf, config=cfg, netting=ns, collateral=cc, source_meta=source_meta)


def _validate_date_like(value: str, field_name: str) -> None:
    if len(value) == 8 and value.isdigit():
        datetime.strptime(value, "%Y%m%d")
        return
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD or YYYYMMDD") from exc


def ensure_tuple[T](values: Iterable[T]) -> Tuple[T, ...]:
    return tuple(values)


def _runtime_from_dict(data: Dict[str, Any]) -> RuntimeConfig:
    pe = data.get("pricing_engine", {})
    tm = data.get("todays_market", {})
    sim = data.get("simulation", {})
    sim_market = data.get("simulation_market", {})
    cam = data.get("cross_asset_model", {})
    xva = data.get("xva_analytic", {})
    cred = data.get("credit_simulation", {})
    conv = data.get("conventions", {})
    cp = data.get("counterparties", {})
    curves = tuple(
        CurveConfig(
            curve_id=c["curve_id"],
            currency=c["currency"],
            tenors=tuple(c.get("tenors", ("1Y", "2Y", "5Y", "10Y"))),
        )
        for c in data.get("curve_configs", [])
    )
    return RuntimeConfig(
        pricing_engine=PricingEngineConfig(
            model=pe.get("model", "BlackScholes"),
            npv_engine=pe.get("npv_engine", "DiscountedCashflows"),
            fx_model=pe.get("fx_model"),
            fx_engine=pe.get("fx_engine"),
            swap_model=pe.get("swap_model", "DiscountedCashflows"),
            swap_engine=pe.get("swap_engine", "DiscountingSwapEngine"),
            bermudan_model=pe.get("bermudan_model", "LGM"),
            bermudan_engine=pe.get("bermudan_engine", "Gaussian1dNonstandardSwaptionEngine"),
            bermudan_reversion=float(pe.get("bermudan_reversion", 0.03)),
            bermudan_volatility=float(pe.get("bermudan_volatility", 0.01)),
            bermudan_shift_horizon=float(pe.get("bermudan_shift_horizon", 0.0)),
            bermudan_sx=float(pe.get("bermudan_sx", 3.0)),
            bermudan_nx=int(pe.get("bermudan_nx", 10)),
            bermudan_sy=float(pe.get("bermudan_sy", 3.0)),
            bermudan_ny=int(pe.get("bermudan_ny", 10)),
        ),
        todays_market=TodaysMarketConfig(
            market_id=tm.get("market_id", "default"),
            discount_curve=tm.get("discount_curve", "EUR-EONIA"),
            fx_pairs=tuple(tm.get("fx_pairs", ("EURUSD",))),
            yield_curves_id=tm.get("yield_curves_id"),
            discounting_curves_id=tm.get("discounting_curves_id"),
            index_forwarding_curves_id=tm.get("index_forwarding_curves_id"),
            fx_spots_id=tm.get("fx_spots_id"),
            fx_volatilities_id=tm.get("fx_volatilities_id"),
            swaption_volatilities_id=tm.get("swaption_volatilities_id"),
            default_curves_id=tm.get("default_curves_id"),
        ),
        curve_configs=curves,
        simulation=SimulationConfig(
            samples=sim.get("samples", 5000),
            seed=sim.get("seed", 42),
            dates=tuple(sim.get("dates", ())),
            strict_template=bool(sim.get("strict_template", False)),
            discretization=sim.get("discretization", "Exact"),
            sequence=sim.get("sequence", "SobolBrownianBridge"),
            scenario=sim.get("scenario", "Simple"),
            closeout_lag=sim.get("closeout_lag", "2W"),
            mpor_mode=sim.get("mpor_mode", "StickyDate"),
            day_counter=sim.get("day_counter", "A365F"),
            calendar=sim.get("calendar"),
            xva_cg_dynamic_im=bool(sim.get("xva_cg_dynamic_im", False)),
            xva_cg_dynamic_im_step_size=int(sim.get("xva_cg_dynamic_im_step_size", 1)),
            xva_cg_regression_order_dynamic_im=(
                None
                if sim.get("xva_cg_regression_order_dynamic_im") is None
                else int(sim.get("xva_cg_regression_order_dynamic_im"))
            ),
            xva_cg_regression_report_time_steps_dynamic_im=tuple(
                int(x) for x in sim.get("xva_cg_regression_report_time_steps_dynamic_im", ())
            ),
        ),
        simulation_market=SimulationMarketConfig(
            base_currency=sim_market.get("base_currency", "EUR"),
            currencies=tuple(sim_market.get("currencies", ("EUR", "USD"))),
            indices=tuple(sim_market.get("indices", ("EUR-ESTER", "USD-SOFR"))),
            default_curve_names=tuple(sim_market.get("default_curve_names", ("BANK", "CPTY_A"))),
            fx_pairs=tuple(sim_market.get("fx_pairs", ("USDEUR",))),
            yield_curve_tenors=tuple(sim_market.get("yield_curve_tenors", ("3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y"))),
            yield_curve_interpolation=sim_market.get("yield_curve_interpolation", "LogLinear"),
            yield_curve_extrapolation=bool(sim_market.get("yield_curve_extrapolation", True)),
            default_curve_tenors=tuple(sim_market.get("default_curve_tenors", ("1Y", "2Y", "5Y", "10Y"))),
            default_simulate_survival_probabilities=bool(sim_market.get("default_simulate_survival_probabilities", True)),
            default_simulate_recovery_rates=bool(sim_market.get("default_simulate_recovery_rates", True)),
            default_curve_calendar=sim_market.get("default_curve_calendar", "TARGET"),
            default_curve_extrapolation=sim_market.get("default_curve_extrapolation", "FlatZero"),
            swaption_simulate=bool(sim_market.get("swaption_simulate", False)),
            swaption_reaction_to_time_decay=sim_market.get("swaption_reaction_to_time_decay", "ForwardVariance"),
            swaption_expiries=tuple(sim_market.get("swaption_expiries", ("6M", "1Y", "2Y", "3Y", "5Y", "10Y", "12Y", "15Y", "20Y"))),
            swaption_terms=tuple(sim_market.get("swaption_terms", ("1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"))),
            fxvol_simulate=bool(sim_market.get("fxvol_simulate", False)),
            fxvol_reaction_to_time_decay=sim_market.get("fxvol_reaction_to_time_decay", "ForwardVariance"),
            fxvol_expiries=tuple(sim_market.get("fxvol_expiries", ("1Y", "2Y", "5Y"))),
        ),
        cross_asset_model=CrossAssetModelConfig(
            domestic_ccy=cam.get("domestic_ccy", "EUR"),
            currencies=tuple(cam.get("currencies", ("EUR", "USD"))),
            ir_model_ccys=tuple(cam.get("ir_model_ccys", ("EUR", "USD"))),
            fx_model_ccys=tuple(cam.get("fx_model_ccys", ("USD",))),
            bootstrap_tolerance=float(cam.get("bootstrap_tolerance", 0.0001)),
            ir_calibration_type=cam.get("ir_calibration_type", "Bootstrap"),
            ir_volatility=float(cam.get("ir_volatility", 0.01)),
            ir_reversion=float(cam.get("ir_reversion", 0.0)),
            ir_shift_horizon=float(cam.get("ir_shift_horizon", 20.0)),
            ir_scaling=float(cam.get("ir_scaling", 1.0)),
            ir_calibration_expiries=tuple(cam.get("ir_calibration_expiries", ("1Y",))),
            ir_calibration_terms=tuple(cam.get("ir_calibration_terms", ("5Y",))),
            fx_calibration_type=cam.get("fx_calibration_type", "Bootstrap"),
            fx_sigma=float(cam.get("fx_sigma", 0.1)),
            fx_calibration_expiries=tuple(cam.get("fx_calibration_expiries", ("1Y",))),
            correlations=tuple(
                (str(a), str(b), float(v)) for a, b, v in cam.get(
                    "correlations",
                    (("IR:EUR", "IR:USD", 0.0), ("IR:EUR", "FX:USDEUR", 0.0), ("IR:USD", "FX:USDEUR", 0.0)),
                )
            ),
        ),
        xva_analytic=XVAAnalyticConfig(
            exposure_allocation_method=xva.get("exposure_allocation_method", "None"),
            exposure_observation_model=xva.get("exposure_observation_model", "Disable"),
            pfe_quantile=float(xva.get("pfe_quantile", 0.95)),
            collateral_calculation_type=xva.get("collateral_calculation_type", "Symmetric"),
            marginal_allocation_limit=float(xva.get("marginal_allocation_limit", 1.0)),
            exercise_next_break=bool(xva.get("exercise_next_break", False)),
            scenario_gen_type=xva.get("scenario_gen_type"),
            netting_set_ids=tuple(xva.get("netting_set_ids", ())),
            full_initial_collateralisation=xva.get("full_initial_collateralisation"),
            flip_view_xva=xva.get("flip_view_xva"),
            collateral_floor_enabled=xva.get("collateral_floor_enabled"),
            dim_model=xva.get("dim_model"),
            dim_quantile=(None if xva.get("dim_quantile") is None else float(xva.get("dim_quantile"))),
            dim_horizon_calendar_days=(
                None if xva.get("dim_horizon_calendar_days") is None else int(xva.get("dim_horizon_calendar_days"))
            ),
            dim_regression_order=(
                None if xva.get("dim_regression_order") is None else int(xva.get("dim_regression_order"))
            ),
            dim_regressors=xva.get("dim_regressors"),
            dim_evolution_file=xva.get("dim_evolution_file"),
            dim_regression_files=xva.get("dim_regression_files"),
            dim_output_grid_points=xva.get("dim_output_grid_points"),
            dim_output_netting_set=xva.get("dim_output_netting_set"),
            dim_local_regression_evaluations=(
                None
                if xva.get("dim_local_regression_evaluations") is None
                else int(xva.get("dim_local_regression_evaluations"))
            ),
            dim_local_regression_bandwidth=(
                None
                if xva.get("dim_local_regression_bandwidth") is None
                else float(xva.get("dim_local_regression_bandwidth"))
            ),
            flip_view_borrowing_curve_postfix=xva.get("flip_view_borrowing_curve_postfix"),
            flip_view_lending_curve_postfix=xva.get("flip_view_lending_curve_postfix"),
            cva_enabled=xva.get("cva_enabled"),
            dva_enabled=xva.get("dva_enabled"),
            fva_enabled=xva.get("fva_enabled"),
            mva_enabled=xva.get("mva_enabled"),
            colva_enabled=xva.get("colva_enabled"),
            dim_enabled=xva.get("dim_enabled"),
            dynamic_credit_enabled=xva.get("dynamic_credit_enabled"),
            dva_name=xva.get("dva_name"),
            fva_borrowing_curve=xva.get("fva_borrowing_curve"),
            fva_lending_curve=xva.get("fva_lending_curve"),
        ),
        credit_simulation=CreditSimulationConfig(
            enabled=bool(cred.get("enabled", False)),
            transition_matrix_name=cred.get("transition_matrix_name", "TransitionMatrix_1"),
            netting_set_ids=tuple(cred.get("netting_set_ids", ())),
            entities=tuple(
                CreditEntityConfig(
                    name=e["name"],
                    factor_loading=float(e.get("factor_loading", 0.4898979485566356)),
                    transition_matrix=e.get("transition_matrix", cred.get("transition_matrix_name", "TransitionMatrix_1")),
                    initial_state=int(e.get("initial_state", 4)),
                )
                for e in cred.get("entities", [])
            ),
            paths=int(cred.get("paths", 1000)),
            seed=int(cred.get("seed", 42)),
            evaluation=cred.get("evaluation", "Analytic"),
            credit_mode=cred.get("credit_mode", "Migration"),
            loan_exposure_mode=cred.get("loan_exposure_mode", "Value"),
            market_risk=bool(cred.get("market_risk", False)),
            credit_risk=bool(cred.get("credit_risk", True)),
            double_default=bool(cred.get("double_default", True)),
            zero_market_pnl=bool(cred.get("zero_market_pnl", False)),
        ),
        conventions=ConventionsConfig(
            day_counter=conv.get("day_counter", "A365"),
            calendar=conv.get("calendar", "TARGET"),
            yield_curve_day_counter=conv.get("yield_curve_day_counter", "A365"),
            cds_day_counter=conv.get("cds_day_counter", "Actual/365 (Fixed)"),
            cds_conventions=conv.get("cds_conventions", "CDS-STANDARD-CONVENTIONS"),
        ),
        counterparties=CounterpartyConfig(
            ids=tuple(cp.get("ids", ())),
            curve_currencies={str(k): str(v) for k, v in cp.get("curve_currencies", {}).items()},
            credit_qualities={str(k): str(v) for k, v in cp.get("credit_qualities", {}).items()},
            bacva_risk_weights={str(k): float(v) for k, v in cp.get("bacva_risk_weights", {}).items()},
            saccr_risk_weights={str(k): float(v) for k, v in cp.get("saccr_risk_weights", {}).items()},
            sacva_risk_buckets={str(k): int(v) for k, v in cp.get("sacva_risk_buckets", {}).items()},
        ),
        store_sensis=bool(data.get("store_sensis", False)),
        curve_sensi_grid=tuple(float(x) for x in data.get("curve_sensi_grid", ())),
        vega_sensi_grid=tuple(float(x) for x in data.get("vega_sensi_grid", ())),
    )
