from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, Protocol

from pythonore.domain.dataclasses import (
    BermudanSwaption,
    CollateralConfig,
    ConventionsConfig,
    CounterpartyConfig,
    CreditSimulationConfig,
    CrossAssetModelConfig,
    EquityForward,
    EquityOption,
    EquitySwap,
    FXForward,
    GenericProduct,
    IRS,
    MporConfig,
    NettingConfig,
    Portfolio,
    PricingEngineConfig,
    RuntimeConfig,
    SimulationConfig,
    SimulationMarketConfig,
    TodaysMarketConfig,
    Trade,
    XVASnapshot,
)
from pythonore.runtime.exceptions import MappingError


@dataclass(frozen=True)
class MappedInputs:
    asof: str
    base_currency: str
    analytics: List[str]
    market_data_lines: List[str]
    fixing_data_lines: List[str]
    xml_buffers: Dict[str, str]


class InputParametersLike(Protocol):
    def setAsOfDate(self, s: str) -> None: ...
    def setBaseCurrency(self, s: str) -> None: ...
    def setAnalytics(self, s: str) -> None: ...
    def setPortfolio(self, xml: str) -> None: ...
    def setNettingSetManager(self, xml: str) -> None: ...
    def setCollateralBalances(self, xml: str) -> None: ...
    def setPricingEngine(self, xml: str) -> None: ...
    def setTodaysMarketParams(self, xml: str) -> None: ...
    def setCurveConfigs(self, xml: str, id: str = "") -> None: ...
    def setConventions(self, xml: str) -> None: ...
    def setRefDataManager(self, xml: str) -> None: ...
    def setDimModel(self, s: str) -> None: ...
    def setXvaCgDynamicIM(self, b: bool) -> None: ...
    def setXvaCgDynamicIMStepSize(self, n: int) -> None: ...
    def setXvaCgRegressionOrderDynamicIm(self, n: int) -> None: ...
    def setXvaCgRegressionReportTimeStepsDynamicIM(self, s: Any) -> None: ...
    def setStoreSensis(self, b: bool) -> None: ...
    def setCurveSensiGrid(self, s: Any) -> None: ...
    def setVegaSensiGrid(self, s: Any) -> None: ...


def map_snapshot(snapshot: XVASnapshot) -> MappedInputs:
    if not snapshot.portfolio.trades:
        raise MappingError(
            "Snapshot portfolio is empty. "
            "Fix: provide at least one trade in snapshot.portfolio, or load the snapshot "
            "from an ORE portfolio.xml via XVALoader.from_files(...)."
        )

    if not snapshot.market.raw_quotes:
        warnings.warn(
            "Snapshot contains no market quotes (raw_quotes is empty). "
            "The XVA engine will run with zero market data and produce meaningless results. "
            "Fix: populate snapshot.market.raw_quotes with at least discount-curve and FX quotes.",
            UserWarning,
            stacklevel=2,
        )

    market_lines = [f"{q.date.replace('-', '')} {q.key} {q.value}" for q in snapshot.market.raw_quotes]
    fixing_lines = [f"{f.date} {f.index} {f.value}" for f in snapshot.fixings.points]
    xml_buffers = dict(snapshot.config.xml_buffers)
    runtime_xml = _runtime_xml_buffers(snapshot)
    for k, v in runtime_xml.items():
        if k not in xml_buffers or _is_generated_placeholder_xml(xml_buffers[k], k):
            xml_buffers[k] = v

    if "portfolio.xml" not in xml_buffers:
        xml_buffers["portfolio.xml"] = _portfolio_to_xml(
            snapshot.portfolio,
            snapshot.config.asof,
            snapshot.config.runtime or RuntimeConfig(),
        )
    if "netting.xml" not in xml_buffers:
        xml_buffers["netting.xml"] = _netting_to_xml(snapshot.netting)
    if "collateralbalances.xml" not in xml_buffers:
        xml_buffers["collateralbalances.xml"] = _collateral_to_xml(snapshot.collateral)

    # These are required by the SWIG/native orchestration layer. For dataclass-
    # only snapshots we synthesize a generated compatibility bundle so the
    # adapter can run without an ORE input folder. This is not a parity-grade
    # setup: for ORE parity, callers should provide the real ORE XML files via
    # XVALoader or explicit xml_buffers.
    for req in ("pricingengine.xml", "todaysmarket.xml", "curveconfig.xml", "simulation.xml"):
        if req not in xml_buffers:
            xml_buffers[req] = runtime_xml.get(req, _empty_xml(req))
    xml_buffers["simulation.xml"] = _apply_num_paths_to_simulation_xml(
        xml_buffers["simulation.xml"], snapshot.config.num_paths
    )
    resolved_mpor = _resolve_mpor_config(snapshot, xml_buffers)
    xml_buffers["simulation.xml"] = _apply_mpor_to_simulation_xml(
        xml_buffers["simulation.xml"], resolved_mpor
    )

    return MappedInputs(
        asof=snapshot.config.asof,
        base_currency=snapshot.config.base_currency,
        analytics=list(snapshot.config.analytics),
        market_data_lines=market_lines,
        fixing_data_lines=fixing_lines,
        xml_buffers=xml_buffers,
    )


def build_input_parameters(snapshot: XVASnapshot, input_parameters: InputParametersLike) -> InputParametersLike:
    mapped = map_snapshot(snapshot)
    input_parameters.setAsOfDate(mapped.asof)
    input_parameters.setBaseCurrency(mapped.base_currency)
    # InputParameters expects internal analytic labels, not report-facing names.
    # EXPOSURE is required for XVA cube generation; SCENARIO alone leads to load-cube mode.
    input_parameters.setAnalytics("PRICING,EXPOSURE,XVA")
    input_parameters.setPortfolio(mapped.xml_buffers["portfolio.xml"])
    input_parameters.setNettingSetManager(mapped.xml_buffers["netting.xml"])
    input_parameters.setCollateralBalances(mapped.xml_buffers["collateralbalances.xml"])
    input_parameters.setPricingEngine(mapped.xml_buffers["pricingengine.xml"])
    input_parameters.setTodaysMarketParams(mapped.xml_buffers["todaysmarket.xml"])
    input_parameters.setCurveConfigs(mapped.xml_buffers["curveconfig.xml"], "")

    _maybe_set(input_parameters, "setConventions", mapped.xml_buffers.get("conventions.xml"))
    refdata_xml = mapped.xml_buffers.get("referencedata.xml") or mapped.xml_buffers.get("reference_data.xml")
    if _looks_like_refdata(refdata_xml):
        _maybe_set(input_parameters, "setRefDataManager", refdata_xml)
    simulation_xml = mapped.xml_buffers.get("simulation.xml")
    _maybe_set(input_parameters, "setScenarioSimMarketParams", simulation_xml)
    _maybe_set(input_parameters, "setExposureSimMarketParams", simulation_xml)
    _maybe_set(input_parameters, "setScenarioGeneratorData", simulation_xml)
    _maybe_set(input_parameters, "setCrossAssetModelData", _extract_xml_section(simulation_xml, "CrossAssetModel"))
    _maybe_set(input_parameters, "setSimulationPricingEngine", mapped.xml_buffers.get("pricingengine.xml"))
    _maybe_set(input_parameters, "setAmcPricingEngine", mapped.xml_buffers.get("pricingengine.xml"))
    _maybe_set_bool(input_parameters, "setAllFixings", True)
    _maybe_set_bool(input_parameters, "setEntireMarket", True)
    _maybe_set_bool(input_parameters, "setLoadCube", False)
    _maybe_set(input_parameters, "setXvaBaseCurrency", snapshot.config.base_currency)
    _maybe_set(input_parameters, "setExposureBaseCurrency", snapshot.config.base_currency)
    _maybe_set(input_parameters, "setMarketConfig", _market_config_args(snapshot.config.params))
    resolved_mpor = _resolve_mpor_config(snapshot, mapped.xml_buffers)
    if resolved_mpor.enabled:
        _maybe_set(input_parameters, "setMporDays", int(resolved_mpor.mpor_days))
        _maybe_set(input_parameters, "setMporCalendar", snapshot.config.base_currency)
        _maybe_set_bool(input_parameters, "setMporForward", True)
    _maybe_set_bool(input_parameters, "setCvaAnalytic", "CVA" in snapshot.config.analytics)
    _maybe_set_bool(input_parameters, "setDvaAnalytic", "DVA" in snapshot.config.analytics)
    _maybe_set_bool(input_parameters, "setFvaAnalytic", "FVA" in snapshot.config.analytics)
    _maybe_set_bool(input_parameters, "setMvaAnalytic", "MVA" in snapshot.config.analytics)
    _maybe_set_bool(input_parameters, "setExposureProfiles", True)
    _maybe_set_bool(input_parameters, "setExposureProfilesByTrade", True)
    _apply_simulation_overrides(snapshot, input_parameters)
    _apply_xva_analytic_overrides(snapshot, input_parameters)
    _apply_credit_simulation(snapshot, mapped, input_parameters)
    return input_parameters


def _portfolio_to_xml(portfolio: Portfolio, asof: str, runtime: RuntimeConfig) -> str:
    lines = ["<Portfolio>"]
    for t in portfolio.trades:
        lines.append(f'  <Trade id="{t.trade_id}">')
        lines.append(f"    <TradeType>{t.trade_type}</TradeType>")
        lines.append("    <Envelope>")
        lines.append(f"      <CounterParty>{t.counterparty}</CounterParty>")
        lines.append(f"      <NettingSetId>{t.netting_set}</NettingSetId>")
        lines.extend(_additional_fields_xml(t.additional_fields))
        lines.append("    </Envelope>")
        lines.extend(_product_xml(t, asof, runtime))
        lines.append("  </Trade>")
    lines.append("</Portfolio>")
    return "\n".join(lines)


def _product_xml(trade: Trade, asof: str, runtime: RuntimeConfig | None = None) -> List[str]:
    p = trade.product
    runtime = runtime or RuntimeConfig()
    conventions = runtime.conventions
    indices = runtime.simulation_market.indices
    if isinstance(p, IRS):
        start_date = _fmt_yyyymmdd(p.start_date or asof)
        end_date = _fmt_yyyymmdd(p.end_date) if p.end_date else _add_months_yyyymmdd(
            start_date, int(round(p.maturity_years * 12.0))
        )
        payer_fixed = str(p.pay_fixed).lower()
        payer_float = str(not p.pay_fixed).lower()
        # This branch is intentionally a convenience fallback, not a full ORE
        # trade serializer. It exposes the main schedule and floating-leg fields
        # from the IRS dataclass, but it still does not model the entire ORE
        # trade schema (stubs, explicit calendars per leg, amortisation, etc.).
        idx = p.float_index or _resolve_index_for_trade_currency(p.ccy, indices)
        calendar = p.calendar or conventions.calendar
        fixed_day_counter = p.fixed_day_counter or conventions.day_counter
        float_day_counter = p.float_day_counter or conventions.day_counter
        fixed_schedule_convention = p.fixed_schedule_convention or p.fixed_payment_convention
        float_schedule_convention = p.float_schedule_convention or p.float_payment_convention
        fixed_term_convention = p.fixed_term_convention or fixed_schedule_convention
        float_term_convention = p.float_term_convention or float_schedule_convention
        return [
            "    <SwapData>",
            "      <StartDate>" + start_date + "</StartDate>",
            "      <EndDate>" + end_date + "</EndDate>",
            "      <LegData>",
            "        <LegType>Fixed</LegType>",
            f"        <Payer>{payer_fixed}</Payer>",
            f"        <Currency>{p.ccy}</Currency>",
            f"        <DayCounter>{fixed_day_counter}</DayCounter>",
            f"        <PaymentConvention>{p.fixed_payment_convention}</PaymentConvention>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            f"            <Tenor>{p.fixed_leg_tenor}</Tenor>",
            f"            <Calendar>{calendar}</Calendar>",
            f"            <Convention>{fixed_schedule_convention}</Convention>",
            f"            <TermConvention>{fixed_term_convention}</TermConvention>",
            f"            <Rule>{p.fixed_schedule_rule}</Rule>",
            f"            <EndOfMonth>{str(p.end_of_month).lower()}</EndOfMonth>",
            "          </Rules>",
            "        </ScheduleData>",
            "        <FixedLegData>",
            "          <Rates>",
            f"            <Rate>{p.fixed_rate}</Rate>",
            "          </Rates>",
            "        </FixedLegData>",
            "      </LegData>",
            "      <LegData>",
            "        <LegType>Floating</LegType>",
            f"        <Payer>{payer_float}</Payer>",
            f"        <Currency>{p.ccy}</Currency>",
            f"        <DayCounter>{float_day_counter}</DayCounter>",
            f"        <PaymentConvention>{p.float_payment_convention}</PaymentConvention>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            f"            <Tenor>{p.float_leg_tenor}</Tenor>",
            f"            <Calendar>{calendar}</Calendar>",
            f"            <Convention>{float_schedule_convention}</Convention>",
            f"            <TermConvention>{float_term_convention}</TermConvention>",
            f"            <Rule>{p.float_schedule_rule}</Rule>",
            f"            <EndOfMonth>{str(p.end_of_month).lower()}</EndOfMonth>",
            "          </Rules>",
            "        </ScheduleData>",
            "        <FloatingLegData>",
            f"          <Index>{idx}</Index>",
            f"          <FixingDays>{p.fixing_days}</FixingDays>",
            "          <Spreads>",
            f"            <Spread>{p.float_spread}</Spread>",
            "          </Spreads>",
            "        </FloatingLegData>",
            "      </LegData>",
            "    </SwapData>",
        ]
    if isinstance(p, FXForward):
        base = p.pair[:3]
        quote = p.pair[3:]
        bought = p.notional if p.buy_base else p.notional * p.strike
        sold = p.notional * p.strike if p.buy_base else p.notional
        value_date = _fmt_yyyymmdd(p.value_date) if p.value_date else _add_months_yyyymmdd(
            _fmt_yyyymmdd(asof), int(round(p.maturity_years * 12.0))
        )
        return [
            "    <FxForwardData>",
            f"      <BoughtCurrency>{base if p.buy_base else quote}</BoughtCurrency>",
            f"      <BoughtAmount>{bought}</BoughtAmount>",
            f"      <SoldCurrency>{quote if p.buy_base else base}</SoldCurrency>",
            f"      <SoldAmount>{sold}</SoldAmount>",
            f"      <ValueDate>{value_date[:4]}-{value_date[4:6]}-{value_date[6:]}</ValueDate>",
            "    </FxForwardData>",
        ]
    if isinstance(p, EquityOption):
        return [
            "    <EquityOptionData>",
            "      <OptionData>",
            f"        <LongShort>{p.long_short}</LongShort>",
            f"        <OptionType>{p.option_type}</OptionType>",
            f"        <Style>{p.style}</Style>",
            f"        <Settlement>{p.settlement}</Settlement>",
            f"        <PayOffAtExpiry>{str(p.payoff_at_expiry).lower()}</PayOffAtExpiry>",
            "        <ExerciseDates>",
            f"          <ExerciseDate>{_fmt_yyyymmdd(p.exercise_date) if p.exercise_date else _fmt_yyyymmdd(asof)}</ExerciseDate>",
            "        </ExerciseDates>",
            "      </OptionData>",
            f"      <Name>{_xml_escape(p.name)}</Name>",
            f"      <Currency>{p.currency}</Currency>",
            f"      <Strike>{p.strike}</Strike>",
            f"      <Quantity>{p.quantity}</Quantity>",
            "    </EquityOptionData>",
        ]
    if isinstance(p, EquityForward):
        maturity_date = _fmt_yyyymmdd(p.maturity_date) if p.maturity_date else _add_months_yyyymmdd(
            _fmt_yyyymmdd(asof), int(round(p.maturity_years * 12.0))
        )
        return [
            "    <EquityForwardData>",
            f"      <LongShort>{p.long_short}</LongShort>",
            f"      <Maturity>{maturity_date[:4]}-{maturity_date[4:6]}-{maturity_date[6:]}</Maturity>",
            f"      <Name>{_xml_escape(p.name)}</Name>",
            f"      <Currency>{p.currency}</Currency>",
            f"      <Strike>{p.strike}</Strike>",
            f"      <StrikeCurrency>{p.strike_currency or p.currency}</StrikeCurrency>",
            f"      <Quantity>{p.quantity}</Quantity>",
            "    </EquityForwardData>",
        ]
    if isinstance(p, EquitySwap):
        start_date = _fmt_yyyymmdd(p.start_date or asof)
        end_date = _fmt_yyyymmdd(p.end_date) if p.end_date else _add_months_yyyymmdd(
            start_date, int(round(p.maturity_years * 12.0))
        )
        calendar = p.calendar or p.currency
        return [
            "    <SwapData>",
            "      <LegData>",
            "        <LegType>Equity</LegType>",
            f"        <Payer>{str(p.equity_payer).lower()}</Payer>",
            f"        <Currency>{p.currency}</Currency>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            f"        <DayCounter>{p.equity_day_counter}</DayCounter>",
            f"        <PaymentConvention>{p.equity_payment_convention}</PaymentConvention>",
            "        <EquityLegData>",
            f"          <ReturnType>{p.return_type}</ReturnType>",
            f"          <Name>{_xml_escape(p.name)}</Name>",
            f"          <InitialPrice>{p.initial_price}</InitialPrice>",
            "        </EquityLegData>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            f"            <Tenor>{p.equity_leg_tenor}</Tenor>",
            f"            <Calendar>{calendar}</Calendar>",
            f"            <Convention>{p.equity_schedule_convention}</Convention>",
            f"            <TermConvention>{p.equity_term_convention}</TermConvention>",
            f"            <Rule>{p.equity_schedule_rule}</Rule>",
            f"            <EndOfMonth>{str(p.end_of_month).lower()}</EndOfMonth>",
            "          </Rules>",
            "        </ScheduleData>",
            "      </LegData>",
            "      <LegData>",
            "        <LegType>Floating</LegType>",
            f"        <Payer>{str(not p.equity_payer).lower()}</Payer>",
            f"        <Currency>{p.currency}</Currency>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            f"        <DayCounter>{p.float_day_counter}</DayCounter>",
            f"        <PaymentConvention>{p.float_payment_convention}</PaymentConvention>",
            "        <FloatingLegData>",
            f"          <Index>{p.float_index}</Index>",
            "          <Spreads>",
            f"            <Spread>{p.float_spread}</Spread>",
            "          </Spreads>",
            "          <IsInArrears>false</IsInArrears>",
            f"          <FixingDays>{p.fixing_days}</FixingDays>",
            "        </FloatingLegData>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            f"            <Tenor>{p.float_leg_tenor}</Tenor>",
            f"            <Calendar>{calendar}</Calendar>",
            f"            <Convention>{p.float_schedule_convention}</Convention>",
            f"            <TermConvention>{p.float_term_convention}</TermConvention>",
            f"            <Rule>{p.float_schedule_rule}</Rule>",
            f"            <EndOfMonth>{str(p.end_of_month).lower()}</EndOfMonth>",
            "          </Rules>",
            "        </ScheduleData>",
            "      </LegData>",
            "    </SwapData>",
        ]
    if isinstance(p, BermudanSwaption):
        start_date = _fmt_yyyymmdd(p.start_date or asof)
        end_date = _fmt_yyyymmdd(p.end_date) if p.end_date else _add_months_yyyymmdd(
            start_date, int(round(p.maturity_years * 12.0))
        )
        payer_fixed = str(p.pay_fixed).lower()
        payer_float = str(not p.pay_fixed).lower()
        float_index = p.float_index or _resolve_index_for_trade_currency(p.ccy, indices)
        calendar = p.calendar or conventions.calendar
        fixed_day_counter = p.fixed_day_counter or conventions.day_counter
        float_day_counter = p.float_day_counter or conventions.day_counter
        fixed_schedule_convention = p.fixed_schedule_convention or p.fixed_payment_convention
        float_schedule_convention = p.float_schedule_convention or p.float_payment_convention
        fixed_term_convention = p.fixed_term_convention or fixed_schedule_convention
        float_term_convention = p.float_term_convention or float_schedule_convention
        exercise_dates = "\n".join(f"          <ExerciseDate>{d}</ExerciseDate>" for d in p.exercise_dates)
        return [
            "    <SwaptionData>",
            "      <OptionData>",
            f"        <LongShort>{p.long_short}</LongShort>",
            f"        <OptionType>{p.option_type}</OptionType>",
            "        <Style>Bermudan</Style>",
            f"        <Settlement>{p.settlement}</Settlement>",
            f"        <PayOffAtExpiry>{str(p.payoff_at_expiry).lower()}</PayOffAtExpiry>",
            "        <ExerciseDates>",
            exercise_dates,
            "        </ExerciseDates>",
            "      </OptionData>",
            "      <LegData>",
            "        <LegType>Floating</LegType>",
            f"        <Payer>{payer_float}</Payer>",
            f"        <Currency>{p.ccy}</Currency>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            f"        <DayCounter>{float_day_counter}</DayCounter>",
            f"        <PaymentConvention>{p.float_payment_convention}</PaymentConvention>",
            "        <FloatingLegData>",
            f"          <Index>{float_index}</Index>",
            "          <Spreads>",
            f"            <Spread>{p.float_spread}</Spread>",
            "          </Spreads>",
            f"          <FixingDays>{p.fixing_days}</FixingDays>",
            f"          <IsInArrears>{str(p.is_in_arrears).lower()}</IsInArrears>",
            "        </FloatingLegData>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            f"            <Tenor>{p.float_leg_tenor}</Tenor>",
            f"            <Calendar>{calendar}</Calendar>",
            f"            <Convention>{float_schedule_convention}</Convention>",
            f"            <TermConvention>{float_term_convention}</TermConvention>",
            f"            <Rule>{p.float_schedule_rule}</Rule>",
            f"            <EndOfMonth>{str(p.end_of_month).lower()}</EndOfMonth>",
            "          </Rules>",
            "        </ScheduleData>",
            "      </LegData>",
            "      <LegData>",
            "        <LegType>Fixed</LegType>",
            f"        <Payer>{payer_fixed}</Payer>",
            f"        <Currency>{p.ccy}</Currency>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            f"        <DayCounter>{fixed_day_counter}</DayCounter>",
            f"        <PaymentConvention>{p.fixed_payment_convention}</PaymentConvention>",
            "        <FixedLegData>",
            "          <Rates>",
            f"            <Rate>{p.fixed_rate}</Rate>",
            "          </Rates>",
            "        </FixedLegData>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            f"            <Tenor>{p.fixed_leg_tenor}</Tenor>",
            f"            <Calendar>{calendar}</Calendar>",
            f"            <Convention>{fixed_schedule_convention}</Convention>",
            f"            <TermConvention>{fixed_term_convention}</TermConvention>",
            f"            <Rule>{p.fixed_schedule_rule}</Rule>",
            f"            <EndOfMonth>{str(p.end_of_month).lower()}</EndOfMonth>",
            "          </Rules>",
            "        </ScheduleData>",
            "      </LegData>",
            "    </SwaptionData>",
        ]
    if isinstance(p, GenericProduct):
        payload_xml = str(p.payload.get("xml", "")).strip()
        if payload_xml:
            return [f"    {line}" if line else "" for line in payload_xml.splitlines()]
    return ["    <Data/>", f"    <!-- Generic product payload omitted for {type(p).__name__} -->"]


def _additional_fields_xml(fields: Dict[str, str]) -> List[str]:
    if not fields:
        return ["      <AdditionalFields/>"]
    lines = ["      <AdditionalFields>"]
    for key, value in fields.items():
        tag = _xml_safe_tag(key)
        escaped = _xml_escape(value)
        lines.append(f"        <{tag}>{escaped}</{tag}>")
    lines.append("      </AdditionalFields>")
    return lines


def _netting_to_xml(netting: NettingConfig) -> str:
    lines = ["<NettingSetDefinitions>"]
    for ns_id, ns in netting.netting_sets.items():
        lines.append("  <NettingSet>")
        lines.append(f"    <NettingSetId>{ns_id}</NettingSetId>")
        lines.append(f"    <ActiveCSAFlag>{str(ns.active_csa).lower()}</ActiveCSAFlag>")
        lines.append("    <CSADetails>")
        if ns.bilateral is not None:
            lines.append(f"      <Bilateral>{_xml_escape(ns.bilateral)}</Bilateral>")
        lines.append(f"      <CSACurrency>{ns.csa_currency or 'EUR'}</CSACurrency>")
        if ns.index is not None:
            lines.append(f"      <Index>{_xml_escape(ns.index)}</Index>")
        lines.append(f"      <ThresholdPay>{ns.threshold_pay or 0.0}</ThresholdPay>")
        lines.append(f"      <ThresholdReceive>{ns.threshold_receive or 0.0}</ThresholdReceive>")
        lines.append(f"      <MinimumTransferAmountPay>{ns.mta_pay or 0.0}</MinimumTransferAmountPay>")
        lines.append(f"      <MinimumTransferAmountReceive>{ns.mta_receive or 0.0}</MinimumTransferAmountReceive>")
        if (
            ns.independent_amount.held is not None
            or ns.independent_amount.posted is not None
            or ns.independent_amount.amount_type is not None
        ):
            lines.append("      <IndependentAmount>")
            if ns.independent_amount.held is not None:
                lines.append(f"        <IndependentAmountHeld>{ns.independent_amount.held}</IndependentAmountHeld>")
            if ns.independent_amount.posted is not None:
                lines.append(f"        <IndependentAmountPosted>{ns.independent_amount.posted}</IndependentAmountPosted>")
            if ns.independent_amount.amount_type is not None:
                lines.append(f"        <IndependentAmountType>{_xml_escape(ns.independent_amount.amount_type)}</IndependentAmountType>")
            lines.append("      </IndependentAmount>")
        if ns.margining_frequency.call_frequency is not None or ns.margining_frequency.post_frequency is not None:
            lines.append("      <MarginingFrequency>")
            if ns.margining_frequency.call_frequency is not None:
                lines.append(f"        <CallFrequency>{_xml_escape(ns.margining_frequency.call_frequency)}</CallFrequency>")
            if ns.margining_frequency.post_frequency is not None:
                lines.append(f"        <PostFrequency>{_xml_escape(ns.margining_frequency.post_frequency)}</PostFrequency>")
            lines.append("      </MarginingFrequency>")
        lines.append(f"      <MarginPeriodOfRisk>{ns.margin_period_of_risk or '0D'}</MarginPeriodOfRisk>")
        if ns.collateral_compounding_spread_receive is not None:
            lines.append(
                f"      <CollateralCompoundingSpreadReceive>{ns.collateral_compounding_spread_receive}</CollateralCompoundingSpreadReceive>"
            )
        if ns.collateral_compounding_spread_pay is not None:
            lines.append(
                f"      <CollateralCompoundingSpreadPay>{ns.collateral_compounding_spread_pay}</CollateralCompoundingSpreadPay>"
            )
        if ns.eligible_collateral_currencies:
            lines.append("      <EligibleCollaterals>")
            lines.append("        <Currencies>")
            for ccy in ns.eligible_collateral_currencies:
                lines.append(f"          <Currency>{_xml_escape(ccy)}</Currency>")
            lines.append("        </Currencies>")
            lines.append("      </EligibleCollaterals>")
        for key, value in ns.raw_csa_fields.items():
            tag = _xml_safe_tag(key)
            lines.append(f"      <{tag}>{_xml_escape(value)}</{tag}>")
        lines.append("    </CSADetails>")
        lines.append("  </NettingSet>")
    lines.append("</NettingSetDefinitions>")
    return "\n".join(lines)


def _collateral_to_xml(collateral: CollateralConfig) -> str:
    lines = ["<CollateralBalances>"]
    for bal in collateral.balances:
        lines.append("  <CollateralBalance>")
        lines.append(f"    <NettingSetId>{bal.netting_set_id}</NettingSetId>")
        lines.append(f"    <Currency>{bal.currency}</Currency>")
        lines.append(f"    <InitialMargin>{bal.initial_margin}</InitialMargin>")
        lines.append(f"    <VariationMargin>{bal.variation_margin}</VariationMargin>")
        if bal.initial_margin_type is not None:
            lines.append(f"    <InitialMarginType>{_xml_escape(bal.initial_margin_type)}</InitialMarginType>")
        if bal.variation_margin_type is not None:
            lines.append(f"    <VariationMarginType>{_xml_escape(bal.variation_margin_type)}</VariationMarginType>")
        for key, value in bal.raw_fields.items():
            tag = _xml_safe_tag(key)
            lines.append(f"    <{tag}>{_xml_escape(value)}</{tag}>")
        lines.append("  </CollateralBalance>")
    lines.append("</CollateralBalances>")
    return "\n".join(lines)


def _empty_xml(name: str) -> str:
    root = name.split(".")[0].replace("_", "")
    return f"<{root}/>"


def _is_generated_placeholder_xml(xml: str, name: str) -> bool:
    try:
        root = ET.fromstring(xml)
    except Exception:
        return False
    expected = name.split(".")[0].replace("_", "").lower()
    text = (root.text or "").strip()
    return root.tag.lower() == expected and not root.attrib and len(root) == 0 and not text


def _fmt_yyyymmdd(asof: str) -> str:
    s = asof.strip()
    if len(s) == 8 and s.isdigit():
        return s
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")


def _add_months_yyyymmdd(start_yyyymmdd: str, months: int) -> str:
    d = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    mdays = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(d.day, mdays[m - 1])
    return f"{y:04d}{m:02d}{day:02d}"


def _currency_from_token(value: str) -> str:
    token = str(value or "").strip().upper()
    return token[:3] if len(token) >= 3 and token[:3].isalpha() else ""


def _xml_safe_tag(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "").strip())
    if not text:
        return "Field"
    if not re.match(r"[A-Za-z_]", text[0]):
        text = f"Field_{text}"
    return text


def _xml_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _currency_from_curve_id(curve_id: str) -> str:
    token = str(curve_id or "").strip().upper()
    if "/" in token:
        for part in token.split("/"):
            if len(part) == 3 and part.isalpha():
                return part
    if "-" in token:
        head = token.split("-", 1)[0]
        if len(head) == 3 and head.isalpha():
            return head
    return _currency_from_token(token)


def _currency_from_index_name(index_name: str) -> str:
    token = str(index_name or "").strip().upper()
    if "-" in token:
        head = token.split("-", 1)[0]
        if len(head) == 3 and head.isalpha():
            return head
    return _currency_from_token(token)


def _ordered_unique(values: List[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _currencies_from_fx_pairs(pairs: tuple[str, ...]) -> tuple[str, ...]:
    values: List[str] = []
    for pair in pairs:
        token = str(pair or "").strip().upper()
        if len(token) >= 6:
            values.extend((token[:3], token[3:6]))
    return _ordered_unique(values)


def _resolve_market_currencies(
    discount_curve: str,
    currencies: tuple[str, ...] = (),
    fx_pairs: tuple[str, ...] = (),
    indices: tuple[str, ...] = (),
) -> tuple[str, ...]:
    values = list(currencies)
    values.extend(_currencies_from_fx_pairs(fx_pairs))
    values.extend(_currency_from_index_name(index) for index in indices)
    discount_ccy = _currency_from_curve_id(discount_curve)
    if discount_ccy:
        values.append(discount_ccy)
    return _ordered_unique([str(v).upper() for v in values if v])


def _curve_convention_for_currency(ccy: str) -> str:
    c = ccy.upper()
    if c == "USD":
        return "USD-ON-DEPOSIT-SOFR"
    if c == "GBP":
        return "GBP-ON-DEPOSIT-SONIA"
    if c == "CHF":
        return "CHF-ON-DEPOSIT-SARON"
    if c == "JPY":
        return "JPY-ON-DEPOSIT-TONAR"
    return "EUR-ON-DEPOSIT-ESTER"


def _curve_path_for(curve_id: str, currency: str) -> str:
    return f"Yield/{currency}/{curve_id}"


def _default_curve_discount_path(curves: tuple, fallback_curve: str) -> str:
    usd_curve = next((c for c in curves if getattr(c, "currency", "").upper() == "USD"), None)
    if usd_curve is not None:
        return _curve_path_for(usd_curve.curve_id, "USD")
    ccy = _currency_from_curve_id(fallback_curve) or "USD"
    return _curve_path_for(fallback_curve, ccy)


def _default_curve_currency(name: str, counterparties: CounterpartyConfig | None, base_currency: str) -> str:
    if counterparties is not None:
        mapped = str(counterparties.curve_currencies.get(name, "")).strip().upper()
        if mapped:
            return mapped
    return str(base_currency or "USD").strip().upper() or "USD"


def _simulation_calendar(cfg: SimulationMarketConfig, cam_cfg: CrossAssetModelConfig | None = None) -> str:
    values = list(cfg.currencies)
    if cam_cfg is not None:
        values.extend(cam_cfg.currencies)
    return ",".join(_ordered_unique([str(v).upper() for v in values if v])) or "EUR,USD"


def _swap_indices_xml_lines(currencies: tuple[str, ...], indices: tuple[str, ...]) -> List[str]:
    lines: List[str] = []
    for ccy in currencies:
        discounting_index = _resolve_index_for_trade_currency(ccy, indices)
        for tenor in ("1Y", "30Y"):
            lines.append(
                f"      <SwapIndex><Name>{ccy}-CMS-{tenor}</Name><DiscountingIndex>{discounting_index}</DiscountingIndex></SwapIndex>"
            )
    return lines


def _default_index_for_ccy(ccy: str) -> str:
    c = ccy.upper()
    if c == "USD":
        return "USD-SOFR"
    if c == "GBP":
        return "GBP-SONIA"
    if c == "CHF":
        return "CHF-SARON"
    if c == "JPY":
        return "JPY-TONAR"
    return "EUR-ESTER"


def _resolve_index_for_trade_currency(ccy: str, configured_indices: tuple[str, ...] = ()) -> str:
    target = ccy.upper()
    for index in configured_indices:
        if _currency_from_index_name(index) == target:
            return str(index).upper()
    return _default_index_for_ccy(target)


def _runtime_xml_buffers(snapshot: XVASnapshot) -> Dict[str, str]:
    runtime = snapshot.config.runtime or RuntimeConfig()
    out = {
        "pricingengine.xml": _pricing_engine_xml(runtime.pricing_engine),
        "todaysmarket.xml": _todays_market_xml(
            runtime.todays_market,
            runtime.simulation_market,
            runtime.counterparties,
            snapshot.config.base_currency,
        ),
        "curveconfig.xml": _curve_config_xml(
            runtime.curve_configs,
            runtime.todays_market.discount_curve,
            runtime.simulation_market.default_curve_names,
            runtime.counterparties,
            snapshot.config.base_currency,
        ),
        "simulation.xml": _simulation_xml(
            runtime.simulation,
            runtime.simulation_market,
            runtime.cross_asset_model,
            snapshot.config.num_paths,
        ),
        "conventions.xml": _conventions_xml(runtime.conventions),
        "counterparty.xml": _counterparty_xml(runtime.counterparties),
    }
    if runtime.credit_simulation.enabled:
        out["creditsimulation.xml"] = _credit_simulation_xml(runtime.credit_simulation)
    return out


def _pricing_engine_xml(cfg: PricingEngineConfig) -> str:
    return "\n".join(
        [
            "<PricingEngines>",
            "  <Product type=\"FxForward\">",
            f"    <Model>{cfg.fx_model or cfg.model}</Model>",
            "    <ModelParameters/>",
            f"    <Engine>{cfg.fx_engine or cfg.npv_engine}</Engine>",
            "    <EngineParameters/>",
            "  </Product>",
            "  <Product type=\"EquityForward\">",
            f"    <Model>{cfg.equity_forward_model}</Model>",
            "    <ModelParameters/>",
            f"    <Engine>{cfg.equity_forward_engine}</Engine>",
            "    <EngineParameters/>",
            "  </Product>",
            "  <Product type=\"EquityOption\">",
            f"    <Model>{cfg.equity_option_model}</Model>",
            "    <ModelParameters/>",
            f"    <Engine>{cfg.equity_option_engine}</Engine>",
            "    <EngineParameters/>",
            "  </Product>",
            "  <Product type=\"EquitySwap\">",
            f"    <Model>{cfg.equity_swap_model}</Model>",
            "    <ModelParameters/>",
            f"    <Engine>{cfg.equity_swap_engine}</Engine>",
            "    <EngineParameters/>",
            "  </Product>",
            "  <Product type=\"Swap\">",
            f"    <Model>{cfg.swap_model}</Model>",
            "    <ModelParameters/>",
            f"    <Engine>{cfg.swap_engine}</Engine>",
            "    <EngineParameters/>",
            "  </Product>",
            "  <Product type=\"BermudanSwaption\">",
            f"    <Model>{cfg.bermudan_model}</Model>",
            "    <ModelParameters>",
            f"      <Parameter name=\"Reversion\">{cfg.bermudan_reversion}</Parameter>",
            f"      <Parameter name=\"Volatility\">{cfg.bermudan_volatility}</Parameter>",
            f"      <Parameter name=\"ShiftHorizon\">{cfg.bermudan_shift_horizon}</Parameter>",
            "    </ModelParameters>",
            f"    <Engine>{cfg.bermudan_engine}</Engine>",
            "    <EngineParameters>",
            f"      <Parameter name=\"sx\">{cfg.bermudan_sx}</Parameter>",
            f"      <Parameter name=\"nx\">{cfg.bermudan_nx}</Parameter>",
            f"      <Parameter name=\"sy\">{cfg.bermudan_sy}</Parameter>",
            f"      <Parameter name=\"ny\">{cfg.bermudan_ny}</Parameter>",
            "    </EngineParameters>",
            "  </Product>",
            "</PricingEngines>",
        ]
    )


def _todays_market_xml(
    cfg: TodaysMarketConfig,
    market_cfg: SimulationMarketConfig,
    counterparties: CounterpartyConfig | None,
    base_currency: str,
) -> str:
    # This generated todaysmarket.xml is intentionally narrow. It is good enough
    # for native smoke tests and fully in-memory demos, but it does not aim to
    # reproduce the full richness of a production ORE todaysmarket.xml.
    currencies = _resolve_market_currencies(
        cfg.discount_curve,
        market_cfg.currencies,
        cfg.fx_pairs,
        market_cfg.indices,
    )
    curve_ccy = _currency_from_curve_id(cfg.discount_curve) or (currencies[0] if currencies else "EUR")
    yield_curves_id = cfg.yield_curves_id or cfg.market_id
    discounting_curves_id = cfg.discounting_curves_id or cfg.market_id
    index_forwarding_curves_id = cfg.index_forwarding_curves_id or cfg.market_id
    fx_spots_id = cfg.fx_spots_id or cfg.market_id
    fx_volatilities_id = cfg.fx_volatilities_id or cfg.market_id
    swaption_volatilities_id = cfg.swaption_volatilities_id or cfg.market_id
    default_curves_id = cfg.default_curves_id or cfg.market_id
    fx_spots = "\n".join(
        f"    <FxSpot pair=\"{pair}\">FX/{pair[:3]}/{pair[3:]}</FxSpot>"
        for pair in cfg.fx_pairs
    ) or "    <FxSpot pair=\"EURUSD\">FX/EUR/USD</FxSpot>"
    discounting_curves = "\n".join(
        f"    <DiscountingCurve currency=\"{ccy}\">{_curve_path_for(cfg.discount_curve, ccy)}</DiscountingCurve>"
        for ccy in currencies
    ) or f"    <DiscountingCurve currency=\"{curve_ccy}\">{_curve_path_for(cfg.discount_curve, curve_ccy)}</DiscountingCurve>"
    forwarding_curves = "\n".join(
        f"    <Index name=\"{str(index).upper()}\">{_curve_path_for(cfg.discount_curve, _currency_from_index_name(index) or curve_ccy)}</Index>"
        for index in _ordered_unique([str(i).upper() for i in market_cfg.indices])
    ) or f"    <Index name=\"{_default_index_for_ccy(curve_ccy)}\">{_curve_path_for(cfg.discount_curve, curve_ccy)}</Index>"
    return "\n".join(
        [
            "<TodaysMarket>",
            f"  <Configuration id=\"{cfg.market_id}\">",
            f"    <YieldCurvesId>{yield_curves_id}</YieldCurvesId>",
            f"    <DiscountingCurvesId>{discounting_curves_id}</DiscountingCurvesId>",
            f"    <IndexForwardingCurvesId>{index_forwarding_curves_id}</IndexForwardingCurvesId>",
            f"    <FxSpotsId>{fx_spots_id}</FxSpotsId>",
            f"    <FxVolatilitiesId>{fx_volatilities_id}</FxVolatilitiesId>",
            f"    <SwaptionVolatilitiesId>{swaption_volatilities_id}</SwaptionVolatilitiesId>",
            f"    <DefaultCurvesId>{default_curves_id}</DefaultCurvesId>",
            "  </Configuration>",
            f"  <YieldCurves id=\"{yield_curves_id}\">",
            f"    <YieldCurve name=\"{cfg.discount_curve}\">{_curve_path_for(cfg.discount_curve, curve_ccy)}</YieldCurve>",
            "  </YieldCurves>",
            f"  <DiscountingCurves id=\"{discounting_curves_id}\">",
            discounting_curves,
            "  </DiscountingCurves>",
            f"  <IndexForwardingCurves id=\"{index_forwarding_curves_id}\">",
            forwarding_curves,
            "  </IndexForwardingCurves>",
            f"  <FxSpots id=\"{fx_spots_id}\">",
            fx_spots,
            "  </FxSpots>",
            f"  <FxVolatilities id=\"{fx_volatilities_id}\"/>",
            f"  <SwaptionVolatilities id=\"{swaption_volatilities_id}\"/>",
            f"  <DefaultCurves id=\"{default_curves_id}\">",
            *[
                f"    <DefaultCurve name=\"{name}\">Default/{_default_curve_currency(name, counterparties, base_currency)}/{name}</DefaultCurve>"
                for name in market_cfg.default_curve_names
            ],
            "  </DefaultCurves>",
            "</TodaysMarket>",
        ]
    )


def _curve_config_xml(
    curves: tuple,
    fallback_curve: str,
    default_curve_names: tuple[str, ...],
    counterparties: CounterpartyConfig | None = None,
    base_currency: str = "USD",
) -> str:
    # Generated curve configs are a compatibility scaffold. They are useful when
    # the caller wants an in-memory native demo, but they should not be treated
    # as equivalent to a case-authored ORE curveconfig.xml for parity work.
    if not curves:
        lines = [
            "<CurveConfiguration>",
            "  <YieldCurves>",
            "    <YieldCurve>",
            f"      <CurveId>{fallback_curve}</CurveId>",
            "      <CurveDescription>Auto-generated flat curve</CurveDescription>",
            f"      <Currency>{_currency_from_curve_id(fallback_curve) or 'EUR'}</Currency>",
            f"      <DiscountCurve>{fallback_curve}</DiscountCurve>",
            "      <Segments>",
            "        <Simple>",
            "          <Type>Deposit</Type>",
            f"          <Quotes><Quote>ZERO/RATE/{_currency_from_curve_id(fallback_curve) or 'EUR'}/1Y</Quote></Quotes>",
            f"          <Conventions>{_curve_convention_for_currency(_currency_from_curve_id(fallback_curve) or 'EUR')}</Conventions>",
            "        </Simple>",
            "      </Segments>",
            "      <InterpolationVariable>Discount</InterpolationVariable>",
            "      <InterpolationMethod>LogLinear</InterpolationMethod>",
            "      <YieldCurveDayCounter>A365</YieldCurveDayCounter>",
            "      <Extrapolation>true</Extrapolation>",
            "    </YieldCurve>",
            "  </YieldCurves>",
            "  <DefaultCurves>",
        ]
        for name in default_curve_names:
            curve_currency = _default_curve_currency(name, counterparties, base_currency)
            lines.extend(
                [
                    "    <DefaultCurve>",
                    f"      <CurveId>{name}</CurveId>",
                    "      <CurveDescription/>",
                    f"      <Currency>{curve_currency}</Currency>",
                    "      <Configurations>",
                    "        <Configuration priority=\"0\">",
                    "          <Type>SpreadCDS</Type>",
                    f"          <DiscountCurve>{_default_curve_discount_path(curves, fallback_curve)}</DiscountCurve>",
                    "          <DayCounter>Actual/365 (Fixed)</DayCounter>",
                    f"          <RecoveryRate>RECOVERY_RATE/RATE/{name}/SNRFOR/{curve_currency}</RecoveryRate>",
                    "          <Quotes>",
                    f"            <Quote>CDS/CREDIT_SPREAD/{name}/SNRFOR/{curve_currency}/*</Quote>",
                    "          </Quotes>",
                    "          <Conventions>CDS-STANDARD-CONVENTIONS</Conventions>",
                    "          <Extrapolation>true</Extrapolation>",
                    "          <AllowNegativeRates>false</AllowNegativeRates>",
                    "        </Configuration>",
                    "      </Configurations>",
                    "    </DefaultCurve>",
                ]
            )
        lines.extend(["  </DefaultCurves>", "</CurveConfiguration>"])
        return "\n".join(lines)
    lines = ["<CurveConfiguration>", "  <YieldCurves>"]
    for c in curves:
        quote_nodes = "".join(f"<Quote>ZERO/RATE/{c.currency}/{t}</Quote>" for t in c.tenors)
        lines.extend(
            [
                "    <YieldCurve>",
                f"      <CurveId>{c.curve_id}</CurveId>",
                "      <CurveDescription>Auto-generated flat curve</CurveDescription>",
                f"      <Currency>{c.currency}</Currency>",
                f"      <DiscountCurve>{c.curve_id}</DiscountCurve>",
                "      <Segments>",
                "        <Simple>",
                "          <Type>Deposit</Type>",
                f"          <Quotes>{quote_nodes}</Quotes>",
                f"          <Conventions>{_curve_convention_for_currency(c.currency)}</Conventions>",
                "        </Simple>",
                "      </Segments>",
                "      <InterpolationVariable>Discount</InterpolationVariable>",
                "      <InterpolationMethod>LogLinear</InterpolationMethod>",
                "      <YieldCurveDayCounter>A365</YieldCurveDayCounter>",
                "      <Extrapolation>true</Extrapolation>",
                "    </YieldCurve>",
            ]
        )
    lines.append("  </YieldCurves>")
    lines.append("  <DefaultCurves>")
    for name in default_curve_names:
        curve_currency = _default_curve_currency(name, counterparties, base_currency)
        lines.extend(
            [
                "    <DefaultCurve>",
                f"      <CurveId>{name}</CurveId>",
                "      <CurveDescription/>",
                f"      <Currency>{curve_currency}</Currency>",
                "      <Configurations>",
                "        <Configuration priority=\"0\">",
                "          <Type>SpreadCDS</Type>",
                f"          <DiscountCurve>{_default_curve_discount_path(curves, fallback_curve)}</DiscountCurve>",
                "          <DayCounter>Actual/365 (Fixed)</DayCounter>",
                f"          <RecoveryRate>RECOVERY_RATE/RATE/{name}/SNRFOR/{curve_currency}</RecoveryRate>",
                "          <Quotes>",
                f"            <Quote>CDS/CREDIT_SPREAD/{name}/SNRFOR/{curve_currency}/*</Quote>",
                "          </Quotes>",
                "          <Conventions>CDS-STANDARD-CONVENTIONS</Conventions>",
                "          <Extrapolation>true</Extrapolation>",
                "          <AllowNegativeRates>false</AllowNegativeRates>",
                "        </Configuration>",
                "      </Configurations>",
                "    </DefaultCurve>",
            ]
        )
    lines.extend(["  </DefaultCurves>", "</CurveConfiguration>"])
    return "\n".join(lines)


def _simulation_xml(
    cfg: SimulationConfig,
    market_cfg: SimulationMarketConfig,
    cam_cfg: CrossAssetModelConfig,
    num_paths: int,
) -> str:
    samples = num_paths if num_paths > 0 else cfg.samples
    strict_mode = bool(cfg.strict_template)
    grid = ",".join(cfg.dates) if cfg.dates else ("88,3M" if strict_mode else "1Y,2Y")
    cam_xml = _cross_asset_model_xml(cam_cfg)
    market_xml = _simulation_market_xml(market_cfg, strict_mode=strict_mode)
    calendar = cfg.calendar or _simulation_calendar(market_cfg, cam_cfg)
    return "\n".join(
        [
            "<Simulation>",
            "  <Parameters>",
            f"    <Discretization>{cfg.discretization}</Discretization>",
            f"    <Grid>{grid}</Grid>",
            f"    <Calendar>{calendar}</Calendar>",
            f"    <Sequence>{cfg.sequence}</Sequence>",
            f"    <Scenario>{cfg.scenario}</Scenario>",
            f"    <Seed>{cfg.seed}</Seed>",
            f"    <Samples>{samples}</Samples>",
            f"    <CloseOutLag>{cfg.closeout_lag}</CloseOutLag>",
            f"    <MporMode>{cfg.mpor_mode}</MporMode>",
            f"    <DayCounter>{cfg.day_counter}</DayCounter>",
            "  </Parameters>",
            cam_xml,
            market_xml,
            "</Simulation>",
        ]
    )


def _cross_asset_model_xml(cfg: CrossAssetModelConfig) -> str:
    currencies = "\n".join(f"    <Currency>{c}</Currency>" for c in cfg.currencies)
    ir_models = []
    for ccy in cfg.ir_model_ccys:
        ir_models.extend(
            [
                f"    <LGM ccy=\"{ccy}\">",
                f"      <CalibrationType>{cfg.ir_calibration_type}</CalibrationType>",
                f"      <Volatility><Calibrate>N</Calibrate><VolatilityType>Hagan</VolatilityType><ParamType>Constant</ParamType><TimeGrid/><InitialValue>{cfg.ir_volatility}</InitialValue></Volatility>",
                f"      <Reversion><Calibrate>N</Calibrate><ReversionType>HullWhite</ReversionType><ParamType>Constant</ParamType><TimeGrid/><InitialValue>{cfg.ir_reversion}</InitialValue></Reversion>",
                f"      <CalibrationSwaptions><Expiries>{','.join(cfg.ir_calibration_expiries)}</Expiries><Terms>{','.join(cfg.ir_calibration_terms)}</Terms><Strikes/></CalibrationSwaptions>",
                f"      <ParameterTransformation><ShiftHorizon>{cfg.ir_shift_horizon}</ShiftHorizon><Scaling>{cfg.ir_scaling}</Scaling></ParameterTransformation>",
                "    </LGM>",
            ]
        )
    fx_models = []
    for fccy in cfg.fx_model_ccys:
        fx_models.extend(
            [
                f"    <CrossCcyLGM foreignCcy=\"{fccy}\">",
                f"      <DomesticCcy>{cfg.domestic_ccy}</DomesticCcy>",
                f"      <CalibrationType>{cfg.fx_calibration_type}</CalibrationType>",
                f"      <Sigma><Calibrate>N</Calibrate><ParamType>Constant</ParamType><TimeGrid/><InitialValue>{cfg.fx_sigma}</InitialValue></Sigma>",
                f"      <CalibrationOptions><Expiries>{','.join(cfg.fx_calibration_expiries)}</Expiries><Strikes/></CalibrationOptions>",
                "    </CrossCcyLGM>",
            ]
        )
    equity_models = []
    for eq in cfg.equity_factors:
        time_grid = ",".join(eq.time_grid)
        expiries = ",".join(eq.expiries)
        strikes = ",".join(eq.strikes)
        initial_values = ",".join([str(eq.sigma)] * (len(eq.time_grid) + 1 if eq.time_grid else 1))
        equity_models.extend(
            [
                f"    <CrossAssetLGM name=\"{_xml_escape(eq.name)}\">",
                f"      <Currency>{eq.currency}</Currency>",
                f"      <CalibrationType>{eq.calibration_type}</CalibrationType>",
                "      <Sigma>",
                "        <Calibrate>N</Calibrate>",
                "        <ParamType>Piecewise</ParamType>",
                f"        <TimeGrid>{time_grid}</TimeGrid>",
                f"        <InitialValue>{initial_values}</InitialValue>",
                "      </Sigma>",
                "      <CalibrationOptions>",
                f"        <Expiries>{expiries}</Expiries>",
                f"        <Strikes>{strikes}</Strikes>",
                "      </CalibrationOptions>",
                "    </CrossAssetLGM>",
            ]
        )
    corrs = "\n".join(
        f"    <Correlation factor1=\"{f1}\" factor2=\"{f2}\">{v}</Correlation>"
        for f1, f2, v in cfg.correlations
    )
    return "\n".join(
        [
            "  <CrossAssetModel>",
            f"    <DomesticCcy>{cfg.domestic_ccy}</DomesticCcy>",
            "    <Currencies>",
            currencies,
            "    </Currencies>",
            f"    <BootstrapTolerance>{cfg.bootstrap_tolerance}</BootstrapTolerance>",
            "    <InterestRateModels>",
            *ir_models,
            "    </InterestRateModels>",
            "    <ForeignExchangeModels>",
            *fx_models,
            "    </ForeignExchangeModels>",
            "    <EquityModels>",
            *equity_models,
            "    </EquityModels>",
            "    <InstantaneousCorrelations>",
            corrs,
            "    </InstantaneousCorrelations>",
            "  </CrossAssetModel>",
        ]
    )


def _simulation_market_xml(cfg: SimulationMarketConfig, strict_mode: bool = False) -> str:
    currencies = "\n".join(f"      <Currency>{c}</Currency>" for c in cfg.currencies)
    indices = "\n".join(f"      <Index>{str(i).upper()}</Index>" for i in cfg.indices)
    names = "\n".join(f"        <Name>{n}</Name>" for n in cfg.default_curve_names)
    pairs = "\n".join(f"        <CurrencyPair>{p}</CurrencyPair>" for p in cfg.fx_pairs)
    swap_indices = _swap_indices_xml_lines(cfg.currencies, cfg.indices)
    yield_tenors = ",".join(cfg.yield_curve_tenors) if cfg.yield_curve_tenors else ""
    default_tenors = ",".join(cfg.default_curve_tenors) if cfg.default_curve_tenors else ""
    swaption_expiries = ",".join(cfg.swaption_expiries)
    swaption_terms = ",".join(cfg.swaption_terms)
    fxvol_expiries = ",".join(cfg.fxvol_expiries)
    return "\n".join(
        [
            "  <Market>",
            f"    <BaseCurrency>{cfg.base_currency}</BaseCurrency>",
            "    <Currencies>",
            currencies,
            "    </Currencies>",
            "    <YieldCurves>",
            "      <Configuration>",
            f"        <Tenors>{yield_tenors or ('1W,1M,2M,3M,4M,5M,6M,7M,8M,10M,1Y,15M,18M,21M,2Y,3Y,4Y,5Y,7Y,8Y,9Y,10Y,12Y,15Y,20Y,30Y' if strict_mode else '3M,6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y')}</Tenors>",
            f"        <Interpolation>{cfg.yield_curve_interpolation}</Interpolation>",
            f"        <Extrapolation>{'Y' if cfg.yield_curve_extrapolation else 'N'}</Extrapolation>",
            "      </Configuration>",
            "    </YieldCurves>",
            "    <Indices>",
            indices,
            "    </Indices>",
            "    <SwapIndices>",
            *swap_indices,
            "    </SwapIndices>",
            "    <DefaultCurves>",
            "      <Names>",
            names,
            "      </Names>",
            f"      <Tenors>{default_tenors or ('6M,1Y,2Y' if strict_mode else '1Y,2Y,5Y,10Y')}</Tenors>",
            f"      <SimulateSurvivalProbabilities>{str(cfg.default_simulate_survival_probabilities).lower()}</SimulateSurvivalProbabilities>",
            f"      <SimulateRecoveryRates>{str(cfg.default_simulate_recovery_rates).lower()}</SimulateRecoveryRates>",
            f"      <Calendars><Calendar name=\"\">{cfg.default_curve_calendar}</Calendar></Calendars>",
            f"      <Extrapolation>{cfg.default_curve_extrapolation}</Extrapolation>",
            "    </DefaultCurves>",
            "    <SwaptionVolatilities>",
            f"      <Simulate>{str(cfg.swaption_simulate).lower()}</Simulate>",
            f"      <ReactionToTimeDecay>{cfg.swaption_reaction_to_time_decay}</ReactionToTimeDecay>",
            "      <Currencies>",
            currencies,
            "      </Currencies>",
            f"      <Expiries>{swaption_expiries}</Expiries>",
            f"      <Terms>{swaption_terms}</Terms>",
            "    </SwaptionVolatilities>",
            "    <FxVolatilities>",
            f"      <Simulate>{str(cfg.fxvol_simulate).lower()}</Simulate>",
            f"      <ReactionToTimeDecay>{cfg.fxvol_reaction_to_time_decay}</ReactionToTimeDecay>",
            "      <CurrencyPairs>",
            pairs,
            "      </CurrencyPairs>",
            f"      <Expiries>{fxvol_expiries or ('6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y' if strict_mode else '1Y,2Y,5Y')}</Expiries>",
            "    </FxVolatilities>",
            "    <AggregationScenarioDataCurrencies>",
            currencies,
            "    </AggregationScenarioDataCurrencies>",
            "    <AggregationScenarioDataIndices>",
            indices,
            "    </AggregationScenarioDataIndices>",
            "  </Market>",
        ]
    )


def _credit_simulation_xml(cfg: CreditSimulationConfig) -> str:
    entities = list(cfg.entities)
    if not entities:
        entities = [
            {
                "name": ns,
                "factor_loading": 0.4898979485566356,
                "transition_matrix": cfg.transition_matrix_name,
                "initial_state": 4,
            }
            for ns in (cfg.netting_set_ids or ("CPTY_A",))
        ]
    matrix = (
        "0.8588,0.0976,0.0048,0.0000,0.0003,0.0000,0.0000,0.0000,"
        "0.0092,0.8487,0.0964,0.0036,0.0015,0.0002,0.0000,0.0004,"
        "0.0008,0.0224,0.8624,0.0609,0.0077,0.0021,0.0000,0.0002,"
        "0.0008,0.0037,0.0602,0.7916,0.0648,0.0130,0.0011,0.0019,"
        "0.0003,0.0008,0.0046,0.0402,0.7676,0.0788,0.0047,0.0140,"
        "0.0001,0.0004,0.0016,0.0053,0.0586,0.7607,0.0274,0.0660,"
        "0.0000,0.0000,0.0000,0.0100,0.0279,0.0538,0.5674,0.2535,"
        "0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000"
    )
    lines = [
        "<CreditSimulation>",
        "  <TransitionMatrices>",
        "    <TransitionMatrix>",
        f"      <Name>{cfg.transition_matrix_name}</Name>",
        f"      <Data t0=\"0.0\" t1=\"1.0\">{matrix}</Data>",
        "    </TransitionMatrix>",
        "  </TransitionMatrices>",
        "  <Entities>",
    ]
    for e in entities:
        name = e.name if hasattr(e, "name") else e["name"]
        factor = e.factor_loading if hasattr(e, "factor_loading") else e["factor_loading"]
        tm = e.transition_matrix if hasattr(e, "transition_matrix") else e["transition_matrix"]
        init_state = e.initial_state if hasattr(e, "initial_state") else e["initial_state"]
        lines.extend(
            [
                "    <Entity>",
                f"      <Name>{name}</Name>",
                f"      <FactorLoadings>{factor}</FactorLoadings>",
                f"      <TransitionMatrix>{tm}</TransitionMatrix>",
                f"      <InitialState>{init_state}</InitialState>",
                "    </Entity>",
            ]
        )
    lines.extend(
        [
            "  </Entities>",
            f"  <NettingSetIds>{','.join(cfg.netting_set_ids)}</NettingSetIds>",
            "  <Risk>",
            f"    <Market>{'Y' if cfg.market_risk else 'N'}</Market>",
            f"    <Credit>{'Y' if cfg.credit_risk else 'N'}</Credit>",
            f"    <ZeroMarketPnl>{'Y' if cfg.zero_market_pnl else 'N'}</ZeroMarketPnl>",
            f"    <DoubleDefault>{'Y' if cfg.double_default else 'N'}</DoubleDefault>",
            f"    <Evaluation>{cfg.evaluation}</Evaluation>",
            f"    <Seed>{cfg.seed}</Seed>",
            f"    <Paths>{cfg.paths}</Paths>",
            f"    <CreditMode>{cfg.credit_mode}</CreditMode>",
            f"    <LoanExposureMode>{cfg.loan_exposure_mode}</LoanExposureMode>",
            "  </Risk>",
            "</CreditSimulation>",
        ]
    )
    return "\n".join(lines)


def _conventions_xml(cfg: ConventionsConfig) -> str:
    return "\n".join(
        [
            "<Conventions>",
            "  <Swap>",
            f"    <DayCounter>{cfg.day_counter}</DayCounter>",
            f"    <Calendar>{cfg.calendar}</Calendar>",
            "  </Swap>",
            "</Conventions>",
        ]
    )


def _counterparty_xml(cfg: CounterpartyConfig) -> str:
    if not cfg.ids:
        return "<CounterpartyInformation><Counterparties/></CounterpartyInformation>"
    lines = ["<CounterpartyInformation>", "  <Counterparties>"]
    for cid in cfg.ids:
        credit_quality = str(cfg.credit_qualities.get(cid, "IG"))
        bacva_weight = cfg.bacva_risk_weights.get(cid, 0.05)
        saccr_weight = cfg.saccr_risk_weights.get(cid, 1.0)
        sacva_bucket = cfg.sacva_risk_buckets.get(cid, 2)
        lines.extend(
            [
                "    <Counterparty>",
                f"      <CounterpartyId>{cid}</CounterpartyId>",
                f"      <CreditQuality>{credit_quality}</CreditQuality>",
                f"      <BaCvaRiskWeight>{bacva_weight}</BaCvaRiskWeight>",
                f"      <SaCcrRiskWeight>{saccr_weight}</SaCcrRiskWeight>",
                f"      <SaCvaRiskBucket>{sacva_bucket}</SaCvaRiskBucket>",
                "    </Counterparty>",
            ]
        )
    lines.extend(["  </Counterparties>", "</CounterpartyInformation>"])
    return "\n".join(lines)


def _apply_num_paths_to_simulation_xml(xml: str, num_paths: int) -> str:
    if num_paths <= 0 or not xml:
        return xml
    if "<Samples>" in xml:
        return re.sub(r"(<Samples>)([^<]+)(</Samples>)", rf"\g<1>{num_paths}\g<3>", xml, count=1)
    if "<Parameters>" in xml:
        return xml.replace("<Parameters>", f"<Parameters><Samples>{num_paths}</Samples>", 1)
    return xml


def _resolve_mpor_config(snapshot: XVASnapshot, xml_buffers: Dict[str, str]) -> MporConfig:
    mode = str(snapshot.config.params.get("python.mpor_mode", "sticky")).strip().lower()
    if mode and mode != "sticky":
        raise MappingError(
            f"Unsupported python.mpor_mode '{mode}'. Only 'sticky' is supported."
        )

    override_period = str(snapshot.config.params.get("python.mpor_source_override", "")).strip()
    override_days = str(snapshot.config.params.get("python.mpor_days", "")).strip()
    has_python_override = bool(override_period or override_days)

    if snapshot.config.mpor != MporConfig() and not has_python_override:
        return snapshot.config.mpor

    period = ""
    source = "disabled"
    sim_xml = xml_buffers.get("simulation.xml", "")
    if sim_xml:
        try:
            root = ET.fromstring(sim_xml)
            period = (root.findtext("./Parameters/CloseOutLag") or "").strip()
            if period:
                source = "simulation.xml"
        except Exception as exc:
            warnings.warn(
                f"Failed to parse simulation.xml when resolving MPOR CloseOutLag: {exc}. "
                "Falling back to netting set or param overrides.",
                UserWarning,
                stacklevel=2,
            )

    for ns in snapshot.netting.netting_sets.values():
        if ns.margin_period_of_risk:
            period = str(ns.margin_period_of_risk).strip()
            source = f"netting:{ns.netting_set_id}"
            break

    if override_period:
        period = override_period
        source = "python.mpor_source_override"
    elif override_days:
        try:
            period = f"{max(int(float(override_days)), 0)}D"
            source = "python.mpor_days"
        except Exception as exc:
            warnings.warn(
                f"Failed to parse python.mpor_days={override_days!r} as a number: {exc}. "
                "The override will be ignored and MPOR remains disabled.",
                UserWarning,
                stacklevel=2,
            )

    years = _period_to_years(period)
    days = _period_to_days(period)
    return MporConfig(
        enabled=years > 0.0 or days > 0,
        mpor_years=years,
        mpor_days=days,
        closeout_lag_period=period,
        sticky=True,
        cashflow_mode="NonePay",
        source=source if (years > 0.0 or days > 0) else "disabled",
    )


def _apply_mpor_to_simulation_xml(xml: str, mpor: MporConfig) -> str:
    if not xml:
        return xml
    period = mpor.closeout_lag_period or "0D"
    injection = ""
    if "<CloseOutLag>" in xml:
        xml = re.sub(r"(<CloseOutLag>)([^<]*)(</CloseOutLag>)", rf"\g<1>{period}\g<3>", xml, count=1)
    else:
        injection += f"<CloseOutLag>{period}</CloseOutLag>"
    if "<MporMode>" in xml:
        xml = re.sub(r"(<MporMode>)([^<]*)(</MporMode>)", r"\g<1>StickyDate\g<3>", xml, count=1)
    else:
        injection += "<MporMode>StickyDate</MporMode>"
    if injection and "<Parameters>" in xml:
        xml = xml.replace("<Parameters>", f"<Parameters>{injection}", 1)
    return xml


def _period_to_years(period: str) -> float:
    p = str(period).strip().upper()
    if not p:
        return 0.0
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([DWMY])", p)
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "D":
        return n / 365.25
    if unit == "W":
        return (7.0 * n) / 365.25
    if unit == "M":
        return n / 12.0
    return n


def _period_to_days(period: str) -> int:
    p = str(period).strip().upper()
    if not p:
        return 0
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([DWMY])", p)
    if not m:
        return 0
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "D":
        return int(round(n))
    if unit == "W":
        return int(round(5.0 * n))
    if unit == "M":
        return int(round(21.0 * n))
    return int(round(252.0 * n))


def _maybe_set(target: object, method_name: str, value: Any) -> None:
    if value is None:
        return
    fn = getattr(target, method_name, None)
    if callable(fn):
        if method_name == "setMarketConfig":
            # value format: "context=config;context2=config2"
            for token in str(value).split(";"):
                token = token.strip()
                if not token or "=" not in token:
                    continue
                context, config = token.split("=", 1)
                fn(config.strip(), context.strip())
            return
        fn(value)


def _maybe_set_bool(target: object, method_name: str, value: bool) -> None:
    fn = getattr(target, method_name, None)
    if callable(fn):
        fn(bool(value))


def _market_config_args(params: Dict[str, str]) -> str | None:
    # Placeholder marker used by _maybe_set for setMarketConfig special handling.
    contexts = {
        k[len("market.") :]: v
        for k, v in params.items()
        if k.startswith("market.") and v
    }
    if not contexts:
        return None
    # Encoded as context=config pairs separated by ';'
    return ";".join(f"{ctx}={cfg}" for ctx, cfg in sorted(contexts.items()))


def _looks_like_refdata(xml: str | None) -> bool:
    if not xml:
        return False
    prefix = xml.lstrip()[:80]
    return "<ReferenceData" in prefix


def _extract_xml_section(xml: str | None, tag: str) -> str | None:
    if not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except Exception:
        return None
    if root.tag == tag:
        return xml
    node = root.find(f".//{tag}")
    if node is None:
        return None
    return ET.tostring(node, encoding="unicode")


def _apply_simulation_overrides(snapshot: XVASnapshot, input_parameters: InputParametersLike) -> None:
    runtime = snapshot.config.runtime
    if runtime is None:
        return
    simulation = runtime.simulation
    _maybe_set_bool(input_parameters, "setXvaCgDynamicIM", simulation.xva_cg_dynamic_im)
    _maybe_set(input_parameters, "setXvaCgDynamicIMStepSize", int(simulation.xva_cg_dynamic_im_step_size))
    if simulation.xva_cg_regression_order_dynamic_im is not None:
        _maybe_set(
            input_parameters,
            "setXvaCgRegressionOrderDynamicIm",
            int(simulation.xva_cg_regression_order_dynamic_im),
        )
    if simulation.xva_cg_regression_report_time_steps_dynamic_im:
        _maybe_set(
            input_parameters,
            "setXvaCgRegressionReportTimeStepsDynamicIM",
            list(simulation.xva_cg_regression_report_time_steps_dynamic_im),
        )
    _maybe_set_bool(input_parameters, "setStoreSensis", runtime.store_sensis)
    if runtime.curve_sensi_grid:
        _maybe_set(input_parameters, "setCurveSensiGrid", list(runtime.curve_sensi_grid))
    if runtime.vega_sensi_grid:
        _maybe_set(input_parameters, "setVegaSensiGrid", list(runtime.vega_sensi_grid))


def _apply_xva_analytic_overrides(snapshot: XVASnapshot, input_parameters: InputParametersLike) -> None:
    runtime = snapshot.config.runtime
    if runtime is None:
        return
    xva = runtime.xva_analytic
    _maybe_set(input_parameters, "setExposureAllocationMethod", xva.exposure_allocation_method)
    _maybe_set(input_parameters, "setExposureObservationModel", xva.exposure_observation_model)
    _maybe_set(input_parameters, "setObservationModel", xva.exposure_observation_model)
    _maybe_set(input_parameters, "setPfeQuantile", float(xva.pfe_quantile))
    _maybe_set(input_parameters, "setCollateralCalculationType", xva.collateral_calculation_type)
    _maybe_set(input_parameters, "setMarginalAllocationLimit", float(xva.marginal_allocation_limit))
    _maybe_set_bool(input_parameters, "setExerciseNextBreak", xva.exercise_next_break)
    _maybe_set(input_parameters, "setScenarioGenType", xva.scenario_gen_type)
    if xva.netting_set_ids:
        _maybe_set(input_parameters, "setNettingSetId", ",".join(xva.netting_set_ids))
    if xva.full_initial_collateralisation is not None:
        _maybe_set_bool(input_parameters, "setFullInitialCollateralisation", bool(xva.full_initial_collateralisation))
    if xva.flip_view_xva is not None:
        _maybe_set_bool(input_parameters, "setFlipViewXVA", bool(xva.flip_view_xva))
    if xva.collateral_floor_enabled is not None:
        _maybe_set_bool(input_parameters, "setCollateralFloorAnalytic", bool(xva.collateral_floor_enabled))
    if xva.dim_model is not None:
        _maybe_set(input_parameters, "setDimModel", str(xva.dim_model))
    if xva.dim_quantile is not None:
        _maybe_set(input_parameters, "setDimQuantile", float(xva.dim_quantile))
    if xva.dim_horizon_calendar_days is not None:
        _maybe_set(input_parameters, "setDimHorizonCalendarDays", int(xva.dim_horizon_calendar_days))
    if xva.dim_regression_order is not None:
        _maybe_set(input_parameters, "setDimRegressionOrder", int(xva.dim_regression_order))
    if xva.dim_regressors is not None:
        _maybe_set(input_parameters, "setDimRegressors", str(xva.dim_regressors))
    if xva.dim_evolution_file is not None:
        _maybe_set(input_parameters, "setDimEvolutionFile", str(xva.dim_evolution_file))
    if xva.dim_regression_files is not None:
        _maybe_set(input_parameters, "setDimRegressionFiles", str(xva.dim_regression_files))
    if xva.dim_output_grid_points is not None:
        _maybe_set(input_parameters, "setDimOutputGridPoints", str(xva.dim_output_grid_points))
    if xva.dim_output_netting_set is not None:
        _maybe_set(input_parameters, "setDimOutputNettingSet", str(xva.dim_output_netting_set))
    if xva.dim_local_regression_evaluations is not None:
        _maybe_set(input_parameters, "setDimLocalRegressionEvaluations", int(xva.dim_local_regression_evaluations))
    if xva.dim_local_regression_bandwidth is not None:
        _maybe_set(input_parameters, "setDimLocalRegressionBandwidth", float(xva.dim_local_regression_bandwidth))
    if xva.flip_view_borrowing_curve_postfix is not None:
        _maybe_set(input_parameters, "setFlipViewBorrowingCurvePostfix", str(xva.flip_view_borrowing_curve_postfix))
    if xva.flip_view_lending_curve_postfix is not None:
        _maybe_set(input_parameters, "setFlipViewLendingCurvePostfix", str(xva.flip_view_lending_curve_postfix))
    if xva.dva_name:
        _maybe_set(input_parameters, "setDvaName", xva.dva_name)
    if xva.fva_borrowing_curve:
        _maybe_set(input_parameters, "setFvaBorrowingCurve", xva.fva_borrowing_curve)
    if xva.fva_lending_curve:
        _maybe_set(input_parameters, "setFvaLendingCurve", xva.fva_lending_curve)
    for method, val in (
        ("setCvaAnalytic", xva.cva_enabled),
        ("setDvaAnalytic", xva.dva_enabled),
        ("setFvaAnalytic", xva.fva_enabled),
        ("setMvaAnalytic", xva.mva_enabled),
        ("setColvaAnalytic", xva.colva_enabled),
        ("setDimAnalytic", xva.dim_enabled),
        ("setDynamicCredit", xva.dynamic_credit_enabled),
    ):
        if val is not None:
            _maybe_set_bool(input_parameters, method, bool(val))


def _apply_credit_simulation(snapshot: XVASnapshot, mapped: MappedInputs, input_parameters: InputParametersLike) -> None:
    runtime = snapshot.config.runtime
    if runtime is None or not runtime.credit_simulation.enabled:
        return
    _maybe_set(input_parameters, "setCreditSimulationParametersFromBuffer", mapped.xml_buffers.get("creditsimulation.xml"))
