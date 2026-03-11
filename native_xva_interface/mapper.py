from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, Protocol

from .dataclasses import (
    BermudanSwaption,
    CollateralConfig,
    ConventionsConfig,
    CounterpartyConfig,
    CreditSimulationConfig,
    CrossAssetModelConfig,
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
from .exceptions import MappingError


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
        xml_buffers["portfolio.xml"] = _portfolio_to_xml(snapshot.portfolio, snapshot.config.asof)
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
    _apply_xva_analytic_overrides(snapshot, input_parameters)
    _apply_credit_simulation(snapshot, mapped, input_parameters)
    return input_parameters


def _portfolio_to_xml(portfolio: Portfolio, asof: str) -> str:
    lines = ["<Portfolio>"]
    for t in portfolio.trades:
        lines.append(f'  <Trade id="{t.trade_id}">')
        lines.append(f"    <TradeType>{t.trade_type}</TradeType>")
        lines.append("    <Envelope>")
        lines.append(f"      <CounterParty>{t.counterparty}</CounterParty>")
        lines.append(f"      <NettingSetId>{t.netting_set}</NettingSetId>")
        lines.append("      <AdditionalFields/>")
        lines.append("    </Envelope>")
        lines.extend(_product_xml(t, asof))
        lines.append("  </Trade>")
    lines.append("</Portfolio>")
    return "\n".join(lines)


def _product_xml(trade: Trade, asof: str) -> List[str]:
    p = trade.product
    if isinstance(p, IRS):
        start_date = _fmt_yyyymmdd(asof)
        end_date = _add_months_yyyymmdd(start_date, int(round(p.maturity_years * 12.0)))
        payer_fixed = str(p.pay_fixed).lower()
        payer_float = str(not p.pay_fixed).lower()
        # This branch is intentionally a convenience fallback, not a full ORE
        # trade serializer. It bakes in generic conventions (TARGET, A360, 6M
        # fixed / 3M float, default index by currency). That is acceptable for
        # lightweight native demos, but it is not sufficient for strict parity
        # against a real ORE portfolio with bespoke schedules or conventions.
        idx = _default_index_for_ccy(p.ccy)
        return [
            "    <SwapData>",
            "      <StartDate>" + start_date + "</StartDate>",
            "      <EndDate>" + end_date + "</EndDate>",
            "      <LegData>",
            "        <LegType>Fixed</LegType>",
            f"        <Payer>{payer_fixed}</Payer>",
            f"        <Currency>{p.ccy}</Currency>",
            "        <DayCounter>A360</DayCounter>",
            "        <PaymentConvention>MF</PaymentConvention>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            "            <Tenor>6M</Tenor>",
            "            <Calendar>TARGET</Calendar>",
            "            <Convention>MF</Convention>",
            "            <TermConvention>MF</TermConvention>",
            "            <Rule>Forward</Rule>",
            "            <EndOfMonth/>",
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
            "        <DayCounter>A360</DayCounter>",
            "        <PaymentConvention>MF</PaymentConvention>",
            "        <Notionals>",
            f"          <Notional>{p.notional}</Notional>",
            "        </Notionals>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            "            <Tenor>3M</Tenor>",
            "            <Calendar>TARGET</Calendar>",
            "            <Convention>MF</Convention>",
            "            <TermConvention>MF</TermConvention>",
            "            <Rule>Forward</Rule>",
            "            <EndOfMonth/>",
            "          </Rules>",
            "        </ScheduleData>",
            "        <FloatingLegData>",
            f"          <Index>{idx}</Index>",
            "          <FixingDays>2</FixingDays>",
            "          <Spreads>",
            "            <Spread>0.0</Spread>",
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
        return [
            "    <FxForwardData>",
            f"      <BoughtCurrency>{base if p.buy_base else quote}</BoughtCurrency>",
            f"      <BoughtAmount>{bought}</BoughtAmount>",
            f"      <SoldCurrency>{quote if p.buy_base else base}</SoldCurrency>",
            f"      <SoldAmount>{sold}</SoldAmount>",
            "      <ValueDate>2030-01-01</ValueDate>",
            "    </FxForwardData>",
        ]
    if isinstance(p, BermudanSwaption):
        start_date = _fmt_yyyymmdd(asof)
        end_date = _add_months_yyyymmdd(start_date, int(round(p.maturity_years * 12.0)))
        payer_fixed = str(p.pay_fixed).lower()
        payer_float = str(not p.pay_fixed).lower()
        float_index = p.float_index or _default_index_for_ccy(p.ccy)
        exercise_dates = "\n".join(f"          <ExerciseDate>{d}</ExerciseDate>" for d in p.exercise_dates)
        return [
            "    <SwaptionData>",
            "      <OptionData>",
            f"        <LongShort>{p.long_short}</LongShort>",
            f"        <OptionType>{p.option_type}</OptionType>",
            "        <Style>Bermudan</Style>",
            f"        <Settlement>{p.settlement}</Settlement>",
            "        <PayOffAtExpiry>false</PayOffAtExpiry>",
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
            "        <DayCounter>A360</DayCounter>",
            "        <PaymentConvention>ModifiedFollowing</PaymentConvention>",
            "        <FloatingLegData>",
            f"          <Index>{float_index}</Index>",
            "          <Spreads>",
            "            <Spread>0.0</Spread>",
            "          </Spreads>",
            "          <FixingDays>2</FixingDays>",
            "          <IsInArrears>false</IsInArrears>",
            "        </FloatingLegData>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            "            <Tenor>6M</Tenor>",
            "            <Calendar>TARGET</Calendar>",
            "            <Convention>Following</Convention>",
            "            <TermConvention>Following</TermConvention>",
            "            <Rule>Forward</Rule>",
            "            <EndOfMonth/>",
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
            "        <DayCounter>ACT/ACT</DayCounter>",
            "        <PaymentConvention>Following</PaymentConvention>",
            "        <FixedLegData>",
            "          <Rates>",
            f"            <Rate>{p.fixed_rate}</Rate>",
            "          </Rates>",
            "        </FixedLegData>",
            "        <ScheduleData>",
            "          <Rules>",
            f"            <StartDate>{start_date}</StartDate>",
            f"            <EndDate>{end_date}</EndDate>",
            "            <Tenor>1Y</Tenor>",
            "            <Calendar>TARGET</Calendar>",
            "            <Convention>Following</Convention>",
            "            <TermConvention>Following</TermConvention>",
            "            <Rule>Forward</Rule>",
            "            <EndOfMonth/>",
            "          </Rules>",
            "        </ScheduleData>",
            "      </LegData>",
            "    </SwaptionData>",
        ]
    return ["    <Data/>", f"    <!-- Generic product payload omitted for {type(p).__name__} -->"]


def _netting_to_xml(netting: NettingConfig) -> str:
    lines = ["<NettingSetDefinitions>"]
    for ns_id, ns in netting.netting_sets.items():
        lines.append("  <NettingSet>")
        lines.append(f"    <NettingSetId>{ns_id}</NettingSetId>")
        lines.append(f"    <ActiveCSAFlag>{str(ns.active_csa).lower()}</ActiveCSAFlag>")
        lines.append("    <CSADetails>")
        lines.append(f"      <CSACurrency>{ns.csa_currency or 'EUR'}</CSACurrency>")
        lines.append(f"      <ThresholdPay>{ns.threshold_pay or 0.0}</ThresholdPay>")
        lines.append(f"      <ThresholdReceive>{ns.threshold_receive or 0.0}</ThresholdReceive>")
        lines.append(f"      <MinimumTransferAmountPay>{ns.mta_pay or 0.0}</MinimumTransferAmountPay>")
        lines.append(f"      <MinimumTransferAmountReceive>{ns.mta_receive or 0.0}</MinimumTransferAmountReceive>")
        lines.append(f"      <MarginPeriodOfRisk>{ns.margin_period_of_risk or '0D'}</MarginPeriodOfRisk>")
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


def _default_index_for_ccy(ccy: str) -> str:
    c = ccy.upper()
    if c == "USD":
        return "USD-LIBOR-3M"
    if c == "GBP":
        return "GBP-LIBOR-3M"
    if c == "CHF":
        return "CHF-LIBOR-3M"
    if c == "JPY":
        return "JPY-LIBOR-3M"
    return "EUR-EURIBOR-3M"


def _runtime_xml_buffers(snapshot: XVASnapshot) -> Dict[str, str]:
    runtime = snapshot.config.runtime or RuntimeConfig()
    # Strict template mode uses a fully in-memory compatibility template set
    # to provide complete XVA runtime wiring without external file loading.
    if runtime.simulation.strict_template:
        try:
            from .stress_classic_templates import stress_classic_xml_buffers

            out = dict(stress_classic_xml_buffers(num_paths=snapshot.config.num_paths))
            if runtime.credit_simulation.enabled:
                out["creditsimulation.xml"] = _credit_simulation_xml(runtime.credit_simulation)
            return out
        except Exception:
            # Fall back to generated builders below if compatibility templates
            # are unavailable in the runtime environment.
            pass
    out = {
        "pricingengine.xml": _pricing_engine_xml(runtime.pricing_engine),
        "todaysmarket.xml": _todays_market_xml(runtime.todays_market, runtime.simulation_market.default_curve_names),
        "curveconfig.xml": _curve_config_xml(
            runtime.curve_configs,
            runtime.todays_market.discount_curve,
            runtime.simulation_market.default_curve_names,
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
            f"    <Model>{cfg.model}</Model>",
            "    <ModelParameters/>",
            f"    <Engine>{cfg.npv_engine}</Engine>",
            "    <EngineParameters/>",
            "  </Product>",
            "  <Product type=\"Swap\">",
            "    <Model>DiscountedCashflows</Model>",
            "    <ModelParameters/>",
            "    <Engine>DiscountingSwapEngine</Engine>",
            "    <EngineParameters/>",
            "  </Product>",
            "</PricingEngines>",
        ]
    )


def _todays_market_xml(cfg: TodaysMarketConfig, default_curve_names: tuple[str, ...]) -> str:
    # This generated todaysmarket.xml is intentionally narrow. It is good enough
    # for native smoke tests and fully in-memory demos, but it does not aim to
    # reproduce the full richness of a production ORE todaysmarket.xml.
    fx_spots = "\n".join(
        f"    <FxSpot pair=\"{pair}\">FX/{pair[:3]}/{pair[3:]}</FxSpot>"
        for pair in cfg.fx_pairs
    ) or "    <FxSpot pair=\"EURUSD\">FX/EUR/USD</FxSpot>"
    return "\n".join(
        [
            "<TodaysMarket>",
            f"  <Configuration id=\"{cfg.market_id}\">",
            f"    <YieldCurvesId>{cfg.market_id}</YieldCurvesId>",
            f"    <DiscountingCurvesId>{cfg.market_id}</DiscountingCurvesId>",
            f"    <IndexForwardingCurvesId>{cfg.market_id}</IndexForwardingCurvesId>",
            f"    <FxSpotsId>{cfg.market_id}</FxSpotsId>",
            f"    <FxVolatilitiesId>{cfg.market_id}</FxVolatilitiesId>",
            f"    <SwaptionVolatilitiesId>{cfg.market_id}</SwaptionVolatilitiesId>",
            f"    <DefaultCurvesId>{cfg.market_id}</DefaultCurvesId>",
            "  </Configuration>",
            f"  <YieldCurves id=\"{cfg.market_id}\">",
            f"    <YieldCurve name=\"{cfg.discount_curve}\">Yield/EUR/{cfg.discount_curve}</YieldCurve>",
            "  </YieldCurves>",
            f"  <DiscountingCurves id=\"{cfg.market_id}\">",
            f"    <DiscountingCurve currency=\"EUR\">Yield/EUR/{cfg.discount_curve}</DiscountingCurve>",
            "  </DiscountingCurves>",
            f"  <IndexForwardingCurves id=\"{cfg.market_id}\">",
            f"    <Index name=\"EUR-ESTER\">Yield/EUR/{cfg.discount_curve}</Index>",
            "  </IndexForwardingCurves>",
            f"  <FxSpots id=\"{cfg.market_id}\">",
            fx_spots,
            "  </FxSpots>",
            f"  <FxVolatilities id=\"{cfg.market_id}\"/>",
            f"  <SwaptionVolatilities id=\"{cfg.market_id}\"/>",
            f"  <DefaultCurves id=\"{cfg.market_id}\">",
            *[
                f"    <DefaultCurve name=\"{name}\">Default/USD/{name}</DefaultCurve>"
                for name in default_curve_names
            ],
            "  </DefaultCurves>",
            "</TodaysMarket>",
        ]
    )


def _curve_config_xml(curves: tuple, fallback_curve: str, default_curve_names: tuple[str, ...]) -> str:
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
            "      <Currency>EUR</Currency>",
            f"      <DiscountCurve>{fallback_curve}</DiscountCurve>",
            "      <Segments>",
            "        <Simple>",
            "          <Type>Deposit</Type>",
            "          <Quotes><Quote>ZERO/RATE/EUR/1Y</Quote></Quotes>",
            "          <Conventions>EUR-ON-DEPOSIT-ESTER</Conventions>",
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
            lines.extend(
                [
                    "    <DefaultCurve>",
                    f"      <CurveId>{name}</CurveId>",
                    "      <CurveDescription/>",
                    "      <Currency>USD</Currency>",
                    "      <Configurations>",
                    "        <Configuration priority=\"0\">",
                    "          <Type>SpreadCDS</Type>",
                    "          <DiscountCurve>Yield/USD/USD-SOFR</DiscountCurve>",
                    "          <DayCounter>Actual/365 (Fixed)</DayCounter>",
                    f"          <RecoveryRate>RECOVERY_RATE/RATE/{name}/SNRFOR/USD</RecoveryRate>",
                    "          <Quotes>",
                    f"            <Quote>CDS/CREDIT_SPREAD/{name}/SNRFOR/USD/*</Quote>",
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
                "          <Conventions>EUR-ON-DEPOSIT-ESTER</Conventions>",
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
        lines.extend(
            [
                "    <DefaultCurve>",
                f"      <CurveId>{name}</CurveId>",
                "      <CurveDescription/>",
                "      <Currency>USD</Currency>",
                "      <Configurations>",
                "        <Configuration priority=\"0\">",
                "          <Type>SpreadCDS</Type>",
                "          <DiscountCurve>Yield/USD/USD-SOFR</DiscountCurve>",
                "          <DayCounter>Actual/365 (Fixed)</DayCounter>",
                f"          <RecoveryRate>RECOVERY_RATE/RATE/{name}/SNRFOR/USD</RecoveryRate>",
                "          <Quotes>",
                f"            <Quote>CDS/CREDIT_SPREAD/{name}/SNRFOR/USD/*</Quote>",
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
    if cfg.strict_template:
        return _simulation_xml_strict(cfg, num_paths)
    samples = num_paths if num_paths > 0 else cfg.samples
    grid = ",".join(cfg.dates) if cfg.dates else "1Y,2Y"
    cam_xml = _cross_asset_model_xml(cam_cfg)
    market_xml = _simulation_market_xml(market_cfg)
    return "\n".join(
        [
            "<Simulation>",
            "  <Parameters>",
            "    <Discretization>Exact</Discretization>",
            f"    <Grid>{grid}</Grid>",
            "    <Calendar>EUR,USD</Calendar>",
            "    <Sequence>SobolBrownianBridge</Sequence>",
            "    <Scenario>Simple</Scenario>",
            f"    <Samples>{samples}</Samples>",
            f"    <Seed>{cfg.seed}</Seed>",
            "    <CloseOutLag>2W</CloseOutLag>",
            "    <MporMode>StickyDate</MporMode>",
            "    <DayCounter>A365F</DayCounter>",
            "  </Parameters>",
            cam_xml,
            market_xml,
            "</Simulation>",
        ]
    )


def _simulation_xml_strict(cfg: SimulationConfig, num_paths: int) -> str:
    samples = num_paths if num_paths > 0 else cfg.samples
    grid = ",".join(cfg.dates) if cfg.dates else "88,3M"
    return f"""<Simulation>
  <Parameters>
    <Discretization>Exact</Discretization>
    <Grid>{grid}</Grid>
    <Calendar>EUR,USD</Calendar>
    <Sequence>SobolBrownianBridge</Sequence>
    <Scenario>Simple</Scenario>
    <Seed>{cfg.seed}</Seed>
    <Samples>{samples}</Samples>
    <CloseOutLag>2W</CloseOutLag>
    <MporMode>StickyDate</MporMode>
    <DayCounter>A365F</DayCounter>
  </Parameters>
  <CrossAssetModel>
    <DomesticCcy>EUR</DomesticCcy>
    <Currencies>
      <Currency>EUR</Currency>
      <Currency>USD</Currency>
    </Currencies>
    <BootstrapTolerance>0.0001</BootstrapTolerance>
    <InterestRateModels>
      <LGM ccy="default">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility><Calibrate>Y</Calibrate><VolatilityType>Hagan</VolatilityType><ParamType>Piecewise</ParamType><TimeGrid>1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0</TimeGrid><InitialValue>0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01</InitialValue></Volatility>
        <Reversion><Calibrate>N</Calibrate><ReversionType>HullWhite</ReversionType><ParamType>Constant</ParamType><TimeGrid/><InitialValue>0.03</InitialValue></Reversion>
        <CalibrationSwaptions><Expiries> 1Y,  2Y,  4Y,  6Y,  8Y, 10Y, 12Y, 14Y, 16Y, 18Y, 19Y</Expiries><Terms>   19Y, 18Y, 16Y, 14Y, 12Y, 10Y,  8Y,  6Y,  4Y,  2Y,  1Y</Terms><Strikes/></CalibrationSwaptions>
        <ParameterTransformation><ShiftHorizon>20.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
      </LGM>
      <LGM ccy="EUR">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility><Calibrate>Y</Calibrate><VolatilityType>Hagan</VolatilityType><ParamType>Piecewise</ParamType><TimeGrid>1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0</TimeGrid><InitialValue>0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01</InitialValue></Volatility>
        <Reversion><Calibrate>N</Calibrate><ReversionType>HullWhite</ReversionType><ParamType>Constant</ParamType><TimeGrid/><InitialValue>0.03</InitialValue></Reversion>
        <CalibrationSwaptions><Expiries> 1Y,  2Y,  4Y,  6Y,  8Y, 10Y, 12Y, 14Y, 16Y, 18Y, 19Y</Expiries><Terms>   19Y, 18Y, 16Y, 14Y, 12Y, 10Y,  8Y,  6Y,  4Y,  2Y,  1Y</Terms><Strikes/></CalibrationSwaptions>
        <ParameterTransformation><ShiftHorizon>20.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
      </LGM>
    </InterestRateModels>
    <ForeignExchangeModels>
      <CrossCcyLGM foreignCcy="default"><DomesticCcy>EUR</DomesticCcy><CalibrationType>Bootstrap</CalibrationType><Sigma><Calibrate>Y</Calibrate><ParamType>Piecewise</ParamType><TimeGrid>1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0</TimeGrid><InitialValue>0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1</InitialValue></Sigma><CalibrationOptions><Expiries>1Y, 2Y, 3Y, 4Y, 5Y, 10Y</Expiries><Strikes/></CalibrationOptions></CrossCcyLGM>
      <CrossCcyLGM foreignCcy="USD"><DomesticCcy>EUR</DomesticCcy><CalibrationType>Bootstrap</CalibrationType><Sigma><Calibrate>Y</Calibrate><ParamType>Piecewise</ParamType><TimeGrid>1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0</TimeGrid><InitialValue>0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1</InitialValue></Sigma><CalibrationOptions><Expiries>1Y, 2Y, 3Y, 4Y, 5Y, 10Y</Expiries><Strikes/></CalibrationOptions></CrossCcyLGM>
    </ForeignExchangeModels>
    <InstantaneousCorrelations>
      <Correlation factor1="IR:EUR" factor2="IR:USD">0</Correlation>
      <Correlation factor1="IR:EUR" factor2="FX:USDEUR">0</Correlation>
      <Correlation factor1="IR:USD" factor2="FX:USDEUR">0</Correlation>
    </InstantaneousCorrelations>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>EUR</BaseCurrency>
    <Currencies><Currency>EUR</Currency><Currency>USD</Currency></Currencies>
    <YieldCurves><Configuration><Tenors>1W, 1M, 2M, 3M, 4M, 5M, 6M, 7M, 8M, 10M, 1Y, 15M, 18M, 21M, 2Y, 3Y,4Y,5Y,7Y,8Y,9Y,10Y,12Y,15Y,20Y,30Y</Tenors><Interpolation>LogLinear</Interpolation><Extrapolation>Y</Extrapolation></Configuration></YieldCurves>
    <Indices><Index>EUR-EURIBOR-6M</Index><Index>EUR-EURIBOR-3M</Index><Index>EUR-EONIA</Index><Index>USD-FedFunds</Index><Index>USD-LIBOR-3M</Index><Index>USD-LIBOR-6M</Index></Indices>
    <SwapIndices><SwapIndex><Name>EUR-CMS-1Y</Name><DiscountingIndex>EUR-EONIA</DiscountingIndex></SwapIndex><SwapIndex><Name>EUR-CMS-30Y</Name><DiscountingIndex>EUR-EONIA</DiscountingIndex></SwapIndex><SwapIndex><Name>USD-CMS-1Y</Name><DiscountingIndex>USD-FedFunds</DiscountingIndex></SwapIndex><SwapIndex><Name>USD-CMS-30Y</Name><DiscountingIndex>USD-FedFunds</DiscountingIndex></SwapIndex></SwapIndices>
    <DefaultCurves><Names/><Tenors>6M,1Y,2Y</Tenors></DefaultCurves>
    <SwaptionVolatilities><Simulate>false</Simulate><ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay><Currencies><Currency>EUR</Currency><Currency>USD</Currency></Currencies><Expiries>6M,1Y,2Y,3Y,5Y,10Y,12Y,15Y,20Y</Expiries><Terms>1Y,2Y,3Y,4Y,5Y,7Y,10Y,15Y,20Y,30Y</Terms></SwaptionVolatilities>
    <FxVolatilities><Simulate>false</Simulate><ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay><CurrencyPairs><CurrencyPair>USDEUR</CurrencyPair></CurrencyPairs><Expiries>6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y</Expiries></FxVolatilities>
    <AggregationScenarioDataCurrencies><Currency>EUR</Currency><Currency>USD</Currency></AggregationScenarioDataCurrencies>
    <AggregationScenarioDataIndices><Index>EUR-EURIBOR-3M</Index><Index>EUR-EONIA</Index><Index>USD-LIBOR-3M</Index></AggregationScenarioDataIndices>
  </Market>
</Simulation>"""


def _cross_asset_model_xml(cfg: CrossAssetModelConfig) -> str:
    currencies = "\n".join(f"    <Currency>{c}</Currency>" for c in cfg.currencies)
    ir_models = []
    for ccy in cfg.ir_model_ccys:
        ir_models.extend(
            [
                f"    <LGM ccy=\"{ccy}\">",
                "      <CalibrationType>Bootstrap</CalibrationType>",
                "      <Volatility><Calibrate>N</Calibrate><VolatilityType>Hagan</VolatilityType><ParamType>Constant</ParamType><TimeGrid/><InitialValue>0.01</InitialValue></Volatility>",
                "      <Reversion><Calibrate>N</Calibrate><ReversionType>HullWhite</ReversionType><ParamType>Constant</ParamType><TimeGrid/><InitialValue>0.0</InitialValue></Reversion>",
                "      <CalibrationSwaptions><Expiries>1Y</Expiries><Terms>5Y</Terms><Strikes/></CalibrationSwaptions>",
                "      <ParameterTransformation><ShiftHorizon>20.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>",
                "    </LGM>",
            ]
        )
    fx_models = []
    for fccy in cfg.fx_model_ccys:
        fx_models.extend(
            [
                f"    <CrossCcyLGM foreignCcy=\"{fccy}\">",
                f"      <DomesticCcy>{cfg.domestic_ccy}</DomesticCcy>",
                "      <CalibrationType>Bootstrap</CalibrationType>",
                "      <Sigma><Calibrate>N</Calibrate><ParamType>Constant</ParamType><TimeGrid/><InitialValue>0.1</InitialValue></Sigma>",
                "      <CalibrationOptions><Expiries>1Y</Expiries><Strikes/></CalibrationOptions>",
                "    </CrossCcyLGM>",
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
            "    <BootstrapTolerance>0.0001</BootstrapTolerance>",
            "    <InterestRateModels>",
            *ir_models,
            "    </InterestRateModels>",
            "    <ForeignExchangeModels>",
            *fx_models,
            "    </ForeignExchangeModels>",
            "    <InstantaneousCorrelations>",
            corrs,
            "    </InstantaneousCorrelations>",
            "  </CrossAssetModel>",
        ]
    )


def _simulation_market_xml(cfg: SimulationMarketConfig) -> str:
    currencies = "\n".join(f"      <Currency>{c}</Currency>" for c in cfg.currencies)
    indices = "\n".join(f"      <Index>{i}</Index>" for i in cfg.indices)
    names = "\n".join(f"        <Name>{n}</Name>" for n in cfg.default_curve_names)
    pairs = "\n".join(f"        <CurrencyPair>{p}</CurrencyPair>" for p in cfg.fx_pairs)
    return "\n".join(
        [
            "  <Market>",
            f"    <BaseCurrency>{cfg.base_currency}</BaseCurrency>",
            "    <Currencies>",
            currencies,
            "    </Currencies>",
            "    <YieldCurves>",
            "      <Configuration>",
            "        <Tenors>3M,6M,1Y,2Y,3Y,4Y,5Y,7Y,10Y,12Y,15Y,20Y</Tenors>",
            "        <Interpolation>LogLinear</Interpolation>",
            "        <Extrapolation>Y</Extrapolation>",
            "      </Configuration>",
            "    </YieldCurves>",
            "    <Indices>",
            indices,
            "    </Indices>",
            "    <SwapIndices>",
            "      <SwapIndex><Name>EUR-CMS-1Y</Name><DiscountingIndex>EUR-ESTER</DiscountingIndex></SwapIndex>",
            "      <SwapIndex><Name>EUR-CMS-30Y</Name><DiscountingIndex>EUR-ESTER</DiscountingIndex></SwapIndex>",
            "    </SwapIndices>",
            "    <DefaultCurves>",
            "      <Names>",
            names,
            "      </Names>",
            "      <Tenors>1Y,2Y,5Y,10Y</Tenors>",
            "      <SimulateSurvivalProbabilities>true</SimulateSurvivalProbabilities>",
            "      <SimulateRecoveryRates>true</SimulateRecoveryRates>",
            "      <Calendars><Calendar name=\"\">TARGET</Calendar></Calendars>",
            "      <Extrapolation>FlatZero</Extrapolation>",
            "    </DefaultCurves>",
            "    <SwaptionVolatilities>",
            "      <ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay>",
            "      <Currencies>",
            currencies,
            "      </Currencies>",
            "      <Expiries>6M,1Y,2Y,3Y,5Y,10Y,12Y,15Y,20Y</Expiries>",
            "      <Terms>1Y,2Y,3Y,4Y,5Y,7Y,10Y,15Y,20Y,30Y</Terms>",
            "    </SwaptionVolatilities>",
            "    <FxVolatilities>",
            "      <Simulate>false</Simulate>",
            "      <ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay>",
            "      <CurrencyPairs>",
            pairs,
            "      </CurrencyPairs>",
            "      <Expiries>1Y,2Y,5Y</Expiries>",
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
        lines.extend(
            [
                "    <Counterparty>",
                f"      <CounterpartyId>{cid}</CounterpartyId>",
                "      <CreditQuality>IG</CreditQuality>",
                "      <BaCvaRiskWeight>0.05</BaCvaRiskWeight>",
                "      <SaCcrRiskWeight>1</SaCcrRiskWeight>",
                "      <SaCvaRiskBucket>2</SaCvaRiskBucket>",
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
        except Exception:
            pass

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
        except Exception:
            pass

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
    if xva.dim_quantile is not None:
        _maybe_set(input_parameters, "setDimQuantile", float(xva.dim_quantile))
    if xva.dim_horizon_calendar_days is not None:
        _maybe_set(input_parameters, "setDimHorizonCalendarDays", int(xva.dim_horizon_calendar_days))
    if xva.dim_regression_order is not None:
        _maybe_set(input_parameters, "setDimRegressionOrder", int(xva.dim_regression_order))
    if xva.dim_regressors is not None:
        _maybe_set(input_parameters, "setDimRegressors", str(xva.dim_regressors))
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
