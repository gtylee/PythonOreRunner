# ORE Snapshot CLI Unsupported Cases

This file tracks the cases that still do not run because they require additional
product or model support in the Python snapshot path.

Scope:
- Includes only current failures caused by unsupported product/model assumptions.
- Excludes missing native output files, bad input references, empty portfolio cases,
  and other setup/data issues.

Current unsupported buckets:

## Missing Float Index Assumption

These fail with:
- `ValueError: FloatingLegData/Index not found for trade '...' in portfolio XML`

This is the current signal that the loader is still assuming a swap-style floating leg.

### Bond / Credit Bond
- `Bond` (`5`)
  - `Examples/CreditRisk/Input/ore.xml`
  - `Examples/CreditRisk/Input/ore1.xml`
  - `Examples/CreditRisk/Input/ore1_PFE.xml`
  - `Examples/Legacy/Example_43/Input/ore.xml`
  - `Examples/Legacy/Example_43/Input/ore1.xml`
- `Bond_1` (`10`)
  - `Examples/CreditRisk/Input/ore100.xml`
  - `Examples/CreditRisk/Input/ore2.xml`
  - `Examples/CreditRisk/Input/ore3.xml`
  - `Examples/CreditRisk/Input/ore3_ts.xml`
  - `Examples/CreditRisk/Input/ore4.xml`
  - `Examples/Legacy/Example_43/Input/ore100.xml`
  - `Examples/Legacy/Example_43/Input/ore2.xml`
  - `Examples/Legacy/Example_43/Input/ore3.xml`
  - `Examples/Legacy/Example_43/Input/ore3_ts.xml`
  - `Examples/Legacy/Example_43/Input/ore4.xml`

### Bermudan / Bermudan-style
- `BermSwp` (`4`)
  - `Examples/AmericanMonteCarlo/Input/ore_scriptedberm.xml`
  - `Examples/Legacy/Example_54/Input/ore.xml`
  - `Examples/ORE-Python/Notebooks/Example_3/Input/ore_amc.xml`
  - `Examples/ORE-Python/Notebooks/Example_3/Input/ore_classic.xml`
- `BermSwp_01` (`1`)
  - `Examples/ORE-Python/Notebooks/Example_3/Input/ore_bermudans.xml`

### Swaptions
- `SwaptionPhysical` (`7`)
  - `Examples/Legacy/Example_13/Input/ore_B0.xml`
  - `Examples/Legacy/Example_13/Input/ore_B0b.xml`
  - `Examples/Legacy/Example_13/Input/ore_B1.xml`
  - `Examples/Legacy/Example_13/Input/ore_B1b.xml`
  - `Examples/Legacy/Example_13/Input/ore_B2.xml`
  - `Examples/Legacy/Example_13/Input/ore_B2b.xml`
  - `Examples/Legacy/Example_13/Input/ore_B3.xml`
- `SwaptionCash` (`2`)
  - `Examples/Legacy/Example_3/Input/ore.xml`
  - `Examples/Legacy/Example_4/Input/ore.xml`

### FX Forward / FX Option / TARF / Barrier
- `FXFWD_EURUSD_10Y` (`3`)
  - `Examples/Exposure/Input/ore_fx.xml`
  - `Examples/Legacy/Example_7/Input/ore.xml`
  - `Examples/ORE-API/Input/ore.xml`
- `FxOption` (`4`)
  - `Examples/Legacy/Example_13/Input/ore_E0.xml`
  - `Examples/Legacy/Example_13/Input/ore_E1.xml`
  - `Examples/Legacy/Example_13/Input/ore_E2.xml`
  - `Examples/Legacy/Example_13/Input/ore_E3.xml`
- `FX_TaRF` (`1`)
  - `Examples/ORE-Python/Notebooks/Example_5/Input/ore.xml`
- `SCRIPTED_FX_TARF` (`4`)
  - `Examples/AmericanMonteCarlo/Input/ore_fxtarf.xml`
  - `Examples/AmericanMonteCarlo/Input/ore_overlapping.xml`
  - `Examples/Legacy/Example_55/Input/ore.xml`
  - `Examples/Legacy/Example_60/Input/ore.xml`
- `generic_barrier_option_fx_raw_kiko` (`1`)
  - `Examples/AmericanMonteCarlo/Input/ore_barrier.xml`

### Bonds / Callable / Forward Bond
- `CallableBondTrade` (`1`)
  - `Examples/Exposure/Input/ore_callable_bond.xml`
- `FwdBond` (`2`)
  - `Examples/AmericanMonteCarlo/Input/ore_forwardbond.xml`
  - `Examples/Legacy/Example_73/Input/ore.xml`

### Commodity / Equity / Inflation / FRA / CapFloor / Credit
- `CommodityForward` (`2`)
  - `Examples/Exposure/Input/ore_commodity.xml`
  - `Examples/Legacy/Example_24/Input/ore_wti.xml`
- `EqCall_SP5` (`2`)
  - `Examples/Exposure/Input/ore_equity.xml`
  - `Examples/Legacy/Example_16/Input/ore.xml`
- `trade_01` (`3`)
  - `Examples/Exposure/Input/ore_inflation_dk.xml`
  - `Examples/Exposure/Input/ore_inflation_jy.xml`
  - `Examples/Legacy/Example_32/Input/ore.xml`
- `fra1` (`2`)
  - `Examples/Exposure/Input/ore_fra.xml`
  - `Examples/Legacy/Example_23/Input/ore.xml`
- `cap_01` (`2`)
  - `Examples/Exposure/Input/ore_capfloor.xml`
  - `Examples/Legacy/Example_6/Input/ore_portfolio_2.xml`
- `cap_02` (`1`)
  - `Examples/Legacy/Example_6/Input/ore_portfolio_3.xml`
- `CPI_Swap_1` (`1`)
  - `Examples/Legacy/Example_17/Input/ore.xml`
- `CDS` (`2`)
  - `Examples/Exposure/Input/ore_credit.xml`
  - `Examples/Legacy/Example_33/Input/ore.xml`
- `SWAP_EUR_CMSSpread_fbc` (`2`)
  - `Examples/Exposure/Input/ore_fbc.xml`
  - `Examples/Legacy/Example_64/Input/ore.xml`

## Missing LGM Model Support

These fail with:
- `ValueError: no LGM node found for ccy 'EUR' or 'default'`

This is a model-family support gap, not a data issue.

### HW2F family
- `missing_lgm_model:EUR` (`3`)
  - `Examples/Exposure/Input/ore_hw2f.xml`
  - `Examples/Exposure/Input/ore_swap_hw2f.xml`
  - `Examples/Legacy/Example_38/Input/ore.xml`

## Recommended Priority

If these are tackled later, a reasonable implementation order is:

1. FX forward / FX option / TARF / barrier
2. Swaptions / Bermudans
3. Bond / callable / forward bond
4. FRA / cap-floor / inflation / commodity / equity / CDS / CMS spread
5. HW2F model-family support or explicit reference-mode fallback for unsupported models

## Recommended Policy

Until explicit support is added, keep these buckets separate from setup/data fixes.

Do not hide them behind generic expected-output fallback unless the intended policy is:
- unsupported Python path
- but acceptable ORE-reference-only completion when native or expected outputs exist

That decision should be made explicitly.
