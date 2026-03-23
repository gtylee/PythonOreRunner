# ORE Snapshot CLI Report

- ore_xml: `/Users/gordonlee/Documents/PythonOreRunner/Examples/Generated/USD_AllRatesProductsSnapshot/Input/ore.xml`
- trade_id: `IRS_USD_0001`
- counterparty: `CPTY_A`
- netting_set_id: `CPTY_A`
- modes: `price, xva, sensi`

## Pricing


## XVA


## Parity

- parity: `not run (price-only mode)`

## Sensitivity

- metric: `CVA`
- python_factor_count: `0`
- ore_factor_count: `0`
- matched_factor_count: `0`
- unmatched_ore_count: `0`
- unmatched_python_count: `0`
- unsupported_factor_count: `0`
- notes: `['sensitivity fallback: Unsupported by PythonLgmAdapter in native-only mode: CAP_USD_SOFR3M_0001:CapFloor, FLOOR_USD_SOFR3M_0001:CapFloor, SWAPTION_USD_0001:Swaption. These trades are supported only through the ORE SWIG fallback.']`

## Input Validation

- input_links_valid: `False`
- issue: `some active todaysmarket curve specs do not resolve to curve configurations`
- issue: `asof-date fixings are present but implyTodaysFixings is disabled`
