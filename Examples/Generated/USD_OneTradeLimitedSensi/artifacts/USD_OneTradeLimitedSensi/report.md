# ORE Snapshot CLI Report

- ore_xml: `/Users/gordonlee/Documents/PythonOreRunner/Examples/Generated/USD_OneTradeLimitedSensi/Input/ore.xml`
- trade_id: `IRS_USD_0001`
- counterparty: `CPTY_A`
- netting_set_id: `CPTY_A`
- modes: `price, sensi`

## Pricing

- py_t0_npv: `1702209820.4057944`
- trade_type: `Swap`
- pricing_mode: `python_native_from_sensitivity`
- report_ccy: `USD`

## XVA


## Parity

- parity: `not run (price-only mode)`

## Sensitivity

- metric: `NPV`
- python_factor_count: `18`
- ore_factor_count: `0`
- matched_factor_count: `0`
- unmatched_ore_count: `0`
- unmatched_python_count: `18`
- unsupported_factor_count: `0`
- notes: `['Pruned 14 native sensitivity factors that are outside the portfolio currencies/index families.']`
- sensitivity_output_file: `sensitivity.csv`
- scenario_output_file: `scenario.csv`

## Input Validation

- input_links_valid: `True`
