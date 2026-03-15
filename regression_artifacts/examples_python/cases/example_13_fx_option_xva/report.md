# ORE Snapshot CLI Report

- ore_xml: `$REPO_ROOT/Examples/Legacy/Example_13/Input/ore_E0.xml`
- trade_id: `FxOption`
- counterparty: `CPTY_A`
- netting_set_id: `CPTY_A`
- modes: `price, xva`

## Pricing

- discount_column: `EUR-EURIBOR-6M`
- forward_column: `EUR-EURIBOR-6M`
- py_t0_npv: `139479.4482852071`
- trade_type: `FxOption`

## XVA

- own_credit_source: `market`
- py_basel_eepe: `150395.50265475397`
- py_basel_epe: `144508.85901073684`
- py_cva: `6782.130835222395`
- py_dva: `2.3287885206009345e-15`
- py_fba: `0.0`
- py_fca: `0.0`
- py_fva: `0.0`

## Parity

- parity: `not run (price-only mode)`

## Input Validation

- input_links_valid: `False`
- issue: `some active mandatory curve-config quotes are missing on the asof date`
