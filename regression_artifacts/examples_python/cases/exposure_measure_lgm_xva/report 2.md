# ORE Snapshot CLI Report

- ore_xml: `$REPO_ROOT/Examples/Exposure/Input/ore_measure_lgm.xml`
- trade_id: `Swap_20`
- counterparty: `CPTY_A`
- netting_set_id: `CPTY_A`
- modes: `price, xva`

## Pricing

- discount_column: `EUR-EURIBOR-6M`
- forward_column: `EUR-EURIBOR-6M`
- leg_source: `flows`
- py_t0_npv: `602.3421823545359`

## XVA

- own_credit_source: `market`
- py_basel_eepe: `296094.6195723826`
- py_basel_epe: `295719.7253922913`
- py_cva: `63531.45186215086`
- py_dva: `47594.55668399515`
- py_fba: `23273.745914321218`
- py_fca: `-6611.003830611647`
- py_fva: `16662.742083709574`

## Parity

- parity: `not run (price-only mode)`

## Input Validation

- input_links_valid: `False`
- issue: `some active mandatory curve-config quotes are missing on the asof date`
