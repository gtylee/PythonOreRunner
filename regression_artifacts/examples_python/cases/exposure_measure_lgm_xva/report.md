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
- py_basel_eepe: `248720.5350385722`
- py_basel_epe: `247214.70345924492`
- py_cva: `43663.237641230786`
- py_dva: `61613.78108976491`
- py_fba: `30216.586111728328`
- py_fca: `-4558.845657651169`
- py_fva: `25657.740454077164`

## Parity

- parity: `not run (price-only mode)`

## Input Validation

- input_links_valid: `False`
- issue: `some active mandatory curve-config quotes are missing on the asof date`
