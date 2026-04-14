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
- py_t0_npv: `602.3422395223752`

## XVA

- own_credit_source: `market`
- py_basel_eepe: `52304.03041350203`
- py_basel_epe: `52300.48786141324`
- py_cva: `4598.2043143641`
- py_dva: `159.08205897150376`
- py_fba: `77.73232929484129`
- py_fca: `-469.24507253485115`
- py_fva: `-391.51274324000985`

## Parity

- parity: `not run (price-only mode)`

## Input Validation

- input_links_valid: `False`
- issue: `some active mandatory curve-config quotes are missing on the asof date`
