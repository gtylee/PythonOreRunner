# ORE Snapshot CLI Report

- ore_xml: `$REPO_ROOT/Examples/Legacy/Example_22/Input/ore_atmOnly.xml`
- trade_id: `EQ_CALL_SP5`
- counterparty: `CPTY_A`
- netting_set_id: `CPTY_A`
- modes: `price`

## Pricing

- currency: `USD`
- discount_factor: `0.973876465105328`
- equity_name: `SP5`
- forward0: `2090.1891193933984`
- long_short: `Long`
- option_type: `Call`
- pricing_mode: `python_equity_option_black`
- py_t0_npv: `132103.23843732217`
- quantity: `775.0`
- spot0: `2147.56`
- strike: `2147.56`
- trade_type: `EquityOption`
- volatility: `0.1697325510963265`

## XVA


## Parity

- parity: `not run (price-only mode)`

## Input Validation

- input_links_valid: `False`
- issue: `missing todaysmarket configurations: ['xois_eur']`
- issue: `some active mandatory curve-config quotes are missing on the asof date`
