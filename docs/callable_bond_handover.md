# Callable Bond Handover

## Scope

This note is only about `CallableBond` support in `ore_snapshot_cli`.

Current status:
- `CallableBond` is supported in Python price-only mode
- the intended parity target is native ORE deterministic `LGM` + `Grid`
- native ORE `CrossAssetModel` + `MC` for the exposure example is not the active parity target here because the local native run crashes in `McCamCallableBondBaseEngine`

Primary Python entrypoints:
- [bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py)
- [ore_snapshot_cli.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/ore_snapshot_cli.py)

## ORE C++ Route Being Mirrored

Main native references:
- [callablebond.cpp](/Users/gordonlee/Documents/Engine/OREData/ored/portfolio/callablebond.cpp)
- [builders/callablebond.cpp](/Users/gordonlee/Documents/Engine/OREData/ored/portfolio/builders/callablebond.cpp)
- [numericlgmcallablebondengine.cpp](/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/numericlgmcallablebondengine.cpp)
- [fdcallablebondevents.cpp](/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/fdcallablebondevents.cpp)
- [effectivebonddiscountcurve.hpp](/Users/gordonlee/Documents/Engine/QuantExt/qle/termstructures/effectivebonddiscountcurve.hpp)
- [numericlgmmultilegoptionengine.cpp](/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp)

High-level native flow:
1. merge callable trade data with callable bond reference data
2. build underlying bond cashflows
3. build reference curve, income curve, credit curve, recovery, security spread
4. calibrate or instantiate LGM under `MarketContext::irCalibration`
5. price via `NumericLgmCallableBondEngine`

## What Is Implemented

### Parsing and reference data

Implemented in [bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py):
- `load_callable_bond_trade_spec(...)`
- `_merge_callable_reference_data(...)`
- `_parse_callability_data(...)`
- `_load_callable_engine_spec(...)`

Supported behavior:
- inline `CallableBondData`
- merge from `CallableBondReferenceData`
- call schedule
- put schedule
- `OnThisDate`
- `FromThisDateOn`
- clean vs dirty call/put prices
- include accrual flag
- ORE-style American expansion on the callable exercise grid

### Pricing route

Implemented in:
- [_price_callable_bond_lgm](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py)
- [_rollback_callable_bond_lgm_value](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py)

Current model path:
- deterministic 1D LGM only
- `Grid` and `FD` config both map into the same Python rollback framework
- current tested parity target is `Grid`

### Calibration route

Implemented in:
- [_build_callable_lgm_market_inputs](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py)
- [_try_calibrate_callable_lgm](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py)

Important current behavior:
- calibration reads callable product model params from [pricingengine_callablebond_lgm_grid.xml](/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/pricingengine_callablebond_lgm_grid.xml)
- calibration basket is coterminal ATM, matching the ORE builder shape
- calibration discount curve now follows ORE `irCalibration` market context more closely by resolving the `collateral_inccy` OIS/1D discount family
- Python calibration backend is QuantLib GSR-based, so ORE `VolatilityType=Hagan` is still approximated with a Hull-White/GSR-compatible calibration path

## What Changed In The Latest Passes

### 1. OIS-style calibration discount curve

Added:
- `_resolve_todaysmarket_discount_handle(...)`
- `_build_ql_discount_curve_for_callable_calibration(...)`

Why:
- native ORE callable builder calibrates under `MarketContext::irCalibration`
- for this example, that points to `collateral_inccy`
- `collateral_inccy` uses the EUR 1D/OIS discount family

Effect:
- deterministic callable parity improved materially versus the older generic per-currency fitted curve calibration input

### 2. More literal rollback state handling

Added:
- `_CallableCashflowState`
- `_build_callable_cashflow_states(...)`
- `_callable_coupon_ratio(...)`
- `_callable_cashflow_reduced_pv(...)`

Changed:
- `_rollback_callable_bond_lgm_value(...)`

Why:
- the previous shortcut recomputed the whole reduced underlying bond at exercise dates
- native ORE evolves `underlyingNpv`, `optionNpv`, and `provisionalNpv` separately
- native ORE also treats coupon and non-coupon cashflows differently via `CashflowInfo`

What now matches better:
- redemptions / simple cashflows stay in the underlying at `t = payDate`
- coupon-ratio exclusion only applies to coupon-like flows
- call exercise uses `call - (underlying + provisional)`
- put exercise uses `put - underlying + provisional`

Effect:
- `CallableBondCertainCall` improved again
- the mixed put/call case is still the weakest

## Native ORE Artifacts To Use

Deterministic parity target input:
- [ore_callable_bond_lgm_grid_npv_only.xml](/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/ore_callable_bond_lgm_grid_npv_only.xml)

Deterministic pricing-engine config:
- [pricingengine_callablebond_lgm_grid.xml](/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/pricingengine_callablebond_lgm_grid.xml)

Portfolio and reference data:
- [portfolio_callablebond.xml](/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/portfolio_callablebond.xml)
- [reference_data_callablebond.xml](/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/reference_data_callablebond.xml)

Native ORE outputs:
- [npv.csv](/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Output/callable_bond_lgm_grid_npv_only/npv.csv)
- [additional_results.csv](/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Output/callable_bond_lgm_grid_npv_additional/additional_results.csv)

Use `additional_results.csv` for:
- `strippedBondNpv`
- `callPutValue`
- event table diagnostics

## Current Deterministic Parity

Current stable numbers after the latest rollback pass:

| Trade | ORE NPV | Python NPV | Abs Diff |
|---|---:|---:|---:|
| `CallableBondTrade` | 107838072.031120 | 106265571.031745 | 1572501.00 |
| `CallableBondNoCall` | 111858499.784284 | 112146381.339940 | 287881.56 |
| `CallableBondCertainCall` | 61960289.618040 | 61982593.977652 | 22304.36 |
| `PutCallBondTrade` | 128191889.607085 | 126072478.908943 | 2119410.70 |

Interpretation:
- `CallableBondNoCall` is the cleanest stripped-underlying check
- `CallableBondCertainCall` is already quite close
- `PutCallBondTrade` is the clearest remaining engine-logic miss

## Known Good Tests

Primary callable regression:
- [test_bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/tests/test_bond_pricing.py)

Run:
```bash
python3 -m pytest /Users/gordonlee/Documents/PythonOreRunner/tests/test_bond_pricing.py -q -k callable
```

Current expected result:
- `6 passed, 14 deselected, 4 subtests passed`

CLI smoke coverage:
- [test_ore_snapshot_cli.py](/Users/gordonlee/Documents/PythonOreRunner/tests/test_ore_snapshot_cli.py)

## Important Experiments Already Tried

### Using exact trade `referenceCurveId` in live pricing

I prototyped explicit curve-id routing for `EUR-EURIBOR-3M` and related named curves.

Result:
- it made callable parity much worse

Conclusion:
- native ORE does care about the named reference curve
- but the current lightweight Python multi-curve bootstrap is not source-faithful enough to help yet
- do not switch live callable pricing to the explicit `referenceCurveId` route unless the multi-curve builder is improved first

### Using `curves.csv` exact discount columns

Also tried earlier for callable work.

Result:
- parity got worse

Conclusion:
- do not assume ORE `curves.csv` columns can be dropped directly into the callable rollback without reproducing the native engine’s exact curve semantics

## What Still Looks Missing

The remaining misses are now mostly engine-state issues, not XML parsing issues.

Highest-signal next targets:

1. Match `NumericLgmCallableBondEngine` state handling even more literally.
Current Python still simplifies:
- no cached `mustBeEstimated()` path
- no explicit future cashflow cache vector
- no exact `provisionalNpv` rollback gating except the trivial final-step case

2. Focus on `PutCallBondTrade`.
This is still the biggest miss and is the best case to debug:
- put-overrides-call precedence
- call then put update ordering on the same event index
- event-date interaction with underlying cashflows

3. Use native event-table additional results more aggressively.
The native engine emits an event table in `additional_results.csv`.
That should be compared line by line against:
- notional
- accrual
- bond flow
- call
- put
- effective discount logic

4. Only revisit named pricing curves after building a better multi-curve bootstrap.
The direction is correct, but the current prototype is not good enough.

## Practical Advice For The Next Agent

- Keep the current callable calibration discount-curve improvement. That was a real gain.
- Keep the more literal cashflow-state rollback. It improved source faithfulness and did not break tests.
- Do not chase more random market-input tweaks first.
- Debug `PutCallBondTrade` against [numericlgmcallablebondengine.cpp](/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/numericlgmcallablebondengine.cpp) section `9.2` to `9.4`.
- If you need a clean internal benchmark, use `CallableBondNoCall` for stripped value and `CallableBondCertainCall` for call exercise on a simpler path.

