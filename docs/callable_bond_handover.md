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

### 1b. GSR calibration was silently falling back

Fixed in:
- [lgm_calibration.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/lgm_calibration.py)

Root cause:
- the QuantLib `Gsr` model had been built with a hardcoded numeraire time of `60.0`
- the calibration discount curve only extended to about `50Y`
- QuantLib then threw `time (60) is past max curve time`
- `_try_calibrate_callable_lgm(...)` swallowed that exception and returned `None`
- callable pricing then fell back to the uncalibrated XML defaults

Current fix:
- `numeraire_time = max(maturity_times) + 2.0`

Why it matters:
- this was a real calibration disable, not a harmless numeric detail
- after the fix, the callable path actually uses the calibrated LGM instead of the flat fallback model

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

Current stable numbers after the callable option/native-curve split plus the
more literal cached rollback state handling:

| Trade | ORE NPV | Python NPV | Abs Diff |
|---|---:|---:|---:|
| `CallableBondTrade` | 107838072.031120 | 108223321.679891 | 385249.65 |
| `CallableBondNoCall` | 111858499.784284 | 112146381.339940 | 287881.56 |
| `CallableBondCertainCall` | 61960289.618040 | 62210523.118371 | 250233.50 |
| `PutCallBondTrade` | 128191889.607085 | 128450669.412103 | 258779.81 |

Interpretation:
- `CallableBondNoCall` is the cleanest stripped-underlying check
- `CallableBondCertainCall` is now also in the same sub-`300k` band
- `PutCallBondTrade` is no longer the outlier it used to be

## Native ORE Proof About `CallableBondNoCall`

One important conceptual point is now settled: native ORE itself does **not**
price `CallableBondNoCall` as the same object as the equivalent standalone
plain bond.

Using a temporary native ORE case with the commented plain bond trade enabled:

| Trade | ORE TradeType | Native ORE NPV |
|---|---|---:|
| `CallableBondNoCall` | `CallableBond` | 111858499.784284 |
| `UnderlyingBondTrade` | `Bond` | 114184634.212881 |

Difference:
- the standalone bond is higher by `2326134.428597`

Conclusion:
- the remaining Python `CallableBondNoCall` miss is **not** a plain-bond parity miss
- it is a callable-engine-underlying parity miss
- so comparing `CallableBondNoCall` against the standalone bond is the wrong target

## Known Good Tests

Primary callable regression:
- [test_bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/tests/test_bond_pricing.py)

Run:
```bash
python3 -m pytest /Users/gordonlee/Documents/PythonOreRunner/tests/test_bond_pricing.py -q -k callable
```

Current expected result:
- `10 passed, 70 deselected, 7 subtests passed` when running the callable slices across bond and CLI tests

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

The remaining misses are now much smaller and concentrated in the callable
engine's internal underlying semantics, not in XML parsing and not in basic
call/put wiring.

Highest-signal next targets if this ever needs to be pushed further:

1. Reproduce native callable-engine underlying valuation directly.
Current Python still mixes:
- callable rollback for the option layer
- standalone risky-bond logic for the stripped layer

2. Use native event-table additional results more aggressively.
The native engine emits an event table in `additional_results.csv`.
That should be compared line by line against:
- notional
- accrual
- bond flow
- call
- put
- effective discount logic

3. Only revisit named pricing curves after building a better multi-curve bootstrap.
The direction is correct, but the current prototype is not good enough.

## Practical Advice For The Next Agent

- Keep the current callable calibration discount-curve improvement. That was a real gain.
- Keep the current callable option/native-reference-curve split. That was the biggest parity gain in the whole callable path.
- Keep the more literal cashflow-state rollback. It improved source faithfulness and did not break tests.
- Do not chase more random market-input tweaks first.
- If you need a clean internal benchmark, use `CallableBondNoCall` for callable-engine underlying behavior and `CallableBondCertainCall` for call exercise on a simpler path.
- Do not assume `CallableBondNoCall` should equal the standalone `Bond`; native ORE proves that it does not.
