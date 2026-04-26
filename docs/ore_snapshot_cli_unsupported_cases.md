# ORE Snapshot CLI Unsupported Cases

This file tracks the remaining example families that still require additional
native Python product/model support if the goal is to eliminate
`unsupported_python_snapshot_fallback`.

Important distinction:
- these cases no longer fail outright in the shipped example sweep
- they pass today via ORE reference fallback
- this file describes what would need to be added to run them natively on the
  Python snapshot path instead of falling back

## Current Live Count

Source:
- `/tmp/ore_snapshot_cli_live_report_after_reclassify/live_summary.json`

Current bucket:
- `unsupported_python_snapshot_fallback`: `68`

Important distinction:
- this file is only about cases that pass today because the CLI falls back away
  from native Python product/model handling
- it is not about `expected_output_fallback_pass`
- `expected_output_fallback_pass` means the case already works via explicit
  `ExpectedOutput` fallback and belongs in
  `docs/ore_snapshot_cli_remaining_buckets.md`, not here

## Product / Model Families Still Missing

### Rates first: remaining non-vanilla and model gaps

The current runtime already has native branches for vanilla IRS, generic
fixed/floating rate swaps, basis swaps, CMS/CMS spread/digital CMS spread swaps,
generic cap/floor trades, deterministic cashflows, and European/Bermudan
swaptions. Those families should not be described as globally missing.

Representative remaining rates-adjacent cases:
- `Examples/Exposure/Input/ore_hw2f.xml`
- `Examples/Exposure/Input/ore_swap_hw2f.xml`
- `Examples/Legacy/Example_38/Input/ore.xml`
- `Examples/AmericanMonteCarlo/Input/ore_scriptedberm.xml`

What still needs to be added or proved:
1. Native rate-future product parsing and pricing, wired into the snapshot
   runtime rather than only curve-building helpers.
2. Full modeled cross-currency rate-swap parity for FX-reset notionals, exchanges,
   reset calendars, and spot/fixing provenance. The native runtime must not use
   ORE `flows.csv` as an input; flows remain reference artifacts for validation
   only. Principal exchange signs and FX forward-point reset notionals are now
   modeled natively, USD holiday/payment-lag dates now follow the xccy schedule,
   FX reset dates now use FX convention calendars, and QuantExt-style overnight
   fixing-day value/fixing ladders are covered, and reporting-currency conversion
   now backs today's FX spot out of spot-date quotes using short FX forwards. The
   remaining modeled cross-currency PV gap should next be checked against ORE's
   coupon projection / curve convention details.
3. Scripted or AMC Bermudan-style rate products that are not expressible through
   the current swaption loaders.
4. HW2F model support in the snapshot runtime, or a model-adapter route that can
   price/expose those rate cases without forcing them through the LGM-only path.
5. ORE reference-generation coverage for rates cases that currently pass only
   through `ExpectedOutput` or no-reference fallback buckets.

### FX products

Representative cases:
- `Examples/Exposure/Input/ore_fx.xml`
- `Examples/AmericanMonteCarlo/Input/ore_fxtarf.xml`
- `Examples/AmericanMonteCarlo/Input/ore_barrier.xml`
- `Examples/AmericanMonteCarlo/Input/ore_overlapping.xml`
- `Examples/Legacy/Example_55/Input/ore.xml`

What would need to be added:
1. Snapshot parsing for FX option / TARF / barrier trade structures.
2. Native Python exposure/pricing for those FX products.
3. For portfolio/XVA parity, shared FX simulation support rather than
   swap-centric single-curve reconstruction.

### Commodity / equity / inflation / credit / structured products

Representative cases:
- `Examples/Exposure/Input/ore_commodity.xml`
- `Examples/Exposure/Input/ore_equity.xml`
- `Examples/Exposure/Input/ore_inflation_dk.xml`
- `Examples/Exposure/Input/ore_inflation_jy.xml`
- `Examples/Exposure/Input/ore_credit.xml`
- `Examples/Exposure/Input/ore_fbc.xml`
- `Examples/Legacy/Example_23/Input/ore.xml`
- `Examples/Legacy/Example_24/Input/ore_wti.xml`
- `Examples/Legacy/Example_32/Input/ore.xml`
- `Examples/Legacy/Example_33/Input/ore.xml`

What would need to be added:
1. Product-specific portfolio loaders for these trade types.
2. Native pricing/exposure functions for each family.
3. Removal of the remaining swap-only assumptions in the current snapshot path,
   especially places that still expect `FloatingLegData/Index` to exist.

## What Not To Count As Product Work

The following are not product-support tasks and should stay out of this file:
- cases that already pass against `ExpectedOutput`
- cases in `expected_output_fallback_pass`, which are already working because
  the explicit fallback is in place
- cases that only need native `Output/` artifacts
- cases that only need a simulation analytic added to the example XML
- report bucketing / classification issues

Those belong in:
- `docs/ore_snapshot_cli_remaining_buckets.md`

## Practical Next Step If This Bucket Is Targeted

The cleanest order is:
1. rates parity cleanup:
   - futures runtime product support
   - modeled cross-currency rate-swap residual PV parity without `flows.csv`
   - scripted/AMC Bermudan-style rate products
   - HW2F snapshot/runtime model support
2. rates reference provenance:
   - generate or vendor native `Output/` for remaining rates cases that are
     currently fallback-only
3. XVA exposure parity on supported rates:
   - close profile-level EPE/ENE/PFE/CVA differences before expanding more
     product families
4. FX options / TARF / barriers
5. commodity / equity / inflation / credit structured products

This order keeps rates product parity ahead of XVA model calibration work, and
keeps both ahead of broader non-rates product expansion.
