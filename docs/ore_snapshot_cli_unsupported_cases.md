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

### Swaptions and Bermudan-style rates products

Representative cases:
- `Examples/Legacy/Example_13/Input/ore_B0.xml`
- `Examples/Legacy/Example_13/Input/ore_B1.xml`
- `Examples/Legacy/Example_13/Input/ore_B2.xml`
- `Examples/Legacy/Example_13/Input/ore_B3.xml`
- `Examples/Legacy/Example_3/Input/ore.xml`
- `Examples/Legacy/Example_4/Input/ore.xml`
- `Examples/AmericanMonteCarlo/Input/ore_scriptedberm.xml`
- `Examples/ORE-Python/Notebooks/Example_3/Input/ore_bermudans.xml`

What would need to be added:
1. Product parsing beyond swap legs in the snapshot loader.
2. Native Python valuation logic for swaption/Bermudan payoff and exercise
   treatment.
3. Exposure-time support for option-style states rather than just swap NPV
   reconstruction from IRS legs.

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

### Commodity / equity / inflation / credit / structured rate products

Representative cases:
- `Examples/Exposure/Input/ore_commodity.xml`
- `Examples/Exposure/Input/ore_equity.xml`
- `Examples/Exposure/Input/ore_inflation_dk.xml`
- `Examples/Exposure/Input/ore_inflation_jy.xml`
- `Examples/Exposure/Input/ore_credit.xml`
- `Examples/Exposure/Input/ore_fbc.xml`
- `Examples/Exposure/Input/ore_fra.xml`
- `Examples/Legacy/Example_23/Input/ore.xml`
- `Examples/Legacy/Example_24/Input/ore_wti.xml`
- `Examples/Legacy/Example_32/Input/ore.xml`
- `Examples/Legacy/Example_33/Input/ore.xml`

What would need to be added:
1. Product-specific portfolio loaders for these trade types.
2. Native pricing/exposure functions for each family.
3. Removal of the remaining swap-only assumptions in the current snapshot path,
   especially places that still expect `FloatingLegData/Index` to exist.

### HW2F and other non-LGM model families

Representative cases:
- `Examples/Exposure/Input/ore_hw2f.xml`
- `Examples/Exposure/Input/ore_swap_hw2f.xml`
- `Examples/Legacy/Example_38/Input/ore.xml`

What would need to be added:
1. Native Python model support for HW2F snapshots, or
2. an explicit model-adapter layer that can price/expose those cases without
   forcing them through the LGM-only path

The repo already handles these safely in fallback mode. The missing piece is
native model support, not error handling.

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
1. swaptions / Bermudan-style rates products
2. FX options / TARF / barriers
3. commodity / equity / inflation / credit structured products
4. HW2F native model support

That order is recommended because it removes the largest rate-adjacent fallback
families first while staying closer to the existing rates/LGM infrastructure.
