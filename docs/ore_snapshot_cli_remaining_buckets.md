# ORE Snapshot CLI Remaining Buckets

This note tracks the remaining non-clean buckets in the live parity report for
`py_ore_tools.ore_snapshot_cli`.

Important distinction:
- all `347` shipped `Examples/**/Input/ore*.xml` cases now complete
- this file is no longer about crash failures
- it is about what would need to be added to reduce fallback-driven buckets

## Current Live Report

Source:
- `/tmp/ore_snapshot_cli_live_report_after_reclassify/live_summary.json`

Current totals:
- Total scanned: `347`
- `clean_pass`: `70`
- `expected_output_fallback_pass`: `167`
- `no_reference_artifacts_pass`: `20`
- `price_only_reference_fallback`: `15`
- `unsupported_python_snapshot_fallback`: `68`

Important distinction:
- `clean_pass` means the case passes without needing the explicit `ExpectedOutput`
  fallback classification
- `expected_output_fallback_pass` means the case passes because the CLI
  explicitly fell back to vendored `ExpectedOutput`
- that bucket is working, but it is not the same as native case-local `Output/`
  parity

There are no remaining hard failures or parity-threshold failures in the shipped
example set.

## Bucket: `price_only_reference_fallback`

Count:
- `15`

Meaning:
- the case runs in price mode only
- the example still lacks a safe local Python pricing route
- this is now mostly cases with no `simulation.xml`, or cases that are not true
  vanilla IRS-style price-only targets
- it falls back to reference price-only mode and emits `ore_t0_npv` only

Representative cases:
- `Examples/Academy/FC003_Reporting_Currency/Input/ore.xml`
- `Examples/Academy/TA001_Equity_Option/Input/ore.xml`
- `Examples/Legacy/Example_19/Input/ore_flat.xml`
- `Examples/Legacy/Example_20/Input/ore.xml`
- `Examples/Legacy/Example_28/Input/ore_eur_base.xml`
- `Examples/Legacy/Example_51/Input/ore.xml`

What would need to be added to eliminate this bucket:
1. For the one real vanilla swap holdout, add a local curve-build fallback for
   price-only mode when `ExpectedOutput` has `npv.csv` and `flows.csv` but no
   `curves.csv`:
   - `Examples/Legacy/Example_51/Input/ore.xml`
2. Reclassify the rest out of this bucket, because they are not meaningful
   vanilla swap price-only targets:
   - equity options
   - swaptions
   - FX forwards
   - credit/structured products
   - non-vanilla swaps with exchanges or cross-currency structure
3. If native Python pricing is actually desired for those families, that work
   belongs in the unsupported-product backlog, not here.

What not to do:
- do not describe `expected_output_fallback_pass` as “native parity”
- do not keep broadening the synthetic vanilla swap path to cover non-vanilla
  products; those should move to unsupported/reference-only buckets instead

## Bucket: `unsupported_python_snapshot_fallback`

Count:
- `68`

Meaning:
- the case passes via ORE reference fallback
- the current Python snapshot path still does not support the product/model path
  needed to run it natively

Representative families:
- `Examples/Exposure/Input/ore_capfloor.xml`
- `Examples/Exposure/Input/ore_fx.xml`
- `Examples/Exposure/Input/ore_credit.xml`
- `Examples/Exposure/Input/ore_commodity.xml`
- `Examples/Exposure/Input/ore_hw2f.xml`
- `Examples/Legacy/Example_13/Input/ore_B*.xml`
- `Examples/AmericanMonteCarlo/Input/ore_barrier.xml`
- `Examples/AmericanMonteCarlo/Input/ore_scriptedberm.xml`

What would need to be added to eliminate this bucket:
1. Native Python product support in the snapshot/pricer path for the remaining
   unsupported trade families.
2. Native model support where the example is not LGM-based, especially the HW2F
   family.
3. Product-aware exposure/pricing loaders where the current swap-centric leg
   extraction expects `FloatingLegData/Index` and similar IRS-only structures.

Where the detailed product backlog now lives:
- `docs/ore_snapshot_cli_unsupported_cases.md`

## Bucket: `expected_output_fallback_pass`

Count:
- `167`

Meaning:
- the case already passes
- and it passes because the CLI explicitly fell back to vendored
  `ExpectedOutput`
- this is a working fallback bucket, not a failure bucket
- it is still distinct from native case-local `Output` parity, because the
  comparison baseline did not come from `Output/` in this checkout

Representative families:
- `Examples/CurveBuilding/Input/ore_centralbank_*.xml`
- `Examples/ExposureWithCollateral/Input/ore_*.xml`
- `Examples/MarketRisk/Input/ore_*.xml`
- `Examples/Performance/Input/ore_*.xml`
- `Examples/Legacy/Example_10/Input/ore_*.xml`
- `Examples/Legacy/Example_31/Input/ore_*.xml`

What would need to be added to eliminate this bucket:
1. Native `Output/` artifacts for those cases in the checkout, or
2. a reproducible case-generation step that runs ORE and materializes the same
   reports locally before parity comparison

Required outputs depend on the mode, but typically:
- `npv.csv`
- `curves.csv`
- `flows.csv`
- `xva.csv`
- trade/netting exposure CSVs where XVA compare is expected

This is not a Python-modeling bug bucket. It is a fallback/provenance bucket.

## Bucket: `no_reference_artifacts_pass`

Count:
- `20`

Meaning:
- the case completes, but there is no native `Output/` and no vendored
  `ExpectedOutput`
- the report keeps these separate because there is nothing parity-grade to
  compare against

Current members:
- `Examples/Legacy/Example_35/Input/ore_FlipView.xml`
- `Examples/Legacy/Example_35/Input/ore_Normal.xml`
- `Examples/Legacy/Example_35/Input/ore_ReversedNormal.xml`
- `Examples/Legacy/Example_36/Input/ore_ba.xml`
- `Examples/Legacy/Example_36/Input/ore_fwd.xml`
- `Examples/Legacy/Example_36/Input/ore_lgm.xml`
- `Examples/Legacy/Example_61/Input/ore.xml`
- `Examples/Legacy/Example_61/Input/ore_ad.xml`
- `Examples/Legacy/Example_61/Input/ore_cg.xml`
- `Examples/Legacy/Example_61/Input/ore_gpu.xml`
- `Examples/Legacy/Example_65/Input/ore_0.xml`
- `Examples/Legacy/Example_65/Input/ore_0_100.xml`
- `Examples/Legacy/Example_65/Input/ore_0_20.xml`
- `Examples/Legacy/Example_65/Input/ore_0_50.xml`
- `Examples/Legacy/Example_65/Input/ore_10.xml`
- `Examples/Legacy/Example_65/Input/ore_15.xml`
- `Examples/Legacy/Example_65/Input/ore_20.xml`
- `Examples/Legacy/Example_65/Input/ore_5.xml`
- `Examples/Legacy/Example_65/Input/ore_50.xml`
- `Examples/ORE-Python/Input/ore.xml`

What would need to be added to eliminate this bucket:
1. Generate and check in native `Output/` artifacts for these examples, or
2. vendor `ExpectedOutput/` artifacts for them, or
3. explicitly decide that these should remain “execution only, no parity
   baseline” examples

Important note:
- `Example_65` is still a product-support-sensitive family even though it now
  completes; if native Python parity is desired there, this may overlap with the
  unsupported-product backlog as well as the missing-artifact backlog

## Practical Priority

If continuing cleanup, the next honest order is:
1. `price_only_reference_fallback`
   - now mostly requires one plain-swap fallback improvement plus reclassification
     of non-vanilla products
2. `no_reference_artifacts_pass`
   - requires generating or vendoring reference artifacts
3. `expected_output_fallback_pass`
   - only if native-output provenance matters; these already pass via explicit
     fallback
4. `unsupported_python_snapshot_fallback`
   - product/model implementation work, not report cleanup
