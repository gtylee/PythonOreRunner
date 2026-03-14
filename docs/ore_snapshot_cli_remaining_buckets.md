# ORE Snapshot CLI Remaining Buckets

This note tracks the remaining non-clean buckets in the live parity report for
`py_ore_tools.ore_snapshot_cli`.

Important distinction:
- all `347` shipped `Examples/**/Input/ore*.xml` cases now complete
- this file is no longer about crash failures
- it is about what would need to be added to reduce fallback-driven buckets

## Current Live Report

Source:
- `/tmp/ore_snapshot_cli_live_report_rebucketed3/live_summary.json`

Current totals:
- Total scanned: `347`
- `clean_pass`: `65`
- `expected_output_fallback_pass`: `164`
- `no_reference_artifacts_pass`: `20`
- `price_only_reference_fallback`: `49`
- `unsupported_python_snapshot_fallback`: `49`

There are no remaining hard failures or parity-threshold failures in the shipped
example set.

## Bucket: `price_only_reference_fallback`

Count:
- `49`

Meaning:
- the case runs in price mode only
- the example XML does not have an active simulation analytic
- the CLI therefore cannot produce Python-side compare/XVA output
- it falls back to reference price-only mode and emits `ore_t0_npv` only

Representative cases:
- `Examples/Academy/FC003_Reporting_Currency/Input/ore.xml`
- `Examples/Academy/TA001_Equity_Option/Input/ore.xml`
- `Examples/Academy/TA002_IR_Swap/Input/ore.xml`
- `Examples/Legacy/Example_2/Input/ore_payer_swaption.xml`
- `Examples/Legacy/Example_17/Input/ore_capfloor.xml`
- `Examples/Legacy/Example_63/Input/ore_parstressconversion.xml`

What would need to be added to eliminate this bucket:
1. Add a simulation analytic to the example XMLs that only request price-style
   reports today.
2. Add or point to a valid `simulationConfigFile` for those cases.
3. Ensure the case also has the minimum native outputs needed by the Python
   compare path:
   - `curves.csv`
   - `npv.csv`
   - `flows.csv` for swap-style cases where parity depends on ORE cashflows
4. Only do this for families where Python compare mode is actually meaningful.
   Some of these are intentionally static/reporting examples where price-only
   reference mode may already be the right end state.

What not to do:
- do not just reclassify this bucket away
- if the goal is true Python-side pricing parity here, the examples need a real
  simulation setup, not another fallback

## Bucket: `unsupported_python_snapshot_fallback`

Count:
- `49`

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
- `164`

Meaning:
- the case already passes
- but it is passing against vendored `ExpectedOutput` instead of native
  case-local `Output`
- the report treats this separately because the parity baseline exists, but the
  artifacts are not native-run outputs from this checkout

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

This is not a Python-modeling bug bucket. It is an artifact-provenance bucket.

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
   - requires real simulation setup additions to the examples
2. `no_reference_artifacts_pass`
   - requires generating or vendoring reference artifacts
3. `expected_output_fallback_pass`
   - only if native-output provenance matters
4. `unsupported_python_snapshot_fallback`
   - product/model implementation work, not report cleanup
