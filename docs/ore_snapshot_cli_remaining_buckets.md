# ORE Snapshot CLI Remaining Buckets

This note is for follow-up work on `py_ore_tools.ore_snapshot_cli`.

Scope:
- Excludes the unsupported product/model bucket already split out separately.
- Focuses only on the remaining large failure buckets after the recent path-resolution and empty-path fixes.

Current scan status:
- Total scanned: `343`
- `completed_zero`: `36`
- `completed_nonzero`: `2`
- `error`: `305`
- `timeout`: `0`

## What Was Already Fixed

Recent passes already addressed:
- forced `price=True` mode inference
- incorrect `Setup/inputPath` resolution
- price-only fallback when simulation analytic is absent
- blank `portfolioFile` / `fixingDataFile` crashing validation
- non-portfolio analytics such as SIMM causing `IsADirectoryError`

Do not reopen those unless a regression appears.

## Excluded From This Handoff

Do not tackle these in the next thread unless explicitly asked:
- unsupported product parsing
  - bond / callable bond / forward bond
  - FX forward / FX option / TARF / barrier
  - Bermudan / swaption variants
  - inflation / commodity / FRA / cap-floor
  - CMS spread / credit products
- unsupported model bucket
  - HW2F cases failing with `no LGM node found for ccy 'EUR' or 'default'`

Those have their own separate add-list.

## Remaining Big Buckets

### 1. Missing Native ORE Outputs

This is now the largest remaining non-product bucket. These cases fail because the local checkout does not contain the expected `Output/*.csv` artifacts.

Current status:
- Many earlier output-missing families were already recovered by `ExpectedOutput` fallback.
- The remaining cases in this bucket are the ones that still have no usable vendored reference data.

Typical remaining patterns:
- no `Output/` and no `ExpectedOutput/`
- or `ExpectedOutput/` exists but does not contain rows/files for the requested trade variant

Current high-signal groups:

#### Example_35 has no vendored reference outputs
- Count: `3`
- Cases:
  - `Examples/Legacy/Example_35/Input/ore_FlipView.xml`
  - `Examples/Legacy/Example_35/Input/ore_Normal.xml`
  - `Examples/Legacy/Example_35/Input/ore_ReversedNormal.xml`
- Notes:
  - these are plain swap/XVA examples
  - they do not ship `Output/` or `ExpectedOutput/`
  - `run.py` expects native ORE to generate subdirectories such as `NormalXVA/`

#### Example_36 has no vendored reference outputs
- Count: `3`
- Cases:
  - `Examples/Legacy/Example_36/Input/ore_ba.xml`
  - `Examples/Legacy/Example_36/Input/ore_fwd.xml`
  - `Examples/Legacy/Example_36/Input/ore_lgm.xml`
- Notes:
  - these are plain swap/XVA measure-comparison examples
  - they do not ship `Output/` or `ExpectedOutput/`
  - `run.py` expects native ORE to generate subdirectories such as `measure_lgm/`

#### Example_33 swap variants have incomplete vendored reference data
- Count: `3`
- Cases:
  - `Examples/Legacy/Example_33/Input/ore_FlipView.xml`
  - `Examples/Legacy/Example_33/Input/ore_Normal.xml`
  - `Examples/Legacy/Example_33/Input/ore_ReversedNormal.xml`
- Notes:
  - `ExpectedOutput/` exists
  - but `npv.csv` only contains trade `CDS`
  - the swap variants need reference rows for trade `Swap_20`

Do not include `Example_65` here anymore:
- that family is a `FlexiSwap` support gap and now belongs in the unsupported-product bucket

Recommended next-thread action:
- Decide whether these should remain hard failures.
- If the goal is “runs in this checkout”, the only real fix is to vendor or generate the missing native `Output` artifacts.
- Do not fake success unless there is a clear reference-mode contract for missing outputs.

### 2. Broken Case-Local Input References

These are not unsupported products. They looked like missing shared inputs or case files that resolved to nowhere.

#### Example_33 status update
- The three swap variants were patched to use the correct shared input paths.
- They are no longer blocked on missing `curveconfig.xml` / `todaysmarket.xml` / `market_20160205.txt`.
- Remaining issue:
  - vendored `ExpectedOutput/npv.csv` only contains trade `CDS`
  - the swap variants need reference data for trade `Swap_20`
- Cases:
  - `Examples/Legacy/Example_33/Input/ore_FlipView.xml`
  - `Examples/Legacy/Example_33/Input/ore_Normal.xml`
  - `Examples/Legacy/Example_33/Input/ore_ReversedNormal.xml`
- Treat these as missing reference-output cases now, not broken-input-reference cases.

#### Exposure historical-calibration cases missing expected shared inputs
- Status:
  - fixed by adding explicit shared `../../Input/...` setup paths
- Fixed cases:
  - `Examples/Exposure/Input/ore_hwhistoricalcalibration.xml`
  - `Examples/Exposure/Input/ore_hwhistoricalcalibration_pca.xml`

Recommended next-thread action:
- This bucket is effectively exhausted.
- Do not reopen unless a new broken-path case appears.

### 3. Invalid or Missing Market Configuration IDs

These fail because `Markets/simulation` or related market config ids do not exist in `todaysmarket.xml`.

Status:
- This bucket has been fixed in the example XMLs.
- Fixed cases:
  - `Examples/MinimalSetup/Input/ore.xml`
  - `Examples/ORE-Python/Notebooks/Example_2/Input/ore_dim.xml`
  - `Examples/ORE-Python/Notebooks/Example_2/Input/ore_threshold_break.xml`

Recommended next-thread action:
- Do not reopen unless a regression appears.

### 4. Portfolio Exists But Contains No Trade

These are not unsupported products; they are empty-or-placeholder portfolio cases.

Status:
- This bucket has been fixed in the CLI.
- Empty `<Portfolio/>` files now degrade to blank case identity for non-pricing analytics instead of hard-failing.
- Fixed cases:
  - `Examples/MarketRisk/Input/ore_zerotoparshift.xml`
  - `Examples/Legacy/Example_69/Input/ore.xml`

Recommended next-thread action:
- Do not reopen unless a regression appears.

## Suggested Next Order

If another thread picks this up, the highest-signal order is:

1. Revisit whether missing native outputs should stay hard-fail or get a separate workflow

## Notes

Do not mix this with the unsupported product/model bucket.

That bucket should remain separate so the next thread does not blur:
- example-data defects
- missing vendored outputs
- unsupported analytics/products
- unsupported model families
