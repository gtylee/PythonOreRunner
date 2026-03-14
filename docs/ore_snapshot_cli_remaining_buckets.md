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

Typical errors:
- `FileNotFoundError: ORE output file not found (run ORE first): .../curves.csv`
- `FileNotFoundError: [Errno 2] No such file or directory: .../Output/npv.csv`

Largest groups:

#### Example_13 output missing
- Count: `10`
- Error: missing `Examples/Legacy/Example_13/Output/curves.csv`
- Cases:
  - `Examples/Legacy/Example_13/Input/ore_A0.xml`
  - `Examples/Legacy/Example_13/Input/ore_A1.xml`
  - `Examples/Legacy/Example_13/Input/ore_A2.xml`
  - `Examples/Legacy/Example_13/Input/ore_A3.xml`
  - `Examples/Legacy/Example_13/Input/ore_A4.xml`
  - `Examples/Legacy/Example_13/Input/ore_A5.xml`
  - `Examples/Legacy/Example_13/Input/ore_C1.xml`
  - `Examples/Legacy/Example_13/Input/ore_C2.xml`
  - `Examples/Legacy/Example_13/Input/ore_D1.xml`
  - `Examples/Legacy/Example_13/Input/ore_D2.xml`

#### Example_65 output missing
- Count: `9`
- Error: missing `Examples/Legacy/Example_65/Output/npv.csv`
- Cases:
  - `Examples/Legacy/Example_65/Input/ore_0.xml`
  - `Examples/Legacy/Example_65/Input/ore_0_100.xml`
  - `Examples/Legacy/Example_65/Input/ore_0_20.xml`
  - `Examples/Legacy/Example_65/Input/ore_0_50.xml`
  - `Examples/Legacy/Example_65/Input/ore_10.xml`
  - `Examples/Legacy/Example_65/Input/ore_15.xml`
  - `Examples/Legacy/Example_65/Input/ore_20.xml`
  - `Examples/Legacy/Example_65/Input/ore_5.xml`
  - `Examples/Legacy/Example_65/Input/ore_50.xml`

#### Example_31 output missing
- Count: `8`
- Error: missing `Examples/Legacy/Example_31/Output/curves.csv`
- Cases:
  - `Examples/Legacy/Example_31/Input/ore.xml`
  - `Examples/Legacy/Example_31/Input/ore_ddv.xml`
  - `Examples/Legacy/Example_31/Input/ore_dim.xml`
  - `Examples/Legacy/Example_31/Input/ore_mpor.xml`
  - `Examples/Legacy/Example_31/Input/ore_mta.xml`
  - `Examples/Legacy/Example_31/Input/ore_threshold.xml`
  - `Examples/Legacy/Example_31/Input/ore_threshold_break.xml`
  - `Examples/Legacy/Example_31/Input/ore_threshold_dim.xml`

#### Example_10 output missing
- Count: `6`
- Error: missing `Examples/Legacy/Example_10/Output/curves.csv`
- Cases:
  - `Examples/Legacy/Example_10/Input/ore.xml`
  - `Examples/Legacy/Example_10/Input/ore_mpor.xml`
  - `Examples/Legacy/Example_10/Input/ore_mta.xml`
  - `Examples/Legacy/Example_10/Input/ore_threshold.xml`
  - `Examples/Legacy/Example_10/Input/ore_threshold_break.xml`
  - `Examples/Legacy/Example_10/Input/ore_threshold_dim.xml`

#### ORE-Python Notebook Example_2 output missing
- Count: `5`
- Error: missing `Examples/ORE-Python/Notebooks/Example_2/Output/curves.csv`
- Cases:
  - `Examples/ORE-Python/Notebooks/Example_2/Input/ore.xml`
  - `Examples/ORE-Python/Notebooks/Example_2/Input/ore_external_im.xml`
  - `Examples/ORE-Python/Notebooks/Example_2/Input/ore_mpor.xml`
  - `Examples/ORE-Python/Notebooks/Example_2/Input/ore_mta.xml`
  - `Examples/ORE-Python/Notebooks/Example_2/Input/ore_threshold.xml`

Other smaller output-missing groups:
- `Examples/Legacy/Example_36` (`3`)
- `Examples/Legacy/Example_35` (`3`)
- `Examples/Legacy/Example_28` (`2`)
- `Examples/Legacy/Example_22` (`2`)
- `Examples/Legacy/Example_2` (`2`)
- `Examples/Legacy/Example_19` (`2`)
- `Examples/Legacy/Example_17` (`2`)
- `Examples/Legacy/Example_12` (`2`)

Recommended next-thread action:
- Decide whether these should remain hard failures.
- If the goal is “runs in this checkout”, the only real fix is to vendor or generate the missing native `Output` artifacts.
- Do not fake success unless there is a clear reference-mode contract for missing outputs.

### 2. Broken Case-Local Input References

These are not unsupported products. They look like missing shared inputs or case files that still resolve to nowhere.

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
- Count: `2`
- Error:
  - `Required ORE input files not found: ['/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/curveconfig.xml', '/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/conventions.xml', '/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/todaysmarket.xml', '/Users/gordonlee/Documents/PythonOreRunner/Examples/Exposure/Input/market.txt']`
- Cases:
  - `Examples/Exposure/Input/ore_hwhistoricalcalibration.xml`
  - `Examples/Exposure/Input/ore_hwhistoricalcalibration_pca.xml`

Recommended next-thread action:
- Check whether these `ore.xml` files are using the wrong relative paths.
- Compare them against neighboring cases that do run.
- If the paths are wrong in the examples, patch the example XMLs, not the CLI.

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

Examples:
- Count: `1`
  - `Examples/MarketRisk/Input/ore_zerotoparshift.xml`
  - Error: `no Trade node found in .../MarketRisk/Input/ZeroToParShift/portfolio.xml`
- Count: `1`
  - `Examples/Legacy/Example_69/Input/ore.xml`
  - Error: `no Trade node found in .../Legacy/Example_69/Input/portfolio.xml`

Recommended next-thread action:
- Decide whether empty-portfolio cases should be allowed to complete as non-pricing summaries, similar to the SIMM/no-portfolio handling.
- If yes, extend the tolerant identity path beyond blank `portfolioFile` to empty trade lists.

## Suggested Next Order

If another thread picks this up, the highest-signal order is:

1. Broken case-local input references
2. Empty portfolio / no-trade cases
3. Only then revisit whether missing native outputs should stay hard-fail or get a separate workflow

## Notes

Do not mix this with the unsupported product/model bucket.

That bucket should remain separate so the next thread does not blur:
- example-data defects
- missing vendored outputs
- unsupported analytics/products
- unsupported model families
