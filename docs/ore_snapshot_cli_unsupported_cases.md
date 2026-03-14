# ORE Snapshot CLI Unsupported Cases

This file tracks the product/model buckets that should still be excluded from
the "fixable setup/data" workstream.

Scope:
- Includes only cases that still fail in the current scan because the Python
  path cannot complete them and there is not enough vendored ORE reference data
  to recover via reference fallback.
- Excludes plain missing-output / missing-reference-data cases for otherwise
  supported products.

Current status after unsupported-product reference fallback:
- Most of the earlier `FloatingLegData/Index not found ...` family no longer
  needs to be excluded.
- Those cases now complete in ORE-reference mode when vendored `Output` or
  `ExpectedOutput` data exists.

## Current Exclude Bucket

### FlexiSwap family

These are mixed-product portfolios that include `FlexiSwap` trades and still do
not ship usable native/reference output files for the CLI to fall back to.

Count: `9`

Cases:
- `Examples/Legacy/Example_65/Input/ore_0.xml`
- `Examples/Legacy/Example_65/Input/ore_0_100.xml`
- `Examples/Legacy/Example_65/Input/ore_0_20.xml`
- `Examples/Legacy/Example_65/Input/ore_0_50.xml`
- `Examples/Legacy/Example_65/Input/ore_10.xml`
- `Examples/Legacy/Example_65/Input/ore_15.xml`
- `Examples/Legacy/Example_65/Input/ore_20.xml`
- `Examples/Legacy/Example_65/Input/ore_5.xml`
- `Examples/Legacy/Example_65/Input/ore_50.xml`

Current observed failure:
- `FileNotFoundError: ORE output file not found (run ORE first): .../Example_65/Output/npv.csv`

Why these remain excluded:
- the portfolio contains unsupported `FlexiSwap` trades
- there is no vendored `ExpectedOutput/` fallback for the case family
- the CLI therefore cannot recover via reference-only mode

## Policy

Keep this bucket separate from the remaining fixable cases.

These cases need one of:
1. explicit `FlexiSwap` support in the Python snapshot/pricing path
2. vendored native ORE `Output` or `ExpectedOutput` files for reference mode

Without one of those, they should stay excluded.
