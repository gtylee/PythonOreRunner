# ORE Snapshot CLI Final Status

This note records the final example-scan state for `py_ore_tools.ore_snapshot_cli`
after the path-resolution, reference-fallback, partial-reference, and
sensitivity-loader fixes.

## Final Scan

Source:
- `/tmp/ore_snapshot_cli_scan_results_final.json`

Totals:
- Total scanned: `343`
- `completed_zero`: `341`
- `completed_nonzero`: `2`
- `error`: `0`
- `timeout`: `0`

## Remaining Nonzero Cases

These complete successfully but exit nonzero because they still fail parity
thresholds, not because the CLI crashes.

- `Examples/Exposure/Input/ore_measure_ba.xml`
- `Examples/Legacy/Example_1/Input/ore.xml`

Current observed pattern:
- only `xva_cva` fails in both cases
- pricing is now effectively aligned
- CVA relative difference remains large even when Python paths are increased to
  match ORE samples (`1000`)
- the remaining mismatch is now an XVA/exposure issue, not a trade rebuild issue

Concrete findings from the parity check:
- both cases are the same economic swap under different example directories
- the portfolio XML is clean:
  - floating leg index is `EUR-EURIBOR-6M`
  - floating spread is `0.000000`
  - floating day counter is `A360`
- the flow-leg rebuild issues that were found earlier are now fixed:
  - `float_index_accrual` from `flows.csv` now reuses exported accruals when the
    true index basis is absent from the report
  - discount-curve interpolation/extrapolation now works in log-discount space
    instead of flat discount-factor extrapolation
- those two fixes collapsed the pricing gap:
  - before: `py_t0_npv = 702.683207`, `ore_t0_npv = 602.485572`,
    absolute diff `100.197635`
  - after: `py_t0_npv = 602.342182`, `ore_t0_npv = 602.485572`,
    absolute diff `0.143390`
- the inferred-spread problem is also gone:
  - `float_spread_abs_max` is now about `1.53e-06`
  - the old "large inferred floating spreads" warning no longer appears
- but XVA is still materially high on the Python side:
  - `ore_measure_ba.xml` at `500` paths:
    - `ore_basel_eepe = 250361.65`
    - `py_basel_eepe = 368113.83`
    - `cva_rel_diff = 0.2692`
  - `Example_1/ore.xml` at `500` paths:
    - `ore_basel_eepe = 272801.00`
    - `py_basel_eepe = 432455.74`
    - `cva_rel_diff = 0.2825`
  - `Example_1/ore.xml` at `1000` paths:
    - `ore_basel_eepe = 272801.00`
    - `py_basel_eepe = 405371.49`
    - `cva_rel_diff = 0.3495`
- that means the remaining parity miss is not caused by:
  - sample-count mismatch
  - coupon/index accrual mismatch
  - bad tail-curve extrapolation
  - t0 pricing misalignment

Suggested parity follow-up:
1. compare Python `epe/eepe` term profiles against ORE exposure output for the
   same dates to locate where the overstatement begins
2. inspect ORE-style cube treatment in `_compute_snapshot_case()`, especially:
   - `deflate_lgm_npv_paths()`
   - `compute_realized_float_coupons()`
   - `swap_npv_from_ore_legs_dual_curve()` for in-period fixed coupons
3. verify whether the remaining gap is due to:
   - numeraire-deflated vs discount-weighted exposure aggregation
   - fixing-date coupon lock-in on the simulation grid
   - ORE cube valuation conventions after coupon fixing but before payment

## Status

There are no remaining hard-failure buckets in the shipped example set.

The previous buckets are now resolved:
- missing native output fallback
- incomplete `ExpectedOutput` trade mapping
- broken example XML/shared-input paths
- invalid market config ids
- empty portfolio / no-trade cases
- unsupported-product fallback to ORE reference mode
- sensitivity-loader input path resolution

## If Work Continues

The next optional work is not crash-fixing. It is:
1. reduce or explain the two remaining parity-threshold nonzero exits
2. clean up docs or code paths that are now legacy from the earlier failure buckets
