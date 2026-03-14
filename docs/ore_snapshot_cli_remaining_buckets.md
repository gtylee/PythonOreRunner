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

## Exposure Decomposition Follow-up

Using the current fixed code path with `500` Python paths (`seed=42`,
`rng=ore_parity`), the residual mismatch starts immediately after `t=0` and is
already visible in the netting-set exposure profile.

Representative points for `ore_measure_ba.xml`:
- `t=0.084699`: ORE `EPE=96,930`, Python `EPE=148,852`
- `t=0.163934`: ORE `EPE=130,896`, Python `EPE=202,395`
- `t=0.333333`: ORE `EPE=182,673`, Python `EPE=285,700`
- `t=0.584699`: ORE `EPE=292,443`, Python `EPE=404,308`

Representative points for `Example_1/ore.xml`:
- `t=0.248634`: ORE `EPE=162,395`, Python `EPE=258,670`
- `t=0.497268`: ORE `EPE=223,660`, Python `EPE=352,932`
- `t=1.003002`: ORE `EPE=366,189`, Python `EPE=588,021`
- `t=1.750947`: ORE `EPE=435,362`, Python `EPE=672,731`

Leg-level decomposition on the Python side shows:
- mean fixed-leg PV stays large and positive, around `+3.27mm`
- mean floating-leg PV stays large and negative, around `-3.26mm`
- mean total PV stays small, so the issue is not a gross sign error
- the excess exposure comes from dispersion, not mean

Representative Python path statistics:
- `ore_measure_ba.xml` at `t=0.084699`:
  - total std about `365,629`
  - fixed std about `77,826`
  - float std about `287,822`
  - corr(fixed,float) about `0.999698`
- `ore_measure_ba.xml` at `t=0.754098`:
  - total std about `1,053,580`
  - fixed std about `232,531`
  - float std about `821,212`
  - corr(fixed,float) about `0.999099`
- `Example_1/ore.xml` at `t=1.003002`:
  - total std about `1,259,340`
  - fixed std about `278,559`
  - float std about `981,004`
  - corr(fixed,float) about `0.998974`

Interpretation:
- the floating leg dominates the pathwise variance
- fixed and floating legs remain highly correlated, but the residual variance of
  their offset is still too large versus ORE
- the remaining gap looks like a pathwise floating-leg dynamics issue, not a
  coupon-date reconstruction issue and not a top-level trade-sign bug

Best next inspection targets:
1. compare pathwise floating coupon / forward projection against ORE on early
   exposure dates
2. inspect whether already-fixed versus not-yet-fixed floating coupons are being
   valued with the same convention as ORE between fixing date and payment date
3. inspect whether the forwarding dynamics in `swap_npv_from_ore_legs_dual_curve()`
   are still too volatile relative to ORE for this single-curve `EUR-EURIBOR-6M`
   setup

Additional narrowing:
- `swap_source=trade` and `swap_source=flows` produce essentially the same EPE on
  these two cases, so the residual gap is not caused by flow-vs-portfolio leg
  source choice
- enabling `use_node_interpolation=True` only changes EPE by a few tens on values
  in the hundreds of thousands, so node interpolation is not the dominant lever
  on this pair of examples
- replacing the exact forward-bond projection with the older discount-to-forward
  basis-ratio transport approximation makes no difference on these cases because
  `discount_column`, `forward_column`, and `xva_discount_column` are all the same
  (`EUR-EURIBOR-6M`)
- splitting the floating leg into already-fixed and still-unfixed coupons shows
  the excess variance comes almost entirely from the unfixed coupons

Representative split for `ore_measure_ba.xml`:
- at `t=0.084699`:
  - fixed floating coupons: mean about `-100,657`, std about `11,835`
  - unfixed floating coupons: mean about `-3,160,786`, std about `277,986`
- at `t=0.584699`:
  - fixed floating coupons: mean about `-96,393`, std about `34,107`
  - unfixed floating coupons: mean about `-3,075,574`, std about `687,719`

Representative split for `Example_1/ore.xml`:
- at `t=0.248634`:
  - fixed floating coupons: mean about `-100,316`, std about `11,929`
  - unfixed floating coupons: mean about `-3,147,361`, std about `484,756`
- at `t=1.003002`:
  - fixed floating coupons: mean about `-96,399`, std about `37,012`
  - unfixed floating coupons: mean about `-3,024,980`, std about `953,156`

Interpretation:
- fixing lock itself is not the main remaining bug
- the dominant residual mismatch is the volatility of the projected unfixed
  floating coupons

## Model / Calibration Status

Additional checks on the underlying LGM state path show:
- the Python loader is now reading simulation metadata correctly
  - `ore_measure_ba.xml` loads with `measure = BA`, `seed = 42`,
    `samples = 1000`
  - `Example_1/ore.xml` loads with `measure = LGM`, `seed = 42`,
    `samples = 1000`
- the LGM parameters currently used on the Python side are the flat initial
  values from `simulation*.xml`
  - `alpha(t) = 0.01`
  - `kappa(t) = 0.03`
- these are not being silently replaced by internal defaults
- the state simulator is internally consistent with those parameters
  - for example on `ore_measure_ba`, simulated `var(x_t)` closely matches the
    analytic `zeta(t)` on checked exposure dates

However, the ORE logs show that the native runs did calibrate the model:
- `LGM Volatility calibrate = 1`
- `IrModelBuilder: calibration for LGM and qualifier EUR`

The shipped output folders do **not** include `calibration.xml`, and the available
calibration report files do not expose the calibrated LGM parameters in a form the
Python loader can reuse:
- `todaysmarketcalibration.csv` contains market / surface calibration output
- it does not contain explicit post-calibration EUR LGM `alpha` / `kappa` values
- there is no checked-in artifact here that is equivalent to the loader's expected
  `calibration.xml`

This means the current Python parity path is still using the *pre-calibration*
LGM parameters from `simulation.xml`, while ORE likely used calibrated parameters
when generating `rawcube.csv` / `xva.csv`.

Current best hypothesis:
- the remaining raw-cube variance gap is primarily due to Python using initial
  LGM parameters while ORE used calibrated ones

Practical next step:
1. rerun the native ORE cases with a persisted `calibration.xml`, or otherwise
   export the calibrated EUR LGM parameters
2. feed those calibrated parameters into the Python path and recompare
   `rawcube.csv` before making further formula changes

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
