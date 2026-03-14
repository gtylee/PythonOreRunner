# ORE Snapshot CLI Final Status

This note records the current end state for `py_ore_tools.ore_snapshot_cli`
after the bond-support routing fix, the non-LGM calibration matching hardening,
and the final default-path-count fix.

## Final Scan

Source:
- `/tmp/ore_snapshot_cli_scan_results_post_paths_fix.json`

Totals:
- Total scanned: `347`
- `completed_zero`: `347`
- `completed_nonzero`: `0`
- `error`: `0`
- `timeout`: `0`
- `pass_all=true`: `347`

## What Closed The Last Failure

The final remaining nonzero case had been:
- `Examples/Exposure/Input/ore_measure_ba.xml`

Root cause:
- the CLI parser defaulted `--paths` to `500`
- that meant a plain default XVA run used `500` Python paths even when the ORE
  case itself had `Samples=1000`
- the resulting `sample_count_mismatch` was enough to leave `xva_cva=false` on
  the BA example

Fix:
- `--paths` now defaults to `None`
- when the user does not pass `--paths`, `_compute_snapshot_case()` now uses the
  sample count embedded in the ORE case (`snap.n_samples`)
- explicit `--paths` still overrides this, so benchmarking and stress runs are
  unchanged

Observed effect on `ore_measure_ba.xml` default run:
- before:
  - `python_paths = 500`
  - `ore_samples = 1000`
  - `sample_count_mismatch = true`
  - `pass_all = false`
- after:
  - `python_paths = 1000`
  - `ore_samples = 1000`
  - `sample_count_mismatch = false`
  - `cva_rel_diff ≈ 0.0184`
  - `pass_all = true`

## Notes

- bond-family examples now default to Python price-only mode unless the user
  explicitly requests XVA/sensitivity modes
- non-LGM examples such as HW2F cases no longer crash calibration matching and
  complete via the intended reference fallback path
- there are no remaining hard-failure or parity-threshold buckets in the shipped
  `Examples/**/Input/ore*.xml` set

## Regression Coverage

Relevant tests:
- `tests/test_ore_snapshot_cli.py`
  - bond-family default dispatch
  - default XVA run inherits ORE sample count when `--paths` is omitted
- `tests/test_ore_snapshot_parity_report.py`
  - cloned-case calibration fallback still resolves calibrated LGM inputs
