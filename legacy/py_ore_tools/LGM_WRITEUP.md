# LGM IRS/CVA Extraction and ORE Parity Write-Up

## Scope
This work extracts a clean LGM 1F implementation and builds a parity workflow against ORE for IRS exposure/CVA.

Primary focus:
- LGM-only (no cross-asset coupling)
- IRS valuation and exposure under LGM
- Unilateral CVA reconciliation to ORE outputs

## Core Model Implementation
Core model code is in [lgm.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/lgm.py).

Implemented capabilities:
- Hagan volatility + Hull-White reversion parameterization
- Deterministic functions:
  - `alpha(t)`, `kappa(t)`
  - `zeta(t)`
  - `H(t)`, `Hprime(t)`
  - `zetan(2,t)` support for BA numeraire terms
- Exact LGM-measure simulation
- BA-measure simulation support (auxiliary state)
- Pricing identities:
  - `discount_bond`
  - `numeraire_lgm`
  - `numeraire_ba`

## IRS/XVA Utilities
Reusable trade/curve/credit helpers are in [irs_xva_utils.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py).

Implemented capabilities:
- Swap leg loading:
  - from ORE trade XML (`trade` source)
  - from ORE flows export (`flows` source)
- Discount and forwarding curve loading from explicit ORE `curves.csv` time-value data
- Dual-curve IRS valuation (discounting vs forwarding)
- Fixing-day and calendar-aware schedule logic
- Optional pathwise fixed coupon handling
- ORE default curve + recovery ingestion for CVA side

## Reconciliation Harness
Main comparator is [compare_ore_python_lgm.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/compare_ore_python_lgm.py).

What it does:
- Loads ORE artifacts (`curves.csv`, `flows.csv`, `exposure_trade_*.csv`, `xva.csv`, and calibration output when present)
- Simulates LGM on ORE exposure dates
- Revalues IRS pathwise
- Computes `EE`, `EPE`, `ENE`, and unilateral CVA decomposition term-by-term
- Writes diagnostics and summaries into parity artifacts

Outputs per run:
- `*_diagnostics.csv`
- `*_cva_terms.csv`
- `*_summary.json`

## Convention Sweep Framework
Sweep script is [convention_sweep_lgm.py](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/convention_sweep_lgm.py).

Swept dimensions:
- `swap_source`: `trade` vs `flows`
- `forward_column`: e.g. `EUR-EURIBOR-6M` vs `EUR-EONIA`
- `pathwise_fixing_lock`: on/off
- `node_tenor_interp`: on/off
- `coupon_spread_calibration`: on/off
- `alpha_scale`: list values

Ranking metric:
- absolute `cva_rel_diff` vs ORE

Sweep artifacts:
- `/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/calibrated/sweep_full_results.csv`
- `/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/calibrated/sweep_full_results.json`

## Current Parity Findings
Best-performing convention set (calibrated scenario):
- `swap_source=trade`
- discount curve: `EUR-EONIA`
- forward curve: `EUR-EURIBOR-6M`
- node-tenor interpolation: ON
- coupon-spread calibration: OFF

Observed CVA parity:
- `alpha_scale=1.05` with 20k paths: ~`0.43%` CVA gap
- `alpha_scale=1.00` with 20k paths: ~`2.82%` single-seed gap
- 10-seed recheck at `alpha_scale=1.00`: mean gap ~`2.54%`, with expected MC dispersion

## Interpreting Alpha
Practical interpretation:
- `alpha` controls LGM variance level and therefore exposure dispersion and CVA sensitivity
- Higher `alpha` tends to widen exposure tails and increase CVA

Recommended usage in this workflow:
- Keep ORE calibrated `alpha` (`alpha_scale=1.0`) as model baseline
- Use `alpha_scale` as explicit sensitivity, not hidden retuning
- Report a small ladder (`0.95x / 1.00x / 1.05x`) if needed

## Baseline Configuration (Accepted)
Current accepted working baseline:
- scenario: `calibrated`
- `swap_source=trade`
- discount: `EUR-EONIA`
- forward: `EUR-EURIBOR-6M`
- node-tenor interpolation: ON
- coupon-spread calibration: OFF
- `anchor-t0-npv`: OFF
- paths: `20000`
- seed: fixed per run

This baseline is currently at an acceptable parity level for the LGM-only prototype.
