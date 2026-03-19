# Python LGM vs ORE SWIG Parity Report

Date: 2026-03-08
Scope: `native_xva_interface` single-trade parity validation for 1x IRS (`EUR`, 10MM, 5Y, pay-fixed), common snapshot inputs, `num_paths=4000`.

## Summary
Current parity is strong for funding/own-side adjustments and acceptable for MVA proxy direction/magnitude. CVA remains the main residual gap.

Latest side-by-side (PY vs ORE):

- `CVA`: `92.140984` vs `17.843903` (high residual)
- `DVA`: `25366.895254` vs `26100.569597` (`2.81%` diff)
- `FVA`: `12755.866956` vs `12789.603030` (`0.26%` diff)
- `MVA`: `-306.519945` vs `-283.156151` (`8.25%` diff)

## Implemented Parity Fixes

1. IRS trade mapping parity improvements
- Generated full two-leg swap XML (fixed + floating + schedule) instead of incomplete fixed-only payload.
- Reused ORE-style leg extraction (`load_swap_legs_from_portfolio`) from generated portfolio XML.
- Added simulation node tenor extraction into IRS leg setup.

2. Market/curve extraction alignment
- Added parsing for real ORE market quote families used in this setup (`IR_SWAP/RATE`, `MM/RATE`), including composite tenors.
- Split discount vs forwarding curve construction (OIS-like vs index-tenor buckets).
- Added safer duplicate tenor handling.

3. XVA convention and decomposition alignment
- DVA sign aligned to ORE report convention (positive component).
- FVA decomposed into `FBA + FCA` with ORE-style orientation:
  - `FBA <- ENE * lending spread`
  - `FCA <- EPE * borrowing spread`
- Funding spreads sourced from market quotes when available:
  - `ZERO/YIELD_SPREAD/<ccy>/<fvaBorrowCurve>/...`
  - `ZERO/YIELD_SPREAD/<ccy>/<fvaLendCurve>/...`
- MVA proxy sign switched to ORE-consistent direction for this setup.

4. Own-credit handling
- Own-name hazard/recovery inputs wired from runtime name mapping (`dvaName` / `BANK`).
- Added own-hazard fallback path when own hazard curve is absent in compact market data.

## Residual Gap

CVA is still materially higher on Python side. Based on diagnostics, the residual is dominated by exposure profile shape (EPE path dynamics), not by reporting convention.

Implication:
- Funding/own-side stack can be treated as parity-ready for this prototype scope.
- CVA needs separate exposure-dynamics calibration/alignment work if tighter parity is required.

## Test Status

Executed:

```bash
python3 -m pytest Tools/PythonOreRunner/native_xva_interface/tests -q
```

Result:

- `26 passed, 1 skipped`

## Performance Note

Python LGM remains significantly faster than ORE SWIG for this single-trade run configuration, while now producing close DVA/FVA and directional MVA alignment.
