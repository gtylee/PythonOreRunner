# Equity Support Handover

## Scope implemented

Current native example-scoped support covers:

- `EquityOption`
- `EquityForward`
- `EquitySwap` mapping and loader support only

Out of scope for now:

- `QuantoEquityOption`
- `EquityOptionAmerican`
- `EquityOptionEuropeanCS`
- barrier / Asian / cliquet / accumulator / TaRF / variance / basket-style equity exotics

## What was added

- Typed domain support for `EquityOption`, `EquityForward`, `EquitySwap`, and `EquityFactorConfig`
- Loader parsing for those products from ORE portfolio XML
- Mapper output for:
  - equity portfolio XML
  - `pricingengine.xml` equity sections
  - `simulation.xml` `<EquityModels><CrossAssetLGM .../></EquityModels>`
- Price-only workflow support in `src/pythonore/workflows/ore_snapshot_cli.py`

## Native pricing behavior

`EquityOption`

- TA001-style premium-surface cases use direct market premium lookup from `EQUITY_OPTION/PRICE/...`
- Legacy smile-vol cases use Black-style pricing from:
  - spot
  - equity forecasting curve
  - pricing discount curve
  - equity vol surface

`EquityForward`

- Native price-only valuation is supported from:
  - spot
  - equity forecasting curve
  - pricing discount curve

`EquitySwap`

- Trade loading and XML mapping are supported
- price-only native valuation is not implemented yet
- current CLI behavior is explicit reference fallback with `pricing_fallback_reason=missing_native_equity_swap_pricer`

## Important pricing detail

For legacy equity cases, the equity forecasting curve must stay separate from the pricing discount curve.

Example:

- `Example_22` uses `ForecastingCurve=USD1D` for the SP5 equity projection
- pricing discounting still comes from the pricing config (`USD-IN-EUR` under `xois_eur`)

This split materially improved parity for the legacy ATM-only example.

## How to use it

Price-only CLI runs:

```bash
PYTHONPATH=src python3 -m pythonore.apps.ore_snapshot_cli Examples/Academy/TA001_Equity_Option/Input/ore.xml --price
PYTHONPATH=src python3 -m pythonore.apps.ore_snapshot_cli Examples/Legacy/Example_22/Input/ore_atmOnly.xml --price
```

Example regression cases currently added:

- `ta001_equity_option_price`
- `example_22_equity_option_price`

Refresh those baselines with:

```bash
PYTHONPATH=src python3 -m pythonore.apps.examples_regression refresh --case ta001_equity_option_price --case example_22_equity_option_price
```

Compare regression baselines with:

```bash
PYTHONPATH=src python3 -m pythonore.apps.examples_regression compare
```

## Current parity level

`TA001_Equity_Option`

- `ore_t0_npv = 236.649138`
- `py_t0_npv = 236.7`
- abs diff `= 0.050862`

`Example_22 ore_atmOnly`

- `ore_t0_npv = 132179.29212`
- `py_t0_npv = 132103.238437`
- abs diff `= 76.053683`

## Tests passing

Focused regression / workflow suite:

```bash
pytest tests/test_snapshot_irs_rules.py tests/test_ore_snapshot_cli.py tests/test_examples_python_regression.py -q
```

Current result when last run:

- `85 passed`

Additional focused checks run during implementation:

- `pytest tests/test_ore_snapshot_cli.py -q`
- `pytest tests/test_examples_python_regression.py -q`

## Next obvious extension

If more equity native pricing is needed, the next step is `EquitySwap` price-only support. The XML and product plumbing are already in place, so the missing part is the native valuation path in `ore_snapshot_cli.py`.
