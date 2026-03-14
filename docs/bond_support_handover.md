# Bond Support Handover

## Scope

`ore_snapshot_cli` now supports price-only handling for:

- `Bond`
- `ForwardBond`
- `CallableBond`

It does **not** support:

- bond XVA / exposure simulation

The implementation is a Python port of the relevant ORE / QuantExt pricing flow. It does **not** call ORE-SWIG and it is not a thin native bridge.

## Main entrypoints

Primary files:

- [py_ore_tools/bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py)
- [py_ore_tools/ore_snapshot_cli.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/ore_snapshot_cli.py)
- [py_ore_tools/ore_snapshot.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/ore_snapshot.py)

Main public entrypoint used by the CLI:

- `price_bond_trade(...)` in [py_ore_tools/bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py)

## What the code does

### 1. Trade parsing

`load_bond_trade_spec(...)` parses a portfolio trade into `BondTradeSpec`.

It handles:

- `Bond`
- `ForwardBond`
- inline `BondData`
- reference-data merge via `SecurityId`
- bond settlement fields
- forward settlement fields
- premium / compensation payment fields

Reference data merge is intentionally minimal and follows the ORE build flow closely enough for snapshot parity.

### 2. Cashflow source

Preferred source is ORE `flows.csv` when available.

That is important because:

- it gives parity with ORE cashflow generation
- it avoids reconstructing every coupon rule locally
- it handles floating / amortizing cases more robustly

Fallback is local reconstruction from `LegData`.

### 3. Bond pricing

The risky bond pricing logic is in `_bond_npv(...)`.

This is the core ORE/QuantExt port:

- discount future cashflows
- weight by survival probability
- add expected recovery on coupon accrual intervals
- special-case zero-coupon recovery stepping
- handle security spread either:
  - on the discount / income curve, or
  - as extra credit spread

### 4. Forward bond pricing

`ForwardBond` pricing in `price_bond_trade(...)` follows the same structure as QuantExt:

- price underlying risky bond forward at forward maturity
- compute strike with dirty vs clean settlement logic
- discount payoff to settlement
- apply long / short sign
- include premium cashflow when applicable

## ORE / C++ references

The implementation comments in [bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py) point at the relevant ORE C++ sources.

The main conceptual mappings are:

- `Bond` construction:
  - OREData `ored/portfolio/bond.cpp`
  - especially `Bond::build()`

- risky bond engine:
  - QuantExt `qle/pricingengines/discountingriskybondengine.cpp`

- `ForwardBond` parsing:
  - OREData `ored/portfolio/forwardbond.cpp`

- forward bond engine:
  - QuantExt `qle/pricingengines/discountingforwardbondengine.cpp`

Important implementation detail:

- the Python code mirrors ORE behavior, but it does not attempt a one-to-one class port
- instead it ports the pricing semantics into a more direct Python representation

## Snapshot CLI integration

`ore_snapshot_cli` no longer rejects `Bond` and `ForwardBond` as unsupported in price mode.

The CLI output now includes bond-specific diagnostics such as:

- `trade_type`
- `security_id`
- `credit_curve_id`
- `reference_curve_id`
- `income_curve_id`
- `security_spread`
- `recovery_rate`
- `spread_on_income_curve`
- `treat_security_spread_as_credit_spread`
- `settlement_dirty`

Existing swap behavior was left intact.

## Vectorized scenario pricing

Bond pricing now has a compiled-trade / scenario-grid API for fast multi-scenario evaluation.

Key dataclasses:

- `CompiledBondTrade`
- `BondScenarioGrid`

Key functions:

- `compile_bond_trade(...)`
- `build_bond_scenario_grid_numpy(...)`
- `price_bond_scenarios_numpy(...)`
- `price_bond_scenarios_torch(...)`
- `price_bond_single_numpy(...)`

Design intent:

- parse / compile once
- evaluate many scenarios with dense arrays
- make single-scenario pricing a degenerate `n_scenarios = 1` case

Current CLI pricing already routes through the single-scenario vectorized NumPy path.

## Parity-critical details

These details matter. If changed casually, parity will regress.

### 1. Anchor flow-derived curves at `t=0`

`_curve_from_flow_discounts(...)` must include `(0.0, 1.0)`.

Without that anchor:

- `curve(0)` inherits a future DF
- bond PVs inflate
- parity breaks badly

### 2. Filter already-occurred cashflows at compile time

`compile_bond_trade(...)` filters out flows before the asof date.

This is required because `_bond_npv(...)` skips occurred flows. If the compiled path keeps them, cases with mid-life valuation dates will diverge.

### 3. Carry ORE spread treatment into the vectorized grid builder

`build_bond_scenario_grid_numpy(...)` must reproduce the same effective curves / hazard treatment as `_bond_npv(...)`:

- spread on curve vs spread as credit spread
- spread on income curve
- conditional survival logic

### 4. Forward premium should only require a discount factor if it is future-dated

Do not require `premium_discount` just because a premium field exists. Only require it when the compensation payment is actually in the future.

### 5. MPS uses `float32`

`price_bond_scenarios_torch(..., device="mps")` uses `float32`, because MPS does not support `float64` here.

That means:

- MPS parity is still very good
- but not machine-epsilon exact like CPU double precision

## Test coverage

Main tests:

- [tests/test_bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/tests/test_bond_pricing.py)
- [tests/test_ore_snapshot_cli.py](/Users/gordonlee/Documents/PythonOreRunner/tests/test_ore_snapshot_cli.py)

Coverage includes:

- bond XML parsing
- flow loading
- curve fallback behavior
- risky bond unit tests
- parity tests against `Example_18`
- parity test against `Example_78`
- vectorized NumPy single-scenario equivalence
- NumPy vs Torch kernel parity
- CLI regression coverage

At the point of handover, the combined run passes:

```bash
python3 -m pytest /Users/gordonlee/Documents/PythonOreRunner/tests/test_bond_pricing.py /Users/gordonlee/Documents/PythonOreRunner/tests/test_ore_snapshot_cli.py -q
```

## Performance status

Benchmark entrypoint:

- [benchmark_bond_pricing_numpy_torch.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/benchmarks/benchmark_bond_pricing_numpy_torch.py)

What it benchmarks:

- scalar loop baseline
- vectorized NumPy
- vectorized Torch CPU
- vectorized Torch MPS when available

Observed behavior on this machine:

- vectorization gives very large wins over scalar loops
- Torch CPU beats NumPy at larger scenario counts
- MPS is slower than Torch CPU for this kernel
- MPS parity remains strong, but it runs in `float32`

Interpretation:

- this bond kernel is small and memory-light
- once Python overhead is removed, CPU SIMD is already extremely strong
- GPU/MPS overhead is not recovered by enough arithmetic intensity

## Current limitations

- no bond exposure / XVA path
- no portfolio-level batched multi-trade GPU kernel yet
- scenario-grid builder is still mostly a scalar-to-grid convenience path
- the benchmark uses synthetic scenario perturbations, not a full market scenario engine

## Hard-fought integration notes

These were easy to get wrong and worth preserving explicitly.

### 1. Default CLI routing matters more than ore.xml analytics for bond-family trades

Several bond-family examples ship active simulation / XVA analytics in `ore.xml`.
If the CLI follows those blindly, it routes the case into the swap/XVA snapshot
path and either falls back unnecessarily or fails for the wrong reason.

Current rule in `ore_snapshot_cli.py`:

- when the user does **not** explicitly pass `--price`, `--xva`, or `--sensi`
- and the first trade type is `Bond`, `ForwardBond`, or `CallableBond`
- default to Python `price` mode and do not force XVA mode

This is why examples like:

- `Examples/AmericanMonteCarlo/Input/ore_forwardbond.xml`
- `Examples/Legacy/Example_73/Input/ore.xml`
- `Examples/Exposure/Input/ore_callable_bond.xml`

now run cleanly out of the box.

### 2. Forward-bond reference ids may include `_FWDEXP_...` suffixes

`ForwardBond` examples can reference security ids like:

- `SECURITY_1_FWDEXP_20251220`

while the reference data is keyed by the base id:

- `SECURITY_1`

If reference-data lookup requires an exact id match, forward-bond examples will
look unsupported even though the data is present.

Current rule:

- if exact security-id lookup fails
- and the id contains `_FWDEXP_`
- retry with the base security id before that suffix

### 3. Reconstructed bond cashflows must be scaled by `BondNotional`

When bond cashflows come from `flows.csv`, notional is already embedded.
When they are reconstructed from `LegData` / reference data, coupon amounts are
often unit-notional unless explicitly scaled.

Failure mode:

- PVs collapse from sensible ORE-sized values to tiny numbers
- forward-bond cases look catastrophically wrong even though discounting logic is fine

Current rule:

- for `Bond` and `ForwardBond`
- if cashflows are locally reconstructed instead of read from `flows.csv`
- scale them by `BondNotional`

## Recommended next steps

If continuing bond work, the highest-value next steps are:

1. Add bond exposure / XVA support if needed, instead of widening price-only routing further.
2. Add a true multi-scenario market-data builder that samples scenario curves directly into `BondScenarioGrid`.
3. Batch multiple compiled trades together for portfolio-level Torch pricing.
4. Only pursue bond GPU work further if the target is portfolio batching, not isolated single-trade kernels.

## Fast orientation for the next agent

If you need to modify bond pricing safely, start here:

1. Read [bond_pricing.py](/Users/gordonlee/Documents/PythonOreRunner/py_ore_tools/bond_pricing.py), especially:
   - `load_bond_trade_spec`
   - `_bond_npv`
   - `compile_bond_trade`
   - `build_bond_scenario_grid_numpy`
   - `price_bond_scenarios_numpy`
   - `price_bond_trade`
2. Run:
   - `python3 -m pytest /Users/gordonlee/Documents/PythonOreRunner/tests/test_bond_pricing.py -q`
3. If parity changes, check:
   - occurred-flow filtering
   - curve anchoring
   - security spread treatment
   - forward settlement / accrued handling
