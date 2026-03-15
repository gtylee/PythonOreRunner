---
name: python-ore-runner
description: Use when working with ORE (Open Risk Engine): configuring AMC/AMC-CG runs, benchmarking, number matching/parity, performance optimization, build setup, SWIG/native XVA, or debugging ORE output. Invoke when the user asks about ORE configuration, XVA calculations, AMC-CG routing, or troubleshooting ORE runs.
---

# ORE Expert Context

This skill provides hard-won, non-obvious knowledge about ORE (Open Risk Engine) accumulated from hands-on build, benchmark, and parity experiments on this codebase. Apply this knowledge whenever working with ORE configs, code, or debugging.

---

## Pure Python HW 2F

### There is now a pure-Python HW 2F model kernel in `py_ore_tools`

The repo now contains a Python extraction of the core QuantExt Hull-White n-factor machinery for the 2-factor case:

- `Tools/PythonOreRunner/py_ore_tools/hw2f.py`
- `Tools/PythonOreRunner/py_ore_tools/hw2f_integration.py`

This is not just an `ore` process wrapper. The Python side now implements:

- piecewise-constant `sigma_x(t)` and `kappa(t)`
- the deterministic HW transforms `y(t)` and `g(t,T)`
- pathwise discount-bond pricing
- BA-measure Euler simulation for the HW state and auxiliary bank-account integral
- swap pricing from ORE-style leg arrays loaded from `portfolio.xml` / `curves.csv`

The formulas were extracted from the existing C++ implementation:

- `QuantExt/qle/models/hwpiecewiseparametrization.hpp`
- `QuantExt/qle/models/hwmodel.cpp`
- `QuantExt/qle/processes/irhwstateprocess.cpp`

### Current usage split: case orchestration vs model calc

There are now two distinct layers:

- `hw2f_ore_runner.py`
  - builds/runs an ORE case directory
  - useful for generating authoritative ORE outputs
- `hw2f.py` + `hw2f_integration.py`
  - pure-Python model calc
  - useful for pricing / exposure work without calling back into ORE's model engine

Do not confuse them:

- if the task is "run the ORE case", use the runner
- if the task is "extract the model calc into Python", use the HW 2F kernel + integration layer

### Current parity status of the pure-Python HW 2F path

On the current generated single-swap case, the Python t0 swap PV is reasonably close to ORE:

- example relative difference was about `-1.5%`

That means:

- the Python model calc is live and coherent
- it is good enough for iterative Python-side diagnostics and XVA experimentation
- it is not yet a claim of full ORE parity across products or all scenarios

Treat the current state as:

- pricing-capable
- exposure-capable
- not yet a complete production-equivalent ORE replica

### How it works with XVA today

The pure-Python HW 2F path can already feed the existing Python XVA formulas.

Current flow:

1. `simulate_hw2f_exposure_paths(...)` in `hw2f_integration.py`
   - produces `times`, `x_paths`, `npv_paths`
2. existing XVA helpers in `irs_xva_utils.py`
   - `aggregate_exposure_profile_from_npv_paths(...)`
   - `compute_xva_from_npv_paths(...)`
   - `compute_portfolio_xva_from_trade_paths(...)`

So the current HW 2F XVA stack is:

- pure-Python HW 2F state simulation
- pure-Python pathwise swap valuation
- existing Python exposure/XVA aggregation

### Current XVA limitations of the HW 2F Python path

What is implemented:

- single-trade swap exposure generation
- pathwise NPVs on a simulation grid
- reuse of existing Python CVA/DVA/FVA formulas

What is not yet fully packaged:

- one-call `run_hw2f_xva(...)` convenience entrypoint
- ORE-style numeraire-deflated cube output for HW 2F
- broad multi-trade / multi-netting-set HW 2F portfolio plumbing
- full counterparty/default/funding loader wiring inside the HW 2F integration layer
- full product coverage beyond the current swap path

So if the user asks "does HW 2F work with XVA?", the accurate answer is:

- yes, for Python-side pathwise exposure and XVA computation on the current swap workflow
- no, not yet as a complete drop-in replacement for all of ORE's XVA runtime features

### Relevant commands

Build/run an ORE HW 2F case:

```bash
python3 Tools/PythonOreRunner/run_hw2f_ore_case.py --case-dir /tmp/hw2f_case
```

Compare pure-Python HW 2F pricing against ORE output on the same case:

```bash
python3 Tools/PythonOreRunner/run_hw2f_python_case.py /tmp/hw2f_case
```

Relevant tests:

```bash
python3 -m pytest Tools/PythonOreRunner/tests/test_hw2f_ore_runner.py Tools/PythonOreRunner/tests/test_hw2f_python_model.py -q
```

---

## AMC-CG Configuration

### `amcTradeTypes` is a routing whitelist — not global enablement

Enabling AMC-CG globally is not enough. `amcTradeTypes` explicitly controls which trade types go on the AMC-CG path. Any trade type not listed falls back to the classic residual path, even if AMC-CG is globally on.

- **FxSwap decomposes**: In `portfolio_amccg.xml`, an `FXSwap` appears as `FXSwap_Near` and `FXSwap_Far` — both are `FxForward` type. If `FxForward` is absent from `amcTradeTypes`, both legs route to classic residual.
- **Always verify the actual trade types** in the portfolio XML against `amcTradeTypes` — not the business product name.
- A "mixed" run (some AMC-CG, some classic) can look nearly correct but destroys benchmark conclusions and performance comparisons.

### Two gates must both be satisfied for AMC-CG coverage

1. Trade type listed in `amcTradeTypes`
2. AMC-CG pricing-engine file has an explicit engine mapping for that trade type

For `CrossCurrencySwap` and similar, both must be present or the run fails/falls back.

### `Discretization=Euler` is required for `XvaEngineCG`

This is a hard compatibility requirement. A simulation config acceptable for other engines is not acceptable for the AMC-CG path. If AMC-CG fails unexpectedly, check simulation discretization early.

### Sample count knobs live in different input files

The "training" and "simulation" sample counts are in separate config files. A visible `Samples=10000` in one pricing-engine config does not mean the dominant benchmark workload uses that value. Always trace each sample-count parameter to the actual active engine/product path.

---

## Run Isolation and Benchmarking

### Use unique output directories per run

The cleanest benchmark isolation comes from using a fresh output directory per run. Shared output directories can produce fake "numerical" or "reporting" bugs — particularly truncated CSV rows like:
```
csv report is finalized with incomplete row, got data for 19 columns out of 27
```
This looks like a serialization bug but is actually overlapping processes writing to the same path.

### Benchmark hygiene checklist

Before comparing two ORE runs:

1. Run exactly one benchmark process at a time.
2. Use a unique output directory per run.
3. Record exact `amcTradeTypes` used.
4. Record whether residual classic trades were present.
5. Record sample count and seed.
6. Treat late CSV/report failures as a separate class from AMC-CG valuation failures.
7. Hold log verbosity constant between runs.

### For timing: direct `ore` invocation beats the Python wrapper

`Examples/AmericanMonteCarlo/run_benchmark.py` includes plotting and overhead. For clean timing of the ORE engine itself, use direct invocation:
```bash
../../build/App/ore Input/ore_amccg.xml
```
The Python wrapper is for convenience and report generation, not precision timing.

### "Fast" runs can be fake if the run failed early

A suspiciously low runtime before configuration is valid is not trustworthy. Confirm all three:
- The path-specific timing/progress lines appear in the log.
- The expected output artifacts were generated.
- The process exited cleanly.

### A successful AMC-CG timing banner ≠ the full job finished

The AMC-CG timing summary can print successfully before a later report failure. For automation, gate on final process exit status and final report checks — not just the presence of AMC-CG timing output.

### AMC-CG baseline on this machine (2000 samples, all-AMC-CG portfolio)

- AMC-CG cube generation: ~2840–2910 ms
- Forward evaluation (`fwdEval`): ~2610–2740 ms
- Wall clock: ~3.95–4.17 s

If a rerun is materially slower, suspect process overlap, file contention, or changed config before blaming stochastic instability.

---

## Number Matching and Parity

### For parity, prefer fresh ORE reruns over baked `Output/` folders

Several benchmark cases under `Tools/PythonOreRunner/parity_artifacts/...` ship with static `Output/` files. Those are useful for inspection, but they are not authoritative when testing:

- sticky MPOR overrides
- toggling `dva=Y/N`
- changes to native report generation
- changes to Python runtime semantics

Use a fresh isolated rerun whenever the comparison depends on runtime settings. The current side-by-side helper is:

```bash
PYTHONPATH=Tools/PythonOreRunner \
python3 Tools/PythonOreRunner/example_ore_snapshot_side_by_side.py \
  --case-dir Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_EUR_5Y_A \
  --paths 2000 \
  --rng ore_parity \
  --mpor-days 10 \
  --rerun-ore \
  --ore-dva
```

This script now:
- clones the case into a fresh run directory
- rewrites `ore.xml` / `simulation.xml`
- runs the local `build/App/ore`
- compares Python LGM against those fresh ORE outputs

### Cloned rerun cases must still resolve calibrated LGM params

Fresh native reruns in `/tmp` can look like a model/XVA mismatch when they are really a calibration provenance bug.

- If a cloned case tree loses `calibration.xml` in its local `Output/`, Python must still resolve the matching calibrated LGM term structure from another example run.
- Matching on absolute shared-input paths is not robust enough because cloned trees rewrite the root path. Match on stable example-relative resource ids instead.
- Symptom of the bad state:
  - repo case loads `alpha_source=calibration`
  - cloned rerun loads `alpha_source=simulation`
  - Python cube variance jumps materially and CVA/EEPE drift high
- Check this early on any fresh rerun parity failure before debugging bond formulas or XVA aggregation.

### Sticky MPOR changes exposure semantics at `t=0`

---

## Callable Bond Parity

### The active callable-bond parity target is native ORE deterministic `LGM + Grid`

For the callable-bond exposure example, do **not** treat the native `CrossAssetModel + MC` path as the primary parity target in this workspace.

Reason:
- the local native run crashes in `McCamCallableBondBaseEngine`
- the stable apples-to-apples target is native deterministic `LGM` + `Grid`

Use:
- `Examples/Exposure/Input/ore_callable_bond_lgm_grid_npv_only.xml`
- `Examples/Exposure/Input/pricingengine_callablebond_lgm_grid.xml`
- `Examples/Exposure/Output/callable_bond_lgm_grid_npv_only/npv.csv`
- `Examples/Exposure/Output/callable_bond_lgm_grid_npv_additional/additional_results.csv`

### The ORE C++ route being mirrored is specific

When debugging Python callable bonds, the relevant native path is:

1. `OREData/ored/portfolio/callablebond.cpp`
2. `OREData/ored/portfolio/builders/callablebond.cpp`
3. `QuantExt/qle/pricingengines/numericlgmcallablebondengine.cpp`
4. `QuantExt/qle/pricingengines/fdcallablebondevents.cpp`
5. `QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp`
6. `QuantExt/qle/termstructures/effectivebonddiscountcurve.hpp`

Important native builder semantics:
- `referenceCurve = market_->yieldCurve(referenceCurveId, pricing)`
- `incomeCurve = referenceCurve` unless explicitly overridden
- `defaultCurve = securitySpecificCreditCurve(...)->curve()`
- `recovery = market_->recoveryRate(securityId)` and falls back to `recoveryRate(creditCurveId)`
- `spread = market_->securitySpread(securityId)`

Important native engine semantics:
- the rollback discounts on `EffectiveBondDiscountCurve`
- that effective curve is:
  - `referenceCurve.discount(t)`
  - times `survivalProbability(t)^(1 - recoveryRate)`
  - times `exp(-securitySpread * t)` if spread exists
- this is **not** the same as the risky-bond helper formula `df * survival`

### The callable path has two valuation layers and they should not be treated identically

This was the hard-fought part.

There are two economically different layers:

1. stripped risky bond
2. callable option rollback

A very plausible but wrong instinct is:
- “native ORE uses `referenceCurveId = EUR-EURIBOR-3M`, so switch the whole Python callable pricer to that native curve”

That made parity **worse**.

What eventually worked:
- keep the **stripped risky bond** on the existing generic fitted EUR curve
- use the **native ORE reference curve from `curves.csv`** for the callable **option rollback only**

This is currently implemented in:
- `py_ore_tools/bond_pricing.py`

Specifically:
- `price_bond_trade(...)` keeps `base_curve = _fit_curve_for_currency(...)`
- `_price_callable_bond_lgm(...)` can receive a callable-specific reference curve
- `_load_callable_option_curve_from_reference_output(...)` looks for native `curves.csv`, including sibling output variants like `_curves`

### Why the split exists

Working backward from native ORE event diagnostics:

- native `additional_results.csv` reports `refDsc`
- for `PutCallBondTrade`, those `refDsc` values match the native `curves.csv` `EUR-EURIBOR-3M` column closely
- so native ORE **is** using the named reference curve in the callable engine

But:
- plugging that same native 3M curve into the Python stripped risky-bond layer overstates the stripped value badly
- the current generic fitted EUR curve was compensating for other mismatches in the stripped-bond approximation

The useful hybrid discovered in this repo is therefore:
- stripped layer: current generic fitted curve
- option layer: native callable reference curve if available

### This one change moved parity a lot

Before the callable option/native curve split, deterministic callable parity was roughly:

- `CallableBondTrade`: about `1.3m` to `1.6m` off depending on intermediate pass
- `PutCallBondTrade`: about `2.1m` off

After the split:

- `CallableBondTrade`: about `385k` off
- `CallableBondNoCall`: about `288k` off
- `CallableBondCertainCall`: about `250k` off
- `PutCallBondTrade`: about `259k` off

That is the current stable deterministic `LGM/Grid` parity level.

### Native ORE itself prices `CallableBondNoCall` differently from the plain bond

This was a crucial checkpoint.

A temporary native ORE run with the commented standalone bond trade enabled gave:

- `CallableBondNoCall`: `111,858,499.784284`
- `UnderlyingBondTrade` (`Bond`): `114,184,634.212881`

So native ORE itself has a gap of about `2.326m` between:
- the callable engine's no-call trade
- the equivalent standalone bond

Implication:
- the remaining Python `CallableBondNoCall` miss is **not** a standalone bond-pricer miss
- it is a callable-engine-underlying miss

Do **not** try to force Python `CallableBondNoCall` to equal the standalone bond.
That would be source-inconsistent with native ORE.

### The calibration path had a silent failure mode

There was a real bug in callable calibration:

- `lgm_calibration.py` built QuantLib `Gsr(..., 60.0)` with a hardcoded `60Y` numeraire time
- the discount curve only extended to about `50Y`
- QuantLib threw `time (60) is past max curve time`
- `_try_calibrate_callable_lgm(...)` swallowed the exception and returned `None`
- pricing silently fell back to flat uncalibrated XML defaults

The fix was:
- `numeraire_time = max(maturity_times) + 2.0`

Do not remove or “simplify” that. It is the difference between real calibration and silent fallback.

### `ReferenceCalibrationGrid` semantics matter

Another real source-faithfulness fix:

- ORE expands `ReferenceCalibrationGrid` via `DateGrid(...)`
- that means `TARGET` calendar + `Following` adjustment
- naive month addition is wrong

This is now mirrored in Python and covered by tests.

If callable calibration starts drifting again, check this before blaming the model.

### What did not fix parity, despite sounding plausible

These were all tried and should not be repeated blindly:

1. Switch the whole callable pricing path to `referenceCurveId`
- made parity materially worse

2. Use native `curves.csv` columns directly everywhere
- also made parity worse

3. Tune only the grid (`nx`, `ny`, `exerciseTimeStepsPerYear`)
- moved values, but did not close the large mixed put/call gap

4. Assume the remaining miss was mostly calibration
- not true after the calibration fixes
- native alpha/kappa substitution did not explain the remaining `PutCallBondTrade` miss

### The native logs are useful and should be read

For this example, native logs already tell you some things:

- security-specific recovery for `SECURITY_PUTCALL` is **not** found
- ORE falls back to the credit-curve recovery for `CPTY_A`

That means:
- a missing security-specific recovery is **not** the remaining parity issue here

### The callable engine is using `EffectiveBondDiscountCurve`, not the risky-bond helper formula

This is easy to miss if you only stare at the Python risky-bond implementation.

Native callable rollback uses:
- `EffectiveBondDiscountCurve(referenceCurve, creditCurve, spread, recovery)`

which implies:
- `effDsc = refDsc * survival^(1 - rr) * secSpreadDiscount`

Native event diagnostics in `additional_results.csv` expose:
- `refDsc`
- `survProb`
- `secSprdDsc`
- `effDsc`

Use those columns.
If Python callable discounting looks off, compare against `effDsc`, not just `refDsc`.

### The remaining hard part used to be the mixed put/call case

The historically hardest trade was:
- `PutCallBondTrade`

Why it was hard:
- pure call was already close
- no-call stripped value was much tighter
- the remaining miss was concentrated in mixed call/put interaction

The eventual large improvement did **not** come from another rollback rewrite.
It came from the option-layer native reference curve split described above.

Also, after the native no-call vs plain-bond check:
- the remaining conceptual gap is no longer “mixed put/call only”
- it is that Python still reuses the standalone risky-bond helper for the
  callable stripped layer, while native ORE reports the callable engine's own
  internal underlying value

So if parity worsens again specifically on `PutCallBondTrade`, check these in order:

1. did callable option rollback still pick up native `curves.csv`?
2. did the code fall back to generic curve because `_curves` output was not found?
3. did somebody accidentally change callable stripped bond to also use the native 3M curve?

### The current callable-specific reference curve loader has sibling fallback logic

Native `curves.csv` is not always written in the exact same callable output directory as `npv.csv`.

The repo now has:
- `_load_curve_from_reference_output(...)`
- `_load_callable_option_curve_from_reference_output(...)`

The second helper tries:
- direct output directory
- sibling variants such as `_curves` when running from `_npv_only` or `_npv_additional`

This is important for parity tests, because the deterministic callable run often has:
- `npv.csv` in `callable_bond_lgm_grid_npv_only`
- `curves.csv` in `callable_bond_lgm_grid_curves`

### Current callable parity tests are much tighter now

Relevant tests live in:
- `tests/test_bond_pricing.py`
- `tests/test_ore_snapshot_cli.py`

Important callable tests:
- native reference curve loader matches native `refDsc`
- sibling `_curves` discovery works
- deterministic callable NPV parity against native ORE `npv.csv`

Current deterministic tolerances are around:
- `CallableBondTrade`: `4e5`
- `CallableBondNoCall`: `5.1e5`
- `CallableBondCertainCall`: `3e5`
- `PutCallBondTrade`: `3e5`

These are much tighter than the earlier `1m+` / `2m+` bands.

### If you need the shortest trustworthy callable debug recipe

1. Use deterministic `LGM/Grid`, not CAM/MC.
2. Compare against native:
   - `npv.csv`
   - `additional_results.csv`
3. Confirm calibration is actually live.
4. Confirm `ReferenceCalibrationGrid` uses ORE `DateGrid` semantics.
5. Confirm callable option rollback resolves native `EUR-EURIBOR-3M` from `curves.csv`.
6. Confirm stripped risky bond is still on the generic fitted curve.
7. Only after that, touch rollback state logic.

### If parity regresses, suspect these first

1. Callable calibration silently fell back again.
2. Callable option curve could not find sibling `_curves` output and fell back to generic.
3. Someone unified callable stripped curve and option curve “for consistency”.
4. Day-count labels drifted between `A365F` and the curve loader’s accepted labels.

Do **not** start by retuning `nx` or rewriting put/call precedence unless the curve split and calibration path are already confirmed.

---

## Torch Backend Dispatch

### Use one CLI/runtime surface, not separate NumPy vs torch workflows

The current preferred pattern is:

- one user-facing CLI
- one case definition
- backend dispatch underneath

Do not present NumPy and torch as separate products unless the user explicitly asks for an internal comparison.

### Unified backend selector

The Python side now has a unified backend selector for torch-capable FX workflows:

- `tensor_backend=auto|numpy|torch-cpu|torch-mps`
- CLI flag: `--tensor-backend auto|numpy|torch-cpu|torch-mps`

Relevant code:

- `Tools/PythonOreRunner/py_ore_tools/lgm_fx_xva_utils.py`
  - `resolve_tensor_backend(...)`
  - `run_fx_forward_profile_xva_backend(...)`
  - `fx_forward_portfolio_npv_paths(...)`
- `Tools/PythonOreRunner/py_ore_tools/ore_snapshot_cli.py`

### Current `auto` policy is case-family aware

Do not assume `auto` always prefers torch.

Current behavior:

- single-trade FX forward profile/XVA:
  - `auto -> numpy`
  - reason: this workflow is too small/lightly batched to justify torch
- batched FX forward portfolio:
  - `auto -> torch-mps` if available
  - else `torch-cpu` if torch is available
  - else `numpy`

This is deliberate. The goal is one front door with capability/perf-aware dispatch, not forcing torch onto cases where it loses.

### Current torch-backed workflows worth using

These are the torch paths that have shown real value in this repo:

- single-ccy LGM simulation
- fused single-ccy swap/XVA benchmark path
- multi-ccy hybrid simulation
- batched FX forward portfolio path

These are not yet blanket guarantees for every ORE-style CLI case.

### Known failure modes

When debugging backend behavior, check these first:

- unsupported product/logic branches in the fused torch path
- MPS numerical drift from `float32`
- small-workload slowdown where launch/setup overhead dominates
- partial offload where data moves back to NumPy/Python too early
- calibration / QuantLib orchestration bottlenecks that torch cannot help

If the user wants “torch everywhere”, push back. The correct target is unified dispatch with explicit support boundaries.

### Useful CLI examples

Built-in examples now include:

- `--example lgm_torch`
- `--example lgm_torch_swap`
- `--example lgm_fx_hybrid`
- `--example lgm_fx_forward`
- `--example lgm_fx_portfolio`
- `--example lgm_fx_portfolio_256`

Example commands:

```bash
PYTHONPATH=Tools/PythonOreRunner python3 -m py_ore_tools.ore_snapshot_cli \
  --example lgm_fx_portfolio \
  --tensor-backend auto
```

```bash
PYTHONPATH=Tools/PythonOreRunner python3 -m py_ore_tools.ore_snapshot_cli \
  --example lgm_fx_portfolio_256 \
  --tensor-backend auto
```

### Current benchmark shape to remember

The batched multi-ccy FX forward portfolio is the strongest current torch example.

Representative result already observed:

- `10k` paths, `256` trades
  - `numpy/cpu`: about `1.55s`
  - `torch/cpu`: about `0.17s`
  - parity max abs: about `1.8e-07`

This is the kind of workload where torch is clearly worthwhile; the earlier single-trade FX profile example was not.

With sticky MPOR enabled, Python XVA uses closeout-date valuation along the same path. That means the first closeout exposure point is not the same object as valuation-date PV.

- **Valuation exposure** at `t=0` should line up with trade PV / ORE `exposure_nettingset_*.csv`
- **Sticky closeout exposure** at `t=0` is effectively the exposure at `t + MPOR` mapped back to the valuation observation point

If Python `EPE(t0)` looks much larger than PV after enabling MPOR, first check whether you are looking at the closeout profile rather than the valuation profile.

The current runtime now exposes both:
- `valuation_epe` / `valuation_ene`
- `closeout_epe` / `closeout_ene`

Do not collapse them into a single printed number in diagnostics.

### DVA mismatches can be a metric-enable mismatch, not a model bug

When an ORE case has `dva = N` in `ore.xml`, the loader will carry `analytics=("CVA",)` unless explicitly overridden. If a fresh native rerun forces `dva=Y` but the Python snapshot still runs with the original analytics tuple, Python DVA will appear to be zero even though the formula path works.

Before debugging DVA numerics, confirm all three:
- `ore.xml` or fresh rerun has `dva=Y`
- Python snapshot `config.analytics` includes `DVA`

---

## Python DIM / SIMM Port

### The native Python DIM port is feeder-driven

The Python-side DIM implementation under `Tools/PythonOreRunner/native_xva_interface/dim.py` does not generate its own AAD / AMC-CG sensitivities. It expects a prepared feeder payload in:

```python
snapshot.config.params["python.dim_feeder"]
```

Current supported Python DIM models are:

- `DynamicIM`
- `SimmAnalytic`
- `DeltaVaR`
- `DeltaGammaNormalVaR`
- `DeltaGammaVaR`

`Regression` is not yet ported to Python and should fail clearly rather than silently falling back.

### There are two feeder families

SIMM-style feeder:

- top-level keys:
  - `currencies`
  - `ir_delta_terms`
  - `ir_vega_terms`
  - `fx_vega_terms`
  - `simm_config`
  - `netting_sets`
- per-slice arrays:
  - `numeraire`: shape `(samples,)`
  - `ir_delta`: shape `(currencies, ir_delta_terms, samples)`
  - `ir_vega`: shape `(currencies, ir_vega_terms, samples)`
  - `fx_delta`: shape `(currencies - 1, samples)`
  - `fx_vega`: shape `(currencies - 1, fx_vega_terms, samples)`

Variance-style feeder:

- top-level keys:
  - `var_config`
  - `netting_sets`
- `var_config`:
  - `quantile`
  - optional `horizon_calendar_days`
- per-netting-set `current_slice`:
  - `covariance`
  - `delta`
  - `gamma`
  - optional `theta`
- per future slice:
  - `time`
  - `date`
  - `days_in_period`
  - `numeraire`
  - `covariance`
  - `delta`
  - `gamma`
  - optional `theta`
  - optional `flow`

### Low `dim_current` often means a toy feeder, not a DIM bug

If `dim_current` looks too small, first check whether the feeder itself is tiny.

For example, a variance feeder like:

```python
covariance = [[4.0, 1.0], [1.0, 9.0]]
delta = [2.0, -1.0]
gamma = 0
```

will naturally produce a small `DeltaVaR` DIM because:

- there are only 2 factors
- the delta magnitudes are small
- gamma is zero
- no `current_im` scaling is applied

In that case a low `dim_current` is expected. It does not mean the live trade notionals from the portfolio are being fed automatically into DIM. The feeder is the source of truth.

### Real standalone native runs need `PYTHONPATH=Tools/PythonOreRunner`

When running the native Python DIM / LGM path directly from the shell, use:

```bash
PYTHONPATH=Tools/PythonOreRunner python3 ...
```

Without that, imports can fail in `py_ore_tools` / `ore_snapshot.py` with errors like:

```text
attempted relative import with no known parent package
```

This is an environment issue, not a DIM calculation issue.

### Current report behavior

The Python DIM port attaches DIM outputs to the main `XVAResult`:

- `result.metadata["dim_mode"]`
- `result.metadata["dim_engine"]`
- `result.metadata["dim_current"]`
- `result.reports["dim_evolution"]`
- `result.reports["dim_distribution"]`
- `result.reports["dim_cube"]`
- optional `result.reports["dim_regression"]`
- `result.cubes["dim_cube"]`

This means native XVA and native DIM are already composable in one run as long as the feeder is provided.
- own-name credit data for `dvaName` exists in the market snapshot

### For exposure comparison, use netting-set exposure files, not trade row 0

For ORE-vs-Python exposure shape checks, compare against:

- `exposure_nettingset_<NETTING_SET>.csv`

not just the first row of `exposure_trade_*.csv`.

Trade-row `t0` is easy to print but mixes the wrong level of aggregation. Netting-set files are the correct counterpart to Python's netting-set exposure cube payload.

### Match Monte Carlo sample counts before calling a parity gap "real"

If Python is rerun at a much larger path count than the stored native ORE outputs, apparent CVA/DVA gaps can be mostly ORE-side MC noise.

This showed up clearly on long-dated convention-`B` USD/CAD swaps:
- Python `5000` vs stored ORE `500` still showed about `6%` CVA gaps
- fresh native ORE `5000` vs Python `5000` collapsed those same cases to about `0.5%` to `0.7%`

Rule of thumb:
- if pricing is already tight and exposure shape looks qualitatively similar, rerun native ORE at the same sample count before debugging model code
- do not treat `5000`-vs-`500` XVA deltas as proof of a structural parity bug

### For multiccy `NPV-only` parity, use flow amounts and the pricing curve, not ORE cashflow PVs

On the generated multicurrency IRS benchmark cases, especially CAD:

- do not use `flows.csv` `PresentValue` as the Python parity result
- do use `flows.csv` `Amount` discounted on each `PayDate`
- reconstruct discount factors from `curves.csv` on the exported curve-date grid, not only by interpolating year-fraction times

The important configuration detail is in `ore.xml`:

- if `Markets/pricing = libor`, ORE is pricing off the in-ccy swap / forwarding curve for this benchmark
- for CAD that means `CAD-CDOR-3M`, not `CAD-CORRA`

Practical implication:

- if the CAD t0 NPV is off by about `1%` while coupons already match the exported fixing values, the remaining miss is usually the discount curve choice / interpolation basis
- summing `Amount * reconstructed_df(PayDate)` on the correct pricing curve should collapse the gap to near-zero without reading ORE `PresentValue`

### For ORE template comparisons, align the trade dates to the template `asofDate`

When benchmarking Python against a file-driven ORE template such as `Example_9`, do not keep generating trades off a synthetic benchmark date like `2026-03-08` if the template market is on `2016-02-05`.

If you do:
- ORE will see supposedly "short-dated" FX forwards as roughly 10-year trades
- PV can still look superficially plausible
- XVA and exposure will be materially distorted in a way that looks like a model bug

Practical rule:
- when the benchmark uses a template-backed ORE case, generate the portfolio off the template `asofDate`
- treat date alignment as a first-order parity precondition, not cleanup

### For ORE `xva.csv`, use the aggregate row instead of summing trade rows

ORE `xva.csv` contains:
- one aggregate netting-set row with blank `TradeId`
- then per-trade rows

If you sum all rows, you double count badly and can conclude that ORE XVA is orders of magnitude larger than Python for no real reason.

Practical rule:
- parse the blank-`TradeId` aggregate row for top-level CVA / DVA / FBA / FCA
- use trade rows only for drill-down, not book totals

### Netting-set exposure must be aggregated pathwise before `max(.,0)`

A common Python-side bug is:

```text
sum_i E[max(V_i, 0)]
```

instead of the correct netting-set quantity:

```text
E[max(sum_i V_i, 0)]
```

On offsetting FX-forward books this can blow up Python EPE/CVA even when PV parity is already good.

Practical rule:
- aggregate trade NPVs pathwise to the netting-set cube first
- only then apply positive/negative exposure clipping
- if PV is close but EPE is wildly too large, check this before touching spreads or hazard curves

### Example_9 own-name credit can come from CDS spreads, not hazard quotes

In `Examples/ORE-Python/Notebooks/Example_9/Input/market_20160205.txt`:
- `CPTY_A` is provided via `HAZARD_RATE/...`
- `BANK` is provided via `CDS/CREDIT_SPREAD/...` plus recovery

If the Python loader only consumes `HAZARD_RATE/...`, own-name DVA / funding terms fall back to a conservative default hazard and appear too large.

Practical rule:
- when explicit hazard quotes are absent, convert CDS spread quotes to hazard using LGD
- if DVA/FBA remain too large after exposure parity is fixed, inspect whether own-name credit came from real quotes or a fallback

### Close the benchmark plumbing gap before blaming the XVA model

The large-FX benchmark work on this repo showed a repeatable order of operations:

1. align portfolio subset
2. align `asofDate`
3. align analytics flags (`cva/dva/fva/...`)
4. fix ORE aggregate parsing
5. fix Python pathwise netting aggregation
6. only then investigate real XVA methodology differences

If you skip those steps, Python-vs-ORE XVA gaps can look huge for purely mechanical reasons.

### On the parity path, prefer fresh ORE output curves over raw market-overlay reconstruction

For an ORE-backed Python run, there are two very different market-input regimes:

- `ore_output_curves`
  Python reads fresh `curves.csv` / calibration artifacts generated by ORE for that case
- `market_overlay`
  Python reconstructs curves from raw market quotes

If the goal is parity with a specific ORE case, `market_overlay` is only a fallback. It can be directionally sensible but still leave a visible PV gap even after spots, vols, and spreads are aligned.

Practical rule:
- for template/file-driven parity runs, execute the prepared ORE case first
- then load the Python snapshot from that same case so the native runtime can use `ore_output_curves`
- record the chosen path in metadata (`input_provenance["market"]`)

If Python PV is still off after the obvious plumbing fixes, check whether Python is still on `market_overlay` instead of `ore_output_curves`.

### FX forward `t=0` price parity should use the static discounted-forward formula

For file-driven `FxForward` price-only parity, do not start with the hybrid-model
forward pricer.

On `Example_28`, ORE's reported `npv.csv` value matched the simple static formula
exactly on the exported maturity-date curve row:

- `NPV = N * (S0 * P_for(T) - K * P_dom(T))`
- with `S0` from the ORE market quote
- and `P_for(T)`, `P_dom(T)` taken from the exported `curves.csv` row for the
  trade maturity date

The hybrid-model forward pricer was the wrong tool for this `t=0` price-only
check and introduced an avoidable gap. Practical rule:

- for `FxForward` `t=0` price parity, use the static discounted-forward formula
- prefer the exact maturity-date curve row when present instead of interpolating
  by `MaturityTime`
- keep the hybrid model for pathwise FX exposure / XVA work, not for this static
  price-only comparison

### `FX/RATE/...` is the spot format used in Example_9

Do not assume FX spot quotes arrive only as `FX/EUR/USD`.

`Example_9` uses:

```text
FX/RATE/EUR/USD
```

If the parser only accepts the shorter key shape, the parity path will silently miss FX spot data and drop onto synthetic defaults.

Practical rule:
- support both `FX/RATE/CCY1/CCY2` and `FX/CCY1/CCY2`
- when strict parity mode is on, missing FX spot quotes should fail fast rather than fall back

### `flows.csv` fixing dates matter for floating-leg exposure parity

`flows.csv` exports in this repo do carry floating fixing dates:
- field name is typically `fixingDate` (lowercase `f`), not `FixingDate`

If the Python loader ignores that field and falls back to `AccrualStartDate`, coupons with fixing lags get treated as fixing too late. That does not necessarily break `t=0` PV, but it can bias future exposure and therefore CVA/DVA.

Practical guidance:
- in `load_ore_legs_from_flows()`, prefer explicit fixing dates from `flows.csv`
- only fall back to accrual start when the fixing-date field is genuinely absent
- this is especially relevant on quarterly float legs with `FixingDays=2`

### Separate coupon accrual from index accrual on floating legs

A subtle but important parity trap:
- the coupon cash amount uses the leg accrual factor
- the floating forward rate must be annualized on the index accrual basis, not necessarily the coupon accrual basis

This matters on synthetic or stress-test conventions where the leg day counter differs from the index basis. In the multicurrency benchmark:
- convention `B` used `A365` coupon accrual with `USD-LIBOR-3M`
- `USD-LIBOR-3M` is still an `A360` index

If the forward projection divides by coupon accrual instead of index accrual:
- the loader will infer a fake persistent floating spread at `t=0`
- that synthetic spread can then leak into every future path
- the result is clean `t=0` PV parity but overstated EPE/CVA on long-dated swaps

Implementation guidance:
- carry a separate `float_index_accrual`
- use `float_index_accrual` for forward-rate annualization and spread calibration
- still use coupon accrual for the final cash amount

### `A360` support in helper time conversions is easy to miss

There are two different year-fraction helper paths in this codebase:
- `_year_fraction(...)`
- `_time_from_dates(...)`

It is not enough for `_year_fraction(...)` to support `A360` if `_time_from_dates(...)` silently falls back to Actual/Actual or `A365`. That can reintroduce the same annualization bug through fixing-time or accrual reconstruction code.

When debugging floating-leg exposure mismatches:
- confirm both helper paths understand `A360`
- check the actual inferred accruals on the offending case, not just the nominal convention labels

### Sticky MPOR verification workflow

Use the focused verification runner:

```bash
PYTHONPATH=Tools/PythonOreRunner \
python3 Tools/PythonOreRunner/run_sticky_mpor_verification.py
```

Current behavior:
- runs fresh Python-vs-ORE reruns on a small vanilla case pack
- forces sticky MPOR (`10D` by default)
- forces DVA on in the fresh ORE rerun
- writes JSON summary to:
  - `Tools/PythonOreRunner/parity_artifacts/sticky_mpor_verification_latest.json`

This is the quickest broader-scope check before claiming sticky-MPOR parity on the Python side.

### If notebook 5 suddenly shows a huge negative IRS PV, check flow-loaded fixed leg dates first

There is a specific regression pattern that looks like a catastrophic model failure but is actually a leg-loading bug:

- Python IRS PV jumps from a small ORE-like number to a large negative number
- for `flat_EUR_5Y_A`, the broken value was about `-940605.84` instead of about `823.56`
- valuation `t=0` exposure shows large ENE and near-zero EPE, as if only the floating leg were present

The root cause on this repo was in:
- `Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py`
- `load_ore_legs_from_flows()`

`flows.csv`-loaded swaps were missing:
- `fixed_start_time`
- `fixed_end_time`

The dual-curve pricer uses `fixed_start_time` to decide whether a fixed coupon is still alive:
- if those fields are missing, the fixed leg can be dropped entirely at `t=0`
- the resulting PV then looks like "just the floating leg", which is a large false negative number on receive-fixed swaps

The fix was to populate fixed accrual dates from `flows.csv`:
- `AccrualStartDate -> fixed_start_time`
- `AccrualEndDate -> fixed_end_time`

Guardrail:
- regression test lives in `native_xva_interface/tests/test_python_lgm_adapter.py`
- test name: `test_flow_loaded_fixed_leg_dates_preserve_t0_pv_parity`

Post-fix status:
- notebook 5 / side-by-side PV is back in line on `flat_EUR_5Y_A`
- example result is again about `Python 823.46` vs `ORE 823.56`
- if PV is broken but the curve loader and trade signs look sane, inspect leg completeness before touching model code

Do not confuse this with the remaining parity gaps:
- this fix restores the missing fixed leg and the `t=0` PV anchor
- CVA/DVA / exposure-shape mismatches can still remain after the PV is fixed

### Current sticky-MPOR verification status on this repo

On the current codebase, the focused fresh-rerun pack is in reasonable shape on selected vanilla cases:
- PV differences are tiny
- CVA is within low single-digit percent on most checked cases
- DVA is within roughly high-single-digit percent on the checked cases
- valuation-start exposure aligns closely with ORE

Do not overclaim from that result:
- this is not a full regression pack
- it is concentrated on vanilla parity artifacts
- it does not prove parity for unsupported products or all native paths

### Native AMC-CG is still a separate sign-off item

The representative native AMC-CG check is not clean enough for broad sign-off yet.

Useful facts from the current repo state:
- `Examples/AmericanMonteCarlo/Input/simulation_amccg.xml` already carries `CloseOutLag=2W` and `MporMode=StickyDate`
- classic native run on that example completes and writes full XVA outputs
- AMC-CG can enter `XvaEngineCG` graph build and start the AMC cube
- but parity against classic is not currently good enough to treat as signed off from this verification pass

So:
- Python sticky-MPOR parity on the checked vanilla cases: reasonably healthy
- native AMC-CG sticky-closeout parity: still requires dedicated investigation

### "Same benchmark" means more than same portfolio

To compare two ORE runs fairly, all of the following must be locked:
- Executable (binary state — reverted source ≠ clean binary without rebuild)
- Portfolio
- Simulation sample count
- `amcTradeTypes` (determines routing)
- Whether residual classic trades were present
- Output mode
- Log verbosity

### Matching ORE externally is a behavioral validation problem, not parameter-copying

Also verify:
- Which engine path actually ran
- Whether all trade types were covered by that path
- Whether simulation method was compatible
- Whether the requested outputs were actually produced
- Whether the run completed cleanly

### Operational failures produce misleading "numerical" symptoms

Common examples:
- Truncated `flows.csv` rows → concurrent process collision, not a model bug
- Valid `#N/A` placeholders in report columns (cap/floor vol fields for non-cap/floor trades) → expected, not data corruption

### Bermudan LGM parity: treat `Output/classic/calibration.xml` as the primary model source

For Bermudan swaption parity against an existing ORE classic run, the main lesson is:
- if `Output/classic/calibration.xml` exists, prefer it over rebuilding a trade-specific model in Python

Why this matters on this repo:
- the Python-side approximation of ORE's trade-specific `LgmBuilder` path can be materially worse than the actual calibrated model emitted by ORE
- on the Bermudan benchmark pack, using ORE's `calibration.xml` collapsed the large price gap, while the Python trade-specific GSR rebuild made parity worse

Concrete outcome from the benchmark cases:
- `berm_100bp`: Python backward moved to about `96382.62` vs ORE `96185.09`
- `berm_200bp`: Python backward moved to about `52294.67` vs ORE `52289.13`
- `berm_300bp`: Python backward moved to about `24337.23` vs ORE `24384.66`

Practical rule:
- when benchmarking Python backward Bermudan pricing against ORE classic outputs, use this precedence:
1. `Output/classic/calibration.xml`
2. only if absent, attempt a Python trade-specific rebuild from `pricingengine.xml`
3. only if that fails, fall back to `simulation.xml`

Do not assume that "more ORE-like builder logic in Python" is automatically more accurate than consuming the calibration ORE already produced.

### Bermudan backward parity depends on reduced-value rollback, not plain PV rollback

For the Python backward Bermudan pricer in `py_ore_tools/lgm_ir_options.py`:
- rollback should operate on reduced values, not raw PVs
- intrinsic exercise values should be converted from PV to reduced value by dividing by the LGM numeraire before `max(intrinsic, continuation)`
- the convolution rollback should mirror `LgmConvolutionSolver2` state-grid scaling (`sx/nx`, `sy/ny`) rather than using a generic Gauss-Hermite expectation step

Observed effect on this repo:
- once reduced-value convolution rollback was implemented, Python backward moved from materially above Python LSMC to near-equality with it
- after also preferring ORE `calibration.xml`, Python backward was effectively in classic ORE parity on the benchmark Bermudans

Interpretation rule:
- if `backward` and `lsmc` are far apart, the problem is likely in the backward solver
- if `backward` and `lsmc` are close but both differ from ORE, the problem is more likely in model source or intrinsic/cashflow treatment

### Do not reuse generic swap cashflow liveness rules for Bermudan intrinsic valuation

There is a specific and easy-to-reintroduce regression here:
- generic swap PV / exposure logic and Bermudan exercise intrinsic logic do **not** use the same coupon liveness rule

What is safe for generic swap valuation:
- keep fixed and floating coupons alive until payment date
- this is appropriate for swap exposure / remaining-swap PV views

What is required for Bermudan intrinsic valuation:
- exercise into whole periods
- a coupon must drop out once exercise is past the accrual start, not only after payment date

Why this matters:
- reverting Bermudan intrinsic to pay-date liveness crushed late-exercise cases on this repo
- the concrete regression was:
  - `single_last` fell from about `5896.18` back to about `1377.96`
  - `berm_200bp` fell from about `52294.67` back to about `49308.75`

Current safe pattern:
- `swap_npv_from_ore_legs_dual_curve()` in `py_ore_tools/irs_xva_utils.py` takes an explicit flag:
  - `exercise_into_whole_periods=False` for generic swap valuation / exposure
  - `exercise_into_whole_periods=True` for Bermudan intrinsic valuation
- `py_ore_tools/lgm_ir_options.py` Bermudan backward code must call the helper with `exercise_into_whole_periods=True`

Guardrail:
- if a Bermudan late-exercise benchmark collapses while generic swap PV tests remain fine, suspect coupon inclusion timing first
- especially inspect `single_last` in `parity_artifacts/bermudan_method_compare`, because it is the fastest canary for this bug
- Surprisingly fast run → likely failed early, not a genuine speedup

### Modern XVA sensitivities come from `xvaSensitivity`, not plain `xva`

If you want ORE factor sensitivities like `xva_zero_sensitivity_cva.csv`, enable the separate `xvaSensitivity` analytic. Plain `xva` does not emit those reports.

- Typical outputs live in `Output/xva_sensitivity/`, not necessarily directly under `Output/`
- In SWIG/native integration, treat this as a separate analytic run, not an extension of the plain `xva` contract

### Rate sensitivity parity: use direct curve-node shocks, not raw quote bumps

For ORE-vs-Python sensitivity comparison, raw quote bumping is too indirect and mixes bootstrapping differences into the result. The better comparison path is:

1. Read ORE zero-sensitivity rows
2. Map `DiscountCurve/...` and `IndexCurve/...` factors to direct node shocks
3. Rerun the Python engine with those curve-node shocks applied

This is the path used by the current comparator and it materially improved parity.

### Freeze floating spreads across shocked reruns

If the Python IRS path recalibrates `float_spread` on each bumped run, forward sensitivities collapse or get muted because the all-in coupon is re-anchored after the shock.

- Calibrate base floating spreads once
- Reuse them for all shocked reruns

This was required to make forward sensitivities like `IndexCurve/EUR-EURIBOR-6M/1/5Y` behave sensibly.

### Bucket labels can be right even when parity is wrong

For the benchmark sensitivity cases, Python bucket weights were already matching ORE bucket weights. The remaining mismatch came from downstream repricing behavior, not from the bucket interpolation formula itself.

- Do not assume a bad bucket delta means the bucket shape is wrong
- First compare the actual shocked curve values on the tenor grid before changing factor mapping logic

### ORE benchmark curve shocks are effectively `Discount + LogLinear`

For the benchmark cases here, ORE discount/index curves are configured with:

- `InterpolationVariable = Discount`
- `InterpolationMethod = LogLinear`

Applying node shocks in Python as direct zero-rate perturbations on an already-built curve overstated short-end discount sensitivities. Much better parity came from:

1. shocking the discount factors at the node tenors
2. interpolating the shocked curve in log-discount space between nodes
3. keeping flat-end behavior outside the node range

This change materially improved `flat_EUR_5Y_A` rate parity:

- `zero:EUR:1Y` moved from a large mismatch to roughly `-6.9%` relative error
- `zero:EUR:5Y` moved to roughly `-5.1%`
- `fwd:EUR:6M:5Y` moved to roughly `+2.0%`

### Credit sensitivity parity is still approximate

ORE `SurvivalProbability/...` XVA sensitivities are generated through ORE's scenario-market machinery and are not reproduced faithfully from the Python snapshot layer alone.

- Keep credit factors explicitly marked unsupported/approximate in the comparator output
- Do not present credit parity numbers as equivalent to the rate-side compare quality

### Useful diagnostics for sensitivity mismatches

When rate parity drifts, use these scripts first:

- `scripts/diagnostics/diagnose_sensitivity_bucket.py`
  - compares ORE bucket weights with Python applied shocks on sample tenors
- `scripts/diagnostics/diagnose_sensitivity_cashflows.py`
  - decomposes fixed-leg and floating-leg t0 PV response for a chosen factor

These are faster and more informative than jumping straight into full XVA debugging.

---

## Performance Optimization Order

Follow this sequence — do not skip to code tuning before confirming routing:

1. **Verify the intended engine path** — confirm it is truly AMC-CG, not mixed classic/CG
2. **Eliminate accidental fallbacks** — fix `amcTradeTypes` and engine-file coverage
3. **Identify the dominant hotspot for the confirmed path** — then optimize it

### Path-specific hotspots

| Path | Dominant hotspot |
|------|-----------------|
| Classic valuation | `ScenarioSimMarket` updates |
| AMC-CG | `fwdEval` in computation-graph forward evaluation |

Do not generalize a hotspot from one path to the other.

### Accepted optimizations from experiments on this repo (~4–5% overall)

- `swaptionvolatilityconverter.cpp`: safe ATM-strike cache with relink-aware invalidation
- `scenariosimmarket.cpp`: narrow observer suppression during construction; eager yield-curve bootstrap; precomputed vol-loop quantities; `unordered_map` for lookup-heavy maps with explicit deterministic-iteration preservation
- `genericyieldvolcurve.cpp`: replaced repeated `std::find` scans with lookup maps; reused computed ATM vols in logging
- `yieldcurve.cpp`: cached `RateHelperData` min/max pillar-date bounds

### Rejected optimization from experiments

- `forwardevaluation.hpp`: deletion-dedup optimization using a bitmap — passed unit tests, degraded benchmark, was reverted

**Lesson**: Benchmark evidence overrules algorithmic intuition. Do not keep AD/graph changes unless they improve the actual benchmark.

### Next high-value area (dominant hotspot still unaddressed)

The remaining ~95% of forward-evaluation cost is in:
- `OREAnalytics/orea/engine/xvaenginecg.cpp`
- `OREData/ored/scripting/engines/scriptedinstrumentpricingenginecg.cpp`
- `QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp`
- `QuantExt/qle/models/lgmconvolutionsolver2.cpp`
- `QuantExt/qle/models/lgmbackwardsolver.hpp`

---

## Optimization Safety Rules

### Global observer deferral is not numerically neutral

`ObservableSettings::disableUpdates(true)` in `TodaysMarket` causes numerical drift (observed in `OREDataTestSuite/TodaysMarketTests/testCorrelationCurve`). It was safe inside `ScenarioSimMarket` but not in `TodaysMarket`. Always apply in a narrow scope and validate against regression tests.

### Market-build sequencing is part of the numerical model

Eager bootstrapping and observer-management changes alter lazy construction order. This can change downstream numerics even with no formula change. Always re-run a regression guard after changing build order, lazy/eager initialization, or observer behavior.

### Cache keys must include effective curve identities for `RelinkableHandle`

A naive cache on `(optionTenor, swapTenor, indexName)` is wrong once forwarding or discount curves are relinked. Keys must reflect the underlying linked objects, or use an equivalent invalidation mechanism.

### Deterministic iteration order is separate from container type

Replacing `std::map` with `std::unordered_map` improves lookup cost but loses deterministic iteration. If downstream code assumes stable ordering (scenario hashing, serialization, regression files), preserve iteration order explicitly via sorted key vectors.

### Safe optimization pattern vs. risky pattern

**Safe (preserve semantic ordering):**
- Replace repeated linear searches with lookup tables
- Memoize repeated local pure computations
- Cache already-derived pillar-date bounds after construction
- Precompute local lookups

**Risky (require explicit numerical validation):**
- Global observer deferral
- Lazy/eager construction changes
- Stateful caching across relinked objects
- Any change that alters observable evaluation order

---

## Build and Test Setup (Apple Silicon / M-series Mac)

### Architecture consistency matters more than "library is installed"

A build can appear mostly configured and still fail at link time with unresolved Boost symbols if CMake cache steers part of the build toward `x86_64`. Use a clean native `arm64` build directory; keep Boost/CMake/compiler architecture consistent. If configure or link behavior looks strange, suspect cached architecture settings.

### Executable path for examples

`Examples/ore_examples_helper.py` expects the `ore` executable at a specific path relative to the examples tree. If an example says `ORE executable not found`, verify the exact path expected by the helper script — not just whether `build/App/ore` exists somewhere.

### Test build is separate from app build

```
-DORE_BUILD_TESTS=ON
-DORE_BUILD_SWIG=OFF   # needed in environments where SWIG configuration fails
```
Do not assume the app build is usable for validation without checking these flags.

### YieldCurveTests failures may be environment, not code regressions

Missing runtime fixture files or wrong working directory can look like numerical breakage. Confirm that input files exist in the expected runtime location before diagnosing code.

### Tracking state during benchmark iteration

- Source state and binary state are separate. Reverting a source change does not clean the binary — rebuild required.
- When comparing runtimes, confirm binary state matches source state.

---

## SWIG / Native XVA

### `app.run()` success ≠ XVA produced

A clean run (no thrown exception, no `app.getErrors()`) can return no `xva` report with `xva_total = 0`. Gate on all of:
- `xva` report exists
- `xva_total != 0`
- Expected XVA metrics present (`CVA`, `DVA`, `FVA`, `MVA` as configured)
- Report count above baseline-only threshold

### Analytic type introspection is diagnostic only

Seeing `['EXPOSURE', 'PRICING', 'XVA']` in analytic types does not guarantee XVA outputs. Treat it as diagnostic, not proof.

### Payload parity beats "reasonable-looking" config

Matching the effective payload shape of known-good stress examples (at the `InputParameters` setter level) is more predictive than conceptual dataclass field parity. Small or "plausible" synthetic payloads frequently produce zero-XVA outcomes.

---

## Time Conventions and Exposure Reports

### ORE model time and exposure-report time are not the same thing

Do not assume the `Time` column in `exposure_*` reports is the same time convention used by the LGM model or MC cashflow engine.

- **Model / MC cashflow time** uses the model term structure day counter.
  - In [`/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/mccashflowinfo.cpp`](/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/mccashflowinfo.cpp), the MC machinery uses `model->irlgm1f(0)->termStructure()->timeFromReference(d)`.
  - In [`/Users/gordonlee/Documents/Engine/OREData/ored/scripting/models/modelcgimpl.cpp`](/Users/gordonlee/Documents/Engine/OREData/ored/scripting/models/modelcgimpl.cpp), `ModelCGImpl::actualTimeFromReference()` is `dayCounter_.yearFraction(referenceDate(), d)`.
  - In [`/Users/gordonlee/Documents/Engine/OREData/ored/scripting/models/gaussiancamcg.cpp`](/Users/gordonlee/Documents/Engine/OREData/ored/scripting/models/gaussiancamcg.cpp), that `dayCounter_` is taken from `curves.front()->dayCounter()`.

- **Exposure report `Time`** is hardcoded to `ActualActual(ISDA)` off `today` and `cube()->dates()`.
  - In [`/Users/gordonlee/Documents/Engine/OREAnalytics/orea/app/reportwriter.cpp`](/Users/gordonlee/Documents/Engine/OREAnalytics/orea/app/reportwriter.cpp), both `addTradeExposures()` and `addNettingSetExposure()` do:
    - `DayCounter dc = ActualActual(ActualActual::ISDA);`
    - `time = dc.yearFraction(today, dates[j]);`
  - This is why exposure report times can disagree with `timeFromReference()` on the live market/model curve even when the underlying valuation is correct.

### In the standard XvaRisk setup here, the scenario grid is A365F

- [`/Users/gordonlee/Documents/Engine/Examples/XvaRisk/Input/simulation.xml`](/Users/gordonlee/Documents/Engine/Examples/XvaRisk/Input/simulation.xml) sets `<DayCounter>A365F</DayCounter>`.
- [`/Users/gordonlee/Documents/Engine/OREAnalytics/orea/scenario/scenariogeneratordata.cpp`](/Users/gordonlee/Documents/Engine/OREAnalytics/orea/scenario/scenariogeneratordata.cpp) parses this into the simulation grid.
- [`/Users/gordonlee/Documents/Engine/OREAnalytics/orea/engine/amcvaluationengine.cpp`](/Users/gordonlee/Documents/Engine/OREAnalytics/orea/engine/amcvaluationengine.cpp) requires the scenario grid day counter to match the model IR term structure day counter.

### SWIG-side probe: inspect the live curve instead of guessing

You do not need to recompile ORE to check the active time convention. With the existing `ORE-SWIG` bindings:

- `app.getAnalytic("EXPOSURE").getMarket().discountCurve("EUR")` returns a live `YieldTermStructureHandle`
- that handle exposes:
  - `dayCounter()`
  - `referenceDate()`
  - `timeFromReference(date)`

In the simple EUR IRS harness, the live SWIG probe showed:

- `dayCounter = Actual/365 (Fixed)`
- `referenceDate = February 5th, 2016`
- `timeFromReference(May 5th, 2016) = 0.2465753425`
- `timeFromReference(August 5th, 2016) = 0.4986301370`
- `timeFromReference(February 6th, 2017) = 1.0054794521`
- `timeFromReference(February 5th, 2019) = 3.0027397260`

### Practical rule for parity work

- If you are matching **path generation, model states, or cashflow timing inside valuation**, mirror the model curve day counter / `timeFromReference()`.
- If you are matching the **printed `Time` column in ORE exposure reports**, mirror `ActualActual(ISDA)` on `today -> report date`.
- Do not mix these two targets in the same parity test or you will chase fake timing bugs.

### Minor XML blocks can flip XVA on/off

`conventions.xml` and `counterparty.xml` differences alone changed zero vs. non-zero XVA behavior — even when major configs (`simulation`, `todaysmarket`, `curveconfig`, `pricingengine`) matched.

### Market data richness is a hidden dependency

Minimal programmatic quote sets can pass baseline pricing but fail to produce XVA reports. Full in-memory market/fixings datasets at stress-classic scale are often required for stable non-zero XVA output.

### macOS rpath failure mode

```
_ORE...so: Library not loaded: @rpath/libOREAnalytics.dylib
Reason: no LC_RPATH's found
```
This is a loader/rpath issue, not a Python package issue. Fix with `DYLD_LIBRARY_PATH` or by fixing the rpath at build time.

### "Strict native works" current definition

- Dataclass-first orchestration

### Native parity is a different debugging problem from `ore_snapshot` parity

The `native_xva_interface` path has its own failure modes. Do not assume a fix proven in
`example_ore_snapshot.py` automatically carries over to `PythonLgmAdapter`.

The native path can diverge from the snapshot parity path because of:
- different XML buffers loaded into the snapshot
- different curve-role resolution
- different IRS leg source (`portfolio.xml` vs `flows.csv`)
- different exposure grid handling
- different analytics requested by the original ORE case

Before diagnosing a native mismatch, print and trust:
- `result.metadata["input_provenance"]`
- `result.metadata["grid_size"]`
- `result.metadata["observation_grid_size"]`
- whether `calibration.xml` is present in `snapshot.config.xml_buffers`

If native metadata says:
- `market = market_overlay`
- `model_params = simulation`

on a case where the snapshot parity path uses ORE output curves and calibrated parameters, the native result is still on the wrong route.

### Native loader must pull `calibration.xml` from ORE output if present

One of the biggest native misses on `Examples/Exposure/Input/ore_measure_lgm.xml` came from a very simple omission:

- `XVALoader.from_files(...)` only loaded input-side XML files
- `PythonLgmAdapter` therefore never saw `Output/measure_lgm/calibration.xml`
- native runtime silently fell back to simulation parameters

This kept native `ore_measure_lgm` on:
- `model_params = simulation`

while the working snapshot parity path was on:
- `alpha_source = calibration`

**Implemented fix**:
- in [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/loader.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/loader.py),
  `_load_known_xml_buffers()` now loads:
  - `outputPath/calibration.xml` into `xml_buffers["calibration.xml"]`

Observed effect on native `ore_measure_lgm` at `1000` paths:
- before: native CVA about `56.85k`
- after: native CVA about `42.44k`
- ORE reference: `42.14k`

This was one of the highest-value fixes in the whole native path. If native metadata still says `model_params = simulation` on an ORE case that produced calibration output, stop there and fix the loader first.

### Native ORE-output curve loading must only resolve currencies actually used

Another major native trap: `_load_ore_output_curves()` originally asked the snapshot helper to resolve discount columns for *all currencies* in the chosen `todaysmarket.xml` configuration.

That is unsafe.

On `ore_measure_lgm`, the `default` config included unused handles like:
- `Yield/USD/USD-IN-EUR`

and `_resolve_discount_columns_by_currency()` threw:

```text
Could not resolve column name for curve handle 'Yield/USD/USD-IN-EUR'
```

Because `_load_ore_output_curves()` swallowed exceptions, the whole native path silently fell back to:
- `market = market_overlay`

This looked like a pricing/model bug, but it was really an exception-masked curve-loading bug.

**Implemented fix**:
- in [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py),
  `_load_ore_output_curves()` now resolves discount columns only for:
  - base currency
  - currencies actually used by the current trade specs

Do not use the “all currencies in config” helper in native runtime unless you actually need every one of them.

### Native curve roles must mirror the snapshot parity split

The working parity split is:
- trade repricing discount curves from the `curves` analytic config
- XVA/funding comparator from the simulation/XVA config

If native runtime uses the simulation config for both, it can still look “reasonable” but it is not matching the working snapshot path.

**Implemented native fix** in [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py):
- pricing discount curves are resolved from the `curves` analytic configuration
- `xva_discount_curve` still comes from the simulation config

This is the same separation used in `py_ore_tools.ore_snapshot`.

Practical rule:
- if native PV is close but CVA/FBA/FCA are still structurally off, check whether pricing and XVA comparator curves have been collapsed onto the same config by mistake

### Native IRS legs should prefer `flows.csv` over reconstructed portfolio legs for parity

For parity work, native runtime should not prefer schedule reconstruction if ORE already exported concrete flows.

**Implemented** in [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py):
- `_build_irs_legs(...)` prefers `Output/flows.csv` when available
- native leg build then:
  - maps times with the simulation/model day counter
  - injects node tenors
  - applies historical fixings

This removed a large class of “cashflow parity” false positives. If `flows.csv` exists and native runtime is still building IRS legs only from `portfolio.xml`, you are not on the best parity path.

### Calibrate native floating spreads after the actual forward curve is known

Loading the right leg dates is not enough. For the native route, the floating spread still needs to be calibrated against the actual forward curve selected for that trade.

**Implemented**:
- after curve construction, native runtime runs
  `calibrate_float_spreads_from_coupon(spec.legs, p_fwd, t0=0.0)`

Without this, native IRS parity can stay biased even when cashflow dates and coupons look superficially correct.

### Price on the augmented grid, but aggregate XVA on ORE observation dates only

This is a native-runtime specific structural rule.

The pricing grid wants extra dates:
- pay dates
- fixing dates
- other trade-state transition dates

But ORE XVA aggregation is still reported and integrated on the ORE observation grid, not on every augmented trade date.

If native runtime aggregates EPE/ENE and XVA on the fully augmented grid, it can overcount XVA even when pathwise pricing is reasonable.

**Implemented** in [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py):
- `_PythonLgmInputs` now carries:
  - `times`
  - `observation_times`
- pricing still runs on `times`
- EPE/ENE, CVA, DVA, FBA, FCA, MVA are sliced back to `observation_times`
- metadata now records:
  - `grid_size`
  - `observation_grid_size`

This is the right architecture even when it is not the final dominant parity lever on a given case.

### Native comparison scripts can lie by omission if the ORE case did not request a metric

A zero in native output is not always a bug.

Example:
- the `flat_EUR_5Y_A` benchmark case only requests `CVA`
- a comparison script printing `DVA = 0` is not evidence that native runtime failed to generate ENE

Observed directly:
- native exposure cube still had substantial negative exposure
- `xva_by_metric` only contained `CVA` because the case analytics requested only `CVA`

Practical rule:
- before diagnosing a missing metric, check `snapshot.config.analytics`
- if the ORE case did not request DVA/FVA/MVA, a zero printout is not a parity failure

### The most important native debugging lesson: do not trust swallowed fallbacks

Two of the largest native gaps came from silent fallback behavior:

1. `calibration.xml` not loaded -> fallback to simulation parameters
2. `_load_ore_output_curves()` exception -> fallback to raw market-overlay curves

Both produced outputs that looked numerically plausible.
Neither threw a hard error.
Both completely changed parity conclusions.

So for native debugging:
- always inspect `input_provenance`
- if a case expected ORE outputs and provenance says `market_overlay` or `simulation`, stop and fix the load path before changing formulas

### Hard-won native sequence that actually worked

For `ore_measure_lgm`, the successful order was:

1. Fix native ORE-output curve loading
2. Fix native curve-role split (pricing vs XVA comparator)
3. Load `calibration.xml` from output
4. Prefer `flows.csv` for IRS legs
5. Calibrate floating spreads after curve build
6. Keep augmented pricing grid but aggregate on ORE observation dates

Trying to tune deflation, funding, or hazard details before steps 1-3 was mostly wasted effort.

### Reflection: the biggest parity gains came from plumbing, not formulas

The native path improved most from:
- loading the same ORE artifacts the snapshot path already trusted
- preventing silent fallback
- separating roles cleanly:
  - pricing curve
  - XVA comparator curve
  - pricing grid
  - observation grid
  - simulation params
  - calibrated params

The expensive mistake pattern was:
- see a large CVA gap
- assume a stochastic/model/formula problem

In practice, the dominant causes were repeatedly:
- wrong artifact source
- wrong fallback path
- wrong config role
- wrong grid role

When parity is badly wrong, first prove that native runtime is consuming the same artifacts and conventions as the known-good path. Only after that should you touch formulas.

---

## Python LGM vs ORE Parity Notes

### Exact RNG parity is only realistic in a narrow mode

For 1F LGM path matching, exact Ore/Python RNG parity is achievable only when all of the following are locked:
- `SequenceType = MersenneTwister`
- same integer seed
- same Gaussian transform (`InverseCumulativeNormal`)
- same variate consumption order

Do **not** target exact parity against Sobol / Brownian-bridge configurations. In this repo:
- Ore/QuantExt pseudo-random path generation is MT19937 uniform -> inverse normal
- Python parity mode now uses QuantLib-backed MT Gaussian draws rather than NumPy's `default_rng`

### Draw ordering matters as much as the seed

The first Python LGM simulator consumed normals in time-major order (`step -> all paths`). Ore consumes them path-by-path. Matching the seed alone did not produce path parity.

- **Implemented in** `py_ore_tools.lgm`: `OreMersenneTwisterGaussianRng`, `make_ore_gaussian_rng()`, and `simulate_lgm_measure(..., draw_order="ore_path_major")` use QuantLib-backed MT Gaussian draws and match Ore's path-by-path variate order.
- **Implemented parity mode**: `draw_order="ore_path_major"` in `simulate_lgm_measure`
- **Implemented native switch**: `snapshot.config.params["python.lgm_rng_mode"] = "ore_parity"` routes `PythonLgmAdapter` (in `native_xva_interface.runtime`) through the Ore-compatible RNG and draw order.
- **Oracle / tests**: `scripts/dumps/dump_ore_lgm_rng_parity_case.py`, artifact `parity_artifacts/lgm_rng_alignment/mt_seed_42_constant.json`, and tests in `tests/test_lgm.py` validate RNG alignment.

If parity tests still differ after aligning the seed, verify draw ordering before changing formulas.

### Do not compare full-stress ORE against simplified Python formulas and call it a bug

If ORE runs with full stress-classic runtime wiring (credit/funding/collateral/XVA decomposition), Python must either:
- match those conventions/decompositions explicitly, or
- be compared under reduced scope (for example CVA-only)

Otherwise large differences are structural by design, not necessarily implementation defects.

### FVA decomposition convention is a major parity lever

In this codebase setup, ORE-aligned funding decomposition behaves as:
- `FBA <- ENE * lending spread`
- `FCA <- EPE * borrowing spread`
- `FVA = FBA + FCA`

Reversing the orientation (or using a single generic spread) can create order-of-magnitude FVA errors.

### Prefer market-implied funding spreads over heuristic constants

When present, source funding spreads from market quotes:
- `ZERO/YIELD_SPREAD/<ccy>/<curve>/...`

Typical examples in this repo:
- `BANK_EUR_BORROW`
- `BANK_EUR_LEND`

Using these improved FVA parity dramatically versus fixed heuristic spread assumptions.

### Own-name hazard curve can be missing even when `dvaName` is configured

`dvaName`/own-name (for example `BANK`) may be present in runtime config while explicit own hazard quotes are absent in compact market sets.
You need an explicit fallback policy; this materially impacts DVA/FVA.

- **Implemented fallback**: `load_ore_default_curve_inputs` (in `irs_xva_utils`) tries `HAZARD_RATE/RATE/<name>/...` first; if no points are found, it falls back to `CDS/CREDIT_SPREAD/<name>/...` and uses those spreads as hazard pillars (e.g. flat market with BANK only has CDS quotes). OreSnapshot loads own credit from market when `dvaName` is set in ore.xml and exposes `own_hazard_times`, `own_hazard_rates`, `own_recovery` when present.

### Important correction: CDS spread fallback is not the same thing as a hazard curve

The initial fallback above was still too crude for own-name curves like `BANK`. In the standard exposure example:

- `CPTY_A` is quoted directly in `HAZARD_RATE/RATE/...`
- `BANK` is quoted in `CDS/CREDIT_SPREAD/...`

Treating those CDS spreads as raw hazards materially understates DVA. The lightweight Python fallback should instead approximate ORE's default-curve bootstrap as:

```text
hazard ~= cds_spread / (1 - recovery)
```

This single correction was decisive on the standard `ore_measure_lgm` case:
- before: Python DVA stayed around `35k-38k` vs ORE `61.2k`
- after converting CDS spreads to `spread / LGD`: Python DVA moved to about `59.9k`

So for compact parity work:
- `HAZARD_RATE` quotes can be used directly
- `CDS/CREDIT_SPREAD` quotes must be converted to hazards, not copied verbatim

This is especially important for own-name (`dvaName`) because ORE examples often quote the bank that way even when counterparties are already in hazard space.

### Node-tenor interpolation is a major XVA-parity trap

The biggest remaining dynamic exposure error on the main `ore_measure_lgm` case was not RNG, cashflow rows, or CVA aggregation. It was the swap pricer's use of node-tenor discount-factor interpolation during pathwise revaluation.

Observed on the same simulated paths:
- with node interpolation:
  - median EPE rel diff about `11.9%`
  - median ENE rel diff about `7.3%`
  - Python CVA about `46.96k`
  - Python DVA about `35.02k`
- with exact bond evaluation (same curves, same paths, same cashflows):
  - median EPE rel diff about `3.5%`
  - median ENE rel diff about `1.2%`
  - Python CVA about `43.29k`
  - Python DVA about `59.91k` after the CDS/LGD fix above

Key lesson:
- the "simulate node tenors, then interpolate" path is a useful approximation mode
- it is **not** the right default for parity against ORE XVA on this repo's benchmark cases

**Implemented fix**:
- `swap_npv_from_ore_legs_dual_curve(..., use_node_interpolation=False)` now defaults to exact discount-bond evaluation
- node interpolation remains available explicitly for diagnostics via `use_node_interpolation=True`

### ORE-style FBA / FCA are curve-increment adjustments, not flat spread times dt

The simple Python funding proxy

```text
FCA ~= borrow_spread * EPE * dt
FBA ~= lend_spread * ENE * dt
```

is structurally wrong for ORE parity.

From `OREAnalytics/orea/aggregation/xvacalculator.cpp` and `staticcreditxvacalculator.cpp`, ORE computes:

- `FCA increment = S_cpty(d0) * S_own(d0) * EPE(d1) * dcf_borrow`
- `FBA increment = S_cpty(d0) * S_own(d0) * ENE(d1) * dcf_lend`

where:

```text
dcf_borrow = P_borrow(d0)/P_borrow(d1) - P_ois(d0)/P_ois(d1)
dcf_lend   = P_lend(d0)/P_lend(d1)   - P_ois(d0)/P_ois(d1)
```

Practical implications:
- ORE FBA/FCA are survival-weighted forward discounting differences between funding and OIS curves
- they are not just "spread times exposure times year fraction"
- both counterparty and own survival at `d0` matter

**Implemented on Python side**:
- `OreSnapshot` now carries optional funding curves from `fvaBorrowingCurve` / `fvaLendingCurve` when they are present in `ore.xml`
- `compute_xva_from_exposure_profile()` accepts:
  - `funding_discount_borrow`
  - `funding_discount_lend`
  - `funding_discount_ois`
- when these are present, it computes ORE-style `fba_terms` / `fca_terms` instead of the flat proxy

### Funding comparator curve matters: for FBA/FCA parity, do not assume it is the same as the Python discount curve

One subtle but decisive point from the `ore_measure_lgm` case:

- the Python trade/exposure parity path uses `discount_column = EUR-EONIA`
- but ORE funding parity on this case is much closer when the FBA/FCA comparator uses the **XVA/simulation base-currency discount curve**
- in this case that curve is:
  - `xva_discount_column = EUR-EURIBOR-6M`

This matches the relevant ORE setup better than using the `curves` analytic discount column as the funding/OIS comparator.

Observed effect on the main case:

- using `EUR-EONIA` as funding comparator:
  - Python FBA about `27.75k`
  - Python FCA about `-6.33k`
- using `EUR-EURIBOR-6M` as funding comparator:
  - Python FBA about `29.36k`
  - Python FCA about `-4.49k`
- ORE reference:
  - FBA `29.68k`
  - FCA `-4.36k`

So for funding parity, the correct comparator curve can differ from the trade repricing discount curve.

**Implemented**:
- `OreSnapshot` now carries:
  - `xva_discount_column`
  - `curve_times_xva_disc`
  - `curve_dfs_xva_disc`
  - `p0_xva_disc`
- `example_ore_snapshot.py` now uses `p0_xva_disc` as the comparator for ORE-style funding increments

### Funding-gap diagnosis order

If FBA/FCA are still off after implementing ORE-style funding increments:

1. Check the comparator curve first
   - Do not assume `discount_column` is the right OIS/base comparator for funding
   - inspect the XVA/simulation config and compare against ORE output

2. Check with ORE EPE/ENE plugged directly into the Python funding formula
   - if the gap remains large, the issue is formula/curve-role related
   - if the gap shrinks, the issue is still mostly exposure shape

3. Compare `exOwnSP` / `exAllSP` style variants
   - this separates survival-probability issues from pure funding-curve increment issues

On the benchmark case here, plugging ORE EPE/ENE into the Python funding formula showed the remaining miss was mostly formula/curve-role, not exposure.

### Current main-case status after the full XVA push

On `Examples/Exposure/Input/ore_measure_lgm.xml` with `2000` paths and seed `42`, after:
- numeraire-deflated XVA aggregation
- exact bond evaluation instead of node interpolation
- CDS spread to hazard conversion for own-name curves

the parity moved to roughly:
- Python CVA: `43.29k` vs ORE `42.14k` (`~2.7%`)
- Python DVA: `59.91k` vs ORE `61.22k` (`~2.1%`)
- median EPE rel diff: `~3.5%`
- median ENE rel diff: `~1.2%`

At that point the remaining task is mainly funding-adjustment parity (FBA/FCA), not core CVA/DVA exposure generation anymore.

### Updated main-case status after funding comparator fix

On `Examples/Exposure/Input/ore_measure_lgm.xml` with `2000` paths, seed `42`, after:
- numeraire-deflated XVA aggregation
- exact bond evaluation instead of node interpolation
- CDS spread to hazard conversion for own-name curves
- ORE-style funding increments with the **XVA/simulation discount curve** as comparator

the parity is now roughly:

- CVA:
  - Python `43.29k`
  - ORE `42.14k`
  - diff about `+2.7%`
- DVA:
  - Python `59.91k`
  - ORE `61.22k`
  - diff about `-2.1%`
- FBA:
  - Python `29.36k`
  - ORE `29.68k`
  - diff about `-1.1%`
- FCA:
  - Python `-4.49k`
  - ORE `-4.36k`
  - diff about `-2.9%`

At this point:
- FBA/FCA are no longer the dominant mismatch
- the remaining residual is mainly:
  - exposure shape in CVA/DVA
  - small `t0` NPV miss

### IRS portfolio payload completeness is mandatory for ORE parity

A fixed-only IRS payload is insufficient for parity. Ensure generated trade XML includes:
- both fixed and floating legs
- schedule blocks and core leg fields
- rule fields (`TermConvention`, `Rule`, `EndOfMonth`) when reconstructing ORE-style schedules

Incomplete payloads can run but produce misleadingly weak or zero-like outputs.

### Remove bundled `portfolio.xml` when testing programmatic SWIG trades

In `native_xva_interface`, if `snapshot.config.xml_buffers` already contains `portfolio.xml`, the mapper keeps that file and ignores the dataclass portfolio. This can silently cause ORE-SWIG to price the template portfolio instead of the simple injected trade.

For clean programmatic trade comparisons:
- remove `portfolio.xml`
- remove `netting.xml`
- remove `collateralbalances.xml`

before building the snapshot used for the SWIG adapter.

### `pv_total` is not always available in XVA-focused ORE runs

Depending on analytic configuration, ORE may not emit an `npv` report. Avoid treating missing/zero `pv_total` as immediate pricing failure without checking report configuration/output set.

In the programmatic SWIG harness (simple receiver-fixed IRS via `native_xva_interface`), `pv_total` can still come back as `0.0` even when exposure and CVA are produced. For that probe, treat exposure/CVA as the trustworthy comparison and do not rely on PV for parity.

### Current parity snapshot (single-IRS prototype)

After convention and mapping fixes:
- DVA: close
- FVA: close
- MVA: direction/magnitude aligned
- CVA: still residual gap (primarily exposure-shape calibration issue)
- In-memory SWIG execution
- No runtime file I/O
- Non-zero XVA achieved

It does **not** yet mean full autogenerated runtime XML parity from a minimal dataclass schema without compatibility-template assistance.

### Updated simple-IRS convergence snapshot

Using:
- single receiver-fixed EUR IRS
- generated programmatic portfolio in `native_xva_interface`
- stress-classic compatible runtime bundle
- `python.lgm_rng_mode = ore_parity`

Observed progression on this repo:
- initial simple IRS gap after RNG wiring: about `10.6%` → `7.1%` with ore_parity
- after schedule-generation fixes: about `6%`
- after forward-curve merge fix: about `3.3%` (e.g. Python CVA 882, ORE CVA 854)

This means the current dominant mismatch is no longer RNG. The remaining gap is in coupon/fixing/forwarding conventions.

**Note**: On the *file-based* comparison (`compare_ore_python_lgm.py` with `--rng-mode ore_parity`), RNG parity did **not** materially improve CVA (gap stayed ~18–20%); the main mismatch there was elsewhere (e.g. schedule/curves). So RNG parity helps the programmatic simple-IRS probe; the file-based fixed LGM IRS case has other dominant factors.

---

## ORE-Python Parity (LGM / IRS / Bermudan / XVA)

These lessons apply when matching Python implementations against ORE output for LGM-model trades.

### Exercise-date mapping must use ORE-style `lower_bound`

ORE maps contractual exercise dates to the **first simulation date ≥ the exercise date** (`lower_bound` semantics). Nearest-grid snapping exercises too early and can overprice Bermudans by ~37%. Use:
```python
searchsorted(grid, exercise, side="left")   # not nearest-date logic
```

### Bermudan exposure after exercise requires a mode switch

For physical settlement, ORE transitions from option/continuation logic to underlying-swap logic post-exercise. Carrying only the pre-exercise continuation surface leaves exposure wrong even when PV looks reasonable. Carry pathwise exercise state and switch valuation mode after exercise.

### Pathwise fixing lock — lock realized fixings

Once a fixing date has passed on a path, lock that realized fixing and reuse it. Reprojecting already-fixed coupons at later valuation dates is a systematic ORE parity error.

### Use ORE cashflow signs as canonical truth

Reconstructing signs from payer/receiver labels is fragile. Exported `Amount` signs in `flows.csv` are more reliable for parity than semantic labels.

- **OreSnapshot** (`py_ore_tools.ore_snapshot`): when `flows.csv` exists under the ORE output path, legs are loaded via `load_ore_legs_from_flows` (Amount signs) instead of `load_swap_legs_from_portfolio` (Payer labels). This removed the t0 NPV sign flip in practice (e.g. flat_EUR_5Y_A: portfolio-only gave Python t0 NPV opposite to ORE; flows-based legs gave same sign and ~9% level gap).
- If the run has no cashflow analytic (no `flows.csv`), snapshot falls back to portfolio XML; for those runs a manual sign flip (e.g. `--flip-npv-perspective` in `example_ore_snapshot.py`) may be needed if t0 NPV is opposite to ORE.

### ORE cashflow report is available directly in SWIG

For programmatic debugging, the SWIG app can emit a trade cashflow report without writing files:
- set analytics to include `CASHFLOW`
- call `setStoreFlows(True)`
- read `app.getReport("cashflow")`

This is the fastest way to compare trade-level ORE cashflows against Python-generated leg arrays when `flows.csv` is not already available on disk.

### Schedule mismatch can survive "same dates" unless accrual boundaries are generated the ORE way

For generated IRS trades in `native_xva_interface`, the initial gap was not just RNG. ORE and Python differed because schedule reconstruction was too simple.

Observed lessons:
- adjusting only payment dates is wrong
- carrying adjusted period ends forward is also wrong
- ORE advances on the **unadjusted anchor schedule** and then adjusts each boundary independently

**Where it was fixed**: In `native_xva_interface.mapper`, added `TermConvention`, `Rule`, `EndOfMonth` to generated IRS schedule XML. In `py_ore_tools.irs_xva_utils`, updated `_build_schedule()` to follow the ORE rule above (advance on unadjusted anchor, then adjust each boundary independently). Getting this right materially improved the fixed-leg coupon rows against the ORE cashflow report.

### Front-end forwarding was a larger remaining gap than RNG after schedule fixes

After RNG parity and schedule fixes, the simple IRS still differed because the Python adapter's forward-curve construction was too crude.

Key lesson:
- do not pick a single EUR forward bucket (`3M`/`6M`/`1M`) and call that the forwarding curve
- merge available non-ON forward quotes across maturities, deduplicate by maturity, and build the forwarding curve from that merged set

In this repo's simple IRS probe, that forward-curve merge reduced the residual CVA gap materially after schedule alignment.

### Historical floating coupons must not be reprojected

The native `PythonLgmAdapter` still has a parity risk if generated IRS legs keep `float_coupon = 0` for already-fixed periods and rely on projection instead of historical fixings. ORE cashflow reports exposed this clearly on the first front coupon.

**Current gap**: `_build_irs_legs_from_trade()` in `native_xva_interface.runtime` still sets `float_coupon = 0` and does not map historical fixings into already-fixed coupons. Fixing injection is not yet implemented.

When trade-level parity matters:
- map snapshot fixings onto generated floating coupons in `PythonLgmAdapter` / `_build_irs_legs_from_trade()`
- treat already-fixed coupons as deterministic cashflows
- only project future coupons

**Suggested next step**: Implement fixing injection; rerun the simple receiver-fixed IRS case with `python.lgm_rng_mode=ore_parity`; compare the first few ORE `cashflow` rows to the resulting Python deterministic front coupons to see if the ~3% CVA gap closes further.

### "Same curves" means same ORE column and same economic role

Discount-vs-forward curve mixups move PV and exposure materially even when market data looks close. Lock down explicitly per case:
- Which ORE column is used for discounting
- Which is used for index forwarding

### Node-tenor interpolation on the forward side is not optional for IRS parity

Turning node-tenor interpolation on materially improves IRS agreement with ORE. Without it, Python can be directionally correct but wrong enough at coupon level to show up in PV and CVA.

### Best accepted IRS/CVA calibrated dual-curve configuration

```
swap_source          = trade          (not flows)
discount curve       = EUR-EONIA
forward curve        = EUR-EURIBOR-6M
node_tenor_interp    = ON
coupon_spread_calib  = OFF
```
More adjustments did not automatically improve parity — prefer the simplest configuration that wins empirically.

### Test `swap_source=trade` and `swap_source=flows` separately

`flows.csv` seems closer to "what ORE actually paid" but `trade` was the winning source in some calibrated baselines. Never assume one is better without testing both. For **OreSnapshot**-based parity, prefer **flows** when available (canonical Amount signs); use trade (portfolio) only when flows.csv is absent or has no interest flows for the trade.

### `alpha_scale` is a dangerous hidden tuning knob

`alpha_scale=1.05` vs `1.00` moved CVA gap from ~0.43% to ~2.82% (single-seed). Alpha can make parity look fixed while actually changing exposure dispersion rather than conventions. Keep `alpha_scale=1.0` as the baseline; use alpha ladders as explicit sensitivity only.

### Single-seed parity results are misleading

Multi-seed reruns show that apparent single-seed gaps are partly Monte Carlo dispersion. Recheck "improvements" with multiple seeds before promoting to accepted conventions.

### More paths reduces noise but does not fix convention mismatches

Solve the convention layer first (fixing treatment, curve roles, source choice, interpolation), then add paths.

### ORE XVA fields that are zero or `#N/A` often reflect setup, not economics

Zero/missing `DVA`, `FVA`, `MVA`, `KVA` usually means analytics toggles or incomplete funding/credit setup. Never interpret XVA columns without checking analytic flags and supporting configuration.

### Rebuild CVA from exposure when trade-level XVA output is sparse

```
CVA = sum(LGD × EPE_i × DF_i × dPD_i)
```
Building from `exposure_trade_*.csv` + discount factors + default probabilities is more robust than relying on trade-level XVA report entries.

### CVA term table is the highest-ROI parity diagnostic

The term-by-term decomposition `LGD × EPE_i × DF_i × dPD_i` reveals whether a mismatch comes from exposure shape, discounting, or default probabilities. Inspect it before changing model choices.

- **PV & EPE/ENE diagnostic** (in `example_ore_snapshot.py`): compare Python t0 NPV vs ORE; then "CVA with Python EPE" vs "CVA with ORE EPE" (same formula) vs ORE CVA, so the gap splits into "from EPE" vs "from formula/df/dpd". Same for DVA with ENE. If "CVA with ORE EPE" is close to ORE CVA, the residual is exposure; if not, the formula or default-prob treatment differs. For DVA, a large "gap from rest" (e.g. using ORE ENE with our formula still far below ORE DVA) means ORE's DVA aggregation or own-credit treatment differs — fix that before chasing ENE profile.

### Build fixed-parameter parity first; calibration can hide bugs

- `fixed` lane: fixed alpha/kappa for strict implementation parity
- `calibrated` lane: only after the fixed lane is green

Calibration differences can mask convention bugs.

### Closeout / MPOR is second-order until PV parity is solved

A simplified `V(t + MPOR)` closeout mapping changes CVA but does not fix the main gap if valuation logic itself is still wrong. Prioritize deterministic PV and exposure conventions first.

### Put parity logic in `.py` modules, not notebooks

Notebook-local pricing logic drifts from module implementations quickly. Keep notebooks thin; put valuation, exposure, and parity logic in `.py` modules.

### OreSnapshot: single entry point and XVA comparison

`load_from_ore_xml(ore_xml_path)` resolves the full ORE chain (ore.xml → simulation, portfolio, todaysmarket, Output). It loads `ore_cva`, `ore_dva`, `ore_fba`, `ore_fca` from the aggregate row of `xva.csv` (empty TradeId), so parity scripts can compare CVA/DVA/FVA without ad-hoc CSV parsing. Legs are built from `flows.csv` when present (Amount signs), else from portfolio XML.

### Highest-value recurring parity diagnostics

- Per-date `ORE_EPE` vs `PY_EPE`
- CVA term table (and "CVA with ORE EPE" vs "CVA with Python EPE" to split gap)
- PV & EPE/ENE diagnostic (t0 NPV, Sum(df×EPE×dt), gap-from-EPE vs gap-from-formula)
- Configuration sweep table (source, curve role, fixing lock, interpolation, alpha scale)
- Selected deterministic PV checkpoints

If you cannot explain the gap using these, you probably do not yet understand the failure mode.

### Artifact layout can silently block a fair parity run

The parity script may expect `exposure_trade_*.csv` directly under the output directory, but ORE may nest artifacts under `exposure/` or `curves/`. A file-not-found failure here is not a modeling problem — validate artifact layout first.

---

## Performance Timing (LGM / Products / General ORE)

### `logMask` has a huge impact on wall-clock time

| logMask | What it logs | Observed impact |
|---------|-------------|-----------------|
| 255 | All levels (Alert+Critical+Error+Warning+Notice+Debug+Data+Memory) | ~7 s, 121k+ line log |
| 15 | Alert+Critical+Error+Warning (ORE default) | moderate |
| 2 | Critical only | ~2.6 s — ~2.7× faster |

Bit values (from `OREData/ored/utilities/log.hpp`): `1`=Alert, `2`=Critical, `4`=Error, `8`=Warning, `16`=Notice, `32`=Debug, `64`=Data, `128`=Memory.

Use `logMask=2` or `15` for performance-sensitive runs; reserve 255 for debugging only.

### Most wall-clock time is not in the pricing engine

For the Products Bermudan example: the LGM grid engine itself was ~2 ms. Time was dominated by:
- `buildMarket()`: ~1.1–1.3 s
- `buildPortfolio()`: ~0.17–0.25 s
- Framework/logging overhead: ~3+ s at full verbosity

Before optimizing a specific engine, check `runtimes.csv` and `pricingstats.csv`. If engine time is tiny, focus on `logMask`, market build scope, and framework overhead.

### Where time in `buildMarket()` actually goes

- LGM calibration: small fraction (~0.19 s)
- Broad market-object construction, curve bootstraps, swaption/vol surface setup: majority
- `implyBondSpreads`: may run unconditionally even when the portfolio doesn't need it
- `buildCalibrationInfo`: can often be disabled for pricing-only runs

For fast iteration: trim market config to curves/vols actually needed, consider `lazyMarketBuilding`, gate bond-spread and calibration-info where possible.

### ORE's internal timers don't account for all wall time

`runtimes.csv` reports in microseconds (e.g. `1310000` = 1.31 s). Wall-clock is often several seconds longer due to framework startup, config loading, and logging. Use wall-clock for end-to-end benchmarking; use `runtimes.csv` for relative phase breakdown only.

### Output files for timing and parity work

| File | Contents |
|------|----------|
| `runtimes.csv` | Phase timings in microseconds (PRICING, buildMarket, buildPortfolio, etc.) |
| `pricingstats.csv` | Per-trade pricing count and average timing; confirms engine is negligible fraction |
| `additional_results.csv` | Engine-level timings if engine writes to `additionalResults_` |
| `log.txt` | At `logMask=255` becomes very large and dominates I/O |

### Bump-and-reval optimization priority

Repeated valuations are dominated by market build and framework overhead, not the numeric engine's inner loop (LGM grid engine batched rollback showed no measurable gain). Highest leverage:
1. Reduce logging
2. Narrow market build (curves/vols needed only for bumped scenario)
3. Reuse structures (preprocessed grid, calibration) where ORE allows
4. Engine-level and SIMD improvements only if profiling shows engine as significant fraction

---

## Native Interface Pipeline (`Tools/PythonOreRunner/native_xva_interface`)

### The pipeline has three separate checkpoints

1. ORE files load into Python snapshot (`XVALoader.from_files(...)`)
2. Snapshot maps into ORE input parameters
3. ORE-SWIG runtime accepts those parameters and runs

"Loader works" does not imply step 3 works. Both `ore_stress_classic.xml` and `ore_swap.xml` loaded cleanly in step 1 but both failed step 3 with `XML node with name Simulation not found`.

### Preferred smoke-test case: `Examples/Exposure/Input/ore_swap.xml`

`ore_swap.xml` (1 trade, 8126 quotes, 5454 fixings) is much faster than the XvaRisk stress setup (5 trades, 7881 quotes) for verifying interface plumbing. Use it as the default real-files integration check before moving to XVA stress setups.

### Minimal end-to-end repro shape

```bash
# real files + one trade + tiny path override
input bundle: Examples/Exposure/Input/
ore file:     ore_swap.xml
override:     --num-paths 10
```
This gives: successful snapshot load + successful toy runtime + reproducible ORE-SWIG failure point.

### Toy adapter vs. real ORE adapter

- Toy adapter: use to prove snapshot construction, portfolio/netting wiring, and runtime session machinery work — not for parity claims
- Real ORE adapter: required for any number matching

### pytest from repo root needs PYTHONPATH

```bash
# fails
pytest Tools/PythonOreRunner/native_xva_interface/tests/test_step4_mapper.py

# works
PYTHONPATH=Tools/PythonOreRunner pytest Tools/PythonOreRunner/native_xva_interface/tests/test_step4_mapper.py
```
Alternatively, run pytest from inside `PythonIntegration/`.

### Separate import failures from ORE failures immediately

`ModuleNotFoundError` (Python packaging) and `XML node with name Simulation not found` (ORE runtime) look similar if all you see is "the test/demo does not run". Confirm module imports first; only then chase ORE XML/runtime issues.

### Useful test commands for LGM / native adapter

From repo root:
```bash
PYTHONPATH=Tools/PythonOreRunner python3 -m pytest Tools/PythonOreRunner/tests/test_lgm.py -q
# expect: 13 passed

PYTHONPATH=Tools/PythonOreRunner python3 -m pytest Tools/PythonOreRunner/native_xva_interface/tests/test_python_lgm_adapter.py -q
# expect: 5 passed
```

The file-based compare script supports `--rng-mode ore_parity` for Ore-compatible RNG/draw order: `py_ore_tools.demos.compare_ore_python_lgm`.

---

## Key Operational Rules Summary

| Situation | Rule |
|-----------|------|
| Run looks like AMC-CG | Verify `amcTradeTypes` covers every trade type in portfolio |
| Numbers don't match | Check routing first, math second |
| Surprisingly fast run | Verify outputs exist and process exited cleanly |
| CSV truncation errors | Check for concurrent ORE processes on same output dir |
| Performance optimization | Verify path → eliminate fallbacks → then tune |
| Code change passes tests | Still benchmark on real AMC-CG flow before keeping it |
| Observer/market-build change | Run regression suite, especially `testCorrelationCurve` |
| SWIG `app.run()` succeeds | Explicitly check for non-zero XVA report |
| Apple Silicon build issues | Check for `x86_64` contamination in CMake cache |
| Bermudan exercise mapping | Use `searchsorted(grid, date, side="left")` |
| IRS/CVA parity off | Check curve roles, fixing lock, node-tenor interp before model |
| CVA gap diagnosis | Inspect CVA term table before changing model choices |
| Python t0 NPV opposite sign to ORE | Prefer legs from `flows.csv` (Amount signs); OreSnapshot does this when flows.csv exists. Else try `--flip-npv-perspective` in example script. |
| DVA gap large even with ORE ENE | Gap is likely formula/own-credit (ORE aggregation or default prob); fix that before chasing ENE profile. |
| Simple IRS CVA gap ~3% (programmatic) | Next lever: map snapshot fixings into generated IRS legs; compare ORE cashflow vs Python front coupons. |
| `runtimes.csv` sum ≠ wall-clock | Normal — gap is framework/logging overhead |
| Native loader succeeds | Still test SWIG runtime step separately |
| pytest import error | Set `PYTHONPATH=PythonIntegration` |

---

## Hard-Won IRS / XVA Reconciliation Notes

### Separate the targets: direct trade pricing, XVA-state exposure, and report time are different objects

This debugging thread only became coherent once the three parity targets were treated as distinct:

1. **Direct trade pricing parity**
   - compare against `npv.csv`
   - target is raw trade NPV at a valuation date
   - uses the trade pricing engine / pricing market config

2. **XVA-state parity**
   - compare against `exposure_trade_*.csv`, netting-set exposures, and `xva.csv`
   - target is numeraire-deflated exposure state, not raw `V(t)`
   - uses cube interpretation, exposure aggregation, and XVA integration

3. **Exposure report `Time` parity**
   - target is the printed `Time` column in ORE reports
   - this is a report-writer convention, not the model time convention

Trying to force all three through one notion of “cashflow parity” or one `discount_column` causes confusion and fake fixes.

### ORE XVA uses numeraire-deflated NPVs, not raw trade values

Confirmed in local source:

- [`/Users/gordonlee/Documents/Engine/OREAnalytics/orea/engine/valuationcalculator.cpp`](/Users/gordonlee/Documents/Engine/OREAnalytics/orea/engine/valuationcalculator.cpp)
  - `NPVCalculator::npv()` stores `trade->instrument()->NPV() * fx / simMarket->numeraire()`
  - `CashflowCalculator` stores MPOR cashflows divided by the numeraire as well

- [`/Users/gordonlee/Documents/Engine/OREAnalytics/orea/aggregation/exposurecalculator.cpp`](/Users/gordonlee/Documents/Engine/OREAnalytics/orea/aggregation/exposurecalculator.cpp)
  - EPE/ENE are built directly from the cube values via `max(npv, 0)` / `max(-npv, 0)`
  - Basel EE is recovered by dividing those exposures by a discount factor later

- [`/Users/gordonlee/Documents/Engine/OREAnalytics/orea/aggregation/staticcreditxvacalculator.cpp`](/Users/gordonlee/Documents/Engine/OREAnalytics/orea/aggregation/staticcreditxvacalculator.cpp)
  - CVA/DVA increments multiply default-probability changes by exposure cube values directly
  - there is no extra `P(0,t)` applied there

Consequence for Python:

- If the target is ORE XVA / exposure parity, aggregate on **deflated** NPVs:
  - `deflated_npv = V(t) / N(t)`
- Do **not** treat ORE `EPE` like raw undiscounted `V(t)` and then multiply by another discount factor again.

### ORE report time and model time are different on purpose

The local source confirms the split:

- model / MC cashflow time:
  - [`/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/mccashflowinfo.cpp`](/Users/gordonlee/Documents/Engine/QuantExt/qle/pricingengines/mccashflowinfo.cpp)
  - uses `model->irlgm1f(0)->termStructure()->timeFromReference(d)`

- CG / scripting model time:
  - [`/Users/gordonlee/Documents/Engine/OREData/ored/scripting/models/modelcgimpl.cpp`](/Users/gordonlee/Documents/Engine/OREData/ored/scripting/models/modelcgimpl.cpp)
  - `actualTimeFromReference(d) = dayCounter_.yearFraction(referenceDate(), d)`

- exposure report `Time`:
  - [`/Users/gordonlee/Documents/Engine/OREAnalytics/orea/app/reportwriter.cpp`](/Users/gordonlee/Documents/Engine/OREAnalytics/orea/app/reportwriter.cpp)
  - hardcodes `ActualActual(ISDA)` on `today -> report date`

Operational rule:

- use **model time** for valuation / coupon timing / simulation
- use **report time** only for comparing to ORE output columns

### The swap cashflow rows were eventually reconciled; the big remaining gap was not coupon generation

The triptych diagnostic added during this effort:

- [`/Users/gordonlee/Documents/PythonOreRunner/scripts/diagnostics/diagnose_cashflow_triptych.py`](/Users/gordonlee/Documents/PythonOreRunner/scripts/diagnostics/diagnose_cashflow_triptych.py)
- output example:
  - [`/Users/gordonlee/Documents/Engine/Examples/Exposure/Output/measure_lgm/cashflow_triptych.csv`](/Users/gordonlee/Documents/Engine/Examples/Exposure/Output/measure_lgm/cashflow_triptych.csv)

This compares:

- ORE direct cashflows from `flows.csv`
- Python portfolio-reconstructed leg rows
- Python active leg rows used in the snapshot / XVA path
- XVA-state classification (`dead`, `fixed_alive`, `projected`) at a chosen observation date

Important result from the final run on `Swap_20`:

- fixed rows matched
- floating pay/start/fixing dates matched
- floating coupons matched
- floating amounts matched

This means:

- the remaining large parity gap was **not** because Python lost the first fixing
- it was **not** because ORE cashflow rows were missing on the Python side
- it was **not** because `flows.csv` sign handling was wrong anymore

### The floating leg itself was reconciled coupon-by-coupon; the large t0 gap came from the pricer shortcut

The decisive debugging step was to break the floating leg into coupon PV contributions at `t=0`.

Observed for `Swap_20` after the cashflow fixes:

- ORE exported float PV:
  - `-3,257,143.6087227073`
- Python reconstructed float PV from the same coupon rows:
  - `-3,257,143.6087783747`
- difference:
  - about `-5.6e-05`

So the float leg itself was already aligned.

The real culprit was the full swap pricer:

- deterministic sum of the exact exported legs at `t=0`:
  - about `464.39`
- `swap_npv_from_ore_legs_dual_curve(..., t=0, x=0)` before the fix:
  - about `-9647.90`

That isolated the gap to the pricing shortcut inside the pricer, not to cashflow generation.

### Major fix: disable node-tenor interpolation at t=0 in the dual-curve swap pricer

The crucial bug was in:

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py)
  - `swap_npv_from_ore_legs_dual_curve()`

The function was interpolating discount bonds off `node_tenors` even at `t=0`.

For this case, that approximation was catastrophically wrong:

- with node interpolation at `t=0`:
  - around `-9647.90`
- with nodes disabled at `t=0`:
  - around `464.39`

Fix:

- when `t == 0`, do **not** use node-tenor interpolation
- use the exact initial discount / forward curves directly

This collapsed the raw `t0` pricing gap:

- before:
  - Python `t0` NPV `-9647.90`
  - ORE `t0` NPV `602.49`
- after:
  - Python `t0` NPV `464.39`
  - ORE `t0` NPV `602.49`

Residual gap after this fix:

- about `-138.09`

That was the biggest single pricing fix in the entire debugging thread.

### Fixed floating coupons should use stored ORE amounts when available

Another correct but smaller fix:

- when a floating coupon is already fixed and `float_amount` exists on the leg
- the dual-curve pricer should discount that concrete stored amount
- not rebuild the amount again from `coupon * accrual * notional`

This fix is now in the pricer and covered by regression tests, but it was **not** the main source of the large `t0` gap on `Swap_20`.

### Coupons remain alive until payment date, not accrual start

Another major IRS exposure bug appeared after the `t0` PV anchor was already back in line:

- PV looked close to ORE
- `EPE` was too low after `t=0`
- `ENE` was too high
- CVA was low and DVA was high

The cause was in:

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py)
  - `swap_npv_from_ore_legs_dual_curve()`

The pricer was treating coupons as dead once accrual had started. For swap valuation that is wrong: a coupon remains part of the NPV until its payment date. Dropping it at accrual start can leave `t0` PV looking acceptable while still materially understating future exposure.

Fix:

- fixed coupons are live while `fixed_pay_time > t`
- floating coupons are live while `float_pay_time > t`

This was the decisive CVA-parity fix for the notebook/file-based IRS cases. After it:

- `flat_EUR_10Y_B` moved to about:
  - Python CVA `34,446`
  - ORE CVA `34,481`
- `flat_USD_5Y_A` moved to about:
  - Python CVA `8,141`
  - ORE CVA `8,156`

If PV is fine but CVA is still structurally low and DVA high, inspect coupon life through payment date before changing model or credit formulas.

### The leg time axis must use the ORE model day counter, not an ad hoc basis

The leg loaders originally turned dates into times using Actual/Actual-style fractions, even when the ORE run was using `A365F`.

This was corrected in:

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py)
  - `load_ore_legs_from_flows(..., time_day_counter=...)`
  - `load_swap_legs_from_portfolio(..., time_day_counter=...)`

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/ore_snapshot.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/ore_snapshot.py)
  - snapshot now passes `model_day_counter` into the leg loaders

This was a real fix, but on the main `ore_measure_lgm` case it was not the dominant driver of the remaining gap once cashflows were already aligned.

### ORE-style XVA aggregation is now implemented; it fixed the formula side, not the exposure shape side

The Python helper path was extended so that it can work in an ORE-style mode:

- numeraire-deflate NPVs first
- aggregate EPE/ENE from deflated NPVs
- skip the extra discount factor in the CVA/DVA/FVA helper in that mode

Files:

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/irs_xva_utils.py)
  - `deflate_lgm_npv_paths()`
  - `compute_xva_from_exposure_profile(..., exposure_discounting=...)`

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/example_ore_snapshot.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/example_ore_snapshot.py)
  - `--xva-mode classic|ore`

Key lesson:

- once fed **ORE exposure** and using `--xva-mode ore`, the Python CVA formula matches ORE essentially exactly
- therefore the remaining gap is not the CVA formula anymore
- it is the underlying Python exposure profile

This is an important diagnostic milestone:

- formula side: mostly solved
- exposure-path side: still open

### Current status at the end of this debugging sequence

For `Examples/Exposure/Input/ore_measure_lgm.xml` using `--xva-mode ore`:

- ORE `t0` NPV:
  - `602.49`
- Python `t0` NPV:
  - `464.39`

- ORE CVA:
  - `42,142.15`
- Python CVA:
  - `46,964.76`

- ORE DVA:
  - `61,224.53`
- Python DVA:
  - `35,016.39`

Interpretation:

- raw `t0` trade pricing is now much closer than before
- CVA formula integration is structurally aligned in ORE mode
- the remaining gap is genuinely dynamic / pathwise:
  - EPE still too high
  - ENE still too low

### What to try next if parity is still off

After this sequence, do **not** keep hacking cashflow dates first. The better next steps are:

1. Compare pathwise / datewise swap PV decomposition beyond `t=0`
   - fixed leg PV
   - float leg PV
   - total
   - on a few early exposure dates where EPE is too high

2. Investigate pricing curve role for this specific ORE run
   - pricing config vs curves analytic config vs XVA config
   - do not assume the same `discount_column` is correct for both direct pricing parity and XVA parity

3. Investigate the sign / perspective of the whole trade in the dynamic path
   - if `t0` sign is now right-ish but later EE is still biased, look at where sign/perspective is applied in the pathwise valuation and XVA aggregation

4. Only after that, revisit closeout / MPOR / own-credit DVA details
   - those matter, but they were not the first-order blocker in this thread

### Commands and verification that were stable during this work

Regression test command:

```bash
PYTHONPATH=Tools/PythonOreRunner python3 -m pytest Tools/PythonOreRunner/tests/test_lgm.py -q
```

Expected after the latest fixes:

```text
21 passed
```

Useful ad hoc diagnostics added during the work:

- cashflow triptych:
  - [`/Users/gordonlee/Documents/PythonOreRunner/scripts/diagnostics/diagnose_cashflow_triptych.py`](/Users/gordonlee/Documents/PythonOreRunner/scripts/diagnostics/diagnose_cashflow_triptych.py)
- leg bias diagnostic:
  - [`/Users/gordonlee/Documents/PythonOreRunner/scripts/diagnostics/diagnose_ore_snapshot_leg_bias.py`](/Users/gordonlee/Documents/PythonOreRunner/scripts/diagnostics/diagnose_ore_snapshot_leg_bias.py)
- snapshot parity example:
  - [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/example_ore_snapshot.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/example_ore_snapshot.py)

### Short takeaway

The “battle” was won in layers:

1. distinguish model time from report time
2. distinguish raw pricing from XVA-state exposure
3. trust `flows.csv` as the cashflow oracle
4. verify the floating leg coupon-by-coupon
5. discover that the huge `t0` gap was actually the node-tenor interpolation shortcut at `t=0`

That last point is the one most likely to save time in future work. If swap parity is wildly wrong at `t=0`, inspect the node-interpolation shortcut before blaming fixings, curves, or model parameters.

### Notebook 5 aligned benchmark should be `flat_EUR_10Y_B`

The notebook-series parity workflow regressed for two different reasons:

1. huge negative PV from missing `fixed_start_time` / `fixed_end_time`
2. large CVA gap from coupons being dropped at accrual start instead of payment date

After those fixes, the aligned benchmark case for notebook 5 should be:

- `flat_EUR_10Y_B`

That case is now good enough for the headline notebook parity section:

- PV essentially exact
- CVA within about `0.1%`

If notebook 5 suddenly looks bad again:

1. verify the notebook is still using `flat_EUR_10Y_B`
2. verify the standalone `ore_snapshot` / `PythonLgmAdapter` path first
3. only then debug notebook presentation code

---

## Regression Pack and Multicurrency Follow-ups

### `market.pricing` is the right first choice for live native trade discount curves

For live native multicurrency IRS cases, the most important curve-role correction after the EUR work was:

- use `Markets/pricing` from `ore.xml` for trade repricing discount curves
- keep the XVA comparator / funding logic separate

Using the `curves` analytic configuration as the pricing-curve source was acceptable on the `measure_lgm` family, but it broke the live non-EUR multicurrency cases badly because it routed discounting through cross-currency handles like:

- `Yield/USD/USD-IN-EUR`
- `Yield/GBP/GBP-IN-EUR`

That produced large live native gaps, for example on `flat_USD_5Y_A`:

- before the fix:
  - Python PV about `299,759`
  - ORE PV about `420,394`
  - Python CVA about `6,169`
  - ORE CVA about `8,156`

- after switching native trade repricing to `market.pricing`:
  - Python PV about `420,392`
  - ORE PV about `420,394`
  - Python CVA about `8,124`
  - ORE CVA about `8,156`

So for native live trade valuation:

- default rule: prefer `market.pricing`
- do **not** assume the `curves` analytic config is the right trade repricing source

### Cross-currency discount handles in `todaysmarket.xml` can collapse to bare-currency columns in `curves.csv`

The multicurrency benchmark cases exposed a non-obvious ORE artifact convention:

- `todaysmarket.xml` may use handles such as:
  - `Yield/USD/USD-IN-EUR`
  - `Yield/GBP/GBP-IN-EUR`

- but `curves.csv` may only expose columns such as:
  - `USD`
  - `GBP`

There may be no explicit `YieldCurve name="USD-IN-EUR"` or `GBP-IN-EUR` entry to bridge them.

This required a shared fallback in [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/ore_snapshot.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/py_ore_tools/ore_snapshot.py):

- when `_handle_to_curve_name()` sees a handle of the form:
  - `Yield/<CCY>/<CCY>-IN-...`
- it now falls back to the bare currency column:
  - `<CCY>`

This change was necessary to get live native USD/GBP cases onto ORE output curves at all.

### Live native multicurrency cases are now good enough to replace stale artifact summaries for CVA

The regression pack originally relied on old `artifact_summary` JSON files for many USD/GBP benchmark cases.
That became misleading once the live native path improved.

The multicurrency IRS cases in the regression pack now run live as `native_ore_case` for:

- `flat_USD_5Y_A`
- `flat_USD_5Y_B`
- `flat_USD_10Y_A`
- `flat_USD_10Y_B`
- `full_USD_5Y_A`
- `full_USD_10Y_A`
- `flat_GBP_5Y_A`
- `flat_GBP_5Y_B`
- `flat_GBP_10Y_A`
- `flat_GBP_10Y_B`
- `full_GBP_5Y_A`
- `full_GBP_10Y_A`

Observed live CVA parity after the fixes:

- `flat_USD_5Y_A`: about `0.39%`
- `flat_USD_10Y_B`: about `1.11%`
- `flat_GBP_5Y_A`: about `0.44%`
- `flat_GBP_10Y_B`: about `1.14%`

This is much better than the stale artifact-based diffs that were previously showing `~7-8%`.

### Pass/fail in the regression pack must respect the case’s configured metric scope

One subtle but important clean-up:

- many multicurrency benchmark cases request only `CVA`
- ORE output files may still contain other columns or zeros for `DVA/FBA/FCA`
- Python may be able to compute those metrics anyway if the runtime is forced into a broader analytic set

If the regression pack blindly computes relative diffs on every metric, the summary becomes noisy and misleading.

The correct rule for the pack is:

- gate pass/fail only on metrics the source case actually requested
- keep other metrics as informational only, or mark them `n/a`

This is now how [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/run_xva_regression_pack.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/run_xva_regression_pack.py) behaves.

### Clean regression outputs matter; `n/a` is better than fake precision

The regression pack was updated to clean inactive metrics instead of emitting misleading giant relative diffs or zero placeholders.

Current outputs:

- machine-readable:
  - [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/xva_regression_pack_latest/summary.json`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/xva_regression_pack_latest/summary.json)
  - [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/xva_regression_pack_latest/summary.csv`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/xva_regression_pack_latest/summary.csv)
- human-readable:
  - [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/xva_regression_pack_latest/summary_pretty.md`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/xva_regression_pack_latest/summary_pretty.md)

Meaningful presentation rules now are:

- inactive metric in JSON/CSV: `null`
- inactive metric in console / markdown: `n/a`
- pack pass/fail: based only on active metrics

### Current clean headline for the regression pack

With the latest live/native and reporting fixes at `1000` paths, seed `42`:

- cases run: `23`
- hard failures: `0`
- passes: `23`

This is the current best top-level “is parity in shape?” sanity signal in this repo.

### Current residual parity gap is mostly DVA, not CVA

After the coupon-life-through-payment-date fix, the big systematic CVA problem on the vanilla notebook/file cases largely collapsed. The remaining visible gap on the representative checked cases is now mostly DVA.

Current rough status:

- `flat_EUR_10Y_B`
  - CVA close enough for notebook parity
  - DVA still a few-to-several percent high
- `flat_EUR_5Y_A`
  - CVA back near ORE
  - DVA still needs more caution
- `flat_USD_5Y_A`
  - CVA back near ORE
  - DVA still not as tight as CVA

So if you see:

- PV close
- CVA close
- DVA still off

do **not** reopen the IRS exposure-dynamics debugging from scratch. The next likely area is own-credit / DVA aggregation detail, not the swap exposure core.

### Modern ORE XVA sensitivities are a separate analytic, not part of plain `xva`

If you want factor-by-factor XVA sensitivities from ORE, plain `xva` is not enough.

- the modern output comes from the separate `xvaSensitivity` analytic
- the key report files are:
  - `xva_zero_sensitivity_<metric>.csv`
  - optionally `xva_par_sensitivity_<metric>.csv`
- for the demo cases, ORE may write them under:
  - `Output/xva_sensitivity/`
  - not only directly under `Output/`

If a case only runs `xva`, you may still see legacy files like `cva_sensitivity_nettingset_*.csv`, but that is not the same format and should not be treated as modern factor parity input.

### Native sensitivity runs should go through runtime preparation, not ad hoc config params

The native Python sensitivity path now has a runtime-side setup hook in [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/runtime.py):

- `XVAEngine.prepare_sensitivity_snapshot(...)`

Use that instead of manually stuffing sensitivity-specific values into `snapshot.config.params` from compare code.

Current runtime-owned knobs:

- `python.curve_fit_mode`
- `python.use_ore_output_curves`
- `python.curve_node_shocks`
- `python.frozen_float_spreads`

This matters because the comparator should decide **what** to shock, while runtime should own **how** shocked reruns are prepared.

### Freeze base floating spreads across shocked reruns or forward sensitivities get washed out

For IRS sensitivities, recalibrating `float_spread` on every bumped run can neutralize the forward shock by preserving the all-in coupon.

The correct native rule for this codebase is:

- calibrate base floating spreads once
- freeze them across the shocked reruns

That is now part of the runtime-side sensitivity preparation path. If forward sensitivity is suspiciously near zero while the fitted forward curve did move, check spread freezing before changing pricing formulas.

### Rate parity is credible; credit parity is still approximate

Current useful rule for the ore-snapshot compare path:

- compare rate factors (`DiscountCurve/...`, `IndexCurve/...`)
- do **not** claim strong parity on credit factors yet

Reason:

- ORE `xvaSensitivity` shocks simulated `SurvivalProbability/...` factors via its scenario-market machinery
- the Python snapshot path only has an approximate today-curve reconstruction for credit

So the compare tool now treats credit factors as unsupported for parity output and prints them separately. This is more honest than reporting misleading credit diffs.

### For this demo case, the remaining rate mismatch is concentrated in short-end forward coverage

On `flat_EUR_5Y_A`, after the runtime/sensitivity fixes:

- `zero:EUR:5Y` is close
- `fwd:EUR:6M:5Y` is in the right sign and ballpark
- `zero:EUR:1Y` still has a noticeable residual gap
- `fwd:EUR:6M:1Y` is still unmatched in the compare output

So if rate parity regresses again, first check:

1. the fitted curve actually moved for the shocked node
2. the trade kept its real floating index tenor through the runtime path
3. frozen floating spreads were reused on the bumped run
4. the compare is using ORE bump-change convention, not derivative-per-shift scaling

---

## SWIG-first curve parity fitter

### What was added

There is now a dedicated SWIG-first curve parity service under:

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/ore_curve_fit_parity/service.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/ore_curve_fit_parity/service.py)

and the trace layer in:

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/ore_curve_fit_parity/curve_trace.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/ore_curve_fit_parity/curve_trace.py)

was extended to expose:

- generic trace-by-handle for yield curves
- richer curve config metadata:
  - `bootstrap_config`
  - `pillar_choice`
  - `extrapolation`
- native ORE calibration nodes extracted from `todaysmarketcalibration.csv`
- handle listing from `todaysmarket.xml`

The public Python surface now includes:

- `CurveBuildRequest`
- `CurveBuildResult`
- `CurveTrace`
- `CurveComparison`
- `build_curves_from_ore_inputs(...)`
- `trace_curve(...)`
- `compare_python_vs_ore(...)`
- `swig_module_available()`

### The intended architecture

For “as close as possible to ORE”, the primary path should be:

1. clone or point at a real ORE case
2. run ORE through ORE-SWIG
3. read the generated:
   - `curves.csv`
   - `todaysmarketcalibration.csv`
4. trace and compare from those native outputs

Do **not** start by reimplementing ORE helper construction in pure Python if the goal is parity.

The current rule is:

- `swig` path = parity baseline
- `python` path = diagnostic comparator only

### How the SWIG-first build works

The build service currently:

1. takes an `ore.xml`
2. clones the whole case to a temp run directory
3. rewrites absolute case-local paths in `ore.xml` to point at the cloned case
4. runs ORE via:
   - `ORE.Parameters().fromFile(...)`
   - `ORE.OREApp(...)`
5. reads the generated output artifacts
6. traces selected yield curves from the fresh run

This is important because it avoids mutating the checked-in parity artifacts while still using the exact ORE builder path.

### Demo case that was verified

The cleanest verified demo is:

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_USD_5Y_B/Input/ore.xml`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/parity_artifacts/multiccy_benchmark_final/cases/flat_USD_5Y_B/Input/ore.xml)

Using:

- `currency='USD'`
- `index_name='USD-LIBOR-6M'`

the SWIG build produced:

- `Yield/USD/USD3M`
- `Yield/USD/USD6M`

with:

- `USD3M` dependencies:
  - `USD1D`
- `USD6M` dependencies:
  - `USD1D`
  - `USD3M`

### Clean demo script

A clean demo of the Python diagnostic fitter now exists at:

- [`/Users/gordonlee/Documents/PythonOreRunner/scripts/checks/demo_python_curve_fitter.py`](/Users/gordonlee/Documents/PythonOreRunner/scripts/checks/demo_python_curve_fitter.py)

Run it with:

```bash
cd /Users/gordonlee/Documents/PythonOreRunner
python3 scripts/checks/demo_python_curve_fitter.py --points 6 --quotes-per-segment 4
```

What it shows:

- input instruments grouped by segment
  - deposits
  - FRAs
  - swaps
- market quotes for each instrument
- mapped ORE pillar outputs
  - time
  - discount factor
  - zero rate
- fitted Python outputs on the ORE report grid
- ORE vs Python error

### Lessons learned

#### 1. The right SWIG module is usually `ORE`, not the repo namespace `OREAnalytics`

In this repo, importing `OREAnalytics` by itself can just resolve the source namespace package directory and **not** the SWIG runtime module.

For actual live runs:

- bootstrap `sys.path` with:
  - `ORE-SWIG/`
  - or `ORE-SWIG/build/lib.*`
- then import:
  - `ORE`

The service now probes module candidates, but `ORE` is the one that actually exposed `Parameters` and `OREApp` in the verified run.

#### 2. `Parameters.fromFile(ore.xml)` is the cleanest parity entry point

If the goal is curve parity against ORE, prefer:

- `Parameters.fromFile(...)`

over trying to synthesize `InputParameters` manually from Python buffers.

Reason:

- it uses the same file-driven path ORE cases already use
- it avoids silent divergence in setup fields
- it makes the parity service behave like a real ORE case run

#### 3. Clone the case first; do not run directly in checked-in parity artifacts

The temp-case clone is not optional hygiene; it is the safe default.

Reason:

- ORE writes `Output/*`
- some runs can rewrite or append logs and reports
- parity tooling should not mutate the checked-in benchmark cases

#### 4. Do not rewrite relative paths like `log.txt`

One bug hit during implementation:

- relative `log.txt` in `ore.xml` was rewritten to an absolute path
- ORE then tried to resolve it under `Output/` and produced a broken doubled path

Rule:

- rewrite absolute case-local paths
- leave relative filenames like `log.txt` alone

#### 5. `curves.csv` and `todaysmarketcalibration.csv` remain the real external parity oracles

Even with a programmatic SWIG run, the parity-grade outputs are still:

- `curves.csv`
- `todaysmarketcalibration.csv`

Use them as the external truth for:

- grid values
- pillar values
- day counter
- zero / forward / discount outputs

#### 6. The current Python fitter is interpolation parity, not bootstrap parity

The Python comparator currently rebuilds the curve from native ORE nodes using:

- `Discount`
- `LogLinear`

and compares that to ORE report-grid values.

This is useful and credible, but it is not the same as claiming:

- Python helper construction matches ORE helper construction
- Python bootstrap numerics match ORE bootstrap numerics

Do not over-claim what the Python side is doing today.

#### 7. Tiny grid differences on the order of `1e-9` are normal here

On the verified USD3M demo case, the Python diagnostic comparator matched ORE’s report grid with:

- max abs error around `4.8e-09`
- max rel error around `5.2e-09`

That is good enough for the current interpolation-parity diagnostic.

Do not set default demo tolerances to `1e-12` and then misread harmless noise as a structural failure.

#### 8. Use one benchmark with mixed instrument types for the cleanest fitter demo

The best “show me the fitter” case so far is the USD3M curve in `flat_USD_5Y_B` because it contains:

- deposits
- FRAs
- swaps

in one curve.

That makes it much easier to show:

- inputs by instrument type
- fitted pillar outputs
- final report-grid comparison

without switching cases.

#### 9. Multi-ccy FX XVA needs one shared portfolio FX simulation and MPOR-aware settlement handling

Do not price each FX forward on its own standalone two-ccy simulation if the target is netting-set XVA parity.

On the large-FX benchmark, three structural fixes mattered:

- use one shared multi-ccy FX scenario set across the whole FX-forward book
- only make FX pairs stochastic if they are actually modeled in the loaded ORE `simulation.xml`
- keep FX forward maturity payoffs alive when maturity falls inside the MPOR closeout window

Without those, PV can look acceptable while CVA/DVA/FVA are overstated by an order of magnitude on short-dated offsetting books.

### 10. A default `--paths` mismatch can look like a model/XVA bug when it is just a CLI bug

One of the last shipped-example parity failures was not a curve, exposure, or
model issue at all. The CLI defaulted `--paths` to `500`, while some ORE cases
were configured with `Samples=1000`.

That created a subtle trap:

- explicit parity investigations at `1000 vs 1000` could pass
- but a plain default CLI run of the same example could still fail parity
- the summary would show:
  - `python_paths = 500`
  - `ore_samples = 1000`
  - `sample_count_mismatch = true`

Rule:

- do not hardcode a CLI default path count for parity-mode runs
- if the user does not pass `--paths`, inherit the sample count from the loaded
  ORE case (`snap.n_samples`)
- only treat a path mismatch as meaningful when it was explicitly requested

This was the final fix needed to make the shipped example sweep go fully green.

### 11. Non-LGM cases must not crash calibration matching

`resolve_calibration_xml_path()` originally assumed every simulation file had an
LGM node and tried to build an LGM signature unconditionally.

That is wrong for cases like:

- HW2F examples
- any future non-LGM simulation config

Failure mode:

- the case crashes while trying to match calibration metadata
- the crash is misleading because the real desired behavior is usually a clean
  fallback to ORE reference mode

Rule:

- calibration matching must be tolerant of non-LGM simulation files
- if a simulation file has no usable LGM signature, treat it as "no calibration
  match available", not as a fatal error

This is especially important when broad example sweeps include mixed model families.

### 12. Cloned reruns can silently lose calibration fallback if matching uses absolute paths

Another subtle parity trap appeared when native ORE cases were copied into `/tmp`
for fresh reruns.

The bad version of the calibration resolver compared absolute shared-input paths.
That meant:

- the repo example and the temp-cloned rerun were logically the same case
- but they no longer matched for calibration fallback
- Python then silently fell back to flat simulation params such as:
  - `alpha = 1%`
  - `kappa = 3%`

Observed consequence:

- PV could still look good
- but raw cube variance, EEPE, and CVA would blow out again
- it looked like a fresh state-dynamics bug even though the real issue was wrong
  parameter provenance

Rule:

- calibration fallback matching must use stable example-relative resource ids or
  another clone-stable identity
- do not use absolute filesystem paths as the parity identity for shared inputs

If a temp-rerun parity result suddenly regresses after looking healthy in-repo,
check calibration provenance before touching the simulator or XVA formulas.
