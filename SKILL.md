---
description: Use when working with ORE (Open Risk Engine): configuring AMC/AMC-CG runs, benchmarking, number matching/parity, performance optimization, build setup, SWIG/native XVA, or debugging ORE output. Invoke when the user asks about ORE configuration, XVA calculations, AMC-CG routing, or troubleshooting ORE runs.
---

# ORE Expert Context

This skill provides hard-won, non-obvious knowledge about ORE (Open Risk Engine) accumulated from hands-on build, benchmark, and parity experiments on this codebase. Apply this knowledge whenever working with ORE configs, code, or debugging.

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
- Surprisingly fast run → likely failed early, not a genuine speedup

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
- **Oracle / tests**: `dump_ore_lgm_rng_parity_case.py`, artifact `parity_artifacts/lgm_rng_alignment/mt_seed_42_constant.json`, and tests in `tests/test_lgm.py` validate RNG alignment.

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

Once a coupon is fixed on a path, lock that realized fixing and reuse it at later valuation dates. Reprojecting post-fixing coupons is a first-order parity bug.

**Implemented in core helpers**:
- Added `compute_realized_float_coupons(...)` to `py_ore_tools.irs_xva_utils`.
- The helper computes pathwise `forward + spread` at each fixing time using the same discount/forward mapping used by `swap_npv_from_ore_legs_dual_curve`.
- `example_ore_snapshot.py` now simulates on `exposure_times ∪ fixing_times` and passes `realized_float_coupon=...` into pricing.

**Observed effect (flat_EUR_5Y_A, 2000 paths, seed 42)**:
- CVA rel diff improved from about `17.9%` to about `4.4%`.
- EPE profile parity improved materially (`p95` roughly from `26.9%` to `4.7%`).

**Native-interface status**:
- `native_xva_interface.runtime.PythonLgmAdapter` still needs the same treatment fully wired for its generated-leg path to reach OreSnapshot-level parity.
- Keep fixing-injection in the next-step list for native adapter parity.

### Portfolio-loaded floating-leg index tenor must be propagated

If legs are loaded from portfolio XML but `float_index_tenor` is missing, fallback forwarding selection can choose the wrong curve bucket and distort floating-leg PV/exposure.

**Implemented**:
- In `load_swap_legs_from_portfolio(...)` (`py_ore_tools.irs_xva_utils`), floating legs now carry:
  - `float_index` (e.g. `EUR-EURIBOR-6M`)
  - `float_index_tenor` (e.g. `6M`)

This is critical for both file-based parity and native adapter runs that consume portfolio-derived legs.

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

### Two native packages exist; capabilities differ

- `Tools/PythonOreRunner/native_xva_interface` includes `PythonLgmAdapter` (Python-LGM parity path).
- `PythonIntegration/native_xva_interface` currently exports toy/SWIG adapters but not `PythonLgmAdapter`.

When a user asks for Python-LGM parity via "native interface", default to the `Tools/PythonOreRunner` package unless/until parity features are ported.

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

- [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/diagnose_cashflow_triptych.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/diagnose_cashflow_triptych.py)
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
  - [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/diagnose_cashflow_triptych.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/diagnose_cashflow_triptych.py)
- leg bias diagnostic:
  - [`/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/diagnose_ore_snapshot_leg_bias.py`](/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/diagnose_ore_snapshot_leg_bias.py)
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
