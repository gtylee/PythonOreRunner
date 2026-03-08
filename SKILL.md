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

---

## Python LGM vs ORE Parity Notes

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

### IRS portfolio payload completeness is mandatory for ORE parity

A fixed-only IRS payload is insufficient for parity. Ensure generated trade XML includes:
- both fixed and floating legs
- schedule blocks and core leg fields

Incomplete payloads can run but produce misleadingly weak or zero-like outputs.

### `pv_total` is not always available in XVA-focused ORE runs

Depending on analytic configuration, ORE may not emit an `npv` report. Avoid treating missing/zero `pv_total` as immediate pricing failure without checking report configuration/output set.

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

`flows.csv` seems closer to "what ORE actually paid" but `trade` was the winning source in the calibrated baseline. Never assume one is better without testing both.

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

### Build fixed-parameter parity first; calibration can hide bugs

- `fixed` lane: fixed alpha/kappa for strict implementation parity
- `calibrated` lane: only after the fixed lane is green

Calibration differences can mask convention bugs.

### Closeout / MPOR is second-order until PV parity is solved

A simplified `V(t + MPOR)` closeout mapping changes CVA but does not fix the main gap if valuation logic itself is still wrong. Prioritize deterministic PV and exposure conventions first.

### Put parity logic in `.py` modules, not notebooks

Notebook-local pricing logic drifts from module implementations quickly. Keep notebooks thin; put valuation, exposure, and parity logic in `.py` modules.

### Highest-value recurring parity diagnostics

- Per-date `ORE_EPE` vs `PY_EPE`
- CVA term table
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
| `runtimes.csv` sum ≠ wall-clock | Normal — gap is framework/logging overhead |
| Native loader succeeds | Still test SWIG runtime step separately |
| pytest import error | Set `PYTHONPATH=PythonIntegration` |
