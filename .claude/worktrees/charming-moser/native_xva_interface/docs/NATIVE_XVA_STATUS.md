# Native XVA Interface Status (March 2026)

## What Is Implemented

1. Dataclass-first domain model:
- Market, fixings, portfolio/trades, netting, collateral, config.
- Runtime config dataclasses for pricing engine, todays market, curve config, simulation, conventions, counterparties.
- XVA-specific runtime dataclasses:
  - `XVAAnalyticConfig`
  - `CreditSimulationConfig`
  - `CreditEntityConfig`

2. Mapper + runtime API:
- Dataclasses map to in-memory ORE-SWIG `InputParameters`.
- Session API supports incremental updates:
  - `update_market(...)`
  - `update_portfolio(...)`
  - `update_config(...)`
- Report extraction parses ORE report layouts and returns typed `XVAResult`.

3. Test coverage:
- Unit/integration tests for dataclasses, loaders, mapper, runtime, results, parity harness.
- Current test status: `18 passed`.

## What Works Reliably Today

1. **Hybrid production path (recommended now)**:
- Programmatic market/portfolio/fixings + file-derived XVA runtime buffers from known-good ORE examples.
- This path produces non-zero XVA and full exposure/XVA reports.

2. **Pure programmatic path with embedded templates**:
- No runtime file loading.
- Market/fixings and required XML templates are embedded in Python and materialized in-memory.
- This now produces non-zero XVA (for example with `num_paths=100`).

3. **Strict pure-native runtime config generation (now non-zero)**:
- Strict mode now runs through SWIG with non-zero XVA and emits `xva` report.
- In strict mode, the mapper injects an internal in-memory compatibility runtime template set (no runtime file reads, no user XML files).

## Current Strict-Mode Caveat

- Strict mode now achieves non-zero XVA by using internal compatibility templates for runtime XML blocks.
- This is still fully in-memory at runtime, but not yet a fully field-by-field dataclass-generated runtime XML implementation.

## Python That Works Now

Use:

- `/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/demos/demo_working_xva.py`
- `/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/demos/run_pure_programmatic_embedded_templates.py`
- `/Users/gordonlee/Documents/Engine/Tools/PythonOreRunner/native_xva_interface/demos/run_alignment_matrix.py`

`demo_working_xva.py`:
- builds market + portfolio programmatically in Python dataclasses,
- imports the known-good XVA config buffers from `Examples/XvaRisk/Input/ore_stress_classic.xml`,
- runs ORE-SWIG and prints non-zero XVA metrics and report names.

`run_pure_programmatic_embedded_templates.py`:
- builds market/fixings/portfolio/netting/config fully in Python,
- uses embedded XML template buffers (packaged in this module),
- runs ORE-SWIG with non-zero XVA output.

`run_alignment_matrix.py`:
- runs all 3 modes side-by-side:
  - strict pure-native runtime generation,
  - pure programmatic embedded-template mode,
  - hybrid preset mode,
- prints report counts, XVA totals, and a concise alignment verdict.

## Next Technical Steps

1. Complete full dataclass-native runtime generation parity:
- replace compatibility-template wiring in strict mode with fully generated XML from dataclass fields while preserving non-zero XVA parity.

2. Keep hard integration gates:
- strict, embedded-template, and hybrid modes must stay non-zero.

3. Keep bridge/fallback modes:
- hybrid and embedded-template modes remain stable runtime options while strict-native parity is finalized.
