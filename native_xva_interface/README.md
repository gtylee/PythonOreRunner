# Native XVA Interface (ORE Add-On)

This package is an add-on layer on top of ORE, not a replacement for ORE.

It exists for two main goals:

1. Add a more native Python layer for using ORE workflows.
2. Provide a lightweight Python LGM model for faster iteration and comparison.

## What this add-on provides

- Python dataclass models for market, portfolio, netting, collateral, and config.
- Loaders that ingest standard ORE input folders into Python objects.
- Mapping from Python objects into runtime inputs (including XML buffers when needed).
- Runtime adapters:
  - `ORESwigAdapter` to run through ORE-SWIG.
  - `PythonLgmAdapter` for a lighter-weight Python LGM path.
- Typed result objects and parity utilities for comparing runs.

## Primary API

```python
from native_xva_interface import XVALoader, XVAEngine

snapshot = XVALoader.from_files("Examples/XvaRisk/Input", ore_file="ore_stress_classic.xml")
engine = XVAEngine()
result = engine.create_session(snapshot).run(return_cubes=False)
```

## Run tests

From repo root:

```bash
PYTHONPATH=Tools/PythonOreRunner python3 -m pytest Tools/PythonOreRunner/native_xva_interface/tests -q
```

## Demos and notebooks

- Demos: `Tools/PythonOreRunner/native_xva_interface/demos`
- Notebooks: `Tools/PythonOreRunner/native_xva_interface/notebooks`
