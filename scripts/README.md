## Scripts

Ad hoc utilities that do not belong in the tested library surface live here instead of the repo root.

### Layout

- `scripts/checks/`: small notebook or model sanity checks
- `scripts/compare/`: ORE-vs-Python comparison runners that generate parity artifacts
- `scripts/diagnostics/`: investigation tools for sensitivities, cashflows, and snapshot behavior
- `scripts/dumps/`: one-off data extractors and artifact generators
- `scripts/plots/`: plotting helpers

### Relevance

- Still active: `compare_bermudan_ore_sensitivities.py`, `dump_ore_lgm_rng_parity_case.py`, and the `diagnose_*` tools referenced by the notebook helpers or local ORE skill
- Still usable but ad hoc: `check_demo_fx_examples.py`, `check_demo_fx_profiles_xva.py`, `check_lgm_irs_xva_calc.py`, `compare_bermudan_singlecurve_sensitivity.py`, `dump_ore_discount_factors.py`, `dump_ore_input_validation.py`, `plot_ore_snapshot_epe_ene_semianalytic.py`, `strict_native_vs_py_lgm_example.py`
- Benchmarks are separate on purpose and remain under `py_ore_tools/benchmarks/`

Run these from the repo root, for example:

```bash
python scripts/diagnostics/diagnose_sensitivity_bucket.py --factor zero:EUR:5Y
python scripts/compare/compare_bermudan_ore_sensitivities.py --case-name berm_200bp
```
