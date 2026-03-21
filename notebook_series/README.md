# Notebook Series

This folder contains the main Python + ORE notebook track plus Python-only companion notebooks generated from
[`build_series.py`](/Users/gordonlee/Documents/PythonOreRunner/notebook_series/build_series.py).

## Start Here

- `05_1_python_only_workflow.ipynb`: canonical Python-first, in-memory workflow
- `01_python_to_ore_swig_dataclasses.ipynb`: object model and mapping contract
- `example_ore_snapshot.py`: ORE XML -> Python snapshot bridge

## Main Track

1. `01_python_to_ore_swig_dataclasses.ipynb`
2. `02_ore_snapshot_capabilities.ipynb`
3. `03_curve_calibration_and_lgm_params.ipynb`
4. `04_python_lgm_xva_model.ipynb`
5. `05_joint_python_and_ore_workflow.ipynb`
6. `06_bermudan_python_vs_ore.ipynb`

## Python-Only Companions

1. `03_1_python_curve_calibration_and_lgm_params.ipynb`
2. `04_1_python_lgm_xva_model.ipynb`
3. `05_1_python_only_workflow.ipynb`
4. `06_1_python_bermudan_swaption_pricing.ipynb`
5. `06_2_python_swap_fxforward_capfloor_pricing.ipynb`

## Notes

- `03_1` covers programmatic curve fitting, rates-futures convexity, and the Python-native curve/LGM setup.
- `04_1` now includes the larger 256-trade multi-ccy portfolio example and the numpy vs torch backend comparison.
- `05_1` is the primary native walkthrough and should be the first notebook for users focused on Python-only XVA.
- `series_helpers.py` provides the shared plotting, loader, and benchmark helpers used across the notebooks.
- The checked-in `.py` files are notebook sources; rebuild the `.ipynb` outputs after editing the builder or those sources.

To regenerate the notebooks after editing the builder:

```bash
python3 notebook_series/build_series.py
```
