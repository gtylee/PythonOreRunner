**Profile Summary**

Compared two end-to-end `ore_snapshot_cli --price --xva` runs:

- legacy USD basis book: `300` trades
  - `USD-LIBOR-3M vs USD-LIBOR-6M`
  - `USD-LIBOR-3M vs USD-SIFMA`
  - `USD-FedFunds vs USD-LIBOR-3M`
- SOFR USD basis book: `200` trades
  - `USD-SOFR-3M vs USD-LIBOR-3M`
  - `USD-SOFR-3M vs USD-SIFMA`

Commands used:

```bash
python3 -m cProfile -o /tmp/usd_basis.prof /Users/gordonlee/Documents/PythonOreRunner/example_ore_snapshot_usd_basis_swaps.py --count-per-type 100 --include-fedfunds --overwrite
python3 -m cProfile -o /tmp/usd_sofr_basis.prof /Users/gordonlee/Documents/PythonOreRunner/example_ore_snapshot_usd_sofr_basis_swaps.py --count-per-type 100 --overwrite
```

Raw profile dumps:

- [legacy_basis_300t.pstats.txt](/Users/gordonlee/Documents/PythonOreRunner/legacy_basis_300t.pstats.txt)
- [sofr_basis_200t.pstats.txt](/Users/gordonlee/Documents/PythonOreRunner/sofr_basis_200t.pstats.txt)

**Wall Clock**

- Legacy basis: `1.775249 sec`
- SOFR basis: `7.440372 sec`
- Same SOFR case with `--lgm-param-source simulation_xml`: `0.657659 sec`

**Primary Cause**

The dominant delta is runtime LGM calibration inside [`calibrate_lgm_params_via_ore()`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore/io/ore_snapshot.py#L775), reached from [`resolve_lgm_params()`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore/io/ore_snapshot.py#L889).

Top cumulative frames:

- Legacy basis:
  - `resolve_lgm_params`: `1.324s`
  - `calibrate_lgm_params_via_ore`: `1.015s`
  - `subprocess.run(... ore ...)`: `1.011s`
- SOFR basis:
  - `resolve_lgm_params`: `6.709s`
  - `calibrate_lgm_params_via_ore`: `6.400s`
  - `subprocess.run(... ore ...)`: `6.397s`

So the extra `~5.7s` is almost entirely the ORE runtime calibration subprocess.

In the SOFR case, that calibration subprocess is not just expensive, it also fails internally. The preserved ORE log shows:

- `Error while building IrModel LGM for qualifier 'USD'`
- `did not find swaption index bases for key 'USD'`
- final ORE message: `Failed to run analytics CALIBRATION`

That means `auto` mode was repeatedly paying for a calibration attempt that produced no `calibration.xml`, then falling back to simulation XML params.

**Secondary Cause**

The SOFR case points at the much larger `Examples/Products/Input` bundle:

- Legacy:
  - `curveconfig.xml`: `131,522` bytes, `3,421` lines
  - `market_20160205_flat.txt`: `439,835` bytes, `7,937` lines
- SOFR:
  - `curveconfig.xml`: `750,993` bytes, `19,746` lines
  - `marketdata.csv`: `3,923,586` bytes, `52,324` lines

This adds some loader cost, visible in [`_load_ore_csv_keys_by_date()`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore/io/ore_snapshot.py#L1919), but it is not the main slowdown.

**Conclusion**

The SOFR book is not slow because SOFR basis trades are intrinsically expensive in the Python XVA path. It is slow because this setup triggers a heavy runtime ORE calibration against a much larger market/config bundle.

**Implemented Mitigation**

[`calibrate_lgm_params_via_ore()`](/Users/gordonlee/Documents/PythonOreRunner/src/pythonore/io/ore_snapshot.py#L775) now persists a failure marker at:

- `<case>/Output/calibration.failed`

If runtime calibration fails once for a case, later `auto` runs skip the doomed calibration subprocess and go straight to simulation XML params.

Verified on the SOFR case:

- first `auto` run: `6.958176 sec`
- second `auto` run on same case: `0.232642 sec`

**Best Next Step**

If you want cleaner production behavior, the two practical options are:

- reuse a matching `calibration.xml` so auto mode does not recalibrate
- build a smaller SOFR-only market/config bundle and keep runtime calibration
- for benchmark runs, force `--lgm-param-source simulation_xml`
