#!/usr/bin/env python3
"""Quickstart for the canonical ORE snapshot loader.

This is the one root-level example script that should stay current.

What it demonstrates
--------------------
This script exercises the canonical snapshot-loading path:

    ore.xml -> referenced ORE input/output files -> pythonore.io.ore_snapshot.OreSnapshot

`load_from_ore_xml()` starts from a single `ore.xml` file and resolves the
linked ORE resources that the Python parity path needs, including:

- `simulation.xml`
- `todaysmarket.xml`
- `curveconfig.xml`
- `portfolio.xml`
- market data / fixing files
- selected ORE output files such as `curves.csv`, `npv.csv`, `xva.csv`,
  and exposure reports when they exist

The returned snapshot is the canonical Python object for:

- inspecting the case in Python
- validating curve / trade / credit inputs
- feeding the Python LGM workflow
- feeding the higher-level ORE snapshot CLI

What you need
-------------
- Run this from the repository checkout, or at least from an environment where
  the repo root is importable.
- Point `--ore-xml` at a valid ORE `ore.xml` file.
- The file should live in a normal ORE-style case structure so the referenced
  files can be resolved.

By default the script uses the vendored case:

    Examples/Exposure/Input/ore_measure_lgm.xml

What this script does not do
----------------------------
- It does not run a full Python pricing/XVA workflow.
- It does not call the ORE binary.
- It does not compare Python results against ORE results.

It is intentionally a read/inspect quickstart. After confirming the snapshot
loads, the next step is usually the canonical Python-only notebook or CLI:

    notebook_series/05_1_python_only_workflow.ipynb
    python3 -m pythonore.apps.ore_snapshot_cli <ore.xml> --xva --paths 10000

Example usage
-------------
    python3 example_ore_snapshot.py
    python3 example_ore_snapshot.py --ore-xml Examples/Exposure/Input/ore_measure_lgm_fixed.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pythonore.io.ore_snapshot import load_from_ore_xml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a single ORE ore.xml into the canonical Python snapshot object and print a high-signal summary."
    )
    parser.add_argument(
        "--ore-xml",
        type=Path,
        default=REPO_ROOT / "Examples" / "Exposure" / "Input" / "ore_measure_lgm.xml",
        help="Path to an ORE ore.xml file. Defaults to the vendored Exposure example.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ore_xml = args.ore_xml.resolve()
    snapshot = load_from_ore_xml(ore_xml)

    print("Loaded ORE snapshot")
    print(f"  ore_xml        : {snapshot.ore_xml_path}")
    print(f"  portfolio_xml  : {snapshot.portfolio_xml_path}")
    print(f"  todaysmarket   : {snapshot.todaysmarket_xml_path}")
    print(f"  simulation_xml : {snapshot.simulation_xml_path}")
    print(f"  curves_csv     : {snapshot.curves_csv_path}")
    print(f"  xva_csv        : {snapshot.xva_csv_path}")
    print(f"  npv_csv        : {snapshot.npv_csv_path}")
    print(f"  asof           : {snapshot.asof_date}")
    print(f"  trade_id       : {snapshot.trade_id}")
    print(f"  counterparty   : {snapshot.counterparty}")
    print(f"  netting_set    : {snapshot.netting_set_id}")
    print(f"  base_ccy       : {snapshot.domestic_ccy}")
    print(f"  measure        : {snapshot.measure}")
    print(f"  alpha_source   : {snapshot.alpha_source}")
    print(f"  model_dc       : {snapshot.model_day_counter}")
    print(f"  report_dc      : {snapshot.report_day_counter}")
    print(f"  discount_curve : {snapshot.discount_column}")
    print(f"  forward_curve  : {snapshot.forward_column}")
    print(f"  xva_disc_curve : {snapshot.xva_discount_column}")
    print(f"  market_quotes  : {len(snapshot.market_data_lines) if hasattr(snapshot, 'market_data_lines') else 'n/a'}")
    print(f"  ore_t0_npv     : {snapshot.ore_t0_npv:.6f}")
    print(f"  ore_cva        : {snapshot.ore_cva:.6f}")
    print(f"  ore_dva        : {snapshot.ore_dva:.6f}")
    print(f"  ore_fba        : {snapshot.ore_fba:.6f}")
    print(f"  ore_fca        : {snapshot.ore_fca:.6f}")
    print(f"  exposure_pts   : {snapshot.exposure_times.size}")
    print(f"  node_tenors    : {len(snapshot.node_tenors)}")
    print(f"  hazard_pts     : {snapshot.hazard_times.size}")
    print()
    print("What this means:")
    print("  - The ORE input bundle was resolved successfully from a single ore.xml.")
    print("  - Curves, trade legs, credit inputs, and selected ORE outputs are now available")
    print("    through one Python object.")
    print("  - This object is the canonical handoff into the Python-only pricing/XVA flow.")
    print()
    print("Canonical modules:")
    print("  pythonore.io.ore_snapshot")
    print("  pythonore.workflows.ore_snapshot_cli")
    print()
    print("Typical next steps:")
    print("  1. Open notebook_series/05_1_python_only_workflow.ipynb for the canonical Python-only flow.")
    print("  2. Inspect snapshot fields programmatically.")
    print("  3. Run the canonical CLI for pricing/XVA.")
    print("  4. Use the snapshot as input to higher-level Python workflows.")
    print()
    print("CLI quickstart:")
    print(f"  python3 -m pythonore.apps.ore_snapshot_cli {snapshot.ore_xml_path} --xva --paths 10000")
    print()
    print("Notebook quickstart:")
    print("  notebook_series/05_1_python_only_workflow.ipynb")
    print()
    print("Programmatic quickstart:")
    print("  from pythonore.io.ore_snapshot import load_from_ore_xml")
    print(f"  snapshot = load_from_ore_xml(r'{snapshot.ore_xml_path}')")


if __name__ == "__main__":
    main()
