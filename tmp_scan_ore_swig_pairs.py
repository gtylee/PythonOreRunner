from __future__ import annotations

import itertools
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


BASE = Path("/private/tmp/USD_Mixed6_Compare")
PORTFOLIO = BASE / "Input" / "portfolio.xml"


CODE = """
import sys, time
from pathlib import Path
from dataclasses import replace
repo = Path('/Users/gordonlee/Documents/PythonOreRunner')
sys.path.insert(0, str(repo / 'src'))
from pythonore.io.loader import XVALoader
from pythonore.runtime.runtime import XVAEngine, ORESwigAdapter
ore_xml = Path(sys.argv[1])
snapshot = XVALoader.from_files(str(ore_xml.parent), ore_file=ore_xml.name)
snapshot = replace(snapshot, config=replace(snapshot.config, analytics=('NPV',)))
start = time.perf_counter()
res = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
print({'sec': time.perf_counter() - start, 'pv': res.pv_total})
"""


def write_case(trades: list[ET.Element], selected: set[str], case_dir: Path) -> None:
    shutil.copytree(BASE, case_dir, dirs_exist_ok=True)
    new_root = ET.Element("Portfolio")
    for trade in trades:
        if trade.attrib.get("id", "") in selected:
            new_root.append(trade)
    ET.ElementTree(new_root).write(case_dir / "Input" / "portfolio.xml", encoding="unicode")


def main() -> int:
    root = ET.parse(PORTFOLIO).getroot()
    trades = root.findall("./Trade")
    trade_ids = [t.attrib.get("id", "") for t in trades]
    print("trades", trade_ids)
    with tempfile.TemporaryDirectory(prefix="ore_pairscan_") as td:
        td_path = Path(td)
        for size in (2, 3):
            print("size", size)
            for idx, combo in enumerate(itertools.combinations(trade_ids, size), start=1):
                case_dir = td_path / f"case_{size}_{idx}"
                write_case(trades, set(combo), case_dir)
                ore_xml = case_dir / "Input" / "ore.xml"
                proc = subprocess.run(
                    [sys.executable, "-c", CODE, str(ore_xml)],
                    capture_output=True,
                    text=True,
                )
                status = "ok" if proc.returncode == 0 else f"rc={proc.returncode}"
                print(combo, status)
                if proc.returncode != 0:
                    err = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
                    print("  err:", " | ".join(err))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
