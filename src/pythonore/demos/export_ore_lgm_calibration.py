#!/usr/bin/env python3
"""Export ORE calibrated LGM parameters into normalized JSON/CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

if __package__ in (None, ""):
    REPO_BOOTSTRAP = Path(__file__).resolve().parents[3]
    if str(REPO_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(REPO_BOOTSTRAP))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-xml", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def _parse_grid(text: str | None) -> list[float]:
    s = (text or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = _parse_args()
    root = ET.parse(args.calibration_xml).getroot()
    models = root.find("./InterestRateModels")
    if models is None:
        raise ValueError("calibration xml missing InterestRateModels")

    rows = []
    payload = {"currencies": {}}

    for lgm in models.findall("./LGM"):
        key = (lgm.attrib.get("key") or lgm.attrib.get("ccy") or "").upper()
        if not key:
            continue
        vol = lgm.find("./Volatility")
        rev = lgm.find("./Reversion")
        trans = lgm.find("./ParameterTransformation")
        if vol is None or rev is None or trans is None:
            continue

        alpha_times = _parse_grid(vol.findtext("./TimeGrid"))
        alpha_values = _parse_grid(vol.findtext("./InitialValue"))
        kappa_times = _parse_grid(rev.findtext("./TimeGrid"))
        kappa_values = _parse_grid(rev.findtext("./InitialValue"))
        shift = float((trans.findtext("./ShiftHorizon") or "0").strip())
        scaling = float((trans.findtext("./Scaling") or "1").strip())

        payload["currencies"][key] = {
            "alpha_times": alpha_times,
            "alpha_values": alpha_values,
            "kappa_times": kappa_times,
            "kappa_values": kappa_values,
            "shift": shift,
            "scaling": scaling,
        }

        rows.append([key, "alpha_times", ",".join(str(x) for x in alpha_times)])
        rows.append([key, "alpha_values", ",".join(str(x) for x in alpha_values)])
        rows.append([key, "kappa_times", ",".join(str(x) for x in kappa_times)])
        rows.append([key, "kappa_values", ",".join(str(x) for x in kappa_values)])
        rows.append([key, "shift", str(shift)])
        rows.append([key, "scaling", str(scaling)])

    if not payload["currencies"]:
        raise ValueError("no LGM entries found in calibration xml")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ccy", "field", "value"])
        w.writerows(rows)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
