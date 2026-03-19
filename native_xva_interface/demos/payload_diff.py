from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

from native_xva_interface.dataclasses import XVASnapshot
from native_xva_interface.loader import XVALoader
from native_xva_interface.mapper import build_input_parameters, map_snapshot
from native_xva_interface.presets import stress_classic_native_preset


class _CaptureInputParameters:
    def __init__(self):
        self.calls: Dict[str, Tuple[Any, ...]] = {}
        self.order: List[str] = []

    def __getattr__(self, name: str):
        if not name.startswith("set"):
            raise AttributeError(name)

        def recorder(*args):
            self.calls[name] = args
            self.order.append(name)

        return recorder


def _xml_signature(s: str) -> Dict[str, Any]:
    raw = s or ""
    sig: Dict[str, Any] = {
        "bytes": len(raw.encode("utf-8")),
        "sha1_12": hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12] if raw else "",
        "root": None,
        "children": [],
        "samples": None,
        "grid": None,
    }
    if not raw:
        return sig
    try:
        root = ET.fromstring(raw)
        sig["root"] = root.tag
        sig["children"] = [c.tag for c in list(root)[:8]]
        samples = root.findtext(".//Samples")
        if samples is not None:
            sig["samples"] = samples.strip()
        grid = root.findtext(".//Grid")
        if grid is not None:
            sig["grid"] = grid.strip()
    except Exception:
        sig["root"] = "parse_error"
    return sig


def _flatten_xml(root: ET.Element, prefix: str = "") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    cur = f"{prefix}/{root.tag}" if prefix else root.tag
    text = (root.text or "").strip()
    out[cur] = {
        "attrs": dict(sorted(root.attrib.items())),
        "text": text,
    }
    for ch in list(root):
        out.update(_flatten_xml(ch, cur))
    return out


def xml_structure_diff(reference_xml: str, candidate_xml: str) -> Dict[str, Any]:
    a_root = ET.fromstring(reference_xml)
    b_root = ET.fromstring(candidate_xml)
    a = _flatten_xml(a_root)
    b = _flatten_xml(b_root)
    a_keys = set(a)
    b_keys = set(b)
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)
    common = sorted(a_keys & b_keys)
    attr_changed = []
    text_changed = []
    for k in common:
        if a[k]["attrs"] != b[k]["attrs"]:
            attr_changed.append((k, a[k]["attrs"], b[k]["attrs"]))
        if a[k]["text"] != b[k]["text"]:
            text_changed.append((k, a[k]["text"], b[k]["text"]))
    return {
        "only_in_reference_paths": only_a,
        "only_in_candidate_paths": only_b,
        "attr_changed": attr_changed,
        "text_changed": text_changed,
    }


def _arg_repr(v: Any) -> Any:
    if isinstance(v, str) and v.lstrip().startswith("<"):
        return {"xml": _xml_signature(v)}
    return v


def _normalize(calls: Dict[str, Tuple[Any, ...]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, args in calls.items():
        out[k] = tuple(_arg_repr(a) for a in args)
    return out


def _capture(snapshot: XVASnapshot) -> Dict[str, Any]:
    cap = _CaptureInputParameters()
    build_input_parameters(snapshot, cap)
    mapped = map_snapshot(snapshot)
    return {
        "calls": cap.calls,
        "order": cap.order,
        "normalized": _normalize(cap.calls),
        "mapped_xml": {k: _xml_signature(v) for k, v in mapped.xml_buffers.items()},
    }


def diff_setter_payloads(reference: XVASnapshot, candidate: XVASnapshot) -> Dict[str, Any]:
    ref = _capture(reference)
    cand = _capture(candidate)

    ref_calls = ref["normalized"]
    cand_calls = cand["normalized"]

    all_methods = sorted(set(ref_calls) | set(cand_calls))
    changed = {}
    only_ref = []
    only_cand = []
    for m in all_methods:
        if m not in cand_calls:
            only_ref.append(m)
            continue
        if m not in ref_calls:
            only_cand.append(m)
            continue
        if ref_calls[m] != cand_calls[m]:
            changed[m] = {"reference": ref_calls[m], "candidate": cand_calls[m]}

    xml_changed = {}
    all_xml = sorted(set(ref["mapped_xml"]) | set(cand["mapped_xml"]))
    for x in all_xml:
        a = ref["mapped_xml"].get(x)
        b = cand["mapped_xml"].get(x)
        if a != b:
            xml_changed[x] = {"reference": a, "candidate": b}

    return {
        "changed_setters": changed,
        "only_in_reference": only_ref,
        "only_in_candidate": only_cand,
        "changed_xml_buffers": xml_changed,
    }


def build_working_vs_strict_native_simulation(repo_root: str | Path, num_paths: int = 10) -> Dict[str, Any]:
    root = Path(repo_root)
    input_dir = root / "Examples" / "XvaRisk" / "Input"
    base = XVALoader.from_files(str(input_dir), ore_file="ore_stress_classic.xml")

    cfg_working = stress_classic_native_preset(root, num_paths=num_paths)
    snap_working = XVASnapshot(
        market=base.market,
        fixings=base.fixings,
        portfolio=base.portfolio,
        config=cfg_working,
        netting=base.netting,
        collateral=base.collateral,
        source_meta=base.source_meta,
    )

    strict_runtime = replace(cfg_working.runtime, simulation=replace(cfg_working.runtime.simulation, strict_template=True))
    # Generate strict native simulation xml in isolation.
    strict_tmp_cfg = replace(
        cfg_working,
        runtime=strict_runtime,
        xml_buffers={},
    )
    strict_tmp_snap = replace(snap_working, config=strict_tmp_cfg)
    strict_generated = map_snapshot(strict_tmp_snap).xml_buffers["simulation.xml"]

    cfg_candidate = replace(
        cfg_working,
        runtime=strict_runtime,
        xml_buffers={**cfg_working.xml_buffers, "simulation.xml": strict_generated},
    )
    snap_candidate = replace(snap_working, config=cfg_candidate)

    base = diff_setter_payloads(snap_working, snap_candidate)
    base["simulation_structure_diff"] = xml_structure_diff(
        cfg_working.xml_buffers["simulation.xml"], strict_generated
    )
    return base


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    diff = build_working_vs_strict_native_simulation(repo_root, num_paths=64)
    print("Setter payload diff (working file-sim vs strict-native sim)")
    print("only_in_reference:", diff["only_in_reference"])
    print("only_in_candidate:", diff["only_in_candidate"])
    print("changed_setter_count:", len(diff["changed_setters"]))
    for m, d in diff["changed_setters"].items():
        print(f" - {m}")
        print("   reference:", d["reference"])
        print("   candidate:", d["candidate"])
    print("changed_xml_buffer_count:", len(diff["changed_xml_buffers"]))
    for name, d in diff["changed_xml_buffers"].items():
        print(f" - {name}")
        print("   reference:", d["reference"])
        print("   candidate:", d["candidate"])
    sim_diff = diff["simulation_structure_diff"]
    print("simulation paths only in reference:", len(sim_diff["only_in_reference_paths"]))
    for p in sim_diff["only_in_reference_paths"][:40]:
        print("  -", p)
    print("simulation paths only in candidate:", len(sim_diff["only_in_candidate_paths"]))
    for p in sim_diff["only_in_candidate_paths"][:20]:
        print("  -", p)
    print("simulation attr changes:", len(sim_diff["attr_changed"]))
    for k, a, b in sim_diff["attr_changed"][:20]:
        print("  -", k, a, "=>", b)
    print("simulation text changes:", len(sim_diff["text_changed"]))
    for k, a, b in sim_diff["text_changed"][:30]:
        if a or b:
            print("  -", k, repr(a), "=>", repr(b))


if __name__ == "__main__":
    main()
