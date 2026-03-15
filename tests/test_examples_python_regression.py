from pathlib import Path

from pythonore.workflows.examples_regression import (
    DEFAULT_BASELINE_ROOT,
    DEFAULT_MANIFEST,
    compare_baselines,
    load_manifest,
)


def test_examples_regression_manifest_resolves_cases() -> None:
    cases = load_manifest(DEFAULT_MANIFEST)
    assert cases
    for case in cases:
        assert case.ore_xml_path.exists(), case.ore_xml_path
        assert (DEFAULT_BASELINE_ROOT / case.case_id).exists(), case.case_id


def test_examples_regression_compare_passes() -> None:
    failures = compare_baselines(manifest_path=DEFAULT_MANIFEST, baseline_root=DEFAULT_BASELINE_ROOT)
    assert failures == {}


def test_canonical_pythonore_import_surface_is_available() -> None:
    import pythonore
    import py_ore_tools

    assert hasattr(pythonore, "XVASnapshot")
    assert hasattr(pythonore, "XVALoader")
    assert hasattr(pythonore, "map_snapshot")
    assert hasattr(pythonore, "XVAEngine")
    assert hasattr(py_ore_tools, "refresh_baselines")
