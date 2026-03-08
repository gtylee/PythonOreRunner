from pathlib import Path

from native_xva_interface import stress_classic_native_preset


def test_stress_classic_native_preset_loads_expected_buffers():
    repo_root = Path(__file__).resolve().parents[4]
    cfg = stress_classic_native_preset(repo_root, num_paths=10)
    assert cfg.num_paths == 10
    assert cfg.runtime is not None
    for k in ("pricingengine.xml", "todaysmarket.xml", "curveconfig.xml", "simulation.xml", "netting.xml"):
        assert k in cfg.xml_buffers
        assert cfg.xml_buffers[k]
