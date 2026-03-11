from native_xva_interface import stress_classic_native_preset
from py_ore_tools.repo_paths import pythonorerunner_root


def test_stress_classic_native_preset_loads_expected_buffers():
    repo_root = pythonorerunner_root()
    cfg = stress_classic_native_preset(repo_root, num_paths=10)
    assert cfg.num_paths == 10
    assert cfg.runtime is not None
    for k in ("pricingengine.xml", "todaysmarket.xml", "curveconfig.xml", "simulation.xml", "netting.xml"):
        assert k in cfg.xml_buffers
        assert cfg.xml_buffers[k]
