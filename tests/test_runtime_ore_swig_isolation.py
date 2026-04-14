from types import SimpleNamespace
from unittest.mock import patch

from pythonore.runtime.results import XVAResult
from pythonore.runtime.swig import ORESwigAdapter


def _fake_result() -> XVAResult:
    return XVAResult(
        run_id="probe",
        pv_total=1.0,
        xva_total=0.0,
        xva_by_metric={},
        exposure_by_netting_set={},
    )


def test_ore_swig_adapter_uses_subprocess_isolation_by_default() -> None:
    adapter = ORESwigAdapter(module=SimpleNamespace(InputParameters=object, OREApp=object))
    assert adapter._process_isolation is True

    sentinel = _fake_result()
    with patch.object(adapter, "_run_via_subprocess", return_value=sentinel) as mocked:
        result = adapter.run(SimpleNamespace(), SimpleNamespace(), "probe")

    mocked.assert_called_once()
    assert result is sentinel


def test_ore_swig_adapter_direct_mode_is_available_for_child_worker() -> None:
    adapter = ORESwigAdapter(module=SimpleNamespace(InputParameters=object, OREApp=object), process_isolation=False)
    sentinel = _fake_result()
    with patch.object(adapter, "_run_direct", return_value=sentinel) as mocked:
        result = adapter.run(SimpleNamespace(), SimpleNamespace(), "probe")

    mocked.assert_called_once()
    assert result is sentinel
