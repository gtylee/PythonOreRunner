"""Compatibility facade for canonical HW2F helpers."""

import sys
from pythonore import hw2f as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if k not in {"__name__", "__package__", "__loader__", "__spec__"}})
sys.modules[__name__] = _impl
