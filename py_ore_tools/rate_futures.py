"""Compatibility facade for the canonical rate futures helpers."""

import sys
from pythonore.compute import rate_futures as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if k not in {"__name__", "__package__", "__loader__", "__spec__"}})
sys.modules[__name__] = _impl
