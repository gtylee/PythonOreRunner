"""Compatibility facade for the canonical shared dataclass model."""

import sys
from pythonore.domain import dataclasses as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if k not in {"__name__", "__package__", "__loader__", "__spec__"}})
sys.modules[__name__] = _impl
