"""Compatibility facade for the canonical LGM IR options helpers."""

import sys
from pythonore.compute import lgm_ir_options as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if k not in {"__name__", "__package__", "__loader__", "__spec__"}})
sys.modules[__name__] = _impl
