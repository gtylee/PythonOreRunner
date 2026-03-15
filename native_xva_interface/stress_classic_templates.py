"""Compatibility facade for the canonical stress-classic templates."""

import sys
from pythonore.runtime import stress_classic_templates as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if k not in {"__name__", "__package__", "__loader__", "__spec__"}})
sys.modules[__name__] = _impl
