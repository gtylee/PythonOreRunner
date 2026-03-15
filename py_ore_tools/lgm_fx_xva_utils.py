"""Compatibility facade for the canonical LGM FX/XVA helpers."""

import sys
from pythonore.compute import lgm_fx_xva_utils as _impl

globals().update({k: v for k, v in _impl.__dict__.items() if k not in {"__name__", "__package__", "__loader__", "__spec__"}})
sys.modules[__name__] = _impl
