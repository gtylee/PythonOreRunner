"""Compatibility facade for :mod:`pythonore.runtime.runtime_impl`."""

from importlib import import_module
import sys

_impl = import_module('pythonore.runtime.runtime_impl')
sys.modules[__name__] = _impl
