from importlib import import_module

_EXPORTS = {
    "MappedInputs": ("pythonore.mapping.mapper", "MappedInputs"),
    "build_input_parameters": ("pythonore.mapping.mapper", "build_input_parameters"),
    "map_snapshot": ("pythonore.mapping.mapper", "map_snapshot"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
