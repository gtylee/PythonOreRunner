from importlib import import_module

_EXPORTS = {
    "XVALoader": ("pythonore.io.loader", "XVALoader"),
    "merge_snapshots": ("pythonore.io.loader", "merge_snapshots"),
    "CurveDFPayload": ("pythonore.io.ore_snapshot", "CurveDFPayload"),
    "OreSnapshot": ("pythonore.io.ore_snapshot", "OreSnapshot"),
    "discount_factors_to_dataframe": ("pythonore.io.ore_snapshot", "discount_factors_to_dataframe"),
    "dump_discount_factors_json": ("pythonore.io.ore_snapshot", "dump_discount_factors_json"),
    "extract_discount_factors_by_currency": ("pythonore.io.ore_snapshot", "extract_discount_factors_by_currency"),
    "extract_market_instruments_by_currency": ("pythonore.io.ore_snapshot", "extract_market_instruments_by_currency"),
    "extract_market_instruments_by_currency_from_quotes": ("pythonore.io.ore_snapshot", "extract_market_instruments_by_currency_from_quotes"),
    "fit_discount_curves_from_ore_market": ("pythonore.io.ore_snapshot", "fit_discount_curves_from_ore_market"),
    "fit_discount_curves_from_programmatic_quotes": ("pythonore.io.ore_snapshot", "fit_discount_curves_from_programmatic_quotes"),
    "fitted_curves_to_dataframe": ("pythonore.io.ore_snapshot", "fitted_curves_to_dataframe"),
    "load_from_ore_xml": ("pythonore.io.ore_snapshot", "load_from_ore_xml"),
    "ore_input_validation_dataframe": ("pythonore.io.ore_snapshot", "ore_input_validation_dataframe"),
    "quote_dicts_from_pairs": ("pythonore.io.ore_snapshot", "quote_dicts_from_pairs"),
    "validate_ore_input_snapshot": ("pythonore.io.ore_snapshot", "validate_ore_input_snapshot"),
    "validate_xva_snapshot_dataclasses": ("pythonore.io.ore_snapshot", "validate_xva_snapshot_dataclasses"),
    "xva_snapshot_validation_dataframe": ("pythonore.io.ore_snapshot", "xva_snapshot_validation_dataframe"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
