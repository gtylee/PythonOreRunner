from importlib import import_module

_EXPORTS = {
    "LGM1F": ("pythonore.compute.lgm", "LGM1F"),
    "LGMParams": ("pythonore.compute.lgm", "LGMParams"),
    "ORE_PARITY_SEQUENCE_TYPE": ("pythonore.compute.lgm", "ORE_PARITY_SEQUENCE_TYPE"),
    "OreMersenneTwisterGaussianRng": ("pythonore.compute.lgm", "OreMersenneTwisterGaussianRng"),
    "make_ore_gaussian_rng": ("pythonore.compute.lgm", "make_ore_gaussian_rng"),
    "simulate_ba_measure": ("pythonore.compute.lgm", "simulate_ba_measure"),
    "simulate_lgm_measure": ("pythonore.compute.lgm", "simulate_lgm_measure"),
    "RateFutureModelParams": ("pythonore.compute.rate_futures", "RateFutureModelParams"),
    "RateFutureQuote": ("pythonore.compute.rate_futures", "RateFutureQuote"),
    "build_discount_curve_from_zero_rate_pairs": ("pythonore.compute.irs_xva_utils", "build_discount_curve_from_zero_rate_pairs"),
    "build_discount_curve_from_discount_pairs": ("pythonore.compute.irs_xva_utils", "build_discount_curve_from_discount_pairs"),
    "swap_npv_from_ore_legs_dual_curve": ("pythonore.compute.irs_xva_utils", "swap_npv_from_ore_legs_dual_curve"),
    "LgmFxHybrid": ("pythonore.compute.lgm_fx_hybrid", "LgmFxHybrid"),
    "MultiCcyLgmParams": ("pythonore.compute.lgm_fx_hybrid", "MultiCcyLgmParams"),
    "TorchDiscountCurve": ("pythonore.compute.lgm_torch_xva", "TorchDiscountCurve"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
