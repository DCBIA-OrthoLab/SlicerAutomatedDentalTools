# Only the resampling helpers remain here: MRI2CBCT_RESAMPLE_CBCT_MRI is still
# run by AREG_IOSCBCT and VFACE. The MRI2CBCT module itself carries its own
# package, MRI2CBCT_core. Each name is resolved to its submodule on first
# access, so a caller only pays for the dependencies it actually uses.

import importlib

_EXPORTS = {
    "create_csv": "resample_create_csv",
    "resample_images": "resample",
    "run_resample": "resample",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
