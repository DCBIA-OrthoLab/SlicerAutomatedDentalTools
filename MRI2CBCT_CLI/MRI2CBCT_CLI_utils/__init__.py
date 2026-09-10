# The submodules of this package do not share the same dependencies: resample
# and resample_create_csv only need SimpleITK/numpy/pandas, while the others
# pull in torch, nibabel, sklearn, itk or torchreg. Importing them all eagerly
# meant that a caller asking only for resample_images (AREG_IOSCBCT, VFACE)
# still had to have torchreg installed, and crashed with a ModuleNotFoundError
# if it wasn't. Each name is therefore resolved to its submodule on first
# access, so a caller only pays for the dependencies it actually uses.

import importlib

_EXPORTS = {
    "create_csv": "resample_create_csv",
    "resample_images": "resample",
    "run_resample": "resample",
    "invert_mri_intensity": "mri_inverse",
    "normalize": "normalize_percentile",
    "apply_mask_f": "apply_mask",
    "registration": "AREG_MRI",
    "approximation": "approximate",
    "segment_condyle": "condyle_segmentation",
    "NMI": "nmi",
    "get_transformation": "crop_approximation",
    "crop_volume": "crop_approximation",
    "crop_mri": "LR_crop",
    "crop_cbct": "LR_crop",
    "GetPatients": "TMJ_crop",
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
