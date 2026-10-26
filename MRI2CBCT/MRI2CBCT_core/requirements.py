"""What a run needs installed, and what it downloads.

The packages are the union of what this module's steps import: ASO and ALI
(the CBCT orientation), AMASSS (the CBCT cranial base), nnU-Net (the MRI
cranial base) and Open3D (the registration). The pins are the ones those
modules already use, copied rather than chosen here:
    ASO    itk, pytorch_lightning, dicom2nifti>=2.6.2, pydicom 3.0.2, monai 1.3.2
    AMASSS itk, blosc2, dicom2nifti>=2.6.2, pydicom 3.0.2, einops, nibabel, nnunetv2>=2.8.0
    numpy  1.26.4, restored LAST: a later install can pull numpy 2 and break torch
torch is not in the list: which build a machine needs is decided from its GPU
(ADTLib.env.cuda.torch_install_arguments), and it is installed first.
"""
import os

PACKAGES = [
    ("itk", None), ("pytorch_lightning", None), ("dicom2nifti", ">=2.6.2"),
    ("pydicom", "==3.0.2"), ("monai", "==1.3.2"),
    ("blosc2", None), ("einops", None), ("nibabel", None), ("scipy", None),
    ("nnunetv2", ">=2.8.0"),
]
NUMPY = "1.26.4"

#: Pure-Python packages `import open3d` needs, installed normally. dash stays on
#: 2.18: dash 3 needs pydantic, a compiled package with the same manylinux problem.
OPEN3D_RUNTIME = ["plotly", "addict", "configargparse", "pyquaternion", "dash==2.18.2",
                  "flask", "werkzeug", "nest_asyncio", "retrying"]

ASO_LANDMARKS = ["Ba", "LPo", "N", "RPo", "S", "LOr", "ROr"]
ASO_GOLD = ("https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/"
            "Frankfurt_Horizontal_Midsagittal_Plane.zip")
ALI_MODELS = "https://github.com/lucanchling/ALI_CBCT/releases/download/models_v01"
AMASSS_MODELS = ("https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/"
                 "download/AMASSS_CBCT/AMASSS_Models.zip")


def downloads_plan(downloads, dataset516_url, dataset516_root):
    """(label, url, root, name) for each download, in the layout ASO, ALI and AMASSS
    use themselves, so a model they already fetched is never fetched again."""
    aso = os.path.join(downloads, "ASO", "ASO_CBCT")
    ali = os.path.join(downloads, "ALI", "ALI_CBCT")
    plan = [("ASO Frankfort reference", ASO_GOLD, aso,
             os.path.join("Reference", "Frankfurt Horizontal and Midsagittal Plane"))]
    plan += [(f"ALI landmark {lm}", f"{ALI_MODELS}/{lm}.zip", ali, lm) for lm in ASO_LANDMARKS]
    plan.append(("AMASSS models", AMASSS_MODELS, os.path.join(downloads, "AMASSS", "Models"),
                 "Segmentation"))
    plan.append(("Dataset516 (MRI cranial base)", dataset516_url,
                 os.path.dirname(dataset516_root), os.path.basename(dataset516_root)))
    return plan


def missing_packages(check):
    """`check(name, spec) -> bool` is ADTLib.env.deps.check_lib_installed."""
    return [(n, v) for n, v in PACKAGES if not check(n, v)]
