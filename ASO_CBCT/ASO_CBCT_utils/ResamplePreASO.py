import shutil
import SimpleITK as sitk
import numpy as np
import argparse
import os
import glob
import sys
import csv
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("ASO_CBCT_Resample_Pre")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

Spacing = []


def resample_fn(img, args):
    output_size = args["size"]
    fit_spacing = args["fit_spacing"]
    iso_spacing = args["iso_spacing"]
    pixel_dimension = args["pixel_dimension"]
    center = args["center"]

    if args["linear"]:
        InterpolatorType = sitk.sitkLinear
    else:
        InterpolatorType = sitk.sitkNearestNeighbor

    spacing = img.GetSpacing()
    size = img.GetSize()

    output_origin = img.GetOrigin()
    output_size = [si if o_si == -1 else o_si for si, o_si in zip(size, output_size)]

    if fit_spacing:
        output_spacing = [
            sp * si / o_si for sp, si, o_si in zip(spacing, size, output_size)
        ]
    else:
        output_spacing = spacing

    if iso_spacing:
        output_spacing_filtered = [
            sp for si, sp in zip(args["size"], output_spacing) if si != -1
        ]
        max_spacing = np.max(output_spacing_filtered)
        output_spacing = [
            sp if si == -1 else max_spacing
            for si, sp in zip(args["size"], output_spacing)
        ]

    if args["spacing"] is not None:
        if isinstance(args["spacing"], float):
            output_spacing = [args["spacing"], args["spacing"], args["spacing"]]
        else:
            output_spacing = args["spacing"]

    if args["origin"] is not None:
        output_origin = args["origin"]

    if center:
        output_physical_size = np.array(output_size) * np.array(output_spacing)
        input_physical_size = np.array(size) * np.array(spacing)
        output_origin = (
            np.array(output_origin) - (output_physical_size - input_physical_size) / 2.0
        )

    if args["direction"]:
        output_direction = np.identity(3).flatten()
    else:
        output_direction = img.GetDirection()
    Spacing.append(output_spacing)

    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetInterpolator(InterpolatorType)
    resampleImageFilter.SetOutputSpacing(output_spacing)
    resampleImageFilter.SetSize(output_size)
    resampleImageFilter.SetOutputDirection(output_direction)
    resampleImageFilter.SetOutputOrigin(output_origin)

    return resampleImageFilter.Execute(img)


def Resample(img_filename, args):

    output_size = args["size"]
    fit_spacing = args["fit_spacing"]
    iso_spacing = args["iso_spacing"]
    img_dimension = args["image_dimension"]
    pixel_dimension = args["pixel_dimension"]
    center = args["center"]

    img = sitk.ReadImage(img_filename)

    return resample_fn(img, args)


def main(args):

    filenames = []
    if args["img"]:
        fobj = {}
        fobj["img"] = args["img"]
        fobj["out"] = args["out"]
        filenames.append(fobj)
    elif args["dir"]:
        out_dir = args["out"]
        normpath = os.path.normpath("/".join([args["dir"], "**", "*"]))
        FROM, WHERE = [], []
        for img in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img) and True in [
                ext in img
                for ext in [
                    ".nrrd",
                    ".nii",
                    ".nii.gz",
                    ".mhd",
                    ".dcm",
                    ".DCM",
                    ".jpg",
                    ".png",
                    "gipl.gz",
                ]
            ]:
                fobj = {}
                fobj["img"] = img
                fobj["out"] = os.path.normpath(
                    out_dir
                    + "/"
                    + "_".join(img.replace(args["dir"], "").split("_")).split(".")[0]
                    + ".nii.gz"
                )
                if args["out_ext"] is not None:
                    out_ext = args["out_ext"]
                    if out_ext[0] != ".":
                        out_ext = "." + out_ext
                    fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext
                if not os.path.exists(os.path.dirname(fobj["out"])):
                    os.makedirs(os.path.dirname(fobj["out"]))
                if not os.path.exists(fobj["out"]) or args["ow"]:
                    filenames.append(fobj)
            if os.path.isfile(img) and True in [ext in img for ext in ["json"]]:
                FROM.append(img)
                WHERE.append(
                    os.path.normpath(out_dir + "/" + img.replace(args["dir"], ""))
                )
    else:
        raise "Set img or dir to resample!"

    if args["rgb"]:
        if args["pixel_dimension"] == 3:
            logger.info("Using: RGB type pixel with unsigned char")
        elif args["pixel_dimension"] == 4:
            logger.info("Using: RGBA type pixel with unsigned char")
        else:
            logger.warning("WARNING: Pixel size not supported!")

    if args["ref"] is not None:
        ref = sitk.ReadImage(args["ref"])
        args["size"] = ref.GetSize()
        args["spacing"] = ref.GetSpacing()
        args["origin"] = ref.GetOrigin()

    for fobj in filenames:

        if not os.path.exists(fobj["out"]):

            try:
                if "ref" in fobj and fobj["ref"] is not None:
                    ref = sitk.ReadImage(fobj["ref"])
                    args["size"] = ref.GetSize()
                    args["spacing"] = ref.GetSpacing()
                    args["origin"] = ref.GetOrigin()

                if args["size"] is not None:
                    img = Resample(fobj["img"], args)
                else:
                    img = sitk.ReadImage(fobj["img"])

                if args["spacing"] is not None:
                    writer = sitk.ImageFileWriter()
                    writer.SetFileName(fobj["out"])
                    writer.UseCompressionOn()
                    writer.Execute(img)

            except Exception as e:
                logger.error("Error during the resampling.")


def PreASOResample(data_dir, out_dir, spacing):

    args = {
        "img": None,
        "dir": data_dir,
        "ref": None,
        "size": [128, 128, 128],
        "spacing": spacing,
        "origin": None,
        "linear": False,
        "center": True,
        "fit_spacing": True,
        "iso_spacing": True,
        "direction": True,
        "image_dimension": 2,
        "pixel_dimension": 1,
        "rgb": False,
        "ow": 1,
        "out": out_dir,
        "out_ext": None,
    }

    main(args)
