#!/usr/bin/env python-real

"""Crop both temporomandibular joints of every oriented CBCT of a folder.

This is the last step of the TMJ_Crop chain. The module's interface runs the
others before it (PRE_ASO_CBCT, ALI_CBCT, SEMI_ASO_CBCT, AMASSS_CLI) and hands
their folders to this one, which cuts each joint out of the oriented scan in a
box of 96 x 112 x 80 voxels at 0.3 mm, writes the per-scan report and a quick
view, and one run report for the whole batch. The algorithm is SADT-VISOR's
TMJ_Crop, see TMJ_Crop_utils/crop.py.
"""

import argparse
import os
import sys

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_root, "ADT", "ADTLib")) \
        and _root != os.path.dirname(_root):
    _root = os.path.dirname(_root)
if os.path.join(_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_root, "ADT"))
# The helpers of the orientation step and of this CLI, found the same way in a
# source tree; an installed build has both on sys.path already.
for _folder in (os.path.join(_root, "ASO_CBCT"),
                os.path.dirname(os.path.realpath(__file__))):
    if os.path.isdir(_folder) and _folder not in sys.path:
        sys.path.append(_folder)

from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import emit_fraction

logger = get_logger("TMJ_Crop_CLI")


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oriented_folder", help="SEMI_ASO_CBCT output: the oriented scans and their landmarks")
    parser.add_argument("landmarks_folder", help="The landmarks that went into SEMI_ASO_CBCT")
    parser.add_argument("mask_folder", help="AMASSS_CLI output: one mandible mask per oriented scan")
    parser.add_argument("reference_folder", help="The reference SEMI_ASO_CBCT registered onto")
    parser.add_argument("output_folder")
    parser.add_argument("--input_folder", default="", help="The folder the scans came from")
    parser.add_argument("--sides", default="Left and Right")
    parser.add_argument("--outputs", default="", help="Comma-separated extra outputs")
    parser.add_argument("--scale_by", default="none")
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--add_inname", default="Or")
    parser.add_argument("--prediction_id", default="Pred")
    return parser.parse_args(argv)


def main(args) -> None:
    from TMJ_Crop_utils import batch

    outputs = [name.strip() for name in args.outputs.split(",") if name.strip()]

    def progress(finished, total):
        emit_fraction(finished / total if total else 1.0)

    def tell(message):
        # Standard output is what Slicer keeps as the CLI's text; the panel's
        # error dialog shows the lines that begin with "Warning".
        print(f"Warning: {message}", flush=True)

    batch.crop_batch(
        oriented_dir=args.oriented_folder,
        landmarks_dir=args.landmarks_folder,
        mask_dir=args.mask_folder,
        reference_dir=args.reference_folder,
        output_dir=args.output_folder,
        input_dir=args.input_folder or None,
        sides=args.sides,
        outputs=outputs,
        scale_by=args.scale_by,
        scale=args.scale,
        suffix=args.add_inname,
        prediction_id=args.prediction_id,
        on_scan_done=progress,
        tell=tell,
    )


if __name__ == "__main__":
    try:
        logger.info("TMJ_Crop_CLI entry point initiated")
        main(parse())
        logger.info("TMJ_Crop_CLI completed successfully")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(f"Fatal error: {e}")
