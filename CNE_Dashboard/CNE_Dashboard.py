#!/usr/bin/env python-real
import argparse
import sys, os

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import emit_fraction

logger = get_logger("CNE_Dashboard")

# realpath, not __file__: a scripted CLI is registered through a flat folder of
# symlinks into the source tree, so __file__ names the link, whose parent holds
# no CNE_CLI. Resolving first lands beside CNE_CLI either way; a built install
# has CNE_CLI_utils on sys.path already.
_here = os.path.dirname(os.path.realpath(__file__))
_cne_cli = os.path.join(os.path.dirname(_here), "CNE_CLI")
if os.path.isdir(_cne_cli) and _cne_cli not in sys.path:
    sys.path.append(_cne_cli)

from CNE_CLI_utils import show_dashboard


def main(args):
    print("<filter-start><filter-name>CNE Dashboard</filter-name></filter-start>", flush=True)
    os.makedirs(args.output_folder, exist_ok=True)
    emit_fraction(0.10)

    show_dashboard(args.summary_folder, args.output_folder)

    emit_fraction(1.0)
    print("<filter-end><filter-name>CNE Dashboard</filter-name></filter-end>", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw the patient dashboard from the extractions CNE_CLI wrote.")
    parser.add_argument("summary_folder", type=str)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()
    main(args)
