"""One logging setup for the whole extension.

A hundred and fifty-one files carried the same ten lines, and every one of them
set `propagate = False` and attached its own StreamHandler. That is what makes
the extension's logging impossible to steer: raising a level, silencing a noisy
module or sending a run to a file would mean editing every file that logs.

Here the handler is attached **once**, to a single parent logger named `ADT`,
and every module's logger is a child of it that propagates normally. Changing
the level or the destination is then one call, in one place.

Names are kept readable rather than dotted module paths: `ADT.ASO_Method.CBCT`
reads the way the old `"ASO_Method_CBCT"` did, and still groups under `ADT`.

Standard library only: this is imported from the Conda environment too.
"""
import logging
import sys

ROOT = "ADT"
FORMAT = "%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s"

_configured = False


def _configure_root():
    """Attach the one handler, the first time anything asks for a logger."""
    global _configured
    if _configured:
        return
    root = logging.getLogger(ROOT)
    root.setLevel(logging.INFO)
    # Not propagating to Python's root keeps Slicer's own console from showing
    # each line twice, which is what the per-file copies were reaching for.
    root.propagate = False
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(handler)
    _configured = True


def get_logger(name):
    """The logger a module should use.

    `name` is the old per-file name (`"ASO_Method_CBCT"`) or `__name__`; either
    way it ends up under `ADT.`, so one call sets the level for all of them.
    """
    _configure_root()
    name = name or ROOT
    if name == ROOT or name.startswith(ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT}.{name}")


def set_level(level):
    """Set the level for every logger of the extension at once."""
    _configure_root()
    logging.getLogger(ROOT).setLevel(level)
    for handler in logging.getLogger(ROOT).handlers:
        handler.setLevel(level)
