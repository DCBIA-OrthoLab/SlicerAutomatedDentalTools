# These re-exports are a convenience for code running inside Slicer, where vtk
# and numpy are available. The guard is kept because this package is also
# reachable from the lean shapeaxi conda environment, which has no vtk: an
# unconditional import here makes every import of any submodule fail there.
# What used to be imported from that environment was install_pytorch, which
# now lives in ADTLib.env and no longer goes through this package.
try:
    from .util import ToothNoExist, NoSegmentationSurf
    from .orientation import orientation_f
except ImportError:
    pass
