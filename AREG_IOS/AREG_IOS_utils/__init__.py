from .dataset import DatasetPatch, SortLower
from .vtkSegTeeth import vtkMeshTeeth
from .ICP import ICP, vtkICP
from .utils import WriteSurf, ReadSurf, LoadJsonLandmarks
from .transformation import TransformSurf, saveMatrixAsTfm
from .mgl_patch import (MGLPatch, DropDoubtfulLandmarks, SharedLandmarks,
                        AlignOnLandmarks,
                        DEFAULT_RADIUS, MGL_ARRAY_NAME)


# The palatal patch network and its stack are loaded only when asked for.
# The mucogingival registration owns no network -- ALI predicts its landmarks
# in a run of its own -- so importing them here made an MGL registration die
# at import on a Slicer carrying torch but not pytorch_lightning, over a model
# that run never touches.
# The result is cached in the module globals, and for `PredPatch` that is not a
# nicety: the submodule holding it is called `PredPatch` too, so importing it
# binds THAT name on this package to the MODULE. `from ... import PredPatch`
# then looks the name up a second time, finds the module rather than this
# accessor, and the caller gets a module it cannot call -- which is exactly how
# the palatal registration died, on the first import, with "'module' object is
# not callable". Rebinding the name to the class is what makes the second
# lookup find the right thing.
def __getattr__(name):
    if name == "PredPatch":
        from .PredPatch import PredPatch as _PredPatch
        globals()["PredPatch"] = _PredPatch
        return _PredPatch
    if name == "MonaiUNetHRes":
        from .net import MonaiUNetHRes as _MonaiUNetHRes
        globals()["MonaiUNetHRes"] = _MonaiUNetHRes
        return _MonaiUNetHRes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
