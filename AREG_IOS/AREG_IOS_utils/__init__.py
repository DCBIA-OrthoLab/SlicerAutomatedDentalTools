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
def __getattr__(name):
    if name == "PredPatch":
        from .PredPatch import PredPatch
        return PredPatch
    if name == "MonaiUNetHRes":
        from .net import MonaiUNetHRes
        return MonaiUNetHRes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
