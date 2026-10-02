import vtk
from ADTLib.geometry import ApplyTransform, RotationMatrix, TransformDict, TransformList, TransformSurf  # noqa: F401  (re-exported)

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("ASO_IOS_Transformation")


def RotateTransform(surf, transform):
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(surf)
    transform_filter.Update()
    return transform_filter.GetOutput()


def TranslationDict(source, transform):
    """
    Apply translation to source dictionary of landmarks

    Parameters
    ----------
    source : Dictionary
        Dictionary containing the source landmarks.
    transform : numpy array
        Translation to be applied to the source.

    Returns
    -------
    Dictionary
        Dictionary containing the translated source landmarks.
    """
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = sourcee[key] + transform
    return sourcee

