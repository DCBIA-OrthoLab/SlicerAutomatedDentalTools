"""Which point array carries the tooth numbers of a mesh.

`GetLabelSurface` and `isLabelSurface` existed in five copies, methods of five
`vtkTeeth` classes that were themselves copies (`ASO_IOS_utils/icp.py`,
`FlexReg_utils/util.py`, `FlexReg_Method/util.py`,
`FlexReg_Method/vtkSegTeeth.py`, `AREG_IOS_utils/vtkSegTeeth.py`).

Four of the five carried the same defect: the loop wrote `out = Preference`
then did `continue` instead of `break`, so that the next turn immediately
overwrote the value just found. The requested name was therefore only returned
when it happened to be the **last** array of the mesh; otherwise the function
returned the one after it. A mesh carrying `Universal_ID` then `Normals` had
the teeth looked for in the normals. Only the `ASO_IOS_utils` copy had `break`,
and that is the one kept: in the other four the `out = Preference` assignment
served no purpose, which is enough to say what was intended.

Depends only on the vtk interface of the mesh it receives: importable from the
Conda environment as well as from Slicer.
"""


def array_names(surf):
    """The names of the point data arrays carried by `surf`."""
    point_data = surf.GetPointData()
    return [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]


def has_label_array(surf, name):
    """Does `surf` carry a point array named `name`?"""
    return name in array_names(surf)


def label_array(surf, preference="Universal_ID"):
    """The array to use as the tooth numbering.

    `preference` when it is there, otherwise the last array of the mesh,
    otherwise `None` when there is none. Falling back on the last one is not
    obvious at all, but it is what the five copies did and nothing says which
    one would be right: changing it would take a decision, not a refactor.
    """
    names = array_names(surf)
    if not names:
        return None
    return preference if preference in names else names[-1]
