"""Read and write a surface mesh, in a single place.

`ReadSurf` existed in five copies and `WriteSurf` in four, and unlike the
functions already shared here, **those had diverged**. The measured differences,
and what was kept:

`ReadSurf`
  - `ASO/IOS_utils/Reader.py` and `ASO_IOS_utils/utils.py` computed `fname` from
    `basename(path)`. The `.mtl` file of an `.obj` was therefore looked up in the
    **current working directory**, not next to the mesh: the "with material"
    branch practically never fired. The other three kept the full path, and that
    is what was kept.
  - Those same two copies set the texture path in an `if/else` whose two
    branches computed the same value -- dead code from a copy-paste. Kept: the
    shape of the other three, which only sets the `../images` path when it
    exists.
  - `.off` was read by only three of the five. It is now read by all of them.
  - An unknown extension returned a never-assigned `surf`, hence an opaque
    `UnboundLocalError`, in three copies out of five. `AREG_IOS` and `ALI_IOS`
    already raised `ValueError`: kept.
  - A missing file raised nothing: the VTK readers silently return an empty
    mesh, and the error only surfaced much later, during registration.
    `ALI_IOS` already checked existence and the point count: kept for everyone.
    **This is the only change that can make a run fail where it "passed"
    before** -- it was already failing, later and without saying why.

`WriteSurf`
  - `ASO_IOS_utils` picked the writer from the extension; the others always
    wrote legacy VTK *under the original name*, hence VTK bytes in a file named
    `.vtp` or `.obj`. Kept: the writer follows the extension, and an extension
    with no known writer forces `.vtk`, name included.
  - `os.mkdir` (three copies) fails when the parent is missing and on a folder
    created meanwhile by another process: `os.makedirs(..., exist_ok=True)`.
  - The ASO one took no `inname` and forced `.vtk` no matter what. The single
    call site concerned converts a mesh for segmentation, and a `.vtk` is indeed
    what it wants: it now asks for that explicitly in the name it passes, rather
    than the function deciding for everyone.

Imported from the Conda environment by the CLIs: vtk is enough, Slicer and Qt
are not required.
"""
import os

import vtk

from ADTLib.logging_setup import get_logger

logger = get_logger(__name__)


class OFFReader:
    """Minimal OFF reader, with the interface of the vtk readers.

    The two copies (`ASO/IOS_utils/Reader.py`, `ASO_IOS_utils/OFFReader.py`) were
    identical character for character, logging preamble aside. Two defects fixed
    along the way:

    - `__init__` assigned local variables `FileName` and `Output` instead of the
      attributes, so that a `GetOutput()` before `Update()` raised
      `AttributeError` instead of returning `None`;
    - an invalid header did `raise ("Not a valid OFF header")`, that is, raising
      a string: `TypeError: exceptions must derive from BaseException`, which
      hides the real message.
    """

    def __init__(self):
        self.FileName = None
        self.Output = None

    def SetFileName(self, file_name):
        self.FileName = file_name

    def GetOutput(self):
        return self.Output

    def Update(self):
        with open(self.FileName) as file:
            if "OFF" != file.readline().strip():
                raise ValueError(f"Not a valid OFF header: {self.FileName}")

            n_verts, n_faces, _ = tuple(
                [int(s) for s in file.readline().strip().split(" ")]
            )

            surf = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            for _ in range(n_verts):
                p = [float(s) for s in file.readline().strip().split(" ")]
                points.InsertNextPoint(p[0], p[1], p[2])

            for _ in range(n_faces):
                t = [int(s) for s in file.readline().strip().split(" ")]

                if t[0] == 1:
                    vertex = vtk.vtkVertex()
                    vertex.GetPointIds().SetId(0, t[1])
                    cells.InsertNextCell(vertex)
                elif t[0] == 2:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, t[1])
                    line.GetPointIds().SetId(1, t[2])
                    cells.InsertNextCell(line)
                elif t[0] == 3:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, t[1])
                    triangle.GetPointIds().SetId(1, t[2])
                    triangle.GetPointIds().SetId(2, t[3])
                    cells.InsertNextCell(triangle)

            surf.SetPoints(points)
            surf.SetPolys(cells)

            self.Output = surf


def _read_obj_with_material(file_name, fname):
    """An `.obj` together with its `.mtl`, imported then flattened into one mesh."""
    obj_import = vtk.vtkOBJImporter()
    obj_import.SetFileName(file_name)
    obj_import.SetFileNameMTL(fname + ".mtl")
    textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
    if os.path.exists(textures_path):
        obj_import.SetTexturePath(textures_path)
    obj_import.Read()

    actors = obj_import.GetRenderer().GetActors()
    actors.InitTraversal()
    append = vtk.vtkAppendPolyData()
    for _ in range(actors.GetNumberOfItems()):
        surf_actor = actors.GetNextActor()
        append.AddInputData(surf_actor.GetMapper().GetInputAsDataSet())
    append.Update()
    return append.GetOutput()


def ReadSurf(file_name):
    """The mesh contained in `fileName`, whatever its format.

    Formats read: `.vtk`, `.vtp`, `.stl`, `.off`, `.obj` (with its `.mtl` when it
    sits next to it). Raises `FileNotFoundError` when the file is missing,
    `ValueError` when the extension is not recognised or when the mesh read is
    empty -- since the vtk readers return an empty mesh rather than an error,
    this is the only way to tell "unreadable" from "empty".
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File does not exist: {file_name}")

    fname, extension = os.path.splitext(file_name)
    extension = extension.lower()

    if extension == ".obj" and os.path.exists(fname + ".mtl"):
        surf = _read_obj_with_material(file_name, fname)
    else:
        if extension == ".vtk":
            reader = vtk.vtkPolyDataReader()
        elif extension == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif extension == ".stl":
            reader = vtk.vtkSTLReader()
        elif extension == ".off":
            reader = OFFReader()
        elif extension == ".obj":
            reader = vtk.vtkOBJReader()
        else:
            raise ValueError(
                f"Unsupported file format: {extension}. Supported formats: "
                f".vtk, .vtp, .stl, .off, .obj. File: {file_name}"
            )
        reader.SetFileName(file_name)
        reader.Update()
        surf = reader.GetOutput()

    if surf.GetNumberOfPoints() == 0:
        raise ValueError(f"Surface has no points: {file_name}")

    logger.debug("Read %d points from %s", surf.GetNumberOfPoints(), file_name)
    return surf


# Which extension goes with which writer. Anything not listed here is written as
# legacy VTK, and the output name takes `.vtk` so as not to lie about its
# contents -- that is what the ASO_IOS copy already did.
_WRITERS = {
    ".vtk": vtk.vtkPolyDataWriter,
    ".vtp": vtk.vtkXMLPolyDataWriter,
    ".obj": vtk.vtkOBJWriter,
}


def WriteSurf(surf, output_folder, name, inname=""):
    """Write `surf` into `output_folder`, under `name` suffixed with `inname`.

    `name` may be a full path: only its tail is used.
    `inname` is inserted between the name and the extension, hence `A2_Seg.vtk`
    + `Or` giving `A2_SegOr.vtk`. Returns the path written.
    """
    name = os.path.basename(name)
    name, extension = os.path.splitext(name)
    extension = extension.lower()
    if extension not in _WRITERS:
        extension = ".vtk"

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{name}{inname}{extension}")

    writer = _WRITERS[extension]()
    writer.SetFileName(output_path)
    writer.SetInputData(surf)
    writer.Update()

    # vtk reports a write failure through a return code nobody reads, and the
    # run went on with a file that was not there.
    if not os.path.exists(output_path):
        raise RuntimeError(f"WriteSurf failed: {output_path} was not created")

    logger.debug("Wrote %s (%d bytes)", output_path, os.path.getsize(output_path))
    return output_path
