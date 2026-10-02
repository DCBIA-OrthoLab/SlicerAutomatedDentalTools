"""Mesh reading and writing: all of it now lives in ADTLib.

This file carried a copy of `OFFReader`, `ReadSurf` and `WriteSurf`. All three
existed identically, or nearly so, in four other modules; the detail of what
diverged and what was kept is in `ADTLib/io/surface.py`. This module stays for
the callers that import it by its original path.

The `WriteSurf` from here forced the output to `.vtk` whatever the input
extension was. Its only caller, the segmentation bypass in `IOS.py`, wanted
exactly that conversion: it now asks for the `.vtk` in the name it passes,
instead of the function deciding it for everyone.
"""
from ADTLib.io.surface import OFFReader, ReadSurf, WriteSurf  # noqa: F401  (re-exported)
