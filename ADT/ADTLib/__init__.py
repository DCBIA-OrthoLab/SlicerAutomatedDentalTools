"""Code shared by every module of the Automated Dental Tools extension.

Slicer puts each extension's `qt-scripted-modules` directory on `sys.path`, so
this package is reachable by a plain `import ADTLib` from any module of this
extension, from a scripted CLI, and -- once Slicer's module search paths are
published into it -- from inside the shapeaxi Conda environment. That last
route is what `ADTLib.env.install_pytorch` relies on: it is not imported by the
widgets, it is run as `python -m ADTLib.env.install_pytorch` by the Conda
interpreter.

Keep this file free of imports. Anything pulled in here runs in the lean Conda
environment too, which has neither vtk, nor qt, nor slicer; a single import of
one of those would break every submodule import there. Submodules that need
them import them themselves.
"""


def package_root():
    """Directory that has to be on PYTHONPATH for `import ADTLib` to work.

    The modules publish Slicer's module search paths into the Conda environment
    so that the tools they run there can import what ships with the extension.
    This package is not on those paths: a search path is a directory Slicer
    found a module in, and this one holds no module, only a package -- passing
    it with --additional-module-path does not put it there either. An installed
    build hides that, because the package then sits in qt-scripted-modules
    beside the modules themselves, and that directory is a search path.

    So the package says where it is, rather than relying on being found.
    """
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
