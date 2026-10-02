"""Driving SlicerConda: the part every module of this extension repeated.

Six modules carried their own copy of each of these -- forty functions, 656
lines -- and the copies had drifted into three shapes: a logger line here, a
docstring there, and one `check_lib_wsl` still holding an unreachable return
that read two names nothing had bound.

This module talks to SlicerConda and to Slicer, so it must not be imported from
inside a Conda environment. `ADTLib.env.install_pytorch`, which is, imports
nothing from here -- that is why `ADTLib/env/__init__.py` stays empty.
"""
import platform
import subprocess

import slicer

import ADTLib
import inspect


def conda_quote(conda, value):
    """Quote `value` only if this SlicerConda joins the command into a shell line.

    Two SlicerConda versions are in circulation and they want the opposite of
    each other. The older one builds a bash line, where a path holding a space --
    and the ';' inside a `python -c` body -- has to be quoted or the line falls
    apart. The newer one hands conda an argv list, where nothing ever strips
    those quotes: they reach PYTHONPATH and argv literally and break exactly what
    they were meant to protect. Reading the installed source tests the property
    that decides it, rather than guessing from a version number.

    Only commands going to SlicerConda come through here. The copies of
    condaRunCommand this extension carries of its own always build a shell line,
    so what they are given keeps its quotes unconditionally.
    """
    try:

        shell = "shell=True" in inspect.getsource(conda.condaRunCommand)
    except Exception:
        # Source unreadable: assume the argv contract, which is the one shipping
        # now, rather than emitting quotes that would land literally.
        shell = False
    return f'"{value}"' if shell else str(value)


def init_conda():
    """The SlicerConda entry point for this platform, or False without it."""
    try:
        import CondaSetUp  # noqa: F401  (availability probe)
    except ImportError:
        return False

    if platform.system() == "Windows":
        from CondaSetUp import CondaSetUpCallWsl

        return CondaSetUpCallWsl()
    from CondaSetUp import CondaSetUpCall

    return CondaSetUpCall()


def check_lib_wsl():
    """Whether WSL carries the system libraries the tools need to render."""
    required_libs_old = ["libxrender1", "libgl1-mesa-glx"]          # Ubuntu < 24.04
    required_libs_new = ["libxrender1", "libgl1", "libglx-mesa0"]   # Ubuntu >= 24.04

    def all_installed(libs):
        return all(
            subprocess.run(
                f'wsl -- bash -c "dpkg -l | grep {lib}"', capture_output=True, text=True
            ).stdout.encode("utf-16-le").decode("utf-8").replace("\x00", "").find(lib) >= 0
            for lib in libs
        )

    return all_installed(required_libs_old) or all_installed(required_libs_new)


def windows_to_linux_path(windows_path):
    """A Windows path as WSL sees it."""
    path = windows_path.strip().replace("\\", "/")
    if ":" in path:
        drive, path_without_drive = path.split(":", 1)
        path = "/mnt/" + drive.lower() + path_without_drive
    return path


def check_pythonpath(conda, env_name, module):
    """Whether `module` is importable by the Python of environment `env_name`."""
    conda_exe = conda.getCondaExecutable()
    command = [
        conda_exe, "run", "-n", env_name, "python", "-c",
        conda_quote(
            conda,
            f"import {module} as check;import os; print(os.path.isfile(check.__file__))"),
    ]
    return "True" in conda.condaRunCommand(command)


def give_pythonpath(conda, env_name):
    """Publish Slicer's module search paths into environment `env_name`.

    ADTLib holds no module, so its directory is never one of those search paths
    and would not reach the environment. It says where it is instead.
    """
    paths = list(slicer.app.moduleManager().factoryManager().searchPaths)
    if ADTLib.package_root() not in paths:
        paths.append(ADTLib.package_root())

    # Quoted only where a shell will strip the quotes again. They used to be
    # unconditional: under the argv-passing SlicerConda they survived into
    # PYTHONPATH, Python read each entry as a relative path and prefixed the
    # cwd, and every sys.path entry pointed nowhere.
    mnt_paths = [conda_quote(conda, windows_to_linux_path(p)) for p in paths]
    argument = [
        conda.getCondaExecutable(), "env", "config", "vars", "set",
        "-n", env_name, "PYTHONPATH=" + ":".join(mnt_paths),
    ]
    return conda.condaRunCommand(argument)
