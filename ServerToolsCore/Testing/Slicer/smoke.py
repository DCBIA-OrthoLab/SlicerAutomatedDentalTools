"""Smoke test run inside a real Slicer, on each system the client ships to.

The suites under */Testing/Python stub Slicer and Qt, so they cannot see a
module that fails to load in the real application, a panel that throws while
building its widgets, or a library that behaves differently in Slicer's own
Python. This script checks exactly those, and nothing that needs a server:

1. every module the extension declares was discovered and loaded by Slicer;
2. every scripted module's panel builds, against an unreachable server (the
   state a fresh install is in), without raising;
3. the transfer, unzip and TLS suites pass in Slicer's interpreter, with the
   requests, ssl and certifi Slicer bundles rather than the runner's.

Run it as:

    Slicer --no-splash --no-main-window --testing \
        [--additional-module-paths <module dirs>...] \
        --python-script smoke.py

The modules to expect are read from the extension's CMakeLists.txt files, so
a new module is checked without editing this script. `python smoke.py
--list-module-paths`, run with any Python, prints their folders for
`--additional-module-paths`. The exit status is 1 on any failure.
"""

import argparse
import importlib
import os
import re
import sys
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_UNIT_TESTS = os.path.join(_REPO, "ServerToolsCore", "Testing", "Python")

# The suites that need nothing but requests and the standard library, and so
# run unchanged in Slicer's interpreter. The others install Slicer and Qt
# stubs, which would shadow the real ones this script is here to exercise.
_UNIT_SUITES = ("test_transfer", "test_tls")


def _declared_modules(folder=_REPO):
    """Every module folder the build would package, as {name: folder}.

    A folder holding `<name>.py` is a module; one that only groups others (a
    folder of CLIs) is walked into, the way CMake walks its add_subdirectory.
    """
    with open(os.path.join(folder, "CMakeLists.txt"), encoding="utf-8") as handle:
        children = re.findall(r"^add_subdirectory\((\w+)\)", handle.read(), re.MULTILINE)
    modules = {}
    for child in children:
        path = os.path.join(folder, child)
        if os.path.isfile(os.path.join(path, child + ".py")):
            modules[child] = path
        else:
            modules.update(_declared_modules(path))
    return modules


def _is_cli(folder, name):
    """Slicer builds a Python CLI's panel from its XML: there is no Widget class."""
    return os.path.isfile(os.path.join(folder, name + ".xml"))


def _check_loaded(names, failures):
    import slicer

    missing = False
    for name in names:
        if hasattr(slicer.modules, name.lower()):
            print(f"[loaded] {name}")
        else:
            missing = True
            failures.append(f"{name}: not loaded by Slicer")
            print(f"[MISSING] {name}")
    if missing:
        factory = slicer.app.moduleManager().factoryManager()
        print(f"[search paths] {factory.searchPaths}")


def _check_panel(name, failures):
    """Has Slicer build the panel, as it does when a clinician opens the module.

    Slicer prints an exception raised in setup() and carries on with a half-built
    panel, so setup is wrapped first: what it raises is recorded here instead.
    """
    import slicer

    module = importlib.import_module(name)
    widget_class = getattr(module, name + "Widget", None)
    if widget_class is None:
        # A library carried as a hidden module, ServerToolsCore: nothing to show.
        print(f"[no panel] {name}")
        return
    raised = []
    original_setup = widget_class.setup

    def recorded_setup(self, *args, **kwargs):
        try:
            return original_setup(self, *args, **kwargs)
        except Exception:
            raised.append(traceback.format_exc())
            raise

    widget_class.setup = recorded_setup
    try:
        slicer.util.getModuleWidget(name)
        slicer.app.processEvents()
    except Exception:
        raised.append(traceback.format_exc())
    finally:
        widget_class.setup = original_setup
    if raised:
        failures.append(f"{name}: panel failed to build\n{raised[0]}")
        print(f"[PANEL FAILED] {name}\n{raised[0]}")
    else:
        print(f"[panel] {name}")


def _settle(timeout=60):
    """Waits for the panels' background requests to end before Slicer quits.

    Quitting while one is in flight crashes Slicer on its way out, which would
    fail this job for a reason none of the checks above is about.
    """
    import threading
    import time

    import slicer

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        slicer.app.processEvents()
        others = [t for t in threading.enumerate() if t is not threading.main_thread()]
        if not others:
            return
        time.sleep(0.1)
    print(f"[warning] threads still running at exit: {[t.name for t in others]}")


# What Slicer's launcher sets for its own libraries. The TLS fixture shells
# out to the system's openssl, which these point at Slicer's copies instead:
# on macOS that openssl then fails outright.
_LAUNCHER_ONLY = ("DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "LD_LIBRARY_PATH",
                  "OPENSSL_CONF", "OPENSSL_MODULES", "OPENSSL_ENGINES")


def _run_unit_suites(failures):
    sys.path.insert(0, _UNIT_TESTS)
    saved = {name: os.environ.pop(name) for name in _LAUNCHER_ONLY if name in os.environ}
    try:
        suite = unittest.defaultTestLoader.loadTestsFromNames(_UNIT_SUITES)
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    finally:
        os.environ.update(saved)
    if not result.wasSuccessful():
        failures.append(
            f"unit suites in Slicer's Python: {len(result.failures)} failures, "
            f"{len(result.errors)} errors"
        )


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-module-paths", action="store_true")
    args = parser.parse_args(argv)
    modules = _declared_modules()
    if args.list_module_paths:
        print("\n".join(modules.values()))
        return 0

    import slicer

    names = list(modules)
    print(f"Slicer {slicer.app.applicationVersion}, Python {sys.version.split()[0]}, {sys.platform}")
    failures = []
    _check_loaded(names, failures)
    for name in names:
        if not _is_cli(modules[name], name) and hasattr(slicer.modules, name.lower()):
            _check_panel(name, failures)
    _run_unit_suites(failures)
    _settle()

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for failure in failures:
            print(" - " + failure.splitlines()[0])
    else:
        print("\nAll smoke checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        status = main(sys.argv[1:])
    except Exception:
        traceback.print_exc()
        status = 1
    if "slicer" in sys.modules:
        sys.modules["slicer"].util.exit(status)
    else:
        sys.exit(status)
