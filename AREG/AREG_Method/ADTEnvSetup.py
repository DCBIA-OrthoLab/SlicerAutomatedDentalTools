"""One startup check for the Conda environment that every IOS pipeline shares.

ASO, AREG, ALI, FlexReg and DOCShapeAXI all drive their IOS tools - crown
segmentation, landmark identification, registration - through a single Conda
environment named `shapeaxi`. Nothing used to look for that environment before a
run: the check lives in each widget's `onCheckRequirements`, which is reached
only from the Run button and, in AREG, only for the IOS type. A fresh Slicer
therefore gave no sign that the environment was missing until a run was
launched, and `run_conda_tool` answers a missing `dentalmodelseg` with a log
line and a `return`, so the segmentation step was skipped and the run carried on
with nothing to show for it.

This module asks once, at startup, and builds the environment on the spot when
the user agrees. The environment it looks after belongs to the whole extension;
it is still hosted by AREG only because the steps it drives are AREG's own
`install_shapeaxi` and `install_pytorch3d`. The pip step itself has moved to
`ADTLib.env.install_pytorch`, so this module is a candidate to follow it into
`ADTLib` once those two steps are shared as well.

It sits in `AREG_Method` rather than next to `AREG.py`: Slicer probes every file
at the root of a module path for a scripted module class, and a helper left
there fails that probe loudly at each startup.
"""

import logging
import os
import platform
import time

import qt
import slicer

logger = logging.getLogger("ADTEnvSetup")

ENV_NAME = "shapeaxi"

# Answering "Never ask again" is remembered per Slicer installation, not per
# machine: a Conda set up for one installation is not shared with the next one
# (SlicerConda keys its path by installation), so a new Slicer has to ask again.
SKIP_KEY = "AutomatedDentalTools/skipSharedEnvironmentCheck"

# The environment specification is deliberately NOT repeated here. installEnvironment
# drives AREG's own install_shapeaxi and install_pytorch3d, so whatever those two
# install is what this module builds, and the two can never drift apart. A copy of
# the specification used to sit here, promising to be "kept identical"; PR #254 moved
# torch out of condaCreateEnv and into ADTLib.env.install_pytorch, and this copy was
# not updated - it was never read, so nothing broke, which is exactly why it went
# unnoticed. Anything this module needs to know about the specification, it asks the
# module that owns it.


def _settings():
    """Settings of the running installation, falling back to the user's own."""
    revision_user_settings = getattr(slicer.app, "revisionUserSettings", None)
    settings = revision_user_settings() if revision_user_settings else None
    if settings and settings.isWritable():
        return settings
    return slicer.app.userSettings()


def _condaCall():
    """The SlicerConda entry point, or None when the extension is not there."""
    try:
        import CondaSetUp
    except ImportError:
        return None
    if platform.system() == "Windows":
        return CondaSetUp.CondaSetUpCallWsl()
    return CondaSetUp.CondaSetUpCall()


def environmentIsPresent(conda):
    """Tells whether `shapeaxi` sits where this Slicer's Conda keeps its envs.

    A directory test rather than `conda info --envs`: this runs at every startup
    and must not pay for a subprocess. Under WSL the envs are not reachable that
    way, so there the question goes to Conda itself.
    """
    if platform.system() == "Windows":
        return conda.condaTestEnv(ENV_NAME) is True
    conda_path = conda.getCondaPath()
    if not conda_path or conda_path == "None":
        return False
    return os.path.isdir(os.path.join(conda_path, "envs", ENV_NAME))


def _formatTime(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return "{:d}:{:02d}".format(minutes, seconds)


def _waitForThread(logic, title, message):
    """Spins the event loop until the worker thread of `logic` is done.

    Nothing here writes to the log while the worker runs. Slicer drains its own
    stdout from the event loop, and the worker fills that pipe: a line written
    from this side would block in the write, stop the draining and freeze the
    run for good, panel up and no error shown. Progress goes to the dialog,
    which is a widget, not a stream.
    """
    dialog = qt.QProgressDialog(message, "Cancel", 0, 0)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(qt.Qt.NonModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.show()

    start = time.time()
    finished = True
    while logic.process.is_alive():
        slicer.app.processEvents()
        dialog.setLabelText("{}\nelapsed: {}".format(message, _formatTime(time.time() - start)))
        if dialog.wasCanceled:
            # The worker holds a conda subprocess that cannot be taken back, so
            # cancelling stops the waiting, not the installation.
            finished = False
            break
        time.sleep(0.05)

    dialog.close()
    return finished


def _clickedRole(box):
    """The role of the button that closed `box`, by role rather than by identity.

    A dialog dismissed from the window frame has no clicked button at all, and
    that counts as a refusal here, not as an acceptance.
    """
    clicked = box.clickedButton()
    if clicked is None:
        return qt.QMessageBox.RejectRole
    return box.buttonRole(clicked)


def _informCondaMissing():
    box = qt.QMessageBox()
    box.setWindowTitle("Automated Dental Tools")
    box.setTextFormat(qt.Qt.RichText)
    box.setText(
        "Conda is not set up, so the IOS tools of ASO, AREG, ALI, FlexReg and "
        "DOCShapeAXI cannot run.<br><br>"
        "Open the <b>CondaSetUp</b> module of SlicerConda to install it, then "
        "restart Slicer."
    )
    box.addButton("OK", qt.QMessageBox.AcceptRole)
    box.addButton("Never ask again", qt.QMessageBox.DestructiveRole)
    box.exec_()
    if _clickedRole(box) == qt.QMessageBox.DestructiveRole:
        _settings().setValue(SKIP_KEY, "true")


def _askToInstall():
    """Offers to build the environment. Returns True when the user accepts."""
    box = qt.QMessageBox()
    box.setWindowTitle("Automated Dental Tools")
    box.setTextFormat(qt.Qt.RichText)
    box.setText(
        "The <b>shapeaxi</b> Conda environment does not exist.<br><br>"
        "Without it, the IOS pipelines of ASO, AREG, ALI, FlexReg and "
        "DOCShapeAXI cannot segment the crowns.<br><br>"
        "Install it now? This downloads several gigabytes - torch, pytorch3d "
        "and shapeaxi - and usually takes 15 to 30 minutes."
    )
    box.addButton("Install now", qt.QMessageBox.AcceptRole)
    box.addButton("Later", qt.QMessageBox.RejectRole)
    box.addButton("Never ask again", qt.QMessageBox.DestructiveRole)
    box.exec_()

    role = _clickedRole(box)
    if role == qt.QMessageBox.DestructiveRole:
        _settings().setValue(SKIP_KEY, "true")
        return False
    return role == qt.QMessageBox.AcceptRole


def installEnvironment():
    """Builds `shapeaxi` and its libraries, the way a module's Run button would.

    The steps are AREG's own, so that an environment built from here and one
    built from a module are the same environment.
    """
    from AREG import AREGLogic

    logic = AREGLogic()

    logic.install_shapeaxi()
    if not _waitForThread(
        logic,
        "Automated Dental Tools",
        "Creating the shapeaxi environment.",
    ):
        return None

    # pytorch3d has no distribution on PyPI, so it is built against the torch
    # that was just installed; shapeaxi rides along with it, which is why it is
    # never asked for on its own.
    if "Error" in logic.check_if_pytorch3d():
        logic.install_pytorch3d()
        if not _waitForThread(
            logic,
            "Automated Dental Tools",
            "Installing pytorch3d and shapeaxi.",
        ):
            return None

    return "Error" not in logic.check_if_pytorch3d()


def _reportOutcome(succeeded):
    box = qt.QMessageBox()
    box.setWindowTitle("Automated Dental Tools")
    if succeeded is None:
        box.setText(
            "The installation is still running in the background.\n\n"
            "Watch it from the Python console, and let it finish before "
            "launching a run."
        )
    elif succeeded:
        box.setText("The shapeaxi environment is ready.")
    else:
        box.setText(
            "The shapeaxi environment was created, but pytorch3d or shapeaxi "
            "did not install.\n\n"
            "The Python console holds the output of the installation."
        )
    box.exec_()


def checkAtStartup():
    """Looks for the shared environment once, and offers to build it if absent."""
    if slicer.app.commandOptions().noMainWindow or slicer.util.mainWindow() is None:
        return
    if _settings().value(SKIP_KEY, "") == "true":
        return

    conda = _condaCall()
    if conda is None:
        # SlicerConda is a declared dependency of the modules; its absence is
        # already reported by the extension manager, not this check's business.
        return

    conda_path = conda.getCondaPath()
    if not conda_path or conda_path == "None":
        _informCondaMissing()
        return

    if environmentIsPresent(conda):
        logger.info("The shapeaxi environment is in place")
        return

    if not _askToInstall():
        return

    _reportOutcome(installEnvironment())
