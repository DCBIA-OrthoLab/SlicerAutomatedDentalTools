"""MRI2CBCT: register TMJ MRIs onto their CBCTs with one button.

The run, for every patient found in both folders (paired by ID, e.g. P01):
    1. prepare the MRI: optional resampling to 0.3 mm, optional left/right split
    2. orient the CBCT to Frankfort horizontal / midsagittal plane (ASO)
    3. cranial base on the CBCT (AMASSS)
    4. cranial base on each MRI joint (nnU-Net Dataset516)
    5. FPFH + RANSAC + ICP registration of the two cranial bases, with the
       FLIP ladder (more draws, 180-degree turn, other side, mirror)
Results are written in the ORIGINAL CBCT's frame. The run can stop after any
step so the result can be checked, the way AREG's review does.

The work itself lives in MRI2CBCT_core (no Slicer imports) and in the
MRI2CBCT_* CLIs; this file is the panel and the step chaining.
"""
import csv
import glob
import json
import os
import sys
import time

import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry, so walk up to the
# holder directory and add it, as the other modules do.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

from MRI2CBCT_core import plan, pairing, layout, env, requirements

SETTINGS = "MRI2CBCT/"


class MRI2CBCT(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MRI2CBCT"
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = []
        self.parent.contributors = ["DCBIA-OrthoLab (University of Michigan, UNC)"]
        self.parent.helpText = (
            "Registers TMJ MRIs onto their CBCTs in one run: MRI preparation, CBCT "
            "orientation (ASO), cranial base on both scans (AMASSS, Dataset516), then a "
            "global surface registration with automatic flip recovery. Choose an MRI "
            "folder and a CBCT folder; patients are paired by ID (P01, B038...).")
        self.parent.acknowledgementText = ""


def downloads_folder():
    docs = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
    return os.path.join(docs, slicer.app.applicationName + "Downloads")


class MRI2CBCTWidget(ScriptedLoadableModuleWidget):

    # ------------------------------------------------------------------ panel
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # The CLIs have no Qt to ask; they read the downloads folder from here.
        os.environ["MRI2CBCT_DOWNLOADS"] = downloads_folder()
        self.logic = MRI2CBCTLogic()
        self.queue, self.done_steps, self.node, self.observer = [], [], None, None
        self.review_items, self.review_index, self.review_nodes = [], 0, []
        self.running = False

        # one-time setup: a single line, gone once everything is installed
        setupRow = qt.QHBoxLayout()
        self.setupLabel = qt.QLabel("")
        self.setupLabel.wordWrap = True
        self.setupButton = qt.QPushButton("Set up this computer")
        self.setupButton.toolTip = ("Install the Python packages and download the models MRI2CBCT "
                                    "needs. Done once per computer; Run does it too if needed.")
        self.setupButton.clicked.connect(self.onSetup)
        setupRow.addWidget(self.setupLabel, 1)
        setupRow.addWidget(self.setupButton)
        self.layout.addLayout(setupRow)
        self._refresh_setup()

        # inputs
        box = ctk.ctkCollapsibleButton()
        box.text = "Inputs"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.mriPath = self._dir_edit("mri")
        self.cbctPath = self._dir_edit("cbct")
        self.outPath = self._dir_edit("out")
        form.addRow("MRI folder:", self.mriPath)
        form.addRow("CBCT folder:", self.cbctPath)
        form.addRow("Output folder:", self.outPath)
        hint = qt.QLabel("Patients are paired by the ID at the start of the file name "
                         "(P01, B038...). One CBCT per patient; one or two MRIs.")
        hint.wordWrap = True
        form.addRow(hint)

        # the two choices
        box = ctk.ctkCollapsibleButton()
        box.text = "Options"
        self.layout.addWidget(box)
        v = qt.QVBoxLayout(box)
        g = qt.QGroupBox("MRI resolution")
        gl = qt.QVBoxLayout(g)
        self.resampleYes = qt.QRadioButton("Resample the MRI to 0.3 mm (CBCT resolution)")
        self.resampleNo = qt.QRadioButton("No resampling")
        self.resampleYes.checked = True
        gl.addWidget(self.resampleYes)
        gl.addWidget(self.resampleNo)
        v.addWidget(g)
        g = qt.QGroupBox("Left / right")
        gl = qt.QFormLayout(g)
        self.separateYes = qt.QRadioButton("Separate left and right (both joints in one MRI)")
        self.separateNo = qt.QRadioButton("No separation (one joint per MRI, or whole head)")
        self.separateYes.checked = True
        gl.addRow(self.separateYes)
        gl.addRow(self.separateNo)
        self.acqSpacing = qt.QDoubleSpinBox()
        self.acqSpacing.setRange(0.5, 10.0)
        self.acqSpacing.setSingleStep(0.5)
        self.acqSpacing.setDecimals(2)
        self.acqSpacing.value = 3.0
        self.acqSpacing.suffix = " mm"
        self.acqSpacing.toolTip = (
            "Slice spacing used at acquisition. A merged bilateral slab is two stacks "
            "with a gap between the joints; NIfTI cannot store the gap, so the true "
            "spacing is needed to rebuild each joint.")
        gl.addRow("Acquisition slice spacing:", self.acqSpacing)
        self.separateYes.toggled.connect(lambda on: self.acqSpacing.setEnabled(on))
        v.addWidget(g)

        adv = ctk.ctkCollapsibleButton()
        adv.text = "Advanced"
        adv.collapsed = True
        v.addWidget(adv)
        af = qt.QFormLayout(adv)
        self.folds = qt.QComboBox()
        self.folds.addItems(["All 5 folds (recommended)", "1 fold (faster, for computers without a GPU)"])
        af.addRow("MRI segmentation:", self.folds)
        self.onCbctGrid = qt.QCheckBox("Put the registered MRI on the CBCT's voxel grid (interpolated)")
        self.onCbctGrid.toolTip = (
            "Off: the registered MRI keeps its own voxels and only its position changes; "
            "it still opens on top of the CBCT. On: it is resampled onto the CBCT grid.")
        self.onCbctGrid.checked = False
        af.addRow(self.onCbctGrid)
        self.keepSteps = qt.QCheckBox("Keep the intermediate files (_steps folder)")
        self.keepSteps.toolTip = ("The oriented CBCT, both segmentations before registration, "
                                  "and the prepared MRIs. Deleted after a successful run otherwise.")
        self.keepSteps.checked = False
        af.addRow(self.keepSteps)
        self.workers = qt.QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.value = 0
        self.workers.specialValueText = "automatic"
        af.addRow("Registration workers:", self.workers)

        # review
        box = ctk.ctkCollapsibleButton()
        box.text = "Review"
        box.collapsed = True
        self.layout.addWidget(box)
        rv = qt.QVBoxLayout(box)
        self.reviewEnable = qt.QCheckBox("Pause during the run to check the results")
        rv.addWidget(self.reviewEnable)
        self.reviewBoxes = {}
        remembered = set(str(qt.QSettings().value(SETTINGS + "review", "")).split(","))
        for sid, title, hint in plan.REVIEW_STEPS:
            cb = qt.QCheckBox(title)
            cb.toolTip = hint
            cb.checked = sid in remembered
            cb.toggled.connect(lambda _c: self._save_review_choice())
            rv.addWidget(cb)
            self.reviewBoxes[sid] = cb
        self.reviewEnable.checked = qt.QSettings().value(SETTINGS + "review_on", "false") == "true"
        self.reviewEnable.toggled.connect(self._on_review_enable)
        self._on_review_enable(self.reviewEnable.checked)

        # run
        self.runButton = qt.QPushButton("Run")
        self.runButton.setStyleSheet("font-weight: bold; padding: 6px;")
        self.layout.addWidget(self.runButton)
        self.runButton.clicked.connect(self.onRun)
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.visible = False
        self.cancelButton.clicked.connect(self.onCancel)
        self.layout.addWidget(self.cancelButton)
        self.progress = qt.QProgressBar()
        self.progress.visible = False
        self.layout.addWidget(self.progress)
        self.status = qt.QLabel("")
        self.status.wordWrap = True
        self.layout.addWidget(self.status)

        # the pause panel
        self.reviewPanel = qt.QFrame()
        self.reviewPanel.setFrameShape(qt.QFrame.StyledPanel)
        self.reviewPanel.visible = False
        pv = qt.QVBoxLayout(self.reviewPanel)
        self.reviewMessage = qt.QLabel("")
        self.reviewMessage.wordWrap = True
        self.reviewMessage.textFormat = qt.Qt.RichText
        pv.addWidget(self.reviewMessage)
        nav = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton("◀ Previous")
        self.itemLabel = qt.QLabel("")
        self.itemLabel.alignment = qt.Qt.AlignCenter
        self.nextButton = qt.QPushButton("Next ▶")
        nav.addWidget(self.prevButton)
        nav.addWidget(self.itemLabel)
        nav.addWidget(self.nextButton)
        pv.addLayout(nav)
        self.swapButton = qt.QPushButton("Swap left / right for this patient")
        pv.addWidget(self.swapButton)
        self.continueButton = qt.QPushButton("Next step")
        self.continueButton.setStyleSheet("font-weight: bold;")
        pv.addWidget(self.continueButton)
        self.layout.addWidget(self.reviewPanel)
        self.prevButton.clicked.connect(lambda: self._show_item(self.review_index - 1))
        self.nextButton.clicked.connect(lambda: self._show_item(self.review_index + 1))
        self.swapButton.clicked.connect(self.onSwapSides)
        self.continueButton.clicked.connect(self.onContinue)

        logBox = ctk.ctkCollapsibleButton()
        logBox.text = "Log"
        logBox.collapsed = True
        self.layout.addWidget(logBox)
        lv = qt.QVBoxLayout(logBox)
        self.log = qt.QPlainTextEdit()
        self.log.readOnly = True
        self.log.setMinimumHeight(200)
        lv.addWidget(self.log)
        self.layout.addStretch(1)

    def _refresh_setup(self):
        try:
            missing = self.logic.quick_status(downloads_folder())
        except Exception as e:                      # never let the check block the panel
            missing = [f"(could not check: {e})"]
        if getattr(self.logic, "restart_needed", False):
            self.setupLabel.text = "<b>Restart Slicer to finish the setup</b> (PyTorch was installed)."
            self.setupButton.visible = False
        elif missing:
            self.setupLabel.text = ("<b>One-time setup needed</b> (packages and models, "
                                    "about 5 GB): " + ", ".join(missing[:6])
                                    + (", ..." if len(missing) > 6 else ""))
            self.setupButton.visible = True
        else:
            self.setupLabel.text = "✓ This computer is ready"
            self.setupButton.visible = False

    def onSetup(self):
        self.setupButton.enabled = False
        try:
            ok = self.logic.ensure_ready(downloads_folder(), self.parent)
        finally:
            self.setupButton.enabled = True
        self._refresh_setup()
        if ok:
            slicer.util.infoDisplay("MRI2CBCT is set up on this computer.")

    def _dir_edit(self, key):
        e = ctk.ctkPathLineEdit()
        e.filters = ctk.ctkPathLineEdit.Dirs
        e.settingKey = SETTINGS + key
        return e

    def _on_review_enable(self, on):
        for cb in self.reviewBoxes.values():
            cb.enabled = on
        qt.QSettings().setValue(SETTINGS + "review_on", "true" if on else "false")

    def _save_review_choice(self):
        chosen = [k for k, cb in self.reviewBoxes.items() if cb.checked]
        qt.QSettings().setValue(SETTINGS + "review", ",".join(chosen))

    def _say(self, text):
        self.log.appendPlainText(text)
        try:
            with open(os.path.join(self.output, layout.STEPS, "run.log"), "a") as f:
                f.write(text + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ run
    def onRun(self):
        mri, cbct, out = self.mriPath.currentPath, self.cbctPath.currentPath, self.outPath.currentPath
        for label, p in (("MRI", mri), ("CBCT", cbct)):
            if not p or not os.path.isdir(p):
                slicer.util.errorDisplay(f"Choose the {label} folder.")
                return
        if not out:
            slicer.util.errorDisplay("Choose an output folder.")
            return
        if pairing.folders_swapped(mri, cbct):
            if not slicer.util.confirmYesNoDisplay(
                    "The two folders look swapped: the MRI folder holds CBCT scans, and the "
                    f"CBCT folder holds MRIs.\n\nMRI folder: {mri}\nCBCT folder: {cbct}\n\n"
                    "Swap them and run?"):
                return
            mri, cbct = cbct, mri
            self.mriPath.currentPath, self.cbctPath.currentPath = mri, cbct
        for e in (self.mriPath, self.cbctPath, self.outPath):
            e.addCurrentPathToHistory()
        patients, problems = pairing.pair_folders(mri, cbct)
        if not patients:
            slicer.util.errorDisplay(
                "No patient has both an MRI and a CBCT. None of the IDs in one folder "
                "appear in the other - check that both folders are the ones you meant.\n\n"
                f"MRI folder:  {mri}\nCBCT folder: {cbct}\n\n" + "\n".join(problems[:20]))
            return
        downloads = downloads_folder()
        ready = self.logic.ensure_ready(downloads, self.parent)
        self._refresh_setup()
        if not ready:
            return
        if problems and not slicer.util.confirmOkCancelDisplay(
                f"{len(patients)} patient(s) will run. These files are left out:\n\n"
                + "\n".join(problems[:20]) + ("\n..." if len(problems) > 20 else "")):
            return

        os.makedirs(out, exist_ok=True)
        self.output = out
        self.log.clear()
        folds = (0, 1, 2, 3, 4) if self.folds.currentIndex == 0 else (0,)
        self.steps = plan.build(mri, cbct, out, self.resampleYes.checked, self.separateYes.checked,
                                self.acqSpacing.value, folds, self.onCbctGrid.checked,
                                self.workers.value, downloads, env.dataset516())
        self.queue = list(self.steps)
        self.done_steps = []
        self.pause_at = ({k for k, cb in self.reviewBoxes.items() if cb.checked}
                         if self.reviewEnable.checked else set())
        self.started = time.time()
        self.running = True
        self._running_ui(True)
        self._say(f"{len(patients)} patient(s): {', '.join(patients)}")
        for p in problems:
            self._say("left out: " + p)
        self._start_next()

    def _running_ui(self, on):
        self.runButton.enabled = not on
        self.cancelButton.visible = on
        self.progress.visible = on

    def _start_next(self):
        if not self.queue:
            self._finish()
            return
        step = self.queue.pop(0)
        self.current = step
        module = getattr(slicer.modules, step["cli"].lower(), None)
        if module is None:
            self._fail(f"The CLI module {step['cli']} is not loaded in Slicer.")
            return
        n = len(self.steps)
        i = len(self.done_steps)
        self.status.text = f"Step {i + 1} of {n}: {step['title']}"
        self._say(f"\n=== {step['title']}")
        self.handled = False
        self.node = slicer.cli.run(module, None, dict(step["params"]), wait_for_completion=False)
        self.observer = self.node.AddObserver("ModifiedEvent", self._on_cli)

    def _on_cli(self, caller, _event):
        if caller is not self.node or self.handled:
            return
        n = len(self.steps)
        frac = (len(self.done_steps) + caller.GetProgress() / 100.0) / n
        self.progress.value = int(100 * frac)
        if caller.IsBusy():
            return
        self.handled = True
        caller.RemoveObserver(self.observer)
        status = caller.GetStatus()
        out = caller.GetOutputText() or ""
        if out.strip():
            self._say("\n".join(l for l in out.splitlines() if "filter-progress" not in l))
        if status == caller.Completed:
            self.done_steps.append(self.current)
            # load nothing inside the VTK callback; let the event loop turn first
            if self.current.get("review") in self.pause_at:
                qt.QTimer.singleShot(0, self._begin_review)
            else:
                qt.QTimer.singleShot(0, self._start_next)
        elif status == caller.Cancelled:
            self._stop("Cancelled.")
        else:
            err = caller.GetErrorText() or "(no error text)"
            self._say(err)
            self._fail(f"{self.current['title']} failed.\n\n{err[-1500:]}")

    def onCancel(self):
        if self.node is not None and self.node.IsBusy():
            self.node.Cancel()
        self._clear_review()
        self._stop("Cancelled.")

    def _stop(self, message):
        self.running = False
        self.queue = []
        self._running_ui(False)
        self.status.text = message

    def _fail(self, message):
        self._stop("Stopped: " + message.splitlines()[0])
        slicer.util.errorDisplay(message)

    def _finish(self):
        self._stop("")
        minutes = (time.time() - self.started) / 60.0
        summary = os.path.join(self.output, "summary.csv")
        lines = []
        if os.path.exists(summary):
            with open(summary) as f:
                for r in csv.DictReader(f):
                    tag = f" {r['tag']}" if r.get("tag") else ""
                    lines.append(f"{r['patient']} {r['side']}: {r.get('qc') or r['status']}{tag}")
        self.status.text = f"Done in {minutes:.1f} min. Results in {self.output}"
        self._say(f"\nDone in {minutes:.1f} min")
        steps = os.path.join(self.output, layout.STEPS)
        if not self.keepSteps.checked and os.path.isdir(steps):
            import shutil
            shutil.rmtree(steps, ignore_errors=True)   # this module's own scratch
        slicer.util.infoDisplay(f"Done in {minutes:.1f} min.\n\n" + "\n".join(lines))

    # ------------------------------------------------------------------ review
    def _begin_review(self):
        sid = self.current["review"]
        try:
            self.review_items = self.logic.review_items(self.output, sid)
        except Exception as e:
            self._say(f"Nothing to review after {self.current['title']}: {e}")
            self.review_items = []
        if not self.review_items:
            self._start_next()
            return
        self.review_step = sid
        self.reviewPanel.visible = True
        self.swapButton.visible = sid == "prepare"
        self.status.text = f"Paused after: {self.current['title']}"
        self._show_item(0)

    def _show_item(self, index):
        if not self.review_items:
            return
        index = max(0, min(index, len(self.review_items) - 1))
        self.review_index = index
        item = self.review_items[index]
        self._clear_review_nodes()
        try:
            self.review_nodes = self.logic.show(item)
        except Exception as e:
            self._say(f"Could not load {item['title']}: {e}")
            self.review_nodes = []
        hint = dict((k, h) for k, _t, h in plan.REVIEW_STEPS)[self.review_step]
        self.reviewMessage.text = (f"<b>{item['title']}</b><br/>{item.get('text', '')}"
                                   f"<br/><i>{hint}</i>")
        self.itemLabel.text = f"{index + 1} / {len(self.review_items)}"
        self.prevButton.enabled = index > 0
        self.nextButton.enabled = index < len(self.review_items) - 1

    def onSwapSides(self):
        item = self.review_items[self.review_index]
        pid = item["patient"]
        msg = self.logic.swap_sides(self.output, pid)
        self._say(msg)
        self.review_items = self.logic.review_items(self.output, "prepare")
        self._show_item(self.review_index)

    def onContinue(self):
        self._clear_review()
        self._start_next()

    def _clear_review_nodes(self):
        for n in self.review_nodes:
            if n is not None and slicer.mrmlScene.IsNodePresent(n):
                slicer.mrmlScene.RemoveNode(n)
        self.review_nodes = []

    def _clear_review(self):
        self._clear_review_nodes()
        self.review_items = []
        self.reviewPanel.visible = False

    def cleanup(self):
        if self.node is not None and self.observer is not None:
            self.node.RemoveObserver(self.observer)


class MRI2CBCTLogic(ScriptedLoadableModuleLogic):
    """First-run setup, what each pause shows, and the side swap.
    The run itself is in the CLIs."""

    # ------------------------------------------------------------ first-run setup
    def quick_status(self, downloads):
        """What is missing, checked in well under a second: installed packages and
        download markers only. No torch import, no GPU query - that is ensure_ready's job."""
        import importlib.util
        from ADTLib.env.deps import check_lib_installed
        from ADTLib import testdata
        missing = [n for n, _v in requirements.missing_packages(check_lib_installed)]
        if not check_lib_installed("torch"):
            missing.append("torch")
        if importlib.util.find_spec("open3d") is None and \
                not os.path.isdir(os.path.join(env.py_deps(), "open3d")):
            missing.append("open3d")
        missing += [d[0] for d in requirements.downloads_plan(downloads, env.DATASET516_URL,
                                                              env.dataset516_root())
                    if not testdata.is_present(os.path.join(d[2], d[3]))]
        return missing

    def ensure_ready(self, downloads, parent=None):
        """Install what is missing, download what is missing; False to abort the run."""
        from ADTLib.env.deps import check_lib_installed, requirement
        from ADTLib.env.cuda import torch_install_arguments
        from ADTLib import testdata

        packages = requirements.missing_packages(check_lib_installed)
        try:
            torch_args = torch_install_arguments()
        except Exception as e:                       # no answer from the GPU: leave torch be
            print(f"MRI2CBCT: could not decide the torch build ({e})")
            torch_args = None
        try:
            env.ensure_open3d()
            need_open3d = False
        except ImportError:
            need_open3d = True

        if packages or torch_args or need_open3d:
            lines = [requirement(n, v) for n, v in packages]
            if torch_args:
                lines.append(f"PyTorch for this GPU ({torch_args})")
            if need_open3d:
                lines.append(env.OPEN3D)
            if not slicer.util.confirmYesNoDisplay(
                    "MRI2CBCT needs these Python packages, which are not installed yet:\n\n  "
                    + "\n  ".join(lines)
                    + "\n\nInstall them now? This takes several minutes and happens once.",
                    parent=parent):
                return False
            try:
                with slicer.util.WaitCursor():
                    self._install(packages, torch_args, need_open3d)
            except Exception as e:
                slicer.util.errorDisplay(f"Installing the packages failed:\n\n{e}")
                return False
            if torch_args:
                self.restart_needed = True
                slicer.util.infoDisplay(
                    "PyTorch was installed or replaced. Restart Slicer, then press Run again.",
                    parent=parent)
                return False

        todo = [d for d in requirements.downloads_plan(downloads, env.DATASET516_URL,
                                                       env.dataset516_root())
                if not testdata.is_present(os.path.join(d[2], d[3]))]
        if todo:
            if not slicer.util.confirmYesNoDisplay(
                    "MRI2CBCT needs these models, which are not on this computer yet:\n\n  "
                    + "\n  ".join(d[0] for d in todo)
                    + "\n\nDownload them now? This happens once.", parent=parent):
                return False
            for i, (label, url, root, name) in enumerate(todo):
                try:
                    testdata.ensure_with_progress(
                        url, root, name, parent=parent,
                        title=f"Downloading {label} ({i + 1}/{len(todo)})")
                except Exception as e:
                    slicer.util.errorDisplay(f"Downloading {label} failed:\n{url}\n\n{e}")
                    return False
        return True

    def _install(self, packages, torch_args, need_open3d):
        from slicer.util import pip_install
        from ADTLib.env.deps import requirement
        if torch_args:                          # first: monai would pull PyPI's default build
            pip_install(torch_args)
        for name, spec in packages:
            pip_install(requirement(name, spec))
        if need_open3d:
            pip_install(" ".join(requirements.OPEN3D_RUNTIME))
            if sys.platform.startswith("linux"):
                glibc = os.confstr("CS_GNU_LIBC_VERSION") or "glibc 0"
                version = tuple(int(x) for x in glibc.split()[-1].split(".")[:2])
                if version < (2, 31):
                    raise RuntimeError(f"Open3D needs glibc 2.31 or newer; this system has {glibc}.")
                # Slicer's pip caps manylinux at 2_28; the wheel is 2_31 and runs here.
                os.makedirs(env.py_deps(), exist_ok=True)
                pip_install(f"--no-deps --only-binary :all: --platform {env.OPEN3D_LINUX_TAG} "
                            f"--python-version {sys.version_info.major}.{sys.version_info.minor} "
                            f"--implementation cp --abi {env.PY_ABI} --upgrade "
                            f"--target {env.py_deps()} {env.OPEN3D}")
            else:
                pip_install(env.OPEN3D)
        import importlib.metadata as md
        if md.version("numpy") != requirements.NUMPY:     # last: anything above may pull numpy 2
            pip_install(f"numpy=={requirements.NUMPY}")

    def review_items(self, output, sid):
        m = layout.read_manifest(output)
        items = []
        for pid, p in m["patients"].items():
            if sid in ("orient", "cbct_seg"):
                it = {"patient": pid, "title": f"{pid} - CBCT",
                      "background": layout.cbct_oriented(output, pid)}
                if sid == "cbct_seg":
                    it["labels"] = [layout.cbct_seg(output, pid)]
                items.append(it)
                continue
            if sid in ("prepare", "mri_seg"):
                for j in p["joints"]:
                    s = j["side"]
                    it = {"patient": pid, "title": f"{pid} - joint {_side_name(s)}",
                          "background": layout.joint_mri(output, pid, s),
                          "text": f"From {os.path.basename(j['source'])}: {j['note']}"}
                    if sid == "mri_seg":
                        it["labels"] = [layout.joint_seg(output, pid, s)]
                    items.append(it)
                continue
            if sid == "register":
                for r in self._summary(output):
                    if r["patient"] != pid or r["status"] != "ok":
                        continue
                    tag = f"_{r['tag']}" if r["tag"] else ""
                    stem = os.path.join(output, pid, f"{pid}_{r['side']}_MRI")
                    text = (f"QC <b>{r['qc']}</b>{(' ' + r['tag']) if r['tag'] else ''} - "
                            f"fitness {r['fitness']}, rotation {r['rotation_deg']} deg, "
                            f"mask {r['mask_cm3']} cm3")
                    if r["ladder"]:
                        text += f"<br/>FLIP recovery tried: {r['ladder']}; resolved by: {r['resolved_by']}"
                    items.append({"patient": pid,
                                  "title": f"{pid} - joint {_side_name(r['side'])} on its CBCT",
                                  "background": p["cbct"],
                                  "foreground": f"{stem}_REG{tag}.nii.gz",
                                  "labels": [f"{stem}_MASK{tag}.nii.gz"],
                                  "text": text})
        return items

    @staticmethod
    def _summary(output):
        path = os.path.join(output, "summary.csv")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return list(csv.DictReader(f))

    def show(self, item):
        nodes = []
        bg = slicer.util.loadVolume(item["background"], {"show": False})
        nodes.append(bg)
        fg = None
        tnode = None
        if item.get("transform"):
            tnode = slicer.util.loadTransform(item["transform"])
            nodes.append(tnode)
        if item.get("foreground"):
            fg = slicer.util.loadVolume(item["foreground"], {"show": False})
            nodes.append(fg)
            if tnode:
                fg.SetAndObserveTransformNodeID(tnode.GetID())
        lab = None
        for p in item.get("labels", []):
            if os.path.exists(p):
                lab = slicer.util.loadLabelVolume(p, {"show": False})
                nodes.append(lab)
                if tnode:
                    lab.SetAndObserveTransformNodeID(tnode.GetID())
        slicer.util.setSliceViewerLayers(background=bg, foreground=fg, label=lab,
                                         foregroundOpacity=0.5, labelOpacity=0.6)
        if lab is not None:
            for name in ("Red", "Yellow", "Green"):
                comp = slicer.app.layoutManager().sliceWidget(name).sliceLogic().GetSliceCompositeNode()
                comp.SetLabelOpacity(0.6)
                slicer.app.layoutManager().sliceWidget(name).sliceLogic().GetLabelLayer() \
                    .GetSliceNode().SetUseLabelOutline(True)
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        focus = fg if fg is not None else bg
        slicer.util.resetSliceViews()
        if focus is not None:
            b = [0.0] * 6
            focus.GetRASBounds(b)
            centre = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
            for name in ("Red", "Yellow", "Green"):
                slicer.app.layoutManager().sliceWidget(name).sliceLogic().GetSliceNode() \
                    .JumpSliceByCentering(*centre)
        return nodes

    def swap_sides(self, output, pid):
        """Exchange L and R for one patient: files and manifest."""
        m = layout.read_manifest(output)
        p = m["patients"][pid]
        swap = {"L": "R", "R": "L"}
        moves = [j for j in p["joints"] if j["side"] in swap]
        if not moves:
            return f"{pid}: nothing to swap (the MRI is kept whole)"
        tmp = []
        for j in moves:
            src = layout.joint_mri(output, pid, j["side"])
            t = src + ".swap"
            os.replace(src, t)
            tmp.append((t, layout.joint_mri(output, pid, swap[j["side"]])))
            j["side"] = swap[j["side"]]
            j["note"] += "; side swapped by hand at review"
        for t, dst in tmp:
            os.replace(t, dst)
        layout.write_manifest(output, m)
        return f"{pid}: left and right swapped by hand"


def _side_name(s):
    return {"L": "LEFT", "R": "RIGHT", "B": "both (whole volume)"}.get(s, s)


class MRI2CBCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.test_plan_builds()

    def test_plan_builds(self):
        steps = plan.build("/m", "/c", "/o", True, True, 3.0, (0,), True, 0,
                           downloads_folder(), env.dataset516())
        self.assertEqual([s["cli"] for s in steps][0], "MRI2CBCT_PREPARE")
        for s in steps:
            self.assertIsNotNone(getattr(slicer.modules, s["cli"].lower(), None), s["cli"])
        self.delayDisplay("MRI2CBCT step list builds, and every CLI is loaded")
