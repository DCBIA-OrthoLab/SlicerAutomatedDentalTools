import json
import os
import shutil
import sys
import time

import qt
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))
# The names the CLI shares with this panel (TMJ_Crop_utils). An installed build
# has the package on sys.path; a source tree finds it next to this folder.
_cli_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                           "TMJ_Crop_CLI")
if os.path.isdir(_cli_folder) and _cli_folder not in sys.path:
    sys.path.append(_cli_folder)

from ADTLib.logging_setup import get_logger
from ADTLib.format import format_timer
from ADTLib.theming import apply_theme
from ADTLib.env.deps import check_lib_installed, requirement
from ADTLib.env.cuda import torch_install_arguments
from ADTLib.model_registry import (
    ASO_CBCT_GOLD, ASO_CBCT_TEST_FILES, AMASSS_CBCT)
from ADTLib.testdata import ensure_with_progress, is_present, TestDataError
from TMJ_Crop_utils import catalogs

logger = get_logger("TMJ_Crop")

SCAN_EXTENSIONS = (".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz")

# What the extension already offers for each of the three things the chain needs.
# The same files ASO and AMASSS download, into the same folders, so a machine
# that ran either of them has nothing left to fetch.
REFERENCE_NAME = "Frankfurt Horizontal and Midsagittal Plane"
REFERENCE_URL = f"{ASO_CBCT_GOLD}/Frankfurt_Horizontal_Midsagittal_Plane.zip"
ALI_MODELS_URL = "https://github.com/lucanchling/ALI_CBCT/releases/download/models_v01/"
AMASSS_MODELS_URL = f"{AMASSS_CBCT}/AMASSS_Models.zip"
TEST_FILES_URL = f"{ASO_CBCT_TEST_FILES}/FullyAuto.zip"

# The steps of the chain, in order: what the progress label says for each.
STEP_TITLES = {
    "PRE_ASO_CBCT": "Centering the scans",
    "ALI_CBCT": "Placing the landmarks",
    "SEMI_ASO_CBCT": "Orienting the scans",
    "AMASSS_CLI": "Segmenting the mandible",
    "TMJ_Crop_CLI": "Cropping the joints",
}


class TMJ_Crop(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "TMJ_Crop"
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jules Grivot Pélisson (UNC)"]
        self.parent.helpText = """
Crops both temporomandibular joints of a CBCT into fixed-size volumes.
See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools#tmj_crop">module documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""


def scan_files(folder):
    """The scans under `folder`, whatever their depth."""
    found = []
    for root, _, names in os.walk(folder):
        for name in sorted(names):
            if name.lower().endswith(SCAN_EXTENSIONS):
                found.append(os.path.join(root, name))
    return found


def scan_stem(path):
    name = os.path.basename(path)
    for extension in sorted(SCAN_EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return name


def check_request(input_path, output_folder, reference, model_ali, model_amasss):
    """What is wrong with the choices made in the panel, as one text, or None.

    Every scan has to carry its own file name: AMASSS_CLI writes every mask into
    one folder under the name of the scan, so two scans of that name in two
    sub-folders would overwrite each other.
    """
    problems = []
    if not input_path or not os.path.exists(input_path):
        problems.append("Please select a scan or a folder of scans")
    else:
        scans = [input_path] if os.path.isfile(input_path) else scan_files(input_path)
        if not scans:
            problems.append("The selected folder must contain scans")
        stems = [scan_stem(path).casefold() for path in scans]
        if len(set(stems)) != len(stems):
            problems.append("Two scans share a file name: give each scan a name of its own")
    if not output_folder:
        problems.append("Please select an output folder")
    if not reference or not os.path.isdir(reference):
        problems.append("Please select the reference folder")
    if not model_ali or not os.path.isdir(model_ali):
        problems.append("Please select the landmark models folder")
    elif not _holds_models(model_ali, catalogs.ORIENTATION_LANDMARKS):
        problems.append("The landmark models folder must hold the models of "
                        + ", ".join(catalogs.ORIENTATION_LANDMARKS))
    if not model_amasss or not os.path.isdir(model_amasss):
        problems.append("Please select the mandible models folder")
    elif not os.path.isdir(os.path.join(model_amasss, "MAND")):
        problems.append("The mandible models folder must hold a MAND folder")
    return "\n".join(problems) if problems else None


def _holds_models(folder, landmarks):
    """Whether a `.pth` sits under `<landmark>/<scale>/` for every landmark."""
    found = set()
    for root, _, names in os.walk(folder):
        if any(".pth" in name for name in names):
            found.add(os.path.basename(os.path.dirname(root)))
    return set(landmarks) <= found


def amasss_models_folder(bundle):
    """The folder of `bundle` that holds the `MAND` models: the bundle itself or a folder below it."""
    for root, dirs, _ in os.walk(bundle):
        if "MAND" in dirs:
            return root
    return bundle


def find_downloads(downloads):
    """The folders a Download button fills, when they are already complete under `downloads`.

    Returns `{"reference", "ali", "amasss"}` with only the ones found: the same
    places and names the three Download buttons write to, so a machine that
    already ran ASO, ALI or AMASSS needs no click.
    """
    found = {}
    reference = os.path.join(downloads, "ASO", "ASO_CBCT", "Reference", REFERENCE_NAME)
    if is_present(reference):
        found["reference"] = reference
    ali = os.path.join(downloads, "ALI")
    if os.path.isdir(ali) and _holds_models(ali, catalogs.ORIENTATION_LANDMARKS):
        found["ali"] = ali
    bundle = os.path.join(downloads, "AMASSS", "Models", "Segmentation")
    if os.path.isdir(bundle):
        folder = amasss_models_folder(bundle)
        if os.path.isdir(os.path.join(folder, "MAND")):
            found["amasss"] = folder
    return found


def build_steps(input_folder, output_folder, reference, model_ali, model_amasss,
                sides, outputs, work, modules):
    """The five steps of the chain, in order, each with the parameters its module takes.

    `modules` maps a module name to the Slicer module (`slicer.modules`), so this
    needs no running application. The first three are ASO's Fully-Automated CBCT
    chain exactly as ASO runs it; the fourth is AMASSS for the mandible alone.
    """
    centred = os.path.join(work, "centred")
    oriented = os.path.join(work, "oriented")
    masks = os.path.join(work, "masks")
    landmarks = " ".join(catalogs.ORIENTATION_LANDMARKS)
    return [
        {
            "Module": "PRE_ASO_CBCT",
            "Process": modules["pre_aso_cbct"],
            "Parameter": {
                "input": input_folder,
                "output_folder": centred,
                # PRE_ASO_CBCT only centres the scan and reads no model: its
                # parameter is kept because the CLI declares it.
                "model_folder": centred,
                "SmallFOV": "false",
                "temp_folder": os.path.join(work, "pre_tmp"),
                "DCMInput": "false",
            },
        },
        {
            "Module": "ALI_CBCT",
            "Process": modules["ali_cbct"],
            "Parameter": {
                "input": centred,
                "dir_models": model_ali,
                "lm_type": ", ".join(f"'{name}'" for name in catalogs.ORIENTATION_LANDMARKS),
                "output_dir": centred,
                "temp_fold": os.path.join(work, "ali_tmp"),
                "DCMInput": "false",
                "spacing": "[1,0.3]",
                "speed_per_scale": "[1,1]",
                "agent_FOV": "[64,64,64]",
                "spawn_radius": "10",
            },
        },
        {
            "Module": "SEMI_ASO_CBCT",
            "Process": modules["semi_aso_cbct"],
            "Parameter": {
                "input": centred,
                "gold_folder": reference,
                "output_folder": oriented,
                "add_inname": catalogs.ORIENTED_SUFFIX,
                "list_landmark": landmarks,
            },
        },
        {
            "Module": "AMASSS_CLI",
            "Process": modules["amasss_cli"],
            "Parameter": {
                "inputVolume": oriented,
                "modelDirectory": model_amasss,
                "skullStructure": "MAND",
                "merge": "SEPARATE",
                "genVtk": False,
                "save_in_folder": False,
                "output_folder": masks,
                "vtk_smooth": 5,
                "prediction_ID": catalogs.PREDICTION_ID,
                "temp_fold": os.path.join(work, "amasss_tmp"),
                "SegmentInput": False,
                "DCMInput": False,
            },
        },
        {
            "Module": "TMJ_Crop_CLI",
            "Process": modules["tmj_crop_cli"],
            "Parameter": {
                "oriented_folder": oriented,
                "landmarks_folder": centred,
                "mask_folder": masks,
                "reference_folder": reference,
                "output_folder": output_folder,
                "input_folder": input_folder,
                "sides": sides,
                "outputs": ",".join(outputs),
                "add_inname": catalogs.ORIENTED_SUFFIX,
                "prediction_id": catalogs.PREDICTION_ID,
            },
        },
    ]


def install_libraries(parent):
    """Check the libraries the chain needs and offer to install what is missing."""
    libs = [("itk", None), ("pytorch_lightning", None), ("dicom2nifti", ">=2.6.2"),
            ("pydicom", "3.0.2"), ("monai", "1.3.2" if sys.version_info >= (3, 10) else "0.7.0"),
            ("einops", None), ("nibabel", None), ("nnunetv2", ">=2.8.0"),
            ("scipy", None), ("matplotlib", None), ("blosc2", None)]
    missing = [(lib, version) for lib, version in libs if not check_lib_installed(lib, version)]
    torch_requirement = torch_install_arguments()
    if not missing and not torch_requirement:
        return True
    message = "The following libraries are not installed or need updating:\n"
    message += "\n".join(requirement(lib, version) for lib, version in missing)
    if torch_requirement:
        message += "\n\nPyTorch build for this GPU:\n" + torch_requirement
    message += "\n\nDo you want to install/update these libraries?\n Doing it could break other modules"
    if not slicer.util.confirmYesNoDisplay(message):
        return False
    # torch first and on its own, as ASO does: the others would pull PyPI's build in.
    if torch_requirement:
        pip_install(torch_requirement)
    for lib, version in missing:
        try:
            pip_install(requirement(lib, version))
        except Exception as error:
            logger.error(f"Error installing {lib}: {error}")
    return True


class TMJ_CropWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = TMJ_CropLogic()
        self.steps = []
        self.process = None
        self.step_index = 0
        self.work = None
        self.start_time = 0

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/TMJ_Crop.ui"))
        self.layout.addWidget(ui_widget)
        self.uiWidget = ui_widget
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        ui_widget.setMRMLScene(slicer.mrmlScene)

        self.ui.label_sides.setText("Joints to crop")

        self.ui.buttonInput.connect("clicked(bool)", self.onBrowseInput)
        self.ui.buttonOutput.connect("clicked(bool)", self.onBrowseOutput)
        self.ui.buttonReference.connect("clicked(bool)", lambda _: self.browseFolder(self.ui.lineEditReference))
        self.ui.buttonModelAli.connect("clicked(bool)", lambda _: self.browseFolder(self.ui.lineEditModelAli))
        self.ui.buttonModelAmasss.connect("clicked(bool)", lambda _: self.browseFolder(self.ui.lineEditModelAmasss))
        self.ui.buttonDownloadReference.connect("clicked(bool)", self.onDownloadReference)
        self.ui.buttonDownloadModelAli.connect("clicked(bool)", self.onDownloadModelAli)
        self.ui.buttonDownloadModelAmasss.connect("clicked(bool)", self.onDownloadModelAmasss)
        self.ui.buttonTestFiles.connect("clicked(bool)", self.onTestFiles)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.cancelButton.connect("clicked(bool)", self.onCancel)

        self.setRunning(False)
        apply_theme(self.uiWidget, self.ui, primary=("applyButton",), danger=("cancelButton",),
                    accents={"violet": ("buttonDownloadReference", "buttonDownloadModelAli",
                                        "buttonDownloadModelAmasss"),
                             "orange": ("buttonTestFiles",)})

    def enter(self):
        """Each time the module is opened: fill the empty model fields from what is already downloaded."""
        found = find_downloads(self.downloadRoot())
        for key, line_edit in (("reference", self.ui.lineEditReference),
                               ("ali", self.ui.lineEditModelAli),
                               ("amasss", self.ui.lineEditModelAmasss)):
            if key in found and not line_edit.text.strip():
                line_edit.setText(found[key])

    def cleanup(self):
        self.removeObservers()

    # ---- where the downloads go ------------------------------------------------

    def downloadRoot(self, *parts):
        documents = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
        return os.path.join(documents, slicer.app.applicationName + "Downloads", *parts)

    # ---- choosing paths --------------------------------------------------------

    def browseFolder(self, line_edit, title="Select a folder"):
        folder = qt.QFileDialog.getExistingDirectory(self.parent, title)
        if folder:
            line_edit.setText(folder)

    def onBrowseInput(self):
        if self.ui.inputType.currentIndex == 0:
            self.browseFolder(self.ui.lineEditInput, "Select a folder of scans")
        else:
            path = qt.QFileDialog.getOpenFileName(self.parent, "Select a scan")
            if path:
                self.ui.lineEditInput.setText(path)

    def onBrowseOutput(self):
        self.browseFolder(self.ui.lineEditOutput, "Select the output folder")

    # ---- downloads -------------------------------------------------------------

    def download(self, url, root, name, title):
        try:
            return ensure_with_progress(url, root, name, parent=self.parent, title=title)
        except (TestDataError, OSError) as error:
            qt.QMessageBox.warning(self.parent, "Warning", f"The download failed:\n{error}")
            return None

    def onDownloadReference(self):
        folder = self.download(REFERENCE_URL, self.downloadRoot("ASO", "ASO_CBCT", "Reference"),
                               REFERENCE_NAME, f"Downloading {REFERENCE_NAME}...")
        if folder:
            self.ui.lineEditReference.setText(folder)

    def onDownloadModelAli(self):
        root = self.downloadRoot("ALI")
        names = catalogs.ORIENTATION_LANDMARKS
        for number, name in enumerate(names, start=1):
            folder = self.download(f"{ALI_MODELS_URL}{name}.zip", root, name,
                                   f"Downloading {name} (File {number}/{len(names)})")
            if folder is None:
                return
        self.ui.lineEditModelAli.setText(root)

    def onDownloadModelAmasss(self):
        bundle = self.download(AMASSS_MODELS_URL, self.downloadRoot("AMASSS", "Models"),
                               "Segmentation", "Downloading the AMASSS models (2 GB)...")
        if bundle:
            bundle = amasss_models_folder(bundle)
            self.ui.lineEditModelAmasss.setText(bundle)

    def onTestFiles(self):
        folder = self.download(TEST_FILES_URL, self.downloadRoot("ASO", "ASO_CBCT", "TestFiles"),
                               "Fully-Automated", "Downloading the test files...")
        if folder:
            self.ui.inputType.setCurrentIndex(0)
            self.ui.lineEditInput.setText(folder)

    # ---- running the chain -----------------------------------------------------

    def selectedOutputs(self):
        """Both extra outputs are always written beside the crops; the panel does not offer them as options."""
        return [catalogs.OUTPUT_ORIENTED_SCAN, catalogs.OUTPUT_MANDIBLE]

    def notify(self, title, text, warning=False):
        box = qt.QMessageBox(self.parent)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(qt.QMessageBox.Warning if warning else qt.QMessageBox.Information)
        box.exec_()

    def onApplyButton(self):
        request = dict(
            input_path=self.ui.lineEditInput.text,
            output_folder=self.ui.lineEditOutput.text,
            reference=self.ui.lineEditReference.text,
            model_ali=self.ui.lineEditModelAli.text,
            model_amasss=self.ui.lineEditModelAmasss.text,
        )
        error = check_request(**request)
        if error:
            self.notify("Warning", error, warning=True)
            return
        if not install_libraries(self.parent):
            self.notify("Warning", "The module will not work properly without the required "
                        "libraries.\nPlease install them and try again.", warning=True)
            return
        self.start(request["input_path"], request["output_folder"], request["reference"],
                   request["model_ali"], request["model_amasss"],
                   self.ui.comboSides.currentText, self.selectedOutputs())

    def start(self, input_path, output_folder, reference, model_ali, model_amasss, sides, outputs):
        """Stage the input, build the five steps and run the first."""
        self.work = slicer.util.tempDirectory()
        self.stageDirectory = os.path.join(self.work, "input")
        input_folder = self.logic.stage_input(input_path, self.stageDirectory)
        self.steps = build_steps(input_folder, output_folder, reference, model_ali,
                                 model_amasss, sides, outputs, self.work, {
                                     "pre_aso_cbct": slicer.modules.pre_aso_cbct,
                                     "ali_cbct": slicer.modules.ali_cbct,
                                     "semi_aso_cbct": slicer.modules.semi_aso_cbct,
                                     "amasss_cli": slicer.modules.amasss_cli,
                                     "tmj_crop_cli": slicer.modules.tmj_crop_cli,
                                 })
        self.output_folder = output_folder
        self.step_index = 0
        self.start_time = time.time()
        self.setRunning(True)
        self.ui.progressBar.setValue(0)
        self.runStep()

    def runStep(self):
        step = self.steps[self.step_index]
        logger.info(f"Step {self.step_index + 1}/{len(self.steps)}: {step['Module']}")
        self.ui.labelStep.setText(
            f"Step {self.step_index + 1} / {len(self.steps)} : {STEP_TITLES[step['Module']]}")
        self.process = slicer.cli.run(step["Process"], None, step["Parameter"])
        self.processObserver = self.process.AddObserver("ModifiedEvent", self.onProcessUpdate)

    def onProcessUpdate(self, caller, event):
        self.ui.labelTimer.setText(format_timer(time.time() - self.start_time))
        progress = caller.GetProgress()
        # A value up to 100 is a fraction of the step; the pulses a CLI sends
        # for "one more scan done" are larger and carry no fraction.
        fraction = progress / 100.0 if 0 < progress <= 100 else 0.0
        self.ui.progressBar.setValue(round((self.step_index + fraction) / len(self.steps) * 100))

        status = caller.GetStatus()
        if not status & caller.Completed:
            return
        caller.RemoveObserver(self.processObserver)
        if status & caller.ErrorsMask:
            logger.error("========= ERROR DETAILS =========")
            error_text = caller.GetErrorText()
            logger.error(f"CLI execution failed: \n{error_text}")
            self.finish()
            self.notify("Error", "There was an error during the process:\n\n" + error_text[-1500:],
                        warning=True)
            return
        logger.info(caller.GetOutputText())
        self.step_index += 1
        if self.step_index < len(self.steps):
            self.runStep()
            return
        self.finish()
        self.notify("End of Process", self.logic.summary(self.output_folder,
                                                         time.time() - self.start_time))

    def finish(self):
        self.setRunning(False)
        if self.work:
            shutil.rmtree(self.work, ignore_errors=True)
            self.work = None

    def onCancel(self):
        if self.process is not None:
            try:
                self.process.Cancel()
            except Exception as error:
                logger.warning(f"Cancellation failed: {error}")
        logger.warning("========= PROCESS CANCELED =========")
        self.finish()

    def setRunning(self, running):
        self.ui.applyButton.setVisible(not running)
        self.ui.cancelButton.setVisible(running)
        self.ui.progressBar.setVisible(running)
        self.ui.labelStep.setVisible(running)
        self.ui.labelTimer.setVisible(running)
        for name in ("CollapsibleButtonInput", "CollapsibleButtonModels", "CollapsibleButtonOptions"):
            getattr(self.ui, name).setEnabled(not running)


class TMJ_CropLogic(ScriptedLoadableModuleLogic):
    """What the panel does that needs no window: staging the input and reading the result."""

    def stage_input(self, input_path, stage):
        """The folder PRE_ASO_CBCT reads: the input folder itself, or a folder holding the one scan."""
        if os.path.isdir(input_path):
            return input_path
        os.makedirs(stage, exist_ok=True)
        target = os.path.join(stage, os.path.basename(input_path))
        try:
            os.symlink(input_path, target)
        except OSError:
            shutil.copyfile(input_path, target)
        return stage

    def summary(self, output_folder, seconds):
        """The message at the end of a run, from the run report the CLI wrote."""
        text = f"Successfully done in {int(seconds // 60)} min and {int(seconds % 60)} sec."
        try:
            with open(os.path.join(output_folder, "run_report.json")) as handle:
                counts = json.load(handle)["summary"]
            text += (f"\n{counts['cropped']} of {counts['cases']} scan(s) cropped"
                     f" ({counts['partial']} partial, {counts['failed']} failed).")
        except (OSError, KeyError, ValueError):
            pass
        return text + f"\nResults are in {output_folder}"


class TMJ_CropTest(ScriptedLoadableModuleTest):
    """Checks the chain the panel builds, without running it."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_the_five_steps_hand_their_folders_on()

    def test_the_five_steps_hand_their_folders_on(self):
        self.delayDisplay("Starting the test")
        names = ("pre_aso_cbct", "ali_cbct", "semi_aso_cbct", "amasss_cli", "tmj_crop_cli")
        steps = build_steps("in", "out", "ref", "ali", "amasss", "Left and Right",
                            [catalogs.OUTPUT_MANDIBLE], "work", {name: name for name in names})
        self.assertEqual([step["Process"] for step in steps], list(names))
        self.assertEqual(steps[0]["Parameter"]["output_folder"], steps[1]["Parameter"]["input"])
        self.assertEqual(steps[2]["Parameter"]["output_folder"], steps[3]["Parameter"]["inputVolume"])
        self.assertEqual(steps[3]["Parameter"]["output_folder"], steps[4]["Parameter"]["mask_folder"])
        self.delayDisplay("Test passed")
