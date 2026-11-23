import slicer
from slicer.ScriptedLoadableModule import *

import os
import sys
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

from BATCHDENTALSEGLib import SegmentationWidget
from BATCHDENTALSEGLib.IconPath import icon

import sys

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("BatchDentalSeg")

class BATCHDENTALSEG(ScriptedLoadableModule):
    def __init__(self, parent):
        from slicer.i18n import tr, translate
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = tr("BatchDentalSegmentator")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Automated Dental Tools")]
        self.parent.dependencies = []
        self.parent.icon = icon("DentalSegmentator_full_icon_2.png")
        self.parent.contributors = [
            "Enzo Tulissi (UoM)",
            "Gauthier DOT (AP-HP)",
            "Laurent GAJNY (ENSAM)",
            "Roman FENIOUX (KITWARE SAS)",
            "Thibault PELLETIER (KITWARE SAS)"
        ]

        self.parent.helpText = tr(
            "Fully automatic AI segmentation tool for Dental CT and CBCT scans based on DentalSegmentator nnU-Net or PediatricDentalSegmentator nnU-Net "
            "model."
        )
        self.parent.acknowledgementText = tr(
            "This module was originally developed for the "
            '<a href="https://orthodontie-ffo.org/">Fédération Française d\'Orthodontie</a> '
            "(FFO) for the analysis of dento-maxillo-facial data."
        )


class BATCHDENTALSEGWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)
        widget = SegmentationWidget()
        self.logic = widget.logic
        self.layout.addWidget(widget)
        self.layout.addStretch()


class DentalSegmentatorTest(ScriptedLoadableModuleTest):
    def runTest(self):
        try:
            from SlicerPythonTestRunnerLib import RunnerLogic, RunnerWidget, RunSettings, isRunningInTestMode  # noqa: F401  (sonde de disponibilite)
            from pathlib import Path
        except ImportError:
            slicer.util.warningDisplay("Please install SlicerPythonTestRunner extension to run the self tests.")
            return

        current_dir_test = Path(__file__).parent.joinpath("Testing")
        results = RunnerLogic().runAndWaitFinished(
            current_dir_test,
            RunSettings(extraPytestArgs=RunSettings.pytestFileFilterArgs("*TestCase.py") + ["-m not slow"]),
            doRunInSubProcess=not isRunningInTestMode()
        )

        if results.failuresNumber:
            raise AssertionError(f"Test failed: \n{results.getFailingCasesString()}")

        slicer.util.delayDisplay(f"Tests OK. {results.getSummaryString()}")

