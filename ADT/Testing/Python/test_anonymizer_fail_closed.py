# The anonymizer writes nothing it could not anonymize.
#
# When Presidio raised, anonymize_text_presidio returned the text it was given.
# It was saved as `<name>_anonymized.txt` and counted as a success: any text
# over spaCy's 1,000,000 characters came out with every name still in it.
#
# Slicer, Qt and Presidio are stood in for: what is pinned is what reaches the
# output folder, and what the user is told.
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", ".."))
if os.path.join(_ROOT, "ADT") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "ADT"))


@contextlib.contextmanager
def _stand_ins(modules):
    """Put `modules` in sys.modules, then put back those names and no other."""
    absent = object()
    before = {name: sys.modules.get(name, absent) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in before.items():
            if module is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_module():
    scripted = types.ModuleType("slicer.ScriptedLoadableModule")
    for suffix in ("", "Widget", "Logic", "Test"):
        name = "ScriptedLoadableModule" + suffix
        setattr(scripted, name, type(name, (), {}))
    slicer = mock.MagicMock()
    with _stand_ins({"slicer": slicer, "slicer.ScriptedLoadableModule": scripted,
                     "qt": mock.MagicMock(), "ctk": mock.MagicMock()}):
        spec = importlib.util.spec_from_file_location(
            "anonymizer_fail_closed",
            os.path.join(_ROOT, "Medical_Data_Anonymizer_Module",
                         "Medical_Data_Anonymizer_Module.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class Analyzer:
    """Refuses a text holding UNREADABLE, as spaCy refuses one that is too long."""

    def analyze(self, text, **_):
        if "UNREADABLE" in text:
            raise ValueError("[E088] Text exceeds maximum")
        return []


class Anonymizer:
    def anonymize(self, text, **_):
        return types.SimpleNamespace(text=text.replace("John Smith", "<PERSON>"))


class FailClosedTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def setUp(self):
        self.input = tempfile.mkdtemp(prefix="anonymizer_in_")
        self.addCleanup(shutil.rmtree, self.input, True)
        self.output = tempfile.mkdtemp(prefix="anonymizer_out_")
        self.addCleanup(shutil.rmtree, self.output, True)
        self.module.qt.reset_mock()

    def write(self, name, text):
        with open(os.path.join(self.input, name), "w", encoding="utf-8") as handle:
            handle.write(text)

    def run_module(self):
        widget = mock.MagicMock()
        widget.logic = self.module.Medical_Data_Anonymizer_ModuleLogic()
        widget.inputDirectoryButton.currentPath = self.input
        widget.outputDirectoryButton.currentPath = self.output
        widget.entityCheckboxes = {"PERSON": mock.MagicMock()}
        widget.anonymizationMethodCombo.currentData = "replace"
        widget.scoreThresholdSlider.value = 0.5

        self.mappings = []
        pandas = mock.MagicMock()

        def data_frame(rows):
            self.mappings = rows
            return mock.MagicMock(empty=not rows)

        pandas.DataFrame = data_frame
        with _stand_ins({"presidio_analyzer": mock.MagicMock(AnalyzerEngine=Analyzer),
                         "presidio_anonymizer": mock.MagicMock(AnonymizerEngine=Anonymizer),
                         "docx": mock.MagicMock(), "pandas": pandas}):
            self.module.Medical_Data_Anonymizer_ModuleWidget.onAnonymizeButton(widget)
        return widget

    def test_the_logic_raises_rather_than_return_the_text_it_was_given(self):
        logic = self.module.Medical_Data_Anonymizer_ModuleLogic()
        with self.assertRaises(ValueError):
            logic.anonymize_text_presidio(
                "John Smith UNREADABLE", Analyzer(), Anonymizer(), ["PERSON"], "replace", 0.5)

    def test_a_file_that_failed_leaves_no_output(self):
        self.write("report.txt", "John Smith UNREADABLE")
        self.run_module()
        self.assertEqual(os.listdir(self.output), [])
        self.assertEqual([row["Anonymized File Name"] for row in self.mappings], ["ERROR"])

    def test_a_failure_is_not_announced_as_a_success(self):
        self.write("good.txt", "John Smith")
        self.write("bad.txt", "John Smith UNREADABLE")
        widget = self.run_module()

        self.module.qt.QMessageBox.information.assert_not_called()
        self.module.qt.QMessageBox.warning.assert_called_once()
        message = self.module.qt.QMessageBox.warning.call_args[0][2]
        self.assertIn("1 of 2", message)
        self.assertIn("bad.txt", message)
        self.assertIn("1 of 2", widget.statusLabel.setText.call_args[0][0])

    def test_a_long_list_of_failures_still_fits_in_the_dialog(self):
        for index in range(25):
            self.write("bad%02d.txt" % index, "UNREADABLE")
        self.run_module()

        self.module.qt.QMessageBox.warning.assert_called_once()
        message = self.module.qt.QMessageBox.warning.call_args[0][2]
        self.assertIn("25 of 25", message)
        self.assertIn("and 15 more", message)

    def test_a_run_without_failure_is_announced_as_one(self):
        self.write("good.txt", "John Smith")
        self.run_module()

        self.module.qt.QMessageBox.warning.assert_not_called()
        self.module.qt.QMessageBox.information.assert_called_once()
        with open(os.path.join(self.output, "good_anonymized.txt"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "<PERSON>")


if __name__ == "__main__":
    unittest.main()
