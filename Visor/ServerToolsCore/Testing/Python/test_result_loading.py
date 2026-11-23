"""What gets opened in the scene when a run finishes, and what does not.

Two defects live here, both of which reached a clinician:

* ASO has declared `("*.mrk.json", "markups")` since it was converted, and no
  loader was ever registered for that kind. Every ASO run with the box ticked
  ended on a red "Some results could not be loaded" -- the landmarks were on
  disk and correct, and the panel said the run had failed to deliver them.
* A chain's own results, returned under `intermediate/`, were poured into the
  scene beside the run's, because the pattern was matched against the file's
  NAME and a name says nothing about which directory it came from.
"""

import ast
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO = os.path.abspath(os.path.join(_CORE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: F401,E402 - installs the stubs
from test_hosted_test_files import _util  # noqa: E402

from ServerToolsCoreLib import slicer_io  # noqa: E402
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase  # noqa: E402


def _declared_pairs():
    """Every `(pattern, kind)` any module names in its `_LOADABLE`, by module.

    Read out of the source rather than imported: a module needs Slicer to
    import, and what is being checked is a DECLARATION, which is right there in
    the text.
    """
    found = {}
    # `<Module>/<Module>.py`, or one level down where a module keeps its panel
    # beside its CLI (`TMJ_Crop/TMJ_Crop_UI/TMJ_Crop.py`).
    paths = []
    for entry in sorted(os.listdir(_REPO)):
        direct = os.path.join(_REPO, entry, entry + ".py")
        if os.path.isfile(direct):
            paths.append(direct)
        elif os.path.isdir(os.path.join(_REPO, entry)):
            for sub in sorted(os.listdir(os.path.join(_REPO, entry))):
                nested = os.path.join(_REPO, entry, sub, entry + ".py")
                if os.path.isfile(nested):
                    paths.append(nested)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            try:
                tree = ast.parse(handle.read())
            except SyntaxError:
                continue  # a module pinned to a newer Python; not ours to parse
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_LOADABLE" not in names or not isinstance(node.value, (ast.Tuple, ast.List)):
                continue
            for pair in node.value.elts:
                if not isinstance(pair, (ast.Tuple, ast.List)) or len(pair.elts) != 2:
                    continue
                pattern, kind = pair.elts
                if all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                       for e in (pattern, kind)):
                    module = os.path.splitext(os.path.basename(path))[0]
                    found.setdefault(module, []).append((pattern.value, kind.value))
    return found


def _declared_kinds():
    """The same, reduced to the set of kinds each module names."""
    return {module: {kind for _pattern, kind in pairs}
            for module, pairs in _declared_pairs().items()}


class LoaderCoverageTest(unittest.TestCase):
    def test_every_kind_a_module_declares_has_a_loader(self):
        """The general form of the markups bug: a module may only name a kind
        this library can actually open. Declaring one nobody registered fails
        at the END of a run, after the GPU time is spent."""
        declared = _declared_kinds()
        self.assertTrue(declared, "no module declared _LOADABLE; the scan is broken")

        missing = {
            module: sorted(kinds - set(slicer_io._LOADERS))
            for module, kinds in declared.items()
            if kinds - set(slicer_io._LOADERS)
        }
        self.assertEqual(missing, {}, "no MRML loader for these declared kinds")

    def test_no_module_declares_two_patterns_that_can_match_one_file(self):
        """A file matching two patterns is opened TWICE -- the same scan in the
        scene as a volume and again as a labelmap. None does today; this is
        what keeps it that way, because the duplicate is only visible in the
        Data module and reads as a tool that ran twice."""
        import fnmatch

        overlapping = {}
        for module, pairs in _declared_pairs().items():
            for index, (pattern, _kind) in enumerate(pairs):
                for other, _k in pairs[index + 1:]:
                    if fnmatch.fnmatch(pattern.replace("*", "x"), other) or \
                            fnmatch.fnmatch(other.replace("*", "x"), pattern):
                        overlapping.setdefault(module, []).append((pattern, other))
        self.assertEqual(overlapping, {})

    def test_markups_are_loadable(self):
        """Named on its own because it is the one that shipped broken."""
        self.assertIn("markups", slicer_io._LOADERS)


class IntermediateResultsTest(unittest.TestCase):
    """A chain's results are delivered, not opened."""

    class _Panel(ServerToolWidgetBase):
        TOOL_NAME = "ASO"
        _LOADABLE = (("*.mrk.json", "markups"), ("*.nii.gz", "volume"))

    def setUp(self):
        _util.loaded = []
        self.panel = self._Panel.__new__(self._Panel)
        self.panel._producedRoot = "/out"
        self.panel._loadResultsCheckBox = None

    def test_the_runs_own_results_are_opened(self):
        self.panel._producedFiles = ["/out/Pat_0002_lm_Or.mrk.json"]
        self.panel._loadResults()

        self.assertEqual(_util.loaded, [("markups", "/out/Pat_0002_lm_Or.mrk.json")])

    def test_what_the_chain_produced_is_left_on_disk(self):
        """The exact file from the report: ASO returned its own oriented
        landmarks AND ALI's raw prediction, and both went into the scene."""
        self.panel._producedFiles = [
            "/out/Pat_0002_lm_Or.mrk.json",
            "/out/intermediate/01_ALI_CBCT/Pat_0002_lm_Pred.mrk.json",
        ]
        self.panel._loadResults()

        self.assertEqual(_util.loaded, [("markups", "/out/Pat_0002_lm_Or.mrk.json")])

    def test_what_the_chain_produced_for_one_patient_is_left_on_disk(self):
        """A tool that sorts its chain's outputs by patient files them one
        level down, under the patient's own folder."""
        self.panel._producedFiles = [
            "/out/Pat_0002/Pat_0002_lm_Or.mrk.json",
            "/out/Pat_0002/intermediate/01_ALI_CBCT/Pat_0002_lm_Pred.mrk.json",
        ]
        self.panel._loadResults()

        self.assertEqual(_util.loaded, [("markups", "/out/Pat_0002/Pat_0002_lm_Or.mrk.json")])

    def test_a_file_merely_named_like_the_folder_is_not_confused_for_one(self):
        """Only folders count, so a result whose own name starts with the
        word is still loaded."""
        self.panel._producedFiles = ["/out/patient/intermediate_scan.mrk.json"]
        self.panel._loadResults()

        self.assertEqual(len(_util.loaded), 1)

    def test_an_output_folder_reached_through_a_symlink_still_filters(self):
        """The archive's members come back resolved; the folder the reader
        picked did not, and every file then looked outside it."""
        real = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, real, True)
        link = os.path.join(tempfile.mkdtemp(), "Slicer Output")
        self.addCleanup(shutil.rmtree, os.path.dirname(link), True)
        try:
            os.symlink(real, link)
        except (OSError, NotImplementedError) as exc:  # Windows without the privilege
            self.skipTest("cannot create a symlink here: %s" % exc)
        self.panel._producedRoot = link
        self.panel._producedFiles = [
            os.path.join(real, "Pat_0002_lm_Or.mrk.json"),
            os.path.join(real, "T1_Oriented", "intermediate", "01_ALI_CBCT", "Pat_0002_lm_Pred.mrk.json"),
        ]
        self.panel._loadResults()

        self.assertEqual(_util.loaded, [("markups", os.path.join(real, "Pat_0002_lm_Or.mrk.json"))])

    def test_with_no_root_recorded_nothing_is_skipped(self):
        """A single-file result records its own folder; an older panel state
        records nothing. Skipping on no information would lose results."""
        self.panel._producedRoot = ""
        self.panel._producedFiles = ["/elsewhere/intermediate/x.mrk.json"]
        self.panel._loadResults()

        self.assertEqual(len(_util.loaded), 1)


class ShowLoadedResultsTest(unittest.TestCase):
    """Loading puts a node in the scene; it decides nothing about what is shown.

    ASO's oriented CBCT arrived correctly and the slice views kept whatever was
    already there, so a 102 MB volume sat in the Data module while the panel
    looked as though only the landmarks had come back.
    """

    class _Panel(ServerToolWidgetBase):
        TOOL_NAME = "ASO"
        _LOADABLE = (("*.nii.gz", "volume"), ("*.mrk.json", "markups"))

    def setUp(self):
        _util.loaded = []
        _util.shown = None
        self.panel = self._Panel.__new__(self._Panel)
        self.panel._producedRoot = "/out"
        self.panel._loadResultsCheckBox = None
        self._realSetLayers = _util.setSliceViewerLayers

    def tearDown(self):
        # One test replaces it to model a view that refuses; put the RECORDING
        # stub back, or every test after it silently observes nothing.
        _util.setSliceViewerLayers = self._realSetLayers

    def test_the_scan_is_put_in_front_of_the_user(self):
        self.panel._producedFiles = ["/out/Pat_0002_Or.nii.gz"]
        self.panel._loadResults()

        background, label, fit = _util.shown
        self.assertEqual(background.path, "/out/Pat_0002_Or.nii.gz")
        self.assertIsNone(label)
        self.assertTrue(fit, "an oriented scan is not where the old view was")

    def test_labelled_voxels_go_in_the_label_layer(self):
        """Which layer a result belongs in is what its KIND already says.

        AMASSS's shape: one pattern, declared as a labelmap because that is
        what the tool writes -- the same extension ASO uses for a grey scan."""
        class _Labels(ServerToolWidgetBase):
            TOOL_NAME = "AMASSS"
            _LOADABLE = (("*.nii.gz", "labelmap"),)

        self.panel = _Labels.__new__(_Labels)
        self.panel._producedRoot = "/out"
        self.panel._loadResultsCheckBox = None
        self.panel._producedFiles = ["/out/scan_Seg.nii.gz"]
        self.panel._loadResults()

        background, label, _fit = _util.shown
        self.assertIsNone(background)
        self.assertEqual(label.kind, "labelmap")

    def test_markups_alone_ask_for_no_slice_layer(self):
        """They carry their own display node and show themselves; there is no
        layer to put them in. The views are left as `_hideScene` emptied them,
        not handed an earlier scan back."""
        self.panel._producedFiles = ["/out/Pat_0002_lm_Or.mrk.json"]
        self.panel._loadResults()

        self.assertEqual(_util.loaded, [("markups", "/out/Pat_0002_lm_Or.mrk.json")])
        self.assertEqual(_util.shown, (None, None, False))

    def test_a_cohort_shows_the_first_scan_and_always_the_same_one(self):
        """Forty patients offer no better answer than a deterministic one."""
        self.panel._producedFiles = [
            "/out/Pat_0001_Or.nii.gz", "/out/Pat_0002_Or.nii.gz"]
        self.panel._loadResults()
        first = _util.shown[0].path

        _util.shown = None
        self.panel._loadResults()
        self.assertEqual(_util.shown[0].path, first)
        self.assertEqual(first, "/out/Pat_0001_Or.nii.gz")

    def test_a_view_that_refuses_does_not_fail_the_run(self):
        """The results are on disk and in the scene either way."""
        def _refuse(**kwargs):
            raise RuntimeError("no slice view in this layout")

        _util.setSliceViewerLayers = _refuse
        self.panel._producedFiles = ["/out/Pat_0002_Or.nii.gz"]
        self.panel._loadResults()  # must not raise

        self.assertEqual(len(_util.loaded), 1)


class OneCheckBoxTest(unittest.TestCase):
    def test_a_module_that_builds_its_own_box_does_not_get_a_second(self):
        """AREG still builds one in `addExtraWidgets`, under the same attribute
        name the base uses. Two boxes would appear, the module's wired to its
        own handler and the base's silently winning the attribute."""
        import qt

        panel = self._panel()
        panel._loadResultsCheckBox = qt.QCheckBox("mine")
        mine = panel._loadResultsCheckBox

        layout = qt.QVBoxLayout()
        panel._addLoadResultsCheckBox(layout)

        self.assertIs(panel._loadResultsCheckBox, mine)
        self.assertEqual(layout.widgets, [])

    @staticmethod
    def _panel():
        class _P(ServerToolWidgetBase):
            TOOL_NAME = "AREG"
            _LOADABLE = (("*.nii.gz", "volume"),)
            # The real panel reads `_schema` to know which subfolder an
            # argument's hosted files come from; a double without one is a
            # double of an older panel.
            _schema = {"arguments": {}}

        panel = _P.__new__(_P)
        panel._producedFiles = []
        panel._producedRoot = ""
        return panel


if __name__ == "__main__":
    unittest.main()


class VolumeRenderingTest(unittest.TestCase):
    """A scan result shown in 3D, with the preset its module named.

    Driven through the Volume Rendering module's own controls rather than by
    writing to the nodes. Writing the curve directly produces the right image
    -- that is what this did first -- but the preset list and the Shift slider
    are the module's own state with no counterpart in MRML, so the picture was
    right while the panel read "no preset, shift 0", and anyone touching the
    slider afterwards started from a curve already moved by an unstated amount.
    """

    def _panel(self, preset="CT-AAA", loadable=(("*.nii.gz", "volume"),)):
        class _P(ServerToolWidgetBase):
            TOOL_NAME = "ASO"
            _LOADABLE = loadable
            VOLUME_RENDERING = preset

        panel = _P.__new__(_P)
        panel._producedRoot = "/out"
        panel._producedFiles = ["/out/Pat_0002_Or.nii.gz"]
        panel._loadResultsCheckBox = None
        return panel

    def setUp(self):
        _util.loaded = []
        _util.shown = None
        module = sys.modules["slicer"].modules.volumerendering
        self._logic = module.logic()
        self._widget = module.widgetRepresentation()
        self._logic.created = []
        self._widget.applied = []
        self._widget.children["PresetComboBox"].panel = self._widget
        self._widget.children["PresetOffsetSlider"].panel = self._widget

    def test_the_rendering_is_created_and_switched_on(self):
        self._panel()._loadResults()

        self.assertEqual(len(self._logic.created), 1)
        node, display = self._logic.created[0]
        self.assertEqual(node.path, "/out/Pat_0002_Or.nii.gz")
        self.assertTrue(display.visible, "a rendering nobody switched on shows nothing")

    def test_the_module_shows_the_preset_and_the_shift(self):
        """What writing to the nodes could never do: the panel reads CT-AAA and
        580 instead of "no preset, shift 0" over a curve already moved."""
        from ServerToolsCoreLib import slicer_io as io

        self._panel()._loadResults()

        self.assertEqual(self._widget.children["PresetComboBox"].current, "CT-AAA")
        self.assertEqual(self._widget.children["PresetOffsetSlider"].value,
                         io.VOLUME_RENDERING_SHIFT)

    def test_the_volume_is_selected_before_anything_is_applied(self):
        """The module applies a preset to whatever IS selected in it, and the
        Shift slider's range is computed from that volume. Setting either one
        first would work on the wrong scan."""
        self._panel()._loadResults()

        self.assertEqual([what for what, _value in self._widget.applied],
                         ["volume", "preset", "shift", "visible"])

    def test_the_shift_comes_after_the_preset(self):
        """Applying a preset RESETS the offset, so a shift set first is thrown
        away without a word -- and the picture would be subtly wrong with the
        panel claiming otherwise."""
        self._panel()._loadResults()

        order = [what for what, _value in self._widget.applied]
        self.assertLess(order.index("preset"), order.index("shift"))

    def test_the_rendering_is_switched_on_after_the_module_has_settled(self):
        """The one that shipped broken: reaching the module instantiates its
        widget, which reacts to the volume being selected and settles the
        rendering's state -- so a visibility set BEFORE that was simply undone.

        What a clinician saw was a 3D view that stayed empty while the module
        showed a rendering loaded, correct, and waiting for someone to find the
        Visibility check box.
        """
        self._panel()._loadResults()

        _node, display = self._logic.created[0]
        self.assertTrue(display.visible, "switched off again by the module")
        self.assertTrue(self._widget.children["VisibilityCheckBox"].checked,
                        "and the module's own box has to read as ticked")

    def test_visibility_comes_last_of_all(self):
        order = [what for what, _value in self._widget.applied]
        self._panel()._loadResults()
        order = [what for what, _value in self._widget.applied]

        self.assertEqual(order[-1], "visible")

    def test_a_module_naming_no_preset_renders_nothing(self):
        """The default. A tool returning an MRI or a mesh must not have a CT
        curve applied to it, and most modules return no grey volume at all."""
        self._panel(preset="")._loadResults()

        self.assertEqual(self._logic.created, [])
        self.assertEqual(self._widget.applied, [])

    def test_a_labelmap_result_is_never_rendered(self):
        """AMASSS returns labelled voxels. They belong in the label layer, and
        a CT preset over them is meaningless."""
        self._panel(loadable=(("*.nii.gz", "labelmap"),))._loadResults()

        self.assertEqual(self._logic.created, [])

    def test_the_presets_scene_is_loaded_before_the_preset_is_asked_for(self):
        """The trap this fixes: presets live in a scene of their own, loaded on
        demand, and `GetPresetByName` looks in it WITHOUT loading it. A module
        nobody had opened yet answered nothing, the default curve stayed, and
        the rendering appeared -- just not the one that was asked for."""
        self._logic.presetsLoaded = False
        self._panel()._loadResults()

        self.assertTrue(self._logic.presetsLoaded)
        self.assertEqual(self._widget.children["PresetComboBox"].current, "CT-AAA")

    def test_a_preset_this_build_does_not_have_leaves_the_rendering_alone(self):
        """Named by a module, installed by a build. A missing one still renders
        -- with the default curve -- rather than not at all, and touches none
        of the module's controls."""
        self._panel(preset="CT-Nonexistent")._loadResults()

        _node, display = self._logic.created[0]
        self.assertTrue(display.visible)
        self.assertEqual(self._widget.applied, [])

    def test_a_renamed_control_does_not_fail_the_run(self):
        """These are Qt object names from Slicer's own .ui -- the one part of
        this a release could rename under us. The results are on disk and the
        rendering is on screen either way."""
        combo = self._widget.children.pop("PresetComboBox")
        try:
            self._panel()._loadResults()  # must not raise
        finally:
            self._widget.children["PresetComboBox"] = combo

        _node, display = self._logic.created[0]
        self.assertTrue(display.visible)



class OneScanOnlyTest(unittest.TestCase):
    """A 3D rendering shows ONE patient, so it is only offered for one.

    A cohort of forty would have one of them rendered and the other
    thirty-nine not, with nothing on screen saying which -- a 3D view showing a
    patient the clinician did not choose is worse than one showing nothing.
    They are all in the slice views and on disk either way.
    """

    def _panel(self, files):
        class _P(ServerToolWidgetBase):
            TOOL_NAME = "ASO"
            _LOADABLE = (("*.nii.gz", "volume"), ("*.mrk.json", "markups"))
            VOLUME_RENDERING = "CT-AAA"

        panel = _P.__new__(_P)
        panel._producedRoot = "/out"
        panel._producedFiles = list(files)
        panel._loadResultsCheckBox = None
        return panel

    def setUp(self):
        _util.loaded = []
        _util.shown = None
        module = sys.modules["slicer"].modules.volumerendering
        self._logic = module.logic()
        self._widget = module.widgetRepresentation()
        self._logic.created = []
        self._widget.applied = []

    def test_one_patient_is_rendered(self):
        self._panel(["/out/Pat_0002_Or.nii.gz"])._loadResults()

        self.assertEqual(len(self._logic.created), 1)

    def test_several_patients_are_loaded_but_none_is_rendered(self):
        files = ["/out/Pat_000%d_Or.nii.gz" % n for n in (1, 2, 3)]
        self._panel(files)._loadResults()

        self.assertEqual(len(_util.loaded), 3, "every scan still reaches the scene")
        self.assertEqual(self._logic.created, [],
                         "no patient may be singled out in the 3D view")

    def test_several_patients_still_get_a_slice_background(self):
        """The slice views show one volume whatever happens -- Slicer's own
        behaviour. What is refused is the 3D rendering, which reads as a
        finding about a patient rather than as a layer."""
        files = ["/out/Pat_000%d_Or.nii.gz" % n for n in (1, 2)]
        self._panel(files)._loadResults()

        background, _label, _fit = _util.shown
        self.assertEqual(background.path, "/out/Pat_0001_Or.nii.gz")

    def test_landmarks_beside_one_scan_do_not_count_as_a_second(self):
        """Only VOLUMES are counted: a run producing one scan and its markups
        is one patient, and must still be rendered."""
        self._panel(["/out/Pat_0002_Or.nii.gz",
                     "/out/Pat_0002_lm_Or.mrk.json"])._loadResults()

        self.assertEqual(len(self._logic.created), 1)


class ImportPreviewTest(unittest.TestCase):
    """A panel reduced to the one thing these tests exercise: a row with a path
    in it, and what reaches the scene because of it. No test cases of its own."""

    class _Picker:
        def __init__(self, path):
            self.currentPath = path

    def _panel(self, preset="CT-AAA", tool="AMASSS"):
        class _P(ServerToolWidgetBase):
            TOOL_NAME = tool
            VOLUME_RENDERING = preset

        panel = _P.__new__(_P)
        panel._scenePreviews = {}
        panel._progressLabel = None
        panel._inputWidgets = {}
        return panel

    def setUp(self):
        _util.loaded = []
        _util.nodes = []
        module = sys.modules["slicer"].modules.volumerendering
        self._logic = module.logic()
        self._logic.created = []

    def _pick(self, path, panel=None):
        panel = panel or self._panel()
        panel._inputWidgets["scans"] = self._Picker(path)
        panel._previewPickedFile("scans")
        return panel


class RenderOnImportTest(ImportPreviewTest):
    """The scan a clinician just picked, shown in 3D before anything is run.

    Same question as a result: one scan going into an empty 3D view, and the
    rule that keeps it to one is `slicer_io.sole_scan_in` -- a cohort never
    previews and cannot flood anything.
    """

    def test_a_picked_scan_is_rendered(self):
        self._pick("/data/Pat_0002.nii.gz")

        self.assertEqual(len(_util.loaded), 1)
        self.assertEqual(len(self._logic.created), 1)

    def test_a_picked_mesh_is_not(self):
        """It loads as a model, which carries its own display node -- there is
        no volume for a CT curve to apply to."""
        self._pick("/data/arch.vtk")

        self.assertEqual(len(_util.loaded), 1)
        self.assertEqual(self._logic.created, [])

    def test_an_archive_or_a_table_is_never_previewed_at_all(self):
        """Nothing a scene can hold, so nothing is loaded and nothing
        rendered -- the rule this feature needed, and it predates it."""
        for path in ("/data/cohort.zip", "/data/notes.csv"):
            self._pick(path)

        self.assertEqual(_util.loaded, [])
        self.assertEqual(self._logic.created, [])

    def test_a_module_that_does_not_work_on_scans_renders_nothing(self):
        self._pick("/data/Pat_0002.nii.gz", self._panel(preset="", tool="Crown_Seg"))

        self.assertEqual(len(_util.loaded), 1, "it is still put in the scene")
        self.assertEqual(self._logic.created, [], "but not rendered in 3D")

    def test_picking_the_same_file_twice_renders_once(self):
        """Re-picking would otherwise stack copies in the scene -- and a second
        rendering over the first."""
        panel = self._pick("/data/Pat_0002.nii.gz")
        self._pick("/data/Pat_0002.nii.gz", panel)

        self.assertEqual(len(self._logic.created), 1)

    def test_a_hosted_test_file_is_rendered_too(self):
        """The SECOND path a scan reaches the scene by, and the one that was
        missed.

        A hand-picked file goes through `_previewPickedFile`; a test file the
        panel downloaded goes through `_useTestFile`, which loads it itself.
        Wiring only the first left a downloaded scan in the scene, correct and
        flat -- and "the test files" is the case this feature was asked for.
        """
        panel = self._panel()
        panel._testFileCache = {}
        # The cache is keyed by (scope, name): the panel reads the scope off
        # its schema, two arguments being able to offer the same entry name.
        panel._schema = {"arguments": {}}
        panel._checkCanApply = lambda *a: None
        panel._lastTestFileTimings = []
        panel._hideProgress = lambda *a: None

        panel._useTestFile("scans", "sample_scan.nii.gz", "/tmp/sample_scan.nii.gz")

        self.assertEqual(_util.loaded,
                         [("volume", "/tmp/sample_scan.nii.gz")])
        self.assertEqual(len(self._logic.created), 1,
                         "a downloaded scan is a scan like any other")

    def test_a_hosted_test_folder_is_neither_loaded_nor_rendered(self):
        """A cohort arrives as a folder and never reaches the scene."""
        panel = self._panel()
        panel._testFileCache = {}
        # The cache is keyed by (scope, name): the panel reads the scope off
        # its schema, two arguments being able to offer the same entry name.
        panel._schema = {"arguments": {}}
        panel._checkCanApply = lambda *a: None
        panel._lastTestFileTimings = []
        panel._hideProgress = lambda *a: None

        folder = os.path.dirname(os.path.abspath(__file__))
        panel._useTestFile("scans", "cohort", folder)

        self.assertEqual(_util.loaded, [])
        self.assertEqual(self._logic.created, [])

    def test_a_file_or_folder_picker_previews_like_any_other(self):
        """ALI is the only module whose input is a `file_or_folder` field, and
        that field is a composite of this repo's own -- not a ctkPathLineEdit.

        Exercised with the REAL widget rather than a stand-in, because what
        could break here is precisely the seam: `_previewPickedFile` reads
        `currentPath` off whatever it finds in `_inputWidgets`, and this one
        supplies it as a plain Python property with no Qt widget behind it at
        all.
        """
        from ServerToolsCoreLib import formgen

        panel = self._panel(tool="ALI")
        widget = formgen.FileOrFolderInput()
        widget.setCurrentPath("/data/Pat_0002.nii.gz")
        panel._inputWidgets["input"] = widget

        panel._previewPickedFile("input")

        self.assertEqual(_util.loaded, [("volume", "/data/Pat_0002.nii.gz")])
        self.assertEqual(len(self._logic.created), 1)

    def test_a_folder_in_that_same_picker_previews_nothing(self):
        """The field takes either, and only the file half may reach the scene."""
        from ServerToolsCoreLib import formgen

        panel = self._panel(tool="ALI")
        widget = formgen.FileOrFolderInput()
        widget.setCurrentPath("/data/cohort.zip")
        panel._inputWidgets["input"] = widget

        panel._previewPickedFile("input")

        self.assertEqual(_util.loaded, [])
        self.assertEqual(self._logic.created, [])


class SoleScanInAFolderTest(ImportPreviewTest):
    """A folder holding ONE patient is one scan, and is shown like one.

    "A folder is never loaded" was the first version of the cohort rule, and it
    was too blunt by half. Of the CBCT test files this extension offers, only
    ALI's is a bare `.nii.gz`: ASO's four are folders, AutoMatrix's are folders,
    two of ASO's are DICOM series. Every one of them holds a single patient, and
    every one of them showed nothing -- so the feature looked absent on every
    module but the one it was built on, which is exactly what was reported.

    What must never happen is forty patients landing in the scene at once. That
    is a count, not a file-or-folder question, and `slicer_io.sole_scan_in` is
    where it is now asked.
    """

    def _folder(self, *names):
        """A directory holding these (possibly nested) files, emptied after."""
        root = tempfile.mkdtemp(prefix="sole_scan_")
        self.addCleanup(shutil.rmtree, root, True)
        for name in names:
            path = os.path.join(root, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(b"x")
        return root

    # -- one patient ----------------------------------------------------

    def test_a_folder_holding_one_scan_is_that_scan(self):
        """ASO's CBCT_FullyAuto, which is one `Pat_0002.nii.gz` in a folder."""
        folder = self._folder("Pat_0002.nii.gz")

        self._pick(folder)

        self.assertEqual(_util.loaded,
                         [("volume", os.path.join(folder, "Pat_0002.nii.gz"))])
        self.assertEqual(len(self._logic.created), 1, "and rendered in 3D")

    def test_landmarks_beside_the_scan_do_not_make_it_a_cohort(self):
        """ASO's CBCT_SemiAuto: a scan and the landmarks to orient it by. Only
        VOLUMES are counted -- the same rule the result side already applies."""
        folder = self._folder("IC_0005.nii.gz", "IC_0005_lm_Pred.mrk.json")

        self._pick(folder)

        self.assertEqual(len(_util.loaded), 1)
        self.assertEqual(len(self._logic.created), 1)

    def test_a_surface_beside_the_scan_does_not_either(self):
        """AutoMatrix's test folder holds a CBCT, a surface and a transform.
        The scan is what a clinician opened it to see; counting the three
        together would make it a cohort and show none of them."""
        folder = self._folder("patient1.nii.gz", "patient1.vtk", "matrix.tfm")

        self._pick(folder)

        self.assertEqual(_util.loaded,
                         [("volume", os.path.join(folder, "patient1.nii.gz"))])

    def test_a_scan_one_level_down_still_counts(self):
        """Per-patient subdirectories are the normal shape of a cohort, so the
        walk is recursive -- counting only the top level would report zero."""
        folder = self._folder(os.path.join("case_a", "scan.nii.gz"))

        self._pick(folder)

        self.assertEqual(len(_util.loaded), 1)

    def test_a_folder_holding_one_surface_is_that_surface(self):
        """No volume at all, one mesh: it behaves like the file it contains,
        and is loaded without being rendered -- a model has no CT curve."""
        folder = self._folder("Upper_gold.vtk")

        self._pick(folder)

        self.assertEqual(_util.loaded,
                         [("model", os.path.join(folder, "Upper_gold.vtk"))])
        self.assertEqual(self._logic.created, [])

    # -- a cohort -------------------------------------------------------

    def test_two_scans_are_a_cohort_and_show_nothing(self):
        """AREG's CBCT_FullyAuto is two timepoints. Showing one of them would
        put a patient in the 3D view with nothing on screen saying which."""
        folder = self._folder("T1.nii.gz", "T2.nii.gz")

        self._pick(folder)

        self.assertEqual(_util.loaded, [])
        self.assertEqual(self._logic.created, [])

    def test_a_folder_with_nothing_a_scene_can_hold_shows_nothing(self):
        folder = self._folder("notes.csv", "matrix.tfm")

        self._pick(folder)

        self.assertEqual(_util.loaded, [])

    def test_a_hidden_file_is_not_a_scan(self):
        """`.DS_Store` sits in every folder copied off a Mac. One counted as a
        second scan would silently turn a single patient into a cohort."""
        folder = self._folder("scan.nii.gz", ".hidden.nii.gz")

        self._pick(folder)

        self.assertEqual(len(_util.loaded), 1)

    def test_re_picking_the_folder_does_not_stack_a_second_copy(self):
        folder = self._folder("Pat_0002.nii.gz")

        panel = self._pick(folder)
        self._pick(folder, panel)

        self.assertEqual(len(_util.loaded), 1)
        self.assertEqual(len(self._logic.created), 1)

    # -- DICOM, the one scan spread over hundreds of files ---------------

    def test_a_dicom_directory_is_one_scan(self):
        """365 single-slice files are one 512x512x365 volume, and Slicer reads
        the series from any one of them. Counting files here would have made
        ASO's CBCT_FullyAuto_DCM a 365-patient cohort."""
        folder = self._folder(*["IMG%04d.dcm" % n for n in range(1, 366)])

        self._pick(folder)

        self.assertEqual(len(_util.loaded), 1)
        self.assertEqual(_util.loaded[0][0], "volume")
        self.assertTrue(_util.loaded[0][1].endswith("IMG0001.dcm"),
                        "the first slice is the archetype: %s" % _util.loaded[0][1])
        self.assertEqual(len(self._logic.created), 1, "and rendered like any CBCT")

    def test_the_series_takes_the_name_of_the_folder_that_was_picked(self):
        """Opened through a slice, it would otherwise arrive in the scene
        called `IMG0001`, which names nothing anyone chose."""
        folder = self._folder("IMG0001.dcm", "IMG0002.dcm")

        self._pick(folder)

        self.assertEqual(_util.nodes[-1].GetName(), os.path.basename(folder))

    def test_a_scan_picked_as_a_file_keeps_its_own_name(self):
        """The rename exists for the one case the file name is not the scan's.
        Renaming everything to its folder would lose the patient's file name,
        which is the thing a clinician recognises in the Data module."""
        folder = self._folder("Pat_0002.nii.gz")

        self._pick(os.path.join(folder, "Pat_0002.nii.gz"))

        self.assertEqual(_util.nodes[-1].GetName(), "Pat_0002.nii.gz")

    def test_two_dicom_directories_are_two_timepoints(self):
        """AREG's DICOM test file is `T1/<case>/` and `T2/<case>/`. Merging 884
        slices into one volume is precisely the accident the count avoids."""
        folder = self._folder(
            os.path.join("T1", "case_a", "IMG0001.dcm"),
            os.path.join("T2", "case_a", "IMG0001.dcm"),
        )

        self._pick(folder)

        self.assertEqual(_util.loaded, [])
        self.assertEqual(self._logic.created, [])

    def test_a_nifti_beside_a_dicom_tree_wins(self):
        """A converted scan sitting next to the slices it came from is one
        scan, and the NIfTI is the one to open."""
        folder = self._folder("scan.nii.gz", os.path.join("dcm", "IMG0001.dcm"))

        self._pick(folder)

        self.assertEqual(_util.loaded,
                         [("volume", os.path.join(folder, "scan.nii.gz"))])


class _Display:
    """A display node's visibility, view list and Modified observers."""

    def __init__(self, views=()):
        self.visible = True
        self.views = list(views)
        self.observers = {}

    def SetVisibility(self, visible):
        self.visible = bool(visible)
        for callback in list(self.observers.values()):
            callback(self, "ModifiedEvent")

    def GetVisibility(self):
        return self.visible

    def GetNumberOfViewNodeIDs(self):
        return len(self.views)

    def GetNthViewNodeID(self, index):
        return self.views[index]

    def RemoveAllViewNodeIDs(self):
        self.views = []

    def AddViewNodeID(self, view):
        self.views.append(view)

    def AddObserver(self, _event, callback):
        tag = len(self.observers) + 1
        self.observers[tag] = callback
        return tag

    def RemoveObserver(self, tag):
        self.observers.pop(tag, None)


class _SceneNode:
    def __init__(self, name, displays=1, hidden_from_editors=False):
        self.name = name
        self.displays = [_Display() for _ in range(displays)]
        self.hidden_from_editors = hidden_from_editors

    def GetName(self):
        return self.name

    def GetHideFromEditors(self):
        return self.hidden_from_editors

    def GetNumberOfDisplayNodes(self):
        return len(self.displays)

    def GetNthDisplayNode(self, index):
        return self.displays[index]


class HideSceneTest(unittest.TestCase):
    """A finished run's results go into a scene with nothing else showing.

    They landed on top of the inputs, the pauses' reviews and earlier runs,
    and what the run had just produced was the hard part to find."""

    class _Panel(ServerToolWidgetBase):
        TOOL_NAME = "AREG"
        _LOADABLE = (("*.nii.gz", "volume"), ("*.mrk.json", "markups"))

    def setUp(self):
        _util.loaded = []
        _util.shown = None
        self.panel = self._Panel.__new__(self._Panel)
        self.panel._producedRoot = "/out"
        self.panel._loadResultsCheckBox = None
        self.addCleanup(setattr, _util, "scene_nodes", {})

    def _scene(self, *nodes):
        _util.scene_nodes = {"vtkMRMLDisplayableNode": list(nodes)}

    def test_everything_already_shown_is_hidden_not_removed(self):
        """A CBCT with its volume rendering (two display nodes) and a mesh."""
        cbct, mesh = _SceneNode("input CBCT", displays=2), _SceneNode("input mesh")
        self._scene(cbct, mesh)
        self.panel._producedFiles = ["/out/case_Or.nii.gz"]
        self.panel._loadResults()

        self.assertFalse(any(d.visible for d in cbct.displays + mesh.displays))
        self.assertEqual(_util.loaded, [("volume", "/out/case_Or.nii.gz")])
        self.assertEqual(_util.shown[0].path, "/out/case_Or.nii.gz",
                         "the result's scan goes back into the emptied views")

    def test_slicers_own_nodes_are_left_alone(self):
        """The slice-plane models are hidden from editors and are not ours."""
        plane = _SceneNode("Red Volume Slice", hidden_from_editors=True)
        self._scene(plane)
        self.panel._producedFiles = ["/out/case_lm.mrk.json"]
        self.panel._loadResults()

        self.assertTrue(plane.displays[0].visible)

    def test_nothing_is_hidden_when_nothing_will_be_loaded(self):
        """Too many results, or none: the scene must not go blank for nothing."""
        mesh = _SceneNode("input mesh")
        self._scene(mesh)
        self.panel._producedFiles = ["/out/report.json"]
        self.panel._loadResults()

        self.assertTrue(mesh.displays[0].visible)

    def test_one_node_that_refuses_does_not_keep_the_others_shown(self):
        broken, mesh = _SceneNode("broken"), _SceneNode("input mesh")
        broken.GetNumberOfDisplayNodes = None
        self._scene(broken, mesh)
        self.panel._producedFiles = ["/out/case_lm.mrk.json"]
        self.panel._loadResults()

        self.assertFalse(mesh.displays[0].visible)
        self.assertEqual(len(_util.loaded), 1)


class ParkedOutOfTheViewsTest(unittest.TestCase):
    """Hidden is not enough for the 3D view's box: Slicer sizes it to every
    node displayable in the view, hidden or not, so at a pause after ASO it
    spanned the original CBCT and the oriented one."""

    def setUp(self):
        self.addCleanup(setattr, _util, "scene_nodes", {})

    def _hide(self, *nodes):
        _util.scene_nodes = {"vtkMRMLDisplayableNode": list(nodes)}
        slicer_io.hide_scene()

    def test_a_hidden_node_is_in_no_view(self):
        node = _SceneNode("input CBCT")
        self._hide(node)
        self.assertEqual(node.displays[0].views, [slicer_io._PARKED_VIEW_ID])

    def test_showing_it_again_gives_it_its_own_views_back(self):
        """Empty means every view; one limited to a view keeps that limit."""
        everywhere, one_view = _SceneNode("a"), _SceneNode("b")
        one_view.displays[0].views = ["vtkMRMLViewNode1"]
        self._hide(everywhere, one_view)

        everywhere.displays[0].SetVisibility(True)
        one_view.displays[0].SetVisibility(True)
        self.assertEqual(everywhere.displays[0].views, [])
        self.assertEqual(one_view.displays[0].views, ["vtkMRMLViewNode1"])
        self.assertEqual(everywhere.displays[0].observers, {}, "the watch ends once shown")

    def test_hiding_twice_keeps_the_readers_views(self):
        node = _SceneNode("a")
        node.displays[0].views = ["vtkMRMLViewNode1"]
        self._hide(node)
        self._hide(node)
        node.displays[0].SetVisibility(True)
        self.assertEqual(node.displays[0].views, ["vtkMRMLViewNode1"])
