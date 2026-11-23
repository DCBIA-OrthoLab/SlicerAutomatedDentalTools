"""Filling an input row from something already open in the scene.

There is ONE affordance for it, and it predates this work: the row's dropdown,
the same one that lists the tool's hosted test files. What was missing is that
it only ever offered scalar VOLUMES, and only to arguments whose declared
extensions named a volume format.

ALI fails both halves of that. Its single `input` declares no extensions -- it
takes a CBCT or an intraoral surface and the tool decides from the data -- so
it was offered nothing at all, and a scene holding fourteen `.vtk` surfaces
left the row with no scene entries.
"""

import os
import shutil
import tempfile
import sys
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: F401,E402 - installs the stubs
from test_hosted_test_files import _util  # noqa: E402

from ServerToolsCoreLib import formgen, slicer_io  # noqa: E402
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase  # noqa: E402


class _SceneNode:
    """A node as the panel handles one: a name and the class it answers to.

    `IsA` is what narrows the dropdown per argument and what decides the format
    the node is written back out in, so a double without it cannot cover the
    mesh case -- which is the case that was broken.
    """

    def __init__(self, name, node_class="vtkMRMLScalarVolumeNode"):
        self._name = name
        self._class = node_class
        # MRML's own "do not offer me in a chooser" flag, off for real data.
        self.hidden = False

    def GetName(self):
        return self._name

    def GetHideFromEditors(self):
        return self.hidden

    def IsA(self, node_class):
        return node_class == self._class


def _Volume(name):
    return _SceneNode(name)


def _Model(name):
    return _SceneNode(name, "vtkMRMLModelNode")


def _SlicePlane(colour):
    """One of the model nodes a slice view keeps for the plane it draws in 3D.

    Named exactly as Slicer names them, because that is what `IsSliceModelNode`
    reads -- these are not tagged, they are recognised.
    """
    return _SceneNode("%s Volume Slice" % colour, "vtkMRMLModelNode")


def _Hidden(name):
    """A node Slicer has flagged as scaffolding rather than data."""
    node = _SceneNode(name, "vtkMRMLModelNode")
    node.hidden = True
    return node


ALI = {"type": "path", "types": ["path", "folder"]}
SCAN = {"type": "path", "types": ["path"], "extensions": {"path": [".nii.gz"]}}
MESH = {"type": "path", "types": ["path"], "extensions": {"path": [".vtk", ".stl"]}}
TABLE = {"type": "csv_file", "types": ["csv_file"],
         "extensions": {"csv_file": [".csv"]}}


class SceneKindsTest(unittest.TestCase):
    """What the scene may answer with follows the ARGUMENT, read off the schema."""

    def test_an_argument_declaring_nothing_gets_nothing_from_the_schema(self):
        """ALI's shape -- and the schema cannot answer for it.

        `describe.py` publishes no extensions for any packaged tool, so reading
        "declares none" as "takes anything" would put a list of scans under
        every file row in the extension, spreadsheets included. The module says
        so instead: see `ServerToolWidgetBase.SCENE_INPUTS`.
        """
        self.assertEqual(formgen.scene_kinds_for(ALI), ())

    def test_a_scan_argument_offers_only_volumes(self):
        self.assertEqual(formgen.scene_kinds_for(SCAN), ("volume",))

    def test_a_mesh_argument_offers_only_models(self):
        self.assertEqual(formgen.scene_kinds_for(MESH), ("model",))

    def test_a_table_argument_offers_nothing(self):
        """A `.csv` input must never be offered a scan."""
        self.assertEqual(formgen.scene_kinds_for(TABLE), ())

    def test_the_row_is_wrapped_for_anything_the_scene_can_answer(self):
        """`accepts_volume` is what decides a row gets the dropdown at all.

        ALI's row has one anyway: it also offers hosted test files, which is
        the other reason `file_widget` wraps a row.
        """
        self.assertTrue(formgen.accepts_volume(SCAN))
        self.assertTrue(formgen.accepts_volume(MESH))
        self.assertFalse(formgen.accepts_volume(TABLE))


class DropdownContentsTest(unittest.TestCase):
    """What each row's dropdown is filled with, from one scene."""

    def _panel(self, scene_inputs=None, **arguments):
        class _P(ServerToolWidgetBase):
            TOOL_NAME = "ALI"
            SCENE_INPUTS = dict(scene_inputs or {})

        panel = _P.__new__(_P)
        panel._schema = {"arguments": arguments}
        panel._checkCanApply = lambda *a: None
        panel._inputWidgets = {
            name: formgen.file_widget(dict(spec, server_selectable="testfile"),
                                      "file_or_folder")
            for name, spec in arguments.items()
        }
        return panel

    def setUp(self):
        _util.scene_nodes = {}

    def test_a_scene_of_surfaces_is_offered_to_ali(self):
        """The reported case: fourteen `.vtk` files loaded, and a row that
        offered nothing."""
        _util.scene_nodes = {"vtkMRMLModelNode": [_Model("L01_T2_L"),
                                                  _Model("L02_T2_L")]}
        panel = self._panel(input=ALI,
                            scene_inputs={"input": ("volume", "model")})

        panel._refreshSceneVolumes()
        self.assertEqual(sorted(panel._inputWidgets["input"]._volume_names),
                         ["L01_T2_L", "L02_T2_L"])

    def test_volumes_and_surfaces_together(self):
        _util.scene_nodes = {"vtkMRMLScalarVolumeNode": [_Volume("Pat_0002")],
                             "vtkMRMLModelNode": [_Model("L01_T2_L")]}
        panel = self._panel(input=ALI,
                            scene_inputs={"input": ("volume", "model")})

        panel._refreshSceneVolumes()
        self.assertEqual(sorted(panel._inputWidgets["input"]._volume_names),
                         ["L01_T2_L", "Pat_0002"])

    def test_a_scan_row_is_not_offered_the_surfaces(self):
        _util.scene_nodes = {"vtkMRMLScalarVolumeNode": [_Volume("Pat_0002")],
                             "vtkMRMLModelNode": [_Model("L01_T2_L")]}
        panel = self._panel(input=SCAN)

        panel._refreshSceneVolumes()
        self.assertEqual(panel._inputWidgets["input"]._volume_names, ["Pat_0002"])

    def test_a_mesh_row_is_not_offered_the_scans(self):
        _util.scene_nodes = {"vtkMRMLScalarVolumeNode": [_Volume("Pat_0002")],
                             "vtkMRMLModelNode": [_Model("L01_T2_L")]}
        panel = self._panel(input=MESH)

        panel._refreshSceneVolumes()
        self.assertEqual(panel._inputWidgets["input"]._volume_names, ["L01_T2_L"])

    def test_two_rows_of_one_panel_are_narrowed_independently(self):
        _util.scene_nodes = {"vtkMRMLScalarVolumeNode": [_Volume("Pat_0002")],
                             "vtkMRMLModelNode": [_Model("L01_T2_L")]}
        panel = self._panel(scan=SCAN, mesh=MESH)

        panel._refreshSceneVolumes()
        self.assertEqual(panel._inputWidgets["scan"]._volume_names, ["Pat_0002"])
        self.assertEqual(panel._inputWidgets["mesh"]._volume_names, ["L01_T2_L"])


class SceneExportTest(unittest.TestCase):
    """A picked node becomes a file at UPLOAD time, in its own format."""

    class _Workspace:
        def __init__(self, root):
            self.root = root

        def file(self, name):
            return os.path.join(self.root, name)

    def _panel(self, spec, node):
        class _P(ServerToolWidgetBase):
            TOOL_NAME = "ALI"
            SCENE_INPUTS = {"input": ("volume", "model")}

        panel = _P.__new__(_P)
        panel._schema = {"arguments": {"input": spec}}
        panel._sceneVolumes = {"picked": node}
        panel._hiddenArgs = set()

        class _Row:
            @staticmethod
            def volume_name():
                return "picked"

        panel._inputWidgets = {"input": _Row()}
        return panel

    def setUp(self):
        _util.saved = []
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_scan_is_written_as_a_scan(self):
        panel = self._panel(ALI, _Volume("Pat_0002"))

        path = panel._prepareOneInputFile(self._Workspace(self.tmp), "input", "file_or_folder")
        self.assertTrue(path.endswith(".nii.gz"))

    def test_a_surface_is_written_as_a_surface(self):
        """Not `.nii.gz`: the tool could not read it."""
        panel = self._panel(ALI, _Model("L01_T2_L"))

        path = panel._prepareOneInputFile(self._Workspace(self.tmp), "input", "file_or_folder")
        self.assertTrue(path.endswith(".vtk"))

    def test_it_is_written_only_when_the_run_starts(self):
        """Not when the entry is picked: a cohort browsed through costs nothing
        until Apply, and the scene is the source of truth until then."""
        panel = self._panel(ALI, _Model("L01_T2_L"))
        self.assertEqual(_util.saved, [])

        panel._prepareOneInputFile(self._Workspace(self.tmp), "input", "file_or_folder")
        self.assertEqual(len(_util.saved), 1)

    def test_a_node_is_sent_under_its_own_name(self):
        """The server reads the patient and the jaw off the name: `AREG_ios.vtk`
        carried neither, and IOS to CBCT could pair nothing."""
        panel = self._panel(ALI, _Model("L01_T2_L"))

        path = panel._prepareOneInputFile(self._Workspace(self.tmp), "input", "file_or_folder")
        self.assertEqual(os.path.basename(path), "L01_T2_L.vtk")

    def test_each_argument_gets_its_own_folder(self):
        """Two nodes sharing a name, in two rows, must not overwrite each other."""
        panel = self._panel(ALI, _Volume("Pat_0002"))

        path = panel._prepareOneInputFile(self._Workspace(self.tmp), "input", "file_or_folder")
        self.assertEqual(os.path.dirname(path), os.path.join(self.tmp, "input"))

    def test_a_name_that_is_no_file_name_is_made_one(self):
        self.assertEqual(slicer_io.upload_name("Pat 0002 (T1)", ".nii.gz"), "Pat_0002_T1")
        self.assertEqual(slicer_io.upload_name("L01_T2_L.vtk", ".vtk"), "L01_T2_L")
        self.assertEqual(slicer_io.upload_name("Pat_0002.nii.gz", ".nii.gz"), "Pat_0002")

    def test_a_node_with_no_usable_name_falls_back_to_the_argument(self):
        panel = self._panel(ALI, _Model("()"))

        path = panel._prepareOneInputFile(self._Workspace(self.tmp), "input", "file_or_folder")
        self.assertEqual(os.path.basename(path), "ALI_input.vtk")


if __name__ == "__main__":
    unittest.main()


class TwoDropdownsTest(unittest.TestCase):
    """Two lists on the row, not one holding both sources.

    They answer different questions -- "fetch the tool's sample data" and "use
    what is already open in Slicer" -- and merged they read as one jumbled
    menu. Separating them also lets a row that cannot take a scene node simply
    not show that list, which a single list has no way to express.
    """

    SPEC = dict(ALI, server_selectable="testfile")

    def setUp(self):
        self.row = formgen.file_widget(self.SPEC, "file_or_folder")
        self.row.setChoices([{"name": "sample_scan.nii.gz",
                              "kind": "file", "size": 1024}])

    def test_a_row_the_scene_can_never_fill_does_not_show_the_list(self):
        """A transform or a spreadsheet has no counterpart in a scene. There is
        nothing to learn from the control, so it is not there."""
        self.assertFalse(self.row.sceneCombo.isVisible())

    def test_a_row_it_could_fill_offers_the_source_while_the_scene_is_empty(self):
        """Without the segment the feature stays invisible until the day the
        scene happens to hold the right kind -- and nobody discovers a control
        that is not there. That is exactly how it looked missing on every
        module but the one whose scene happened to match. Offered and greyed,
        the row says it CAN be filled this way, and why it cannot right now."""
        self.row.setSceneSupported(True)
        self.row.sourceButtons[formgen.ServerFileInput.SOURCE_SCENE].click()

        self.assertTrue(self.row.sceneCombo.isVisible())
        self.assertFalse(self.row.sceneCombo._enabled)

    def test_it_comes_alive_once_the_scene_has_something_to_offer(self):
        self.row.setSceneSupported(True)
        self.row.sourceButtons[formgen.ServerFileInput.SOURCE_SCENE].click()
        self.row.setVolumeChoices(["L01_T2_L"])

        self.assertTrue(self.row.sceneCombo.isVisible())
        self.assertTrue(self.row.sceneCombo._enabled)

    def test_the_two_lists_hold_their_own_source_only(self):
        self.row.setSceneSupported(True)
        self.row.setVolumeChoices(["L01_T2_L"])

        hosted = [self.row.combo.itemText(i) for i in range(self.row.combo.count)]
        scene = [self.row.sceneCombo.itemText(i)
                 for i in range(self.row.sceneCombo.count)]
        self.assertTrue(any("sample_scan" in e for e in hosted))
        self.assertFalse(any("L01_T2_L" in e for e in hosted))
        self.assertTrue(any("L01_T2_L" in e for e in scene))

    def test_picking_a_scene_node_resets_the_test_file_list(self):
        """One source at a time. A precedence rule the user cannot see is how
        you end up sending a file you thought you had replaced."""
        self.row.setVolumeChoices(["L01_T2_L"])
        self.row.combo.setCurrentIndex(1)

        self.row.sceneCombo.setCurrentIndex(1)

        self.assertEqual(self.row.volume_name(), "L01_T2_L")
        self.assertEqual(self.row.combo.currentIndex, 0)

    def test_picking_a_test_file_resets_the_scene_list(self):
        self.row.setVolumeChoices(["L01_T2_L"])
        self.row.sceneCombo.setCurrentIndex(1)

        self.row.combo.setCurrentIndex(1)

        self.assertEqual(self.row.volume_name(), "")
        self.assertEqual(self.row.sceneCombo.currentIndex, 0)

    def test_a_browsed_path_resets_both(self):
        """Nothing is typed into the row any more -- the two browse dialogs and
        `set_local_path` are its only writers -- but a path arriving from any
        of them still has to let go of both lists."""
        self.row.setVolumeChoices(["L01_T2_L"])
        self.row.sceneCombo.setCurrentIndex(1)

        self.row.local.setCurrentPath("/data/mine.vtk")

        self.assertEqual(self.row.sceneCombo.currentIndex, 0)
        self.assertEqual(self.row.combo.currentIndex, 0)
        self.assertEqual(self.row.currentPath, "/data/mine.vtk")


class EmptyListsTest(unittest.TestCase):
    """A dropdown that can only ever offer nothing is not shown at all.

    Most arguments host no test files and take no scene node, so before this
    every file row in the extension carried a dead list beside its buttons. A
    control a user opens once and finds empty is a control they stop trusting.
    """

    def test_a_plain_file_argument_shows_neither_list(self):
        row = formgen.file_widget(dict(SCAN, server_selectable="testfile"),
                                  "single_file")

        self.assertFalse(row.combo.isVisible())
        self.assertFalse(row.sceneCombo.isVisible())

    def test_the_hosted_source_appears_once_the_server_offers_one(self):
        row = formgen.file_widget(dict(SCAN, server_selectable="testfile"),
                                  "single_file")
        self.assertNotIn(formgen.ServerFileInput.SOURCE_HOSTED, row.sourceButtons)

        row.setChoices([{"name": "sample_scan.nii.gz", "kind": "file", "size": 1}])

        self.assertIn(formgen.ServerFileInput.SOURCE_HOSTED, row.sourceButtons)
        row.sourceButtons[formgen.ServerFileInput.SOURCE_HOSTED].click()
        self.assertTrue(row.combo.isVisible())

    def test_it_goes_again_when_the_server_offers_nothing(self):
        """A tool whose DATA/ folder was emptied between two runs."""
        row = formgen.file_widget(dict(SCAN, server_selectable="testfile"),
                                  "single_file")
        row.setChoices([{"name": "sample_scan.nii.gz", "kind": "file", "size": 1}])
        row.sourceButtons[formgen.ServerFileInput.SOURCE_HOSTED].click()

        row.setChoices([])

        self.assertNotIn(formgen.ServerFileInput.SOURCE_HOSTED, row.sourceButtons)
        self.assertFalse(row.combo.isVisible())


class CaptionNamesItsSourceTest(unittest.TestCase):
    """The row's second line says WHICH source satisfied it.

    "sample_scan.nii.gz - NIfTI volume, 94 MB" does not distinguish a file
    fetched from the server from one the user browsed to, and a path alone does
    not distinguish either from a node picked out of the scene. The label does,
    and it is the first word on the line.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.row = formgen.file_widget(dict(ALI, server_selectable="testfile"),
                                       "file_or_folder")

    def _file(self, name, size=2048):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as handle:
            handle.write(b"x" * size)
        return path

    def test_nothing_chosen_says_so(self):
        self.assertEqual(self.row.caption.text, formgen.NOTHING_CHOSEN)

    def test_a_browsed_file_is_labelled_File(self):
        formgen.set_local_path(self.row, self._file("scan.nii.gz"))

        self.assertTrue(self.row.caption.text.startswith("File: "),
                        self.row.caption.text)

    def test_a_browsed_file_shows_its_name_and_keeps_the_path_one_hover_away(self):
        """It showed the whole path, back when it had a line of its own and
        wrapped. The box is half a row wide now, and a path elided into it is
        the one part of itself that means nothing: `/tmp/tmpgt7qr_f5/pati...`
        where a reader is looking for `scan.nii.gz`."""
        path = self._file("scan.nii.gz")
        formgen.set_local_path(self.row, path)

        self.assertIn("scan.nii.gz", self.row.caption.text)
        self.assertNotIn(self.dir, self.row.caption.text)
        self.assertEqual(self.row.local.container.toolTip(), path)

    def test_a_browsed_folder_is_labelled_Folder_and_says_what_it_holds(self):
        self._file("a.vtk")
        self._file("b.vtk")
        formgen.set_local_path(self.row, self.dir)

        text = self.row.caption.text
        self.assertTrue(text.startswith("Folder: "), text)
        self.assertIn("2 VTK surfaces", text)

    def test_a_fetched_test_file_is_labelled_Test_File(self):
        """And by its hosted NAME: the path it landed on is a session directory
        swept on exit, and a user who takes it for their own copy will look for
        it later."""
        self.row.setChoices([{"name": "scan.nii.gz", "kind": "file", "size": 2048}])
        formgen.set_local_path(self.row, self._file("scan.nii.gz"))

        self.assertTrue(self.row.caption.text.startswith("Test File: "),
                        self.row.caption.text)

    def test_a_scene_pick_is_labelled_by_its_kind(self):
        self.row.setSceneLabel("ROI")
        self.row.setVolumeChoices(["Crop box"])
        self.row.sceneCombo.setCurrentIndex(1)

        self.assertEqual(self.row.caption.text, "ROI: Crop box")

    def test_a_volume_and_a_surface_are_both_simply_a_scan(self):
        """Read off the kinds' own labels rather than off their number: ALI
        takes a CBCT or an intraoral surface through one argument, and both are
        a scan. Counting would have answered "Scene" -- the container rather
        than the thing -- on the one row where a true word exists."""
        self.assertEqual(formgen.scene_label_for(("volume", "model")), "Scan")
        self.assertEqual(formgen.scene_label_for(("volume",)), "Scan")
        self.assertEqual(formgen.scene_label_for(("roi",)), "ROI")
        self.assertEqual(formgen.scene_label_for(("volume", "roi")), "Scene")

    def test_the_dropdown_says_what_it_holds(self):
        self.assertEqual(formgen.scene_prompt_for("Scan"), "Imported scan...")
        self.assertEqual(formgen.scene_prompt_for("ROI"), "Drawn ROI...")

    def test_every_row_has_the_second_line_even_without_dropdowns(self):
        """A `.csv` argument is wrapped in nothing, and used to show nothing at
        all once the path field went."""
        plain = formgen.file_widget(TABLE, "single_file")

        self.assertNotIsInstance(plain, formgen.ServerFileInput)
        self.assertEqual(plain.caption.text, formgen.NOTHING_CHOSEN)
        formgen.set_local_path(plain, self._file("notes.csv"))
        self.assertTrue(plain.caption.text.startswith("File: "))


class ScenePermissionTest(unittest.TestCase):
    """Only SCANS may come from the scene -- a CBCT volume or an intraoral
    surface. Landmarks may not, and were offered until 2026-09-24: a set of
    points one row below a scan in the same dropdown reads as another scan,
    and picking the wrong one is a run that fails on a file the tool cannot
    open."""

    def test_a_volume_and_a_surface_may(self):
        self.assertEqual(formgen.scene_kinds_for(SCAN), ("volume",))
        self.assertEqual(formgen.scene_kinds_for(MESH), ("model",))

    def test_landmarks_may_not(self):
        self.assertEqual(
            formgen.scene_kinds_for(
                {"types": ["path"], "extensions": {"path": [".mrk.json"]}}),
            ())

    def test_nothing_else_may(self):
        """A spreadsheet or an archive has no counterpart in a scene, and
        offering one would be offering a conversion nobody asked for."""
        for spec in (TABLE,
                     {"types": ["zip_file"], "extensions": {"zip_file": [".zip"]}},
                     {"types": ["file"], "extensions": {"file": [".txt"]}}):
            self.assertEqual(formgen.scene_kinds_for(spec), ())


class ValueFieldLeadsTheRowTest(unittest.TestCase):
    """What the row holds is the FIRST thing on the control line, and the one
    control that changes it is the last.

    It began life on a line of its own underneath, and before that nested
    inside the picker's own container -- where it started where the browse
    buttons did, indented past the dropdowns, reading as a caption for the
    buttons rather than for the row. No test looked at WHERE it was, which is
    why both of those shipped.
    """

    WRAPPED = dict(ALI, server_selectable="testfile")

    def test_a_wrapped_row_leads_its_control_line_with_it(self):
        row = formgen.file_widget(self.WRAPPED, "file_or_folder")

        controls = row.container.layout.widgets[1]
        self.assertIs(controls.layout.widgets[0], row.caption,
                      "the value is not the first thing on the control line")
        self.assertNotIn(row.caption, row.container.layout.widgets)

    def test_the_picker_no_longer_holds_it(self):
        """Two layouts both believing they hold one widget is how a caption
        ends up drawn in the wrong place, or twice."""
        row = formgen.file_widget(self.WRAPPED, "file_or_folder")

        self.assertNotIn(row.caption,
                         getattr(row.local.container.layout, "widgets", []))

    def test_an_unwrapped_row_keeps_it_on_its_own_line(self):
        """There is nothing to wrap it in, so it is already at the left edge."""
        plain = formgen.file_widget(TABLE, "single_file")

        self.assertIs(plain.container.layout.widgets[0], plain.caption)

    def test_the_one_button_closes_the_control_line(self):
        """`Select`, on the right, whatever the row accepts: which dialog it
        opens follows the segmented control above it, so the button never has
        to name a kind the pressed segment already names."""
        row = formgen.file_widget(self.WRAPPED, "file_or_folder")

        controls = row.container.layout.widgets[1]
        self.assertIn(row.local.selectButton, controls.layout.widgets)
        self.assertEqual(row.local.selectButton.text, formgen.SELECT_LABEL)


class NameVocabularyTest(unittest.TestCase):
    """Which arguments the scene may fill, decided from their NAME.

    Not a preference: `describe.py` publishes no extensions for any packaged
    tool, so the schema cannot narrow a row on its own and every file argument
    in the extension would be offered nothing. The name is the only signal
    left, and it lives on this side for the reason PRETTY_NAMES does -- it is
    a dental vocabulary, and the server is built not to hold one.
    """

    SPEC = {"type": "path", "types": ["path"]}

    def _kinds(self, name):
        return formgen.scene_kinds_for(self.SPEC, name)

    def test_a_scan_argument_offers_volumes(self):
        for name in ("scans", "cbct", "t1_masks", "masks"):
            self.assertEqual(self._kinds(name), ("volume",), name)

    def test_a_landmark_argument_offers_nothing(self):
        """And the rule that answers it is FIRST in the table: `cbct_landmarks`
        holds `cbct`, so falling through would offer it the scene's volumes --
        the one answer that is certainly wrong."""
        for name in ("landmarks", "cbct_landmarks", "ios_landmarks",
                     "mgl_landmarks"):
            self.assertEqual(self._kinds(name), (), name)

    def test_a_surface_argument_offers_models(self):
        for name in ("meshes", "surfaces", "ios"):
            self.assertEqual(self._kinds(name), ("model",), name)

    def test_an_ambiguous_name_offers_both(self):
        """`t1` is a CBCT volume in AREG_CBCT and GreedyReg and an intraoral
        SURFACE in AREG_IOS. One name, two meanings, and the scene listing is
        the honest answer -- what the user picks is what they meant."""
        self.assertEqual(self._kinds("t1"), ("volume", "model"))
        self.assertEqual(self._kinds("t2"), ("volume", "model"))
        self.assertEqual(formgen.scene_label_for(self._kinds("t1")), "Scan")

    def test_a_transform_or_a_table_offers_nothing(self):
        """No counterpart in a scene, and guessing one wrong is worse than
        offering none."""
        for name in ("transforms", "initial_transforms", "measurements",
                     "reference", "something_new"):
            self.assertEqual(self._kinds(name), (), name)

    def test_a_declared_extension_still_wins_over_the_name(self):
        """The schema is authoritative wherever it speaks: a tool that does
        publish its formats is not second-guessed by a word."""
        spec = dict(self.SPEC, extensions={"path": [".csv"]})

        self.assertEqual(formgen.scene_kinds_for(spec, "scans"), ())

    def test_every_scan_argument_in_the_extension_is_covered(self):
        """The ask, stated as a test: every argument that takes a scan can be
        filled from the scene -- not ALI's alone. Landmark arguments are
        deliberately absent from this list; see ScenePermissionTest."""
        covered = {
            "scans": ("volume",), "cbct": ("volume",), "masks": ("volume",),
            "t1_masks": ("volume",), "meshes": ("model",),
            "surfaces": ("model",), "ios": ("model",),
        }
        for name, kinds in covered.items():
            self.assertEqual(self._kinds(name), kinds, name)


class TheSceneFurnitureIsNotOfferedTest(unittest.TestCase):
    """A Slicer scene holds more than what was loaded into it.

    Every slice view keeps a model node for the plane it draws in the 3D view --
    `Red Volume Slice`, `Yellow Volume Slice`, `Green Volume Slice`. They are
    there from the moment Slicer opens, before anyone has loaded anything, so a
    surface argument's dropdown led with three rectangles that no tool could use
    and that sat above the meshes the user actually had.
    """

    def _panel(self, **arguments):
        class _P(ServerToolWidgetBase):
            TOOL_NAME = "ALI"

        panel = _P.__new__(_P)
        panel._schema = {"arguments": arguments}
        panel._checkCanApply = lambda *a: None
        panel._inputWidgets = {
            name: formgen.file_widget(dict(spec, server_selectable="testfile"),
                                      "file_or_folder")
            for name, spec in arguments.items()
        }
        return panel

    def setUp(self):
        _util.scene_nodes = {}

    def _names(self, panel):
        panel._refreshSceneVolumes()
        return sorted(panel._inputWidgets["input"]._volume_names)

    def test_the_three_slice_planes_are_not_offered_as_surfaces(self):
        _util.scene_nodes = {"vtkMRMLModelNode": [
            _SlicePlane("Red"), _SlicePlane("Yellow"), _SlicePlane("Green"),
            _Model("upper_arch_segmented"),
        ]}

        self.assertEqual(self._names(self._panel(input=MESH)),
                         ["upper_arch_segmented"])

    def test_a_scene_holding_only_them_offers_nothing(self):
        """Which is what an empty scene really looks like: not zero model
        nodes, but three. The row must read as empty, not as "three meshes"."""
        _util.scene_nodes = {"vtkMRMLModelNode": [
            _SlicePlane("Red"), _SlicePlane("Yellow"), _SlicePlane("Green"),
        ]}

        self.assertEqual(self._names(self._panel(input=MESH)), [])

    def test_a_node_hidden_from_editors_is_not_offered_either(self):
        """The general case beside the slice planes: a node Slicer has already
        flagged as not belonging in a chooser."""
        _util.scene_nodes = {"vtkMRMLModelNode": [_Hidden("internal"),
                                                  _Model("upper_arch_segmented")]}

        self.assertEqual(self._names(self._panel(input=MESH)),
                         ["upper_arch_segmented"])

    def test_a_mesh_of_the_users_own_is_still_offered(self):
        """The filter has to be narrow: refusing one node too many hides a
        surface someone loaded, which is worse than the problem it fixes."""
        _util.scene_nodes = {"vtkMRMLModelNode": [_Model("Red_arch_T2")]}

        self.assertEqual(self._names(self._panel(input=MESH)), ["Red_arch_T2"])


def _Roi(name):
    """A crop box as Slicer makes one: NOT a markups fiducial.

    `vtkMRMLMarkupsROINode.IsA("vtkMRMLMarkupsFiducialNode")` is 0 -- verified
    against a real Slicer -- so a row offered "markups" is offered no box at
    all. That is why it is a kind of its own.
    """
    return _SceneNode(name, "vtkMRMLMarkupsROINode")


ROI = {"type": "path", "types": ["path"]}


class ARoiIsItsOwnKindTest(unittest.TestCase):
    """AutoCrop3D crops to a box the clinician DRAWS in Slicer, so the box is
    already a node when the panel opens. Asking for it as a file would mean
    saving it first, for no reason."""

    def _panel(self, **arguments):
        class _P(ServerToolWidgetBase):
            TOOL_NAME = "AutoCrop3D"
            SCENE_INPUTS = {"roi": ("roi",)}

        panel = _P.__new__(_P)
        panel._schema = {"arguments": arguments}
        panel._checkCanApply = lambda *a: None
        panel._inputWidgets = {
            name: formgen.file_widget(dict(spec, server_selectable="testfile"),
                                      "file_or_folder")
            for name, spec in arguments.items()
        }
        return panel

    def setUp(self):
        _util.scene_nodes = {}

    def test_the_argument_name_alone_answers_roi(self):
        """`describe.py` publishes no extensions for a packaged tool, so the
        name is all there is to go on -- and `roi` means nothing to the generic
        rule."""
        self.assertEqual(formgen.scene_kinds_for(ROI, "roi"), ("roi",))

    def test_a_box_in_the_scene_is_offered(self):
        _util.scene_nodes = {"vtkMRMLMarkupsROINode": [_Roi("Crop box")]}
        panel = self._panel(roi=ROI)

        panel._refreshSceneVolumes()
        self.assertEqual(panel._inputWidgets["roi"]._volume_names, ["Crop box"])

    def test_landmarks_are_not_a_box(self):
        """A fiducial list is points, not an extent. Offering one here would
        let a clinician send something the tool cannot read as a box."""
        _util.scene_nodes = {
            "vtkMRMLMarkupsFiducialNode": [_SceneNode("L", "vtkMRMLMarkupsFiducialNode")],
        }
        panel = self._panel(roi=ROI)

        panel._refreshSceneVolumes()
        self.assertEqual(panel._inputWidgets["roi"]._volume_names, [])

    def test_a_scan_row_beside_it_is_narrowed_independently(self):
        """The two rows of AutoCrop3D: one takes the volume, one takes the box,
        and each is offered only its own."""
        _util.scene_nodes = {
            "vtkMRMLScalarVolumeNode": [_Volume("Pat_0002")],
            "vtkMRMLMarkupsROINode": [_Roi("Crop box")],
        }
        panel = self._panel(scans=SCAN, roi=ROI)

        panel._refreshSceneVolumes()
        self.assertEqual(panel._inputWidgets["scans"]._volume_names, ["Pat_0002"])
        self.assertEqual(panel._inputWidgets["roi"]._volume_names, ["Crop box"])

    def test_the_row_says_what_it_takes(self):
        self.assertEqual(formgen.scene_label_for(("roi",)), "ROI")

    def test_a_box_is_written_back_out_as_markups(self):
        """The file format is the same as a fiducial list's -- only the node
        class differs -- so a picked box uploads as `.mrk.json`."""
        self.assertEqual(formgen.SCENE_NODE_KINDS["roi"][1], ".mrk.json")


class PickingFromTheSceneReEvaluatesApplyTest(unittest.TestCase):
    """An imported scan leaves NO local path behind -- it is exported at
    upload time -- so the only thing that can announce it is the scene list
    itself. It was the one source `connect_changed` did not listen to, and
    Apply stayed grey over a row the user had just filled."""

    def _row(self):
        """A row switched to its scene source, the way a click on the segment
        leaves it."""
        row = formgen.file_widget(SCAN, "single_file", "scans")
        row.setSceneSupported(True)
        row.setVolumeChoices(["CBCT_patient1"])
        row.sourceButtons[formgen.ServerFileInput.SOURCE_SCENE].click()
        return row

    def test_choosing_one_announces_itself(self):
        row = self._row()
        seen = []
        formgen.connect_changed(row, lambda *args: seen.append(1))

        row.sceneCombo.setCurrentIndex(1)

        self.assertEqual(row.volume_name(), "CBCT_patient1")
        self.assertTrue(seen, "the row was filled and nothing was told")

    def test_switching_source_away_from_it_announces_that_too(self):
        """Switching source empties the row, which is exactly the moment Apply
        has to go back to grey."""
        row = self._row()
        row.sceneCombo.setCurrentIndex(1)
        seen = []
        formgen.connect_changed(row, lambda *args: seen.append(1))

        row.sourceButtons[formgen.ServerFileInput.SOURCE_FILE].click()

        self.assertEqual(row.volume_name(), "")
        self.assertTrue(seen, "the row was emptied and nothing was told")
