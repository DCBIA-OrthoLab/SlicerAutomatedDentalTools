"""Unit tests for ServerToolsCoreLib.formgen - run outside Slicer, `qt`/`ctk`/
`slicer` stubbed (see qt_stubs.py).

Everything here is driven by EXAMPLE_TOOL_SCHEMA, the real `GET /tools` entry
for `example_tool`: it is the one tool exercising every argument shape the
panel has to render - free text, int, float, a single-choice dropdown, a
multi-choice checkbox group, and a file argument that also accepts a folder.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_formgen.py
"""

import json
import io
import os
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import qt_stubs

qt, ctk = qt_stubs.install()

from ServerToolsCoreLib import design, formgen
from ServerToolsCoreLib.client import ToolServerClient

# The server's actual GET /tools payload for example_tool, verbatim.
EXAMPLE_TOOL_SCHEMA = {
    "name": "example_tool",
    "output_kind": "files",
    "arguments": {
        "label": {
            "type": "str", "types": ["str"], "required": True,
            "description": "Free-text label for this run",
            "server_selectable": None, "choices": None, "initial": None,
        },
        "input": {
            "type": "csv_file", "types": ["csv_file", "folder"], "required": True,
            "description": "A single .csv file, or a folder of .csv/.xlsx/.ods files sent as a .zip archive",
            "server_selectable": None, "choices": None, "initial": None,
        },
        "threshold": {
            "type": "float", "types": ["float"], "required": True,
            "description": "Numeric threshold parameter",
            "server_selectable": None, "choices": None, "initial": None,
        },
        # `initial` is null here on purpose: run()'s own default is None
        # ("not specified"), so there is no value to pre-fill. Contrast with
        # PREFILLED_SCHEMA below.
        "iterations": {
            "type": "int", "types": ["int"], "required": False,
            "description": "Optional number of iterations",
            "server_selectable": None, "choices": None, "initial": None,
        },
        "outputs": {
            "type": "multichoice", "types": ["multichoice"], "required": False,
            "description": "Which result files to produce",
            "server_selectable": None, "initial": None,
            "choices": {"summary": True, "preview": True, "columns": False},
        },
        "preview_format": {
            "type": "choice", "types": ["choice"], "required": False,
            "description": "Format of the preview file",
            "server_selectable": None, "initial": None,
            "choices": {"csv": True, "json": False},
        },
    },
}


def _build():
    layout = qt.QFormLayout()
    return formgen.build(EXAMPLE_TOOL_SCHEMA["arguments"], layout), layout


# Scalar arguments whose tool declares an `initial`, the way AMASSS declares
# surface_smoothing=5. Shaped like a real GET /tools payload.
PREFILLED_SCHEMA = {
    "smoothing": {
        "type": "int", "types": ["int"], "required": False,
        "description": "Smoothing iterations", "server_selectable": None,
        "choices": None, "initial": 5,
    },
    "ratio": {
        "type": "float", "types": ["float"], "required": False,
        "description": "A ratio", "server_selectable": None,
        "choices": None, "initial": 0.25,
    },
    "enabled": {
        "type": "bool", "types": ["bool"], "required": False,
        "description": "A flag on by default", "server_selectable": None,
        "choices": None, "initial": True,
    },
    "suffix": {
        "type": "str", "types": ["str"], "required": False,
        "description": "A name suffix", "server_selectable": None,
        "choices": None, "initial": "Pred",
    },
}



class HiddenArgumentTest(unittest.TestCase):
    """`hidden` is never rendered, whatever else the panel holds.

    It carries the arguments a clinician has no business being asked -- which
    CUDA device, nnUNet's tile step size -- named by the deployment rather than
    by the tool. The tool still declares them and still applies its own
    defaults; the client simply does not ask.
    """

    def test_a_hidden_argument_is_not_visible(self):
        self.assertFalse(formgen.is_visible({"type": "float", "hidden": True}, {}))

    def test_hidden_beats_a_satisfied_visible_when(self):
        spec = {"type": "float", "hidden": True, "visible_when": {"mode": "CBCT"}}
        self.assertFalse(formgen.is_visible(spec, {"mode": "CBCT"}))

    def test_an_argument_without_the_key_is_unaffected(self):
        self.assertTrue(formgen.is_visible({"type": "float"}, {}))
        self.assertTrue(formgen.is_visible({"type": "float", "hidden": False}, {}))

class ScalarInitialValueTest(unittest.TestCase):
    """A scalar argument's `initial` reaches its widget.

    This is not cosmetic. collect() sends EVERY widget, so a field the user
    never touched still travels: a spin box left at Qt's own 0 sent 0, and the
    tool's Python default never applied. That is what shipped AMASSS surfaces
    with 0 smoothing iterations while its run() signature read 5.
    """

    def setUp(self):
        self.layout = qt.QFormLayout()
        self.widgets = formgen.build(PREFILLED_SCHEMA, self.layout)

    def test_int_starts_at_the_declared_value(self):
        self.assertEqual(self.widgets["smoothing"].value, 5)

    def test_float_starts_at_the_declared_value(self):
        self.assertAlmostEqual(self.widgets["ratio"].value, 0.25)

    def test_bool_starts_checked_when_declared_true(self):
        self.assertTrue(self.widgets["enabled"].isChecked())

    def test_str_starts_at_the_declared_value(self):
        self.assertEqual(self.widgets["suffix"].text, "Pred")

    def test_collect_returns_the_declared_values_untouched(self):
        """The whole point: an untouched form sends the tool's own defaults."""
        self.assertEqual(
            formgen.collect(self.widgets),
            {"smoothing": 5, "ratio": 0.25, "enabled": True, "suffix": "Pred"},
        )

    def test_absent_initial_leaves_the_qt_default(self):
        """`initial: None` must not be coerced -- iterations means "unset"."""
        layout = qt.QFormLayout()
        widgets = formgen.build(
            {"iterations": EXAMPLE_TOOL_SCHEMA["arguments"]["iterations"]}, layout
        )
        self.assertEqual(widgets["iterations"].value, 0)


class ChoiceWidgetTest(unittest.TestCase):
    """`"choice"`: one QComboBox, items in declaration order, the single true
    entry preselected."""

    def setUp(self):
        self.widgets, self.layout = _build()
        self.combo = self.widgets["preview_format"]

    def test_is_a_combobox(self):
        self.assertIsInstance(self.combo, qt.QComboBox)

    def test_items_are_the_choice_keys_in_declaration_order(self):
        self.assertEqual([self.combo.itemText(i) for i in range(self.combo.count)], ["csv", "json"])

    def test_initial_selection_is_the_true_entry(self):
        self.assertEqual(self.combo.currentText, "csv")
        self.assertEqual(formgen.collect(self.widgets)["preview_format"], "csv")

    def test_reads_back_the_selected_option_name(self):
        self.combo.setCurrentText("json")

        self.assertEqual(formgen.collect(self.widgets)["preview_format"], "json")

    def test_selection_is_sent_in_clear(self):
        self.combo.setCurrentText("json")
        data = ToolServerClient._stringify(formgen.collect(self.widgets))

        self.assertEqual(data["preview_format"], "json")

    def test_true_entry_is_preselected_wherever_it_sits(self):
        # Nothing may assume the default is the first option.
        spec = {"type": "choice", "choices": {"a": False, "b": False, "c": True}}
        widget = formgen._make_widget("k", spec)

        self.assertEqual(widget.currentText, "c")

    def test_description_becomes_the_tooltip(self):
        self.assertEqual(self.combo.toolTip(), "Format of the preview file")


class MultiChoiceWidgetTest(unittest.TestCase):
    """`"multichoice"`: one QCheckBox per option, in declaration order, each
    starting at its declared boolean."""

    def setUp(self):
        self.widgets, self.layout = _build()
        self.group = self.widgets["outputs"]

    def test_is_a_multichoice_group(self):
        self.assertIsInstance(self.group, formgen.MultiChoiceGroup)

    def test_one_checkbox_per_option_in_declaration_order(self):
        self.assertEqual(list(self.group.boxes), ["summary", "preview", "columns"])
        for box in self.group.boxes.values():
            self.assertIsInstance(box, qt.QCheckBox)

    def test_checkboxes_are_laid_out_in_declaration_order(self):
        laid_out = [w for w in self.group.container.layout.widgets if isinstance(w, qt.QCheckBox)]
        self.assertEqual([box.text for box in laid_out], ["summary", "preview", "columns"])

    def test_nothing_but_the_options_is_printed_in_the_field(self):
        """The description used to be rendered here, as a small grey paragraph
        above the boxes. Several of those stacked down a panel is text a reader
        scrolls past; it is the label's tooltip now."""
        printed = [w.text for w in self.group.container.layout.widgets
                   if getattr(w, "text", None)]
        self.assertNotIn("Which result files to produce", printed)

    def test_initial_state_matches_the_declared_booleans(self):
        self.assertEqual(
            [self.group.boxes[option].isChecked() for option in ("summary", "preview", "columns")],
            [True, True, False],
        )

    def test_reads_back_the_full_state_after_toggling(self):
        self.group.boxes["preview"].setChecked(False)
        self.group.boxes["columns"].setChecked(True)

        self.assertEqual(
            formgen.collect(self.widgets)["outputs"],
            {"summary": True, "preview": False, "columns": True},
        )

    def test_every_option_is_reported_even_when_none_is_checked(self):
        # Server-side, what is sent *is* the selection: an option left out of
        # the payload counts as unchecked whatever its declared default, and
        # omitting the argument entirely is what applies the defaults. So
        # "everything unchecked" has to be spelled out in full, not dropped.
        for box in self.group.boxes.values():
            box.setChecked(False)

        self.assertEqual(
            formgen.collect(self.widgets)["outputs"],
            {"summary": False, "preview": False, "columns": False},
        )

    def test_the_block_has_air_under_it(self):
        """A multichoice is several rows tall where every other field is one, so
        its last option sat as close to the NEXT argument's label as its own
        options sit to each other. The other three margins stay zero: the row's
        label has to line up with the first option, not with a gap."""
        group = formgen.MultiChoiceGroup({"MERGED": True, "SEPARATE": False})
        left, top, right, bottom = group.container.layout.margins
        self.assertEqual((left, top, right), (0, 0, 0))
        self.assertGreater(bottom, 0)


    def test_the_description_is_hovered_on_the_label(self):
        """Not on the container, which Qt would hand to every child that has
        none -- ALI's 304-character note on `landmarks` popped up under each of
        its 236 chips that way. The label has no children to hand it down to,
        and is where a reader looks for what a field means."""
        label = dict((field, label) for label, field in self.layout.rows)[
            self.group.container]

        self.assertEqual(label.toolTip(), "Which result files to produce")
        self.assertFalse(self.group.container.toolTip())


class MultiChoiceEncodingTest(unittest.TestCase):
    """How the checkbox state reaches the wire (client._stringify)."""

    def setUp(self):
        self.widgets, _ = _build()
        self.group = self.widgets["outputs"]

    def test_sent_as_json(self):
        self.group.boxes["preview"].setChecked(False)
        self.group.boxes["columns"].setChecked(True)

        data = ToolServerClient._stringify(formgen.collect(self.widgets))

        self.assertEqual(
            json.loads(data["outputs"]),
            {"summary": True, "preview": False, "columns": True},
        )

    def test_json_uses_lowercase_booleans_not_python_repr(self):
        data = ToolServerClient._stringify(formgen.collect(self.widgets))

        self.assertIn('"summary": true', data["outputs"])
        self.assertNotIn("True", data["outputs"])

    def test_all_unchecked_is_not_an_empty_field(self):
        for box in self.group.boxes.values():
            box.setChecked(False)

        data = ToolServerClient._stringify(formgen.collect(self.widgets))

        self.assertEqual(json.loads(data["outputs"]), {"summary": False, "preview": False, "columns": False})

    def test_option_name_containing_a_comma_survives(self):
        # Why JSON and not the server's `a,b` shortcut: that spelling is for
        # curl and cannot represent an option name with a comma in it.
        group = formgen.MultiChoiceGroup({"a,b": True, "c": False})

        payload = ToolServerClient._stringify({"outputs": group.value()})["outputs"]

        self.assertEqual(json.loads(payload), {"a,b": True, "c": False})


class OtherArgumentTypesTest(unittest.TestCase):
    """`choices: null` arguments must produce neither of the two new widgets."""

    def setUp(self):
        self.widgets, self.layout = _build()

    def test_str_int_float_are_unchanged(self):
        self.assertIsInstance(self.widgets["label"], qt.QLineEdit)
        self.assertIsInstance(self.widgets["threshold"], qt.QDoubleSpinBox)
        self.assertIsInstance(self.widgets["iterations"], qt.QSpinBox)

    def test_no_choice_widget_for_a_null_choices_argument(self):
        for name in ("label", "threshold", "iterations"):
            widget = self.widgets[name]
            self.assertNotIsInstance(widget, qt.QComboBox, name)
            self.assertNotIsInstance(widget, qt.QCheckBox, name)
            self.assertNotIsInstance(widget, formgen.MultiChoiceGroup, name)

    def test_file_argument_is_not_a_generated_field(self):
        # "input" is provided by base_widget from FILE_INPUTS, not by build().
        self.assertNotIn("input", self.widgets)

    def test_every_other_argument_gets_exactly_one_row(self):
        self.assertEqual(
            [name for name, _spec in EXAMPLE_TOOL_SCHEMA["arguments"].items() if name != "input"],
            list(self.widgets),
        )
        self.assertEqual(len(self.layout.rows), len(self.widgets))

    def test_collect_returns_one_entry_per_generated_field(self):
        collected = formgen.collect(self.widgets)

        self.assertEqual(set(collected), set(self.widgets))
        self.assertEqual(collected["label"], "")
        self.assertEqual(collected["threshold"], 0.0)

    def test_required_flag_drives_the_apply_button(self):
        arguments = EXAMPLE_TOOL_SCHEMA["arguments"]
        self.assertFalse(formgen.all_required_filled(self.widgets, arguments))  # "label" still empty

        self.widgets["label"].setText("run-1")

        self.assertTrue(formgen.all_required_filled(self.widgets, arguments))


class MultiChoiceRequiredTest(unittest.TestCase):
    def test_an_all_unchecked_required_multichoice_still_counts_as_filled(self):
        schema = {"outputs": {"type": "multichoice", "required": True, "choices": {"a": False, "b": False}}}
        layout = qt.QFormLayout()
        widgets = formgen.build(schema, layout)

        self.assertTrue(formgen.all_required_filled(widgets, schema))


class ChangeSignalTest(unittest.TestCase):
    def setUp(self):
        self.widgets, _ = _build()
        self.calls = []

    def test_toggling_any_checkbox_notifies(self):
        formgen.connect_changed(self.widgets["outputs"], lambda *_a: self.calls.append(1))

        self.widgets["outputs"].boxes["columns"].setChecked(True)

        self.assertEqual(len(self.calls), 1)

    def test_changing_the_choice_notifies(self):
        formgen.connect_changed(self.widgets["preview_format"], lambda *_a: self.calls.append(1))

        self.widgets["preview_format"].setCurrentText("json")

        self.assertEqual(len(self.calls), 1)


class AutoFileModeTest(unittest.TestCase):
    """The one rule deciding what a file argument's picker looks like - and,
    downstream, whether base_widget zips the selection before uploading."""

    def test_folder_plus_a_file_type_offers_both(self):
        self.assertEqual(formgen.auto_file_mode(EXAMPLE_TOOL_SCHEMA["arguments"]["input"]), "file_or_folder")

    def test_no_folder_means_a_single_file(self):
        self.assertEqual(formgen.auto_file_mode({"types": ["nifti_file"]}), "single_file")

    def test_folder_only_means_a_zipped_folder(self):
        self.assertEqual(formgen.auto_file_mode({"types": ["folder"]}), "folder_zip")

    def test_an_unknown_argument_falls_back_to_a_single_file(self):
        # What base_widget resolves against when the schema could not be loaded.
        self.assertEqual(formgen.auto_file_mode({}), "single_file")


class FileInputModesTest(unittest.TestCase):
    """Which arguments get an input row, and with which picker, is the
    schema's answer; a module's FILE_INPUTS only overrides it."""

    ARGUMENTS = EXAMPLE_TOOL_SCHEMA["arguments"]

    def test_every_file_argument_is_offered_without_being_declared(self):
        self.assertEqual(formgen.file_input_modes(self.ARGUMENTS), {"input": "file_or_folder"})

    def test_scalar_arguments_are_not_file_inputs(self):
        modes = formgen.file_input_modes(self.ARGUMENTS)

        for name in ("label", "threshold", "iterations", "outputs", "preview_format"):
            self.assertNotIn(name, modes)

    def test_declaring_auto_changes_nothing(self):
        # What ExampleTool used to spell out by hand.
        self.assertEqual(
            formgen.file_input_modes(self.ARGUMENTS, {"input": "auto"}),
            formgen.file_input_modes(self.ARGUMENTS),
        )

    def test_an_override_wins_over_the_derived_mode(self):
        # SurgMovPred: the schema types "input" as a zip_file, the module wants
        # to hand the user a folder picker and zip it client-side.
        arguments = {"input": {"type": "zip_file", "types": ["zip_file"], "required": True}}

        self.assertEqual(formgen.file_input_modes(arguments), {"input": "single_file"})
        self.assertEqual(
            formgen.file_input_modes(arguments, {"input": "folder_zip"}), {"input": "folder_zip"}
        )

    def test_a_scene_node_input_can_only_come_from_an_override(self):
        # AMASSS: the server declares a nifti_file; that it is filled from a
        # MRML volume node is knowledge the server does not have.
        arguments = {"file": {"type": "nifti_file", "types": ["nifti_file"], "required": True}}

        self.assertEqual(
            formgen.file_input_modes(arguments, {"file": "volume_node"}), {"file": "volume_node"}
        )

    def test_none_leaves_an_optional_file_argument_out(self):
        arguments = {
            "input": {"type": "csv_file", "types": ["csv_file"], "required": True},
            "attachment": {"type": "file", "types": ["file"], "required": False},
        }

        self.assertEqual(
            formgen.file_input_modes(arguments, {"attachment": "none"}), {"input": "single_file"}
        )

    def test_rows_follow_schema_order_even_when_overridden(self):
        arguments = {
            "first": {"type": "csv_file", "types": ["csv_file"]},
            "second": {"type": "zip_file", "types": ["zip_file"]},
        }

        modes = formgen.file_input_modes(arguments, {"first": "folder_zip"})

        self.assertEqual(list(modes), ["first", "second"])

    def test_no_schema_yields_no_inputs(self):
        self.assertEqual(formgen.file_input_modes({}), {})


class ResultKindTest(unittest.TestCase):
    """RESULT_KIND is derived from the tool's output_kind, except where the
    server genuinely cannot answer."""

    def test_derived_from_output_kind(self):
        self.assertEqual(formgen.result_kind_for("text"), "text")
        self.assertEqual(formgen.result_kind_for("segmentation"), "segmentation")
        self.assertEqual(formgen.result_kind_for("files"), "save_as")

    def test_example_tool_needs_no_declaration(self):
        self.assertEqual(formgen.result_kind_for(EXAMPLE_TOOL_SCHEMA["output_kind"]), "save_as")

    def test_a_single_file_defaults_to_saving_it(self):
        # "file" is the ambiguous one: the server says a file comes back, not
        # whether it is a volume, a mesh, or something to write to disk.
        self.assertEqual(formgen.result_kind_for("file"), "save_as")

    def test_a_declaration_always_wins(self):
        self.assertEqual(formgen.result_kind_for("file", "volume"), "volume")
        self.assertEqual(formgen.result_kind_for("files", "text"), "text")

    def test_an_unknown_or_missing_output_kind_falls_back_to_text(self):
        # Also the no-schema case (an unreachable server).
        self.assertEqual(formgen.result_kind_for(None), "text")
        self.assertEqual(formgen.result_kind_for("something_new"), "text")


class FileOrFolderInputTest(unittest.TestCase):
    """The `input` argument: `types` = ["csv_file", "folder"] - one row taking
    either, with the client working out which it got.

    **There is nothing to type a path into.** The row shows what it holds in a
    READ-ONLY box on the left and a `Select` button on the right -- the
    ordinary file-picker shape -- and an editable field there would be a box a
    clinician can type a path into which is then ignored, since the dialog and
    `set_local_path` are the only writers. `currentPath` is what the rest of
    the panel reads, and the full path is one hover away on the container.
    """

    def setUp(self):
        self.spec = EXAMPLE_TOOL_SCHEMA["arguments"]["input"]
        self.field = formgen.file_widget(self.spec)
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work, True)
        self.csv = os.path.join(self.work, "measurements.csv")
        with open(self.csv, "w") as fh:
            fh.write("a,b\n1,2\n")
        self.folder = os.path.join(self.work, "cohort")
        os.makedirs(self.folder)

    def test_folder_in_types_gives_one_row_accepting_both(self):
        """An argument taking a file OR a folder is ONE row with ONE button -
        not two rows, and not a kind selector the user has to set correctly
        before picking. Which dialog the button opens follows the segmented
        control above it."""
        self.assertIsInstance(self.field, formgen.FileOrFolderInput)
        self.assertEqual(self.field.modes, ("file", "folder"))
        self.assertEqual(self.field.selectButton.text, formgen.SELECT_LABEL)

    def test_a_single_file_argument_accepts_no_folder(self):
        """A source whose result the server is going to refuse is worse than no
        source at all: the folder is zipped and uploaded first, and the run
        fails after the bytes have travelled."""
        field = formgen.file_widget({"type": "csv_file", "types": ["csv_file"]})

        self.assertEqual(field.modes, ("file",))

    def test_a_folder_argument_accepts_no_file(self):
        """The mirror image: an argument taking a whole folder (zipped on the
        way out) must not invite a single file it cannot use -- and its one
        button must open the folder dialog without being told."""
        field = formgen.file_widget({"type": "folder", "types": ["folder"]})

        self.assertEqual(field.modes, ("folder",))
        qt.QFileDialog.next_directory = self.folder
        self.addCleanup(setattr, qt.QFileDialog, "next_directory", "")
        field.selectButton.clicked.emit()
        self.assertEqual(field.currentPath, self.folder)

    def test_there_is_nothing_to_type_a_path_into(self):
        """The dialog and `set_local_path` are the only writers.

        Pinned rather than left to the class docstring: the box on the left IS
        a QLineEdit now, and one that accepted typing would take a path the row
        then ignores.
        """
        self.assertFalse(hasattr(self.field, "pathEdit"))
        self.assertTrue(self.field.caption.isReadOnly())

    def test_the_file_dialog_is_restricted_to_the_declared_extensions(self):
        # The extensions still come from `types` - that is the whole point of
        # driving the dialog here rather than letting ctkPathLineEdit do it.
        self.field.selectButton.clicked.emit()

        self.assertIn("*.csv", qt.QFileDialog.last_open_file_args[3])

    def test_a_generic_file_argument_is_left_unrestricted(self):
        """A `file` argument declares no extension, and no restriction must
        stay no restriction: a dialog handed "Supported files ()" shows the
        user an empty folder and no way out of it.

        Read off the dialog the button opens, the extensions no longer having a
        widget of their own to be read back from.
        """
        field = formgen.file_widget({"type": "file", "types": ["file"]})

        field.selectButton.clicked.emit()

        self.assertEqual(field._extensions, ())
        self.assertEqual(qt.QFileDialog.last_open_file_args[3], "")

    def test_nothing_selected_is_not_a_folder(self):
        self.assertEqual(self.field.currentPath, "")
        self.assertFalse(self.field.is_folder())

    def test_a_file_is_detected_as_a_file(self):
        self.field.setCurrentPath(self.csv)

        self.assertEqual(self.field.currentPath, self.csv)
        self.assertFalse(self.field.is_folder())

    def test_a_folder_is_detected_as_a_folder(self):
        # The user says nothing: picking the path *is* saying which it is.
        self.field.setCurrentPath(self.folder)

        self.assertTrue(self.field.is_folder())

    def test_a_folder_chosen_after_a_file_is_still_detected(self):
        # A kind selector made this a wrong request: a folder left under
        # "File" was uploaded as a file and failed at open(). The kind is read
        # off the filesystem on every selection, so it cannot go stale.
        self.field.setCurrentPath(self.csv)
        self.assertFalse(self.field.is_folder())

        self.field.setCurrentPath(self.folder)

        self.assertTrue(self.field.is_folder())

    def test_surrounding_whitespace_is_ignored(self):
        # Nobody types here any more, but a path still arrives from elsewhere:
        # `set_local_path` when a download lands, or a module filling its own
        # row. A trailing newline makes `is_folder` answer no for a folder, and
        # the upload then fails at open().
        self.field.setCurrentPath(f"  {self.folder}\n")

        self.assertEqual(self.field.currentPath, self.folder)
        self.assertTrue(self.field.is_folder())

    def test_a_nonexistent_path_is_not_taken_for_a_folder(self):
        self.field.setCurrentPath(os.path.join(self.work, "gone"))

        self.assertFalse(self.field.is_folder())

    def test_the_one_button_opens_whichever_dialog_the_mode_asks_for(self):
        qt.QFileDialog.next_directory = self.folder
        qt.QFileDialog.next_file = self.csv
        self.addCleanup(setattr, qt.QFileDialog, "next_directory", "")
        self.addCleanup(setattr, qt.QFileDialog, "next_file", "")

        self.field.setBrowseMode("folder")
        self.field.selectButton.clicked.emit()
        self.assertEqual(self.field.currentPath, self.folder)
        self.assertTrue(self.field.is_folder())

        self.field.setBrowseMode("file")
        self.field.selectButton.clicked.emit()
        self.assertEqual(self.field.currentPath, self.csv)
        self.assertFalse(self.field.is_folder())

    def test_a_mode_it_does_not_know_leaves_the_button_alone(self):
        """It is set from a source key, and a key this row never offered must
        not silently turn a file picker into a folder picker."""
        self.field.setBrowseMode("file")
        self.field.setBrowseMode("scene")

        self.assertEqual(self.field._mode, "file")

    def test_a_cancelled_dialog_keeps_the_current_selection(self):
        self.field.setCurrentPath(self.csv)
        qt.QFileDialog.next_directory = ""  # what Qt returns when cancelled
        qt.QFileDialog.next_file = ""

        self.field.setBrowseMode("folder")
        self.field.selectButton.clicked.emit()
        self.field.setBrowseMode("file")
        self.field.selectButton.clicked.emit()

        self.assertEqual(self.field.currentPath, self.csv)

    def test_every_selection_notifies_whatever_its_kind(self):
        # Regression: on a ctkPathLineEdit, a *.csv name filter silences
        # currentPathChanged for every folder (and every non-matching file),
        # so the Apply button would never enable after choosing a folder. The
        # row keeps its own listener list now, and it has to fire for all
        # three ways a path arrives: browsed, or written in from outside.
        calls = []
        formgen.connect_changed(self.field, lambda *_a: calls.append(1))
        qt.QFileDialog.next_directory = self.folder
        self.addCleanup(setattr, qt.QFileDialog, "next_directory", "")

        self.field.setCurrentPath(self.csv)
        self.field.setBrowseMode("folder")
        self.field.selectButton.clicked.emit()
        self.field.setCurrentPath(os.path.join(self.work, "other.xlsx"))

        self.assertEqual(len(calls), 3)

    def test_the_same_path_written_twice_notifies_once(self):
        """A notification re-runs the readiness check and rewrites the caption,
        and `set_local_path` is called on every refresh of a row -- including
        ones that change nothing. Re-announcing a selection nobody made is how
        a panel ends up redrawing itself in a loop."""
        calls = []
        formgen.connect_changed(self.field, lambda *_a: calls.append(1))

        self.field.setCurrentPath(self.csv)
        self.field.setCurrentPath(self.csv)

        self.assertEqual(len(calls), 1)

    def test_an_explicit_mode_overrides_the_schema_rule(self):
        # SurgMovPred's "input" is typed zip_file and the module still wants a
        # folder picker (it zips it): a declared mode wins over the derived one,
        # and decides which source the row accepts at all.
        field = formgen.file_widget({"type": "zip_file", "types": ["zip_file"]}, "folder_zip")

        self.assertIsInstance(field, formgen.FileOrFolderInput)
        self.assertEqual(field.modes, ("folder",))
        self.assertEqual(field._mode, "folder")

    def test_a_volume_argument_gets_the_sources_dropdown_around_its_picker(self):
        field = formgen.file_widget({"type": "nifti_file", "types": ["nifti_file"]})

        # accepts_volume: a scan input can also be satisfied by a volume open
        # in the scene, so its picker comes wrapped in the sources dropdown.
        self.assertIsInstance(field, formgen.ServerFileInput)
        self.assertIsInstance(field.local, formgen.FileOrFolderInput)
        self.assertEqual(field.local.modes, ("file",))

        # Wrapping it changes nothing about what the local half offers: the
        # declared extensions still reach the dialog.
        field.local.selectButton.clicked.emit()

        self.assertIn("Supported files (*.nii *.nii.gz)",
                      qt.QFileDialog.last_open_file_args[3])

    def test_a_non_volume_file_argument_gets_a_plain_row(self):
        """A csv input cannot be satisfied by a scan open in the scene, so it
        gets no sources dropdown at all - the row IS the picker."""
        field = formgen.file_widget({"type": "csv_file", "types": ["csv_file"]})

        self.assertIsInstance(field, formgen.FileOrFolderInput)
        self.assertNotIsInstance(field, formgen.ServerFileInput)


# ---------------------------------------------------------------------------
# Presentation hints (ArgSpec.section / visible_when / ui / groups)
# ---------------------------------------------------------------------------

def _multichoice(choices, **hints):
    spec = {
        "type": "multichoice", "types": ["multichoice"], "required": False,
        "description": "", "server_selectable": None, "choices": choices,
        "initial": None,
    }
    spec.update(hints)
    return spec


# Four options, two groups, one option left out of every group on purpose.
_LAYOUT_CHOICES = {"a": True, "b": False, "c": True, "d": False}
_LAYOUT_GROUPS = {"First": ["a", "b"], "Second": ["c"]}


class MultiChoiceLayoutTest(unittest.TestCase):
    """The four layouts must be indistinguishable from the outside.

    This is the property that makes a presentation hint safe: a layout the
    client renders badly is ugly, never wrong on the wire. Everything
    downstream (collect, connect_changed, all_required_filled, and the JSON
    client.py builds) reads `boxes` and `value()`, so those must not depend on
    where the boxes were put.
    """

    def _group(self, layout, groups=None):
        return formgen.MultiChoiceGroup(_LAYOUT_CHOICES, layout=layout, groups=groups)

    def test_every_layout_reads_back_identically(self):
        for layout, groups in ((None, None), ("inline", None),
                               ("grid", _LAYOUT_GROUPS), ("tabs", _LAYOUT_GROUPS)):
            group = self._group(layout, groups)
            self.assertEqual(list(group.boxes), list(_LAYOUT_CHOICES), layout)
            self.assertEqual(group.value(), _LAYOUT_CHOICES, layout)

    def test_a_grouped_layout_keeps_declaration_order_not_group_order(self):
        # "Second" holds "c", declared third; the read-back order stays the
        # schema's, because that is the order the server matches against.
        group = self._group("tabs", {"Second": ["c"], "First": ["a", "b"]})
        self.assertEqual(list(group.boxes), ["a", "b", "c", "d"])

    def test_an_option_left_out_of_every_group_is_still_offered(self):
        # The server rejects a group naming an unknown option, but not an
        # option no group mentions. Dropping it would hide a selection the
        # tool genuinely offers.
        group = self._group("tabs", _LAYOUT_GROUPS)
        self.assertIn("d", group.boxes)

    def test_an_unknown_layout_falls_back_to_the_flat_column(self):
        # A presentation hint from a newer server must never break an older
        # client: the panel still renders, just plainly.
        group = self._group("carousel")
        self.assertEqual(group.value(), _LAYOUT_CHOICES)

    def test_tabs_makes_one_tab_per_group(self):
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)]
        self.assertEqual(len(tabs), 1)
        self.assertEqual([title for title, _w in tabs[0].tabs], ["First", "Second", "Other"])

    def test_a_tab_packs_its_options_at_the_top_left(self):
        """Otherwise the grid shares the scroll area's height between its rows.

        Measured in Slicer on ALI's cranial base: eleven 20 px check boxes sat
        94 px apart, three sparse lines floating in a tall empty box, which is
        what "the table looks ugly, badly proportioned" describes. Qt has no
        pack flag; a trailing stretched row and column is the idiom.
        """
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        for _title, tab in tabs.tabs:
            # tab -> [scroll area, the group toggle]; the grid is in the area.
            area = tab.layout.widgets[0]
            grid = area.widget.layout
            rows = max((row for row, _column in grid.cells), default=-1) + 1
            self.assertEqual(grid.rowStretch.get(rows), 1,
                             "the spare height must go below the options")
            columns = formgen._columns_for(_LAYOUT_CHOICES)
            self.assertEqual(grid.columnStretch.get(columns), 1,
                             "and the spare width to their right")

    def test_the_tab_box_is_bounded_both_ways(self):
        """A minimum alone let the panel's spare vertical space stretch the box
        -- 380 px for ALI's ten cranial landmarks, mostly empty."""
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        self.assertEqual(tabs.minimumHeight(), tabs.maximumHeight())
        self.assertGreaterEqual(tabs.maximumHeight(), design.TABS_MIN_HEIGHT)
        self.assertLessEqual(tabs.maximumHeight(), design.TABS_MAX_HEIGHT)

    def test_the_box_follows_the_tab_on_screen(self):
        """Sizing every tab to the tallest one still left ALI's four-landmark
        tab in a box built for fifty-seven. The box follows the number of boxes,
        which is what the reader is actually looking at."""
        few = ["a"]
        many = ["opt{}".format(i) for i in range(40)]
        choices = {option: False for option in few + many}
        group = formgen.MultiChoiceGroup(
            choices, layout="tabs", groups={"Few": few, "Many": many})
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        small = tabs.maximumHeight()
        tabs.setCurrentIndex(1)
        large = tabs.maximumHeight()

        self.assertLess(small, large)
        self.assertEqual(small, design.tabs_height_for(1))

    def test_the_height_comes_back_when_the_small_tab_does(self):
        few, many = ["a"], ["opt{}".format(i) for i in range(40)]
        group = formgen.MultiChoiceGroup(
            {option: False for option in few + many},
            layout="tabs", groups={"Few": few, "Many": many})
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        tabs.setCurrentIndex(1)
        tabs.setCurrentIndex(0)

        self.assertEqual(tabs.maximumHeight(), design.tabs_height_for(1))

    def test_a_long_catalogue_is_capped_rather_than_pushing_apply_off_screen(self):
        many = {"opt{}".format(i): False for i in range(200)}
        group = formgen.MultiChoiceGroup(many, layout="tabs", groups=None)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        self.assertEqual(tabs.maximumHeight(), design.TABS_MAX_HEIGHT)

    def test_a_short_catalogue_no_longer_gets_a_tall_empty_box(self):
        """The floor used to be 220 px whatever the content held."""
        few = {"a": False, "b": True}
        group = formgen.MultiChoiceGroup(few, layout="tabs", groups=None)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        self.assertEqual(tabs.maximumHeight(), design.tabs_height_for(1))
        self.assertLess(tabs.maximumHeight(), 220)

    def _buttons_of(self, tabs, index):
        """tab -> [scroll area, the button pair]. {label: button}."""
        tab = tabs.tabs[index][1]
        bar = tab.layout.widgets[1]
        return {button.text: button for button in bar.layout.widgets}

    def test_a_tab_can_be_taken_in_one_click(self):
        """The original extension's `Switch group selection`, restored.

        Dropping it made a ten-landmark region cost ten clicks, which is what
        "ticking a region turns on every landmark in that region" is asking
        for. Scoped to the tab: the other groups are untouched.
        """
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]
        first = _LAYOUT_GROUPS["First"]

        # What the OTHER group held before, so the assertion is "untouched"
        # rather than "off" -- `c` is on by default in this catalogue.
        others = {option: group.boxes[option].isChecked()
                  for option in _LAYOUT_GROUPS["Second"]}

        self._buttons_of(tabs, 0)[formgen.SELECT_GROUP_LABEL].clicked.emit()

        for option in first:
            self.assertTrue(group.boxes[option].isChecked(), option)
        for option, before in others.items():
            self.assertEqual(group.boxes[option].isChecked(), before, option)

    def test_deselect_all_clears_only_its_own_tab(self):
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]
        others = {option: group.boxes[option].isChecked()
                  for option in _LAYOUT_GROUPS["Second"]}

        self._buttons_of(tabs, 0)[formgen.CLEAR_GROUP_LABEL].clicked.emit()

        for option in _LAYOUT_GROUPS["First"]:
            self.assertFalse(group.boxes[option].isChecked(), option)
        for option, before in others.items():
            self.assertEqual(group.boxes[option].isChecked(), before, option)

    def test_each_tab_names_both_actions_rather_than_one_that_changes(self):
        """A single toggle has to say which of the two a click will do, so its
        label moves under the pointer as the group fills. Two named actions are
        always true."""
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]
        buttons = self._buttons_of(tabs, 0)

        self.assertEqual(set(buttons),
                         {formgen.SELECT_GROUP_LABEL, formgen.CLEAR_GROUP_LABEL})

        buttons[formgen.SELECT_GROUP_LABEL].clicked.emit()

        self.assertEqual(set(self._buttons_of(tabs, 0)), set(buttons),
                         "the labels must not move under the pointer")

    def test_the_pair_reads_as_add_and_take_away(self):
        """Blue adds, red takes away -- the extension's own vocabulary, where
        Apply is blue and Cancel is red. Grey was tried first and read as
        disabled: two evenly weighted slabs of DISABLED_BG next to each other
        look like a control that is off."""
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]
        buttons = self._buttons_of(tabs, 0)
        tokens = design.tokens()

        # Asserted by ROLE, not by hex: the colours live in design.py's stop
        # tables and a restyle there must not have to be restated here.
        self.assertEqual(buttons[formgen.SELECT_GROUP_LABEL]._stylesheet,
                         design._button_stylesheet("primary", tokens))
        self.assertEqual(buttons[formgen.CLEAR_GROUP_LABEL]._stylesheet,
                         design._button_stylesheet("danger", tokens))

    def test_the_group_buttons_change_nothing_on_the_wire(self):
        """Whatever the layout does, `value()` is still the complete state."""
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        self._buttons_of(tabs, 0)[formgen.SELECT_GROUP_LABEL].clicked.emit()

        self.assertEqual(set(group.value()), set(_LAYOUT_CHOICES))
        self.assertTrue(all(group.value()[option] for option in _LAYOUT_GROUPS["First"]))

    def test_a_dense_option_is_the_label_itself(self):
        """A check box puts an 18 px target beside the word the clinician is
        reading. Over 119 landmarks that is a chore, and a grid of small ticks
        does not read as selected-or-not at a glance."""
        group = self._group("tabs", _LAYOUT_GROUPS)
        chip = group.boxes["a"]

        self.assertIsInstance(chip, qt.QPushButton)
        self.assertEqual(chip.text, "a")
        self.assertTrue(chip._checkable, "it has to remain a real toggle")

    def test_a_chip_reads_back_exactly_as_a_check_box_did(self):
        """The invariant every layout here is held to: a wrong layout may be
        ugly, it is never wrong on the wire."""
        group = self._group("tabs", _LAYOUT_GROUPS)

        self.assertEqual(group.value(), _LAYOUT_CHOICES)
        group.boxes["b"].setChecked(True)
        self.assertTrue(group.value()["b"])
        for box in group.boxes.values():
            box.setChecked(False)
        self.assertFalse(any(group.value().values()))

    def test_a_short_catalogue_gets_more_columns_than_a_long_one(self):
        """One number had to be chosen for the worst case, and wasted half the
        width on every short catalogue: `Ba`, `S`, `N` fit six across where
        `UR3OIP` fits four."""
        self.assertGreater(formgen._columns_for(["Ba", "S", "N", "RPo"]),
                           formgen._columns_for(["UR3OIP", "LFZyg", "RFZyg"]))
        self.assertLessEqual(formgen._columns_for(["x" * 40]), formgen._MAX_COLUMNS)
        self.assertGreaterEqual(formgen._columns_for(["x" * 40]), formgen._MIN_COLUMNS)

    def test_a_tab_of_short_labels_gets_more_columns_than_one_of_long(self):
        """One count for the whole argument had to be the worst case, and wasted
        half the width on every short region. It changes only the arrangement
        INSIDE the box, which already resizes with the tab."""
        short = ["Ba", "S", "N", "RPo", "LPo", "C2"]
        long = ["UR3OIPxx", "LFZygxxx", "RFZygxxx", "UL6Oxxxx"]
        group = formgen.MultiChoiceGroup(
            {option: False for option in short + long},
            layout="tabs", groups={"Short": short, "Long": long})
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]

        def columns_of(index):
            grid = tabs.tabs[index][1].layout.widgets[0].widget.layout
            return max(column for _row, column in grid.cells) + 1

        self.assertGreater(columns_of(0), columns_of(1))

    def test_the_columns_are_further_apart_than_the_rows(self):
        """Chips carry their own padding: touching columns read as one long
        word, touching rows read as a list."""
        group = self._group("tabs", _LAYOUT_GROUPS)
        tabs = [w for w in group.container.layout.widgets if isinstance(w, qt.QTabWidget)][0]
        grid = tabs.tabs[0][1].layout.widgets[0].widget.layout

        self.assertGreater(grid.horizontalSpacing, grid.verticalSpacing)

    def test_a_small_list_keeps_the_native_check_box(self):
        """Slicer is the application around this panel. Two or three options are
        a check box's own idiom, and restyling them buys nothing."""
        group = self._group("inline")
        self.assertIsInstance(group.boxes["a"], qt.QCheckBox)

    def test_the_chart_stretches_its_rows_but_never_its_columns(self):
        """The columns ARE the arch. Spreading them across whatever width the
        panel happens to have destroys the adjacency the layout exists to show,
        which is why only the rows take the slack here."""
        grid = self._chart_grid(self._group("grid", _LAYOUT_GROUPS))

        self.assertEqual(grid.rowStretch.get(grid.rowCount()), 1)
        self.assertEqual(grid.columnStretch, {})

    def test_the_chart_is_drawn_on_a_table_surface(self):
        """A tabbed layout gets its frame from QTabWidget::pane; this one has
        no pane, and without a frame thirty-two chips sat on the panel with no
        edge saying where the table stopped."""
        group = self._group("grid", _LAYOUT_GROUPS)
        frame = group.container.layout.widgets[-1]

        self.assertIn("tableFrame", frame._stylesheet)
        self.assertIn(design.tokens()["SURFACE_TABLE"], frame._stylesheet)
        self.assertIn("1px solid {}".format(design.tokens()["BORDER"]),
                      frame._stylesheet)

    @staticmethod
    def _chart_grid(group):
        """The chart's own QGridLayout, through the table frame it now sits
        in."""
        frame = group.container.layout.widgets[-1]
        area = [w for w in frame.layout.widgets
                if isinstance(w, qt.QScrollArea)][0]
        return area.widget.layout

    def test_no_layout_carries_a_global_selection_bar(self):
        """`All` / `None` / `Default` were three small links under the options.
        They read as a different control language from everything around them,
        and on a tabbed group they duplicated the per-tab button that does the
        same thing. A group's own widgets are now the only way to select.
        """
        for layout, groups in ((None, None), ("inline", None),
                               ("grid", _LAYOUT_GROUPS), ("tabs", _LAYOUT_GROUPS)):
            group = self._group(layout, groups)
            texts = [getattr(w, "text", "") for w in group.container.layout.widgets]
            for gone in ("All", "None", "Default"):
                self.assertNotIn(gone, texts, layout)

    def test_the_layout_reaches_the_widget_through_the_schema(self):
        # Not just constructible by hand: _make_widget has to read `ui`/`groups`
        # off the spec, or a tool declaring them renders flat anyway.
        widgets = formgen.build(
            {"picks": _multichoice(_LAYOUT_CHOICES, ui="tabs", groups=_LAYOUT_GROUPS)},
            qt.QFormLayout(),
        )
        tabs = [w for w in widgets["picks"].container.layout.widgets
                if isinstance(w, qt.QTabWidget)]
        self.assertEqual(len(tabs), 1)


class FacadeGroupsTest(unittest.TestCase):
    """A facade puts two engines behind one argument, and they do not group the
    same options the same way. ALI's four anatomical regions are not another
    spelling of its five intraoral families."""

    SPEC = {
        "type": "multichoice", "types": ["multichoice"], "required": False,
        "choices": {"Ba": False, "S": False, "L0MG": False, "UR1O": False},
        "ui": "tabs",
        "groups": {"Cranial base": ["Ba", "S"]},
        "options_when": {"mode": {"CBCT": ["Ba", "S"],
                                  "Intraoral scan": ["L0MG", "UR1O"]}},
        "groups_when": {"mode": {"CBCT": {"Cranial base": ["Ba", "S"]},
                                 "Intraoral scan": {"Mucogingival Lower": ["L0MG"],
                                                    "Occlusal Upper": ["UR1O"]}}},
    }

    def test_the_groups_follow_the_mode(self):
        self.assertEqual(
            list(formgen.allowed_groups(self.SPEC, {"mode": "Intraoral scan"})),
            ["Mucogingival Lower", "Occlusal Upper"])
        self.assertEqual(list(formgen.allowed_groups(self.SPEC, {"mode": "CBCT"})),
                         ["Cranial base"])

    def test_no_rule_means_render_the_declared_groups(self):
        self.assertIsNone(formgen.allowed_groups({"groups": {"a": ["x"]}}, {}))

    def test_a_mode_the_rule_does_not_name_falls_back_to_the_declared_groups(self):
        self.assertIsNone(formgen.allowed_groups(self.SPEC, {"mode": "Something else"}))

    def test_rebuilding_keeps_what_survives_and_defaults_the_rest(self):
        """Switching mode and back must not silently clear a selection."""
        group = formgen.MultiChoiceGroup(
            {"Ba": False, "S": True}, layout="tabs", groups={"Cranial base": ["Ba", "S"]})
        group.boxes["Ba"].setChecked(True)

        group.rebuild({"Ba": False, "L0MG": True}, {"Mucogingival Lower": ["L0MG"]})

        self.assertEqual(set(group.boxes), {"Ba", "L0MG"})
        self.assertTrue(group.boxes["Ba"].isChecked(), "a surviving option keeps its state")
        self.assertTrue(group.boxes["L0MG"].isChecked(), "a new one takes its declared default")

    def test_rebuilding_with_the_same_options_redraws_nothing(self):
        group = formgen.MultiChoiceGroup({"a": True, "b": False}, layout="tabs")
        before = group.boxes["a"]

        group.rebuild({"a": True, "b": False}, None)

        self.assertIs(group.boxes["a"], before)

    def test_the_group_still_reads_back_the_complete_state(self):
        group = formgen.MultiChoiceGroup({"a": True, "b": False}, layout="tabs")

        group.rebuild({"b": False, "c": True}, None)

        self.assertEqual(set(group.value()), {"b", "c"})


class LabelTest(unittest.TestCase):
    """The words a user reads are the tool's, not this file's."""

    def test_the_declared_label_wins(self):
        self.assertEqual(
            formgen.label_for("input", {"label": "Scan / Landmark Folder"}),
            "Scan / Landmark Folder",
        )

    def test_the_fallback_prettifies_the_argument_name(self):
        self.assertEqual(formgen.label_for("output_suffix", {}), "Output suffix")
        self.assertEqual(formgen.label_for("input", {"label": None}), "Input")
        self.assertEqual(formgen.label_for("input", {"label": "   "}), "Input")

    def test_the_vocabulary_is_spelled_here_because_the_server_may_not_know_it(self):
        """The server hands over "Cbct landmarks" and cannot do better.

        Knowing that CBCT is a word would make it know its tools, which is the
        one thing that server is built not to know -- an executable claim,
        checked on every build. This side IS the dental extension, so the table
        lives here and the word comes out whole.
        """
        self.assertEqual(formgen.label_for("cbct_landmarks", {}), "CBCT landmarks")
        self.assertEqual(formgen.label_for("ios_networks", {}), "IOS networks")
        self.assertEqual(formgen.label_for("prediction_ID", {}), "Prediction ID")

    def test_a_declared_label_gets_the_same_courtesy(self):
        """A tool that typed "Cbct regions" by hand reads the same as one that
        declared nothing: one rule, applied to whatever is displayed."""
        self.assertEqual(
            formgen.label_for("cbct_regions", {"label": "Cbct regions"}),
            "CBCT regions",
        )

    def test_a_phrase_no_rule_could_invent_survives_untouched(self):
        """The case that matters most: no token matches, nothing is rewritten.

        No naming rule can recover "Scan / Landmark Folder" from `input`, which
        is why a tool declares it — and why this pass must never damage one.
        """
        for phrase in ("Scan / Landmark Folder", "Reference (gold) file",
                       "Number of workers", "T1"):
            self.assertEqual(formgen.label_for("whatever", {"label": phrase}), phrase)

    def test_the_pass_matches_whole_words_only(self):
        """"identifier" is not "id", and "action" is not "ct"."""
        self.assertEqual(formgen.label_for("x", {"label": "Patient identifier"}),
                         "Patient identifier")
        self.assertEqual(formgen.label_for("x", {"label": "No action taken"}),
                         "No action taken")

    def test_one_rule_for_generated_fields_and_file_inputs(self):
        # Regression: build() used the raw schema name while base_widget
        # prettified it, so one panel showed "Reference" above "cbct_landmarks".
        schema = {
            "plain": {"type": "str", "types": ["str"], "required": True},
            "named": {"type": "str", "types": ["str"], "label": "A Real Name"},
        }
        layout = qt.QFormLayout()
        formgen.build(schema, layout)
        self.assertEqual([label.text for label, _f in layout.rows], ["Plain *", "A Real Name"])


class SectionTest(unittest.TestCase):
    def test_an_argument_declaring_nothing_lands_in_the_default_section(self):
        self.assertEqual(formgen.section_of({}), formgen.DEFAULT_SECTION)
        self.assertEqual(formgen.section_of({"section": None}), formgen.DEFAULT_SECTION)

    def test_sections_are_ordered_by_first_mention(self):
        schema = {
            "a": {"type": "str", "section": "Setup"},
            "b": {"type": "str"},
            "c": {"type": "str", "section": "Setup"},
            "d": {"type": "str", "section": "Tuning"},
        }
        self.assertEqual(
            formgen.sections_of(schema), ["Setup", formgen.DEFAULT_SECTION, "Tuning"]
        )

    def test_extra_sections_are_appended_unless_already_claimed(self):
        schema = {"a": {"type": "str", "section": "Outputs"}}
        self.assertEqual(formgen.sections_of(schema, ["Outputs", "Extra"]), ["Outputs", "Extra"])

    def test_every_example_tool_argument_keeps_the_single_default_section(self):
        # The compatibility guarantee: a tool declaring no section must render
        # exactly as it did before sections existed.
        self.assertEqual(
            formgen.sections_of(EXAMPLE_TOOL_SCHEMA["arguments"]), [formgen.DEFAULT_SECTION]
        )

    def test_build_routes_each_argument_to_its_section(self):
        schema = {
            "here": {"type": "str", "types": ["str"], "section": "Setup"},
            "there": {"type": "str", "types": ["str"], "section": "Tuning"},
            "nowhere": {"type": "str", "types": ["str"]},
        }
        fallback = qt.QFormLayout()
        sections = {"Setup": qt.QFormLayout(), "Tuning": qt.QFormLayout()}
        rows = {}
        formgen.build(schema, fallback, sections=sections, rows=rows)

        self.assertEqual(len(sections["Setup"].rows), 1)
        self.assertEqual(len(sections["Tuning"].rows), 1)
        # No layout was created for the default section, so it falls back.
        self.assertEqual(len(fallback.rows), 1)
        self.assertEqual(set(rows), {"here", "there", "nowhere"})
        # A row is the pair a caller has to hide together.
        self.assertEqual(len(rows["here"]), 2)


class VisibilityTest(unittest.TestCase):
    def test_an_argument_declaring_nothing_is_always_visible(self):
        self.assertTrue(formgen.is_visible({}, {}))

    def test_every_condition_must_match(self):
        spec = {"visible_when": {"modality": "CBCT", "automation": "Fully-Automated"}}
        self.assertTrue(
            formgen.is_visible(spec, {"modality": "CBCT", "automation": "Fully-Automated"})
        )
        self.assertFalse(
            formgen.is_visible(spec, {"modality": "CBCT", "automation": "Semi-Automated"})
        )
        self.assertFalse(
            formgen.is_visible(spec, {"modality": "IOS", "automation": "Fully-Automated"})
        )

    def test_a_list_of_values_means_any_of_them(self):
        spec = {"visible_when": {"mode": ["a", "b"]}}
        self.assertTrue(formgen.is_visible(spec, {"mode": "b"}))
        self.assertFalse(formgen.is_visible(spec, {"mode": "c"}))

    def test_an_unevaluable_condition_hides_the_field(self):
        # Only reachable when the schema could not be fetched. A field whose
        # precondition is unknown is one the user cannot fill in meaningfully,
        # and hiding it is the answer that cannot produce a wrong request.
        self.assertFalse(formgen.is_visible({"visible_when": {"modality": "CBCT"}}, {}))

    def test_controlling_arguments_are_collected_across_the_schema(self):
        schema = {
            "modality": {"type": "choice"},
            "automation": {"type": "choice"},
            "a": {"type": "str", "visible_when": {"modality": "CBCT"}},
            "b": {"type": "str", "visible_when": {"modality": "IOS", "automation": "Semi"}},
            "c": {"type": "str"},
        }
        self.assertEqual(
            formgen.controlling_arguments(schema), {"modality", "automation"}
        )

    def test_a_hidden_required_field_does_not_block_apply(self):
        schema = {"needed": {"type": "str", "types": ["str"], "required": True,
                             "visible_when": {"mode": "on"}}}
        widgets = formgen.build(schema, qt.QFormLayout())
        self.assertFalse(formgen.all_required_filled(widgets, schema))
        self.assertTrue(formgen.all_required_filled(widgets, schema, hidden={"needed"}))


# ---------------------------------------------------------------------------
# Input sources: the tool's server-hosted test files and the scene's volumes
# ---------------------------------------------------------------------------

_VOLUME_SPEC = {
    "type": "volume_or_zip_file", "types": ["volume_or_zip_file", "folder"],
    "required": True, "description": "", "server_selectable": "testfile",
    "choices": None, "initial": None,
}

# Same shape, but the entries are WEIGHTS: never downloaded, named in the run.
_MODEL_SPEC = dict(_VOLUME_SPEC, server_selectable="model")


class AcceptsVolumeTest(unittest.TestCase):
    """Which file arguments may be satisfied by a volume open in the scene:
    read off the schema, never off a module override."""

    def test_a_volume_type_name_qualifies(self):
        self.assertTrue(formgen.accepts_volume({"types": ["volume_or_zip_file"]}))
        self.assertTrue(formgen.accepts_volume({"types": ["nifti_file"]}))

    def test_a_volume_extension_qualifies(self):
        spec = {"types": ["scan_file"], "extensions": {"scan_file": [".nrrd"]}}
        self.assertTrue(formgen.accepts_volume(spec))

    def test_a_csv_input_never_offers_scene_volumes(self):
        self.assertFalse(formgen.accepts_volume(EXAMPLE_TOOL_SCHEMA["arguments"]["input"]))

    def test_a_surface_only_input_qualifies_now_that_meshes_count(self):
        """It did not, and that was the narrower question: "can a scalar volume
        satisfy this". What the row actually asks is whether the SCENE can --
        and a mesh argument is satisfied by a model open in Slicer exactly as a
        scan argument is by a volume. `scene_kinds_for` says which kinds."""
        spec = {"types": ["surface_file"], "extensions": {"surface_file": [".vtk", ".stl"]}}
        self.assertTrue(formgen.accepts_volume(spec))
        self.assertEqual(formgen.scene_kinds_for(spec), ("model",))


class HumanSizeTest(unittest.TestCase):
    """A test file's size is shown to a clinician deciding whether to click it,
    so it has to read as a size."""

    def test_bytes_megabytes_and_gigabytes(self):
        self.assertEqual(formgen.human_size(2969), "2.9 KB")
        self.assertEqual(formgen.human_size(94 * 1024 * 1024), "94 MB")
        self.assertEqual(formgen.human_size(648 * 1024 * 1024), "648 MB")
        self.assertEqual(formgen.human_size(7 * 1024 * 1024 + 419430), "7.4 MB")
        self.assertEqual(formgen.human_size(3 * 1024 ** 3), "3.0 GB")
        self.assertEqual(formgen.human_size(512), "512 B")

    def test_an_unknown_size_renders_nothing_rather_than_zero(self):
        """A backend that cannot size a tree cheaply sends null, and "0 B"
        would be a claim where the server made none."""
        self.assertEqual(formgen.human_size(None), "")
        self.assertEqual(formgen.human_size(0), "")
        self.assertEqual(formgen.human_size("355640000"), "")


class HostedEntryLabelTest(unittest.TestCase):
    """What one server-hosted test file reads as in the picker."""

    def test_kind_and_size_are_both_shown(self):
        self.assertEqual(
            formgen.hosted_entry_label(
                {"name": "CBCT_FullyAuto", "kind": "folder", "size": 355640000}
            ),
            "CBCT_FullyAuto  (folder, 339 MB)",
        )

    def test_a_known_kind_alone_still_reads(self):
        self.assertEqual(
            formgen.hosted_entry_label({"name": "ROI_box.mrk.json", "kind": "file", "size": None}),
            "ROI_box.mrk.json  (file)",
        )

    def test_an_entry_that_says_nothing_extra_is_just_its_name(self):
        self.assertEqual(
            formgen.hosted_entry_label({"name": "scan.nii.gz", "kind": None, "size": None}),
            "scan.nii.gz",
        )


class InputSourcesTest(unittest.TestCase):
    """The one-line input row: [sources dropdown][local picker], with the
    tool's server-hosted test files above the scene's open volumes.

    There is no "Upload my own file..." entry any more: it was a MODE dressed
    as a file, and what says whether the argument has been given anything is
    the picker's own `currentPath`.
    """

    def setUp(self):
        self.widget = formgen.file_widget(_VOLUME_SPEC, "file_or_folder")
        self.widget.setChoices([
            {"name": "sample_scan.nii.gz", "kind": "file", "size": 94 * 1024 * 1024},
        ])
        self.widget.setVolumeChoices(["CBCT_patient1", "CBCT_patient2"])
        # Real files on disk: the caption reports a size, and a stub size would
        # test the stub.
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, True)

    def test_the_row_is_three_lines_source_control_and_answer(self):
        """The segmented control on top, the chosen source's own control under
        it, and what came of it underneath.

        The control stayed on ONE line -- that was the point of the row and
        still is. What changed is that only one control is on it: the four
        sources used to sit side by side and be kept exclusive by a rule
        nobody could see.
        """
        column = self.widget.container.layout
        self.assertIsInstance(column, qt.QVBoxLayout)
        self.assertIs(column.widgets[0], self.widget.sourceBar)
        controls = column.widgets[1]
        self.assertIsInstance(controls.layout, qt.QHBoxLayout)
        self.assertIn(self.widget.combo, controls.layout.widgets)
        # The last line lives on the PICKER, so a row without extra sources
        # has one too -- a `.csv` argument used to show nothing at all.
        self.assertIs(self.widget.caption, self.widget.local.caption)

    def test_the_caption_says_so_when_nothing_is_chosen(self):
        """It used to hide, on the reasoning that an empty line is noise. That
        held while the row had a path field: its placeholder said "nothing here
        yet". The field is gone, and a row of two buttons with no caption reads
        as a row whose state nobody thought to show -- with Apply greyed and
        nothing saying which input is the one still missing."""
        self.assertTrue(self.widget.caption.isVisible())
        self.assertEqual(self.widget.caption.text, formgen.NOTHING_CHOSEN)

    def test_a_downloaded_test_file_says_its_name_and_its_kind(self):
        """The complaint this exists for: the row showed
        `:TestFiles2026-09-09_09+26+53.117/sample_scan.nii.gz` and the dropdown
        had gone back to its prompt, so nothing on screen said which file was
        loaded, still less that it was a NIfTI volume."""
        path = os.path.join(self.temp, "sample_scan.nii.gz")
        with open(path, "wb") as handle:
            handle.write(b"x" * 2048)

        formgen.set_local_path(self.widget, path)

        caption = self.widget.caption.text
        self.assertIn("sample_scan.nii.gz", caption)
        self.assertIn("NIfTI volume", caption)
        self.assertIn("2.0 KB", caption)
        self.assertTrue(self.widget.caption.isVisible())

    def test_it_says_a_fetched_file_is_not_the_user_s_own_copy(self):
        """A download lands in a session folder swept on exit. A user who takes
        it for their own copy will look for it next week and not find it."""
        path = os.path.join(self.temp, "sample_scan.nii.gz")
        open(path, "wb").close()

        formgen.set_local_path(self.widget, path)

        self.assertIn("Test File:", self.widget.caption.text)

    def test_a_file_the_user_chose_themselves_claims_nothing_of_the_sort(self):
        path = os.path.join(self.temp, "my_own_patient.nii.gz")
        open(path, "wb").close()

        formgen.set_local_path(self.widget, path)

        self.assertIn("my_own_patient.nii.gz", self.widget.caption.text)
        self.assertNotIn("test data", self.widget.caption.text)

    def test_a_surface_reads_as_a_surface(self):
        path = os.path.join(self.temp, "upper_arch_segmented.vtk")
        open(path, "wb").close()

        formgen.set_local_path(self.widget, path)

        self.assertIn("VTK surface", self.widget.caption.text)

    def test_a_name_that_says_nothing_is_not_given_a_kind(self):
        """Better a bare name than a confident guess at what it holds."""
        path = os.path.join(self.temp, "measurements.weird")
        open(path, "wb").close()

        formgen.set_local_path(self.widget, path)

        self.assertIn("measurements.weird", self.widget.caption.text)
        self.assertNotIn(" - ", self.widget.caption.text.replace(
            "measurements.weird", ""))

    def test_the_full_path_stays_reachable_as_a_tooltip(self):
        """The caption names the file; the tooltip says where it sits. Neither
        costs a line the panel does not have.

        It sits on the picker's CONTAINER now, the field it used to sit on
        having gone: the pointer is over the button group, and a tooltip on a
        widget nobody hovers is a path nobody can read.
        """
        path = os.path.join(self.temp, "sample_scan.nii.gz")
        open(path, "wb").close()

        formgen.set_local_path(self.widget, path)

        self.assertEqual(self.widget.local.container.toolTip(), path)

    def test_an_imported_scan_says_it_is_one(self):
        """Nothing is on disk for it, so `describe_file` has nothing to read --
        and "no file chosen" would be a lie about a satisfied argument."""
        self.widget.sceneCombo.setCurrentIndex(1)

        # "Scan", which is true of a CBCT volume and of an intraoral surface
        # alike -- ALI takes either through the one argument.
        self.assertIn("Scan:", self.widget.caption.text)
        self.assertIn("CBCT_patient1", self.widget.caption.text)

    def test_the_caption_empties_when_the_input_does(self):
        path = os.path.join(self.temp, "sample_scan.nii.gz")
        open(path, "wb").close()
        formgen.set_local_path(self.widget, path)

        formgen.set_local_path(self.widget, "")

        self.assertEqual(self.widget.caption.text, formgen.NOTHING_CHOSEN)

    def test_each_source_has_a_list_of_its_own(self):
        """They used to share one. Two unrelated questions -- "fetch the tool's
        sample data" and "use what is already open in Slicer" -- read as a
        single jumbled menu, and one list cannot be hidden for a row that
        takes no scene node while the other stays."""
        combo, scene = self.widget.combo, self.widget.sceneCombo
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count)],
            [formgen.ServerFileInput.PROMPT_HOSTED,
             "sample_scan.nii.gz  (file, 94 MB)"],
        )
        self.assertEqual(
            [scene.itemText(i) for i in range(scene.count)],
            [formgen.scene_prompt_for("Scan"), "CBCT_patient1", "CBCT_patient2"],
        )

    def test_the_prompt_names_what_the_list_holds(self):
        """It used to be the path field's own placeholder, word for word.

        Photographed, the panel showed "Select a file or a folder" twice side by
        side, and the dropdown read as a duplicate of the field beside it rather
        than as the one place a tool's test data is reached from. Nobody opens a
        control that appears to repeat its neighbour.

        The field is gone, so the duplication cannot happen on screen any more,
        but the words are still there (`PATH_PLACEHOLDER`, the fallback for an
        empty list) and a list that HAS test data in it must not fall back to
        them: "Select a file or a folder" says nothing about what is inside.
        """
        self.assertEqual(self.widget.combo.itemText(0),
                         formgen.ServerFileInput.PROMPT_HOSTED)
        self.assertNotEqual(self.widget.combo.itemText(0),
                            formgen.PATH_PLACEHOLDER)

    def test_the_prompt_offers_only_what_is_there(self):
        """Naming a source the list does not have would be worse than saying
        nothing: a user opens it, finds no test data, and stops trusting it."""
        self.widget.setSceneSupported(True)
        self.widget.setVolumeChoices([])
        self.assertEqual(self.widget.combo.itemText(0),
                         formgen.ServerFileInput.PROMPT_HOSTED)

        # Offered but GREY: without the segment nobody learns the row can be
        # filled that way at all -- which is exactly how the feature looked
        # missing on every module but the one whose scene happened to match.
        self.widget._chooseSource(formgen.ServerFileInput.SOURCE_SCENE)
        self.assertTrue(self.widget.sceneCombo.isVisible())
        self.assertFalse(self.widget.sceneCombo._enabled)

        self.widget.setChoices([])
        self.widget.setVolumeChoices(["CBCT_patient1"])
        self.assertEqual(self.widget.combo.itemText(0),
                         formgen.ServerFileInput.CHOOSE_OPTION)
        self.assertTrue(self.widget.sceneCombo.isVisible())
        self.assertTrue(self.widget.sceneCombo._enabled)

    def test_an_empty_list_keeps_the_neutral_words(self):
        self.widget.setChoices([])
        self.widget.setVolumeChoices([])
        self.assertEqual(self.widget.combo.itemText(0),
                         formgen.ServerFileInput.CHOOSE_OPTION)

    def test_a_model_row_says_model_because_nothing_is_fetched(self):
        """Those entries are not test data: they are the value that travels,
        and the weights never leave the server."""
        widget = formgen.file_widget(_MODEL_SPEC, "single_file")
        widget.setChoices([{"name": "AMASSS_Models", "kind": "folder", "size": None}])

        self.assertEqual(widget.combo.itemText(0),
                         formgen.ServerFileInput.PROMPT_MODEL)

    def test_the_prompt_is_also_the_collapsed_box_tooltip(self):
        """The box stays narrow on purpose, so the prompt is the first thing
        elided -- the tooltip is where it survives."""
        self.assertEqual(self.widget.combo.toolTip(),
                         formgen.ServerFileInput.PROMPT_HOSTED)
        self.assertEqual(self.widget.sceneCombo.toolTip(),
                         formgen.scene_prompt_for("Scan"))

    def test_the_default_state_names_nothing(self):
        self.assertEqual(self.widget.hosted_name(), "")
        self.assertEqual(self.widget.volume_name(), "")
        self.assertEqual(self.widget.currentPath, "")

    def test_choosing_a_test_file_hands_its_name_to_the_callback(self):
        """formgen never talks HTTP: it reports the pick and base_widget
        downloads it."""
        picked = []
        self.widget.setHostedCallback(picked.append)

        self.widget.combo.setCurrentIndex(1)

        self.assertEqual(picked, ["sample_scan.nii.gz"])

    def test_a_test_file_pick_clears_a_previously_chosen_path(self):
        """Cleared when the pick starts, not when the download lands: a run
        launched mid-download must not send the file it replaced."""
        self.widget.local.setCurrentPath("/data/my_own_scan.nii.gz")

        self.widget.combo.setCurrentIndex(1)

        self.assertEqual(self.widget.currentPath, "")

    def test_the_downloaded_path_becomes_an_ordinary_local_selection(self):
        self.widget.setHostedCallback(
            lambda name: formgen.set_local_path(self.widget, "/tmp/session/" + name)
        )

        self.widget.combo.setCurrentIndex(1)

        self.assertEqual(self.widget.currentPath, "/tmp/session/sample_scan.nii.gz")
        # The dropdown is an action list, not a mode: once the file is on disk
        # the local path is the whole of the state.
        self.assertEqual(self.widget.combo.currentIndex, 0)
        self.assertEqual(self.widget.hosted_name(), "")

    def test_choosing_a_volume_is_not_a_test_file_selection(self):
        self.widget.sceneCombo.setCurrentIndex(1)

        self.assertEqual(self.widget.volume_name(), "CBCT_patient1")
        self.assertEqual(self.widget.hosted_name(), "")
        # Nothing to read off disk either: the node is exported at upload time.
        self.assertEqual(self.widget.currentPath, "")

    def test_choosing_a_volume_clears_the_local_path(self):
        self.widget.local.setCurrentPath("/data/scan.nii.gz")

        self.widget.sceneCombo.setCurrentIndex(2)

        self.assertEqual(self.widget.local.currentPath, "")

    def test_a_local_path_resets_the_dropdown(self):
        """The other half of the mutual exclusion, and nothing is typed to get
        it any more: a browse dialog, or `set_local_path` once a download has
        landed, writes the path and the chosen entry has to let go. A
        precedence rule the user cannot see is how you end up sending a file
        you thought you had replaced."""
        self.widget.sceneCombo.setCurrentIndex(2)

        self.widget.local.setCurrentPath("/data/scan.nii.gz")

        self.assertEqual(self.widget.volume_name(), "")
        self.assertEqual(self.widget.sceneCombo.currentIndex, 0)
        self.assertEqual(self.widget.currentPath, "/data/scan.nii.gz")

    def test_a_chosen_volume_survives_a_test_file_list_refresh(self):
        self.widget.sceneCombo.setCurrentIndex(1)

        self.widget.setChoices([
            {"name": "sample_scan.nii.gz", "kind": "file", "size": 94 * 1024 * 1024},
            {"name": "cohort_10_patients.zip", "kind": "file", "size": None},
        ])

        self.assertEqual(self.widget.volume_name(), "CBCT_patient1")

    def test_a_gone_volume_falls_back_to_the_prompt(self):
        self.widget.sceneCombo.setCurrentIndex(1)

        self.widget.setVolumeChoices([])

        self.assertEqual(self.widget.volume_name(), "")
        # And the list goes with it: a dropdown holding only its own prompt is
        # a control that can only disappoint.
        self.assertFalse(self.widget.sceneCombo.isVisible())

    def test_a_refresh_starts_no_download_of_its_own(self):
        """clear()+addItems reselects index 0 and would otherwise fire the
        selection handler for a choice nobody made."""
        picked = []
        self.widget.setHostedCallback(picked.append)

        self.widget.setChoices([{"name": "sample_scan.nii.gz", "kind": "file", "size": 1}])
        self.widget.setVolumeChoices(["CBCT_patient1"])

        self.assertEqual(picked, [])

    def test_a_test_file_named_like_a_volume_entry_is_not_misread(self):
        # Selection kind is decided by index, so even a hosted file named
        # exactly like a scene entry stays a hosted selection.
        tricky = "CBCT_patient1"
        self.widget.setChoices([{"name": tricky, "kind": None, "size": None}])

        self.widget.combo.setCurrentIndex(1)

        self.assertEqual(self.widget.hosted_name(), tricky)
        self.assertEqual(self.widget.volume_name(), "")

    def test_bare_names_are_accepted_as_entries(self):
        """A caller holding only names -- an older server publishes no
        `entries` at all -- still gets a working list."""
        self.widget.setChoices(["a.nii.gz", "b.nii.gz"])

        self.assertEqual(self.widget.combo.itemText(1), "a.nii.gz")
        self.assertEqual(
            self.widget.hosted_entries(),
            [{"name": "a.nii.gz", "kind": None, "size": None},
             {"name": "b.nii.gz", "kind": None, "size": None}],
        )


class SetLocalPathTest(unittest.TestCase):
    """Writing a downloaded path into an input row, whichever shape it has."""

    def test_set_local_path_reaches_the_path_field_of_either_shape(self):
        wrapped = formgen.file_widget(_VOLUME_SPEC, "file_or_folder")
        bare = formgen.file_widget(EXAMPLE_TOOL_SCHEMA["arguments"]["input"], "file_or_folder")

        formgen.set_local_path(wrapped, "/downloads/scan.nii.gz")
        formgen.set_local_path(bare, "/downloads/cohort")

        self.assertEqual(wrapped.currentPath, "/downloads/scan.nii.gz")
        self.assertEqual(bare.currentPath, "/downloads/cohort")

    def test_an_argument_with_no_extra_source_is_a_bare_picker(self):
        """No dropdown at all where there is nothing to put in it: example_tool's
        csv input is neither server_selectable nor volume-ish."""
        bare = formgen.file_widget(EXAMPLE_TOOL_SCHEMA["arguments"]["input"], "file_or_folder")
        self.assertIsInstance(bare, formgen.FileOrFolderInput)
        self.assertFalse(hasattr(bare, "combo"))


# ---------------------------------------------------------------------------
# Sliders (ArgSpec.ui = "slider" on int/float) and bounds on spin boxes
# ---------------------------------------------------------------------------

def _numeric(arg_type, **hints):
    spec = {
        "type": arg_type, "types": [arg_type], "required": False,
        "description": "", "server_selectable": None, "choices": None,
        "initial": None,
    }
    spec.update(hints)
    return spec


class SliderWidgetTest(unittest.TestCase):
    """`ui: "slider"` on a bounded int/float renders the combined
    slider+spinbox; everything else stays a spin box."""

    def _one(self, spec):
        return formgen.build({"k": spec}, qt.QFormLayout())["k"]

    def test_bounded_float_with_the_hint_becomes_a_slider(self):
        widget = self._one(_numeric("float", ui="slider", min=-180, max=180, step=0.5, initial=10))

        self.assertIsInstance(widget, ctk.ctkSliderWidget)
        self.assertEqual(widget.minimum, -180.0)
        self.assertEqual(widget.maximum, 180.0)
        self.assertEqual(widget.singleStep, 0.5)
        self.assertEqual(widget.value, 10.0)

    def test_an_int_slider_reads_back_as_an_int(self):
        widget = self._one(_numeric("int", ui="slider", min=0, max=95, step=5, initial=5))

        self.assertEqual(widget.decimals, 0)
        value = formgen.collect({"k": widget})["k"]
        self.assertEqual(value, 5)
        self.assertIsInstance(value, int)

    def test_float_decimals_follow_the_step(self):
        widget = self._one(_numeric("float", ui="slider", min=0, max=1, step=0.05))
        self.assertEqual(widget.decimals, 2)

    def test_declared_decimals_win_over_the_step(self):
        widget = self._one(_numeric("float", ui="slider", min=0, max=1, step=0.05, decimals=4))
        self.assertEqual(widget.decimals, 4)

    def test_a_float_spin_box_without_decimals_shows_two_places(self):
        # A flat 6 drew a temperature of 0 as `0.000000`.
        self.assertEqual(self._one(_numeric("float", initial=0.0)).decimals, 2)

    def test_a_float_spin_box_keeps_the_places_its_initial_value_needs(self):
        self.assertEqual(self._one(_numeric("float", initial=0.001)).decimals, 3)

    def test_a_slider_without_bounds_falls_back_to_a_spin_box(self):
        # An unbounded slider has no geometry; the panel must render anyway.
        self.assertIsInstance(self._one(_numeric("float", ui="slider")), qt.QDoubleSpinBox)
        self.assertIsInstance(self._one(_numeric("int", ui="slider", min=0)), qt.QSpinBox)

    def test_bounds_without_the_hint_constrain_the_spin_box(self):
        # min/max alone must not switch the widget kind: a bound added
        # server-side for validation cannot silently produce a slider.
        widget = self._one(_numeric("int", min=1, max=10, step=2))

        self.assertIsInstance(widget, qt.QSpinBox)
        self.assertEqual((widget.minimum, widget.maximum), (1, 10))
        self.assertEqual(widget.singleStep, 2)

    def test_an_unknown_scalar_ui_falls_back_to_a_spin_box(self):
        self.assertIsInstance(self._one(_numeric("float", ui="dial")), qt.QDoubleSpinBox)

    def test_changing_the_slider_notifies(self):
        widget = self._one(_numeric("float", ui="slider", min=0, max=10))
        calls = []
        formgen.connect_changed(widget, lambda *_a: calls.append(1))

        widget.value = 3.5

        self.assertEqual(len(calls), 1)

    def test_slider_value_is_sent_in_clear(self):
        widget = self._one(_numeric("float", ui="slider", min=0, max=10, initial=2.5))
        data = ToolServerClient._stringify(formgen.collect({"k": widget}))
        self.assertEqual(data["k"], "2.5")

    def test_a_required_slider_at_zero_still_counts_as_filled(self):
        schema = {"k": _numeric("float", ui="slider", min=-10, max=10, required=True)}
        widgets = formgen.build(schema, qt.QFormLayout())
        self.assertTrue(formgen.all_required_filled(widgets, schema))


# ---------------------------------------------------------------------------
# vec2 (two numbers set together, ui = "joystick" for the 2D pad)
# ---------------------------------------------------------------------------

def _vec2(**hints):
    spec = {
        "type": "vec2", "types": ["vec2"], "required": False,
        "description": "", "server_selectable": None, "choices": None,
        "initial": None,
    }
    spec.update(hints)
    return spec


class JoystickWidgetTest(unittest.TestCase):
    def _one(self, spec):
        return formgen.build({"k": spec}, qt.QFormLayout())["k"]

    def test_vec2_with_the_hint_gets_a_pad_with_the_declared_ranges(self):
        widget = self._one(_vec2(ui="joystick", x_range=[-15, 15], y_range=[-5, 5]))

        self.assertIsInstance(widget, formgen.JoystickInput)
        self.assertIsNotNone(widget.pad)
        self.assertEqual((widget.pad.x_start, widget.pad.x_end), (-15.0, 15.0))
        self.assertEqual((widget.pad.y_start, widget.pad.y_end), (-5.0, 5.0))

    def test_vec2_without_the_hint_is_two_plain_spin_boxes(self):
        widget = self._one(_vec2(x_range=[0, 1], y_range=[0, 1]))
        self.assertIsInstance(widget, formgen.JoystickInput)
        self.assertIsNone(widget.pad)

    def test_an_unknown_vec2_ui_falls_back_to_the_boxes_alone(self):
        widget = self._one(_vec2(ui="trackball", x_range=[0, 1], y_range=[0, 1]))
        self.assertIsNone(widget.pad)

    def test_the_declared_initial_reaches_boxes_and_pad(self):
        widget = self._one(_vec2(ui="joystick", x_range=[-15, 15], y_range=[-5, 5], initial=[3, -2]))

        self.assertEqual(widget.value(), [3.0, -2.0])
        self.assertEqual((widget.pad.value_x, widget.pad.value_y), (3.0, -2.0))

    def test_no_initial_opens_at_the_centre_of_both_axes(self):
        widget = self._one(_vec2(ui="joystick", x_range=[0, 10], y_range=[-5, 5]))
        self.assertEqual(widget.value(), [5.0, 0.0])

    def test_collect_returns_the_pair_and_it_travels_as_json(self):
        widget = self._one(_vec2(ui="joystick", x_range=[-15, 15], y_range=[-5, 5], initial=[3, -2]))

        collected = formgen.collect({"k": widget})
        self.assertEqual(collected["k"], [3.0, -2.0])
        self.assertEqual(json.loads(ToolServerClient._stringify(collected)["k"]), [3.0, -2.0])

    def test_editing_a_box_moves_the_pad(self):
        widget = self._one(_vec2(ui="joystick", x_range=[-15, 15], y_range=[-5, 5], initial=[0, 0]))

        widget.xBox.setValue(7.5)

        self.assertEqual(widget.pad.value_x, 7.5)

    def test_moving_the_pad_updates_the_boxes(self):
        widget = self._one(_vec2(ui="joystick", x_range=[-15, 15], y_range=[-5, 5], initial=[0, 0]))

        widget.pad.setValues(4.0, -1.0, notify=True)

        self.assertEqual(widget.value(), [4.0, -1.0])

    def test_any_input_path_notifies(self):
        widget = self._one(_vec2(ui="joystick", x_range=[-15, 15], y_range=[-5, 5], initial=[0, 0]))
        calls = []
        formgen.connect_changed(widget, lambda *_a: calls.append(1))

        widget.yBox.setValue(2.0)       # typing
        widget.pad.setValues(1.0, 2.0, notify=True)  # dragging

        self.assertGreaterEqual(len(calls), 2)

    def test_a_spring_back_pad_accumulates_displacements(self):
        widget = self._one(_vec2(ui="joystick", x_range=[-10, 10], y_range=[-10, 10],
                                 initial=[0, 0], spring_back=True))
        pad = widget.pad

        # One push: offset (2, 1) from the rest position, then release.
        pad.setValues(2.0, 1.0, notify=True)
        self.assertEqual(widget.value(), [2.0, 1.0])
        pad.mouseReleaseEvent(None)
        self.assertEqual((pad.value_x, pad.value_y), (0.0, 0.0))  # sprang home

        # A second push adds to the committed base instead of replacing it.
        pad.setValues(1.0, 1.0, notify=True)
        self.assertEqual(widget.value(), [3.0, 2.0])

    def test_the_description_is_hovered_rather_than_printed(self):
        """It was printed above the pad, as a small grey paragraph. Every
        argument's description is the row label's tooltip now; a pad that also
        printed its own would be the one field on the panel saying it twice."""
        schema = {"k": _vec2(ui="joystick", description="Move the landmark")}
        layout = qt.QFormLayout()
        formgen.build(schema, layout)
        label, field = layout.rows[0]

        printed = [w.text for w in field.layout.widgets if getattr(w, "text", None)]
        self.assertNotIn("Move the landmark", printed)
        self.assertEqual(label.toolTip(), "Move the landmark")

    def test_an_invalid_range_falls_back_to_the_unit_axis(self):
        widget = self._one(_vec2(ui="joystick", x_range=[3], y_range=[0, 1]))
        self.assertEqual((widget.pad.x_start, widget.pad.x_end), (0.0, 1.0))

    def test_a_required_vec2_always_counts_as_filled(self):
        schema = {"k": _vec2(ui="joystick", x_range=[0, 1], y_range=[0, 1], required=True)}
        widgets = formgen.build(schema, qt.QFormLayout())
        self.assertTrue(formgen.all_required_filled(widgets, schema))

class HostedEntryReadabilityTest(unittest.TestCase):
    """A hosted entry has to be readable in the popup, whatever the collapsed
    box's width. Elided, `CBCT_Or_FullyAuto_DCM (folder, 532 MB)` and
    `CBCT_Or_FullyAuto (folder, 267 MB)` are the same text -- and what gets cut
    is exactly what the entry exists to say."""

    def _input(self, hosted):
        widget = formgen.ServerFileInput(qt.QLineEdit())
        widget.setChoices(hosted)
        return widget

    def test_the_popup_is_widened_to_its_longest_entry(self):
        widget = self._input([
            {"name": "a.nii.gz", "kind": "file", "size": 1},
            {"name": "CBCT_Or_FullyAuto_DCM", "kind": "folder", "size": 532 * 1024 ** 2},
        ])

        longest = max(len(entry) for entry in widget._entries())
        self.assertGreaterEqual(widget.combo.view().minimumWidth, longest)

    def test_the_width_comes_from_the_view_not_from_font_metrics(self):
        """The trap this cost a release to find.

        `combo.fontMetrics` is a SLOT under PythonQt, not a property: reading
        through it without calling it raises `AttributeError`, which the bare
        `except` here swallowed -- so the list was never widened once, while the
        stub (which modelled it as a property) said it was. Measured in Slicer:
        `view.minimumWidth` 0, open popup 166 px, three AREG cohorts reading
        `CBCT_Or_Full...`.

        The view is asked instead, and it measures its own items.
        """
        widget = self._input([{"name": "CBCT_Or_FullyAuto_DCM", "kind": "folder",
                               "size": 532 * 1024 ** 2}])

        # Nothing may reach for the metrics through the slot object.
        with self.assertRaises(AttributeError):
            widget.combo.fontMetrics.horizontalAdvance("x")
        self.assertGreater(widget.combo.view().minimumWidth, 0)

    def test_a_view_that_cannot_measure_leaves_the_panel_standing(self):
        """Cosmetic, always: a list that is merely narrow must never be a
        traceback in a clinician's panel."""
        widget = self._input([{"name": "a.nii.gz", "kind": "file", "size": 1}])

        def refuse(_column):
            raise RuntimeError("no view today")

        widget.combo.view().sizeHintForColumn = refuse
        widget.setChoices([{"name": "b.nii.gz", "kind": "file", "size": 2}])

        self.assertEqual(widget.combo.itemText(1), "b.nii.gz  (file, 2 B)")

    def test_the_popup_is_bounded(self):
        """Wider than its box, never wider than a screen."""
        widget = self._input([{"name": "x" * 4000, "kind": "file", "size": 1}])

        self.assertLessEqual(widget.combo.view().minimumWidth,
                             formgen._POPUP_MAX_WIDTH)

    def test_the_collapsed_box_stays_narrow(self):
        """The row still has to fit: only the LIST grows."""
        widget = self._input([{"name": "x" * 120, "kind": "file", "size": 1}])

        self.assertEqual(widget.combo.minimumContentsLength, 14)

    def test_widening_survives_a_toolkit_that_cannot_do_it(self):
        """Every failure here is cosmetic, so none of them may take the panel
        down with it."""
        widget = self._input([{"name": "a.nii.gz", "kind": "file", "size": 1}])

        def explode():
            raise RuntimeError("no view in this build")

        widget.combo.view = explode
        widget.setChoices([{"name": "b.nii.gz", "kind": "file", "size": 2}])

        self.assertIn("b.nii.gz", " ".join(widget._entries()))

    def test_an_entry_says_what_it_is_and_what_it_costs(self):
        widget = self._input([
            {"name": "CBCT_FullyAuto", "kind": "folder", "size": 339 * 1024 ** 2},
        ])

        entry = [e for e in widget._entries() if "CBCT_FullyAuto" in e][0]
        self.assertIn("folder", entry)
        self.assertIn("339", entry)



class ToolTipStyleTest(unittest.TestCase):
    """The hover bubble is Qt's own until something styles it, and Qt's own is
    an opaque pale yellow that belongs to no theme -- it reads as a system
    warning sitting on top of the panel rather than as part of it."""

    def test_the_bubble_is_styled_in_both_themes(self):
        for theme in (design._LIGHT, design._DARK):
            sheet = design._base_stylesheet(theme)
            self.assertIn("QToolTip", sheet)

    def test_every_colour_it_uses_comes_from_the_token_table(self):
        """The trap this guards: a hard-coded hex survives light mode and
        disappears in dark, which is the theme nobody tests in. Each colour the
        rule resolves to has to be a value the token table actually holds."""
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            rule = re.search(r"QToolTip \{(.*?)\n    \}",
                             design._base_stylesheet(theme), re.S).group(1)
            colours = re.findall(r"#[0-9a-fA-F]{3,8}", rule)
            self.assertTrue(colours, name)
            for colour in colours:
                self.assertIn(colour, theme.values(), "{} in {}".format(colour, name))

    def test_the_two_themes_do_not_resolve_to_the_same_bubble(self):
        """A rule that renders identically in both is one that took its colours
        from somewhere other than the palette."""
        light = re.search(r"QToolTip \{(.*?)\n    \}",
                          design._base_stylesheet(design._LIGHT), re.S).group(1)
        dark = re.search(r"QToolTip \{(.*?)\n    \}",
                         design._base_stylesheet(design._DARK), re.S).group(1)
        self.assertNotEqual(light, dark)



class MultiChoiceTooltipTest(unittest.TestCase):
    """A group's description never lands on the group itself."""

    CHOICES = {"Ba": False, "S": False, "N": False}
    NOTE = ("Predict exactly these landmarks -- naming any of them REPLACES the "
            "region selection rather than narrowing it, which is what lets a "
            "caller ask for the seven points it needs.")

    def test_it_is_not_also_put_on_the_container(self):
        """Qt hands a container's tooltip to every child that has none, so this
        paragraph popped up under each of ALI's 236 chips -- printed and hovered
        at once, and the hovered copy is the one nobody asked for."""
        group = formgen.MultiChoiceGroup(self.CHOICES)
        group.setToolTip(self.NOTE)
        self.assertFalse(group.container._tooltip)

    def test_every_other_composite_still_takes_one(self):
        """Only the multichoice refuses it. A file picker's tooltip says which
        path it holds, and nothing else says that."""
        field = formgen.FileOrFolderInput()
        field.setToolTip("/data/patient/scan.nii.gz")
        self.assertEqual(field.container._tooltip, "/data/patient/scan.nii.gz")



class RowLabelTest(unittest.TestCase):
    """A long label wraps instead of widening its section's label column,
    which pushed every field of the section to the side."""

    def test_a_long_label_wraps_at_the_cap(self):
        label = design.fit_row_label(design.section_title("x" * 80))
        self.assertTrue(label.wordWrap)
        self.assertEqual(label.maximumWidth, design.ROW_LABEL_MAX_WIDTH)
        self.assertEqual(label.minimumWidth, design.ROW_LABEL_MAX_WIDTH)

    def test_a_label_that_fits_is_left_alone(self):
        """Given a measured minimum, "Mode *" wrapped as soon as the
        measurement fell a pixel short of what Qt drew."""
        label = design.fit_row_label(design.required_label("Mode"))
        self.assertFalse(getattr(label, "wordWrap", False))
        self.assertFalse(hasattr(label, "minimumWidth"))


class ToggleRowTest(unittest.TestCase):
    """A yes/no spans the row, box first: in the label column its words set
    the column's width and pushed every field of the section aside."""

    def test_a_bool_takes_the_whole_row(self):
        layout = qt.QFormLayout()
        formgen.build({
            "mode": {"type": "str"},
            "orient": {"type": "bool", "label": "Standardise the CBCT orientation"},
        }, layout)
        (text_label, _field), (no_label, row) = layout.rows
        self.assertIsNotNone(text_label)
        self.assertIsNone(no_label)
        placed = row.layout.widgets
        self.assertIsInstance(placed[0], qt.QCheckBox)
        self.assertEqual(placed[1].text, "Standardise the CBCT orientation")


class NoteTest(unittest.TestCase):
    """`note` -- one line above a multichoice's options, always visible.

    AREG's "Extra outputs" read as three bare words; what comes back whatever
    is ticked, and in which space the rest lands, was only in a tooltip.
    """

    CHOICES = {"T1 scan": False}
    NOTE = "Saved beside the result, in the same 3D space."

    def _notes(self, group):
        return [widget.text for widget in group._column.widgets
                if isinstance(widget, qt.QLabel)]

    def test_the_note_is_drawn_above_the_options(self):
        group = formgen.MultiChoiceGroup(self.CHOICES, note=self.NOTE)
        self.assertEqual(self._notes(group), [self.NOTE])
        self.assertIsInstance(group._column.widgets[0], qt.QLabel)

    def test_a_redraw_for_another_mode_keeps_it_once(self):
        group = formgen.MultiChoiceGroup(self.CHOICES, note=self.NOTE)
        group.rebuild({"CBCT scan": False, "Moved landmarks": False})
        self.assertEqual(self._notes(group), [self.NOTE])

    def test_without_one_nothing_is_added(self):
        group = formgen.MultiChoiceGroup(self.CHOICES)
        self.assertEqual(self._notes(group), [])

    def test_the_schema_key_reaches_the_group(self):
        widget = formgen._make_widget("outputs", {
            "type": "multichoice", "choices": self.CHOICES, "note": self.NOTE})
        self.assertEqual(self._notes(widget), [self.NOTE])


class OptionHelpTest(unittest.TestCase):
    """`option_help` -- one line per option, on the option itself.

    A catalogue of CODES needs it: `Ba` and `UR1MB` tell a clinician nothing,
    and the argument's own description covers all 236 of them at once.
    """

    CHOICES = {"Ba": False, "S": False, "N": False}
    HELP = {"Ba": "Basion -- most anterior point of the foramen magnum"}

    def _group(self, layout=None, help_texts=None, groups=None):
        return formgen.MultiChoiceGroup(
            self.CHOICES, layout=layout, groups=groups,
            option_help=self.HELP if help_texts is None else help_texts)

    def test_the_named_option_carries_its_line(self):
        group = self._group()
        self.assertEqual(group.boxes["Ba"].toolTip(), self.HELP["Ba"])

    def test_an_option_the_table_skips_carries_nothing(self):
        """A half-filled table is the normal state while the words are being
        written, and it must leave the rest exactly as it was."""
        group = self._group()
        self.assertFalse(group.boxes["S"].toolTip())

    def test_a_tool_declaring_none_is_unchanged(self):
        group = self._group(help_texts={})
        self.assertFalse(any(box.toolTip() for box in group.boxes.values()))

    def test_every_layout_explains_identically(self):
        """The same property as `test_every_layout_reads_back_identically`: a
        layout changes the arrangement, never what the panel says."""
        for layout in (None, "inline", "grid", "tabs"):
            group = self._group(layout=layout, groups={"Cranial base": list(self.CHOICES)})
            self.assertEqual(group.boxes["Ba"].toolTip(), self.HELP["Ba"], layout)

    def test_a_malformed_table_costs_nothing(self):
        """This is the seam between two repositories. A field that arrives as
        something other than a mapping must leave the panel standing, not take
        it down -- the same rule the server applies to a key it does not know."""
        for broken in ("not a mapping", ["Ba"], {"Ba": 17}):
            group = self._group(help_texts=broken)
            self.assertEqual(sorted(group.boxes), sorted(self.CHOICES))
            self.assertFalse(group.boxes["Ba"].toolTip(), repr(broken))

    def test_it_reaches_the_widget_through_the_schema(self):
        """Declared by the tool, not built here: the whole point is that a
        landmark gains its line with no client release."""
        spec = {"type": "multichoice",
                "choices": {"Ba": False, "S": False},
                "option_help": {"Ba": "Basion"}}
        widget = formgen._make_widget("landmarks", spec)
        self.assertEqual(widget.boxes["Ba"].toolTip(), "Basion")



class ChipsLayoutTest(unittest.TestCase):
    """`ui: "chips"` -- the tabbed grid without the tabs.

    For a handful of options: AMASSS's nine structures fit on two lines, and
    putting them behind a single tab would be a tab bar with nowhere to go.
    """

    STRUCTURES = ["MAND", "MAX", "CB", "CV", "UAW", "SKIN",
                  "CBMASK", "MANDMASK", "MAXMASK"]

    def _group(self, groups=None, help_texts=None):
        return formgen.MultiChoiceGroup(
            {option: False for option in self.STRUCTURES},
            layout="chips", groups=groups, option_help=help_texts)

    def _grid(self, group, index=0):
        grids = [w.layout for w in group.container.layout.widgets
                 if isinstance(getattr(w, "layout", None), qt.QGridLayout)]
        return grids[index]

    def test_the_options_are_chips_not_check_boxes(self):
        """The point of the layout: the label IS the control, as in the tabbed
        catalogue -- not a column of boxes with their captions beside them."""
        group = self._group()
        self.assertTrue(all(isinstance(box, qt.QPushButton)
                            for box in group.boxes.values()))
        self.assertTrue(all(box.isCheckable() for box in group.boxes.values()))

    def test_they_wrap_onto_several_lines(self):
        """Not one long row: nine chips on one line is what `grid` does, and it
        is a chart layout, not this."""
        group = self._group()
        rows = {row for row, _column in self._grid(group).cells}
        self.assertGreater(len(rows), 1)

    def test_it_carries_no_group_button(self):
        """Nine chips are nine clicks. A control that takes all of them earns
        its place at a hundred options, not at nine."""
        group = self._group()
        texts = [getattr(w, "text", "") for w in group.container.layout.widgets]
        self.assertNotIn(formgen.SELECT_GROUP_LABEL, texts)
        self.assertNotIn(formgen.CLEAR_GROUP_LABEL, texts)

    def test_a_declared_group_becomes_a_heading_not_a_tab(self):
        """Which keeps a two-group argument readable without hiding half of it
        behind a click."""
        group = self._group(groups={"Structures": self.STRUCTURES[:6],
                                    "Masks": self.STRUCTURES[6:]})
        titles = [getattr(w, "text", "") for w in group.container.layout.widgets]
        self.assertIn("Structures", titles)
        self.assertIn("Masks", titles)
        self.assertFalse([w for w in group.container.layout.widgets
                          if isinstance(w, qt.QTabWidget)])

    def test_it_reads_back_exactly_as_every_other_layout(self):
        """The invariant every layout here is held to: a layout may be ugly, it
        is never wrong on the wire."""
        group = self._group()
        self.assertEqual(list(group.boxes), self.STRUCTURES)
        group.boxes["MAND"].setChecked(True)
        self.assertEqual(group.value()["MAND"], True)
        self.assertEqual(sorted(group.value()), sorted(self.STRUCTURES))

    def test_each_chip_still_carries_its_line(self):
        group = self._group(help_texts={"MAND": "Mandible"})
        self.assertEqual(group.boxes["MAND"].toolTip(), "Mandible")
        self.assertFalse(group.boxes["MAX"].toolTip())



class MinimumSelectionTest(unittest.TestCase):
    """`min_selected` -- a tool saying an empty multichoice is not an answer."""

    SCHEMA = {"merge": {"type": "multichoice", "required": False,
                        "choices": {"MERGED": True, "SEPARATE": False},
                        "min_selected": 1}}

    def _widgets(self):
        return formgen.build(self.SCHEMA, qt.QFormLayout())

    def test_an_empty_selection_blocks_apply(self):
        widgets = self._widgets()
        for box in widgets["merge"].boxes.values():
            box.setChecked(False)
        self.assertFalse(formgen.all_required_filled(widgets, self.SCHEMA))

    def test_one_tick_is_enough(self):
        widgets = self._widgets()
        for box in widgets["merge"].boxes.values():
            box.setChecked(False)
        widgets["merge"].boxes["SEPARATE"].setChecked(True)
        self.assertTrue(formgen.all_required_filled(widgets, self.SCHEMA))

    def test_a_multichoice_without_it_is_still_filled_when_empty(self):
        """The default everywhere else, and ALI relies on it: an empty
        `landmarks` is how a caller says "let the regions decide"."""
        schema = {"landmarks": {"type": "multichoice", "required": True,
                                "choices": {"Ba": False, "S": False}}}
        widgets = formgen.build(schema, qt.QFormLayout())
        self.assertTrue(formgen.all_required_filled(widgets, schema))

    def test_a_hidden_argument_cannot_dead_lock_apply(self):
        """Hidden rows are not sent, so the server applies the default. An
        unreachable widget must not be able to grey Apply out forever with
        nothing on screen to explain why."""
        widgets = self._widgets()
        for box in widgets["merge"].boxes.values():
            box.setChecked(False)
        self.assertTrue(
            formgen.all_required_filled(widgets, self.SCHEMA, hidden=("merge",)))


if __name__ == "__main__":
    unittest.main()


class ServerFileInputClearTest(unittest.TestCase):
    """`clear()` exists for a caller outside the dropdown that decides an
    argument is no longer satisfied — AutoMatrix's mirror check box, which fills
    Transforms when ticked and must not leave the matrix behind when unticked.

    What it must NOT do is read as the user having chosen something: the write is
    guarded the way every other programmatic write to this widget is, so a panel
    watching for changes is not told a hosted file was picked.
    """

    def _input(self):
        widget = formgen.ServerFileInput(qt.QLineEdit())
        widget.setChoices([{"name": "Mirror", "kind": "folder", "size": 468}])
        return widget

    def test_a_filled_input_is_emptied(self):
        widget = self._input()
        formgen._set_local_path(widget.local, "/tmp/downloads/Mirror")
        self.assertEqual(widget.currentPath, "/tmp/downloads/Mirror")

        widget.clear()

        self.assertEqual(widget.currentPath, "")
        self.assertEqual(widget.combo.currentIndex, 0, "the dropdown kept its pick")

    def test_clearing_does_not_fire_the_hosted_action(self):
        picked = []
        widget = self._input()
        widget.setHostedCallback(picked.append)
        formgen._set_local_path(widget.local, "/tmp/downloads/Mirror")

        widget.clear()

        self.assertEqual(picked, [], "clearing was mistaken for picking an entry")


class OverridesMayOnlyNameRealArgumentsTest(unittest.TestCase):
    """A FILE_INPUTS entry MODIFIES an argument the tool declares.

    Naming one it does not is always a module left behind by a rename, and the
    merge used to take it at its word: SurgMovPred's `input` became
    `measurements` when the tool was packaged, and the panel kept showing an
    "Input" picker -- styled optional, since there was no spec to say
    otherwise -- that uploaded to an argument the server would have refused.
    """

    SCHEMA = {"measurements": {"type": "path"}, "model": {"type": "str"}}

    def test_an_override_naming_no_argument_adds_no_row(self):
        modes = formgen.file_input_modes(self.SCHEMA, {"input": "folder_zip"})

        self.assertNotIn("input", modes)
        self.assertEqual(list(modes), ["measurements"])

    def test_an_override_on_a_real_argument_still_applies(self):
        modes = formgen.file_input_modes(self.SCHEMA, {"measurements": "folder_zip"})

        self.assertEqual(modes["measurements"], "folder_zip")

    def test_a_failed_schema_fetch_drops_no_override(self):
        """No schema means no form either, and warning once per override about
        a tool nobody could reach says nothing useful."""
        modes = formgen.file_input_modes({}, {"input": "folder_zip"})

        self.assertEqual(modes, {"input": "folder_zip"})


class DescribeFolderTest(unittest.TestCase):
    """What a folder HOLDS, which is the only thing that confirms it is the
    right one.

    "2_TAD_VTKFiles_L_T2 - folder, 73 MB" says nothing a wrong folder would not
    also say. The count and the kind do: pointing one level too high shows a
    different count, and pointing at nothing shows `empty`.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _write(self, relative, size=1024):
        path = os.path.join(self.dir, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"x" * size)

    def test_one_kind_is_named_on_its_own(self):
        for index in range(14):
            self._write("L%02d_T2_L.vtk" % index)

        self.assertIn("14 VTK surfaces", formgen.describe_file(self.dir))

    def test_a_single_file_is_not_pluralised(self):
        self._write("only.vtk")

        self.assertIn("1 VTK surface,", formgen.describe_file(self.dir))

    def test_a_kind_already_ending_in_s_is_left_alone(self):
        """`FILE_KINDS` holds "Slicer markups"; a blind `+ "s"` put
        "Slicer markupss" on screen."""
        self._write("a.mrk.json")
        self._write("b.mrk.json")

        self.assertIn("2 Slicer markups,", formgen.describe_file(self.dir))

    def test_a_mixed_folder_names_its_two_largest_groups(self):
        for index in range(5):
            self._write("scan%d.nii.gz" % index)
        self._write("a.mrk.json")
        self._write("b.mrk.json")
        self._write("notes.csv")

        text = formgen.describe_file(self.dir)
        self.assertIn("8 files", text)
        self.assertIn("5 NIfTI volumes", text)
        self.assertIn("2 Slicer markups", text)
        self.assertIn("and more", text)

    def test_it_counts_the_whole_tree(self):
        """A cohort whose scans sit under per-patient directories is the normal
        shape; counting only the top level would report zero for it."""
        self._write(os.path.join("Pat_01", "scan.nii.gz"))
        self._write(os.path.join("Pat_02", "scan.nii.gz"))

        self.assertIn("2 NIfTI volumes", formgen.describe_file(self.dir))

    def test_an_empty_folder_says_so(self):
        """The signal that you pointed one level too high."""
        self.assertIn("empty", formgen.describe_file(self.dir))

    def test_hidden_files_are_not_counted(self):
        """`.DS_Store` beside fourteen surfaces must not read as fifteen."""
        for index in range(3):
            self._write("s%d.vtk" % index)
        self._write(".DS_Store")

        text = formgen.describe_file(self.dir)
        self.assertIn("3 VTK surfaces", text)
        # And ONE kind, so the caption stays "3 VTK surfaces" rather than
        # becoming "4 files (3 VTK surfaces, 1 file)" -- which is what counting
        # it produced, and reads as a folder holding something unexpected.
        self.assertNotIn("4 files", text)

    def test_a_folder_of_empty_files_still_reads_as_a_folder(self):
        """`human_size(0)` is empty, and a trailing comma with nothing after it
        reads as a value that failed to load."""
        self._write("empty.vtk", size=0)

        text = formgen.describe_file(self.dir)
        self.assertFalse(text.rstrip().endswith(","), text)

    def test_a_file_is_still_described_as_a_file(self):
        self._write("scan.nii.gz", size=2048)

        self.assertIn("NIfTI volume",
                      formgen.describe_file(os.path.join(self.dir, "scan.nii.gz")))


class ValueFieldTest(unittest.TestCase):
    """The box on the LEFT of an input row, saying what the row holds.

    It replaced a wrapped label on a line of its own under the controls --
    three lines per input, on a panel where ASO has four of them. The row is
    the ordinary file-picker shape now: the value on the left, the button that
    changes it on the right.
    """

    def test_it_is_read_only_and_says_so_to_Qt(self):
        """There IS no typing path into a row, so a box a clinician can type a
        path into which is then ignored is worse than no box. Enforced rather
        than implied."""
        self.assertTrue(design.value_field("Nothing selected").isReadOnly())

    def test_it_shows_its_text_from_the_start(self):
        """A path is longest on its left and a file name is what a reader is
        looking for, so a box scrolled to the end shows the one part that means
        nothing."""
        self.assertEqual(design.value_field("/very/long/path/scan.nii.gz").cursorPosition, 0)

    def test_it_is_the_largest_text_on_the_row(self):
        """Raised twice before it read as feedback: 8pt muted was a footnote,
        10pt was still close enough to the surrounding text to be scanned
        past. Everything else on the row is a control offering a choice; this
        is the answer, and it should be the thing the eye lands on."""
        value = design.value_field("Folder: /data/cohort")
        hint = design.hint_label("CBCT only: ignored for intraoral scans")

        self.assertIn("font-size: 12pt", value._stylesheet)
        self.assertIn("font-size: 8pt", hint._stylesheet)

    def test_it_only_shouts_once_the_row_holds_something(self):
        """Empty it reads "Nothing selected" -- a prompt in full-strength
        semi-bold is a panel of four inputs all demanding attention. The weight
        and the colour are what change when a scan actually lands."""
        field = design.value_field("Nothing selected")
        empty = field._stylesheet

        design.set_input_filled(design.input_card(), field, True)

        self.assertIn("font-weight: 500", empty)
        self.assertIn("font-weight: 600", field._stylesheet)

    def test_it_is_not_a_second_slot_inside_the_first(self):
        """It sits INSIDE the input card, which is itself a filled slot: a box
        in a box, in the same colour, is to say invisible. Both declarations
        are explicit because the panel's own QLineEdit rule would otherwise
        fill and round it like a field to type in."""
        style = design.value_field("Folder: /data/cohort")._stylesheet

        self.assertIn("background: transparent", style)
        self.assertIn("border: none", style)

    def test_a_filled_row_uses_the_body_colour_not_the_muted_one(self):
        field = design.value_field("Folder: /data/cohort")
        design.set_input_filled(design.input_card(), field, True)

        self.assertIn(design.tokens()["TEXT"], field._stylesheet)
        self.assertNotIn(design.tokens()["TEXT_MUTED"], field._stylesheet)

    def test_both_themes_define_what_it_needs(self):
        """A colour defined in one theme and not the other is a KeyError in
        the theme nobody tests in."""
        for theme in (design._LIGHT, design._DARK):
            self.assertIn("TEXT", theme)


class ThemeSymmetryTest(unittest.TestCase):
    """A token defined in one theme and not the other is a KeyError in the
    theme nobody tests in -- and the theme nobody tests in is whichever one the
    author was not running."""

    def test_the_two_tables_hold_exactly_the_same_names(self):
        self.assertEqual(set(design._LIGHT), set(design._DARK))

    def test_no_token_has_the_same_value_in_both(self):
        """Not a style rule -- a detection. A value that survived a palette
        rewrite unchanged in both tables is one that was never re-chosen for
        the second theme."""
        shared = [name for name in design._LIGHT
                  if design._LIGHT[name] == design._DARK[name]
                  and name not in ("DANGER",)]
        self.assertEqual(shared, [], "carried over rather than chosen")

    def test_the_ground_is_told_from_everything_standing_on_it(self):
        """The one separation a line cannot make: the panel's ground has no
        edge of its own, so a card standing on it has to differ in colour or
        the panel is one flat sheet."""
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            self.assertNotEqual(theme["BACKGROUND"], theme["SURFACE"], name)

    def test_a_chosen_slot_cannot_be_mistaken_for_an_empty_one(self):
        """`FIELD` and `ACCENT_SOFT` are the two states of an input row, and
        the only thing that tells them apart now that neither is outlined."""
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            self.assertNotEqual(theme["FIELD"], theme["ACCENT_SOFT"], name)


class DropdownArrowTest(unittest.TestCase):
    """A dropdown has to look like one. It had Qt's default arrow -- a grey
    triangle a few pixels across -- so next to a spin box of the same size and
    the same outline, nothing said there was a list behind it.

    **And then, for a while, it had no arrow at all.** The chevron shipped as
    `url("data:image/svg+xml,<svg .../>")` and on this Slicer Qt that draws
    nothing: the sheet parses, the rule applies, the image is never resolved,
    and the only symptom is a light blue square at the end of the box. The
    icons are written to real files now and referenced by path.
    """

    def test_the_stylesheet_points_at_a_file_that_exists(self):
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            sheet = design._base_stylesheet(theme)
            paths = re.findall(r'image: url\("([^"]+)"\)', sheet)
            self.assertTrue(paths, name)
            for path in paths:
                self.assertTrue(os.path.exists(path), "{}: {}".format(name, path))

    def test_it_is_never_a_data_uri(self):
        """The whole reason the arrow was missing. Pinned so it cannot come
        back looking like a tidy-up."""
        self.assertNotIn("data:image", design._base_stylesheet(design._LIGHT))

    def test_the_chevron_is_a_filled_path_and_not_a_stroked_line(self):
        """Qt renders SVG Tiny, where a fill is the one thing every renderer
        of that profile agrees on. A stroked polyline needs four properties to
        all land, and when one does not the shape is not wrong -- it is
        absent."""
        svg = design._chevron_svg("#1f6fbf")

        self.assertIn("fill='#1f6fbf'", svg)
        self.assertNotIn("stroke", svg)

    def test_the_open_state_points_the_other_way(self):
        """The only feedback a collapsed combo box gives that its list is
        down."""
        self.assertNotEqual(design._chevron_svg("#1f6fbf"),
                            design._chevron_svg("#1f6fbf", up=True))

    def test_each_theme_gets_a_file_of_its_own(self):
        """The file is named after a digest of its contents, so a changed
        colour is a different file and a stale one is never picked up."""
        light = design._icon_url(design._chevron_svg(design._LIGHT["PRIMARY"]))
        dark = design._icon_url(design._chevron_svg(design._DARK["PRIMARY"]))

        self.assertNotEqual(light, dark)
        self.assertIn(design._LIGHT["PRIMARY"], io.open(light).read())
        self.assertIn(design._DARK["PRIMARY"], io.open(dark).read())

    def test_an_icon_that_cannot_be_written_leaves_the_rule_out(self):
        """Which falls back to the platform's own arrow. Drawing `url("")`
        instead would be the missing-arrow bug again, by another route."""
        design._ICON_FILES.clear()
        self.addCleanup(design._ICON_FILES.clear)
        original = design._icon_dir
        # A folder inside a regular file: uncreatable on every system, where
        # /proc/nowhere is only uncreatable on Linux.
        handle, blocker = tempfile.mkstemp()
        os.close(handle)
        self.addCleanup(os.remove, blocker)
        design._icon_dir = lambda: os.path.join(blocker, "sadt")
        self.addCleanup(setattr, design, "_icon_dir", original)

        self.assertEqual(design._image_rule("<svg/>"), "")

    def test_both_themes_draw_an_arrow_of_their_own(self):
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            sheet = design._base_stylesheet(theme)
            self.assertIn("QComboBox::down-arrow", sheet, name)
            self.assertIn("QComboBox::drop-down", sheet, name)

    def test_the_text_clears_the_arrow_zone(self):
        """Without the right padding a long hosted entry runs under the
        chevron rather than being elided before it."""
        sheet = design._base_stylesheet(design._LIGHT)
        self.assertIn(
            "padding-right: {}px".format(
                design.DROPDOWN_ARROW_WIDTH + design.SPACING_MD),
            sheet)


class TableSurfaceTest(unittest.TestCase):
    """ASO's landmark chooser is a table, and a table is an object you look
    into -- not a region of the panel's own ground with a hairline round it."""

    def test_the_pane_is_an_outlined_surface(self):
        """It was filled in its own colour and given a border of that same
        colour -- which on this palette is a white table on a white card, with
        nothing at all saying where it starts."""
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            rule = re.search(r"QTabWidget::pane \{(.*?)\n    \}",
                             design._base_stylesheet(theme), re.S).group(1)
            self.assertIn("background-color: {}".format(theme["SURFACE_TABLE"]),
                          rule, name)
            self.assertIn("border: 1px solid {}".format(theme["BORDER"]), rule, name)

    def test_a_tab_changes_nothing_but_its_colours_when_chosen(self):
        """Qt sizes a tab from what it holds when the bar is laid out, so
        anything that changed its box with the selection would make the open
        tab wider than its own slot and clip its label -- `Cranial base`
        rendered as `ranial bas`. Only fills and text colours move."""
        sheet = design._base_stylesheet(design._LIGHT)
        selected = re.search(r"QTabBar::tab:selected \{(.*?)\n    \}",
                             sheet, re.S).group(1)
        for property_name in ("border", "padding", "margin", "font"):
            self.assertNotIn(property_name + ":", selected)

    def test_a_table_frame_styles_itself_and_not_its_children(self):
        """An id selector, because every child of it holds a control that has
        to keep the styling the panel's own sheet gives it."""
        frame = design.table_frame()
        self.assertTrue(frame._stylesheet.startswith("#tableFrame"))


class ChipGroupSpacingTest(unittest.TestCase):
    """AMASSS declares three groups -- Bones, Soft tissue, Masks -- and they
    were drawn at the column's own 4px option spacing, so `Soft tissue` sat as
    close to the last chip of `Bones` as two chips of one group sit to each
    other. Three groups, read as one run of nine."""

    STRUCTURES = {"MAND": False, "MAX": False, "SKIN": False, "UAW": False}
    GROUPS = {"Bones": ["MAND", "MAX"], "Soft tissue": ["SKIN", "UAW"]}

    def _headings(self):
        group = formgen.MultiChoiceGroup(
            dict(self.STRUCTURES), layout="chips", groups=self.GROUPS)
        return [w for w in group.container.layout.widgets
                if getattr(w, "text", None) in self.GROUPS]

    def test_every_group_still_gets_its_heading(self):
        self.assertEqual([w.text for w in self._headings()],
                         ["Bones", "Soft tissue"])

    def test_a_heading_carries_its_air_above_it_and_none_below(self):
        """The heading and the chips it names are ONE block: every pixel spent
        separating them is a pixel that pushes the third group out of the eye's
        first pass, and comparing the three at a glance is the whole point of
        grouping them."""
        for heading in self._headings():
            self.assertIn("margin-top", heading._stylesheet)
            self.assertIn("padding: 0px", heading._stylesheet)
            self.assertNotIn("border-bottom", heading._stylesheet)


    def test_the_heading_is_not_shrunk_to_make_room(self):
        """Compactness comes from taking out the padding and the rule, never
        from making the words smaller: this panel has been told twice that its
        small text cannot be read."""
        self.assertNotIn("font-size", design.group_heading("Soft tissue")._stylesheet)

    def test_it_is_not_the_plain_section_title_it_used_to_be(self):
        self.assertNotEqual(design.group_heading("Bones")._stylesheet,
                            design.section_title("Bones")._stylesheet)


class ExplainedLabelTest(unittest.TestCase):
    """The argument's description used to be printed under the field, as a
    small grey paragraph. It is the label's tooltip now, and the label says so
    with a dotted rule -- the oldest convention there is for "there is more
    here if you hover", and one that costs the label no words."""

    def _label(self, name):
        """The row label of one argument, found by the words it shows."""
        spec = EXAMPLE_TOOL_SCHEMA["arguments"][name]
        wanted = formgen.label_for(name, spec)
        layout = qt.QFormLayout()
        formgen.build(EXAMPLE_TOOL_SCHEMA["arguments"], layout)
        return [label for label, _field in layout.rows
                if label.text.startswith(wanted)][0]

    def test_an_explained_field_is_marked_and_hovered(self):
        label = self._label("outputs")
        self.assertIn("dotted", label._stylesheet)
        self.assertEqual(label.toolTip(), "Which result files to produce")

    def test_a_field_with_nothing_to_say_carries_no_rule(self):
        plain = design.section_title("Jaws")
        self.assertNotIn("dotted", plain._stylesheet)
        self.assertNotIn("border-bottom", plain._stylesheet)

    def test_the_mark_survives_the_required_star(self):
        self.assertIn("dotted", design.required_label("Input", True)._stylesheet)
        self.assertIn("dotted", design.optional_label("Landmarks", True)._stylesheet)


class InputCardTest(unittest.TestCase):
    """One input row is up to five controls on one line -- two dropdowns, two
    browse buttons -- and a sentence under them. Laid out bare that is five
    shapes and no edge anywhere, and the question a clinician actually has
    ("have I given this tool its scan yet?") was answered only by a line of
    12pt text among all of it.
    """

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.scan = os.path.join(self.temp, "patient1.nii.gz")
        with open(self.scan, "wb") as handle:
            handle.write(b"0" * 32)

    def test_an_untouched_row_goes_white_against_the_panels_ground(self):
        """Which is the whole of what it needs to say -- see
        EverythingIsOutlinedTest for why this one block carries no line."""
        row = formgen.FileOrFolderInput()
        self.assertIn(design.tokens()["SURFACE"], row.container._stylesheet)
        self.assertIn("border: none", row.container._stylesheet)

    def test_a_filled_row_takes_the_accent(self):
        row = formgen.FileOrFolderInput()
        row.setCurrentPath(self.scan)

        self.assertIn(design.tokens()["ACCENT_SOFT"], row.container._stylesheet)

    def test_emptying_it_again_takes_the_accent_back(self):
        row = formgen.FileOrFolderInput()
        row.setCurrentPath(self.scan)
        row.setCurrentPath("")

        self.assertNotIn(design.tokens()["ACCENT_SOFT"], row.container._stylesheet)

    def test_only_the_fill_changes_with_the_state(self):
        """It is the outermost thing on the row: anything that changed its box
        would move every control inside it by a pixel the moment a file was
        chosen."""
        row = formgen.FileOrFolderInput()
        empty = row.container._stylesheet
        row.setCurrentPath(self.scan)
        filled = row.container._stylesheet

        strip = lambda sheet: re.sub(r"background-color:[^;]*;", "", sheet)
        self.assertEqual(strip(empty), strip(filled))

    def test_it_styles_itself_and_not_the_controls_inside_it(self):
        """Every child of the card holds a control that has to keep the
        styling the panel's own sheet gives it."""
        self.assertTrue(formgen.FileOrFolderInput().container._stylesheet
                        .startswith("#inputCard"))

    def test_a_wrapped_row_paints_the_box_the_panel_actually_shows(self):
        """The picker inside a `ServerFileInput` is never added to a layout --
        only its buttons are -- so painting its own card would leave the box a
        clinician can see saying the row is still empty."""
        widget = formgen.file_widget(_VOLUME_SPEC, "file_or_folder")
        widget.setChoices([{"name": "sample_scan.nii.gz", "kind": "file", "size": 94}])
        formgen.set_local_path(widget, self.scan)

        self.assertIn(design.tokens()["ACCENT_SOFT"], widget.container._stylesheet)

    def test_a_scene_pick_fills_the_row_though_no_path_was_chosen(self):
        """An open volume is exported at upload time and has no local path at
        all, so a card driven off `currentPath` would call a satisfied row
        empty."""
        widget = formgen.file_widget(_VOLUME_SPEC, "file_or_folder")
        widget.setSceneSupported(True)
        widget.setVolumeChoices(["CBCT_patient1"])
        widget.sceneCombo.setCurrentIndex(1)

        self.assertTrue(widget.volume_name(), "the scene pick did not register")
        self.assertIn(design.tokens()["ACCENT_SOFT"], widget.container._stylesheet)

    def test_the_line_inside_it_says_the_same_thing_the_box_does(self):
        """The box says THAT the row is satisfied and the line says WITH WHAT.
        A panel where those two disagreed would be worse than either alone."""
        row = formgen.FileOrFolderInput()
        row.setCurrentPath(self.scan)

        self.assertIn("patient1.nii.gz", row.caption.text)
        self.assertIn("font-weight: 600", row.caption._stylesheet)


class EverythingIsOutlinedTest(unittest.TestCase):
    """Every control a clinician operates is a white surface with a hairline
    round it -- and the line is what separates it from the card it sits on,
    which is why the two can be the same colour.

    A borderless treatment was built and compared against this one on a real
    panel: a control as a tinted slot sunk into its card, no lines anywhere. It
    read as washed out on the screens this actually runs on, and the fills it
    depended on kept drifting together. What is pinned here is the decision.
    """

    #: The rules that draw a control the user acts on.
    CONTROLS = ("QLineEdit", "QComboBox", "QSpinBox", "QCheckBox::indicator",
                "QTabWidget::pane", "QTabBar::tab", "QProgressBar")

    def _sheet(self, theme):
        return re.sub(r"/\*.*?\*/", "", design._base_stylesheet(theme), flags=re.S)

    def test_every_control_carries_the_hairline(self):
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            sheet = self._sheet(theme)
            for selector in self.CONTROLS:
                rule = re.search(re.escape(selector) + r"[^{}]*\{([^{}]*)\}", sheet)
                self.assertIsNotNone(rule, "{}: {}".format(name, selector))
                self.assertIn("1px solid {}".format(theme["BORDER"]), rule.group(1),
                              "{}: {}".format(name, selector))

    def test_the_edge_is_one_pixel_in_every_state(self):
        """Only the COLOUR moves. A hairline that thickened on focus would grow
        its field by a pixel under the pointer, Qt laying a row out from the
        border box -- and a tab that did it would clip its own label."""
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            widths = set(re.findall(r"border:\s*(\d+)px solid", self._sheet(theme)))
            self.assertEqual(widths, {"1"}, name)

    def test_the_accent_is_what_a_focused_control_takes(self):
        sheet = self._sheet(design._LIGHT)
        self.assertIn("border-color: {}".format(design._LIGHT["PRIMARY"]), sheet)

    def test_a_filled_surface_factory_carries_it_too(self):
        """The trap the borderless pass left behind: on this palette FIELD and
        SURFACE are both white, so a chip or a frame filled with one and given
        no edge is white on white -- an option nobody can see until they hover
        it."""
        for widget in (design.option_chip("MAND"), design.segment_button("File"),
                       design.ghost_button("Select all"),
                       design.table_frame(), design.cohort_frame()):
            self.assertIn("1px solid", widget._stylesheet, widget)

    def test_the_input_row_is_the_one_exception(self):
        """A hairline separates a CONTROL from the card it sits on. An input
        row is not a control -- it is a block standing on the panel's own grey
        ground, and going white against that ground already says where it
        starts. An outline on it was the fifth line in a row of four, drawn
        round something that was not in any doubt."""
        card = design.input_card()

        self.assertIn("border: none", card._stylesheet)
        self.assertIn(design.tokens()["SURFACE"], card._stylesheet)


class EveryColourComesFromTheTablesTest(unittest.TestCase):
    """The trap: a hard-coded hex survives light mode and disappears in dark,
    which is the theme nobody tests in. Checked over the WHOLE sheet rather
    than one rule at a time -- the tooltip-only version of this test was
    passing while six other rules carried their own colours."""

    def _sources(self, theme, fills):
        known = set(theme.values())
        known |= {colour for role in fills.values() for colour in role.values()}
        known |= {design._TOGGLE_OFF, design._TOGGLE_ON}
        return known

    def test_no_rule_invents_a_colour(self):
        for name, theme, fills in (
                ("light", design._LIGHT, design._BUTTON_FILLS_LIGHT),
                ("dark", design._DARK, design._BUTTON_FILLS_DARK)):
            known = self._sources(theme, fills)
            body = re.sub(r"/\*.*?\*/", "", design._base_stylesheet(theme), flags=re.S)
            loose = {found for found in re.findall(r"#[0-9a-fA-F]{6}", body)
                     if found not in known}
            self.assertEqual(loose, set(), name)

    def test_the_chevron_is_drawn_in_the_accent_of_its_own_theme(self):
        """It is a drawn icon, so its colour is baked into a file -- the one
        place a theme colour could be frozen without anyone noticing."""
        for name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            drawn = design._icon_url(design._chevron_svg(theme["PRIMARY"]))
            self.assertIn(theme["PRIMARY"], io.open(drawn).read(), name)

    def test_a_sheet_rendered_for_one_theme_holds_that_themes_buttons(self):
        """`_button_stylesheet` took a palette and then asked the application
        which theme it was, so a caller handing it one got the buttons of the
        other. Harmless while only `apply` calls it -- and exactly the kind of
        agreement that holds until it does not."""
        dark = design._base_stylesheet(design._DARK)
        self.assertIn(design._BUTTON_FILLS_DARK["primary"]["base"], dark)
        self.assertNotIn(design._BUTTON_FILLS_LIGHT["primary"]["base"], dark)


class OneSourceAtATimeTest(unittest.TestCase):
    """A file argument can be satisfied four ways -- a file on this machine, a
    folder, the test data the server hosts, a scan already open in Slicer --
    and exactly one at a time.

    That rule used to be enforced invisibly: all four controls sat on the row
    at once and picking in one silently emptied the others. Now the rule IS the
    interface -- one segment pressed, one control on the row.
    """

    SOURCES = formgen.ServerFileInput

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.widget = formgen.file_widget(_VOLUME_SPEC, "file_or_folder")
        self.widget.setChoices([
            {"name": "sample_scan.nii.gz", "kind": "file", "size": 94},
        ])
        self.widget.setSceneSupported(True)
        self.widget.setVolumeChoices(["CBCT_patient1"])

    def _file(self, name="patient1.nii.gz"):
        path = os.path.join(self.temp, name)
        with open(path, "wb") as handle:
            handle.write(b"0" * 16)
        return path

    def test_all_four_sources_are_offered(self):
        self.assertEqual(list(self.widget.sourceButtons),
                         [self.SOURCES.SOURCE_FILE, self.SOURCES.SOURCE_FOLDER,
                          self.SOURCES.SOURCE_HOSTED, self.SOURCES.SOURCE_SCENE])

    def test_exactly_one_segment_is_pressed(self):
        for key in list(self.widget.sourceButtons):
            self.widget.sourceButtons[key].click()
            pressed = [name for name, button in self.widget.sourceButtons.items()
                       if button.isChecked()]
            self.assertEqual(pressed, [key])

    def test_exactly_one_control_is_on_the_right_of_the_row(self):
        """Three controls share the right-hand slot -- the `Select` button
        serving both local sources, and the two lists -- and which one is SHOWN
        is the whole answer to "where is this input coming from"."""
        controls = {
            "select": self.widget.local.selectButton,
            "hosted": self.widget.combo,
            "scene": self.widget.sceneCombo,
        }
        expected = {
            self.SOURCES.SOURCE_FILE: "select",
            self.SOURCES.SOURCE_FOLDER: "select",
            self.SOURCES.SOURCE_HOSTED: "hosted",
            self.SOURCES.SOURCE_SCENE: "scene",
        }
        for key in list(self.widget.sourceButtons):
            self.widget.sourceButtons[key].click()
            shown = [name for name, widget in controls.items() if widget.isVisible()]
            self.assertEqual(shown, [expected[key]], key)

    def test_the_value_field_is_never_the_one_that_hides(self):
        """Every source fills the same row, and what it holds is the one thing
        that does not depend on where it came from."""
        for key in list(self.widget.sourceButtons):
            self.widget.sourceButtons[key].click()
            self.assertTrue(self.widget.caption.isVisible(), key)

    def test_the_button_opens_the_dialog_the_chosen_source_asks_for(self):
        """Which is why it can say `Select` rather than naming a kind the
        pressed segment already names."""
        self.widget.sourceButtons[self.SOURCES.SOURCE_FOLDER].click()
        self.assertEqual(self.widget.local._mode, "folder")

        self.widget.sourceButtons[self.SOURCES.SOURCE_FILE].click()
        self.assertEqual(self.widget.local._mode, "file")

    def test_switching_source_empties_the_row(self):
        """Emptying is the point rather than a side effect: a row that kept its
        scan while showing the folder button would be saying two things at
        once."""
        self.widget.sourceButtons[self.SOURCES.SOURCE_FILE].click()
        formgen.set_local_path(self.widget, self._file())
        self.assertTrue(self.widget.currentPath)

        self.widget.sourceButtons[self.SOURCES.SOURCE_SCENE].click()

        self.assertEqual(self.widget.currentPath, "")
        self.assertEqual(self.widget.caption.text, formgen.NOTHING_CHOSEN)

    def test_a_row_with_one_source_shows_no_segments_at_all(self):
        """A choice of one is not a choice, and a single pressed segment over
        a lone control is a decoration that looks like a decision."""
        alone = formgen.file_widget(dict(_VOLUME_SPEC, server_selectable=None),
                                    "single_file")
        self.assertEqual(alone.sourceButtons, {})
        self.assertFalse(alone.sourceBar.isVisible())

    def test_a_source_that_arrives_late_gets_its_segment(self):
        """The hosted list arrives with the schema and the scene list is
        refreshed on every enter(): what a row can be filled from moves under
        the panel, so the bar is rebuilt rather than drawn once."""
        late = formgen.file_widget(_VOLUME_SPEC, "single_file")
        self.assertNotIn(self.SOURCES.SOURCE_HOSTED, late.sourceButtons)

        late.setChoices([{"name": "sample_scan.nii.gz", "kind": "file", "size": 1}])

        self.assertIn(self.SOURCES.SOURCE_HOSTED, late.sourceButtons)

    def test_the_chosen_source_survives_a_refresh_that_still_offers_it(self):
        """Both lists are refreshed on every enter(), and a refresh must not
        silently move the row back to its first source."""
        self.widget.sourceButtons[self.SOURCES.SOURCE_SCENE].click()

        self.widget.setVolumeChoices(["CBCT_patient1", "CBCT_patient2"])

        self.assertTrue(
            self.widget.sourceButtons[self.SOURCES.SOURCE_SCENE].isChecked())

    def test_a_source_that_goes_away_hands_the_row_to_another(self):
        """A tool whose DATA/ folder was emptied between two runs, while the
        row was sitting on its test data."""
        self.widget.sourceButtons[self.SOURCES.SOURCE_HOSTED].click()

        self.widget.setChoices([])

        self.assertNotIn(self.SOURCES.SOURCE_HOSTED, self.widget.sourceButtons)
        self.assertEqual(self.widget._source, self.SOURCES.SOURCE_FILE)
        self.assertTrue(self.widget.local.selectButton.isVisible())

    def test_a_hosted_model_calls_its_segment_what_it_is(self):
        """A model is never fetched -- the weights stay on the server and the
        run names them -- so `Test data` would be wrong twice over."""
        row = formgen.file_widget(dict(_VOLUME_SPEC, server_selectable="model"),
                                  "single_file")
        row.setChoices([{"name": "AMASSS_models", "kind": "folder", "size": 1}])

        self.assertEqual(row.sourceButtons[self.SOURCES.SOURCE_HOSTED].text,
                         self.SOURCES.SOURCE_MODEL_LABEL)

    def test_every_segment_says_what_it_means_on_hover(self):
        """Four one-word labels on a narrow panel: `Imported` alone does not
        say imported into WHAT."""
        for button in self.widget.sourceButtons.values():
            self.assertTrue(button.toolTip())


class TheSurfacesAreTellingApartTest(unittest.TestCase):
    """With no outlines anywhere, the only thing separating a control from what
    holds it is how far apart their two fills are. The first borderless pass
    put BACKGROUND at #e8ecf2 and FIELD at #e7ebf1 -- ONE count apart -- and
    the panel read as washed out because it was."""

    #: Perceived-brightness gap two neighbouring surfaces must clear. Eight is
    #: about where a step stops being deniable on a mid-range clinical monitor
    #: in a bright room, which is the screen this runs on.
    STEP = 8

    #: The pairs that actually touch on screen, with the pair each one is.
    #: The pairs that have to be told apart by COLOUR. `SURFACE` and `FIELD`
    #: are deliberately absent, and are the same white: the hairline between a
    #: control and its card is what separates those two, which is the whole
    #: bargain of an outlined design.
    NEIGHBOURS = (
        ("BACKGROUND", "SURFACE"),      # a section card on the panel's ground
        ("FIELD", "FIELD_HOVER"),       # a control under the pointer
        ("FIELD", "ACCENT_SOFT"),       # empty against chosen
    )

    @staticmethod
    def _brightness(colour):
        red, green, blue = (int(colour[index:index + 2], 16) for index in (1, 3, 5))
        return 0.299 * red + 0.587 * green + 0.114 * blue

    def test_every_pair_that_touches_is_a_step_you_can_see(self):
        for theme_name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            for lower, upper in self.NEIGHBOURS:
                gap = abs(self._brightness(theme[lower]) - self._brightness(theme[upper]))
                self.assertGreaterEqual(
                    gap, self.STEP,
                    "{}: {} and {} are {:.0f} apart".format(theme_name, lower, upper, gap))

    def test_text_stands_well_clear_of_the_surface_it_sits_on(self):
        """The muted colour is the one at risk: it is the quieter of the two
        and it carries every label on the panel."""
        for theme_name, theme in (("light", design._LIGHT), ("dark", design._DARK)):
            for ground in ("SURFACE", "FIELD"):
                gap = abs(self._brightness(theme["TEXT_MUTED"])
                          - self._brightness(theme[ground]))
                self.assertGreater(gap, 80, "{}: TEXT_MUTED on {}".format(theme_name, ground))


class OutputFolderRowTest(unittest.TestCase):
    """The output folder is built by the PANEL, not by the schema -- and it was
    the last ctkPathLineEdit in a generated panel: an editable box with a small
    grey `...` at its end, sitting under four rows that had stopped looking
    anything like it. It is the same row as every other now, in folder mode.
    """

    def setUp(self):
        # Exactly as base_widget builds it.
        self.row = formgen.FileOrFolderInput(modes=("folder",), destination=True)
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, True)

    def test_it_shows_the_whole_path(self):
        """A patient's scan is identified by its NAME and the directory above
        it is noise. A folder results are about to be written into is
        identified by WHERE it is -- `out` or `Documents` on its own says
        nothing about which of them it is."""
        formgen.set_local_path(self.row, self.temp)

        self.assertIn(self.temp, self.row.caption.text)

    def test_it_never_takes_the_accent(self):
        """The accent answers "have I given this tool its scan yet?", and this
        row fills ITSELF in the moment the panel opens -- so accented it would
        be a blue block sitting permanently on every panel, saying something
        that was never in doubt and pulling the eye off the rows where it is."""
        formgen.set_local_path(self.row, self.temp)

        self.assertNotIn(design.tokens()["ACCENT_SOFT"], self.row.container._stylesheet)
        self.assertIn(design.tokens()["SURFACE"], self.row.container._stylesheet)

    def test_its_text_still_goes_to_full_strength(self):
        """What is there is worth reading whether or not the box is lit."""
        formgen.set_local_path(self.row, self.temp)

        self.assertIn("font-weight: 600", self.row.caption._stylesheet)

    def test_an_input_row_is_the_other_way_round(self):
        """The same two decisions, both inverted -- which is what makes
        `destination` one flag rather than two."""
        scan = os.path.join(self.temp, "patient1.nii.gz")
        with open(scan, "wb") as handle:
            handle.write(b"0" * 16)
        row = formgen.FileOrFolderInput(modes=("file",))

        row.setCurrentPath(scan)

        self.assertNotIn(self.temp, row.caption.text)
        self.assertIn("patient1.nii.gz", row.caption.text)
        self.assertIn(design.tokens()["ACCENT_SOFT"], row.container._stylesheet)

    def test_it_offers_one_primary_select_button(self):
        self.assertEqual(self.row.selectButton.text, formgen.SELECT_LABEL)
        self.assertIn(design._fills_for(design.tokens())["primary"]["base"],
                      self.row.selectButton._stylesheet)

    def test_its_one_button_opens_the_folder_dialog(self):
        qt.QFileDialog.next_directory = self.temp
        self.addCleanup(setattr, qt.QFileDialog, "next_directory", "")

        self.row.selectButton.clicked.emit()

        self.assertEqual(self.row.currentPath, self.temp)

    def test_a_proposed_folder_is_written_through_the_generic_writer(self):
        """`_suggestOutputFolder` fills this row in so Apply works on a panel
        nobody set up, and it has to announce it or Apply stays grey."""
        seen = []
        formgen.connect_changed(self.row, lambda *args: seen.append(1))

        formgen.set_local_path(self.row, self.temp)

        self.assertEqual(self.row.currentPath, self.temp)
        self.assertTrue(seen, "the folder was filled in and nothing was told")

    def test_assigning_currentPath_directly_is_refused(self):
        """The trap this row replaced a ctkPathLineEdit into: that widget takes
        `widget.currentPath = path`, and this one raises. Pinned so the panel
        keeps writing through `set_local_path`, which serves both."""
        with self.assertRaises(AttributeError):
            self.row.currentPath = self.temp
