# The output folder the user typed must survive a change of mode.
#
# `SwitchType` runs on every `activated` of either combo -- re-picking the entry
# already selected included -- and it called `ClearAllLineEdits`, which emptied
# `lineEditOutputPath` along with the scan and model paths. Nothing said so. The
# user then clicked `Test Files`, which fills that field *only when it is empty*
# with the test set's OWN folder, in whichever tree that mode downloads to. On
# 2026-10-06 a prod run wrote 2.5 GB into `Oriented-Automated/Registered` while
# its author was watching an empty `Fully-Automated/Registered`, and spent the
# run believing the pipeline had produced nothing.
#
# ASO.py carried the identical mistake in its own `SwitchType`, so both are
# covered here.
#
# Neither module can be imported outside a running Slicer (`qt` is the
# in-application PythonQt shim), so this reads the source, the way
# test_pip_requirements.py does. Two invariants:
#
#   - nothing in AREG.py ever empties the output field;
#   - the clearing that remains happens only when the method really changed.
import ast
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
OUTPUT_FIELD = "lineEditOutputPath"

#: module -> (source, the fields a change of method really does invalidate).
#: ASO carried the identical mistake in its own `SwitchType`.
MODULES = {
    "AREG": (
        os.path.join(_ROOT, "AREG", "AREG.py"),
        {"lineEditScanT1LmPath", "lineEditScanT2LmPath",
         "lineEditModel1", "lineEditModel2"},
    ),
    "ASO": (
        os.path.join(_ROOT, "ASO", "ASO.py"),
        {"lineEditScanLmPath", "lineEditRefFolder",
         "lineEditModelAli", "lineEditModelSegOr"},
    ),
}


def _tree(path):
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read())


def _function(path, name):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _fields_set_to_empty(node):
    """The `self.ui.<field>.setText("")` this node performs, by field name."""
    found = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "setText"):
            continue
        if not (len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == ""):
            continue
        target = func.value          # self.ui.<field>
        if isinstance(target, ast.Attribute):
            found.add(target.attr)
    return found


class OutputFolderIsNeverClearedTest(unittest.TestCase):

    def test_the_sources_are_where_we_think_they_are(self):
        """Without this, every case below would pass on an empty tree."""
        for module, (path, _) in MODULES.items():
            with self.subTest(module=module):
                self.assertTrue(os.path.isfile(path), path)
                self.assertIsNotNone(_function(path, "SwitchType"))

    #: The one place allowed to empty the output field, and only inside a branch
    #: guarded by `_auto_output` -- i.e. only the path `Test Files` itself wrote.
    #: This exception did not exist on 2026-10-09: the rule then was "nothing
    #: empties it", which protected the user's folder and froze ours, so clicking
    #: `Test Files` in one mode kept offering another mode's path.
    MAY_CLEAR_WHEN_GUARDED = {"SwitchType"}

    def test_nothing_empties_the_output_field_unconditionally(self):
        """The whole defect in one assertion, for both modules."""
        for module, (path, _) in MODULES.items():
            with self.subTest(module=module):
                culprits = sorted(
                    node.name for node in ast.walk(_tree(path))
                    if isinstance(node, ast.FunctionDef)
                    and OUTPUT_FIELD in _fields_set_to_empty(node)
                    and node.name not in self.MAY_CLEAR_WHEN_GUARDED
                )
                self.assertEqual(culprits, [],
                                 f"{module}: {culprits} empties {OUTPUT_FIELD}")

    def test_the_one_exception_is_really_guarded(self):
        """`SwitchType` may clear it, but only the path it filled in itself."""
        for module, (path, _) in MODULES.items():
            switch = _function(path, "SwitchType")
            if switch is None or OUTPUT_FIELD not in _fields_set_to_empty(switch):
                continue
            with self.subTest(module=module):
                guarded = [b for b in ast.walk(switch)
                           if isinstance(b, ast.If)
                           and "_auto_output" in ast.dump(b.test)
                           and OUTPUT_FIELD in _fields_set_to_empty(b)]
                self.assertTrue(
                    guarded,
                    f"{module}: SwitchType empties {OUTPUT_FIELD} outside any "
                    "branch that checks it was auto-filled")

    def test_the_mode_specific_paths_are_still_cleared(self):
        """The fix must not turn into "clear nothing": an IOS path left in a
        CBCT run is worse than an empty field."""
        for module, (path, fields) in MODULES.items():
            with self.subTest(module=module):
                node = _function(path, "ClearModeSpecificPaths")
                self.assertIsNotNone(node, f"{module}: the method is gone")
                self.assertEqual(_fields_set_to_empty(node), fields)


class ClearingIsGuardedTest(unittest.TestCase):

    def test_switchtype_clears_only_when_the_method_changed(self):
        for module, (path, _) in MODULES.items():
            with self.subTest(module=module):
                switch = _function(path, "SwitchType")
                guarded = []
                for branch in ast.walk(switch):
                    if not isinstance(branch, ast.If):
                        continue
                    calls = {c.func.attr for c in ast.walk(branch)
                             if isinstance(c, ast.Call)
                             and isinstance(c.func, ast.Attribute)}
                    if "ClearModeSpecificPaths" not in calls:
                        continue
                    names = {n.id for n in ast.walk(branch.test)
                             if isinstance(n, ast.Name)}
                    attrs = {n.attr for n in ast.walk(branch.test)
                             if isinstance(n, ast.Attribute)}
                    guarded.append(names | attrs)
                self.assertTrue(guarded,
                                f"{module}: the clearing is in no `if` at all")
                for condition in guarded:
                    self.assertIn("previous_method", condition)
                    self.assertTrue(
                        any(c.startswith("ActualMeth") for c in condition),
                        f"{module}: the guard does not look at the method: "
                        f"{sorted(condition)}")

    def test_the_previous_method_is_read_before_it_is_reassigned(self):
        """`previous_method` only means anything if it is captured first."""
        for module, (path, _) in MODULES.items():
            with self.subTest(module=module):
                switch = _function(path, "SwitchType")
                capture = assign = None
                for node in ast.walk(switch):
                    if not (isinstance(node, ast.Assign)
                            and len(node.targets) == 1):
                        continue
                    target = node.targets[0]
                    if (isinstance(target, ast.Name)
                            and target.id == "previous_method"):
                        capture = node.lineno
                    if (isinstance(target, ast.Attribute)
                            and target.attr.startswith("ActualMeth")
                            and assign is None):
                        assign = node.lineno
                self.assertIsNotNone(capture,
                                     f"{module}: previous_method never captured")
                self.assertIsNotNone(assign,
                                     f"{module}: the method is never reassigned")
                self.assertLess(capture, assign)


if __name__ == "__main__":
    unittest.main()


class AutoFilledOutputFollowsTheModeTest(unittest.TestCase):
    """`Test Files` fills the output folder; that path belongs to the mode.

    Protecting the user's folder from a change of mode is one half of the rule.
    The other half was missed and showed up three days later: a path THIS button
    had filled in survived into another mode, where it is simply wrong -- every
    mode downloads its own test set, under its own scan type. Clicking
    `Test Files` in IOS-to-CBCT kept offering
    `AREG_CBCT/Test_Files/Oriented-Automated/Registered`, a CBCT path from a
    CBCT session, because `== ""` cannot replace a field that is not empty.

    The two are told apart by memory, not by guessing: `_auto_output` holds what
    the button wrote, and the field counts as ours only while it still matches.
    """

    AREG = MODULES["AREG"][0]

    def source(self):
        with open(self.AREG, encoding="utf-8") as handle:
            return handle.read()

    def function(self, name, inside="AREGWidget"):
        """A method of the WIDGET class.

        Not the first `__init__` in the file: that one belongs to the
        ScriptedLoadableModule and sets `title`, `categories` and such. Looking
        it up by name alone is how the first version of this test failed.
        """
        for cls in ast.walk(ast.parse(self.source())):
            if not (isinstance(cls, ast.ClassDef) and cls.name == inside):
                continue
            for node in ast.walk(cls):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return node
        return None

    def test_the_auto_filled_path_is_remembered(self):
        """Without this attribute there is nothing to compare against."""
        self.assertTrue("_auto_output" in self.source(),
                        "AREG.py never mentions _auto_output")
        init = self.function("__init__")
        assigned = {t.attr for node in ast.walk(init)
                    if isinstance(node, ast.Assign)
                    for t in node.targets
                    if isinstance(t, ast.Attribute)}
        self.assertIn("_auto_output", assigned,
                      "_auto_output must start empty, or the first comparison "
                      "raises AttributeError")

    def test_test_files_no_longer_refuses_a_non_empty_field(self):
        """`== ""` alone is what left the stale CBCT path in place."""
        fill = self.function("FillFromTestFiles")
        self.assertIsNotNone(fill)
        body = ast.dump(fill)
        self.assertTrue("_auto_output" in body,
                      "FillFromTestFiles must compare the field against what "
                      "it last wrote, not only against the empty string")

    def test_a_change_of_method_drops_our_path_and_only_ours(self):
        switch = self.function("SwitchType")
        self.assertIsNotNone(switch)
        guarded = [branch for branch in ast.walk(switch)
                   if isinstance(branch, ast.If)
                   and "_auto_output" in ast.dump(branch.test)]
        self.assertTrue(guarded,
                        "SwitchType must drop the auto-filled output folder "
                        "when the method changes")
        for branch in guarded:
            # it may only clear it, never set it to something else
            cleared = [c for c in ast.walk(branch)
                       if isinstance(c, ast.Call)
                       and isinstance(c.func, ast.Attribute)
                       and c.func.attr == "setText"]
            for call in cleared:
                self.assertTrue(
                    call.args and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "",
                    "a change of mode may clear the field, not refill it")

