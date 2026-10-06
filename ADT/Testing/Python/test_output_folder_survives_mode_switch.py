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

    def test_nothing_empties_the_output_field(self):
        """The whole defect in one assertion, for both modules."""
        for module, (path, _) in MODULES.items():
            with self.subTest(module=module):
                culprits = sorted(
                    node.name for node in ast.walk(_tree(path))
                    if isinstance(node, ast.FunctionDef)
                    and OUTPUT_FIELD in _fields_set_to_empty(node)
                )
                self.assertEqual(culprits, [],
                                 f"{module}: {culprits} empties {OUTPUT_FIELD}")

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
