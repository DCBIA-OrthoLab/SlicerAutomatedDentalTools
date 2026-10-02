# Every (library, version) pair the repository hands to pip must parse.
#
# Written because grepping was not enough: `f'{lib}=={version}'` is correct
# only while every entry of a list carries a bare version, and three modules
# had quietly gained a constraint with an operator -- producing
# `dicom2nifti==>=2.6.2`, which pip refuses outright. The defect is invisible
# on a machine that already has the library, so no amount of running the
# module locally shows it.
#
# This walks the source instead of a list written by hand, so a list that
# gains a constraint tomorrow is covered without anyone remembering to come
# back here.
import ast
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from packaging.requirements import Requirement  # noqa: E402

from ADTLib.env.deps import requirement  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))

# A package name, then a bare version or a PEP 440 constraint.
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{1,40}$")
_VERSION = re.compile(r"^\s*(?:[<>=!~]{1,2}\s*)?[0-9][0-9A-Za-z.\-+*,<>=!~ ]*$")

SKIP_DIRS = {".git", "__pycache__", "build", "archive"}


def _library_pairs():
    """Every `('name', 'version')` literal that looks like a pip requirement.

    A false positive costs nothing: it is still a pair that `requirement()`
    turns into a string, and the test only fails when the string does not
    parse. A false negative is what matters, so the filter stays loose.
    """
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                try:
                    tree = ast.parse(handle.read())
                except SyntaxError:                  # pragma: no cover
                    continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Tuple, ast.List)):
                    continue
                items = node.elts
                if not 2 <= len(items) <= 3:
                    continue
                if not isinstance(items[0], ast.Constant) or not isinstance(items[0].value, str):
                    continue
                if not _NAME.match(items[0].value):
                    continue
                second = items[1]
                if isinstance(second, ast.Constant) and second.value is None:
                    yield os.path.relpath(path, REPO), items[0].value, None
                elif isinstance(second, ast.Constant) and isinstance(second.value, str) \
                        and _VERSION.match(second.value):
                    yield os.path.relpath(path, REPO), items[0].value, second.value


class PipRequirementsTest(unittest.TestCase):

    def test_the_scan_finds_the_known_lists(self):
        """Guards the test below from passing because it found nothing."""
        found = {(lib, version) for _, lib, version in _library_pairs()}
        self.assertIn(("dicom2nifti", ">=2.6.2"), found)
        self.assertIn(("pydicom", "3.0.2"), found)
        self.assertGreater(len(found), 20, "the scan sees almost nothing")

    def test_pip_accepts_every_pair_in_the_repository(self):
        broken = []
        for path, lib, version in _library_pairs():
            text = requirement(lib, version)
            try:
                Requirement(text)
            except Exception as error:
                broken.append("%s: %r + %r -> %r (%s)" % (path, lib, version, text, error))
        self.assertEqual(broken, [], "\n".join(broken))


if __name__ == "__main__":
    unittest.main()
