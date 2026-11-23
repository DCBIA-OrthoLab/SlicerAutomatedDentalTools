"""Reading the report a tool writes beside its results.

Four modules carried a byte-identical copy of this, differing only in the file
name they looked for, and not one line of it was covered by a test. It lives in
`ServerToolWidgetBase` now, driven by `RUN_REPORT`.

What the shared version has to keep is the reason every copy existed: the
report is a SUMMARY. The results are already on disk and are what the user
asked for, so a report that is missing, unreadable or not JSON costs them the
summary and nothing else. Every failure here returns None; none of them raises.
"""

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: E402,F401 - installs the stubs

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase  # noqa: E402


class Reader(ServerToolWidgetBase):
    RUN_REPORT = "Tool_report.json"


class RunReportTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, relative, content):
        path = os.path.join(self.dir, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def test_a_report_at_the_top_is_read(self):
        self.write("Tool_report.json", json.dumps({"summary": "3 scans"}))
        self.assertEqual(Reader._readRunReport(self.dir), {"summary": "3 scans"})

    def test_a_report_below_the_top_is_found(self):
        """A tool that mirrors its input tree files the report with the results.

        AutoMatrix, GreedyReg and DOCShapeAXI all globbed recursively for this
        reason; ALI looked only at the top. The shared reader does both, so
        neither module loses its summary.
        """
        self.write(os.path.join("patient1", "Tool_report.json"), '{"summary": "ok"}')
        self.assertEqual(Reader._readRunReport(self.dir), {"summary": "ok"})

    def test_the_top_level_report_wins_over_a_nested_one(self):
        self.write("Tool_report.json", '{"where": "top"}')
        self.write(os.path.join("patient1", "Tool_report.json"), '{"where": "nested"}')
        self.assertEqual(Reader._readRunReport(self.dir), {"where": "top"})

    def test_two_nested_reports_always_pick_the_same_one(self):
        """Sorted, so a rerun does not summarize a different patient."""
        self.write(os.path.join("b", "Tool_report.json"), '{"which": "b"}')
        self.write(os.path.join("a", "Tool_report.json"), '{"which": "a"}')
        first = Reader._readRunReport(self.dir)
        self.assertEqual(first, {"which": "a"})
        self.assertEqual(Reader._readRunReport(self.dir), first)

    def test_no_report_is_None_and_never_raises(self):
        self.assertIsNone(Reader._readRunReport(self.dir))

    def test_a_missing_output_folder_is_None_and_never_raises(self):
        self.assertIsNone(Reader._readRunReport(os.path.join(self.dir, "gone")))

    def test_malformed_json_is_None_and_never_raises(self):
        """Truncated is what a run killed mid-write leaves behind."""
        self.write("Tool_report.json", '{"summary": "3 sca')
        self.assertIsNone(Reader._readRunReport(self.dir))

    def test_a_module_declaring_no_report_asks_for_nothing(self):
        """The default, and it must not walk the output folder to find it out."""
        self.write("Tool_report.json", '{"summary": "ok"}')

        class NoReport(ServerToolWidgetBase):
            pass

        self.assertIsNone(NoReport._readRunReport(self.dir))


if __name__ == "__main__":
    unittest.main()
