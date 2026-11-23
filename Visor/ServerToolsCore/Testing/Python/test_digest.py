"""Unit tests for ServerToolsCoreLib.digest, run outside Slicer.

This module is what decides how much of a reviewed cohort travels back to the
server, so it is tested against real files on a real disk rather than against
a mocked filesystem: the one question it answers is whether two sequences of
bytes are the same, and a mock cannot be wrong about that in the way a disk
can.

**No qt/slicer stubs are imported here, deliberately.** `digest.py` must stay
free of both -- it is pure logic over a directory, and the day it grows a
`slicer.util` call it stops being testable in plain CI. This file failing to
import IS that regression.

Usage:
    python3 -m unittest test_digest
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ServerToolsCoreLib import digest


class DigestTreeTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _write(self, relative, data="{}"):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(data)
        return path

    def test_every_file_is_keyed_by_its_path_relative_to_the_folder(self):
        self._write("p1.mrk.json")
        self._write(os.path.join("sub", "p2.mrk.json"))

        marks = digest.digest_tree(self.root)

        self.assertEqual(sorted(marks),
                         ["p1.mrk.json", os.path.join("sub", "p2.mrk.json")])

    def test_two_files_of_identical_content_have_identical_digests(self):
        self._write("a.json", "same")
        self._write("b.json", "same")

        marks = digest.digest_tree(self.root)

        self.assertEqual(marks["a.json"], marks["b.json"])

    def test_a_folder_that_does_not_exist_is_an_empty_mapping(self):
        """A checkpoint that unpacked nothing must not raise on the way to
        reporting that the reader changed nothing."""
        self.assertEqual(digest.digest_tree(os.path.join(self.root, "absent")), {})
        self.assertEqual(digest.digest_tree(""), {})

    def test_a_file_read_in_several_blocks_digests_as_its_whole_content(self):
        """The read is chunked because a step is hundreds of megabytes; a
        digest that only covered the first block would call two different
        cohorts identical."""
        self.addCleanup(setattr, digest, "_READ_BYTES", digest._READ_BYTES)
        digest._READ_BYTES = 8
        self._write("long.json", "x" * 100)
        chunked = digest.digest_tree(self.root)["long.json"]

        digest._READ_BYTES = 1024 * 1024
        whole = digest.digest_tree(self.root)["long.json"]

        self.assertEqual(chunked, whole)

    def test_a_file_that_cannot_be_read_is_left_out_rather_than_recorded(self):
        path = self._write("locked.json")
        os.chmod(path, 0)
        self.addCleanup(os.chmod, path, 0o600)
        if os.access(path, os.R_OK):
            self.skipTest("running as a user that can read a mode-0 file")

        # assertLogs as well as an assertion: it pins that the file is
        # REPORTED rather than silently dropped, and it keeps the warning out
        # of the suite's output.
        with self.assertLogs("ServerToolsCore.digest", "WARNING"):
            self.assertEqual(digest.digest_tree(self.root), {})


class ChangedSinceTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self._write("p1.json", "one")
        self._write("p2.json", "two")
        self.baseline = digest.digest_tree(self.root)

    def _write(self, relative, data):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(data)
        return path

    def test_an_untouched_folder_has_changed_nothing(self):
        self.assertEqual(digest.changed_since(self.baseline, self.root), [])

    def test_only_the_file_whose_bytes_moved_is_reported(self):
        self._write("p1.json", "corrected")

        self.assertEqual(digest.changed_since(self.baseline, self.root),
                         ["p1.json"])

    def test_a_file_the_reader_added_is_reported(self):
        self._write("p1_adjust.tfm", "matrix")

        self.assertEqual(digest.changed_since(self.baseline, self.root),
                         ["p1_adjust.tfm"])

    def test_a_rewrite_with_the_same_bytes_is_not_a_change(self):
        """The reviewer saves on the way OUT of a patient, not only when
        something moved, so mtime moves for files nobody touched. Stat-ing
        would send the cohort back; this is the reason for hashing."""
        self._write("p1.json", "one")
        os.utime(os.path.join(self.root, "p1.json"), (0, 0))

        self.assertEqual(digest.changed_since(self.baseline, self.root), [])

    def test_a_file_that_was_deleted_is_not_reported(self):
        """Deletions are deliberately not honoured: the server lays a
        correction over the step's output, so an absent file reads as
        unchanged and a partial set cannot be told from a deletion."""
        os.remove(os.path.join(self.root, "p2.json"))

        self.assertEqual(digest.changed_since(self.baseline, self.root), [])

    def test_the_answer_is_sorted_so_one_review_builds_one_archive(self):
        self._write("z.json", "new")
        self._write("a.json", "new")

        self.assertEqual(digest.changed_since(self.baseline, self.root),
                         ["a.json", "z.json"])

    def test_an_empty_baseline_makes_everything_new(self):
        """What a checkpoint that failed to unpack leaves behind. Sending too
        much is the safe direction; sending a file the reader corrected under
        a stale baseline's name is not."""
        self.assertEqual(digest.changed_since({}, self.root),
                         ["p1.json", "p2.json"])


class PathsUnderTest(unittest.TestCase):

    def test_a_subfolder_strips_its_own_name_from_the_paths(self):
        found = digest.paths_under(
            [os.path.join("01_ALI", "p1.json"), os.path.join("02_ASO", "p1.nii.gz")],
            "/root", "/root/01_ALI")

        self.assertEqual(found, ["p1.json"])

    def test_a_nested_path_keeps_the_rest_of_its_tree(self):
        """A step mirrors its input tree, so a patient's correction can be two
        directories down; flattening it would unpack into the wrong folder."""
        found = digest.paths_under(
            [os.path.join("01_ALI", "p1", "lm.json")], "/root", "/root/01_ALI")

        self.assertEqual(found, [os.path.join("p1", "lm.json")])

    def test_the_folder_itself_is_the_flattened_single_step_and_strips_nothing(self):
        found = digest.paths_under(["p1.json"], "/root", "/root")

        self.assertEqual(found, ["p1.json"])

    def test_a_file_at_the_root_belongs_to_no_named_step(self):
        found = digest.paths_under(["stray.json"], "/root", "/root/01_ALI")

        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
