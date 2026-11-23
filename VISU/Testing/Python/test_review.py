"""The reviewer's list: what was marked, and whether it survived.

    python3 -m unittest test_review
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from VISULib import review  # noqa: E402


class ReviewTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = self.folder.name

    def test_a_folder_nobody_has_reviewed_starts_with_nothing_marked(self):
        self.assertEqual(review.load(self.root), set())

    def test_marks_survive_being_written_and_read_back(self):
        self.assertTrue(review.save(self.root, {"p2", "p1"}))
        self.assertEqual(review.load(self.root), {"p1", "p2"})

    def test_the_list_is_written_beside_the_data(self):
        review.save(self.root, {"p1"})
        self.assertTrue(os.path.exists(os.path.join(self.root, "visu-review.json")))

    def test_it_is_written_sorted_so_two_sessions_do_not_churn_the_file(self):
        review.save(self.root, {"c", "a", "b"})
        with open(review.path_for(self.root), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["flagged"], ["a", "b", "c"])

    def test_a_folder_that_will_not_take_it_says_so_rather_than_raising(self):
        # A hosted sample lands in a temp directory that the next download
        # deletes, and a cohort on a read-only share is perfectly normal.
        missing = os.path.join(self.root, "not-there")
        self.assertFalse(review.save(missing, {"p1"}))

    def test_no_half_written_file_is_left_behind(self):
        review.save(self.root, {"p1"})
        leftovers = [n for n in os.listdir(self.root) if n.endswith(".visu-tmp")]
        self.assertEqual(leftovers, [])

    def test_a_file_something_else_wrote_reads_as_nothing_marked(self):
        with open(review.path_for(self.root), "w", encoding="utf-8") as handle:
            handle.write("not json at all")
        self.assertEqual(review.load(self.root), set())

    def test_a_document_of_the_wrong_shape_reads_as_nothing_marked(self):
        with open(review.path_for(self.root), "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "flagged": "p1"}, handle)
        self.assertEqual(review.load(self.root), set())

    def test_clearing_every_mark_leaves_a_file_saying_so(self):
        review.save(self.root, {"p1"})
        review.save(self.root, set())
        self.assertEqual(review.load(self.root), set())

    def test_the_text_is_something_to_paste_into_a_ticket(self):
        self.assertEqual(review.as_text(set()), "Nothing flagged.")
        self.assertEqual(review.as_text({"p2", "p1"}, "/data/cohort"),
                         "2 flagged in /data/cohort:\n  p1\n  p2")


if __name__ == "__main__":
    unittest.main()
