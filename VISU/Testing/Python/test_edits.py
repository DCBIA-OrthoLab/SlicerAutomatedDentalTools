"""Writing a corrected landmark back, and the frame that makes it wrong.

    python3 -m unittest test_edits
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from VISULib import edits  # noqa: E402


def document(system="LPS"):
    """What ALI writes, down to the fields nothing here understands."""
    return {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/.../markups-schema.json",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": system,
            "locked": False,
            "labelFormat": "%N-%d",
            "controlPoints": [
                {"id": "1", "label": "Ba", "description": "predicted",
                 "position": [1.0, 2.0, 3.0], "locked": True, "visibility": True},
                {"id": "2", "label": "S", "description": "",
                 "position": [4.0, 5.0, 6.0], "locked": True, "visibility": True},
            ],
            "measurements": [],
            "display": {"visibility": True, "glyphScale": 2.0, "color": [0.4, 1, 1]},
        }],
    }


def written(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class FrameTest(unittest.TestCase):
    def test_a_scene_position_is_flipped_back_into_an_lps_file(self):
        # Slicer flips x and y reading an LPS file. Writing a scene position
        # straight back mirrors every point through the midsagittal plane --
        # which on a skull looks like a plausible landmark set.
        self.assertEqual(edits.to_file_frame([10, 20, 30], "LPS"), [-10.0, -20.0, 30.0])
        self.assertEqual(edits.to_file_frame([10, 20, 30], "RAS"), [10.0, 20.0, 30.0])

    def test_a_file_that_does_not_say_is_read_as_lps(self):
        self.assertEqual(edits.coordinate_system_of({"markups": [{}]}), "LPS")
        self.assertEqual(edits.to_file_frame([1, 2, 3], ""), [-1.0, -2.0, 3.0])

    def test_the_flip_is_its_own_inverse(self):
        there = edits.to_file_frame([1.5, -2.5, 3.5], "LPS")
        self.assertEqual(edits.to_file_frame(there, "LPS"), [1.5, -2.5, 3.5])


class SaveTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = os.path.join(self.folder.name, "p1_lm_Pred.mrk.json")
        self.original = document()
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.original, handle)

    def scene_position(self, label):
        """Where Slicer would hold the point this file writes."""
        for point in self.original["markups"][0]["controlPoints"]:
            if point["label"] == label:
                x, y, z = point["position"]
                return [-x, -y, z]
        raise KeyError(label)

    def test_nothing_moved_means_nothing_written(self):
        before = os.stat(self.path).st_mtime_ns
        moved = edits.save_markups(self.path, {
            "Ba": self.scene_position("Ba"), "S": self.scene_position("S"),
        })
        self.assertEqual(moved, 0)
        self.assertEqual(os.stat(self.path).st_mtime_ns, before,
                         "an unchanged file was rewritten")

    def test_one_dragged_point_is_written_and_only_that_one(self):
        moved_to = self.scene_position("Ba")
        moved_to[0] += 5.0
        count = edits.save_markups(self.path, {
            "Ba": moved_to, "S": self.scene_position("S"),
        })
        self.assertEqual(count, 1)
        after = written(self.path)["markups"][0]["controlPoints"]
        self.assertEqual(after[0]["position"], [-4.0, 2.0, 3.0])   # 1.0 - 5.0
        self.assertEqual(after[1]["position"], [4.0, 5.0, 6.0])

    def test_everything_the_tool_wrote_survives(self):
        moved_to = self.scene_position("Ba")
        moved_to[2] += 1.0
        edits.save_markups(self.path, {"Ba": moved_to})
        after = written(self.path)
        point = after["markups"][0]["controlPoints"][0]
        self.assertEqual(point["description"], "predicted")
        self.assertTrue(point["locked"], "the tool's own flag was rewritten")
        self.assertEqual(after["markups"][0]["display"]["color"], [0.4, 1, 1])
        self.assertEqual(after["@schema"], self.original["@schema"])
        self.assertEqual(after["markups"][0]["labelFormat"], "%N-%d")

    def test_points_are_matched_by_label_not_by_position_in_the_list(self):
        # A reader may have deleted one, and the second point of the file is
        # then not the second point of the node.
        moved_to = self.scene_position("S")
        moved_to[1] += 2.0
        edits.save_markups(self.path, {"S": moved_to})
        after = written(self.path)["markups"][0]["controlPoints"]
        self.assertEqual(after[0]["position"], [1.0, 2.0, 3.0])
        self.assertEqual(after[1]["position"], [4.0, 3.0, 6.0])   # 5.0 - 2.0

    def test_a_label_the_file_does_not_hold_is_ignored(self):
        count = edits.save_markups(self.path, {"Nasion": [9.0, 9.0, 9.0]})
        self.assertEqual(count, 0)
        self.assertEqual(written(self.path), self.original)

    def test_a_point_that_moved_by_a_rounding_bit_did_not_move(self):
        nudged = self.scene_position("Ba")
        nudged[0] += edits.UNMOVED_MM / 10
        self.assertEqual(edits.save_markups(self.path, {"Ba": nudged}), 0)

    def test_no_half_written_file_is_left_behind(self):
        moved_to = self.scene_position("Ba")
        moved_to[0] += 1.0
        edits.save_markups(self.path, {"Ba": moved_to})
        leftovers = [n for n in os.listdir(self.folder.name) if n.endswith(".visu-tmp")]
        self.assertEqual(leftovers, [])

    def test_a_round_trip_through_slicer_s_frame_changes_nothing(self):
        # Load, do not touch, save: the file must be byte-for-byte what it was.
        positions = {label: self.scene_position(label) for label in ("Ba", "S")}
        self.assertEqual(edits.save_markups(self.path, positions), 0)
        self.assertEqual(written(self.path), self.original)


if __name__ == "__main__":
    unittest.main()
