# Invariants of the filter that keeps a marked face only when it can belong to
# the tooth the cameras were aimed at. Synthetic geometry, so the suite needs
# no scan, no model and no GPU.
#
# What it guards is invisible on screen: without it, a neighbour's mucogingival
# point marked in the same picture is averaged into this tooth's answer, and
# the two landmarks come out on the same spot -- with a high confidence, since
# both are on a real mucogingival point.
#
# Run with:  python -m unittest discover ALI_IOS/Testing/Python
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ALI_IOS_utils"))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import pick_patch  # noqa: E402


def Arch(pitch=1.0, labels=range(19, 32), radius=10.0):
    """Tooth centres laid on a circle, `pitch` apart along it.

    Returns the per-vertex region ids and vertex positions the pitch is read
    from, one vertex per tooth being enough for what is measured.
    """
    labels = list(labels)
    step = pitch / radius                      # chord ~ arc for a shallow step
    angles = np.arange(len(labels)) * step
    positions = np.column_stack([radius * np.cos(angles),
                                 radius * np.sin(angles),
                                 np.zeros(len(labels))])
    return np.array(labels), positions


class ToothPitchTest(unittest.TestCase):
    def test_evenly_spaced_teeth_give_that_spacing(self):
        ids, positions = Arch(pitch=1.0)
        self.assertAlmostEqual(pick_patch.ToothPitch(ids, positions), 1.0, places=2)

    def test_a_missing_tooth_does_not_double_the_pitch(self):
        """The median is what makes this hold, and it is why a median is used.

        A tooth absent from the segmentation leaves one double-width step. Read
        as a mean it would widen the bound for the whole arch, which is exactly
        where the neighbour's answer would then slip back in.
        """
        ids, positions = Arch(pitch=1.0)
        keep = ids != 24
        self.assertAlmostEqual(pick_patch.ToothPitch(ids[keep], positions[keep]),
                               1.0, places=2)

    def test_a_crowded_arch_gets_a_tighter_bound_than_a_spaced_one(self):
        crowded = pick_patch.ToothPitch(*Arch(pitch=0.6))
        spaced = pick_patch.ToothPitch(*Arch(pitch=1.4))
        self.assertLess(crowded, spaced)

    def test_too_few_teeth_to_measure_anything(self):
        ids, positions = Arch(labels=range(19, 21))
        self.assertIsNone(pick_patch.ToothPitch(ids, positions))

    def test_labels_that_are_not_teeth_are_ignored(self):
        """33 is the gingiva, and it is most of the mesh."""
        ids, positions = Arch(pitch=1.0)
        ids = np.concatenate([ids, [33] * 5])
        positions = np.vstack([positions, np.zeros((5, 3))])
        self.assertAlmostEqual(pick_patch.ToothPitch(ids, positions), 1.0, places=2)


class PickNearAimTest(unittest.TestCase):
    def setUp(self):
        # two spots two units apart: the aimed tooth's, and its neighbour's
        self.vertices = np.array([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0],   # at the aim
            [2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0],   # next tooth
        ])
        self.faces = np.array([[0, 1, 2], [3, 4, 5]])
        self.aim = np.array([0.0, 0.0, 0.0])

    def test_the_neighbours_spot_is_dropped(self):
        kept, off_aim = pick_patch.PickNearAim([0, 1], self.faces, self.vertices,
                                               self.aim, radius=1.0)
        self.assertEqual(list(kept), [0])
        self.assertFalse(off_aim)

    def test_a_single_spot_is_left_alone(self):
        kept, off_aim = pick_patch.PickNearAim([0], self.faces, self.vertices,
                                               self.aim, radius=1.0)
        self.assertEqual(list(kept), [0])
        self.assertFalse(off_aim)

    def test_nothing_within_reach_is_reported_not_emptied(self):
        """The point found is the neighbour's, and nothing can recover this one.

        Returning no point would be worse than returning a wrong one silently
        is: what fixes it is saying so, so the tools downstream leave it out
        and rebuild it from its neighbours.
        """
        kept, off_aim = pick_patch.PickNearAim([0, 1], self.faces, self.vertices,
                                               np.array([50.0, 0.0, 0.0]), radius=1.0)
        self.assertEqual(list(kept), [0, 1])
        self.assertTrue(off_aim)

    def test_no_pitch_measured_means_no_filtering(self):
        kept, off_aim = pick_patch.PickNearAim([0, 1], self.faces, self.vertices,
                                               self.aim, radius=None)
        self.assertEqual(list(kept), [0, 1])
        self.assertFalse(off_aim)

    def test_a_wide_bound_keeps_both(self):
        kept, off_aim = pick_patch.PickNearAim([0, 1], self.faces, self.vertices,
                                               self.aim, radius=10.0)
        self.assertEqual(list(kept), [0, 1])
        self.assertFalse(off_aim)

    def test_no_marked_face_is_not_an_error(self):
        kept, off_aim = pick_patch.PickNearAim([], self.faces, self.vertices,
                                               self.aim, radius=1.0)
        self.assertEqual(list(kept), [])
        self.assertFalse(off_aim)

    def test_the_note_is_the_one_the_other_tools_read(self):
        """AREG, FlexReg and FillGaps all match on this text."""
        self.assertEqual(pick_patch.OFF_AIM_NOTE, "off the aim")


class ResolveCollisionsTest(unittest.TestCase):
    """Two mucogingival landmarks cannot share a spot: they are a tooth apart."""

    def Line(self, spacing=5.0):
        names = pick_patch.MGL_ORDER
        return {name: {"x": float(i * spacing), "y": 0.0, "z": 0.0}
                for i, name in enumerate(names)}

    def Aims(self, spacing=5.0):
        return {name: np.array([i * spacing, 0.0, 0.0])
                for i, name in enumerate(pick_patch.MGL_ORDER)}

    def test_a_well_spread_line_is_left_alone(self):
        line, aims = self.Line(), self.Aims()
        self.assertEqual(pick_patch.ResolveCollisions(line, aims, 5.0, note="off the aim"), [])
        self.assertFalse(any("desc" in entry for entry in line.values()))

    def test_the_one_further_from_its_own_aim_is_marked(self):
        line, aims = self.Line(), self.Aims()
        # LL5MG dragged onto LL6MG's point: LL6MG's aim is the one it sits on
        line["LL5MG"]["x"] = line["LL6MG"]["x"]
        marked = pick_patch.ResolveCollisions(line, aims, 5.0, note="off the aim")
        self.assertEqual(marked, ["LL5MG"])
        self.assertIn("off the aim", line["LL5MG"]["desc"])
        self.assertNotIn("desc", line["LL6MG"])

    def test_the_verdict_follows_the_aims_not_the_order(self):
        """Reverse which one owns the spot and the other name is marked."""
        line, aims = self.Line(), self.Aims()
        line["LL6MG"]["x"] = line["LL5MG"]["x"]
        marked = pick_patch.ResolveCollisions(line, aims, 5.0, note="off the aim")
        self.assertEqual(marked, ["LL6MG"])

    def test_an_existing_note_is_kept(self):
        line, aims = self.Line(), self.Aims()
        line["LL5MG"]["x"] = line["LL6MG"]["x"]
        line["LL5MG"]["desc"] = "confidence 0.900"
        pick_patch.ResolveCollisions(line, aims, 5.0, note="off the aim")
        self.assertIn("confidence 0.900", line["LL5MG"]["desc"])
        self.assertIn("off the aim", line["LL5MG"]["desc"])

    def test_only_one_of_a_pair_is_ever_marked(self):
        """Marking both would leave the line with a hole and no owner."""
        line, aims = self.Line(), self.Aims()
        line["LL5MG"]["x"] = line["LL6MG"]["x"]
        line["LL4MG"]["x"] = line["LL6MG"]["x"]
        marked = pick_patch.ResolveCollisions(line, aims, 5.0, note="off the aim")
        self.assertEqual(len(marked), len(set(marked)))
        self.assertNotIn("LL6MG", marked)

    def test_no_pitch_means_no_verdict(self):
        line, aims = self.Line(), self.Aims()
        line["LL5MG"]["x"] = line["LL6MG"]["x"]
        self.assertEqual(pick_patch.ResolveCollisions(line, aims, None, note="off the aim"), [])

    def test_a_landmark_with_no_recorded_aim_is_left_alone(self):
        line, aims = self.Line(), self.Aims()
        line["LL5MG"]["x"] = line["LL6MG"]["x"]
        del aims["LL5MG"]
        self.assertEqual(pick_patch.ResolveCollisions(line, aims, 5.0, note="off the aim"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
