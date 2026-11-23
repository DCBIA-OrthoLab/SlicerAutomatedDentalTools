"""Unit tests for ServerToolsCoreLib.joystick, run outside Slicer with `qt`/
`ctk`/`slicer` stubbed (see qt_stubs.py).

Painting is not exercised (there is no real Qt to paint with). What is tested
is everything the panel's correctness rests on: the mapping between values and
pixels, the clamping, and the gesture handlers' arithmetic: the same logic
FlexReg ships, minus the arch-specific mirrors.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_joystick.py
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import qt_stubs

qt, ctk = qt_stubs.install()

from ServerToolsCoreLib import joystick
from ServerToolsCoreLib.joystick import JoystickPad


class FakeMouseEvent:
    def __init__(self, x, y):
        self.pos = types.SimpleNamespace(x=float(x), y=float(y))


class FakeWheelEvent:
    def __init__(self, notches):
        self.angleDelta = types.SimpleNamespace(y=120.0 * notches)


class FakeKeyEvent:
    def __init__(self, key):
        self.key = key


def _pad(**kwargs):
    kwargs.setdefault("x_range", (-10.0, 10.0))
    kwargs.setdefault("y_range", (-10.0, 10.0))
    kwargs.setdefault("size", 128)
    return JoystickPad(**kwargs)


class ValueTest(unittest.TestCase):
    def test_opens_at_the_centre_of_both_axes(self):
        pad = _pad(x_range=(0.0, 10.0), y_range=(-5.0, 5.0))
        self.assertEqual((pad.value_x, pad.value_y), (5.0, 0.0))

    def test_set_values_clamps_to_the_ranges(self):
        pad = _pad()
        pad.setValues(99.0, -99.0)
        self.assertEqual((pad.value_x, pad.value_y), (10.0, -10.0))

    def test_an_inverted_range_still_clamps_both_ends(self):
        pad = _pad(x_range=(15.0, -15.0))
        pad.setValues(99.0, 0.0)
        self.assertEqual(pad.value_x, 15.0)
        pad.setValues(-99.0, 0.0)
        self.assertEqual(pad.value_x, -15.0)

    def test_notify_calls_on_changed_with_the_pad(self):
        pad = _pad()
        seen = []
        pad.onChanged = seen.append

        pad.setValues(1.0, 2.0, notify=True)

        self.assertEqual(seen, [pad])

    def test_an_unchanged_value_does_not_notify(self):
        pad = _pad()
        pad.setValues(1.0, 2.0)
        calls = []
        pad.onChanged = calls.append

        pad.setValues(1.0, 2.0, notify=True)

        self.assertEqual(calls, [])

    def test_default_step_is_a_hundredth_of_the_axis(self):
        pad = _pad(x_range=(-15.0, 15.0), y_range=(-5.0, 5.0))
        self.assertAlmostEqual(pad.x_step, 0.3)
        self.assertAlmostEqual(pad.y_step, 0.1)


class GeometryTest(unittest.TestCase):
    """The pixel/value mapping. size=128, no labels: box (0,0,128,128), knob
    area inset by KNOB+2=9 → (9,9,110,110)."""

    def test_no_labels_means_no_gutters(self):
        self.assertEqual(_pad()._box(), (0, 0, 128, 128))

    def test_labels_reserve_their_gutters(self):
        pad = _pad(x_labels=("R", "L"), y_labels=("POST", "ANT"))
        left, top, width, height = pad._box()
        self.assertEqual((left, top), (JoystickPad.SIDE_GUTTER, JoystickPad.GUTTER))
        self.assertEqual(width, 128 - 2 * JoystickPad.SIDE_GUTTER)
        self.assertEqual(height, 128 - 2 * JoystickPad.GUTTER)

    def test_the_area_corners_map_to_the_range_extremes(self):
        pad = _pad(x_range=(0.0, 1.0), y_range=(0.0, 1.0))
        left, top, width, height = pad._area()

        self.assertEqual(pad._valuesAt(left, top), (0.0, 1.0))                    # top-left
        self.assertEqual(pad._valuesAt(left + width, top + height), (1.0, 0.0))   # bottom-right

    def test_positions_outside_the_area_are_clamped(self):
        pad = _pad(x_range=(0.0, 1.0), y_range=(0.0, 1.0))
        self.assertEqual(pad._valuesAt(-999, 999), (0.0, 0.0))

    def test_knob_position_and_values_at_are_inverses(self):
        pad = _pad(x_range=(-15.0, 15.0), y_range=(-5.0, 5.0))
        pad.setValues(3.0, -2.0)

        x, y = pad._valuesAt(*pad._knobPosition())

        self.assertAlmostEqual(x, 3.0, places=6)
        self.assertAlmostEqual(y, -2.0, places=6)

    def test_an_inverted_axis_mirrors_the_knob(self):
        # x_range [15, -15]: 15 is the LEFT end, by construction of the
        # schema contract (index 0 = left/bottom).
        pad = _pad(x_range=(15.0, -15.0))
        left, _top, width, _height = pad._area()

        pad.setValues(15.0, 0.0)
        self.assertAlmostEqual(pad._knobPosition()[0], left)

        pad.setValues(-15.0, 0.0)
        self.assertAlmostEqual(pad._knobPosition()[0], left + width)


class MouseTest(unittest.TestCase):
    def test_an_absolute_press_jumps_the_knob_under_the_cursor(self):
        pad = _pad(x_range=(0.0, 1.0), y_range=(0.0, 1.0))
        seen = []
        pad.onChanged = lambda p: seen.append((p.value_x, p.value_y))
        left, top, width, height = pad._area()

        pad.mousePressEvent(FakeMouseEvent(left + width, top))

        self.assertEqual(seen, [(1.0, 1.0)])

    def test_a_release_without_spring_back_changes_nothing(self):
        pad = _pad()
        pad.setValues(3.0, 4.0)
        released = []
        pad.onReleased = released.append

        pad.mouseReleaseEvent(None)

        self.assertEqual((pad.value_x, pad.value_y), (3.0, 4.0))
        self.assertEqual(released, [])

    def test_a_spring_back_drag_is_relative_then_springs_home(self):
        pad = _pad(spring_back=True)
        released = []
        pad.onReleased = released.append
        centre_x, centre_y = pad._knobPosition()

        # Press anywhere: the knob holds (relative drag), no value change.
        pad.mousePressEvent(FakeMouseEvent(20, 20))
        self.assertEqual((pad.value_x, pad.value_y), (0.0, 0.0))

        # Move 11px right and 11px up from the press point: the knob moves by
        # the same delta from where it was, 11/110 of each axis span of 20.
        pad.mouseMoveEvent(FakeMouseEvent(31, 9))
        self.assertAlmostEqual(pad.value_x, 2.0)
        self.assertAlmostEqual(pad.value_y, 2.0)

        # Release: home, silently, and the gesture is reported as ended.
        pad.mouseReleaseEvent(None)
        self.assertEqual((pad.value_x, pad.value_y), (0.0, 0.0))
        self.assertEqual(released, [pad])
        self.assertAlmostEqual(pad._knobPosition()[0], centre_x)
        self.assertAlmostEqual(pad._knobPosition()[1], centre_y)


class KeyAndWheelTest(unittest.TestCase):
    def test_arrows_walk_one_step_each(self):
        pad = _pad()  # steps: 20/100 = 0.2

        pad.keyPressEvent(FakeKeyEvent(qt.Qt.Key_Right))
        pad.keyPressEvent(FakeKeyEvent(qt.Qt.Key_Up))
        pad.keyPressEvent(FakeKeyEvent(qt.Qt.Key_Up))

        self.assertAlmostEqual(pad.value_x, 0.2)
        self.assertAlmostEqual(pad.value_y, 0.4)

    def test_arrows_are_screen_directional_on_an_inverted_axis(self):
        # Right must always walk the knob right; on x_range [15, -15] that is
        # the numeric value going DOWN.
        pad = _pad(x_range=(15.0, -15.0))

        pad.keyPressEvent(FakeKeyEvent(qt.Qt.Key_Right))

        self.assertAlmostEqual(pad.value_x, -0.3)

    def test_the_wheel_walks_the_vertical_axis(self):
        pad = _pad()

        pad.wheelEvent(FakeWheelEvent(2))

        self.assertAlmostEqual(pad.value_y, 0.4)
        self.assertAlmostEqual(pad.value_x, 0.0)

    def test_shift_wheel_walks_the_horizontal_axis(self):
        pad = _pad()
        original = joystick._modifiers
        joystick._modifiers = lambda: qt.Qt.ShiftModifier
        self.addCleanup(setattr, joystick, "_modifiers", original)

        pad.wheelEvent(FakeWheelEvent(1))

        self.assertAlmostEqual(pad.value_x, 0.2)
        self.assertAlmostEqual(pad.value_y, 0.0)

    def test_on_wheel_takes_the_wheel_over_entirely(self):
        pad = _pad()
        seen = []
        pad.onWheel = lambda p, steps: seen.append(steps)

        pad.wheelEvent(FakeWheelEvent(3))

        self.assertEqual(seen, [3.0])
        self.assertEqual((pad.value_x, pad.value_y), (0.0, 0.0))

    def test_double_click_returns_to_the_defaults(self):
        pad = _pad()
        pad.setDefaults(1.0, -1.0)
        pad.setValues(5.0, 5.0)

        pad.mouseDoubleClickEvent(None)

        self.assertEqual((pad.value_x, pad.value_y), (1.0, -1.0))

    def test_a_spring_back_nudge_reports_then_springs_home(self):
        pad = _pad(spring_back=True)
        changes, released = [], []
        pad.onChanged = lambda p: changes.append((p.value_x, p.value_y))
        pad.onReleased = released.append

        pad.keyPressEvent(FakeKeyEvent(qt.Qt.Key_Up))

        # The nudge was dealt out (visible to onChanged), then sprung home.
        self.assertEqual(changes, [(0.0, 0.2)])
        self.assertEqual((pad.value_x, pad.value_y), (0.0, 0.0))
        self.assertEqual(released, [pad])


if __name__ == "__main__":
    unittest.main()
