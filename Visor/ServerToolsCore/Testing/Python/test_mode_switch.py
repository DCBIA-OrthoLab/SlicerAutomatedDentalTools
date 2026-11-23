"""The bar that links a classic panel to its VISOR twin, minus Qt.

What is pinned here is what the bar says, what it remembers and what it asks
Slicer to open. The widget itself, its injection into a classic panel and its
survival of a Reload need Slicer: see Testing/Slicer/mode_switch_check.py.
"""

import os
import sys
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _CORE)

from ServerToolsCoreLib import mode_switch as ms  # noqa: E402


class FakeSettings:
    """The part of QSettings the bar uses."""

    def __init__(self, **initial):
        self.data = dict(initial)

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):
        self.data[key] = value


class BrokenSettings:
    def value(self, key, default=None):
        raise RuntimeError("unreadable")

    def setValue(self, key, value):
        raise RuntimeError("read only")


class RememberedModeTest(unittest.TestCase):
    def test_the_default_is_classic(self):
        self.assertEqual(ms.DEFAULT_MODE, "classic")
        self.assertEqual(ms.get_mode("ALI", FakeSettings()), "classic")

    def test_the_key_is_per_module(self):
        self.assertEqual(ms.settings_key("ALI"), "AutomatedDentalTools/Version/ALI")
        settings = FakeSettings()
        ms.set_mode("ALI", "visor", settings)
        self.assertEqual(settings.data, {"AutomatedDentalTools/Version/ALI": "visor"})
        self.assertEqual(ms.get_mode("ALI", settings), "visor")
        self.assertEqual(ms.get_mode("AREG", settings), "classic", "another module is not affected")

    def test_a_value_nobody_wrote_is_the_default(self):
        for junk in ("", "VISOR", "remote", None, 3):
            settings = FakeSettings(**{"AutomatedDentalTools/Version/ALI": junk})
            self.assertEqual(ms.get_mode("ALI", settings), "classic", junk)

    def test_unreadable_settings_give_the_default(self):
        self.assertEqual(ms.get_mode("ALI", BrokenSettings()), "classic")

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            ms.set_mode("ALI", "remote", FakeSettings())


class SwitchTest(unittest.TestCase):
    def test_classic_to_visor_writes_then_opens_the_twin(self):
        settings, opened = FakeSettings(), []
        order = []
        original = settings.setValue
        settings.setValue = lambda k, v: (order.append("write"), original(k, v))
        self.assertTrue(ms.switch("classic", "ALI", lambda n: (order.append("open"), opened.append(n)), settings))
        self.assertEqual(opened, ["ALIVisor"])
        self.assertEqual(ms.get_mode("ALI", settings), "visor")
        self.assertEqual(order, ["write", "open"], "the choice is written before the module is selected")

    def test_visor_to_classic(self):
        settings, opened = FakeSettings(**{"AutomatedDentalTools/Version/AREG": "visor"}), []
        self.assertTrue(ms.switch("visor", "AREG", opened.append, settings))
        self.assertEqual(opened, ["AREG"])
        self.assertEqual(ms.get_mode("AREG", settings), "classic")

    def test_a_module_that_cannot_be_selected_leaves_the_choice_alone(self):
        settings = FakeSettings()

        def refuse(name):
            raise RuntimeError("no such module")

        self.assertFalse(ms.switch("classic", "ALI", refuse, settings))
        self.assertEqual(ms.get_mode("ALI", settings), "classic")
        settings = FakeSettings(**{"AutomatedDentalTools/Version/ALI": "visor"})
        self.assertFalse(ms.switch("visor", "ALI", refuse, settings))
        self.assertEqual(ms.get_mode("ALI", settings), "visor")

    def test_settings_that_cannot_be_written_do_not_stop_the_switch(self):
        opened = []
        self.assertTrue(ms.switch("classic", "ALI", opened.append, BrokenSettings()))
        self.assertEqual(opened, ["ALIVisor"])


class BarStateTest(unittest.TestCase):
    def test_the_classic_bar_offers_the_visor_version(self):
        state = ms.bar_state("classic", "ALI")
        self.assertEqual(state.button, "Passer à la version VISOR locale")
        self.assertTrue(state.enabled)
        self.assertIn("classique", state.heading)
        self.assertTrue(state.explanation)

    def test_the_visor_bar_offers_the_classic_version(self):
        state = ms.bar_state("visor", "ALI")
        self.assertEqual(state.button, "Revenir à la version classique")
        self.assertTrue(state.enabled)
        self.assertIn("VISOR", state.heading)

    def test_a_running_treatment_greys_the_bar_and_says_why(self):
        for side in ("classic", "visor"):
            state = ms.bar_state(side, "ALI", busy_reason="2 traitement(s) lancé(s) depuis ce panneau")
            self.assertFalse(state.enabled, side)
            self.assertIn("ALI", state.explanation)
            self.assertIn("2 traitement(s)", state.explanation)
            self.assertTrue(state.tooltip)

    def test_a_missing_twin_greys_the_bar_and_says_so(self):
        state = ms.bar_state("classic", "ALI", target_loaded=False)
        self.assertFalse(state.enabled)
        self.assertIn("pas chargée", state.explanation)

    def test_busy_wins_over_a_missing_twin(self):
        self.assertIn("en cours", ms.bar_state("visor", "ALI", "x", target_loaded=False).explanation)

    def test_an_unknown_side_is_refused(self):
        with self.assertRaises(ValueError):
            ms.bar_state("remote", "ALI")

    def test_the_target_of_each_side(self):
        self.assertEqual(ms.target_module("classic", "FlexReg"), "FlexRegVisor")
        self.assertEqual(ms.target_module("visor", "FlexReg"), "FlexReg")


class BusyTest(unittest.TestCase):
    def test_a_visor_panel_with_no_run_is_idle(self):
        class Widget:
            _runs = []

        self.assertEqual(ms.visor_busy_reason(Widget()), "")
        self.assertEqual(ms.visor_busy_reason(object()), "")

    def test_a_visor_panel_with_a_run_is_busy(self):
        class Widget:
            _runs = [object(), object()]

        self.assertIn("2", ms.visor_busy_reason(Widget()))

    def test_classic_busy_outside_slicer_is_idle(self):
        self.assertEqual(ms.classic_busy_reason("ALI"), "")
        self.assertEqual(ms.classic_busy_reason("CLIC"), "", "no command line module to look at")


class TablesTest(unittest.TestCase):
    def test_greedyreg_has_no_bar_and_says_why(self):
        self.assertFalse(ms.has_bar("GreedyReg"))
        self.assertIn("GreedyReg", ms.NO_PARITY)
        self.assertTrue(ms.NO_PARITY["GreedyReg"])

    def test_modules_without_a_twin_have_no_bar(self):
        for name in ms.CLASSIC_ONLY + ms.VISOR_ONLY:
            self.assertFalse(ms.has_bar(name), name)

    def test_the_tables_do_not_overlap(self):
        groups = [set(ms.EQUIVALENTS), set(ms.NO_PARITY), set(ms.CLASSIC_ONLY), set(ms.VISOR_ONLY)]
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                self.assertEqual(a & b, set())

    def test_names_round_trip(self):
        for classic in ms.EQUIVALENTS:
            self.assertEqual(ms.classic_name(ms.visor_name(classic)), classic)
        self.assertIsNone(ms.classic_name("ALI"))
        self.assertIsNone(ms.classic_name("VISU"))

    def test_the_tool_names_of_sadt_visor_are_not_the_module_names(self):
        """TOOL_NAME stays `ALI` on the panel called `ALIVisor`."""
        import re

        path = os.path.join(_CORE, "..", "ALIVisor", "ALIVisor.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertRegex(text, r'TOOL_NAME = "ALI"')
        self.assertRegex(text, r"class ALIVisorWidget\(")
        self.assertNotRegex(text, re.compile(r'TOOL_NAME = "ALIVisor"'))


if __name__ == "__main__":
    unittest.main()
