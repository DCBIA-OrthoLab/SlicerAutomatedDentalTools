# What the panel offers is what the SADT-VISOR tool offers (tools/TMJ_Crop,
# commit 11f3cb8): the same joints, the same extra outputs under the same names,
# the same defaults, and no scale_by or scale on the screen.
import os
import re
import sys
import unittest
import xml.etree.ElementTree as ElementTree

_HERE = os.path.dirname(os.path.abspath(__file__))
_UI = os.path.normpath(os.path.join(_HERE, "..", ".."))
_CLI = os.path.normpath(os.path.join(_UI, "..", "TMJ_Crop_CLI"))
sys.path.insert(0, _CLI)

from TMJ_Crop_utils import catalogs  # noqa: E402


def read(*parts):
    with open(os.path.join(*parts)) as handle:
        return handle.read()


def ui_items(widget_name):
    root = ElementTree.parse(os.path.join(_UI, "Resources", "UI", "TMJ_Crop.ui")).getroot()
    widget = next(w for w in root.iter("widget") if w.get("name") == widget_name)
    return [item.find("property/string").text for item in widget.findall("item")]


class OptionsTest(unittest.TestCase):

    def test_the_joints_are_the_ones_the_visor_tool_offers_and_the_default_comes_first(self):
        self.assertEqual(ui_items("comboSides"), ["Left and Right", "Left", "Right"])
        self.assertEqual(list(catalogs.SIDES), ["Left and Right", "Left", "Right"])

    def test_the_extra_outputs_are_the_two_the_visor_tool_offers(self):
        self.assertEqual(catalogs.OUTPUT_ORIENTED_SCAN, "Oriented scan")
        self.assertEqual(catalogs.OUTPUT_MANDIBLE, "Mandible segmentation")
        self.assertEqual(sorted(catalogs.OUTPUT_HELP),
                         ["Mandible segmentation", "Oriented scan"])
        self.assertEqual(catalogs.OUTPUT_LABEL, "Extra outputs")

    def test_nothing_is_ticked_by_default(self):
        ui = read(_UI, "Resources", "UI", "TMJ_Crop.ui")
        self.assertNotIn('name="checked"', ui)

    def test_the_panel_has_no_scale_option(self):
        ui = read(_UI, "Resources", "UI", "TMJ_Crop.ui")
        self.assertNotRegex(ui, r"(?i)scale")

    def test_the_cli_takes_scale_by_and_scale_for_research_use(self):
        parameters = [n.text for n in ElementTree.parse(
            os.path.join(_CLI, "TMJ_Crop_CLI.xml")).getroot().iter("name")]
        for name in ("sides", "outputs", "scale_by", "scale"):
            self.assertIn(name, parameters)
        self.assertEqual(catalogs.SCALE_BY, ("none", "mandible-volume"))

    def test_the_cli_flags_are_the_ones_the_xml_declares(self):
        source = read(_CLI, "TMJ_Crop_CLI.py")
        flags = set(re.findall(r'add_argument\("--(\w+)"', source))
        root = ElementTree.parse(os.path.join(_CLI, "TMJ_Crop_CLI.xml")).getroot()
        declared = {e.text for e in root.iter("longflag")}
        self.assertEqual(flags, declared)

    def test_the_chain_orients_on_the_seven_landmarks_aso_defaults_to(self):
        self.assertEqual(catalogs.ORIENTATION_LANDMARKS,
                         ("Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"))


if __name__ == "__main__":
    unittest.main()
