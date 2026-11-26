# A phone number is found at the default threshold.
#
# Presidio scores every phone number 0.4, raised only when an English word such
# as "phone" or "mobile" stands by it. The slider starts at 0.5, so "Call
# 212-555-0198." came out untouched.
#
# At 0.4 the same recognizer also takes rows of decimals for phone numbers: in
# a sample, half the voxel spacings and four coordinate triples in ten. Those
# must not be erased from a clinical report.
#
# The analyzer is stood in for by one that scores as Presidio does and honours
# the threshold it is given.
import contextlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", ".."))
if os.path.join(_ROOT, "ADT") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "ADT"))

# By name: test_pip_requirements reads a ("NAME", "212-...") literal as a pin.
PHONE = "PHONE_NUMBER"


@contextlib.contextmanager
def _stand_ins(modules):
    """Put `modules` in sys.modules, then put back those names and no other."""
    absent = object()
    before = {name: sys.modules.get(name, absent) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in before.items():
            if module is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_module():
    scripted = types.ModuleType("slicer.ScriptedLoadableModule")
    for suffix in ("", "Widget", "Logic", "Test"):
        name = "ScriptedLoadableModule" + suffix
        setattr(scripted, name, type(name, (), {}))
    with _stand_ins({"slicer": mock.MagicMock(), "slicer.ScriptedLoadableModule": scripted,
                     "qt": mock.MagicMock(), "ctk": mock.MagicMock()}):
        spec = importlib.util.spec_from_file_location(
            "anonymizer_phone_threshold",
            os.path.join(_ROOT, "Medical_Data_Anonymizer_Module",
                         "Medical_Data_Anonymizer_Module.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class Analyzer:
    """Reports `found` in the text, as (entity, substring, score)."""

    def __init__(self, found):
        self.found = found

    def analyze(self, text, entities, score_threshold, **_):
        return [types.SimpleNamespace(entity_type=entity, score=score,
                                      start=text.index(part),
                                      end=text.index(part) + len(part))
                for entity, part, score in self.found
                if entity in entities and score >= score_threshold]


class Anonymizer:
    def anonymize(self, text, analyzer_results, **_):
        for result in sorted(analyzer_results, key=lambda r: r.start, reverse=True):
            text = text[:result.start] + "<%s>" % result.entity_type + text[result.end:]
        return types.SimpleNamespace(text=text)


class PhoneThresholdTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logic = _load_module().Medical_Data_Anonymizer_ModuleLogic()

    def anonymize(self, text, found, threshold=0.5, entities=(PHONE, "PERSON")):
        return self.logic.anonymize_text_presidio(
            text, Analyzer(found), Anonymizer(), list(entities), "replace", threshold)

    def phone(self, text, part, score=0.4, threshold=0.5):
        return self.anonymize(text, [(PHONE, part, score)], threshold)

    def test_a_phone_number_is_found_without_a_context_word(self):
        for number in ("212-555-0198", "(415) 555-2671", "+36 30 123 4567", "212.555.0198",
                       "06 30 123 4567", "2125550198"):
            with self.subTest(number=number):
                self.assertEqual(self.phone("Reach %s." % number, number),
                                 "Reach <PHONE_NUMBER>.")

    def test_a_stricter_slider_does_not_lose_phone_numbers(self):
        self.assertEqual(self.phone("Call 212-555-0198", "212-555-0198", threshold=0.9),
                         "Call <PHONE_NUMBER>")

    def test_the_other_entities_still_follow_the_slider(self):
        self.assertEqual(self.anonymize("Seen by Molar", [("PERSON", "Molar", 0.45)]),
                         "Seen by Molar")

    def test_a_lower_slider_still_applies_to_phone_numbers(self):
        self.assertEqual(self.phone("x 555-0198", "555-0198", score=0.3, threshold=0.2),
                         "x <PHONE_NUMBER>")

    def test_phone_numbers_left_unchecked_are_left_alone(self):
        self.assertEqual(
            self.anonymize("Call 212-555-0198", [(PHONE, "212-555-0198", 0.4)],
                           entities=("PERSON",)),
            "Call 212-555-0198")

    def test_rows_of_measurements_are_not_phone_numbers(self):
        for text, part in (("Spacing 0.400000 0.400000", "0.400000 0.400000"),
                           ("Voxel 0.3 0.3 0.3.", "0.3 0.3 0.3"),
                           ("Point 12.345 -45.678 102.334", "12.345 -45.678 102.334"),
                           ("Point 12.345 -45.678 102.334", "345 -45.678"),
                           ("Widths 10.25\t11.50\t12.75", "10.25\t11.50")):
            with self.subTest(text=text, part=part):
                self.assertEqual(self.phone(text, part), text)

    def test_a_number_beside_a_measurement_is_still_found(self):
        self.assertEqual(self.phone("3.5 212-555-0198", "212-555-0198"), "3.5 <PHONE_NUMBER>")

    def test_what_the_slider_already_caught_is_never_released(self):
        # With a context word Presidio scores 0.75: the filter has no say there.
        self.assertEqual(self.phone("phone 20.79 46.09", "20.79 46.09", score=0.75),
                         "phone <PHONE_NUMBER>")


if __name__ == "__main__":
    unittest.main()
