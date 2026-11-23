"""Server text is shown as text: message boxes, labels and tooltips never
render what the server sent as HTML, and a server message is capped."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import qt_stubs  # noqa: E402

qt, ctk = qt_stubs.install()

from ServerToolsCoreLib import design, formgen  # noqa: E402
from ServerToolsCoreLib.client import _SERVER_MESSAGE_MAX_LEN, ToolServerClient  # noqa: E402
from ServerToolsCoreLib.errors import plain_text  # noqa: E402

_LINK = "<a href='https://evil.example'>click</a>"


class PlainTextTest(unittest.TestCase):
    def test_text_without_markup_is_untouched(self):
        for text in ("Argument 'x' is required.", "A & B", "line 1\nline 2", ""):
            self.assertEqual(plain_text(text), text)
        self.assertEqual(plain_text(None), "")

    def test_markup_is_escaped_into_inert_rich_text(self):
        shown = plain_text(_LINK + "\nsecond line")
        self.assertTrue(shown.startswith("<qt>"))
        self.assertNotIn("<a ", shown)
        self.assertIn("&lt;a href=&#x27;https://evil.example&#x27;&gt;click&lt;/a&gt;", shown)
        self.assertIn("<br>second line", shown)


class SchemaTextTest(unittest.TestCase):
    def test_labels_are_plain_text(self):
        for factory in (design.section_title, design.required_label, design.optional_label,
                        design.group_heading, design.hint_label, design.warning_label):
            with self.subTest(factory=factory.__name__):
                self.assertEqual(factory(_LINK).textFormat, qt.Qt.PlainText)

    def test_a_description_with_markup_is_an_escaped_tooltip(self):
        layout = qt.QFormLayout()
        widgets = formgen.build(
            {"mode": {"type": "str", "required": False, "description": _LINK}}, layout)
        label = dict((field, label) for label, field in layout.rows)[
            formgen.row_widget(widgets["mode"])]
        for tooltip in (widgets["mode"].toolTip(), label.toolTip()):
            self.assertNotIn("<a ", tooltip)
            self.assertIn("&lt;a href=", tooltip)


class ServerMessageCapTest(unittest.TestCase):
    def _response(self, payload):
        response = mock.Mock()
        response.json.return_value = payload
        response.text = json.dumps(payload)
        return response

    def test_a_json_detail_is_capped(self):
        message = ToolServerClient._server_message(self._response({"detail": "x" * 10_000}))
        self.assertEqual(len(message), _SERVER_MESSAGE_MAX_LEN)

    def test_a_structured_detail_is_text_and_capped(self):
        detail = [{"loc": ["body", "x"], "msg": "y" * 10_000}]
        message = ToolServerClient._server_message(self._response({"detail": detail}))
        self.assertIsInstance(message, str)
        self.assertEqual(len(message), _SERVER_MESSAGE_MAX_LEN)


if __name__ == "__main__":
    unittest.main()
