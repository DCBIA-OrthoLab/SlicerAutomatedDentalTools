"""A panel built from yesterday's schema is a panel that lies about the tool.

The schema is read once, at `setup()`. Nothing re-read it afterwards: `enter()`
refreshes the hosted FILES -- a model dropped into DATA/ -- but not the shape
of the form. So a field the server started publishing while Slicer was open is
simply absent, and a field it stopped publishing is still there and still
clickable. That is indistinguishable, from the panel, from a server that never
changed; it cost an afternoon of "the checkbox is not there" against a server
that had been publishing it for twenty minutes.

Rebuilding on EVERY enter() would be the wrong cure: a rebuild throws the form
away, and a path somebody typed and has not run yet must survive a trip to
another module. So the rule these tests pin is: re-read always, rebuild only on
a difference.
"""

import os
import sys
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: E402,F401 - installs the stubs

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase  # noqa: E402
from ServerToolsCoreLib.errors import ServerToolError  # noqa: E402

A = {"arguments": {"t1": {"type": "path"}}}
B = {"arguments": {"t1": {"type": "path"}, "segmentations": {"type": "multichoice"}}}


class _Client:
    def __init__(self, schema, error=None):
        self.schema = schema
        self.error = error
        self.reads = 0

    def get_tool_schema(self, name, force_refresh=False):
        self.reads += 1
        if self.error:
            raise self.error
        return self.schema


class SchemaRefreshTest(unittest.TestCase):
    def _panel(self, client, schema=A):
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel.TOOL_NAME = "AREG_CBCT"
        panel.client = client
        panel._schema = schema
        panel.rebuilds = []
        panel._buildForm = lambda force_refresh=False: panel.rebuilds.append(force_refresh)
        return panel

    def test_a_schema_that_changed_rebuilds_the_form(self):
        panel = self._panel(_Client(B))

        panel._refreshSchema()

        self.assertEqual(panel.rebuilds, [True],
                         "the form must be rebuilt from the server's schema")

    def test_an_unchanged_schema_leaves_the_form_alone(self):
        """The common case, and the one that must cost nothing: a rebuild would
        discard whatever the user has already filled in."""
        panel = self._panel(_Client(A))

        panel._refreshSchema()

        self.assertEqual(panel.rebuilds, [])

    def test_a_server_that_cannot_be_reached_leaves_the_panel_usable(self):
        """The panel already works; a server that went away between two visits
        must not empty it. Same rule the hosted-file refresh follows."""
        panel = self._panel(_Client(None, error=ServerToolError("down")))

        panel._refreshSchema()

        self.assertEqual(panel.rebuilds, [])

    def test_an_unreachable_server_is_reported_once_per_visit(self):
        """The schema re-read and the hosted-file refresh both reach the same
        server on enter(). When the first could not, the second would only log
        the same outage again -- two red lines per module, per visit."""
        panel = self._panel(_Client(None, error=ServerToolError("down")))
        self.assertIs(panel._refreshSchema(), False)

        asked = []
        panel.uiWidget = None
        panel._collapsePending = False
        panel._sweepLeftoverTestFiles = lambda: None
        panel._refreshServerSelectables = lambda: asked.append("files")
        panel._refreshSceneVolumes = lambda: None
        panel._refreshServerStatus = lambda: asked.append("status")

        panel.enter()

        self.assertEqual(asked, ["status"],
                         "the hosted files must not be asked of a server already found down")

    def test_a_reachable_server_still_has_its_files_refreshed(self):
        panel = self._panel(_Client(A))
        self.assertIs(panel._refreshSchema(), True)

    def test_a_panel_that_never_built_is_left_to_the_build_path(self):
        """`_schema` is None while the panel could not be built at all. That
        case has its own retry; re-entering must not race with it."""
        panel = self._panel(_Client(B), schema=None)

        panel._refreshSchema()

        self.assertEqual(panel.rebuilds, [])
        self.assertEqual(panel.client.reads, 0, "nothing to compare against")


if __name__ == "__main__":
    unittest.main()
