"""An argument that draws from one subfolder, not from the tool's whole catalogue.

A tool can serve several modalities -- AREG registers CBCT volumes, intraoral
surfaces, and one onto the other -- and the hosted-file list is published per
TOOL. Flat, every argument was offered every cohort: the CBCT baseline picker
listed the intraoral meshes, and picking one sends a folder of `.vtk` where a
volume is expected.

The server now publishes a subfolder on the argument and a list per subfolder
beside the flat one. What this pins is the client half: the scoped argument
reads its own list, an unscoped one reads the flat list exactly as before, and
a scope the server did not send falls back rather than emptying the picker.
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

DATA = {
    "testfiles": ["CBCT", "IOS"],
    "scoped": {
        "CBCT": {"testfiles": ["CBCT_FullyAuto", "CBCT_SemiAuto"], "models": []},
        "IOS": {"testfiles": ["IOS_test_scans"], "models": []},
    },
}


class _Combo:
    def __init__(self):
        self.items = []
        self.currentText = ""

    def clear(self):
        self.items = []

    def addItems(self, items):
        self.items.extend(items)

    def setCurrentIndex(self, index):
        self.currentText = self.items[index]


class ScopedHostedFilesTest(unittest.TestCase):
    def _panel(self, spec):
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel._schema = {"arguments": {"t1": dict(spec, required=False)}}
        panel._inputWidgets = {}
        panel._argWidgets = {"t1": _Combo()}
        return panel

    def test_a_scoped_argument_is_offered_only_its_own_folder(self):
        panel = self._panel({"server_selectable": "testfile", "selectable_scope": "CBCT"})

        panel._fillServerSelectable("t1", "testfile", DATA)

        assert "IOS_test_scans" not in panel._argWidgets["t1"].items
        assert "CBCT_FullyAuto" in panel._argWidgets["t1"].items

    def test_an_unscoped_argument_reads_the_flat_list(self):
        """Every deployment that scopes nothing, which is all of them until one
        says otherwise."""
        panel = self._panel({"server_selectable": "testfile"})

        panel._fillServerSelectable("t1", "testfile", DATA)

        assert "CBCT" in panel._argWidgets["t1"].items

    def test_a_scope_the_server_did_not_send_falls_back(self):
        """A panel built against a newer schema than the server it is talking
        to must not show an empty picker with no explanation."""
        panel = self._panel({"server_selectable": "testfile", "selectable_scope": "MRI"})

        panel._fillServerSelectable("t1", "testfile", DATA)

        assert "CBCT" in panel._argWidgets["t1"].items


class ScopedDownloadTest(unittest.TestCase):
    """The name a scoped picker listed must be downloadable from that scope.

    The run path learned this first: a hosted file is resolved through the
    ARGUMENT it was sent for, so the server knows which subfolder to look in.
    The DOWNLOAD path did not, and the server answered "No such testfile
    'IOSCBCT_TestFile' for tool 'AREG'" for an entry its own picker had just
    offered -- a list the user could read and not use.
    """

    def test_the_scope_travels_with_the_download(self):
        import ServerToolsCoreLib.client as client_module

        captured = {}

        def _probe(session, url, **kwargs):
            # `probe_ranged(session, url, ...)`: the URL is the SECOND argument.
            captured["url"] = url
            raise RuntimeError("stop here: the URL is the subject")

        original = client_module.transfer.probe_ranged
        client_module.transfer.probe_ranged = _probe
        try:
            client = client_module.ToolServerClient.__new__(client_module.ToolServerClient)
            client._server_url = "https://example.org"
            client._token = "t"
            client._session = object()
            try:
                client.download_testfile("AREG", "IOSCBCT_TestFile", "/tmp/x", None, "IOSCBCT")
            except Exception:
                pass
        finally:
            client_module.transfer.probe_ranged = original

        assert "scope=IOSCBCT" in captured.get("url", ""), captured


class ScopedCacheTest(unittest.TestCase):
    """Two scopes may offer the SAME entry name, and they are not the same file.

    `IOSCBCT_TestFile` is the intraoral surfaces under one scope and the CBCT
    volume under another. Cached by name alone, picking it for the second
    argument handed back the first one's download -- meshes in the CBCT field,
    from a cache hit that looked perfectly healthy.
    """

    def _panel(self):
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel._schema = {"arguments": {
            "ios": {"selectable_scope": "IOSCBCT_IOS"},
            "cbct": {"selectable_scope": "IOSCBCT_CBCT"},
            "plain": {},
        }}
        return panel

    def test_the_same_name_under_two_scopes_is_two_cache_entries(self):
        panel = self._panel()

        assert panel._selectableScope("ios") == "IOSCBCT_IOS"
        assert panel._selectableScope("cbct") == "IOSCBCT_CBCT"
        assert panel._selectableScope("ios") != panel._selectableScope("cbct")

    def test_an_unscoped_argument_has_no_scope(self):
        """Every deployment that scopes nothing, where the name alone was
        always enough."""
        assert self._panel()._selectableScope("plain") == ""


if __name__ == "__main__":
    unittest.main()
