"""Unit tests for ServerToolsCoreLib.client - run outside Slicer, requests mocked.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_client.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import requests

from ServerToolsCoreLib import transfer
from ServerToolsCoreLib.client import (
    ToolResult,
    ToolServerClient,
    _HEALTH_CHECK_TIMEOUT,
    _TOOLS_FETCH_TIMEOUT,
    accepts_folder,
    argument_types,
    file_extensions_for,
    is_file_type,
    testfile_entries,
)
from ServerToolsCoreLib.errors import ServerToolError

TOOLS_RESPONSE = [
    {
        "name": "example_tool",
        "arguments": {
            "label": {"type": "str", "required": True},
            "file": {"type": "file", "required": True},
        },
        "output_kind": "text",
    },
]


def _response(status_code=200, json_data=None, content=b"", headers=None, text=""):
    response = mock.Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.headers = headers or {"Content-Type": "application/json"}
    response.content = content
    response.text = text
    # The client downloads file results via iter_content (stream=True), never
    # via .content; a mock without it would hand the writer a Mock object.
    response.iter_content = lambda chunk_size: iter([content] if content else [])
    if json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.side_effect = ValueError("no json body")
    return response


def _zip_bytes(members=None) -> bytes:
    """A genuinely valid zip archive, for tests whose result filename ends in
    .zip: the client CRC-checks those before accepting them, so `b"zip-bytes"`
    placeholders no longer pass."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in (members or {"result.txt": "ok"}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


class ToolServerClientTest(unittest.TestCase):
    def setUp(self):
        self.client = ToolServerClient("https://example.org/", "secret-token")

    def test_server_url_is_normalized(self):
        self.assertEqual(self.client._server_url, "https://example.org")

    # -- configure() (live settings panel updates) -----------------------

    def test_configure_updates_fields(self):
        self.client.configure(server_url="https://other.org/", token="new-token", timeout=42)

        self.assertEqual(self.client.server_url, "https://other.org")
        self.assertEqual(self.client.token, "new-token")
        self.assertEqual(self.client.timeout, 42)

    def test_configure_partial_update_leaves_other_fields(self):
        self.client.configure(token="only-token-changes")

        self.assertEqual(self.client.server_url, "https://example.org")
        self.assertEqual(self.client.token, "only-token-changes")
        self.assertEqual(self.client.timeout, 600)

    @mock.patch("requests.Session.get")
    def test_configure_drops_cached_tools(self, mock_get):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)
        self.client.list_tools()
        self.assertEqual(mock_get.call_count, 1)

        self.client.configure(server_url="https://other.org")

        mock_get.return_value = _response(json_data=[])
        self.client.list_tools()
        self.assertEqual(mock_get.call_count, 2)  # re-fetched, not served from the old cache

    def test_properties_reflect_constructor_defaults(self):
        self.assertEqual(self.client.server_url, "https://example.org")
        self.assertEqual(self.client.token, "secret-token")
        self.assertEqual(self.client.timeout, 600)

    # -- list_tools / get_tool_schema ---------------------------------

    @mock.patch("requests.Session.get")
    def test_list_tools_caches(self, mock_get):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)

        tools = self.client.list_tools()
        tools_again = self.client.list_tools()

        self.assertEqual(mock_get.call_count, 1)
        self.assertIn("example_tool", tools)
        self.assertIs(tools, tools_again)

    @mock.patch("requests.Session.get")
    def test_list_tools_force_refresh(self, mock_get):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)

        self.client.list_tools()
        self.client.list_tools(force_refresh=True)

        self.assertEqual(mock_get.call_count, 2)

    @mock.patch("requests.Session.get")
    def test_get_tool_schema_unknown_tool_lists_available(self, mock_get):
        mock_get.return_value = _response(
            json_data=[
                {"name": "beta_tool", "arguments": {}, "output_kind": "text"},
                {"name": "alpha_tool", "arguments": {}, "output_kind": "text"},
            ]
        )

        with self.assertRaises(ServerToolError) as ctx:
            self.client.get_tool_schema("does_not_exist")

        self.assertIn("does_not_exist", str(ctx.exception))
        self.assertIn("alpha_tool, beta_tool", str(ctx.exception))

    @mock.patch("requests.Session.get")
    def test_get_tool_schema_can_bypass_the_cache(self, mock_get):
        # What a panel retrying after a failure needs: the cached list may be
        # the very reason the tool wasn't found.
        mock_get.return_value = _response(json_data=[])
        with self.assertRaises(ServerToolError):
            self.client.get_tool_schema("example_tool")

        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)

        self.assertIn("label", self.client.get_tool_schema("example_tool", force_refresh=True)["arguments"])
        self.assertEqual(mock_get.call_count, 2)

    @mock.patch("requests.Session.get")
    def test_get_tool_schema_uses_the_cache_by_default(self, mock_get):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)
        self.client.get_tool_schema("example_tool")
        self.client.get_tool_schema("example_tool")

        self.assertEqual(mock_get.call_count, 1)

    @mock.patch("requests.Session.get")
    def test_list_tools_uses_short_timeout(self, mock_get):
        # get_tool_schema() runs synchronously from a module's setup(); it must
        # not be able to freeze Slicer for up to self._timeout (600s).
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)
        self.client.list_tools()

        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["timeout"], _TOOLS_FETCH_TIMEOUT)
        self.assertLess(_TOOLS_FETCH_TIMEOUT, self.client._timeout)

    @mock.patch("requests.Session.get")
    def test_fetch_tools_network_error_wrapped(self, mock_get):
        mock_get.side_effect = requests.RequestException("boom")
        with self.assertRaises(ServerToolError):
            self.client.list_tools()

    # -- list_tool_data (server-side models/testfiles) -----------------

    @mock.patch("requests.Session.get")
    def test_list_tool_data_request_shape_and_result(self, mock_get):
        mock_get.return_value = _response(
            json_data={"models": ["stacking_v1.zip", "stacking_v2.zip"], "testfiles": ["demo.zip"]}
        )

        data = self.client.list_tool_data("SurgMovPred")

        self.assertEqual(
            data,
            {
                "models": ["stacking_v1.zip", "stacking_v2.zip"],
                "testfiles": ["demo.zip"],
                # {} rather than absent, so no caller has to ask which server
                # version it is talking to. Same for `scoped`, the per-subfolder
                # lists a deployment that scopes an argument sends.
                "entries": {},
                "scoped": {},
            },
        )
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "https://example.org/tools/SurgMovPred/data")
        # The endpoint is Bearer-protected, unlike /tools.
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-token")
        # Called synchronously from a module's setup(), like the schema fetch:
        # must use the short timeout, never the 600s tool-execution one.
        self.assertEqual(kwargs["timeout"], _TOOLS_FETCH_TIMEOUT)

    @mock.patch("requests.Session.get")
    def test_list_tool_data_tolerates_missing_keys(self, mock_get):
        mock_get.return_value = _response(json_data={})

        self.assertEqual(
            self.client.list_tool_data("SurgMovPred"),
            {"models": [], "testfiles": [], "entries": {}, "scoped": {}},
        )

    @mock.patch("requests.Session.get")
    def test_list_tool_data_passes_the_described_entries_through(self, mock_get):
        """The richer half of the payload: a name, what it is, and how big.
        Passed through verbatim -- shaping it is testfile_entries' job."""
        entries = {"models": [], "testfiles": [
            {"name": "CBCT_FullyAuto", "kind": "folder", "size": 355640000}
        ]}
        mock_get.return_value = _response(
            json_data={"models": [], "testfiles": ["CBCT_FullyAuto"], "entries": entries}
        )

        data = self.client.list_tool_data("AREG")

        self.assertEqual(data["entries"], entries)

    @mock.patch("requests.Session.get")
    def test_list_tool_data_network_error_wrapped(self, mock_get):
        mock_get.side_effect = requests.RequestException("boom")
        with self.assertRaises(ServerToolError):
            self.client.list_tool_data("SurgMovPred")

    @mock.patch("requests.Session.get")
    def test_list_tool_data_maps_error_status(self, mock_get):
        mock_get.return_value = _response(status_code=401, json_data={})
        with self.assertRaises(ServerToolError) as ctx:
            self.client.list_tool_data("SurgMovPred")
        self.assertEqual(ctx.exception.status_code, 401)

    # -- local validation ----------------------------------------------

    @mock.patch("requests.Session.get")
    def test_run_rejects_unexpected_argument_locally(self, mock_get):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)

        with self.assertRaises(ServerToolError):
            self.client.run("example_tool", args={"typo": "x"}, files={"file": __file__}, output_dir="/tmp")

    @mock.patch("requests.Session.get")
    def test_run_rejects_missing_required_argument_locally(self, mock_get):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)

        with self.assertRaises(ServerToolError):
            self.client.run("example_tool", args={}, files={"file": __file__}, output_dir="/tmp")

    @mock.patch("requests.Session.get")
    def test_run_rejects_missing_required_file_locally(self, mock_get):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)

        with self.assertRaises(ServerToolError):
            self.client.run("example_tool", args={"label": "x"}, files={}, output_dir="/tmp")

    @mock.patch("requests.Session.get")
    def test_run_allows_missing_optional_file_locally(self, mock_get):
        mock_get.return_value = _response(
            json_data=[
                {
                    "name": "optional_file_tool",
                    "arguments": {"attachment": {"type": "file", "required": False}},
                    "output_kind": "text",
                }
            ]
        )

        # Should pass local validation (no file argument required); network call
        # itself isn't exercised here (no requests.post mock), so a failure would
        # surface as a real connection attempt/error, not a ServerToolError.
        try:
            self.client._validate_against_schema(
                self.client.get_tool_schema("optional_file_tool"), {}, {}
            )
        except ServerToolError as exc:
            self.fail(f"Optional file argument should not be required: {exc}")

    @mock.patch("requests.Session.get")
    def test_run_rejects_file_when_tool_has_no_file_argument_locally(self, mock_get):
        mock_get.return_value = _response(
            json_data=[{"name": "no_file_tool", "arguments": {}, "output_kind": "text"}]
        )

        with self.assertRaises(ServerToolError):
            self.client.run("no_file_tool", args={}, files={"file": __file__}, output_dir="/tmp")

    @mock.patch("requests.Session.get")
    def test_run_rejects_file_argument_name_not_declared_as_file_locally(self, mock_get):
        # "label" exists in the schema but is type "str", not "file".
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)

        with self.assertRaises(ServerToolError):
            self.client.run("example_tool", args={}, files={"label": __file__}, output_dir="/tmp")

    # -- multi-file tools (e.g. real SurgMovPred: "model" + "input") ------

    @mock.patch("requests.Session.get")
    def test_run_rejects_when_only_one_of_two_required_files_given(self, mock_get):
        mock_get.return_value = _response(
            json_data=[
                {
                    "name": "SurgMovPred",
                    "arguments": {
                        "model": {"type": "zip_file", "required": True},
                        "input": {"type": "zip_file", "required": True},
                    },
                    "output_kind": "file",
                }
            ]
        )

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("SurgMovPred", args={}, files={"input": __file__}, output_dir="/tmp")

        self.assertIn("model", str(ctx.exception))

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_sends_each_file_under_its_own_argument_name(self, mock_get, mock_post):
        mock_get.return_value = _response(
            json_data=[
                {
                    "name": "SurgMovPred",
                    "arguments": {
                        "model": {"type": "zip_file", "required": True},
                        "input": {"type": "zip_file", "required": True},
                    },
                    "output_kind": "file",
                }
            ]
        )
        mock_post.return_value = _response(
            content=_zip_bytes(), headers={"Content-Type": "application/zip"}
        )

        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            self.client.run(
                "SurgMovPred", args={}, files={"model": __file__, "input": __file__}, output_dir=out_dir
            )

        _, kwargs = mock_post.call_args
        self.assertEqual(set(kwargs["files"].keys()), {"model", "input"})
        model_filename, model_handle = kwargs["files"]["model"]
        input_filename, input_handle = kwargs["files"]["input"]
        self.assertEqual(model_filename, os.path.basename(__file__))
        self.assertEqual(input_filename, os.path.basename(__file__))
        self.assertTrue(model_handle.closed)
        self.assertTrue(input_handle.closed)

    # -- request shape ----------------------------------------------------

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_sends_filename_with_upload(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)
        mock_post.return_value = _response(json_data={"result": "hello"})

        self.client.run("example_tool", args={"label": "x"}, files={"file": __file__}, output_dir="/tmp")

        _, kwargs = mock_post.call_args
        filename, file_handle = kwargs["files"]["file"]
        self.assertEqual(filename, os.path.basename(__file__))
        self.assertTrue(file_handle.closed)  # closed after the call

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_text_result_and_request_shape(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)
        mock_post.return_value = _response(json_data={"result": "hello"})

        result = self.client.run(
            "example_tool", args={"label": "x"}, files={"file": __file__}, output_dir="/tmp"
        )

        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.kind, "text")
        self.assertEqual(result.text, "hello")

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-token")
        self.assertEqual(kwargs["data"], {"label": "x"})

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_stringifies_bool(self, mock_get, mock_post):
        mock_get.return_value = _response(
            json_data=[
                {
                    "name": "flag_tool",
                    "arguments": {"flag": {"type": "bool", "required": True}},
                    "output_kind": "text",
                }
            ]
        )
        mock_post.return_value = _response(json_data={"result": "ok"})

        self.client.run("flag_tool", args={"flag": True}, output_dir="/tmp")

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["data"], {"flag": "true"})

    # -- error mapping ----------------------------------------------------

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_maps_422_to_server_message(self, mock_get, mock_post):
        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "text"}]
        )
        mock_post.return_value = _response(
            status_code=422, json_data={"detail": "Missing required argument 'x'"}
        )

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("no_args_tool", args={}, output_dir="/tmp")

        self.assertIn("Missing required argument", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 422)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_maps_401(self, mock_get, mock_post):
        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "text"}]
        )
        mock_post.return_value = _response(status_code=401, json_data={})

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("no_args_tool", args={}, output_dir="/tmp")

        self.assertEqual(ctx.exception.status_code, 401)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_error_message_falls_back_to_plain_text_body(self, mock_get, mock_post):
        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "text"}]
        )
        mock_post.return_value = _response(
            status_code=400, headers={"Content-Type": "text/plain"}, text="disallowed extension: .exe"
        )

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("no_args_tool", args={}, output_dir="/tmp")

        self.assertIn("disallowed extension", str(ctx.exception))

    # -- result filename / extension ---------------------------------------

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_segmentation_result_gets_nii_gz_extension(self, mock_get, mock_post):
        mock_get.return_value = _response(
            json_data=[{"name": "seg_tool", "arguments": {}, "output_kind": "segmentation"}]
        )
        mock_post.return_value = _response(
            content=b"binary", headers={"Content-Type": "application/octet-stream"}
        )

        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run("seg_tool", args={}, output_dir=out_dir)

        self.assertTrue(result.path.endswith(".nii.gz"), result.path)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_result_filename_prefers_content_disposition(self, mock_get, mock_post):
        mock_get.return_value = _response(
            json_data=[{"name": "seg_tool", "arguments": {}, "output_kind": "segmentation"}]
        )
        mock_post.return_value = _response(
            content=b"binary",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="custom_name.nrrd"',
            },
        )

        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run("seg_tool", args={}, output_dir=out_dir)

        self.assertEqual(os.path.basename(result.path), "custom_name.nrrd")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_xlsx_result_keeps_its_own_extension_not_zip(self, mock_get, mock_post):
        # Regression: an .xlsx is a zip container internally (OOXML). The
        # resolved result filename must never end in .zip, or downstream code
        # deciding "should this be unpacked as an archive" purely from the
        # extension (see slicer_io.is_extractable_archive) would wrongly
        # extract a spreadsheet into raw XML parts instead of keeping it.
        mock_get.return_value = _response(
            json_data=[{"name": "SurgMovPred", "arguments": {}, "output_kind": "file"}]
        )
        mock_post.return_value = _response(
            content=b"PK\x03\x04fake-xlsx-bytes",
            headers={
                "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "Content-Disposition": 'attachment; filename="predictions_outputs.xlsx"',
            },
        )

        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run("SurgMovPred", args={}, output_dir=out_dir)

        self.assertEqual(os.path.basename(result.path), "predictions_outputs.xlsx")
        self.assertFalse(result.path.lower().endswith(".zip"))

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_result_filename_falls_back_to_mimetype_extension_without_content_disposition(self, mock_get, mock_post):
        # No Content-Disposition this time: the extension must still come from
        # a real MIME lookup (mirroring the server's own mimetypes.guess_type),
        # not the generic .bin/.gz guess.
        mock_get.return_value = _response(
            json_data=[{"name": "SurgMovPred", "arguments": {}, "output_kind": "file"}]
        )
        mock_post.return_value = _response(
            content=b"csv,data",
            headers={"Content-Type": "text/csv"},
        )

        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run("SurgMovPred", args={}, output_dir=out_dir)

        self.assertTrue(result.path.endswith(".csv"), result.path)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_writes_binary_result_to_output_dir(self, mock_get, mock_post):
        import tempfile

        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "file"}]
        )
        mock_post.return_value = _response(
            content=b"binary-bytes", headers={"Content-Type": "application/gzip"}
        )

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run("no_args_tool", args={}, output_dir=out_dir)

            self.assertEqual(result.kind, "file")
            self.assertTrue(result.path.endswith(".gz"))
            with open(result.path, "rb") as fh:
                self.assertEqual(fh.read(), b"binary-bytes")

    # -- download integrity ------------------------------------------------

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_run_streams_the_response(self, mock_get, mock_post):
        # stream=True is what keeps a multi-hundred-MB result archive out of
        # Slicer's RAM: the body must be consumed by iter_content, not
        # pre-buffered inside requests.post.
        mock_get.return_value = _response(json_data=TOOLS_RESPONSE)
        mock_post.return_value = _response(json_data={"result": "ok"})

        self.client.run("example_tool", args={"label": "x"}, files={"file": __file__}, output_dir="/tmp")

        _, kwargs = mock_post.call_args
        self.assertIs(kwargs["stream"], True)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_truncated_download_is_rejected_and_removed(self, mock_get, mock_post):
        import tempfile

        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "file"}]
        )
        mock_post.return_value = _response(
            content=b"only-half-the-bytes",
            headers={"Content-Type": "application/gzip", "Content-Length": "999999"},
        )

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(ServerToolError) as ctx:
                self.client.run("no_args_tool", args={}, output_dir=out_dir)
            self.assertIn("Truncated", str(ctx.exception))
            # The partial file must not survive: a later step (or the user)
            # finding it would mistake it for a real result.
            self.assertEqual(os.listdir(out_dir), [])

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_corrupt_zip_result_is_rejected_and_removed(self, mock_get, mock_post):
        import tempfile

        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "files"}]
        )
        # Looks like a zip (magic bytes, .zip filename) but has no readable
        # central directory -- exactly what a connection cut mid-body leaves.
        mock_post.return_value = _response(
            content=b"PK\x03\x04this-is-not-a-whole-archive",
            headers={
                "Content-Type": "application/zip",
                "Content-Disposition": 'attachment; filename="AMASSS_Pred.zip"',
            },
        )

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(ServerToolError) as ctx:
                self.client.run("no_args_tool", args={}, output_dir=out_dir)
            self.assertIn("unreadable", str(ctx.exception))
            self.assertEqual(os.listdir(out_dir), [])

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_intact_zip_result_passes_verification(self, mock_get, mock_post):
        import tempfile

        content = _zip_bytes({"scan_Pred_MAND.nii.gz": "seg", "scan_Pred_MAND.vtk": "surf"})
        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "files"}]
        )
        mock_post.return_value = _response(
            content=content,
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(len(content)),
                "Content-Disposition": 'attachment; filename="AMASSS_Pred.zip"',
            },
        )

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run("no_args_tool", args={}, output_dir=out_dir)

            self.assertEqual(result.kind, "file")
            with open(result.path, "rb") as fh:
                self.assertEqual(fh.read(), content)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_length_check_skipped_when_body_travels_compressed(self, mock_get, mock_post):
        import tempfile

        # With Content-Encoding, Content-Length counts wire bytes while
        # iter_content yields the decompressed stream: a mismatch there is
        # normal and must not be reported as truncation.
        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "file"}]
        )
        mock_post.return_value = _response(
            content=b"decompressed-and-longer-than-the-wire-count",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": "10",
                "Content-Encoding": "gzip",
                "Content-Disposition": 'attachment; filename="result.bin"',
            },
        )

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run("no_args_tool", args={}, output_dir=out_dir)
            self.assertEqual(result.kind, "file")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_download_reports_progress_with_a_total(self, mock_get, mock_post):
        # A silent multi-minute run is what made a user cancel a job that was
        # working; the download phase must report bytes as they land.
        import tempfile

        content = _zip_bytes({"a.nii.gz": "x" * 5_000_000})
        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "files"}]
        )
        mock_post.return_value = _response(
            content=content,
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(len(content)),
                "Content-Disposition": 'attachment; filename="out.zip"',
            },
        )

        messages = []
        with tempfile.TemporaryDirectory() as out_dir:
            self.client.run(
                "no_args_tool", args={}, output_dir=out_dir, progress_cb=messages.append
            )

        downloads = [m for m in messages if m.startswith("Downloading results")]
        self.assertTrue(downloads, messages)
        self.assertIn("MB", downloads[-1])
        self.assertIn("100%", downloads[-1])

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_download_progress_omits_the_total_when_unusable(self, mock_get, mock_post):
        # Content-Encoding makes Content-Length count wire bytes, so a
        # percentage computed from it would run past 100.
        import tempfile

        mock_get.return_value = _response(
            json_data=[{"name": "no_args_tool", "arguments": {}, "output_kind": "file"}]
        )
        mock_post.return_value = _response(
            content=b"z" * 2_000_000,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": "10",
                "Content-Encoding": "gzip",
                "Content-Disposition": 'attachment; filename="out.bin"',
            },
        )

        messages = []
        with tempfile.TemporaryDirectory() as out_dir:
            self.client.run(
                "no_args_tool", args={}, output_dir=out_dir, progress_cb=messages.append
            )

        downloads = [m for m in messages if m.startswith("Downloading results")]
        self.assertTrue(downloads, messages)
        self.assertNotIn("%", downloads[-1])

    # -- health --------------------------------------------------------

    @mock.patch("requests.Session.get")
    def test_health_true(self, mock_get):
        mock_get.return_value = _response(json_data={"status": "ok"})
        self.assertTrue(self.client.health())

    @mock.patch("requests.Session.get")
    def test_health_uses_short_timeout(self, mock_get):
        mock_get.return_value = _response(json_data={"status": "ok"})
        self.client.health()

        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["timeout"], _HEALTH_CHECK_TIMEOUT)
        self.assertLess(_HEALTH_CHECK_TIMEOUT, self.client._timeout)

    @mock.patch("requests.Session.get")
    def test_health_false_on_network_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("boom")
        self.assertFalse(self.client.health())

    @mock.patch("requests.Session.get")
    def test_health_does_not_swallow_unrelated_errors(self, mock_get):
        response = _response(json_data={"status": "ok"})
        response.json.side_effect = KeyError("programming error, not a network/parse issue")
        mock_get.return_value = response

        with self.assertRaises(KeyError):
            self.client.health()


class IsFileTypeTest(unittest.TestCase):
    def test_literal_file(self):
        self.assertTrue(is_file_type("file"))

    def test_suffixed_types(self):
        self.assertTrue(is_file_type("zip_file"))
        self.assertTrue(is_file_type("nifti_file"))
        self.assertTrue(is_file_type("csv_file"))  # not seen yet, but the convention must cover it

    def test_scalar_types_are_not_file_types(self):
        self.assertFalse(is_file_type("str"))
        self.assertFalse(is_file_type("int"))
        self.assertFalse(is_file_type("float"))
        self.assertFalse(is_file_type("bool"))

    def test_missing_type_is_not_a_file_type(self):
        self.assertFalse(is_file_type(""))


class PublishedExtensionsTest(unittest.TestCase):
    """The server publishes each file type's extensions next to `types`, so
    the client keeps no copy of its FILE_TYPES table."""

    def test_extensions_come_from_the_schema(self):
        spec = {
            "type": "csv_file",
            "types": ["csv_file", "folder"],
            "extensions": {"csv_file": [".csv"], "folder": [".zip"]},
        }

        # "folder"'s .zip is what a zipped folder uploads as, not something a
        # file picker should offer.
        self.assertEqual(file_extensions_for(spec), (".csv",))

    def test_a_multi_format_type_needs_no_client_side_table(self):
        # AMASSS's input. Nothing in the name says .nii/.nrrd/.gipl.
        spec = {
            "type": "volume_or_zip_file",
            "types": ["volume_or_zip_file"],
            "extensions": {"volume_or_zip_file": [".nii", ".nii.gz", ".nrrd", ".zip"]},
        }

        self.assertEqual(file_extensions_for(spec), (".nii", ".nii.gz", ".nrrd", ".zip"))

    def test_a_type_the_server_declines_to_restrict_is_unrestricted(self):
        spec = {"type": "file", "types": ["file"], "extensions": {"file": None}}

        self.assertEqual(file_extensions_for(spec), ())

    def test_the_schema_wins_over_the_fallback_table(self):
        # A server that changes an extension must not be second-guessed.
        spec = {
            "type": "nifti_file",
            "types": ["nifti_file"],
            "extensions": {"nifti_file": [".nii", ".nii.gz", ".nrrd"]},
        }

        self.assertEqual(file_extensions_for(spec), (".nii", ".nii.gz", ".nrrd"))

    def test_a_server_predating_the_field_still_works(self):
        # No "extensions" key at all: fall back to the local table.
        self.assertEqual(file_extensions_for({"types": ["nifti_file"]}), (".nii", ".nii.gz"))


class ArgumentTypesTest(unittest.TestCase):
    """An argument may accept several types (`types`), e.g. example_tool's
    `input`: a .csv file *or* a whole folder.

    These exercise the fallback path - a server that does not publish
    `extensions` (see PublishedExtensionsTest for the normal one).
    """

    _INPUT = {"type": "csv_file", "types": ["csv_file", "folder"]}

    def test_types_wins_over_the_single_type(self):
        self.assertEqual(argument_types(self._INPUT), ["csv_file", "folder"])

    def test_falls_back_to_the_single_type(self):
        # A schema predating the `types` field must still work.
        self.assertEqual(argument_types({"type": "zip_file"}), ["zip_file"])
        self.assertEqual(argument_types({}), [])

    def test_accepts_folder(self):
        self.assertTrue(accepts_folder(self._INPUT))
        self.assertFalse(accepts_folder({"type": "zip_file", "types": ["zip_file"]}))
        self.assertFalse(accepts_folder({"type": "str", "types": ["str"]}))

    def test_extensions_come_from_the_non_folder_types(self):
        self.assertEqual(file_extensions_for(self._INPUT), (".csv",))

    def test_extensions_of_a_multi_extension_type(self):
        self.assertEqual(file_extensions_for({"types": ["nifti_file"]}), (".nii", ".nii.gz"))

    def test_extensions_of_several_file_types(self):
        self.assertEqual(file_extensions_for({"types": ["csv_file", "zip_file"]}), (".csv", ".zip"))

    def test_generic_file_type_is_unrestricted(self):
        self.assertEqual(file_extensions_for({"types": ["file"]}), ())
        self.assertEqual(file_extensions_for({"types": ["csv_file", "file"]}), ())

    def test_folder_only_argument_has_no_extensions(self):
        self.assertEqual(file_extensions_for({"types": ["folder"]}), ())

    def test_unknown_file_type_derives_its_extension_from_its_name(self):
        # Same convention as is_file_type: a new "<x>_file" the server adds
        # needs no client-side change.
        self.assertEqual(file_extensions_for({"types": ["vtk_file"]}), (".vtk",))

    def test_a_multi_format_type_lists_every_extension(self):
        # The server's "volume_or_zip_file" (used by its AMASSS tool): a name
        # that spells out no extension at all. Its entry has to mirror the
        # server's FILE_TYPES table, which /tools does not publish.
        self.assertEqual(
            file_extensions_for({"types": ["volume_or_zip_file"]}),
            (".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz", ".zip"),
        )

    def test_an_unknown_compound_type_name_is_not_turned_into_a_filter(self):
        # Guessing ".scan_or_mesh" from "scan_or_mesh_file" would give a file
        # dialog matching nothing - strictly worse than not filtering. A
        # compound name we don't know degrades to an unrestricted picker.
        self.assertEqual(file_extensions_for({"types": ["scan_or_mesh_file"]}), ())


class RealServerSchemaTest(unittest.TestCase):
    """Regression coverage for the actual /tools payload returned by the dev
    server: file arguments are typed "nifti_file"/"zip_file", never the
    literal "file" - see ARCHITECTURE.md."""

    def setUp(self):
        self.client = ToolServerClient("https://localhost:8000", "dev-token")

    @mock.patch("requests.Session.get")
    def test_example_tool_nifti_file_argument_is_recognized(self, mock_get):
        mock_get.return_value = _response(
            json_data=[
                {
                    "name": "example_tool",
                    "arguments": {
                        "label": {"type": "str", "required": True},
                        "file": {"type": "nifti_file", "required": True},
                        "threshold": {"type": "float", "required": True},
                        "iterations": {"type": "int", "required": False},
                    },
                    "output_kind": "text",
                }
            ]
        )

        with self.assertRaises(ServerToolError) as ctx:
            # Missing the required "file" upload and "threshold" argument.
            self.client.run("example_tool", args={"label": "x"}, output_dir="/tmp")

        # Whichever is checked first, it must be a *local* validation error
        # (schema-driven), not a generic/unrelated failure.
        self.assertIn("example_tool", str(ctx.exception))

    # The real surg_mov_pred schema since the model moved fully server-side:
    # "model" is a scalar str (the *name* of a server-hosted model, picked
    # from GET /tools/{tool}/data), only "input" is still uploaded.
    _SURG_MOV_PRED_SCHEMA = [
        {
            "name": "SurgMovPred",
            "arguments": {
                "model": {"type": "str", "required": True, "server_selectable": "model"},
                "input": {"type": "zip_file", "required": True, "server_selectable": "testfile"},
            },
            "output_kind": "file",
        }
    ]

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_surg_mov_pred_sends_model_name_as_form_value(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=self._SURG_MOV_PRED_SCHEMA)
        mock_post.return_value = _response(content=_zip_bytes(), headers={"Content-Type": "application/zip"})

        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            self.client.run(
                "SurgMovPred",
                args={"model": "stacking_v2.zip"},
                files={"input": __file__},
                output_dir=out_dir,
            )

        _, kwargs = mock_post.call_args
        # The model travels as a plain form value (its server-side name)...
        self.assertEqual(kwargs["data"], {"model": "stacking_v2.zip"})
        # ...and only "input" is uploaded as a file.
        self.assertEqual(set(kwargs["files"].keys()), {"input"})

    @mock.patch("requests.Session.get")
    def test_surg_mov_pred_missing_model_name_fails_locally(self, mock_get):
        mock_get.return_value = _response(json_data=self._SURG_MOV_PRED_SCHEMA)

        with self.assertRaises(ServerToolError) as ctx:
            self.client._validate_against_schema(
                self.client.get_tool_schema("SurgMovPred"), {}, {"input": __file__}
            )
        self.assertIn("model", str(ctx.exception))

    @mock.patch("requests.Session.get")
    def test_surg_mov_pred_model_is_not_a_file_argument(self, mock_get):
        # Uploading a local model package is no longer supported: "model" is a
        # scalar, so passing it under `files` must be rejected before any
        # network round-trip.
        mock_get.return_value = _response(json_data=self._SURG_MOV_PRED_SCHEMA)

        with self.assertRaises(ServerToolError):
            self.client._validate_against_schema(
                self.client.get_tool_schema("SurgMovPred"),
                {},
                {"model": __file__, "input": __file__},
            )


class ExampleToolRequestTest(unittest.TestCase):
    """The full request/response round-trip for `example_tool` - the tool that
    exercises everything the client has to do: a choice, a multichoice, an
    input accepting a file or a folder, and a multi-file (.zip) result.

    The schema is the server's real GET /tools entry, verbatim.
    """

    EXAMPLE_TOOL = {
        "name": "example_tool",
        "output_kind": "files",
        "arguments": {
            "label": {
                "type": "str", "types": ["str"], "required": True,
                "description": "Free-text label for this run",
                "server_selectable": None, "choices": None,
            },
            "input": {
                "type": "csv_file", "types": ["csv_file", "folder"], "required": True,
                "description": "A single .csv file, or a folder of .csv/.xlsx/.ods files sent as a .zip archive",
                "server_selectable": None, "choices": None,
            },
            "threshold": {
                "type": "float", "types": ["float"], "required": True,
                "description": "Numeric threshold parameter",
                "server_selectable": None, "choices": None,
            },
            "iterations": {
                "type": "int", "types": ["int"], "required": False,
                "description": "Optional number of iterations",
                "server_selectable": None, "choices": None,
            },
            "outputs": {
                "type": "multichoice", "types": ["multichoice"], "required": False,
                "description": "Which result files to produce",
                "server_selectable": None,
                "choices": {"summary": True, "preview": True, "columns": False},
            },
            "preview_format": {
                "type": "choice", "types": ["choice"], "required": False,
                "description": "Format of the preview file",
                "server_selectable": None,
                "choices": {"csv": True, "json": False},
            },
        },
    }

    def setUp(self):
        self.client = ToolServerClient("https://localhost:8000", "dev-token")

    def _args(self, **overrides):
        args = {
            "label": "test",
            "threshold": 1.0,
            "preview_format": "json",
            "outputs": {"summary": True, "preview": True, "columns": False},
        }
        args.update(overrides)
        return args

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_form_fields_match_the_contract(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        mock_post.return_value = _response(
            content=_zip_bytes(), headers={"Content-Type": "application/zip"}
        )

        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            self.client.run("example_tool", args=self._args(), files={"input": __file__}, output_dir=out_dir)

        _, kwargs = mock_post.call_args
        data = kwargs["data"]
        # A "choice" travels as the option name, in clear.
        self.assertEqual(data["preview_format"], "json")
        # A "multichoice" travels as the complete dict, as JSON.
        self.assertEqual(json.loads(data["outputs"]), {"summary": True, "preview": True, "columns": False})
        self.assertEqual(data["label"], "test")
        # Only the file argument is uploaded.
        self.assertEqual(set(kwargs["files"]), {"input"})

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_batch_of_a_cohort_says_which_batch_it_is(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        mock_post.return_value = _response(content=_zip_bytes(), headers={"Content-Type": "application/zip"})
        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            self.client.run("example_tool", args=self._args(), files={"input": __file__}, output_dir=out_dir,
                            batch={"id": "cohort-aaaaaaaa", "index": 2, "total": 5})
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["X-Batch-Id"], "cohort-aaaaaaaa")
        self.assertEqual(headers["X-Batch-Index"], "2")
        self.assertEqual(headers["X-Batch-Total"], "5")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_run_that_is_not_a_batch_sends_no_batch_header(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        mock_post.return_value = _response(content=_zip_bytes(), headers={"Content-Type": "application/zip"})
        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            self.client.run("example_tool", args=self._args(), files={"input": __file__}, output_dir=out_dir)
        headers = mock_post.call_args.kwargs["headers"]
        self.assertFalse([name for name in headers if name.startswith("X-Batch-")])

    @mock.patch("requests.Session.get")
    def test_the_batch_policy_is_read_and_an_older_server_answers_none(self, mock_get):
        mock_get.return_value = _response(json_data={"batches": "parallel", "max_parallel": 4})
        self.assertEqual(self.client.batch_policy(), {"batches": "parallel", "max_parallel": 4})
        mock_get.return_value = _response(status_code=404, json_data={"detail": "Not Found"})
        self.assertIsNone(self.client.batch_policy())
        mock_get.side_effect = requests.ConnectionError("down")
        self.assertIsNone(self.client.batch_policy())

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_zipped_folder_is_uploaded_under_the_same_argument_name(self, mock_get, mock_post):
        # The client zips a folder selection before sending; the server sees an
        # ordinary .zip under "input" and unpacks it.
        import tempfile

        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        mock_post.return_value = _response(content=_zip_bytes(), headers={"Content-Type": "application/zip"})

        with tempfile.TemporaryDirectory() as work_dir:
            archive = os.path.join(work_dir, "example_tool_input.zip")
            with open(archive, "wb") as fh:
                fh.write(b"PK\x03\x04")
            self.client.run("example_tool", args=self._args(), files={"input": archive}, output_dir=work_dir)

        _, kwargs = mock_post.call_args
        filename, _handle = kwargs["files"]["input"]
        self.assertEqual(filename, "example_tool_input.zip")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_files_output_kind_is_saved_under_its_real_name(self, mock_get, mock_post):
        import tempfile

        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        mock_post.return_value = _response(
            content=_zip_bytes(),
            headers={
                "Content-Type": "application/zip",
                "Content-Disposition": 'attachment; filename="example_tool_results.zip"',
            },
        )

        with tempfile.TemporaryDirectory() as out_dir:
            result = self.client.run(
                "example_tool", args=self._args(), files={"input": __file__}, output_dir=out_dir
            )

        self.assertEqual(result.kind, "file")
        # A .zip of several results: base_widget's "save_as" handling unpacks
        # it into the output folder (see slicer_io.is_extractable_archive).
        self.assertEqual(os.path.basename(result.path), "example_tool_results.zip")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_unknown_option_422_is_shown_verbatim(self, mock_get, mock_post):
        # No client-side fallback for an out-of-list value: the server's own
        # message names the offending option and lists the valid ones.
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        detail = "Argument 'preview_format': unknown option 'xml'. Expected one of: csv, json"
        mock_post.return_value = _response(status_code=422, json_data={"detail": detail})

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run(
                "example_tool",
                args=self._args(preview_format="xml"),
                files={"input": __file__},
                output_dir="/tmp",
            )

        self.assertEqual(str(ctx.exception), detail)
        self.assertEqual(ctx.exception.status_code, 422)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_413_reports_the_server_s_actual_limit(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        mock_post.return_value = _response(
            status_code=413, json_data={"detail": "File exceeds the 500 MB limit."}
        )

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("example_tool", args=self._args(), files={"input": __file__}, output_dir="/tmp")

        self.assertEqual(str(ctx.exception), "File exceeds the 500 MB limit.")
        self.assertEqual(ctx.exception.status_code, 413)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_400_reports_the_allowed_extensions(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        detail = "Unsupported file extension for 'input'. Allowed: ('.csv', '.zip')"
        mock_post.return_value = _response(status_code=400, json_data={"detail": detail})

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("example_tool", args=self._args(), files={"input": __file__}, output_dir="/tmp")

        self.assertEqual(str(ctx.exception), detail)
        self.assertEqual(ctx.exception.status_code, 400)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_401_reports_the_server_message(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])
        mock_post.return_value = _response(status_code=401, json_data={"detail": "Invalid token."})

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("example_tool", args=self._args(), files={"input": __file__}, output_dir="/tmp")

        self.assertEqual(str(ctx.exception), "Invalid token.")

    @mock.patch("requests.Session.get")
    def test_missing_input_file_fails_locally(self, mock_get):
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])

        with self.assertRaises(ServerToolError) as ctx:
            self.client.run("example_tool", args=self._args(), files={}, output_dir="/tmp")

        self.assertIn("input", str(ctx.exception))

    @mock.patch("requests.Session.get")
    def test_optional_choice_arguments_may_be_omitted(self, mock_get):
        # Omitting a multichoice entirely is what applies the server's declared
        # defaults - it must not be forced into the payload.
        mock_get.return_value = _response(json_data=[self.EXAMPLE_TOOL])

        self.client._validate_against_schema(
            self.client.get_tool_schema("example_tool"),
            {"label": "test", "threshold": 1.0},
            {"input": __file__},
        )


class BulkTransferWiringTest(unittest.TestCase):
    """Which transfer path `run()` picks, and what it does with a result the
    server hands back by reference. The transfer machinery itself is covered
    against a real socket in test_transfer.py; what is tested here is the
    decision and the plumbing around it."""

    TOOL = {
        "name": "big_tool",
        "output_kind": "files",
        "arguments": {
            "input": {"type": "nifti_file", "types": ["nifti_file"], "required": True},
        },
    }

    def setUp(self):
        self.client = ToolServerClient("https://server", "tok", chunk_bytes=1024)
        self.work = tempfile.mkdtemp(prefix="wiring_test_")
        self.addCleanup(shutil.rmtree, self.work, True)

    def _input(self, size):
        path = os.path.join(self.work, "scan.nii.gz")
        with open(path, "wb") as handle:
            handle.write(b"x" * size)
        return path

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_big_input_goes_up_through_the_upload_endpoints(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data={"result": "done"})

        with mock.patch.object(transfer, "upload_file", return_value="up123") as upload:
            self.client.run("big_tool", files={"input": self._input(8192)}, output_dir=self.work)

        upload.assert_called_once()
        # The run request then carries a REFERENCE, not the bytes.
        sent = mock_post.call_args.kwargs
        self.assertIsNone(sent["files"])
        self.assertEqual(json.loads(sent["data"]["__uploads__"]), {"input": "up123"})

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_small_input_stays_on_the_single_request_path(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data={"result": "done"})

        with mock.patch.object(transfer, "upload_file") as upload:
            self.client.run("big_tool", files={"input": self._input(100)}, output_dir=self.work)

        upload.assert_not_called()
        sent = mock_post.call_args.kwargs
        self.assertIn("input", sent["files"])
        self.assertNotIn("__uploads__", sent["data"])

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_server_without_the_endpoints_falls_back_silently(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data={"result": "done"})

        with mock.patch.object(
            transfer, "upload_file", side_effect=transfer.UnsupportedByServer
        ) as upload:
            self.client.run("big_tool", files={"input": self._input(8192)}, output_dir=self.work)

        upload.assert_called_once()
        # The file still travelled, the old way, and the run still worked.
        self.assertIn("input", mock_post.call_args.kwargs["files"])

        # And the verdict sticks: a second run does not probe again.
        with mock.patch.object(transfer, "upload_file") as second:
            self.client.run("big_tool", files={"input": self._input(8192)}, output_dir=self.work)
        second.assert_not_called()

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_the_run_request_asks_for_reference_delivery(self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data={"result": "done"})

        self.client.run("big_tool", files={"input": self._input(100)}, output_dir=self.work)

        self.assertEqual(
            mock_post.call_args.kwargs["headers"].get("X-Result-Delivery"), "reference"
        )

    @mock.patch("requests.Session.delete")
    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_referenced_result_is_pulled_in_ranges_and_then_released(
        self, mock_get, mock_post, mock_delete
    ):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(
            json_data={
                "result_ref": {
                    "result_id": "res123",
                    "filename": "big_tool_output.zip",
                    "size": 4096,
                    "media_type": "application/zip",
                }
            }
        )

        def fake_download(_session, _url, destination, *_a, **_kw):
            with open(destination, "wb") as handle:
                handle.write(_zip_bytes())
            return destination

        with mock.patch.object(transfer, "download_ranged", side_effect=fake_download) as ranged:
            result = self.client.run(
                "big_tool", files={"input": self._input(100)}, output_dir=self.work
            )

        self.assertEqual(result.kind, "file")
        self.assertEqual(result.path, os.path.join(self.work, "big_tool_output.zip"))
        self.assertEqual(ranged.call_args.args[1], "https://server/results/res123")
        # Released once it is safely on disk, so TEMP_DIR does not hold every
        # result for the full server-side TTL.
        self.assertEqual(mock_delete.call_args.args[0], "https://server/results/res123")

    def _reference_response(self, filename="big_tool_output.zip", size=4096):
        return _response(
            json_data={
                "result_ref": {
                    "result_id": "res123",
                    "filename": filename,
                    "size": size,
                    "media_type": "application/zip",
                }
            }
        )

    @mock.patch("requests.Session.delete")
    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_failed_download_still_releases_the_result(self, mock_get, mock_post, mock_delete):
        """The server holds the result until it is told it can go. A download
        that died halfway is exactly when a return-only cleanup would leave
        patient data sitting there until the idle reaper got to it."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = self._reference_response()

        with mock.patch.object(
            transfer, "download_ranged", side_effect=ServerToolError("connection lost")
        ):
            with self.assertRaises(ServerToolError):
                self.client.run(
                    "big_tool", files={"input": self._input(100)}, output_dir=self.work
                )

        self.assertEqual(mock_delete.call_args.args[0], "https://server/results/res123")

    @mock.patch("requests.Session.delete")
    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_corrupt_archive_still_releases_the_result(self, mock_get, mock_post, mock_delete):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = self._reference_response()

        def truncated(_session, _url, destination, *_a, **_kw):
            with open(destination, "wb") as handle:
                handle.write(_zip_bytes()[:20])   # a .zip that cannot be read
            return destination

        with mock.patch.object(transfer, "download_ranged", side_effect=truncated):
            with self.assertRaises(ServerToolError):
                self.client.run(
                    "big_tool", files={"input": self._input(100)}, output_dir=self.work
                )

        self.assertEqual(mock_delete.call_args.args[0], "https://server/results/res123")

    @mock.patch("requests.Session.delete")
    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_releasing_is_retried_and_never_fails_a_finished_run(
        self, mock_get, mock_post, mock_delete
    ):
        """A dropped packet should not decide whether the file goes away now or
        waits for the reaper. But a run that produced a good result must not be
        reported as failed because the tidy-up call did not get through."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = self._reference_response()
        mock_delete.side_effect = requests.RequestException("nope")

        def fake_download(_session, _url, destination, *_a, **_kw):
            with open(destination, "wb") as handle:
                handle.write(_zip_bytes())
            return destination

        with mock.patch.object(transfer, "download_ranged", side_effect=fake_download):
            result = self.client.run(
                "big_tool", files={"input": self._input(100)}, output_dir=self.work
            )

        self.assertEqual(result.kind, "file")
        self.assertEqual(mock_delete.call_count, 2)

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_result_filename_can_never_escape_the_output_folder(self, mock_get, mock_post):
        """The name is the server's to choose, so it is treated as untrusted:
        a path separator in it must not place the file outside output_dir."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(
            json_data={
                "result_ref": {
                    "result_id": "res123",
                    "filename": "../../../../etc/cron.d/pwned",
                    "size": 10,
                }
            }
        )

        def fake_download(_session, _url, destination, *_a, **_kw):
            with open(destination, "wb") as handle:
                handle.write(b"0123456789")
            return destination

        with mock.patch.object(transfer, "download_ranged", side_effect=fake_download):
            with mock.patch("requests.Session.delete"):
                result = self.client.run(
                    "big_tool", files={"input": self._input(100)}, output_dir=self.work
                )

        self.assertEqual(os.path.dirname(result.path), self.work)
        self.assertEqual(os.path.basename(result.path), "pwned")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_plain_text_result_is_still_a_text_result(self, mock_get, mock_post):
        """Reference delivery only ever applies to file results; a "text"
        tool's JSON must keep being read exactly as before."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data={"result": "42"})

        result = self.client.run(
            "big_tool", files={"input": self._input(100)}, output_dir=self.work
        )

        self.assertEqual(result.kind, "text")
        self.assertEqual(result.text, "42")


class TestfileEntriesTest(unittest.TestCase):
    """`GET /tools/{tool}/data` answers in two shapes, and both have to work:
    a current server describes each test file, an older one lists names only."""

    def test_described_entries_carry_their_kind_and_size(self):
        data = {
            "testfiles": ["CBCT_FullyAuto", "ROI_box.mrk.json"],
            "entries": {"testfiles": [
                {"name": "CBCT_FullyAuto", "kind": "folder", "size": 355640000},
                {"name": "ROI_box.mrk.json", "kind": "file", "size": 2969},
            ]},
        }

        self.assertEqual(
            testfile_entries(data),
            [
                {"name": "CBCT_FullyAuto", "kind": "folder", "size": 355640000},
                {"name": "ROI_box.mrk.json", "kind": "file", "size": 2969},
            ],
        )

    def test_a_server_publishing_only_names_still_yields_entries(self):
        entries = testfile_entries({"testfiles": ["demo.zip"]})
        self.assertEqual(entries, [{"name": "demo.zip", "kind": None, "size": None}])

    def test_an_unsizeable_entry_is_unknown_and_not_zero(self):
        """A backend that cannot size a tree cheaply sends null, and 0 would be
        indistinguishable from an empty folder."""
        data = {
            "testfiles": ["cohort"],
            "entries": {"testfiles": [{"name": "cohort", "kind": "folder", "size": None}]},
        }
        self.assertEqual(testfile_entries(data)[0]["size"], None)

    def test_junk_in_the_described_half_never_costs_an_entry(self):
        """The seam between two repositories: a field the other side changes
        must not empty the picker."""
        data = {
            "testfiles": ["demo.zip"],
            "entries": {"testfiles": [
                "not a dict",
                {"name": "demo.zip", "kind": "directory", "size": "big"},
            ]},
        }
        self.assertEqual(
            testfile_entries(data), [{"name": "demo.zip", "kind": None, "size": None}]
        )

    def test_the_flat_list_decides_which_names_exist(self):
        """An entry the server does not also list is not offered: the name is
        what a download would ask for, and it has to be one the server serves."""
        data = {
            "testfiles": ["demo.zip"],
            "entries": {"testfiles": [
                {"name": "demo.zip", "kind": "file", "size": 1},
                {"name": "ghost.zip", "kind": "file", "size": 1},
            ]},
        }
        self.assertEqual([entry["name"] for entry in testfile_entries(data)], ["demo.zip"])


class DownloadTestFileTest(unittest.TestCase):
    """`download_testfile`: the tool's own hosted test data, fetched so the
    user can open it beside the panel. Bearer-protected, and pulled over
    parallel ranges when the server offers them."""

    def setUp(self):
        self.client = ToolServerClient("https://example.org/", "secret-token")
        self.work = tempfile.mkdtemp(prefix="testfile_download_")
        self.addCleanup(shutil.rmtree, self.work, True)
        self.destination = os.path.join(self.work, "sample_scan.nii.gz")
        # No ranges by default, so these cover the sequential path -- and, more
        # importantly, so none of them can reach the network.
        self._head = mock.patch.object(
            requests.Session, "head", return_value=self._head_response(None)
        )
        self._head.start()
        self.addCleanup(self._head.stop)

    @staticmethod
    def _head_response(size, accept_ranges="bytes"):
        response = mock.MagicMock()
        response.ok = True
        response.headers = {}
        if size is not None:
            response.headers = {"Accept-Ranges": accept_ranges, "Content-Length": str(size)}
        return response

    def _streaming_response(self, chunks, headers=None, status=200):
        response = mock.MagicMock()
        response.status_code = status
        response.ok = 200 <= status < 300
        response.headers = headers or {}
        response.iter_content = lambda chunk_size: iter(chunks)
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.json.side_effect = ValueError("no json body")
        return response

    def test_streams_the_body_to_the_destination_with_the_token(self):
        response = self._streaming_response([b"abc", b"def"])

        with mock.patch.object(requests.Session, "get", return_value=response) as get:
            result = self.client.download_testfile(
                "AMASSS", "sample_scan.nii.gz", self.destination
            )

        self.assertEqual(result, self.destination)
        with open(self.destination, "rb") as handle:
            self.assertEqual(handle.read(), b"abcdef")
        self.assertEqual(
            get.call_args.args[0], "https://example.org/tools/AMASSS/testfiles/sample_scan.nii.gz"
        )
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer secret-token")
        # stream=True is what keeps a 648 MB cohort out of Slicer's RAM.
        self.assertTrue(get.call_args.kwargs.get("stream"))

    def test_a_name_is_escaped_into_the_path_rather_than_read_as_one(self):
        response = self._streaming_response([b""])

        with mock.patch.object(requests.Session, "get", return_value=response) as get:
            self.client.download_testfile("AMASSS", "a b/c.nii.gz", self.destination)

        self.assertEqual(
            get.call_args.args[0], "https://example.org/tools/AMASSS/testfiles/a%20b%2Fc.nii.gz"
        )

    def test_progress_reports_percentages_from_content_length(self):
        response = self._streaming_response(
            [b"a" * 512, b"b" * 512], headers={"Content-Length": "1024"}
        )
        messages = []

        with mock.patch.object(requests.Session, "get", return_value=response):
            self.client.download_testfile(
                "AMASSS", "sample_scan.nii.gz", self.destination, progress_cb=messages.append
            )

        self.assertEqual(len(messages), 2)
        self.assertIn("(50%)", messages[0])
        self.assertIn("(100%)", messages[1])
        # The label names the file being fetched, not the tool-run wording.
        self.assertIn("sample_scan.nii.gz", messages[0])

    def test_an_error_status_is_a_ServerToolError_not_a_raw_HTTPError(self):
        """The panel shows the message verbatim; a 404 here means the server
        no longer hosts that name, which is worth saying."""
        response = self._streaming_response([], status=404)

        with mock.patch.object(requests.Session, "get", return_value=response):
            with self.assertRaises(ServerToolError) as caught:
                self.client.download_testfile("AMASSS", "gone.nii.gz", self.destination)

        self.assertEqual(caught.exception.status_code, 404)
        self.assertFalse(os.path.exists(self.destination))

    def test_a_big_file_goes_over_parallel_ranges(self):
        """The same path a large result takes (`_download_reference`): one
        connection is bound by its own congestion window, and these are
        hundreds of megabytes."""
        size = transfer.MIN_CHUNKED_BYTES + 1
        with mock.patch.object(
            requests.Session, "head", return_value=self._head_response(size)
        ):
            with mock.patch.object(
                transfer, "download_ranged", return_value=self.destination
            ) as ranged:
                self.client.download_testfile(
                    "AREG", "CBCT_SemiAuto_DCM", self.destination
                )

        self.assertEqual(ranged.call_args.args[2], self.destination)
        self.assertEqual(ranged.call_args.args[3], size)
        self.assertEqual(
            ranged.call_args.kwargs["headers"]["Authorization"], "Bearer secret-token"
        )

    def test_a_small_file_is_not_worth_a_parallel_plan(self):
        response = self._streaming_response([b"x"])
        with mock.patch.object(
            requests.Session, "head", return_value=self._head_response(1024)
        ):
            with mock.patch.object(transfer, "download_ranged") as ranged:
                with mock.patch.object(requests.Session, "get", return_value=response):
                    self.client.download_testfile("AMASSS", "small.nii.gz", self.destination)

        ranged.assert_not_called()

class ToolNameSpellingTest(unittest.TestCase):
    """A rename that only moved separators must not read as a missing tool.

    Found by running every Slicer module against a live server: four of six
    named a tool the server no longer serves under that spelling, and the panel
    said "Unknown tool 'AREG'" -- the same message a typo produces. Two of those
    were splits, which no rule can fix; `SurgMovPred` -> `Surg_Mov_Pred` was
    only a rename.
    """

    def setUp(self):
        self.client = ToolServerClient("https://example.org/", "secret-token")
        self.client._tools_cache = {"Surg_Mov_Pred": {"arguments": {}}}

    def test_the_old_spelling_finds_the_renamed_tool(self):
        schema = self.client.get_tool_schema("SurgMovPred")
        self.assertEqual(schema, {"arguments": {}})

    def test_a_name_matching_nothing_is_still_unknown(self):
        with self.assertRaises(ServerToolError) as caught:
            self.client.get_tool_schema("AREG")
        self.assertIn("Unknown tool 'AREG'", str(caught.exception))
        self.assertIn("Surg_Mov_Pred", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
