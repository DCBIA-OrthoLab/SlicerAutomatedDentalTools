"""Following SADT-VISOR's releases: which exist, which this client drives, their code.

No network: the three HTTPS answers are stood in for. What is checked is what the
client does with them -- the ordering of versions, the interface level it
accepts, what it remembers offline, and an archive that tries to escape.
"""

import io
import json
import os
import sys
import tarfile
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from SADTCoreLib import releases  # noqa: E402
from SADTCoreLib import tools_store  # noqa: E402
from SADTCoreLib.tools_store import ToolStore  # noqa: E402


def _advertisement(*tags):
    lines = [b"001e# service=git-upload-pack\n", b"0000"]
    for tag in tags:
        lines.append(b"003f" + b"a" * 40 + b" refs/tags/" + tag.encode() + b"\n")
        lines.append(b"0042" + b"b" * 40 + b" refs/tags/" + tag.encode() + b"^{}\n")
    lines.append(b"003f" + b"c" * 40 + b" refs/heads/deploy\n")
    return b"".join(lines)


def _run_tool(level):
    body = b"import os\n"
    if level is not None:
        body += b"SUPERVISOR_API = %d\n" % level
    return body


class _FakeGitHub:
    """Answers the three URLs the module asks, and counts the questions."""

    def __init__(self, tags, levels, archives=None):
        self.tags, self.levels, self.archives = tags, levels, archives or {}
        self.asked = []
        self.offline = False

    def get(self, url, timeout):
        self.asked.append(url)
        if self.offline:
            raise releases.ReleaseError("offline")
        if "info/refs" in url:
            return _advertisement(*("tools-" + tag for tag in self.tags))
        if "raw.githubusercontent" in url:
            version = url.split("/tools-")[1].split("/")[0]
            return _run_tool(self.levels.get(version))
        version = url.rsplit("/tools-", 1)[1]
        return self.archives[version]


class ReleasesTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        self._real_get = releases._get
        self.addCleanup(setattr, releases, "_get", self._real_get)

    def _github(self, tags, levels, archives=None):
        fake = _FakeGitHub(tags, levels, archives)
        releases._get = fake.get
        return fake

    # -- which releases exist ------------------------------------------------

    def test_the_tools_tags_are_read_newest_first_and_nothing_else(self):
        advertisement = _advertisement("tools-2026.11.24-2", "tools-2026.11.24-10", "tools-2026.10.02-1",
                                       "wheels-pytorch3d-0.7.9-pt2141cpu-macos", "data-2026.10.09-1")

        self.assertEqual(releases.parse_refs(advertisement),
                         ["2026.11.24-10", "2026.11.24-2", "2026.10.02-1"])

    def test_the_level_is_read_from_the_release_supervisor(self):
        self.assertEqual(releases.parse_level(_run_tool(2)), 2)
        self.assertIsNone(releases.parse_level(_run_tool(None)))

    # -- which one this client offers ----------------------------------------

    def test_the_newest_release_this_client_drives_is_the_one_offered(self):
        self._github(["2026.11.24-3", "2026.11.24-2", "2026.11.19-1"],
                     {"2026.11.24-3": 9, "2026.11.24-2": 2})

        state = releases.check(self.root)

        self.assertEqual(state["latest"], "2026.11.24-2")
        self.assertEqual(releases.latest_known(self.root), "2026.11.24-2")
        self.assertEqual(state["error"], "")

    def test_a_release_without_a_level_is_never_offered(self):
        """Older than the driven supervisor: this client cannot run it."""
        self._github(["2026.11.19-1"], {})

        self.assertIsNone(releases.check(self.root)["latest"])

    def test_a_release_level_is_asked_once(self):
        fake = self._github(["2026.11.24-2"], {"2026.11.24-2": 2})
        releases.check(self.root)
        releases.check(self.root)

        self.assertEqual(sum("raw.githubusercontent" in url for url in fake.asked), 1)

    def test_offline_the_last_release_seen_is_kept(self):
        fake = self._github(["2026.11.24-2"], {"2026.11.24-2": 2})
        releases.check(self.root)
        fake.offline = True

        state = releases.check(self.root)

        self.assertEqual(state["latest"], "2026.11.24-2")
        self.assertIn("offline", state["error"])

    def test_a_check_is_due_every_interval(self):
        self.assertTrue(releases.due(self.root))
        self._github(["2026.11.24-2"], {"2026.11.24-2": 2})
        releases.check(self.root)
        self.assertFalse(releases.due(self.root))
        state = releases.read_state(self.root)
        state["checked"] = time.time() - releases.CHECK_INTERVAL_SECONDS - 1
        with open(os.path.join(self.root, releases.STATE_FILE), "w") as handle:
            json.dump(state, handle)
        self.assertTrue(releases.due(self.root))

    # -- the code of a release -----------------------------------------------

    def _archive(self, members):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    def test_a_release_code_is_unpacked_once_under_the_tools_folder(self):
        archive = self._archive({
            "SADT-VISOR-tools-2026.11.24-2/tools/AMASSS/pyproject.toml": b"[project]\n",
            "SADT-VISOR-tools-2026.11.24-2/scripts/run_tool.py": _run_tool(2),
        })
        fake = self._github([], {}, {"2026.11.24-2": archive})

        first = releases.fetch_source(self.root, "2026.11.24-2")
        second = releases.fetch_source(self.root, "2026.11.24-2")

        self.assertEqual(first, releases.source_dir(self.root, "2026.11.24-2"))
        self.assertTrue(os.path.isfile(os.path.join(first, "tools", "AMASSS", "pyproject.toml")))
        self.assertEqual(first, second)
        self.assertEqual(sum("codeload" in url for url in fake.asked), 1)

    def test_an_archive_writing_outside_its_folder_is_refused(self):
        archive = self._archive({
            "SADT-VISOR-tools-2026.11.24-2/tools/x.txt": b"x",
            "../escaped.txt": b"x",
        })
        self._github([], {}, {"2026.11.24-2": archive})

        with self.assertRaises(releases.ReleaseError):
            releases.fetch_source(self.root, "2026.11.24-2")
        self.assertFalse(os.path.exists(os.path.join(self.root, "..", "escaped.txt")))
        self.assertFalse(os.path.isdir(releases.source_dir(self.root, "2026.11.24-2")))


class DescribeCheckTest(unittest.TestCase):
    """The line a user reads after pressing Check for updates."""

    def test_a_newer_release_is_named(self):
        text = releases.describe_check("2026.11.24-2", {"latest": "2026.11.24-10", "error": "", "checked": 0})

        self.assertIn("2026.11.24-10 is available", text)
        self.assertIn("2026.11.24-2", text)

    def test_the_same_release_is_up_to_date(self):
        text = releases.describe_check("2026.11.24-3", {"latest": "2026.11.24-3", "error": "", "checked": 0})

        self.assertTrue(text.startswith("Up to date"))

    def test_offline_says_so_and_keeps_what_was_seen(self):
        text = releases.describe_check("2026.11.24-2", {"latest": "2026.11.24-3", "error": "offline"})

        self.assertIn("Could not reach GitHub", text)
        self.assertIn("2026.11.24-3", text)


class ReleaseVersionTest(unittest.TestCase):
    """The version a Slicer runs: pinned, else the newest seen, else its own."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        import SADTCoreLib

        self.pick = SADTCoreLib.release_version

    def test_a_first_start_offline_uses_the_extension_version(self):
        self.assertEqual(self.pick(self.root), (tools_store.TOOLS_VERSION, False))

    def test_the_newest_release_seen_is_followed(self):
        with open(os.path.join(self.root, releases.STATE_FILE), "w") as handle:
            json.dump({"latest": "2026.12.01-1"}, handle)

        self.assertEqual(self.pick(self.root), ("2026.12.01-1", False))

    def test_a_study_stays_on_its_version(self):
        with open(os.path.join(self.root, releases.STATE_FILE), "w") as handle:
            json.dump({"latest": "2026.12.01-1"}, handle)

        self.assertEqual(self.pick(self.root, "2026.11.24-2"), ("2026.11.24-2", True))


class ChainModelsTest(unittest.TestCase):
    """A model removed from a tool another tool calls makes the caller not ready."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        self.store = ToolStore(self.root)
        for name, calls in (("AREG_CBCT", ["AMASSS", "ASO"]), ("AMASSS", []), ("ASO", ["ALI_CBCT"]),
                            ("ALI_CBCT", [])):
            folder = os.path.join(self.root, f"tools-{self.store.version}", "tools", name)
            os.makedirs(folder)
            open(os.path.join(folder, "pyproject.toml"), "w").close()
            with open(os.path.join(folder, tools_store.INSTALL_RECORD), "w") as handle:
                json.dump({"calls": calls, "version": self.store.version}, handle)

    def test_a_callee_without_its_model_is_named(self):
        self.store.models_missing = lambda tool: tool == "ALI_CBCT"

        self.assertEqual(self.store.chain_models_missing("AREG_CBCT"), ["ALI_CBCT"])
        self.assertEqual(self.store.chain_models_missing("AMASSS"), [])

    def test_a_chain_with_every_model_is_complete(self):
        self.store.models_missing = lambda tool: False

        self.assertEqual(self.store.chain_models_missing("AREG_CBCT"), [])


if __name__ == "__main__":
    unittest.main()
