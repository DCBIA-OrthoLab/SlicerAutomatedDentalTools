"""A local run writes its temporary files in the tools folder, not the system's.

A computer whose system disk is nearly full keeps its tools on a second disk;
the scans a tool hands the tools it calls (gigabytes for one registration) must
land on that disk too, and be gone once the run ends.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import test_hosted_test_files  # noqa: E402,F401 - importing installs the Slicer stubs

from SADTCoreLib import local_client, local_schema, slicer_io  # noqa: E402
from SADTCoreLib.tools_store import SCRATCH_FOLDER, ToolStore  # noqa: E402


class _Job:
    """Stands for run_tool's Job: writes in the temporary folder it was given."""

    log_file = "job.log"

    def __init__(self, environment):
        self.environment = environment
        for variable in ("TMPDIR", "TEMP", "TMP"):
            with open(os.path.join(environment[variable], variable + ".part"), "w") as handle:
                handle.write("a callee's output")

    def new_progress(self):
        return []

    def poll(self):
        return 0

    def result(self):
        return {}


class _RunTool:
    def __init__(self):
        self.jobs = []
        self.job_dirs = []

    def start(self, target, params, job_dir, env=None, data_dir=None):
        self.job_dirs.append(job_dir)
        self.jobs.append(_Job(env))
        return self.jobs[-1]


class ScratchFolderTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.store = ToolStore(os.path.join(self.root, "sadt-tools"))
        os.makedirs(self.store.data_dir)

    def _client(self):
        client = local_client.LocalToolClient(environment=lambda: {"PATH": "/bin"}, store=self.store)
        fake = _RunTool()
        client._supervisor = lambda _tool: fake
        client.get_tool_schema = lambda _tool: {}
        client._tool_folders = lambda _tool=None: {"AMASSS": self.root}
        return client, fake

    def test_a_run_gets_a_temporary_folder_inside_the_tools_folder_and_loses_it_after(self):
        client, fake = self._client()
        output = os.path.join(self.root, "results")
        with mock.patch.object(local_schema, "resolve_facade", lambda name, values: (name, values)), \
                mock.patch.object(local_schema, "wire_arguments", lambda *a: {}):
            client.run("AMASSS", output_dir=output)

        environment = fake.jobs[0].environment
        scratch = os.path.join(self.store.root, SCRATCH_FOLDER)
        self.assertEqual(environment["TMPDIR"], environment["TEMP"])
        self.assertEqual(environment["TMPDIR"], environment["TMP"])
        self.assertEqual(os.path.dirname(environment["TMPDIR"]), scratch)
        self.assertEqual(os.path.dirname(fake.job_dirs[0]), scratch)
        self.assertFalse(os.path.exists(environment["TMPDIR"]))

    def test_the_scene_inputs_are_exported_inside_the_tools_folder(self):
        client, _fake = self._client()

        with slicer_io.TempWorkspace(dir=client.scratch_dir()) as workspace:
            path = workspace.path
            self.assertEqual(os.path.dirname(path), self.store.scratch_dir)
        self.assertFalse(os.path.exists(path))

    def test_without_a_store_the_system_folder_is_kept(self):
        client = local_client.LocalToolClient(environment=lambda: {}, store=None)

        self.assertIsNone(client.scratch_dir())
        folder = client._scratch("sadt_test_")
        self.addCleanup(shutil.rmtree, folder, True)
        self.assertEqual(os.path.dirname(folder), tempfile.gettempdir())

    def test_what_a_crashed_run_left_is_removed_and_a_running_one_kept(self):
        old = self.store.scratch("run_AMASSS_")
        recent = self.store.scratch("run_ALI_CBCT_")
        long_ago = time.time() - 3 * 24 * 3600
        os.utime(old, (long_ago, long_ago))

        self.assertEqual(self.store.clean_scratch(), 1)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.isdir(recent))


if __name__ == "__main__":
    unittest.main()
