"""tools_store: where the shared tools folder is, what needs what, what may be removed.

Plain unittest, no Slicer: installed tools are faked by their install record,
which is all the Manager reads.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ServerToolsCoreLib import tools_store  # noqa: E402
from ServerToolsCoreLib.tools_store import InstallError, ToolStore  # noqa: E402


def _fake_install(store, name, calls, version=None):
    folder = os.path.join(store.root, f"tools-{version or store.version}", "tools", name)
    os.makedirs(folder)
    open(os.path.join(folder, "pyproject.toml"), "w").close()
    with open(os.path.join(folder, tools_store.INSTALL_RECORD), "w") as handle:
        json.dump({"calls": calls}, handle)
    return folder


class LocationTest(unittest.TestCase):
    def test_the_development_variable_wins(self):
        os.environ[tools_store.ROOT_VARIABLE] = "/somewhere"
        try:
            self.assertEqual(tools_store.resolve_root("/chosen")[0], "/somewhere")
        finally:
            del os.environ[tools_store.ROOT_VARIABLE]

    def test_a_choice_for_this_slicer_beats_the_shared_default(self):
        os.environ.pop(tools_store.ROOT_VARIABLE, None)
        self.assertEqual(tools_store.resolve_root("/chosen")[0], "/chosen")
        self.assertEqual(tools_store.resolve_root("")[0], tools_store.default_root())

    def test_the_default_is_one_folder_per_user_named_for_what_it_holds(self):
        self.assertEqual(os.path.basename(tools_store.default_root()), "sadt-tools")


class InstalledToolsTest(unittest.TestCase):
    def setUp(self):
        self.store = ToolStore(tempfile.mkdtemp())
        _fake_install(self.store, "ALI_CBCT", [])
        _fake_install(self.store, "ASO", ["ALI_CBCT"])
        _fake_install(self.store, "TMJ_Crop", ["ASO", "AMASSS"])
        _fake_install(self.store, "AMASSS", [])

    def test_a_tool_is_needed_by_its_callers_and_theirs(self):
        self.assertEqual(self.store.needed_by("ALI_CBCT"), ["ASO", "TMJ_Crop"])
        self.assertEqual(self.store.needed_by("TMJ_Crop"), [])

    def test_an_old_version_is_listed_but_needs_nothing(self):
        _fake_install(self.store, "ASO", ["ALI_CBCT"], version="2026.01.01-1")
        old = [row for row in self.store.installed() if not row["current"]]
        self.assertEqual([row["name"] for row in old], ["ASO"])
        self.assertEqual(self.store.remove_unused_versions(), ["tools-2026.01.01-1"])

    def test_only_an_installed_tool_of_this_folder_can_be_removed(self):
        with self.assertRaises(InstallError):
            self.store.remove_tool(tempfile.mkdtemp())
        path = self.store.tool_dir("ASO")
        self.store.remove_tool(path)
        self.assertFalse(self.store.is_installed("ASO"))

    def test_every_slicer_that_uses_the_folder_is_recorded(self):
        self.store.register_user("/apps/slicer/Prod", "5.12")
        self.store.register_user("/apps/slicer/Dev", "5.13")
        self.assertEqual(sorted(self.store.users()), ["/apps/slicer/Dev", "/apps/slicer/Prod"])
        self.assertTrue(os.path.isfile(os.path.join(self.store.root, "README.txt")))


class RemovalTest(unittest.TestCase):
    """What leaves with a tool: the tools it brought that nothing else needs, and their data."""

    def setUp(self):
        self.store = ToolStore(tempfile.mkdtemp())
        _fake_install(self.store, "ALI_CBCT", [])
        _fake_install(self.store, "ALI_IOS", ["Crown_Seg"])
        _fake_install(self.store, "Crown_Seg", [])
        _fake_install(self.store, "ASO", ["ALI_CBCT"])
        _fake_install(self.store, "TMJ_Crop", ["ASO"])

    def test_a_tool_takes_along_what_only_it_needed_and_its_module(self):
        self.assertEqual(self.store.removable_with("TMJ_Crop"),
                         ["ALI_CBCT", "ALI_IOS", "ASO", "Crown_Seg"])

    def test_a_tool_another_one_still_needs_stays(self):
        _fake_install(self.store, "VFACE", ["ASO"])
        self.assertEqual(self.store.removable_with("TMJ_Crop"), [])

    def test_data_a_kept_tool_uses_is_not_offered(self):
        models = os.path.join(self.store.data_dir, "ALI", "models")
        os.makedirs(os.path.join(models, "ALI_CBCT_Models"))
        os.makedirs(os.path.join(models, "ALI_IOS_Models"))
        rows = self.store.data_of(["ALI_CBCT"], keep=["ALI_IOS"])
        self.assertEqual([(row["name"], row["used_by"]) for row in rows], [("ALI_CBCT_Models", [])])


if __name__ == "__main__":
    unittest.main()


class VersionOrderTest(unittest.TestCase):
    def test_versions_compare_as_numbers(self):
        key = tools_store.version_key
        self.assertLess(key("2026.11.19-2"), key("2026.11.19-10"))
        self.assertLess(key("2026.9.1-1"), key("2026.10.1-1"))

    def test_the_newest_version_is_picked_numerically(self):
        store = ToolStore(tempfile.mkdtemp(), version="2026.11.19-1")
        for version in ("2026.11.19-2", "2026.11.19-10"):
            _fake_install(store, "ASO", [], version=version)
        self.assertEqual(store.versions(), ["2026.11.19-2", "2026.11.19-10"])
        self.assertEqual(store.installed_version("ASO"), "2026.11.19-10")

    def test_a_newer_version_is_neither_old_nor_an_update(self):
        store = ToolStore(tempfile.mkdtemp(), version="2026.11.19-1")
        _fake_install(store, "ASO", [], version="2026.12.01-1")
        _fake_install(store, "AMASSS", [], version="2026.01.01-1")
        self.assertFalse(store.update_available("ASO"))
        self.assertTrue(store.update_available("AMASSS"))
        self.assertEqual(store.remove_unused_versions(), ["tools-2026.01.01-1"])
        self.assertTrue(os.path.isdir(os.path.join(store.root, "tools-2026.12.01-1")))


class StoreLockTest(unittest.TestCase):
    def test_a_second_holder_is_refused_while_the_first_is_alive(self):
        root = tempfile.mkdtemp()
        with open(os.path.join(root, tools_store.StoreLock.NAME), "w") as handle:
            handle.write(str(os.getppid()))  # a live process that is not this one
        with self.assertRaises(tools_store.StoreBusy):
            with ToolStore(root).lock():
                pass

    def test_a_lock_left_by_a_dead_process_is_taken_over(self):
        root = tempfile.mkdtemp()
        with open(os.path.join(root, tools_store.StoreLock.NAME), "w") as handle:
            handle.write("999999999")
        with ToolStore(root).lock():
            pass
        self.assertFalse(os.path.exists(os.path.join(root, tools_store.StoreLock.NAME)))

    def test_another_thread_of_this_slicer_is_refused(self):
        import threading

        store = ToolStore(tempfile.mkdtemp())
        refused = []

        def other():
            try:
                with store.lock():
                    pass
            except tools_store.StoreBusy:
                refused.append(True)

        with store.lock():
            with store.lock():  # the same thread may nest (an install pruning)
                pass
            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
        self.assertEqual(refused, [True])


class InstallTest(unittest.TestCase):
    """An update that fails leaves the tool that worked."""

    def setUp(self):
        self.source = tempfile.mkdtemp()
        origin = os.path.join(self.source, "tools", "Demo")
        os.makedirs(os.path.join(origin, "src", "sadt_demo"))
        with open(os.path.join(origin, "pyproject.toml"), "w") as handle:
            handle.write("[project]\nname = 'demo'\n")
        with open(os.path.join(origin, "src", "sadt_demo", "__init__.py"), "w") as handle:
            handle.write("VERSION = 2\n")
        self.store = ToolStore(tempfile.mkdtemp(), source=self.source)
        self.target = _fake_install(self.store, "Demo", [])
        with open(os.path.join(self.target, "old.txt"), "w") as handle:
            handle.write("the version that worked")
        self.saved = (tools_store._calls, ToolStore._install_scripts, ToolStore._uv_sync,
                      ToolStore._install_data, ToolStore.prune_cache)
        tools_store._calls = lambda source, folder: []
        ToolStore._install_scripts = lambda self, source: None
        ToolStore._install_data = lambda self, *a, **k: None
        ToolStore.prune_cache = lambda self: 0

    def tearDown(self):
        (tools_store._calls, ToolStore._install_scripts, ToolStore._uv_sync,
         ToolStore._install_data, ToolStore.prune_cache) = self.saved

    def test_a_failed_update_puts_the_previous_tool_back(self):
        def fail(self, *args, **kwargs):
            raise InstallError("no network")

        ToolStore._uv_sync = fail
        self.assertTrue(self.store.outdated("Demo", self.source))
        with self.assertRaises(InstallError):
            self.store.install("Demo", self.source, "")
        self.assertTrue(os.path.isfile(os.path.join(self.target, "old.txt")))
        self.assertTrue(self.store.is_installed("Demo"))
        self.assertFalse(os.path.exists(self.target + ".previous"))

    def test_a_successful_update_replaces_it(self):
        ToolStore._uv_sync = lambda self, *args, **kwargs: None
        self.assertEqual(self.store.install("Demo", self.source, ""), ["Demo"])
        self.assertFalse(os.path.exists(os.path.join(self.target, "old.txt")))
        self.assertFalse(self.store.outdated("Demo", self.source))
        self.assertFalse(os.path.exists(self.target + ".previous"))

    def test_removal_deletes_the_record_first_and_frees_the_lock(self):
        self.store.remove_tool(self.target)
        self.assertFalse(os.path.exists(self.target))
        self.assertFalse(os.path.exists(os.path.join(self.store.root, tools_store.StoreLock.NAME)))


def _lock(packages):
    """A uv.lock whose packages each have one wheel that installs anywhere."""
    blocks = ['version = 1\n']
    for name, version, size in packages:
        wheel = f"{name}-{version}-py3-none-any.whl"
        blocks.append(f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
                      f'wheels = [\n    {{ url = "https://x/{wheel}", hash = "h", size = {size} }},\n]\n')
    blocks.append('[[package]]\nname = "tool"\nversion = "1"\nsource = { editable = "." }\n')
    return "\n".join(blocks)


class EstimateTest(unittest.TestCase):
    """What an install says it takes: a package on disk already, or shared, counts once."""

    def setUp(self):
        self.source = tempfile.mkdtemp()
        for tool, packages in (("One", [("torch", "2.11.0+cu128", 1000), ("Small-Lib", "1.0", 10)]),
                               ("Two", [("torch", "2.11.0+cu128", 1000), ("other", "3", 5)])):
            folder = os.path.join(self.source, "tools", tool)
            os.makedirs(folder)
            with open(os.path.join(folder, "pyproject.toml"), "w") as handle:
                handle.write('[project]\nrequires-python = ">=3.11,<3.12"\n')
            with open(os.path.join(folder, "uv.lock"), "w") as handle:
                handle.write(_lock(packages))
        self.store = ToolStore(tempfile.mkdtemp(), source=self.source)
        self.saved = (ToolStore.incomplete, ToolStore.missing, ToolStore._manifest_download)
        ToolStore.incomplete = ToolStore.missing = lambda self, name, source: ["One", "Two"]
        ToolStore._manifest_download = lambda self, tool, kind, counted: 0

    def tearDown(self):
        ToolStore.incomplete, ToolStore.missing, ToolStore._manifest_download = self.saved

    def test_the_lock_names_each_package_as_its_dist_info_does(self):
        packages = tools_store.lock_packages(os.path.join(self.source, "tools", "One", "uv.lock"), "cp311")
        self.assertEqual(packages, {"torch-2.11.0+cu128": 1000, "small_lib-1.0": 10})

    def test_a_package_two_tools_share_is_counted_once(self):
        rows = self.store.estimate("One", self.source, "")["tools"]
        self.assertEqual([row["environment"] for row in rows], [1010, 5])

    def test_a_package_already_in_the_cache_costs_nothing(self):
        archive = os.path.join(self.store.uv_cache, "archive-v0", "abc")
        os.makedirs(os.path.join(archive, "torch-2.11.0+cu128.dist-info"))
        os.makedirs(os.path.join(archive, "Small_Lib-1.0.dist-info"))
        self.assertEqual(self.store.estimate("One", self.source, "")["environment"], 5)


class UvDownloadTest(unittest.TestCase):
    """The store fetches its own uv: a clinician installs nothing by hand."""

    def _archive(self, folder):
        import hashlib
        import io
        import tarfile

        archive = os.path.join(folder, "uv-test.tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            data = b"#!/bin/sh\necho uv\n"
            member = tarfile.TarInfo("uv-test/uv")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        with open(archive, "rb") as handle:
            return archive, hashlib.sha256(handle.read()).hexdigest()

    def test_a_checked_archive_gives_an_executable_uv(self):
        folder = tempfile.mkdtemp()
        archive, digest = self._archive(folder)
        destination = os.path.join(folder, "bin", "uv")
        tools_store._download_uv(("uv-test.tar.gz", digest), destination, url="file://" + archive)
        self.assertTrue(os.access(destination, os.X_OK))

    def test_a_damaged_archive_is_refused_and_leaves_nothing(self):
        folder = tempfile.mkdtemp()
        archive, _digest = self._archive(folder)
        destination = os.path.join(folder, "bin", "uv")
        with self.assertRaises(InstallError):
            tools_store._download_uv(("uv-test.tar.gz", "0" * 64), destination, url="file://" + archive)
        self.assertEqual(os.listdir(os.path.dirname(destination)), [])

    def test_slicer_under_rosetta_still_gets_the_apple_silicon_tools(self):
        from unittest import mock

        answer = mock.Mock(stdout="1\n")
        with mock.patch.object(tools_store.platform, "system", return_value="Darwin"), \
                mock.patch.object(tools_store.platform, "machine", return_value="x86_64"), \
                mock.patch.object(tools_store.subprocess, "run", return_value=answer):
            self.assertEqual(tools_store._uv_platform(), ("Darwin", "arm64"))
            self.assertIn("arm64", tools_store._platform_tags())

    def test_every_platform_a_clinician_uses_has_a_release(self):
        for system, machine in (("Darwin", "arm64"), ("Darwin", "x86_64"), ("Windows", "x86_64"),
                                ("Linux", "x86_64")):
            self.assertIn((system, machine), tools_store.UV_RELEASES)


class ManifestTargetTest(unittest.TestCase):
    def test_an_extracted_archive_lands_without_its_zip_suffix(self):
        target = tools_store._entry_target
        self.assertEqual(target({"name": "AMASSS_Models.zip", "extract": True}), "AMASSS_Models")
        self.assertEqual(target({"name": "model.pth"}), "model.pth")
        self.assertEqual(target({"name": "a.zip", "dest": "nnunet/one"}), os.path.join("nnunet", "one"))

    def test_a_network_failure_is_said_in_words(self):
        self.assertIn("internet", tools_store._network_hint("tunnel error: failed to connect"))
        self.assertEqual(tools_store._network_hint("No space left on device"), "")


class RecoverTest(unittest.TestCase):
    def test_a_tool_moved_aside_by_an_update_cut_short_is_put_back(self):
        store = ToolStore(tempfile.mkdtemp())
        target = _fake_install(store, "Demo", [])
        os.replace(target, target + tools_store.PREVIOUS_SUFFIX)
        os.makedirs(target)  # the successor, half built, no record
        self.assertEqual([row["name"] for row in store.installed()], [])
        self.assertEqual(store.recover(), ["Demo"])
        self.assertTrue(store.is_installed("Demo"))
        self.assertFalse(os.path.exists(target + tools_store.PREVIOUS_SUFFIX))

    def test_a_finished_update_only_loses_the_old_copy(self):
        store = ToolStore(tempfile.mkdtemp())
        target = _fake_install(store, "Demo", [])
        os.makedirs(target + tools_store.PREVIOUS_SUFFIX)
        self.assertEqual(store.recover(), [])
        self.assertTrue(store.is_installed("Demo"))
        self.assertFalse(os.path.exists(target + tools_store.PREVIOUS_SUFFIX))


class ModelSizesTest(unittest.TestCase):
    def test_sizes_are_kept_outside_the_folders_tools_read(self):
        data = os.path.join(tempfile.mkdtemp(), "DATA")
        models = os.path.join(data, "ASO", "models")
        os.makedirs(os.path.join(models, "Bundle"))
        with open(os.path.join(models, "Bundle", "ref.mrk.json"), "w") as handle:
            handle.write("{}")
        tools_store._record_model_size(models, "Bundle")
        self.assertEqual(sorted(os.listdir(models)), ["Bundle"])
        self.assertEqual(tools_store._model_sizes(models), {"Bundle": 2})

    def test_sizes_written_inside_by_an_earlier_pilot_are_moved_out(self):
        data = os.path.join(tempfile.mkdtemp(), "DATA")
        models = os.path.join(data, "ASO", "models")
        os.makedirs(models)
        with open(os.path.join(models, tools_store.MODEL_SIZES), "w") as handle:
            json.dump({"Bundle": 7}, handle)
        self.assertEqual(tools_store.migrate_model_sizes(data), 1)
        self.assertEqual(os.listdir(models), [])
        self.assertEqual(tools_store._model_sizes(models), {"Bundle": 7})
