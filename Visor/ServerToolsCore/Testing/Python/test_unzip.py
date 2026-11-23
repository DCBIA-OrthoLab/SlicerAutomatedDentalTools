"""slicer_io.unzip_folder: an archive from the server is untrusted.

It returns only the paths extraction really wrote, refuses any entry that would
land outside the destination (absolute, `..`, a symbolic link, a symlinked
destination), and refuses an archive past the size or entry-count ceiling
before a byte is written.
"""

import os
import shutil
import stat
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import test_hosted_test_files as fixtures  # noqa: F401,E402 - installs the stubs

from ServerToolsCoreLib import config, slicer_io  # noqa: E402
from ServerToolsCoreLib.errors import ServerToolError  # noqa: E402


class UnzipFolderTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="unzip_test_")
        self.addCleanup(shutil.rmtree, self.work, True)
        self.dest = os.path.join(self.work, "dest")
        self.archive = os.path.join(self.work, "result.zip")

    def _zip(self, entries):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in entries:
                archive.writestr(name, data)
        return self.archive

    def _nothing_written(self):
        return not os.path.exists(self.dest) or not os.listdir(self.dest)

    def test_the_files_written_are_what_it_returns(self):
        self._zip([("a/scan.nii.gz", b"1"), ("report.json", b"{}"), ("a/", b"")])
        written = slicer_io.unzip_folder(self.archive, self.dest)
        real = os.path.realpath(self.dest)
        self.assertEqual(sorted(written), sorted([os.path.join(real, "a", "scan.nii.gz"),
                                                  os.path.join(real, "report.json")]))
        for path in written:
            self.assertTrue(os.path.isfile(path))

    def test_an_entry_that_climbs_out_is_refused_before_anything_is_written(self):
        for name in ("../escaped.txt", "a/../../escaped.txt", "/etc/escaped.txt",
                     "C:/escaped.txt", "..\\escaped.txt"):
            with self.subTest(name=name):
                self._zip([("fine.txt", b"ok"), (name, b"x")])
                with self.assertRaises(ServerToolError):
                    slicer_io.unzip_folder(self.archive, self.dest)
                self.assertTrue(self._nothing_written())
                self.assertFalse(os.path.exists(os.path.join(self.work, "escaped.txt")))

    def test_a_symbolic_link_entry_is_refused(self):
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self._zip([(link, "/etc/passwd")])
        with self.assertRaises(ServerToolError):
            slicer_io.unzip_folder(self.archive, self.dest)
        self.assertTrue(self._nothing_written())

    def test_an_entry_through_a_symlinked_folder_is_refused(self):
        outside = os.path.join(self.work, "outside")
        os.makedirs(outside)
        os.makedirs(self.dest)
        try:
            os.symlink(outside, os.path.join(self.dest, "sub"))
        except OSError as exc:
            # Windows without Developer Mode or the symlink privilege: the
            # fixture cannot be built, which says nothing about the unzip.
            self.skipTest(f"cannot create a symlink here: {exc}")
        self._zip([("sub/escaped.txt", b"x")])
        with self.assertRaises(ServerToolError):
            slicer_io.unzip_folder(self.archive, self.dest)
        self.assertEqual(os.listdir(outside), [])

    def test_too_many_entries_is_refused(self):
        self._zip([(f"f{i}.txt", b"") for i in range(5)])
        with mock.patch.object(config, "UNZIP_MAX_MEMBERS", 4):
            with self.assertRaises(ServerToolError):
                slicer_io.unzip_folder(self.archive, self.dest)
        self.assertTrue(self._nothing_written())

    def test_too_many_bytes_is_refused_from_the_directory_alone(self):
        self._zip([("big.bin", b"\0" * 10_000)])
        with mock.patch.object(config, "UNZIP_MAX_BYTES", 9_999):
            with self.assertRaises(ServerToolError):
                slicer_io.unzip_folder(self.archive, self.dest)
        self.assertTrue(self._nothing_written())

    def test_the_ceilings_are_generous(self):
        self.assertGreaterEqual(config.UNZIP_MAX_BYTES, 20 * 1024 ** 3)
        self.assertGreaterEqual(config.UNZIP_MAX_MEMBERS, 100_000)


if __name__ == "__main__":
    unittest.main()
