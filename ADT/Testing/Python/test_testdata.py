# What the shared test-data fetcher does, and the three defects it exists for.
#
# All three were reproduced against the real links before this was written:
# a .nii.gz handed to zipfile (ALI CBCT), a releases/tag/ link that serves an
# HTML page with code 200 (ALI IOS), and a destination directory created
# before the download so an interruption leaves a cache that lies.
#
# No network: every case goes through a file:// URL.
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from ADTLib.testdata import (  # noqa: E402
    MARKER, TestDataError, ensure, is_present)


def _url(path):
    return "file://" + path


class TestDataTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="adt-testdata-")
        self.root = os.path.join(self.tmp, "downloads")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _zip(self, name="payload.zip", member="scan.nii.gz", body=b"x" * 32):
        path = os.path.join(self.tmp, name)
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(member, body)
        return path

    def _file(self, name, body=b"\x1f\x8b" + b"y" * 64):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(body)
        return path

    # ---------------------------------------------------------------- zip

    def test_an_archive_is_extracted(self):
        out = ensure(_url(self._zip()), self.root, "Set")
        self.assertTrue(os.path.isfile(os.path.join(out, "scan.nii.gz")))
        self.assertTrue(is_present(out))

    def test_the_marker_is_what_says_it_is_complete(self):
        out = ensure(_url(self._zip()), self.root, "Set")
        os.remove(os.path.join(out, MARKER))
        self.assertFalse(is_present(out))

    # -------------------------------------------------- not an archive

    def test_a_plain_file_is_kept_whole(self):
        """ALI CBCT: MG_test_scan.nii.gz was handed to zipfile.ZipFile."""
        out = ensure(_url(self._file("MG_test_scan.nii.gz")), self.root, "Scan")
        self.assertTrue(os.path.isfile(os.path.join(out, "MG_test_scan.nii.gz")))
        self.assertTrue(is_present(out))

    # --------------------------------------------------- an HTML page

    def test_a_web_page_is_refused_with_a_reason(self):
        """ALI IOS: releases/tag/... serves a page and answers 200."""
        page = self._file("v1.0.4.html", b"<!DOCTYPE html>\n<html><body>404</body></html>")
        with self.assertRaises(TestDataError) as caught:
            ensure(_url(page), self.root, "Set")
        self.assertIn("releases/download", str(caught.exception))

    def test_nothing_is_left_behind_when_it_fails(self):
        page = self._file("v1.0.4.html", b"<!DOCTYPE html><html></html>")
        with self.assertRaises(TestDataError):
            ensure(_url(page), self.root, "Set")
        self.assertFalse(os.path.exists(os.path.join(self.root, "Set")))
        self.assertEqual([e for e in os.listdir(self.root)], [],
                         "a working directory was left behind")

    # --------------------------------------------- already downloaded

    def test_a_dataset_already_there_is_not_downloaded_again(self):
        source = self._zip()
        first = ensure(_url(source), self.root, "Set")
        os.remove(source)                       # the source is gone
        again = ensure(_url(source), self.root, "Set")   # must ask for nothing
        self.assertEqual(first, again)
        self.assertTrue(os.path.isfile(os.path.join(again, "scan.nii.gz")))

    def test_an_interrupted_download_is_not_taken_for_a_complete_one(self):
        """The defect of the seven copies: the empty folder passed for the
        data set."""
        half = os.path.join(self.root, "Set")
        os.makedirs(half)
        self.assertFalse(is_present(half))
        out = ensure(_url(self._zip()), self.root, "Set")
        self.assertTrue(os.path.isfile(os.path.join(out, "scan.nii.gz")))

    def test_a_directory_left_by_an_interruption_is_replaced_not_merged(self):
        half = os.path.join(self.root, "Set")
        os.makedirs(half)
        open(os.path.join(half, "half.tmp"), "w").close()
        out = ensure(_url(self._zip()), self.root, "Set")
        self.assertFalse(os.path.exists(os.path.join(out, "half.tmp")))

    # --------------------------------------------------------- progress

    def test_progress_is_reported(self):
        seen = []
        ensure(_url(self._zip()), self.root, "Set", progress=lambda r, t: seen.append((r, t)))
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], os.path.getsize(os.path.join(self.tmp, "payload.zip")))


if __name__ == "__main__":
    unittest.main()
