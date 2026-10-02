"""Fetch a test dataset once, and only once.

Seven modules carried their own `DownloadUnzip`, and between them they had
three ways of being wrong. All three are real, all three were reproduced:

  - **Not everything published is a zip.** ALI's CBCT entry points at
    `MG_test_scan.nii.gz` and went straight into `zipfile.ZipFile`, which
    answers `BadZipFile: File is not a zip file`.
  - **A wrong link still answers 200.** ALI's IOS entry used
    `releases/tag/...` instead of `releases/download/...`; GitHub serves a
    202 KB HTML page with `Content-Type: text/html`, which the copies then
    tried to unzip. The user sees "not a zip file" and has no way to guess
    the link is wrong.
  - **The cache lies after an interrupted download.** The copies create the
    destination directory *before* downloading. Cancel, lose the network, or
    hit an error, and the empty directory stays: from then on
    `if not os.path.exists(out_path)` answers "already there" and the module
    silently works against nothing.

So: the destination is only ever created by moving a completed directory into
place, and a dataset counts as present only when it carries the marker file
written after a successful extraction.

Standard library only in the core. `ensure_with_progress` imports `qt` lazily,
so importing this module from the Conda environment stays possible.
"""
import logging
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

#: Written into the folder once extraction has finished. Its presence -- and
#: its presence alone -- means "this dataset is complete".
MARKER = ".adt-testdata-complete"

_HTML_STARTS = (b"<!doctype", b"<html", b"<?xml")


class TestDataError(RuntimeError):
    """A test dataset could not be obtained, with a reason worth showing."""


def is_present(directory):
    """Whether `directory` holds a dataset that finished downloading."""
    return os.path.isfile(os.path.join(directory, MARKER))


def _check_not_a_web_page(head, content_type, url):
    """A release link that is wrong answers with a page, not with the file."""
    if content_type and content_type.split(";")[0].strip().lower() == "text/html":
        raise TestDataError(
            "%s served a web page, not a file. A release asset link looks like "
            "releases/download/<tag>/<file>; releases/tag/<tag>/<file> is the "
            "web page and answers 200 all the same." % url)
    start = head.lstrip()[:16].lower()
    if any(start.startswith(marker) for marker in _HTML_STARTS):
        raise TestDataError("%s served a web page, not a file." % url)


def fetch(url, target, progress=None):
    """Download `url` to the file `target`. Returns `target`.

    `progress` is called with (read_bytes, total_bytes); total is 0 when the
    server does not say. No Qt here -- the caller decides what a progress
    report looks like.
    """
    with urllib.request.urlopen(url) as response:
        total = int(response.info().get("Content-Length") or 0)
        head = response.read(1024)
        _check_not_a_web_page(head, response.info().get("Content-Type"), url)
        read = len(head)
        with open(target, "wb") as out:
            out.write(head)
            if progress:
                progress(read, total)
            block = max(65536, total // 100 if total else 65536)
            while True:
                buffer = response.read(block)
                if not buffer:
                    break
                out.write(buffer)
                read += len(buffer)
                if progress:
                    progress(read, total)
    return target


def _unpack(archive, destination, url):
    """Extract an archive, or keep the file as it is when it is not one."""
    os.makedirs(destination, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
        return
    # Not an archive: the file itself is what is wanted, under its original
    # name. This is the case of MG_test_scan.nii.gz, which ALI was unzipping.
    name = os.path.basename(urllib.parse.urlsplit(url).path) or "testdata"
    shutil.move(archive, os.path.join(destination, name))


def ensure(url, root, name, progress=None):
    """The directory holding this dataset, downloading it only if missing.

    `root` is where datasets live (one directory per `name`). Returns the
    directory. A dataset already there is returned untouched -- no request is
    made, which is the whole point of pressing the button twice.
    """
    destination = os.path.join(root, name)
    if is_present(destination):
        logger.debug("%s is already downloaded, in %s", name, destination)
        return destination

    # A folder that is present but carries no marker comes from an interrupted
    # download: it is worth nothing, and keeping it would make the next run
    # believe the dataset is there.
    if os.path.isdir(destination):
        logger.info("%s was left incomplete, downloading it again", name)
        shutil.rmtree(destination, ignore_errors=True)

    os.makedirs(root, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".%s-" % name.replace(os.sep, "_"), dir=root)
    try:
        archive = os.path.join(staging, "payload")
        fetch(url, archive, progress)
        content = os.path.join(staging, "content")
        _unpack(archive, content, url)
        if os.path.exists(archive):
            os.remove(archive)
        open(os.path.join(content, MARKER), "w").close()
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        os.replace(content, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    logger.info("%s downloaded into %s", name, destination)
    return destination


def ensure_with_progress(url, root, name, parent=None, title=None):
    """`ensure`, with Slicer's modal progress dialog. Qt imported lazily."""
    if is_present(os.path.join(root, name)):
        return os.path.join(root, name)

    import qt
    dialog = qt.QProgressDialog(title or "Downloading %s..." % name,
                                "Cancel", 0, 100, parent)
    dialog.setCancelButton(None)
    dialog.setWindowModality(qt.Qt.WindowModal)
    dialog.setWindowTitle(title or "Downloading %s..." % name)
    dialog.show()

    def report(read, total):
        dialog.setValue(int(read * 100.0 / total) if total else 0)
        qt.QApplication.processEvents()

    try:
        return ensure(url, root, name, progress=report)
    finally:
        dialog.close()
