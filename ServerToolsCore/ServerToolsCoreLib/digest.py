"""What changed in a folder, by CONTENT rather than by timestamp.

A quality-control checkpoint puts a step's whole output in front of a reader,
and a step of AMASSS is a cohort of segmentations. The reader moves one point
in one landmark file -- eight kilobytes -- and the run has to carry on with
that correction. Sending the step back whole to deliver it is the expensive
half of the feature, and it is the half this module removes: digest the tree
when it is unpacked, digest it again when the reader presses Continue, and
send only what differs.

**Why a hash and not a stat.** Slicer rewrites a file whenever it is saved,
and the reviewer saves on the way OUT of a patient rather than only when
something moved -- so mtime and even size move for files whose bytes are
identical. Trusting either would send the cohort back after all, which is the
behaviour this replaces. The bytes are the only thing that answers the
question that is actually being asked.

Deliberately free of `qt` and `slicer`: this is a walk over a directory, it is
what the panel's correctness now rests on, and it is tested without either.
"""

import hashlib
import logging
import os

logger = logging.getLogger("ServerToolsCore.digest")

# Read size. A step is hundreds of megabytes and the reader waits through two
# passes over it, so files are never read whole into memory -- a cohort of
# CBCT volumes would be the panel's resident set.
_READ_BYTES = 4 * 1024 * 1024

# blake2b truncated to 128 bits, not sha256: nothing here is a security
# property -- no adversary picks these files, and the answer is only ever
# "same bytes or not". blake2b is the fastest hash the standard library
# offers on a machine with no SHA extensions, which is the machine a
# clinician reviews on, and 128 bits is far past the point where an
# accidental collision is the reason a correction went missing.
_DIGEST_BYTES = 16


def digest_tree(folder: str) -> dict:
    """{path relative to `folder`: a digest of that file's bytes}.

    Every file under `folder`, at any depth: a step's output mirrors its
    input tree, so a patient's correction can be two directories down.

    A file that cannot be read is left OUT rather than recorded as empty, and
    the asymmetry that produces is the safe one. Unreadable when the baseline
    is taken and readable afterwards reads as new, so it is sent; unreadable
    when the reader continues means it never reaches `changed_since`, so it is
    not sent -- and the server treats a file it was not sent as unchanged,
    which is what it was.
    """
    if not folder or not os.path.isdir(folder):
        return {}
    marks = {}
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            mark = digest_file(path)
            if mark is not None:
                marks[os.path.relpath(path, folder)] = mark
    return marks


def digest_file(path: str):
    """The digest of one file's bytes, or None if it could not be read."""
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(_READ_BYTES)
                if not block:
                    break
                hasher.update(block)
    except OSError as exc:
        logger.warning("Could not digest %s: %s", os.path.basename(path), exc)
        return None
    return hasher.hexdigest()


def changed_since(baseline: dict, folder: str) -> list:
    """The paths under `folder`, relative to it, whose bytes are not `baseline`'s.

    New files count; a file whose digest still matches does not. **A file that
    has gone does not appear at all, and that is deliberate** -- the server
    lays a correction OVER the step's output file by file, so an absent file
    reads as "unchanged" there, and a partial set could not be told from
    "delete everything I did not send". A reader who wants a file gone deletes
    it from the result, not from the checkpoint.

    Sorted, so the same review always builds the same archive.
    """
    now = digest_tree(folder)
    return sorted(path for path, mark in now.items() if baseline.get(path) != mark)


def paths_under(paths, folder: str, subfolder: str) -> list:
    """Those of `paths` -- relative to `folder` -- that live under `subfolder`,
    renamed relative to `subfolder` instead.

    What this is for: the checkpoint is digested once, as one tree, but it is
    sent back one archive per STEP, and each archive's member names have to be
    relative to that step's own directory or the server unpacks them a level
    too deep. `subfolder` equal to `folder` is the single-step shape, where the
    server's zip flattened the lone step to the archive root and there is
    nothing to strip.
    """
    inside = os.path.relpath(subfolder, folder)
    if inside == os.curdir:
        return list(paths)
    head = inside + os.sep
    return [path[len(head):] for path in paths if path.startswith(head)]
