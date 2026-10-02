"""Find the files of a folder by extension.

`search` existed in fifteen copies, in six variants. All of them now come down
to this one, and the only two substantive differences became parameters:

- six were a `search(self, ...)` method, three the same thing as a free
  function -- identical byte for byte once the `self` is removed;
- three wrapped the result in a `sorted()`, hence `sort`. The order in which
  patients are traversed depends on it in those;
- the one in `ASO_IOS_utils/data_file.py` additionally discarded anything that
  is not a file, hence `files_only`. A folder whose name ends with the
  extension being looked for -- `patient.nrrd/` -- is counted as a scan by the
  other fourteen. This is probably a defect everywhere, but nobody has ever
  seen its effect, so the original behaviour stays the default and only the
  caller that asked for the filter keeps getting it.

The VFACE one did not lay the result out the same way -- a single walk of the
tree, sorted, then split by key -- but returns exactly what `sort=True` returns.

Standard library only: called from the Conda environment.
"""
import glob
import os


def search(path, *args, sort=False, files_only=False):
    """The files of `path` grouped by requested extension.

    Returns a dictionary whose every key is an item of `args` and whose value
    is the list of files under `path` that end with that key. A list passed in
    `args` is flattened.

        search(path, 'json', ['.nii.gz', '.nrrd'])
        {'json': ['path/a.json', ...], '.nii.gz': [...], '.nrrd': [...]}

    `sort` returns each list sorted: three of the fifteen original sites did
    so, and the order in which patients are traversed depends on it in those.

    `files_only` discards the directories whose name ends with the key: a
    single original site did so.
    """
    arguments = []
    for arg in args:
        if isinstance(arg, list):
            arguments.extend(arg)
        else:
            arguments.append(arg)

    # An empty path gave the pattern `/**/*`: glob started over from the root
    # of the disk and Slicer froze. It is reached by running AREG IOS with the
    # "Registration Model Folder" field empty (AREG_Method/IOS.py). A
    # non-existent folder already gave an empty result -- glob finds nothing
    # there -- so the same contract is applied here, without walking anything
    # at all.
    if not path or not os.path.isdir(path):
        return {key: [] for key in arguments}

    entries = list(
        glob.iglob(os.path.normpath("/".join([path, "**", "*"])), recursive=True)
    )
    if files_only:
        entries = [entry for entry in entries if os.path.isfile(entry)]
    if sort:
        entries.sort()

    return {key: [entry for entry in entries if entry.endswith(key)] for key in arguments}
