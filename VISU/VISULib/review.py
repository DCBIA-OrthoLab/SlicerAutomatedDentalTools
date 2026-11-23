"""Which patients a reader marked as needing work, and where that is kept.

The one thing a reviewer produces that is not a corrected file: a list. Six
cases look fine, two do not, and what happens next -- re-running those two,
handing them to someone else, coming back tomorrow -- depends on the list
surviving the session.

**It is written beside the data, not into a preference store.** A mark is
about the cohort, not about the person who made it: whoever opens the folder
next should see what was already judged, including on another machine. A
QSettings key would have made two readers of one folder disagree silently.

The file is ours and its name says so. It is never confused with a tool's
output: `index` skips it like every other report.
"""

import json
import os

FILENAME = "visu-review.json"

# The shape written. A version, because the next thing this file wants to
# carry is a note per patient, and a reader that predates that must not
# choke on it.
VERSION = 1


def path_for(folder: str) -> str:
    return os.path.join(folder, FILENAME)


def load(folder: str) -> set:
    """The flagged case keys, or an empty set for a folder with no marks."""
    try:
        with open(path_for(folder), encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        # A folder nobody has reviewed, one on a read-only mount, or a file
        # something else wrote. None of the three is an error here: the
        # reader starts with nothing marked.
        return set()
    flagged = document.get("flagged")
    return {str(key) for key in flagged} if isinstance(flagged, list) else set()


def save(folder: str, flagged) -> bool:
    """Write the marks beside the data. False when the folder will not take it.

    Not raising: a hosted sample is unpacked into a temporary directory that
    is deleted on the next download, and a cohort on a read-only share is
    perfectly normal. Losing the marks is worth a line in the panel, never a
    dialog over a scan.
    """
    document = {"version": VERSION, "flagged": sorted(flagged)}
    staging = path_for(folder) + ".visu-tmp"
    try:
        with open(staging, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
        os.replace(staging, path_for(folder))
    except OSError:
        try:
            os.remove(staging)
        except OSError:
            pass
        return False
    return True


def as_text(flagged, folder: str = "") -> str:
    """The list as something to paste into a message or a ticket."""
    marked = sorted(flagged)
    if not marked:
        return "Nothing flagged."
    head = f"{len(marked)} flagged"
    if folder:
        head += f" in {folder}"
    return "\n".join([head + ":"] + [f"  {key}" for key in marked])
