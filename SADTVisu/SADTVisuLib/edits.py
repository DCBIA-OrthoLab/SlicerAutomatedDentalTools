"""Writing back what a reader changed, without disturbing what they did not.

A landmark file is not ours. ALI wrote it, it carries descriptions, per-point
locks, a display block and fields this module has never heard of -- and a
reader who dragged one point expects to find the other hundred exactly as
they were. So nothing here SERIALISES a markups node: it reads the document
that is on disk, replaces the positions that actually moved, and writes the
same document back.

**The frame is the trap.** A `.mrk.json` declares its own coordinate system
and Slicer's reader honours it: loading an LPS file flips x and y into the
scene's RAS. Writing a scene position straight back into an LPS file
therefore moves every point by twice its own x and y -- a mirror through the
midsagittal plane, which on a skull looks like a plausible landmark set and
is wrong. `to_file_frame` is the flip back, and it is applied to every
position this module writes.
"""

import json
import os

# Below this, in millimetres, a point is where it was. A control point read
# back from Slicer is a float32 promoted to a double and can differ from what
# was loaded in the last bits; rewriting the file for that would make every
# visit to a patient a modification.
UNMOVED_MM = 1e-4

# What a document means when it does not say. The Slicer markups schema has
# defaulted to LPS since it gained the field, and every writer in this
# ecosystem states it outright.
DEFAULT_COORDINATE_SYSTEM = "LPS"


def to_file_frame(position, coordinate_system: str = DEFAULT_COORDINATE_SYSTEM) -> list:
    """A scene position (RAS) in the frame the file is written in."""
    if (coordinate_system or DEFAULT_COORDINATE_SYSTEM).upper().startswith("LPS"):
        return [-float(position[0]), -float(position[1]), float(position[2])]
    return [float(value) for value in position]


def read_markups(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def coordinate_system_of(document: dict) -> str:
    markups = document.get("markups") or [{}]
    return markups[0].get("coordinateSystem") or DEFAULT_COORDINATE_SYSTEM


def labels_of(document: dict) -> list:
    markups = document.get("markups") or [{}]
    return [point.get("label", "") for point in markups[0].get("controlPoints") or []]


def has_moved(before, after) -> bool:
    return any(abs(float(a) - float(b)) > UNMOVED_MM for a, b in zip(before, after))


def apply_positions(document: dict, positions: dict) -> int:
    """Replace the positions of the points named in `positions`, IN the file's
    own frame. Returns how many actually moved.

    Matched by LABEL, never by index: a reader may have deleted a point, and
    the second point of the file is then not the second point of the node.
    """
    markups = (document.get("markups") or [{}])[0]
    moved = 0
    for point in markups.get("controlPoints") or []:
        wanted = positions.get(point.get("label", ""))
        if wanted is None:
            continue
        current = point.get("position") or [0.0, 0.0, 0.0]
        if not has_moved(current, wanted):
            continue
        point["position"] = [float(value) for value in wanted]
        moved += 1
    return moved


def write_markups(path: str, document: dict) -> str:
    """Write the document back, through a temporary file in the same folder.

    A half-written landmark file is worse than an unchanged one: the reader
    would have neither their correction nor what the tool produced. The
    rename is atomic on every filesystem this runs on.
    """
    staging = path + ".visu-tmp"
    with open(staging, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    os.replace(staging, path)
    return path


def save_markups(path: str, scene_positions: dict) -> int:
    """Fold `{label: RAS position}` into the file at `path`. Returns how many
    points moved; writes nothing at all when that is zero."""
    document = read_markups(path)
    system = coordinate_system_of(document)
    wanted = {label: to_file_frame(position, system)
              for label, position in scene_positions.items()}
    moved = apply_positions(document, wanted)
    if moved:
        write_markups(path, document)
    return moved
