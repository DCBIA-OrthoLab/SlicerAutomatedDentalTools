"""Ask for new SADT-VISOR releases in the background, every 15 minutes.

Started once, when Slicer loads SADTCore; `check_now` asks at once, for the
"check" link of a tool panel and the Manager's button. It only PREVENTS: a newer release
this client can drive becomes the version the panels offer as "Update to ...";
nothing is installed until the user presses it, so a run is never disturbed.

Nothing happens for a developer's checkout (SADT_VISOR_SOURCE) or a Slicer kept
on one version (SADTTools/PinnedVersion). Several Slicers sharing one tools
folder share its `releases.json`, so they ask once between them.
"""

import time

import qt

from . import releases, reset_client, tool_store
from .worker import BackgroundJob

_TICK_MS = 60 * 1000
# "waiting": what asked for the check in progress (a panel's "check" link, the
# Manager's button), each told when it ends. One check at a time, however many
# asked: a click in a panel and one in the Manager share it.
_state = {"timer": None, "job": None, "waiting": []}


def install() -> None:
    """Start the watch (idempotent: a module reload does not start a second)."""
    if _state["timer"] is not None:
        return
    timer = qt.QTimer()
    timer.setInterval(_TICK_MS)
    timer.timeout.connect(_tick)
    timer.start()
    _state["timer"] = timer
    qt.QTimer.singleShot(5000, _tick)  # once Slicer has settled


def follows_releases(store) -> bool:
    """Whether this Slicer takes its tools from SADT-VISOR's releases and may move on."""
    return bool(getattr(store, "from_release", False)) and not getattr(store, "pinned", False)


def check_now(on_done=None) -> bool:
    """Ask SADT-VISOR now, without waiting for the next tick.

    `on_done(before, state)` is called on the main thread when the check ends,
    with the version this Slicer ran and what `releases.check` wrote. Returns
    False, and asks nothing, when this Slicer does not follow the releases (a
    checkout, or a version kept for a study).
    """
    store = tool_store()
    if not follows_releases(store):
        return False
    if on_done is not None:
        _state["waiting"].append(on_done)
    if _state["job"] is None:
        _start(store)
    return True


def checking() -> bool:
    return _state["job"] is not None


def _tick() -> None:
    if _state["job"] is not None:
        return
    store = tool_store()
    if not follows_releases(store) or not releases.due(store.root):
        return
    _start(store)


def _start(store) -> None:
    before, root = store.version, store.root

    def failed(error):
        _checked(before, {"latest": releases.latest_known(root), "checked": time.time(), "error": str(error)})

    job = BackgroundJob(lambda _progress: releases.check(root),
                        on_success=lambda state: _checked(before, state), on_error=failed)
    _state["job"] = job
    job.start()


def _checked(before: str, state: dict) -> None:
    _state["job"] = None
    latest = state.get("latest")
    if latest and latest != before:
        # The panels read the version from the client: a fresh one sees it, and
        # each panel's badge says "update available → <latest>" when opened.
        reset_client()
        print(f"SADTCore: SADT-VISOR tools {latest} is available (this Slicer runs {before}).")
    waiting, _state["waiting"] = _state["waiting"], []
    for callback in waiting:
        try:
            callback(before, state)
        except Exception as error:  # noqa: BLE001 -- a closed panel must not stop the others
            print(f"SADTCore: a release check callback failed: {error}")
