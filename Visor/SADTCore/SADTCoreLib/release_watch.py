"""Ask for new SADT-VISOR releases in the background, every 15 minutes.

Started once, when Slicer loads SADTCore. It only PREVENTS: a newer release
this client can drive becomes the version the panels offer as "Update to ...";
nothing is installed until the user presses it, so a run is never disturbed.

Nothing happens for a developer's checkout (SADT_VISOR_SOURCE) or a Slicer kept
on one version (SADTTools/PinnedVersion). Several Slicers sharing one tools
folder share its `releases.json`, so they ask once between them.
"""

import qt

from . import releases, reset_client, tool_store
from .worker import BackgroundJob

_TICK_MS = 60 * 1000
_state = {"timer": None, "job": None}


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


def _tick() -> None:
    if _state["job"] is not None:
        return
    store = tool_store()
    if not getattr(store, "from_release", False) or getattr(store, "pinned", False):
        return
    if not releases.due(store.root):
        return
    before = store.version
    job = BackgroundJob(lambda _progress: releases.check(store.root),
                        on_success=lambda state: _checked(before, state),
                        on_error=lambda _error: _state.update(job=None))
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
