"""Persists a user-editable override of config.py's defaults via Slicer's
native qt.QSettings (an ini/plist file on disk, independent of the Slicer
process - survives restarts).

Not imported by `__init__.py` on purpose: it depends on `qt`, and the package
must stay importable outside Slicer for client.py's unit tests. Only Slicer-
side code (ServerToolsCore.py at startup, ServerToolsSettingsWidget) uses it.

`config.py` remains the compiled-in defaults; this module is the optional
layer on top, edited via the "Server Tools Settings" module.
"""

import logging
from urllib.parse import urlsplit

import qt

from .client import checked_server_url, is_loopback
from .errors import plain_text

logger = logging.getLogger(__name__)

_GROUP = "ServerTools"
_KEY_SERVER_URL = f"{_GROUP}/ServerUrl"
_KEY_API_TOKEN = f"{_GROUP}/ApiToken"
_KEY_VERIFY_TLS = f"{_GROUP}/VerifyTls"
_KEY_TIMEOUT = f"{_GROUP}/Timeout"
_KEY_UPLOAD_NOTICE_HOST = f"{_GROUP}/UploadNoticeShownFor"


def load_overrides() -> dict:
    """Return only the settings the user has actually saved, e.g. {} if none."""
    settings = qt.QSettings()
    overrides = {}
    if settings.contains(_KEY_SERVER_URL):
        saved = str(settings.value(_KEY_SERVER_URL))
        # Any http:// URL -- loopback included, the default of older clients --
        # is dropped rather than applied, for the reason `verify_tls` is no
        # longer read back either: the server has no plain-HTTP listener, and
        # a value saved before this client refused that scheme must not decide
        # anything. The other settings still apply, Slicer still starts, the
        # URL falls back to config.py's placeholder, and the settings panel
        # shows it the moment they open it.
        try:
            overrides["server_url"] = checked_server_url(saved)
        except ValueError as exc:
            # A warning, not an error: nothing failed, Slicer starts and the
            # settings panel shows the URL to fix.
            logger.warning("The saved server URL was ignored: %s", exc)
    if settings.contains(_KEY_API_TOKEN):
        overrides["token"] = str(settings.value(_KEY_API_TOKEN))
    # `verify_tls` is deliberately NOT read back. It was a saved setting until
    # 2026-10-13, and a machine that had stored False would otherwise keep
    # certificate verification off for ever -- the one setting a user must not
    # be able to turn off on a client that sends medical images. It is not a
    # setting anywhere any more, and a stale key in the file decides nothing.
    if settings.contains(_KEY_TIMEOUT):
        overrides["timeout"] = int(settings.value(_KEY_TIMEOUT, 600))
    return overrides


def save_overrides(server_url: str, token: str, timeout: int) -> None:
    """Persist what a user may change. Certificate verification is not in that
    set -- it is always on, see tls.py."""
    settings = qt.QSettings()
    settings.setValue(_KEY_SERVER_URL, server_url)
    settings.setValue(_KEY_API_TOKEN, token)
    settings.setValue(_KEY_TIMEOUT, timeout)
    # Drop the key this used to write, so a value an older client saved stops
    # sitting in the file looking as though it still decided something.
    settings.remove(_KEY_VERIFY_TLS)
    settings.sync()


def upload_notice_due(server_url: str) -> bool:
    """Whether the "nothing is de-identified" notice still has to be shown for
    `server_url`.

    Never for loopback: nothing leaves the machine, so there is nothing to warn
    about and a dialog on every save would only teach people to dismiss it.

    Tracked per host rather than once and for all. Pointing the client at a
    different server is the moment the question matters again, and the answer
    may differ: one server may be the operator's own machine and the next one
    an institution's, under another agreement.
    """
    host = urlsplit(server_url).hostname or ""
    if not host or is_loopback(host):
        return False
    return str(qt.QSettings().value(_KEY_UPLOAD_NOTICE_HOST, "")) != host


def mark_upload_notice_shown(server_url: str) -> None:
    """Remember that the notice was shown for this host, so it is not repeated."""
    host = urlsplit(server_url).hostname or ""
    if not host:
        return
    settings = qt.QSettings()
    settings.setValue(_KEY_UPLOAD_NOTICE_HOST, host)
    settings.sync()


def notify_uploads_are_not_deidentified(server_url: str) -> None:
    """Say once, per server, that this client de-identifies nothing.

    The README has said so since the client existed, and a README is not where
    somebody finds out what leaves their machine. This fires at the one moment
    the question is live: a server that is not this machine has just been
    pointed at, and nothing has been sent yet.

    Best effort by design -- failing to show a dialog must not cost the user
    the setting they just saved.
    """
    if not upload_notice_due(server_url):
        return

    import slicer
    from slicer.i18n import tr as _

    host = urlsplit(server_url).hostname or server_url
    try:
        slicer.util.warningDisplay(plain_text(_(
            "Files are uploaded unmodified.\n\n"
            "This client does not de-identify anything. What you select is sent to "
            "{0} as it sits on disk: the file names, the DICOM headers, and whatever "
            "identifiers they carry.\n\n"
            "De-identify your data before selecting it, or confine use to data that "
            "carries no protected health information."
        ).format(host)))
    except Exception:  # noqa: BLE001 - a missing dialog must not undo a save
        return
    mark_upload_notice_shown(server_url)


def clear_overrides() -> None:
    settings = qt.QSettings()
    settings.remove(_GROUP)
    settings.sync()


def apply_saved_overrides(client) -> bool:
    """Apply whatever was saved onto `client` (e.g. get_client()). Returns
    whether an override was found and applied - used at Slicer startup."""
    overrides = load_overrides()
    if overrides:
        client.configure(**overrides)
    return bool(overrides)
