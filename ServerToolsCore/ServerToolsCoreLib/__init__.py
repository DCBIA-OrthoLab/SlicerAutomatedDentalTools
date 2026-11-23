"""Shared infrastructure for remote-server-backed Slicer tool modules.

Deliberately minimal: this module imports only `client`/`errors`/`config`/`tls`, none
of which touch `slicer` or `qt`. That keeps `import ServerToolsCoreLib` safe
outside Slicer, which is what lets client.py be unit-tested in plain CI.

GUI-facing pieces (design, formgen, slicer_io, worker, base_widget) depend on
`slicer`/`qt`/`ctk` and must be imported explicitly by modules that run inside
Slicer, e.g. `from ServerToolsCoreLib.base_widget import ServerToolWidgetBase`.
"""

from . import config
from .client import (
    RunCheckpoint,
    ToolResult,
    ToolServerClient,
    accepts_folder,
    argument_types,
    file_extensions_for,
    is_file_type,
    is_log_event,
    new_run_id,
    testfile_entries,
)
from .errors import RunCancelled, ServerToolError
from .tls import InsecureServerUrl

_client = None


def _startup_environment() -> dict:
    """The environment Slicer was launched with, before it set up its own Python.

    A tool's venv must not inherit PYTHONHOME/PYTHONPATH pointing at Slicer's
    library. Outside Slicer (the unit tests) it is just this process's.
    """
    try:
        import slicer

        return dict(slicer.util.startupEnvironment())
    except Exception:  # noqa: BLE001 -- no Slicer: plain environment
        import os

        return dict(os.environ)


# Where this Slicer was told to keep the tools, in ITS OWN settings: empty means
# the per-user default every Slicer shares (see tools_store).
LOCATION_SETTING = "SADTTools/Location"
# The pilot's sources, when not given by the environment: this Slicer's settings.
SOURCE_SETTING = "SADTTools/Source"
DATA_SOURCE_SETTING = "SADTTools/DataSource"


def slicer_settings():
    """THIS Slicer installation's own settings (`slicer.org/Slicer-<revision>.ini`).

    Not `qt.QSettings()`, which is the file every Slicer of the user shares:
    a choice made "for this Slicer" has to stay with this Slicer. None outside
    Slicer.
    """
    try:
        import slicer

        return slicer.app.revisionUserSettings()
    except Exception:  # noqa: BLE001 -- no Slicer
        return None


def tool_store():
    """The installed tools of this Slicer: the shared folder, or the one it was told to use."""
    from . import tools_store

    settings = slicer_settings()

    def setting(key):
        return str(settings.value(key, "") or "") if settings is not None else ""

    import os

    root, _why = tools_store.resolve_root(setting(LOCATION_SETTING))
    return tools_store.ToolStore(
        root,
        source=os.environ.get(tools_store.SOURCE_VARIABLE, "") or setting(SOURCE_SETTING),
        data_source=os.environ.get(tools_store.DATA_SOURCE_VARIABLE, "") or setting(DATA_SOURCE_SETTING))


def reset_client() -> None:
    """Forget the client, so the next `get_client()` reads the location again."""
    global _client
    _client = None


def get_client():
    """Singleton client for the whole extension, so its schema cache is shared.

    In this extension the tools run on this machine: the client is the LOCAL
    one, which has the remote client's interface. `config.LOCAL_TOOLS = False`
    goes back to a VISOR server.
    """
    global _client
    if _client is None and getattr(config, "LOCAL_TOOLS", True):
        from .local_client import LocalToolClient

        _client = LocalToolClient(environment=_startup_environment, store=tool_store())
    if _client is None:
        _client = ToolServerClient(
            server_url=config.SERVER_URL,
            token=config.API_TOKEN,
            timeout=config.TIMEOUT,
            parallelism=config.TRANSFER_PARALLELISM,
            chunk_bytes=config.TRANSFER_CHUNK_MB * 1024 * 1024,
            compress_uploads=config.TRANSFER_COMPRESS,
            detached_runs=getattr(config, "DETACHED_RUNS", False),
        )
    return _client


__all__ = [
    "get_client",
    "ToolServerClient",
    "ToolResult",
    "RunCheckpoint",
    "ServerToolError",
    "RunCancelled",
    "InsecureServerUrl",
    "new_run_id",
    "is_file_type",
    "is_log_event",
    "argument_types",
    "accepts_folder",
    "file_extensions_for",
    "testfile_entries",
]
