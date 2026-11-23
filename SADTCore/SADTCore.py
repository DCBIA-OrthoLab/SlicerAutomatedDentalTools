import os

import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule


class SADTCore(ScriptedLoadableModule):
    """Hidden module. Its only purpose is to make SADTCoreLib importable
    as `from SADTCoreLib import ...` by every other scripted module in
    this extension - see ARCHITECTURE.md.
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Server Tools Core")
        self.parent.categories = ["Automated Dental Tools Remote.Advanced"]
        self.parent.dependencies = []
        self.parent.contributors = ["Automated Dental Tools team"]
        self.parent.hidden = True
        self.parent.helpText = _(
            "Shared infrastructure for remote-server-backed tools: HTTP client, async "
            "execution, schema-driven GUI generation and shared styling. Not user-facing "
            " - see ARCHITECTURE.md at the repository root."
        )
        self.parent.acknowledgementText = _("This module was supported by NIDCR R01 024450.")

        # Runs once, when Slicer discovers this module at startup - before any
        # tool module is opened. Applies any server URL/API key/etc. the user
        # saved via "Server Tools Settings" in a previous session, on top of
        # config.py's compiled-in defaults.
        # The console filter goes in first, so the warnings of the very first
        # module to load are already shown in their own colour.
        from SADTCoreLib import console, get_client
        from SADTCoreLib.settings_qt import apply_saved_overrides

        console.install()
        apply_saved_overrides(get_client())
        # Say that this Slicer uses the shared tools folder, so the SADT Tools
        # Manager of every Slicer can tell who does (users.json).
        try:
            client = get_client()
            store = getattr(client, "store", None)
            if store is not None and os.path.isdir(store.root):
                store.register_user(os.path.normpath(slicer.app.slicerHome), slicer.app.applicationVersion)
                restored = store.recover()  # an update a crash cut short
                if restored:
                    print(f"SADTCore: put back {', '.join(restored)} after an interrupted update")
        except Exception as error:  # noqa: BLE001 -- bookkeeping must never stop Slicer
            print(f"SADTCore: could not record this Slicer: {error}")
