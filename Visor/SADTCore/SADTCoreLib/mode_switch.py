"""Switch between the classic ADT panel of a tool and its VISOR local panel.

Two sets of modules live side by side in this extension. The classic set is
what upstream ships (`ALI`, `AREG`, ...), byte for byte. The VISOR set lives in
`Visor/` and carries the suffix `Visor` (`ALIVisor`, `AREGVisor`, ...). The
tool names of SADT-VISOR do not change: `ALIVisorWidget.TOOL_NAME` is still
`ALI`.

Every panel of a pair gets a thin bar at its top, with one button that opens the
other version. The VISOR panels build it themselves (`base_widget.setup`). The
classic panels must stay untouched, so the bar is INJECTED into them from here:

* `install()` is called once by the hidden `SADTCore` module;
* `ensure_bar(name)` puts the bar into the layout of the classic panel `name`,
  or does nothing if it is already there. It is idempotent on purpose:
  `slicer.util.reloadScriptedModule` hides every child of a panel and empties
  its layout, so a bar injected once is gone after a Reload. The bar is
  therefore re-checked when a module is selected AND every `POLL_MS` for the
  panels that exist, which also covers a Reload.

What the user chose is remembered per module in the settings shared by all
the user's Slicers, `AutomatedDentalTools/Version/<Name>` = `classic` | `visor`,
default `classic`. It is written when the button is pressed.

Pairs, and why some modules have no bar
---------------------------------------
* `EQUIVALENTS`: classic module -> the CLI modules it runs (used to tell
  whether a classic treatment is running). These get a bar.
* `NO_PARITY`: both versions exist but the VISOR one does not do what the
  classic one does. A bar would promise "the same thing, another engine", so
  there is none, on either side. GreedyReg: the classic panel has sliders and
  Center/Harden/transform export that the VISOR batch form does not have.
* `CLASSIC_ONLY`: no VISOR panel: no bar, nothing is injected. FlexReg is
  here until its VISOR port is ready.
* `VISOR_ONLY`: no classic panel: no bar.

Parity of the tools in `EQUIVALENTS` with their classic panel has not been
checked tool by tool, only GreedyReg was shown not to have it.

What the bar cannot see
-----------------------
A classic treatment is detected only when it runs through a command line module
of the extension (a `vtkMRMLCommandLineModuleNode` that is busy). The classic
modules that run conda commands by themselves (ALI, AREG, ASO, CLIC, DOCShapeAXI)
are not detected, and their bar is not greyed out.
"""

import contextlib
import logging
import os
import weakref

logger = logging.getLogger(__name__)

SETTINGS_ROOT = "AutomatedDentalTools/Version"
CLASSIC = "classic"
VISOR = "visor"
DEFAULT_MODE = CLASSIC
SUFFIX = "Visor"
BAR_NAME = "adtVersionBar"
POLL_MS = 1000  # how often the classic panels that exist are checked for the bar
REFRESH_MS = 1000  # how often a bar re-reads "is a treatment running"

# classic module -> the command line modules its treatments go through
EQUIVALENTS = {
    "ALI": ("ALI_CBCT", "ALI_IOS"),
    "AMASSS": ("AMASSS_CLI",),
    "AREG": ("AREG_CBCT", "AREG_IOS", "AREG_IOSCBCT"),
    "ASO": ("PRE_ASO_CBCT", "SEMI_ASO_CBCT", "PRE_ASO_IOS", "SEMI_ASO_IOS"),
    "AutoCrop3D": ("AutoCrop3D_CLI",),
    "AutoMatrix": ("Automatrix_CLI",),
    "BATCHDENTALSEG": (),
    "CLIC": (),
    "CNE": ("CNE_CLI",),
    "DOCShapeAXI": ("DOCShapeAXI_CLI",),
    "SurgMovPred": ("SurgMovPred_CLI",),
    "TMJ_Crop": ("TMJ_Crop_CLI",),
    "VFACE": ("VFACE_CLI",),
}
NO_PARITY = {
    "GreedyReg": "the classic panel has sliders, Center, Harden and a .tfm export that the "
                 "VISOR batch form does not have",
}
# FlexReg: its VISOR port is not ready, upstream ships the classic module alone for now.
CLASSIC_ONLY = ("MRI2CBCT", "Agent", "CNE_Dashboard", "Medical_Data_Anonymizer_Module", "FlexReg")
VISOR_ONLY = ("SADTVisu", "Convert", "CrownSeg", "SADTToolsManager")

# Every module of the VISOR set, with the classic one it is the twin of.
VISOR_MODULES = {name + SUFFIX: name for name in (*EQUIVALENTS, *NO_PARITY)}


def _tr(text: str) -> str:
    try:
        from slicer.i18n import tr

        return tr(text)
    except Exception:  # noqa: BLE001 -- outside Slicer
        return text


# --------------------------------------------------------------------------
# Names and the remembered mode (no Qt, no Slicer: tested as plain Python)
# --------------------------------------------------------------------------


def visor_name(classic: str) -> str:
    return classic + SUFFIX


def classic_name(visor: str):
    """The classic twin of a VISOR module name, or None."""
    return VISOR_MODULES.get(visor)


def has_bar(classic: str) -> bool:
    """Whether the pair `classic` / `classic + Visor` gets a bar."""
    return classic in EQUIVALENTS


def _settings(settings=None):
    if settings is not None:
        return settings
    import qt

    return qt.QSettings()


def settings_key(classic: str) -> str:
    return f"{SETTINGS_ROOT}/{classic}"


def get_mode(classic: str, settings=None) -> str:
    """`classic` or `visor`; anything else, or no settings at all, is the default."""
    try:
        value = str(_settings(settings).value(settings_key(classic), DEFAULT_MODE) or DEFAULT_MODE)
    except Exception:  # noqa: BLE001 -- unreadable settings must not break a panel
        return DEFAULT_MODE
    return value if value in (CLASSIC, VISOR) else DEFAULT_MODE


def set_mode(classic: str, mode: str, settings=None) -> None:
    if mode not in (CLASSIC, VISOR):
        raise ValueError(f"unknown mode {mode!r}")
    _settings(settings).setValue(settings_key(classic), mode)


# --------------------------------------------------------------------------
# What the bar says (pure: the same function feeds the Qt widgets and the tests)
# --------------------------------------------------------------------------


class BarState:
    """Texts and enabled state of a bar. A plain object so tests can compare it."""

    def __init__(self, heading, explanation, button, enabled, tooltip=""):
        self.heading = heading
        self.explanation = explanation
        self.button = button
        self.enabled = enabled
        self.tooltip = tooltip

    def as_tuple(self):
        return (self.heading, self.explanation, self.button, self.enabled, self.tooltip)

    def __repr__(self):
        return f"BarState{self.as_tuple()!r}"


def bar_state(side: str, classic: str, busy_reason: str = "", target_loaded: bool = True) -> BarState:
    """What the bar of the `side` panel of `classic` shows.

    `busy_reason` is non-empty while a treatment runs in the panel you are on;
    the button is then disabled and the line says why. `target_loaded` is
    False when the other version's module failed to load in this session.
    """
    if side == CLASSIC:
        heading = _tr("Version classique (calcul dans Slicer, environnement conda)")
        button = _tr("Passer à la version VISOR locale")
        other = _tr("la version VISOR locale")
        explanation = _tr(
            "La version VISOR locale utilise l'outil VISOR installé sur cet ordinateur, dans son "
            "propre environnement. Changer de version n'arrête aucun traitement.")
    elif side == VISOR:
        heading = _tr("Version VISOR locale (outil installé sur cet ordinateur)")
        button = _tr("Revenir à la version classique")
        other = _tr("la version classique")
        explanation = _tr(
            "La version classique calcule dans Slicer, avec l'environnement conda de l'extension. "
            "Changer de version n'arrête aucun traitement.")
    else:
        raise ValueError(f"unknown side {side!r}")

    if busy_reason:
        return BarState(
            heading,
            _tr("Un traitement {tool} est en cours dans cette version ({why}). Le changement de "
                "version est désactivé tant qu'il dure : il ne l'arrêterait pas, et vous le "
                "perdriez de vue.").format(tool=classic, why=busy_reason),
            button, False, _tr("Désactivé : un traitement est en cours dans cette version."))
    if not target_loaded:
        return BarState(
            heading,
            _tr("{other} n'est pas chargée dans cette session de Slicer (voir le journal des "
                "modules). Il n'y a rien vers quoi basculer.").format(other=other.capitalize()),
            button, False, _tr("Désactivé : le module de l'autre version n'est pas chargé."))
    return BarState(heading, explanation, button, True)


def target_module(side: str, classic: str) -> str:
    """The module the button of the `side` panel opens."""
    return visor_name(classic) if side == CLASSIC else classic


def switch(side: str, classic: str, select=None, settings=None) -> bool:
    """Remember the choice and open the other version. False if it could not.

    The choice is written first, then undone if the module cannot be selected.
    """
    new_mode = VISOR if side == CLASSIC else CLASSIC
    before = get_mode(classic, settings)
    try:
        set_mode(classic, new_mode, settings)
    except Exception as error:  # noqa: BLE001 -- the switch still works without memory
        logger.warning("Could not remember the version of %s: %s", classic, error)
        before = None
    try:
        if select is None:
            import slicer

            select = slicer.util.selectModule
        select(target_module(side, classic))
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not open %s: %s", target_module(side, classic), error)
        if before is not None:
            with contextlib.suppress(Exception):
                set_mode(classic, before, settings)
        return False
    return True


# --------------------------------------------------------------------------
# Is a treatment running? (needs Slicer)
# --------------------------------------------------------------------------


def classic_busy_reason(classic: str) -> str:
    """Why the classic panel of `classic` is busy, or "".

    Only command line modules are seen; see the module docstring.
    """
    names = {n.lower() for n in EQUIVALENTS.get(classic, ())}
    if not names:
        return ""
    try:
        import slicer

        nodes = slicer.util.getNodesByClass("vtkMRMLCommandLineModuleNode")
    except Exception:  # noqa: BLE001 -- no scene: nothing runs
        return ""
    for node in nodes:
        with contextlib.suppress(Exception):
            if node.GetModuleTitle().lower() in names and node.IsBusy():
                return _tr("module en ligne de commande {name} actif").format(name=node.GetModuleTitle())
    return ""


def visor_busy_reason(widget) -> str:
    """Why a VISOR panel is busy, or "": runs it queued or started and not yet finished."""
    runs = list(getattr(widget, "_runs", None) or [])
    if not runs:
        return ""
    return _tr("{n} traitement(s) lancé(s) depuis ce panneau").format(n=len(runs))


# --------------------------------------------------------------------------
# The widget
# --------------------------------------------------------------------------

_BARS = {}  # (side, classic) -> ModeSwitchBar, kept alive here and replaced on a rebuild


class ModeSwitchBar:
    """The bar of one panel: a frame, a heading, a button, a line of explanation."""

    def __init__(self, side, classic, busy, select=None, settings=None, loaded=None):
        import qt

        self.side = side
        self.classic = classic
        self._busy = busy
        self._select = select
        self._settings = settings
        self._loaded = loaded or (lambda: module_loaded(target_module(side, classic)))

        frame = qt.QFrame()
        frame.setObjectName(BAR_NAME)
        frame.setFrameShape(qt.QFrame.StyledPanel)
        frame.setStyleSheet(f"QFrame#{BAR_NAME} {{ border: 1px solid palette(mid); border-radius: 4px; }}")
        outer = qt.QVBoxLayout(frame)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)
        row = qt.QHBoxLayout()
        self.heading = qt.QLabel()
        self.heading.setWordWrap(True)
        font = self.heading.font
        font.setBold(True)
        self.heading.setFont(font)
        self.button = qt.QPushButton()
        self.button.setObjectName("adtVersionSwitchButton")
        row.addWidget(self.heading, 1)
        row.addWidget(self.button)
        outer.addLayout(row)
        self.explanation = qt.QLabel()
        self.explanation.setWordWrap(True)
        outer.addWidget(self.explanation)
        self.frame = frame
        self.button.connect("clicked()", self._clicked)

        self.timer = qt.QTimer(frame)
        self.timer.setInterval(REFRESH_MS)
        self.timer.connect("timeout()", self.refresh)
        self.timer.start()
        self.refresh()

    def state(self) -> BarState:
        try:
            reason = self._busy() or ""
        except Exception:  # noqa: BLE001
            reason = ""
        try:
            loaded = bool(self._loaded())
        except Exception:  # noqa: BLE001
            loaded = True
        return bar_state(self.side, self.classic, reason, loaded)

    def refresh(self) -> None:
        try:
            state = self.state()
            self.heading.setText(state.heading)
            self.explanation.setText(state.explanation)
            self.button.setText(state.button)
            self.button.setEnabled(state.enabled)
            self.button.setToolTip(state.tooltip)
        except RuntimeError:  # the frame was deleted under us (a Reload)
            self.dispose()

    def _clicked(self) -> None:
        if self.state().enabled:
            switch(self.side, self.classic, self._select, self._settings)

    def dispose(self) -> None:
        with contextlib.suppress(Exception):
            self.timer.stop()
        key = (self.side, self.classic)
        if _BARS.get(key) is self:
            del _BARS[key]


def module_loaded(module_name: str) -> bool:
    try:
        import slicer

        return getattr(slicer.modules, module_name.lower(), None) is not None
    except Exception:  # noqa: BLE001
        return False


def _register(bar: ModeSwitchBar) -> ModeSwitchBar:
    key = (bar.side, bar.classic)
    old = _BARS.get(key)
    if old is not None and old is not bar:
        old.dispose()
        with contextlib.suppress(Exception):
            old.frame.hide()
            old.frame.deleteLater()
    _BARS[key] = bar
    return bar


# --------------------------------------------------------------------------
# VISOR side: called by ServerToolWidgetBase.setup()
# --------------------------------------------------------------------------


def add_visor_bar(widget) -> bool:
    """Put the bar at the top of a VISOR panel. False if this panel has none."""
    classic = classic_name(getattr(widget, "moduleName", "") or type(widget).__name__.replace("Widget", ""))
    if classic is None or not has_bar(classic):
        return False
    ref = weakref.ref(widget)

    def busy():
        target = ref()
        return visor_busy_reason(target) if target is not None else ""

    try:
        bar = _register(ModeSwitchBar(VISOR, classic, busy))
        widget.layout.insertWidget(0, bar.frame)
    except Exception as error:  # noqa: BLE001 -- a missing bar must never cost the panel
        logger.warning("Could not add the version bar to %s: %s", classic, error)
        return False
    return True


def remove_visor_bar(widget) -> None:
    classic = classic_name(getattr(widget, "moduleName", "") or type(widget).__name__.replace("Widget", ""))
    bar = _BARS.get((VISOR, classic))
    if bar is not None:
        bar.dispose()


# --------------------------------------------------------------------------
# Classic side: injected, the classic files are never edited
# --------------------------------------------------------------------------


def _layout_of(classic: str):
    import slicer

    module = getattr(slicer.modules, classic.lower(), None)
    if module is None:
        return None
    return module.widgetRepresentation().layout()


def ensure_bar(classic: str, layout=None) -> bool:
    """Make sure the classic panel of `classic` has its bar; True if it does.

    Idempotent. A bar that is no longer in the layout (a Reload removed it and
    hid it) is thrown away and rebuilt rather than revived.
    """
    if not has_bar(classic):
        return False
    try:
        if layout is None:
            layout = _layout_of(classic)
        if layout is None:
            return False
        bar = _BARS.get((CLASSIC, classic))
        if bar is not None:
            try:
                if layout.indexOf(bar.frame) >= 0:
                    return True
            except RuntimeError:  # the frame no longer exists
                pass
        bar = _register(ModeSwitchBar(CLASSIC, classic, lambda: classic_busy_reason(classic)))
        layout.insertWidget(0, bar.frame)
        return True
    except Exception as error:  # noqa: BLE001 -- the classic panel works without its bar
        logger.warning("Could not inject the version bar into %s: %s", classic, error)
        return False


def ensure_existing() -> list:
    """Re-check every classic panel that has been created; the names that now have a bar."""
    try:
        import slicer
    except ImportError:
        return []
    done = []
    for classic in EQUIVALENTS:
        if getattr(slicer.modules, classic + "Widget", None) is not None and ensure_bar(classic):
            done.append(classic)
    return done


class _Injector:
    def __init__(self):
        self.timer = None
        self.connected = False

    def start(self) -> None:
        import qt

        if self.timer is None:
            self.timer = qt.QTimer()
            self.timer.setInterval(POLL_MS)
            self.timer.connect("timeout()", ensure_existing)
            self.timer.start()
        self.connect_selector()

    def connect_selector(self) -> bool:
        """Inject at once when a module is selected, instead of waiting for the poll.

        Needs the main window; without one the poll is all there is.
        """
        if self.connected:
            return True
        try:
            import slicer

            window = slicer.util.mainWindow()
            if window is None:
                return False
            window.moduleSelector().connect("moduleSelected(QString)", self.on_selected)
            self.connected = True
        except Exception as error:  # noqa: BLE001
            logger.debug("No module selector to follow: %s", error)
            return False
        return True

    def on_selected(self, name) -> None:
        name = str(name)
        if has_bar(name):
            import qt

            qt.QTimer.singleShot(0, lambda: ensure_bar(name))


_INJECTOR = _Injector()


def install() -> None:
    """Start following the classic panels. Idempotent. Never raises."""
    try:
        _INJECTOR.start()
        with contextlib.suppress(Exception):
            import slicer

            # The main window may not exist yet when the module loads.
            slicer.app.connect("startupCompleted()", _INJECTOR.connect_selector)
    except Exception as error:  # noqa: BLE001
        logger.warning("The version bar of the classic panels is not available: %s", error)


# --------------------------------------------------------------------------
# Icons
# --------------------------------------------------------------------------


def legacy_icon(module_file: str, icon_file: str):
    """The classic module's icon, for its VISOR twin.

    Installed, both modules share one `Resources/Icons` folder, next to the
    module file. In a source checkout the classic icon is in the classic
    module's own folder. Returns an empty icon when neither exists.
    """
    import glob

    import qt

    here = os.path.dirname(os.path.abspath(module_file))
    classic = os.path.splitext(os.path.basename(module_file))[0]
    if classic.endswith(SUFFIX):
        classic = classic[: -len(SUFFIX)]
    repo = os.path.dirname(os.path.dirname(here))
    candidates = [os.path.join(here, "Resources", "Icons", icon_file)]
    candidates += glob.glob(os.path.join(repo, classic, "**", "Resources", "Icons", icon_file), recursive=True)
    for path in candidates:
        if os.path.isfile(path):
            return qt.QIcon(path)
    return qt.QIcon()
