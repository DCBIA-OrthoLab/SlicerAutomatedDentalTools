"""
SADT Tools Manager: where the tools of this extension are, what they take, and
how to remove them.

The tools (SADT-VISOR's) are downloaded on demand into ONE folder shared by
every Slicer of this user. This panel is the one place to see that folder and
its size, every tool and every model installed with what needs it, which Slicer
installs use it, and to install or remove tools, models, old versions, or
everything.
"""

import os
import platform
import shutil
import subprocess

import ctk
import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

from SADTCoreLib import LOCATION_SETTING, PIN_SETTING, reset_client, slicer_settings, tool_store
from SADTCoreLib import design, tools_store, tools_ui


class SADTToolsManager(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SADT Tools Manager")
        self.parent.categories = ["Automated Dental Tools.VISOR local"]
        self.parent.dependencies = ["SADTCore"]
        self.parent.contributors = ["Jules Grivot Pélisson (UNC)"]
        self.parent.helpText = _("""
        Shows where the tools of Automated Dental Tools and their models are stored on this
        computer, how much space they take and which Slicer installations use them; installs
        and removes them.
        """)
        self.parent.acknowledgementText = ""


def _open_folder(path: str) -> None:
    if not os.path.isdir(path):
        slicer.util.infoDisplay(_("The folder does not exist yet: nothing is installed."))
        return
    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # noqa: S606 -- the user asked to see the folder
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(path))


def _table_style() -> str:
    """A table as a card, like the sections and input rows of the tool panels."""
    t = design.tokens()
    radius = design.RADIUS_LG
    return f"""
    QTableWidget {{
      background-color: {t['SURFACE_TABLE']};
      alternate-background-color: {t['SURFACE_HOVER']};
      border: 1px solid {t['BORDER']};
      border-radius: {radius}px;
      gridline-color: transparent;
      color: {t['TEXT']};
    }}
    QTableWidget::item {{ padding: 4px 8px; border: none; }}
    QHeaderView {{ background: transparent; border: none; }}
    QHeaderView::section {{
      background-color: {t['BACKGROUND']};
      color: {t['TEXT_MUTED']};
      font-weight: 600;
      padding: 6px 8px;
      border: none;
      border-bottom: 1px solid {t['BORDER']};
    }}
    QHeaderView::section:first {{ border-top-left-radius: {radius}px; }}
    QHeaderView::section:last {{ border-top-right-radius: {radius}px; }}
    """


def _table(headers) -> qt.QTableWidget:
    table = qt.QTableWidget(0, len(headers))
    table.setStyleSheet(_table_style())
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(qt.QAbstractItemView.NoSelection)
    table.setShowGrid(False)
    table.setAlternatingRowColors(True)
    table.setFocusPolicy(qt.Qt.NoFocus)
    table.verticalHeader().setDefaultSectionSize(34)
    return table


def _fit(table: qt.QTableWidget, stretch=-2) -> None:
    """Every column to its content, the first one taking what is left; no inner scroll."""
    header = table.horizontalHeader()
    # Names and sizes keep their width; the "needed by / used by" column, the
    # one before the buttons, takes what is left and elides.
    for column in range(table.columnCount - 1):
        header.setSectionResizeMode(column, qt.QHeaderView.ResizeToContents)
    # Several stretching columns share what is left: in a narrow panel the
    # model NAME must not be squeezed to "..." by a long "used by" list.
    for column in (stretch if isinstance(stretch, (tuple, list)) else (stretch,)):
        header.setSectionResizeMode(column % table.columnCount, qt.QHeaderView.Stretch)
    table.setTextElideMode(qt.Qt.ElideMiddle)
    table.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    # The last column holds the Remove buttons, which resizeColumnsToContents
    # ignores: as wide as the widest of them, plus its margins.
    last = table.columnCount - 1
    widest = max([table.cellWidget(row, last).sizeHint.width()
                  for row in range(table.rowCount) if table.cellWidget(row, last)] or [80])
    table.horizontalHeader().setSectionResizeMode(last, qt.QHeaderView.Fixed)
    table.setColumnWidth(last, widest + 16)
    height = table.horizontalHeader().height + 4
    for row in range(table.rowCount):
        height += table.rowHeight(row)
    table.setMinimumHeight(height)
    table.setMaximumHeight(height)


def _section(title: str) -> tuple:
    box = ctk.ctkCollapsibleButton()
    box.text = title
    layout = qt.QVBoxLayout(box)
    return box, layout


class SADTToolsManagerWidget(ScriptedLoadableModuleWidget):
    TOOL_COLUMNS = (_("Tool"), _("Version"), _("Size"), _("Needed by"), "")
    MODEL_COLUMNS = (_("Name"), _("Kind"), _("Size"), _("Used by"), "")

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self._folderSize = {}  # tools folder -> bytes, as last measured
        self._sizeJob = None
        # What removing an old copy really gives back, as last measured:
        # installed tool path -> bytes, and the old versions together.
        self._freed = {}
        self._oldJob = None
        self.uiWidget = qt.QWidget()
        self.layout.addWidget(self.uiWidget)
        root = qt.QVBoxLayout(self.uiWidget)

        # ---- summary ----
        self.summaryLabel = design.section_title("")
        root.addWidget(self.summaryLabel)
        self.pathLabel = qt.QLabel()
        self.pathLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.pathLabel.setWordWrap(True)
        root.addWidget(self.pathLabel)
        self.whyLabel = design.hint_label("")
        root.addWidget(self.whyLabel)
        # The version line wraps, and so does everything below that names tools:
        # a label that does not wrap, or a button that names them, sets the
        # panel's minimum width to the length of the list.
        versionRow = qt.QHBoxLayout()
        self.versionLabel = qt.QLabel()
        self.versionLabel.setTextFormat(qt.Qt.RichText)
        self.versionLabel.setWordWrap(True)
        versionRow.addWidget(self.versionLabel, 1)
        # The watch asks every 15 minutes; this asks now, after a push.
        self.checkButton = design.secondary_button(_("Check for updates"))
        self.checkButton.clicked.connect(lambda: self.onCheckForUpdates())
        versionRow.addWidget(self.checkButton, 0, qt.Qt.AlignTop)
        root.addLayout(versionRow)
        self.checkLabel = design.hint_label("")
        self.checkLabel.setWordWrap(True)
        self.checkLabel.setVisible(False)
        root.addWidget(self.checkLabel)
        self._checkJob = None
        # A study keeps the version its results came from; everyone else follows.
        self.pinCheckBox = qt.QCheckBox(_("Stay on this tools version (for a study): do not offer updates"))
        self.pinCheckBox.toggled.connect(lambda checked: self.onPin(checked))
        root.addWidget(self.pinCheckBox)
        self.updateAllButton = design.primary_button("")
        self.updateAllButton.clicked.connect(lambda: self.onUpdateAll())
        root.addWidget(self.updateAllButton)
        # The names of the tools it updates: here, wrapped, not on the button.
        self.updateAllNames = design.hint_label("")
        self.updateAllNames.setWordWrap(True)
        root.addWidget(self.updateAllNames)
        buttons = qt.QHBoxLayout()
        openButton = design.secondary_button(_("Open folder"))
        changeButton = design.secondary_button(_("Change for this Slicer..."))
        resetButton = design.secondary_button(_("Use the shared folder"))
        openButton.clicked.connect(lambda: _open_folder(self.store.root))
        changeButton.clicked.connect(lambda: self.onChangeLocation())
        resetButton.clicked.connect(lambda: self.onResetLocation())
        for button in (openButton, changeButton, resetButton):
            buttons.addWidget(button)
        root.addLayout(buttons)

        # ---- tools ----
        box, layout = _section(_("Tools"))
        self.toolsEmpty = design.hint_label(_("No tool is installed yet."))
        layout.addWidget(self.toolsEmpty)
        self.toolsTable = _table(self.TOOL_COLUMNS)
        layout.addWidget(self.toolsTable)
        root.addWidget(box)

        # ---- models ----
        box, layout = _section(_("Models and test files"))
        self.modelsEmpty = design.hint_label(_("No model or test file is installed yet."))
        layout.addWidget(self.modelsEmpty)
        self.modelsTable = _table(self.MODEL_COLUMNS)
        layout.addWidget(self.modelsTable)
        root.addWidget(box)

        # ---- install ----
        installButton = design.primary_button(_("Install tools..."))
        installButton.clicked.connect(
            lambda: tools_ui.choose_and_install(self.parent, on_done=lambda _installed: self.refresh()))
        root.addWidget(installButton)

        # ---- who uses it, and the big removals ----
        box, layout = _section(_("Shared with"))
        self.usersLabel = qt.QLabel()
        self.usersLabel.setWordWrap(True)
        self.usersLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        layout.addWidget(self.usersLabel)
        removals = qt.QHBoxLayout()
        # What an update leaves behind: the tools of the version it replaced,
        # kept so a run in progress or a tool not yet updated still works.
        self.oldButton = design.secondary_button("")
        self.oldButton.setToolTip(_(
            "After an update, the tools of the previous version stay installed until "
            "you remove them. This removes them; the current version is kept."))
        everythingButton = design.danger_button(_("Remove everything"))
        self.oldButton.clicked.connect(lambda: self.onRemoveOldVersions())
        everythingButton.clicked.connect(lambda: self.onRemoveEverything())
        removals.addStretch(1)
        removals.addWidget(self.oldButton)
        removals.addWidget(everythingButton)
        layout.addLayout(removals)
        root.addWidget(box)

        root.addStretch(1)
        design.apply(self.uiWidget)

        self.refresh()

    def enter(self):
        self.refresh()

    # ---- filling ---------------------------------------------------------

    def refresh(self):
        self.store = tool_store()
        choice = str(slicer_settings().value(LOCATION_SETTING, "") or "")
        _root, why = tools_store.resolve_root(choice)
        self.pathLabel.setText(f"<b>{self.store.root}</b>")
        self.whyLabel.setText(why[0].upper() + why[1:])
        self.tools = sorted(self.store.installed(),
                            key=lambda row: (row["name"].lower(), not row["current"]))
        self.modelRows = self.store.models()
        self._fillTools()
        self._fillModels()
        users = self.store.users()
        self.usersLabel.setText("<br>".join(
            f"Slicer {info.get('version', '?')} &nbsp;<span style='color: gray;'>{home} · "
            f"last used {info.get('last_used', '?')}</span>"
            for home, info in sorted(users.items())) or _("No Slicer has used this folder yet."))
        self._showSizes()
        self._showVersions()

    def _outdated(self) -> list:
        """Tools installed only at an older version, or changed since: what an update brings."""
        current = {row["name"] for row in self.tools if row["current"]}
        # Only OLDER versions: a newer one belongs to a newer Slicer sharing the folder.
        older = {row["name"] for row in self.tools
                 if not row["current"] and self.store.is_older(row["version"])} - current
        changed = {name for name in current
                   if self.store.source and self.store.outdated(name, self.store.source)}
        return sorted(older | changed)

    def _showVersions(self):
        t = design.tokens()
        outdated = self._outdated()
        state = (f"<span style='color: #c77700;'>{len(outdated)} tool(s) to update</span>" if outdated
                 else f"<span style='color: {t['SUCCESS']};'>all tools up to date</span>" if self.tools
                 else "")
        self.versionLabel.setText(
            f"Tools version: <b>{self.store.version}</b>" + self._followText()
            + (f" &nbsp;·&nbsp; {state}" if state else ""))
        from_release = getattr(self.store, "from_release", False)
        self.pinCheckBox.setVisible(from_release)
        self.pinCheckBox.blockSignals(True)
        self.pinCheckBox.setChecked(bool(getattr(self.store, "pinned", False)))
        self.pinCheckBox.blockSignals(False)
        self.updateAllButton.setText(_("Update all ({count} tool(s)) to {version}").format(
            count=len(outdated), version=self.store.version))
        self.updateAllButton.setToolTip(", ".join(outdated))
        self.updateAllButton.setVisible(bool(outdated))
        self.updateAllNames.setText(", ".join(outdated))
        self.updateAllNames.setVisible(bool(outdated))
        self._showCheckButton()
        self._showOldButton()

    def _oldRows(self) -> list:
        return [row for row in self.tools if not row["current"] and self.store.is_older(row["version"])]

    def _showOldButton(self):
        old = self._oldRows()
        versions = sorted({row["version"] for row in old}, key=tools_store.version_key)
        freed = self._freed.get(tuple(versions))
        self.oldButton.setText(_("Remove old version {versions} ({size})").format(
            versions=", ".join(versions),
            size=_("frees {size}").format(size=tools_store.human_size(freed)) if freed is not None
            else _("measuring...")))
        self.oldButton.setVisible(bool(old))

    def _followText(self) -> str:
        """Where the version comes from, and when SADT-VISOR was last asked."""
        if not getattr(self.store, "from_release", False):
            return " &nbsp;·&nbsp; from the SADT-VISOR checkout " + self.store.source
        if getattr(self.store, "pinned", False):
            return " &nbsp;·&nbsp; kept for a study"
        import time

        from SADTCoreLib import releases

        state = releases.read_state(self.store.root)
        if not state.get("checked"):
            return " &nbsp;·&nbsp; following SADT-VISOR's releases (not asked yet)"
        when = time.strftime("%H:%M", time.localtime(float(state["checked"])))
        if state.get("error"):
            return f" &nbsp;·&nbsp; following SADT-VISOR's releases (could not ask at {when}: offline?)"
        return f" &nbsp;·&nbsp; following SADT-VISOR's releases (asked at {when})"

    def _showCheckButton(self):
        """Enabled when this Slicer follows the releases; the tooltip says why not."""
        if self._checkJob is not None:
            return
        self.checkButton.setText(_("Check for updates"))
        if not getattr(self.store, "from_release", False):
            enabled, tip = False, _("The tools come from a SADT-VISOR checkout, not from its releases.")
        elif getattr(self.store, "pinned", False):
            enabled, tip = False, _("Kept on {version} for a study: untick the box below to follow "
                                    "the releases again.").format(version=self.store.version)
        else:
            enabled, tip = True, _("Ask SADT-VISOR now for a newer tools release "
                                   "(otherwise asked every 15 minutes).")
        self.checkButton.setEnabled(enabled)
        self.checkButton.setToolTip(tip)

    def onCheckForUpdates(self):
        from SADTCoreLib import releases
        from SADTCoreLib.worker import BackgroundJob

        if self._checkJob is not None:
            return
        root, before = self.store.root, self.store.version
        self.checkButton.setEnabled(False)
        self.checkButton.setText(_("Checking..."))

        def finished(state):
            self._checkJob = None
            reset_client()
            self.refresh()
            self.checkLabel.setText(releases.describe_check(before, state))
            self.checkLabel.setVisible(True)

        def failed(error):
            finished({"error": str(error), "latest": releases.latest_known(root)})

        self._checkJob = BackgroundJob(lambda _progress: releases.check(root),
                                       on_success=finished, on_error=failed)
        self._checkJob.start()

    def onPin(self, checked: bool):
        settings = slicer_settings()
        if settings is not None:
            settings.setValue(PIN_SETTING, self.store.version if checked else "")
        reset_client()
        self.refresh()

    def onUpdateAll(self):
        outdated = self._outdated()
        if outdated:
            tools_ui.install_many(self.parent, outdated, on_done=lambda _installed: self.refresh())

    def _removeButton(self, handler):
        """A compact Remove, sized to the row rather than to a form."""
        button = qt.QPushButton(_("Remove"))
        t = design.tokens()
        button.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {t['DANGER']};"
            f" border: 1px solid {t['DANGER']}; border-radius: {design.RADIUS_SM}px;"
            f" padding: 2px 10px; margin: 4px 8px 4px 4px; font-weight: 600; }}"
            f"QPushButton:hover {{ background-color: {t['DANGER']}; color: white; }}")
        button.setCursor(qt.QCursor(qt.Qt.PointingHandCursor))
        button.clicked.connect(handler)
        return button

    def _fillTools(self):
        table = self.toolsTable
        table.setRowCount(len(self.tools))
        t = design.tokens()
        current = {row["name"] for row in self.tools if row["current"]}
        for index, row in enumerate(self.tools):
            needed = ", ".join(self.store.needed_by(row["name"])) if row["current"] else ""
            if row["current"] and self.store.source and self.store.outdated(row["name"], self.store.source):
                version, color, hint = f"{row['version']} ↑", "#c77700", _(
                    "Its code or its dependencies changed since it was installed: Update, at the top")
            elif row["current"]:
                version, color, hint = f"{row['version']} ✓", t["SUCCESS"], _("Up to date")
            elif not self.store.is_older(row["version"]):
                version, color, hint = f"{row['version']} (newer)", t["TEXT_MUTED"], _(
                    "Installed by a newer Slicer that shares this folder; this Slicer does not use it")
            elif row["name"] in current:
                version, color, hint = f"{row['version']} (old)", t["TEXT_MUTED"], _(
                    "A previous version, kept after the update: Remove old version frees it")
            else:
                version, color, hint = f"{row['version']} ↑", "#c77700", _(
                    "An update is available: Update, at the top")
            for column, value in enumerate((row["name"], version, "", needed or "—")):
                item = qt.QTableWidgetItem(value)
                if column == 3:
                    item.setToolTip(value)  # the whole list, when the cell elides it
                if column == 1:
                    item.setForeground(qt.QBrush(qt.QColor(color)))
                    item.setToolTip(hint)
                table.setItem(index, column, item)
            table.setCellWidget(index, 4, self._removeButton(
                lambda _checked=False, r=row: self.onRemoveTool(r)))
        self.toolsEmpty.setVisible(not self.tools)
        table.setVisible(bool(self.tools))
        _fit(table)

    def _fillModels(self):
        table = self.modelsTable
        table.setRowCount(len(self.modelRows))
        for index, row in enumerate(self.modelRows):
            values = (row["name"], row.get("kind", "model"), "", ", ".join(row["used_by"]) or "—")
            for column, value in enumerate(values):
                item = qt.QTableWidgetItem(value)
                item.setToolTip(value)  # the whole text, when the cell elides it
                table.setItem(index, column, item)
            table.setCellWidget(index, 4, self._removeButton(
                lambda _checked=False, r=row: self.onRemoveModel(r)))
        self.modelsEmpty.setVisible(not self.modelRows)
        table.setVisible(bool(self.modelRows))
        # Model names are long ("CBCT_Gold_Frankfurt_Horizontal_..."): the name elides.
        _fit(table, stretch=(0, 3))

    def _showSizes(self):
        """Sizes as recorded when each tool and model was installed: nothing to measure."""
        size = tools_store.human_size
        total = 0
        for table, rows, column in ((self.toolsTable, self.tools, 2),
                                    (self.modelsTable, self.modelRows, 2)):
            for index, row in enumerate(rows):
                value = row.get("size")
                total += value or 0
                table.setItem(index, column, qt.QTableWidgetItem(size(value) if value is not None else "—"))
            _fit(table, stretch=(0, 3) if table is self.modelsTable else -2)
        self._showOldSizes()
        self._showSummary()
        self._measureFolder()
        self._measureOld()

    def _showOldSizes(self):
        """An old copy shows what removing it frees, not its recorded size.

        The recorded size counts every file of its environment, but those are
        hard links to uv's package cache, shared with the current copy of the
        same tool: an old AMASSS "of 4 GB" frees a few MB. Summed over twenty
        tools, the Manager read 86.6 GB for versions that free 0.9 GB.
        """
        size = tools_store.human_size
        old = {id(row) for row in self._oldRows()}
        for index, row in enumerate(self.tools):
            if id(row) not in old:
                continue
            freed = self._freed.get(row["path"])
            recorded = size(row.get("size") or 0)
            item = qt.QTableWidgetItem(size(freed) if freed is not None else "…")
            item.setToolTip(
                _("Removing it frees {freed}. The rest of its {recorded} is shared with the "
                  "current tools (the package cache) and stays.").format(freed=size(freed), recorded=recorded)
                if freed is not None else _("Measuring what removing it frees..."))
            self.toolsTable.setItem(index, 2, item)

    def _measureOld(self):
        """What removing each old copy, and the old versions together, gives back.

        Measured on a worker: it reads every file of the old copies and of the
        package cache, the same exact count the removal's confirmation gives.
        """
        from SADTCoreLib.worker import BackgroundJob

        old = self._oldRows()
        if self._oldJob is not None or not old:
            return
        store = self.store
        paths = [row["path"] for row in old]
        versions = tuple(sorted({row["version"] for row in old}, key=tools_store.version_key))
        folders = [os.path.join(store.root, f"tools-{version}") for version in versions]

        def measure(_progress):
            extra = store._cache_inodes() if store.private_cache else None
            freed = {path: store.reclaim(path, extra=extra)["freed"] for path in paths}
            freed[versions] = store.reclaim(folders, extra=extra)["freed"]
            return freed

        def measured(freed):
            self._oldJob = None
            self._freed.update(freed)
            if self.store.root == store.root:
                self._showOldSizes()
                self._showOldButton()

        def failed(_error):
            self._oldJob = None

        self._oldJob = BackgroundJob(measure, on_success=measured, on_error=failed)
        self._oldJob.start()

    def _showSummary(self):
        """The counts now; the folder's size as last measured, or "measuring"."""
        tools = len([row for row in self.tools if row["current"]])
        measured = self._folderSize.get(self.store.root)
        self.summaryLabel.setText(
            _("{tools} tools · {models} models · {size}").format(
                tools=tools, models=len(self.modelRows),
                size=tools_store.human_size(measured) if measured is not None else _("measuring...")))

    def _measureFolder(self):
        """The folder as it is on disk, each hard-linked file once (the per-tool
        sizes each count the packages they share, torch, in full).

        Measured on a worker: the walk visits every file of every environment
        and of uv's cache, half a million on a full install, which took three
        seconds each time the Manager was opened.
        """
        from SADTCoreLib.worker import BackgroundJob

        root = self.store.root
        if self._sizeJob is not None or not os.path.isdir(root):
            return

        def measured(total):
            self._sizeJob = None
            self._folderSize[root] = total
            if self.store.root == root:
                self._showSummary()

        def failed(_error):
            self._sizeJob = None

        self._sizeJob = BackgroundJob(lambda _progress: tools_store.folder_size(root),
                                      on_success=measured, on_error=failed)
        self._sizeJob.start()

    # ---- actions ---------------------------------------------------------

    def _changed(self):
        reset_client()
        self.refresh()

    def _dataSize(self, path: str) -> int:
        for row in self.modelRows:
            if row["path"] == path:
                return row.get("size") or 1
        return 1

    def _removeInBackground(self, title, steps, prune=True):
        """Run removal steps off the main thread, behind the install's progress window.

        `steps` are `(message, weight, action)`: the bar advances by weight (the
        recorded size), so removing an 8 GB environment moves it more than a
        6 MB model. The package cache is cleaned last, its own step.
        """
        from SADTCoreLib.worker import BackgroundJob

        window = tools_ui.ProgressWindow(self.parent, title)
        total = max(1, sum(weight for _message, weight, _action in steps))

        def task(progress_cb):
            done = 0
            for message, weight, action in steps:
                progress_cb({"message": message, "detail": "", "fraction": min(0.99, done / total)})
                action()
                done += weight
            if prune:
                progress_cb({"message": "Cleaning the package cache",
                             "detail": "packages no tool uses any more", "fraction": 0.99})
                freed = self.store.prune_cache()
                progress_cb({"message": "Done", "detail": f"{tools_store.human_size(freed)} of packages freed",
                             "fraction": 1.0})
            return None

        def finish(_result=None):
            window.close()
            self._changed()

        def failed(error):
            window.close()
            slicer.util.errorDisplay(f"The removal stopped:\n\n{error}")
            self._changed()

        job = BackgroundJob(task, on_success=finish, on_error=failed, on_progress=window.update)
        job.start()
        window.job = job

    def _reclaimText(self, paths) -> str:
        """How much space the removal gives back, and how much stays because it is shared."""
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            space = self.store.reclaim_with_cache(paths)
        finally:
            slicer.app.restoreOverrideCursor()
        size = tools_store.human_size
        text = f"\n\nFrees {size(space['freed'])} on disk."
        if space["shared"]:
            text += (f"\n{size(space['shared'])} more is shared with other files (another tool, "
                     "the package cache, the folder the models came from) and stays until those go too.")
        return text

    def onRemoveTool(self, row):
        """Remove a tool, and if asked what came with it: its data, the tools it needed."""
        if tools_ui.refuse_if_busy(self.parent, self.store, [row["name"]], "remove it"):
            return
        if not row["current"]:
            text = f"Remove the tool {row['name']} ({row['version']})?" + self._reclaimText([row["path"]])
            if slicer.util.confirmOkCancelDisplay(text, windowTitle=_("Remove a tool")):
                self._removeInBackground(
                    f"Removing {row['name']}",
                    [(f"Removing {row['name']} ({row['version']})", row.get("size") or 1,
                      lambda: self.store.remove_tool(row["path"]))])
            return
        name = row["name"]
        needed = self.store.needed_by(name)
        brought = self.store.removable_with(name)
        everything = set(self.store._needs())

        dialog = qt.QDialog(self.parent)
        dialog.setWindowTitle(_("Remove a tool"))
        dialog.setMinimumWidth(520)
        layout = qt.QVBoxLayout(dialog)
        layout.addWidget(design.section_title(_("Remove {tool}?").format(tool=name)))
        if needed:
            warning = qt.QLabel(_("{tools} need it: it will be downloaded again the next time "
                                  "one of them is installed.").format(tools=", ".join(needed)))
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #c77700;")
            layout.addWidget(warning)
        withData = qt.QCheckBox()
        withTools = qt.QCheckBox(_("Also remove the tools it needs that nothing else uses: {tools}")
                                 .format(tools=", ".join(brought)))
        withTools.setVisible(bool(brought))
        withTools.setChecked(bool(brought))
        layout.addWidget(withTools)
        layout.addWidget(withData)
        dataNote = design.hint_label("")
        dataNote.setWordWrap(True)
        layout.addWidget(dataNote)
        freed = qt.QLabel()
        freed.setWordWrap(True)
        layout.addWidget(freed)
        buttons = qt.QHBoxLayout()
        buttons.addStretch(1)
        cancel = design.secondary_button(_("Cancel"))
        remove = design.danger_button(_("Remove"))
        buttons.addWidget(cancel)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        plan = {}

        def compute():
            tools = [name] + (brought if withTools.checked else [])
            # A tool installed at another version (an older one still kept, a
            # newer Slicer's) uses the same DATA folder: its models stay.
            keep = (everything - set(tools)) | {
                tool["name"] for tool in self.tools if not tool["current"]}
            data = self.store.data_of(tools, keep)
            free = [item for item in data if not item["used_by"]]
            kept = [item for item in data if item["used_by"]]
            withData.blockSignals(True)
            withData.setText(_("Also remove their models and test files ({count})").format(count=len(free))
                             if len(tools) > 1 else
                             _("Also remove its models and test files ({count})").format(count=len(free)))
            withData.setEnabled(bool(free))
            if not plan:
                withData.setChecked(bool(free))
            withData.blockSignals(False)
            dataNote.setText(_("Kept, still used: ") + ", ".join(
                f"{item['name']} ({', '.join(item['used_by'])})" for item in kept) if kept else "")
            paths = [tool["path"] for tool in self.tools if tool["current"] and tool["name"] in tools]
            if withData.checked:
                paths += [item["path"] for item in free]
            plan.update(tools=tools, data=free if withData.checked else [], paths=paths)
            freed.setText("<b>" + self._reclaimText(paths).strip().replace("\n", "<br>") + "</b>")

        withTools.toggled.connect(lambda _on: compute())
        withData.toggled.connect(lambda _on: compute())
        cancel.clicked.connect(lambda: dialog.reject())
        remove.clicked.connect(lambda: dialog.accept())
        compute()
        design.apply(dialog)
        if dialog.exec_() != qt.QDialog.Accepted:
            return
        sizes = {tool["name"]: tool.get("size") or 1 for tool in self.tools if tool["current"]}
        steps = []
        for tool in plan["tools"]:
            folder = self.store.tool_dir(tool)
            if folder:
                steps.append((f"Removing {tool}", sizes.get(tool, 1),
                              lambda f=folder: self.store.remove_tool(f, prune=False)))
        for item in plan["data"]:
            steps.append((f"Removing the {item['kind']} {item['name']}",
                          self._dataSize(item["path"]),
                          lambda p=item["path"]: self.store.remove_data(p)))
        self._removeInBackground(f"Removing {name}", steps)

    def onRemoveModel(self, row):
        if tools_ui.refuse_if_busy(self.parent, self.store, row["used_by"], "remove it"):
            return
        text = f"Remove the model {row['name']}?"
        if row["used_by"] and row.get("kind", "model") == "model":
            text += (f"\n\n{', '.join(row['used_by'])} use it and will not run without it "
                     "until it is installed again.")
        text += self._reclaimText([row["path"]])
        if slicer.util.confirmOkCancelDisplay(text, windowTitle=_("Remove")):
            self._removeInBackground(
                f"Removing {row['name']}",
                [(f"Removing the {row.get('kind', 'model')} {row['name']}", row.get("size") or 1,
                  lambda: self.store.remove_model(row["path"]))])

    def onRemoveOldVersions(self):
        old = sorted({row["version"] for row in self.tools
                      if not row["current"] and self.store.is_older(row["version"])},
                     key=tools_store.version_key)
        if not old:
            slicer.util.infoDisplay(_("There is no old version to remove."))
            return
        paths = [os.path.join(self.store.root, f"tools-{version}") for version in old]
        if slicer.util.confirmOkCancelDisplay(
                f"Remove the tools of the old versions {', '.join(old)}?" + self._reclaimText(paths),
                windowTitle=_("Remove old versions")):
            self._removeInBackground(
                "Removing old versions",
                [(f"Removing the tools of versions {', '.join(old)}", 1,
                  self.store.remove_unused_versions)])

    def onRemoveEverything(self):
        if tools_ui.refuse_if_busy(self.parent, self.store, [r["name"] for r in self.tools],
                                   "remove everything"):
            return
        if slicer.util.confirmOkCancelDisplay(
                f"Remove EVERYTHING in\n{self.store.root}\n\nevery tool, environment and model, "
                "for every Slicer that shares this folder? They are downloaded again when needed."
                + self._reclaimText([self.store.root]),
                windowTitle=_("Remove everything")):
            root = self.store.root
            if self.store.lock().holder():
                slicer.util.warningDisplay(_("Another Slicer is installing or removing tools in "
                                             "this folder. Try again when it has finished."))
                return
            entries = sorted(entry for entry in (os.listdir(root) if os.path.isdir(root) else [])
                             if entry != tools_store.StoreLock.NAME)
            steps = [(f"Removing {entry}", 1,
                      lambda p=os.path.join(root, entry): (tools_store._rmtree(p) if os.path.isdir(p)
                                                           else os.remove(p)))
                     for entry in entries]
            steps.append(("Removing the folder", 0, lambda: shutil.rmtree(root, ignore_errors=True)))
            self._removeInBackground("Removing everything", steps, prune=False)

    def onChangeLocation(self):
        folder = qt.QFileDialog.getExistingDirectory(
            self.parent, _("Where should this Slicer keep the tools?"), self.store.root)
        if not folder:
            return
        slicer_settings().setValue(LOCATION_SETTING, os.path.join(folder, tools_store.FOLDER_NAME))
        self._changed()

    def onResetLocation(self):
        slicer_settings().remove(LOCATION_SETTING)
        self._changed()
