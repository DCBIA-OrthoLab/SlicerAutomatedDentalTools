"""The Qt side of installing tools: the confirmation, the progress, the Manager link.

Shared by the SADT Tools Manager module and by every tool panel ("Install
this tool" when it is not installed yet), so the two say the same thing.
"""

import os

import qt
import slicer

from . import reset_client, tool_store
from . import release_watch, releases, tools_store
from .worker import BackgroundJob

MANAGER_MODULE = "SADTToolsManager"


def tools_source() -> tuple:
    """`(tools source, models source)` installs read from (see tool_store).

    Following releases, the source is that release's code, downloaded the first
    time it is needed (~3 MB). Offline, there is none: the caller says so. On a
    tools folder that was never asked, the newest release is asked for first.
    """
    store = tool_store()
    if release_watch.follows_releases(store) and not releases.latest_known(store.root):
        # Never asked in this tools folder: ask now, so the first install is the
        # newest release and not the older one the extension was built with.
        with slicer.util.WaitCursor():
            releases.ensure_known(store.root)
        reset_client()
        store = tool_store()
    if getattr(store, "from_release", False) and not os.path.isdir(store.source):
        try:
            with slicer.util.WaitCursor():
                releases.fetch_source(store.root, store.version)
        except releases.ReleaseError as error:
            print(f"SADTCore: could not download tools {store.version}: {error}")
            return "", store.data_source
    return store.source, store.data_source


def open_manager() -> None:
    slicer.util.selectModule(MANAGER_MODULE)


def confirm_and_install(parent, tool: str, on_done=None) -> None:
    """Ask, then install `tool` and every tool it calls, without freezing Slicer.

    `on_done(installed: list)` is called on success, after the client was reset
    so the panels see the new tools.
    """
    source, data_source = tools_source()
    if not source:
        slicer.util.errorDisplay(
            "The tools could not be downloaded from SADT-VISOR's releases. Check "
            "this computer's internet connection, then try again.", parent=parent)
        return
    store = tool_store()
    try:
        missing = store.incomplete(tool, source)
    except tools_store.InstallError as error:
        slicer.util.errorDisplay(str(error), parent=parent)
        return
    if not missing:
        slicer.util.infoDisplay(f"{tool} is already installed.", parent=parent)
        return
    verb = "Update" if all(store.is_installed(name) for name in missing) else "Install"
    current = store.installed_version(tool)
    if current and current != store.version:
        headline = f"Update {tool} from tools {current} to {store.version}?"
    else:
        headline = f"{verb} {tool} (tools {store.version})?"
    text = f"{headline}\n\n{describe_download(store, tool, source, data_source)}"
    text += (f"\n\nWhere: {store.root}\n\n"
             "You can see and remove installed tools at any time in "
             "Automated Dental Tools > SADT Tools Manager.")
    box = qt.QMessageBox(parent)
    box.setWindowTitle("Install a tool")
    box.setText(text)
    box.setStandardButtons(qt.QMessageBox.Ok | qt.QMessageBox.Cancel)
    tests = qt.QCheckBox(f"Also install the test files "
                         f"({tools_store.human_size(store.testfiles_size(tool, source, data_source))})")
    box.setCheckBox(tests)
    if box.exec_() != qt.QMessageBox.Ok:
        return
    with_tests = tests.checked

    _install_with_progress(parent, store, [tool], source, data_source, on_done,
                           testfiles=with_tests)


def describe_download(store, tool: str, source: str, data_source: str,
                      testfiles: bool = False) -> str:
    """What installing `tool` downloads, tool by tool, in plain words with sizes."""
    from . import local_schema

    estimate = store.estimate(tool, source, data_source, testfiles=testfiles)
    size = tools_store.human_size
    engines = local_schema.facade_targets(tool) or []
    lines = []
    for row in estimate["tools"]:
        if row["name"] == tool:
            role = ""
        elif row["name"] in engines:
            role = f" ({tool} engine)"
        else:
            role = f" (used by {tool})"
        models = f"models {size(row['models'])}" if row["models"] else ""
        environment = f"environment {size(row['environment'])}" if row["environment"] else ""
        if store.is_installed(row["name"]):
            role += " — update" if row["environment"] else " — missing models"
        lines.append(f"  • {row['name']}{role}: " + " + ".join(part for part in (environment, models) if part))
    pending = {row["name"] for row in estimate["tools"]}
    already = [name for name in store.plan(tool, source)
               if store.is_installed(name) and name not in pending]
    if already:
        lines.append(f"  • Already installed: {', '.join(already)}")
    tests = f", test files {size(estimate['testfiles'])}" if testfiles else ""
    lines.append(f"Download: {size(estimate['total'])} "
                 f"(environments {size(estimate['environment'])}, models {size(estimate['models'])}{tests})")
    lines.append(disk_line(store, estimate, html=False))
    return "\n".join(lines)


def disk_line(store, estimate: dict, html: bool = True) -> str:
    """"Needs about X on disk · Y free": in red, with a warning, when it does not fit."""
    size = tools_store.human_size
    needed = tools_store.needed_on_disk(estimate)
    free = tools_store.free_space(store.root)
    text = f"Needs about {size(needed)} on disk · {size(free)} free on this disk"
    if needed > free:
        text += " — NOT ENOUGH SPACE: free some, or change the location in the SADT Tools Manager"
        if html:
            return f"<span style='color: #d13c3c; font-weight: 600;'>{text}</span>"
    return text


def not_installed_card(parent, layout, tool: str, on_installed) -> None:
    """What a tool panel shows instead of its form when the tool is not installed.

    Nothing to fill in, nothing to run: what the tool is missing, how much it
    downloads, where it goes, and one button.
    """
    from . import design

    store = tool_store()
    source, data_source = tools_source()
    layout.addWidget(design.section_title(f"{tool} is not installed on this computer"))
    text = ("To use it, install it once: the tool runs on this computer, "
            "in its own environment, and nothing is sent anywhere.")
    details = ""
    if source:
        try:
            details = describe_download(store, tool, source, data_source)
        except tools_store.InstallError as error:
            details = str(error)
    else:
        details = "The tools could not be downloaded: check this computer's internet connection."
    body = qt.QLabel(f"{text}\n\n{details}\n\nInstalled into: {store.root}")
    body.setWordWrap(True)
    body.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
    layout.addWidget(body)
    install = design.primary_button(f"Install {tool}...")
    install.clicked.connect(lambda: confirm_and_install(parent, tool, on_done=on_installed))
    layout.addWidget(install)
    manage = design.secondary_button("Open the SADT Tools Manager")
    manage.clicked.connect(open_manager)
    layout.addWidget(manage)


def available_tools(source: str) -> list:
    """Every tool the source offers, facades (ALI, AREG) included, sorted."""
    from . import local_schema

    names = []
    tools = os.path.join(source, "tools") if source else ""
    if tools and os.path.isdir(tools):
        for walk_root, dirs, files in os.walk(tools):
            dirs[:] = [d for d in dirs if not d.startswith((".", "_")) and d not in ("src", "tests")]
            if "pyproject.toml" in files:
                with open(os.path.join(walk_root, "pyproject.toml"), encoding="utf-8") as handle:
                    text = handle.read()
                if "[tool.sadt]" in text and "tool = true" in text:
                    names.append(os.path.basename(walk_root))
                    dirs[:] = []
    names += [facade for facade in local_schema.FACADES
              if all(target in names for target in local_schema.facade_targets(facade))]
    return sorted(set(names), key=str.lower)


def choose_and_install(parent, on_done=None) -> None:
    """A checklist of every tool: tick what you want, what they need is ticked for you.

    A tool another ticked tool needs is ticked and locked, saying by whom; an
    installed tool is shown as such. The download is summed as you tick.
    """
    from . import design

    source, data_source = tools_source()
    if not source:
        slicer.util.errorDisplay("The tools could not be downloaded from SADT-VISOR's releases: "
                                 "check this computer's internet connection.", parent=parent)
        return
    store = tool_store()
    names = available_tools(source)
    plans = {}
    for name in names:
        try:
            plans[name] = store.plan(name, source)
        except tools_store.InstallError:
            plans[name] = [name]

    dialog = qt.QDialog(parent)
    dialog.setWindowTitle("Install tools")
    dialog.resize(520, 640)
    layout = qt.QVBoxLayout(dialog)
    layout.addWidget(design.section_title("Which tools do you want on this computer?"))
    layout.addWidget(design.hint_label(
        "The tools a ticked tool needs are ticked for you and cannot be unticked."))
    scroll = qt.QScrollArea()
    scroll.setWidgetResizable(True)
    inner = qt.QWidget()
    listLayout = qt.QVBoxLayout(inner)
    boxes = {}
    for name in names:
        box = qt.QCheckBox(name)
        listLayout.addWidget(box)
        boxes[name] = box
    listLayout.addStretch(1)
    scroll.setWidget(inner)
    layout.addWidget(scroll, 1)
    showInstalled = qt.QCheckBox("Show installed tools")
    layout.addWidget(showInstalled)
    withTests = qt.QCheckBox("Also install the test files of the ticked tools")
    layout.addWidget(withTests)
    summary = qt.QLabel()
    summary.setWordWrap(True)
    layout.addWidget(summary)
    buttons = qt.QHBoxLayout()
    buttons.addStretch(1)
    cancel = design.secondary_button("Cancel")
    install = design.primary_button("Install")
    buttons.addWidget(cancel)
    buttons.addWidget(install)
    layout.addLayout(buttons)
    chosen = set()

    done = {name: all(store.is_installed(tool) for tool in plans[name]) for name in names}
    from . import local_schema

    engines = {engine for facade in local_schema.FACADES for engine in local_schema.facade_targets(facade)}

    def installed(name):
        return done[name]

    def refresh():
        required = {}
        for name in chosen:
            for needed in plans[name]:
                if needed != name and needed in boxes:
                    required.setdefault(needed, []).append(name)
        for name, box in boxes.items():
            box.blockSignals(True)
            # Installed tools are hidden unless asked for; an engine of a module
            # (ALI_CBCT of ALI) is listed only when a ticked tool needs it.
            if installed(name):
                box.setVisible(showInstalled.checked)
            else:
                box.setVisible(name not in engines or name in required)
            if installed(name):
                box.setChecked(True)
                box.setEnabled(False)
                box.setText(f"{name}   ✓ installed")
                box.setStyleSheet(f"QCheckBox {{ color: {design.tokens()['SUCCESS']}; }}")
            elif name in required and name not in chosen:
                box.setChecked(True)
                box.setEnabled(False)
                box.setText(f"{name}   (needed by {', '.join(sorted(required[name]))})")
            else:
                box.setEnabled(True)
                box.setChecked(name in chosen)
                box.setText(name)
            box.blockSignals(False)
        wanted = sorted(chosen)
        if not wanted:
            summary.setText("Nothing to install yet.")
            install.setEnabled(False)
            return
        estimate = store.estimate(wanted, source, data_source, testfiles=withTests.checked)
        size = tools_store.human_size
        tests = f", test files {size(estimate['testfiles'])}" if withTests.checked else ""
        summary.setText(
            f"<b>{len(estimate['tools'])} tools to install · download {size(estimate['total'])}</b>"
            f" (environments {size(estimate['environment'])}, models {size(estimate['models'])}{tests})"
            f"<br>{disk_line(store, estimate)}"
            f"<br>Into {store.root}")
        install.setEnabled(bool(estimate["tools"]))

    def toggled(name, on):
        if on:
            chosen.add(name)
        else:
            chosen.discard(name)
        refresh()

    for name, box in boxes.items():
        box.toggled.connect(lambda on, n=name: toggled(n, on))
    # A lambda: PythonQt hands clicked() its `checked` flag, which reject() refuses.
    cancel.clicked.connect(lambda: dialog.reject())
    showInstalled.toggled.connect(lambda _on: refresh())
    withTests.toggled.connect(lambda _on: refresh())

    def go():
        wanted = sorted(chosen)
        dialog.accept()
        _install_with_progress(parent, store, wanted, source, data_source, on_done,
                               testfiles=withTests.checked)

    install.clicked.connect(lambda: go())
    refresh()
    design.apply(dialog)
    dialog.exec_()


class _ProgressWindow:
    """What an install is doing, so nobody wonders whether it froze.

    The step, the overall percentage (weighted by download size), a live detail
    (what uv has written, what has been downloaded), the elapsed time, and a
    spinner that turns even while a step reports nothing.
    """

    SPINNER = "◐◓◑◒"

    def __init__(self, parent, title):
        from . import design

        self.dialog = qt.QDialog(parent)
        self.dialog.setWindowTitle(title)
        self.dialog.setModal(True)
        self.dialog.setMinimumWidth(480)
        layout = qt.QVBoxLayout(self.dialog)
        self.step = design.section_title("Starting...")
        layout.addWidget(self.step)
        self.bar = qt.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)
        self.detail = qt.QLabel("")
        layout.addWidget(self.detail)
        self.clock = design.hint_label("")
        layout.addWidget(self.clock)
        self._started = __import__("time").monotonic()
        self._tick = 0
        self._timer = qt.QTimer()
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._turn)
        self._timer.start()
        design.apply(self.dialog)
        # Esc, or the window's close button, would hide the window and leave the
        # job running unseen, with Slicer free to start a second one: shown
        # again until the job itself ends.
        self._open = True
        self.dialog.rejected.connect(lambda: self.dialog.show() if self._open else None)
        self.dialog.show()

    def _turn(self):
        import time

        self._tick += 1
        seconds = int(time.monotonic() - self._started)
        self.clock.setText(f"{self.SPINNER[self._tick % 4]}  {seconds // 60}:{seconds % 60:02d} elapsed")

    def update(self, payload):
        if isinstance(payload, dict):
            if payload.get("message"):
                self.step.setText(payload["message"])
            if payload.get("detail") is not None:
                self.detail.setText(payload["detail"])
            if payload.get("fraction") is not None:
                self.bar.setValue(int(round(payload["fraction"] * 100)))
        elif payload:
            self.step.setText(str(payload))

    def close(self):
        self._open = False
        self._timer.stop()
        self.dialog.close()


# Public: the Manager shows its removals in the same window.
ProgressWindow = _ProgressWindow


def busy_with(store, tools) -> list:
    """Which of `tools` a run in this Slicer is using now, itself or through a caller."""
    from .local_client import running_tools

    running = running_tools()
    if not running:
        return []
    in_use = set(running) | store._closure(running, store._needs())
    return sorted(set(tools) & in_use)


def refuse_if_busy(parent, store, tools, action: str) -> bool:
    """Say so and return True when a run in this Slicer uses one of `tools`."""
    busy = busy_with(store, tools)
    if busy:
        slicer.util.warningDisplay(
            f"{', '.join(busy)} is running in this Slicer. Wait until the run has ended, "
            f"then {action} again.", parent=parent)
    return bool(busy)


def _install_with_progress(parent, store, wanted, source, data_source, on_done,
                           testfiles: bool = False) -> None:
    try:
        replaced = [tool for tool in store.missing(wanted, source) if store.is_installed(tool)]
    except tools_store.InstallError:
        replaced = []
    if refuse_if_busy(parent, store, replaced, "update"):
        return
    window = _ProgressWindow(parent, "Installing " + ", ".join(wanted))

    def done(installed):
        window.close()
        store.register_user(os.path.normpath(slicer.app.slicerHome), slicer.app.applicationVersion)
        reset_client()
        slicer.util.infoDisplay(
            f"Installed: {', '.join(installed) or 'nothing new'}\nin {store.root}", parent=parent)
        if on_done:
            on_done(installed)

    def failed(error):
        window.close()
        slicer.util.errorDisplay(f"The installation failed:\n\n{error}", parent=parent)

    job = BackgroundJob(lambda progress_cb: store.install(wanted, source, data_source,
                                                         progress=progress_cb, testfiles=testfiles),
                        on_success=done, on_error=failed, on_progress=window.update)
    job.start()
    # Kept alive with the window: a job nobody references is collected mid-run.
    window.job = job


def install_many(parent, names, on_done=None) -> None:
    """Install several tools (and what they need) after one confirmation: Update all."""
    source, data_source = tools_source()
    store = tool_store()
    estimate = store.estimate(list(names), source, data_source)
    size = tools_store.human_size
    # The names on a line of their own, which wraps: in the headline, a long
    # list made the dialog as wide as the list.
    text = (f"Update {len(names)} tool(s) to tools {store.version}?\n\n"
            f"{', '.join(names)}\n\n"
            f"{len(estimate['tools'])} tool(s) with what they call, download {size(estimate['total'])} "
            f"(environments {size(estimate['environment'])}, models {size(estimate['models'])}).\n\n"
            "The previous version stays installed until you remove it.")
    if slicer.util.confirmOkCancelDisplay(text, windowTitle="Update tools", parent=parent):
        _install_with_progress(parent, store, list(names), source, data_source, on_done)
