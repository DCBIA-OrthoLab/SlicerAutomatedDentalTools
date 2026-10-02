import logging
import os
import qt
import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
import json
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

# Max number of automatic "fix the parameters from the error and retry"
# attempts after a real tool execution fails (see AgentWidget.runToolWithRepair).
MAX_REPAIR_ATTEMPTS = 2

# Ollama model used by the agent. Kept in sync with Agent_CLI (qwen3:8b); the
# CheckDependencies button pulls this model.
MODEL_NAME = "qwen3:8b"


# ---------------------------------------------------------------------------
# Ollama auto-installation
#
# Most end users of this extension are clinicians/researchers who are not
# comfortable installing command-line software. When Ollama is missing we can
# download the *official* build (which detects and uses the GPU) into a
# user-writable folder - no admin/sudo, no terminal - and launch its server.
#
# We deliberately avoid the Linux "curl | sudo sh" installer and the Ubuntu
# snap package: the former needs a root password we can't supply from the GUI,
# the latter is sandboxed and cannot reach the GPU (falls back to CPU, ~80-90s
# per answer). The portable archives below ship the CUDA/Metal libraries and
# run entirely from a user folder.
# ---------------------------------------------------------------------------

def _agent_install_dir():
    """User-writable folder where we keep our bundled Ollama."""
    base = os.environ.get("SLICER_AGENT_HOME") or os.path.join(
        os.path.expanduser("~"), ".slicer_agent")
    return os.path.join(base, "ollama")


def _bundled_ollama_binary():
    """Path to an Ollama binary we installed previously, or None."""
    d = _agent_install_dir()
    for c in (
        os.path.join(d, "bin", "ollama"),                                   # Linux .tgz
        os.path.join(d, "ollama.exe"),                                      # Windows .zip
        os.path.join(d, "Ollama.app", "Contents", "Resources", "ollama"),  # macOS .app
    ):
        if os.path.exists(c):
            return c
    return None


def _ollama_env(binary):
    """Environment for running our bundled Ollama (so it finds its own libs)."""
    env = dict(os.environ)
    libdir = os.path.join(_agent_install_dir(), "lib", "ollama")  # Linux layout
    if os.path.isdir(libdir):
        env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _ollama_responding(timeout=0.5):
    """True if an Ollama server answers on the default port (127.0.0.1:11434)."""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", 11434))
        return True
    except Exception:
        return False
    finally:
        s.close()


def _download_with_progress(url, dest, title):
    """Download url -> dest showing a modal Qt progress dialog. Raises on cancel."""
    import urllib.request
    progress = qt.QProgressDialog(title, "Cancel", 0, 100)
    progress.setWindowModality(qt.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    slicer.app.processEvents()

    def hook(count, block_size, total_size):
        if total_size > 0:
            progress.setValue(min(100, int(count * block_size * 100 / total_size)))
        slicer.app.processEvents()
        if progress.wasCanceled():
            raise RuntimeError("Download cancelled by the user")

    try:
        urllib.request.urlretrieve(url, dest, hook)
    finally:
        progress.close()


def install_official_ollama():
    """Download the official (GPU-enabled) Ollama into a user folder.

    Returns the path to the ollama binary. Raises on any failure so the caller
    can fall back to the manual-install message.
    """

    system = platform.system()
    machine = platform.machine().lower()
    install_dir = _agent_install_dir()
    os.makedirs(install_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp()

    if system == "Linux":
        arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
        url = f"https://ollama.com/download/ollama-linux-{arch}.tgz"
        archive = os.path.join(tmpdir, "ollama.tgz")
        _download_with_progress(url, archive, "Downloading Ollama (GPU build)…")
        with tarfile.open(archive) as tf:
            tf.extractall(install_dir)
        binary = os.path.join(install_dir, "bin", "ollama")
        os.chmod(binary, 0o755)

    elif system == "Windows":
        arch = "arm64" if "arm" in machine else "amd64"
        url = f"https://ollama.com/download/ollama-windows-{arch}.zip"
        archive = os.path.join(tmpdir, "ollama.zip")
        _download_with_progress(url, archive, "Downloading Ollama (GPU build)…")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(install_dir)
        binary = os.path.join(install_dir, "ollama.exe")

    elif system == "Darwin":
        url = "https://ollama.com/download/Ollama-darwin.zip"
        archive = os.path.join(tmpdir, "ollama.zip")
        _download_with_progress(url, archive, "Downloading Ollama…")
        # Use ditto so the .app bundle's symlinks and exec bits survive.
        subprocess.run(["ditto", "-x", "-k", archive, install_dir], check=True)
        binary = os.path.join(install_dir, "Ollama.app", "Contents", "Resources", "ollama")
        # Strip the Gatekeeper quarantine flag so it launches without a prompt.
        subprocess.run(
            ["xattr", "-dr", "com.apple.quarantine",
             os.path.join(install_dir, "Ollama.app")],
            check=False,
        )
        os.chmod(binary, 0o755)

    else:
        raise RuntimeError(f"Unsupported operating system: {system}")

    if not os.path.exists(binary):
        raise RuntimeError("Ollama download finished but the binary was not found.")
    return binary


def ensure_ollama_server(binary):
    """Make sure an Ollama server is reachable, starting our bundled one if needed.

    Returns True once the server answers, False on timeout.
    """

    if _ollama_responding():
        return True

    creationflags = 0
    if platform.system() == "Windows":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    subprocess.Popen(
        [binary, "serve"],
        env=_ollama_env(binary),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    # Give the server up to ~30s to come up, keeping the UI responsive.
    for _ in range(60):
        if _ollama_responding():
            return True
        slicer.app.processEvents()
        qt.QThread.msleep(500)
    return _ollama_responding()


def ensure_agent_ollama_running():
    """Best-effort: make sure an Ollama server is up before talking to it.

    Cheap when the server is already running (a single socket probe). If it is
    down but we have a system or previously-bundled Ollama, start it. Used both
    at dependency-check time and before each conversation, since a server we
    launched ourselves does not survive a Slicer restart.
    """
    if _ollama_responding():
        return True
    binary = shutil.which("ollama") or _bundled_ollama_binary()
    if binary is None:
        return False
    return ensure_ollama_server(binary)


class DropZone(qt.QFrame):
    """Drag & drop area for multiple files/folders. Emits a list of local paths."""
    dropped = qt.Signal(list)

    def __init__(self, parent=None, title="Drop files or folders here",objectName = "dropZone"):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName(objectName)
        self.setFrameShape(qt.QFrame.StyledPanel)
        self.setFrameShadow(qt.QFrame.Plain)
        self.setMinimumHeight(100)

        self._label = qt.QLabel(title)
        self._label.alignment = qt.Qt.AlignCenter

        lay = qt.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(self._label)

        # style "drop zone" - palette(...) functions resolve against the
        # active Slicer theme (light or dark) at render time, so this never
        # needs to special-case dark mode itself.
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed palette(mid);
                border-radius: 8px;
                background: palette(base);
            }
            QLabel {
                color: palette(window-text);
                font-weight: 800;
            }
        """)

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = []
        for u in urls:
            if u.isLocalFile():
                p = u.toLocalFile()
                if p:
                    paths.append(p)

        if paths:
            self.dropped.emit(paths)

        event.acceptProposedAction()
    
    def setSummary(self, paths):
        if not paths:
            self._label.setText("Drop files or folders here")
            return

        preview = "\n".join(paths[:3])
        more = "" if len(paths) <= 3 else f"\n... (+{len(paths)-3} more)"
        self._label.setText(f"Dropped {len(paths)} item(s):\n{preview}{more}")

#
# Agent
#


class Agent(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Agent")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Automated Dental Tools")]
        self.parent.dependencies = []
        self.parent.contributors = ["Alexandre Buisson (University of North Carolina)"]
        self.parent.helpText = _("""
Agent is the graphical front-end of the AI Agent extension. Describe what you
want to do with your dental/orthodontic imaging data in natural language and the
agent (powered locally by the qwen3:8b model through Ollama) selects the right
tool, extracts its parameters and runs it. Use the 'Check' button once to install
the required dependencies and pull the model.
""")
        self.parent.acknowledgementText = _("""
This module was developed at the University of North Carolina as part of the
AI Agent extension for 3D Slicer.
""")


#
# AgentParameterNode
#


@parameterNodeWrapper
class AgentParameterNode:
    """
    Parameters needed by the module.

    prompt - The natural-language request typed by the user.
    folders - The list of input files/folders dropped in the drop zone.
    modeagent - The interaction mode ("Agent (Automated)" or consultant).
    """

    prompt: str
    # list[str], not a bare list: the serializer logs
    # "Unexpected list[] type arg length" at every Slicer start for the bare
    # form, then falls back to list[Any]. The paths put here come from
    # os.path.normpath, so they are strings.
    folders: list[str]
    modeagent: str

#
# AgentWidget
#

# import qt

class TextEditEnterFilter(qt.QObject):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.callback = callback

    def eventFilter(self, obj, event):
        import qt

        if event.type() == qt.QEvent.KeyPress:
            key = event.key()
            modifiers = event.modifiers()

            # Enter
            if key in (qt.Qt.Key_Return, qt.Qt.Key_Enter):
                # Shift+Enter -> New line
                if modifiers & qt.Qt.ShiftModifier:
                    return False
                self.callback()
                return True

        return False



class AgentWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.CliStartTime=0
        # Earlier turns of this conversation, sent to Agent_CLI on every
        # request so the model isn't limited to the latest message (e.g. a
        # follow-up that only supplies a previously-missing parameter).
        self.conversationHistory = []

    def setup(self) -> None:
        import qt
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/Agent.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        self.dropZone = DropZone(objectName="dropZoneInput")

        self.dropZoneButton = qt.QPushButton("x")

        self.dropZoneButton.setStyleSheet("""
        QPushButton {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #e74c3c, /* bright red */
                                            stop:1 #c0392b); /* slightly darker red */
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 10pt;
            padding: 8px;
            margin-top: 4px;
        }

        QPushButton:hover:!pressed {
            /* lighter red on hover */
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #e9685a,
                                            stop:1 #d64d3c);
        }

        QPushButton:pressed {
            /* darker red on click (pressed effect) */
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #a93226,
                                            stop:1 #8e241b);
        }

        QPushButton:disabled {
            /* disabled state (grey) */
            background-color: #bdc3c7;
            color: #95a5a6;
        }
        """)

        self.dropZoneLayout = qt.QHBoxLayout()

        self.dropZoneLayout.setContentsMargins(0, 0, 0, 0)
        self.dropZoneLayout.setSpacing(5)

        self.dropZoneLayout.addWidget(self.dropZone,98)
        self.dropZoneLayout.addWidget(self.dropZoneButton,2)

        self.dropZoneContainer = qt.QWidget()
        self.dropZoneContainer.setLayout(self.dropZoneLayout)

        self.ui.formLayout_2.addRow("Drop zone", self.dropZoneContainer)

        self.dropZone.dropped.connect(self.onDroppedPaths)

        # Retrieve the list
        self.droppedInputPaths = []

        self.enterFilter = TextEditEnterFilter(self.ui.textEdit_2, self.onReturnPressed)
        self.ui.textEdit_2.installEventFilter(self.enterFilter)

        te = self.ui.textEdit_2

        fm = te.fontMetrics()
        line_height = fm.lineSpacing()

        min_lines = 1
        max_lines = 6

        min_height = int(line_height * min_lines + te.frameWidth * 2 + 8)
        max_height = int(line_height * max_lines + te.frameWidth * 2 + 8)

        te.setMinimumHeight(min_height)
        te.setMaximumHeight(max_height)

        te.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)

        te.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)

        te.setFixedHeight(min_height)

        def _autoResizeTextEdit():
            doc_height = te.document.size.height()
            h = int(doc_height) + 10  # little padding
            if h < min_height:
                h = min_height
            if h > max_height:
                h = max_height
            te.setFixedHeight(h)

        te.textChanged.connect(_autoResizeTextEdit)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        ui_widget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AgentLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.SaveButton.connect("clicked(bool)", self.OnSaveButton)
        self.ui.ClearButton.connect("clicked(bool)", self.OnClearButton)
        self.ui.RetrieveButton.connect("clicked(bool)",self.OnRetrieveButton)
        self.ui.CheckButton.connect("clicked(bool)",self.CheckDependencies)
        
        # Connect UI signals to checkCanApply
        self.ui.textEdit_2.connect("textChanged()", self._checkCanApply)
        self.ui.comboBox.currentIndexChanged.connect(self._checkCanApply)
        self.dropZoneButton.connect("clicked(bool)",self.clearDropzone)


        self.ui.label_4.hide()

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Re-color the "LLM response will appear here..." placeholder once
        # the widget has actually been painted (grab() right now, before the
        # first paint event, can't sample real pixels yet).
        qt.QTimer.singleShot(0, self._applyPlaceholderTheme)

    def onDroppedPaths(self, paths):
        norm = []
        seen = set()
        for p in paths:
            p = os.path.normpath(p)
            if p not in seen:
                seen.add(p)
                norm.append(p)
        for path in norm:
            self.droppedInputPaths.append(path)

        # Show summary in the drop area
        self.dropZone.setSummary(self.droppedInputPaths)
        self._checkCanApply()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        # if self._parameterNode:
        #     self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        #     self._parameterNodeGuiTag = None
        #     self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes only if they are not already set
        if not self._parameterNode.prompt:
            self._parameterNode.prompt = ""
        if not self._parameterNode.folders:
            self._parameterNode.folders = []
        if not self._parameterNode.modeagent:
            self._parameterNode.modeagent = "Agent (Automated)"


    def setParameterNode(self, input_parameter_node: AgentParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        self._parameterNode = input_parameter_node

        if self._parameterNode:
            self.ui.textEdit_2.blockSignals(True)
            self.ui.comboBox.blockSignals(True)

            self.ui.textEdit_2.setPlainText(self._parameterNode.prompt or "")
            self.ui.comboBox.setCurrentText(self._parameterNode.modeagent or "Agent (Automated)")

            self.ui.textEdit_2.blockSignals(False)
            self.ui.comboBox.blockSignals(False)

            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if not self._parameterNode:
            self.ui.applyButton.enabled = False
            self.ui.SaveButton.enabled = False
            self.ui.ClearButton.enabled = False
            self.ui.CheckButton.enabled = False
            return
        
        # Update parameter node values from UI
        self._parameterNode.prompt = self.ui.textEdit_2.toPlainText()
        self._parameterNode.modeagent = self.ui.comboBox.currentText
        self._parameterNode.folders = self.droppedInputPaths
        
        # Check if all required fields are filled
        has_prompt = self._parameterNode.prompt.strip() != ""
        has_folders = self._parameterNode.folders != []

        if self._parameterNode.modeagent == "Agent (Automated)":
            if has_prompt and has_folders and not self.ui.label_4.isVisible():
                self.ui.applyButton.toolTip = _("Click to give your prompt to the agent")
                self.ui.applyButton.enabled = True
            else:
                missing = []
                if not has_prompt:
                    missing.append("prompt")
                if not has_folders:
                    missing.append("folders")
                self.ui.applyButton.toolTip = _(f"Fill: {', '.join(missing)}")
                self.ui.applyButton.enabled = False
        else:
            if has_prompt and not self.ui.label_4.isVisible():
                self.ui.applyButton.toolTip = _("Click to give your prompt to the agent")
                self.ui.applyButton.enabled = True
            else:
                missing = []
                if not has_prompt:
                    missing.append("prompt")
                self.ui.applyButton.toolTip = _(f"Fill: {', '.join(missing)}")
                self.ui.applyButton.enabled = False


        if self.ui.textEdit.toPlainText()!="":
            self.ui.SaveButton.toolTip = _("Click to save your chat with the agent")
            self.ui.SaveButton.enabled = True
            self.ui.ClearButton.toolTip = _("Click to clear your chat with the agent")
            self.ui.ClearButton.enabled = True
        else:
            self.ui.ClearButton.toolTip = _(f"Start a chat with the LLM to be able to clear it")
            self.ui.ClearButton.enabled = False
            self.ui.SaveButton.toolTip = _(f"Start a chat with the LLM to be able to save it")
            self.ui.SaveButton.enabled = False

    def _isDarkBackground(self, widget):
        """
        Sample the actual rendered pixels of `widget` to tell light from
        dark theme. QPalette roles (even WindowText/Text) turned out to
        keep returning the default *light*-theme values in Slicer's dark
        style, so they can't be trusted here - grabbing the real on-screen
        pixmap is the only thing that reflects what's actually painted,
        regardless of how Slicer implements the theme under the hood.
        """
        try:
            pixmap = widget.grab()
            if pixmap.isNull() or pixmap.width() < 1 or pixmap.height() < 1:
                return False
            image = pixmap.toImage()
            x = min(5, image.width() - 1)
            y = min(5, image.height() - 1)
            color = image.pixelColor(x, y)
            luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            return luminance < 128
        except Exception:
            return False

    def _applyPlaceholderTheme(self):
        """
        Recolor the "LLM response will appear here..." placeholder text.
        The static "color: palette(mid);" from the .ui ended up unreadable
        in both themes (too light on light, invisible on dark) - same root
        cause as the chat bubbles, so reuse the pixel-sampling detection
        and bake in a literal color instead of trusting palette roles.
        """
        base_style = self.ui.textEdit.property("_baseStyleSheet")
        if base_style is None:
            base_style = self.ui.textEdit.styleSheet
            self.ui.textEdit.setProperty("_baseStyleSheet", base_style)

        is_dark = self._isDarkBackground(self.ui.textEdit)
        placeholder_color = "#cfd8dc" if is_dark else "#5a6268"
        self.ui.textEdit.setStyleSheet(
            f"{base_style}\nQTextEdit {{ color: {placeholder_color}; }}"
        )

    def _scrollChatToEnd(self):
        """Move the cursor to the end of the chat view and keep it visible."""
        cursor = self.ui.textEdit.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        self.ui.textEdit.setTextCursor(cursor)
        self.ui.textEdit.ensureCursorVisible()

    def _insertChatBubble(self, inner_html, color, align):
        """
        Insert a single chat bubble into the chat view. `align` is "right" for
        user messages and "left" for agent messages; the margins mirror
        accordingly. Shared by add_user_message and add_agent_message so the
        bubble markup lives in a single place.
        """
        margin = "5px 20px 5px 0" if align == "right" else "5px 0 5px 20px"
        bubble = (
            f'<div style="color: {color}; padding: 6px 15px; border-radius: 999px; '
            f'margin: {margin}; display: inline-block; white-space: pre-wrap; '
            f'font-family: Segoe UI, Arial, sans-serif; font-size: 11pt;">{inner_html}</div>'
        )
        if align == "right":
            row = (
                f'<table width="100%"><tr><td width="20%"></td>'
                f'<td width="80%" align="right">{bubble}</td></tr></table>'
            )
        else:
            row = (
                f'<table width="100%"><tr><td width="80%" align="left">{bubble}</td>'
                f'<td width="20%"></td></tr></table>'
            )
        self.ui.textEdit.insertHtml(row)
        self._scrollChatToEnd()

    def add_user_message(self, msg):
        accent = "#5dade2" if self._isDarkBackground(self.ui.textEdit) else "#3498db"
        self._insertChatBubble(self.logic.to_html(msg), accent, "right")

        content = msg[2:] if msg.startswith("👨:") else msg
        self._appendHistory("user", content)

    def add_agent_message(self, msg):
        text_color = "#ecf0f1" if self._isDarkBackground(self.ui.textEdit) else "#1c2833"
        self._insertChatBubble(f'<b>🤖:</b> {self.logic.to_html(msg)}', text_color, "left")

        self._appendHistory("assistant", msg)

    def _appendHistory(self, role, content):
        """Append a turn to the conversation history sent to Agent_CLI, capped
        to the last MAX_HISTORY_ENTRIES so the prompt doesn't grow unbounded
        over a long chat."""
        MAX_HISTORY_ENTRIES = 40
        self.conversationHistory.append({"role": role, "content": content})
        if len(self.conversationHistory) > MAX_HISTORY_ENTRIES:
            self.conversationHistory = self.conversationHistory[-MAX_HISTORY_ENTRIES:]

    def onApplyButton(self) -> None:
        self.CliStartTime = time.time()
        self.ui.label_4.setVisible(True)
        slicer.app.processEvents()

        # A server we auto-installed does not survive a Slicer restart, so make
        # sure one is running before we hand the request to Agent_CLI.
        ensure_agent_ollama_running()

        # Snapshot history BEFORE adding this turn's user message, since
        # Agent_CLI.py appends the current prompt itself - including it here
        # too would duplicate the last user turn.
        history_snapshot = list(self.conversationHistory)

        message = "👨:" + self._parameterNode.prompt
        self.add_user_message(message)

        self.droppedInputPaths = self.logic.normalize_folders(self.droppedInputPaths)

        if not self.droppedInputPaths:
            self.droppedInputPaths.append('nothing')

        cli_params = {
            # Agent_CLI declares `folders` as a plain string parameter and
            # splits it on "," - pass a comma-separated string, not a list.
            "folders": ",".join(self.droppedInputPaths),
            "prompt": self._parameterNode.prompt,
            "modeagent": self._parameterNode.modeagent,
            "temp_folder":slicer.util.tempDirectory(),
            "history": json.dumps(history_snapshot)
        }
        CLI = slicer.modules.agent_cli
            
        self.cliNode = slicer.cli.run(CLI, None, cli_params)

        if 'nothing' in self.droppedInputPaths:
            self.droppedInputPaths.remove('nothing')

        self.addObserver(self.cliNode, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)

        self.ui.applyButton.enabled = False
        self.ui.textEdit_2.clear()


    def onCliUpdated(self, caller, event):
        cli_node = caller

        status = cli_node.GetStatus()

        if status & slicer.vtkMRMLCommandLineModuleNode.Completed or \
           status & slicer.vtkMRMLCommandLineModuleNode.Cancelled:

            self.removeObserver(cli_node, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)

            self.ui.applyButton.enabled = True

            output_text = cli_node.GetOutputText()
            print(output_text)

            if self._parameterNode.modeagent == "Agent (Automated)":
                try:
                    message = json.loads(output_text)
                except (json.JSONDecodeError, TypeError):
                    error_text = cli_node.GetErrorText() or "no output from Agent_CLI."
                    self.add_agent_message(
                        f"Agent_CLI failed to run.\n\n{error_text[-1000:]}\n\n"
                        f"{self.logic._suggestFixFor(error_text)}"
                    )
                    self.ui.label_4.setVisible(False)
                    self._checkCanApply()
                    return

                if message.get("error"):
                    traceback_text = cli_node.GetErrorText()
                    details = f"\n\nFull traceback (see also the Slicer Python console):\n{traceback_text[-1500:]}" if traceback_text else ""
                    self.add_agent_message(
                        f"The agent hit an error and couldn't process your request:\n\n{message['error']}"
                        f"{details}\n\n"
                        f"{self.logic._suggestFixFor(message['error'] + ' ' + (traceback_text or ''))}"
                    )
                    self.ui.label_4.setVisible(False)
                    self._checkCanApply()
                    return

                selected_tool = message.get("tool",None)
                missing_required = message.get("missing_required",[])
                params = message.get("parameters",{})
                cli_args = message.get("command",[])

                if selected_tool:
                    if missing_required == []:
                        output_text = f"""After reflection, I would like to run {selected_tool}. Click the Yes button to launch the module if the parameters look good to you."""
                        self.add_agent_message(output_text)

                        parameters = "\n-".join(f"{key}={value}" for key, value in params.items())

                        reply = qt.QMessageBox.question(
                            None,
                            f"Run {selected_tool}",
                            f"The agent wants to run {selected_tool} with these parameters:\n\n-{parameters}\n\nIf it looks good to you, click the Yes button, otherwise click No.",
                            qt.QMessageBox.Yes | qt.QMessageBox.No
                        )

                        if reply == qt.QMessageBox.Yes:
                            output_text=f"\nRunning:{selected_tool}\n"
                            self.add_agent_message(output_text)

                            self.runToolWithRepair(selected_tool, params, cli_args)

                    else:
                        missing = "\n-".join(missing_required)
                        known_section = ""
                        if params:
                            known = "\n-".join(f"{key}={value}" for key, value in params.items())
                            known_section = f"\n\nAlready known parameters:\n-{known}"
                        output_text = (
                            f"After reflection, I would like to run {selected_tool}, but for this I need these parameters:\n\n-{missing}"
                            f"{known_section}\n\nPlease give me the missing value(s) above."
                        )
                        self.add_agent_message(output_text)
                        
                else:
                    output_text = f"""After reflection, I wasn't able to choose a module to run. Try to explain your need in another way."""
                    self.add_agent_message(output_text)
            else:
                self.add_agent_message(output_text)

            self.ui.label_4.setVisible(False)
            self._checkCanApply()
        
        act_time = time.time()
        total_time = round(act_time-self.CliStartTime,2)
        new_text = f"LLM is thinking ({total_time}s)"
        self.ui.label_4.setText(new_text)

    def runToolWithRepair(self, tool_name, params, cli_args):
        """
        Run cli_args (the real underlying CLI tool, e.g. ALI_CBCT.py). On
        failure, ask the LLM to propose corrected parameters from the error
        output and retry, bounded to MAX_REPAIR_ATTEMPTS. A Yes/No
        confirmation is required before each retry since these are real,
        potentially expensive medical-imaging jobs.
        """

        attempt = 0
        current_params = dict(params or {})
        current_cli_args = list(cli_args)

        while True:
            self.ui.label_4.setText(f"LLM is running {tool_name}")
            slicer.app.processEvents()

            try:
                result = subprocess.run(current_cli_args, capture_output=True, text=True)
            except Exception as e:
                # The tool couldn't even be launched (e.g. missing interpreter
                # or script); report it and stop rather than crashing the UI.
                self.add_agent_message(f"Could not launch {tool_name}:\n\n{e}")
                return
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            print(stdout)
            print(stderr)

            if result.returncode == 0:
                self.add_agent_message(f"{tool_name} completed successfully.")
                return

            if attempt >= MAX_REPAIR_ATTEMPTS:
                self.add_agent_message(
                    f"{tool_name} failed after {attempt} repair attempt(s) and I couldn't fix it automatically.\n\nLast error:\n{stderr[-1000:]}"
                )
                return

            self.add_agent_message(
                f"{tool_name} failed (exit code {result.returncode}). Trying to fix the parameters from the error..."
            )

            repaired = self.logic._proposeRepair(tool_name, current_params, current_cli_args, stderr)
            if repaired is None:
                self.add_agent_message(f"I couldn't propose a fix for this error.\n\nLast error:\n{stderr[-1000:]}")
                return

            new_params, new_cli_args = repaired
            diff_lines = "\n".join(
                f"-{k}: {current_params.get(k)!r} -> {v!r}"
                for k, v in new_params.items()
                if current_params.get(k) != v
            )
            if not diff_lines:
                self.add_agent_message(f"I couldn't find a different set of parameters to try.\n\nLast error:\n{stderr[-1000:]}")
                return

            reply = qt.QMessageBox.question(
                None,
                f"Retry {tool_name}?",
                f"{tool_name} failed. Here is the fix I propose:\n\n{diff_lines}\n\nRetry with these corrected parameters?",
                qt.QMessageBox.Yes | qt.QMessageBox.No
            )
            if reply != qt.QMessageBox.Yes:
                self.add_agent_message("Okay, not retrying.")
                return

            current_params, current_cli_args = new_params, new_cli_args
            attempt += 1

    def OnSaveButton(self):
        from pathlib import Path

        text = self.ui.textEdit.toPlainText()
        filename = f"{Path.home()}/Chat_LLM_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(text)
            print(f"The chat has been saved to {filename}")
        except Exception as e:
            qt.QMessageBox.warning(
                None,
                "Could not save the chat",
                f"Saving the chat to '{filename}' failed:\n\n{e}"
            )

        self._checkCanApply()

    def OnClearButton(self):
        self.ui.textEdit.clear()
        self.conversationHistory = []
        self._checkCanApply()

    def onReturnPressed(self):
        if self.ui.applyButton.isEnabled():
            self.onApplyButton()
            
    def OnRetrieveButton(self):
        import qt
        filepath = qt.QFileDialog.getOpenFileName(
            None,
            "Choose a text file",
            "",
            "Text files (*.txt)"
        )

        # The dialog returns an empty string when the user cancels.
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            qt.QMessageBox.warning(
                None,
                "Could not open the file",
                f"Reading '{filepath}' failed:\n\n{e}"
            )
            return

        content = content.replace("🤖:", "👨:")
        output_list = content.split("👨:")
        output_list.pop(0)

        if len(output_list) < 2:
            qt.QMessageBox.warning(None, "Text file incompatible", "Please choose a text (.txt) file from a previous discussion with the agent")
        else:
            for i in range(len(output_list)):
                if i % 2 == 0:
                    self.add_user_message("👨:" + output_list[i])
                else:
                    self.add_agent_message(output_list[i])

            self._checkCanApply()

    def CheckDependencies(self):

        # Python packages required by Agent_CLI. Order matters: the pinned
        # versions must be installed before the packages that would otherwise
        # pull in a broken/incompatible version as a transitive dependency.
        #   - transformers>=4.53.0 has a known regression that breaks the
        #     cross-encoder import with "NameError: name 'nn' is not defined"
        #     (huggingface/transformers#43784), so it is pinned first.
        #   - numpy>=2 breaks the ABI most pip-installed torch wheels were built
        #     against ("RuntimeError: Numpy is not available"), so it is pinned
        #     before torch is pulled in as a sentence-transformers dependency.
        # Agent_CLI.py runs as a separate "python-real" CLI process, but it
        # shares Slicer's site-packages, so pip-installing here makes these
        # importable there too.
        packages = [
            "ollama",
            "pyyaml",
            "transformers<4.53.0",
            "numpy<2",
            "sentence-transformers",
        ]

        failed_packages = []
        for package in packages:
            try:
                slicer.util.pip_install(package)
            except Exception as e:
                # Keep going so one failing package doesn't block the others;
                # report everything that failed at the end.
                print(f"Failed to install '{package}': {e}")
                failed_packages.append(package)

        if failed_packages:
            qt.QMessageBox.warning(
                None,
                "Dependency installation issues",
                "Some Python packages could not be installed:\n\n- "
                + "\n- ".join(failed_packages)
                + "\n\nCheck your network connection and try again."
            )
            return

        # pip_install("ollama") only installs the Python client library - the
        # actual Ollama application/server (the "ollama" binary used below)
        # has to be installed separately and isn't something pip can provide.
        # Prefer a system install, otherwise fall back to one we bundled before.
        ollama_path = shutil.which("ollama") or _bundled_ollama_binary()

        # Nothing installed at all: offer to download the official GPU build for
        # the user (most of them are not comfortable doing this by hand).
        if ollama_path is None:
            choice = qt.QMessageBox.question(
                None,
                "Ollama is not installed",
                "The agent needs Ollama to run the AI model, and it is not installed on "
                "this computer.\n\n"
                "Download and install it automatically now? This fetches the official "
                "build (a few hundred MB, GPU-enabled) - no admin rights needed.",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
                qt.QMessageBox.Yes,
            )
            if choice != qt.QMessageBox.Yes:
                qt.QMessageBox.information(
                    None,
                    "Manual installation",
                    "No problem. You can install Ollama yourself from https://ollama.com "
                    "(on Linux, use the official installer, NOT the snap/apt package, so it "
                    "can use your GPU), then click Check again."
                )
                return
            try:
                ollama_path = install_official_ollama()
            except Exception as e:
                print(f"Automatic Ollama install failed: {e}")
                qt.QMessageBox.warning(
                    None,
                    "Automatic installation failed",
                    "Ollama could not be installed automatically:\n\n"
                    f"{e}\n\n"
                    "Please install it manually from https://ollama.com, then click Check again."
                )
                return

        # On Linux the snap build of Ollama runs inside a confined sandbox that
        # cannot reach the NVIDIA GPU, so it silently falls back to CPU - a 8B
        # model then takes ~80-90s per answer instead of a few seconds. We can't
        # safely auto-fix this (the snap server already holds port 11434 and
        # removing it needs sudo), so warn and point to the official build.
        elif "/snap/" in os.path.realpath(ollama_path):
            proceed = qt.QMessageBox.warning(
                None,
                "Ollama installed via snap (CPU-only)",
                "Ollama was installed through snap. The snap build is sandboxed and "
                "cannot use your GPU, so the agent will run on CPU and be very slow "
                "(around 80-90 seconds per answer for the 8B model).\n\n"
                "To get GPU speed, install the official build instead:\n\n"
                "    sudo snap remove ollama\n"
                "    curl -fsSL https://ollama.com/install.sh | sh\n\n"
                "Then run 'ollama serve' and click Check again.\n\n"
                "Continue anyway with the current (CPU-only) install?",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
                qt.QMessageBox.No,
            )
            if proceed != qt.QMessageBox.Yes:
                return

        # Make sure a server is actually reachable. For a system install this is
        # usually already running; for one we just downloaded, start it.
        ollama_env = _ollama_env(ollama_path)
        if not ensure_ollama_server(ollama_path):
            qt.QMessageBox.warning(
                None,
                "Ollama server not responding",
                "Ollama is installed but its server did not start. Try running "
                "'ollama serve' in a terminal, then click Check again."
            )
            return

        list_model = [MODEL_NAME]
        installed = True

        for model in list_model:
            try:
                result = subprocess.run(
                    [ollama_path, 'pull', model],
                    capture_output=True,
                    text=True,
                    env=ollama_env,
                )
                if result.returncode == 0:
                    print(f"Model {model} has successfully been installed")
                else:
                    print(f"Error pulling model {model}: {result.stderr.strip()}")
                    installed = False
            except Exception as e:
                print(f"Error getting models: {e}")
                installed = False

        if installed:
            qt.QMessageBox.information(None, "Dependencies checked and installed", "All the dependencies have been checked and installed, now you can start to talk with your personal agent.")

        else:
            qt.QMessageBox.warning(
                None,
                "Dependencies installation issues",
                "Ollama is installed but pulling the models failed. Make sure Ollama is running "
                "('ollama serve') and that you have network access, then try again."
            )


    def clearDropzone(self):
        self.droppedInputPaths = []
        self.dropZone.setSummary(self.droppedInputPaths)
        self._checkCanApply()

class AgentLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return AgentParameterNode(super().getParameterNode())

    def process(self,
                folders: str,
                prompt: str) -> str:
        """
        Run the Agent_CLI routing/execution algorithm.
        Can be used without the GUI widget.
        :param folders: input files/folders passed to the tools
        :param prompt: the natural-language request for the agent
        :return: the CLI output text
        """

        start_time = time.time()
        logging.info("Processing started")

        # Delegate the actual work to the Agent_CLI command-line module.
        cli_params = {
            "folders": folders,
            "prompt": prompt
        }
        CLI = slicer.modules.agent_cli
        self.cliNode = slicer.cli.run(CLI, None, cli_params, wait=False)
        output_text = self.cliNode.GetOutputText()
        stop_time = time.time()
        logging.info(f"Processing completed in {stop_time-start_time:.2f} seconds")
        return output_text
    def to_html(self, text):
        """Escape text for HTML while preserving basic formatting (newlines, spaces)."""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&#39;")

        text = text.replace("\n", "<br>")

        text = text.replace("  ", "&nbsp;&nbsp;")
        return text

    def normalize_folders(self,folders):
        if folders is None:
            return []
        if isinstance(folders, (list, tuple)):
            return list(folders)
        try:
            # ObservedList, Qt list, etc.
            return list(folders)
        except TypeError:
            return [str(folders)]

    def _suggestFixFor(self, error_text):
        """Best-effort, keyword-based remediation hint for an Agent_CLI failure."""

        text = (error_text or "")
        lower = text.lower()

        model_match = re.search(r"model ['\"]([^'\"]+)['\"] not found", lower)
        if model_match:
            model_name = model_match.group(1)
            return (
                f"The Ollama model '{model_name}' isn't pulled on this machine yet. Run this in a "
                f"terminal: ollama pull {model_name.split(':')[0]}\nThen try again."
            )
        if "modulenotfounderror" in lower or "no module named" in lower:
            return (
                "A required Python package is missing in Slicer's environment. "
                "Click the 'Check' button to (re)install the dependencies, then try again."
            )
        if "nameerror" in lower and "'nn'" in lower:
            return (
                "Known regression in transformers>=4.53.0 (a missing 'import torch.nn as nn' in "
                "transformers/integrations/accelerate.py - see huggingface/transformers#43784), not "
                "a bug in the agent. Click 'Check' to reinstall with the pinned, working version, "
                "or run manually in Slicer's Python console: "
                "slicer.util.pip_install('transformers<4.53.0')"
            )
        if "numpy is not available" in lower:
            return (
                "Likely a numpy 2.x / torch ABI mismatch (numpy>=2 breaks most pip-installed torch "
                "wheels' .numpy() calls). Click 'Check' to reinstall with numpy pinned below 2, or "
                "run manually in Slicer's Python console: slicer.util.pip_install('numpy<2')"
            )
        if "ollama" in lower or "connection" in lower:
            return (
                "This usually means Ollama isn't installed/running - install it from "
                "https://ollama.com, make sure 'ollama serve' is running, then try again."
            )
        return "Click the 'Check' button to verify dependencies, then try again."

    def _proposeRepair(self, tool_name, params, cli_args, stderr):
        """
        Ask the LLM to correct `params` given the failing `cli_args`/`stderr`.
        Returns (new_params, new_cli_args), or None if no usable correction
        could be produced. Reuses Agent_CLI_utils (no duplicated extraction
        logic) by adding Agent_CLI's own directory to sys.path.
        """

        agent_cli_dir = os.path.dirname(slicer.modules.agent_cli.path)
        if agent_cli_dir not in sys.path:
            sys.path.insert(0, agent_cli_dir)

        from Agent_CLI_utils.utils import (
            load_manifest, build_repair_prompt, build_cli_args, complete_with_defaults,
            get_tool_def, chat_with_auto_pull, get_router_model
        )
        from Agent_CLI_utils.parameter_validator import ParameterValidator

        manifest_path = os.path.join(agent_cli_dir, "manifest.yaml")
        if not os.path.isfile(manifest_path):
            manifest_path = os.path.join(agent_cli_dir, "Resources", "manifest.yaml")

        try:
            manifest = load_manifest(manifest_path)
            tool_spec = get_tool_def(manifest, tool_name)
            if not tool_spec:
                return None

            prompt = build_repair_prompt(tool_name, tool_spec, params, cli_args, stderr)
            model = get_router_model()

            response = chat_with_auto_pull(
                model,
                messages=[
                    {"role": "system", "content": "You are a parameter-repair expert. Output ONLY valid JSON on one line."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            data = json.loads(response["message"]["content"])
            corrections = data.get("extracted", {})
            if not corrections:
                return None

            merged = dict(params)
            merged.update(corrections)

            validator = ParameterValidator(manifest_path)
            validation_result = validator.validate(tool_name, merged)
            new_params = complete_with_defaults(manifest, tool_name, validation_result["params"])
            new_cli_args = build_cli_args(tool_name, new_params, manifest, agent_cli_dir)
            return new_params, new_cli_args
        except Exception as e:
            print(f"Repair attempt failed: {e}")
            return None



#
# AgentTest
#


class AgentTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_Agent1()

    def test_Agent1(self):
        """
        Smoke test: the logic instantiates and its parameter node exposes the
        expected fields (prompt, folders, modeagent). The full agent pipeline
        depends on Ollama and the qwen3:8b model, which are out of scope for an
        automated unit test, so this only checks the module wiring.
        """

        self.delayDisplay("Starting the test")

        logic = AgentLogic()
        parameter_node = logic.getParameterNode()

        # The parameter node should expose the module's parameters.
        self.assertTrue(hasattr(parameter_node, "prompt"))
        self.assertTrue(hasattr(parameter_node, "folders"))
        self.assertTrue(hasattr(parameter_node, "modeagent"))

        self.delayDisplay("Test passed")
