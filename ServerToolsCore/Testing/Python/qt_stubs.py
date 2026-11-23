"""Minimal stand-ins for `qt`, `ctk` and `slicer`, so the GUI-facing parts of
ServerToolsCoreLib can be unit-tested outside Slicer.

Only what `formgen` (and the `design` module it pulls in) actually touches is
implemented - enough to assert *which* widgets a schema produces, in which
order, with which initial state, and what they read back as. This obviously
tests the schema-to-widget logic, not Qt itself; the Qt calls it makes are the
same ones the previously shipped widgets already used.

Call `install()` before importing ServerToolsCoreLib.formgen.
"""

import sys
import types


class Signal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class QObject:
    def __init__(self, *_args, **_kwargs):
        self._properties = {}
        self._tooltip = ""
        self._stylesheet = ""
        self._visible = True

    def setVisible(self, visible):
        self._visible = bool(visible)

    def isVisible(self):
        return self._visible

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        """Enabled until something disables it, as Qt is. Read back so a panel
        that greys a control can be tested on whether it greyed the right one."""
        return getattr(self, "_enabled", True)

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)

    def setToolTip(self, text):
        self._tooltip = text

    def toolTip(self):
        return self._tooltip

    def setParent(self, parent):
        """`rebuild` detaches a widget this way; the stub only has to accept it."""
        self.parent = parent

    def setObjectName(self, name):
        """An id selector is how a container styles ITSELF without the rule
        reaching every child, which is what `design.table_frame` relies on."""
        self.objectName = name

    def setStyleSheet(self, sheet):
        self._stylesheet = sheet
        # Readable under the name Qt publishes it under, so a test asserting
        # what a widget was painted with reads the same on every stub class.
        self.styleSheet = sheet

    def setCursor(self, _cursor):
        pass

    def setMinimumHeight(self, height):
        self._minimum_height = height

    def setMaximumHeight(self, height):
        """Recorded because it is half of a real property: a tab box with only a
        MINIMUM is stretched by the panel's spare vertical space, which is how
        ten cranial landmarks ended up in a 380 px box."""
        self._maximum_height = height

    def minimumHeight(self):
        return getattr(self, "_minimum_height", 0)

    def maximumHeight(self):
        return getattr(self, "_maximum_height", None)

    def setFixedSize(self, width, height):
        self._fixed_size = (width, height)

    def setFocusPolicy(self, _policy):
        pass

    def update(self):
        pass


class QWidget(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self)
        self.parent = parent
        self.layout = None
        self.deleted = False

    def setParent(self, parent):
        """Reparenting is what removes a widget from its layout in Qt, which is
        how base_widget swaps a container wholesale (_buildForm,
        _rebuildRunCancelButtons). Recorded so a test can check the old one
        really left."""
        self.parent = parent

    def deleteLater(self):
        self.deleted = True


class QLayout(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self)
        self.widgets = []
        self.stretches = []
        self.margins = (0, 0, 0, 0)
        if parent is not None:
            parent.layout = self

    def setContentsMargins(self, *margins):
        """Recorded rather than dropped. Where a block's air sits is the
        difference between a group a reader can see the end of and one whose
        last option touches the next argument's label."""
        self.margins = tuple(margins) if len(margins) == 4 else (0, 0, 0, 0)

    def setSpacing(self, _spacing):
        pass

    def addWidget(self, widget, stretch=0):
        self.widgets.append(widget)

    def addLayout(self, layout, stretch=0):
        """A nested layout is an item of its own, and its widgets are still in
        the tree. Recorded in `widgets` so a panel's contents can be walked
        whether a row was built as one layout or two."""
        self.widgets.append(layout)

    def addStretch(self, stretch=1):
        """Recorded, not discarded. WHERE the stretch sits is the difference
        between links that start at the left edge of the options they act on and
        links that float at the far edge of the panel."""
        self.stretches.append(stretch)

    def count(self):
        return len(self.widgets)

    def takeAt(self, index):
        """Qt hands back an item wrapping the widget; `rebuild` empties a layout
        through it, so the stub answers the same shape."""
        widget = self.widgets.pop(index)

        class _Item:
            @staticmethod
            def widget():
                return widget

        return _Item()


class QVBoxLayout(QLayout):
    pass


class QHBoxLayout(QLayout):
    pass


class QGridLayout(QLayout):
    """Records the (row, column) each widget was placed at - which is the whole
    point of the "grid" multichoice layout: a chart is only a chart if the
    positions are right (see formgen._build_grid_boxes)."""

    def __init__(self, parent=None):
        QLayout.__init__(self, parent)
        self.cells = {}  # {(row, column): widget}
        # Which trailing row/column was given the layout's spare space. Recorded
        # rather than ignored: without a stretch, a grid inside a resizable
        # scroll area spreads its rows across the whole height -- ALI's check
        # boxes sat 94 px apart -- and that is a property worth a test.
        self.rowStretch = {}
        self.columnStretch = {}
        self.verticalSpacing = None
        self.horizontalSpacing = None

    def addWidget(self, widget, row=0, column=0, *_args):
        self.widgets.append(widget)
        self.cells[(row, column)] = widget

    def setVerticalSpacing(self, spacing):
        self.verticalSpacing = spacing

    def setHorizontalSpacing(self, spacing):
        """Recorded separately: chips carry their own padding, so the two gaps
        are deliberately different -- touching columns read as one long word."""
        self.horizontalSpacing = spacing

    def setRowStretch(self, row, stretch):
        self.rowStretch[row] = stretch

    def setColumnStretch(self, column, stretch):
        self.columnStretch[column] = stretch

    def rowCount(self):
        return max((row for row, _column in self.cells), default=-1) + 1


class QScrollArea(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.widget = None
        self.widgetResizable = False
        self.verticalScrollBarPolicy = None
        self.horizontalScrollBarPolicy = None

    def setWidget(self, widget):
        self.widget = widget

    def setWidgetResizable(self, resizable):
        self.widgetResizable = resizable

    def setVerticalScrollBarPolicy(self, policy):
        self.verticalScrollBarPolicy = policy

    def setHorizontalScrollBarPolicy(self, policy):
        self.horizontalScrollBarPolicy = policy


class QTabWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.tabs = []  # [(title, widget)]
        self.currentIndex = 0
        self.currentChanged = Signal()

    def addTab(self, widget, title):
        self.tabs.append((title, widget))

    def setCurrentIndex(self, index):
        """Emits, because the height of the box follows the tab on screen."""
        self.currentIndex = index
        self.currentChanged.emit(index)


class QStandardPaths:
    """Where this OS puts things. The stub answers a fixed location, which is
    the point: a test asserts what the code does WITH the answer, never what
    the answer is on the machine running it."""

    DocumentsLocation = 1
    documents = "/home/tester/Documents"

    @staticmethod
    def writableLocation(_location):
        return QStandardPaths.documents


class QCursor:
    def __init__(self, shape=0):
        self.shape = shape


class Qt:
    # A read-only box takes no focus: it reports, it does not accept.
    NoFocus = 0
    """The Qt namespace enum values design/formgen/joystick reach for."""

    ScrollBarAlwaysOff = 1
    ScrollBarAsNeeded = 0
    PointingHandCursor = 13
    StrongFocus = 11
    AlignCenter = 0x84
    ControlModifier = 0x04000000
    ShiftModifier = 0x02000000
    Key_Left = 0x01000012
    Key_Up = 0x01000013
    Key_Right = 0x01000014
    Key_Down = 0x01000015
    PlainText = 0
    RichText = 1
    AutoText = 2


class QFileDialog:
    """Test seam: `next_file`/`next_directory` are what the static helpers
    return, so a test can drive the browse buttons without a real (modal)
    dialog, and inspect the arguments they were called with - the file
    dialog's filter string among them."""

    next_file = ""
    next_directory = ""
    last_open_file_args = None
    last_existing_directory_args = None

    @staticmethod
    def getOpenFileName(*args, **_kwargs):
        QFileDialog.last_open_file_args = args
        return QFileDialog.next_file

    @staticmethod
    def getExistingDirectory(*args, **_kwargs):
        QFileDialog.last_existing_directory_args = args
        return QFileDialog.next_directory


class QFormLayout(QLayout):
    def __init__(self, parent=None):
        QLayout.__init__(self, parent)
        self.rows = []  # [(label, widget)]

    def addRow(self, label, widget=None):
        """`addRow(widget)` alone is a row spanning both columns, recorded
        with no label."""
        if widget is None:
            label, widget = None, label
        self.rows.append((label, widget))
        self.widgets.append(widget)

    def insertRow(self, row, label, widget):
        self.rows.insert(row, (label, widget))
        self.widgets.insert(row, widget)


class QFont:
    DemiBold = 63

    def __init__(self, *_args):
        pass

    def setWeight(self, _weight):
        pass


class QFontMetrics:
    """Seven pixels a character: enough to tell a short label from one past
    the cap, which is all the tests ask of it."""

    def __init__(self, _font):
        pass

    @staticmethod
    def horizontalAdvance(text):
        return 7 * len(text)


class QLabel(QObject):
    def __init__(self, text=""):
        QObject.__init__(self)
        self.text = text
        self.textFormat = None

    def setTextFormat(self, text_format):
        self.textFormat = text_format

    def setText(self, text):
        self.text = text

    def setWordWrap(self, wrap):
        """Recorded, not dropped. Whether a caption WRAPS is the difference
        between showing a whole path and showing a fragment of one, which is
        the reason that line replaced the path field."""
        self.wordWrap = bool(wrap)

    def setMaximumWidth(self, width):
        self.maximumWidth = width

    def setMinimumWidth(self, width):
        self.minimumWidth = width

    @property
    def font(self):
        return QFont()


class QPushButton(QObject):
    def __init__(self, text=""):
        QObject.__init__(self)
        self.text = text
        self.clicked = Signal()
        self.toggled = Signal()
        self._checkable = False
        self._checked = False

    def connect(self, signature, slot):
        """PythonQt's own spelling: `button.connect("clicked()", callable)`.

        Only the signals the extension actually wires are honoured; anything
        else is recorded and never fired, which is what a stub should do rather
        than pretend.
        """
        if signature.startswith("clicked"):
            self.clicked.connect(slot)
        elif signature.startswith("toggled"):
            self.toggled.connect(slot)

    def click(self):
        """What a user does. Emits, so whatever was connected actually runs."""
        self.clicked.emit()

    def isCheckable(self):
        """Whether this button IS an option rather than an action -- the whole
        distinction between a chip and a plain button."""
        return self._checkable

    def setChecked(self, checked):
        """A checkable button IS the option in a dense multichoice, so it has to
        read back exactly as a check box does (see design.option_chip)."""
        checked = bool(checked)
        if checked != self._checked:
            self._checked = checked
            self.toggled.emit(checked)

    def isChecked(self):
        return self._checked

    def setText(self, text):
        """PythonQt exposes both the property and the setter; so does this."""
        self.text = text

    def setCheckable(self, checkable):
        self._checkable = bool(checkable)


class QLineEdit(QObject):
    def __init__(self, text=""):
        QObject.__init__(self)
        self._text = text
        self.placeholderText = ""
        self.textChanged = Signal()

    def setPlaceholderText(self, text):
        self.placeholderText = text

    def setReadOnly(self, read_only):
        """A value field is a QLineEdit you cannot type into: it shows what the
        row holds and the button beside it is what changes that. Recorded, so a
        test can tell it from a field a user is meant to fill."""
        self.readOnly = bool(read_only)

    def isReadOnly(self):
        return getattr(self, "readOnly", False)

    def setCursorPosition(self, position):
        """Where the visible text starts when it is longer than the box. A path
        shown from its END is a path whose file name is off screen."""
        self.cursorPosition = position

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        self.textChanged.emit(value)

    def setText(self, value):
        self.text = value


class QCheckBox(QObject):
    def __init__(self, text=""):
        QObject.__init__(self)
        self.text = text
        self._checked = False
        self.toggled = Signal()

    def setChecked(self, checked):
        self._checked = bool(checked)
        self.toggled.emit(self._checked)

    def isChecked(self):
        return self._checked


class _ListView:
    """Just enough of the popup for `_widenPopup`.

    `sizeHintForColumn` is what the real view answers: the width its own items
    need. One pixel per character here -- real metrics are proportional, but the
    property under test is "the widest entry decides", not the exact pixels.
    """

    def __init__(self):
        self.minimumWidth = 0
        self.items = []

    def sizeHintForColumn(self, _column):
        return max((len(text) for text in self.items), default=0)

    def setMinimumWidth(self, width):
        self.minimumWidth = width


class _Metrics:
    """What `fontMetrics()` RETURNS. One pixel per character."""

    @staticmethod
    def horizontalAdvance(text):
        return len(text)


class _FontMetricsSlot:
    """A SLOT, exactly as PythonQt exposes it: `combo.fontMetrics` is not a
    property, it is a callable, and reading through it without calling it gives
    an object with no metrics on it at all.

    This stub used to BE the metrics, so `combo.fontMetrics.horizontalAdvance`
    worked here and raised `AttributeError` in Slicer -- the popup-widening code
    was dead for as long as it existed and 728 passing tests said it was fine. A
    stub that lies in the same direction as the binding is worse than no stub.
    """

    def __call__(self):
        return _Metrics()


class QComboBox(QObject):
    AdjustToMinimumContentsLengthWithIcon = 2

    def __init__(self):
        QObject.__init__(self)
        self._items = []
        self._data = []
        self._index = -1
        self.sizeAdjustPolicy = 0
        self.minimumContentsLength = 0
        self._view = _ListView()
        self.fontMetrics = _FontMetricsSlot()
        self.currentTextChanged = Signal()
        self.currentIndexChanged = Signal()

    def view(self):
        return self._view

    def addItems(self, items):
        self._items.extend(items)
        self._data.extend([None] * len(items))
        # The real view measures the model it is showing; so does this one.
        self._view.items = list(self._items)
        if self._index < 0 and self._items:
            self.setCurrentIndex(0)

    def addItem(self, item, userData=None):
        """`userData` is what a combo carries besides its label.

        Real Qt has always had it; this stub did not, so a panel storing the
        value behind each entry — an address, an id, anything not fit to show
        — could not be tested at all. Optional, so every existing caller is
        unaffected.
        """
        self.addItems([item])
        self._data[-1] = userData

    def itemData(self, index):
        if 0 <= index < len(self._data):
            return self._data[index]
        return None

    def clear(self):
        self._items = []
        self._data = []
        self._index = -1
        self._view.items = []

    @property
    def count(self):
        # A property, not a method: PythonQt exposes a Qt property whose getter
        # shares its name as an attribute, and it shadows the method - real
        # Slicer raises "'int' object is not callable" on `combo.count()`.
        return len(self._items)

    def itemText(self, index):
        return self._items[index]

    @property
    def currentIndex(self):
        return self._index

    def setCurrentIndex(self, index):
        """Emits only on a REAL change, as Qt does.

        It emitted unconditionally, which is a stub lying in the direction of
        "everything notifies" -- and it hid a live defect: picking a node in
        the scene list re-evaluated Apply only because resetting the OTHER
        list to an index it was already on fired a signal here. Against real
        Qt it fired nothing, and the panel kept Apply greyed over a row the
        user had just filled.
        """
        if index == self._index:
            return
        self._index = index
        # Both, as Qt does. A panel that reacts to the SELECTION rather than to
        # the label has to connect currentIndexChanged — two entries can show
        # the same text, and currentTextChanged does not fire between them.
        self.currentIndexChanged.emit(index)
        self.currentTextChanged.emit(self.currentText)

    @property
    def currentText(self):
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return ""

    def setCurrentText(self, text):
        self.setCurrentIndex(self._items.index(text))


class QSpinBox(QObject):
    def __init__(self):
        QObject.__init__(self)
        self.value = 0
        self.minimum = 0
        self.maximum = 99
        self.singleStep = 1
        self.valueChanged = Signal()

    def setRange(self, minimum, maximum):
        self.minimum, self.maximum = minimum, maximum

    def setSingleStep(self, step):
        self.singleStep = step

    def setValue(self, value):
        # Clamped like the real widget: JoystickInput's spring-back mode
        # relies on the boxes clamping the accumulated displacement.
        self.value = min(max(value, self.minimum), self.maximum)
        self.valueChanged.emit(self.value)


class QDoubleSpinBox(QSpinBox):
    def __init__(self):
        QSpinBox.__init__(self)
        self.value = 0.0
        self.decimals = 2

    def setDecimals(self, decimals):
        self.decimals = decimals

    def setReadOnly(self, readOnly):
        self.readOnly = readOnly

    def setButtonSymbols(self, symbols):
        self.buttonSymbols = symbols


class QAbstractSpinBox:
    """Only the enum formgen names: a read-only box hides its arrows."""

    NoButtons = 2


class QTimer(QObject):
    """Records its state instead of running an event loop.

    `BackgroundJob` drives two of these -- one draining its queue, one handing
    the GIL to the worker thread -- and what a test needs to see is which of
    them is running when, not Qt's scheduling.
    """

    def __init__(self, *_args, **_kwargs):
        QObject.__init__(self)
        self.interval = None
        self.running = False
        self.starts = 0
        self.stops = 0
        self.timeout = Signal()

    def setInterval(self, milliseconds):
        self.interval = milliseconds

    def start(self, *_args):
        self.running = True
        self.starts += 1

    def stop(self):
        self.running = False
        self.stops += 1

    def fire(self):
        """Call the connected slots, the way a real timeout would."""
        self.timeout.emit()


class QProgressBar(QObject):
    """Records its range and value. Determinate only: base_widget shows it
    exclusively for a fraction a tool really reported, so there is no
    indeterminate (0, 0) mode to model."""

    def __init__(self):
        QObject.__init__(self)
        self.minimum = 0
        self.maximum = 100
        self.value = 0
        self.textVisible = True

    def setRange(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def setValue(self, value):
        self.value = value

    def setTextVisible(self, visible):
        self.textVisible = bool(visible)

    def setMaximumHeight(self, height):
        """The slim bar one batch of a cohort gets (design.batch_bar)."""
        self.maximumHeight = height


class QFrame(QWidget):
    """The box a cohort's progress is drawn in (design.cohort_frame).

    A QWidget that takes a stylesheet and nothing else: everything else about
    it is the layout put inside it, which is modelled already.
    """

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.styleSheet = ""


class QPalette:
    Window = 0


class ctkPathLineEdit(QObject):
    """Counts assignments to `filters`/`nameFilters`.

    Reconfiguring a real, live ctkPathLineEdit corrupts it and crashes Slicer
    (see formgen.path_widget); the counters let a test assert each picker is
    configured exactly once, at construction, and never again.
    """

    Files = 1
    Dirs = 2

    def __init__(self):
        QObject.__init__(self)
        self._path = ""
        self._filters = ctkPathLineEdit.Files
        self._name_filters = []
        self.filterAssignments = 0
        self.nameFilterAssignments = 0
        self.currentPathChanged = Signal()

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, value):
        self._filters = value
        self.filterAssignments += 1

    @property
    def nameFilters(self):
        return self._name_filters

    @nameFilters.setter
    def nameFilters(self, value):
        self._name_filters = value
        self.nameFilterAssignments += 1

    @property
    def currentPath(self):
        return self._path

    @currentPath.setter
    def currentPath(self, value):
        self._path = value
        self.currentPathChanged.emit(value)


class ctkCollapsibleButton(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.text = ""
        # Open unless a panel folds it. Which sections open folded is a
        # property a test asserts on, so it is recorded rather than ignored.
        self._collapsed = False
        # How many rows the box held at the moment it folded. Real CTK hides the
        # children a box HAS when it collapses, so a box folded before its rows
        # exist folds nothing -- which is what made a "Reference volume" picker
        # appear crushed into the "Advanced" header. Recorded so a test can tell
        # a box that folded once it was built from one that folded empty.
        self.rowsWhenCollapsed = None
        # How many times CTK actually ran its hide-the-children pass. Counted
        # because "is it collapsed" cannot tell a fold that DID something from
        # one that was skipped -- see the setter.
        self.foldsApplied = 0

    @property
    def collapsed(self):
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value):
        # `ctkCollapsibleButton::setCollapsed` returns immediately when handed
        # the value it already holds, so folding a box that already believes it
        # is folded runs NOTHING: no child pass, no hiding. A panel that folds
        # during its build and folds again once on screen therefore has to step
        # through the other state to make the second one count. Modelled here
        # because the real behaviour is invisible from Python and cost two
        # wrong fixes.
        if value == self._collapsed:
            return
        self._collapsed = value
        if value:
            self.foldsApplied += 1
            self.rowsWhenCollapsed = len(getattr(self.layout, "widgets", ()) or ())


class ctkSliderWidget(QObject):
    """PythonQt exposes minimum/maximum/decimals/singleStep/value as writable
    properties. `value` clamps into [minimum, maximum] like the real widget."""

    def __init__(self):
        QObject.__init__(self)
        self.minimum = 0.0
        self.maximum = 99.0
        self.decimals = 2
        self.singleStep = 1.0
        self.pageStep = 1.0
        self._value = 0.0
        self.valueChanged = Signal()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = min(max(float(new_value), self.minimum), self.maximum)
        self.valueChanged.emit(self._value)

    def setValue(self, new_value):
        self.value = new_value


def install():
    """Register the fake `qt`, `ctk` and `slicer` modules in sys.modules.

    `slicer` is deliberately empty: `design.is_dark_mode()` reaches for
    `slicer.app.palette()` inside a try/except, so the missing attribute makes
    it fall back to the light palette - which is all these tests need.
    """
    qt = types.ModuleType("qt")
    for name, value in globals().items():
        if name.startswith("Q") or name == "Signal":
            setattr(qt, name, value)

    ctk = types.ModuleType("ctk")
    ctk.ctkPathLineEdit = ctkPathLineEdit
    ctk.ctkCollapsibleButton = ctkCollapsibleButton
    ctk.ctkSliderWidget = ctkSliderWidget

    sys.modules.setdefault("qt", qt)
    sys.modules.setdefault("ctk", ctk)
    sys.modules.setdefault("slicer", types.ModuleType("slicer"))
    return qt, ctk
