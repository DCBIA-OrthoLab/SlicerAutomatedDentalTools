"""The dark palette the modules of this extension share.

Eleven modules carried their own copy: nine of `applyDarkModeStyles` and eleven
of the recursive pass below, 1926 lines in all, and the copies had started to
drift -- a docstring here, an inert `!important` there, and one missing the
`hasattr` guard that keeps the recursion off a parent with no children.

Two modules are deliberately NOT on this. ASO and AREG style widgets the others
do not have (ctkCollapsibleButton, qMRMLNodeComboBox) and AREG uses a darker
palette throughout; folding them in is a decision about how they should look,
not a refactor. They keep their own stylesheet and share the recursive pass.

Qt's stylesheet subset has no `!important`: the token one copy carried was
inert, and dropping it changes nothing.
"""
import qt


def is_dark_mode():
    """Whether Slicer is running under a dark palette."""
    palette = qt.QApplication.instance().palette()
    return palette.color(qt.QPalette.Window).lightness() < 128


def apply_dark_mode(ui_widget):
    """Give `uiWidget` the dark palette, when Slicer is running dark.

    Does nothing under a light palette, so a module can call it
    unconditionally from its setup().
    """
    app = qt.QApplication.instance()
    palette = app.palette()
    bg_color = palette.color(qt.QPalette.Window)
    if bg_color.lightness() < 128:
        # Complete dark mode stylesheet
        dark_stylesheet = """
QLineEdit, QTextEdit {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 4px;
  padding: 6px;
  color: #ffffff;
  selection-background-color: #5dade2;
}
QLineEdit:focus, QTextEdit:focus {
  border: 2px solid #5dade2;
}
QComboBox {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 4px;
  padding: 4px 6px;
  color: #ffffff;
}
QComboBox:focus {
  border: 2px solid #5dade2;
}
QComboBox::drop-down {
  width: 20px;
  border: none;
}
QComboBox QAbstractItemView {
  background-color: #3c3c3c;
  color: #ffffff;
  selection-background-color: #5dade2;
}
QLabel {
  color: #ffffff;
  font-weight: 500;
  background-color: transparent;
}
QPushButton {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 #3498db);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 600;
  font-size: 10pt;
  padding: 8px;
  margin-top: 4px;
}
QPushButton:hover:!pressed {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7bbcef, stop:1 #5dade2);
}
QPushButton:pressed {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1e638d);
}
QPushButton:disabled {
  background-color: #555555;
  color: #888888;
}
QCheckBox {
  color: #ffffff;
  font-weight: 500;
  spacing: 6px;
  background-color: transparent;
}
QCheckBox::indicator {
  width: 18px;
  height: 18px;
  border: 1px solid #555555;
  border-radius: 3px;
  background-color: #3c3c3c;
}
QCheckBox::indicator:hover {
  border: 1px solid #5dade2;
}
QCheckBox::indicator:checked {
  width: 18px;
  height: 18px;
  border: 1px solid #5dade2;
  border-radius: 3px;
  background-color: #5dade2;
  image: url(:/Icons/SmallCheckMark.png);
}
QCheckBox::indicator:checked:hover {
  border: 1px solid #7bbcef;
  background-color: #7bbcef;
}
QProgressBar {
  border: 1px solid #555555;
  border-radius: 4px;
  background-color: #3c3c3c;
  padding: 2px;
  color: #ffffff;
}
QProgressBar::chunk {
  background-color: #5dade2;
  border-radius: 3px;
}
QSpinBox, QDoubleSpinBox {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 4px;
  padding: 4px 6px;
  color: #ffffff;
}
QSpinBox:focus, QDoubleSpinBox:focus {
  border: 2px solid #5dade2;
}
QSlider::groove:horizontal {
  background-color: #555555;
  border-radius: 4px;
}
QSlider::handle:horizontal {
  background-color: #5dade2;
  width: 12px;
  margin: -4px 0;
  border-radius: 6px;
}
QSlider::handle:horizontal:hover {
  background-color: #7bbcef;
}
        """
        ui_widget.setStyleSheet(dark_stylesheet)
        
        # Update QLineEdit, QComboBox, and QLabel for dark mode
        update_line_edit_and_combo_box(ui_widget)

def update_line_edit_and_combo_box(parent):
    """
    Recursively apply dark mode styles to QLineEdit, QComboBox, and QLabel widgets.
    """
    # Update QLabel
    if isinstance(parent, qt.QLabel):
        try:
            parent.setStyleSheet("""
                QLabel {
                  color: #ffffff;
                  font-weight: 500;
                }
            """)
        except (AttributeError, RuntimeError):
            # A widget without that method, or whose C++ object is already gone.
            pass
    
    # Update QLineEdit
    if isinstance(parent, qt.QLineEdit):
        try:
            parent.setStyleSheet("""
                QLineEdit {
                  background-color: #3c3c3c;
                  border: 1px solid #555555;
                  border-radius: 4px;
                  padding: 6px;
                  color: #ffffff;
                }
                QLineEdit:focus {
                  border: 2px solid #5dade2;
                }
            """)
        except (AttributeError, RuntimeError):
            pass
    
    # Update QComboBox
    if isinstance(parent, qt.QComboBox):
        try:
            parent.setStyleSheet("""
                QComboBox {
                  background-color: #3c3c3c;
                  border: 1px solid #555555;
                  border-radius: 4px;
                  padding: 4px 6px;
                  color: #ffffff;
                }
                QComboBox:focus {
                  border: 2px solid #5dade2;
                }
                QComboBox::drop-down {
                  width: 20px;
                  border: none;
                }
                QComboBox QAbstractItemView {
                  background-color: #3c3c3c;
                  color: #ffffff;
                  selection-background-color: #5dade2;
                }
            """)
        except (AttributeError, RuntimeError):
            pass
    
    # Recursively update all children
    if hasattr(parent, 'children'):
        for child in parent.children():
            update_line_edit_and_combo_box(child)

# ---------------------------------------------------------------------------
# The button sheet VFACE carried four times.
#
# Those four blocks -- standard/cancel x dark/light -- were byte-identical once
# every colour was replaced by a token: one 477-character template, seven
# colours. And dark and light differed only in the two `:disabled` colours; the
# gradients were the same. So what looked like four stylesheets is one template,
# two accents and two disabled pairs.
#
# Only the accent and the disabled pair are parameters. Everything else -- the
# radius, the weight, the padding, the indentation -- is reproduced to the
# character, because these sheets are what the user sees and this move is meant
# to change nothing on screen.
# ---------------------------------------------------------------------------

#: Each accent gives the three gradients: normal, hover, pressed.
BUTTON_ACCENTS = {
    "primary": (("#4ba3ff", "#3498db"), ("#5cb3ff", "#2980b9"), ("#2980b9", "#1f618d")),
    "danger": (("#e74c3c", "#c0392b"), ("#ec7063", "#a93226"), ("#a93226", "#922b21")),
    # Quiet slate for the buttons that are not the point of the panel
    # (Browse, Download, Test Files): the same shape, so it sits beside the others.
    "secondary": (("#6c7a89", "#5d6d7e"), ("#7b8a99", "#6c7a89"), ("#5d6d7e", "#4d5b6a")),
    # Two more, for buttons a panel wants told apart from Browse: a transfer
    # (Download) and a trial run (Test Files).
    "violet": (("#a569bd", "#8e44ad"), ("#b07cc6", "#7d3c98"), ("#7d3c98", "#6c3483")),
    "orange": (("#f5b041", "#e67e22"), ("#f8c471", "#ca6f1e"), ("#ca6f1e", "#af601a")),
}

#: Background and foreground of a disabled button, per palette.
BUTTON_DISABLED = {True: ("#555555", "#888888"), False: ("#bdc3c7", "#95a5a6")}


def button_stylesheet(accent="primary", dark=None):
    """The stylesheet of a push button, for one accent and one palette.

    `accent` is a key of `BUTTON_ACCENTS`; `dark` defaults to what the
    application currently shows.
    """
    if dark is None:
        dark = is_dark_mode()
    (normal, hover, pressed) = BUTTON_ACCENTS[accent]
    (off_background, off_text) = BUTTON_DISABLED[bool(dark)]
    gradient = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 %s, stop:1 %s)"
    return (
        "\n"
        "            QPushButton {\n"
        "              background-color: " + gradient % normal + ";\n"
        "              color: white;\n"
        "              border: none;\n"
        "              border-radius: 6px;\n"
        "              font-weight: 600;\n"
        "              font-size: 10pt;\n"
        "              padding: 8px;\n"
        "              margin-top: 4px;\n"
        "            }\n"
        "            QPushButton:hover:!pressed {\n"
        "              background-color: " + gradient % hover + ";\n"
        "            }\n"
        "            QPushButton:pressed {\n"
        "              background-color: " + gradient % pressed + ";\n"
        "            }\n"
        "            QPushButton:disabled {\n"
        "              background-color: " + off_background + ";\n"
        "              color: " + off_text + ";\n"
        "            }\n"
        "            "
    )


def apply_button_style(ui, names, accent="primary", dark=None):
    """Set that sheet on every widget of `ui` named in `names` that exists."""
    sheet = button_stylesheet(accent, dark)
    for name in names:
        widget = getattr(ui, name, None)
        if widget is not None:
            widget.setStyleSheet(sheet)


# ---------------------------------------------------------------------------
# The same look for the inputs and the drop-down lists, in both palettes.
#
# `update_line_edit_and_combo_box` above paints them for the dark palette only,
# with the colours written into the call, so under a light palette a panel showed
# Slicer's own fields next to styled buttons. One template, two colour sets: the
# dark set is the one the function above already uses, colour for colour. The
# old `::drop-down { border: none }` rule hid the arrow; here the arrow is
# painted back, in the text colour, so a list looks like a list.
# ---------------------------------------------------------------------------

#: background, border, text, focus border and selection, per palette.
INPUT_COLORS = {
    True: ("#3c3c3c", "#555555", "#ffffff", "#5dade2", "#5dade2"),
    False: ("#ffffff", "#bdc3c7", "#2c3e50", "#3498db", "#3498db"),
}


def input_stylesheet(dark=None, arrow=None):
    """The stylesheet of a QLineEdit and a QComboBox, for one palette.

    `dark` defaults to what the application currently shows. `arrow` is the path
    of an image for the drop-down arrow (see `_arrow_image`): without one the
    list is styled but Qt draws no arrow of its own once its background is set,
    and a list with no arrow does not look like a list.
    """
    if dark is None:
        dark = is_dark_mode()
    (background, border, text, focus, selection) = INPUT_COLORS[bool(dark)]
    arrow_rules = ""
    if arrow:
        arrow_rules = (
            "            QComboBox::drop-down {\n"
            "              subcontrol-origin: padding;\n"
            "              subcontrol-position: center right;\n"
            "              width: 22px;\n"
            "              border: none;\n"
            "            }\n"
            "            QComboBox::down-arrow {\n"
            "              image: url(" + arrow.replace("\\", "/") + ");\n"
            "              width: 10px;\n"
            "              height: 6px;\n"
            "            }\n"
        )
    return (
        "\n"
        "            QLineEdit {\n"
        "              background-color: " + background + ";\n"
        "              border: 1px solid " + border + ";\n"
        "              border-radius: 4px;\n"
        "              padding: 6px;\n"
        "              color: " + text + ";\n"
        "            }\n"
        "            QLineEdit:focus {\n"
        "              border: 2px solid " + focus + ";\n"
        "            }\n"
        "            QComboBox {\n"
        "              background-color: " + background + ";\n"
        "              border: 1px solid " + border + ";\n"
        "              border-radius: 4px;\n"
        "              padding: 4px 6px;\n"
        "              color: " + text + ";\n"
        "            }\n"
        "            QComboBox:focus {\n"
        "              border: 2px solid " + focus + ";\n"
        "            }\n"
        + arrow_rules +
        "            QComboBox QAbstractItemView {\n"
        "              background-color: " + background + ";\n"
        "              color: " + text + ";\n"
        "              selection-background-color: " + selection + ";\n"
        "            }\n"
        "            "
    )


_ARROW_FOLDER = "adtlib-arrows"


def _arrow_image(color):
    """Path of a small down-pointing triangle of `color`, drawn once and kept.

    A stylesheet can only point at an image file, and the extension installs
    none, so the arrow is painted at run time into the temporary folder.
    """
    import os
    import tempfile
    folder = os.path.join(tempfile.gettempdir(), _ARROW_FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "down_%s.png" % color.lstrip("#"))
    if not os.path.isfile(path):
        pixmap = qt.QPixmap(10, 6)
        pixmap.fill(qt.QColor(0, 0, 0, 0))
        painter = qt.QPainter(pixmap)
        painter.setRenderHint(qt.QPainter.Antialiasing)
        painter.setPen(qt.Qt.NoPen)
        painter.setBrush(qt.QColor(color))
        triangle = qt.QPolygonF()
        for x, y in ((0.5, 0.5), (9.5, 0.5), (5.0, 5.5)):
            triangle.append(qt.QPointF(x, y))
        painter.drawPolygon(triangle)
        painter.end()
        pixmap.save(path, "PNG")
    return path


def _named_widgets(ui):
    """The widgets a `slicer.util.childWidgetVariables(...)` result holds, by name."""
    return {name: widget for name, widget in vars(ui).items() if not name.startswith("_")}


def apply_input_style(ui, dark=None):
    """Give every QLineEdit and QComboBox of `ui` the input sheet of the palette."""
    if dark is None:
        dark = is_dark_mode()
    try:
        arrow = _arrow_image(INPUT_COLORS[bool(dark)][2])
    except Exception:  # no arrow rather than no panel
        arrow = None
    sheet = input_stylesheet(dark, arrow)
    for widget in _named_widgets(ui).values():
        if isinstance(widget, (qt.QLineEdit, qt.QComboBox)):
            try:
                widget.setStyleSheet(sheet)
            except (AttributeError, RuntimeError):
                # A widget whose C++ object is already gone.
                pass


def apply_theme(ui_widget, ui, *, primary=(), danger=(), secondary=None, accents=None):
    """One call that gives a panel the extension's look, dark or light.

    `ui_widget` is the loaded panel and `ui` its `childWidgetVariables`.
    The inputs and the drop-down lists get the input sheet; the buttons named in
    `primary` and `danger` get those accents; every other QPushButton gets
    `secondary`, unless `secondary` is a tuple of names, in which case only
    those do. Pass `secondary=()` to leave the other buttons as Slicer draws them.
    `accents` names more groups, `{"violet": ("download",), "orange": ("test",)}`:
    any key of `BUTTON_ACCENTS`.
    A module can call it, or keep its own styling: nothing else depends on it.
    """
    dark = is_dark_mode()
    apply_dark_mode(ui_widget)  # does nothing under a light palette
    apply_input_style(ui, dark)
    apply_button_style(ui, primary, "primary", dark)
    apply_button_style(ui, danger, "danger", dark)
    accents = accents or {}
    for accent, names in accents.items():
        apply_button_style(ui, names, accent, dark)
    if secondary is None:
        taken = set(primary) | set(danger) | {name for names in accents.values() for name in names}
        secondary = [name for name, widget in _named_widgets(ui).items()
                     if isinstance(widget, qt.QPushButton) and name not in taken]
    apply_button_style(ui, secondary, "secondary", dark)
