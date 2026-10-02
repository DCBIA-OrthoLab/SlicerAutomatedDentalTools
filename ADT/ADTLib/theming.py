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
