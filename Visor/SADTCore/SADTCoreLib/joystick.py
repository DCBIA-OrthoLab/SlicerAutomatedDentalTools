"""A 2D joystick pad: both values of a `"vec2"` argument set with one gesture.

Ported from FlexReg's JoystickPad (the butterfly-patch corner pads), minus the
arch-specific mirrors (`outward_right`, `adjust_sign`): here an axis is
described by its range alone, and a mirrored axis is declared by inverting the
range: `x_range: [15, -15]` puts 15 at the left end. By construction
`x_range[0]`/`y_range[0]` is the left/bottom end of its axis and index 1 the
right/top end, and the optional end labels pair with the range by index.

This is a genuine QWidget subclass, unlike formgen's holder classes: painting
and mouse handling cannot be composed from stock widgets. FlexReg ships the
same kind of subclass under Slicer's PythonQt, which is what makes it safe.

No Qt signals (PythonQt cannot declare new ones on a Python subclass), so
callers assign plain callables, exactly like FlexReg:

- `pad.onChanged(pad)` after every value change made through the pad;
- `pad.onReleased(pad)` when a spring-back gesture ends;
- `pad.onWheel(pad, steps)` to take the wheel over entirely.

Gestures (all funnel into setValues): drag is absolute (the knob jumps under
the cursor), Ctrl+drag is anchored and `FINE` times finer, the wheel walks the
vertical axis one step per notch (Shift+wheel the horizontal one), arrow keys
one step each, double-click returns to the defaults. A `spring_back` pad is
relative instead: each drag deals out a displacement and the knob returns to
its defaults on release, so repeated pushes never saturate against the ends.
"""

import qt
import slicer

from . import design


def _event_position(event):
    """Read a mouse position out of a Qt event. PythonQt exposes some getters
    as plain attributes and others as methods depending on the build, so try
    both rather than betting on one."""
    position = event.pos
    if callable(position):
        position = position()
    x = position.x() if callable(position.x) else position.x
    y = position.y() if callable(position.y) else position.y
    return float(x), float(y)


def _wheel_steps(event):
    """Number of notches scrolled, positive upwards."""
    try:
        delta = event.angleDelta
        delta = delta() if callable(delta) else delta
        value = delta.y() if callable(delta.y) else delta.y
    except AttributeError:
        value = event.delta
        if callable(value):
            value = value()
    return float(value) / 120.0


def _modifiers():
    """Current keyboard modifiers, 0 outside a running Slicer (unit tests)."""
    try:
        return slicer.app.keyboardModifiers()
    except Exception:
        return 0


class JoystickPad(qt.QWidget):

    GUTTER = 11       # room above and below the pad, for the y-axis end labels
    SIDE_GUTTER = 15  # room either side, for the x-axis end labels
    KNOB = 7
    FINE = 0.2        # sensitivity multiplier while Ctrl is held

    def __init__(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0), x_step=None, y_step=None,
                 x_labels=None, y_labels=None, spring_back=False, size=None, parent=None):
        qt.QWidget.__init__(self, parent)
        self.x_start, self.x_end = float(x_range[0]), float(x_range[1])
        self.y_start, self.y_end = float(y_range[0]), float(y_range[1])
        # Clamping bounds, whichever way the axis is declared.
        self._x_lo, self._x_hi = sorted((self.x_start, self.x_end))
        self._y_lo, self._y_hi = sorted((self.y_start, self.y_end))
        # Which numeric direction moves the knob right/up: screen-directional
        # gestures (arrows, wheel) multiply their step by this.
        self._x_dir = 1.0 if self.x_end >= self.x_start else -1.0
        self._y_dir = 1.0 if self.y_end >= self.y_start else -1.0
        # One wheel notch, or one arrow key, walks a hundredth of the axis.
        self.x_step = float(x_step) if x_step else abs(self.x_end - self.x_start) / 100.0
        self.y_step = float(y_step) if y_step else abs(self.y_end - self.y_start) / 100.0
        self.x_labels = tuple(x_labels) if x_labels else None
        self.y_labels = tuple(y_labels) if y_labels else None
        self.spring_back = bool(spring_back)
        self.SIDE = int(size or design.PAD_SIZE)

        centre_x = (self.x_start + self.x_end) / 2.0
        centre_y = (self.y_start + self.y_end) / 2.0
        # NOT self.x / self.y: QWidget already owns those as its position, and
        # PythonQt refuses the assignment outright -- "Property 'x' of
        # JoystickPad object is not writable", which takes the whole panel down
        # with it. Invisible against a stub that is not a real QWidget.
        self.value_x = centre_x
        self.value_y = centre_y
        self.default_x = centre_x
        self.default_y = centre_y

        self.onChanged = None
        self.onReleased = None
        self.onWheel = None
        self._dragging = False
        self._anchor = None

        self.setFixedSize(self.SIDE, self.SIDE)
        self.setFocusPolicy(qt.Qt.StrongFocus)
        self.setToolTip(
            "Drag to set both values at once.\n"
            "Ctrl+drag : five times finer\n"
            "Wheel : one vertical step, Shift+wheel : one horizontal step\n"
            "Arrow keys : one step, double-click : back to the default"
        )

    # ---- values ---------------------------------------------------------

    def setValues(self, x, y, notify=False):
        x = min(max(float(x), self._x_lo), self._x_hi)
        y = min(max(float(y), self._y_lo), self._y_hi)
        if x == self.value_x and y == self.value_y:
            return
        self.value_x = x
        self.value_y = y
        self.update()
        if notify and self.onChanged:
            self.onChanged(self)

    def setDefaults(self, x, y):
        """Where a double-click (and a spring-back release) sends the knob."""
        self.default_x = float(x)
        self.default_y = float(y)

    # ---- geometry -------------------------------------------------------

    def _box(self):
        """The pad itself. Labels sit outside it, the knob never leaves it.
        A gutter is only reserved when its axis actually has end labels."""
        side = self.SIDE_GUTTER if self.x_labels else 0
        vertical = self.GUTTER if self.y_labels else 0
        return side, vertical, self.SIDE - 2 * side, self.SIDE - 2 * vertical

    def _area(self):
        """Where the centre of the knob is allowed to travel."""
        left, top, width, height = self._box()
        inset = self.KNOB + 2
        return left + inset, top + inset, width - 2 * inset, height - 2 * inset

    def _knobPosition(self):
        left, top, width, height = self._area()
        fraction_x = (self.value_x - self.x_start) / (self.x_end - self.x_start)
        fraction_y = (self.value_y - self.y_start) / (self.y_end - self.y_start)
        # Screen y grows downwards; the axis end (index 1) is the top.
        return left + fraction_x * width, top + (1.0 - fraction_y) * height

    def _valuesAt(self, x, y):
        left, top, width, height = self._area()
        fraction_x = min(max((x - left) / float(width), 0.0), 1.0)
        fraction_y = min(max(1.0 - (y - top) / float(height), 0.0), 1.0)
        return (self.x_start + fraction_x * (self.x_end - self.x_start),
                self.y_start + fraction_y * (self.y_end - self.y_start))

    # ---- interaction ----------------------------------------------------

    def _isFine(self):
        return bool(_modifiers() & qt.Qt.ControlModifier)

    def mousePressEvent(self, event):
        self._dragging = True
        self._anchor = None
        x, y = _event_position(event)
        if self.spring_back or self._isFine():
            # Relative drag: hold the knob where it is and move from there,
            # rather than jumping it under the cursor.
            self._anchor = ((x, y), self._knobPosition())
            return
        self.setValues(*self._valuesAt(x, y), notify=True)

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        x, y = _event_position(event)
        if self.spring_back or self._isFine():
            if self._anchor is None:
                # Ctrl pressed mid-drag: rebase so the knob does not jump.
                self._anchor = ((x, y), self._knobPosition())
        else:
            self._anchor = None
        if self._anchor is not None:
            (anchor_x, anchor_y), (knob_x, knob_y) = self._anchor
            scale = self.FINE if self._isFine() else 1.0
            x = knob_x + (x - anchor_x) * scale
            y = knob_y + (y - anchor_y) * scale
        self.setValues(*self._valuesAt(x, y), notify=True)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._anchor = None
        if self.spring_back:
            # The displacement has been dealt out: back to the rest position,
            # silently, ready for the next push.
            self.setValues(self.default_x, self.default_y)
            if self.onReleased:
                self.onReleased(self)

    def mouseDoubleClickEvent(self, event):
        self.setValues(self.default_x, self.default_y, notify=True)

    def wheelEvent(self, event):
        """The wheel is a vertical gesture, so it drives the vertical axis:
        scrolling up walks the knob up. Shift swaps it onto the other axis,
        where up means right."""
        steps = _wheel_steps(event)
        if self.onWheel is not None:
            self.onWheel(self, steps)
            return
        if _modifiers() & qt.Qt.ShiftModifier:
            self.setValues(self.value_x + self.x_step * steps * self._x_dir, self.value_y, notify=True)
        else:
            self.setValues(self.value_x, self.value_y + self.y_step * steps * self._y_dir, notify=True)
        self._springNudgeBack()

    def keyPressEvent(self, event):
        key = event.key
        if callable(key):
            key = key()
        # Arrows are screen-directional: Right always walks the knob right,
        # whichever way the axis is declared.
        if key == qt.Qt.Key_Left:
            self.setValues(self.value_x - self.x_step * self._x_dir, self.value_y, notify=True)
        elif key == qt.Qt.Key_Right:
            self.setValues(self.value_x + self.x_step * self._x_dir, self.value_y, notify=True)
        elif key == qt.Qt.Key_Up:
            self.setValues(self.value_x, self.value_y + self.y_step * self._y_dir, notify=True)
        elif key == qt.Qt.Key_Down:
            self.setValues(self.value_x, self.value_y - self.y_step * self._y_dir, notify=True)
        else:
            return
        self._springNudgeBack()

    def _springNudgeBack(self):
        """On a spring-back pad every wheel/key nudge is one dealt-out
        displacement: spring home silently and report the gesture as ended."""
        if not self.spring_back:
            return
        self.setValues(self.default_x, self.default_y)
        if self.onReleased:
            self.onReleased(self)

    # ---- painting -------------------------------------------------------

    def paintEvent(self, event):
        colors = {name: qt.QColor(value) for name, value in design.pad_palette().items()}
        box_left, box_top, box_width, box_height = self._box()
        centre_x = box_left + box_width / 2.0
        centre_y = box_top + box_height / 2.0

        painter = qt.QPainter(self)
        painter.setRenderHint(qt.QPainter.Antialiasing)

        painter.setPen(qt.QPen(colors["border"], 1))
        painter.setBrush(qt.QBrush(colors["background"]))
        painter.drawRoundedRect(box_left, box_top, box_width - 1, box_height - 1, 6, 6)

        painter.setPen(qt.QPen(colors["grid"], 1))
        painter.drawLine(int(box_left + 4), int(centre_y), int(box_left + box_width - 4), int(centre_y))
        painter.drawLine(int(centre_x), int(box_top + 4), int(centre_x), int(box_top + box_height - 4))

        if self.y_labels:
            painter.setFont(qt.QFont("", 6))
            painter.setPen(qt.QPen(colors["text"], 1))
            painter.drawText(qt.QRect(box_left, 0, box_width, self.GUTTER),
                             qt.Qt.AlignCenter, str(self.y_labels[1]))
            painter.drawText(qt.QRect(box_left, self.SIDE - self.GUTTER, box_width, self.GUTTER),
                             qt.Qt.AlignCenter, str(self.y_labels[0]))

        if self.x_labels:
            painter.setFont(qt.QFont("", 6))
            painter.setPen(qt.QPen(colors["label"], 1))
            painter.drawText(qt.QRect(0, 0, self.SIDE_GUTTER, self.SIDE),
                             qt.Qt.AlignCenter, "\n".join(str(self.x_labels[0])))
            painter.drawText(qt.QRect(self.SIDE - self.SIDE_GUTTER, 0, self.SIDE_GUTTER, self.SIDE),
                             qt.Qt.AlignCenter, "\n".join(str(self.x_labels[1])))

        knob_x, knob_y = self._knobPosition()
        painter.setPen(qt.QPen(colors["trail"], 2))
        painter.drawLine(int(centre_x), int(centre_y), int(knob_x), int(knob_y))

        painter.setPen(qt.QPen(colors["knob"].darker(120), 1))
        painter.setBrush(qt.QBrush(colors["knob"]))
        painter.drawEllipse(int(knob_x - self.KNOB), int(knob_y - self.KNOB), 2 * self.KNOB, 2 * self.KNOB)
        painter.end()
