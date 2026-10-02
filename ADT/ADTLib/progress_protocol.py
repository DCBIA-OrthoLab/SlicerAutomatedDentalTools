"""The `<filter-progress>` channel, named once.

The extension's CLIs talk to their interface through a single route: a line
`<filter-progress>x</filter-progress>` on standard output. Slicer reads it,
**multiplies x by a hundred**, and stores the result on the CLI node, from
where `caller.GetProgress()` hands it back on the widget side.

That factor of a hundred is written down nowhere, and it is what produced the
three conventions found in the repository:

- a CLI that prints a **fraction** `0.42` makes 42 show up: that is the
  intended use, and the bar advances;
- a CLI that prints the integer `2` makes 200 show up, which no bar can
  represent. That is not progress, it is an **event**: the interfaces compare
  the value to `200` to learn that one more patient is done. The CLI sends it
  as a **pulse** `0 -> 2 -> 0`, because Slicer only notifies its widget when
  the value *changes*;
- and AMASSS, which received both, guessed at run time which one it was holding
  (`if progress > 1: progress /= 100`).

Here the three are named. The constants carry the value **as the interface sees
it**, since that is where it is compared; the division by a hundred lives in a
single place, just below.

Standard library only: imported from the Conda environment by the CLIs, and
from Slicer by the widgets.
"""
import sys
import time

# What Slicer does to the printed value before handing it to the widget.
SCALE = 100

# The two events, in the unit the interface reads them in.
STEP_DONE = 100      # the CLI printed 1: one more step is done
PATIENT_DONE = 200   # the CLI printed 2: one more patient is done

# Slicer only reports value changes: a pulse must therefore come back down,
# and leave the event loop the time to see it. The delay is the one used by the
# four CLIs that already sent 0 -> 2 -> 0.
PULSE_PAUSE = 0.2


def emit(value, flush=True):
    """Write a raw value on the channel. The next two functions call it."""
    print(f"<filter-progress>{value}</filter-progress>")
    if flush:
        sys.stdout.flush()


def emit_fraction(fraction):
    """How far along the run is, between 0 and 1.

    This is the intended use of the channel: the progress bar follows.
    """
    emit(f"{fraction:.4f}")


def emit_event(event, pause=PULSE_PAUSE):
    """Report an event to the interface, as a pulse.

    `event` is one of the constants above. The value comes back down to zero
    because Slicer only wakes the widget on a change: without the return to
    zero, two events in a row would pass for one.
    """
    emit(0)
    time.sleep(pause)
    emit(event // SCALE)
    time.sleep(pause)
    emit(0)
    time.sleep(pause)


def is_event(progress, event):
    """Is the value the widget received this event?"""
    return progress == event


def as_fraction(progress):
    """The received value, brought back between 0 and 1.

    A widget that receives both fractions and pulses has no way to tell them
    apart other than by scale: past 1, the value comes from the factor of a
    hundred. This is what AMASSS did by hand.
    """
    return progress / SCALE if progress > 1 else progress
