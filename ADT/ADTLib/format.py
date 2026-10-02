"""Formatting shared by the modules' installation and progress labels.

Pure functions, standard library only: this module is imported by the widgets
inside Slicer and must stay importable from the lean Conda environment too.
"""
import time


def format_elapsed(seconds):
    """Seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def format_timer(seconds):
    """The duration as the progress labels display it.

    These three branches existed in twelve copies, word for word, in the
    widgets of ALI, ASO, AREG, AutoMatrix and MedX -- the last one archived
    since, which leaves ten in the tree. The format is preserved character for
    character -- it is a label the user reads during the run, not a format to
    modernise along the way.

        5      -> "Time : 5s"
        192    -> "Time : 3min and 12s"
        3723   -> "Time : 1h, 2min and 3s"

    A thirteenth copy exists, in `MRI2CBCT.py`, and it writes `"Time: "`
    without a space before the colon. It stays where it is: aligning it would
    change a label the user reads during their run, which is not what this pass
    is about.
    """
    if seconds < 60:
        return f"Time : {int(seconds)}s"
    if seconds < 3600:
        return f"Time : {int(seconds/60)}min and {int(seconds%60)}s"
    return (f"Time : {int(seconds/3600)}h, {int(seconds%3600/60)}min "
            f"and {int(seconds%60)}s")


def elapsed_since(start_time):
    """Seconds since `start_time`, as returned by `time.time()`."""
    return time.time() - start_time
