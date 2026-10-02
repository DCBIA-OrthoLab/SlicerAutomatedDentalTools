from abc import ABC, abstractmethod
import os
from typing import Tuple

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import PATIENT_DONE, STEP_DONE, is_event

logger = get_logger("ASO_Progress")


class Display(ABC):
    def __init__(self) -> None:
        self.progress: int = 0
        self.progress_bar: float = 0
        self.message: str = 0

    @abstractmethod
    def __call__(self, *args, **kwds) -> Tuple[float, str]:
        return self.progress_bar, self.message

    @abstractmethod
    def isProgress(self, **kwds) -> bool:
        pass


class DisplayCrownSeg(Display):
    def __init__(self, nb_scan, log_path) -> None:
        self.nb_scan_total = nb_scan
        self.time_log = 0
        self.log_path = log_path
        super().__init__()

    def __call__(self) -> Tuple[float, str]:
        self.progress += 1
        if self.nb_scan_total == 0:
            self.progress_bar = 0
        else:
            self.progress_bar = self.progress / self.nb_scan_total * 100
        self.message = f"Scan : { self.progress} / {self.nb_scan_total}"

        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True

        return out


class DisplayALIIOS(Display):
    def __init__(self, nb_landmark, nb_scan) -> None:
        self.nb_landmark = nb_landmark
        self.nb_scan_total = nb_scan
        super().__init__()

    def __call__(self) -> Tuple[float, str]:
        self.progress += 1
        self.progress_bar = (
            self.progress / (self.nb_landmark * self.nb_scan_total)
        ) * 100
        nb_scan_treat = int(self.progress // self.nb_landmark)
        self.message = f"Scan : {nb_scan_treat} / {self.nb_scan_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if is_event(kwds["progress"], STEP_DONE) and kwds["updateProgressBar"] == False:
            out = True
        return out


class DisplayASOIOS(Display):
    def __init__(self, nb_progress, mode, log_path) -> None:
        self.nb_progress_total = nb_progress
        self.mode = mode
        self.log_path = log_path
        self.time_log = 0
        super().__init__()

    def __call__(self, **kwds) -> Tuple[float, str]:
        self.progress += 1
        if self.nb_progress_total!=0:
            self.progress_bar = self.progress / self.nb_progress_total * 100
        st = "Scan"
        if "/" in self.mode:
            st = "Patient"

        self.message = f"{st} : {self.progress} / {self.nb_progress_total}"

        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True

        return out


class DisplayASOCBCT(Display):
    def __init__(self, nb_progress) -> None:
        self.nb_progress_total = nb_progress
        self.time_log = 0
        super().__init__()

    def __call__(self, **kwds) -> Tuple[float, str]:
        self.progress += 1
        self.progress_bar = self.progress / self.nb_progress_total * 100
        self.message = f"Scan : {self.progress} / {self.nb_progress_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if is_event(kwds["progress"], PATIENT_DONE) and kwds["updateProgressBar"] == False:
            out = True
        return out


class DisplayALICBCT(Display):
    def __init__(self, nb_landmark, nb_scan) -> None:
        self.nb_landmark = nb_landmark
        self.nb_scan_total = nb_scan
        self.pred_step = 0
        super().__init__()

    def __call__(self) -> Tuple[float, str]:
        self.progress += 0.39
        self.progress_bar = (
            self.progress / (self.nb_landmark * self.nb_scan_total)
        ) * 100
        nb_scan_treat = int(self.progress // self.nb_landmark)
        self.message = f"Landmarks : {round(self.progress)} / {self.nb_landmark * self.nb_scan_total} | Patient : {nb_scan_treat} / {self.nb_scan_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """WARNING -- nothing ever fires this method.

        It expects the events `emit_event` produces, but the CLI that feeds
        it, `ALI_CBCT.py`, sends percentages: the window sees 500 to 10000
        there, never 100 nor 200. See the note next to
        `ALI_CBCT.update_slicer_progress`. The code is left as it is because
        fixing it means deciding what the bar should show.
        """
        out = False
        if is_event(kwds["progress"], PATIENT_DONE):
            self.pred_step += 1
        if is_event(kwds["progress"], STEP_DONE) and kwds["updateProgressBar"] == False:
            if self.pred_step > 3:
                out = True
        return out