from abc import ABC, abstractmethod
import os
from typing import Tuple


class Display(ABC):
    def __init__(self) -> None:
        self.progress: float = 0
        self.progress_bar: float = 0
        self.message: str = 0

    @abstractmethod
    def __call__(self, *args, **kwds) -> Tuple[float, str]:
        return self.progress_bar, self.message

    @abstractmethod
    def isProgress(self, **kwds) -> bool:
        pass


class DisplayCrownSeg(Display):
    def __init__(self, nb_scan, log_path, msg) -> None:
        self.nb_scan_total = nb_scan
        self.time_log = 0
        self.log_path = log_path
        self.msg = msg
        super().__init__()
        self.progress = -1

    def __call__(self) -> Tuple[float, str]:
        self.progress += 1
        if self.nb_scan_total == 0:
            self.progress_bar = 0
        else:
            self.progress_bar = self.progress / self.nb_scan_total * 100
        self.message = f"{self.msg} : {self.progress} / {self.nb_scan_total}"

        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True

        return out


class DisplayASOIOS(Display):
    def __init__(self, nb_progress, log_path, msg) -> None:
        self.nb_progress_total = nb_progress
        self.log_path = log_path
        self.time_log = 0
        self.msg = msg
        super().__init__()

    def __call__(self, **kwds) -> Tuple[float, str]:
        self.progress += 1
        self.progress_bar = self.progress / self.nb_progress_total * 100

        self.message = f"{self.msg}: {self.progress} / {self.nb_progress_total}"

        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True

        return out


class DisplayAREGIOS(Display):
    def __init__(self, nb_progress, log_path) -> None:
        self.nb_progress_total = nb_progress
        self.log_path = log_path
        self.time_log = 0
        super().__init__()
        self.progress = -1

    def __call__(self, **kwds) -> Tuple[float, str]:
        self.progress += 1
        self.progress_bar = self.progress / (self.nb_progress_total * 3) * 100
        self.message = f"Patient : {self.progress // 3} / {self.nb_progress_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True

        return out


class DisplayAREGCBCT(Display):
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
        if kwds["progress"] == 200 and kwds["updateProgessBar"] == False:
            out = True
        return out


class DisplayAMASSS(Display):
    def __init__(self, nb_patient, nb_seg_struc, nb_reg=None) -> None:
        self.nb_struct = nb_seg_struc
        self.nb_scan_total = nb_patient * nb_reg if nb_reg is not None else nb_patient
        self.pred_step = 0
        super().__init__()

    def __call__(self) -> Tuple[float, str]:
        self.progress += 1 / (12 + self.nb_struct)
        self.progress_bar = (
            (self.progress / (self.nb_scan_total)) * 100
            if self.progress_bar < 100
            else 99
        )
        nb_scan_done = int(self.progress)
        self.message = f"Scan : {nb_scan_done} / {self.nb_scan_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if kwds["progress"] == 200:
            self.pred_step += 1
        if kwds["progress"] == 100 and kwds["updateProgessBar"] == False:
            if self.pred_step > 3:
                out = True
        return out


class DisplayAREGIOSCBCT(Display):
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
        if kwds["progress"] == 200 and kwds["updateProgessBar"] == False:
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
        if kwds["progress"] == 200 and kwds["updateProgessBar"] == False:
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
        self.message = f"Scan : {nb_scan_treat} / {self.nb_scan_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if kwds["progress"] == 200:
            self.pred_step += 1
        if kwds["progress"] == 100 and kwds["updateProgessBar"] == False:
            if self.pred_step > 3:
                out = True
        return out
