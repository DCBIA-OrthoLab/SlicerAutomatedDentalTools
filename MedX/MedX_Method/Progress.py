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


class DisplayMedX(Display):
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
        self.message = f"{self.msg}: {self.progress} / {self.nb_scan_total}"

        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True

        return out