"""
Progress Display Utilities for VFACE Module

This module provides progress tracking and display classes for various
processing steps in the VFACE analysis pipeline.

Each Display class is responsible for tracking progress and formatting
progress messages for different types of processing modules.
"""

from abc import ABC, abstractmethod
import os
from typing import Tuple


class Display(ABC):
    """
    Abstract base class for progress display.
    
    Defines the interface for tracking and displaying progress information
    during processing tasks.
    """
    
    def __init__(self) -> None:
        """Initialize progress tracking attributes."""
        self.progress: float = 0
        self.progress_bar: float = 0
        self.message: str = 0

    @abstractmethod
    def __call__(self, *args, **kwds) -> Tuple[float, str]:
        """
        Update and return progress information.
        
        Returns:
            Tuple[float, str]: Progress percentage (0-100) and status message
        """
        return self.progress_bar, self.message

    @abstractmethod
    def isProgress(self, **kwds) -> bool:
        """
        Check if progress should be updated.
        
        Returns:
            bool: True if progress should be displayed, False otherwise
        """
        pass


class DisplayCrownSeg(Display):
    """Progress display for crown segmentation processing."""
    
    def __init__(self, nb_scan: int, log_path: str, msg: str) -> None:
        """
        Initialize crown segmentation progress tracker.
        
        Args:
            nb_scan: Total number of scans to process
            log_path: Path to log file for monitoring progress
            msg: Custom message prefix for progress display
        """
        self.nb_scan_total = nb_scan
        self.time_log = 0
        self.log_path = log_path
        self.msg = msg
        super().__init__()
        self.progress = -1

    def __call__(self) -> Tuple[float, str]:
        """Update and return segmentation progress."""
        self.progress += 1
        if self.nb_scan_total == 0:
            self.progress_bar = 0
        else:
            self.progress_bar = self.progress / self.nb_scan_total * 100
        self.message = f"{self.msg}: {self.progress} / {self.nb_scan_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Check if log file has been updated."""
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True
        return out


class DisplayASOIOS(Display):
    """Progress display for ASO IOS (iOS-based registration) processing."""
    
    def __init__(self, nb_progress: int, log_path: str, msg: str) -> None:
        """
        Initialize ASO IOS progress tracker.
        
        Args:
            nb_progress: Total number of progress steps
            log_path: Path to log file for monitoring progress
            msg: Custom message prefix for progress display
        """
        self.nb_progress_total = nb_progress
        self.log_path = log_path
        self.time_log = 0
        self.msg = msg
        super().__init__()

    def __call__(self, **kwds) -> Tuple[float, str]:
        """Update and return processing progress."""
        self.progress += 1
        self.progress_bar = self.progress / self.nb_progress_total * 100
        self.message = f"{self.msg}: {self.progress} / {self.nb_progress_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Check if log file has been updated."""
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True
        return out


class DisplayAREGIOS(Display):
    """Progress display for AREG IOS (Automated Registration - iOS) processing."""
    
    def __init__(self, nb_progress: int, log_path: str) -> None:
        """
        Initialize AREG IOS progress tracker.
        
        Args:
            nb_progress: Total number of progress steps
            log_path: Path to log file for monitoring progress
        """
        self.nb_progress_total = nb_progress
        self.log_path = log_path
        self.time_log = 0
        super().__init__()
        self.progress = -1

    def __call__(self, **kwds) -> Tuple[float, str]:
        """Update and return registration progress."""
        self.progress += 1
        self.progress_bar = self.progress / (self.nb_progress_total * 3) * 100
        self.message = f"Patient: {self.progress // 3} / {self.nb_progress_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Check if log file has been updated."""
        out = False
        if os.path.isfile(self.log_path):
            path_time = os.path.getmtime(self.log_path)
            if path_time != self.time_log:
                self.time_log = path_time
                out = True
        return out


class DisplayAREGCBCT(Display):
    """Progress display for AREG CBCT (Automated Registration - CBCT) processing."""
    
    def __init__(self, nb_progress: int) -> None:
        """
        Initialize AREG CBCT progress tracker.
        
        Args:
            nb_progress: Total number of scans to process
        """
        self.nb_progress_total = nb_progress
        self.time_log = 0
        super().__init__()

    def __call__(self, **kwds) -> Tuple[float, str]:
        """Update and return registration progress."""
        self.progress += 1
        self.progress_bar = self.progress / self.nb_progress_total * 100
        self.message = f"Scan: {self.progress} / {self.nb_progress_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Determine if progress should be updated based on CLI progress."""
        out = False
        if kwds["progress"] == 200 and kwds["updateProgessBar"] == False:
            out = True
        return out


class DisplayAMASSS(Display):
    """Progress display for AMASSS (Automatic Mandible and Maxilla Segmentation) processing."""
    
    def __init__(self, nb_patient: int, nb_seg_struc: int, nb_reg=None) -> None:
        """
        Initialize AMASSS segmentation progress tracker.
        
        Args:
            nb_patient: Number of patients to process
            nb_seg_struc: Number of structures to segment
            nb_reg: Optional number of registrations per patient
        """
        self.nb_struct = nb_seg_struc
        self.nb_scan_total = nb_patient * nb_reg if nb_reg is not None else nb_patient
        self.pred_step = 0
        super().__init__()

    def __call__(self) -> Tuple[float, str]:
        """Update and return segmentation progress."""
        self.progress += 1 / (12 + self.nb_struct)
        self.progress_bar = (
            (self.progress / (self.nb_scan_total)) * 100
            if self.progress_bar < 100
            else 99
        )
        nb_scan_done = int(self.progress)
        self.message = f"Scan: {nb_scan_done} / {self.nb_scan_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Determine if progress should be updated based on CLI progress."""
        out = False
        if kwds["progress"] == 200:
            self.pred_step += 1
        if kwds["progress"] == 100 and kwds["updateProgessBar"] == False:
            if self.pred_step > 3:
                out = True
        return out


class DisplayAREGIOSCBCT(Display):
    """Progress display for combined AREG IOS and CBCT processing."""
    
    def __init__(self, nb_progress: int) -> None:
        """
        Initialize combined AREG registration progress tracker.
        
        Args:
            nb_progress: Total number of scans to process
        """
        self.nb_progress_total = nb_progress
        self.time_log = 0
        super().__init__()

    def __call__(self, **kwds) -> Tuple[float, str]:
        """Update and return registration progress."""
        self.progress += 1
        self.progress_bar = self.progress / self.nb_progress_total * 100
        self.message = f"Scan: {self.progress} / {self.nb_progress_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Determine if progress should be updated based on CLI progress."""
        out = False
        if kwds["progress"] == 200 and kwds["updateProgessBar"] == False:
            out = True
        return out


class DisplayASOCBCT(Display):
    """Progress display for ASO CBCT (Automated Skull Orientation - CBCT) processing."""
    
    def __init__(self, nb_progress: int) -> None:
        """
        Initialize ASO CBCT progress tracker.
        
        Args:
            nb_progress: Total number of scans to process
        """
        self.nb_progress_total = nb_progress
        self.time_log = 0
        super().__init__()

    def __call__(self, **kwds) -> Tuple[float, str]:
        """Update and return orientation progress."""
        self.progress += 1
        self.progress_bar = self.progress / self.nb_progress_total * 100
        self.message = f"Scan: {self.progress} / {self.nb_progress_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Determine if progress should be updated based on CLI progress."""
        out = False
        if kwds["progress"] == 200 and kwds["updateProgessBar"] == False:
            out = True
        return out


class DisplayALICBCT(Display):
    """Progress display for ALI CBCT (Automated Landmark Identification - CBCT) processing."""
    
    def __init__(self, nb_landmark: int, nb_scan: int) -> None:
        """
        Initialize ALI CBCT landmark detection progress tracker.
        
        Args:
            nb_landmark: Total number of landmarks to detect
            nb_scan: Total number of scans to process
        """
        self.nb_landmark = nb_landmark
        self.nb_scan_total = nb_scan
        self.pred_step = 0
        super().__init__()

    def __call__(self) -> Tuple[float, str]:
        """Update and return landmark detection progress."""
        self.progress += 0.39
        self.progress_bar = (
            self.progress / (self.nb_landmark * self.nb_scan_total)
        ) * 100
        nb_scan_treat = int(self.progress // self.nb_landmark)
        self.message = f"Landmarks: {round(self.progress)} / {self.nb_landmark * self.nb_scan_total} | Patient: {nb_scan_treat} / {self.nb_scan_total}"
        return self.progress_bar, self.message

    def isProgress(self, **kwds) -> bool:
        """Determine if progress should be updated based on CLI progress."""
        out = False
        if kwds["progress"] == 200:
            self.pred_step += 1
        if kwds["progress"] == 100 and kwds["updateProgessBar"] == False:
            if self.pred_step > 3:
                out = True
        return out