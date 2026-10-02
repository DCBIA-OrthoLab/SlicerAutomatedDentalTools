import json
import zipfile
from pathlib import Path
from typing import Optional, Callable

import qt
import slicer
from github import Github, GithubException


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
import shutil
import traceback

logger = get_logger("BatchDentalSeg_pythonDependency")

def hasInternetConnection(time_out_sec=2) -> bool:
    """
    Check if user has access to the internet.
    """
    import requests
    try:
        requests.get("https://www.github.com", timeout=time_out_sec)
        return True
    except requests.ConnectionError:
        return False


class PythonDependencyChecker:
    """
    Class responsible for installing the Modules dependencies and downloading the model weights.
    """

    def __init__(
            self,
            repoPath: Optional[str] = None,
            destWeightFolder: Optional[Path] = None,
            hasInternetConnectionF: Optional[Callable[[], bool]] = None,
            errorDisplayF=None
    ):
        """
        :param repoPath: Optional path to the github repository from which the weights will be downloaded from.
        :param destWeightFolder: Optional path to where the weights will be saved.
        :param hasInternetConnectionF: Optional function returning True when internet connection is available, False
            otherwise.
        :param errorDisplayF: Optional function used to display error information.
        """
        from .SegmentationWidget import SegmentationWidget
        self.dependencyChecked = False
        self.destWeightFolder = Path(destWeightFolder or SegmentationWidget.nnUnetFolder())
        self.repo_path = repoPath or "gaudot/SlicerDentalSegmentator"
        self.hasInternetConnectionF = hasInternetConnectionF or hasInternetConnection
        self.errorDisplay = errorDisplayF or slicer.util.errorDisplay

    @classmethod
    def areDependenciesSatisfied(cls):
        try:
            import torch  # noqa: F401  (sonde de disponibilite)
            import nnunetv2  # noqa: F401  (sonde de disponibilite)
            return True
        except ImportError:
            return False

    def downloadWeightsIfNeeded(self, progress_callback):
        if self.areWeightsMissing():
            return self.downloadWeights(progress_callback)

        elif self.areWeightsOutdated():
            if qt.QMessageBox.question(
                    None,
                    "New weights are available",
                    "New weights are available. Would you like to download them?"
            ):
                return self.downloadWeights(progress_callback)
        return True

    def areWeightsMissing(self):
        dataset = self.getDatasetPath()
        if dataset is None:
            return True

        # The weights, not the manifest lying next to them. dataset.json,
        # plans.json and download_info.json are committed to this
        # repository, so every install ships the two markers of a finished
        # download without a single .pth. Asking "is dataset.json there?"
        # therefore answered False on a tree that had no weights at all:
        # nothing was ever downloaded, and nnUNet refused with
        #   Model folder is missing the following folds : ['fold_0']
        # which reads as a broken installation rather than as a model that
        # was never fetched. DentalSegmentator is the first entry of the
        # model list, so this is what anybody gets by default.
        #
        # Same trap as the 0-byte checkpoint_final.pth that blocked AMASSS
        # for good: look at the weight, never at the marker beside it.
        return not any(dataset.parent.glob("fold_*/*.pth"))

    def getLatestReleaseUrl(self):
        g = Github()
        repo = g.get_repo(self.repo_path)
        assets = [asset for release in repo.get_releases() for asset in release.get_assets()]
        return assets[0].browser_download_url

    def areWeightsOutdated(self) -> bool:
        """
        :returns: True if weights information are missing or internet connection is available and weights information
            don't match the ones on the GitHub page. False otherwise.
        """
        if not self.getWeightDownloadInfoPath().exists():
            return True

        if not self.hasInternetConnectionF():
            return False

        try:
            return self.getLastDownloadedWeights() != self.getLatestReleaseUrl()
        except GithubException:
            return False

    def getDestWeightFolder(self):
        return self.destWeightFolder

    def getDatasetPath(self):
        try:
            return next(self.destWeightFolder.rglob("dataset.json"))
        except StopIteration:
            return None

    def getWeightDownloadInfoPath(self):
        return self.destWeightFolder / "download_info.json"

    def getLastDownloadedWeights(self):
        if not self.getWeightDownloadInfoPath().exists():
            return None

        with open(self.getWeightDownloadInfoPath(), "r") as f:
            return json.loads(f.read()).get("download_url")

    def downloadWeights(self, progress_callback) -> bool:
        """
        Removes the weight folder and tries to download the weights from the GitHub page.
        If an internet connection is not available, keeps the current weights unchanged.

        :returns: True if download was successful. False in case of no internet or failure during download.
        """
        import requests

        progress_callback("Downloading model weights...")
        if not self.hasInternetConnectionF():
            self.errorDisplay(
                "Failed to download weights (no internet connection). "
                "Please retry or manually install them to proceed.\n"
                "To manually install the weights, please refer to the documentation here :\n"
                "https://github.com/gaudot/SlicerDentalSegmentator",
            )
            return False

        if self.destWeightFolder.exists():
            shutil.rmtree(self.destWeightFolder)
        self.destWeightFolder.mkdir(parents=True, exist_ok=True)

        try:
            download_url = self.getLatestReleaseUrl()
            session = requests.Session()
            response = session.get(download_url, stream=True)
            response.raise_for_status()

            file_name = download_url.split("/")[-1]
            dest_zip_path = self.destWeightFolder / file_name
            with open(dest_zip_path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)

            self.extractWeightsToWeightsFolder(dest_zip_path)
            self.writeDownloadInfoURL(download_url)
            return True
        except Exception:  # noqa
            self.errorDisplay(
                "Failed to download weights. Please retry or manually install them to proceed.\n"
                "To manually install the weights, please refer to the documentation here :\n"
                "https://github.com/gaudot/SlicerDentalSegmentator",
                detailedText=traceback.format_exc()
            )
            return False

    def extractWeightsToWeightsFolder(self, zip_path):
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(self.destWeightFolder)

    def writeDownloadInfoURL(self, download_url):
        with open(self.destWeightFolder / "download_info.json", "w") as f:
            f.write(json.dumps({"download_url": download_url}))
