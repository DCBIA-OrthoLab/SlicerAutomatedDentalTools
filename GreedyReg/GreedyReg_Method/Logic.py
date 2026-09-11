import os
import sys
import json
import shutil

import vtk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic


class GreedyRegLogic(ScriptedLoadableModuleLogic):
  """Non-UI logic for Greedy Registration: locating/downloading the Greedy
  binary, building parameters for the GreedyReg_CLI and ALI_CBCT CLI
  modules, parsing ALI landmark output, and small geometry helpers. The
  actual Greedy registration and ALI landmark detection run out-of-process
  through slicer.cli.run; this class never calls them with a blocking
  subprocess itself."""

  # Landmark sets per region, used by ALI-based distant registration
  REGION_CONFIG = {
    "MANDMASK": {
      # RGo, LGo in Lower_Bones_1; Gn, Me, Pog in Lower_Bones_2
      "landmarks": ["RGo", "LGo", "Gn", "Me", "Pog"],
      "model_dirs": ["Lower_Bones_1", "Lower_Bones_2"],
    },
    "MAXMASK": {
      "landmarks": ["A", "ANS", "LOr", "ROr", "PNS"],
      "model_dirs": ["Upper_Bones_v2"],
    },
    "CBMASK": {
      "landmarks": ["S", "N", "RPo", "LPo"],
      "model_dirs": ["Cranial_Base"],
    },
  }

  # Same release the ALI module's own "Download latest models" button uses
  ALI_MODEL_DOWNLOAD_BASE_URL = (
    "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/"
    "releases/download/v0.1-v2.0_models/")

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)

  # ------------------------------------------------------------------ #
  #  Greedy binary (lives alongside GreedyReg_CLI, which is the module
  #  that actually invokes it)
  # ------------------------------------------------------------------ #

  def _platformBinDir(self):
    import platform
    system = platform.system()
    if system == "Linux":
      return "linux", "greedy"
    elif system == "Darwin":
      return "mac", "greedy"
    elif system == "Windows":
      return "windows", "greedy.exe"
    raise RuntimeError("Unsupported platform!")

  def greedyBinaryPath(self):
    platformDir, binaryName = self._platformBinDir()
    try:
      cliModuleDir = os.path.dirname(slicer.modules.greedyreg_cli.path)
    except AttributeError:
      cliModuleDir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "GreedyReg_CLI")
    return os.path.join(cliModuleDir, "bin", platformDir, binaryName)

  def isGreedyAvailable(self):
    path = self.greedyBinaryPath()
    return bool(path) and os.path.exists(path)

  # ------------------------------------------------------------------ #
  #  Python dependencies
  #
  #  GreedyReg_CLI.py runs as "python-real", i.e. Slicer's own bundled
  #  Python interpreter, so a package installed here via pip_install is
  #  immediately importable from the CLI subprocess too - no separate
  #  environment to manage.
  # ------------------------------------------------------------------ #

  def ensureNibabelInstalled(self):
    """Used to binarize masks and bake landmark-based affines into NIfTI
    files. Not bundled with Slicer by default. Prompts the user once and
    pip-installs it into Slicer's Python if missing. Returns True if
    nibabel is importable by the time this returns."""
    try:
      import nibabel  # noqa: F401
      return True
    except ImportError:
      pass

    if not slicer.util.confirmYesNoDisplay(
        "GreedyReg requires the 'nibabel' Python package (used to read/write "
        "NIfTI masks and transforms) which is not installed in Slicer's Python "
        "environment.\n\nInstall it now?"):
      return False

    slicer.util.pip_install("nibabel")
    try:
      import nibabel  # noqa: F401
      return True
    except ImportError:
      return False

  def downloadGreedyBinary(self, statusCallback=None):
    """Download and extract the Greedy binary for the current platform
    into GreedyReg_CLI's bin folder. statusCallback, if given, is called
    with progress strings. Returns the path to the extracted binary;
    raises on failure or unsupported platform."""
    import platform

    def report(text):
      if statusCallback:
        statusCallback(text)

    system = platform.system()
    if system == "Windows":
      return self._downloadGreedyBinaryWindows(report)
    elif system == "Linux":
      url = "https://sourceforge.net/projects/itk-snap/files/itk-snap/4.2.2/itksnap-4.2.2-20241202-Linux-x86_64.tar.gz/download"
      binaryInArchive = "itksnap-4.2.2-20241202-Linux-x86_64/bin/greedy"
    elif system == "Darwin":
      url = "https://sourceforge.net/projects/itk-snap/files/itk-snap/4.2.2/itksnap-4.2.2-20241202-MacOS-arm64.tar.gz/download"
      binaryInArchive = "itksnap-4.2.2-20241202-MacOS-arm64/bin/greedy"
    else:
      raise RuntimeError("Unsupported platform!")

    import urllib.request, tarfile, tempfile

    destBinary = self.greedyBinaryPath()
    destDir = os.path.dirname(destBinary)

    tmpFile = tempfile.mktemp(suffix=".tar.gz")
    report("Downloading Greedy (~60MB)...")
    urllib.request.urlretrieve(url, tmpFile)
    report("Extracting...")
    os.makedirs(destDir, exist_ok=True)
    with tarfile.open(tmpFile, "r:gz") as tar:
      member = tar.getmember(binaryInArchive)
      member.name = os.path.basename(member.name)
      tar.extract(member, destDir)
    os.chmod(destBinary, 0o755)
    os.remove(tmpFile)
    return destBinary

  def _find7zExecutable(self):
    """Locate a system 7-Zip install, if any. Used to pull greedy.exe
    straight out of the ITK-SNAP NSIS installer without running it."""
    candidate = shutil.which("7z") or shutil.which("7z.exe")
    if candidate:
      return candidate
    for envVar in ("ProgramFiles", "ProgramFiles(x86)"):
      base = os.environ.get(envVar)
      if base:
        path = os.path.join(base, "7-Zip", "7z.exe")
        if os.path.exists(path):
          return path
    return None

  def _downloadGreedyBinaryWindows(self, report):
    """ITK-SNAP only ships Windows builds as an NSIS installer (no portable
    archive like Linux/Mac), so getting greedy.exe out of it needs an extra
    step. Two strategies, in order of preference:

    1. If 7-Zip is installed, open the installer as an archive and pull
       greedy.exe out directly - no install, no admin rights, no leftovers.
    2. Otherwise, run the installer silently into a throwaway, space-free
       directory (NSIS's /D= switch cannot be quoted, so a path containing
       spaces would be truncated at the first space), copy greedy.exe out,
       then silently uninstall and delete the directory.

    Raises RuntimeError with an actionable message if both fail.
    """
    import urllib.request, tempfile, subprocess

    url = "https://sourceforge.net/projects/itk-snap/files/itk-snap/4.2.2/itksnap-4.2.2-20241202-win64-AMD64.exe/download"
    destBinary = self.greedyBinaryPath()
    destDir = os.path.dirname(destBinary)
    os.makedirs(destDir, exist_ok=True)

    installerPath = tempfile.mktemp(suffix=".exe")
    report("Downloading ITK-SNAP installer (~150MB)...")
    urllib.request.urlretrieve(url, installerPath)

    try:
      sevenZip = self._find7zExecutable()
      if sevenZip:
        report("Extracting greedy.exe with 7-Zip...")
        extractDir = tempfile.mkdtemp(prefix="greedyreg_7z_")
        try:
          result = subprocess.run(
            [sevenZip, "x", installerPath, f"-o{extractDir}", "-y"],
            capture_output=True, text=True, timeout=300)
          if result.returncode == 0:
            found = self._findFileRecursive(extractDir, "greedy.exe")
            if found:
              shutil.copy2(found, destBinary)
              return destBinary
          report("7-Zip extraction did not contain greedy.exe, falling back to silent install...")
        finally:
          shutil.rmtree(extractDir, ignore_errors=True)

      return self._installGreedyBinaryWindowsViaNsis(installerPath, destBinary, report)
    finally:
      if os.path.exists(installerPath):
        os.remove(installerPath)

  def _installGreedyBinaryWindowsViaNsis(self, installerPath, destBinary, report):
    import subprocess

    # NSIS's /D=dir switch cannot be quoted, so a path containing spaces
    # gets truncated at the first space. Pick a short, space-free
    # directory off the system drive instead of reusing destDir (which may
    # live under a path with spaces or non-ASCII characters).
    systemDrive = os.environ.get("SystemDrive", "C:")
    installDir = os.path.join(systemDrive + "\\", "_greedyreg_nsis_tmp")
    if os.path.exists(installDir):
      shutil.rmtree(installDir, ignore_errors=True)
    try:
      os.makedirs(installDir, exist_ok=True)
    except OSError:
      import tempfile
      installDir = os.path.join(tempfile.gettempdir(), "_greedyreg_nsis_tmp")
      if os.path.exists(installDir):
        shutil.rmtree(installDir, ignore_errors=True)
      os.makedirs(installDir, exist_ok=True)
    if " " in installDir:
      raise RuntimeError(
        f"Could not find a space-free directory to silently install ITK-SNAP into "
        f"(tried '{installDir}'). Install 7-Zip, or install ITK-SNAP manually and "
        f"copy its bin\\greedy.exe to: {destBinary}")

    report("Installing ITK-SNAP silently (this can take a minute)...")
    try:
      result = subprocess.run(
        [installerPath, "/S", f"/D={installDir}"],
        capture_output=True, text=True, timeout=300)
      if result.returncode != 0:
        raise RuntimeError(
          f"Silent ITK-SNAP install failed (exit code {result.returncode}). "
          f"Install ITK-SNAP manually and copy its bin\\greedy.exe to: {destBinary}")

      report("Locating greedy.exe...")
      found = self._findFileRecursive(installDir, "greedy.exe")
      if not found:
        raise RuntimeError(
          f"greedy.exe not found inside the ITK-SNAP install. "
          f"Install ITK-SNAP manually and copy its bin\\greedy.exe to: {destBinary}")
      shutil.copy2(found, destBinary)
      return destBinary
    finally:
      uninstaller = os.path.join(installDir, "Uninstall.exe")
      if os.path.exists(uninstaller):
        try:
          subprocess.run([uninstaller, "/S", f"_?={installDir}"],
                         capture_output=True, timeout=60)
        except Exception:
          pass
      shutil.rmtree(installDir, ignore_errors=True)

  def _findFileRecursive(self, rootDir, fileName):
    for root, _dirs, files in os.walk(rootDir):
      if fileName in files:
        return os.path.join(root, fileName)
    return None

  # ------------------------------------------------------------------ #
  #  GreedyReg_CLI parameters
  # ------------------------------------------------------------------ #

  def exportMask(self, maskNode, maskPath):
    """Export a mask/segmentation MRML node to a NIfTI file. Binarizing
    for Greedy's -gm mask argument is handled by GreedyReg_CLI."""
    slicer.util.exportNode(maskNode, maskPath)

  def writeInitTransform(self, initPath, matrix):
    """Write a vtkMatrix4x4 to a Greedy-format .mat init file, nudging a
    zero translation slightly so Greedy doesn't treat it as identity."""
    if matrix.GetElement(0, 3) == 0 and matrix.GetElement(1, 3) == 0 and matrix.GetElement(2, 3) == 0:
      matrix.SetElement(0, 3, 0.001)
    with open(initPath, 'w') as f:
      for i in range(4):
        f.write(' '.join([str(matrix.GetElement(i, j)) for j in range(4)]) + '\n')

  def buildGreedyCliParameters(self, t1Folder, t2Folder, outputFolder,
                                metricIndex, dofIndex, maskFolder=None, initFolder=None):
    metric = ["NMI", "NCC", "SSD"][metricIndex]
    transformType = "Rigid" if dofIndex == 0 else "Affine"
    return {
      "t1Folder": t1Folder,
      "t2Folder": t2Folder,
      "maskFolder": maskFolder or "",
      "initFolder": initFolder or "",
      "outputFolder": outputFolder,
      "greedyBinary": self.greedyBinaryPath(),
      "metric": metric,
      "transformType": transformType,
    }

  def runGreedyCli(self, parameters):
    return slicer.cli.run(slicer.modules.greedyreg_cli, None, parameters)

  def findBatchPairs(self, t1Folder, t2Folder, maskFolder=None):
    """Preview the pairs GreedyReg_CLI would find, for the 'Found N pairs'
    label. Matching logic must stay consistent with GreedyReg_CLI.py."""
    import re
    id_pattern = re.compile(r'^([A-Za-z]+\d+)', re.IGNORECASE)

    def getNiftiFiles(folder):
      ids = {}
      if not folder or not os.path.exists(folder):
        return ids
      for fname in os.listdir(folder):
        if fname.endswith('.nii.gz') or fname.endswith('.nii'):
          m = id_pattern.match(fname)
          if m:
            ids[m.group(1).upper()] = os.path.join(folder, fname)
      return ids

    t1s = getNiftiFiles(t1Folder)
    t2s = getNiftiFiles(t2Folder)
    masks = getNiftiFiles(maskFolder) if maskFolder else {}

    pairs = []
    for patientId in sorted(set(t1s.keys()) & set(t2s.keys())):
      pairs.append((patientId, t1s[patientId], t2s[patientId], masks.get(patientId)))
    return pairs

  # ------------------------------------------------------------------ #
  #  ALI_CBCT Python dependencies (torch is assumed already present via
  #  this extension's own NNUNet/PyTorch dependency chain - matches what
  #  the ALI module itself checks before predicting).
  # ------------------------------------------------------------------ #

  def _aliRequiredLibs(self):
    monaiVersion = '1.3.2' if sys.version_info >= (3, 10) else '0.7.0'
    return [('itk', None), ('dicom2nifti', '2.6.2'), ('pydicom', '3.0.2'), ('monai', monaiVersion)]

  def _checkLibInstalled(self, libName, requiredVersion=None):
    import importlib.metadata
    try:
      installedVersion = importlib.metadata.version(libName)
      if requiredVersion and installedVersion != requiredVersion:
        return False
      return True
    except importlib.metadata.PackageNotFoundError:
      return False

  def aliLibrariesReady(self):
    """Quick, non-installing check for the Python libraries ALI_CBCT.py
    needs."""
    return all(self._checkLibInstalled(lib, version) for lib, version in self._aliRequiredLibs())

  def ensureAliLibrariesInstalled(self):
    """Prompt-and-install missing/mismatched libraries ALI_CBCT.py needs
    to run landmark detection for Distant Registration. Mirrors the ALI
    module's own install_function. Returns True once all required
    libraries are present (including when nothing needed installing)."""
    libsToInstall = [(lib, version) for lib, version in self._aliRequiredLibs()
                      if not self._checkLibInstalled(lib, version)]
    if not libsToInstall:
      return True

    message = "The following libraries are required for ALI-based Distant Registration:\n"
    message += "\n".join(f"{lib}=={version}" if version else lib for lib, version in libsToInstall)
    message += "\n\nInstall/update them now? Doing so could affect other modules."
    if not slicer.util.confirmYesNoDisplay(message):
      return False

    for lib, version in libsToInstall:
      libSpec = f"{lib}=={version}" if version else lib
      slicer.util.pip_install(libSpec)

    return all(self._checkLibInstalled(lib, version) for lib, version in self._aliRequiredLibs())

  # ------------------------------------------------------------------ #
  #  Manual alignment helpers
  # ------------------------------------------------------------------ #

  def computeCenteringTranslation(self, fixed, moving):
    """Return (tx, ty, tz) in RAS mm that centers moving's image center
    on fixed's image center."""
    fixedDims = fixed.GetImageData().GetDimensions()
    fixedIJKCenter = [fixedDims[0] / 2, fixedDims[1] / 2, fixedDims[2] / 2, 1]
    fixedIJKToRAS = vtk.vtkMatrix4x4()
    fixed.GetIJKToRASMatrix(fixedIJKToRAS)
    fixedRASCenter = [0, 0, 0, 1]
    fixedIJKToRAS.MultiplyPoint(fixedIJKCenter, fixedRASCenter)

    movingDims = moving.GetImageData().GetDimensions()
    movingIJKCenter = [movingDims[0] / 2, movingDims[1] / 2, movingDims[2] / 2, 1]
    movingIJKToRAS = vtk.vtkMatrix4x4()
    moving.GetIJKToRASMatrix(movingIJKToRAS)
    movingRASCenter = [0, 0, 0, 1]
    movingIJKToRAS.MultiplyPoint(movingIJKCenter, movingRASCenter)

    tx = fixedRASCenter[0] - movingRASCenter[0]
    ty = fixedRASCenter[1] - movingRASCenter[1]
    tz = fixedRASCenter[2] - movingRASCenter[2]
    return tx, ty, tz

  # ------------------------------------------------------------------ #
  #  Distant registration (ALI landmark-based, via slicer.modules.ali_cbct)
  # ------------------------------------------------------------------ #

  def defaultAliModelsDir(self):
    import qt
    documents = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
    return os.path.join(documents, slicer.app.applicationName + "Downloads", "GreedyReg", "ALIModels")

  def _allAliModelDirs(self):
    return sorted({d for cfg in self.REGION_CONFIG.values() for d in cfg["model_dirs"]})

  def aliModelsReady(self, aliModelsDir, regions=None):
    """True if every model subdirectory needed by the given regions (or
    all regions if None) already exists and is non-empty under
    aliModelsDir."""
    if regions:
      dirNames = sorted({d for r in regions for d in self.REGION_CONFIG[r]["model_dirs"]})
    else:
      dirNames = self._allAliModelDirs()
    return all(
      os.path.isdir(os.path.join(aliModelsDir, d)) and os.listdir(os.path.join(aliModelsDir, d))
      for d in dirNames)

  def downloadAliModels(self, aliModelsDir, regions=None, statusCallback=None):
    """Download and extract the ALI landmark-detection models Distant
    Registration needs (the same release the ALI module's own "Download
    latest models" button uses) into aliModelsDir/<model_dir>/, e.g.
    aliModelsDir/Lower_Bones_1/. Skips any model_dir that's already
    present. Prompts for confirmation before downloading. Returns
    aliModelsDir; raises on failure or if the user declines."""
    import urllib.request, zipfile, tempfile

    def report(text):
      if statusCallback:
        statusCallback(text)

    if regions:
      dirNames = sorted({d for r in regions for d in self.REGION_CONFIG[r]["model_dirs"]})
    else:
      dirNames = self._allAliModelDirs()

    missing = [
      d for d in dirNames
      if not (os.path.isdir(os.path.join(aliModelsDir, d)) and os.listdir(os.path.join(aliModelsDir, d)))]
    if not missing:
      return aliModelsDir

    if not slicer.util.confirmYesNoDisplay(
        "The following ALI landmark-detection models used by Distant Registration "
        "are missing:\n" + "\n".join(missing) +
        f"\n\nDownload them now into:\n{aliModelsDir}\n(this can take a while)?"):
      raise RuntimeError("ALI model download cancelled by user")

    os.makedirs(aliModelsDir, exist_ok=True)
    for i, dirName in enumerate(missing):
      destDir = os.path.join(aliModelsDir, dirName)
      url = f"{self.ALI_MODEL_DOWNLOAD_BASE_URL}{dirName}.zip"
      report(f"Downloading {dirName} ({i + 1}/{len(missing)})...")
      tmpZip = tempfile.mktemp(suffix=".zip")
      urllib.request.urlretrieve(url, tmpZip)
      report(f"Extracting {dirName}...")
      os.makedirs(destDir, exist_ok=True)
      with zipfile.ZipFile(tmpZip, "r") as zf:
        zf.extractall(destDir)
      os.remove(tmpZip)
    return aliModelsDir

  def buildAliParameters(self, scanPath, modelSubDir, aliModelDir, landmarks, outputDir, tmpDir):
    modelPath = os.path.join(aliModelDir, modelSubDir)
    if not os.path.exists(modelPath):
      raise RuntimeError(f"ALI model folder not found: {modelPath}")
    os.makedirs(outputDir, exist_ok=True)
    subTmp = os.path.join(tmpDir, f"ali_tmp_{modelSubDir}")
    os.makedirs(subTmp, exist_ok=True)
    lm_str = ",".join(f"\"{lm}\"" for lm in landmarks)
    return {
      "input": scanPath,
      "dir_models": modelPath,
      "lm_type": lm_str,
      "output_dir": outputDir,
      "temp_fold": subTmp,
      "DCMInput": "false",
      "spacing": "[1,0.3]",
      "speed_per_scale": "[1,1]",
      "agent_FOV": "[64,64,64]",
      "spawn_radius": "10",
    }

  def runAliCli(self, parameters):
    return slicer.cli.run(slicer.modules.ali_cbct, None, parameters)

  def buildAliJobQueue(self, scans, aliModelDir, region, tmpDir):
    """scans: dict like {"fixed": scanPath, "moving": scanPath}.
    Returns a list of job dicts, one per (scan, model subdir) combination,
    each carrying the slicer.cli parameters needed to run ALI_CBCT and the
    output dir to parse its landmark JSON from afterwards."""
    cfg = self.REGION_CONFIG[region]
    landmarks = cfg["landmarks"]
    jobs = []
    for scanKey, scanPath in scans.items():
      outputDir = os.path.join(tmpDir, f"ali_{scanKey}")
      for subdir in cfg["model_dirs"]:
        subOutputDir = os.path.join(outputDir, subdir)
        jobs.append({
          "scanKey": scanKey,
          "subdir": subdir,
          "outputDir": subOutputDir,
          "landmarks": landmarks,
          "parameters": self.buildAliParameters(
            scanPath, subdir, aliModelDir, landmarks, subOutputDir, tmpDir),
        })
    return jobs

  def parseAliLandmarksFromOutput(self, outputDir, landmarks):
    """Parse ALI_CBCT's output markups JSON. Coordinates are in LPS;
    converted to RAS (flip X and Y)."""
    found = {}
    if not os.path.isdir(outputDir):
      return found
    for fname in os.listdir(outputDir):
      if not fname.endswith(".json"):
        continue
      with open(os.path.join(outputDir, fname)) as f:
        data = json.load(f)
      for cp in data.get("markups", [{}])[0].get("controlPoints", []):
        name = cp.get("label", "")
        pos = cp.get("position", [0, 0, 0])
        if name in landmarks:
          found[name] = [-pos[0], -pos[1], pos[2]]
    return found

  def rigidFromLandmarks(self, fixedPts, movingPts):
    """Compute rigid 4x4 RAS transform from matched landmark arrays using
    SVD. fixedPts, movingPts: Nx3 numpy arrays of corresponding points."""
    import numpy as np
    fc = fixedPts.mean(axis=0)
    mc = movingPts.mean(axis=0)
    fC = fixedPts - fc
    mC = movingPts - mc
    H = mC.T @ fC
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
      Vt[-1, :] *= -1
      R = Vt.T @ U.T
    t = fc - R @ mc
    mat4 = np.eye(4)
    mat4[:3, :3] = R
    mat4[:3, 3] = t
    return mat4

  # ------------------------------------------------------------------ #
  #  Batch pairing (used by Distant Registration batch, which still
  #  drives one ALI job-queue per pair from the widget)
  # ------------------------------------------------------------------ #

  def findBatchPairsDistant(self, t1Folder, t2Folder):
    return [(pid, t1, t2) for pid, t1, t2, _mask in self.findBatchPairs(t1Folder, t2Folder)]
