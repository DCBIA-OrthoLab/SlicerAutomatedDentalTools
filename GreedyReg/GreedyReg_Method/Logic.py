import os
import sys
import json
import shutil

import vtk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
import platform
import re


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
    system = platform.system()
    if system == "Linux":
      return "linux", "greedy"
    elif system == "Darwin":
      return "mac", "greedy"
    elif system == "Windows":
      return "windows", "greedy.exe"
    raise RuntimeError("Unsupported platform!")

  def greedyBinaryPath(self):
    platform_dir, binary_name = self._platformBinDir()
    try:
      cli_module_dir = os.path.dirname(slicer.modules.greedyreg_cli.path)
    except AttributeError:
      cli_module_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "GreedyReg_CLI")
    return os.path.join(cli_module_dir, "bin", platform_dir, binary_name)

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

  def startGreedyDownload(self, onStatus, onProgress, onFinished):
    """Download Greedy into GreedyReg_CLI's bin folder in a separate
    PythonSlicer process (greedy_download.py), so a slow network, a hung
    installer or a crash can't freeze or take down Slicer. Returns
    immediately; the callbacks run on the GUI thread:
      onStatus(text), onProgress(percent 0-100),
      onFinished(success, errorMessage)."""
    python_slicer = shutil.which("PythonSlicer")
    if not python_slicer:
      raise RuntimeError("PythonSlicer executable not found")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greedy_download.py")
    errors = []

    def onLine(line):
      if line.startswith("STATUS: "):
        onStatus(line[len("STATUS: "):])
      elif line.startswith("PROGRESS: "):
        try:
          onProgress(int(line[len("PROGRESS: "):]))
        except ValueError:
          pass
      elif line.startswith("ERROR: "):
        errors.append(line[len("ERROR: "):])
      else:
        print(f"[greedy download] {line}")

    def onCompleted(return_code):
      success = return_code == 0 and self.isGreedyAvailable()
      message = "" if success else (errors[-1] if errors else f"download process exited with code {return_code}")
      onFinished(success, message)

    return slicer.util.launchConsoleProcess(
      [python_slicer, script, self.greedyBinaryPath()],
      useStartupEnvironment=False, blocking=False,
      logCallback=onLine, completedCallback=onCompleted)

  # ------------------------------------------------------------------ #
  #  GreedyReg_CLI parameters
  # ------------------------------------------------------------------ #

  def exportMask(self, mask_node, mask_path):
    """Export a mask/segmentation MRML node to a NIfTI file. Binarizing
    for Greedy's -gm mask argument is handled by GreedyReg_CLI."""
    slicer.util.exportNode(mask_node, mask_path)

  def writeInitTransform(self, init_path, matrix):
    """Write a vtkMatrix4x4 to a Greedy-format .mat init file, nudging a
    zero translation slightly so Greedy doesn't treat it as identity."""
    if matrix.GetElement(0, 3) == 0 and matrix.GetElement(1, 3) == 0 and matrix.GetElement(2, 3) == 0:
      matrix.SetElement(0, 3, 0.001)
    with open(init_path, 'w') as f:
      for i in range(4):
        f.write(' '.join([str(matrix.GetElement(i, j)) for j in range(4)]) + '\n')

  def buildGreedyCliParameters(self, t1_folder, t2_folder, output_folder,
                                metric_index, dof_index, maskFolder=None, initFolder=None):
    metric = ["NMI", "NCC", "SSD"][metric_index]
    transform_type = "Rigid" if dof_index == 0 else "Affine"
    return {
      "t1Folder": t1_folder,
      "t2Folder": t2_folder,
      "maskFolder": maskFolder or "",
      "initFolder": initFolder or "",
      "outputFolder": output_folder,
      "greedyBinary": self.greedyBinaryPath(),
      "metric": metric,
      "transformType": transform_type,
    }

  def runGreedyCli(self, parameters):
    return slicer.cli.run(slicer.modules.greedyreg_cli, None, parameters)

  def findBatchPairs(self, t1_folder, t2_folder, maskFolder=None):
    """Preview the pairs GreedyReg_CLI would find, for the 'Found N pairs'
    label. Matching logic must stay consistent with GreedyReg_CLI.py."""
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

    t1s = getNiftiFiles(t1_folder)
    t2s = getNiftiFiles(t2_folder)
    masks = getNiftiFiles(maskFolder) if maskFolder else {}

    pairs = []
    for patient_id in sorted(set(t1s.keys()) & set(t2s.keys())):
      pairs.append((patient_id, t1s[patient_id], t2s[patient_id], masks.get(patient_id)))
    return pairs

  # ------------------------------------------------------------------ #
  #  ALI_CBCT Python dependencies (torch is assumed already present via
  #  this extension's own NNUNet/PyTorch dependency chain - matches what
  #  the ALI module itself checks before predicting).
  # ------------------------------------------------------------------ #

  def _aliRequiredLibs(self):
    monai_version = '1.3.2' if sys.version_info >= (3, 10) else '0.7.0'
    return [('itk', None), ('dicom2nifti', '2.6.2'), ('pydicom', '3.0.2'), ('monai', monai_version)]

  def _checkLibInstalled(self, lib_name, required_version=None):
    """Delegated to ADTLib, like every other module's copy.

    This one was a tenth copy, out of reach of the sweep that replaced the
    others because it sits in the method package rather than at the module
    root. It compared bare strings, so `2.2.0+cu118` read as "wrong version"
    and a constraint such as `>=2.6.2` could never be satisfied at all.
    """
    from ADTLib.env.deps import check_lib_installed
    return check_lib_installed(lib_name, required_version)

  def aliLibrariesReady(self):
    """Quick, non-installing check for the Python libraries ALI_CBCT.py
    needs."""
    return all(self._checkLibInstalled(lib, version) for lib, version in self._aliRequiredLibs())

  def ensureAliLibrariesInstalled(self):
    """Prompt-and-install missing/mismatched libraries ALI_CBCT.py needs
    to run landmark detection for Distant Registration. Mirrors the ALI
    module's own install_function. Returns True once all required
    libraries are present (including when nothing needed installing)."""
    libs_to_install = [(lib, version) for lib, version in self._aliRequiredLibs()
                      if not self._checkLibInstalled(lib, version)]
    if not libs_to_install:
      return True

    from ADTLib.env.deps import requirement
    message = "The following libraries are required for ALI-based Distant Registration:\n"
    message += "\n".join(requirement(lib, version) for lib, version in libs_to_install)
    message += "\n\nInstall/update them now? Doing so could affect other modules."
    if not slicer.util.confirmYesNoDisplay(message):
      return False

    for lib, version in libs_to_install:
      slicer.util.pip_install(requirement(lib, version))

    return all(self._checkLibInstalled(lib, version) for lib, version in self._aliRequiredLibs())

  # ------------------------------------------------------------------ #
  #  Manual alignment helpers
  # ------------------------------------------------------------------ #

  def computeCenteringTranslation(self, fixed, moving):
    """Return (tx, ty, tz) in RAS mm that centers moving's image center
    on fixed's image center."""
    fixed_dims = fixed.GetImageData().GetDimensions()
    fixed_ijk_center = [fixed_dims[0] / 2, fixed_dims[1] / 2, fixed_dims[2] / 2, 1]
    fixed_ijk_to_ras = vtk.vtkMatrix4x4()
    fixed.GetIJKToRASMatrix(fixed_ijk_to_ras)
    fixed_ras_center = [0, 0, 0, 1]
    fixed_ijk_to_ras.MultiplyPoint(fixed_ijk_center, fixed_ras_center)

    moving_dims = moving.GetImageData().GetDimensions()
    moving_ijk_center = [moving_dims[0] / 2, moving_dims[1] / 2, moving_dims[2] / 2, 1]
    moving_ijk_to_ras = vtk.vtkMatrix4x4()
    moving.GetIJKToRASMatrix(moving_ijk_to_ras)
    moving_ras_center = [0, 0, 0, 1]
    moving_ijk_to_ras.MultiplyPoint(moving_ijk_center, moving_ras_center)

    tx = fixed_ras_center[0] - moving_ras_center[0]
    ty = fixed_ras_center[1] - moving_ras_center[1]
    tz = fixed_ras_center[2] - moving_ras_center[2]
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

  def aliModelsReady(self, ali_models_dir, regions=None):
    """True if every model subdirectory needed by the given regions (or
    all regions if None) already exists and is non-empty under
    aliModelsDir."""
    if regions:
      dir_names = sorted({d for r in regions for d in self.REGION_CONFIG[r]["model_dirs"]})
    else:
      dir_names = self._allAliModelDirs()
    return all(
      os.path.isdir(os.path.join(ali_models_dir, d)) and os.listdir(os.path.join(ali_models_dir, d))
      for d in dir_names)

  def downloadAliModels(self, ali_models_dir, regions=None, statusCallback=None):
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
      dir_names = sorted({d for r in regions for d in self.REGION_CONFIG[r]["model_dirs"]})
    else:
      dir_names = self._allAliModelDirs()

    missing = [
      d for d in dir_names
      if not (os.path.isdir(os.path.join(ali_models_dir, d)) and os.listdir(os.path.join(ali_models_dir, d)))]
    if not missing:
      return ali_models_dir

    if not slicer.util.confirmYesNoDisplay(
        "The following ALI landmark-detection models used by Distant Registration "
        "are missing:\n" + "\n".join(missing) +
        f"\n\nDownload them now into:\n{ali_models_dir}\n(this can take a while)?"):
      raise RuntimeError("ALI model download cancelled by user")

    os.makedirs(ali_models_dir, exist_ok=True)
    for i, dir_name in enumerate(missing):
      dest_dir = os.path.join(ali_models_dir, dir_name)
      url = f"{self.ALI_MODEL_DOWNLOAD_BASE_URL}{dir_name}.zip"
      report(f"Downloading {dir_name} ({i + 1}/{len(missing)})...")
      tmp_zip = tempfile.mktemp(suffix=".zip")
      urllib.request.urlretrieve(url, tmp_zip)
      report(f"Extracting {dir_name}...")
      os.makedirs(dest_dir, exist_ok=True)
      with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(dest_dir)
      os.remove(tmp_zip)
    return ali_models_dir

  def buildAliParameters(self, scanPath, model_sub_dir, ali_model_dir, landmarks, outputDir, tmp_dir):
    model_path = os.path.join(ali_model_dir, model_sub_dir)
    if not os.path.exists(model_path):
      raise RuntimeError(f"ALI model folder not found: {model_path}")
    os.makedirs(outputDir, exist_ok=True)
    sub_tmp = os.path.join(tmp_dir, f"ali_tmp_{model_sub_dir}")
    os.makedirs(sub_tmp, exist_ok=True)
    lm_str = ",".join(f"\"{lm}\"" for lm in landmarks)
    return {
      "input": scanPath,
      "dir_models": model_path,
      "lm_type": lm_str,
      "output_dir": outputDir,
      "temp_fold": sub_tmp,
      "DCMInput": "false",
      "spacing": "[1,0.3]",
      "speed_per_scale": "[1,1]",
      "agent_FOV": "[64,64,64]",
      "spawn_radius": "10",
    }

  def runAliCli(self, parameters):
    return slicer.cli.run(slicer.modules.ali_cbct, None, parameters)

  def buildAliJobQueue(self, scans, ali_model_dir, region, tmp_dir):
    """scans: dict like {"fixed": scanPath, "moving": scanPath}.
    Returns a list of job dicts, one per (scan, model subdir) combination,
    each carrying the slicer.cli parameters needed to run ALI_CBCT and the
    output dir to parse its landmark JSON from afterwards."""
    cfg = self.REGION_CONFIG[region]
    landmarks = cfg["landmarks"]
    jobs = []
    for scan_key, scan_path in scans.items():
      output_dir = os.path.join(tmp_dir, f"ali_{scan_key}")
      for subdir in cfg["model_dirs"]:
        sub_output_dir = os.path.join(output_dir, subdir)
        jobs.append({
          "scanKey": scan_key,
          "subdir": subdir,
          "outputDir": sub_output_dir,
          "landmarks": landmarks,
          "parameters": self.buildAliParameters(
            scan_path, subdir, ali_model_dir, landmarks, sub_output_dir, tmp_dir),
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

  def rigidFromLandmarks(self, fixed_pts, moving_pts):
    """Compute rigid 4x4 RAS transform from matched landmark arrays using
    SVD. fixedPts, movingPts: Nx3 numpy arrays of corresponding points."""
    import numpy as np
    fc = fixed_pts.mean(axis=0)
    mc = moving_pts.mean(axis=0)
    f_c = fixed_pts - fc
    m_c = moving_pts - mc
    H = m_c.T @ f_c
    U, S, vt = np.linalg.svd(H)
    R = vt.T @ U.T
    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
      vt[-1, :] *= -1
      R = vt.T @ U.T
    t = fc - R @ mc
    mat4 = np.eye(4)
    mat4[:3, :3] = R
    mat4[:3, 3] = t
    return mat4

  # ------------------------------------------------------------------ #
  #  Batch pairing (used by Distant Registration batch, which still
  #  drives one ALI job-queue per pair from the widget)
  # ------------------------------------------------------------------ #

  def findBatchPairsDistant(self, t1_folder, t2_folder):
    return [(pid, t1, t2) for pid, t1, t2, _mask in self.findBatchPairs(t1_folder, t2_folder)]
  def _centerSensitivityDemoWheel(self, pad):
    """Keep a wheel centered in its parent slice view.

    This is called both when the overlay is attached and by the zoom timer, so
    it also tracks layout/window resizing without needing a fragile Qt event
    filter. The wheel remains a child of the 2D slice view only, never the 3D
    view.
    """
    parent = pad.parent()
    if parent is None:
      return
    try:
      x = int((parent.width - pad.width) / 2)
      y = int((parent.height - pad.height) / 2)
    except Exception:
      try:
        x = int((parent.width() - pad.width()) / 2)
        y = int((parent.height() - pad.height()) / 2)
      except Exception:
        return
    pad.move(max(0, x), max(0, y))


  def _currentSliceFovMean(self, slice_name):
    try:
      lm = slicer.app.layoutManager()
      slice_widget = lm.sliceWidget(slice_name)
      slice_node = slice_widget.mrmlSliceNode()
      fov = slice_node.GetFieldOfView()
      return max(1e-3, (float(fov[0]) + float(fov[1])) / 2.0)
    except Exception:
      return None


  def _rotationAxisForSlice(self, slice_name):
    # Slicer default slice conventions:
    # Red = axial plane -> rotate around Z.
    # Yellow = sagittal plane -> rotate around X.
    # Green = coronal plane -> rotate around Y.
    if slice_name == "Yellow":
      return "X"
    if slice_name == "Green":
      return "Y"
    return "Z"


  def _translationDeltasForSlice(self, slice_name, dx_pixels, dy_pixels, mm_per_pixel):
    # Background drag translates in the visible plane of the slice. This removes
    # the axis buttons entirely while still giving access to X/Y/Z translation:
    #   Red axial:     horizontal=X, vertical=Y
    #   Yellow sagittal: horizontal=Y, vertical=Z
    #   Green coronal: horizontal=X, vertical=Z
    # Screen Y grows downward. For this demo we intentionally keep the mapping
    # cursor-following instead of anatomically inverted: drag down -> positive
    # vertical movement in the displayed slice plane.
    if slice_name == "Yellow":
      return {"X": 0.0, "Y": dx_pixels * mm_per_pixel, "Z": dy_pixels * mm_per_pixel}
    if slice_name == "Green":
      return {"X": dx_pixels * mm_per_pixel, "Y": 0.0, "Z": dy_pixels * mm_per_pixel}
    return {"X": dx_pixels * mm_per_pixel, "Y": dy_pixels * mm_per_pixel, "Z": 0.0}


