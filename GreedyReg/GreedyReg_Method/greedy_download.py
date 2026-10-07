"""Standalone Greedy downloader, run by GreedyReg as a separate PythonSlicer
process (see GreedyRegLogic.startGreedyDownload) so a slow network, a hung
installer or a crash here can never freeze or take down Slicer itself.

Usage: PythonSlicer greedy_download.py <destBinaryPath>

Standard library only - it must not import slicer, qt or vtk. Reports back to
the GUI with one message per stdout line:
  STATUS: <text>       human-readable step
  PROGRESS: <0-100>    download percentage
  ERROR: <text>        fatal error (process then exits non-zero)
"""

import os
import sys
import shutil
import platform
import tempfile
import subprocess
import urllib.request

ITKSNAP_BASE_URL = "https://sourceforge.net/projects/itk-snap/files/itk-snap/4.2.2/"
ITKSNAP_BUILD = "itksnap-4.2.2-20241202"


def status(text):
  print(f"STATUS: {text}", flush=True)


def progress(percent):
  print(f"PROGRESS: {int(percent)}", flush=True)


def downloadFile(url, destPath, label):
  """Stream url to destPath, reporting percentage when the size is known."""
  status(f"Downloading {label}...")
  request = urllib.request.Request(url, headers={"User-Agent": "Wget/1.21"})
  with urllib.request.urlopen(request, timeout=60) as response, open(destPath, "wb") as out:
    total = int(response.headers.get("Content-Length") or 0)
    done = 0
    lastPercent = -1
    while True:
      chunk = response.read(1024 * 1024)
      if not chunk:
        break
      out.write(chunk)
      done += len(chunk)
      if total:
        percent = done * 100 // total
        if percent != lastPercent:
          progress(percent)
          lastPercent = percent
  if total and done < total:
    raise RuntimeError(f"Download of {label} was cut off ({done} of {total} bytes)")


def findFileRecursive(rootDir, fileName):
  for root, _dirs, files in os.walk(rootDir):
    if fileName in files:
      return os.path.join(root, fileName)
  return None


# ---------------------------------------------------------------------- #
#  Per-platform extraction. Each writes the binary to partialPath.
# ---------------------------------------------------------------------- #

def fetchLinux(workDir, partialPath):
  import tarfile
  archiveName = f"{ITKSNAP_BUILD}-Linux-x86_64.tar.gz"
  archivePath = os.path.join(workDir, archiveName)
  downloadFile(ITKSNAP_BASE_URL + archiveName + "/download", archivePath, "ITK-SNAP (~200MB)")
  status("Extracting greedy...")
  memberName = f"{ITKSNAP_BUILD}-Linux-x86_64/bin/greedy"
  with tarfile.open(archivePath, "r:gz") as tar:
    source = tar.extractfile(tar.getmember(memberName))
    with open(partialPath, "wb") as out:
      shutil.copyfileobj(source, out)


def fetchMac(workDir, partialPath):
  # ITK-SNAP ships macOS builds only as .dmg disk images, one per CPU type.
  arch = "arm64" if platform.machine() == "arm64" else "x86_64"
  imageName = f"{ITKSNAP_BUILD}-Darwin-{arch}.dmg"
  imagePath = os.path.join(workDir, imageName)
  downloadFile(ITKSNAP_BASE_URL + imageName + "/download", imagePath, f"ITK-SNAP for {arch}")
  status("Extracting greedy...")
  mountPoint = os.path.join(workDir, "mnt")
  os.makedirs(mountPoint)
  # "Y" accepts a license agreement if the image carries one.
  result = subprocess.run(
    ["hdiutil", "attach", imagePath, "-nobrowse", "-readonly", "-noautoopen",
     "-mountpoint", mountPoint],
    input="Y\n", capture_output=True, text=True, timeout=300)
  if result.returncode != 0:
    raise RuntimeError(f"Could not open the ITK-SNAP disk image: {result.stderr.strip()}")
  try:
    found = findFileRecursive(mountPoint, "greedy")
    if not found:
      raise RuntimeError("greedy was not found inside the ITK-SNAP disk image")
    shutil.copyfile(found, partialPath)
  finally:
    subprocess.run(["hdiutil", "detach", mountPoint, "-force"], capture_output=True, timeout=120)


def find7zExecutable():
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


def fetchWindows(workDir, partialPath):
  """ITK-SNAP only ships Windows builds as an NSIS installer. If 7-Zip is
  installed, pull greedy.exe straight out of it; otherwise run the
  installer silently into a throwaway, space-free directory (NSIS's /D=
  switch cannot be quoted), copy greedy.exe out, then uninstall."""
  installerName = f"{ITKSNAP_BUILD}-win64-AMD64.exe"
  installerPath = os.path.join(workDir, installerName)
  downloadFile(ITKSNAP_BASE_URL + installerName + "/download", installerPath, "ITK-SNAP installer (~150MB)")

  sevenZip = find7zExecutable()
  if sevenZip:
    status("Extracting greedy.exe with 7-Zip...")
    extractDir = os.path.join(workDir, "7z")
    result = subprocess.run(
      [sevenZip, "x", installerPath, f"-o{extractDir}", "-y"],
      capture_output=True, text=True, timeout=300)
    found = findFileRecursive(extractDir, "greedy.exe") if result.returncode == 0 else None
    if found:
      shutil.copyfile(found, partialPath)
      return
    status("7-Zip could not extract greedy.exe, falling back to a silent install...")

  systemDrive = os.environ.get("SystemDrive", "C:")
  installDir = os.path.join(systemDrive + "\\", "_greedyreg_nsis_tmp")
  shutil.rmtree(installDir, ignore_errors=True)
  try:
    os.makedirs(installDir, exist_ok=True)
  except OSError:
    installDir = os.path.join(tempfile.gettempdir(), "_greedyreg_nsis_tmp")
    shutil.rmtree(installDir, ignore_errors=True)
    os.makedirs(installDir, exist_ok=True)
  if " " in installDir:
    raise RuntimeError(
      f"No space-free folder available for a silent ITK-SNAP install (tried '{installDir}'). "
      "Install 7-Zip and retry, or install ITK-SNAP manually and copy its bin\\greedy.exe.")

  status("Installing ITK-SNAP silently (this can take a minute)...")
  try:
    try:
      result = subprocess.run(
        [installerPath, "/S", f"/D={installDir}"],
        capture_output=True, text=True, timeout=600)
    except OSError as e:
      # e.g. WinError 740: the installer requires administrator rights
      raise RuntimeError(
        f"Could not run the ITK-SNAP installer ({e}). Install 7-Zip and retry, "
        "or install ITK-SNAP manually and copy its bin\\greedy.exe.")
    if result.returncode != 0:
      raise RuntimeError(
        f"Silent ITK-SNAP install failed (exit code {result.returncode}). Install 7-Zip "
        "and retry, or install ITK-SNAP manually and copy its bin\\greedy.exe.")
    found = findFileRecursive(installDir, "greedy.exe")
    if not found:
      raise RuntimeError("greedy.exe was not found inside the ITK-SNAP install")
    shutil.copyfile(found, partialPath)
  finally:
    uninstaller = os.path.join(installDir, "Uninstall.exe")
    if os.path.exists(uninstaller):
      try:
        subprocess.run([uninstaller, "/S", f"_?={installDir}"], capture_output=True, timeout=120)
      except Exception as e:
        # Best effort: the install directory is removed outright just below,
        # so a failed uninstaller is not fatal -- but say so rather than
        # letting it vanish.
        print(f"ITK-SNAP uninstaller failed ({e}); removing its directory directly", flush=True)
    shutil.rmtree(installDir, ignore_errors=True)


def main(destBinary):
  system = platform.system()
  fetchers = {"Linux": fetchLinux, "Darwin": fetchMac, "Windows": fetchWindows}
  if system not in fetchers:
    raise RuntimeError(f"Unsupported platform: {system}")

  destDir = os.path.dirname(destBinary)
  os.makedirs(destDir, exist_ok=True)
  # Build the binary under a temporary name and only move it into place once
  # it has been verified to run, so an interrupted or broken download never
  # leaves a file behind that makes GreedyReg think Greedy is installed.
  partialPath = destBinary + ".part"
  workDir = tempfile.mkdtemp(prefix="greedyreg_download_")
  try:
    fetchers[system](workDir, partialPath)
    if system != "Windows":
      os.chmod(partialPath, 0o755)
    status("Checking that greedy runs...")
    result = subprocess.run([partialPath, "-version"], capture_output=True, text=True, timeout=60)
    if "Greedy" not in (result.stdout + result.stderr):
      raise RuntimeError(
        "The downloaded greedy binary does not run on this computer: "
        + (result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"))
    os.replace(partialPath, destBinary)
  finally:
    shutil.rmtree(workDir, ignore_errors=True)
    if os.path.exists(partialPath):
      os.remove(partialPath)
  status("Greedy installed")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("ERROR: usage: greedy_download.py <destBinaryPath>", flush=True)
    sys.exit(2)
  try:
    main(sys.argv[1])
  except Exception as e:
    print(f"ERROR: {e}", flush=True)
    sys.exit(1)
