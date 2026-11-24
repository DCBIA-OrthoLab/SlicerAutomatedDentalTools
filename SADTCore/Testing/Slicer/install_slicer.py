"""Download, verify and unpack a Slicer release for the smoke test.

    python install_slicer.py --stability release|nightly --dest DIR

Prints the path of the Slicer executable. Runs with any Python 3.8+, standard
library only, on Linux, Windows and macOS: the CI job calls this one script
rather than three shell branches.

The archive is checked against the SHA-512 that download.slicer.org publishes
before anything in it is run.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

_SERVER = "https://download.slicer.org"
_OS = {"linux": "linux", "win32": "win", "darwin": "macosx"}[sys.platform]


def _native_arch():
    machine = platform.machine().lower()
    return "arm64" if machine in ("arm64", "aarch64") else "amd64"


def _find(stability):
    """The newest build for this system, native architecture first.

    Slicer Stable has no arm64 macOS build, so an Apple silicon runner falls
    back to the amd64 one, which runs under Rosetta.
    """
    for arch in dict.fromkeys((_native_arch(), "amd64")):
        url = f"{_SERVER}/find?os={_OS}&stability={stability}&arch={arch}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
                build = json.load(response)
        except urllib.error.HTTPError:
            continue
        if build.get("download_url") and build.get("sha512"):
            return build
    raise SystemExit(f"No {stability} Slicer build for {_OS}")


def _download(build, dest):
    path = os.path.join(dest, "slicer-package")
    digest = hashlib.sha512()
    # download_url is a path on the same https server the build was found on.
    response = urllib.request.urlopen(_SERVER + build["download_url"], timeout=600)  # nosec B310
    with response, open(path, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != build["sha512"]:
        raise SystemExit("Slicer package does not match its published SHA-512")
    return path


def _unpack(package, dest):
    if _OS == "linux":
        with tarfile.open(package) as archive:
            archive.extractall(dest, filter="data")
        return glob.glob(os.path.join(dest, "Slicer-*", "Slicer"))[0]
    if _OS == "win":
        # Backslashes only: NSIS takes a /D with a forward slash in it for no
        # path at all and installs under Program Files instead.
        target = os.path.abspath(os.path.join(dest, "Slicer"))
        installer = package + ".exe"
        os.replace(package, installer)
        # NSIS: /S is silent, and /D must come last, unquoted.
        subprocess.run([installer, "/S", "/D=" + target], check=True)
        return _wait_for(os.path.join(target, "Slicer.exe"))
    mount = os.path.join(dest, "mount")
    subprocess.run(["hdiutil", "attach", package, "-nobrowse", "-quiet", "-mountpoint", mount],
                   check=True)
    try:
        app = glob.glob(os.path.join(mount, "*.app"))[0]
        shutil.copytree(app, os.path.join(dest, "Slicer.app"), symlinks=True)
    finally:
        subprocess.run(["hdiutil", "detach", mount, "-quiet"], check=False)
    return os.path.join(dest, "Slicer.app", "Contents", "MacOS", "Slicer")


def _wait_for(executable, timeout=600):
    """An NSIS installer can hand over to a copy of itself and return early."""
    deadline = time.monotonic() + timeout
    while not os.path.isfile(executable):
        if time.monotonic() > deadline:
            raise SystemExit(f"{executable} never appeared")
        time.sleep(5)
    return executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability", choices=("release", "nightly"), required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    build = _find(args.stability)
    print(f"Slicer {build['version']} r{build['revision']} {build['arch']}", file=sys.stderr)
    print(_unpack(_download(build, args.dest), args.dest))


if __name__ == "__main__":
    main()
