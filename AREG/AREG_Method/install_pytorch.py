#!/usr/bin/env python3
"""
Install a pytorch3d build that matches the torch already present in the env.

PyTorch3D publishes no wheel on PyPI for modern Python/torch combinations, and
the historical index (dl.fbaipublicfiles.com) stopped at Python 3.9 / torch 2.0,
which is what forced this environment onto Python 3.9 for years. Prebuilt
wheels for cp310-cp313 are served from a PEP 503 index instead:

    https://ImageMindAnalytics.github.io/pytorch3d-wheels/simple/

Wheels there carry a local version tag describing what they were built against
(``0.7.9+pt2110cu128`` = torch 2.11.0, CUDA 12.8). Installing one whose tag does
not match the installed torch produces an ``undefined symbol`` crash on import,
so the tag is computed from torch rather than left to pip's resolver.
"""
import logging
import re
import subprocess
import sys
import urllib.parse
import urllib.request

WHEEL_INDEX = "https://ImageMindAnalytics.github.io/pytorch3d-wheels/simple/"
WHEEL_LISTING = WHEEL_INDEX + "pytorch3d/"

# ===== Logging Configuration =====
logger = logging.getLogger("AREG_install_pytorch")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def torch_build_tag():
    """torch 2.11.0+cu128 -> 'pt2110cu128'; a CPU build -> 'pt2110cpu'."""
    import torch

    base = torch.__version__.split("+")[0]
    major, minor, micro = (base.split(".") + ["0", "0"])[:3]
    cuda = torch.version.cuda
    suffix = "cu" + cuda.replace(".", "") if cuda else "cpu"
    return "pt{}{}{}{}".format(major, minor, micro, suffix)


def platform_tag():
    if sys.platform.startswith("win"):
        return "win_amd64"
    if sys.platform == "darwin":
        return "macosx"
    return "manylinux"


def list_wheels():
    """Return [(filename, url)] published on the index, or [] if unreachable."""
    try:
        with urllib.request.urlopen(WHEEL_LISTING, timeout=60) as response:
            html = response.read().decode("utf-8", "replace")
    except Exception as exc:
        logger.warning("Could not read the pytorch3d wheel index: {}".format(exc))
        return []

    wheels = []
    for href, text in re.findall(r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>', html):
        name = text.strip()
        if name.endswith(".whl"):
            wheels.append((name, urllib.parse.urljoin(WHEEL_LISTING, href)))
    return wheels


def select_wheel(wheels, py_tag, plat_tag, torch_tag):
    """Highest-version wheel matching this interpreter, platform and torch."""
    matches = [
        (name, url) for name, url in wheels
        if "-{}-".format(py_tag) in name and plat_tag in name and "+{}-".format(torch_tag) in name
    ]
    if not matches:
        return None
    return sorted(matches)[-1]


def run_pip(pip_path, args):
    cmd = [pip_path, "install", "--no-cache-dir"] + args
    logger.info("Running: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr.strip())
    return result.returncode == 0


def verify_gpu():
    """
    Run a real pytorch3d CUDA kernel.

    Imports are not proof of anything: a wheel built without kernels for the
    installed GPU imports cleanly and then fails on the first operation with
    'no kernel image is available for execution on the device'. Catching that
    here turns a silent, mid-pipeline crash into an actionable message.
    """
    try:
        import torch
    except Exception as exc:
        logger.error("torch is not importable: {}".format(exc))
        return False

    try:
        import pytorch3d
        import pytorch3d.renderer  # noqa: F401
        logger.info("pytorch3d {} imported".format(pytorch3d.__version__))
    except Exception as exc:
        logger.error("pytorch3d is installed but not importable: {}".format(exc))
        return False

    if not torch.cuda.is_available():
        logger.warning("No CUDA device visible - pytorch3d will run on CPU.")
        return True

    major, minor = torch.cuda.get_device_capability(0)
    capability = "sm_{}{}".format(major, minor)
    arch = "{}.{}".format(major, minor)
    device_name = torch.cuda.get_device_name(0)
    try:
        from pytorch3d.ops import knn_points

        points = torch.rand(1, 16, 3, device="cuda")
        knn_points(points, points, K=2)
        logger.info("pytorch3d CUDA kernels run on {} ({})".format(device_name, capability))
        return True
    except Exception as exc:
        logger.error("=" * 70)
        logger.error("pytorch3d imports but CANNOT run on this GPU.")
        logger.error("  GPU        : {} ({})".format(device_name, capability))
        logger.error("  torch built for: {}".format(" ".join(torch.cuda.get_arch_list())))
        logger.error("  error      : {}".format(exc))
        logger.error("")
        logger.error("The installed pytorch3d wheel contains no kernels for this")
        logger.error("architecture. Either ask for wheels built with '{}' in".format(capability))
        logger.error("cuda_arch_list, or rebuild locally:")
        logger.error("    TORCH_CUDA_ARCH_LIST=\"{}\" FORCE_CUDA=1 \\".format(arch))
        logger.error("    pip install --no-build-isolation \\")
        logger.error("        'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'")
        logger.error("=" * 70)
        return False


def install_pytorch3d(pip_path):
    try:
        torch_tag = torch_build_tag()
    except Exception as exc:
        logger.error("Cannot read the installed torch version: {}".format(exc))
        return False

    py_tag = "cp3{}".format(sys.version_info.minor)
    plat_tag = platform_tag()
    logger.info("Looking for pytorch3d matching {} / {} / {}".format(py_tag, plat_tag, torch_tag))

    installed = False
    selected = select_wheel(list_wheels(), py_tag, plat_tag, torch_tag)
    if selected:
        name, url = selected
        logger.info("Selected wheel: {}".format(name))
        installed = run_pip(pip_path, [url])
    else:
        logger.warning(
            "No prebuilt wheel for {} / {} / {}. Letting pip resolve from the "
            "index; if it picks a build tagged for another torch, pytorch3d "
            "will fail to import.".format(py_tag, plat_tag, torch_tag)
        )
        installed = run_pip(pip_path, ["pytorch3d", "--extra-index-url", WHEEL_INDEX])

    if not installed:
        logger.error("pytorch3d installation failed.")
        return False

    logger.info("PyTorch3D installed in the environment")
    return verify_gpu()


def main(pip_path):
    install_pytorch3d(pip_path)


if __name__ == "__main__":
    main(sys.argv[1])
