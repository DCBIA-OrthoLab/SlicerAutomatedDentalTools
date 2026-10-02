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

torch itself is installed here too, because computing the tag from whatever
torch happens to be present only works if that torch has a wheel. It used to be
pulled in beforehand by ``condaCreateEnv(["torch>=2.8,<2.13", ...])``, which
cannot pass an index URL and so took PyPI's default CUDA variant - on
2026-09-16 that was ``2.12.1+cu130``, a combination this index does not publish
and never will, since it builds against release torch versions and other CUDA
minors.

The pin in ``TORCH_PINS`` is what a *new* environment gets. An existing torch is
only replaced when the index publishes no wheel for it; one that is supported is
kept, whatever the pin says. See ``install_torch``.
"""
import re
import subprocess
import sys
import urllib.parse
import urllib.request

WHEEL_INDEX = "https://ImageMindAnalytics.github.io/pytorch3d-wheels/simple/"
WHEEL_LISTING = WHEEL_INDEX + "pytorch3d/"

# The torch build every other package in this environment has to agree with.
#
# pytorch3d and torchvision both link against libtorch, so version *and* CUDA
# minor have to match the wheel exactly. Each mismatch fails differently and
# none of them says "wrong version": pytorch3d built for another torch raises
# `undefined symbol: _ZN3c104cuda...` on import, torchvision built for another
# torch raises `RuntimeError: operator torchvision::nms does not exist`, and
# that second one only surfaces from inside shapeaxi.dental_model_seg, long
# after pytorch3d itself imports and runs CUDA kernels fine.
#
# 2.11.0+cu128 is chosen on the CUDA minor, not on the torch version. The
# wheels carry no PTX, so a GPU whose architecture was not compiled in cannot
# fall back to JIT - it fails on the first kernel with "no kernel image is
# available". Read out of the published cp312 wheels with cuobjdump:
#
#     cu126 (pt2110, pt2120)  sm_70 75 80 86 89 90
#     cu128 (pt2110)          sm_70 75 80 86 89 90 120
#     cu132 (pt2120, pt2140)  sm_75 80 86 89 90 120
#
# cu128 is a strict superset of cu126, so it costs nothing and adds Blackwell
# (sm_120, the RTX 50 series). cu132 would drop Volta and needs an r580+ driver
# for the CUDA 13 runtime - a machine reporting "CUDA Version: 12.9" in
# nvidia-smi cannot run it. None of the three has Maxwell or Pascal.
#
# 2.11.0 is then the newest torch cu128 carries, and shapeaxi wants torch<2.13.
# To move: change the three values together, check WHEEL_INDEX has the matching
# pt<ver>cu<minor> for cp310-cp313 on manylinux and win_amd64, and re-read the
# architectures rather than assuming a newer CUDA covers more.
TORCH_PINS = {
    "manylinux": ("2.11.0+cu128", "0.26.0+cu128",
                  "https://download.pytorch.org/whl/cu128"),
    "win_amd64": ("2.11.0+cu128", "0.26.0+cu128",
                  "https://download.pytorch.org/whl/cu128"),
    # No CUDA wheel is published for macOS; the index only carries pt280cpu.
    "macosx": ("2.8.0", "0.23.0", None),
}

# The torch range shapeaxi itself accepts - `torch>=2.8,<2.13` as of 2.0.3.
#
# A torch outside it is not worth keeping even when the index does publish a
# wheel for it, which is the case for 2.14.0+cu132. Keeping it would install the
# matching pytorch3d, and then `pip install shapeaxi` would move torch back
# under 2.13 to satisfy its own requirement - taking PyPI's default variant on
# the way - leaving the wheel installed one step earlier built against a torch
# that is no longer there. Bump this whenever SHAPEAXI_REQUIREMENT moves.
SHAPEAXI_TORCH_RANGE = ((2, 8), (2, 13))

# Declared by shapeaxi anyway, but installed here so it resolves against the
# pinned torch. Listed in condaCreateEnv it did the opposite: it declares an
# unbounded `torch` dependency, so it dragged PyPI's default build in first.
EXTRA_REQUIREMENTS = ["ocnn==2.2.1"]

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger(__name__)


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


_wheel_cache = None


def list_wheels():
    """Return [(filename, url)] published on the index, or [] if unreachable.

    Read once per run: install_torch and install_pytorch3d both need it, and
    one listing is one fewer thing that can fail between them.
    """
    global _wheel_cache
    if _wheel_cache is not None:
        return _wheel_cache
    _wheel_cache = _read_index()
    return _wheel_cache


def _read_index():
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


def wheels_for(wheels, py_tag, plat_tag):
    """Everything published for this interpreter and platform, any torch."""
    return [
        (name, url) for name, url in wheels
        if "-{}-".format(py_tag) in name and plat_tag in name
    ]


def select_wheel(wheels, py_tag, plat_tag, torch_tag):
    """Highest-version wheel matching this interpreter, platform and torch."""
    matches = [
        (name, url) for name, url in wheels_for(wheels, py_tag, plat_tag)
        if "+{}-".format(torch_tag) in name
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


def torch_in_shapeaxi_range(torch_tag):
    """Does this build tag name a torch shapeaxi will accept as it stands?

    'pt2120cu126' -> 2.12 -> True; 'pt2140cu132' -> 2.14 -> False. The tag packs
    major, minor and micro with no separator, so it is read back from the ends:
    first digit major, last digit micro, the rest minor.
    """
    match = re.match(r"pt(\d+)(?:cu\d+|cpu)$", torch_tag)
    if not match:
        return False
    digits = match.group(1)
    version = (int(digits[0]), int(digits[1:-1] or 0))
    low, high = SHAPEAXI_TORCH_RANGE
    return low <= version < high


def install_torch(pip_path):
    """Put torch on a build the index publishes a pytorch3d wheel for.

    Only when it is not on one already. Any torch with a published wheel is
    left exactly as it is, even though it is not the pin: replacing it can only
    lose ground. It would mean a 3 GB download, and it would swap whatever GPU
    coverage that build has for the pin's - an environment on 2.11.0+cu128
    moved to cu126 loses the Blackwell kernels CUDA 12.8 added.

    This has to hold on every launch, not just the first: FlexReg calls
    install_pytorch3d unconditionally (it defines check_if_pytorch3d and never
    calls it), so this module runs on each of its boots. ASO, AREG, ALI and
    DOCShapeAXI only reach it when check_if_pytorch3d has already failed.
    """
    plat_tag = platform_tag()
    py_tag = "cp3{}".format(sys.version_info.minor)
    pin = TORCH_PINS.get(plat_tag)
    if pin is None:
        logger.error("No torch pin for platform '{}'.".format(plat_tag))
        return False

    wheels = list_wheels()
    if not wheels:
        # Without the listing there is no way to tell a supported torch from an
        # unsupported one. Reinstalling on a guess is the expensive, damaging
        # direction, so stop here and leave the environment untouched.
        logger.error(
            "The pytorch3d wheel index is unreachable, so which torch builds "
            "are supported cannot be established. Leaving torch alone.")
        return False

    if not wheels_for(wheels, py_tag, plat_tag):
        # An interpreter the index does not build for at all, rather than one
        # whose torch is wrong: a Python 3.9 environment left from before the
        # move to 3.12, say. Nothing below could succeed either - the pinned
        # torch has no cp39 wheel on the PyTorch index - so pip would fail on
        # resolution with a message about torch that says nothing about the
        # actual problem, which is the interpreter.
        logger.error(
            "The wheel index publishes no pytorch3d for {} / {}; it starts at "
            "cp310. This environment runs Python {}.{}.".format(
                py_tag, plat_tag, sys.version_info.major, sys.version_info.minor))
        logger.error(
            "Delete the environment and let the module rebuild it - it is "
            "created at a supported Python now. Nothing was installed.")
        return False

    try:
        current = torch_build_tag()
    except Exception:
        current = None  # no torch in this environment yet

    if current and torch_in_shapeaxi_range(current) \
            and select_wheel(wheels, py_tag, plat_tag, current):
        logger.info(
            "torch is at {}, which has a published pytorch3d wheel and suits "
            "shapeaxi - keeping it rather than moving to the {} pin.".format(
                current, pin[0]))
        return True

    if current and not torch_in_shapeaxi_range(current):
        logger.info(
            "torch {} is outside the range shapeaxi accepts ({}).".format(
                current, SHAPEAXI_REQUIREMENT))
    elif current:
        logger.info("torch {} has no published pytorch3d wheel.".format(current))

    torch_version, vision_version, index = pin
    args = ["torch=={}".format(torch_version), "torchvision=={}".format(vision_version)]
    if index:
        # --index-url replaces PyPI, so the +cuXXX local versions resolve from
        # the pytorch index; --extra-index-url puts PyPI back for everything
        # else these two pull in.
        args += ["--index-url", index, "--extra-index-url", "https://pypi.org/simple"]

    logger.info("Pinning torch {} / torchvision {}".format(torch_version, vision_version))
    if run_pip(pip_path, args):
        return True
    logger.error("Could not install the pinned torch build.")
    return False


def install_extras(pip_path):
    """Install the torch-dependent packages that used to sit in condaCreateEnv."""
    if not EXTRA_REQUIREMENTS:
        return True
    if run_pip(pip_path, list(EXTRA_REQUIREMENTS)):
        return True
    logger.error("Could not install {}.".format(", ".join(EXTRA_REQUIREMENTS)))
    return False


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

    selected = select_wheel(list_wheels(), py_tag, plat_tag, torch_tag)
    if not selected:
        # Deliberately not falling back to `pip install pytorch3d
        # --extra-index-url`. pip has no way to know which build matches this
        # torch, so it takes the highest local version on the index - that is
        # how an environment on torch 2.12.1 ended up with 0.7.9+pt2140cu132
        # and an `undefined symbol` on every import. Worse, that install
        # reports success and sticks: check_if_pytorch3d then fails on every
        # launch, and the module reinstalls the same broken wheel each time.
        expected = TORCH_PINS.get(plat_tag)
        logger.error("No pytorch3d wheel published for {} / {} / {}.".format(
            py_tag, plat_tag, torch_tag))
        if expected:
            logger.error(
                "This environment is not on the pinned torch build. Expected "
                "torch {}; install_torch should have put it there.".format(expected[0]))
        logger.error(
            "Refusing to let pip pick another build: it would install one "
            "tagged for a different torch, which imports as 'undefined symbol' "
            "and has to be uninstalled by hand.")
        return False

    name, url = selected
    logger.info("Selected wheel: {}".format(name))
    installed = run_pip(pip_path, [url])

    if not installed:
        logger.error("pytorch3d installation failed.")
        return False

    logger.info("PyTorch3D installed in the environment")
    return verify_gpu()


SHAPEAXI_REQUIREMENT = "shapeaxi>=2.0.2"


def install_shapeaxi(pip_path):
    """Install shapeaxi once pytorch3d is in place.

    shapeaxi declares pytorch3d as a hard requirement, and PyPI serves no
    distribution for it at all, so asking pip for shapeaxi in a bare
    environment ends on "No matching distribution found for pytorch3d" and
    leaves nothing behind. Installing it here, after the wheel above, gives
    pip an already-satisfied requirement to resolve against.
    """
    if run_pip(pip_path, [SHAPEAXI_REQUIREMENT]):
        logger.info("{} installed in the environment".format(SHAPEAXI_REQUIREMENT))
        return True
    logger.error("{} installation failed.".format(SHAPEAXI_REQUIREMENT))
    return False


STALE_CALL = "saxi_nets.DentalModelSeg"
STALE_IMPORT = "from shapeaxi import saxi_nets, utils"
FIXED_IMPORT = "from shapeaxi import saxi_nets_lightning, utils"


def patch_dentalmodelseg():
    """Repoint dentalmodelseg at the class it needs, on shapeaxi 2.0.0 - 2.0.2.

    `dental_model_seg.py` calls `saxi_nets.DentalModelSeg`, but the class moved
    to `saxi_nets_lightning` in the 2.0 split and `saxi_nets` never re-exported
    it, so every crown segmentation dies on

        AttributeError: module 'shapeaxi.saxi_nets' has no attribute 'DentalModelSeg'

    Reported upstream (ImageMindAnalytics/ShapeAXI); this rewrites the two stale
    references until a release carries the fix. It is a no-op on any version
    that does not have the problem, so it disappears on its own once shapeaxi is
    updated. `saxi_nets` is not used anywhere else in that file.
    """
    try:
        from shapeaxi import dental_model_seg
        path = dental_model_seg.__file__
        with open(path) as handle:
            source = handle.read()
    except Exception as exc:
        logger.warning("Could not read shapeaxi.dental_model_seg: {}".format(exc))
        return False

    if STALE_CALL not in source:
        logger.info("dentalmodelseg needs no patching on this shapeaxi")
        return True

    try:
        with open(path, "w") as handle:
            handle.write(source.replace(STALE_IMPORT, FIXED_IMPORT)
                               .replace(STALE_CALL, "saxi_nets_lightning.DentalModelSeg"))
    except Exception as exc:
        logger.error("Could not patch {}: {}".format(path, exc))
        return False

    logger.info("Patched {} so dentalmodelseg finds DentalModelSeg".format(path))
    return True


def main(pip_path):
    # Order matters: every step below resolves against the torch installed by
    # the one before it.
    if not install_torch(pip_path):
        logger.error(
            "Not installing pytorch3d: it has to be built against the torch in "
            "this environment, and that torch is not the pinned one.")
        return
    if not install_pytorch3d(pip_path):
        logger.error(
            "Not installing shapeaxi: it requires a working pytorch3d, and pip "
            "cannot resolve pytorch3d from PyPI on its own.")
        return
    if not install_extras(pip_path):
        return
    if install_shapeaxi(pip_path):
        patch_dentalmodelseg()


if __name__ == "__main__":
    main(sys.argv[1])
