"""Pick the pytorch.org wheel channel this machine's GPU can actually run.

A torch wheel carries compiled kernels for a fixed list of architectures. The
cu118 and cu121 wheels carry those cubins alone, with no PTX to fall back on, so
a GPU outside their list cannot be JIT-compiled for: it fails on the *first*
kernel launch with

    CUDA error: no kernel image is available for execution on the device

while ``torch.cuda.is_available()`` keeps answering True, because the driver and
the runtime are both fine. Nothing in that message names the wheel, so the
diagnosis goes looking at the machine, the driver, the data - anywhere but the
pin. That is how an RTX 5070 Ti (sm_120) lost 7 landmarks out of 7 in ALI_CBCT
with no line saying why.

The lists below were read out of the published cp312 wheels rather than assumed,
with the arch string that ``torch.cuda.get_arch_list()`` returns:

    cu118 (2.2.0)           sm_37 50 60 70 75 80 86 90
    cu121 (2.2.0)           sm_50 60 70 75 80 86 90
    cu128 (2.7.1, 2.11.0)   sm_75 80 86 90 100 120 + compute_120

No channel covers everything: cu128 is what Blackwell needs and it is also the
one that dropped Maxwell, Pascal and Volta. A single pin for the whole fleet
therefore cannot exist - which is why this picks per machine.
"""
import re
import subprocess
import sys

from ADTLib.logging_setup import get_logger

logger = get_logger(__name__)


class Channel:
    """One pytorch.org wheel channel and the torch trio published in it."""

    def __init__(self, name, torch, torchvision, torchaudio, architectures,
                 ptx=None):
        self.name = name
        self.torch = torch
        self.torchvision = torchvision
        self.torchaudio = torchaudio
        self.architectures = architectures
        # The `compute_NNN` entry, when the wheel carries one. PTX is source
        # the driver compiles on first launch, so it covers architectures the
        # wheel was never built for -- at the cost of a slow first kernel.
        self.ptx = ptx

    @property
    def index_url(self):
        return "https://download.pytorch.org/whl/" + self.name

    def requirements(self):
        """The three pins, which have to be resolved in a single pip call.

        torchvision and torchaudio link against libtorch, so a separate call is
        free to satisfy one of them by *replacing* torch - the failure then
        surfaces much later as `operator torchvision::nms does not exist`, or
        an `undefined symbol`, with nothing pointing back at the install.
        """
        return [
            "torch=={}+{}".format(self.torch, self.name),
            "torchvision=={}+{}".format(self.torchvision, self.name),
            "torchaudio=={}+{}".format(self.torchaudio, self.name),
        ]

    def pip_arguments(self):
        """The requirements plus the index that publishes them, as one string.

        ``--index-url`` rather than ``--extra-index-url``: the `+cuXXX` local
        versions only exist on the pytorch index, and leaving PyPI in the
        running is what lets pip answer a `torch==2.7.1` with the default
        variant instead. PyPI stays reachable for everything else through
        ``--extra-index-url``.
        """
        return " ".join(self.requirements() + [
            "--index-url", self.index_url,
            "--extra-index-url", "https://pypi.org/simple",
        ])

    def runs_on(self, capability):
        """Whether a device of this compute capability can run these kernels.

        CUDA guarantees binary compatibility *within* a major version only: an
        sm_86 cubin runs on a device of capability 8.9, which is why every Ada
        card works fine on a cu121 wheel whose list stops at sm_86 - and why an
        sm_90 cubin does nothing at all for a 12.0 device.

        PTX lifts that ceiling upwards. A wheel carrying `compute_120` can be
        JIT-compiled by the driver for any device at 12.0 or above, across
        majors, so the next generation is served without a new pin here. Only
        cu128 ships any: cu118 and cu121 are cubins alone, which is exactly why
        a card they do not list fails on its first launch instead of falling
        back to a slow one.
        """
        if capability is None:
            return False
        major, minor = capability
        if any(cmaj == major and cminor <= minor
               for cmaj, cminor in self.architectures):
            return True
        return self.ptx is not None and capability >= self.ptx

    def __repr__(self):
        return "<Channel {} torch {}>".format(self.name, self.torch)


# Ordered from the most conservative usable channel to the newest, and read in
# that order by `select_channel`. This is deliberate and load-bearing: every
# machine in the lab is on a cu121 torch 2.2.0 today, so putting cu121 first
# means a Blackwell card is the only thing this moves. Newest-first would drag
# every working Turing, Ampere and Ada machine onto a torch four minor versions
# up, and with it monai, nnunetv2 and the numpy pin - a fleet-wide upgrade
# dressed up as a bug fix.
CHANNELS = (
    Channel("cu121", "2.2.0", "0.17.0", "2.2.0",
            ((5, 0), (6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0))),
    Channel("cu118", "2.2.0", "0.17.0", "2.2.0",
            ((3, 7), (5, 0), (6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0))),
    Channel("cu128", "2.7.1", "0.22.1", "2.7.1",
            ((7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0)), ptx=(12, 0)),
)

_ARCH = re.compile(r"sm_(\d+)")
_PTX = re.compile(r"compute_(\d+)")


def _capability_from_digits(digits):
    """`'86'` -> `(8, 6)`, `'120'` -> `(12, 0)`.

    The last digit is the minor: sm_120 is Blackwell 12.0, not 1.20. Getting
    this backwards reads an sm_100 wheel as covering capability 1.0 and quietly
    excludes every Blackwell card from the channel that is meant to serve it.
    """
    return int(digits[:-1]), int(digits[-1])


def parse_architecture(name):
    """`'sm_86'` -> `(8, 6)`. A `compute_NNN` entry is PTX, not a cubin: None."""
    match = _ARCH.fullmatch(name.strip())
    return _capability_from_digits(match.group(1)) if match else None


def parse_ptx(name):
    """`'compute_120'` -> `(12, 0)`. Anything else: None."""
    match = _PTX.fullmatch(name.strip())
    return _capability_from_digits(match.group(1)) if match else None


def split_arch_list(names):
    """A `torch.cuda.get_arch_list()` split into cubins and the lowest PTX.

    Reading the list for cubins alone was wrong about the very wheel this
    module installs: torch 2.7.1+cu128 answers
    `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120`, and dropping that last
    entry loses the JIT path that serves whatever comes after Blackwell.
    """
    cubins, ptx = [], []
    for name in names:
        parsed = parse_architecture(name)
        if parsed:
            cubins.append(parsed)
            continue
        parsed = parse_ptx(name)
        if parsed:
            ptx.append(parsed)
    return cubins, (min(ptx) if ptx else None)


def capability_from_nvidia_smi():
    """The compute capability of GPU 0, without needing torch to be installed.

    The install path runs before torch is there on a fresh machine, so the
    channel cannot be chosen from `torch.cuda.get_device_capability`.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        logger.info("nvidia-smi could not be run: %s", error)
        return None
    if output.returncode != 0:
        logger.info("nvidia-smi failed: %s", (output.stderr or "").strip())
        return None
    first = (output.stdout or "").strip().splitlines()
    if not first:
        return None
    try:
        major, minor = first[0].strip().split(".")
        return int(major), int(minor)
    except ValueError:
        logger.info("could not read a compute capability from %r", first[0])
        return None


def capability_from_torch():
    """Same, asked of an installed torch. Second choice, and only a fallback.

    A machine with no driver at all answers nothing here *and* nothing to
    nvidia-smi, which is the CPU-only case and not an error.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception as error:
        logger.info("torch could not report a device capability: %s", error)
        return None


def device_capability():
    """`(major, minor)` for GPU 0, or None when there is no usable GPU."""
    return capability_from_nvidia_smi() or capability_from_torch()


def select_channel(capability=None):
    """The channel to install for this machine, or None when there is no GPU.

    None means "install the CPU build": there is no card to serve, and picking
    a CUDA channel anyway only downloads a gigabyte of kernels nothing runs.
    """
    if capability is None:
        capability = device_capability()
    if capability is None:
        return None

    for channel in CHANNELS:
        if channel.runs_on(capability):
            logger.info("GPU capability sm_%d%d -> %s (torch %s)",
                        capability[0], capability[1], channel.name, channel.torch)
            return channel

    logger.error(
        "No published torch wheel has kernels for a device of capability "
        "sm_%d%d. The channels checked were: %s. This GPU is either older than "
        "Kepler or newer than anything pinned here; the architecture lists at "
        "the top of this file have to be re-read before it can be served.",
        capability[0], capability[1],
        ", ".join(channel.name for channel in CHANNELS))
    return None


def torch_pip_arguments(capability=None):
    """What to hand pip for torch, as one string, or None for the CPU build."""
    channel = select_channel(capability)
    if channel is None:
        return None
    return channel.pip_arguments()


def installed_torch_serves_this_gpu():
    """Whether the torch already installed has kernels for this GPU.

    This is the check the install path was missing: `check_lib_installed` only
    compares version strings, so a torch that satisfies the pin and cannot run
    a single kernel here reads as "nothing to do".
    """
    capability = device_capability()
    if capability is None:
        return True                             # no GPU to serve, CPU is fine

    try:
        import torch
    except ImportError:
        return False

    architectures, ptx = split_arch_list(torch.cuda.get_arch_list())
    if not architectures and ptx is None:
        return False

    major, minor = capability
    if any(cmaj == major and cminor <= minor for cmaj, cminor in architectures):
        return True
    return ptx is not None and capability >= ptx


TORCH_STACK = ("torch", "torchvision", "torchaudio")


def torch_stack_is_usable(libs=TORCH_STACK):
    """Whether the installed torch trio is complete, consistent and runnable.

    Three separate ways it can fail to be, and only the third one is visible in
    a version number:

    - one of the three is simply missing;
    - they are present but built against different CUDA minors, which imports
      fine and fails much later inside a model with an `undefined symbol`;
    - they are present and consistent and have no kernels for this GPU.
    """
    import importlib.metadata

    for name in libs:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            logger.info("%s is not installed", name)
            return False
        except Exception as error:
            logger.warning("could not read the version of %s: %s", name, error)
            return False

    from ADTLib.env.deps import torch_cuda_conflict

    conflict = torch_cuda_conflict(libs)
    if conflict:
        logger.info("the torch family disagrees on a CUDA build: %s", conflict)
        return False

    return installed_torch_serves_this_gpu()


def torch_version_after_install(libs=TORCH_STACK):
    """The torch version that will be in place once the install has run.

    Callers use it to decide what has to agree with torch -- the numpy pin, in
    practice. Asking the *installed* torch instead answers for the environment
    being replaced, which is how a machine moving to 2.7 still got the numpy
    below 2.0 that only a torch older than 2.3 ever needed.
    """
    if not torch_stack_is_usable(libs):
        channel = select_channel()
        if channel is not None:
            return channel.torch
    try:
        import torch
        return torch.__version__.split("+")[0]
    except ImportError:
        return None


def torch_needs_numpy1(libs=TORCH_STACK):
    """Whether the torch that will be in place has to have numpy below 2.

    torch was built against numpy 1.x up to and including 2.2, and works with
    either from 2.3 on. Below that the mismatch shows up as "Failed to
    initialize NumPy: _ARRAY_API not found", from inside monai, long after pip
    reported success -- which is why this is pinned rather than left to resolve.
    """
    version = torch_version_after_install(libs)
    if version is None:
        return True

    try:
        from packaging.version import Version
        return Version(version) < Version("2.3.0")
    except Exception as error:
        logger.warning("could not compare the torch version %r: %s", version, error)
        return True


def torch_install_arguments(libs=TORCH_STACK):
    """The pip arguments to repair the torch stack, or None when it is fine.

    None also covers the machine with no GPU at all: there is nothing to serve,
    and pulling a CUDA channel there downloads a gigabyte of kernels that
    nothing will ever launch.
    """
    if torch_stack_is_usable(libs):
        return None
    return torch_pip_arguments()


_kernels_usable = None


def cuda_kernels_usable():
    """Whether a CUDA kernel actually *runs* here. Cached; never raises.

    `torch.cuda.is_available()` answers for the driver and the runtime, not for
    the wheel: on a GPU the wheel was not compiled for it says True and every
    launch afterwards fails. Only launching one settles it, so this launches
    the smallest possible one and synchronises, because the error is reported
    asynchronously and would otherwise land on some unrelated later call.
    """
    global _kernels_usable
    if _kernels_usable is not None:
        return _kernels_usable

    try:
        import torch
        if not torch.cuda.is_available():
            _kernels_usable = False
            return _kernels_usable
    except ImportError:
        _kernels_usable = False
        return _kernels_usable

    try:
        probe = torch.ones(8, device="cuda")
        (probe + probe).sum().item()
        torch.cuda.synchronize()
        _kernels_usable = True
    except Exception as error:
        capability = capability_from_torch()
        logger.error(
            "This GPU cannot run the installed torch. %s (sm_%s) needs kernels "
            "this wheel does not carry - it was built for: %s. torch reports "
            "CUDA as available, which is why nothing failed until the first "
            "launch. The error was: %s",
            _device_name(), "".join(str(part) for part in (capability or ("?",))),
            " ".join(_arch_list()), error)
        _kernels_usable = False

    return _kernels_usable


def _device_name():
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "the GPU"


def _arch_list():
    try:
        import torch
        return torch.cuda.get_arch_list()
    except Exception:
        return ["unknown"]


def preferred_device():
    """The torch device to run on: cuda when its kernels work here, else cpu.

    Every module that used `torch.device("cuda" if torch.cuda.is_available()
    else "cpu")` sent work to a GPU that could not run it and reported the
    result as "not found". Falling back to the CPU is slow, and slow beats a
    run that loses every landmark without saying so.
    """
    import torch

    if cuda_kernels_usable():
        return torch.device("cuda")

    if torch.cuda.is_available():
        logger.warning(
            "Falling back to the CPU: the GPU is visible but this torch build "
            "has no kernels for it. Expect the run to be much slower. Install "
            "a torch from %s to use the GPU.",
            (select_channel() or CHANNELS[0]).index_url)
    return torch.device("cpu")


if __name__ == "__main__":                       # a one-line answer per machine
    capability = device_capability()
    print("compute capability :", capability)
    print("channel to install :", select_channel(capability))
    print("pip arguments      :", torch_pip_arguments(capability))
    try:
        import torch
        print("installed torch    :", torch.__version__, torch.cuda.get_arch_list())
        print("serves this GPU    :", installed_torch_serves_this_gpu())
        print("kernels usable     :", cuda_kernels_usable())
    except ImportError:
        print("installed torch    : absent")
    sys.exit(0)
