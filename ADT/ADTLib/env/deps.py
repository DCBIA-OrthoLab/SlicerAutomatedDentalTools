"""Is a library present, and does it satisfy the constraint the module asked for.

Nine modules carried their own `check_lib_installed`, under three incompatible
answers to the same question:

  - AREG and FlexReg: `if required_version: pass` -- true as soon as the library
    is installed at all, whatever version. The constraint was decoration.
  - ALI: exact equality against a bare version string (`'2.6.2'`).
  - MedX: a hand-rolled regex over one operator.
  - MRI2CBCT: `packaging.SpecifierSet`, which handles every PEP 440 form.

The last one is the only one that is both correct and complete, so it is the one
kept here. Constraints are now actually applied, which is the point: the pins
that were silently ignored are what let torch and pytorch3d drift apart.

Two spellings from the old call sites are accepted so no list has to be rewritten
in the same breath: a bare version (`'2.6.2'`) means `'==2.6.2'`, and a constraint
carried inside the name (`'numpy<2.0.0'`) is split off.

Standard library plus `packaging`, which Slicer ships.
"""
import importlib.metadata
import logging
import re

from packaging.specifiers import SpecifierSet
from packaging.version import Version

logger = logging.getLogger(__name__)

_NAME_WITH_SPEC = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*([<>=!~].*)$")
_BARE_VERSION = re.compile(r"^\s*[0-9][0-9A-Za-z.\-+*]*\s*$")


def normalise_spec(lib_name, required_version=None):
    """(name, specifier) from the spellings the call sites use.

    Returns the specifier as a PEP 440 string, or None when the call site asked
    for no particular version.
    """
    embedded = _NAME_WITH_SPEC.match(lib_name or "")
    if embedded:
        lib_name, carried = embedded.group(1), embedded.group(2).strip()
        required_version = required_version or carried
    if required_version and _BARE_VERSION.match(required_version):
        # '2.6.2' meant '==2.6.2' at the one call site that wrote it that way.
        required_version = "==" + required_version.strip()
    return lib_name.strip(), (required_version.strip() if required_version else None)


def requirement(lib_name, required_version=None):
    """The string to hand pip, from the spellings the call sites use.

    `('dicom2nifti', '>=2.6.2')` must become `dicom2nifti>=2.6.2`, not
    `dicom2nifti==>=2.6.2`, which pip rejects outright:

        Invalid requirement: 'dicom2nifti==>=2.6.2'

    ALI and ASO used to glue `==` in unconditionally, which was correct only as
    long as every entry of their list carried a bare version. It stopped being
    correct the day one of them carried an operator, and the failure lands on a
    machine that does not have the library at all -- a fresh install, that is.
    FlexReg had the right test inline; this is that test, in one place.
    """
    lib_name, spec = normalise_spec(lib_name, required_version)
    return lib_name + (spec or "")


def check_lib_installed(lib_name, required_version=None):
    """Whether `lib_name` is installed and satisfies `required_version`."""
    lib_name, required_version = normalise_spec(lib_name, required_version)
    try:
        installed = Version(importlib.metadata.version(lib_name))
    except importlib.metadata.PackageNotFoundError:
        logger.info("%s is not installed", lib_name)
        return False
    except Exception as error:                      # unreadable metadata
        logger.warning("could not read the version of %s: %s", lib_name, error)
        return False

    if not required_version:
        return True
    if installed in SpecifierSet(required_version):
        return True
    logger.info("%s %s does not satisfy %s", lib_name, installed, required_version)
    return False


TORCH_FAMILY = ("torch", "torchvision", "torchaudio")

_CUDA_LABEL = re.compile(r"\+cu(\d+)")


def torch_cuda_labels(libs=TORCH_FAMILY, lookup=None):
    """The CUDA build each installed member of the torch family declares.

    `{'torch': '118', 'torchvision': None}`: present but silent is None, absent
    is missing from the mapping. Wheels from PyPI carry no `+cuXXX` at all,
    those from `download.pytorch.org/whl/cu118` do -- which is why the two
    cases have to stay distinct.
    """
    lookup = lookup or importlib.metadata.version
    found = {}
    for name in libs:
        try:
            version = lookup(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as error:
            logger.warning("could not read the version of %s: %s", name, error)
            continue
        match = _CUDA_LABEL.search(version or "")
        found[name] = match.group(1) if match else None
    return found


def torch_cuda_conflict(libs=TORCH_FAMILY, lookup=None):
    """The disagreement, when two members declare *different* CUDA builds.

    Returns `{name: label}` for the members that disagree, or None. Only an
    explicit disagreement counts: a wheel with no `+cuXXX` says nothing about
    what it was built against, and calling that a mismatch would fire on every
    plain PyPI install. This is the check to run when a warning must not be a
    false alarm.

    A torch and a torchvision from different CUDA minors import fine and fail
    much later, inside a model, with an `undefined symbol` nobody can read back
    to its cause.
    """
    labels = {name: label for name, label in torch_cuda_labels(libs, lookup).items()
              if label is not None}
    if len(set(labels.values())) <= 1:
        return None
    return labels


def torch_cuda_builds_agree(libs=TORCH_FAMILY, lookup=None):
    """Whether the whole family is installed *and* all from the same CUDA build.

    Stricter than `torch_cuda_conflict`: a missing member, or one without a
    `+cuXXX` label, is a no. That is what AMASSS wants, because it installs the
    three together from `download.pytorch.org/whl/cuXXX` and a wheel that came
    from anywhere else is one it means to replace. Anywhere the answer only
    feeds a warning, use `torch_cuda_conflict` instead.

    AMASSS's own copy compared the first pair and returned on it, so a
    torchaudio out of step with the other two answered "agree". This one
    compares the whole set.
    """
    labels = torch_cuda_labels(libs, lookup)
    if set(labels) != set(libs):
        return False
    if any(label is None for label in labels.values()):
        return False
    return len(set(labels.values())) == 1
