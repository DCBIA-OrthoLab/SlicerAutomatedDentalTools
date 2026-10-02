"""
Dependency checker and fixer for torchvision/pytorch compatibility
This module automatically detects and fixes version mismatches
"""

import sys
import subprocess
import os
from pathlib import Path

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("AREG_IOS_checkdeps")


def remove_broken_image_so():
    """Remove broken image.so file that causes import warnings"""
    try:
        # Ask torchvision where it actually lives rather than guessing a
        # lib/pythonX.Y directory - the interpreter version is not knowable
        # in advance and hardcoding it silently skipped the cleanup.
        import torchvision

        image_so_path = Path(torchvision.__file__).parent / "image.so"
        if image_so_path.exists():
            image_so_path.unlink()
            return True
    except (ImportError, OSError):
        # torchvision missing, or the file already removed by another pass.
        pass

    return False


def get_torch_version():
    """Get installed torch version"""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def get_torchvision_version():
    """Get installed torchvision version"""
    try:
        import torchvision
        return torchvision.__version__
    except ImportError:
        return None


def parse_version(version_str):
    """Parse version string to tuple"""
    try:
        parts = version_str.split('.')
        return tuple(int(p) for p in parts[:2])
    except Exception:
        return None


def get_compatible_torchvision(torch_version):
    """The torchvision release that ships alongside a given torch release.

    These are the pairings pytorch.org publishes together: torchvision 0.x is
    built against torch 2.(x-15). Naming a LATER torchvision here does not
    merely fail -- pip installs it, and it brings its own torch with it, so a
    repair meant to align torchvision quietly replaces the interpreter's
    torch. Measured: the table read 0.22.1 for torch 2.5, and one run turned
    a working 2.5.1+cu121 into 2.7.1+cu128, after which every compiled
    extension built against the old ABI -- pytorch3d, which ALI_IOS needs --
    failed to load.
    """
    compatibility_map = {
        (2, 8): "0.23.0",
        (2, 7): "0.22.0",
        (2, 6): "0.21.0",
        (2, 5): "0.20.0",
        (2, 4): "0.19.0",
        (2, 3): "0.18.0",
        (2, 2): "0.17.0",
        (2, 1): "0.16.0",
    }
    
    torch_ver = parse_version(torch_version)
    if torch_ver and torch_ver in compatibility_map:
        return compatibility_map[torch_ver]
    return None


def get_cuda_channel():
    """The pytorch.org wheel channel matching the installed torch.

    torch reports its build as `2.5.1+cu121`; the torchvision that goes with
    it lives in the channel of the same name. Hardcoding one meant a cu121
    install was handed cu128 wheels.
    """
    version = get_torch_version() or ""
    if "+" in version:
        local = version.split("+", 1)[1]
        if local.startswith("cu"):
            return local
    return None


def check_compatibility():
    """
    Check torch and torchvision compatibility
    Returns: (is_compatible, torch_version, tv_version, expected_tv_version)
    """
    torch_ver = get_torch_version()
    tv_ver = get_torchvision_version()
    
    if not torch_ver or not tv_ver:
        return None, torch_ver, tv_ver, None
    
    expected_tv = get_compatible_torchvision(torch_ver)
    
    torch_major_minor = parse_version(torch_ver)
    tv_major_minor = parse_version(tv_ver)
    expected_major_minor = parse_version(expected_tv) if expected_tv else None
    
    is_compatible = (tv_major_minor == expected_major_minor) if expected_major_minor else True
    
    return is_compatible, torch_ver, tv_ver, expected_tv


def fix_torchvision_auto():
    """Automatically fix torchvision if incompatible"""
    is_compatible, torch_ver, tv_ver, expected_tv = check_compatibility()
    
    if is_compatible is None:
        return False
    
    if is_compatible:
        return True
    
    logger.info("torchvision {} does not match torch {}, expected {}"
                .format(tv_ver, torch_ver, expected_tv))

    try:
        # --no-deps is what keeps this a repair rather than an upgrade. The
        # job here is to bring torchvision to the torch that is installed;
        # without it pip is free to satisfy torchvision by replacing torch,
        # which breaks every extension compiled against the old one.
        base = [
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--force-reinstall", "--no-cache-dir", "--no-deps",
        ]
        channel = get_cuda_channel()

        attempts = []
        if channel:
            attempts.append((["torchvision=={}+{}".format(expected_tv, channel)],
                             ["--index-url",
                              "https://download.pytorch.org/whl/{}".format(channel)]))
        attempts.append((["torchvision=={}".format(expected_tv)], []))

        for package, index in attempts:
            result = subprocess.run(base + package + index,
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return True
        
        # Try conda as another fallback
        try:
            cmd_conda = [
                "conda", "install", "-n", get_conda_env(),
                "torchvision={}".format(expected_tv.split('.')[0] + '.' + expected_tv.split('.')[1]),
                "-c", "pytorch", "-y"
            ]
            result = subprocess.run(cmd_conda, capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            logger.debug("Installation par conda impossible", exc_info=True)
        
        return False
            
    except Exception as e:
        return False


def get_conda_env():
    """Get current conda environment name"""
    return os.environ.get('CONDA_DEFAULT_ENV', 'base')


def ensure_compatible():
    """
    Ensure torch and torchvision are compatible
    Call this at the beginning of your module
    """
    # First, try to remove broken image.so to suppress warnings
    remove_broken_image_so()
    
    is_compatible, torch_ver, tv_ver, expected_tv = check_compatibility()
    
    if is_compatible is False:
        if fix_torchvision_auto():
            # Force reload to get new versions
            if 'torch' in sys.modules:
                del sys.modules['torch']
            if 'torchvision' in sys.modules:
                del sys.modules['torchvision']
            return True
    
    return True


if __name__ == "__main__":
    # Test mode
    ensure_compatible()
