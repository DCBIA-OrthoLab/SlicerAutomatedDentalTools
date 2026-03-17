"""
Dependency checker and fixer for torchvision/pytorch compatibility
This module automatically detects and fixes version mismatches
"""

import sys
import subprocess
import os
from pathlib import Path
from distutils.version import StrictVersion


def remove_broken_image_so():
    """Remove broken image.so file that causes import warnings"""
    try:
        # Find and remove broken image.so
        python_prefix = sys.prefix
        image_so_path = Path(python_prefix) / "lib" / "python3.9" / "site-packages" / "torchvision" / "image.so"
        
        if image_so_path.exists():
            try:
                image_so_path.unlink()
                return True
            except Exception as e:
                return False
        
        # Also check other potential locations
        other_paths = [
            Path(python_prefix) / "lib" / "python3.8" / "site-packages" / "torchvision" / "image.so",
            Path(python_prefix) / "lib" / "python3.10" / "site-packages" / "torchvision" / "image.so",
            Path(python_prefix) / "lib" / "python3.11" / "site-packages" / "torchvision" / "image.so",
        ]
        
        for path in other_paths:
            if path.exists():
                try:
                    path.unlink()
                    return True
                except:
                    pass
        
    except Exception as e:
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
    except:
        return None


def get_compatible_torchvision(torch_version):
    """Get compatible torchvision version for given torch version"""
    # Map of PyTorch version to available torchvision versions
    # Use versions available on pytorch.org wheels
    compatibility_map = {
        (2, 7): "0.23.0",  # PyTorch 2.7 -> TorchVision 0.23.0+cu128
        (2, 6): "0.23.0", 
        (2, 5): "0.22.1",
        (2, 4): "0.22.0",
        (2, 3): "0.22.0",
        (2, 2): "0.17.0",
        (2, 1): "0.16.0",
    }
    
    torch_ver = parse_version(torch_version)
    if torch_ver and torch_ver in compatibility_map:
        return compatibility_map[torch_ver]
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
    
    try:
        # Try with +cu128 format (available on pytorch.org)
        torchvision_pkg = "torchvision=={}+cu128".format(expected_tv)
        
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "--force-reinstall", "--no-cache-dir",
            torchvision_pkg,
            "--index-url", "https://download.pytorch.org/whl/cu128"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        
        # Try without +cu128 suffix as fallback
        cmd[3] = "torchvision=={}".format(expected_tv)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
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
        except:
            pass
        
        return False
            
    except Exception as e:
        return False


def get_conda_env():
    """Get current conda environment name"""
    import os
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
            import importlib
            if 'torch' in sys.modules:
                del sys.modules['torch']
            if 'torchvision' in sys.modules:
                del sys.modules['torchvision']
            return True
    
    return True


if __name__ == "__main__":
    # Test mode
    ensure_compatible()
