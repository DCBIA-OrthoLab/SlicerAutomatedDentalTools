#!/usr/bin/env python3
"""
Enhanced PyTorch3D Installation Script for 3D Slicer
Automatically installs CUDA toolkit if nvcc is not found
Designed to be run within Slicer's conda environment using conda run
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(cmd, shell=False, check=True, env=None):
    """Run a command and return the result"""
    print(f"[CMD] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, 
                              capture_output=True, text=True, env=env)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        if check:
            raise
        return e

def find_conda_executable():
    """Find the conda executable in the current environment"""
    print("[INFO] Looking for conda executable...")
    
    # Method 1: Check if conda is directly available
    conda_cmd = shutil.which("conda")
    if conda_cmd:
        print(f"[INFO] Found conda in PATH: {conda_cmd}")
        return conda_cmd
    
    # Method 2: Check environment variables
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        possible_paths = [
            Path(conda_prefix) / "bin" / "conda",
            Path(conda_prefix).parent / "bin" / "conda",
            Path(conda_prefix).parent / "condabin" / "conda",
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"[INFO] Found conda via CONDA_PREFIX: {path}")
                return str(path)
    
    # Method 3: Relative to Python executable (for Slicer environments)
    python_path = Path(sys.executable)
    possible_paths = [
        python_path.parent / "conda",
        python_path.parent.parent / "bin" / "conda",
        python_path.parent.parent / "condabin" / "conda",
        python_path.parent.parent.parent / "bin" / "conda",  # For deep env structures
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"[INFO] Found conda relative to Python: {path}")
            return str(path)
    
    # Method 4: Check common installation paths
    common_paths = [
        "/usr/local/bin/conda",
        "/opt/conda/bin/conda",
        "/opt/miniconda3/bin/conda",
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
    ]
    
    for path in common_paths:
        if Path(path).exists():
            print(f"[INFO] Found conda in common location: {path}")
            return path
    
    print("[WARNING] Could not find conda executable")
    return None

def install_cuda_toolkit():
    """Install CUDA toolkit via conda if not already present"""
    print("[INFO] Installing CUDA toolkit via conda...")
    
    # Find conda executable
    conda_cmd = find_conda_executable()
    
    if not conda_cmd:
        print("[ERROR] Cannot install CUDA toolkit - conda not found")
        return False
    
    try:
        # Get current environment name
        env_name = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        print(f"[INFO] Installing CUDA toolkit in environment: {env_name}")
        
        # Install CUDA toolkit from nvidia channel
        cmd = [conda_cmd, "install", "nvidia/label/cuda-12.6.0::cuda-toolkit", "-c", "nvidia", "-y"]
        
        # If we're in a specific environment, specify it
        if env_name != 'base':
            cmd.extend(["-n", env_name])
        
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            print("[SUCCESS] CUDA toolkit installed via conda")
            
            # Update PATH to include new CUDA installation
            cuda_bin_paths = [
                Path(sys.prefix) / "bin",
                Path(os.environ.get('CONDA_PREFIX', sys.prefix)) / "bin"
            ]
            
            current_path = os.environ.get('PATH', '')
            for cuda_bin in cuda_bin_paths:
                if cuda_bin.exists() and str(cuda_bin) not in current_path:
                    os.environ['PATH'] = f"{cuda_bin}:{current_path}"
                    print(f"[INFO] Added to PATH: {cuda_bin}")
            
            return True
        else:
            print("[ERROR] CUDA toolkit installation failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] CUDA toolkit installation failed: {e}")
        return False

def check_and_install_gcc():
    """Check GCC version and install newer version if needed"""
    print("[INFO] Checking GCC version...")
    
    try:
        result = run_command(["gcc", "--version"], check=False)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"[INFO] Current GCC: {version_line}")
            
            # Extract version number
            import re
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_line)
            if match:
                major = int(match.group(1))
                if major >= 9:
                    print(f"[SUCCESS] GCC {major} is sufficient")
                    return True
                else:
                    print(f"[WARNING] GCC {major} is too old, need GCC 9+")
                    return install_newer_gcc()
        
        print("[WARNING] Could not determine GCC version")
        return install_newer_gcc()
        
    except Exception as e:
        print(f"[WARNING] GCC check failed: {e}")
        return install_newer_gcc()

def install_newer_gcc():
    """Install newer GCC via conda"""
    print("[INFO] Installing newer GCC via conda...")
    
    conda_cmd = find_conda_executable()
    if not conda_cmd:
        print("[ERROR] Cannot install GCC - conda not found")
        return False
    
    try:
        # Install GCC 9 from conda-forge
        cmd = [conda_cmd, "install", "-c", "conda-forge", 
               "gcc_linux-64=9", "gxx_linux-64=9", "-y"]
        
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            print("[SUCCESS] Newer GCC installed via conda")
            
            # Set up environment variables for the new compilers
            gcc_path = shutil.which("x86_64-conda-linux-gnu-gcc")
            gxx_path = shutil.which("x86_64-conda-linux-gnu-g++")
            
            if gcc_path and gxx_path:
                os.environ['CC'] = gcc_path
                os.environ['CXX'] = gxx_path
                print(f"[INFO] Set CC to: {gcc_path}")
                print(f"[INFO] Set CXX to: {gxx_path}")
                return True
            else:
                print("[WARNING] New GCC installed but not found in PATH")
                return False
        else:
            print("[ERROR] GCC installation failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] GCC installation failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"[INFO] Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 10:
        raise RuntimeError(f"Python {version.major}.{version.minor} is not supported. Need Python 3.10+")
    
    return f"{version.major}.{version.minor}"

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    try:
        import torch
        print(f"[INFO] PyTorch version: {torch.__version__}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"[INFO] CUDA version: {torch.version.cuda}")
            print(f"[INFO] GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return torch.__version__, torch.cuda.is_available()
    except ImportError:
        print("[WARNING] PyTorch not found")
        return None, False

def check_and_fix_torchvision():
    """Check torchvision compatibility with PyTorch and fix if needed"""
    try:
        import torch
        import torchvision
        
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        torchvision_version = tuple(map(int, torchvision.__version__.split('.')[:2]))
        
        print(f"[INFO] Checking torch/torchvision compatibility...")
        print(f"[INFO] PyTorch: {torch.__version__}")
        print(f"[INFO] TorchVision: {torchvision.__version__}")
        
        # Define compatible pairs (PyTorch major.minor -> TorchVision major.minor)
        compatible = {
            (2, 7): (0, 18),  # PyTorch 2.7 -> TorchVision 0.18
            (2, 6): (0, 17),
            (2, 5): (0, 17),
            (2, 4): (0, 16),
            (2, 3): (0, 16),
            (2, 2): (0, 15),
            (2, 1): (0, 15),
        }
        
        expected_tv_version = compatible.get(torch_version)
        
        if expected_tv_version is None:
            print(f"[WARNING] Unknown PyTorch version {torch_version}, skipping compatibility check")
            return False
        
        if torchvision_version[:2] != expected_tv_version:
            print(f"[ERROR] TorchVision {torchvision.__version__} is incompatible with PyTorch {torch.__version__}")
            print(f"[INFO] Expected TorchVision 0.{expected_tv_version[1]}.x for PyTorch {torch.__version__}")
            print(f"[INFO] Fixing torchvision installation...")
            
            # Reinstall compatible torchvision
            expected_tv_full = f"0.{expected_tv_version[1]}.1"
            cmd = [
                sys.executable, "-m", "pip", "install", "--force-reinstall",
                f"torchvision=={expected_tv_full}",
                "--index-url", "https://download.pytorch.org/whl/cu128"
            ]
            
            result = run_command(cmd, check=False)
            
            if result.returncode == 0:
                print(f"[SUCCESS] TorchVision fixed to {expected_tv_full}")
                return True
            else:
                print(f"[ERROR] Failed to fix TorchVision")
                return False
        else:
            print(f"[SUCCESS] PyTorch and TorchVision are compatible")
            return False
            
    except ImportError:
        print("[WARNING] Could not check torchvision compatibility")
        return False
    except Exception as e:
        print(f"[WARNING] Error checking torchvision: {e}")
        return False

def install_pytorch_if_needed():
    """Install PyTorch with CUDA support if not present or insufficient"""
    torch_version, has_cuda = check_pytorch()
    
    if torch_version is None or not has_cuda:
        print("[INFO] Installing PyTorch with CUDA support...")
        
        # Install PyTorch with CUDA 12.8 (adjust as needed)
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch==2.7.0", "torchvision==0.18.1", "torchaudio==2.7.0",
            "--index-url", "https://download.pytorch.org/whl/cu128"
        ]
        run_command(cmd)
        
        # Verify installation
        torch_version, has_cuda = check_pytorch()
        if not has_cuda:
            raise RuntimeError("PyTorch installation failed or CUDA not available")
    
    return torch_version

def setup_cuda_environment():
    """Set up CUDA environment variables, installing CUDA toolkit if needed"""
    print("[INFO] Setting up CUDA environment...")
    
    # Check if nvcc is available
    nvcc_path = shutil.which("nvcc")
    
    if nvcc_path:
        print(f"[INFO] Found nvcc at: {nvcc_path}")
    else:
        print("[WARNING] nvcc not found in PATH")
        print("[INFO] Attempting to install CUDA toolkit...")
        
        cuda_installed = install_cuda_toolkit()
        
        if cuda_installed:
            # Check again for nvcc after installation
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                print(f"[SUCCESS] Found nvcc after installation: {nvcc_path}")
            else:
                print("[WARNING] nvcc still not found after CUDA installation")
        else:
            print("[WARNING] CUDA toolkit installation failed")
    
    if nvcc_path:
        # Set CUDA_HOME
        cuda_home = str(Path(nvcc_path).parent.parent)
        os.environ['CUDA_HOME'] = cuda_home
        print(f"[INFO] Set CUDA_HOME to: {cuda_home}")
        
        # Get CUDA version
        try:
            result = run_command(["nvcc", "--version"])
            print(f"[INFO] NVCC version info:\n{result.stdout}")
        except:
            pass
    else:
        print("[WARNING] Proceeding without CUDA toolkit")
        print("[WARNING] PyTorch3D compilation may fail or use CPU-only")
    
    # Set compilation flags regardless (PyTorch might have CUDA even without nvcc)
    os.environ['FORCE_CUDA'] = '1'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '6.0;6.1;7.0;7.5;8.0;8.6;9.0'
    
    print(f"[INFO] FORCE_CUDA: {os.environ.get('FORCE_CUDA')}")
    print(f"[INFO] TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    
    return nvcc_path is not None

def install_pytorch3d():
    """Install PyTorch3D from GitHub"""
    print("[INFO] Installing PyTorch3D from GitHub...")
    print("[INFO] This may take 10-30 minutes depending on your system...")
    
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/facebookresearch/pytorch3d.git",
        "--verbose"
    ]
    
    run_command(cmd)

def verify_installation():
    """Verify PyTorch3D installation with GPU support"""
    print("[INFO] Verifying PyTorch3D installation...")
    # this is needed to check if compile with GPU support
    
    try:
        import torch
        import pytorch3d
        print(f"[SUCCESS] PyTorch3D version: {pytorch3d.__version__}")
        
        # Test GPU operations
        from pytorch3d.ops import sample_points_from_meshes
        from pytorch3d.structures import Meshes
        
        # Create simple mesh on GPU
        verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], 
                           dtype=torch.float32).cuda()
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long).cuda()
        mesh = Meshes(verts=[verts], faces=[faces])
        
        # Test the critical operation
        points = sample_points_from_meshes(mesh, 1000)
        print(f"[SUCCESS] GPU operations successful!")
        print(f"[SUCCESS] Sampled points shape: {points.shape}")
        print(f"[SUCCESS] Points device: {points.device}")
        
        # Test renderer import
        import pytorch3d.renderer
        print(f"[SUCCESS] PyTorch3D renderer import successful!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False

def main():
    print("="*60)
    print("PyTorch3D Installation for 3D Slicer")
    print("="*60)


    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"[INFO] Installing in environment: {env_name}")

    
    try:
        # Install/verify PyTorch
        torch_version = install_pytorch_if_needed()
        print(f"[SUCCESS] PyTorch {torch_version} ready")
        
        # Check and fix torchvision compatibility
        check_and_fix_torchvision()
        
        # Check and install GCC if needed
        gcc_ok = check_and_install_gcc()
        if gcc_ok:
            print("[SUCCESS] GCC environment ready")
        else:
            print("[WARNING] GCC setup failed - compilation might fail")
        
        # Set up CUDA environment (includes auto-installation)
        cuda_available = setup_cuda_environment()
        
        if cuda_available:
            print("[SUCCESS] CUDA environment ready")
        else:
            print("[WARNING] CUDA setup incomplete - GPU support may be limited")
                        
        # Install PyTorch3D
        install_pytorch3d()
        
        # Verify installation
        if verify_installation():
            print("\n" + "="*60)
            print("INSTALLATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Python: {sys.version}")
            print(f"PyTorch: {torch_version}")
            
            import pytorch3d
            print(f"PyTorch3D: {pytorch3d.__version__}")
            print("GPU support: ✓ Verified")
            print("CUDA toolkit: ✓ Installed" if cuda_available else "⚠ Limited")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("INSTALLATION COMPLETED WITH ISSUES")
            print("="*60)
            print("PyTorch3D was installed but GPU verification failed")
            print("Check the error messages above for details")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Installation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()