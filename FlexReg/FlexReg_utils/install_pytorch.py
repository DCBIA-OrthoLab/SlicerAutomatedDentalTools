#!/usr/bin/env python3
import subprocess
import logging
import sys
# ===== Logging Configuration =====
logger = logging.getLogger("FlexReg_install_pytorch")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def install_pytorch3d(pip_path):
    import torch

    pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
    version_str="".join([
        f"py3{sys.version_info.minor}_cu",
        torch.version.cuda.replace(".",""),
        f"_pyt{pyt_version_str}"
    ])
    cmd = [pip_path, 'install','--no-index', '--no-cache-dir' ,'pytorch3d', '-f', f'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html']

    result = subprocess.run(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
    logger.info(f"Result : {result.stdout}")
    logger.info(f"Error : {result.stderr}")
    logger.info("\nPyTorch3D installed in the environnement")


def main(pip_path):
    install_pytorch3d(pip_path)

if __name__ == "__main__":
    main(sys.argv[1])