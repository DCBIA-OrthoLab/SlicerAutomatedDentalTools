import sys
import torch
import subprocess


def main(pip_path):
    pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
    version_str="".join([
        f"py3{sys.version_info.minor}_cu",
        torch.version.cuda.replace(".",""),
        f"_pyt{pyt_version_str}"
    ])
    print(version_str)
    # command = [pip_path, 'install',f'--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html']
    command = [pip_path, 'install','--no-index', '--no-cache-dir' ,'pytorch3d', '-f', f'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html']
    result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
    print("Result : ",result.stdout)
    print("Error : ",result.stderr)
    

if __name__ == "__main__":
    

    print(sys.argv)
    main(sys.argv[1])

