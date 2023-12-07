import subprocess
import platform
import os
import sys

# subprocess.run('dentalmodelseg --vtk T1_nosegmented.vtk --out /home/luciacev/Documents/Gaelle/Data/CrownSegmentation/vtk_non_seg/T1_nosegmented.vtk --mount_point /home/luciacev/Documents/Gaelle/Data/CrownSegmentation/vtk_non_seg --overwrite True',check=True, shell=True)

def checkMiniconda():
    print("je suis dans checkminiconda")
    user_home = os.path.expanduser("~")
    default_install_path = os.path.join(user_home, "miniconda3")
    return(os.path.exists(default_install_path),default_install_path)

def InstallConda(default_install_path):
      system = platform.system()
      machine = platform.machine()

      miniconda_base_url = "https://repo.anaconda.com/miniconda/"

      # Construct the filename based on the operating system and architecture
      if system == "Windows":
          if machine.endswith("64"):
              filename = "Miniconda3-latest-Windows-x86_64.exe"
          else:
              filename = "Miniconda3-latest-Windows-x86.exe"
      elif system == "Linux":
          if machine == "x86_64":
              filename = "Miniconda3-latest-Linux-x86_64.sh"
          else:
              filename = "Miniconda3-latest-Linux-x86.sh"
      else:
          raise NotImplementedError(f"Unsupported system: {system} {machine}")

      print(f"Selected Miniconda installer file: {filename}")

      miniconda_url = miniconda_base_url + filename
    #   miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh"
    #   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      print(f"Full download URL: {miniconda_url}")

      print(f"Default Miniconda installation path: {default_install_path}")

      path_sh = os.path.join(default_install_path,"miniconda.sh")
      path_conda = os.path.join(default_install_path,"bin","conda")

      print(f"path_sh : {path_sh}")
      print(f"path_conda : {path_conda}")

      if not os.path.exists(default_install_path):
          os.makedirs(default_install_path)



      subprocess.run(f"mkdir -p {default_install_path}",capture_output=True, shell=True)
      subprocess.run(f"wget --continue --tries=3 {miniconda_url} -O {path_sh}",capture_output=True, shell=True)
      subprocess.run(f"chmod +x {path_sh}",capture_output=True, shell=True)

      try:
          print("Le fichier est valide.")
          subprocess.run(f"bash {path_sh} -b -u -p {default_install_path}",capture_output=True, shell=True)
          subprocess.run(f"rm -rf {path_sh}",shell=True)
          subprocess.run(f"{path_conda} init bash",shell=True)
          # subprocess.run(f"{path_conda} init zsh",shell=True)
          return True
      except:
          print("Le fichier est invalide.")
          return (False)
      


def run(args):
    print(args)
    miniconda,default_install_path = checkMiniconda()
    
    if not miniconda :
        InstallConda()

    python_path = os.path.join(default_install_path,"bin","python") #python path in miniconda3
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    path_func_miniconda = os.path.join(current_directory,'seg2.py') #Next files to call

    command_to_execute = [python_path,path_func_miniconda,args['file'],args['out'],args['overwrite'],args['mount_point'],args['name_env']]  

    env = dict(os.environ)
    if 'PYTHONPATH' in env:
        del env['PYTHONPATH']
    if 'PYTHONHOME' in env:
        del env['PYTHONHOME']

    result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,env=env)


    if result.returncode != 0:
        print(f"Error creating the environment. Return code: {result.returncode}")
        print("result.stdout : ","*"*150)
        print(result.stdout)
        print("result.stderr : ","*"*150)
        print(result.stderr)
    else:
        print(result.stdout)
        print("Environment created successfully.")



if __name__ == "__main__":
    args = {
    "file": sys.argv[1],
    "out": sys.argv[2],
    "overwrite": sys.argv[3],
    "mount_point": sys.argv[4],
    "name_env":sys.argv[5]
    
    }
    
    run(args)