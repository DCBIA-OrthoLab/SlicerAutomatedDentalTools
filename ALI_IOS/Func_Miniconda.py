import os
import subprocess
import platform
import sys

def createCondaEnv(name:str,default_install_path:str,path_conda:str,path_activate:str) :
      python_path = "/home/luciacev/miniconda3/bin/python3"
      command_to_execute = [python_path,path_conda, "create", "--name", name, "python=3.9", "-y"]  
      print(f"command_to_execute : {command_to_execute}")
      result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


      if result.returncode != 0:
        print(f"Error creating the environment. Return code: {result.returncode}")
        print("result.stdout : ","*"*150)
        print(result.stdout)
        print("result.stderr : ","*"*150)
        print(result.stderr)
      else:
        print("Environment created successfully.")

      install_commands = [
      f"source {path_activate} {name} && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113",
      f"source {path_activate} {name} && pip install monai==0.7.0",
      f"source {path_activate} {name} && pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113",
      f"source {path_activate} {name} && pip install fvcore",
      f"source {path_activate} {name} && pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html",   
      f"source {path_activate} {name} && pip install rpyc",
      ]


      # Exécution des commandes d'installation
      for command in install_commands:
          print("command : ",command)
          result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',  executable="/bin/bash")
          if result.returncode == 0:
              print(f"Successfully executed: {command}")
              print(result.stdout)
          else:
              print(f"Failed to execute: {command}")
              print(result.stderr)

      if result.returncode == 0:
          print("Environment created successfully:", result.stdout)
      else:
          print("Failed to create environment:", result.stderr)


def checkEnvConda(name:str,default_install_path:str):
      path_conda = os.path.join(default_install_path,"bin","conda")
      command_to_execute = [path_conda, "info", "--envs"]

      result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      if result.returncode == 0:
          output = result.stdout.decode("utf-8")
          env_lines = output.strip().split("\n")
          for line in env_lines:
              env_name = line.split()[0].strip()
              if env_name == name:
                  print('Env conda exist')
                  return True  # L'environnement Conda existe déjà
          
      print("Env conda doesn't exist")
      return False  # L'environnement Conda n'existe pas


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
      miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh"
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
      

def appel(args1,args2):
    print(args1*150)
    miniconda,default_install_path = checkMiniconda()
    path_conda = os.path.join(default_install_path,"bin","conda")
    path_activate = os.path.join(default_install_path, "bin", "activate")
    success_install = miniconda

    if not miniconda : 
            print("appelle InstallConda")
            success_install = InstallConda(default_install_path)

    if success_install:
        print("miniconda installed")

        name = "aliIOSCondaCli"
        if not checkEnvConda(name,default_install_path):
            createCondaEnv(name,default_install_path,path_conda,path_activate)

        command_to_execute = [path_conda, "info", "--envs"]
        print(f"commande de verif : {command_to_execute}")

        result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            output = result.stdout.decode("utf-8")
            print("Environnements Conda disponibles :\n", output)

        # Lister les packages installés dans l'environnement
        print("List les packages du nouvel environment")
        list_packages_command = f"source {path_activate} {name} && conda list"
        result = subprocess.run(list_packages_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, executable="/bin/bash")
        if result.returncode == 0:
            print("List of installed packages:")
            print(result.stdout)
        else:
            print(f"Failed to list installed packages: {result.stderr}")
            
    print(args2*150)

if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "appel":
        appel(sys.argv[2], sys.argv[3])