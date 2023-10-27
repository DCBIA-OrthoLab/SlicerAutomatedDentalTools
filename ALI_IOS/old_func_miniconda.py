import os
import subprocess
import platform
import sys
import time
import importlib.util

def is_module_installed(module_name):
    """Vérifie si un module est installé"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def install_module(module_name):
    """Installe un module via pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])

if not is_module_installed("rpyc"):
    install_module("rpyc")

# Maintenant, vous pouvez importer rpyc et l'utiliser
import rpyc

def createCondaEnv(name:str,default_install_path:str,path_conda:str,path_activate:str) :
      python_path =  os.path.join(default_install_path,"python")
      print("python path ::::", python_path)
    
      command_to_execute = [python_path, "-m", "conda", "create", "--name", name, "python=3.9", "-y"]
      
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
        
      path_conda = os.path.join(default_install_path,"Scripts","conda")
      path_pip = os.path.join(default_install_path,"envs",name,"Scripts","pip")
      
      activate_command = f"conda {path_activate} {name}"
      subprocess.run(activate_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
      
      result = subprocess.run(f"{path_conda} list pip", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
      print(result.stdout)

      install_commands = [
      f"{path_pip} install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113",
    #   f"{path_pip} install monai==0.7.0",
      f"{path_pip} install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113",
    #   f"{path_pip} install fvcore",
      f"{path_pip} install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html",   
      f"{path_pip} install rpyc",
    #   f"{path_pip} install vtk",
    #   f"{path_pip} install scipy"
      ]


      # Exécution des commandes d'installation
      for command in install_commands:
          print("command : ",command)
          result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
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
      path_conda = os.path.join(default_install_path,"_conda")
      command_to_execute = [path_conda, "info", "--envs"]
      

      result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
      if result.returncode == 0:
            output = result.stdout
            env_lines = output.strip().split("\n")
            for line in env_lines:
                env_name = os.path.basename(line)
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





def setup(default_install_path,args):
    
    miniconda,default_install_path = checkMiniconda()
    path_conda = os.path.join(default_install_path,"_conda")
    path_conda2 = os.path.join(default_install_path,"Scripts","conda")
    path_conda3 = os.path.join(default_install_path,"Scripts","conda.exe")
    path_activate = os.path.join(default_install_path, "Scripts", "activate")
    
    
    name = "aliIOSCondaCli"
    if not checkEnvConda(name,default_install_path):
        createCondaEnv(name,default_install_path,path_conda2,path_activate)

    command_to_execute = [path_conda, "info", "--envs"]
    print(f"commande de verif : {command_to_execute}")

    result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        output = result.stdout.decode("utf-8")
        print("Environnements Conda disponibles :\n", output)

    # activate_command = f"{path_conda2} activate {name}"
    # subprocess.run(activate_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    
    activate_command = f"conda {path_activate} {name}"
    subprocess.run(activate_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

    # Lister les packages installés dans l'environnement
    path_env = os.path.join(default_install_path,"envs",name)
    print("List les packages du nouvel environment")
    list_packages_command = f"{path_conda2} list -p {path_env}"
    result = subprocess.run(list_packages_command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if result.returncode == 0:
        print("List of installed packages:")
        print(result.stdout)
    else:
        print(f"Failed to list installed packages: {result.stderr}")

    # call(default_install_path,args,name)


def call(default_install_path,args,name):

    activate_env = os.path.join(default_install_path, "bin", "activate")
    python_executable = os.path.join(default_install_path, "envs",name,"python")  # Modifiez selon votre système d'exploitation et votre installation


    current_file_path = os.path.abspath(__file__)

    # Répertoire contenant le script en cours d'exécution
    current_directory = os.path.dirname(current_file_path)

    path_activate = os.path.join(default_install_path, "Scripts", "activate")
    activate_command = f"conda {path_activate} {name}"
    subprocess.run(activate_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    # Chemin absolu du fichier souhaité qui est à côté du script en cours d'exécution
    path_server = os.path.join(current_directory, 'server.py')
    command = f"{python_executable} {path_server}"

    # Start server
    server_process = subprocess.Popen(command, shell=True)
    
    # To be sure the server start
    time.sleep(2)
    
    conn = rpyc.connect("localhost", 18817)
    # wait_for_server_ready(conn)
    time.sleep(2)
    conn.root.running(args)

    # Stop process
    result = conn.root.stop()
    if result == "DISCONNECTING":
        conn.close()

    print("on a ferme le server")



if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "setup":

        args = {
        "input": sys.argv[3],
        "dir_models": sys.argv[4],
        "lm_type": sys.argv[5].split(" "),
        "teeth": sys.argv[6].split(" "),
        "save_in_folder": sys.argv[7] == "true",
        "output_dir": sys.argv[8],

        "image_size": 224,
        "blur_radius": 0,
        "faces_per_pixel": 1,
        # "sphere_radius": 0.3,
    }
        setup(sys.argv[2], args)