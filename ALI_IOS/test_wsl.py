import os
import platform
import subprocess
import time



def check_miniconda():
    """Check if Miniconda is installed on WSL."""
    result = subprocess.run(["wsl", "test", "-d", "~/miniconda3"], capture_output=True)
    default_install_path = "~/miniconda3"
    return result.returncode == 0, default_install_path

def install_conda(default_install_path):
    """Install Miniconda on WSL."""
    system = platform.system()
    machine = platform.machine()
    miniconda_base_url = "https://repo.anaconda.com/miniconda/"

    if system == "Windows":
        filename = "Miniconda3-latest-Linux-x86_64.sh"
    else:
        raise NotImplementedError(f"Unsupported system: {system} {machine}")

    miniconda_url = miniconda_base_url + filename
    path_sh = os.path.join(default_install_path, "miniconda.sh")
    path_conda = os.path.join(default_install_path, "bin", "conda")

    subprocess.run(["wsl", "wget", "--continue", "--tries=3", miniconda_url, "-O", path_sh], capture_output=True)
    subprocess.run(["wsl", "chmod", "+x", path_sh], capture_output=True)

    try:
        print("Installing Miniconda...")
        subprocess.run(["wsl", "bash", path_sh, "-b", "-u", "-p", default_install_path], capture_output=True)
        subprocess.run(["wsl", "rm", "-rf", path_sh])
        subprocess.run(["wsl", path_conda, "init", "bash"])
        print("Miniconda installed successfully!")
    except:
        print("An error occurred during Miniconda installation.")
        return False

    return True

# is_installed, default_install_path = check_miniconda()
# if not is_installed:
#     install_conda(default_install_path)
# else:
#     print("Miniconda is already installed.")
      
      
# x,default_install_path=check_miniconda()   
# InstallConda(default_install_path)   

      
def is_miniconda_installed_on_wsl():
    try:
        # Exécute la commande "which conda" dans WSL
        activate_command = ["wsl","--","bash", "which", "conda"]
        result = subprocess.run(activate_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        # return result
        # Si un chemin est renvoyé, Miniconda est installé
        print("oui")
        print(result.stderr)
        print(result.returncode)
        if result:
            print(result.stdout)
            return True
    except subprocess.CalledProcessError:
        print("non")
        # La commande "which conda" renvoie une erreur si "conda" n'est pas trouvé
        pass

    return False


def install_miniconda_on_wsl():
    try:
        # Télécharge l'installateur Miniconda pour Linux
        subprocess.check_call(["wsl", "--","wget", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"])

        # Rend l'installateur exécutable
        subprocess.check_call(["wsl", "--","chmod", "+x", "Miniconda3-latest-Linux-x86_64.sh"])

        # Exécute l'installateur
        subprocess.check_call(["wsl","--","bash", "Miniconda3-latest-Linux-x86_64.sh", "-b"])

        # Supprime l'installateur après l'installation
        subprocess.check_call(["wsl","--", "rm", "Miniconda3-latest-Linux-x86_64.sh"])
        
        # subprocess.check_call(["wsl","--", "bash", "-c","\"echo 'export PATH=\"$HOME/miniconda3/bin:$PATH\"' >> ~/.bashrc\""])
        subprocess.check_call(["wsl", "--", "bash", "-c", "echo 'export PATH=\"$HOME/miniconda3/bin:$PATH\"' >> ~/.bashrc"])

        
        subprocess.check_call(["wsl", "--", "bash", "-c", "source ~/.bashrc"])
        
        command = f"wsl -- bash -c \"~/miniconda3/bin/pip install rpyc\""
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

        print("Miniconda a été installé avec succès sur WSL.")
        
    except subprocess.CalledProcessError as e:
        print(f"Une erreur s'est produite lors de l'installation de Miniconda sur WSL: {e}")

# install_miniconda_on_wsl()
    
# exist,default_install_path = check_miniconda()
# if not exist:
#     install_miniconda_on_wsl()

# result = subprocess.run(["wsl","--","~/miniconda3/bin/python3","~/miniconda3/bin/conda","--version"],capture_output=True)
# subprocess.run(['wsl', '--', 'bash', '-i', '-c', '~/miniconda3/bin/conda --version'])


# print(result.stdout)


def checkEnvConda(name:str,default_install_path:str):
      path_conda = "~/miniconda3/bin/conda"
      path_python = "~/miniconda3/bin/python3"
      command_to_execute = ["wsl","--","~/miniconda3/bin/python3","~/miniconda3/bin/conda","info","--envs"]
      

      result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
      if result.returncode == 0:
            output = result.stdout
            print("output : ",output)
            env_lines = output.strip().split("\n")
            for line in env_lines:
                env_name = os.path.basename(line)
                if env_name == name:
                    print('Env conda exist')
                    return True  # L'environnement Conda existe déjà
          
      print("Env conda doesn't exist")
      return False  # L'environnement Conda n'existe pas
  
# name = "aliIOSCondaCLI"
# env = checkEnvConda(name,default_install_path)


def createCondaEnv(name:str) :
      path_conda = "~/miniconda3/bin/conda"
      path_python = "~/miniconda3/bin/python3"
      
      default_path = "~/miniconda3"
    
    #   command_to_execute = [python_path, "-m", "conda", "create", "--name", name, "python=3.9", "-y"]
      command_to_execute = ["wsl","--","~/miniconda3/bin/conda", "create", "-y","-n", name, "python=3.9","pip","numpy-base"]
      
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
        
      
      result = subprocess.run(f"{path_conda} list pip", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
      print(result.stdout)

      path_pip=f"~/miniconda3/envs/{name}/bin/pip"
      path_activate = f"~/miniconda3/bin/activate"
      install_commands = [
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113\"",
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install monai==0.7.0\"",
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113\"",
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install fvcore\"",
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html\"",
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install rpyc\"",
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install vtk\"",
      f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install scipy\""
      ]


    #   Exécution des commandes d'installation
      for command in install_commands:
          print("command : ",command)
          result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
          if result.returncode == 0:
              print(f"Successfully executed: {command}")
            #   print(result.stdout)
          else:
              print(f"Failed to execute: {command}")
            #   print(result.stderr)

      if result.returncode == 0:
          print("Environment created successfully:", result.stdout)
      else:
          print("Failed to create environment:", result.stderr)
          
# env = checkEnvConda(name,default_install_path)
# print("env1 : ",env)       
          
# if not env : 
#     createCondaEnv(name)
    
# env = checkEnvConda(name,default_install_path)
# print("env2 : ",env)

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
    
    




def windows_to_linux_path(windows_path):
    # Supprime le caractère de retour chariot
    windows_path = windows_path.strip()

    # Remplace les backslashes par des slashes
    path = windows_path.replace('\\', '/')

    # Remplace le lecteur par '/mnt/lettre_du_lecteur'
    if ':' in path:
        drive, path_without_drive = path.split(':', 1)
        path = "/mnt/" + drive.lower() + path_without_drive

    return path


# current_file_path = os.path.abspath(__file__)

#     # Répertoire contenant le script en cours d'exécution
# current_directory = os.path.dirname(current_file_path)
# python_path = "~/miniconda3/bin/python"
# lien_path = os.path.join(current_directory,"test.py")
# lien_path = windows_to_linux_path(lien_path)
# command = f"wsl -- bash -c \"{python_path} {lien_path}\""
# print("command : ",command)
# result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

# print("oui")
# print(result.stdout)