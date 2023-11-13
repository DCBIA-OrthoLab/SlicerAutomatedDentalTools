import os
import subprocess
import platform
import sys
import time
import importlib.util

def checkMinicondaWsl():
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


def install_miniconda_on_wsl():
    try:
        # Télécharge l'installateur Miniconda pour Linux
        subprocess.check_call(["wsl", "--","wget", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"])

        # Rend l'installateur exécutable
        subprocess.check_call(["wsl", "--","chmod", "+x", "Miniconda3-latest-Linux-x86_64.sh"])

        # Exécute l'installateur
        subprocess.check_call(["wsl","--","bash", "Miniconda3-latest-Linux-x86_64.sh", "-b","-p", "~/miniconda3"])

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
        
        
def checkEnvCondaWsl(name:str):
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

def is_ubuntu_installed():
    result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
    output = result.stdout.encode('utf-16-le').decode('utf-8')
    clean_output = output.replace('\x00', '')  # Enlève tous les octets null

    print("clean_output :", clean_output)

    return 'Ubuntu' in clean_output

def run_command_with_input_and_delays(command, input_data, delay):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in input_data:
        process.stdin.write(line + '\n')
        process.stdin.flush()  # Assurez-vous que les données sont bien envoyées
        time.sleep(delay)  # Attendez pendant le délai spécifié

    return process




def setup(default_install_path,args):
            
    
    miniconda,default_install_path = checkMinicondaWsl()
    if not miniconda:
        install_miniconda_on_wsl()
        
    
    name = "aliIOSCondaCli"
    if not checkEnvCondaWsl(name):
        createCondaEnv(name)

    
    current_file_path = os.path.abspath(__file__)

    # Répertoire contenant le script en cours d'exécution
    current_directory = os.path.dirname(current_file_path)
    # python_path = "~/miniconda3/bin/python"
    python_path = os.path.join(default_install_path,"python")
    lien_path = os.path.join(current_directory,"link.py")
    lien_path = windows_to_linux_path(lien_path)
   
    home_directory = subprocess.check_output(['wsl', 'echo', '$HOME']).decode().strip()

    # Remplacer le ~ dans python_path avec le répertoire personnel
    python_path = python_path.replace('~', home_directory)

    command = f"wsl -- bash -c \"{python_path} {lien_path} {sys.argv[3]} {sys.argv[4]} {sys.argv[5]} {sys.argv[6]} {sys.argv[7]} {sys.argv[8]} {name}\""
    command_to_execute = ["wsl", "--", "bash", "-c"]
    command_inside_wsl = [python_path, lien_path] + sys.argv[3:9] +[name]
    command_string_inside_wsl = " ".join(['"' + arg + '"' for arg in command_inside_wsl])
    command_to_execute.append(command_string_inside_wsl)
    command = command_to_execute
    
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

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