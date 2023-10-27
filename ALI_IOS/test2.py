import platform
import subprocess
import os
import urllib.request
import shutil
import time
import sys
print(sys.executable)
try:
    import rpyc
except :
    subprocess.run("pip install --upgrade pip", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    print("Le module rpyc n'est pas installé. Installation en cours...")
    python_path = sys.executable  # Obtenez le chemin vers l'interpréteur Python en cours d'exécution
    install_command = [python_path, "-m", "pip", "install", "rpyc"]
    result = subprocess.run(install_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

    if result.returncode == 0:
        import rpyc
        print("Le module rpyc a été installé avec succès.")
    else:
        print("Échec de l'installation du module rpyc.")
        print(result.stderr)

def checkMiniconda():
    print("je suis dans checkminiconda")
    user_home = os.path.expanduser("~")
    default_install_path = os.path.join(user_home, "miniconda3")
    return(os.path.exists(default_install_path),default_install_path)



def InstallConda(default_install_path):
    system = platform.system()
    machine = platform.machine()

    # Define the Anaconda installer URL for Windows
    if system == "Windows":
        if machine.endswith("64"):
            filename = "Miniconda3-latest-Windows-x86_64.exe"
        else:
            filename = "Miniconda3-latest-Windows-x86.exe"
        miniconda_url = f"https://repo.anaconda.com/miniconda/{filename}"
        print(f"Selected Miniconda installer file: {filename}")

        print(f"Full download URL: {miniconda_url}")

        print(f"Default Miniconda installation path: {default_install_path}")


        path_exe = os.path.join(os.path.expanduser("~"), "tempo")
       
        os.makedirs(path_exe, exist_ok=True)
        # Define paths for the installer and conda executable
        path_installer = os.path.join(path_exe, filename)
        path_conda = os.path.join(default_install_path, "Scripts", "conda.exe")
        
        

        print(f"path_installer : {path_installer}")
        print(f"path_conda : {path_conda}")

        if not os.path.exists(default_install_path):
            os.makedirs(default_install_path)

        try:
            # Download the Anaconda installer
            urllib.request.urlretrieve(miniconda_url, path_installer)
            print("Installer downloaded successfully.")
            print("Installing Miniconda...")
            
            # Run the Anaconda installer with silent mode
            print("path_installer : ",path_installer)
            print("default_install_path : ",default_install_path)
            # subprocess.run('start /wait "C:\\Users\\luciacev.UMROOT\\miniconda3\\Miniconda3-latest-Windows-x86_64.exe" /InstallationType=JustMe /S /D=C:\\Users\\luciacev.UMROOT\\miniconda3', shell=True)
            # Commande PowerShell
            # Chemin vers l'installateur Miniconda pour Windows
            # path_installer = "C:\\Users\\luciacev.UMROOT\\oui\\Miniconda3-latest-Windows-x86_64.exe"
            path_miniconda = os.path.join(default_install_path,"miniconda")

            # Commande pour une installation silencieuse avec Miniconda
            install_command = f'"{path_installer}" /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D={default_install_path}'

            # Exécutez la commande d'installation
            subprocess.run(install_command, shell=True)


            # os.remove(path_installer)  # Remove the installer file after installation
            print("bonjour")
            subprocess.run(f"{path_conda} init cmd.exe", shell=True)
            print("Miniconda installed successfully.")
            
            try:
                shutil.rmtree(path_exe)
                print(f"Dossier {path_exe} et son contenu ont été supprimés avec succès.")
            except Exception as e:
                print(f"Une erreur s'est produite lors de la suppression du dossier : {str(e)}")
            return True
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False
    else:
        print("Unsupported system. This code is intended for Windows.")
        return False

      
    
# exist,defaut_install_path = checkMiniconda()

# print(exist,defaut_install_path)

# if not exist : 
    # InstallConda(defaut_install_path)
    

##on arrive a installer miniconda au dessus

    


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
      f"{path_pip} install monai==0.7.0",
      f"{path_pip} install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113",
      f"{path_pip} install fvcore",
      f"{path_pip} install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html",   
      f"{path_pip} install rpyc",
      f"{path_pip} install vtk",
      f"{path_pip} install scipy"
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




def setup(default_install_path):

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
        
    call(default_install_path,name)
    
    
def wait_for_server_ready(conn):
    while not conn.root.is_ready():
        print("En attente du serveur...")
        time.sleep(1)
    
    
def call(default_install_path,name):
    
    activate_env = os.path.join(default_install_path, "bin", "activate")
    python_executable = os.path.join(default_install_path, "envs",name,"python")  # Modifiez selon votre système d'exploitation et votre installation


    current_file_path = os.path.abspath(__file__)

    # Répertoire contenant le script en cours d'exécution
    current_directory = os.path.dirname(current_file_path)

    path_activate = os.path.join(default_install_path, "Scripts", "activate")
    activate_command = f"conda {path_activate} {name}"
    subprocess.run(activate_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    # Chemin absolu du fichier souhaité qui est à côté du script en cours d'exécution
    path_server = os.path.join(current_directory, 'server2.py')
    command = f"{python_executable} {path_server}"

    # Start server
    server_process = subprocess.Popen(command, shell=True)
    
    # To be sure the server start
    time.sleep(2)
    
    conn = rpyc.connect("localhost", 18817)
    # wait_for_server_ready(conn)
    time.sleep(2)


    print("on lui demande le resultat")
    x=6
    result=conn.root.running(x)
    print(f"{x} au carre = {result}")

    # Stop process
    
    # server_process.terminate()
    # server_process.wait()
    
    result = conn.root.stop()
    if result == "DISCONNECTING":
        conn.close()

    print("on a ferme le server")
    
# setup(defaut_install_path)