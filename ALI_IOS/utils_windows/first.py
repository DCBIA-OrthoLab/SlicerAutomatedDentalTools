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


def install_miniconda_on_wsl():
    '''
    Install miniconda3 on wsl
    '''
    try:
        # Download executable of miniconda3 for linux
        subprocess.check_call(["wsl", "--","wget", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"])

        # Makes the installer executable
        subprocess.check_call(["wsl", "--","chmod", "+x", "Miniconda3-latest-Linux-x86_64.sh"])

        # Execution of the installer
        subprocess.check_call(["wsl","--","bash", "Miniconda3-latest-Linux-x86_64.sh", "-b","-p", "~/miniconda3"])

        # Delete installer
        subprocess.check_call(["wsl","--", "rm", "Miniconda3-latest-Linux-x86_64.sh"])

        subprocess.check_call(["wsl", "--", "bash", "-c", "echo 'export PATH=\"$HOME/miniconda3/bin:$PATH\"' >> ~/.bashrc"])


        subprocess.check_call(["wsl", "--", "bash", "-c", "source ~/.bashrc"])

        command = f"wsl -- bash -c \"~/miniconda3/bin/pip install rpyc\""
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

        print("Miniconda has been successfully installed on WSL.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred when installing Miniconda on WSL: {e}")


def checkEnvCondaWsl(name:str):
    '''
    Check if env conda exist on wsl
    '''
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
    '''
    Create the new env to run ali_ios on wsl
    It will install all the require librairie
    '''
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
    '''
    Convert a windows path to a path that wsl can read
    '''
    windows_path = windows_path.strip()

    path = windows_path.replace('\\', '/')

    if ':' in path:
        drive, path_without_drive = path.split(':', 1)
        path = "/mnt/" + drive.lower() + path_without_drive

    return path



def write_txt(message):
    '''
    Write in a temporary file
    '''
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_path,"tempo.txt")
    path_parts = os.path.split(file_path)
    new_dir = path_parts[0].replace(os.path.join('ALI_IOS','utils_windows'), 'ALI')
    new_path = os.path.join(new_dir, path_parts[1])

    with open(new_path, 'a') as file:
        file.write(message + '\n')  # Écrire le message suivi d'une nouvelle ligne

def setup(default_install_path,args):


    miniconda,default_install_path = checkMinicondaWsl()
    if not miniconda:
        write_txt("Installing miniconda3 on wsl, this task can take a few minutes")
        install_miniconda_on_wsl()


    name = "aliIOSCondaCli"
    if not checkEnvCondaWsl(name):
        write_txt("Creating the new environement, this task can take a few minutes")
        createCondaEnv(name)


    current_file_path = os.path.abspath(__file__)

    # Répertoire contenant le script en cours d'exécution
    current_directory = os.path.dirname(current_file_path)
    python_path = "~/miniconda3/bin/python" #in wsl
    lien_path = os.path.join(current_directory,"in_wsl.py")
    lien_path = windows_to_linux_path(lien_path)

    home_directory = subprocess.check_output(['wsl', 'echo', '$HOME']).decode().strip()

    python_path = python_path.replace('~', home_directory)

    write_txt("Process the file(s), creation of landmark(s)")
    # command = f"wsl -- bash -c \"{python_path} {lien_path} {sys.argv[3]} {sys.argv[4]} {sys.argv[5]} {sys.argv[6]} {sys.argv[7]} {sys.argv[8]} {name}\""

    command_to_execute = ["wsl", "--", "bash", "-c"]
    command_inside_wsl = [python_path, lien_path] + sys.argv[3:9] +[name]
    command_string_inside_wsl = " ".join(['"' + arg + '"' for arg in command_inside_wsl])
    command_to_execute.append(command_string_inside_wsl)
    command = command_to_execute #command to call in_wsl.py with python in miniconda3 on wsl and give it the argument

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