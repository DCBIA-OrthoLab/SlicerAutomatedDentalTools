import subprocess
import platform
import os
import sys


def checkEnvConda(name:str,default_install_path:str):
    path_conda = os.path.join(default_install_path,"bin","conda")
    command_to_execute = [path_conda, "info", "--envs"]

    result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        output = result.stdout.decode("utf-8")
        env_lines = output.strip().split("\n")

        for line in env_lines:
            env_name = line.split()[0].strip()
            print("name : ",name,"     env_name : ",env_name)
            if env_name == name:
                print('Env conda exist')
                return True  # L'environnement Conda existe déjà

    print("Env conda doesn't exist")
    return False  # L'environnement Conda n'existe pas


def createCondaEnv(name:str,default_install_path:str,path_conda:str,path_activate:str) :
    command_to_execute = [path_conda, "create", "--name", name, "python=3.9", "-y"]  
    print(f"command_to_execute : {command_to_execute}")
    result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    install_commands = [
    f"source {path_activate} {name} && pip install shapeaxi"
    ]


    # Exécution des commandes d'installation
    for command in install_commands:
        print("command : ",command)
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',  executable="/bin/bash")
        if result.returncode == 0:
            print(f"Successfully first command executed : {command}")
            print(result.stdout)
        else:
            print(f"Failed to execute first command : {command}")
            print(result.stderr)


    python_script = """
import sys
import torch
pyt_version_str = torch.__version__.split('+')[0].replace('.', '')
version_str = ''.join([
    f'py3{sys.version_info.minor}_cu',
    torch.version.cuda.replace('.', ''),
    f'_pyt{pyt_version_str}'
])
print(version_str)
"""
    python_executable = os.path.join(default_install_path,"envs",name,"bin","python")
    result = subprocess.run([python_executable, "-c", python_script], capture_output=True, text=True,errors='ignore')

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Output:\n{result.stdout}")

        command = f"source {path_activate} {name} && pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{result.stdout}/download.html"
        command = command.replace('\n','')
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',  executable="/bin/bash")
        if result.returncode == 0:
            print(f"Successfully execute second command : {command}")
            print(result.stdout)
        else:
            print(f"Failed to execute second command : {command}")
            print(result.stderr)

def checkUpgrade(name:str,path_activate:str):

    install_commands = [
    f"source {path_activate} {name} && pip install --upgrade shapeaxi"
    ]

    for command in install_commands:
        print("command : ",command)
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',  executable="/bin/bash")
        if result.returncode == 0:
            print(f"Successfully first command executed : {command}")
            print(result.stdout)
        else:
            print(f"Failed to execute first command : {command}")
            print(result.stderr)
    

def run(args):
    print("ON EST DANS SEG2")
    print(args)

    user_home = os.path.expanduser("~")
    default_install_path = os.path.join(user_home, "miniconda3")
    path_activate = f"~/miniconda3/bin/activate"

    env_exist = checkEnvConda(args['name_env'],default_install_path)

    if not env_exist :
        print("666"*150)
        path_conda = os.path.join(default_install_path,"bin","conda")
        createCondaEnv(args['name_env'],default_install_path,path_conda,path_activate)


    checkUpgrade(args['name_env'],path_activate)

    command = f"bash -c 'source {path_activate} {args['name_env']} && dentalmodelseg --vtk {args['file']} --out {args['out']} --mount_point {args['mount_point']} --overwrite {args['overwrite']}'"
    print("command : ",command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore',  executable="/bin/bash")
    if result.returncode == 0:
        print(f"Successfully executed: {command}")
        print(result.stdout)
    else:
        print(f"Failed to execute: {command}")
        print(result.stderr)



if __name__ == "__main__":
    args = {
    "file": sys.argv[1],
    "out": sys.argv[2],
    "overwrite": sys.argv[3] == "true",
    "mount_point": sys.argv[4],
    "name_env":sys.argv[5]
    
    }
    
    run(args)