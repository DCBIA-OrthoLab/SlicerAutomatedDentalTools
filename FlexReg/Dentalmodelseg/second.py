'''
This file is running on the python of miniconda3. It's checking the environnement and create it if not. 
It's running the dentalmodelseg in the environnement created.
'''

import subprocess
import platform
import os
import sys

def checkEnvCondaWsl(name:str):
    '''
    Check if env conda exist on wsl
    '''
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


def checkEnvConda(name:str,default_install_path:str)->bool:
    '''
    check if the environnement 'name' exist in miniconda3. return bool
    '''

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
                return True 

    print("Env conda doesn't exist")
    return False  

def createCondaEnvWsl(name:str) :
    '''
    Create the new env to run shapeaxi (dentalmodelseg) on wsl
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
    f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install shapeaxi\""
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
       

    command = f"wsl -- bash -c \"~/miniconda3/envs/{name}/bin/python -c "
    command = command + " 'import sys; import torch; pyt_version_str = torch.__version__.split(\\\"+\\\")[0].replace(\\\".\\\", \\\"\\\"); version_str = \\\"\\\".join([f\\\"py3{sys.version_info.minor}_cu\\\", torch.version.cuda.replace(\\\".\\\", \\\"\\\"), f\\\"_pyt{pyt_version_str}\\\"]); print(version_str)'\""

    result = subprocess.run(command, capture_output=True, text=True,errors='ignore')
    
    command = f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{result.stdout}/download.html\""
    command = command.replace('\n','')
    print("command : ",command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if result.returncode == 0:
        print(f"Successfully execute second command : {command}")
        print(result.stdout)
    else:
        print(f"Failed to execute second command : {command}")
        print(result.stderr)

    
    


def createCondaEnv(name:str,default_install_path:str,path_conda:str,path_activate:str)->None :
    ''''
    create the environnement 'name' -> install shapeaxi and the good version of pytorch3d on it
    '''

    command_to_execute = [path_conda, "create", "--name", name, "python=3.9", "-y"]  
    print(f"command_to_execute : {command_to_execute}")
    result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    install_commands = [
    f"source {path_activate} {name} && pip install shapeaxi"
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

def checkUpgrade(name:str,path_activate:str)->None:
    '''
    do the upgrade for shapeaxi on the environnement 'name'
    '''

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
            
def checkUpgradeWsl(name:str)->None:
    '''
    do the upgrade for shapeaxi on the environnement 'name'
    '''
    path_pip=f"~/miniconda3/envs/{name}/bin/pip"
    path_activate = f"~/miniconda3/bin/activate"
    install_commands = [
    f"wsl -- bash -c \"source {path_activate} {name} && {path_pip} install shapeaxi\""
    ]

    for command in install_commands:
        print("command : ",command)
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print(f"Successfully first command executed : {command}")
            print(result.stdout)
        else:
            print(f"Failed to execute first command : {command}")
            print(result.stderr)
    
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

def run(args):
    '''
    main function, checking if the environnement is created and running shapeaxi on the environnement
    '''
    system = platform.system()
    print("ON EST DANS SECOND.PY")
    if system=="Windows":
        minicondawsl, default_install_path = checkMinicondaWsl()
        if not minicondawsl :
            install_miniconda_on_wsl()
            print("ON INSTALL MINICONDA SUR WSL")
            
        env_exist_wsl = checkEnvCondaWsl(args['name_env'])
        if not env_exist_wsl :
            createCondaEnvWsl(args['name_env'])
            
        checkUpgradeWsl(args['name_env'])
        
        name = args['name_env']
        file = args['file']
        out = windows_to_linux_path(args['file'])
        mount_point = windows_to_linux_path(args['mount_point'])
        overwrite = args['overwrite']
        path_activate = f"~/miniconda3/bin/activate"
        command = f"wsl -- bash -c \"source {path_activate} {name} && dentalmodelseg --vtk {file} --out {out} --mount_point {mount_point} --overwrite {overwrite}\""
        print("DERNIERE LIGNE DROITEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print("command : ",command)
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            print(f"Successfully first command executed : {command}")
            print(result.stdout)
        else:
            print(f"Failed to execute first command : {command}")
            print(result.stderr)
        
        
            
    else : 
        user_home = os.path.expanduser("~")
        default_install_path = os.path.join(user_home, "miniconda3")
        path_activate = f"~/miniconda3/bin/activate"

        env_exist = checkEnvConda(args['name_env'],default_install_path)

        if not env_exist :
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