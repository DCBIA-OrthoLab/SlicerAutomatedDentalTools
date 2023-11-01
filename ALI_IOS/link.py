import sys
import os
import time
import subprocess
import rpyc

def call(name,args):
    home_dir = os.path.expanduser("~")
    default_install_path = "~/miniconda3"
    path_activate = "~/miniconda3/bin/activate"
    python_path_env = f"{home_dir}/miniconda3/envs/{name}/bin/python"
    

    current_file_path = os.path.abspath(__file__)

    # Répertoire contenant le script en cours d'exécution
    current_directory = os.path.dirname(current_file_path)

    # command = f"wsl -- bash -c \"source {path_activate} {name} && {python_path_env} {path_server}"
    path_server = os.path.join(current_directory, 'server3.py')
    
    path_activate = f"{home_dir}/miniconda3/bin/activate"
    # command = f"/bin/bash -c 'source {path_activate} {name} && {python_path_env} {path_server}'"
   
    command = f"/bin/bash -c 'source {path_activate} {name} && {python_path_env} {path_server} \"{sys.argv[1]}\" \"{sys.argv[2]}\" \"{sys.argv[3]}\" \"{sys.argv[4]}\" \"{sys.argv[5]}\" \"{sys.argv[6]}\"'"
    
  
    
     
    
    result = subprocess.run(command,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
            print(f"Error processing the code in server3t. Return code: {result.returncode}")
            print("result.stdout : ","1"*150)
            print(result.stdout)
            print("result.stderr : ","1"*150)
            print(result.stderr)
    else:
        print(result.stdout)
        print("Process run succesfully")




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




if __name__ == "__main__":
    if len(sys.argv) > 3 :

        args = {
        "input": windows_to_linux_path(sys.argv[1]),
        "dir_models": windows_to_linux_path(sys.argv[2]),
        "lm_type": sys.argv[3].split(" "),
        "teeth": sys.argv[4].split(" "),
        "save_in_folder": sys.argv[5] == "true",
        "output_dir": windows_to_linux_path(sys.argv[6]),

        "image_size": 224,
        "blur_radius": 0,
        "faces_per_pixel": 1,
        # "sphere_radius": 0.3,
    }
        
        
        name = sys.argv[7]
        
    else:
        args = []
        name = "aliIOSCondaCli"
        
    call(name, args)