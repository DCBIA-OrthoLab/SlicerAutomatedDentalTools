import sys
import os
import time
import subprocess
import rpyc

def call(name,args):
    home_dir = os.path.expanduser("~")
    print("home_dir  : ",home_dir)
    default_install_path = "~/miniconda3"
    path_activate = "~/miniconda3/bin/activate"
    python_path_env = f"{home_dir}/miniconda3/envs/{name}/bin/python"
    print("name  : ",{name})
    
    # activate_env = os.path.join(default_install_path, "bin", "activate")
    # python_executable = os.path.join(default_install_path, "envs",name,"python")  # Modifiez selon votre système d'exploitation et votre installation


    current_file_path = os.path.abspath(__file__)
    print("current_file_path : ",current_file_path)

    # Répertoire contenant le script en cours d'exécution
    current_directory = os.path.dirname(current_file_path)
    print("current_directory : ",current_directory)

    # path_activate = os.path.join(default_install_path, "Scripts", "activate")
    # activate_command = f"conda {path_activate} {name}"
    # subprocess.run(activate_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    # Chemin absolu du fichier souhaité qui est à côté du script en cours d'exécution
    
    # command = f"wsl -- bash -c \"source {path_activate} {name} && {python_path_env} {path_server}"
    path_server = os.path.join(current_directory, 'server.py')
    print("path_server : ",path_server)
    
    path_activate = f"{home_dir}/miniconda3/bin/activate"
    command = f"source {path_activate} {name} && {python_path_env} {path_server}"
    command = f"/bin/bash -c 'source {path_activate} {name} && {python_path_env} {path_server}'"

    print("command : ",command)

    # Start server
    server_process = subprocess.Popen(command, shell=True)
    
    # To be sure the server start
    time.sleep(5)
    print("on essaye de se connecter")
    
    conn = rpyc.connect("localhost", 18817, config={"sync_request_timeout": 600})
    # wait_for_server_ready(conn)
    time.sleep(2)


    # print("on lui demande le resultat")
    # x=7
    # result=conn.root.running(x)
    # print(f"{x} au carre = {result}")
    print("args : ",args)
    conn.root.running(args)
    print(time.sleep(5))
    
    
    # Stop process
    
    # server_process.terminate()
    # server_process.wait()
    
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