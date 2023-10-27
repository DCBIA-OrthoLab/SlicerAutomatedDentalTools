import subprocess
import time
import os
import rpyc

def call(name):
    default_install_path = "~/miniconda3"
    path_activate = "~/miniconda3/bin/activate"
    python_path_env = f"~/miniconda3/envs/{name}/bin/python"
    print("absbdfhirhfurhn")
    
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
    path_server = os.path.join(current_directory, 'server2.py')
    print("path_server : ",path_server)
    command = f"{python_path_env} {path_server}"

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


# name = "aliIOSCondaCLI"
# print("*"*200)
# call(name)