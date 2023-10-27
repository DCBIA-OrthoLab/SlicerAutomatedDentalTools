
import subprocess
import rpyc
import time
import os 


# python_path_env = f"~/miniconda3/envs/{name}/bin/python"

# current_file_path = os.path.abspath(__file__)
# print("current_file_path : ",current_file_path)

# # Répertoire contenant le script en cours d'exécution
# current_directory = os.path.dirname(current_file_path)
# print("current_directory : ",current_directory)


# path_server = os.path.join(current_directory, 'server2.py')
# print("path_server : ",path_server)
# command = f"{python_path_env} {path_server}"

# # Start server
# server_process = subprocess.Popen(command, shell=True)

# # To be sure the server start
# time.sleep(2)
print("on essaye de se connecter")

conn = rpyc.connect("localhost", 18817)
# wait_for_server_ready(conn)
time.sleep(2)


print("on lui demande le resultat")
x=9
result=conn.root.running(x)
print(f"{x} au carre = {result}")


result = conn.root.stop()
if result == "DISCONNECTING":
    conn.close()

print("on a ferme le server")
