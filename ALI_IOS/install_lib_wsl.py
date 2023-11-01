# import subprocess
# import time

# delay = 30
# mdp = input('Enter your mdp here :') 
# print("mdp : ",mdp)

# print("c'est partis")
# process = subprocess.Popen("wsl -- bash -c \"sudo apt update && sudo apt install libxrender1\"", 
#                            shell=True, 
#                            stdout=subprocess.PIPE, 
#                            stderr=subprocess.PIPE, 
#                            stdin=subprocess.PIPE,
#                            text=True, 
#                            encoding='utf-8', 
#                            errors='replace')

# time.sleep(10)

# process.stdin.write(mdp + '\n')
# print("line : abc", mdp, "abc")
# process.stdin.flush()  # Assurez-vous que les données sont bien envoyées
# # time.sleep(delay)  # Attendez pendant le délai spécifié
# process.wait()
