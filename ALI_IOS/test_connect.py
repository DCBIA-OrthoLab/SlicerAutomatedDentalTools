import subprocess

def is_ubuntu_installed():
    result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
    output = result.stdout.encode('utf-16-le').decode('utf-8')
    clean_output = output.replace('\x00', '')  # Enlève tous les octets null

    # print("clean_output :", clean_output)

    return 'Ubuntu' in clean_output


print(is_ubuntu_installed())

def check_lib():
    result1 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libxrender1\"", capture_output=True, text=True)
    output1 = result1.stdout.encode('utf-16-le').decode('utf-8')
    clean_output1 = output1.replace('\x00', '') 
    
    result2 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libgl1-mesa-glx\"", capture_output=True, text=True)
    output2 = result2.stdout.encode('utf-16-le').decode('utf-8')
    clean_output2 = output2.replace('\x00', '')

    return "libxrender1" in clean_output1 and "libgl1-mesa-glx" in clean_output2

print(check_lib())

# import subprocess
# import time
# import codecs

# def run_command_with_input_and_delays(command, input_data, delay):
#     process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     time.sleep(20)

#     for line in input_data:
#         process.stdin.write(line + '\n')
#         print("line : abc",line,"abc")
#         process.stdin.flush()  # Assurez-vous que les données sont bien envoyées
#         time.sleep(delay)  # Attendez pendant le délai spécifié

#     return process

# def is_ubuntu_installed():
#     result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
#     output = result.stdout.encode('utf-16-le').decode('utf-8')
#     clean_output = output.replace('\x00', '')  # Enlève tous les octets null

#     # print("clean_output :", clean_output)

#     return 'Ubuntu' in clean_output




# if __name__ == "__main__":
#     cmd = ['wsl', '--install']
#     input_data = ['user1', 'oui', 'oui']
#     delay = 20  # Délai de 2 secondes entre les entrées
    
#     process=None
#     if is_ubuntu_installed():
#         print("Ubuntu deja installe")
        

#     else : 
#         process = run_command_with_input_and_delays(cmd, input_data, delay)
#         max_attempts = 30  # Par exemple, vérifier pendant 5 minutes
#         attempts = 0

#         while attempts < max_attempts:
#             if is_ubuntu_installed():
#                 time.sleep(20)
#                 print("Ubuntu a été correctement installé!")
#                 # process.kill()  # Ubuntu est installé, nous pouvons tuer le processus
#                 # subprocess.run("wsl -- bash -c \"sudo apt update && sudo apt install libxrender1\"", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
#                 # print("premiere installation")
#                 # subprocess.run("wsl -- bash -c \"sudo apt install -y libgl1-mesa-glx\"", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
#                 # print("et la seconde")
#                 break
#             else:
#                 time.sleep(10)  # Attendre 10 secondes avant de vérifier à nouveau
#                 attempts += 1
#                 print("attemps : ",attempts)

#             if attempts == max_attempts:
#                 print("Erreur: Ubuntu n'a pas été détecté après plusieurs tentatives.")


