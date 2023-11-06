import ctypes
import sys
import os
import time
import subprocess
import urllib.request
import tempfile

# Check if ubuntu is installed on the computer
def is_ubuntu_installed():
    result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
    output = result.stdout.encode('utf-16-le').decode('utf-8')
    clean_output = output.replace('\x00', '')  # Enlève tous les octets null
    return 'Ubuntu' in clean_output

# check if the code is running in administrator
def is_admin():
    # Créer un fichier temporaire pour stocker la sortie du script secondaire
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    # Préparer la commande avec les privilèges d'administrateur
    exe_dir = os.path.dirname(sys.executable)
                
    bash_script = "secondaire.py"

    script_path = os.path.join(exe_dir, bash_script)
    
    params = f'cmd /c "python {script_path} > {temp_file_path} 2>&1"'
    shell_command = "runas"
    
    ctypes.windll.shell32.ShellExecuteW(None, shell_command, "cmd.exe", params, None, 1)
    
    # Attendre que l'utilisateur ferme manuellement le terminal admin ou que le script se termine
    
    # Lire la sortie du fichier temporaire
    with open(temp_file_path, 'r') as file:
        output = file.read()
        while "LetsContinue" not in output :
            print("Secondaire is not finish")
            time.sleep(5)
            output = file.read()
            
    
    # Supprimer le fichier temporaire
    os.unlink(temp_file_path)
    
    print("output : ",output)
    
    input("Finish ? ")
    
if __name__ == "__main__":
    is_admin()