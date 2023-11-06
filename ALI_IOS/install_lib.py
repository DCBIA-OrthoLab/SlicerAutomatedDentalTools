import subprocess
import time
import os



# Check if ubuntu is installed on the computer
def is_ubuntu_installed():
    result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
    output = result.stdout.encode('utf-16-le').decode('utf-8')
    clean_output = output.replace('\x00', '')  # Enlève tous les octets null
    return 'Ubuntu' in clean_output

def main():
    if not is_ubuntu_installed():
        subprocess.run("wsl --set-default-version 2", shell=True)
        # Utilise sys.executable pour obtenir le chemin du .exe et trouve le dossier contenant le .exe
        exe_dir = os.path.dirname(sys.executable)
        
        # Nom du script bash que vous voulez exécuter
        bash_script = "install_libs.sh"

        # Construit le chemin complet pour le script bash
        full_bash_script_path = os.path.join(exe_dir, bash_script)

        # Remplace les séparateurs de chemin pour Windows par ceux de Linux
        # et supprime le lecteur (ex: 'C:') pour le formatage WSL
        wsl_path = "/mnt/" + full_bash_script_path[0].lower() + full_bash_script_path[2:].replace("\\", "/")


        # Affiche un message et attend que l'utilisateur appuie sur Entrée
        input("During the installation you will create an username with a password, please note that whilst entering the Password, nothing will appear on screen. This is called blind typing. You won't see what you are typing, this is completely normal.\nWhen the line will start by your username please enter 'exit' to continue the installation\nPress enter to start the installation")
        subprocess.check_call(["wsl", "--install"])
        
        # Wait for Ubuntu to be installed on WSL
        while not is_ubuntu_installed():
            time.sleep(10)
        
    # Once WSL is initialized, execute the bash script
    subprocess.run(["wsl", "bash", wsl_path])
    
    input("Installation succesfull, press enter to finish it")

if __name__ == "__main__":
    main()


