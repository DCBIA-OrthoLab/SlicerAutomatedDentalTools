import ctypes
import sys
import os
import time
import subprocess
import urllib.request

# Check if ubuntu is installed on the computer
def is_ubuntu_installed():
    result = subprocess.run(['wsl', '--list'], capture_output=True, text=True)
    output = result.stdout.encode('utf-16-le').decode('utf-8')
    clean_output = output.replace('\x00', '')  # Enlève tous les octets null
    return 'Ubuntu' in clean_output

# check if the code is running in administrator
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def download_wsl2_kernel():
    kernel_url = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
    kernel_path = os.path.join(os.path.expanduser("~"), "Downloads", "wsl_update_x64.msi")
    urllib.request.urlretrieve(kernel_url, kernel_path)
    return kernel_path

# Install the kernel of linux
def install_wsl2_kernel(kernel_path):
    subprocess.run(["msiexec", "/i", kernel_path, "/quiet"], check=True)
    
def main():
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1) # restart the code in admin mode
    else:
        print("Le terminal est lancé en tant qu'administrateur.")

        # Check is WSL activate in parameters of windows
        wsl_status = subprocess.run("dism.exe /online /get-featureinfo /featurename:Microsoft-Windows-Subsystem-Linux",
                                    capture_output=True, text=True, shell=True)

        if "Enabled" not in wsl_status.stdout:
            print("WSL n'est pas activé. Activation en cours...")
            # Activate WSL and virtual machine platform for wsl2
            subprocess.run("dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart", shell=True)
            subprocess.run("dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart", shell=True)
            
            # Install kernel linux (for wsl2)
            kernel_path = download_wsl2_kernel()
            print("Installation du noyau Linux...")
            install_wsl2_kernel(kernel_path)
            
            # set default version of wsl
            subprocess.run("wsl --set-default-version 2", shell=True)
            
            # need to restart the computer
            print("WSL a été activé avec la version 2 comme défaut. Redémarrez votre ordinateur et relancer ce script pour terminer l'installation.")
        else:
            print("WSL est déjà activé.")
            if not is_ubuntu_installed(): # check is ubuntu is install on wsl
                #define version of wsl at 2
                subprocess.run("wsl --set-default-version 2", shell=True)
                
                exe_dir = os.path.dirname(sys.executable)
                
                bash_script = "install_libs.sh"

                full_bash_script_path = os.path.join(exe_dir, bash_script)

                # Path to the .sh with the command to install librairie on wsl
                wsl_path = "/mnt/" + full_bash_script_path[0].lower() + full_bash_script_path[2:].replace("\\", "/")


                # Write a message for the user before installing
                input("During the installation you will create an username with a password, please note that whilst entering the Password, nothing will appear on screen. This is called blind typing. You won't see what you are typing, this is completely normal.\nWhen the line will start by your username please enter 'exit' to continue the installation\nPress enter to start the installation")
                subprocess.check_call(["wsl", "--install"])
                
                # Wait for Ubuntu to be installed on WSL
                while not is_ubuntu_installed():
                    time.sleep(10)
                
            # Once WSL is initialized, execute the bash script (to install the librairies)
            subprocess.run(["wsl", "bash", wsl_path])
            
            input("Installation succesfull, press enter to finish it")

if __name__ == "__main__":
    main()