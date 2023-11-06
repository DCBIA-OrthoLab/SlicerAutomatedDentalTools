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
    
    print("Le terminal est lancé en tant qu'administrateur.")

    # Check is WSL activate in parameters of windows
    wsl_status = subprocess.run("dism.exe /online /get-featureinfo /featurename:Microsoft-Windows-Subsystem-Linux",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)

    print("output : ",wsl_status.stdout)
    print("Est ce que c'est enable ou pas ? Reponse : ","Enabled" not in wsl_status.stdout)
    
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
    
    print("LetsContinue")
        
        
        
if __name__ == "__main__":
    main()