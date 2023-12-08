'''
This file is running on the python of slicer. It's checking the installation of miniconda and install it if not.
'''


import subprocess
import platform
import os
import sys
import urllib
import shutil



def checkMiniconda():
    '''
    check if miniconda3 is installed on the computer. It will check if the folder miniconda3 in the home folder.
    return bool and the path of miniconda3
    '''
    user_home = os.path.expanduser("~")
    default_install_path = os.path.join(user_home, "miniconda3")
    return(os.path.exists(default_install_path),default_install_path)

def InstallConda(default_install_path:str)->None:
        '''
        install conda on linux
        '''
        system = platform.system()
        machine = platform.machine()

        miniconda_base_url = "https://repo.anaconda.com/miniconda/"

        # Construct the filename based on the operating system and architecture
        if system == "Windows":
            if machine.endswith("64"):
                filename = "Miniconda3-latest-Windows-x86_64.exe"
            else:
                filename = "Miniconda3-latest-Windows-x86.exe"
        elif system == "Linux":
            if machine == "x86_64":
                filename = "Miniconda3-latest-Linux-x86_64.sh"
            else:
                filename = "Miniconda3-latest-Linux-x86.sh"
        else:
            raise NotImplementedError(f"Unsupported system: {system} {machine}")

        miniconda_url = miniconda_base_url + filename


        path_sh = os.path.join(default_install_path,"miniconda.sh")
        path_conda = os.path.join(default_install_path,"bin","conda")


        if not os.path.exists(default_install_path):
            os.makedirs(default_install_path)
            

        if system == "Windows":
            try:
                path_exe = os.path.join(os.path.expanduser("~"), "tempo")
       
                os.makedirs(path_exe, exist_ok=True)
                # Define paths for the installer and conda executable
                path_installer = os.path.join(path_exe, filename)
                path_conda = os.path.join(default_install_path, "Scripts", "conda.exe")
                # Download the Anaconda installer
                urllib.request.urlretrieve(miniconda_url, path_installer)
                print("Installer downloaded successfully.")
                print("Installing Miniconda...")
                
                # Run the Anaconda installer with silent mode
                print("path_installer : ",path_installer)
                print("default_install_path : ",default_install_path)
                path_miniconda = os.path.join(default_install_path,"miniconda")

                # Commande pour une installation silencieuse avec Miniconda
                install_command = f'"{path_installer}" /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D={default_install_path}'

                # Exécutez la commande d'installation
                subprocess.run(install_command, shell=True)

                subprocess.run(f"{path_conda} init cmd.exe", shell=True)
                print("Miniconda installed successfully.")
                
                try:
                    shutil.rmtree(path_exe)
                    print(f"Dossier {path_exe} et son contenu ont été supprimés avec succès.")
                except Exception as e:
                    print(f"Une erreur s'est produite lors de la suppression du dossier : {str(e)}")
                    return True
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return False

        else : 

            subprocess.run(f"mkdir -p {default_install_path}",capture_output=True, shell=True)
            subprocess.run(f"wget --continue --tries=3 {miniconda_url} -O {path_sh}",capture_output=True, shell=True)
            subprocess.run(f"chmod +x {path_sh}",capture_output=True, shell=True)

            try:
                subprocess.run(f"bash {path_sh} -b -u -p {default_install_path}",capture_output=True, shell=True)
                subprocess.run(f"rm -rf {path_sh}",shell=True)
                subprocess.run(f"{path_conda} init bash",shell=True)
                return True
            except:
                print("Le fichier est invalide.")
                return (False)
      


def run(args):
    '''
    main function. It will check if miniconda is installed and call second.py on the python of miniconda3
    '''
    system = platform.system()
    miniconda,default_install_path = checkMiniconda()
    
    if not miniconda :
        InstallConda(default_install_path)

    if system=="Windows" :
        python_path = os.path.join(default_install_path,"python") #python path in miniconda3
    else :
        python_path = os.path.join(default_install_path,"bin","python") #python path in miniconda3
        
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    path_func_miniconda = os.path.join(current_directory,'second.py') #Next files to call

    command_to_execute = [python_path,path_func_miniconda,args['file'],args['out'],args['overwrite'],args['mount_point'],args['name_env']]  

    env = dict(os.environ)
    if 'PYTHONPATH' in env:
        del env['PYTHONPATH']
    if 'PYTHONHOME' in env:
        del env['PYTHONHOME']

    result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,env=env)


    if result.returncode != 0:
        print(f"Error creating the environment. Return code: {result.returncode}")
        print("result.stdout : ","*"*150)
        print(result.stdout)
        print("result.stderr : ","*"*150)
        print(result.stderr)
    else:
        print(result.stdout)
        print("Environment created successfully.")



if __name__ == "__main__":
    args = {
    "file": sys.argv[1],
    "out": sys.argv[2],
    "overwrite": sys.argv[3],
    "mount_point": sys.argv[4],
    "name_env":sys.argv[5]
    
    }
    
    run(args)