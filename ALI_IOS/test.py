import sys

print(f"Python executable used by Slicer: {sys.executable}")
import subprocess


print("OUI "*150)

python_path = "/home/luciacev/miniconda3/bin/python3"
command_to_execute = [python_path,"/home/luciacev/Desktop/SlicerAutomatedDentalTools/ALI_IOS/Func_Miniconda.py"]  
print(f"command_to_execute : {command_to_execute}")
result = subprocess.run(command_to_execute, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


if result.returncode != 0:
    print(f"Error creating the environment. Return code: {result.returncode}")
    print("result.stdout : ","*"*150)
    print(result.stdout)
    print("result.stderr : ","*"*150)
    print(result.stderr)
else:
    print("Environment created successfully.")



