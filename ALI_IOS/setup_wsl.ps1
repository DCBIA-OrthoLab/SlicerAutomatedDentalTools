# ##############Downloading and installing the app###################
# # Enable wsl subsystems for linux (if powershell is ran in admin mode)
# Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

# # Set Tls12 protocol to be able to download the wsl application
# [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# # check to see if ubuntu1804 installation file exists and download the app otherwise
# $fileToCheck = "Ubuntu1804.appx"
# if (Test-Path $fileToCheck -PathType leaf) 
# {"File does Exist"}
# else
# {Invoke-WebRequest -Uri https://aka.ms/wsl-ubuntu-1804 -OutFile Ubuntu1804.appx -UseBasicParsing}

# # Actually install the wsl ubuntu 18.04 app
# Add-AppxPackage .\Ubuntu1804.appx
# Write-Output "Installed the ubuntu18.04"

# # backup installation command if the first command did not function properly
# invoke-expression -Command "Add-AppxPackage .\Ubuntu1804.appx"
# Write-Output "Installed the ubuntu with backup attempt"


# ##############Initializing the wsl ubuntu 18.04 app without requiring user input###################

# # First define path to the installed ubuntu1804.exe
# $str1="/Users/"
# $str2="/AppData/Local/Microsoft/WindowsApps/ubuntu1804"
# $hdd_name=(Get-WmiObject Win32_OperatingSystem).SystemDrive
# $username=$env:UserName
# [String] $ubuntu1804_path=$hdd_name+$str1+$username+$str2

# # let root be default username
# $str1=" install --root"
# $set_user=$ubuntu1804_path+$str1
# invoke-expression -Command $set_user 

# Write-Host "Done with setup."