cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerAutomatedDentalTools)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/blob/main/README.md")
set(EXTENSION_CATEGORY "SlicerCMF")
set(EXTENSION_CONTRIBUTORS "Maxime Gillot (University of Michigan), Baptiste Baquero (UoM), Juan Carlos Prieto (University of North Carolina), Nathan Hutin (University of Michigan), Luc Anchling (University of Michigan)")
set(EXTENSION_DESCRIPTION "This extension will allow clinicians to perform automatic CBCT scan segmentation as well as automatic lamndmark identification in CBCT and IOS using machine learning tools.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/main/SlicerAutomaticTools.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/main/ADT-exemple.png")
set(EXTENSION_DEPENDS SlicerDentalModelSeg) # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AMASSS)
add_subdirectory(AMASSS_CLI)
add_subdirectory(ALI)
add_subdirectory(ALI_CBCT)
add_subdirectory(ALI_IOS)
add_subdirectory(ASO)
add_subdirectory(ASO_IOS)
add_subdirectory(ASO_CBCT)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
