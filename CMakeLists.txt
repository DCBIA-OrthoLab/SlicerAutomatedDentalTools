cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Slicer Automatic Ortho Tools)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerAutomaticTools")
set(EXTENSION_CATEGORY "Segmentation & Quantification")
set(EXTENSION_CONTRIBUTORS "Maxime Gillot (University of Michigan), Baptiste Baquero (University of Michigan), Lucia Cevidanes (University of Michigan), Juan Carlos Prieto (University of North Carolina)")
set(EXTENSION_DESCRIPTION "Slicer extension for segmentation and quantification automatic tools such as 
Automatic Multi Skull Structure Segmentation (AMASSS) and Automatic Landmark Identification ALI")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SlicerAutomaticTools.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SlicerAutomaticTools/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AMASSS)
add_subdirectory(AMASSS_CLI)
add_subdirectory(ALI)
add_subdirectory(ALI_CLI)

## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
