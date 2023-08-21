cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Crop_Volume)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/Crop_Volume")
set(EXTENSION_CATEGORY "Test_crop_volume")
set(EXTENSION_CONTRIBUTORS "Jeanne Claret (DCBIA)")
set(EXTENSION_DESCRIPTION "This is a simple extension to crop volumes.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/Crop_Volume.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/Crop_Volume/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(t_crop_volumes)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
