cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(AutoMatrix)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/Apply_matrix_bis")
set(EXTENSION_CATEGORY "Tuto")
set(EXTENSION_CONTRIBUTORS "Gaelle Leroux (DCBIA)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/Apply_matrix_bis.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/Apply_matrix_bis/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Matrix_CLI)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})