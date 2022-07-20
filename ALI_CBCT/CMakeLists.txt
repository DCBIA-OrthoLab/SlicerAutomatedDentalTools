#-----------------------------------------------------------------------------
set(MODULE_NAME ALI_CBCT)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
  )

set(CLI_DEST "${CMAKE_BINARY_DIR}/${Slicer_CLIMODULES_BIN_DIR}")
set(CLI_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")