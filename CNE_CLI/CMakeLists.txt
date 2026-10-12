#-----------------------------------------------------------------------------
set(MODULE_NAME CNE_CLI)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
  )

# Installs CNE_CLI_utils, the package CNE_Dashboard imports.
include(ImportLibrary.cmake)
