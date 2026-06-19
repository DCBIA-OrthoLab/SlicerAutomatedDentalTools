#-----------------------------------------------------------------------------
set(MODULE_NAME AREG_IOSCBCT)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY AREG_IOSCBCT_utils)

set(MODULE_PYTHON_SCRIPTS
)


slicerMacroBuildScriptedModule(
  NAME ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)