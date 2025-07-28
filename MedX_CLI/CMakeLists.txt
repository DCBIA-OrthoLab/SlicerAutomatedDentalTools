#-----------------------------------------------------------------------------
set(MODULE_NAME MedX_CLI)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY MedX_CLI_utils)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/utils.py
)

slicerMacroBuildScriptedModule(
  NAME ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)