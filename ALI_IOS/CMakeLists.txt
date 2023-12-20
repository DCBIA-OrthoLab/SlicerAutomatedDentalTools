#-----------------------------------------------------------------------------
set(MODULE_NAME ALI_IOS)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY utils_windows)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/first.py
  ${FOLDER_LIBRARY}/in_wsl.py
  ${FOLDER_LIBRARY}/server.py
)

slicerMacroBuildScriptedModule(
  NAME ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)