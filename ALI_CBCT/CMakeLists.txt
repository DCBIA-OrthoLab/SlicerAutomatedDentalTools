#-----------------------------------------------------------------------------
set(MODULE_NAME ALI_CBCT)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY ALI_CBCT_utils)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/agent.py
  ${FOLDER_LIBRARY}/brain.py
  ${FOLDER_LIBRARY}/constants.py
  ${FOLDER_LIBRARY}/environment.py
  ${FOLDER_LIBRARY}/io.py
  ${FOLDER_LIBRARY}/preprocess.py
)

slicerMacroBuildScriptedModule(
  NAME ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)