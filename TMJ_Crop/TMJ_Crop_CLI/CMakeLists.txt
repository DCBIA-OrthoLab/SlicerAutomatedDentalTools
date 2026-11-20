#-----------------------------------------------------------------------------
set(MODULE_NAME TMJ_Crop_CLI)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY TMJ_Crop_utils)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/batch.py
  ${FOLDER_LIBRARY}/catalogs.py
  ${FOLDER_LIBRARY}/crop.py
)

slicerMacroBuildScriptedModule(
  NAME ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
