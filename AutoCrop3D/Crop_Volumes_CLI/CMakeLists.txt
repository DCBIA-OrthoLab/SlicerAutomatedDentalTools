#-----------------------------------------------------------------------------
set(MODULE_NAME AutoCrop3D_CLI)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY Crop_Volumes_utils)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/CropCBCT.py
  ${FOLDER_LIBRARY}/FilesType.py
  ${FOLDER_LIBRARY}/GenerateVTKfromSeg.py
)

slicerMacroBuildScriptedModule(
  NAME ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
