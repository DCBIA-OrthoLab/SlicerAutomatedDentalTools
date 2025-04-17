# #-----------------------------------------------------------------------------
set(MODULE_NAME ALI_IOS)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY ALI_IOS_utils)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/agent.py
  ${FOLDER_LIBRARY}/io.py
  ${FOLDER_LIBRARY}/mask_renderer.py
  ${FOLDER_LIBRARY}/model.py
  ${FOLDER_LIBRARY}/render.py
  ${FOLDER_LIBRARY}/surface.py
)

slicerMacroBuildScriptedModule(
  NAME ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
# #-----------------------------------------------------------------------------