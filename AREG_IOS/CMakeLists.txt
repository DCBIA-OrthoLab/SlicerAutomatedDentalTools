#-----------------------------------------------------------------------------
set(MODULE_NAME AREG_IOS)



SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY AREG_IOS_utils)

set(MODULE_PYTHON_SCRIPTS
${FOLDER_LIBRARY}/__init__.py
${FOLDER_LIBRARY}/dataset.py
${FOLDER_LIBRARY}/ICP.py
${FOLDER_LIBRARY}/net.py
${FOLDER_LIBRARY}/orientation.py
${FOLDER_LIBRARY}/post_process.py
${FOLDER_LIBRARY}/PredPatch.py
${FOLDER_LIBRARY}/transformation.py
${FOLDER_LIBRARY}/utils.py
${FOLDER_LIBRARY}/vtkSegTeeth.py
)


slicerMacroBuildScriptedModule(
  NAME  ${FOLDER_LIBRARY}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)