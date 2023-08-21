#-----------------------------------------------------------------------------
set(MODULE_NAME Matrix_CLI)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY Apply_matrix_utils)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py 
  ${FOLDER_LIBRARY}/General_tools.py 
  ${FOLDER_LIBRARY}/Matrix_tools.py 
  ${FOLDER_LIBRARY}/OFFReader.py 
  ${FOLDER_LIBRARY}/VTK_tools.py 
  )

  slicerMacroBuildScriptedModule(
    NAME  ${FOLDER_LIBRARY}
    SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)