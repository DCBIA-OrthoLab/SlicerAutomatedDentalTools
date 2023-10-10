#-----------------------------------------------------------------------------
set(MODULE_NAME Flex_Reg_CLI)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

set(FOLDER_LIBRARY Method)

set(MODULE_PYTHON_SCRIPTS
  ${FOLDER_LIBRARY}/__init__.py 
  ${FOLDER_LIBRARY}/draw.py 
  ${FOLDER_LIBRARY}/ICP.py 
  ${FOLDER_LIBRARY}/make_butterfly.py 
  ${FOLDER_LIBRARY}/orientation.py 
  ${FOLDER_LIBRARY}/propagation.py 
  ${FOLDER_LIBRARY}/transformation.py 
  ${FOLDER_LIBRARY}/util.py 
  ${FOLDER_LIBRARY}/utils.py 
  ${FOLDER_LIBRARY}/vtkSegTeeth.py 
  )

  slicerMacroBuildScriptedModule(
    NAME  ${FOLDER_LIBRARY}
    SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)