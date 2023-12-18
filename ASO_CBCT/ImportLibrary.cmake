#-----------------------------------------------------------------------------
set(MODULE_NAME ASO_CBCT_utils)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
    ${MODULE_NAME}/__init__.py
    ${MODULE_NAME}/Net.py 
    ${MODULE_NAME}/ResamplePreASO.py 
    ${MODULE_NAME}/utils.py 

  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  )