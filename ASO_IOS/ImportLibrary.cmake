#-----------------------------------------------------------------------------
set(MODULE_NAME ASO_IOS_utils)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}/__init__.py
  ${MODULE_NAME}/data_file.py
  ${MODULE_NAME}/icp.py
  ${MODULE_NAME}/OFFReader.py
  ${MODULE_NAME}/pre_icp.py
  ${MODULE_NAME}/transformation.py
  ${MODULE_NAME}/utils.py
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
