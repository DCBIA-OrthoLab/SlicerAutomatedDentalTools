#-----------------------------------------------------------------------------
set(MODULE_NAME ASO_IOS_utils)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
ASO_IOS_utils/__init__.py
ASO_IOS_utils/data_file.py 
ASO_IOS_utils/icp.py 
ASO_IOS_utils/OFFReader.py 
ASO_IOS_utils/pre_icp.py 
ASO_IOS_utils/transformation.py 
ASO_IOS_utils/utils.py
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  )