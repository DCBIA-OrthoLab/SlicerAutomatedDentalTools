#-----------------------------------------------------------------------------
set(MODULE_NAME MRI2CBCT_CLI_utils)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}/__init__.py
  ${MODULE_NAME}/resample_create_csv.py
  ${MODULE_NAME}/resample.py
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
