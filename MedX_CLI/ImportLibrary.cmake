#-----------------------------------------------------------------------------
set(MODULE_NAME MedX_CLI_utils)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}/__init__.py
  ${MODULE_NAME}/dashboard_utils.py
  ${MODULE_NAME}/display_figure.py
  ${MODULE_NAME}/utils.py
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
