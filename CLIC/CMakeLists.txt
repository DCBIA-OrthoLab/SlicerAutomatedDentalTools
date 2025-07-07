#-----------------------------------------------------------------------------
set(MODULE_NAME CLIC)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  runner/clic_runner.py


)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/CLIC.png
  Resources/UI/CLIC.ui

)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)
#-----------------------------------------------------------------
