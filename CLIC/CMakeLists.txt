#-----------------------------------------------------------------------------
set(MODULE_NAME CLIC)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  runner/clic_runner.py


)

set(MODULE_PYTHON_RESOURCES
  Ressources/Icons/CLIC.png
  Ressources/UI/CLIC.ui

)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)
#-----------------------------------------------------------------
