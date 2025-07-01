#-----------------------------------------------------------------------------
set(MODULE_NAME CMAKE_LINK_INTERFACE_LIBRARIES)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  runner/clic_runenr.py


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
