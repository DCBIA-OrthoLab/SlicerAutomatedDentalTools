#-----------------------------------------------------------------------------
# The VISOR local twin of the classic DOCShapeAXI module. Same tool name on the SADT-VISOR
# side; the module name carries the suffix so both can live in one extension.
set(MODULE_NAME DOCShapeAXIVisor)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

# No icon of its own: the module takes the classic one (mode_switch.legacy_icon).
set(MODULE_PYTHON_RESOURCES
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
