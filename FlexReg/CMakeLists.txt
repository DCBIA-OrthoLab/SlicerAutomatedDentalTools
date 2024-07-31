#-----------------------------------------------------------------------------
set(MODULE_NAME FlexReg)

#-----------------------------------------------------------------------------

set(FOLDER_LIBRARY FlexReg_utils)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  FlexReg_utils/__init__.py
  FlexReg_utils/orientation.py
  FlexReg_utils/transform.py
  FlexReg_utils/util.py
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
