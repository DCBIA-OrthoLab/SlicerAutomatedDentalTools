#-----------------------------------------------------------------------------
set(MODULE_NAME AutoCrop3D)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui

  # The test data set behind the "Test Files" button and the self-test. Left
  # undeclared, it existed in a development tree -- where everything is there
  # -- and vanished from the compiled extension, where the button would have
  # found nothing to unzip.
  Resources/testfiles/${MODULE_NAME}/Segmentation.zip
  Resources/testfiles/${MODULE_NAME}/ROI.mrk.zip
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
)

#-----------------------------------------------------------------------------
if (BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif ()
