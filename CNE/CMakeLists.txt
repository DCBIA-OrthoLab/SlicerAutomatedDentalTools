#-----------------------------------------------------------------------------
set(MODULE_NAME CNE)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

# The sample notes are copied at run time from Resources/testfiles into the
# user's download folder (see `CNE.py`). They were missing from the manifest:
# the sample button found nothing in a compiled install.
set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/testfiles/input_TMJ/TMJ_note1.txt
  Resources/testfiles/input_TMJ/TMJ_note2.txt
  Resources/testfiles/input_Ortho/Ortho_note1.txt
  Resources/testfiles/input_Ortho/Ortho_note2.txt
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
