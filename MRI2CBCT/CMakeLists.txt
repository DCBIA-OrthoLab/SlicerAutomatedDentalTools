#-----------------------------------------------------------------------------
set(MODULE_NAME MRI2CBCT)
set(FOLDER_LIBRARY MRI2CBCT_utils)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${FOLDER_LIBRARY}/Approx_MRI2CBCT.py
  ${FOLDER_LIBRARY}/Method.py
  ${FOLDER_LIBRARY}/Preprocess_CBCT_MRI.py
  ${FOLDER_LIBRARY}/Preprocess_CBCT.py
  ${FOLDER_LIBRARY}/Preprocess_MRI.py
  ${FOLDER_LIBRARY}/Reg_MRI2CBCT.py
  ${FOLDER_LIBRARY}/LR_crop.py
  ${FOLDER_LIBRARY}/utils_CBCT.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
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
