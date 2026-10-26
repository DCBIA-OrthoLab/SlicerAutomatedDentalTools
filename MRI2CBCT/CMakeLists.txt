#-----------------------------------------------------------------------------
set(MODULE_NAME MRI2CBCT)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  MRI2CBCT_core/__init__.py
  MRI2CBCT_core/env.py
  MRI2CBCT_core/frames.py
  MRI2CBCT_core/ladder.py
  MRI2CBCT_core/layout.py
  MRI2CBCT_core/mri_prep.py
  MRI2CBCT_core/pairing.py
  MRI2CBCT_core/plan.py
  MRI2CBCT_core/requirements.py
  MRI2CBCT_core/seg516.py
  MRI2CBCT_core/reg/__init__.py
  MRI2CBCT_core/reg/global_reg.py
  MRI2CBCT_core/reg/itk_io.py
  MRI2CBCT_core/reg/pipeline.py
  MRI2CBCT_core/reg/surface.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
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
