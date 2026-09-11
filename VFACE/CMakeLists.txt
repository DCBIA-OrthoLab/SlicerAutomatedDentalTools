#-----------------------------------------------------------------------------
set(MODULE_NAME VFACE)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  VFACE_utils/__init__.py
  VFACE_utils/_batch_worker.py
  VFACE_utils/createlistprocess.py
  VFACE_utils/functionaq3dc.py
  VFACE_utils/Group_landmark.py
  VFACE_utils/Line.py
  VFACE_utils/Measure.py
  VFACE_utils/Point.py
  VFACE_utils/Progress.py
  VFACE_utils/review_steps.py
  VFACE_utils/segmentation_logic.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  # The nnUNet model description travels with the module: without these the
  # installed extension has an empty Resources/ML, the weight downloader finds
  # no URL and the dental segmentation fails on every scan. Only the checkpoint
  # itself is left out, it is fetched at first use.
  Resources/ML/download_info.json
  Resources/ML/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json
  Resources/ML/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json
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
