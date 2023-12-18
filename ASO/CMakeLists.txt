#-----------------------------------------------------------------------------
set(MODULE_NAME ASO)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  Methode/Methode.py
  Methode/IOS.py
  Methode/CBCT.py
  Methode/Progress.py
  Methode/IOS_utils/Reader.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/Image/1_resize.png
  Resources/Image/2_resize.png
  Resources/Image/3_resize.png
  Resources/Image/4_resize.png
  Resources/Image/5_resize.png
  Resources/Image/6_resize.png
  Resources/Image/7_resize.png
  Resources/Image/8_resize.png
  Resources/Image/9_resize.png
  Resources/Image/10_resize.png
  Resources/Image/11_resize.png
  Resources/Image/12_resize.png
  Resources/Image/13_resize.png
  Resources/Image/14_resize.png
  Resources/Image/15_resize.png
  Resources/Image/16_resize.png
  Resources/Image/17_resize.png
  Resources/Image/18_resize.png
  Resources/Image/19_resize.png
  Resources/Image/20_resize.png
  Resources/Image/21_resize.png
  Resources/Image/22_resize.png
  Resources/Image/23_resize.png
  Resources/Image/24_resize.png
  Resources/Image/25_resize.png
  Resources/Image/26_resize.png
  Resources/Image/27_resize.png
  Resources/Image/28_resize.png
  Resources/Image/29_resize.png
  Resources/Image/30_resize.png
  Resources/Image/31_resize.png
  Resources/Image/32_resize.png
  Resources/Image/1_resize_child.png
  Resources/Image/2_resize_child.png
  Resources/Image/3_resize_child.png
  Resources/Image/4_resize_child.png
  Resources/Image/5_resize_child.png
  Resources/Image/6_resize_child.png
  Resources/Image/7_resize_child.png
  Resources/Image/8_resize_child.png
  Resources/Image/9_resize_child.png
  Resources/Image/10_resize_child.png
  Resources/Image/11_resize_child.png
  Resources/Image/12_resize_child.png
  Resources/Image/13_resize_child.png
  Resources/Image/14_resize_child.png
  Resources/Image/15_resize_child.png
  Resources/Image/16_resize_child.png
  Resources/Image/17_resize_child.png
  Resources/Image/18_resize_child.png
  Resources/Image/19_resize_child.png
  Resources/Image/20_resize_child.png

  

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
