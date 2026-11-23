#-----------------------------------------------------------------------------
set(MODULE_NAME VISU)

#-----------------------------------------------------------------------------
# VISULib is a plain Python package, not a Slicer module in its own right.
# Listing it as extra SCRIPTS of this module lands it next to VISU.py and makes
# `from VISULib import index` work — the same arrangement SlicerCloud uses for
# SlicerCloudLib, see ARCHITECTURE.md.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  VISULib/__init__.py
  VISULib/edits.py
  VISULib/index.py
  VISULib/review.py
  )

# No icon: it uses Slicer's default one.
set(MODULE_PYTHON_RESOURCES
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
  add_subdirectory(Testing)
endif()
