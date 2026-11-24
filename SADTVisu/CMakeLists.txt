#-----------------------------------------------------------------------------
set(MODULE_NAME SADTVisu)

#-----------------------------------------------------------------------------
# SADTVisuLib is a plain Python package, not a Slicer module in its own right.
# Listing it as extra SCRIPTS of this module lands it next to SADTVisu.py and makes
# `from SADTVisuLib import index` work — the same arrangement SlicerCloud uses for
# SlicerCloudLib, see ARCHITECTURE.md.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  SADTVisuLib/__init__.py
  SADTVisuLib/edits.py
  SADTVisuLib/index.py
  SADTVisuLib/review.py
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
