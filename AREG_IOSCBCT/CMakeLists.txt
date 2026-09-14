#-----------------------------------------------------------------------------
set(MODULE_NAME AREG_IOSCBCT)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
)

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
