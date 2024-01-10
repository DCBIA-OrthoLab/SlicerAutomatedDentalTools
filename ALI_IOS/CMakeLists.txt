#-----------------------------------------------------------------------------
set(MODULE_NAME ALI_IOS)

SlicerMacroBuildScriptedCLI(
  NAME ${MODULE_NAME}
  )



  slicerMacroBuildScriptedModule(
    NAME  ${FOLDER_LIBRARY}
    SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)