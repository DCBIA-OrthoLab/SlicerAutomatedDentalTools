#-----------------------------------------------------------------------------
set(MODULE_NAME SADTCore)

#-----------------------------------------------------------------------------
# SADTCoreLib is a plain Python package, not a Slicer module in its own
# right. It is listed here as extra SCRIPTS of the hidden SADTCore
# module so it lands next to it and is importable via `from SADTCoreLib
# import ...` from any scripted module in the extension (see ARCHITECTURE.md,
# "Build system" section, for why this mirrors FlexRegUILib rather than
# using a raw install(DIRECTORY ...) into Slicer_QTSCRIPTEDMODULES_LIB_DIR).
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  SADTCoreLib/__init__.py
  SADTCoreLib/config.py
  SADTCoreLib/console.py
  SADTCoreLib/errors.py
  SADTCoreLib/client.py
  SADTCoreLib/local_client.py
  SADTCoreLib/tools_store.py
  SADTCoreLib/transfer.py
  SADTCoreLib/digest.py
  SADTCoreLib/joystick.py
  SADTCoreLib/slicer_io.py
  SADTCoreLib/design.py
  SADTCoreLib/formgen.py
  SADTCoreLib/worker.py
  SADTCoreLib/base_widget.py
  SADTCoreLib/settings_qt.py
  SADTCoreLib/tls.py
  SADTCoreLib/local_schema.py
  SADTCoreLib/tools_ui.py
  SADTCoreLib/mode_switch.py
  )

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
