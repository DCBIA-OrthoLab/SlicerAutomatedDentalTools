#-----------------------------------------------------------------------------
set(MODULE_NAME ServerToolsCore)

#-----------------------------------------------------------------------------
# ServerToolsCoreLib is a plain Python package, not a Slicer module in its own
# right. It is listed here as extra SCRIPTS of the hidden ServerToolsCore
# module so it lands next to it and is importable via `from ServerToolsCoreLib
# import ...` from any scripted module in the extension (see ARCHITECTURE.md,
# "Build system" section, for why this mirrors FlexRegLib rather than
# using a raw install(DIRECTORY ...) into Slicer_QTSCRIPTEDMODULES_LIB_DIR).
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ServerToolsCoreLib/__init__.py
  ServerToolsCoreLib/config.py
  ServerToolsCoreLib/console.py
  ServerToolsCoreLib/errors.py
  ServerToolsCoreLib/client.py
  ServerToolsCoreLib/local_client.py
  ServerToolsCoreLib/tools_store.py
  ServerToolsCoreLib/transfer.py
  ServerToolsCoreLib/digest.py
  ServerToolsCoreLib/joystick.py
  ServerToolsCoreLib/slicer_io.py
  ServerToolsCoreLib/design.py
  ServerToolsCoreLib/formgen.py
  ServerToolsCoreLib/worker.py
  ServerToolsCoreLib/base_widget.py
  ServerToolsCoreLib/settings_qt.py
  ServerToolsCoreLib/tls.py
  ServerToolsCoreLib/local_schema.py
  ServerToolsCoreLib/tools_ui.py
  ServerToolsCoreLib/mode_switch.py
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
