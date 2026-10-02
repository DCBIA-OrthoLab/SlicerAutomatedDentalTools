#-----------------------------------------------------------------------------
set(MODULE_NAME Agent_CLI)
set(FOLDER_LIBRARY Agent_CLI_utils)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/utils.py
  ${FOLDER_LIBRARY}/parameter_extraction_improved.py
  ${FOLDER_LIBRARY}/parameter_validator.py
  )

# Agent_CLI is a CLI: `Agent.py` calls it through `slicer.modules.agent_cli`.
# Slicer only recognises it as one if it finds its XML description next to the
# script. That description was missing here: in an installed build the script
# ended up alone, Slicer tried to load it as a scripted module, failed for want
# of a widget class, and `slicer.modules.agent_cli` did not exist. The defect
# does not show in a development tree, where the .xml sits beside the script.
set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  ${MODULE_NAME}.xml
  manifest.yaml
  )

SlicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)
