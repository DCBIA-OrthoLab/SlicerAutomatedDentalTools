#-----------------------------------------------------------------------------
set(MODULE_NAME BATCHDENTALSEG)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  BATCHDENTALSEGLib/__init__.py
  BATCHDENTALSEGLib/IconPath.py
  BATCHDENTALSEGLib/PythonDependencyChecker.py
  BATCHDENTALSEGLib/SegmentationWidget.py
  BATCHDENTALSEGLib/Signal.py
  BATCHDENTALSEGLib/Utils.py
  Testing/__init__.py
  Testing/IntegrationTestCase.py
  Testing/SegmentationWidgetTestCase.py
  Testing/Utils.py
)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/DentalSegmentator.png
  Resources/Icons/DentalSegmentator_full_icon.png
  Resources/Icons/DentalSegmentator_full_icon_2.png
  Resources/Icons/info.png
  Resources/Icons/loading.gif
  Resources/Icons/start_icon.png

)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)
#-----------------------------------------------------------------
