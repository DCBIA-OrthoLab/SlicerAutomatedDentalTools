#-----------------------------------------------------------------------------
set(MODULE_NAME BATCHDENTALSEG)

if (BUILD_TESTING)
  set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Testing/Data")
  file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

  set(TEST_RELEASE_URL
    "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/TEST_FILES_BATCHDENTALSEG"
  )

  foreach(testfile
      "PostDentalSurgery_Segmentation.nii.gz"
      "PostDentalSurgery_Segmentation_1_3_5.nii.gz"
    )
    file(DOWNLOAD
      "${TEST_RELEASE_URL}/${testfile}"
      "${TEST_DATA_DIR}/${testfile}"
      SHOW_PROGRESS
      STATUS _download_status
      LOG LOG_DOWNLOAD
    )
    list(GET _download_status 0 _status_code)
    if (NOT _status_code EQUAL 0)
      message(FATAL_ERROR
        "Impossible to download ${testfile}:\n${LOG_DOWNLOAD}"
      )
    endif()
  endforeach()
endif()

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
  Resources/Icons/info.png
  Resources/Icons/loading.gif
  Resources/Icons/start_icon.png

  # on pointe vers les fichiers de Tests/Data qui viennent d'être téléchargés
  Testing/Data/PostDentalSurgery_Segmentation.nii.gz
  Testing/Data/PostDentalSurgery_Segmentation_1_3_5.nii.gz
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
)

if (BUILD_TESTING)
  # Enregistre les tests Python en CTest
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
-----------------------------------------------------------------
