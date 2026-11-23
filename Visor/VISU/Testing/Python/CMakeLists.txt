#-----------------------------------------------------------------------------
# test_index.py imports VISULib.index only, which is `os` and `re` and nothing
# else. test_panel.py does need qt/ctk/slicer, but only as the stand-ins in
# ServerToolsCore's qt_stubs.py. Both are therefore plain Python3 ctests rather
# than slicer_add_python_unittest — no need to pay for a full Slicer
# interpreter launch. Same rationale as ServerToolsCore/Testing/Python, see
# ARCHITECTURE.md, "Debatable decisions".
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
  NAME VISU_test_review
  COMMAND ${Python3_EXECUTABLE} -m unittest test_review
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_test(
  NAME VISU_test_index
  COMMAND ${Python3_EXECUTABLE} -m unittest test_index
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_test(
  NAME VISU_test_edits
  COMMAND ${Python3_EXECUTABLE} -m unittest test_edits
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_test(
  NAME VISU_test_panel
  COMMAND ${Python3_EXECUTABLE} -m unittest test_panel
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
