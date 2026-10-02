
#slicer_add_python_unittest(SCRIPT ${MODULE_NAME}ModuleTest.py)

# The interface description each method carries: the values are the ones the
# widget's `if/elif` chain decided.
slicer_add_python_unittest(SCRIPT test_ui_description.py)
