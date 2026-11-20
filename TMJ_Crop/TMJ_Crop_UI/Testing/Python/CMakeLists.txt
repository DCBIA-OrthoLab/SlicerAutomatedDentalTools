# The crop on a synthetic head, and the batch around it: no scan, no weights.
slicer_add_python_unittest(SCRIPT test_crop.py)

# What the panel offers is what the VISOR tool offers: the same options,
# the same extra outputs, and no scale_by or scale on the screen.
slicer_add_python_unittest(SCRIPT test_options.py)

# The model fields fill from what a Download button already left on disk.
slicer_add_python_unittest(SCRIPT test_downloads.py)
