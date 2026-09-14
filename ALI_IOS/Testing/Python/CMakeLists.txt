
# Geometry tests of the filter that keeps a marked face only when it can
# belong to the tooth aimed at: synthetic, no scan, no model and no GPU.
slicer_add_python_unittest(SCRIPT test_pick_patch.py)

# What a rebuilt landmark may span, and what it may not.
slicer_add_python_unittest(SCRIPT test_fill_gaps.py)

# What painting the line onto the user's own scan may and may not do to it.
slicer_add_python_unittest(SCRIPT test_paint_scan.py)

# Placing a landmark the prediction could not give.
slicer_add_python_unittest(SCRIPT test_complete_line.py)
