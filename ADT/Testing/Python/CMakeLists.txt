
# The identifier read off a file name, checked against the six chains it
# replaces: standard library only, no scan and no Slicer.
slicer_add_python_unittest(SCRIPT test_naming.py)

# check_lib_installed: version constraints, and the inherited spellings.
slicer_add_python_unittest(SCRIPT test_deps.py)

# The progress channel, at both ends: what ALI CBCT prints, and what its
# widget makes of it. The two disagreed and the bar never moved.
slicer_add_python_unittest(SCRIPT test_ali_progress_contract.py)

# ASO IOS keeps the landmarks both files carry, and refuses below three --
# which is what the CBCT path always did.
slicer_add_python_unittest(SCRIPT test_aso_ios_landmark_intersection.py)

# Which torch wheel channel a GPU gets: Blackwell needs cu128, and Maxwell,
# Pascal and Volta must never be moved onto it.
slicer_add_python_unittest(SCRIPT test_cuda_channel.py)

# A `with` that works: decorated and yielding, or neither. VFACE carried
# both halves of the mistake, and neither raised until the step ran.
slicer_add_python_unittest(SCRIPT test_context_managers.py)

# The Semi-Automated ASO CBCT mode needs no prior transform: the published
# test set has none, and the identity is the neutral value of that step.
slicer_add_python_unittest(SCRIPT test_aso_cbct_optional_transform.py)

# ReadSurf, WriteSurf and OFFReader: the five copies had drifted apart, these
# cases freeze what was kept and the four behaviours that change.
slicer_add_python_unittest(SCRIPT test_surface.py)

# search: the two parameters that carry what the fifteen copies did
# differently, and nothing else.
slicer_add_python_unittest(SCRIPT test_fs.py)

# The tooth numbering array: the preferred array was ignored as soon as it was
# not the last array of the mesh.
slicer_add_python_unittest(SCRIPT test_labels.py)

# The <filter-progress> channel: the bytes emitted are what counts, the window
# comparing the value it receives against hard-coded integers.
slicer_add_python_unittest(SCRIPT test_progress_protocol.py)

# The typed requests: what matters is that an absent field raises exactly what
# the dictionary raised.
slicer_add_python_unittest(SCRIPT test_requests.py)

# Walks the repository: every (library, version) pair must produce a string
# that pip accepts. What grepping had missed twice.
slicer_add_python_unittest(SCRIPT test_pip_requirements.py)

# Fetching a test data set: archive or not, an HTML page refused, and an
# interrupted download that must not pass for a complete one.
slicer_add_python_unittest(SCRIPT test_testdata.py)

# The button sheet VFACE carried four times: one template, two accents.
# Pinned because it is what the user sees.
slicer_add_python_unittest(SCRIPT test_button_style.py)

# What stops an ALI CBCT search: a number of steps, not wall-clock seconds.
# The same scan must place the same landmark on a busy machine and on an idle
# one.
slicer_add_python_unittest(SCRIPT test_ali_cbct_budget.py)

# A landmark ALI CBCT did not find is absent from the output file with
# nothing to say so: the list now goes out beside the predictions.
slicer_add_python_unittest(SCRIPT test_ali_cbct_missing.py)

# A landmark whose name has no group: written out or not, it must not cost
# the patient the thirty others that were found.
slicer_add_python_unittest(SCRIPT test_ali_cbct_unknown_group.py)

# ALI's not-found report shares the folder with the landmarks: the readers must
# skip it, not die on its missing "markups" key nor hand it to ICP.
slicer_add_python_unittest(SCRIPT test_aso_cbct_report_is_not_landmarks.py)
