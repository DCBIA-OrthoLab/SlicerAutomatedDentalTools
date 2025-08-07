#-----------------------------------------------------------------------------
set(MODULE_NAME MRI2CBCT_CLI_utils)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}/__init__.py
  ${MODULE_NAME}/apply_mask.py
  ${MODULE_NAME}/approx_utils.py
  ${MODULE_NAME}/approximate.py
  ${MODULE_NAME}/AREG_MRI.py
  ${MODULE_NAME}/crop_approximation.py
  ${MODULE_NAME}/LR_crop.py
  ${MODULE_NAME}/mri_inverse.py
  ${MODULE_NAME}/nmi.py
  ${MODULE_NAME}/normalize_percentile.py
  ${MODULE_NAME}/resample_create_csv.py
  ${MODULE_NAME}/resample.py
  ${MODULE_NAME}/TMJ_crop.py
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
