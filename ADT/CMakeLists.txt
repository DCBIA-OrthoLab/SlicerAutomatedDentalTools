#-----------------------------------------------------------------------------
# Shared Python code for every module of the extension.
#
# Built as a scripted module with no module of its own: the macro installs the
# listed files under qt-scripted-modules/ADTLib/, which Slicer has on sys.path,
# so `import ADTLib` works from any module, from a scripted CLI, and from the
# Conda environment once the search paths are published into it.
#
# The directory holding this file is named ADT, not ADTLib, on purpose. A
# directory carrying the package's own name shadows it as a namespace package
# the moment its parent lands on sys.path, and the package then imports with an
# empty __file__ and no submodules. Every other package in this repository is
# built the same way -- AREG_IOS/AREG_IOS_utils, ALI_IOS/ALI_IOS_utils -- with
# a holder directory whose name differs from the package it carries.
#
# Declared before its consumers in the top-level CMakeLists.txt.
#-----------------------------------------------------------------------------
set(MODULE_NAME ADTLib)

#-----------------------------------------------------------------------------
# Every file has to be listed: this is a manifest, nothing checks it against
# the source tree, and a file left out is simply absent from an installed
# build while it keeps working in a development tree.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}/__init__.py
  ${MODULE_NAME}/env/__init__.py
  ${MODULE_NAME}/env/conda.py
  ${MODULE_NAME}/env/cuda.py
  ${MODULE_NAME}/env/deps.py
  ${MODULE_NAME}/env/install_pytorch.py
  ${MODULE_NAME}/format.py
  ${MODULE_NAME}/geometry.py
  ${MODULE_NAME}/io/__init__.py
  ${MODULE_NAME}/io/fs.py
  ${MODULE_NAME}/io/landmarks.py
  ${MODULE_NAME}/io/surface.py
  ${MODULE_NAME}/labels.py
  ${MODULE_NAME}/logging_setup.py
  ${MODULE_NAME}/method.py
  ${MODULE_NAME}/model_registry.py
  ${MODULE_NAME}/naming.py
  ${MODULE_NAME}/progress_protocol.py
  ${MODULE_NAME}/requests.py
  ${MODULE_NAME}/testdata.py
  ${MODULE_NAME}/theming.py
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
