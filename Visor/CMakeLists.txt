#-----------------------------------------------------------------------------
# The VISOR local set: tools that run on this machine, from SADT-VISOR.
#
# Nothing here replaces a classic module. Every module of this folder has its own
# name (`ALIVisor`, `AREGVisor`, ...); the classic ones (`ALI`, `AREG`, ...) are
# built from the extension's root as upstream has them. See
# ServerToolsCoreLib/mode_switch.py for the bar that links the two.
#-----------------------------------------------------------------------------

# Shared code first: every module below imports ServerToolsCoreLib.
add_subdirectory(ServerToolsCore)
add_subdirectory(SADTToolsManager)
add_subdirectory(VISU)
add_subdirectory(Crown_Seg)
add_subdirectory(Convert)

add_subdirectory(ALIVisor)
add_subdirectory(AMASSSVisor)
add_subdirectory(AREGVisor)
add_subdirectory(ASOVisor)
add_subdirectory(AutoCrop3DVisor)
add_subdirectory(AutoMatrixVisor)
add_subdirectory(BATCHDENTALSEGVisor)
add_subdirectory(CLICVisor)
add_subdirectory(CNEVisor)
add_subdirectory(DOCShapeAXIVisor)
add_subdirectory(FlexRegVisor)
add_subdirectory(GreedyRegVisor)
add_subdirectory(SurgMovPredVisor)
add_subdirectory(TMJ_CropVisor)
add_subdirectory(VFACEVisor)
