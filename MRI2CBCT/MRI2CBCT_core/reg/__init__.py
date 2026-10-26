"""FPFH -> RANSAC -> point-to-plane ICP registration of an MRI cranial-base mask
onto a CBCT cranial-base mask. See ../REGISTRATION.md."""
from .pipeline import register_joint, cbct_surface, qc_tag
from .itk_io import write_tfm, ras_to_itk, slicer_matrix, resample_onto
from .surface import surface_points_normals, largest_cc

__all__ = ["register_joint", "cbct_surface", "qc_tag", "write_tfm", "ras_to_itk",
           "slicer_matrix", "resample_onto", "surface_points_normals", "largest_cc"]
