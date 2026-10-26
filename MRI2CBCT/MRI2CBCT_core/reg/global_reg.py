"""Global feature-based registration: FPFH descriptors + RANSAC, then ICP.

Why this and not plain ICP: ICP is LOCAL. It polishes whatever pose you hand it
and never asks whether the fragment belongs somewhere else, which is why it
settled on a 172-degree flip for B108. A human matching these masks by eye is
doing a GLOBAL search over shape correspondences.

FPFH is a rotation-invariant descriptor of local surface geometry. Matching
descriptors between the two masks gives candidate correspondences; RANSAC then
finds the pose that the largest consistent set of them agrees on. That handles
partial overlap and unknown initial pose - the textbook tool for this problem.
ICP runs afterwards only to refine.
"""
import numpy as np
import open3d as o3d


def to_cloud(pts, nrm, voxel):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    p.normals = o3d.utility.Vector3dVector(np.asarray(nrm, dtype=np.float64))
    p = p.voxel_down_sample(voxel)
    p.orient_normals_consistent_tangent_plane(20)
    return p


def register(mri_pts, mri_nrm, cb_pts, cb_nrm, voxel=1.5, seed=0):
    """Rigid transform MRI -> CBCT. Returns (4x4, fitness, inlier_rmse)."""
    src = to_cloud(mri_pts, mri_nrm, voxel)
    tgt = to_cloud(cb_pts, cb_nrm, voxel)
    r = voxel * 5
    fs = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    ft = o3d.pipelines.registration.compute_fpfh_feature(
        tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    dist = voxel * 1.5
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, fs, ft, True, dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999))
    # refine with point-to-plane ICP from the global solution
    fine = o3d.pipelines.registration.registration_icp(
        src, tgt, voxel * 1.5, res.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return np.asarray(fine.transformation), float(fine.fitness), float(fine.inlier_rmse)
