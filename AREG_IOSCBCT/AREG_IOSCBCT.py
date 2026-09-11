#!/usr/bin/env python-real

import os
import sys
import shutil
import argparse
import platform
import logging

import pyvista as pv
import SimpleITK as sitk
import numpy as np
import json
import vtk

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("AREG_IOSCBCT")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# vtkLandmarkTransform needs as many source points as target points: given
# anything else it logs an error on stderr and returns the identity, so the
# mesh is never pre-aligned and nothing says so in the log. ALI_CBCT regularly
# finds only part of the landmarks, so the IOS and CBCT lists match neither in
# size nor in content, and pairing them by position would align landmarks that
# have nothing to do with each other. Only keep what exists on both sides.
MIN_LANDMARK_PAIRS = 3
MAX_LANDMARK_RESIDUAL_MM = float(os.environ.get("AREG_MAX_LANDMARK_RESIDUAL", 10.0))

# How flat a landmark set is allowed to be, as the ratio of its second spread to
# its first. Landmarks strung out along a line leave the rotation about that
# line to be decided by nothing but their noise, and the residual does not show
# it: three points always fit three points exactly, whichever way the arch ends
# up facing. A full arch measures about 0.32 here, half an arch 0.15, and three
# anterior landmarks 0.05 to 0.09.
MIN_LANDMARK_SPREAD_RATIO = 0.10

# An ICP correspondence is kept when the two surfaces face the same way there.
# In a closed bite the opposing crowns sit 1 to 3 mm apart, well inside the
# capture radius, but they face each other: the maxillary occlusal surface
# points down and the mandibular one points up. Without this test a maxillary
# IOS that starts a little low locks onto the mandible and reports a perfect
# fitness while sitting millimetres from the truth.
MIN_NORMAL_AGREEMENT = float(os.environ.get("AREG_MIN_NORMAL_AGREEMENT", 0.5))

# A registration that found next to nothing to match is not a registration. The
# floor is deliberately low: an IOS carries gingiva that no CBCT surface answers
# to, so a healthy run does not match all of its points either.
MIN_ICP_FITNESS = float(os.environ.get("AREG_MIN_ICP_FITNESS", 0.05))


def _labeled_landmarks(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    out = {}
    if data.get("markups"):
        for cp in data["markups"][0].get("controlPoints", []):
            label = cp.get("label")
            if label:
                out[label] = list(cp["position"])
    return out


def _pair_landmarks(cbct_json, ios_json, jaw, patient_id):
    cbct = _labeled_landmarks(cbct_json)
    ios = _labeled_landmarks(ios_json)
    common = [l for l in ios if l in cbct]
    missing = [l for l in ios if l not in cbct]
    logger.info("%s / %s: %d common landmark pair(s) %s%s" % (
        patient_id, jaw, len(common), common,
        "  |  missing from the CBCT: %s" % missing if missing else ""))
    return (common,
            np.array([cbct[l] for l in common], dtype=float).reshape(-1, 3),
            np.array([ios[l] for l in common], dtype=float).reshape(-1, 3))


def _landmark_spread_ratio(landmarks):
    """Second spread over first: 1 is a disc, 0 is a straight line."""
    if len(landmarks) < 3:
        return 0.0
    centred = np.asarray(landmarks, dtype=float)
    centred = centred - centred.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return float(singular[1] / singular[0]) if singular[0] > 0 else 0.0


def _alignment_residual(moving_lms, fixed_lms, matrix):
    """How far each moved landmark still sits from its counterpart, as an RMS.

    A rigid transform preserves distances, so a large residual proves the two
    sets do not describe the same anatomy and the transform fitted to them is a
    meaningless compromise.
    """
    res = []
    for src, dst in zip(moving_lms, fixed_lms):
        moved = matrix.MultiplyPoint([src[0], src[1], src[2], 1])[:3]
        res.append(np.linalg.norm(np.array(moved) - np.array(dst)))
    return float(np.sqrt(np.mean(np.square(res)))) if res else float("inf")


def _write_positions(cbct_json, positions, labels):
    """Store the registered positions in the CBCT json, matched by label.

    This used to walk the CBCT control points and index the IOS array by
    position, so a CBCT holding one landmark against six on the IOS side saved
    the IOS UL1O coordinates under the UR6O label.
    """
    if not cbct_json.get("markups"):
        return
    control_points = cbct_json["markups"][0].get("controlPoints", [])
    if labels is None:
        for i, cp in enumerate(control_points):
            if i < len(positions):
                cp["position"] = list(positions[i])
        return
    by_label = dict(zip(labels, positions))
    kept = []
    for cp in control_points:
        label = cp.get("label")
        if label in by_label:
            cp["position"] = list(by_label[label])
            kept.append(cp)
        else:
            logger.warning("CBCT landmark %s has no IOS counterpart: dropped from the output" % label)
    cbct_json["markups"][0]["controlPoints"] = kept


def align_by_landmarks(moving_mesh, moving_lms, fixed_lms, jaw="", patient_id=""):
    # 1. Create VTK transformation object
    landmark_transform = vtk.vtkLandmarkTransform()
    
    # 2. Assign points (convert landmarks to vtkPoints)
    points_moving = vtk.vtkPoints()
    points_fixed = vtk.vtkPoints()
    
    for p in moving_lms: points_moving.InsertNextPoint(p)
    for p in fixed_lms: points_fixed.InsertNextPoint(p)
    
    landmark_transform.SetSourceLandmarks(points_moving)
    landmark_transform.SetTargetLandmarks(points_fixed)
    landmark_transform.SetModeToRigidBody() # Important: preserve shape (no deformation)
    landmark_transform.Update()
    
    # 3. Apply matrix to moving mesh
    # Get the 4x4 matrix
    matrix = landmark_transform.GetMatrix()

    # Falling back to the identity leaves the ICP to start from the raw pose,
    # which is what already happened whenever the two lists had different
    # sizes -- only now it is a decision, and it is written down.
    n = len(moving_lms)
    if n < MIN_LANDMARK_PAIRS:
        logger.warning(
            "%s / %s: %d landmark pair(s), %d needed. A rigid fit is "
            "underdetermined below that (one point is a plain translation, two "
            "leave a free rotation about the axis). Skipping the pre-alignment."
            % (patient_id, jaw, n, MIN_LANDMARK_PAIRS))
        matrix = vtk.vtkMatrix4x4()
    else:
        rms = _alignment_residual(moving_lms, fixed_lms, matrix)
        if rms > MAX_LANDMARK_RESIDUAL_MM:
            logger.warning(
                "%s / %s: %.1f mm residual over %d pairs (threshold %.1f). The "
                "IOS and CBCT landmarks do not describe the same points. "
                "Skipping the pre-alignment."
                % (patient_id, jaw, rms, n, MAX_LANDMARK_RESIDUAL_MM))
            matrix = vtk.vtkMatrix4x4()
        else:
            logger.info("%s / %s: pre-alignment accepted, %.1f mm residual over %d pairs"
                        % (patient_id, jaw, rms, n))
            # Kept rather than refused: being under-determined is not being
            # wrong, and the identity would start the ICP from the raw pose,
            # which is further still. If the free rotation has in fact turned
            # the arch away, the ICP finds nothing to match and the run says so.
            spread = _landmark_spread_ratio(moving_lms)
            if spread < MIN_LANDMARK_SPREAD_RATIO:
                logger.warning(
                    "%s / %s: the %d landmarks are nearly in a straight line "
                    "(spread %.2f, under the %.2f floor; a full arch measures "
                    "about 0.32). The rotation about that line rests on their "
                    "noise alone, and the %.1f mm residual cannot show it. "
                    "Landmarks further apart on both sides of the arch would "
                    "pin it down."
                    % (patient_id, jaw, n, spread, MIN_LANDMARK_SPREAD_RATIO, rms))
    # Apply it with PyVista
    aligned_mesh = moving_mesh.transform(matrix,inplace=False)
    
    aligned_lms = []
    for p in moving_lms:
        p_transformed = matrix.MultiplyPoint([p[0], p[1], p[2], 1])
        aligned_lms.append(p_transformed[:3])
    
    aligned_lms = np.array(aligned_lms)
    
    return aligned_mesh, matrix, aligned_lms

def _point_normals(mesh, name):
    """Per-point outward normals, in the order of `mesh.points`.

    Cell normals were taken first here and never used, which was as well: they
    are one per triangle, so indexing them with a point index read the normal of
    an unrelated part of the surface.
    """
    if "Normals" in mesh.point_data:
        return np.asarray(mesh.point_data["Normals"], dtype=float)
    try:
        with_normals = mesh.compute_normals(
            point_normals=True, cell_normals=False,
            auto_orient_normals=False, inplace=False)
        return np.asarray(with_normals.point_data["Normals"], dtype=float)
    except Exception as e:
        logger.warning("%s: no surface normals could be computed (%s). The "
                       "opposing arch cannot be told apart by orientation."
                       % (name, e))
        return None


def _point_to_plane_step(source, target, normals):
    """Rigid step minimising the distance to the target's tangent plane.

    Point-to-point pulls a surface towards particular neighbours, which on the
    smooth, near-flat occlusal surfaces here means it slides along them and
    stalls. Measuring along the target normal lets the surface slide freely and
    only resists what actually separates it from the other one.

    Linearised in the rotation: for a correspondence (p, q, n) the residual is
    (p - q).n + w.(p x n) + t.n, which is linear in the six unknowns [w, t].
    """
    A = np.hstack([np.cross(source, normals), normals])
    b = np.einsum("ij,ij->i", target - source, normals)

    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 6 or not np.all(np.isfinite(solution)):
        return None

    omega, translation = solution[:3], solution[3:]
    # A linearised step is only meaningful while it stays small; a large one
    # means the system is being driven by outliers rather than by the surface.
    if np.linalg.norm(omega) > 0.5:
        return None

    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    delta[:3, 3] = translation
    return delta


def _point_to_point_step(source, target):
    """Procrustes fit, the fallback when the point-to-plane system is degenerate."""
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)

    H = (source - source_center).T @ (target - target_center)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    delta = np.eye(4)
    delta[:3, :3] = R
    delta[:3, 3] = target_center - R @ source_center
    return delta


def _icp_run(moving_pts, moving_normals, fixed_pts, fixed_normals, kdtree,
             sign, max_dist, max_iterations, label):
    """One ICP, under one reading of which way the two meshes wind their faces.

    `sign` is +1 when a normal means the same thing on both meshes and -1 when
    one of them is wound the other way; it multiplies the CBCT normals before
    they are compared and before they are used as tangent planes.
    """
    use_normals = moving_normals is not None and fixed_normals is not None

    transformation = np.eye(4)
    moving_pts_transformed = moving_pts.copy()
    moving_normals_transformed = moving_normals.copy() if use_normals else None

    prev_rmse = np.inf
    prev_fitness = 0
    inlier_rmse = float("inf")
    fitness = 0.0
    n_pairs = 0
    rejected_by_normal = 0

    for iteration in range(max_iterations):
        # workers=-1 spreads the query over every core: it is the dominant cost
        # of the loop (one lookup per moving point, per iteration) and the
        # default of a single worker left the other cores idle.
        distances, indices = kdtree.query(moving_pts_transformed, k=1, workers=-1)

        near = distances < max_dist
        valid_mask = near
        if use_normals:
            agreement = np.einsum("ij,ij->i", moving_normals_transformed,
                                  sign * fixed_normals[indices])
            valid_mask = near & (agreement > MIN_NORMAL_AGREEMENT)
            # The peak, not the last iteration: once the arch has settled on
            # its own side nothing nearby faces the wrong way any more, so the
            # final count says nothing about how ambiguous the start was.
            rejected_by_normal = max(
                rejected_by_normal, int(np.sum(near & ~valid_mask)))

        valid_indices = indices[valid_mask]
        valid_moving = moving_pts_transformed[valid_mask]
        valid_distances = distances[valid_mask]

        inlier_rmse = (float(np.sqrt(np.mean(valid_distances ** 2)))
                       if len(valid_distances) > 0 else float("inf"))
        fitness = float(np.sum(valid_mask)) / len(moving_pts)
        n_pairs = int(np.sum(valid_mask))

        if (abs(prev_rmse - inlier_rmse) < 1e-8
                and abs(prev_fitness - fitness) < 1e-8):
            logger.debug("%s: converged at iteration %d" % (label, iteration))
            break

        prev_rmse = inlier_rmse
        prev_fitness = fitness

        if n_pairs < 3:
            logger.debug("%s: iteration %d has %d usable correspondence(s), "
                         "the ICP stops here" % (label, iteration, n_pairs))
            break

        target = fixed_pts[valid_indices]
        delta = None
        if use_normals:
            delta = _point_to_plane_step(
                valid_moving, target, sign * fixed_normals[valid_indices])
        if delta is None:
            delta = _point_to_point_step(valid_moving, target)

        transformation = delta @ transformation

        moving_pts_homogeneous = np.hstack(
            [moving_pts, np.ones((moving_pts.shape[0], 1))])
        moving_pts_transformed = (moving_pts_homogeneous @ transformation.T)[:, :3]
        if use_normals:
            moving_normals_transformed = moving_normals @ transformation[:3, :3].T

    return transformation, {
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "pairs": n_pairs,
        "rejected_by_normal": rejected_by_normal,
        "used_normals": use_normals,
        "sign": sign,
    }


def run_icp_point_to_plane(moving_mesh, fixed_mesh, max_dist=1.5, label=""):
    """Register `moving_mesh` onto `fixed_mesh`.

    Returns the registered mesh, the 4x4 matrix, and what the run is worth:
    fitness (the share of moving points that found a match), the inlier RMSE,
    and how many correspondences the answer stands on.

    Whether a normal points out of the tooth or into it is a property of how
    each file was written, and the two modalities do not have to agree. It
    cannot be read off the starting pose either: that is precisely where an
    IOS sitting between the two arches has most of its nearest neighbours on
    the wrong one, and averaging over them reads the bite as an inversion and
    then locks the registration onto the opposing arch. So both readings are
    registered and the one that actually fits the CBCT better is kept -- the
    IOS crowns are the same anatomy as their own arch in the CBCT and nothing
    else, so the right reading wins on the merits.
    """
    label = label or "IOS"
    moving_pts = np.asarray(moving_mesh.points)
    fixed_pts = np.asarray(fixed_mesh.points)

    fixed_normals = _point_normals(fixed_mesh, "CBCT surface")
    moving_normals = _point_normals(moving_mesh, label)
    use_normals = fixed_normals is not None and moving_normals is not None

    max_iterations = 2000
    # Only the moving points change from one iteration to the next, so the tree
    # over the fixed points is built once instead of being rebuilt up to
    # max_iterations times over the very same coordinates.
    kdtree = cKDTree(fixed_pts)

    attempts = []
    for sign in ((1.0, -1.0) if use_normals else (1.0,)):
        transformation, quality = _icp_run(
            moving_pts, moving_normals, fixed_pts, fixed_normals, kdtree,
            sign, max_dist, max_iterations, label)
        attempts.append((transformation, quality))
        if use_normals:
            logger.debug("%s: normals read as %s gives %.1f%% matched at %.3f mm"
                         % (label, "aligned" if sign > 0 else "opposed",
                            100 * quality["fitness"], quality["inlier_rmse"]))

    # More of the IOS matched is the first thing that matters; a tie on that is
    # broken by how closely it matched.
    transformation, quality = max(
        attempts, key=lambda a: (round(a[1]["fitness"], 3), -a[1]["inlier_rmse"]))

    if use_normals and quality["sign"] < 0:
        logger.info("%s: the IOS and the CBCT surface wind their faces the "
                    "opposite way; the IOS normals were flipped to compare them"
                    % label)

    final_mesh = moving_mesh.transform(transformation, inplace=False)

    logger.info(
        "%s: ICP done, %.1f%% of the IOS matched (%d points) at %.2f mm RMSE%s"
        % (label, 100 * quality["fitness"], quality["pairs"],
           quality["inlier_rmse"],
           ", up to %d nearby points dropped as the opposing surface"
           % quality["rejected_by_normal"] if quality["rejected_by_normal"] else ""))

    return final_mesh, transformation, quality

def save_registered_ios(registered_vtk_upper,registered_vtk_lower,output_path,num_patient):
    file_path_U = os.path.join(output_path,f"{num_patient}_Reg_U.vtk")
    registered_vtk_upper.save(file_path_U)
    file_path_L = os.path.join(output_path,f"{num_patient}_Reg_L.vtk")
    registered_vtk_lower.save(file_path_L)

def apply_matrix_and_save_landmarks(aligned_upper_lm,aligned_lower_lm,mat_u,mat_l,json_output_path,num_patient,landmarks_json_cbct_U,landmarks_json_cbct_L,labels_u=None,labels_l=None):
    # Apply transformations using NumPy (no Open3D dependency)
    # Convert landmarks to homogeneous coordinates, apply transformation, convert back
    
    # Upper landmarks transformation
    aligned_icp_lm_upper_homo = np.hstack([aligned_upper_lm, np.ones((aligned_upper_lm.shape[0], 1))])
    aligned_icp_lm_upper = (aligned_icp_lm_upper_homo @ mat_u.T)[:, :3]
    
    # Lower landmarks transformation
    aligned_icp_lm_lower_homo = np.hstack([aligned_lower_lm, np.ones((aligned_lower_lm.shape[0], 1))])
    aligned_icp_lm_lower = (aligned_icp_lm_lower_homo @ mat_l.T)[:, :3]

    json_output_path_IOS_U = os.path.join(json_output_path,f"{num_patient}_lm_Reg_U.mrk.json")
    json_output_path_IOS_L = os.path.join(json_output_path,f"{num_patient}_lm_Reg_L.mrk.json")

    _write_positions(landmarks_json_cbct_U, aligned_icp_lm_upper, labels_u)

    with open(json_output_path_IOS_U, "w") as file:
        json.dump(landmarks_json_cbct_U, file,indent=4, ensure_ascii=False)

    _write_positions(landmarks_json_cbct_L, aligned_icp_lm_lower, labels_l)

    with open(json_output_path_IOS_L, "w") as file:
        json.dump(landmarks_json_cbct_L, file,indent=4, ensure_ascii=False)

def get_landmarks (json_path):
    with open(json_path, 'r') as f:
        landmarks_json = json.load(f)

    landmarks = []
    if 'markups' in landmarks_json:
        for markup in landmarks_json['markups'][0]['controlPoints']:
            x, y, z = markup['position']
            landmarks.append([x, y, z])
    
    return np.array(landmarks)

def load_data(scan_path,json_path_CBCT_U,json_path_CBCT_L,json_path_IOS_U,json_path_IOS_L):
    
    lm_cbct_U = get_landmarks(json_path_CBCT_U)
    lm_cbct_L = get_landmarks(json_path_CBCT_L)
    lm_ios_U = get_landmarks(json_path_IOS_U)
    lm_ios_L = get_landmarks(json_path_IOS_L)

    image = sitk.ReadImage(scan_path)
    image_array = sitk.GetArrayFromImage(image)
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    ijk_to_lps = np.eye(4)
    ijk_to_lps[:3, :3] = direction @ np.diag(spacing)
    ijk_to_lps[:3, 3] = origin

    vol = pv.wrap(image_array.transpose(2, 1, 0))
    cbct_raw_mesh = vol.contour(isosurfaces=[400])

    cbct_surface = cbct_raw_mesh.transform(ijk_to_lps, inplace=False)

    return lm_cbct_U,lm_cbct_L,lm_ios_U,lm_ios_L,cbct_surface

def getPatients(ios_folder, cbct_folder, ios_lm_folder, cbct_lm_folder):
    """
    Scans the 4 folders and generates a dictionary with patient IDs as keys
    and paths to IOS scan, CBCT scan, and landmarks files as values.
    
    Uses pattern matching to identify timepoints (T0, T1, T2) and jaws (U/u, L/l)
    """
    import re
    
    patients = {}
    
    def extract_timepoint(filename):
        """Extract timepoint from filename (T0, T1, T2, t0, t1, t2)"""
        match = re.search(r'[Tt]([0-2])', filename)
        return match.group(0) if match else None
    
    def extract_jaw(filename):
        """Extract jaw from filename (_u, _U, u_, _l, _L, l_, _upper, _lower)

        The letter has to be a token of its own, delimited by an underscore,
        the start of the name or a dot. Matching a bare "u" anywhere used to
        read "Dupont_003_T1_L.vtk" or "P001_T1_L_Surface.vtk" as upper, which
        registers the lower arch against the upper CBCT landmarks without any
        error being raised.
        """
        if re.search(r'(?:^|_)(?:u|upper)(?=_|\.|$)', filename, re.IGNORECASE):
            return 'upper'
        elif re.search(r'(?:^|_)(?:l|lower)(?=_|\.|$)', filename, re.IGNORECASE):
            return 'lower'
        return None
    
    def extract_patient_id(filename):
        """Extract patient ID from filename (letter + digits before timepoint)"""
        match = re.search(r'([A-Za-z]+)[_]?([0-9]+)[_]?[Tt][0-2]', filename)
        if match:
            return match.group(1) + match.group(2)  # Combine letter and digits
        return None
    
    def normalize_patient_id(patient_id):
        """Normalize patient ID by:
        1. Removing underscores (P_0001 -> P0001)
        2. Removing leading zeros from digits (P0001 -> P1, P00001 -> P1)
        E.g., P001, P_0001, and P00001 all become P1"""
        if patient_id:
            # Remove underscores
            patient_id = patient_id.replace('_', '')
            
            # Separate letters and digits
            match = re.match(r'([A-Za-z]*)([0-9]*)', patient_id)
            if match:
                letters = match.group(1)
                digits = match.group(2)
                if digits:
                    # Remove leading zeros from digits
                    digits = str(int(digits))
                return letters + digits
            return patient_id
        return None
    
    # Parse IOS VTK files
    ios_files = os.listdir(ios_folder)
    for filename in ios_files:
        if filename.endswith('.vtk'):
            timepoint = extract_timepoint(filename)
            jaw = extract_jaw(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            if timepoint and jaw and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                jaw_key = f"ios_{'upper' if jaw == 'upper' else 'lower'}"
                patients[key][jaw_key] = os.path.join(ios_folder, filename)
    
    # Parse CBCT files
    cbct_files = os.listdir(cbct_folder)
    for filename in cbct_files:
        if filename.endswith('.nii.gz'):
            timepoint = extract_timepoint(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            if timepoint and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                patients[key]["cbct"] = os.path.join(cbct_folder, filename)
    
    # Parse IOS JSON landmarks
    ios_lm_files = os.listdir(ios_lm_folder)
    for filename in ios_lm_files:
        if filename.endswith('.json'):
            timepoint = extract_timepoint(filename)
            jaw = extract_jaw(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            # Check if it's an IOS landmark file
            if timepoint and jaw and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                jaw_key = f"ios_lm_{'upper' if jaw == 'upper' else 'lower'}"
                patients[key][jaw_key] = os.path.join(ios_lm_folder, filename)
    
    # Parse CBCT JSON landmarks
    cbct_lm_files = os.listdir(cbct_lm_folder)
    for filename in cbct_lm_files:
        if filename.endswith('.json'):
            timepoint = extract_timepoint(filename)
            jaw = extract_jaw(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            # Check if it's a CBCT landmark file
            if timepoint and jaw and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                jaw_key = f"cbct_lm_{'upper' if jaw == 'upper' else 'lower'}"
                patients[key][jaw_key] = os.path.join(cbct_lm_folder, filename)
    
    # Log found patients
    logger.info(f"Found {len(patients)} patients")
    for patient_key in sorted(patients.keys()):
        logger.debug(f"Patient {patient_key}: {patients[patient_key]}")
    
    return patients

def main(args):
    patients = getPatients(args.IOS_folder, args.CBCT_folder, args.IOS_lm_folder, args.CBCT_lm_folder)
    logger.info("Running AREG_IOSCBCT for all patients")

    if not patients:
        logger.warning("No files to process has been found. Please check the input folders and folder_name")
    
    # Create output directory if it doesn't exist
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Process each patient
    registered = 0
    for patient_id, patient_data in sorted(patients.items()):
        logger.info(f"Processing patient {patient_id}...")
        
        try:
            # 1. LOAD DATA
            logger.debug(f"Loading data for patient {patient_id}")
            lm_cbct_U, lm_cbct_L, lm_ios_U, lm_ios_L, cbct_surface = load_data(
                patient_data["cbct"],
                patient_data["cbct_lm_upper"],
                patient_data["cbct_lm_lower"],
                patient_data["ios_lm_upper"],
                patient_data["ios_lm_lower"]
            )
            
            # Keep only the landmarks present on both sides, in the same order
            labels_U, lm_cbct_U, lm_ios_U = _pair_landmarks(
                patient_data["cbct_lm_upper"], patient_data["ios_lm_upper"], "Upper", patient_id)
            labels_L, lm_cbct_L, lm_ios_L = _pair_landmarks(
                patient_data["cbct_lm_lower"], patient_data["ios_lm_lower"], "Lower", patient_id)

            # Load IOS meshes (VTK files)
            ios_upper_mesh = pv.read(patient_data["ios_upper"])
            ios_lower_mesh = pv.read(patient_data["ios_lower"])
            logger.debug(f"Loaded IOS upper and lower meshes")
            logger.debug(f"IOS Upper mesh: n_points={ios_upper_mesh.n_points}, n_cells={ios_upper_mesh.n_cells}, bounds={ios_upper_mesh.bounds}")
            logger.debug(f"IOS Lower mesh: n_points={ios_lower_mesh.n_points}, n_cells={ios_lower_mesh.n_cells}, bounds={ios_lower_mesh.bounds}")
            
            # 2. ALIGN BY LANDMARKS
            logger.debug(f"Aligning IOS upper jaw by landmarks")
            aligned_ios_upper, mat_ios_upper, aligned_lms_ios_upper = align_by_landmarks(
                ios_upper_mesh, lm_ios_U, lm_cbct_U, "Upper", patient_id
            )
            logger.info(f"IOS Upper landmarks after alignment:\n{aligned_lms_ios_upper}")
            
            logger.debug(f"Aligning IOS lower jaw by landmarks")
            aligned_ios_lower, mat_ios_lower, aligned_lms_ios_lower = align_by_landmarks(
                ios_lower_mesh, lm_ios_L, lm_cbct_L, "Lower", patient_id
            )
            logger.debug(f"IOS Lower landmarks after alignment shape: {aligned_lms_ios_lower.shape}")
            logger.debug(f"IOS Lower landmarks after alignment:\n{aligned_lms_ios_lower}")
            
            # 3. RUN ICP REGISTRATION
            logger.debug(f"Running ICP for upper jaw")
            registered_ios_upper, mat_icp_upper, quality_upper = run_icp_point_to_plane(
                aligned_ios_upper, cbct_surface, max_dist=1.0,
                label=f"{patient_id} / Upper"
            )
            
            logger.debug(f"Running ICP for lower jaw")
            registered_ios_lower, mat_icp_lower, quality_lower = run_icp_point_to_plane(
                aligned_ios_lower, cbct_surface, max_dist=1.0,
                label=f"{patient_id} / Lower"
            )
            logger.info(f"ICP registration completed for patient {patient_id}")

            # An ICP that matched nothing still returns a matrix, and writing it
            # out put an untouched IOS in the results folder under the name of a
            # registered one. It happens whenever the pre-alignment is skipped
            # and the raw pose is nowhere near the CBCT.
            starved = [jaw for jaw, quality in (("Upper", quality_upper),
                                                ("Lower", quality_lower))
                       if quality["fitness"] < MIN_ICP_FITNESS]
            if starved:
                raise RuntimeError(
                    "the %s arch matched under %.0f%% of its points to the CBCT "
                    "surface (upper %.1f%%, lower %.1f%%). Nothing was written "
                    "for this patient: check its landmarks, the pre-alignment "
                    "above says whether it was accepted."
                    % (" and ".join(starved), 100 * MIN_ICP_FITNESS,
                       100 * quality_upper["fitness"], 100 * quality_lower["fitness"]))

            # 4. SAVE RESULTS
            logger.info(f"Saving registered meshes and landmarks")
            
            # Save registered meshes
            save_registered_ios(registered_ios_upper, registered_ios_lower,output_dir,patient_id)
            
            # Load landmark JSON files to update them
            with open(patient_data["cbct_lm_upper"], 'r') as f:
                landmarks_json_cbct_U = json.load(f)
            with open(patient_data["cbct_lm_lower"], 'r') as f:
                landmarks_json_cbct_L = json.load(f)
            
            # Save registered landmarks
            apply_matrix_and_save_landmarks(
                aligned_lms_ios_upper, aligned_lms_ios_lower,
                mat_icp_upper, mat_icp_lower,
                output_dir, patient_id,
                landmarks_json_cbct_U, landmarks_json_cbct_L,
                labels_U, labels_L
            )
            registered += 1
            logger.info(f"Patient {patient_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}", exc_info=True)
            continue
    
    logger.info("AREG_IOSCBCT processing completed")
    # Every patient can fail on a missing landmark file and the loop still ends
    # normally, which used to exit 0 and let Slicer report the whole pipeline as
    # a success while the output folder stayed empty.
    if not registered:
        raise RuntimeError(
            "No patient could be registered: check that every patient has an "
            "IOS surface, a CBCT, and the landmark files for both arches.")
    logger.info(f"{registered}/{len(patients)} patient(s) registered")


if __name__ == "__main__":
    try:
        logger.info("AREG_IOSCBCT entry point initiated")
        
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("IOS_folder", type=str)
            parser.add_argument("CBCT_folder", type=str)
            parser.add_argument("IOS_lm_folder", type=str)
            parser.add_argument("CBCT_lm_folder", type=str)
            parser.add_argument("output", type=str)

            args = parser.parse_args()
            logger.debug(f"Arguments parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing command line arguments: {e}")
            raise

        try:
            logger.info("Calling main() function")
            main(args)
            logger.info("AREG_IOSCBCT completed successfully")
        except Exception as e:
            logger.error(f"Error in main() execution: {e}")
            raise

    except SystemExit as e:
        logger.info(f"Script exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error in entry point: {e}")
        sys.exit(f"Fatal error: {e}")
