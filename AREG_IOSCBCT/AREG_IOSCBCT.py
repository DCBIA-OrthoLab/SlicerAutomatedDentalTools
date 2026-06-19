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

def align_by_landmarks(moving_mesh, moving_lms, fixed_lms):
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
    # Apply it with PyVista
    aligned_mesh = moving_mesh.transform(matrix,inplace=False)
    
    aligned_lms = []
    for p in moving_lms:
        p_transformed = matrix.MultiplyPoint([p[0], p[1], p[2], 1])
        aligned_lms.append(p_transformed[:3])
    
    aligned_lms = np.array(aligned_lms)
    
    return aligned_mesh, matrix, aligned_lms

def run_icp_point_to_plane(moving_mesh, fixed_mesh, max_dist=1.5):
    """
    ICP Point-to-Plane
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    
    # 1. Extract points
    moving_pts = np.asarray(moving_mesh.points)
    fixed_pts = np.asarray(fixed_mesh.points)
    
    fixed_mesh_copy = fixed_mesh.copy()
    fixed_mesh_copy.compute_normals(inplace=True)
    fixed_normals = np.asarray(fixed_mesh_copy.cell_data['Normals'] 
                               if 'Normals' in fixed_mesh_copy.cell_data 
                               else fixed_mesh_copy.point_data['Normals'])
    
    transformation = np.eye(4)
    moving_pts_transformed = moving_pts.copy()
    
    print("Optimisation Point-to-Plane on going...")
    
    max_iterations = 2000
    prev_rmse = np.inf
    rmse_threshold = 1e-8
    fitness_threshold = 1e-8
    prev_fitness = 0
    
    for iteration in range(max_iterations):
        # 2. Find correspondances (nearest neighbors)
        kdtree = cKDTree(fixed_pts)
        distances, indices = kdtree.query(moving_pts_transformed, k=1)
        
        # Filter the points too far
        valid_mask = distances < max_dist
        valid_indices = indices[valid_mask]
        valid_moving = moving_pts_transformed[valid_mask]
        valid_distances = distances[valid_mask]
        
        # Calcul fitness and RMSE
        inlier_rmse = np.sqrt(np.mean(valid_distances**2)) if len(valid_distances) > 0 else np.inf
        fitness = np.sum(valid_mask) / len(moving_pts)
        
        # check convergence
        rmse_change = abs(prev_rmse - inlier_rmse)
        fitness_change = abs(prev_fitness - fitness)
        
        if rmse_change < rmse_threshold and fitness_change < fitness_threshold:
            print(f"Convergence reached at iteration {iteration}")
            break
        
        prev_rmse = inlier_rmse
        prev_fitness = fitness
        
        # 3. Calcul of the transformation (Point-to-Plane with SVD)
        if len(valid_indices) < 3:
            print(f"Iteration {iteration}: Not enough correspondances")
            break
        
        # corresponding Points 
        source = valid_moving
        target = fixed_pts[valid_indices]
        
        # Center points
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)
        source_centered = source - source_center
        target_centered = target - target_center
        
        # covariance Matrix 
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Assure a clean rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = target_center - R @ source_center
        
        # 4. Apply the transformation
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = R
        delta_transform[:3, 3] = t
        
        transformation = delta_transform @ transformation
        
        # Transform points
        moving_pts_homogeneous = np.hstack([moving_pts, np.ones((moving_pts.shape[0], 1))])
        moving_pts_transformed = (moving_pts_homogeneous @ transformation.T)[:, :3]
        
        if (iteration + 1) % 100 == 0 or iteration < 10:
            print(f"  Iteration {iteration + 1}: RMSE = {inlier_rmse:.6f}, Fitness = {fitness:.4f}")
    
    # 5. Apply final transformation
    final_mesh = moving_mesh.transform(transformation, inplace=False)
    
    print(f"ICP Finished. Fitness: {fitness:.4f}, Inlier RMSE: {inlier_rmse:.4f}")
    
    return final_mesh, transformation

def save_registered_ios(registered_vtk_upper,registered_vtk_lower,output_path,num_patient):
    file_path_U = os.path.join(output_path,f"{num_patient}_Reg_U.vtk")
    registered_vtk_upper.save(file_path_U)
    file_path_L = os.path.join(output_path,f"{num_patient}_Reg_L.vtk")
    registered_vtk_lower.save(file_path_L)

def apply_matrix_and_save_landmarks(aligned_upper_lm,aligned_lower_lm,mat_u,mat_l,json_output_path,num_patient,landmarks_json_cbct_U,landmarks_json_cbct_L):
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

    if 'markups' in landmarks_json_cbct_U:
        i=0
        for markup in landmarks_json_cbct_U['markups'][0]['controlPoints']:
            markup['position'] = list(aligned_icp_lm_upper[i])
            i+=1

    with open(json_output_path_IOS_U, "w") as file:
        json.dump(landmarks_json_cbct_U, file,indent=4, ensure_ascii=False)

    if 'markups' in landmarks_json_cbct_L:
        i=0
        for markup in landmarks_json_cbct_L['markups'][0]['controlPoints']:
            markup['position'] = list(aligned_icp_lm_lower[i])
            i+=1

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
        """Extract jaw from filename (_u, _U, u_, _l, _L, l_)"""
        # Check for patterns: _u, _U, u_, _l, _L, l_
        if re.search(r'[_]?[uU][_]?', filename):
            return 'upper'
        elif re.search(r'[_]?[lL][_]?', filename):
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
            
            # Load IOS meshes (VTK files)
            ios_upper_mesh = pv.read(patient_data["ios_upper"])
            ios_lower_mesh = pv.read(patient_data["ios_lower"])
            logger.debug(f"Loaded IOS upper and lower meshes")
            logger.debug(f"IOS Upper mesh: n_points={ios_upper_mesh.n_points}, n_cells={ios_upper_mesh.n_cells}, bounds={ios_upper_mesh.bounds}")
            logger.debug(f"IOS Lower mesh: n_points={ios_lower_mesh.n_points}, n_cells={ios_lower_mesh.n_cells}, bounds={ios_lower_mesh.bounds}")
            
            # 2. ALIGN BY LANDMARKS
            logger.debug(f"Aligning IOS upper jaw by landmarks")
            aligned_ios_upper, mat_ios_upper, aligned_lms_ios_upper = align_by_landmarks(
                ios_upper_mesh, lm_ios_U, lm_cbct_U
            )
            logger.info(f"IOS Upper landmarks after alignment:\n{aligned_lms_ios_upper}")
            
            logger.debug(f"Aligning IOS lower jaw by landmarks")
            aligned_ios_lower, mat_ios_lower, aligned_lms_ios_lower = align_by_landmarks(
                ios_lower_mesh, lm_ios_L, lm_cbct_L
            )
            logger.debug(f"IOS Lower landmarks after alignment shape: {aligned_lms_ios_lower.shape}")
            logger.debug(f"IOS Lower landmarks after alignment:\n{aligned_lms_ios_lower}")
            
            # 3. RUN ICP REGISTRATION
            logger.debug(f"Running ICP for upper jaw")
            registered_ios_upper, mat_icp_upper = run_icp_point_to_plane(
                aligned_ios_upper, cbct_surface, max_dist=1.0
            )
            
            logger.debug(f"Running ICP for lower jaw")
            registered_ios_lower, mat_icp_lower = run_icp_point_to_plane(
                aligned_ios_lower, cbct_surface, max_dist=1.0
            )
            logger.info(f"ICP registration completed for patient {patient_id}")
            
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
                landmarks_json_cbct_U, landmarks_json_cbct_L
            )
            logger.info(f"Patient {patient_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}", exc_info=True)
            continue
    
    logger.info("AREG_IOSCBCT processing completed")


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
