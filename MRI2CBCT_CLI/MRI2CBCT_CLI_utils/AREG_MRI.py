import argparse
import os
import itk
import SimpleITK as sitk
import numpy as np

def ComputeFinalMatrix(Transforms):
    """Compute the final matrix from the list of matrices and translations"""
    Rotation, Translation = [], []
    for i in range(len(Transforms)):
        Rotation.append(Transforms[i].GetMatrix())
        Translation.append(Transforms[i].GetTranslation())

    # Compute the final rotation matrix
    final_rotation = np.reshape(np.asarray(Rotation[0]), (3, 3))
    for i in range(1, len(Rotation)):
        final_rotation = final_rotation @ np.reshape(np.asarray(Rotation[i]), (3, 3))

    # Compute the final translation matrix
    final_translation = np.reshape(np.asarray(Translation[0]), (1, 3))
    for i in range(1, len(Translation)):
        final_translation = final_translation + np.reshape(
            np.asarray(Translation[i]), (1, 3)
        )

    # Create the final transform
    final_transform = sitk.Euler3DTransform()
    final_transform.SetMatrix(final_rotation.flatten().tolist())
    final_transform.SetTranslation(final_translation[0].tolist())

    return final_transform


def ElastixReg(fixed_image, moving_image, initial_transform=None):
    """Perform a registration using elastix with a rigid transform and possibly an initial transform"""

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)

    # ParameterMap
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    parameter_object.SetParameter("ErodeMask", "true")
    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("MaximumNumberOfIterations", "10000")
    parameter_object.SetParameter("NumberOfResolutions", "1")
    parameter_object.SetParameter("NumberOfSpatialSamples", "10000")

    elastix_object.SetParameterObject(parameter_object)
    if initial_transform is not None:
        elastix_object.SetInitialTransformParameterObject(initial_transform)

    # Additional parameters
    elastix_object.SetLogToConsole(False)

    # Execute registration
    elastix_object.UpdateLargestPossibleRegion()

    TransParamObj = elastix_object.GetTransformParameterObject()

    return TransParamObj

def MatrixRetrieval(TransformParameterMapObject):
    """Retrieve the matrix from the transform parameter map"""
    ParameterMap = TransformParameterMapObject.GetParameterMap(0)

    if ParameterMap["Transform"][0] == "AffineTransform":
        matrix = [float(i) for i in ParameterMap["TransformParameters"]]
        # Convert to a sitk transform
        transform = sitk.AffineTransform(3)
        transform.SetParameters(matrix)

    elif ParameterMap["Transform"][0] == "EulerTransform":
        A = [float(i) for i in ParameterMap["TransformParameters"][0:3]]
        B = [float(i) for i in ParameterMap["TransformParameters"][3:6]]
        # Convert to a sitk transform
        transform = sitk.Euler3DTransform()
        transform.SetRotation(angleX=A[0], angleY=A[1], angleZ=A[2])
        transform.SetTranslation(B)

    return transform

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def registration(cbct_folder,mri_folder,cbct_mask_folder,output_folder,mri_original_folder):
    """
    Registers CBCT and MRI images using CBCT masks, saving the results in the specified output folder.

    Arguments:
    cbct_folder (str): Folder containing CBCT files (.nii.gz).
    mri_folder (str): Folder containing corresponding MRI files (.nii.gz).
    cbct_mask_folder (str): Folder containing CBCT masks (.nii.gz).
    output_folder (str): Folder to save the registration results.
    mri_original_folder (str): Folder containing original MRI files (.nii.gz), if available.

    For each CBCT file in cbct_folder:
    - Extract patient ID from the filename.
    - Find corresponding MRI and CBCT mask files.
    - Optionally, find the original MRI file.
    - Call process_images to perform registration and save the results.
    """
    
    for cbct_file in os.listdir(cbct_folder):
        if cbct_file.endswith(".nii.gz") and "_CBCT_" in cbct_file:
            patient_id = cbct_file.split("_CBCT_")[0]
        
            mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")
            if mri_original_folder!="None":
                mri_path_original = get_corresponding_file(mri_original_folder, patient_id, "_MR_")
            

            cbct_mask_path = get_corresponding_file(cbct_mask_folder, patient_id, "_CBCT_")

            process_images(mri_path, cbct_mask_path, output_folder,patient_id,mri_path_original,)

def process_images(mri_path, cbct_mask_path, output_folder, patient_id,mri_path_original):
    """
    Processes MRI and CBCT mask images, performs registration, and saves the results.

    Arguments:
    mri_path (str): Path to the MRI file.
    cbct_mask_path (str): Path to the CBCT mask file.
    output_folder (str): Folder to save the registration results.
    patient_id (str): Identifier for the patient.
    mri_path_original (str): Path to the original MRI file.

    Steps:
    - Reads the MRI and CBCT mask images.
    - Performs registration using Elastix.
    - Retrieves the transformation matrix and computes the final transform.
    - Saves the transformed image and transformation matrix in the output folder.
    """
    
    try : 
        mri_path = itk.imread(mri_path, itk.F)
        cbct_mask_path = itk.imread(cbct_mask_path, itk.F)
    except KeyError as e:
        print("An error occurred while reading the images of the patient : {patient_id}")
        print(e)
        print(f"{patient_id} failed")
        return

    Transforms = []

    try : 
        TransformObj_Fine = ElastixReg(cbct_mask_path, mri_path, initial_transform=None)
    except Exception as e:
        print("An error occurred during the registration process on the patient {patient_id} :")
        print(e)
        return
    
    transforms_Fine = MatrixRetrieval(TransformObj_Fine)
    Transforms.append(transforms_Fine)
    transform = ComputeFinalMatrix(Transforms)
    
    os.makedirs(output_folder, exist_ok=True)
    
    output_image_path = os.path.join(output_folder,os.path.basename(mri_path_original).replace('.nii.gz', f'_reg.nii.gz'))
    output_image_path_transform = os.path.join(output_folder,os.path.basename(mri_path_original).replace('.nii.gz', f'_reg_transform.tfm'))
    
    sitk.WriteTransform(transform, output_image_path_transform)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AREG MRI folder')

    parser.add_argument("--cbct_folder", type=str,  help="Folder containing CBCT images.", default=".")
    parser.add_argument("--cbct_mask_folder", type=str, help="Folder containing CBCT masks.", default=".")
    
    parser.add_argument("--mri_folder", type=str,  help="Folder containing MRI images.", default=".")
    parser.add_argument("--mri_original_folder", type=str,  help="Folder containing original MRI.", default=".")
    
    parser.add_argument("--output_folder", type=str,  help="Folder to save the output files.",default=".")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    cbct_folder = args.cbct_folder
    mri_folder = args.mri_folder
    cbct_mask_folder = args.cbct_mask_folder
    output_folder = args.output_folder
    mri_original_folder = args.mri_original_folder
        
    registration(cbct_folder,mri_folder,cbct_mask_folder,output_folder,mri_original_folder)
