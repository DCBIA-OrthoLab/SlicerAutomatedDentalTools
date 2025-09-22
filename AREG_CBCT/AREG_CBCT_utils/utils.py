"""
8888888 888b     d888 8888888b.   .d88888b.  8888888b.  88888888888  .d8888b.
  888   8888b   d8888 888   Y88b d88P" "Y88b 888   Y88b     888     d88P  Y88b
  888   88888b.d88888 888    888 888     888 888    888     888     Y88b.
  888   888Y88888P888 888   d88P 888     888 888   d88P     888      "Y888b.
  888   888 Y888P 888 8888888P"  888     888 8888888P"      888         "Y88b.
  888   888  Y8P  888 888        888     888 888 T88b       888           "888
  888   888   "   888 888        Y88b. .d88P 888  T88b      888     Y88b  d88P
8888888 888       888 888         "Y88888P"  888   T88b     888      "Y8888P"
"""
import numpy as np
import time
from glob import iglob
import os, json
import SimpleITK as sitk

# from slicer.util import pip_install, pip_uninstall

from pkg_resources import working_set
import dicom2nifti
import itk
# if "itk-elastix" in [f"{i.key}" for i in working_set]:
#     import itk
# else:
#     pip_install("itk-elastix -q")
#     import itk

# try:
#     import dicom2nifti
# except ImportError:
#     pip_install("dicom2nifti -q")
#     import dicom2nifti

"""
8888888888 8888888 888      8888888888  .d8888b.
888          888   888      888        d88P  Y88b
888          888   888      888        Y88b.
8888888      888   888      8888888     "Y888b.
888          888   888      888            "Y88b.
888          888   888      888              "888
888          888   888      888        Y88b  d88P
888        8888888 88888888 8888888888  "Y8888P"
"""


def GetListNamesSegType(segmentationType):
    dic = {
        "CB": ["cb"],
        "MAND": ["mand", "md"],
        "MAX": ["max", "mx"],
    }
    return dic[segmentationType]


def GetListFiles(folder_path, file_extension):
    """Return a list of files in folder_path finishing by file_extension"""
    file_list = []
    for extension_type in file_extension:
        file_list += search(folder_path, file_extension)[extension_type]
    return file_list


def GetPatients(folder_path, time_point="T1", segmentationType=None, mask_folder=None):
    """Return a dictionary with patient id as key"""
    file_extension = [".nii.gz", ".nii", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz"]
    json_extension = [".json"]
    file_list = GetListFiles(folder_path, file_extension + json_extension)

    patients = {}

    for file in file_list:
        basename = os.path.basename(file)
        patient = (
            basename.split("_Scan")[0]
            .split("_scan")[0]
            .split("_Or")[0]
            .split("_OR")[0]
            .split("_MAND")[0]
            .split("_MD")[0]
            .split("_MAX")[0]
            .split("_MX")[0]
            .split("_CB")[0]
            .split("_lm")[0]
            .split("_T2")[0]
            .split("_T1")[0]
            .split("_Cl")[0]
            .split(".")[0]
        )

        if patient not in patients:
            patients[patient] = {}

        if True in [i in basename for i in file_extension]:
            if True in [i in basename.lower() for i in ["mask", "seg", "pred"]]:
                if segmentationType is None:
                    patients[patient]["seg" + time_point] = file
                # Otherwise, skip for now (handled below)
            else:
                patients[patient]["scan" + time_point] = file

        if True in [i in basename for i in json_extension]:
            if time_point == "T2":
                patients[patient]["lm" + time_point] = file

    # If segmentationType is specified, look for masks in mask_folder or fallback to folder_path
    if segmentationType:
        target_keywords = GetListNamesSegType(segmentationType)
        search_folder = mask_folder if mask_folder else folder_path
        mask_files = GetListFiles(search_folder, file_extension)

        for file in mask_files:
            basename = os.path.basename(file)
            patient = (
                basename.split("_Scan")[0]
                .split("_scan")[0]
                .split("_Or")[0]
                .split("_OR")[0]
                .split("_MAND")[0]
                .split("_MD")[0]
                .split("_MAX")[0]
                .split("_MX")[0]
                .split("_CB")[0]
                .split("_lm")[0]
                .split("_T2")[0]
                .split("_T1")[0]
                .split("_Cl")[0]
                .split(".")[0]
            )
            if True in [kw in basename.lower() for kw in target_keywords]:
                if patient not in patients:
                    patients[patient] = {}
                if "seg" + time_point not in patients[patient]:
                    patients[patient]["seg" + time_point] = file

    return patients



def GetMatrixPatients(folder_path):
    """Return a dictionary with patient id as key and matrix path as data"""
    file_extension = [".tfm"]
    file_list = GetListFiles(folder_path, file_extension)

    patients = {}
    for file in file_list:
        basename = os.path.basename(file)
        patient = basename.split("reg_")[1].split("_Cl")[0]
        if patient not in patients and True in [i in basename for i in file_extension]:
            patients[patient] = {}
            patients[patient]["mat"] = file

    return patients


def GetDictPatients(
    folder_t1_path,
    folder_t2_path,
    segmentationType=None,
    todo_str="",
    matrix_folder=None,
    mask_folder_t1=None,
):
    """Return a dictionary with patients for both time points"""
    patients_t1 = GetPatients(
        folder_t1_path, time_point="T1", segmentationType=segmentationType, mask_folder=mask_folder_t1
    )
    patients_t2 = GetPatients(folder_t2_path, time_point="T2", segmentationType=None)
    patients = MergeDicts(patients_t1, patients_t2)

    if matrix_folder is not None:
        patient_matrix = GetMatrixPatients(matrix_folder)
        patients = MergeDicts(patients, patient_matrix)
    patients = ModifiedDictPatients(patients, todo_str)
    return patients


def MergeDicts(dict1, dict2):
    """Merge t1 and t2 dictionaries for each patient"""
    patients = {}
    for patient in dict1:
        patients[patient] = dict1[patient]
        try:
            patients[patient].update(dict2[patient])
        except KeyError:
            continue
    return patients


def ModifiedDictPatients(patients, todo_str):
    """Modify the dictionary of patients to only keep the ones in the todo_str"""

    if todo_str != "":
        liste_todo = todo_str.split(",")
        todo_patients = {}
        for i in liste_todo:
            patient = list(patients.keys())[int(i) - 1]
            todo_patients[patient] = patients[patient]
        patients = todo_patients

    return patients


def search(path, *args):
    """
    Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

    Example:
    args = ('json',['.nii.gz','.nrrd'])
    return:
        {
            'json' : ['path/a.json', 'path/b.json','path/c.json'],
            '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
            '.nrrd.gz' : ['path/c.nrrd']
        }
    """
    arguments = []
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
        else:
            arguments.append(arg)
    return {
        key: sorted(
            [
                i
                for i in iglob(
                    os.path.normpath("/".join([path, "**", "*"])), recursive=True
                )
                if i.endswith(key)
            ]
        )
        for key in arguments
    }


"""
888             d8888 888b    888 8888888b.  888b     d888        d8888 8888888b.  888    d8P
888            d88888 8888b   888 888  "Y88b 8888b   d8888       d88888 888   Y88b 888   d8P
888           d88P888 88888b  888 888    888 88888b.d88888      d88P888 888    888 888  d8P
888          d88P 888 888Y88b 888 888    888 888Y88888P888     d88P 888 888   d88P 888d88K
888         d88P  888 888 Y88b888 888    888 888 Y888P 888    d88P  888 8888888P"  8888888b
888        d88P   888 888  Y88888 888    888 888  Y8P  888   d88P   888 888 T88b   888  Y88b
888       d8888888888 888   Y8888 888  .d88P 888   "   888  d8888888888 888  T88b  888   Y88b
88888888 d88P     888 888    Y888 8888888P"  888       888 d88P     888 888   T88b 888    Y88b
"""


def applyTransformLandmarks(landmarks, transform):
    """Apply a transform to a set of landmarks."""
    copy = landmarks.copy()
    for lm, pt in landmarks.items():
        copy[lm] = transform.TransformPoint(pt)
    return copy


def GenControlePoint(landmarks):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark, data in landmarks.items():
        id += 1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data[0], data[1], data[2]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "defined",
        }
        lm_lst.append(controle_point)

    return lm_lst


def WriteJson(landmarks, out_path):
    false = False
    true = True
    file = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": false,
                "labelFormat": "%N-%d",
                "controlPoints": GenControlePoint(landmarks),
                "measurements": [],
                "display": {
                    "visibility": false,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 0.0],
                    "color": [0.5, 0.5, 0.5],
                    "selectedColor": [
                        0.26666666666666669,
                        0.6745098039215687,
                        0.39215686274509806,
                    ],
                    "propertiesLabelVisibility": false,
                    "pointLabelsVisibility": true,
                    "textScale": 2.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 2.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": true,
                    "sliceProjection": false,
                    "sliceProjectionUseFiducialColor": true,
                    "sliceProjectionOutlinedBehindSlicePlane": false,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": false,
                    "snapMode": "toVisibleSurface",
                },
            }
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close


def LoadOnlyLandmarks(ldmk_path, ldmk_list=None):
    """
    Load landmarks from json file without using the img as input

    Parameters
    ----------
    ldmk_path : str
        Path to the json file
    gold : bool, optional
        If True, load gold standard landmarks, by default False

    Returns
    -------
    dict
        Dictionary of landmarks

    Raises
    ------
    ValueError
        If the json file is not valid
    """
    with open(ldmk_path) as f:
        data = json.load(f)

    markups = data["markups"][0]["controlPoints"]

    landmarks = {}
    for markup in markups:
        try:
            lm_ph_coord = np.array(
                [markup["position"][0], markup["position"][1], markup["position"][2]]
            )
            # lm_coord = ((lm_ph_coord - origin) / spacing).astype(np.float16)
            lm_coord = lm_ph_coord.astype(np.float64)
            landmarks[markup["label"]] = lm_coord
        except:
            continue
    if ldmk_list is not None:
        return {key: landmarks[key] for key in ldmk_list if key in landmarks.keys()}

    return landmarks


"""
8888888 888b     d888        d8888  .d8888b.  8888888888  .d8888b.
  888   8888b   d8888       d88888 d88P  Y88b 888        d88P  Y88b
  888   88888b.d88888      d88P888 888    888 888        Y88b.
  888   888Y88888P888     d88P 888 888        8888888     "Y888b.
  888   888 Y888P 888    d88P  888 888  88888 888            "Y88b.
  888   888  Y8P  888   d88P   888 888    888 888              "888
  888   888   "   888  d8888888888 Y88b  d88P 888        Y88b  d88P
8888888 888       888 d88P     888  "Y8888P88 8888888888  "Y8888P"
"""


def ResampleImage(image, transform):
    """
    Resample image using SimpleITK

    Parameters
    ----------
    image : SimpleITK.Image
        Image to be resampled
    target : SimpleITK.Image
        Target image
    transform : SimpleITK transform
        Transform to be applied to the image.

    Returns
    -------
    SimpleITK image
        Resampled image.
    """
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # Resample image
    resampled_image = resampler.Execute(image)

    return resampled_image


def applyMask(image, mask, label):
    """Apply a mask to an image."""
    # Cast the image to float32
    # image = sitk.Cast(image, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(mask)
    if label is not None and label in np.unique(array):
        array = np.where(array == label, 1, 0)
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(image)

    return sitk.Mask(image, mask)


"""
8888888b.  8888888888  .d8888b.  8888888  .d8888b.  88888888888
888   Y88b 888        d88P  Y88b   888   d88P  Y88b     888
888    888 888        888    888   888   Y88b.          888
888   d88P 8888888    888          888    "Y888b.       888
8888888P"  888        888  88888   888       "Y88b.     888
888 T88b   888        888    888   888         "888     888
888  T88b  888        Y88b  d88P   888   Y88b  d88P     888
888   T88b 8888888888  "Y8888P88 8888888  "Y8888P"      888
"""


def make_rigid_param_map_stochastic():
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap("rigid")

    # Sampling aléatoire rapide
    pm["ImageSampler"] = ["RandomCoordinate"]
    pm["RandomSeed"] = ["1"]                  # reproductibilité
    pm["NewSamplesEveryIteration"] = ["true"] # on change les points à chaque itération
    pm["NumberOfSpatialSamples"] = ["10000"]  # rapide
    pm["MaximumNumberOfIterations"] = ["2000"]# run court
    pm["NumberOfResolutions"] = ["2"]
    pm["Metric"] = ["AdvancedMattesMutualInformation"]
    pm["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    pm["ErodeMask"] = ["true"]
    pm["WriteResultImage"] = ["false"]

    po.AddParameterMap(pm)
    return po


def make_rigid_param_map_deterministic():
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap("rigid")  # EulerTransform

    # Déterminisme
    pm["NumberOfThreads"] = ["1"]
    pm["UseDirectionCosines"] = ["true"]
    pm["ImageSampler"] = ["Grid"]
    pm["NewSamplesEveryIteration"] = ["false"]

    # Pyramide (coarse → fine)
    pm["NumberOfResolutions"] = ["3"]
    pm["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    pm["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    pm["ImagePyramidSchedule"] = ["8","8", "4","4", "2","2"]

    # Metric & interpolation
    pm["Metric"] = ["AdvancedMattesMutualInformation"]
    pm["NumberOfHistogramBins"] = ["64"]
    pm["NormalizeGradient"] = ["true"]
    pm["Interpolator"] = ["LinearInterpolator"]

    # Optimiseur (stable en rigide avec Grid)
    pm["Optimizer"] = ["ConjugateGradient"]
    pm["MaximumNumberOfIterations"] = ["1500"]
    pm["MaximumStepLength"] = ["2.0"]
    pm["MinimumStepLength"] = ["0.001"]
    pm["ValueTolerance"] = ["1e-6"]
    pm["GradientTolerance"] = ["1e-6"]

    # Initialisation & échelles
    pm["AutomaticTransformInitialization"] = ["true"]
    pm["AutomaticScalesEstimation"] = ["true"]

    # Masque & sorties
    pm["ErodeMask"] = ["true"]
    pm["WriteResultImage"] = ["false"]

    # Compat (ignoré avec Grid, mais ne gêne pas)
    pm["NumberOfSpatialSamples"] = ["30000"]

    po.AddParameterMap(pm)
    return po



def ElastixApprox(fixed_image, moving_image):
    elastix = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix.SetParameterObject(make_rigid_param_map_stochastic())
    elastix.SetLogToConsole(False)
    elastix.UpdateLargestPossibleRegion()
    return elastix.GetTransformParameterObject()


def ElastixReg(fixed_image, moving_image, initial_transform=None):
    elastix = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix.SetParameterObject(make_rigid_param_map_deterministic())
    if initial_transform is not None:
        elastix.SetInitialTransformParameterObject(initial_transform)
    elastix.SetLogToConsole(False)
    elastix.UpdateLargestPossibleRegion()
    return elastix.GetTransformParameterObject()



def MaskedImage(fixed_image_path, fixed_seg_path, temp_folder, SegLabel=None):
    """Mask the fixed image with the fixed segmentation and write it to a file"""
    fixed_image_sitk = sitk.ReadImage(fixed_image_path)
    fixed_seg_sitk = sitk.ReadImage(fixed_seg_path)
    fixed_seg_sitk.SetOrigin(fixed_image_sitk.GetOrigin())
    fixed_image_masked = applyMask(fixed_image_sitk, fixed_seg_sitk, label=SegLabel)

    # Write the masked image
    output_path = os.path.join(temp_folder, "fixed_image_masked.nii.gz")
    sitk.WriteImage(sitk.Cast(fixed_image_masked, sitk.sitkInt16), output_path)

    return output_path


def VoxelBasedRegistration(
    fixed_image_path,
    moving_image_path,
    fixed_seg_path,
    temp_folder,
    approx=False,
    SegLabel=None,
):
    """Perform a voxel-based registration using elastix"""

    # Read images and segmentations
    fixed_image = itk.imread(fixed_image_path, itk.F)
    moving_image = itk.imread(moving_image_path, itk.F)

    # Mask the fixed image (using sitk)
    masked_image_path = MaskedImage(
        fixed_image_path, fixed_seg_path, temp_folder, SegLabel=SegLabel
    )
    fixed_image_masked = itk.imread(masked_image_path, itk.F)

    # Register images
    Transforms = []

    # if approx:
    #     # Approximate registration
    #     TransformObj_Approx = ElastixApprox(fixed_image, moving_image)
    #     transforms_Approx = MatrixRetrieval(TransformObj_Approx)
    #     Transforms.append(transforms_Approx)
    # else:
    #     TransformObj_Approx = None

    # Fine tuning

    TransformObj_Fine = ElastixReg(
        fixed_image_masked, moving_image, initial_transform=None
    )

    transforms_Fine = MatrixRetrieval(TransformObj_Fine)
    Transforms.append(transforms_Fine)

    print("ccccc")

    # Combine transforms
    transform = ComputeFinalMatrix(Transforms)

    # Resample images and segmentations using the final transform
    resample_t2 = sitk.Cast(
        ResampleImage(sitk.ReadImage(moving_image_path), transform), sitk.sitkInt16
    )

    return transform, resample_t2


"""
888     888 88888888888 8888888 888       .d8888b.
888     888     888       888   888      d88P  Y88b
888     888     888       888   888      Y88b.
888     888     888       888   888       "Y888b.
888     888     888       888   888          "Y88b.
888     888     888       888   888            "888
Y88b. .d88P     888       888   888      Y88b  d88P
 "Y88888P"      888     8888888 88888888  "Y8888P"
"""


def translate(shortname):
    """Translate a shortname to a full name for the different structures"""
    dic = {"CB": "Cranial Base", "MAND": "Mandible", "MAX": "Maxilla"}
    return dic[shortname]


def convertdicom2nifti(input_folder, output_folder=None):
    """Convert a folder of dicom files to nifti files using SimpleITK"""

    patients_folders = [
        folder
        for folder in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, folder)) and folder != "NIFTI"
    ]

    if output_folder is None:
        output_folder = os.path.join(input_folder, "NIFTI")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for patient in patients_folders:
        if not os.path.exists(os.path.join(output_folder, patient + ".nii.gz")):
            print("Converting patient: {}...".format(patient))
            current_directory = os.path.join(input_folder, patient)
            try:
                reader = sitk.ImageSeriesReader()
                sitk.ProcessObject_SetGlobalWarningDisplay(False)
                dicom_names = reader.GetGDCMSeriesFileNames(current_directory)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                sitk.ProcessObject_SetGlobalWarningDisplay(True)
                sitk.WriteImage(
                    image,
                    os.path.join(
                        output_folder, os.path.basename(current_directory) + ".nii.gz"
                    ),
                )
            except RuntimeError:
                dicom2nifti.convert_directory(current_directory, output_folder)
                nifti_file = search(output_folder, "nii.gz")["nii.gz"][0]
                os.rename(nifti_file, os.path.join(output_folder, patient + ".nii.gz"))


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
