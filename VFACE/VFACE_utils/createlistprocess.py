import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .Progress import DisplayASOCBCT,DisplayAMASSS,DisplayAREGCBCT,DisplayALICBCT
from glob import iglob
import slicer
from .functionaq3dc import AQ3DCLogic, AQ3DCWidget, patientIdFromFileName
import qt
import re
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import traceback

import logging

# ===== Logging Configuration =====
logger = logging.getLogger("VFACE_createlistprocess")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("Warning: psutil not available - memory monitoring disabled")
import gc

def check_memory_usage(threshold_percent=80):
    if psutil is None:
        return False
    
    try:
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent > threshold_percent:
            logger.warning(f"Warning: High memory usage detected: {usage_percent:.1f}%")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking memory: {e}")
        return False

def force_memory_cleanup():
    logger.info("Forcing memory cleanup...")
    
    collected_total = 0
    for i in range(3):
        collected = gc.collect()
        collected_total += collected
        if collected > 0:
            logger.debug(f"Garbage collection round {i+1}: freed {collected} objects")
    
    if collected_total > 0:
        logger.debug(f"Total objects freed: {collected_total}")
    
    if 'slicer' in globals():
        slicer.app.processEvents()
    
    if psutil:
        memory = psutil.virtual_memory()
        logger.debug(f"Memory usage after cleanup: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB used / {memory.total / 1024**3:.1f}GB total)")

def get_memory_info():
    if not psutil:
        return "Memory monitoring not available (psutil not installed)"
    
    try:
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'used_gb': memory.used / 1024**3,
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3
        }
    except Exception as e:
        return f"Error getting memory info: {e}"

class LocalAQ3DCLogic(AQ3DCLogic):
    
    def __init__(self):
        
        modules_to_remove = [name for name in sys.modules.keys() if name.startswith('Classes.')]
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        local_classes_path = os.path.join(current_dir, "Classes")
        if local_classes_path not in sys.path:
            sys.path.insert(0, local_classes_path)
        
        super().__init__()
    
    def computeMeasurement(self, list_measure: list, dict_patient: dict):
        for measure in list_measure:
            if not hasattr(measure, 'keep_sign') or measure.keep_sign is None:
                import qt
                measure.keep_sign = qt.QCheckBox()
                measure.keep_sign.setChecked(True)
                logger.debug(f"Initialized keep_sign for {measure.__class__.__name__} from {measure.__class__.__module__}")
        
        return super().computeMeasurement(list_measure, dict_patient)

def CreateListProcess(**kwargs):

    list_process = []

    # Load all slicer modules at the beginning
    ResampleProcess = slicer.modules.mri2cbct_resample_cbct_mri
    PreOrientProcess = slicer.modules.pre_aso_cbct
    ALIProcess = slicer.modules.ali_cbct
    SEMI_ASOProcess = slicer.modules.semi_aso_cbct
    AMASSSProcess = slicer.modules.amasss_cli
    AutomatrixProcess = slicer.modules.automatrix_cli
    AREGProcess = slicer.modules.areg_cbct
    AsymProcess = slicer.modules.vface_cli

    # NumberScan is just len(GetPatients(...)), so scan the input folder once.
    patients = GetPatients(kwargs["InputFolder"], time_point="T1")
    nb_scan = len(patients)

    if kwargs["bool_quantification"]:
        cb_measurements_path, mand_measurements_path, max_measurements_path,feature_path = SplitMeasurements(kwargs["measurements_folder"],kwargs["mode2"])
        
        if not cb_measurements_path or not max_measurements_path or not mand_measurements_path:
            logger.warning("There is an issue, it miss measurements lists in the list of measurements folder")
        elif not feature_path and kwargs["mode2"] != "Longitudinal studies":
                logger.warning("There is an issue, it miss feature list in the ML folder")
        
        # The MAND and CB measurement lists share landmarks, and concatenating them
        # made ALI spawn a second agent for each shared one and search it twice for
        # the same position. dict.fromkeys keeps the original order.
        list_landmark = list(dict.fromkeys(
            create_list_landmark(mand_measurements_path)
            + create_list_landmark(cb_measurements_path)
        ))
        
        list_landmark_max = create_list_landmark(max_measurements_path)

        list_measure_cb = create_list_measure(cb_measurements_path)
        list_measure_max = create_list_measure(max_measurements_path)
        list_measure_mand = create_list_measure(mand_measurements_path)

    if kwargs["mode"] != "File already Registered":
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        documents = qt.QStandardPaths.writableLocation(documentsLocation)
        tempAMASSS_folder = os.path.join(documents, slicer.app.applicationName + "_temp_AMASSS")

        if kwargs["mode2"] != "Longitudinal studies":
            t2scan_folder_path = os.path.join(kwargs["OutputFolder"],"T2_Scan")
            os.makedirs(t2scan_folder_path, exist_ok=True)

            t2scan_max_folder_path = os.path.join(t2scan_folder_path,"MAX")
            os.makedirs(t2scan_max_folder_path, exist_ok=True)

            t2scan_cb_folder_path = os.path.join(t2scan_folder_path,"CB")
            os.makedirs(t2scan_cb_folder_path, exist_ok=True)
        else:
            t2_centered_folder_path = os.path.join(kwargs["OutputFolder"],"T2 Centered")
            os.makedirs(t2_centered_folder_path, exist_ok=True)

    orientation_folder_path = os.path.join(kwargs["OutputFolder"],"Oriented T1 Scans")
    os.makedirs(orientation_folder_path, exist_ok=True)

    orientation_cb_folder_path = os.path.join(orientation_folder_path,"CB")
    os.makedirs(orientation_cb_folder_path, exist_ok=True)

    orientation_max_folder_path = os.path.join(orientation_folder_path,"MAX")
    os.makedirs(orientation_max_folder_path, exist_ok=True)

    if kwargs["mode"] != "Full pipeline":
        oriented_files = SplitOriented(kwargs["InputFolder"])

        if oriented_files:
            for file_info in oriented_files:
                destination_path = os.path.join(orientation_folder_path, file_info['destination_folder'])
                os.makedirs(destination_path, exist_ok=True)
                shutil.copy2(file_info['path'], os.path.join(destination_path, file_info['path'].split("/")[-1]))
                logger.info(f"Copied {file_info['type']} file: {file_info['path'].split('/')[-1]} to {file_info['destination_folder']}")
        else:
            logger.error("Issue, it seems to miss some oriented cbct T1 folder")
            return

    else:

        resample_folder_path = os.path.join(kwargs["OutputFolder"],"T1 Resample")
        os.makedirs(resample_folder_path, exist_ok=True)

        preaso_folder_path = os.path.join(kwargs["OutputFolder"],"Centered T1 Scans")
        os.makedirs(preaso_folder_path, exist_ok=True)

        preaso_CB_folder_path = os.path.join(preaso_folder_path,"CB")
        os.makedirs(preaso_CB_folder_path, exist_ok=True)

        preaso_MAX_folder_path = os.path.join(preaso_folder_path,"MAX")
        os.makedirs(preaso_MAX_folder_path, exist_ok=True)

        parameter_resample = {
            "input_folder_MRI": "None",
            "input_folder_T2_MRI": "None",
            "input_folder_CBCT": kwargs["InputFolder"],
            "input_folder_T2_CBCT": "None",
            "input_folder_Seg": "None",
            "input_folder_T2_Seg": "None",
            "output_folder": resample_folder_path,
            "resample_size": "None",
            "spacing": [0.3,0.3,0.3],
            "center": "True"
        }
        list_process.append(
            {
                "Process": ResampleProcess,
                "Parameter": parameter_resample,
                "Module": "Resample T1",
                "Display": DisplayASOCBCT(
                    nb_scan
                )
            }
        )

        parameter_pre_aso_max = {
            "input": os.path.join(resample_folder_path, "CBCT"),
            "output_folder": preaso_MAX_folder_path,
            "model_folder": False,
            "SmallFOV": False,
            "temp_folder": _unique_temp_dir("work"),
            "DCMInput": False,
        }
        list_process.append(
            {
                "Process": PreOrientProcess,
                "Parameter": parameter_pre_aso_max,
                "Module": "Centering T1",
                "Display": DisplayASOCBCT(
                    nb_scan
                )
           }
        )
        
        parameter_ali_aso_max = {
            "input": preaso_MAX_folder_path,
            "dir_models": kwargs["model_folder_ali"],
            "lm_type": "'ANS','IF','PNS','UL6O','UR1O','UR6O'",
            "output_dir": preaso_MAX_folder_path,
            "temp_fold": _unique_temp_dir("work"),
            "DCMInput": False,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }

        list_process.append(
            {
                "Process": ALIProcess,
                "Parameter": parameter_ali_aso_max,
                "Module": "Orient T1 (MAX)",
                "Display": DisplayALICBCT(6,
                    nb_scan
                ),
                "ReviewTitle": "Maxilla orientation landmarks",
                "ReviewHint": (
                    "These points decide how the scan is oriented, and every step after it inherits that orientation. Drag any that sits off its anatomy. Your changes are saved when you click Continue - you do not need to save in Slicer."
                ),
                "ReviewFolder": preaso_MAX_folder_path,
                "ReviewVolumeFolder": preaso_MAX_folder_path,
                "ReviewEditable": True,
                "ReviewId": "t1_landmarks_orientation_max",
                "pause_for_visualization": True,
            }
        )
        
        parameter_semi_aso_max = {
            "input": preaso_MAX_folder_path,
            "gold_folder": os.path.join(kwargs["gold_folder"],"Occlusal and Midsagittal Plane"),
            "output_folder": orientation_max_folder_path,
            "add_inname": "MAX_Or",
            "list_landmark": 'ANS IF PNS UL6O UR1O UR6O',
        }

        list_process.append(
            {
                "Process": SEMI_ASOProcess,
                "Parameter": parameter_semi_aso_max,
                "Module": "Orient T1 (MAX)",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
                "ReviewTitle": "Maxilla orientation of the original scan",
                "ReviewHint": (
                    "Check the scan is oriented on the occlusal and mid-sagittal "
                    "planes. Nothing to edit here - look at the result, then click "
                    "Continue."
                ),
                "ReviewFolder": orientation_max_folder_path,
                "ReviewId": "t1_oriented_max",
                "pause_for_visualization": True,
            }
        )

        parameter_pre_aso_cb = {
            "input": os.path.join(resample_folder_path, "CBCT"),
            "output_folder": preaso_CB_folder_path,
            "model_folder": False,
            "SmallFOV": False,
            "temp_folder": _unique_temp_dir("work"),
            "DCMInput": False,
        }

        list_process.append(
            {
                "Process": PreOrientProcess,
                "Parameter": parameter_pre_aso_cb,
                "Module": "Centering T1",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            }
        )

        parameter_ali_aso_cb = {
            "input": preaso_CB_folder_path,
            "dir_models": kwargs["model_folder_ali"],
            "lm_type": "'Ba', 'LPo', 'N', 'RPo', 'S', 'LOr', 'ROr'",
            "output_dir": preaso_CB_folder_path,
            "temp_fold": _unique_temp_dir("work"),
            "DCMInput": False,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }

        list_process.append(
            {
                "Process": ALIProcess,
                "Parameter": parameter_ali_aso_cb,
                "Module": "Orient T1 (CB)",
                "Display": DisplayALICBCT(6,
                    nb_scan
                ),
                "ReviewTitle": "Cranial base orientation landmarks",
                "ReviewHint": (
                    "These points decide how the scan is oriented, and every step after it inherits that orientation. Drag any that sits off its anatomy. Your changes are saved when you click Continue - you do not need to save in Slicer."
                ),
                "ReviewFolder": preaso_CB_folder_path,
                "ReviewVolumeFolder": preaso_CB_folder_path,
                "ReviewEditable": True,
                "ReviewId": "t1_landmarks_orientation_cb",
                "pause_for_visualization": True,
            }
        )

        parameter_semi_aso_CBMand = {
            "input": preaso_CB_folder_path,
            "gold_folder": os.path.join(kwargs["gold_folder"],"Frankfurt Horizontal and Midsagittal Plane"),
            "output_folder": orientation_cb_folder_path,
            "add_inname": "CB_Or",
            "list_landmark": 'Ba LPo N RPo S LOr ROr',
        }

        list_process.append(
            {
                "Process": SEMI_ASOProcess,
                "Parameter": parameter_semi_aso_CBMand,
                "Module": "Orient T1 (CB)",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
                "ReviewTitle": "Cranial base orientation of the original scan",
                "ReviewHint": (
                    "Check the scan is oriented on the Frankfort horizontal and "
                    "mid-sagittal planes. Nothing to edit here - look at the "
                    "result, then click Continue."
                ),
                "ReviewFolder": orientation_cb_folder_path,
                "ReviewId": "t1_oriented_cb",
                "pause_for_visualization": True,
            }
        )
    
    if kwargs["mode"] != "File already Registered":

        if kwargs["mode2"] == "Longitudinal studies":
            t2_cb_folder = t2_centered_folder_path
            t2_max_folder = t2_centered_folder_path
        else:
            t2_cb_folder = t2scan_cb_folder_path
            t2_max_folder = t2scan_max_folder_path

        if kwargs["mode2"] == "Longitudinal studies":

            t2_resample_folder_path = os.path.join(kwargs["OutputFolder"],"T2 Resample")
            os.makedirs(t2_resample_folder_path, exist_ok=True)

            parameter_resample = {
                "input_folder_MRI": "None",
                "input_folder_T2_MRI": "None",
                "input_folder_CBCT": os.path.join(kwargs["t2_folder"]),
                "input_folder_T2_CBCT": "None",
                "input_folder_Seg": "None",
                "input_folder_T2_Seg": "None",
                "output_folder": t2_resample_folder_path,
                "resample_size": "None",
                "spacing": [0.3,0.3,0.3],
                "center": "True"
            }
            list_process.append(
                {
                    "Process": ResampleProcess,
                    "Parameter": parameter_resample,
                    "Module": "Resample T2",
                    "Display": DisplayASOCBCT(
                        nb_scan
                    )
                }
            )

            parameter_pre_aso = {
                "input": os.path.join(t2_resample_folder_path,"CBCT"),
                "output_folder": t2_centered_folder_path,
                "model_folder": False,
                "SmallFOV": False,
                "temp_folder": _unique_temp_dir("work"),
                "DCMInput": False,
            }

            list_process.append(
                {
                    "Process": PreOrientProcess,
                    "Parameter": parameter_pre_aso,
                    "Module": "Centering T2(CB)",
                    "Display": DisplayASOCBCT(
                        nb_scan
                    ),
                }
            )

        mask_folder_path = os.path.join(kwargs["OutputFolder"],"T1 Masks")
        os.makedirs(mask_folder_path, exist_ok=True)

        full_reg_struct = ["Cranial Base","Mandible"]
        reg_struct = TranslateModels(full_reg_struct, True)

        parameter_amasss_mask_t1 = {
            "inputVolume": orientation_cb_folder_path,
            "modelDirectory": os.path.join(kwargs["model_folder"], "AMASSS_Models"),
            "skullStructure": reg_struct,
            "merge": "SEPARATE",
            "genVtk": False,
            "save_in_folder": False,
            "output_folder": mask_folder_path,
            "vtk_smooth": 5,
            "prediction_ID": "seg",
            "temp_fold": tempAMASSS_folder,
            "SegmentInput": False,
            "DCMInput": False,
        }
        list_process.append(
            {
                "Process": AMASSSProcess,
                "Parameter": parameter_amasss_mask_t1,
                "Module": "Masks Generation for T1 (CB,MAND)",
                "Display": DisplayAMASSS(
                    nb_scan, len(full_reg_struct)
                ),
            },
        )

        parameter_amasss_mask = {
            "inputVolume": orientation_max_folder_path,
            "modelDirectory": os.path.join(kwargs["model_folder"], "AMASSS_Models"),
            "skullStructure": TranslateModels(["Maxilla"], True),       
            "merge": "SEPARATE",
            "genVtk": False,
            "save_in_folder": False,
            "output_folder": mask_folder_path,
            "vtk_smooth": 5,
            "prediction_ID": "seg",
            "temp_fold": tempAMASSS_folder,
            "SegmentInput": False,
            "DCMInput": False,
        }
        list_process.append(
            {
                "Process": AMASSSProcess,
                "Parameter": parameter_amasss_mask,
                "Module": "Masks Generation for T1 (MAX)",
                "Display": DisplayAMASSS(
                    nb_scan, len(full_reg_struct)
                ),
                "ReviewTitle": "Bone segmentation of the original scan",
                "ReviewHint": (
                    "Check the cranial base, mandible and maxilla masks follow the "
                    "bone - they guide the registration that comes next. Nothing to "
                    "edit here - look at the result, then click Continue."
                ),
                "ReviewFolder": mask_folder_path,
                "ReviewId": "t1_masks",
                "pause_for_visualization": True,
            },
        )
        if kwargs["mode2"] == "Asymmetry Assesment":
            if kwargs["reg_type"] == "CMFReg":
                t2mask_folder_path = os.path.join(kwargs["OutputFolder"],"T2_Masks")
                os.makedirs(t2mask_folder_path, exist_ok=True)

                parameter_automatrix_mask = {
                    "input_patient": mask_folder_path,
                    "input_matrix": kwargs["mirror_matrix"],
                    "reference_file": "None",
                    "suffix": "_mir",
                    "matrix_name": False,
                    "fromAreg": False,
                    "output_folder": t2mask_folder_path,
                    "log_path": _unique_temp_dir("log"),
                    "is_seg": True
                }
                list_process.append(
                    {
                        "Process": AutomatrixProcess,
                        "Parameter": parameter_automatrix_mask,
                        "Module": "Mirroring Masks",
                        "Display": DisplayAMASSS(
                            nb_scan, len(full_reg_struct)
                        ),
                        "ReviewTitle": "Mirrored bone segmentation",
                        "ReviewHint": (
                            "Check the mirrored masks. Nothing to edit here - look at the "
                            "result, then click Continue."
                        ),
                        "ReviewFolder": t2mask_folder_path,
                        "ReviewId": "mirror_masks",
                        "pause_for_visualization": True,
                    },
                )

            parameter_automatrix_scan = {
                "input_patient": orientation_cb_folder_path,
                "input_matrix": kwargs["mirror_matrix"],
                "reference_file": "None",
                "suffix": "_mir",
                "matrix_name": False,
                "fromAreg": False,
                "output_folder": t2_cb_folder,
                "log_path": _unique_temp_dir("log"),
                "is_seg": False
            }

            list_process.append(
                {
                    "Process": AutomatrixProcess,
                    "Parameter": parameter_automatrix_scan,
                    "Module": "Mirroring CB Oriented Scan",
                    "Display": DisplayAMASSS(
                        nb_scan, len(full_reg_struct)
                    ),
                },
            )

            parameter_automatrix_scan_max = {
                "input_patient": orientation_max_folder_path,
                "input_matrix": kwargs["mirror_matrix"],
                "reference_file": "None",
                "suffix": "_mir",
                "matrix_name": False,
                "fromAreg": False,
                "output_folder": t2_max_folder,
                "log_path": _unique_temp_dir("log"),
                "is_seg": False
            }

            list_process.append(
                {
                    "Process": AutomatrixProcess,
                    "Parameter": parameter_automatrix_scan_max,
                    "Module": "Mirroring MAX Oriented Scan",
                    "Display": DisplayAMASSS(
                        nb_scan, len(full_reg_struct)
                    ),
                    "ReviewTitle": "Mirrored scan, the side being compared against",
                    "ReviewHint": (
                        "Check the mirror of the patient's own scan. Nothing to edit "
                        "here - look at the result, then click Continue."
                    ),
                    # Both mirrored scans, not just the maxilla this step
                    # happened to write last. In a longitudinal run the two
                    # already share one folder.
                    "ReviewFolder": (
                        t2scan_folder_path
                        if kwargs["mode2"] != "Longitudinal studies"
                        else t2_max_folder
                    ),
                    "ReviewId": "mirror_scans",
                    "pause_for_visualization": True,
                },
            )

        registeredscan_folder_path = os.path.join(kwargs["OutputFolder"],"Registered Scan")
        os.makedirs(registeredscan_folder_path, exist_ok=True)

        parameter_areg_cbct = {
            "t1_folder": orientation_cb_folder_path,
            "t2_folder": t2_cb_folder,
            "reg_type": "CB",
            "output_folder": registeredscan_folder_path,
            "add_name": "_Reg",
            "DCMInput": False,
            "SegmentationLabel": "0",
            "temp_folder": _unique_temp_dir("work"),
            "ApproxReg": False,
            "mask_folder_t1": mask_folder_path,
        }

        list_process.append(
            {
                "Process": AREGProcess,
                "Parameter": parameter_areg_cbct,
                "Module": "AREG - Registering Scan (CB)",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
                "ReviewTitle": "Mirrored scan registered on the cranial base",
                "ReviewHint": (
                    "Check how closely the mirrored scan (shown half-transparent "
                    "over the original) matches it. If it is off, drag it into "
                    "place - the correction is applied to the landmarks too, "
                    "automatically, when you click Continue."
                ),
                "ReviewFolder": os.path.join(registeredscan_folder_path, "Cranial Base"),
                "ReviewVolumeFolder": orientation_cb_folder_path,
                "ReviewAdjustable": True,
                "ReviewId": "registration_cb",
                "pause_for_visualization": True,
            },
        )

        parameter_areg_cbct_2 = {
            "t1_folder": orientation_max_folder_path,
            "t2_folder": t2_max_folder,
            "reg_type": "MAX",
            "output_folder": registeredscan_folder_path,
            "add_name": "_Reg",
            "DCMInput": False,
            "SegmentationLabel": "0",
            "temp_folder": _unique_temp_dir("work"),
            "ApproxReg": False,
            "mask_folder_t1": mask_folder_path,
        }

        list_process.append(
            {
                "Process": AREGProcess,
                "Parameter": parameter_areg_cbct_2,
                "Module": "AREG - Registering Scan (MAX)",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
                "ReviewTitle": "Mirrored scan registered on the maxilla",
                "ReviewHint": (
                    "Check how closely the mirrored scan (shown half-transparent "
                    "over the original) matches it. If it is off, drag it into "
                    "place - the correction is applied to the landmarks too, "
                    "automatically, when you click Continue."
                ),
                "ReviewFolder": os.path.join(registeredscan_folder_path, "Maxilla"),
                "ReviewVolumeFolder": orientation_max_folder_path,
                "ReviewAdjustable": True,
                "ReviewId": "registration_max",
                "pause_for_visualization": True,
            },
        )

        parameter_areg_cbct_3 = {
            "t1_folder": orientation_cb_folder_path,
            "t2_folder": t2_cb_folder,
            "reg_type": "MAND",
            "output_folder": registeredscan_folder_path,
            "add_name": "_Reg",
            "DCMInput": False,
            "SegmentationLabel": "0",
            "temp_folder": _unique_temp_dir("work"),
            "ApproxReg": False,
            "mask_folder_t1": mask_folder_path,
        }

        list_process.append(
            {
                "Process": AREGProcess,
                "Parameter": parameter_areg_cbct_3,
                "Module": "AREG - Registering Scan (MAND)",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
                "ReviewTitle": "Mirrored scan registered on the mandible",
                "ReviewHint": (
                    "Check how closely the mirrored scan (shown half-transparent "
                    "over the original) matches it. If it is off, drag it into "
                    "place - the correction is applied to the landmarks too, "
                    "automatically, when you click Continue."
                ),
                "ReviewFolder": os.path.join(registeredscan_folder_path, "Mandible"),
                "ReviewVolumeFolder": orientation_cb_folder_path,
                "ReviewAdjustable": True,
                "ReviewId": "registration_mand",
                "pause_for_visualization": True,
            },
        )
    else:
        registeredscan_folder_path = os.path.join(kwargs["OutputFolder"],"Registered Scan")
        os.makedirs(registeredscan_folder_path, exist_ok=True)

        if kwargs["bool_visualization"] and not kwargs["bool_quantification"]:
            t2_files = SplitT2(kwargs["t2_folder"], transform=False)
        else:
            t2_files = SplitT2(kwargs["t2_folder"], transform=True)

        if t2_files:
            for file_info in t2_files:
                destination_path = os.path.join(registeredscan_folder_path, file_info['destination_folder'])
                os.makedirs(destination_path, exist_ok=True)
                shutil.copy2(file_info['path'], os.path.join(destination_path, file_info['path'].split("/")[-1]))
                logger.info(f"Copied {file_info['type']} {file_info['file_type']} file: {file_info['path'].split('/')[-1]} to {file_info['destination_folder']}")
        else:
            logger.error("Issue, it seems to miss some cbct or transform files in the T2 folder")
            return
        
    # Quantification runs before visualization: nothing here depends on the
    # surfaces or the heatmaps, and the landmark review is the only step the
    # user can act on. Behind the visualization block it meant waiting through
    # five segmentations and three distance runs before being able to correct
    # a single point.
    if kwargs["bool_quantification"]:

        landmarks_folder_path = os.path.join(kwargs["OutputFolder"],"T1 Landmarks")
        os.makedirs(landmarks_folder_path, exist_ok=True)

        landmarks_cb_folder_path = os.path.join(landmarks_folder_path,"CB")
        os.makedirs(landmarks_cb_folder_path, exist_ok=True)

        landmarks_max_folder_path = os.path.join(landmarks_folder_path,"MAX")
        os.makedirs(landmarks_max_folder_path, exist_ok=True)

        # ALI's agent may sample inside the border ALI adds, but it refuses to
        # step outside the unpadded image, so a landmark at the edge is out of
        # reach. Hand it a roomier copy: the scans on disk are untouched and
        # every later step still reads the originals.
        # tempfile.mkdtemp, not slicer.util.tempDirectory: the latter names its
        # folder to the millisecond, and the calls that build this list run
        # microseconds apart, so it handed the padded scans and ALI's own
        # temp_fold the same directory. ALI then found the leftovers of an
        # earlier step there - elastix's fixed_image_masked - and landmarked
        # that instead of the patients.
        padded_scans_path = _unique_temp_dir("padded")

        list_process.append(
            {
                "Process": pad_scans_for_landmarks,
                "Parameter": {
                    "input_folder": orientation_cb_folder_path,
                    "output_folder": padded_scans_path,
                    "margin_mm": 30,
                },
                "Module": "Making room for the landmark search",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_ali = {
                "input": padded_scans_path,
                "dir_models": kwargs["model_folder_ali"],
                "lm_type": ",".join([f"'{e}'" for e in list_landmark]),
                "output_dir": landmarks_cb_folder_path,
                "temp_fold": _unique_temp_dir("ali"),
                "DCMInput": False,
                "spacing": "[1,0.3]",
                "speed_per_scale": "[1,1]",
                "agent_FOV": "[64,64,64]",
                "spawn_radius": "10"
                }

        list_process.append(
            {
                "Process": ALIProcess,
                "Parameter": parameter_ali,
                "Module": "ALI - Identifying T1 Landmarks (CB)",
                "Display": DisplayALICBCT(30,
                    nb_scan
                ),
                "ReviewTitle": "All T1 landmarks",
                "ReviewHint": (
                    "Every landmark the measurements use is here, maxilla "
                    "included - the maxilla set is derived from these, so this is "
                    "the only review. Drag any misplaced point onto the correct "
                    "anatomy. Your changes are saved automatically when you click "
                    "Continue - you do not need to save in Slicer."
                ),
                "ReviewFolder": landmarks_cb_folder_path,
                "ReviewVolumeFolder": orientation_cb_folder_path,
                "ReviewEditable": True,
                "ReviewId": "t1_landmarks",
                "pause_for_visualization": True,
            },
        )

        # The maxilla landmarks are the same anatomical points the cranial base
        # run already found, wanted in the other oriented frame. Both frames come
        # from the same centred scan, so a rigid transform places them exactly,
        # in seconds instead of minutes - and it keeps one point from landing in
        # two slightly different places depending on which run found it. It also
        # means the review above is the only one: a correction there flows here.
        parameter_derive_max = {
            "source_folder": landmarks_cb_folder_path,
            "source_tfm_folder": orientation_cb_folder_path,
            "target_tfm_folder": orientation_max_folder_path,
            "target_scan_folder": orientation_max_folder_path,
            "keep_landmarks": list_landmark_max,
            "output_folder": landmarks_max_folder_path,
        }

        list_process.append(
            {
                "Process": derive_landmarks,
                "Parameter": parameter_derive_max,
                "Module": "Deriving T1 Landmarks (MAX)",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )
        if kwargs["mode2"] == "Asymmetry Assesment":
            mirrored_landmarks_folder_path = os.path.join(kwargs["OutputFolder"],"Mirrored Landmarks")
            os.makedirs(mirrored_landmarks_folder_path, exist_ok=True)

            mirrored_landmarks_cb_folder_path = os.path.join(mirrored_landmarks_folder_path,"CB")
            os.makedirs(mirrored_landmarks_cb_folder_path, exist_ok=True)

            mirrored_landmarks_max_folder_path = os.path.join(mirrored_landmarks_folder_path,"MAX")
            os.makedirs(mirrored_landmarks_max_folder_path, exist_ok=True)

            parameter_automatrix_ldm = {        
                "input_patient": landmarks_cb_folder_path,
                "input_matrix": kwargs["mirror_matrix"],
                "reference_file": "None",
                "suffix": "_mir",
                "matrix_name": False,
                "fromAreg": False,
                "output_folder": mirrored_landmarks_cb_folder_path,
                "log_path": _unique_temp_dir("log"),
                "is_seg": False
                }

            list_process.append(
                {
                    "Process": AutomatrixProcess,
                    "Parameter": parameter_automatrix_ldm,
                    "Module": "Mirorring T1 Landmarks (CB)",
                    "Display": DisplayAREGCBCT(
                        nb_scan
                    ),
                },
            )

            parameter_automatrix_ldm_max = {        
                "input_patient": landmarks_max_folder_path,
                "input_matrix": kwargs["mirror_matrix"],
                "reference_file": "None",
                "suffix": "_mir",
                "matrix_name": False,
                "fromAreg": False,
                "output_folder": mirrored_landmarks_max_folder_path,
                "log_path": _unique_temp_dir("log"),
                "is_seg": False
                }


            list_process.append(
                {
                    "Process": AutomatrixProcess,
                    "Parameter": parameter_automatrix_ldm_max,
                    "Module": "Mirorring T1 Landmarks (MAX)",
                    "Display": DisplayAREGCBCT(
                        nb_scan
                    ),
                },
            )

            mirrored_registered_landmarks_folder_path = os.path.join(kwargs["OutputFolder"],"Mirrored & Registered Landmarks")
            os.makedirs(mirrored_registered_landmarks_folder_path, exist_ok=True)

            mirrored_registered_cb_landmarks_folder_path = os.path.join(mirrored_registered_landmarks_folder_path,"CB")
            os.makedirs(mirrored_registered_cb_landmarks_folder_path, exist_ok=True)

            mirrored_registered_mand_landmarks_folder_path = os.path.join(mirrored_registered_landmarks_folder_path,"MAND")
            os.makedirs(mirrored_registered_mand_landmarks_folder_path, exist_ok=True)

            mirrored_registered_max_landmarks_folder_path = os.path.join(mirrored_registered_landmarks_folder_path,"MAX")
            os.makedirs(mirrored_registered_max_landmarks_folder_path, exist_ok=True)

            # One run per structure, not one per patient: AutoMatrix pairs each
            # matrix with its patient by name when input_matrix is a folder. Passing
            # a single .tfm made it apply that patient's matrix to every landmark
            # file in the folder, and the constant suffix made each run overwrite
            # the previous one, so only the last patient's matrix survived.
            parameter_automatrix_register_ldm_cb = {
                "input_patient": mirrored_landmarks_cb_folder_path,
                "input_matrix": os.path.join(registeredscan_folder_path,"Cranial Base"),
                "reference_file": "None",
                "suffix": "_CB_reg",
                "matrix_name": False,
                "fromAreg": False,
                "output_folder": mirrored_registered_cb_landmarks_folder_path,
                "log_path": _unique_temp_dir("log"),
                "is_seg": False
                }

            list_process.append(
                {
                    "Process": AutomatrixProcess,
                    "Parameter": parameter_automatrix_register_ldm_cb,
                    "Module": "Apply matrixes T1 to landmarks (CB)",
                    "Display": DisplayAREGCBCT(
                        nb_scan
                    ),
                },
            )

            parameter_automatrix_register_ldm_mand = {
                "input_patient": mirrored_landmarks_cb_folder_path,
                "input_matrix": os.path.join(registeredscan_folder_path,"Mandible"),
                "reference_file": "None",
                "suffix": "_MAND_reg",
                "matrix_name": False,
                "fromAreg": False,
                "output_folder": mirrored_registered_mand_landmarks_folder_path,
                "log_path": _unique_temp_dir("log"),
                "is_seg": False
                }

            list_process.append(
                {
                    "Process": AutomatrixProcess,
                    "Parameter": parameter_automatrix_register_ldm_mand,
                    "Module": "Apply matrixes T1 to landmarks (MAND)",
                    "Display": DisplayAREGCBCT(
                        nb_scan
                    ),
                },
            )

            parameter_automatrix_register_ldm_MAX = {
                "input_patient": mirrored_landmarks_max_folder_path,
                "input_matrix": os.path.join(registeredscan_folder_path,"Maxilla"),
                "reference_file": "None",
                "suffix": "_MAX_reg",
                "matrix_name": False,
                "fromAreg": False,
                "output_folder": mirrored_registered_max_landmarks_folder_path,
                "log_path": _unique_temp_dir("log"),
                "is_seg": False
                }

            list_process.append(
                {
                    "Process": AutomatrixProcess,
                    "Parameter": parameter_automatrix_register_ldm_MAX,
                    "Module": "Apply matrixes to T1 landmarks (MAX)",
                    "Display": DisplayAREGCBCT(
                        nb_scan
                    ),
                },
            )
        else:
            t2_landmarks_folder_path = os.path.join(kwargs["OutputFolder"],"T2 Landmarks")
            os.makedirs(t2_landmarks_folder_path, exist_ok=True)

            t2_landmarks_cb_folder_path = os.path.join(t2_landmarks_folder_path,"CB")
            os.makedirs(t2_landmarks_cb_folder_path, exist_ok=True)

            t2_landmarks_max_folder_path = os.path.join(t2_landmarks_folder_path,"MAX")
            os.makedirs(t2_landmarks_max_folder_path, exist_ok=True)

            t2_landmarks_mand_folder_path = os.path.join(t2_landmarks_folder_path,"MAND")
            os.makedirs(t2_landmarks_mand_folder_path, exist_ok=True)

            parameter_ali = {
                    "input": os.path.join(registeredscan_folder_path,"Cranial Base"),
                    "dir_models": kwargs["model_folder_ali"],
                    "lm_type": ",".join([f"'{e}'" for e in list_landmark]),
                    "output_dir": t2_landmarks_cb_folder_path,
                    "temp_fold": _unique_temp_dir("ali"),
                    "DCMInput": False,
                    "spacing": "[1,0.3]",
                    "speed_per_scale": "[1,1]",
                    "agent_FOV": "[64,64,64]",
                    "spawn_radius": "10"
                    }

            list_process.append(
                {
                    "Process": ALIProcess,
                    "Parameter": parameter_ali,
                    "Module": "ALI - Identifying T2 Landmarks (CB)",
                    "Display": DisplayALICBCT(30,
                        nb_scan
                    ),
                },
            )

            parameter_ali_max = {
                    "input": os.path.join(registeredscan_folder_path,"Maxilla"),
                    "dir_models": kwargs["model_folder_ali"],
                    "lm_type": ",".join([f"'{e}'" for e in list_landmark_max]),
                    "output_dir": t2_landmarks_max_folder_path,
                    "temp_fold": _unique_temp_dir("ali"),
                    "DCMInput": False,
                    "spacing": "[1,0.3]",
                    "speed_per_scale": "[1,1]",
                    "agent_FOV": "[64,64,64]",
                    "spawn_radius": "10"
                    }

            list_process.append(
                {
                    "Process": ALIProcess,
                    "Parameter": parameter_ali_max,
                    "Module": "ALI - Identifying T2 Landmarks (MAX)",
                    "Display": DisplayALICBCT(30,
                        nb_scan
                    )
                },
            )

            parameter_ali_mand = {
                    "input": os.path.join(registeredscan_folder_path,"Mandible"),
                    "dir_models": kwargs["model_folder_ali"],
                    "lm_type": ",".join([f"'{e}'" for e in list_landmark_max]),
                    "output_dir": t2_landmarks_mand_folder_path,
                    "temp_fold": _unique_temp_dir("ali"),
                    "DCMInput": False,
                    "spacing": "[1,0.3]",
                    "speed_per_scale": "[1,1]",
                    "agent_FOV": "[64,64,64]",
                    "spawn_radius": "10"
                    }

            list_process.append(
                {
                    "Process": ALIProcess,
                    "Parameter": parameter_ali_mand,
                    "Module": "ALI - Identifying T2 Landmarks (MAND)",
                    "Display": DisplayALICBCT(30,
                        nb_scan
                    )
                },
            )
        
        measurements_folder_path = os.path.join(kwargs["OutputFolder"],"Measurements")
        os.makedirs(measurements_folder_path, exist_ok=True)

        if kwargs["mode2"] == "Asymmetry Assesment":
            t2_mand_landmarks = mirrored_registered_mand_landmarks_folder_path
            t2_max_landmarks = mirrored_registered_max_landmarks_folder_path
            t2_cb_landmarks = mirrored_registered_cb_landmarks_folder_path
        else:
            t2_mand_landmarks = t2_landmarks_mand_folder_path
            t2_max_landmarks = t2_landmarks_max_folder_path
            t2_cb_landmarks = t2_landmarks_cb_folder_path

        parameter_aq3dc_cb = {
            "t1_path":landmarks_cb_folder_path,
            "t2_path":t2_cb_landmarks,
            "list_measure":list_measure_cb,
            "output_path":measurements_folder_path,
            "filename":"Measurements_CB.xlsx"
            }

        AQ3DCProcess = run_aq3dc

        list_process.append(
            {
                "Process": AQ3DCProcess,
                "Parameter": parameter_aq3dc_cb,
                "Module": "AQ3DC - CB Measurements",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_aq3dc_mand = {
            "t1_path":landmarks_cb_folder_path,
            "t2_path":t2_mand_landmarks,
            "list_measure":list_measure_mand,
            "output_path":measurements_folder_path,
            "filename":"Measurements_MAND.xlsx"
            }

        list_process.append(
            {
                "Process": AQ3DCProcess,
                "Parameter": parameter_aq3dc_mand,
                "Module": "AQ3DC - MAND Measurements",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_aq3dc_max = {
            "t1_path":landmarks_max_folder_path,
            "t2_path":t2_max_landmarks,
            "list_measure":list_measure_max,
            "output_path":measurements_folder_path,
            "filename":"Measurements_MAX.xlsx"
            }

        list_process.append(
            {
                "Process": AQ3DCProcess,
                "Parameter": parameter_aq3dc_max,
                "Module": "AQ3DC - MAX Measurements",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        if kwargs["mode2"] == "Asymmetry Assesment":
            PostProcessAQ3DC = postprocess

            parameter_postprocessaq3dc = {
                "cb_path":os.path.join(measurements_folder_path,"Measurements_CB.xlsx"),
                "mand_path":os.path.join(measurements_folder_path,"Measurements_MAND.xlsx"),
                "max_path":os.path.join(measurements_folder_path,"Measurements_MAX.xlsx"),
                "exemple_path":feature_path,
                "outputfolder": measurements_folder_path
                }

            list_process.append(
                {
                    "Process": PostProcessAQ3DC,
                    "Parameter": parameter_postprocessaq3dc,
                    "Module": "Post process AQ3DC",
                    "Display": DisplayAREGCBCT(
                        nb_scan
                    ),
                },
            )
            classification_folder_path = os.path.join(kwargs["OutputFolder"],"Classification")
            os.makedirs(classification_folder_path, exist_ok=True)

            parameter_asymclass = {
                    "model_path": kwargs["model_vface"],
                    "excel_path": os.path.join(measurements_folder_path,"PostProcess_Measurements.xlsx"),
                    "output_path": os.path.join(classification_folder_path,"Classification.xlsx")
            }

            list_process.append(
                {
                    "Process": AsymProcess,
                    "Parameter": parameter_asymclass,
                    "Module": "Asym_Class - Identifying Asymmetry type",
                    "Display": DisplayAREGCBCT(
                        nb_scan
                    ),
                },
            )

    if kwargs["bool_visualization"]:

        vtk_folder_path = os.path.join(kwargs["OutputFolder"],"VTK Files")
        os.makedirs(vtk_folder_path, exist_ok=True)

        BDSProcess = run_bds
            
        t1_cb_vtk_folder_path = os.path.join(vtk_folder_path,"T1 CB")
        os.makedirs(t1_cb_vtk_folder_path, exist_ok=True)

        t1_max_vtk_folder_path = os.path.join(vtk_folder_path,"T1 MAX")
        os.makedirs(t1_max_vtk_folder_path, exist_ok=True)

        t2_cb_vtk_folder_path = os.path.join(vtk_folder_path,"T2 CB")
        os.makedirs(t2_cb_vtk_folder_path, exist_ok=True)

        t2_mand_vtk_folder_path = os.path.join(vtk_folder_path,"T2 MAND")
        os.makedirs(t2_mand_vtk_folder_path, exist_ok=True)

        t2_max_vtk_folder_path = os.path.join(vtk_folder_path,"T2 MAX")
        os.makedirs(t2_max_vtk_folder_path, exist_ok=True)

        parameter_bds_t1_cb = {
            "input_path":orientation_cb_folder_path,
            "output_path":t1_cb_vtk_folder_path,
            }

        list_process.append(
            {
                "Process": BDSProcess,
                "Parameter": parameter_bds_t1_cb,
                "Module": "BDS - Segmentation T1 CB",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_bds_t1_max = {
            "input_path":orientation_max_folder_path,
            "output_path":t1_max_vtk_folder_path,
            }

        list_process.append(
            {
                "Process": BDSProcess,
                "Parameter": parameter_bds_t1_max,
                "Module": "BDS - Segmentation T1 MAX",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_bds_t2_cb = {
            "input_path":os.path.join(registeredscan_folder_path,"Cranial Base"),
            "output_path":t2_cb_vtk_folder_path,
            }
        
        if kwargs["mode"] == "File already Registered":
            parameter_bds_t2_cb["input_path"] = os.path.join(registeredscan_folder_path,"Cranial Base")

        list_process.append(
            {
                "Process": BDSProcess,
                "Parameter": parameter_bds_t2_cb,
                "Module": "BDS - Segmentation T2 CB",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_bds_t2_mand = {
            "input_path":os.path.join(registeredscan_folder_path,"Mandible"),
            "output_path":t2_mand_vtk_folder_path,
            }
        
        if kwargs["mode"] == "File already Registered":
            parameter_bds_t2_mand["input_path"] = os.path.join(registeredscan_folder_path,"Mandible")

        list_process.append(
            {
                "Process": BDSProcess,
                "Parameter": parameter_bds_t2_mand,
                "Module": "BDS - Segmentation T2 MAND",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_bds_t2_max = {
            "input_path":os.path.join(registeredscan_folder_path,"Maxilla"),
            "output_path":t2_max_vtk_folder_path,
            }
        
        if kwargs["mode"] == "File already Registered":
            parameter_bds_t2_max["input_path"] = os.path.join(registeredscan_folder_path,"Maxilla")

        list_process.append(
            {
                "Process": BDSProcess,
                "Parameter": parameter_bds_t2_max,
                "Module": "BDS - Segmentation T2 MAX",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
                "ReviewTitle": "Bone surfaces used for the heatmaps",
                "ReviewHint": (
                    "Check the surfaces before the distance maps are computed. "
                    "Nothing to edit here - look at the result, then click "
                    "Continue."
                ),
                # The whole VTK folder, not just the last one written: the
                # heatmaps compare T1 against T2, and reviewing one side alone
                # cannot tell whether they are worth comparing. It also keeps
                # the pause from showing nothing when that folder is empty.
                "ReviewFolder": vtk_folder_path,
                # One surface per scan rather than the six BDS writes: the
                # merged one already holds every structure, and eighteen skulls
                # at once is a blob nobody can judge.
                "ReviewNameContains": "_merged",
                "ReviewId": "bone_surfaces",
                "pause_for_visualization": True,
            },
        )

        heatmap_folder_path = os.path.join(kwargs["OutputFolder"],"Heatmaps")
        os.makedirs(heatmap_folder_path, exist_ok=True)
        HeatmapProcess = batch_process

        parameter_heatmap_cb = {
            "t1_dir":t1_cb_vtk_folder_path,
            "t2_dir":t2_cb_vtk_folder_path,
            "patient_list":patients.keys(),
            "output_dir":heatmap_folder_path,
            "zone_type":"merged"
            }

        list_process.append(
            {
                "Process": HeatmapProcess,
                "Parameter": parameter_heatmap_cb,
                "Module": "ModelToModel Distance CB",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_heatmap_mand = {
            "t1_dir":t1_cb_vtk_folder_path,
            "t2_dir":t2_mand_vtk_folder_path,
            "patient_list":patients.keys(),
            "output_dir":heatmap_folder_path,
            "zone_type":"Mandible"
            }

        list_process.append(
            {
                "Process": HeatmapProcess,
                "Parameter": parameter_heatmap_mand,
                "Module": "ModelToModel Distance MAND",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )

        parameter_heatmap_max = {
            "t1_dir":t1_max_vtk_folder_path,
            "t2_dir":t2_max_vtk_folder_path,
            "patient_list":patients.keys(),
            "output_dir":heatmap_folder_path,
            "zone_type":"Upper_Skull"
            }

        list_process.append(
            {
                "Process": HeatmapProcess,
                "Parameter": parameter_heatmap_max,
                "Module": "ModelToModel Distance MAX",
                "Display": DisplayAREGCBCT(
                    nb_scan
                ),
            },
        )
    
    return list_process

def run_aq3dc(t1_path, t2_path, list_measure, output_path, filename):
    
    modules_to_remove = [name for name in sys.modules.keys() if name.startswith('Classes.')]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    
    logic = LocalAQ3DCLogic()
    
    for measure in list_measure:
        if not hasattr(measure, 'keep_sign') or measure.keep_sign is None:
            import qt
            measure.keep_sign = qt.QCheckBox()
            measure.keep_sign.setChecked(True)
    
    patient_T1, x = logic.createDictPatient(t1_path)
    patient_T2,x = logic.createDictPatient(t2_path)
    cat_patient = logic.concatenateT1T2Patient(patient_T1,patient_T2)
    compute = logic.computeMeasurement(list_measure,cat_patient)
    compute = reorganizeStat(compute)
        
    logic.writeMeasurementExcel(compute,output_path,filename)



def _ali_group_labels():
    """ALI's landmark-to-group map, read from its source without importing torch."""
    import ast

    constants = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "ALI_CBCT", "ALI_CBCT_utils", "constants.py",
    )
    try:
        tree = ast.parse(open(constants).read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "GROUP_LABELS" for t in node.targets
            ):
                groups = ast.literal_eval(node.value)
                return {lm: g for g, lms in groups.items() for lm in lms}
    except Exception as e:
        logger.warning(f"Could not read ALI landmark groups ({e}), writing one group")
    return {}


def _read_landmarks(folder):
    """{patient: {label: position}} for every .mrk.json under folder."""
    patients = {}
    for path in sorted(GetListFiles(folder, [".mrk.json"])):
        patient = patientIdFromFileName(os.path.basename(path))
        try:
            data = json.load(open(path))
        except Exception as e:
            logger.warning(f"Could not read {os.path.basename(path)}: {e}")
            continue
        for markup in data.get("markups", []):
            for point in markup.get("controlPoints", []):
                patients.setdefault(patient, {})[point["label"]] = list(point["position"])
    return patients


def _transforms_by_patient(folder):
    """{patient: path} for the ASO .tfm files written next to the oriented scans."""
    found = {}
    for path in sorted(GetListFiles(folder, [".tfm"])):
        found[patientIdFromFileName(os.path.basename(path))] = path
    return found


def _scan_stems_by_patient(folder):
    """{patient: scan basename without extension}, to name outputs as ALI does."""
    stems = {}
    for ext in [".nii.gz", ".nii", ".nrrd", ".nrrd.gz", ".gipl.gz", ".gipl"]:
        for path in sorted(GetListFiles(folder, [ext])):
            name = os.path.basename(path)
            stems.setdefault(patientIdFromFileName(name), name[: -len(ext)])
    return stems


def derive_landmarks(source_folder, source_tfm_folder, target_tfm_folder,
                     target_scan_folder, keep_landmarks, output_folder):
    """
    Express landmarks found in one oriented frame in another oriented frame.

    Both orientations come from the same centred scan, so a landmark's position
    in the target frame is inv(target_transform) . source_transform applied to
    its position in the source frame. Running the landmark search a second time
    on the other orientation costs minutes per patient and makes the same
    anatomical point land in two slightly different places; a rigid transform
    is exact and instant.

    Args:
        source_folder: folder holding the landmarks already found
        source_tfm_folder: folder holding the source orientation's ASO transform
        target_tfm_folder: folder holding the target orientation's ASO transform
        target_scan_folder: oriented scans of the target frame, used for naming
        keep_landmarks: labels to write out
        output_folder: where the derived landmark files go

    Returns:
        bool: True if every patient with landmarks produced an output
    """
    import SimpleITK as sitk

    source = _read_landmarks(source_folder)
    source_tfm = _transforms_by_patient(source_tfm_folder)
    target_tfm = _transforms_by_patient(target_tfm_folder)
    stems = _scan_stems_by_patient(target_scan_folder)
    group_of = _ali_group_labels()

    if not source:
        logger.error(f"No landmarks to derive from in {source_folder}")
        return False

    wanted = list(keep_landmarks)
    os.makedirs(output_folder, exist_ok=True)
    ok = True

    for patient, points in sorted(source.items()):
        if patient not in source_tfm or patient not in target_tfm:
            logger.error(
                f"{patient}: missing an orientation transform, cannot derive its "
                f"landmarks (source={patient in source_tfm}, target={patient in target_tfm})"
            )
            ok = False
            continue

        try:
            to_centred = sitk.ReadTransform(source_tfm[patient])
            from_centred = sitk.ReadTransform(target_tfm[patient]).GetInverse()
        except Exception as e:
            logger.error(f"{patient}: could not read an orientation transform: {e}")
            ok = False
            continue

        missing = [lm for lm in wanted if lm not in points]
        if missing:
            logger.warning(f"{patient}: not found in the source landmarks: {missing}")

        by_group = {}
        for label in wanted:
            if label not in points:
                continue
            moved = from_centred.TransformPoint(to_centred.TransformPoint(points[label]))
            by_group.setdefault(group_of.get(label, "U"), []).append((label, moved))

        if not by_group:
            logger.error(f"{patient}: none of the requested landmarks were available")
            ok = False
            continue

        stem = stems.get(patient, f"{patient}_derived")
        for group, entries in by_group.items():
            control_points = [
                {
                    "id": str(i + 1),
                    "label": label,
                    "description": "",
                    "associatedNodeID": "",
                    "position": [float(c) for c in position],
                    "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "selected": True,
                    "locked": True,
                    "visibility": True,
                    "positionStatus": "defined",
                }
                for i, (label, position) in enumerate(entries)
            ]
            out_path = os.path.join(output_folder, f"{stem}_lm_Pred_{group}.mrk.json")
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "@schema": "https://raw.githubusercontent.com/slicer/slicer/"
                                   "master/Modules/Loadable/Markups/Resources/Schema/"
                                   "markups-schema-v1.0.0.json#",
                        "markups": [
                            {
                                "type": "Fiducial",
                                "coordinateSystem": "LPS",
                                "locked": False,
                                "labelFormat": "%N-%d",
                                "controlPoints": control_points,
                                "measurements": [],
                                "display": {
                                    "visibility": False,
                                    "opacity": 1.0,
                                    "color": [0.4, 1.0, 0.0],
                                    "selectedColor": [1.0, 0.5, 0.5],
                                    "activeColor": [0.4, 1.0, 0.0],
                                    "propertiesLabelVisibility": False,
                                    "pointLabelsVisibility": True,
                                    "textScale": 3.0,
                                },
                            }
                        ],
                    },
                    f,
                    indent=2,
                )
            logger.info(f"{patient}: {len(entries)} landmark(s) derived -> {os.path.basename(out_path)}")

    return ok


def run_bds(input_path, output_path, model_name="DentalSegmentator", device="cuda"):
    
    import sys
    import os
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from VFACE_utils.segmentation_logic import run_dental_segmentation, ExportFormat
        
        logger.info(f"[BDS] Starting dental segmentation...")
        logger.info(f"[BDS] Input folder: {input_path}")
        logger.info(f"[BDS] Output folder: {output_path}")
        logger.info(f"[BDS] Model: {model_name}")
        logger.info(f"[BDS] Device: {device}")
        
        export_formats = ExportFormat.VTK | ExportFormat.VTK_MERGED
        
        success = run_dental_segmentation(
            input_folder=input_path,
            output_folder=output_path,
            model_name=model_name,
            device=device,
            export_formats=export_formats
        )
        
        if success:
            logger.info(f"[BDS] Dental segmentation completed successfully")
            logger.info(f"[BDS] Results saved to: {output_path}")
        else:
            logger.error(f"[BDS] Dental segmentation failed")
            
        return success
        
    except ImportError as e:
        logger.error(f"[BDS] Failed to import segmentation logic: {str(e)}")
        logger.error(f"[BDS] Make sure segmentation_logic.py is in the parent directory")
        return False
    except Exception as e:
        logger.error(f"[BDS] Error in dental segmentation: {str(e)}")
        return False

def reorganizeStat(patient_compute):
        dic_stats = {
                "ID":[],
                "Landmarks":[],
                "Transverse":[],
                "AP":[],
                "Vertical":[],
                "3D":[],
                "Yaw":[],
                "Pitch":[],
                "Roll":[],
                "BL":[],
                "MD":[],
                "Rotation":[],
                "Arch":[],
                "Segment":[]

            }

        TOOTHS = ["UR8", "UR7", "UR6", "UR5", "UR4", "UR3","UR1", "UR2","UL8", "UL7", "UL6", "UL5", "UL4", "UL3","UL1", "UL2",
                  "LR8", "LR7", "LR6", "LR5", "LR4", "LR3","LR1", "LR2","LL8", "LL7", "LL6", "LL5", "LL4", "LL3","LL1", "LL2"]

        for i in range(len(patient_compute["Patient"])) :


            # Strip a P/Pat/Patient prefix only when it is glued to the number
            # (P1 -> 1), which is what this was for. Chopping the first character
            # unconditionally turned P_0001 into "_0001" and "P" into "", and an
            # empty ID leaves postprocess with nothing to group the rows by.
            patient = str(patient_compute["Patient"][i])
            numbered = re.fullmatch(r"(?:patient|pat|p)[ _-]?(\d+)", patient, re.IGNORECASE)
            dic_stats["ID"].append(numbered.group(1) if numbered else patient)

            dic_stats["Landmarks"].append(patient_compute["Landmarks"][i])

            T1=False
            T2=False
            if "T1" in patient_compute["Type of measurement"][i]:
                T1=True
            if "T2" in patient_compute["Type of measurement"][i]:
                T2=True


            type = "skeletal"
            tooth = None
            for t in TOOTHS :
                if t in patient_compute["Landmarks"][i] :
                    tooth = t

            if tooth != None:
                #Arch : upper=0, lower=1
                if "U" in tooth :
                    dic_stats["Arch"].append(0)
                else :
                    dic_stats["Arch"].append(1)

                #Segment : posterior=0,anterior=1
                if "1" in tooth or "2" in tooth:
                    dic_stats["Segment"].append(1)
                else :
                    dic_stats["Segment"].append(0)


                if dic_stats["Segment"][len(dic_stats["Segment"])-1] == 1 : # is anterior teeth

                    #AP
                    ap = patient_compute["A-P Component"][i]
                    if ap!="x" and ap!="":
                        ap=float(ap)
                        if patient_compute["A-P Meaning"][i]=="L":
                            ap=-ap
                    dic_stats["AP"].append(str(ap))

                    #Transverse-RL
                    rl = patient_compute["R-L Component"][i]
                    if rl!="x" and rl!="":
                        rl=float(rl)
                        if patient_compute["R-L Meaning"][i]=="D":
                            rl=-rl
                    dic_stats["Transverse"].append(str(rl))


                    #BL
                    pitch = patient_compute["Pitch Component"][i]
                    if pitch!="x" and pitch!="":
                        pitch=float(pitch)
                        if patient_compute["Pitch Meaning"][i]=="L":
                            pitch=-pitch
                    dic_stats["BL"].append(str(pitch))

                    #MD
                    roll = patient_compute["Roll Component"][i]
                    if roll!="x" and roll!="":
                        roll=float(roll)
                        if patient_compute["Roll Meaning"][i]=="D":
                            roll=-roll
                    dic_stats["MD"].append(str(roll))


                else :
                    #AP
                    ap = patient_compute["A-P Component"][i]
                    if ap!="x" and ap!="":
                        ap=float(ap)
                        if patient_compute["A-P Meaning"][i]=="D":
                            ap=-ap
                    dic_stats["AP"].append(str(ap))

                    #Transverse-RL
                    rl = patient_compute["R-L Component"][i]
                    if rl!="x" and rl!="":
                        rl=float(rl)
                        if patient_compute["R-L Meaning"][i]=="B":
                            rl=-rl
                    dic_stats["Transverse"].append(str(rl))

                    #MD
                    pitch = patient_compute["Pitch Component"][i]
                    if pitch!="x" and pitch!="":
                        pitch=float(pitch)
                        if patient_compute["Pitch Meaning"][i]=="D":
                            pitch=-pitch
                    dic_stats["MD"].append(str(pitch))

                    #BL
                    roll = patient_compute["Roll Component"][i]
                    if roll!="x" and roll!="":
                        roll=float(roll)
                        if patient_compute["Roll Meaning"][i]=="L":
                            roll=-roll
                    dic_stats["BL"].append(str(roll))

                #Vertical
                si = patient_compute["S-I Component"][i]
                if si!="x" and si!="":
                    si=float(si)
                    if patient_compute["S-I Meaning"][i]=="I":
                        si=-si
                dic_stats["Vertical"].append(str(si))

                #Rotation
                yaw = patient_compute["Yaw Component"][i]
                if yaw!="x" and yaw!="":
                    yaw=float(yaw)
                    if patient_compute["Yaw Meaning"][i]=="DR":
                        yaw=-yaw
                dic_stats["Rotation"].append(str(yaw))

                #3D
                ThreeD = patient_compute["3D Distance"][i]
                dic_stats["3D"].append(str(ThreeD))

                dic_stats["Yaw"].append(str("x"))
                dic_stats["Pitch"].append(str("x"))
                dic_stats["Roll"].append(str("x"))

            else :
                dic_stats["Arch"].append("x")
                dic_stats["Segment"].append("x")

                dic_stats["BL"].append(str("x"))
                dic_stats["MD"].append(str("x"))
                dic_stats["Rotation"].append(str("x"))

                #Transverse-RL
                rl = patient_compute["R-L Component"][i]
                if rl!="x" and rl!="":
                    rl=float(rl)
                    if patient_compute["R-L Meaning"][i]=="Medial" or patient_compute["R-L Meaning"][i]=="L":
                        rl=-rl
                dic_stats["Transverse"].append(str(rl))


                #AP
                ap = patient_compute["A-P Component"][i]
                if ap!="x" and ap!="":
                    ap=float(ap)
                    if patient_compute["A-P Meaning"][i]=="P":
                        ap=-ap
                dic_stats["AP"].append(str(ap))

                #Vertical
                si = patient_compute["S-I Component"][i]
                if si!="x" and si!="":
                    si=float(si)
                    if patient_compute["S-I Meaning"][i]=="S":
                        si=-si
                dic_stats["Vertical"].append(str(si))

                #Yaw
                yaw = patient_compute["Yaw Component"][i]
                if yaw!="x" and yaw!="":
                    yaw=float(yaw)
                    if patient_compute["Yaw Meaning"][i]=="CounterC":
                        yaw=-yaw
                dic_stats["Yaw"].append(str(yaw))

                #Pitch
                pitch = patient_compute["Pitch Component"][i]
                if pitch!="x" and pitch!="":
                    pitch=float(pitch)
                    if patient_compute["Pitch Meaning"][i]=="CounterC":
                        pitch=-pitch
                dic_stats["Pitch"].append(str(pitch))

                #BL
                roll = patient_compute["Roll Component"][i]
                if roll!="x" and roll!="":
                    roll=float(roll)
                    if patient_compute["Roll Meaning"][i]=="CounterC":
                        roll=-roll
                dic_stats["Roll"].append(str(roll))

                #3D
                ThreeD = patient_compute["3D Distance"][i]
                dic_stats["3D"].append(str(ThreeD))


        keys_to_delete = []
        for key, value in dic_stats.items():
            if not value:  # Check if the list is empty
                keys_to_delete.append(key)
            elif all(item == "x" for item in value):  # Check if all items in the list are "x"
                keys_to_delete.append(key)

        # Deleting the keys where the condition is not met
        for key in keys_to_delete:
            del dic_stats[key]


        return dic_stats

def SplitMeasurements(measurements_folder,mode2):
    
    measurements_folder = Path(measurements_folder)
    
    cb_path = None
    mand_path = None 
    max_path = None
    features_path = None
    
    excel_files = list(measurements_folder.glob("*.xlsx")) + list(measurements_folder.glob("*.xls"))
    
    if len(excel_files) == 0:
        raise FileNotFoundError(f"No Excel files in the folder: {measurements_folder}")
    

    for file_path in excel_files:
        filename = file_path.name.upper()
        
        if "CB" in filename or "CRANIAL" in filename or "CRANIOFACIAL" in filename:
            cb_path = str(file_path)
        
        elif "MAND" in filename or "MANDIBLE" in filename or "MANDIBULAR" in filename:
            mand_path = str(file_path)
        
        elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename:
            max_path = str(file_path)

        if mode2 == "Asymmetry Assesment":
            if "FEAT" in filename or "FEATURE" in filename:
                features_path = str(file_path)
    
    missing_files = []
    if cb_path is None:
        missing_files.append("CB (Cranial Base)")
    if mand_path is None:
        missing_files.append("MAND (Mandible)")
    if max_path is None:
        missing_files.append("MAX (Maxilla)")
    if features_path is None:
        if mode2 == "Asymmetry Assesment":
            missing_files.append("Features")
    
    if missing_files:
        logger.warning(f"Missing Excel Files: {', '.join(missing_files)}")
    
    logger.info(f"All measurements files have been successfully identify")
    return cb_path, mand_path, max_path,features_path

def SplitOriented(t1_folder):
    
    t1_folder = Path(t1_folder)
    
    oriented_files = []
    
    cbct_files = list(t1_folder.glob("*.nii")) + list(t1_folder.glob("*.nii.gz")) + list(t1_folder.glob("*.nrrd")) + list(t1_folder.glob("*.nrrd.gz"))
    
    if len(cbct_files) == 0:
        raise FileNotFoundError(f"No CBCT files in the folder: {t1_folder}")

    for file_path in cbct_files:
        filename = file_path.name.upper()

        if "CB" in filename or "CRANIAL" in filename or "CRANIOFACIAL" in filename:
            oriented_files.append({
                'type': 'CB',
                'path': str(file_path),
                'destination_folder': 'CB'
            })
        
        elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename:
            oriented_files.append({
                'type': 'MAX',
                'path': str(file_path),
                'destination_folder': 'MAX'
            })
    
    if not oriented_files:
        logger.warning(f"No oriented files found in {t1_folder}")
    else:
        logger.info(f"Found {len(oriented_files)} oriented file(s)")
    
    return oriented_files

def SplitT2(t2_folder, transform):
    
    t2_folder = Path(t2_folder)
    
    t2_files = []
    
    cbct_files = list(t2_folder.glob("*.nii")) + list(t2_folder.glob("*.nii.gz")) + list(t2_folder.glob("*.nrrd")) + list(t2_folder.glob("*.nrrd.gz"))
    transform_files = list(t2_folder.glob("*.tfm")) if transform else []

    if len(cbct_files) == 0 or (transform and len(transform_files) == 0):
        raise FileNotFoundError(f"No CBCT or transform files in the folder: {t2_folder}")

    # Process CBCT files
    for file_path in cbct_files:
        filename = file_path.name.upper()

        if "CB" in filename or "CRANIAL" in filename or "CRANIOFACIAL" in filename:
            t2_files.append({
                'type': 'CB',
                'path': str(file_path),
                'destination_folder': 'Cranial Base',
                'file_type': 'cbct'
            })

        elif "MAND" in filename or "MANDIBLE" in filename or "MANDIBULAR" in filename or "MD" in filename:
            t2_files.append({
                'type': 'MAND',
                'path': str(file_path),
                'destination_folder': 'Mandible',
                'file_type': 'cbct'
            })
        
        elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename or "MX" in filename:
            t2_files.append({
                'type': 'MAX',
                'path': str(file_path),
                'destination_folder': 'Maxilla',
                'file_type': 'cbct'
            })

    # Process transform files if requested
    if transform:
        for file_path in transform_files:
            filename = file_path.name.upper()

            if "CB" in filename or "CRANIAL" in filename or "CRANIOFACIAL" in filename:
                t2_files.append({
                    'type': 'CB',
                    'path': str(file_path),
                    'destination_folder': 'Cranial Base',
                    'file_type': 'transform'
                })

            elif "MAND" in filename or "MANDIBLE" in filename or "MANDIBULAR" in filename:
                t2_files.append({
                    'type': 'MAND',
                    'path': str(file_path),
                    'destination_folder': 'Mandible',
                    'file_type': 'transform'
                })
            
            elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename:
                t2_files.append({
                    'type': 'MAX',
                    'path': str(file_path),
                    'destination_folder': 'Maxilla',
                    'file_type': 'transform'
                })

    if not t2_files:
        logger.warning(f"No T2 files found in {t2_folder}")
    else:
        cbct_count = len([f for f in t2_files if f['file_type'] == 'cbct'])
        transform_count = len([f for f in t2_files if f['file_type'] == 'transform'])
        logger.info(f"Found {cbct_count} CBCT file(s)" + (f" and {transform_count} transform file(s)" if transform else ""))
    
    return t2_files

def TranslateModels(listeModels, mask=False):
    dicTranslate = {
        "Models": {
            "Mandible": "MAND",
            "Maxilla": "MAX",
            "Cranial Base": "CB",
            "Cervical Vertebra": "CV",
            "Root Canal": "RC",
            "Mandibular Canal": "MCAN",
            "Upper Airway": "UAW",
            "Skin": "SKIN",
        },
        "Masks": {
            "Cranial Base": "CBMASK",
            "Mandible": "MANDMASK",
            "Maxilla": "MAXMASK",
        },
    }

    translate = ""
    for i, model in enumerate(listeModels):
        if i < len(listeModels) - 1:
            if mask:
                translate += dicTranslate["Masks"][model] + ","
            else:
                translate += dicTranslate["Models"][model] + ","
        else:
            if mask:
                translate += dicTranslate["Masks"][model]
            else:
                translate += dicTranslate["Models"][model]

    return translate

def NumberScan(scan_folder_t1: str):
    return len(GetPatients(scan_folder_t1))

def GetListNamesSegType(segmentationType):
    dic = {
        "CB": ["cb"],
        "MAND": ["mand", "md"],
        "MAX": ["max", "mx"],
    }
    return dic[segmentationType]

def create_list_measure(df_path):

    df = pd.read_excel(df_path,sheet_name=None)
    list_measure = []

    for sheet in list(df.keys()):
        sheet_df = df[sheet]
        if "Type of measurement" in sheet_df.columns and "Point 1" in sheet_df.columns and "Point 2 / Line" in sheet_df.columns:
            for row in sheet_df[['Type of measurement', 'Point 1', 'Point 2 / Line']].itertuples(index=False):
                list_measure = list_measure + AQ3DCLogic.createMeasurement(AQ3DCLogic(),[row[0]],list(row[1:]))

        elif "Type of measurement" in sheet_df.columns and "Line 1" in sheet_df.columns and "Line 2" in sheet_df.columns:
            for row in sheet_df[['Type of measurement', 'Line 1', 'Line 2']].itertuples(index=False):
                list_measure = list_measure + AQ3DCLogic.createMeasurement(AQ3DCLogic(),[row[0]],list(row[1].split("-") + row[2].split("-")))

        else:
            logger.error("There is an issue in the xlsx file")
    return list_measure

def _unique_temp_dir(what):
    """A directory no other step can be handed.

    slicer.util.tempDirectory() names its folder from the clock to the
    millisecond, and the steps of a pipeline are built microseconds apart, so two
    of them are regularly given the *same* directory. That is not theoretical:
    the padded scans and ALI's own temp_fold collided, ALI found elastix's
    leftover fixed_image_masked there and landmarked it instead of the patients.

    Kept under Slicer's temporary root so its own cleanup still applies.

    Args:
        what: short tag that ends up in the folder name, to make a stray one
            traceable back to the step that made it

    Returns:
        str: path of a fresh, empty directory
    """
    root = getattr(slicer.app, "temporaryPath", None) or tempfile.gettempdir()
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"VFACE_{what}_", dir=root)


def pad_scans_for_landmarks(input_folder, output_folder, margin_mm=30):
    """Copy each scan with empty space around it so ALI can work at the edges.

    ALI border-pads the image it samples (agent_fov/2 + 1 voxels), but Agent.Move
    refuses any position outside the *unpadded* size. A landmark sitting a voxel
    or two from the border therefore sends the agent into a bounds bounce: random
    restart, attempt counter up, and after three tries it gives up. Me is 1.2 mm
    above the floor of these CBCTs, and the very same model finds it at once when
    there is room around it.

    Fixing that mismatch belongs in ALI and would serve every module that calls
    it. This is the safe half of the answer: give the scan room, touch nothing
    that AREG and ASO also depend on.

    Physical coordinates are preserved - the origin moves with the padding - so
    the landmarks come back in the original scan's space. Measured against an
    unpadded run: points away from the border moved by at most one voxel.

    Args:
        input_folder: Folder of scans to copy
        output_folder: Where the padded copies go
        margin_mm: Room to add on every side, in millimetres

    Returns:
        str: output_folder, so the caller can feed it straight to ALI
    """
    import SimpleITK as sitk

    os.makedirs(output_folder, exist_ok=True)
    scans = GetListFiles(input_folder, [".nii", ".nii.gz", ".nrrd", ".gipl", ".gipl.gz"])
    if not scans:
        logger.warning(f"No scan to pad in {input_folder}; landmarks will run on it as it is")
        return input_folder

    for path in scans:
        target = os.path.join(output_folder, os.path.basename(path))
        try:
            image = sitk.ReadImage(path)
            pad = [max(1, int(round(margin_mm / sp))) for sp in image.GetSpacing()]
            # The scan's own minimum, not zero: an air value that already exists
            # in the volume keeps the intensity rescaling ALI applies unchanged.
            background = float(sitk.GetArrayViewFromImage(image).min())
            sitk.WriteImage(sitk.ConstantPad(image, pad, pad, background), target)
        except Exception as e:
            # A scan that cannot be padded is still worth landmarking as it is.
            logger.warning(f"Could not pad {os.path.basename(path)}, using it unpadded: {e}")
            shutil.copy(path, target)

    logger.info(f"{len(scans)} scan(s) given {margin_mm} mm of room for the landmark search")
    return output_folder


def create_list_landmark(df_path):

    df = pd.read_excel(df_path)
    list_landmark = []

    if "Type of measurement" in df.columns and "Point 1" in df.columns and "Point 2 / Line" in df.columns:
        for row in df[['Point 1', 'Point 2 / Line']].itertuples(index=False):
            if row[0] not in list_landmark:
                list_landmark.append(row[0])
            if row[1] not in list_landmark:
                list_landmark.append(row[1])
    else:
        logger.error("There is an issue in the xlsx file")
    return list_landmark

def GetListFiles(folder_path, file_extension):
    """Return a list of files in folder_path finishing by file_extension"""
    # search() already returns every extension at once, and each of its keys walks
    # the tree: calling it once per extension walked the tree len(file_extension)**2
    # times and threw away all but one result each round.
    found = search(folder_path, file_extension)
    file_list = []
    for extension_type in file_extension:
        file_list += found[extension_type]
    return file_list


def GetPatients(folder_path, time_point="T1", segmentationType=None, folder_mask=None):
    """Return a dictionary with patient id as key"""
    file_extension = [".nii.gz", ".nii", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz"]
    json_extension = [".json"]
    
    # Get files from main folder
    file_list = GetListFiles(folder_path, file_extension + json_extension)
    
    # Get mask files from mask folder if provided
    mask_files = []
    if folder_mask and os.path.exists(folder_mask):
        mask_files = GetListFiles(folder_mask, file_extension)
    
    # Combine both lists
    all_files = file_list + mask_files
    
    patients = {}

    # TIMEPOINT-SUFFIX: only _T1/_T2 are stripped here, so _T3/_T4 inputs break
    # patient pairing. See the full note above GetPatients in
    # AREG_CBCT/AREG_CBCT_utils/utils.py before changing this.
    for file in all_files:
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

        # Handle mask files separately
        if file in mask_files:
            if segmentationType is None:
                patients[patient]["seg" + time_point] = file
            else:
                if any(
                    kw in basename.lower()
                    for kw in GetListNamesSegType(segmentationType)
                ):
                    patients[patient]["seg" + time_point] = file
                    
        # Handle main folder files
        elif True in [i in basename for i in file_extension]:
            # If it's a segmentation file in main folder
            if True in [i in basename.lower() for i in ["mask", "seg", "pred"]]:
                if segmentationType is None:
                    patients[patient]["seg" + time_point] = file
                else:
                    if any(
                        kw in basename.lower()
                        for kw in GetListNamesSegType(segmentationType)
                    ):
                        patients[patient]["seg" + time_point] = file
            else:
                patients[patient]["scan" + time_point] = file

        # Handle JSON landmark files
        elif True in [i in basename for i in json_extension]:
            if time_point == "T2":
                patients[patient]["lm" + time_point] = file

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
    folder_t1_mask=None,
    segmentationType=None,
    todo_str="",
    matrix_folder=None,
):
    """Return a dictionary with patients for both time points"""
    patients_t1 = GetPatients(folder_t1_path, time_point="T1", segmentationType=segmentationType, folder_mask=folder_t1_mask)
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

def batch_process(t1_dir, t2_dir, patient_list, output_dir, signed=True, output_text=".vtk", zone_type="merged"):
    """
    A batch process that executes each pair of files in a separate subprocess
    to ensure that memory is completely freed between each case.
    
    The fundamental problem is that vtkDistancePolyDataFilter allocates memory
    on the C++ side (cell locators, BSP trees) that neither gc.collect() nor DeepCopy can
    fully free. Only a separate process guarantees memory release via the OS.
    """
    import subprocess
    import json
    import tempfile
    import time
    
    input_dir1 = Path(t1_dir)
    input_dir2 = Path(t2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_zones = ["merged", "Upper_Skull", "Mandible"]
    if zone_type not in valid_zones:
        raise ValueError(f"zone_type must be one of {valid_zones}, got: {zone_type}")
    
    def extract_patient_id_and_zone(filename):
        base_name = filename.replace("_T1_", "_").replace("_T2_", "_")
        zone = None
        if "Upper_Skull" in filename:
            zone = "Upper_Skull"
            patient_id = base_name.split("_Segmentation_Upper_Skull")[0]
        elif "Mandible" in filename:
            zone = "Mandible"
            patient_id = base_name.split("_Segmentation_Mandible")[0]
        elif "merged" in filename:
            zone = "merged"
            patient_id = base_name.split("_Segmentation_merged")[0]
        else:
            parts = base_name.split("_")
            if len(parts) > 1:
                patient_id = "_".join(parts[:-1])
                zone = "general"
            else:
                patient_id = parts[0]
                zone = "general"
        return patient_id, zone
    
    def is_patient_in_list(patient_id, patient_list):
        if not patient_list:
            return True
        patient_id_clean = str(patient_id).strip()
        for list_patient in patient_list:
            list_patient_clean = str(list_patient).strip()
            if patient_id_clean.lower() == list_patient_clean.lower():
                return True
            try:
                pattern_exact = r'\b' + re.escape(list_patient_clean) + r'\b'
                if re.search(pattern_exact, patient_id_clean, re.IGNORECASE):
                    return True
            except:
                pass
            if list_patient_clean.isdigit():
                if patient_id_clean.lower() == f"pat{list_patient_clean}":
                    return True
                if patient_id_clean.lower() == f"patient{list_patient_clean}":
                    return True
                if patient_id_clean.lower() == f"p{list_patient_clean}":
                    return True
        return False

    # TIMEPOINT-SUFFIX: only _T1/_T2 are stripped here, so _T3/_T4 inputs break
    # patient pairing. See the full note above GetPatients in
    # AREG_CBCT/AREG_CBCT_utils/utils.py before changing this.
    def clean_patient_id(patient_id):
        return (
            patient_id.split("_Scan")[0]
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

    t2_files = {}
    for file2 in input_dir2.iterdir():
        if file2.suffix.lower() not in ['.vtk', '.vtp']:
            continue
        patient_id, zone = extract_patient_id_and_zone(file2.name)
        patient = clean_patient_id(patient_id)
        if not is_patient_in_list(patient, patient_list):
            continue
        if zone != zone_type:
            continue
        t2_files[f"{patient}_{zone}"] = file2

    pairs_to_process = []
    for file1 in input_dir1.iterdir():
        if file1.suffix.lower() not in ['.vtk', '.vtp']:
            continue
        patient_id, zone = extract_patient_id_and_zone(file1.name)
        patient = clean_patient_id(patient_id)
        if not is_patient_in_list(patient, patient_list):
            continue
        if zone != zone_type:
            continue
        key = f"{patient}_{zone}"
        file2 = t2_files.get(key)
        if not file2 or not file2.exists():
            continue
        pairs_to_process.append({
            'file1': str(file1),
            'file2': str(file2),
            'patient_id': patient_id,
            'patient': patient,
            'zone': zone,
        })

    total_files = len(pairs_to_process)
    logger.info(f"\nFound {total_files} pairs to process for zone '{zone_type}'")

    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_batch_worker.py")
    
    slicer_python = None
    
    if 'slicer' in globals() or 'slicer' in dir():
        try:
            slicer_home = os.path.dirname(slicer.app.slicerHome)
            candidates = [
                os.path.join(slicer.app.slicerHome, "bin", "PythonSlicer"),
                os.path.join(slicer.app.slicerHome, "bin", "python-real"),
                os.path.join(slicer.app.slicerHome, "bin", "python3"),
                os.path.join(slicer.app.slicerHome, "bin", "python"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    slicer_python = c
                    break
        except Exception:
            pass
    
    if slicer_python is None:
        if 'python' in os.path.basename(sys.executable).lower():
            slicer_python = sys.executable
    
    if slicer_python is None:
        exe_dir = os.path.dirname(sys.executable)
        for name in ["PythonSlicer", "python-real", "python3", "python"]:
            candidate = os.path.join(exe_dir, name)
            if os.path.isfile(candidate):
                slicer_python = candidate
                break
    
    if slicer_python is None:
        slicer_python = sys.executable
    
    logger.info(f"Using Python executable: {slicer_python}")
    logger.info(f"Worker script: {worker_script}")

    # Each pair already runs in its own process, and pairs are independent, so
    # run a few at a time instead of one. Capped low because each worker holds a
    # full pair of meshes plus the distance filter's locators in memory.
    try:
        max_workers = max(1, min(4, (os.cpu_count() or 2) // 2))
    except Exception:
        max_workers = 2
    logger.info(f"Running up to {max_workers} worker(s) at a time")

    processed_pairs = []
    running = []          # (Popen, pair, output_filename, deadline, log_dir)
    queue = list(pairs_to_process)
    launched = 0

    def _launch(pair):
        nonlocal launched
        launched += 1
        output_filename = f"{pair['patient']}_{pair['zone']}_ModelDistance{output_text}"
        output_path = str(output_dir / output_filename)

        logger.info(f"Processing [{launched}/{total_files}]: {Path(pair['file1']).name}")
        logger.info(f"  with: {Path(pair['file2']).name}")
        logger.info(f"  Patient: {pair['patient_id']}, Zone: {pair['zone']}")

        cmd = [
            slicer_python, worker_script,
            "--file1", pair['file1'],
            "--file2", pair['file2'],
            "--output", output_path,
            "--signed" if signed else "--unsigned",
        ]
        # Pipes would deadlock: nothing reads them until the worker exits, so a
        # worker writing more than the pipe buffer blocks for ever. Files never do.
        log_dir = tempfile.mkdtemp(prefix="vface_worker_")
        out_log = os.path.join(log_dir, "stdout.txt")
        err_log = os.path.join(log_dir, "stderr.txt")
        proc = subprocess.Popen(cmd, stdout=open(out_log, "w"), stderr=open(err_log, "w"), text=True)
        return (proc, pair, output_filename, time.monotonic() + 600, log_dir)

    def _read_log(path):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        except OSError:
            return ""

    def _collect(proc, pair, output_filename, timed_out, log_dir):
        out_log = os.path.join(log_dir, "stdout.txt")
        err_log = os.path.join(log_dir, "stderr.txt")
        try:
            if timed_out:
                proc.kill()
                proc.wait()
                logger.error(f"  TIMEOUT processing {Path(pair['file1']).name} (>10 min)")
                return

            stdout = _read_log(out_log)
            if stdout:
                for line in stdout.split('\n'):
                    logger.info(f"  [worker] {line}")

            if proc.returncode != 0:
                logger.error(f"  [worker] ERROR (exit code {proc.returncode}):")
                stderr = _read_log(err_log)
                if stderr:
                    for line in stderr.split('\n')[-5:]:
                        logger.error(f"  [worker] {line}")
                return

            processed_pairs.append({
                'patient_id': pair['patient_id'],
                'zone': pair['zone'],
                't1_file': Path(pair['file1']).name,
                't2_file': Path(pair['file2']).name,
                'output_file': output_filename,
            })
            logger.info(f"Successfully processed {output_filename}")
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    while queue or running:
        # Hold back a new worker while memory is already tight.
        while queue and len(running) < max_workers and not (running and check_memory_usage()):
            try:
                running.append(_launch(queue.pop(0)))
            except Exception as e:
                logger.error(f"  Error: {e}")
                traceback.print_exc()

        still_running = []
        for proc, pair, output_filename, deadline, log_dir in running:
            timed_out = proc.poll() is None and time.monotonic() > deadline
            if proc.poll() is None and not timed_out:
                still_running.append((proc, pair, output_filename, deadline, log_dir))
                continue
            try:
                _collect(proc, pair, output_filename, timed_out, log_dir)
            except Exception as e:
                logger.error(f"  Error: {e}")
                traceback.print_exc()
        running = still_running

        if psutil:
            mem = psutil.virtual_memory()
            logger.debug(f"Memory: {mem.percent:.1f}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")

        if 'slicer' in globals():
            slicer.app.processEvents()
        if running:
            time.sleep(0.2)

    logger.info(f"Processing complete. {len(processed_pairs)}/{total_files} pairs processed.")
    for pair in processed_pairs:
        logger.info(f"  {pair['patient_id']} ({pair['zone']}): {pair['output_file']}")

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
    # Walk the tree once and bucket by extension rather than re-globbing per key.
    entries = sorted(
        iglob(os.path.normpath("/".join([path, "**", "*"])), recursive=True)
    )
    return {key: [i for i in entries if i.endswith(key)] for key in arguments}

def _landmark_label(dic_features, composant, i):
    """Rebuild the "Landmarks" cell a feature column refers to."""
    first = dic_features.get("Landmarks"+str(2*i+1))
    second = dic_features.get("Landmarks"+str(2*i+2))
    if composant in ["Transverse","Vertical","AP"]:
        return first+" - "+second
    return (first+" / "+second).replace("_","-")


def _measurement(by_landmark, dic_features, composant, i, feature, patient):
    """One measurement for a feature column, or None if the run never made it.

    A column can name a landmark the pipeline does not produce - the reference
    sheet asks for "Me", which ALI's mandible model does not predict - and
    indexing that blind ends the whole post-processing on a KeyError. The Excel
    is then never written, and every later step fails looking for it: that is how
    one missing landmark turned into a failed classification.

    A column left empty is visible and recoverable; a run stopped halfway is not.
    """
    label = _landmark_label(dic_features, composant, i)
    row = by_landmark.get(label)
    if row is None:
        logger.warning(
            f"{patient}: no '{label}' measurement, leaving '{feature}' empty"
        )
        return None
    try:
        return float(row[composant])
    except (KeyError, TypeError, ValueError):
        logger.warning(
            f"{patient}: '{label}' carries no usable {composant}, "
            f"leaving '{feature}' empty"
        )
        return None


def _index_measurements(df):
    """{patient: {landmark: row}}, keeping the first row of a repeated landmark."""
    index = {}
    for row in df.to_dict("records"):
        index.setdefault(row["ID"], {}).setdefault(row["Landmarks"], row)
    return index


def postprocess (cb_path,mand_path,max_path,exemple_path,outputfolder):
    file_cb = pd.read_excel(cb_path)
    file_mand = pd.read_excel(mand_path)
    file_max = pd.read_excel(max_path)
    file_alls = pd.read_excel(exemple_path)

    file_all = file_alls.drop(labels=["ID","Asymmetry","Mand","Max"], axis = 1)

    list_position = ["CB","MAND","MAX"]
    dic_composant = {"RL":"Transverse","IS":"Vertical","AP":"AP","Pitch":"Pitch","Yaw":"Yaw","Roll":"Roll"}

    dic_columns = {}
    for col in file_all.columns:
        dic_columns[col] = {"Location":"Unknow","Composant":"Unknow","Average":"No","Landmarks1":"Unknow","Landmarks2":"Unknow","Nbr_Landmarks":0}
        copycol = col
        for pos in list_position:
            if pos in col:
                dic_columns[col]["Location"] = pos
                copycol = copycol.replace(pos+"_","")
        for comp,translation in dic_composant.items():
            if comp in col:
                dic_columns[col]["Composant"] = translation
                copycol = copycol.replace("_"+comp,"")
        if "/" not in copycol:
            split = copycol.split("_")
            if len(split) == 2:
                dic_columns[col]["Landmarks1"] = split [0]
                dic_columns[col]["Landmarks2"] = split [1]
                dic_columns[col]["Nbr_Landmarks"] = 2
            else:
                dic_columns[col]["Average"] = "Yes"
                dic_columns[col]["Nbr_Landmarks"] = len(split)
                for i in range(len(split)):
                    dic_columns[col]["Landmarks"+str(i+1)] = split [i]
        else :
            split = copycol.split("/")
            if len(split) == 2:
                dic_columns[col]["Landmarks1"] = split [0]
                dic_columns[col]["Landmarks2"] = split [1]
                dic_columns[col]["Nbr_Landmarks"] = 2
            else:
                dic_columns[col]["Average"] = "Yes"
                dic_columns[col]["Nbr_Landmarks"] = len(split) + 1
                for i in range(len(split)):
                    if i%2==0:
                        dic_columns[col]["Landmarks"+str(1 + (i//2 * 3))] = split [i]
                    else:
                        scndsplit = split[i].split("_")
                        if len(scndsplit) == 4:
                            dic_columns[col]["Landmarks"+str((3*(i-1))+2)] = scndsplit[0]+"_"+scndsplit[1]
                            dic_columns[col]["Landmarks"+str(3*i)] = scndsplit[2]+"_"+scndsplit[3]
                        else:
                            logger.error("Issue")

    # Index each measurement table by patient then by landmark once, instead of
    # rescanning the whole frame with a boolean mask for every single feature.
    measurements = {
        "CB": _index_measurements(file_cb),
        "MAX": _index_measurements(file_max),
        "MAND": _index_measurements(file_mand),
    }

    unique_cb = file_cb["ID"].unique()
    unique_max = file_max["ID"].unique()
    unique_mand = file_mand["ID"].unique()

    if unique_cb.all() != unique_max.all() or unique_cb.all() !=unique_mand.all():
        logger.error("Issue on the ID patient")

    records = []
    for val in unique_cb:
        record = {"ID": val}

        for feature, dic_features in dic_columns.items():
            location = dic_features.get("Location")
            if location not in measurements:
                continue

            by_landmark = measurements[location].get(val)
            if by_landmark is None:
                logger.warning(f"No {location} measurement for this patient {val}")
                continue

            composant = dic_features.get("Composant")
            if dic_features.get("Average") == "No":
                value = _measurement(by_landmark, dic_features, composant, 0, feature, val)
                if value is None:
                    continue
                record[feature] = value
            else:
                nbr = dic_features.get("Nbr_Landmarks")//2
                average = 0
                for i in range(nbr):
                    value = _measurement(by_landmark, dic_features, composant, i, feature, val)
                    if value is None:
                        # Averaging what is left would quietly report a different
                        # measurement than the column claims to be.
                        average = None
                        break
                    average += value
                if average is None:
                    continue
                record[feature] = average / nbr

        records.append(record)

    output_df = pd.DataFrame(records, columns=file_alls.columns)

    output_df.to_excel(os.path.join(outputfolder,"PostProcess_Measurements.xlsx"),index=False)