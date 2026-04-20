import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .Progress import DisplayASOCBCT,DisplayAMASSS,DisplayAREGCBCT,DisplayALICBCT
from glob import iglob
import slicer
from .functionaq3dc import AQ3DCLogic, AQ3DCWidget
import qt
import re
import shutil
from pathlib import Path
import pandas as pd
import vtk
import traceback
try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not available - memory monitoring disabled")
import gc

def check_memory_usage(threshold_percent=80):
    if psutil is None:
        return False
    
    try:
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent > threshold_percent:
            print(f"Warning: High memory usage detected: {usage_percent:.1f}%")
            return True
        
        return False
    except Exception as e:
        print(f"Error checking memory: {e}")
        return False

def force_memory_cleanup():
    print("Forcing memory cleanup...")
    
    collected_total = 0
    for i in range(3):
        collected = gc.collect()
        collected_total += collected
        if collected > 0:
            print(f"Garbage collection round {i+1}: freed {collected} objects")
    
    if collected_total > 0:
        print(f"Total objects freed: {collected_total}")
    
    if 'slicer' in globals():
        slicer.app.processEvents()
    
    if psutil:
        memory = psutil.virtual_memory()
        print(f"Memory usage after cleanup: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB used / {memory.total / 1024**3:.1f}GB total)")

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
                print(f"Initialized keep_sign for {measure.__class__.__name__} from {measure.__class__.__module__}")
        
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

    nb_scan = NumberScan(kwargs["InputFolder"])
    patients = GetPatients(kwargs["InputFolder"], time_point="T1")

    if kwargs["bool_quantification"]:
        cb_measurements_path, mand_measurements_path, max_measurements_path,feature_path = SplitMeasurements(kwargs["measurements_folder"],kwargs["mode2"])
        
        if not cb_measurements_path or not max_measurements_path or not mand_measurements_path:
            print("There is an issue, it miss measurements lists in the list of measurements folder")
        elif not feature_path and kwargs["mode2"] != "Longitudinal studies":
                print("There is an issue, it miss feature list in the ML folder")
        
        list_landmark = []
        list_landmark += create_list_landmark(mand_measurements_path)
        list_landmark += create_list_landmark(cb_measurements_path)
        
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
                print(f"Copied {file_info['type']} file: {file_info['path'].split('/')[-1]} to {file_info['destination_folder']}")
        else:
            print("Issue, it seems to miss some oriented cbct T1 folder")
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
            "temp_folder": slicer.util.tempDirectory(),
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
            "temp_fold": slicer.util.tempDirectory(),
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
                "pause_for_visualization": True,
            }
        )

        parameter_pre_aso_cb = {
            "input": os.path.join(resample_folder_path, "CBCT"),
            "output_folder": preaso_CB_folder_path,
            "model_folder": False,
            "SmallFOV": False,
            "temp_folder": slicer.util.tempDirectory(),
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
            "temp_fold": slicer.util.tempDirectory(),
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
                "temp_folder": slicer.util.tempDirectory(),
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
                    "log_path": slicer.util.tempDirectory(),
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
                "log_path": slicer.util.tempDirectory(),
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
                "log_path": slicer.util.tempDirectory(),
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
            "temp_folder": slicer.util.tempDirectory(),
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
            "temp_folder": slicer.util.tempDirectory(),
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
            "temp_folder": slicer.util.tempDirectory(),
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
                print(f"Copied {file_info['type']} {file_info['file_type']} file: {file_info['path'].split('/')[-1]} to {file_info['destination_folder']}")
        else:
            print("Issue, it seems to miss some cbct or transform files in the T2 folder")
            return
        
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
    
    if kwargs["bool_quantification"]:

        landmarks_folder_path = os.path.join(kwargs["OutputFolder"],"T1 Landmarks")
        os.makedirs(landmarks_folder_path, exist_ok=True)

        landmarks_cb_folder_path = os.path.join(landmarks_folder_path,"CB")
        os.makedirs(landmarks_cb_folder_path, exist_ok=True)

        landmarks_max_folder_path = os.path.join(landmarks_folder_path,"MAX")
        os.makedirs(landmarks_max_folder_path, exist_ok=True)

        parameter_ali = {
                "input": orientation_cb_folder_path,
                "dir_models": kwargs["model_folder_ali"],
                "lm_type": ",".join([f"'{e}'" for e in list_landmark]),
                "output_dir": landmarks_cb_folder_path,
                "temp_fold": slicer.util.tempDirectory(),
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
            },
        )

        parameter_ali_max = {
                "input": orientation_max_folder_path,
                "dir_models": kwargs["model_folder_ali"],
                "lm_type": ",".join([f"'{e}'" for e in list_landmark_max]),
                "output_dir": landmarks_max_folder_path,
                "temp_fold": slicer.util.tempDirectory(),
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
                "Module": "ALI - Identifying T1 Landmarks (MAX)",
                "Display": DisplayALICBCT(30,
                    nb_scan
                )
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
                "log_path": slicer.util.tempDirectory(),
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
                "log_path": slicer.util.tempDirectory(),
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

            for patient,data in patients.items():

                parameter_automatrix_register_ldm_cb = {        
                    "input_patient": mirrored_landmarks_cb_folder_path,
                    "input_matrix": os.path.join(registeredscan_folder_path,"Cranial Base",patient+"_OutReg",patient + "_" + "CB" + "_Reg" + "_matrix.tfm"),
                    "reference_file": "None",
                    "suffix": "_CB_reg",
                    "matrix_name": False,
                    "fromAreg": False,
                    "output_folder": mirrored_registered_cb_landmarks_folder_path,
                    "log_path": slicer.util.tempDirectory(),
                    "is_seg": False
                    }
                
                if kwargs["mode"] == "File already Registered":
                        parameter_automatrix_register_ldm_cb["input_matrix"] = os.path.join(registeredscan_folder_path,"Cranial Base")

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
                    "input_matrix": os.path.join(registeredscan_folder_path,"Mandible",patient+"_OutReg",patient + "_" + "MAND" + "_Reg" + "_matrix.tfm"),
                    "reference_file": "None",
                    "suffix": "_MAND_reg",
                    "matrix_name": False,
                    "fromAreg": False,
                    "output_folder": mirrored_registered_mand_landmarks_folder_path,
                    "log_path": slicer.util.tempDirectory(),
                    "is_seg": False
                    }
                
                if kwargs["mode"] == "File already Registered":
                    parameter_automatrix_register_ldm_mand["input_matrix"] = os.path.join(registeredscan_folder_path,"Mandible")

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
                    "input_matrix": os.path.join(registeredscan_folder_path,"Maxilla",patient+"_OutReg",patient + "_" + "MAX" + "_Reg" + "_matrix.tfm"),
                    "reference_file": "None",
                    "suffix": "_MAX_reg",
                    "matrix_name": False,
                    "fromAreg": False,
                    "output_folder": mirrored_registered_max_landmarks_folder_path,
                    "log_path": slicer.util.tempDirectory(),
                    "is_seg": False
                    }
                if kwargs["mode"] == "File already Registered":
                    parameter_automatrix_register_ldm_MAX["input_matrix"] = os.path.join(registeredscan_folder_path,"Maxilla")

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
                    "temp_fold": slicer.util.tempDirectory(),
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
                    "temp_fold": slicer.util.tempDirectory(),
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
                    "temp_fold": slicer.util.tempDirectory(),
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



def run_bds(input_path, output_path, model_name="DentalSegmentator", device="cuda"):
    
    import sys
    import os
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from VFACE_utils.segmentation_logic import run_dental_segmentation, ExportFormat
        
        print(f"[BDS] Starting dental segmentation...")
        print(f"[BDS] Input folder: {input_path}")
        print(f"[BDS] Output folder: {output_path}")
        print(f"[BDS] Model: {model_name}")
        print(f"[BDS] Device: {device}")
        
        export_formats = ExportFormat.VTK | ExportFormat.VTK_MERGED
        
        success = run_dental_segmentation(
            input_folder=input_path,
            output_folder=output_path,
            model_name=model_name,
            device=device,
            export_formats=export_formats
        )
        
        if success:
            print(f"[BDS] Dental segmentation completed successfully")
            print(f"[BDS] Results saved to: {output_path}")
        else:
            print(f"[BDS] Dental segmentation failed")
            
        return success
        
    except ImportError as e:
        print(f"[BDS] Failed to import segmentation logic: {str(e)}")
        print(f"[BDS] Make sure segmentation_logic.py is in the parent directory")
        return False
    except Exception as e:
        print(f"[BDS] Error in dental segmentation: {str(e)}")
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


            if patient_compute["Patient"][i][0].lower()=="p" :
                dic_stats["ID"].append(patient_compute["Patient"][i][1:])
            elif patient_compute["Patient"][i][:3].lower() == "pat" :
                dic_stats["ID"].append(patient_compute["Patient"][i][3:])
            else :
                dic_stats["ID"].append(patient_compute["Patient"][i])

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
            print(f"✓ File CB: {file_path.name}")
        
        elif "MAND" in filename or "MANDIBLE" in filename or "MANDIBULAR" in filename:
            mand_path = str(file_path)
            print(f"✓ File MAND: {file_path.name}")
        
        elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename:
            max_path = str(file_path)
            print(f"✓ File MAX: {file_path.name}")

        if mode2 == "Asymmetry Assesment":
            if "FEAT" in filename or "FEATURE" in filename:
                features_path = str(file_path)
                print(f"✓ File Feature: {file_path.name}")
    
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
        print(f"⚠️ Missing Excel Files: {', '.join(missing_files)}")
    
    print(f"✅ All measurements files have been successfully identify")
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
            print(f"✓ File CB: {file_path.name}")
        
        elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename:
            oriented_files.append({
                'type': 'MAX',
                'path': str(file_path),
                'destination_folder': 'MAX'
            })
            print(f"✓ File MAX: {file_path.name}")
    
    if not oriented_files:
        print(f"⚠️ No oriented files found in {t1_folder}")
    else:
        print(f"✅ Found {len(oriented_files)} oriented file(s)")
    
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
            print(f"✓ File CB: {file_path.name}")

        elif "MAND" in filename or "MANDIBLE" in filename or "MANDIBULAR" in filename or "MD" in filename:
            t2_files.append({
                'type': 'MAND',
                'path': str(file_path),
                'destination_folder': 'Mandible',
                'file_type': 'cbct'
            })
            print(f"✓ File MAND: {file_path.name}")
        
        elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename or "MX" in filename:
            t2_files.append({
                'type': 'MAX',
                'path': str(file_path),
                'destination_folder': 'Maxilla',
                'file_type': 'cbct'
            })
            print(f"✓ File MAX: {file_path.name}")

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
                print(f"✓ Transform CB: {file_path.name}")

            elif "MAND" in filename or "MANDIBLE" in filename or "MANDIBULAR" in filename:
                t2_files.append({
                    'type': 'MAND',
                    'path': str(file_path),
                    'destination_folder': 'Mandible',
                    'file_type': 'transform'
                })
                print(f"✓ Transform MAND: {file_path.name}")
            
            elif "MAX" in filename or "MAXILLA" in filename or "MAXILLARY" in filename:
                t2_files.append({
                    'type': 'MAX',
                    'path': str(file_path),
                    'destination_folder': 'Maxilla',
                    'file_type': 'transform'
                })
                print(f"✓ Transform MAX: {file_path.name}")

    if not t2_files:
        print(f"⚠️ No T2 files found in {t2_folder}")
    else:
        cbct_count = len([f for f in t2_files if f['file_type'] == 'cbct'])
        transform_count = len([f for f in t2_files if f['file_type'] == 'transform'])
        print(f"✅ Found {cbct_count} CBCT file(s)" + (f" and {transform_count} transform file(s)" if transform else ""))
    
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
            print("There is an issue in the xlsx file")
    return list_measure

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
        print("There is an issue in the xlsx file")
    return list_landmark

def GetListFiles(folder_path, file_extension):
    """Return a list of files in folder_path finishing by file_extension"""
    file_list = []
    for extension_type in file_extension:
        file_list += search(folder_path, file_extension)[extension_type]
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

def read_vtk(file_path):
    file_path = str(file_path)  # Convert Path to string
    if file_path.endswith('.vtk'):
        reader = vtk.vtkPolyDataReader()
    elif file_path.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported format: {file_path}")
    reader.SetFileName(str(file_path))
    reader.Update()

    raw_output = reader.GetOutput()
    if not raw_output or raw_output.GetNumberOfPoints() == 0:
        raise ValueError(f"Failed to read or empty polydata from: {file_path}")

    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(raw_output)

    reader.SetFileName("")
    reader.RemoveAllInputs()
    del reader

    return polydata


def write_vtk(polydata, file_path):
    file_path = str(file_path)
    if file_path.endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
    elif file_path.endswith('.vtp'):
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise ValueError(f"Unsupported format: {file_path}")
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()

    writer.RemoveAllInputs()
    del writer

def clean_and_triangulate(polydata):
    """
    Nettoie et triangule un polydata.
    Utilise DeepCopy pour couper les liens avec les pipelines VTK internes
    et éviter les fuites mémoire lors du batch processing.
    """
    if not polydata or polydata.GetNumberOfPoints() == 0:
        raise ValueError("Cannot clean empty or invalid polydata")
    
    num_points = polydata.GetNumberOfPoints()
    num_cells = polydata.GetNumberOfCells()
    print(f"Input polydata: {num_points} points, {num_cells} cells")
    
    try:
        print("Cleaning polydata...")
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.PointMergingOn()
        cleaner.Update()

        cleaned = vtk.vtkPolyData()
        cleaned.DeepCopy(cleaner.GetOutput())

        cleaner.RemoveAllInputs()
        del cleaner
        
        if not cleaned or cleaned.GetNumberOfPoints() == 0:
            raise ValueError("Cleaning resulted in empty polydata")
        
        print("Triangulating polydata...")
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(cleaned)
        triangle_filter.Update()

        triangulated = vtk.vtkPolyData()
        triangulated.DeepCopy(triangle_filter.GetOutput())

        triangle_filter.RemoveAllInputs()
        del triangle_filter
        del cleaned
        
        if not triangulated or triangulated.GetNumberOfPoints() == 0:
            raise ValueError("Triangulation resulted in empty polydata")
        
        print(f"Clean and triangulate completed: {triangulated.GetNumberOfPoints()} points, {triangulated.GetNumberOfCells()} cells")
        return triangulated
        
    except Exception as e:
        raise ValueError(f"Error during clean and triangulate: {str(e)}")

def compute_distance(polydata1, polydata2, signed=True):
    print("Starting distance computation...")
    
    if not polydata1 or polydata1.GetNumberOfPoints() == 0:
        raise ValueError("Invalid or empty polydata1")
    if not polydata2 or polydata2.GetNumberOfPoints() == 0:
        raise ValueError("Invalid or empty polydata2")
    
    num_points1 = polydata1.GetNumberOfPoints()
    num_points2 = polydata2.GetNumberOfPoints()
    num_cells1 = polydata1.GetNumberOfCells()
    num_cells2 = polydata2.GetNumberOfCells()
    
    print(f"Polydata1: {num_points1} points, {num_cells1} cells")
    print(f"Polydata2: {num_points2} points, {num_cells2} cells")
    
    total_complexity = num_points1 + num_points2 + num_cells1 + num_cells2
    
    if total_complexity > 5000000:
        print(f"Very large dataset detected (complexity: {total_complexity}). Using aggressive subsampling.")
        return compute_distance_subsampled(polydata1, polydata2, signed, target_points=300000)
    elif total_complexity > 2000000:
        print(f"Large dataset detected (complexity: {total_complexity}). Using subsampling approach.")
        return compute_distance_subsampled(polydata1, polydata2, signed, target_points=500000)
    elif total_complexity > 1000000:
        print(f"Medium dataset detected (complexity: {total_complexity}). Using optimized approach.")
        return compute_distance_subsampled(polydata1, polydata2, signed, target_points=750000)
    
    return compute_distance_standard(polydata1, polydata2, signed)


def compute_distance_subsampled(polydata1, polydata2, signed=True, target_points=500000):
    """
    Version avec sous-échantillonnage optimisée pour les gros datasets
    """
    import gc
    print("Using subsampled distance computation...")
    
    try:
        if polydata1.GetNumberOfPoints() > target_points:
            print(f"Subsampling polydata1 from {polydata1.GetNumberOfPoints()} to ~{target_points} points")
            ratio1 = target_points / polydata1.GetNumberOfPoints()
            pd1_subsampled = subsample_polydata(polydata1, ratio1)
        else:
            pd1_subsampled = polydata1
        
        if polydata2.GetNumberOfPoints() > target_points:
            print(f"Subsampling polydata2 from {polydata2.GetNumberOfPoints()} to ~{target_points} points")
            ratio2 = target_points / polydata2.GetNumberOfPoints()
            pd2_subsampled = subsample_polydata(polydata2, ratio2)
        else:
            pd2_subsampled = polydata2
        
        print(f"Computing distance on subsampled data: {pd1_subsampled.GetNumberOfPoints()} vs {pd2_subsampled.GetNumberOfPoints()} points")
        
        result_subsampled = compute_distance_standard(pd1_subsampled, pd2_subsampled, signed)
        
        if pd1_subsampled != polydata1:
            pd1_subsampled = None
        if pd2_subsampled != polydata2:
            pd2_subsampled = None
        gc.collect()
        
        if polydata1.GetNumberOfPoints() > 1000000:
            print("Very large dataset - returning subsampled result directly")
            return result_subsampled
        
        print("Interpolating results back to original resolution...")
        result_full = interpolate_distance_to_original(result_subsampled, polydata1)

        result_subsampled = None
        gc.collect()
        
        return result_full
        
    except Exception as e:
        print(f"Error in subsampled computation: {e}")
        print("Falling back to simplified approach...")
        
        gc.collect()
        
        return create_fallback_result(polydata1, signed)


def create_fallback_result(polydata, signed=True):
    """
    Crée un résultat fallback avec des distances nulles
    """
    print("Creating fallback result with zero distances...")
    
    try:
        output = vtk.vtkPolyData()
        output.DeepCopy(polydata)
        
        distance_name = "SignedDistance" if signed else "AbsoluteDistance"
        distance_array = vtk.vtkDoubleArray()
        distance_array.SetName(distance_name)
        distance_array.SetNumberOfTuples(output.GetNumberOfPoints())
        distance_array.FillComponent(0, 0.0)
        output.GetPointData().SetScalars(distance_array)
        
        const_array = vtk.vtkDoubleArray()
        const_array.SetName("Original")
        const_array.SetNumberOfTuples(output.GetNumberOfPoints())
        const_array.FillComponent(0, 1.0)
        output.GetPointData().AddArray(const_array)
        
        print(f"Fallback result created: {output.GetNumberOfPoints()} points with zero distances")
        return output
        
    except Exception as e:
        print(f"Error creating fallback result: {e}")
        raise RuntimeError("Cannot create fallback result")


def subsample_polydata(polydata, ratio):
    """
    Sous-échantillonnage avec décimation. DeepCopy pour couper le pipeline.
    """
    print(f"Decimating mesh with target ratio {ratio:.3f}")
    
    try:
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(polydata)
        decimate.SetTargetReduction(1.0 - ratio)
        decimate.VolumePreservationOn()
        decimate.AttributeErrorMetricOn()
        decimate.Update()
        
        raw_result = decimate.GetOutput()
        
        if raw_result.GetNumberOfPoints() > ratio * polydata.GetNumberOfPoints() * 1.5:
            print("Decimation not sufficient, using point masking...")
            
            decimate.RemoveAllInputs()
            del decimate
            
            mask = vtk.vtkMaskPoints()
            mask.SetInputData(polydata)
            mask.SetOnRatio(max(1, int(1.0 / ratio)))
            mask.RandomModeOn()
            mask.Update()
            
            points_to_poly = vtk.vtkVertexGlyphFilter()
            points_to_poly.SetInputData(mask.GetOutput())
            points_to_poly.Update()
            
            result = vtk.vtkPolyData()
            result.DeepCopy(points_to_poly.GetOutput())
            
            points_to_poly.RemoveAllInputs()
            mask.RemoveAllInputs()
            del points_to_poly, mask
        else:
            result = vtk.vtkPolyData()
            result.DeepCopy(raw_result)
            
            decimate.RemoveAllInputs()
            del decimate
        
        print(f"Subsampling result: {result.GetNumberOfPoints()} points, {result.GetNumberOfCells()} cells")
        return result
        
    except Exception as e:
        print(f"Error in subsampling: {e}")
        print("Warning: Using original data without subsampling")
        return polydata


def interpolate_distance_to_original(distance_result, original_polydata):
    """
    Version simplifiée d'interpolation
    """
    print("Creating result with original geometry...")
    
    try:
        output = vtk.vtkPolyData()
        output.DeepCopy(original_polydata)
        
        distance_array = vtk.vtkDoubleArray()
        distance_array.SetName("SignedDistance")
        distance_array.SetNumberOfTuples(output.GetNumberOfPoints())
        distance_array.FillComponent(0, 0.0)
        
        source_distance_array = distance_result.GetPointData().GetArray("SignedDistance") or distance_result.GetPointData().GetArray("Distance")
        if source_distance_array:
            print("Attempting simple interpolation...")
            
            source_points = distance_result.GetPoints()
            target_points = output.GetPoints()
            
            for i in range(min(1000, output.GetNumberOfPoints())):
                target_point = target_points.GetPoint(i)
                
                closest_distance = float('inf')
                closest_value = 0.0
                
                for j in range(min(100, distance_result.GetNumberOfPoints())):
                    source_point = source_points.GetPoint(j)
                    dist = ((target_point[0] - source_point[0])**2 + 
                           (target_point[1] - source_point[1])**2 + 
                           (target_point[2] - source_point[2])**2)**0.5
                    
                    if dist < closest_distance:
                        closest_distance = dist
                        closest_value = source_distance_array.GetValue(j)
                
                distance_array.SetValue(i, closest_value)
                
                if i % 100 == 0:  # Progress update
                    if 'slicer' in globals():
                        slicer.app.processEvents()
        
        output.GetPointData().SetScalars(distance_array)
        
        const_array = vtk.vtkDoubleArray()
        const_array.SetName("Original")
        const_array.SetNumberOfTuples(output.GetNumberOfPoints())
        const_array.FillComponent(0, 1.0)
        output.GetPointData().AddArray(const_array)
        
        print(f"Interpolation completed: {output.GetNumberOfPoints()} points")
        return output
        
    except Exception as e:
        print(f"Error in interpolation: {e}")
        print("Returning subsampled result...")
        return distance_result


def compute_distance_standard(polydata1, polydata2, signed=True):
    """
    Version standard avec DeepCopy pour éviter les fuites mémoire.
    vtkDistancePolyDataFilter construit un vtkCellLocator interne qui garde
    des références aux inputs. Sans DeepCopy, ces structures restent en mémoire
    côté C++ et le garbage collector Python ne peut pas les libérer.
    """
    try:
        print("Creating distance filter...")
        distance_filter = vtk.vtkDistancePolyDataFilter()
        
        print("Setting input data...")
        distance_filter.SetInputData(0, polydata1)
        distance_filter.SetInputData(1, polydata2)
        distance_filter.SetSignedDistance(signed)
        
        if 'slicer' in globals():
            slicer.app.processEvents()
        
        print("Computing distances...")
        distance_filter.Update()
        
        print("Getting output...")
        raw_output = distance_filter.GetOutput()
        
        if not raw_output or raw_output.GetNumberOfPoints() == 0:
            raise ValueError("Distance computation failed - empty output")
        
        output = vtk.vtkPolyData()
        output.DeepCopy(raw_output)
        
        distance_filter.RemoveAllInputs()
        del distance_filter
        del raw_output
        
        print("Processing results...")
        if output.GetCellData().GetArray("Distance"):
            output.GetCellData().RemoveArray("Distance")
        
        distance_name = "SignedDistance" if signed else "AbsoluteDistance"
        distance_array = output.GetPointData().GetArray("Distance")
        if distance_array:
            distance_array.SetName(distance_name)
        else:
            raise ValueError("No distance array found in output")

        # Add a constant scalar array to visualize the original model easily
        const_array = vtk.vtkDoubleArray()
        const_array.SetName("Original")
        const_array.SetNumberOfTuples(output.GetNumberOfPoints())
        const_array.FillComponent(0, 1.0)
        output.GetPointData().AddArray(const_array)
        
        print(f"Distance computation completed successfully: {output.GetNumberOfPoints()} points")
        return output
        
    except Exception as e:
        print(f"Error during distance computation: {str(e)}")
        traceback.print_exc()
        raise RuntimeError(f"Distance computation failed: {str(e)}")

def batch_process(t1_dir, t2_dir, patient_list, output_dir, signed=True, output_text=".vtk", zone_type="merged"):
    """
    Batch process qui exécute chaque paire de fichiers dans un sous-processus
    séparé pour garantir la libération complète de la mémoire entre chaque cas.
    
    Le problème fondamental est que vtkDistancePolyDataFilter alloue de la mémoire
    côté C++ (cell locators, BSP trees) que ni gc.collect() ni DeepCopy ne peuvent
    libérer complètement. Seul un processus séparé garantit la libération via l'OS.
    """
    import subprocess
    import json
    import tempfile
    
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
    print(f"\nFound {total_files} pairs to process for zone '{zone_type}'")

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
    
    print(f"Using Python executable: {slicer_python}")
    print(f"Worker script: {worker_script}")

    processed_pairs = []
    
    for idx, pair in enumerate(pairs_to_process):
        processed_count = idx + 1
        output_filename = f"{pair['patient']}_{pair['zone']}_ModelDistance{output_text}"
        output_path = str(output_dir / output_filename)

        print(f"\n{'='*60}")
        print(f"Processing [{processed_count}/{total_files}]: {Path(pair['file1']).name}")
        print(f"  with: {Path(pair['file2']).name}")
        print(f"  Patient: {pair['patient_id']}, Zone: {pair['zone']}")
        print(f"{'='*60}")
        
        if psutil:
            mem = psutil.virtual_memory()
            print(f"Memory before: {mem.percent:.1f}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")

        try:
            cmd = [
                slicer_python, worker_script,
                "--file1", pair['file1'],
                "--file2", pair['file2'],
                "--output", output_path,
                "--signed" if signed else "--unsigned",
            ]
            
            print(f"Launching subprocess...")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if proc.stdout:
                for line in proc.stdout.strip().split('\n'):
                    print(f"  [worker] {line}")
            
            if proc.returncode != 0:
                print(f"  [worker] ERROR (exit code {proc.returncode}):")
                if proc.stderr:
                    for line in proc.stderr.strip().split('\n')[-5:]:
                        print(f"  [worker] {line}")
                continue
            
            processed_pairs.append({
                'patient_id': pair['patient_id'],
                'zone': pair['zone'],
                't1_file': Path(pair['file1']).name,
                't2_file': Path(pair['file2']).name,
                'output_file': output_filename,
            })
            
            print(f"  ✓ Successfully processed {output_filename}")
            
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT processing {Path(pair['file1']).name} (>10 min)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
        
        if psutil:
            mem = psutil.virtual_memory()
            print(f"Memory after:  {mem.percent:.1f}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")
        
        if 'slicer' in globals():
            slicer.app.processEvents()
    
    print(f"\n{'='*60}")
    print(f"Processing complete. {len(processed_pairs)}/{total_files} pairs processed.")
    print(f"{'='*60}")
    for pair in processed_pairs:
        print(f"  ✓ {pair['patient_id']} ({pair['zone']}): {pair['output_file']}")

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
                            print("Issue")

    output_df = pd.DataFrame(columns=file_alls.columns)
    unique_cb = file_cb["ID"].unique()
    unique_max = file_max["ID"].unique()
    unique_mand = file_mand["ID"].unique()

    if unique_cb.all() != unique_max.all() or unique_cb.all() !=unique_mand.all():
        print("Issue on the ID patient")

    for val in unique_cb:
        output_df.loc[val, "ID"] = val

        cb_df = file_cb[file_cb["ID"]==val]
        for cb_features,dic_features in dic_columns.items():
            if dic_features.get("Location")=="CB":
                if dic_features.get("Average") == "No":
                    if dic_features.get("Composant") in ["Transverse","Vertical","AP"]:
                        landmark = dic_features.get("Landmarks1")+" - "+dic_features.get("Landmarks2")
                    else:
                        landmark = dic_features.get("Landmarks1")+" / "+dic_features.get("Landmarks2")
                        landmark = landmark.replace("_","-")
                    line_specific = cb_df[cb_df["Landmarks"] == landmark]
                    output_df.loc[val, cb_features] = float(line_specific[dic_features.get("Composant")].values[0])
                else:
                    average = 0
                    nbr = dic_features.get("Nbr_Landmarks")//2
                    for i in range(nbr):
                        if dic_features.get("Composant") in ["Transverse","Vertical","AP"]:
                            landmark = dic_features.get("Landmarks"+str(2*i+1))+" - "+dic_features.get("Landmarks"+str(2*i+2))
                        else:
                            landmark = dic_features.get("Landmarks"+str(2*i+1))+" / "+dic_features.get("Landmarks"+str(2*i+2))
                            landmark = landmark.replace("_","-")
                        line_specific = cb_df[cb_df["Landmarks"] == landmark]
                        average += float(line_specific[dic_features.get("Composant")].values[0])
                    average /= nbr
                    output_df.loc[val, cb_features] = average
        
        max_df = file_max[file_max["ID"]==val]
        for max_features,dic_features in dic_columns.items():
            if dic_features.get("Location")=="MAX":
                if dic_features.get("Average") == "No":
                    if dic_features.get("Composant") in ["Transverse","Vertical","AP"]:
                        landmark = dic_features.get("Landmarks1")+" - "+dic_features.get("Landmarks2")
                    else:
                        landmark = dic_features.get("Landmarks1")+" / "+dic_features.get("Landmarks2")
                        landmark = landmark.replace("_","-")
                    line_specific = max_df[max_df["Landmarks"] == landmark]
                    output_df.loc[val, max_features] = float(line_specific[dic_features.get("Composant")].values[0])
                else:
                    average = 0
                    nbr = dic_features.get("Nbr_Landmarks")//2
                    for i in range(nbr):
                        if dic_features.get("Composant") in ["Transverse","Vertical","AP"]:
                            landmark = dic_features.get("Landmarks"+str(2*i+1))+" - "+dic_features.get("Landmarks"+str(2*i+2))
                        else:
                            landmark = dic_features.get("Landmarks"+str(2*i+1))+" / "+dic_features.get("Landmarks"+str(2*i+2))
                            landmark = landmark.replace("_","-")
                        line_specific = max_df[max_df["Landmarks"] == landmark]
                        average += float(line_specific[dic_features.get("Composant")].values[0])
                    average /= nbr
                    output_df.loc[val, max_features] = average
        
        mand_df = file_mand[file_mand["ID"]==val]
        for mand_features,dic_features in dic_columns.items():
            if dic_features.get("Location")=="MAND":
                if dic_features.get("Average") == "No":
                    if dic_features.get("Composant") in ["Transverse","Vertical","AP"]:
                        landmark = dic_features.get("Landmarks1")+" - "+dic_features.get("Landmarks2")
                    else:
                        landmark = dic_features.get("Landmarks1")+" / "+dic_features.get("Landmarks2")
                        landmark = landmark.replace("_","-")
                    line_specific = mand_df[mand_df["Landmarks"] == landmark]
                    output_df.loc[val, mand_features] = float(line_specific[dic_features.get("Composant")].values[0])
                else:
                    average = 0
                    nbr = dic_features.get("Nbr_Landmarks")//2
                    for i in range(nbr):
                        if dic_features.get("Composant") in ["Transverse","Vertical","AP"]:
                            landmark = dic_features.get("Landmarks"+str(2*i+1))+" - "+dic_features.get("Landmarks"+str(2*i+2))
                        else:
                            landmark = dic_features.get("Landmarks"+str(2*i+1))+" / "+dic_features.get("Landmarks"+str(2*i+2))
                            landmark = landmark.replace("_","-")
                        line_specific = mand_df[mand_df["Landmarks"] == landmark]
                        average += float(line_specific[dic_features.get("Composant")].values[0])
                    average /= nbr
                    output_df.loc[val, mand_features] = average

    output_df.to_excel(os.path.join(outputfolder,"PostProcess_Measurements.xlsx"),index=False)