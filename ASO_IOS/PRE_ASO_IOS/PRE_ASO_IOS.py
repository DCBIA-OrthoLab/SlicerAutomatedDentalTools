#!/usr/bin/env python-real
import glob
import os
import sys
import time
import argparse
import platform
import logging
import numpy as np

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("PRE_ASO_IOS")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from tqdm import tqdm
from itertools import chain


fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

def check_platform():
    if platform.system() == 'Windows':
        return "Windows"
    elif platform.system() == 'Linux':
        if 'microsoft' in platform.release().lower():
            return "WSL"
        else:
            return "Linux"
    else:
        return "Unknown"

# Import from utils
if check_platform()=="WSL":
    from ASO_IOS_utils.utils import UpperOrLower, search, ReadSurf, WriteSurf, WritefileError, saveMatrixAsTfm, PatientNumber
    from ASO_IOS_utils.icp import vtkICP, vtkMeanTeeth, InitIcp, ICP, ToothNoExist, NoSegmentationSurf
    from ASO_IOS_utils.data_file import Files_vtk_link, Jaw, Lower, Upper
    from ASO_IOS_utils.transformation import TransformSurf
    from ASO_IOS_utils.pre_icp import PrePreAso
    
else:
    from ASO_IOS_utils import (
        UpperOrLower, search, ReadSurf, WriteSurf, WritefileError, saveMatrixAsTfm, PatientNumber,
        vtkICP, vtkMeanTeeth, InitIcp, ICP, ToothNoExist, NoSegmentationSurf,
        Files_vtk_link, Jaw, Lower, Upper,
        TransformSurf,
        PrePreAso,
    )
    
# import ASO_IOS_utils

def main(args):
    """
    Main processing function for IOS teeth mean registration preprocessing.
    
    Performs ICP-based registration of dental surfaces with comprehensive error handling.
    Supports both Upper and Lower jaw processing with optional occlusion linking.
    
    Args:
        args: Argument namespace with attributes:
            - list_teeth: CSV string of teeth to process
            - gold_folder: Path to gold standard reference surfaces
            - input: Path to input surfaces
            - output_folder: Path for output transforms
            - log_path: Path to processing log file
            - folder_error: Path for error files
            - occlusion: "true"/"false" for jaw linking
            - jaw: "Upper"/"Lower" for jaw selection (if occlusion=true)
            - add_inname: Suffix to add to output filenames
    
    Returns:
        dict: Processing summary with keys:
            - total_files: Total files processed
            - successful: Number of successful registrations
            - failed: Number of failed registrations
            - errors: List of error details
    """
    logger.info("ICP mean teeth registration preprocessing started")
    
    # Initialize processing variables
    list_extension = [".vtk", ".stl", ".off", ".obj", ".vtp"]
    lower = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    dic_teeth = {"Upper": [], "Lower": []}
    
    # Track processing results
    failed_indices = []
    success_count = 0
    error_details = []
    
    try:
        # ===== Stage 1: Argument and Input Validation =====
        logger.debug("Stage 1: Validating arguments and inputs")
        try:
            if not hasattr(args, 'list_teeth') or not args.list_teeth or not args.list_teeth[0]:
                raise ValueError("list_teeth argument is missing or empty")
            if not hasattr(args, 'gold_folder') or not args.gold_folder or not args.gold_folder[0]:
                raise ValueError("gold_folder argument is missing or empty")
            if not hasattr(args, 'input') or not args.input or not args.input[0]:
                raise ValueError("input argument is missing or empty")
            if not hasattr(args, 'output_folder') or not args.output_folder or not args.output_folder[0]:
                raise ValueError("output_folder argument is missing or empty")
            if not hasattr(args, 'log_path') or not args.log_path or not args.log_path[0]:
                raise ValueError("log_path argument is missing or empty")
            if not hasattr(args, 'folder_error') or not args.folder_error or not args.folder_error[0]:
                raise ValueError("folder_error argument is missing or empty")
            logger.debug("All required arguments present")
        except ValueError as ve:
            logger.error(f"Argument validation failed: {str(ve)}")
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": [{"stage": "argument_validation", "message": str(ve)}]
            }
        
        # ===== Stage 2: Teeth Dictionary Parsing =====
        logger.debug("Stage 2: Building teeth dictionary from input list")
        try:
            list_teeth = args.list_teeth[0].split(",")
            dic = {
                "UR8": 1, "UR7": 2, "UR6": 3, "UR5": 4, "UR4": 5, "UR3": 6, "UR2": 7, "UR1": 8,
                "UL1": 9, "UL2": 10, "UL3": 11, "UL4": 12, "UL5": 13, "UL6": 14, "UL7": 15, "UL8": 16,
                "LL8": 17, "LL7": 18, "LL6": 19, "LL5": 20, "LL4": 21, "LL3": 22, "LL2": 23, "LL1": 24,
                "LR1": 25, "LR2": 26, "LR3": 27, "LR4": 28, "LR5": 29, "LR6": 30, "LR7": 31, "LR8": 32,
            }
            
            for tooth in list_teeth:
                tooth = tooth.strip()
                if tooth not in dic:
                    raise ValueError(f"Unknown tooth identifier: {tooth}")
                tooth_code = dic[tooth]
                if tooth_code in lower:
                    dic_teeth["Lower"].append(tooth_code)
                else:
                    dic_teeth["Upper"].append(tooth_code)
            
            if not dic_teeth["Upper"] and not dic_teeth["Lower"]:
                raise ValueError("No valid teeth identified from input list")
            logger.info(f"Teeth dictionary built: Upper={dic_teeth['Upper']}, Lower={dic_teeth['Lower']}")
        except (ValueError, KeyError) as e:
            logger.error(f"Teeth dictionary parsing failed: {str(e)}")
            error_details.append({"stage": "teeth_parsing", "message": str(e)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 3: Gold Reference Loading =====
        logger.debug("Stage 3: Loading gold standard reference surfaces")
        gold = {}
        try:
            if not os.path.exists(args.gold_folder[0]):
                raise FileNotFoundError(f"Gold folder does not exist: {args.gold_folder[0]}")
            
            gold_files = list(
                chain.from_iterable(search(args.gold_folder[0], list_extension).values())
            )
            
            if len(gold_files) < 2:
                raise ValueError(f"Expected at least 2 gold reference files, found {len(gold_files)}")
            
            logger.debug(f"Loading {len(gold_files)} gold reference files")
            for i, gf in enumerate(gold_files[:2]):
                try:
                    jaw_type = UpperOrLower(gf)
                    gold_surf = ReadSurf(gf)
                    gold[jaw_type] = gold_surf
                    logger.debug(f"Gold reference [{jaw_type}] loaded from {os.path.basename(gf)}")
                except Exception as gf_err:
                    logger.error(f"Failed to load gold file {i+1}: {str(gf_err)}")
                    raise
            
            if "Upper" not in gold or "Lower" not in gold:
                raise ValueError("Could not load both Upper and Lower gold references")
            logger.info("Gold reference surfaces loaded successfully")
        except (FileNotFoundError, ValueError, Exception) as ge:
            logger.error(f"Gold reference loading failed: {str(ge)}")
            error_details.append({"stage": "gold_loading", "message": str(ge)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 4: Output Directory Setup =====
        logger.debug("Stage 4: Setting up output directories")
        try:
            # Setup output folder
            if not os.path.exists(args.output_folder[0]):
                os.makedirs(args.output_folder[0], exist_ok=True)
                logger.debug(f"Created output folder: {args.output_folder[0]}")
            
            # Setup error folder
            if not os.path.exists(args.folder_error[0]):
                os.makedirs(args.folder_error[0], exist_ok=True)
                logger.debug(f"Created error folder: {args.folder_error[0]}")
            
            # Setup log file
            log_dir = os.path.split(args.log_path[0])[0]
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            with open(args.log_path[0], "w") as log_f:
                log_f.truncate(0)
            logger.debug(f"Log file initialized: {args.log_path[0]}")
        except OSError as oe:
            logger.error(f"Directory setup failed: {str(oe)}")
            error_details.append({"stage": "directory_setup", "message": str(oe)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 5: Jaw Configuration =====
        logger.debug("Stage 5: Configuring jaw parameters")
        try:
            link = False
            jaw = None
            
            if args.occlusion[0].lower() == "true":
                link = True
                if args.jaw[0] == "Upper":
                    jaw = Jaw(Upper())
                    logger.debug("Jaw linking enabled for Upper jaw")
                elif args.jaw[0] == "Lower":
                    jaw = Jaw(Lower())
                    logger.debug("Jaw linking enabled for Lower jaw")
                else:
                    raise ValueError(f"Invalid jaw specification: {args.jaw[0]}")
            else:
                logger.debug("Jaw linking disabled")
        except (ValueError, Exception) as je:
            logger.error(f"Jaw configuration failed: {str(je)}")
            error_details.append({"stage": "jaw_config", "message": str(je)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 6: File Discovery =====
        logger.debug("Stage 6: Discovering input surface files")
        try:
            if not os.path.exists(args.input[0]):
                raise FileNotFoundError(f"Input folder does not exist: {args.input[0]}")
            
            if link:
                list_files = Files_vtk_link(args.input[0])
            else:
                list_files = list(
                    chain.from_iterable(search(args.input[0], list_extension).values())
                )
            
            if not list_files:
                raise ValueError(f"No surface files found in {args.input[0]}")
            logger.info(f"Discovered {len(list_files)} surface files for processing")
        except (FileNotFoundError, ValueError, Exception) as fe:
            logger.error(f"File discovery failed: {str(fe)}")
            error_details.append({"stage": "file_discovery", "message": str(fe)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 7: ICP Method Initialization =====
        logger.debug("Stage 7: Initializing ICP registration methods")
        try:
            Method = [InitIcp(), vtkICP()]
            option_upper = vtkMeanTeeth(dic_teeth["Upper"],property="Universal_ID")
            option_lower = vtkMeanTeeth(dic_teeth["Lower"],property="Universal_ID")
            icp_upper = ICP(Method, option=option_upper)
            icp_lower = ICP(Method, option=option_lower)
            icp = {"Upper": icp_upper, "Lower": icp_lower}
            logger.debug("ICP methods initialized successfully")
        except Exception as ic:
            logger.error(f"ICP initialization failed: {str(ic)}")
            error_details.append({"stage": "icp_init", "message": str(ic)})
            return {
                "total_files": len(list_files),
                "successful": 0,
                "failed": len(list_files),
                "errors": error_details
            }
        
        # ===== Stage 8: Main Processing Loop =====
        logger.info(f"Starting registration processing for {len(list_files)} files")
        for index, file in tqdm(enumerate(list_files), total=len(list_files)):
            try:
                # Determine jaw and file path
                file_vtk = file
                if link:
                    file_vtk = file[jaw()]
                if not link:
                    jaw = Jaw(file_vtk)
                
                logger.debug(f"Processing [{index+1}/{len(list_files)}]: {os.path.basename(file_vtk)} ({jaw()})")
                
                # ===== Stage 8.1: Surface Loading =====
                try:
                    surf = ReadSurf(file_vtk)
                    logger.debug(f"Surface loaded for {os.path.basename(file_vtk)}")
                except Exception as sl_err:
                    error_msg = f"Failed to load surface: {str(sl_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_vtk),
                        "stage": "surface_loading",
                        "message": str(sl_err)
                    })
                    try:
                        WritefileError(file_vtk, args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 8.2: PreProcessing =====
                try:
                    surf, matrix = PrePreAso(surf, gold[jaw()], dic_teeth[jaw()])
                    logger.debug(f"Preprocessing completed for {os.path.basename(file_vtk)}")
                except ToothNoExist as tne:
                    error_msg = f"Tooth not found: {str(tne)}"
                    logger.error(f"[{index}] {error_msg} for {file_vtk}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_vtk),
                        "stage": "preprocessing",
                        "error_type": "ToothNoExist",
                        "message": str(tne)
                    })
                    try:
                        WritefileError(file_vtk, args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    with open(args.log_path[0], "a") as log_f:
                        log_f.write(f"{index},ToothNoExist\n")
                    continue
                except NoSegmentationSurf as nss:
                    error_msg = f"No segmentation surface: {str(nss)}"
                    logger.error(f"[{index}] {error_msg} for {file_vtk}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_vtk),
                        "stage": "preprocessing",
                        "error_type": "NoSegmentationSurf",
                        "message": str(nss)
                    })
                    try:
                        WritefileError(file_vtk, args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    with open(args.log_path[0], "a") as log_f:
                        log_f.write(f"{index},NoSegmentationSurf\n")
                    continue
                except Exception as pre_err:
                    error_msg = f"Preprocessing error: {str(pre_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_vtk),
                        "stage": "preprocessing",
                        "message": str(pre_err)
                    })
                    try:
                        WritefileError(file_vtk, args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 8.3: ICP Registration =====
                try:
                    output_icp = icp[jaw()].run(surf, gold[jaw()])
                    logger.debug(f"ICP registration completed for {os.path.basename(file_vtk)}")
                except Exception as icp_err:
                    error_msg = f"ICP registration failed: {str(icp_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_vtk),
                        "stage": "icp_registration",
                        "message": str(icp_err)
                    })
                    try:
                        WritefileError(file_vtk, args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 8.4: Matrix Computation =====
                try:
                    final_matrix = np.matmul(output_icp["matrix"], matrix)
                    patient_id = PatientNumber(file_vtk)
                    tfm_path = os.path.join(args.output_folder[0], f"{patient_id}_SegOr.tfm")
                    logger.debug(f"Saving transform matrix to {os.path.basename(tfm_path)}")
                    saveMatrixAsTfm(final_matrix, tfm_path)
                    logger.debug(f"Transform matrix saved successfully")
                except Exception as mat_err:
                    error_msg = f"Matrix computation/saving failed: {str(mat_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_vtk),
                        "stage": "matrix_saving",
                        "message": str(mat_err)
                    })
                    try:
                        WritefileError(file_vtk, args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 8.5: Surface Output Writing =====
                try:
                    logger.debug(f"Writing registered surface for {os.path.basename(file_vtk)}")
                    WriteSurf(
                        output_icp["source_Or"],
                        args.output_folder[0],
                        os.path.basename(file_vtk),
                        args.add_inname[0],
                    )
                    logger.debug(f"Registered surface saved successfully")
                except Exception as write_err:
                    error_msg = f"Surface writing failed: {str(write_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    logger.error(f"  Output folder: {args.output_folder[0]}, File: {os.path.basename(file_vtk)}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_vtk),
                        "stage": "surface_writing",
                        "message": str(write_err)
                    })
                    try:
                        WritefileError(file_vtk, args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 8.6: Linked Jaw Processing (if applicable) =====
                if link:
                    try:
                        logger.debug(f"Processing linked jaw for {os.path.basename(file_vtk)}")
                        surf_lower = ReadSurf(file[jaw.inv()])
                        output_lower = TransformSurf(surf_lower, matrix)
                        output_lower = TransformSurf(output_lower, output_icp["matrix"])
                        
                        WriteSurf(
                            output_lower,
                            args.output_folder[0],
                            os.path.basename(file[jaw.inv()]),
                            args.add_inname[0],
                        )
                        logger.debug(f"Linked jaw surface written successfully")
                    except Exception as link_err:
                        error_msg = f"Linked jaw processing failed: {str(link_err)}"
                        logger.error(f"[{index}] {error_msg}")
                        failed_indices.append(index)
                        error_details.append({
                            "index": index,
                            "file": os.path.basename(file[jaw.inv()]),
                            "stage": "linked_jaw_processing",
                            "message": str(link_err)
                        })
                        try:
                            WritefileError(file[jaw.inv()], args.folder_error[0], error_msg)
                        except Exception as we:
                            logger.warning(f"Could not write error file: {str(we)}")
                        continue
                
                # ===== Stage 8.7: Log Update =====
                try:
                    with open(args.log_path[0], "a") as log_f:
                        log_f.write(f"{index+1},success\n")
                    success_count += 1
                    logger.info(f"[{index+1}/{len(list_files)}] File processed successfully")
                except Exception as log_err:
                    logger.warning(f"Could not update log file: {str(log_err)}")
            
            except Exception as outer_err:
                logger.error(f"Unhandled error in processing loop at index {index}: {str(outer_err)}")
                failed_indices.append(index)
                error_details.append({
                    "index": index,
                    "stage": "outer_loop",
                    "message": str(outer_err)
                })
        
        # ===== Stage 9: Final Report =====
        logger.info(f"Processing complete: {success_count} successful, {len(failed_indices)} failed")
        return {
            "total_files": len(list_files),
            "successful": success_count,
            "failed": len(failed_indices),
            "failed_indices": failed_indices,
            "errors": error_details
        }
    
    except Exception as main_err:
        logger.error(f"Fatal error in main processing: {str(main_err)}", exc_info=True)
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "errors": [{"stage": "main_function", "message": str(main_err)}]
        }


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PRE_ASO_IOS - IOS Teeth Mean Registration Preprocessing")
    logger.info("="*80)
    logger.debug(f"Command line arguments: {sys.argv}")
    
    try:
        # ===== Argument Parsing =====
        try:
            parser = argparse.ArgumentParser(description="IOS teeth mean registration preprocessing")
            
            parser.add_argument("input", nargs=1, help="Path to input surface files")
            parser.add_argument("gold_folder", nargs=1, help="Path to gold standard reference folder")
            parser.add_argument("output_folder", nargs=1, help="Path to output folder")
            parser.add_argument("add_inname", nargs=1, help="Suffix to add to output filenames")
            parser.add_argument("list_teeth", nargs=1, help="CSV list of teeth to process")
            parser.add_argument("occlusion", nargs=1, help="Enable occlusion linking (true/false)")
            parser.add_argument("jaw", nargs=1, help="Jaw selection (Upper/Lower)")
            parser.add_argument("folder_error", nargs=1, help="Path to error folder")
            parser.add_argument("log_path", nargs=1, help="Path to log file")
            
            args = parser.parse_args()
            logger.info("Arguments parsed successfully")
            logger.debug(f"Teeth: {args.list_teeth[0]}, Gold: {args.gold_folder[0]}")
            logger.debug(f"Input: {args.input[0]}, Output: {args.output_folder[0]}")
        except SystemExit as se:
            if se.code != 0:
                logger.error(f"Argument parsing failed with exit code {se.code}")
            raise
        except Exception as parse_err:
            logger.error(f"Failed to parse arguments: {str(parse_err)}")
            raise
        
        # ===== Main Processing =====
        logger.info("Starting main processing")
        result = main(args)
        
        # ===== Process Results =====
        logger.info(f"Processing Results:")
        logger.info(f"  Total files: {result['total_files']}")
        logger.info(f"  Successful: {result['successful']}")
        logger.info(f"  Failed: {result['failed']}")
        
        if result['errors']:
            logger.warning(f"Errors encountered during processing:")
            for error in result['errors'][:10]:  # Log first 10 errors
                if 'stage' in error:
                    logger.warning(f"  [{error.get('stage')}] {error.get('message', 'Unknown error')}")
                else:
                    logger.warning(f"  {error}")
            if len(result['errors']) > 10:
                logger.warning(f"  ... and {len(result['errors']) - 10} more errors")
        
        if result['successful'] > 0:
            logger.info(f"Successfully processed {result['successful']} files")
        
        if result['failed'] > 0:
            logger.warning(f"Failed to process {result['failed']} files")
            logger.info(f"Failed file indices: {result.get('failed_indices', [])[:20]}")
        
        logger.info("="*80)
        logger.info("PRE_ASO_IOS processing completed")
        logger.info("="*80)
        
        # Exit with appropriate status
        sys.exit(0 if result['failed'] == 0 else 1)
    
    except Exception as e:
        logger.error(f"Fatal error in PRE_ASO_IOS: {str(e)}", exc_info=True)
        logger.error("="*80)
        logger.error("PRE_ASO_IOS terminated with error")
        logger.error("="*80)
        sys.exit(1)

    main(args)