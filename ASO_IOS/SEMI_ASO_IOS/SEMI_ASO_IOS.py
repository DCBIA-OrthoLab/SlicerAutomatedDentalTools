#!/usr/bin/env python-real


import glob
import os

import sys
import argparse
import logging
from tqdm import tqdm
import numpy as np

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("SEMI_ASO_IOS")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from ASO_IOS_utils import (
    vtkICP,
    InitIcp,
    SelectKey,
    ICP,
    TransformSurf,
    UpperOrLower,
    LoadJsonLandmarks,
    WriteSurf,
    WriteJsonLandmarks,
    listlandmark2diclandmark,
    ReadSurf,
    Files_vtk_json,
    Files_vtk_json_semilink,
    Jaw,
    Lower,
    Upper,
    ApplyTransform,
    WritefileError,
)

def main(args):
    """
    Main processing function for IOS semi-automatic registration with landmarks.
    
    Performs ICP registration of dental surfaces with landmark alignment.
    Supports both Upper and Lower jaw processing with optional occlusion linking.
    Handles both surface and landmark transformation and output.
    
    Args:
        args: Argument namespace with attributes:
            - list_landmark: CSV string of landmarks to use for registration
            - gold_folder: Path to gold standard reference folder
            - input: Path to input surface and landmark files
            - output_folder: Path for output transforms and surfaces
            - log_path: Path to processing log file
            - add_inname: Suffix to add to output filenames
            - occlusion: "true"/"false" for jaw linking
            - jaw: "Upper"/"Lower" for jaw selection (if occlusion=true)
            - folder_error: Path for error files
    
    Returns:
        dict: Processing summary with keys:
            - total_files: Total files processed
            - successful: Number of successful registrations
            - failed: Number of failed registrations
            - errors: List of error details
    """
    logger.info("ICP landmark registration processing started")
    
    # Initialize processing variables
    failed_indices = []
    success_count = 0
    error_details = []
    
    try:
        # ===== Stage 1: Landmark Dictionary Parsing =====
        logger.debug("Stage 1: Parsing landmark list from arguments")
        try:
            if not hasattr(args, 'list_landmark') or not args.list_landmark or not args.list_landmark[0]:
                raise ValueError("list_landmark argument is missing or empty")
            
            dic_landmark = listlandmark2diclandmark(args.list_landmark[0])
            
            if not dic_landmark.get("Upper") and not dic_landmark.get("Lower"):
                raise ValueError("No valid landmarks identified from input list")
            logger.info(f"Landmarks parsed: Upper={len(dic_landmark.get('Upper', []))}, Lower={len(dic_landmark.get('Lower', []))}")
        except (ValueError, KeyError) as e:
            logger.error(f"Landmark parsing failed: {str(e)}")
            error_details.append({"stage": "landmark_parsing", "message": str(e)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 2: Gold Reference Loading =====
        logger.debug("Stage 2: Loading gold standard reference landmark files")
        dic_gold = {}
        try:
            if not hasattr(args, 'gold_folder') or not args.gold_folder or not args.gold_folder[0]:
                raise ValueError("gold_folder argument is missing or empty")
            
            if not os.path.exists(args.gold_folder[0]):
                raise FileNotFoundError(f"Gold folder does not exist: {args.gold_folder[0]}")
            
            gold_json = glob.glob(args.gold_folder[0] + "/*json")
            
            if len(gold_json) < 2:
                raise ValueError(f"Expected at least 2 gold reference JSON files, found {len(gold_json)}")
            
            logger.debug(f"Loading {len(gold_json)} gold reference landmark files")
            dic_gold[UpperOrLower(gold_json[0])] = gold_json[0]
            dic_gold[UpperOrLower(gold_json[1])] = gold_json[1]
            
            if "Upper" not in dic_gold or "Lower" not in dic_gold:
                raise ValueError("Could not load both Upper and Lower gold references")
            logger.info("Gold reference landmark files loaded successfully")
        except (FileNotFoundError, ValueError, Exception) as ge:
            logger.error(f"Gold reference loading failed: {str(ge)}")
            error_details.append({"stage": "gold_loading", "message": str(ge)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 3: Output Directory Setup =====
        logger.debug("Stage 3: Setting up output directories")
        try:
            # Validate required arguments
            if not hasattr(args, 'output_folder') or not args.output_folder or not args.output_folder[0]:
                raise ValueError("output_folder argument is missing or empty")
            if not hasattr(args, 'log_path') or not args.log_path or not args.log_path[0]:
                raise ValueError("log_path argument is missing or empty")
            if not hasattr(args, 'folder_error') or not args.folder_error or not args.folder_error[0]:
                raise ValueError("folder_error argument is missing or empty")
            
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
        except (OSError, ValueError) as oe:
            logger.error(f"Directory setup failed: {str(oe)}")
            error_details.append({"stage": "directory_setup", "message": str(oe)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 4: ICP Method Initialization =====
        logger.debug("Stage 4: Initializing ICP registration methods")
        try:
            Method = [InitIcp(), vtkICP()]
            option_upper = SelectKey(dic_landmark["Upper"])
            option_lower = SelectKey(dic_landmark["Lower"])
            icp_upper = ICP(Method, option=option_upper)
            icp_lower = ICP(Method, option=option_lower)
            icp = {"Lower": icp_lower, "Upper": icp_upper}
            logger.debug("ICP methods initialized successfully")
            logger.debug(f"Landmark dictionary: {dic_landmark}")
        except Exception as ic:
            logger.error(f"ICP initialization failed: {str(ic)}")
            error_details.append({"stage": "icp_init", "message": str(ic)})
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
        logger.debug("Stage 6: Discovering input surface and landmark files")
        try:
            if not os.path.exists(args.input[0]):
                raise FileNotFoundError(f"Input folder does not exist: {args.input[0]}")
            
            if link:
                list_file = Files_vtk_json_semilink(args.input[0])
            else:
                list_file = Files_vtk_json(args.input[0])
            
            if not list_file:
                raise ValueError(f"No surface/landmark file pairs found in {args.input[0]}")
            logger.info(f"Discovered {len(list_file)} surface/landmark file pairs for processing")
        except (FileNotFoundError, ValueError, Exception) as fe:
            logger.error(f"File discovery failed: {str(fe)}")
            error_details.append({"stage": "file_discovery", "message": str(fe)})
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "errors": error_details
            }
        
        # ===== Stage 7: Main Processing Loop =====
        logger.info(f"Starting landmark-based registration processing for {len(list_file)} file pairs")
        for index, file in tqdm(enumerate(list_file), total=len(list_file)):
            try:
                # Determine jaw and file path
                file_jaw = file
                if link:
                    file_jaw = file[jaw()]
                else:
                    jaw = Jaw(file_jaw["json"])
                
                logger.debug(f"Processing [{index+1}/{len(list_file)}]: Surface {os.path.basename(file_jaw.get('vtk', 'N/A'))}, Landmarks {os.path.basename(file_jaw.get('json', 'N/A'))} ({jaw()})")
                
                # ===== Stage 7.1: Landmark JSON Validation =====
                try:
                    if file_jaw.get("json") is None:
                        logger.warning(f"[{index}] Skipping: No JSON landmark file associated")
                        with open(args.log_path[0], "a") as log_f:
                            log_f.write(f"{index},skipped_no_json\n")
                        continue
                    
                    if not os.path.exists(file_jaw["json"]):
                        raise FileNotFoundError(f"Landmark JSON file not found: {file_jaw['json']}")
                    logger.debug(f"JSON landmark file validated: {os.path.basename(file_jaw['json'])}")
                except (FileNotFoundError, Exception) as jv_err:
                    error_msg = f"Landmark JSON validation failed: {str(jv_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_jaw.get('json', 'unknown')),
                        "stage": "json_validation",
                        "message": str(jv_err)
                    })
                    try:
                        WritefileError(file_jaw.get("json"), args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 7.2: ICP Registration with Landmarks =====
                try:
                    logger.debug(f"Starting ICP registration with landmarks for jaw {jaw()}")
                    output_icp = icp[jaw()].run(file_jaw["json"], dic_gold[jaw()])
                    logger.debug(f"ICP registration completed, transformation matrix obtained")
                except KeyError as k:
                    error_msg = f"Landmark key not found in gold reference: {str(k)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_jaw.get('json', 'unknown')),
                        "stage": "icp_registration",
                        "error_type": "KeyError",
                        "message": f"Landmark {str(k)} not found in gold reference {os.path.basename(dic_gold.get(jaw(), 'unknown'))}"
                    })
                    try:
                        error_detail = f'Please verify landmark file {file_jaw["json"]} or gold reference {dic_gold[jaw()]}, we dont find this landmark {k}'
                        WritefileError(file_jaw["json"], args.folder_error[0], error_detail)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    with open(args.log_path[0], "a") as log_f:
                        log_f.write(f"{index},KeyError_{str(k)}\n")
                    continue
                except Exception as icp_err:
                    error_msg = f"ICP registration failed: {str(icp_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_jaw.get('json', 'unknown')),
                        "stage": "icp_registration",
                        "message": str(icp_err)
                    })
                    try:
                        WritefileError(file_jaw["json"], args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 7.3: Surface Transformation =====
                try:
                    logger.debug(f"Loading input surface for transformation")
                    surf_input = ReadSurf(file_jaw["vtk"])
                    surf_output = TransformSurf(surf_input, output_icp["matrix"])
                    logger.debug(f"Surface transformation completed")
                except Exception as surf_err:
                    error_msg = f"Surface transformation failed: {str(surf_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_jaw.get('vtk', 'unknown')),
                        "stage": "surface_transformation",
                        "message": str(surf_err)
                    })
                    try:
                        WritefileError(file_jaw["vtk"], args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 7.4: Output Writing (Landmarks) =====
                try:
                    logger.debug(f"Writing transformed landmarks")
                    WriteJsonLandmarks(
                        output_icp["source_Or"],
                        file_jaw["json"],
                        file_jaw["json"],
                        args.add_inname[0],
                        args.output_folder[0],
                    )
                    logger.debug(f"Transformed landmarks saved successfully")
                except Exception as lm_err:
                    error_msg = f"Landmark output writing failed: {str(lm_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_jaw.get('json', 'unknown')),
                        "stage": "landmark_writing",
                        "message": str(lm_err)
                    })
                    try:
                        WritefileError(file_jaw["json"], args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 7.5: Output Writing (Surface) =====
                try:
                    logger.debug(f"Writing transformed surface")
                    WriteSurf(
                        surf_output, args.output_folder[0], file_jaw["vtk"], args.add_inname[0]
                    )
                    logger.debug(f"Transformed surface saved successfully")
                except Exception as surf_write_err:
                    error_msg = f"Surface output writing failed: {str(surf_write_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    failed_indices.append(index)
                    error_details.append({
                        "index": index,
                        "file": os.path.basename(file_jaw.get('vtk', 'unknown')),
                        "stage": "surface_writing",
                        "message": str(surf_write_err)
                    })
                    try:
                        WritefileError(file_jaw["vtk"], args.folder_error[0], error_msg)
                    except Exception as we:
                        logger.warning(f"Could not write error file: {str(we)}")
                    continue
                
                # ===== Stage 7.6: Transform Matrix Output =====
                try:
                    logger.debug(f"Saving transformation matrix")
                    file_name = file.get("name", f"file_{index}") if isinstance(file, dict) else f"file_{index}"
                    matrix_path = os.path.join(args.output_folder[0], f'matrix_{file_name}.npy')
                    np.save(matrix_path, output_icp["matrix"])
                    logger.debug(f"Transformation matrix saved: {os.path.basename(matrix_path)}")
                except Exception as matrix_err:
                    error_msg = f"Matrix output writing failed: {str(matrix_err)}"
                    logger.error(f"[{index}] {error_msg}")
                    logger.warning("Continuing despite matrix save failure")
                    error_details.append({
                        "index": index,
                        "stage": "matrix_writing",
                        "message": str(matrix_err)
                    })
                
                # ===== Stage 7.7: Linked Jaw Processing (if applicable) =====
                if link:
                    try:
                        logger.debug(f"Processing linked jaw")
                        link_file = file[jaw.inv()]
                        
                        # Process linked surface
                        try:
                            logger.debug(f"Loading linked jaw surface")
                            surf_input_link = ReadSurf(link_file["vtk"])
                            surf_output_link = TransformSurf(surf_input_link, output_icp["matrix"])
                            WriteSurf(
                                surf_output_link,
                                args.output_folder[0],
                                link_file["vtk"],
                                args.add_inname[0],
                            )
                            logger.debug(f"Linked jaw surface processed successfully")
                        except Exception as link_surf_err:
                            error_msg = f"Linked jaw surface processing failed: {str(link_surf_err)}"
                            logger.error(f"[{index}] {error_msg}")
                            logger.warning("Continuing with main jaw processing")
                        
                        # Process linked landmarks if available
                        if link_file.get("json") is not None:
                            try:
                                logger.debug(f"Processing linked jaw landmarks")
                                json_input_link = LoadJsonLandmarks(link_file["json"])
                                json_output_link = ApplyTransform(json_input_link, output_icp["matrix"])
                                WriteJsonLandmarks(
                                    json_output_link,
                                    link_file["json"],
                                    link_file["json"],
                                    args.add_inname[0],
                                    args.output_folder[0],
                                )
                                logger.debug(f"Linked jaw landmarks processed successfully")
                            except Exception as link_lm_err:
                                error_msg = f"Linked jaw landmark processing failed: {str(link_lm_err)}"
                                logger.error(f"[{index}] {error_msg}")
                                logger.warning("Continuing with main processing")
                        else:
                            logger.debug("No landmarks for linked jaw")
                    except Exception as link_err:
                        error_msg = f"Linked jaw processing failed: {str(link_err)}"
                        logger.error(f"[{index}] {error_msg}")
                        failed_indices.append(index)
                        error_details.append({
                            "index": index,
                            "stage": "linked_jaw_processing",
                            "message": str(link_err)
                        })
                        continue
                
                # ===== Stage 7.8: Log Update =====
                try:
                    with open(args.log_path[0], "a") as log_f:
                        log_f.write(f"{index+1},success\n")
                    success_count += 1
                    logger.info(f"[{index+1}/{len(list_file)}] File pair processed successfully")
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
        
        # ===== Stage 8: Final Report =====
        logger.info(f"Processing complete: {success_count} successful, {len(failed_indices)} failed")
        return {
            "total_files": len(list_file),
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
    logger.info("SEMI_ASO_IOS - IOS Semi-Automatic Landmark-Based Registration")
    logger.info("="*80)
    logger.debug(f"Command line arguments: {sys.argv}")
    
    try:
        # ===== Argument Parsing =====
        try:
            parser = argparse.ArgumentParser(
                description="IOS semi-automatic landmark-based registration"
            )
            parser.add_argument("input", nargs=1, help="Path to input surface and landmark files")
            parser.add_argument("gold_folder", nargs=1, help="Path to gold standard reference folder")
            parser.add_argument("output_folder", nargs=1, help="Path to output folder")
            parser.add_argument("add_inname", nargs=1, help="Suffix to add to output filenames")
            parser.add_argument("list_landmark", nargs=1, help="CSV list of landmarks to use for registration")
            parser.add_argument("occlusion", nargs=1, help="Enable occlusion linking (true/false)")
            parser.add_argument("jaw", nargs=1, help="Jaw selection (Upper/Lower)")
            parser.add_argument("folder_error", nargs=1, help="Path to error folder")
            parser.add_argument("log_path", nargs=1, help="Path to log file")
            
            args = parser.parse_args()
            logger.info("Arguments parsed successfully")
            logger.debug(f"Landmarks: {args.list_landmark[0]}, Gold: {args.gold_folder[0]}")
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
        logger.info(f"  Total file pairs: {result['total_files']}")
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
            logger.info(f"Successfully processed {result['successful']} file pairs")
        
        if result['failed'] > 0:
            logger.warning(f"Failed to process {result['failed']} file pairs")
            logger.info(f"Failed file indices: {result.get('failed_indices', [])[:20]}")
        
        logger.info("="*80)
        logger.info("SEMI_ASO_IOS processing completed")
        logger.info("="*80)
        
        # Exit with appropriate status
        sys.exit(0 if result['failed'] == 0 else 1)
    
    except Exception as e:
        logger.error(f"Fatal error in SEMI_ASO_IOS: {str(e)}", exc_info=True)
        logger.error("="*80)
        logger.error("SEMI_ASO_IOS terminated with error")
        logger.error("="*80)
        sys.exit(1)
