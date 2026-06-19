# Utilities for generating landmark JSON outputs and label translation
import json
import logging
import os
import sys

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("ALI_IOS_IO")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def GenControlPoint(group_data, selected_lm):
    """Generate control points for landmarks with error handling."""
    try:
        if not group_data:
            logger.warning("group_data is empty")
            return []
        
        if not selected_lm:
            logger.warning("selected_lm is empty")
            return []
        
        lm_lst = []
        for i, (label, data) in enumerate(group_data.items(), 1):
            try:
                if label in selected_lm:
                    if not all(k in data for k in ["x", "y", "z"]):
                        logger.warning(f"Missing coordinates for landmark {label}")
                        continue
                    
                    lm_lst.append({
                        "id": str(i),
                        "label": label,
                        "description": "",
                        "associatedNodeID": "",
                        "position": [data["x"], data["y"], data["z"]],
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": True,
                        "visibility": True,
                        "positionStatus": "defined"
                    })
            except Exception as e:
                logger.error(f"Error processing landmark {label}: {e}")
                continue
        
        logger.debug(f"Generated {len(lm_lst)} control points")
        return lm_lst
    except Exception as e:
        logger.error(f"Error in GenControlPoint: {e}")
        raise

def WriteJson(lm_lst, out_path):
    """Write landmarks to JSON file with error handling."""
    try:
        if not out_path:
            logger.error("Output path cannot be empty")
            raise ValueError("Output path is empty")
        
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
                logger.debug(f"Created output directory: {out_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory {out_dir}: {e}")
                raise
        
        content = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
            "markups": [{
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": lm_lst,
                "measurements": [],
                "display": {
                    "visibility": False,
                    "opacity": 1.0,
                    "color": [0.5, 0.5, 0.5],
                    "selectedColor": [0.27, 0.67, 0.39],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 2.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 2.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "snapMode": "toVisibleSurface"
                }
            }]
        }
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=4)
        
        logger.info(f"Successfully wrote JSON to {out_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file {out_path}: {e}")
        raise

def TradLabel(teeth_list):
    """Translate tooth labels to FDI numbering system with error handling."""
    try:
        mapping = {
            'LL7': 18, 'LL6': 19, 'LL5': 20, 'LL4': 21, 'LL3': 22, 'LL2': 23, 'LL1': 24,
            'LR1': 25, 'LR2': 26, 'LR3': 27, 'LR4': 28, 'LR5': 29, 'LR6': 30, 'LR7': 31,
            'UL7': 15, 'UL6': 14, 'UL5': 13, 'UL4': 12, 'UL3': 11, 'UL2': 10, 'UL1': 9,
            'UR1': 8, 'UR2': 7, 'UR3': 6, 'UR4': 5, 'UR5': 4, 'UR6': 3, 'UR7': 2
        }
        
        if not teeth_list:
            logger.warning("teeth_list is empty")
            return {'Lower': [], 'Upper': []}
        
        result = {'Lower': [], 'Upper': []}
        unknown_teeth = []
        
        for tooth in teeth_list:
            try:
                logger.debug(f"Processing tooth: {tooth}")
                if tooth in mapping:
                    fdi_number = mapping[tooth]
                    if tooth.startswith('L'):
                        result['Lower'].append(fdi_number)
                    else:
                        result['Upper'].append(fdi_number)
                else:
                    logger.warning(f"Unknown tooth notation: {tooth}")
                    unknown_teeth.append(tooth)
            except Exception as e:
                logger.error(f"Error processing tooth {tooth}: {e}")
                unknown_teeth.append(tooth)
        
        if unknown_teeth:
            logger.warning(f"Failed to map teeth: {unknown_teeth}")
        
        logger.info(f"Translated teeth - Upper: {result['Upper']}, Lower: {result['Lower']}")
        return result
    except Exception as e:
        logger.error(f"Error in TradLabel: {e}")
        raise
